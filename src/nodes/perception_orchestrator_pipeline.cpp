#include "concrete_block_world_model/nodes/perception_orchestrator_node.hpp"

#include "concrete_block_world_model/utils/img_utils.hpp"
#include "concrete_block_world_model/utils/visu_utils.hpp"
#include "concrete_block_world_model/utils/world_model_utils.hpp"
#include "concrete_block_world_model/world_model/continuous_perception.hpp"
#include "concrete_block_world_model/world_model/scene_discovery_flow.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace
{

void publishTimingScalar(
  const rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr & publisher,
  double value)
{
  if (!publisher) {
    return;
  }
  std_msgs::msg::Float64 msg;
  msg.data = value;
  publisher->publish(msg);
}

}  // namespace

void PerceptionOrchestratorNode::publishWorldMarkers(const std_msgs::msg::Header & header, const std::vector<Block> & blocks)
  {
    auto marker_header = header;
    marker_header.stamp = rclcpp::Time(0, 0, get_clock()->get_clock_type());
    const auto static_scene_world = staticSceneObjectsInWorld();
    const auto markers = cbpwm::buildWorldMarkers(
      marker_header, blocks, static_scene_world, world_frame_, block_dimensions_m_);
    marker_pub_->publish(markers);

    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 3000,
      "Published marker array: markers=%zu blocks=%zu static_objects=%zu frame=%s",
      markers.markers.size(),
      blocks.size(),
      static_scene_world.size(),
      world_frame_.c_str());

    if (blocks.size() != last_published_block_count_) {
      last_published_block_count_ = blocks.size();
      if (!blocks.empty()) {
        const auto & b0 = blocks.front();
        RCLCPP_INFO(
          get_logger(),
          "Published markers: %zu blocks + %zu static objects in frame '%s' (first block: id=%s pos=[%.3f, %.3f, %.3f])",
          blocks.size(),
          static_scene_world.size(),
          world_frame_.c_str(),
          b0.id.c_str(),
          b0.pose.position.x,
          b0.pose.position.y,
          b0.pose.position.z);
      } else if (!static_scene_objects_.empty()) {
        RCLCPP_INFO(
          get_logger(),
          "Published markers: 0 blocks + %zu static objects in frame '%s'",
          static_scene_world.size(),
          world_frame_.c_str());
      }
    }
  }

void PerceptionOrchestratorNode::publishPersistentWorld(const std_msgs::msg::Header & header)
  {
    BlockArray out;
    out.header = header;

    const rclcpp::Time now_stamp(header.stamp);
    {
      std::lock_guard<std::mutex> lock(persistent_world_mutex_);
      for (auto it = persistent_world_.begin(); it != persistent_world_.end();) {
        const rclcpp::Time seen(it->second.last_seen);
        const bool task_protected =
          protect_task_blocks_from_timeout_ &&
          it->second.task_status != Block::TASK_UNKNOWN &&
          it->second.task_status != Block::TASK_FREE;
        if (seeded_block_ids_.count(it->first) > 0U || task_protected) {
          it->second.last_seen = header.stamp;
          out.blocks.push_back(it->second);
          ++it;
          continue;
        }
        if ((now_stamp - seen).seconds() > runtime_cfg_.object_timeout_s) {
          it = persistent_world_.erase(it);
          continue;
        }
        out.blocks.push_back(it->second);
        ++it;
      }
    }

    world_pub_->publish(out);
    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 3000,
      "Published world model: blocks=%zu frame=%s",
      out.blocks.size(),
      out.header.frame_id.c_str());
    updateLatestWorldCache(out);
    {
      std::lock_guard<std::mutex> lock(latest_planning_scene_mutex_);
      latest_planning_scene_ = buildPlanningSceneSnapshot(out.header, out.blocks);
    }
    publishWorldMarkers(out.header, out.blocks);
  }

void PerceptionOrchestratorNode::publishDetectionOverlay(
    const sensor_msgs::msg::Image::ConstSharedPtr & image,
    const vision_msgs::msg::Detection2DArray & detections,
    const sensor_msgs::msg::Image & mask_msg)
  {
    cv::Mat img = toCvBgr(*image);

    if (!mask_msg.data.empty()) {
      cv::Mat mask = toCvMono(mask_msg);
      overlayMask(img, mask, cv::Scalar(255, 255, 0), 0.35);
    }

    drawDetectionBoxes(img, detections, cv::Scalar(0, 255, 0));
    auto out = cv_bridge::CvImage(image->header, "bgr8", img).toImageMsg();
    det_debug_pub_->publish(*out);
  }


void PerceptionOrchestratorNode::syncCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr image,
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud)
  {
    bool have_one_shot = false;
    {
      std::lock_guard<std::mutex> lock(one_shot_mutex_);
      have_one_shot = active_one_shot_.mode != cbpwm::OneShotMode::kNone;
    }

    if (!have_one_shot) {
      if (perception_mode_.load() != PerceptionMode::kContinuous) {
        return;
      }
      const uint64_t frame_index = ++continuous_seen_frames_;
      if ((frame_index % static_cast<uint64_t>(continuous_cfg_.process_every_n_frames)) != 0U) {
        return;
      }
    }

    bool expected = false;
    if (!busy_.compare_exchange_strong(expected, true)) {
      dropped_busy_frames_.fetch_add(1);
      return;
    }

    const rclcpp::Time t_img(image->header.stamp);
    const rclcpp::Time t_cloud(cloud->header.stamp);
    const double dt = std::abs((t_img - t_cloud).seconds());

    if (dt > runtime_cfg_.max_sync_delta_s) {
      dropped_sync_frames_.fetch_add(1);
      RCLCPP_WARN(
        get_logger(),
        "Dropped frame pair due to sync delta %.4f s (> %.4f s)",
        dt,
        runtime_cfg_.max_sync_delta_s);
      resetBusy();
      return;
    }

    if (have_one_shot) {
      processFrame(image, cloud);
    } else {
      processContinuousFrame(image, cloud);
    }
  }

void PerceptionOrchestratorNode::handleOneShotSegmentationResponse(
    rclcpp::Client<SegmentSrv>::SharedFuture seg_future,
    const sensor_msgs::msg::Image::ConstSharedPtr & image,
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr & cloud,
    std::chrono::steady_clock::time_point t_start,
    const OneShotRequest & run_request)
  {
    try {
      auto seg_res = seg_future.get();
      const auto t_after_seg = std::chrono::steady_clock::now();
      const size_t seg_detection_count =
        seg_res ? seg_res->detections.detections.size() : 0U;

      if (!seg_res || !seg_res->success) {
        publishPersistentWorld(cloud->header);
        completeOneShotRequest(run_request.sequence, false, "Segmentation failed.");
        resetBusy();
        return;
      }

      RCLCPP_INFO(
        get_logger(),
        "One-shot %s: segmentation detections=%zu",
        cbpwm::oneShotModeToString(run_request.mode),
        seg_detection_count);

      if (debug_detection_overlay_enabled_.load() && det_debug_pub_) {
        publishDetectionOverlay(image, seg_res->detections, seg_res->mask);
      }

      const auto t_after_track = t_after_seg;
      const size_t tracked_count = seg_detection_count;
      RCLCPP_INFO(
        get_logger(),
        "One-shot %s: detections=%zu tracked=%zu (tracker bypassed)",
        cbpwm::oneShotModeToString(run_request.mode),
        seg_detection_count,
        tracked_count);

      if (seg_res->detections.detections.empty()) {
        const auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
          t_after_seg - t_start).count();
        recordTiming(total_ms, 0, 0, total_ms);
        publishPersistentWorld(cloud->header);

        if (run_request.mode == cbpwm::OneShotMode::kSceneDiscovery) {
          completeOneShotRequest(
            run_request.sequence,
            true,
            "Scene discovery finished (detections=0, tracked=0, registrations=0).");
        } else {
          completeOneShotRequest(
            run_request.sequence,
            false,
            "Requested block was not detected.");
        }
        resetBusy();
        return;
      }

      auto candidates = cbpwm::buildRegistrationCandidates(
        *seg_res, run_request.mode, run_request.target_block_id);

      const size_t registration_candidates = candidates.size();
      RCLCPP_INFO(
        get_logger(),
        "One-shot %s: detections=%zu tracked=%zu registration_candidates=%zu",
        cbpwm::oneShotModeToString(run_request.mode),
        seg_detection_count,
        tracked_count,
        registration_candidates);

      if (registration_candidates == 0) {
        const auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
          t_after_track - t_start).count();
        recordTiming(total_ms, 0, 0, total_ms);
        publishPersistentWorld(cloud->header);

        if (run_request.mode == cbpwm::OneShotMode::kSceneDiscovery) {
          completeOneShotRequest(
            run_request.sequence,
            true,
            "Scene discovery finished (detections=" + std::to_string(seg_detection_count) +
            ", tracked=" + std::to_string(tracked_count) +
            ", registration_candidates=0, registrations=0).");
        } else {
          completeOneShotRequest(
            run_request.sequence,
            false,
            "Requested block was not available for registration "
            "(detections=" + std::to_string(seg_detection_count) +
            ", tracked=" + std::to_string(tracked_count) + ").");
        }

        resetBusy();
        return;
      }

      if (!action_client_->action_server_is_ready()) {
        RCLCPP_WARN(get_logger(), "Registration action unavailable.");
        completeOneShotRequest(run_request.sequence, false, "Registration action unavailable.");
        resetBusy();
        return;
      }

      const auto t_reg_start = std::chrono::steady_clock::now();
      cbpwm::RegistrationCounters counters;
      Eigen::Vector3d camera_origin_world = Eigen::Vector3d::Zero();
      std::string camera_origin_reason;
      const bool have_camera_origin = lookupFrameOriginInWorld(
        cloud->header, camera_origin_world, camera_origin_reason);
      if (!have_camera_origin) {
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 3000,
          "Scene-discovery coarse center offset disabled for this frame: %s",
          camera_origin_reason.c_str());
      }
      cbpwm::SceneFlowRequest scene_req;
      scene_req.mode = run_request.mode;
      scene_req.target_block_id = run_request.target_block_id;
      scene_req.registration_timeout_s = run_request.registration_timeout_s;
      scene_req.refine_target_max_distance_m = runtime_cfg_.refine_target_max_distance_m;

      cbpwm::SceneFlowRuntime scene_rt;
      scene_rt.logger = get_logger();
      scene_rt.run_registration =
        [this](
        uint32_t detection_id,
        const sensor_msgs::msg::Image & mask,
        const sensor_msgs::msg::PointCloud2 & in_cloud,
        const std_msgs::msg::Header & header,
        double timeout_s,
        Block & out_block,
        std::string & out_reason) {
          return runRegistrationSync(
            detection_id,
            mask,
            in_cloud,
            header,
            timeout_s,
            out_block,
            out_reason);
        };
      scene_rt.upsert_block =
        [this, &run_request, cloud](Block & block, std::string & assigned_id, std::string & reason) {
          std::lock_guard<std::mutex> lock(persistent_world_mutex_);
          return cbpwm::upsertRegisteredBlock(
            persistent_world_,
            world_block_counter_,
            block,
            run_request.mode,
            run_request.target_block_id,
            cloud->header,
            *get_clock(),
            associationConfig(),
            assigned_id,
            reason);
        };
      scene_rt.get_expected_target =
        [this](const std::string & target_id, Block & out_target) {
          std::lock_guard<std::mutex> lock(persistent_world_mutex_);
          const auto it = persistent_world_.find(target_id);
          if (it == persistent_world_.end()) {
            return false;
          }
          out_target = it->second;
          return true;
        };
      scene_rt.try_coarse_fallback =
        [this, have_camera_origin, &camera_origin_world](
        uint32_t detection_id,
        const std::string & registration_reason,
        const sensor_msgs::msg::Image & mask,
        const sensor_msgs::msg::PointCloud2 & in_cloud,
        const cbpwm::SceneFlowRequest & request,
        cbpwm::RegistrationCounters & out_counters) {
          return trySceneDiscoveryCoarseFallback(
            detection_id,
            registration_reason,
            mask,
            in_cloud,
            request,
            have_camera_origin ? &camera_origin_world : nullptr,
            out_counters);
        };

      cbpwm::processRegistrationCandidates(
        candidates,
        *cloud,
        scene_req,
        scene_rt,
        counters);

      const auto t_reg_end = std::chrono::steady_clock::now();
      const auto seg_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        t_after_seg - t_start).count();
      const auto track_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        t_after_track - t_after_seg).count();
      const auto reg_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        t_reg_end - t_reg_start).count();
      const auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        t_reg_end - t_start).count();
      recordTiming(seg_ms, track_ms, reg_ms, total_ms);

      publishPersistentWorld(cloud->header);
      completeOneShotRequest(
        run_request.sequence,
        counters.registrations_ok > 0 || run_request.mode == cbpwm::OneShotMode::kSceneDiscovery,
        "Pose estimation finished (detections=" + std::to_string(seg_detection_count) +
        ", tracked=" + std::to_string(tracked_count) +
        ", registration_candidates=" + std::to_string(registration_candidates) +
        ", registrations=" + std::to_string(counters.registrations_ok) +
        ", coarse_upserts=" + std::to_string(counters.coarse_upserts_ok) + ").");

      resetBusy();
    } catch (const std::exception & e) {
      RCLCPP_ERROR(get_logger(), "Segmentation stage failed: %s", e.what());
      completeOneShotRequest(run_request.sequence, false, "Segmentation stage exception.");
      resetBusy();
    }
  }

void PerceptionOrchestratorNode::processContinuousFrame(
    const sensor_msgs::msg::Image::ConstSharedPtr & image,
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr & cloud)
  {
    if (!segment_client_->service_is_ready()) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 3000,
        "Continuous perception skipped: segmentation service unavailable.");
      publishPersistentWorld(cloud->header);
      resetBusy();
      return;
    }

    auto seg_req = std::make_shared<SegmentSrv::Request>();
    seg_req->image = *image;
    seg_req->return_debug = debug_detection_overlay_enabled_.load();

    const auto t_start = std::chrono::steady_clock::now();
    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 1000,
      "Continuous frame: requesting YOLO segmentation stamp=%u.%u return_debug=%s",
      image->header.stamp.sec,
      image->header.stamp.nanosec,
      seg_req->return_debug ? "true" : "false");
    segment_client_->async_send_request(
      seg_req,
      [this, image, cloud, t_start](rclcpp::Client<SegmentSrv>::SharedFuture seg_future) {
        handleContinuousSegmentationResponse(seg_future, image, cloud, t_start);
      });
  }

void PerceptionOrchestratorNode::handleContinuousSegmentationResponse(
    rclcpp::Client<SegmentSrv>::SharedFuture seg_future,
    const sensor_msgs::msg::Image::ConstSharedPtr & image,
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr & cloud,
    std::chrono::steady_clock::time_point t_start)
  {
    try {
      auto seg_res = seg_future.get();
      const auto t_after_seg = std::chrono::steady_clock::now();
      const auto publish_continuous_timing =
        [this](
          double seg_ms,
          double cutout_ms,
          double coarse_ms,
          double registration_ms,
          double upsert_ms,
          double total_ms,
          double detections,
          double accepted,
          double rejected) {
          publishTimingScalar(continuous_timing_seg_ms_pub_, seg_ms);
          publishTimingScalar(continuous_timing_cutout_ms_pub_, cutout_ms);
          publishTimingScalar(continuous_timing_coarse_ms_pub_, coarse_ms);
          publishTimingScalar(continuous_timing_registration_ms_pub_, registration_ms);
          publishTimingScalar(continuous_timing_upsert_ms_pub_, upsert_ms);
          publishTimingScalar(continuous_timing_total_ms_pub_, total_ms);
          publishTimingScalar(continuous_timing_detections_pub_, detections);
          publishTimingScalar(continuous_timing_accepted_pub_, accepted);
          publishTimingScalar(continuous_timing_rejected_pub_, rejected);
        };
      if (!seg_res || !seg_res->success) {
        RCLCPP_WARN(
          get_logger(),
          "Continuous frame: YOLO segmentation failed or returned empty response.");
        publishPersistentWorld(cloud->header);
        resetBusy();
        return;
      }

      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "Continuous frame: YOLO segmentation success detections=%zu mask_bytes=%zu debug_bytes=%zu",
        seg_res->detections.detections.size(),
        seg_res->mask.data.size(),
        seg_res->debug_image.data.size());

      if (yolo_service_debug_pub_ && !seg_res->debug_image.data.empty()) {
        yolo_service_debug_pub_->publish(seg_res->debug_image);
        RCLCPP_INFO_THROTTLE(
          get_logger(), *get_clock(), 3000,
          "Continuous frame: published YOLO service debug image.");
      }

      if (seg_res->detections.detections.empty() || seg_res->mask.data.empty()) {
        RCLCPP_INFO_THROTTLE(
          get_logger(), *get_clock(), 1000,
          "Continuous frame: no detections or mask; publishing world snapshot unchanged.");
        publishPersistentWorld(cloud->header);
        const auto seg_ms =
          std::chrono::duration_cast<std::chrono::milliseconds>(t_after_seg - t_start).count();
        recordTiming(seg_ms, 0, 0, seg_ms);
        publish_continuous_timing(
          static_cast<double>(seg_ms), 0.0, 0.0, 0.0, 0.0, static_cast<double>(seg_ms), 0.0,
          0.0, 0.0);
        resetBusy();
        return;
      }

      const cv::Mat full_mask = toCvMono(seg_res->mask);
      cv::Mat debug_image;
      cv::Mat accepted_mask = cv::Mat::zeros(full_mask.size(), CV_8UC1);
      cv::Mat rejected_mask = cv::Mat::zeros(full_mask.size(), CV_8UC1);
      std::vector<bool> accepted_by_detection(seg_res->detections.detections.size(), false);

      Eigen::Vector3d camera_origin_world = Eigen::Vector3d::Zero();
      std::string camera_origin_reason;
      const bool have_camera_origin = lookupFrameOriginInWorld(
        cloud->header, camera_origin_world, camera_origin_reason);
      if (have_camera_origin) {
        RCLCPP_INFO_THROTTLE(
          get_logger(), *get_clock(), 3000,
          "Continuous frame: camera origin in world [%.3f, %.3f, %.3f]",
          camera_origin_world.x(),
          camera_origin_world.y(),
          camera_origin_world.z());
      }
      if (!have_camera_origin) {
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 3000,
          "Continuous coarse center offset disabled for this frame: %s",
          camera_origin_reason.c_str());
      }

      size_t accepted_count = 0;
      size_t rejected_count = 0;
      int64_t cutout_sum_ms = 0;
      int64_t coarse_sum_ms = 0;
      int64_t registration_sum_ms = 0;
      int64_t upsert_sum_ms = 0;
      int continuous_registration_attempts = 0;
      std::vector<cbpwm::ContinuousMaskCandidate> candidates;
      for (size_t i = 0; i < seg_res->detections.detections.size(); ++i) {
        const auto & det = seg_res->detections.detections[i];
        cv::Mat det_mask = extract_mask_roi(full_mask, det);
        cv::Mat binary_mask;
        cv::threshold(det_mask, binary_mask, 0, 255, cv::THRESH_BINARY);

        const auto quality = cbpwm::evaluateContinuousMaskQuality(
          binary_mask,
          det,
          continuous_cfg_.min_mask_pixels,
          continuous_cfg_.min_mask_fill_ratio);
        cbpwm::logContinuousQuality(get_logger(), *get_clock(), i, det, quality);

        if (!quality.accepted) {
          ++rejected_count;
          cv::bitwise_or(rejected_mask, binary_mask, rejected_mask);
          RCLCPP_INFO_THROTTLE(
            get_logger(), *get_clock(), 2000,
            "Continuous mask rejected: idx=%zu pixels=%d bbox_area=%d fill=%.3f reason=%s",
            i,
            quality.mask_pixels,
            quality.bbox_area_px,
            quality.fill_ratio,
            quality.reason.c_str());
          continue;
        }

        Block coarse_block;
        std::string coarse_reason;
        const auto mask_msg =
          cv_bridge::CvImage(cloud->header, "mono8", binary_mask).toImageMsg();
        RCLCPP_INFO_THROTTLE(
          get_logger(), *get_clock(), 1000,
          "Continuous cutout input: idx=%zu detection_id=%u mask_pixels=%d min_cutout_points=%d",
          i,
          static_cast<uint32_t>(i + 1U),
          quality.mask_pixels,
          continuous_cfg_.min_valid_cloud_points);

        sensor_msgs::msg::PointCloud2 cutout_cloud;
        std::string cutout_reason;
        const auto t_cutout_start = std::chrono::steady_clock::now();
        const bool got_cutout = extractMaskCutoutSync(
          *mask_msg,
          *cloud,
          continuous_cfg_.cutout_timeout_s,
          cutout_cloud,
          cutout_reason);
        cutout_sum_ms += std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - t_cutout_start).count();
        if (!got_cutout) {
          ++rejected_count;
          cv::bitwise_or(rejected_mask, binary_mask, rejected_mask);
          RCLCPP_INFO_THROTTLE(
            get_logger(), *get_clock(), 2000,
            "Continuous cutout rejected: idx=%zu pixels=%d fill=%.3f reason=%s",
            i,
            quality.mask_pixels,
            quality.fill_ratio,
            cutout_reason.c_str());
          continue;
        }

        const auto t_coarse_start = std::chrono::steady_clock::now();
        const bool coarse_ok = buildCoarseBlockFromCloudCentroid(
          static_cast<uint32_t>(i + 1U),
          *mask_msg,
          cutout_cloud,
          cutout_cloud.header,
          have_camera_origin ? &camera_origin_world : nullptr,
          coarse_block,
          coarse_reason,
          continuous_cfg_.min_valid_cloud_points);
        coarse_sum_ms += std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - t_coarse_start).count();
        if (!coarse_ok) {
          ++rejected_count;
          cv::bitwise_or(rejected_mask, binary_mask, rejected_mask);
          RCLCPP_INFO_THROTTLE(
            get_logger(), *get_clock(), 2000,
            "Continuous coarse pose rejected: idx=%zu pixels=%d fill=%.3f cutout_points=%u reason=%s",
            i,
            quality.mask_pixels,
            quality.fill_ratio,
            cutout_cloud.width * cutout_cloud.height,
            coarse_reason.c_str());
          continue;
        }

        RCLCPP_INFO_THROTTLE(
          get_logger(), *get_clock(), 1000,
          "Continuous fragment coarse pose: idx=%zu id=%s pos=[%.3f, %.3f, %.3f] cutout_points=%u reason=%s",
          i,
          coarse_block.id.c_str(),
          coarse_block.pose.position.x,
          coarse_block.pose.position.y,
          coarse_block.pose.position.z,
          cutout_cloud.width * cutout_cloud.height,
          coarse_reason.c_str());

        coarse_block.confidence = static_cast<float>(cbpwm::detectionConfidence(det));
        coarse_block.last_seen = cloud->header.stamp;
        candidates.push_back(
          cbpwm::ContinuousMaskCandidate{
            i,
            binary_mask.clone(),
            quality,
            coarse_block,
            cbpwm::detectionConfidence(det),
            cutout_cloud.width * cutout_cloud.height});
      }

      const auto groups = cbpwm::groupContinuousCandidates(
        candidates,
        continuous_cfg_.mask_merge_enabled,
        continuous_cfg_.mask_merge_max_centroid_distance_m,
        get_logger(),
        *get_clock());

      cv::Mat merged_mask_debug = cv::Mat::zeros(full_mask.size(), CV_8UC1);
      for (size_t group_index = 0; group_index < groups.size(); ++group_index) {
        const auto & group = groups[group_index];
        const int merged_pixels = cv::countNonZero(group.merged_mask);
        int source_pixels = 0;
        uint32_t source_cutout_points = 0;
        double max_confidence = 0.0;
        std::string fragment_indices;
        for (const size_t candidate_index : group.candidate_indices) {
          const auto & candidate = candidates.at(candidate_index);
          source_pixels += candidate.quality.mask_pixels;
          source_cutout_points += candidate.cutout_points;
          max_confidence = std::max(max_confidence, candidate.confidence);
          if (!fragment_indices.empty()) {
            fragment_indices += ",";
          }
          fragment_indices += std::to_string(candidate.detection_index);
        }

        const auto mask_msg =
          cv_bridge::CvImage(cloud->header, "mono8", group.merged_mask).toImageMsg();
        RCLCPP_INFO_THROTTLE(
          get_logger(), *get_clock(), 1000,
          "Continuous merged cutout input: group=%zu fragments=[%s] merged_pixels=%d source_pixels=%d source_cutout_points=%u",
          group_index,
          fragment_indices.c_str(),
          merged_pixels,
          source_pixels,
          source_cutout_points);

        sensor_msgs::msg::PointCloud2 merged_cutout_cloud;
        std::string cutout_reason;
        const auto t_cutout_start = std::chrono::steady_clock::now();
        const bool got_cutout = extractMaskCutoutSync(
          *mask_msg,
          *cloud,
          continuous_cfg_.cutout_timeout_s,
          merged_cutout_cloud,
          cutout_reason);
        cutout_sum_ms += std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - t_cutout_start).count();
        if (!got_cutout) {
          ++rejected_count;
          cv::bitwise_or(rejected_mask, group.merged_mask, rejected_mask);
          RCLCPP_INFO_THROTTLE(
            get_logger(), *get_clock(), 2000,
            "Continuous merged cutout rejected: group=%zu fragments=[%s] merged_pixels=%d reason=%s",
            group_index,
            fragment_indices.c_str(),
            merged_pixels,
            cutout_reason.c_str());
          continue;
        }

        Block coarse_block;
        std::string coarse_reason;
        const auto t_coarse_start = std::chrono::steady_clock::now();
        const bool coarse_ok = buildCoarseBlockFromCloudCentroid(
          static_cast<uint32_t>(group_index + 1U),
          *mask_msg,
          merged_cutout_cloud,
          merged_cutout_cloud.header,
          have_camera_origin ? &camera_origin_world : nullptr,
          coarse_block,
          coarse_reason,
          continuous_cfg_.min_valid_cloud_points);
        coarse_sum_ms += std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - t_coarse_start).count();
        if (!coarse_ok) {
          ++rejected_count;
          cv::bitwise_or(rejected_mask, group.merged_mask, rejected_mask);
          RCLCPP_INFO_THROTTLE(
            get_logger(), *get_clock(), 2000,
            "Continuous merged coarse pose rejected: group=%zu fragments=[%s] merged_pixels=%d cutout_points=%u reason=%s",
            group_index,
            fragment_indices.c_str(),
            merged_pixels,
            merged_cutout_cloud.width * merged_cutout_cloud.height,
            coarse_reason.c_str());
          continue;
        }

        coarse_block.confidence = static_cast<float>(max_confidence);
        coarse_block.last_seen = cloud->header.stamp;
        Block block_to_upsert = coarse_block;
        bool precise_pose_used = false;

        if (continuous_cfg_.registration_enabled) {
          if (!action_client_ || !action_client_->action_server_is_ready()) {
            RCLCPP_WARN_THROTTLE(
              get_logger(), *get_clock(), 2000,
              "Continuous precise registration skipped: registration action unavailable.");
          } else if (continuous_registration_attempts >= continuous_cfg_.registration_max_per_frame) {
            RCLCPP_INFO_THROTTLE(
              get_logger(), *get_clock(), 2000,
              "Continuous precise registration skipped: group=%zu fragments=[%s] max_per_frame=%d reached.",
              group_index,
              fragment_indices.c_str(),
              continuous_cfg_.registration_max_per_frame);
          } else {
            ++continuous_registration_attempts;
            Block precise_block;
            std::string registration_reason;
            RCLCPP_INFO_THROTTLE(
              get_logger(), *get_clock(), 1000,
              "Continuous precise registration input: group=%zu fragments=[%s] merged_pixels=%d timeout=%.2fs",
              group_index,
              fragment_indices.c_str(),
              merged_pixels,
              continuous_cfg_.registration_timeout_s);
            const auto t_registration_start = std::chrono::steady_clock::now();
            const bool registration_ok = runRegistrationSync(
              static_cast<uint32_t>(group_index + 1U),
              *mask_msg,
              *cloud,
              cloud->header,
              continuous_cfg_.registration_timeout_s,
              precise_block,
              registration_reason);
            registration_sum_ms += std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::steady_clock::now() - t_registration_start).count();

            if (registration_ok) {
              block_to_upsert = precise_block;
              precise_pose_used = true;
              RCLCPP_INFO_THROTTLE(
                get_logger(), *get_clock(), 1000,
                "Continuous precise registration accepted: group=%zu fragments=[%s] pos=[%.3f, %.3f, %.3f] confidence=%.3f",
                group_index,
                fragment_indices.c_str(),
                precise_block.pose.position.x,
                precise_block.pose.position.y,
                precise_block.pose.position.z,
                precise_block.confidence);
            } else {
              RCLCPP_WARN_THROTTLE(
                get_logger(), *get_clock(), 2000,
                "Continuous precise registration failed: group=%zu fragments=[%s] reason=%s; falling back to coarse pose.",
                group_index,
                fragment_indices.c_str(),
                registration_reason.c_str());
            }
          }
        }

        std::string assigned_id;
        std::string upsert_reason;
        bool upsert_ok = false;
        auto assoc_cfg = associationConfig();
        assoc_cfg.association_max_distance_m = continuous_cfg_.association_max_distance_m;
        assoc_cfg.association_max_age_s = continuous_cfg_.association_max_age_s;
        const auto t_upsert_start = std::chrono::steady_clock::now();
        {
          std::lock_guard<std::mutex> lock(persistent_world_mutex_);
          upsert_ok = cbpwm::upsertRegisteredBlock(
            persistent_world_,
            world_block_counter_,
            block_to_upsert,
            cbpwm::OneShotMode::kSceneDiscovery,
            "",
            cloud->header,
            *get_clock(),
            assoc_cfg,
            assigned_id,
            upsert_reason);
        }
        upsert_sum_ms += std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - t_upsert_start).count();

        if (!upsert_ok) {
          ++rejected_count;
          cv::bitwise_or(rejected_mask, group.merged_mask, rejected_mask);
          RCLCPP_INFO_THROTTLE(
            get_logger(), *get_clock(), 2000,
            "Continuous upsert rejected: group=%zu fragments=[%s] confidence=%.3f reason=%s",
            group_index,
            fragment_indices.c_str(),
            block_to_upsert.confidence,
            upsert_reason.c_str());
          continue;
        }

        RCLCPP_INFO_THROTTLE(
          get_logger(), *get_clock(), 1000,
          "Continuous world-model upsert: group=%zu fragments=[%s] incoming=%s assigned=%s confidence=%.3f pose_status=%d",
          group_index,
          fragment_indices.c_str(),
          block_to_upsert.id.c_str(),
          assigned_id.c_str(),
          block_to_upsert.confidence,
          block_to_upsert.pose_status);

        ++accepted_count;
        block_to_upsert.id = assigned_id;
        for (const size_t candidate_index : group.candidate_indices) {
          accepted_by_detection[candidates.at(candidate_index).detection_index] = true;
        }
        cv::bitwise_or(accepted_mask, group.merged_mask, accepted_mask);
        cv::bitwise_or(merged_mask_debug, group.merged_mask, merged_mask_debug);
        RCLCPP_INFO_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "Continuous merged upsert accepted: group=%zu fragments=[%s] assigned=%s pose=%s merged_pixels=%d cutout_points=%u confidence=%.3f",
          group_index,
          fragment_indices.c_str(),
          assigned_id.c_str(),
          precise_pose_used ? "PRECISE" : "COARSE",
          merged_pixels,
          merged_cutout_cloud.width * merged_cutout_cloud.height,
          block_to_upsert.confidence);
      }

      if (continuous_merged_mask_pub_) {
        auto merged_debug_msg =
          cv_bridge::CvImage(image->header, "mono8", merged_mask_debug).toImageMsg();
        continuous_merged_mask_pub_->publish(*merged_debug_msg);
      }

      if (debug_detection_overlay_enabled_.load() && det_debug_pub_) {
        debug_image = toCvBgr(*image);
        overlayMask(debug_image, rejected_mask, cv::Scalar(0, 0, 255), 0.35);
        overlayMask(debug_image, accepted_mask, cv::Scalar(0, 255, 0), 0.35);
        for (size_t i = 0; i < seg_res->detections.detections.size(); ++i) {
          drawBoundingBox(
            debug_image,
            seg_res->detections.detections[i].bbox,
            accepted_by_detection[i] ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255),
            2);
        }
        auto debug_msg = cv_bridge::CvImage(image->header, "bgr8", debug_image).toImageMsg();
        det_debug_pub_->publish(*debug_msg);
      }

      publishPersistentWorld(cloud->header);
      const auto t_end = std::chrono::steady_clock::now();
      const auto seg_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(t_after_seg - t_start).count();
      const auto total_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
      recordTiming(seg_ms, 0, 0, total_ms);
      publish_continuous_timing(
        static_cast<double>(seg_ms),
        static_cast<double>(cutout_sum_ms),
        static_cast<double>(coarse_sum_ms),
        static_cast<double>(registration_sum_ms),
        static_cast<double>(upsert_sum_ms),
        static_cast<double>(total_ms),
        static_cast<double>(seg_res->detections.detections.size()),
        static_cast<double>(accepted_count),
        static_cast<double>(rejected_count));
      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Continuous perception frame: detections=%zu accepted=%zu rejected=%zu seg=%ld cutout=%ld coarse=%ld registration=%ld upsert=%ld total=%ld ms",
        seg_res->detections.detections.size(),
        accepted_count,
        rejected_count,
        seg_ms,
        cutout_sum_ms,
        coarse_sum_ms,
        registration_sum_ms,
        upsert_sum_ms,
        total_ms);
      resetBusy();
    } catch (const std::exception & e) {
      RCLCPP_ERROR(get_logger(), "Continuous perception failed: %s", e.what());
      publishPersistentWorld(cloud->header);
      resetBusy();
    }
  }

void PerceptionOrchestratorNode::processFrame(
    const sensor_msgs::msg::Image::ConstSharedPtr & image,
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr & cloud)
  {
    const auto t_start = std::chrono::steady_clock::now();

    OneShotRequest run_request;
    {
      std::lock_guard<std::mutex> lock(one_shot_mutex_);
      run_request = active_one_shot_;
    }

    if (run_request.mode == cbpwm::OneShotMode::kNone) {
      resetBusy();
      return;
    }

    if (run_request.mode == cbpwm::OneShotMode::kRefineGrasped && refine_grasped_use_fk_roi_) {
      processRefineGraspedWithFkRoi(image, cloud, run_request, t_start);
      return;
    }
    if (run_request.mode == cbpwm::OneShotMode::kRefineBlock &&
      tryProcessRefineBlockWithPoseRoi(image, cloud, run_request, t_start))
    {
      return;
    }

    if (!segment_client_->service_is_ready()) {
      RCLCPP_WARN(get_logger(), "Segmentation service unavailable.");
      resetBusy();
      return;
    }

    auto seg_req = std::make_shared<SegmentSrv::Request>();
    seg_req->image = *image;
    seg_req->return_debug = false;

    segment_client_->async_send_request(
      seg_req,
      [this, image, cloud, t_start, run_request](rclcpp::Client<SegmentSrv>::SharedFuture seg_future) {
        handleOneShotSegmentationResponse(seg_future, image, cloud, t_start, run_request);
      });
  }
