#include "concrete_block_world_model/nodes/perception_orchestrator_node.hpp"

#include "concrete_block_world_model/utils/img_utils.hpp"
#include "concrete_block_world_model/utils/visu_utils.hpp"
#include "concrete_block_world_model/utils/world_model_utils.hpp"
#include "concrete_block_world_model/world_model/scene_discovery_flow.hpp"

#include <algorithm>
#include <cmath>

namespace
{

struct ContinuousMaskQuality
{
  int mask_pixels{0};
  int bbox_area_px{0};
  double fill_ratio{0.0};
  bool accepted{false};
  std::string reason;
};

double detectionConfidence(const vision_msgs::msg::Detection2D & det)
{
  if (det.results.empty()) {
    return 1.0;
  }
  return det.results.front().hypothesis.score;
}

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

cv::Rect clippedDetectionRect(
  const vision_msgs::msg::Detection2D & det,
  const cv::Size & image_size)
{
  const cv::Rect image_rect(0, 0, image_size.width, image_size.height);
  const cv::Rect raw = toCvRect(det);
  return raw & image_rect;
}

ContinuousMaskQuality evaluateContinuousMaskQuality(
  const cv::Mat & binary_mask,
  const vision_msgs::msg::Detection2D & det,
  int min_mask_pixels,
  double min_fill_ratio)
{
  ContinuousMaskQuality quality;
  if (binary_mask.empty()) {
    quality.reason = "empty mask";
    return quality;
  }

  const cv::Rect bbox = clippedDetectionRect(det, binary_mask.size());
  quality.bbox_area_px = bbox.area();
  if (quality.bbox_area_px <= 0) {
    quality.reason = "empty bbox";
    return quality;
  }

  quality.mask_pixels = cv::countNonZero(binary_mask);
  quality.fill_ratio =
    static_cast<double>(quality.mask_pixels) / static_cast<double>(quality.bbox_area_px);

  if (quality.mask_pixels < min_mask_pixels) {
    quality.reason = "mask_pixels below threshold";
    return quality;
  }
  if (quality.fill_ratio < min_fill_ratio) {
    quality.reason = "fill_ratio below threshold";
    return quality;
  }

  quality.accepted = true;
  quality.reason = "accepted";
  return quality;
}

void logContinuousQuality(
  const rclcpp::Logger & logger,
  rclcpp::Clock & clock,
  size_t index,
  const vision_msgs::msg::Detection2D & det,
  const ContinuousMaskQuality & quality)
{
  RCLCPP_INFO_THROTTLE(
    logger, clock, 1000,
    "Continuous quality idx=%zu bbox[cx=%.1f cy=%.1f w=%.1f h=%.1f] score=%.3f "
    "mask_pixels=%d bbox_area=%d fill=%.3f accepted=%s reason=%s",
    index,
    det.bbox.center.position.x,
    det.bbox.center.position.y,
    det.bbox.size_x,
    det.bbox.size_y,
    detectionConfidence(det),
    quality.mask_pixels,
    quality.bbox_area_px,
    quality.fill_ratio,
    quality.accepted ? "true" : "false",
    quality.reason.c_str());
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
        ", coarse_registrations=" + std::to_string(counters.coarse_upserts_ok) + ").");

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
          double upsert_ms,
          double total_ms,
          double detections,
          double accepted,
          double rejected) {
          publishTimingScalar(continuous_timing_seg_ms_pub_, seg_ms);
          publishTimingScalar(continuous_timing_cutout_ms_pub_, cutout_ms);
          publishTimingScalar(continuous_timing_coarse_ms_pub_, coarse_ms);
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
          static_cast<double>(seg_ms), 0.0, 0.0, 0.0, static_cast<double>(seg_ms), 0.0, 0.0,
          0.0);
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
      int64_t upsert_sum_ms = 0;
      for (size_t i = 0; i < seg_res->detections.detections.size(); ++i) {
        const auto & det = seg_res->detections.detections[i];
        cv::Mat det_mask = extract_mask_roi(full_mask, det);
        cv::Mat binary_mask;
        cv::threshold(det_mask, binary_mask, 0, 255, cv::THRESH_BINARY);

        const auto quality = evaluateContinuousMaskQuality(
          binary_mask,
          det,
          continuous_cfg_.min_mask_pixels,
          continuous_cfg_.min_mask_fill_ratio);
        logContinuousQuality(get_logger(), *get_clock(), i, det, quality);

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
        const bool got_cutout = runRegistrationServiceCutoutSync(
          *mask_msg,
          *cloud,
          continuous_cfg_.segmentation_timeout_s,
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
          "Continuous coarse pose output: idx=%zu id=%s pos=[%.3f, %.3f, %.3f] reason=%s",
          i,
          coarse_block.id.c_str(),
          coarse_block.pose.position.x,
          coarse_block.pose.position.y,
          coarse_block.pose.position.z,
          coarse_reason.c_str());

        coarse_block.confidence = static_cast<float>(detectionConfidence(det));
        coarse_block.last_seen = cloud->header.stamp;

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
            coarse_block,
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
          cv::bitwise_or(rejected_mask, binary_mask, rejected_mask);
          RCLCPP_INFO_THROTTLE(
            get_logger(), *get_clock(), 2000,
            "Continuous upsert rejected: idx=%zu confidence=%.3f reason=%s",
            i,
            coarse_block.confidence,
            upsert_reason.c_str());
          continue;
        }

        RCLCPP_INFO_THROTTLE(
          get_logger(), *get_clock(), 1000,
          "Continuous world-model upsert: idx=%zu incoming=%s assigned=%s confidence=%.3f pose_status=%d",
          i,
          coarse_block.id.c_str(),
          assigned_id.c_str(),
          coarse_block.confidence,
          coarse_block.pose_status);

        ++accepted_count;
        accepted_by_detection[i] = true;
        cv::bitwise_or(accepted_mask, binary_mask, accepted_mask);
        RCLCPP_INFO_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "Continuous coarse upsert accepted: idx=%zu assigned=%s pixels=%d fill=%.3f cutout_points=%u confidence=%.3f",
          i,
          assigned_id.c_str(),
          quality.mask_pixels,
          quality.fill_ratio,
          cutout_cloud.width * cutout_cloud.height,
          coarse_block.confidence);
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
        static_cast<double>(upsert_sum_ms),
        static_cast<double>(total_ms),
        static_cast<double>(seg_res->detections.detections.size()),
        static_cast<double>(accepted_count),
        static_cast<double>(rejected_count));
      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Continuous perception frame: detections=%zu accepted=%zu rejected=%zu seg=%ld cutout=%ld coarse=%ld upsert=%ld total=%ld ms",
        seg_res->detections.detections.size(),
        accepted_count,
        rejected_count,
        seg_ms,
        cutout_sum_ms,
        coarse_sum_ms,
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
