#include "concrete_block_world_model/nodes/perception_orchestrator_node.hpp"

#include "concrete_block_world_model/utils/img_utils.hpp"

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
        recordContinuousTrackMisses({}, rclcpp::Time(cloud->header.stamp).seconds());
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
      int continuous_registration_attempts = 0;
      ContinuousStageTimings timings;
      const Eigen::Vector3d * camera_origin =
        have_camera_origin ? &camera_origin_world : nullptr;
      std::unordered_set<std::string> observed_track_ids;

      const auto candidates = buildContinuousCandidates(
        *seg_res, cloud, full_mask, camera_origin, timings, rejected_mask, rejected_count);

      const auto groups = cbpwm::groupContinuousCandidates(
        candidates,
        continuous_cfg_.mask_merge,
        get_logger(),
        *get_clock());

      cv::Mat merged_mask_debug = cv::Mat::zeros(full_mask.size(), CV_8UC1);
      for (size_t group_index = 0; group_index < groups.size(); ++group_index) {
        const auto & group = groups[group_index];

        cbpwm::BlockObservation observation;
        if (!buildContinuousObservation(
            group,
            group_index,
            candidates,
            cloud,
            camera_origin,
            continuous_registration_attempts,
            timings,
            observation))
        {
          ++rejected_count;
          cv::bitwise_or(rejected_mask, group.merged_mask, rejected_mask);
          continue;
        }

        std::string assigned_id;
        std::string upsert_reason;
        if (!applyContinuousObservation(
            observation, cloud->header, timings, assigned_id, upsert_reason))
        {
          if (!assigned_id.empty()) {
            observed_track_ids.insert(assigned_id);
          }
          ++rejected_count;
          cv::bitwise_or(rejected_mask, group.merged_mask, rejected_mask);
          RCLCPP_INFO_THROTTLE(
            get_logger(), *get_clock(), 2000,
            "Continuous upsert rejected: group=%zu fragments=[%s] confidence=%.3f reason=%s",
            group_index,
            cbpwm::fragmentIndices(group, candidates).c_str(),
            observation.block.confidence,
            upsert_reason.c_str());
          continue;
        }

        ++accepted_count;
        if (!assigned_id.empty()) {
          observed_track_ids.insert(assigned_id);
        }
        for (const size_t candidate_index : group.candidate_indices) {
          accepted_by_detection[candidates.at(candidate_index).detection_index] = true;
        }
        cv::bitwise_or(accepted_mask, group.merged_mask, accepted_mask);
        cv::bitwise_or(merged_mask_debug, group.merged_mask, merged_mask_debug);
        RCLCPP_INFO_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "Continuous observation accepted: group=%zu fragments=[%s] assigned=%s pose=%s mask_pixels=%d cutout_points=%u confidence=%.3f",
          group_index,
          cbpwm::fragmentIndices(group, candidates).c_str(),
          assigned_id.c_str(),
          observation.precise ? "PRECISE" : "COARSE",
          observation.mask_pixels,
          observation.cutout_points,
          observation.block.confidence);
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

      recordContinuousTrackMisses(
        observed_track_ids,
        rclcpp::Time(cloud->header.stamp).seconds());
      publishPersistentWorld(cloud->header);
      const auto t_end = std::chrono::steady_clock::now();
      const auto seg_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(t_after_seg - t_start).count();
      const auto total_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
      recordTiming(seg_ms, 0, 0, total_ms);
      publish_continuous_timing(
        static_cast<double>(seg_ms),
        static_cast<double>(timings.cutout_ms),
        static_cast<double>(timings.coarse_ms),
        static_cast<double>(timings.registration_ms),
        static_cast<double>(timings.upsert_ms),
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
        timings.cutout_ms,
        timings.coarse_ms,
        timings.registration_ms,
        timings.upsert_ms,
        total_ms);
      resetBusy();
    } catch (const std::exception & e) {
      RCLCPP_ERROR(get_logger(), "Continuous perception failed: %s", e.what());
      publishPersistentWorld(cloud->header);
      resetBusy();
    }
  }

void PerceptionOrchestratorNode::recordContinuousTrackMisses(
    const std::unordered_set<std::string> & observed_track_ids,
    double now_s)
  {
    if (
      !continuous_cfg_.filtering_enabled ||
      !continuous_cfg_.filtering.operational_confidence_enabled)
    {
      return;
    }

    std::lock_guard<std::mutex> lock(persistent_world_mutex_);
    for (auto & kv : continuous_tracks_) {
      if (observed_track_ids.count(kv.first) > 0U) {
        continue;
      }
      cbpwm::predict(
        kv.second,
        now_s - kv.second.last_update_s,
        continuous_cfg_.filtering);
      kv.second.last_update_s = now_s;
      cbpwm::recordMiss(kv.second, continuous_cfg_.filtering);
    }
  }

std::vector<cbpwm::ContinuousMaskCandidate> PerceptionOrchestratorNode::buildContinuousCandidates(
    const SegmentSrv::Response & seg_res,
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr & cloud,
    const cv::Mat & full_mask,
    const Eigen::Vector3d * camera_origin_world,
    ContinuousStageTimings & timings,
    cv::Mat & rejected_mask,
    size_t & rejected_count)
  {
    std::vector<cbpwm::ContinuousMaskCandidate> candidates;
    for (size_t i = 0; i < seg_res.detections.detections.size(); ++i) {
      const auto & det = seg_res.detections.detections[i];
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
      timings.cutout_ms += std::chrono::duration_cast<std::chrono::milliseconds>(
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
        camera_origin_world,
        coarse_block,
        coarse_reason,
        continuous_cfg_.min_valid_cloud_points);
      timings.coarse_ms += std::chrono::duration_cast<std::chrono::milliseconds>(
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
      const cv::Rect bbox = toCvRect(det) & cv::Rect(0, 0, full_mask.cols, full_mask.rows);
      candidates.push_back(
        cbpwm::ContinuousMaskCandidate{
          i,
          det.results.empty() ? std::string{} : det.results.front().hypothesis.class_id,
          bbox,
          binary_mask.clone(),
          quality,
          coarse_block,
          cbpwm::detectionConfidence(det),
          cutout_cloud.width * cutout_cloud.height});
    }

    return candidates;
  }

bool PerceptionOrchestratorNode::findContinuousRegistrationPrior(
    const Block & coarse_block,
    const std_msgs::msg::Header & header,
    Block & out_prior)
  {
    const rclcpp::Time now_stamp(header.stamp, get_clock()->get_clock_type());
    double best_distance = std::numeric_limits<double>::infinity();
    bool found = false;

    std::lock_guard<std::mutex> lock(persistent_world_mutex_);
    for (const auto & kv : persistent_world_) {
      const auto & candidate = kv.second;
      if (candidate.task_status == Block::TASK_MOVE ||
        candidate.task_status == Block::TASK_REMOVED ||
        candidate.pose_status == Block::POSE_UNKNOWN)
      {
        continue;
      }

      const rclcpp::Time seen(candidate.last_seen, get_clock()->get_clock_type());
      if ((now_stamp - seen).seconds() > continuous_cfg_.association_max_age_s) {
        continue;
      }

      const double distance = cbpwm::blockDistance(coarse_block, candidate);
      if (distance <= continuous_cfg_.association_max_distance_m && distance < best_distance) {
        best_distance = distance;
        out_prior = candidate;
        found = true;
      }
    }

    return found;
  }

bool PerceptionOrchestratorNode::buildContinuousObservation(
    const cbpwm::ContinuousMaskGroup & group,
    size_t group_index,
    const std::vector<cbpwm::ContinuousMaskCandidate> & candidates,
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr & cloud,
    const Eigen::Vector3d * camera_origin_world,
    int & registration_attempts,
    ContinuousStageTimings & timings,
    cbpwm::BlockObservation & out_observation)
  {
    const int merged_pixels = cv::countNonZero(group.merged_mask);
    int source_pixels = 0;
    uint32_t source_cutout_points = 0;
    double max_confidence = 0.0;
    for (const size_t candidate_index : group.candidate_indices) {
      const auto & candidate = candidates.at(candidate_index);
      source_pixels += candidate.quality.mask_pixels;
      source_cutout_points += candidate.cutout_points;
      max_confidence = std::max(max_confidence, candidate.confidence);
    }
    const std::string fragments = cbpwm::fragmentIndices(group, candidates);

    const auto mask_msg =
      cv_bridge::CvImage(cloud->header, "mono8", group.merged_mask).toImageMsg();
    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 1000,
      "Continuous merged cutout input: group=%zu fragments=[%s] mask_pixels=%d source_pixels=%d source_cutout_points=%u",
      group_index,
      fragments.c_str(),
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
    timings.cutout_ms += std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - t_cutout_start).count();
    if (!got_cutout) {
      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Continuous merged cutout rejected: group=%zu fragments=[%s] merged_pixels=%d reason=%s",
        group_index,
        fragments.c_str(),
        merged_pixels,
        cutout_reason.c_str());
      return false;
    }

    Block coarse_block;
    std::string coarse_reason;
    const auto t_coarse_start = std::chrono::steady_clock::now();
    const bool coarse_ok = buildCoarseBlockFromCloudCentroid(
      static_cast<uint32_t>(group_index + 1U),
      *mask_msg,
      merged_cutout_cloud,
      merged_cutout_cloud.header,
      camera_origin_world,
      coarse_block,
      coarse_reason,
      continuous_cfg_.min_valid_cloud_points);
    timings.coarse_ms += std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - t_coarse_start).count();
    if (!coarse_ok) {
      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Continuous merged coarse pose rejected: group=%zu fragments=[%s] merged_pixels=%d cutout_points=%u reason=%s",
        group_index,
        fragments.c_str(),
        merged_pixels,
        merged_cutout_cloud.width * merged_cutout_cloud.height,
        coarse_reason.c_str());
      return false;
    }

    coarse_block.confidence = static_cast<float>(max_confidence);
    coarse_block.last_seen = cloud->header.stamp;
    out_observation.block = coarse_block;
    out_observation.mask_pixels = merged_pixels;
    out_observation.cutout_points = merged_cutout_cloud.width * merged_cutout_cloud.height;
    out_observation.fragment_count = group.candidate_indices.size();
    out_observation.precise = false;
    const cv::Moments mask_moments = cv::moments(group.merged_mask, true);
    if (std::abs(mask_moments.m00) > 1.0e-6) {
      out_observation.has_mask_centroid = true;
      out_observation.mask_centroid_x_px = mask_moments.m10 / mask_moments.m00;
      out_observation.mask_centroid_y_px = mask_moments.m01 / mask_moments.m00;
    }

    Block registration_prior;
    const bool have_registration_prior =
      continuous_cfg_.registration_pose_prior_enabled &&
      findContinuousRegistrationPrior(coarse_block, cloud->header, registration_prior);
    out_observation.has_registration_prior = have_registration_prior;

    if (continuous_cfg_.registration_enabled) {
      if (!action_client_ || !action_client_->action_server_is_ready()) {
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "Continuous precise registration skipped: registration action unavailable.");
      } else if (continuous_cfg_.registration_pose_prior_enabled && !have_registration_prior) {
        RCLCPP_INFO_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "Continuous precise registration skipped: group=%zu fragments=[%s] no registration prior; keeping coarse/filter observation.",
          group_index,
          fragments.c_str());
      } else if (registration_attempts >= continuous_cfg_.registration_max_per_frame) {
        RCLCPP_INFO_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "Continuous precise registration skipped: group=%zu fragments=[%s] max_per_frame=%d reached.",
          group_index,
          fragments.c_str(),
          continuous_cfg_.registration_max_per_frame);
      } else {
        ++registration_attempts;
        Block precise_block;
        std::string registration_reason;
        RCLCPP_INFO_THROTTLE(
          get_logger(), *get_clock(), 1000,
          "Continuous precise registration input: group=%zu fragments=[%s] mask_pixels=%d timeout=%.2fs pose_prior=%s%s%s",
          group_index,
          fragments.c_str(),
          merged_pixels,
          continuous_cfg_.registration_timeout_s,
          have_registration_prior ? "true" : "false",
          have_registration_prior ? " id=" : "",
          have_registration_prior ? registration_prior.id.c_str() : "");
        const auto t_registration_start = std::chrono::steady_clock::now();
        const bool registration_ok = runRegistrationSync(
          static_cast<uint32_t>(group_index + 1U),
          *mask_msg,
          *cloud,
          cloud->header,
          continuous_cfg_.registration_timeout_s,
          precise_block,
          registration_reason,
          "",
          have_registration_prior ? &registration_prior : nullptr);
        timings.registration_ms += std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - t_registration_start).count();

        bool accept_precise_registration = registration_ok;
        if (accept_precise_registration && have_registration_prior) {
          const double prior_result_distance =
            cbpwm::blockDistance(precise_block, registration_prior);
          if (prior_result_distance >
            continuous_cfg_.registration_pose_prior_max_result_distance_m)
          {
            accept_precise_registration = false;
            RCLCPP_WARN_THROTTLE(
              get_logger(), *get_clock(), 2000,
              "Continuous precise registration rejected: group=%zu fragments=[%s] prior=%s result_distance=%.3fm limit=%.3fm; falling back to coarse pose.",
              group_index,
              fragments.c_str(),
              registration_prior.id.c_str(),
              prior_result_distance,
              continuous_cfg_.registration_pose_prior_max_result_distance_m);
          }
        }

        if (accept_precise_registration) {
          out_observation.block = precise_block;
          out_observation.precise = true;
          RCLCPP_INFO_THROTTLE(
            get_logger(), *get_clock(), 1000,
            "Continuous precise registration accepted: group=%zu fragments=[%s] pos=[%.3f, %.3f, %.3f] confidence=%.3f",
            group_index,
            fragments.c_str(),
            precise_block.pose.position.x,
            precise_block.pose.position.y,
            precise_block.pose.position.z,
            precise_block.confidence);
        } else if (!registration_ok) {
          RCLCPP_WARN_THROTTLE(
            get_logger(), *get_clock(), 2000,
            "Continuous precise registration failed: group=%zu fragments=[%s] reason=%s; falling back to coarse pose.",
            group_index,
            fragments.c_str(),
            registration_reason.c_str());
        }
      }
    }

    return true;
  }

bool PerceptionOrchestratorNode::applyContinuousObservation(
    const cbpwm::BlockObservation & observation,
    const std_msgs::msg::Header & header,
    ContinuousStageTimings & timings,
    std::string & assigned_id,
    std::string & reason)
  {
    auto assoc_cfg = associationConfig();
    assoc_cfg.association_max_distance_m = continuous_cfg_.association_max_distance_m;
    assoc_cfg.association_max_age_s = continuous_cfg_.association_max_age_s;

    Block block_to_upsert = observation.block;
    bool upsert_ok = false;
    const auto t_upsert_start = std::chrono::steady_clock::now();
    if (!continuous_cfg_.filtering_enabled) {
      {
        std::lock_guard<std::mutex> lock(persistent_world_mutex_);
        upsert_ok = cbpwm::upsertRegisteredBlock(
          persistent_world_,
          world_block_counter_,
          block_to_upsert,
          cbpwm::OneShotMode::kSceneDiscovery,
          "",
          header,
          *get_clock(),
          assoc_cfg,
          assigned_id,
          reason);
      }
      timings.upsert_ms += std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t_upsert_start).count();

      if (upsert_ok) {
        RCLCPP_INFO_THROTTLE(
          get_logger(), *get_clock(), 1000,
          "Continuous world-model upsert: incoming=%s assigned=%s confidence=%.3f pose_status=%d",
          observation.block.id.c_str(),
          assigned_id.c_str(),
          block_to_upsert.confidence,
          block_to_upsert.pose_status);
      }
      return upsert_ok;
    }

    const rclcpp::Time now_stamp(header.stamp, get_clock()->get_clock_type());
    const double now_s = now_stamp.seconds();
    {
      std::lock_guard<std::mutex> lock(persistent_world_mutex_);

      for (auto it = continuous_tracks_.begin(); it != continuous_tracks_.end();) {
        if (it->second.state == cbpwm::FilteredBlockTrackState::kTentative &&
          (now_s - it->second.first_update_s) > continuous_cfg_.filtering.tentative_max_age_s)
        {
          persistent_world_.erase(it->first);
          it = continuous_tracks_.erase(it);
          continue;
        }
        ++it;
      }

      double best_score = std::numeric_limits<double>::infinity();
      bool best_has_track = false;
      for (const auto & kv : continuous_tracks_) {
        auto predicted = kv.second;
        cbpwm::predict(
          predicted,
          now_s - predicted.last_update_s,
          continuous_cfg_.filtering);
        const double d2 = cbpwm::mahalanobisDistanceSquared(predicted, observation);
        if (d2 <= continuous_cfg_.filtering.mahalanobis_gate_threshold && d2 < best_score) {
          best_score = d2;
          assigned_id = kv.first;
          best_has_track = true;
        }
      }

      if (!best_has_track) {
        for (const auto & kv : persistent_world_) {
          if (continuous_tracks_.count(kv.first) > 0U) {
            continue;
          }
          const rclcpp::Time seen(kv.second.last_seen, get_clock()->get_clock_type());
          if ((now_stamp - seen).seconds() > assoc_cfg.association_max_age_s) {
            continue;
          }
          const double dist = cbpwm::blockDistance(observation.block, kv.second);
          if (dist <= assoc_cfg.association_max_distance_m && dist < best_score) {
            best_score = dist;
            assigned_id = kv.first;
          }
        }
      }

      if (!assigned_id.empty()) {
        const auto world_it = persistent_world_.find(assigned_id);
        if (world_it != persistent_world_.end() && world_it->second.task_status == Block::TASK_MOVE) {
          reason = "continuous filtering skipped TASK_MOVE block";
          RCLCPP_INFO_THROTTLE(
            get_logger(), *get_clock(), 2000,
            "Continuous filtering skipped TASK_MOVE block: assigned=%s incoming=%s",
            assigned_id.c_str(),
            observation.block.id.c_str());
          timings.upsert_ms += std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t_upsert_start).count();
          return false;
        }
      }

      if (assigned_id.empty()) {
        ++world_block_counter_;
        assigned_id = "wm_block_" + std::to_string(world_block_counter_);
        auto track = cbpwm::initializeTrack(observation, now_s, continuous_cfg_.filtering);
        continuous_tracks_[assigned_id] = track;
        reason = "tentative track created";
        RCLCPP_INFO_THROTTLE(
          get_logger(), *get_clock(), 1000,
          "Continuous filtering tentative track: assigned=%s incoming=%s hits=%zu/%d",
          assigned_id.c_str(),
          observation.block.id.c_str(),
          track.hit_history.size(),
          continuous_cfg_.filtering.confirmation_hits);
        timings.upsert_ms += std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - t_upsert_start).count();
        return false;
      }

      auto track_it = continuous_tracks_.find(assigned_id);
      if (track_it == continuous_tracks_.end()) {
        const auto world_it = persistent_world_.find(assigned_id);
        if (world_it != persistent_world_.end()) {
          cbpwm::BlockObservation seed_obs;
          seed_obs.block = world_it->second;
          seed_obs.precise = world_it->second.pose_status == Block::POSE_PRECISE;
          track_it = continuous_tracks_.emplace(
            assigned_id,
            cbpwm::initializeTrack(seed_obs, now_s, continuous_cfg_.filtering)).first;
        } else {
          track_it = continuous_tracks_.emplace(
            assigned_id,
            cbpwm::initializeTrack(observation, now_s, continuous_cfg_.filtering)).first;
        }
      }

      cbpwm::predict(
        track_it->second,
        now_s - track_it->second.last_update_s,
        continuous_cfg_.filtering);
      track_it->second.last_update_s = now_s;
      const bool update_ok = cbpwm::gateAndUpdate(
        track_it->second,
        observation,
        continuous_cfg_.filtering,
        reason);
      if (!update_ok) {
        RCLCPP_INFO_THROTTLE(
          get_logger(), *get_clock(), 1000,
          "Continuous filtering rejected: assigned=%s incoming=%s reason=%s",
          assigned_id.c_str(),
          observation.block.id.c_str(),
          reason.c_str());
        timings.upsert_ms += std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - t_upsert_start).count();
        return false;
      }
      track_it->second.last_observation_s = now_s;

      const bool confirmed =
        track_it->second.state == cbpwm::FilteredBlockTrackState::kConfirmed;
      if (!confirmed) {
        reason = "tentative track not yet confirmed";
        RCLCPP_INFO_THROTTLE(
          get_logger(), *get_clock(), 1000,
          "Continuous filtering tentative update: assigned=%s incoming=%s history=%zu/%d",
          assigned_id.c_str(),
          observation.block.id.c_str(),
          track_it->second.hit_history.size(),
          continuous_cfg_.filtering.confirmation_hits);
        timings.upsert_ms += std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - t_upsert_start).count();
        return false;
      }

      const auto world_it = persistent_world_.find(assigned_id);
      const bool temporal_bootstrap_ok =
        cbpwm::trackHasStableTemporalBootstrap(track_it->second, continuous_cfg_.filtering);
      if (world_it == persistent_world_.end() &&
        !observation.has_registration_prior &&
        !continuous_cfg_.filtering.publish_new_tracks_without_prior &&
        !temporal_bootstrap_ok)
      {
        reason = "confirmed no-prior track awaiting temporal stability";
        RCLCPP_INFO_THROTTLE(
          get_logger(), *get_clock(), 1000,
          "Continuous filtering kept no-prior track internal: assigned=%s incoming=%s history=%zu/%d stable=%d/%d",
          assigned_id.c_str(),
          observation.block.id.c_str(),
          track_it->second.hit_history.size(),
          continuous_cfg_.filtering.confirmation_hits,
          track_it->second.stable_observations,
          continuous_cfg_.filtering.temporal_bootstrap_min_stable_observations);
        timings.upsert_ms += std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - t_upsert_start).count();
        return false;
      }

      block_to_upsert = cbpwm::toBlockMsg(
        track_it->second,
        observation.block,
        continuous_cfg_.filtering,
        now_s);
      block_to_upsert.id = assigned_id;
      block_to_upsert.last_seen = header.stamp;
      if (world_it != persistent_world_.end()) {
        if (rclcpp::Time(observation.block.last_seen, get_clock()->get_clock_type()) <
          rclcpp::Time(world_it->second.last_seen, get_clock()->get_clock_type()))
        {
          reason = "stale update (incoming older than stored state)";
          timings.upsert_ms += std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - t_upsert_start).count();
          return false;
        }
        if (world_it->second.task_status != Block::TASK_UNKNOWN) {
          block_to_upsert.task_status = world_it->second.task_status;
        }
        if (world_it->second.pose_status == Block::POSE_PRECISE || observation.precise) {
          block_to_upsert.pose_status = Block::POSE_PRECISE;
        }
      } else {
        block_to_upsert.task_status = Block::TASK_FREE;
      }
      if (block_to_upsert.pose_status == Block::POSE_UNKNOWN) {
        block_to_upsert.pose_status =
          observation.precise ? Block::POSE_PRECISE : Block::POSE_COARSE;
      }

      persistent_world_[assigned_id] = block_to_upsert;
      upsert_ok = true;
      reason = "filtered update accepted";
    }
    timings.upsert_ms += std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - t_upsert_start).count();

    if (upsert_ok) {
      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "Continuous filtered world-model upsert: incoming=%s assigned=%s confidence=%.3f pose_status=%d",
        observation.block.id.c_str(),
        assigned_id.c_str(),
        block_to_upsert.confidence,
        block_to_upsert.pose_status);
    }
    return upsert_ok;
  }
