#include "concrete_block_world_model/nodes/perception_orchestrator_node.hpp"

#include "concrete_block_world_model/utils/img_utils.hpp"
#include "concrete_block_world_model/utils/visu_utils.hpp"
#include "concrete_block_world_model/utils/world_model_utils.hpp"
#include "concrete_block_world_model/world_model/continuous_perception.hpp"
#include "concrete_block_world_model/world_model/scene_discovery_flow.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

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
          continuous_tracks_.erase(it->first);
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
