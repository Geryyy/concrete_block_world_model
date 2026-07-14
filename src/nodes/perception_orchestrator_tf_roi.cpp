#include "concrete_block_world_model/nodes/perception_orchestrator_node.hpp"

#include "concrete_block_world_model/world_model/roi_processing.hpp"
#include "concrete_block_world_model/world_model/refine_flow.hpp"

Eigen::Matrix4d PerceptionOrchestratorNode::transformToEigen(const geometry_msgs::msg::TransformStamped & tf)
  {
    Eigen::Quaterniond q(
      tf.transform.rotation.w,
      tf.transform.rotation.x,
      tf.transform.rotation.y,
      tf.transform.rotation.z);
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = q.normalized().toRotationMatrix();
    T(0, 3) = tf.transform.translation.x;
    T(1, 3) = tf.transform.translation.y;
    T(2, 3) = tf.transform.translation.z;
    return T;
  }

void PerceptionOrchestratorNode::cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
  {
    if (msg->k.size() < 9) {
      return;
    }
    CameraIntrinsics intr;
    intr.fx = msg->k[0];
    intr.fy = msg->k[4];
    intr.cx = msg->k[2];
    intr.cy = msg->k[5];
    intr.width = msg->width;
    intr.height = msg->height;
    intr.valid = intr.fx > 0.0 && intr.fy > 0.0;

    if (!intr.valid) {
      return;
    }
    std::lock_guard<std::mutex> lock(camera_info_mutex_);
    camera_intrinsics_ = intr;
    camera_info_frame_id_ = msg->header.frame_id;
  }

bool PerceptionOrchestratorNode::lookupPredictedGraspedPose(
    const std::string & block_id,
    const std_msgs::msg::Header & header,
    Eigen::Vector3d & p_world,
    Eigen::Vector3d & p_camera,
    Eigen::Quaterniond & q_world,
    std::string & reason)
  {
    if (!tf_buffer_) {
      reason = "TF buffer not initialized";
      return false;
    }

    try {
      const auto tf_world_tcp = tf_buffer_->lookupTransform(
        world_frame_,
        refine_grasped_tcp_frame_,
        rclcpp::Time(header.stamp),
        rclcpp::Duration::from_seconds(0.2));
      std::string camera_frame = refine_grasped_camera_frame_;
      if (camera_frame.empty()) {
        camera_frame = header.frame_id;
      }
      if (camera_frame.empty()) {
        std::lock_guard<std::mutex> lock(camera_info_mutex_);
        camera_frame = camera_info_frame_id_;
      }
      if (camera_frame.empty()) {
        reason = "camera frame unresolved (no override, image frame, or camera_info frame)";
        return false;
      }

      const auto tf_camera_world = tf_buffer_->lookupTransform(
        camera_frame,
        world_frame_,
        rclcpp::Time(header.stamp),
        rclcpp::Duration::from_seconds(0.2));

      const Eigen::Matrix4d T_world_tcp = transformToEigen(tf_world_tcp);
      const Eigen::Matrix4d T_camera_world = transformToEigen(tf_camera_world);
      // Prefer the per-block captured grasp offset (populated on TASK_MOVE); fall back to
      // the nominal T_tcp_block_ when none is available, so refine_grasped and FK tracking
      // share the same offset and their predicted poses agree.
      const Eigen::Matrix4d T_world_block_pred = T_world_tcp * resolveGraspOffset(block_id);
      p_world = T_world_block_pred.block<3, 1>(0, 3);
      q_world = Eigen::Quaterniond(T_world_block_pred.block<3, 3>(0, 0)).normalized();
      const Eigen::Vector4d p_block_world_h(
        p_world.x(),
        p_world.y(),
        p_world.z(),
        1.0);
      const Eigen::Vector4d p_block_camera_h = T_camera_world * p_block_world_h;

      p_camera = p_block_camera_h.head<3>();
      return true;
    } catch (const tf2::TransformException & ex) {
      reason = std::string("TF lookup failed: ") + ex.what();
      return false;
    }
  }

bool PerceptionOrchestratorNode::lookupTcpInWorld(
    const std_msgs::msg::Header & header,
    Eigen::Matrix4d & T_world_tcp,
    std::string & reason)
  {
    if (!tf_buffer_) {
      reason = "TF buffer not initialized";
      return false;
    }

    try {
      const auto tf_world_tcp = tf_buffer_->lookupTransform(
        world_frame_,
        refine_grasped_tcp_frame_,
        rclcpp::Time(header.stamp),
        rclcpp::Duration::from_seconds(0.2));
      T_world_tcp = transformToEigen(tf_world_tcp);
      return true;
    } catch (const tf2::TransformException & ex) {
      reason = std::string("TF lookup failed: ") + ex.what();
      return false;
    }
  }

bool PerceptionOrchestratorNode::captureGraspOffsetFromPose(
    const geometry_msgs::msg::Pose & block_pose,
    Eigen::Matrix4d & out_offset,
    std::string & reason)
  {
    // Latest TCP pose (zero stamp => tf2 returns the most recent available transform,
    // which is the TCP at the moment of grasping).
    std_msgs::msg::Header tcp_header;
    Eigen::Matrix4d T_world_tcp = Eigen::Matrix4d::Identity();
    if (!lookupTcpInWorld(tcp_header, T_world_tcp, reason)) {
      return false;
    }

    Eigen::Matrix4d T_world_block = Eigen::Matrix4d::Identity();
    T_world_block.block<3, 3>(0, 0) =
      Eigen::Quaterniond(
      block_pose.orientation.w, block_pose.orientation.x,
      block_pose.orientation.y, block_pose.orientation.z).normalized().toRotationMatrix();
    T_world_block.block<3, 1>(0, 3) =
      Eigen::Vector3d(block_pose.position.x, block_pose.position.y, block_pose.position.z);

    const Eigen::Matrix4d T_tcp_block = T_world_tcp.inverse() * T_world_block;

    // Plausibility gate: reject captures that stray too far from a configured nominal.
    if (grasp_offset_nominal_configured_) {
      const double dev =
        (T_tcp_block.block<3, 1>(0, 3) - T_tcp_block_.block<3, 1>(0, 3)).norm();
      if (dev > refine_grasped_grasp_offset_max_deviation_m_) {
        reason = "captured grasp offset deviates " + std::to_string(dev) +
          " m from nominal (max " +
          std::to_string(refine_grasped_grasp_offset_max_deviation_m_) + " m)";
        return false;
      }
    }

    out_offset = T_tcp_block;
    return true;
  }

Eigen::Matrix4d PerceptionOrchestratorNode::resolveGraspOffset(const std::string & block_id)
  {
    std::lock_guard<std::mutex> lock(persistent_world_mutex_);
    const auto it = task_move_grasp_offsets_.find(block_id);
    if (it != task_move_grasp_offsets_.end()) {
      return it->second;
    }
    return T_tcp_block_;
  }

bool PerceptionOrchestratorNode::resolveCameraFrame(
    const std_msgs::msg::Header & header,
    std::string & camera_frame,
    std::string & reason)
  {
    camera_frame = refine_grasped_camera_frame_;
    if (camera_frame.empty()) {
      camera_frame = header.frame_id;
    }
    if (camera_frame.empty()) {
      std::lock_guard<std::mutex> lock(camera_info_mutex_);
      camera_frame = camera_info_frame_id_;
    }
    if (camera_frame.empty()) {
      reason = "camera frame unresolved (no override, image frame, or camera_info frame)";
      return false;
    }
    return true;
  }

bool PerceptionOrchestratorNode::worldPointToCamera(
    const std_msgs::msg::Header & header,
    const Eigen::Vector3d & p_world,
    Eigen::Vector3d & p_camera,
    std::string & reason)
  {
    if (!tf_buffer_) {
      reason = "TF buffer not initialized";
      return false;
    }

    std::string camera_frame;
    if (!resolveCameraFrame(header, camera_frame, reason)) {
      return false;
    }

    try {
      const auto tf_camera_world = tf_buffer_->lookupTransform(
        camera_frame,
        world_frame_,
        rclcpp::Time(header.stamp),
        rclcpp::Duration::from_seconds(0.2));
      const Eigen::Matrix4d T_camera_world = transformToEigen(tf_camera_world);
      const Eigen::Vector4d p_world_h(p_world.x(), p_world.y(), p_world.z(), 1.0);
      const Eigen::Vector4d p_camera_h = T_camera_world * p_world_h;
      p_camera = p_camera_h.head<3>();
      return true;
    } catch (const tf2::TransformException & ex) {
      reason = std::string("TF lookup failed: ") + ex.what();
      return false;
    }
  }

cbpwm::RefineFlowRuntime PerceptionOrchestratorNode::makeRefineFlowRuntime()
  {
    cbpwm::RefineFlowRuntime rt;
    rt.logger = get_logger();
    rt.registration_ready = [this]() {
        return action_client_ && action_client_->action_server_is_ready();
      };
    rt.publish_persistent_world = [this](const std_msgs::msg::Header & header) {
        publishPersistentWorld(header);
      };
    rt.complete_one_shot = [this](uint64_t sequence, bool success, const std::string & message) {
        completeOneShotRequest(sequence, success, message);
      };
    rt.reset_busy = [this]() {resetBusy();};
    rt.record_timing = [this](int64_t seg_ms, int64_t track_ms, int64_t reg_ms, int64_t total_ms) {
        recordTiming(seg_ms, track_ms, reg_ms, total_ms);
      };
    rt.publish_debug_overlay = [this](const sensor_msgs::msg::Image & image_msg) {
        if (det_debug_pub_) {
          det_debug_pub_->publish(image_msg);
        }
      };
    rt.publish_roi_input = [this](const sensor_msgs::msg::Image & image_msg) {
        if (refine_grasped_roi_input_pub_) {
          refine_grasped_roi_input_pub_->publish(image_msg);
        }
      };
    rt.get_expected_target =
      [this](const std::string & target_id, Block & out_target) {
        std::lock_guard<std::mutex> lock(persistent_world_mutex_);
        const auto it = persistent_world_.find(target_id);
        if (it == persistent_world_.end()) {
          return false;
        }
        out_target = it->second;
        return true;
      };
    rt.get_projection_intrinsics = [this](cbpwm::ProjectionIntrinsics & out_intr) {
        std::lock_guard<std::mutex> lock(camera_info_mutex_);
        out_intr.valid = camera_intrinsics_.valid;
        out_intr.fx = camera_intrinsics_.fx;
        out_intr.fy = camera_intrinsics_.fy;
        out_intr.cx = camera_intrinsics_.cx;
        out_intr.cy = camera_intrinsics_.cy;
        out_intr.width = camera_intrinsics_.width;
        out_intr.height = camera_intrinsics_.height;
        return out_intr.valid;
      };
    rt.lookup_predicted_grasped_pose =
      [this](
      const std::string & block_id,
      const std_msgs::msg::Header & header,
      Eigen::Vector3d & p_world,
      Eigen::Vector3d & p_camera,
      Eigen::Quaterniond & q_world,
      std::string & reason) {
        return lookupPredictedGraspedPose(block_id, header, p_world, p_camera, q_world, reason);
      };
    rt.world_point_to_camera =
      [this](
      const std_msgs::msg::Header & header,
      const Eigen::Vector3d & p_world,
      Eigen::Vector3d & p_camera,
      std::string & reason) {
        return worldPointToCamera(header, p_world, p_camera, reason);
      };
    rt.run_segmentation_sync =
      [this](
      const sensor_msgs::msg::Image & in_image,
      double timeout_s,
      SegmentSrv::Response::SharedPtr & out_response,
      std::string & out_reason) {
        return runSegmentationSync(in_image, timeout_s, out_response, out_reason);
      };
    rt.run_registration_sync =
      [this](
      uint32_t detection_id,
      const sensor_msgs::msg::Image & mask,
      const sensor_msgs::msg::PointCloud2 & cloud,
      const std_msgs::msg::Header & header,
      double timeout_s,
      Block & out_block,
      std::string & out_reason,
      const std::string & object_class_override) {
        return runRegistrationSync(
          detection_id,
          mask,
          cloud,
          header,
          timeout_s,
          out_block,
          out_reason,
          object_class_override);
      };
    rt.upsert_block = [](Block &, std::string &, std::string &) {
        return false;
      };
    return rt;
  }

void PerceptionOrchestratorNode::processRefineGraspedWithFkRoi(
    const sensor_msgs::msg::Image::ConstSharedPtr & image,
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr & cloud,
    const OneShotRequest & run_request,
    const std::chrono::steady_clock::time_point & t_start)
  {
    cbpwm::RefineRequest req;
    req.sequence = run_request.sequence;
    req.target_block_id = run_request.target_block_id;
    req.registration_timeout_s = run_request.registration_timeout_s;

    cbpwm::RefineGraspedConfig cfg;
    cfg.roi_cfg = refine_grasped_roi_cfg_;
    cfg.debug_detection_overlay_enabled = debug_detection_overlay_enabled_.load();
    cfg.debug_refine_grasped_roi_input_enabled = debug_refine_grasped_roi_input_enabled_.load();
    cfg.object_class = object_class_;
    cfg.pose_fusion = refine_grasped_pose_fusion_;

    auto rt = makeRefineFlowRuntime();
    rt.upsert_block = [this, &run_request, cloud](Block & block, std::string & assigned_id, std::string & reason) {
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

    cbpwm::processRefineGraspedWithFkRoi(req, cfg, rt, image, cloud, t_start);
  }

bool PerceptionOrchestratorNode::tryProcessRefineBlockWithPoseRoi(
    const sensor_msgs::msg::Image::ConstSharedPtr & image,
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr & cloud,
    const OneShotRequest & run_request,
    const std::chrono::steady_clock::time_point & t_start)
  {
    cbpwm::RefineRequest req;
    req.sequence = run_request.sequence;
    req.target_block_id = run_request.target_block_id;
    req.registration_timeout_s = run_request.registration_timeout_s;

    cbpwm::RefineBlockConfig cfg;
    cfg.use_pose_roi = refine_block_use_pose_roi_;
    cfg.roi_cfg = refine_block_roi_cfg_;
    cfg.refine_target_max_distance_m = runtime_cfg_.refine_target_max_distance_m;
    cfg.debug_detection_overlay_enabled = debug_detection_overlay_enabled_.load();

    auto rt = makeRefineFlowRuntime();
    rt.upsert_block = [this, &run_request, cloud](Block & block, std::string & assigned_id, std::string & reason) {
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

    return cbpwm::tryProcessRefineBlockWithPoseRoi(req, cfg, rt, image, cloud, t_start);
  }
