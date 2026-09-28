#include "concrete_block_world_model/nodes/perception_orchestrator_node.hpp"

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
    // One consumer: publishSceneDiscoveryPoseOverlay, which projects block wireframes onto
    // the rectified Blackfly image with P (not K).  A message without a usable P is of no
    // use to it, so it is not cached at all.
    if (msg->p[0] <= 0.0 || msg->p[5] <= 0.0) {
      return;
    }
    std::lock_guard<std::mutex> lock(camera_info_mutex_);
    scene_discovery_camera_infos_.push_back(msg);
    constexpr std::size_t kCameraInfoCacheCapacity = 30U;
    while (scene_discovery_camera_infos_.size() > kCameraInfoCacheCapacity) {
      scene_discovery_camera_infos_.pop_front();
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
    // FK tracking runs from a wall timer as well as from sensor callbacks.
    // A timer tick at `now()` can be a few milliseconds newer than the most
    // recent joint-state TF, especially in Gazebo.  The latest transform is
    // the correct physical state for that use case; do not leave a carried
    // block frozen at its pickup pose merely because an exact stamped lookup
    // extrapolates forward.
    try {
      const auto latest_tf_world_tcp = tf_buffer_->lookupTransform(
        world_frame_, refine_grasped_tcp_frame_, tf2::TimePointZero,
        tf2::durationFromSec(0.2));
      T_world_tcp = transformToEigen(latest_tf_world_tcp);
      RCLCPP_DEBUG(
        get_logger(),
        "TCP FK lookup at stamp failed (%s); using latest transform", ex.what());
      return true;
    } catch (const tf2::TransformException & latest_ex) {
      reason = std::string("TF lookup failed at requested stamp: ") + ex.what() +
        "; latest lookup failed: " + latest_ex.what();
      return false;
    }
  }
}

bool PerceptionOrchestratorNode::graspOffsetIsPlausible(
    const Eigen::Matrix4d & T_tcp_block, std::string & reason) const
  {
    if (!grasp_offset_nominal_configured_) {
      return true;
    }
    const double dev = (T_tcp_block.block<3, 1>(0, 3) - T_tcp_block_.block<3, 1>(0, 3)).norm();
    if (dev > refine_grasped_grasp_offset_max_deviation_m_) {
      reason = "grasp offset deviates " + std::to_string(dev) +
        " m from nominal (max " +
        std::to_string(refine_grasped_grasp_offset_max_deviation_m_) + " m)";
      return false;
    }
    return true;
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

    const Eigen::Matrix4d T_tcp_block = cbpwm::graspOffsetFromPose(T_world_tcp, block_pose);
    if (!graspOffsetIsPlausible(T_tcp_block, reason)) {
      return false;
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
