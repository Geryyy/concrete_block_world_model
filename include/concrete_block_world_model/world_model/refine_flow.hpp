#pragma once

#include <chrono>
#include <functional>
#include <string>

#include <Eigen/Dense>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/header.hpp>

#include "concrete_block_world_model_interfaces/msg/block.hpp"
#include "concrete_block_world_model/world_model/roi_processing.hpp"

namespace cbp::world_model
{

struct PoseFusionConfig
{
  bool enabled{true};
  std::string mode{"position_from_registration_orientation_from_fk"};
  double max_translation_jump_m{0.35};
  double max_z_delta_m{0.25};
  bool debug_log{true};
};

struct RefineRequest
{
  uint64_t sequence{0};
  std::string target_block_id;
  double registration_timeout_s{3.0};
};

struct RefineGraspedConfig
{
  RoiInputConfig roi_cfg;
  PoseFusionConfig pose_fusion;
  bool debug_detection_overlay_enabled{true};
  bool debug_refine_grasped_roi_input_enabled{true};
  std::string object_class{"concrete_block"};
};

struct RefineBlockConfig
{
  bool use_pose_roi{false};
  RoiInputConfig roi_cfg;
  double refine_target_max_distance_m{1.2};
  bool debug_detection_overlay_enabled{true};
};

struct RefineFlowRuntime
{
  rclcpp::Logger logger{rclcpp::get_logger("refine_flow")};

  std::function<bool()> registration_ready;
  std::function<void(const std_msgs::msg::Header &)> publish_persistent_world;
  std::function<void(uint64_t, bool, const std::string &)> complete_one_shot;
  std::function<void()> reset_busy;
  std::function<void(int64_t, int64_t, int64_t, int64_t)> record_timing;

  std::function<void(const sensor_msgs::msg::Image &)> publish_debug_overlay;
  std::function<void(const sensor_msgs::msg::Image &)> publish_roi_input;
  std::function<bool(const std::string &, concrete_block_world_model_interfaces::msg::Block &)> get_expected_target;
  std::function<bool(ProjectionIntrinsics &)> get_projection_intrinsics;

  std::function<bool(
      const std_msgs::msg::Header &,
      Eigen::Vector3d &,
      Eigen::Vector3d &,
      Eigen::Quaterniond &,
      std::string &)> lookup_predicted_grasped_pose;
  std::function<bool(
      const std_msgs::msg::Header &,
      const Eigen::Vector3d &,
      Eigen::Vector3d &,
      std::string &)> world_point_to_camera;

  std::function<bool(
      const sensor_msgs::msg::Image &,
      double,
      ros2_yolos_cpp::srv::SegmentImage::Response::SharedPtr &,
      std::string &)> run_segmentation_sync;

  std::function<bool(
      uint32_t,
      const sensor_msgs::msg::Image &,
      const sensor_msgs::msg::PointCloud2 &,
      const std_msgs::msg::Header &,
      double,
      concrete_block_world_model_interfaces::msg::Block &,
      std::string &,
      const std::string &)> run_registration_sync;

  std::function<bool(
      concrete_block_world_model_interfaces::msg::Block &,
      std::string &,
      std::string &)> upsert_block;
};

void processRefineGraspedWithFkRoi(
  const RefineRequest & request,
  const RefineGraspedConfig & cfg,
  const RefineFlowRuntime & rt,
  const sensor_msgs::msg::Image::ConstSharedPtr & image,
  const sensor_msgs::msg::PointCloud2::ConstSharedPtr & cloud,
  const std::chrono::steady_clock::time_point & t_start);

bool tryProcessRefineBlockWithPoseRoi(
  const RefineRequest & request,
  const RefineBlockConfig & cfg,
  const RefineFlowRuntime & rt,
  const sensor_msgs::msg::Image::ConstSharedPtr & image,
  const sensor_msgs::msg::PointCloud2::ConstSharedPtr & cloud,
  const std::chrono::steady_clock::time_point & t_start);

}  // namespace cbp::world_model
