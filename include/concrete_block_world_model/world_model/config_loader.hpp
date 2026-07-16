#pragma once

#include <array>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "concrete_block_world_model/world_model/refine_flow.hpp"

namespace cbp::world_model
{

struct InitialBlockConfig
{
  std::string id;
  std::string frame_id{"world"};
  std::array<double, 3> position{0.0, 0.0, 0.0};
  double yaw_deg{0.0};
  int pose_status{1};
  int task_status{3};
  double confidence{1.0};
};

struct StaticSceneObjectConfig
{
  std::string id;
  std::string frame_id{"world"};
  std::array<double, 3> position{0.0, 0.0, 0.0};
  std::array<double, 3> rpy_deg{0.0, 0.0, 0.0};
  std::array<double, 3> dimensions{1.0, 1.0, 1.0};
};

struct WorldModelConfig
{
  double min_fitness{0.3};
  double max_rmse{0.05};
  std::string object_class{"concrete_block"};
  std::string world_frame{"world"};
  double max_sync_delta_s{0.06};
  double association_max_distance_m{0.45};
  double association_max_age_s{20.0};
  double min_update_confidence{0.25};
  bool task_move_fk_tracking_enabled{true};
  double refine_target_max_distance_m{1.2};
  bool scene_discovery_merge_enabled{true};
  double scene_discovery_merge_containment_ratio{0.3};
  double scene_discovery_merge_iou_threshold{0.5};
  bool scene_discovery_coarse_fallback_enabled{true};
  int scene_discovery_coarse_fallback_min_points{120};
  double coarse_surface_square_ratio_thresh{1.35};
  double coarse_front_center_offset_square_m{0.45};
  double coarse_front_center_offset_rect_m{0.30};
  bool debug_detection_overlay_enabled{true};
  bool debug_refine_grasped_roi_input_enabled{true};
  bool debug_scene_discovery_dump_enabled{false};
  std::string debug_scene_discovery_dump_dir{"scene_discovery_dump"};
  bool perf_log_timing_enabled{true};
  int perf_log_every_n_frames{20};
  double marker_refresh_period_s{0.5};

  bool refine_grasped_use_fk_roi{true};
  std::string refine_grasped_tcp_frame{"elastic/K8_tool_center_point"};
  std::string refine_grasped_camera_frame{};
  std::string refine_grasped_camera_info_topic{"/blackfly_rotated/camera_info"};
  double refine_grasped_min_depth_m{0.5};
  double refine_grasped_max_depth_m{30.0};
  double refine_grasped_segmentation_timeout_s{3.0};
  bool refine_grasped_use_black_bg{false};
  int refine_grasped_blur_kernel_size{31};
  std::vector<double> refine_grasped_tcp_to_block_xyz{0.0, 0.0, 0.0};
  std::vector<double> refine_grasped_tcp_to_block_rpy{0.0, 0.0, 0.0};
  double refine_grasped_grasp_offset_max_deviation_m{1.0};
  std::vector<double> refine_grasped_roi_size_m{0.60, 0.40};
  PoseFusionConfig refine_grasped_pose_fusion;

  bool refine_block_use_pose_roi{false};
  std::vector<double> refine_block_roi_size_m{1.20, 1.00};
  double refine_block_min_depth_m{0.5};
  double refine_block_max_depth_m{30.0};
  double refine_block_segmentation_timeout_s{3.0};
  bool refine_block_use_black_bg{false};
  int refine_block_blur_kernel_size{31};
  std::string initial_blocks_yaml{};
  std::vector<InitialBlockConfig> initial_blocks;
  std::string static_scene_objects_yaml{};
  std::vector<StaticSceneObjectConfig> static_scene_objects;
  std::array<double, 3> block_dimensions_m{0.6, 0.9, 0.6};
};

WorldModelConfig loadWorldModelConfig(rclcpp::Node & node);

void normalizeWorldModelConfig(rclcpp::Logger logger, WorldModelConfig & cfg);

double vectorComponent(
  rclcpp::Logger logger,
  const std::vector<double> & values,
  size_t index,
  double fallback,
  const char * param_name);

}  // namespace cbp::world_model
