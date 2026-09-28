#pragma once

#include <array>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include "concrete_block_world_model/utils/world_model_utils.hpp"

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
  std::string world_frame{"world"};
  double association_max_distance_m{0.45};
  double association_max_age_s{20.0};
  double min_update_confidence{0.25};
  double scene_discovery_min_detector_confidence{0.25};
  double scene_discovery_association_max_distance_m{0.45};
  bool task_move_fk_tracking_enabled{true};
  // Read from `debug.scene_discovery_dump.tf_frames` -- see the declaration for why that name
  // outlived the dump.
  std::vector<std::string> scene_discovery_capture_tf_frames{"K0_mounting_base"};
  double marker_refresh_period_s{0.5};
  // How long an unchanged /crane/collision_scene may go unrepublished, s. The planner ages this
  // topic against its own `max_scene_age` (crane_planning issue 096) and the subscription is
  // transient-local, so a scene published once and then suppressed as "unchanged" is
  // indistinguishable there from a world model that has died. Republishing the same scene on this
  // period is what makes the age mean "the world model has stopped" rather than "nothing moved".
  // It rides `marker_refresh_period_s`, so it is only ever honoured to that resolution.
  double collision_scene_heartbeat_s{2.0};

  // What is left of `refine_grasped`: the TCP frame FK tracking and the REFINE_GRASPED prior
  // are built from, the nominal grasp offset and its plausibility bound, and the camera_info
  // topic the scene-discovery pose overlay takes its projection from.
  std::string refine_grasped_tcp_frame{"elastic/K8_tool_center_point"};
  std::string refine_grasped_camera_info_topic{"/blackfly_rotated/camera_info"};
  std::vector<double> refine_grasped_tcp_to_block_xyz{0.0, 0.0, 0.0};
  std::vector<double> refine_grasped_tcp_to_block_rpy{0.0, 0.0, 0.0};
  double refine_grasped_grasp_offset_max_deviation_m{1.0};

  std::string initial_blocks_yaml{};
  std::vector<InitialBlockConfig> initial_blocks;
  std::string static_scene_objects_yaml{};
  std::vector<StaticSceneObjectConfig> static_scene_objects;
  VehicleBoxConfig vehicle_box;
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
