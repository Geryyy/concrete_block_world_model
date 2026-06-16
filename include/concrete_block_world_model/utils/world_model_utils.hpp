#pragma once

#include <array>
#include <string>
#include <vector>

#include <std_msgs/msg/header.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "concrete_block_world_model_interfaces/msg/block.hpp"
#include "concrete_block_world_model_interfaces/msg/planning_scene_object.hpp"

namespace cbp::world_model
{

enum class OneShotMode
{
  kNone,
  kSceneDiscovery,
  kRefineBlock,
  kRefineGrasped
};

std::string normalizeMode(std::string mode);

OneShotMode parseOneShotMode(const std::string & mode);
const char * oneShotModeToString(OneShotMode mode);

bool isValidTaskTransition(int32_t from_status, int32_t to_status);
const char * taskStatusToString(int32_t status);
bool shouldAssociateByDistance(
  double distance_m,
  double max_distance_m,
  double confidence,
  double min_confidence);

visualization_msgs::msg::MarkerArray buildWorldMarkers(
  const std_msgs::msg::Header & header,
  const std::vector<concrete_block_world_model_interfaces::msg::Block> & blocks,
  const std::vector<concrete_block_world_model_interfaces::msg::PlanningSceneObject> & static_objects,
  const std::string & world_frame,
  const std::array<double, 3> & block_dimensions_m);

// Opaque cubes at each block's assembly goal_pose (where goal_status == GOAL_SET),
// for visualizing the target wall alongside the live (translucent) blocks.
visualization_msgs::msg::MarkerArray buildGoalMarkers(
  const std_msgs::msg::Header & header,
  const std::vector<concrete_block_world_model_interfaces::msg::Block> & blocks,
  const std::string & world_frame,
  const std::array<double, 3> & block_dimensions_m);

}  // namespace cbp::world_model
