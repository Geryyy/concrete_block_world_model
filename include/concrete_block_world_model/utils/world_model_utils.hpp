#pragma once

#include <array>
#include <string>
#include <vector>

#include <std_msgs/msg/header.hpp>
#include <vision_msgs/msg/detection2_d.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "concrete_block_world_model_interfaces/msg/block.hpp"
#include "concrete_block_world_model_interfaces/msg/planning_scene_object.hpp"

namespace cbp::world_model
{

// Detection confidence = top hypothesis score, or 1.0 when no hypothesis is present.
double detectionConfidence(const vision_msgs::msg::Detection2D & det);

// Union-merge detections whose 2D boxes overlap strongly, transitively. Two detections are
// merged when the intersection over the smaller box is >= containment_ratio OR their IoU is
// >= iou_threshold. Each merged detection gets the union bounding box; its class/hypothesis
// are taken from the highest-confidence member. Non-overlapping detections pass through
// unchanged. Because the registration cutout is the shared semantic mask cropped to a
// detection's bbox, the union box yields a more complete cutout (e.g. recovering the front
// face that a nested duplicate box would clip away). Output order is first-seen group order.
vision_msgs::msg::Detection2DArray mergeOverlappingDetections(
  const vision_msgs::msg::Detection2DArray & detections,
  double containment_ratio,
  double iou_threshold);

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
