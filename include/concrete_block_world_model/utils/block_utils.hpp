#pragma once

#include <cstdint>
#include <string>

#include <geometry_msgs/msg/pose.hpp>
#include "concrete_block_world_model_interfaces/msg/block.hpp"
#include "concrete_block_world_model_interfaces/msg/block_array.hpp"

std::string poseStatusToString(int status);
std::string taskStatusToString(int status);

bool isKnownPoseStatus(int status);
bool isKnownTaskStatus(int status);

// Auto-assigned ids follow the "block_<detection_id>" convention; anything
// else (e.g. seeded "b0") is a user-named block.
std::string detectionBlockId(uint32_t detection_id);
bool isAutoAssignedBlockId(const std::string & id);

double poseDistance(
  const geometry_msgs::msg::Pose & a,
  const geometry_msgs::msg::Pose & b);
