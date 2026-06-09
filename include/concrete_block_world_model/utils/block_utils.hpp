#pragma once

#include <geometry_msgs/msg/pose.hpp>
#include "concrete_block_world_model_interfaces/msg/block.hpp"
#include "concrete_block_world_model_interfaces/msg/block_array.hpp"

std::string poseStatusToString(int status);
std::string taskStatusToString(int status);

double poseDistance(
  const geometry_msgs::msg::Pose & a,
  const geometry_msgs::msg::Pose & b);
