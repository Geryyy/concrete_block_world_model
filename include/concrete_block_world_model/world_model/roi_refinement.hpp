#pragma once

#include <string>

#include <Eigen/Dense>
#include <geometry_msgs/msg/pose.hpp>

namespace cbp::world_model
{

struct PoseFusionResult
{
  bool success{false};
  double residual_norm{0.0};
  double z_delta{0.0};
  std::string reason;
};

PoseFusionResult fuseRegistrationPositionWithFkOrientation(
  geometry_msgs::msg::Pose & in_out_pose,
  const Eigen::Vector3d & p_world_fk,
  const Eigen::Quaterniond & q_world_fk,
  double max_translation_jump_m,
  double max_z_delta_m);

}  // namespace cbp::world_model

