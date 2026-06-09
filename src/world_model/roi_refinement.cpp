#include "concrete_block_world_model/world_model/roi_refinement.hpp"

#include <cmath>

namespace cbp::world_model
{

namespace
{

geometry_msgs::msg::Quaternion toGeometryMsgQuaternion(const Eigen::Quaterniond & q)
{
  geometry_msgs::msg::Quaternion out;
  out.x = q.x();
  out.y = q.y();
  out.z = q.z();
  out.w = q.w();
  return out;
}

}  // namespace

PoseFusionResult fuseRegistrationPositionWithFkOrientation(
  geometry_msgs::msg::Pose & in_out_pose,
  const Eigen::Vector3d & p_world_fk,
  const Eigen::Quaterniond & q_world_fk,
  double max_translation_jump_m,
  double max_z_delta_m)
{
  PoseFusionResult result;
  const Eigen::Vector3d p_reg(
    in_out_pose.position.x,
    in_out_pose.position.y,
    in_out_pose.position.z);
  result.residual_norm = (p_reg - p_world_fk).norm();
  result.z_delta = std::abs(p_reg.z() - p_world_fk.z());

  if (result.residual_norm > max_translation_jump_m) {
    result.reason = "translation jump too large: residual=" +
      std::to_string(result.residual_norm) + "m > " + std::to_string(max_translation_jump_m) + "m";
    return result;
  }
  if (result.z_delta > max_z_delta_m) {
    result.reason = "z jump too large: |dz|=" + std::to_string(result.z_delta) +
      "m > " + std::to_string(max_z_delta_m) + "m";
    return result;
  }

  in_out_pose.orientation = toGeometryMsgQuaternion(q_world_fk);
  result.success = true;
  return result;
}

}  // namespace cbp::world_model
