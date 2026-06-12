#include <cmath>
#include "concrete_block_world_model/utils/block_utils.hpp"


using Block = concrete_block_world_model_interfaces::msg::Block;
using BlockArray = concrete_block_world_model_interfaces::msg::BlockArray;

std::string poseStatusToString(int status)
{
  switch (status) {
    case Block::POSE_COARSE:
      return "coarse";
    case Block::POSE_PRECISE:
      return "precise";
    case Block::POSE_UNKNOWN:
    default:
      return "unknown";
  }
}


std::string taskStatusToString(int status)
{
  switch (status) {
    case Block::TASK_FREE:
      return "free";
    case Block::TASK_PLACED:
      return "placed";
    case Block::TASK_MOVE:
      return "move";
    case Block::TASK_REMOVED:
      return "removed";
    case Block::TASK_UNKNOWN:
    default:
      return "unknown";
  }
}


bool isKnownPoseStatus(int status)
{
  return status == Block::POSE_UNKNOWN ||
         status == Block::POSE_COARSE ||
         status == Block::POSE_PRECISE;
}


bool isKnownTaskStatus(int status)
{
  return status == Block::TASK_UNKNOWN ||
         status == Block::TASK_FREE ||
         status == Block::TASK_MOVE ||
         status == Block::TASK_PLACED ||
         status == Block::TASK_REMOVED;
}


double poseDistance(
  const geometry_msgs::msg::Pose & a,
  const geometry_msgs::msg::Pose & b)
{
  const double dx = a.position.x - b.position.x;
  const double dy = a.position.y - b.position.y;
  const double dz = a.position.z - b.position.z;
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}
