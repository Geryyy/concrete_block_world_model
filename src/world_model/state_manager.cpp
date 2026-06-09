#include "concrete_block_world_model/world_model/state_manager.hpp"

#include <cmath>

namespace cbp::world_model
{

namespace
{

std::string nextWorldBlockId(uint64_t & world_block_counter)
{
  world_block_counter++;
  return "wm_block_" + std::to_string(world_block_counter);
}

}  // namespace

double blockDistance(
  const concrete_block_world_model_interfaces::msg::Block & a,
  const concrete_block_world_model_interfaces::msg::Block & b)
{
  const double dx = a.pose.position.x - b.pose.position.x;
  const double dy = a.pose.position.y - b.pose.position.y;
  const double dz = a.pose.position.z - b.pose.position.z;
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

std::string resolveGraspedBlockId(
  const std::unordered_map<std::string, concrete_block_world_model_interfaces::msg::Block> & persistent_world,
  const rclcpp::Clock & clock)
{
  rclcpp::Time newest_time(0, 0, clock.get_clock_type());
  std::string best_id;
  for (const auto & kv : persistent_world) {
    if (kv.second.task_status != concrete_block_world_model_interfaces::msg::Block::TASK_MOVE) {
      continue;
    }
    const rclcpp::Time seen(kv.second.last_seen, clock.get_clock_type());
    if (best_id.empty() || seen > newest_time) {
      newest_time = seen;
      best_id = kv.first;
    }
  }
  return best_id;
}

bool upsertRegisteredBlock(
  std::unordered_map<std::string, concrete_block_world_model_interfaces::msg::Block> & persistent_world,
  uint64_t & world_block_counter,
  concrete_block_world_model_interfaces::msg::Block incoming,
  OneShotMode run_mode,
  const std::string & target_block_id,
  const std_msgs::msg::Header & header,
  const rclcpp::Clock & clock,
  const AssociationConfig & config,
  std::string & assigned_id,
  std::string & reason)
{
  const rclcpp::Time now_stamp(header.stamp, clock.get_clock_type());
  if (incoming.confidence < config.min_update_confidence) {
    reason = "confidence below min_update_confidence";
    return false;
  }

  std::string forced_id;
  if ((run_mode == OneShotMode::kRefineBlock || run_mode == OneShotMode::kRefineGrasped) &&
    !target_block_id.empty())
  {
    forced_id = target_block_id;
  }

  const concrete_block_world_model_interfaces::msg::Block * best_match = nullptr;
  double best_dist = std::numeric_limits<double>::infinity();
  std::string best_id;

  for (const auto & kv : persistent_world) {
    const auto & existing = kv.second;
    const rclcpp::Time seen(existing.last_seen, clock.get_clock_type());
    const double age_s = (now_stamp - seen).seconds();
    if (age_s > config.association_max_age_s) {
      continue;
    }

    const double dist = blockDistance(incoming, existing);
    if (!shouldAssociateByDistance(
        dist, config.association_max_distance_m, incoming.confidence,
        config.min_update_confidence))
    {
      continue;
    }
    if (dist < best_dist) {
      best_dist = dist;
      best_match = &existing;
      best_id = kv.first;
    }
  }

  if (!forced_id.empty()) {
    assigned_id = forced_id;
  } else if (best_match != nullptr) {
    assigned_id = best_id;
  } else {
    assigned_id = nextWorldBlockId(world_block_counter);
  }

  auto it = persistent_world.find(assigned_id);
  if (it != persistent_world.end()) {
    const auto & previous = it->second;
    const rclcpp::Time prev_stamp(previous.last_seen, clock.get_clock_type());
    const rclcpp::Time incoming_stamp(incoming.last_seen, clock.get_clock_type());
    if (incoming_stamp < prev_stamp) {
      reason = "stale update (incoming older than stored state)";
      return false;
    }

    if (!isValidTaskTransition(previous.task_status, incoming.task_status)) {
      reason = std::string("invalid task transition ") +
        taskStatusToString(previous.task_status) + " -> " +
        taskStatusToString(incoming.task_status);
      return false;
    }

    if (previous.task_status != concrete_block_world_model_interfaces::msg::Block::TASK_UNKNOWN) {
      incoming.task_status = previous.task_status;
    }
  } else {
    incoming.task_status = concrete_block_world_model_interfaces::msg::Block::TASK_FREE;
  }

  incoming.id = assigned_id;
  if (incoming.pose_status == concrete_block_world_model_interfaces::msg::Block::POSE_UNKNOWN) {
    incoming.pose_status = concrete_block_world_model_interfaces::msg::Block::POSE_PRECISE;
  }
  persistent_world[assigned_id] = incoming;
  return true;
}

}  // namespace cbp::world_model

