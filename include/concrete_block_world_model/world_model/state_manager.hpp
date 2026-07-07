#pragma once

#include <limits>
#include <string>
#include <unordered_map>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/header.hpp>

#include "concrete_block_world_model_interfaces/msg/block.hpp"
#include "concrete_block_world_model/utils/world_model_utils.hpp"

namespace cbp::world_model
{

struct AssociationConfig
{
  double association_max_distance_m{0.45};
  double association_max_age_s{20.0};
  double min_update_confidence{0.25};
};

double blockDistance(
  const concrete_block_world_model_interfaces::msg::Block & a,
  const concrete_block_world_model_interfaces::msg::Block & b);

std::string resolveGraspedBlockId(
  const std::unordered_map<std::string, concrete_block_world_model_interfaces::msg::Block> & persistent_world,
  const rclcpp::Clock & clock);

std::string nextWorldBlockId(
  const std::unordered_map<std::string, concrete_block_world_model_interfaces::msg::Block> & persistent_world,
  uint64_t & world_block_counter);

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
  std::string & reason);

}  // namespace cbp::world_model
