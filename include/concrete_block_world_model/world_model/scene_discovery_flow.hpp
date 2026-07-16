#pragma once

#include <functional>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include "concrete_block_world_model_interfaces/msg/block.hpp"
#include "concrete_block_world_model/world_model/registration_flow.hpp"
#include "concrete_block_world_model/utils/world_model_utils.hpp"

namespace cbp::world_model
{

struct RegistrationCounters
{
  size_t registrations_ok{0};
  size_t coarse_upserts_ok{0};
};

struct SceneFlowRequest
{
  OneShotMode mode{OneShotMode::kNone};
  std::string target_block_id;
  double registration_timeout_s{3.0};
  double refine_target_max_distance_m{1.2};
};

struct SceneFlowRuntime
{
  rclcpp::Logger logger{rclcpp::get_logger("scene_discovery_flow")};

  std::function<bool(
      uint32_t,
      const sensor_msgs::msg::Image &,
      const sensor_msgs::msg::PointCloud2 &,
      const std_msgs::msg::Header &,
      double,
      concrete_block_world_model_interfaces::msg::Block &,
      std::string &)> run_registration;

  std::function<bool(
      concrete_block_world_model_interfaces::msg::Block &,
      std::string &,
      std::string &)> upsert_block;

  std::function<bool(const std::string &, concrete_block_world_model_interfaces::msg::Block &)> get_expected_target;

  std::function<bool(
      uint32_t,
      const std::string &,
      const sensor_msgs::msg::Image &,
      const sensor_msgs::msg::PointCloud2 &,
      const SceneFlowRequest &,
      RegistrationCounters &)> try_coarse_fallback;

  std::function<void(
      uint32_t,
      bool,
      const concrete_block_world_model_interfaces::msg::Block &,
      bool,
      const std::string &,
      const std::string &)> on_registration_result;
};

void processRegistrationCandidates(
  const std::vector<DetectionCandidate> & candidates,
  const sensor_msgs::msg::PointCloud2 & cloud,
  const SceneFlowRequest & request,
  const SceneFlowRuntime & rt,
  RegistrationCounters & counters);

}  // namespace cbp::world_model
