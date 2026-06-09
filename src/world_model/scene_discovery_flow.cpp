#include "concrete_block_world_model/world_model/scene_discovery_flow.hpp"

#include <limits>

namespace cbp::world_model
{

void processRegistrationCandidates(
  const std::vector<DetectionCandidate> & candidates,
  const sensor_msgs::msg::PointCloud2 & cloud,
  const SceneFlowRequest & request,
  const SceneFlowRuntime & rt,
  RegistrationCounters & counters)
{
  const bool targeted_refine =
    (request.mode == OneShotMode::kRefineBlock || request.mode == OneShotMode::kRefineGrasped) &&
    !request.target_block_id.empty() &&
    request.target_block_id.rfind("block_", 0) != 0;

  concrete_block_world_model_interfaces::msg::Block expected_target;
  const bool have_expected_target =
    targeted_refine && rt.get_expected_target &&
    rt.get_expected_target(request.target_block_id, expected_target);

  if (have_expected_target) {
    concrete_block_world_model_interfaces::msg::Block best_block;
    double best_dist = std::numeric_limits<double>::infinity();
    const bool best_found = selectBestCandidateByExpectedPose(
      candidates,
      expected_target,
      request.refine_target_max_distance_m,
      [&rt, &cloud, &request](
        uint32_t detection_id,
        const sensor_msgs::msg::Image & mask,
        concrete_block_world_model_interfaces::msg::Block & out_block,
        std::string & out_reason) {
        return rt.run_registration(
          detection_id,
          mask,
          cloud,
          cloud.header,
          request.registration_timeout_s,
          out_block,
          out_reason);
      },
      best_block,
      best_dist);

    if (!best_found) {
      RCLCPP_WARN(
        rt.logger,
        "Targeted refine failed for '%s': no candidate within %.3f m of expected pose.",
        request.target_block_id.c_str(),
        request.refine_target_max_distance_m);
      return;
    }

    std::string assigned_id;
    std::string upsert_reason;
    const bool upsert_ok = rt.upsert_block(best_block, assigned_id, upsert_reason);
    if (!upsert_ok) {
      RCLCPP_WARN(
        rt.logger,
        "Targeted refine rejected after association checks: %s",
        upsert_reason.c_str());
      return;
    }

    ++counters.registrations_ok;
    RCLCPP_INFO(
      rt.logger,
      "Targeted refine accepted: target=%s assigned=%s dist=%.3f m",
      request.target_block_id.c_str(),
      assigned_id.c_str(),
      best_dist);
    return;
  }

  for (const auto & candidate : candidates) {
    concrete_block_world_model_interfaces::msg::Block block;
    std::string reason;
    const bool ok = rt.run_registration(
      candidate.first,
      candidate.second,
      cloud,
      cloud.header,
      request.registration_timeout_s,
      block,
      reason);

    if (ok) {
      std::string assigned_id;
      std::string upsert_reason;
      const bool upsert_ok = rt.upsert_block(block, assigned_id, upsert_reason);
      if (upsert_ok) {
        ++counters.registrations_ok;
        RCLCPP_INFO(
          rt.logger,
          "Registration accepted and associated: incoming=%s assigned=%s",
          block.id.c_str(),
          assigned_id.c_str());
      } else {
        RCLCPP_WARN(
          rt.logger,
          "Registration rejected after association checks for %s: %s",
          block.id.c_str(),
          upsert_reason.c_str());
      }
      continue;
    }

    const bool fallback_ok =
      rt.try_coarse_fallback &&
      rt.try_coarse_fallback(
      candidate.first,
      reason,
      candidate.second,
      cloud,
      request,
      counters);
    if (!fallback_ok) {
      RCLCPP_WARN(
        rt.logger,
        "Registration rejected for block_%u: %s",
        candidate.first,
        reason.c_str());
    }
  }
}

}  // namespace cbp::world_model
