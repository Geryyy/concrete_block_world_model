#include "concrete_block_world_model/world_model/scene_discovery_flow.hpp"

#include <limits>

#include "concrete_block_world_model/utils/block_utils.hpp"

namespace cbp::world_model
{

void processRegistrationCandidates(
  const std::vector<DetectionCandidate> & candidates,
  const sensor_msgs::msg::PointCloud2 & cloud,
  const SceneFlowRequest & request,
  const SceneFlowRuntime & rt,
  RegistrationCounters & counters)
{
  // Any refine with a named target (whether the id is auto-assigned like "block_0"
  // or user-named) is matched to a detection by pose, using the target block's
  // known world pose. Detection ids are per-frame indices and cannot be matched to
  // persistent world-model ids directly.
  const bool targeted_refine =
    (request.mode == OneShotMode::kRefineBlock || request.mode == OneShotMode::kRefineGrasped) &&
    !request.target_block_id.empty();

  if (targeted_refine) {
    concrete_block_world_model_interfaces::msg::Block expected_target;
    if (!rt.get_expected_target ||
      !rt.get_expected_target(request.target_block_id, expected_target))
    {
      RCLCPP_WARN(
        rt.logger,
        "Targeted refine skipped: target '%s' is not present in the world model.",
        request.target_block_id.c_str());
      return;
    }

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
      if (rt.on_registration_result) {
        concrete_block_world_model_interfaces::msg::Block empty_block;
        rt.on_registration_result(
          0U,
          false,
          empty_block,
          false,
          "",
          "targeted refine: no candidate within expected-pose gate");
      }
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
      if (rt.on_registration_result) {
        rt.on_registration_result(
          0U,
          true,
          best_block,
          false,
          "",
          upsert_reason);
      }
      RCLCPP_WARN(
        rt.logger,
        "Targeted refine rejected after association checks: %s",
        upsert_reason.c_str());
      return;
    }

    ++counters.registrations_ok;
    if (rt.on_registration_result) {
      rt.on_registration_result(
        0U,
        true,
        best_block,
        true,
        assigned_id,
        upsert_reason);
    }
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
    RCLCPP_INFO(
      rt.logger,
      "Registration candidate start: detection_id=%u timeout=%.2fs",
      candidate.first,
      request.registration_timeout_s);
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
        if (rt.on_registration_result) {
          rt.on_registration_result(
            candidate.first,
            true,
            block,
            true,
            assigned_id,
            upsert_reason);
        }
        RCLCPP_INFO(
          rt.logger,
          "Registration accepted and associated: detection_id=%u incoming=%s assigned=%s confidence=%.3f pose_status=%d pos=[%.3f, %.3f, %.3f]",
          candidate.first,
          block.id.c_str(),
          assigned_id.c_str(),
          block.confidence,
          block.pose_status,
          block.pose.position.x,
          block.pose.position.y,
          block.pose.position.z);
      } else {
        if (rt.on_registration_result) {
          rt.on_registration_result(
            candidate.first,
            true,
            block,
            false,
            "",
            upsert_reason);
        }
        RCLCPP_WARN(
          rt.logger,
          "Registration rejected after association checks: detection_id=%u incoming=%s confidence=%.3f reason=%s",
          candidate.first,
          block.id.c_str(),
          block.confidence,
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
      if (rt.on_registration_result) {
        concrete_block_world_model_interfaces::msg::Block empty_block;
        rt.on_registration_result(
          candidate.first,
          false,
          empty_block,
          false,
          "",
          reason);
      }
      RCLCPP_WARN(
        rt.logger,
        "Registration rejected: detection_id=%u reason=%s",
        candidate.first,
        reason.c_str());
    }
  }
}

}  // namespace cbp::world_model
