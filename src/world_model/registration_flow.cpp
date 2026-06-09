#include "concrete_block_world_model/world_model/registration_flow.hpp"

#include <limits>

#include <cv_bridge/cv_bridge.h>

#include "concrete_block_world_model/utils/img_utils.hpp"
#include "concrete_block_world_model/world_model/state_manager.hpp"

namespace cbp::world_model
{

std::vector<DetectionCandidate> buildRegistrationCandidates(
  const ros2_yolos_cpp::srv::SegmentImage::Response & seg_res,
  OneShotMode run_mode,
  const std::string & target_block_id)
{
  std::vector<DetectionCandidate> candidates;
  candidates.reserve(seg_res.detections.detections.size());

  cv::Mat full_mask = toCvMono(seg_res.mask);
  const auto & detections = seg_res.detections.detections;
  for (size_t i = 0; i < detections.size(); ++i) {
    const auto & det = detections[i];
    const uint32_t detection_id = static_cast<uint32_t>(i + 1U);
    const std::string det_id = "block_" + std::to_string(detection_id);

    if ((run_mode == OneShotMode::kRefineBlock || run_mode == OneShotMode::kRefineGrasped) &&
      !target_block_id.empty() &&
      target_block_id.rfind("block_", 0) == 0 &&
      det_id != target_block_id)
    {
      continue;
    }

    cv::Mat det_mask = extract_mask_roi(full_mask, det);
    if (det_mask.empty() || cv::countNonZero(det_mask) == 0) {
      continue;
    }

    auto mask_msg = cv_bridge::CvImage(
      seg_res.mask.header,
      "mono8",
      det_mask).toImageMsg();
    candidates.emplace_back(detection_id, *mask_msg);
  }

  return candidates;
}

bool selectBestCandidateByExpectedPose(
  const std::vector<DetectionCandidate> & candidates,
  const concrete_block_world_model_interfaces::msg::Block & expected_target,
  double max_distance_m,
  const std::function<bool(
    uint32_t, const sensor_msgs::msg::Image &, concrete_block_world_model_interfaces::msg::Block &,
    std::string &)> & run_registration,
  concrete_block_world_model_interfaces::msg::Block & out_best_block,
  double & out_best_dist)
{
  bool best_valid = false;
  out_best_dist = std::numeric_limits<double>::infinity();

  for (const auto & candidate : candidates) {
    concrete_block_world_model_interfaces::msg::Block block;
    std::string reason;
    if (!run_registration(candidate.first, candidate.second, block, reason)) {
      continue;
    }

    const double dist = blockDistance(block, expected_target);
    if (dist < out_best_dist) {
      out_best_dist = dist;
      out_best_block = block;
      best_valid = true;
    }
  }

  return best_valid && out_best_dist <= max_distance_m;
}

}  // namespace cbp::world_model

