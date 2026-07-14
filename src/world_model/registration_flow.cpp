#include "concrete_block_world_model/world_model/registration_flow.hpp"

#include <limits>

#include <cv_bridge/cv_bridge.h>

#include "concrete_block_world_model/utils/block_utils.hpp"
#include "concrete_block_world_model/utils/img_utils.hpp"
#include "concrete_block_world_model/utils/world_model_utils.hpp"
#include "concrete_block_world_model/world_model/state_manager.hpp"

namespace cbp::world_model
{

std::vector<DetectionCandidate> buildRegistrationCandidates(
  const ros2_yolos_cpp::srv::SegmentImage::Response & seg_res,
  OneShotMode run_mode,
  const std::string & target_block_id,
  const rclcpp::Logger & logger)
{
  // Targeted refine is resolved by pose in processRegistrationCandidates(), not by
  // matching a per-frame detection index to a persistent world-model id here, so
  // these are unused when building the candidate list.
  (void)run_mode;
  (void)target_block_id;

  std::vector<DetectionCandidate> candidates;
  candidates.reserve(seg_res.detections.detections.size());

  cv::Mat full_mask = toCvMono(seg_res.mask);
  const auto & detections = seg_res.detections.detections;
  for (size_t i = 0; i < detections.size(); ++i) {
    const auto & det = detections[i];
    const uint32_t detection_id = static_cast<uint32_t>(i + 1U);
    const double confidence = detectionConfidence(det);
    const std::string class_id =
      det.results.empty() ? std::string{} : det.results.front().hypothesis.class_id;

    cv::Mat det_mask = extract_mask_roi(full_mask, det);
    const int mask_pixels = det_mask.empty() ? 0 : cv::countNonZero(det_mask);
    if (mask_pixels == 0) {
      RCLCPP_WARN(
        logger,
        "One-shot candidate rejected: idx=%zu detection_id=%u class=%s confidence=%.3f bbox[cx=%.1f cy=%.1f w=%.1f h=%.1f] reason=empty_mask",
        i,
        detection_id,
        class_id.empty() ? "<unknown>" : class_id.c_str(),
        confidence,
        det.bbox.center.position.x,
        det.bbox.center.position.y,
        det.bbox.size_x,
        det.bbox.size_y);
      continue;
    }

    auto mask_msg = cv_bridge::CvImage(
      seg_res.mask.header,
      "mono8",
      det_mask).toImageMsg();
    candidates.emplace_back(detection_id, *mask_msg);
    RCLCPP_INFO(
      logger,
      "One-shot registration candidate accepted: idx=%zu detection_id=%u class=%s confidence=%.3f mask_pixels=%d bbox[cx=%.1f cy=%.1f w=%.1f h=%.1f]",
      i,
      detection_id,
      class_id.empty() ? "<unknown>" : class_id.c_str(),
      confidence,
      mask_pixels,
      det.bbox.center.position.x,
      det.bbox.center.position.y,
      det.bbox.size_x,
      det.bbox.size_y);
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
