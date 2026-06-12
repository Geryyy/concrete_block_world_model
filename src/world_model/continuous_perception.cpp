#include "concrete_block_world_model/world_model/continuous_perception.hpp"

#include <algorithm>
#include <limits>

#include "concrete_block_world_model/utils/img_utils.hpp"
#include "concrete_block_world_model/world_model/state_manager.hpp"

namespace cbp::world_model
{

namespace
{

cv::Rect clippedDetectionRect(
  const vision_msgs::msg::Detection2D & det,
  const cv::Size & image_size)
{
  const cv::Rect image_rect(0, 0, image_size.width, image_size.height);
  const cv::Rect raw = toCvRect(det);
  return raw & image_rect;
}

}  // namespace

double detectionConfidence(const vision_msgs::msg::Detection2D & det)
{
  if (det.results.empty()) {
    return 1.0;
  }
  return det.results.front().hypothesis.score;
}

ContinuousMaskQuality evaluateContinuousMaskQuality(
  const cv::Mat & binary_mask,
  const vision_msgs::msg::Detection2D & det,
  int min_mask_pixels,
  double min_fill_ratio)
{
  ContinuousMaskQuality quality;
  if (binary_mask.empty()) {
    quality.reason = "empty mask";
    return quality;
  }

  const cv::Rect bbox = clippedDetectionRect(det, binary_mask.size());
  quality.bbox_area_px = bbox.area();
  if (quality.bbox_area_px <= 0) {
    quality.reason = "empty bbox";
    return quality;
  }

  quality.mask_pixels = cv::countNonZero(binary_mask);
  quality.fill_ratio =
    static_cast<double>(quality.mask_pixels) / static_cast<double>(quality.bbox_area_px);

  if (quality.mask_pixels < min_mask_pixels) {
    quality.reason = "mask_pixels below threshold";
    return quality;
  }
  if (quality.fill_ratio < min_fill_ratio) {
    quality.reason = "fill_ratio below threshold";
    return quality;
  }

  quality.accepted = true;
  quality.reason = "accepted";
  return quality;
}

void logContinuousQuality(
  const rclcpp::Logger & logger,
  rclcpp::Clock & clock,
  size_t index,
  const vision_msgs::msg::Detection2D & det,
  const ContinuousMaskQuality & quality)
{
  RCLCPP_INFO_THROTTLE(
    logger, clock, 1000,
    "Continuous quality idx=%zu bbox[cx=%.1f cy=%.1f w=%.1f h=%.1f] score=%.3f "
    "mask_pixels=%d bbox_area=%d fill=%.3f accepted=%s reason=%s",
    index,
    det.bbox.center.position.x,
    det.bbox.center.position.y,
    det.bbox.size_x,
    det.bbox.size_y,
    detectionConfidence(det),
    quality.mask_pixels,
    quality.bbox_area_px,
    quality.fill_ratio,
    quality.accepted ? "true" : "false",
    quality.reason.c_str());
}

bool candidateFitsGroup(
  const ContinuousMaskCandidate & candidate,
  const ContinuousMaskGroup & group,
  const std::vector<ContinuousMaskCandidate> & candidates,
  double max_centroid_distance_m,
  double & nearest_distance_m)
{
  bool fits = false;
  nearest_distance_m = std::numeric_limits<double>::infinity();
  for (const size_t grouped_index : group.candidate_indices) {
    const double dist =
      blockDistance(candidate.coarse_block, candidates.at(grouped_index).coarse_block);
    if (dist < nearest_distance_m) {
      nearest_distance_m = dist;
    }
    if (dist <= max_centroid_distance_m) {
      fits = true;
    }
  }
  return fits;
}

std::vector<ContinuousMaskGroup> groupContinuousCandidates(
  const std::vector<ContinuousMaskCandidate> & candidates,
  bool merge_enabled,
  double max_centroid_distance_m,
  const rclcpp::Logger & logger,
  rclcpp::Clock & clock)
{
  std::vector<ContinuousMaskGroup> groups;
  for (size_t candidate_index = 0; candidate_index < candidates.size(); ++candidate_index) {
    const auto & candidate = candidates[candidate_index];
    size_t best_group_index = groups.size();
    double best_group_distance_m = std::numeric_limits<double>::infinity();
    if (merge_enabled && max_centroid_distance_m > 0.0) {
      for (size_t group_index = 0; group_index < groups.size(); ++group_index) {
        double nearest_distance_m = std::numeric_limits<double>::infinity();
        if (candidateFitsGroup(
            candidate,
            groups[group_index],
            candidates,
            max_centroid_distance_m,
            nearest_distance_m) &&
          nearest_distance_m < best_group_distance_m)
        {
          best_group_index = group_index;
          best_group_distance_m = nearest_distance_m;
        }
      }
    }

    if (best_group_index == groups.size()) {
      ContinuousMaskGroup group;
      group.candidate_indices.push_back(candidate_index);
      group.merged_mask = candidate.mask.clone();
      groups.push_back(std::move(group));
      RCLCPP_INFO_THROTTLE(
        logger, clock, 1000,
        "Continuous mask merge: started group=%zu from idx=%zu pos=[%.3f, %.3f, %.3f]",
        groups.size() - 1U,
        candidate.detection_index,
        candidate.coarse_block.pose.position.x,
        candidate.coarse_block.pose.position.y,
        candidate.coarse_block.pose.position.z);
      continue;
    }

    groups[best_group_index].candidate_indices.push_back(candidate_index);
    cv::bitwise_or(
      groups[best_group_index].merged_mask,
      candidate.mask,
      groups[best_group_index].merged_mask);
    RCLCPP_INFO_THROTTLE(
      logger, clock, 1000,
      "Continuous mask merge: added idx=%zu to group=%zu nearest_dist=%.3fm threshold=%.3fm",
      candidate.detection_index,
      best_group_index,
      best_group_distance_m,
      max_centroid_distance_m);
  }

  RCLCPP_INFO_THROTTLE(
    logger, clock, 1000,
    "Continuous mask merge summary: candidates=%zu groups=%zu enabled=%s threshold=%.3fm",
    candidates.size(),
    groups.size(),
    merge_enabled ? "true" : "false",
    max_centroid_distance_m);

  return groups;
}

std::string fragmentIndices(
  const ContinuousMaskGroup & group,
  const std::vector<ContinuousMaskCandidate> & candidates)
{
  std::string out;
  for (const size_t candidate_index : group.candidate_indices) {
    if (!out.empty()) {
      out += ",";
    }
    out += std::to_string(candidates.at(candidate_index).detection_index);
  }
  return out;
}

}  // namespace cbp::world_model
