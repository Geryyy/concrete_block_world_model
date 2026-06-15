#include "concrete_block_world_model/world_model/continuous_perception.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

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

bool sameClassOrUnknown(
  const ContinuousMaskCandidate & lhs,
  const ContinuousMaskCandidate & rhs)
{
  return lhs.class_id.empty() || rhs.class_id.empty() || lhs.class_id == rhs.class_id;
}

double overlapRatioOfSmaller(const cv::Rect & lhs, const cv::Rect & rhs)
{
  const cv::Rect intersection = lhs & rhs;
  const int smaller_area = std::min(lhs.area(), rhs.area());
  if (smaller_area <= 0) {
    return 0.0;
  }
  return static_cast<double>(intersection.area()) / static_cast<double>(smaller_area);
}

double axisOverlapRatio(
  int a_min,
  int a_max,
  int b_min,
  int b_max)
{
  const int overlap = std::max(0, std::min(a_max, b_max) - std::max(a_min, b_min));
  const int smaller = std::min(a_max - a_min, b_max - b_min);
  if (smaller <= 0) {
    return 0.0;
  }
  return static_cast<double>(overlap) / static_cast<double>(smaller);
}

double rectGapPx(const cv::Rect & lhs, const cv::Rect & rhs)
{
  const int lhs_right = lhs.x + lhs.width;
  const int rhs_right = rhs.x + rhs.width;
  const int horizontal_gap = std::max({lhs.x - rhs_right, rhs.x - lhs_right, 0});
  const int lhs_bottom = lhs.y + lhs.height;
  const int rhs_bottom = rhs.y + rhs.height;
  const int vertical_gap = std::max({lhs.y - rhs_bottom, rhs.y - lhs_bottom, 0});
  if (horizontal_gap == 0) {
    return static_cast<double>(vertical_gap);
  }
  if (vertical_gap == 0) {
    return static_cast<double>(horizontal_gap);
  }
  return std::hypot(static_cast<double>(horizontal_gap), static_cast<double>(vertical_gap));
}

bool unionShapePlausible(
  const ContinuousMaskCandidate & candidate,
  const ContinuousMaskGroup & group,
  const ContinuousMaskMergeConfig & cfg)
{
  const cv::Rect union_rect = candidate.bbox | cv::boundingRect(group.merged_mask);
  if (union_rect.area() <= 0) {
    return false;
  }

  const double longer = static_cast<double>(std::max(union_rect.width, union_rect.height));
  const double shorter = static_cast<double>(std::max(1, std::min(union_rect.width, union_rect.height)));
  if ((longer / shorter) > cfg.max_union_aspect_ratio) {
    return false;
  }

  cv::Mat merged;
  cv::bitwise_or(group.merged_mask, candidate.mask, merged);
  const cv::Mat roi = merged(union_rect);
  const double fill_ratio =
    static_cast<double>(cv::countNonZero(roi)) / static_cast<double>(union_rect.area());
  return fill_ratio >= cfg.min_union_fill_ratio;
}

bool bboxOcclusionCompatible(
  const ContinuousMaskCandidate & candidate,
  const ContinuousMaskCandidate & grouped,
  const ContinuousMaskGroup & group,
  const ContinuousMaskMergeConfig & cfg)
{
  if (!cfg.occlusion_aware_enabled || candidate.bbox.area() <= 0 || grouped.bbox.area() <= 0) {
    return false;
  }
  if (!sameClassOrUnknown(candidate, grouped)) {
    return false;
  }

  const double overlap_ratio = overlapRatioOfSmaller(candidate.bbox, grouped.bbox);
  if (overlap_ratio >= cfg.min_bbox_overlap_ratio) {
    return unionShapePlausible(candidate, group, cfg);
  }

  const double x_overlap = axisOverlapRatio(
    candidate.bbox.x,
    candidate.bbox.x + candidate.bbox.width,
    grouped.bbox.x,
    grouped.bbox.x + grouped.bbox.width);
  const double y_overlap = axisOverlapRatio(
    candidate.bbox.y,
    candidate.bbox.y + candidate.bbox.height,
    grouped.bbox.y,
    grouped.bbox.y + grouped.bbox.height);
  const bool side_by_side =
    rectGapPx(candidate.bbox, grouped.bbox) <= cfg.max_bbox_gap_px &&
    std::max(x_overlap, y_overlap) >= cfg.min_bbox_axis_overlap;

  return side_by_side && unionShapePlausible(candidate, group, cfg);
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
  const ContinuousMaskMergeConfig & cfg,
  double & nearest_distance_m)
{
  bool fits = false;
  nearest_distance_m = std::numeric_limits<double>::infinity();
  for (const size_t grouped_index : group.candidate_indices) {
    const auto & grouped = candidates.at(grouped_index);
    if (!sameClassOrUnknown(candidate, grouped)) {
      continue;
    }

    const double dist = blockDistance(candidate.coarse_block, grouped.coarse_block);
    if (dist < nearest_distance_m) {
      nearest_distance_m = dist;
    }
    if (
      (cfg.enabled && dist <= cfg.max_centroid_distance_m) ||
      bboxOcclusionCompatible(candidate, grouped, group, cfg))
    {
      fits = true;
    }
  }
  return fits;
}

std::vector<ContinuousMaskGroup> groupContinuousCandidates(
  const std::vector<ContinuousMaskCandidate> & candidates,
  const ContinuousMaskMergeConfig & cfg,
  const rclcpp::Logger & logger,
  rclcpp::Clock & clock)
{
  std::vector<ContinuousMaskGroup> groups;
  for (size_t candidate_index = 0; candidate_index < candidates.size(); ++candidate_index) {
    const auto & candidate = candidates[candidate_index];
    size_t best_group_index = groups.size();
    double best_group_distance_m = std::numeric_limits<double>::infinity();
    if (
      (cfg.enabled && cfg.max_centroid_distance_m > 0.0) ||
      cfg.occlusion_aware_enabled)
    {
      for (size_t group_index = 0; group_index < groups.size(); ++group_index) {
        double nearest_distance_m = std::numeric_limits<double>::infinity();
        if (candidateFitsGroup(
            candidate,
            groups[group_index],
            candidates,
            cfg,
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
      cfg.max_centroid_distance_m);
  }

  RCLCPP_INFO_THROTTLE(
    logger, clock, 1000,
    "Continuous mask merge summary: candidates=%zu groups=%zu enabled=%s threshold=%.3fm occlusion_aware=%s",
    candidates.size(),
    groups.size(),
    cfg.enabled ? "true" : "false",
    cfg.max_centroid_distance_m,
    cfg.occlusion_aware_enabled ? "true" : "false");

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
