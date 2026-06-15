#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <rclcpp/rclcpp.hpp>
#include <vision_msgs/msg/detection2_d.hpp>

#include "concrete_block_world_model_interfaces/msg/block.hpp"

namespace cbp::world_model
{

struct ContinuousMaskQuality
{
  int mask_pixels{0};
  int bbox_area_px{0};
  double fill_ratio{0.0};
  bool accepted{false};
  std::string reason;
};

struct ContinuousMaskCandidate
{
  size_t detection_index{0};
  std::string class_id;
  cv::Rect bbox;
  cv::Mat mask;
  ContinuousMaskQuality quality;
  concrete_block_world_model_interfaces::msg::Block coarse_block;
  double confidence{1.0};
  uint32_t cutout_points{0};
};

struct ContinuousMaskGroup
{
  std::vector<size_t> candidate_indices;
  cv::Mat merged_mask;
};

struct ContinuousMaskMergeConfig
{
  bool enabled{true};
  double max_centroid_distance_m{0.6};
  bool occlusion_aware_enabled{true};
  double max_bbox_gap_px{24.0};
  double min_bbox_axis_overlap{0.35};
  double min_bbox_overlap_ratio{0.20};
  double max_union_aspect_ratio{2.8};
  double min_union_fill_ratio{0.08};
};

// One pose measurement extracted from a continuous frame, decoupled from the
// world-model update. The quality metadata is kept alongside the pose so a
// probabilistic filter can derive a measurement noise model from it.
struct BlockObservation
{
  concrete_block_world_model_interfaces::msg::Block block;
  int mask_pixels{0};
  uint32_t cutout_points{0};
  size_t fragment_count{1};
  bool precise{false};
  bool has_registration_prior{false};
};

double detectionConfidence(const vision_msgs::msg::Detection2D & det);

ContinuousMaskQuality evaluateContinuousMaskQuality(
  const cv::Mat & binary_mask,
  const vision_msgs::msg::Detection2D & det,
  int min_mask_pixels,
  double min_fill_ratio);

void logContinuousQuality(
  const rclcpp::Logger & logger,
  rclcpp::Clock & clock,
  size_t index,
  const vision_msgs::msg::Detection2D & det,
  const ContinuousMaskQuality & quality);

bool candidateFitsGroup(
  const ContinuousMaskCandidate & candidate,
  const ContinuousMaskGroup & group,
  const std::vector<ContinuousMaskCandidate> & candidates,
  const ContinuousMaskMergeConfig & cfg,
  double & nearest_distance_m);

std::vector<ContinuousMaskGroup> groupContinuousCandidates(
  const std::vector<ContinuousMaskCandidate> & candidates,
  const ContinuousMaskMergeConfig & cfg,
  const rclcpp::Logger & logger,
  rclcpp::Clock & clock);

// Comma-separated detection indices of a group's fragments, for log messages.
std::string fragmentIndices(
  const ContinuousMaskGroup & group,
  const std::vector<ContinuousMaskCandidate> & candidates);

}  // namespace cbp::world_model
