#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>

#include "concrete_block_world_model/world_model/continuous_perception.hpp"

namespace cbpwm = cbp::world_model;
using concrete_block_world_model_interfaces::msg::Block;

namespace
{

Block makeBlock(double x, double y, double z)
{
  Block block;
  block.pose.position.x = x;
  block.pose.position.y = y;
  block.pose.position.z = z;
  block.confidence = 1.0F;
  return block;
}

cbpwm::ContinuousMaskCandidate makeCandidate(
  size_t detection_index,
  const std::string & class_id,
  const cv::Rect & bbox,
  const Block & coarse_block)
{
  cv::Mat mask = cv::Mat::zeros(120, 160, CV_8UC1);
  cv::rectangle(mask, bbox, cv::Scalar(255), cv::FILLED);

  cbpwm::ContinuousMaskQuality quality;
  quality.mask_pixels = cv::countNonZero(mask);
  quality.bbox_area_px = bbox.area();
  quality.fill_ratio =
    static_cast<double>(quality.mask_pixels) / static_cast<double>(quality.bbox_area_px);
  quality.accepted = true;
  quality.reason = "accepted";

  cbpwm::ContinuousMaskCandidate candidate;
  candidate.detection_index = detection_index;
  candidate.class_id = class_id;
  candidate.bbox = bbox;
  candidate.mask = mask;
  candidate.quality = quality;
  candidate.coarse_block = coarse_block;
  candidate.confidence = 1.0;
  candidate.cutout_points = 100;
  return candidate;
}

std::vector<cbpwm::ContinuousMaskGroup> group(
  const std::vector<cbpwm::ContinuousMaskCandidate> & candidates,
  cbpwm::ContinuousMaskMergeConfig cfg)
{
  rclcpp::Clock clock(RCL_ROS_TIME);
  return cbpwm::groupContinuousCandidates(
    candidates,
    cfg,
    rclcpp::get_logger("test_continuous_perception"),
    clock);
}

}  // namespace

TEST(ContinuousPerception, OcclusionAwareMergeCombinesGripperSplitMasks)
{
  cbpwm::ContinuousMaskMergeConfig cfg;
  cfg.enabled = false;
  cfg.occlusion_aware_enabled = true;
  cfg.max_bbox_gap_px = 16.0;
  cfg.min_bbox_axis_overlap = 0.5;

  const auto groups = group(
    {
      makeCandidate(0, "block", cv::Rect(40, 40, 28, 42), makeBlock(0.0, 0.0, 0.0)),
      makeCandidate(1, "block", cv::Rect(76, 42, 30, 40), makeBlock(2.0, 0.0, 0.0)),
    },
    cfg);

  ASSERT_EQ(groups.size(), 1U);
  EXPECT_EQ(groups.front().candidate_indices.size(), 2U);
  const int expected_pixels =
    cv::countNonZero(groups.front().merged_mask(cv::Rect(40, 40, 28, 42))) +
    cv::countNonZero(groups.front().merged_mask(cv::Rect(76, 42, 30, 40)));
  EXPECT_EQ(cv::countNonZero(groups.front().merged_mask), expected_pixels);
}

TEST(ContinuousPerception, OcclusionAwareMergeCombinesNestedMasks)
{
  cbpwm::ContinuousMaskMergeConfig cfg;
  cfg.enabled = false;
  cfg.occlusion_aware_enabled = true;
  cfg.min_bbox_overlap_ratio = 0.5;

  const auto groups = group(
    {
      makeCandidate(0, "block", cv::Rect(38, 36, 56, 48), makeBlock(0.0, 0.0, 0.0)),
      makeCandidate(1, "block", cv::Rect(52, 46, 24, 20), makeBlock(2.0, 0.0, 0.0)),
    },
    cfg);

  ASSERT_EQ(groups.size(), 1U);
  EXPECT_EQ(groups.front().candidate_indices.size(), 2U);
}

TEST(ContinuousPerception, OcclusionAwareMergeDoesNotCombineDifferentClasses)
{
  cbpwm::ContinuousMaskMergeConfig cfg;
  cfg.enabled = false;
  cfg.occlusion_aware_enabled = true;
  cfg.max_bbox_gap_px = 16.0;
  cfg.min_bbox_axis_overlap = 0.5;

  const auto groups = group(
    {
      makeCandidate(0, "block", cv::Rect(40, 40, 28, 42), makeBlock(0.0, 0.0, 0.0)),
      makeCandidate(1, "other", cv::Rect(76, 42, 30, 40), makeBlock(2.0, 0.0, 0.0)),
    },
    cfg);

  EXPECT_EQ(groups.size(), 2U);
}
