#pragma once

#include <string>

#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/header.hpp>

#include "concrete_block_world_model_interfaces/msg/block.hpp"

namespace cbp::world_model
{

struct CoarsePoseConfig
{
  int min_points{120};
  double square_ratio_thresh{1.35};
  double front_center_offset_square_m{0.45};
  double front_center_offset_rect_m{0.30};
  float min_confidence{0.3F};
};

struct CoarsePoseInput
{
  uint32_t detection_id{0};
  std_msgs::msg::Header header;
  cv::Mat mask;
  const Eigen::Vector3d * camera_origin_world{nullptr};
};

bool buildCoarseBlockFromOrganizedCloud(
  const CoarsePoseInput & in,
  const sensor_msgs::msg::PointCloud2 & cloud_msg,
  const CoarsePoseConfig & cfg,
  concrete_block_world_model_interfaces::msg::Block & out_block,
  std::string & reason);

bool buildCoarseBlockFromCutoutCloud(
  const CoarsePoseInput & in,
  const sensor_msgs::msg::PointCloud2 & cutout_cloud_msg,
  const CoarsePoseConfig & cfg,
  concrete_block_world_model_interfaces::msg::Block & out_block,
  std::string & reason);

}  // namespace cbp::world_model

