#pragma once

#include <functional>
#include <string>

#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "ros2_yolos_cpp/srv/segment_image.hpp"

namespace cbp::world_model
{

struct RoiInputConfig
{
  double roi_size_x_m{0.60};
  double roi_size_y_m{0.40};
  double min_depth_m{0.5};
  double max_depth_m{30.0};
  double segmentation_timeout_s{3.0};
  bool use_black_bg{false};
  int blur_kernel_size{31};
};

struct ProjectionIntrinsics
{
  bool valid{false};
  double fx{0.0};
  double fy{0.0};
  double cx{0.0};
  double cy{0.0};
  uint32_t width{0};
  uint32_t height{0};
};

using RunSegmentationSyncFn = std::function<bool(
    const sensor_msgs::msg::Image &,
    double,
    ros2_yolos_cpp::srv::SegmentImage::Response::SharedPtr &,
    std::string &)>;

sensor_msgs::msg::Image::SharedPtr buildRoiSegmentationInputImage(
  const sensor_msgs::msg::Image::ConstSharedPtr & image,
  const cv::Rect & roi_rect,
  const RoiInputConfig & roi_cfg);

bool buildRoiMaskFromPrediction(
  const ProjectionIntrinsics & intr,
  const sensor_msgs::msg::Image::ConstSharedPtr & image,
  const Eigen::Vector3d & p_camera,
  const RoiInputConfig & roi_cfg,
  cv::Mat & roi_mask,
  cv::Rect & roi_rect,
  std::string & reason);

bool roiSegmentationToFullMask(
  const sensor_msgs::msg::Image::ConstSharedPtr & image,
  const cv::Mat & roi_mask,
  const sensor_msgs::msg::Image::SharedPtr & roi_image_msg,
  const RoiInputConfig & roi_cfg,
  const RunSegmentationSyncFn & run_segmentation_sync,
  cv::Mat & full_seg_mask,
  size_t & detections_count,
  std::string & reason);

sensor_msgs::msg::Image::SharedPtr buildRoiSegmentationDebugOverlay(
  const sensor_msgs::msg::Image::ConstSharedPtr & image,
  const cv::Rect & roi_rect,
  const cv::Mat & full_seg_mask);

}  // namespace cbp::world_model
