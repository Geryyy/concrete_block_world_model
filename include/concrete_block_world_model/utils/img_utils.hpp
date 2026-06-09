#pragma once

#include <opencv2/imgproc.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <vision_msgs/msg/detection2_d.hpp>
#include <vision_msgs/msg/pose2_d.hpp>

#include <sensor_msgs/msg/image.hpp>

#include <opencv2/opencv.hpp>


cv::Mat extract_mask_roi(
  const cv::Mat & full_mask,
  const vision_msgs::msg::Detection2D & det);
double bboxCenterDistance(
  const vision_msgs::msg::Detection2D & a,
  const vision_msgs::msg::Detection2D & b);
double bboxIoU(const cv::Rect & a, const cv::Rect & b);
double maskIoU(const cv::Mat & a, const cv::Mat & b);
cv::Rect toCvRect(const vision_msgs::msg::Detection2D & det);


cv::Mat toCvBgr(
  const sensor_msgs::msg::Image & image);

cv::Mat toCvMono(
  const sensor_msgs::msg::Image & image);

void overlayMask(
  cv::Mat & image,
  const cv::Mat & mask,
  const cv::Scalar & color,
  double alpha = 0.3);

void drawBoundingBox(
  cv::Mat & image,
  const vision_msgs::msg::BoundingBox2D & bbox,
  const cv::Scalar & color,
  int thickness = 2);

void drawDetectionBoxes(
  cv::Mat & image,
  const vision_msgs::msg::Detection2DArray & detections,
  const cv::Scalar & color);
