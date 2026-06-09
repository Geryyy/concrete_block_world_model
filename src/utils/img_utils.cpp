#include "concrete_block_world_model/utils/img_utils.hpp"
#include <cv_bridge/cv_bridge.h>

cv::Mat extract_mask_roi(
  const cv::Mat & full_mask,
  const vision_msgs::msg::Detection2D & det)
{
  // Create empty mask with same size/type
  cv::Mat det_mask = cv::Mat::zeros(
    full_mask.rows,
    full_mask.cols,
    full_mask.type()
  );

  const int cx = static_cast<int>(std::round(det.bbox.center.position.x));
  const int cy = static_cast<int>(std::round(det.bbox.center.position.y));
  const int w = static_cast<int>(std::round(det.bbox.size_x));
  const int h = static_cast<int>(std::round(det.bbox.size_y));

  int x0 = std::max(0, cx - w / 2);
  int y0 = std::max(0, cy - h / 2);
  int x1 = std::min(full_mask.cols, cx + w / 2);
  int y1 = std::min(full_mask.rows, cy + h / 2);

  if (x1 <= x0 || y1 <= y0) {
    return det_mask;  // valid, but empty
  }

  // Copy only the ROI into the same-sized mask
  full_mask(cv::Rect(x0, y0, x1 - x0, y1 - y0))
  .copyTo(det_mask(cv::Rect(x0, y0, x1 - x0, y1 - y0)));

  return det_mask;
}

double bboxCenterDistance(
  const vision_msgs::msg::Detection2D & a,
  const vision_msgs::msg::Detection2D & b)
{
  const auto & ca = a.bbox.center.position;
  const auto & cb = b.bbox.center.position;

  const double dx = ca.x - cb.x;
  const double dy = ca.y - cb.y;

  return std::sqrt(dx * dx + dy * dy);
}


double bboxIoU(const cv::Rect & a, const cv::Rect & b)
{
  const int x1 = std::max(a.x, b.x);
  const int y1 = std::max(a.y, b.y);
  const int x2 = std::min(a.x + a.width, b.x + b.width);
  const int y2 = std::min(a.y + a.height, b.y + b.height);

  const int inter_area =
    std::max(0, x2 - x1) * std::max(0, y2 - y1);

  const int union_area =
    a.area() + b.area() - inter_area;

  if (union_area <= 0) {
    return 0.0;
  }

  return static_cast<double>(inter_area) / union_area;
}


double maskIoU(const cv::Mat & a, const cv::Mat & b)
{
  CV_Assert(a.size() == b.size());
  CV_Assert(a.type() == b.type());

  cv::Mat inter, uni;

  cv::bitwise_and(a, b, inter);
  cv::bitwise_or(a, b, uni);

  const double inter_area = static_cast<double>(cv::countNonZero(inter));
  const double union_area = static_cast<double>(cv::countNonZero(uni));

  if (union_area <= 0.0) {
    return 0.0;
  }

  return inter_area / union_area;
}

cv::Rect toCvRect(const vision_msgs::msg::Detection2D & det)
{
  const int cx = static_cast<int>(std::round(det.bbox.center.position.x));
  const int cy = static_cast<int>(std::round(det.bbox.center.position.y));
  const int w = static_cast<int>(std::round(det.bbox.size_x));
  const int h = static_cast<int>(std::round(det.bbox.size_y));

  return cv::Rect(cx - w / 2, cy - h / 2, w, h);
}


cv::Mat toCvBgr(
  const sensor_msgs::msg::Image & image)
{
  return cv_bridge::toCvCopy(image, "bgr8")->image;
}

cv::Mat toCvMono(
  const sensor_msgs::msg::Image & image)
{
  return cv_bridge::toCvCopy(image, "mono8")->image;
}

// ==========================================================
// Mask overlay (vectorized)
// ==========================================================
void overlayMask(
  cv::Mat & image,
  const cv::Mat & mask,
  const cv::Scalar & color,
  double alpha)
{
  if (mask.empty()) {
    return;
  }

  cv::Mat mask_binary;
  cv::Mat mask_8u;
  mask.convertTo(mask_8u, CV_8U, 255.0);
  cv::threshold(mask_8u, mask_binary, 127, 255, cv::THRESH_BINARY);

  cv::Mat colored(image.size(), CV_8UC3, color);

  cv::Mat blended;
  cv::addWeighted(
    image, 1.0 - alpha,
    colored, alpha,
    0.0, blended);

  blended.copyTo(image, mask_binary);
}

// ==========================================================
// Bounding box drawing
// ==========================================================
void drawBoundingBox(
  cv::Mat & image,
  const vision_msgs::msg::BoundingBox2D & bbox,
  const cv::Scalar & color,
  int thickness)
{
  int x = bbox.center.position.x - bbox.size_x / 2;
  int y = bbox.center.position.y - bbox.size_y / 2;

  cv::rectangle(
    image,
    cv::Rect(x, y, bbox.size_x, bbox.size_y),
    color,
    thickness);
}

void drawDetectionBoxes(
  cv::Mat & image,
  const vision_msgs::msg::Detection2DArray & detections,
  const cv::Scalar & color)
{
  for (const auto & d : detections.detections) {
    drawBoundingBox(image, d.bbox, color);
  }
}
