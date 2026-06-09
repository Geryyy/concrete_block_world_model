#include "concrete_block_world_model/world_model/roi_processing.hpp"

#include <algorithm>
#include <cmath>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc.hpp>

#include "concrete_block_world_model/utils/img_utils.hpp"

namespace cbp::world_model
{

sensor_msgs::msg::Image::SharedPtr buildRoiSegmentationInputImage(
  const sensor_msgs::msg::Image::ConstSharedPtr & image,
  const cv::Rect & roi_rect,
  const RoiInputConfig & roi_cfg)
{
  cv::Mat image_bgr = toCvBgr(*image);
  cv::Mat roi_image_full;
  if (roi_cfg.use_black_bg) {
    roi_image_full = cv::Mat::zeros(image_bgr.size(), image_bgr.type());
  } else {
    const int ksize = std::max(1, roi_cfg.blur_kernel_size | 1);
    cv::GaussianBlur(image_bgr, roi_image_full, cv::Size(ksize, ksize), 0.0);
  }
  image_bgr(roi_rect).copyTo(roi_image_full(roi_rect));

  auto out = cv_bridge::CvImage(image->header, "bgr8", roi_image_full).toImageMsg();
  return out;
}

bool buildRoiMaskFromPrediction(
  const ProjectionIntrinsics & intr,
  const sensor_msgs::msg::Image::ConstSharedPtr & image,
  const Eigen::Vector3d & p_camera,
  const RoiInputConfig & roi_cfg,
  cv::Mat & roi_mask,
  cv::Rect & roi_rect,
  std::string & reason)
{
  if (!intr.valid) {
    reason = "camera_info not received or invalid intrinsics";
    return false;
  }

  const double z = p_camera.z();
  if (!std::isfinite(z) || z < roi_cfg.min_depth_m || z > roi_cfg.max_depth_m) {
    reason = "predicted block depth out of bounds: z=" + std::to_string(z);
    return false;
  }

  const double u = (intr.fx * p_camera.x() / z) + intr.cx;
  const double v = (intr.fy * p_camera.y() / z) + intr.cy;
  if (!std::isfinite(u) || !std::isfinite(v)) {
    reason = "projected image point is invalid";
    return false;
  }

  const int roi_w_px = std::max(1, static_cast<int>(std::lround(intr.fx * roi_cfg.roi_size_x_m / z)));
  const int roi_h_px = std::max(1, static_cast<int>(std::lround(intr.fy * roi_cfg.roi_size_y_m / z)));
  cv::Rect requested_roi(
    static_cast<int>(std::lround(u)) - roi_w_px / 2,
    static_cast<int>(std::lround(v)) - roi_h_px / 2,
    roi_w_px,
    roi_h_px);
  const cv::Rect image_rect(0, 0, static_cast<int>(image->width), static_cast<int>(image->height));
  roi_rect = requested_roi & image_rect;
  if (roi_rect.width < 2 || roi_rect.height < 2) {
    reason = "ROI outside image or too small after clamping";
    return false;
  }

  roi_mask = cv::Mat::zeros(image_rect.height, image_rect.width, CV_8UC1);
  cv::rectangle(roi_mask, roi_rect, cv::Scalar(255), cv::FILLED);
  return true;
}

bool roiSegmentationToFullMask(
  const sensor_msgs::msg::Image::ConstSharedPtr & image,
  const cv::Mat & roi_mask,
  const sensor_msgs::msg::Image::SharedPtr & roi_image_msg,
  const RoiInputConfig & roi_cfg,
  const RunSegmentationSyncFn & run_segmentation_sync,
  cv::Mat & full_seg_mask,
  size_t & detections_count,
  std::string & reason)
{
  ros2_yolos_cpp::srv::SegmentImage::Response::SharedPtr seg_response;
  if (!run_segmentation_sync(
      *roi_image_msg,
      roi_cfg.segmentation_timeout_s,
      seg_response,
      reason))
  {
    return false;
  }

  cv::Mat seg_mask_full = toCvMono(seg_response->mask);
  if (seg_mask_full.size() != roi_mask.size()) {
    reason = "segmentation mask/image size mismatch";
    return false;
  }

  full_seg_mask = cv::Mat::zeros(seg_mask_full.size(), CV_8UC1);
  seg_mask_full.copyTo(full_seg_mask, roi_mask);
  detections_count = seg_response->detections.detections.size();

  if (cv::countNonZero(full_seg_mask) == 0) {
    reason = "ROI segmentation produced empty mask";
    return false;
  }

  (void)image;
  return true;
}

sensor_msgs::msg::Image::SharedPtr buildRoiSegmentationDebugOverlay(
  const sensor_msgs::msg::Image::ConstSharedPtr & image,
  const cv::Rect & roi_rect,
  const cv::Mat & full_seg_mask)
{
  cv::Mat dbg = toCvBgr(*image);
  cv::rectangle(dbg, roi_rect, cv::Scalar(0, 0, 255), 2);
  overlayMask(dbg, full_seg_mask, cv::Scalar(255, 0, 0), 0.35);
  return cv_bridge::CvImage(image->header, "bgr8", dbg).toImageMsg();
}

}  // namespace cbp::world_model
