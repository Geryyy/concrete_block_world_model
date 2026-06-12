#include "concrete_block_world_model/utils/coarse_pose_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "concrete_block_world_model/utils/block_utils.hpp"

namespace cbp::world_model
{

using Block = concrete_block_world_model_interfaces::msg::Block;

namespace
{

bool readXyzOffsets(
  const sensor_msgs::msg::PointCloud2 & cloud_msg,
  int & x_offset,
  int & y_offset,
  int & z_offset,
  std::string & reason)
{
  x_offset = -1;
  y_offset = -1;
  z_offset = -1;
  for (const auto & field : cloud_msg.fields) {
    if (field.name == "x") {
      x_offset = static_cast<int>(field.offset);
    } else if (field.name == "y") {
      y_offset = static_cast<int>(field.offset);
    } else if (field.name == "z") {
      z_offset = static_cast<int>(field.offset);
    }
  }
  if (x_offset < 0 || y_offset < 0 || z_offset < 0) {
    reason = "cloud missing x/y/z fields";
    return false;
  }
  return true;
}

bool estimateSurfaceNormalFromPoints(
  const std::vector<Eigen::Vector3d> & points_world,
  const Eigen::Vector3d & center_world,
  const Eigen::Vector3d & camera_origin_world,
  Eigen::Vector3d & out_normal_world,
  std::string & reason)
{
  if (points_world.size() < 20U) {
    reason = "not enough points for surface-normal PCA";
    return false;
  }

  Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
  for (const auto & p : points_world) {
    const Eigen::Vector3d d = p - center_world;
    cov.noalias() += d * d.transpose();
  }
  cov /= static_cast<double>(points_world.size());

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov);
  if (solver.info() != Eigen::Success) {
    reason = "eigendecomposition failed";
    return false;
  }

  Eigen::Vector3d n = solver.eigenvectors().col(0);
  const double n_norm = n.norm();
  if (n_norm < 1e-6) {
    reason = "degenerate normal";
    return false;
  }
  n /= n_norm;

  const Eigen::Vector3d cam_to_center = center_world - camera_origin_world;
  if (n.dot(cam_to_center) < 0.0) {
    n = -n;
  }

  out_normal_world = n;
  return true;
}

void applyShapeAwareOffset(
  const cv::Mat & mask,
  const CoarsePoseConfig & cfg,
  const Eigen::Vector3d * camera_origin_world,
  const std::vector<Eigen::Vector3d> & selected_points,
  Eigen::Vector3d & center_world)
{
  if (mask.empty() || camera_origin_world == nullptr) {
    return;
  }

  const cv::Rect bbox = cv::boundingRect(mask);
  if (bbox.width <= 0 || bbox.height <= 0) {
    return;
  }

  const double ratio = static_cast<double>(std::max(bbox.width, bbox.height)) /
    static_cast<double>(std::max(1, std::min(bbox.width, bbox.height)));
  const bool is_square = ratio <= cfg.square_ratio_thresh;
  const double depth_offset_m = is_square ? cfg.front_center_offset_square_m : cfg.front_center_offset_rect_m;

  Eigen::Vector3d n_surface = Eigen::Vector3d::Zero();
  std::string normal_reason;
  if (estimateSurfaceNormalFromPoints(
      selected_points, center_world, *camera_origin_world, n_surface, normal_reason))
  {
    center_world += n_surface * depth_offset_m;
    return;
  }

  const Eigen::Vector3d ray = center_world - *camera_origin_world;
  const double ray_norm = ray.norm();
  if (ray_norm > 1e-6) {
    center_world += (ray / ray_norm) * depth_offset_m;
  }
}

void fillCoarseBlock(
  uint32_t detection_id,
  const std_msgs::msg::Header & header,
  const Eigen::Vector3d & center_world,
  float confidence,
  Block & out_block)
{
  out_block.id = detectionBlockId(detection_id);
  out_block.pose.position.x = center_world.x();
  out_block.pose.position.y = center_world.y();
  out_block.pose.position.z = center_world.z();
  out_block.pose.orientation.x = 0.0;
  out_block.pose.orientation.y = 0.0;
  out_block.pose.orientation.z = 0.0;
  out_block.pose.orientation.w = 1.0;
  out_block.confidence = confidence;
  out_block.last_seen = header.stamp;
  out_block.pose_status = Block::POSE_COARSE;
  out_block.task_status = Block::TASK_FREE;
  setDiagonalPoseCovariance(out_block, kCoarsePositionSigmaM, kCoarseOrientationSigmaRad);
}

}  // namespace

bool buildCoarseBlockFromOrganizedCloud(
  const CoarsePoseInput & in,
  const sensor_msgs::msg::PointCloud2 & cloud_msg,
  const CoarsePoseConfig & cfg,
  Block & out_block,
  std::string & reason)
{
  if (cloud_msg.width == 0 || cloud_msg.height == 0) {
    reason = "cloud has zero size";
    return false;
  }
  if (in.mask.empty()) {
    reason = "empty mask";
    return false;
  }
  if (in.mask.cols != static_cast<int>(cloud_msg.width) ||
    in.mask.rows != static_cast<int>(cloud_msg.height))
  {
    reason =
      "mask/cloud size mismatch (mask=" + std::to_string(in.mask.cols) + "x" +
      std::to_string(in.mask.rows) + ", cloud=" + std::to_string(cloud_msg.width) + "x" +
      std::to_string(cloud_msg.height) + ")";
    return false;
  }

  int x_offset = -1;
  int y_offset = -1;
  int z_offset = -1;
  if (!readXyzOffsets(cloud_msg, x_offset, y_offset, z_offset, reason)) {
    return false;
  }

  double sum_x = 0.0;
  double sum_y = 0.0;
  double sum_z = 0.0;
  int valid_points = 0;
  std::vector<Eigen::Vector3d> selected_points;
  selected_points.reserve(
    static_cast<size_t>(cloud_msg.width) * static_cast<size_t>(cloud_msg.height) / 16U);

  for (uint32_t v = 0; v < cloud_msg.height; ++v) {
    const auto * row_ptr = cloud_msg.data.data() + static_cast<size_t>(v) * cloud_msg.row_step;
    for (uint32_t u = 0; u < cloud_msg.width; ++u) {
      if (in.mask.at<uint8_t>(static_cast<int>(v), static_cast<int>(u)) == 0U) {
        continue;
      }
      const auto * point_ptr = row_ptr + static_cast<size_t>(u) * cloud_msg.point_step;
      float x = 0.0F;
      float y = 0.0F;
      float z = 0.0F;
      std::memcpy(&x, point_ptr + x_offset, sizeof(float));
      std::memcpy(&y, point_ptr + y_offset, sizeof(float));
      std::memcpy(&z, point_ptr + z_offset, sizeof(float));
      if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
        continue;
      }
      sum_x += static_cast<double>(x);
      sum_y += static_cast<double>(y);
      sum_z += static_cast<double>(z);
      selected_points.emplace_back(x, y, z);
      ++valid_points;
    }
  }

  if (valid_points < cfg.min_points) {
    reason = "too few valid masked points for coarse pose: " + std::to_string(valid_points);
    return false;
  }

  const double inv_n = 1.0 / static_cast<double>(valid_points);
  Eigen::Vector3d center(sum_x * inv_n, sum_y * inv_n, sum_z * inv_n);
  applyShapeAwareOffset(in.mask, cfg, in.camera_origin_world, selected_points, center);
  fillCoarseBlock(in.detection_id, in.header, center, cfg.min_confidence, out_block);
  reason = "coarse centroid from masked cloud points=" + std::to_string(valid_points);
  return true;
}

bool buildCoarseBlockFromCutoutCloud(
  const CoarsePoseInput & in,
  const sensor_msgs::msg::PointCloud2 & cutout_cloud_msg,
  const CoarsePoseConfig & cfg,
  Block & out_block,
  std::string & reason)
{
  if (cutout_cloud_msg.data.empty()) {
    reason = "empty cutout cloud";
    return false;
  }

  int x_offset = -1;
  int y_offset = -1;
  int z_offset = -1;
  if (!readXyzOffsets(cutout_cloud_msg, x_offset, y_offset, z_offset, reason)) {
    return false;
  }

  const size_t points_n = static_cast<size_t>(cutout_cloud_msg.width) *
    static_cast<size_t>(cutout_cloud_msg.height);
  if (points_n == 0) {
    reason = "cutout cloud has zero points";
    return false;
  }

  double sum_x = 0.0;
  double sum_y = 0.0;
  double sum_z = 0.0;
  int valid_points = 0;
  std::vector<Eigen::Vector3d> selected_points;
  selected_points.reserve(points_n);
  for (size_t i = 0; i < points_n; ++i) {
    const auto * point_ptr = cutout_cloud_msg.data.data() + i * cutout_cloud_msg.point_step;
    float x = 0.0F;
    float y = 0.0F;
    float z = 0.0F;
    std::memcpy(&x, point_ptr + x_offset, sizeof(float));
    std::memcpy(&y, point_ptr + y_offset, sizeof(float));
    std::memcpy(&z, point_ptr + z_offset, sizeof(float));
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
      continue;
    }
    sum_x += static_cast<double>(x);
    sum_y += static_cast<double>(y);
    sum_z += static_cast<double>(z);
    selected_points.emplace_back(x, y, z);
    ++valid_points;
  }

  if (valid_points < cfg.min_points) {
    reason = "too few valid cutout points for coarse pose: " + std::to_string(valid_points);
    return false;
  }

  const double inv_n = 1.0 / static_cast<double>(valid_points);
  Eigen::Vector3d center(sum_x * inv_n, sum_y * inv_n, sum_z * inv_n);
  applyShapeAwareOffset(in.mask, cfg, in.camera_origin_world, selected_points, center);
  fillCoarseBlock(in.detection_id, in.header, center, cfg.min_confidence, out_block);
  reason = "coarse centroid from cutout cloud points=" + std::to_string(valid_points);
  return true;
}

}  // namespace cbp::world_model
