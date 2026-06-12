#include "concrete_block_world_model/world_model/refine_flow.hpp"

#include <cv_bridge/cv_bridge.h>
#include <opencv2/core.hpp>
#include <rclcpp/rclcpp.hpp>

#include "concrete_block_world_model/world_model/roi_refinement.hpp"
#include "concrete_block_world_model/world_model/state_manager.hpp"

namespace cbp::world_model
{

namespace
{

bool buildRoiMask(
  const RefineFlowRuntime & rt,
  const sensor_msgs::msg::Image::ConstSharedPtr & image,
  const Eigen::Vector3d & p_camera,
  const RoiInputConfig & roi_cfg,
  cv::Mat & roi_mask,
  cv::Rect & roi_rect,
  std::string & reason)
{
  ProjectionIntrinsics intr;
  if (!rt.get_projection_intrinsics || !rt.get_projection_intrinsics(intr)) {
    reason = "camera_info not received or invalid intrinsics";
    return false;
  }
  return buildRoiMaskFromPrediction(intr, image, p_camera, roi_cfg, roi_mask, roi_rect, reason);
}

}  // namespace

void processRefineGraspedWithFkRoi(
  const RefineRequest & request,
  const RefineGraspedConfig & cfg,
  const RefineFlowRuntime & rt,
  const sensor_msgs::msg::Image::ConstSharedPtr & image,
  const sensor_msgs::msg::PointCloud2::ConstSharedPtr & cloud,
  const std::chrono::steady_clock::time_point & t_start)
{
  if (!rt.registration_ready || !rt.registration_ready()) {
    rt.complete_one_shot(request.sequence, false, "Registration action unavailable.");
    rt.reset_busy();
    return;
  }

  Eigen::Vector3d p_world_fk = Eigen::Vector3d::Zero();
  Eigen::Vector3d p_camera = Eigen::Vector3d::Zero();
  Eigen::Quaterniond q_world_fk = Eigen::Quaterniond::Identity();
  std::string reason;
  if (!rt.lookup_predicted_grasped_pose ||
    !rt.lookup_predicted_grasped_pose(image->header, p_world_fk, p_camera, q_world_fk, reason))
  {
    RCLCPP_WARN(rt.logger, "REFINE_GRASPED FK+ROI failed: %s", reason.c_str());
    rt.publish_persistent_world(cloud->header);
    rt.complete_one_shot(request.sequence, false, "REFINE_GRASPED FK prediction failed: " + reason);
    rt.reset_busy();
    return;
  }

  cv::Mat roi_mask;
  cv::Rect roi_rect;
  if (!buildRoiMask(rt, image, p_camera, cfg.roi_cfg, roi_mask, roi_rect, reason)) {
    RCLCPP_WARN(rt.logger, "REFINE_GRASPED FK+ROI failed: %s", reason.c_str());
    rt.publish_persistent_world(cloud->header);
    rt.complete_one_shot(request.sequence, false, "REFINE_GRASPED ROI construction failed: " + reason);
    rt.reset_busy();
    return;
  }

  auto roi_image_msg = buildRoiSegmentationInputImage(image, roi_rect, cfg.roi_cfg);
  if (cfg.debug_refine_grasped_roi_input_enabled && rt.publish_roi_input) {
    rt.publish_roi_input(*roi_image_msg);
  }

  cv::Mat full_seg_mask;
  size_t detections_count = 0;
  if (!roiSegmentationToFullMask(
      image,
      roi_mask,
      roi_image_msg,
      cfg.roi_cfg,
      rt.run_segmentation_sync,
      full_seg_mask,
      detections_count,
      reason))
  {
    RCLCPP_WARN(rt.logger, "REFINE_GRASPED FK+ROI segmentation failed: %s", reason.c_str());
    rt.publish_persistent_world(cloud->header);
    rt.complete_one_shot(request.sequence, false, "REFINE_GRASPED ROI segmentation failed: " + reason);
    rt.reset_busy();
    return;
  }

  if (cfg.debug_detection_overlay_enabled && rt.publish_debug_overlay) {
    if (auto dbg = buildRoiSegmentationDebugOverlay(image, roi_rect, full_seg_mask)) {
      rt.publish_debug_overlay(*dbg);
    }
  }

  RCLCPP_INFO(
    rt.logger,
    "REFINE_GRASPED FK+ROI segmentation: roi=[x=%d y=%d w=%d h=%d] detections=%zu mask_pixels=%d",
    roi_rect.x,
    roi_rect.y,
    roi_rect.width,
    roi_rect.height,
    detections_count,
    cv::countNonZero(full_seg_mask));

  auto full_mask_msg = cv_bridge::CvImage(image->header, "mono8", full_seg_mask).toImageMsg();
  concrete_block_world_model_interfaces::msg::Block block;
  std::string reg_reason;
  const auto t_reg_start = std::chrono::steady_clock::now();
  const bool registration_ok = rt.run_registration_sync(
    kRefineDetectionId,
    *full_mask_msg,
    *cloud,
    cloud->header,
    request.registration_timeout_s,
    block,
    reg_reason,
    cfg.object_class + "#REFINE_GRASPED");
  const auto t_reg_end = std::chrono::steady_clock::now();

  size_t registrations_ok = 0;
  if (registration_ok) {
    bool fusion_ok = true;
    std::string fusion_reason;

    if (cfg.pose_fusion.enabled) {
      if (cfg.pose_fusion.mode != "position_from_registration_orientation_from_fk") {
        fusion_ok = false;
        fusion_reason = "unsupported pose_fusion.mode='" + cfg.pose_fusion.mode + "'";
      } else {
        const auto fusion_result = fuseRegistrationPositionWithFkOrientation(
          block.pose,
          p_world_fk,
          q_world_fk,
          cfg.pose_fusion.max_translation_jump_m,
          cfg.pose_fusion.max_z_delta_m);
        fusion_ok = fusion_result.success;
        fusion_reason = fusion_result.reason;

        if (cfg.pose_fusion.debug_log) {
          const Eigen::Vector3d p_reg(
            block.pose.position.x,
            block.pose.position.y,
            block.pose.position.z);
          RCLCPP_INFO(
            rt.logger,
            "REFINE_GRASPED pose fusion: source=FK_ORIENTATION fk_pos=[%.3f %.3f %.3f] reg_pos=[%.3f %.3f %.3f] fused_pos=[%.3f %.3f %.3f] residual=%.3f z_delta=%.3f gate=%s",
            p_world_fk.x(), p_world_fk.y(), p_world_fk.z(),
            p_reg.x(), p_reg.y(), p_reg.z(),
            block.pose.position.x, block.pose.position.y, block.pose.position.z,
            fusion_result.residual_norm,
            fusion_result.z_delta,
            fusion_ok ? "PASS" : "FAIL");
        }
      }
    }

    if (!fusion_ok) {
      RCLCPP_WARN(rt.logger, "REFINE_GRASPED pose fusion rejected: %s", fusion_reason.c_str());
      reg_reason = fusion_reason;
    } else {
      std::string assigned_id;
      std::string upsert_reason;
      const bool upsert_ok = rt.upsert_block(block, assigned_id, upsert_reason);
      if (upsert_ok) {
        ++registrations_ok;
        RCLCPP_INFO(
          rt.logger,
          "REFINE_GRASPED FK+ROI accepted: target=%s assigned=%s roi=[x=%d y=%d w=%d h=%d]",
          request.target_block_id.c_str(),
          assigned_id.c_str(),
          roi_rect.x,
          roi_rect.y,
          roi_rect.width,
          roi_rect.height);
      } else {
        RCLCPP_WARN(
          rt.logger,
          "REFINE_GRASPED FK+ROI rejected after association checks: %s",
          upsert_reason.c_str());
        reg_reason = upsert_reason;
      }
    }
  } else {
    RCLCPP_WARN(rt.logger, "REFINE_GRASPED FK+ROI registration failed: %s", reg_reason.c_str());
  }

  const auto reg_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_reg_end - t_reg_start).count();
  const auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_reg_end - t_start).count();
  rt.record_timing(0, 0, reg_ms, total_ms);
  rt.publish_persistent_world(cloud->header);
  rt.complete_one_shot(
    request.sequence,
    registrations_ok > 0,
    registrations_ok > 0 ?
    ("Pose estimation finished (mode=REFINE_GRASPED_FK_ROI, registration_candidates=1, registrations=1).") :
    ("Pose estimation finished (mode=REFINE_GRASPED_FK_ROI, registration_candidates=1, registrations=0, reason=" +
    reg_reason + ")."));
  rt.reset_busy();
}

bool tryProcessRefineBlockWithPoseRoi(
  const RefineRequest & request,
  const RefineBlockConfig & cfg,
  const RefineFlowRuntime & rt,
  const sensor_msgs::msg::Image::ConstSharedPtr & image,
  const sensor_msgs::msg::PointCloud2::ConstSharedPtr & cloud,
  const std::chrono::steady_clock::time_point & t_start)
{
  if (!cfg.use_pose_roi) {
    return false;
  }
  if (request.target_block_id.empty()) {
    return false;
  }
  if (!rt.registration_ready || !rt.registration_ready()) {
    rt.complete_one_shot(request.sequence, false, "Registration action unavailable.");
    rt.reset_busy();
    return true;
  }

  concrete_block_world_model_interfaces::msg::Block expected_target;
  if (!rt.get_expected_target || !rt.get_expected_target(request.target_block_id, expected_target)) {
    RCLCPP_WARN(
      rt.logger,
      "REFINE_BLOCK pose+ROI fallback: target '%s' not found in world model.",
      request.target_block_id.c_str());
    return false;
  }

  const Eigen::Vector3d p_world(
    expected_target.pose.position.x,
    expected_target.pose.position.y,
    expected_target.pose.position.z);
  Eigen::Vector3d p_camera = Eigen::Vector3d::Zero();
  std::string reason;
  if (!rt.world_point_to_camera || !rt.world_point_to_camera(image->header, p_world, p_camera, reason)) {
    RCLCPP_WARN(rt.logger, "REFINE_BLOCK pose+ROI fallback: %s", reason.c_str());
    return false;
  }

  cv::Mat roi_mask;
  cv::Rect roi_rect;
  if (!buildRoiMask(rt, image, p_camera, cfg.roi_cfg, roi_mask, roi_rect, reason)) {
    RCLCPP_WARN(rt.logger, "REFINE_BLOCK pose+ROI fallback: %s", reason.c_str());
    return false;
  }

  auto roi_image_msg = buildRoiSegmentationInputImage(image, roi_rect, cfg.roi_cfg);
  cv::Mat full_seg_mask;
  size_t detections_count = 0;
  if (!roiSegmentationToFullMask(
      image,
      roi_mask,
      roi_image_msg,
      cfg.roi_cfg,
      rt.run_segmentation_sync,
      full_seg_mask,
      detections_count,
      reason))
  {
    RCLCPP_WARN(rt.logger, "REFINE_BLOCK pose+ROI failed: %s", reason.c_str());
    rt.publish_persistent_world(cloud->header);
    rt.complete_one_shot(request.sequence, false, "REFINE_BLOCK ROI segmentation failed: " + reason);
    rt.reset_busy();
    return true;
  }

  if (cfg.debug_detection_overlay_enabled && rt.publish_debug_overlay) {
    if (auto dbg = buildRoiSegmentationDebugOverlay(image, roi_rect, full_seg_mask)) {
      rt.publish_debug_overlay(*dbg);
    }
  }

  RCLCPP_INFO(
    rt.logger,
    "REFINE_BLOCK pose+ROI: target=%s roi=[x=%d y=%d w=%d h=%d] detections=%zu mask_pixels=%d",
    request.target_block_id.c_str(),
    roi_rect.x,
    roi_rect.y,
    roi_rect.width,
    roi_rect.height,
    detections_count,
    cv::countNonZero(full_seg_mask));

  auto full_mask_msg = cv_bridge::CvImage(image->header, "mono8", full_seg_mask).toImageMsg();
  concrete_block_world_model_interfaces::msg::Block block;
  std::string reg_reason;
  const auto t_reg_start = std::chrono::steady_clock::now();
  const bool registration_ok = rt.run_registration_sync(
    kRefineDetectionId,
    *full_mask_msg,
    *cloud,
    cloud->header,
    request.registration_timeout_s,
    block,
    reg_reason,
    "");
  const auto t_reg_end = std::chrono::steady_clock::now();

  size_t registrations_ok = 0;
  if (registration_ok) {
    const double dist = blockDistance(block, expected_target);
    if (dist > cfg.refine_target_max_distance_m) {
      reg_reason = "registered pose too far from target: dist=" + std::to_string(dist) + " m";
      RCLCPP_WARN(rt.logger, "REFINE_BLOCK pose+ROI rejected: %s", reg_reason.c_str());
    } else {
      std::string assigned_id;
      std::string upsert_reason;
      const bool upsert_ok = rt.upsert_block(block, assigned_id, upsert_reason);
      if (upsert_ok) {
        ++registrations_ok;
        RCLCPP_INFO(
          rt.logger,
          "REFINE_BLOCK pose+ROI accepted: target=%s assigned=%s dist=%.3f m",
          request.target_block_id.c_str(),
          assigned_id.c_str(),
          dist);
      } else {
        reg_reason = upsert_reason;
        RCLCPP_WARN(rt.logger, "REFINE_BLOCK pose+ROI rejected: %s", reg_reason.c_str());
      }
    }
  } else {
    RCLCPP_WARN(rt.logger, "REFINE_BLOCK pose+ROI registration failed: %s", reg_reason.c_str());
  }

  const auto reg_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_reg_end - t_reg_start).count();
  const auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_reg_end - t_start).count();
  rt.record_timing(0, 0, reg_ms, total_ms);
  rt.publish_persistent_world(cloud->header);
  rt.complete_one_shot(
    request.sequence,
    registrations_ok > 0,
    registrations_ok > 0 ?
    ("Pose estimation finished (mode=REFINE_BLOCK_POSE_ROI, registration_candidates=1, registrations=1).") :
    ("Pose estimation finished (mode=REFINE_BLOCK_POSE_ROI, registration_candidates=1, registrations=0, reason=" +
    reg_reason + ")."));
  rt.reset_busy();
  return true;
}

}  // namespace cbp::world_model

