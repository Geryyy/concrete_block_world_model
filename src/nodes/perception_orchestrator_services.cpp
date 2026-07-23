#include "concrete_block_world_model/nodes/perception_orchestrator_node.hpp"

#include "concrete_block_world_model/utils/block_utils.hpp"
#include "concrete_block_world_model/utils/img_utils.hpp"
#include "concrete_block_world_model/utils/world_model_utils.hpp"

#include <array>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <unordered_set>

#include <opencv2/imgproc.hpp>

namespace
{

int64_t stampNanoseconds(const builtin_interfaces::msg::Time & stamp)
{
  return static_cast<int64_t>(stamp.sec) * 1000000000LL +
    static_cast<int64_t>(stamp.nanosec);
}

bool projectPoint(
  const Eigen::Vector3d & point_camera,
  double fx,
  double fy,
  double cx,
  double cy,
  cv::Point & pixel)
{
  constexpr double kMinDepthM = 1e-3;
  if (!point_camera.allFinite() || point_camera.z() <= kMinDepthM) {
    return false;
  }
  const double u = fx * point_camera.x() / point_camera.z() + cx;
  const double v = fy * point_camera.y() / point_camera.z() + cy;
  constexpr double kMaxPixelMagnitude = 1e6;
  if (!std::isfinite(u) || !std::isfinite(v) ||
    std::abs(u) > kMaxPixelMagnitude || std::abs(v) > kMaxPixelMagnitude)
  {
    return false;
  }
  pixel = cv::Point(static_cast<int>(std::lround(u)), static_cast<int>(std::lround(v)));
  return true;
}

}  // namespace

void PerceptionOrchestratorNode::cacheSceneDiscoveryImage(
  const sensor_msgs::msg::Image::ConstSharedPtr msg)
{
  std::lock_guard<std::mutex> lock(scene_discovery_image_mutex_);
  scene_discovery_images_.push_back(std::move(msg));
  constexpr std::size_t kImageCacheCapacity = 30U;
  while (scene_discovery_images_.size() > kImageCacheCapacity) {
    scene_discovery_images_.pop_front();
  }
}

void PerceptionOrchestratorNode::publishSceneDiscoveryPoseOverlay(
  const std_msgs::msg::Header & cloud_header,
  const std::vector<Block> & blocks)
{
  if (!scene_discovery_pose_overlay_pub_ || blocks.empty()) {
    return;
  }

  sensor_msgs::msg::Image::ConstSharedPtr image;
  const int64_t cloud_stamp_ns = stampNanoseconds(cloud_header.stamp);
  int64_t best_delta_ns = std::numeric_limits<int64_t>::max();
  {
    std::lock_guard<std::mutex> lock(scene_discovery_image_mutex_);
    for (const auto & candidate : scene_discovery_images_) {
      const int64_t delta_ns = std::llabs(stampNanoseconds(candidate->header.stamp) - cloud_stamp_ns);
      if (delta_ns < best_delta_ns) {
        best_delta_ns = delta_ns;
        image = candidate;
      }
    }
  }
  const int64_t max_delta_ns = static_cast<int64_t>(
    std::llround(scene_discovery_overlay_max_image_delta_s_ * 1e9));
  const int64_t fallback_max_delta_ns = static_cast<int64_t>(
    std::llround(scene_discovery_overlay_fallback_max_image_delta_s_ * 1e9));
  if (!image || best_delta_ns > fallback_max_delta_ns) {
    RCLCPP_WARN(
      get_logger(),
      "Scene-discovery pose overlay skipped: no Blackfly image within %.3f s of cloud stamp (nearest %.3f s).",
      scene_discovery_overlay_fallback_max_image_delta_s_,
      image ? static_cast<double>(best_delta_ns) * 1e-9 : -1.0);
    return;
  }
  const bool stale_rgb = best_delta_ns > max_delta_ns;
  if (stale_rgb) {
    RCLCPP_WARN(
      get_logger(),
      "Scene-discovery pose overlay uses STALE RGB: image/cloud delta is %.3f s (strict limit %.3f s).",
      static_cast<double>(best_delta_ns) * 1e-9,
      scene_discovery_overlay_max_image_delta_s_);
  }
  if (image->header.frame_id.empty()) {
    RCLCPP_WARN(get_logger(), "Scene-discovery pose overlay skipped: Blackfly image has no frame_id.");
    return;
  }

  sensor_msgs::msg::CameraInfo::SharedPtr camera_info;
  int64_t best_camera_info_delta_ns = std::numeric_limits<int64_t>::max();
  {
    std::lock_guard<std::mutex> lock(camera_info_mutex_);
    const int64_t image_stamp_ns = stampNanoseconds(image->header.stamp);
    for (const auto & candidate : scene_discovery_camera_infos_) {
      const int64_t delta_ns = std::llabs(
        stampNanoseconds(candidate->header.stamp) - image_stamp_ns);
      if (delta_ns < best_camera_info_delta_ns) {
        best_camera_info_delta_ns = delta_ns;
        camera_info = candidate;
      }
    }
  }
  if (!camera_info) {
    RCLCPP_WARN(get_logger(), "Scene-discovery pose overlay skipped: no cached CameraInfo.");
    return;
  }
  if (best_camera_info_delta_ns > fallback_max_delta_ns) {
    RCLCPP_WARN(
      get_logger(),
      "Scene-discovery pose overlay skipped: CameraInfo is %.3f s from selected image.",
      static_cast<double>(best_camera_info_delta_ns) * 1e-9);
    return;
  }
  if (camera_info->header.frame_id != image->header.frame_id) {
    RCLCPP_WARN(
      get_logger(),
      "Scene-discovery pose overlay skipped: CameraInfo frame '%s' does not match image frame '%s'.",
      camera_info->header.frame_id.c_str(), image->header.frame_id.c_str());
    return;
  }
  if (camera_info->width != image->width || camera_info->height != image->height) {
    RCLCPP_WARN(
      get_logger(),
      "Scene-discovery pose overlay skipped: CameraInfo dimensions %ux%u do not match image %ux%u.",
      camera_info->width, camera_info->height, image->width, image->height);
    return;
  }
  CameraIntrinsics intrinsics;
  intrinsics.projection_fx = camera_info->p[0];
  intrinsics.projection_fy = camera_info->p[5];
  intrinsics.projection_cx = camera_info->p[2];
  intrinsics.projection_cy = camera_info->p[6];
  if (intrinsics.projection_fx <= 0.0 || intrinsics.projection_fy <= 0.0) {
    RCLCPP_WARN(get_logger(), "Scene-discovery pose overlay skipped: invalid CameraInfo projection matrix.");
    return;
  }

  Eigen::Matrix4d T_camera_world = Eigen::Matrix4d::Identity();
  try {
    const auto tf_camera_world = tf_buffer_->lookupTransform(
      image->header.frame_id,
      world_frame_,
      rclcpp::Time(image->header.stamp),
      rclcpp::Duration::from_seconds(0.2));
    T_camera_world = transformToEigen(tf_camera_world);
  } catch (const tf2::TransformException & ex) {
    RCLCPP_WARN(
      get_logger(), "Scene-discovery pose overlay skipped: camera<-world TF failed: %s", ex.what());
    return;
  }

  cv::Mat output;
  try {
    output = toCvBgr(*image);
  } catch (const std::exception & ex) {
    RCLCPP_WARN(get_logger(), "Scene-discovery pose overlay skipped: image conversion failed: %s", ex.what());
    return;
  }

  constexpr std::array<std::array<int, 2>, 12> kEdges{{
    {{0, 1}}, {{0, 2}}, {{0, 4}}, {{1, 3}}, {{1, 5}}, {{2, 3}},
    {{2, 6}}, {{3, 7}}, {{4, 5}}, {{4, 6}}, {{5, 7}}, {{6, 7}}
  }};
  const Eigen::Vector3d half_dimensions(
    block_dimensions_m_[0] / 2.0,
    block_dimensions_m_[1] / 2.0,
    block_dimensions_m_[2] / 2.0);
  for (const auto & block : blocks) {
    const Eigen::Quaterniond q_world_block(
      block.pose.orientation.w,
      block.pose.orientation.x,
      block.pose.orientation.y,
      block.pose.orientation.z);
    if (!q_world_block.coeffs().allFinite() || q_world_block.norm() < 1e-8 ||
      !std::isfinite(block.pose.position.x) || !std::isfinite(block.pose.position.y) ||
      !std::isfinite(block.pose.position.z))
    {
      RCLCPP_WARN(get_logger(), "Scene-discovery pose overlay skipped a block with an invalid pose.");
      continue;
    }
    const Eigen::Matrix3d R_world_block = q_world_block.normalized().toRotationMatrix();
    const Eigen::Vector3d p_world(
      block.pose.position.x,
      block.pose.position.y,
      block.pose.position.z);
    std::array<cv::Point, 8> pixels;
    std::array<bool, 8> visible{};
    for (int corner = 0; corner < 8; ++corner) {
      const Eigen::Vector3d local(
        (corner & 1) == 0 ? -half_dimensions.x() : half_dimensions.x(),
        (corner & 2) == 0 ? -half_dimensions.y() : half_dimensions.y(),
        (corner & 4) == 0 ? -half_dimensions.z() : half_dimensions.z());
      const Eigen::Vector3d corner_world = p_world + R_world_block * local;
      const Eigen::Vector4d p_world_h(
        corner_world.x(), corner_world.y(), corner_world.z(), 1.0);
      visible[corner] = projectPoint(
        (T_camera_world * p_world_h).head<3>(),
        intrinsics.projection_fx,
        intrinsics.projection_fy,
        intrinsics.projection_cx,
        intrinsics.projection_cy,
        pixels[corner]);
    }
    const cv::Scalar black(0, 0, 0);
    const cv::Scalar outline = stale_rgb ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 255);
    for (const auto & edge : kEdges) {
      if (visible[edge[0]] && visible[edge[1]]) {
        cv::line(output, pixels[edge[0]], pixels[edge[1]], black, 4, cv::LINE_AA);
        cv::line(output, pixels[edge[0]], pixels[edge[1]], outline, 2, cv::LINE_AA);
      }
    }
    const Eigen::Vector4d center_world_h(p_world.x(), p_world.y(), p_world.z(), 1.0);
    cv::Point center_pixel;
    if (projectPoint(
        (T_camera_world * center_world_h).head<3>(),
        intrinsics.projection_fx,
        intrinsics.projection_fy,
        intrinsics.projection_cx,
        intrinsics.projection_cy,
        center_pixel)) {
      std::ostringstream label;
      label << (stale_rgb ? "STALE RGB " : "")
            << (block.id.empty() ? "detected_block" : block.id)
            << " coarse score=" << std::fixed << std::setprecision(2) << block.confidence
            << " dt=" << std::setprecision(3) << static_cast<double>(best_delta_ns) * 1e-9 << "s";
      cv::putText(output, label.str(), center_pixel + cv::Point(8, -8),
        cv::FONT_HERSHEY_SIMPLEX, 0.5, black, 3, cv::LINE_AA);
      cv::putText(output, label.str(), center_pixel + cv::Point(8, -8),
        cv::FONT_HERSHEY_SIMPLEX, 0.5, outline, 1, cv::LINE_AA);
    }
  }
  scene_discovery_pose_overlay_pub_->publish(
    *cv_bridge::CvImage(image->header, "bgr8", output).toImageMsg());
}

void PerceptionOrchestratorNode::completeOneShotRequest(uint64_t sequence, bool success, const std::string & message)
  {
    std::lock_guard<std::mutex> lock(one_shot_mutex_);
    if (sequence == 0 || sequence != active_one_shot_.sequence) {
      return;
    }
    active_one_shot_ = OneShotRequest{};
    one_shot_done_sequence_ = sequence;
    one_shot_last_success_ = success;
    one_shot_last_message_ = message;
    one_shot_cv_.notify_all();
  }

bool PerceptionOrchestratorNode::runDetectorSceneDiscovery(
    double timeout_s, RunPoseSrv::Response & response)
  {
    const auto service_wait = std::chrono::duration<double>(std::min(timeout_s, 1.0));
    if (!discover_blocks_client_->wait_for_service(service_wait)) {
      response.success = false;
      response.message = "Detector discovery service '" + detector_discover_service_ + "' is unavailable.";
      response.blocks = latestWorldSnapshot();
      return false;
    }

    auto request = std::make_shared<DiscoverBlocksSrv::Request>();
    request->timeout_s = static_cast<float>(timeout_s);
    auto future = discover_blocks_client_->async_send_request(request);
    if (future.wait_for(std::chrono::duration<double>(timeout_s)) != std::future_status::ready) {
      response.success = false;
      response.message = "Timed out waiting for detector scene discovery.";
      response.blocks = latestWorldSnapshot();
      return false;
    }

    DiscoverBlocksSrv::Response::SharedPtr detector_response;
    try {
      detector_response = future.get();
    } catch (const std::exception & error) {
      response.success = false;
      response.message = std::string("Detector scene discovery transport failed: ") + error.what();
      response.blocks = latestWorldSnapshot();
      return false;
    }
    if (!detector_response->success) {
      response.success = false;
      response.message = "Detector scene discovery failed: " + detector_response->message;
      response.blocks = latestWorldSnapshot();
      return false;
    }

    std_msgs::msg::Header header = detector_response->blocks.header;
    header.frame_id = world_frame_;
    if (header.stamp.sec == 0 && header.stamp.nanosec == 0U) {
      header.stamp = now();
    }
    std::size_t accepted = 0;
    std::size_t rejected = 0;
    std::vector<Block> accepted_blocks;
    {
      // Associate every detector observation against the same pre-discovery
      // snapshot.  Inserting each observation immediately would let later
      // observations associate to earlier observations from this same cloud,
      // collapsing distinct nearby blocks into one world-model entry.
      std::lock_guard<std::mutex> lock(persistent_world_mutex_);
      const auto association_snapshot = persistent_world_;
      std::unordered_set<std::string> claimed_existing_ids;
      std::vector<std::pair<std::string, Block>> pending_updates;
      auto discovery_association = associationConfig();
      discovery_association.min_update_confidence =
        runtime_cfg_.scene_discovery_min_detector_confidence;
      discovery_association.association_max_distance_m =
        runtime_cfg_.scene_discovery_association_max_distance_m;
      for (auto incoming : detector_response->blocks.blocks) {
        incoming.id.clear();
        incoming.pose_status = Block::POSE_COARSE;
        incoming.task_status = Block::TASK_FREE;
        incoming.last_seen = header.stamp;

        // Remove already claimed prior-world entries.  This makes association
        // one-to-one within the detector response while preserving the stable
        // snapshot for every other decision.
        auto candidate_world = association_snapshot;
        for (const auto & id : claimed_existing_ids) {
          candidate_world.erase(id);
        }
        std::string assigned_id;
        std::string reason;
        if (cbpwm::upsertRegisteredBlock(
            candidate_world, world_block_counter_, incoming,
            cbpwm::OneShotMode::kSceneDiscovery, "", header, *get_clock(),
            discovery_association, assigned_id, reason))
        {
          if (association_snapshot.find(assigned_id) != association_snapshot.end()) {
            claimed_existing_ids.insert(assigned_id);
          }
          pending_updates.emplace_back(assigned_id, std::move(candidate_world.at(assigned_id)));
          accepted_blocks.push_back(pending_updates.back().second);
          ++accepted;
        } else {
          ++rejected;
          RCLCPP_WARN(get_logger(), "Detector discovery observation rejected: %s", reason.c_str());
        }
      }
      for (auto & update : pending_updates) {
        persistent_world_[update.first] = std::move(update.second);
      }
    }
    publishPersistentWorld(header);
    publishSceneDiscoveryPoseOverlay(header, accepted_blocks);
    response.blocks = latestWorldSnapshot();
    response.success = true;
    response.message = "Detector scene discovery: " + std::to_string(accepted) +
      " associated, " + std::to_string(rejected) + " rejected.";
    return true;
  }

void PerceptionOrchestratorNode::handleRunPoseEstimation(
    const std::shared_ptr<RunPoseSrv::Request> request,
    std::shared_ptr<RunPoseSrv::Response> response)
  {
    const cbpwm::OneShotMode run_mode = cbpwm::parseOneShotMode(request->mode);
    if (run_mode == cbpwm::OneShotMode::kNone) {
      response->success = false;
      response->message = "Unsupported mode: " + request->mode;
      return;
    }

    if (run_mode == cbpwm::OneShotMode::kSceneDiscovery) {
      const double timeout_s = request->timeout_s > 0.0f ? request->timeout_s : 5.0;
      runDetectorSceneDiscovery(timeout_s, *response);
      return;
    }

    OneShotRequest run_request;
    run_request.mode = run_mode;
    run_request.target_block_id = request->target_block_id;
    run_request.enable_debug = request->enable_debug;
    run_request.registration_timeout_s =
      request->timeout_s > 0.0f ? static_cast<double>(request->timeout_s) : 3.0;

    if (run_mode == cbpwm::OneShotMode::kRefineGrasped && run_request.target_block_id.empty()) {
      run_request.target_block_id = resolveGraspedBlockId();
      if (run_request.target_block_id.empty()) {
        response->success = false;
        response->message = "No grasped block found (TASK_MOVE) and no target_block_id provided.";
        return;
      }
    }

    {
      std::lock_guard<std::mutex> lock(one_shot_mutex_);
      if (active_one_shot_.mode != cbpwm::OneShotMode::kNone) {
        response->success = false;
        response->message = "Another one-shot request is already running.";
        return;
      }
      run_request.sequence = ++one_shot_sequence_counter_;
      active_one_shot_ = run_request;
    }

    debug_detection_overlay_enabled_ = request->enable_debug;

    RCLCPP_INFO(
      get_logger(),
      "Scheduled one-shot pose estimation: mode=%s target=%s",
      cbpwm::oneShotModeToString(run_mode),
      run_request.target_block_id.empty() ? "<all>" : run_request.target_block_id.c_str());

    const double timeout_s = request->timeout_s > 0.0f ? request->timeout_s : 5.0;
    {
      std::unique_lock<std::mutex> lock(one_shot_mutex_);
      const bool done = one_shot_cv_.wait_for(
        lock,
        std::chrono::duration<double>(timeout_s),
        [this, &run_request]() {
          return one_shot_done_sequence_ >= run_request.sequence;
        });
      if (!done) {
        if (active_one_shot_.sequence == run_request.sequence) {
          active_one_shot_ = OneShotRequest{};
        }
        response->success = false;
        response->message = "Timed out waiting for one-shot result.";
        response->blocks = latestWorldSnapshot();
        return;
      }
      response->success = one_shot_last_success_;
      response->message = one_shot_last_message_;
    }

    response->blocks = latestWorldSnapshot();
  }

void PerceptionOrchestratorNode::handleGetCoarseBlocks(
    const std::shared_ptr<GetCoarseSrv::Request> request,
    std::shared_ptr<GetCoarseSrv::Response> response)
  {
    (void)request;
    response->success = true;
    response->blocks = latestWorldSnapshot();
    response->message = "ok";
    RCLCPP_INFO(
      get_logger(),
      "GetCoarseBlocks -> %zu blocks (stamp=%u.%u)",
      response->blocks.blocks.size(),
      response->blocks.header.stamp.sec,
      response->blocks.header.stamp.nanosec);
  }

void PerceptionOrchestratorNode::handleGetPlanningScene(
    const std::shared_ptr<GetPlanningSceneSrv::Request> request,
    std::shared_ptr<GetPlanningSceneSrv::Response> response)
  {
    (void)request;
    response->success = true;
    response->scene = latestPlanningSceneSnapshot();
    response->message = "ok";
    RCLCPP_INFO(
      get_logger(),
      "GetPlanningScene -> %zu objects (stamp=%u.%u)",
      response->scene.objects.size(),
      response->scene.header.stamp.sec,
      response->scene.header.stamp.nanosec);
  }

void PerceptionOrchestratorNode::handleSetBlockTaskStatus(
    const std::shared_ptr<SetBlockTaskStatusSrv::Request> request,
    std::shared_ptr<SetBlockTaskStatusSrv::Response> response)
  {
    const std::string block_id = request->block_id;
    const int32_t target_task_status = request->task_status;

    if (block_id.empty()) {
      response->success = false;
      response->message = "block_id must not be empty.";
      RCLCPP_WARN(get_logger(), "SetBlockTaskStatus rejected: %s", response->message.c_str());
      return;
    }
    if (!isKnownTaskStatus(target_task_status)) {
      response->success = false;
      response->message = "Unsupported task_status: " + std::to_string(target_task_status);
      RCLCPP_WARN(get_logger(), "SetBlockTaskStatus rejected: %s", response->message.c_str());
      return;
    }

    std_msgs::msg::Header publish_header;
    publish_header.stamp = now();
    {
      const auto snapshot = latestWorldSnapshot();
      publish_header.frame_id = snapshot.header.frame_id.empty() ? world_frame_ : snapshot.header.frame_id;
    }

    int32_t prev_task_status = Block::TASK_UNKNOWN;
    {
      std::lock_guard<std::mutex> lock(persistent_world_mutex_);
      const auto it = persistent_world_.find(block_id);
      if (it == persistent_world_.end()) {
        response->success = false;
        response->message = "Unknown block_id: " + block_id;
        RCLCPP_WARN(get_logger(), "SetBlockTaskStatus rejected: %s", response->message.c_str());
        return;
      }

      prev_task_status = it->second.task_status;
      if (!cbpwm::isValidTaskTransition(prev_task_status, target_task_status)) {
        response->success = false;
        response->message =
          std::string("Invalid task transition: ") +
          cbpwm::taskStatusToString(prev_task_status) + " -> " +
          cbpwm::taskStatusToString(target_task_status);
        RCLCPP_WARN(
          get_logger(),
          "SetBlockTaskStatus rejected for block '%s': %s",
          block_id.c_str(),
          response->message.c_str());
        return;
      }

      it->second.task_status = target_task_status;
      // Keep block alive and reflect semantic state change immediately.
      it->second.last_seen = publish_header.stamp;

      // Capture the grasp offset T_tcp_block for FK pose tracking while carrying.
      // Priority: explicit request offset > auto-capture from the block's registered
      // pose + current TCP > nominal fallback. Cleared when leaving TASK_MOVE.
      if (target_task_status == Block::TASK_MOVE) {
        Eigen::Matrix4d offset = Eigen::Matrix4d::Identity();
        const char * offset_src = "nominal_fallback";
        std::string capture_reason;
        if (request->has_grasp_offset) {
          const auto & p = request->grasp_offset.position;
          const auto & o = request->grasp_offset.orientation;
          offset.block<3, 3>(0, 0) =
            Eigen::Quaterniond(o.w, o.x, o.y, o.z).normalized().toRotationMatrix();
          offset.block<3, 1>(0, 3) = Eigen::Vector3d(p.x, p.y, p.z);
          offset_src = "request";
        } else if (captureGraspOffsetFromPose(it->second.pose, offset, capture_reason)) {
          offset_src = "auto_capture";
        } else {
          offset = T_tcp_block_;
          RCLCPP_WARN(
            get_logger(),
            "Grasp offset auto-capture failed for '%s' (%s); using nominal T_tcp_block_.",
            block_id.c_str(),
            capture_reason.c_str());
        }
        task_move_grasp_offsets_[block_id] = offset;
        RCLCPP_INFO(
          get_logger(),
          "Grasp offset for '%s' set from %s: t=[%.3f %.3f %.3f]",
          block_id.c_str(),
          offset_src,
          offset(0, 3),
          offset(1, 3),
          offset(2, 3));
      } else {
        task_move_grasp_offsets_.erase(block_id);
      }
    }

    publishPersistentWorld(publish_header);

    response->success = true;
    response->message =
      std::string("Updated block '") + block_id + "' task_status to " +
      cbpwm::taskStatusToString(target_task_status);
    RCLCPP_INFO(
      get_logger(),
      "SetBlockTaskStatus applied: block '%s' %s -> %s",
      block_id.c_str(),
      cbpwm::taskStatusToString(prev_task_status),
      cbpwm::taskStatusToString(target_task_status));
  }


void PerceptionOrchestratorNode::handleUpsertBlock(
    const std::shared_ptr<UpsertBlockSrv::Request> request,
    std::shared_ptr<UpsertBlockSrv::Response> response)
  {
    if (request->block_id.empty()) {
      response->success = false;
      response->message = "block_id must not be empty.";
      return;
    }
    if (!request->frame_id.empty() && request->frame_id != world_frame_) {
      response->success = false;
      response->message =
        "frame_id '" + request->frame_id + "' != world_frame '" + world_frame_ + "'";
      RCLCPP_WARN(get_logger(), "UpsertBlock rejected: %s", response->message.c_str());
      return;
    }

    Block block;
    block.id = request->block_id;
    block.pose = request->pose;
    block.pose_status = request->pose_status;
    block.task_status = request->task_status;
    block.confidence = request->confidence;
    block.last_seen = now();
    setDefaultPoseCovariance(block);

    {
      std::lock_guard<std::mutex> lock(persistent_world_mutex_);
      // Preserve any goal already assigned to this block: an actual-pose upsert
      // (e.g. placed-from-gripper) must not wipe goal_pose/goal_status, otherwise
      // the goal (target) marker would vanish the moment the block is placed.
      Block & existing = persistent_world_[block.id];
      block.goal_pose = existing.goal_pose;
      block.goal_status = existing.goal_status;
      existing = block;
      seeded_block_ids_.insert(block.id);
    }

    // Update the cached snapshot so markers and queries reflect the change
    std_msgs::msg::Header header;
    header.stamp = now();
    header.frame_id = world_frame_;
    publishPersistentWorld(header);

    response->success = true;
    response->message = "OK";
    RCLCPP_INFO(
      get_logger(),
      "UpsertBlock: '%s' at (%.2f, %.2f, %.2f) in '%s'",
      block.id.c_str(),
      block.pose.position.x,
      block.pose.position.y,
      block.pose.position.z,
      world_frame_.c_str());
  }

void PerceptionOrchestratorNode::handleSetBlockGoal(
    const std::shared_ptr<SetBlockGoalSrv::Request> request,
    std::shared_ptr<SetBlockGoalSrv::Response> response)
  {
    if (request->block_id.empty()) {
      response->success = false;
      response->message = "block_id must not be empty.";
      return;
    }
    if (!request->frame_id.empty() && request->frame_id != world_frame_) {
      response->success = false;
      response->message =
        "frame_id '" + request->frame_id + "' != world_frame '" + world_frame_ + "'";
      RCLCPP_WARN(get_logger(), "SetBlockGoal rejected: %s", response->message.c_str());
      return;
    }

    {
      std::lock_guard<std::mutex> lock(persistent_world_mutex_);
      // Find-or-create: a block may already exist (with a perceived/actual pose)
      // and only gain a goal, or it may be goal-only until perceived/placed.
      Block & block = persistent_world_[request->block_id];
      if (block.id.empty()) {
        block.id = request->block_id;
        block.pose_status = Block::POSE_UNKNOWN;
        block.task_status = Block::TASK_FREE;
        block.confidence = 1.0f;
        setDefaultPoseCovariance(block);
      }
      block.goal_pose = request->goal_pose;
      block.goal_status = Block::GOAL_SET;
      block.last_seen = now();
      seeded_block_ids_.insert(request->block_id);
    }

    std_msgs::msg::Header header;
    header.stamp = now();
    header.frame_id = world_frame_;
    publishPersistentWorld(header);

    response->success = true;
    response->message = "OK";
    RCLCPP_INFO(
      get_logger(),
      "SetBlockGoal: '%s' goal=(%.2f, %.2f, %.2f) in '%s'",
      request->block_id.c_str(),
      request->goal_pose.position.x,
      request->goal_pose.position.y,
      request->goal_pose.position.z,
      world_frame_.c_str());
  }

void PerceptionOrchestratorNode::handleClearBlockGoals(
    const std::shared_ptr<ClearBlockGoalsSrv::Request> request,
    std::shared_ptr<ClearBlockGoalsSrv::Response> response)
  {
    const bool clear_all = request->block_ids.empty();
    const std::unordered_set<std::string> requested_ids(
      request->block_ids.begin(), request->block_ids.end());

    uint32_t cleared_count = 0;
    {
      std::lock_guard<std::mutex> lock(persistent_world_mutex_);
      for (auto it = persistent_world_.begin(); it != persistent_world_.end();) {
        const bool selected =
          clear_all || requested_ids.find(it->first) != requested_ids.end();
        if (!selected || it->second.goal_status != Block::GOAL_SET) {
          ++it;
          continue;
        }

        ++cleared_count;
        if (it->second.pose_status == Block::POSE_UNKNOWN) {
          seeded_block_ids_.erase(it->first);
          it = persistent_world_.erase(it);
          continue;
        }

        it->second.goal_status = Block::GOAL_NONE;
        it->second.goal_pose = geometry_msgs::msg::Pose{};
        it->second.last_seen = now();
        ++it;
      }
    }

    std_msgs::msg::Header header;
    header.stamp = now();
    header.frame_id = world_frame_;
    publishPersistentWorld(header);

    response->success = true;
    response->cleared_count = cleared_count;
    response->message =
      clear_all ?
      "Cleared all block goals." :
      "Cleared selected block goals.";
    RCLCPP_INFO(
      get_logger(),
      "ClearBlockGoals: cleared=%u scope=%s",
      cleared_count,
      clear_all ? "<all>" : "<selected>");
  }

void PerceptionOrchestratorNode::handleClearWorldModel(
    const std::shared_ptr<ClearWorldModelSrv::Request> request,
    std::shared_ptr<ClearWorldModelSrv::Response> response)
  {
    (void)request;

    uint32_t cleared_count = 0;
    {
      std::lock_guard<std::mutex> lock(persistent_world_mutex_);
      cleared_count = static_cast<uint32_t>(persistent_world_.size());
      persistent_world_.clear();
      seeded_block_ids_.clear();
      task_move_grasp_offsets_.clear();
      world_block_counter_ = 0;
    }

    std_msgs::msg::Header header;
    header.stamp = now();
    header.frame_id = world_frame_;
    publishPersistentWorld(header);

    response->success = true;
    response->cleared_count = cleared_count;
    response->message = "Cleared world model.";
    RCLCPP_INFO(get_logger(), "ClearWorldModel: cleared=%u", cleared_count);
  }
