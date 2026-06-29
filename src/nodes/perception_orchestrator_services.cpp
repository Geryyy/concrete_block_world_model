#include "concrete_block_world_model/nodes/perception_orchestrator_node.hpp"

#include "concrete_block_world_model/utils/block_utils.hpp"
#include "concrete_block_world_model/utils/world_model_utils.hpp"

#include <unordered_set>

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

      // Track the captured grasp offset for FK pose tracking while carrying;
      // clear it once the block is no longer being moved.
      if (target_task_status == Block::TASK_MOVE && request->has_grasp_offset) {
        const auto & p = request->grasp_offset.position;
        const auto & o = request->grasp_offset.orientation;
        Eigen::Matrix4d m = Eigen::Matrix4d::Identity();
        m.block<3, 3>(0, 0) =
          Eigen::Quaterniond(o.w, o.x, o.y, o.z).normalized().toRotationMatrix();
        m.block<3, 1>(0, 3) = Eigen::Vector3d(p.x, p.y, p.z);
        task_move_grasp_offsets_[block_id] = m;
      } else if (target_task_status != Block::TASK_MOVE) {
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

void PerceptionOrchestratorNode::handleSetPerceptionMode(
    const std::shared_ptr<SetPerceptionModeSrv::Request> request,
    std::shared_ptr<SetPerceptionModeSrv::Response> response)
  {
    const std::string mode = cbpwm::normalizeMode(request->mode);
    if (mode == "CONTINUOUS") {
      perception_mode_.store(PerceptionMode::kContinuous);
      debug_detection_overlay_enabled_ = request->enable_debug;
      response->success = true;
      response->message = "Perception mode set to CONTINUOUS.";
      RCLCPP_INFO(
        get_logger(),
        "SetPerceptionMode: CONTINUOUS target=%s debug=%s",
        request->target_block_id.empty() ? "<all>" : request->target_block_id.c_str(),
        request->enable_debug ? "true" : "false");
      return;
    }

    if (mode == "IDLE" || mode == "ON_DEMAND") {
      perception_mode_.store(PerceptionMode::kIdle);
      response->success = true;
      response->message = "Perception mode set to IDLE. One-shot run_pose_estimation remains available.";
      RCLCPP_INFO(get_logger(), "SetPerceptionMode: IDLE");
      return;
    }

    response->success = false;
    response->message = "Unsupported perception mode: " + request->mode +
      " (expected IDLE, ON_DEMAND, or CONTINUOUS).";
    RCLCPP_WARN(get_logger(), "SetPerceptionMode rejected: %s", response->message.c_str());
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
          continuous_tracks_.erase(it->first);
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
