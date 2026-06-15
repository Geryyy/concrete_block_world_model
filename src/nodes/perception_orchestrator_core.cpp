#include "concrete_block_world_model/nodes/perception_orchestrator_node.hpp"

#include "concrete_block_world_model/utils/block_utils.hpp"
#include "concrete_block_world_model/utils/coarse_pose_utils.hpp"
#include "concrete_block_world_model/utils/img_utils.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

#include <cmath>

namespace
{

geometry_msgs::msg::Vector3 toVector3(const std::array<double, 3> & values)
{
  geometry_msgs::msg::Vector3 out;
  out.x = values[0];
  out.y = values[1];
  out.z = values[2];
  return out;
}

double diagonalSigma(
  const Block & block,
  size_t offset,
  double fallback_sigma)
{
  double sum_var = 0.0;
  size_t valid = 0;
  for (size_t i = 0; i < 3; ++i) {
    const double value = block.pose_covariance[(i + offset) * 6 + i + offset];
    if (std::isfinite(value) && value > 0.0) {
      sum_var += value;
      ++valid;
    }
  }
  if (valid == 0U) {
    return fallback_sigma;
  }
  return std::sqrt(sum_var / static_cast<double>(valid));
}

}  // namespace

void PerceptionOrchestratorNode::resetPerfCounters()
  {
    std::lock_guard<std::mutex> lock(perf_mutex_);
    perf_timing_count_ = 0;
    perf_seg_sum_ms_ = 0;
    perf_track_sum_ms_ = 0;
    perf_reg_sum_ms_ = 0;
    perf_total_sum_ms_ = 0;
    dropped_busy_frames_.store(0);
    dropped_sync_frames_.store(0);
  }

std::string PerceptionOrchestratorNode::resolveGraspedBlockId()
  {
    std::lock_guard<std::mutex> lock(persistent_world_mutex_);
    return cbpwm::resolveGraspedBlockId(persistent_world_, *get_clock());
  }

cbpwm::AssociationConfig PerceptionOrchestratorNode::associationConfig() const
  {
    cbpwm::AssociationConfig cfg;
    cfg.association_max_distance_m = runtime_cfg_.association_max_distance_m;
    cfg.association_max_age_s = runtime_cfg_.association_max_age_s;
    cfg.min_update_confidence = runtime_cfg_.min_update_confidence;
    return cfg;
  }

void PerceptionOrchestratorNode::resetBusy()
  {
    busy_.store(false);
  }

void PerceptionOrchestratorNode::updateLatestWorldCache(const BlockArray & out)
  {
    std::lock_guard<std::mutex> lock(latest_world_mutex_);
    latest_world_ = out;
  }

BlockArray PerceptionOrchestratorNode::latestWorldSnapshot()
  {
    BlockArray snapshot;
    {
      std::lock_guard<std::mutex> lock(latest_world_mutex_);
      snapshot = latest_world_;
    }

    if (!continuous_cfg_.filtering_enabled) {
      return snapshot;
    }

    const double now_s = now().seconds();
    std::lock_guard<std::mutex> lock(persistent_world_mutex_);
    for (auto & block : snapshot.blocks) {
      refreshContinuousBlockConfidenceLocked(block, now_s);
    }
    return snapshot;
  }

PlanningScene PerceptionOrchestratorNode::latestPlanningSceneSnapshot()
  {
    PlanningScene scene;
    {
      std::lock_guard<std::mutex> lock(latest_planning_scene_mutex_);
      scene = latest_planning_scene_;
    }

    if (!continuous_cfg_.filtering_enabled) {
      return scene;
    }

    const double now_s = now().seconds();
    std::lock_guard<std::mutex> lock(persistent_world_mutex_);
    for (auto & object : scene.objects) {
      if (object.source_type != PlanningSceneObject::SOURCE_BLOCK) {
        continue;
      }
      const auto track_it = continuous_tracks_.find(object.id);
      if (track_it == continuous_tracks_.end()) {
        continue;
      }
      object.confidence =
        static_cast<float>(
          cbpwm::trackConfidence(track_it->second, continuous_cfg_.filtering, now_s));
    }
    return scene;
  }

std::vector<PlanningSceneObject> PerceptionOrchestratorNode::staticSceneObjectsInWorld() const
  {
    std::vector<PlanningSceneObject> out;
    out.reserve(static_scene_objects_.size());

    for (const auto & object : static_scene_objects_) {
      PlanningSceneObject world_object = object;
      world_object.frame_id = world_frame_;

      if (!tf_buffer_ || object.frame_id.empty() || object.frame_id == world_frame_) {
        out.push_back(std::move(world_object));
        continue;
      }

      geometry_msgs::msg::PoseStamped src_pose;
      src_pose.header.frame_id = object.frame_id;
      src_pose.header.stamp = rclcpp::Time(0, 0, get_clock()->get_clock_type());
      src_pose.pose = object.pose;

      try {
        const auto tf = tf_buffer_->lookupTransform(
          world_frame_,
          object.frame_id,
          tf2::TimePointZero,
          tf2::durationFromSec(0.2));
        geometry_msgs::msg::PoseStamped dst_pose;
        tf2::doTransform(src_pose, dst_pose, tf);
        world_object.pose = dst_pose.pose;
      } catch (const tf2::TransformException & ex) {
        RCLCPP_WARN(
          get_logger(),
          "Could not transform static scene object '%s' from %s to %s: %s",
          object.id.c_str(),
          object.frame_id.c_str(),
          world_frame_.c_str(),
          ex.what());
      }

      out.push_back(std::move(world_object));
    }

    return out;
  }

PlanningScene PerceptionOrchestratorNode::buildPlanningSceneSnapshot(
    const std_msgs::msg::Header & header,
    const std::vector<Block> & blocks)
  {
    PlanningScene scene;
    scene.header = header;
    scene.header.frame_id = world_frame_;

    const auto static_scene_world = staticSceneObjectsInWorld();
    scene.objects.reserve(static_scene_world.size() + blocks.size());
    for (const auto & object : static_scene_world) {
      scene.objects.push_back(object);
    }

    for (const auto & block : blocks) {
      PlanningSceneObject object;
      object.id = block.id;
      object.frame_id = world_frame_;
      object.shape_type = PlanningSceneObject::SHAPE_BOX;
      object.source_type = PlanningSceneObject::SOURCE_BLOCK;
      object.pose = block.pose;
      object.dimensions = toVector3(block_dimensions_m_);
      object.pose_status = block.pose_status;
      object.task_status = block.task_status;
      object.confidence = block.confidence;
      scene.objects.push_back(std::move(object));
    }
    return scene;
  }

void PerceptionOrchestratorNode::recordTiming(int64_t seg_ms, int64_t track_ms, int64_t reg_ms, int64_t total_ms)
  {
    if (!perf_log_timing_enabled_) {
      return;
    }

    std::lock_guard<std::mutex> lock(perf_mutex_);
    perf_timing_count_++;
    perf_seg_sum_ms_ += seg_ms;
    perf_track_sum_ms_ += track_ms;
    perf_reg_sum_ms_ += reg_ms;
    perf_total_sum_ms_ += total_ms;

    const uint64_t n = perf_timing_count_;
    if ((n % static_cast<uint64_t>(perf_log_every_n_frames_)) != 0U) {
      return;
    }

    const double avg_seg = static_cast<double>(perf_seg_sum_ms_) / static_cast<double>(n);
    const double avg_track = static_cast<double>(perf_track_sum_ms_) / static_cast<double>(n);
    const double avg_reg = static_cast<double>(perf_reg_sum_ms_) / static_cast<double>(n);
    const double avg_total = static_cast<double>(perf_total_sum_ms_) / static_cast<double>(n);

    RCLCPP_INFO(
      get_logger(),
      "Timing avg over %llu frames | seg %.1f ms | track %.1f ms | reg %.1f ms | total %.1f ms | dropped busy %llu | dropped sync %llu",
      static_cast<unsigned long long>(n),
      avg_seg,
      avg_track,
      avg_reg,
      avg_total,
      static_cast<unsigned long long>(dropped_busy_frames_.load()),
      static_cast<unsigned long long>(dropped_sync_frames_.load()));
  }

cbpwm::CoarsePoseConfig PerceptionOrchestratorNode::coarsePoseConfig(int min_points_override) const
  {
    cbpwm::CoarsePoseConfig cfg;
    cfg.min_points =
      min_points_override > 0 ? min_points_override : scene_discovery_coarse_fallback_min_points_;
    cfg.square_ratio_thresh = coarse_surface_square_ratio_thresh_;
    cfg.front_center_offset_square_m = coarse_front_center_offset_square_m_;
    cfg.front_center_offset_rect_m = coarse_front_center_offset_rect_m_;
    cfg.min_confidence = std::max(0.3F, static_cast<float>(runtime_cfg_.min_update_confidence));
    return cfg;
  }

bool PerceptionOrchestratorNode::buildCoarseBlockFromMaskAndCloud(
    uint32_t detection_id,
    const sensor_msgs::msg::Image & mask_msg,
    const sensor_msgs::msg::PointCloud2 & cloud_msg,
    const std_msgs::msg::Header & header,
    const Eigen::Vector3d * camera_origin_world,
    Block & out_block,
    std::string & reason) const
  {
    cbpwm::CoarsePoseInput in;
    in.detection_id = detection_id;
    in.header = header;
    in.mask = toCvMono(mask_msg);
    in.camera_origin_world = camera_origin_world;

    return cbpwm::buildCoarseBlockFromOrganizedCloud(
      in, cloud_msg, coarsePoseConfig(), out_block, reason);
  }

bool PerceptionOrchestratorNode::buildCoarseBlockFromCloudCentroid(
    uint32_t detection_id,
    const sensor_msgs::msg::Image & mask_msg,
    const sensor_msgs::msg::PointCloud2 & cutout_cloud_msg,
    const std_msgs::msg::Header & header,
    const Eigen::Vector3d * camera_origin_world,
    Block & out_block,
    std::string & reason,
    int min_points_override) const
  {
    cbpwm::CoarsePoseInput in;
    in.detection_id = detection_id;
    in.header = header;
    in.mask = toCvMono(mask_msg);
    in.camera_origin_world = camera_origin_world;

    return cbpwm::buildCoarseBlockFromCutoutCloud(
      in, cutout_cloud_msg, coarsePoseConfig(min_points_override), out_block, reason);
  }

bool PerceptionOrchestratorNode::extractMaskCutoutSync(
    const sensor_msgs::msg::Image & mask,
    const sensor_msgs::msg::PointCloud2 & cloud,
    double timeout_s,
    sensor_msgs::msg::PointCloud2 & out_cutout_cloud,
    std::string & reason)
  {
    if (!extract_mask_cutout_client_ || !extract_mask_cutout_client_->service_is_ready()) {
      reason = "extract_mask_cutout service unavailable";
      return false;
    }

    auto req = std::make_shared<ExtractMaskCutoutSrv::Request>();
    req->mask = mask;
    req->cloud = cloud;

    auto future = extract_mask_cutout_client_->async_send_request(req);
    const auto ret = future.wait_for(std::chrono::duration<double>(timeout_s));
    if (ret != std::future_status::ready) {
      reason = "extract_mask_cutout service timeout";
      return false;
    }
    const auto res = future.get();
    if (!res) {
      reason = "empty extract_mask_cutout response";
      return false;
    }
    if (!res->success) {
      reason = res->reason.empty() ? "extract_mask_cutout reported success=false" : res->reason;
      return false;
    }
    if (res->cutout_cloud.data.empty()) {
      reason = "extract_mask_cutout returned empty cutout_cloud";
      return false;
    }
    out_cutout_cloud = res->cutout_cloud;
    reason = res->reason.empty() ? "cutout_cloud received from extract_mask_cutout" : res->reason;
    return true;
  }


bool PerceptionOrchestratorNode::lookupFrameOriginInWorld(
    const std_msgs::msg::Header & stamped_header,
    Eigen::Vector3d & origin_world,
    std::string & reason)
  {
    if (!tf_buffer_) {
      reason = "TF buffer unavailable";
      return false;
    }
    std::string source_frame = stamped_header.frame_id;
    if (source_frame.empty()) {
      if (!resolveCameraFrame(stamped_header, source_frame, reason)) {
        return false;
      }
    }
    try {
      const auto tf_w_s = tf_buffer_->lookupTransform(
        world_frame_, source_frame, stamped_header.stamp, tf2::durationFromSec(0.1));
      origin_world =
        Eigen::Vector3d(tf_w_s.transform.translation.x, tf_w_s.transform.translation.y,
        tf_w_s.transform.translation.z);
      return true;
    } catch (const tf2::TransformException & ex) {
      reason = std::string("TF lookup failed for camera origin (") + world_frame_ + " <- " +
        source_frame + "): " + ex.what();
      return false;
    }
  }

bool PerceptionOrchestratorNode::runRegistrationSync(
    uint32_t detection_id,
    const sensor_msgs::msg::Image & mask,
    const sensor_msgs::msg::PointCloud2 & cloud,
    const std_msgs::msg::Header & header,
    double timeout_s,
    Block & out_block,
    std::string & reason,
    const std::string & object_class_override,
    const Block * pose_prior)
  {
    RegisterBlock::Goal goal;
    goal.mask = mask;
    goal.cloud = cloud;
    goal.object_class = object_class_override.empty() ? object_class_ : object_class_override;
    if (pose_prior) {
      goal.use_pose_prior = true;
      goal.prior_pose = pose_prior->pose;
      goal.prior_position_sigma_m =
        static_cast<float>(diagonalSigma(*pose_prior, 0, kCoarsePositionSigmaM));
      goal.prior_orientation_sigma_rad =
        static_cast<float>(diagonalSigma(*pose_prior, 3, kCoarseOrientationSigmaRad));
    }

    std::mutex reg_mutex;
    std::condition_variable reg_cv;
    bool done = false;
    bool goal_rejected = false;
    rclcpp_action::ResultCode result_code = rclcpp_action::ResultCode::UNKNOWN;
    RegisterBlock::Result::SharedPtr action_result;

    rclcpp_action::Client<RegisterBlock>::SendGoalOptions options;
    options.goal_response_callback =
      [&reg_mutex, &reg_cv, &done, &goal_rejected](GoalHandleRegisterBlock::SharedPtr goal_handle)
      {
        if (goal_handle) {
          return;
        }
        std::lock_guard<std::mutex> lock(reg_mutex);
        goal_rejected = true;
        done = true;
        reg_cv.notify_all();
      };
    options.result_callback =
      [&reg_mutex, &reg_cv, &done, &result_code, &action_result](
      const GoalHandleRegisterBlock::WrappedResult & wrapped)
      {
        std::lock_guard<std::mutex> lock(reg_mutex);
        result_code = wrapped.code;
        action_result = wrapped.result;
        done = true;
        reg_cv.notify_all();
      };

    (void)action_client_->async_send_goal(goal, options);

    {
      std::unique_lock<std::mutex> lock(reg_mutex);
      const bool completed = reg_cv.wait_for(
        lock,
        std::chrono::duration<double>(timeout_s),
        [&done]() {return done;});
      if (!completed) {
        reason = "action timeout";
        return false;
      }
      if (goal_rejected) {
        reason = "goal rejected";
        return false;
      }
    }

    if (result_code != rclcpp_action::ResultCode::SUCCEEDED) {
      reason = "result code " + std::to_string(static_cast<int>(result_code));
      return false;
    }

    if (!action_result) {
      reason = "empty result response";
      return false;
    }
    if (!action_result->success) {
      reason = "registration reported success=false";
      return false;
    }
    if (action_result->fitness < runtime_cfg_.min_fitness || action_result->rmse > runtime_cfg_.max_rmse) {
      reason = "thresholds failed (fitness=" + std::to_string(action_result->fitness) +
        ", rmse=" + std::to_string(action_result->rmse) + ")";
      return false;
    }

    out_block.id = detectionBlockId(detection_id);
    out_block.pose = action_result->pose;
    out_block.confidence = action_result->fitness;
    out_block.last_seen = header.stamp;
    out_block.pose_status = Block::POSE_PRECISE;
    out_block.task_status = Block::TASK_FREE;
    setDiagonalPoseCovariance(
      out_block,
      std::max(kPrecisePositionSigmaMinM, 2.0 * action_result->rmse),
      kPreciseOrientationSigmaRad);
    return true;
  }

bool PerceptionOrchestratorNode::runSegmentationSync(
    const sensor_msgs::msg::Image & image,
    double timeout_s,
    SegmentSrv::Response::SharedPtr & out_response,
    std::string & reason)
  {
    if (!segment_client_ || !segment_client_->service_is_ready()) {
      reason = "segmentation service unavailable";
      return false;
    }

    auto seg_req = std::make_shared<SegmentSrv::Request>();
    seg_req->image = image;
    seg_req->return_debug = false;

    auto future = segment_client_->async_send_request(seg_req);
    const auto ret = future.wait_for(std::chrono::duration<double>(timeout_s));
    if (ret != std::future_status::ready) {
      reason = "segmentation timeout";
      return false;
    }

    out_response = future.get();
    if (!out_response) {
      reason = "empty segmentation response";
      return false;
    }
    if (!out_response->success) {
      reason = "segmentation returned success=false";
      return false;
    }

    return true;
  }

bool PerceptionOrchestratorNode::trySceneDiscoveryCoarseFallback(
    uint32_t detection_id,
    const std::string & registration_reason,
    const sensor_msgs::msg::Image & mask,
    const sensor_msgs::msg::PointCloud2 & cloud,
    const cbpwm::SceneFlowRequest & run_request,
    const Eigen::Vector3d * camera_origin_world,
    cbpwm::RegistrationCounters & counters)
  {
    if (run_request.mode != cbpwm::OneShotMode::kSceneDiscovery ||
      !scene_discovery_coarse_fallback_enabled_)
    {
      return false;
    }

    Block coarse_block;
    std::string coarse_reason;
    const bool coarse_ok = buildCoarseBlockFromMaskAndCloud(
      detection_id,
      mask,
      cloud,
      cloud.header,
      camera_origin_world,
      coarse_block,
      coarse_reason);
    if (!coarse_ok) {
      sensor_msgs::msg::PointCloud2 cutout_cloud;
      std::string cutout_reason;
      const bool got_cutout = extractMaskCutoutSync(
        mask, cloud, run_request.registration_timeout_s, cutout_cloud, cutout_reason);
      if (!got_cutout) {
        RCLCPP_WARN(
          get_logger(),
          "Registration rejected for block_%u (%s). Coarse fallback unavailable: %s",
          detection_id,
          registration_reason.c_str(),
          cutout_reason.c_str());
        return false;
      }

      const bool coarse_from_cutout_ok = buildCoarseBlockFromCloudCentroid(
        detection_id,
        mask,
        cutout_cloud,
        cutout_cloud.header,
        camera_origin_world,
        coarse_block,
        coarse_reason);
      if (!coarse_from_cutout_ok) {
        RCLCPP_WARN(
          get_logger(),
          "Registration rejected for block_%u (%s). Coarse fallback unavailable: %s",
          detection_id,
          registration_reason.c_str(),
          coarse_reason.c_str());
        return false;
      }
    }

    std::lock_guard<std::mutex> lock(persistent_world_mutex_);
    std::string assigned_id;
    std::string upsert_reason;
    const bool upsert_ok = cbpwm::upsertRegisteredBlock(
      persistent_world_,
      world_block_counter_,
      coarse_block,
      run_request.mode,
      run_request.target_block_id,
      cloud.header,
      *get_clock(),
      associationConfig(),
      assigned_id,
      upsert_reason);
    if (!upsert_ok) {
      RCLCPP_WARN(
        get_logger(),
        "Registration rejected for block_%u (%s). Coarse fallback rejected: %s",
        detection_id,
        registration_reason.c_str(),
        upsert_reason.c_str());
      return false;
    }

    ++counters.coarse_upserts_ok;
    RCLCPP_WARN(
      get_logger(),
      "Registration rejected for block_%u (%s). Coarse fallback accepted: incoming=%s assigned=%s",
      detection_id,
      registration_reason.c_str(),
      coarse_block.id.c_str(),
      assigned_id.c_str());
    return true;
  }
