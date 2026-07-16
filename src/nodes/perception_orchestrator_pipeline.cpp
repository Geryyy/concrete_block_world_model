#include "concrete_block_world_model/nodes/perception_orchestrator_node.hpp"

#include "concrete_block_world_model/utils/block_utils.hpp"
#include "concrete_block_world_model/utils/img_utils.hpp"
#include "concrete_block_world_model/utils/world_model_utils.hpp"
#include "concrete_block_world_model/world_model/scene_discovery_flow.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

namespace
{

std::string yamlEscape(const std::string & value)
{
  std::string out;
  out.reserve(value.size());
  for (const char c : value) {
    if (c == '\\' || c == '"') {
      out.push_back('\\');
    }
    out.push_back(c);
  }
  return out;
}

std::string stampBaseName(const builtin_interfaces::msg::Time & stamp, uint64_t sequence)
{
  std::ostringstream ss;
  ss << stamp.sec << "_"
     << std::setw(9) << std::setfill('0') << stamp.nanosec
     << "_seq" << sequence;
  return ss.str();
}

bool writeImagePng(
  const std::filesystem::path & path,
  const sensor_msgs::msg::Image & image,
  const rclcpp::Logger & logger,
  const std::string & label)
{
  try {
    auto cv_ptr = cv_bridge::toCvCopy(image);
    cv::Mat out = cv_ptr->image;
    const std::string & encoding = cv_ptr->encoding;
    if (encoding == "rgb8") {
      cv::cvtColor(out, out, cv::COLOR_RGB2BGR);
    } else if (encoding == "rgba8") {
      cv::cvtColor(out, out, cv::COLOR_RGBA2BGRA);
    }
    if (!cv::imwrite(path.string(), out)) {
      RCLCPP_WARN(logger, "Scene-discovery dump: failed to write %s image to %s",
        label.c_str(), path.c_str());
      return false;
    }
    return true;
  } catch (const std::exception & exc) {
    RCLCPP_WARN(logger, "Scene-discovery dump: failed to write %s image: %s",
      label.c_str(), exc.what());
    return false;
  }
}

bool writeMaskPng(
  const std::filesystem::path & path,
  const sensor_msgs::msg::Image & mask,
  const rclcpp::Logger & logger)
{
  try {
    auto cv_ptr = cv_bridge::toCvCopy(mask, "mono8");
    if (!cv::imwrite(path.string(), cv_ptr->image)) {
      RCLCPP_WARN(logger, "Scene-discovery dump: failed to write mask to %s", path.c_str());
      return false;
    }
    return true;
  } catch (const std::exception & exc) {
    RCLCPP_WARN(logger, "Scene-discovery dump: failed to write mask: %s", exc.what());
    return false;
  }
}

bool writeCloudPcd(
  const std::filesystem::path & path,
  const sensor_msgs::msg::PointCloud2 & cloud,
  const rclcpp::Logger & logger)
{
  try {
    sensor_msgs::PointCloud2ConstIterator<float> iter_x(cloud, "x");
    sensor_msgs::PointCloud2ConstIterator<float> iter_y(cloud, "y");
    sensor_msgs::PointCloud2ConstIterator<float> iter_z(cloud, "z");

    std::vector<std::array<float, 3>> points;
    points.reserve(static_cast<size_t>(cloud.width) * static_cast<size_t>(cloud.height));
    for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
      if (std::isfinite(*iter_x) && std::isfinite(*iter_y) && std::isfinite(*iter_z)) {
        points.push_back({*iter_x, *iter_y, *iter_z});
      }
    }

    std::ofstream out(path);
    if (!out.is_open()) {
      RCLCPP_WARN(logger, "Scene-discovery dump: failed to open cloud path %s", path.c_str());
      return false;
    }

    out << "# .PCD v0.7 - Point Cloud Data file format\n"
        << "VERSION 0.7\n"
        << "FIELDS x y z\n"
        << "SIZE 4 4 4\n"
        << "TYPE F F F\n"
        << "COUNT 1 1 1\n"
        << "WIDTH " << points.size() << "\n"
        << "HEIGHT 1\n"
        << "VIEWPOINT 0 0 0 1 0 0 0\n"
        << "POINTS " << points.size() << "\n"
        << "DATA ascii\n";
    out << std::setprecision(9);
    for (const auto & p : points) {
      out << p[0] << " " << p[1] << " " << p[2] << "\n";
    }
    return true;
  } catch (const std::exception & exc) {
    RCLCPP_WARN(logger, "Scene-discovery dump: failed to write cloud: %s", exc.what());
    return false;
  }
}

}  // namespace

void PerceptionOrchestratorNode::publishWorldMarkers(
  const std_msgs::msg::Header & header,
  const std::vector<Block> & blocks)
{
  auto marker_header = header;
  marker_header.stamp = rclcpp::Time(0, 0, get_clock()->get_clock_type());
  const auto static_scene_world = staticSceneObjectsInWorld();
  const auto markers = cbpwm::buildWorldMarkers(
    marker_header, blocks, static_scene_world, world_frame_, block_dimensions_m_);
  marker_pub_->publish(markers);

  const auto goal_markers = cbpwm::buildGoalMarkers(
    marker_header, blocks, world_frame_, block_dimensions_m_);
  goal_marker_pub_->publish(goal_markers);

  RCLCPP_INFO_THROTTLE(
    get_logger(), *get_clock(), 10000,
    "Published marker array: markers=%zu blocks=%zu static_objects=%zu frame=%s",
    markers.markers.size(),
    blocks.size(),
    static_scene_world.size(),
    world_frame_.c_str());

  if (blocks.size() != last_published_block_count_) {
    last_published_block_count_ = blocks.size();
    if (!blocks.empty()) {
      const auto & b0 = blocks.front();
      RCLCPP_INFO(
        get_logger(),
        "Published markers: %zu blocks + %zu static objects in frame '%s' (first block: id=%s pos=[%.3f, %.3f, %.3f])",
        blocks.size(),
        static_scene_world.size(),
        world_frame_.c_str(),
        b0.id.c_str(),
        b0.pose.position.x,
        b0.pose.position.y,
        b0.pose.position.z);
    } else if (!static_scene_objects_.empty()) {
      RCLCPP_INFO(
        get_logger(),
        "Published markers: 0 blocks + %zu static objects in frame '%s'",
        static_scene_world.size(),
        world_frame_.c_str());
    }
  }
}

void PerceptionOrchestratorNode::publishPersistentWorld(const std_msgs::msg::Header & header)
{
  updateTaskMoveBlocksFromFk(header);

  BlockArray out;
  out.header = header;

  // Single-shot world model: blocks persist until explicitly cleared
  // (clear_world_model / clear_block_goals) or overwritten by a new observation.
  // There is no staleness timeout -- nothing periodically re-observes blocks, so
  // an age-based eviction would delete exactly the results the caller wants to keep.
  {
    std::lock_guard<std::mutex> lock(persistent_world_mutex_);
    out.blocks.reserve(persistent_world_.size());
    for (const auto & kv : persistent_world_) {
      out.blocks.push_back(kv.second);
    }
  }

  world_pub_->publish(out);
  RCLCPP_INFO_THROTTLE(
    get_logger(), *get_clock(), 10000,
    "Published world model: blocks=%zu frame=%s",
    out.blocks.size(),
    out.header.frame_id.c_str());
  updateLatestWorldCache(out);
  {
    std::lock_guard<std::mutex> lock(latest_planning_scene_mutex_);
    latest_planning_scene_ = buildPlanningSceneSnapshot(out.header, out.blocks);
  }
  publishWorldMarkers(out.header, out.blocks);
}

void PerceptionOrchestratorNode::updateTaskMoveBlocksFromFk(const std_msgs::msg::Header & header)
{
  if (!task_move_fk_tracking_enabled_) {
    return;
  }

  std::vector<std::string> task_move_ids;
  {
    std::lock_guard<std::mutex> lock(persistent_world_mutex_);
    for (const auto & kv : persistent_world_) {
      if (kv.second.task_status == Block::TASK_MOVE) {
        task_move_ids.push_back(kv.first);
      }
    }
  }

  if (task_move_ids.empty()) {
    return;
  }

  Eigen::Matrix4d T_world_tcp;
  std::string reason;
  if (!lookupTcpInWorld(header, T_world_tcp, reason)) {
    RCLCPP_WARN_THROTTLE(
      get_logger(), *get_clock(), 1000,
      "TASK_MOVE FK tracking skipped for %zu block(s): %s",
      task_move_ids.size(),
      reason.c_str());
    return;
  }

  std::lock_guard<std::mutex> lock(persistent_world_mutex_);
  for (const auto & id : task_move_ids) {
    auto it = persistent_world_.find(id);
    if (it == persistent_world_.end() || it->second.task_status != Block::TASK_MOVE) {
      continue;
    }

    // Use the per-block captured grasp offset if available, else the nominal.
    const auto off_it = task_move_grasp_offsets_.find(id);
    const Eigen::Matrix4d T_tcp_block =
      (off_it != task_move_grasp_offsets_.end()) ? off_it->second : T_tcp_block_;
    const Eigen::Matrix4d T_world_block = T_world_tcp * T_tcp_block;
    const Eigen::Vector3d p_world = T_world_block.block<3, 1>(0, 3);
    const Eigen::Quaterniond q_world =
      Eigen::Quaterniond(T_world_block.block<3, 3>(0, 0)).normalized();
    geometry_msgs::msg::Pose fk_pose;
    fk_pose.position.x = p_world.x();
    fk_pose.position.y = p_world.y();
    fk_pose.position.z = p_world.z();
    fk_pose.orientation.x = q_world.x();
    fk_pose.orientation.y = q_world.y();
    fk_pose.orientation.z = q_world.z();
    fk_pose.orientation.w = q_world.w();

    it->second.pose = fk_pose;
    it->second.pose_status = Block::POSE_PRECISE;
    it->second.confidence = 1.0f;
    it->second.last_seen = header.stamp;
    setDiagonalPoseCovariance(
      it->second,
      kPrecisePositionSigmaMinM,
      kPreciseOrientationSigmaRad);
  }
}

void PerceptionOrchestratorNode::publishDetectionOverlay(
  const sensor_msgs::msg::Image::ConstSharedPtr & image,
  const vision_msgs::msg::Detection2DArray & detections,
  const sensor_msgs::msg::Image & mask_msg)
{
  cv::Mat img = toCvBgr(*image);

  if (!mask_msg.data.empty()) {
    cv::Mat mask = toCvMono(mask_msg);
    overlayMask(img, mask, cv::Scalar(255, 255, 0), 0.35);
  }

  drawDetectionBoxes(img, detections, cv::Scalar(0, 255, 0));
  auto out = cv_bridge::CvImage(image->header, "bgr8", img).toImageMsg();
  det_debug_pub_->publish(*out);
}


auto PerceptionOrchestratorNode::makeSegmentationRequest(
  const sensor_msgs::msg::Image & image) const
  -> std::shared_ptr<SegmentSrv::Request>
{
  auto seg_req = std::make_shared<SegmentSrv::Request>();
  seg_req->image = image;
  // Only pay for the annotated debug render when we are going to publish it.
  seg_req->return_debug = debug_detection_overlay_enabled_.load();
  return seg_req;
}


void PerceptionOrchestratorNode::publishYoloServiceDebugImage(
  const sensor_msgs::msg::Image & debug_image)
{
  if (debug_detection_overlay_enabled_.load() && yolo_service_debug_pub_ &&
    !debug_image.data.empty())
  {
    yolo_service_debug_pub_->publish(debug_image);
  }
}

std::filesystem::path PerceptionOrchestratorNode::createSceneDiscoveryDump(
  const sensor_msgs::msg::Image & image,
  const sensor_msgs::msg::PointCloud2 & cloud,
  const std::vector<cbpwm::DetectionCandidate> & candidates,
  const SegmentSrv::Response & seg_res,
  const OneShotRequest & run_request)
{
  if (!debug_scene_discovery_dump_enabled_ ||
    run_request.mode != cbpwm::OneShotMode::kSceneDiscovery)
  {
    return {};
  }

  const auto dump_dir =
    debug_scene_discovery_dump_dir_ / stampBaseName(cloud.header.stamp, run_request.sequence);
  const auto masks_dir = dump_dir / "masks";
  try {
    std::filesystem::create_directories(masks_dir);
  } catch (const std::exception & exc) {
    RCLCPP_WARN(
      get_logger(),
      "Scene-discovery dump: failed to create %s: %s",
      dump_dir.c_str(),
      exc.what());
    return {};
  }

  writeImagePng(dump_dir / "rgb.png", image, get_logger(), "rgb");
  writeCloudPcd(dump_dir / "cloud.pcd", cloud, get_logger());
  if (!seg_res.mask.data.empty()) {
    writeMaskPng(dump_dir / "segmentation_mask.png", seg_res.mask, get_logger());
  }

  for (const auto & candidate : candidates) {
    std::ostringstream name;
    name << "candidate_" << std::setw(3) << std::setfill('0') << candidate.first << "_mask.png";
    writeMaskPng(masks_dir / name.str(), candidate.second, get_logger());
  }

  std::ofstream meta(dump_dir / "metadata.yaml");
  if (meta.is_open()) {
    meta << "mode: \"" << cbpwm::oneShotModeToString(run_request.mode) << "\"\n";
    meta << "sequence: " << run_request.sequence << "\n";
    meta << "target_block_id: \"" << yamlEscape(run_request.target_block_id) << "\"\n";
    meta << "world_frame: \"" << yamlEscape(world_frame_) << "\"\n";
    meta << "image:\n";
    meta << "  frame_id: \"" << yamlEscape(image.header.frame_id) << "\"\n";
    meta << "  stamp:\n";
    meta << "    sec: " << image.header.stamp.sec << "\n";
    meta << "    nanosec: " << image.header.stamp.nanosec << "\n";
    meta << "  width: " << image.width << "\n";
    meta << "  height: " << image.height << "\n";
    meta << "  encoding: \"" << yamlEscape(image.encoding) << "\"\n";
    meta << "  file: rgb.png\n";
    meta << "cloud:\n";
    meta << "  frame_id: \"" << yamlEscape(cloud.header.frame_id) << "\"\n";
    meta << "  stamp:\n";
    meta << "    sec: " << cloud.header.stamp.sec << "\n";
    meta << "    nanosec: " << cloud.header.stamp.nanosec << "\n";
    meta << "  width: " << cloud.width << "\n";
    meta << "  height: " << cloud.height << "\n";
    meta << "  file: cloud.pcd\n";
    meta << "detections: " << seg_res.detections.detections.size() << "\n";
    meta << "registration_candidates: " << candidates.size() << "\n";
    meta << "candidate_masks:\n";
    for (const auto & candidate : candidates) {
      meta << "  - detection_id: " << candidate.first << "\n";
      meta << "    file: masks/candidate_"
           << std::setw(3) << std::setfill('0') << candidate.first << "_mask.png\n";
    }
  }

  RCLCPP_INFO(get_logger(), "Scene-discovery dump started: %s", dump_dir.c_str());
  return dump_dir;
}

void PerceptionOrchestratorNode::writeSceneDiscoveryDumpSummary(
  const std::filesystem::path & dump_dir,
  const std::vector<std::string> & registration_records,
  const cbpwm::RegistrationCounters & counters)
{
  if (dump_dir.empty()) {
    return;
  }

  std::ofstream out(dump_dir / "registrations.yaml");
  if (!out.is_open()) {
    RCLCPP_WARN(
      get_logger(),
      "Scene-discovery dump: failed to write registrations.yaml in %s",
      dump_dir.c_str());
    return;
  }

  out << "registrations_ok: " << counters.registrations_ok << "\n";
  out << "coarse_upserts_ok: " << counters.coarse_upserts_ok << "\n";
  out << "records:\n";
  for (const auto & record : registration_records) {
    out << record;
  }
}

std::string PerceptionOrchestratorNode::formatSceneDiscoveryDumpRecord(
  uint32_t detection_id,
  bool registration_ok,
  const Block & block,
  bool upsert_ok,
  const std::string & assigned_id,
  const std::string & reason) const
{
  std::ostringstream out;
  out << "  - detection_id: " << detection_id << "\n";
  out << "    registration_ok: " << (registration_ok ? "true" : "false") << "\n";
  out << "    upsert_ok: " << (upsert_ok ? "true" : "false") << "\n";
  out << "    assigned_id: \"" << yamlEscape(assigned_id) << "\"\n";
  out << "    reason: \"" << yamlEscape(reason) << "\"\n";
  if (registration_ok) {
    out << "    incoming_id: \"" << yamlEscape(block.id) << "\"\n";
    out << "    confidence: " << block.confidence << "\n";
    out << "    pose_status: " << block.pose_status << "\n";
    out << "    task_status: " << block.task_status << "\n";
    out << "    pose:\n";
    out << "      frame_id: \"" << yamlEscape(world_frame_) << "\"\n";
    out << "      position: ["
        << block.pose.position.x << ", "
        << block.pose.position.y << ", "
        << block.pose.position.z << "]\n";
    out << "      orientation_xyzw: ["
        << block.pose.orientation.x << ", "
        << block.pose.orientation.y << ", "
        << block.pose.orientation.z << ", "
        << block.pose.orientation.w << "]\n";
  }
  return out.str();
}


void PerceptionOrchestratorNode::syncCallback(
  const sensor_msgs::msg::Image::ConstSharedPtr image,
  const sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud)
{
  {
    // Single-shot only: a synced frame pair is processed exclusively to serve an
    // active run_pose_estimation request. When idle we drop the pair immediately.
    std::lock_guard<std::mutex> lock(one_shot_mutex_);
    if (active_one_shot_.mode == cbpwm::OneShotMode::kNone) {
      return;
    }
  }

  bool expected = false;
  if (!busy_.compare_exchange_strong(expected, true)) {
    dropped_busy_frames_.fetch_add(1);
    return;
  }

  const rclcpp::Time t_img(image->header.stamp);
  const rclcpp::Time t_cloud(cloud->header.stamp);
  const double dt = std::abs((t_img - t_cloud).seconds());

  if (dt > runtime_cfg_.max_sync_delta_s) {
    dropped_sync_frames_.fetch_add(1);
    RCLCPP_WARN(
      get_logger(),
      "Dropped frame pair due to sync delta %.4f s (> %.4f s)",
      dt,
      runtime_cfg_.max_sync_delta_s);
    resetBusy();
    return;
  }

  processFrame(image, cloud);
}

void PerceptionOrchestratorNode::handleOneShotSegmentationResponse(
  rclcpp::Client<SegmentSrv>::SharedFuture seg_future,
  const sensor_msgs::msg::Image::ConstSharedPtr & image,
  const sensor_msgs::msg::PointCloud2::ConstSharedPtr & cloud,
  std::chrono::steady_clock::time_point t_start,
  const OneShotRequest & run_request)
{
  try {
    auto seg_res = seg_future.get();
    const auto t_after_seg = std::chrono::steady_clock::now();
    const size_t seg_detection_count =
      seg_res ? seg_res->detections.detections.size() : 0U;

    if (!seg_res || !seg_res->success) {
      publishPersistentWorld(cloud->header);
      completeOneShotRequest(run_request.sequence, false, "Segmentation failed.");
      resetBusy();
      return;
    }

    RCLCPP_INFO(
      get_logger(),
      "One-shot %s: segmentation detections=%zu",
      cbpwm::oneShotModeToString(run_request.mode),
      seg_detection_count);
    for (size_t i = 0; i < seg_detection_count; ++i) {
      const auto & det = seg_res->detections.detections[i];
      const std::string class_id =
        det.results.empty() ? std::string{} : det.results.front().hypothesis.class_id;
      RCLCPP_INFO(
        get_logger(),
        "One-shot %s detection: idx=%zu class=%s confidence=%.3f bbox[cx=%.1f cy=%.1f w=%.1f h=%.1f]",
        cbpwm::oneShotModeToString(run_request.mode),
        i,
        class_id.empty() ? "<unknown>" : class_id.c_str(),
        cbpwm::detectionConfidence(det),
        det.bbox.center.position.x,
        det.bbox.center.position.y,
        det.bbox.size_x,
        det.bbox.size_y);
    }

    // Collapse strongly-overlapping detections (e.g. a nested duplicate box on the same
    // block) into one, taking the union bbox so the cutout keeps every face. Done before
    // the overlay so the debug image shows the merged result the registration will use.
    if (scene_discovery_merge_enabled_) {
      const size_t before = seg_res->detections.detections.size();
      seg_res->detections = cbpwm::mergeOverlappingDetections(
        seg_res->detections,
        scene_discovery_merge_containment_ratio_,
        scene_discovery_merge_iou_threshold_);
      const size_t after = seg_res->detections.detections.size();
      if (after != before) {
        RCLCPP_INFO(
          get_logger(),
          "One-shot %s: overlap-merged detections %zu -> %zu (containment>=%.2f or IoU>=%.2f)",
          cbpwm::oneShotModeToString(run_request.mode),
          before,
          after,
          scene_discovery_merge_containment_ratio_,
          scene_discovery_merge_iou_threshold_);
      }
    }

    if (debug_detection_overlay_enabled_.load() && det_debug_pub_) {
      publishDetectionOverlay(image, seg_res->detections, seg_res->mask);
    }
    publishYoloServiceDebugImage(seg_res->debug_image);

    const auto t_after_track = t_after_seg;
    const size_t tracked_count = seg_res->detections.detections.size();
    RCLCPP_INFO(
      get_logger(),
      "One-shot %s: detections=%zu tracked=%zu (overlap-merged)",
      cbpwm::oneShotModeToString(run_request.mode),
      seg_detection_count,
      tracked_count);

    if (seg_res->detections.detections.empty()) {
      const auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        t_after_seg - t_start).count();
      recordTiming(total_ms, 0, 0, total_ms);
      publishPersistentWorld(cloud->header);

      if (run_request.mode == cbpwm::OneShotMode::kSceneDiscovery) {
        completeOneShotRequest(
          run_request.sequence,
          true,
          "Scene discovery finished (detections=0, tracked=0, registrations=0).");
      } else {
        completeOneShotRequest(
          run_request.sequence,
          false,
          "Requested block was not detected.");
      }
      resetBusy();
      return;
    }

    auto candidates = cbpwm::buildRegistrationCandidates(
      *seg_res, run_request.mode, run_request.target_block_id, get_logger());
    const auto scene_dump_dir =
      createSceneDiscoveryDump(*image, *cloud, candidates, *seg_res, run_request);
    std::vector<std::string> scene_dump_registration_records;

    const size_t registration_candidates = candidates.size();
    RCLCPP_INFO(
      get_logger(),
      "One-shot %s: detections=%zu tracked=%zu registration_candidates=%zu",
      cbpwm::oneShotModeToString(run_request.mode),
      seg_detection_count,
      tracked_count,
      registration_candidates);

    if (registration_candidates == 0) {
      const auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        t_after_track - t_start).count();
      recordTiming(total_ms, 0, 0, total_ms);
      publishPersistentWorld(cloud->header);
      cbpwm::RegistrationCounters counters;
      writeSceneDiscoveryDumpSummary(scene_dump_dir, {}, counters);

      if (run_request.mode == cbpwm::OneShotMode::kSceneDiscovery) {
        completeOneShotRequest(
          run_request.sequence,
          true,
          "Scene discovery finished (detections=" + std::to_string(seg_detection_count) +
          ", tracked=" + std::to_string(tracked_count) +
          ", registration_candidates=0, registrations=0).");
      } else {
        completeOneShotRequest(
          run_request.sequence,
          false,
          "Requested block was not available for registration "
          "(detections=" + std::to_string(seg_detection_count) +
          ", tracked=" + std::to_string(tracked_count) + ").");
      }

      resetBusy();
      return;
    }

    if (!action_client_->action_server_is_ready()) {
      RCLCPP_WARN(get_logger(), "Registration action unavailable.");
      completeOneShotRequest(run_request.sequence, false, "Registration action unavailable.");
      resetBusy();
      return;
    }

    const auto t_reg_start = std::chrono::steady_clock::now();
    cbpwm::RegistrationCounters counters;
    Eigen::Vector3d camera_origin_world = Eigen::Vector3d::Zero();
    std::string camera_origin_reason;
    const bool have_camera_origin = lookupFrameOriginInWorld(
      cloud->header, camera_origin_world, camera_origin_reason);
    if (!have_camera_origin) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 3000,
        "Scene-discovery coarse center offset disabled for this frame: %s",
        camera_origin_reason.c_str());
    }
    cbpwm::SceneFlowRequest scene_req;
    scene_req.mode = run_request.mode;
    scene_req.target_block_id = run_request.target_block_id;
    scene_req.registration_timeout_s = run_request.registration_timeout_s;
    scene_req.refine_target_max_distance_m = runtime_cfg_.refine_target_max_distance_m;

    cbpwm::SceneFlowRuntime scene_rt;
    scene_rt.logger = get_logger();
    scene_rt.run_registration =
      [this](
      uint32_t detection_id,
      const sensor_msgs::msg::Image & mask,
      const sensor_msgs::msg::PointCloud2 & in_cloud,
      const std_msgs::msg::Header & header,
      double timeout_s,
      Block & out_block,
      std::string & out_reason) {
        return runRegistrationSync(
          detection_id,
          mask,
          in_cloud,
          header,
          timeout_s,
          out_block,
          out_reason);
      };
    scene_rt.upsert_block =
      [this, &run_request, cloud](Block & block, std::string & assigned_id, std::string & reason) {
        std::lock_guard<std::mutex> lock(persistent_world_mutex_);
        return cbpwm::upsertRegisteredBlock(
          persistent_world_,
          world_block_counter_,
          block,
          run_request.mode,
          run_request.target_block_id,
          cloud->header,
          *get_clock(),
          associationConfig(),
          assigned_id,
          reason);
      };
    scene_rt.get_expected_target =
      [this](const std::string & target_id, Block & out_target) {
        std::lock_guard<std::mutex> lock(persistent_world_mutex_);
        const auto it = persistent_world_.find(target_id);
        if (it == persistent_world_.end()) {
          return false;
        }
        out_target = it->second;
        return true;
      };
    scene_rt.try_coarse_fallback =
      [this, have_camera_origin, &camera_origin_world](
      uint32_t detection_id,
      const std::string & registration_reason,
      const sensor_msgs::msg::Image & mask,
      const sensor_msgs::msg::PointCloud2 & in_cloud,
      const cbpwm::SceneFlowRequest & request,
      cbpwm::RegistrationCounters & out_counters) {
        return trySceneDiscoveryCoarseFallback(
          detection_id,
          registration_reason,
          mask,
          in_cloud,
          request,
          have_camera_origin ? &camera_origin_world : nullptr,
          out_counters);
      };
    scene_rt.on_registration_result =
      [this, &scene_dump_dir, &scene_dump_registration_records](
      uint32_t detection_id,
      bool registration_ok,
      const Block & block,
      bool upsert_ok,
      const std::string & assigned_id,
      const std::string & reason) {
        if (scene_dump_dir.empty()) {
          return;
        }
        scene_dump_registration_records.push_back(
          formatSceneDiscoveryDumpRecord(
            detection_id,
            registration_ok,
            block,
            upsert_ok,
            assigned_id,
            reason));
      };

    cbpwm::processRegistrationCandidates(
      candidates,
      *cloud,
      scene_req,
      scene_rt,
      counters);
    writeSceneDiscoveryDumpSummary(scene_dump_dir, scene_dump_registration_records, counters);

    const auto t_reg_end = std::chrono::steady_clock::now();
    const auto seg_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      t_after_seg - t_start).count();
    const auto track_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      t_after_track - t_after_seg).count();
    const auto reg_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      t_reg_end - t_reg_start).count();
    const auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      t_reg_end - t_start).count();
    recordTiming(seg_ms, track_ms, reg_ms, total_ms);

    publishPersistentWorld(cloud->header);
    completeOneShotRequest(
      run_request.sequence,
      counters.registrations_ok > 0 || run_request.mode == cbpwm::OneShotMode::kSceneDiscovery,
      "Pose estimation finished (detections=" + std::to_string(seg_detection_count) +
      ", tracked=" + std::to_string(tracked_count) +
      ", registration_candidates=" + std::to_string(registration_candidates) +
      ", registrations=" + std::to_string(counters.registrations_ok) +
      ", coarse_upserts=" + std::to_string(counters.coarse_upserts_ok) + ").");

    resetBusy();
  } catch (const std::exception & e) {
    RCLCPP_ERROR(get_logger(), "Segmentation stage failed: %s", e.what());
    completeOneShotRequest(run_request.sequence, false, "Segmentation stage exception.");
    resetBusy();
  }
}

void PerceptionOrchestratorNode::processFrame(
  const sensor_msgs::msg::Image::ConstSharedPtr & image,
  const sensor_msgs::msg::PointCloud2::ConstSharedPtr & cloud)
{
  const auto t_start = std::chrono::steady_clock::now();

  OneShotRequest run_request;
  {
    std::lock_guard<std::mutex> lock(one_shot_mutex_);
    run_request = active_one_shot_;
  }

  if (run_request.mode == cbpwm::OneShotMode::kNone) {
    resetBusy();
    return;
  }

  if (run_request.mode == cbpwm::OneShotMode::kRefineGrasped && refine_grasped_use_fk_roi_) {
    processRefineGraspedWithFkRoi(image, cloud, run_request, t_start);
    return;
  }
  if (run_request.mode == cbpwm::OneShotMode::kRefineBlock &&
    tryProcessRefineBlockWithPoseRoi(image, cloud, run_request, t_start))
  {
    return;
  }

  if (!segment_client_->service_is_ready()) {
    RCLCPP_WARN(get_logger(), "Segmentation service unavailable.");
    resetBusy();
    return;
  }

  auto seg_req = makeSegmentationRequest(*image);

  segment_client_->async_send_request(
    seg_req,
    [this, image, cloud, t_start, run_request](
      rclcpp::Client<SegmentSrv>::SharedFuture seg_future) {
      handleOneShotSegmentationResponse(seg_future, image, cloud, t_start, run_request);
    });
}
