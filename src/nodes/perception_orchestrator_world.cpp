#include "concrete_block_world_model/nodes/perception_orchestrator_node.hpp"

#include "concrete_block_world_model/utils/block_utils.hpp"
#include "concrete_block_world_model/utils/world_model_utils.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2/time.h>

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

// Frame ids may contain '/' (e.g. "elastic/K8_tool_center_point"); sanitize for the
// cosmetic "T_parent_child" name field so the YAML key stays readable.
std::string frameToken(const std::string & frame)
{
  std::string out;
  out.reserve(frame.size());
  for (const char c : frame) {
    out.push_back((c == '/' || c == ' ') ? '_' : c);
  }
  return out;
}

// URDF/tf convention: R = Rz(yaw) * Ry(pitch) * Rx(roll). Returns [roll, pitch, yaw].
std::array<double, 3> rpyFromRotation(const Eigen::Matrix3d & rot)
{
  const double pitch = std::asin(std::clamp(-rot(2, 0), -1.0, 1.0));
  const double roll = std::atan2(rot(2, 1), rot(2, 2));
  const double yaw = std::atan2(rot(1, 0), rot(0, 0));
  return {roll, pitch, yaw};
}

void writeTfEntry(
  std::ostream & out,
  const std::string & parent,
  const std::string & child,
  const std::string & lookup,
  bool available,
  const std::string & reason,
  const geometry_msgs::msg::Transform & tf)
{
  out << "  - name: \"T_" << frameToken(parent) << "_" << frameToken(child) << "\"\n";
  out << "    parent: \"" << yamlEscape(parent) << "\"\n";
  out << "    child: \"" << yamlEscape(child) << "\"\n";
  out << "    lookup: \"" << lookup << "\"\n";
  out << "    available: " << (available ? "true" : "false") << "\n";
  if (!available) {
    out << "    reason: \"" << yamlEscape(reason) << "\"\n";
    return;
  }
  if (!reason.empty()) {
    out << "    note: \"" << yamlEscape(reason) << "\"\n";
  }

  const auto & t = tf.translation;
  const auto & q = tf.rotation;
  const Eigen::Quaterniond quat = Eigen::Quaterniond(q.w, q.x, q.y, q.z).normalized();
  const Eigen::Matrix3d rot = quat.toRotationMatrix();
  const auto rpy = rpyFromRotation(rot);
  const std::array<double, 3> trans{t.x, t.y, t.z};

  std::ostringstream body;
  body << std::setprecision(9);
  body << "    xyz: [" << t.x << ", " << t.y << ", " << t.z << "]\n";
  body << "    rpy: [" << rpy[0] << ", " << rpy[1] << ", " << rpy[2] << "]\n";
  body << "    quaternion_xyzw: [" << q.x << ", " << q.y << ", " << q.z << ", " << q.w << "]\n";
  body << "    matrix:\n";
  for (int r = 0; r < 3; ++r) {
    body << "      - [" << rot(r, 0) << ", " << rot(r, 1) << ", " << rot(r, 2) << ", "
         << trans[static_cast<size_t>(r)] << "]\n";
  }
  body << "      - [0.0, 0.0, 0.0, 1.0]\n";
  out << body.str();
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
  PlanningScene scene;
  {
    std::lock_guard<std::mutex> lock(latest_planning_scene_mutex_);
    latest_planning_scene_ = buildPlanningSceneSnapshot(out.header, out.blocks);
    scene = latest_planning_scene_;
  }
  publishCollisionScene(scene);
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
    it->second.pose = cbpwm::poseFromGraspOffset(T_world_tcp, T_tcp_block);
    it->second.pose_status = Block::POSE_PRECISE;
    it->second.confidence = 1.0f;
    it->second.last_seen = header.stamp;
    setDiagonalPoseCovariance(
      it->second,
      kPrecisePositionSigmaMinM,
      kPreciseOrientationSigmaRad);
  }
}

bool PerceptionOrchestratorNode::lookupCaptureTransform(
  const std::string & parent,
  const std::string & child,
  const builtin_interfaces::msg::Time & stamp,
  geometry_msgs::msg::TransformStamped & out,
  std::string & lookup,
  std::string & reason)
{
  if (parent == child) {
    out = geometry_msgs::msg::TransformStamped();
    out.header.frame_id = parent;
    out.child_frame_id = child;
    out.transform.rotation.w = 1.0;
    lookup = "identity";
    return true;
  }
  if (!tf_buffer_) {
    reason = "TF buffer unavailable";
    return false;
  }
  try {
    out = tf_buffer_->lookupTransform(parent, child, stamp, tf2::durationFromSec(0.1));
    lookup = "stamp";
    return true;
  } catch (const tf2::TransformException & ex_stamp) {
    try {
      out = tf_buffer_->lookupTransform(parent, child, tf2::TimePointZero);
      lookup = "latest";
      reason = std::string("stamp lookup failed, used latest available: ") + ex_stamp.what();
      return true;
    } catch (const tf2::TransformException & ex_latest) {
      reason = ex_latest.what();
      return false;
    }
  }
}

void PerceptionOrchestratorNode::writeSceneDiscoveryTfSnapshot(
  const std::filesystem::path & capture_dir,
  const sensor_msgs::msg::Image & image,
  const sensor_msgs::msg::PointCloud2 & cloud)
{
  if (capture_dir.empty()) {
    return;
  }

  std::ofstream out(capture_dir / "tf.yaml");
  if (!out.is_open()) {
    RCLCPP_WARN(
      get_logger(),
      "Scene-discovery capture: failed to write tf.yaml in %s",
      capture_dir.c_str());
    return;
  }

  const std::string image_frame = image.header.frame_id;
  const std::string cloud_frame = cloud.header.frame_id;

  out << "# TF snapshot for this scene-discovery capture.\n";
  out << "# Convention: T_parent_child maps points from child coords into parent coords\n";
  out << "#   (p_parent = T_parent_child * p_child).\n";
  out << "# lookup: \"stamp\" = at the cloud stamp, \"latest\" = newest available (see note),\n";
  out << "#   \"identity\" = parent == child.\n";
  out << "# rgb.png is in image_frame, cloud.pcd is in cloud_frame, block poses are in"
      << " reference_frame.\n";
  out << "reference_frame: \"" << yamlEscape(world_frame_) << "\"\n";
  out << "stamp:\n";
  out << "  sec: " << cloud.header.stamp.sec << "\n";
  out << "  nanosec: " << cloud.header.stamp.nanosec << "\n";
  out << "image_frame: \"" << yamlEscape(image_frame) << "\"\n";
  out << "cloud_frame: \"" << yamlEscape(cloud_frame) << "\"\n";
  out << "transforms:\n";

  std::vector<std::pair<std::string, std::string>> pairs;
  std::unordered_set<std::string> seen;
  const auto add_pair =
    [&pairs, &seen](const std::string & parent, const std::string & child) {
      if (parent.empty() || child.empty()) {
        return;
      }
      if (seen.insert(parent + '\n' + child).second) {
        pairs.emplace_back(parent, child);
      }
    };

  // world <- sensor frames of the exported rgb.png / cloud.pcd.
  add_pair(world_frame_, image_frame);
  add_pair(world_frame_, cloud_frame);
  // Direct camera -> lidar extrinsic for projecting rgb.png onto cloud.pcd.
  add_pair(cloud_frame, image_frame);
  // world <- extra bodies of interest (e.g. crane base K0_mounting_base).
  for (const auto & frame : scene_discovery_capture_tf_frames_) {
    add_pair(world_frame_, frame);
  }

  size_t resolved = 0;
  for (const auto & pair : pairs) {
    geometry_msgs::msg::TransformStamped tf;
    std::string lookup;
    std::string reason;
    const bool ok =
      lookupCaptureTransform(pair.first, pair.second, cloud.header.stamp, tf, lookup, reason);
    if (ok) {
      ++resolved;
    }
    writeTfEntry(out, pair.first, pair.second, lookup, ok, reason, tf.transform);
  }

  RCLCPP_INFO(
    get_logger(),
    "Scene-discovery capture: wrote tf.yaml (%zu/%zu transforms resolved) in %s",
    resolved,
    pairs.size(),
    capture_dir.c_str());
}
