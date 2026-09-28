#include "concrete_block_world_model/world_model/config_loader.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <unordered_set>

#include "concrete_block_world_model_interfaces/msg/block.hpp"
#include "concrete_block_world_model/utils/block_utils.hpp"
#include <yaml-cpp/yaml.h>

namespace cbp::world_model
{

namespace
{

using concrete_block_world_model_interfaces::msg::Block;

int parsePoseStatus(const YAML::Node & node, int fallback)
{
  if (!node || node.IsNull()) {
    return fallback;
  }
  if (node.IsScalar()) {
    const std::string value = node.as<std::string>("");
    if (value == "POSE_UNKNOWN") {
      return Block::POSE_UNKNOWN;
    }
    if (value == "POSE_COARSE") {
      return Block::POSE_COARSE;
    }
    if (value == "POSE_PRECISE") {
      return Block::POSE_PRECISE;
    }
    try {
      return std::stoi(value);
    } catch (...) {
      return fallback;
    }
  }
  return fallback;
}

int parseTaskStatus(const YAML::Node & node, int fallback)
{
  if (!node || node.IsNull()) {
    return fallback;
  }
  if (node.IsScalar()) {
    const std::string value = node.as<std::string>("");
    if (value == "TASK_UNKNOWN") {
      return Block::TASK_UNKNOWN;
    }
    if (value == "TASK_FREE") {
      return Block::TASK_FREE;
    }
    if (value == "TASK_MOVE") {
      return Block::TASK_MOVE;
    }
    if (value == "TASK_PLACED") {
      return Block::TASK_PLACED;
    }
    if (value == "TASK_REMOVED") {
      return Block::TASK_REMOVED;
    }
    try {
      return std::stoi(value);
    } catch (...) {
      return fallback;
    }
  }
  return fallback;
}

std::vector<InitialBlockConfig> parseInitialBlocksYaml(
  rclcpp::Logger logger,
  const std::string & world_frame,
  const std::string & yaml_payload)
{
  std::vector<InitialBlockConfig> out;
  if (yaml_payload.empty()) {
    return out;
  }

  YAML::Node root;
  try {
    root = YAML::Load(yaml_payload);
  } catch (const std::exception & exc) {
    RCLCPP_ERROR(logger, "Failed to parse world_model.initial_blocks YAML: %s", exc.what());
    return out;
  }

  if (!root || !root.IsSequence()) {
    RCLCPP_ERROR(logger, "world_model.initial_blocks must be a YAML sequence.");
    return out;
  }

  std::unordered_set<std::string> seen_ids;
  for (std::size_t idx = 0; idx < root.size(); ++idx) {
    const YAML::Node node = root[idx];
    if (!node.IsMap()) {
      RCLCPP_WARN(logger, "Skipping initial block %zu: expected mapping.", idx + 1);
      continue;
    }

    InitialBlockConfig block;
    block.id = node["id"].as<std::string>("");
    if (block.id.empty()) {
      RCLCPP_WARN(logger, "Skipping initial block %zu: id must not be empty.", idx + 1);
      continue;
    }
    if (!seen_ids.insert(block.id).second) {
      RCLCPP_WARN(logger, "Skipping initial block '%s': duplicate id.", block.id.c_str());
      continue;
    }

    block.frame_id = node["frame_id"].as<std::string>(world_frame);
    if (block.frame_id.empty()) {
      block.frame_id = world_frame;
    }
    if (block.frame_id != world_frame) {
      RCLCPP_WARN(
        logger,
        "Skipping initial block '%s': frame_id '%s' must match world_frame '%s'.",
        block.id.c_str(),
        block.frame_id.c_str(),
        world_frame.c_str());
      continue;
    }

    const YAML::Node position = node["position"];
    if (!position || !position.IsSequence() || position.size() != 3) {
      RCLCPP_WARN(
        logger,
        "Skipping initial block '%s': position must be a 3-element sequence.",
        block.id.c_str());
      continue;
    }
    try {
      for (std::size_t axis = 0; axis < 3; ++axis) {
        block.position[axis] = position[axis].as<double>();
      }
    } catch (...) {
      RCLCPP_WARN(
        logger,
        "Skipping initial block '%s': position values must be numeric.",
        block.id.c_str());
      continue;
    }
    if (!std::isfinite(block.position[0]) ||
      !std::isfinite(block.position[1]) ||
      !std::isfinite(block.position[2]))
    {
      RCLCPP_WARN(
        logger,
        "Skipping initial block '%s': position values must be finite.",
        block.id.c_str());
      continue;
    }

    try {
      block.yaw_deg = node["yaw_deg"].as<double>(0.0);
      block.confidence = node["confidence"].as<double>(1.0);
    } catch (...) {
      RCLCPP_WARN(
        logger,
        "Skipping initial block '%s': yaw_deg/confidence must be numeric.",
        block.id.c_str());
      continue;
    }
    if (!std::isfinite(block.yaw_deg) || !std::isfinite(block.confidence)) {
      RCLCPP_WARN(
        logger,
        "Skipping initial block '%s': yaw_deg/confidence must be finite.",
        block.id.c_str());
      continue;
    }

    block.pose_status = parsePoseStatus(node["pose_status"], Block::POSE_COARSE);
    block.task_status = parseTaskStatus(node["task_status"], Block::TASK_PLACED);
    if (!isKnownPoseStatus(block.pose_status)) {
      RCLCPP_WARN(
        logger,
        "Skipping initial block '%s': unsupported pose_status.",
        block.id.c_str());
      continue;
    }
    if (!isKnownTaskStatus(block.task_status)) {
      RCLCPP_WARN(
        logger,
        "Skipping initial block '%s': unsupported task_status.",
        block.id.c_str());
      continue;
    }

    out.push_back(block);
  }

  return out;
}

// A three-element parameter into a fixed triple. A wrong-sized list keeps the struct's defaults
// on the axes it does not name, and says so, rather than being silently padded with zeros.
void copyVectorInto(
  rclcpp::Logger logger,
  const std::vector<double> & values,
  const char * param_name,
  std::array<double, 3> & out)
{
  if (values.size() != out.size()) {
    RCLCPP_WARN(
      logger, "%s has %zu entries, expected 3; using the defaults for the rest.", param_name,
      values.size());
  }
  for (std::size_t idx = 0; idx < out.size() && idx < values.size(); ++idx) {
    out[idx] = values[idx];
  }
}

std::vector<StaticSceneObjectConfig> parseStaticSceneObjectsYaml(
  rclcpp::Logger logger,
  const std::string & world_frame,
  const std::string & yaml_payload)
{
  std::vector<StaticSceneObjectConfig> out;
  if (yaml_payload.empty()) {
    return out;
  }

  YAML::Node root;
  try {
    root = YAML::Load(yaml_payload);
  } catch (const std::exception & exc) {
    RCLCPP_ERROR(logger, "Failed to parse world_model.static_scene_objects YAML: %s", exc.what());
    return out;
  }

  if (!root || !root.IsSequence()) {
    RCLCPP_ERROR(logger, "world_model.static_scene_objects must be a YAML sequence.");
    return out;
  }

  std::unordered_set<std::string> seen_ids;
  for (std::size_t idx = 0; idx < root.size(); ++idx) {
    const YAML::Node node = root[idx];
    if (!node.IsMap()) {
      RCLCPP_WARN(logger, "Skipping static scene object %zu: expected mapping.", idx + 1);
      continue;
    }

    StaticSceneObjectConfig object;
    object.id = node["id"].as<std::string>("");
    if (object.id.empty()) {
      RCLCPP_WARN(logger, "Skipping static scene object %zu: id must not be empty.", idx + 1);
      continue;
    }
    if (!seen_ids.insert(object.id).second) {
      RCLCPP_WARN(logger, "Skipping static scene object '%s': duplicate id.", object.id.c_str());
      continue;
    }

    object.frame_id = node["frame_id"].as<std::string>(world_frame);
    if (object.frame_id.empty()) {
      object.frame_id = world_frame;
    }
    const YAML::Node position = node["position"];
    const YAML::Node dimensions = node["dimensions"];
    if (!position || !position.IsSequence() || position.size() != 3) {
      RCLCPP_WARN(
        logger,
        "Skipping static scene object '%s': position must be a 3-element sequence.",
        object.id.c_str());
      continue;
    }
    if (!dimensions || !dimensions.IsSequence() || dimensions.size() != 3) {
      RCLCPP_WARN(
        logger,
        "Skipping static scene object '%s': dimensions must be a 3-element sequence.",
        object.id.c_str());
      continue;
    }

    try {
      for (std::size_t axis = 0; axis < 3; ++axis) {
        object.position[axis] = position[axis].as<double>();
        object.dimensions[axis] = dimensions[axis].as<double>();
      }
      const YAML::Node rpy_deg = node["rpy_deg"];
      if (rpy_deg && rpy_deg.IsSequence() && rpy_deg.size() == 3) {
        for (std::size_t axis = 0; axis < 3; ++axis) {
          object.rpy_deg[axis] = rpy_deg[axis].as<double>();
        }
      }
    } catch (...) {
      RCLCPP_WARN(
        logger,
        "Skipping static scene object '%s': position/dimensions/rpy_deg must be numeric.",
        object.id.c_str());
      continue;
    }

    bool finite = true;
    for (double value : object.position) {
      finite = finite && std::isfinite(value);
    }
    for (double value : object.dimensions) {
      finite = finite && std::isfinite(value) && value > 0.0;
    }
    for (double value : object.rpy_deg) {
      finite = finite && std::isfinite(value);
    }
    if (!finite) {
      RCLCPP_WARN(
        logger,
        "Skipping static scene object '%s': values must be finite and dimensions > 0.",
        object.id.c_str());
      continue;
    }

    out.push_back(object);
  }

  return out;
}

}  // namespace

WorldModelConfig loadWorldModelConfig(rclcpp::Node & node)
{
  WorldModelConfig cfg;

  (void)node.declare_parameter<std::string>("pipeline_mode", "full");
  // "perception_mode" is still accepted from launch files for backward compatibility
  // but is ignored: the world model is single-shot only (run_pose_estimation).
  (void)node.declare_parameter<std::string>("perception_mode", "IDLE");
  cfg.world_frame = node.declare_parameter<std::string>("world_frame", "world");
  cfg.association_max_distance_m =
    node.declare_parameter<double>("world_model.association_max_distance_m", 0.45);
  cfg.association_max_age_s = node.declare_parameter<double>(
    "world_model.association_max_age_s",
    20.0);
  cfg.min_update_confidence = node.declare_parameter<double>(
    "world_model.min_update_confidence",
    0.25);
  cfg.scene_discovery_min_detector_confidence = node.declare_parameter<double>(
    "world_model.scene_discovery.min_detector_confidence",
    cfg.min_update_confidence);
  cfg.scene_discovery_association_max_distance_m = node.declare_parameter<double>(
    "world_model.scene_discovery.association_max_distance_m",
    cfg.association_max_distance_m);
  cfg.task_move_fk_tracking_enabled = node.declare_parameter<bool>(
    "task_move.fk_tracking.enabled",
    true);
  // Extra frames snapshotted as T_world_<frame> in each scene-discovery *capture's* tf.yaml.
  // The RGB dump this key is named after is gone, but the name is not free to fix here: replay
  // capture profiles outside this package set it (the six gripper-rail frames an annotator
  // needs), and rclcpp silently ignores an override whose parameter is not declared -- renaming
  // it would drop those frames with no error anywhere. Rename it together with those profiles.
  cfg.scene_discovery_capture_tf_frames =
    node.declare_parameter<std::vector<std::string>>(
    "debug.scene_discovery_dump.tf_frames", std::vector<std::string>{"K0_mounting_base"});
  cfg.marker_refresh_period_s = node.declare_parameter<double>(
    "world_model.marker_refresh_period_s", 0.5);
  cfg.collision_scene_heartbeat_s = node.declare_parameter<double>(
    "world_model.collision_scene_heartbeat_s", 2.0);

  cfg.refine_grasped_tcp_frame =
    node.declare_parameter<std::string>("refine_grasped.tcp_frame", "elastic/K8_tool_center_point");
  cfg.refine_grasped_camera_info_topic = node.declare_parameter<std::string>(
    "refine_grasped.camera_info_topic", "/blackfly_rotated/camera_info");
  cfg.refine_grasped_tcp_to_block_xyz =
    node.declare_parameter<std::vector<double>>("refine_grasped.tcp_to_block.xyz", {0.0, 0.0, 0.0});
  cfg.refine_grasped_tcp_to_block_rpy =
    node.declare_parameter<std::vector<double>>("refine_grasped.tcp_to_block.rpy", {0.0, 0.0, 0.0});
  cfg.refine_grasped_grasp_offset_max_deviation_m = node.declare_parameter<double>(
    "refine_grasped.grasp_offset_capture.max_deviation_m", 1.0);

  cfg.initial_blocks_yaml =
    node.declare_parameter<std::string>("world_model.initial_blocks", "");
  cfg.static_scene_objects_yaml =
    node.declare_parameter<std::string>("world_model.static_scene_objects", "");

  // The vehicle the crane is bolted to, described in exactly one place and emitted as the
  // reserved `truck` primitive. Off by default so a deployment that has not measured its truck
  // publishes no vehicle at all rather than a guessed one.
  cfg.vehicle_box.enabled = node.declare_parameter<bool>("world_model.vehicle_box.enable", false);
  cfg.vehicle_box.frame_id =
    node.declare_parameter<std::string>("world_model.vehicle_box.frame_id", cfg.world_frame);
  if (cfg.vehicle_box.frame_id.empty()) {
    cfg.vehicle_box.frame_id = cfg.world_frame;
  }
  copyVectorInto(
    node.get_logger(),
    node.declare_parameter<std::vector<double>>(
      "world_model.vehicle_box.position", {0.0, 0.0, 0.0}),
    "world_model.vehicle_box.position", cfg.vehicle_box.position);
  copyVectorInto(
    node.get_logger(),
    node.declare_parameter<std::vector<double>>("world_model.vehicle_box.rpy_deg", {0.0, 0.0, 0.0}),
    "world_model.vehicle_box.rpy_deg", cfg.vehicle_box.rpy_deg);
  copyVectorInto(
    node.get_logger(),
    node.declare_parameter<std::vector<double>>(
      "world_model.vehicle_box.dimensions", {0.0, 0.0, 0.0}),
    "world_model.vehicle_box.dimensions", cfg.vehicle_box.dimensions);
  cfg.vehicle_box.runge_length_m =
    node.declare_parameter<double>("world_model.vehicle_box.runge_length_m", 0.0);
  cfg.vehicle_box.runge_stations_m = node.declare_parameter<std::vector<double>>(
    "world_model.vehicle_box.runge_stations_m", std::vector<double>{});

  const auto block_dimensions =
    node.declare_parameter<std::vector<double>>("world_model.block_dimensions_m", {0.6, 0.9, 0.6});
  for (std::size_t idx = 0; idx < cfg.block_dimensions_m.size() && idx < block_dimensions.size();
    ++idx)
  {
    cfg.block_dimensions_m[idx] = block_dimensions[idx];
  }

  cfg.initial_blocks = parseInitialBlocksYaml(
    node.get_logger(), cfg.world_frame, cfg.initial_blocks_yaml);
  cfg.static_scene_objects = parseStaticSceneObjectsYaml(
    node.get_logger(), cfg.world_frame, cfg.static_scene_objects_yaml);
  return cfg;
}

void normalizeWorldModelConfig(rclcpp::Logger logger, WorldModelConfig & cfg)
{
  auto clamp_min = [logger](double & value, double min_value, const char * name) {
      if (value < min_value) {
        RCLCPP_WARN(logger, "Invalid %s=%.3f, clamping to %.3f", name, value, min_value);
        value = min_value;
      }
    };
  clamp_min(cfg.association_max_distance_m, 0.01, "world_model.association_max_distance_m");
  clamp_min(cfg.association_max_age_s, 0.1, "world_model.association_max_age_s");
  clamp_min(cfg.min_update_confidence, 0.0, "world_model.min_update_confidence");
  clamp_min(
    cfg.scene_discovery_min_detector_confidence, 0.0,
    "world_model.scene_discovery.min_detector_confidence");
  clamp_min(
    cfg.scene_discovery_association_max_distance_m, 0.01,
    "world_model.scene_discovery.association_max_distance_m");

  for (std::size_t idx = 0; idx < cfg.block_dimensions_m.size(); ++idx) {
    if (!std::isfinite(cfg.block_dimensions_m[idx]) || cfg.block_dimensions_m[idx] <= 0.0) {
      RCLCPP_WARN(
        logger,
        "Invalid world_model.block_dimensions_m[%zu]=%.3f, resetting to 0.6",
        idx,
        cfg.block_dimensions_m[idx]);
      cfg.block_dimensions_m[idx] = 0.6;
    }
  }
  if (!cfg.initial_blocks.empty()) {
    RCLCPP_INFO(
      logger, "Configured %zu seeded world-model blocks for startup.",
      cfg.initial_blocks.size());
  }
  if (!cfg.static_scene_objects.empty()) {
    RCLCPP_INFO(
      logger, "Configured %zu static planning-scene objects for startup.",
      cfg.static_scene_objects.size());
  }
  if (cfg.vehicle_box.enabled) {
    RCLCPP_INFO(
      logger,
      "Vehicle box configured in frame '%s': centre [%.3f %.3f %.3f], extent [%.3f %.3f %.3f]; "
      "it leaves the world model as the reserved 'truck' primitive.",
      cfg.vehicle_box.frame_id.c_str(),
      cfg.vehicle_box.position[0], cfg.vehicle_box.position[1], cfg.vehicle_box.position[2],
      cfg.vehicle_box.dimensions[0], cfg.vehicle_box.dimensions[1], cfg.vehicle_box.dimensions[2]);
  }
}

double vectorComponent(
  rclcpp::Logger logger,
  const std::vector<double> & values,
  size_t index,
  double fallback,
  const char * param_name)
{
  if (index < values.size()) {
    return values[index];
  }
  RCLCPP_WARN(
    logger,
    "Parameter '%s' expected at least %zu entries, got %zu. Using fallback %.3f for index %zu.",
    param_name,
    index + 1,
    values.size(),
    fallback,
    index);
  return fallback;
}

}  // namespace cbp::world_model
