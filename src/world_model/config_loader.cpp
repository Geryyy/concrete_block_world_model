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
  cfg.min_fitness = node.declare_parameter<double>("min_fitness", 0.3);
  cfg.max_rmse = node.declare_parameter<double>("max_rmse", 0.05);
  cfg.object_class = node.declare_parameter<std::string>("object_class", "concrete_block");
  cfg.world_frame = node.declare_parameter<std::string>("world_frame", "world");
  cfg.max_sync_delta_s = node.declare_parameter<double>("sync.max_delta_s", 0.06);
  cfg.association_max_distance_m =
    node.declare_parameter<double>("world_model.association_max_distance_m", 0.45);
  cfg.association_max_age_s = node.declare_parameter<double>(
    "world_model.association_max_age_s",
    20.0);
  cfg.min_update_confidence = node.declare_parameter<double>(
    "world_model.min_update_confidence",
    0.25);
  cfg.task_move_fk_tracking_enabled = node.declare_parameter<bool>(
    "task_move.fk_tracking.enabled",
    true);
  cfg.refine_target_max_distance_m =
    node.declare_parameter<double>("world_model.refine_target_max_distance_m", 1.2);
  cfg.scene_discovery_merge_enabled =
    node.declare_parameter<bool>("world_model.scene_discovery_merge.enable", true);
  cfg.scene_discovery_merge_containment_ratio = node.declare_parameter<double>(
    "world_model.scene_discovery_merge.containment_ratio", 0.3);
  cfg.scene_discovery_merge_iou_threshold = node.declare_parameter<double>(
    "world_model.scene_discovery_merge.iou_threshold", 0.5);
  cfg.scene_discovery_coarse_fallback_enabled =
    node.declare_parameter<bool>("world_model.scene_discovery_coarse_fallback.enable", true);
  cfg.scene_discovery_coarse_fallback_min_points =
    node.declare_parameter<int>("world_model.scene_discovery_coarse_fallback.min_points", 120);
  cfg.coarse_surface_square_ratio_thresh = node.declare_parameter<double>(
    "world_model.scene_discovery_coarse_fallback.surface_shape.square_ratio_thresh", 1.35);
  cfg.coarse_front_center_offset_square_m = node.declare_parameter<double>(
    "world_model.scene_discovery_coarse_fallback.center_offset.square_m", 0.45);
  cfg.coarse_front_center_offset_rect_m = node.declare_parameter<double>(
    "world_model.scene_discovery_coarse_fallback.center_offset.rect_m", 0.30);
  cfg.debug_detection_overlay_enabled = node.declare_parameter<bool>(
    "debug.publish_detection_overlay", true);
  cfg.debug_refine_grasped_roi_input_enabled =
    node.declare_parameter<bool>("debug.publish_refine_grasped_roi_input", true);
  cfg.debug_scene_discovery_dump_enabled =
    node.declare_parameter<bool>("debug.scene_discovery_dump.enable", false);
  cfg.debug_scene_discovery_dump_dir =
    node.declare_parameter<std::string>("debug.scene_discovery_dump.dir", "scene_discovery_dump");
  cfg.perf_log_timing_enabled = node.declare_parameter<bool>("perf.log_timing", true);
  cfg.perf_log_every_n_frames = node.declare_parameter<int>("perf.log_every_n_frames", 100);
  cfg.marker_refresh_period_s = node.declare_parameter<double>(
    "world_model.marker_refresh_period_s", 0.5);

  cfg.refine_grasped_use_fk_roi = node.declare_parameter<bool>("refine_grasped.use_fk_roi", true);
  cfg.refine_grasped_tcp_frame =
    node.declare_parameter<std::string>("refine_grasped.tcp_frame", "elastic/K8_tool_center_point");
  cfg.refine_grasped_camera_frame =
    node.declare_parameter<std::string>("refine_grasped.camera_frame", "");
  cfg.refine_grasped_camera_info_topic = node.declare_parameter<std::string>(
    "refine_grasped.camera_info_topic", "/blackfly_rotated/camera_info");
  cfg.refine_grasped_min_depth_m =
    node.declare_parameter<double>("refine_grasped.min_depth_m", 0.5);
  cfg.refine_grasped_max_depth_m =
    node.declare_parameter<double>("refine_grasped.max_depth_m", 30.0);
  cfg.refine_grasped_segmentation_timeout_s =
    node.declare_parameter<double>("refine_grasped.segmentation_timeout_s", 3.0);
  cfg.refine_grasped_use_black_bg =
    node.declare_parameter<bool>("refine_grasped.segmentation_input.use_black_background", false);
  cfg.refine_grasped_blur_kernel_size =
    node.declare_parameter<int>("refine_grasped.segmentation_input.blur_kernel_size", 31);
  cfg.refine_grasped_pose_fusion.enabled =
    node.declare_parameter<bool>("refine_grasped.pose_fusion.enable", true);
  cfg.refine_grasped_pose_fusion.mode = node.declare_parameter<std::string>(
    "refine_grasped.pose_fusion.mode",
    "position_from_registration_orientation_from_fk");
  cfg.refine_grasped_pose_fusion.max_translation_jump_m = node.declare_parameter<double>(
    "refine_grasped.pose_fusion.max_translation_jump_m", 0.35);
  cfg.refine_grasped_pose_fusion.max_z_delta_m = node.declare_parameter<double>(
    "refine_grasped.pose_fusion.max_z_delta_m", 0.25);
  cfg.refine_grasped_pose_fusion.debug_log =
    node.declare_parameter<bool>("refine_grasped.pose_fusion.debug_log", true);
  cfg.refine_grasped_tcp_to_block_xyz =
    node.declare_parameter<std::vector<double>>("refine_grasped.tcp_to_block.xyz", {0.0, 0.0, 0.0});
  cfg.refine_grasped_tcp_to_block_rpy =
    node.declare_parameter<std::vector<double>>("refine_grasped.tcp_to_block.rpy", {0.0, 0.0, 0.0});
  cfg.refine_grasped_grasp_offset_max_deviation_m = node.declare_parameter<double>(
    "refine_grasped.grasp_offset_capture.max_deviation_m", 1.0);
  cfg.refine_grasped_roi_size_m =
    node.declare_parameter<std::vector<double>>("refine_grasped.roi_size_m", {0.60, 0.40});

  cfg.refine_block_use_pose_roi = node.declare_parameter<bool>("refine_block.use_pose_roi", false);
  cfg.refine_block_roi_size_m =
    node.declare_parameter<std::vector<double>>("refine_block.roi_size_m", {1.20, 1.00});
  cfg.refine_block_min_depth_m = node.declare_parameter<double>("refine_block.min_depth_m", 0.5);
  cfg.refine_block_max_depth_m = node.declare_parameter<double>("refine_block.max_depth_m", 30.0);
  cfg.refine_block_segmentation_timeout_s =
    node.declare_parameter<double>("refine_block.segmentation_timeout_s", 3.0);
  cfg.refine_block_use_black_bg =
    node.declare_parameter<bool>("refine_block.segmentation_input.use_black_background", false);
  cfg.refine_block_blur_kernel_size =
    node.declare_parameter<int>("refine_block.segmentation_input.blur_kernel_size", 31);
  cfg.initial_blocks_yaml =
    node.declare_parameter<std::string>("world_model.initial_blocks", "");
  cfg.static_scene_objects_yaml =
    node.declare_parameter<std::string>("world_model.static_scene_objects", "");
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
  auto normalize_blur = [logger](int & kernel, const char * name) {
      if (kernel < 1) {
        RCLCPP_WARN(logger, "Invalid %s=%d, clamping to 1", name, kernel);
        kernel = 1;
      }
      if ((kernel % 2) == 0) {
        RCLCPP_WARN(
          logger, "Invalid %s=%d (must be odd), incrementing to %d", name, kernel,
          kernel + 1);
        kernel += 1;
      }
    };
  auto clamp_min_i = [logger](int & value, int min_value, const char * name) {
      if (value < min_value) {
        RCLCPP_WARN(logger, "Invalid %s=%d, clamping to %d", name, value, min_value);
        value = min_value;
      }
    };

  clamp_min(cfg.min_fitness, 0.0, "min_fitness");
  clamp_min(cfg.max_rmse, 0.0, "max_rmse");
  clamp_min(cfg.association_max_distance_m, 0.01, "world_model.association_max_distance_m");
  clamp_min(cfg.association_max_age_s, 0.1, "world_model.association_max_age_s");
  clamp_min(cfg.min_update_confidence, 0.0, "world_model.min_update_confidence");
  clamp_min(cfg.refine_target_max_distance_m, 0.01, "world_model.refine_target_max_distance_m");
  clamp_min_i(
    cfg.scene_discovery_coarse_fallback_min_points, 1,
    "scene_discovery_coarse_fallback.min_points");
  clamp_min(
    cfg.coarse_surface_square_ratio_thresh, 1.0,
    "scene_discovery_coarse_fallback.surface_shape.square_ratio_thresh");
  clamp_min(
    cfg.coarse_front_center_offset_square_m, 0.0,
    "scene_discovery_coarse_fallback.center_offset.square_m");
  clamp_min(
    cfg.coarse_front_center_offset_rect_m, 0.0,
    "scene_discovery_coarse_fallback.center_offset.rect_m");

  if (cfg.refine_grasped_min_depth_m > cfg.refine_grasped_max_depth_m) {
    RCLCPP_WARN(
      logger,
      "Invalid refine_grasped depth range [%.3f, %.3f], swapping bounds",
      cfg.refine_grasped_min_depth_m, cfg.refine_grasped_max_depth_m);
    std::swap(cfg.refine_grasped_min_depth_m, cfg.refine_grasped_max_depth_m);
  }
  if (cfg.refine_block_min_depth_m > cfg.refine_block_max_depth_m) {
    RCLCPP_WARN(
      logger,
      "Invalid refine_block depth range [%.3f, %.3f], swapping bounds",
      cfg.refine_block_min_depth_m, cfg.refine_block_max_depth_m);
    std::swap(cfg.refine_block_min_depth_m, cfg.refine_block_max_depth_m);
  }
  normalize_blur(
    cfg.refine_grasped_blur_kernel_size, "refine_grasped.segmentation_input.blur_kernel_size");
  normalize_blur(
    cfg.refine_block_blur_kernel_size,
    "refine_block.segmentation_input.blur_kernel_size");
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
