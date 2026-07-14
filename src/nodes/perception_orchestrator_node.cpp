#include "concrete_block_world_model/nodes/perception_orchestrator_node.hpp"

#include "concrete_block_world_model/utils/block_utils.hpp"
#include "concrete_block_world_model/world_model/config_loader.hpp"

#include <tf2/LinearMath/Quaternion.h>

#define WM_LOG(logger, ...) RCLCPP_INFO(logger, __VA_ARGS__)

PerceptionOrchestratorNode::PerceptionOrchestratorNode()
: Node("block_world_model_node")
{
    run_pose_cb_group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    action_client_cb_group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    auto startup = cbpwm::loadWorldModelConfig(*this);
    cbpwm::normalizeWorldModelConfig(get_logger(), startup);
    runtime_cfg_.min_fitness = startup.min_fitness;
    runtime_cfg_.max_rmse = startup.max_rmse;
    object_class_ = startup.object_class;
    world_frame_ = startup.world_frame;
    runtime_cfg_.max_sync_delta_s = startup.max_sync_delta_s;
    runtime_cfg_.association_max_distance_m = startup.association_max_distance_m;
    runtime_cfg_.association_max_age_s = startup.association_max_age_s;
    runtime_cfg_.min_update_confidence = startup.min_update_confidence;
    runtime_cfg_.refine_target_max_distance_m = startup.refine_target_max_distance_m;
    scene_discovery_merge_enabled_ = startup.scene_discovery_merge_enabled;
    scene_discovery_merge_containment_ratio_ = startup.scene_discovery_merge_containment_ratio;
    scene_discovery_merge_iou_threshold_ = startup.scene_discovery_merge_iou_threshold;
    scene_discovery_coarse_fallback_enabled_ = startup.scene_discovery_coarse_fallback_enabled;
    scene_discovery_coarse_fallback_min_points_ = startup.scene_discovery_coarse_fallback_min_points;
    coarse_surface_square_ratio_thresh_ = startup.coarse_surface_square_ratio_thresh;
    coarse_front_center_offset_square_m_ = startup.coarse_front_center_offset_square_m;
    coarse_front_center_offset_rect_m_ = startup.coarse_front_center_offset_rect_m;
    debug_detection_overlay_enabled_ = startup.debug_detection_overlay_enabled;
    debug_refine_grasped_roi_input_enabled_ = startup.debug_refine_grasped_roi_input_enabled;
    task_move_fk_tracking_enabled_ = startup.task_move_fk_tracking_enabled;
    perf_log_timing_enabled_ = startup.perf_log_timing_enabled;
    perf_log_every_n_frames_ = startup.perf_log_every_n_frames;
    refine_grasped_use_fk_roi_ = startup.refine_grasped_use_fk_roi;
    refine_grasped_tcp_frame_ = startup.refine_grasped_tcp_frame;
    refine_grasped_camera_frame_ = startup.refine_grasped_camera_frame;
    refine_grasped_camera_info_topic_ = startup.refine_grasped_camera_info_topic;
    refine_grasped_roi_cfg_.min_depth_m = startup.refine_grasped_min_depth_m;
    refine_grasped_roi_cfg_.max_depth_m = startup.refine_grasped_max_depth_m;
    refine_grasped_roi_cfg_.segmentation_timeout_s = startup.refine_grasped_segmentation_timeout_s;
    refine_grasped_roi_cfg_.use_black_bg = startup.refine_grasped_use_black_bg;
    refine_grasped_roi_cfg_.blur_kernel_size = startup.refine_grasped_blur_kernel_size;
    refine_grasped_pose_fusion_ = startup.refine_grasped_pose_fusion;
    refine_block_use_pose_roi_ = startup.refine_block_use_pose_roi;
    refine_block_roi_cfg_.min_depth_m = startup.refine_block_min_depth_m;
    refine_block_roi_cfg_.max_depth_m = startup.refine_block_max_depth_m;
    refine_block_roi_cfg_.segmentation_timeout_s = startup.refine_block_segmentation_timeout_s;
    refine_block_roi_cfg_.use_black_bg = startup.refine_block_use_black_bg;
    refine_block_roi_cfg_.blur_kernel_size = startup.refine_block_blur_kernel_size;
    block_dimensions_m_ = startup.block_dimensions_m;

    constexpr double kDegToRad = 3.14159265358979323846 / 180.0;
    static_scene_objects_.reserve(startup.static_scene_objects.size());
    for (const auto & cfg_object : startup.static_scene_objects) {
      PlanningSceneObject object;
      object.id = cfg_object.id;
      object.frame_id = world_frame_;
      object.shape_type = PlanningSceneObject::SHAPE_BOX;
      object.source_type = PlanningSceneObject::SOURCE_STATIC_OBSTACLE;
      object.pose.position.x = cfg_object.position[0];
      object.pose.position.y = cfg_object.position[1];
      object.pose.position.z = cfg_object.position[2];
      tf2::Quaternion quat;
      quat.setRPY(
        cfg_object.rpy_deg[0] * kDegToRad,
        cfg_object.rpy_deg[1] * kDegToRad,
        cfg_object.rpy_deg[2] * kDegToRad);
      object.pose.orientation.x = quat.x();
      object.pose.orientation.y = quat.y();
      object.pose.orientation.z = quat.z();
      object.pose.orientation.w = quat.w();
      object.dimensions.x = cfg_object.dimensions[0];
      object.dimensions.y = cfg_object.dimensions[1];
      object.dimensions.z = cfg_object.dimensions[2];
      object.pose_status = Block::POSE_UNKNOWN;
      object.task_status = Block::TASK_UNKNOWN;
      object.confidence = 1.0F;
      static_scene_objects_.push_back(std::move(object));
    }

    if (perf_log_every_n_frames_ < 1) {
      perf_log_every_n_frames_ = 1;
    }

    if (debug_detection_overlay_enabled_.load()) {
      const auto debug_image_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();
      det_debug_pub_ = create_publisher<sensor_msgs::msg::Image>(
        "debug/detection_overlay", debug_image_qos);
      yolo_service_debug_pub_ = create_publisher<sensor_msgs::msg::Image>(
        "debug/yolo_service_debug_image", debug_image_qos);
    }
    if (debug_refine_grasped_roi_input_enabled_.load()) {
      const auto debug_image_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();
      refine_grasped_roi_input_pub_ =
        create_publisher<sensor_msgs::msg::Image>(
        "debug/refine_grasped_roi_input", debug_image_qos);
    }
    camera_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
      refine_grasped_camera_info_topic_,
      rclcpp::SensorDataQoS(),
      std::bind(&PerceptionOrchestratorNode::cameraInfoCallback, this, std::placeholders::_1));

    const double tx = cbpwm::vectorComponent(
      get_logger(),
      startup.refine_grasped_tcp_to_block_xyz,
      0,
      0.0,
      "refine_grasped.tcp_to_block.xyz");
    const double ty = cbpwm::vectorComponent(
      get_logger(),
      startup.refine_grasped_tcp_to_block_xyz,
      1,
      0.0,
      "refine_grasped.tcp_to_block.xyz");
    const double tz = cbpwm::vectorComponent(
      get_logger(),
      startup.refine_grasped_tcp_to_block_xyz,
      2,
      0.0,
      "refine_grasped.tcp_to_block.xyz");
    const double rr = cbpwm::vectorComponent(
      get_logger(),
      startup.refine_grasped_tcp_to_block_rpy,
      0,
      0.0,
      "refine_grasped.tcp_to_block.rpy");
    const double rp = cbpwm::vectorComponent(
      get_logger(),
      startup.refine_grasped_tcp_to_block_rpy,
      1,
      0.0,
      "refine_grasped.tcp_to_block.rpy");
    const double ry = cbpwm::vectorComponent(
      get_logger(),
      startup.refine_grasped_tcp_to_block_rpy,
      2,
      0.0,
      "refine_grasped.tcp_to_block.rpy");
    const Eigen::Matrix3d rot_tcp_block =
      (Eigen::AngleAxisd(ry, Eigen::Vector3d::UnitZ()) *
      Eigen::AngleAxisd(rp, Eigen::Vector3d::UnitY()) *
      Eigen::AngleAxisd(rr, Eigen::Vector3d::UnitX())).toRotationMatrix();
    T_tcp_block_ = Eigen::Matrix4d::Identity();
    T_tcp_block_.block<3, 3>(0, 0) = rot_tcp_block;
    T_tcp_block_.block<3, 1>(0, 3) = Eigen::Vector3d(tx, ty, tz);
    // A configured (non-identity) nominal enables the deviation gate on auto-captured
    // offsets; left at identity it only serves as a last-resort fallback.
    grasp_offset_nominal_configured_ =
      !T_tcp_block_.isApprox(Eigen::Matrix4d::Identity(), 1e-6);
    refine_grasped_grasp_offset_max_deviation_m_ =
      startup.refine_grasped_grasp_offset_max_deviation_m;

    refine_grasped_roi_cfg_.roi_size_x_m = cbpwm::vectorComponent(
      get_logger(), startup.refine_grasped_roi_size_m, 0, 0.60, "refine_grasped.roi_size_m");
    refine_grasped_roi_cfg_.roi_size_y_m = cbpwm::vectorComponent(
      get_logger(), startup.refine_grasped_roi_size_m, 1, 0.40, "refine_grasped.roi_size_m");
    refine_block_roi_cfg_.roi_size_x_m =
      cbpwm::vectorComponent(
      get_logger(), startup.refine_block_roi_size_m, 0, 1.20, "refine_block.roi_size_m");
    refine_block_roi_cfg_.roi_size_y_m =
      cbpwm::vectorComponent(
      get_logger(), startup.refine_block_roi_size_m, 1, 1.00, "refine_block.roi_size_m");

    world_pub_ = create_publisher<BlockArray>("block_world_model", 10);
    const auto marker_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();
    marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
      "block_world_model_markers", marker_qos);
    goal_marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
      "block_goal_markers", marker_qos);

    image_sub_.subscribe(this, "image");
    cloud_sub_.subscribe(this, "points");
    sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
      SyncPolicy(10), image_sub_, cloud_sub_);
    sync_->registerCallback(
      std::bind(
        &PerceptionOrchestratorNode::syncCallback,
        this,
        std::placeholders::_1,
        std::placeholders::_2));
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    segment_client_ = create_client<SegmentSrv>(
      "/yolos_segmentor_service/segment",
      rmw_qos_profile_services_default,
      action_client_cb_group_);
    extract_mask_cutout_client_ = create_client<ExtractMaskCutoutSrv>(
      "/extract_mask_cutout",
      rmw_qos_profile_services_default,
      action_client_cb_group_);
    action_client_ = rclcpp_action::create_client<RegisterBlock>(
      this, "register_block", action_client_cb_group_);

    get_coarse_srv_ = create_service<GetCoarseSrv>(
      "~/get_coarse_blocks",
      std::bind(
        &PerceptionOrchestratorNode::handleGetCoarseBlocks,
        this,
        std::placeholders::_1,
        std::placeholders::_2));

    get_planning_scene_srv_ = create_service<GetPlanningSceneSrv>(
      "~/get_planning_scene",
      std::bind(
        &PerceptionOrchestratorNode::handleGetPlanningScene,
        this,
        std::placeholders::_1,
        std::placeholders::_2));

    run_pose_srv_ = create_service<RunPoseSrv>(
      "~/run_pose_estimation",
      std::bind(
        &PerceptionOrchestratorNode::handleRunPoseEstimation,
        this,
        std::placeholders::_1,
        std::placeholders::_2),
      rmw_qos_profile_services_default,
      run_pose_cb_group_);

    set_block_task_status_srv_ = create_service<SetBlockTaskStatusSrv>(
      "~/set_block_task_status",
      std::bind(
        &PerceptionOrchestratorNode::handleSetBlockTaskStatus,
        this,
        std::placeholders::_1,
        std::placeholders::_2));

    upsert_block_srv_ = create_service<UpsertBlockSrv>(
      "~/upsert_block",
      std::bind(
        &PerceptionOrchestratorNode::handleUpsertBlock,
        this,
        std::placeholders::_1,
        std::placeholders::_2));

    set_block_goal_srv_ = create_service<SetBlockGoalSrv>(
      "~/set_block_goal",
      std::bind(
        &PerceptionOrchestratorNode::handleSetBlockGoal,
        this,
        std::placeholders::_1,
        std::placeholders::_2));

    clear_block_goals_srv_ = create_service<ClearBlockGoalsSrv>(
      "~/clear_block_goals",
      std::bind(
        &PerceptionOrchestratorNode::handleClearBlockGoals,
        this,
        std::placeholders::_1,
        std::placeholders::_2));

    clear_world_model_srv_ = create_service<ClearWorldModelSrv>(
      "~/clear_world_model",
      std::bind(
        &PerceptionOrchestratorNode::handleClearWorldModel,
        this,
        std::placeholders::_1,
        std::placeholders::_2));

    marker_refresh_timer_ = create_wall_timer(
      std::chrono::duration<double>(startup.marker_refresh_period_s),
      [this]() {
        const BlockArray snapshot = latestWorldSnapshot();
        if (!snapshot.blocks.empty() || !static_scene_objects_.empty()) {
          std_msgs::msg::Header header;
          header.stamp = now();
          header.frame_id =
            snapshot.header.frame_id.empty() ? world_frame_ : snapshot.header.frame_id;
          publishPersistentWorld(header);
        }
      });

    initializeSeededWorld(startup);

    WM_LOG(
      get_logger(),
      "PerceptionOrchestratorNode ready | trigger_policy=ON_DEMAND (single-shot run_pose_estimation) task_move_fk_tracking=%s",
      task_move_fk_tracking_enabled_ ? "true" : "false");
    if (refine_grasped_use_fk_roi_) {
      WM_LOG(
        get_logger(),
        "REFINE_GRASPED FK+ROI enabled | tcp_frame=%s camera_frame_override=%s camera_info_topic=%s roi_size=[%.2f, %.2f]m",
        refine_grasped_tcp_frame_.c_str(),
        refine_grasped_camera_frame_.empty() ? "<image.header.frame_id>" : refine_grasped_camera_frame_.c_str(),
        refine_grasped_camera_info_topic_.c_str(),
        refine_grasped_roi_cfg_.roi_size_x_m,
        refine_grasped_roi_cfg_.roi_size_y_m);
      RCLCPP_INFO(
        get_logger(),
        "REFINE_GRASPED segmentation input: background=%s blur_kernel=%d seg_timeout=%.2fs",
        refine_grasped_roi_cfg_.use_black_bg ? "black" : "blur",
        refine_grasped_roi_cfg_.blur_kernel_size,
        refine_grasped_roi_cfg_.segmentation_timeout_s);
      RCLCPP_INFO(
        get_logger(),
        "REFINE_GRASPED pose fusion: enabled=%s mode=%s max_jump=%.3fm max_z_delta=%.3fm debug_log=%s",
        refine_grasped_pose_fusion_.enabled ? "true" : "false",
        refine_grasped_pose_fusion_.mode.c_str(),
        refine_grasped_pose_fusion_.max_translation_jump_m,
        refine_grasped_pose_fusion_.max_z_delta_m,
        refine_grasped_pose_fusion_.debug_log ? "true" : "false");
    }
    if (refine_block_use_pose_roi_) {
      RCLCPP_INFO(
        get_logger(),
        "REFINE_BLOCK pose+ROI enabled | roi_size=[%.2f, %.2f]m depth=[%.2f, %.2f]m seg_timeout=%.2fs",
        refine_block_roi_cfg_.roi_size_x_m,
        refine_block_roi_cfg_.roi_size_y_m,
        refine_block_roi_cfg_.min_depth_m,
        refine_block_roi_cfg_.max_depth_m,
        refine_block_roi_cfg_.segmentation_timeout_s);
    }
}

void PerceptionOrchestratorNode::initializeSeededWorld(const cbpwm::WorldModelConfig & startup)
{
  if (!static_scene_objects_.empty()) {
    RCLCPP_INFO(
      get_logger(),
      "Loaded %zu static planning-scene object(s) into world model.",
      static_scene_objects_.size());
  }
  if (startup.initial_blocks.empty()) {
    std_msgs::msg::Header header;
    header.stamp = now();
    header.frame_id = world_frame_;
    publishPersistentWorld(header);
    return;
  }
  constexpr double kDegToRad = 3.14159265358979323846 / 180.0;

  const auto stamp = now();
  std_msgs::msg::Header header;
  header.stamp = stamp;
  header.frame_id = world_frame_;

  {
    std::lock_guard<std::mutex> lock(persistent_world_mutex_);
    for (const auto & cfg_block : startup.initial_blocks) {
      Block block;
      block.id = cfg_block.id;
      block.pose_status = cfg_block.pose_status;
      block.task_status = cfg_block.task_status;
      block.pose.position.x = cfg_block.position[0];
      block.pose.position.y = cfg_block.position[1];
      block.pose.position.z = cfg_block.position[2];
      tf2::Quaternion quat;
      quat.setRPY(0.0, 0.0, cfg_block.yaw_deg * kDegToRad);
      block.pose.orientation.x = quat.x();
      block.pose.orientation.y = quat.y();
      block.pose.orientation.z = quat.z();
      block.pose.orientation.w = quat.w();
      block.confidence = static_cast<float>(cfg_block.confidence);
      block.last_seen = stamp;
      setDefaultPoseCovariance(block);
      persistent_world_[block.id] = block;
      seeded_block_ids_.insert(block.id);
    }
  }

  publishPersistentWorld(header);
  RCLCPP_INFO(
    get_logger(),
    "Seeded world model with %zu startup block(s).",
    startup.initial_blocks.size());
}

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<PerceptionOrchestratorNode>();
  rclcpp::executors::MultiThreadedExecutor exec(rclcpp::ExecutorOptions(), 4);
  exec.add_node(node);
  exec.spin();

  rclcpp::shutdown();
  return 0;
}
