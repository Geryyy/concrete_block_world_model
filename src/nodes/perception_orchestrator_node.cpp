#include "concrete_block_world_model/nodes/perception_orchestrator_node.hpp"

#include "concrete_block_world_model/utils/block_utils.hpp"
#include "concrete_block_world_model/world_model/config_loader.hpp"

#include <tf2/LinearMath/Quaternion.h>

#define WM_LOG(logger, ...) RCLCPP_INFO(logger, __VA_ARGS__)

PerceptionOrchestratorNode::PerceptionOrchestratorNode()
: Node("block_world_model_node")
{
    run_pose_cb_group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    detector_client_cb_group_ = create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    auto startup = cbpwm::loadWorldModelConfig(*this);
    cbpwm::normalizeWorldModelConfig(get_logger(), startup);
    world_frame_ = startup.world_frame;
    runtime_cfg_.association_max_distance_m = startup.association_max_distance_m;
    runtime_cfg_.association_max_age_s = startup.association_max_age_s;
    runtime_cfg_.min_update_confidence = startup.min_update_confidence;
    runtime_cfg_.scene_discovery_min_detector_confidence =
      startup.scene_discovery_min_detector_confidence;
    runtime_cfg_.scene_discovery_association_max_distance_m =
      startup.scene_discovery_association_max_distance_m;
    detector_discover_service_ = declare_parameter<std::string>(
      "scene_discovery.detector_service", "/concrete_block_detector/discover_blocks");
    scene_discovery_overlay_enabled_ = declare_parameter<bool>(
      "scene_discovery.overlay.enabled", true);
    scene_discovery_overlay_max_image_delta_s_ = declare_parameter<double>(
      "scene_discovery.overlay.max_image_delta_s", 0.08);
    scene_discovery_overlay_fallback_max_image_delta_s_ = declare_parameter<double>(
      "scene_discovery.overlay.fallback_max_image_delta_s", 0.50);
    scene_discovery_capture_enabled_ = declare_parameter<bool>(
      "scene_discovery.capture.enabled", false);
    scene_discovery_capture_dir_ = declare_parameter<std::string>(
      "scene_discovery.capture.dir", "scene_discovery_capture");
    scene_discovery_capture_cloud_topic_ = declare_parameter<std::string>(
      "scene_discovery.capture.cloud_topic", "/seyond/points");
    scene_discovery_capture_cloud_transport_ = declare_parameter<std::string>(
      "scene_discovery.capture.cloud_transport", "cloudini");
    scene_discovery_capture_cloud_max_delta_s_ = declare_parameter<double>(
      "scene_discovery.capture.cloud_max_delta_s", 0.005);
    scene_discovery_registered_priors_enabled_ = declare_parameter<bool>(
      "scene_discovery.priors.registered_blocks.enabled", false);
    scene_discovery_wall_plan_priors_enabled_ = declare_parameter<bool>(
      "scene_discovery.priors.wall_plan.enabled", false);
    scene_discovery_registered_prior_weight_ = declare_parameter<double>(
      "scene_discovery.priors.registered_blocks.weight", 0.20);
    scene_discovery_wall_plan_prior_weight_ = declare_parameter<double>(
      "scene_discovery.priors.wall_plan.weight", 0.15);
    scene_discovery_prior_translation_tolerance_m_ = declare_parameter<double>(
      "scene_discovery.priors.translation_tolerance_m", 0.35);
    scene_discovery_prior_orientation_tolerance_rad_ = declare_parameter<double>(
      "scene_discovery.priors.orientation_tolerance_rad", 0.70);
    if (scene_discovery_overlay_max_image_delta_s_ <= 0.0) {
      RCLCPP_WARN(
        get_logger(),
        "scene_discovery.overlay.max_image_delta_s must be positive; using 0.08 s");
      scene_discovery_overlay_max_image_delta_s_ = 0.08;
    }
    if (scene_discovery_overlay_fallback_max_image_delta_s_ <
      scene_discovery_overlay_max_image_delta_s_)
    {
      RCLCPP_WARN(
        get_logger(),
        "scene_discovery.overlay.fallback_max_image_delta_s must be at least the strict image delta; using %.2f s",
        scene_discovery_overlay_max_image_delta_s_);
      scene_discovery_overlay_fallback_max_image_delta_s_ =
        scene_discovery_overlay_max_image_delta_s_;
    }
    if (scene_discovery_capture_cloud_max_delta_s_ < 0.0) {
      RCLCPP_WARN(
        get_logger(),
        "scene_discovery.capture.cloud_max_delta_s must be non-negative; using 0.005 s");
      scene_discovery_capture_cloud_max_delta_s_ = 0.005;
    }
    if (scene_discovery_registered_prior_weight_ < 0.0 ||
      scene_discovery_wall_plan_prior_weight_ < 0.0 ||
      scene_discovery_prior_translation_tolerance_m_ <= 0.0 ||
      scene_discovery_prior_orientation_tolerance_rad_ <= 0.0)
    {
      throw std::invalid_argument("scene_discovery pose-prior weights and tolerances must be positive");
    }
    task_move_fk_tracking_enabled_ = startup.task_move_fk_tracking_enabled;
    collision_scene_heartbeat_s_ = startup.collision_scene_heartbeat_s;
    refine_grasped_tcp_frame_ = startup.refine_grasped_tcp_frame;
    refine_grasped_camera_info_topic_ = startup.refine_grasped_camera_info_topic;
    scene_discovery_capture_tf_frames_ = startup.scene_discovery_capture_tf_frames;
    block_dimensions_m_ = startup.block_dimensions_m;
    vehicle_box_ = startup.vehicle_box;

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

    if (scene_discovery_overlay_enabled_) {
      const auto debug_image_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();
      scene_discovery_pose_overlay_pub_ = create_publisher<sensor_msgs::msg::Image>(
        "debug/scene_discovery_pose_overlay", debug_image_qos);
      // The overlay is the sole consumer of the CameraInfo cache. Ungated it would subscribe to
      // a camera nothing reads, on a deployment that turned the overlay off.
      camera_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
        refine_grasped_camera_info_topic_,
        rclcpp::SensorDataQoS(),
        std::bind(&PerceptionOrchestratorNode::cameraInfoCallback, this, std::placeholders::_1));
    }

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

    world_pub_ = create_publisher<BlockArray>("block_world_model", 10);
    const auto marker_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();
    marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
      "block_world_model_markers", marker_qos);
    goal_marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
      "block_goal_markers", marker_qos);
    // The contract QoS and the contract name (ros2_interfaces 4): reliable, keep-last-one,
    // transient-local, so a planner that starts later still gets the latched scene. The name is
    // absolute because this node runs under its own namespace and the topic is /crane's.
    collision_scene_pub_ = create_publisher<crane_msgs::msg::CollisionScene>(
      "/crane/collision_scene",
      rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local());

    if (scene_discovery_overlay_enabled_ || scene_discovery_capture_enabled_) {
      scene_discovery_image_sub_ = create_subscription<sensor_msgs::msg::Image>(
        "image",
        rclcpp::SensorDataQoS(),
        std::bind(
          &PerceptionOrchestratorNode::cacheSceneDiscoveryImage,
          this,
          std::placeholders::_1));
    }
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    discover_blocks_client_ = create_client<DiscoverBlocksSrv>(
      detector_discover_service_, rmw_qos_profile_services_default, detector_client_cb_group_);

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
        // Unconditional. This timer carries the collision-scene heartbeat, whose whole job is to
        // keep saying "the world model is alive" on a transient-local topic, so it may not be
        // conditional on what the world holds. It used to be guarded by "blocks or static
        // objects", which enumerated the scene's sources and so missed `vehicle_box`: a blockless
        // site with a configured vehicle published the `truck` primitive once at startup and then
        // went silent, and clear_world_model stopped the heartbeat on any profile. Enumerating at
        // all is the bug -- an empty world is a statement too. This is also the retry for a
        // startup publish lost to the `world` -> K0_mounting_base lookup racing
        // robot_state_publisher.
        const BlockArray snapshot = latestWorldSnapshot();
        std_msgs::msg::Header header;
        header.stamp = now();
        header.frame_id =
          snapshot.header.frame_id.empty() ? world_frame_ : snapshot.header.frame_id;
        publishPersistentWorld(header);
      });

    initializeSeededWorld(startup);

    WM_LOG(
      get_logger(),
      "PerceptionOrchestratorNode ready | every run_pose_estimation mode runs on the detector at "
      "%s | task_move_fk_tracking=%s tcp_frame=%s",
      detector_discover_service_.c_str(),
      task_move_fk_tracking_enabled_ ? "true" : "false",
      refine_grasped_tcp_frame_.c_str());
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

void PerceptionOrchestratorNode::start()
{
  if (!scene_discovery_capture_enabled_ || scene_discovery_cloud_sub_) {
    return;
  }
  scene_discovery_cloud_sub_ = point_cloud_transport::create_subscription(
    shared_from_this(),
    scene_discovery_capture_cloud_topic_,
    std::bind(
      &PerceptionOrchestratorNode::cacheSceneDiscoveryCloud,
      this,
      std::placeholders::_1),
    scene_discovery_capture_cloud_transport_,
    rclcpp::SensorDataQoS().get_rmw_qos_profile());
  RCLCPP_INFO(
    get_logger(),
    "Scene-discovery capture enabled: dir=%s cloud_topic=%s transport=%s",
    scene_discovery_capture_dir_.c_str(), scene_discovery_capture_cloud_topic_.c_str(),
    scene_discovery_capture_cloud_transport_.c_str());
}

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<PerceptionOrchestratorNode>();
  node->start();
  rclcpp::executors::MultiThreadedExecutor exec(rclcpp::ExecutorOptions(), 4);
  exec.add_node(node);
  exec.spin();

  rclcpp::shutdown();
  return 0;
}
