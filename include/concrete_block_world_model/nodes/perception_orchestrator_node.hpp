#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <filesystem>
#include <future>
#include <limits>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <cv_bridge/cv_bridge.h>
#include <Eigen/Dense>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <std_msgs/msg/float64.hpp>
#include <tf2/exceptions.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <visualization_msgs/msg/marker_array.hpp>

#include "concrete_block_perception_interfaces/action/register_block.hpp"
#include "concrete_block_world_model_interfaces/msg/block.hpp"
#include "concrete_block_world_model_interfaces/msg/block_array.hpp"
#include "concrete_block_world_model_interfaces/msg/planning_scene.hpp"
#include "concrete_block_world_model_interfaces/msg/planning_scene_object.hpp"
#include "concrete_block_world_model_interfaces/srv/get_coarse_blocks.hpp"
#include "concrete_block_world_model_interfaces/srv/get_planning_scene.hpp"
#include "concrete_block_perception_interfaces/srv/extract_mask_cutout.hpp"
#include "concrete_block_world_model_interfaces/srv/run_pose_estimation.hpp"
#include "concrete_block_world_model_interfaces/srv/set_perception_mode.hpp"
#include "concrete_block_world_model_interfaces/srv/set_block_task_status.hpp"
#include "concrete_block_world_model_interfaces/srv/upsert_block.hpp"
#include "concrete_block_world_model/utils/coarse_pose_utils.hpp"
#include "concrete_block_world_model/world_model/config_loader.hpp"
#include "concrete_block_world_model/world_model/continuous_perception.hpp"
#include "concrete_block_world_model/world_model/refine_flow.hpp"
#include "concrete_block_world_model/world_model/scene_discovery_flow.hpp"
#include "concrete_block_world_model/world_model/state_manager.hpp"
#include "ros2_yolos_cpp/srv/segment_image.hpp"

using concrete_block_world_model_interfaces::msg::Block;
using concrete_block_world_model_interfaces::msg::BlockArray;
using concrete_block_world_model_interfaces::msg::PlanningScene;
using concrete_block_world_model_interfaces::msg::PlanningSceneObject;

namespace cbpwm = cbp::world_model;

class PerceptionOrchestratorNode : public rclcpp::Node
{
  using SegmentSrv = ros2_yolos_cpp::srv::SegmentImage;
  using SetBlockTaskStatusSrv = concrete_block_world_model_interfaces::srv::SetBlockTaskStatus;
  using GetCoarseSrv = concrete_block_world_model_interfaces::srv::GetCoarseBlocks;
  using GetPlanningSceneSrv = concrete_block_world_model_interfaces::srv::GetPlanningScene;
  using ExtractMaskCutoutSrv = concrete_block_perception_interfaces::srv::ExtractMaskCutout;
  using RunPoseSrv = concrete_block_world_model_interfaces::srv::RunPoseEstimation;
  using SetPerceptionModeSrv = concrete_block_world_model_interfaces::srv::SetPerceptionMode;
  using UpsertBlockSrv = concrete_block_world_model_interfaces::srv::UpsertBlock;
  using RegisterBlock = concrete_block_perception_interfaces::action::RegisterBlock;
  using GoalHandleRegisterBlock = rclcpp_action::ClientGoalHandle<RegisterBlock>;

  using SyncPolicy = message_filters::sync_policies::ApproximateTime<
    sensor_msgs::msg::Image,
    sensor_msgs::msg::PointCloud2>;

  struct OneShotRequest
  {
    uint64_t sequence{0};
    cbpwm::OneShotMode mode{cbpwm::OneShotMode::kNone};
    std::string target_block_id;
    bool enable_debug{true};
    double registration_timeout_s{3.0};
  };

  struct CameraIntrinsics
  {
    bool valid{false};
    double fx{0.0};
    double fy{0.0};
    double cx{0.0};
    double cy{0.0};
    uint32_t width{0};
    uint32_t height{0};
  };

  enum class PerceptionMode
  {
    kIdle,
    kContinuous
  };

  struct ContinuousConfig
  {
    int process_every_n_frames{3};
    double segmentation_timeout_s{1.0};
    double cutout_timeout_s{1.0};
    int min_mask_pixels{2000};
    double min_mask_fill_ratio{0.15};
    int min_valid_cloud_points{120};
    bool mask_merge_enabled{true};
    double mask_merge_max_centroid_distance_m{0.6};
    bool registration_enabled{false};
    double registration_timeout_s{3.0};
    int registration_max_per_frame{1};
    double association_max_distance_m{0.8};
    double association_max_age_s{20.0};
  };

  struct ContinuousStageTimings
  {
    int64_t cutout_ms{0};
    int64_t coarse_ms{0};
    int64_t registration_ms{0};
    int64_t upsert_ms{0};
  };

  struct RuntimeConfig
  {
    double min_fitness{0.3};
    double max_rmse{0.05};
    double max_sync_delta_s{0.06};
    double object_timeout_s{10.0};
    double association_max_distance_m{0.45};
    double association_max_age_s{20.0};
    double min_update_confidence{0.25};
    double refine_target_max_distance_m{1.2};
  };

public:
  PerceptionOrchestratorNode();

private:
  void resetPerfCounters();
  std::string resolveGraspedBlockId();
  cbpwm::AssociationConfig associationConfig() const;
  cbpwm::CoarsePoseConfig coarsePoseConfig(int min_points_override = -1) const;
  bool buildCoarseBlockFromMaskAndCloud(
    uint32_t detection_id,
    const sensor_msgs::msg::Image & mask_msg,
    const sensor_msgs::msg::PointCloud2 & cloud_msg,
    const std_msgs::msg::Header & header,
    const Eigen::Vector3d * camera_origin_world,
    Block & out_block,
    std::string & reason) const;
  bool buildCoarseBlockFromCloudCentroid(
    uint32_t detection_id,
    const sensor_msgs::msg::Image & mask_msg,
    const sensor_msgs::msg::PointCloud2 & cutout_cloud_msg,
    const std_msgs::msg::Header & header,
    const Eigen::Vector3d * camera_origin_world,
    Block & out_block,
    std::string & reason,
    int min_points_override = -1) const;
  bool extractMaskCutoutSync(
    const sensor_msgs::msg::Image & mask,
    const sensor_msgs::msg::PointCloud2 & cloud,
    double timeout_s,
    sensor_msgs::msg::PointCloud2 & out_cutout_cloud,
    std::string & reason);
  bool lookupFrameOriginInWorld(
    const std_msgs::msg::Header & stamped_header,
    Eigen::Vector3d & origin_world,
    std::string & reason);
  void resetBusy();
  void updateLatestWorldCache(const BlockArray & out);
  BlockArray latestWorldSnapshot();
  PlanningScene latestPlanningSceneSnapshot();
  PlanningScene buildPlanningSceneSnapshot(const std_msgs::msg::Header & header, const std::vector<Block> & blocks);
  std::vector<PlanningSceneObject> staticSceneObjectsInWorld() const;
  bool runRegistrationSync(
    uint32_t detection_id,
    const sensor_msgs::msg::Image & mask,
    const sensor_msgs::msg::PointCloud2 & cloud,
    const std_msgs::msg::Header & header,
    double timeout_s,
    Block & out_block,
    std::string & reason,
    const std::string & object_class_override = "");
  bool runSegmentationSync(
    const sensor_msgs::msg::Image & image,
    double timeout_s,
    SegmentSrv::Response::SharedPtr & out_response,
    std::string & reason);
  bool trySceneDiscoveryCoarseFallback(
    uint32_t detection_id,
    const std::string & registration_reason,
    const sensor_msgs::msg::Image & mask,
    const sensor_msgs::msg::PointCloud2 & cloud,
    const cbpwm::SceneFlowRequest & run_request,
    const Eigen::Vector3d * camera_origin_world,
    cbpwm::RegistrationCounters & counters);
  static Eigen::Matrix4d transformToEigen(const geometry_msgs::msg::TransformStamped & tf);
  void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);
  bool lookupPredictedGraspedPose(
    const std_msgs::msg::Header & header,
    Eigen::Vector3d & p_world,
    Eigen::Vector3d & p_camera,
    Eigen::Quaterniond & q_world,
    std::string & reason);
  bool resolveCameraFrame(
    const std_msgs::msg::Header & header,
    std::string & camera_frame,
    std::string & reason);
  bool worldPointToCamera(
    const std_msgs::msg::Header & header,
    const Eigen::Vector3d & p_world,
    Eigen::Vector3d & p_camera,
    std::string & reason);
  cbpwm::RefineFlowRuntime makeRefineFlowRuntime();
  void processRefineGraspedWithFkRoi(
    const sensor_msgs::msg::Image::ConstSharedPtr & image,
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr & cloud,
    const OneShotRequest & run_request,
    const std::chrono::steady_clock::time_point & t_start);
  bool tryProcessRefineBlockWithPoseRoi(
    const sensor_msgs::msg::Image::ConstSharedPtr & image,
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr & cloud,
    const OneShotRequest & run_request,
    const std::chrono::steady_clock::time_point & t_start);

  void completeOneShotRequest(uint64_t sequence, bool success, const std::string & message);
  void handleRunPoseEstimation(
    const std::shared_ptr<RunPoseSrv::Request> request,
    std::shared_ptr<RunPoseSrv::Response> response);
  void handleGetCoarseBlocks(
    const std::shared_ptr<GetCoarseSrv::Request> request,
    std::shared_ptr<GetCoarseSrv::Response> response);
  void handleGetPlanningScene(
    const std::shared_ptr<GetPlanningSceneSrv::Request> request,
    std::shared_ptr<GetPlanningSceneSrv::Response> response);
  void handleSetBlockTaskStatus(
    const std::shared_ptr<SetBlockTaskStatusSrv::Request> request,
    std::shared_ptr<SetBlockTaskStatusSrv::Response> response);
  void handleSetPerceptionMode(
    const std::shared_ptr<SetPerceptionModeSrv::Request> request,
    std::shared_ptr<SetPerceptionModeSrv::Response> response);
  void handleUpsertBlock(
    const std::shared_ptr<UpsertBlockSrv::Request> request,
    std::shared_ptr<UpsertBlockSrv::Response> response);
  void publishWorldMarkers(const std_msgs::msg::Header & header, const std::vector<Block> & blocks);
  void publishPersistentWorld(const std_msgs::msg::Header & header);
  void publishDetectionOverlay(
    const sensor_msgs::msg::Image::ConstSharedPtr & image,
    const vision_msgs::msg::Detection2DArray & detections,
    const sensor_msgs::msg::Image & mask_msg);
  void recordTiming(int64_t seg_ms, int64_t track_ms, int64_t reg_ms, int64_t total_ms);
  void initializeSeededWorld(const cbpwm::WorldModelConfig & startup);
  void syncCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr image,
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr cloud);
  void handleOneShotSegmentationResponse(
    rclcpp::Client<SegmentSrv>::SharedFuture seg_future,
    const sensor_msgs::msg::Image::ConstSharedPtr & image,
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr & cloud,
    std::chrono::steady_clock::time_point t_start,
    const OneShotRequest & run_request);
  void handleContinuousSegmentationResponse(
    rclcpp::Client<SegmentSrv>::SharedFuture seg_future,
    const sensor_msgs::msg::Image::ConstSharedPtr & image,
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr & cloud,
    std::chrono::steady_clock::time_point t_start);
  void processContinuousFrame(
    const sensor_msgs::msg::Image::ConstSharedPtr & image,
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr & cloud);
  std::vector<cbpwm::ContinuousMaskCandidate> buildContinuousCandidates(
    const SegmentSrv::Response & seg_res,
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr & cloud,
    const cv::Mat & full_mask,
    const Eigen::Vector3d * camera_origin_world,
    ContinuousStageTimings & timings,
    cv::Mat & rejected_mask,
    size_t & rejected_count);
  bool buildContinuousObservation(
    const cbpwm::ContinuousMaskGroup & group,
    size_t group_index,
    const std::vector<cbpwm::ContinuousMaskCandidate> & candidates,
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr & cloud,
    const Eigen::Vector3d * camera_origin_world,
    int & registration_attempts,
    ContinuousStageTimings & timings,
    cbpwm::BlockObservation & out_observation);
  // World-update seam for the continuous stream: a future probabilistic
  // filter replaces this association + overwrite-upsert.
  bool applyContinuousObservation(
    const cbpwm::BlockObservation & observation,
    const std_msgs::msg::Header & header,
    ContinuousStageTimings & timings,
    std::string & assigned_id,
    std::string & reason);
  void processFrame(
    const sensor_msgs::msg::Image::ConstSharedPtr & image,
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr & cloud);

  message_filters::Subscriber<sensor_msgs::msg::Image> image_sub_;
  message_filters::Subscriber<sensor_msgs::msg::PointCloud2> cloud_sub_;
  std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;

  rclcpp::Client<SegmentSrv>::SharedPtr segment_client_;
  rclcpp::Client<ExtractMaskCutoutSrv>::SharedPtr extract_mask_cutout_client_;
  rclcpp_action::Client<RegisterBlock>::SharedPtr action_client_;
  rclcpp::Service<SetBlockTaskStatusSrv>::SharedPtr set_block_task_status_srv_;
  rclcpp::Service<UpsertBlockSrv>::SharedPtr upsert_block_srv_;
  rclcpp::Service<GetCoarseSrv>::SharedPtr get_coarse_srv_;
  rclcpp::Service<GetPlanningSceneSrv>::SharedPtr get_planning_scene_srv_;
  rclcpp::Service<RunPoseSrv>::SharedPtr run_pose_srv_;
  rclcpp::Service<SetPerceptionModeSrv>::SharedPtr set_perception_mode_srv_;
  rclcpp::CallbackGroup::SharedPtr run_pose_cb_group_;
  rclcpp::CallbackGroup::SharedPtr action_client_cb_group_;
  rclcpp::TimerBase::SharedPtr marker_refresh_timer_;

  rclcpp::Publisher<BlockArray>::SharedPtr world_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  size_t last_published_block_count_{0};
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr det_debug_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr yolo_service_debug_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr continuous_merged_mask_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr refine_grasped_roi_input_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr continuous_timing_seg_ms_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr continuous_timing_cutout_ms_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr continuous_timing_coarse_ms_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr continuous_timing_registration_ms_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr continuous_timing_upsert_ms_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr continuous_timing_total_ms_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr continuous_timing_detections_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr continuous_timing_accepted_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr continuous_timing_rejected_pub_;

  std::unordered_map<std::string, Block> persistent_world_;
  std::mutex persistent_world_mutex_;
  std::unordered_set<std::string> seeded_block_ids_;

  BlockArray latest_world_;
  std::mutex latest_world_mutex_;
  PlanningScene latest_planning_scene_;
  std::mutex latest_planning_scene_mutex_;

  std::atomic<bool> busy_{false};
  std::atomic<uint64_t> dropped_busy_frames_{0};
  std::atomic<uint64_t> dropped_sync_frames_{0};

  std::string object_class_;
  std::string world_frame_{"world"};
  std::vector<PlanningSceneObject> static_scene_objects_;
  std::array<double, 3> block_dimensions_m_{0.6, 0.9, 0.6};
  RuntimeConfig runtime_cfg_;
  ContinuousConfig continuous_cfg_;
  std::atomic<PerceptionMode> perception_mode_{PerceptionMode::kIdle};
  std::atomic<uint64_t> continuous_seen_frames_{0};
  bool protect_task_blocks_from_timeout_{true};
  std::atomic<bool> debug_detection_overlay_enabled_{true};
  std::atomic<bool> debug_refine_grasped_roi_input_enabled_{true};
  bool perf_log_timing_enabled_{true};
  int perf_log_every_n_frames_{20};
  uint64_t world_block_counter_{0};

  std::mutex perf_mutex_;
  uint64_t perf_timing_count_{0};
  int64_t perf_seg_sum_ms_{0};
  int64_t perf_track_sum_ms_{0};
  int64_t perf_reg_sum_ms_{0};
  int64_t perf_total_sum_ms_{0};

  std::mutex one_shot_mutex_;
  std::condition_variable one_shot_cv_;
  OneShotRequest active_one_shot_;
  uint64_t one_shot_sequence_counter_{0};
  uint64_t one_shot_done_sequence_{0};
  bool one_shot_last_success_{false};
  std::string one_shot_last_message_;

  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  std::mutex camera_info_mutex_;
  CameraIntrinsics camera_intrinsics_;
  std::string camera_info_frame_id_;
  bool refine_grasped_use_fk_roi_{true};
  std::string refine_grasped_tcp_frame_{"elastic/K8_tool_center_point"};
  std::string refine_grasped_camera_frame_{};
  std::string refine_grasped_camera_info_topic_{"/zed2i/warped/left/camera_info"};
  Eigen::Matrix4d T_tcp_block_{Eigen::Matrix4d::Identity()};
  cbpwm::RoiInputConfig refine_grasped_roi_cfg_;
  bool refine_block_use_pose_roi_{false};
  cbpwm::RoiInputConfig refine_block_roi_cfg_;
  bool scene_discovery_coarse_fallback_enabled_{true};
  int scene_discovery_coarse_fallback_min_points_{120};
  double coarse_surface_square_ratio_thresh_{1.35};
  double coarse_front_center_offset_square_m_{0.45};
  double coarse_front_center_offset_rect_m_{0.30};
  cbpwm::PoseFusionConfig refine_grasped_pose_fusion_;
};
