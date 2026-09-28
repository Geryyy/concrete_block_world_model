#pragma once

#include <array>
#include <string>
#include <vector>

#include <Eigen/Geometry>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <std_msgs/msg/header.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "concrete_block_world_model_interfaces/msg/block.hpp"
#include "concrete_block_world_model_interfaces/msg/planning_scene.hpp"
#include "concrete_block_world_model_interfaces/msg/planning_scene_object.hpp"
#include "crane_msgs/msg/collision_scene.hpp"

namespace cbp::world_model
{

enum class OneShotMode
{
  kNone,
  kSceneDiscovery,
  kRefineBlock,
  kRefineGrasped
};

std::string normalizeMode(std::string mode);

OneShotMode parseOneShotMode(const std::string & mode);
const char * oneShotModeToString(OneShotMode mode);

bool isValidTaskTransition(int32_t from_status, int32_t to_status);
const char * taskStatusToString(int32_t status);
bool shouldAssociateByDistance(
  double distance_m,
  double max_distance_m,
  double confidence,
  double min_confidence);

// REFINE_BLOCK: pick the detector observation that re-measures `target`. Returns the index of
// the closest observation that clears `min_confidence` and whose translation and yaw both stay
// inside tolerance, -1 when none does. Yaw differences are folded by pi -- a cuboid at yaw and
// at yaw+pi occupies the same volume, so a flipped re-measurement is still the same block.
// The confidence floor is the one the replaced registration path applied: without it a weak
// spurious hypothesis that happens to be nearest would overwrite a good measured pose.
int selectRefineMatch(
  const concrete_block_world_model_interfaces::msg::Block & target,
  const std::vector<concrete_block_world_model_interfaces::msg::Block> & observations,
  double translation_tolerance_m,
  double yaw_tolerance_rad,
  double min_confidence);

// Where a carried block is, given the TCP and the grasp offset: the pose FK tracking
// publishes for a TASK_MOVE block, and the prior REFINE_GRASPED sends the detector.
geometry_msgs::msg::Pose poseFromGraspOffset(
  const Eigen::Matrix4d & T_world_tcp, const Eigen::Matrix4d & T_tcp_block);

// The inverse: the grasp offset that makes FK tracking reproduce `block_pose`. This is how a
// measurement of a carried block is kept -- writing the pose alone would be overwritten by the
// next FK update, and the corrected offset also carries the measurement forward as the crane
// moves on.
Eigen::Matrix4d graspOffsetFromPose(
  const Eigen::Matrix4d & T_world_tcp, const geometry_msgs::msg::Pose & block_pose);

visualization_msgs::msg::MarkerArray buildWorldMarkers(
  const std_msgs::msg::Header & header,
  const std::vector<concrete_block_world_model_interfaces::msg::Block> & blocks,
  const std::vector<concrete_block_world_model_interfaces::msg::PlanningSceneObject> &
  static_objects,
  const std::string & world_frame,
  const std::array<double, 3> & block_dimensions_m);

// Opaque cubes at each block's assembly goal_pose (where goal_status == GOAL_SET),
// for visualizing the target wall alongside the live (translucent) blocks.
visualization_msgs::msg::MarkerArray buildGoalMarkers(
  const std_msgs::msg::Header & header,
  const std::vector<concrete_block_world_model_interfaces::msg::Block> & blocks,
  const std::string & world_frame,
  const std::array<double, 3> & block_dimensions_m);

// Ids crane_msgs reserves (wiki/implementation/ros2_interfaces.md 6). The planner refuses a
// scene *whole* when it carries `payload`, and reads `truck` as a vehicle pose it expands into
// bed and runges -- neither is something this world model may emit as an ordinary obstacle.
extern const char kReservedPayloadId[];
extern const char kReservedTruckId[];

// Outcome of toCollisionScene: the message plus one line per object that did not make it in.
struct CollisionSceneConversion
{
  crane_msgs::msg::CollisionScene scene;
  std::vector<std::string> dropped;
};

// The vehicle the crane is bolted to, as one configured box: pose at the vehicle's centre,
// `dimensions` its extent in its own axes with x along the bed and z up. It leaves the world
// model as the single primitive with the reserved id `truck`, which the planner expands into a
// bed slab at the box's top face, the six runges of trajectory_planning 4.2 standing on it, and
// the headboard closing its cab end.
// Publishing the vehicle as ordinary boxes instead would lose the runges and would report the
// mounting base colliding with the vehicle it is part of.
//
// `pose` is a static configured value and not a measurement -- the seam a vehicle-pose estimator
// later replaces, with no planner change, because the planner keys every expanded piece to
// whatever pose the primitive carries.
struct VehicleBoxConfig
{
  bool enabled{false};
  std::string frame_id{};
  std::array<double, 3> position{0.0, 0.0, 0.0};
  std::array<double, 3> rpy_deg{0.0, 0.0, 0.0};
  std::array<double, 3> dimensions{0.0, 0.0, 0.0};
  // Check only, no geometry: `crane_planning` owns the runge model and refuses a scene whose bed
  // is too short for it. These two mirror its `truck.runge_dimensions` x component and its
  // `truck.runge_stations`, so a box that does not fit is reported where it is configured
  // instead of arriving at the planner as a refused scene. A non-positive length or an empty
  // station list turns the check off.
  double runge_length_m{0.0};
  std::vector<double> runge_stations_m{};
};

// Convert a planning-scene snapshot into the planner's collision scene, in the frame the given
// transform targets (`K0_mounting_base`; ros2_interfaces 4). Pure: no node, no clock, no graph.
//
// `mounting_base_from_world` is the transform as tf2 returns it for
// lookupTransform(K0_mounting_base, world): header.frame_id is the target frame and becomes the
// scene's frame, child_frame_id is the frame the objects are expected to be in.
//
// Objects are filtered here because the planner refuses a scene whole on any bad primitive, so
// one bad detection would otherwise blank the entire obstacle set. Dropped: an empty id, an id
// already seen, a reserved id, a shape the planner has no enumerator for, a non-positive or
// non-finite extent, a pose that is not finite or whose quaternion has no usable direction, and
// an object in a frame the transform does not come from. Repaired rather than dropped: a
// finite, non-zero quaternion that is not unit length is normalized.
//
// `vehicle_box`, when enabled, is emitted first as the reserved `truck` primitive, `structural`
// and outside the object stream -- the object stream drops that id, and this is the one thing
// allowed to carry it. A vehicle box that cannot be emitted is reported in `dropped` rather than
// emitted silently: a bad frame, a non-positive or non-finite extent, an unusable pose, or a
// runge station that the configured bed is too short to carry.
CollisionSceneConversion toCollisionScene(
  const concrete_block_world_model_interfaces::msg::PlanningScene & planning_scene,
  const geometry_msgs::msg::TransformStamped & mounting_base_from_world,
  const VehicleBoxConfig & vehicle_box = VehicleBoxConfig{});

}  // namespace cbp::world_model
