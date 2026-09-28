#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <geometry_msgs/msg/point.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include "concrete_block_world_model/utils/block_utils.hpp"
#include "concrete_block_world_model/utils/world_model_utils.hpp"

namespace cbp::world_model
{

using concrete_block_world_model_interfaces::msg::Block;
using concrete_block_world_model_interfaces::msg::PlanningSceneObject;

namespace
{

constexpr double kPi = 3.14159265358979323846;

// Yaw of the pose's z-rotation component, the only orientation a block on (roughly) level
// ground is free in.
double poseYaw(const geometry_msgs::msg::Pose & pose)
{
  const auto & q = pose.orientation;
  return std::atan2(
    2.0 * (q.w * q.z + q.x * q.y),
    1.0 - 2.0 * (q.y * q.y + q.z * q.z));
}

builtin_interfaces::msg::Duration markerLifetime(double seconds)
{
  builtin_interfaces::msg::Duration lifetime;
  lifetime.sec = static_cast<int32_t>(seconds);
  lifetime.nanosec =
    static_cast<uint32_t>((seconds - static_cast<double>(lifetime.sec)) * 1'000'000'000.0);
  return lifetime;
}

// The snapshot's shape enumerator as the planner's. The two lists share the numeric value of a
// box, and the conversion is written out rather than cast so that a shape either message adds
// later becomes a dropped primitive here instead of whatever the cast landed on.
bool collisionShapeFromPlanningScene(uint8_t shape_type, uint8_t & shape)
{
  if (shape_type == PlanningSceneObject::SHAPE_BOX) {
    shape = crane_msgs::msg::CollisionScene::SHAPE_BOX;
    return true;
  }
  return false;
}

// `dimensions` is the full extent per axis in both messages -- never a half-extent and never a
// radius -- so a zero anywhere is a primitive with no thickness rather than a shorthand.
bool hasPositiveExtent(const geometry_msgs::msg::Vector3 & dimensions)
{
  const std::array<double, 3> axes{dimensions.x, dimensions.y, dimensions.z};
  for (const double extent : axes) {
    if (!std::isfinite(extent) || extent <= 0.0) {
      return false;
    }
  }
  return true;
}

// A pose the planner will accept, with a non-unit quaternion repaired rather than dropped.
bool normalizedPose(const geometry_msgs::msg::Pose & pose, geometry_msgs::msg::Pose & out)
{
  const std::array<double, 7> values{
    pose.position.x, pose.position.y, pose.position.z,
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w};
  for (const double value : values) {
    if (!std::isfinite(value)) {
      return false;
    }
  }
  const double norm = std::sqrt(
    pose.orientation.x * pose.orientation.x + pose.orientation.y * pose.orientation.y +
    pose.orientation.z * pose.orientation.z + pose.orientation.w * pose.orientation.w);
  if (!(norm > 0.0)) {
    return false;
  }
  out = pose;
  out.orientation.x /= norm;
  out.orientation.y /= norm;
  out.orientation.z /= norm;
  out.orientation.w /= norm;
  return true;
}

// Metres the way crane_planning writes them in its refusals, so the two messages read alike.
std::string metres(double value)
{
  std::ostringstream text;
  text << std::fixed << std::setprecision(3) << value << " m";
  return text.str();
}

// The configured vehicle box as the reserved `truck` primitive, in the frame the transform
// targets. Returns false and fills `report` when the box must not be emitted.
bool truckPrimitiveFromVehicleBox(
  const VehicleBoxConfig & vehicle_box,
  const geometry_msgs::msg::TransformStamped & mounting_base_from_world,
  crane_msgs::msg::CollisionPrimitive & primitive,
  std::string & report)
{
  const std::string & source_frame = mounting_base_from_world.child_frame_id;
  if (!vehicle_box.frame_id.empty() && vehicle_box.frame_id != source_frame) {
    report = "the vehicle box is in frame '" + vehicle_box.frame_id +
      "' and the transform comes from '" + source_frame + "'";
    return false;
  }

  geometry_msgs::msg::Vector3 dimensions;
  dimensions.x = vehicle_box.dimensions[0];
  dimensions.y = vehicle_box.dimensions[1];
  dimensions.z = vehicle_box.dimensions[2];
  if (!hasPositiveExtent(dimensions)) {
    report = "the vehicle box has no finite, positive extent along every axis";
    return false;
  }

  geometry_msgs::msg::Pose configured;
  configured.position.x = vehicle_box.position[0];
  configured.position.y = vehicle_box.position[1];
  configured.position.z = vehicle_box.position[2];
  for (const double angle : vehicle_box.rpy_deg) {
    if (!std::isfinite(angle)) {
      report = "the vehicle box orientation is not finite";
      return false;
    }
  }
  constexpr double kDegToRad = 3.14159265358979323846 / 180.0;
  tf2::Quaternion quaternion;
  quaternion.setRPY(
    vehicle_box.rpy_deg[0] * kDegToRad,
    vehicle_box.rpy_deg[1] * kDegToRad,
    vehicle_box.rpy_deg[2] * kDegToRad);
  configured.orientation.x = quaternion.x();
  configured.orientation.y = quaternion.y();
  configured.orientation.z = quaternion.z();
  configured.orientation.w = quaternion.w();

  geometry_msgs::msg::Pose normalized;
  if (!normalizedPose(configured, normalized)) {
    report = "the vehicle box pose is not finite or its quaternion has no direction";
    return false;
  }

  // The planner refuses a station that hangs a runge off the end of the bed, naming both
  // numbers. A box that trips that is a model of a different vehicle, so it is reported here --
  // where the box is configured -- instead of blanking the whole scene at the planner.
  if (vehicle_box.runge_length_m > 0.0 && !vehicle_box.runge_stations_m.empty()) {
    for (std::size_t station = 0; station < vehicle_box.runge_stations_m.size(); ++station) {
      const double x = vehicle_box.runge_stations_m[station];
      if (!std::isfinite(x) ||
        std::abs(x) + 0.5 * vehicle_box.runge_length_m > 0.5 * dimensions.x)
      {
        report = "runge station " + std::to_string(station) + " sits at " + metres(x) +
          " from the centre of a bed " + metres(dimensions.x) +
          " long, so the runge would hang off the end of it";
        return false;
      }
    }
  }

  primitive.id = kReservedTruckId;
  primitive.shape = crane_msgs::msg::CollisionScene::SHAPE_BOX;
  tf2::doTransform(normalized, primitive.pose, mounting_base_from_world);
  primitive.dimensions = dimensions;
  // From a model, never perceived: this is the truck model of trajectory_planning 4.2.
  primitive.structural = true;
  return true;
}

}  // namespace

std::string normalizeMode(std::string mode)
{
  std::transform(mode.begin(), mode.end(), mode.begin(), [](unsigned char c) {
    return static_cast<char>(std::toupper(c));
  });
  return mode;
}

OneShotMode parseOneShotMode(const std::string & mode)
{
  const std::string m = normalizeMode(mode);
  if (m == "SCENE_DISCOVERY") {
    return OneShotMode::kSceneDiscovery;
  }
  if (m == "REFINE_BLOCK") {
    return OneShotMode::kRefineBlock;
  }
  if (m == "REFINE_GRASPED") {
    return OneShotMode::kRefineGrasped;
  }
  return OneShotMode::kNone;
}

const char * oneShotModeToString(OneShotMode mode)
{
  switch (mode) {
    case OneShotMode::kSceneDiscovery:
      return "SCENE_DISCOVERY";
    case OneShotMode::kRefineBlock:
      return "REFINE_BLOCK";
    case OneShotMode::kRefineGrasped:
      return "REFINE_GRASPED";
    case OneShotMode::kNone:
    default:
      return "NONE";
  }
}

const char * taskStatusToString(int32_t status)
{
  switch (status) {
    case Block::TASK_FREE:
      return "TASK_FREE";
    case Block::TASK_MOVE:
      return "TASK_MOVE";
    case Block::TASK_PLACED:
      return "TASK_PLACED";
    case Block::TASK_REMOVED:
      return "TASK_REMOVED";
    case Block::TASK_UNKNOWN:
    default:
      return "TASK_UNKNOWN";
  }
}

bool isValidTaskTransition(int32_t from_status, int32_t to_status)
{
  if (from_status == to_status) {
    return true;
  }
  if (from_status == Block::TASK_UNKNOWN) {
    return true;
  }

  switch (from_status) {
    case Block::TASK_FREE:
      return to_status == Block::TASK_MOVE || to_status == Block::TASK_REMOVED;
    case Block::TASK_MOVE:
      return to_status == Block::TASK_FREE || to_status == Block::TASK_PLACED;
    case Block::TASK_PLACED:
      return to_status == Block::TASK_MOVE || to_status == Block::TASK_REMOVED;
    case Block::TASK_REMOVED:
      return false;
    case Block::TASK_UNKNOWN:
    default:
      return true;
  }
}

bool shouldAssociateByDistance(
  double distance_m,
  double max_distance_m,
  double confidence,
  double min_confidence)
{
  if (!std::isfinite(distance_m) || !std::isfinite(max_distance_m) ||
    !std::isfinite(confidence) || !std::isfinite(min_confidence))
  {
    return false;
  }
  if (distance_m < 0.0 || max_distance_m <= 0.0) {
    return false;
  }
  if (confidence < min_confidence) {
    return false;
  }
  return distance_m <= max_distance_m;
}

int selectRefineMatch(
  const Block & target,
  const std::vector<Block> & observations,
  double translation_tolerance_m,
  double yaw_tolerance_rad,
  double min_confidence)
{
  if (!std::isfinite(yaw_tolerance_rad) || yaw_tolerance_rad < 0.0) {
    return -1;
  }

  const double target_yaw = poseYaw(target.pose);
  int best = -1;
  double best_distance = std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < observations.size(); ++i) {
    const double distance = poseDistance(observations[i].pose, target.pose);
    if (distance >= best_distance ||
      !shouldAssociateByDistance(
        distance, translation_tolerance_m, observations[i].confidence, min_confidence))
    {
      continue;
    }
    double yaw_delta = std::fmod(std::abs(poseYaw(observations[i].pose) - target_yaw), kPi);
    if (yaw_delta > 0.5 * kPi) {
      yaw_delta = kPi - yaw_delta;
    }
    if (!std::isfinite(yaw_delta) || yaw_delta > yaw_tolerance_rad) {
      continue;
    }
    best_distance = distance;
    best = static_cast<int>(i);
  }
  return best;
}

geometry_msgs::msg::Pose poseFromGraspOffset(
  const Eigen::Matrix4d & T_world_tcp, const Eigen::Matrix4d & T_tcp_block)
{
  const Eigen::Matrix4d T_world_block = T_world_tcp * T_tcp_block;
  const Eigen::Quaterniond q(Eigen::Matrix3d(T_world_block.block<3, 3>(0, 0)));
  const Eigen::Quaterniond q_n = q.normalized();
  geometry_msgs::msg::Pose pose;
  pose.position.x = T_world_block(0, 3);
  pose.position.y = T_world_block(1, 3);
  pose.position.z = T_world_block(2, 3);
  pose.orientation.x = q_n.x();
  pose.orientation.y = q_n.y();
  pose.orientation.z = q_n.z();
  pose.orientation.w = q_n.w();
  return pose;
}

Eigen::Matrix4d graspOffsetFromPose(
  const Eigen::Matrix4d & T_world_tcp, const geometry_msgs::msg::Pose & block_pose)
{
  Eigen::Matrix4d T_world_block = Eigen::Matrix4d::Identity();
  T_world_block.block<3, 3>(0, 0) =
    Eigen::Quaterniond(
    block_pose.orientation.w, block_pose.orientation.x,
    block_pose.orientation.y, block_pose.orientation.z).normalized().toRotationMatrix();
  T_world_block.block<3, 1>(0, 3) =
    Eigen::Vector3d(block_pose.position.x, block_pose.position.y, block_pose.position.z);
  return T_world_tcp.inverse() * T_world_block;
}

visualization_msgs::msg::MarkerArray buildWorldMarkers(
  const std_msgs::msg::Header & header,
  const std::vector<Block> & blocks,
  const std::vector<PlanningSceneObject> & static_objects,
  const std::string & world_frame,
  const std::array<double, 3> & block_dimensions_m)
{
  visualization_msgs::msg::MarkerArray ma;
  auto marker_header = header;
  marker_header.frame_id = world_frame;

  visualization_msgs::msg::Marker clear;
  clear.header = marker_header;
  clear.ns = "";
  clear.id = 0;
  clear.type = visualization_msgs::msg::Marker::CUBE;
  clear.action = visualization_msgs::msg::Marker::DELETEALL;
  ma.markers.push_back(clear);

  int marker_id = 1;
  const auto dynamic_marker_lifetime = markerLifetime(2.0);
  for (const auto & object : static_objects) {
    visualization_msgs::msg::Marker marker;
    marker.header = marker_header;
    marker.ns = "cbp_static_scene";
    marker.id = marker_id++;
    marker.type = visualization_msgs::msg::Marker::CUBE;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose = object.pose;
    marker.scale.x = object.dimensions.x;
    marker.scale.y = object.dimensions.y;
    marker.scale.z = object.dimensions.z;
    marker.color.r = 0.2f;
    marker.color.g = 0.55f;
    marker.color.b = 0.95f;
    marker.color.a = 0.28f;
    ma.markers.push_back(std::move(marker));

    visualization_msgs::msg::Marker label;
    label.header = marker_header;
    label.ns = "cbp_static_scene_ids";
    label.id = marker_id++;
    label.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    label.action = visualization_msgs::msg::Marker::ADD;
    label.pose = object.pose;
    label.pose.position.z += 0.5 * object.dimensions.z + 0.2;
    label.scale.z = 0.18;
    label.color.r = 0.7f;
    label.color.g = 0.9f;
    label.color.b = 1.0f;
    label.color.a = 0.9f;
    label.text = object.id;
    ma.markers.push_back(std::move(label));
  }

  for (const auto & b : blocks) {
    // Goal-only placeholders (no actual pose yet) are shown via goal markers, not here.
    if (b.pose_status == Block::POSE_UNKNOWN && b.goal_status == Block::GOAL_SET) {
      continue;
    }
    visualization_msgs::msg::Marker m;
    m.header = marker_header;
    m.ns = "cbp_blocks";
    m.id = marker_id++;
    m.type = (b.pose_status == Block::POSE_COARSE) ?
      visualization_msgs::msg::Marker::SPHERE :
      visualization_msgs::msg::Marker::CUBE;
    m.action = visualization_msgs::msg::Marker::ADD;
    m.lifetime = dynamic_marker_lifetime;
    m.pose = b.pose;
    m.scale.x = block_dimensions_m[0];
    m.scale.y = block_dimensions_m[1];
    m.scale.z = block_dimensions_m[2];

    // Pose status takes precedence in marker color so coarse/precise is always visible in RViz.
    if (b.pose_status == Block::POSE_PRECISE) {
      m.color.r = 0.1f;
      m.color.g = 0.8f;
      m.color.b = 0.2f;
    } else if (b.pose_status == Block::POSE_COARSE) {
      m.color.r = 1.0f;
      m.color.g = 0.8f;
      m.color.b = 0.1f;
    } else {
      m.color.r = 0.5f;
      m.color.g = 0.5f;
      m.color.b = 0.5f;
    }
    m.color.a = 0.6f;

    // Keep some task-state visibility via alpha changes without hiding pose status color.
    if (b.task_status == Block::TASK_REMOVED) {
      m.color.a = 0.25f;
    } else if (b.task_status == Block::TASK_MOVE) {
      m.color.a = 0.85f;
    } else if (b.task_status == Block::TASK_PLACED) {
      m.color.a = 0.95f;   // placed block: near-solid so the actual pose reads clearly
    }
    ma.markers.push_back(std::move(m));

    visualization_msgs::msg::Marker label;
    label.header = marker_header;
    label.ns = "cbp_block_ids";
    label.id = marker_id++;
    label.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    label.action = visualization_msgs::msg::Marker::ADD;
    label.lifetime = dynamic_marker_lifetime;
    label.pose = b.pose;
    label.pose.position.z += 0.7;
    label.scale.z = 0.2;
    label.color.r = 1.0f;
    label.color.g = 1.0f;
    label.color.b = 1.0f;
    label.color.a = 0.95f;
    label.text = b.id;
    ma.markers.push_back(std::move(label));

    if (b.pose_status == Block::POSE_PRECISE) {
      // Draw local X/Y/Z axes as arrows to verify orientation.
      const double qx = b.pose.orientation.x;
      const double qy = b.pose.orientation.y;
      const double qz = b.pose.orientation.z;
      const double qw = b.pose.orientation.w;

      // Columns of the rotation matrix (world directions for block +X, +Y, +Z).
      const double ax[3] = {
        1 - 2 * (qy * qy + qz * qz),
        2 * (qx * qy + qw * qz),
        2 * (qx * qz - qw * qy)
      };
      const double ay[3] = {
        2 * (qx * qy - qw * qz),
        1 - 2 * (qx * qx + qz * qz),
        2 * (qy * qz + qw * qx)
      };
      const double az[3] = {
        2 * (qx * qz + qw * qy),
        2 * (qy * qz - qw * qx),
        1 - 2 * (qx * qx + qy * qy)
      };

      constexpr double kAxisLen = 0.5;
      struct AxisDef { const double * dir; float r, g, bl; };
      const AxisDef axes[3] = {
        {ax, 1.0f, 0.0f, 0.0f},   // X → red
        {ay, 0.0f, 1.0f, 0.0f},   // Y → green
        {az, 0.0f, 0.0f, 1.0f},   // Z → blue
      };

      for (const auto & a : axes) {
        visualization_msgs::msg::Marker arrow;
        arrow.header = marker_header;
        arrow.ns = "cbp_block_axes";
        arrow.id = marker_id++;
        arrow.type = visualization_msgs::msg::Marker::ARROW;
        arrow.action = visualization_msgs::msg::Marker::ADD;
        arrow.lifetime = dynamic_marker_lifetime;
        arrow.scale.x = 0.025;   // shaft diameter
        arrow.scale.y = 0.05;    // head diameter
        arrow.scale.z = 0.0;
        arrow.color.r = a.r;
        arrow.color.g = a.g;
        arrow.color.b = a.bl;
        arrow.color.a = 1.0f;

        geometry_msgs::msg::Point p0, p1;
        p0.x = b.pose.position.x;
        p0.y = b.pose.position.y;
        p0.z = b.pose.position.z;
        p1.x = p0.x + kAxisLen * a.dir[0];
        p1.y = p0.y + kAxisLen * a.dir[1];
        p1.z = p0.z + kAxisLen * a.dir[2];
        arrow.points.push_back(p0);
        arrow.points.push_back(p1);
        ma.markers.push_back(std::move(arrow));
      }
    }
  }

  return ma;
}

visualization_msgs::msg::MarkerArray buildGoalMarkers(
  const std_msgs::msg::Header & header,
  const std::vector<Block> & blocks,
  const std::string & world_frame,
  const std::array<double, 3> & block_dimensions_m)
{
  visualization_msgs::msg::MarkerArray ma;
  auto marker_header = header;
  marker_header.frame_id = world_frame;

  visualization_msgs::msg::Marker clear;
  clear.header = marker_header;
  clear.ns = "";
  clear.id = 0;
  clear.type = visualization_msgs::msg::Marker::CUBE;
  clear.action = visualization_msgs::msg::Marker::DELETEALL;
  ma.markers.push_back(clear);

  int marker_id = 1;
  for (const auto & b : blocks) {
    if (b.goal_status != Block::GOAL_SET) {
      continue;
    }

    visualization_msgs::msg::Marker m;
    m.header = marker_header;
    m.ns = "cbp_block_goals";
    m.id = marker_id++;
    m.type = visualization_msgs::msg::Marker::CUBE;
    m.action = visualization_msgs::msg::Marker::ADD;
    m.pose = b.goal_pose;
    m.scale.x = block_dimensions_m[0];
    m.scale.y = block_dimensions_m[1];
    m.scale.z = block_dimensions_m[2];
    // Translucent steel-blue target cube. Kept see-through so the actual placed
    // block (solid green) and its goal pose stay visible together, making the
    // placement error easy to read off in RViz.
    m.color.r = 0.2f;
    m.color.g = 0.45f;
    m.color.b = 0.9f;
    m.color.a = 0.3f;
    ma.markers.push_back(std::move(m));

    visualization_msgs::msg::Marker label;
    label.header = marker_header;
    label.ns = "cbp_block_goal_ids";
    label.id = marker_id++;
    label.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    label.action = visualization_msgs::msg::Marker::ADD;
    label.pose = b.goal_pose;
    label.pose.position.z += 0.5 * block_dimensions_m[2] + 0.2;
    label.scale.z = 0.15;
    label.color.r = 1.0f;
    label.color.g = 1.0f;
    label.color.b = 1.0f;
    label.color.a = 1.0f;
    label.text = b.id;
    ma.markers.push_back(std::move(label));
  }

  return ma;
}

const char kReservedPayloadId[] = "payload";
const char kReservedTruckId[] = "truck";

CollisionSceneConversion toCollisionScene(
  const concrete_block_world_model_interfaces::msg::PlanningScene & planning_scene,
  const geometry_msgs::msg::TransformStamped & mounting_base_from_world,
  const VehicleBoxConfig & vehicle_box)
{
  CollisionSceneConversion out;
  out.scene.header.stamp = planning_scene.header.stamp;
  out.scene.header.frame_id = mounting_base_from_world.header.frame_id;
  out.scene.primitives.reserve(planning_scene.objects.size() + 1U);

  const std::string & source_frame = mounting_base_from_world.child_frame_id;
  std::vector<std::string> seen_ids;
  seen_ids.reserve(planning_scene.objects.size());

  // The vehicle first, and outside the loop below: the object stream drops the reserved id, and
  // the configured box is the one thing allowed to carry it.
  if (vehicle_box.enabled) {
    crane_msgs::msg::CollisionPrimitive truck;
    std::string report;
    if (truckPrimitiveFromVehicleBox(vehicle_box, mounting_base_from_world, truck, report)) {
      out.scene.primitives.push_back(std::move(truck));
    } else {
      out.dropped.push_back(std::string(kReservedTruckId) + ": " + report);
    }
  }

  for (const auto & object : planning_scene.objects) {
    const std::string named = object.id.empty() ? std::string("<no id>") : object.id;

    if (object.id.empty()) {
      out.dropped.push_back(named + ": an object with no id -- the planner needs one to name it");
      continue;
    }
    if (object.id == kReservedPayloadId || object.id == kReservedTruckId) {
      out.dropped.push_back(named + ": the id is reserved by crane_msgs and refuses the scene");
      continue;
    }
    if (std::find(seen_ids.begin(), seen_ids.end(), object.id) != seen_ids.end()) {
      out.dropped.push_back(named + ": the id arrived twice");
      continue;
    }
    if (!object.frame_id.empty() && object.frame_id != source_frame) {
      out.dropped.push_back(
        named + ": it is in frame '" + object.frame_id + "' and the transform comes from '" +
        source_frame + "'");
      continue;
    }
    uint8_t shape = 0;
    if (!collisionShapeFromPlanningScene(object.shape_type, shape)) {
      out.dropped.push_back(
        named + ": shape " + std::to_string(static_cast<int>(object.shape_type)) +
        " is none the planner enumerates");
      continue;
    }
    if (!hasPositiveExtent(object.dimensions)) {
      out.dropped.push_back(named + ": no finite, positive extent along every axis");
      continue;
    }
    geometry_msgs::msg::Pose normalized;
    if (!normalizedPose(object.pose, normalized)) {
      out.dropped.push_back(named + ": the pose is not finite or its quaternion has no direction");
      continue;
    }

    crane_msgs::msg::CollisionPrimitive primitive;
    primitive.id = object.id;
    primitive.shape = shape;
    tf2::doTransform(normalized, primitive.pose, mounting_base_from_world);
    primitive.dimensions = object.dimensions;
    // `structural` is "from a model" as against "perceived" (ros2_interfaces 6), which is
    // exactly the static-obstacle / block distinction the snapshot already carries.
    primitive.structural = object.source_type == PlanningSceneObject::SOURCE_STATIC_OBSTACLE;

    seen_ids.push_back(object.id);
    out.scene.primitives.push_back(std::move(primitive));
  }

  return out;
}

}  // namespace cbp::world_model
