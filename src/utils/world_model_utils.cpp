#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <geometry_msgs/msg/point.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include "concrete_block_world_model/utils/world_model_utils.hpp"

namespace cbp::world_model
{

using concrete_block_world_model_interfaces::msg::Block;
using concrete_block_world_model_interfaces::msg::PlanningSceneObject;

namespace
{

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

// Axis-aligned box in image pixels, used for detection overlap tests.
struct DetAabb
{
  double x0, y0, x1, y1;
};

DetAabb detAabb(const vision_msgs::msg::Detection2D & det)
{
  const double hw = det.bbox.size_x * 0.5;
  const double hh = det.bbox.size_y * 0.5;
  const double cx = det.bbox.center.position.x;
  const double cy = det.bbox.center.position.y;
  return DetAabb{cx - hw, cy - hh, cx + hw, cy + hh};
}

double aabbArea(const DetAabb & b)
{
  return std::max(0.0, b.x1 - b.x0) * std::max(0.0, b.y1 - b.y0);
}

double aabbIntersection(const DetAabb & a, const DetAabb & b)
{
  const double x0 = std::max(a.x0, b.x0);
  const double y0 = std::max(a.y0, b.y0);
  const double x1 = std::min(a.x1, b.x1);
  const double y1 = std::min(a.y1, b.y1);
  return std::max(0.0, x1 - x0) * std::max(0.0, y1 - y0);
}

int dsuFind(std::vector<int> & parent, int i)
{
  while (parent[i] != i) {
    parent[i] = parent[parent[i]];  // path halving
    i = parent[i];
  }
  return i;
}

}  // namespace

double detectionConfidence(const vision_msgs::msg::Detection2D & det)
{
  if (det.results.empty()) {
    return 1.0;
  }
  return det.results.front().hypothesis.score;
}

vision_msgs::msg::Detection2DArray mergeOverlappingDetections(
  const vision_msgs::msg::Detection2DArray & detections,
  double containment_ratio,
  double iou_threshold)
{
  const auto & dets = detections.detections;
  const std::size_t n = dets.size();

  vision_msgs::msg::Detection2DArray out;
  out.header = detections.header;
  if (n <= 1) {
    out.detections = dets;
    return out;
  }

  std::vector<DetAabb> boxes(n);
  std::vector<double> areas(n);
  for (std::size_t i = 0; i < n; ++i) {
    boxes[i] = detAabb(dets[i]);
    areas[i] = aabbArea(boxes[i]);
  }

  // Union-find grouping of strongly-overlapping detections (transitive).
  std::vector<int> parent(n);
  for (std::size_t i = 0; i < n; ++i) {
    parent[i] = static_cast<int>(i);
  }

  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = i + 1; j < n; ++j) {
      const double inter = aabbIntersection(boxes[i], boxes[j]);
      if (inter <= 0.0) {
        continue;
      }
      const double min_area = std::min(areas[i], areas[j]);
      const double containment = (min_area > 0.0) ? inter / min_area : 0.0;
      const double union_area = areas[i] + areas[j] - inter;
      const double iou = (union_area > 0.0) ? inter / union_area : 0.0;
      if (containment >= containment_ratio || iou >= iou_threshold) {
        parent[dsuFind(parent, static_cast<int>(i))] =
          dsuFind(parent, static_cast<int>(j));
      }
    }
  }

  // Collect members per group, preserving first-seen order for deterministic output.
  std::vector<int> group_order;
  group_order.reserve(n);
  std::unordered_map<int, std::vector<std::size_t>> groups;
  for (std::size_t i = 0; i < n; ++i) {
    const int root = dsuFind(parent, static_cast<int>(i));
    auto it = groups.find(root);
    if (it == groups.end()) {
      group_order.push_back(root);
      groups.emplace(root, std::vector<std::size_t>{i});
    } else {
      it->second.push_back(i);
    }
  }

  out.detections.reserve(group_order.size());
  for (const int root : group_order) {
    const auto & members = groups[root];
    if (members.size() == 1) {
      out.detections.push_back(dets[members.front()]);
      continue;
    }

    // Union box + highest-confidence member drives class/hypothesis/header.
    DetAabb u = boxes[members.front()];
    std::size_t best = members.front();
    double best_conf = detectionConfidence(dets[best]);
    for (std::size_t k = 1; k < members.size(); ++k) {
      const std::size_t m = members[k];
      u.x0 = std::min(u.x0, boxes[m].x0);
      u.y0 = std::min(u.y0, boxes[m].y0);
      u.x1 = std::max(u.x1, boxes[m].x1);
      u.y1 = std::max(u.y1, boxes[m].y1);
      const double conf = detectionConfidence(dets[m]);
      if (conf > best_conf) {
        best_conf = conf;
        best = m;
      }
    }

    vision_msgs::msg::Detection2D merged = dets[best];
    merged.bbox.center.position.x = 0.5 * (u.x0 + u.x1);
    merged.bbox.center.position.y = 0.5 * (u.y0 + u.y1);
    merged.bbox.size_x = u.x1 - u.x0;
    merged.bbox.size_y = u.y1 - u.y0;
    out.detections.push_back(std::move(merged));
  }

  return out;
}

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
