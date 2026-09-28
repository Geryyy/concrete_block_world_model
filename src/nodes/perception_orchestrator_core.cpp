#include "concrete_block_world_model/nodes/perception_orchestrator_node.hpp"

#include "concrete_block_world_model/utils/block_utils.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

namespace
{

// The frame the planner plans in and the only one it accepts a scene in
// (wiki/nomenclature.md 3, wiki/implementation/ros2_interfaces.md 4).
constexpr char kMountingBaseFrame[] = "K0_mounting_base";

geometry_msgs::msg::Vector3 toVector3(const std::array<double, 3> & values)
{
  geometry_msgs::msg::Vector3 out;
  out.x = values[0];
  out.y = values[1];
  out.z = values[2];
  return out;
}

}  // namespace

std::string PerceptionOrchestratorNode::resolveGraspedBlockId()
  {
    std::lock_guard<std::mutex> lock(persistent_world_mutex_);
    return cbpwm::resolveGraspedBlockId(persistent_world_, *get_clock());
  }

cbpwm::AssociationConfig PerceptionOrchestratorNode::associationConfig() const
  {
    cbpwm::AssociationConfig cfg;
    cfg.association_max_distance_m = runtime_cfg_.association_max_distance_m;
    cfg.association_max_age_s = runtime_cfg_.association_max_age_s;
    cfg.min_update_confidence = runtime_cfg_.min_update_confidence;
    return cfg;
  }

void PerceptionOrchestratorNode::updateLatestWorldCache(const BlockArray & out)
  {
    std::lock_guard<std::mutex> lock(latest_world_mutex_);
    latest_world_ = out;
  }

BlockArray PerceptionOrchestratorNode::latestWorldSnapshot()
  {
    std::lock_guard<std::mutex> lock(latest_world_mutex_);
    return latest_world_;
  }

PlanningScene PerceptionOrchestratorNode::latestPlanningSceneSnapshot()
  {
    std::lock_guard<std::mutex> lock(latest_planning_scene_mutex_);
    return latest_planning_scene_;
  }

std::vector<PlanningSceneObject> PerceptionOrchestratorNode::staticSceneObjectsInWorld() const
  {
    std::vector<PlanningSceneObject> out;
    out.reserve(static_scene_objects_.size());

    for (const auto & object : static_scene_objects_) {
      PlanningSceneObject world_object = object;
      world_object.frame_id = world_frame_;

      if (!tf_buffer_ || object.frame_id.empty() || object.frame_id == world_frame_) {
        out.push_back(std::move(world_object));
        continue;
      }

      geometry_msgs::msg::PoseStamped src_pose;
      src_pose.header.frame_id = object.frame_id;
      src_pose.header.stamp = rclcpp::Time(0, 0, get_clock()->get_clock_type());
      src_pose.pose = object.pose;

      try {
        const auto tf = tf_buffer_->lookupTransform(
          world_frame_,
          object.frame_id,
          tf2::TimePointZero,
          tf2::durationFromSec(0.2));
        geometry_msgs::msg::PoseStamped dst_pose;
        tf2::doTransform(src_pose, dst_pose, tf);
        world_object.pose = dst_pose.pose;
      } catch (const tf2::TransformException & ex) {
        RCLCPP_WARN(
          get_logger(),
          "Could not transform static scene object '%s' from %s to %s: %s",
          object.id.c_str(),
          object.frame_id.c_str(),
          world_frame_.c_str(),
          ex.what());
      }

      out.push_back(std::move(world_object));
    }

    return out;
  }

PlanningScene PerceptionOrchestratorNode::buildPlanningSceneSnapshot(
    const std_msgs::msg::Header & header,
    const std::vector<Block> & blocks)
  {
    PlanningScene scene;
    scene.header = header;
    scene.header.frame_id = world_frame_;

    const auto static_scene_world = staticSceneObjectsInWorld();
    scene.objects.reserve(static_scene_world.size() + blocks.size());
    for (const auto & object : static_scene_world) {
      scene.objects.push_back(object);
    }

    for (const auto & block : blocks) {
      if (block.pose_status == Block::POSE_UNKNOWN) {
        continue;
      }
      PlanningSceneObject object;
      object.id = block.id;
      object.frame_id = world_frame_;
      object.shape_type = PlanningSceneObject::SHAPE_BOX;
      object.source_type = PlanningSceneObject::SOURCE_BLOCK;
      object.pose = block.pose;
      object.dimensions = toVector3(block_dimensions_m_);
      object.pose_status = block.pose_status;
      object.task_status = block.task_status;
      object.confidence = block.confidence;
      scene.objects.push_back(std::move(object));
    }
    return scene;
  }

void PerceptionOrchestratorNode::publishCollisionScene(const PlanningScene & scene)
  {
    if (!collision_scene_pub_ || !tf_buffer_) {
      return;
    }

    geometry_msgs::msg::TransformStamped mounting_base_from_world;
    try {
      mounting_base_from_world = tf_buffer_->lookupTransform(
        kMountingBaseFrame,
        scene.header.frame_id,
        tf2::TimePointZero,
        tf2::durationFromSec(0.2));
    } catch (const tf2::TransformException & ex) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "No collision scene published: cannot transform %s to %s: %s",
        scene.header.frame_id.c_str(),
        kMountingBaseFrame,
        ex.what());
      return;
    }

    auto converted = cbpwm::toCollisionScene(scene, mounting_base_from_world, vehicle_box_);
    for (const auto & dropped : converted.dropped) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "Collision scene: dropped %s", dropped.c_str());
    }

    // The stamp moves on every snapshot, so the comparison is over the primitives and the frame
    // -- otherwise every republish would be a "change". The planner *does* read the stamp
    // (crane_planning issue 096: `max_scene_age`), so suppressing an unchanged scene forever is
    // not an option: on this transient-local topic a scene published once and then held back is
    // indistinguishable, at the planner, from a world model that has died. So an unchanged scene
    // is republished anyway once `collision_scene_heartbeat_s` has passed, and the planner's age
    // check then means "the world model has stopped" rather than "nothing has moved lately".
    const rclcpp::Time publish_time = now();
    const bool unchanged = collision_scene_published_ &&
      last_collision_scene_.header.frame_id == converted.scene.header.frame_id &&
      last_collision_scene_.primitives == converted.scene.primitives;
    const bool heartbeat_due = !collision_scene_published_ ||
      (publish_time - last_collision_scene_publish_).seconds() >= collision_scene_heartbeat_s_;
    if (unchanged && !heartbeat_due) {
      return;
    }

    collision_scene_pub_->publish(converted.scene);
    last_collision_scene_ = converted.scene;
    last_collision_scene_publish_ = publish_time;
    collision_scene_published_ = true;
    if (unchanged) {
      // The heartbeat is not news; logging it at INFO once every two seconds would bury the
      // changes that are.
      return;
    }
    RCLCPP_INFO(
      get_logger(),
      "Published collision scene: primitives=%zu dropped=%zu frame=%s",
      converted.scene.primitives.size(),
      converted.dropped.size(),
      converted.scene.header.frame_id.c_str());
  }
