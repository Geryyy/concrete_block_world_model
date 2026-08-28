#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

#include <cstddef>
#include <limits>
#include <string>
#include <vector>

#include "concrete_block_world_model_interfaces/msg/block.hpp"
#include "concrete_block_world_model/utils/world_model_utils.hpp"

namespace cbpwm = cbp::world_model;
using concrete_block_world_model_interfaces::msg::Block;

TEST(WorldModelUtils, TaskTransitionRules) {
  EXPECT_TRUE(cbpwm::isValidTaskTransition(Block::TASK_UNKNOWN, Block::TASK_FREE));
  EXPECT_TRUE(cbpwm::isValidTaskTransition(Block::TASK_FREE, Block::TASK_MOVE));
  EXPECT_TRUE(cbpwm::isValidTaskTransition(Block::TASK_MOVE, Block::TASK_PLACED));
  EXPECT_TRUE(cbpwm::isValidTaskTransition(Block::TASK_PLACED, Block::TASK_MOVE));

  EXPECT_FALSE(cbpwm::isValidTaskTransition(Block::TASK_FREE, Block::TASK_PLACED));
  EXPECT_FALSE(cbpwm::isValidTaskTransition(Block::TASK_REMOVED, Block::TASK_FREE));
}

TEST(WorldModelUtils, AssociationDistanceAndConfidenceGating) {
  EXPECT_TRUE(cbpwm::shouldAssociateByDistance(0.20, 0.45, 0.9, 0.25));
  EXPECT_FALSE(cbpwm::shouldAssociateByDistance(0.60, 0.45, 0.9, 0.25));
  EXPECT_FALSE(cbpwm::shouldAssociateByDistance(0.20, 0.45, 0.1, 0.25));
}

TEST(WorldModelUtils, TaskStatusToString) {
  EXPECT_STREQ(cbpwm::taskStatusToString(Block::TASK_FREE), "TASK_FREE");
  EXPECT_STREQ(cbpwm::taskStatusToString(Block::TASK_MOVE), "TASK_MOVE");
  EXPECT_STREQ(cbpwm::taskStatusToString(Block::TASK_PLACED), "TASK_PLACED");
}

TEST(WorldModelUtils, ParseOneShotModes) {
  EXPECT_EQ(
    cbpwm::parseOneShotMode("SCENE_DISCOVERY"),
    cbpwm::OneShotMode::kSceneDiscovery);
  EXPECT_EQ(
    cbpwm::parseOneShotMode("refine_block"),
    cbpwm::OneShotMode::kRefineBlock);
  EXPECT_EQ(
    cbpwm::parseOneShotMode("Refine_Grasped"),
    cbpwm::OneShotMode::kRefineGrasped);
  EXPECT_EQ(
    cbpwm::parseOneShotMode("unsupported"),
    cbpwm::OneShotMode::kNone);
}

namespace
{

using concrete_block_world_model_interfaces::msg::PlanningScene;
using concrete_block_world_model_interfaces::msg::PlanningSceneObject;

// The transform tf2 returns for lookupTransform(K0_mounting_base, world): a pure translation of
// one metre along +x, so a world point lands one metre lower in x in the mounting base.
geometry_msgs::msg::TransformStamped mountingBaseFromWorld()
{
  geometry_msgs::msg::TransformStamped tf;
  tf.header.frame_id = "K0_mounting_base";
  tf.child_frame_id = "world";
  tf.transform.translation.x = -1.0;
  tf.transform.rotation.w = 1.0;
  return tf;
}

PlanningSceneObject makeObject(const std::string & id, uint8_t source_type)
{
  PlanningSceneObject object;
  object.id = id;
  object.frame_id = "world";
  object.shape_type = PlanningSceneObject::SHAPE_BOX;
  object.source_type = source_type;
  object.pose.position.x = 2.0;
  object.pose.position.y = 3.0;
  object.pose.position.z = 0.5;
  object.pose.orientation.w = 1.0;
  object.dimensions.x = 0.8;
  object.dimensions.y = 0.4;
  object.dimensions.z = 0.2;
  return object;
}

PlanningScene sceneWith(const std::vector<PlanningSceneObject> & objects)
{
  PlanningScene scene;
  scene.header.frame_id = "world";
  scene.objects = objects;
  return scene;
}

// A scene of one good block plus the object under test, so that a filter dropping the bad one
// is visibly not dropping the rest of the scene.
PlanningScene sceneWithBadObject(const PlanningSceneObject & bad)
{
  return sceneWith({makeObject("block_ok", PlanningSceneObject::SOURCE_BLOCK), bad});
}

}  // namespace

TEST(CollisionScene, StructuralFollowsTheSourceTypeInBothDirections) {
  const auto scene = sceneWith(
    {makeObject("wall", PlanningSceneObject::SOURCE_STATIC_OBSTACLE),
      makeObject("block_1", PlanningSceneObject::SOURCE_BLOCK)});

  const auto converted = cbpwm::toCollisionScene(scene, mountingBaseFromWorld());

  ASSERT_EQ(converted.scene.primitives.size(), 2u);
  EXPECT_TRUE(converted.dropped.empty());
  EXPECT_EQ(converted.scene.primitives[0].id, "wall");
  EXPECT_TRUE(converted.scene.primitives[0].structural);
  EXPECT_EQ(converted.scene.primitives[1].id, "block_1");
  EXPECT_FALSE(converted.scene.primitives[1].structural);
}

TEST(CollisionScene, PoseAndFrameReachTheMountingBase) {
  const auto scene = sceneWith({makeObject("block_1", PlanningSceneObject::SOURCE_BLOCK)});

  const auto converted = cbpwm::toCollisionScene(scene, mountingBaseFromWorld());

  ASSERT_EQ(converted.scene.primitives.size(), 1u);
  EXPECT_EQ(converted.scene.header.frame_id, "K0_mounting_base");
  const auto & primitive = converted.scene.primitives[0];
  EXPECT_DOUBLE_EQ(primitive.pose.position.x, 1.0);
  EXPECT_DOUBLE_EQ(primitive.pose.position.y, 3.0);
  EXPECT_DOUBLE_EQ(primitive.pose.position.z, 0.5);
  EXPECT_EQ(primitive.shape, crane_msgs::msg::CollisionScene::SHAPE_BOX);
  // Full extents in both messages, carried across unchanged.
  EXPECT_DOUBLE_EQ(primitive.dimensions.x, 0.8);
  EXPECT_DOUBLE_EQ(primitive.dimensions.y, 0.4);
  EXPECT_DOUBLE_EQ(primitive.dimensions.z, 0.2);
}

TEST(CollisionScene, DropsUnknownShape) {
  auto bad = makeObject("mystery", PlanningSceneObject::SOURCE_BLOCK);
  bad.shape_type = PlanningSceneObject::SHAPE_UNKNOWN;

  const auto converted = cbpwm::toCollisionScene(sceneWithBadObject(bad), mountingBaseFromWorld());

  ASSERT_EQ(converted.scene.primitives.size(), 1u);
  EXPECT_EQ(converted.scene.primitives[0].id, "block_ok");
  EXPECT_EQ(converted.dropped.size(), 1u);
}

TEST(CollisionScene, DropsNonPositiveAndNonFiniteExtents) {
  auto flat = makeObject("flat", PlanningSceneObject::SOURCE_BLOCK);
  flat.dimensions.z = 0.0;
  auto infinite = makeObject("infinite", PlanningSceneObject::SOURCE_BLOCK);
  infinite.dimensions.y = std::numeric_limits<double>::infinity();

  const auto converted = cbpwm::toCollisionScene(
    sceneWith({makeObject("block_ok", PlanningSceneObject::SOURCE_BLOCK), flat, infinite}),
    mountingBaseFromWorld());

  ASSERT_EQ(converted.scene.primitives.size(), 1u);
  EXPECT_EQ(converted.scene.primitives[0].id, "block_ok");
  EXPECT_EQ(converted.dropped.size(), 2u);
}

TEST(CollisionScene, DropsDuplicateAndEmptyIds) {
  auto duplicate = makeObject("block_ok", PlanningSceneObject::SOURCE_BLOCK);
  auto nameless = makeObject("", PlanningSceneObject::SOURCE_BLOCK);

  const auto converted = cbpwm::toCollisionScene(
    sceneWith({makeObject("block_ok", PlanningSceneObject::SOURCE_BLOCK), duplicate, nameless}),
    mountingBaseFromWorld());

  ASSERT_EQ(converted.scene.primitives.size(), 1u);
  EXPECT_EQ(converted.scene.primitives[0].id, "block_ok");
  EXPECT_EQ(converted.dropped.size(), 2u);
}

TEST(CollisionScene, DropsUnusablePoseAndRepairsAnUnnormalizedQuaternion) {
  auto zero_quaternion = makeObject("zero_quaternion", PlanningSceneObject::SOURCE_BLOCK);
  zero_quaternion.pose.orientation.w = 0.0;
  auto not_finite = makeObject("not_finite", PlanningSceneObject::SOURCE_BLOCK);
  not_finite.pose.position.z = std::numeric_limits<double>::quiet_NaN();
  auto unnormalized = makeObject("unnormalized", PlanningSceneObject::SOURCE_BLOCK);
  unnormalized.pose.orientation.w = 2.0;

  const auto converted = cbpwm::toCollisionScene(
    sceneWith(
      {makeObject("block_ok", PlanningSceneObject::SOURCE_BLOCK), zero_quaternion, not_finite,
        unnormalized}),
    mountingBaseFromWorld());

  ASSERT_EQ(converted.scene.primitives.size(), 2u);
  EXPECT_EQ(converted.scene.primitives[0].id, "block_ok");
  EXPECT_EQ(converted.scene.primitives[1].id, "unnormalized");
  EXPECT_DOUBLE_EQ(converted.scene.primitives[1].pose.orientation.w, 1.0);
  EXPECT_EQ(converted.dropped.size(), 2u);
}

TEST(CollisionScene, DropsReservedIds) {
  const auto converted = cbpwm::toCollisionScene(
    sceneWith(
      {makeObject("block_ok", PlanningSceneObject::SOURCE_BLOCK),
        makeObject(cbpwm::kReservedPayloadId, PlanningSceneObject::SOURCE_BLOCK),
        makeObject(cbpwm::kReservedTruckId, PlanningSceneObject::SOURCE_STATIC_OBSTACLE)}),
    mountingBaseFromWorld());

  ASSERT_EQ(converted.scene.primitives.size(), 1u);
  EXPECT_EQ(converted.scene.primitives[0].id, "block_ok");
  EXPECT_EQ(converted.dropped.size(), 2u);
}

TEST(CollisionScene, DropsAnObjectTheTransformDoesNotComeFrom) {
  auto elsewhere = makeObject("elsewhere", PlanningSceneObject::SOURCE_STATIC_OBSTACLE);
  elsewhere.frame_id = "camera_link";

  const auto converted = cbpwm::toCollisionScene(
    sceneWithBadObject(elsewhere), mountingBaseFromWorld());

  ASSERT_EQ(converted.scene.primitives.size(), 1u);
  EXPECT_EQ(converted.scene.primitives[0].id, "block_ok");
  EXPECT_EQ(converted.dropped.size(), 1u);
}

namespace
{

// The vehicle box the shipped config carries, as the tests need it: a load area long enough for
// the planner's three runge stations, keyed to the same world frame the transform comes from.
cbpwm::VehicleBoxConfig configuredVehicleBox()
{
  cbpwm::VehicleBoxConfig box;
  box.enabled = true;
  box.frame_id = "world";
  box.position = {-3.512, 0.35, 0.235575};
  box.dimensions = {6.594, 2.46, 0.66285};
  box.runge_length_m = 0.28;
  box.runge_stations_m = {-2.0, 0.0, 2.0};
  return box;
}

}  // namespace

TEST(CollisionSceneVehicleBox, EmittedOnceWithTheReservedIdAndStructural) {
  const auto converted = cbpwm::toCollisionScene(
    sceneWith({makeObject("block_ok", PlanningSceneObject::SOURCE_BLOCK)}),
    mountingBaseFromWorld(), configuredVehicleBox());

  ASSERT_EQ(converted.scene.primitives.size(), 2u);
  EXPECT_TRUE(converted.dropped.empty());

  std::size_t truck_count = 0;
  for (const auto & primitive : converted.scene.primitives) {
    truck_count += primitive.id == cbpwm::kReservedTruckId ? 1u : 0u;
  }
  EXPECT_EQ(truck_count, 1u);

  const auto & truck = converted.scene.primitives[0];
  EXPECT_EQ(truck.id, std::string(cbpwm::kReservedTruckId));
  EXPECT_TRUE(truck.structural);
  EXPECT_EQ(truck.shape, crane_msgs::msg::CollisionScene::SHAPE_BOX);
  // Reaches the planner in K0_mounting_base, one metre back along x like every other primitive.
  EXPECT_DOUBLE_EQ(truck.pose.position.x, -4.512);
  EXPECT_DOUBLE_EQ(truck.pose.position.y, 0.35);
  EXPECT_DOUBLE_EQ(truck.pose.position.z, 0.235575);
  // Full extents, carried across unchanged: x along the bed, z up.
  EXPECT_DOUBLE_EQ(truck.dimensions.x, 6.594);
  EXPECT_DOUBLE_EQ(truck.dimensions.y, 2.46);
  EXPECT_DOUBLE_EQ(truck.dimensions.z, 0.66285);
}

TEST(ShippedWorldModelConfig, DescribesTheVehicleInExactlyOnePlace) {
  const YAML::Node root = YAML::LoadFile(CBP_WORLD_MODEL_CONFIG_PATH);
  const YAML::Node world_model =
    root["world_model_node"]["ros__parameters"]["world_model"];
  ASSERT_TRUE(world_model);

  // The three former truck entries are gone from the static-object stream, so the only truck
  // that can reach the planner is the reserved primitive below.
  const YAML::Node statistics = YAML::Load(world_model["static_scene_objects"].as<std::string>(""));
  for (std::size_t idx = 0; statistics && idx < statistics.size(); ++idx) {
    const std::string id = statistics[idx]["id"].as<std::string>("");
    EXPECT_NE(id, "truck_main");
    EXPECT_NE(id, "truck_rear");
    EXPECT_NE(id, "truck_bed");
  }

  const YAML::Node vehicle_box = world_model["vehicle_box"];
  ASSERT_TRUE(vehicle_box);
  EXPECT_TRUE(vehicle_box["enable"].as<bool>(false));

  // And it is a box the planner will accept: every configured station carries its runge on the
  // configured bed.
  cbpwm::VehicleBoxConfig box;
  box.enabled = true;
  box.frame_id = vehicle_box["frame_id"].as<std::string>("world");
  for (std::size_t axis = 0; axis < 3; ++axis) {
    box.position[axis] = vehicle_box["position"][axis].as<double>();
    box.dimensions[axis] = vehicle_box["dimensions"][axis].as<double>();
    box.rpy_deg[axis] = vehicle_box["rpy_deg"][axis].as<double>();
  }
  box.runge_length_m = vehicle_box["runge_length_m"].as<double>(0.0);
  for (const auto & station : vehicle_box["runge_stations_m"]) {
    box.runge_stations_m.push_back(station.as<double>());
  }

  const auto converted = cbpwm::toCollisionScene(sceneWith({}), mountingBaseFromWorld(), box);
  EXPECT_TRUE(converted.dropped.empty());
  ASSERT_EQ(converted.scene.primitives.size(), 1u);
  EXPECT_EQ(converted.scene.primitives[0].id, std::string(cbpwm::kReservedTruckId));
  EXPECT_TRUE(converted.scene.primitives[0].structural);
}

// A blockless world -- a fresh seed_none profile, or one just cleared through clear_world_model
// -- still has the vehicle to say, so what the node republishes on the heartbeat is not an empty
// message. The node-level half of this (that the refresh timer keeps calling at all with no
// blocks) is the guard removed in perception_orchestrator_node.cpp; only the conversion is
// reachable offline.
TEST(CollisionSceneVehicleBox, AWorldWithNoBlocksStillCarriesTheTruck) {
  const auto converted = cbpwm::toCollisionScene(
    sceneWith({}), mountingBaseFromWorld(), configuredVehicleBox());

  EXPECT_TRUE(converted.dropped.empty());
  ASSERT_EQ(converted.scene.primitives.size(), 1u);
  EXPECT_EQ(converted.scene.primitives[0].id, std::string(cbpwm::kReservedTruckId));
  EXPECT_EQ(converted.scene.header.frame_id, "K0_mounting_base");
}

TEST(CollisionSceneVehicleBox, DisabledEmitsNothingAndReportsNothing) {
  const auto converted = cbpwm::toCollisionScene(
    sceneWith({makeObject("block_ok", PlanningSceneObject::SOURCE_BLOCK)}),
    mountingBaseFromWorld(), cbpwm::VehicleBoxConfig{});

  ASSERT_EQ(converted.scene.primitives.size(), 1u);
  EXPECT_EQ(converted.scene.primitives[0].id, "block_ok");
  EXPECT_TRUE(converted.dropped.empty());
}

TEST(CollisionSceneVehicleBox, ARungeOffTheEndOfTheBedIsReportedNotEmitted) {
  auto too_short = configuredVehicleBox();
  // 2.0 m from the centre plus half a 0.28 m runge needs 4.28 m of bed; this one has 4.0.
  too_short.dimensions[0] = 4.0;

  const auto converted = cbpwm::toCollisionScene(
    sceneWith({makeObject("block_ok", PlanningSceneObject::SOURCE_BLOCK)}),
    mountingBaseFromWorld(), too_short);

  ASSERT_EQ(converted.scene.primitives.size(), 1u);
  EXPECT_EQ(converted.scene.primitives[0].id, "block_ok");
  ASSERT_EQ(converted.dropped.size(), 1u);
  // Both numbers are named, as the planner's own refusal names them.
  EXPECT_NE(converted.dropped[0].find("2.000 m"), std::string::npos);
  EXPECT_NE(converted.dropped[0].find("4.000 m"), std::string::npos);
}

TEST(CollisionSceneVehicleBox, ABoxJustLongEnoughForTheOutermostStationIsEmitted) {
  auto exact = configuredVehicleBox();
  exact.dimensions[0] = 4.28;

  const auto converted = cbpwm::toCollisionScene(
    sceneWith({}), mountingBaseFromWorld(), exact);

  ASSERT_EQ(converted.scene.primitives.size(), 1u);
  EXPECT_EQ(converted.scene.primitives[0].id, std::string(cbpwm::kReservedTruckId));
  EXPECT_TRUE(converted.dropped.empty());
}

TEST(CollisionSceneVehicleBox, ReportsAnUnusableBoxRatherThanEmittingIt) {
  auto no_extent = configuredVehicleBox();
  no_extent.dimensions[2] = 0.0;
  auto not_finite = configuredVehicleBox();
  not_finite.position[1] = std::numeric_limits<double>::quiet_NaN();
  auto elsewhere = configuredVehicleBox();
  elsewhere.frame_id = "camera_link";

  for (const auto & box : {no_extent, not_finite, elsewhere}) {
    const auto converted = cbpwm::toCollisionScene(
      sceneWith({makeObject("block_ok", PlanningSceneObject::SOURCE_BLOCK)}),
      mountingBaseFromWorld(), box);

    ASSERT_EQ(converted.scene.primitives.size(), 1u);
    EXPECT_EQ(converted.scene.primitives[0].id, "block_ok");
    ASSERT_EQ(converted.dropped.size(), 1u);
    EXPECT_EQ(converted.dropped[0].rfind(cbpwm::kReservedTruckId, 0), 0u);
  }
}
