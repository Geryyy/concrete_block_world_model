#include <cmath>
#include <string>

#include <gtest/gtest.h>

#include "concrete_block_world_model_interfaces/msg/block.hpp"
#include "concrete_block_world_model/world_model/block_filter.hpp"

namespace cbpwm = cbp::world_model;
using concrete_block_world_model_interfaces::msg::Block;

namespace
{

Block makeBlock(
  double x,
  double y,
  double z,
  const Eigen::Quaterniond & orientation = Eigen::Quaterniond::Identity(),
  bool precise = true)
{
  Block block;
  block.id = "block_1";
  block.pose_status = precise ? Block::POSE_PRECISE : Block::POSE_COARSE;
  block.task_status = Block::TASK_FREE;
  block.pose.position.x = x;
  block.pose.position.y = y;
  block.pose.position.z = z;
  const Eigen::Quaterniond q = orientation.normalized();
  block.pose.orientation.x = q.x();
  block.pose.orientation.y = q.y();
  block.pose.orientation.z = q.z();
  block.pose.orientation.w = q.w();
  block.confidence = 1.0F;

  block.pose_covariance.fill(0.0);
  const double pos_var = precise ? 0.02 * 0.02 : 0.20 * 0.20;
  const double rot_var = precise ? 0.10 * 0.10 : M_PI * M_PI;
  for (size_t i = 0; i < 3; ++i) {
    block.pose_covariance[i * 6 + i] = pos_var;
    block.pose_covariance[(i + 3) * 6 + i + 3] = rot_var;
  }
  return block;
}

cbpwm::BlockObservation makeObservation(
  const Block & block,
  bool precise = true)
{
  cbpwm::BlockObservation obs;
  obs.block = block;
  obs.mask_pixels = 1000;
  obs.cutout_points = 200;
  obs.fragment_count = 1;
  obs.precise = precise;
  return obs;
}

}  // namespace

TEST(BlockFilter, JumpRejectionLeavesConfirmedTrackUnchanged)
{
  cbpwm::BlockFilterConfig cfg;
  cfg.confirmation_hits = 1;
  cfg.confirmation_window = 1;
  cfg.max_consecutive_rejections = 5;

  auto track = cbpwm::initializeTrack(makeObservation(makeBlock(0.0, 0.0, 0.0)), 0.0, cfg);
  ASSERT_EQ(track.state, cbpwm::FilteredBlockTrackState::kConfirmed);
  const Eigen::Vector3d before = track.position;

  std::string reason;
  const bool ok = cbpwm::gateAndUpdate(
    track,
    makeObservation(makeBlock(5.0, 0.0, 0.0)),
    cfg,
    reason);

  EXPECT_FALSE(ok);
  EXPECT_NE(reason.find("position gate rejected"), std::string::npos);
  EXPECT_TRUE(track.position.isApprox(before));
  EXPECT_EQ(track.consecutive_rejections, 1);
}

TEST(BlockFilter, ReinitializesAfterConsecutiveRejections)
{
  cbpwm::BlockFilterConfig cfg;
  cfg.confirmation_hits = 1;
  cfg.confirmation_window = 1;
  cfg.max_consecutive_rejections = 2;

  auto track = cbpwm::initializeTrack(makeObservation(makeBlock(0.0, 0.0, 0.0)), 0.0, cfg);
  std::string reason;
  EXPECT_FALSE(cbpwm::gateAndUpdate(track, makeObservation(makeBlock(3.0, 0.0, 0.0)), cfg, reason));
  EXPECT_TRUE(cbpwm::gateAndUpdate(track, makeObservation(makeBlock(3.0, 0.0, 0.0)), cfg, reason));

  EXPECT_NE(reason.find("reinitialized"), std::string::npos);
  EXPECT_NEAR(track.position.x(), 3.0, 1.0e-9);
  EXPECT_EQ(track.consecutive_rejections, 0);
}

TEST(BlockFilter, SpuriousSingleHitNeverConfirmsAfterMisses)
{
  cbpwm::BlockFilterConfig cfg;
  cfg.confirmation_hits = 2;
  cfg.confirmation_window = 3;

  auto track = cbpwm::initializeTrack(makeObservation(makeBlock(0.0, 0.0, 0.0)), 0.0, cfg);
  cbpwm::recordMiss(track, cfg);
  cbpwm::recordMiss(track, cfg);

  EXPECT_EQ(track.state, cbpwm::FilteredBlockTrackState::kTentative);
}

TEST(BlockFilter, ConfirmsAfterMOfNHits)
{
  cbpwm::BlockFilterConfig cfg;
  cfg.confirmation_hits = 2;
  cfg.confirmation_window = 3;

  auto track = cbpwm::initializeTrack(makeObservation(makeBlock(0.0, 0.0, 0.0)), 0.0, cfg);
  std::string reason;
  EXPECT_TRUE(cbpwm::gateAndUpdate(track, makeObservation(makeBlock(0.01, 0.0, 0.0)), cfg, reason));

  EXPECT_EQ(track.state, cbpwm::FilteredBlockTrackState::kConfirmed);
}

TEST(BlockFilter, CoarseObservationDoesNotChangeOrientationButPreciseDoes)
{
  cbpwm::BlockFilterConfig cfg;
  cfg.confirmation_hits = 1;
  cfg.confirmation_window = 1;

  auto track = cbpwm::initializeTrack(makeObservation(makeBlock(0.0, 0.0, 0.0)), 0.0, cfg);
  const Eigen::Quaterniond before = track.orientation;
  const Eigen::Quaterniond yaw90(Eigen::AngleAxisd(M_PI / 2.0, Eigen::Vector3d::UnitZ()));

  std::string reason;
  EXPECT_TRUE(cbpwm::gateAndUpdate(
    track,
    makeObservation(makeBlock(0.0, 0.0, 0.0, yaw90, false), false),
    cfg,
    reason));
  EXPECT_NEAR(std::abs(track.orientation.dot(before)), 1.0, 1.0e-9);

  EXPECT_TRUE(cbpwm::gateAndUpdate(
    track,
    makeObservation(makeBlock(0.0, 0.0, 0.0, yaw90, true), true),
    cfg,
    reason));
  EXPECT_LT(std::abs(track.orientation.dot(before)), 0.999);
}

TEST(BlockFilter, SymmetryCanonicalizationRemovesPiFlipInnovation)
{
  const Eigen::Quaterniond estimate = Eigen::Quaterniond::Identity();
  const Eigen::Quaterniond flipped(Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX()));

  const Eigen::Quaterniond canonical =
    cbpwm::canonicalizeSymmetricOrientation(estimate, flipped);
  const Eigen::Vector3d innovation =
    cbpwm::rotationVectorInnovation(estimate, canonical);

  EXPECT_NEAR(innovation.norm(), 0.0, 1.0e-9);
}

TEST(BlockFilter, PredictionGrowthReducesDerivedConfidence)
{
  cbpwm::BlockFilterConfig cfg;
  cfg.confirmation_hits = 1;
  cfg.confirmation_window = 1;

  auto track = cbpwm::initializeTrack(makeObservation(makeBlock(0.0, 0.0, 0.0)), 0.0, cfg);
  const float before = cbpwm::toBlockMsg(track, makeBlock(0.0, 0.0, 0.0)).confidence;

  cbpwm::predict(track, 100.0, cfg);
  const float after = cbpwm::toBlockMsg(track, makeBlock(0.0, 0.0, 0.0)).confidence;

  EXPECT_LT(after, before);
}
