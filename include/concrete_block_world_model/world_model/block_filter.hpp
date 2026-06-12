#pragma once

#include <deque>
#include <string>

#include <Eigen/Dense>
#include <geometry_msgs/msg/pose.hpp>

#include "concrete_block_world_model_interfaces/msg/block.hpp"
#include "concrete_block_world_model/world_model/continuous_perception.hpp"

namespace cbp::world_model
{

struct BlockFilterConfig
{
  double position_process_sigma_m_per_sqrt_s{0.03};
  double orientation_process_sigma_rad_per_sqrt_s{0.05};
  double mahalanobis_gate_threshold{7.81};
  int confirmation_hits{2};
  int confirmation_window{3};
  int max_consecutive_rejections{3};
  double tentative_max_age_s{2.0};
};

enum class FilteredBlockTrackState
{
  kTentative,
  kConfirmed
};

struct FilteredBlockTrack
{
  Eigen::Vector3d position{Eigen::Vector3d::Zero()};
  Eigen::Matrix3d position_covariance{Eigen::Matrix3d::Identity()};
  Eigen::Quaterniond orientation{Eigen::Quaterniond::Identity()};
  Eigen::Matrix3d orientation_covariance{Eigen::Matrix3d::Identity()};
  std::deque<bool> hit_history;
  int consecutive_rejections{0};
  FilteredBlockTrackState state{FilteredBlockTrackState::kTentative};
  double first_update_s{0.0};
  double last_update_s{0.0};
};

Eigen::Vector3d blockPosition(
  const concrete_block_world_model_interfaces::msg::Block & block);

Eigen::Quaterniond blockOrientation(
  const concrete_block_world_model_interfaces::msg::Block & block);

FilteredBlockTrack initializeTrack(
  const BlockObservation & observation,
  double stamp_s,
  const BlockFilterConfig & cfg);

void predict(
  FilteredBlockTrack & track,
  double dt_s,
  const BlockFilterConfig & cfg);

double mahalanobisDistanceSquared(
  const FilteredBlockTrack & track,
  const BlockObservation & observation);

void recordMiss(
  FilteredBlockTrack & track,
  const BlockFilterConfig & cfg);

bool gateAndUpdate(
  FilteredBlockTrack & track,
  const BlockObservation & observation,
  const BlockFilterConfig & cfg,
  std::string & reason);

concrete_block_world_model_interfaces::msg::Block toBlockMsg(
  const FilteredBlockTrack & track,
  concrete_block_world_model_interfaces::msg::Block block);

Eigen::Quaterniond canonicalizeSymmetricOrientation(
  const Eigen::Quaterniond & estimate,
  const Eigen::Quaterniond & measurement);

Eigen::Vector3d rotationVectorInnovation(
  const Eigen::Quaterniond & estimate,
  const Eigen::Quaterniond & measurement);

}  // namespace cbp::world_model
