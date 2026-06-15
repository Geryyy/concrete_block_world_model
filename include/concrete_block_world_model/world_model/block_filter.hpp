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
  bool publish_new_tracks_without_prior{false};
  bool temporal_bootstrap_enabled{true};
  int temporal_bootstrap_min_stable_observations{3};
  double temporal_bootstrap_max_mask_area_change_ratio{0.35};
  double temporal_bootstrap_max_mask_centroid_jump_px{45.0};
  double temporal_bootstrap_max_position_jump_m{0.20};
  bool operational_confidence_enabled{false};
  double confidence_stale_after_s{1.0};
  double confidence_age_half_life_s{5.0};
  double confidence_miss_penalty{0.5};
  double confidence_rejection_penalty{0.25};
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
  double last_observation_s{0.0};
  int stable_observations{0};
  int last_mask_pixels{0};
  bool has_last_mask_centroid{false};
  double last_mask_centroid_x_px{0.0};
  double last_mask_centroid_y_px{0.0};
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

bool trackHasStableTemporalBootstrap(
  const FilteredBlockTrack & track,
  const BlockFilterConfig & cfg);

concrete_block_world_model_interfaces::msg::Block toBlockMsg(
  const FilteredBlockTrack & track,
  concrete_block_world_model_interfaces::msg::Block block);

double trackConfidence(
  const FilteredBlockTrack & track,
  const BlockFilterConfig & cfg,
  double now_s);

concrete_block_world_model_interfaces::msg::Block toBlockMsg(
  const FilteredBlockTrack & track,
  concrete_block_world_model_interfaces::msg::Block block,
  const BlockFilterConfig & cfg,
  double now_s);

Eigen::Quaterniond canonicalizeSymmetricOrientation(
  const Eigen::Quaterniond & estimate,
  const Eigen::Quaterniond & measurement);

Eigen::Vector3d rotationVectorInnovation(
  const Eigen::Quaterniond & estimate,
  const Eigen::Quaterniond & measurement);

}  // namespace cbp::world_model
