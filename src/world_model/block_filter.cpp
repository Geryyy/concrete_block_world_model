#include "concrete_block_world_model/world_model/block_filter.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

#include "concrete_block_world_model/utils/block_utils.hpp"

namespace cbp::world_model
{
namespace
{

using Block = concrete_block_world_model_interfaces::msg::Block;

constexpr double kMinVariance = 1.0e-9;

Eigen::Matrix3d covarianceFromBlock(
  const Block & block,
  size_t offset,
  double fallback_sigma)
{
  Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
  bool valid = true;
  for (size_t i = 0; i < 3; ++i) {
    const double value = block.pose_covariance[(i + offset) * 6 + i + offset];
    if (value <= 0.0 || !std::isfinite(value)) {
      valid = false;
      break;
    }
    cov(static_cast<int>(i), static_cast<int>(i)) = std::max(value, kMinVariance);
  }

  if (valid) {
    return cov;
  }

  const double variance = fallback_sigma * fallback_sigma;
  return Eigen::Matrix3d::Identity() * variance;
}

Eigen::Matrix3d positionObservationCovariance(const BlockObservation & observation)
{
  const double fallback_sigma =
    observation.precise ? kPrecisePositionSigmaMinM : kCoarsePositionSigmaM;
  return covarianceFromBlock(observation.block, 0, fallback_sigma);
}

Eigen::Matrix3d orientationObservationCovariance(const BlockObservation & observation)
{
  const double fallback_sigma =
    observation.precise ? kPreciseOrientationSigmaRad : kCoarseOrientationSigmaRad;
  return covarianceFromBlock(observation.block, 3, fallback_sigma);
}

void pushHistory(
  FilteredBlockTrack & track,
  bool hit,
  const BlockFilterConfig & cfg)
{
  track.hit_history.push_back(hit);
  while (track.hit_history.size() > static_cast<size_t>(std::max(1, cfg.confirmation_window))) {
    track.hit_history.pop_front();
  }

  const int hits = static_cast<int>(
    std::count(track.hit_history.begin(), track.hit_history.end(), true));
  if (hits >= std::max(1, cfg.confirmation_hits)) {
    track.state = FilteredBlockTrackState::kConfirmed;
  }
}

double covarianceTraceToConfidence(const Eigen::Matrix3d & covariance)
{
  // A 2 cm isotropic position sigma is near-certain; a 50 cm isotropic sigma is
  // effectively unusable for block association.
  constexpr double kBestTrace = 3.0 * 0.02 * 0.02;
  constexpr double kWorstTrace = 3.0 * 0.50 * 0.50;
  const double trace = covariance.trace();
  const double confidence = 1.0 - ((trace - kBestTrace) / (kWorstTrace - kBestTrace));
  return std::clamp(confidence, 0.0, 1.0);
}

}  // namespace

Eigen::Vector3d blockPosition(const Block & block)
{
  return Eigen::Vector3d(
    block.pose.position.x,
    block.pose.position.y,
    block.pose.position.z);
}

Eigen::Quaterniond blockOrientation(const Block & block)
{
  Eigen::Quaterniond q(
    block.pose.orientation.w,
    block.pose.orientation.x,
    block.pose.orientation.y,
    block.pose.orientation.z);
  if (q.norm() <= 0.0 || !std::isfinite(q.norm())) {
    return Eigen::Quaterniond::Identity();
  }
  q.normalize();
  return q;
}

FilteredBlockTrack initializeTrack(
  const BlockObservation & observation,
  double stamp_s,
  const BlockFilterConfig & cfg)
{
  FilteredBlockTrack track;
  track.position = blockPosition(observation.block);
  track.position_covariance = positionObservationCovariance(observation);
  track.orientation = blockOrientation(observation.block);
  track.orientation_covariance = orientationObservationCovariance(observation);
  track.first_update_s = stamp_s;
  track.last_update_s = stamp_s;
  pushHistory(track, true, cfg);
  return track;
}

void predict(
  FilteredBlockTrack & track,
  double dt_s,
  const BlockFilterConfig & cfg)
{
  const double dt = std::max(0.0, dt_s);
  const double position_variance =
    cfg.position_process_sigma_m_per_sqrt_s *
    cfg.position_process_sigma_m_per_sqrt_s * dt;
  const double orientation_variance =
    cfg.orientation_process_sigma_rad_per_sqrt_s *
    cfg.orientation_process_sigma_rad_per_sqrt_s * dt;
  track.position_covariance += Eigen::Matrix3d::Identity() * position_variance;
  track.orientation_covariance += Eigen::Matrix3d::Identity() * orientation_variance;
}

double mahalanobisDistanceSquared(
  const FilteredBlockTrack & track,
  const BlockObservation & observation)
{
  const Eigen::Vector3d innovation = blockPosition(observation.block) - track.position;
  const Eigen::Matrix3d innovation_covariance =
    track.position_covariance + positionObservationCovariance(observation);
  const Eigen::LDLT<Eigen::Matrix3d> solver(innovation_covariance);
  if (solver.info() != Eigen::Success) {
    return std::numeric_limits<double>::infinity();
  }
  return innovation.dot(solver.solve(innovation));
}

void recordMiss(
  FilteredBlockTrack & track,
  const BlockFilterConfig & cfg)
{
  pushHistory(track, false, cfg);
}

Eigen::Quaterniond canonicalizeSymmetricOrientation(
  const Eigen::Quaterniond & estimate,
  const Eigen::Quaterniond & measurement)
{
  const Eigen::Quaterniond flips[] = {
    Eigen::Quaterniond::Identity(),
    Eigen::Quaterniond(Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitX())),
    Eigen::Quaterniond(Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitY())),
    Eigen::Quaterniond(Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitZ())),
  };

  Eigen::Quaterniond best = measurement.normalized();
  double best_dot = std::abs(estimate.normalized().dot(best));
  for (const auto & flip : flips) {
    Eigen::Quaterniond candidate = (measurement * flip).normalized();
    const double dot = std::abs(estimate.normalized().dot(candidate));
    if (dot > best_dot) {
      best = candidate;
      best_dot = dot;
    }
  }
  return best;
}

Eigen::Vector3d rotationVectorInnovation(
  const Eigen::Quaterniond & estimate,
  const Eigen::Quaterniond & measurement)
{
  Eigen::Quaterniond error = estimate.normalized().conjugate() * measurement.normalized();
  if (error.w() < 0.0) {
    error.coeffs() *= -1.0;
  }
  Eigen::AngleAxisd axis_angle(error);
  if (!std::isfinite(axis_angle.angle()) || axis_angle.axis().hasNaN()) {
    return Eigen::Vector3d::Zero();
  }
  return axis_angle.axis() * axis_angle.angle();
}

bool gateAndUpdate(
  FilteredBlockTrack & track,
  const BlockObservation & observation,
  const BlockFilterConfig & cfg,
  std::string & reason)
{
  const double d2 = mahalanobisDistanceSquared(track, observation);
  if (d2 > cfg.mahalanobis_gate_threshold) {
    ++track.consecutive_rejections;
    pushHistory(track, false, cfg);
    std::ostringstream oss;
    oss << "position gate rejected: mahalanobis_d2=" << d2 <<
      " threshold=" << cfg.mahalanobis_gate_threshold <<
      " consecutive_rejections=" << track.consecutive_rejections;
    reason = oss.str();

    if (track.consecutive_rejections >= std::max(1, cfg.max_consecutive_rejections)) {
      track = initializeTrack(observation, track.last_update_s, cfg);
      reason += "; reinitialized track after consecutive rejections";
      return true;
    }
    return false;
  }

  const Eigen::Vector3d z = blockPosition(observation.block);
  const Eigen::Matrix3d r = positionObservationCovariance(observation);
  const Eigen::Matrix3d s = track.position_covariance + r;
  const Eigen::Matrix3d gain = track.position_covariance * s.inverse();
  const Eigen::Vector3d innovation = z - track.position;
  track.position += gain * innovation;
  track.position_covariance =
    (Eigen::Matrix3d::Identity() - gain) * track.position_covariance;

  if (observation.precise) {
    const Eigen::Quaterniond measured =
      canonicalizeSymmetricOrientation(track.orientation, blockOrientation(observation.block));
    const Eigen::Vector3d rot_innovation =
      rotationVectorInnovation(track.orientation, measured);
    const Eigen::Matrix3d rot_r = orientationObservationCovariance(observation);
    const Eigen::Matrix3d rot_s = track.orientation_covariance + rot_r;
    const Eigen::Matrix3d rot_gain = track.orientation_covariance * rot_s.inverse();
    const Eigen::Vector3d delta = rot_gain * rot_innovation;
    if (delta.norm() > 1.0e-12) {
      track.orientation =
        (track.orientation * Eigen::Quaterniond(Eigen::AngleAxisd(delta.norm(), delta.normalized()))).
        normalized();
    }
    track.orientation_covariance =
      (Eigen::Matrix3d::Identity() - rot_gain) * track.orientation_covariance;
  }

  track.consecutive_rejections = 0;
  pushHistory(track, true, cfg);
  reason = "accepted";
  return true;
}

Block toBlockMsg(
  const FilteredBlockTrack & track,
  Block block)
{
  block.pose.position.x = track.position.x();
  block.pose.position.y = track.position.y();
  block.pose.position.z = track.position.z();
  const Eigen::Quaterniond q = track.orientation.normalized();
  block.pose.orientation.x = q.x();
  block.pose.orientation.y = q.y();
  block.pose.orientation.z = q.z();
  block.pose.orientation.w = q.w();

  block.pose_covariance.fill(0.0);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      block.pose_covariance[i * 6 + j] =
        track.position_covariance(static_cast<int>(i), static_cast<int>(j));
      block.pose_covariance[(i + 3) * 6 + j + 3] =
        track.orientation_covariance(static_cast<int>(i), static_cast<int>(j));
    }
  }
  block.confidence = static_cast<float>(covarianceTraceToConfidence(track.position_covariance));
  return block;
}

}  // namespace cbp::world_model
