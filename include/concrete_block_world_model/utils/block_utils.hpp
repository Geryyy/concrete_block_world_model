#pragma once

#include <cstdint>
#include <string>

#include <geometry_msgs/msg/pose.hpp>
#include "concrete_block_world_model_interfaces/msg/block.hpp"
#include "concrete_block_world_model_interfaces/msg/block_array.hpp"

std::string poseStatusToString(int status);
std::string taskStatusToString(int status);

bool isKnownPoseStatus(int status);
bool isKnownTaskStatus(int status);

// Auto-assigned ids follow the "block_<detection_id>" convention; anything
// else (e.g. seeded "b0") is a user-named block.
std::string detectionBlockId(uint32_t detection_id);
bool isAutoAssignedBlockId(const std::string & id);

// Heuristic 1-sigma measurement uncertainties, to be tuned against field
// data once the probabilistic filter is in place. Coarse poses come from a
// cloud centroid with an offset guess (occlusion-biased), and their identity
// orientation is not a measurement, hence the large sigmas.
inline constexpr double kPrecisePositionSigmaMinM = 0.02;
inline constexpr double kPreciseOrientationSigmaRad = 0.10;
inline constexpr double kCoarsePositionSigmaM = 0.20;
inline constexpr double kCoarseOrientationSigmaRad = 3.14159265358979323846;

// Fills pose_covariance with a diagonal (x, y, z, rot_x, rot_y, rot_z)
// matrix from isotropic position/orientation sigmas.
void setDiagonalPoseCovariance(
  concrete_block_world_model_interfaces::msg::Block & block,
  double sigma_pos_m,
  double sigma_rot_rad);

// Default covariance derived from pose_status, for blocks entering the world
// without a per-measurement estimate (seeded blocks, external upserts).
void setDefaultPoseCovariance(concrete_block_world_model_interfaces::msg::Block & block);

double poseDistance(
  const geometry_msgs::msg::Pose & a,
  const geometry_msgs::msg::Pose & b);
