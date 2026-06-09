#pragma once

#include <string>
#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <std_msgs/msg/color_rgba.h>
#include "concrete_block_world_model_interfaces/msg/block.hpp"
#include "concrete_block_world_model_interfaces/msg/block_array.hpp"

using Block = concrete_block_world_model_interfaces::msg::Block;
using BlockArray = concrete_block_world_model_interfaces::msg::BlockArray;


std_msgs::msg::ColorRGBA make_color(
  float r, float g, float b, float a);

visualization_msgs::msg::Marker
make_block_marker(
  const Block & block,
  const std::string & frame,
  int id,
  const rclcpp::Time & stamp);

visualization_msgs::msg::Marker
make_text_marker(const Block & block, std::string frame, int id, const rclcpp::Time & stamp);
