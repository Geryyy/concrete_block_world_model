#include "concrete_block_world_model/utils/block_utils.hpp"
#include "concrete_block_world_model/utils/visu_utils.hpp"

std_msgs::msg::ColorRGBA make_color(
  float r, float g, float b, float a)
{
  std_msgs::msg::ColorRGBA c;
  c.r = r; c.g = g; c.b = b; c.a = a;
  return c;
}

visualization_msgs::msg::Marker
make_block_marker(
  const Block & block,
  const std::string & frame,
  int id,
  const rclcpp::Time & stamp)
{
  visualization_msgs::msg::Marker m;
  m.header.frame_id = frame;
  m.header.stamp = stamp;

  m.ns = "blocks";
  m.id = id;
  m.action = visualization_msgs::msg::Marker::ADD;

  if (block.pose_status == Block::POSE_PRECISE) {
    // Only in precise pose state the orientation is known to orient a cube properly.
    m.type = visualization_msgs::msg::Marker::CUBE;
    m.scale.x = 0.9;
    m.scale.y = 0.6;
    m.scale.z = 0.6;
  } else {
    m.type = visualization_msgs::msg::Marker::SPHERE;
    m.scale.x = 0.9;
    m.scale.y = 0.9;
    m.scale.z = 0.9;
  }

  m.pose = block.pose;

  // Rough concrete block size (adjust!)


  // Color by pose status
  if (block.pose_status == Block::POSE_PRECISE) {
    m.color = make_color(0.0, 1.0, 0.0, 0.8);   // green
  } else {
    m.color = make_color(1.0, 0.5, 0.0, 0.8);   // orange
  }

  return m;
}

visualization_msgs::msg::Marker
make_text_marker(const Block & block, std::string frame, int id, const rclcpp::Time & stamp)
{
  visualization_msgs::msg::Marker m;
  m.header.frame_id = frame;
  m.header.stamp = stamp;

  m.ns = "block_labels";
  m.id = id;
  m.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;

  m.pose = block.pose;
  m.pose.position.z += 0.3;

  m.scale.z = 0.12;
  m.color = make_color(1, 1, 1, 1);

  m.text =
    block.id +
    "\npose=" + poseStatusToString(block.pose_status) +
    "\ntask=" + taskStatusToString(block.task_status);

  return m;
}
