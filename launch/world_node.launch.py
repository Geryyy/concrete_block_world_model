from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_world_model_params = PathJoinSubstitution(
        [
            FindPackageShare("concrete_block_world_model"),
            "config",
            "world_model.yaml",
        ]
    )
    default_scene_discovery_params = PathJoinSubstitution(
        [
            FindPackageShare("concrete_block_world_model"),
            "config",
            "scene_discovery_defaults.yaml",
        ]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="true",
            ),
            DeclareLaunchArgument(
                "pipeline_mode",
                default_value="full",
                description="Deprecated; ignored.",
            ),
            DeclareLaunchArgument(
                "perception_mode",
                default_value="IDLE",
                description="World-model perception mode: IDLE or CONTINUOUS.",
            ),
            DeclareLaunchArgument(
                "params_file",
                default_value=default_world_model_params,
            ),
            DeclareLaunchArgument(
                "scene_discovery_params",
                default_value=default_scene_discovery_params,
            ),
            Node(
                package="concrete_block_world_model",
                executable="world_model_node",
                name="world_model_node",
                parameters=[
                    LaunchConfiguration("params_file"),
                    LaunchConfiguration("scene_discovery_params"),
                    {
                        "use_sim_time": LaunchConfiguration("use_sim_time"),
                        "perception_mode": LaunchConfiguration("perception_mode"),
                    },
                ],
                remappings=[
                    # The only image consumer left is the scene-discovery pose overlay /
                    # capture cache; the raw cloud the capture pairs with is read on the
                    # absolute topic in scene_discovery.capture.cloud_topic.
                    ("image", "/blackfly_rotated/image_rect"),
                    ("block_world_model", "/cbp/block_world_model"),
                    ("block_world_model_markers", "/cbp/block_world_model_markers"),
                    ("block_goal_markers", "/cbp/block_goal_markers"),
                    (
                        "debug/scene_discovery_pose_overlay",
                        "/cbp/debug/scene_discovery_pose_overlay",
                    ),
                ],
                additional_env={
                    "RCUTILS_COLORIZED_OUTPUT": "1",
                    "RCUTILS_CONSOLE_OUTPUT_FORMAT": (
                        "\033[33m[{name}] [{severity}] {message}\033[0m"
                    ),
                },
                output="screen",
            ),
        ]
    )
