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
                description="Deprecated; ignored.",
            ),
            DeclareLaunchArgument(
                "params_file",
                default_value=default_world_model_params,
            ),
            Node(
                package="concrete_block_world_model",
                executable="world_model_node",
                name="world_model_node",
                parameters=[
                    LaunchConfiguration("params_file"),
                    {
                        "use_sim_time": LaunchConfiguration("use_sim_time"),
                    },
                ],
                remappings=[
                    ("image", "/zed2i/warped/left/image_rect_color/image_raw"),
                    ("points", "/seyond_points"),
                    ("block_world_model", "/cbp/block_world_model"),
                    ("block_world_model_markers", "/cbp/block_world_model_markers"),
                    ("debug/detection_overlay", "/cbp/debug/detection_overlay"),
                    ("debug/tracking_overlay", "/cbp/debug/tracking_overlay"),
                    ("debug/registration_cutout", "/cbp/debug/registration_cutout"),
                    ("debug/registration_template", "/cbp/debug/registration_template"),
                    ("debug/refine_grasped_roi_input", "/cbp/debug/refine_grasped_roi_input"),
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
