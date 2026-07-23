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
                    ("image", "/blackfly_rotated/image_rect"),
                    ("points", "/seyond/points"),
                    ("block_world_model", "/cbp/block_world_model"),
                    ("block_world_model_markers", "/cbp/block_world_model_markers"),
                    ("block_goal_markers", "/cbp/block_goal_markers"),
                    ("debug/detection_overlay", "/cbp/debug/detection_overlay"),
                    ("debug/yolo_service_debug_image", "/cbp/debug/yolo_service_debug_image"),
                    ("debug/continuous_merged_mask", "/cbp/debug/continuous_merged_mask"),
                    ("debug/tracking_overlay", "/cbp/debug/tracking_overlay"),
                    ("debug/refine_grasped_roi_input", "/cbp/debug/refine_grasped_roi_input"),
                    ("timing/continuous_seg_ms", "/cbp/timing/continuous_seg_ms"),
                    ("timing/continuous_cutout_ms", "/cbp/timing/continuous_cutout_ms"),
                    ("timing/continuous_coarse_ms", "/cbp/timing/continuous_coarse_ms"),
                    (
                        "timing/continuous_registration_ms",
                        "/cbp/timing/continuous_registration_ms",
                    ),
                    ("timing/continuous_upsert_ms", "/cbp/timing/continuous_upsert_ms"),
                    ("timing/continuous_total_ms", "/cbp/timing/continuous_total_ms"),
                    ("timing/continuous_detections", "/cbp/timing/continuous_detections"),
                    ("timing/continuous_accepted", "/cbp/timing/continuous_accepted"),
                    ("timing/continuous_rejected", "/cbp/timing/continuous_rejected"),
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
