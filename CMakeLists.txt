cmake_minimum_required(VERSION 3.8)
project(concrete_block_world_model)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(concrete_block_perception_interfaces REQUIRED)
find_package(concrete_block_world_model_interfaces REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(OpenCV REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(ros2_yolos_cpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(yaml-cpp REQUIRED)

add_executable(world_model_node
  src/nodes/perception_orchestrator_core.cpp
  src/nodes/perception_orchestrator_node.cpp
  src/nodes/perception_orchestrator_pipeline.cpp
  src/nodes/perception_orchestrator_services.cpp
  src/nodes/perception_orchestrator_tf_roi.cpp
  src/utils/block_utils.cpp
  src/utils/coarse_pose_utils.cpp
  src/utils/img_utils.cpp
  src/utils/world_model_utils.cpp
  src/world_model/config_loader.cpp
  src/world_model/refine_flow.cpp
  src/world_model/registration_flow.cpp
  src/world_model/roi_processing.cpp
  src/world_model/roi_refinement.cpp
  src/world_model/scene_discovery_flow.cpp
  src/world_model/state_manager.cpp
)

target_include_directories(world_model_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(world_model_node
  Eigen3::Eigen
  ${OpenCV_LIBS}
  yaml-cpp
)

ament_target_dependencies(world_model_node
  concrete_block_perception_interfaces
  concrete_block_world_model_interfaces
  cv_bridge
  geometry_msgs
  message_filters
  rclcpp
  rclcpp_action
  ros2_yolos_cpp
  sensor_msgs
  std_msgs
  tf2
  tf2_geometry_msgs
  tf2_ros
  vision_msgs
  visualization_msgs
)

install(TARGETS
  world_model_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  config
  launch
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_cpplint_FOUND TRUE)
  set(ament_cmake_copyright_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()

  ament_add_gtest(test_world_model_utils
    test/test_world_model_utils.cpp
    src/utils/world_model_utils.cpp
  )
  if(TARGET test_world_model_utils)
    target_include_directories(test_world_model_utils PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
    )
    ament_target_dependencies(test_world_model_utils
      concrete_block_world_model_interfaces
      std_msgs
      vision_msgs
      visualization_msgs
    )
  endif()

endif()

ament_package()
