cmake_minimum_required(VERSION 3.8)
project(concrete_block_world_model)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(concrete_block_world_model_interfaces REQUIRED)
find_package(crane_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(point_cloud_transport REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(yaml-cpp REQUIRED)

add_executable(world_model_node
  src/nodes/perception_orchestrator_core.cpp
  src/nodes/perception_orchestrator_grasp_tf.cpp
  src/nodes/perception_orchestrator_node.cpp
  src/nodes/perception_orchestrator_services.cpp
  src/nodes/perception_orchestrator_world.cpp
  src/utils/block_utils.cpp
  src/utils/world_model_utils.cpp
  src/world_model/config_loader.cpp
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
  concrete_block_world_model_interfaces
  crane_msgs
  cv_bridge
  geometry_msgs
  point_cloud_transport
  rclcpp
  sensor_msgs
  std_msgs
  tf2
  tf2_geometry_msgs
  tf2_ros
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
  # Formatting is clang-format's, via pre-commit (CLAUDE.md, .pre-commit-config.yaml), and
  # uncrustify disagrees with it on this package's whole tree -- it would reformat files
  # clang-format is happy with, and pre-commit would then reformat them back.
  set(ament_cmake_uncrustify_FOUND TRUE)
  # Same conflict on the Python side: ruff-format (pre-commit) writes double quotes and
  # flake8's Q000 demands single ones, so every launch file fails a lint nobody can satisfy.
  # Pyflakes' actual checks are not lost -- pre-commit's ruff-check runs the F rules.
  set(ament_cmake_flake8_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()

  ament_add_gtest(test_world_model_utils
    test/test_world_model_utils.cpp
    src/utils/block_utils.cpp
    src/utils/world_model_utils.cpp
  )
  if(TARGET test_world_model_utils)
    target_include_directories(test_world_model_utils PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
    )
    ament_target_dependencies(test_world_model_utils
      concrete_block_world_model_interfaces
      crane_msgs
      geometry_msgs
      std_msgs
      tf2
      tf2_geometry_msgs
      visualization_msgs
    )
    # The shipped config is an oracle here: the vehicle has to be described in exactly one place,
    # which is a claim about the file and not about any function.
    target_compile_definitions(test_world_model_utils PRIVATE
      CBP_WORLD_MODEL_CONFIG_PATH="${CMAKE_CURRENT_SOURCE_DIR}/config/world_model.yaml")
    target_link_libraries(test_world_model_utils yaml-cpp Eigen3::Eigen)
  endif()

endif()

ament_package()
