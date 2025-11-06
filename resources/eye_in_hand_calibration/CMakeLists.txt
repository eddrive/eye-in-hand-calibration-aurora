cmake_minimum_required(VERSION 3.8)
project(eye_in_hand_calibration)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ========================================
# FIND DEPENDENCIES
# ========================================
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(aurora_ndi_ros2_driver REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

# Find OpenCV
find_package(OpenCV 4 REQUIRED)

# Find Eigen3
find_package(Eigen3 REQUIRED)

# Find yaml-cpp
find_package(PkgConfig REQUIRED)
pkg_check_modules(YAML_CPP REQUIRED yaml-cpp)

# ========================================
# INCLUDE DIRECTORIES
# ========================================
include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${YAML_CPP_INCLUDE_DIRS}
)

# ========================================
# BUILD LIBRARY
# ========================================
add_library(eye_in_hand_calibration_lib
  src/chessboard_detector.cpp
  src/pose_estimator.cpp
  src/aurora_synchronizer.cpp
  src/sample_manager.cpp
  src/calibration_solver.cpp
  src/hand_eye_calibrator.cpp
)

target_include_directories(eye_in_hand_calibration_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(eye_in_hand_calibration_lib
  ${OpenCV_LIBRARIES}
  ${YAML_CPP_LIBRARIES}
  Eigen3::Eigen
)

ament_target_dependencies(eye_in_hand_calibration_lib
  rclcpp
  sensor_msgs
  geometry_msgs
  cv_bridge
  aurora_ndi_ros2_driver
  tf2
  tf2_ros
  tf2_geometry_msgs
)

# ========================================
# BUILD EXECUTABLE
# ========================================
add_executable(hand_eye_calibrator_node
  nodes/hand_eye_calibrator_node.cpp
)

target_link_libraries(hand_eye_calibrator_node
  eye_in_hand_calibration_lib
)

ament_target_dependencies(hand_eye_calibrator_node
  rclcpp
)

# ========================================
# INSTALL
# ========================================
# Install library
install(TARGETS eye_in_hand_calibration_lib
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

# Install executable
install(TARGETS
  hand_eye_calibrator_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install header files
install(DIRECTORY include/
  DESTINATION include
)

# Install config files
install(DIRECTORY
  config/
  DESTINATION share/${PROJECT_NAME}/config
)

# Install launch files
install(DIRECTORY
  launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# ========================================
# EXPORT
# ========================================
ament_export_include_directories(include)
ament_export_libraries(eye_in_hand_calibration_lib)
ament_export_dependencies(
  rclcpp
  sensor_msgs
  geometry_msgs
  cv_bridge
  aurora_pub
  tf2
  tf2_ros
  tf2_geometry_msgs
  OpenCV
  Eigen3
  yaml-cpp
)

# ========================================
# TESTING
# ========================================
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
  
  # Optional: Add unit tests
  # find_package(ament_cmake_gtest REQUIRED)
  # ament_add_gtest(test_chessboard_detector test/test_chessboard_detector.cpp)
  # target_link_libraries(test_chessboard_detector eye_in_hand_calibration_lib)
endif()

ament_package()