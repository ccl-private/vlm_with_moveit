cmake_minimum_required(VERSION 3.22)
project(vision_moveit_bridge)

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)

add_executable(moveit_target_bridge src/moveit_target_bridge.cpp)
ament_target_dependencies(moveit_target_bridge
  geometry_msgs moveit_msgs moveit_ros_planning_interface rclcpp sensor_msgs std_msgs)
target_compile_features(moveit_target_bridge PUBLIC cxx_std_17)

install(TARGETS moveit_target_bridge DESTINATION lib/${PROJECT_NAME})
install(PROGRAMS scripts/activate_fake_controllers.py DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
ament_package()
