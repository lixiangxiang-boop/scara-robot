cmake_minimum_required(VERSION 3.8)
project(scara_gripper_plugin)

find_package(ament_cmake REQUIRED)
find_package(gazebo_dev REQUIRED)
find_package(gazebo_ros REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

add_library(scara_gripper_plugin SHARED src/gripper_plugin.cpp)
target_include_directories(scara_gripper_plugin PRIVATE
  ${gazebo_dev_INCLUDE_DIRS} ${gazebo_ros_INCLUDE_DIRS} ${GAZEBO_INCLUDE_DIRS})
target_link_libraries(scara_gripper_plugin
  ${gazebo_ros_LIBRARIES} ${GAZEBO_LIBRARIES}
  rclcpp::rclcpp std_msgs::std_msgs)

install(TARGETS scara_gripper_plugin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin)

ament_export_libraries(scara_gripper_plugin)
ament_package()
