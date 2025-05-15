cmake_minimum_required(VERSION 3.0.2)
project(local_motion_planning)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  geometry_msgs
  message_generation
)

# add_message_files(
#   FILES
#   DynamicObstacles.msg
#   DynamicObstacle.msg
#   OptimalConfiguration.msg
# )

# generate_messages(
#   DEPENDENCIES
#   std_msgs
#   geometry_msgs
# )

# catkin_package(
#   CATKIN_DEPENDS
#   roscpp
#   rospy
#   std_msgs
#   geometry_msgs
#   message_runtime
# )

# include_directories(
#   ${catkin_INCLUDE_DIRS}
# )

# catkin_install_python(PROGRAMS
#     scripts/dynamic_obstacle_processor.py
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )