cmake_minimum_required(VERSION 3.0.2)
project(global_path_planning)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
)

# Find SNOPT library
set(SNOPT_DIR "/home/plg/libsnopt7_cpp")
include_directories(${SNOPT_DIR})
link_directories(${SNOPT_DIR})

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

catkin_package(
  CATKIN_DEPENDS roscpp std_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(
  PROGRAMS 
    scripts/global_path_planning.py
    scripts/plot_global_path.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

# add_executable(sntoya src/sntoya.cpp)
# target_link_libraries(sntoya ${catkin_LIBRARIES} snopt7_cpp)
