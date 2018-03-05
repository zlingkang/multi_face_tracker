cmake_minimum_required(VERSION 2.8.3)
project(multi_face_tracker)

option(POSE_ESTIMATION "estimate detected faces' pose using dlib" ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, option is: Release" FORCE)
endif(NOT CMAKE_BUILD_TYPE)
#or catkin_make -DCMAKE_BUILD_TYPE=Release

if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
    set (X86 TRUE)
else()
    set (X86 FALSE)
endif ()

if (X86)
    set(USE_AVX_INSTRUCTIONS ON CACHE BOOL "Enable Dlib AVX Instructions")
    include(/home/lingkang/projects/dlib-19.4/dlib/cmake)
    # Use MSSE4 Instrcution for high speed
    add_definitions(-msse4 -std=c++11)
else()
    include(/home/odroid/projects/dlib_for_arm/dlib/dlib/cmake)
    add_definitions(-mfpu=neon -std=c++11)
endif()

# Add header files
include_directories(include)

## Find catkin and any catkin packages
find_package(catkin REQUIRED COMPONENTS 
    roscpp 
    rospy 
    std_msgs
    genmsg
    actionlib
    actionlib_msgs
    image_transport
    cv_bridge
    sensor_msgs
    geometry_msgs
    message_generation
    face_expression)

## Declare ROS messages and services
# add_message_files(FILES Num.msg)
# add_service_files(FILES AddTwoInts.srv)

add_action_files(
    DIRECTORY action
    FILES FaceDetection.action
)

## Generate added messages and services
generate_messages(
    DEPENDENCIES 
    std_msgs
    actionlib_msgs
)

## Declare a catkin package
catkin_package(
    CATKIN_DEPENDS message_runtime actionlib_msgs actionlib
    )

find_package(OpenCV)
if(OpenCV_FOUND)
    include_directories(include ${OpenCV_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS} ${actionlib_msgs_PACKAGE_PATH})
    add_executable(face_ctrl src/face_ctrl.cpp src/head_pose_estimation.cpp)
    add_executable(face_detection src/face_detection.cpp)
    target_link_libraries(face_ctrl ${catkin_LIBRARIES}  dlib ${OpenCV_LIBS})
    target_link_libraries(face_detection ${catkin_LIBRARIES} dlib ${OpenCV_LIBS})
endif()

