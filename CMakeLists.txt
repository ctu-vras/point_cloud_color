cmake_minimum_required(VERSION 3.20)
project(point_cloud_color)
set (CMAKE_CXX_STANDARD 20)

find_package(ament_cmake REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(point_cloud_transport REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(OpenCV REQUIRED) #COMPONENTS calib3d core)

add_executable(point_cloud_color src/point_cloud2_color.cpp)
ament_target_dependencies(point_cloud_color
  cv_bridge
  rclcpp
  rclcpp_components
  sensor_msgs
  tf2_eigen
  tf2_ros
  image_transport
  point_cloud_transport
  OpenCV
)
# target_include_directories(point_cloud_color PUBLIC
#     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
#     $<INSTALL_INTERFACE:include>)

target_compile_features(point_cloud_color PUBLIC c_std_99 cxx_std_23)

install(TARGETS point_cloud_color
    DESTINATION lib/${PROJECT_NAME})

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()