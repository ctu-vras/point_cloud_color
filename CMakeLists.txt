cmake_minimum_required(VERSION 3.10.2)
project(point_cloud_color)

find_package(catkin REQUIRED
  COMPONENTS
    cv_bridge
    image_transport
    nodelet
    pcl_conversions
    roscpp
    sensor_msgs
    tf2_eigen
    tf2_ros
)
find_package(OpenCV REQUIRED COMPONENTS core)

catkin_package(
  CATKIN_DEPENDS
    cv_bridge
    image_transport
    nodelet
    pcl_conversions
    roscpp
    sensor_msgs
    tf2_eigen
    tf2_ros
  DEPENDS OpenCV
  LIBRARIES ${PROJECT_NAME}_plugins
)

include_directories(${catkin_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

add_library(${PROJECT_NAME}_plugins src/point_cloud_color/point_cloud2_color.cpp)
target_link_libraries(${PROJECT_NAME}_plugins ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})

install(
  TARGETS ${PROJECT_NAME}_plugins
  DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)
install(
  FILES ${PROJECT_NAME}_plugins.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

