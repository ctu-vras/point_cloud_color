/**
 * Point cloud coloring from calibrated cameras.
 * Static image masks can be used to denote ROI for coloring.
 *
 * TODO: Mask out robot model dynamically.
 */

#include <functional>
#include <boost/format.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <limits>
#include <map>
#include <nodelet/nodelet.h>
#include <opencv2/opencv.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pluginlib/class_list_macros.hpp>
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_eigen/tf2_eigen.h>
// needs to be included after tf2_eigen
#include <opencv2/core/eigen.hpp>

namespace point_cloud_color {

size_t getFieldIndex(const sensor_msgs::PointCloud2 &cloud, const std::string &fieldName) {
  for (size_t i = 0; i < cloud.fields.size(); i++) {
    if (cloud.fields[i].name == fieldName) {
      return i;
    }
  }
  throw std::runtime_error("Field " + fieldName + " not found in pointcloud.");
}

cv::Mat matFromCloud(sensor_msgs::PointCloud2::Ptr cloud, const std::string& fieldName, const int numElements) {
  const int numPoints = cloud->width * cloud->height;
  size_t fieldIndex = getFieldIndex(*cloud, fieldName);
  void *data = static_cast<void *>(&cloud->data[cloud->fields[fieldIndex].offset]);
  int matType;
  switch (cloud->fields[fieldIndex].datatype) {
  case sensor_msgs::PointField::FLOAT32: matType = CV_32FC1; break;
  case sensor_msgs::PointField::FLOAT64: matType = CV_64FC1; break;
  case sensor_msgs::PointField::UINT8:   matType = CV_8UC1;  break;
  default:
    assert(0);
    break;
  }
  return cv::Mat(numPoints, numElements, matType, data, cloud->point_step);
}

sensor_msgs::PointCloud2::Ptr createCloudWithColorFrom(const sensor_msgs::PointCloud2::ConstPtr inCloud) {
  // Create a copy to fix fields - use correct count 1 instead of 0.
  pcl::PCLPointCloud2 inCloudFixed;
  pcl_conversions::toPCL(*inCloud, inCloudFixed);
  for (auto &f : inCloudFixed.fields) {
    if (f.count == 0) {
      f.count = 1;
    }
  }

  pcl::PCLPointField rgbField;
  rgbField.name = "rgb";
  rgbField.datatype = sensor_msgs::PointField::FLOAT32;
  rgbField.count = 1;
  rgbField.offset = 0;
  pcl::PCLPointCloud2 rgbCloud;
  rgbCloud.header = inCloudFixed.header;
  rgbCloud.width = inCloudFixed.width;
  rgbCloud.height = inCloudFixed.height;
  rgbCloud.fields.push_back(rgbField);
  rgbCloud.is_bigendian = inCloudFixed.is_bigendian;
  rgbCloud.is_dense = inCloudFixed.is_dense;
  rgbCloud.point_step = 4;
  rgbCloud.row_step = rgbCloud.width * rgbCloud.point_step;
  rgbCloud.data.resize(rgbCloud.row_step * rgbCloud.height, 0);

  pcl::PCLPointCloud2 outPcl;
  pcl::concatenateFields(inCloudFixed, rgbCloud, outPcl);

  sensor_msgs::PointCloud2::Ptr outCloud(new sensor_msgs::PointCloud2);
  pcl_conversions::moveFromPCL(outPcl, *outCloud);
  outCloud->header.stamp = inCloud->header.stamp; // PCL conversion crops precision of the timestamp
  return outCloud;
}

/**
 * @brief A nodelet for coloring point clouds from calibrated cameras.
 */
class PointCloudColor : public nodelet::Nodelet {
public:
  PointCloudColor();
  ~PointCloudColor() override = default;
  void onInit() override;
private:
  tf2_ros::Buffer tfBuffer;
  tf2_ros::TransformListener transformListener;
  std::vector<image_transport::CameraSubscriber> cameraSubscribers;
  ros::Subscriber pointCloudSub;
  ros::Publisher pointCloudPub;
  std::vector<cv_bridge::CvImage::ConstPtr> images;
  std::vector<sensor_msgs::CameraInfo::ConstPtr> cameraInfos;
  std::vector<cv::Mat> cameraMasks;
  float defaultColor;
  std::string fixedFrame;
  size_t numCameras;
  double maxImageAge;
  bool useFirstValid;
  int imageQueueSize;
  uint32_t pointCloudQueueSize;
  double waitForTransform;
  std::map<std::string, ros::Time> lastCamWarning;
  double minCamWarningInterval;
  void cameraCallback(const sensor_msgs::Image::ConstPtr &imageMsg,
                      const sensor_msgs::CameraInfo::ConstPtr &cameraInfoMsg, int iCam);
  void pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr &cloudMsg);
};

PointCloudColor::PointCloudColor() :
    tfBuffer(ros::Duration(15.0)),
    transformListener(tfBuffer),
    defaultColor(0.0),
    fixedFrame("odom"),
    numCameras(1),
    maxImageAge(10.0),
    useFirstValid(true),
    imageQueueSize(1),
    pointCloudQueueSize(1),
    waitForTransform(1.0),
    minCamWarningInterval(1.0) {
}

void PointCloudColor::onInit() {
  NODELET_DEBUG("%s: Initializing...", getName().c_str());

  ros::NodeHandle &nh = getNodeHandle();
  ros::NodeHandle &pnh = getPrivateNodeHandle();

  // Get and process parameters.
  pnh.param("fixed_frame", fixedFrame, fixedFrame);
  NODELET_INFO("%s: Fixed frame: %s.", getName().c_str(), fixedFrame.c_str());
  std::string defaultColorStr("0x00000000");
  pnh.param("default_color", defaultColorStr, defaultColorStr);
  unsigned long defaultColorUl = 0xfffffffful & strtoul(defaultColorStr.c_str(), nullptr, 0);
  defaultColor = *reinterpret_cast<float *>(&defaultColorUl);
  NODELET_INFO("%s: Default color: %#lx.", getName().c_str(), defaultColorUl);
  int _numCameras;
  pnh.param("num_cameras", _numCameras, static_cast<int>(numCameras));
  numCameras = static_cast<size_t>(_numCameras);
  numCameras = numCameras >= 0 ? numCameras : 0;
  NODELET_INFO("%s: Number of cameras: %zu.", getName().c_str(), numCameras);
  pnh.param("max_image_age", maxImageAge, maxImageAge);
  NODELET_INFO("%s: Maximum image age: %.1f s.", getName().c_str(), maxImageAge);
  pnh.param("use_first_valid", useFirstValid, useFirstValid);
  NODELET_INFO("%s: Use first valid projection: %s.", getName().c_str(), useFirstValid ? "yes" : "no");
  pnh.param("image_queue_size", imageQueueSize, imageQueueSize);
  imageQueueSize = imageQueueSize >= 1 ? imageQueueSize : 1;
  NODELET_INFO("%s: Image queue size: %i.", getName().c_str(), imageQueueSize);
  int _pointCloudQueueSize;
  pnh.param("point_cloud_queue_size", _pointCloudQueueSize, static_cast<int>(pointCloudQueueSize));
  pointCloudQueueSize = static_cast<uint32_t>(_pointCloudQueueSize);
  pointCloudQueueSize = pointCloudQueueSize >= 1 ? pointCloudQueueSize : 1;
  NODELET_INFO("%s: Point cloud queue size: %u.", getName().c_str(), pointCloudQueueSize);
  pnh.param("wait_for_transform", waitForTransform, waitForTransform);
  waitForTransform = waitForTransform >= 0.0 ? waitForTransform : 0.0;
  NODELET_INFO("%s: Wait for transform timeout: %.2f s.", getName().c_str(), waitForTransform);

  // Subscribe list of camera topics.
  image_transport::ImageTransport it(nh);
  cameraSubscribers.resize(numCameras);
  cameraMasks.resize(numCameras);
  images.resize(numCameras);
  cameraInfos.resize(numCameras);
  for (int iCam = 0; iCam < numCameras; iCam++) {
    std::string cameraTopic = nh.resolveName((boost::format("camera_%i/image") % iCam).str(), true);
    std::string cameraMaskParam = (boost::format("camera_%i/mask") % iCam).str();
    std::string cameraMaskPath;
    pnh.param(cameraMaskParam, cameraMaskPath, cameraMaskPath);
    if (!cameraMaskPath.empty()) {
      cameraMasks[iCam] = cv::imread(cameraMaskPath, cv::IMREAD_GRAYSCALE);
      NODELET_INFO("%s: Camera %i: Using camera mask from %s.", getName().c_str(), iCam, cameraMaskPath.c_str());
    }
    NODELET_INFO("%s: Camera %i: Subscribing camera topic %s.", getName().c_str(), iCam, cameraTopic.c_str());
    cameraSubscribers[iCam] = it.subscribeCamera(cameraTopic, imageQueueSize,
                                                 (std::bind(&PointCloudColor::cameraCallback, this, std::placeholders::_1, std::placeholders::_2, iCam)));
  }

  // Advertise colored point cloud topic.
  pointCloudPub = nh.advertise<sensor_msgs::PointCloud2>("cloud_out", pointCloudQueueSize);

  // Subscribe point cloud topic.
  pointCloudSub = nh.subscribe<sensor_msgs::PointCloud2>("cloud_in", pointCloudQueueSize,
                                                         &PointCloudColor::pointCloudCallback, this);
}

void PointCloudColor::cameraCallback(const sensor_msgs::Image::ConstPtr &imageMsg, const sensor_msgs::CameraInfo::ConstPtr &cameraInfoMsg, const int iCam) {
  NODELET_DEBUG("%s: Camera %i: Image frame: %s, camera info frame: %s.",
                getName().c_str(), iCam, imageMsg->header.frame_id.c_str(), cameraInfoMsg->header.frame_id.c_str());
  // Use frame id from image message for both image and camera info to ensure consistency.
  images[iCam] = cv_bridge::toCvShare(imageMsg, sensor_msgs::image_encodings::BGR8);
  cameraInfos[iCam] = cameraInfoMsg;
}

void PointCloudColor::pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr &cloudMsg) {
  const size_t numPoints = cloudMsg->width * cloudMsg->height;
  const std::string cloudFrame = cloudMsg->header.frame_id;
  if (numPoints == 0 || cloudMsg->data.empty()) {
    NODELET_WARN("%s: Skipping empty point cloud in frame %s.", getName().c_str(), cloudFrame.c_str());
    return;
  }
  sensor_msgs::PointCloud2::Ptr outCloud = createCloudWithColorFrom(cloudMsg);
  cv::Mat rgbMat = matFromCloud(outCloud, "rgb", 1);
  rgbMat.setTo(defaultColor);
  // Initialize vector with projection distances from image center, used as a quality measure.
  std::vector<float> dist(numPoints, std::numeric_limits<float>::infinity());
  for (int iCam = 0; iCam < numCameras; iCam++) {
    if (!images[iCam] || !cameraInfos[iCam]) {
      NODELET_DEBUG("%s: Camera image %i has not been received yet...", getName().c_str(), iCam);
      continue;
    }
    // Check relative age of the point cloud and the image. Skip the image if the time span is to large.
    const double imageAge = (outCloud->header.stamp - images[iCam]->header.stamp).toSec();
    if (imageAge > maxImageAge) {
      if ((ros::Time::now() - lastCamWarning[images[iCam]->header.frame_id]).toSec() >= minCamWarningInterval) {
        NODELET_WARN("%s: Skipping image %s %.1f (> %.1f) s older than point cloud...",
                     getName().c_str(), images[iCam]->header.frame_id.c_str(), imageAge, maxImageAge);
        lastCamWarning[images[iCam]->header.frame_id] = ros::Time::now();
      }
      continue;
    }

    geometry_msgs::TransformStamped cloudToCamStamped;
    try
    {
      cloudToCamStamped = tfBuffer.lookupTransform(
        images[iCam]->header.frame_id, images[iCam]->header.stamp, // target frame and time
        cloudFrame, cloudMsg->header.stamp, // source frame and time
        fixedFrame, ros::Duration(waitForTransform));
    } catch (tf2::TransformException &e) {
      if ((ros::Time::now() - lastCamWarning[images[iCam]->header.frame_id]).toSec() >= minCamWarningInterval) {
        NODELET_WARN("%s: Could not transform point cloud from %s to %s. Skipping the image...",
                     getName().c_str(), cloudFrame.c_str(), images[iCam]->header.frame_id.c_str());
        lastCamWarning[images[iCam]->header.frame_id] = ros::Time::now();
      }
      continue;
    }

    cv::Mat camRotation, camTranslation;
    cv::eigen2cv(Eigen::Matrix3d(tf2::transformToEigen(cloudToCamStamped).linear()), camRotation);
    cv::eigen2cv(Eigen::Vector3d(tf2::transformToEigen(cloudToCamStamped).translation()), camTranslation);
    cv::Mat objectPoints;
    matFromCloud(outCloud, "x", 3).convertTo(objectPoints, CV_64FC1);
    // Rigid transform with points in rows: (X_C)^T = (R * X_L + t)^T.
    objectPoints = objectPoints * camRotation.t() + cv::repeat(camTranslation.t(), objectPoints.rows, 1);
    cv::Mat cameraMatrix(3, 3, CV_64FC1, const_cast<void *>(reinterpret_cast<const void *>(&cameraInfos[iCam]->K[0])));
    cv::Mat distCoeffs(1, static_cast<int>(cameraInfos[iCam]->D.size()), CV_64FC1, const_cast<void *>(reinterpret_cast<const void *>(&cameraInfos[iCam]->D[0])));
    cv::Mat imagePoints;
    cv::projectPoints(objectPoints.reshape(3), cv::Mat::zeros(3, 1, CV_64FC1), cv::Mat::zeros(3, 1, CV_64FC1), cameraMatrix, distCoeffs, imagePoints);

    for (int iPoint = 0; iPoint < numPoints; iPoint++) {
      // Continue if we already have got a color.
      if (useFirstValid && dist[iPoint] < DBL_MAX) {
        continue;
      }
      const double z = objectPoints.at<double>(iPoint, 2);
      // Skip points behind the camera.
      if (z <= 0.0) {
        continue;
      }
      const cv::Vec2d pt = imagePoints.at<cv::Vec2d>(iPoint, 0);
      const double x = round(pt.val[0]);
      const double y = round(pt.val[1]);
      // Skip points outside the image.
      if (x < 0.0 || y < 0.0 || x >= static_cast<double>(cameraInfos[iCam]->width) || y >= static_cast<double>(cameraInfos[iCam]->height)) {
        continue;
      }
      // Apply static mask with image ROI to be used for coloring.
      const int yi = static_cast<int>(y);
      const int xi = static_cast<int>(x);
      if (!cameraMasks[iCam].empty() && !cameraMasks[iCam].at<uint8_t>(yi, xi)) {
        // Pixel masked out.
        continue;
      }
      const float r = hypot(static_cast<float>(cameraInfos[iCam]->width / 2.0 - xi), static_cast<float>(cameraInfos[iCam]->height / 2.0 - yi));
      if (r >= dist[iPoint]) {
        // Keep color from the previous projection closer to image center.
        continue;
      }
      dist[iPoint] = r;
      const cv::Vec3b px = images[iCam]->image.at<cv::Vec3b>(static_cast<int>(y), static_cast<int>(x));
      uint32_t rgb = static_cast<uint32_t>(px.val[2]) << 16 | static_cast<uint32_t>(px.val[1]) << 8 | static_cast<uint32_t>(px.val[0]);
      rgbMat.at<float>(iPoint, 0) = *reinterpret_cast<float*>(&rgb);
    }
  }
  pointCloudPub.publish(outCloud);
}

} /* namespace point_cloud_color */

PLUGINLIB_EXPORT_CLASS(point_cloud_color::PointCloudColor, nodelet::Nodelet) // NOLINT(cert-err58-cpp)
