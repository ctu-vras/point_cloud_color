/**
 * Point cloud coloring from calibrated cameras.
 * Static image masks can be used to denote ROI for coloring.
 *
 * Configured field data type must match image data type.
 * RGB image must be used with float type (default).
 * Grayscale image must use corresponding unsigned data type.
 */

#include <functional>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <limits>
#include <nodelet/nodelet.h>
#include <opencv2/opencv.hpp>
#include <pluginlib/class_list_macros.hpp>
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_eigen/tf2_eigen.h>
#include <unordered_map>
// needs to be included after tf2_eigen
#include <opencv2/core/eigen.hpp>

namespace {

size_t point_field_type_size(sensor_msgs::PointField::_datatype_type datatype)
{
  switch (datatype)
  {
    case sensor_msgs::PointField::FLOAT32: return 4;
    case sensor_msgs::PointField::FLOAT64: return 8;
    case sensor_msgs::PointField::INT8:    return 1;
    case sensor_msgs::PointField::INT16:   return 2;
    case sensor_msgs::PointField::INT32:   return 4;
    case sensor_msgs::PointField::UINT8:   return 1;
    case sensor_msgs::PointField::UINT16:  return 2;
    default: throw std::runtime_error("Unknown point field data type.");
  }
}

void append_field(const std::string& name,
                  uint32_t count,
                  sensor_msgs::PointField::_datatype_type datatype,
                  sensor_msgs::PointCloud2& cloud)
{
  sensor_msgs::PointField field;
  field.name = name;
  field.offset = cloud.point_step;
  field.datatype = datatype;
  field.count = count;
  cloud.fields.emplace_back(field);
  cloud.point_step += count * point_field_type_size(datatype);
  cloud.row_step = cloud.width * cloud.point_step;
}

float rgb_to_float(const cv::Vec3b& px)
{
  uint32_t rgb = uint32_t(0xff) << 24 | uint32_t(px.val[2]) << 16 | uint32_t(px.val[1]) << 8 | uint32_t(px.val[0]);
  return *reinterpret_cast<float*>(&rgb);
}

bool camera_calibrated(const sensor_msgs::CameraInfo& camera_info)
{
  return camera_info.K[0] != 0.0;
}

void copy_cloud_metadata(const sensor_msgs::PointCloud2& input,
                         sensor_msgs::PointCloud2& output)
{
  output.header = input.header;
  output.height = input.height;
  output.width = input.width;
  output.fields = input.fields;
  output.is_bigendian = input.is_bigendian;
  output.point_step = input.point_step;
  output.row_step = input.row_step;
  output.is_dense = input.is_dense;
}

/**
 * Copy points, assume compatible fields in the head of each point.
 * Only point step is actually checked so that the input fits in the output.
 * Point step is always used, row padding is neglected.
 *
 * @param input Input cloud.
 * @param output Output cloud.
 */
void copy_cloud_data(const sensor_msgs::PointCloud2& input,
                     sensor_msgs::PointCloud2& output)
{
  assert(input.point_step <= output.point_step);
  assert(size_t(input.height) * input.width <= size_t(output.height) * output.width);
  assert(input.width * input.point_step == input.row_step);
  assert(output.width * output.point_step == output.row_step);
  assert(size_t(input.height) * input.width * input.point_step == input.data.size());
  assert(size_t(output.height) * output.width * output.point_step == output.data.size());
  size_t n = output.height * output.width;
  auto in_ptr = input.data.data();
  auto out_ptr = output.data.data();
  for (size_t i = 0; i < n; ++i, in_ptr += input.point_step, out_ptr += output.point_step)
  {
    std::copy(in_ptr, in_ptr + input.point_step, out_ptr);
  }
}

template<class T>
T clip(T value, T min, T max)
{
  if (value < min || std::isnan(value))
    value = min;
  if (value > max)
    value = max;
  return value;
}

enum WarningType
{
  camera_not_ready,
  uncalibrated_camera,
  incompatible_image_type,
  image_too_old,
  transform_not_found
};

struct PairHash
{
  template <class T1, class T2>
  std::size_t operator() (const std::pair<T1, T2> &pair) const
  {
    return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
  }
};

}

namespace point_cloud_color {

/**
 * @brief A nodelet for coloring point clouds from calibrated cameras.
 */
class PointCloudColor : public nodelet::Nodelet {
public:
  PointCloudColor() = default;
  ~PointCloudColor() override = default;
  void onInit() override;
private:
  std::string fixed_frame_ = "odom";
  std::string field_name_ = "rgb";
  int field_type_ = sensor_msgs::PointField::FLOAT32;
  float default_color_ = 0.0;
  int num_cameras_ = 1;
  bool synchronize_ = false;
  double max_image_age_ = 5.0;
  double max_cloud_age_ = 5.0;
  bool use_first_valid_ = true;
  double min_depth_ = 1e-3;
  int image_queue_size_ = 1;
  int cloud_queue_size_ = 1;
  double wait_for_transform_ = 1.0;
  double min_warn_period_ = 10.0;

  tf2_ros::Buffer tf_buffer_ = {ros::Duration(15.0)};
  tf2_ros::TransformListener tf_sub_ = {tf_buffer_};
  std::vector<image_transport::CameraSubscriber> camera_subs_;
  std::vector<image_transport::Subscriber> image_subs_;
  std::vector<ros::Subscriber> camera_info_subs_;
  ros::Subscriber cloud_sub_;
  ros::Publisher cloud_pub_;
  std::vector<cv_bridge::CvImage::ConstPtr> images_;
  std::vector<sensor_msgs::CameraInfo::ConstPtr> cam_infos_;
  std::vector<cv::Mat> camera_masks_;
  std::unordered_map<std::pair<int, int>, ros::Time, PairHash> last_cam_warning_;

  void readParams();
  void setupPublishers();
  void setupSubscribers();
  bool cameraWarnedRecently(int i_cam, int type);
  void updateWarningTime(int i_cam, int type);
  bool imageCompatible(const sensor_msgs::Image& image) const;
  void cameraCallback(const sensor_msgs::Image::ConstPtr &image,
                      const sensor_msgs::CameraInfo::ConstPtr &camera_info, int i);
  void imageCallback(const sensor_msgs::Image::ConstPtr& image, int i);
  void camInfoCallback(const sensor_msgs::CameraInfo::ConstPtr& cam_info, int i);
  void cloudCallback(const sensor_msgs::PointCloud2::ConstPtr &cloud_in);
};

void PointCloudColor::readParams()
{
  ros::NodeHandle &pnh = getPrivateNodeHandle();

  pnh.param("fixed_frame", fixed_frame_, fixed_frame_);
  NODELET_INFO("Fixed frame: %s.", fixed_frame_.c_str());

  pnh.param("field_name", field_name_, field_name_);
  NODELET_INFO("Field name: %s.", field_name_.c_str());

  pnh.param("field_type", field_type_, field_type_);
  if (field_type_ != sensor_msgs::PointField::FLOAT32
      && field_type_ != sensor_msgs::PointField::UINT8
      && field_type_ != sensor_msgs::PointField::UINT16)
  {
    NODELET_ERROR("Unsupported field data type %i used.", field_type_);
    throw std::runtime_error("Unsupported field data type used.");
  }
  NODELET_INFO("Field type: %i.", field_type_);

  if (field_type_ == sensor_msgs::PointField::FLOAT32) {
    // Reinterpret as RGB float.
    std::string default_color_str("0x00000000");
    pnh.param("default_color", default_color_str, default_color_str);
    uint32_t default_color_uint = 0xfffffffful & strtoul(default_color_str.c_str(), nullptr, 0);
    default_color_ = *reinterpret_cast<float *>(&default_color_uint);
    NODELET_INFO("Default color: %#x.", default_color_uint);
  } else {
    // Use as literal.
    pnh.param("default_color", default_color_, default_color_);
    // Clip to valid ranges.
    if (field_type_ == sensor_msgs::PointField::UINT8)
      default_color_ = clip<float>(default_color_, 0.0f, std::numeric_limits<uint8_t>::max());
    else if (field_type_ == sensor_msgs::PointField::UINT16)
      default_color_ = clip<float>(default_color_, 0.0f, std::numeric_limits<uint16_t>::max());
    NODELET_INFO("Default color: %.0f.", default_color_);
  }

  pnh.param("num_cameras", num_cameras_, num_cameras_);
  num_cameras_ = num_cameras_ >= 0 ? num_cameras_ : 0;
  NODELET_INFO("Number of cameras: %i.", num_cameras_);

  camera_masks_.resize(num_cameras_);
  for (int i = 0; i < num_cameras_; i++)
  {
    std::stringstream ss;
    ss << "camera_" << i << "/mask";
    std::string mask_param = ss.str();
    std::string mask_path;
    pnh.param(mask_param, mask_path, mask_path);
    if (!mask_path.empty())
    {
      NODELET_INFO("Camera %i uses mask from %s.", i, mask_path.c_str());
      try
      {
        camera_masks_[i] = cv::imread(mask_path, cv::IMREAD_GRAYSCALE);
      }
      catch (const std::exception& e)
      {
        NODELET_ERROR_STREAM("Error loading mask image: " << e.what());
      }
      if (camera_masks_[i].empty())
      {
        NODELET_ERROR("Reading the mask image failed.");
      }
    }
  }

  pnh.param("synchronize", synchronize_, synchronize_);
  NODELET_INFO("Synchronize image with camera info: %i.", synchronize_);

  pnh.param("max_image_age", max_image_age_, max_image_age_);
  NODELET_INFO("Maximum image age: %.1f s.", max_image_age_);

  pnh.param("max_cloud_age", max_cloud_age_, max_cloud_age_);
  NODELET_INFO("Maximum cloud age: %.1f s.", max_cloud_age_);

  pnh.param("use_first_valid", use_first_valid_, use_first_valid_);
  NODELET_INFO("Use first valid projection: %s.", use_first_valid_ ? "yes" : "no");

  pnh.param("min_depth", min_depth_, min_depth_);
  NODELET_INFO("Minimum depth: %.3f.", min_depth_);

  pnh.param("image_queue_size", image_queue_size_, image_queue_size_);
  image_queue_size_ = image_queue_size_ >= 1 ? image_queue_size_ : 1;
  NODELET_INFO("Image queue size: %i.", image_queue_size_);

  pnh.param("point_cloud_queue_size", cloud_queue_size_, cloud_queue_size_);  // backward compatibility
  pnh.param("cloud_queue_size", cloud_queue_size_, cloud_queue_size_);
  cloud_queue_size_ = cloud_queue_size_ >= 1 ? cloud_queue_size_ : 1;
  NODELET_INFO("Point cloud queue size: %i.", cloud_queue_size_);

  pnh.param("wait_for_transform", wait_for_transform_, wait_for_transform_);
  wait_for_transform_ = wait_for_transform_ >= 0.0 ? wait_for_transform_ : 0.0;
  NODELET_INFO("Wait for transform timeout: %.2f s.", wait_for_transform_);

  pnh.param("min_warn_period", min_warn_period_, min_warn_period_);
  wait_for_transform_ = min_warn_period_ >= 0.0 ? min_warn_period_ : 0.0;
  NODELET_INFO("Minimum period between warnings: %.2f s.", min_warn_period_);
}

void PointCloudColor::setupPublishers()
{
  ros::NodeHandle &nh = getNodeHandle();

  // Advertise colored point cloud topic.
  cloud_pub_ = nh.advertise<sensor_msgs::PointCloud2>("cloud_out", cloud_queue_size_);
}

void PointCloudColor::setupSubscribers()
{
  ros::NodeHandle &nh = getNodeHandle();

  // Subscribe list of camera topics.
  image_transport::ImageTransport it(nh);
  camera_subs_.resize(num_cameras_);
  image_subs_.resize(num_cameras_);
  camera_info_subs_.resize(num_cameras_);
  images_.resize(num_cameras_);
  cam_infos_.resize(num_cameras_);
  image_transport::TransportHints transport_hints("raw", {}, getPrivateNodeHandle());
  for (int i = 0; i < num_cameras_; i++)
  {
    std::stringstream ss;
    ss << "camera_" << i << "/image";
    std::string topic = ss.str();
    topic = nh.resolveName(topic, true);
    NODELET_INFO("Camera %i subscribes to %s.", i, nh.resolveName(topic, true).c_str());
    if (synchronize_)
    {
      camera_subs_[i] = it.subscribeCamera(
          topic, image_queue_size_, (std::bind(&PointCloudColor::cameraCallback,
                                               this, std::placeholders::_1, std::placeholders::_2, i)), {}, transport_hints);
    }
    else
    {
      image_subs_[i] = it.subscribe(
          topic, image_queue_size_, (std::bind(&PointCloudColor::imageCallback, this, std::placeholders::_1, i)), {}, transport_hints);
      std::stringstream ss;
      ss << "camera_" << i << "/camera_info";
      topic = ss.str();
      topic = nh.resolveName(topic, true);
      camera_info_subs_[i] = nh.subscribe<sensor_msgs::CameraInfo>(
          topic, image_queue_size_, (std::bind(&PointCloudColor::camInfoCallback, this, std::placeholders::_1, i)));
    }
  }
  // Subscribe to cloud topic.
  cloud_sub_ = nh.subscribe<sensor_msgs::PointCloud2>("cloud_in", cloud_queue_size_, &PointCloudColor::cloudCallback, this);
}

void PointCloudColor::onInit()
{
  readParams();
  setupPublishers();
  setupSubscribers();
}

bool PointCloudColor::imageCompatible(const sensor_msgs::Image& image) const
{
  // Check image type is compatible with field data type.
  size_t elem_size = image.step / image.width;
  return (field_type_ == sensor_msgs::PointField::FLOAT32 and elem_size == 3)
      || (field_type_ != sensor_msgs::PointField::FLOAT32 and point_field_type_size(field_type_) == elem_size);
}

void PointCloudColor::imageCallback(const sensor_msgs::Image::ConstPtr& image, int i)
{
  NODELET_DEBUG("Image %i received in frame %s.", i, image->header.frame_id.c_str());
  if (!imageCompatible(*image))
  {
    if (!cameraWarnedRecently(i, incompatible_image_type))
    {
      NODELET_WARN("Image with encoding %s cannot be used with field type %i and size %lu.",
                   image->encoding.c_str(), field_type_, point_field_type_size(field_type_));
      updateWarningTime(i, incompatible_image_type);
    }
    return;
  }
  if (field_type_ == sensor_msgs::PointField::FLOAT32)
  {
    images_[i] = cv_bridge::toCvShare(image, sensor_msgs::image_encodings::BGR8);
  }
  else
  {
    images_[i] = cv_bridge::toCvShare(image);
  }
}

void PointCloudColor::camInfoCallback(const sensor_msgs::CameraInfo::ConstPtr& cam_info, int i)
{
  NODELET_DEBUG("Camera info %i received in frame %s.", i, cam_info->header.frame_id.c_str());
  if (!camera_calibrated(*cam_info))
  {
    if (!cameraWarnedRecently(i, uncalibrated_camera))
    {
      NODELET_WARN("Camera %i is not calibrated.", i);
      updateWarningTime(i, uncalibrated_camera);
    }
    return;
  }
  cam_infos_[i] = cam_info;
}

void PointCloudColor::cameraCallback(const sensor_msgs::Image::ConstPtr& image,
                                     const sensor_msgs::CameraInfo::ConstPtr& camera_info,
                                     const int i) {
  NODELET_DEBUG("Camera %i received with image frame %s and camera info frame %s.",
                i, image->header.frame_id.c_str(), camera_info->header.frame_id.c_str());
  imageCallback(image, i);
  camInfoCallback(camera_info, i);
}

bool PointCloudColor::cameraWarnedRecently(int i, int type)
{
  std::pair<int, int> key(i, type);
  auto it = last_cam_warning_.find(key);
  if (it == last_cam_warning_.end())
  {
    return false;
  }
  return (ros::Time::now() - last_cam_warning_[key]).toSec() < min_warn_period_;
}

void PointCloudColor::updateWarningTime(int i, int type)
{
  std::pair<int, int> key(i, type);
  last_cam_warning_[key] = ros::Time::now();
}

void PointCloudColor::cloudCallback(const sensor_msgs::PointCloud2::ConstPtr &cloud_in) {
  auto cloud_age = (ros::Time::now() - cloud_in->header.stamp).toSec();
  if (cloud_age > max_cloud_age_)
  {
    NODELET_WARN("Skipping old cloud (%.1f s > %.1f s).", cloud_age, max_cloud_age_);
    return;
  }

  if (cloud_in->width == 0 || cloud_in->height == 0) {
    NODELET_WARN("Skipping empty cloud %s.", cloud_in->header.frame_id.c_str());
    return;
  }

  const size_t num_points = size_t(cloud_in->width) * cloud_in->height;

  // Create cloud copy with extra field.
  auto cloud_out = boost::make_shared<sensor_msgs::PointCloud2>();
  copy_cloud_metadata(*cloud_in, *cloud_out);
  // TODO: Allow re-using existing field.
  append_field(field_name_, 1, field_type_, *cloud_out);
  cloud_out->data.resize(size_t(cloud_out->height) * cloud_out->width * cloud_out->point_step);
  copy_cloud_data(*cloud_in, *cloud_out);

  sensor_msgs::PointCloud2Iterator<float> x_begin(*cloud_out, "x");
  sensor_msgs::PointCloud2Iterator<float> color_begin_f(*cloud_out, field_name_);
  sensor_msgs::PointCloud2Iterator<uint8_t> color_begin_u8(*cloud_out, field_name_);
  sensor_msgs::PointCloud2Iterator<uint16_t> color_begin_u16(*cloud_out, field_name_);

  // Set default color.
  for (size_t j = 0; j < num_points; ++j)
  {
    switch (field_type_)
    {
      case sensor_msgs::PointField::UINT8:
      {
        *(color_begin_u8 + j) = uint8_t(default_color_);
        break;
      }
      case sensor_msgs::PointField::UINT16:
      {
        *(color_begin_u16 + j) = uint16_t(default_color_);
        break;
      }
      case sensor_msgs::PointField::FLOAT32:
      {
        *(color_begin_f + j) = default_color_;
        break;
      }
    }
  }

  // Initialize vector with projection distances from image center, used as a quality measure.
  std::vector<float> dist(num_points, std::numeric_limits<float>::infinity());
  cv::Mat empty_mat;
  cv::Mat zero_vec = cv::Mat::zeros(3, 1, CV_32FC1);

  for (int i = 0; i < num_cameras_; ++i) {
    if (!images_[i] || !cam_infos_[i])
    {
      if (!cameraWarnedRecently(i, camera_not_ready))
      {
        NODELET_WARN("Camera %i has not been received yet.", i);
        updateWarningTime(i, camera_not_ready);
      }
      continue;
    }

    // Check relative age of the point cloud and the image.
    // Skip the image if the time span is too large.
    const double image_age = (cloud_out->header.stamp - images_[i]->header.stamp).toSec();
    if (image_age > max_image_age_)
    {
      if (!cameraWarnedRecently(i, image_too_old))
      {
        NODELET_WARN("Skipping image %s much older than cloud (%.1f s > %.1f s).",
                     images_[i]->header.frame_id.c_str(), image_age, max_image_age_);
        updateWarningTime(i, image_too_old);
      }
      continue;
    }

    cv::Mat camera_matrix(3, 3, CV_64FC1, const_cast<void *>(reinterpret_cast<const void *>(&cam_infos_[i]->K[0])));
    cv::Mat dist_coeffs(1, int(cam_infos_[i]->D.size()), CV_64FC1, const_cast<void *>(reinterpret_cast<const void *>(&cam_infos_[i]->D[0])));
    camera_matrix.convertTo(camera_matrix, CV_32FC1);
    dist_coeffs.convertTo(dist_coeffs, CV_32FC1);

    geometry_msgs::TransformStamped cloud_to_cam_tf;
    try
    {
      auto wait = wait_for_transform_ - (ros::Time::now() - images_[i]->header.stamp).toSec();
      cloud_to_cam_tf = tf_buffer_.lookupTransform(
          images_[i]->header.frame_id, images_[i]->header.stamp, // target frame and time
          cloud_out->header.frame_id.c_str(), cloud_in->header.stamp, // source frame and time
          fixed_frame_, ros::Duration(wait));
    }
    catch (tf2::TransformException &e)
    {
      if (!cameraWarnedRecently(i, transform_not_found))
      {
        NODELET_WARN("Could not transform cloud from %s to %s. Skipping the image.",
                     cloud_out->header.frame_id.c_str(), images_[i]->header.frame_id.c_str());
        updateWarningTime(i, transform_not_found);
      }
      continue;
    }
    Eigen::Isometry3f cloud_to_cam = Eigen::Isometry3f(tf2::transformToEigen(cloud_to_cam_tf));

    // Gather points in front of camera for projection.
    sensor_msgs::PointCloud2Iterator<float> x_iter = x_begin;
    std::vector<size_t> indices;
    std::vector<cv::Vec3f> x_cam_vec;
    for (size_t j = 0; j < num_points; ++j, ++x_iter)
    {
      // Continue if we already have got a color.
      if (use_first_valid_ && std::isfinite(dist[j]))
      {
        // TODO: Default
        continue;
      }

      // Transform to camera frame.
      Eigen::Map<Eigen::Vector3f> x_cloud(&x_iter[0]);
      Eigen::Vector3f x_cam = cloud_to_cam * x_cloud;

      // Skip NaN points and points behind camera.
      if (!std::isfinite(x_cam(0)) || !std::isfinite(x_cam(1)) || !std::isfinite(x_cam(2)) || x_cam(2) < min_depth_)
      {
        // TODO: Default
        continue;
      }
      indices.push_back(j);
      x_cam_vec.emplace_back(cv::Vec3f(x_cam(0), x_cam(1), x_cam(2)));
    }

    if (indices.empty())
    {
      continue;
    }

    std::vector<cv::Vec2f> u_vec;
    cv::projectPoints(x_cam_vec, zero_vec, zero_vec, camera_matrix, dist_coeffs, u_vec);

    for (int j = 0; j < indices.size(); ++j)
    {
      // Skip points outside the image.
      const float x = u_vec[j][0];
      const float y = u_vec[j][1];
      if (x < 0.0 || x > float(cam_infos_[i]->width - 1) || !std::isfinite(x)
          || y < 0.0 || y > float(cam_infos_[i]->height - 1) || !std::isfinite(y))
      {
        continue;
      }

      int xi = int(std::round(x));
      int yi = int(std::round(y));

      // Apply static mask with image ROI to be used for coloring.
      if (!camera_masks_[i].empty() && !camera_masks_[i].at<uint8_t>(yi, xi))
      {
        // Pixel masked out.
        continue;
      }

      // Keep color from projection closest to image center.
      const float r = hypot(float(cam_infos_[i]->width) / 2 - x,
                            float(cam_infos_[i]->height) / 2 - y);
      if (r >= dist[indices[j]])
      {
        continue;
      }
      dist[indices[j]] = r;

      int offset = int(indices[j]);
      switch (field_type_)
      {
        case sensor_msgs::PointField::UINT8:
        {
          *(color_begin_u8 + offset) =  images_[i]->image.at<uint8_t>(yi, xi);
          break;
        }
        case sensor_msgs::PointField::UINT16:
        {
          *(color_begin_u16 + offset) =  images_[i]->image.at<uint16_t>(yi, xi);
          break;
        }
        case sensor_msgs::PointField::FLOAT32:
        {
          *(color_begin_f + offset) = rgb_to_float(images_[i]->image.at<cv::Vec3b>(yi, xi));
          break;
        }
      }
    }
  }
  cloud_pub_.publish(cloud_out);
}

} /* namespace point_cloud_color */

PLUGINLIB_EXPORT_CLASS(point_cloud_color::PointCloudColor, nodelet::Nodelet) // NOLINT(cert-err58-cpp)
