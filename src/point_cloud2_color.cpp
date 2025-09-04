// SPDX-License-Identifier: BSD-3-Clause
// SPDX-FileCopyrightText: Czech Technical University in Prague

/**
 * Point cloud coloring from calibrated cameras.
 * Static image masks can be used to denote ROI for coloring.
 *
 * Configured field data type must match image data type.
 * RGB image must be used with float type (default).
 * Grayscale image must use corresponding unsigned data type.
 */


#include <functional>
#include <limits>
#include <unordered_map>

#include <cv_bridge/cv_bridge.hpp>
#include <image_transport/image_transport.hpp>          
#include <point_cloud_transport/point_cloud_transport.hpp>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>

#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_eigen/tf2_eigen.hpp>                  

#include <opencv2/opencv.hpp>
// needs to be included after tf2_eigen
#include <opencv2/core/eigen.hpp>

namespace {

using sensor_msgs::msg::PointCloud2;
using sensor_msgs::msg::PointField;
using sensor_msgs::msg::CameraInfo;

auto other_c = cv::Vec3b(0, 0, 0);
auto road_c = cv::Vec3b(128, 64, 128);
auto sky_c = cv::Vec3b(235, 206, 135);

size_t point_field_type_size(uint8_t datatype)
{
  switch (datatype)
  {
    case PointField::FLOAT32: return 4;
    case PointField::FLOAT64: return 8;
    case PointField::INT8:    return 1;
    case PointField::INT16:   return 2;
    case PointField::INT32:   return 4;
    case PointField::UINT8:   return 1;
    case PointField::UINT16:  return 2;
    default: throw std::runtime_error("Unknown point field data type.");
  }
}

void append_field(const std::string& name,
                  uint32_t count,
                  uint8_t datatype,
                  PointCloud2& cloud)
{
  PointField field;
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
  // ABGR (little-endian float reinterpretation, same as ROS1 trick)
  uint32_t rgb = (uint32_t(0xff) << 24) |
                 (uint32_t(px.val[2]) << 16) |
                 (uint32_t(px.val[1]) << 8)  |
                  uint32_t(px.val[0]);
  float f;
  std::memcpy(&f, &rgb, sizeof(f)); // defined type-pun
  return f;

  // used to be 
  // return *reinterpret_cast<float*>(&rgb);
}

bool camera_calibrated(const CameraInfo& camera_info)
{
  return camera_info.k[0] != 0.0;
}

void copy_cloud_metadata(const PointCloud2& input,
                         PointCloud2& output)
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
void copy_cloud_data(const PointCloud2& input,
                     PointCloud2& output)
{
  assert(input.point_step <= output.point_step);
  assert(static_cast<size_t>(input.height) * input.width <=
         static_cast<size_t>(output.height) * output.width);
  assert(input.width * input.point_step == input.row_step);
  assert(output.width * output.point_step == output.row_step);
  assert(static_cast<size_t>(input.height) * input.width * input.point_step == input.data.size());
  assert(static_cast<size_t>(output.height) * output.width * output.point_step == output.data.size());

  size_t n = static_cast<size_t>(output.height) * output.width;
  const auto* in_ptr  = input.data.data();
  auto*       out_ptr = output.data.data();
  for (size_t i = 0; i < n; ++i, in_ptr += input.point_step, out_ptr += output.point_step)
  {
    std::copy(in_ptr, in_ptr + input.point_step, out_ptr);
  }
}

template<class T, std::enable_if_t<std::is_floating_point<T>::value, int> = 0>
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
class PointCloudColor : public rclcpp::Node {

public:
  PointCloudColor()
  : Node("point_cloud_color"),
    tf_buffer_(this->get_clock()),
    tf_sub_(tf_buffer_)
  {
  }
  
  PointCloudColor(const rclcpp::NodeOptions & options)
  : Node("point_cloud_color", options),
    tf_buffer_(this->get_clock()),
    tf_sub_(tf_buffer_)
  {
  }

  void onInit() {
    readParams();
    setupPublishers();
    setupSubscribers();
  }
  
  ~PointCloudColor() override = default;

private:
  std::string fixed_frame_ = "odom";
  std::string field_name_ = "rgb";
  int field_type_ = sensor_msgs::msg::PointField::FLOAT32;
  float default_color_ = 0.0;
  bool semantic_segmentation_ = false;
  int road_cost_ = 0;
  int other_cost_ = 0;
  int num_cameras_ = 1;
  bool synchronize_ = false;
  bool print_delay_ = false;
  double max_image_age_ = 5.0;
  double max_cloud_age_ = 5.0;
  bool use_first_valid_ = true;
  double min_depth_ = 1e-3;
  int image_queue_size_ = 1;
  int cloud_queue_size_ = 1;
  double wait_for_transform_ = 1.0;
  double min_warn_period_ = 10.0;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_sub_;

  std::vector<image_transport::CameraSubscriber> camera_subs_;
  std::vector<image_transport::Subscriber> image_subs_;
  std::vector<rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr> camera_info_subs_;
  point_cloud_transport::Subscriber cloud_sub_;
  point_cloud_transport::Publisher cloud_pub_;

  std::vector<cv_bridge::CvImage::ConstPtr> images_;
  std::vector<sensor_msgs::msg::CameraInfo::ConstSharedPtr> cam_infos_;
  std::vector<cv::Mat> camera_masks_;
  std::unordered_map<std::pair<int, int>, rclcpp::Time, PairHash> last_cam_warning_;

  void readParams();
  void setupPublishers();
  void setupSubscribers();
  bool cameraWarnedRecently(int i_cam, int type);
  void updateWarningTime(int i_cam, int type);
  bool imageCompatible(const sensor_msgs::msg::Image& image) const;
  void cameraCallback(const sensor_msgs::msg::Image::ConstSharedPtr &image,
                      const sensor_msgs::msg::CameraInfo::ConstSharedPtr &camera_info, int i);
  void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr& image, int i);
  void camInfoCallback(const sensor_msgs::msg::CameraInfo::ConstSharedPtr& cam_info, int i);
  void cloudCallback(const sensor_msgs::msg::PointCloud2::ConstPtr &cloud_in);
};

void PointCloudColor::readParams()
{  
  fixed_frame_        = this->declare_parameter("fixed_frame", fixed_frame_);
  RCLCPP_INFO(this->get_logger(), "Fixed frame: %s.", fixed_frame_.c_str());

  field_name_         = this->declare_parameter("field_name", field_name_);
  RCLCPP_INFO(this->get_logger(), "Field name: %s.", field_name_.c_str());

  field_type_         = this->declare_parameter("field_type", field_type_);
  if (field_type_ != sensor_msgs::msg::PointField::FLOAT32 &&
      field_type_ != sensor_msgs::msg::PointField::UINT8 &&
      field_type_ != sensor_msgs::msg::PointField::UINT16)
  {
    RCLCPP_ERROR(this->get_logger(), "Unsupported field data type %i used.", field_type_);
    throw std::runtime_error("Unsupported field data type used.");
  }
  RCLCPP_INFO(this->get_logger(), "Field type: %i.", field_type_);



  if (field_type_ == sensor_msgs::msg::PointField::FLOAT32)
  {
    std::string default_color_str = this->declare_parameter("default_color", std::string("0x00000000"));
    uint32_t default_color_uint = strtoul(default_color_str.c_str(), nullptr, 0);
    std::memcpy(&default_color_, &default_color_uint, sizeof(default_color_));
    RCLCPP_INFO(this->get_logger(), "Default color: %#x.", default_color_uint);
  }
  else
  {
    default_color_ = this->declare_parameter("default_color", default_color_);
    if (field_type_ == sensor_msgs::msg::PointField::UINT8) {
      default_color_ = clip<float>(default_color_, 0.0f, static_cast<float>(std::numeric_limits<uint8_t>::max()));
    } else if (field_type_ == sensor_msgs::msg::PointField::UINT16)
      default_color_ = clip<float>(default_color_, 0.0f, static_cast<float>(std::numeric_limits<uint16_t>::max()));
    RCLCPP_INFO(this->get_logger(), "Default color: %.0f.", default_color_);
  }

  num_cameras_ = this->declare_parameter("num_cameras", num_cameras_);
  num_cameras_ = std::max(0, num_cameras_);
  RCLCPP_INFO(this->get_logger(), "Number of cameras: %i.", num_cameras_);

  camera_masks_.resize(num_cameras_);
  for (int i = 0; i < num_cameras_; i++)
  {
    std::string mask_param = "camera_" + std::to_string(i) + "/mask";
    std::string mask_path  = this->declare_parameter(mask_param, std::string(""));
    if (!mask_path.empty())
    {
      RCLCPP_INFO(this->get_logger(), "Camera %i uses mask from %s.", i, mask_path.c_str());
      try
      {
        camera_masks_[i] = cv::imread(mask_path, cv::IMREAD_GRAYSCALE);
      }
      catch (const std::exception & e)
      {
        RCLCPP_ERROR(this->get_logger(), "Error loading mask image: %s", e.what());
      }
      if (camera_masks_[i].empty())
      {
        RCLCPP_ERROR(this->get_logger(), "Reading the mask image failed.");
      }
    }
  }

  semantic_segmentation_ = this->declare_parameter("semantic_segmentation", semantic_segmentation_);
  synchronize_        = this->declare_parameter("synchronize", synchronize_);
  print_delay_        = this->declare_parameter("print_delay", print_delay_);
  max_image_age_      = this->declare_parameter("max_image_age", max_image_age_);
  max_cloud_age_      = this->declare_parameter("max_cloud_age", max_cloud_age_);
  use_first_valid_    = this->declare_parameter("use_first_valid", use_first_valid_);
  min_depth_          = this->declare_parameter("min_depth", min_depth_);
  image_queue_size_   = std::max(1, static_cast<int>(this->declare_parameter("image_queue_size", image_queue_size_)));
  cloud_queue_size_   = std::max(1, static_cast<int>(this->declare_parameter("cloud_queue_size", cloud_queue_size_)));
  wait_for_transform_ = std::max(0.0, this->declare_parameter("wait_for_transform", wait_for_transform_));
  min_warn_period_    = std::max(0.0, this->declare_parameter("min_warn_period", min_warn_period_));
  road_cost_ = this->declare_parameter("road_cost", road_cost_);
  other_cost_ = this->declare_parameter("other_cost", other_cost_);

  RCLCPP_INFO(this->get_logger(), "Semantic segmentation: %s.", semantic_segmentation_ ? "yes" : "no");
  RCLCPP_INFO(this->get_logger(), "Road cost: %d Other cost: %d", road_cost_, other_cost_);
  RCLCPP_INFO(this->get_logger(), "Synchronize: %i", synchronize_);
  RCLCPP_INFO(this->get_logger(), "Print delay: %i", print_delay_);
  RCLCPP_INFO(this->get_logger(), "Maximum image age: %.1f s.", max_image_age_);
  RCLCPP_INFO(this->get_logger(), "Maximum cloud age: %.1f s.", max_cloud_age_);
  RCLCPP_INFO(this->get_logger(), "Use first valid projection: %s.", use_first_valid_ ? "yes" : "no");
  RCLCPP_INFO(this->get_logger(), "Minimum depth: %.3f.", min_depth_);
  RCLCPP_INFO(this->get_logger(), "Image queue size: %i", image_queue_size_);
  RCLCPP_INFO(this->get_logger(), "Cloud queue size: %i", cloud_queue_size_);
  RCLCPP_INFO(this->get_logger(), "Wait for transform timeout: %.2f s.", wait_for_transform_);
  RCLCPP_INFO(this->get_logger(), "Minimum period between warnings: %.2f s.", min_warn_period_);
}

void PointCloudColor::setupPublishers()
{
  RCLCPP_WARN(this->get_logger(), "A\n");
  point_cloud_transport::PointCloudTransport pct(shared_from_this());
  RCLCPP_WARN(this->get_logger(), "B\n");
  cloud_pub_ = pct.advertise("cloud_out", rclcpp::SystemDefaultsQoS().get_rmw_qos_profile());
  RCLCPP_WARN(this->get_logger(), "C\n");
}

void PointCloudColor::setupSubscribers()
{
  RCLCPP_WARN(this->get_logger(), "D\n");
  image_transport::ImageTransport it(shared_from_this());

  camera_subs_.resize(num_cameras_);
  image_subs_.resize(num_cameras_);
  camera_info_subs_.resize(num_cameras_);
  images_.resize(num_cameras_);
  cam_infos_.resize(num_cameras_);

  image_transport::TransportHints transport_hints(this);
  RCLCPP_WARN(this->get_logger(), "E\n");

  for (int i = 0; i < num_cameras_; i++)
  {
    std::string image_topic = "camera_" + std::to_string(i) + "/image";
    RCLCPP_INFO(this->get_logger(), "Camera %i subscribes to %s.", i, image_topic.c_str());

    if (synchronize_)
    {
      camera_subs_[i] = it.subscribeCamera(
        image_topic, image_queue_size_,
        (std::bind(&PointCloudColor::cameraCallback, this, std::placeholders::_1, std::placeholders::_2, i))
      );
      RCLCPP_WARN(this->get_logger(), "F\n");

    }
    else
    {
      image_subs_[i] = it.subscribe(
        image_topic, image_queue_size_,
        std::bind(&PointCloudColor::imageCallback, this, std::placeholders::_1, i)
      );
      RCLCPP_WARN(this->get_logger(), "G\n");

      std::string info_topic = "camera_" + std::to_string(i) + "/camera_info";
      camera_info_subs_[i] = this->create_subscription<sensor_msgs::msg::CameraInfo>(
        info_topic, rclcpp::SensorDataQoS(),
        [this, i](sensor_msgs::msg::CameraInfo::SharedPtr msg)
        { camInfoCallback(msg, i); }
      );
      RCLCPP_WARN(this->get_logger(), "H\n");

    }
  }

  point_cloud_transport::PointCloudTransport pct(shared_from_this());
  cloud_sub_ = pct.subscribe(
    "cloud_in", cloud_queue_size_,
    std::bind(&PointCloudColor::cloudCallback, this, std::placeholders::_1)
  );
  RCLCPP_WARN(this->get_logger(), "I\n");

}



// void PointCloudColor::setupSubscribers()
// {
//   image_transport::ImageTransport it(shared_from_this());

//   camera_subs_.resize(num_cameras_);
//   image_subs_.resize(num_cameras_);
//   camera_info_subs_.resize(num_cameras_);
//   images_.resize(num_cameras_);
//   cam_infos_.resize(num_cameras_);

//   image_transport::TransportHints transport_hints(this);
  
//   for (int i = 0; i < num_cameras_; i++)
//   {
//     std::string image_topic = "camera_" + std::to_string(i) + "/image";
//     RCLCPP_INFO(this->get_logger(), "Camera %i subscribes to %s.", i, image_topic.c_str());

//     if (synchronize_)
//     {
//       // You will need to use message_filters for synchronization manually
//       image_subs_[i] = this->create_subscription<sensor_msgs::msg::Image>(
//         image_topic, rclcpp::QoS(image_queue_size_),
//         [this, i](sensor_msgs::msg::Image::SharedPtr msg)
//         {
//           this->images_[i] = msg;
//           this->trySyncCameraData(i);
//         }
//       );

//       std::string info_topic = "camera_" + std::to_string(i) + "/camera_info";
//       camera_info_subs_[i] = this->create_subscription<sensor_msgs::msg::CameraInfo>(
//         info_topic, rclcpp::QoS(image_queue_size_),
//         [this, i](sensor_msgs::msg::CameraInfo::SharedPtr msg)
//         {
//           this->cam_infos_[i] = msg;
//           this->trySyncCameraData(i);
//         }
//       );
//     }
//     else
//     {
//       image_subs_[i] = it.subscribe(
//         image_topic, image_queue_size_,
//         std::bind(&PointCloudColor::imageCallback, this, std::placeholders::_1, i)
//       );

//       std::string info_topic = "camera_" + std::to_string(i) + "/camera_info";
//       camera_info_subs_[i] = this->create_subscription<sensor_msgs::msg::CameraInfo>(
//         info_topic, rclcpp::QoS(image_queue_size_),
//         [this, i](sensor_msgs::msg::CameraInfo::SharedPtr msg)
//         { camInfoCallback(msg, i); }
//       );
//     }
//   }

//   point_cloud_transport::PointCloudTransport pct(shared_from_this());
//   cloud_sub_ = pct.subscribe(
//     "cloud_in", cloud_queue_size_,
//     std::bind(&PointCloudColor::cloudCallback, this, std::placeholders::_1)
//   );
// }


bool PointCloudColor::imageCompatible(const sensor_msgs::msg::Image & image) const
{
  // Check image type is compatible with field data type.
  size_t elem_size = image.step / image.width;
  return ((field_type_ == sensor_msgs::msg::PointField::FLOAT32 && elem_size == 3) ||
          (field_type_ == sensor_msgs::msg::PointField::FLOAT32 && elem_size == 1) ||
          (field_type_ != sensor_msgs::msg::PointField::FLOAT32 && point_field_type_size(field_type_) == elem_size));
}

void PointCloudColor::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr & image, int i)
{
  RCLCPP_DEBUG(this->get_logger(), "Image %i received in frame %s.", i, image->header.frame_id.c_str());

  if (!imageCompatible(*image))
  {
    if (!cameraWarnedRecently(i, static_cast<int>(WarningType::incompatible_image_type)))
    {
      RCLCPP_WARN(this->get_logger(),
                  "Image with encoding %s cannot be used with field type %i (size %zu).",
                  image->encoding.c_str(), field_type_, point_field_type_size(field_type_));
      updateWarningTime(i, static_cast<int>(WarningType::incompatible_image_type));
    }
    return;
  }

  if (field_type_ == sensor_msgs::msg::PointField::FLOAT32)
  {
    images_[i] = cv_bridge::toCvShare(image, sensor_msgs::image_encodings::BGR8);
  }
  else
  {
    images_[i] = cv_bridge::toCvShare(image);
  }
}

void PointCloudColor::camInfoCallback(const sensor_msgs::msg::CameraInfo::ConstSharedPtr & cam_info, int i)
{
  RCLCPP_DEBUG(this->get_logger(),
               "Camera info %i received in frame %s.", i, cam_info->header.frame_id.c_str());

  if (!camera_calibrated(*cam_info))
  {
    if (!cameraWarnedRecently(i, static_cast<int>(WarningType::uncalibrated_camera)))
    {
      RCLCPP_WARN(this->get_logger(), "Camera %i is not calibrated.", i);
      updateWarningTime(i, static_cast<int>(WarningType::uncalibrated_camera));
    }
    return;
  }
  cam_infos_[i] = cam_info;
}


void PointCloudColor::cameraCallback(
  const sensor_msgs::msg::Image::ConstSharedPtr & image,
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & camera_info,
  const int i)
{
  RCLCPP_DEBUG(this->get_logger(),
               "Camera %i received with image frame %s and camera info frame %s.",
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
  return (this->now() - last_cam_warning_[key]).seconds() < min_warn_period_;
}

void PointCloudColor::updateWarningTime(int i, int type)
{
  std::pair<int, int> key(i, type);
  last_cam_warning_[key] = this->now();
}



void PointCloudColor::cloudCallback(const sensor_msgs::msg::PointCloud2::ConstPtr & cloud_in)
{
  auto start = std::chrono::high_resolution_clock::now();

  double cloud_age = (this->now() - rclcpp::Time(cloud_in->header.stamp)).seconds();
  if (cloud_age > max_cloud_age_)
  {
    RCLCPP_WARN(this->get_logger(),
                "Skipping old cloud (%.1f s > %.1f s).", cloud_age, max_cloud_age_);
    return;
  }

  if (cloud_in->width == 0 || cloud_in->height == 0)
  {
    RCLCPP_WARN(this->get_logger(),
                "Skipping empty cloud %s.", cloud_in->header.frame_id.c_str());
    return;
  }

  const size_t num_points = static_cast<size_t>(cloud_in->width) * cloud_in->height;
  
  // Create cloud copy with extra field
  auto cloud_out = std::make_shared<sensor_msgs::msg::PointCloud2>();
  copy_cloud_metadata(*cloud_in, *cloud_out);
  append_field(field_name_, 1, field_type_, *cloud_out);
  cloud_out->data.resize(static_cast<size_t>(cloud_out->height) * cloud_out->width * cloud_out->point_step);
  copy_cloud_data(*cloud_in, *cloud_out);

  sensor_msgs::PointCloud2Iterator<float> x_begin(*cloud_out, "x");
  sensor_msgs::PointCloud2Iterator<float> color_begin_f(*cloud_out, field_name_);
  sensor_msgs::PointCloud2Iterator<uint8_t> color_begin_u8(*cloud_out, field_name_);
  sensor_msgs::PointCloud2Iterator<uint16_t> color_begin_u16(*cloud_out, field_name_);

  // Set default color
  if (semantic_segmentation_) {
    for (size_t j = 0; j < num_points; ++j)
    {
      *(color_begin_u8 + j) = static_cast<uint8_t>(other_cost_);
    }
  } else {
    for (size_t j = 0; j < num_points; ++j)
    {
      switch (field_type_)
      {
        case sensor_msgs::msg::PointField::UINT8:
          *(color_begin_u8 + j) = static_cast<uint8_t>(default_color_);
          break;
        case sensor_msgs::msg::PointField::UINT16:
          *(color_begin_u16 + j) = static_cast<uint16_t>(default_color_);
          break;
        case sensor_msgs::msg::PointField::FLOAT32:
          *(color_begin_f + j) = default_color_;
          break;
      }
    }
  }
  
  
  // Initialize projection distances (used as quality metric)
  std::vector<float> dist(num_points, std::numeric_limits<float>::infinity());
  cv::Mat zero_vec = cv::Mat::zeros(3, 1, CV_32FC1);

  for (int i = 0; i < num_cameras_; ++i)
  {
    if (!images_[i] || !cam_infos_[i])
    {
      if (!cameraWarnedRecently(i, static_cast<int>(WarningType::camera_not_ready)))
      {
        RCLCPP_WARN(this->get_logger(), "Camera %i has not been received yet.", i);
        updateWarningTime(i, static_cast<int>(WarningType::camera_not_ready));
      }
      continue;
    }
    
    // Check relative age of the point cloud and the image.
    // Skip the image if the time span is too large.
    const double image_age = (rclcpp::Time(cloud_out->header.stamp) -
                        rclcpp::Time(images_[i]->header.stamp)).seconds();
    if (image_age > max_image_age_)
    {
      if (!cameraWarnedRecently(i, static_cast<int>(WarningType::image_too_old)))
      {
        RCLCPP_WARN(this->get_logger(),
                    "Skipping image %s much older than cloud (%.1f s > %.1f s).",
                    images_[i]->header.frame_id.c_str(), image_age, max_image_age_);
        updateWarningTime(i, static_cast<int>(WarningType::image_too_old));
      }
      continue;
    }

    // Camera calibration matrices
    cv::Mat camera_matrix(3, 3, CV_64FC1,
                          const_cast<void *>(reinterpret_cast<const void *>(&cam_infos_[i]->k[0])));
    cv::Mat dist_coeffs(1, static_cast<int>(cam_infos_[i]->d.size()), CV_64FC1,
                        const_cast<void *>(reinterpret_cast<const void *>(&cam_infos_[i]->d[0])));
    camera_matrix.convertTo(camera_matrix, CV_32FC1);
    dist_coeffs.convertTo(dist_coeffs, CV_32FC1);

    // Transform lookup
    geometry_msgs::msg::TransformStamped cloud_to_cam_tf;
    try
    {
      double wait = wait_for_transform_ -
                    (this->now() - rclcpp::Time(images_[i]->header.stamp)).seconds();
      cloud_to_cam_tf = tf_buffer_.lookupTransform(
        images_[i]->header.frame_id, rclcpp::Time(images_[i]->header.stamp),
        cloud_out->header.frame_id, rclcpp::Time(cloud_in->header.stamp),
        fixed_frame_, tf2::durationFromSec(wait));
    }
    catch (tf2::TransformException & e)
    {
      if (!cameraWarnedRecently(i, static_cast<int>(WarningType::transform_not_found)))
      {
        RCLCPP_WARN(this->get_logger(),
                    "Could not transform cloud from %s to %s. Skipping the image.",
                    cloud_out->header.frame_id.c_str(),
                    images_[i]->header.frame_id.c_str());
        updateWarningTime(i, static_cast<int>(WarningType::transform_not_found));
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

      int xi = static_cast<int>(std::round(x));
      int yi = static_cast<int>(std::round(y));

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

      if (semantic_segmentation_) {
        if (images_[i]->image.at<cv::Vec3b>(yi, xi) == road_c) {
          *(color_begin_u8 + offset) = road_cost_;
        }
      } else {
        switch (field_type_)
        {
          case sensor_msgs::msg::PointField::UINT8:
          {
            *(color_begin_u8 + offset) =  images_[i]->image.at<uint8_t>(yi, xi);
            break;
          }
          case sensor_msgs::msg::PointField::UINT16:
          {
            *(color_begin_u16 + offset) =  images_[i]->image.at<uint16_t>(yi, xi);
            break;
          }
          case sensor_msgs::msg::PointField::FLOAT32:
          {
            RCLCPP_WARN(this->get_logger(), "barva %d %d %d", images_[i]->image.at<cv::Vec3b>(yi, xi)[0],
                  images_[i]->image.at<cv::Vec3b>(yi, xi)[1],
                  images_[i]->image.at<cv::Vec3b>(yi, xi)[2]);
            *(color_begin_f + offset) = rgb_to_float(images_[i]->image.at<cv::Vec3b>(yi, xi));
            break;
          }
        }
      }
    }
  }
  cloud_pub_.publish(cloud_out);

  if (print_delay_) {
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    RCLCPP_INFO(this->get_logger(), "Callback execution time: %ld µs", duration.count());
  }
}

} /* namespace point_cloud_color */

RCLCPP_COMPONENTS_REGISTER_NODE(point_cloud_color::PointCloudColor)

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<point_cloud_color::PointCloudColor>(rclcpp::NodeOptions());
  node->onInit();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}