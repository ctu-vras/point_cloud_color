# point_cloud_color
 
Package point_cloud_color provides a nodelet for coloring point clouds using calibrated cameras.

## Nodelets

## point_cloud_color/point_cloud_color

### Topics subscribed
- `cloud_in` (`sensor_msgs::PointCloud2`) Input point cloud.
- `camera_0/image` (`sensor_msgs::Image`) Subscribed cameras images.
- ..., `camera_<num_cameras - 1>/image`
- `camera_0/camera_info` (`sensor_msgs::CameraInfo`) Subscribed camera calibration messages.
- ..., `camera_<num_cameras - 1>/camera`

### Topics published
- `cloud_out` (`sensor_msgs::PointCloud2`) Colored point cloud, with field rgb.

### Parameters
- `fixed_frame` (`str`) Fixed frame to use when transforming point clouds to camera frame.
- `default_color` (`str`) Default color to be assigned to the point.
- `num_cameras` (`int`) Number of cameras to subscribe.
- `max_image_age` (`double`) Maximum image age to be used for coloring.
- `use_first_valid` (`bool`) Use first valid point projection, or best, otherwise.
- `image_queue_size` (`int`) Image queue size.
- `point_cloud_queue_size` (`int`) Point cloud queue size.
- `wait_for_transform` (`double`) Duration for waiting for the transform to become available.
- `camera_0/mask` (`str`) Static camera mask, zero elements denote pixels not to use in coloring.
- ..., `camera_<num_cameras - 1>/mask`
