^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package point_cloud_color
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1.2.1 (2023-06-19)
------------------
* Added point_cloud_transport integration.
* Support MONO8 images.
* Noetic compatibility.
* Fix passing image_transport parameters
* Fixed coloring of clouds with NaNs and fix publishing color as rgba
* Compute remaining duration of waiting for transform.
* Update row step to pass asserts in debug, clip values according to types.
* Refactored for various image types, custom field name, minor speed up.
* Shortening cloud queue size option (backward compatible).
* Set default value before coloring.
* Camera warnings refactored, warning type enum.
* Refactored to handle various image types.
* OpenCV constant updated for compatibility with 20.04/Noetic.
* Initial commit.
* Contributors: Martin Pecka, Tomas Petricek
