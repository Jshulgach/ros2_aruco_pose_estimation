#!/usr/bin/env python3
"""
ROS2 wrapper code taken from:
https://github.com/JMU-ROBOTICS-VIVA/ros2_aruco/tree/main

This node locates Aruco AR markers in images and publishes their ids and poses.

Subscriptions:
   /camera/color/image_raw (sensor_msgs.msg.Image)
   /camera/camera_info (sensor_msgs.msg.CameraInfo)

Published Topics:
    /aruco_poses (geometry_msgs.msg.PoseArray)
       Pose of all detected markers (suitable for rviz visualization)

    /aruco_markers (aruco_interfaces.msg.ArucoMarkers)
       Provides an array of all poses along with the corresponding
       marker ids.

    /aruco_image (sensor_msgs.msg.Image)
       Annotated image with marker locations and ids, with markers drawn on it

Parameters:
    marker_size - size of the markers in meters (default .065)
    aruco_dictionary_id - dictionary that was used to generate markers (default DICT_5X5_250)
    camera1_image_topic - image topic to subscribe to (default /camera/color/image_raw)
    camera2_image_topic - image topic to subscribe to (default /camera/color/image_raw)
    camera1_info_topic - camera info topic to subscribe to (default /camera/camera_info)
    camera2_info_topic - camera info topic to subscribe to (default /camera/camera_info)
    camera_frame - camera optical frame to use (default "camera_depth_optical_frame")
    detected_markers_topic - topic to publish detected markers (default /aruco_markers)
    markers_visualization_topic - topic to publish markers visualization (default /aruco_poses)
    output_image_topic - topic to publish annotated image (default /aruco_image)

Author: Simone Giampà. Version: 2024-01-29
Modified by: Jonathan Shulgach, Version: 2024-12-24

"""

# ROS2 imports
import rclpy
import rclpy.node
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
import message_filters
from packaging.version import Version

# Python imports
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

# Local imports for custom defined functions
from aruco_pose_estimation.utils import ARUCO_DICT
from aruco_pose_estimation.pose_estimation import (
    apply_transform_to_pose,
    pose_to_matrix,
    pose_estimation_single_camera,
    pose_estimation_dual_cameras
)

# ROS2 message imports
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseArray, Pose
from aruco_interfaces.msg import ArucoMarkers


class ArucoNode(rclpy.node.Node):
    def __init__(self):
        super().__init__("aruco_node")

        # Initialize ROS2 parameters
        self.initialize_parameters()

        # Initialize publishers and subscribers
        self.initialize_publishers()
        self.initialize_subscribers()

        # Load the camera calibration parameters
        #self.load_calibration_parameters()

        # Set the origin transformation from the parameters
        #self.set_origin_from_parameters()

        # Make sure we have a valid dictionary id:
        try:
            dictionary_id_name = self.get_param("aruco_dictionary_id")
            dictionary_id = cv2.aruco.__getattribute__(dictionary_id_name)
            # check if the dictionary_id is a valid dictionary inside ARUCO_DICT values
            if dictionary_id not in ARUCO_DICT.values():
                raise AttributeError
        except AttributeError:
            self.get_logger().error("bad aruco_dictionary_id: {}".format(dictionary_id_name))
            options = "\n".join([s for s in ARUCO_DICT])
            self.get_logger().error("valid options: {}".format(options))

        # code for updated version of cv2 (4.7.0)
        if Version(cv2.__version__) >= Version("4.7.0"):
            self.aruco_dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[self.get_param('aruco_dictionary_id')])
            self.aruco_parameters = cv2.aruco.DetectorParameters()
            self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dictionary, self.aruco_parameters)
        else:
            # old code version (4.6.0.66 and older)
            self.aruco_dictionary = cv2.aruco.Dictionary_get(ARUCO_DICT[self.get_param('aruco_dictionary_id')])
            self.aruco_parameters = cv2.aruco.DetectorParameters_create()
            self.aruco_detector = None

        # Set up fields for camera parameters
        self.info_msg = None
        self.intrinsic = []
        self.distortion = []
        self.projection = []

        self.bridge = CvBridge()

    def initialize_parameters(self):
        # Declare and read parameters from aruco params.yaml
        # Aruco detection parameters
        self.declare_parameter("marker_size", 0.0625)
        self.declare_parameter("aruco_dictionary_id", "DICT_5X5_100")

        # Input topics
        self.declare_parameter("camera_frame", "")
        self.declare_parameter("dual_cameras", False)
        self.declare_parameter("camera1_image_topic", "/camera1/color/image_raw")
        self.declare_parameter("camera1_info_topic", "/camera1/color/camera_info")  # Camera intrinsic parameters
        self.get_logger().info(f"Camera 1 input image topic: {self.get_param('camera1_image_topic')}")
        if self.get_param("dual_cameras"):
            self.declare_parameter("camera2_image_topic", "/camera2/color/image_raw")
            self.declare_parameter("camera2_info_topic", "/camera2/color/camera_info")  # Camera intrinsic parameters
            self.get_logger().info(f"Camera 2 input image topic: {self.get_param('camera2_image_topic')}")

        # Output topics
        self.declare_parameter("detected_markers_topic", "/aruco_markers")
        self.declare_parameter("markers_visualization_topic", "/aruco_poses")
        self.declare_parameter("output1_image_topic", "/aruco/camera1/image")
        if self.get_param("dual_cameras"):
            self.declare_parameter("output2_image_topic", "/aruco/camera2/image")

        self.declare_parameter("origin_position", [0.0, 0.0, 0.0])  # Default position
        self.declare_parameter("origin_orientation", [0.0, 0.0, 0.0, 1.0])  # Default quaternion [x, y, z, w]

        self.get_logger().info(f"Dual cameras: {self.get_param('dual_cameras')}")
        self.get_logger().info(f"Marker size: {self.get_param('marker_size')}")
        self.get_logger().info(f"Camera frame: {self.get_param('camera_frame')}")
        self.get_logger().info(f"Detected markers topic: {self.get_param('detected_markers_topic')}")
        self.get_logger().info(f"Markers visualization topic: {self.get_param('markers_visualization_topic')}")
        #self.get_logger().info(f"Output image topic: {self.get_param('output_image_topic')}")
        #self.get_logger().info(f"Origin position: {self.get_param('origin_position')}")
        #self.get_logger().info(f"Origin orientation: {self.get_param('origin_orientation')}")

    def initialize_publishers(self):
        # Set up publishers
        self.poses_pub = self.create_publisher(PoseArray, self.get_param('markers_visualization_topic'), 10)
        self.markers_pub = self.create_publisher(ArucoMarkers, self.get_param('detected_markers_topic'), 10)
        self.image1_pub = self.create_publisher(Image, self.get_param('output1_image_topic'), 10)
        if self.get_param("dual_cameras"):
            self.image2_pub = self.create_publisher(Image, self.get_param('output2_image_topic'), 10)

    def initialize_subscribers(self):
        # Set up subscriptions to the camera info and camera image topics
        self.info1_sub = self.create_subscription(
            CameraInfo, self.get_param('camera1_info_topic'), self.camera1_info_cb, qos_profile_sensor_data
        )

        if not self.get_param("dual_cameras"):
            self.image1_sub = self.create_subscription(
                Image, self.get_param('camera1_image_topic'), self.single_camera_image_cb, qos_profile_sensor_data
            )
        else:
            self.info2_sub = self.create_subscription(
                CameraInfo, self.get_param('camera2_info_topic'), self.camera2_info_cb, qos_profile_sensor_data
            )
            self.image1_sub = message_filters.Subscriber(self, Image, self.get_param('camera1_image_topic'),
                                                         qos_profile=qos_profile_sensor_data)
            self.image2_sub = message_filters.Subscriber(self, Image, self.get_param('camera2_image_topic'),
                                                         qos_profile=qos_profile_sensor_data)
            # Create synchronizer between image topics using message filters and approximate time
            self.synchronizer = message_filters.ApproximateTimeSynchronizer(
                [self.image1_sub, self.image2_sub], queue_size=10, slop=0.1  # slop is max time diff between messages
            )
            self.synchronizer.registerCallback(self.dual_camera_image_cb)

    def camera1_info_cb(self, msg: CameraInfo):
        self.info_msg = msg
        self.intrinsic.append(np.reshape(np.array(msg.k), (3, 3)))
        self.distortion.append(np.array(msg.d))
        self.projection.append(np.reshape(np.array(msg.p), (3, 4)))

        self.logger("Camera 1 intrinsic parameters received")
        self.logger("Intrinsic matrix: {}".format(self.intrinsic[0]))
        self.logger("Distortion coefficients: {}".format(self.distortion[0]))
        self.logger("Projection matrix: {}".format(self.projection[0]))
        self.info1_sub.destroy()

    def camera2_info_cb(self, msg: CameraInfo):
        self.info_msg = msg
        self.intrinsic.append(np.reshape(np.array(msg.k), (3, 3)))
        self.distortion.append(np.array(msg.d))
        self.projection.append(np.reshape(np.array(msg.p), (3, 4)))

        self.logger("Camera 2 intrinsic parameters received")
        self.logger("Intrinsic matrix: {}".format(self.intrinsic[1]))
        self.logger("Distortion coefficients: {}".format(self.distortion[1]))
        self.logger("Projection matrix: {}".format(self.projection[1]))
        self.info2_sub.destroy()

    def logger(self, *argv, msg_type='info'):
        """ Robust printing function to log events"""
        msg = ''.join(argv)
        if msg_type == 'info':
            self.get_logger().info(msg)
        elif msg_type == 'warn' or msg_type == 'warning':
            self.get_logger().warn(msg)
        elif msg_type == 'error' or msg_type == 'err':
            self.get_logger().error(msg)

    def get_param(self, name):
        # Helper function to get the ROS2 parameter by key name
        return self.get_parameter(name).value

    def single_camera_image_cb(self, img_msg: Image):

        # convert the image messages to cv2 format
        cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="rgb8")

        # create the ArucoMarkers and PoseArray messages
        markers = ArucoMarkers()
        pose_array = PoseArray()

        # Set the frame id and timestamp for the markers and pose array
        camera_frame = self.get_param('camera_frame')
        if camera_frame == "":
            markers.header.frame_id = self.info_msg.header.frame_id
            pose_array.header.frame_id = self.info_msg.header.frame_id
        else:
            markers.header.frame_id = camera_frame
            pose_array.header.frame_id = camera_frame

        # Check if the stamp field sec or nanosec are both zero then add the current time
        if img_msg.header.stamp.sec == 0 and img_msg.header.stamp.nanosec == 0:
            img_msg.header.stamp = self.get_clock().now().to_msg()
        else:
            img_msg.header.stamp = img_msg.header.stamp

        markers.header.stamp = img_msg.header.stamp
        pose_array.header.stamp = img_msg.header.stamp

        # call the pose estimation function
        frame, pose_array, markers = pose_estimation_single_camera(
            frame=cv_image,
            aruco_dict_type=self.aruco_dictionary,
            aruco_params=self.aruco_parameters,
            aruco_detector=self.aruco_detector,
            marker_size=self.get_param('marker_size'),
            calibration_coeff=self.intrinsic,
            distortion_coeff=self.distortion,
            pose_array=pose_array,
            markers=markers
        )

        if len(markers.marker_ids) > 0:
            # Publish the results with the poses and marker positions
            self.poses_pub.publish(pose_array)
            self.markers_pub.publish(markers)

        # publish the image frame with computed markers positions over the image
        self.image1_pub.publish(self.bridge.cv2_to_imgmsg(frame, "rgb8"))

    def dual_camera_image_cb(self, img_msg1: Image, img_msg2: Image):
        # convert the image messages to cv2 format
        cv_image1 = self.bridge.imgmsg_to_cv2(img_msg1, desired_encoding="rgb8")
        cv_image2 = self.bridge.imgmsg_to_cv2(img_msg2, desired_encoding="rgb8")

        # create the ArucoMarkers and PoseArray messages
        markers = ArucoMarkers()
        pose_array = PoseArray()

        # Set the frame id and timestamp for the markers and pose array
        camera_frame = self.get_param('camera_frame')
        if camera_frame == "":
            markers.header.frame_id = self.info_msg.header.frame_id
            pose_array.header.frame_id = self.info_msg.header.frame_id
        else:
            markers.header.frame_id = camera_frame
            pose_array.header.frame_id = camera_frame

        # Check if the stamp field sec or nanosec are both zero then add the current time
        if img_msg1.header.stamp.sec == 0 and img_msg1.header.stamp.nanosec == 0:
            img_msg1.header.stamp = self.get_clock().now().to_msg()
        else:
            img_msg1.header.stamp = img_msg1.header.stamp

        markers.header.stamp = img_msg1.header.stamp
        pose_array.header.stamp = img_msg1.header.stamp

        # call the pose estimation function
        if len(self.intrinsic) == 0 and len(self.distortion) == 0 and len(self.projection) == 0:
            return

        #self.get_logger().info(f"P: {self.projection}")
        frame1, frame2, pose_array, markers = pose_estimation_dual_cameras(
            frame0=cv_image1, frame1=cv_image2,
            aruco_dict_type=self.aruco_dictionary,
            aruco_params=self.aruco_parameters,
            aruco_detector=self.aruco_detector,
            marker_size=self.get_param('marker_size'),
            K=self.intrinsic, D=self.distortion, P=self.projection[0],  # identical projection
            pose_array=pose_array, markers=markers
        )

        if len(markers.marker_ids) > 0:
            # Publish the results with the poses and marker positions
            self.poses_pub.publish(pose_array)
            self.markers_pub.publish(markers)

        # publish the image frame with computed markers positions over the image
        self.image1_pub.publish(self.bridge.cv2_to_imgmsg(frame1, "rgb8"))
        self.image2_pub.publish(self.bridge.cv2_to_imgmsg(frame2, "rgb8"))

def main():
    rclpy.init()
    node = ArucoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        #rclpy.shutdown()

if __name__ == "__main__":
    main()
