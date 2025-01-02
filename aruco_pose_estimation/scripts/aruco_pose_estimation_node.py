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
    image_topic - image topic to subscribe to (default /camera/color/image_raw)
    camera_info_topic - camera info topic to subscribe to (default /camera/camera_info)
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

# Python imports
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

# Local imports for custom defined functions
from aruco_pose_estimation.utils import ARUCO_DICT
from aruco_pose_estimation.pose_estimation import apply_transform_to_pose, pose_to_matrix, pose_estimation

# ROS2 message imports
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray, Pose
from aruco_interfaces.msg import ArucoMarkers
from rcl_interfaces.msg import ParameterDescriptor, ParameterType


class ArucoNode(rclpy.node.Node):
    def __init__(self):
        super().__init__("aruco_node")
        self.calibration_coeff = None
        self.distortion_coeff = None
        self.origin_transform = None

        # Initialize ROS2 parameters
        self.initialize_parameters()

        # Load the camera calibration parameters
        self.load_calibration_parameters()

        # Set the origin transformation from the parameters
        self.set_origin_from_parameters()

        # Make sure we have a valid dictionary id:
        try:
            dictionary_id = cv2.aruco.__getattribute__(self.dictionary_id_name)
            # check if the dictionary_id is a valid dictionary inside ARUCO_DICT values
            if dictionary_id not in ARUCO_DICT.values():
                raise AttributeError
        except AttributeError:
            self.get_logger().error("bad aruco_dictionary_id: {}".format(self.dictionary_id_name))
            options = "\n".join([s for s in ARUCO_DICT])
            self.get_logger().error("valid options: {}".format(options))

        # Set up subscriptions to the camera info and camera image topics
        self.image_sub = self.create_subscription(Image, self.image_topic, self.image_cb, qos_profile_sensor_data)

        # Set up publishers
        self.poses_pub = self.create_publisher(PoseArray, self.markers_visualization_topic, 10)
        self.markers_pub = self.create_publisher(ArucoMarkers, self.detected_markers_topic, 10)
        self.image_pub = self.create_publisher(Image, self.output_image_topic, 10)

        # Set up fields for camera parameters
        self.info_msg = None

        # code for updated version of cv2 (4.7.0)
        if cv2.__version__ >= "4.7.0":
            self.aruco_dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
            self.aruco_parameters = cv2.aruco.DetectorParameters()
            self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dictionary, self.aruco_parameters)
        else:
            # old code version (4.6.0.66 and older)
            self.aruco_dictionary = cv2.aruco.Dictionary_get(dictionary_id)
            self.aruco_parameters = cv2.aruco.DetectorParameters_create()
            self.aruco_detector = None

        self.bridge = CvBridge()

    def initialize_parameters(self):
        # Declare and read parameters from aruco_params.yaml
        self.declare_parameter(
            name="marker_size",
            value=0.0625,
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description="Size of the markers in meters.",
            ),
        )

        self.declare_parameter(
            name="aruco_dictionary_id",
            value="DICT_5X5_100",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Dictionary that was used to generate markers.",
            ),
        )

        self.declare_parameter(
            name="image_topic",
            value="/camera/color/image_raw",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Image topic to subscribe to.",
            ),
        )

        self.declare_parameter(
            name="camera_info_topic",
            value="/camera/camera_info",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Camera info topic to subscribe to.",
            ),
        )

        self.declare_parameter(
            name="camera_frame",
            value="",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Camera optical frame to use.",
            ),
        )

        self.declare_parameter(
            name="detected_markers_topic",
            value="/aruco_markers",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Topic to publish detected markers as array of marker ids and poses",
            ),
        )

        self.declare_parameter(
            name="markers_visualization_topic",
            value="/aruco_poses",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Topic to publish markers as pose array",
            ),
        )

        self.declare_parameter(
            name="output_image_topic",
            value="/aruco_image",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Topic to publish annotated images with markers drawn on them",
            ),
        )

        self.declare_parameter(
            name="calibration_coefficients_file",
            value="",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description=".npy file containing the camera calibration matrix",
            ),
        )

        self.declare_parameter(
            name="distortion_coefficients_file",
            value="",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description=".npy file containing the camera distortion coefficients",
            ),
        )

        self.declare_parameter(
            name="origin_position",
            value=[0.0, 0.0, 0.0],  # Default position
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE_ARRAY,
                description="Origin position [x, y, z] in meters."
            )
        )

        self.declare_parameter(
            name="origin_orientation",
            value=[0.0, 0.0, 0.0, 1.0],  # Default quaternion [x, y, z, w]
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE_ARRAY,
                description="Origin orientation [x, y, z, w] quaternion."
            )
        )

        # read parameters from aruco_params.yaml and store them
        self.marker_size = self.get_parameter("marker_size").get_parameter_value().double_value
        self.dictionary_id_name = self.get_parameter("aruco_dictionary_id").get_parameter_value().string_value
        self.image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
        self.camera_frame = self.get_parameter("camera_frame").get_parameter_value().string_value
        self.detected_markers_topic = self.get_parameter("detected_markers_topic").get_parameter_value().string_value
        self.markers_visualization_topic = self.get_parameter("markers_visualization_topic").get_parameter_value().string_value
        self.output_image_topic = self.get_parameter("output_image_topic").get_parameter_value().string_value
        self.origin_position = self.get_parameter("origin_position").get_parameter_value().double_array_value
        self.origin_orientation = self.get_parameter("origin_orientation").get_parameter_value().double_array_value

        self.get_logger().info(f"Marker size: {self.marker_size}")
        self.get_logger().info(f"Input image topic: {self.image_topic}")
        self.get_logger().info(f"Camera frame: {self.camera_frame}")
        self.get_logger().info(f"Detected markers topic: {self.detected_markers_topic}")
        self.get_logger().info(f"Markers visualization topic: {self.markers_visualization_topic}")
        self.get_logger().info(f"Output image topic: {self.output_image_topic}")
        self.get_logger().info(f"Origin position: {self.origin_position}")
        self.get_logger().info(f"Origin orientation: {self.origin_orientation}")

    def load_calibration_parameters(self):
        calibration_file = self.get_parameter("calibration_coefficients_file").get_parameter_value().string_value
        distortion_file = self.get_parameter("distortion_coefficients_file").get_parameter_value().string_value
        print("calibration_file: ", calibration_file)
        print("distortion_file: ", distortion_file)

        # load the calibration matrix and distortion coefficients from the files
        try:
            self.calibration_coeff = np.load(calibration_file)
            self.distortion_coeff = np.load(distortion_file)
            self.get_logger().info("Camera calibration parameters loaded.")
            self.get_logger().info("Calibration coefficients: \n{}".format(self.calibration_coeff))
            self.get_logger().info("Distortion coefficients: {}".format(self.distortion_coeff))
        except Exception as e:
            self.get_logger().error("Error loading calibration parameters: {}".format(e))

    def image_cb(self, img_msg: Image):

        # convert the image messages to cv2 format
        cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="rgb8")

        # create the ArucoMarkers and PoseArray messages
        markers = ArucoMarkers()
        pose_array = PoseArray()

        # Set the frame id and timestamp for the markers and pose array
        if self.camera_frame == "":
            markers.header.frame_id = self.info_msg.header.frame_id
            pose_array.header.frame_id = self.info_msg.header.frame_id
        else:
            #markers.header.frame_id = "world"
            #pose_array.header.frame_id = "/world"
            markers.header.frame_id = self.camera_frame
            pose_array.header.frame_id = self.camera_frame

        # Check if the stamp field sec or nanosec are both zero then add the current time
        if img_msg.header.stamp.sec == 0 and img_msg.header.stamp.nanosec == 0:
            img_msg.header.stamp = self.get_clock().now().to_msg()
        else:
            img_msg.header.stamp = img_msg.header.stamp

        markers.header.stamp = img_msg.header.stamp
        pose_array.header.stamp = img_msg.header.stamp

        # call the pose estimation function
        frame, pose_array, markers = pose_estimation(frame=cv_image,
                                                     aruco_dict_type=self.aruco_dictionary,
                                                     aruco_params=self.aruco_parameters,
                                                     aruco_detector=self.aruco_detector,
                                                     marker_size=self.marker_size,
                                                     calibration_coeff=self.calibration_coeff,
                                                     distortion_coeff=self.distortion_coeff,
                                                     pose_array=pose_array,
                                                     markers=markers)

        if len(markers.marker_ids) > 0:
            # Transform poses relative to the origin
            transformed_pose_array = self.transform_all_poses(pose_array)

            # Publish the results with the poses and markes positions
            self.poses_pub.publish(transformed_pose_array)
            self.markers_pub.publish(markers)

        # publish the image frame with computed markers positions over the image
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(frame, "rgb8"))

    def set_origin_from_parameters(self):
        """ Sets th origin transformation using the position and orientation form the parameters. """
        self.origin_transform = pose_to_matrix(self.origin_position, self.origin_orientation)
        self.get_logger().info("Origin set to: {}".format(self.origin_transform))

    def transform_all_poses(self, pose_array):
        """
        Transform all detected marker poses to the origin coordinate frame.
        :param pose_array: PoseArray to be transformed.
        :return: Transformed PoseArray.
        """
        if self.origin_transform is None:
            self.get_logger().warn("Origin is not set. Cannot transform poses.")
            return pose_array  # Return the original array if no origin is set

        transformed_pose_array = PoseArray()
        transformed_pose_array.header = pose_array.header

        for pose in pose_array.poses:
            transformed_pose = apply_transform_to_pose(pose, self.origin_transform)
            transformed_pose_array.poses.append(transformed_pose)

        return transformed_pose_array


def main():
    rclpy.init()
    node = ArucoNode()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
