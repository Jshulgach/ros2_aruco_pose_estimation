#!/usr/bin/env python3
"""

"""
import rclpy
import rclpy.node
from geometry_msgs.msg import PoseArray
from aruco_interfaces.msg import ArucoMarkers

from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped



class ArucoTransformPublisher(rclpy.node.Node):
    def __init__(self):
        super().__init__('aruco_transform_publisher')

        # Initialize ROS2 parameters
        self.init_parameters()

        # Create a subscriber to the marker topic
        self.subscription = self.create_subscription(ArucoMarkers, self.pose_topic, self.pose_callback, 1)

        # Make the transform broadcaster
        self.broadcaster = TransformBroadcaster(self)
        #self.marker_id = 0  # Use this to specify which marker's pose to broadcast

    def init_parameters(self):
        # Declare ROS2 parameters
        self.declare_parameter('pose_topic', '/aruco/markers')
        self.declare_parameter('parent_frame', '/camera_color_optical_frame')

        self.pose_topic = self.get_parameter('pose_topic').get_parameter_value().string_value
        self.parent_frame = self.get_parameter('parent_frame').get_parameter_value().string_value

    def pose_callback(self, msg):
        # Check if there are any poses in the PoseArray
        #self.get_logger().info("Pose message received")

        # The ArucoMarkers() message type contains:
        #
        # std_msgs/Header header
        # int64[] marker_ids
        # geometry_msgs/Pose[] poses

        for i, pose in enumerate(msg.poses):
            #self.get_logger().info(f"Pose {i}: {pose}")
            #self.get_logger().info(f"marker id: {msg.marker_ids[i]}")

            # Create a TransformStamped message
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = self.parent_frame  # The camera's frame
            t.child_frame_id = f'aruco_marker_{msg.marker_ids[i]}'  # The marker's frame

            # Set the translation
            t.transform.translation.x = pose.position.x
            t.transform.translation.y = pose.position.y
            t.transform.translation.z = pose.position.z

            # Set the rotation
            t.transform.rotation.x = pose.orientation.x
            t.transform.rotation.y = pose.orientation.y
            t.transform.rotation.z = pose.orientation.z
            t.transform.rotation.w = pose.orientation.w
            #t.transform.rotation = pose.orientation


            # Broadcast the transform
            self.broadcaster.sendTransform(t)
            #self.get_logger().info(f'Broadcasted transform for marker {msg.marker_ids[i]}')


def main():
    rclpy.init()
    node = ArucoTransformPublisher()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

