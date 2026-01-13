# ROS2 imports
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration, TextSubstitution
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch_ros.substitutions import FindPackageShare
from launch.conditions import IfCondition, UnlessCondition


def generate_launch_description():

    # Declare launch arguments
    declared_arguments = []
    declared_arguments.append(DeclareLaunchArgument(
        name="launch_rviz", default_value="false", description="Launch RViz?", choices=['true', 'false', 'True', 'False']
    ))
    launch_rviz = LaunchConfiguration('launch_rviz')

    # Aruco node with parameters yaml file
    aruco_params = PathJoinSubstitution([
        FindPackageShare('aruco_pose_estimation'), 'config', 'aruco_parameters.yaml',
    ])
    aruco_pose_estimator_node = Node(
        package='aruco_pose_estimation',
        executable='aruco_pose_estimation_node.py',
        parameters=[aruco_params],
        output='screen',
        emulate_tty=True
    )

    # Static TF publisher
    static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="static_transform_publisher",
        output="log",
        # These pose values are estimated by visual inspection of the webcam position representation of an
        # array of aruco markers placed on the surface around a robot arm.
        #arguments=["0.57", "0.03", "0.86", "1.571", "0.0", "3.69", "world", "camera_color_optical_frame"],
        #arguments=["0.55", "0.0", "0.87", "1.571", "-0.03", "3.69", "world", "camera_color_optical_frame"],
        arguments=["0.55", "0.0", "0.87", "0.0", "-0.7", "-0.87", "world", "camera_color_optical_frame"],
    )

    # Aruco state publisher
    aruco_tf_broadcaster_node = Node(
        package='aruco_pose_estimation',
        executable='aruco_broadcaster_node.py',
        output='log',
    )

    # RViz2 node
    rviz_file = PathJoinSubstitution([FindPackageShare('aruco_pose_estimation'), 'rviz', 'cam_detect.rviz'])
    rviz2_node = Node(
        package='rviz2',
        condition=IfCondition(launch_rviz),
        executable='rviz2',
        arguments=['-d', rviz_file]
    )

    launch_nodes = [
        aruco_pose_estimator_node,
        static_tf,
        aruco_tf_broadcaster_node,
        rviz2_node
    ]

    return LaunchDescription(declared_arguments + launch_nodes)