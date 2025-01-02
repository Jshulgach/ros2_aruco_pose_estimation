#!/usr/bin/env python3

# Code taken and readapted from:
# https://github.com/GSNCodes/ArUCo-Markers-Pose-Estimation-Generation-Python/tree/main

# Python imports
import numpy as np
import cv2
import tf_transformations

# ROS2 message imports
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseArray
from aruco_interfaces.msg import ArucoMarkers

# utils import python code
from aruco_pose_estimation.utils import aruco_display


def pose_estimation(frame, aruco_dict_type, aruco_params, aruco_detector, marker_size, calibration_coeff, distortion_coeff, pose_array, markers):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if aruco_detector is None:
        (corners, marker_ids, rejected) = cv2.aruco.detectMarkers(gray, aruco_dict_type, parameters=aruco_params)
    else:
        # new code version
        corners, marker_ids, rejected = aruco_detector.detectMarkers(image=gray)

    frame_processed = frame
    if marker_ids is not None:
        for i, marker_id in enumerate(marker_ids):
            try:
                # Fix corners if the shape is wrong
                if corners[i].shape != (4, 1, 2):
                    corners[i] = corners[i].reshape((1, 4, 2))
                corners[i] = np.array(corners[i], dtype=np.float32)

                # Ensure inputs are numpy arrays with the correct type
                calibration_coeff = np.array(calibration_coeff, dtype=np.float32)
                distortion_coeff = np.array(distortion_coeff, dtype=np.float32)

                # Estimate pose of each marker and return the values rvec and tvec
                rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], marker_size, calibration_coeff, distortion_coeff)
                rvec = rvec.reshape(3, 1)
                tvec = tvec.reshape(3, 1)

                rot, jacobian = cv2.Rodrigues(rvec)
                rot_matrix = np.eye(4, dtype=np.float32)
                rot_matrix[0:3, 0:3] = rot

                # convert rotation matrix to quaternion
                quaternion = tf_transformations.quaternion_from_matrix(rot_matrix)
                norm_quat = np.linalg.norm(quaternion)
                quat = quaternion / norm_quat

                # alternative code version using solvePnP
                #tvec, rvec, quat = my_estimatePoseSingleMarkers(corners=corners[i], marker_size=marker_size,
                #                                                        camera_matrix=matrix_coefficients,
                #                                                        distortion=distortion_coefficients)

                # show the detected markers bounding boxes
                frame_processed = aruco_display(corners=corners, ids=marker_ids,
                                                image=frame_processed)

                # draw frame axes
                frame_processed = cv2.drawFrameAxes(image=frame_processed, cameraMatrix=calibration_coeff,
                                                    distCoeffs=distortion_coeff, rvec=rvec, tvec=tvec,
                                                    length=0.05, thickness=3)

                pose = Pose()
                pose.position.x = float(tvec[0])
                pose.position.y = float(tvec[1])
                pose.position.z = float(tvec[2])
                pose.orientation.x = quat[0]
                pose.orientation.y = quat[1]
                pose.orientation.z = quat[2]
                pose.orientation.w = quat[3]

                # add the pose and marker id to the pose_array and markers messages
                pose_array.poses.append(pose)
                markers.poses.append(pose)
                markers.marker_ids.append(marker_id[0])

            except Exception as e:
                print(f"Error: {e}")
                continue

    return frame_processed, pose_array, markers

def my_estimatePoseSingleMarkers(corners, marker_size, camera_matrix, distortion):# -> tuple[np.array, np.array, np.array]:
    '''
    This will estimate the rvec and tvec for each of the marker corners detected by:
       corners, ids, rejectedImgPoints = detector.detectMarkers(image)

    corners - is an array of detected corners for each detected marker in the image
    marker_size - is the size of the detected markers in meters
    mtx - is the camera intrinsic matrix
    distortion - is the camera distortion matrix
    RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
    '''
    print("Camera Matrix:", camera_matrix)
    print("Type:", type(camera_matrix))
    print("Shape:", camera_matrix.shape if hasattr(camera_matrix, 'shape') else 'Not a numpy array')

    marker_points = np.array([[-marker_size / 2.0, marker_size / 2.0, 0],
                              [marker_size / 2.0, marker_size / 2.0, 0],
                              [marker_size / 2.0, -marker_size / 2.0, 0],
                              [-marker_size / 2.0, -marker_size / 2.0, 0]], dtype=np.float32)

    # solvePnP returns the rotation and translation vectors
    retval, rvec, tvec = cv2.solvePnP(objectPoints=marker_points, imagePoints=corners,
                                        cameraMatrix=camera_matrix, distCoeffs=distortion, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    rvec = rvec.reshape(3, 1)
    tvec = tvec.reshape(3, 1)
       
    rot, jacobian = cv2.Rodrigues(rvec)
    rot_matrix = np.eye(4, dtype=np.float32)
    rot_matrix[0:3, 0:3] = rot

    # convert rotation matrix to quaternion
    quaternion = tf_transformations.quaternion_from_matrix(rot_matrix)
    norm_quat = np.linalg.norm(quaternion)
    quaternion = quaternion / norm_quat

    return tvec, rvec, quaternion


def depth_to_pointcloud_centroid(depth_image: np.array, intrinsic_matrix: np.array,
                                 corners: np.array) -> np.array:
    """
    This function takes a depth image and the corners of a quadrilateral as input,
    and returns the centroid of the corresponding pointcloud.

    Args:
        depth_image: A 2D numpy array representing the depth image.
        corners: A list of 4 tuples, each representing the (x, y) coordinates of a corner.

    Returns:
        A tuple (x, y, z) representing the centroid of the segmented pointcloud.
    """

    # Get image parameters
    height, width = depth_image.shape
    

    # Check if all corners are within image bounds
    # corners has shape (1, 4, 2)
    corners_indices = np.array([(int(x), int(y)) for x, y in corners[0]])

    for x, y in corners_indices:
        if x < 0 or x >= width or y < 0 or y >= height:
            raise ValueError("One or more corners are outside the image bounds.")

    # bounding box of the polygon
    x_min = int(min(corners_indices[:, 0]))
    x_max = int(max(corners_indices[:, 0]))
    y_min = int(min(corners_indices[:, 1]))
    y_max = int(max(corners_indices[:, 1]))

    # create array of pixels inside the polygon defined by the corners
    # search for pixels inside the squared bounding box of the polygon
    points = []
    for x in range(x_min, x_max):
        for y in range(y_min, y_max):
            if is_pixel_in_polygon(pixel=(x, y), corners=corners_indices):
                # add point to the list of points
                points.append([x, y, depth_image[y, x]])

    # Convert points to numpy array
    points = np.array(points, dtype=np.uint16)
   
    # convert to open3d image
    #depth_segmented = geometry.Image(points)
    # create pinhole camera model
    #pinhole_matrix = camera.PinholeCameraIntrinsic(width=width, height=height, 
    #                                               intrinsic_matrix=intrinsic_matrix)
    # Convert points to Open3D pointcloud
    #pointcloud = geometry.PointCloud.create_from_depth_image(depth=depth_segmented, intrinsic=pinhole_matrix,
    #                                                         depth_scale=1000.0)

    # apply formulas to pointcloud, where 
    # fx = intrinsic_matrix[0, 0], fy = intrinsic_matrix[1, 1]
    # cx = intrinsic_matrix[0, 2], cy = intrinsic_matrix[1, 2], 
    # u = x, v = y, d = depth_image[y, x], depth_scale = 1000.0,
    # z = d / depth_scale
    # x = (u - cx) * z / fx
    # y = (v - cy) * z / fy

    # create pointcloud
    pointcloud = []
    for x, y, d in points:
        z = d / 1000.0
        x = (x - intrinsic_matrix[0, 2]) * z / intrinsic_matrix[0, 0]
        y = (y - intrinsic_matrix[1, 2]) * z / intrinsic_matrix[1, 1]
        pointcloud.append([x, y, z])

    # Calculate centroid from pointcloud
    centroid = np.mean(np.array(pointcloud, dtype=np.uint16), axis=0)

    return centroid

def apply_transform_to_pose(pose: Pose, transform: np.array) -> Pose:
    """
    This function takes a Pose message and a transformation matrix as input, and returns a new Pose message
    that is the result of applying the transformation to the original pose.

    Args:
        pose: A Pose message representing the original pose.
        transform: A 4x4 numpy array representing the transformation matrix.

    Returns:
        A Pose message representing the transformed pose.
    """

    # Convert the position and orientation of the pose to numpy arrays
    position = np.array([pose.position.x, pose.position.y, pose.position.z], dtype=np.float32)
    orientation = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w], dtype=np.float32)

    # Apply the transformation to the position
    position_homogeneous = np.append(position, 1.0)  # Convert to homogeneous coordinates
    position_transformed = np.dot(transform, position_homogeneous)[:3]  # Transform and extract [x, y, z]

    # Apply the transformation to the orientation
    rot_matrix = tf_transformations.quaternion_matrix(orientation)[:3, :3]  # Extract the 3x3 rotation matrix
    rot_matrix_transformed = np.dot(transform[:3, :3], rot_matrix)  # Transform the rotation
    orientation_transformed = tf_transformations.quaternion_from_matrix(
        np.vstack([np.hstack([rot_matrix_transformed, [[0], [0], [0]]]), [0, 0, 0, 1]])
    )

    # Create a new Pose message with the transformed position and orientation
    pose_transformed = Pose()
    pose_transformed.position.x = position_transformed[0]
    pose_transformed.position.y = position_transformed[1]
    pose_transformed.position.z = position_transformed[2]
    pose_transformed.orientation.x = orientation_transformed[0]
    pose_transformed.orientation.y = orientation_transformed[1]
    pose_transformed.orientation.z = orientation_transformed[2]
    pose_transformed.orientation.w = orientation_transformed[3]

    return pose_transformed


def pose_to_matrix(position: np.array, orientation: np.array) -> np.array:
    """
    This function takes a position and orientation as input, and returns a 4x4 transformation matrix.

    Args:
        position: A numpy array representing the position [x, y, z].
        orientation: A numpy array representing the orientation [x, y, z, w].

    Returns:
        A 4x4 numpy array representing the transformation matrix.
    """

    # Create a 4x4 transformation matrix
    matrix = np.eye(4, dtype=np.float32)

    # Set the position in the transformation matrix
    matrix[0:3, 3] = position

    # Set the orientation in the transformation matrix
    rot_matrix = tf_transformations.quaternion_matrix(orientation)
    matrix[0:3, 0:3] = rot_matrix[0:3, 0:3]

    return matrix

def is_pixel_in_polygon(pixel: tuple, corners: np.array) -> bool:
    """
    This function takes a pixel and a list of corners as input, and returns whether the pixel is inside the polygon
    defined by the corners. This function uses the ray casting algorithm to determine if the pixel is inside the polygon.
    This algorithm works by casting a ray from the pixel in the positive x-direction, and counting the number of times
    the ray intersects with the edges of the polygon. If the number of intersections is odd, the pixel is inside the
    polygon, otherwise it is outside. This algorithm works for both convex and concave polygons.

    Args:
        pixel: A tuple (x, y) representing the pixel coordinates.
        corners: A list of 4 tuples in a numpy array, each representing the (x, y) coordinates of a corner.

    Returns:
        A boolean indicating whether the pixel is inside the polygon.
    """

    # Initialize counter for number of intersections
    num_intersections = 0

    # Iterate over each edge of the polygon
    for i in range(len(corners)):
        x1, y1 = corners[i]
        x2, y2 = corners[(i + 1) % len(corners)]

        # Check if the pixel is on the same y-level as the edge
        if (y1 <= pixel[1] < y2) or (y2 <= pixel[1] < y1):
            # Calculate the x-coordinate of the intersection point
            x_intersection = (x2 - x1) * (pixel[1] - y1) / (y2 - y1) + x1

            # Check if the intersection point is to the right of the pixel
            if x_intersection > pixel[0]:
                num_intersections += 1

    # Return whether the number of intersections is odd
    return num_intersections % 2 == 1
