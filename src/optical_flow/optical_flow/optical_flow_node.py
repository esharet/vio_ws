#!/usr/bin/env python3
from dataclasses import dataclass
import math
from scipy.spatial.transform import Rotation as R
import cv2
import traceback
import numpy as np

import rclpy
from rclpy.qos import qos_profile_sensor_data
from rclpy.node import Node

from cv_bridge import CvBridge
from tf2_ros import TransformListener
from tf2_ros.buffer import Buffer
from tf2_ros import TransformException

from sensor_msgs.msg import Image, CameraInfo
from mavros_msgs.msg import Altitude
from builtin_interfaces.msg import Time
from geometry_msgs.msg import TwistStamped

from projection import project_points_to_ground
from lk import LK, LKResult, VelocityKalmanFilter2D

import matplotlib.pyplot as plt


CAMERA_IMAGE_TOPIC = "/gimbal_camera/image_raw"
# CAMERA_IMAGE_TOPIC = "/camera/image_ae"
CAMERA_INFO_TOPIC = "/gimbal_camera/camera_info"
ALTITUDE_TOPIC = "/rome/altitude"

WORLD_FRAME = "map"
CAMERA_OPTICAL_FRAME = "camera_optical"

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 512

MAX_TIME_DIFF = 0.5
MIN_TIME_DIFF = 0.05

VELOCITY_SIZE_LIMIT = 30

VELOCITY_DEADZONE = 0.05

@dataclass
class CameraInfoData:
    K: np.ndarray
    distortion_coef: np.ndarray


class VIONode(Node):
    def __init__(self):
        node_name="optical_flow_node"
        super().__init__(node_name)

        self.camera_info_data: CameraInfoData = None
        self.camera_height: float = None
        self.last_image_timestamp: Time = None

        self.bridge = CvBridge()

        self._vio = LK(IMAGE_WIDTH, IMAGE_HEIGHT, self.get_logger())

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.kf = VelocityKalmanFilter2D()

        self._init_subscribers()

        self._init_publishers()

    def _init_publishers(self):
        self.twist_pub = self.create_publisher(TwistStamped, "of_twist", 10)
        self.twist_filtered_pub = self.create_publisher(TwistStamped, "of_twist_filtered", 10)


    def _init_subscribers(self):
        self.camera_image_sub = self.create_subscription(Image,
            CAMERA_IMAGE_TOPIC,
            self.image_callback,
            qos_profile=qos_profile_sensor_data)
        self.camera_info_sub = self.create_subscription(CameraInfo,
            CAMERA_INFO_TOPIC,
            self.camera_info_callback,
            qos_profile=qos_profile_sensor_data)
        self.altitude_sub = self.create_subscription(Altitude,
            ALTITUDE_TOPIC,
            self.altitude_callback,
            qos_profile=qos_profile_sensor_data)
    
    def show_features(self, frame, lk_result: LKResult):
        COLOR_NEW = (0, 0, 255) # RED
        COLOR_OLD = (255, 0, 0) # BLUE
        for new, old in zip(lk_result.good_new, lk_result.good_old):
            a, b = new.ravel().astype(int)
            cv2.circle(frame, (a, b), 3, COLOR_NEW, -1)
            c, d = old.ravel().astype(int)
            cv2.circle(frame, (c, d), 3, COLOR_OLD, -1)
        
        cv2.imshow("LK Optical Flow CUDA", frame)
        cv2.waitKey(1)
    
    def is_all_inputs_valid(self) -> bool: 
        if self.camera_info_data is None:
            self.get_logger().warning(f"camera info data is None")
            return False
        if self.camera_height is None:
            self.get_logger().warning(f"camera height is None. No {ALTITUDE_TOPIC} terrain value ?")
            return False
        return True

    def get_rotation_matrix(self):
        current_transform = self.tf_buffer.lookup_transform(WORLD_FRAME, CAMERA_OPTICAL_FRAME, rclpy.time.Time()) # TODO: verify transform exists
        quat = [
            current_transform.transform.rotation.x,
            current_transform.transform.rotation.y,
            current_transform.transform.rotation.z,
            current_transform.transform.rotation.w
        ]
        return R.from_quat(quat).as_matrix()

    def check_time_stamps(self, current_image_timestamp: Time):
        if self.last_image_timestamp is None:
            self.get_logger().warning(f"last image timestamp is None")
            self.last_image_timestamp = current_image_timestamp
            return False, 0
        
        time_diff = self.ros_stamp_to_sec(current_image_timestamp) - self.ros_stamp_to_sec(self.last_image_timestamp)
        self.last_image_timestamp = current_image_timestamp

        if time_diff > MAX_TIME_DIFF:
            self.get_logger().warning(f"low frame rate, time diff is bigger then max: {time_diff} > {MAX_TIME_DIFF}")
            return False, 0
        elif time_diff < MIN_TIME_DIFF:
            self.get_logger().error(f"too close frames, time diff is lower then min: {time_diff} < {MIN_TIME_DIFF}")
            return False, 0
        
        return True, time_diff

    def ros_stamp_to_sec(self, msg: Time) -> float:
        return float(msg.sec + msg.nanosec * 1e-9)
    
    def calc_estimated_velocity(self, lk_result, rotation_matrix, time_diff):
        old_points: np.ndarray = project_points_to_ground(
            features=lk_result.good_old,
            K=self.camera_info_data.K,
            D=self.camera_info_data.distortion_coef,
            R=rotation_matrix,
            camera_height=self.camera_height)
        
        new_points: np.ndarray = project_points_to_ground(
            features=lk_result.good_new,
            K=self.camera_info_data.K,
            D=self.camera_info_data.distortion_coef,
            R=rotation_matrix,
            camera_height=self.camera_height)
        
        points_flow_on_ground = new_points[:, :2] - old_points[:, :2]
        
        # 1. Calculate median, and filter by 1.8 STD around mean
        med = np.median(points_flow_on_ground, axis=0)
        std = np.std(points_flow_on_ground, axis=0)

        std_thresh = 1.8
        lower = med - std_thresh * std
        upper = med + std_thresh * std
        inlier_mask = np.all((points_flow_on_ground >= lower) & (points_flow_on_ground <= upper), axis=1)
        
        if np.sum(inlier_mask) == 0:
            mean_flow_on_ground = med  # fallback
        else:
            mean_flow_on_ground = np.mean(points_flow_on_ground[inlier_mask], axis=0)

        # # 2. Calculate histogram. and filter by 1.8 STD around peack
        # self.get_logger().info(f"points_flow_on_ground shape {points_flow_on_ground.shape}")
        # self.get_logger().info(f"{points_flow_on_ground=}")
        # filtered_velocities = []
        # std_thresh = 2.5
        # for i in range(2):
        #     comp = points_flow_on_ground[:, i]
        #     hist, bin_edges = np.histogram(comp, bins=15)
        #     max_bin_idx = np.argmax(hist)
        #     bin_start = bin_edges[max_bin_idx]
        #     bin_end   = bin_edges[max_bin_idx + 1]

        #     values_in_bin = comp[(comp >= bin_start) & (comp < bin_end)]

        #     mean_val = np.mean(values_in_bin)
        #     std_val  = np.std(values_in_bin)

        #     mask = (comp >= mean_val - std_thresh * std_val) & (comp <= mean_val + std_thresh * std_val)
        #     filtered_velocities.append(mask)
            
        # combined_mask = filtered_velocities[0] & filtered_velocities[1]
        # mean_flow_on_ground = points_flow_on_ground[combined_mask]
        # self.get_logger().info(f"mean_flow_on_ground shape {mean_flow_on_ground.shape}")

        estimated_velocity = -mean_flow_on_ground / time_diff
        return estimated_velocity#[0], estimated_velocity[1]

    def image_callback(self, msg: Image):
        # Make sure there are camera info and updated camera height
        if not self.is_all_inputs_valid():
            return

        try:
            # Convert to OpenCV image and find features
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            lk_result: LKResult = self._vio.process_frame(frame)
            # Calculate the rotation matrix using TF2
            rotation_matrix = self.get_rotation_matrix()
        except TransformException as e:
            self.get_logger().error(f"Failed to lookup transform: {e}")
            lk_result = None
        except Exception as e:
            self.get_logger().error(f"Failed to process frame: {e}")
            lk_result = None
        finally:
            # exit if no lk result, and show them on cv2 window if exists
            if lk_result is None:
                return
            self.show_features(frame, lk_result)
        
        # Check time difference is in the valid range
        current_image_timestamp = msg.header.stamp
        time_valid, time_diff = self.check_time_stamps(current_image_timestamp)
        if not time_valid:
            self.get_logger().warning(f"time diff not valid")
            return
        
        # Use the LK optical flow, then find the ground points, filter them, and calculate average velocity
        est_vx, est_vy = self.calc_estimated_velocity(lk_result, rotation_matrix, time_diff)
        
        # filter large noise in velocity estimation
        velocity_size = np.linalg.norm([est_vx, est_vy])
        if velocity_size > VELOCITY_SIZE_LIMIT:
            self.get_logger().warning(f"velocity size is bigger then max: {velocity_size} > {VELOCITY_SIZE_LIMIT}")
            return
        elif velocity_size < VELOCITY_DEADZONE:
            est_vx = 0
            est_vy = 0

        # Send optical flow answer as TwistStamped message
        self.pub_est_vel(current_image_timestamp, est_vx, est_vy)
        self.pub_filtered_est_vel(current_image_timestamp, est_vx, est_vy)


    def pub_est_vel(self, current_image_timestamp, est_vx, est_vy):
        # TODO: calculate covariance and use TwistWithCovarianceStamped
        msg = TwistStamped()
        msg.header.stamp = current_image_timestamp
        msg.header.frame_id = WORLD_FRAME
        msg.twist.linear.x = float(est_vx)
        msg.twist.linear.y = float(est_vy)
        self.twist_pub.publish(msg)

    def pub_filtered_est_vel(self, current_image_timestamp, est_vx, est_vy):
        filtered_vx, filtered_vy = self.kf.update([est_vx, est_vy])
        msg = TwistStamped()
        msg.header.stamp = current_image_timestamp
        msg.header.frame_id = WORLD_FRAME
        msg.twist.linear.x = float(filtered_vx)
        msg.twist.linear.y = float(filtered_vy)
        self.twist_filtered_pub.publish(msg)


    def camera_info_callback(self, msg: CameraInfo):
        self.camera_info_data = CameraInfoData(
            K=np.array(msg.k).reshape((3,3)),
            distortion_coef=np.array(msg.d)
            )

    def altitude_callback(self, msg: Altitude):
        # TODO: make sure there is always updated height, create last_height_update variable
        # TODO: use ahrs2 if there is no rangefinder
        if not math.isnan(msg.terrain):
            self.camera_height = msg.terrain




def main(args=None):
    rclpy.init(args=args)
    node = VIONode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()