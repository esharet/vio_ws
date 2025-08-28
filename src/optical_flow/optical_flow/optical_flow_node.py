#!/usr/bin/env python3

from dataclasses import dataclass
import math
import rclpy
from rclpy.qos import qos_profile_sensor_data
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from mavros_msgs.msg import Altitude
from scipy.spatial.transform import Rotation as R
from projection import undistort_pixels, project_points_to_ground

import cv2
import traceback
import numpy as np
from lk import LK, LKResult
from tf2_ros import TransformListener
from tf2_ros.buffer import Buffer

CAMERA_IMAGE_TOPIC = "/gimbal_camera/image_raw"
CAMERA_INFO_TOPIC = "/gimbal_camera/camera_info"
ALTITUDE_TOPIC = "/rome/altitude"

WORLD_FRAME = "map"
CAMERA_OPTICAL_FRAME = "camera_optical"

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 512

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

        self.bridge = CvBridge()

        self._vio = LK(IMAGE_WIDTH, IMAGE_HEIGHT, self.get_logger())

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self._init_subscribers()

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
    
    def calc_position_diff(self):
        pass # use the real camera height, K matrix and R matrix (transform)
        

    def get_rotation_matrix(self):
        current_transform = self.tf_buffer.lookup_transform(WORLD_FRAME, CAMERA_OPTICAL_FRAME, rclpy.time.Time()) # TODO: verify transform exists
        quat = [
            current_transform.transform.rotation.x,
            current_transform.transform.rotation.y,
            current_transform.transform.rotation.z,
            current_transform.transform.rotation.w
        ]
        return R.from_quat(quat).as_matrix()
        
        

    def image_callback(self, msg: Image):
        if self.camera_height is None:
            self.get_logger().warning(f"camera height is none. No {ALTITUDE_TOPIC} terrain value")
            return
        
        try:
            # Convert to OpenCV image
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # Find features
            lk_result: LKResult = self._vio.process_frame(frame)
            # Show image using OpenCV
            self.show_features(frame, lk_result)
        except Exception as e:
            self.get_logger().error(f"Failed to process frame: {e}")
            return
        
        try:
            rotation_matrix = self.get_rotation_matrix()
        except Exception as e:
            self.get_logger().error(f"Failed to lookup transform: {e}")
            return
        
        
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
        


    def camera_info_callback(self, msg: CameraInfo):
        self.camera_info_data = CameraInfoData(
            K=np.array(msg.k).reshape((3,3)),
            distortion_coef=np.array(msg.d)
            )

    def altitude_callback(self, msg: Altitude):
        self.get_logger().info(f"rome altitude: {msg}")
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