#!/usr/bin/env python3

import rclpy
from rclpy.qos import qos_profile_sensor_data
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import cv2
import traceback
import numpy as np
from vio_core.optical_flow.lk import LK, LKResult

TOPIC_IMAGE = "/gimbal_camera/image_raw"
WIDTH = 640
HEIGHT = 512

class VIONode(Node):
    def __init__(self):
        node_name="optical_flow_node"
        super().__init__(node_name)
        self.bridge = CvBridge()
        self._init_subscribers()
        self._vio = LK(WIDTH, HEIGHT)

    

    def _init_subscribers(self):
        self.sub_img = self.create_subscription(Image,
            TOPIC_IMAGE,
            self.img_handler,
            qos_profile=qos_profile_sensor_data)

    
    #region subscribers handler
    def img_handler(self, msg: Image):
        COLOR_NEW = (0, 0, 255) # RED
        COLOR_OLD = (255, 0, 0) # BLUE
        try:
            # Convert to OpenCV image
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Show image using OpenCV
            good_new, good_old = self._vio.process_frame(frame)

            for new, old in zip(good_new, good_old):
                a, b = new.ravel().astype(int)
                c, d = old.ravel().astype(int)
                # cv2.arrowedLine(img, (c, d), (a, b), (0, 255, 255), 2, tipLength=0.3)
                cv2.circle(frame, (a, b), 3, COLOR_NEW, -1)
                cv2.circle(frame, (c, d), 3, COLOR_OLD, -1)

            cv2.imshow("LK Optical Flow CUDA", frame)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}\n\n---\n{traceback.format_exc()}')
            exit()
    #end region subscribers handler

def main(args=None):
    rclpy.init(args=args)
    node = VIONode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()