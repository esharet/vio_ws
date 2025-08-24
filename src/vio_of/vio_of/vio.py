#!/usr/bin/env python3

import rclpy
from rclpy.qos import qos_profile_sensor_data
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import cv2
import traceback
import numpy as np


TOPIC_IMAGE = "/gimbal_camera/image_raw"
WIDTH = 640
HEIGHT = 512

class MyNode(Node):
    def __init__(self):
        node_name="minimal"
        super().__init__(node_name)
        self.bridge = CvBridge()

        PAGE_LOCKED = 1
        self.img_source_pin = cv2.cuda.HostMem(HEIGHT, WIDTH, cv2.CV_8UC3, PAGE_LOCKED)
        self.img_source = self.img_source_pin.createMatHeader() 
        self.gpu_source = cv2.cuda.GpuMat()
        self.p0_gpu = cv2.cuda.GpuMat()
        self._init_of()
        self._init_subscribers()

    def _init_of(self):
        if cv2.cuda.getCudaEnabledDeviceCount() == 0:
            self.get_logger().error("GPU Not support")
        self.gpu_detector = cv2.cuda.createGoodFeaturesToTrackDetector(
            cv2.CV_8UC1,
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )

        self.lk = cv2.cuda.SparsePyrLKOpticalFlow_create(
            winSize=(15, 15), 
            maxLevel=2,
            iters=10, 
            useInitialFlow=False
    )

    def _init_subscribers(self):
        self.sub_img = self.create_subscription(Image,
            TOPIC_IMAGE,
            self.img_handler,
            qos_profile=qos_profile_sensor_data)

    def calc(self, img):
        self.gpu_source.upload(img)
        gpu_gray = cv2.cuda.cvtColor(self.gpu_source, cv2.COLOR_BGR2GRAY)

        good_new = np.array([], dtype=np.float32).reshape(0,2)
        good_old = np.array([], dtype=np.float32).reshape(0,2)

        if self.p0_gpu.empty():
            self.get_logger().info("---- detect -----")
            for i in range(9):
                self.p0_gpu = self.gpu_detector.detect(gpu_gray)
                self.gpu_gray_prev = gpu_gray.clone()
        else:
            p1_gpu, status_gpu, error_gpu = None, None, None
            for i in range(9):
                p1_gpu, status_gpu, error_gpu = self.lk.calc(
                    self.gpu_gray_prev,
                    gpu_gray,
                    self.p0_gpu,
                    None
                )

            p0 = self.p0_gpu.download().reshape(-1, 2)
            p1 = p1_gpu.download().reshape(-1, 2)
            status = status_gpu.download().reshape(-1).astype(bool)

            good_new = p1[status]
            good_old = p0[status]

            if len(good_new) > 10:
                self.p0_gpu.upload(good_new.reshape(1, -1, 2).astype(np.float32))
            else:
                self.p0_gpu = cv2.cuda.GpuMat()

        
        
            for new, old in zip(good_new, good_old):
                a, b = new.ravel().astype(int)
                c, d = old.ravel().astype(int)
                # cv2.arrowedLine(img, (c, d), (a, b), (0, 255, 255), 2, tipLength=0.3)
                cv2.circle(img, (a, b), 3, (0, 0, 255), -1)
                cv2.circle(img, (c, d), 3, (255, 0, 0), -1)

        self.gpu_gray_prev = gpu_gray.clone()
        return img
       

    #region subscribers handler
    def img_handler(self, msg: Image):
        try:
            # Convert to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Show image using OpenCV
            img = self.calc(cv_image)
            # cv2.imshow('Camera Image', img)
            # cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}\n\n---\n{traceback.format_exc()}')
            exit()
    #end region subscribers handler

def main(args=None):
    rclpy.init(args=args)
    node = MyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()