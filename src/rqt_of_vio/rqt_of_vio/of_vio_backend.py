import os
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from rcl_interfaces.srv import GetParameters
from rclpy.clock import Clock
import pathlib
import numpy as np
from threading import Thread
from . import Event

TOPIC_ESTIMATE_VELOCITY = "xxx"
TOPIC_MAVROS_VELOCITY = "yyy"
TEST = False

def sine_generator(frequency=1.0, sample_rate=10, amplitude=10.0):
    """
    Infinite generator of sine wave values.
    Each call to next() returns the next sample.
    """
    t = 0.0
    dt = 1.0 / sample_rate
    while True:
        yield amplitude * np.sin(2 * np.pi * frequency * t)
        t += dt

def cos_generator(frequency=1.0, sample_rate=10, amplitude=10.0):
    """
    Infinite generator of sine wave values.
    Each call to next() returns the next sample.
    """
    t = 0.0
    dt = 1.0 / sample_rate
    while True:
        yield amplitude * np.cos(2 * np.pi * frequency * t)
        t += dt

class BackendNode(Node):
    def __init__(self):
        super().__init__('of_vio')

        self.on_estimate_velocity = Event()
        self.on_truth_velocity = Event()

        self.create_subscription(TwistStamped, TOPIC_ESTIMATE_VELOCITY, self.__estimate_velocity_handler, 10)
        self.create_subscription(TwistStamped, TOPIC_MAVROS_VELOCITY, self.__mavros_velocity_handler, 10)

        if TEST:
            self.test_pub = self.create_publisher(TwistStamped, TOPIC_ESTIMATE_VELOCITY, 10)
            self.test_mavros_pub = self.create_publisher(TwistStamped, TOPIC_MAVROS_VELOCITY, 10)
            self.t = sine_generator(frequency=1, sample_rate=10)
            self.tc = cos_generator(frequency=1, sample_rate=10)

            self.timer = self.create_timer(1.0, self.sim_estimate_velocity)

    #region handlers
    def __mavros_velocity_handler(self, msg: TwistStamped):
        x = msg.twist.linear.x
        y = msg.twist.linear.y
        self.on_truth_velocity.fire(x, y)

    def __estimate_velocity_handler(self, msg: TwistStamped):
        x = msg.twist.linear.x
        y = msg.twist.linear.y
        self.on_estimate_velocity.fire(x, y)

    def sim_estimate_velocity(self):
        msg = TwistStamped()
        
        msg.header.stamp = Clock().now().to_msg()
        msg.twist.linear.x = next(self.t)
        msg.twist.linear.y = next(self.t)
        self.test_pub.publish(msg)

        msg.header.stamp = Clock().now().to_msg()
        msg.twist.linear.x = next(self.tc)
        msg.twist.linear.y = next(self.tc)
        self.test_mavros_pub.publish(msg)

    #endregion
    def start(self):
        # Start a thread to spin the node so GUI stays responsive
        self.executor_thread = Thread(target=self._spin)
        self.executor_thread.daemon = True
        self.executor_thread.start()

    def _spin(self):
        executor = rclpy.executors.SingleThreadedExecutor()
        executor.add_node(self)
        executor.spin()