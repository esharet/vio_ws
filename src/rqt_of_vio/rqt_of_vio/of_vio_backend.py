import os
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger, SetBool
from rcl_interfaces.srv import GetParameters
import subprocess
import pathlib
import yaml


class BackendNode(Node):
    def __init__(self):
        super().__init__('of_vio')