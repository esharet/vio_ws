import rclpy
from rclpy.node import Node

from mavros_msgs.msg import GimbalDeviceAttitudeStatus
import math
from scipy.spatial.transform import Rotation as R

GIMBAL_STATUS_TOPIC = '/mavros/gimbal_control/device/attitude_status'


class AltitudeManager(Node):
    def __init__(self):
        super().__init__('altitude_manager')
        self.get_logger().info('AltitudeManager node has been started.')
        self.subscription_attitude = self.create_subscription(
            GimbalDeviceAttitudeStatus,
            GIMBAL_STATUS_TOPIC,
            self.gimbal_attitude_status_callback,
            10
        )
        # Timer to call pub_alt 9 times per second (period = 1/9 seconds)
        # self.timer = self.create_timer(1.0 / 9.0, self.pub_alt)

    def gimbal_attitude_status_callback(self, msg):
        # Extract quaternion from message
        q = msg.q  # geometry_msgs/Quaternion

        # Convert quaternion to [x, y, z, w] format for scipy
        quat = [q.x, q.y, q.z, q.w]

        # Use scipy to convert quaternion to roll, pitch, yaw
        r = R.from_quat(quat)
        roll, pitch, yaw = r.as_euler('xyz', degrees=True)

        self.get_logger().info(
            f'Received attitude status message: roll={roll:.2f}, pitch={pitch:.2f}, yaw={yaw:.2f}'
        )

    def pub_alt(self):
        self.get_logger().info('pub_alt called')

def main(args=None):
    rclpy.init(args=args)
    node = AltitudeManager()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()