import math
import time

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, Path


class OrchardNavigation(Node):
    def __init__(self):
        super().__init__("orchard_navigation")

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.path_pub = self.create_publisher(Path, "/orchard_search_path", 10)

        self.scan_sub = self.create_subscription(
            LaserScan,
            "/scan",
            self.scan_callback,
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            "/odom",
            self.odom_callback,
            10
        )

        self.timer = self.create_timer(0.1, self.control_loop)

        self.state = "SEARCHING"

        self.front_distance = float("inf")
        self.left_distance = float("inf")
        self.right_distance = float("inf")

        self.forward_speed = 0.15
        self.turn_speed = 0.35
        self.safe_front_distance = 0.45

        self.start_time = time.time()
        self.search_duration = 120.0

        self.path_msg = Path()
        self.path_msg.header.frame_id = "odom"

        self.search_complete_logged = False

        self.get_logger().info("Orchard navigation node started.")

    def scan_callback(self, msg):
        ranges = list(msg.ranges)

        front_ranges = ranges[:20] + ranges[-20:]
        left_ranges = ranges[60:120]
        right_ranges = ranges[240:300]

        self.front_distance = self.get_min_valid_distance(front_ranges)
        self.left_distance = self.get_min_valid_distance(left_ranges)
        self.right_distance = self.get_min_valid_distance(right_ranges)

    def get_min_valid_distance(self, ranges):
        valid_ranges = [
            r for r in ranges
            if not math.isinf(r) and not math.isnan(r) and r > 0.0
        ]

        if valid_ranges:
            return min(valid_ranges)

        return float("inf")

    def odom_callback(self, msg):
        pose = PoseStamped()
        pose.header = msg.header
        pose.pose = msg.pose.pose

        self.path_msg.header.stamp = self.get_clock().now().to_msg()
        self.path_msg.poses.append(pose)

        self.path_pub.publish(self.path_msg)

    def control_loop(self):
        cmd = Twist()

        elapsed_time = time.time() - self.start_time

        if elapsed_time >= self.search_duration:
            self.state = "SEARCH_COMPLETE"
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.cmd_pub.publish(cmd)

            if not self.search_complete_logged:
                self.get_logger().info("2-minute search completed. Robot stopped.")
                self.search_complete_logged = True

            return

        if self.front_distance > self.safe_front_distance:
            self.state = "MOVING_FORWARD"
            cmd.linear.x = self.forward_speed

            # Simple lane balancing
            if self.left_distance < self.right_distance:
                cmd.angular.z = -0.12
            elif self.right_distance < self.left_distance:
                cmd.angular.z = 0.12
            else:
                cmd.angular.z = 0.0

        else:
            self.state = "TURNING"

            cmd.linear.x = 0.0

            if self.left_distance > self.right_distance:
                cmd.angular.z = self.turn_speed
            else:
                cmd.angular.z = -self.turn_speed

        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = OrchardNavigation()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt received. Stopping node.")

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()