import math
import time

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TwistStamped, PoseStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, Path


class OrchardNavigation(Node):
    def __init__(self):
        super().__init__("orchard_navigation")

        self.cmd_pub = self.create_publisher(TwistStamped, "/cmd_vel", 10)
        self.path_pub = self.create_publisher(Path, "/orchard_search_path", 10)

        self.scan_sub = self.create_subscription(LaserScan, "/scan", self.scan_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, "/odom", self.odom_callback, 10)

        self.timer = self.create_timer(0.1, self.control_loop)

        self.state = "SEARCHING"

        self.front_distance = float("inf")
        self.front_left_distance = float("inf")
        self.front_right_distance = float("inf")
        self.left_distance = float("inf")
        self.right_distance = float("inf")

        self.forward_speed = 0.07
        self.turn_speed = 0.25

        self.safe_front_distance = 1.05
        self.corner_safe_distance = 0.75

        self.correction_speed = 0.05
        self.side_difference_threshold = 0.20

        # Fixed in-place turn. No arc movement.
        self.turn_duration = 2.5
        self.turn_start_time = None
        self.turn_direction = None
        self.default_turn_direction = "LEFT"

        self.start_time = time.time()
        self.search_duration = 150.0

        self.path_msg = Path()
        self.path_msg.header.frame_id = "odom"

        self.search_complete_logged = False

        self.get_logger().info("Orchard navigation node started using TwistStamped.")

    def scan_callback(self, msg):
        ranges = list(msg.ranges)

        front_ranges = ranges[:70] + ranges[-70:]
        front_left_ranges = ranges[20:80]
        front_right_ranges = ranges[-80:-20]

        left_ranges = ranges[55:125]
        right_ranges = ranges[235:305]

        self.front_distance = self.get_min_valid_distance(front_ranges)
        self.front_left_distance = self.get_min_valid_distance(front_left_ranges)
        self.front_right_distance = self.get_min_valid_distance(front_right_ranges)
        self.left_distance = self.get_min_valid_distance(left_ranges)
        self.right_distance = self.get_min_valid_distance(right_ranges)

    def get_min_valid_distance(self, ranges):
        valid_ranges = [
            r for r in ranges
            if not math.isinf(r) and not math.isnan(r) and r > 0.0
        ]
        return min(valid_ranges) if valid_ranges else float("inf")

    def odom_callback(self, msg):
        pose = PoseStamped()
        pose.header = msg.header
        pose.pose = msg.pose.pose

        self.path_msg.header.stamp = self.get_clock().now().to_msg()
        self.path_msg.poses.append(pose)
        self.path_pub.publish(self.path_msg)

    def create_cmd(self):
        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = "base_link"
        return cmd

    def stop_robot(self):
        cmd = self.create_cmd()
        cmd.twist.linear.x = 0.0
        cmd.twist.angular.z = 0.0
        self.cmd_pub.publish(cmd)

    def reset_turn_lock(self):
        self.turn_start_time = None
        self.turn_direction = None

    def choose_turn_direction(self):
        return self.default_turn_direction

    def front_is_clear(self):
        return (
            self.front_distance > self.safe_front_distance and
            self.front_left_distance > self.corner_safe_distance and
            self.front_right_distance > self.corner_safe_distance
        )

    def apply_turning_motion(self, cmd):
        self.state = f"ROTATING_{self.turn_direction}"
        cmd.twist.linear.x = 0.0

        if self.turn_direction == "LEFT":
            cmd.twist.angular.z = self.turn_speed
        else:
            cmd.twist.angular.z = -self.turn_speed

    def control_loop(self):
        cmd = self.create_cmd()

        if time.time() - self.start_time >= self.search_duration:
            self.state = "SEARCH_COMPLETE"
            self.stop_robot()

            if not self.search_complete_logged:
                self.get_logger().info("2.5-minute search completed. Robot stopped.")
                self.search_complete_logged = True
            return

        if self.turn_direction is not None:
            if time.time() - self.turn_start_time < self.turn_duration:
                self.apply_turning_motion(cmd)
                self.cmd_pub.publish(cmd)
                return

            self.reset_turn_lock()

        if self.front_is_clear():
            self.state = "MOVING_FORWARD"
            cmd.twist.linear.x = self.forward_speed

            side_difference = self.left_distance - self.right_distance

            if side_difference < -self.side_difference_threshold:
                cmd.twist.angular.z = -self.correction_speed
            elif side_difference > self.side_difference_threshold:
                cmd.twist.angular.z = self.correction_speed
            else:
                cmd.twist.angular.z = 0.0

        else:
            self.state = "TURNING"
            self.turn_direction = self.choose_turn_direction()
            self.turn_start_time = time.time()

            self.apply_turning_motion(cmd)

            self.get_logger().info(
                f"Obstacle zone detected | front={self.front_distance:.2f}m, "
                f"front_left={self.front_left_distance:.2f}m, "
                f"front_right={self.front_right_distance:.2f}m. "
                f"Mode={self.state}."
            )

        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = OrchardNavigation()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt received. Stopping robot.")
        node.stop_robot()

    finally:
        node.stop_robot()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()