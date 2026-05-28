import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import TwistStamped, PoseStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, Path


class OrchardLaneNavigator(Node):
    def __init__(self):
        super().__init__("orchard_lane_navigator")

        self.cmd_pub = self.create_publisher(TwistStamped, "/cmd_vel", 10)
        self.path_pub = self.create_publisher(Path, "/orchard_search_path", 10)

        self.scan_sub = self.create_subscription(
            LaserScan, "/scan", self.scan_callback, 10
        )

        self.odom_sub = self.create_subscription(
            Odometry, "/odom", self.odom_callback, 10
        )

        self.timer = self.create_timer(0.1, self.control_loop)

        self.front_distance = float("inf")
        self.front_left_distance = float("inf")
        self.front_right_distance = float("inf")
        self.left_distance = float("inf")
        self.right_distance = float("inf")

        self.front_blocked_ratio = 0.0

        self.forward_speed = 0.06
        self.gap_move_speed = 0.04
        self.enter_lane_speed = 0.04
        self.turn_speed = 0.22
        self.gap_curve_speed = 0.12

        self.end_wall_distance = 0.75
        self.corner_safe_distance = 0.85

        self.enter_gap_front_distance = 0.95
        self.enter_gap_corner_distance = 0.80

        self.wall_block_threshold = 0.55

        self.max_correction_speed = 0.05
        self.correction_gain = 0.06
        self.side_difference_deadband = 0.20

        self.state = "FOLLOW_LANE"
        self.turn_direction = None

        self.search_duration = 150.0
        self.start_time = self.get_clock().now()
        self.state_start_time = self.get_clock().now()

        self.move_toward_gap_duration = 1.4
        self.min_rotate_time = 1.2
        self.max_rotate_time = 5.0
        self.enter_lane_duration = 2.0

        self.path_msg = Path()
        self.path_msg.header.frame_id = "odom"

        self.search_complete_logged = False

        self.get_logger().info("Autonomous orchard lane navigator started.")

    def scan_callback(self, msg):
        ranges = list(msg.ranges)

        front_ranges = ranges[:90] + ranges[-90:]
        front_left_ranges = ranges[20:100]
        front_right_ranges = ranges[-100:-20]

        left_ranges = ranges[55:125]
        right_ranges = ranges[235:305]

        self.front_distance = self.get_min_valid_distance(front_ranges)
        self.front_left_distance = self.get_min_valid_distance(front_left_ranges)
        self.front_right_distance = self.get_min_valid_distance(front_right_ranges)
        self.left_distance = self.get_min_valid_distance(left_ranges)
        self.right_distance = self.get_min_valid_distance(right_ranges)

        self.front_blocked_ratio = self.get_blocked_ratio(
            front_ranges,
            self.end_wall_distance
        )

    def get_min_valid_distance(self, ranges):
        valid_ranges = [
            r for r in ranges
            if not math.isinf(r) and not math.isnan(r) and r > 0.0
        ]
        return min(valid_ranges) if valid_ranges else float("inf")

    def get_blocked_ratio(self, ranges, threshold):
        valid_ranges = [
            r for r in ranges
            if not math.isinf(r) and not math.isnan(r) and r > 0.0
        ]

        if not valid_ranges:
            return 0.0

        blocked_count = sum(1 for r in valid_ranges if r < threshold)
        return blocked_count / len(valid_ranges)

    def odom_callback(self, msg):
        pose = PoseStamped()
        pose.header = msg.header
        pose.pose = msg.pose.pose

        self.path_msg.header.stamp = self.get_clock().now().to_msg()
        self.path_msg.poses.append(pose)
        self.path_pub.publish(self.path_msg)

    def seconds_since(self, start_time):
        return (self.get_clock().now() - start_time).nanoseconds / 1e9

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

    def set_state(self, new_state):
        if self.state != new_state:
            self.state = new_state
            self.state_start_time = self.get_clock().now()
            self.get_logger().info(f"State changed to: {self.state}")

    def wall_detected(self):
        return self.front_blocked_ratio >= self.wall_block_threshold

    def front_is_safe(self):
        return (
            self.front_left_distance > self.corner_safe_distance and
            self.front_right_distance > self.corner_safe_distance
        )

    def gap_is_safe_to_enter(self):
        return (
            self.front_distance > self.enter_gap_front_distance and
            self.front_left_distance > self.enter_gap_corner_distance and
            self.front_right_distance > self.enter_gap_corner_distance
        )

    def choose_open_side(self):
        if self.left_distance > self.right_distance:
            return "LEFT"
        return "RIGHT"

    def calculate_lane_correction(self):
        side_difference = self.left_distance - self.right_distance

        if abs(side_difference) < self.side_difference_deadband:
            return 0.0

        correction = -side_difference * self.correction_gain

        return max(
            -self.max_correction_speed,
            min(self.max_correction_speed, correction)
        )

    def apply_lane_following(self, cmd, speed):
        cmd.twist.linear.x = speed
        cmd.twist.angular.z = self.calculate_lane_correction()

    def apply_move_toward_gap(self, cmd):
        cmd.twist.linear.x = self.gap_move_speed

        if self.turn_direction == "LEFT":
            cmd.twist.angular.z = self.gap_curve_speed
        else:
            cmd.twist.angular.z = -self.gap_curve_speed

    def apply_rotate_to_gap(self, cmd):
        cmd.twist.linear.x = 0.0

        if self.turn_direction == "LEFT":
            cmd.twist.angular.z = self.turn_speed
        else:
            cmd.twist.angular.z = -self.turn_speed

    def control_loop(self):
        cmd = self.create_cmd()

        elapsed = self.seconds_since(self.start_time)
        state_elapsed = self.seconds_since(self.state_start_time)

        if elapsed >= self.search_duration:
            self.set_state("SEARCH_COMPLETE")
            self.stop_robot()

            if not self.search_complete_logged:
                self.get_logger().info("2.5-minute autonomous search completed.")
                self.search_complete_logged = True
            return

        if self.state == "FOLLOW_LANE":
            if self.wall_detected():
                self.turn_direction = self.choose_open_side()
                self.set_state("MOVE_TOWARD_GAP")

                self.get_logger().info(
                    f"End wall detected. "
                    f"blocked_ratio={self.front_blocked_ratio:.2f}, "
                    f"left={self.left_distance:.2f}m, "
                    f"right={self.right_distance:.2f}m, "
                    f"chosen={self.turn_direction}."
                )

                self.apply_move_toward_gap(cmd)

            elif self.front_is_safe():
                self.apply_lane_following(cmd, self.forward_speed)

            else:
                self.get_logger().info(
                    f"Tree/obstacle nearby, correcting only. "
                    f"front_left={self.front_left_distance:.2f}m, "
                    f"front_right={self.front_right_distance:.2f}m."
                )
                self.apply_lane_following(cmd, self.forward_speed * 0.7)

        elif self.state == "MOVE_TOWARD_GAP":
            self.apply_move_toward_gap(cmd)

            if state_elapsed >= self.move_toward_gap_duration:
                self.set_state("ROTATE_TO_GAP")

        elif self.state == "ROTATE_TO_GAP":
            self.apply_rotate_to_gap(cmd)

            if state_elapsed >= self.min_rotate_time and self.gap_is_safe_to_enter():
                self.set_state("ENTER_LANE")

            elif state_elapsed >= self.max_rotate_time:
                self.set_state("ENTER_LANE")

        elif self.state == "ENTER_LANE":
            if self.gap_is_safe_to_enter():
                self.apply_lane_following(cmd, self.enter_lane_speed)
            else:
                self.apply_rotate_to_gap(cmd)

            if state_elapsed >= self.enter_lane_duration:
                self.set_state("FOLLOW_LANE")

        elif self.state == "SEARCH_COMPLETE":
            self.stop_robot()
            return

        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = OrchardLaneNavigator()

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