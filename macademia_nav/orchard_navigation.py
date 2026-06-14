import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped, PoseStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, Path


def quat_to_yaw(x, y, z, w):
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def yaw_error(target, current):
    err = target - current
    while err > math.pi:
        err -= 2 * math.pi
    while err < -math.pi:
        err += 2 * math.pi
    return err


class OrchardNavigator(Node):
    """
    Layered architecture orchard navigator for ROSbot Pro.
    Serpentine (boustrophedon) coverage pattern.

    ── Quick-config block ─────────────────────────────────────────────
    Change only these values before each demo run:
      NUM_LANES         — how many lanes to cover
      NUM_TREES_PER_ROW — reduce if lab space is tight (min 2)
      TREE_SPACING      — measured distance between tree centres (m)
      LANE_WIDTH        — measured distance between lane centres (m)
    Everything else auto-calculates from these four.
    ───────────────────────────────────────────────────────────────────
    """

    # ── Quick-config ───────────────────────────────────────────────── #
    NUM_LANES         = 3
    NUM_TREES_PER_ROW = 3      # drop to 3 or 2 if space is tight
    TREE_SPACING      = 0.75    # metres between tree centres in a row
    LANE_WIDTH        = 1.0    # metres between lane centres

    # ── Speeds ────────────────────────────────────────────────────── #
    FORWARD_SPEED     = 0.08    # m/s along lane
    TURN_SPEED        = 0.25    # rad/s in-place rotation
    CROSS_SPEED       = 0.06    # m/s crossing headland

    # ── Row-end detection ─────────────────────────────────────────── #
    ROW_END_OPEN_DIST = 1.0     # both sides must open beyond this (m)
    ROW_END_CONFIRM   = 0.5     # seconds both sides must stay open

    # ── Straight-line heading hold ────────────────────────────────── #
    HEADING_GAIN      = 0.8
    MAX_HEADING_CORR  = 0.30    # rad/s max correction

    # ── Reactive thresholds ───────────────────────────────────────── #
    FRONT_STOP_LANE   = 0.25    # stop distance during lane following
    FRONT_STOP_CROSS  = 0.20    # stop distance during headland moves
    SIDE_DANGER_DIST  = 0.18    # hard steer if side closer than this
    SIDE_SLOW_DIST    = 0.30    # slow to 50% if side closer than this

    # ── Lane centering (secondary, gentle) ────────────────────────── #
    CENTER_GAIN       = 0.04
    CENTER_DEADBAND   = 0.15

    # ── Headland geometry ─────────────────────────────────────────── #
    CLEAR_DIST        = 0.30    # drive past last tree before turning
    TURN_TOLERANCE    = 0.08    # radians — close enough to target yaw

    # ── Auto-calculated from quick-config ─────────────────────────── #
    @property
    def row_length(self):
        """Total row length based on tree count and spacing."""
        return self.NUM_TREES_PER_ROW * self.TREE_SPACING

    @property
    def lane_min_dist(self):
        """
        Minimum distance before row-end can trigger.
        60% of row length — robot must be well into the row
        before both-sides-open detection activates.
        """
        return self.row_length * 0.60

    def __init__(self):
        super().__init__("orchard_navigator")

        self.cmd_pub  = self.create_publisher(TwistStamped, "/cmd_vel", 10)
        self.path_pub = self.create_publisher(Path, "/orchard_path", 10)
        self.scan_sub = self.create_subscription(
            LaserScan, "/scan", self.scan_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, "/odometry/filtered", self.odom_callback, 10)
        self.timer = self.create_timer(0.1, self.control_loop)

        self.num_ranges       = None
        self.scan_ranges      = []
        self.front_dist       = float("inf")
        self.left_dist        = float("inf")
        self.right_dist       = float("inf")
        self.front_left_dist  = float("inf")
        self.front_right_dist = float("inf")

        self.current_yaw = 0.0
        self.current_x   = 0.0
        self.current_y   = 0.0

        self.state            = "INIT"
        self.state_start_time = self.get_clock().now()
        self.mission_start    = self.get_clock().now()

        self.lane_number        = 0
        self.lane_heading       = None
        self.cross_heading      = None
        self.lane_start_x       = 0.0
        self.lane_start_y       = 0.0
        self.row_end_open_since = None
        self.turn_target_yaw    = None
        self.headland_start_x   = 0.0
        self.headland_start_y   = 0.0

        self.path_msg = Path()
        self.path_msg.header.frame_id = "odom"

        self.get_logger().info(
            f"Orchard navigator ready — "
            f"{self.NUM_LANES} lanes x {self.NUM_TREES_PER_ROW} trees, "
            f"row length={self.row_length:.2f}m, "
            f"min trigger dist={self.lane_min_dist:.2f}m, "
            f"lane width={self.LANE_WIDTH}m.")

    # ================================================================== #
    # SENSING LAYER                                                        #
    # ================================================================== #

    def angle_to_index(self, angle_deg):
        angle_deg = angle_deg % 360
        return int(round(angle_deg / 360.0 * self.num_ranges)) % self.num_ranges

    def get_arc(self, from_deg, to_deg):
        if self.num_ranges is None:
            return []
        start = self.angle_to_index(from_deg)
        end   = self.angle_to_index(to_deg)
        if start <= end:
            return list(self.scan_ranges[start:end + 1])
        return list(self.scan_ranges[start:]) + list(self.scan_ranges[:end + 1])

    def min_valid(self, ranges):
        valid = [r for r in ranges
                 if not math.isinf(r) and not math.isnan(r) and r > 0.05]
        return min(valid) if valid else float("inf")

    def scan_callback(self, msg):
        self.scan_ranges = list(msg.ranges)
        self.num_ranges  = len(self.scan_ranges)
        self.front_dist       = self.min_valid(self.get_arc(330, 30))
        self.front_left_dist  = self.min_valid(self.get_arc(10,  60))
        self.front_right_dist = self.min_valid(self.get_arc(300, 350))
        self.left_dist        = self.min_valid(self.get_arc(60,  120))
        self.right_dist       = self.min_valid(self.get_arc(240, 300))

    def odom_callback(self, msg):
        p = msg.pose.pose
        self.current_x   = p.position.x
        self.current_y   = p.position.y
        self.current_yaw = quat_to_yaw(
            p.orientation.x, p.orientation.y,
            p.orientation.z, p.orientation.w)
        pose = PoseStamped()
        pose.header = msg.header
        pose.pose   = p
        self.path_msg.header.stamp = self.get_clock().now().to_msg()
        self.path_msg.poses.append(pose)
        self.path_pub.publish(self.path_msg)

    # ================================================================== #
    # UTILITIES                                                            #
    # ================================================================== #

    def create_cmd(self):
        cmd = TwistStamped()
        cmd.header.stamp    = self.get_clock().now().to_msg()
        cmd.header.frame_id = "base_link"
        return cmd

    def stop(self):
        try:
            if rclpy.ok():
                self.cmd_pub.publish(self.create_cmd())
        except Exception:
            pass

    def set_state(self, s):
        if self.state != s:
            self.state            = s
            self.state_start_time = self.get_clock().now()
            self.get_logger().info(
                f"-> {s}  [lane {self.lane_number + 1}/{self.NUM_LANES}]")

    def state_elapsed(self):
        return (self.get_clock().now() - self.state_start_time).nanoseconds / 1e9

    def dist_from(self, x, y):
        return math.hypot(self.current_x - x, self.current_y - y)

    def normalise_yaw(self, yaw):
        return math.atan2(math.sin(yaw), math.cos(yaw))

    # ================================================================== #
    # REACTIVE LAYER                                                       #
    # ================================================================== #

    def get_front_stop_dist(self):
        if self.state in {"TURN_INTO_HEADLAND", "TURN_INTO_LANE"}:
            return 0.0
        elif self.state in {"CLEAR_TREE", "CROSS_TO_LANE"}:
            return self.FRONT_STOP_CROSS
        else:
            return self.FRONT_STOP_LANE

    def reactive_override(self, cmd):
        front_limit = self.get_front_stop_dist()
        if front_limit > 0.0 and self.front_dist < front_limit:
            self.get_logger().warn(
                f"[REACTIVE] Front {self.front_dist:.2f}m "
                f"limit={front_limit:.2f}m state={self.state} — stopping.")
            return True

        if self.state == "FOLLOW_LANE":
            if self.left_dist < self.SIDE_DANGER_DIST:
                cmd.twist.linear.x  = self.FORWARD_SPEED * 0.4
                cmd.twist.angular.z = -self.TURN_SPEED * 0.6
                self.get_logger().warn(
                    f"[REACTIVE] Left {self.left_dist:.2f}m — steering right.")
                return True
            if self.right_dist < self.SIDE_DANGER_DIST:
                cmd.twist.linear.x  = self.FORWARD_SPEED * 0.4
                cmd.twist.angular.z = self.TURN_SPEED * 0.6
                self.get_logger().warn(
                    f"[REACTIVE] Right {self.right_dist:.2f}m — steering left.")
                return True

        return False

    # ================================================================== #
    # EXECUTIVE LAYER                                                      #
    # ================================================================== #

    def drive_straight(self, cmd, speed=None, heading=None):
        if speed is None:
            speed = self.FORWARD_SPEED
        if heading is None:
            heading = self.lane_heading

        if self.state == "FOLLOW_LANE":
            if (self.left_dist  < self.SIDE_SLOW_DIST or
                    self.right_dist < self.SIDE_SLOW_DIST):
                speed *= 0.5

        heading_corr = 0.0
        if heading is not None:
            err = yaw_error(heading, self.current_yaw)
            heading_corr = max(-self.MAX_HEADING_CORR,
                               min(self.MAX_HEADING_CORR,
                                   self.HEADING_GAIN * err))

        center_corr = 0.0
        if self.state == "FOLLOW_LANE":
            side_diff = self.left_dist - self.right_dist
            if abs(side_diff) > self.CENTER_DEADBAND:
                center_corr = max(-0.08, min(0.08,
                                             -side_diff * self.CENTER_GAIN))

        cmd.twist.linear.x  = speed
        cmd.twist.angular.z = heading_corr + center_corr

    def row_end_detected(self):
        dist_travelled = self.dist_from(self.lane_start_x, self.lane_start_y)

        # Must travel at least lane_min_dist (auto-calculated from tree count)
        if dist_travelled < self.lane_min_dist:
            return False

        both_open = (self.left_dist  > self.ROW_END_OPEN_DIST and
                     self.right_dist > self.ROW_END_OPEN_DIST)

        now = self.get_clock().now()
        if both_open:
            if self.row_end_open_since is None:
                self.row_end_open_since = now
            elapsed = (now - self.row_end_open_since).nanoseconds / 1e9
            if elapsed >= self.ROW_END_CONFIRM:
                self.get_logger().info(
                    f"Row end detected — "
                    f"dist={dist_travelled:.2f}m "
                    f"(row={self.row_length:.2f}m) "
                    f"L={self.left_dist:.2f}m R={self.right_dist:.2f}m")
                return True
        else:
            self.row_end_open_since = None
        return False

    def rotate_to(self, cmd, target_yaw):
        err = yaw_error(target_yaw, self.current_yaw)
        if abs(err) < self.TURN_TOLERANCE:
            return True
        cmd.twist.angular.z = math.copysign(self.TURN_SPEED, err)
        return False

    # ================================================================== #
    # MISSION LAYER                                                        #
    # ================================================================== #

    def start_lane(self):
        self.lane_heading       = self.current_yaw
        self.lane_start_x       = self.current_x
        self.lane_start_y       = self.current_y
        self.row_end_open_since = None
        self.get_logger().info(
            f"Lane {self.lane_number + 1} start — "
            f"heading={math.degrees(self.lane_heading):.1f}deg "
            f"pos=({self.current_x:.2f}, {self.current_y:.2f}) "
            f"trees={self.NUM_TREES_PER_ROW} "
            f"row={self.row_length:.2f}m "
            f"trigger@{self.lane_min_dist:.2f}m")

    def get_turn_angle(self):
        """
        Serpentine pattern:
        Even lane index (0, 2) -> turn RIGHT (-90deg)
        Odd  lane index (1, 3) -> turn LEFT  (+90deg)
        """
        return -math.pi / 2 if self.lane_number % 2 == 0 else math.pi / 2

    # ================================================================== #
    # MAIN CONTROL LOOP                                                    #
    # ================================================================== #

    def control_loop(self):
        if self.num_ranges is None:
            return

        cmd = self.create_cmd()

        if self.state == "INIT":
            self.start_lane()
            self.set_state("FOLLOW_LANE")
            return

        if self.reactive_override(cmd):
            self.cmd_pub.publish(cmd)
            return

        if self.state == "FOLLOW_LANE":
            if int(self.state_elapsed() * 5) % 10 == 0:
                self.get_logger().info(
                    f"[lane {self.lane_number+1}] "
                    f"L={self.left_dist:.2f} R={self.right_dist:.2f} "
                    f"F={self.front_dist:.2f} "
                    f"dist={self.dist_from(self.lane_start_x, self.lane_start_y):.2f}/"
                    f"{self.row_length:.2f}m "
                    f"trigger@{self.lane_min_dist:.2f}m")

            if self.row_end_detected():
                self.headland_start_x = self.current_x
                self.headland_start_y = self.current_y
                # If this was the last lane, stop here — no headland needed
                if self.lane_number >= self.NUM_LANES - 1:
                    self.set_state("MISSION_COMPLETE")
                else:
                    self.set_state("CLEAR_TREE")
                return

            self.drive_straight(cmd)

        elif self.state == "CLEAR_TREE":
            if self.dist_from(self.headland_start_x,
                              self.headland_start_y) >= self.CLEAR_DIST:
                turn_angle = self.get_turn_angle()
                self.turn_target_yaw = self.normalise_yaw(
                    self.lane_heading + turn_angle)
                self.get_logger().info(
                    f"Cleared — turning "
                    f"{'RIGHT' if turn_angle < 0 else 'LEFT'} "
                    f"to {math.degrees(self.turn_target_yaw):.1f}deg")
                self.set_state("TURN_INTO_HEADLAND")
            else:
                self.drive_straight(cmd, speed=self.CROSS_SPEED)

        elif self.state == "TURN_INTO_HEADLAND":
            if self.rotate_to(cmd, self.turn_target_yaw):
                self.cross_heading    = self.current_yaw
                self.headland_start_x = self.current_x
                self.headland_start_y = self.current_y
                self.get_logger().info(
                    f"Aligned for headland — "
                    f"cross_heading={math.degrees(self.cross_heading):.1f}deg "
                    f"crossing {self.LANE_WIDTH}m to next lane")
                self.set_state("CROSS_TO_LANE")

        elif self.state == "CROSS_TO_LANE":
            crossed = self.dist_from(self.headland_start_x,
                                     self.headland_start_y)
            if int(self.state_elapsed() * 5) % 10 == 0:
                self.get_logger().info(
                    f"[crossing] {crossed:.2f}m / {self.LANE_WIDTH}m "
                    f"F={self.front_dist:.2f}m")

            if crossed >= self.LANE_WIDTH:
                turn_angle = self.get_turn_angle()
                self.turn_target_yaw = self.normalise_yaw(
                    self.lane_heading + turn_angle + turn_angle)
                self.get_logger().info(
                    f"Crossed {crossed:.2f}m — aligning into "
                    f"lane {self.lane_number + 2} "
                    f"target={math.degrees(self.turn_target_yaw):.1f}deg")
                self.set_state("TURN_INTO_LANE")
            else:
                self.drive_straight(cmd, speed=self.CROSS_SPEED,
                                    heading=self.cross_heading)

        elif self.state == "TURN_INTO_LANE":
            if self.rotate_to(cmd, self.turn_target_yaw):
                self.lane_number += 1
                if self.lane_number >= self.NUM_LANES:
                    self.set_state("MISSION_COMPLETE")
                else:
                    self.start_lane()
                    self.set_state("FOLLOW_LANE")

        elif self.state == "MISSION_COMPLETE":
            self.stop()
            total = (self.get_clock().now() -
                     self.mission_start).nanoseconds / 1e9
            self.get_logger().info(
                f"Mission complete — {self.NUM_LANES} lanes x "
                f"{self.NUM_TREES_PER_ROW} trees covered in {total:.1f}s.")
            return

        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = OrchardNavigator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted.")
    finally:
        node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
