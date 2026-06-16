import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped, PoseStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Int32


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

    Outbound: lane 1 → lane 2 → ... → lane N  (serpentine)
    Return:   rotate 180° → lane N back → cross → lane N-1 back
              → ... → lane 1 back → AT_HOME

    Return turn logic:
      On outbound, crossing from lane i to lane i+1:
        even lane_number → turn RIGHT twice
        odd  lane_number → turn LEFT  twice
      On return, the robot faces the OPPOSITE direction.
      So every turn is the opposite of what outbound does
      for the same crossing index.

    ── Quick-config ───────────────────────────────────────────────────
    NUM_LANES         — how many lanes to cover
    NUM_TREES_PER_ROW — reduce if lab space is tight (min 2)
    TREE_SPACING      — measured distance between tree centres (m)
    LANE_WIDTH        — measured distance between lane centres (m)
    ───────────────────────────────────────────────────────────────────
    """

    # ── Quick-config ───────────────────────────────────────────────── #
    NUM_LANES         = 2
    NUM_TREES_PER_ROW = 2
    TREE_SPACING      = 1.0
    LANE_WIDTH        = 0.75

    # ── Speeds ────────────────────────────────────────────────────── #
    FORWARD_SPEED     = 0.08
    TURN_SPEED        = 0.25
    CROSS_SPEED       = 0.06

    # ── Row-end detection ─────────────────────────────────────────── #
    ROW_END_OPEN_DIST = 0.90
    ROW_END_CONFIRM   = 0.30
    ROW_END_MIN_TIME  = 3.0
    ROW_END_WALL_DIST = 0.35

    # ── Straight-line heading hold ────────────────────────────────── #
    HEADING_GAIN      = 0.8
    MAX_HEADING_CORR  = 0.30

    # ── Reactive thresholds ───────────────────────────────────────── #
    FRONT_STOP_LANE   = 0.25
    FRONT_STOP_CROSS  = 0.20
    SIDE_DANGER_DIST  = 0.18
    SIDE_SLOW_DIST    = 0.30

    # ── Lane centering ────────────────────────────────────────────── #
    CENTER_GAIN         = 0.04
    CENTER_DEADBAND     = 0.15
    CENTER_START_DIST   = 0.30
    CENTER_START_TIME   = 2.0

    # ── Headland geometry ─────────────────────────────────────────── #
    CLEAR_DIST        = 0.30
    TURN_TOLERANCE    = 0.08

    # ── Coverage ──────────────────────────────────────────────────── #
    ROBOT_WIDTH       = 0.30

    @property
    def row_length(self):
        return self.NUM_TREES_PER_ROW * self.TREE_SPACING

    @property
    def lane_min_dist(self):
        return self.row_length * 0.75

    def __init__(self):
        super().__init__("orchard_navigator")

        self.cmd_pub  = self.create_publisher(TwistStamped, "/cmd_vel", 10)
        self.path_pub = self.create_publisher(Path, "/path_explore", 10)
        self.scan_sub = self.create_subscription(
            LaserScan, "/scan", self.scan_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, "/odometry/filtered", self.odom_callback, 10)
        self.nut_sub  = self.create_subscription(
            Int32, "/nut_count", self.nut_callback, 10)
        self.timer = self.create_timer(0.1, self.control_loop)

        # Sensing
        self.num_ranges       = None
        self.scan_ranges      = []
        self.front_dist       = float("inf")
        self.left_dist        = float("inf")
        self.right_dist       = float("inf")
        self.front_left_dist  = float("inf")
        self.front_right_dist = float("inf")

        # Odometry
        self.current_yaw = 0.0
        self.current_x   = 0.0
        self.current_y   = 0.0

        # State machine
        self.state            = "INIT"
        self.state_start_time = self.get_clock().now()
        self.mission_start    = self.get_clock().now()

        # Lane tracking
        self.lane_number        = 0
        self.lane_heading       = None
        self.cross_heading      = None
        self.lane_start_x       = 0.0
        self.lane_start_y       = 0.0
        self.lane_start_time    = self.get_clock().now()
        self.row_end_open_since = None
        self.turn_target_yaw    = None
        self.headland_start_x   = 0.0
        self.headland_start_y   = 0.0

        # Return tracking
        self.returning          = False
        self.return_lane        = 0

        # Coverage
        self.total_path_length  = 0.0
        self.last_x             = None
        self.last_y             = None

        # Nut tracking
        self.current_lane_nuts  = 0
        self.nuts_per_lane      = []
        self.nuts_return        = []
        self.total_nuts         = 0

        # Path
        self.path_msg = Path()
        self.path_msg.header.frame_id = "odom"

        self.get_logger().info(
            f"Orchard navigator ready — "
            f"{self.NUM_LANES} lanes x {self.NUM_TREES_PER_ROW} trees, "
            f"row={self.row_length:.2f}m "
            f"trigger@{self.lane_min_dist:.2f}m "
            f"lane width={self.LANE_WIDTH}m "
            f"[OUTBOUND + RETURN]")

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

        if self.last_x is not None:
            self.total_path_length += math.hypot(
                self.current_x - self.last_x,
                self.current_y - self.last_y)
        self.last_x = self.current_x
        self.last_y = self.current_y

        pose = PoseStamped()
        pose.header = msg.header
        pose.pose   = p
        self.path_msg.header.stamp = self.get_clock().now().to_msg()
        self.path_msg.poses.append(pose)
        self.path_pub.publish(self.path_msg)

    def nut_callback(self, msg):
        if msg.data > 0 and self.state == "FOLLOW_LANE":
            self.current_lane_nuts = max(self.current_lane_nuts, msg.data)

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
            direction = "RETURN" if self.returning else "OUTBOUND"
            self.get_logger().info(
                f"-> {s}  [{direction} lane "
                f"{self.lane_number + 1}/{self.NUM_LANES}]")

    def state_elapsed(self):
        return (self.get_clock().now() - self.state_start_time).nanoseconds / 1e9

    def lane_elapsed(self):
        return (self.get_clock().now() - self.lane_start_time).nanoseconds / 1e9

    def dist_from(self, x, y):
        return math.hypot(self.current_x - x, self.current_y - y)

    def normalise_yaw(self, yaw):
        return math.atan2(math.sin(yaw), math.cos(yaw))

    def calculate_coverage(self):
        orchard_area = self.row_length * (self.LANE_WIDTH * self.NUM_LANES)
        if orchard_area <= 0:
            return 0.0
        covered = self.total_path_length * self.ROBOT_WIDTH
        return min(100.0, (covered / orchard_area) * 100.0)

    # ================================================================== #
    # REACTIVE LAYER                                                       #
    # ================================================================== #

    def get_front_stop_dist(self):
        if self.state in {"TURN_INTO_HEADLAND", "TURN_INTO_LANE",
                          "RETURN_START_ROTATE"}:
            return 0.0
        elif self.state in {"CLEAR_TREE", "CROSS_TO_LANE"}:
            return self.FRONT_STOP_CROSS
        elif self.state == "FINAL_CLEAR":
            return 0.0
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
            dist = self.dist_from(self.lane_start_x, self.lane_start_y)
            if dist > self.row_length * 0.80:
                speed *= 0.6

        heading_corr = 0.0
        if heading is not None:
            err = yaw_error(heading, self.current_yaw)
            heading_corr = max(-self.MAX_HEADING_CORR,
                               min(self.MAX_HEADING_CORR,
                                   self.HEADING_GAIN * err))

        # Centering — requires distance AND time to avoid startup curve
        center_corr = 0.0
        if self.state == "FOLLOW_LANE":
            dist = self.dist_from(self.lane_start_x, self.lane_start_y)
            if (dist > self.CENTER_START_DIST and
                    self.lane_elapsed() > self.CENTER_START_TIME):
                side_diff = self.left_dist - self.right_dist
                if abs(side_diff) > self.CENTER_DEADBAND:
                    center_corr = max(-0.08, min(0.08,
                                                 -side_diff * self.CENTER_GAIN))

        cmd.twist.linear.x  = speed
        cmd.twist.angular.z = heading_corr + center_corr

    def row_end_detected(self):
        if self.lane_elapsed() < self.ROW_END_MIN_TIME:
            return False

        dist_travelled = self.dist_from(self.lane_start_x, self.lane_start_y)
        if dist_travelled < self.lane_min_dist:
            return False

        # Hard distance overshoot — prevents headland wall collision
        if dist_travelled > self.row_length * 1.15:
            self.get_logger().warn(
                f"Row end forced — overshot "
                f"{dist_travelled:.2f}m > {self.row_length * 1.15:.2f}m")
            return True

        # Wall fallback
        if self.front_dist < self.ROW_END_WALL_DIST:
            self.get_logger().warn(
                f"Row end forced — wall {self.front_dist:.2f}m "
                f"dist={dist_travelled:.2f}m")
            return True

        # Primary bilateral detection
        both_open = (self.left_dist  > self.ROW_END_OPEN_DIST and
                     self.right_dist > self.ROW_END_OPEN_DIST)

        now = self.get_clock().now()
        if both_open:
            if self.row_end_open_since is None:
                self.row_end_open_since = now
            elapsed = (now - self.row_end_open_since).nanoseconds / 1e9
            if elapsed >= self.ROW_END_CONFIRM:
                self.get_logger().info(
                    f"Row end — dist={dist_travelled:.2f}m "
                    f"t={self.lane_elapsed():.1f}s "
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

    def get_turn_angle(self):
        """
        Outbound crossing index = lane_number (0-based).
        Return  crossing index = return_lane (0-based).

        For BOTH, the pattern is:
          even crossing index → RIGHT (-90°) on OUTBOUND → LEFT (+90°) on RETURN
          odd  crossing index → LEFT  (+90°) on OUTBOUND → RIGHT(-90°) on RETURN

        Using the crossing index directly (not lane_number during return)
        ensures the direction is based on which crossing we are on,
        not which lane we are currently in.
        """
        if not self.returning:
            # Outbound: use lane_number as crossing index
            return -math.pi / 2 if self.lane_number % 2 == 0 else math.pi / 2
        else:
            # Return: use return_lane as crossing index, then negate
            # return_lane=0 mirrors outbound lane_number=N-1 crossing
            # The crossing that was RIGHT on outbound is LEFT on return
            base = -math.pi / 2 if self.return_lane % 2 == 0 else math.pi / 2
            return -base  # negate = mirror

    def record_lane_nuts(self):
        if self.returning:
            self.nuts_return.append(self.current_lane_nuts)
        else:
            self.nuts_per_lane.append(self.current_lane_nuts)
        self.total_nuts       += self.current_lane_nuts
        self.current_lane_nuts = 0

    def start_lane(self):
        self.lane_heading       = self.current_yaw
        self.lane_start_x       = self.current_x
        self.lane_start_y       = self.current_y
        self.lane_start_time    = self.get_clock().now()
        self.row_end_open_since = None
        direction = "RETURN" if self.returning else "OUTBOUND"
        self.get_logger().info(
            f"[{direction}] Lane {self.lane_number + 1} start — "
            f"heading={math.degrees(self.lane_heading):.1f}deg "
            f"pos=({self.current_x:.2f}, {self.current_y:.2f}) "
            f"row={self.row_length:.2f}m "
            f"trigger@{self.lane_min_dist:.2f}m")

    def log_summary(self):
        total        = (self.get_clock().now() -
                        self.mission_start).nanoseconds / 1e9
        coverage     = self.calculate_coverage()
        orchard_area = self.row_length * self.LANE_WIDTH * self.NUM_LANES

        self.get_logger().info("=" * 50)
        self.get_logger().info("MISSION COMPLETE — FULL SUMMARY")
        self.get_logger().info(f"  Lanes (outbound): {self.NUM_LANES}")
        self.get_logger().info(f"  Lanes (return)  : {self.NUM_LANES}")
        self.get_logger().info(f"  Trees per row   : {self.NUM_TREES_PER_ROW}")
        self.get_logger().info(f"  Row length      : {self.row_length:.2f}m")
        self.get_logger().info(f"  Lane width      : {self.LANE_WIDTH:.2f}m")
        self.get_logger().info(f"  Mission time    : {total:.1f}s")
        self.get_logger().info(f"  Total distance  : {self.total_path_length:.2f}m")
        self.get_logger().info(f"  Orchard area    : {orchard_area:.2f}m²")
        self.get_logger().info(f"  Coverage        : {coverage:.1f}%")
        self.get_logger().info("── Nut Detection (outbound) ────────")
        for i, c in enumerate(self.nuts_per_lane):
            self.get_logger().info(f"  Lane {i+1} [OUT]   : {c} nuts")
        self.get_logger().info("── Nut Detection (return) ──────────")
        for i, c in enumerate(self.nuts_return):
            lane_num = self.NUM_LANES - i
            self.get_logger().info(f"  Lane {lane_num} [RET]   : {c} nuts")
        self.get_logger().info(f"  Total nuts seen : {self.total_nuts}")
        self.get_logger().info("=" * 50)

    # ================================================================== #
    # MAIN CONTROL LOOP                                                    #
    # ================================================================== #

    def control_loop(self):
        if self.num_ranges is None:
            return

        cmd = self.create_cmd()

        # INIT
        if self.state == "INIT":
            self.start_lane()
            self.set_state("FOLLOW_LANE")
            return

        # Reactive
        if self.reactive_override(cmd):
            self.cmd_pub.publish(cmd)
            return

        # FOLLOW_LANE
        if self.state == "FOLLOW_LANE":
            if int(self.state_elapsed() * 5) % 10 == 0:
                direction = "RET" if self.returning else "OUT"
                self.get_logger().info(
                    f"[{direction} lane {self.lane_number+1}] "
                    f"L={self.left_dist:.2f} R={self.right_dist:.2f} "
                    f"F={self.front_dist:.2f} "
                    f"dist={self.dist_from(self.lane_start_x, self.lane_start_y):.2f}/"
                    f"{self.row_length:.2f}m "
                    f"t={self.lane_elapsed():.1f}s "
                    f"nuts={self.current_lane_nuts}")

            if self.row_end_detected():
                self.headland_start_x = self.current_x
                self.headland_start_y = self.current_y
                self.record_lane_nuts()

                if not self.returning:
                    if self.lane_number >= self.NUM_LANES - 1:
                        self.returning   = True
                        self.return_lane = 0
                        self.get_logger().info(
                            "Outbound complete — rotating 180° to start return")
                        self.turn_target_yaw = self.normalise_yaw(
                            self.lane_heading + math.pi)
                        self.set_state("RETURN_START_ROTATE")
                    else:
                        self.set_state("CLEAR_TREE")
                else:
                    if self.return_lane >= self.NUM_LANES - 1:
                        self.set_state("FINAL_CLEAR")
                    else:
                        self.set_state("CLEAR_TREE")
                return

            self.drive_straight(cmd)

        # RETURN_START_ROTATE
        elif self.state == "RETURN_START_ROTATE":
            if self.rotate_to(cmd, self.turn_target_yaw):
                self.get_logger().info(
                    f"180° rotation complete — "
                    f"now facing {math.degrees(self.current_yaw):.1f}deg "
                    f"starting return through lane {self.lane_number + 1}")
                self.start_lane()
                self.set_state("FOLLOW_LANE")

        # CLEAR_TREE
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

        # TURN_INTO_HEADLAND
        elif self.state == "TURN_INTO_HEADLAND":
            if self.rotate_to(cmd, self.turn_target_yaw):
                self.cross_heading    = self.current_yaw
                self.headland_start_x = self.current_x
                self.headland_start_y = self.current_y
                self.get_logger().info(
                    f"Aligned — cross_heading="
                    f"{math.degrees(self.cross_heading):.1f}deg "
                    f"crossing {self.LANE_WIDTH}m")
                self.set_state("CROSS_TO_LANE")

        # CROSS_TO_LANE
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
                    f"Crossed {crossed:.2f}m — aligning "
                    f"target={math.degrees(self.turn_target_yaw):.1f}deg")
                self.set_state("TURN_INTO_LANE")
            else:
                self.drive_straight(cmd, speed=self.CROSS_SPEED,
                                    heading=self.cross_heading)

        # TURN_INTO_LANE
        elif self.state == "TURN_INTO_LANE":
            if self.rotate_to(cmd, self.turn_target_yaw):
                if not self.returning:
                    self.lane_number += 1
                    self.start_lane()
                    self.set_state("FOLLOW_LANE")
                else:
                    self.return_lane += 1
                    self.lane_number -= 1
                    self.start_lane()
                    self.set_state("FOLLOW_LANE")

        # FINAL_CLEAR
        elif self.state == "FINAL_CLEAR":
            if self.dist_from(self.headland_start_x,
                              self.headland_start_y) >= self.CLEAR_DIST:
                self.set_state("AT_HOME")
            else:
                self.drive_straight(cmd, speed=self.CROSS_SPEED)

        # AT_HOME
        elif self.state == "AT_HOME":
            self.stop()
            self.log_summary()
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
