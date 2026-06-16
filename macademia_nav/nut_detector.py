import cv2
import numpy as np
import json
import math
from collections import deque

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32, String
from nav_msgs.msg import Odometry
from visualization_msgs.msg import MarkerArray, Marker
from cv_bridge import CvBridge


def quat_to_yaw(x, y, z, w):
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class TrackedNut:
    """A nut that has been detected and localised in the odom frame."""
    def __init__(self, x, y, nut_id):
        self.x    = x
        self.y    = y
        self.id   = nut_id
        self.hits = 1


class NutDetector(Node):
    """
    Detects simulated macadamia nuts (red/orange paper circles).

    Key features:
      - Spatial deduplication — each nut counted only once
      - Odometry-based ground position estimate
      - RViz MarkerArray showing nut locations
      - Persistent nut registry across entire mission

    Publishes:
      /nut_count      (Int32)        — unique nuts found so far
      /nut_detections (String)       — JSON with all known nuts
      /nut_image      (Image)        — annotated camera feed
      /nuts/markers   (MarkerArray)  — RViz spheres at nut positions
    """

    # ── Camera geometry ────────────────────────────────────────────── #
    CAMERA_HEIGHT    = 0.20    # metres above ground
    CAMERA_TILT_DEG  = 15.0   # degrees downward from horizontal
    CAMERA_FOV_H_DEG = 69.0   # OAK-D horizontal FOV
    IMAGE_WIDTH      = 1280
    IMAGE_HEIGHT     = 800

    # ── Deduplication ─────────────────────────────────────────────── #
    DEDUP_RADIUS     = 0.25   # metres — same nut if within this distance

    def __init__(self):
        super().__init__("nut_detector")

        # ── Parameters ─────────────────────────────────────────────── #
        self.declare_parameter("min_area",          50)
        self.declare_parameter("max_area",         8000)
        self.declare_parameter("min_circularity",   0.25)
        self.declare_parameter("min_aspect_ratio",  0.5)
        self.declare_parameter("max_aspect_ratio",  2.0)
        self.declare_parameter("smoothing_window",  3)
        self._refresh_params()

        # ── Bridge ─────────────────────────────────────────────────── #
        self.bridge = CvBridge()

        # ── Subscribers ────────────────────────────────────────────── #
        self.image_sub = self.create_subscription(
            Image, "/oak/rgb/image_raw",
            self.image_callback, 10)
        self.odom_sub  = self.create_subscription(
            Odometry, "/odometry/filtered",
            self.odom_callback, 10)

        # ── Publishers ─────────────────────────────────────────────── #
        self.count_pub  = self.create_publisher(Int32,       "/nut_count",      10)
        self.det_pub    = self.create_publisher(String,      "/nut_detections", 10)
        self.image_pub  = self.create_publisher(Image,       "/nut_image",      10)
        self.marker_pub = self.create_publisher(MarkerArray, "/nuts/markers",   10)

        # ── State ──────────────────────────────────────────────────── #
        self.recent_counts = deque(maxlen=self.smoothing_window)
        self.frame_count   = 0

        self.robot_x    = 0.0
        self.robot_y    = 0.0
        self.robot_yaw  = 0.0
        self.odom_ready = False

        self.known_nuts  = []
        self.next_nut_id = 1

        self.get_logger().info(
            "NutDetector v3 started — "
            "with spatial deduplication and RViz markers.")

    # ================================================================== #

    def _refresh_params(self):
        self.min_area         = self.get_parameter("min_area").value
        self.max_area         = self.get_parameter("max_area").value
        self.min_circularity  = self.get_parameter("min_circularity").value
        self.min_aspect_ratio = self.get_parameter("min_aspect_ratio").value
        self.max_aspect_ratio = self.get_parameter("max_aspect_ratio").value
        self.smoothing_window = self.get_parameter("smoothing_window").value

    def odom_callback(self, msg):
        p = msg.pose.pose
        self.robot_x   = p.position.x
        self.robot_y   = p.position.y
        self.robot_yaw = quat_to_yaw(
            p.orientation.x, p.orientation.y,
            p.orientation.z, p.orientation.w)
        self.odom_ready = True

    def _build_red_mask(self, hsv):
        mask = cv2.bitwise_or(
            cv2.inRange(hsv, np.array([0,    80, 60]),
                             np.array([12,  255, 255])),
            cv2.inRange(hsv, np.array([158,  80, 60]),
                             np.array([180, 255, 255])),
        )
        kernel = np.ones((5, 5), np.uint8)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def _circularity(self, contour):
        perimeter = cv2.arcLength(contour, True)
        area      = cv2.contourArea(contour)
        return (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0

    def pixel_to_ground(self, cx, cy):
        """
        Estimate ground position (odom frame) from pixel coordinates.
        Returns (world_x, world_y).
        """
        cy_norm = (cy - self.IMAGE_HEIGHT / 2) / (self.IMAGE_HEIGHT / 2)
        tilt_rad = math.radians(self.CAMERA_TILT_DEG)
        angle = tilt_rad + cy_norm * math.radians(20)

        # Clamp angle to prevent division by zero or negative distances
        # tan(angle) must be positive and non-zero
        angle = max(math.radians(2.0), angle)

        tan_val = math.tan(angle)

        # Extra safety — should never be zero after clamp but just in case
        if abs(tan_val) < 1e-6:
            tan_val = 1e-6

        forward_dist = self.CAMERA_HEIGHT / tan_val
        forward_dist = max(0.1, min(forward_dist, 1.5))  # clamp 0.1-1.5m

        cx_norm      = (cx - self.IMAGE_WIDTH / 2) / (self.IMAGE_WIDTH / 2)
        fov_rad      = math.radians(self.CAMERA_FOV_H_DEG / 2)
        lateral_dist = forward_dist * math.tan(fov_rad * cx_norm)

        world_x = (self.robot_x
                   + forward_dist * math.cos(self.robot_yaw)
                   - lateral_dist * math.sin(self.robot_yaw))
        world_y = (self.robot_y
                   + forward_dist * math.sin(self.robot_yaw)
                   + lateral_dist * math.cos(self.robot_yaw))

        return world_x, world_y

    def find_known_nut(self, world_x, world_y):
        for nut in self.known_nuts:
            dist = math.hypot(world_x - nut.x, world_y - nut.y)
            if dist < self.DEDUP_RADIUS:
                return nut
        return None

    def publish_markers(self):
        marker_array = MarkerArray()

        # Clear old markers first
        clear_marker              = Marker()
        clear_marker.action       = Marker.DELETEALL
        clear_marker.header.frame_id = "odom"
        clear_marker.header.stamp = self.get_clock().now().to_msg()
        marker_array.markers.append(clear_marker)

        for nut in self.known_nuts:
            # Sphere
            m                    = Marker()
            m.header.frame_id    = "odom"
            m.header.stamp       = self.get_clock().now().to_msg()
            m.ns                 = "nuts"
            m.id                 = nut.id
            m.type               = Marker.SPHERE
            m.action             = Marker.ADD
            m.pose.position.x    = nut.x
            m.pose.position.y    = nut.y
            m.pose.position.z    = 0.05
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.12
            m.color.r = 1.0
            m.color.g = 0.5
            m.color.b = 0.0
            m.color.a = 0.9
            m.lifetime.sec = 0
            marker_array.markers.append(m)

            # Label
            t                    = Marker()
            t.header.frame_id    = "odom"
            t.header.stamp       = self.get_clock().now().to_msg()
            t.ns                 = "nuts_label"
            t.id                 = nut.id + 10000
            t.type               = Marker.TEXT_VIEW_FACING
            t.action             = Marker.ADD
            t.pose.position.x    = nut.x
            t.pose.position.y    = nut.y
            t.pose.position.z    = 0.20
            t.pose.orientation.w = 1.0
            t.scale.z            = 0.08
            t.color.r = t.color.g = t.color.b = t.color.a = 1.0
            t.text               = f"Nut {nut.id}"
            t.lifetime.sec       = 0
            marker_array.markers.append(t)

        self.marker_pub.publish(marker_array)

    # ================================================================== #

    def image_callback(self, msg):
        self.frame_count += 1
        self._refresh_params()

        frame  = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        h, w   = frame.shape[:2]
        self.IMAGE_WIDTH  = w
        self.IMAGE_HEIGHT = h

        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask    = self._build_red_mask(hsv)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        new_nuts_this_frame = 0
        detections          = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if not (self.min_area < area < self.max_area):
                continue

            x, y, bw, bh = cv2.boundingRect(contour)
            aspect_ratio  = (bw / bh) if bh > 0 else 0
            circularity   = self._circularity(contour)

            if not (self.min_aspect_ratio < aspect_ratio < self.max_aspect_ratio):
                continue
            if circularity < self.min_circularity:
                continue

            cx = x + bw // 2
            cy = y + bh // 2

            is_new   = False
            world_x  = None
            world_y  = None

            if self.odom_ready:
                try:
                    world_x, world_y = self.pixel_to_ground(cx, cy)
                    existing = self.find_known_nut(world_x, world_y)

                    if existing is None:
                        nut = TrackedNut(world_x, world_y, self.next_nut_id)
                        self.known_nuts.append(nut)
                        self.next_nut_id    += 1
                        new_nuts_this_frame += 1
                        is_new = True
                        self.get_logger().info(
                            f"NEW NUT #{nut.id} at "
                            f"({world_x:.2f}, {world_y:.2f}) "
                            f"total={len(self.known_nuts)}")
                    else:
                        existing.hits += 1
                except Exception as e:
                    self.get_logger().warn(f"pixel_to_ground error: {e}")
                    continue

            # Annotate
            colour = (0, 255, 0) if is_new else (0, 165, 255)
            label  = f"NEW c={circularity:.2f}" if is_new else f"Known c={circularity:.2f}"
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), colour, 2)
            cv2.circle(frame, (cx, cy), 4, colour, -1)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 2)

            detections.append({
                "cx": cx, "cy": cy,
                "area": int(area),
                "circularity": round(circularity, 3),
                "is_new": is_new,
                "world_x": round(world_x, 3) if world_x is not None else None,
                "world_y": round(world_y, 3) if world_y is not None else None,
            })

        # Smoothing
        self.recent_counts.append(len(detections))
        stable_count = round(
            sum(self.recent_counts) / len(self.recent_counts))
        total_unique = len(self.known_nuts)

        # HUD
        cv2.putText(frame, f"Detected (frame): {len(detections)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Unique nuts total: {total_unique}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"New this frame:   {new_nuts_this_frame}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

        if self.frame_count % 30 == 0:
            self.get_logger().info(
                f"Frame {self.frame_count} | "
                f"in_frame={len(detections)} "
                f"unique_total={total_unique} "
                f"new_this_frame={new_nuts_this_frame}")

        # Publish
        count_msg      = Int32()
        count_msg.data = total_unique
        self.count_pub.publish(count_msg)

        det_msg      = String()
        det_msg.data = json.dumps({
            "frame":        self.frame_count,
            "in_frame":     len(detections),
            "unique_total": total_unique,
            "known_nuts": [
                {"id": n.id, "x": round(n.x, 3),
                 "y": round(n.y, 3), "hits": n.hits}
                for n in self.known_nuts
            ],
            "detections": detections,
        })
        self.det_pub.publish(det_msg)

        self.image_pub.publish(
            self.bridge.cv2_to_imgmsg(frame, encoding="bgr8"))

        if new_nuts_this_frame > 0:
            self.publish_markers()


def main(args=None):
    rclpy.init(args=args)
    node = NutDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info(
            f"Shutting down — unique nuts: {len(node.known_nuts)}")
        for nut in node.known_nuts:
            node.get_logger().info(
                f"  Nut #{nut.id}: ({nut.x:.2f}, {nut.y:.2f}) "
                f"seen {nut.hits} times")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
