import cv2
import numpy as np
import json
from collections import deque

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32, String
from cv_bridge import CvBridge


class NutDetector(Node):
    """
    Detects simulated macadamia nuts (red/radish paper circles) using
    HSV colour segmentation on the OAK-D RGB camera.

    Publishes:
      /nut_count      (Int32)  — temporally smoothed nut count
      /nut_detections (String) — JSON array of bounding boxes + centroids
      /nut_image      (Image)  — annotated camera feed for RViz

    Uses only standard ROS 2 messages — no extra packages required.
    Tune live: ros2 param set /nut_detector min_area 200
    """

    def __init__(self):
        super().__init__("nut_detector")

        # ── ROS 2 parameters ───────────────────────────────────────── #
        self.declare_parameter("min_area",          50)
        self.declare_parameter("max_area",         8000)
        self.declare_parameter("min_circularity",   0.25)
        self.declare_parameter("min_aspect_ratio",  0.6)
        self.declare_parameter("max_aspect_ratio",  1.6)
        self.declare_parameter("smoothing_window",  3)

        self._refresh_params()

        # ── Camera + bridge ────────────────────────────────────────── #
        self.bridge    = CvBridge()
        self.image_sub = self.create_subscription(
            Image, "/oak/rgb/image_raw", self.image_callback, 10)

        # ── Publishers ─────────────────────────────────────────────── #
        self.count_pub = self.create_publisher(Int32,  "/nut_count",      10)
        self.det_pub   = self.create_publisher(String, "/nut_detections", 10)
        self.image_pub = self.create_publisher(Image,  "/nut_image",      10)

        # ── State ──────────────────────────────────────────────────── #
        self.recent_counts: deque = deque(maxlen=self.smoothing_window)
        self.frame_count          = 0
        self.session_max          = 0

        self.get_logger().info(
            "NutDetector started — "
            "subscribing to /oak/rgb/image_raw — "
            "no extra ROS packages required.")

    # ================================================================== #

    def _refresh_params(self):
        self.min_area         = self.get_parameter("min_area").value
        self.max_area         = self.get_parameter("max_area").value
        self.min_circularity  = self.get_parameter("min_circularity").value
        self.min_aspect_ratio = self.get_parameter("min_aspect_ratio").value
        self.max_aspect_ratio = self.get_parameter("max_aspect_ratio").value
        self.smoothing_window = self.get_parameter("smoothing_window").value

    def _build_red_mask(self, hsv):
        """
        Red wraps around in HSV — combine two ranges.
        Permissive S/V thresholds handle variable lab lighting.
        """
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

    # ================================================================== #

    def image_callback(self, msg):
        self.frame_count += 1
        self._refresh_params()

        # Pre-processing
        frame   = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask    = self._build_red_mask(hsv)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        nuts_this_frame = 0
        detections      = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if not (self.min_area < area < self.max_area):
                continue

            x, y, w, h   = cv2.boundingRect(contour)
            aspect_ratio  = (w / h) if h > 0 else 0
            circularity   = self._circularity(contour)

            # Shape filters
            if not (self.min_aspect_ratio < aspect_ratio < self.max_aspect_ratio):
                continue
            if circularity < self.min_circularity:
                continue

            nuts_this_frame += 1

            cx = x + w // 2
            cy = y + h // 2

            # Store detection info
            detections.append({
                "id":          nuts_this_frame,
                "cx":          cx,
                "cy":          cy,
                "width":       w,
                "height":      h,
                "area":        int(area),
                "circularity": round(circularity, 3),
            })

            # Annotate frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
            cv2.putText(
                frame,
                f"Nut {nuts_this_frame} c={circularity:.2f}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (0, 255, 0), 2)

        # Temporal smoothing
        self.recent_counts.append(nuts_this_frame)
        stable_count     = round(
            sum(self.recent_counts) / len(self.recent_counts))
        self.session_max = max(self.session_max, nuts_this_frame)

        # HUD overlay
        cv2.putText(frame, f"Nuts (raw):    {nuts_this_frame}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Nuts (stable): {stable_count}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Session max:   {self.session_max}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 200, 255), 2)

        # Log every 30 frames
        if self.frame_count % 30 == 0:
            self.get_logger().info(
                f"Frame {self.frame_count} | "
                f"raw={nuts_this_frame} "
                f"stable={stable_count} "
                f"session_max={self.session_max}")

        # Publish count
        count_msg      = Int32()
        count_msg.data = stable_count
        self.count_pub.publish(count_msg)

        # Publish detections as JSON string
        det_msg      = String()
        det_msg.data = json.dumps({
            "frame":        self.frame_count,
            "count":        nuts_this_frame,
            "stable_count": stable_count,
            "detections":   detections,
        })
        self.det_pub.publish(det_msg)

        # Publish annotated image
        self.image_pub.publish(
            self.bridge.cv2_to_imgmsg(frame, encoding="bgr8"))


def main(args=None):
    rclpy.init(args=args)
    node = NutDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info(
            f"Shutting down — session max nuts seen: {node.session_max}")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

