import cv2
import numpy as np
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D
from geometry_msgs.msg import Pose2D
from cv_bridge import CvBridge


class NutDetector(Node):
    """
    Detects simulated macadamia nuts (red/radish paper circles) using
    HSV colour segmentation on the OAK-D RGB camera.

    Improvements over v1:
      - Circularity filtering (not just display colour)
      - Aspect-ratio filtering to reject elongated blobs
      - Gaussian blur before HSV conversion for cleaner masks
      - Wider / lower-threshold HSV ranges for variable lighting
      - Temporal smoothing via a rolling window (replaces lifetime-max)
      - Publishes Detection2DArray with bounding boxes + centroids
      - All thresholds exposed as ROS 2 parameters (tune without restart)
    """

    def __init__(self):
        super().__init__("nut_detector")

        # ── ROS 2 parameters (tune with: ros2 param set /nut_detector <name> <val>) ──
        self.declare_parameter("min_area",         200)
        self.declare_parameter("max_area",        8000)
        self.declare_parameter("min_circularity",  0.45)
        self.declare_parameter("min_aspect_ratio", 0.6)
        self.declare_parameter("max_aspect_ratio", 1.6)
        self.declare_parameter("smoothing_window", 10)

        self._refresh_params()

        # ── Bridge & subscriptions ──────────────────────────────────────────────────
        self.bridge = CvBridge()

        self.image_sub = self.create_subscription(
            Image,
            "/oak/rgb/image_raw",
            self.image_callback,
            10,
        )

        # ── Publishers ──────────────────────────────────────────────────────────────
        self.count_pub      = self.create_publisher(Int32,           "/nut_count",       10)
        self.det_pub        = self.create_publisher(Detection2DArray, "/nut_detections",  10)
        self.image_pub      = self.create_publisher(Image,            "/nut_image",       10)

        # ── State ───────────────────────────────────────────────────────────────────
        self.recent_counts: deque[int] = deque(maxlen=self.smoothing_window)
        self.frame_count = 0

        self.get_logger().info(
            "NutDetector v2 started — "
            "subscribing to /oak/rgb/image_raw"
        )

    # ────────────────────────────────────────────────────────────────────────────────
    def _refresh_params(self) -> None:
        """Pull current parameter values into instance attributes."""
        self.min_area          = self.get_parameter("min_area").value
        self.max_area          = self.get_parameter("max_area").value
        self.min_circularity   = self.get_parameter("min_circularity").value
        self.min_aspect_ratio  = self.get_parameter("min_aspect_ratio").value
        self.max_aspect_ratio  = self.get_parameter("max_aspect_ratio").value
        self.smoothing_window  = self.get_parameter("smoothing_window").value

    # ────────────────────────────────────────────────────────────────────────────────
    def _build_red_mask(self, hsv: np.ndarray) -> np.ndarray:
        """
        Return a binary mask for red hues.
        Red wraps around in HSV, so two ranges are ORed together.
        Thresholds are deliberately permissive (lower S/V) to handle
        variable lighting on radish paper.
        """
        lower_red1 = np.array([0,    80, 60])
        upper_red1 = np.array([12,  255, 255])
        lower_red2 = np.array([158,  80, 60])
        upper_red2 = np.array([180, 255, 255])

        mask = cv2.bitwise_or(
            cv2.inRange(hsv, lower_red1, upper_red1),
            cv2.inRange(hsv, lower_red2, upper_red2),
        )

        # Morphological cleanup: open removes speckles, close fills holes
        kernel = np.ones((5, 5), np.uint8)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    # ────────────────────────────────────────────────────────────────────────────────
    def _circularity(self, contour) -> float:
        perimeter = cv2.arcLength(contour, True)
        area      = cv2.contourArea(contour)
        return (4 * np.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0

    # ────────────────────────────────────────────────────────────────────────────────
    def image_callback(self, msg: Image) -> None:
        self.frame_count += 1
        self._refresh_params()           # honour live parameter changes

        # ── Pre-processing ──────────────────────────────────────────────────────────
        frame  = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        # Gaussian blur reduces salt-and-pepper noise → cleaner HSV mask
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask    = self._build_red_mask(hsv)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # ── Detection ───────────────────────────────────────────────────────────────
        det_array          = Detection2DArray()
        det_array.header   = msg.header
        nuts_this_frame    = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if not (self.min_area < area < self.max_area):
                continue

            x, y, w, h   = cv2.boundingRect(contour)
            aspect_ratio  = (w / h) if h > 0 else 0
            circularity   = self._circularity(contour)

            # ── Shape filters (the key addition over v1) ──────────────────────────
            if not (self.min_aspect_ratio < aspect_ratio < self.max_aspect_ratio):
                continue                          # elongated blob → not a nut
            if circularity < self.min_circularity:
                continue                          # non-circular blob → not a nut

            nuts_this_frame += 1

            # Annotate frame
            colour = (0, 255, 0)   # green = high confidence
            cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
            cv2.putText(
                frame,
                f"Nut c={circularity:.2f}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                colour,
                2,
            )

            # Build Detection2D message with centroid + bounding box
            det                      = Detection2D()
            det.header               = msg.header
            bbox                     = BoundingBox2D()
            bbox.center              = Pose2D()
            bbox.center.x            = float(x + w / 2)
            bbox.center.y            = float(y + h / 2)
            bbox.size_x              = float(w)
            bbox.size_y              = float(h)
            det.bbox                 = bbox
            det_array.detections.append(det)

        # ── Temporal smoothing ───────────────────────────────────────────────────────
        self.recent_counts.append(nuts_this_frame)
        stable_count = round(sum(self.recent_counts) / len(self.recent_counts))

        # ── HUD overlay ─────────────────────────────────────────────────────────────
        cv2.putText(frame, f"Nuts (raw):    {nuts_this_frame}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Nuts (stable): {stable_count}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # ── Logging (every 30 frames) ────────────────────────────────────────────────
        if self.frame_count % 30 == 0:
            self.get_logger().info(
                f"Frame {self.frame_count} | "
                f"raw={nuts_this_frame} stable={stable_count}"
            )

        # ── Publish ──────────────────────────────────────────────────────────────────
        count_msg      = Int32()
        count_msg.data = stable_count
        self.count_pub.publish(count_msg)
        self.det_pub.publish(det_array)
        self.image_pub.publish(
            self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        )


# ────────────────────────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = NutDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down NutDetector.")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()