import cv2
import numpy as np
import json
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Int32, String
from cv_bridge import CvBridge


class NutDetector(Node):
    """
    Macadamia Nut Detector v4 — ROS 2 Node
    ───────────────────────────────────────
    Fixes over v2:
      - Much wider HSV ranges to catch nuts at all distances / lighting angles
      - Multi-scale pyramid detection (catches far & near nuts in same frame)
      - Relaxed shape filters (far nuts look smaller & less circular)
      - Reliable QoS profile to prevent publisher context crash
      - Per-lane nut accumulator — counts unique nuts, not re-detections
      - Overlap suppression (NMS) to avoid double-counting one nut
      - Configurable via ROS 2 params — tune without restarting

    Publishes:
      /nut_count      (Int32)  — smoothed nut count this frame
      /nut_detections (String) — JSON array of bounding boxes + confidence
      /nut_image      (Image)  — annotated camera feed for RViz

    Tune live: ros2 param set /nut_detector min_area 100
    """

    def __init__(self):
        super().__init__("nut_detector")

        # ── Parameters ─────────────────────────────────────────────────────────
        self.declare_parameter("min_area",           80)    # lowered from 200
        self.declare_parameter("max_area",        40000)
        self.declare_parameter("min_circularity",   0.30)   # relaxed from 0.45
        self.declare_parameter("min_solidity",      0.60)   # relaxed from 0.75
        self.declare_parameter("min_aspect_ratio",  0.40)   # relaxed
        self.declare_parameter("max_aspect_ratio",  2.20)   # relaxed
        self.declare_parameter("smoothing_window",  8)
        self.declare_parameter("nms_iou_thresh",    0.35)
        self.declare_parameter("scales",            [0.5, 1.0, 1.5])
        self.declare_parameter("total_expected",    12)     # for % reporting

        self._load_params()

        # ── QoS — reliable prevents the "publisher context invalid" crash ──────
        reliable_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.bridge = CvBridge()

        self.image_sub = self.create_subscription(
            Image,
            "/oak/rgb/image_raw",
            self.image_callback,
            qos_profile=10,
        )

        self.count_pub = self.create_publisher(Int32,  "/nut_count",      reliable_qos)
        self.det_pub   = self.create_publisher(String, "/nut_detections", reliable_qos)
        self.image_pub = self.create_publisher(Image,  "/nut_image",      reliable_qos)

        # ── State ───────────────────────────────────────────────────────────────
        self.recent_counts:  deque = deque(maxlen=self.smoothing_window)
        self.unique_nut_positions: list  = []   # [(cx, cy)] across frames
        self.total_unique_nuts = 0
        self.frame_count = 0

        self.get_logger().info(
            "NutDetector v4 started | "
            f"expected nuts={self.total_expected} | "
            "subscribing /oak/rgb/image_raw"
        )

    # ───────────────────────────────────────────────────────────────────────────
    def _load_params(self):
        self.min_area          = self.get_parameter("min_area").value
        self.max_area          = self.get_parameter("max_area").value
        self.min_circularity   = self.get_parameter("min_circularity").value
        self.min_solidity      = self.get_parameter("min_solidity").value
        self.min_aspect_ratio  = self.get_parameter("min_aspect_ratio").value
        self.max_aspect_ratio  = self.get_parameter("max_aspect_ratio").value
        self.smoothing_window  = self.get_parameter("smoothing_window").value
        self.nms_iou_thresh    = self.get_parameter("nms_iou_thresh").value
        self.scales            = self.get_parameter("scales").value
        self.total_expected    = self.get_parameter("total_expected").value

    # ───────────────────────────────────────────────────────────────────────────
    def _build_mask(self, hsv: np.ndarray) -> np.ndarray:
        """
        Wide HSV coverage for real macadamia nuts + red/radish paper proxies.
        Three layers:
          1. Red/radish paper (hue wraps around 0/180)
          2. Tan/beige — raw macadamia shell colour
          3. Dark brown — roasted / shadowed nut
        """
        ranges = [
            # Red proxy markers (low hue)
            (np.array([0,   60,  60]),  np.array([12,  255, 255])),
            # Red proxy markers (high hue wrap)
            (np.array([155, 60,  60]),  np.array([180, 255, 255])),
            # Tan / light brown (raw macadamia)
            (np.array([8,   30,  100]), np.array([25,  200, 255])),
            # Medium brown
            (np.array([5,   50,  50]),  np.array([22,  210, 200])),
            # Dark brown (shadowed nuts or far away)
            (np.array([3,   60,  25]),  np.array([18,  255, 130])),
        ]

        combined = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in ranges:
            combined = cv2.bitwise_or(combined, cv2.inRange(hsv, lo, hi))

        # Open removes speckles; close fills holes in nut body
        k3 = np.ones((3, 3), np.uint8)
        k7 = np.ones((7, 7), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  k3, iterations=2)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, k7, iterations=2)
        return combined

    # ───────────────────────────────────────────────────────────────────────────
    def _shape_metrics(self, contour):
        area      = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circ      = (4 * np.pi * area / perimeter ** 2) if perimeter > 0 else 0.0
        x, y, w, h = cv2.boundingRect(contour)
        aspect    = (w / h) if h > 0 else 0.0
        hull      = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity  = (area / hull_area) if hull_area > 0 else 0.0
        return area, circ, aspect, solidity, x, y, w, h

    # ───────────────────────────────────────────────────────────────────────────
    def _confidence(self, area, circ, solidity) -> float:
        mid        = (self.min_area + self.max_area) / 2
        area_score = max(0.0, 1 - abs(area - mid) / mid) * 33
        circ_score = min(circ,     1.0) * 34
        sol_score  = min(solidity, 1.0) * 33
        return area_score + circ_score + sol_score

    # ───────────────────────────────────────────────────────────────────────────
    def _detect_on_frame(self, frame: np.ndarray) -> list:
        """
        Run detection at multiple scales and return all candidate boxes.
        Returns list of dicts: {x, y, w, h, score, cx, cy}
        """
        candidates = []

        for scale in self.scales:
            if scale != 1.0:
                h, w   = frame.shape[:2]
                resized = cv2.resize(frame, (int(w * scale), int(h * scale)),
                                     interpolation=cv2.INTER_LINEAR)
            else:
                resized = frame

            blurred = cv2.GaussianBlur(resized, (5, 5), 0)
            hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
            mask    = self._build_mask(hsv)

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for cnt in contours:
                area, circ, aspect, solidity, bx, by, bw, bh = \
                    self._shape_metrics(cnt)

                # Scale area back to original for filtering
                real_area = area / (scale ** 2)
                if not (self.min_area < real_area < self.max_area):
                    continue
                if circ     < self.min_circularity:  continue
                if solidity < self.min_solidity:      continue
                if not (self.min_aspect_ratio < aspect < self.max_aspect_ratio):
                    continue

                # Scale box back to original coords
                ox = int(bx / scale)
                oy = int(by / scale)
                ow = int(bw / scale)
                oh = int(bh / scale)

                score = self._confidence(real_area, circ, solidity)
                candidates.append({
                    "x": ox, "y": oy, "w": ow, "h": oh,
                    "cx": ox + ow // 2, "cy": oy + oh // 2,
                    "score": score, "circ": circ, "solidity": solidity,
                })

        return candidates

    # ───────────────────────────────────────────────────────────────────────────
    def _nms(self, candidates: list) -> list:
        """Non-maximum suppression to remove duplicate detections."""
        if not candidates:
            return []

        candidates.sort(key=lambda d: d["score"], reverse=True)
        kept = []

        for cand in candidates:
            x1, y1 = cand["x"], cand["y"]
            x2, y2 = x1 + cand["w"], y1 + cand["h"]
            area_c = cand["w"] * cand["h"]
            duplicate = False

            for kept_d in kept:
                kx1, ky1 = kept_d["x"], kept_d["y"]
                kx2, ky2 = kx1 + kept_d["w"], ky1 + kept_d["h"]
                area_k   = kept_d["w"] * kept_d["h"]

                ix1, iy1 = max(x1, kx1), max(y1, ky1)
                ix2, iy2 = min(x2, kx2), min(y2, ky2)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                iou   = inter / (area_c + area_k - inter + 1e-6)

                if iou > self.nms_iou_thresh:
                    duplicate = True
                    break

            if not duplicate:
                kept.append(cand)

        return kept

    # ───────────────────────────────────────────────────────────────────────────
    def _update_unique_count(self, detections: list) -> None:
        """
        Track unique nut positions across frames.
        A new detection is 'new' if its centroid is >50px from all known nuts.
        """
        POSITION_THRESHOLD = 50  # pixels

        for det in detections:
            cx, cy = det["cx"], det["cy"]
            is_new = all(
                np.hypot(cx - px, cy - py) > POSITION_THRESHOLD
                for px, py in self.unique_nut_positions
            )
            if is_new:
                self.unique_nut_positions.append((cx, cy))
                self.total_unique_nuts += 1

    # ───────────────────────────────────────────────────────────────────────────
    def _draw(self, frame: np.ndarray, detections: list) -> np.ndarray:
        out = frame.copy()

        for det in detections:
            x, y, w, h = det["x"], det["y"], det["w"], det["h"]
            cx, cy     = det["cx"], det["cy"]
            score      = det["score"]

            if score >= 75:
                colour = (0, 255, 0)      # green  — high confidence
            elif score >= 50:
                colour = (0, 200, 255)    # yellow — medium
            else:
                colour = (0, 100, 255)    # orange — low

            cv2.rectangle(out, (x, y), (x + w, y + h), colour, 2)
            cv2.drawMarker(out, (cx, cy), colour,
                           cv2.MARKER_CROSS, markerSize=12, thickness=1)

            label = f"Nut {score:.0f}%"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(out, (x, y - th - 8), (x + tw + 4, y), colour, -1)
            cv2.putText(out, label, (x + 2, y - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # HUD
        n         = len(detections)
        pct       = (self.total_unique_nuts / self.total_expected * 100
                     if self.total_expected else 0)
        hud_lines = [
            f"Frame nuts   : {n}",
            f"Unique total : {self.total_unique_nuts}/{self.total_expected}",
            f"Coverage     : {pct:.0f}%",
            f"Frame        : {self.frame_count}",
        ]
        cv2.rectangle(out, (0, 0), (270, 20 + 25 * len(hud_lines)),
                      (20, 20, 20), -1)
        for i, line in enumerate(hud_lines):
            cv2.putText(out, line, (8, 20 + 25 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 1)

        return out

    # ───────────────────────────────────────────────────────────────────────────
    def image_callback(self, msg: Image) -> None:
        self.frame_count += 1
        self._load_params()

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")
            return

        # ── Detect ─────────────────────────────────────────────────────────────
        candidates = self._detect_on_frame(frame)
        detections = self._nms(candidates)

        # ── Track unique nuts across the run ───────────────────────────────────
        self._update_unique_count(detections)

        # ── Smoothed count ─────────────────────────────────────────────────────
        self.recent_counts.append(len(detections))
        stable = round(sum(self.recent_counts) / len(self.recent_counts))

        # ── Logging ────────────────────────────────────────────────────────────
        if self.frame_count % 30 == 0:
            pct = (self.total_unique_nuts / self.total_expected * 100
                   if self.total_expected else 0)
            self.get_logger().info(
                f"Frame {self.frame_count} | "
                f"this_frame={len(detections)} stable={stable} | "
                f"unique={self.total_unique_nuts}/{self.total_expected} "
                f"({pct:.0f}%)"
            )

        # ── Publish — wrapped in try/except to survive context errors ──────────
        try:
            count_msg      = Int32()
            count_msg.data = stable
            self.count_pub.publish(count_msg)

            det_payload = [
                {"x": d["x"], "y": d["y"],
                 "w": d["w"], "h": d["h"],
                 "cx": d["cx"], "cy": d["cy"],
                 "score": round(d["score"], 1)}
                for d in detections
            ]
            det_msg      = String()
            det_msg.data = json.dumps(det_payload)
            self.det_pub.publish(det_msg)

            annotated = self._draw(frame, detections)
            self.image_pub.publish(
                self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8"))

        except Exception as e:
            self.get_logger().warn(f"Publish error (skipping frame): {e}")


# ──────────────────────────────────────────────────────────────────────────────
def main(args=None):
    rclpy.init(args=args)
    node = NutDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pct = (node.total_unique_nuts / node.total_expected * 100
               if node.total_expected else 0)
        node.get_logger().info(
            f"Shutdown | unique nuts found: "
            f"{node.total_unique_nuts}/{node.total_expected} ({pct:.0f}%)"
        )
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()