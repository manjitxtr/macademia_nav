import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from cv_bridge import CvBridge


class NutDetector(Node):
    """
    Detects simulated macadamia nuts (red/radish paper circles) using
    HSV colour segmentation on the OAK-D RGB camera.
    Red wraps around in HSV so two ranges are combined.
    """

    MIN_CONTOUR_AREA = 200
    MAX_CONTOUR_AREA = 8000

    def __init__(self):
        super().__init__("nut_detector")

        self.bridge = CvBridge()

        self.image_sub = self.create_subscription(
            Image,
            "/oak/rgb/image_raw",
            self.image_callback,
            10
        )

        self.count_pub = self.create_publisher(Int32, "/nut_detections", 10)
        self.image_pub = self.create_publisher(Image, "/nut_image", 10)

        self.total_nuts_seen = 0
        self.frame_count     = 0

        self.get_logger().info(
            "Nut detector started — "
            "red/radish HSV range — "
            "subscribing to /oak/rgb/image_raw")

    def image_callback(self, msg):
        self.frame_count += 1

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Red wraps around in HSV — need two ranges
        lower_red1 = np.array([0,   100, 80])
        upper_red1 = np.array([10,  255, 255])
        lower_red2 = np.array([160, 100, 80])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask  = cv2.bitwise_or(mask1, mask2)

        # Morphological cleanup
        kernel = np.ones((5, 5), np.uint8)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        nuts_this_frame = 0

        for contour in contours:
            area = cv2.contourArea(contour)

            if self.MIN_CONTOUR_AREA < area < self.MAX_CONTOUR_AREA:
                nuts_this_frame += 1
                x, y, w, h = cv2.boundingRect(contour)

                perimeter = cv2.arcLength(contour, True)
                circularity = (4 * np.pi * area / (perimeter ** 2)
                               if perimeter > 0 else 0)

                colour = (0, 255, 0) if circularity > 0.5 else (0, 165, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
                cv2.putText(
                    frame,
                    f"Nut ({circularity:.2f})",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    colour,
                    2
                )

        self.total_nuts_seen = max(self.total_nuts_seen, nuts_this_frame)

        cv2.putText(frame, f"Nuts this frame: {nuts_this_frame}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Max seen: {self.total_nuts_seen}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)

        if self.frame_count % 30 == 0:
            self.get_logger().info(
                f"Frame {self.frame_count} — "
                f"nuts visible: {nuts_this_frame} "
                f"max seen: {self.total_nuts_seen}")

        count_msg      = Int32()
        count_msg.data = nuts_this_frame
        self.count_pub.publish(count_msg)

        self.image_pub.publish(
            self.bridge.cv2_to_imgmsg(frame, encoding="bgr8"))


def main(args=None):
    rclpy.init(args=args)
    node = NutDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info(
            f"Shutting down — max nuts seen: {node.total_nuts_seen}")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
