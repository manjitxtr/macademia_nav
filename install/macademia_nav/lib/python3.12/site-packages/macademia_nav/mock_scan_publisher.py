import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


class MockScanPublisher(Node):
    def __init__(self):
        super().__init__("mock_scan_publisher")

        self.scan_pub = self.create_publisher(LaserScan, "/scan", 10)
        self.timer = self.create_timer(0.5, self.publish_scan)

        self.counter = 0
        self.get_logger().info("Mock orchard scan publisher started.")

    def publish_scan(self):
        msg = LaserScan()

        msg.header.frame_id = "laser"
        msg.angle_min = -3.14
        msg.angle_max = 3.14
        msg.angle_increment = 6.28 / 360
        msg.range_min = 0.1
        msg.range_max = 10.0

        ranges = [4.0] * 360
        self.counter += 1

        # Stage 1: robot is inside a clear orchard lane
        # Left and right tree rows are visible, front is open
        if self.counter < 20:
            self.set_zone(ranges, 60, 120, 0.9)     # left tree row
            self.set_zone(ranges, 240, 300, 0.9)    # right tree row
            self.set_front(ranges, 3.0)             # front open

        # Stage 2: robot is slightly closer to left trees
        # It should correct slightly to the right
        elif self.counter < 40:
            self.set_zone(ranges, 60, 120, 0.55)
            self.set_zone(ranges, 240, 300, 1.2)
            self.set_front(ranges, 3.0)

        # Stage 3: robot is slightly closer to right trees
        # It should correct slightly to the left
        elif self.counter < 60:
            self.set_zone(ranges, 60, 120, 1.2)
            self.set_zone(ranges, 240, 300, 0.55)
            self.set_front(ranges, 3.0)

        # Stage 4: row end detected
        # Front becomes blocked, left is open, so robot should turn left
        elif self.counter < 85:
            self.set_zone(ranges, 60, 120, 2.5)
            self.set_zone(ranges, 240, 300, 0.8)
            self.set_front(ranges, 0.3)

        # Stage 5: new lane after turning
        elif self.counter < 110:
            self.set_zone(ranges, 60, 120, 0.9)
            self.set_zone(ranges, 240, 300, 0.9)
            self.set_front(ranges, 3.0)

        # Stage 6: another row end, right is open
        elif self.counter < 135:
            self.set_zone(ranges, 60, 120, 0.8)
            self.set_zone(ranges, 240, 300, 2.5)
            self.set_front(ranges, 0.3)

        # Stage 7: continue searching
        else:
            self.set_zone(ranges, 60, 120, 0.9)
            self.set_zone(ranges, 240, 300, 0.9)
            self.set_front(ranges, 3.0)

        msg.ranges = ranges
        self.scan_pub.publish(msg)

    def set_front(self, ranges, distance):
        for i in range(20):
            ranges[i] = distance
            ranges[-(i + 1)] = distance

    def set_zone(self, ranges, start, end, distance):
        for i in range(start, end):
            ranges[i] = distance


def main(args=None):
    rclpy.init(args=args)
    node = MockScanPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()

    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()