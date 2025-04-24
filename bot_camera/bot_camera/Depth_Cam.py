#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class DualDepthObstacleDetector(Node):
    def __init__(self):
        super().__init__('dual_depth_obstacle_detector')
        
        # Subscribers for top and bottom depth cameras
        self.depth_sub_top = self.create_subscription(
            Image,
            '/camera4/depth/image_raw',  # Topic for top depth camera
            self.top_depth_callback,
            10)

        self.depth_sub_bottom = self.create_subscription(
            Image,
            '/camera3/depth/image_raw',  # Topic for bottom depth camera
            self.bottom_depth_callback,
            10)

        self.bridge = CvBridge()
        self.top_depth_image = None
        self.bottom_depth_image = None

    def top_depth_callback(self, msg):
        # Convert ROS Image message to OpenCV format for top depth camera
        self.top_depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")

    def bottom_depth_callback(self, msg):
        # Convert ROS Image message to OpenCV format for bottom depth camera
        self.bottom_depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")

        if self.top_depth_image is not None and self.bottom_depth_image is not None:
            self.compare_depth_images()

    def compare_depth_images(self):
        # Ensure that both images have the same size
        if self.top_depth_image.shape != self.bottom_depth_image.shape:
            self.get_logger().error('Depth images from top and bottom cameras have different sizes!')
            return

        # Example: Take the average depth of the images
        avg_depth_top = np.mean(self.top_depth_image)
        avg_depth_bottom = np.mean(self.bottom_depth_image)

        self.get_logger().info(f"Average Depth - Top: {avg_depth_top}, Bottom: {avg_depth_bottom}")

        # Define a threshold to detect a significant difference
        depth_difference_threshold = 0.5  # Meters

        # Detect if there's a significant drop
        if (avg_depth_bottom - avg_depth_top) > depth_difference_threshold:
            self.get_logger().info('Negative obstacle detected! (Significant depth difference)')
        else:
            self.get_logger().info('No significant negative obstacle detected.')

        # Optionally, you can visualize the depth images for debugging
        depth_top_normalized = cv2.normalize(self.top_depth_image, None, 0, 255, cv2.NORM_MINMAX)
        depth_bottom_normalized = cv2.normalize(self.bottom_depth_image, None, 0, 255, cv2.NORM_MINMAX)

        depth_top_normalized = np.uint8(depth_top_normalized)
        depth_bottom_normalized = np.uint8(depth_bottom_normalized)

        cv2.imshow('Top Depth Camera', depth_top_normalized)
        cv2.imshow('Bottom Depth Camera', depth_bottom_normalized)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = DualDepthObstacleDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
