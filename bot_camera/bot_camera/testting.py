#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from message_filters import ApproximateTimeSynchronizer, Subscriber

class NegativeObstacleDetector(Node):
    def __init__(self):
        super().__init__('negative_obstacle_detector')

        # Create subscribers for both cameras
        self.top_camera_sub = Subscriber(self, Image, '/camera3/image_raw')
        self.bottom_camera_sub = Subscriber(self, Image, '/camera4/image_raw')

        # Synchronize the two camera feeds with an approximate time sync policy
        self.ts = ApproximateTimeSynchronizer([self.top_camera_sub, self.bottom_camera_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.image_callback)

        self.bridge = CvBridge()

    def image_callback(self, top_img_msg, bottom_img_msg):
        # Convert ROS Image messages to OpenCV format
        top_frame = self.bridge.imgmsg_to_cv2(top_img_msg, desired_encoding='bgr8')
        bottom_frame = self.bridge.imgmsg_to_cv2(bottom_img_msg, desired_encoding='bgr8')

        # Preprocess the images (convert to grayscale, blur, etc.)
        top_blurred, bottom_blurred = self.preprocess_images(top_frame, bottom_frame)

        # Detect edges
        top_edges, bottom_edges = self.detect_edges(top_blurred, bottom_blurred)

        # Find contours
        top_contours, bottom_contours = self.find_contours(top_edges, bottom_edges)

        # Display contours for debugging
        self.display_contours(top_frame, top_contours, bottom_frame, bottom_contours)

        # Calculate the difference between the two grayscale images
        diff_score = self.calculate_difference(top_blurred, bottom_blurred)

        self.get_logger().info(f'Difference score: {diff_score}')

        # Decision-making based on contours and image difference
        if self.is_negative_obstacle(top_contours, bottom_contours, diff_score):
            self.get_logger().info('Negative obstacle detected!')
        else:
            self.get_logger().info('No obstacle detected.')

        cv2.waitKey(1)

    def preprocess_images(self, top_frame, bottom_frame):
        """Convert images to grayscale and apply Gaussian blur."""
        top_gray = cv2.cvtColor(top_frame, cv2.COLOR_BGR2GRAY)
        bottom_gray = cv2.cvtColor(bottom_frame, cv2.COLOR_BGR2GRAY)
        
        top_blurred = cv2.GaussianBlur(top_gray, (5, 5), 0)
        bottom_blurred = cv2.GaussianBlur(bottom_gray, (5, 5), 0)
        
        return top_blurred, bottom_blurred

    def detect_edges(self, top_blurred, bottom_blurred):
        """Detect edges using Canny edge detector."""
        top_edges = cv2.Canny(top_blurred, 50, 150)
        bottom_edges = cv2.Canny(bottom_blurred, 50, 150)
        
        return top_edges, bottom_edges

    def find_contours(self, top_edges, bottom_edges):
        """Find contours from the edge-detected images."""
        top_contours, _ = cv2.findContours(top_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        bottom_contours, _ = cv2.findContours(bottom_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        return top_contours, bottom_contours

    def display_contours(self, top_frame, top_contours, bottom_frame, bottom_contours):
        """Display the contours on both camera images for debugging."""
        top_contour_image = cv2.drawContours(top_frame.copy(), top_contours, -1, (0, 255, 0), 3)
        bottom_contour_image = cv2.drawContours(bottom_frame.copy(), bottom_contours, -1, (0, 255, 0), 3)

        cv2.imshow('Top Camera Contours', top_contour_image)
        cv2.imshow('Bottom Camera Contours', bottom_contour_image)

    def calculate_difference(self, top_gray, bottom_gray):
        """Calculate the pixel-wise absolute difference between the two grayscale images."""
        difference = cv2.absdiff(top_gray, bottom_gray)
        diff_score = np.mean(difference)
        return diff_score

    def is_negative_obstacle(self, top_contours, bottom_contours, diff_score, diff_threshold=20):
        """Logic for detecting negative obstacles using contour and difference analysis."""
        # Significant difference between the two images
        if diff_score > diff_threshold:
            self.get_logger().info(f"Significant difference detected: {diff_score}")
            return True

        # Further contour analysis can be added here for more accurate detection

        return False

def main(args=None):
    rclpy.init(args=args)
    node = NegativeObstacleDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
