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
            '/camera4/depth/image_raw',  # Topic for bottom depth camera
            self.bottom_depth_callback,
            10)

        self.bridge = CvBridge()
        self.top_depth_image = None
        self.bottom_depth_image = None

        # YOLO setup
        self.net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        self.classes = ['hole', 'other_class']  # Classes to detect, you will train this for 'hole'

    def top_depth_callback(self, msg):
        self.top_depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")

    def bottom_depth_callback(self, msg):
        self.bottom_depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")

        if self.top_depth_image is not None and self.bottom_depth_image is not None:
            # If both depth images are available, compare them and detect obstacles
            self.compare_depth_images()

    def detect_obstacle(self, frame):
        height, width, channels = frame.shape
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)

        boxes = []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:  # Set a confidence threshold
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])

        return boxes

    def compare_depth_images(self):
        # Convert ROS Image to OpenCV image for YOLO detection
        top_frame = self.top_depth_image.copy()
        bottom_frame = self.bottom_depth_image.copy()

        # Detect obstacles using YOLO on top image (depth-based region)
        boxes = self.detect_obstacle(top_frame)

        # Check for depth differences in detected regions
        for box in boxes:
            x, y, w, h = box
            roi_top = self.top_depth_image[y:y+h, x:x+w]
            roi_bottom = self.bottom_depth_image[y:y+h, x:x+w]

            avg_depth_top = np.mean(roi_top)
            avg_depth_bottom = np.mean(roi_bottom)

            self.get_logger().info(f"Detected Obstacle. Avg Depth - Top: {avg_depth_top}, Bottom: {avg_depth_bottom}")

            depth_difference_threshold = 0.5  # meters
            if (avg_depth_bottom - avg_depth_top) > depth_difference_threshold:
                self.get_logger().info('Negative obstacle (hole) detected!')
            else:
                self.get_logger().info('No negative obstacle detected.')

            # Visualize for debugging
            roi_top_normalized = cv2.normalize(roi_top, None, 0, 255, cv2.NORM_MINMAX)
            roi_bottom_normalized = cv2.normalize(roi_bottom, None, 0, 255, cv2.NORM_MINMAX)

            cv2.imshow('Top Depth ROI', np.uint8(roi_top_normalized))
            cv2.imshow('Bottom Depth ROI', np.uint8(roi_bottom_normalized))
            cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = DualDepthObstacleDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
