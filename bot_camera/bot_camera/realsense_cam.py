#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from message_filters import ApproximateTimeSynchronizer, Subscriber
from tensorflow.keras.models import load_model

class NegativeObstacleDetector(Node):
    def __init__(self):
        super().__init__('negative_obstacle_detector')
        
        # Create subscribers for both cameras
        # self.img_sub = self.create_subscription(Image, '/camera/color/image_raw',self.image_callback,10)

        self.bridge = CvBridge()

        # Load the CNN model
        try:
            model_path = '/home/david/nonav_ws/src/bot_camera/model/unwrap_lr_1e-0.001_model11.h5'
            self.cnn_model = load_model(model_path)
            self.get_logger().info(f'Loaded CNN model from {model_path}')
        except Exception as e:
            self.get_logger().error(f'Failed to load CNN model: {e}')
            raise

        
    def image_callback(self, top_img_msg):
        # Convert ROS Image message to OpenCV format for both cameras
        top_frame = self.bridge.imgmsg_to_cv2(top_img_msg, desired_encoding='bgr8')

        # Preprocess the images (e.g., grayscale, blur)
        top_gray = cv2.cvtColor(top_frame, cv2.COLOR_BGR2GRAY)

        top_blurred = cv2.GaussianBlur(top_gray, (5, 5), 0)

        # Detect edges in both images
        top_edges = cv2.Canny(top_blurred, 50, 150)

        # Find contours in both images
        top_contours, _ = cv2.findContours(top_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Display the contours (optional for debugging)
        top_contour_image = cv2.drawContours(top_frame.copy(), top_contours, -1, (0, 255, 0), 3)

        cv2.imshow('Top Camera Contours', top_contour_image)

        # Compare the two images using pixel-wise difference
        depth_score = np.mean(top_gray)

        if self.is_negative_obstacle(top_contours, depth_score):
            self.get_logger().info('Potential negative obstacle detected!')

            # Extract ROI from the bottom camera's image
            roi = self.extract_roi(top_frame, top_contours)
            if roi is not None:
                # Resize the ROI to match the CNN input size
                resized_roi = cv2.resize(roi, (32, 32))  # Assuming CNN input size is 64x64
                prediction = self.classify_with_cnn(resized_roi)
                self.get_logger().info(f'CNN Prediction: {prediction}')
                
                if prediction == 1:  # Assuming 1 means a negative obstacle
                    self.get_logger().info('Negative obstacle confirmed by CNN!')
                else:
                    self.get_logger().info('No negative obstacle detected by CNN.')
        else:
            self.get_logger().info('No obstacle detected by edge detection.')

        cv2.waitKey(1)

    def is_negative_obstacle(self, top_contours, depth_score, diff_threshold=20):
        # Simple logic for initial negative obstacle detection using edge detection
        if depth_score > diff_threshold:
            self.get_logger().info(f"Significant depth: {depth_score}")
            return True
        return False

    def extract_roi(self, frame, contours):
        # Extract the Region of Interest (ROI) from the detected contours
        if contours:
            x, y, w, h = cv2.boundingRect(contours[0])  # Use the first contour found
            roi = frame[y:y+h, x:x+w]
            return roi
        return None

    def classify_with_cnn(self, roi):
        # Preprocess the ROI and classify it using the CNN model
        roi_normalized = roi / 255.0  # Normalize the pixel values
        roi_reshaped = np.expand_dims(roi_normalized, axis=0)  # Add batch dimension
        prediction = self.cnn_model.predict(roi_reshaped)
        return np.argmax(prediction)  # Return the class with the highest probability

def main(args=None):
    rclpy.init(args=args)
    node = NegativeObstacleDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
