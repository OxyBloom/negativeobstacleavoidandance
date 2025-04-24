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
        self.top_camera_sub = self.create_subscription(Image, '/camera/color/image_raw',self.image_callback,10)
        self.depth_sub = self.create_subscription(Image, '/camera/aligned_depth_to_color/image_raw',self.depth_callback,10)
        # self.bottom_camera_sub = Subscriber(self, Image, '/camera4/image_raw')

        # Synchronize the two camera feeds
        # self.ts = ApproximateTimeSynchronizer([self.top_camera_sub, self.bottom_camera_sub], queue_size=10, slop=0.1)
        # self.ts.registerCallback(self.image_callback)

        self.bridge = CvBridge()

        # Load the CNN model for final classification
        self.cnn_model = load_model('/home/david/nonav_ws/src/bot_camera/model/wrap_lr_1e-03_model11.h5')  # Replace with the actual path to your trained model

    def depth_callback(self,msg):
    
        try:
            self.depth_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().warn(f"Error in depth callback: {e}")

    def image_callback(self, top_img_msg):
        # Convert ROS Image message to OpenCV format for both cameras
        top_frame = self.bridge.imgmsg_to_cv2(top_img_msg, desired_encoding='bgr8')
        # bottom_frame = self.bridge.imgmsg_to_cv2(bottom_img_msg, desired_encoding='bgr8')

        # Preprocess the images (e.g., grayscale, blur)
        top_gray = cv2.cvtColor(top_frame, cv2.COLOR_BGR2GRAY)
        # bottom_gray = cv2.cvtColor(bottom_frame, cv2.COLOR_BGR2GRAY)

        top_blurred = cv2.GaussianBlur(top_gray, (5, 5), 0)
        # bottom_blurred = cv2.GaussianBlur(bottom_gray, (5, 5), 0)

        # Detect edges in both images
        top_edges = cv2.Canny(top_blurred, 50, 150)
        # bottom_edges = cv2.Canny(bottom_blurred, 50, 150)

        # Find contours in both images
        top_contours, _ = cv2.findContours(top_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # bottom_contours, _ = cv2.findContours(bottom_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Display the contours (optional for debugging)
        top_contour_image = cv2.drawContours(top_frame.copy(), top_contours, -1, (0, 255, 0), 3)
        # bottom_contour_image = cv2.drawContours(bottom_frame.copy(), bottom_contours, -1, (0, 255, 0), 3)

        cv2.imshow('Top Camera Contours', top_contour_image)
        # cv2.imshow('Bottom Camera Contours', bottom_contour_image)

        # Compare the two images using pixel-wise difference
        # difference = cv2.absdiff(top_gray, bottom_gray)
        # diff_score = np.mean(difference)

        # self.get_logger().info(f'Difference score: {diff_score}')

        # Extract Region of Interest (ROI) for CNN classification if a potential obstacle is detected
        # if self.is_negative_obstacle(top_frame):
        #     self.get_logger().info('Potential negative obstacle detected!')

        #     # Extract ROI from the bottom camera's image
        #     roi = self.extract_roi(top_frame)
        #     if roi is not None:
        #         # Resize the ROI to match the CNN input size
        #         resized_roi = cv2.resize(roi, (32, 32))  # Assuming CNN input size is 64x64
        #         prediction = self.classify_with_cnn(resized_roi)
        #         self.get_logger().info(f'CNN Prediction: {prediction}')
                
        #         if prediction == 1:  # Assuming 1 means a negative obstacle
        #             self.get_logger().info('Negative obstacle confirmed by CNN!')
        #         else:
        #             self.get_logger().info('No negative obstacle detected by CNN.')
        # else:
        #     self.get_logger().info('No obstacle detected by edge detection.')

        roi = self.extract_roi(top_frame,top_contours)
        cv2.imshow('roi',roi)
        if roi is not None:
            # Resize the ROI to match the CNN input size
            resized_roi = cv2.resize(roi, (32, 32))  # Assuming CNN input size is 64x64
            prediction = self.classify_with_cnn(resized_roi)
            self.get_logger().info(f'CNN Prediction: {prediction}')
            
            if prediction == 1:  # Assuming 1 means a negative obstacle
                self.get_logger().info('Negative obstacle confirmed by CNN!')
            else:
                self.get_logger().info('No negative obstacle detected by CNN.')

        cv2.waitKey(1)

    def is_negative_obstacle(self, top_contours, bottom_contours, diff_score, diff_threshold=20):
        # Simple logic for initial negative obstacle detection using edge detection
        if diff_score > diff_threshold:
            self.get_logger().info(f"Significant difference between top and bottom cameras: {diff_score}")
            return True
        return False

    def extract_roi(self, frame, contours):
        # Extract the Region of Interest (ROI) from the detected contours
        if contours:
            x, y, w, h = cv2.boundingRect(contours[0])  # Use the first contour found
            roi = frame[y:y+h, x:x+w]
            depth = self.depth_frame[y:y+h,x:x+w]
            self.mean_depth = np.mean(depth)
            self.get_logger().info(f"Mean depth: {self.mean_depth}")
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
