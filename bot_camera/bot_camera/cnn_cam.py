#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from message_filters import ApproximateTimeSynchronizer, Subscriber
from tensorflow.keras.models import load_model
from math import dist

class NegativeObstacleDetector(Node):
    def __init__(self):
        super().__init__('negative_obstacle_detector')
        
        # Create subscribers for both cameras
        # self.img_sub = self.create_subscription(Image, '/camera/color/image_raw',self.image_callback,10)
        self.depth_sub = self.create_subscription(Image, '/camera/aligned_depth_to_color/image_raw',self.depth_callback,10)
        self.color_sub = self.create_subscription(Image, '/camera/color/image_raw',self.color_callback,10)
        self.bridge = CvBridge()
        self.cimg_frame = None
        self.depth_frame = None

        # Load the CNN model
        try:
            model_path = '/home/david/nonav_ws/src/bot_camera/model/wrap_lr_1e-03_model11.h5'
            self.cnn_model = load_model(model_path)
            self.get_logger().info(f'Loaded CNN model from {model_path}')
        except Exception as e:
            self.get_logger().error(f'Failed to load CNN model: {e}')
            raise

        self.process = 0.2
        self.timer = self.create_timer(self.process,self.process_image)

        
    def depth_callback(self,msg):
        
        try:
            self.depth_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().warn(f"Error in depth callback: {e}")

    def color_callback(self, img_msg):
        
        try:
            # Convert the image message to OpenCV format
            self.cimg_frame = self.bridge.imgmsg_to_cv2(img_msg, 'bgr8')
        except Exception as e:
            self.get_logger().warn(f"Error in color callback: {e}")
    
    def image_callback(self, top_img_msg):
        # Convert ROS Image message to OpenCV format for both cameras
        top_frame = self.bridge.imgmsg_to_cv2(top_img_msg, desired_encoding='bgr8')

        # Preprocess the images (e.g., grayscale, blur)
        # top_gray = cv2.cvtColor(top_frame, cv2.COLOR_BGR2GRAY)

        # top_blurred = cv2.GaussianBlur(top_gray, (5, 5), 0)

        # # Detect edges in both images
        # top_edges = cv2.Canny(top_blurred, 50, 150)

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
        # roi_normalized = roi / 255.0  # Normalize the pixel values
        
        # prediction = self.cnn_model.predict(roi_reshaped)
        roi_new = cv2.resize(roi,(32,32))
        roi_normalized = roi_new / 255.0
        roi_reshaped = np.expand_dims(roi_normalized, axis=0)  # Add batch dimension
        prediction = self.cnn_model.predict(roi_reshaped)
        return np.argmax(prediction)  # Return the class with the highest probability
    
    def process_image(self):
        
        try:
            # Convert the image message to OpenCV format
            top_frame = self.cimg_frame
            
            top_img = top_frame
            # print("hello")

            if self.cimg_frame is not None:
                # print("hello")
                image_width = 1280  # Example width
                image_height = 720  # Example height

                # Parameters for trapezium
                top_width = 0.20 * image_width
                bottom_width = 0.80 * image_width
                height = 0.7 * image_height

                # Trapezium vertices
                x1, y1 = int((image_width - top_width) // 2), int(image_height - height)
                x2, y2 = int((image_width + top_width) // 2), int(image_height - height)
                x3, y3 = int((image_width - bottom_width) // 2), int(image_height - 0.0*height)
                x4, y4 = int((image_width + bottom_width) // 2), int(image_height - 0.0*height)
                # cX = int((x1+x2+x3+x4)//4)
                # cY = int((y1+y2+y3+y4)//4)
                # print(f'({cX},{cY})')
                # cv2.circle(self.colored_image, (cX, cY), 5, (0, 255, 0), -1)
                # depth = self.depth_frame[cY, cX] / 1000.0

                # trapezium_vertices = [(x1, y1), (x2, y2), (x4, y4), (x3, y3)]
                trapezium_vertices = np.array([[x1, y1], [x2, y2], [x4, y4], [x3, y3]], dtype=np.int32)
                cv2.polylines(top_img,[trapezium_vertices],isClosed=True, color=(0, 255, 0), thickness=3)

                src_points = np.float32(trapezium_vertices)

                # depth = self.depth_frame[y2, x2] #/ 1000.0
                # depth = np.mean(self.depth_frame)

                trapezium_mask = np.zeros(self.depth_frame.shape, dtype=np.uint8)
                trapezium_mask1 = np.zeros(top_img.shape, dtype=np.uint8)
                cv2.fillPoly(trapezium_mask, [trapezium_vertices], 255)
                cv2.fillPoly(trapezium_mask1, [trapezium_vertices], 255)

                roi = cv2.bitwise_and(top_img, top_img, mask=trapezium_mask)

                masked_depth = cv2.bitwise_and(self.depth_frame, self.depth_frame, mask=trapezium_mask)
                mean_depth = np.mean(masked_depth[masked_depth != 0])

                cv2.putText(top_img,f"Depth: {mean_depth}",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA)

                # Define destination points (e.g., map to a rectangle)
                new_width = max(dist(src_points[0], src_points[1]), dist(src_points[2], src_points[3]))
                new_height = max(dist(src_points[0], src_points[3]), dist(src_points[1], src_points[2]))

                # dst_points = np.float32([[0, 0], [300, 0], [300, 300], [0, 300]])

                dst_points = np.float32([[0, 0], [new_width, 0], [new_width, new_height], [0, new_height]])


                # Compute the perspective transform matrix
                matrix = cv2.getPerspectiveTransform(src_points, dst_points)

                # Perform the perspective warp
                output = cv2.warpPerspective(top_img, matrix, (int(new_width), int(new_height)))
                self.get_logger().info(f'Depth: {mean_depth}')

                # pred_unwarpped = self.classify_with_cnn(roi)
                pred_warpped = self.classify_with_cnn(output)
                # self.get_logger().info(f'unwarpped prediction: {pred_unwarpped}')
                self.get_logger().info(f'Warpped prediction: {pred_warpped}')
                

                # if mean_depth >= 338:
                #     #save to negative obstacle folder
                #     self.get_logger().info(f'Negative obstacle detected!')
                # else:
                #     #save non_negative floor
                #     # image_name = os.path.join(self.unwarp_save_path, f'top_image_{rclpy.clock.Clock().now().to_msg().sec}.jpg')
                #     # image_name_warp = os.path.join(self.warp_save_path, f'warp_image_{rclpy.clock.Clock().now().to_msg().sec}.jpg')
                #     # cv2.imwrite(image_name, roi)
                #     # cv2.imwrite(image_name_warp, output)
                #     pass

                

                # Save the image
                # cv2.imwrite(image_name, top_img)
                cv2.imshow("negtive_obs", top_img)
                cv2.imshow('Warped Image', output)
                # cv2.imshow('Depth Image', self.depth_frame)
                cv2.waitKey(1)
        except Exception as e:
            self.get_logger().warn(f'Error: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = NegativeObstacleDetector()
    rclpy.spin(node)
    # node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
