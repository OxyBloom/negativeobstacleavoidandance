#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import numpy as np
from math import dist

class DataCollector(Node):
    def __init__(self):
        super().__init__('data_collector')

        self.bridge = CvBridge()

        # Create subscriptions for both cameras
        self.depth_sub = self.create_subscription(Image, '/camera/aligned_depth_to_color/image_raw',self.depth_callback,10)
        self.color_sub = self.create_subscription(Image, '/camera/color/image_raw',self.color_callback,10)
        
        # Define save paths for both cameras
        self.unwarp_negative_save_path = "/home/david/nonav_ws/Dataset/unwarp/negative"
        self.unwarp_save_path = "/home/david/nonav_ws/Dataset/unwarp/nonegative"
        self.warp_negative_save_path = "/home/david/nonav_ws/Dataset/warp/negative"
        self.warp_save_path = "/home/david/nonav_ws/Dataset/warp/nonegative"
        # self.bottom_camera_save_path = "/home/shonde/nav2_ws/src/bot_camera/Dataset"

        # Create directories if they do not exist
        os.makedirs(self.unwarp_negative_save_path, exist_ok=True)
        os.makedirs(self.unwarp_save_path, exist_ok=True)
        os.makedirs(self.warp_negative_save_path, exist_ok=True)
        os.makedirs(self.warp_save_path, exist_ok=True)
        self.process = 0.2
        self.timer = self.create_timer(self.process,self.process_image)
        # os.makedirs(self.bottom_camera_save_path, exist_ok=True)

    def depth_callback(self,msg):
        try:
            self.depth_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().warn(f"Error in depth callback: {e}")

    def color_callback(self, img_msg):
        try:
            # Convert the image message to OpenCV format
            self.top_frame = self.bridge.imgmsg_to_cv2(img_msg, 'bgr8')
        except Exception as e:
            self.get_logger().warn(f"Error in color callback: {e}")

    
    def process_image(self):
        try:
            # Convert the image message to OpenCV format
            top_frame = self.top_frame
            

            # top_blurred = cv2.GaussianBlur(top_gray, (5, 5), 0)

            # Detect edges in both images
            # top_edges = cv2.Canny(top_blurred, 50, 150)
            # top_img = top_edges
            top_img = top_frame
            # if top_img is not None:
                # cv2.imshow("negtive_obs", top_img)
                # cv2.waitKey(1)

            if self.top_frame is not None:
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
                # cv2.putText(top_img,f"1",(x1,y1),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA)
                # cv2.putText(top_img,f"2",(x2,y2),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA)
                # cv2.putText(top_img,f"3",(x3,y3),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA)
                # cv2.putText(top_img,f"4",(x4,y4),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA)
                
            
                # Generate image filename based on the current time
                # image_name = os.path.join(self.top_camera_save_path, f'top_image_{rclpy.clock.Clock().now().to_msg().sec}.jpg')

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
                

                # if mean_depth >= 338:
                #     #save to negative obstacle folder
                #     image_name = os.path.join(self.unwarp_negative_save_path, f'{mean_depth}_top_image_{rclpy.clock.Clock().now().to_msg().sec}.jpg')
                #     image_name_warp = os.path.join(self.warp_negative_save_path, f'{mean_depth}_warp_image_{rclpy.clock.Clock().now().to_msg().sec}.jpg')
                #     cv2.imwrite(image_name, roi)
                #     cv2.imwrite(image_name_warp, output)
                # else:
                #     #save non_negative floor
                #     image_name = os.path.join(self.unwarp_save_path, f'top_image_{rclpy.clock.Clock().now().to_msg().sec}.jpg')
                #     image_name_warp = os.path.join(self.warp_save_path, f'warp_image_{rclpy.clock.Clock().now().to_msg().sec}.jpg')
                #     cv2.imwrite(image_name, roi)
                #     cv2.imwrite(image_name_warp, output)

                

                # Save the image
                # cv2.imwrite(image_name, top_img)
                # cv2.imshow("negtive_obs", top_img)
                # cv2.imshow('Warped Image', output)
                cv2.imshow('Depth Image', self.depth_frame)
                cv2.waitKey(1)
        except Exception as e:
            self.get_logger().warn(f'Error: {e}')



def main():
    rclpy.init()
    node = DataCollector()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
