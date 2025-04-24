#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from ultralytics import YOLO
from math import dist


class YoloDetectorNode(Node):
    def __init__(self):
        super().__init__('yolo_detector')

        # Initialize YOLOv8 model with the custom weights
        # self.model = YOLO('/home/david/ws_ 2/runs/detect/train/weights/best.pt') #/home/david/nav2_ws/Dataset_unwrap_yolo_complete/runs/detect/train/weights/best.pt
        self.model = YOLO('/home/david/nav2_ws/Dataset_wrap_yolo/runs/detect/train/weights/best.pt')

        # Define class names (adjust according to your training)
        self.class_names = ['holes', 'boxes', 'tools', 'negative']

        # Initialize the CvBridge class to convert ROS images to OpenCV format
        self.bridge = CvBridge()

        # Subscribe to the camera topic
        self.subscription = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.image_callback,
            10
        )
        self.depth_sub = self.create_subscription(Image, '/camera/aligned_depth_to_color/image_raw',self.depth_callback,10)
        self.depth_frame = None

    def depth_callback(self,msg):
        self.depth_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
    
    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

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

            trapezium_vertices = np.array([[x1, y1], [x2, y2], [x4, y4], [x3, y3]], dtype=np.int32)
            cv2.polylines(frame,[trapezium_vertices],isClosed=True, color=(0, 255, 0), thickness=3)

            trapezium_mask = np.zeros(self.depth_frame.shape, dtype=np.uint8)
            trapezium_mask1 = np.zeros(frame.shape, dtype=np.uint8)
            cv2.fillPoly(trapezium_mask, [trapezium_vertices], 255)
            cv2.fillPoly(trapezium_mask1, [trapezium_vertices], 255)

            masked_depth = cv2.bitwise_and(self.depth_frame, self.depth_frame, mask=trapezium_mask)
            mean_depth = np.mean(masked_depth[masked_depth != 0])
            cv2.putText(frame,f"Depth: {mean_depth}",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA)
            roi = cv2.bitwise_and(frame, frame, mask=trapezium_mask)

            src_points = np.float32(trapezium_vertices)
            new_width = max(dist(src_points[0], src_points[1]), dist(src_points[2], src_points[3]))
            new_height = max(dist(src_points[0], src_points[3]), dist(src_points[1], src_points[2]))
            dst_points = np.float32([[0, 0], [new_width, 0], [new_width, new_height], [0, new_height]])

             # Compute the perspective transform matrix
            matrix = cv2.getPerspectiveTransform(src_points, dst_points)

            # Perform the perspective warp
            output = cv2.warpPerspective(frame, matrix, (int(new_width), int(new_height)))
            # Run YOLOv8 model on the current frame
            # detections = self.run_yolo(frame)
            detections = self.run_yolo(output)

            # Process the detection results
            for detection in detections:
                class_name, x1, y1, x2, y2, conf = detection

                wrap_src_points = np.float32([[0, 0], [new_width, 0], [new_width, new_height], [0, new_height]])
                wrap_dst_points = np.float32(trapezium_vertices)
                wrap_matrix = cv2.getPerspectiveTransform(wrap_src_points, wrap_dst_points)

                unwrapped = cv2.warpPerspective(output, wrap_matrix, (int(image_width), int(image_height)))

                # Draw bounding boxes and labels on the image
                cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
                cv2.putText(output, f"{class_name} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                
                wrap_src_points = np.float32([[0, 0], [new_width, 0], [new_width, new_height], [0, new_height]])
                wrap_dst_points = np.float32(trapezium_vertices)
                wrap_matrix = cv2.getPerspectiveTransform(wrap_src_points, wrap_dst_points)

                unwrapped = cv2.warpPerspective(output, wrap_matrix, (int(image_width), int(image_height)))
                frame = unwrapped
                
                if mean_depth >= 338:
                    cv2.putText(frame,f"Negative Obstacle confirmed!",(50,200),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA)

            # Display the frame with detections (for debugging)
            cv2.imshow('YOLOv8 Detection', frame)
            cv2.imshow('Wrapped YOLOv8 Detection', output)
            cv2.waitKey(1)  # Required to display OpenCV window

        except Exception as e:
            self.get_logger().error(f"CvBridgeError: {str(e)}")

    def run_yolo(self, frame):
        results = self.model(frame)
        fruits = []

        for result in results:
            boxes = result.boxes  # YOLOv8 detection results

            for box in boxes:
                # Extract coordinates and class index
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                conf = box.conf.item()  # Confidence score
                class_id = int(box.cls.item())  # Class ID

                # Get class name
                class_name = self.class_names[class_id]
                fruits.append([class_name, x1, y1, x2, y2, conf])

        return fruits


def main(args=None):
    rclpy.init(args=args)

    # Create the YOLO detector node
    yolo_detector = YoloDetectorNode()

    # Spin to process callbacks
    rclpy.spin(yolo_detector)

    # Shutdown ROS2 when done
    yolo_detector.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
