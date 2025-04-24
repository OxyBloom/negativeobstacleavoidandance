import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np

class DepthChangeDetector(Node):
    def __init__(self):
        super().__init__('depth_change_detector')
        # self.ir_sub = self.create_subscription(Image, '/camera/infra1/image_raw',self.ir_callback,10)
        self.depth_sub = self.create_subscription(Image, '/camera/aligned_depth_to_color/image_raw',self.depth_callback,10)
        self.color_sub = self.create_subscription(Image, '/camera/color/image_raw',self.color_callback,10)
        self.bridge = CvBridge()
        self.ir_image = None
        self.depth_frame = None
        self.colored_image = None 
        self.depth_threshold = None
        image_processing_rate = 0.2   
        self.timer = self.create_timer(image_processing_rate, self.process_image) 

    def ir_callback(self,msg):
        self.ir_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        
    def depth_callback(self,msg):
        self.depth_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def color_callback(self,msg):
        self.colored_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
    def process_image(self):
        if self.colored_image is None or self.depth_frame is None:
            self.get_logger().info("No frame detected.")
            return
        if self.colored_image is not None:
            image_width = 1280  # Example width
            image_height = 720  # Example height

            # Parameters for trapezium
            top_width = 0.1 * image_width
            bottom_width = 0.3 * image_width
            height = 0.7 * image_height

            # Trapezium vertices
            x1, y1 = int((image_width - top_width) // 2), int(image_height - height)
            x2, y2 = int((image_width + top_width) // 2), int(image_height - height)
            x3, y3 = int((image_width - bottom_width) // 2), int(image_height - 0.8*height)
            x4, y4 = int((image_width + bottom_width) // 2), int(image_height - 0.8*height)
            cX = int((x1+x2+x3+x4)//4)
            cY = int((y1+y2+y3+y4)//4)
            print(f'({cX},{cY})')
            cv2.circle(self.colored_image, (cX, cY), 5, (0, 255, 0), -1)
            depth = self.depth_frame[cY, cX] / 1000.0

            # trapezium_vertices = [(x1, y1), (x2, y2), (x4, y4), (x3, y3)]
            trapezium_vertices = np.array([[x1, y1], [x2, y2], [x4, y4], [x3, y3]], dtype=np.int32)
            cv2.polylines(self.colored_image,[trapezium_vertices],isClosed=True, color=(0, 255, 0), thickness=3)

            mean_depth = depth #based on this negative obstacles are below 0.25 and above 0.78

            cv2.putText(self.colored_image,f"Depth: {mean_depth}",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA)

            # Check for depth anomalies
            # if mean_depth > self.depth_threshold:
            #     signal_negative_obstacle()


            self.get_logger().info(f"Depth: {mean_depth}")
            cv2.imshow("Image",self.colored_image)
            cv2.waitKey(1)
        pass

def main(args=None):
    rclpy.init(args=args)
    node = DepthChangeDetector()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
