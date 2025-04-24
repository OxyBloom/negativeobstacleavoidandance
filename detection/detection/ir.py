import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np

class DepthChangeDetector(Node):
    def __init__(self):
        super().__init__('depth_change_detector')
        self.ir_sub = self.create_subscription(Image, '/camera/infra1/image_raw',self.ir_callback,10)
        self.depth_sub = self.create_subscription(Image, '/camera/aligned_depth_to_color/image_raw',self.depth_callback,10)
        self.color_sub = self.create_subscription(Image, '/camera/color/image_raw',self.color_callback,10)
        self.bridge = CvBridge()
        self.ir_image = None
        self.depth_image = None
        self.colored_image = None 

    def ir_callback(self,msg):
        self.ir_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        # print(self.ir_image)

    def depth_callback(self,msg):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def color_callback(self,msg):
        self.colored_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        # print(self.ir_msg)
        # print(self.ir_image)

        if self.ir_image is not None and self.depth_image is not None:
            # print("hello")
            h, w = self.ir_image.shape
            depth_center = self.depth_image[h//2,w//2]

            self.get_logger().info(f"Depth at center: {depth_center: .2f} meters")
            cv2.circle(self.colored_image, (622, 431), 5, (0, 255, 0), -1)
            cv2.putText(self.colored_image,f"Depth: {depth_center: .2f}",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA,True)

            cv2.imshow("IR Image", self.colored_image)
            # cv2.imshow("Depth Image", self.depth_image / np.max(self.depth_image))
            cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = DepthChangeDetector()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
