import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
import cv2
import numpy as np

class ColorDepthSubscriber(Node):
    def __init__(self):
        super().__init__('color_depth_subscriber')
        self.bridge = CvBridge()

        # Subscribe to color and depth topics
        self.color_sub = Subscriber(self, Image, '/camera/color/image_raw')
        self.depth_sub = Subscriber(self, Image, '/camera/aligned_depth_to_color/image_raw')

        # Synchronize color and depth images using ApproximateTimeSynchronizer
        self.ts = ApproximateTimeSynchronizer([self.color_sub, self.depth_sub], 10, 0.1)
        self.ts.registerCallback(self.color_depth_callback)

    def color_depth_callback(self, color_msg, depth_msg):
        # Convert ROS Image messages to OpenCV images
        color_image = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

        # Process color and depth images
        # depth_image_color = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        # combined_image = cv2.addWeighted(color_image, 0.7, depth_image_color, 0.3, 0)
        combined_image = np.concatenate((color_image,np.expand_dims(depth_image,axis=2)),axis=1)
        cv2.imshow('Depth-Color Image', combined_image)
        cv2.waitKey(1)
        # self.get_logger().info(f"Received color image with shape {color_image.shape} and depth image with shape {depth_image.shape}")

def main(args=None):
    rclpy.init(args=args)
    node = ColorDepthSubscriber()
    try:
        rclpy.spin(node)
    except ExternalShutdownException:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()