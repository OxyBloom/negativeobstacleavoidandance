#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from ultralytics import YOLO
import tf2_ros
from rclpy.executors import MultiThreadedExecutor
import threading
import queue
import traceback
import sys
from ament_index_python.packages import get_package_share_directory
import os
# import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import TransformStamped
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from std_msgs.msg import Header
from geometry_msgs.msg import Pose2D,Point
import struct
from std_msgs.msg import Float64MultiArray, MultiArrayDimension
image_queue = queue.Queue()

def display_thread():
    while True:
        image = image_queue.get()
        if image is None:
            break
        cv2.imshow("Image", image)
        key = cv2.waitKey(1)
        if key == 27:  # ESC to quit
            break
    cv2.destroyAllWindows()

# Start the thread
threading.Thread(target=display_thread, daemon=True).start()


class YoloDetectorNode(Node):
    def __init__(self):
        super().__init__('yolo_detector')

        package_dir = get_package_share_directory('bot_camera')
        model_path = os.path.join(package_dir, 'model', 'best.pt')
        self.model = YOLO(model_path)
        self.class_names = ['holes', 'boxes', 'tools', 'negative']
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.buffer.Buffer()        
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.br = tf2_ros.TransformBroadcaster(self)
        self.camera_height=None
        self.theta = np.radians(-11)  # negative because camera is tilted downward
        self.Ry = np.array([
                [1, 0, 0],
                [0, np.cos(self.theta), -np.sin(self.theta)],
                [0, np.sin(self.theta),  np.cos(self.theta)]
            ])

        self.timer = self.create_timer(1.0, self.lookup_camera_height)

        self.subscription = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.image_callback,
            10
        )
        self.depth_sub = self.create_subscription(
            Image,
            '/camera/aligned_depth_to_color/image_raw',
            self.depth_callback,
            10
        )
        self.pointcloud_pub = self.create_publisher(PointCloud2,'/virtual_obstacles',10)
        self.detection_pub = self.create_publisher(Detection2DArray,'/detections',10)
        self.point_publisher = self.create_publisher(Float64MultiArray,'lpoints',10)
        self.depth_frame = None
        self.timer = self.create_timer(0.2, self.process_image)

    def depth_callback(self, msg):
        self.depth_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        # self.depth_frame = cv2.GaussianBlur(self.depth, (5, 5), 0)

    def image_callback(self, data):
        
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data,desired_encoding='bgr8')
            # self.cv_image = cv2.rotate(self.cv_image,cv2.ROTATE_90_CLOCKWISE)

            # np_arr = np.frombuffer(msg.data, np.uint8)
            # frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except CvBridgeError as e:
            self.get_logger().error(f"Color image conversion: {e}")
    
    def process_image(self):
        try:
            # Camera intrinsics
            sizeCamX = 1280
            sizeCamY = 720
            centerCamX = 640 
            centerCamY = 360
            focalX = 931.1829833984375
            focalY = 931.1829833984375

            if self.depth_frame is None or self.cv_image is None:
                return

            frame = self.cv_image.copy()
            image_height, image_width = frame.shape[:2]

            # === 1. Define trapezium ROI ===
            trapezium_vertices = self.define_trapezium_roi(image_width, image_height)
            cv2.polylines(frame, [trapezium_vertices], isClosed=True, color=(0, 255, 0), thickness=3)

            # trap_distance_info = self.calculate_trapezium_top_distance()
            # if trap_distance_info:
            #     distance, point_3d, pixel_pos = trap_distance_info
            #     # Draw marker at top center
            #     cv2.circle(frame, pixel_pos, 5, (255, 0, 0), -1)
            #     cv2.putText(frame, f"Top Center: {distance:.2f}m", 
            #             (pixel_pos[0] - 50, pixel_pos[1] - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # === 2. Create mask and apply to depth and RGB image ===
            mask = np.zeros(self.depth_frame.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [trapezium_vertices], 255)
            roi = cv2.bitwise_and(frame, frame, mask=mask)

            # === 3. Run YOLO within ROI ===
            detections = self.run_yolo(roi)
            negative_points = None
            bottom_points_robot = []
            all_bottom_points_robot = []

            self.get_logger().info(f"length of points test")
            for det in detections:
                try:
                    label, x1, y1, x2, y2, conf = det
                    box_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    if cv2.pointPolygonTest(trapezium_vertices, box_center, False) < 0:
                        continue  # Ignore boxes outside ROI

                    # === 4. Subdivide box vertically to check depth profile ===
                    depth_profile = self.compute_depth_profile(x1, y1, x2, y2, num_strips=10)
                    for i in range(len(depth_profile) - 1):
                        d1, d2 = depth_profile[i], depth_profile[i + 1]
                        if d1 > 0 and d2 > 0:
                            delta = d2 - d1
                            if delta > 0.015 or conf > 0.65:  # Depth drop in meters
                                # Mark detection
                                cv2.putText(frame, f"NEGATIVE: {label}", (x1, y1 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

                                # === 5. Estimate 3D position (bottom center of bbox) ===
                                bottom_points_robot = []
                                cX = int((x1 + x2) / 2)
                                cY = int(y2)

                                depth_m = self.depth_frame[cY, cX] / 1000.0

                                x = depth_m * (sizeCamX - cX - centerCamX) / focalX
                                y = depth_m * (sizeCamY -cY - centerCamY) / focalY
                                z = depth_m 
                                
                                point_cam = np.array([x, y, z])
                                negative_points = point_cam.copy()
                                pt_robot = self.Ry @ negative_points
                                # self.publish_negative_obstacle_point(pt_robot[2],pt_robot[0], pt_robot[1])
                                for cx in np.arange(x1, x2 + 1, 1):  # sample every 5 pixels (tune step for performance)
                                    cy = y2
                                    depth = self.depth_frame[cy, cx] / 1000.0  # convert to meters
                                    # if depth <= 0.1 or depth != depth:  # ignore 0 or NaN
                                    #     continue

                                    x_cam = depth * (sizeCamX - cx - centerCamX) / focalX
                                    y_cam = depth * (sizeCamY - cy - centerCamY) / focalY
                                    z_cam = depth
                                    pt_cam = np.array([x_cam, y_cam, z_cam])
                                    pt_robot = self.Ry @ pt_cam
                                    bottom_points_robot.append((pt_robot[2], pt_robot[0], pt_robot[1]))  # to match your earlier x,y,z

                                if bottom_points_robot:
                                    # p = [(x1,y2),(x2,y2)]
                                    self.publish_laser_points(bottom_points_robot)
                                    all_bottom_points_robot.extend(bottom_points_robot)
                                    # self.publish_laser_points(p)
                                    self.publish_negative_obstacle_point(bottom_points_robot)
                                break
                
                except Exception as inner_e:
                    self.get_logger().warn(f"Skipping detection due to error: {inner_e}")


            # === 6. Calculate angle and height drop, then project to robot frame ===
            if negative_points is not None:
                q_deg, h_prime = self.compute_floor_slope_and_drop(negative_points)
                point_robot = self.Ry @ negative_points
                t = TransformStamped()
                t.header.stamp = self.get_clock().now().to_msg()
                t.header.frame_id = 'base_link'    
                t.child_frame_id = f'neg' 
                # Translation (position of the box)
                
                t.transform.translation.x = point_robot[2]
                t.transform.translation.y = point_robot[0]                                    
                t.transform.translation.z = point_robot[1]
                # Orientation (custom quaternion for the box's rotation)

                self.br.sendTransform(t)
                # self.publish_negative_obstacle_point(point_robot[2],point_robot[0], point_robot[1])
                # cv2.putText(frame, f"Angle: {q_deg:.2f}  Height: {h_prime:.3f}m  Distance: {point_robot[2]:.3f}m",
                #             (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(frame, f"Height: {h_prime:.3f}m  Distance: {point_robot[2]:.3f}m",
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if all_bottom_points_robot:
                self.publish_laser_points(all_bottom_points_robot)

            image_queue.put(frame)

        except Exception as e:
            self.get_logger().error(f"Image callback error: {str(e)}")
            tb = sys.exc_info()[2]
            lineno = tb.tb_lineno
            self.get_logger().error(f"Image callback error at line: {str(lineno)}")

    # def publish_laser_points(self, bottom_points_robot):
    #     self.get_logger().info(f"length of point: {len(bottom_points_robot)}")
    #     for pt in bottom_points_robot:
    #         msg = Float64MultiArray()
    #         msg.x = float(pt[0])  # pt_robot[2]
    #         msg.y = float(pt[1])  # pt_robot[0]
    #         msg.z = 0.0 #float(pt[2])          # optional, or use pt[2] if needed
    #         self.point_publisher.publish(msg)
    #         self.get_logger().debug(f"Published point: x={msg.x}, y={msg.y}")

    def publish_laser_points(self, bottom_points_robot):
        self.get_logger().info(f"length of points: {len(bottom_points_robot)}")

        msg = Float64MultiArray()

        # Flatten the list of 3D points into a single list [x1, y1, z1, x2, y2, z2, ...]
        flat_data = [coord for pt in bottom_points_robot for coord in (pt[0], pt[1], 0.0)]
        msg.data = flat_data

        # Define the layout if the subscriber wants to reshape it
        msg.layout.dim.append(MultiArrayDimension(label="points", size=len(bottom_points_robot), stride=len(bottom_points_robot)*3))
        msg.layout.dim.append(MultiArrayDimension(label="xyz", size=3, stride=3))

        self.point_publisher.publish(msg)
        self.get_logger().debug(f"Published {len(bottom_points_robot)} points.")

    def run_yolo(self, frame):
        results = self.model(frame)
        detections = []
        detections_msg = Detection2DArray()
        detections_msg.header.stamp = self.get_clock().now().to_msg()
        detections_msg.header.frame_id = "camera_link"

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf.item()
                class_id = int(box.cls.item())
                class_name = self.class_names[class_id]
                detections.append([class_name, x1, y1, x2, y2, conf])

                detect = Detection2D()
                detect.bbox.center.position.x = (x1 + x2) / 2.0
                detect.bbox.center.position.y = (y1 + y2) / 2.0
                detect.bbox.size_x = float(x2 - x1)
                detect.bbox.size_y = float(y2 - y1)

                hypothesis = ObjectHypothesisWithPose()
                hypothesis.hypothesis.class_id = str(class_id)
                hypothesis.hypothesis.score = conf
                detect.results.append(hypothesis)

                detections_msg.detections.append(detect)

        self.detection_pub.publish(detections_msg)

        return detections

    def define_trapezium_roi(self, image_width, image_height):
        top_width = 0.20 * image_width
        bottom_width = 0.80 * image_width
        height = 0.7 * image_height
        x1, y1 = int((image_width - top_width) // 2), int(image_height - height)
        x2, y2 = int((image_width + top_width) // 2), int(image_height - height)
        x3, y3 = int((image_width - bottom_width) // 2), int(image_height)
        x4, y4 = int((image_width + bottom_width) // 2), int(image_height)
        return np.array([[x1, y1], [x2, y2], [x4, y4], [x3, y3]], dtype=np.int32)

    def compute_depth_profile(self, x1, y1, x2, y2, num_strips=6):
        strip_height = max(1, (y2 - y1) // num_strips)
        profile = []
        for i in range(num_strips):
            y_start = y1 + i * strip_height
            y_end = y1 + (i + 1) * strip_height
            crop = self.depth_frame[y_start:y_end, x1:x2]
            nonzero = crop[crop > 0]
            avg = np.mean(nonzero) / 1000.0 if nonzero.size > 0 else 0  # in meters
            profile.append(avg)
        return profile

    def lookup_camera_height(self):
        try:
            # Replace these with your actual frame names
            from_frame = 'camera_link'
            to_frame = 'odom'  # or 'odom' if you want height from world

            t = self.tf_buffer.lookup_transform(
                to_frame,
                from_frame,
                rclpy.time.Time()
            )

            self.camera_height = t.transform.translation.z
            # self.get_logger().info(f'Camera height relative to {to_frame}: {self.camera_height:.3f} m')

        except Exception as e:
            self.get_logger().warn(f'Could not get transform: {e}')

    def compute_floor_slope_and_drop(self,point):
        
        point_cam = np.array(point)  # In camera frame

        # The camera is at (0, 0, 0), and the floor point is at `point_cam`
        # Slope of the ramp (α): angle between vector and horizontal plane (x-z)
        vec = point_cam
        horizontal_proj = np.linalg.norm([vec[0], vec[2]])  # magnitude in xz plane
        alpha = np.arctan2(-vec[1], horizontal_proj)  # negative because y is "down" from camera
        q = np.pi/2 - alpha  # angle from vertical

        # Drop height relative to base of robot
        h_prime = self.camera_height - (-vec[1])  # y is negative if point is below camera

        return np.degrees(q), h_prime
    
    def publish_negative_obstacle_point(self,points_3d):
        # for point in points_3d:
        #     points_3d.append((point[0],point[1],(point[2]-0.05)))
        cloud = PointCloud2()
        cloud.header.stamp = self.get_clock().now().to_msg()
        cloud.header.frame_id = 'base_link'
        cloud.height = 1
        cloud.width = len(points_3d)
        cloud.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        cloud.is_bigendian = False
        cloud.point_step = 12
        cloud.row_step = cloud.point_step * cloud.width
        cloud.is_dense = True
        # cloud.data = struct.pack('fff', x, y, z)
        cloud.data = b''.join([struct.pack('fff', x, y, z) for x, y, z in points_3d])
        self.pointcloud_pub.publish(cloud)

    def calculate_trapezium_top_distance(self):
        """Calculate distance from base_footprint to center of trapezium top"""
        try:
            if self.cv_image is None or self.depth_frame is None:
                return None
                
            # Camera intrinsics
            sizeCamX = 1280
            sizeCamY = 720
            centerCamX = 640 
            centerCamY = 360
            focalX = 931.1829833984375
            focalY = 931.1829833984375
            
            frame = self.cv_image.copy()
            image_height, image_width = frame.shape[:2]
            
            # Get trapezium vertices
            trapezium_vertices = self.define_trapezium_roi(image_width, image_height)
            
            # Calculate center of top edge
            top_left = trapezium_vertices[0]   # [x1, y1]
            top_right = trapezium_vertices[1]  # [x2, y2]
            top_center_x = int((top_left[0] + top_right[0]) / 2)
            top_center_y = int((top_left[1] + top_right[1]) / 2)
            
            # Get depth at top center
            depth_mm = self.depth_frame[top_center_y, top_center_x]
            if depth_mm <= 0:
                self.get_logger().warn("Invalid depth at trapezium top center")
                return None
                
            depth_m = depth_mm / 1000.0
            
            # Convert pixel to 3D camera coordinates
            x_cam = depth_m * (sizeCamX - top_center_x - centerCamX) / focalX
            y_cam = depth_m * (sizeCamY - top_center_y - centerCamY) / focalY
            z_cam = depth_m
            
            point_cam = np.array([x_cam, y_cam, z_cam])
            
            # Transform to robot frame
            point_robot = self.Ry @ point_cam
            
            # Get transform from base_link to base_footprint
            try:
                transform = self.tf_buffer.lookup_transform(
                    'base_link', 'camera_link', rclpy.time.Time())
                
                # Apply transform (usually just a z-offset)
                base_footprint_offset_z = transform.transform.translation.z
                point_footprint = point_robot.copy()
                point_footprint[1] += base_footprint_offset_z  # Adjust z coordinate
                
            except Exception as tf_e:
                self.get_logger().warn(f"Could not get base_footprint transform: {tf_e}")
                point_footprint = point_robot  # Use base_link coordinates
            
            # Calculate distance
            distance = np.linalg.norm(point_footprint)
            
            self.get_logger().info(f"Trapezium top center - Pixel: ({top_center_x}, {top_center_y}), "
                                f"3D robot: ({point_robot[2]:.3f}, {point_robot[0]:.3f}, {point_robot[1]:.3f}), "
                                f"Distance from base_footprint: {distance:.3f}m")
            
            return distance, point_footprint, (top_center_x, top_center_y)
            
        except Exception as e:
            self.get_logger().error(f"Error calculating trapezium distance: {e}")
            return None


def main(args=None):
    rclpy.init(args=args)
    node = YoloDetectorNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
