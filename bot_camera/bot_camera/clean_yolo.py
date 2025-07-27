#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image, PointCloud2, PointField
from geometry_msgs.msg import TransformStamped
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from std_msgs.msg import Float64MultiArray, MultiArrayDimension
from cv_bridge import CvBridge, CvBridgeError

import cv2
import numpy as np
from ultralytics import YOLO
import tf2_ros
import threading
import queue
import struct
from dataclasses import dataclass
from typing import List, Tuple, Optional
import traceback
from ament_index_python.packages import get_package_share_directory
import os

# Configuration constants
@dataclass
class CameraConfig:
    WIDTH: int = 1280
    HEIGHT: int = 720
    CENTER_X: int = 640
    CENTER_Y: int = 360
    FOCAL_X: float = 931.1829833984375
    FOCAL_Y: float = 931.1829833984375
    TILT_ANGLE: float = np.radians(-11)

@dataclass
class ROIConfig:
    TOP_WIDTH_RATIO: float = 0.20
    BOTTOM_WIDTH_RATIO: float = 0.80
    HEIGHT_RATIO: float = 0.7

@dataclass
class DetectionConfig:
    DEPTH_THRESHOLD: float = 0.015
    CONFIDENCE_THRESHOLD: float = 0.65
    DEPTH_STRIPS: int = 10
    SAMPLE_STEP: int = 1

class ImageDisplay:
    """Handles OpenCV display in a separate thread"""
    def __init__(self):
        self.image_queue = queue.Queue()
        self.display_thread = threading.Thread(target=self._display_worker, daemon=True)
        self.display_thread.start()
    
    def _display_worker(self):
        while True:
            image = self.image_queue.get()
            if image is None:
                break
            cv2.imshow("YOLO Detection", image)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break
        cv2.destroyAllWindows()
    
    def show(self, image: np.ndarray):
        try:
            self.image_queue.put_nowait(image)
        except queue.Full:
            pass  # Drop frame if queue is full

class CameraCalibration:
    """Handles camera transformations and 3D projections"""
    def __init__(self, config: CameraConfig):
        self.config = config
        self.rotation_matrix = self._create_rotation_matrix()
    
    def _create_rotation_matrix(self) -> np.ndarray:
        """Create rotation matrix for camera tilt correction"""
        theta = self.config.TILT_ANGLE
        return np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ])
    
    def pixel_to_camera_coords(self, u: int, v: int, depth: float) -> np.ndarray:
        """Convert pixel coordinates to 3D camera coordinates"""
        x = depth * (self.config.WIDTH - u - self.config.CENTER_X) / self.config.FOCAL_X
        y = depth * (self.config.HEIGHT - v - self.config.CENTER_Y) / self.config.FOCAL_Y
        z = depth
        return np.array([x, y, z])
    
    def camera_to_robot_coords(self, point_cam: np.ndarray) -> np.ndarray:
        """Transform from camera frame to robot frame"""
        return self.rotation_matrix @ point_cam

class ROIProcessor:
    """Handles Region of Interest operations"""
    def __init__(self, config: ROIConfig):
        self.config = config
    
    def create_trapezium_vertices(self, width: int, height: int) -> np.ndarray:
        """Create trapezium ROI vertices"""
        top_w = self.config.TOP_WIDTH_RATIO * width
        bot_w = self.config.BOTTOM_WIDTH_RATIO * width
        roi_h = self.config.HEIGHT_RATIO * height
        
        x1, y1 = int((width - top_w) // 2), int(height - roi_h)
        x2, y2 = int((width + top_w) // 2), int(height - roi_h)
        x3, y3 = int((width - bot_w) // 2), int(height)
        x4, y4 = int((width + bot_w) // 2), int(height)
        
        return np.array([[x1, y1], [x2, y2], [x4, y4], [x3, y3]], dtype=np.int32)
    
    def create_mask(self, shape: Tuple[int, int], vertices: np.ndarray) -> np.ndarray:
        """Create binary mask from vertices"""
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.fillPoly(mask, [vertices], 255)
        return mask

class DepthAnalyzer:
    """Analyzes depth information for obstacle detection"""
    def __init__(self, config: DetectionConfig):
        self.config = config
    
    def compute_depth_profile(self, depth_frame: np.ndarray, 
                            x1: int, y1: int, x2: int, y2: int) -> List[float]:
        """Compute vertical depth profile of bounding box"""
        strip_height = max(1, (y2 - y1) // self.config.DEPTH_STRIPS)
        profile = []
        
        for i in range(self.config.DEPTH_STRIPS):
            y_start = y1 + i * strip_height
            y_end = y1 + (i + 1) * strip_height
            crop = depth_frame[y_start:y_end, x1:x2]
            nonzero = crop[crop > 0]
            avg = np.mean(nonzero) / 1000.0 if nonzero.size > 0 else 0
            profile.append(avg)
        
        return profile
    
    def has_depth_drop(self, profile: List[float], confidence: float) -> bool:
        """Check if depth profile indicates a drop/hole"""
        for i in range(len(profile) - 1):
            d1, d2 = profile[i], profile[i + 1]
            if d1 > 0 and d2 > 0:
                delta = d2 - d1
                if delta > self.config.DEPTH_THRESHOLD or confidence > self.config.CONFIDENCE_THRESHOLD:
                    return True
        return False

class YoloDetectorNode(Node):
    def __init__(self):
        super().__init__('yolo_detector')
        
        # Initialize configurations
        self.cam_config = CameraConfig()
        self.roi_config = ROIConfig()
        self.det_config = DetectionConfig()
        
        # Initialize components
        self.display = ImageDisplay()
        self.calibration = CameraCalibration(self.cam_config)
        self.roi_processor = ROIProcessor(self.roi_config)
        self.depth_analyzer = DepthAnalyzer(self.det_config)
        
        # Initialize YOLO and bridge
        package_dir = get_package_share_directory('bot_camera')
        model_path = os.path.join(package_dir, 'model', 'best.pt')
        self.model = YOLO(model_path)
        self.class_names = ['holes', 'boxes', 'tools', 'negative']
        self.bridge = CvBridge()
        
        # TF2 setup
        self.tf_buffer = tf2_ros.buffer.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        
        # State variables
        self.camera_height: Optional[float] = None
        self.cv_image: Optional[np.ndarray] = None
        self.depth_frame: Optional[np.ndarray] = None
        
        # Publishers and Subscribers
        self._setup_communication()
        
        # Timers
        self.create_timer(1.0, self._lookup_camera_height)
        self.create_timer(0.2, self._process_image)
    
    def _setup_communication(self):
        """Setup ROS2 publishers and subscribers"""
        # Subscribers
        self.create_subscription(Image, '/camera/color/image_raw', 
                               self._image_callback, 10)
        self.create_subscription(Image, '/camera/aligned_depth_to_color/image_raw',
                               self._depth_callback, 10)
        
        # Publishers
        self.pointcloud_pub = self.create_publisher(PointCloud2, '/virtual_obstacles', 10)
        self.detection_pub = self.create_publisher(Detection2DArray, '/detections', 10)
        self.point_publisher = self.create_publisher(Float64MultiArray, 'lpoints', 10)
    
    def _image_callback(self, msg: Image):
        """Handle incoming RGB image"""
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"Color image conversion: {e}")
    
    def _depth_callback(self, msg: Image):
        """Handle incoming depth image"""
        try:
            self.depth_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except CvBridgeError as e:
            self.get_logger().error(f"Depth image conversion: {e}")
    
    def _process_image(self):
        """Main processing loop"""
        if self.cv_image is None or self.depth_frame is None:
            return
        
        try:
            frame = self.cv_image.copy()
            height, width = frame.shape[:2]
            
            # Create ROI
            trapezium_vertices = self.roi_processor.create_trapezium_vertices(width, height)
            cv2.polylines(frame, [trapezium_vertices], isClosed=True, color=(0, 255, 0), thickness=2)
            
            # Apply mask and run YOLO
            mask = self.roi_processor.create_mask(self.depth_frame.shape, trapezium_vertices)
            roi_image = cv2.bitwise_and(frame, frame, mask=mask)
            
            detections = self._run_yolo_inference(roi_image)
            obstacle_points = self._process_detections(detections, trapezium_vertices, frame)
            
            if obstacle_points:
                self._publish_obstacle_points(obstacle_points)
            
            self.display.show(frame)
            
        except Exception as e:
            self.get_logger().error(f"Processing error: {e}")
            self.get_logger().error(traceback.format_exc())
    
    def _run_yolo_inference(self, image: np.ndarray) -> List:
        """Run YOLO inference and publish detection messages"""
        results = self.model(image)
        detections = []
        detections_msg = Detection2DArray()
        detections_msg.header.stamp = self.get_clock().now().to_msg()
        detections_msg.header.frame_id = "camera_link"
        
        for result in results:
            if result.boxes is None:
                continue
                
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf.item())
                class_id = int(box.cls.item())
                class_name = self.class_names[class_id]
                
                detections.append((class_name, x1, y1, x2, y2, conf))
                
                # Create detection message
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
    
    def _process_detections(self, detections: List, roi_vertices: np.ndarray, 
                          frame: np.ndarray) -> List[Tuple[float, float, float]]:
        """Process detections to find obstacles"""
        all_obstacle_points = []
        
        for detection in detections:
            label, x1, y1, x2, y2, conf = detection
            
            # Check if detection is within ROI
            box_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            if cv2.pointPolygonTest(roi_vertices, box_center, False) < 0:
                continue
            
            # Analyze depth profile
            depth_profile = self.depth_analyzer.compute_depth_profile(
                self.depth_frame, x1, y1, x2, y2)
            
            if self.depth_analyzer.has_depth_drop(depth_profile, conf):
                # Mark as obstacle
                cv2.putText(frame, f"OBSTACLE: {label}", (x1, y1 - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # Generate 3D points
                obstacle_points = self._generate_3d_points(x1, x2, y2)
                all_obstacle_points.extend(obstacle_points)
                
                # Add info to frame
                if obstacle_points:
                    first_point = obstacle_points[0]
                    distance = first_point[0]  # x in robot frame
                    cv2.putText(frame, f"Distance: {distance:.2f}m", (x1, y2 + 20),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        return all_obstacle_points
    
    def _generate_3d_points(self, x1: int, x2: int, y2: int) -> List[Tuple[float, float, float]]:
        """Generate 3D points along the bottom edge of detection"""
        points_3d = []
        
        for cx in range(x1, x2 + 1, self.det_config.SAMPLE_STEP):
            depth = self.depth_frame[y2, cx] / 1000.0  # Convert to meters
            if depth <= 0 or np.isnan(depth):
                continue
            
            point_cam = self.calibration.pixel_to_camera_coords(cx, y2, depth)
            point_robot = self.calibration.camera_to_robot_coords(point_cam)
            
            # Format as (x, y, z) in robot frame
            points_3d.append((point_robot[2], point_robot[0], point_robot[1]))
        
        return points_3d
    
    def _publish_obstacle_points(self, points_3d: List[Tuple[float, float, float]]):
        """Publish obstacle points as PointCloud2 and Float64MultiArray"""
        # Publish as PointCloud2
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
        cloud.data = b''.join([struct.pack('fff', x, y, z) for x, y, z in points_3d])
        
        self.pointcloud_pub.publish(cloud)
        
        # Publish as Float64MultiArray
        array_msg = Float64MultiArray()
        flat_data = [coord for pt in points_3d for coord in (pt[0], pt[1], 0.0)]
        array_msg.data = flat_data
        
        array_msg.layout.dim.append(MultiArrayDimension(
            label="points", size=len(points_3d), stride=len(points_3d) * 3))
        array_msg.layout.dim.append(MultiArrayDimension(
            label="xyz", size=3, stride=3))
        
        self.point_publisher.publish(array_msg)
        self.get_logger().debug(f"Published {len(points_3d)} obstacle points")
    
    def _lookup_camera_height(self):
        """Get camera height from TF tree"""
        try:
            transform = self.tf_buffer.lookup_transform(
                'odom', 'camera_link', rclpy.time.Time())
            self.camera_height = transform.transform.translation.z
        except Exception as e:
            self.get_logger().warn(f'Could not get camera transform: {e}')

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