#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PointStamped,Point
from rclpy.duration import Duration
import math
import numpy as np
import copy
import tf2_ros
import tf2_geometry_msgs
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from action_msgs.msg import GoalInfo


class ScanMerger(Node):
    def __init__(self):
        super().__init__('scan_merger')

        self.declare_parameter('input_scan_topic', '/oscan')
        self.declare_parameter('output_scan_topic', '/scan')
        self.declare_parameter('target_frame', 'laser_link')

        self.input_topic = self.get_parameter('input_scan_topic').value
        self.output_topic = self.get_parameter('output_scan_topic').value
        self.target_frame = self.get_parameter('target_frame').value

        self.scan_sub = self.create_subscription(
            LaserScan,
            self.input_topic,
            self.scan_callback,
            10
        )

        self.point_sub = self.create_subscription(
            Float64MultiArray,
            'lpoints',
            self.lpoints_callback,
            10
        )

        self.scan_pub = self.create_publisher(
            LaserScan,
            self.output_topic,
            10
        )

        self.nav_monitor = self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback, 10)
        self.nav_action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')


        self.laser_points = []  # List of (x, y) points

        self.timer = self.create_timer(0.2,self.publish_laser)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.latest_scan = None
        self.merged_scan =None
        self.x,self.y = None,None
        self.current_goal = None
        self.get_logger().info("ScanMerger Node started.")

    # def publish_laser(self):
    #     self.get_logger().info(f"Recieved x: {self.x} y: {self.y}")
    #     if self.latest_scan is None :
    #         return

    #     try:
    #         if self.laser_points:
    #             self.merged_scan = copy.deepcopy(self.latest_scan)
    #             for x, y in self.laser_points:
    #                 self.inject_negative_obstacle(self.merged_scan, x, y)

    #             self.scan_pub.publish(self.merged_scan)
    #             # ✅ Clear the laser points so they aren't reused next time
    #             self.laser_points = []
    #         else:
    #             self.scan_pub.publish(self.latest_scan)
    #     except Exception as e:
    #         self.get_logger().warn(f"failedx: {e}")

    def goal_callback(self, msg):
        self.current_goal = msg

    def publish_laser(self):
        if self.latest_scan is None:
            return

        try:
            self.merged_scan = copy.deepcopy(self.latest_scan)

            if hasattr(self, 'laser_points') and self.laser_points:
                for x, y in self.laser_points:
                    self.inject_negative_obstacle(self.merged_scan, x, y)
                self.scan_pub.publish(self.merged_scan)
                #do nav stuffs here
                if self.current_goal is not None:
                    goal_pose = self.current_goal.pose
                    self.get_logger().info(f"Current goal: {goal_pose.position.x}, {goal_pose.position.y}")

                    # Example: Send a navigation action to the robot
                    if not self.nav_action_client.wait_for_server(timeout_sec=1.0):
                        self.get_logger().warn("Navigation action server not available")
                        return
                    
                    #cancel previous goal if needed
                    
                    active_goals = self.current_goal
                    self.get_logger().info(f"Active goals: {len(active_goals)}")
                    # Cancel any active goals before sending a new one

                    if active_goals:
                        print(f"Active goal(s) found: {len(active_goals)}")
                        for goal in active_goals:
                            # goal_info = GoalInfo(goal.goal_id, goal.goal)
                            self.cancel_goal(goal_info)
                    self.get_logger().info("Sending navigation goal...")
                    # Create and send the navigation goal
                    # Ensure the goal pose is in the correct frame
                    goal_pose.header.stamp = self.get_clock().now().to_msg()

                    goal_msg = NavigateToPose.Goal()
                    goal_msg.pose = goal_pose
                    self.nav_action_client.send_goal_async(goal_msg)
                # Clear the laser points after publishing
                self.laser_points = []
            else:
                self.scan_pub.publish(self.latest_scan)

        except Exception as e:
            self.get_logger().warn(f"failed in publish_laser: {e}")

    # def lpoints_callback(self, msg: Float64MultiArray):
    #     try:
    #         num_points = len(msg.data) // 3
    #         self.get_logger().info(f"Received {num_points} points")

    #         self.laser_points = []
    #         for i in range(0, len(msg.data), 3):
    #             x, y, _ = msg.data[i], msg.data[i+1], msg.data[i+2]
    #             self.laser_points.append((x, y))
    #     except Exception as e:
    #         self.get_logger().warn(f"Failed to process laser points: {e}")

    # def lpoints_callback(self, msg: Float64MultiArray):
    #     try:
    #         num_points = len(msg.data) 
    #         self.laser_points = []
    #         for i in range(num_points):
    #             x = msg.data[i]
    #             y = msg.data[i+1]
    #             self.laser_points.append((x, y))
    #         self.get_logger().info(f"Received {len(self.laser_points)} laser points")

    #     except Exception as e:
    #         self.get_logger().warn(f"Failed to process laser points: {e}")
    
    def cancel_goal(self, goal_info: GoalInfo):
        print(f"Cancelling goal: {goal_info.goal_id.uuid}")
        cancel_future = self.nav_action_client._cancel_goal_async
        # cancel_future = self.nav_action_client.cancel_goal_async(goal_info.goal_id)

        def done_callback(future):
            cancel_response = future.result()
            if cancel_response.return_code == 0:
                print("Goal successfully cancelled")
            else:
                print(f"Failed to cancel goal. Code: {cancel_response.return_code}")

        cancel_future.add_done_callback(done_callback)
        
    def lpoints_callback(self, msg: Float64MultiArray):
        try:
            num_points = len(msg.data) // 2
            self.laser_points = []

            for i in range(0, len(msg.data), 2):
                x = msg.data[i]
                y = msg.data[i + 1]

                # Create PointStamped in the source frame (e.g., "map")
                point_stamped = PointStamped()
                point_stamped.header.stamp = self.get_clock().now().to_msg()
                point_stamped.header.frame_id = "camera_link"  # ⚠️ Update to your actual point frame
                point_stamped.point.x = x
                point_stamped.point.y = y
                point_stamped.point.z = 0.0

                # Transform to laser frame
                try:
                    transform = self.tf_buffer.lookup_transform(
                        self.target_frame,           # to_frame
                        point_stamped.header.frame_id,  # from_frame
                        rclpy.time.Time(),           # use latest available
                        timeout=Duration(seconds=0.5)
                    )
                    transformed_point = tf2_geometry_msgs.do_transform_point(point_stamped, transform)
                    self.laser_points.append((transformed_point.point.x, transformed_point.point.y))
                except Exception as tf_e:
                    self.get_logger().warn(f"TF transform failed: {tf_e}")

            self.get_logger().info(f"Transformed {len(self.laser_points)} laser points to {self.target_frame}")

        except Exception as e:
            self.get_logger().warn(f"Failed to process laser points: {e}")




    def scan_callback(self, msg):
        self.latest_scan = copy.deepcopy(msg)

        # Deep copy to preserve original scan
        merged_scan = copy.deepcopy(msg)

        # Optionally inject stored negative obstacles here
        # Example: inject a test negative obstacle at (0.5, 0.2)
        # self.inject_negative_obstacle(merged_scan, 0.5, 0.2)
        # q= -1.0
        # for i in range(20):
        #     self.inject_negative_obstacle(merged_scan, 0.5, 0.2)
        #     self.scan_pub.publish(merged_scan)
        #     q-=0.1

    # def inject_negative_obstacle(self, scan_msg, x, y, width_degrees=2.0):
    #     """
    #     Injects a negative obstacle into the LaserScan ranges array over a short angular span.
    #     Ensures injection only happens in the front arc of the robot.
    #     """
    #     angle = math.atan2(y, x)
    #     range_val = math.hypot(x, y)

    #     # Only inject obstacles that are roughly in front of the robot
    #     FRONTAL_ANGLE_LIMIT = math.radians(30)
    #     if abs(angle) > FRONTAL_ANGLE_LIMIT:
    #         self.get_logger().debug(f"Skipping point ({x:.2f},{y:.2f}) with angle {math.degrees(angle):.1f}°")
    #         return

    #     if not scan_msg.angle_min <= angle <= scan_msg.angle_max:
    #         return

    #     center_index = int((angle - scan_msg.angle_min) / scan_msg.angle_increment)
    #     width_radians = math.radians(width_degrees)
    #     num_indices = int(width_radians / scan_msg.angle_increment)

    #     for offset in range(-num_indices // 2, num_indices // 2 + 1):
    #         index = center_index + offset
    #         if 0 <= index < len(scan_msg.ranges):
    #             if scan_msg.range_min < range_val < scan_msg.range_max:
    #                 current_range = scan_msg.ranges[index]
    #                 if math.isnan(current_range) or range_val < current_range:
    #                     scan_msg.ranges[index] = range_val


    # def inject_negative_obstacle(self, scan_msg, x, y):
    #     """
    #     Injects a negative obstacle point into the scan as if detected by LiDAR
    #     x, y are in the frame of `scan_msg.header.frame_id`
    #     """
    #     angle = math.atan2(y, x)
    #     range_val = math.hypot(x, y)
        

    #     if not scan_msg.angle_min <= angle <= scan_msg.angle_max:
    #         self.get_logger().debug(f"Point ({x:.2f},{y:.2f}) angle {angle:.2f} out of bounds.")
    #         return  # Outside scan FOV

    #     index = int((angle - scan_msg.angle_min) / scan_msg.angle_increment)

    #     # Check bounds
    #     if index < 0 or index >= len(scan_msg.ranges):
    #         self.get_logger().debug(f"Point ({x:.2f},{y:.2f}) index {index} out of range.")
    #         return
        
    #     if scan_msg.range_min < range_val < scan_msg.range_max:
    #         current_range = scan_msg.ranges[index]
    #         # self.get_logger().info(f"publishing...")
    #         if math.isnan(current_range) or range_val < current_range:

    #             scan_msg.ranges[index] = range_val
    #             self.get_logger().info(f"Injected at angle {angle:.2f}, range {range_val:.2f}")

    # def inject_negative_obstacle_from_point(self, point_stamped: PointStamped):
    #     """
    #     Transforms a point into the LaserScan frame and injects it
    #     """
    #     if self.latest_scan is None:
    #         return

    #     try:
    #         transform = self.tf_buffer.lookup_transform(
    #             self.latest_scan.header.frame_id,
    #             point_stamped.header.frame_id,
    #             rclpy.time.Time()
    #         )
    #         transformed_point = tf2_geometry_msgs.do_transform_point(point_stamped, transform)
    #         self.inject_negative_obstacle(self.latest_scan, transformed_point.point.x, transformed_point.point.y)
    #     except Exception as e:
    #         self.get_logger().warn(f"TF transform failed: {e}")

    def inject_negative_obstacle(self, scan_msg, x, y, length_across=0.3, num_points=2):
        """
        Inject a fake negative obstacle into a LaserScan message at position (x, y) in robot frame.
        The injection will simulate an obstacle of `length_across` meters wide in front of the robot.
        """
        import math

        angle_to_point = math.atan2(y, x)
        range_val = math.hypot(x, y)

        # Prevent division by zero or invalid injections
        if range_val <= 0.05:
            return

        # Compute angular width to span a fixed physical width
        width_radians = length_across / range_val
        start_angle = angle_to_point - width_radians / 2
        end_angle = angle_to_point + width_radians / 2
        angle_step = (end_angle - start_angle) / max(num_points - 1, 1)

        for i in range(num_points):
            injected_angle = start_angle + i * angle_step

            # Find index in scan ranges
            index = int((injected_angle - scan_msg.angle_min) / scan_msg.angle_increment)

            if 0 <= index < len(scan_msg.ranges):
                current_range = scan_msg.ranges[index]
                if math.isinf(current_range) or range_val < current_range:
                    scan_msg.ranges[index] = range_val


def main(args=None):
    rclpy.init(args=args)
    node = ScanMerger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()

if __name__ =='__main__':
    main()
