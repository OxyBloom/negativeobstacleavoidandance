#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from action_msgs.srv import CancelGoal
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from nav2_simple_commander.robot_navigator import BasicNavigator

import time

class Replanner(Node):
    def __init__(self):
        super().__init__('replanner_node')

        # To receive latest RViz goal
        self.subscription = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.goal_callback,
            10
        )
        self.latest_goal = None
        self.latest_pose = None

        # Action client to resend goal
        self.nav_action_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')

        # CancelGoal service
        self.cancel_client = self.create_client(CancelGoal, '/navigate_to_pose/_action/cancel_goal')
        while not self.cancel_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for cancel_goal service...')

    def goal_callback(self, msg):
        self.latest_goal = msg.pose
        self.latest_pose = PoseStamped()
        # self.latest_pose.header = msg.header
        self.latest_pose.pose = self.latest_goal
        self.get_logger().info('Received and stored new goal from RViz.')

    def cancel_all_goals(self):
        from action_msgs.srv import CancelGoal
        req = CancelGoal.Request()
        # req.goal_info = []
        future = self.cancel_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result():
            self.get_logger().info("Cancelled all active goals.")
        else:
            self.get_logger().error("Failed to cancel goals.")

    def send_goal(self):
        if not self.latest_pose or not self.latest_goal
            self.get_logger().warn("No goal to send.")
            return

        # Update timestamp
        se
        self.latest_pose.header.stamp = self.get_clock().now().to_msg()

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = self.latest_goal.pose
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

        self.nav_action_client.wait_for_server()
        self.get_logger().info("Sending new goal to NavigateToPose...")
        future = self.nav_action_client.send_goal_async(goal_msg)
        
        rclpy.spin_until_future_complete(self, future)

        if future.result():
            self.get_logger().info('Goal accepted.')
        else:
            self.get_logger().error('Goal rejected.')

    def replan(self):
        self.get_logger().info("Starting replan process...")
        self.cancel_all_goals()
        time.sleep(1.0)  # Wait for cancel to take effect
        self.send_goal()

def main():
    rclpy.init()
    node = Replanner()

    # Wait a bit to receive RViz goal
    print("Waiting 10 seconds to receive a goal from RViz...")
    rclpy.spin_once(node, timeout_sec=10.0)

    # Trigger the replan
    node.replan()

    rclpy.shutdown()

if __name__ == '__main__':
    main()
