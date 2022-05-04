#!/usr/bin/env python

import time

import numpy as np
import rospy

import actionlib
from geometry_msgs.msg import PoseWithCovarianceStamped
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Joy
from tf.transformations import quaternion_from_euler



class goal_publisher_node():
    def __init__(self, hz=1, wait_time=5):
        rospy.init_node('goal_publisher_node')

        self.start = time.time()
        self.wait_time = wait_time
        self.rate = rospy.Rate(hz)

        self.e_stop = False

        self.current_ball_belief = None
        self.ball_belief_covariance_threshold = 10000
        self.current_robot_location = None
        
        
        rospy.Subscriber('/ball_belief', PoseWithCovarianceStamped, self.ball_belief_callback)
        rospy.Subscriber('/t265/odom/sample', Odometry, self.robot_odometry_callback)
        rospy.Subscriber('/joy', Joy, self.joystick_callback)

        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.client.wait_for_server()

    def run(self):
        while not rospy.is_shutdown():
            if time.time() - self.start > self.wait_time:
                self.publish_goal()
            print('wait time!!!!!!!!!!', self.wait_time, 'diff', time.time() - self.start)
            self.rate.sleep()
    
    def joystick_callback(self, msg):
        # (b) is emergency stop
        self.e_stop = bool(msg.buttons[1])

        # (y) resets the 5 second count down
        if msg.buttons[3]:
            self.start = time.time()

    def publish_goal(self):
        if self.e_stop:
            self.client.cancel_all_goals()
            return
        
        
        if self.current_ball_belief is None or self.current_robot_location is None:
            return
        
        xr, yr = self.current_robot_location
        xt, yt = self.current_ball_belief

        dx, dy = xt - xr, yt - yr
        theta = np.arctan2(dy, dx)

        distance = np.sqrt(((yt - yr) ** 2) + ((xt - xr) ** 2))
        if distance < 0.3:
            self.client.cancel_all_goals()
            return

        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()

        goal.target_pose.pose.position.x = xt
        goal.target_pose.pose.position.y = yt

        x, y, z, w = quaternion_from_euler(0, 0, theta)
        goal.target_pose.pose.orientation.x = x
        goal.target_pose.pose.orientation.y = y
        goal.target_pose.pose.orientation.z = z
        goal.target_pose.pose.orientation.w = w

        self.client.send_goal(goal)
    
    def robot_odometry_callback(self, msg):
        self.current_robot_location = [msg.pose.pose.position.x, msg.pose.pose.position.y]
    
    def ball_belief_callback(self, msg):
        if np.trace(np.reshape(msg.pose.covariance, (6, 6))) < self.ball_belief_covariance_threshold:
            self.current_ball_belief = [msg.pose.pose.position.x, msg.pose.pose.position.y]
            print('Setting new ball belief goal')
        else :
            self.current_ball_belief = None
            print('Ball belief is to uncertain to publish goal')

if __name__ == '__main__':
    goal_publisher_node(wait_time=rospy.get_param('/wait_time')).run()
