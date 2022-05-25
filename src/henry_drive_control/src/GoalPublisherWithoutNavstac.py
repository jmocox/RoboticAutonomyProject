#!/usr/bin/env python

import time

import numpy as np
import rospy

from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Joy
from tf.transformations import quaternion_from_euler, euler_from_quaternion



class goal_publisher_node():
    def __init__(self, hz=1, wait_time=5):
        rospy.init_node('goal_publisher_node')

        self.start = time.time()
        self.wait_time = wait_time
        self.rate = rospy.Rate(hz)

        self.e_stop = True

        self.current_ball_belief = None
        self.ball_belief_covariance_threshold = 10000
        self.current_robot_location = None
        
        
        rospy.Subscriber('/ball_belief', PoseWithCovarianceStamped, self.ball_belief_callback)
        rospy.Subscriber('/t265/odom/sample', Odometry, self.robot_odometry_callback)
        rospy.Subscriber('/joy', Joy, self.joystick_callback)

        self.pub = rospy.Publisher('/chassis/cmd_vel', Twist, queue_size=10)

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
            msg = Twist()
            msg.linear.x = 0
            msg.angular.z = 0
            self.pub.publish(msg)
            return
        
        
        if self.current_ball_belief is None or self.current_robot_location is None:
            return
        
        xr, yr = self.current_robot_location
        xt, yt = self.current_ball_belief

        dx, dy = xt - xr, yt - yr
        theta = np.arctan2(dy, dx)

        distance = np.sqrt(((yt - yr) ** 2) + ((xt - xr) ** 2))

        if distance < 0.3:
            msg = Twist()
            msg.linear.x = 0
            msg.angular.z = 0
            self.pub.publish(msg)

        

        throttle = 0
        if distance > 0 and distance < 0.75:
            throttle = distance - 0.3
        elif distance >= 0.75:
            throttle = 0.45
        
        turn_diff = theta - self.current_robot_theta
        turn_diff = self.correct_angle(turn_diff)

        turn_max = 0.1
        turn_throttle = max(min(turn_diff * 2, turn_max), -turn_max)
        

        msg = Twist()
        msg.linear.x = throttle
        msg.angular.z = turn_throttle
        self.pub.publish(msg)

    def correct_angle(self, a):
        return np.arctan2(np.sin(a), np.cos(a))
    
    def robot_odometry_callback(self, msg):
        self.current_robot_location = [msg.pose.pose.position.x, msg.pose.pose.position.y]
        self.current_robot_theta = euler_from_quaternion([
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        ])[2]
    
    def ball_belief_callback(self, msg):
        if np.trace(np.reshape(msg.pose.covariance, (6, 6))) < self.ball_belief_covariance_threshold:
            self.current_ball_belief = [msg.pose.pose.position.x, msg.pose.pose.position.y]
            print('Setting new ball belief goal')
        else :
            self.current_ball_belief = None
            print('Ball belief is to uncertain to publish goal')

if __name__ == '__main__':
    goal_publisher_node(wait_time=rospy.get_param('/wait_time')).run()
