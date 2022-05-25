#!/usr/bin/env python
import rospy
import time
from geometry_msgs.msg import Twist, Pose
#from sensor_msgs.msg import Joy
#from std_msgs.msg import Float32


# Author: Jeronimo 
# This ROS Node converts Joystick inputs from the joy node
# into commands for turtlesim

# Receives joystick messages (subscribed to Joy topic)
# then converts the joysick inputs into Twist commands
# axis 1 aka left stick vertical controls linear speed
# axis 4 aka right stick vertical controls angular speed
def callback(data):
    twist = Twist()
    # vertical left stick axis = linear rate
    #twist.linear.x = 4*data.axes[1]
    omega_left = data.axes[1]
    
    # horizontal left stick axis = turn rate
    #twist.angular.z = 4*data.axes[4]
    omega_right = data.axes[4]
    #x_dot = 
    twist.angular.z = omega_left-omega_right
    twist.linear.x = (omega_left+omega_right)/2

    pub.publish(twist)
    #pub2.publish(omega_right)

# Intializes everything
def start():
    # publishing to "turtle1/cmd_vel" to control turtle1
    global pub#1, pub2
    
    ### THIS PUBLISHES MOTION FROM JOYSTICK
    #pub = rospy.Publisher('chassis/cmd_vel', Twist, queue_size=5)

    ### THIS PUBLISHES A POSE GIVEN CONTROL INPUTS
    pub = rospy.Publisher('chassis/pose', Pose, callback)
    #pub2 = rospy.Publisher('chassis/right_wheels', Float32)

    # subscribed to Pose Commands Produced by Nav Stack
    rospy.Subscriber("chassis/cmd_vel", Twist, callback)
    # starts the node
    rospy.init_node('Joy2Nook')
    rospy.spin()

if __name__ == '__main__':
    start = time.perf_counter()
    while (not rospy.is_shutdown()):
        toc = time.perf_counter()
        if (toc-start>5): ## Set time here
            start()

