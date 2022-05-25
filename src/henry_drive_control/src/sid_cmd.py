#!/usr/bin/env python
import rospy
from geometry_msgs.msg import Twist 
from sensor_msgs.msg import Joy

running = False

def joystick_callback(msg):
    global running

    # (a) start the run
    if msg.buttons[0]:
        running = True
    
    # (b) stop the run
    if msg.buttons[1]:
        running = False
    
    print('preseed joy stock', running)

if __name__=='__main__':
    

    rospy.init_node('fixed_cmdvel')
    pub = rospy.Publisher('/chassis/cmd_vel', Twist, queue_size=10)
    rospy.Subscriber('/joy', Joy, joystick_callback)
    
    

    rate = rospy.Rate(20.0)
    dT = rospy.Duration(secs=20)
    while not rospy.is_shutdown():

        if running:
            t_init = rospy.get_rostime()

            while not rospy.is_shutdown() and running:
                t_now = rospy.get_rostime()
                if (t_now - t_init < dT):
                    msg = Twist()
                    msg.linear.x = 0.2
                    msg.angular.z = 0
                    pub.publish(msg)
                else:
                    running = False
                    msg = Twist()
                    msg.linear.x = 0
                    msg.angular.z = 0
                    print('I shall stop now.')
                    pub.publish(msg)
                
                rate.sleep()
        
        else:
            msg = Twist()
            msg.linear.x = 0
            msg.angular.z = 0
            print('I have stopped now.')
            pub.publish(msg)

        rate.sleep()
        

        