#!/usr/bin/env python

"""Detect balls in camera view

I used 'roslaunch realsense2_camera rs_camera.launch filters:=pointcloud' to
start the camera. Pretty sure pointcloud is not necessary for this particular
use case.
"""

import imutils
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

ball_colors = {
    'purple': {'lower': (110, 30, 60), 'upper': (150, 205, 205), 'bgr': (255, 0, 0)},
    'blue':   {'lower': (70, 150, 80), 'upper': (150, 255, 255), 'bgr': (255, 255, 0)},
}

class BallDetector:
    def __init__(self):
        self.bridge = CvBridge()
        
        rospy.init_node('ball_detector')
        
        self.ball_image_pub = rospy.Publisher('/ball_image', Image, queue_size=5)
        rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
        
        self.kernel = np.ones((5, 5), np.uint8)
        
    def run(self):
        while not rospy.is_shutdown():
            rospy.spin()
    
    def image_callback(self, image_msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, 'bgr8')
        except CvBridgeError as e:
            rospy.logerr(e)
            return
        
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        found_balls = False
        
        for color, thresholds in ball_colors.items():
            # use color to find ball candidates
            mask = cv2.inRange(hsv_image, thresholds['lower'], thresholds['upper'])
            smooth_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
            
            # use contours to filter noise
            contours = cv2.findContours(smooth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            better_contours = imutils.grab_contours(contours)
            sorted_contours = sorted(better_contours, key=cv2.contourArea, reverse=True)
            
            if len(sorted_contours) == 0:
                continue
            
            for i, contour in enumerate(sorted_contours):
                ((x, y), radius) = cv2.minEnclosingCircle(contour)
                M = cv2.moments(contour)
                center = (
                    int(M["m10"] / M["m00"]), 
                    int(M["m01"] / M["m00"])
                )
                if radius > 10:
                    cv2.circle(hsv_image, (int(x), int(y)), 
                               int(radius), thresholds['bgr'], 5)
                    cv2.circle(hsv_image, center, 5, (0, 0, 0), -1)
                    cv2.putText(hsv_image, '{0}.{1}'.format(color, i), 
                                (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 
                                1, (0,0,0), 2)
                    found_balls = True
                    
            
            
            
            #ball_img = cv2.bitwise_and(hsv_image, hsv_image, mask=smooth_mask)
            
            #if combined_balls is None:
            #    combined_balls = ball_img
            #else:
            #    combined_balls = cv2.bitwise_or(combined_balls, ball_img)
        
        if found_balls:
            bgr_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
            try:
                img_msg_out = self.bridge.cv2_to_imgmsg(bgr_image)
                self.ball_image_pub.publish(img_msg_out)
                
            except CvBridgeError as e:
                rospy.logerr(e)
  
if __name__ == '__main__':
    BallDetector().run()

