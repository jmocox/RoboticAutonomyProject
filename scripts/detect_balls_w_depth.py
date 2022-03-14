#!/usr/bin/env python

"""Detect balls in camera view

I used 'roslaunch realsense2_camera rs_camera.launch align_depth:=true' to
start the camera. 
"""
import json
import imutils
import rospy
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

ball_colors = {
    'purple': {'lower': (113, 35, 40), 'upper': (145, 160, 240)},
    # 'blue':   {'lower': (70, 150, 80), 'upper': (150, 255, 255)},
}

class BallDetector:
    def __init__(self):
        self.bridge = CvBridge()
        
        rospy.init_node('ball_detector')
        
        self.ball_image_pub = rospy.Publisher('/ball_image', Image, queue_size=5)
        self.ball_marker_pub = rospy.Publisher('/ball_marker', Marker, queue_size=5)
        rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
        rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.depth_callback)
        
        self.kernel = np.ones((5, 5), np.uint8)
        self.once = True
        
        self.last_depth = None
        
        self.save_index = 0
        self.save_distanceRadius = []
        self.save_directory = '/home/team1/Desktop/inverse_regression/'
        
        self.height, self.width = None, None   # pixels
        self.horizontal_half_angle = 0.436332  # radians
        self.vertical_half_angle = 0.375246  # radians
        
        
    def run(self):
        while not rospy.is_shutdown():
            rospy.spin()
    
    def squared_mean(self, distance, radius):
        return (radius - (6.37 + (38.33 / distance))) ** 2
    
    def tranform_and_publish_marker(self, pixel_x, pixel_y, distance):
        horizontal_theta = (((pixel_x * 2) / self.width) - 1) * self.horizontal_half_angle
        vertical_theta = -1 * (((pixel_y * 2) / self.height) - 1) * self.vertical_half_angle
        
        
        y = distance * np.cos(vertical_theta) * np.sin(horizontal_theta)
        x = distance * np.sin(vertical_theta) * np.cos(horizontal_theta)
        z = distance * np.cos(vertical_theta) * np.cos(horizontal_theta)
        
        marker = Marker()
        marker.header.frame_id = "/camera_link"
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.a = 1.0
        marker.color.r = 238.0 / 255.0
        marker.color.g = 130.0 / 255.0
        marker.color.b = 238.0 / 255.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = z
        marker.pose.position.y = -y
        marker.pose.position.z = x
        
        self.ball_marker_pub.publish(marker)
        
    
    def save(self, distance, radius):
        self.save_distanceRadius.append([distance, radius])
        self.save_index += 1
        print(self.save_index)
        if self.save_index % 500 == 0:
            with open('{0}data{1}.json'.format(self.save_directory, self.save_index), 'w') as f:
                json.dump(
                    self.save_distanceRadius, 
                    f
                )
        
            
    def depth_callback(self, image_msg):
        try:
            self.last_depth = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='passthrough')
        except CvBridgeError as e:
            rospy.logerr(e)
    
    def image_callback(self, image_msg):
        if self.last_depth is None:
            return
        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, 'bgr8')
        except CvBridgeError as e:
            rospy.logerr(e)
            return
        
        if self.width is None:
            self.height, self.width, _ = cv_image.shape
        
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
            
            sorted_contours = sorted_contours[:1]
            for i, contour in enumerate(sorted_contours):
                ((x, y), radius) = cv2.minEnclosingCircle(contour)
                M = cv2.moments(contour)
                center = (
                    int(M["m10"] / M["m00"]), 
                    int(M["m01"] / M["m00"])
                )
                ball_d = float(self.last_depth[int(y), int(x)]) / 1000
                if ball_d > 0.1:
                    error = self.squared_mean(ball_d, radius)
                    if radius > 10 and error < 200:
                        cv2.circle(hsv_image, (int(x), int(y)), 
                                   int(radius), (1, 1, 1), 5)
                        #cv2.circle(hsv_image, center, 5, (0, 0, 0), -1)
                        #text = '{0}.{1}'.format(color, i)
                        if True:
                            
                            self.tranform_and_publish_marker(x, y, ball_d)
                            #self.save(ball_d, radius)
                            text = '  {0} m  {1:.2f} error'.format(ball_d, error)
                            cv2.putText(hsv_image, text, 
                                    (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 
                                    1, (0,0,0), 2)
                            

                    
            
            
            
            #ball_img = cv2.bitwise_and(hsv_image, hsv_image, mask=smooth_mask)
            
            #if combined_balls is None:
            #    combined_balls = ball_img
            #else:
            #    combined_balls = cv2.bitwise_or(combined_balls, ball_img)
        
        # if found_balls:
        bgr_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        try:
            img_msg_out = self.bridge.cv2_to_imgmsg(bgr_image)
            self.ball_image_pub.publish(img_msg_out)
            
        except CvBridgeError as e:
            rospy.logerr(e)
  
if __name__ == '__main__':
    BallDetector().run()

