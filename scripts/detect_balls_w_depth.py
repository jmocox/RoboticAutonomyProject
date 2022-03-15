#!/usr/bin/env python

"""Detect balls in camera view

I used 'roslaunch realsense2_camera rs_camera.launch align_depth:=true' to
start the camera. 
"""
# import json
import imutils
import rospy
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

ball_hsv_thresholds = {
    'Purple': {'lower': (113, 35, 40), 'upper': (145, 160, 240)},
    'Blue': {'lower': (95, 150, 80), 'upper': (100, 255, 250)},
    'Green': {'lower': (43, 60, 40), 'upper': (71, 240, 200)},
    'Yellow': {'lower': (19, 60, 100), 'upper': (23, 255, 255)},
    'Orange': {'lower': (11, 150, 100), 'upper': (16, 255, 250)},
}

marker_rgb_colors = {
    'Purple': (238, 130, 238),
    'Blue': (0, 0, 255),
    'Green': (0, 255, 0),
    'Yellow': (255, 255, 0),
    'Orange': (25, 140, 0),
}
for c in marker_rgb_colors.keys():
    r, g, b = marker_rgb_colors[c]
    marker_rgb_colors[c] = (r / 255, g / 255, b / 255)


class BallDetector:
    def __init__(self):
        rospy.init_node('ball_detector')

        self.bridge = CvBridge()

        self.ball_image_pub = rospy.Publisher('/ball_image', Image, queue_size=5)
        self.ball_marker_pubs = {c: rospy.Publisher(f'/ball_marker/{c}', Marker, queue_size=5) for c in
                                 ball_hsv_thresholds.keys()}
        self.ball_marker_pub = rospy.Publisher('/ball_marker', Marker, queue_size=5)
        rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
        rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.depth_callback)

        self.kernel = np.ones((5, 5), np.uint8)

        self.last_depth = None

        # self.save_index = 0
        # self.save_distanceRadius = []
        # self.save_directory = '/home/team1/Desktop/inverse_regression/'

        self.height, self.width = None, None  # pixels
        self.horizontal_half_angle = 0.436332  # radians
        self.vertical_half_angle = 0.375246  # radians

    def run(self):
        while not rospy.is_shutdown():
            rospy.spin()

    def squared_mean(self, distance, radius):
        return (radius - (6.37 + (38.33 / distance))) ** 2

    def transform_and_publish_marker(self, pixel_x, pixel_y, distance, color, diameter=0.15):
        horizontal_theta = (((pixel_x * 2) / self.width) - 1) * self.horizontal_half_angle
        vertical_theta = -1 * (((pixel_y * 2) / self.height) - 1) * self.vertical_half_angle

        y = distance * np.cos(vertical_theta) * np.sin(horizontal_theta)
        x = distance * np.sin(vertical_theta) * np.cos(horizontal_theta)
        z = distance * np.cos(vertical_theta) * np.cos(horizontal_theta)

        r, g, b = marker_rgb_colors[color]

        marker = Marker()
        marker.header.frame_id = '/camera_link'
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.ns = f'{color}Ball'
        marker.id = 0
        marker.scale.x = diameter
        marker.scale.y = diameter
        marker.scale.z = diameter
        marker.color.a = 1.0
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = z
        marker.pose.position.y = -y
        marker.pose.position.z = x

        self.ball_marker_pubs[color].publish(marker)

    # def save(self, distance, radius):
    #     self.save_distanceRadius.append([distance, radius])
    #     self.save_index += 1
    #     print(self.save_index)
    #     if self.save_index % 500 == 0:
    #         with open('{0}data{1}.json'.format(self.save_directory, self.save_index), 'w') as f:
    #             json.dump(
    #                 self.save_distanceRadius,
    #                 f
    #             )

    def depth_callback(self, image_msg):
        try:
            self.last_depth = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='passthrough')
        except CvBridgeError as e:
            rospy.logerr(e)
            print(e)

    def image_callback(self, image_msg):
        if self.last_depth is None:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, 'bgr8')
        except CvBridgeError as e:
            rospy.logerr(e)
            print(e)
            return

        if self.width is None:
            self.height, self.width, _ = cv_image.shape

        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        for color, thresholds in ball_hsv_thresholds.items():
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
                ball_d = float(self.last_depth[int(y), int(x)]) / 1000
                if ball_d > 0.1:
                    error = self.squared_mean(ball_d, radius)
                    if radius > 10 and error < 200:
                        self.transform_and_publish_marker(x, y, ball_d, color)

                        cv2.circle(hsv_image, (int(x), int(y)), int(radius), (1, 1, 1), 5)
                        text = '  {0} m  {1:.2f} error'.format(ball_d, error)
                        cv2.putText(hsv_image, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        bgr_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        try:
            img_msg_out = self.bridge.cv2_to_imgmsg(bgr_image)
            self.ball_image_pub.publish(img_msg_out)

        except CvBridgeError as e:
            rospy.logerr(e)


if __name__ == '__main__':
    BallDetector().run()
