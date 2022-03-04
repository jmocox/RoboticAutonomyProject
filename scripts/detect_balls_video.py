#!/usr/bin/env python

"""Detect balls in camera view

I used 'roslaunch realsense2_camera rs_camera.launch filters:=pointcloud' to
start the camera. Pretty sure pointcloud is not necessary for this particular
use case.
"""

import imutils
import cv2
import numpy as np
from sklearn.mixture import GaussianMixture

random_state = np.random.RandomState(seed=1)

ball_colors = {
    'purple': {'lower': (110, 30, 60), 'upper': (150, 205, 205)},
    #'blue': {'lower': (70, 150, 100), 'upper': (150, 255, 255)},
    'green': {'lower': (45, 30, 60), 'upper': (75, 255, 205)},
}

kernel = np.ones((5, 5), np.uint8)

# purple = cv2.VideoCapture('../ball_training_data/{0}BallRecording.mp4'.format('Purple'))
purple = cv2.VideoCapture(0)
ret, frame = purple.read()
i = 0
while ret:
    i += 1
    print(i)

    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    found_balls = False

    for color, thresholds in ball_colors.items():
        # use color to find ball candidates
        mask = cv2.inRange(hsv_image, thresholds['lower'], thresholds['upper'])
        smooth_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # use contours to filter noise
        contours = cv2.findContours(smooth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        better_contours = imutils.grab_contours(contours)
        sorted_contours = sorted(better_contours, key=cv2.contourArea, reverse=True)

        for i, contour in enumerate(sorted_contours):
            ((x, y), radius) = cv2.minEnclosingCircle(contour)
            M = cv2.moments(contour)
            center = (
                int(M["m10"] / M["m00"]),
                int(M["m01"] / M["m00"])
            )
            if radius > 10:
                cv2.circle(hsv_image, (int(x), int(y)), int(radius), (0, 0, 0), 5)
                cv2.circle(hsv_image, center, 5, (0, 0, 0), -1)
                cv2.putText(hsv_image, '{0}.{1}'.format(color, i),
                            (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 0), 2)
                found_balls = True

    out = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    cv2.imshow('purple', out)

    ret, frame = purple.read()
    if cv2.waitKey(10) & 0xFF == ord('q'):
        # break out of the while loop
        c = False
        break
