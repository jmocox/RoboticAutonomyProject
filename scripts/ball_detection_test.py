import time

import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits import mplot3d

from scripts.probability_utils import gauss2d, Space2d

thresh = {
    'Purple': {'lower': (113, 35, 40), 'upper': (145, 160, 240)},
    'Blue': {'lower': (95, 150, 80), 'upper': (100, 255, 250)},
    'Green': {'lower': (43, 60, 40), 'upper': (71, 240, 200)},
    'Yellow': {'lower': (19, 60, 100), 'upper': (23, 255, 255)},
    'Orange': {'lower': (11, 150, 100), 'upper': (16, 255, 250)},
}


def rbe(height, width, space, sensor_mean, sensor_sigma):
    belief = np.ones((height, width)) / (height * width)

    measurement_sigma = np.asarray(sensor_sigma)

    while True:
        x, y = yield belief

        

        measurement_mean = np.asarray(sensor_mean) + np.asarray([x, y])

        sensor_belief = gauss2d(space.xi, space.yi, measurement_mean, measurement_sigma)

        combined_belief = sensor_belief * belief
        belief = combined_belief / np.sum(combined_belief)



map_h, map_w = 100, 100
space = Space2d(-map_w, map_w, map_w * 2, -map_h, map_h, map_h * 2)
gen = rbe(map_w, map_w, space, sensor_mean=[0, 0], sensor_sigma=[[25, 0], [0, 25]])
_ = next(gen)

start = time.time()

colors = ['Purple', 'Blue', 'Green', 'Orange', 'Yellow']
for color in colors[:1]:
    print(color, time.time() - start)

    video_object = cv2.VideoCapture(f'../ball_training_data/{color}BallRecording.mp4')
    ret, frame = video_object.read()
    print(frame.shape[:2])
    j = 0
    while ret:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        blur = cv2.GaussianBlur(hsv, (11, 11), 0)
        mask = cv2.inRange(blur, thresh[color]['lower'], thresh[color]['upper'])

        kernel = np.ones((10, 10), np.uint8)

        smooth_mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours = cv2.findContours(smooth_mask_open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
                cv2.circle(frame, (int(x), int(y)),
                           int(radius), (1, 1, 1), 5)
                cv2.circle(frame, center, 5, (0, 0, 0), -1)
                cv2.putText(frame, '{0}.{1}'.format(color, i),
                            (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 0), 2)
                cur_belief = gen.send((x, y))
                ax = plt.axes()
                # ax.plot_surface(space.xi, space.yi, cur_belief, rstride=3, cstride=3, linewidth=1, antialiased=True,
                #                 cmap=cm.viridis)
                ax.pcolormesh(space.xi, space.yi, cur_belief, cmap='RdBu', vmin=0, vmax=np.max(cur_belief), shading='nearest')
                print(cur_belief, x, y)
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                plt.show()

            # break

        cv2.imshow(color, frame)
        cv2.imshow('mask', smooth_mask_open)

        wait = cv2.waitKey(5)
        if wait & 0xFF == ord('y'):
            pass
        elif wait & 0xFF == ord('n'):
            print(f'has errors {j}')
        elif wait & 0xFF == ord('q'):
            # break out of the while loop
            c = False
            break

        for _ in range(100):
            ret, frame = video_object.read()
        j += 1
