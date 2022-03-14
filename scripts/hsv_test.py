import cv2
import matplotlib.pyplot as plt
import numpy as np
import random


# img = cv2.imread('/home/team1/Desktop/RoboticAutonomyProject/image_negative_410.png')
#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#in_img = cv2.inRange(img, 90, 255)
#kernel = np.ones((5, 5), np.uint8)
#smooth_mask = cv2.morphologyEx(in_img, cv2.MORPH_OPEN, kernel)

#cv2.imshow('adsf', smooth_mask)
#cv2.waitKey(5000)

#quit()

video_object = cv2.VideoCapture('/dev/video4')
ret, frame = video_object.read()
height, width, _ = frame.shape

out_saver = cv2.VideoWriter('ball_stat_' + str(random.randint(1, 1000)) + '.mp4',cv2.VideoWriter_fourcc(*'MP4V'), 20, (width,height))

#cv2.imwrite('/home/team1/Desktop/RoboticAutonomyProject/image_negative_' + str(random.randint(1, 10000)) + '.png', frame)

px = 1 / plt.rcParams['figure.dpi']
fig = plt.figure(figsize=(width * px, height * px))

x = np.asarray(range(1, 256))
y_ball = np.zeros(255, np.float32)
y_everything_else = np.zeros(255, np.float32)

plt.ylim(0, 1)
line1, = plt.plot(x, y_ball, 'bo', label='Circle Interior')
line2, = plt.plot(x, y_everything_else, 'r+', label='Circle Exterior')

while ret:
    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)    
    out_saver.write(frame)
    cv2.imshow('vid', frame)
    ret, frame = video_object.read()
    if cv2.waitKey(10) & 0xFF == ord('q'):
        # break out of the while loop
        break
    continue
    
    test_center = (int(frame.shape[1] / 2), int(frame.shape[0] / 2))
    test_radius = int(min(frame.shape[:2]) / 4)

    test_mask = np.zeros((frame.shape[0], frame.shape[1], 1), np.uint8)
    test_mask = cv2.circle(test_mask, test_center, test_radius, (255), -1)

    color_ball = cv2.bitwise_and(frame, frame, mask=test_mask)
    color_everything_else = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(test_mask))

    hsv_ball = cv2.cvtColor(color_ball, cv2.COLOR_BGR2HSV)
    hsv_everything_else = cv2.cvtColor(color_everything_else, cv2.COLOR_BGR2HSV)

    hue_ball = [h for h in hsv_ball[:, :, 0].flatten() if h > 0]
    hue_everything_else = [h for h in hsv_everything_else[:, :, 0].flatten() if h > 0]
    
    hue_histogram_ball, _ = np.histogram(hue_ball, 255)
    hue_histogram_everything_else, _ = np.histogram(hue_everything_else, 255)

    scale = max(max(hue_histogram_ball), max(hue_histogram_everything_else))

    hue_histogram_ball = hue_histogram_ball.astype(np.float32) / scale
    hue_histogram_everything_else = hue_histogram_everything_else.astype(np.float32) / scale

    # line1.set_color()  # todo: change display color to best fit color
    line1.set_ydata(hue_histogram_ball)
    line2.set_ydata(hue_histogram_everything_else)
    plt.title('Hue Value Distributions')
    plt.xlabel('Hue Value')
    plt.ylabel('Relative Frequency')
    plt.legend()

    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    annotated_frame = cv2.circle(frame, test_center, test_radius, (255, 255, 255), 10)

    top_cat = np.concatenate((annotated_frame, hsv_ball), axis=1)
    bottom_cat = np.concatenate((hsv_everything_else, img), axis=1)
    full_image = np.concatenate((top_cat, bottom_cat), axis=0)

    cv2.imshow('HSV Test', full_image)

    ret, frame = video_object.read()

    if cv2.waitKey(10) & 0xFF == ord('q'):
        # break out of the while loop
        break
        
        
        
cap.release()
out.release()
cv2.destroyAllWindows()
