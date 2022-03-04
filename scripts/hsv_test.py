import cv2
import matplotlib.pyplot as plt
import numpy as np

video_object = cv2.VideoCapture(0)
ret, frame = video_object.read()
height, width, _ = frame.shape

px = 1 / plt.rcParams['figure.dpi']
fig = plt.figure(figsize=(width * px, height * px))

x = np.asarray(range(1, 256))
y_ball = np.zeros(255, np.float32)
y_everything_else = np.ones(255, np.float32)

line1, = plt.plot(x, y_ball, 'bo', label='Circle Interior')
line2, = plt.plot(x, y_everything_else, 'r+', label='Circle Exterior')

while ret:
    test_center = (int(frame.shape[1] / 2), int(frame.shape[0] / 2))
    test_radius = int(min(frame.shape[:2]) / 4)

    test_mask = np.zeros((frame.shape[0], frame.shape[1], 1), np.uint8)
    test_mask = cv2.circle(test_mask, test_center, test_radius, (255), -1)

    color_ball = cv2.bitwise_and(frame, frame, mask=test_mask)
    color_everything_else = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(test_mask))

    hsv_ball = cv2.cvtColor(color_ball, cv2.COLOR_BGR2HSV)
    hsv_everything_else = cv2.cvtColor(color_everything_else, cv2.COLOR_BGR2HSV)

    hue_ball = hsv_ball[:, :, 0]
    hue_everything_else = hsv_everything_else[:, :, 0]

    hue_histogram_ball, _ = np.histogram(hue_ball, 256)
    hue_histogram_everything_else, _ = np.histogram(hue_everything_else, 256)

    hue_histogram_ball = hue_histogram_ball[1:]
    hue_histogram_everything_else = hue_histogram_everything_else[1:]

    scale = max(max(hue_histogram_ball), max(hue_histogram_everything_else))

    hue_histogram_ball = hue_histogram_ball / scale
    hue_histogram_everything_else = hue_histogram_everything_else / scale

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
