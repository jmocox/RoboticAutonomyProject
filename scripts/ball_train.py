import cv2
import matplotlib.pyplot as plt
import numpy as np
import random

info = {
    'green':   {'in_c': (351, 122), 'in_r': 80, 'len': 4513},
    'blue':    {'in_c': (356, 122), 'in_r': 80, 'len': 4638},
    'purple':  {'in_c': (355, 125), 'in_r': 80, 'len': 5242},
    'orange':  {'in_c': (355, 120), 'in_r': 80, 'len': 4638},
    'yellow':  {'in_c': (349, 120), 'in_r': 79, 'len': 4528},
}

height, width = None, None

for color in info.keys():
    print(color)
    choices = []
    for _ in range(10):
        choice = random.randint(1, info[color]['len'])
        while choice in choices:
            choice = random.randint(1, info[color]['len'])
        choices.append(choice)
    
    video_object = cv2.VideoCapture(
        '/home/team1/Desktop/{0}BallRecording.mp4'.format(color[0].upper() + color[1:])
    )
    ret, frame = video_object.read()
    height, width, _ = frame.shape
    frames = []
    counter = 1
    while ret:
        
        if counter in choices:
            frames.append(frame)
        counter += 1
        ret, frame = video_object.read()
        
    video_object.release()
    
    info[color]['frames'] = frames

print(len(repr(info)))

for color in info.keys():
    data_interior = [0 for _ in range(255)]
    data_exterior = [0 for _ in range(255)]
    
    test_mask = np.zeros((height, width, 1), np.uint8)
    test_mask = cv2.circle(test_mask, info[color]['in_c'], info[color]['in_r'], (255), -1)

    every_mask = np.zeros((height, width, 1), np.uint8)
    every_mask = cv2.bitwise_not(cv2.circle(every_mask, info[color]['in_c'], info[color]['in_r'] + 5, (255), -1))

    j = 0
    for frame in info[color]['frames']:
        j += 1
        print(j, 'out of', len(info[color]['frames']))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        hsv_ball = cv2.bitwise_and(hsv, hsv, mask=test_mask)
        hsv_everything_else = cv2.bitwise_and(hsv, hsv, mask=every_mask)
        
        hue_ball = [h for h in hsv_ball[:, :, 0].flatten() if h > 0]
        hue_everything_else = [h for h in hsv_everything_else[:, :, 0].flatten() if h > 0]
        
        hue_histogram_ball, _ = np.histogram(hue_ball, 255)
        hue_histogram_everything_else, _ = np.histogram(hue_everything_else, 255)
        
        for i in range(255):
            data_interior[i] += hue_histogram_ball[i]
            data_exterior[i] += hue_histogram_everything_else[i]
    
    y_interior = np.asarray(data_interior).astype(np.float32)
    y_exterior = np.asarray(data_exterior).astype(np.float32)
    
    #scale = sum(y_interior) + sum(y_exterior)
    
    y_interior = y_interior / max(y_interior)
    y_exterior = y_exterior / max(y_exterior)
    
    px = 1 / plt.rcParams['figure.dpi']
    fig = plt.figure(figsize=(width * px, height * px))

    x = np.asarray(range(1, 256))

    plt.ylim(0, 1)
    plt.xlim(1, 255)
    line1, = plt.plot(x, y_interior, 'bo-', label='Circle Interior')
    line2, = plt.plot(x, y_exterior, 'r+', label='Circle Exterior')
    
    line1.set_color(color)
    plt.show()
    
    

quit()
color = 'purple'

video_object = cv2.VideoCapture(
    '/home/team1/Desktop/{0}BallRecording.mp4'.format(color[0].upper() + color[1:])
)
frames = []
ret, frame = video_object.read()
height, width, _ = frame.shape
while ret:
    frames.append(frame)
    ret, frame = video_object.read()
video_object.release()

print(color, len(frames))

data_interior = [0 for _ in range(255)]
data_exterior = [0 for _ in range(255)]

test_mask = np.zeros((height, width, 1), np.uint8)
test_mask = cv2.circle(test_mask, info[color]['in_c'], info[color]['in_r'], (255), -1)

every_mask = np.zeros((height, width, 1), np.uint8)
every_mask = cv2.bitwise_not(cv2.circle(every_mask, info[color]['in_c'], info[color]['in_r'] + 5, (255), -1))

j = 0
for frame in frames:
    j += 1
    if j > 100:
        break
    print(j, 'out of', len(frames))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    hsv_ball = cv2.bitwise_and(hsv, hsv, mask=test_mask)
    hsv_everything_else = cv2.bitwise_and(hsv, hsv, mask=every_mask)
    
    hue_ball = [h for h in hsv_ball[:, :, 0].flatten() if h > 0]
    hue_everything_else = [h for h in hsv_everything_else[:, :, 0].flatten() if h > 0]
    
    hue_histogram_ball, _ = np.histogram(hue_ball, 255)
    hue_histogram_everything_else, _ = np.histogram(hue_everything_else, 255)
    
    for i in range(255):
        data_interior[i] += hue_histogram_ball[i]
        data_exterior[i] += hue_histogram_everything_else[i]


px = 1 / plt.rcParams['figure.dpi']
fig = plt.figure(figsize=(width * px, height * px))

x = np.asarray(range(1, 256))

plt.ylim(0, max(max(data_exterior), max(data_interior)))
plt.xlim(1, 255)
line1, = plt.plot(x, data_interior, 'bo-', label='Circle Interior')
line2, = plt.plot(x, data_exterior, 'r+', label='Circle Exterior')
plt.show()

quit()

px = 1 / plt.rcParams['figure.dpi']
fig = plt.figure(figsize=(width * px, height * px))

x = np.asarray(range(1, 256))
y_ball = np.zeros(255, np.float32)
y_everything_else = np.zeros(255, np.float32)

plt.ylim(0, 1)
plt.xlim(1, 255)
line1, = plt.plot(x, y_ball, 'bo-', label='Circle Interior')
line2, = plt.plot(x, y_everything_else, 'r+', label='Circle Exterior')

plots = []
j = 0
for frame in frames:
    print(j, 'out of', len(frames))
    j += 1
    if j > 50:
        break
    test_mask = np.zeros((frame.shape[0], frame.shape[1], 1), np.uint8)
    test_mask = cv2.circle(test_mask, info[color]['in_c'], info[color]['in_r'], (255), -1)
    color_ball = cv2.bitwise_and(frame, frame, mask=test_mask)
    
    every_mask = np.zeros((frame.shape[0], frame.shape[1], 1), np.uint8)
    every_mask = cv2.circle(test_mask, info[color]['in_c'], info[color]['in_r'] + 5, (255), -1)
    color_everything_else = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(every_mask))
    
    hsv_ball = cv2.cvtColor(color_ball, cv2.COLOR_BGR2HSV)
    hsv_everything_else = cv2.cvtColor(color_everything_else, cv2.COLOR_BGR2HSV)

    hue_ball = [h for h in hsv_ball[:, :, 0].flatten() if h > 0]
    hue_everything_else = [h for h in hsv_everything_else[:, :, 0].flatten() if h > 0]
    
    hue_histogram_ball, _ = np.histogram(hue_ball, 255)
    hue_histogram_everything_else, _ = np.histogram(hue_everything_else, 255)

    scale = max(max(hue_histogram_ball), max(hue_histogram_everything_else))

    hue_histogram_ball = hue_histogram_ball.astype(np.float32) / scale
    hue_histogram_everything_else = hue_histogram_everything_else.astype(np.float32) / scale
    
    line1.set_color(color)
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
    plots.append(img)

print('plots', len(plots))

out_saver = cv2.VideoWriter('/home/team1/Desktop/' + color + '_stacked_plot0.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 40, (width,height*2))

for i in range(len(plots)):
    circled = cv2.circle(frames[i], info[color]['in_c'], info[color]['in_r'], (255, 255, 255), 1)
    thing = np.concatenate((frames[i], plots[i]), axis=0)
    out_saver.write(thing)
    
out_saver.release()


quit()

tester = 'orange'



ret, frame = video_object.read()
while ret:
    circled = cv2.circle(frame, info[tester]['in_c'], info[tester]['in_r'], (255, 255, 255), 1)
    cv2.imshow(tester, circled)
    
    ret, frame = video_object.read()
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
quit()
circled = cv2.circle(frame, info[tester]['in_c'], info[tester]['in_r'], (255, 255, 255), 1)
#circled = cv2.circle(circled, info[tester]['ex_c'], info[tester]['ex_r'], (255, 255, 255), 1)

cv2.imshow(tester, circled)
cv2.waitKey(10000)
quit()
ret, frame = video_object.read()
cv2.imshow(tester, frame)
print(1)

video_object.release()
cv2.destroyAllWindows()
quit()

video_object = cv2.VideoCapture('/home/team1/Desktop/GreenBallRecording.mp4')
hard_coded_center = (349, 120)
hard_coded_radius = 79

ret, frame = video_object.read()
"""
height, width, _ = frame.shape

px = 1 / plt.rcParams['figure.dpi']
fig = plt.figure(figsize=(width * px, height * px))

x = np.asarray(range(1, 256))
y_ball = np.zeros(255, np.float32)
y_everything_else = np.zeros(255, np.float32)

plt.ylim(0, 1)
plt.xlim(1, 255)
line1, = plt.plot(x, y_ball, 'bo', label='Circle Interior')
line2, = plt.plot(x, y_everything_else, 'r+', label='Circle Exterior')
"""

while ret:

    test_mask = np.zeros((frame.shape[0], frame.shape[1], 1), np.uint8)
    test_mask = cv2.circle(test_mask, hard_coded_center, hard_coded_radius, (255), -1)
    color_ball = cv2.bitwise_and(frame, frame, mask=test_mask)
    
    every_mask = np.zeros((frame.shape[0], frame.shape[1], 1), np.uint8)
    every_mask = cv2.circle(test_mask, hard_coded_center, hard_coded_radius + 10, (255), -1)
    color_everything_else = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(every_mask))
    
    hsv_ball = cv2.cvtColor(color_ball, cv2.COLOR_BGR2HSV)
    hsv_everything_else = cv2.cvtColor(color_everything_else, cv2.COLOR_BGR2HSV)

    hue_ball = [h for h in hsv_ball[:, :, 0].flatten() if h > 0]
    hue_everything_else = [h for h in hsv_everything_else[:, :, 0].flatten() if h > 0]
    
    hue_histogram_ball, _ = np.histogram(hue_ball, 255)
    hue_histogram_everything_else, _ = np.histogram(hue_everything_else, 255)

    scale = max(max(hue_histogram_ball), max(hue_histogram_everything_else))

    hue_histogram_ball = hue_histogram_ball.astype(np.float32) / scale
    hue_histogram_everything_else = hue_histogram_everything_else.astype(np.float32) / scale
    
    line1.set_color('green')
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
    
    circled = cv2.circle(frame, hard_coded_center, hard_coded_radius, (255, 255, 255), 1)
    
    top_cat = np.concatenate((circled, color_ball), axis=1)
    bottom_cat = np.concatenate((color_everything_else, img), axis=1)
    full_image = np.concatenate((top_cat, bottom_cat), axis=0)

    cv2.imshow('vid', full_image)
    
    ret, frame = video_object.read()
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

