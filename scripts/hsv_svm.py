import time

import imutils
import cv2
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt

def main():
    start = time.time()

    kernel = np.ones((10, 10), np.uint8)

    for color in ['Purple', 'Blue', 'Green', 'Orange', 'Yellow'][:1]:
        video_object = cv2.VideoCapture('../ball_training_data/{0}BallRecording.mp4'.format(color))
        ret, frame = video_object.read()

        i, c = 0, True
        while ret and c:
            print(i, time.time() - start)
            i += 1

            if i < 100:
                ret, frame = video_object.read()
                continue

            # if i % 10 > 0:
            #     ret, frame = video_object.read()
            #     continue

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            ball_mask = cv2.inRange(hsv, (110, 50, 0), (160, 255, 255))
            ball_mask_smooth = cv2.morphologyEx(ball_mask, cv2.MORPH_OPEN, kernel)

            contours = cv2.findContours(ball_mask_smooth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            better_contours = imutils.grab_contours(contours)
            sorted_contours = sorted(better_contours, key=cv2.contourArea, reverse=True)

            for j, contour in enumerate(sorted_contours):
                ((x, y), radius) = cv2.minEnclosingCircle(contour)
                M = cv2.moments(contour)
                center = (
                    int(M["m10"] / M["m00"]),
                    int(M["m01"] / M["m00"])
                )
                if radius > 10:
                    cv2.circle(frame, (int(x), int(y)), int(radius), (0, 0, 0), 5)
                    cv2.circle(frame, center, 5, (0, 0, 0), -1)
                    cv2.putText(frame, '{0}.{1}'.format(color, j),
                                (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 0, 0), 2)


            cv2.imshow(color, frame)
            cv2.imshow('ball', ball_mask_smooth)

            ret, frame = video_object.read()
            if cv2.waitKey(70) & 0xFF == ord('q'):
                # break out of the while loop
                c = False
                break

def plot_hue_frequencies():
    start = time.time()

    background_mask = cv2.imread('../ball_training_data/background_mask.png')[:, :, 0].flatten()
    ball_mask = cv2.imread('../ball_training_data/ball_mask.png')[:, :, 0].flatten()
    for color in ['Purple', 'Blue', 'Green', 'Orange', 'Yellow']:
        video_object = cv2.VideoCapture('../ball_training_data/{0}BallRecording.mp4'.format(color))
        ret, frame = video_object.read()

        hues_background = np.zeros(180, np.uint64)
        hues_ball = np.zeros(180, np.uint64)

        i, c = 0, True
        while ret and c:
            print(i, time.time() - start)
            i += 1

            if i < 100:
                ret, frame = video_object.read()
                continue

            if i % 50 > 0:
                ret, frame = video_object.read()
                continue
            if i  > 400:
                break

            # if i > 200:
            #     break

            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hues = hsv_frame[:, :, 0].flatten()
            for hb in hues[background_mask > 0]:
                hues_background[hb] += 1
            for hb in hues[ball_mask > 0]:
                hues_ball[hb] += 1

            # cv2.imshow(color, frame)
            # cv2.imshow('background', cv2.bitwise_and(background_mask, frame))
            # cv2.imshow('ball', cv2.bitwise_and(ball_mask, frame))

            ret, frame = video_object.read()
            # if cv2.waitKey() & 0xFF == ord('q'):
            #     # break out of the while loop
            #     c = False
            #     break

        hues_background_normalized = hues_background / sum(hues_background)
        hues_ball_normalized = hues_ball / sum(hues_ball)
        combined = hues_background + hues_ball
        combined_normalized = combined / sum(combined)

        x = np.arange(180)

        fig, (ax0, ax1, ax2) = plt.subplots(3, 1)
        ax0.bar(x, hues_background_normalized)
        ax1.bar(x, hues_ball_normalized)
        ax2.bar(x, combined_normalized)

        ax0.set_ylabel('Background')
        ax1.set_ylabel(f'{color} Ball')
        ax2.set_ylabel('Combined')

        ax0.set_title(f'Relative Hue Frequencies ({color})')

        ax2.set_xlabel(f'Hue Values (0 to 179 degrees)')

        plt.show()

        # h_value = np.arange(180)
        # plt.plot()

        # quit()


# def get_foreground_mask(frames):
#     vars = np.var(frames, axis=0)
#     vars_mono = np.sum(vars, axis=2)
#
#     vars_mono /= np.max(vars_mono)
#     vars_mono *= 256
#
#     cv2.imshow('mono', vars_mono)

# variances = np.ndarray((frames[0].shape[0], frames[0].shape[1], 3)).astype(np.float32)
# min_a, max_a = 255, 0
# for y in range(variances.shape[0]):
#     for x in range(variances.shape[1]):
#         pixels = [frame[y, x, :] for frame in frames]
#         var = np.var(pixels)
#         min_a = min(min_a, var)
#         max_a = max(max_a, var)
#         variances[y, x, :] = [var, var, var]
#
# print(min_a, max_a)
# print(np.min(variances), np.max(variances))
# print(variances, variances.shape)


# ball_mask = cv2.imread('../tmp/ball_mask.png')
# grey = cv2.cvtColor(ball_mask, cv2.COLOR_BGR2GRAY)
# _, mask = cv2.threshold(grey, 200, 255, cv2.THRESH_BINARY)
# g = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
#
# cv2.imwrite('../ball_training_data/ball_mask.png', g)
# cv2.imshow('asdagf', g)
# cv2.waitKey(10000)
# quit()
#
# for i in range(7, 43):
#     img = cv2.imread('../tmp/{0}_asdf.png'.format(i))
#     img = cv2.rectangle(img, (0, 0), (250, img.shape[0]), (255, 255, 255), -1)
#
#     grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     for i in range(3):
#         print(i)
#         _, mask = cv2.threshold(grey, 105 + i, 255, cv2.THRESH_BINARY)
#         cv2.imshow('asdf', mask)
#
#         cv2.waitKey(300)
# quit()

def build_hsv_plots():
    background_mask = cv2.imread('../ball_training_data/background_mask.png')
    ball_mask = cv2.imread('../ball_training_data/ball_mask.png')
    for color in ['Purple', 'Blue', 'Green', 'Orange', 'Yellow']:
        fig, axes = plt.subplots(3, 2)
        (ax00, ax01, ax10, ax11, ax20, ax21) = axes.flat
        x_180 = np.asarray(range(180))
        x_256 = np.asarray(range(256))
        y_ck_h = np.zeros(180, np.float32)
        y_ck_s = np.zeros(256, np.float32)
        y_ck_v = np.zeros(256, np.float32)
        y_ll_h = np.zeros(180, np.float32)
        y_ll_s = np.zeros(256, np.float32)
        y_ll_v = np.zeros(256, np.float32)

        line_ck_h, = ax00.plot(x_180, y_ck_h, 'k+')
        line_ck_s, = ax10.plot(x_256, y_ck_s, 'r+')
        line_ck_v, = ax20.plot(x_256, y_ck_v, 'b+')
        line_ll_h, = ax01.plot(x_180, y_ck_h, 'k+')
        line_ll_s, = ax11.plot(x_256, y_ck_s, 'r+')
        line_ll_v, = ax21.plot(x_256, y_ck_v, 'b+')

        ax00.set_title('Background')
        ax01.set_title('{0} Ball'.format(color))

        ax00.set_ylabel('Hue')
        ax10.set_ylabel('Saturation')
        ax20.set_ylabel('Value')

        fig.canvas.draw()
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        background_hues = np.zeros(180, dtype=np.uint32)
        background_saturations = np.zeros(256, dtype=np.uint32)
        background_values = np.zeros(256, dtype=np.uint32)
        ball_hues = np.zeros(180, dtype=np.uint32)
        ball_saturations = np.zeros(256, dtype=np.uint32)
        ball_values = np.zeros(256, dtype=np.uint32)

        blue = cv2.VideoCapture('../ball_training_data/{0}BallRecording.mp4'.format(color))
        ret, frame = blue.read()
        i = 0
        while ret:
            print(i)
            i += 1
            if i < 30:
                ret, frame = blue.read()
                continue
            if i % 100 > 0:
                ret, frame = blue.read()
                continue

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            for x in range(hsv.shape[1]):
                for y in range(hsv.shape[0]):
                    h, s, v = hsv[y, x]

                    # print(h, s, v)

                    if background_mask[y, x, 0]:
                        background_hues[h] += 1
                        background_saturations[s] += 1
                        background_values[v] += 1
                    elif ball_mask[y, x, 0]:
                        ball_hues[h] += 1
                        ball_saturations[s] += 1
                        ball_values[v] += 1

            cv2.imshow('HSV', img)
            cv2.imshow('Frame', frame)

            ret, frame = blue.read()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                # break out of the while loop
                c = False
                break

            if i % 1000 == 0:
                continue

            line_ck_h.set_ydata(background_hues)
            line_ck_s.set_ydata(background_saturations)
            line_ck_v.set_ydata(background_values)
            line_ll_h.set_ydata(ball_hues)
            line_ll_s.set_ydata(ball_saturations)
            line_ll_v.set_ydata(ball_values)

            ax00.set_ylim(0, max(background_hues))
            ax10.set_ylim(0, max(background_saturations))
            ax20.set_ylim(0, max(background_values))
            ax01.set_ylim(0, max(ball_hues))
            ax11.set_ylim(0, max(ball_saturations))
            ax21.set_ylim(0, max(ball_values))

            fig.canvas.draw()
            img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        cv2.imwrite('../ball_training_data/hsv_{0}.png'.format(color), img)


def make_rainbow_hsv(height, width):
    rainbow = np.ndarray((height, width, 3)).astype(np.uint8)

    for x in range(rainbow.shape[1]):
        for y in range(rainbow.shape[0]):
            rainbow[y][x][0] = int((x / rainbow.shape[1]) * 180)  # Hue
            rainbow[y][x][1] = int((1 - (y / rainbow.shape[0])) * 255)  # Saturation
            rainbow[y][x][2] = 255  # Value
    return rainbow


def overlay(background, object, mask, top, left, height, width):
    background_height, background_width, _ = background.shape
    if top + height >= background_height or left + width >= background_width:
        print(top, height, background_height, left, width, background_width)
        raise ValueError('cannot fit object into background there')

    object = cv2.resize(cv2.bitwise_and(object, object, mask=mask), (width, height))
    mask = cv2.resize(mask, (width, height))

    object_layer = cv2.copyMakeBorder(
        object,
        top,
        background_height - top - height,
        left,
        background_width - left - width,
        cv2.BORDER_CONSTANT
    )

    object_mask_layer = cv2.copyMakeBorder(
        mask,
        top,
        background_height - top - height,
        left,
        background_width - left - width,
        cv2.BORDER_CONSTANT
    )

    inverse_object_mask_layer = cv2.bitwise_not(object_mask_layer)
    masked_background = cv2.bitwise_and(background, background, mask=inverse_object_mask_layer)
    full = cv2.addWeighted(masked_background, 1, object_layer, 1, 0)
    return full


def do_thing():
    background = cv2.imread('/Users/tylerhaden/school/AUTO/RoboticAutonomyProject/background.png')
    purple_ball = cv2.imread('/Users/tylerhaden/school/AUTO/RoboticAutonomyProject/purple_ball.jpeg')
    background_height, background_width, _ = background.shape
    tmp = cv2.cvtColor(purple_ball, cv2.COLOR_BGR2GRAY)
    purple_ball_mask = cv2.inRange(tmp, (0), (252))

    purp = cv2.bitwise_and(purple_ball, purple_ball, mask=purple_ball_mask)
    hsv_purp = cv2.cvtColor(purp, cv2.COLOR_BGR2HSV)
    h_purp = hsv_purp[:, :, 0].flatten()
    s_purp = hsv_purp[:, :, 1].flatten()
    v_purp = hsv_purp[:, :, 2].flatten()
    purple_centroid = np.average(h_purp), np.average(s_purp), np.average(v_purp)
    # print(purple_centroid)

    hsv_background = cv2.cvtColor(background, cv2.COLOR_BGR2HSV)

    hsv_purp = hsv_purp.reshape((hsv_purp.shape[0] * hsv_purp.shape[1], hsv_purp.shape[2]))
    hsv_background = hsv_background.reshape(
        (hsv_background.shape[0] * hsv_background.shape[1], hsv_background.shape[2]))
    print(hsv_purp.shape, hsv_background.shape)
    hsv_purp = np.asarray([hsv for hsv in hsv_purp if (255 * 2) + 150 > sum(hsv) > 0])
    hsv_background = np.asarray([hsv for hsv in hsv_background if (255 * 2) + 130 > sum(hsv) > 30])
    print(hsv_purp.shape, hsv_background.shape)
    # quit()
    # y, = np.histogram(hsv_purp, )

    sample = 10

    X_background = np.random.choice(range(len(hsv_background)), sample)
    X_purple = np.random.choice(range(len(hsv_purp)), sample)

    X = np.concatenate((hsv_background[X_background], hsv_purp[X_purple]), axis=0)
    Y = np.concatenate((np.zeros(X_background.shape[0]), np.ones(X_purple.shape[0])), axis=0)

    # print(X, Y)

    # clf = svm.NuSVC(gamma="auto")
    clf = svm.SVC(gamma="auto")
    clf.fit(X, Y)

    rainbow = cv2.cvtColor(make_rainbow_hsv(int(background_height / 5), background_width), cv2.COLOR_HSV2BGR)
    # r = clf.predict(rainbow.reshape((rainbow.shape[0] * rainbow.shape[1], rainbow.shape[2])))
    # print(rainbow.shape, r.shape)
    # cv2.imshow('asdfasdf', r.reshape(rainbow.shape[:2]))

    out_shape = (rainbow.shape[0] + background_height, background_width)
    out = cv2.VideoWriter(
        '/Users/tylerhaden/school/AUTO/RoboticAutonomyProject/nusvc.mp4',
        cv2.VideoWriter_fourcc(*'mp4v'),
        10,
        out_shape
    )
    print(out_shape)

    c = True
    while c:
        c = False
        for i in range(30, 100):
            # hsv_ball = cv2.cvtColor(purple_ball, cv2.COLOR_BGR2HSV)
            # hsv_ball[:, :, 0] = int(150 * (i / 100))
            # color_ball = cv2.cvtColor(hsv_ball, cv2.COLOR_HSV2BGR)

            flying = overlay(
                background,
                purple_ball,
                purple_ball_mask,
                top=int((background_height - 100) * (i / 100)),
                left=int((background_width - 100) * (i / 100)),
                height=i,
                width=i
            )

            combined = np.concatenate((flying, rainbow), axis=0)
            print(combined.shape)

            # dist = np.ndarray(combined.shape).astype(np.float32)
            # max_distance = 0
            # for x in range(dist.shape[1]):
            #     for y in range(dist.shape[0]):
            #         ay, ax, az = combined[y,x,:]
            #         by, bx, bz = purple_centroid
            #         distance = np.sqrt(((by - ay) ** 2) + ((bx - ax) ** 2) + ((bz - az) ** 2))
            #         #print(distance)
            #         max_distance = max(distance, max_distance)
            #         dist[y,x,:] = [distance, distance, distance]
            # #print(dist)
            # dist = 55 + ((dist * 200) / max_distance)
            # dist = dist.astype(np.uint8)

            combined_hsv = cv2.cvtColor(combined, cv2.COLOR_BGR2HSV)

            # comb_copy = combined.copy()
            #
            # print(comb_copy.shape)
            #
            # for x in range(comb_copy.shape[1]):
            #     for y in range(comb_copy.shape[0]):
            #         try:
            #             j = comb_copy[y, x]
            #         except IndexError as e:
            #             print(e)
            #             k = 0
            #         print('j', j)
            #         k = clf.predict([j])[0]
            #         print('k', k)
            #         comb_copy = np.asarray([k * 255, k * 255, k * 255]).astype(np.uint8)

            pre = clf.predict(combined_hsv.reshape((combined.shape[0] * combined.shape[1], combined.shape[2])))
            print('predicted sum', sum(pre))
            pre = pre.reshape(combined.shape[:2])
            sh = (*pre.shape, 1)
            # print(pre.reshape(sh).shape)
            # pre = cv2.cvtColor(pre.reshape(sh, cv2.COLOR_GRAY2BGR))
            pre = pre.reshape(sh)
            # print(pre.shape, combined.shape)

            # print('d', dist)

            # combined0 = np.concatenate((combined, pre.reshape(combined.shape[:2])), axis=0)
            # print(combined)
            # cv2.imshow('asdf', np.where(pre, combined, np.zeros(combined.shape)))
            thing = np.where(pre, combined, np.ones(combined.shape) * 15).astype(np.uint8)
            # print('thing', thing.shape, combined.shape)
            out.write(thing)

            cv2.imshow('color', combined)
            cv2.imshow('color mask', thing)
            cv2.imshow('mask', pre)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                # break out of the while loop
                c = False
                break

    out.release()


if __name__ == '__main__':
    main()
    quit()
