import json
import os
import random
import time

import cv2
from matplotlib import colors as plt_colors
import matplotlib.pyplot as plt
import numpy as np

pixel_data_filename = '../ball_training_data/pixel_data.json'

thresh = {
    'Purple': {'lower': (113, 35, 40), 'upper': (145, 160, 220)},
    'Blue': {'lower': (95, 150, 80), 'upper': (100, 255, 250)},
    'Green': {'lower': (43, 60, 40), 'upper': (71, 240, 200)},
    'Yellow': {'lower': (19, 60, 100), 'upper': (23, 255, 255)},
    'Orange': {'lower': (11, 150, 100), 'upper': (16, 255, 250)},
}


def gauss1d(x, m, S):
    K = np.sqrt(2 * np.pi * S)
    M = (-(x - m) ** 2) / (2 * S)
    return np.exp(M) / K


def norm(a):
    return np.asarray(a) / sum(a)


def build_plot_colors():
    h = np.arange(180) / 179
    s = np.ones(180)
    v = np.ones(180)

    return plt_colors.hsv_to_rgb(list(zip(h, s, v)))


def build_hue_pmf_with_subtract_background():
    with open(pixel_data_filename) as pixel_data_json:
        pixel_data = json.load(pixel_data_json)

        colors_and_background = ['Purple', 'Blue', 'Green', 'Yellow', 'Orange', 'Background']
        for color in colors_and_background:
            pixel_data[color]['hues'] = norm(pixel_data[color]['hues'])

        background_inverse = pixel_data[colors_and_background[-1]]['hues']
        background_inverse = np.ones(background_inverse.shape) - background_inverse

        for color in colors_and_background[:-1]:
            pixel_data[color]['hues'] *= background_inverse
            pixel_data[color]['hues'] = norm(pixel_data[color]['hues'])

        x = np.arange(180)
        plot_colors = build_plot_colors()
        fig, axes = plt.subplots(len(colors_and_background), 1)
        for color, ax in zip(colors_and_background, axes):
            ax.bar(x, pixel_data[color]['hues'], color=plot_colors)
            ax.set_ylabel(color)

        axes[0].set_title('Colored Ball - Hue PMF')
        axes[-1].set_xlabel('Hue')

        plt.show()


def build_hue_pmf_with_percentiles(hsv_choice='hues'):
    with open(pixel_data_filename) as pixel_data_json:
        pixel_data = json.load(pixel_data_json)

        colors_and_background = ['Purple', 'Blue', 'Green', 'Yellow', 'Orange', 'Background']
        for color in colors_and_background:
            pixel_data[color][hsv_choice] = norm(pixel_data[color][hsv_choice])

        background_inverse = pixel_data[colors_and_background[-1]][hsv_choice]
        background_inverse = np.ones(background_inverse.shape) - background_inverse

        x = np.arange(len(background_inverse))

        for color in colors_and_background[:-1]:
            pixel_data[color][hsv_choice] *= background_inverse
            pixel_data[color][hsv_choice] = norm(pixel_data[color][hsv_choice])

            # discrete = (pixel_data[color][hsv_choice] * 10000).astype(np.uint32)
            # d_samples = np.repeat(x, discrete)
            #
            # pixel_data[color][hsv_choice + '_p05'] = int(np.percentile(d_samples, 5))
            # pixel_data[color][hsv_choice + '_p25'] = int(np.percentile(d_samples, 25))
            # pixel_data[color][hsv_choice + '_median'] = int(np.percentile(d_samples, 50))
            # pixel_data[color][hsv_choice + '_p75'] = int(np.percentile(d_samples, 75))
            # pixel_data[color][hsv_choice + '_p95'] = int(np.percentile(d_samples, 95))

        plot_colors = build_plot_colors()

        fig, axes = plt.subplots(len(colors_and_background), 1)
        for color, ax in zip(colors_and_background, axes):
            if hsv_choice == 'hues':
                current_plot_colors = plot_colors
            elif color == 'Background':
                current_plot_colors = 'lightgrey'
            else:
                current_plot_colors = color.lower()

            ax.bar(x, pixel_data[color][hsv_choice], color=current_plot_colors)

            # if color != 'Background':
            #     ax.axvline(x=pixel_data[color][hsv_choice + '_p05'], color='grey')
            #     ax.axvline(x=pixel_data[color][hsv_choice + '_p25'], color='grey', ls=':')
            #     ax.axvline(x=pixel_data[color][hsv_choice + '_median'], color='grey', ls='--')
            #     ax.axvline(x=pixel_data[color][hsv_choice + '_p75'], color='grey', ls=':')
            #     ax.axvline(x=pixel_data[color][hsv_choice + '_p95'], color='grey')
            # ax.bar(x, pixel_data[color]['hues'], color=plot_colors)

            if color != 'Background':
                hsv_index = ['hues', 'saturations', 'values'].index(hsv_choice)
                ax.axvline(x=thresh[color]['lower'][hsv_index], color='grey', ls='--')
                ax.axvline(x=thresh[color]['upper'][hsv_index], color='grey', ls='--')


            ax.set_ylabel(color)
            if hsv_choice == 'hues':
                ax.set_xticks(list(range(0, 181, 10)))
            else:
                ax.set_xticks(list(range(0, 255, 20)) + [255])

        label = hsv_choice.upper()[0] + hsv_choice[1:-1]
        axes[-1].set_xlabel(label)
        axes[0].set_title(f'Colored Ball - {label} PMF')

        plt.show()


def compile_and_save_data(samples_n=200):
    start = time.time()

    data = {}

    background_hsv = np.zeros((256, 3), np.uint64)
    background_mask = cv2.imread('../ball_training_data/AllBackgroundMask.png')[:, :, 0].flatten()

    colors = ['Purple', 'Blue', 'Green', 'Orange', 'Yellow']
    for color in colors:
        print(color, time.time() - start)

        video_object = cv2.VideoCapture(f'../ball_training_data/{color}BallRecording.mp4')
        ret, frame = video_object.read()
        frames = []
        while ret:
            frames.append(frame)
            ret, frame = video_object.read()
        video_object.release()

        ball_hsv = np.zeros((256, 3), np.uint64)
        ball_mask = cv2.imread(f'../ball_training_data/{color}BallMask.png')[:, :, 0].flatten()

        samples = random.sample(frames, samples_n)
        for sample in samples:
            # blurring!
            hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)
            hsv = cv2.GaussianBlur(hsv, (9, 9), 0)

            for hsv_i in range(3):
                hue_or_sat_or_val = hsv[:, :, hsv_i].flatten()
                for hb in hue_or_sat_or_val[background_mask > 0]:
                    background_hsv[hb, hsv_i] += 1

                for i in hue_or_sat_or_val[ball_mask > 0]:
                    ball_hsv[i, hsv_i] += 1

        data[color] = {
            'frames': len(frames),
            'samples': samples_n,
            'hues': ball_hsv[:180, 0].tolist(),
            'saturations': ball_hsv[:, 1].tolist(),
            'values': ball_hsv[:, 2].tolist(),
        }
        print(data[color])

    data['Background'] = {
        'samples': samples_n * len(colors),
        'hues': background_hsv[:180, 0].tolist(),
        'saturations': background_hsv[:, 1].tolist(),
        'values': background_hsv[:, 2].tolist(),
    }
    print(data['Background'])

    if os.path.isfile(pixel_data_filename):
        os.rename(
            pixel_data_filename,
            pixel_data_filename.replace('.json', str(random.randint(0, 10000)) + '.json')
        )

    data['meta'] = 'blur after hsv'

    with open(pixel_data_filename, 'w') as f:
        json.dump(data, f, indent=2)


if __name__ == '__main__':
    # compile_and_save_data(samples_n=200)
    # build_hue_pmf_with_subtract_background()
    # build_hue_pmf_with_percentiles('hues')
    # build_hue_pmf_with_percentiles('saturations')
    build_hue_pmf_with_percentiles('values')
