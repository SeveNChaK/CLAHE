# import os

import cv2
import numpy as np
from matplotlib import pyplot as plot
from pathlib import Path

RESOURCES_FOLDER = "resources/"
OUTPUT_FOLDER = "output/"
CURRENT_IMAGE_NAME = "park"
CURRENT_IMAGE_FORMAT = ".jpg"

SEGMENT_SIZE = 8
CLIP_LIMIT = 40

# 8-2, 20, 40
# 6-2, 20, 40
# 32-2, 20, 40


def start():
    image = cv2.imread(
        RESOURCES_FOLDER + CURRENT_IMAGE_NAME + CURRENT_IMAGE_FORMAT,
        cv2.IMREAD_GRAYSCALE
    )
    # draw_image(image)

    # clahe = cv2.createCLAHE(clipLimit=15.0, tileGridSize=(73, 73))
    # cl1 = clahe.apply(image)
    # draw_image(cl1)
    #
    # temp = (os.stat(RESOURCES_FOLDER + CURRENT_IMAGE_NAME + CURRENT_IMAGE_FORMAT).st_size * 8) / (
    #             image.shape[0] * image.shape[1])
    # print(temp)
    # temp1 = np.power(2, temp)
    # print(temp1)
    # hist = [0] * 256
    # for i in range(0, image.shape[0]):
    #     for j in range(0, image.shape[1]):
    #         hist[image[i, j]] = hist[image[i, j]] + 1
    # aaa = 0
    # for i in range(0, len(hist) - 1):
    #     if hist[i] > 0:
    #         aaa += 1
    # auto_segment_size = np.sqrt(image.shape[0] * image.shape[1] * (256 / np.max(hist)))
    # auto_segment_size = int(round(auto_segment_size))
    # print(auto_segment_size)

    segment_size = SEGMENT_SIZE

    image_height, image_width = image.shape
    top_point = 0
    bottom_point = segment_size - 1

    equ_image = np.zeros(image.shape)
    while bottom_point < image_height:
        left_point = 0
        right_point = segment_size - 1
        while right_point < image_width:
            equ_image += convert_image(image, top_point, bottom_point, left_point, right_point)
            left_point = right_point
            if right_point + segment_size < image_width or (right_point == image_width - 1):
                right_point += segment_size
            else:
                right_point = image_width - 1
        top_point = bottom_point
        if bottom_point + segment_size < image_height or (bottom_point == image_height - 1):
            bottom_point += segment_size
        else:
            bottom_point = image_height - 1
    # draw_image(equ_image, need_save=False)
    #
    # inter_image = cv2.resize(equ_image, None, fx=10, fy=10, interpolation=cv2.INTER_CUBIC)
    # draw_image(inter_image, need_save=False)


def convert_image(image, top_point, bottom_point, left_point, right_point):
    result = np.zeros(image.shape)
    cumulative_hist = crete_cumulative_hist(image, top_point, bottom_point, left_point, right_point)
    for i in range(top_point, bottom_point):
        for j in range(left_point, right_point):
            result[i, j] = cumulative_hist[image[i, j]] * 255 \
                if abs((cumulative_hist[image[i, j]] * 255) - image[i, j]) < CLIP_LIMIT \
                else image[i, j] + CLIP_LIMIT \
                if cumulative_hist[image[i, j]] * 255 >= image[i, j] \
                else image[i, j] - CLIP_LIMIT

    return result


def crete_cumulative_hist(image, top_point, bottom_point, left_point, right_point):
    # create hist
    hist = [0] * 256
    for i in range(top_point, bottom_point):
        for j in range(left_point, right_point):
            hist[image[i, j]] = hist[image[i, j]] + 1

    # clip limit hist
    excess_pixels = 0
    for i in range(0, 255):
        if hist[i] > CLIP_LIMIT:
            excess_pixels += hist[i] - CLIP_LIMIT
            hist[i] = CLIP_LIMIT
    i = 0
    already_counter = 0
    while excess_pixels > 0:
        if hist[i] < CLIP_LIMIT:
            hist[i] += 1
            excess_pixels -= 1
            already_counter = 0
        if i < 255:
            i += 1
        else:
            i = 0
        already_counter += 1
        if already_counter > 255:
            break

    # norm hist
    for i in range(0, len(hist)):
        hist[i] = hist[i] / ((bottom_point - top_point + 1) * (right_point - left_point + 1))

    # сumulativе hist
    for i in range(1, len(hist)):
        hist[i] = hist[i - 1] + hist[i]

    draw_graph(hist)
    # result = np.zeros(image.shape)
    # for i in range(top_point, bottom_point):
    #     for j in range(left_point, right_point):
    #         result[i, j] = hist[image[i, j]] * 255 \
    #             if abs((hist[image[i, j]] * 255) - image[i, j]) < CLIP_LIMIT \
    #             else image[i, j] + CLIP_LIMIT \
    #             if hist[image[i, j]] * 255 >= image[i, j] \
    #             else image[i, j] - CLIP_LIMIT

    return hist


def draw_image(image, need_save=False):
    plot.tick_params(labelsize=0, length=0)
    plot.imshow(image, cmap='gray')
    if need_save:
        Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)
        plot.savefig(
            OUTPUT_FOLDER + CURRENT_IMAGE_NAME + "_s_" + str(SEGMENT_SIZE) + "_cl_" + str(CLIP_LIMIT),
            bbox_inches='tight',
            pad_inches=0
        )
    plot.show()


def draw_graph(arr):
    plot.bar(np.arange(len(arr)), arr)
    plot.grid(True)
    plot.xlabel("x")
    plot.ylabel("y")
    # Path(OUT_FOLDER).mkdir(parents=True, exist_ok=True)
    # plt.savefig(OUT_FOLDER + "end", bbox_inches='tight', pad_inches=0)
    plot.show()


if __name__ == '__main__':
    start()
