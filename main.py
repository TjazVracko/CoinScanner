import argparse
import glob
import cv2
import copy
from coinsegmentation import get_coin_segments
from util import show_image
from profiler import print_prof_data, clear_prof_data
from featuredetection import FeatureDetector

import random

if __name__ == '__main__':
    # main here
    # parse input
    parser = argparse.ArgumentParser(description='Edge detector')
    parser.add_argument('-i', '--images', required=True, help='Path to images (directory)')

    args = parser.parse_args()

    dirname = args.images

    # loop over all images
    extensions = ("*.png", "*.jpg", "*.jpeg", "*.JPG")
    list_e = []
    for extension in extensions:
        list_e.extend(glob.glob(dirname + "/"+extension))
    # list_e.sort()  # da bo po abecedi
    random.shuffle(list_e)

    # init feature detection
    fd = FeatureDetector()
    fd.learn()

    for filename in list_e:
        # read image
        img = cv2.imread(filename)
        show_image(img, "original")

        # get singular coins (probably coins)
        potential_coins = get_coin_segments(img)

        image_with_circles = copy.copy(img)  # kopija
        for a, x, y, r, pc in potential_coins:
            cv2.circle(image_with_circles, (x, y), r, (0, 0, 255), 8, cv2.LINE_AA)
        show_image(image_with_circles, "najdeni, filtrirani krogi")

        # klasificiramo po barvi
        for a, x, y, r, im in potential_coins:
            coin_type = fd.classify_by_color(im)
            coin_value_list = [a[0] for a in coin_type]
            cv2.putText(image_with_circles, str(coin_value_list), (x - r - 5, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0))
            print("TA KOVANEC JE: \n" + str(coin_type))
            show_image(im, 'trenutni kovanec')

        show_image(image_with_circles, "vrednosti")

    print_prof_data()
