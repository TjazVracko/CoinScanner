import argparse
import glob
import cv2
import copy
from coinsegmentation import get_coin_segments
from util import show_image, print_yes_no, reset_yes_no
from profiler import print_prof_data, clear_prof_data
from classification import Classificator

import random
import numpy as np

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
    list_e.sort()  # da bo po abecedi
    # random.shuffle(list_e)

    # init feature detection
    csf = Classificator()
    csf.learn()
    csf.learn_bow()
    # csf.set_bow_from_file()

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

        # klasificiramo
        for a, x, y, r, im in potential_coins:
            # po barvi
            # show_image(im, 'trenutni kovanec')

            coin_type = csf.classify_by_color(im)
            coin_value_list = [a[0] for a in coin_type]
            cv2.putText(image_with_circles, str(coin_value_list), (x - r - 5, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0))
            print("TA KOVANEC JE: \n", coin_type)

            # # TEST
            # r = 100
            # ym, xm = np.ogrid[-r:r, -r:r]
            # coin_mask = xm**2 + ym**2 > r**2  # ta maska definira krog (oziroma elemente zunaj kroga (manj nek rob) na kvadratu, saj teh ne upoštevamo)
            # coin_mask = np.dstack((coin_mask, coin_mask, coin_mask))

            # # posebne maske za 1€ in 2€
            # edge_width = 25  # 25 pri NEW_SIZE=200
            # coin_edge_mask = (xm**2 + ym**2 > r**2) | (xm**2 + ym**2 < (r - edge_width)**2)
            # coin_inside_mask = xm**2 + ym**2 > (r - edge_width)**2
            # coin_edge_mask = np.dstack((coin_edge_mask, coin_edge_mask, coin_edge_mask))
            # coin_inside_mask = np.dstack((coin_inside_mask, coin_inside_mask, coin_inside_mask))

            # ime = im.copy()
            # imi = im.copy()
            # ime[coin_edge_mask] = 0
            # imi[coin_inside_mask] = 0
            # show_image(ime, "edge")
            # show_image(imi, "inside")

            # po teksturi
            tex_coin_class_hog = csf.classify_by_texture_hog(im)
            tex_coin_class_sift = csf.classify_by_texture_sift_bow(im)

            print("PO HOG: ", tex_coin_class_hog)

            print("PO SIFT: ", tex_coin_class_sift)

            show_image(im, 'trenutni kovanec')

        # print yes no values
        print_yes_no()
        reset_yes_no()

        show_image(image_with_circles, "vrednosti")

    print_prof_data()
