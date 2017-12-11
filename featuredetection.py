import cv2
import glob
import numpy as np
from profiler import profile

from coinsegmentation import get_coin_segments
from util import show_image
import copy
from colormath import color_objects, color_diff

import random


class FeatureDetector:

    learning_images_base_path = '/home/comemaster/Documents/Projects/Diploma/EdgeDetect/slike/ucenje/'
    learning_images_folder = {'1c': '_1c', '2c': '_2c', '5c': '_5c', '10c': '_10c', '20c': '_20c', '50c': '_50c', '1e': '_1e', '2e': '_2e'}
    coin_values = ('1c', '2c', '5c', '10c', '20c', '50c', '1e', '2e')

    def __init__(self):
        r = 100  # NEW_SIZE / 2
        ym, xm = np.ogrid[-r:r, -r:r]
        coin_mask = xm**2 + ym**2 > r**2  # ta maska definira krog (oziroma elemente zunaj kroga (manj nek rob) na kvadratu, saj teh ne upoštevamo)
        coin_mask_3 = np.dstack((coin_mask, coin_mask, coin_mask))
        self.coin_mask_1 = coin_mask
        self.coin_mask_3 = coin_mask_3

        # posebne maske za 1€ in 2€
        edge_width = 25  # 25 pri NEW_SIZE=200
        self.coin_edge_mask_1 = (xm**2 + ym**2 > r**2) | (xm**2 + ym**2 < (r - edge_width)**2)
        self.coin_inside_mask_1 = xm**2 + ym**2 > (r - edge_width)**2
        self.coin_edge_mask_3 = np.dstack((self.coin_edge_mask_1, self.coin_edge_mask_1, self.coin_edge_mask_1))
        self.coin_inside_mask_3 = np.dstack((self.coin_inside_mask_1, self.coin_inside_mask_1, self.coin_inside_mask_1))

        self.color_knowledge = {}

    def get_color_caracteristics(self, coin_image):
        '''
        Retuns avarage color and standard deviation within coin cirlce
        glede na https://en.wikipedia.org/wiki/Color_difference
        se nam splača samo Lab prostor, in pol uporabimo tiste formule za razdaljo, ki kao predstavlja človeško zaznavo
        paper: http://www2.ece.rochester.edu/~gsharma/ciede2000/
        '''

        # spremenimo v float 0 do 1 image, da nam konverzija pol vrne prave vrednosti
        # fpi = coin_image.astype('float32') / 255
        fpi = np.float32(coin_image)
        fpi = fpi * (1.0/255)

        lab_coin = cv2.cvtColor(fpi, cv2.COLOR_BGR2Lab)
        masked_lab_coin = np.ma.array(lab_coin, mask=self.coin_mask_3)
        # avg_lab = masked_lab_coin.mean(axis=(0, 1))
        std_lab = masked_lab_coin.std(axis=(0, 1))

        masked_edge = np.ma.array(lab_coin, mask=self.coin_edge_mask_3)
        avg_edge = masked_edge.mean(axis=(0, 1))
        # std_edge = masked_edge.std(axis=(0, 1))

        masked_inside = np.ma.array(lab_coin, mask=self.coin_inside_mask_3)
        avg_inside = masked_inside.mean(axis=(0, 1))
        # std_inside = masked_inside.std(axis=(0, 1))

        return avg_edge.data, avg_inside.data, std_lab.data  # , std_inside.data  # (avg_gray, *avg_hsv, std_gray, *std_hsv)  # * je "Unpack" operator, RGB data se mi zdi neuporabna

    # https://python-colormath.readthedocs.io/en/latest/delta_e.html
    @staticmethod
    def color_difference(color1, color2):
        c1 = color_objects.LabColor(*color1)
        c2 = color_objects.LabColor(*color2)
        return color_diff.delta_e_cie2000(c1, c2)  # najnovejša formula, kao d best

    @staticmethod
    def color_difference_no_luminance(color1, color2):
        l = 50  # pač fiksni L
        c1 = color_objects.LabColor(l, color1[1], color1[2])
        c2 = color_objects.LabColor(l, color2[1], color2[2])
        return color_diff.delta_e_cie2000(c1, c2)  # najnovejša formula, kao d best

    @profile
    def learn(self):
        '''
        Vzameš vsak set kovancev in zračunaš potrebne podatke za vektor
        '''
        all_color_chars = {}
        # čez vse kovance
        for coin_value, folder_name in self.learning_images_folder.items():
            all_color_chars[coin_value] = []

            dirname = self.learning_images_base_path + folder_name
            # loop over all images
            extensions = ("*.png", "*.jpg", "*.jpeg", "*.JPG")
            list_e = []
            for extension in extensions:
                list_e.extend(glob.glob(dirname + "/"+extension))
            list_e.sort()

            # vsak kovanec enega tipa
            for filename in list_e:
                img = cv2.imread(filename)

                # barva
                color_chars = self.get_color_caracteristics(img)
                all_color_chars[coin_value].append(color_chars)

                #
                #
                #

        # imamo povprečne barve in deviacijo za vsak kovanec, zračunamo povprečje teh čez vse kovance
        for coin_value, color_chars in all_color_chars.items():
            cc = np.array(color_chars)
            avg_color_of_coins = np.mean(cc, axis=0)

            print("COIN: " + coin_value)
            print("COLOR: " + str(avg_color_of_coins))

            # shranimo
            self.color_knowledge[coin_value] = avg_color_of_coins

        # print(self.color_knowledge)

    def classify_by_color(self, coin):
        '''
        gets coin image as input, checks it against the color_knowledge
        and finds the most suitable matches
        returs coin descriptor(s), or empty array if no coins match
        '''
        out_class = []
        color_char_of_coin = self.get_color_caracteristics(coin)
        print("THIS COIN: \n" + str(color_char_of_coin))

        # print("KNOWLEDGE: \n" + str(self.color_knowledge))

        for coin_value, coin_knowledge in self.color_knowledge.items():
            # diff = abs(color_knowledge - color_char_of_coin)
            # bigger_then_std = diff > std_color_knowledge*1.5

            # get color diference in lab via formulas
            # diff_color = FeatureDetector.color_difference(coin_knowledge[0], color_char_of_coin[0])
            diff_color_edge = FeatureDetector.color_difference_no_luminance(coin_knowledge[0], color_char_of_coin[0])
            diff_color_inside = FeatureDetector.color_difference_no_luminance(coin_knowledge[1], color_char_of_coin[1])
            # diff_std_edge = abs(coin_knowledge[2] - color_char_of_coin[2])
            # diff_std_inside = abs(coin_knowledge[3] - color_char_of_coin[3])
            diff_std = abs(coin_knowledge[2] - color_char_of_coin[2])

            print("COIN: " + coin_value)
            print("DIFF COL: " + str(diff_color_edge) + "\n" + str(diff_color_inside))
            # print("DIFF STD: " + str(diff_std_edge) + "\n" + str(diff_std_inside))
            print("DIFF STD: " + str(diff_std))

            # razred je enak, če je razlika v povprečni barvi dovolj majhna, in če se stadnardna deviacija ne razlikuje preveč
            if diff_color_edge < 11 and diff_color_inside < 11 and diff_std[1] < 4 and diff_std[2] < 4:
                out_class.append((coin_value, diff_color_edge, diff_color_inside))

        # print(str(out_class))
        out_class = sorted(out_class, key=lambda c: c[1] + c[2])
        return out_class

# if __name__ == '__main__':
#     fd = FeatureDetector()
#     fd.learn()

#     # TEST TEST TEST
#     # loop over all images
#     extensions = ("*.png", "*.jpg", "*.jpeg", "*.JPG")
#     list_e = []
#     for extension in extensions:
#         list_e.extend(glob.glob('/home/comemaster/Documents/Projects/Diploma/EdgeDetect/slike/izbrane' + "/"+extension))
#     # list_e.sort()  # da bo po abecedi
#     random.shuffle(list_e)

#     for filename in list_e:
#         # read image
#         img = cv2.imread(filename)
#         show_image(img, "original")

#         # get singular coins (probably coins)
#         potential_coins = get_coin_segments(img)

#         image_with_circles = copy.copy(img)  # kopija
#         for a, x, y, r, pc in potential_coins:
#             cv2.circle(image_with_circles, (x, y), r, (0, 0, 255), 8, cv2.LINE_AA)
#         show_image(image_with_circles, "najdeni, filtrirani krogi")

#         # klasificiramo po barvi
#         for a, x, y, r, im in potential_coins:
#             coin_type = fd.classify_by_color(im)
#             print("TA KOVANEC JE: \n" + str(coin_type))
#             show_image(im, 'trenutni kovanec')
