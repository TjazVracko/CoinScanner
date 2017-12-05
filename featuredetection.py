import cv2
import glob
import numpy as np
from profiler import profile

from coinsegmentation import get_coin_segments
from util import show_image
import copy
from colormath import color_objects, color_diff
    

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

        self.color_knowledge = {}

    def get_color_caracteristics(self, coin_image):
        '''
        Retuns avarage color and standard deviation within coin cirlce
        glede na https://en.wikipedia.org/wiki/Color_difference
        se nam splača samo Lab prostor, in pol uporabimo tiste formule za razdaljo, ki kao predstavlja človeško zaznavo
        paper: http://www2.ece.rochester.edu/~gsharma/ciede2000/
        '''
        # gray_coin = cv2.cvtColor(coin_image, cv2.COLOR_BGR2GRAY)
        # hsv_coin = cv2.cvtColor(coin_image, cv2.COLOR_BGR2HSV)
        # masked_coin = np.ma.array(coin_image, mask=self.coin_mask_3)
        # masked_gray_coin = np.ma.array(gray_coin, mask=self.coin_mask_1)
        # masked_hsv_coin = np.ma.array(hsv_coin, mask=self.coin_mask_3)

        # avg_bgr = masked_coin.mean(axis=(0, 1))
        # avg_gray = masked_gray_coin.mean()
        # avg_hsv = masked_hsv_coin.mean(axis=(0, 1))
        # std_bgr = masked_coin.std(axis=(0, 1))
        # std_gray = masked_gray_coin.std()
        # std_hsv = masked_hsv_coin.std(axis=(0, 1))

        lab_coin = cv2.cvtColor(coin_image, cv2.COLOR_BGR2LAB)
        masked_lab_coin = np.ma.array(lab_coin, mask=self.coin_mask_3)
        avg_lab = masked_lab_coin.mean(axis=(0, 1))
        std_lab = masked_lab_coin.std(axis=(0, 1))

        return avg_lab, std_lab  # (avg_gray, *avg_hsv, std_gray, *std_hsv)  # * je "Unpack" operator, RGB data se mi zdi neuporabna

    @staticmethod
    def color_difference(color1, color2):
        c1 = color_objects.LabColor(*color1)
        c2 = color_objects.LabColor(*color2)
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

        for coin_value, color_knowledge in self.color_knowledge.items():
            # diff = abs(color_knowledge - color_char_of_coin)
            # bigger_then_std = diff > std_color_knowledge*1.5

            # get color diference in lab via formulas
            diff = FeatureDetector.color_difference(color_knowledge[0], color_char_of_coin[0])

            print("COIN: " + coin_value)
            print("DIFF: " + str(diff))
            # print("BIG: " + str(bigger_then_std))

            if diff < 15:
                out_class.append((coin_value, diff))

        # print(str(out_class))
        out_class = sorted(out_class, key=lambda c: c[1])
        return out_class

if __name__ == '__main__':
    fd = FeatureDetector()
    fd.learn()

    # TEST TEST TEST
    # loop over all images
    extensions = ("*.png", "*.jpg", "*.jpeg", "*.JPG")
    list_e = []
    for extension in extensions:
        list_e.extend(glob.glob('/home/comemaster/Documents/Projects/Diploma/EdgeDetect/slike/izbrane' + "/"+extension))
    list_e.sort()  # da bo po abecedi

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
            print("TA KOVANEC JE: \n" + str(coin_type))
            show_image(im, 'trenutni kovanec')
