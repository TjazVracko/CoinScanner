import cv2
import glob
import numpy as np
from profiler import profile

from coinsegmentation import get_coin_segments
from util import show_image
import copy
from colormath import color_objects, color_diff

import random

import matplotlib.pyplot as plt


def draw_hist(hist, bins):
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center', width=width)
    plt.show()


def draw_4_hist(hists, bins):
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    sp = 221
    for i in range(len(hists)):
        plt.subplot(sp)
        sp += 1
        plt.bar(center, hists[i], align='center', width=width)
    plt.show()


def draw_hist_comparison(hist_array1, hist_array2, bins):
    sp = 221
    for i in range(len(hist_array1)):
        plt.subplot(sp)
        sp += 1
        plt.plot(bins[:-1], hist_array1[i], "b-", bins[:-1], hist_array2[i], "r-")
    plt.show()


def draw_2d_hist(hist2d):
    plt.imshow(hist2d, interpolation='nearest')
    plt.show()


class FeatureDetector:

    learning_images_base_path = '/home/comemaster/Documents/Projects/Diploma/EdgeDetect/slike/ucenje/'
    learning_images_folder = {'1c': '_1c', '2c': '_2c', '5c': '_5c', '10c': '_10c', '20c': '_20c', '50c': '_50c', '1e': '_1e', '2e': '_2e'}
    coin_values = ('1c', '2c', '5c', '10c', '20c', '50c', '1e', '2e')
    color_groups = {'bron', 'zlato', '1e', '2e'}

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
        self.color_group_knowledge = {}

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

    def get_color_histograms(self, coin_image):
        '''
        Returns nomalized histograms for edge and inside the coin
        for a and b channels seperately (4 hists total) of the CIE Lab space

        L←L∗255/100,a←a+128,b←b+128
        '''
        lab_coin = cv2.cvtColor(coin_image, cv2.COLOR_BGR2Lab)
        l, a, b = np.dsplit(lab_coin, 3)
        masked_a_edge = np.ma.array(a, mask=self.coin_edge_mask_1)
        masked_b_edge = np.ma.array(a, mask=self.coin_inside_mask_1)

        masked_a_inside = np.ma.array(b, mask=self.coin_edge_mask_1)
        masked_b_inside = np.ma.array(b, mask=self.coin_inside_mask_1)

        bins = np.arange(257)

        hist_a_edge, bins = np.histogram(masked_a_edge.compressed(), bins=bins, density=True)
        hist_b_edge, bins = np.histogram(masked_b_edge.compressed(), bins=bins, density=True)
        hist_a_inside, bins = np.histogram(masked_a_inside.compressed(), bins=bins, density=True)
        hist_b_inside, bins = np.histogram(masked_b_inside.compressed(), bins=bins, density=True)

        return hist_a_edge, hist_b_edge, hist_a_inside, hist_b_inside

    # https://docs.opencv.org/3.3.1/dd/d0d/tutorial_py_2d_histogram.html
    def get_2d_color_histograms_lab(self, coin_image):
        '''
        Retuns normalized 2d histogram of the a and b channels of L*a*b*
        '''
        lab_coin = cv2.cvtColor(coin_image, cv2.COLOR_BGR2Lab)
        l, a, b = np.dsplit(lab_coin, 3)

        masked_a_edge = np.ma.array(a, mask=self.coin_edge_mask_1)
        masked_b_edge = np.ma.array(b, mask=self.coin_edge_mask_1)

        masked_a_inside = np.ma.array(a, mask=self.coin_inside_mask_1)
        masked_b_inside = np.ma.array(b, mask=self.coin_inside_mask_1)

        bins = np.arange(257)
        # print(masked_a_edge.ravel().compressed().shape, "   ", masked_a_edge.ravel().compressed().shape)

        hist_edge, xbins, ybins = np.histogram2d(masked_a_edge.ravel().compressed(), masked_b_edge.ravel().compressed(), bins=bins, normed=True)
        hist_inside, xbins, ybins = np.histogram2d(masked_a_inside.ravel().compressed(), masked_b_inside.ravel().compressed(), bins=bins, normed=True)

        return hist_edge, hist_inside

    def get_2d_color_histograms_hsv(self, coin_image):
        '''
        Retuns normalized 2d histogram of the h and s channels of HSV
        '''
        hsv_coin = cv2.cvtColor(coin_image, cv2.COLOR_BGR2HSV)
        h, s, v = np.dsplit(hsv_coin, 3)

        masked_h_edge = np.ma.array(h, mask=self.coin_edge_mask_1)
        masked_s_edge = np.ma.array(s, mask=self.coin_edge_mask_1)

        masked_h_inside = np.ma.array(h, mask=self.coin_inside_mask_1)
        masked_s_inside = np.ma.array(s, mask=self.coin_inside_mask_1)

        bins = np.arange(257)
        # print(masked_a_edge.ravel().compressed().shape, "   ", masked_a_edge.ravel().compressed().shape)

        hist_edge, xbins, ybins = np.histogram2d(masked_h_edge.ravel().compressed(), masked_s_edge.ravel().compressed(), bins=bins, normed=True)
        hist_inside, xbins, ybins = np.histogram2d(masked_h_inside.ravel().compressed(), masked_s_inside.ravel().compressed(), bins=bins, normed=True)

        return hist_edge, hist_inside

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

    @staticmethod
    def histogram_intersection(h1, h2, bins):
        bins = np.diff(bins)
        sm = 0
        for i in range(len(bins)):
            sm += min(bins[i]*h1[i], bins[i]*h2[i])
        return sm

    @staticmethod
    def histogram_2d_compare(h1, h2):
        return cv2.compareHist(np.array(h1, dtype='float32'), np.array(h2, dtype='float32'), cv2.HISTCMP_INTERSECT)

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
                # color_chars = self.get_color_caracteristics(img)
                # color_chars = self.get_color_histograms(img)
                # color_chars = self.get_2d_color_histograms_lab(img)
                color_chars = self.get_2d_color_histograms_hsv(img)
                all_color_chars[coin_value].append(color_chars)

                #
                #
                #

        # imamo histograme za vsak kovanec, zračunamo povprečje teh čez vse kovance
        for coin_value, color_chars in all_color_chars.items():
            cc = np.array(color_chars)
            avg_color_of_coins = np.mean(cc, axis=0)

            print("COIN: " + coin_value)
            print("COLOR CHARS SHAPE:" + str(avg_color_of_coins.shape))
            # draw_4_hist(avg_color_of_coins, np.arange(257))
            # draw_2d_hist(avg_color_of_coins[0])
            # draw_2d_hist(avg_color_of_coins[1])

            # shranimo
            self.color_knowledge[coin_value] = avg_color_of_coins

        # združimo barvne skupine
        self.color_group_knowledge['bron'] = (self.color_knowledge['1c'] + self.color_knowledge['2c'] + self.color_knowledge['5c']) / 3
        self.color_group_knowledge['zlato'] = (self.color_knowledge['10c'] + self.color_knowledge['20c'] + self.color_knowledge['50c']) / 3
        self.color_group_knowledge['1e'] = self.color_knowledge['1e']
        self.color_group_knowledge['2e'] = self.color_knowledge['2e']

        # print(self.color_knowledge)

    def classify_by_color(self, coin):
        '''
        gets coin image as input, checks it against the color_knowledge
        and finds the most suitable matches
        returs coin descriptor(s), or empty array if no coins match
        '''
        out_class = []
        # color_char_of_coin = self.get_2d_color_histograms_lab(coin)
        color_char_of_coin = self.get_2d_color_histograms_hsv(coin)
        print("NEW COIN:")

        # print("KNOWLEDGE: \n" + str(self.color_knowledge))

        # for coin_value, coin_knowledge in self.color_knowledge.items():
        for coin_value, coin_knowledge in self.color_group_knowledge.items():
            # diff = abs(color_knowledge - color_char_of_coin)
            # bigger_then_std = diff > std_color_knowledge*1.5

            # get color diference in lab via formulas
            # diff_color = FeatureDetector.color_difference(coin_knowledge[0], color_char_of_coin[0])
            # diff_color_edge = FeatureDetector.color_difference_no_luminance(coin_knowledge[0], color_char_of_coin[0])
            # diff_color_inside = FeatureDetector.color_difference_no_luminance(coin_knowledge[1], color_char_of_coin[1])
            # diff_std_edge = abs(coin_knowledge[2] - color_char_of_coin[2])
            # diff_std_inside = abs(coin_knowledge[3] - color_char_of_coin[3])
            # diff_std = abs(coin_knowledge[2] - color_char_of_coin[2])

            # razdalja histogramov
            # bins = np.arange(257)
            # d_a_edge = FeatureDetector.histogram_intersection(coin_knowledge[0], color_char_of_coin[0], bins)
            # d_b_edge = FeatureDetector.histogram_intersection(coin_knowledge[1], color_char_of_coin[1], bins)
            # d_a_inside = FeatureDetector.histogram_intersection(coin_knowledge[2], color_char_of_coin[2], bins)
            # d_b_inside = FeatureDetector.histogram_intersection(coin_knowledge[3], color_char_of_coin[3], bins)

            # draw_2d_hist(coin_knowledge[0])
            # draw_2d_hist(coin_knowledge[1])
            # draw_2d_hist(color_char_of_coin[0])
            # draw_2d_hist(color_char_of_coin[1])

            # razdalja 2d histogramov
            distance_edge = FeatureDetector.histogram_2d_compare(coin_knowledge[0], color_char_of_coin[0])
            distance_inside = FeatureDetector.histogram_2d_compare(coin_knowledge[1], color_char_of_coin[1])

            print("COIN: " + coin_value)
            # print("DIFF COL: " + str(diff_color_edge) + "\n" + str(diff_color_inside))
            # print("DIFF STD: " + str(diff_std_edge) + "\n" + str(diff_std_inside))
            # print("DIFF STD: " + str(diff_std))
            # print("PODOBNOST HISTOGRAMOV: \n" + str(d_a_edge) + "  " + str(d_b_edge) + " " + str(d_a_inside) + " " + str(d_b_inside))
            print("PODOBNOST HISTOGRAMOV: ", distance_edge, "   ", distance_inside)

            # show graph
            # draw_hist_comparison(coin_knowledge, color_char_of_coin, bins)
            # draw_2d_hist(coin_knowledge[0])
            # draw_2d_hist(color_char_of_coin[0])
            # draw_2d_hist(coin_knowledge[1])
            # draw_2d_hist(color_char_of_coin[1])

            # razred je enak, če je razlika v povprečni barvi dovolj majhna, in če se stadnardna deviacija ne razlikuje preveč
            # if diff_color_edge < 11 and diff_color_inside < 11 and diff_std[1] < 4 and diff_std[2] < 4:
            # out_class.append((coin_value, diff_color_edge, diff_color_inside))
            # if d_a_edge + d_b_edge + d_a_inside + d_b_inside > 1:
            #     out_class.append((coin_value, d_a_edge, d_b_edge, d_a_inside, d_b_inside))

            if distance_edge > 0.07 and distance_inside > 0.07:
                out_class.append((coin_value, distance_edge, distance_inside))

        # print(str(out_class))
        out_class = sorted(out_class, key=lambda c: sum(c[1:]), reverse=True)
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
