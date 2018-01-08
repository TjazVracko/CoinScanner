import cv2
import numpy as np
from util import COIN_IMG_SIZE
from colormath import color_objects, color_diff


class ColorFeatureDetector:

    r = COIN_IMG_SIZE / 2  # NEW_SIZE / 2
    ym, xm = np.ogrid[-r:r, -r:r]
    coin_mask_1 = xm**2 + ym**2 > r**2  # ta maska definira krog (oziroma elemente zunaj kroga (manj nek rob) na kvadratu, saj teh ne upoštevamo)
    coin_mask_3 = np.dstack((coin_mask_1, coin_mask_1, coin_mask_1))

    # posebne maske za 1€ in 2€
    edge_width = COIN_IMG_SIZE / 8  # 25 pri NEW_SIZE=200
    coin_edge_mask_1 = (xm**2 + ym**2 > r**2) | (xm**2 + ym**2 < (r - edge_width)**2)
    coin_inside_mask_1 = xm**2 + ym**2 > (r - edge_width)**2
    coin_edge_mask_3 = np.dstack((coin_edge_mask_1, coin_edge_mask_1, coin_edge_mask_1))
    coin_inside_mask_3 = np.dstack((coin_inside_mask_1, coin_inside_mask_1, coin_inside_mask_1))

    @staticmethod
    def get_color_characteristics(coin_image):
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
        masked_lab_coin = np.ma.array(lab_coin, mask=coin_mask_3)
        # avg_lab = masked_lab_coin.mean(axis=(0, 1))
        std_lab = masked_lab_coin.std(axis=(0, 1))

        masked_edge = np.ma.array(lab_coin, mask=coin_edge_mask_3)
        avg_edge = masked_edge.mean(axis=(0, 1))
        # std_edge = masked_edge.std(axis=(0, 1))

        masked_inside = np.ma.array(lab_coin, mask=coin_inside_mask_3)
        avg_inside = masked_inside.mean(axis=(0, 1))
        # std_inside = masked_inside.std(axis=(0, 1))

        return avg_edge.data, avg_inside.data, std_lab.data  # , std_inside.data  # (avg_gray, *avg_hsv, std_gray, *std_hsv)  # * je "Unpack" operator, RGB data se mi zdi neuporabna

    @staticmethod
    def get_color_histograms(coin_image):
        '''
        Returns nomalized histograms for edge and inside the coin
        for a and b channels seperately (4 hists total) of the CIE Lab space

        L←L∗255/100,a←a+128,b←b+128
        '''
        lab_coin = cv2.cvtColor(coin_image, cv2.COLOR_BGR2Lab)
        l, a, b = np.dsplit(lab_coin, 3)
        masked_a_edge = np.ma.array(a, mask=ColorFeatureDetector.coin_edge_mask_1)
        masked_b_edge = np.ma.array(a, mask=ColorFeatureDetector.coin_inside_mask_1)

        masked_a_inside = np.ma.array(b, mask=ColorFeatureDetector.coin_edge_mask_1)
        masked_b_inside = np.ma.array(b, mask=ColorFeatureDetector.coin_inside_mask_1)

        bins = np.arange(257)

        hist_a_edge, bins = np.histogram(masked_a_edge.compressed(), bins=bins, density=True)
        hist_b_edge, bins = np.histogram(masked_b_edge.compressed(), bins=bins, density=True)
        hist_a_inside, bins = np.histogram(masked_a_inside.compressed(), bins=bins, density=True)
        hist_b_inside, bins = np.histogram(masked_b_inside.compressed(), bins=bins, density=True)

        return hist_a_edge, hist_b_edge, hist_a_inside, hist_b_inside

    # https://docs.opencv.org/3.3.1/dd/d0d/tutorial_py_2d_histogram.html
    @staticmethod
    def get_2d_color_histograms_lab(coin_image):
        '''
        Retuns normalized 2d histogram of the a and b channels of L*a*b*
        '''
        lab_coin = cv2.cvtColor(coin_image, cv2.COLOR_BGR2Lab)
        l, a, b = np.dsplit(lab_coin, 3)

        masked_a_edge = np.ma.array(a, mask=ColorFeatureDetector.coin_edge_mask_1)
        masked_b_edge = np.ma.array(b, mask=ColorFeatureDetector.coin_edge_mask_1)

        masked_a_inside = np.ma.array(a, mask=ColorFeatureDetector.coin_inside_mask_1)
        masked_b_inside = np.ma.array(b, mask=ColorFeatureDetector.coin_inside_mask_1)

        bins = np.arange(257)
        # print(masked_a_edge.ravel().compressed().shape, "   ", masked_a_edge.ravel().compressed().shape)

        hist_edge, xbins, ybins = np.histogram2d(masked_a_edge.ravel().compressed(), masked_b_edge.ravel().compressed(), bins=bins, normed=True)
        hist_inside, xbins, ybins = np.histogram2d(masked_a_inside.ravel().compressed(), masked_b_inside.ravel().compressed(), bins=bins, normed=True)

        return hist_edge, hist_inside

    @staticmethod
    def get_2d_color_histograms_hsv(coin_image):
        '''
        Retuns normalized 2d histogram of the h and s channels of HSV
        '''
        hsv_coin = cv2.cvtColor(coin_image, cv2.COLOR_BGR2HSV)
        h, s, v = np.dsplit(hsv_coin, 3)

        masked_h_edge = np.ma.array(h, mask=ColorFeatureDetector.coin_edge_mask_1)
        masked_s_edge = np.ma.array(s, mask=ColorFeatureDetector.coin_edge_mask_1)

        masked_h_inside = np.ma.array(h, mask=ColorFeatureDetector.coin_inside_mask_1)
        masked_s_inside = np.ma.array(s, mask=ColorFeatureDetector.coin_inside_mask_1)

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
