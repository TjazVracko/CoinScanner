import cv2
import glob
import numpy as np
from profiler import profile

from coinsegmentation import get_coin_segments
from util import show_image
import copy


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
        Retuns tuple of avarage colors within coin cirlce: (gray, blue, green, red, hue, saturation ,value)
        '''
        gray_coin = cv2.cvtColor(coin_image, cv2.COLOR_BGR2GRAY)
        hsv_coin = cv2.cvtColor(coin_image, cv2.COLOR_BGR2HSV)
        masked_coin = np.ma.array(coin_image, mask=self.coin_mask_3)
        masked_gray_coin = np.ma.array(gray_coin, mask=self.coin_mask_1)
        masked_hsv_coin = np.ma.array(hsv_coin, mask=self.coin_mask_3)

        avg_color = masked_coin.mean(axis=(0, 1))
        avg_gray = masked_gray_coin.mean()
        avg_hsv = masked_hsv_coin.mean(axis=(0, 1))
        # print(str(avg_color))
        # print(str(avg_gray))
        return (avg_gray, *avg_color, *avg_hsv)  # * je "Unpack" operator

    @profile
    def learn(self):
        '''
        Vzameš vsak set kovacev in zračunaš potrebne podatke za vektor
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

        # imamo povprečne barve za vsak kovanec, zračunamo std_dev in avg za vse barvne kanale
        for coin_value, color_chars in all_color_chars.items():
            cc = np.array(color_chars)
            avg_color_of_coins = np.mean(cc, axis=0)
            std_dev_of_coins = np.std(cc, axis=0)

            # print("COIN: " + coin_value)
            # print("AVG: " + str(avg_color_of_coins))
            # print("STD: " + str(std_dev_of_coins))

            # shranimo
            self.color_knowledge[coin_value] = (avg_color_of_coins, std_dev_of_coins)

        # print(self.color_knowledge)

    def classify_by_color(self, coin):
        '''
        gets coin image as input, checks it against the color_knowledge
        and finds the most suitable match (avg color is within std_dev of the learned color)
        returs coin descriptor or False
        '''
        out_class = []
        color_char = self.get_color_caracteristics(coin)
        # print("THIS COIN COLOR: \n" + str(color_char))

        # print("KNOWLEDGE: \n" + str(self.color_knowledge))

        for coin_value, cck in self.color_knowledge.items():
            avg_color_knowledge, std_color_knowledge = cck  # odpakiramo da bo bolj razumljivo

            diff = abs(avg_color_knowledge - color_char)
            bigger_then_std = diff > std_color_knowledge+5

            # print("COIN: " + coin_value)
            # print("DIFF: " + str(diff))
            # print("BIG: " + str(bigger_then_std))

            # recimo da do 2 sta lahko true
            s = sum(bigger_then_std.astype('uint8'))  # iz true false na 0 1
            if s <= 2:
                out_class.append(coin_value)

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
            print("TA KOVANEC JE: " + str(coin_type))
            show_image(im, 'trenutni kovanec')
