from colorfeaturedetection import ColorFeatureDetector
from texturefeaturedetection import TextureFeatureDetector
from profiler import profile
import glob
import cv2
import numpy as np


class Classificator:

    learning_images_base_path = '/home/comemaster/Documents/Projects/Diploma/EdgeDetect/slike/ucenje/'
    learning_images_folder = {'1c': '_1c', '2c': '_2c', '5c': '_5c', '10c': '_10c', '20c': '_20c', '50c': '_50c', '1e': '_1e', '2e': '_2e'}
    coin_values = ('1c', '2c', '5c', '10c', '20c', '50c', '1e', '2e')
    coin_value_string_to_int = {'1c': 0, '2c': 1, '5c': 2, '10c': 3, '20c': 4, '50c': 5, '1e': 6, '2e': 7}
    coin_value_int_to_string = dict((v, k) for k, v in coin_value_string_to_int.items())  # menja key in value od zgoraj
    color_groups = ('bron', 'zlato', '1e', '2e')

    def __init__(self):
        self.color_knowledge = {}
        self.color_group_knowledge = {}
        self.texture_knowledge = {}

    @profile
    def learn(self):
        '''
        Vzameš vsak set kovancev in zračunaš potrebne podatke za vektor
        '''
        all_color_chars = {}
        all_texture_chars = {}
        # čez vse kovance
        for coin_value, folder_name in self.learning_images_folder.items():
            all_color_chars[coin_value] = []
            all_texture_chars[coin_value] = []

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
                # color_chars = ColorFeatureDetector.get_color_characteristics(img)
                # color_chars = ColorFeatureDetector.get_color_histograms(img)
                # color_chars = ColorFeatureDetector.get_2d_color_histograms_lab(img)
                color_chars = ColorFeatureDetector.get_2d_color_histograms_hsv(img)
                all_color_chars[coin_value].append(color_chars)

                # tekstura
                # kp, des = TextureFeatureDetector.get_texture_characteristics_orb(img)
                # # če je manj značilnic od neke vrednosti, zavržemo
                # if hasattr(des, '__len__') and len(des) > 50:
                #     all_texture_chars[coin_value].append((kp, des))
                hog_des = TextureFeatureDetector.get_texture_characteristics_hog(img)
                all_texture_chars[coin_value].append(hog_des)

        # imamo histograme barv za vsak kovanec, zračunamo povprečje teh čez vse kovance
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

        # teksture
        for coin_value, tex_chars in all_texture_chars.items():
            tex_chars = np.array(tex_chars)
            # shranimo
            self.texture_knowledge[coin_value] = tex_chars
            print((tex_chars.shape))

        # init and train svm
        self.init_and_train_SVM()

    def init_and_train_SVM(self):
        # https://stackoverflow.com/questions/37715160/how-do-i-train-an-svm-classifier-using-hog-features-in-opencv-3-0-in-python

        # pripravimo podatke
        samples = []
        labels = []
        for coin_value, tex_chars in self.texture_knowledge.items():
            for tc in tex_chars:
                samples.append(tc)
                labels.append(self.coin_value_string_to_int[coin_value])

        print("INIT SVM")

        # Convert objects to Numpy Objects
        samples = np.array(samples, dtype='float32')
        labels = np.array(labels)

        print(samples.shape)
        print(labels.shape)

        # randomize order
        rand = np.random.RandomState(321)
        shuffle = rand.permutation(len(samples))
        samples = samples[shuffle]
        labels = labels[shuffle]

        # Create SVM
        svm = cv2.ml.SVM_create()
        svm.setType(cv2.ml.SVM_C_SVC)
        svm.setKernel(cv2.ml.SVM_RBF)  # cv2.ml.SVM_LINEAR
        # svm.setDegree(0.0)
        svm.setGamma(5.383)
        # svm.setCoef0(0.0)
        svm.setC(2.67)
        # svm.setNu(0.0)
        # svm.setP(0.0)
        # svm.setClassWeights(None)

        # Train
        svm.train(samples, cv2.ml.ROW_SAMPLE, labels)
        svm.save('svm_data.dat')

        self.tex_svm = svm

    def classify_by_texture(self, coin):
        '''
        Uses SVM to find coin class
        '''

        # get tex features from coin
        tex_des = TextureFeatureDetector.get_texture_characteristics_hog(coin)
        print("SHAPE:", tex_des.shape, " TYPE: ", tex_des.dtype)
        # use SVM
        result = self.tex_svm.predict(tex_des)  # TODO: zakaj to ne dela FFS

        print(result)

        return None

    def classify_by_color(self, coin):
        '''
        gets coin image as input, checks it against the color_knowledge
        and finds the most suitable matches
        returs coin descriptor(s), or empty array if no coins match
        '''
        out_class = []
        # color_char_of_coin = ColorFeatureDetector.get_2d_color_histograms_lab(coin)
        color_char_of_coin = ColorFeatureDetector.get_2d_color_histograms_hsv(coin)
        print("NEW COIN:")

        # print("KNOWLEDGE: \n" + str(self.color_knowledge))

        # for coin_value, coin_knowledge in self.color_knowledge.items():
        for coin_value, coin_knowledge in self.color_group_knowledge.items():
            # diff = abs(color_knowledge - color_char_of_coin)
            # bigger_then_std = diff > std_color_knowledge*1.5

            # get color diference in lab via formulas
            # diff_color = ColorFeatureDetector.color_difference(coin_knowledge[0], color_char_of_coin[0])
            # diff_color_edge = ColorFeatureDetector.color_difference_no_luminance(coin_knowledge[0], color_char_of_coin[0])
            # diff_color_inside = ColorFeatureDetector.color_difference_no_luminance(coin_knowledge[1], color_char_of_coin[1])
            # diff_std_edge = abs(coin_knowledge[2] - color_char_of_coin[2])
            # diff_std_inside = abs(coin_knowledge[3] - color_char_of_coin[3])
            # diff_std = abs(coin_knowledge[2] - color_char_of_coin[2])

            # razdalja histogramov
            # bins = np.arange(257)
            # d_a_edge = ColorFeatureDetector.histogram_intersection(coin_knowledge[0], color_char_of_coin[0], bins)
            # d_b_edge = ColorFeatureDetector.histogram_intersection(coin_knowledge[1], color_char_of_coin[1], bins)
            # d_a_inside = ColorFeatureDetector.histogram_intersection(coin_knowledge[2], color_char_of_coin[2], bins)
            # d_b_inside = ColorFeatureDetector.histogram_intersection(coin_knowledge[3], color_char_of_coin[3], bins)

            # draw_2d_hist(coin_knowledge[0])
            # draw_2d_hist(coin_knowledge[1])
            # draw_2d_hist(color_char_of_coin[0])
            # draw_2d_hist(color_char_of_coin[1])

            # razdalja 2d histogramov
            distance_edge = ColorFeatureDetector.histogram_2d_compare(coin_knowledge[0], color_char_of_coin[0])
            distance_inside = ColorFeatureDetector.histogram_2d_compare(coin_knowledge[1], color_char_of_coin[1])

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
