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
    parser.add_argument('-l', '--load', action='store_true', help='If set, trained data will be loaded from file, otherwise train anew')

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
    csf = Classificator()

    if args.load:
        csf.load_color_knowledge_from_file()

        csf.load_hog_svm()

        csf.load_bow_from_file()
        csf.load_sift_svm()
        # csf.init_and_train_SIFT_BOW_SVM()

        csf.load_combo_svm()
        # csf.learn_combo_svm()

        csf.load_sift_ann()
        # csf.init_and_train_SIFT_BOW_ANN()
    else:
        csf.learn_color()
        csf.learn_hog()
        # csf.load_color_knowledge_from_file()
        # csf.load_hog_svm()

        csf.learn_sift_bow()
        # csf.load_bow_from_file()
        # csf.init_and_train_SIFT_BOW_SVM()

        csf.learn_combo_svm()

        csf.init_and_train_SIFT_BOW_ANN()

    print_prof_data()

    for filename in list_e:
        # read image
        img = cv2.imread(filename)
        # show_image(img, "original")

        # get singular coins (probably coins)
        potential_coins = get_coin_segments(img)

        image_with_circles = copy.copy(img)  # kopija
        for a, x, y, r, pc in potential_coins:
            cv2.circle(image_with_circles, (x, y), r, (0, 0, 255), 8, cv2.LINE_AA)
        # show_image(image_with_circles, "najdeni, filtrirani krogi")
        img_out = copy.copy(image_with_circles)

        # za potrebe size classifikacije
        # all_radii = [r for a, x, y, r, pc in potential_coins]
        # print(all_radii)

        coin_outputs = []
        # klasificiramo
        for a, x, y, r, im in potential_coins:
            # print("NEXT COIN:")
            # po barvi
            # show_image(im, 'trenutni kovanec')

            coin_type_color = csf.classify_by_color(im)
            # print("PO BARVI: ", coin_type_color)
            # coin_value_list = [a[0] for a in coin_type_color]
            # cv2.putText(image_with_circles, str(coin_value_list), (x - r - 5, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0))

            # po teksturi
            tex_coin_class_hog = csf.classify_by_texture_hog(im)
            tex_coin_class_sift = csf.classify_by_texture_sift_bow(im)

            # print("PO HOG: ", tex_coin_class_hog)
            # print("PO SIFT: ", tex_coin_class_sift)

            # combo test
            tex_coin_class_combo = csf.classify_by_texture_combo(im)
            # print("PO COMBO: ", tex_coin_class_combo)

            # ann
            tex_coin_class_ann = csf.classify_by_texture_sift_bow_ann(im)
            # print("PO ANN: ", tex_coin_class_ann)

            # združi rezultate v skupni count
            co = [0]*8
            # če je barva prazna pol tak ni kovanec
            if coin_type_color is not None:
                if coin_type_color == "bron":
                    co[0] = co[1] = co[2] = 5
                elif coin_type_color == "zlato":
                    co[3] = co[4] = co[5] = 5
                elif coin_type_color == "1e":
                    co[6] = 5
                elif coin_type_color == "2e":
                    co[7] = 5
            # še ostale
            co[Classificator.coin_value_string_to_int[tex_coin_class_hog]] += 1
            co[Classificator.coin_value_string_to_int[tex_coin_class_sift]] += 1
            co[Classificator.coin_value_string_to_int[tex_coin_class_combo]] += 1
            co[Classificator.coin_value_string_to_int[tex_coin_class_ann]] += 1

            # print("SKUP:\n", co)

            # začasno, največji score je ta pravi, damo gor
            coin_class = "None"
            if coin_type_color is not None:
                ind = np.argmax(co)
                coin_class = Classificator.coin_value_int_to_string[ind]
                if coin_type_color == "1e" and coin_class != "1e":
                    coin_class = "None"

            coin_outputs.append((im, x, y, r, coin_type_color, co, coin_class))

            cv2.putText(image_with_circles, coin_class, (x - r - 5, y), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), thickness=2)

            # print("SKUP:\n", coin_class)
            # show_image(im, 'trenutni kovanec')

        # imamo "odgovore" za vsak kovanec na sliki. Zaj lahko še preverimo s pomočjo velikosti
        # kovance sortiramo po skupnih odgovorih (če jih je več blo za en class je verjetno tisti a ne)
        sorted_coin_outputs = sorted(coin_outputs, key=lambda c: max(c[5]), reverse=True)  # sort po najbolj možni na začetku
        print(len(sorted_coin_outputs))
        # izločimo tiste, ko je barva prazna
        trunc_coin_outputs = [x for x in sorted_coin_outputs if max(x[5]) >= 5]

        results_for_each_coin = []
        for current, co in enumerate(trunc_coin_outputs):
            # tegale izberemo kot "pravega"
            radius_compare = co[3]
            coin_class_compare = np.argmax(co[5])  # kateri kovanec je, z nekim radiusom
            ratios_compare = csf.coin_size_ratios[coin_class_compare]  # ratios tega kovanca (glede na druge)
            results_for_current_coin = []
            sum_for_current_coin = 0
            for i in range(0, len(trunc_coin_outputs)):
                # if i == current:
                #     continue
                other_coin_radius = trunc_coin_outputs[i][3]
                # v coin size ratio tabeli pogledaj v vrstico izbranega kovanca
                # kateri ratio je najbližji temu zdaj zračunanemu. To je pol ta coin.  zapišeš keri coin in razlika ratiotov
                ratio = radius_compare / other_coin_radius
                diff = np.absolute(np.array(ratios_compare) - np.array([ratio]*8))
                # TODO: kaj če bi dovolili primerjati samo znotraj barvnega razreda??
                # razlike zunaj barvnega razreda nas ne zanimajo
                cts = trunc_coin_outputs[i][6]
                to_add = [10]*8
                if cts == "2e":
                    to_add[7] = 0
                elif cts == "1e":
                    to_add[6] = 0
                elif cts == "10c" or cts == "20c" or cts == "50c":
                    to_add[3] = to_add[4] = to_add[5] = 0
                elif cts == "1c" or cts == "2c" or cts == "5c":
                    to_add[0] = to_add[1] = to_add[2] = 0
                diff = diff + np.array(to_add)
                # print("DIFF: ", diff)
                # katera razlika je najmanjša?
                min_index = np.argmin(diff)
                min_diff_coin = Classificator.coin_value_int_to_string[min_index]

                results_for_current_coin.append((min_diff_coin, i, diff[min_index]))
                sum_for_current_coin += diff[min_index]

            # shranimo rezultate za ta coin
            curr_coin_tup = (Classificator.coin_value_int_to_string[coin_class_compare], current, sum_for_current_coin)
            results_for_each_coin.append((curr_coin_tup, results_for_current_coin))

        # mamo vse rezultate, poiščemo najbolši vnos v tabeli vseh rezultatov
        # tistega z najmanjšim skupnim odstopanjem
        # print(results_for_each_coin)

        best_entry = min(results_for_each_coin, key=lambda c: c[0][2])  # suma po razlikah ratiotov, vzameš tistega z najmanjšo sumo
        print("NAJMANJŠA SUMA\n", best_entry)
        for coin_string, ind, diff in best_entry[1]:
            if diff <= 0.06:
                # če je diff dovolj mali spremenimo razred kovanca
                im, x, y, r, coin_type_color, co, cc = sorted_coin_outputs[ind]  # cc se discarda oz zamenja z coin_string, sotalo ostane
                sorted_coin_outputs[ind] = (im, x, y, r, coin_type_color, co, coin_string)

        # print img
        for ind, (im, x, y, r, coin_type_color, co, coin_class) in enumerate(sorted_coin_outputs):
            cv2.putText(img_out, coin_class + "," + str(ind), (x - r - 5, y), cv2.FONT_HERSHEY_SIMPLEX, 4, (128, 255, 0), thickness=2)

        chosen_index = best_entry[0][1]
        chosen = sorted_coin_outputs[chosen_index]
        cv2.putText(img_out, "___", (chosen[1] - chosen[3] - 5, chosen[2]), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), thickness=2)

        # print yes no values
        # print_yes_no()
        # reset_yes_no()

        img_skup = np.hstack((image_with_circles, img_out))

        show_image(img_skup, "levo po klasifikaciji, desno po cekiranju velikosti", size=(2048, 1024))

    print_prof_data()
