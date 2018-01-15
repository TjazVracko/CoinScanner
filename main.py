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
        show_image(img, "original")

        # get singular coins (probably coins)
        potential_coins = get_coin_segments(img)

        image_with_circles = copy.copy(img)  # kopija
        for a, x, y, r, pc in potential_coins:
            cv2.circle(image_with_circles, (x, y), r, (0, 0, 255), 8, cv2.LINE_AA)
        show_image(image_with_circles, "najdeni, filtrirani krogi")

        # za potrebe size classifikacije
        # all_radii = [r for a, x, y, r, pc in potential_coins]
        # print(all_radii)

        coin_outputs = []
        # klasificiramo
        for a, x, y, r, im in potential_coins:
            print("NEXT COIN:")
            # po barvi
            # show_image(im, 'trenutni kovanec')

            coin_type_color = csf.classify_by_color(im)
            print("PO BARVI: ", coin_type_color)
            # coin_value_list = [a[0] for a in coin_type_color]
            # cv2.putText(image_with_circles, str(coin_value_list), (x - r - 5, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0))

            # po teksturi
            tex_coin_class_hog = csf.classify_by_texture_hog(im)
            tex_coin_class_sift = csf.classify_by_texture_sift_bow(im)

            print("PO HOG: ", tex_coin_class_hog)
            print("PO SIFT: ", tex_coin_class_sift)

            # combo test
            tex_coin_class_combo = csf.classify_by_texture_combo(im)
            print("PO COMBO: ", tex_coin_class_combo)

            # ann
            tex_coin_class_ann = csf.classify_by_texture_sift_bow_ann(im)
            print("PO ANN: ", tex_coin_class_ann)

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
            coin_outputs.append((im, x, y, r, coin_type_color, co))
            
            # začasno, največji score je ta pravi, damo gor
            coin_class = "None"
            if coin_type_color is not None:
                ind = np.argmax(co)
                coin_class = Classificator.coin_value_int_to_string[ind]
                if coin_type_color == "1e" and coin_class != "1e":
                    coin_class = "None"

            cv2.putText(image_with_circles, coin_class, (x - r - 5, y), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), thickness=2)

            print("SKUP:\n", coin_class)
            show_image(im, 'trenutni kovanec')

        # TODO: imamo "odgovore" za vsak kovanec na sliki. Zaj lahko še preverimo s pomočjo velikosti
        # kovance sortiramo po skupnih odgovorih (če jih je več blo za en class je verjetno tisti a ne)

        # print yes no values
        print_yes_no()
        reset_yes_no()

        show_image(image_with_circles, "vrednosti")

    print_prof_data()
