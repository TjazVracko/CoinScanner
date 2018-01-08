import cv2
import sys
import uuid
from profiler import print_prof_data
import matplotlib.pyplot as plt

COIN_IMG_SIZE = 256


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


UTIL_YES = 0
UTIL_NO = 0


def show_image(img, title=""):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 720, 720)

    cv2.imshow(title, img)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()

    # escape
    if key == 27:
        print_prof_data()
        sys.exit()

    # s
    if key == 115:
        # save
        print("saving")
        znj = uuid.uuid4()
        cv2.imwrite(str(znj) + ".png", img)

    global UTIL_NO
    global UTIL_YES
    # y
    if key == 121:
        print("yes")
        UTIL_YES += 1

    # n
    if key == 110:
        print("no")
        UTIL_NO += 1


def print_yes_no():
    print("CORRECT = ", UTIL_YES, "\nWRONG = ", UTIL_NO)


def reset_yes_no():
    global UTIL_NO
    global UTIL_YES
    UTIL_YES = 0
    UTIL_NO = 0
