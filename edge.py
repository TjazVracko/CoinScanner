import cv2
import argparse
import numpy as np
import glob
# from matplotlib import pyplot as plt
from skimage.transform import hough_circle, hough_circle_peaks
import sys


def show_image(img, title=""):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 1280, 720)

    cv2.imshow(title, img)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()

    if key == 27:
        sys.exit()


# https://stackoverflow.com/a/45196250
# https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
def auto_canny_threshold(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # calculate lower and upper bou
    # lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    lower = upper // 2
    print("lower=" + str(lower) + "  upper=" + str(upper))

    return lower, upper


# uporabimo otsu threshold in 0.X*otsu
def auto_canny_threshold_otsu(image):
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    ret, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    print(str(ret))

    return ret*0.45, ret


# https://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight=houghcircles
def getCircles_old(gray_img):
    # predn gremo po kroge, zmanjšamo rezolucijo
    # print("orig size= " + str(gray_img.shape))

    # naj bo pod 1000xNekaj
    faktor = 1
    if max(gray_img.shape) > 1000:
        faktor = 1 / (max(gray_img.shape) // 500)

    # print("skalirni faktor= " + str(faktor))
    small = cv2.resize(gray_img, (0, 0), fx=faktor, fy=faktor)

    # print("new size = " + str(small.shape))

    # houghCircles
    dp = 2  # inverse ratio of the accumulator resolution to the image resolution. For example, if dp=1 , the accumulator has the same resolution as the input image. If dp=2 , the accumulator has half as big width and height.
    minDist = 10  # Minimum distance between the centers of the detected circles. If the parameter is too small, multiple neighbor circles may be falsely detected in addition to a true one. If it is too large, some circles may be missed.
    x, param1 = auto_canny_threshold(small)  # it is the higher threshold of the two passed to the Canny() edge detector (the lower one is twice smaller).

    param2 = 70  # it is the accumulator threshold for the circle centers at the detection stage. The smaller it is, the more false circles may be detected. Circles, corresponding to the larger accumulator values, will be returned first.
    # minRadius = 10
    # maxRadius = np.array(gray_image).shape[0] // 2
    circles = cv2.HoughCircles(small, cv2.HOUGH_GRADIENT, dp, minDist, param1=param1+1, param2=param2, minRadius=0, maxRadius=30)

    if circles is None:
        raise Exception("circles je prazn")

    circles = np.uint16(np.around(circles))  # v uint pretvori basicaly

    # kroge skaliramo nazaj
    f = int(1 / faktor)
    for index, circle in enumerate(circles[0]):
        circles[0][index] *= f

    # print(str(circles))
    return circles


# http://scikit-image.org/docs/stable/api/skimage.transform.html#hough-circle
# http://scikit-image.org/docs/dev/api/skimage.transform.html#hough-circle-peaks
def getCircles(edge_img):
    # predn gremo po kroge, zmanjšamo rezolucijo
    # naj bo pod 1000xNekaj
    faktor = 1
    if max(edge_img.shape) > 1000:
        faktor = 1 / (max(edge_img.shape) // 500)
    small = cv2.resize(edge_img, (0, 0), fx=faktor, fy=faktor)

    # circels
    radii = np.arange(5, 30, 1)
    res = hough_circle(small, radii, normalize=False, full_output=False)
    accums, cx, cy, rad = hough_circle_peaks(res, radii, min_xdistance=30, min_ydistance=30, threshold=20, num_peaks=np.inf, total_num_peaks=np.inf, normalize=False)

    # skaliraj nazaj
    f = int(1 / faktor)
    cx = cx * f
    cy = cy * f
    rad = rad * f
    return accums, cx, cy, rad


if __name__ == '__main__':
    # main here
    # parse input
    parser = argparse.ArgumentParser(description='Edge detector')
    parser.add_argument('-i', '--images', required=True, help='Path to images (directory)')
    parser.add_argument('-t1', help='Threshold 1 for Canny edge', type=int)
    parser.add_argument('-t2', help='Threshold 2 for Canny edge', type=int)

    args = parser.parse_args()

    dirname = args.images
    threshold1 = args.t1
    threshold2 = args.t2
    auto = False

    if threshold1 is None or threshold2 is None:
        auto = True

    # loop over all images
    extensions = ("*.pgn", "*.jpg", "*.jpeg", "*.JPG")
    list = []
    for extension in extensions:
        list.extend(glob.glob(dirname + "/"+extension))

    list.sort()  # da bo po abecedi

    for filename in list:

        # read image
        img = cv2.imread(filename)

        # show image
        show_image(img, 'original')

        # predpriprava
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # show_image(gray_img)

        # poglejmo si thresholding
        # histogram:
        # plt.hist(gray_img.ravel(), 256, [0, 256])
        # plt.show()

        # blur = cv2.GaussianBlur(gray_img, (5, 5), 0)

        # plt.hist(blur.ravel(), 256, [0, 256])
        # plt.show()

        # otsu
        # threshold, binarizirana = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # print("otsu thresh: " + str(threshold))

        # show_image(binarizirana, 'otsu')

        # adaptive
        # adaptive = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 13, 2)
        # show_image(adaptive, 'adaptive gaussian')

        adaptive2 = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 13, 2)
        show_image(adaptive2, 'adaptive normal')

        # odpiranje, da gre šum proč
        kernel = np.ones((5, 5), np.uint8)
        opened = cv2.morphologyEx(adaptive2, cv2.MORPH_OPEN, kernel)

        show_image(opened, 'open od adaptive normal')

        blured = cv2.medianBlur(adaptive2, 9)
        show_image(blured, 'medianBlur 9x9 od adaptive normal')

        # canny edge
        # cv2.Canny(image, threshold1, threshold2[, edges[, apertureSize[, L2gradient]]]) → edges
        if auto:
            threshold1, threshold2 = auto_canny_threshold(gray_img)
            # threshold1, threshold2 = auto_canny_threshold_otsu(gray_img)

        edges = cv2.Canny(gray_img, threshold1, threshold2)
        show_image(edges, 'Canny nad grayscale od originala')

        threshold1, threshold2 = auto_canny_threshold(opened)

        edges_open = cv2.Canny(opened, threshold1, threshold2)
        show_image(edges_open, 'Canny nad opened sliko')

        threshold1, threshold2 = auto_canny_threshold(blured)

        edges_blur = cv2.Canny(blured, threshold1, threshold2)
        show_image(edges_blur, 'Canny nad median blur')

        #
        #
        #
        #
        # probamo še Houghcircles
        accums, cx, cy, radii = getCircles(edges)

        print(str(accums))

        # draw circles
        image_with_circles = img
        for a, x, y, r in zip(accums, cx, cy, radii):
            # print(str(circle))
            cv2.circle(image_with_circles, (x, y), r, (255, 0, 0), 8, cv2.LINE_AA)

        show_image(image_with_circles, "circles nad Canny od gray")

        # 2

        # circles = getCircles(edges_open)

        # # draw circles
        # image_with_circles = img
        # for circle in circles[0]:
        #     # print(str(circle))
        #     cv2.circle(image_with_circles, (circle[0], circle[1]), circle[2], (255, 0, 0), 8, cv2.LINE_AA)

        # show_image(image_with_circles, "circles nad Canny od opened")

        # # 3

        # circles = getCircles(edges_blur)

        # # draw circles
        # image_with_circles = img
        # for circle in circles[0]:
        #     # print(str(circle))
        #     cv2.circle(image_with_circles, (circle[0], circle[1]), circle[2], (255, 0, 0), 8, cv2.LINE_AA)

        # show_image(image_with_circles, "circles nad Canny od median blur")
