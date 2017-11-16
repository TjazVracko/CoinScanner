import cv2
import argparse
import numpy as np
import glob
# from matplotlib import pyplot as plt
from skimage.transform import hough_circle, hough_circle_peaks
from skimage import img_as_float
import sys
import copy

np.set_printoptions(threshold=np.nan)


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
    lower = int(max(0, upper // 2))
    # print("madian= " + str(v) + "  lower=" + str(lower) + "  upper=" + str(upper))

    return lower, upper


# uporabimo otsu threshold in 0.X*otsu
def auto_canny_threshold_otsu(image):
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    ret, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    print(str(ret))

    return ret*0.45, ret


# https://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight=houghcircles
def get_circles_old(gray_img):
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
    circles[0] *= f
    # print(str(circles))
    return circles


# http://scikit-image.org/docs/stable/api/skimage.transform.html#hough-circle
# http://scikit-image.org/docs/dev/api/skimage.transform.html#hough-circle-peaks
def get_circles(edge_img):
    # predn gremo po kroge, zmanjšamo rezolucijo
    # naj bo pod 1000xNekaj
    faktor = 1
    if max(edge_img.shape) > 1000:
        faktor = 1 / (max(edge_img.shape) // 500)
    small = cv2.resize(edge_img, (0, 0), fx=faktor, fy=faktor)

    small = img_as_float(small)
    # circels
    radii = np.arange(10, 25, 1)
    res = hough_circle(small, radii, normalize=False, full_output=False)
    accums, cx, cy, rad = hough_circle_peaks(res, radii, min_xdistance=30, min_ydistance=30, threshold=20, num_peaks=np.inf, total_num_peaks=np.inf, normalize=False)

    # '''test'''
    # for a, x, y, r in zip(accums, cx, cy, rad):
    #     # print(str(circle))
    #     cv2.circle(small, (x, y), r, (255, 0, 0), 1, cv2.LINE_AA)
    # show_image(small, 'small test image')
    # '''test'''

    # skaliraj nazaj
    f = int(1 / faktor)
    cx = cx * f
    cy = cy * f
    rad = rad * f

    meja = 70**2
    circles = list(zip(accums, cx, cy, rad))
    proc = []
    for i in range(len(circles)):
        for j in range(i + 1, len(circles)):
            if (circles[i][1] - circles[j][1])**2 + (circles[i][2] - circles[j][2])**2 < meja:  # če sta dovolj blizu
                if circles[i][0] < circles[j][0]:
                    proc.append(i)
                else:
                    proc.append(j)

    proc = np.unique(np.array(proc))
    proc = np.sort(proc)
    proc = proc[::-1]  # reverse list
    # print("PROC: " + str(proc))

    for ix in proc:
        del circles[ix]

    print("circles: " + str(circles))

    return circles


if __name__ == '__main__':
    # main here
    # parse input
    parser = argparse.ArgumentParser(description='Edge detector')
    parser.add_argument('-i', '--images', required=True, help='Path to images (directory)')

    args = parser.parse_args()

    dirname = args.images

    # loop over all images
    extensions = ("*.pgn", "*.jpg", "*.jpeg", "*.JPG")
    list_e = []
    for extension in extensions:
        list_e.extend(glob.glob(dirname + "/"+extension))
    list_e.sort()  # da bo po abecedi

    for filename in list_e:

        # read image
        img = cv2.imread(filename)

        # show image
        show_image(img, 'original')

        # predpriprava
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # show_image(gray_img, 'grayscale')

        # preverimo druge barvne modele
        # rgb_b = img[:, :, 0]
        # rgb_g = img[:, :, 1]
        # rgb_r = img[:, :, 2]
        rgb_b, rgb_g, rgb_r = cv2.split(img)

        # show_image(rgb_r, 'r')
        # show_image(rgb_g, 'g')
        # show_image(rgb_b, 'b')

        lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab_l, lab_a, lab_b = cv2.split(lab_img)

        # show_image(lab_l, 'l')
        # show_image(lab_a, 'a')
        # show_image(lab_b, 'b')

        # cv2.imwrite('lab_a.png', lab_a)
        # cv2.imwrite('lab_b.png', lab_b)

        # eqalizajmo histograme

        # lab_a_eq = cv2.equalizeHist(lab_a)
        # lab_b_eq = cv2.equalizeHist(lab_b)

        # show_image(lab_a_eq, 'a equalized')
        # show_image(lab_b_eq, 'b equalized')

        luv_img = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
        luv_l, luv_u, luv_v = cv2.split(luv_img)

        # show_image(luv_l, 'l')
        # show_image(luv_u, 'u')
        # show_image(luv_v, 'v')

        # canny edge
        # cv2.Canny(image, threshold1, threshold2[, edges[, apertureSize[, L2gradient]]]) → edges

        threshold1, threshold2 = auto_canny_threshold(gray_img)
        # threshold1, threshold2 = auto_canny_threshold_otsu(gray_img)

        gray_edges = cv2.Canny(gray_img, threshold1, threshold2)
        # show_image(gray_edges, 'Canny nad grayscale')

        # luv_u
        threshold1, threshold2 = auto_canny_threshold(luv_u, sigma=-0.75)
        luv_u_edges = cv2.Canny(luv_u, threshold1, threshold2)
        # show_image(luv_u_edges, 'luv u edges')

        # luv_v
        threshold1, threshold2 = auto_canny_threshold(luv_v, sigma=-0.75)
        luv_v_edges = cv2.Canny(luv_u, threshold1, threshold2)
        # show_image(luv_v_edges, 'luv v edges')

        m = cv2.merge((gray_edges, luv_u_edges, luv_v_edges))  # samo za pokazat
        show_image(m, 'blue=gray_edges, green=luv u, red=luv v')

        merged_edges = cv2.add(cv2.add(gray_edges, luv_u_edges), luv_v_edges)  # skrbi za overflow

        # show_image(np.hstack((lab_a_edges, lab_b_edges, ab_edges)))

        # threshold1, threshold2 = auto_canny_threshold(opened)

        # edges_open = cv2.Canny(opened, threshold1, threshold2)
        # show_image(edges_open, 'Canny nad opened sliko')

        # threshold1, threshold2 = auto_canny_threshold(blured)

        # edges_blur = cv2.Canny(blured, threshold1, threshold2)
        # show_image(edges_blur, 'Canny nad median blur')

        #
        #
        #
        #
        # probamo še Houghcircles
        circles = get_circles(merged_edges)  # accumulator_value, x_coord, y_coord, radius

        # print(str(accums))

        # draw circles
        image_with_circles = copy.copy(img)  # kopija
        for a, x, y, r in circles:
            # print(str(circle))
            cv2.circle(image_with_circles, (x, y), r, (255, 0, 0), 8, cv2.LINE_AA)
            # cv2.putText(image_with_circles, str(a), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))

        # print(str(accums))
        show_image(image_with_circles, "circles")

        # # print("PROC: " + str(proc))

        # for a, x, y, r in circles:
        #     cv2.circle(image_with_circles, (x, y), r, (0, 0, 255), 8, cv2.LINE_AA)
        #     cv2.circle(image_with_circles, (x, y), 5, (0, 0, 255), 5, cv2.LINE_4)

        # show_image(image_with_circles, "blue = all, red = selected")

        # # true image
        # image_with_circles_true = copy.copy(img)
        # for a, x, y, r in circles:
        #     cv2.circle(image_with_circles_true, (x, y), r, (0, 0, 255), 8, cv2.LINE_AA)
        #     cv2.circle(image_with_circles_true, (x, y), 5, (0, 0, 255), 5, cv2.LINE_4)

        # show_image(image_with_circles_true, "only red cirles")

        # # stara metoda
        # circles = get_circles_old(cv2.add(merged_edges, gray_img))
        # # draw circles
        # image_with_circles2 = img
        # for circle in circles[0]:
        #     # print(str(circle))
        #     cv2.circle(image_with_circles, (circle[0], circle[1]), circle[2], (255, 0, 0), 8, cv2.LINE_AA)

        # show_image(image_with_circles2, "circles old method")

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
