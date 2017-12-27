import cv2
import numpy as np
# from matplotlib import pyplot as plt
from skimage.transform import hough_circle, hough_circle_peaks
from skimage import img_as_float
import copy
from util import show_image
from profiler import profile

np.set_printoptions(threshold=np.nan)


# https://stackoverflow.com/a/45196250
# https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
def auto_canny_threshold(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # calculate lower and upper bound
    # lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    lower = int(max(0, upper // 2))
    # print("madian= " + str(v) + "  lower=" + str(lower) + "  upper=" + str(upper))

    return lower, upper


# http://scikit-image.org/docs/stable/api/skimage.transform.html#hough-circle
# http://scikit-image.org/docs/dev/api/skimage.transform.html#hough-circle-peaks
@profile
def get_circles(edge_img):
    # predn zmanjšao rezolucijo, naredimo malo bolj debele robove
    edge_img = cv2.dilate(edge_img, np.ones((5, 5), np.uint8), iterations=1)
    # predn gremo po kroge, zmanjšamo rezolucijo
    # naj bo pod 1000xNekaj
    faktor = 1
    if max(edge_img.shape) > 1000:
        faktor = 1 / (max(edge_img.shape) // 500)
    small = cv2.resize(edge_img, (0, 0), fx=faktor, fy=faktor)
    # show_image(small, "small")

    small = img_as_float(small)
    # circels
    radii = np.arange(10, 50, 1)  # TODO: max in min radij sta odvisa od tega kak daleč je kamera od kovancev. Bi se dalo to nekak ugotovit??? Da nimamo hardcoded vrednosti
    hspace = hough_circle(small, radii, normalize=False, full_output=False)
    # threshold: Minimum intensity of peaks in each Hough space. Default is (0.5 * np.amax(hspace)).
    accums, cx, cy, rad = hough_circle_peaks(hspace, radii, min_xdistance=30, min_ydistance=30, threshold=10, num_peaks=np.inf, total_num_peaks=np.inf, normalize=False)

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

    # izločimo bližnje kroge
    # meja = 70**2
    circles = list(zip(accums, cx, cy, rad))
    # all_circles = copy.copy(circles)
    to_remove = []
    meja = 50**2
    for i in range(len(circles)):
        # meja = (circles[i][3] - 10)**2  # meja je malo manj kot radius - torej prečekiramo vse bližnje kroge glede na krog
        for j in range(i + 1, len(circles)):
            if j not in to_remove and (circles[i][1] - circles[j][1])**2 + (circles[i][2] - circles[j][2])**2 < meja:  # če sta dovolj blizu
                if circles[i][0] < circles[j][0]:  # izločimo onega z manjšim akumulatorjem
                    to_remove.append(i)
                else:
                    to_remove.append(j)

    to_remove = np.unique(np.array(to_remove))
    to_remove = np.sort(to_remove)
    to_remove = to_remove[::-1]  # reverse list

    for ix in to_remove:
        del circles[ix]
    # print("circles: " + str(circles))

    return circles  # , all_circles


def get_coin_segments(img):
    # show image
    # show_image(img, 'original')

    # predpriprava
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    luv_img = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
    luv_l, luv_u, luv_v = cv2.split(luv_img)

    # canny edge
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

    # samo za pokazat
    # m = cv2.merge((gray_edges, luv_u_edges, luv_v_edges))
    # show_image(m, 'blue=gray_edges, green=luv u, red=luv v')

    merged_edges = cv2.add(cv2.add(gray_edges, luv_u_edges), luv_v_edges)  # skrbi za overflow
    # show_image(merged_edges, "merged edges")

    # probamo še Hough circles
    circles = get_circles(merged_edges)  # ircles, all_circles = get_circles(merged_edges)

    # print(str(accums))

    # # draw circles
    # image_with_circles = copy.copy(img)  # kopija
    # # for a, x, y, r in all_circles:
    # #     cv2.circle(image_with_circles, (x, y), r, (255, 0, 0), 8, cv2.LINE_AA)
    # for a, x, y, r in circles:
    #     cv2.circle(image_with_circles, (x, y), r, (0, 0, 255), 8, cv2.LINE_AA)
    # # print(str(accums))
    # show_image(image_with_circles, "all circles")

    #
    #
    #
    #
    # nekateri krogi so slabi, jih izločimo:
    #
    #
    # dobimo kroge. Na večini so konvanci, na nekaterih je več kovancev, nekateri so le del kovanca, ponavadi znotraj drugega kroga.
    # krog z največjim akumulatorjem je ponavadi kovanec, razn če je ozadje ful neka slika(ampak s tem se ne ubadamo)
    # izločimo vse kroge, ki so za nek faktor večjo od tega "ziher" kovanca

    # izrežemo vsak krog v svojo sliko
    NEW_SIZE = 200
    potential_coins = []
    for a, x, y, r in circles:
        c = img[y - r:y + r, x - r:x + r, :].copy()
        # okoli kovanca naj bo črno
        ym, xm = np.ogrid[-r:r, -r:r]
        mask = xm**2 + ym**2 > r**2
        if c.shape[0] != mask.shape[0] or c.shape[1] != mask.shape[1]:
            continue
        c[mask] = 0
        # resize na 200x200 ??? samo zgubimo relative size s tem
        c = cv2.resize(c, (NEW_SIZE, NEW_SIZE))
        potential_coins.append((a, x, y, r, c))

    # izločimo tiste z več kovanci,
    to_remove = []
    # glede na radius
    SIZE_FACTOR = 1.7
    size_limit = SIZE_FACTOR * potential_coins[0][3]
    STD_DEV_LIMIT = 110
    STD_DEV_EDGE_LIMIT = 90
    # glede na razlike v barvah čez krog - standardna deviacija
    # 1€ in še posebej 2€ imata večjo standardno deviacijo od ostatih kovancev, zato ju sedaj definirana meja včasih izvrže
    # IDEA: zračunamo std_dev za rob in za sredino, tako bi ujeli srebrni rob in zlato sredico posebej
    # 2€ 25pixlov roba, torej 1/8
    # 1€ 28pixlov roba, neki tu
    # torej recimo 25 bo dovolj (kar je 1/8 celega)
    r = NEW_SIZE / 2
    ym, xm = np.ogrid[-r:r, -r:r]
    coin_mask = xm**2 + ym**2 > r**2  # ta maska definira krog (oziroma elemente zunaj kroga (manj nek rob) na kvadratu, saj teh ne upoštevamo)
    coin_mask = np.dstack((coin_mask, coin_mask, coin_mask))

    # posebne maske za 1€ in 2€
    edge_width = NEW_SIZE / 8  # 25 pri NEW_SIZE=200
    coin_edge_mask = (xm**2 + ym**2 > r**2) | (xm**2 + ym**2 < (r - edge_width)**2)
    coin_inside_mask = xm**2 + ym**2 > (r - edge_width)**2
    coin_edge_mask = np.dstack((coin_edge_mask, coin_edge_mask, coin_edge_mask))
    coin_inside_mask = np.dstack((coin_inside_mask, coin_inside_mask, coin_inside_mask))

    for i, pc in enumerate(potential_coins):
        # print("NOV CIRCLE")
        # radius
        if pc[3] >= size_limit:
            to_remove.append(i)
            # print("REMOVED VIA RADIUS: " + str(pc[3]))
            continue  # skip other part

        coin = np.ma.array(pc[4], mask=coin_mask)
        # avg_color = coin.mean(axis=(0, 1))
        std_dev = coin.std(axis=(0, 1))  # vrne deviacijo za r g b posebej : (std_b, std_g, std_r)
        coin_edge = np.ma.array(pc[4], mask=coin_edge_mask)
        coin_inside = np.ma.array(pc[4], mask=coin_inside_mask)
        std_dev_edge = coin_edge.std(axis=(0, 1))
        std_dev_inside = coin_inside.std(axis=(0, 1))

        # print("std_dev: " + str(std_dev))
        # print("std_dev_edge: " + str(std_dev_edge))
        # print("std_dev_inside: " + str(std_dev_inside))

        # testiramo če odklon ustreza
        if sum(std_dev) > STD_DEV_LIMIT:
            # poglejmo še edge in inside

            if sum(std_dev_edge) > STD_DEV_EDGE_LIMIT and sum(std_dev_inside) > STD_DEV_EDGE_LIMIT:
                to_remove.append(i)
                # print("REMOVED VIA STD_DEV SUM: " + str(std_dev) + "\n" + str(std_dev_edge) + "\n" + str(std_dev_inside))
                continue

        # izločimo tiste, ki so pod mejo, pa je odklon na posameznih kanalih dovolj različen
        # a b c so razlike odklnov na dveh kanalih
        a = abs(std_dev[0] - std_dev[1])
        b = abs(std_dev[0] - std_dev[2])
        c = abs(std_dev[1] - std_dev[2])
        # print("a: " + str(a) + " b: " + str(b) + "c: " + str(c))
        # te meje so določene experimentalno (od oke)
        if a > 10 and b > 10 and c > 10:
            to_remove.append(i)
            # print("REMOVED VIA STD DEV POSAMIČNO:" + str(std_dev))
            # show_image(pc[4], "REMOVED VIA STD DEV POSAMIČNO")
            continue

        # print("NOT REMOVED")
        # show_image(pc[4], "save?")

    for ix in to_remove[::-1]:
        del potential_coins[ix]

    # for a, x, a, r, pc in potential_coins:
    #     show_image(pc, "rad: " + str(r) + " acum: " + str(a))
    # draw Circles
    # image_with_circles = copy.copy(img)  # kopija
    # for a, x, y, r, pc in potential_coins:
    #     cv2.circle(image_with_circles, (x, y), r, (0, 0, 255), 8, cv2.LINE_AA)
    # show_image(image_with_circles, "brez velikoh krogov")
    #
    #
    # odstranimo še male kroge znotraj večjih
    # (x - center_x)^2 + (y - center_y)^2 < radius^2
    to_remove = []
    for i in range(len(potential_coins)):
        center_x = potential_coins[i][1]
        center_y = potential_coins[i][2]
        radius = potential_coins[i][3]
        for j in range(i + 1, len(potential_coins)):
            x = potential_coins[j][1]
            y = potential_coins[j][2]
            # če j coin leži v i coinu
            if (x - center_x)**2 + (y - center_y)**2 < radius**2:
                to_remove.append(j)

    # print("pc len: " + str(len(potential_coins)) + " to_remove len: " + str(len(to_remove)))
    to_remove = np.unique(np.array(to_remove))
    for ix in to_remove[::-1]:
        del potential_coins[ix]
    #
    #
    # draw circles
    # print("num of circles: " + str(len(potential_coins)))
    # image_with_circles = copy.copy(img)  # kopija
    # for a, x, y, r, pc in potential_coins:
    #     cv2.circle(image_with_circles, (x, y), r, (0, 0, 255), 8, cv2.LINE_AA)
    # show_image(image_with_circles, "brez malih krogov")

    return potential_coins
