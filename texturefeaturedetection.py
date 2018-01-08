import cv2
import numpy as np
import util

from skimage import feature, exposure


class TextureFeatureDetector:

    sift = cv2.xfeatures2d.SIFT_create(nfeatures=500)
    orb = cv2.ORB_create(nfeatures=500)
    # HOG
    winSize = (64, 64)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 9
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 64
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma, histogramNormType, L2HysThreshold, gammaCorrection, nlevels)

    @staticmethod
    def get_texture_characteristics_sift(coin_image):
        '''
        Return SIFT  keypoints and descriptors
        '''

        gray = cv2.cvtColor(coin_image, cv2.COLOR_BGR2GRAY)

        # kp = TextureFeatureDetector.sift.detect(gray, None)
        # img = cv2.drawKeypoints(gray, kp, coin_image, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # util.show_image(img)

        kp, des = TextureFeatureDetector.sift.detectAndCompute(gray, None)
        return kp, des

    @staticmethod
    def get_texture_characteristics_orb(coin_image):
        '''
        Return ORB  keypoints and descriptors
        '''
        # kp = orb.detect(coin_image, None)
        # img = cv2.drawKeypoints(coin_image, kp, coin_image, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # util.show_image(img)

        kp, des = TextureFeatureDetector.orb.detectAndCompute(coin_image, None)

        return kp, des

    @staticmethod
    def get_texture_characteristics_hog(coin_image):
        '''
        Returns HOG histogram of the coin image
        TODO: hog ima tudi nastavitve, preveri:
        https://stackoverflow.com/questions/28390614/opencv-hogdescripter-python
        '''

        # winStride = (8, 8)
        # padding = (8, 8)
        # locations = ((10, 20),)

        # h = TextureFeatureDetector.hog.compute(coin_image, winStride, padding, locations)
        # h = h.reshape(len(h))

        # probamo z skimage
        # http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.hog
        gray = cv2.cvtColor(coin_image, cv2.COLOR_BGR2GRAY)
        # (hog, hogImage) = feature.hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys', transform_sqrt=True, visualise=True)
        hog = feature.hog(gray, orientations=9, pixels_per_cell=(32, 32), cells_per_block=(2, 2), block_norm='L2-Hys', transform_sqrt=True, visualise=False)
        # hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
        # hogImage = hogImage.astype("uint8")

        # util.show_image(hogImage, "HOG image")
        # print(hog.shape)
        # print(h.shape)

        return hog.astype('float32')
