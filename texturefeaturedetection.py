import cv2
import numpy as np
import util


class TextureFeatureDetector:

    @staticmethod
    def get_texture_characteristics_sift(coin_image):
        '''
        Return SIFT  keypoints and descriptors
        '''

        gray = cv2.cvtColor(coin_image, cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d.SIFT_create()

        # kp = sift.detect(gray, None)
        # img = cv2.drawKeypoints(gray, kp, coin_image, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # util.show_image(img)

        kp, des = sift.detectAndCompute(gray, None)
        return kp, des

    @staticmethod
    def get_texture_characteristics_orb(coin_image):
        '''
        Return ORB  keypoints and descriptors
        '''

        orb = cv2.ORB_create()

        # kp = orb.detect(coin_image, None)
        # img = cv2.drawKeypoints(coin_image, kp, coin_image, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # util.show_image(img)

        kp, des = orb.detectAndCompute(coin_image, None)
        return kp, des
