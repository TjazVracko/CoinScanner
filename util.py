import cv2
import sys
import uuid


def show_image(img, title=""):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 1280, 720)

    cv2.imshow(title, img)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()

    if key == 27:
        sys.exit()

    if key == 115:
        # save
        print("saving")
        znj = uuid.uuid4()
        cv2.imwrite(str(znj) + ".png", img)
