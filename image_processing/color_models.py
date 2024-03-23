import cv2
from utils import FILE_NAME, bgr_to_yuv, show_img


def run():
    img = cv2.imread(FILE_NAME, 3)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    yuv = bgr_to_yuv(img)

    merged = cv2.hconcat([img, hsv, yuv])
    show_img(merged, "Color models")


if __name__ == '__main__':
    run()
