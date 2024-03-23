import cv2
import numpy as np
import histogram
from image_maker import FILE_NAME
from utils import bgr_to_yuv, show_img, resize


def binarization_constant(img, THRESHOLD):
    img[img > THRESHOLD] = 255
    img[img <= THRESHOLD] = 0


def binarization_histogram(img):
    _histogram = histogram.get(img)
    THRESHOLD = np.argmax(_histogram)
    binarization_constant(img, THRESHOLD)


def binarization_multiple_thresholds(img, TH1, TH2):
    img[img > TH2] = 0
    img[img < TH1] = 0
    img[(img >= TH1) & (img <= TH2)] = 255


def run():
    img = cv2.imread(FILE_NAME, 3)
    img_gray_constant = bgr_to_yuv(img)[:, :, 0]
    img_gray_histogram = img_gray_constant.copy()
    img_gray_multiple_thresholds = img_gray_constant.copy()

    binarization_constant(img_gray_constant, 127)
    binarization_histogram(img_gray_histogram)
    binarization_multiple_thresholds(img_gray_multiple_thresholds, 100, 150)

    merged = cv2.hconcat(
        [resize(img_gray_constant),
         resize(img_gray_histogram),
         resize(img_gray_multiple_thresholds)])

    show_img(merged, "Binarization")


if __name__ == "__main__":
    run()
