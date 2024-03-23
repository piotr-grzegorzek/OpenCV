import cv2
import numpy as np
from utils import FILE_NAME, show_img


def subtract_images(a, b):
    return np.abs(b - a)


def run():
    img_orig = cv2.imread(FILE_NAME, 0)
    img = img_orig.copy().astype(np.int16)

    shifted_horiz = img.copy()
    shifted_horiz[:, 1:] = img[:, :-1]
    horiz_diff = subtract_images(img, shifted_horiz)

    shifted_vertic = img.copy()
    shifted_vertic[1:, :] = img[:-1, :]
    vertic_diff = subtract_images(img, shifted_vertic)

    shifted_diag = img.copy()
    shifted_diag[1:, 1:] = img[:-1, :-1]
    diag_diff = subtract_images(img, shifted_diag)

    edges = np.minimum(horiz_diff + vertic_diff +
                       diag_diff, 255).astype(np.uint8)
    merged = cv2.hconcat([img_orig, edges])
    show_img(merged, "Edge detection")


if __name__ == '__main__':
    run()
