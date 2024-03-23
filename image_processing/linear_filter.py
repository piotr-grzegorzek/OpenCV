import cv2
import numpy as np
from utils import FILE_NAME, extend_border, show_img


def filter2D(img, filter):
    offset = filter.shape[0] // 2
    filtered_img = np.zeros(img.shape, dtype=np.float32)

    # Add border pixels using custom function
    img_with_border = extend_border(img, offset)

    # Apply the filter to the entire image, including the border pixels
    for i in range(offset, img_with_border.shape[0] - offset):
        for j in range(offset, img_with_border.shape[1] - offset):
            for c in range(img.shape[2]):
                weightened_area = img_with_border[i-offset:i +
                                                  offset+1, j-offset:j+offset+1, c] * filter
                filter_sum = np.sum(filter)
                if filter_sum == 0:
                    filter_sum = 1
                new_val = np.sum(weightened_area) / filter_sum
                filtered_img[i-offset, j-offset, c] = np.clip(new_val, 0, 255)

    return filtered_img.astype(np.uint8)


def run():
    filters = [
        np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [
                 1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]),
        np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]),
        np.array([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]),
        np.array([[0, -1, 0], [0, 1, 0], [0, 0, 0]]),
        np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 0]]),
        np.array([[0, 0, -1], [0, 1, 0], [0, 0, 0]])
    ]
    origin_img = cv2.imread(FILE_NAME, 3)

    test = False
    id = 1
    if test:
        filtered_img = filter2D(origin_img, filters[id])
        correct_filter = cv2.filter2D(origin_img, -1, filters[id])
        merged = cv2.hconcat([origin_img, filtered_img, correct_filter])
        show_img(merged, "Test linear filter " + str(id))
        return

    show_img(cv2.hconcat([origin_img, filter2D(
        origin_img, filters[0]), filter2D(
        origin_img, filters[1]), filter2D(
        origin_img, filters[2]), filter2D(
        origin_img, filters[3]), filter2D(
        origin_img, filters[4]), filter2D(
        origin_img, filters[5])]), "Linear filter")


if __name__ == "__main__":
    run()
