import cv2
import numpy as np
from utils import FILE_NAME, extend_border, show_img


def filterr(img, method, size):
    assert method in ['med', 'min', 'max']
    offset = size // 2
    filtered_img = np.zeros(img.shape, dtype=np.float32)

    # Add border pixels using custom function
    img_with_border = extend_border(img, offset)

    # Apply the filter to the entire image, including the border pixels
    for i in range(offset, img_with_border.shape[0] - offset):
        for j in range(offset, img_with_border.shape[1] - offset):
            for c in range(img.shape[2]):
                area = img_with_border[i-offset:i +
                                       offset+1, j-offset:j+offset+1, c]
                if method == 'med':
                    new_val = np.median(area)
                elif method == 'min':
                    new_val = np.min(area)
                else:
                    new_val = np.max(area)
                filtered_img[i-offset, j-offset, c] = np.clip(new_val, 0, 255)

    return filtered_img.astype(np.uint8)


def run():
    SIZE = 3
    filters = [
        'med',
        'min',
        'max'
    ]
    origin_img = cv2.imread(FILE_NAME, 3)

    test = False
    id = 1
    if test:
        filtered_img = filterr(origin_img, filters[id], size=SIZE)
        if id == 0:
            correct_filter = cv2.medianBlur(origin_img, SIZE)
        elif id == 1:
            correct_filter = cv2.erode(origin_img, np.ones((SIZE, SIZE)))
        else:
            correct_filter = cv2.dilate(origin_img, np.ones((SIZE, SIZE)))
        concatenated_img = cv2.hconcat(
            [origin_img, filtered_img, correct_filter])
        show_img(concatenated_img, "Test")
        return

    for filter in filters:
        filtered_img = filterr(origin_img, filter, size=SIZE)
        if filter == 'med':
            correct_filter = cv2.medianBlur(origin_img, SIZE)
        elif filter == 'min':
            correct_filter = cv2.erode(origin_img, np.ones((SIZE, SIZE)))
        else:
            correct_filter = cv2.dilate(origin_img, np.ones((SIZE, SIZE)))
        concatenated_img = cv2.hconcat(
            [origin_img, filtered_img, correct_filter])
        show_img(concatenated_img, f"Filter: {filter}")


if __name__ == "__main__":
    run()
