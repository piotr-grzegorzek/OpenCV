import cv2
from utils import FILE_NAME, bgr_to_yuv, show_img


def run():
    img = cv2.imread(FILE_NAME, 3)
    # Grayscale using the average method
    gray_avg = (img[:, :, 2] // 3 + img[:, :, 1] // 3 + img[:, :, 0] // 3)

    # Grayscale using the YUV method
    gray_yuv = bgr_to_yuv(img)[:, :, 0]

    show_img(img)
    show_img(gray_avg)
    show_img(gray_yuv)


if __name__ == '__main__':
    run()
