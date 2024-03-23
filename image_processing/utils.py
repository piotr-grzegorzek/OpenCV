import cv2
import numpy as np


FILE_NAME = "C:\\Users\\piotr\\go\\src\\github.com\\pgdev01\\PM\OpenCV\\test_photo\\filtr.JPG"


def show_img(img, title="img"):
    """
    Show image in a window.
    """
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def set_channels(img: np.ndarray, rgb: tuple):
    """
    Set channels of image to given values.
    :param img: image
    :param rgb: tuple of 3 values (blue, green, red)
    """
    for i in range(3):
        if rgb[i] is not None:
            img[:, :, i] = rgb[i]


def get_areas(img: np.ndarray, width: int, height: int) -> dict:
    """
    Divide image into 4 areas.
    :return: {
        "NW": north-west: numpy.ndarray,
        "NE": north-east: numpy.ndarray,
        "SW": south-west: numpy.ndarray,
        "SE": south-east: numpy.ndarray}
    """
    areas = {"NW": img[0:height // 2, 0:width // 2],
             "NE": img[0:height // 2, width // 2:],
             "SW": img[height // 2:, 0:width // 2],
             "SE": img[height // 2:, width // 2:]
             }
    return areas


def bgr_to_yuv(img):
    # Change each pixel by the following formula:
    # Y = 0.299 * R + 0.587 * G + 0.114 * B
    # U = 0,493 * (B - Y)
    # V = 0,877 * (R - Y)
    yuv = img.copy()
    yuv[:, :, 0] = 0.299 * img[:, :, 2] + 0.587 * \
        img[:, :, 1] + 0.114 * img[:, :, 0]
    yuv[:, :, 1] = 0.493 * (img[:, :, 0] - yuv[:, :, 0])
    yuv[:, :, 2] = 0.877 * (img[:, :, 2] - yuv[:, :, 0])
    return yuv


def resize(img):
    """
    Resize image to 250x250.
    """
    return cv2.resize(img, (250, 250), interpolation=cv2.INTER_AREA)


def extend_border(img, offset):
    # Create a new image with the same size as the original image, but with an additional border
    img_with_border = np.zeros(
        (img.shape[0] + 2 * offset, img.shape[1] + 2 * offset, img.shape[2]), dtype=np.float32)

    # Copy the original image into the center of the new image
    img_with_border[offset:img_with_border.shape[0] - offset,
                    offset:img_with_border.shape[1] - offset, :] = img

    # Copy the border pixels from the original image to the new image
    img_with_border[0:offset, offset:img_with_border.shape[1] -
                    offset, :] = img[0:offset, :, :]
    img_with_border[img_with_border.shape[0] - offset:img_with_border.shape[0],
                    offset:img_with_border.shape[1] - offset, :] = img[img.shape[0] - offset:img.shape[0], :, :]
    img_with_border[offset:img_with_border.shape[0] -
                    offset, 0:offset, :] = img[:, 0:offset, :]
    img_with_border[offset:img_with_border.shape[0] - offset, img_with_border.shape[1] -
                    offset:img_with_border.shape[1], :] = img[:, img.shape[1] - offset:img.shape[1], :]

    # Copy the corner pixels from the original image to the new image
    img_with_border[0:offset, 0:offset, :] = img[0:offset, 0:offset, :]
    img_with_border[0:offset, img_with_border.shape[1] - offset:img_with_border.shape[1],
                    :] = img[0:offset, img.shape[1] - offset:img.shape[1], :]
    img_with_border[img_with_border.shape[0] - offset:img_with_border.shape[0],
                    0:offset, :] = img[img.shape[0] - offset:img.shape[0], 0:offset, :]
    img_with_border[img_with_border.shape[0] - offset:img_with_border.shape[0], img_with_border.shape[1] - offset:img_with_border.shape[1], :] = img[img.shape[0] -
                                                                                                                                                     offset:img.shape[0], img.shape[1] - offset:img.shape[1], :]
    return img_with_border
