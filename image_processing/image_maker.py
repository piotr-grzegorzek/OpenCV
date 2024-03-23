import cv2
import numpy as np
from utils import FILE_NAME, get_areas, set_channels, show_img


def run():
    img = cv2.imread(FILE_NAME, 0)
    height, width = img.shape
    new_img = np.full((height, width, 3), 255, np.uint8)

    areas = get_areas(new_img, width, height)
    # Grey
    set_channels(areas["NW"], (128, 128, 128))
    # Pink
    set_channels(areas["NE"], (254, 0, 255))
    # Yellow
    set_channels(areas["SW"], (1, 255, 255))
    # Cyan
    set_channels(areas["SE"], (255, 255, 1))

    show_img(new_img, "Image maker")


if __name__ == "__main__":
    run()
