import cv2
from utils import FILE_NAME, show_img, set_channels


def run():
    img = cv2.imread(FILE_NAME, 3)
    show_img(img)
    # Show img in 3 colors
    # Red
    colored_img = img.copy()
    set_channels(colored_img, (0, 0, None))
    show_img(colored_img)
    # Green
    colored_img = img.copy()
    set_channels(colored_img, (0, None, 0))
    show_img(colored_img)
    # Blue
    colored_img = img.copy()
    set_channels(colored_img, (None, 0, 0))
    show_img(colored_img)


if __name__ == "__main__":
    run()
