import cv2
import numpy as np
from utils import resize, show_img


def contour_color(cnt, img):
    # Center of gravity of the contour
    # https://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html
    M = cv2.moments(cnt)
    center_x = int(M["m10"] / M["m00"])
    center_y = int(M["m01"] / M["m00"])

    # Get pixel in center of gravity (color not grayscale, so take img)
    bgr = img[int(center_y), int(center_x)]
    colors = {
        "red": (237, 28, 36),
        "green": (34, 177, 76),
        "blue": (0, 162, 232),
        "dark blue": (63, 72, 204),
        "orange": (255, 127, 39),
        "purple": (163, 73, 164),
        "black": (0, 0, 0)
    }
    cnt_color = "unknown"
    # Find the color of the pixel
    for name, rgb in colors.items():
        if np.array_equal(bgr[::-1], rgb):
            cnt_color = name
            break
    return cnt_color


if __name__ == "__main__":
    img = resize(cv2.imread(
        'C:\\Users\\piotr\\go\\src\\github.com\\pgdev01\\PM\\OpenCV2\\test.png'))

    show_img(img, "original", False)

    # Colors not needed for shape detection
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Thresholding to get geometric shapes (binary_inv - black inside threshold, white outside)
    _, thresh = cv2.threshold(img_gray, 254, 255, cv2.THRESH_BINARY_INV)

    # Each geometric shape is a contour
    # RETR_TREE - contour retrieval mode, CHAIN_APPROX_SIMPLE - contour approximation method
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        color = contour_color(cnt, img)
        # Approximate the contour
        # Second argument - epsilon, maximum distance from contour to approximated contour, small value = more precise
        # Third argument - closed contour
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        # Draw the red approx contour size 5 on the original image
        cv2.drawContours(img, [approx], 0, (0, 0, 255), 5)
        # Get the coordinates of the contour
        x, y, w, _ = cv2.boundingRect(cnt)
        # Put the name of the contour and its color
        if approx.shape[0] == 3:
            shp = "triangle"
        elif approx.shape[0] == 4:
            shp = "rectangle"
        elif 6 < approx.shape[0] < 15:
            shp = "ellipse"
        else:
            shp = "circle"
        cv2.putText(img, color + " " + shp, (x+w//4, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 40, 255), 2)

    show_img(img, "shape detection")
