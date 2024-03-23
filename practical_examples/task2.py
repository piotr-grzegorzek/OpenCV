import cv2
from utils import show_img

if __name__ == "__main__":
    img = cv2.imread(
        'task2_img/base_scheme/ride.png')

    show_img(img, "original", False)

    # Colors not needed for shape detection
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur the image to remove noise
    img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)

    # Thresholding to get geometric shapes (binary_inv - black inside threshold, white outside)
    _, thresh = cv2.threshold(img_gray, 240, 255, cv2.THRESH_BINARY_INV)

    # Each geometric shape is a contour
    # RETR_TREE - contour retrieval mode, CHAIN_APPROX_SIMPLE - contour approximation method
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        # Approximate the contour
        # Second argument - epsilon, maximum distance from contour to approximated contour, small value = more precise
        # Third argument - closed contour
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        # Draw the red approx contour size 5 on the original image
        if approx.shape[0] == 10:
            cv2.drawContours(img, [approx], 0, (0, 0, 255), 2)
            # Get the sizes of the contour
            _, y, w, _ = cv2.boundingRect(approx)
            order = "ride" if y / w > 8 else "stop"
            print(order)

    show_img(img, "shape detection")
