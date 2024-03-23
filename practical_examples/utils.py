import cv2

def show_img(img, title="img", end=True):
    """
    Show image in a window.
    """
    cv2.imshow(title, img)
    if end:
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def resize(img):
    """
    Resize image to fit in the screen.
    """
    w, h = img.shape[1], img.shape[0]
    ratio = 450 / w
    img = cv2.resize(img, (int(w*ratio), int(h*ratio)))
    return img