import cv2
import numpy as np


def sierpinski_triangle(img, x, y, size, depth):
    if depth == 0:
        return

    cv2.line(img, (x, y+size), (x + size, y+size), (0, 0, 0), 1)
    cv2.line(img, (x + size, y + size), (x+size//2, y), (0, 0, 0), 1)
    cv2.line(img, (x, y+size), (x + size // 2, y), (0, 0, 0), 1)

    sierpinski_triangle(img, x + size // 4, y, size // 2, depth - 1)
    sierpinski_triangle(img, x, y+size // 2, size // 2, depth - 1)
    sierpinski_triangle(img, x+size // 2, y+size // 2, size // 2, depth - 1)


def sierpinski_carpet(img, x, y, size, depth):
    if depth == 0:
        return

    cv2.rectangle(img, (x, y), (x + size, y + size), (0, 0, 0), 1)

    sierpinski_carpet(img, x, y, size // 3, depth - 1)
    sierpinski_carpet(img, x + size // 3, y, size // 3, depth - 1)
    sierpinski_carpet(img, x + 2 * size // 3, y, size // 3, depth - 1)
    sierpinski_carpet(img, x, y + size // 3, size // 3, depth - 1)
    sierpinski_carpet(img, x + 2 * size // 3, y +
                      size // 3, size // 3, depth - 1)
    sierpinski_carpet(img, x, y + 2 * size // 3, size // 3, depth - 1)
    sierpinski_carpet(img, x + size // 3, y + 2 *
                      size // 3, size // 3, depth - 1)
    sierpinski_carpet(img, x + 2 * size // 3, y + 2 *
                      size // 3, size // 3, depth - 1)


if __name__ == "__main__":
    img = (np.zeros((512, 512, 3), np.uint8)+1)*255

    sierpinski_triangle(img, 0, 0, 512, 6)
    cv2.imshow("sierpinski_triangle", img)

    img = (np.zeros((512, 512, 3), np.uint8)+1)*255

    sierpinski_carpet(img, 0, 0, 512, 5)
    cv2.imshow("sierpinski_carpet", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
