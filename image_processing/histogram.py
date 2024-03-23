import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import FILE_NAME, bgr_to_yuv

def get(img_gray):
    histogram = np.zeros(256, dtype=np.uint32)
    for intensity in np.unique(img_gray.flatten()):
        histogram[intensity] = np.sum(img_gray == intensity)
    histogram = (histogram - np.min(histogram)) / (np.max(histogram) - np.min(histogram))
    return histogram

def run():
    img = cv2.imread(FILE_NAME, 3)
    img_gray = bgr_to_yuv(img)[:, :, 0]
    histogram = get(img_gray)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    plt.xlim(0, 255)
    plt.ylim(0, np.max(histogram))
    plt.plot(histogram, color='black', linewidth=1.5)
    plt.show()


if __name__ == '__main__':
    run()
