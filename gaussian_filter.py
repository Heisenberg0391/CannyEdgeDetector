from __future__ import division
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage
def gaussian(img):
    # 常用5×5的高斯模板
    kernel = np.array([
        [2, 4,  5,  4,  2],
        [4, 9,  12, 9,  4],
        [5, 12, 15, 12, 5],
        [4, 9,  12, 9,  4],
        [2, 4,  5,  4,  2]]) / 159

    fil_img = ndimage.convolve(img.copy(), kernel)

    return abs(fil_img)

if __name__ == "__main__":
    img_path = './valve.png'
    img = np.array(Image.open(img_path))

    # from sys import argv
    # if len(argv) < 2:
    #     print("Usage: python %s <image>" % argv[0])
    #     exit()
    #
    # im = np.array(Image.open(argv[1]))
    img = img[:, :, 0]
    plt.gray()

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title('Original')

    plt.subplot(1, 2, 2)
    plt.imshow(gaussian(img))
    plt.axis('off')
    plt.title('Filtered')

    plt.show()
