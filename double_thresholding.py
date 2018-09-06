from __future__ import division
from gaussian_filter import gaussian
from gradient import gradient
from nonmax_suppression import suppression
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

def thresholding(input):
    img = input.copy()
    # 阈值图
    thres  = np.zeros(img.shape)

    # 定义弱边缘为强边缘像素值的一半
    strong = 255
    weak   = 128
    low, high = 40, 120
    strongs = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            grad = img[i][j]
            if grad >= high:
                thres[i][j] = strong
                strongs.append((i, j))
            elif grad >= low and grad < high:
                thres[i][j] = weak

    return thres, strongs

if __name__ == '__main__':
    # 读取图像
    img_path = './valve.png'
    img = np.array(Image.open(img_path)).astype('int32')
    img = img[:, :, 0]

    blurred_img = gaussian(img)
    grad_img, grad_phase = gradient(blurred_img)
    grad_max = suppression(grad_img, grad_phase)
    thres, strongs = thresholding(grad_max)

    plt.gray()

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title('Original')

    plt.subplot(1, 2, 2)
    plt.imshow(thres)
    plt.axis('off')
    plt.title('Double thresholding')

    plt.show()
