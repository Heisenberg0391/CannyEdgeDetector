from __future__ import division
from gaussian_filter import gaussian
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage

def gradient(img):
    # 左Sobel
    sobelL = np.array(
        [[1, 0, -1],
         [2, 0, -2],
         [1, 0, -1]])

    # 上sobel
    sobelT = np.array(
        [[ 1,  2,  1],
         [ 0,  0,  0],
         [-1, -2, -1]])

    sobelM = np.array(
        [[ 0,  1,  2],
         [-1,  0,  1],
         [-2, -1,  0]], np.int32)

    sobelP = np.array(
        [[0, -1, -2],
         [1,  0, -1],
         [2,  1,  0]], np.int32)

    # 卷积
    # Gx = cv2.filter2D(img, -1, sobelM)
    # Gy = cv2.filter2D(img, -1, sobelP)
    Gx = ndimage.convolve(img.copy(), sobelL)
    Gy = ndimage.convolve(img.copy(), sobelT)

    # 求梯度值，hypot给定两条直角边，返回斜边长
    # 等价于sqrt(x1**2 + x2**2)，以元素为单位运算
    G = np.hypot(Gx, Gy)

    # 梯度方向，arctan2允许分母为0或分子为inf
    # arctan2结果为弧度，转成角度
    theta = np.arctan2(Gy, Gx) * 180 / np.pi

    return G, theta

if __name__ == '__main__':
    # 读取图像
    img_path = './valve.png'
    img = np.array(Image.open(img_path)).astype('int32')
    img = img[:, :, 0]

    # 高斯滤波
    blurred_img = gaussian(img)

    # 计算梯度
    grad_img, grad_phase = gradient(blurred_img)

    plt.gray()

    plt.subplot(2, 2, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title('Original')

    plt.subplot(2, 2, 2)
    plt.imshow(blurred_img)
    plt.axis('off')
    plt.title('Gaussian')

    plt.subplot(2, 2, 3)
    plt.imshow(grad_img)
    plt.axis('off')
    plt.title('Gradient')

    plt.show()
