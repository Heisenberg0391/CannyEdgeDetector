from __future__ import division
from gaussian_filter import gaussian
from gradient import gradient
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def suppression(input, phase):
    img = input.copy()
    gmax = np.zeros(img.copy().shape)

    # 遍历每个像素
    for i in range(gmax.shape[0]):
        for j in range(gmax.shape[1]):
            # 当前像素梯度角
            theta = phase[i][j]
            # 梯度方向归一化到0-360
            if theta < 0:
                theta += 360

            # 限定边界防止超出
            if ((j + 1) < gmax.shape[1]) and ((j - 1) >= 0) \
                    and ((i + 1) < gmax.shape[0]) and ((i - 1) >= 0):

                # 梯度角=0和180°，与左右两个相邻像素比较，决定去留
                if (theta >= 337.5 or theta < 22.5) or (theta >= 157.5 and theta < 202.5):
                    if img[i][j] >= img[i][j + 1] and img[i][j] >= img[i][j - 1]:
                        gmax[i][j] = img[i][j]

                # 梯度角=45和225°
                if (phase[i][j] >= 22.5 and phase[i][j] < 67.5) \
                        or (phase[i][j] >= 202.5 and phase[i][j] < 247.5):
                    if img[i][j] >= img[i - 1][j + 1] and img[i][j] >= img[i + 1][j - 1]:
                        gmax[i][j] = img[i][j]

                # 梯度角=90和270°
                if (phase[i][j] >= 67.5 and phase[i][j] < 112.5) \
                        or (phase[i][j] >= 247.5 and phase[i][j] < 292.5):
                    if img[i][j] >= img[i - 1][j] and img[i][j] >= img[i + 1][j]:
                        gmax[i][j] = img[i][j]

                # 梯度角=135和315°
                if (phase[i][j] >= 112.5 and phase[i][j] < 157.5) \
                        or (phase[i][j] >= 292.5 and phase[i][j] < 337.5):
                    if img[i][j] >= img[i - 1][j - 1] and img[i][j] >= img[i + 1][j + 1]:
                        gmax[i][j] = img[i][j]
    return gmax

if __name__ == '__main__':

    # 读取图像
    img_path = './valve.png'
    img = np.array(Image.open(img_path)).astype('int32')
    img = img[:, :, 0]

    # 高斯滤波
    blurred_img = gaussian(img)

    # 计算梯度
    grad_img, grad_phase = gradient(blurred_img)

    # 非极大值抑制
    grad_max = suppression(grad_img, grad_phase)

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

    plt.subplot(2, 2, 4)
    plt.imshow(grad_max)
    plt.axis('off')
    plt.title('Non-Maximum suppression')

    plt.show()
