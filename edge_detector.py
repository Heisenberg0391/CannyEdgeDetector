from __future__ import division
from gaussian_filter import gaussian
from gradient import gradient
from nonmax_suppression import suppression
from double_thresholding import thresholding
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def tracking(input, weak, strong=255):
    '''
    :param img:
    :param weak:
    :param strong:
    检测弱边缘像素的8邻域内是否有强边缘像素，若有则保留
    '''
    img = input.copy()
    M, N = img.shape

    # 遍历所有像素
    for i in range(M):
        for j in range(N):
            # 当前像素
            pixel = img[i, j]
            if pixel == weak:
                # check if one of the 8 neighbours is strong (=255 by default)
                try:
                    if ((img[i + 1, j] == strong) or (img[i - 1, j] == strong)
                            or (img[i, j + 1] == strong) or (img[i, j - 1] == strong)
                            or (img[i + 1, j + 1] == strong) or (img[i - 1, j - 1] == strong)
                            or (img[i + 1, j - 1] == strong) or (img[i - 1, j + 1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img


if __name__ == '__main__':

    # 读取图像
    img_path = './ba.jpg'
    img = np.array(Image.open(img_path)).astype('int32')
    if len(img.shape) > 2:
        img = img[:, :, 0]

    blurred_img = gaussian(img)
    grad_img, grad_phase = gradient(blurred_img)
    grad_max = suppression(grad_img, grad_phase)
    thres, strongs = thresholding(grad_max)
    edge = tracking(thres, weak=128, strong=255)

    plt.gray()

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title('Original')

    plt.subplot(1, 2, 2)
    plt.imshow(edge)
    plt.axis('off')
    plt.title('Edges')

    plt.show()
