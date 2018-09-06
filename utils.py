""" Canny 边缘检测分为以下5步:
    1. 高斯滤波
    2. 计算梯度
    3. 非极大值抑制
    4. 双门限检测
    5. 边缘跟踪
"""

from scipy import misc
import numpy as np

def to_ndarray(input):
    img = misc.imread(input, flatten=True)
    img = img.astype('int32')
    return img


def round_angle(angle):
    """ Input angle must be \in [0,180) """
    angle = np.rad2deg(angle) % 180
    if (0 <= angle < 22.5) or (157.5 <= angle < 180):
        angle = 0
    elif (22.5 <= angle < 67.5):
        angle = 45
    elif (67.5 <= angle < 112.5):
        angle = 90
    elif (112.5 <= angle < 157.5):
        angle = 135
    return angle