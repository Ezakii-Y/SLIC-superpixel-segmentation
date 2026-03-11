import numpy as np


def slic(image, num_segments=100, compactness=10, max_iter=10):
    """占位函数：SLIC超像素分割算法实现"""
    # TODO: 添加完整实现
    h, w = image.shape[:2]
    labels = np.zeros((h, w), dtype=np.int32)
    return labels
