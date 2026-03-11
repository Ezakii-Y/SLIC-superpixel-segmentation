import cv2
import numpy as np


def read_image(path):
    return cv2.imread(path)


def save_image(path, image):
    cv2.imwrite(path, image)
