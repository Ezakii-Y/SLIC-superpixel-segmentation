import cv2
import numpy as np


def draw_boundaries(image, labels, color=(0, 0, 255), thickness=1):
    h, w = labels.shape
    boundaries = np.zeros((h, w), dtype=np.uint8)

    for y in range(h - 1):
        for x in range(w - 1):
            if labels[y, x] != labels[y, x + 1] or labels[y, x] != labels[y + 1, x]:
                boundaries[y, x] = 255

    overlay = image.copy()
    overlay[boundaries == 255] = color
    out = cv2.addWeighted(image, 0.75, overlay, 0.25, 0)
    return out
