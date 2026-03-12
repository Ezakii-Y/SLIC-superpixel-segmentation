import numpy as np
from src.slic import slic


def test_slic_basic():
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[16:48, 16:48] = [255, 255, 255]

    labels = slic(image, num_segments=16, compactness=20, max_iter=5)
    assert labels.shape == (64, 64)
    assert labels.dtype == np.int32
    assert np.max(labels) >= 0
    assert np.unique(labels).size <= 32
