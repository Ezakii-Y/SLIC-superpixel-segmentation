import numpy as np
from src import evaluation


def test_undersegmentation_error_perfect():
    labels = np.array([
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [2, 2, 3, 3],
        [2, 2, 3, 3],
    ], dtype=np.int32)
    ground_truth = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
    ], dtype=np.int32)

    ue = evaluation.undersegmentation_error(labels, ground_truth)
    assert np.isclose(ue, 0.0)
    asa = evaluation.achievable_segmentation_accuracy(labels, ground_truth)
    assert np.isclose(asa, 1.0)


def test_boundary_recall_identical():
    labels = np.array([
        [0, 0, 1],
        [0, 1, 1],
        [0, 1, 1],
    ], dtype=np.int32)
    ground_truth = labels.copy()

    br = evaluation.boundary_recall(labels, ground_truth, tolerance=1)
    assert np.isclose(br, 1.0)
