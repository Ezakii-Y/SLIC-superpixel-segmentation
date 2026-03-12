import numpy as np


def boundary_recall(labels, ground_truth, tolerance=2):
    # 计算边界召回率：GT边界点中可匹配上的比例（TP / (TP + FN)）
    def boundary(mask):
        h, w = mask.shape
        out = np.zeros_like(mask)
        for y in range(h - 1):
            for x in range(w - 1):
                if mask[y, x] != mask[y, x + 1] or mask[y, x] != mask[y + 1, x]:
                    out[y, x] = 1
        return out

    b_pred = boundary(labels)
    b_gt = boundary(ground_truth)

    pred_points = np.argwhere(b_pred)
    gt_points = np.argwhere(b_gt)

    if len(gt_points) == 0:
        return 1.0
    if len(pred_points) == 0:
        return 0.0

    hits = 0
    tol2 = tolerance * tolerance
    for y, x in gt_points:
        d_sq = (pred_points[:, 0] - y) ** 2 + (pred_points[:, 1] - x) ** 2
        if np.min(d_sq) <= tol2:
            hits += 1

    return float(hits) / len(gt_points)


def undersegmentation_error(labels, ground_truth):
    # 经典 UE：每个超像素与最大的GT区域匹配后多余像素之和 / 总像素
    # 参考：Achanta et al., 2012
    if labels.shape != ground_truth.shape:
        raise ValueError('labels and ground_truth must have same shape')

    h, w = labels.shape
    N = h * w
    labels_flat = labels.ravel()
    gt_flat = ground_truth.ravel()

    sp_labels, sp_inv = np.unique(labels_flat, return_inverse=True)
    gt_labels, gt_inv = np.unique(gt_flat, return_inverse=True)

    # 计算每个superpixel与GT组合的交叉计数
    M = len(gt_labels)
    pair_index = sp_inv * M + gt_inv
    pairs, counts = np.unique(pair_index, return_counts=True)

    # 将统计还原为 confusion matrix 形式
    sp_idx = pairs // M
    gt_idx = pairs % M

    # 以 (num_superpixels, num_gt) 矩阵形式进行聚合
    conf = np.zeros((len(sp_labels), len(gt_labels)), dtype=np.int32)
    conf[sp_idx, gt_idx] = counts

    # 对每个超像素，取覆盖最大GT的像素数
    max_match = np.max(conf, axis=1)
    sp_sizes = np.sum(conf, axis=1)

    error = np.sum(sp_sizes - max_match)
    return float(error) / float(N)


def achievable_segmentation_accuracy(labels, ground_truth):
    ue = undersegmentation_error(labels, ground_truth)
    return 1.0 - ue


def evaluate(labels, ground_truth):
    br = boundary_recall(labels, ground_truth)
    ue = undersegmentation_error(labels, ground_truth)
    asa = achievable_segmentation_accuracy(labels, ground_truth)
    return {
        'boundary_recall': br,
        'undersegmentation_error': ue,
        'achievable_segmentation_accuracy': asa,
    }
