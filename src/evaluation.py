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
    # 近似 UE
    h, w = labels.shape
    lab_flat = labels.reshape(-1)
    gt_flat = ground_truth.reshape(-1)

    error = 0.0
    for sp_id in np.unique(lab_flat):
        mask = lab_flat == sp_id
        gt_ids, counts = np.unique(gt_flat[mask], return_counts=True)
        if len(counts) == 0:
            continue
        max_count = counts.max()
        error += mask.sum() - max_count

    return error / (h * w)


def evaluate(labels, ground_truth):
    br = boundary_recall(labels, ground_truth)
    ue = undersegmentation_error(labels, ground_truth)
    asa = 1.0 - ue
    return {
        'boundary_recall': br,
        'undersegmentation_error': ue,
        'achievable_segmentation_accuracy': asa,
    }
