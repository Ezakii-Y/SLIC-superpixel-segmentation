import numpy as np


def boundary_recall(labels, ground_truth, tolerance=2):
    # 简单边界召回：将标签边界点匹配到gt边界点
    def boundary(mask):
        h, w = mask.shape
        out = np.zeros_like(mask)
        for y in range(h - 1):
            for x in range(w - 1):
                if mask[y, x] != mask[y, x + 1] or mask[y, x] != mask[y + 1, x]:
                    out[y, x] = 1
        return out

    b1 = boundary(labels)
    b2 = boundary(ground_truth)

    gt_points = np.argwhere(b2)
    if len(gt_points) == 0:
        return 1.0

    hit = 0
    for y, x in np.argwhere(b1):
        d = np.sqrt((gt_points[:, 0] - y) ** 2 + (gt_points[:, 1] - x) ** 2)
        if np.min(d) <= tolerance:
            hit += 1

    return float(hit) / max(1, np.sum(b1))


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
