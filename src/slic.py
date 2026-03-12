import numpy as np
import cv2


def _make_grid_centers(h, w, S):
    centers = []
    for y in np.arange(S / 2, h, S):
        for x in np.arange(S / 2, w, S):
            centers.append([int(y), int(x)])
    return centers


def _perturb_center(lab, cy, cx):
    h, w = lab.shape[:2]
    min_grad = float('inf')
    best = (cy, cx)
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            ny = np.clip(cy + dy, 0, h - 2)
            nx = np.clip(cx + dx, 0, w - 2)
            c = lab[ny, nx].astype(np.float32)
            diff_x = lab[ny, nx + 1] - c
            diff_y = lab[ny + 1, nx] - c
            gx = np.dot(diff_x, diff_x)
            gy = np.dot(diff_y, diff_y)
            g = gx + gy
            if g < min_grad:
                min_grad = g
                best = (ny, nx)
    return best


def slic(image, num_segments=100, compactness=10, max_iter=10):
    """SLIC超像素分割算法实现"""
    if image is None:
        raise ValueError('image is None')

    h, w = image.shape[:2]
    if h * w < num_segments:
        raise ValueError('num_segments must be smaller than number of pixels')

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    N = h * w
    S = int(np.sqrt(N / num_segments) + 0.5)
    if S <= 0:
        S = 1

    centers = []  # [y,x,l,a,b]
    for cy, cx in _make_grid_centers(h, w, S):
        py, px = _perturb_center(lab, cy, cx)
        l, a, b = lab[py, px]
        centers.append([py, px, l, a, b])

    centers = np.array(centers, dtype=np.float32)
    labels = -1 * np.ones((h, w), dtype=np.int32)
    distances = np.full((h, w), np.inf, dtype=np.float32)

    invS = compactness / float(S)
    invS2 = invS * invS

    # 预计算网格坐标，避免循环里重复创建
    yy_all, xx_all = np.indices((h, w))
    yy_all = yy_all.reshape(-1)
    xx_all = xx_all.reshape(-1)
    lab_flat = lab.reshape(-1, 3)

    for it in range(max_iter):
        distances.fill(np.inf)

        for idx, center in enumerate(centers):
            cy, cx, cl, ca, cb = center
            y0 = int(max(cy - S, 0))
            y1 = int(min(cy + S, h - 1))
            x0 = int(max(cx - S, 0))
            x1 = int(min(cx + S, w - 1))

            patch = lab[y0:y1 + 1, x0:x1 + 1]
            yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1]

            dc = (patch[..., 0] - cl) ** 2 + (patch[..., 1] - ca) ** 2 + (patch[..., 2] - cb) ** 2
            ds = (yy - cy) ** 2 + (xx - cx) ** 2
            d = dc + invS2 * ds

            mask = d < distances[y0:y1 + 1, x0:x1 + 1]
            distances[y0:y1 + 1, x0:x1 + 1][mask] = d[mask]
            labels[y0:y1 + 1, x0:x1 + 1][mask] = idx

        # 重新计算质心（矢量化）
        labels_flat = labels.ravel()
        valid = labels_flat >= 0
        if not np.any(valid):
            break

        idx_valid = labels_flat[valid].astype(np.int32)
        y_valid = yy_all[valid].astype(np.float32)
        x_valid = xx_all[valid].astype(np.float32)
        lab_valid = lab_flat[valid]

        count = np.bincount(idx_valid, minlength=len(centers)).astype(np.float32)

        # 避免除0错误
        valid_centers = count > 0
        y_sum = np.bincount(idx_valid, weights=y_valid, minlength=len(centers))
        x_sum = np.bincount(idx_valid, weights=x_valid, minlength=len(centers))
        l_sum = np.bincount(idx_valid, weights=lab_valid[:, 0], minlength=len(centers))
        a_sum = np.bincount(idx_valid, weights=lab_valid[:, 1], minlength=len(centers))
        b_sum = np.bincount(idx_valid, weights=lab_valid[:, 2], minlength=len(centers))

        centers[valid_centers, 0] = y_sum[valid_centers] / count[valid_centers]
        centers[valid_centers, 1] = x_sum[valid_centers] / count[valid_centers]
        centers[valid_centers, 2] = l_sum[valid_centers] / count[valid_centers]
        centers[valid_centers, 3] = a_sum[valid_centers] / count[valid_centers]
        centers[valid_centers, 4] = b_sum[valid_centers] / count[valid_centers]

    # 强制连通性（基于 OpenCV connectedComponents，在 C 代码里高效执行）
    new_labels = np.zeros_like(labels, dtype=np.int32)
    next_label = 0

    for old_label in np.unique(labels):
        mask = (labels == old_label).astype(np.uint8)
        if mask.sum() == 0:
            continue

        num_cc, cc_map = cv2.connectedComponents(mask, connectivity=4)
        for comp_id in range(1, num_cc):
            component_mask = (cc_map == comp_id)
            comp_size = int(component_mask.sum())
            if comp_size == 0:
                continue

            if comp_size <= S * S // 4 and next_label > 0:
                # 小组件合并到最近邻(label作为空间邻接关系)
                # 通过查找旁边像素的已有 new_labels
                ys, xs = np.where(component_mask)
                merged_label = -1
                for (yy, xx) in zip(ys, xs):
                    for dy, dx in ((-1,0),(1,0),(0,-1),(0,1)):
                        ny, nx = yy + dy, xx + dx
                        if 0 <= ny < h and 0 <= nx < w and new_labels[ny, nx] >= 0:
                            merged_label = new_labels[ny, nx]
                            break
                    if merged_label >= 0:
                        break

                if merged_label < 0:
                    merged_label = next_label
                    next_label += 1
                new_labels[component_mask] = merged_label
            else:
                new_labels[component_mask] = next_label
                next_label += 1

    return new_labels
