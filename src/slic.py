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
            gx = np.linalg.norm(lab[ny, nx + 1].astype(np.float32) - c)
            gy = np.linalg.norm(lab[ny + 1, nx].astype(np.float32) - c)
            g = gx * gx + gy * gy
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

        # 更新质心
        new_centers = np.zeros_like(centers)
        count = np.zeros((len(centers),), dtype=np.int32)

        for y in range(h):
            for x in range(w):
                l, a, b = lab[y, x]
                idx = labels[y, x]
                if idx >= 0:
                    new_centers[idx, 0] += y
                    new_centers[idx, 1] += x
                    new_centers[idx, 2] += l
                    new_centers[idx, 3] += a
                    new_centers[idx, 4] += b
                    count[idx] += 1

        for i in range(len(centers)):
            if count[i] > 0:
                centers[i] = new_centers[i] / count[i]

    # 强制连通性（简单重标）
    label_id = 0
    new_labels = -1 * np.ones_like(labels)
    dx4 = [-1, 0, 1, 0]
    dy4 = [0, -1, 0, 1]

    for y in range(h):
        for x in range(w):
            if new_labels[y, x] == -1:
                stack = [(y, x)]
                color = labels[y, x]
                component = []
                while stack:
                    cy, cx = stack.pop()
                    if cy < 0 or cy >= h or cx < 0 or cx >= w:
                        continue
                    if new_labels[cy, cx] != -1 or labels[cy, cx] != color:
                        continue
                    new_labels[cy, cx] = label_id
                    component.append((cy, cx))
                    for k in range(4):
                        stack.append((cy + dy4[k], cx + dx4[k]))
                if len(component) <= S * S // 4 and label_id > 0:
                    # 合并小组件
                    for cy, cx in component:
                        new_labels[cy, cx] = new_labels[y, x - 1] if x > 0 else (new_labels[y - 1, x] if y > 0 else 0)
                else:
                    label_id += 1

    return new_labels
