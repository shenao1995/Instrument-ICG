"""ICG correspondence lines from a rendered silhouette (SRT3D / ICG region)."""
from __future__ import annotations

import numpy as np


def _contours(mask, min_pixels=12):
    import cv2
    binary = (np.asarray(mask) > 0.5).astype(np.uint8)
    if binary.sum() < min_pixels:
        return np.zeros((0, 2), np.int32), None, None
    eroded = cv2.erode(binary, np.ones((3, 3), np.uint8), iterations=1)
    contour = binary & (1 - eroded)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    gy, gx = np.gradient(dist.astype(np.float32))
    ys, xs = np.where(contour > 0)
    if not len(xs):
        return np.zeros((0, 2), np.int32), dist, (gx, gy)
    return np.stack((xs, ys), 1).astype(np.int32), dist, (gx, gy)


def sample_correspondences(mask, xyz, part_id, posterior, n_lines=200, min_gap=3,
                           validate_segments=3, scale=2):
    """Sparse contour samples with 2D outward normals and 3D surface points.

    ``mask`` is the predicted region silhouette [H,W]. ``posterior`` is P(fg)
    from color histograms or an observation mask. Correspondence lines that do
    not have a continuous interior / exterior (Mb-ICG validation) are dropped.
    """
    points, dist, grads = _contours(mask)
    if not len(points):
        return []
    gx, gy = grads
    h, w = mask.shape
    if len(points) > n_lines:
        stride = max(1, len(points) // n_lines)
        points = points[::stride][:n_lines]

    lines = []
    min_run = max(2, int(validate_segments) * max(1, int(scale)))
    for x, y in points:
        nx, ny = -float(gx[y, x]), -float(gy[y, x])
        norm = (nx * nx + ny * ny) ** 0.5
        if norm < 1e-6:
            continue
        nx, ny = nx / norm, ny / norm
        if not _valid_line(mask, x, y, nx, ny, min_run):
            continue
        observed, weight = search_contour(posterior, x + 0.5, y + 0.5, nx, ny,
                                          length=max(8, 6 * scale), scale=scale)
        if observed is None:
            continue
        X = xyz[y, x]
        if X[2] <= 1e-4:
            continue
        pid = int(np.rint(part_id[y, x]))
        lines.append({
            "c": np.array([x + 0.5, y + 0.5], np.float64),
            "n": np.array([nx, ny], np.float64),
            "xyz": np.asarray(X, np.float64),
            "part_id": pid,
            "region_id": ("left_" if pid < 4 else "right_") +
                         ("shaft", "wrist", "grippers", "grippers")[pid % 4],
            "observed": np.asarray(observed, np.float64),
            "weight": float(weight),
        })
    if min_gap > 0 and len(lines) > 1:
        lines = _thin(lines, min_gap)
    return lines


def _valid_line(mask, x, y, nx, ny, run):
    h, w = mask.shape
    inside = outside = 0
    for step in range(1, run + 1):
        xi = int(round(x - nx * step))
        yi = int(round(y - ny * step))
        xo = int(round(x + nx * step))
        yo = int(round(y + ny * step))
        if 0 <= xi < w and 0 <= yi < h and mask[yi, xi] > 0.5:
            inside += 1
        if 0 <= xo < w and 0 <= yo < h and mask[yo, xo] <= 0.5:
            outside += 1
    return inside >= run * 0.7 and outside >= run * 0.7


def search_contour(
    posterior,
    cx,
    cy,
    nx,
    ny,
    length=12,
    scale=1,
):
    h, w = posterior.shape

    radius = int(length)
    step = max(1, int(scale))

    rs = np.arange(
        -radius,
        radius + 1,
        step,
        dtype=np.float64,
    )

    values = []
    coords = []
    valid_rs = []

    for r in rs:
        x = cx + nx * r
        y = cy + ny * r

        xi = int(round(x))
        yi = int(round(y))

        if not (0 <= xi < w and 0 <= yi < h):
            continue

        values.append(float(posterior[yi, xi]))
        coords.append(np.array([x, y], np.float64))
        valid_rs.append(float(r))

    if len(values) < 4:
        return None, 0.0

    values = np.asarray(values)
    valid_rs = np.asarray(valid_rs)

    candidates = []

    for i in range(len(values) - 1):
        a = values[i]
        b = values[i + 1]

        # outward normal:
        # correct transition must be foreground -> background
        if a >= 0.5 and b < 0.5:
            if abs(b - a) < 1e-6:
                continue

            t = (0.5 - a) / (b - a)

            crossing = (
                coords[i] * (1.0 - t)
                + coords[i + 1] * t
            )

            r_cross = (
                valid_rs[i] * (1.0 - t)
                + valid_rs[i + 1] * t
            )

            candidates.append(
                (abs(r_cross), crossing, r_cross)
            )

    if not candidates:
        return None, 0.0

    # choose crossing nearest current predicted contour
    candidates.sort(key=lambda x: x[0])

    distance, crossing, _ = candidates[0]

    # reject very remote correspondence
    if distance > 12.0:
        return None, 0.0

    confidence = max(
        0.15,
        1.0 - distance / 12.0,
    )

    return crossing, confidence


def _thin(lines, min_gap):
    kept = []
    used = np.zeros(len(lines), dtype=bool)
    centers = np.stack([row["c"] for row in lines])
    for i, row in enumerate(lines):
        if used[i]:
            continue
        kept.append(row)
        used |= np.linalg.norm(centers - row["c"], axis=1) < min_gap
    return kept
