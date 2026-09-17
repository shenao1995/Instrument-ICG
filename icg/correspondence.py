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


def search_contour(posterior, cx, cy, nx, ny, length=12, scale=2):
    """Find the FG/BG crossing along the correspondence line.

    Returns the 2D observed contour point and a confidence in (0, 1].
    """
    h, w = posterior.shape
    half = int(length)
    rs = np.arange(-half, half + 1, max(1, int(scale)), dtype=np.float64)
    values = []
    coords = []
    for r in rs:
        x, y = cx + nx * r, cy + ny * r
        xi, yi = int(round(x)), int(round(y))
        if not (0 <= xi < w and 0 <= yi < h):
            continue
        values.append(float(posterior[yi, xi]))
        coords.append(np.array([x, y], np.float64))
    if len(values) < 4:
        return None, 0.0
    values = np.asarray(values, np.float64)
    # Outward: posterior should drop through 0.5.
    crossing = None
    for i in range(len(values) - 1):
        a, b = values[i], values[i + 1]
        if (a - 0.5) * (b - 0.5) <= 0 and abs(b - a) > 1e-6:
            t = (0.5 - a) / (b - a)
            crossing = coords[i] * (1 - t) + coords[i + 1] * t
            confidence = min(1.0, abs(b - a) * 2.0)
            return crossing, max(0.15, confidence)
    # Fallback: location closest to 0.5.
    i = int(np.argmin(np.abs(values - 0.5)))
    if abs(values[i] - 0.5) > 0.35:
        return None, 0.0
    return coords[i], 0.25


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
