"""ICG+ keypoint texture modality (optional; metallic instruments may have little texture)."""
from __future__ import annotations

import numpy as np


class TextureModality:
    def __init__(self, n_features=400):
        import cv2
        self.orb = cv2.ORB_create(nfeatures=n_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.prev_kp = None
        self.prev_des = None
        self.prev_xyz = None
        self.prev_uv = None

    def correspondences(self, rgb, mask, xyz, min_matches=8):
        import cv2
        image = np.clip(np.asarray(rgb) * 255.0, 0, 255).astype(np.uint8)
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        binary = (np.asarray(mask) > 0.5).astype(np.uint8) * 255
        keypoints, descriptors = self.orb.detectAndCompute(gray, binary)
        matches = []
        if (self.prev_des is not None and descriptors is not None and
                len(keypoints) and len(self.prev_kp)):
            paired = self.matcher.match(self.prev_des, descriptors)
            paired = sorted(paired, key=lambda m: m.distance)[:80]
            h, w = gray.shape
            for m in paired:
                if m.distance > 48:
                    continue
                u0, v0 = self.prev_kp[m.queryIdx].pt
                u1, v1 = keypoints[m.trainIdx].pt
                xi, yi = int(round(u0)), int(round(v0))
                if not (0 <= xi < w and 0 <= yi < h):
                    continue
                X = self.prev_xyz[yi, xi] if self.prev_xyz is not None else xyz[yi, xi]
                if X[2] <= 1e-4:
                    continue
                matches.append({
                    "xyz": np.asarray(X, np.float64),
                    "observed": np.array([u1, v1], np.float64),
                    "weight": float(np.exp(-m.distance / 24.0)),
                    "part_id": None,
                })
        self.prev_kp, self.prev_des, self.prev_xyz = keypoints, descriptors, np.asarray(xyz)
        if len(matches) < min_matches:
            return []
        return matches
