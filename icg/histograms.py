"""Foreground / background color histograms and region posteriors (ICG / ICG+)."""
from __future__ import annotations

import numpy as np


class RegionHistogram:
    """Normalized RGB histograms for one ICG region (foreground vs background)."""

    def __init__(self, bins=16, learn_rate=0.2):
        self.bins = int(bins)
        self.learn_rate = float(learn_rate)
        n = self.bins ** 3
        self.fg = np.ones(n, np.float64)
        self.bg = np.ones(n, np.float64)

    def _index(self, rgb):
        q = np.clip((np.asarray(rgb, np.float32).reshape(-1, 3) * self.bins).astype(np.int32), 0, self.bins - 1)
        return q[:, 0] * self.bins * self.bins + q[:, 1] * self.bins + q[:, 2]

    def _accumulate(self, rgb):
        hist = np.ones(self.bins ** 3, np.float64)
        if len(rgb):
            np.add.at(hist, self._index(rgb), 1.0)
        hist /= hist.sum()
        return hist

    def update(self, rgb, foreground, background):
        fg_rgb = rgb[foreground]
        bg_rgb = rgb[background]
        if len(fg_rgb):
            self.fg = (1.0 - self.learn_rate) * self.fg + self.learn_rate * self._accumulate(fg_rgb)
        if len(bg_rgb):
            self.bg = (1.0 - self.learn_rate) * self.bg + self.learn_rate * self._accumulate(bg_rgb)
        self.fg /= self.fg.sum()
        self.bg /= self.bg.sum()

    def posterior(self, rgb):
        """P(foreground | color) for an HxWx3 RGB image in [0, 1]."""
        idx = self._index(rgb)
        pf = self.fg[idx].reshape(rgb.shape[:2])
        pb = self.bg[idx].reshape(rgb.shape[:2])
        return pf / np.clip(pf + pb, 1e-12, None)


def mask_posterior(target_mask):
    """Oracle region posterior from a binary observation mask (simulation / debug)."""
    return np.clip(np.asarray(target_mask, np.float32), 0.0, 1.0)


def histogram_samples(pred_mask, width=6):
    """Interior / exterior bands around a predicted silhouette for histogram updates."""
    import cv2
    binary = (np.asarray(pred_mask) > 0.5).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    inner = cv2.erode(binary, kernel, iterations=max(1, width // 2))
    outer = cv2.dilate(binary, kernel, iterations=width)
    return inner.astype(bool), (outer > 0) & (binary == 0)
