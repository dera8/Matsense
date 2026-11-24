"""Trajectory metrics (no CARLA dependency)."""
from __future__ import annotations

import numpy as np


class Polyline:
    """Route polyline with projection: progress along it and lateral distance."""

    def __init__(self, xy: np.ndarray):
        self.p = np.asarray(xy, dtype=np.float64)[:, :2]
        seg = np.diff(self.p, axis=0)
        self.seg = seg
        self.len2 = np.maximum(np.sum(seg * seg, axis=1), 1e-12)
        self.cum = np.concatenate([[0.0], np.cumsum(np.sqrt(self.len2))])
        self.length = float(self.cum[-1])

    def project(self, x: float, y: float) -> tuple[float, float]:
        q = np.array([x, y])
        t = np.clip(np.sum((q - self.p[:-1]) * self.seg, axis=1) / self.len2, 0.0, 1.0)
        foot = self.p[:-1] + t[:, None] * self.seg
        d = np.hypot(*(foot - q).T)
        k = int(np.argmin(d))
        return float(self.cum[k] + t[k] * np.sqrt(self.len2[k])), float(d[k])


def discrete_frechet(a: np.ndarray, b: np.ndarray) -> float:
    """Discrete Frechet distance between two (n, 2) polylines."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    d = np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1])
    n, m = d.shape
    ca = np.empty((n, m))
    ca[0, 0] = d[0, 0]
    for j in range(1, m):
        ca[0, j] = max(ca[0, j - 1], d[0, j])
    for i in range(1, n):
        ca[i, 0] = max(ca[i - 1, 0], d[i, 0])
        prev, row, di = ca[i - 1], ca[i], d[i]
        for j in range(1, m):
            row[j] = max(min(prev[j], prev[j - 1], row[j - 1]), di[j])
    return float(ca[-1, -1])


def resample_time(t: np.ndarray, xy: np.ndarray, hz: float) -> np.ndarray:
    if len(t) < 2:
        return xy
    grid = np.arange(t[0], t[-1] + 1e-9, 1.0 / hz)
    return np.column_stack([np.interp(grid, t, xy[:, 0]), np.interp(grid, t, xy[:, 1])])


def peak_deceleration(t: np.ndarray, speed: np.ndarray, window_s: float = 0.4,
                      skip_s: float = 1.0) -> float:
    """Largest deceleration sustained over `window_s`, ignoring the first `skip_s`."""
    m = t >= t[0] + skip_s
    t, v = t[m], speed[m]
    best = 0.0
    j = 0
    for i in range(len(t)):
        while j < len(t) and t[j] - t[i] < window_s:
            j += 1
        if j >= len(t):
            break
        dec = (v[i] - v[j]) / max(t[j] - t[i], 1e-6)
        best = max(best, dec)
    return float(best)


def steering_jitter(steer: np.ndarray) -> float:
    return float(np.mean(np.abs(np.diff(steer)))) if len(steer) > 1 else 0.0
