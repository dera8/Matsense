"""Associate each LiDAR return with the CARLA tag of the nearest semantic ray.

The semantic LiDAR is spawned with the same attributes and the same mounting as
the agent's LiDAR, so both fire the same rays. A return is matched to the
semantic ray with the closest direction, if the angle between them is below the
tolerance (0.40 deg in the manuscript). The search uses a k-d tree on unit
vectors, so a 30k-point frame takes a few milliseconds.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree

SEMANTIC_LIDAR_DTYPE = np.dtype([
    ("x", np.float32), ("y", np.float32), ("z", np.float32),
    ("CosAngle", np.float32), ("ObjIdx", np.uint32), ("ObjTag", np.uint32),
])


def _unit(xyz: np.ndarray) -> np.ndarray:
    r = np.linalg.norm(xyz, axis=1, keepdims=True)
    return xyz / np.maximum(r, 1e-9)


def match_tags(lidar_xyz: np.ndarray, semantic: np.ndarray | None,
               tol_deg: float = 0.40) -> tuple[np.ndarray, np.ndarray]:
    """Return (tags, matched) for every LiDAR return."""
    n = lidar_xyz.shape[0]
    tags = np.zeros(n, dtype=np.int64)
    matched = np.zeros(n, dtype=bool)
    if semantic is None or len(semantic) == 0 or n == 0:
        return tags, matched
    sem_xyz = np.column_stack([semantic["x"], semantic["y"], semantic["z"]]).astype(np.float64)
    tree = cKDTree(_unit(sem_xyz))
    chord = 2.0 * math.sin(math.radians(tol_deg) / 2.0)
    dist, idx = tree.query(_unit(lidar_xyz.astype(np.float64)), k=1,
                           distance_upper_bound=chord)
    ok = np.isfinite(dist)
    tags[ok] = semantic["ObjTag"][idx[ok]].astype(np.int64)
    matched[ok] = True
    return tags, matched
