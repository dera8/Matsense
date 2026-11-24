"""The MatSense sensing operator and its control arms, as pure numpy.

Every arm receives the same inputs:

    xyz        (N, 3) float32  points in the sensor frame
    intensity  (N,)   float32  CARLA intensity
    cls        (N,)   int      index into profile.classes() for mapped returns,
                               UNMATCHED (-1) when no semantic ray was found,
                               UNMAPPED (-2) when the tag has no calibrated class

Unmatched and unmapped returns are always left as the simulator produced them
(identity policy), so no result depends on a coefficient that was not estimated.

Operator (manuscript Eqs. 7-10):

    f_i  = beta_m alpha_m,c / phi_max,c            (f_i <= 1, nothing clips)
    p_i  = min(p_max, lambda_d max(1 - alpha_m,c, 0))   return loss
    x'_i = x_i + eps_i u_i,  eps_i ~ N(0, (lambda_j p_i)^2)  range jitter

Arms:

    unmodified        pass-through
    matsense          loss + jitter + intensity f_i
    uniform           same retained points and jitter as matsense (geometry
                      "matched", default) or no loss/jitter ("none", as in the
                      manuscript); one scalar s = sum(f I)/sum(I) per frame
    shuffled          matsense factors permuted among the retained points inside
                      range bins, rescaled so sum(f I) is unchanged
    shuffled_class    factors of a class-deranged profile, same geometry as
                      matsense, rescaled to the same level
    intensity_only    intensity f_i, no loss, no jitter        (ablation)
    geometry_only     loss + jitter, intensity unchanged        (ablation)

All arms draw the same random numbers in the same order, so with the same seed
the same returns are dropped and displaced in every arm that removes points.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .profile import Profile

UNMATCHED = -1
UNMAPPED = -2

ARMS = ("unmodified", "matsense", "uniform", "shuffled", "shuffled_class",
        "intensity_only", "geometry_only")
GEOMETRY_ARMS = {"matsense", "shuffled", "shuffled_class", "geometry_only"}


@dataclass(frozen=True)
class OperatorParams:
    dropout_gain: float = 0.4        # lambda_d
    dropout_cap: float = 0.55        # p_max
    jitter_gain: float = 0.10        # lambda_j (m)
    range_bin_m: float = 5.0         # bins for the point-level shuffle
    uniform_geometry: str = "matched"  # "matched" | "none"
    intensity_clip: tuple = (0.0, 1.0)


def class_index(tags: np.ndarray, matched: np.ndarray, profile: Profile) -> np.ndarray:
    """Map CARLA ObjTags to class indices with the two fallback codes."""
    classes = profile.classes()
    lut = {t: classes.index(m) for t, m in profile.semantic_to_material.items() if m in classes}
    out = np.full(tags.shape[0], UNMAPPED, dtype=np.int32)
    for t, i in lut.items():
        out[tags == t] = i
    out[~matched] = UNMATCHED
    return out


def _tables(profile: Profile, condition: str):
    classes = profile.classes()
    f = profile.factor_table(condition)
    f_tab = np.array([f[c] for c in classes], dtype=np.float64)
    a_tab = np.array([profile.alpha[condition][c] for c in classes], dtype=np.float64)
    return f_tab, a_tab


def factors(profile: Profile, condition: str, cls: np.ndarray) -> np.ndarray:
    f_tab, _ = _tables(profile, condition)
    f = np.ones(cls.shape[0], dtype=np.float64)
    m = cls >= 0
    f[m] = f_tab[cls[m]]
    return f


def loss_probability(profile: Profile, condition: str, cls: np.ndarray,
                     p: OperatorParams) -> np.ndarray:
    _, a_tab = _tables(profile, condition)
    out = np.zeros(cls.shape[0], dtype=np.float64)
    m = cls >= 0
    out[m] = np.minimum(p.dropout_cap, p.dropout_gain * np.maximum(1.0 - a_tab[cls[m]], 0.0))
    return out


def _permute_within_bins(values: np.ndarray, rng_key: np.ndarray,
                         rng: np.random.Generator) -> np.ndarray:
    out = values.copy()
    for b in np.unique(rng_key):
        idx = np.flatnonzero(rng_key == b)
        if idx.size > 1:
            out[idx] = values[rng.permutation(idx)]
    return out


def apply_arm(arm: str, xyz: np.ndarray, intensity: np.ndarray, cls: np.ndarray,
              condition: str, profile: Profile, params: OperatorParams,
              rng: np.random.Generator, shuffle_rng: np.random.Generator | None = None,
              class_shuffled_profile: Profile | None = None):
    """Return (xyz_out, intensity_out, keep_mask, stats)."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm '{arm}'")
    n = xyz.shape[0]
    I = intensity.astype(np.float64)
    stats = {"n_in": n, "n_unmatched": int(np.sum(cls == UNMATCHED)),
             "n_unmapped": int(np.sum(cls == UNMAPPED))}

    # The random draws happen in every arm, in the same order, so arms stay paired.
    u_drop = rng.random(n)
    z_jit = rng.standard_normal(n)

    if arm == "unmodified":
        stats.update(n_out=n, level_ratio=1.0, scale=1.0, n_clipped=0)
        return xyz, intensity, np.ones(n, dtype=bool), stats

    p = loss_probability(profile, condition, cls, params)
    use_geometry = arm in GEOMETRY_ARMS or (arm == "uniform" and params.uniform_geometry == "matched")
    if use_geometry:
        keep = u_drop >= p
        r = np.linalg.norm(xyz, axis=1, keepdims=True)
        u = xyz / np.maximum(r, 1e-6)
        eps = z_jit * params.jitter_gain * p
        xyz_new = xyz + (eps[:, None] * u)
    else:
        keep = np.ones(n, dtype=bool)
        xyz_new = xyz.astype(np.float64)

    f = factors(profile, condition, cls)
    xyz_k, I_k, f_k, cls_k = xyz_new[keep], I[keep], f[keep], cls[keep]
    target = float(np.sum(f_k * I_k))          # level delivered by matsense
    denom = float(np.sum(I_k)) or 1.0
    scale = 1.0

    if arm in ("matsense", "intensity_only"):
        out = f_k * I_k
    elif arm == "geometry_only":
        out = I_k
    elif arm == "uniform":
        scale = target / denom
        out = scale * I_k
    elif arm == "shuffled":
        rng2 = shuffle_rng or np.random.default_rng(0)
        bins = np.floor(np.linalg.norm(xyz_k, axis=1) / params.range_bin_m).astype(np.int64)
        f_perm = _permute_within_bins(f_k, bins, rng2)
        got = float(np.sum(f_perm * I_k)) or 1.0
        scale = target / got
        out = f_perm * I_k * scale
    elif arm == "shuffled_class":
        if class_shuffled_profile is None:
            raise ValueError("shuffled_class needs class_shuffled_profile")
        f_sc = factors(class_shuffled_profile, condition, cls_k)
        got = float(np.sum(f_sc * I_k)) or 1.0
        scale = target / got
        out = f_sc * I_k * scale
    else:  # pragma: no cover
        raise AssertionError(arm)

    lo, hi = params.intensity_clip
    n_clipped = int(np.sum((out < lo) | (out > hi)))
    out = np.clip(out, lo, hi)
    stats.update(
        n_out=int(keep.sum()),
        level_ratio=float(np.sum(out) / denom),
        scale=float(scale),
        n_clipped=n_clipped,
        mean_factor=float(np.mean(f_k)) if f_k.size else 1.0,
    )
    for i, c in enumerate(profile.classes()):
        stats[f"n_{c}"] = int(np.sum(cls_k == i))
    return xyz_k.astype(np.float32), out.astype(np.float32), keep, stats
