"""The four paired sensing modes of phase 1, intensity-only, as pure functions.

Every mode receives the SAME cloud and returns a cloud with the same points, in
the same order, with the same coordinates: only the intensity column may change.
No return is removed and none is moved (no dropout, no jitter), so the modes can
be compared point by point.

    RAW        CARLA's intensity, unchanged
    UNIFORM    I' = s * I, one scale per cloud, s = sum(f I) / sum(I): the
               intensity-weighted level MATSENSE delivers on this cloud
    MATSENSE   I' = f_m * I, f_m = beta_m alpha_m,c / phi_max,c from the frozen
               profile, for returns whose class is calibrated
    SHUFFLED   I' = f_pi(i) * I, the MATSENSE factors permuted among the
               supported returns of each range bin. A permutation keeps the
               multiset of factors, NOT the mean transformed intensity: the
               residual level difference is measured and reported, never
               corrected inside this mode.
    SHUFFLED_LEVEL (optional, separate control)
               SHUFFLED rescaled by one scalar so that sum(I') equals MATSENSE.
               It restores the level and gives up the exact factor multiset
               (every factor is multiplied by the same scalar, and may clip).

Returns whose class is not calibrated keep f = 1 (explicit identity fallback)
in the material-dependent modes, MATSENSE, SHUFFLED and SHUFFLED_LEVEL:
unmatched (no semantic partner) and unmapped (a CARLA tag the profile does not
map). They stay out of the shuffle pool. UNIFORM is material-independent by
definition and scales every return, these included; its scale counts them with
f = 1, so its level equals MATSENSE's.

The coefficients and the tag -> class mapping are read from the campaign
package (campaign/matsense_campaign), the code the paper's runs used.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CAMPAIGN = Path(__file__).resolve().parents[2] / "campaign"
if str(CAMPAIGN) not in sys.path:
    sys.path.insert(0, str(CAMPAIGN))

from matsense_campaign.operator import UNMAPPED, UNMATCHED, class_index, factors  # noqa: E402
from matsense_campaign.profile import Profile, load_profile  # noqa: E402

MODES = ("RAW", "UNIFORM", "MATSENSE", "SHUFFLED")
OPTIONAL_MODES = ("SHUFFLED_LEVEL",)
CLIP = (0.0, 1.0)


@dataclass(frozen=True)
class ModeSettings:
    condition: str = "nominal"
    range_bin_m: float = 5.0
    shuffle_seed: int = 0


def classes_for(tags: np.ndarray, matched: np.ndarray, profile: Profile) -> np.ndarray:
    """Class index per return, UNMATCHED (-1) or UNMAPPED (-2), as the campaign."""
    return class_index(tags.astype(np.int64), matched.astype(bool), profile)


def _permute_supported_within_bins(f: np.ndarray, rng_bins: np.ndarray, supported: np.ndarray,
                                   rng: np.random.Generator) -> np.ndarray:
    out = f.copy()
    idx_sup = np.flatnonzero(supported)
    bins = rng_bins[idx_sup]
    for b in np.unique(bins):
        cell = idx_sup[bins == b]
        if cell.size > 1:
            out[cell] = f[cell][rng.permutation(cell.size)]
    return out


def make_modes(xyz: np.ndarray, intensity: np.ndarray, cls: np.ndarray, profile: Profile,
               settings: ModeSettings, with_optional: bool = True) -> tuple[dict, dict]:
    """Return ({mode: intensity array}, diagnostics). xyz is never modified."""
    I = intensity.astype(np.float64)
    f = factors(profile, settings.condition, cls)            # 1 where cls < 0
    supported = cls >= 0
    out, diag = {}, {}

    out["RAW"] = I.copy()
    matsense = f * I
    out["MATSENSE"] = matsense
    denom = float(I.sum())
    s = float(matsense.sum() / denom) if denom > 0 else 1.0
    out["UNIFORM"] = s * I

    rng = np.random.default_rng(settings.shuffle_seed)
    r = np.linalg.norm(xyz.astype(np.float64), axis=1)
    bins = np.floor(r / settings.range_bin_m).astype(np.int64)
    f_perm = _permute_supported_within_bins(f, bins, supported, rng)
    out["SHUFFLED"] = f_perm * I

    m_sum = float(out["MATSENSE"].sum())
    sh_sum = float(out["SHUFFLED"].sum())
    diag.update(
        uniform_scale=s,
        shuffle_moved_frac=float(np.mean(f_perm[supported] != f[supported])) if supported.any() else 0.0,
        shuffle_level_residual=(sh_sum - m_sum) / m_sum if m_sum > 0 else 0.0,
        factor_multiset_preserved=bool(np.array_equal(np.sort(f_perm[supported]), np.sort(f[supported]))),
    )
    if with_optional:
        # only the supported returns are rescaled, so the fallback still holds:
        # sum over supported of k * f_perm * I = MATSENSE's sum over supported
        sup_m = float(out["MATSENSE"][supported].sum())
        sup_s = float(out["SHUFFLED"][supported].sum())
        k = sup_m / sup_s if sup_s > 0 else 1.0
        lvl = out["SHUFFLED"].copy()
        lvl[supported] *= k
        out["SHUFFLED_LEVEL"] = lvl
        diag["shuffled_level_scale"] = k

    lo, hi = CLIP
    for m in list(out):
        v = out[m]
        diag[f"n_clipped_{m}"] = int(np.sum((v < lo) | (v > hi)))
        out[m] = np.clip(v, lo, hi).astype(np.float32)
    diag.update(n=int(I.size), n_supported=int(supported.sum()),
                n_unmatched=int(np.sum(cls == UNMATCHED)), n_unmapped=int(np.sum(cls == UNMAPPED)))
    return out, diag


__all__ = ["MODES", "OPTIONAL_MODES", "ModeSettings", "classes_for", "make_modes", "load_profile",
           "UNMATCHED", "UNMAPPED"]
