"""Frozen sensing profile: loading, hashing, and class-level permutation.

A profile is a plain JSON object:

    name                  identifier written in every log
    semantic_to_material  CARLA ObjTag (as string) -> material class
    nominal_base          beta_m per class
    weather_ratio         condition -> {class: alpha_m,c}
    measured_entries      which entries were estimated from recordings
    default_material      fallback class name (declared, never measured)

The profile is passed to the operator as a value. What a run records is the
hash of the profile the operator actually applied.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def sha256_of(obj) -> str:
    return hashlib.sha256(_canonical(obj).encode("utf-8")).hexdigest()


@dataclass
class Profile:
    name: str
    semantic_to_material: dict[int, str]
    beta: dict[str, float]
    alpha: dict[str, dict[str, float]]
    measured: list[str]
    default_material: str = "unknown"
    raw: dict = field(default_factory=dict, repr=False)

    @property
    def sha256(self) -> str:
        return sha256_of(self.raw)

    def classes(self) -> list[str]:
        """Measured classes, sorted, so every index below is reproducible."""
        return sorted(self.measured)

    def conditions(self) -> list[str]:
        return sorted(self.alpha)

    def phi(self, cls: str, condition: str) -> float:
        return self.beta[cls] * self.alpha[condition][cls]

    def phi_max(self, condition: str) -> float:
        """Normaliser of Eq. 8: largest beta*alpha over measured classes only."""
        return max(self.phi(c, condition) for c in self.classes())

    def factor_table(self, condition: str) -> dict[str, float]:
        """f_m = phi_m / phi_max for every measured class (all <= 1)."""
        pm = self.phi_max(condition)
        return {c: self.phi(c, condition) / pm for c in self.classes()}

    def check_condition(self, condition: str) -> None:
        if condition not in self.alpha:
            raise ValueError(f"profile {self.name} has no condition '{condition}'")
        declared = set(self.raw.get("measured_entries", {}).get("declared", []))
        if f"{condition}:*" in declared:
            raise ValueError(
                f"condition '{condition}' is declared, not measured, in profile "
                f"{self.name}; refusing to run it as an empirical condition")


def load_profile(path: str | Path) -> Profile:
    return profile_from_raw(json.loads(Path(path).read_text(encoding="utf-8")))


def profile_from_raw(raw: dict) -> Profile:
    measured = list(raw.get("measured_entries", {}).get("nominal_base", []))
    if not measured:
        raise ValueError("profile lists no measured classes (measured_entries.nominal_base)")
    prof = Profile(
        name=str(raw["name"]),
        semantic_to_material={int(k): str(v) for k, v in raw["semantic_to_material"].items()},
        beta={str(k): float(v) for k, v in raw["nominal_base"].items()},
        alpha={str(c): {str(k): float(v) for k, v in row.items()}
               for c, row in raw["weather_ratio"].items()},
        measured=measured,
        default_material=str(raw.get("default_material", "unknown")),
        raw=raw,
    )
    for c in prof.classes():
        if c not in prof.beta:
            raise ValueError(f"measured class '{c}' has no beta")
    return prof


# ---------------------------------------------------------------------------
# Class-level permutation (the "shuffled_class" arm)
# ---------------------------------------------------------------------------

def derangement(n: int, seed: int) -> np.ndarray:
    """Seeded random permutation with no fixed point."""
    rng = np.random.default_rng(seed)
    while True:
        p = rng.permutation(n)
        if n < 2 or not np.any(p == np.arange(n)):
            return p


def class_shuffled(profile: Profile, seed: int) -> tuple[Profile, dict]:
    """Copy of `profile` in which each measured class receives the beta and the
    whole alpha column of another class. Labels are untouched, so geometry and
    class populations do not change. Returns the profile and a record with the
    mapping and its rank correlation with the true assignment."""
    classes = profile.classes()
    perm = derangement(len(classes), seed)
    donor = {classes[i]: classes[perm[i]] for i in range(len(classes))}
    raw = copy.deepcopy(profile.raw)
    for cls, src in donor.items():
        raw["nominal_base"][cls] = profile.beta[src]
        for cond in raw["weather_ratio"]:
            raw["weather_ratio"][cond][cls] = profile.alpha[cond][src]
    raw["name"] = f"{profile.name}__class_shuffle_s{seed}"
    true_b = np.array([profile.beta[c] for c in classes])
    new_b = np.array([profile.beta[donor[c]] for c in classes])
    rank = lambda v: np.argsort(np.argsort(v))
    rho = float(np.corrcoef(rank(true_b), rank(new_b))[0, 1])
    record = {"seed": seed, "donor": donor, "assignment_spearman": round(rho, 3)}
    return profile_from_raw(raw), record
