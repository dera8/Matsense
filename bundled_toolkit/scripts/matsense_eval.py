#!/usr/bin/env python3
"""The evaluation layer: the measurements shared by the three evidence levels.

This exists because of a question whose honest answer was no. The architecture
diagram drew an "Evaluation Engine" with three levels inside it, but the
repository had no such thing: it had about thirty-five independent scripts,
each recomputing its own statistics, with five shared imports in total. A
component that is drawn and does not exist is the first thing anyone who opens
the code will notice.

What lives here is the set of measurements the experiments were passing around
by copy, organised along the three levels the paper makes its claims on:

  sensor fidelity   how closely the simulated response matches the measured one
  propagation       how much a perception model notices
  behaviour         what the vehicle does, and how the run ends

plus two things that were written in every script and in no single place: the
paired statistic, and writing an artefact together with its provenance.

Two decisions this module enforces, both here because getting them wrong cost
time:

  a run that died inside the agent node is NOT a failure of its arm. It never
  drove. Counting it inflates the evidence with something that has nothing to
  do with the sensor, and that happened before the counts were separated.

  a run_summary.json written less than ninety seconds ago may still be
  incomplete. Reading one mid-write produced three wrong conclusions in a
  single session, and since then the guard lives in the loader rather than in
  the memory of whoever uses it.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import re
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

APP = Path(__file__).resolve().parents[1]

SETTLE_S = 90.0
HARNESS_MODES = ("agent_error",)
BAND = (20.0, 60.0)
MIN_SUPPORT = 50


# --------------------------------------------------------------------- runs


@dataclass
class Run:
    """One execution, carrying what all three levels need."""
    arm: str
    seed: int
    mode: str                 # completed | collision | stopped | agent_error
    reason: str
    distance_m: float
    cross_track_m: float
    path: Path
    harness_failure: bool = False

    @property
    def completed(self) -> bool:
        return self.mode == "completed"


def load_runs(root, settle_s: float = SETTLE_S, rename=None) -> tuple[list[Run], int]:
    """The runs under `root`, with the settling guard.

    Returns (runs, how many were still being written). Runs that died inside
    the agent node enter the list with harness_failure=True: whoever counts
    outcomes excludes them, whoever takes an inventory still sees them.
    """
    runs, unsettled = [], 0
    for f in sorted(glob.glob(f"{root}/*/run_summary.json")):
        if time.time() - os.path.getmtime(f) < settle_s:
            unsettled += 1
            continue
        d = json.loads(Path(f).read_text())
        name = os.path.basename(os.path.dirname(f))
        m = re.search(r"_0_(\w+?)_(?:nominal|rain|snow)", name)
        arm = m.group(1) if m else d.get("mode", "?")
        if rename:
            arm = rename.get(arm, arm)
        mode = d.get("termination_mode", "?")
        runs.append(Run(
            arm=arm, seed=int(d.get("seed", -1)), mode=mode,
            reason=str(d.get("termination_reason", "")),
            distance_m=float(d.get("route_completion_m") or 0.0),
            cross_track_m=float(d.get("final_cross_track_error_m") or 0.0),
            path=Path(f).parent, harness_failure=mode in HARNESS_MODES))
    return runs, unsettled


def by_arm(runs, drop_harness: bool = True) -> dict[str, list[Run]]:
    out = defaultdict(list)
    for r in runs:
        if drop_harness and r.harness_failure:
            continue
        out[r.arm].append(r)
    return dict(out)


# ------------------------------------------------- level 1: sensor fidelity


def response_by_material(intensity, xyz, materials, band=BAND,
                         min_support: int = MIN_SUPPORT) -> dict[str, float]:
    """Median response per class inside the calibration band.

    Outside the band the classes do not have support in every recording, and a
    median over twenty points is not a response: it is noise with a name.
    """
    r = np.linalg.norm(xyz, axis=1)
    sel_band = (r >= band[0]) & (r < band[1])
    out = {}
    for k in np.unique(materials):
        sel = sel_band & (materials == k)
        if sel.sum() > min_support:
            out[str(k)] = float(np.median(intensity[sel]))
    return out


def normalised(response: dict[str, float]) -> dict[str, float]:
    """Normalised by the strongest class: it is a ratio, not a reflectance."""
    if not response:
        return {}
    mx = max(response.values())
    return {k: v / mx for k, v in response.items()} if mx > 0 else dict(response)


def profile_mae(response: dict[str, float], reference: dict[str, float]) -> float:
    """Mean absolute error against the profile measured on the real data."""
    keys = [k for k in reference if k in response]
    if not keys:
        return float("nan")
    a, b = normalised(response), normalised(reference)
    return float(np.mean([abs(a[k] - b[k]) for k in keys]))


# ------------------------------------- levels 2 and 3: paired statistics


def paired(a, b, label: str = "") -> dict:
    """Paired difference: mean, 95 per cent interval and Wilcoxon.

    Paired because the arms see the same frames or the same seeds: comparing
    unpaired means would throw away exactly the pairing that makes the
    difference attributable.
    """
    from scipy import stats
    d = np.asarray(a, float) - np.asarray(b, float)
    if not np.any(d):
        return {"label": label, "n": int(d.size), "mean": 0.0,
                "ci": [0.0, 0.0], "p": 1.0}
    ci = stats.t.interval(0.95, d.size - 1, loc=d.mean(), scale=stats.sem(d))
    return {"label": label, "n": int(d.size), "mean": float(d.mean()),
            "ci": [float(ci[0]), float(ci[1])],
            "p": float(stats.wilcoxon(d).pvalue)}


def sign_test(d) -> tuple[int, int, float]:
    """How many differences are positive, and how unlikely that is by chance.

    It lives here rather than in a single experiment because the sign is the
    right statistic when the magnitude is not comparable across scenarios: half
    a metre on a bend and half a metre behind a parked car are not the same
    thing, but "it moved in the same direction" is.
    """
    from scipy import stats
    d = np.asarray(d, float)
    k = int((d > 0).sum())
    return k, int(d.size), float(stats.binomtest(k, d.size, 0.5).pvalue)


def fisher(ok_a: int, n_a: int, ok_b: int, n_b: int) -> float:
    """Exact test on two counts: the outcome is categorical, not a mean."""
    from scipy import stats
    return float(stats.fisher_exact([[ok_a, n_a - ok_a], [ok_b, n_b - ok_b]])[1])


def completion(runs: dict[str, list[Run]]) -> dict[str, dict]:
    """How many finish per arm, with the breakdown of outcomes."""
    out = {}
    for arm, v in runs.items():
        modes = defaultdict(int)
        for r in v:
            modes[r.mode] += 1
        out[arm] = {"n": len(v), "completed": sum(1 for r in v if r.completed),
                    "modes": dict(modes),
                    "median_distance_m": float(np.median([r.distance_m for r in v]))
                    if v else float("nan")}
    return out


# ----------------------------------------------------------- provenance


def _git_rev() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=APP,
                              capture_output=True, text=True, timeout=5
                              ).stdout.strip() or "?"
    except Exception:                                      # noqa: BLE001
        return "?"


def _digest(path) -> str:
    p = Path(path)
    if not p.exists():
        return "missing"
    if p.is_dir():
        return f"dir:{len(list(p.glob('*')))} entries"
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def artefact(path, payload: dict, tool: str = "", inputs=(), quiet: bool = False):
    """Write the JSON with its provenance attached.

    Every script used to write a bare dictionary, and weeks later nobody could
    tell which version of the code had produced it, or from what data. Here the
    file carries the tool, the repository revision, the moment, and a digest of
    its inputs.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    out = dict(payload)
    out["_provenance"] = {
        "tool": tool or Path(os.sys.argv[0]).name,
        "git": _git_rev(),
        "written": time.strftime("%Y-%m-%d %H:%M:%S"),
        "inputs": {str(i): _digest(i) for i in inputs},
    }
    p.write_text(json.dumps(out, indent=1))
    if not quiet:
        print(f"  written {p}")
    return p
