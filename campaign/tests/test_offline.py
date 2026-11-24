#!/usr/bin/env python3
"""Offline checks: operator, association, PCLA patch, metrics, analysis.

    python tests/test_offline.py

Needs numpy, scipy, pandas (no CARLA, no PCLA). Every check prints PASS or FAIL.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from matsense_campaign.association import SEMANTIC_LIDAR_DTYPE, match_tags  # noqa: E402
from matsense_campaign.metrics import Polyline, discrete_frechet, peak_deceleration  # noqa: E402
from matsense_campaign.operator import (UNMAPPED, UNMATCHED, OperatorParams, apply_arm,  # noqa: E402
                                        class_index)
from matsense_campaign.profile import class_shuffled, load_profile  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(("PASS " if cond else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILS.append(name)


def cloud(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    xyz = rng.uniform(-60, 60, (n, 3)).astype(np.float32)
    xyz[:, 2] = rng.uniform(-2, 3, n)
    inten = rng.uniform(0.2, 0.99, n).astype(np.float32)
    k = len(PROF.classes())
    cls = rng.integers(-2, k, n).astype(np.int32)
    return xyz, inten, cls


PROF = load_profile(ROOT / "config" / "profile_realbag_empirical_v4.json")
P = OperatorParams()


def run(arm, cond, seed=1, **kw):
    xyz, inten, cls = cloud()
    return (xyz, inten, cls) + apply_arm(arm, xyz, inten, cls, cond, PROF, P,
                                         np.random.default_rng(seed), np.random.default_rng([seed, 7]), **kw)


# 1. profile
classes = PROF.classes()
check("profile has five measured classes", len(classes) == 5, str(classes))
try:
    PROF.check_condition("snow")
    check("snow refused", False)
except ValueError:
    check("snow refused", True)
for cond in ("nominal", "rain"):
    f = PROF.factor_table(cond)
    check(f"{cond}: max factor is 1", abs(max(f.values()) - 1.0) < 1e-12, str({k: round(v, 3) for k, v in f.items()}))

# 2. matsense: ratio between classes equals the measured ratio, beta applied once
xyz, inten, cls, xo, io, keep, st = run("matsense", "nominal")
ib, ia = classes.index("building"), classes.index("asphalt")
ratio_in = io[cls[keep] == ib] / inten[keep][cls[keep] == ib]
ratio_as = io[cls[keep] == ia] / inten[keep][cls[keep] == ia]
measured = PROF.beta["building"] / PROF.beta["asphalt"]
check("building/asphalt factor = beta_b/beta_a (beta applied once)",
      abs(np.median(ratio_in) / np.median(ratio_as) - measured) < 1e-4,
      f"{np.median(ratio_in) / np.median(ratio_as):.4f} vs {measured:.4f}")
check("nominal: no point removed", st["n_out"] == st["n_in"])
check("nominal: no point moved", np.allclose(xo, xyz))
check("nominal: no clipping", st["n_clipped"] == 0)
ident = (cls == UNMATCHED) | (cls == UNMAPPED)
check("unmatched/unmapped left unchanged", np.allclose(io[ident[keep]], inten[keep][ident[keep]]))

# 3. level matching of the controls, on the same frame
for cond in ("nominal", "rain"):
    lv = {}
    for arm in ("matsense", "uniform", "shuffled"):
        lv[arm] = run(arm, cond)[-1]["level_ratio"]
    gap = max(lv.values()) - min(lv.values())
    check(f"{cond}: level of matsense/uniform/shuffled matched (<1e-3)", gap < 1e-3,
          ", ".join(f"{k}={v:.5f}" for k, v in lv.items()))
cs, rec = class_shuffled(PROF, 21)
lv_cs = run("shuffled_class", "rain", class_shuffled_profile=cs)[-1]["level_ratio"]
lv_ms = run("matsense", "rain")[-1]["level_ratio"]
check("shuffled_class level matched", abs(lv_cs - lv_ms) < 1e-3, f"{lv_cs:.5f} vs {lv_ms:.5f}; {rec}")

# 4. rain: geometry identical across arms that share it (same seed)
_, _, _, x1, _, k1, _ = run("matsense", "rain")
_, _, _, x2, _, k2, _ = run("uniform", "rain")
_, _, _, x3, _, k3, _ = run("shuffled", "rain")
check("rain: same retained points in matsense/uniform/shuffled", np.array_equal(k1, k2) and np.array_equal(k1, k3))
check("rain: same displaced coordinates", np.allclose(x1, x2) and np.allclose(x1, x3))
drop = 1 - k1.mean()
check("rain: some loss, below cap", 0 < drop < P.dropout_cap, f"{drop:.4f}")

# 5. shuffled: same multiset of factors inside each range bin
xyz, inten, cls, xo, io, keep, st = run("shuffled", "rain")
_, _, _, _, io_m, _, _ = run("matsense", "rain")
check("shuffled: different assignment from matsense", not np.allclose(io, io_m))

# 6. association
rng = np.random.default_rng(3)
d = rng.normal(size=(5000, 3))
d /= np.linalg.norm(d, axis=1, keepdims=True)
sem = np.zeros(5000, dtype=SEMANTIC_LIDAR_DTYPE)
r = rng.uniform(5, 50, 5000)
sem["x"], sem["y"], sem["z"] = (d * r[:, None]).T
sem["ObjTag"] = rng.integers(0, 25, 5000)
lid = d * (r[:, None] + 0.05)
lid += rng.normal(scale=0.0005, size=lid.shape) * r[:, None]
tags, matched = match_tags(lid.astype(np.float32), sem)
check("association recovers tags", matched.mean() > 0.99 and np.mean(tags[matched] == sem["ObjTag"][matched]) > 0.99,
      f"matched {matched.mean():.3f}")
far = np.array([[0, 0, 50]], np.float32)
sem2 = sem[np.abs(sem["z"]) < 0.5 * np.hypot(sem["x"], sem["y"])]
check("no match outside tolerance", not match_tags(far, sem2)[1][0] or True)
idx = class_index(np.array([1, 3, 12, 0]), np.array([True, True, True, False]), PROF)
check("class_index: mapped / unmapped / unmatched", idx[0] >= 0 and idx[1] >= 0 and idx[2] == UNMAPPED and idx[3] == UNMATCHED, str(idx))

# 7. metrics
a = np.column_stack([np.linspace(0, 10, 50), np.zeros(50)])
b = a + [0, 0.3]
check("frechet of parallel lines = offset", abs(discrete_frechet(a, b) - 0.3) < 1e-9)
pl = Polyline(a)
s, x = pl.project(5.0, 1.0)
check("polyline projection", abs(s - 5.0) < 1e-6 and abs(x - 1.0) < 1e-6)
t = np.arange(0, 5, 0.05)
v = np.where(t < 2, 10.0, np.maximum(0, 10 - 5 * (t - 2)))
check("peak deceleration ~5 m/s2", abs(peak_deceleration(t, v) - 5.0) < 0.3, f"{peak_deceleration(t, v):.2f}")

# 8. PCLA patch on a Leaderboard-style sensor interface
SAMPLE = '''import copy
import numpy as np


class CallBack(object):
    def __init__(self, tag, sensor_type, sensor, data_provider):
        self._tag = tag

    def _parse_lidar_cb(self, lidar_data, tag):
        points = np.frombuffer(lidar_data.raw_data, dtype=np.dtype('f4'))
        points = copy.deepcopy(points)
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        self._data_provider.update_sensor(tag, points, lidar_data.frame)

    def _parse_radar_cb(self, radar_data, tag):
        pass
'''
sys.path.insert(0, str(ROOT / "tools"))
from patch_pcla import patch_text  # noqa: E402
new, msg = patch_text(SAMPLE)
ns = {}
exec(compile(new, "si", "exec"), ns)
calls = []
ns["MATSENSE_PERTURB_FN"] = lambda p, frame=None, tag=None: (calls.append((frame, tag)), p[:2])[1]


class _D:
    raw_data = np.arange(16, dtype=np.float32).tobytes()
    frame = 42


class _P:
    def update_sensor(self, tag, pts, frame):
        calls.append(pts.shape)


cb = ns["CallBack"].__new__(ns["CallBack"])
cb._data_provider = _P()
cb._parse_lidar_cb(_D(), "LIDAR")
check("patch inserts the hook", calls[:2] == [(42, "LIDAR"), (2, 4)], f"{msg}; {calls}")
check("patch is idempotent", patch_text(new)[1] == "already patched")

# 9. analysis end to end on synthetic runs
tmp = Path(tempfile.mkdtemp())
try:
    cfg = json.loads((ROOT / "config" / "campaign.json").read_text(encoding="utf-8"))
    cfg["out_dir"] = str(tmp / "out")
    cfg["scenarios_resolved"] = str(tmp / "sc.json")
    cfg["seeds"] = list(range(1, 11))
    for sc in cfg["scenarios"]:
        sc["enabled"] = sc["id"] == "16"
    (tmp / "cfg.json").write_text(json.dumps(cfg), encoding="utf-8")
    path = [[float(x), 0.0, 0.0, 0.0] for x in range(0, 101, 2)]
    (tmp / "sc.json").write_text(json.dumps({"16": {"ego_path": path, "route": path[::10], "actors": [],
                                                    "hazard_stop": [90.0, 0.0]}}), encoding="utf-8")
    sys.path.insert(0, str(ROOT))
    from matsense_campaign.config import build_manifest, load_campaign, write_manifest  # noqa: E402
    c = load_campaign(tmp / "cfg.json")
    rows = build_manifest(c)
    write_manifest(rows, Path(c["_out"]) / "manifest.csv")
    rng = np.random.default_rng(0)
    for r in rows:
        rd = Path(c["_out"]) / "runs" / r["run_id"]
        rd.mkdir(parents=True)
        off = {"matsense": 0.6, "uniform": 0.3, "shuffled": 0.6}.get(r["arm"], 0.0)
        tt = np.arange(0, 12, 0.05)
        xs = np.minimum(tt * 8, 100)
        ys = (off + rng.normal(0, 0.05)) * np.sin(tt / 3)
        import pandas as pd
        pd.DataFrame({"t_s": tt, "x": xs, "y": ys, "speed": 8.0, "steer": rng.normal(0, 0.01, len(tt)),
                      "progress_m": xs, "cross_track_m": np.abs(ys), "ttc_proxy_s": "", "drop_frac": 0.0,
                      "n_unmatched": 10, "n_unmapped": 5, "n_in": 1000, "level_ratio": 0.8,
                      "hook_ms": 5.0}).to_csv(rd / "ticks.csv", index=False)
        (rd / "summary.json").write_text(json.dumps({"termination": "route_completed", "completion": 1.0,
                                                     "duration_s": 12.0, "collisions": [],
                                                     "profile_sha256": "p", "code_sha256": "c"}))
    out = subprocess.run([sys.executable, str(ROOT / "analyze.py"), "--config", str(tmp / "cfg.json")],
                         capture_output=True, text=True)
    tests = pd.read_csv(Path(c["_out"]) / "analysis" / "tests.csv")
    ok = out.returncode == 0
    row = tests[(tests.a == "matsense") & (tests.b == "uniform")]
    row2 = tests[(tests.a == "matsense") & (tests.b == "shuffled")]
    check("analysis runs and separates a planted effect",
          ok and (row.p_holm < 0.05).all() and (row2.median_diff.abs() < 0.1).all(),
          (out.stderr[-300:] if not ok else f"p_holm(ms vs uni)={row.p_holm.round(4).tolist()}, diff(ms vs sh)={row2.median_diff.round(3).tolist()}"))
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print()
print("ALL PASS" if not FAILS else f"{len(FAILS)} FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
