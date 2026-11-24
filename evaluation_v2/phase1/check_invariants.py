#!/usr/bin/env python3
"""Automatic invariant checks on a paired dataset written by make_modes.py.

FAIL stops the phase. Checks, per frame:

  geometry      every mode has the RAW coordinates, bit for bit, in the same
                order and count, and no field other than intensity
  raw           RAW intensity equals the acquired intensity
  operator (A)  MATSENSE / RAW equals the loaded coefficient f_m of the class of
                each supported return; this checks the operator against the
                profile, it is NOT evidence of realism
  uniform       one scale for the whole cloud, equal to sum(MATSENSE)/sum(RAW)
  shuffled      per range bin, the multiset of applied factors equals MATSENSE's
  fallback      unmatched and unmapped returns keep RAW intensity in every
                material-dependent mode (UNIFORM scales every return by design)
  association   (sequences) on a random subset, the matched semantic ray is the
                nearest by direction and within the tolerance, checked by brute force
  validity      finite intensities in [0, 1]; clipped counts reported

Reported, not failed: the SHUFFLED level residual (a permutation does not keep
the mean transformed intensity) and the unmatched / unmapped shares.

    python check_invariants.py ../data/paired_town03_001
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from modes import load_profile  # noqa: E402
from matsense_campaign.operator import factors  # noqa: E402

RTOL = 1e-5


def main() -> int:
    root = Path(sys.argv[1])
    man = json.loads((root / "manifest.json").read_text())
    st = man["settings"]
    profile = load_profile(man["profile"]["path"])
    if profile.sha256 != man["profile"]["sha256"]:
        print("FAIL profile changed since the modes were made"); return 1
    modes = st["modes"]
    src = man["source"]
    results = {k: {"pass": 0, "fail": 0, "examples": []} for k in
               ("geometry", "raw", "operator_A", "uniform", "shuffled", "fallback", "association", "validity")}
    report = {"frames": []}
    rng = np.random.default_rng(0)

    def ok(name, cond, msg=""):
        results[name]["pass" if cond else "fail"] += 1
        if not cond and len(results[name]["examples"]) < 5:
            results[name]["examples"].append(msg)

    for fr in man["frames"]:
        name = fr["frame"]
        lab = np.load(root / "labels" / f"{name}.npz")
        cls, matched, tag = lab["cls"], lab["matched"], lab["tag"]
        M = {m: np.load(root / "modes" / m / f"{name}.npz") for m in modes}
        raw = M["RAW"]
        xyz0, I0 = raw["xyz"], raw["intensity"].astype(np.float64)
        n = xyz0.shape[0]
        for m in modes:
            d = M[m]
            ok("geometry", set(d.files) == {"xyz", "intensity"} and d["xyz"].dtype == xyz0.dtype
               and d["xyz"].shape == xyz0.shape and np.array_equal(d["xyz"], xyz0)
               and d["intensity"].shape == (n,), f"{name}/{m}")
            v = d["intensity"]
            ok("validity", bool(np.all(np.isfinite(v)) and v.min() >= 0.0 and v.max() <= 1.0), f"{name}/{m}")

        # RAW against the acquisition
        if src["kind"] == "sequence":
            a = np.load(Path(src["path"]) / "frames" / f"{name}.npz")
            ok("raw", np.array_equal(a["raw_intensity"].astype(np.float32), raw["intensity"])
               and np.array_equal(a["raw_xyz"].astype(np.float32), xyz0), name)
        else:
            a = np.load(Path(src["path"]) / "lidar_raw" / f"{name}.npz", allow_pickle=True)
            ok("raw", np.array_equal(a["intensity"].astype(np.float32), raw["intensity"])
               and np.array_equal(a["xyz"].astype(np.float32), xyz0), name)

        sup = cls >= 0
        pos = I0 > 1e-9
        f = factors(profile, st["condition"], cls)
        Im = M["MATSENSE"]["intensity"].astype(np.float64)
        # operator correctness against the loaded coefficients (A)
        sel = sup & pos
        ok("operator_A", bool(np.allclose(Im[sel] / I0[sel], f[sel], rtol=RTOL, atol=1e-6)), name)
        # fallback
        fb = ~sup
        # UNIFORM is material-independent by definition and scales every return;
        # the identity fallback concerns the modes that depend on the class
        ok("fallback", all(np.array_equal(M[m]["intensity"][fb], raw["intensity"][fb])
                           for m in modes if m != "UNIFORM"), name)
        # uniform: a single scale
        Iu = M["UNIFORM"]["intensity"].astype(np.float64)
        s_exp = Im.sum() / I0.sum() if I0.sum() > 0 else 1.0
        ok("uniform", bool(np.allclose(Iu[pos] / I0[pos], s_exp, rtol=RTOL)), name)
        # shuffled: same factor multiset per bin among supported returns
        Is = M["SHUFFLED"]["intensity"].astype(np.float64)
        bins = np.floor(np.linalg.norm(xyz0.astype(np.float64), axis=1) / st["range_bin_m"]).astype(int)
        good = True
        for b in np.unique(bins[sup]):
            c = sup & pos & (bins == b)
            if c.sum() and not np.allclose(np.sort(Is[c] / I0[c]), np.sort(Im[c] / I0[c]), rtol=RTOL, atol=1e-6):
                good = False
                break
        ok("shuffled", good, name)
        # association, brute force on a subset (sequences only)
        if src["kind"] == "sequence":
            sem = a["sem"]
            su = np.column_stack([sem["x"], sem["y"], sem["z"]]).astype(np.float64)
            su /= np.maximum(np.linalg.norm(su, axis=1, keepdims=True), 1e-9)
            pu = xyz0.astype(np.float64) / np.maximum(np.linalg.norm(xyz0, axis=1, keepdims=True), 1e-9)
            pick = rng.choice(n, size=min(300, n), replace=False)
            cosmax = su @ pu[pick].T
            best = np.argmax(cosmax, axis=0)
            ang = np.degrees(np.arccos(np.clip(cosmax[best, np.arange(pick.size)], -1, 1)))
            tol = float(st["tolerance_deg"])
            expect_m = ang <= tol + 1e-6
            agree = (expect_m == matched[pick]) | (np.abs(ang - tol) < 1e-4)
            tag_ok = ~matched[pick] | (sem["ObjTag"][best].astype(int) == tag[pick])
            ok("association", bool(np.all(agree) and np.all(tag_ok)), name)

        m_sum, s_sum = Im.sum(), Is.sum()
        report["frames"].append({
            "frame": name, "n": int(n), "supported_share": float(sup.mean()),
            "unmatched_share": float(np.mean(cls == -1)), "unmapped_share": float(np.mean(cls == -2)),
            "shuffle_level_residual": float((s_sum - m_sum) / m_sum) if m_sum else 0.0,
            "clipped": {m: fr.get(f"n_clipped_{m}", 0) for m in modes},
        })

    fails = sum(r["fail"] for r in results.values())
    print(f"  {len(man['frames'])} frames, modes {modes}\n")
    for k, r in results.items():
        if r["pass"] + r["fail"] == 0:
            print(f"  {'n/a':4s} {k}"); continue
        print(f"  {'PASS' if r['fail'] == 0 else 'FAIL':4s} {k:12s} {r['pass']} ok, {r['fail']} failed"
              + (f"   e.g. {r['examples']}" if r["fail"] else ""))
    res = np.array([f["shuffle_level_residual"] for f in report["frames"]])
    um = np.array([f["unmatched_share"] for f in report["frames"]])
    up = np.array([f["unmapped_share"] for f in report["frames"]])
    clipped = sum(sum(f["clipped"].values()) for f in report["frames"])
    print(f"\n  reported: SHUFFLED level residual median {100 * np.median(res):+.3f}% "
          f"(range {100 * res.min():+.3f}..{100 * res.max():+.3f}%), not corrected")
    print(f"            unmatched {100 * np.median(um):.2f}%  unmapped {100 * np.median(up):.2f}%  "
          f"clipped values {clipped}")
    report["checks"] = results
    report["verdict"] = "PASS" if fails == 0 else "FAIL"
    (root / "invariants.json").write_text(json.dumps(report, indent=1))
    print(f"\n  {report['verdict']} -> {root / 'invariants.json'}")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
