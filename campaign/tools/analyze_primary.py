#!/usr/bin/env python3
"""Holm corrections as fixed in ANALYSIS_PLAN.md.

    python analyze.py                  (writes tests.csv with the raw Wilcoxon p values)
    python tools/analyze_primary.py

Primary family: matsense vs uniform and matsense vs shuffled, every scenario and
condition, one Holm correction. Manipulation checks (each realisation vs noise):
their own Holm correction. Writes <out_dir>/analysis/PRIMARY.md and tests_primary.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from matsense_campaign.config import load_campaign  # noqa: E402


def holm(p) -> np.ndarray:
    p = np.asarray(p, float)
    out, run = np.empty(len(p)), 0.0
    for k, i in enumerate(np.argsort(p)):
        run = max(run, min(1.0, (len(p) - k) * p[i]))
        out[i] = run
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/campaign.json")
    a = ap.parse_args()
    cfg = load_campaign(a.config)
    an = Path(cfg["_out"]) / "analysis"
    t = pd.read_csv(an / "tests.csv", dtype={"scenario_id": str})
    alpha = float(cfg["analysis"]["alpha"])
    prim = t[(t.a == "matsense") & t.b.isin(["uniform", "shuffled"])].copy()
    prim["family"] = "primary"
    chk = t[(t.b == "noise") & t.a.isin(["matsense", "uniform", "shuffled"])].copy()
    chk["family"] = "manipulation_check"
    out = []
    for fam in (prim, chk):
        fam["p_holm_family"] = holm(fam["p"].fillna(1.0).to_numpy())
        fam["significant"] = fam["p_holm_family"] < alpha
        out.append(fam)
    res = pd.concat(out)
    cols = ["family", "scenario_id", "condition", "a", "b", "n_pairs", "median_a", "median_b",
            "median_diff", "p", "p_holm_family", "significant"]
    res[cols].to_csv(an / "tests_primary.csv", index=False)
    md = ["# Tests as fixed in ANALYSIS_PLAN.md", "",
          f"Primary family: {len(prim)} tests; manipulation checks: {len(chk)} tests. "
          f"Smallest attainable p with {int(prim.n_pairs.min())} pairs: {2 / 2 ** int(prim.n_pairs.min()):.4f}.", "",
          "## Primary family", "", prim[cols[1:]].round(4).to_markdown(index=False), "",
          "## Manipulation checks", "", chk[cols[1:]].round(4).to_markdown(index=False), ""]
    (an / "PRIMARY.md").write_text("\n".join(md), encoding="utf-8")
    print(prim[cols[1:]].round(3).to_string(index=False))
    print(f"written {an / 'PRIMARY.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
