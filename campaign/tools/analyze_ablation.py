#!/usr/bin/env python3
"""Tests of the operator ablation (experiments/ablation/README.md).

    python analyze.py --config experiments/ablation/campaign.json     (writes pairs.csv)
    python tools/analyze_ablation.py

Per scenario (rain only), paired Wilcoxon over the blocks on the Frechet distance to
the block's unmodified run:
    intensity_only vs noise, geometry_only vs noise,
    intensity_only vs matsense, geometry_only vs matsense,
with one Holm correction over the whole family (4 tests x scenarios). Writes
<out_dir>/analysis/ABLATION.md and tests_ablation.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from matsense_campaign.config import load_campaign  # noqa: E402

CONTRASTS = [("intensity_only", "noise"), ("geometry_only", "noise"),
             ("intensity_only", "matsense"), ("geometry_only", "matsense")]


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    out, run = np.empty(len(p)), 0.0
    for k, i in enumerate(np.argsort(p)):
        run = max(run, min(1.0, (len(p) - k) * p[i]))
        out[i] = run
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="experiments/ablation/campaign.json")
    a = ap.parse_args()
    cfg = load_campaign(a.config)
    an = Path(cfg["_out"]) / "analysis"
    pairs = pd.read_csv(an / "pairs.csv", dtype={"scenario_id": str})
    rows = []
    for (sc, cond), g in pairs.groupby(["scenario_id", "condition"]):
        wide = g.pivot_table(index="block", columns="arm", values="frechet")
        for x, y in CONTRASTS:
            if x not in wide or y not in wide:
                continue
            w = wide[[x, y]].dropna()
            n = len(w)
            d = w[x] - w[y]
            p = float(wilcoxon(w[x], w[y]).pvalue) if n >= 2 and np.any(d != 0) else np.nan
            rows.append({"scenario_id": sc, "condition": cond, "a": x, "b": y, "n_pairs": n,
                         "median_a": w[x].median(), "median_b": w[y].median(), "median_diff": d.median(),
                         "a_below_b": int((d < 0).sum()), "p": p, "min_attainable_p": 2.0 / 2 ** n})
    t = pd.DataFrame(rows)
    t["p_holm"] = holm(t["p"].fillna(1.0).to_numpy())
    t["significant"] = t["p_holm"] < float(cfg["analysis"]["alpha"])
    t.to_csv(an / "tests_ablation.csv", index=False)
    med = pairs.groupby(["scenario_id", "condition", "arm"]).frechet.describe()[["count", "25%", "50%", "75%"]]
    n_runs = len(pairs)
    md = ["# Operator ablation", "",
          f"Pairs available: {n_runs}. Holm over the {len(t)} tests below (family declared in experiments/ablation/README.md).",
          "", "## Frechet distance to the block's unmodified run (m)", "", med.round(3).to_markdown(), "",
          "## Paired tests", "", t.round(4).to_markdown(index=False), ""]
    (an / "ABLATION.md").write_text("\n".join(md), encoding="utf-8")
    print(t.round(3).to_string(index=False))
    print(f"written {an / 'ABLATION.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
