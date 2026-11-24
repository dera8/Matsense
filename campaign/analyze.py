#!/usr/bin/env python3
"""Analyse a completed (or partial) campaign.

    python analyze.py --config config/campaign.json

Outputs in <out_dir>/analysis/:
    runs.csv        one row per run: outcome, behaviour metrics, sensing statistics
    pairs.csv       Frechet distance of every run to the unmodified run of its block
    tests.csv       paired Wilcoxon tests, Holm-adjusted over the whole family
    REPORT.md       tables in reading order
    fig_<scenario>.png

Design of the comparison
------------------------
Each block (scenario, condition, seed, replicate) holds one run per arm plus a
second unmodified run. The reference trajectory of a block is its first
unmodified run. The second unmodified run measures how far two runs with the same
input drift apart: that is the noise reference every effect must exceed.

Primary tests, per scenario and condition, paired over blocks, on the Frechet
distance to the block reference:
    matsense vs uniform      does the material structure matter beyond level?
    matsense vs shuffled     does the correct assignment matter?
    <arm> vs noise           does the arm move the vehicle more than chance?
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from matsense_campaign.config import load_campaign, read_manifest, resolve_path  # noqa: E402
from matsense_campaign.metrics import (discrete_frechet, peak_deceleration,  # noqa: E402
                                       resample_time, steering_jitter)


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    ok = np.isfinite(p)
    out = np.full_like(p, np.nan)
    idx = np.flatnonzero(ok)
    order = idx[np.argsort(p[idx])]
    m, running = len(order), 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, (m - k) * p[i]))
        out[i] = running
    return out


def run_metrics(run_dir: Path, inst: dict, safety_offset: float) -> dict | None:
    s_path, t_path = run_dir / "summary.json", run_dir / "ticks.csv"
    if not s_path.exists() or not t_path.exists():
        return None
    s = json.loads(s_path.read_text(encoding="utf-8"))
    d = pd.read_csv(t_path)
    if d.empty:
        return None
    t, v = d["t_s"].to_numpy(), d["speed"].to_numpy()
    failed = s["termination"] in ("route_deviation", "collision", "stationary", "never_started")
    keep = d["progress_m"] <= d["progress_m"].max() - (5.0 if failed else 0.0)
    out = {
        "termination": s["termination"], "completion": s["completion"],
        "n_collisions": len(s.get("collisions", [])), "duration_s": s["duration_s"],
        "cross_track_mean": float(d.loc[keep, "cross_track_m"].mean()) if keep.any() else np.nan,
        "steer_jitter": steering_jitter(d["steer"].to_numpy()),
        "peak_decel": peak_deceleration(t, v),
        "min_ttc": pd.to_numeric(d["ttc_proxy_s"], errors="coerce").min(),
        "drop_frac_mean": pd.to_numeric(d["drop_frac"], errors="coerce").mean(),
        "unmatched_frac": (pd.to_numeric(d["n_unmatched"], errors="coerce")
                           / pd.to_numeric(d["n_in"], errors="coerce")).mean(),
        "unmapped_frac": (pd.to_numeric(d["n_unmapped"], errors="coerce")
                          / pd.to_numeric(d["n_in"], errors="coerce")).mean(),
        "level_ratio_mean": pd.to_numeric(d["level_ratio"], errors="coerce").mean(),
        "hook_ms_p95": pd.to_numeric(d["hook_ms"], errors="coerce").quantile(0.95),
        "profile_sha256": s.get("profile_sha256"), "code_sha256": s.get("code_sha256"),
    }
    hs = inst.get("hazard_stop")
    if hs:
        x, y = d["x"].iloc[-1], d["y"].iloc[-1]
        out["safety_margin"] = float(np.hypot(x - hs[0], y - hs[1]) - safety_offset)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/campaign.json")
    args = ap.parse_args()
    cfg = load_campaign(args.config)
    out = Path(cfg["_out"])
    an = out / "analysis"
    an.mkdir(parents=True, exist_ok=True)
    scen = json.loads(resolve_path(cfg["scenarios_resolved"]).read_text(encoding="utf-8"))
    hz = float(cfg["analysis"]["frechet_hz"])
    rows = read_manifest(out / "manifest.csv")

    recs, traj = [], {}
    for r in rows:
        rd = out / "runs" / r["run_id"]
        m = run_metrics(rd, scen[r["scenario_id"]], float(cfg["analysis"]["safety_offset_m"]))
        if m is None:
            continue
        d = pd.read_csv(rd / "ticks.csv", usecols=["t_s", "x", "y"])
        traj[r["run_id"]] = resample_time(d["t_s"].to_numpy(), d[["x", "y"]].to_numpy(), hz)
        recs.append({**r, **m})
    if not recs:
        print("no completed runs")
        return 1
    runs = pd.DataFrame(recs)
    runs["repeat"] = runs["repeat"].astype(int)
    runs.to_csv(an / "runs.csv", index=False)

    # every profile/code hash must be identical, or the campaign mixes versions
    for col in ("profile_sha256", "code_sha256"):
        if runs[col].nunique() > 1:
            print(f"WARNING: {col} differs across runs; do not pool them")

    pairs = []
    for block, g in runs.groupby("block"):
        ref = g[(g.arm == "unmodified") & (g.repeat == 0)]
        if ref.empty:
            continue
        ref_id = ref.iloc[0]["run_id"]
        for _, x in g.iterrows():
            if x["run_id"] == ref_id:
                continue
            label = "noise" if (x["arm"] == "unmodified" and x["repeat"] == 1) else x["arm"]
            pairs.append({"block": block, "scenario_id": x["scenario_id"], "condition": x["condition"],
                          "seed": x["seed"], "replicate": x["replicate"], "arm": label,
                          "frechet": discrete_frechet(traj[x["run_id"]], traj[ref_id])})
    pairs = pd.DataFrame(pairs)
    pairs.to_csv(an / "pairs.csv", index=False)

    contrasts = [("matsense", "uniform"), ("matsense", "shuffled"), ("matsense", "noise"),
                 ("uniform", "noise"), ("shuffled", "noise"), ("shuffled_class", "noise"),
                 ("matsense", "shuffled_class")]
    tests = []
    for (sc, cond), g in pairs.groupby(["scenario_id", "condition"]):
        wide = g.pivot_table(index="block", columns="arm", values="frechet")
        for a, b in contrasts:
            if a not in wide or b not in wide:
                continue
            w = wide[[a, b]].dropna()
            n = len(w)
            p = np.nan
            if n >= 2 and np.any(w[a] - w[b] != 0):
                p = float(wilcoxon(w[a], w[b], zero_method="wilcox").pvalue)
            tests.append({"scenario_id": sc, "condition": cond, "a": a, "b": b, "n_pairs": n,
                          "median_a": float(w[a].median()) if n else np.nan,
                          "median_b": float(w[b].median()) if n else np.nan,
                          "median_diff": float((w[a] - w[b]).median()) if n else np.nan,
                          "p": p, "min_attainable_p": 2.0 / 2 ** n if n else np.nan})
    tests = pd.DataFrame(tests)
    if not tests.empty:
        tests["p_holm"] = holm(tests["p"].to_numpy())
        tests["significant"] = tests["p_holm"] < float(cfg["analysis"]["alpha"])
    tests.to_csv(an / "tests.csv", index=False)

    # descriptive tables
    desc_cols = ["completion", "cross_track_mean", "steer_jitter", "peak_decel", "min_ttc",
                 "n_collisions", "drop_frac_mean", "unmatched_frac", "unmapped_frac",
                 "level_ratio_mean", "hook_ms_p95"]
    if "safety_margin" in runs:
        desc_cols.append("safety_margin")
    runs["arm_label"] = np.where((runs.arm == "unmodified") & (runs.repeat == 1),
                                 "unmodified(repeat)", runs.arm)
    desc = runs.groupby(["scenario_id", "condition", "arm_label"])[desc_cols].median().round(3)
    term = runs.groupby(["scenario_id", "condition", "arm_label"])["termination"] \
               .value_counts().unstack(fill_value=0)
    fre = pairs.groupby(["scenario_id", "condition", "arm"])["frechet"] \
               .describe()[["count", "25%", "50%", "75%"]].round(3)

    with open(an / "REPORT.md", "w", encoding="utf-8") as f:
        f.write(f"# Campaign {cfg['name']}\n\n")
        f.write(f"Completed runs: {len(runs)} of {len(rows)}. Agent: {cfg['pcla']['agent']}. "
                f"Profile: {runs['profile_sha256'].iloc[0][:12]}. "
                f"Code: {runs['code_sha256'].iloc[0][:12]}.\n\n")
        f.write("## Frechet distance to the block's unmodified run (m)\n\n")
        f.write("`noise` is the second unmodified run of the block: the reference every "
                "effect has to exceed.\n\n")
        f.write(fre.to_markdown() + "\n\n")
        f.write("## Paired tests (Wilcoxon, Holm over all rows)\n\n")
        f.write("`min_attainable_p` is the smallest p a Wilcoxon test can give with that many "
                "pairs: if it is above 0.05 the cell cannot reach significance by design.\n\n")
        f.write((tests.round(4).to_markdown(index=False) if not tests.empty else "none") + "\n\n")
        f.write("## Termination counts\n\n" + term.to_markdown() + "\n\n")
        f.write("## Medians of the run metrics\n\n" + desc.to_markdown() + "\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        order = ["noise", "uniform", "matsense", "shuffled", "shuffled_class"]
        for sc, g in pairs.groupby("scenario_id"):
            conds = sorted(g.condition.unique())
            fig, axes = plt.subplots(1, len(conds), figsize=(5 * len(conds), 4), squeeze=False)
            for ax, cond in zip(axes[0], conds):
                gg = g[g.condition == cond]
                arms = [a for a in order if a in set(gg.arm)]
                ax.boxplot([gg[gg.arm == a]["frechet"].dropna() for a in arms], labels=arms)
                noise = gg[gg.arm == "noise"]["frechet"].median()
                ax.axhline(noise, ls="--", lw=1, color="grey")
                ax.set_title(f"scenario {sc}, {cond}")
                ax.set_ylabel("Frechet distance to unmodified run (m)")
            fig.tight_layout()
            fig.savefig(an / f"fig_{sc}.png", dpi=150)
            plt.close(fig)
    except Exception as e:  # figures are optional
        print("figures skipped:", e)

    print(f"analysis written to {an}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
