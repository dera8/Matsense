#!/usr/bin/env python3
"""Per-class, per-range measurements of a paired dataset, and the two checks.

Tables (CSV and Markdown) in <dataset>/results/:
  by_class_range.csv   mode x class x range bin: points, share of the cloud,
                       median, IQR, MAD, p10, p90 of intensity
  contrast.csv         mode x class: median intensity in the band divided by the
                       reference class's (all frames pooled, and the per-frame
                       median with its 10-90% spread across frames)
  level.csv            mode: mean intensity and its ratio to RAW, per frame summarised
  coverage.csv         shares of supported, unmatched, unmapped returns
  checks.md            A (operator against loaded coefficients) and B (held-out real)

Frames of one sequence are not independent observations: the per-frame spread
is descriptive, and the sequence is the unit of replication.

Check B needs real per-class measurements excluded from the profile's
estimation (--real-reference JSON: {"source": ..., "contrast": {class: value}}).
Without it, B is reported as not executable rather than replaced by A.

    python measure.py ../data/paired_town03_001 --config ../config/phase1.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from modes import load_profile  # noqa: E402


def iqr(v):
    return float(np.subtract(*np.percentile(v, [75, 25]))) if v.size else np.nan


def mad(v):
    return float(np.median(np.abs(v - np.median(v)))) if v.size else np.nan


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--config", required=True)
    ap.add_argument("--real-reference", default=None)
    args = ap.parse_args()
    import yaml
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))["measure"]
    root = Path(args.dataset)
    man = json.loads((root / "manifest.json").read_text())
    inv = json.loads((root / "invariants.json").read_text()) if (root / "invariants.json").exists() else None
    if inv is None or inv.get("verdict") != "PASS":
        raise SystemExit("run check_invariants.py first: measurements are only made on a dataset that passes")
    profile = load_profile(man["profile"]["path"])
    classes = profile.classes()
    names = {i: c for i, c in enumerate(classes)} | {-1: "unmatched", -2: "unmapped"}
    modes = man["settings"]["modes"]
    edges = list(cfg["range_bins_m"])
    band = cfg["band_m"]
    ref = cfg["reference_class"]
    out = root / "results"
    out.mkdir(exist_ok=True)

    rows, per_frame_contrast, level_rows, cov_rows = [], [], [], []
    pooled = {m: {c: [] for c in names.values()} for m in modes}
    for fr in man["frames"]:
        name = fr["frame"]
        lab = np.load(root / "labels" / f"{name}.npz")
        cls = lab["cls"]
        d0 = np.load(root / "modes" / "RAW" / f"{name}.npz")
        r = np.linalg.norm(d0["xyz"].astype(np.float64), axis=1)
        inband = (r >= band[0]) & (r < band[1])
        cov_rows.append({"frame": name, "n": int(cls.size), "supported": float(np.mean(cls >= 0)),
                         "unmatched": float(np.mean(cls == -1)), "unmapped": float(np.mean(cls == -2))})
        raw_mean = float(d0["intensity"].mean())
        for m in modes:
            I = np.load(root / "modes" / m / f"{name}.npz")["intensity"].astype(np.float64)
            level_rows.append({"frame": name, "mode": m, "mean": float(I.mean()),
                               "ratio_to_raw": float(I.mean() / raw_mean) if raw_mean else np.nan})
            med = {}
            for k, c in names.items():
                sel = (cls == k) & inband
                pooled[m][c].append(I[sel])
                med[c] = float(np.median(I[sel])) if sel.sum() >= 20 else np.nan
            if not np.isnan(med.get(ref, np.nan)) and med[ref] > 0:
                for c in classes:
                    per_frame_contrast.append({"frame": name, "mode": m, "class": c, "contrast": med[c] / med[ref]})

    # by class and range bin, pooled over frames
    for m in modes:
        allI, allR, allC = [], [], []
        for fr in man["frames"]:
            name = fr["frame"]
            cls = np.load(root / "labels" / f"{name}.npz")["cls"]
            d = np.load(root / "modes" / m / f"{name}.npz")
            allI.append(d["intensity"].astype(np.float64)); allC.append(cls)
            allR.append(np.linalg.norm(d["xyz"].astype(np.float64), axis=1))
        I, R, C = np.concatenate(allI), np.concatenate(allR), np.concatenate(allC)
        N = I.size
        for k, c in names.items():
            for lo, hi in zip(edges[:-1], edges[1:]):
                sel = (C == k) & (R >= lo) & (R < hi)
                v = I[sel]
                rows.append({"mode": m, "class": c, "range_m": f"{lo}-{hi}", "points": int(sel.sum()),
                             "share": float(sel.sum() / N), "median": float(np.median(v)) if v.size else np.nan,
                             "iqr": iqr(v), "mad": mad(v),
                             "p10": float(np.percentile(v, 10)) if v.size else np.nan,
                             "p90": float(np.percentile(v, 90)) if v.size else np.nan})
    bcr = pd.DataFrame(rows)
    bcr.to_csv(out / "by_class_range.csv", index=False)

    # contrast in the band: pooled and per-frame
    crow = []
    pf = pd.DataFrame(per_frame_contrast)
    for m in modes:
        refv = np.concatenate(pooled[m][ref]) if pooled[m][ref] else np.array([])
        refmed = float(np.median(refv)) if refv.size else np.nan
        for c in classes:
            v = np.concatenate(pooled[m][c]) if pooled[m][c] else np.array([])
            sub = pf[(pf["mode"] == m) & (pf["class"] == c)]["contrast"] if len(pf) else pd.Series(dtype=float)
            crow.append({"mode": m, "class": c, "points_in_band": int(v.size),
                         "contrast_pooled": float(np.median(v) / refmed) if v.size and refmed else np.nan,
                         "contrast_frame_median": float(sub.median()) if len(sub) else np.nan,
                         "contrast_frame_p10": float(sub.quantile(0.1)) if len(sub) else np.nan,
                         "contrast_frame_p90": float(sub.quantile(0.9)) if len(sub) else np.nan,
                         "frames": int(len(sub))})
    con = pd.DataFrame(crow)
    con.to_csv(out / "contrast.csv", index=False)
    lev = pd.DataFrame(level_rows).groupby("mode").agg(mean=("mean", "median"),
                                                       ratio_to_raw=("ratio_to_raw", "median"),
                                                       ratio_p10=("ratio_to_raw", lambda s: s.quantile(.1)),
                                                       ratio_p90=("ratio_to_raw", lambda s: s.quantile(.9)))
    lev.reindex(modes).to_csv(out / "level.csv")
    cov = pd.DataFrame(cov_rows)
    cov.to_csv(out / "coverage.csv", index=False)

    # checks A and B
    beta_f = profile.factor_table(man["settings"]["condition"])
    expected = {c: beta_f[c] / beta_f[ref] for c in classes}
    ms = con[con["mode"] == "MATSENSE"].set_index("class")["contrast_pooled"]
    raw = con[con["mode"] == "RAW"].set_index("class")["contrast_pooled"]
    a_lines = ["| class | profile contrast f_m/f_ref | RAW contrast | MATSENSE contrast | MATSENSE / RAW / expected |",
               "|---|---:|---:|---:|---:|"]
    for c in classes:
        ratio = ms[c] / raw[c] / expected[c] if raw.get(c) and not np.isnan(raw.get(c, np.nan)) else np.nan
        a_lines.append(f"| {c} | {expected[c]:.4f} | {raw.get(c, np.nan):.4f} | {ms.get(c, np.nan):.4f} | {ratio:.4f} |")
    if args.real_reference:
        rr = json.loads(Path(args.real_reference).read_text())
        b_lines = [f"Reference: {rr.get('source', '?')}", "",
                   "| mode | MAE of contrast against the real reference |", "|---|---:|"]
        for m in modes:
            cm = con[con["mode"] == m].set_index("class")["contrast_pooled"]
            errs = [abs(cm[c] - rr["contrast"][c]) for c in rr["contrast"] if c in cm and not np.isnan(cm[c])]
            b_lines.append(f"| {m} | {np.mean(errs):.4f} (n={len(errs)}) |")
        b_text = "\n".join(b_lines)
    else:
        b_text = ("**Not executable with the data available here.** It needs per-class medians from real "
                  "recordings excluded from the profile's estimation, on the same street as the simulated "
                  "sequence (the paper's E2 reference). Check A below is not a substitute: MATSENSE reproduces "
                  "the loaded coefficients by construction.")
    lv = lev.reindex(modes)
    md = ["# Phase 1 checks", "",
          f"Dataset: `{root}`  profile `{man['profile']['name']}` ({man['profile']['sha256'][:12]})  "
          f"condition `{man['settings']['condition']}`  frames {len(man['frames'])}", "",
          "## A. Operator against the loaded coefficients (correctness, not realism)", "",
          f"Contrast = median intensity in {band[0]}-{band[1]} m relative to `{ref}`. MATSENSE multiplies RAW by the "
          "profile factors exactly, so its class contrast should be RAW's contrast times the profile's: "
          "the last column should be 1 wherever both classes have points.", "", *a_lines, "",
          "## B. Agreement with held-out real measurements", "", b_text, "",
          "## Global level", "", "| mode | mean intensity | ratio to RAW (median, p10-p90 over frames) |",
          "|---|---:|---:|",
          *[f"| {m} | {lv.loc[m, 'mean']:.4f} | {lv.loc[m, 'ratio_to_raw']:.4f} "
            f"({lv.loc[m, 'ratio_p10']:.4f}-{lv.loc[m, 'ratio_p90']:.4f}) |" for m in modes], "",
          "## Coverage", "",
          f"Supported {100 * cov.supported.median():.1f}%, unmatched {100 * cov.unmatched.median():.1f}%, "
          f"unmapped {100 * cov.unmapped.median():.1f}% of returns (median over frames). Unmatched and "
          "unmapped returns keep RAW intensity in every material-dependent mode; UNIFORM scales every return.", ""]
    (out / "checks.md").write_text("\n".join(md))

    # figure: band contrast per class and mode
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.5, 3.2))
        w = 0.8 / len(modes)
        for j, m in enumerate(modes):
            sub = con[con["mode"] == m].set_index("class").reindex(classes)
            x = np.arange(len(classes)) + j * w
            ax.bar(x, sub["contrast_pooled"], w, label=m)
            ax.errorbar(x, sub["contrast_frame_median"],
                        yerr=[sub["contrast_frame_median"] - sub["contrast_frame_p10"],
                              sub["contrast_frame_p90"] - sub["contrast_frame_median"]],
                        fmt="none", ecolor="k", lw=0.8)
        ax.set_xticks(np.arange(len(classes)) + w * (len(modes) - 1) / 2)
        ax.set_xticklabels(classes)
        ax.set_ylabel(f"median intensity / {ref}\n({band[0]}-{band[1]} m)")
        ax.legend(fontsize=7, ncol=len(modes), frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0))
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(out / "contrast.png", dpi=160)
    except Exception as exc:  # noqa: BLE001
        print(f"  figure not drawn: {exc}")

    print((out / "checks.md").read_text())
    print(f"\n  tables and figure -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
