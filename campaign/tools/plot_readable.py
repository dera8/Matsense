#!/usr/bin/env python3
"""Figures that can be read without the paper.

    python tools/plot_readable.py              (Italian labels)
    python tools/plot_readable.py --lang en

Writes to <out_dir>/analysis/readable/:
  path_<scenario>.png        where the car drives: sideways offset along the route
  change_<scenario>.png      how much the driving changes with each LiDAR version
  matsense_vs_controls.png   is MatSense closer to normal driving than its controls?
Run analyze.py first (it writes pairs.csv).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from matsense_campaign.config import load_campaign, resolve_path  # noqa: E402

# Reference palette (dataviz skill): first three categorical slots, validated
# all-pairs; the reference arm is neutral gray, not a categorical hue.
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COLOR = {"unmodified": "#898781", "noise": "#898781",
         "uniform": "#2a78d6", "matsense": "#eb6834", "shuffled": "#1baf7a"}
ORDER = ["uniform", "matsense", "shuffled"]

TXT = {
    "it": {
        "arm": {"unmodified": "LiDAR originale",
                "noise": "LiDAR originale, ripetuto",
                "uniform": "Uniforme",
                "matsense": "MatSense",
                "shuffled": "Materiali mescolati"},
        "arm_note": {"noise": "quanto cambia da sola, ripetendo la stessa prova",
                     "uniform": "più scuro, uguale per tutti i materiali",
                     "matsense": "più scuro secondo il materiale (misurato)",
                     "shuffled": "stessi valori di MatSense, assegnati a caso"},
        "scen": {"05": "Auto che passa col rosso", "09": "Auto che si immette da una fila ferma",
                 "15": "Auto contromano in curva", "16": "Frenata dell'auto davanti",
                 "18": "Ciclista dietro un'auto parcheggiata"},
        "cond": {"nominal": "asciutto", "rain": "pioggia"},
        "path_title": "Dove guida l'auto",
        "path_sub": "Spostamento laterale dalla rotta prevista lungo il percorso. Linee sottili: singole prove; spessa: mediana.",
        "path_x": "metri percorsi lungo la rotta",
        "path_y": "spostamento laterale (m)\n← sinistra    destra →",
        "hazard": "pericolo più vicino",
        "change_title": "Quanto cambia la guida",
        "change_sub": "Distanza tra il percorso di ogni prova e quello con il LiDAR originale nella stessa ripetizione "
                      "(0 = identico). La fascia grigia è la variabilità naturale.",
        "change_x": "differenza dal percorso di riferimento (m)",
        "natural": "variabilità naturale",
        "beyond": "riferimento ripetuto fuori scala: {} prova a {} m →",
        "off": "{} prova fuori scala ({} m) →",
        "cmp_title": "MatSense guida più vicino al normale dei suoi controlli?",
        "cmp_sub": "Per ogni ripetizione: distanza di MatSense dal riferimento meno quella del controllo. "
                   "A sinistra dello 0 MatSense è più vicino alla guida normale.",
        "cmp_x": "← MatSense più vicino al normale        MatSense più lontano →   (m)",
        "vs": "MatSense − {}",
        "count": "{} su {} a sinistra",
        "n": "{} ripetizioni",
    },
    "en": {
        "arm": {"unmodified": "Original LiDAR", "noise": "Original LiDAR, repeated",
                "uniform": "Uniform", "matsense": "MatSense", "shuffled": "Shuffled materials"},
        "arm_note": {"noise": "how much it changes on its own when the same test is repeated",
                     "uniform": "darker, same for every material",
                     "matsense": "darker by material (measured)",
                     "shuffled": "MatSense's values, assigned at random"},
        "scen": {"05": "Car running a red light", "09": "Car cutting in from a parked row",
                 "15": "Oncoming car in a bend", "16": "Lead car braking",
                 "18": "Cyclist behind a parked car"},
        "cond": {"nominal": "dry", "rain": "rain"},
        "path_title": "Where the car drives",
        "path_sub": "Sideways offset from the planned route along the way. Thin lines: single runs; thick: median.",
        "path_x": "metres driven along the route",
        "path_y": "sideways offset (m)\n← left    right →",
        "hazard": "hazard closest",
        "change_title": "How much the driving changes",
        "change_sub": "Distance between each run's path and the original-LiDAR path of the same repetition "
                      "(0 = identical). The gray band is the natural variability.",
        "change_x": "difference from the reference path (m)",
        "natural": "natural variability",
        "beyond": "repeated reference off scale: {} run at {} m →",
        "off": "{} run off scale ({} m) →",
        "cmp_title": "Does MatSense drive closer to normal than its controls?",
        "cmp_sub": "Per repetition: MatSense's distance from the reference minus the control's. "
                   "Left of 0, MatSense is closer to normal driving.",
        "cmp_x": "← MatSense closer to normal        MatSense further away →   (m)",
        "vs": "MatSense − {}",
        "count": "{} of {} on the left",
        "n": "{} repetitions",
    },
}


DPI = 160


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=9, length=3)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def header(fig, title, sub):
    fig.text(0.01, 0.985, title, ha="left", va="top", fontsize=14, color=INK, weight="bold")
    fig.text(0.01, 0.935, sub, ha="left", va="top", fontsize=9.5, color=INK2, wrap=True)


def signed_offset(route_xy: np.ndarray, x: np.ndarray, y: np.ndarray):
    """Progress along the route and signed lateral offset (+ = right in CARLA)."""
    seg = np.diff(route_xy, axis=0)
    len2 = np.maximum((seg ** 2).sum(1), 1e-9)
    cum = np.concatenate([[0], np.cumsum(np.sqrt(len2))])
    prog, off = np.empty(len(x)), np.empty(len(x))
    for k, (qx, qy) in enumerate(zip(x, y)):
        d = np.array([qx, qy]) - route_xy[:-1]
        t = np.clip((d * seg).sum(1) / len2, 0, 1)
        foot = route_xy[:-1] + t[:, None] * seg
        dist = np.hypot(qx - foot[:, 0], qy - foot[:, 1])
        j = int(np.argmin(dist))
        prog[k] = cum[j] + t[j] * np.sqrt(len2[j])
        cross = seg[j, 0] * (qy - foot[j, 1]) - seg[j, 1] * (qx - foot[j, 0])
        off[k] = np.sign(cross) * dist[j]
    return prog, off


def plot_paths(runs, out_runs, scen, T, dest):
    for sid, g in runs.groupby("scenario_id"):
        conds = [c for c in ("nominal", "rain") if c in set(g.condition)]
        fig, axes = plt.subplots(len(conds), 1, figsize=(10, 3.4 * len(conds) + 1.1), squeeze=False,
                                 facecolor=SURFACE)
        route = np.asarray(scen[sid]["ego_path"])[:, :2]
        grid = np.arange(0, np.hypot(*np.diff(route, axis=0).T).sum(), 1.0)
        lo, hi = -0.5, 0.5
        for ax, cond in zip(axes[:, 0], conds):
            style(ax)
            hz, ends = [], []
            for arm in ["unmodified"] + ORDER:
                curves = []
                for _, r in g[(g.condition == cond) & (g.arm == arm)].iterrows():
                    tk = pd.read_csv(out_runs / r.run_id / "ticks.csv")
                    p, o = signed_offset(route, tk.x.to_numpy(), tk.y.to_numpy())
                    p = np.maximum.accumulate(p)
                    keep = np.concatenate([[True], np.diff(p) > 1e-3])
                    c = np.interp(grid, p[keep], o[keep], right=np.nan)
                    curves.append(c)
                    ax.plot(grid, c, color=COLOR[arm], lw=0.8, alpha=0.25)
                    if arm == "unmodified" and "hazards" in tk:
                        dmin = []
                        for hjs, ex, ey in zip(tk.hazards, tk.x, tk.y):
                            h = json.loads(hjs) if isinstance(hjs, str) and hjs else {}
                            dmin.append(min((np.hypot(hx - ex, hy - ey) for hx, hy in h.values()), default=np.inf))
                        hz.append(p[int(np.argmin(dmin))])
                if curves:
                    med = np.nanmedian(np.vstack(curves), axis=0)
                    ax.plot(grid, med, color=COLOR[arm], lw=2.2, solid_capstyle="round")
                    lo = min(lo, np.nanmin(np.vstack(curves)))
                    hi = max(hi, np.nanmax(np.vstack(curves)))
                    last = np.where(np.isfinite(med))[0][-1]
                    ends.append([med[last], grid[last], arm])
            ax.axhline(0, color=AXIS, lw=1)
            ends.sort(key=lambda e: e[0])
            gap = 0.09 * (hi - lo + 1.0)          # keep end labels apart
            for k in range(1, len(ends)):
                ends[k][0] = max(ends[k][0], ends[k - 1][0] + gap)
            for y_lab, x_end, arm in ends:
                ax.text(x_end + 1.5, y_lab, T["arm"][arm], va="center", fontsize=9, color=INK2)
                ax.plot([x_end + 0.3, x_end + 1.2], [y_lab, y_lab], color=COLOR[arm], lw=2.2)
            if hz:
                hx = float(np.median(hz))
                ax.axvline(hx, color=INK2, lw=0.8)
                ax.text(hx, 1.0, " " + T["hazard"], transform=ax.get_xaxis_transform(), va="top",
                        fontsize=8.5, color=INK2)
            ax.set_title(f"{T['scen'].get(sid, sid)} — {T['cond'][cond]}", loc="left", fontsize=11, color=INK)
            ax.set_ylabel(T["path_y"], fontsize=9, color=INK2)
            ax.set_xlim(0, grid[-1] * 1.18)
        pad = 0.12 * (hi - lo)
        for ax in axes[:, 0]:
            ax.set_ylim(lo - pad, hi + 2 * pad)
        axes[-1, 0].set_xlabel(T["path_x"], fontsize=9.5, color=INK2)
        header(fig, T["path_title"], T["path_sub"])
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        fig.savefig(dest / f"path_{sid}.png", dpi=DPI, facecolor=SURFACE)
        plt.close(fig)


def plot_change(pairs, T, dest):
    for sid, g in pairs.groupby("scenario_id"):
        conds = [c for c in ("nominal", "rain") if c in set(g.condition)]
        fig, axes = plt.subplots(1, len(conds), figsize=(6.2 * len(conds), 4.4), squeeze=False,
                                 sharex=True, facecolor=SURFACE)
        rows = ORDER[::-1]
        # A run that stopped or failed sits tens of metres away and would flatten everything else:
        # values above 10x the typical distance are listed beside their row instead of setting the scale.
        typical = max(g.frechet.median(), g[g.arm == "noise"].frechet.median())
        thr = 10 * typical
        xmax = max(g[g.frechet <= thr].frechet.max(), 2 * typical) * 1.15
        for ax, cond in zip(axes[0], conds):
            style(ax)
            ax.grid(True, axis="x", color=GRID, linewidth=0.6)
            ax.grid(False, axis="y")
            c = g[g.condition == cond]
            nz = c[c.arm == "noise"].frechet
            if len(nz):
                inr, out = nz[nz <= xmax], nz[nz > xmax]
                ax.axvspan(inr.min(), inr.max(), color="#f0efec", zorder=0)
                if len(out):
                    ax.text(xmax, -0.5, T["beyond"].format(len(out), ", ".join(f"{v:.1f}" for v in out)) + " ",
                            ha="right", va="bottom", fontsize=8.5, color=INK2)
                ax.axvline(nz.median(), color=MUTED, lw=1, zorder=1)
                ax.text(nz.median(), len(rows) - 0.45, " " + T["natural"], fontsize=8.5, color=INK2, va="bottom")
            rng = np.random.default_rng(0)
            for i, arm in enumerate(rows):
                v = c[c.arm == arm].frechet.to_numpy()
                if not len(v):
                    continue
                vin, vout = v[v <= xmax], v[v > xmax]
                ax.scatter(vin, i + rng.uniform(-0.12, 0.12, len(vin)), s=40, color=COLOR[arm],
                           edgecolor=SURFACE, linewidth=1.5, zorder=3)
                if len(vout):
                    ax.text(xmax, i - 0.3, T["off"].format(len(vout), ", ".join(f"{x:.0f}" for x in sorted(vout))) + " ",
                            ha="right", va="center", fontsize=8.5, color=COLOR[arm])
                m = float(np.median(v))
                ax.plot([m, m], [i - 0.28, i + 0.28], color=INK, lw=2, zorder=4, solid_capstyle="round")
                ax.text(m, i + 0.32, f"{m:.2f} m", ha="center", va="bottom", fontsize=8.5, color=INK)
            ax.set_yticks(range(len(rows)))
            ax.set_yticklabels([f"{T['arm'][a]}\n{T['arm_note'][a]}" for a in rows], fontsize=8.5, color=INK2)
            ax.set_ylim(-0.6, len(rows) - 0.1)
            ax.set_xlim(0, xmax)
            n = c[c.arm == "matsense"].shape[0]
            ax.set_title(f"{T['cond'][cond]}  ·  {T['n'].format(n)}", loc="left", fontsize=11, color=INK)
            ax.set_xlabel(T["change_x"], fontsize=9.5, color=INK2)
        header(fig, f"{T['change_title']}: {T['scen'].get(sid, sid)}", T["change_sub"])
        fig.tight_layout(rect=(0, 0, 1, 0.86))
        fig.savefig(dest / f"change_{sid}.png", dpi=DPI, facecolor=SURFACE)
        plt.close(fig)


def plot_compare(pairs, T, dest):
    wide = pairs.pivot_table(index=["scenario_id", "condition", "block"], columns="arm", values="frechet")
    cells = [(s, c) for s, c in wide.index.droplevel("block").unique()]
    fig, axes = plt.subplots(len(cells), 1, figsize=(9, 1.9 * len(cells) + 1.2), squeeze=False,
                             sharex=True, facecolor=SURFACE)
    lim = 0.2
    for (s, c) in cells:
        w = wide.loc[(s, c)]
        for ctl in ("uniform", "shuffled"):
            if {"matsense", ctl} <= set(w.columns):
                d = (w["matsense"] - w[ctl]).dropna()
                if len(d):
                    lim = max(lim, d.abs().max())
    for ax, (s, c) in zip(axes[:, 0], cells):
        style(ax)
        ax.grid(False, axis="y")
        w = wide.loc[(s, c)]
        rng = np.random.default_rng(1)
        labels = []
        for i, ctl in enumerate(("shuffled", "uniform")):
            if not {"matsense", ctl} <= set(w.columns):
                labels.append("")
                continue
            d = (w["matsense"] - w[ctl]).dropna().to_numpy()
            ax.scatter(d, i + rng.uniform(-0.1, 0.1, len(d)), s=40, color=COLOR[ctl],
                       edgecolor=SURFACE, linewidth=1.5, zorder=3)
            if len(d):
                m = float(np.median(d))
                ax.plot([m, m], [i - 0.25, i + 0.25], color=INK, lw=2, zorder=4)
                ax.text(lim * 1.12, i, T["count"].format(int((d < 0).sum()), len(d)), va="center",
                        fontsize=8.5, color=INK2)
            labels.append(T["vs"].format(T["arm"][ctl]))
        ax.axvline(0, color=INK2, lw=1)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(labels, fontsize=9, color=INK2)
        ax.set_ylim(-0.6, 1.6)
        ax.set_title(f"{T['scen'].get(s, s)} — {T['cond'][c]}", loc="left", fontsize=10.5, color=INK)
    axes[-1, 0].set_xlim(-lim * 1.1, lim * 1.1)
    axes[-1, 0].set_xlabel(T["cmp_x"], fontsize=9.5, color=INK2)
    header(fig, T["cmp_title"], T["cmp_sub"])
    fig.tight_layout(rect=(0, 0, 0.88, 0.9))
    fig.savefig(dest / "matsense_vs_controls.png", dpi=DPI, facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/campaign.json")
    ap.add_argument("--lang", default="it", choices=sorted(TXT))
    ap.add_argument("--dpi", type=int, default=160)
    a = ap.parse_args()
    cfg = load_campaign(a.config)
    out = Path(cfg["_out"])
    an = out / "analysis"
    if not (an / "pairs.csv").exists():
        print("run analyze.py first")
        return 1
    dest = an / ("readable" if a.lang == "it" else f"readable_{a.lang}")
    dest.mkdir(exist_ok=True)
    global DPI
    DPI = a.dpi
    T = TXT[a.lang]
    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans", "sans-serif"]
    scen = json.loads(resolve_path(cfg["scenarios_resolved"]).read_text(encoding="utf-8"))
    pairs = pd.read_csv(an / "pairs.csv", dtype={"scenario_id": str})
    runs = pd.read_csv(an / "runs.csv", dtype={"scenario_id": str})
    runs = runs[runs.termination.notna() & (runs.termination != "error")]
    plot_paths(runs, out / "runs", scen, T, dest)
    plot_change(pairs, T, dest)
    plot_compare(pairs, T, dest)
    print(f"figures written to {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
