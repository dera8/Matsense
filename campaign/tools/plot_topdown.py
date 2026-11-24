#!/usr/bin/env python3
"""Top-down map of where the car drove, with a true-scale zoom on the offset.

    python tools/plot_topdown.py [--config CFG] [--lang it|en]
           [--xodr-dir C:/Users/debor/Desktop/carla/carla/Unreal/CarlaUE4/Content/Carla/Maps/OpenDrive]

Reads the road network from the town's OpenDRIVE file (no simulator needed) and
writes <out_dir>/analysis/readable/map_<scenario>.png: for each condition, the
whole route on the map (left) and a true-scale zoom around the hazard (right),
with every run as a thin line and the median path of each LiDAR version thick.
Run analyze.py first (it writes runs.csv).
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
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from matsense_campaign.config import load_campaign, resolve_path  # noqa: E402
from plot_readable import COLOR, INK, INK2, MUTED, ORDER, SURFACE, TXT, signed_offset  # noqa: E402

ROAD, MARK = "#dcdbd4", "#ffffff"
CONE = "#eda100"   # status-free accent for construction cones (not one of the four arm colours)
XODR_DIR = Path(r"C:\Users\debor\Desktop\carla\carla\Unreal\CarlaUE4\Content\Carla\Maps\OpenDrive")
EXTRA = {
    "it": {"title": "Dove guida l'auto, vista dall'alto",
           "sub": "A sinistra il percorso intero; a destra lo zoom a scala reale sulla zona del pericolo "
                  "(riquadro). Linee sottili: singole prove; spessa: percorso mediano.",
           "start": "partenza", "hazard_path": "percorso del pericolo", "cones": "coni (ostacolo)", "parked": "auto ferme",
           "zoom": "zoom a scala reale", "offset": "{:.1f} m"},
    "en": {"title": "Where the car drives, seen from above",
           "sub": "Left: the whole route; right: a true-scale zoom on the hazard area (box). "
                  "Thin lines: single runs; thick: median path.",
           "start": "start", "hazard_path": "hazard path", "cones": "cones (obstacle)", "parked": "parked cars",
           "zoom": "true-scale zoom", "offset": "{:.1f} m"},
}


DPI = 160


def lane_geometry(xodr: Path, box):
    """Road quads and lane-edge segments inside box = (xmin, xmax, ymin, ymax)."""
    import carla
    m = carla.Map(xodr.stem, xodr.read_text(encoding="utf-8"))
    quads, edges = [], []
    x0, x1, y0, y1 = box
    for w in m.generate_waypoints(1.0):
        loc = w.transform.location
        if not (x0 - 5 < loc.x < x1 + 5 and y0 - 5 < loc.y < y1 + 5):
            continue
        for n in w.next(1.0):
            pts = []
            for a in (w, n):
                r = a.transform.get_right_vector()
                c, h = a.transform.location, a.lane_width / 2.0
                pts.append(((c.x - r.x * h, c.y - r.y * h), (c.x + r.x * h, c.y + r.y * h)))
            (al, ar), (bl, br) = pts
            quads.append([al, ar, br, bl])
            if not w.is_junction:
                edges += [[al, bl], [ar, br]]
    return quads, edges


def draw_map(ax, quads, edges):
    ax.add_collection(PolyCollection(quads, facecolors=ROAD, edgecolors=ROAD, linewidths=0.6, zorder=0))
    ax.add_collection(LineCollection(edges, colors=MARK, linewidths=1.4, zorder=1))


def median_path(route, curves, grid):
    """Back to x, y: route point + median offset along the route's right normal."""
    med = np.nanmedian(np.vstack(curves), axis=0)
    seg = np.diff(route, axis=0)
    cum = np.concatenate([[0], np.cumsum(np.hypot(*seg.T))])
    px, py = np.interp(grid, cum, route[:, 0]), np.interp(grid, cum, route[:, 1])
    tx, ty = np.gradient(px), np.gradient(py)
    n = np.hypot(tx, ty) + 1e-9
    # right normal in CARLA's left-handed frame (x forward, y right): (-ty, tx)
    rx, ry = -ty / n, tx / n
    return px + med * rx, py + med * ry, med


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/campaign.json")
    ap.add_argument("--lang", default="it", choices=sorted(TXT))
    ap.add_argument("--dpi", type=int, default=160)
    ap.add_argument("--xodr-dir", default=str(XODR_DIR))
    ap.add_argument("--zoom-m", type=float, default=24.0, help="length of the zoom window along the route")
    ap.add_argument("--zoom-at", action="append", default=[], metavar="SCENARIO:METRES",
                    help="centre the zoom at this progress along the route, e.g. 15:70 (default: where the hazard is closest)")
    a = ap.parse_args()
    zoom_at = {k: float(v) for k, v in (z.split(":") for z in a.zoom_at)}
    cfg = load_campaign(a.config)
    out = Path(cfg["_out"])
    an = out / "analysis"
    dest = an / ("readable" if a.lang == "it" else f"readable_{a.lang}")
    dest.mkdir(parents=True, exist_ok=True)
    global DPI
    DPI = a.dpi
    T, E = TXT[a.lang], EXTRA[a.lang]
    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans", "sans-serif"]
    scen = json.loads(resolve_path(cfg["scenarios_resolved"]).read_text(encoding="utf-8"))
    runs = pd.read_csv(an / "runs.csv", dtype={"scenario_id": str})
    runs = runs[runs.termination.notna() & (runs.termination != "error")]

    for sid, g in runs.groupby("scenario_id"):
        inst = scen[sid]
        route = np.asarray(inst["ego_path"])[:, :2]
        hz_paths = [np.asarray(h["path"])[:, :2] for h in inst["actors"]]
        roles = [h["role"] for h in inst["actors"]]
        allxy = np.vstack([route] + hz_paths)
        box = (allxy[:, 0].min() - 15, allxy[:, 0].max() + 15, allxy[:, 1].min() - 15, allxy[:, 1].max() + 15)
        quads, edges = lane_geometry(Path(a.xodr_dir) / f"{inst['town']}.xodr", box)
        grid = np.arange(0, np.hypot(*np.diff(route, axis=0).T).sum(), 1.0)
        conds = [c for c in ("nominal", "rain") if c in set(g.condition)]
        wide = (box[1] - box[0]) > 2 * (box[3] - box[2])
        if wide:   # long horizontal route: overview on top at full width, zoom below
            fig, ax_ = plt.subplots(2 * len(conds), 1, figsize=(13, 7.2 * len(conds) + 0.8), facecolor=SURFACE,
                                    gridspec_kw={"height_ratios": [1, 2.2] * len(conds)})
            pairs_ax = [(ax_[2 * i], ax_[2 * i + 1]) for i in range(len(conds))]
        else:
            fig, ax_ = plt.subplots(len(conds), 2, figsize=(13, 6.2 * len(conds) + 0.8), squeeze=False,
                                    gridspec_kw={"width_ratios": [1, 1.25]}, facecolor=SURFACE)
            pairs_ax = [tuple(r) for r in ax_]
        for row, cond in enumerate(conds):
            ax0, ax1 = pairs_ax[row]
            paths, hz_at = {}, []
            for arm in ["unmodified"] + ORDER:
                curves = []
                for _, r in g[(g.condition == cond) & (g.arm == arm)].iterrows():
                    tk = pd.read_csv(out / "runs" / r.run_id / "ticks.csv")
                    x, y = tk.x.to_numpy(), tk.y.to_numpy()
                    p, o = signed_offset(route, x, y)
                    p = np.maximum.accumulate(p)
                    keep = np.concatenate([[True], np.diff(p) > 1e-3])
                    curves.append(np.interp(grid, p[keep], o[keep], right=np.nan))
                    for ax in (ax0, ax1):
                        ax.plot(x, y, color=COLOR[arm], lw=0.7, alpha=0.3, zorder=3)
                    if arm == "unmodified":
                        d = [min((np.hypot(hx - ex, hy - ey) for hx, hy in json.loads(h).values()), default=np.inf)
                             for h, ex, ey in zip(tk.hazards, x, y)]
                        hz_at.append(p[int(np.argmin(d))])
                if curves:
                    paths[arm] = median_path(route, curves, grid)
            # zoom window around the hazard, along the route
            if sid in zoom_at:
                c_prog = zoom_at[sid]
            else:
                c_prog = float(np.median(hz_at)) if hz_at else grid[len(grid) // 2]
            sel = (grid > c_prog - a.zoom_m / 2) & (grid < c_prog + a.zoom_m / 2)
            for ax in (ax0, ax1):
                draw_map(ax, quads, edges)
                for role, hp in zip(roles, hz_paths):
                    if "cone" in role:
                        ax.plot(hp[0, 0], hp[0, 1], "^", color=CONE, ms=5, zorder=3)
                    elif len(hp) > 1:
                        ax.plot(hp[:, 0], hp[:, 1], color=INK, lw=1, zorder=2)
                        ax.annotate("", hp[-1], hp[-2], arrowprops=dict(arrowstyle="-|>", color=INK, lw=1))
                    else:
                        ax.add_patch(Rectangle((hp[0, 0] - 2.3, hp[0, 1] - 1.0), 4.6, 2.0, color=INK2, zorder=2))
                for arm, (px, py, _) in paths.items():
                    ax.plot(px, py, color=COLOR[arm], lw=2.6, zorder=4, solid_capstyle="round")
                ax.set_aspect("equal")
                ax.invert_yaxis()   # CARLA top view (x right, y down)
                ax.set_xticks([])
                ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_color("#c3c2b7")
                ax.set_facecolor(SURFACE)
            ax0.plot(*route[0], "o", color=INK, ms=7, zorder=5)
            ax0.annotate(E["start"], route[0], xytext=(8, 0), textcoords="offset points", fontsize=9, color=INK2)
            ax0.set_xlim(box[0], box[1])
            ax0.set_ylim(box[3], box[2])
            # zoom extent: the selected stretch of every median path, padded to a square
            zx = np.concatenate([px[sel] for px, _, _ in paths.values()] + [route[:, 0][:0]])
            zy = np.concatenate([py[sel] for _, py, _ in paths.values()])
            cx, cy = np.nanmean(zx), np.nanmean(zy)
            half = max(np.nanmax(zx) - np.nanmin(zx), np.nanmax(zy) - np.nanmin(zy)) / 2 + 4
            hw, hh = (half, half / 2.2) if wide else (half, half)   # zoom frame follows the panel shape
            ax1.set_xlim(cx - hw, cx + hw)
            ax1.set_ylim(cy + hh, cy - hh)
            ax0.add_patch(Rectangle((cx - hw, cy - hh), 2 * hw, 2 * hh, fill=False, ec=INK, lw=1, zorder=6))
            # offset annotation: largest median gap from the reference inside the zoom
            if "unmodified" in paths:
                rx, ry, rmed = paths["unmodified"]
                best = None
                for arm in ORDER:
                    if arm in paths:
                        _, _, med = paths[arm]
                        gap = np.abs(med - rmed)
                        gap[~sel] = np.nan
                        if np.isfinite(gap).any():
                            k = int(np.nanargmax(gap))
                            if best is None or gap[k] > best[0]:
                                best = (gap[k], k, arm)
                if best:
                    gap, k, arm = best
                    ax_, ay_ = paths[arm][0][k], paths[arm][1][k]
                    ax1.annotate("", (ax_, ay_), (rx[k], ry[k]),
                                 arrowprops=dict(arrowstyle="<->", color=INK, lw=1.2), zorder=7)
                    ax1.annotate(E["offset"].format(gap), ((ax_ + rx[k]) / 2, (ay_ + ry[k]) / 2),
                                 xytext=(8, 8), textcoords="offset points", fontsize=11, color=INK, weight="bold",
                                 zorder=7)
            ax0.set_title(f"{T['scen'].get(sid, sid)} — {T['cond'][cond]}", loc="left", fontsize=11, color=INK)
            ax1.set_title(E["zoom"], loc="left", fontsize=10, color=INK2)
        handles = [plt.Line2D([], [], color=COLOR[k], lw=2.6) for k in ["unmodified"] + ORDER]
        labels = [T["arm"]["unmodified"]] + [f"{T['arm'][k]}: {T['arm_note'][k]}" for k in ORDER]
        handles.append(plt.Line2D([], [], color=INK, lw=1))
        labels.append(E["hazard_path"])
        if any("cone" in r for r in roles):
            handles.append(plt.Line2D([], [], color=CONE, marker="^", ls="none", ms=6))
            labels.append(E["cones"])
        fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.935), ncol=3, frameon=False,
                   fontsize=9, labelcolor=INK2)
        fig.text(0.01, 0.99, E["title"], ha="left", va="top", fontsize=14, color=INK, weight="bold")
        sub = E["sub"]
        if wide:
            sub = sub.replace("A sinistra", "Sopra").replace("a destra", "sotto") \
                     .replace("Left:", "Top:").replace("right:", "below:")
        fig.text(0.01, 0.96, sub, ha="left", va="top", fontsize=9.5, color=INK2)
        fig.tight_layout(rect=(0, 0, 1, 0.88))
        fig.savefig(dest / f"map_{sid}.png", dpi=DPI, facecolor=SURFACE)
        plt.close(fig)
        print("written", dest / f"map_{sid}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
