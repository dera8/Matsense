#!/usr/bin/env python3
"""Check that the pilot runs really did what they were meant to do.

    python tools/check_pilot.py                 every run with an output
    python tools/check_pilot.py --runs 5        the first five runs of the manifest

For each run it reads summary.json (or error.json), ticks.csv and the frozen
scenario, and checks:

  ego      spawned at the start of the route, moved, followed the route
           (lateral distance), progressed along it
  hazard   every moving hazard was triggered, moved, and came near the ego
  hook     PCLA called the MatSense hook, the semantic LiDAR was mirrored,
           the share of returns without a material is low, the hook is fast
  arms     nominal removes no point; rain removes some in the geometry arms;
           matsense, uniform and shuffled deliver the same level in each block

and draws checks/<run_id>.png: route (blue), ego trajectory (black), hazard
paths (red dashed) and positions (red), speed and brake over time.
Prints PASS / WARN / FAIL per check and the time per run, with the estimate
for the whole manifest.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from matsense_campaign.config import load_campaign, read_manifest, resolve_path  # noqa: E402
from matsense_campaign.metrics import Polyline  # noqa: E402

GEOMETRY_ARMS = {"matsense", "uniform", "shuffled", "shuffled_class", "geometry_only"}
LEVEL_ARMS = ("matsense", "uniform", "shuffled")


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def read_ticks(path: Path) -> dict:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    cols = {k: np.array([_f(r[k]) for r in rows]) for k in rows[0]} if rows else {}
    cols["hazards"] = [json.loads(r["hazards"]) if r.get("hazards") else {} for r in rows]
    return cols


class Report:
    def __init__(self):
        self.lines, self.worst = [], "PASS"

    def add(self, level: str, what: str, detail: str = ""):
        order = {"PASS": 0, "WARN": 1, "FAIL": 2}
        if order[level] > order[self.worst]:
            self.worst = level
        self.lines.append(f"  {level:4}  {what}" + (f"  ({detail})" if detail else ""))


def check_run(row: dict, run_dir: Path, inst: dict, cfg: dict, plot_dir: Path) -> tuple[Report, dict]:
    rep, info = Report(), {}
    summ_p, err_p = run_dir / "summary.json", run_dir / "error.json"
    if not summ_p.exists():
        if err_p.exists():
            err = json.loads(err_p.read_text(encoding="utf-8"))
            last = (err.get("traceback") or err.get("error") or "").strip().splitlines()[-1:]
            rep.add("FAIL", "run ended in error", " ".join(last))
        else:
            rep.add("FAIL", "no summary.json or error.json", "see stdout.txt")
        return rep, info
    summ = json.loads(summ_p.read_text(encoding="utf-8"))
    info["termination"] = summ.get("termination")
    try:
        t0 = datetime.strptime(summ["started_at"], "%Y-%m-%d %H:%M:%S")
        t1 = datetime.strptime(summ["finished_at"], "%Y-%m-%d %H:%M:%S")
        info["wall_s"] = (t1 - t0).total_seconds()
    except (KeyError, ValueError):
        pass
    if not (run_dir / "ticks.csv").exists():
        rep.add("FAIL", "no ticks.csv")
        return rep, info
    tk = read_ticks(run_dir / "ticks.csv")
    if len(tk.get("x", [])) == 0:
        rep.add("FAIL", "ticks.csv is empty: the loop never ran")
        return rep, info
    rc = cfg["run"]
    route = Polyline(np.asarray(inst["ego_path"])[:, :2])
    xy = np.column_stack([tk["x"], tk["y"]])

    # --- ego ---------------------------------------------------------------
    d0 = math.hypot(xy[0, 0] - inst["route"][0][0], xy[0, 1] - inst["route"][0][1])
    rep.add("PASS" if d0 < 3.0 else "FAIL", "ego spawned at the route start", f"{d0:.1f} m away")
    travelled = float(np.sum(np.hypot(*np.diff(xy, axis=0).T))) if len(xy) > 1 else 0.0
    rep.add("PASS" if travelled > 10 else "FAIL", "ego moved", f"{travelled:.0f} m driven, termination {summ.get('termination')}")
    xt = np.abs(tk["cross_track_m"])
    xt_med, xt_max = float(np.nanmedian(xt)), float(np.nanmax(xt))
    lvl = "PASS" if xt_med < 0.8 and xt_max < float(rc["deviation_m"]) else ("WARN" if xt_med < 1.5 else "FAIL")
    rep.add(lvl, "ego follows the route", f"lateral distance median {xt_med:.2f} m, max {xt_max:.2f} m")
    prog = tk["progress_m"]
    info["completion"] = float(np.nanmax(prog)) / route.length
    back = float(np.nanmax(np.maximum.accumulate(np.nan_to_num(prog)) - np.nan_to_num(prog)))
    rep.add("PASS" if back < 3.0 else "WARN", "progress along the route only increases", f"max step back {back:.1f} m")
    rep.add("PASS" if summ.get("termination") not in ("never_started", "route_deviation", "error") else "WARN",
            "termination", f"{summ.get('termination')} after {summ.get('duration_s')} s, completion {info['completion']:.0%}")

    # --- hazards -------------------------------------------------------------
    hz = tk["hazards"]
    for a in inst["actors"]:
        role = a["role"]
        pos = np.array([h.get(role, [np.nan, np.nan]) for h in hz], dtype=float)
        if np.all(np.isnan(pos)):
            rep.add("FAIL", f"hazard '{role}' present", "never recorded")
            continue
        start = np.asarray(a["path"][0][:2])
        rep.add("PASS" if math.hypot(*(pos[0] - start)) < 2.0 else "FAIL",
                f"hazard '{role}' spawned where planned", f"{math.hypot(*(pos[0] - start)):.1f} m off")
        dmin = float(np.nanmin(np.hypot(*(pos - xy).T)))
        if float(a.get("speed", 0.0)) > 0 and len(a["path"]) > 1:
            moved = np.hypot(*(pos - pos[0]).T)
            if np.nanmax(moved) < 1.0:
                rep.add("FAIL", f"hazard '{role}' triggered and moved", "never moved: trigger not reached")
            else:
                k = int(np.argmax(moved > 0.2))
                rep.add("PASS", f"hazard '{role}' triggered and moved",
                        f"started at t={tk['t_s'][k]:.1f} s, moved {np.nanmax(moved):.0f} m")
        rep.add("PASS" if dmin < 25 else "WARN", f"hazard '{role}' came near the ego", f"closest {dmin:.1f} m")
    if summ.get("collisions"):
        rep.add("WARN", "collisions", ", ".join(sorted({c['other'] for c in summ['collisions']})))

    # --- hook / sensing --------------------------------------------------------
    n_calls = int(summ.get("n_hook_calls") or 0)
    rep.add("PASS" if n_calls > 0 else "FAIL", "MatSense hook called by PCLA",
            f"{n_calls} calls, modules {summ.get('hook_modules', '?')}")
    rep.add("PASS" if summ.get("semantic_lidar") else "FAIL", "semantic LiDAR mirrored on the agent's LiDAR",
            json.dumps(summ.get("semantic_lidar"))[:120])
    n_in = tk.get("n_in", np.array([]))
    ok = n_in > 0
    if ok.any():
        um = tk["n_unmatched"][ok] / n_in[ok]
        info["unmatched"] = float(np.nanmedian(um))
        rep.add("PASS" if info["unmatched"] < 0.15 else "FAIL", "returns without a material (unmatched)",
                f"median {info['unmatched']:.1%}, max {np.nanmax(um):.1%}")
        hm = tk["hook_ms"][ok]
        rep.add("PASS" if np.nanmedian(hm) < 50 else "FAIL", "hook time", f"median {np.nanmedian(hm):.1f} ms, p95 {np.nanpercentile(hm, 95):.1f} ms")
        lag = tk["sem_lag_frames"][ok]
        frac_lag = float(np.mean(np.nan_to_num(lag, nan=99) != 0))
        rep.add("PASS" if frac_lag < 0.05 else "WARN", "semantic scan from the same frame", f"{frac_lag:.0%} of frames used another frame")
        drop = tk["drop_frac"][ok]
        info["drop"] = float(np.nanmean(drop))
        # mean, not median: with half a LiDAR turn per frame the level alternates between two
        # values on even and odd ticks, and the median of such a series jumps between them
        info["level"] = float(np.nanmean(tk["level_ratio"][ok]))
        arm, cond = row["arm"], row["condition"]
        if cond == "nominal" or arm not in GEOMETRY_ARMS:
            rep.add("PASS" if np.nanmax(drop) == 0 else "FAIL", "no return removed", f"max drop {np.nanmax(drop):.4f}")
        else:
            rep.add("PASS" if info["drop"] > 0 else "FAIL", "rain removes some returns", f"mean drop {info['drop']:.2%}")
    else:
        rep.add("FAIL", "sensing statistics", "n_in is never > 0: the hook never saw a cloud")

    _plot(row, inst, tk, xy, plot_dir / f"{row['run_id']}.png")
    return rep, info


def _plot(row, inst, tk, xy, path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return
    ego = np.asarray(inst["ego_path"])
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 5.5), gridspec_kw={"width_ratios": [1.2, 1]})
    ax.plot(ego[:, 0], ego[:, 1], "b-", lw=6, alpha=0.25, label="planned route")
    ax.plot(xy[:, 0], xy[:, 1], "k-", lw=1.5, label="ego (driven)")
    ax.plot(xy[0, 0], xy[0, 1], "ko")
    for a in inst["actors"]:
        p = np.asarray(a["path"])
        ax.plot(p[:, 0], p[:, 1], "r--" if len(p) > 1 else "rs", lw=1)
        pos = np.array([h.get(a["role"], [np.nan, np.nan]) for h in tk["hazards"]], dtype=float)
        ax.plot(pos[:, 0], pos[:, 1], "r-", lw=1.5)
        ax.annotate(a["role"], p[0, :2], fontsize=7, color="r")
    allxy = np.vstack([ego[:, :2], xy] + [np.asarray(a["path"])[:, :2] for a in inst["actors"]])
    cx, cy = (allxy.min(0) + allxy.max(0)) / 2
    half = max(float(np.ptp(allxy[:, 0])), float(np.ptp(allxy[:, 1])), 40.0) / 2 + 10
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy + half, cy - half)       # inverted: CARLA's y points down on a top view
    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="best")
    ax.set_title(f"{row['run_id']}  sc {row['scenario_id']}  {row['condition']}  {row['arm']}", fontsize=9)
    ax2.plot(tk["t_s"], tk["speed"], label="speed (m/s)")
    ax2.plot(tk["t_s"], tk["brake"] * 10, label="brake x10")
    ax2.plot(tk["t_s"], tk["cross_track_m"], label="lateral distance from route (m)")
    dist = []
    for h, p in zip(tk["hazards"], xy):
        d = [math.hypot(v[0] - p[0], v[1] - p[1]) for v in h.values()]
        dist.append(min(d) if d else np.nan)
    ax2.plot(tk["t_s"], np.clip(dist, 0, 30), "r:", label="nearest hazard (m, capped at 30)")
    ax2.set_xlabel("t (s)")
    ax2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/campaign.json")
    ap.add_argument("--runs", type=int, default=0, help="only the first N runs of the manifest")
    a = ap.parse_args()
    cfg = load_campaign(a.config)
    out = Path(cfg["_out"])
    rows = read_manifest(out / "manifest.csv")
    scen = json.loads(resolve_path(cfg["scenarios_resolved"]).read_text(encoding="utf-8"))
    todo = rows[: a.runs] if a.runs else [r for r in rows if (out / "runs" / r["run_id"]).exists()]
    plot_dir = out / "checks"
    plot_dir.mkdir(parents=True, exist_ok=True)
    infos, worst = {}, []
    for r in todo:
        rep, info = check_run(r, out / "runs" / r["run_id"], scen[r["scenario_id"]], cfg, plot_dir)
        infos[r["run_id"]] = (r, info)
        worst.append(rep.worst)
        print(f"{r['run_id']}  sc {r['scenario_id']}  {r['condition']:7}  {r['arm']:10}  -> {rep.worst}")
        print("\n".join(rep.lines))
    # level matched inside each block
    blocks = {}
    for rid, (r, info) in infos.items():
        if r["arm"] in LEVEL_ARMS and "level" in info:
            blocks.setdefault(r["block"], {})[r["arm"]] = info["level"]
    print("\nlevel_ratio inside each block (matsense, uniform and shuffled must agree):")
    n_block_checks = 0
    for b, lv in blocks.items():
        if len(lv) < 2:
            print(f"  {b}: only {list(lv)} so far")
            continue
        spread = max(lv.values()) - min(lv.values())
        lvl = "PASS" if spread < 0.01 else "FAIL"
        worst.append(lvl)
        n_block_checks += 1
        print(f"  {lvl}  {b}: " + ", ".join(f"{k}={v:.4f}" for k, v in lv.items()) + f"  (spread {spread:.4f})")
    walls = [i["wall_s"] for _, i in infos.values() if "wall_s" in i]
    if walls:
        per = float(np.median(walls))
        print(f"\ntime per run: median {per:.0f} s; the whole manifest ({len(rows)} runs) "
              f"needs about {len(rows) * (per + float(cfg['orchestrator'].get('pause_between_runs_s', 3))) / 3600:.1f} h")
    n_fail = worst.count("FAIL")
    runs_v, blocks_v = worst[:len(worst) - n_block_checks], worst[len(worst) - n_block_checks:]
    print(f"\n{len(todo)} runs checked: {runs_v.count('PASS')} PASS, {runs_v.count('WARN')} WARN, "
          f"{runs_v.count('FAIL')} FAIL; {len(blocks_v)} block level checks: {blocks_v.count('FAIL')} FAIL. "
          f"Pictures in {plot_dir}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
