#!/usr/bin/env python3
"""Find a location for each scenario and freeze it (CARLA must be running).

    python tools/resolve_scenarios.py                       all enabled scenarios
    python tools/resolve_scenarios.py --only 15 --candidate 2   try another location

Writes config/scenarios_resolved.json and config/scenario_<id>.png (top view of
the map with the ego route in blue, hazard paths in red, triggers as circles).
Look at every picture before planning the campaign. If a location is unsuitable,
rerun with --candidate N for that scenario. Once the campaign has started, do not
resolve again: every run must use the same coordinates.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from matsense_campaign.config import load_campaign, load_world_message, resolve_path, town_matches  # noqa: E402
from matsense_campaign.scenarios import resolve  # noqa: E402


def plot(inst, cmap, path: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    wps = cmap.generate_waypoints(3.0)
    xy = np.array([[w.transform.location.x, w.transform.location.y] for w in wps])
    ego = np.array(inst["ego_path"])
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(xy[:, 0], xy[:, 1], s=0.3, c="0.8")
    ax.plot(ego[:, 0], ego[:, 1], "b-", lw=2, label="ego route")
    ax.plot(ego[0, 0], ego[0, 1], "bo")
    for a in inst["actors"]:
        p = np.array(a["path"])
        ax.plot(p[:, 0], p[:, 1], "r-" if len(p) > 1 else "rs", lw=2)
        ax.annotate(a["role"], p[0, :2], fontsize=7, color="r")
        tr = a.get("trigger")
        if tr and tr["type"] == "ego_near":
            ax.add_patch(plt.Circle(tr["point"], tr["distance"], fill=False, ls="--", color="orange"))
    pad = 40
    allp = np.vstack([ego[:, :2]] + [np.array(a["path"])[:, :2] for a in inst["actors"]])
    ax.set_xlim(allp[:, 0].min() - pad, allp[:, 0].max() + pad)
    ax.set_ylim(allp[:, 1].min() - pad, allp[:, 1].max() + pad)
    ax.invert_yaxis()   # CARLA y points down on the top view
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/campaign.json")
    ap.add_argument("--only", default=None, help="scenario id to (re)resolve")
    ap.add_argument("--candidate", type=int, default=None, help="candidate_index override")
    a = ap.parse_args()
    import carla

    cfg = load_campaign(a.config)
    out = resolve_path(cfg["scenarios_resolved"])
    resolved = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    client = carla.Client(cfg["carla"]["host"], int(cfg["carla"]["port"]))
    client.set_timeout(float(cfg["carla"]["timeout_s"]))
    for sc in cfg["scenarios"]:
        if a.only and sc["id"] != a.only:
            continue
        if not a.only and not sc.get("enabled", True):
            continue
        world = client.get_world()
        if not town_matches(world.get_map().name, sc["town"]):
            if not cfg["carla"].get("allow_load_world", True):
                print(f"skip {sc['id']}: " + load_world_message(world.get_map().name, sc["town"]))
                continue
            world = client.load_world(sc["town"])
        params = dict(sc.get("params", {}))
        if a.candidate is not None:
            params["candidate_index"] = a.candidate
        inst = resolve(sc["kind"], world.get_map(), world, params)
        inst.update(id=sc["id"], kind=sc["kind"], town=sc["town"], params=params)
        resolved[sc["id"]] = inst
        png = out.parent / f"scenario_{sc['id']}.png"
        plot(inst, world.get_map(), png, f"{sc['id']} {sc['kind']} ({sc['town']})")
        print(f"{sc['id']} {sc['kind']}: ego start {inst['route'][0][:2]}, "
              f"{len(inst['actors'])} actors -> {png}")
    out.write_text(json.dumps(resolved, indent=1), encoding="utf-8")
    print(f"written {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
