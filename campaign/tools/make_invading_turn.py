#!/usr/bin/env python3
"""Rewrite the frozen scenario-15 entry as the Leaderboard 2.0 InvadingTurn.

    python tools/make_invading_turn.py            (dry run: prints what it would write)
    python tools/make_invading_turn.py --apply

Leaderboard 2.0 (scenario_runner, srunner/scenarios/invading_turn.py):
  * construction cones in the oncoming lane, one every 5 m over 50 m, displaced
    0.8 * lane_width / 2 towards the outside of that lane;
  * a flow of oncoming vehicles, one every 40 m, driving with a lateral offset
    towards the ego lane of offset + lane_width / 2 - 0.7 m (offset = 0.25).
Here the cones are centred on the bend, three vehicles 40 m apart are released
together when the ego is 55 m from the apex, and everything else (ego route,
vehicle speed, trigger) is kept from the existing entry. The geometry is read
from the town's OpenDRIVE file, so no simulator is needed. Only data changes:
the campaign code, and therefore its hash, is untouched.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
XODR = Path(r"C:\Users\debor\Desktop\carla\carla\Unreal\CarlaUE4\Content\Carla\Maps\OpenDrive")
OFFSET, OFFSET_CONST = 0.25, 0.7           # Leaderboard defaults
CONE_GAP, CONE_RUN, FLOW_GAP, N_VEH = 5.0, 50.0, 40.0, 3
ROAD_Z = 0.21                               # Town02 road surface above the waypoint z (measured 2026-10-19)


def pose(wp, lateral=0.0, z=0.0):
    t = wp.transform
    r = t.get_right_vector()
    return [round(t.location.x + lateral * r.x, 3), round(t.location.y + lateral * r.y, 3),
            round(t.location.z + z, 3), round(t.rotation.yaw, 2)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    import carla
    path = ROOT / "config" / "scenarios_resolved.json"
    sc = json.loads(path.read_text(encoding="utf-8"))
    inst = sc["15"]
    m = carla.Map(inst["town"], (XODR / f"{inst['town']}.xodr").read_text(encoding="utf-8"))
    ego = np.asarray(inst["ego_path"])
    old = inst["actors"][0]
    apex = inst.get("apex") or old["trigger"]["point"]

    # oncoming lane: left neighbour of the ego lane at the end of the ego route,
    # extended upstream so that three vehicles 40 m apart fit before the bend
    end_wp = m.get_waypoint(carla.Location(float(ego[-1, 0]), float(ego[-1, 1]), 0.0))
    onc = end_wp.get_left_lane()
    if onc is None or onc.lane_id * end_wp.lane_id > 0:
        raise RuntimeError("no opposite lane next to the end of the ego route")
    up = [onc]
    for _ in range(int((N_VEH - 1) * FLOW_GAP / 2.0) + 5):
        prv = up[-1].previous(2.0)
        if not prv:
            break
        up.append(prv[0])
    down = [onc]
    while True:   # follow the oncoming lane through the bend, past the ego start
        nxt = down[-1].next(2.0)
        if not nxt:
            break
        w = nxt[0]
        down.append(w)
        if math.hypot(w.transform.location.x - ego[0, 0], w.transform.location.y - ego[0, 1]) < 4 or len(down) > 200:
            break
    lane = up[::-1] + down[1:]                    # upstream -> downstream (oncoming direction)
    lw = lane[len(lane) // 2].lane_width
    shift = -(OFFSET + lw / 2.0 - OFFSET_CONST)   # negative = left of the oncoming lane = towards the ego
    veh_path = [pose(w, shift, ROAD_Z) for w in lane]

    # cones: centred on the oncoming-lane point nearest the apex, outer side of the lane
    d = [math.hypot(w.transform.location.x - apex[0], w.transform.location.y - apex[1]) for w in lane]
    k_apex = int(np.argmin(d))
    k0 = max(0, k_apex - int(CONE_RUN / 4.0))
    cones = []
    for i in range(int(CONE_RUN / CONE_GAP)):
        k = k0 + int(i * CONE_GAP / 2.0)
        if k < len(lane):
            cones.append(pose(lane[k], 0.8 * lw / 2.0, ROAD_Z))

    # the flow: as in the Leaderboard, a running flow with one vehicle every 40 m over
    # the whole oncoming path, moving from the first tick, so that whatever the ego's
    # speed a vehicle is in the bend when the ego reaches it
    # (no vehicle starts within 20 m of the end of the path, which lies beside the ego's start)
    starts = list(range(0, len(veh_path) - 11, int(FLOW_GAP / 2.0)))
    actors = []
    for j, s in enumerate(starts):
        actors.append({"role": f"oncoming_{j}", "blueprint": old["blueprint"], "path": veh_path[s:],
                       "speed": old["speed"], "trigger": {"type": "time", "t": 0.0}})
    for i, c in enumerate(cones):
        actors.append({"role": f"cone_{i}", "blueprint": "static.prop.constructioncone", "path": [c],
                       "speed": 0.0, "trigger": None})

    new = dict(inst, actors=actors, kind="invading_bend", apex=apex,
               note_invading_turn=(f"Leaderboard 2.0 InvadingTurn: {len(cones)} cones every {CONE_GAP} m on the outer "
                                   f"side of the oncoming lane around the bend; a running flow of {len(starts)} oncoming "
                                   f"vehicles {FLOW_GAP} m apart over the whole oncoming path, moving from t=0, offset "
                                   f"{shift:.2f} m towards the ego lane (offset {OFFSET}, lane {lw} m). "
                                   f"Generated by tools/make_invading_turn.py; z includes the Town02 road surface "
                                   f"({ROAD_Z} m)."))
    new.pop("note_z", None)
    # vehicle centre is lw + shift from the ego-lane centre; half width of the Audi A2 ~0.89 m
    print(f"lane width {lw} m, vehicle offset {shift:.2f} m, body inside ego lane about "
          f"{lw / 2 - (lw + shift - 0.89):.2f} m")
    print(f"vehicles start at path index {starts}, "
          f"path {len(veh_path)} points; cones {len(cones)} from {cones[0][:2]} to {cones[-1][:2]}")
    if a.apply:
        sc["15"] = new
        path.write_text(json.dumps(sc, indent=1), encoding="utf-8")
        print("written", path)
    else:
        print("dry run: add --apply to write")
    return 0


if __name__ == "__main__":
    sys.exit(main())
