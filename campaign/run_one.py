#!/usr/bin/env python3
"""Execute one closed-loop run of the manifest.

    python run_one.py --config config/campaign.json --run-id r0007

Writes into <out_dir>/runs/<run_id>/:
    ticks.csv      one row per simulation tick
    summary.json   termination, collisions, hashes, sensing statistics
    route.xml      the route given to PCLA
    points/        input and output of the operator every N frames
A run that ends normally writes summary.json last, so its presence marks a
completed run.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from matsense_campaign import code_sha256  # noqa: E402
from matsense_campaign.config import (load_campaign, load_world_message, read_manifest,  # noqa: E402
                                      resolve_path, town_matches)
from matsense_campaign.hook import SensingSession  # noqa: E402
from matsense_campaign.metrics import Polyline  # noqa: E402
from matsense_campaign.operator import OperatorParams  # noqa: E402
from matsense_campaign.profile import class_shuffled, load_profile  # noqa: E402
from matsense_campaign.routes import write_route  # noqa: E402
from matsense_campaign.scenarios import ScenarioRuntime  # noqa: E402

TICK_FIELDS = ["tick", "t_s", "x", "y", "z", "yaw", "speed", "throttle", "brake", "steer",
               "progress_m", "cross_track_m", "ttc_proxy_s", "collision",
               "n_in", "n_out", "drop_frac", "n_unmatched", "n_unmapped", "sem_lag_frames",
               "level_ratio", "scale", "n_clipped", "hook_ms", "hazards"]


def install_hook(fn) -> str:
    """Point the patched PCLA sensor interface at our perturbation function."""
    hits = []
    for name, mod in list(sys.modules.items()):
        # vars(), not hasattr(): a module with a module-level __getattr__ answers
        # hasattr() for any name and would pass for a patched PCLA that is not.
        if mod is not None and "MATSENSE_PERTURB_FN" in vars(mod):
            setattr(mod, "MATSENSE_PERTURB_FN", fn)
            hits.append(name)
    if not hits:
        raise RuntimeError("PCLA is not patched: no module defines MATSENSE_PERTURB_FN. "
                           "Run tools/patch_pcla.py first.")
    return ",".join(hits)


def import_pcla(pcla_dir: str):
    if pcla_dir not in sys.path:
        sys.path.insert(0, pcla_dir)
    from PCLA import PCLA  # noqa
    # make sure the sensor interface is imported, so the hook variable exists
    for cand in ("leaderboard_codes.sensor_interface", "sensor_interface"):
        try:
            __import__(cand)
        except Exception:
            pass
    return PCLA


def ttc_proxy(ego_xy, ego_yaw, speed, hazards: dict) -> float:
    fx, fy = math.cos(math.radians(ego_yaw)), math.sin(math.radians(ego_yaw))
    best = float("inf")
    for hx, hy in hazards.values():
        dx, dy = hx - ego_xy[0], hy - ego_xy[1]
        if dx * fx + dy * fy <= 0:
            continue
        d = math.hypot(dx, dy)
        if d < 40.0:
            best = min(best, d / max(speed, 0.1))
    return round(best, 3) if math.isfinite(best) else ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/campaign.json")
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()

    import carla

    cfg = load_campaign(args.config)
    out_root = Path(cfg["_out"])
    row = next(r for r in read_manifest(out_root / "manifest.csv") if r["run_id"] == args.run_id)
    run_dir = out_root / "runs" / row["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").unlink(missing_ok=True)

    scen_all = json.loads(resolve_path(cfg["scenarios_resolved"]).read_text(encoding="utf-8"))
    inst = scen_all[row["scenario_id"]]
    profile = load_profile(resolve_path(cfg["profile"]))
    op = cfg["operator"]
    params = OperatorParams(dropout_gain=op["dropout_gain"], dropout_cap=op["dropout_cap"],
                            jitter_gain=op["jitter_gain"], range_bin_m=op["range_bin_m"],
                            uniform_geometry=op["uniform_geometry"])
    cs_profile, cs_record = (class_shuffled(profile, int(cfg["class_shuffle_seed"]))
                             if row["arm"] == "shuffled_class" else (None, None))
    rc, cc = cfg["run"], cfg["carla"]
    dt = 1.0 / float(cc["fps"])
    seed = int(row["seed"]) * 100 + int(row["replicate"])

    client = carla.Client(cc["host"], int(cc["port"]))
    client.set_timeout(float(cc["timeout_s"]))
    world = client.get_world()
    current_map = world.get_map().name
    if not town_matches(current_map, row["town"]):
        if not cc.get("allow_load_world", True):
            raise RuntimeError(load_world_message(current_map, row["town"]))
        world = client.load_world(row["town"])
    elif cc.get("reload_world_each_run", True):
        world = client.reload_world()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = dt
    world.apply_settings(settings)
    world.set_weather(getattr(carla.WeatherParameters, cc.get("weather", "ClearNoon")))
    for tl in world.get_actors().filter("traffic.traffic_light*"):
        tl.set_state(carla.TrafficLightState.Green)
        tl.freeze(True)
    world.tick()

    actors, session, pcla, scenario = [], None, None, None
    summary = {"run": row, "code_sha256": code_sha256(), "profile_name": profile.name,
               "profile_sha256": profile.sha256, "operator": op, "agent": cfg["pcla"]["agent"],
               "class_shuffle": cs_record, "started_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    term, t = "max_time", 0.0
    collisions = []
    try:
        x, y, z, yaw = inst["route"][0]
        bp = world.get_blueprint_library().find(cfg["ego_blueprint"])
        bp.set_attribute("role_name", "hero")
        ego = world.spawn_actor(bp, carla.Transform(carla.Location(x=x, y=y, z=z + 0.5),
                                                    carla.Rotation(yaw=yaw)))
        actors.append(ego)
        ego.apply_control(carla.VehicleControl(brake=1.0, hand_brake=True))
        world.tick()

        coll_bp = world.get_blueprint_library().find("sensor.other.collision")
        coll = world.spawn_actor(coll_bp, carla.Transform(), attach_to=ego)
        coll.listen(lambda e: collisions.append(
            {"frame": int(e.frame),
             "other": e.other_actor.type_id if e.other_actor is not None else "static"}))
        actors.append(coll)

        scenario = ScenarioRuntime(world, inst)
        scenario.setup()
        world.tick()

        route_sha = write_route(inst["route"], row["town"], run_dir / "route.xml",
                                cfg["pcla"].get("route_format", "pcla_waypoint"))
        summary["route_sha256"] = route_sha

        dump = None
        if int(rc.get("dump_points_every", 0)) > 0:
            dump = run_dir / "points"
            dump.mkdir(exist_ok=True)
        session = SensingSession(world, ego, profile, row["condition"], row["arm"], params, seed,
                                 tol_deg=float(op.get("tolerance_deg", 0.40)),
                                 class_shuffled_profile=cs_profile, dump_dir=dump,
                                 dump_every=int(rc.get("dump_points_every", 20)) or 20)
        # The operator must be applying the profile the run records.
        if session.profile.sha256 != summary["profile_sha256"]:
            raise RuntimeError("profile mismatch between requested and applied profile")

        PCLA = import_pcla(cfg["pcla"]["dir"])
        summary["hook_modules"] = install_hook(session.perturb)
        pcla = PCLA(cfg["pcla"]["agent"], ego, str(run_dir / "route.xml"), client)
        session.attach()
        summary["semantic_lidar"] = session.mirrored_attributes
        ego.apply_control(carla.VehicleControl(hand_brake=False))

        path = Polyline(np.asarray(inst["ego_path"])[:, :2])
        f = open(run_dir / "ticks.csv", "w", newline="", encoding="utf-8")
        w = csv.DictWriter(f, fieldnames=TICK_FIELDS)
        w.writeheader()
        moved, still_since, tick = False, None, 0
        start_xy = (x, y)
        max_ticks = int(float(rc["max_seconds"]) / dt)
        while tick < max_ticks:
            world.tick()
            t = tick * dt
            tf = ego.get_transform()
            ego_xy = (tf.location.x, tf.location.y)
            scenario.step(dt, ego_xy)
            control = pcla.get_action()
            ego.apply_control(control)
            v = ego.get_velocity()
            speed = math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
            prog, xt = path.project(*ego_xy)
            hz = scenario.hazard_positions()
            st = session.last_stats
            w.writerow({
                "tick": tick, "t_s": round(t, 3), "x": round(ego_xy[0], 3), "y": round(ego_xy[1], 3),
                "z": round(tf.location.z, 3), "yaw": round(tf.rotation.yaw, 2), "speed": round(speed, 3),
                "throttle": round(control.throttle, 3), "brake": round(control.brake, 3),
                "steer": round(control.steer, 4), "progress_m": round(prog, 3),
                "cross_track_m": round(xt, 3), "ttc_proxy_s": ttc_proxy(ego_xy, tf.rotation.yaw, speed, hz),
                "collision": len(collisions),
                "n_in": st.get("n_in", ""), "n_out": st.get("n_out", ""),
                "drop_frac": round(1 - st["n_out"] / max(st["n_in"], 1), 5) if st else "",
                "n_unmatched": st.get("n_unmatched", ""), "n_unmapped": st.get("n_unmapped", ""),
                "sem_lag_frames": st.get("sem_lag_frames", ""), "level_ratio": round(st.get("level_ratio", 1.0), 5) if st else "",
                "scale": round(st.get("scale", 1.0), 5) if st else "", "n_clipped": st.get("n_clipped", ""),
                "hook_ms": st.get("hook_ms", ""), "hazards": json.dumps(hz)})
            if math.hypot(ego_xy[0] - start_xy[0], ego_xy[1] - start_xy[1]) > 1.0:
                moved = True
            if prog >= path.length - 2.0:
                term = "route_completed"
                break
            if xt > float(rc["deviation_m"]):
                term = "route_deviation"
                break
            if collisions and rc.get("terminate_on_collision", False):
                term = "collision"
                break
            if not moved and t > float(rc["never_started_s"]):
                term = "never_started"
                break
            if moved and speed < 0.1:
                still_since = t if still_since is None else still_since
                if t - still_since > float(rc["stationary_s"]):
                    term = "stationary"
                    break
            else:
                still_since = None
            tick += 1
        f.close()
        summary.update(termination=term, duration_s=round(t, 3), route_length_m=round(path.length, 2),
                       final_progress_m=round(prog, 2), completion=round(min(1.0, prog / path.length), 4),
                       collisions=collisions, n_hook_calls=session.n_calls)
        status = 0
    except Exception as e:
        summary.update(termination="error", error=repr(e), traceback=traceback.format_exc())
        print(traceback.format_exc(), flush=True)
        status = 2
    finally:
        for obj in (pcla, session, scenario):
            try:
                if obj is not None:
                    obj.cleanup()
            except Exception:
                pass
        for a in reversed(actors):
            try:
                if hasattr(a, "stop"):
                    a.stop()
                a.destroy()
            except Exception:
                pass
        try:
            s = world.get_settings()
            s.synchronous_mode = False
            s.fixed_delta_seconds = None
            world.apply_settings(s)
        except Exception:
            pass
    summary["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    name = "summary.json" if status == 0 else "error.json"
    (run_dir / name).write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"[run_one] {row['run_id']} {row['scenario_id']} {row['condition']} {row['arm']} "
          f"-> {summary.get('termination')}", flush=True)
    return status


if __name__ == "__main__":
    sys.exit(main())
