#!/usr/bin/env python3
"""Acquire ONE LiDAR sequence in CARLA with no driving agent, for phase 1.

The ego is never driven by an ADS, by PCLA or by CARLA's autopilot: it is
teleported, with physics off, along fixed poses. The poses come either from a
JSON file (for example real poses, whose ids and timestamps are kept) or from a
prescribed path along the road network. Two LiDARs with identical attributes and
the same mounting record every frame: the ray-cast LiDAR (geometry and CARLA's
native intensity) and the semantic LiDAR (one object tag per ray). Every mode of
phase 1 is later derived from this single acquisition, so the modes cannot
differ in sampling.

Per frame it writes frames/<frame>.npz with

    raw_xyz (N,3) float32, raw_intensity (N,) float32     ray-cast LiDAR
    sem (M,) structured x,y,z,CosAngle,ObjIdx,ObjTag      semantic LiDAR
    pose  [x, y, z, roll, pitch, yaw] of the ego (world)
    sensor_to_world (4,4)                                  LiDAR pose (world)

and actors/<frame>.json with every vehicle and walker: id, type, world pose,
bounding box, and the box centre and yaw in the LiDAR frame (the reference for
phase 2). manifest.json records the CARLA versions, the map, the sensor
attributes as CARLA reports them, the pose source, and the hashes of the
configuration and of this script.

It never loads a map (open the town yourself), never overwrites an existing
output folder, and restores the world settings on exit.

    python acquire_sequence.py --config ../config/phase1.yaml --out ../data/seq_town10_001
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import queue
import sys
import time
from pathlib import Path

import numpy as np

SEM_DTYPE = np.dtype([("x", np.float32), ("y", np.float32), ("z", np.float32),
                      ("CosAngle", np.float32), ("ObjIdx", np.uint32), ("ObjTag", np.uint32)])


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_config(path: Path) -> dict:
    import yaml
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def poses_from_file(path: Path) -> list[dict]:
    """[{id, t?, x, y, z, roll?, pitch?, yaw}], kept as given."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = data["poses"] if isinstance(data, dict) else data
    out = []
    for k, r in enumerate(rows):
        out.append({"id": r.get("id", k), "t": r.get("t"), "x": float(r["x"]), "y": float(r["y"]),
                    "z": float(r.get("z", 0.0)), "roll": float(r.get("roll", 0.0)),
                    "pitch": float(r.get("pitch", 0.0)), "yaw": float(r["yaw"])})
    return out


def poses_along_road(world_map, start_index: int, step_m: float, n: int, z_offset: float) -> list[dict]:
    """A prescribed path: follow the lane from a spawn point, one pose every step_m."""
    sp = world_map.get_spawn_points()[start_index]
    wp = world_map.get_waypoint(sp.location)
    out = []
    for k in range(n):
        tr = wp.transform
        out.append({"id": k, "t": None, "x": tr.location.x, "y": tr.location.y,
                    "z": tr.location.z + z_offset, "roll": 0.0, "pitch": tr.rotation.pitch,
                    "yaw": tr.rotation.yaw})
        nxt = wp.next(step_m)
        if not nxt:
            break
        wp = nxt[0]
    return out


def to_sensor_frame(world_to_sensor: np.ndarray, loc) -> list[float]:
    p = world_to_sensor @ np.array([loc.x, loc.y, loc.z, 1.0])
    return [float(p[0]), float(p[1]), float(p[2])]


def actor_record(a, world_to_sensor: np.ndarray, sensor_yaw: float) -> dict:
    tr = a.get_transform()
    bb = a.bounding_box
    centre = tr.transform(bb.location)
    return {
        "id": a.id, "type": a.type_id,
        "world": {"x": tr.location.x, "y": tr.location.y, "z": tr.location.z,
                  "roll": tr.rotation.roll, "pitch": tr.rotation.pitch, "yaw": tr.rotation.yaw},
        "bbox_extent": [bb.extent.x, bb.extent.y, bb.extent.z],
        "bbox_centre_world": [centre.x, centre.y, centre.z],
        "bbox_centre_sensor": to_sensor_frame(world_to_sensor, centre),
        "yaw_sensor_deg": (tr.rotation.yaw - sensor_yaw + 180.0) % 360.0 - 180.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists: refusing to overwrite an earlier acquisition")

    import carla
    host = args.host or cfg["carla"]["host"]
    port = args.port or cfg["carla"]["port"]
    client = carla.Client(host, port)
    client.set_timeout(float(cfg["carla"].get("timeout_s", 30.0)))
    world = client.get_world()
    want = cfg["scene"]["town"]
    have = world.get_map().name.split("/")[-1]
    if want.split("/")[-1] not in have:
        raise SystemExit(f"STOP: open {want} in CARLA (now {have}); this script never loads a map")

    pose_src = cfg["scene"]["poses"]
    if pose_src["source"] == "file":
        poses = poses_from_file(Path(pose_src["path"]))
    else:
        poses = poses_along_road(world.get_map(), int(pose_src["spawn_index"]), float(pose_src["step_m"]),
                                 int(pose_src["n_frames"]), float(pose_src.get("z_offset", 0.0)))
    if not poses:
        raise SystemExit("no poses")

    original = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / float(cfg["carla"]["fps"])
    actors, sensors = [], []
    try:
        world.apply_settings(settings)
        bpl = world.get_blueprint_library()

        def tr_of(p):
            return carla.Transform(carla.Location(x=p["x"], y=p["y"], z=p["z"]),
                                   carla.Rotation(roll=p["roll"], pitch=p["pitch"], yaw=p["yaw"]))

        ego = world.spawn_actor(bpl.find(cfg["scene"]["ego_blueprint"]), tr_of(poses[0]))
        ego.set_simulate_physics(False)
        actors.append(ego)

        # static vehicles, so that the vehicle class is represented; no autopilot
        sv = cfg["scene"].get("static_vehicles", {})
        if int(sv.get("count", 0)) > 0:
            rng = np.random.default_rng(int(sv.get("seed", 0)))
            path = np.array([[p["x"], p["y"]] for p in poses])
            cands = [s for s in world.get_map().get_spawn_points()
                     if np.min(np.hypot(path[:, 0] - s.location.x, path[:, 1] - s.location.y))
                     < float(sv.get("max_distance_m", 60.0))
                     and np.min(np.hypot(path[:, 0] - s.location.x, path[:, 1] - s.location.y))
                     > float(sv.get("min_distance_m", 4.0))]
            rng.shuffle(cands)
            vbps = [b for b in bpl.filter("vehicle.*") if int(b.get_attribute("number_of_wheels")) == 4]
            for s in cands[: int(sv["count"])]:
                a = world.try_spawn_actor(vbps[int(rng.integers(len(vbps)))], s)
                if a is not None:
                    a.set_simulate_physics(False)
                    actors.append(a)

        lid = cfg["lidar"]
        mount = carla.Transform(carla.Location(*lid["mount_xyz"]))
        queues = {}
        for kind in ("ray_cast", "ray_cast_semantic"):
            bp = bpl.find(f"sensor.lidar.{kind}")
            for k, v in lid["attributes"].items():
                if bp.has_attribute(k):
                    bp.set_attribute(k, str(v))
            s = world.spawn_actor(bp, mount, attach_to=ego)
            q = queue.Queue()
            s.listen(q.put)
            sensors.append(s)
            queues[kind] = (s, q)

        def take(q, frame, timeout):
            end = time.time() + timeout
            while time.time() < end:
                m = q.get(timeout=max(0.01, end - time.time()))
                if m.frame == frame:
                    return m
            raise TimeoutError(f"sensor data for frame {frame} not received")

        (out / "frames").mkdir(parents=True)
        (out / "actors").mkdir()
        index = []
        for k in range(int(cfg["carla"].get("warmup_ticks", 5))):
            world.tick()
        for p in poses:
            ego.set_transform(tr_of(p))
            frame = world.tick()
            raw = take(queues["ray_cast"][1], frame, 10.0)
            sem = take(queues["ray_cast_semantic"][1], frame, 10.0)
            pts = np.frombuffer(raw.raw_data, dtype=np.float32).reshape(-1, 4)
            semd = np.frombuffer(sem.raw_data, dtype=SEM_DTYPE).copy()
            s_tr = queues["ray_cast"][0].get_transform()
            s2w = np.array(s_tr.get_matrix())
            w2s = np.array(s_tr.get_inverse_matrix())
            e_tr = ego.get_transform()
            np.savez_compressed(out / "frames" / f"{frame:08d}.npz",
                                raw_xyz=pts[:, :3].copy(), raw_intensity=pts[:, 3].copy(), sem=semd,
                                pose=np.array([e_tr.location.x, e_tr.location.y, e_tr.location.z,
                                               e_tr.rotation.roll, e_tr.rotation.pitch, e_tr.rotation.yaw]),
                                sensor_to_world=s2w, pose_id=str(p["id"]))
            others = [a for a in world.get_actors()
                      if (a.type_id.startswith("vehicle.") or a.type_id.startswith("walker.")) and a.id != ego.id]
            (out / "actors" / f"{frame:08d}.json").write_text(json.dumps(
                [actor_record(a, w2s, s_tr.rotation.yaw) for a in others], indent=1))
            index.append({"frame": frame, "pose_id": p["id"], "pose_t": p.get("t"),
                          "sim_t": raw.timestamp, "n_raw": int(pts.shape[0]), "n_sem": int(semd.shape[0])})
            print(f"  frame {frame}  pose {p['id']}  raw {pts.shape[0]}  sem {semd.shape[0]}", flush=True)

        manifest = {
            "tool": "evaluation_v2/phase1/acquire_sequence.py",
            "tool_sha256": sha256_file(Path(__file__)),
            "config": cfg, "config_sha256": sha256_file(Path(args.config)),
            "carla_client_version": getattr(carla, "__version__", "unknown"),
            "carla_server_version": client.get_server_version(),
            "map": world.get_map().name,
            "sensor_attributes": {kind: dict(s.attributes) for kind, (s, _) in queues.items()},
            "mount_xyz": lid["mount_xyz"],
            "pose_source": pose_src, "n_frames": len(index),
            "ego_driven_by": "teleport, physics off (no ADS, no autopilot)",
            "static_vehicles": len(actors) - 1,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "frames": index,
        }
        (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
        print(f"\n  {len(index)} frames -> {out}")
    finally:
        for s in sensors:
            try:
                s.stop(); s.destroy()
            except Exception:
                pass
        for a in actors:
            try:
                a.destroy()
            except Exception:
                pass
        world.apply_settings(original)
    return 0


if __name__ == "__main__":
    sys.exit(main())
