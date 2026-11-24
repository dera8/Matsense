#!/usr/bin/env python3
"""Ten-second check that a semantic LiDAR can run next to a normal LiDAR.

    python tools/check_semantic_lidar.py

On some CARLA builds the pair stops delivering data after a few ticks. This
script spawns an ego with a 64-channel LiDAR and a mirrored semantic LiDAR in
synchronous mode, ticks 200 times, and reports how many frames each sensor
delivered, how many returns were associated with a tag, and how long the
association took.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from matsense_campaign.association import SEMANTIC_LIDAR_DTYPE, match_tags  # noqa: E402
from matsense_campaign.hook import mirror_attributes  # noqa: E402


def main() -> int:
    import carla
    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(60.0)
    world = client.get_world()
    s = world.get_settings()
    s.synchronous_mode, s.fixed_delta_seconds = True, 0.05
    world.apply_settings(s)
    bpl = world.get_blueprint_library()
    ego = world.spawn_actor(bpl.find("vehicle.lincoln.mkz_2020"), world.get_map().get_spawn_points()[0])
    # The attributes PCLA gives every LiDAR (rotation 10 Hz at 20 fps: half a turn
    # per frame). The semantic mirror is spawned one tick later, as in run_one.py.
    attrs = {"channels": "64", "range": "85", "points_per_second": "600000",
             "rotation_frequency": "10", "upper_fov": "10", "lower_fov": "-30"}
    tf = carla.Transform(carla.Location(z=2.4))
    got = {"lidar": {}, "sem": {}}
    sensors = []
    for kind, bpname in (("lidar", "sensor.lidar.ray_cast"), ("sem", "sensor.lidar.ray_cast_semantic")):
        bp = bpl.find(bpname)
        for k, v in (attrs if kind == "lidar" else mirror_attributes(attrs, 20.0)).items():
            bp.set_attribute(k, v)
        sen = world.spawn_actor(bp, tf, attach_to=ego)
        sen.listen(lambda m, k=kind: got[k].__setitem__(m.frame, bytes(m.raw_data)))
        sensors.append(sen)
        world.tick()
    ego.set_autopilot(False)
    ms, fracs = [], []
    try:
        for _ in range(200):
            fr = world.tick()
            time.sleep(0.01)
            if fr in got["lidar"] and fr in got["sem"]:
                pts = np.frombuffer(got["lidar"][fr], dtype=np.float32).reshape(-1, 4)
                sem = np.frombuffer(got["sem"][fr], dtype=SEMANTIC_LIDAR_DTYPE)
                t0 = time.perf_counter()
                _, matched = match_tags(pts[:, :3], sem)
                ms.append(1000 * (time.perf_counter() - t0))
                fracs.append(matched.mean() if len(matched) else 0)
    finally:
        for sen in sensors:
            sen.stop()
            sen.destroy()
        ego.destroy()
        s.synchronous_mode = False
        world.apply_settings(s)
    print(f"lidar frames: {len(got['lidar'])}, semantic frames: {len(got['sem'])} (expected ~200 each)")
    if ms:
        print(f"matched fraction: median {np.median(fracs):.3f}; association time: median {np.median(ms):.1f} ms")
    ok = len(got["lidar"]) > 150 and len(got["sem"]) > 150
    if not ok:
        print("PROBLEM: one of the sensors stopped delivering data")
        return 1
    if not fracs or np.percentile(fracs, 5) < 0.9:
        print("PROBLEM: the semantic mirror does not cover the LiDAR returns (matched fraction < 0.9)")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
