#!/usr/bin/env python3
"""End-to-end check: does the intensity of each mode reach the detector's input?

Run on the Autoware machine while one mode's bag is replayed:

    ros2 bag play <bags>/MATSENSE --clock            # terminal 1, Autoware running with use_sim_time
    python probe_detector_input.py --bags <bags> --mode MATSENSE \
        --topic <input topic of the detector node> --out probe_MATSENSE.json

Find the input topic from the running system, not from documentation:

    ros2 node list | grep -Ei 'centerpoint|transfusion'
    ros2 node info /perception/object_recognition/detection/centerpoint/lidar_centerpoint

The probe subscribes to that topic, and for every received cloud whose stamp is
in the bag it compares the cloud with what the bag published:

    layout       PointXYZIRC fields (a different layout makes CenterPoint/TransFusion
                 skip the cloud)
    points       count; if equal, intensity byte per point in order
    intensity    when the counts differ (cropping, downsampling, concatenation),
                 the received intensity values must be a sub-multiset of the bag's
                 values, and their histogram is reported
    geometry     xyz equal (same frame) or a constant offset (rigid transform
                 without rotation, e.g. velodyne_top -> base_link)

Verdict REACHES when the received intensity bytes are the bag's (per point or
as a sub-multiset) in every matched cloud, NOT_REACHED when they are constant
or unrelated, NO_DATA when no cloud matched a bag stamp. Running the probe for
two modes and comparing their outputs shows whether the detector input differs
between modes; it does not show whether the model uses intensity (the default
CenterPoint does not; see AUDIT.md).

The comparison itself (compare_cloud) is pure and is unit-tested without ROS.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from to_rosbag2 import POINT_DTYPE, TOPIC  # noqa: E402
from verify_rosbag2 import EXPECTED  # noqa: E402


def compare_cloud(expected: np.ndarray, got: np.ndarray, fields: list[tuple]) -> dict:
    """expected, got: POINT_DTYPE arrays; fields: [(name, offset, datatype)] of the received cloud."""
    res = {"layout_ok": [tuple(f) for f in fields[:6]] == EXPECTED, "n_expected": int(expected.size),
           "n_got": int(got.size)}
    if got.size == 0:
        res["intensity"] = "empty"
        return res
    gi, ei = got["intensity"], expected["intensity"]
    res["got_intensity_unique"] = int(np.unique(gi).size)
    if got.size == expected.size:
        res["intensity_equal_per_point"] = bool(np.array_equal(gi, ei))
        d = np.stack([got[c].astype(np.float64) - expected[c].astype(np.float64) for c in ("x", "y", "z")], 1)
        res["xyz_offset"] = d.mean(0).round(4).tolist()
        res["xyz_rigid_translation"] = bool(np.all(np.abs(d - d.mean(0)) < 1e-3))
    ce, cg = Counter(ei.tolist()), Counter(gi.tolist())
    res["intensity_sub_multiset"] = all(cg[v] <= ce.get(v, 0) for v in cg)
    res["reaches"] = bool(res.get("intensity_equal_per_point") or res["intensity_sub_multiset"]) \
        and not (res["got_intensity_unique"] == 1 and np.unique(ei).size > 1)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bags", required=True)
    ap.add_argument("--mode", required=True)
    ap.add_argument("--topic", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-clouds", type=int, default=50)
    ap.add_argument("--timeout-s", type=float, default=120.0)
    args = ap.parse_args()

    from verify_rosbag2 import read_bag
    bm = json.loads((Path(args.bags) / "bags_manifest.json").read_text())
    exp = {msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec:
           np.frombuffer(bytes(msg.data), dtype=POINT_DTYPE).copy()
           for _, msg in read_bag(Path(bm["modes"][args.mode]["bag"]))}

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import PointCloud2

    results = []

    class Probe(Node):
        def __init__(self):
            super().__init__("matsense_input_probe")
            self.create_subscription(PointCloud2, args.topic, self.cb, qos_profile_sensor_data)

        def cb(self, msg):
            t = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
            if t not in exp:
                return
            fields = [(f.name, f.offset, f.datatype) for f in msg.fields]
            if msg.point_step != 16 or [tuple(f) for f in fields[:6]] != EXPECTED:
                results.append({"stamp_ns": t, "frame_id": msg.header.frame_id, "layout_ok": False,
                                "fields": fields, "point_step": msg.point_step})
                return
            got = np.frombuffer(bytes(msg.data), dtype=POINT_DTYPE)
            results.append({"stamp_ns": t, "frame_id": msg.header.frame_id, **compare_cloud(exp[t], got, fields)})

    rclpy.init()
    node = Probe()
    import time
    end = time.time() + args.timeout_s
    while rclpy.ok() and time.time() < end and len(results) < args.max_clouds:
        rclpy.spin_once(node, timeout_sec=0.5)
    node.destroy_node()
    rclpy.shutdown()

    if not results:
        verdict = "NO_DATA"
    elif all(r.get("layout_ok") and r.get("reaches") for r in results):
        verdict = "REACHES"
    else:
        verdict = "NOT_REACHED"
    report = {"mode": args.mode, "topic": args.topic, "source_topic": TOPIC, "verdict": verdict,
              "clouds_matched": len(results), "clouds_in_bag": len(exp), "clouds": results}
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"  {args.mode}: {verdict} ({len(results)} clouds matched on {args.topic}) -> {args.out}")
    return 0 if verdict == "REACHES" else 1


if __name__ == "__main__":
    sys.exit(main())
