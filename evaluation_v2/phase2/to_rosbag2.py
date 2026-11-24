#!/usr/bin/env python3
"""Middleware layer: the four modes of a paired dataset as four ROS 2 bags.

The sensing logic stays in phase1 (modes.py); this script only encodes clouds
for Autoware, the way autoware_carla_interface does for a live CARLA (carla_ros.py,
lidar()), so that a replayed cloud is what the CARLA sensor kit would publish:

    layout       PointXYZIRC, 16 bytes: x, y, z FLOAT32 at 0/4/8, intensity UINT8
                 at 12, return_type UINT8 at 13, channel UINT16 at 14
    intensity    uint8(clip(I, 0, 1) * 255), truncated, as the interface does
    y            negated (CARLA left-handed -> ROS right-handed)
    return_type  0;  channel: the ring of each point when the acquisition kept it
    topic        /sensing/lidar/top/pointcloud_before_sync, frame velodyne_top
    tf_static    base_link -> velodyne_top at the acquisition's mounting

Every bag has the same messages at the same stamps: only the intensity bytes
differ between modes. One folder per mode, never overwritten.

Autoware's default detector (CenterPoint v4.1) does not read intensity; these
bags are meant for a detector that does (TransFusion), chosen before any run and
kept fixed across modes, and for the end-to-end input check.

    python to_rosbag2.py ../data/paired_town03_001 --out ../data/bags_town03_001
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

TOPIC = "/sensing/lidar/top/pointcloud_before_sync"
FRAME = "velodyne_top"
POINT_DTYPE = np.dtype({"names": ["x", "y", "z", "intensity", "return_type", "channel"],
                        "formats": ["<f4", "<f4", "<f4", "u1", "u1", "<u2"],
                        "offsets": [0, 4, 8, 12, 13, 14], "itemsize": 16})


def encode(xyz: np.ndarray, intensity: np.ndarray, channel: np.ndarray | None) -> np.ndarray:
    pts = np.zeros(xyz.shape[0], dtype=POINT_DTYPE)
    pts["x"] = xyz[:, 0]
    pts["y"] = -xyz[:, 1]
    pts["z"] = xyz[:, 2]
    pts["intensity"] = (np.clip(intensity.astype(np.float64), 0, 1) * 255).astype(np.uint8)
    pts["return_type"] = 0
    pts["channel"] = 0 if channel is None else channel
    return pts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mount-xyz", type=float, nargs=3, default=None,
                    help="LiDAR mounting in base_link (default: the acquisition manifest, else 0 0 2.40)")
    ap.add_argument("--period-s", type=float, default=0.05, help="stamp step when the source has none")
    ap.add_argument("--distro", default="humble", choices=["humble", "jazzy"])
    args = ap.parse_args()

    from rosbags.rosbag2 import Writer
    from rosbags.typesys import Stores, get_typestore
    ts = get_typestore(Stores.ROS2_HUMBLE if args.distro == "humble" else Stores.ROS2_JAZZY)
    PointCloud2 = ts.types["sensor_msgs/msg/PointCloud2"]
    PointField = ts.types["sensor_msgs/msg/PointField"]
    Header = ts.types["std_msgs/msg/Header"]
    Time = ts.types["builtin_interfaces/msg/Time"]
    TFMessage = ts.types["tf2_msgs/msg/TFMessage"]
    TransformStamped = ts.types["geometry_msgs/msg/TransformStamped"]
    Transform = ts.types["geometry_msgs/msg/Transform"]
    Vector3 = ts.types["geometry_msgs/msg/Vector3"]
    Quaternion = ts.types["geometry_msgs/msg/Quaternion"]

    root = Path(args.dataset)
    man = json.loads((root / "manifest.json").read_text())
    inv = json.loads((root / "invariants.json").read_text()) if (root / "invariants.json").exists() else {}
    if inv.get("verdict") != "PASS":
        raise SystemExit("the paired dataset has not passed check_invariants.py")
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists: refusing to overwrite")
    out.mkdir(parents=True)

    mount = args.mount_xyz
    stamps = {}
    src = man["source"]
    if src["kind"] == "sequence":
        smf = Path(src["path"]) / "manifest.json"
        if smf.exists():
            sm = json.loads(smf.read_text())
            mount = mount or sm.get("mount_xyz")
            stamps = {f"{f['frame']:08d}": float(f["sim_t"]) for f in sm.get("frames", []) if f.get("sim_t") is not None}
    mount = mount or [0.0, 0.0, 2.40]

    fields = [PointField(name="x", offset=0, datatype=7, count=1),
              PointField(name="y", offset=4, datatype=7, count=1),
              PointField(name="z", offset=8, datatype=7, count=1),
              PointField(name="intensity", offset=12, datatype=2, count=1),
              PointField(name="return_type", offset=13, datatype=2, count=1),
              PointField(name="channel", offset=14, datatype=4, count=1)]

    def stamp_of(k, name):
        t = stamps.get(name, k * args.period_s)
        sec = int(t); return Time(sec=sec, nanosec=int(round((t - sec) * 1e9)))

    frames = [f["frame"] for f in man["frames"]]
    report = {"source_dataset": str(root.resolve()), "topic": TOPIC, "frame_id": FRAME,
              "mount_xyz": mount, "layout": "PointXYZIRC 16 B", "intensity": "uint8(clip(I,0,1)*255)",
              "channel_from_acquisition": None, "modes": {}}
    for mode in man["settings"]["modes"]:
        bag = out / mode
        h = hashlib.sha256()
        with Writer(bag, version=8) as w:
            c_pc = w.add_connection(TOPIC, PointCloud2.__msgtype__, typestore=ts)
            c_tf = w.add_connection("/tf_static", TFMessage.__msgtype__, typestore=ts)
            st0 = stamp_of(0, frames[0])
            tf = TransformStamped(header=Header(stamp=st0, frame_id="base_link"), child_frame_id=FRAME,
                                  transform=Transform(translation=Vector3(x=mount[0], y=mount[1], z=mount[2]),
                                                      rotation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)))
            t0 = st0.sec * 10**9 + st0.nanosec
            w.write(c_tf, t0, ts.serialize_cdr(TFMessage(transforms=[tf]), TFMessage.__msgtype__))
            has_ch = 0
            for k, name in enumerate(frames):
                d = np.load(root / "modes" / mode / f"{name}.npz")
                lab = np.load(root / "labels" / f"{name}.npz")
                ch = lab["channel"] if "channel" in lab.files else None
                has_ch += ch is not None
                pts = encode(d["xyz"], d["intensity"], ch)
                data = np.frombuffer(pts.tobytes(), dtype=np.uint8)
                h.update(data.tobytes())
                st = stamp_of(k, name)
                msg = PointCloud2(header=Header(stamp=st, frame_id=FRAME), height=1, width=pts.size,
                                  fields=fields, is_bigendian=False, point_step=16, row_step=16 * pts.size,
                                  data=data, is_dense=True)
                w.write(c_pc, st.sec * 10**9 + st.nanosec, ts.serialize_cdr(msg, PointCloud2.__msgtype__))
        report["channel_from_acquisition"] = has_ch == len(frames)
        report["modes"][mode] = {"bag": str(bag), "messages": len(frames), "payload_sha256": h.hexdigest()}
        print(f"  {mode:15s} {len(frames)} clouds -> {bag}")
    (out / "bags_manifest.json").write_text(json.dumps(report, indent=1))
    if not report["channel_from_acquisition"]:
        print("  note: no ring index in the source; channel written as 0 (the CARLA kit does not use it)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
