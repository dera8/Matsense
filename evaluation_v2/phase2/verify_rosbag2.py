#!/usr/bin/env python3
"""Read the per-mode bags back and check that only intensity differs.

Checks (FAIL stops):
  layout      fields, offsets and datatypes of PointXYZIRC, point_step 16,
              little endian, as autoware::point_types::is_data_layout_compatible_with_point_xyzirc
  pairing     same message count and stamps in every mode
  geometry    x, y, z, return_type and channel bytes identical across modes
  encoding    intensity bytes equal uint8(clip(I,0,1)*255) of the paired dataset

Reported: how many returns change intensity byte with respect to RAW, and the
per-class contrast after the 8-bit quantization (median byte / reference class).

    python verify_rosbag2.py ../data/bags_town03_001 --dataset ../data/paired_town03_001
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from to_rosbag2 import POINT_DTYPE, TOPIC, encode  # noqa: E402

EXPECTED = [("x", 0, 7), ("y", 4, 7), ("z", 8, 7), ("intensity", 12, 2), ("return_type", 13, 2), ("channel", 14, 4)]


def read_bag(path: Path):
    from rosbags.highlevel import AnyReader
    out = []
    with AnyReader([path]) as r:
        conns = [c for c in r.connections if c.topic == TOPIC]
        for conn, t, raw in r.messages(connections=conns):
            out.append((t, r.deserialize(raw, conn.msgtype)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bags")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--reference-class", default="building")
    args = ap.parse_args()
    bags, ds = Path(args.bags), Path(args.dataset)
    bm = json.loads((bags / "bags_manifest.json").read_text())
    man = json.loads((ds / "manifest.json").read_text())
    modes = list(bm["modes"])
    frames = [f["frame"] for f in man["frames"]]
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase1"))
    from modes import load_profile  # noqa: E402
    classes = load_profile(man["profile"]["path"]).classes()

    fails = []
    data = {m: read_bag(Path(bm["modes"][m]["bag"])) for m in modes}
    for m in modes:
        msgs = data[m]
        if len(msgs) != len(frames):
            fails.append(f"{m}: {len(msgs)} messages, expected {len(frames)}")
        for _, msg in msgs[:1]:
            got = [(f.name, f.offset, f.datatype) for f in msg.fields]
            if got != EXPECTED or msg.point_step != 16 or msg.is_bigendian:
                fails.append(f"{m}: layout {got} step {msg.point_step}")
    stamps0 = [t for t, _ in data["RAW"]]
    changed, byte_contrast = {m: [] for m in modes}, {m: {c: [] for c in classes} for m in modes}
    for k, name in enumerate(frames):
        lab = np.load(ds / "labels" / f"{name}.npz")
        cls = lab["cls"]
        ch = lab["channel"] if "channel" in lab.files else None
        base = np.frombuffer(bytes(data["RAW"][k][1].data), dtype=POINT_DTYPE)
        for m in modes:
            t, msg = data[m][k]
            pts = np.frombuffer(bytes(msg.data), dtype=POINT_DTYPE)
            if t != stamps0[k]:
                fails.append(f"{m}/{name}: stamp differs")
            for fld in ("x", "y", "z", "return_type", "channel"):
                if not np.array_equal(pts[fld], base[fld]):
                    fails.append(f"{m}/{name}: field {fld} differs from RAW")
            d = np.load(ds / "modes" / m / f"{name}.npz")
            exp = encode(d["xyz"], d["intensity"], ch)
            if not np.array_equal(pts["intensity"], exp["intensity"]):
                fails.append(f"{m}/{name}: intensity bytes differ from the dataset encoding")
            changed[m].append(float(np.mean(pts["intensity"] != base["intensity"])))
            r = np.linalg.norm(d["xyz"].astype(np.float64), axis=1)
            band = (r >= 20) & (r < 60)
            ref_i = classes.index(args.reference_class)
            refb = pts["intensity"][(cls == ref_i) & band].astype(float)
            if refb.size >= 20 and np.median(refb) > 0:
                for i, c in enumerate(classes):
                    v = pts["intensity"][(cls == i) & band].astype(float)
                    if v.size >= 20:
                        byte_contrast[m][c].append(np.median(v) / np.median(refb))

    print(f"  {len(modes)} bags, {len(frames)} clouds each, topic {TOPIC}\n")
    print(f"  {'PASS' if not fails else 'FAIL'}  layout, pairing, geometry, encoding"
          + ("" if not fails else f"\n        {fails[:5]}"))
    print("\n  returns whose intensity byte differs from RAW (median over clouds):")
    for m in modes:
        print(f"    {m:15s} {100 * np.median(changed[m]):6.2f}%")
    print(f"\n  contrast after 8-bit quantization, median byte / {args.reference_class} (20-60 m):")
    print("    " + "".join(f"{c:>12s}" for c in classes))
    for m in modes:
        print(f"    {m:15s}" + "".join(f"{np.median(byte_contrast[m][c]) if byte_contrast[m][c] else float('nan'):12.3f}"
                                      for c in classes))
    (bags / "verify.json").write_text(json.dumps({"verdict": "PASS" if not fails else "FAIL", "fails": fails,
                                                  "changed_vs_raw": {m: float(np.median(v)) for m, v in changed.items()},
                                                  "byte_contrast": {m: {c: float(np.median(v)) if v else None
                                                                        for c, v in d.items()}
                                                                    for m, d in byte_contrast.items()}}, indent=1))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
