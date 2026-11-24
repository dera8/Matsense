#!/usr/bin/env python3
"""Add the MatSense hook to a PCLA checkout.

    python tools/patch_pcla.py C:/Users/debor/Desktop/carla/PCLA          # dry run
    python tools/patch_pcla.py C:/Users/debor/Desktop/carla/PCLA --apply

Finds every file defining `_parse_lidar_cb` (the LiDAR callback of the
Leaderboard sensor interface that PCLA ships), adds a module variable

    MATSENSE_PERTURB_FN = None

and, inside the callback, right after the decoded points are reshaped to (N, 4):

    if MATSENSE_PERTURB_FN is not None:
        points = MATSENSE_PERTURB_FN(points, frame=<data>.frame, tag=<tag>)

run_one.py sets the variable before PCLA spawns its sensors. When it is None,
PCLA behaves exactly as before. A copy of each original file is kept as .orig.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

MARK = "MATSENSE_PERTURB_FN"


def patch_text(src: str) -> tuple[str, str]:
    if MARK in src:
        return src, "already patched"
    m = re.search(r"^([ \t]*)def _parse_lidar_cb\(\s*self\s*,\s*(\w+)\s*,\s*(\w+)\s*\)\s*:", src, re.M)
    if not m:
        return src, "no _parse_lidar_cb"
    indent, data_name, tag_name = m.group(1), m.group(2), m.group(3)
    body_start = m.end()
    nxt = re.search(r"^" + re.escape(indent) + r"(def |class )", src[body_start:], re.M)
    body_end = body_start + (nxt.start() if nxt else len(src) - body_start)
    body = src[body_start:body_end]
    rm = None
    for rm in re.finditer(r"^([ \t]+)(\w+)\s*=\s*np\.reshape\(\s*\2\b.*$", body, re.M):
        break
    if rm is None:
        return src, "no 'points = np.reshape(points, ...)' line in _parse_lidar_cb"
    bind, var = rm.group(1), rm.group(2)
    insert = (f"\n{bind}if {MARK} is not None:\n"
              f"{bind}    {var} = {MARK}({var}, frame={data_name}.frame, tag={tag_name})")
    body = body[:rm.end()] + insert + body[rm.end():]
    src = src[:body_start] + body + src[body_end:]
    cls = re.search(r"^class CallBack\b", src, re.M)
    pos = cls.start() if cls else 0
    src = src[:pos] + f"{MARK} = None  # set by the MatSense campaign (run_one.py)\n\n\n" + src[pos:]
    return src, f"patched (points variable '{var}', data '{data_name}', tag '{tag_name}')"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pcla_dir")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    root = Path(a.pcla_dir)
    if not root.exists():
        print(f"{root} does not exist")
        return 1
    files = [p for p in root.rglob("*.py") if "_parse_lidar_cb" in p.read_text(encoding="utf-8", errors="ignore")]
    if not files:
        print("no file defines _parse_lidar_cb: is this a PCLA checkout?")
        return 1
    for p in files:
        src = p.read_text(encoding="utf-8")
        if "perturb_fn" in src:
            print(f"note: {p} already contains an older 'perturb_fn' fork; it is left in place")
        new, msg = patch_text(src)
        print(f"{p}: {msg}")
        if a.apply and new != src:
            shutil.copy2(p, p.with_suffix(p.suffix + ".orig"))
            p.write_text(new, encoding="utf-8")
    if not a.apply:
        print("dry run: add --apply to write the changes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
