"""Route files for PCLA.

`pcla_waypoint` is the format the June 2026 runs used successfully with PCLA:

    <route id="0" town="Town02">
      <waypoint x=".." y=".." z=".." yaw=".." pitch="0.0" roll="0.0"/>
    </route>

`leaderboard2` is the Leaderboard 2.0 format (<routes><route><waypoints><position/>),
in case the installed PCLA version expects it.
"""
from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path


def write_route(route: list, town: str, path: Path, fmt: str = "pcla_waypoint") -> str:
    if fmt == "pcla_waypoint":
        root = ET.Element("route", id="0", town=town)
        for x, y, z, yaw in route:
            ET.SubElement(root, "waypoint", x=f"{x:.4f}", y=f"{y:.4f}", z=f"{z:.4f}",
                          yaw=f"{yaw:.4f}", pitch="0.0", roll="0.0")
    elif fmt == "leaderboard2":
        root = ET.Element("routes")
        r = ET.SubElement(root, "route", id="0", town=town)
        wps = ET.SubElement(r, "waypoints")
        for x, y, z, _ in route:
            ET.SubElement(wps, "position", x=f"{x:.4f}", y=f"{y:.4f}", z=f"{z:.4f}")
        ET.SubElement(r, "scenarios")
    else:
        raise ValueError(f"unknown route format {fmt}")
    path = Path(path)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return hashlib.sha256(path.read_bytes()).hexdigest()
