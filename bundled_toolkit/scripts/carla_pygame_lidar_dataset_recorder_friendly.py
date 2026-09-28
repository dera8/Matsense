#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CARLA + pygame LiDAR viewer with intensity / pseudo-reflectance coloring.

What it does
------------
- Connects to a running CARLA server.
- Spawns an ego vehicle.
- Attaches:
  * RGB camera (left panel)
  * Semantic segmentation camera (hidden, used for optional class fusion)
  * Ray-cast LiDAR (point cloud source)
- Renders a top-down LiDAR BEV in pygame (right panel).
- Colors LiDAR points by either:
  * raw intensity
  * pseudo-reflectance = intensity * distance^2 correction * semantic/material prior

Keys
----
P : toggle color mode (intensity / global)
R : respawn ego vehicle
WASD or arrows : drive manually
Space : hand brake
Q : reverse gear
ESC : quit

Notes
-----
- This is designed as a practical starting point. The pseudo-reflectance model is
  intentionally simple and explicit, so it can be replaced with your own material-
  aware model calibrated from real LiDAR data.
- Semantic fusion uses a forward-facing semantic camera. If a point is outside its
  FOV, the script falls back to a default material coefficient.
- Compatible with CARLA 0.9.14+ / 0.9.16 style APIs.
"""

import argparse
import csv
import hashlib
import os
from collections import OrderedDict, deque
import json
import math
import sys
import traceback
import threading
import time
import weakref
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pygame
from material_aware_toolkit.material_aware_tool_config import (
    DEFAULT_CONFIG_PATH as DEFAULT_TOOL_CONFIG_PATH,
    get_profile,
    load_tool_config,
    normalize_profile,
)

try:
    import carla
except ImportError as e:
    raise SystemExit(
        "Cannot import carla. Run this from a Python environment that has the CARLA egg/wheel installed."
    ) from e


# -----------------------------
# Material-aware profile
# -----------------------------
_DEFAULT_TOOL_CONFIG, _DEFAULT_TOOL_CONFIG_PATH = load_tool_config(DEFAULT_TOOL_CONFIG_PATH)
_DEFAULT_PROFILE_NAME, _DEFAULT_PROFILE = get_profile(_DEFAULT_TOOL_CONFIG)
_DEFAULT_PROFILE = normalize_profile(_DEFAULT_PROFILE)

SEMANTIC_TO_MATERIAL = dict(_DEFAULT_PROFILE["semantic_to_material"])
DEFAULT_MATERIAL = _DEFAULT_PROFILE["default_material"]
DISPLAY_MODE_SEQUENCE = list(_DEFAULT_PROFILE["display_mode_sequence"])
DISPLAY_MODE_SEQUENCE = [mode for mode in DISPLAY_MODE_SEQUENCE if mode != "material_effect"]
CAMERA_TRIPLE_MODE = "camera_triple"

BG_COLOR = (17, 20, 26)
CARD_BG = (27, 33, 43)
CARD_BORDER = (62, 77, 96)
TEXT_MAIN = (238, 242, 247)
TEXT_MUTED = (167, 180, 196)
ACCENT = (84, 161, 255)
SUCCESS = (77, 201, 136)
WARNING = (255, 196, 84)

MATERIAL_DISPLAY_COLORS = {
    name: np.array(color, dtype=np.uint8)
    for name, color in _DEFAULT_PROFILE["display_colors"].items()
}

NOMINAL_BASE = dict(_DEFAULT_PROFILE["nominal_base"])
WEATHER_RATIO = dict(_DEFAULT_PROFILE["weather_ratio"])
PLANAR_MATERIALS = set(_DEFAULT_PROFILE["planar_materials"])
MATERIAL_PROFILE_NAME = _DEFAULT_PROFILE_NAME
MATERIAL_CONFIG_PATH = _DEFAULT_TOOL_CONFIG_PATH
DEFAULT_DISPLAY_VMAX = 2.0
REAR_OFFSET = -1.393

FIT_RANGE_MIN = 2.0
FIT_RANGE_MAX = 30.0
REF_RANGE_R0 = 10.0
COS_EPS = 0.2
A2 = 0.01592119599839734
A1 = -0.6848984378845165
A0 = 12.437483628508096
G_R0 = 7.180618849502665


def apply_material_profile(profile_name: str, profile: dict, config_path: Path) -> None:
    global SEMANTIC_TO_MATERIAL
    global DEFAULT_MATERIAL
    global DISPLAY_MODE_SEQUENCE
    global MATERIAL_DISPLAY_COLORS
    global NOMINAL_BASE
    global WEATHER_RATIO
    global PLANAR_MATERIALS
    global MATERIAL_PROFILE_NAME
    global MATERIAL_CONFIG_PATH

    normalized = normalize_profile(profile)
    SEMANTIC_TO_MATERIAL = dict(normalized["semantic_to_material"])
    DEFAULT_MATERIAL = normalized["default_material"]
    DISPLAY_MODE_SEQUENCE = list(normalized["display_mode_sequence"])
    MATERIAL_DISPLAY_COLORS = {
        name: np.array(color, dtype=np.uint8)
        for name, color in normalized["display_colors"].items()
    }
    NOMINAL_BASE = dict(normalized["nominal_base"])
    WEATHER_RATIO = dict(normalized["weather_ratio"])
    PLANAR_MATERIALS = set(normalized["planar_materials"])
    MATERIAL_PROFILE_NAME = profile_name
    MATERIAL_CONFIG_PATH = Path(config_path)


def clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def g_of_r(r: np.ndarray) -> np.ndarray:
    return A2 * (r ** 2) + A1 * r + A0


def empirical_range_corrected(intensity: np.ndarray, r: np.ndarray) -> np.ndarray:
    r = np.clip(r, FIT_RANGE_MIN, FIT_RANGE_MAX)
    gr = np.maximum(g_of_r(r), 1e-6)
    return intensity * (G_R0 / gr)


def angle_corrected(values: np.ndarray, cos_theta: np.ndarray | None) -> np.ndarray:
    if cos_theta is None:
        return values
    return values / np.maximum(np.abs(cos_theta), COS_EPS)


def compress_for_display(values: np.ndarray, vmax: float = DEFAULT_DISPLAY_VMAX) -> np.ndarray:
    x = np.log1p(np.maximum(values, 0.0))
    xmax = math.log1p(max(vmax, 1e-6))
    return clamp01(x / xmax)


def compress_for_display_percentile(values: np.ndarray, percentile: float = 99.0) -> tuple[np.ndarray, float]:
    safe = np.maximum(values, 0.0)
    if safe.size == 0:
        return safe.astype(np.float32), 1.0
    vmax = float(np.percentile(safe, percentile))
    vmax = max(vmax, 1e-6)
    return compress_for_display(safe, vmax=vmax), vmax


def colormap_palette(v: np.ndarray, stops: list[tuple[float, tuple[int, int, int]]]) -> np.ndarray:
    """Piecewise-linear RGB colormap."""
    v = clamp01(v)
    out = np.zeros((v.size, 3), dtype=np.float32)
    for idx, value in enumerate(v):
        for stop_idx in range(len(stops) - 1):
            left_pos, left_color = stops[stop_idx]
            right_pos, right_color = stops[stop_idx + 1]
            if value <= right_pos or stop_idx == len(stops) - 2:
                span = max(right_pos - left_pos, 1e-6)
                t = float(clamp01(np.asarray([(value - left_pos) / span]))[0])
                a = np.asarray(left_color, dtype=np.float32)
                b = np.asarray(right_color, dtype=np.float32)
                out[idx] = a * (1.0 - t) + b * t
                break
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def colormap_intensity(v: np.ndarray) -> np.ndarray:
    return colormap_palette(
        v,
        [
            (0.0, (0, 0, 0)),
            (0.50, (0, 220, 255)),
            (1.0, (255, 255, 255)),
        ],
    )


def colormap_pseudo(v: np.ndarray) -> np.ndarray:
    return colormap_palette(
        v,
        [
            (0.0, (7, 59, 76)),
            (0.25, (17, 138, 178)),
            (0.50, (6, 214, 160)),
            (0.75, (255, 209, 102)),
            (1.0, (239, 71, 111)),
        ],
    )


def material_to_rgb(materials: np.ndarray) -> np.ndarray:
    return np.stack(
        [MATERIAL_DISPLAY_COLORS.get(str(m), MATERIAL_DISPLAY_COLORS["unknown"]) for m in materials],
        axis=0,
    )


def visual_material_name(name: str) -> str:
    return str(name)


def known_material_mask(materials: np.ndarray) -> np.ndarray:
    return np.asarray([visual_material_name(m).lower() != "unknown" for m in materials], dtype=bool)


def mode_title(mode: str) -> str:
    return {
        "intensity": "Raw Intensity",
        "global": "Global Scaling",
        "material": "Material Classes",
        CAMERA_TRIPLE_MODE: "RGB + LiDAR Overlays",
    }.get(mode, mode.title())


def mode_description(mode: str) -> str:
    return {
        "intensity": "Direct LiDAR return strength from the sensor",
        "global": "Uniform 20% point dropout applied globally (weather-independent baseline)",
        "material": "Semantic-material overlay for asphalt, sidewalk, building, vegetation and car",
        CAMERA_TRIPLE_MODE: "Same CARLA camera view with material, raw intensity and pseudo-reflectance overlays",
    }.get(mode, mode)


def material_response_values(materials: np.ndarray, weather_name: str) -> np.ndarray:
    weather_lookup = WEATHER_RATIO.get(weather_name, WEATHER_RATIO["nominal"])
    vals = []
    for material in materials:
        m = str(material)
        base = NOMINAL_BASE.get(m, NOMINAL_BASE[DEFAULT_MATERIAL])
        ratio = weather_lookup.get(m, weather_lookup.get(DEFAULT_MATERIAL, 1.0))
        vals.append(base * ratio)
    return np.asarray(vals, dtype=np.float32)


def nominal_material_values(materials: np.ndarray) -> np.ndarray:
    return np.asarray(
        [NOMINAL_BASE.get(str(material), NOMINAL_BASE[DEFAULT_MATERIAL]) for material in materials],
        dtype=np.float32,
    )


def material_effect_display_vmax() -> float:
    vmax = 0.0
    weather_names = list(WEATHER_RATIO.keys()) or ["nominal"]
    for weather_name in weather_names:
        weather_lookup = WEATHER_RATIO.get(weather_name, {})
        material_names = set(NOMINAL_BASE.keys()) | set(weather_lookup.keys())
        for material_name in material_names:
            base = NOMINAL_BASE.get(material_name, NOMINAL_BASE[DEFAULT_MATERIAL])
            ratio = weather_lookup.get(material_name, weather_lookup.get(DEFAULT_MATERIAL, 1.0))
            vmax = max(vmax, float(base * ratio))
    return max(vmax, 1e-6)


def load_trajectory_json(json_path: str) -> list[dict]:
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    poses: list[dict] = []
    for item in data:
        loc = item["transform"]["location"]
        rot = item["transform"]["rotation"]
        poses.append(
            {
                "frame_id": item.get("frame_id"),
                "timestamp": float(item.get("timestamp", 0.0)),
                "x": float(loc["x"]),
                "y": float(loc["y"]),
                "z": float(loc.get("z", 0.0)),
                "yaw": float(rot.get("yaw", 0.0)),
            }
        )
    return poses


def utm_to_carla_xy(utm_e: float, utm_n: float, offset_x: float, offset_y: float) -> tuple[float, float]:
    local_x = utm_e - offset_x
    local_y = utm_n - offset_y
    return local_x, -local_y


def utm_yaw_to_carla_yaw(utm_yaw_rad: float) -> float:
    return 90.0 - math.degrees(utm_yaw_rad)


def load_trajectory_txt(txt_path: str, offset_x: float, offset_y: float, step: int = 1) -> list[dict]:
    poses: list[dict] = []
    with Path(txt_path).open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 5:
                continue
            frame_id = int(parts[0])
            timestamp = float(parts[1])
            utm_e = float(parts[2])
            utm_n = float(parts[3])
            utm_yaw = float(parts[4])
            x, y = utm_to_carla_xy(utm_e, utm_n, offset_x, offset_y)
            poses.append(
                {
                    "frame_id": frame_id,
                    "timestamp": timestamp,
                    "x": x,
                    "y": y,
                    "z": 0.0,
                    "yaw": utm_yaw_to_carla_yaw(utm_yaw),
                }
            )
    return poses[::max(1, step)]


def transform_trajectory_poses(
    poses: list[dict],
    shift_x: float = 0.0,
    shift_y: float = 0.0,
    shift_z: float = 0.0,
    scale: float = 1.0,
) -> list[dict]:
    if not poses:
        return poses
    transformed: list[dict] = []
    for pose in poses:
        updated = dict(pose)
        updated["x"] = float(pose["x"]) * scale + shift_x
        updated["y"] = float(pose["y"]) * scale + shift_y
        updated["z"] = float(pose["z"]) * scale + shift_z
        transformed.append(updated)
    return transformed


def pose_to_transform(pose: dict, z_offset: float = 0.0) -> "carla.Transform":
    return carla.Transform(
        carla.Location(x=pose["x"], y=pose["y"], z=pose["z"] + z_offset),
        carla.Rotation(yaw=pose["yaw"]),
    )


def pose_to_road_transform(world: "carla.World", pose: dict, z_offset: float = 0.0) -> "carla.Transform":
    road_map = world.get_map()
    query_loc = carla.Location(x=float(pose["x"]), y=float(pose["y"]), z=float(pose.get("z", 0.0)))
    snapped_wp = road_map.get_waypoint(
        query_loc,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if snapped_wp is None:
        return pose_to_transform(pose, z_offset=z_offset)
    return carla.Transform(
        carla.Location(
            x=float(pose["x"]),
            y=float(pose["y"]),
            z=float(snapped_wp.transform.location.z) + z_offset,
        ),
        carla.Rotation(yaw=float(pose["yaw"])),
    )


def distance_to_nearest_driving_waypoint(road_map: "carla.Map", x: float, y: float, z: float = 0.0) -> float:
    query_loc = carla.Location(x=float(x), y=float(y), z=float(z))
    wp = road_map.get_waypoint(
        query_loc,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if wp is None:
        return float("inf")
    road_loc = wp.transform.location
    return math.hypot(query_loc.x - road_loc.x, query_loc.y - road_loc.y)


def sample_alignment_distances(
    road_map: "carla.Map",
    points: list[tuple[float, float, float]],
    max_samples: int = 48,
) -> np.ndarray:
    if not points:
        return np.array([], dtype=np.float32)
    step = max(1, len(points) // max_samples)
    sampled = points[::step][:max_samples]
    distances = [
        distance_to_nearest_driving_waypoint(road_map, x, y, z)
        for x, y, z in sampled
    ]
    return np.asarray(distances, dtype=np.float32)


def parked_transform_from_entry(world: "carla.World", item: dict, z_offset: float) -> "carla.Transform":
    """Where a parked vehicle stands, taking the road's own attitude if given.

    Yaw alone is right only on flat ground. Parked vehicles have their physics
    switched off immediately after spawning, so nothing settles them onto a
    slope afterwards: on the parking lane beside route 05, which climbs from
    2.5 m to 8.0 m, a car placed with pitch zero sits with one end buried and
    the other in the air and stays that way for the whole run. Pitch and roll
    are optional and default to zero, so every scenario written before this
    keeps the transform it had.
    """
    raw = item["start"]
    return carla.Transform(
        carla.Location(x=float(raw[0]), y=float(raw[1]), z=float(raw[2]) + z_offset),
        carla.Rotation(yaw=float(item.get("heading", 0.0)),
                       pitch=float(item.get("pitch", 0.0)),
                       roll=float(item.get("roll", 0.0))),
    )


def parked_color_tuple(item: dict) -> tuple[int, int, int]:
    color_value = item.get("color")
    if not color_value:
        return 128, 128, 128
    try:
        r, g, b = [int(v) for v in str(color_value).split(",")]
    except Exception:
        return 128, 128, 128
    return r, g, b


def apply_blueprint_attributes(bp, item: dict):
    attrs = item.get("attributes") or {}
    for key, value in attrs.items():
        if not bp.has_attribute(str(key)):
            continue
        try:
            bp.set_attribute(str(key), str(value))
        except Exception:
            pass


def pick_parked_blueprint(blueprints, item: dict, library=None):
    """Resolve a spec's blueprint.

    An explicit blueprint_id is looked up in the FULL library, not in the
    curated list: get_filtered_vehicle_blueprints() exists to keep RANDOM picks
    sane (no police cars, no two-wheelers whose physics the follow controller
    cannot drive), and applying it to an explicit choice silently substitutes a
    different vehicle. That is not hypothetical - scenario 18 asked for
    vehicle.diamondback.century and vehicle.mercedes.sprinter, both of which the
    filter drops ("diamondback", "century", 2 wheels; "sprinter"), so the cyclist
    spawned as a random saloon that could not follow a bicycle's arc out of a
    parking bay and drove into a lamp post, and the van meant to occlude it was
    another random saloon. Nothing warned.
    """
    explicit_id = str(item.get("blueprint_id", "")).strip()
    if explicit_id:
        pool = list(blueprints)
        if library is not None:
            try:
                pool = list(library.filter("*")) or pool
            except Exception:
                pass
        for candidate in pool:
            if candidate.id == explicit_id:
                bp = candidate
                apply_blueprint_attributes(bp, item)
                return bp
        print(f"[toolkit] blueprint richiesto non trovato: {explicit_id} "
              f"- uso un ripiego casuale", flush=True)
    r, g, b = parked_color_tuple(item)
    idx = (r + g * 3 + b * 7) % len(blueprints)
    bp = blueprints[idx]
    if bp.has_attribute("color"):
        try:
            bp.set_attribute("color", f"{r},{g},{b}")
        except Exception:
            pass
    apply_blueprint_attributes(bp, item)
    return bp


def parse_vehicle_light_state(item: dict):
    raw = item.get("vehicle_light_state") or item.get("light_state")
    if not raw:
        return None
    if isinstance(raw, str):
        parts = [p.strip() for p in raw.split("|") if p.strip()]
    elif isinstance(raw, list):
        parts = [str(p).strip() for p in raw if str(p).strip()]
    else:
        return None
    if not parts:
        return None
    mapping = {
        "position": carla.VehicleLightState.Position,
        "low_beam": carla.VehicleLightState.LowBeam,
        "high_beam": carla.VehicleLightState.HighBeam,
        "brake": carla.VehicleLightState.Brake,
        "right_blinker": carla.VehicleLightState.RightBlinker,
        "left_blinker": carla.VehicleLightState.LeftBlinker,
        "reverse": carla.VehicleLightState.Reverse,
        "fog": carla.VehicleLightState.Fog,
        "interior": carla.VehicleLightState.Interior,
        "special1": carla.VehicleLightState.Special1,
        "special2": carla.VehicleLightState.Special2,
        "all": carla.VehicleLightState.All,
    }
    state = carla.VehicleLightState.NONE
    for part in parts:
        bit = mapping.get(part.lower())
        if bit is not None:
            state |= bit
    if state == carla.VehicleLightState.NONE:
        return None
    return state


def transform_parked_spawn_positions(
    positions: list[dict],
    shift_x: float = 0.0,
    shift_y: float = 0.0,
    shift_z: float = 0.0,
    scale: float = 1.0,
) -> list[dict]:
    if not positions:
        return positions
    transformed: list[dict] = []
    for item in positions:
        updated = dict(item)
        for key in ("start", "end"):
            if key in updated and updated[key]:
                coords = list(updated[key])
                while len(coords) < 3:
                    coords.append(0.0)
                coords[0] = float(coords[0]) * scale + shift_x
                coords[1] = float(coords[1]) * scale + shift_y
                coords[2] = float(coords[2]) * scale + shift_z
                updated[key] = coords
        transformed.append(updated)
    return transformed


def validate_points_against_map(road_map, points, label, max_median_m,
                                mode="abort"):
    """Refuse to place anything on a map it does not belong to.

    A trajectory and a parked layout are both just numbers, and numbers are
    right only relative to a convention. The one that bites here is the sign of
    y: CARLA's y runs opposite to UTM northing, so a file stored before the flip
    describes the mirror image of the street. Every internal check on such a
    file passes, because it is self-consistent; it is only against the map that
    it is eighty metres out, and the symptom is a vehicle placed where no
    geometry was ever built, falling forever.

    So the map is asked. If the points sit far from every driving lane, the run
    stops and says so, and it also reports whether flipping y would fix it,
    because that names the cause instead of leaving a distance to interpret.
    """
    if not points:
        return True
    def median_distance(pts):
        out = []
        for x, y in pts:
            wp = road_map.get_waypoint(carla.Location(x=float(x), y=float(y), z=0.0),
                                       project_to_road=True,
                                       lane_type=carla.LaneType.Driving)
            if wp is not None:
                out.append(math.hypot(wp.transform.location.x - float(x),
                                      wp.transform.location.y - float(y)))
        if not out:
            return float("inf")
        out.sort()
        return out[len(out) // 2]

    sample = points[::max(1, len(points) // 200)]
    d = median_distance(sample)
    if d <= max_median_m:
        print(f"[toolkit] {label}: {d:.2f} m mediani dalla corsia piu' vicina, "
              "coerente con questa mappa", flush=True)
        return True

    flipped = median_distance([(x, -y) for x, y in sample])
    msg = (f"{label} NON appartiene a questa mappa: {d:.1f} m mediani dalla "
           f"corsia piu' vicina, contro un limite di {max_median_m:.1f} m.")
    if flipped <= max_median_m:
        msg += (f" Invertendo il segno di y lo scarto scende a {flipped:.2f} m: "
                "il file e' salvato prima del ribaltamento che CARLA richiede, "
                "e va convertito invece che caricato cosi'.")
    else:
        msg += (" Nemmeno invertendo y i punti cadono sulla strada: probabile "
                "offset UTM sbagliato, o file costruito per un'altra mappa.")
    if mode == "abort":
        raise SystemExit(f"[toolkit] {msg}")
    print(f"[toolkit] ATTENZIONE: {msg}", flush=True)
    return False


def lateral_offset_from_trajectory(
    x: float, y: float, poses: list[dict]
) -> tuple[float, float, int]:
    """Signed distance from the driven line, measured across its local tangent.

    Distance to the nearest recorded pose is not the quantity that decides
    whether a parked car blocks the ego: between two samples the line runs
    straight, and a car beside the midpoint of a long segment is nearer the
    segment than it is to either endpoint. The clearance a vehicle actually has
    is its distance to the segment, taken across that segment's own direction,
    and the sign of that distance says which side it stands on.

    Returns the signed lateral offset, the distance travelled along the line to
    the foot of the perpendicular, and the index of the segment used. Positive
    is left of the direction of travel.
    """
    if len(poses) < 2:
        return 0.0, 0.0, 0
    best = (float("inf"), 0.0, 0.0, 0)
    travelled = 0.0
    for i in range(len(poses) - 1):
        ax, ay = poses[i]["x"], poses[i]["y"]
        bx, by = poses[i + 1]["x"], poses[i + 1]["y"]
        dx, dy = bx - ax, by - ay
        seg = math.hypot(dx, dy)
        if seg < 1e-9:
            continue
        ux, uy = dx / seg, dy / seg
        t = max(0.0, min(seg, (x - ax) * ux + (y - ay) * uy))
        fx, fy = ax + ux * t, ay + uy * t
        d = math.hypot(x - fx, y - fy)
        if d < best[0]:
            # Left normal of the direction of travel, so the sign is the side.
            lateral = (x - fx) * (-uy) + (y - fy) * ux
            best = (d, lateral, travelled + t, i)
        travelled += seg
    return best[1], best[2], best[3]


def resolve_parked_ego_lane_conflicts(
    positions: list[dict],
    poses: list[dict],
    clearance_m: float,
) -> tuple[list[dict], list[dict]]:
    """Push aside only the parked vehicles that stand in the ego's way.

    A layout recovered from a recording carries the recording's own localization
    error, and on a reconstruction without parking lanes some vehicles land
    inside the single drivable lane. Discarding them loses scene content that is
    real; moving every vehicle to a common offset discards the measurement. So
    only those below the clearance are moved, and only far enough to reach it,
    along the local normal and away from the line, keeping the side they were
    found on.

    The input list is never mutated: the caller's JSON stays the record of what
    was recovered, and what was driven is written to the validation report.
    """
    if not positions or len(poses) < 2 or clearance_m <= 0.0:
        return [dict(p) for p in positions], []
    out: list[dict] = []
    corrections: list[dict] = []
    for item in positions:
        updated = dict(item)
        start = list(updated.get("start") or [0.0, 0.0, 0.0])
        while len(start) < 3:
            start.append(0.0)
        x, y = float(start[0]), float(start[1])
        lateral, along, seg = lateral_offset_from_trajectory(x, y, poses)
        if abs(lateral) >= clearance_m:
            updated["start"] = start
            out.append(updated)
            continue
        ax, ay = poses[seg]["x"], poses[seg]["y"]
        bx, by = poses[seg + 1]["x"], poses[seg + 1]["y"]
        seg_len = math.hypot(bx - ax, by - ay) or 1.0
        ux, uy = (bx - ax) / seg_len, (by - ay) / seg_len
        nx, ny = -uy, ux
        # A vehicle sitting exactly on the line has no recovered side; push it
        # right, the side a parked car takes where traffic drives on the right.
        side = 1.0 if lateral > 0 else (-1.0 if lateral < 0 else -1.0)
        shift = clearance_m - abs(lateral)
        start[0] = x + side * nx * shift
        start[1] = y + side * ny * shift
        updated["start"] = start
        out.append(updated)
        corrections.append({
            "from": [round(x, 3), round(y, 3)],
            "to": [round(start[0], 3), round(start[1], 3)],
            "lateral_before_m": round(lateral, 3),
            "lateral_after_m": round(side * clearance_m, 3),
            "shift_m": round(shift, 3),
            "along_m": round(along, 1),
        })
    return out, corrections


def normalize_dynamic_actor_specs(
    specs: list[dict],
    shift_x: float = 0.0,
    shift_y: float = 0.0,
    shift_z: float = 0.0,
    scale: float = 1.0,
) -> list[dict]:
    if not specs:
        return []
    out: list[dict] = []
    for item in specs:
        updated = dict(item)
        for key in ("start", "trigger_point"):
            if key in updated and updated[key]:
                coords = list(updated[key])
                while len(coords) < 3:
                    coords.append(0.0)
                coords[0] = float(coords[0]) * scale + shift_x
                coords[1] = float(coords[1]) * scale + shift_y
                coords[2] = float(coords[2]) * scale + shift_z
                updated[key] = coords
        out.append(updated)
    return out


def load_material_overrides(path: str | None) -> dict:
    if not path:
        return {"actor_ids": {}, "type_ids": {}}
    p = Path(path)
    if not p.exists():
        return {"actor_ids": {}, "type_ids": {}}
    data = json.loads(p.read_text(encoding="utf-8"))
    return {
        "actor_ids": {str(k): str(v) for k, v in data.get("actor_ids", {}).items()},
        "type_ids": {str(k): str(v) for k, v in data.get("type_ids", {}).items()},
    }


def _angles_from_xyz(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xy = np.hypot(xyz[:, 0], xyz[:, 1])
    az = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))
    el = np.degrees(np.arctan2(xyz[:, 2], np.maximum(xy, 1e-6)))
    return az.astype(np.float32), el.astype(np.float32)


def match_semantic_lidar_metadata(
    lidar_xyz: np.ndarray,
    semantic_lidar: np.ndarray | None,
    az_bin_deg: float = 0.25,
    el_bin_deg: float = 0.5,
    max_xyz_dist: float = 0.35,
) -> tuple[np.ndarray, np.ndarray]:
    matched_obj_ids = np.zeros((len(lidar_xyz),), dtype=np.uint32)
    matched_obj_tags = np.zeros((len(lidar_xyz),), dtype=np.int32)
    if semantic_lidar is None or len(semantic_lidar) == 0 or len(lidar_xyz) == 0:
        return matched_obj_ids, matched_obj_tags

    sem_xyz = np.column_stack([semantic_lidar["x"], semantic_lidar["y"], semantic_lidar["z"]]).astype(np.float32)
    sem_az, sem_el = _angles_from_xyz(sem_xyz)
    sem_bins: dict[tuple[int, int], list[int]] = {}
    for idx, (az, el) in enumerate(zip(sem_az, sem_el)):
        key = (int(round(float(az) / az_bin_deg)), int(round(float(el) / el_bin_deg)))
        sem_bins.setdefault(key, []).append(idx)

    lidar_az, lidar_el = _angles_from_xyz(lidar_xyz.astype(np.float32))
    neighbor_offsets = (-1, 0, 1)
    max_dist2 = max_xyz_dist * max_xyz_dist
    for i, (az, el, pt) in enumerate(zip(lidar_az, lidar_el, lidar_xyz)):
        base_key = (int(round(float(az) / az_bin_deg)), int(round(float(el) / el_bin_deg)))
        candidates: list[int] = []
        for da in neighbor_offsets:
            for de in neighbor_offsets:
                candidates.extend(sem_bins.get((base_key[0] + da, base_key[1] + de), []))
        if not candidates:
            continue
        cand_xyz = sem_xyz[candidates]
        d2 = np.sum((cand_xyz - pt.astype(np.float32)) ** 2, axis=1)
        best_rel = int(np.argmin(d2))
        if float(d2[best_rel]) > max_dist2:
            continue
        best_idx = candidates[best_rel]
        matched_obj_ids[i] = np.uint32(semantic_lidar["ObjIdx"][best_idx])
        matched_obj_tags[i] = int(semantic_lidar["ObjTag"][best_idx])
    return matched_obj_ids, matched_obj_tags


def get_filtered_vehicle_blueprints(world: "carla.World"):
    blueprint_library = world.get_blueprint_library()
    all_vehicles = sorted(blueprint_library.filter("vehicle.*"), key=lambda bp: bp.id)
    forbidden_keywords = [
        "mustang", "police", "impala", "carlacola", "cybertruck", "t2", "sprinter",
        "firetruck", "ambulance", "bus", "truck", "van", "bingle", "microlino",
        "vespa", "yamaha", "kawasaki", "harley", "bh", "gazelle", "diamondback",
        "crossbike", "century", "omafiets", "low_rider", "ninja", "zx125", "yzf",
        "fuso", "rosa", "isetta",
    ]
    vehicles = []
    for bp in all_vehicles:
        if bp.has_attribute("number_of_wheels") and int(bp.get_attribute("number_of_wheels")) != 4:
            continue
        if any(keyword in bp.id.lower() for keyword in forbidden_keywords):
            continue
        vehicles.append(bp)
    return vehicles


def vehicle_speed(vehicle: "carla.Vehicle") -> float:
    vel = vehicle.get_velocity()
    return math.sqrt(vel.x * vel.x + vel.y * vel.y + vel.z * vel.z)


def spectator_transform_from_vehicle(vehicle_tf: "carla.Transform", mode: str) -> "carla.Transform":
    loc = vehicle_tf.location
    rot = vehicle_tf.rotation
    forward = vehicle_tf.get_forward_vector()
    right = vehicle_tf.get_right_vector()

    if mode == "hood":
        cam_loc = carla.Location(
            x=loc.x + 1.6 * forward.x,
            y=loc.y + 1.6 * forward.y,
            z=loc.z + 1.6,
        )
        cam_rot = carla.Rotation(pitch=-4.0, yaw=rot.yaw, roll=0.0)
        return carla.Transform(cam_loc, cam_rot)

    if mode == "roof":
        cam_loc = carla.Location(
            x=loc.x + 0.2 * forward.x,
            y=loc.y + 0.2 * forward.y,
            z=loc.z + 2.8,
        )
        cam_rot = carla.Rotation(pitch=-12.0, yaw=rot.yaw, roll=0.0)
        return carla.Transform(cam_loc, cam_rot)

    cam_loc = carla.Location(
        x=loc.x - 6.0 * forward.x + 0.8 * right.x,
        y=loc.y - 6.0 * forward.y + 0.8 * right.y,
        z=loc.z + 2.8,
    )
    cam_rot = carla.Rotation(pitch=-14.0, yaw=rot.yaw, roll=0.0)
    return carla.Transform(cam_loc, cam_rot)


def nearest_index(poses: list[dict], vehicle_loc: "carla.Location", start_idx: int, search_ahead: int) -> int:
    best_idx = start_idx
    best_dist = float("inf")
    end_idx = min(len(poses), start_idx + max(2, search_ahead))
    for i in range(start_idx, end_idx):
        dx = poses[i]["x"] - vehicle_loc.x
        dy = poses[i]["y"] - vehicle_loc.y
        d = dx * dx + dy * dy
        if d < best_dist:
            best_dist = d
            best_idx = i
    return best_idx


def lookahead_index(current_idx: int, poses: list[dict], speed_mps: float) -> int:
    lookahead = 6
    if speed_mps > 8.0:
        lookahead = 12
    elif speed_mps > 4.0:
        lookahead = 8
    return min(len(poses) - 1, current_idx + lookahead)


def compute_target_speed(poses: list[dict], idx: int, fallback_fps: float) -> float:
    if idx >= len(poses) - 1:
        return 0.0
    a = poses[idx]
    b = poses[idx + 1]
    dist = math.hypot(b["x"] - a["x"], b["y"] - a["y"])
    dt = b["timestamp"] - a["timestamp"]
    if dt <= 1e-6:
        dt = 1.0 / fallback_fps
    return dist / dt


def smooth_target_speed(poses: list[dict], idx: int, fallback_fps: float, window: int = 8) -> float:
    speeds = []
    end = min(len(poses) - 1, idx + max(1, window))
    for i in range(idx, end):
        speeds.append(compute_target_speed(poses, i, fallback_fps))
    return sum(speeds) / len(speeds) if speeds else 0.0


def blend_control(previous: "carla.VehicleControl | None", current: "carla.VehicleControl", alpha: float) -> "carla.VehicleControl":
    if previous is None:
        return current
    out = carla.VehicleControl()
    out.throttle = previous.throttle * (1.0 - alpha) + current.throttle * alpha
    out.brake = previous.brake * (1.0 - alpha) + current.brake * alpha
    out.steer = previous.steer * (1.0 - alpha) + current.steer * alpha
    out.hand_brake = current.hand_brake
    out.reverse = current.reverse
    out.manual_gear_shift = current.manual_gear_shift
    return out


def compute_follow_control(vehicle: "carla.Vehicle", target_pose: dict, target_speed_mps: float) -> "carla.VehicleControl":
    tf = vehicle.get_transform()
    loc = tf.location
    yaw_rad = math.radians(tf.rotation.yaw)
    dx = target_pose["x"] - loc.x
    dy = target_pose["y"] - loc.y
    local_x = dx * math.cos(yaw_rad) + dy * math.sin(yaw_rad)
    local_y = -dx * math.sin(yaw_rad) + dy * math.cos(yaw_rad)

    raw_steer = math.atan2(local_y, max(local_x, 0.1)) / math.radians(75.0)
    steer = max(-0.85, min(0.85, raw_steer))

    speed = vehicle_speed(vehicle)
    speed_error = target_speed_mps - speed
    throttle = 0.0
    brake = 0.0
    if speed_error > 0.15:
        throttle = max(0.0, min(0.60, 0.18 + 0.14 * speed_error))
    elif speed_error > -0.15:
        throttle = 0.15
    else:
        brake = max(0.0, min(0.45, 0.08 + 0.18 * (-speed_error)))
    control = carla.VehicleControl()
    control.throttle = throttle
    control.brake = brake
    control.steer = steer
    control.hand_brake = False
    control.reverse = False
    control.manual_gear_shift = False
    return control


def normalize_material_effect(values: np.ndarray) -> np.ndarray:
    return clamp01(np.maximum(values, 0.0) / material_effect_display_vmax())


def values_to_mode_colors(mode: str, intensity_norm: np.ndarray, pseudo_norm: np.ndarray,
                          materials: np.ndarray, material_effect: np.ndarray) -> np.ndarray:
    if mode == "intensity":
        return colormap_intensity(intensity_norm)
    if mode == "global":
        return colormap_pseudo(pseudo_norm)
    if mode == "material_effect":
        return colormap_pseudo(normalize_material_effect(material_effect))
    return material_to_rgb(materials)


def parse_image(image: "carla.Image") -> np.ndarray:
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = arr.reshape((image.height, image.width, 4))
    return arr


def semantic_raw_to_grayscale_bgra(raw_bgra: np.ndarray) -> np.ndarray:
    # CARLA semantic camera stores the class tag in the R channel of the raw BGRA image.
    tags = raw_bgra[:, :, 2].astype(np.uint8)
    if tags.size == 0:
        return raw_bgra.copy()
    max_tag = int(tags.max())
    if max_tag <= 0:
        gray = tags
    else:
        gray = np.rint(tags.astype(np.float32) * (255.0 / max_tag)).astype(np.uint8)
    vis = np.empty_like(raw_bgra)
    vis[:, :, 0] = gray
    vis[:, :, 1] = gray
    vis[:, :, 2] = gray
    vis[:, :, 3] = 255
    return vis


def bgra_to_rgb_surface(arr_bgra: np.ndarray) -> pygame.Surface:
    rgb = arr_bgra[:, :, :3][:, :, ::-1]  # BGRA -> RGB
    surf = pygame.surfarray.make_surface(np.swapaxes(rgb, 0, 1))
    return surf


def draw_text(surface: pygame.Surface, font: pygame.font.Font, text: str, pos: tuple[int, int],
              color: tuple[int, int, int] = TEXT_MAIN) -> None:
    surface.blit(font.render(text, True, color), pos)


def draw_text_block(surface: pygame.Surface, font: pygame.font.Font, text: str, pos: tuple[int, int],
                    max_width: int, line_gap: int = 4,
                    color: tuple[int, int, int] = TEXT_MAIN) -> int:
    words = text.split()
    if not words:
        return 0

    x, y = pos
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if font.size(candidate)[0] <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)

    step = font.get_linesize() + line_gap
    for idx, line in enumerate(lines):
        draw_text(surface, font, line, (x, y + idx * step), color)
    return len(lines) * step


def draw_card(surface: pygame.Surface, rect: pygame.Rect) -> None:
    pygame.draw.rect(surface, CARD_BG, rect, border_radius=14)
    pygame.draw.rect(surface, CARD_BORDER, rect, width=1, border_radius=14)


@dataclass
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


class SensorBuffer:
    def __init__(self, maxsize: int = 16):
        self.maxsize = max(3, int(maxsize))
        self.frames: OrderedDict[int, object] = OrderedDict()
        self.latest_frame: int | None = None
        self.latest_data = None
        self.lock = threading.Lock()
        self.closed = False

    def put(self, frame_id: int, data) -> None:
        frame_id = int(frame_id)
        with self.lock:
            if self.closed:
                return
            self.frames[frame_id] = data
            self.frames.move_to_end(frame_id)
            self.latest_frame = frame_id
            self.latest_data = data
            while len(self.frames) > self.maxsize:
                self.frames.popitem(last=False)

    def get_latest(self):
        with self.lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame, self.latest_data

    def get(self, frame_id: int):
        with self.lock:
            return self.frames.get(int(frame_id))

    def discard_older_than(self, frame_id: int) -> None:
        cutoff = int(frame_id)
        with self.lock:
            while self.frames:
                oldest = next(iter(self.frames))
                if oldest < cutoff:
                    self.frames.popitem(last=False)
                else:
                    break

    def discard_through(self, frame_id: int) -> None:
        cutoff = int(frame_id)
        with self.lock:
            while self.frames:
                oldest = next(iter(self.frames))
                if oldest <= cutoff:
                    self.frames.popitem(last=False)
                else:
                    break

    def available_frames(self) -> set[int]:
        with self.lock:
            return set(self.frames.keys())

    def close(self) -> None:
        with self.lock:
            self.closed = True
            self.frames.clear()
            self.latest_frame = None
            self.latest_data = None


class DatasetRecorder:
    def __init__(self, root_dir: str, scene_id: str, scenario_name: str, args):
        self.root = Path(root_dir) / scene_id / scenario_name
        self.scene_id = scene_id
        self.scenario_name = scenario_name
        self.args = args

        self.rgb_dir = self.root / 'rgb'
        self.semantic_dir = self.root / 'semantic'
        self.semantic_vis_dir = self.root / 'semantic_vis'
        self.projection_debug_dir = self.root / 'projection_debug'
        self.lidar_raw_dir = self.root / 'lidar_raw'
        self.lidar_labels_dir = self.root / 'lidar_labels'
        # The agent's own cloud, before and after the closed-loop operator. It is
        # a different sensor from lidar_raw/ above, and a different quantity from
        # pseudo_final: this is what the driving agent actually reads.
        self.agent_cloud_dir = self.root / 'agent_cloud'
        self.lmdrive_rgb_dir = self.root / 'lmdrive_rgb'
        for d in [self.rgb_dir, self.semantic_dir, self.semantic_vis_dir,
                  self.projection_debug_dir, self.lidar_raw_dir, self.lidar_labels_dir,
                  self.agent_cloud_dir]:
            d.mkdir(parents=True, exist_ok=True)
        if getattr(args, "lmdrive_rig", False):
            for view in ("rgb_front", "rgb_left", "rgb_right", "rgb_rear"):
                (self.lmdrive_rgb_dir / view).mkdir(parents=True, exist_ok=True)

        self.frame_meta_csv = self.root / 'frame_metadata.csv'
        # A cloud and an image are only usable together if the consumer can
        # reproduce the projection, and only comparable across runs if the
        # sensor geometry is on record. Both live in calibration.json.
        self.calibration_json = self.root / 'calibration.json'
        # Ego pose alone supports neither tracking nor detection: those need the
        # other actors, and their boxes.
        self.actors_csv = self.root / 'actors.csv'
        self._saved = 0
        self._accepted = 0
        self._pending_frames = deque()
        self._buffer_mode = getattr(self.args, "save_last_seconds", 0.0) > 0.0
        self._buffer_capacity = self._compute_buffer_capacity()
        self._init_frame_metadata_csv()
        self._init_actors_csv()
        self._write_scenario_metadata()

    def _compute_buffer_capacity(self) -> int:
        if not self._buffer_mode:
            return 0
        seconds = max(0.0, float(getattr(self.args, "save_last_seconds", 0.0)))
        fps = max(1.0, float(getattr(self.args, "fps", 20)))
        stride = max(1, int(getattr(self.args, "save_every", 1)))
        capacity = int(math.ceil(seconds * fps / stride))
        if getattr(self.args, "max_save_frames", 0) > 0:
            capacity = min(capacity, int(self.args.max_save_frames))
        return max(1, capacity)

    def _init_frame_metadata_csv(self):
        # A run restarted into an existing directory would otherwise append
        # wide rows under a narrow header, and the file would no longer parse.
        # The old file is kept, not discarded: it is somebody's recording.
        if self.frame_meta_csv.exists():
            try:
                with open(self.frame_meta_csv, newline='', encoding='utf-8') as f:
                    head = next(csv.reader(f), [])
            except Exception:
                head = []
            if head and 'num_actors' not in head:
                stale = self.frame_meta_csv.with_suffix('.pre_v2.csv')
                self.frame_meta_csv.replace(stale)
                print(f"[toolkit] frame_metadata.csv nel formato precedente, "
                      f"spostato in {stale.name}", flush=True)
        if not self.frame_meta_csv.exists():
            with open(self.frame_meta_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'scene_id', 'scenario', 'frame_id', 'timestamp',
                    'ego_x', 'ego_y', 'ego_z',
                    'ego_roll', 'ego_pitch', 'ego_yaw',
                    'weather', 'num_points',
                    'projected_points', 'projection_ratio',
                    'known_material_points', 'known_material_ratio',
                    # The pose says where the ego was; these say what it was
                    # doing, which is what a behaviour model has to predict.
                    'ego_vx', 'ego_vy', 'ego_vz', 'ego_speed',
                    'ego_throttle', 'ego_steer', 'ego_brake',
                    'num_actors',
                ])

    def _init_actors_csv(self):
        """One row per surrounding actor per saved frame.

        Kept beside the frames rather than inside them because it is the same
        few hundred bytes whether or not a consumer wants it, and because a flat
        table is what a tracking or prediction benchmark reads. The extents are
        half-sizes in the actor's own frame, which with the pose gives the 3D
        box without a second pass over the simulator.
        """
        if not self.actors_csv.exists():
            with open(self.actors_csv, 'w', newline='', encoding='utf-8') as f:
                csv.writer(f).writerow([
                    'frame_id', 'timestamp', 'actor_id', 'type_id', 'category',
                    'x', 'y', 'z', 'roll', 'pitch', 'yaw',
                    'vx', 'vy', 'vz', 'speed',
                    'extent_x', 'extent_y', 'extent_z',
                    'bbox_offset_x', 'bbox_offset_y', 'bbox_offset_z',
                    'distance_to_ego',
                ])

    def write_calibration(self, intrinsics, sensors: dict, extra: dict | None = None) -> None:
        """Record the sensor geometry once per run.

        Without this a consumer holds a point cloud and an image that cannot be
        put in correspondence, which defeats the point of saving both. The
        camera-from-lidar transform is read from the live actors rather than
        recomputed from the mounting constants, so what is written is the
        transform the projection code itself used; the sensors are rigidly
        attached to the same body, so it is constant for the run.
        """
        def _tf(tf) -> dict | None:
            if tf is None:
                return None
            loc, rot = tf.location, tf.rotation
            return {'x': float(loc.x), 'y': float(loc.y), 'z': float(loc.z),
                    'roll': float(rot.roll), 'pitch': float(rot.pitch),
                    'yaw': float(rot.yaw)}

        cam = sensors.get('rgb_camera') or sensors.get('semantic_camera')
        lidar = sensors.get('lidar')
        payload: dict = {
            'schema': 'matsense-calibration-1',
            'scene_id': self.scene_id,
            'scenario': self.scenario_name,
            'conventions': {
                'sensor_frame': 'CARLA/UE4, left-handed: x forward, y right, z up, in metres',
                'rotations_deg': 'roll, pitch, yaw as reported by carla.Rotation',
                'camera_frame': 'x right, y down, z forward (the pinhole frame K applies to)',
                'ue_to_camera': [[0, 1, 0], [0, 0, -1], [1, 0, 0]],
                'note': ('project as: p_cam_ue = T_camera_from_lidar @ p_lidar; '
                         'p_cam = ue_to_camera @ p_cam_ue; uv = K @ p_cam / p_cam.z'),
            },
        }
        if intrinsics is not None:
            payload['camera_intrinsics'] = {
                'width': int(intrinsics.width), 'height': int(intrinsics.height),
                'fx': float(intrinsics.fx), 'fy': float(intrinsics.fy),
                'cx': float(intrinsics.cx), 'cy': float(intrinsics.cy),
                'fov_deg': float(getattr(self.args, 'cam_fov', 0.0) or 0.0),
                'K': [[float(intrinsics.fx), 0.0, float(intrinsics.cx)],
                      [0.0, float(intrinsics.fy), float(intrinsics.cy)],
                      [0.0, 0.0, 1.0]],
            }
        # get_transform() on an attached sensor returns its pose in the world,
        # not its mounting offset, so the offset is recovered against the ego.
        ego = getattr(sensors.get('lidar'), 'parent', None)
        ego_from_world = None
        if ego is not None:
            try:
                ego_from_world = np.array(ego.get_transform().get_inverse_matrix(), dtype=np.float64)
            except Exception:
                ego_from_world = None
        mounts = {}
        for name, actor in sensors.items():
            if actor is None:
                continue
            entry = {'type_id': getattr(actor, 'type_id', None),
                     'pose_in_world_at_calibration': _tf(actor.get_transform())}
            if ego_from_world is not None:
                try:
                    world_from_sensor = np.array(actor.get_transform().get_matrix(), dtype=np.float64)
                    entry['T_ego_from_sensor'] = (ego_from_world @ world_from_sensor).tolist()
                except Exception:
                    pass
            mounts[name] = entry
        payload['sensors'] = mounts
        if cam is not None and lidar is not None:
            try:
                world_from_lidar = np.array(lidar.get_transform().get_matrix(), dtype=np.float64)
                cam_from_world = np.array(cam.get_transform().get_inverse_matrix(), dtype=np.float64)
                payload['T_camera_from_lidar'] = (cam_from_world @ world_from_lidar).tolist()
            except Exception as exc:                              # pragma: no cover
                payload['T_camera_from_lidar_error'] = repr(exc)
        payload['lidar_config'] = {
            'channels': getattr(self.args, 'channels', None),
            'range_m': getattr(self.args, 'lidar_range', None),
            'points_per_second': getattr(self.args, 'pps', None),
            'rotation_frequency_hz': getattr(self.args, 'fps', None),
            'upper_fov_deg': getattr(self.args, 'upper_fov', None),
            'lower_fov_deg': getattr(self.args, 'lower_fov', None),
            'horizontal_fov_deg': getattr(self.args, 'horizontal_fov', None),
        }
        if extra:
            payload.update(extra)
        with open(self.calibration_json, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)

    def _write_scenario_metadata(self):
        meta = {
            'scene_id': self.scene_id,
            'scenario': self.scenario_name,
            'map_name': getattr(self.args, 'map_name', 'unknown'),
            'weather': getattr(self.args, 'weather', 'unknown'),
            'material_config_path': str(getattr(self.args, 'material_config', MATERIAL_CONFIG_PATH)),
            'material_profile': getattr(self.args, 'profile_name', MATERIAL_PROFILE_NAME),
            'spawn': {
                'x': getattr(self.args, 'spawn_x', None),
                'y': getattr(self.args, 'spawn_y', None),
                'z': getattr(self.args, 'spawn_z', None),
                'back_m': getattr(self.args, 'spawn_back_m', 0.0),
                'z_offset': getattr(self.args, 'spawn_z_offset', 0.5),
            },
            'trajectory': {
                'source': 'txt' if getattr(self.args, 'traj_txt', None) else ('json' if getattr(self.args, 'traj_json', None) else None),
                'json_path': getattr(self.args, 'traj_json', None),
                'txt_path': getattr(self.args, 'traj_txt', None),
                'follow_mode': getattr(self.args, 'follow_mode', None),
                'z_offset': getattr(self.args, 'traj_z_offset', 0.5),
                'num_poses': len(getattr(self, 'trajectory_poses', []) or []),
            },
            'parked_vehicles': {
                'json_path': getattr(self.args, 'parked_json', None),
                'z_offset': getattr(self.args, 'parked_z_offset', 0.5),
                'limit': getattr(self.args, 'parked_limit', 0),
                'requested_spawn_positions': len(getattr(self, 'parked_spawn_positions', []) or []),
                'seed': getattr(self.args, 'seed', 42),
            },
            'sensor_config': {
                'cam_fov': getattr(self.args, 'cam_fov', None),
                'channels': getattr(self.args, 'channels', None),
                'pps': getattr(self.args, 'pps', None),
                'rotation_frequency': getattr(self.args, 'rotation_frequency', None),
                'lidar_range': getattr(self.args, 'lidar_range', None),
                'horizontal_fov': getattr(self.args, 'horizontal_fov', None),
                'upper_fov': getattr(self.args, 'upper_fov', None),
                'lower_fov': getattr(self.args, 'lower_fov', None),
                'mode': getattr(self.args, 'mode', None),
                'use_base_nominal': getattr(self.args, 'use_base_nominal', False),
                'display_mode_sequence': DISPLAY_MODE_SEQUENCE,
                'strict_sync': getattr(self.args, 'strict_sync', False),
                'strict_sync_timeout': getattr(self.args, 'strict_sync_timeout', 0.5),
                'save_every': getattr(self.args, 'save_every', None),
                'save_last_seconds': getattr(self.args, 'save_last_seconds', 0.0),
            },
            'calibration': {
                'fit_range_min': FIT_RANGE_MIN,
                'fit_range_max': FIT_RANGE_MAX,
                'ref_range_r0': REF_RANGE_R0,
                'cos_eps': COS_EPS,
                'coefficients_high_to_low': [A2, A1, A0],
                'g_r0': G_R0,
                'weather_ratio': WEATHER_RATIO,
                'nominal_base': NOMINAL_BASE,
            }
        }
        with open(self.root / 'scenario_metadata.json', 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2)

    def should_save(self, frame_id: int) -> bool:
        del frame_id
        if self._buffer_mode:
            return True
        if self.args.max_save_frames > 0 and self._saved >= self.args.max_save_frames:
            return False
        self._accepted += 1
        return ((self._accepted - 1) % max(1, self.args.save_every)) == 0

    def _append_frame(self, payload: dict) -> None:
        if self._buffer_mode:
            self._pending_frames.append(payload)
            while len(self._pending_frames) > self._buffer_capacity:
                self._pending_frames.popleft()
            return
        self._write_frame(payload)

    def finalize(self) -> None:
        if not self._buffer_mode or not self._pending_frames:
            return
        for payload in list(self._pending_frames):
            self._write_frame(payload)
        self._pending_frames.clear()

    def save_frame(self, frame_id: int, timestamp: float, ego_transform, weather_name: str,
                   rgb_bgra: np.ndarray | None, semantic_bgra: np.ndarray | None,
                   semantic_vis_bgra: np.ndarray | None, projection_debug_bgra: np.ndarray | None,
                   xyz: np.ndarray, raw_intensity: np.ndarray, semantic_tags: np.ndarray,
                   material_labels: np.ndarray, ranges: np.ndarray, pseudo_final: np.ndarray,
                   intensity_norm: np.ndarray | None = None, pseudo_norm: np.ndarray | None = None,
                   projected_points: int = 0, known_material_points: int = 0,
                   instance_ids: np.ndarray | None = None,
                   ego_state: dict | None = None,
                   actor_states: list | None = None,
                   agent_cloud: dict | None = None):
        payload = {
            'agent_cloud': agent_cloud,
            'instance_ids': None if instance_ids is None else np.asarray(instance_ids).astype(np.uint32, copy=True),
            'ego_state': dict(ego_state or {}),
            'actor_states': list(actor_states or []),
            'frame_id': int(frame_id),
            'timestamp': float(timestamp),
            'ego_transform': carla.Transform(ego_transform.location, ego_transform.rotation),
            'weather_name': weather_name,
            'rgb_bgra': None if rgb_bgra is None else np.array(rgb_bgra, copy=True),
            'semantic_bgra': None if semantic_bgra is None else np.array(semantic_bgra, copy=True),
            'semantic_vis_bgra': None if semantic_vis_bgra is None else np.array(semantic_vis_bgra, copy=True),
            'projection_debug_bgra': None if projection_debug_bgra is None else np.array(projection_debug_bgra, copy=True),
            'xyz': xyz.astype(np.float32, copy=True),
            'raw_intensity': raw_intensity.astype(np.float32, copy=True),
            'semantic_tags': semantic_tags.astype(np.int32, copy=True),
            'material_labels': np.asarray(material_labels).copy(),
            'ranges': ranges.astype(np.float32, copy=True),
            'pseudo_final': pseudo_final.astype(np.float32, copy=True),
            'intensity_norm': None if intensity_norm is None else intensity_norm.astype(np.float32, copy=True),
            'pseudo_norm': None if pseudo_norm is None else pseudo_norm.astype(np.float32, copy=True),
            'projected_points': int(projected_points),
            'known_material_points': int(known_material_points),
        }
        self._append_frame(payload)

    def _write_frame(self, payload: dict):
        frame_id = int(payload['frame_id'])
        timestamp = float(payload['timestamp'])
        ego_transform = payload['ego_transform']
        weather_name = payload['weather_name']
        rgb_bgra = payload['rgb_bgra']
        semantic_bgra = payload['semantic_bgra']
        semantic_vis_bgra = payload['semantic_vis_bgra']
        projection_debug_bgra = payload['projection_debug_bgra']
        xyz = payload['xyz']
        raw_intensity = payload['raw_intensity']
        semantic_tags = payload['semantic_tags']
        material_labels = payload['material_labels']
        ranges = payload['ranges']
        pseudo_final = payload['pseudo_final']
        intensity_norm = payload['intensity_norm']
        pseudo_norm = payload['pseudo_norm']
        projected_points = int(payload['projected_points'])
        known_material_points = int(payload['known_material_points'])
        instance_ids = payload.get('instance_ids')
        ego_state = payload.get('ego_state') or {}
        actor_states = payload.get('actor_states') or []
        stem = f'frame_{frame_id:06d}'

        if rgb_bgra is not None:
            pygame.image.save(bgra_to_rgb_surface(rgb_bgra), str(self.rgb_dir / f'{stem}.png'))
        if semantic_bgra is not None:
            pygame.image.save(bgra_to_rgb_surface(semantic_bgra), str(self.semantic_dir / f'{stem}.png'))
        if semantic_vis_bgra is not None:
            pygame.image.save(bgra_to_rgb_surface(semantic_vis_bgra), str(self.semantic_vis_dir / f'{stem}.png'))
        if projection_debug_bgra is not None:
            pygame.image.save(bgra_to_rgb_surface(projection_debug_bgra), str(self.projection_debug_dir / f'{stem}.png'))

        np.savez_compressed(self.lidar_raw_dir / f'{stem}.npz',
                            xyz=xyz.astype(np.float32),
                            intensity=raw_intensity.astype(np.float32))

        pack = {
            'material_label': np.asarray(material_labels),
            'semantic_tag': semantic_tags.astype(np.int32),
            'range': ranges.astype(np.float32),
            'pseudo_final': pseudo_final.astype(np.float32),
        }
        if intensity_norm is not None:
            pack['intensity_norm'] = intensity_norm.astype(np.float32)
        if pseudo_norm is not None:
            pack['pseudo_norm'] = pseudo_norm.astype(np.float32)
        # CARLA's semantic LiDAR reports the object each return came from, which
        # is instance segmentation for free; it was being matched and discarded.
        # Zero means the point found no partner, not "instance zero".
        if instance_ids is not None and len(instance_ids) == len(ranges):
            pack['instance_id'] = np.asarray(instance_ids).astype(np.uint32)
        np.savez_compressed(self.lidar_labels_dir / f'{stem}.npz', **pack)

        ac = payload.get('agent_cloud')
        if ac:
            np.savez_compressed(self.agent_cloud_dir / f'{stem}.npz', **ac)

        if actor_states:
            with open(self.actors_csv, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                for a in actor_states:
                    writer.writerow([
                        frame_id, timestamp, a['actor_id'], a['type_id'], a['category'],
                        a['x'], a['y'], a['z'], a['roll'], a['pitch'], a['yaw'],
                        a['vx'], a['vy'], a['vz'], a['speed'],
                        a['extent_x'], a['extent_y'], a['extent_z'],
                        a['bbox_offset_x'], a['bbox_offset_y'], a['bbox_offset_z'],
                        a['distance_to_ego'],
                    ])

        loc = ego_transform.location
        rot = ego_transform.rotation
        num_points = int(len(xyz))
        with open(self.frame_meta_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.scene_id, self.scenario_name, frame_id, timestamp,
                loc.x, loc.y, loc.z, rot.roll, rot.pitch, rot.yaw,
                weather_name, num_points,
                projected_points, float(projected_points / max(num_points, 1)),
                known_material_points, float(known_material_points / max(num_points, 1)),
                ego_state.get('vx', ''), ego_state.get('vy', ''), ego_state.get('vz', ''),
                ego_state.get('speed', ''),
                ego_state.get('throttle', ''), ego_state.get('steer', ''),
                ego_state.get('brake', ''),
                len(actor_states),
            ])
        self._saved += 1


class CarlaLidarViewer:
    def __init__(self, args):
        self.args = args
        self.client = carla.Client(args.host, args.port)
        self._validate_server_compatibility()
        self.world = self._connect_initial_world()
        # What to put back on exit, which is not simply what was found. This
        # tool is the one that drives synchronous mode, so a world already in it
        # is the residue of a previous run that died before restoring it, and
        # capturing that as "original" makes every later run faithfully restore
        # a world nothing is ticking. A vehicle spawned there receives gravity
        # once and never the steps that would settle it, so it falls and keeps
        # falling: the symptom reads as a broken map or a bad trajectory, and
        # survives every restart, because each run carefully reinstates it.
        self.original_settings = self.world.get_settings()
        if self.original_settings.synchronous_mode:
            print("[toolkit] il mondo era gia' in sincrono: residuo di una run "
                  "interrotta, al termine verra' rimesso in asincrono", flush=True)
            self.original_settings.synchronous_mode = False
            self.original_settings.fixed_delta_seconds = None
        self.tm = self.client.get_trafficmanager(args.tm_port)
        self.tm.set_synchronous_mode(True)

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / args.fps
        self.world.apply_settings(settings)

        self.blueprints = self.world.get_blueprint_library()
        self.map = self.world.get_map()
        self.actors = []
        self.vehicle = None
        self._srunner = None
        self.rgb_cam = None
        self.sem_cam = None
        self.lidar = None
        self.semantic_lidar = None

        self.rgb_buffer = SensorBuffer(maxsize=24)
        self.sem_buffer = SensorBuffer(maxsize=24)
        self.lidar_buffer = SensorBuffer(maxsize=24)
        self.semantic_lidar_buffer = SensorBuffer(maxsize=24)
        self.birdseye_buffer = SensorBuffer(maxsize=8)

        self.rgb_array = None
        self.birdseye_array = None
        self.birdseye_cam = None
        self.sem_array = None
        self.sem_vis_array = None
        self.last_lidar = None
        self.last_semantic_lidar = None
        self.rgb_frame = None
        self.sem_frame = None
        self.lidar_frame = None
        self.semantic_lidar_frame = None
        self.last_saved_frame = -1
        self.recording_start_time = None
        self.spectator = self.world.get_spectator()
        self.last_material_counts = {}
        self.last_projection_stats = {
            "total_points": 0,
            "projected_points": 0,
            "projection_ratio": 0.0,
            "known_material_points": 0,
            "known_material_ratio": 0.0,
        }
        self.last_display_stats = {
            "pseudo_vmax": float(self.args.display_vmax),
            "display_mode": getattr(self.args, "display_normalization", "percentile"),
        }
        self.last_color_info = None
        self.material_overrides = load_material_overrides(getattr(args, "material_overrides", None))
        self.recorder = None
        self.screenshot_count = 0
        self.screenshot_dir = Path(args.screenshot_dir) if getattr(args, "screenshot_dir", "") else None
        if self.screenshot_dir is not None:
            self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        self.color_mode = args.mode
        self.clock = pygame.time.Clock()
        self.font = None
        self.font_small = None
        self.font_title = None
        self.font_panel = None
        self.intr = None

        self.screen = None
        self.display_size = (args.width, args.height)
        self.left_panel_w = args.width // 2
        self.right_panel_w = args.width - self.left_panel_w
        self.previous_follow_control = None
        self.last_respawn_time = 0.0
        self.min_respawn_interval = 1.5
        self.render_stride = 3
        # Drawing the pygame window costs 33 ms per loop in `material` and
        # 176 ms in `global`, measured across eleven campaign runs; with the
        # scatter_max fix in place that was half the simulation step. A campaign
        # runs under SDL_VIDEODRIVER=dummy, so those pixels reach no one. The
        # switch is here rather than automatic because the same script drives
        # the preview and screenshot tools, which do want the window.
        self.no_draw = bool(getattr(args, "no_draw", False))
        if self.no_draw and self.screenshot_dir is not None:
            # screenshots come out of the draw path; asking for both is a
            # contradiction, and silently dropping the screenshots would be worse
            print("[toolkit] --no-draw ignorato: sono stati chiesti gli screenshot",
                  flush=True)
            self.no_draw = False
        if getattr(args, "traj_txt", None):
            self.trajectory_poses = load_trajectory_txt(
                args.traj_txt,
                offset_x=float(getattr(args, "utm_offset_x", 0.0)),
                offset_y=float(getattr(args, "utm_offset_y", 0.0)),
                step=int(getattr(args, "traj_step", 1)),
            )
        else:
            self.trajectory_poses = load_trajectory_json(args.traj_json) if getattr(args, "traj_json", None) else []
        self.trajectory_poses = transform_trajectory_poses(
            self.trajectory_poses,
            shift_x=float(getattr(args, "traj_shift_x", 0.0)),
            shift_y=float(getattr(args, "traj_shift_y", 0.0)),
            shift_z=float(getattr(args, "traj_shift_z", 0.0)),
            scale=float(getattr(args, "traj_scale", 1.0)),
        )
        self.trajectory_idx = 0
        self.parked_spawn_positions = []
        if getattr(args, "parked_json", None):
            parked_data = json.loads(Path(args.parked_json).read_text(encoding="utf-8"))
            self.parked_spawn_positions = list(parked_data.get("spawn_positions", []))
            self.parked_spawn_positions = transform_parked_spawn_positions(
                self.parked_spawn_positions,
                shift_x=float(getattr(args, "parked_shift_x", 0.0)),
                shift_y=float(getattr(args, "parked_shift_y", 0.0)),
                shift_z=float(getattr(args, "parked_shift_z", 0.0)),
                scale=float(getattr(args, "parked_scale", 1.0)),
            )
            self.parked_lane_corrections = []
            if getattr(args, "parked_avoid_ego_lane", False):
                (self.parked_spawn_positions,
                 self.parked_lane_corrections) = resolve_parked_ego_lane_conflicts(
                    self.parked_spawn_positions,
                    self.trajectory_poses,
                    float(getattr(args, "parked_clearance_m", 2.5)),
                )
                print(f"[toolkit] parked_json: {len(self.parked_lane_corrections)} "
                      f"veicoli spostati fuori dalla corsia dell'ego "
                      f"(clearance {float(getattr(args, 'parked_clearance_m', 2.5)):.2f} m)",
                      flush=True)
        # Both inputs are checked before anything is spawned: the trajectory
        # because the ego follows it, the layout because a vehicle standing in
        # the void still returns nothing to the LiDAR and quietly empties the
        # scene the measurement depends on.
        _mode = str(getattr(args, "validate_against_map", "abort"))
        if _mode != "off":
            _max = float(getattr(args, "validate_max_median_m", 8.0))
            if self.trajectory_poses:
                validate_points_against_map(
                    self.map, [(p["x"], p["y"]) for p in self.trajectory_poses],
                    "traiettoria", _max, _mode)
            if self.parked_spawn_positions:
                validate_points_against_map(
                    self.map,
                    [(p["start"][0], p["start"][1])
                     for p in self.parked_spawn_positions if p.get("start")],
                    "veicoli in sosta", _max, _mode)

        self.walker_specs = []
        if getattr(args, "walkers_json", None):
            walker_data = json.loads(Path(args.walkers_json).read_text(encoding="utf-8"))
            self.walker_specs = list(walker_data.get("walkers", []))
        self.walker_states = []
        self.dynamic_actor_specs = []
        self.dynamic_actor_states = []
        self.parked_actor_states = []
        # exists even when no traffic is requested, so a consumer never has to
        # guard the attribute to find out whether the run had any
        self.tm_vehicles = []
        self.ambient_walkers = []
        self.hazard_obstacle_actor = None
        self.hazard_obstacle_position = None
        self.first_obstacle_in_range_progress_m = None
        self.collision_events = []
        self.collision_sensor = None
        self.termination_mode = None
        self._pcla_stopped_since = None
        self.termination_reason = ""
        self._last_tick_time_s = None
        self._last_tick_speed_mps = None
        self._max_deceleration_mps2 = 0.0
        self._min_ttc_proxy_s = float("inf")
        self._first_brake_progress_m = None
        self._brake_threshold = float(getattr(args, "brake_threshold", 0.05))
        self._route_file_hash = self._sha256_path(getattr(args, "pcla_route", ""))
        self._profile_config_sha256 = self._sha256_path(getattr(args, "material_config", ""))
        self._profile_version = str(getattr(args, "profile_version", "")) or str(getattr(args, "profile_name", ""))
        self._carla_client_version = ""
        self._carla_server_version = ""
        try:
            self._carla_client_version = str(self.client.get_client_version())
        except Exception:
            self._carla_client_version = ""
        try:
            self._carla_server_version = str(self.client.get_server_version())
        except Exception:
            self._carla_server_version = ""
        if getattr(args, "dynamic_json", None):
            dynamic_data = json.loads(Path(args.dynamic_json).read_text(encoding="utf-8"))
            self.dynamic_actor_specs = normalize_dynamic_actor_specs(
                list(dynamic_data.get("dynamic_actors", [])),
                shift_x=float(getattr(args, "dynamic_shift_x", 0.0)),
                shift_y=float(getattr(args, "dynamic_shift_y", 0.0)),
                shift_z=float(getattr(args, "dynamic_shift_z", 0.0)),
                scale=float(getattr(args, "dynamic_scale", 1.0)),
            )
        self._validate_map_alignment()

        if getattr(args, "save_dataset", False):
            args.map_name = self.map.name
            self.recorder = DatasetRecorder(args.dataset_root, args.scene_id, args.scenario_name, args)

    @staticmethod
    def _sha256_path(path_like: str | os.PathLike | None) -> str:
        if not path_like:
            return ""
        path = Path(path_like)
        if not path.exists() or not path.is_file():
            return ""
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def _normalize_town_name(name: str) -> str:
        town = str(name).split("/")[-1]
        if town.endswith("_Opt"):
            town = town[:-4]
        if town == "Town10":
            town = "Town10HD"
        return town

    def _validate_server_compatibility(self) -> None:
        if getattr(self.args, "allow_version_mismatch", False):
            return

        client_version = "unknown"
        server_version = "unknown"
        try:
            self.client.set_timeout(min(float(getattr(self.args, "client_timeout", 30.0)), 10.0))
            client_version = str(self.client.get_client_version())
            server_version = str(self.client.get_server_version())
        except RuntimeError as exc:
            print(
                f"[toolkit] startup: version probe skipped ({exc})",
                flush=True,
            )
            return

        if client_version != server_version:
            raise RuntimeError(
                "CARLA client/server version mismatch detected. "
                f"client={client_version} server={server_version}. "
                "This launcher expects a matching CARLA build and may crash with native errors such as std::bad_alloc "
                "when the simulator is incompatible. "
                "Start a CARLA 0.9.16 server for this repo, or rerun with --allow-version-mismatch if you need to bypass this check."
            )

    def _get_world_with_retry(self, timeout_seconds: float, attempts: int, sleep_seconds: float) -> carla.World:
        last_error = None
        for attempt in range(1, attempts + 1):
            self.client.set_timeout(timeout_seconds)
            try:
                return self.client.get_world()
            except RuntimeError as exc:
                last_error = exc
                if attempt < attempts:
                    print(
                        f"[toolkit] startup: get_world attempt {attempt}/{attempts} failed "
                        f"({exc}); retrying in {sleep_seconds:.1f}s",
                        flush=True,
                    )
                    time.sleep(sleep_seconds)
        raise last_error

    def _connect_initial_world(self) -> carla.World:
        steady_timeout = float(getattr(self.args, "client_timeout", 30.0))
        if getattr(self.args, "pcla_agent", ""):
            town = getattr(self.args, "pcla_town", "Town02")
            requested_town = self._normalize_town_name(town)
            skip_world_load = os.environ.get("MATSENSE_PCLA_SKIP_WORLD_LOAD", "").strip() == "1"
            world = self._get_world_with_retry(
                timeout_seconds=max(steady_timeout, 30.0),
                attempts=2,
                sleep_seconds=2.0,
            )
            current_town = self._normalize_town_name(world.get_map().name)
            if current_town != requested_town:
                if skip_world_load:
                    print(
                        f"[toolkit] startup: reusing current world {current_town} "
                        f"despite requested {requested_town} "
                        f"(MATSENSE_PCLA_SKIP_WORLD_LOAD=1)",
                        flush=True,
                    )
                else:
                    print(f"[toolkit] startup: loading {town} before first get_world", flush=True)
                    self.client.set_timeout(60.0)
                    self.client.load_world(town)
                    world = self._get_world_with_retry(timeout_seconds=60.0, attempts=2, sleep_seconds=2.0)
            else:
                print(
                    f"[toolkit] startup: reusing current world {current_town} for requested {requested_town}",
                    flush=True,
                )
            self.client.set_timeout(steady_timeout)
            return world

        world = self._get_world_with_retry(timeout_seconds=max(steady_timeout, 30.0), attempts=3, sleep_seconds=2.0)
        self.client.set_timeout(steady_timeout)
        return world

    def _save_display_screenshot(self, loop_count: int) -> bool:
        if self.screenshot_dir is None:
            return False
        every = max(1, int(getattr(self.args, "screenshot_every", 1)))
        if loop_count % every != 0:
            return False
        stem = f"{self.args.weather}_{self.color_mode}_{self.screenshot_count:04d}.png"
        pygame.image.save(self.screen, str(self.screenshot_dir / stem))
        self.screenshot_count += 1
        max_frames = int(getattr(self.args, "screenshot_max_frames", 0))
        return max_frames > 0 and self.screenshot_count >= max_frames

    def _validate_map_alignment(self) -> None:
        warnings: list[str] = []
        if self.trajectory_poses:
            traj_points = [
                (float(p["x"]), float(p["y"]), float(p.get("z", 0.0)))
                for p in self.trajectory_poses
            ]
            distances = sample_alignment_distances(self.map, traj_points)
            if distances.size:
                median = float(np.median(distances))
                p90 = float(np.percentile(distances, 90))
                print(
                    f"[toolkit] alignment: trajectory nearest-road median={median:.2f}m p90={p90:.2f}m map={self.map.name}",
                    flush=True,
                )
                if median > 8.0 or p90 > 18.0:
                    raise RuntimeError(
                        "Trajectory/map mismatch: sampled trajectory points are too far from the loaded CARLA road network "
                        f"(median={median:.2f}m, p90={p90:.2f}m, map={self.map.name}).\n"
                        "Use a trajectory and OpenDRIVE offset generated from the same map coordinates."
                    )
                if median > 3.0 or p90 > 8.0:
                    warnings.append(f"trajectory alignment is loose: median={median:.2f}m p90={p90:.2f}m")

        if self.parked_spawn_positions:
            parked_points = []
            for item in self.parked_spawn_positions:
                raw = item.get("start")
                if not raw or len(raw) < 2:
                    continue
                z = float(raw[2]) if len(raw) > 2 else 0.0
                parked_points.append((float(raw[0]), float(raw[1]), z))
            distances = sample_alignment_distances(self.map, parked_points)
            if distances.size:
                median = float(np.median(distances))
                p90 = float(np.percentile(distances, 90))
                print(
                    f"[toolkit] alignment: parked nearest-road median={median:.2f}m p90={p90:.2f}m map={self.map.name}",
                    flush=True,
                )
                if median > 12.0 or p90 > 28.0:
                    raise RuntimeError(
                        "Parked-vehicle/map mismatch: sampled parked vehicles are too far from the loaded CARLA road network "
                        f"(median={median:.2f}m, p90={p90:.2f}m, map={self.map.name}).\n"
                        "Use parked vehicle positions generated from the same map/XODR coordinate system."
                    )
                if median > 6.0 or p90 > 16.0:
                    warnings.append(f"parked alignment is loose: median={median:.2f}m p90={p90:.2f}m")

        for warning in warnings:
            print(f"[toolkit] alignment warning: {warning}", flush=True)

    def setup_pygame(self):
        requested_driver = str(getattr(self.args, "sdl_driver", "") or "").strip()
        if requested_driver:
            os.environ["SDL_VIDEODRIVER"] = requested_driver
        elif not os.environ.get("SDL_VIDEODRIVER") and os.environ.get("DISPLAY") and os.environ.get("XDG_SESSION_TYPE") == "x11":
            os.environ["SDL_VIDEODRIVER"] = "x11"

        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode(self.display_size, pygame.DOUBLEBUF)
        pygame.display.set_caption("CARLA Material-Aware LiDAR Demo")
        print(
            f"[toolkit] pygame: driver={pygame.display.get_driver()} "
            f"display={os.environ.get('DISPLAY', '')} "
            f"sdl_videodriver={os.environ.get('SDL_VIDEODRIVER', '')}",
            flush=True,
        )
        self.font = pygame.font.SysFont("segoeui", 17)
        self.font_small = pygame.font.SysFont("segoeui", 13)
        self.font_panel = pygame.font.SysFont("segoeui", 20, bold=True)
        self.font_title = pygame.font.SysFont("segoeui", 26, bold=True)

    def get_layout(self) -> dict[str, pygame.Rect]:
        margin = 18
        gap = 14
        title_h = 62
        left_x = margin
        right_x = self.left_panel_w + gap
        left_w = self.left_panel_w - margin - gap
        right_w = self.args.width - right_x - margin
        content_top = margin + title_h + 10
        content_bottom = self.args.height - margin
        content_h = content_bottom - content_top

        left_h = content_h
        status_h = max(230, int(content_h * 0.34))
        controls_h = 126
        bev_h = content_h - status_h - controls_h - 2 * gap

        cam_h = int(left_h * 0.60)
        birdseye_h = left_h - cam_h - gap

        return {
            "title": pygame.Rect(margin, margin, self.args.width - 2 * margin, title_h),
            "camera": pygame.Rect(left_x, content_top, left_w, cam_h),
            "birdseye": pygame.Rect(left_x, content_top + cam_h + gap, left_w, birdseye_h),
            "status": pygame.Rect(right_x, content_top, right_w, status_h),
            "controls": pygame.Rect(right_x, content_top + status_h + gap, right_w, controls_h),
            "bev": pygame.Rect(right_x, content_top + status_h + gap + controls_h + gap, right_w, bev_h),
        }

    def _register_hero_with_provider(self):
        """Put the ego where a ScenarioRunner-based agent expects to find it.

        Agents written for the Leaderboard reach the ego through
        CarlaDataProvider.get_hero_actor(), which walks the provider's own actor
        pool looking for role_name 'hero'. The Leaderboard runner fills that pool;
        this harness spawns the ego itself, so the lookup returns None and the
        agent dies on the first tick with an AttributeError about NoneType - a
        message that says nothing about the actual cause.

        There is more than one such registry, and which one an agent reads is
        decided by an import that can fail for an unrelated reason. Autoware's
        agent asks for srunner's provider and the Leaderboard's Track in a single
        try block, so on a machine without the leaderboard package - this one -
        the whole block falls through to PCLA's bundled copy under
        leaderboard_codes, a different class holding a different, empty pool.
        Registering in one and reading from the other looks exactly like not
        registering at all: get_hero_actor() returns None and the agent dies on
        the first tick with an AttributeError naming neither.

        So every provider that imports gets the ego, and the count is printed.
        The classes are independent registries, so filling a spare one costs a
        dictionary entry and removes a whole class of silent mismatch.
        """
        if self.vehicle is None:
            return
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        try:
            import srunner_bridge
            srunner_bridge.available()
        except Exception:
            pass
        pcla_dir = getattr(self.args, "pcla_dir", "") or ""
        if pcla_dir and pcla_dir not in sys.path:
            sys.path.insert(0, pcla_dir)

        done = []
        for mod in ("srunner.scenariomanager.carla_data_provider",
                    "leaderboard_codes.carla_data_provider"):
            try:
                provider = __import__(mod, fromlist=["CarlaDataProvider"])
                cdp = provider.CarlaDataProvider
                cdp.set_client(self.client)
                cdp.set_world(self.world)
                # The pool entry is the whole fix, and register_actor is not part
                # of it. get_hero_actor() walks _carla_actor_pool alone; what
                # register_actor fills are the velocity, location and transform
                # caches, which is why PCLA calling it during setup_sensors still
                # left the lookup empty. Calling it here would be worse than
                # useless: it refuses an actor it already holds, so PCLA's own
                # call moments later raises KeyError and takes the run with it.
                cdp._carla_actor_pool[self.vehicle.id] = self.vehicle
                done.append(mod.split(".")[0])
            except Exception as exc:
                print(f"[toolkit] provider {mod.split('.')[0]} non registrato: {exc}",
                      flush=True)
        if done:
            print(f"[toolkit] ego registrato in {', '.join(done)} "
                  f"(id={self.vehicle.id}, role={self.vehicle.attributes.get('role_name')})",
                  flush=True)

    def _start_srunner_scenario(self):
        """Build the official scenario this run was asked for, if any.

        Only reached once the ego and every hand-placed actor exist, because the
        scenario measures from the ego's transform and spawns its own actors
        relative to it. A scenario that cannot be built stops the run rather than
        letting it continue as an unmarked baseline: a campaign cell that
        silently lost its hazard is worse than one that failed.
        """
        name = getattr(self.args, "srunner_scenario", "") or ""
        route = getattr(self.args, "srunner_route", "") or ""
        if not name and not route:
            return
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import srunner_bridge
        if not srunner_bridge.available():
            raise RuntimeError(
                "--srunner-scenario chiesto ma ScenarioRunner non si importa. "
                "Indica SCENARIO_RUNNER_ROOT e CARLA_PYTHONAPI_ROOT.")
        if route:
            self._srunner = srunner_bridge.RouteScenarios(
                self.world, self.client, self.vehicle, route,
                tm_port=int(getattr(self.args, "tm_port", 8000)))
            self._srunner.build_tree(self.vehicle)
            s = self._srunner.summary()
            print(f"[srunner] route {Path(route).name}: {s['built']} scenari "
                  f"costruiti, {s['failed']} falliti, {s['actors']} attori, "
                  f"tipi {', '.join(s['types'])}", flush=True)
            for nm, why in self._srunner.failed:
                print(f"[srunner]   NON costruito {nm}: {why}", flush=True)
            return
        params = {}
        for item in getattr(self.args, "srunner_param", None) or []:
            if "=" not in item:
                raise ValueError(f"--srunner-param vuole chiave=valore, non {item!r}")
            k, v = item.split("=", 1)
            params[k.strip()] = v.strip()
        self._srunner = srunner_bridge.ScenarioBridge(
            self.world, self.client, self.vehicle, name, params,
            tm_port=int(getattr(self.args, "tm_port", 8000)))
        print(f"[srunner] scenario {name} costruito, "
              f"{len(self._srunner.actors)} attori, parametri {params or '(default)'}",
              flush=True)

    def _tick_srunner(self):
        """Advance the scenario's behaviour tree by one step.

        Called after the world tick in every driving branch, never before: the
        data provider caches poses per frame, so a tree ticked first would act on
        the previous frame. Failures here are reported once and then left alone,
        because a scenario that has finished its story is not an error and the
        run still has to record the frames it promised.
        """
        if self._srunner is None:
            return
        try:
            self._srunner.tick()
        except Exception as exc:
            print(f"[srunner] tick fallito, scenario disattivato: {exc}", flush=True)
            self._srunner = None

    def _stop_srunner_scenario(self):
        if self._srunner is None:
            return
        print(f"[srunner] {self._srunner.summary()}", flush=True)
        self._srunner.cleanup()
        self._srunner = None

    def destroy(self):
        self._stop_srunner_scenario()
        # A failure before the session exists used to raise here and replace the
        # real traceback with an AttributeError about a missing attribute, which
        # says nothing about what actually went wrong. The summary is best-effort
        # during teardown; the original exception is the one worth seeing.
        try:
            self._write_run_summary()
        except Exception as exc:
            print(f"[toolkit] run_summary non scritto: {exc}", flush=True)
        batch_fast_shutdown = os.environ.get("MATSENSE_FORCE_EXIT_ON_SHUTDOWN", "").strip() == "1"
        for buffer_name in (
            "rgb_buffer",
            "sem_buffer",
            "lidar_buffer",
            "semantic_lidar_buffer",
            "birdseye_buffer",
        ):
            try:
                buffer_obj = getattr(self, buffer_name, None)
                if buffer_obj is not None:
                    buffer_obj.close()
            except Exception:
                pass
        if self.recorder is not None:
            self.recorder.finalize()
        if hasattr(self, "_pcla_logger") and self._pcla_logger is not None:
            try:
                self._pcla_logger.close()
            except Exception:
                pass
            self._pcla_logger = None
        # Switch CARLA back to async mode FIRST so the server doesn't hang
        # waiting for a world.tick() that will never come.
        try:
            settings = self.world.get_settings()
            if settings.synchronous_mode:
                self.world.apply_settings(self.original_settings)
        except Exception:
            pass
        try:
            self.tm.set_synchronous_mode(False)
        except Exception:
            pass
        for actor in self.actors[::-1]:
            try:
                if "sensor." in actor.type_id and actor.is_alive:
                    actor.stop()
            except Exception:
                pass
        try:
            # Let sensor callback threads drain before actor destruction/finalizer shutdown.
            time.sleep(0.2)
        except Exception:
            pass
        if batch_fast_shutdown:
            for actor in self.actors[::-1]:
                try:
                    actor.destroy()
                except Exception:
                    pass
            self.actors = []
            return
        # Cleanup PCLA before destroying actors
        try:
            if getattr(self, '_pcla', None) is not None:
                self._pcla.cleanup()
        except Exception:
            pass
        try:
            if getattr(self, '_pcla_session', None) is not None:
                self._pcla_session.cleanup()
        except Exception:
            pass
        for actor in self.actors[::-1]:
            try:
                actor.destroy()
            except Exception:
                pass
        self.actors = []
        pygame.quit()

    def _resolve_spawn_transform(self):
        if self.trajectory_poses:
            spawn = pose_to_road_transform(self.world, self.trajectory_poses[0], float(getattr(self.args, "traj_z_offset", 0.0)))
            return spawn

        # Preferred path: explicit target location snapped to the nearest driving waypoint.
        if self.args.spawn_x is not None and self.args.spawn_y is not None:
            z = 0.0 if self.args.spawn_z is None else self.args.spawn_z
            target = carla.Location(x=float(self.args.spawn_x), y=float(self.args.spawn_y), z=float(z))
            wp = self.map.get_waypoint(
                target,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            if wp is None:
                raise RuntimeError(
                    f"Could not project target spawn ({target.x:.3f}, {target.y:.3f}, {target.z:.3f}) to a driving waypoint"
                )
            if self.args.spawn_back_m > 0.0:
                prev = wp.previous(float(self.args.spawn_back_m))
                if prev:
                    wp = prev[0]
            spawn = wp.transform
            spawn.location.z += float(self.args.spawn_z_offset)
            return spawn

        # Fallback: regular CARLA spawn points.
        spawn_points = self.map.get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points found in current CARLA map")
        spawn = spawn_points[self.args.spawn_index % len(spawn_points)]
        return spawn

    def _spawn_parked_vehicles_from_json(self):
        if not self.parked_spawn_positions:
            print("[toolkit] parked_json: no positions loaded", flush=True)
            return
        self.parked_actor_states = []
        blueprints = get_filtered_vehicle_blueprints(self.world)
        if not blueprints:
            print("[toolkit] parked_json: no vehicle blueprints available", flush=True)
            return
        positions = self.parked_spawn_positions
        limit = int(getattr(self.args, "parked_limit", 0))
        if limit > 0:
            positions = positions[:limit]
        print(f"[toolkit] parked_json: attempting spawn for {len(positions)} entries", flush=True)
        spawned = 0
        failures = 0
        for item in positions:
            bp = pick_parked_blueprint(blueprints, item, self.world.get_blueprint_library())
            transform = parked_transform_from_entry(self.world, item, float(getattr(self.args, "parked_z_offset", 0.5)))
            actor = self.world.try_spawn_actor(bp, transform)
            if actor is None:
                failures += 1
                continue
            # Freeze immediately; batch spawning with do_tick=True lets cars receive
            # one physics frame and can make them slide before they are parked.
            try:
                actor.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                actor.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                actor.set_simulate_physics(False)
            except Exception:
                pass
            self.actors.append(actor)
            self.parked_actor_states.append({"actor": actor, "spec": item})
            if getattr(self.args, "hazard_obstacle_source", "none") == "first_parked" and self.hazard_obstacle_actor is None:
                self.hazard_obstacle_actor = actor
                loc = actor.get_location()
                self.hazard_obstacle_position = (float(loc.x), float(loc.y), float(loc.z))
            spawned += 1
        print(f"[toolkit] parked_json: spawned {spawned}/{len(positions)} failed={failures}", flush=True)
        self._settle_parked_actors()
        self._write_parked_validation_report(len(positions), spawned, failures)

    def _write_startup_diagnostics(self, spawn, used_fallback):
        """What the run was actually given, written where it can be read later.

        A vehicle that ends up somewhere with no ground under it looks the same
        from the outside whatever put it there: a trajectory in the wrong frame,
        a spawn that fell back, a world left in synchronous mode by a previous
        run. Those have different fixes, and telling them apart after the fact
        needs the state at startup, which otherwise exists only in a terminal
        someone has already closed.

        Always to the same path, so it can be read without arranging anything
        in advance.
        """
        path = Path(getattr(self.args, "startup_diagnostics_json", "")
                    or "/tmp/matsense_last_startup.json")
        poses = self.trajectory_poses or []
        try:
            settings = self.world.get_settings()
            sync = bool(settings.synchronous_mode)
        except Exception:                                  # noqa: BLE001
            sync = None
        info = {
            "map": str(self.world.get_map().name),
            "synchronous_mode": sync,
            "traj_json": getattr(self.args, "traj_json", "") or "",
            "traj_txt": getattr(self.args, "traj_txt", "") or "",
            "follow_mode": getattr(self.args, "follow_mode", ""),
            "n_poses": len(poses),
            "first_pose": ({"x": round(poses[0]["x"], 3),
                            "y": round(poses[0]["y"], 3),
                            "z": round(poses[0].get("z", 0.0), 3),
                            "yaw": round(poses[0]["yaw"], 2)} if poses else None),
            "pose_extent": ({"x": [round(min(p["x"] for p in poses), 1),
                                   round(max(p["x"] for p in poses), 1)],
                             "y": [round(min(p["y"] for p in poses), 1),
                                   round(max(p["y"] for p in poses), 1)]}
                            if poses else None),
            "ego_spawned_at": {"x": round(float(spawn.location.x), 3),
                               "y": round(float(spawn.location.y), 3),
                               "z": round(float(spawn.location.z), 3),
                               "yaw": round(float(spawn.rotation.yaw), 2)},
            "used_fallback_spawn": bool(used_fallback),
            "trajectory_idx_at_start": int(self.trajectory_idx),
            "parked_json": getattr(self.args, "parked_json", "") or "",
            "n_parked_requested": len(self.parked_spawn_positions or []),
            "physics_disabled": getattr(self.args, "follow_mode", "") == "teleport",
        }
        try:
            path.write_text(json.dumps(info, indent=2), encoding="utf-8")
            print(f"[toolkit] diagnostica di avvio -> {path}", flush=True)
        except Exception as exc:                           # noqa: BLE001
            print(f"[toolkit] diagnostica non scritta: {exc}", flush=True)

    def _write_parked_validation_report(self, requested, spawned, failures):
        """Where every parked vehicle ended up, and how far it is from the drive.

        The layout that was driven is not the layout on disk: vehicles are
        lifted, settled, and some are pushed out of the ego's lane. Without a
        record of the difference, a scene cannot be checked afterwards and a
        hovering or blocking car is found only by watching the video, which is
        how the last three defects were found.
        """
        path = getattr(self.args, "parked_report_json", "") or ""
        if not path:
            root = getattr(self.args, "dataset_root", "") or "."
            name = getattr(self.args, "scenario_name", "run") or "run"
            path = str(Path(root) / "parked_reports" / f"{name}_parked.json")
        rows = []
        for state in self.parked_actor_states:
            actor = state["actor"]
            try:
                tf = actor.get_transform()
            except Exception:                              # noqa: BLE001
                continue
            lateral, along, _seg = lateral_offset_from_trajectory(
                float(tf.location.x), float(tf.location.y), self.trajectory_poses)
            rows.append({
                "type_id": actor.type_id,
                "final": [round(float(tf.location.x), 3),
                          round(float(tf.location.y), 3),
                          round(float(tf.location.z), 3)],
                "requested": [round(float(v), 3) for v in state["spec"].get("start", [])],
                "yaw": round(float(tf.rotation.yaw), 2),
                "pitch": round(float(tf.rotation.pitch), 2),
                "roll": round(float(tf.rotation.roll), 2),
                "lateral_from_trajectory_m": round(lateral, 3),
                "along_trajectory_m": round(along, 1),
            })
        lat = [abs(r["lateral_from_trajectory_m"]) for r in rows]
        report = {
            "scenario_name": getattr(self.args, "scenario_name", ""),
            "parked_json": getattr(self.args, "parked_json", ""),
            "requested": requested,
            "spawned": spawned,
            "failed": failures,
            "corrected_into_clearance": len(getattr(self, "parked_lane_corrections", [])),
            "clearance_m": float(getattr(self.args, "parked_clearance_m", 2.5)),
            "avoid_ego_lane": bool(getattr(self.args, "parked_avoid_ego_lane", False)),
            "settle_ticks": int(getattr(self.args, "parked_settle_ticks", 0) or 0),
            "z_offset": float(getattr(self.args, "parked_z_offset", 0.5)),
            "min_lateral_m": round(min(lat), 3) if lat else None,
            "median_lateral_m": round(sorted(lat)[len(lat) // 2], 3) if lat else None,
            "corrections": getattr(self, "parked_lane_corrections", []),
            "vehicles": rows,
        }
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(f"[toolkit] parked_json: report di validazione -> {path}", flush=True)
        except Exception as exc:                           # noqa: BLE001
            print(f"[toolkit] parked_json: report non scritto: {exc}", flush=True)

    def _settle_parked_actors(self):
        """Let the lifted vehicles fall onto the road, then freeze them there.

        A vehicle is spawned above its target height because try_spawn_actor
        refuses a transform whose mesh intersects the road, and its physics is
        switched off immediately afterwards so that nothing pushes it out of
        place. The two together mean the lift never comes off: measured on the
        fortiss layout the gap under the wheels equals the lift exactly, so the
        default 0.5 m leaves every parked car hovering half a metre for the
        whole run, in the clouds as well as on screen.

        Running physics for a few frames first costs those frames and puts the
        wheels on the ground. Default zero, so scenarios that were authored
        against the old behaviour are untouched.
        """
        ticks = int(getattr(self.args, "parked_settle_ticks", 0) or 0)
        actors = [s["actor"] for s in self.parked_actor_states]
        if ticks <= 0 or not actors:
            return
        before = [a.get_location().z for a in actors]
        for a in actors:
            try:
                a.set_simulate_physics(True)
            except Exception:
                pass
        for _ in range(ticks):
            try:
                self.world.tick()
            except Exception:
                break
        for a in actors:
            try:
                a.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                a.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                a.set_simulate_physics(False)
            except Exception:
                pass
        after = [a.get_location().z for a in actors]
        drop = [b - a_ for b, a_ in zip(before, after)]
        print(f"[toolkit] parked_json: assestamento {ticks} tick, "
              f"caduta mediana {sorted(drop)[len(drop) // 2]:.2f} m", flush=True)

    def _spawn_dynamic_actors_from_json(self):
        self.dynamic_actor_states = []
        if not self.dynamic_actor_specs:
            return
        blueprints = get_filtered_vehicle_blueprints(self.world)
        if not blueprints:
            print("[toolkit] dynamic_json: no vehicle blueprints available", flush=True)
            return
        spawned = 0
        failures = 0
        map_spawn_points = []
        try:
            map_spawn_points = list(self.map.get_spawn_points())
        except Exception:
            map_spawn_points = []
        for item in self.dynamic_actor_specs:
            bp = pick_parked_blueprint(blueprints, item, self.world.get_blueprint_library())
            transform = parked_transform_from_entry(
                self.world,
                item,
                float(getattr(self.args, "dynamic_z_offset", 0.5)),
            )
            actor = self.world.try_spawn_actor(bp, transform)
            if actor is None and map_spawn_points:
                for fallback_tf in map_spawn_points:
                    actor = self.world.try_spawn_actor(bp, fallback_tf)
                    if actor is not None:
                        try:
                            actor.set_transform(transform)
                            print(
                                "[toolkit] dynamic_json: spawned via fallback and teleported "
                                f"to ({transform.location.x:.2f}, {transform.location.y:.2f})",
                                flush=True,
                            )
                        except Exception as exc:
                            print(
                                f"[toolkit] dynamic_json: fallback teleport failed ({exc})",
                                flush=True,
                            )
                            try:
                                actor.destroy()
                            except Exception:
                                pass
                            actor = None
                        break
            if actor is None:
                failures += 1
                continue
            try:
                actor.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                actor.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                actor.set_simulate_physics(False)
            except Exception:
                pass
            light_state = parse_vehicle_light_state(item)
            if light_state is not None:
                try:
                    actor.set_light_state(light_state)
                except Exception:
                    pass
            self.actors.append(actor)
            trigger_point = item.get("trigger_point") or item.get("start") or [0.0, 0.0, 0.0]
            heading = float(item.get("motion_heading", item.get("heading", 0.0)))
            speed = float(item.get("target_speed_mps", item.get("speed_mps", 3.0)))
            duration = float(item.get("active_duration_s", 6.0))
            segment_specs = item.get("motion_segments") or []
            if segment_specs:
                segments = []
                for seg in segment_specs:
                    segments.append(
                        {
                            "heading_deg": float(seg.get("heading", seg.get("motion_heading", heading))),
                            "speed_mps": float(seg.get("speed_mps", seg.get("target_speed_mps", speed))),
                            "duration_s": float(seg.get("duration_s", seg.get("active_duration_s", duration))),
                        }
                    )
            else:
                segments = [
                    {
                        "heading_deg": heading,
                        "speed_mps": speed,
                        "duration_s": duration,
                    }
                ]
            total_duration = float(sum(max(0.0, float(seg["duration_s"])) for seg in segments))
            self.dynamic_actor_states.append(
                {
                    "actor": actor,
                    "spec": item,
                    "trigger_point": np.array(
                        [
                            float(trigger_point[0]),
                            float(trigger_point[1]),
                            float(trigger_point[2]) if len(trigger_point) > 2 else 0.0,
                        ],
                        dtype=np.float32,
                    ),
                    "trigger_distance_m": float(item.get("trigger_distance_m", 14.0)),
                    "heading_deg": heading,
                    "speed_mps": speed,
                    "active_duration_s": total_duration,
                    "motion_segments": segments,
                    "current_segment_idx": None,
                    "triggered": False,
                    "done": False,
                    "start_time_s": None,
                }
            )
            if getattr(self.args, "hazard_obstacle_source", "none") == "first_dynamic" and self.hazard_obstacle_actor is None:
                self.hazard_obstacle_actor = actor
                loc = actor.get_location()
                self.hazard_obstacle_position = (float(loc.x), float(loc.y), float(loc.z))
            spawned += 1
        print(f"[toolkit] dynamic_json: spawned {spawned}/{len(self.dynamic_actor_specs)} failed={failures}", flush=True)

    def _spawn_walkers_from_json(self):
        """Pedestrians, driven by WalkerControl rather than by target velocity.

        Vehicles in this harness are moved by setting a target velocity, which
        is right for a rigid body. A pedestrian is not one: driven that way the
        mesh slides along the ground in its idle pose, and the LiDAR sees a
        rigid shape where a real return comes off swinging limbs. WalkerControl
        is CARLA's own interface for them and animates the skeleton, so the
        cloud carries the structure a pedestrian actually has, which is the
        whole reason for putting one in a sensing study.

        Walkers are also spawned NOT invincible: a collision that the harness
        cannot register is a failure it cannot report.
        """
        if not self.walker_specs:
            return
        library = self.world.get_blueprint_library()
        spawned, failures = 0, 0
        for item in self.walker_specs:
            bp_id = str(item.get("blueprint_id", "")).strip()
            # Never a random choice: picking a blueprint from the RNG would tie
            # the scene to a stream that advances differently in each sensing
            # arm, and the three would stop being comparable. Named, or the
            # first of the library in its own order.
            candidates = list(library.filter("walker.pedestrian.*"))
            if not candidates:
                print("[toolkit] walkers: nessun blueprint pedone disponibile",
                      flush=True)
                return
            try:
                bp = library.find(bp_id) if bp_id else candidates[0]
            except Exception:
                print(f"[toolkit] walkers: blueprint '{bp_id}' non trovato, "
                      f"uso {candidates[0].id}", flush=True)
                bp = candidates[0]
            if bp.has_attribute("is_invincible"):
                bp.set_attribute("is_invincible", "false")
            raw = item["start"]
            tf = carla.Transform(
                carla.Location(x=float(raw[0]), y=float(raw[1]),
                               z=float(raw[2]) + float(getattr(self.args, "walker_z_offset", 0.6))),
                carla.Rotation(yaw=float(item.get("heading", 0.0))),
            )
            actor = self.world.try_spawn_actor(bp, tf)
            if actor is None:
                failures += 1
                continue
            segments = [{"heading_deg": float(g.get("heading", item.get("heading", 0.0))),
                         "speed_mps": float(g.get("speed_mps", 1.4)),
                         "duration_s": float(g.get("duration_s", 4.0))}
                        for g in (item.get("motion_segments") or [])]
            if not segments:
                segments = [{"heading_deg": float(item.get("heading", 0.0)),
                             "speed_mps": float(item.get("speed_mps", 1.4)),
                             "duration_s": float(item.get("active_duration_s", 4.0))}]
            trig = item.get("trigger_point") or item.get("start")
            if (getattr(self.args, "hazard_obstacle_source", "none") == "first_walker"
                    and self.hazard_obstacle_actor is None):
                self.hazard_obstacle_actor = actor
                loc = actor.get_location()
                self.hazard_obstacle_position = (float(loc.x), float(loc.y), float(loc.z))
            self.actors.append(actor)
            self.walker_states.append({
                "actor": actor,
                "trigger_point": (float(trig[0]), float(trig[1])),
                "trigger_distance_m": float(item.get("trigger_distance_m", 12.0)),
                "motion_segments": segments,
                "active_duration_s": sum(g["duration_s"] for g in segments),
                "triggered": False, "done": False, "start_time_s": None,
            })
            spawned += 1
        print(f"[toolkit] walkers: spawned {spawned}/{len(self.walker_specs)} "
              f"failed={failures}", flush=True)

    def _update_walkers(self):
        if self.vehicle is None or not self.walker_states:
            return
        ego = self.vehicle.get_location()
        now = float(self.world.get_snapshot().timestamp.elapsed_seconds)
        for st in self.walker_states:
            actor = st["actor"]
            if st["done"] or actor is None or not actor.is_alive:
                continue
            if not st["triggered"]:
                d = math.hypot(float(ego.x) - st["trigger_point"][0],
                               float(ego.y) - st["trigger_point"][1])
                if d > st["trigger_distance_m"]:
                    continue
                st["triggered"] = True
                st["start_time_s"] = now
                print(f"[toolkit] walker triggered: dist={d:.2f}m", flush=True)
            elapsed = now - float(st["start_time_s"] or now)
            if elapsed >= st["active_duration_s"]:
                try:
                    actor.apply_control(carla.WalkerControl(
                        carla.Vector3D(0.0, 0.0, 0.0), 0.0, False))
                except Exception:
                    pass
                st["done"] = True
                continue
            acc = 0.0
            seg = st["motion_segments"][-1]
            for g in st["motion_segments"]:
                if elapsed < acc + g["duration_s"]:
                    seg = g
                    break
                acc += g["duration_s"]
            h = math.radians(seg["heading_deg"])
            try:
                actor.apply_control(carla.WalkerControl(
                    carla.Vector3D(math.cos(h), math.sin(h), 0.0),
                    float(seg["speed_mps"]), False))
            except Exception:
                pass

    def _spawn_ambient_walkers(self):
        """Background pedestrians that walk, rather than follow a line.

        CARLA has an AI controller for walkers backed by the map's navigation
        mesh: given a destination it routes there, keeps to walkable surfaces,
        steps around obstacles and animates properly. That is what a person in
        the background should do, and scripting them with headings and
        durations would be inventing a worse version of it.

        The scenario's OWN hazard pedestrian is deliberately not one of these.
        A CPNCO test turns on the pedestrian reaching the ego's path at a
        defined moment, and a navigating walker arrives when its route lets it;
        the physical test uses a dummy driven along a rail for the same reason.
        So: ambient pedestrians navigate, the hazard is scripted, and the two
        are different mechanisms on purpose.

        Determinism is pinned with set_pedestrians_seed, which is CARLA's own
        hook for it. Without that the destinations are drawn from a stream that
        differs between runs and the three sensing arms would each get their
        own crowd.
        """
        n = int(getattr(self.args, "ambient_walkers", 0) or 0)
        if n <= 0:
            return
        seed = int(getattr(self.args, "pcla_perturb_seed", 1234))
        try:
            self.world.set_pedestrians_seed(seed)
        except Exception as exc:
            print(f"[toolkit] pedoni ambiente: seme non fissato ({exc}). Le tre "
                  f"configurazioni possono divergere: NON usare per un confronto.",
                  flush=True)

        library = self.world.get_blueprint_library()
        walker_bps = list(library.filter("walker.pedestrian.*"))
        ctrl_bp = library.find("controller.ai.walker")
        if not walker_bps:
            print("[toolkit] pedoni ambiente: nessun blueprint disponibile", flush=True)
            return

        ego = self.vehicle.get_location() if self.vehicle is not None else None
        keep = float(getattr(self.args, "ambient_keep_clear_m", 8.0))
        spawned = 0
        for i in range(n * 4):          # some draws land off the mesh
            if spawned >= n:
                break
            loc = self.world.get_random_location_from_navigation()
            if loc is None:
                continue
            if ego is not None and loc.distance(ego) < keep:
                continue
            bp = walker_bps[i % len(walker_bps)]
            if bp.has_attribute("is_invincible"):
                bp.set_attribute("is_invincible", "false")
            walker = self.world.try_spawn_actor(bp, carla.Transform(loc))
            if walker is None:
                continue
            ctrl = self.world.try_spawn_actor(ctrl_bp, carla.Transform(),
                                              attach_to=walker)
            if ctrl is None:
                walker.destroy()
                continue
            try:
                ctrl.start()
                dest = self.world.get_random_location_from_navigation()
                if dest is not None:
                    ctrl.go_to_location(dest)
                ctrl.set_max_speed(float(getattr(self.args,
                                                 "ambient_walker_speed", 1.4)))
            except Exception:
                ctrl.destroy(); walker.destroy()
                continue
            self.actors.extend([ctrl, walker])
            self.ambient_walkers.append(walker)
            spawned += 1
        print(f"[toolkit] pedoni ambiente: {spawned}/{n} in navigazione "
              f"(seme {seed})", flush=True)

    def _spawn_traffic_manager_vehicles(self):
        """Background traffic driven by CARLA, not by a table of headings.

        The scenario JSON drives its actors open loop: constant heading and
        speed for a duration, no steering, no lane following and no reaction to
        anything. That is adequate for a hazard scripted to cross at a known
        moment, and it is visibly wrong for traffic, which has to hold a lane
        through a bend, stop at a light and brake behind whoever is in front.
        The Traffic Manager already exists here for the ego's autopilot, so
        traffic is handed to it instead.

        DETERMINISM IS THE WHOLE DIFFICULTY. The three sensing configurations
        must see the same scene or their difference is not about sensing, and
        the Traffic Manager is stochastic. Two things are pinned: its random
        device, and the spawn points, which are taken in map order rather than
        sampled. What is NOT pinned, and cannot be from here, is that the
        traffic reacts to the ego -- if the ego brakes earlier under one
        configuration the traffic behind it will too. That is a real coupling
        and it is the point of adding traffic, but it means the three arms are
        no longer bit-comparable and the comparison has to be read as
        behavioural rather than as a controlled perturbation.
        """
        n = int(getattr(self.args, "tm_traffic", 0) or 0)
        if n <= 0:
            return
        seed = int(getattr(self.args, "pcla_perturb_seed", 1234))
        try:
            self.tm.set_random_device_seed(seed)
        except Exception as exc:
            print(f"[toolkit] traffico: seme del Traffic Manager non fissato ({exc}). "
                  f"Le tre configurazioni possono divergere: NON usare per un confronto.",
                  flush=True)

        spawns = self.world.get_map().get_spawn_points()
        ego_loc = self.vehicle.get_location() if self.vehicle is not None else None
        keep_clear = float(getattr(self.args, "tm_keep_clear_m", 12.0))
        if ego_loc is not None:
            spawns = [s for s in spawns
                      if s.location.distance(ego_loc) > keep_clear]
        # map order, never sampled: a shuffled list would tie the layout to the
        # RNG stream, and the stream advances with the number of points in the
        # cloud, which differs between the arms
        blueprints = get_filtered_vehicle_blueprints(self.world)
        if not blueprints or not spawns:
            print("[toolkit] traffico: nessun punto di spawn o blueprint disponibile",
                  flush=True)
            return

        for i, tf in enumerate(spawns):
            if len(self.tm_vehicles) >= n:
                break
            bp = blueprints[i % len(blueprints)]
            actor = self.world.try_spawn_actor(bp, tf)
            if actor is None:
                continue
            try:
                actor.set_autopilot(True, self.tm.get_port())
                self.tm.ignore_lights_percentage(actor, 0.0)
                self.tm.vehicle_percentage_speed_difference(
                    actor, float(getattr(self.args, "tm_speed_delta_pct", 0.0)))
            except Exception:
                actor.destroy()
                continue
            self.actors.append(actor)
            self.tm_vehicles.append(actor)
        print(f"[toolkit] traffico: {len(self.tm_vehicles)}/{n} veicoli in autopilot "
              f"(seme {seed}, punti di spawn in ordine di mappa)", flush=True)

    def _spawn_collision_sensor(self):
        if self.vehicle is None:
            return
        collision_bp = self.blueprints.find("sensor.other.collision")
        collision_tf = carla.Transform(carla.Location(x=0.0, z=0.0))
        self.collision_sensor = self.world.spawn_actor(collision_bp, collision_tf, attach_to=self.vehicle)
        self.actors.append(self.collision_sensor)
        self.collision_sensor.listen(self._on_collision_event)

    def _on_collision_event(self, event):
        try:
            other = getattr(event, "other_actor", None)
            other_id = int(other.id) if other is not None else -1
            other_type = str(other.type_id) if other is not None else ""
        except Exception:
            other_id = -1
            other_type = ""
        self.collision_events.append(
            {
                "frame": int(getattr(event, "frame", -1)),
                "other_actor_id": other_id,
                "other_actor_type": other_type,
            }
        )

    def _update_dynamic_actors(self):
        if self.vehicle is None or not self.dynamic_actor_states:
            return
        ego_loc = self.vehicle.get_location()
        snapshot = self.world.get_snapshot()
        sim_time_s = float(snapshot.timestamp.elapsed_seconds)
        for state in self.dynamic_actor_states:
            actor = state["actor"]
            if state["done"] or actor is None or not actor.is_alive:
                continue
            if not state["triggered"]:
                dx = float(ego_loc.x) - float(state["trigger_point"][0])
                dy = float(ego_loc.y) - float(state["trigger_point"][1])
                dist = math.hypot(dx, dy)
                if dist <= float(state["trigger_distance_m"]):
                    state["triggered"] = True
                    state["start_time_s"] = sim_time_s
                    try:
                        actor.set_simulate_physics(True)
                    except Exception:
                        pass
                    print(
                        f"[toolkit] dynamic_actor triggered: dist={dist:.2f}m "
                        f"speed={state['speed_mps']:.2f} heading={state['heading_deg']:.1f}",
                        flush=True,
                    )
            if not state["triggered"]:
                continue
            elapsed = sim_time_s - float(state["start_time_s"] or sim_time_s)
            if elapsed >= float(state["active_duration_s"]):
                try:
                    actor.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                    actor.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                except Exception:
                    pass
                state["done"] = True
                continue
            segments = state.get("motion_segments") or [
                {
                    "heading_deg": float(state["heading_deg"]),
                    "speed_mps": float(state["speed_mps"]),
                    "duration_s": float(state["active_duration_s"]),
                }
            ]
            seg_elapsed = 0.0
            active_idx = len(segments) - 1
            for idx, seg in enumerate(segments):
                seg_duration = float(seg["duration_s"])
                if elapsed < seg_elapsed + seg_duration:
                    active_idx = idx
                    break
                seg_elapsed += seg_duration
            if state.get("current_segment_idx") != active_idx:
                state["current_segment_idx"] = active_idx
                try:
                    loc = actor.get_transform().location
                    actor.set_transform(
                        carla.Transform(
                            loc,
                            carla.Rotation(yaw=float(segments[active_idx]["heading_deg"])),
                        )
                    )
                except Exception:
                    pass
            heading_rad = math.radians(float(segments[active_idx]["heading_deg"]))
            speed = float(segments[active_idx]["speed_mps"])
            vx = speed * math.cos(heading_rad)
            vy = speed * math.sin(heading_rad)
            try:
                actor.set_target_velocity(carla.Vector3D(vx, vy, 0.0))
            except Exception:
                pass

    def spawn_vehicle_and_sensors(self):
        print("[toolkit] spawn_vehicle_and_sensors: start", flush=True)
        self._destroy_runtime_actors_only()
        print("[toolkit] spawn_vehicle_and_sensors: runtime actors cleared", flush=True)
        self._clear_world_runtime_actors()
        self.hazard_obstacle_actor = None
        self.hazard_obstacle_position = None
        self.first_obstacle_in_range_progress_m = None
        self.collision_events = []
        self.collision_sensor = None
        self.termination_mode = None
        self.termination_reason = ""
        self._last_tick_time_s = None
        self._last_tick_speed_mps = None
        self._max_deceleration_mps2 = 0.0
        self._min_ttc_proxy_s = float("inf")
        self._first_brake_progress_m = None

        veh_bp = self.blueprints.filter(self.args.vehicle_filter)[0]
        if veh_bp.has_attribute("role_name"):
            veh_bp.set_attribute("role_name", "hero")

        spawn = self._resolve_spawn_transform()
        print(
            f"[toolkit] spawn_vehicle_and_sensors: ego spawn candidate "
            f"({spawn.location.x:.2f}, {spawn.location.y:.2f}, {spawn.location.z:.2f}) "
            f"yaw={spawn.rotation.yaw:.2f}",
            flush=True,
        )

        self.vehicle = None
        self.trajectory_idx = 0
        if self.trajectory_poses:
            # try_spawn_actor refuses a transform whose mesh intersects the
            # road, and a pose taken from the road surface puts the vehicle's
            # origin exactly on it. Every pose is then refused, the run falls
            # back to a map spawn point, and CARLA's own spawn points carry a
            # lift for precisely this reason. So attempt with the same lift and
            # keep the unlifted pose for the reset, which happens once physics
            # is off and cannot be refused.
            spawn_lift = float(getattr(self.args, "traj_spawn_lift", 0.3))
            traj_z = float(getattr(self.args, "traj_z_offset", 0.0))
            for idx, pose in enumerate(self.trajectory_poses):
                candidate = pose_to_road_transform(self.world, pose, traj_z + spawn_lift)
                self.vehicle = self.world.try_spawn_actor(veh_bp, candidate)
                if self.vehicle is not None:
                    spawn = pose_to_road_transform(self.world, pose, traj_z)
                    self.trajectory_idx = idx
                    if idx:
                        print(
                            f"[toolkit] trajectory spawn: skipped {idx} initial poses; "
                            f"first spawnable pose is ({spawn.location.x:.2f}, {spawn.location.y:.2f}, {spawn.location.z:.2f})",
                            flush=True,
                        )
                    break
        else:
            self.vehicle = self.world.try_spawn_actor(veh_bp, spawn)
        used_fallback_spawn = False
        if self.vehicle is None:
            # fallback search across spawn points
            spawn_points = self.map.get_spawn_points()
            for sp in spawn_points:
                self.vehicle = self.world.try_spawn_actor(veh_bp, sp)
                if self.vehicle is not None:
                    spawn = sp
                    used_fallback_spawn = True
                    print(
                        "[toolkit] ATTENZIONE: nessuna posa della traiettoria era "
                        "libera; ripiego su un punto di spawn della mappa in "
                        f"({sp.location.x:.2f}, {sp.location.y:.2f}, {sp.location.z:.2f})",
                        flush=True,
                    )
                    break
        if used_fallback_spawn and self.trajectory_poses:
            # The fallback reassigns `spawn`, and the reset below teleports the
            # ego to whatever `spawn` now holds. Left as it was, a run that fell
            # back would freeze the vehicle on an arbitrary map spawn point
            # instead of the trajectory, which on a partial reconstruction can
            # be a point with no ground under it at all.
            spawn = pose_to_road_transform(
                self.world, self.trajectory_poses[0],
                float(getattr(self.args, "traj_z_offset", 0.0)))
            print("[toolkit] il reset riporta comunque l'ego sulla prima posa "
                  "della traiettoria", flush=True)
        if self.vehicle is None:
            raise RuntimeError("Could not spawn vehicle")
        print(f"[toolkit] spawn_vehicle_and_sensors: ego spawned id={self.vehicle.id}", flush=True)
        self._write_startup_diagnostics(spawn, used_fallback_spawn)
        self.actors.append(self.vehicle)
        self._spawn_collision_sensor()
        self.previous_follow_control = None
        if self.trajectory_poses:
            if self.args.follow_mode == "teleport":
                self.vehicle.set_simulate_physics(False)
                print("[toolkit] spawn_vehicle_and_sensors: teleport mode enabled", flush=True)
            self.vehicle.set_transform(spawn)
            self.vehicle.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            self.vehicle.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            print("[toolkit] spawn_vehicle_and_sensors: ego reset to first trajectory pose", flush=True)

        self._spawn_parked_vehicles_from_json()
        print("[toolkit] spawn_vehicle_and_sensors: parked spawn done", flush=True)
        self._spawn_dynamic_actors_from_json()
        if self.dynamic_actor_states:
            print("[toolkit] spawn_vehicle_and_sensors: dynamic actor spawn done", flush=True)
        self._spawn_walkers_from_json()
        self._spawn_ambient_walkers()
        self._spawn_traffic_manager_vehicles()
        self._register_hero_with_provider()
        self._start_srunner_scenario()

        if self.args.autopilot and not getattr(self.args, 'pcla_agent', ''):
            self.vehicle.set_autopilot(True, self.tm.get_port())
            self.tm.ignore_lights_percentage(self.vehicle, 0.0)

        sensor_w = getattr(self.args, 'sensor_width', 0) or 0
        cam_w = int(sensor_w) if sensor_w > 0 else self.left_panel_w
        cam_h = int(cam_w * self.args.height / self.left_panel_w)
        fov = self.args.cam_fov
        fx = cam_w / (2.0 * math.tan(math.radians(fov) / 2.0))
        self.intr = CameraIntrinsics(
            width=cam_w, height=cam_h, fx=fx, fy=fx, cx=cam_w / 2.0, cy=cam_h / 2.0
        )

        cam_tf = carla.Transform(
            carla.Location(x=1.4, z=1.8),
            carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
        )
        sensor_dt = 1.0 / float(self.args.fps)

        viewer_cameras_disabled = bool(getattr(self.args, "disable_viewer_cameras", False))
        if viewer_cameras_disabled:
            self.rgb_cam = None
            self.sem_cam = None
            self.birdseye_cam = None
            self.rgb_frame = None
            self.sem_frame = None
            self.rgb_array = None
            self.sem_array = None
            self.sem_vis_array = None
            self.birdseye_array = None
            print("[toolkit] spawn_vehicle_and_sensors: viewer cameras disabled", flush=True)
        else:
            rgb_bp = self.blueprints.find("sensor.camera.rgb")
            rgb_bp.set_attribute("image_size_x", str(cam_w))
            rgb_bp.set_attribute("image_size_y", str(cam_h))
            rgb_bp.set_attribute("fov", str(fov))
            rgb_bp.set_attribute("sensor_tick", str(sensor_dt))
            self.rgb_cam = self.world.spawn_actor(rgb_bp, cam_tf, attach_to=self.vehicle)
            self.actors.append(self.rgb_cam)
            self.rgb_cam.listen(self._make_camera_callback(self.rgb_buffer, semantic=False))

            sem_bp = self.blueprints.find("sensor.camera.semantic_segmentation")
            sem_bp.set_attribute("image_size_x", str(cam_w))
            sem_bp.set_attribute("image_size_y", str(cam_h))
            sem_bp.set_attribute("fov", str(fov))
            sem_bp.set_attribute("sensor_tick", str(sensor_dt))
            self.sem_cam = self.world.spawn_actor(sem_bp, cam_tf, attach_to=self.vehicle)
            self.actors.append(self.sem_cam)
            self.sem_cam.listen(self._make_camera_callback(self.sem_buffer, semantic=True))

        if getattr(self.args, "lmdrive_rig", False):
            self._spawn_lmdrive_rig(sensor_dt)

        lidar_bp = self.blueprints.find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("channels", str(self.args.channels))
        lidar_bp.set_attribute("range", str(self.args.lidar_range))
        lidar_bp.set_attribute("points_per_second", str(self.args.pps))
        lidar_bp.set_attribute("rotation_frequency", str(float(self.args.fps)))
        lidar_bp.set_attribute("upper_fov", str(self.args.upper_fov))
        lidar_bp.set_attribute("lower_fov", str(self.args.lower_fov))
        lidar_bp.set_attribute("sensor_tick", str(sensor_dt))
        if lidar_bp.has_attribute("horizontal_fov"):
            lidar_bp.set_attribute("horizontal_fov", str(self.args.horizontal_fov))
        lidar_tf = carla.Transform(carla.Location(x=0.0, z=2.2))
        self.lidar = self.world.spawn_actor(lidar_bp, lidar_tf, attach_to=self.vehicle)
        self.actors.append(self.lidar)
        self.lidar.listen(self._make_lidar_callback(self.lidar_buffer))

        if self.args.enable_semantic_lidar:
            semantic_lidar_bp = self.blueprints.find("sensor.lidar.ray_cast_semantic")
            semantic_lidar_bp.set_attribute("channels", str(self.args.channels))
            semantic_lidar_bp.set_attribute("range", str(self.args.lidar_range))
            semantic_lidar_bp.set_attribute("points_per_second", str(self.args.pps))
            semantic_lidar_bp.set_attribute("rotation_frequency", str(float(self.args.fps)))
            semantic_lidar_bp.set_attribute("upper_fov", str(self.args.upper_fov))
            semantic_lidar_bp.set_attribute("lower_fov", str(self.args.lower_fov))
            semantic_lidar_bp.set_attribute("sensor_tick", str(sensor_dt))
            if semantic_lidar_bp.has_attribute("horizontal_fov"):
                semantic_lidar_bp.set_attribute("horizontal_fov", str(self.args.horizontal_fov))
            self.semantic_lidar = self.world.spawn_actor(semantic_lidar_bp, lidar_tf, attach_to=self.vehicle)
            self.actors.append(self.semantic_lidar)
            self.semantic_lidar.listen(self._make_semantic_lidar_callback(self.semantic_lidar_buffer))
        else:
            self.semantic_lidar = None
            self.last_semantic_lidar = None
            self.semantic_lidar_frame = None

        if not viewer_cameras_disabled:
            birdseye_h_m = float(getattr(self.args, "birdseye_height", 18.0))
            bev_cam_w = cam_w
            bev_cam_h = int(cam_h * 0.40)
            birdseye_bp = self.blueprints.find("sensor.camera.rgb")
            birdseye_bp.set_attribute("image_size_x", str(bev_cam_w))
            birdseye_bp.set_attribute("image_size_y", str(bev_cam_h))
            birdseye_bp.set_attribute("fov", "90")
            birdseye_bp.set_attribute("sensor_tick", str(sensor_dt))
            birdseye_tf = carla.Transform(
                carla.Location(x=0.0, z=birdseye_h_m),
                carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0),
            )
            self.birdseye_cam = self.world.spawn_actor(birdseye_bp, birdseye_tf, attach_to=self.vehicle)
            self.actors.append(self.birdseye_cam)
            self.birdseye_cam.listen(self._make_camera_callback(self.birdseye_buffer, semantic=False))

        # Written here rather than at construction: the transforms have to be
        # read off the live sensors, which do not exist until now.
        if self.recorder is not None:
            try:
                self.recorder.write_calibration(
                    self.intr,
                    {'rgb_camera': self.rgb_cam, 'semantic_camera': self.sem_cam,
                     'lidar': self.lidar, 'semantic_lidar': self.semantic_lidar},
                    extra={'ego_type_id': getattr(self.vehicle, 'type_id', None),
                           'ego_extent': (lambda bb: {'x': float(bb.extent.x),
                                                      'y': float(bb.extent.y),
                                                      'z': float(bb.extent.z)})(
                               self.vehicle.bounding_box) if self.vehicle is not None else None},
                )
            except Exception as exc:
                print(f"[toolkit] calibrazione non scritta: {exc!r}", flush=True)

        self._apply_weather()
        print(
            f"[toolkit] spawn_vehicle_and_sensors: sensors spawned, priming ticks dt={sensor_dt:.4f} "
            f"lidar_hz={float(self.args.fps):.1f}",
            flush=True,
        )

        # Prime sensors before the first PCLA action. Some agents time out if the
        # first full sensor bundle arrives a few ticks late after actor spawn.
        prime_ticks = int(max(2, getattr(self.args, "pcla_sensor_prime_ticks", 4)))
        for _ in range(prime_ticks):
            self._advance_world_once()
        print("[toolkit] spawn_vehicle_and_sensors: ready", flush=True)
        self._drain_buffers()
        snapshot = self.world.get_snapshot()
        self.recording_start_time = float(snapshot.timestamp.elapsed_seconds)

    def request_respawn(self, reason: str, force: bool = False):
        now = time.monotonic()
        if not force and (now - self.last_respawn_time) < self.min_respawn_interval:
            print(
                f"[toolkit] respawn ignored: reason={reason} cooldown={self.min_respawn_interval:.1f}s",
                flush=True,
            )
            return
        self.last_respawn_time = now
        print(f"[toolkit] respawn requested: reason={reason}", flush=True)
        self.spawn_vehicle_and_sensors()

    def _advance_world_once(self, timeout_s: float = 2.0):
        settings = self.world.get_settings()
        if settings.synchronous_mode:
            return self.world.tick()
        snapshot = self.world.wait_for_tick(seconds=timeout_s)
        if snapshot is None:
            raise RuntimeError(
                f"Timed out waiting for async CARLA tick after {timeout_s:.1f}s during sensor priming"
            )
        return snapshot.frame

    def _update_spectator_follow(self):
        mode = str(getattr(self.args, "spectator_follow", "off")).strip().lower()
        if mode == "off" or self.vehicle is None or not self.vehicle.is_alive or self.spectator is None:
            return
        try:
            self.spectator.set_transform(spectator_transform_from_vehicle(self.vehicle.get_transform(), mode))
        except Exception:
            pass

    def _destroy_runtime_actors_only(self):
        for actor in self.actors[::-1]:
            try:
                if "sensor." in actor.type_id and actor.is_alive:
                    actor.stop()
            except Exception:
                pass
            try:
                actor.destroy()
            except Exception:
                pass
        self.actors = []
        self.vehicle = None
        self.rgb_cam = None
        self.sem_cam = None
        self.lidar = None
        self.rgb_array = None
        self.birdseye_array = None
        self.birdseye_cam = None
        self.sem_array = None
        self.sem_vis_array = None
        self.last_lidar = None

    def _clear_world_runtime_actors(self):
        destroy_ids = []
        for actor in self.world.get_actors().filter("vehicle.*"):
            destroy_ids.append(actor.id)
        for actor in self.world.get_actors().filter("sensor.*"):
            destroy_ids.append(actor.id)
        if not destroy_ids:
            print("[toolkit] world cleanup: no stale vehicles/sensors", flush=True)
            return
        commands = [carla.command.DestroyActor(actor_id) for actor_id in destroy_ids]
        results = self.client.apply_batch_sync(commands, True)
        destroyed = 0
        for result in results:
            if getattr(result, "error", None):
                continue
            destroyed += 1
        print(
            f"[toolkit] world cleanup: destroyed {destroyed}/{len(destroy_ids)} stale vehicles/sensors",
            flush=True,
        )

    def _apply_weather(self):
        preset = self.args.weather.lower()
        cam_degrade = getattr(self.args, "camera_degrade", False)
        if preset == "nominal":
            weather = carla.WeatherParameters(
                cloudiness=15.0,
                precipitation=0.0,
                precipitation_deposits=0.0,
                wetness=0.0,
                fog_density=0.0,
                sun_altitude_angle=35.0,
            )
        elif preset == "rain":
            # fog_density=15 and precipitation_deposits=80 reduce RGB camera
            # visibility so that Transfuser relies more on LiDAR.
            # --camera-degrade boosts fog further (density=25, dist=40m).
            weather = carla.WeatherParameters(
                cloudiness=75.0,
                precipitation=75.0,
                precipitation_deposits=80.0,
                wetness=85.0,
                fog_density=25.0 if cam_degrade else 15.0,
                fog_distance=40.0 if cam_degrade else 80.0,
                fog_falloff=0.25 if cam_degrade else 0.15,
                sun_altitude_angle=25.0,
            )
        elif preset == "snow":
            # CARLA does not expose native snowfall in WeatherParameters. This
            # preset is kept visually distinct from rain: no rain particles,
            # very overcast sky, maximum surface deposits, low wetness, cold
            # low-angle light, wind, and a pale winter haze.
            # With --camera-degrade: boost fog density/falloff further.
            weather = carla.WeatherParameters(
                cloudiness=100.0,
                precipitation=0.0,
                precipitation_deposits=100.0,
                wetness=5.0,
                wind_intensity=45.0,
                fog_density=55.0 if cam_degrade else 34.0,
                fog_distance=22.0 if cam_degrade else 38.0,
                fog_falloff=0.35 if cam_degrade else 0.12,
                scattering_intensity=1.35,
                mie_scattering_scale=0.015,
                rayleigh_scattering_scale=0.06,
                sun_altitude_angle=3.0,
            )
        else:
            weather = getattr(carla.WeatherParameters, "ClearNoon")
        self.world.set_weather(weather)

    def _spawn_lmdrive_rig(self, sensor_dt):
        """LMDrive's four cameras, with its own positions and fields of view.

        Read off its sensors(): front at x=1.3 straight ahead 1200x900, left
        and right at the same x at minus and plus sixty degrees, rear at
        x=-1.3 turned by
        one hundred and eighty, all at fov 100. The fifth view the model
        receives, rgb_center, is not recorded: it is derived by cropping the
        front view, which is what the agent does.

        The images are written straight from the callback, with CARLA's frame
        number in the name: that is the same key the clouds are saved under, so
        the offline pairing is exact and they need not go through the buffer
        pipeline.
        """
        import cv2
        # The directories belong to DatasetRecorder, not to the viewer: the
        # viewer reaches them through self.recorder, which exists only with
        # --save-dataset. Without it the rig would have nowhere to write, and
        # above all there would be no clouds to pair it with.
        rec = getattr(self, "recorder", None)
        if rec is None or not hasattr(rec, "lmdrive_rgb_dir"):
            print("[toolkit] rig di LMDrive richiesto ma il dataset non viene "
                  "salvato: niente telecamere, servirebbe --save-dataset",
                  flush=True)
            return
        root = rec.lmdrive_rgb_dir
        rig = [("rgb_front", 1.3, 0.0, 1200, 900),
               ("rgb_left", 1.3, -60.0, 400, 300),
               ("rgb_right", 1.3, 60.0, 400, 300),
               ("rgb_rear", -1.3, 180.0, 400, 300)]

        def _writer(folder):
            def _cb(image):
                arr = np.frombuffer(image.raw_data, dtype=np.uint8)
                arr = arr.reshape((image.height, image.width, 4))[:, :, :3]
                cv2.imwrite(str(folder / f"frame_{image.frame:06d}.png"), arr)
            return _cb

        for view, x, yaw, w, h in rig:
            bp = self.blueprints.find("sensor.camera.rgb")
            bp.set_attribute("image_size_x", str(w))
            bp.set_attribute("image_size_y", str(h))
            bp.set_attribute("fov", "100")
            bp.set_attribute("sensor_tick", str(sensor_dt))
            tf = carla.Transform(carla.Location(x=x, z=2.3),
                                 carla.Rotation(yaw=yaw))
            cam = self.world.spawn_actor(bp, tf, attach_to=self.vehicle)
            self.actors.append(cam)
            cam.listen(_writer(root / view))
        print(f"[toolkit] rig di LMDrive: {len(rig)} viste sotto {root}",
              flush=True)

    @staticmethod
    def _make_camera_callback(buffer_obj: SensorBuffer, semantic=False):
        def _callback(image):
            if semantic:
                raw_arr = np.frombuffer(image.raw_data, dtype=np.uint8).copy()
                raw_arr = raw_arr.reshape((image.height, image.width, 4))
                vis_arr = semantic_raw_to_grayscale_bgra(raw_arr)

                buffer_obj.put(image.frame, {"raw": raw_arr, "vis": vis_arr})
            else:
                arr = parse_image(image)
                buffer_obj.put(image.frame, arr)
        return _callback

    @staticmethod
    def _make_lidar_callback(buffer_obj: SensorBuffer):
        def _callback(measurement):
            pts = np.frombuffer(measurement.raw_data, dtype=np.float32)
            pts = np.reshape(pts, (-1, 4))
            # CARLA LiDAR: x, y, z, intensity
            buffer_obj.put(measurement.frame, pts)
        return _callback

    @staticmethod
    def _make_semantic_lidar_callback(buffer_obj: SensorBuffer):
        semantic_dtype = np.dtype(
            [
                ("x", np.float32),
                ("y", np.float32),
                ("z", np.float32),
                ("CosAngle", np.float32),
                ("ObjIdx", np.uint32),
                ("ObjTag", np.uint32),
            ]
        )

        def _callback(measurement):
            pts = np.frombuffer(measurement.raw_data, dtype=semantic_dtype).copy()
            buffer_obj.put(measurement.frame, pts)

        return _callback

    def _drain_buffers(self):
        rgb = self.rgb_buffer.get_latest()
        if rgb is not None:
            self.rgb_frame, self.rgb_array = rgb

        sem = self.sem_buffer.get_latest()
        if sem is not None:
            self.sem_frame, sem_payload = sem
            if isinstance(sem_payload, dict):
                self.sem_array = sem_payload.get("raw")
                self.sem_vis_array = sem_payload.get("vis")
            else:
                self.sem_array = sem_payload
                self.sem_vis_array = sem_payload

        lidar = self.lidar_buffer.get_latest()
        if lidar is not None:
            self.lidar_frame, self.last_lidar = lidar

        semantic_lidar = self.semantic_lidar_buffer.get_latest()
        if semantic_lidar is not None:
            self.semantic_lidar_frame, self.last_semantic_lidar = semantic_lidar

        birdseye = self.birdseye_buffer.get_latest()
        if birdseye is not None:
            _, self.birdseye_array = birdseye

    def _update_display_from_synced_frame(self, target_frame: int | None = None):
        synced = None
        if target_frame is not None:
            synced = self._wait_for_exact_frame(target_frame)
        if synced is None:
            synced = self._get_synchronized_frame()
        if synced is None:
            return False

        frame_id, rgb_payload, sem_payload, lidar_payload, semantic_lidar_payload = synced
        self.rgb_frame = frame_id
        self.rgb_array = rgb_payload
        self.lidar_frame = frame_id
        self.last_lidar = lidar_payload
        # Open-loop replay uses the recorded ego pose but still needs the same
        # sensor operator as a PCLA run. Apply it at the synchronized sensor
        # boundary, before projection and dataset serialization.
        if getattr(self.args, "open_loop_matsense", False) and self._pcla_session is not None:
            self.last_lidar = self._pcla_session.get_perturbed_cloud(
                lidar_payload, frame=frame_id
            )
            lidar_payload = self.last_lidar
        self.semantic_lidar_frame = frame_id
        self.last_semantic_lidar = semantic_lidar_payload
        self.sem_frame = frame_id
        if isinstance(sem_payload, dict):
            self.sem_array = sem_payload.get("raw")
            self.sem_vis_array = sem_payload.get("vis")
        else:
            self.sem_array = sem_payload
            self.sem_vis_array = sem_payload
        if self.last_lidar is not None and self.sem_array is not None:
            self.last_color_info = self.compute_color_values(
                self.last_lidar,
                semantic_image=self.sem_array,
                semantic_lidar=self.last_semantic_lidar,
            )
        else:
            self.last_color_info = None
        return True

    def _get_synchronized_frame(self):
        common = (
            self.rgb_buffer.available_frames()
            & self.sem_buffer.available_frames()
            & self.lidar_buffer.available_frames()
        )
        if self.semantic_lidar is not None:
            common &= self.semantic_lidar_buffer.available_frames()
        if not common:
            latest_candidates = [f for f in (self.rgb_frame, self.sem_frame, self.lidar_frame) if f is not None]
            if latest_candidates:
                min_latest = min(int(f) for f in latest_candidates)
                self.rgb_buffer.discard_older_than(min_latest)
                self.sem_buffer.discard_older_than(min_latest)
                self.lidar_buffer.discard_older_than(min_latest)
            return None

        frame_id = max(common)
        rgb = self.rgb_buffer.get(frame_id)
        sem = self.sem_buffer.get(frame_id)
        lidar = self.lidar_buffer.get(frame_id)
        semantic_lidar = self.semantic_lidar_buffer.get(frame_id) if self.semantic_lidar is not None else None
        if rgb is None or sem is None or lidar is None:
            return None
        return int(frame_id), rgb, sem, lidar, semantic_lidar

    def _wait_for_exact_frame(self, frame_id: int):
        deadline = time.monotonic() + max(0.05, float(getattr(self.args, "strict_sync_timeout", 0.5)))
        target = int(frame_id)
        while time.monotonic() < deadline:
            self._drain_buffers()
            rgb = self.rgb_buffer.get(target)
            sem = self.sem_buffer.get(target)
            lidar = self.lidar_buffer.get(target)
            semantic_lidar = self.semantic_lidar_buffer.get(target) if self.semantic_lidar is not None else None
            if rgb is not None and sem is not None and lidar is not None:
                return target, rgb, sem, lidar, semantic_lidar
            time.sleep(0.001)
        return None

    def project_lidar_to_camera(self, xyz_lidar: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Project lidar points in sensor frame into the semantic camera image.
        Assumes camera and lidar share the same vehicle rigid body and uses world transforms.
        Returns:
          uv_valid : Nx2 integer pixel coordinates for valid projected points
          valid_mask : boolean mask aligned to xyz_lidar rows
        """
        if xyz_lidar.size == 0 or self.sem_cam is None or self.intr is None:
            return np.empty((0, 2), dtype=np.int32), np.zeros((xyz_lidar.shape[0],), dtype=bool)

        # lidar local -> world
        lidar_tf = self.lidar.get_transform()
        cam_tf = self.sem_cam.get_transform()

        lidar_to_world = np.array(lidar_tf.get_matrix())
        world_to_cam = np.array(cam_tf.get_inverse_matrix())

        pts_h = np.concatenate([xyz_lidar, np.ones((xyz_lidar.shape[0], 1), dtype=np.float32)], axis=1)
        pts_world = (lidar_to_world @ pts_h.T).T
        pts_cam_ue = (world_to_cam @ pts_world.T).T[:, :3]

        # Unreal -> conventional camera coordinates
        # UE4 camera axes: X forward, Y right, Z up
        # Standard camera: z forward, x right, y down
        x_ue = pts_cam_ue[:, 0]
        y_ue = pts_cam_ue[:, 1]
        z_ue = pts_cam_ue[:, 2]
        z = x_ue
        x = y_ue
        y = -z_ue

        eps = 1e-6
        valid = z > eps
        u = self.intr.fx * (x[valid] / z[valid]) + self.intr.cx
        v = self.intr.fy * (y[valid] / z[valid]) + self.intr.cy

        in_img = (
            (u >= 0) & (u < self.intr.width) &
            (v >= 0) & (v < self.intr.height)
        )

        valid_indices = np.where(valid)[0][in_img]
        uv = np.stack([u[in_img], v[in_img]], axis=1).astype(np.int32)
        final_mask = np.zeros((xyz_lidar.shape[0],), dtype=bool)
        final_mask[valid_indices] = True
        return uv, final_mask

    @staticmethod
    def make_projection_debug_image(semantic_vis_bgra: np.ndarray | None, uv: np.ndarray) -> np.ndarray | None:
        if semantic_vis_bgra is None:
            return None
        debug = semantic_vis_bgra.copy()
        if uv.size == 0:
            return debug

        h, w = debug.shape[:2]
        stride = max(1, len(uv) // 3000)
        for px, py in uv[::stride]:
            x = int(px)
            y = int(py)
            if not (0 <= x < w and 0 <= y < h):
                continue
            debug[y, x, :3] = np.array([0, 0, 255], dtype=np.uint8)
            if x + 1 < w:
                debug[y, x + 1, :3] = np.array([0, 255, 255], dtype=np.uint8)
            if y + 1 < h:
                debug[y + 1, x, :3] = np.array([0, 255, 255], dtype=np.uint8)
        return debug

    def make_rgb_lidar_overlay(self, rgb_bgra: np.ndarray | None, color_info: dict, mode: str) -> np.ndarray | None:
        if rgb_bgra is None:
            return None

        overlay = rgb_bgra.copy()
        uv = color_info.get("uv")
        if uv is None or uv.size == 0:
            return overlay

        colors = values_to_mode_colors(
            mode,
            color_info["intensity_norm"],
            color_info["pseudo_norm"],
            color_info["materials"],
            color_info["material_effect"],
        )
        h, w = overlay.shape[:2]
        stride = max(1, len(uv) // 4000)
        projected_render_mask = color_info.get("projected_render_mask")
        if mode != "material" or projected_render_mask is None or len(projected_render_mask) != len(uv):
            projected_render_mask = np.ones((len(uv),), dtype=bool)
        uv_filtered = uv[projected_render_mask]
        if uv_filtered.size == 0:
            return overlay
        projected_colors = colors[color_info["projected_indices"]][projected_render_mask]
        uv_sub = uv_filtered[::stride]
        color_sub = projected_colors[::stride]

        for (px, py), rgb in zip(uv_sub, color_sub):
            x = int(px)
            y = int(py)
            if not (0 <= x < w and 0 <= y < h):
                continue
            bgr = np.array([rgb[2], rgb[1], rgb[0]], dtype=np.uint8)
            overlay[y, x, :3] = bgr
            if x + 1 < w:
                overlay[y, x + 1, :3] = bgr
            if y + 1 < h:
                overlay[y + 1, x, :3] = bgr
        return overlay

    def _phase_add(self, name: str, dt: float) -> None:
        """Accumulate a phase of the main loop.

        The campaign runs at about a twentieth of real time and nothing said
        where that went. Guessing from the sensor inventory is not evidence, and
        the process cannot be attached to with py-spy under snap confinement, so
        the loop reports on itself. These are prints and counters only.
        """
        acc = getattr(self, "_phase_acc", None)
        if acc is None:
            acc = self._phase_acc = {}
        acc[name] = acc.get(name, 0.0) + float(dt)

    def _phase_report(self, loop_count: int, every: int = 40) -> None:
        now = time.perf_counter()
        prev = getattr(self, "_phase_last_t", None)
        self._phase_last_t = now
        if prev is not None:
            self._phase_add("_giro", now - prev)
        if loop_count == 0 or loop_count % every != 0:
            return
        acc = getattr(self, "_phase_acc", {}) or {}
        total = acc.get("_giro", 0.0)
        n = getattr(self, "_phase_n", 0) or 1
        named = {k: v for k, v in acc.items() if not k.startswith("_")}
        # what is left over is the drawing, the event pump and the pygame clock:
        # reported rather than dropped, so the parts sum to the whole
        rest = max(total - sum(named.values()), 0.0)
        span = loop_count - getattr(self, "_phase_last_loop", 0)
        span = max(span, 1)
        parts = " ".join(
            f"{k}={v / span * 1000:.0f}ms({v / max(total, 1e-9) * 100:.0f}%)"
            for k, v in sorted(named.items(), key=lambda kv: -kv[1]))
        print(f"[toolkit] tempi loop={loop_count} giro={total / span * 1000:.0f}ms  "
              f"{parts} resto={rest / span * 1000:.0f}ms"
              f"({rest / max(total, 1e-9) * 100:.0f}%)", flush=True)
        self._phase_acc = {}
        self._phase_last_loop = loop_count
        self._phase_n = n

    def _agent_cloud_snapshot(self) -> dict | None:
        """The agent's cloud before and after the operator, if a session owns it.

        Distinct from lidar_raw/ (a different sensor) and from pseudo_final (a
        different quantity, computed with range and incidence corrections and
        without the level renormalisation). This is the pair the closed-loop
        claim rests on, and until now it existed only inside the process.
        """
        sess = getattr(self, "_pcla_session", None)
        c = getattr(sess, "last_cloud", None) if sess is not None else None
        if not c:
            return None
        try:
            out = {
                "xyz_in": c["xyz_in"], "intensity_in": c["intensity_in"],
                "material": np.asarray(c["material"]).astype(str),
                "xyz_out": c["xyz_out"], "intensity_out": c["intensity_out"],
            }
            # The session already recovers the surviving points' materials; not
            # carrying them here left every consumer to redo that work, or to
            # guess. Approximate where jitter moved a point, which is why it is
            # a separate field rather than an index into `material`.
            if c.get("material_out") is not None:
                out["material_out"] = np.asarray(c["material_out"]).astype(str)
            # exactly which returns survived, so a consumer never has to guess
            if c.get("keep") is not None:
                out["keep"] = np.asarray(c["keep"], dtype=bool)
            return out
        except Exception:
            return None

    def _collect_ego_state(self) -> dict:
        if self.vehicle is None:
            return {}
        try:
            v = self.vehicle.get_velocity()
            c = self.vehicle.get_control()
            return {'vx': float(v.x), 'vy': float(v.y), 'vz': float(v.z),
                    'speed': float(math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)),
                    'throttle': float(c.throttle), 'steer': float(c.steer),
                    'brake': float(c.brake)}
        except Exception:
            return {}

    def _collect_actor_states(self, ego_transform, radius_m: float = 120.0) -> list:
        """Pose, velocity and box of every vehicle and pedestrian near the ego.

        Cut off well beyond LiDAR range so a consumer can tell an object that
        was out of range from one the sensor missed. The ego itself is excluded:
        its own state is a column of frame_metadata.csv.
        """
        if self.world is None:
            return []
        ego_id = getattr(self.vehicle, 'id', None)
        ego_loc = ego_transform.location
        out = []
        try:
            actors = list(self.world.get_actors().filter('vehicle.*'))
            actors += list(self.world.get_actors().filter('walker.pedestrian.*'))
        except Exception:
            return []
        for a in actors:
            if a.id == ego_id:
                continue
            try:
                tf = a.get_transform()
                loc, rot = tf.location, tf.rotation
                d = math.sqrt((loc.x - ego_loc.x) ** 2 + (loc.y - ego_loc.y) ** 2
                              + (loc.z - ego_loc.z) ** 2)
                if d > radius_m:
                    continue
                v = a.get_velocity()
                bb = a.bounding_box
                out.append({
                    'actor_id': int(a.id), 'type_id': a.type_id,
                    'category': 'pedestrian' if a.type_id.startswith('walker') else 'vehicle',
                    'x': float(loc.x), 'y': float(loc.y), 'z': float(loc.z),
                    'roll': float(rot.roll), 'pitch': float(rot.pitch), 'yaw': float(rot.yaw),
                    'vx': float(v.x), 'vy': float(v.y), 'vz': float(v.z),
                    'speed': float(math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)),
                    'extent_x': float(bb.extent.x), 'extent_y': float(bb.extent.y),
                    'extent_z': float(bb.extent.z),
                    'bbox_offset_x': float(bb.location.x), 'bbox_offset_y': float(bb.location.y),
                    'bbox_offset_z': float(bb.location.z),
                    'distance_to_ego': float(d),
                })
            except Exception:
                continue
        return out

    def compute_color_values(
        self,
        lidar_pts: np.ndarray,
        semantic_image: np.ndarray | None = None,
        semantic_lidar: np.ndarray | None = None,
    ) -> dict:
        xyz = lidar_pts[:, :3]
        intensity = lidar_pts[:, 3].astype(np.float32)

        i_lo, i_hi = np.percentile(intensity, [2, 98]) if intensity.size else (0.0, 1.0)
        intensity_norm = clamp01((intensity - i_lo) / max(i_hi - i_lo, 1e-6))

        dist = np.linalg.norm(xyz, axis=1).astype(np.float32)
        pseudo = empirical_range_corrected(intensity, dist)

        materials = np.full((xyz.shape[0],), DEFAULT_MATERIAL, dtype=object)
        tags_full = np.full((xyz.shape[0],), -1, dtype=np.int32)
        obj_ids = np.zeros((xyz.shape[0],), dtype=np.uint32)
        uv = np.empty((0, 2), dtype=np.int32)
        mask = np.zeros((xyz.shape[0],), dtype=bool)
        projected_indices = np.empty((0,), dtype=np.int32)
        projected_render_mask = np.empty((0,), dtype=bool)
        sem_image = self.sem_array if semantic_image is None else semantic_image
        if sem_image is not None:
            uv, mask = self.project_lidar_to_camera(xyz)
            if uv.size > 0:
                projected_indices = np.where(mask)[0]
                tags = sem_image[uv[:, 1], uv[:, 0], 2].astype(np.int32)
                tags_full[mask] = tags
                mat_vals = np.array([SEMANTIC_TO_MATERIAL.get(int(t), DEFAULT_MATERIAL) for t in tags], dtype=object)
                materials[mask] = mat_vals

        if semantic_lidar is not None and len(semantic_lidar) > 0:
            actor_ids, actor_tags = match_semantic_lidar_metadata(xyz, semantic_lidar)
            obj_ids = actor_ids
            type_override_cache: dict[int, str] = {}
            for idx, actor_id in enumerate(actor_ids):
                actor_id_int = int(actor_id)
                override = self.material_overrides["actor_ids"].get(str(actor_id_int))
                if override is None and actor_id_int != 0:
                    if actor_id_int not in type_override_cache:
                        actor = self.world.get_actor(actor_id_int)
                        if actor is not None:
                            type_override_cache[actor_id_int] = self.material_overrides["type_ids"].get(actor.type_id, "")
                        else:
                            type_override_cache[actor_id_int] = ""
                    override = type_override_cache[actor_id_int] or None
                if override:
                    materials[idx] = override
                    tags_full[idx] = actor_tags[idx]

        uniq_m, cnt_m = np.unique(materials.astype(str), return_counts=True)
        self.last_material_counts = dict(zip(uniq_m.tolist(), cnt_m.tolist()))
        projected_points = int(np.count_nonzero(mask))
        render_mask = known_material_mask(materials)
        if projected_indices.size > 0:
            projected_render_mask = render_mask[projected_indices]
        known_material_points = int(np.count_nonzero(render_mask))
        projected_known_points = int(np.count_nonzero(projected_render_mask))
        total_points = int(xyz.shape[0])
        self.last_projection_stats = {
            "total_points": total_points,
            "projected_points": projected_points,
            "projection_ratio": float(projected_points / max(total_points, 1)),
            "known_material_points": known_material_points,
            "known_material_ratio": float(known_material_points / max(total_points, 1)),
            "rendered_points": projected_known_points,
            "rendered_ratio": float(projected_known_points / max(projected_points, 1)),
        }

        material_base = nominal_material_values(materials)
        if self.args.use_base_nominal:
            pseudo = material_base.copy()
        else:
            dirs = xyz / np.maximum(dist[:, None], 1e-6)
            frontal_cos = np.abs(dirs[:, 0])
            planar_mask = np.array([m in PLANAR_MATERIALS for m in materials], dtype=bool)
            if np.any(planar_mask):
                pseudo[planar_mask] = angle_corrected(pseudo[planar_mask], frontal_cos[planar_mask])
            pseudo = pseudo * material_base

        weather_lookup = WEATHER_RATIO.get(self.args.weather, WEATHER_RATIO['nominal'])
        ratios = np.array([weather_lookup.get(str(m), weather_lookup.get('unknown', 1.0)) for m in materials], dtype=np.float32)
        pseudo = pseudo * ratios
        material_effect = material_base * ratios

        if self.args.display_normalization == "fixed":
            pseudo_norm = compress_for_display(pseudo, vmax=self.args.display_vmax)
            pseudo_vmax = float(self.args.display_vmax)
        else:
            pseudo_norm, pseudo_vmax = compress_for_display_percentile(
                pseudo, percentile=self.args.display_percentile
            )
        self.last_display_stats = {
            "pseudo_vmax": pseudo_vmax,
            "display_mode": self.args.display_normalization,
        }
        return {
            'intensity_norm': intensity_norm,
            'pseudo_norm': pseudo_norm,
            'materials': materials,
            'material_effect': material_effect,
            'semantic_tags': tags_full,
            'obj_ids': obj_ids,
            'render_mask': render_mask,
            'projected_indices': projected_indices,
            'projected_render_mask': projected_render_mask,
            'ranges': dist,
            'pseudo_raw': pseudo,
            'uv': uv,
            'projected_points': projected_points,
            'known_material_points': known_material_points,
            'pseudo_vmax': pseudo_vmax,
        }

    def bev_surface_from_lidar(self, lidar_pts: np.ndarray, mode: str) -> pygame.Surface:
        surf = pygame.Surface((self.right_panel_w, self.args.height))
        surf.fill((10, 10, 10))

        if lidar_pts is None or lidar_pts.size == 0:
            return surf

        xyz = lidar_pts[:, :3]
        color_info = self.compute_color_values(lidar_pts)
        intensity_norm = color_info["intensity_norm"]
        pseudo_norm = color_info["pseudo_norm"]
        material_effect = color_info["material_effect"]
        materials = color_info["materials"]
        if mode == "intensity":
            values = intensity_norm
        elif mode == "global":
            values = pseudo_norm
        elif mode == "material_effect":
            values = normalize_material_effect(material_effect)
        else:
            values = np.zeros_like(intensity_norm)
        colors = values_to_mode_colors(mode, intensity_norm, pseudo_norm, materials, material_effect)
        render_mask = color_info["render_mask"]

        # BEV coordinates: x forward, y right in vehicle frame.
        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]

        # Filter range and height to reduce clutter.
        mask = (
            (x > -self.args.behind_m) & (x < self.args.forward_m) &
            (np.abs(y) < self.args.side_m) &
            (z > self.args.min_z) & (z < self.args.max_z) &
            render_mask
        )
        x = x[mask]
        y = y[mask]
        colors = colors[mask]
        values = values[mask]

        if x.size == 0:
            return surf

        px = ((y + self.args.side_m) / (2.0 * self.args.side_m) * (self.right_panel_w - 1)).astype(np.int32)
        py = ((self.args.forward_m - x) / (self.args.forward_m + self.args.behind_m) * (self.args.height - 1)).astype(np.int32)

        bev = np.zeros((self.right_panel_w, self.args.height, 3), dtype=np.uint8)
        bev[px, py] = colors

        # draw ego footprint
        ego_w = int(self.right_panel_w * (2.2 / (2.0 * self.args.side_m)))
        ego_h = int(self.args.height * (4.8 / (self.args.forward_m + self.args.behind_m)))
        cx = self.right_panel_w // 2
        cy = int(self.args.height * (self.args.forward_m / (self.args.forward_m + self.args.behind_m)))
        x0 = max(cx - ego_w // 2, 0)
        x1 = min(cx + ego_w // 2, self.right_panel_w - 1)
        y0 = max(cy - ego_h // 2, 0)
        y1 = min(cy + ego_h // 2, self.args.height - 1)
        bev[x0:x1, y0:y1] = np.array([255, 255, 255], dtype=np.uint8)

        surf = pygame.surfarray.make_surface(bev)
        return surf

    def draw_hud(self, panel_mode: str, npts: int, layout: dict[str, pygame.Rect]):
        title_rect = layout["title"]
        status_rect = layout["status"]
        controls_rect = layout["controls"]

        draw_card(self.screen, title_rect)
        draw_card(self.screen, status_rect)
        draw_card(self.screen, controls_rect)

        draw_text(self.screen, self.font_title, "CARLA Material-Aware LiDAR Demo", (title_rect.x + 16, title_rect.y + 12))
        subtitle = f"Weather: {self.args.weather}   |   View: {panel_mode}   |   Autopilot: {'ON' if self.args.autopilot else 'OFF'}"
        draw_text_block(self.screen, self.font_small, subtitle, (title_rect.x + 18, title_rect.y + 41), title_rect.w - 36, 2, TEXT_MUTED)

        draw_text(self.screen, self.font_panel, "Runtime Status", (status_rect.x + 16, status_rect.y + 14))
        status_rows = [
            ("Response model", "Base nominal" if self.args.use_base_nominal else "Empirical + material priors"),
            ("Display scaling", f"{self.last_display_stats['display_mode']}  (vmax {self.last_display_stats['pseudo_vmax']:.3f})"),
            ("LiDAR points", f"{npts:,}"),
            ("Projected points", f"{self.last_projection_stats['projected_points']:,}  ({self.last_projection_stats['projection_ratio']:.1%})"),
            ("Visible materials", f"{self.last_projection_stats['rendered_points']:,}  ({self.last_projection_stats['rendered_ratio']:.1%})"),
        ]
        y = status_rect.y + 50
        value_x = status_rect.x + 150
        value_width = status_rect.right - value_x - 16
        for label, value in status_rows:
            draw_text(self.screen, self.font_small, label, (status_rect.x + 16, y), TEXT_MUTED)
            block_h = draw_text_block(self.screen, self.font_small, value, (value_x, y), value_width, 2, TEXT_MAIN)
            y += max(23, block_h + 2)

        top_materials_y = y + 4
        draw_text(self.screen, self.font_panel, "Top Materials", (status_rect.x + 16, top_materials_y))
        y = top_materials_y + 26
        merged_counts: dict[str, int] = {}
        for raw_name, cnt in self.last_material_counts.items():
            name = visual_material_name(raw_name)
            if name.lower() == "unknown":
                continue
            merged_counts[name] = merged_counts.get(name, 0) + cnt
        for name, cnt in sorted(merged_counts.items(), key=lambda kv: kv[1], reverse=True)[:3]:
            color = tuple(int(c) for c in MATERIAL_DISPLAY_COLORS.get(name, MATERIAL_DISPLAY_COLORS["unknown"]))
            pygame.draw.circle(self.screen, color, (status_rect.x + 24, y + 8), 6)
            draw_text(self.screen, self.font_small, name, (status_rect.x + 38, y), TEXT_MAIN)
            draw_text(self.screen, self.font_small, f"{cnt:,}", (status_rect.right - 60, y), TEXT_MUTED)
            y += 18

        draw_text(self.screen, self.font_panel, "Controls", (controls_rect.x + 16, controls_rect.y + 12))
        controls = [
            "P  cycle view",
            "R  respawn ego vehicle",
            "WASD / arrows  drive",
            "ESC  quit",
        ]
        y = controls_rect.y + 42
        for text in controls:
            block_h = draw_text_block(self.screen, self.font_small, text, (controls_rect.x + 16, y), controls_rect.w - 32, 2, TEXT_MUTED)
            y += max(19, block_h)

        self._draw_colorbar(panel_mode, layout)

    def _draw_colorbar(self, panel_mode: str, layout: dict[str, pygame.Rect]):
        bev_rect = layout["bev"]
        if panel_mode == "material":
            legend_w = 188
            legend_h = min(178, max(118, bev_rect.h - 70))
            legend_rect = pygame.Rect(
                bev_rect.right - legend_w - 14,
                bev_rect.bottom - legend_h - 14,
                legend_w,
                legend_h,
            )
            pygame.draw.rect(self.screen, (12, 16, 24), legend_rect, border_radius=12)
            pygame.draw.rect(self.screen, CARD_BORDER, legend_rect, width=1, border_radius=12)
            x0 = legend_rect.x + 12
            y0 = legend_rect.y + 52
            items = [
                ("asphalt", MATERIAL_DISPLAY_COLORS["asphalt"]),
                ("sidewalk", MATERIAL_DISPLAY_COLORS["sidewalk"]),
                ("building", MATERIAL_DISPLAY_COLORS["building"]),
                ("vegetation", MATERIAL_DISPLAY_COLORS["vegetation"]),
                ("car", MATERIAL_DISPLAY_COLORS["car"]),
            ]
            draw_text(self.screen, self.font_panel, "Material Legend", (legend_rect.x + 12, legend_rect.y + 10), TEXT_MAIN)
            draw_text(self.screen, self.font_small, "Unknown points hidden", (legend_rect.x + 12, legend_rect.y + 32), TEXT_MUTED)
            for idx, (name, color) in enumerate(items):
                yy = y0 + idx * 22
                if yy + 18 > legend_rect.bottom - 10:
                    break
                pygame.draw.rect(self.screen, tuple(int(c) for c in color), pygame.Rect(x0, yy, 18, 18), border_radius=4)
                draw_text(self.screen, self.font_small, name.title(), (x0 + 28, yy - 1), TEXT_MAIN)
            return

        card_w = 86
        card_top = bev_rect.y + 54
        max_card_h = max(72, bev_rect.bottom - card_top - 14)
        h = max(48, min(188, max_card_h - 56))
        w = 18
        card_h = h + 56
        card_rect = pygame.Rect(bev_rect.right - card_w - 14, bev_rect.bottom - card_h - 14, card_w, card_h)
        pygame.draw.rect(self.screen, (12, 16, 24), card_rect, border_radius=12)
        pygame.draw.rect(self.screen, CARD_BORDER, card_rect, width=1, border_radius=12)
        x0 = card_rect.x + 48
        y0 = card_rect.y + 18
        vals = np.linspace(1.0, 0.0, h)
        colors = colormap_intensity(vals) if panel_mode == "intensity" else colormap_pseudo(vals)
        for i in range(h):
            pygame.draw.line(self.screen, tuple(int(c) for c in colors[i]), (x0, y0 + i), (x0 + w, y0 + i))
        pygame.draw.rect(self.screen, TEXT_MUTED, pygame.Rect(x0, y0, w, h), width=1)
        draw_text(self.screen, self.font_small, panel_mode.title(), (card_rect.x + 12, card_rect.y + h + 26), TEXT_MAIN)
        draw_text(self.screen, self.font_small, "high", (card_rect.x + 10, y0 - 4), TEXT_MUTED)
        draw_text(self.screen, self.font_small, "low", (card_rect.x + 16, y0 + h - 10), TEXT_MUTED)

    def _draw_camera_overlay_legend(self, rect: pygame.Rect, mode: str):
        legend_w = 168 if mode == "material" else 96
        legend_h = 142 if mode == "material" else 118
        legend_rect = pygame.Rect(rect.right - legend_w - 18, rect.bottom - legend_h - 18, legend_w, legend_h)
        pygame.draw.rect(self.screen, (12, 16, 24), legend_rect, border_radius=10)
        pygame.draw.rect(self.screen, CARD_BORDER, legend_rect, width=1, border_radius=10)

        if mode == "material":
            draw_text(self.screen, self.font_small, "Material Legend", (legend_rect.x + 10, legend_rect.y + 8), TEXT_MAIN)
            items = [
                ("Asphalt", MATERIAL_DISPLAY_COLORS["asphalt"]),
                ("Sidewalk", MATERIAL_DISPLAY_COLORS["sidewalk"]),
                ("Building", MATERIAL_DISPLAY_COLORS["building"]),
                ("Vegetation", MATERIAL_DISPLAY_COLORS["vegetation"]),
                ("Car", MATERIAL_DISPLAY_COLORS["car"]),
            ]
            y = legend_rect.y + 32
            for name, color in items:
                pygame.draw.rect(self.screen, tuple(int(c) for c in color), pygame.Rect(legend_rect.x + 10, y, 14, 14), border_radius=3)
                draw_text(self.screen, self.font_small, name, (legend_rect.x + 32, y - 2), TEXT_MAIN)
                y += 20
            return

        draw_text(self.screen, self.font_small, "Scale", (legend_rect.x + 10, legend_rect.y + 8), TEXT_MAIN)
        x0 = legend_rect.x + legend_w - 34
        y0 = legend_rect.y + 28
        h = legend_h - 50
        vals = np.linspace(1.0, 0.0, h)
        colors = colormap_intensity(vals) if mode == "intensity" else colormap_pseudo(vals)
        for i in range(h):
            pygame.draw.line(self.screen, tuple(int(c) for c in colors[i]), (x0, y0 + i), (x0 + 16, y0 + i))
        pygame.draw.rect(self.screen, TEXT_MUTED, pygame.Rect(x0, y0, 16, h), width=1)
        draw_text(self.screen, self.font_small, "high", (legend_rect.x + 10, y0 - 3), TEXT_MUTED)
        draw_text(self.screen, self.font_small, "low", (legend_rect.x + 14, y0 + h - 11), TEXT_MUTED)

    def _draw_camera_overlay_panel(self, rect: pygame.Rect, color_info: dict | None, mode: str):
        draw_card(self.screen, rect)
        image_rect = pygame.Rect(rect.x + 8, rect.y + 54, rect.w - 16, rect.h - 62)
        if self.rgb_array is not None:
            panel_image = self.rgb_array
            if color_info is not None:
                overlay = self.make_rgb_lidar_overlay(self.rgb_array, color_info, mode)
                if overlay is not None:
                    panel_image = overlay
            rgb_surf = bgra_to_rgb_surface(panel_image)
            scaled = pygame.transform.smoothscale(rgb_surf, (image_rect.w, image_rect.h))
            self.screen.blit(scaled, (image_rect.x, image_rect.y))
            self._draw_camera_overlay_legend(image_rect, mode)
        else:
            draw_text(self.screen, self.font_small, "Waiting for RGB sensor...", (image_rect.x + 10, image_rect.y + 10), TEXT_MUTED)

        draw_text(self.screen, self.font_panel, mode_title(mode), (rect.x + 16, rect.y + 12), TEXT_MAIN)
        draw_text_block(
            self.screen,
            self.font_small,
            mode_description(mode),
            (rect.x + 16, rect.y + 36),
            rect.w - 32,
            1,
            TEXT_MUTED,
        )

    def draw_triple_camera_view(self, color_info: dict | None):
        margin = 18
        gap = 14
        title_h = 62
        title_rect = pygame.Rect(margin, margin, self.args.width - 2 * margin, title_h)
        draw_card(self.screen, title_rect)
        draw_text(self.screen, self.font_title, "CARLA RGB + LiDAR Overlay Comparison", (title_rect.x + 16, title_rect.y + 12))
        subtitle = (
            f"Weather: {self.args.weather}   |   View: material classes / CARLA raw intensity / MatSense pseudo-reflectance"
        )
        draw_text_block(self.screen, self.font_small, subtitle, (title_rect.x + 18, title_rect.y + 41), title_rect.w - 36, 2, TEXT_MUTED)

        top = title_rect.bottom + 14
        bottom = self.args.height - margin
        panel_h = bottom - top
        panel_w = (self.args.width - 2 * margin - 2 * gap) // 3
        modes = ["material", "intensity", "global"]
        for idx, mode in enumerate(modes):
            x = margin + idx * (panel_w + gap)
            width = panel_w if idx < 2 else self.args.width - margin - x
            self._draw_camera_overlay_panel(pygame.Rect(x, top, width, panel_h), color_info, mode)

        npts = 0 if self.last_lidar is None else int(self.last_lidar.shape[0])
        status = f"LiDAR points: {npts:,}   |   Projected: {self.last_projection_stats['projected_points']:,} ({self.last_projection_stats['projection_ratio']:.1%})   |   P cycles view"
        draw_text(self.screen, self.font_small, status, (margin + 16, bottom - 23), TEXT_MUTED)

    def manual_control(self):
        if self.vehicle is None or self.args.autopilot or self.trajectory_poses:
            return
        keys = pygame.key.get_pressed()
        control = carla.VehicleControl()
        control.throttle = 0.6 if (keys[pygame.K_w] or keys[pygame.K_UP]) else 0.0
        control.brake = 0.8 if (keys[pygame.K_s] or keys[pygame.K_DOWN]) else 0.0
        steer = 0.0
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            steer = -0.45
        elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            steer = 0.45
        control.steer = steer
        control.hand_brake = bool(keys[pygame.K_SPACE])
        control.reverse = bool(keys[pygame.K_q])
        self.vehicle.apply_control(control)

    def follow_trajectory_step(self):
        if self.vehicle is None or not self.trajectory_poses:
            return
        if self.args.follow_mode == "teleport":
            if self.trajectory_idx >= len(self.trajectory_poses):
                # A teleported replay has nothing left to do once the recorded
                # poses run out: the ego stops at the last one and every further
                # tick re-records the same static view. The first run of this
                # experiment spent 45000 loops that way, which is most of its
                # wall clock, and only ended because it was killed by hand.
                if not getattr(self, "_traj_exhausted", False):
                    self._traj_exhausted = True
                    print(f"[toolkit] traiettoria esaurita a {len(self.trajectory_poses)} "
                          "pose: fine del replay", flush=True)
                return
            tf = pose_to_road_transform(self.world, self.trajectory_poses[self.trajectory_idx], float(getattr(self.args, "traj_z_offset", 0.0)))
            self.vehicle.set_transform(tf)
            self.trajectory_idx += 1
            return

        loc = self.vehicle.get_transform().location
        self.trajectory_idx = nearest_index(self.trajectory_poses, loc, self.trajectory_idx, search_ahead=25)
        target_idx = lookahead_index(self.trajectory_idx, self.trajectory_poses, vehicle_speed(self.vehicle))
        target_pose = self.trajectory_poses[target_idx]
        target_speed = smooth_target_speed(self.trajectory_poses, self.trajectory_idx, float(self.args.fps), window=4)
        raw_control = compute_follow_control(self.vehicle, target_pose, target_speed)
        alpha = float(getattr(self.args, "control_smoothing", 0.30))
        control = blend_control(self.previous_follow_control, raw_control, alpha)
        self.vehicle.apply_control(control)
        self.previous_follow_control = control

    def try_record_current_frame(self, target_frame: int | None = None):
        if self.recorder is None:
            return

        if getattr(self.args, "strict_sync", False) and target_frame is not None:
            synced = self._wait_for_exact_frame(target_frame)
        else:
            synced = self._get_synchronized_frame()
        if synced is None:
            return
        frame_id, rgb_payload, sem_payload, lidar_payload, semantic_lidar_payload = synced
        if frame_id == self.last_saved_frame:
            return
        if not self.recorder.should_save(frame_id):
            return

        if isinstance(sem_payload, dict):
            sem_array = sem_payload.get("raw")
            sem_vis_array = sem_payload.get("vis")
        else:
            sem_array = sem_payload
            sem_vis_array = sem_payload

        rgb_array = rgb_payload
        lidar_array = lidar_payload
        color_info = self.compute_color_values(lidar_array, semantic_image=sem_array, semantic_lidar=semantic_lidar_payload)
        snapshot = self.world.get_snapshot()
        timestamp = float(snapshot.timestamp.elapsed_seconds)
        start_delay = max(0.0, float(getattr(self.args, "save_start_delay_seconds", 0.0)))
        if self.recording_start_time is not None and (timestamp - self.recording_start_time) < start_delay:
            return
        ego_transform = self.vehicle.get_transform() if self.vehicle is not None else carla.Transform()
        self.recorder.save_frame(
            frame_id=frame_id,
            timestamp=timestamp,
            ego_transform=ego_transform,
            weather_name=self.args.weather,
            rgb_bgra=rgb_array,
            semantic_bgra=sem_array,
            semantic_vis_bgra=sem_vis_array,
            projection_debug_bgra=self.make_projection_debug_image(sem_vis_array, color_info['uv']),
            xyz=lidar_array[:, :3],
            raw_intensity=lidar_array[:, 3],
            semantic_tags=color_info['semantic_tags'],
            material_labels=color_info['materials'],
            ranges=color_info['ranges'],
            pseudo_final=color_info['pseudo_raw'],
            intensity_norm=color_info['intensity_norm'],
            pseudo_norm=color_info['pseudo_norm'],
            projected_points=color_info['projected_points'],
            known_material_points=color_info['known_material_points'],
            instance_ids=color_info.get('obj_ids'),
            ego_state=self._collect_ego_state(),
            actor_states=self._collect_actor_states(ego_transform),
            agent_cloud=self._agent_cloud_snapshot(),
        )
        self.rgb_buffer.discard_through(frame_id)
        self.sem_buffer.discard_through(frame_id)
        self.lidar_buffer.discard_through(frame_id)
        self.last_saved_frame = frame_id

    def _build_pcla_route_xml(self, pcla_dir: str, town: str) -> str:
        """Build a short, straight route XML for PCLA.

        For Town02 we use the known-good east-road segment (y≈109, x=69→180)
        extracted from longest6 route id=6 waypoints 28-30.  This is a clear
        110 m straight stretch with no traffic lights mid-road, well past the
        western intersection.  For other towns we fall back to longest6.
        """
        import xml.etree.ElementTree as _ET
        import tempfile as _tmp

        # Straight segments taken directly from longest6 benchmark positions.
        # Each tuple is (x, y, z) in CARLA world coordinates.
        # Town02: use the N-S road at x≈189 (longest6 route wps 30-32).
        # y=121→174 is a clear 53m straight with NO cross-streets
        # (horizontal roads are at y≈109 above and y≈192 below).
        STRAIGHT_SEGMENTS: dict = {
            'Town02': [
                (189.53, 121.70, 0.0),  # eastern N-S road, start (past y=109 junction)
                (189.53, 147.83, 0.0),  # mid-road
                (189.54, 174.08, 0.0),  # approaching y=192 junction (not reached)
            ],
        }

        if town in STRAIGHT_SEGMENTS:
            positions = STRAIGHT_SEGMENTS[town]
        else:
            longest6_path = Path(pcla_dir) / 'pcla_agents' / 'plant' / 'data' / 'longest6.xml'
            tree = _ET.parse(str(longest6_path))
            positions = []
            for route_el in tree.iter('route'):
                if route_el.attrib.get('town') == town:
                    for pos in route_el.iter('position'):
                        positions.append((
                            float(pos.attrib['x']),
                            float(pos.attrib['y']),
                            float(pos.attrib.get('z', '0')),
                        ))
                    break
            if not positions:
                raise RuntimeError(f"No longest6 route found for {town}")

        root = _ET.Element("route", id="0", town=town)
        for x, y, z in positions:
            wp = self.world.get_map().get_waypoint(carla.Location(x=x, y=y, z=z))
            _ET.SubElement(root, "waypoint",
                x=str(round(wp.transform.location.x, 4)),
                y=str(round(wp.transform.location.y, 4)),
                z=str(round(wp.transform.location.z, 4)),
                yaw=str(round(wp.transform.rotation.yaw, 4)),
                pitch="0.0", roll="0.0",
            )

        tf = _tmp.NamedTemporaryFile(suffix=".xml", delete=False, mode="w", encoding="utf-8")
        _ET.ElementTree(root).write(tf, encoding="unicode", xml_declaration=True)
        tf.close()
        print(f"[toolkit] PCLA: route for {town} ({len(positions)} wps) -> {tf.name}", flush=True)
        return tf.name

    def _generate_road_route_xml(self, start_transform, length_m: float = 600,
                                  step_m: float = 50.0, skip_m: float = 25.0) -> str:
        """Follow the road from start_transform for length_m and write a route XML to a temp file.

        Uses SPARSE waypoints (step_m=50m by default) so that GlobalRoutePlanner's
        trace_route() has enough distance to resolve lane ambiguities at intersections.
        skip_m: advance this far along the road before placing the first waypoint,
        to move away from the spawn-point intersection.
        """
        import xml.etree.ElementTree as _ET
        import tempfile as _tmp

        wp = self.world.get_map().get_waypoint(start_transform.location)

        # Advance skip_m past the spawn to avoid intersection snapping issues.
        skipped = 0.0
        while wp is not None and skipped < skip_m:
            nexts = wp.next(2.0)
            if not nexts:
                break
            wp = nexts[0]
            skipped += 2.0

        waypoints = []
        dist = 0.0
        while wp is not None and dist < length_m:
            waypoints.append(wp)
            nexts = wp.next(step_m)
            if not nexts:
                break
            wp = nexts[0]
            dist += step_m

        town = self.map.name.split("/")[-1]
        root = _ET.Element("route", id="0", town=town)
        for w in waypoints:
            _ET.SubElement(root, "waypoint",
                x=str(round(w.transform.location.x, 4)),
                y=str(round(w.transform.location.y, 4)),
                z=str(round(w.transform.location.z, 4)),
                yaw=str(round(w.transform.rotation.yaw, 4)),
                pitch="0.0", roll="0.0",
            )
        tf = _tmp.NamedTemporaryFile(suffix=".xml", delete=False, mode="w", encoding="utf-8")
        _ET.ElementTree(root).write(tf, encoding="unicode", xml_declaration=True)
        tf.close()
        print(f"[toolkit] PCLA: generated road route {len(waypoints)} sparse wps -> {tf.name}", flush=True)
        return tf.name

    def _prepare_pcla_route_xml(self, route_path: str) -> str:
        """Normalize authored route XML before handing it to PCLA.

        Short leaderboard routes are easy to author sparsely, but sparse points
        near intersections can make GlobalRoutePlanner reconstruct a different
        branch. Densify short/sparse routes onto the map first so PCLA receives
        a lane-consistent path.
        """
        import tempfile as _tmp
        import xml.etree.ElementTree as _ET

        tree = _ET.parse(route_path)
        root = tree.getroot()
        route_el = root if root.tag == "route" else next(tree.iter("route"))
        authored = []
        for waypoint in route_el.iter("waypoint"):
            authored.append(
                carla.Location(
                    x=float(waypoint.attrib["x"]),
                    y=float(waypoint.attrib["y"]),
                    z=float(waypoint.attrib.get("z", "0.0")),
                )
            )
        if len(authored) < 2:
            return route_path

        authored_len = 0.0
        max_gap = 0.0
        for a, b in zip(authored, authored[1:]):
            seg_len = a.distance(b)
            authored_len += seg_len
            max_gap = max(max_gap, seg_len)

        should_densify = authored_len <= 120.0 and (len(authored) <= 12 or max_gap > 4.0)
        if not should_densify:
            return route_path

        dense_points: list[carla.Transform] = []
        try:
            from leaderboard_codes.route_manipulation import interpolate_trajectory

            _, dense_route = interpolate_trajectory(self.world, authored, hop_resolution=2.0)
            for tf, _cmd in dense_route:
                if dense_points and tf.location.distance(dense_points[-1].location) < 0.5:
                    continue
                dense_points.append(tf)
        except Exception as exc:
            print(f"[toolkit] PCLA route densify failed for {route_path}: {exc}", flush=True)
            return route_path

        if len(dense_points) < 2:
            return route_path

        dense_root = _ET.Element(
            "route",
            id=route_el.attrib.get("id", "0"),
            town=route_el.attrib.get("town", self.map.name.split("/")[-1]),
        )
        for tf in dense_points:
            _ET.SubElement(
                dense_root,
                "waypoint",
                x=str(round(tf.location.x, 4)),
                y=str(round(tf.location.y, 4)),
                z=str(round(tf.location.z, 4)),
                yaw=str(round(tf.rotation.yaw, 4)),
                pitch=str(round(tf.rotation.pitch, 4)),
                roll=str(round(tf.rotation.roll, 4)),
            )

        tf = _tmp.NamedTemporaryFile(suffix=".xml", delete=False, mode="w", encoding="utf-8")
        _ET.ElementTree(dense_root).write(tf, encoding="unicode", xml_declaration=True)
        tf.close()
        print(
            f"[toolkit] PCLA: densified authored route {len(authored)} -> {len(dense_points)} waypoints ({authored_len:.1f} m) -> {tf.name}",
            flush=True,
        )
        return tf.name

    def _init_pcla(self):
        pcla_dir = getattr(self.args, 'pcla_dir', '')
        if not pcla_dir:
            script_path = Path(__file__).resolve()
            candidates = (
                script_path.parents[2] / "PCLA",
                script_path.parents[4] / "PCLA",
            )
            for candidate in candidates:
                if candidate.exists():
                    pcla_dir = str(candidate)
                    break
            else:
                pcla_dir = str(candidates[0])
        if pcla_dir not in sys.path:
            sys.path.insert(0, pcla_dir)
        from PCLA import PCLA  # noqa — forked repo

        # Map is already correct — load_world was called in run() before any sensor spawn.
        # Apply sync mode.
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / self.args.fps
        self.world.apply_settings(settings)

        # Freeze all traffic lights to green so the agent is never blocked.
        for _tl in self.world.get_actors().filter('traffic.traffic_light*'):
            _tl.set_state(carla.TrafficLightState.Green)
            _tl.freeze(True)
        print("[toolkit] PCLA: traffic lights frozen green", flush=True)

        route = getattr(self.args, 'pcla_route', '')
        if not route:
            town = getattr(self.args, 'pcla_town', 'Town02')
            route = self._build_pcla_route_xml(pcla_dir, town)
        authored_route = route
        # kept for the cross-track reference: the densified/downsampled versions
        # cut corners, the authored file follows the lane
        self._pcla_authored_route = route
        route = self._prepare_pcla_route_xml(route)

        # Spawn ego at the FIRST WAYPOINT of the route. For normal road routes
        # we still snap to the lane center via map.get_waypoint. For ParkingExit-
        # style routes, if the authored first waypoint is materially off-road,
        # keep the raw route transform so the ego truly starts from the parking
        # bay instead of being projected back onto the lane.
        #
        # Read the pose from the AUTHORED route, not the densified one: densify
        # runs the waypoints through interpolate_trajectory, which only knows
        # Driving lanes, so a start inside a parking bay comes back already
        # projected onto the carriageway and the check below can never see it.
        import xml.etree.ElementTree as _ET
        first_wp_el = next(_ET.parse(authored_route).iter('waypoint'))
        first_loc = carla.Location(
            x=float(first_wp_el.attrib['x']),
            y=float(first_wp_el.attrib['y']),
            z=float(first_wp_el.attrib.get('z', '0')),
        )
        road_wp = self.world.get_map().get_waypoint(first_loc)
        raw_spawn_tf = carla.Transform(
            first_loc,
            carla.Rotation(
                yaw=float(first_wp_el.attrib.get('yaw', '0')),
                pitch=float(first_wp_el.attrib.get('pitch', '0')),
                roll=float(first_wp_el.attrib.get('roll', '0')),
            ),
        )
        snap_distance = raw_spawn_tf.location.distance(road_wp.transform.location)
        # Decide by lane type, not just by distance. A route that deliberately
        # starts off the carriageway (Parking bay, shoulder) must keep its
        # authored pose: snapping moves the ego AND hands it the neighbouring
        # Driving lane's heading. In Town03's parking strip that lane is a
        # junction arm at yaw 119.7 vs the bay's 180.9 -- 61 deg off, which
        # wedges the car against the kerb at full throttle. The distance test
        # alone does not catch it (the offset there is 1.99 m, just under 2.0).
        raw_lane_wp = self.world.get_map().get_waypoint(
            first_loc, project_to_road=False, lane_type=carla.LaneType.Any)
        off_carriageway = (raw_lane_wp is not None
                           and raw_lane_wp.lane_type != carla.LaneType.Driving)
        use_raw_spawn = off_carriageway or snap_distance > 2.0
        spawn_tf = raw_spawn_tf if use_raw_spawn else road_wp.transform

        # Explicit pose override, for scenarios where the ego must NOT start on
        # its own route. ParkingExit is the case: a route whose first waypoint
        # lies in a Parking lane cannot be expressed, because PCLA re-plans it
        # with GlobalRoutePlanner, which only knows Driving lanes and returns a
        # path leaving from somewhere else entirely. The route therefore starts
        # at the point where the ego rejoins the carriageway, and the bay pose is
        # supplied here instead.
        # centimetre-scale, deterministic per (seed, replicate): same run, same pose
        _jit = float(getattr(self.args, "pcla_spawn_jitter_m", 0.0) or 0.0)
        if _jit > 0.0:
            import hashlib as _hl
            _key = f"{getattr(self.args,'seed','')}|{getattr(self.args,'replicate_index','')}"
            _h = int(_hl.sha256(_key.encode()).hexdigest()[:16], 16)
            _u = ((_h & 0xFFFF) / 0xFFFF) * 2.0 - 1.0            # along the lane
            _v = (((_h >> 16) & 0xFFFF) / 0xFFFF) * 2.0 - 1.0    # across it
            _yawr = math.radians(spawn_tf.rotation.yaw)
            spawn_tf.location.x += _jit * (_u * math.cos(_yawr) - 0.35 * _v * math.sin(_yawr))
            spawn_tf.location.y += _jit * (_u * math.sin(_yawr) + 0.35 * _v * math.cos(_yawr))
            print(f"[toolkit] PCLA: posa iniziale variata di {_jit*_u:+.3f} m lungo la corsia, "
                  f"{_jit*0.35*_v:+.3f} m di traverso", flush=True)

        _ox = getattr(self.args, "pcla_spawn_x", None)
        _oy = getattr(self.args, "pcla_spawn_y", None)
        if _ox is not None and _oy is not None:
            _oyaw = getattr(self.args, "pcla_spawn_yaw", None)
            spawn_tf = carla.Transform(
                carla.Location(x=float(_ox), y=float(_oy),
                               z=float(getattr(self.args, "pcla_spawn_z", None) or 0.0)),
                carla.Rotation(yaw=float(_oyaw) if _oyaw is not None
                               else raw_spawn_tf.rotation.yaw))
            use_raw_spawn = True
            _ow = self.world.get_map().get_waypoint(
                spawn_tf.location, project_to_road=False, lane_type=carla.LaneType.Any)
            _lane_ovr = "off-road" if _ow is None else str(_ow.lane_type).split(".")[-1]
            print(f"[toolkit] PCLA: posa di spawn forzata a "
                  f"({spawn_tf.location.x:.1f}, {spawn_tf.location.y:.1f}) "
                  f"yaw={spawn_tf.rotation.yaw:.1f} corsia={_lane_ovr}", flush=True)
        _lane_desc = "off-road" if raw_lane_wp is None else str(raw_lane_wp.lane_type).split(".")[-1]
        spawn_tf.location.z += 0.5
        if self.vehicle and self.vehicle.is_alive:
            # The viewer may have spawned or teleported the ego before PCLA starts.
            # Clear all residual motion/control so the agent's first IMU/UKF sample
            # represents the route heading instead of the previous vehicle state.
            self.vehicle.set_simulate_physics(True)
            self.vehicle.set_transform(spawn_tf)
            self.vehicle.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            self.vehicle.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
            self.vehicle.apply_control(carla.VehicleControl(
                steer=0.0, throttle=0.0, brake=1.0, hand_brake=True,
            ))
        self.world.tick()
        print(f"[toolkit] PCLA: ego placed at route start "
              f"({spawn_tf.location.x:.1f}, {spawn_tf.location.y:.1f}) "
              f"yaw={spawn_tf.rotation.yaw:.1f} "
              f"mode={'raw_route' if use_raw_spawn else 'lane_snapped'} "
              f"authored_lane={_lane_desc} "
              f"snap_dist={snap_distance:.2f}m", flush=True)

        # Import ClosedLoopSession from our integration module
        scripts_dir = str(Path(__file__).resolve().parent)
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        import matsense_closedloop as _mcl
        from matsense_closedloop import ClosedLoopSession, PerturbConfig, BehaviourLogger
        # The operator keeps its own coefficient tables, built at import from the
        # config's default_profile. apply_material_profile above rebinds this
        # module's globals only, so without this call the two disagree silently:
        # campaign 19 ran on realbag_empirical_v1 while every filename and the
        # clog's profile_name column recorded realbag_empirical_v3.
        _requested = str(getattr(self.args, "profile_name", "") or "")
        _in_force = _mcl.apply_profile(_requested or None)
        if _requested and _in_force != _requested:
            raise RuntimeError(
                f"profile mismatch: asked for {_requested!r}, operator applied "
                f"{_in_force!r}. Refusing to run a campaign whose filenames "
                f"would not describe its coefficients.")
        print(f"[toolkit] operatore: profilo {_in_force} "
              f"(beta = {dict(zip(_mcl._MATERIALS, _mcl._BASE_BY_CODE.round(4)))})",
              flush=True)
        # 'shuffled' is the material-assignment randomization control: the
        # operator's own per-point factors, permuted inside range bins. It is a
        # mode of the operator, not a profile, so the coefficients on the
        # filename are still the calibrated ones and the profile guard above
        # still means what it says.
        # 'level' is the flat arm at a level chosen through MATSENSE_LEVEL,
        # not at the level MatSense would deliver: it serves the sweep that
        # looks for the intensity tolerance threshold.
        _mode_map = {"material": "matsense", "intensity": "standard",
                     "global": "global", "shuffled": "shuffled",
                     "level": "level"}
        _session_mode = _mode_map.get(getattr(self.args, "mode", "intensity"), "standard")
        self._pcla_session = ClosedLoopSession(
            self.vehicle, self.world,
            weather=self.args.weather,
            mode=_session_mode,
            cfg=PerturbConfig(
                dropout_gain=float(getattr(self.args, "pcla_dropout_gain", 0.4)),
                jitter_gain=float(getattr(self.args, "pcla_jitter_gain", 0.10)),
                apply_intensity=not bool(getattr(self.args, "pcla_no_intensity", False)),
                intensity_mode=str(getattr(self.args, "pcla_intensity_mode", "headroom")),
                seed=int(getattr(self.args, "pcla_perturb_seed", 1234)),
            ),
        )
        _dg = float(getattr(self.args, "pcla_dropout_gain", 0.4))
        _perturb_seed = int(getattr(self.args, "pcla_perturb_seed", 1234))
        print(
            f"[toolkit] PCLA session: mode={_session_mode} weather={self.args.weather} "
            f"dropout_gain={_dg} seed={_perturb_seed}",
            flush=True,
        )
        _log_dir = Path(getattr(self.args, "dataset_root", "output_dataset")) / "clog"
        _log_dir.mkdir(parents=True, exist_ok=True)
        _scenario = getattr(self.args, "scenario_name", f"{self.args.weather}_{self.args.mode}")
        self._pcla_logger = BehaviourLogger(_log_dir / f"clog_{_scenario}.csv")
        os.environ["MATSENSE_LAV_PERCEPTION_LOG"] = str(_log_dir / f"perception_{_scenario}.csv")
        self._pcla = PCLA(
            self.args.pcla_agent, self.vehicle, route, self.client,
            perturb_fn=self._pcla_session.perturb_fn,
        )
        print(f"[toolkit] PCLA ready: agent={self.args.pcla_agent} route={route}", flush=True)

        # Print the downsampled route waypoints and build a stop condition:
        # halt when the vehicle travels >5 m past the last waypoint.
        self._pcla_route_end = None
        self._pcla_route_locations = []
        try:
            world_plan = self._pcla.agent_instance._global_plan_world_coord

            # Cross-track is measured against the AUTHORED route, not against
            # _global_plan_world_coord. The latter is the leaderboard's downsampled
            # plan: it keeps only the points where a RoadOption changes, so a 67
            # waypoint route round a bend comes back as 4 points and the polyline
            # cuts the corner. An ego driving perfectly in lane then measures 8 m of
            # "deviation" against the chord and the run is discarded as off-route.
            # The authored file follows the lane, so it is the right reference; the
            # downsampled plan is still used for the end-of-route stop condition,
            # where only the last point matters.
            self._pcla_route_locations = []
            try:
                import xml.etree.ElementTree as _ET2
                for _w in _ET2.parse(getattr(self, "_pcla_authored_route", "")).iter("waypoint"):
                    self._pcla_route_locations.append(carla.Location(
                        x=float(_w.attrib["x"]), y=float(_w.attrib["y"]),
                        z=float(_w.attrib.get("z", "0.0"))))
            except Exception as _re:
                print(f"[toolkit] route autorata illeggibile ({_re}), "
                      f"uso il piano ridotto per lo scarto", flush=True)
            if len(self._pcla_route_locations) < 2:
                self._pcla_route_locations = [tf.location for tf, _ in world_plan]
                _src = f"piano ridotto ({len(self._pcla_route_locations)} wp)"
            else:
                _src = f"route autorata ({len(self._pcla_route_locations)} wp)"
            self._pcla_route_len_cached = None
            _rl = self._pcla_route_locations
            if len(_rl) >= 2:
                self._pcla_route_len_cached = float(sum(
                    math.hypot(b.x - a.x, b.y - a.y) for a, b in zip(_rl, _rl[1:])))
            print(f"[toolkit] scarto dalla route misurato su: {_src}"
                  + (f", lunghezza {self._pcla_route_len_cached:.0f} m"
                     if self._pcla_route_len_cached else ""), flush=True)

            pts = world_plan[:10]
            print(f"[toolkit] PCLA route world coords (first {len(pts)} of {len(world_plan)} wps):", flush=True)
            for i, (tf, cmd) in enumerate(pts):
                loc = tf.location
                print(f"  wp[{i}] x={loc.x:.2f} y={loc.y:.2f} cmd={cmd}", flush=True)
            if len(world_plan) >= 2:
                first_loc = world_plan[0][0].location
                last_loc  = world_plan[-1][0].location
                dx = last_loc.x - first_loc.x
                dy = last_loc.y - first_loc.y
                dist = (dx ** 2 + dy ** 2) ** 0.5
                if dist > 0:
                    self._pcla_route_end = (dx / dist, dy / dist,
                                            last_loc.x, last_loc.y, 5.0)
                    print(
                        f"[toolkit] PCLA stop condition: >5 m past "
                        f"({last_loc.x:.1f}, {last_loc.y:.1f})",
                        flush=True,
                    )
        except Exception as _e:
            print(f"[toolkit] PCLA route debug failed: {_e}", flush=True)

    def _pcla_route_deviation_m(self, location) -> float:
        points = getattr(self, "_pcla_route_locations", [])
        if not points:
            return float("nan")
        if len(points) == 1:
            return location.distance(points[0])

        best = float("inf")
        px, py = location.x, location.y
        for a, b in zip(points, points[1:]):
            vx, vy = b.x - a.x, b.y - a.y
            denom = vx * vx + vy * vy
            if denom <= 1e-9:
                dist = math.hypot(px - a.x, py - a.y)
            else:
                t = np.clip(((px - a.x) * vx + (py - a.y) * vy) / denom, 0.0, 1.0)
                dist = math.hypot(px - (a.x + t * vx), py - (a.y + t * vy))
            best = min(best, dist)
        return float(best)

    def _pcla_route_progress_m(self, location) -> float:
        points = getattr(self, "_pcla_route_locations", [])
        if not points:
            return float("nan")
        if len(points) == 1:
            return 0.0
        best_progress = 0.0
        best_dist = float("inf")
        traversed = 0.0
        px, py = location.x, location.y
        for a, b in zip(points, points[1:]):
            vx, vy = b.x - a.x, b.y - a.y
            denom = vx * vx + vy * vy
            seg_len = math.hypot(vx, vy)
            if denom <= 1e-9:
                dist = math.hypot(px - a.x, py - a.y)
                progress = traversed
            else:
                t = float(np.clip(((px - a.x) * vx + (py - a.y) * vy) / denom, 0.0, 1.0))
                proj_x = a.x + t * vx
                proj_y = a.y + t * vy
                dist = math.hypot(px - proj_x, py - proj_y)
                progress = traversed + t * seg_len
            if dist < best_dist:
                best_dist = dist
                best_progress = progress
            traversed += seg_len
        return float(best_progress)

    def _pcla_route_length_m(self) -> float:
        points = getattr(self, "_pcla_route_locations", [])
        if len(points) < 2:
            return 0.0
        return float(sum(a.distance(b) for a, b in zip(points, points[1:])))

    def _hazard_obstacle_distance_m(self) -> float:
        actor = self.hazard_obstacle_actor
        if self.vehicle is None or actor is None or not self.vehicle.is_alive or not actor.is_alive:
            return float("nan")
        return float(self.vehicle.get_location().distance(actor.get_location()))

    def _finalize_termination(self, mode: str, reason: str) -> None:
        if self.termination_mode is None:
            self.termination_mode = mode
            self.termination_reason = reason
            print(f"[toolkit] run_termination: mode={mode} reason={reason}", flush=True)

    def _write_run_summary(self) -> None:
        if not getattr(self.args, "dataset_root", "") or not getattr(self.args, "scene_id", ""):
            return
        scenario_dir = Path(self.args.dataset_root) / self.args.scene_id / self.args.scenario_name
        scenario_dir.mkdir(parents=True, exist_ok=True)
        summary_path = scenario_dir / "run_summary.json"
        route_length_m = self._pcla_route_length_m()
        final_progress_m = float("nan")
        final_route_dev_m = float("nan")
        final_distance_to_obstacle_m = float("nan")
        if self.vehicle is not None and self.vehicle.is_alive:
            loc = self.vehicle.get_location()
            final_progress_m = self._pcla_route_progress_m(loc)
            final_route_dev_m = self._pcla_route_deviation_m(loc)
            final_distance_to_obstacle_m = self._hazard_obstacle_distance_m()
        obstacle_pos = self.hazard_obstacle_position or ("", "", "")
        summary = {
            "scenario_name": self.args.scenario_name,
            "mode": self.args.mode,
            "weather": self.args.weather,
            "seed": int(getattr(self.args, "pcla_perturb_seed", 0)),
            "profile_name": getattr(self.args, "profile_name", ""),
            "profile_version": self._profile_version,
            "profile_config_path": getattr(self.args, "material_config", ""),
            "profile_config_sha256": self._profile_config_sha256,
            "route_file": getattr(self.args, "pcla_route", ""),
            "route_file_hash": self._route_file_hash,
            "condition_type": getattr(self.args, "condition_type", ""),
            "replicate": int(getattr(self.args, "replicate_index", 0)),
            "carla_client_version": self._carla_client_version,
            "carla_server_version": self._carla_server_version,
            "termination_mode": self.termination_mode or "agent_error",
            "termination_reason": self.termination_reason or "unclassified",
            "collision_count": int(len(self.collision_events)),
            "route_length_m": route_length_m,
            "route_completion_m": final_progress_m,
            "route_completion_ratio": final_progress_m / route_length_m if route_length_m > 0 and math.isfinite(final_progress_m) else float("nan"),
            "final_cross_track_error_m": final_route_dev_m,
            "first_brake_progress_m": self._first_brake_progress_m,
            "max_deceleration_mps2": self._max_deceleration_mps2,
            "min_ttc_proxy_s": self._min_ttc_proxy_s if math.isfinite(self._min_ttc_proxy_s) else float("nan"),
            "final_distance_to_obstacle_m": final_distance_to_obstacle_m,
            "first_obstacle_in_range_progress_m": self.first_obstacle_in_range_progress_m,
            "hazard_obstacle_x": obstacle_pos[0],
            "hazard_obstacle_y": obstacle_pos[1],
            "hazard_obstacle_z": obstacle_pos[2],
            # getattr, not the attribute: the session is created late, so any
            # failure before that point reaches destroy() with the attribute
            # absent and an AttributeError here would replace the real error
            # with a useless one.
            "global_mean_ratio": (_sess.last_drop_stats.get("global_mean_ratio")
                                   if (_sess := getattr(self, "_pcla_session", None))
                                   is not None else float("nan")),
            "material_point_counts_json": (self._pcla_session.last_drop_stats.get("material_point_counts_json", "")
                                            if getattr(self, "_pcla_session", None) is not None else ""),
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def _pcla_ttc_proxy_s(self) -> float:
        """TTC to the closest vehicle ahead, assuming constant velocity."""
        ego_tf = self.vehicle.get_transform()
        ego_vel = self.vehicle.get_velocity()
        forward = ego_tf.get_forward_vector()
        best = float("inf")
        for actor in self.world.get_actors().filter("vehicle.*"):
            if actor.id == self.vehicle.id:
                continue
            rel = actor.get_location() - ego_tf.location
            longitudinal = rel.x * forward.x + rel.y * forward.y
            lateral = abs(rel.x * (-forward.y) + rel.y * forward.x)
            if longitudinal <= 0.0 or lateral > 3.0:
                continue
            other_vel = actor.get_velocity()
            closing = (
                (ego_vel.x - other_vel.x) * forward.x
                + (ego_vel.y - other_vel.y) * forward.y
            )
            if closing > 0.1:
                best = min(best, longitudinal / closing)
        return best

    def _apply_pcla_lane_guard(self, control, loop_count: int):
        """Keep PCLA near the current driving-lane center without replacing its route decisions."""
        if self.vehicle is None or control is None:
            return control

        ego_tf = self.vehicle.get_transform()
        road_wp = self.map.get_waypoint(
            ego_tf.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if road_wp is None:
            return control

        lane_change_ahead = False
        route_locs = getattr(self, "_pcla_route_locations", None) or []
        if route_locs:
            try:
                nearest_idx = min(
                    range(len(route_locs)),
                    key=lambda idx: ego_tf.location.distance(route_locs[idx]),
                )
                for loc in route_locs[nearest_idx: min(len(route_locs), nearest_idx + 8)]:
                    if abs(loc.y - road_wp.transform.location.y) > 1.25:
                        lane_change_ahead = True
                        break
            except Exception:
                lane_change_ahead = False

        lane_tf = road_wp.transform
        yaw_rad = math.radians(lane_tf.rotation.yaw)
        right_x = -math.sin(yaw_rad)
        right_y = math.cos(yaw_rad)
        dx = ego_tf.location.x - lane_tf.location.x
        dy = ego_tf.location.y - lane_tf.location.y
        right_offset_m = dx * right_x + dy * right_y
        heading_error_deg = (
            lane_tf.rotation.yaw - ego_tf.rotation.yaw + 180.0
        ) % 360.0 - 180.0

        # CARLA positive steer increases yaw. Correct both heading and lateral
        # drift, but leave small model steering commands untouched near center.
        correction = 0.025 * heading_error_deg - 0.22 * right_offset_m
        correction_strength = min(1.0, max(
            abs(right_offset_m) / 0.75,
            abs(heading_error_deg) / 8.0,
        ))
        original_steer = float(control.steer)
        if lane_change_ahead:
            if loop_count < 20 or loop_count % 40 == 0:
                print(
                    f"[toolkit] pcla_lane_guard loop={loop_count} lane_change_ahead=1 "
                    f"offset={right_offset_m:.2f}m heading_err={heading_error_deg:.2f}deg "
                    f"steer={original_steer:.3f}->{original_steer:.3f}",
                    flush=True,
                )
            return control
        control.steer = float(np.clip(
            original_steer + correction * correction_strength,
            -0.65,
            0.65,
        ))

        lane_distance = ego_tf.location.distance(lane_tf.location)
        if lane_distance > 1.25:
            control.throttle = min(float(control.throttle), 0.25)
        if lane_distance > 2.25:
            control.throttle = 0.0
            control.brake = max(float(control.brake), 0.5)

        if loop_count < 20 or loop_count % 40 == 0:
            print(
                f"[toolkit] pcla_lane_guard loop={loop_count} "
                f"offset={right_offset_m:.2f}m heading_err={heading_error_deg:.2f}deg "
                f"steer={original_steer:.3f}->{control.steer:.3f}",
                flush=True,
            )
        return control

    def _init_open_loop_matsense(self):
        """Create the sensor operator without starting a driving agent."""
        scripts_dir = str(Path(__file__).resolve().parent)
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        import matsense_closedloop as _mcl
        from matsense_closedloop import ClosedLoopSession, PerturbConfig
        requested = str(getattr(self.args, "profile_name", "") or "")
        applied = _mcl.apply_profile(requested or None)
        if requested and applied != requested:
            raise RuntimeError(f"profile mismatch: asked for {requested!r}, applied {applied!r}")
        arm = str(getattr(self.args, "open_loop_arm", "matsense") or "matsense")
        self._pcla_session = ClosedLoopSession(
            self.vehicle, self.world, weather=self.args.weather, mode=arm,
            cfg=PerturbConfig(
                dropout_gain=float(getattr(self.args, "pcla_dropout_gain", 0.4)),
                jitter_gain=float(getattr(self.args, "pcla_jitter_gain", 0.10)),
                apply_intensity=not bool(getattr(self.args, "pcla_no_intensity", False)),
                intensity_mode=str(getattr(self.args, "pcla_intensity_mode", "headroom")),
                seed=int(getattr(self.args, "pcla_perturb_seed", 1234)),
            ),
        )
        print(f"[toolkit] open-loop session: arm={arm} profile={applied} "
              f"seed={getattr(self.args, 'pcla_perturb_seed', 1234)}", flush=True)

    def run(self):
        print("[toolkit] run: setup_pygame", flush=True)
        self.setup_pygame()

        # If PCLA requests a specific town, load it NOW — before spawning any sensors.
        # load_world after sensors are alive causes a LibCarla assertion crash.
        if getattr(self.args, 'pcla_agent', ''):
            town = getattr(self.args, 'pcla_town', 'Town02')
            requested_town = self._normalize_town_name(town)
            current_map_name = self._normalize_town_name(self.map.name)
            if current_map_name != requested_town:
                print(f"[toolkit] run: loading {town} (was {current_map_name}) before sensor spawn …", flush=True)
                self.client.set_timeout(60.0)
                self.client.load_world(town)
                self.world = self.client.get_world()
                self.map = self.world.get_map()
                self.blueprints = self.world.get_blueprint_library()
                self.client.set_timeout(10.0)
                print(f"[toolkit] run: {town} ready", flush=True)

        print("[toolkit] run: spawning vehicle and sensors", flush=True)
        self.request_respawn("startup", force=True)
        self._pcla = None
        self._pcla_session = None
        if getattr(self.args, 'pcla_agent', ''):
            print("[toolkit] run: initialising PCLA agent …", flush=True)
            self._init_pcla()
        elif getattr(self.args, "open_loop_matsense", False):
            print("[toolkit] run: initialising standalone MatSense session …", flush=True)
            self._init_open_loop_matsense()
        print("[toolkit] run: entering main loop", flush=True)
        loop_count = 0

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return
                    if event.key == pygame.K_p:
                        mode_sequence = list(DISPLAY_MODE_SEQUENCE)
                        if CAMERA_TRIPLE_MODE not in mode_sequence:
                            mode_sequence.append(CAMERA_TRIPLE_MODE)
                        try:
                            idx = mode_sequence.index(self.color_mode)
                        except ValueError:
                            idx = 0
                        self.color_mode = mode_sequence[(idx + 1) % len(mode_sequence)]
                    if event.key == pygame.K_r and self._pcla is None:
                        self.request_respawn("keyboard_r")

            if self._pcla is not None:
                self._update_dynamic_actors()
                self._update_walkers()
                if loop_count < 3:
                    print(f"[toolkit] before_tick loop={loop_count}", flush=True)
                _t0 = time.perf_counter()
                target_frame = self.world.tick()
                self._phase_add("tick", time.perf_counter() - _t0)
                self._tick_srunner()
                if loop_count < 3:
                    print(f"[toolkit] after_tick loop={loop_count} frame={target_frame}", flush=True)
                try:
                    _t0 = time.perf_counter()
                    ego_action = self._pcla.get_action()
                    self._phase_add("agente", time.perf_counter() - _t0)
                    if ego_action is not None:
                        if getattr(self.args, "pcla_lane_guard", False):
                            ego_action = self._apply_pcla_lane_guard(ego_action, loop_count)
                        self.vehicle.apply_control(ego_action)
                        if loop_count < 20 or loop_count % 40 == 0:
                            print(
                                f"[toolkit] pcla_ctrl loop={loop_count} "
                                f"steer={ego_action.steer:.3f} "
                                f"throttle={ego_action.throttle:.3f} "
                                f"brake={ego_action.brake:.3f}",
                                flush=True,
                            )
                except Exception as _e:
                    print(f"[toolkit] PCLA get_action error: {_e}", flush=True)
                    self._finalize_termination("agent_error", f"pcla_get_action:{_e}")
                    if hasattr(self, '_pcla_logger') and self._pcla_logger is not None:
                        self._pcla_logger.close()
                        self._pcla_logger = None
                    return
            elif self.trajectory_poses:
                self.follow_trajectory_step()
                if getattr(self, "_traj_exhausted", False) and \
                        not bool(getattr(self.args, "keep_running_after_traj", False)):
                    if hasattr(self, '_pcla_logger') and self._pcla_logger is not None:
                        self._pcla_logger.close()
                        self._pcla_logger = None
                    return
                self._update_dynamic_actors()
                self._update_walkers()
                if loop_count < 3:
                    print(f"[toolkit] before_tick loop={loop_count}", flush=True)
                target_frame = self.world.tick()
                self._tick_srunner()
                if loop_count < 3:
                    print(f"[toolkit] after_tick loop={loop_count} frame={target_frame}", flush=True)
            else:
                self.manual_control()
                self._update_dynamic_actors()
                self._update_walkers()
                if loop_count < 3:
                    print(f"[toolkit] before_tick loop={loop_count}", flush=True)
                target_frame = self.world.tick()
                self._tick_srunner()
                if loop_count < 3:
                    print(f"[toolkit] after_tick loop={loop_count} frame={target_frame}", flush=True)
            loop_count += 1
            self._update_spectator_follow()

            max_seconds = float(getattr(self.args, "pcla_max_seconds", 0.0))
            if self._pcla is not None and max_seconds > 0.0 \
                    and loop_count >= int(round(max_seconds * self.args.fps)):
                print(
                    f"[toolkit] PCLA max duration reached ({max_seconds:.1f}s), stopping.",
                    flush=True,
                )
                self._finalize_termination("timeout", f"max_seconds={max_seconds:.1f}")
                if hasattr(self, '_pcla_logger') and self._pcla_logger is not None:
                    self._pcla_logger.close()
                    self._pcla_logger = None
                return

            # Stop when the PCLA vehicle has driven >5 m past the route end.
            if self._pcla is not None and self._pcla_route_end is not None \
                    and self.vehicle is not None and self.vehicle.is_alive:
                rx, ry, lx, ly, margin = self._pcla_route_end
                loc = self.vehicle.get_location()
                past = rx * (loc.x - lx) + ry * (loc.y - ly)
                # The half-plane alone is not enough. Its direction is the chord
                # from the route's first waypoint to its last, so on a route that
                # turns, the region "beyond the end" reaches back across the map
                # and the ego enters it while still far from the finish. On the
                # official 529 m Leaderboard route it fires at 124 m and the run
                # is recorded as completed having driven a fifth of it. Arc
                # length along the authored route settles it; the short, straight
                # campaign routes satisfy both at the same place, so their
                # behaviour is unchanged.
                _prog = self._pcla_route_progress_m(loc)
                _len = getattr(self, "_pcla_route_len_cached", None)
                if past > margin and _len and math.isfinite(_prog) \
                        and _prog < _len - 10.0:
                    past = -1.0        # beyond the chord but not along the road
                if past > margin:
                    print(
                        f"[toolkit] PCLA route end reached (past={past:.1f} m) "
                        f"at loop={loop_count}, stopping.",
                        flush=True,
                    )
                    self._finalize_termination("completed", f"past_route_end={past:.2f}")
                    if hasattr(self, '_pcla_logger') and self._pcla_logger is not None:
                        self._pcla_logger.close()
                        self._pcla_logger = None
                    return

            route_dev_threshold = float(getattr(self.args, "pcla_route_dev_threshold", 0.0))
            if self._pcla is not None and route_dev_threshold > 0.0 and self.vehicle is not None and self.vehicle.is_alive:
                _loc_dev = self.vehicle.get_location()
                route_dev_now = self._pcla_route_deviation_m(_loc_dev)
                # Deviation is distance to the nearest point of the route
                # polyline, and past the final waypoint that point stops moving:
                # a vehicle continuing straight through the finish accumulates
                # "deviation" that is entirely longitudinal. Every run of one
                # campaign arm ended labelled route_deviation at 5.03 m while its
                # lateral offset was 0.04 m - a completed route recorded as a
                # departure. Once the ego has covered the route, the end-of-route
                # condition is the one that applies, so this one stands down.
                _len_dev = getattr(self, "_pcla_route_len_cached", None)
                if _len_dev:
                    _prog_dev = self._pcla_route_progress_m(_loc_dev)
                    if math.isfinite(_prog_dev) and _prog_dev >= _len_dev - 2.0:
                        route_dev_now = 0.0
                if math.isfinite(route_dev_now) and route_dev_now > route_dev_threshold:
                    print(
                        f"[toolkit] PCLA route deviation termination: route_dev={route_dev_now:.2f}m "
                        f"threshold={route_dev_threshold:.2f}m",
                        flush=True,
                    )
                    self._finalize_termination("route_deviation", f"cross_track={route_dev_now:.3f}")
                    if hasattr(self, '_pcla_logger') and self._pcla_logger is not None:
                        self._pcla_logger.close()
                        self._pcla_logger = None
                    return

            # Early stop once the outcome is settled. Scenarios whose expected
            # behaviour is "come to a halt" (lead-vehicle brake, obstacle ahead,
            # yielding) reach their final state well before --pcla-max-seconds and
            # then log identical frames until the timeout. The safety metric is the
            # resting position, which is already fixed by then. Ending here keeps the
            # outcome explicit in the taxonomy instead of hiding it under "timeout".
            settle_s = float(getattr(self.args, "pcla_stopped_seconds", 0.0))
            if (self._pcla is not None and settle_s > 0.0
                    and self.vehicle is not None and self.vehicle.is_alive):
                v = self.vehicle.get_velocity()
                speed_now = math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
                elapsed_pcla = loop_count / float(self.args.fps)
                if speed_now < float(getattr(self.args, "pcla_stopped_speed", 0.1)):
                    if getattr(self, "_pcla_stopped_since", None) is None:
                        self._pcla_stopped_since = elapsed_pcla
                    elif elapsed_pcla - self._pcla_stopped_since >= settle_s:
                        held = elapsed_pcla - self._pcla_stopped_since
                        loc = self.vehicle.get_location()
                        print(f"[toolkit] PCLA stopped termination: fermo da {held:.1f}s "
                              f"a ({loc.x:.2f}, {loc.y:.2f})", flush=True)
                        self._finalize_termination(
                            "stopped", f"held={held:.1f}s x={loc.x:.3f} y={loc.y:.3f}")
                        if hasattr(self, '_pcla_logger') and self._pcla_logger is not None:
                            self._pcla_logger.close()
                            self._pcla_logger = None
                        return
                else:
                    self._pcla_stopped_since = None

            if self._pcla is not None and self.collision_events:
                collision = self.collision_events[0]
                self._finalize_termination(
                    "collision",
                    f"frame={collision.get('frame', -1)} other={collision.get('other_actor_type', '')}",
                )
                if hasattr(self, '_pcla_logger') and self._pcla_logger is not None:
                    self._pcla_logger.close()
                    self._pcla_logger = None
                return

            if loop_count % 40 == 0 and self.vehicle is not None:
                tf = self.vehicle.get_transform()
                speed = vehicle_speed(self.vehicle)
                vehicle_count = len(self.world.get_actors().filter("vehicle.*"))
                print(
                    f"[toolkit] loop={loop_count} hero_alive={self.vehicle.is_alive} "
                    f"x={tf.location.x:.2f} y={tf.location.y:.2f} yaw={tf.rotation.yaw:.2f} "
                    f"speed={speed:.2f} world_vehicles={vehicle_count}",
                    flush=True,
                )
            _t0 = time.perf_counter()
            # Per-tick behaviour log for RQ4 closed-loop analysis
            if self._pcla is not None and hasattr(self, '_pcla_logger') \
                    and self._pcla_logger is not None and self.vehicle is not None:
                try:
                    _tf  = self.vehicle.get_transform()
                    _vel = self.vehicle.get_velocity()
                    _spd = (_vel.x**2 + _vel.y**2 + _vel.z**2) ** 0.5
                    _snap = self.world.get_snapshot()
                    _drop = self._pcla_session.last_drop_stats if self._pcla_session else {}
                    _ctrl = self.vehicle.get_control()
                    _ttc = self._pcla_ttc_proxy_s()
                    _progress = self._pcla_route_progress_m(_tf.location)
                    _cross_track = self._pcla_route_deviation_m(_tf.location)
                    _obs_dist = self._hazard_obstacle_distance_m()
                    _obs_in_range = int(math.isfinite(_obs_dist) and _obs_dist <= float(getattr(self.args, "hazard_sensor_range_m", self.args.lidar_range)))
                    if _obs_in_range and self.first_obstacle_in_range_progress_m is None:
                        self.first_obstacle_in_range_progress_m = _progress
                    if _ctrl.brake > self._brake_threshold and self._first_brake_progress_m is None:
                        self._first_brake_progress_m = _progress
                    _now_t = float(_snap.timestamp.elapsed_seconds)
                    if self._last_tick_time_s is not None and self._last_tick_speed_mps is not None:
                        _dt = max(_now_t - self._last_tick_time_s, 1e-6)
                        _decel = max((self._last_tick_speed_mps - _spd) / _dt, 0.0)
                        self._max_deceleration_mps2 = max(self._max_deceleration_mps2, float(_decel))
                    self._last_tick_time_s = _now_t
                    self._last_tick_speed_mps = _spd
                    if math.isfinite(_ttc):
                        self._min_ttc_proxy_s = min(self._min_ttc_proxy_s, float(_ttc))
                    _obs_pos = self.hazard_obstacle_position or ("", "", "")
                    self._pcla_logger.log(
                        frame=target_frame,
                        t_s=round(float(_snap.timestamp.elapsed_seconds), 3),
                        x=round(_tf.location.x, 3), y=round(_tf.location.y, 3),
                        speed_mps=round(_spd, 3),
                        throttle=round(_ctrl.throttle, 4),
                        brake=round(_ctrl.brake, 4),
                        steer=round(_ctrl.steer, 5),
                        n_raw=_drop.get("n_raw", 0),
                        n_perturbed=_drop.get("n_perturbed", 0),
                        drop_frac=_drop.get("drop_frac", 0.0),
                        # the two halves of `unknown`, kept apart so a mounting
                        # error cannot hide inside a taxonomy gap again
                        n_unmatched=_drop.get("n_unmatched", 0),
                        n_unmapped=_drop.get("n_unmapped", 0),
                        sem_frame_lag=_drop.get("sem_frame_lag", -999),
                        route_dev_m=round(_cross_track, 3),
                        cross_track_error_m=round(_cross_track, 3),
                        progress_m=round(_progress, 3),
                        ttc_proxy_s=round(_ttc, 3) if math.isfinite(_ttc) else "",
                        obstacle_distance_m=round(_obs_dist, 3) if math.isfinite(_obs_dist) else "",
                        obstacle_in_sensor_range=_obs_in_range,
                        first_obstacle_in_range_progress_m=round(self.first_obstacle_in_range_progress_m, 3)
                        if self.first_obstacle_in_range_progress_m is not None else "",
                        hazard_obstacle_x=_obs_pos[0],
                        hazard_obstacle_y=_obs_pos[1],
                        hazard_obstacle_z=_obs_pos[2],
                        mode=getattr(self.args, "mode", ""),
                        weather=getattr(self.args, "weather", ""),
                        seed=int(getattr(self.args, "pcla_perturb_seed", 1234)),
                        replicate=int(getattr(self.args, "replicate_index", 0)),
                        condition_type=getattr(self.args, "condition_type", ""),
                        profile_name=getattr(self.args, "profile_name", ""),
                        profile_version=self._profile_version,
                        profile_config_path=getattr(self.args, "material_config", ""),
                        profile_config_sha256=self._profile_config_sha256,
                        route_file=getattr(self.args, "pcla_route", ""),
                        route_file_hash=self._route_file_hash,
                        global_mean_ratio=round(float(_drop.get("global_mean_ratio", float("nan"))), 6)
                        if math.isfinite(float(_drop.get("global_mean_ratio", float("nan")))) else "",
                        material_point_counts_json=_drop.get("material_point_counts_json", ""),
                        # What the operator applied, alongside profile_name,
                        # which is only what was asked for. The two differed for
                        # every run of campaign 19 and nothing said so.
                        operator_profile=_drop.get("operator_profile", ""),
                        intensity_mode=_drop.get("intensity_mode", ""),
                        carla_client_version=self._carla_client_version,
                        carla_server_version=self._carla_server_version,
                    )
                except Exception as exc:
                    # Swallowing this silently is how two new columns came out
                    # empty for a whole probe run without anyone noticing. Say
                    # it once: a per-tick message would flood the log.
                    if not getattr(self, "_clog_error_reported", False):
                        self._clog_error_reported = True
                        print(f"[toolkit] clog: riga non scritta ({type(exc).__name__}: "
                              f"{exc}). Le successive non verranno segnalate.",
                              flush=True)
            self._phase_add("clog", time.perf_counter() - _t0)

            _t0 = time.perf_counter()
            self._update_display_from_synced_frame(target_frame)
            self._phase_add("sensori_toolkit", time.perf_counter() - _t0)
            _t0 = time.perf_counter()
            self.try_record_current_frame(target_frame)
            self._phase_add("salvataggio", time.perf_counter() - _t0)
            self._phase_report(loop_count)

            if self.no_draw or (loop_count % self.render_stride) != 0:
                self.clock.tick(self.args.fps)
                continue

            self.screen.fill(BG_COLOR)
            layout = self.get_layout()
            current_color_info = self.last_color_info
            if self.color_mode == CAMERA_TRIPLE_MODE:
                self.draw_triple_camera_view(current_color_info)
                pygame.display.flip()
                if self._save_display_screenshot(loop_count):
                    return
                self.clock.tick(self.args.fps)
                continue

            frame_rect = layout["camera"]
            draw_card(self.screen, frame_rect)
            if self.rgb_array is not None:
                left_image = self.rgb_array
                if current_color_info is not None and self.color_mode in ("intensity", "global", "material"):
                    overlay = self.make_rgb_lidar_overlay(self.rgb_array, current_color_info, self.color_mode)
                    if overlay is not None:
                        left_image = overlay
                rgb_surf = bgra_to_rgb_surface(left_image)
                scaled = pygame.transform.smoothscale(rgb_surf, (frame_rect.w - 16, frame_rect.h - 16))
                self.screen.blit(scaled, (frame_rect.x + 8, frame_rect.y + 8))
            draw_text(self.screen, self.font_panel, "Camera View", (frame_rect.x + 16, frame_rect.y + 14))
            draw_text(
                self.screen,
                self.font_small,
                "RGB + LiDAR overlay" if self.color_mode in ("intensity", "global", "material") else "Raw RGB",
                (frame_rect.x + 18, frame_rect.y + 42),
                TEXT_MUTED,
            )
            draw_text_block(
                self.screen,
                self.font_small,
                mode_description(self.color_mode),
                (frame_rect.x + 18, frame_rect.y + 60),
                frame_rect.w - 36,
                2,
                TEXT_MUTED,
            )
            if self.rgb_array is None:
                draw_text(self.screen, self.font_small, "Waiting for RGB sensor...", (frame_rect.x + 18, frame_rect.y + 84), TEXT_MUTED)

            birdseye_rect = layout["birdseye"]
            draw_card(self.screen, birdseye_rect)
            if self.birdseye_array is not None:
                birdseye_surf = bgra_to_rgb_surface(self.birdseye_array)
                scaled_be = pygame.transform.smoothscale(birdseye_surf, (birdseye_rect.w - 16, birdseye_rect.h - 16))
                self.screen.blit(scaled_be, (birdseye_rect.x + 8, birdseye_rect.y + 8))
            draw_text(self.screen, self.font_panel, "Bird's Eye View", (birdseye_rect.x + 16, birdseye_rect.y + 14))
            if self.birdseye_array is None:
                draw_text(self.screen, self.font_small, "Waiting for camera...", (birdseye_rect.x + 18, birdseye_rect.y + 42), TEXT_MUTED)

            bev = self.bev_surface_from_lidar(self.last_lidar, self.color_mode)
            bev_rect = layout["bev"]
            draw_card(self.screen, bev_rect)
            scaled_bev = pygame.transform.smoothscale(bev, (bev_rect.w - 16, bev_rect.h - 16))
            self.screen.blit(scaled_bev, (bev_rect.x + 8, bev_rect.y + 8))
            draw_text(self.screen, self.font_panel, "Top-Down LiDAR", (bev_rect.x + 16, bev_rect.y + 14))
            draw_text(self.screen, self.font_small, mode_title(self.color_mode), (bev_rect.x + 16, bev_rect.y + 42), TEXT_MAIN)
            draw_text_block(
                self.screen,
                self.font_small,
                mode_description(self.color_mode),
                (bev_rect.x + 16, bev_rect.y + 60),
                bev_rect.w - 32,
                2,
                TEXT_MUTED,
            )
            npts = 0 if self.last_lidar is None else int(self.last_lidar.shape[0])
            self.draw_hud(self.color_mode, npts, layout)
            pygame.display.flip()
            if self._save_display_screenshot(loop_count):
                return
            self.clock.tick(self.args.fps)


def build_argparser():
    ap = argparse.ArgumentParser(description="CARLA pygame LiDAR pseudo-reflectance viewer")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--tm-port", type=int, default=8000)
    ap.add_argument("--client-timeout", type=float, default=30.0,
                    help="Timeout in seconds for steady-state CARLA RPC calls after startup/world load.")
    ap.add_argument("--allow-version-mismatch", action="store_true",
                    help="Bypass the CARLA client/server compatibility check. Unsafe: incompatible simulator builds can crash natively.")
    ap.add_argument("--sdl-driver", type=str, default="",
                    help="Force a specific SDL video driver such as x11, wayland, offscreen, or dummy before pygame init.")
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=800)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--vehicle-filter", default="vehicle.tesla.model3")
    ap.add_argument("--spawn-index", type=int, default=0)
    ap.add_argument("--spawn-x", type=float, default=None,
                    help="Target X to snap onto the nearest driving waypoint")
    ap.add_argument("--spawn-y", type=float, default=None,
                    help="Target Y to snap onto the nearest driving waypoint")
    ap.add_argument("--spawn-z", type=float, default=None,
                    help="Optional target Z before road projection; editor helper actors often sit high above the road")
    ap.add_argument("--spawn-back-m", type=float, default=0.0,
                    help="Move this many meters before the snapped waypoint, useful to start before a RoutePlanner trigger")
    ap.add_argument("--spawn-z-offset", type=float, default=0.5,
                    help="Vertical offset added after snapping to the road to avoid collision/ground penetration at spawn")
    ap.add_argument("--autopilot", action="store_true")
    ap.add_argument("--traj-json", type=str, default=None,
                    help="Optional trajectory JSON to follow instead of manual driving/spawn point logic")
    ap.add_argument("--traj-txt", type=str, default=None,
                    help="Optional trajectory TXT in UTM format: frame_id,timestamp,easting,northing,yaw_rad")
    ap.add_argument("--traj-step", type=int, default=1,
                    help="Use one point every N when loading --traj-txt")
    ap.add_argument("--traj-shift-x", type=float, default=0.0,
                    help="Apply this X shift to every trajectory pose after loading the JSON")
    ap.add_argument("--traj-shift-y", type=float, default=0.0,
                    help="Apply this Y shift to every trajectory pose after loading the JSON")
    ap.add_argument("--traj-shift-z", type=float, default=0.0,
                    help="Apply this Z shift to every trajectory pose after loading the JSON")
    ap.add_argument("--traj-scale", type=float, default=1.0,
                    help="Apply this scale to trajectory XYZ before shifts")
    ap.add_argument("--utm-offset-x", type=float, default=0.0,
                    help="UTM X offset used to convert --traj-txt into local CARLA coordinates")
    ap.add_argument("--utm-offset-y", type=float, default=0.0,
                    help="UTM Y offset used to convert --traj-txt into local CARLA coordinates")
    ap.add_argument("--traj-z-offset", type=float, default=0.0,
                    help="Vertical offset applied when spawning/following a trajectory JSON")
    ap.add_argument("--follow-mode", choices=["control", "teleport"], default="control",
                    help="How to follow --traj-json: vehicle control or exact teleport")
    ap.add_argument("--control-smoothing", type=float, default=0.30,
                    help="Blend factor for successive control commands when follow-mode=control")
    ap.add_argument("--walkers-json", type=str, default=None,
                    help="pedestrians for the scenario. Same trigger-and-segments "
                         "shape as --dynamic-json, but driven with WalkerControl "
                         "so the skeleton animates: a pedestrian slid along by "
                         "target velocity presents the LiDAR a rigid shape.")
    ap.add_argument("--walker-z-offset", type=float, default=0.6,
                    help="spawn height above the ground point, so a walker does "
                         "not spawn inside the road surface")
    ap.add_argument("--parked-json", type=str, default=None,
                    help="Optional vehicle_data JSON with spawn_positions for parked cars")
    ap.add_argument("--parked-shift-x", type=float, default=0.0,
                    help="Apply this X shift to every parked start/end position after loading the JSON")
    ap.add_argument("--parked-shift-y", type=float, default=0.0,
                    help="Apply this Y shift to every parked start/end position after loading the JSON")
    ap.add_argument("--parked-shift-z", type=float, default=0.0,
                    help="Apply this Z shift to every parked start/end position after loading the JSON")
    ap.add_argument("--parked-scale", type=float, default=1.0,
                    help="Apply this scale to parked vehicle XYZ before shifts")
    ap.add_argument("--parked-settle-ticks", type=int, default=15,
                    help="frames of physics given to parked vehicles after "
                         "spawning, so the spawn lift comes off and the wheels "
                         "reach the road. Zero keeps the previous behaviour.")
    ap.add_argument("--parked-z-offset", type=float, default=0.5,
                    help="Initial lift used while spawning parked vehicles before physical settling")
    ap.add_argument("--startup-diagnostics-json", type=str, default="",
                    help="where the startup state is recorded; defaults to "
                         "/tmp/matsense_last_startup.json so a run that went "
                         "wrong can be diagnosed afterwards")
    ap.add_argument("--traj-spawn-lift", type=float, default=0.3,
                    help="vertical lift used only for the ego's spawn attempt "
                         "along the trajectory. Without it every pose is "
                         "refused for intersecting the road and the run falls "
                         "back to an arbitrary map spawn point.")
    ap.add_argument("--validate-against-map",
                    choices=["abort", "warn", "off"], default="abort",
                    help="check the trajectory and the parked layout against "
                         "the loaded map's road network before spawning. A file "
                         "in the wrong frame is self-consistent and only the map "
                         "can catch it.")
    ap.add_argument("--validate-max-median-m", type=float, default=8.0,
                    help="median distance from the nearest driving lane above "
                         "which the input is judged not to belong to this map")
    ap.add_argument("--parked-avoid-ego-lane", action="store_true",
                    help="move aside only the parked vehicles whose lateral "
                         "distance from the driven trajectory is below "
                         "--parked-clearance-m, keeping the side they were "
                         "recovered on. The layout on disk is not modified.")
    ap.add_argument("--parked-clearance-m", type=float, default=2.5,
                    help="lateral clearance a parked vehicle must leave the "
                         "trajectory. Half the ego plus half a parked car is "
                         "1.85 m; the rest is the agent's own tracking error.")
    ap.add_argument("--parked-report-json", type=str, default="",
                    help="where the validation report is written; defaults to "
                         "<dataset-root>/parked_reports/<scenario>_parked.json")
    ap.add_argument("--parked-limit", type=int, default=0,
                    help="Spawn only the first N parked vehicles from --parked-json; 0 means all")
    ap.add_argument("--srunner-scenario", type=str, default="",
                    help="run an official ScenarioRunner hazard instead of a "
                         "hand-built --dynamic-json one. The catalogue entries "
                         "matching this campaign are parking_crossing_pedestrian, "
                         "invading_turn, static_cut_in, no_signal_junction_crossing "
                         "and follow_leading_vehicle. These carry the NHTSA "
                         "provenance the hand-built copies lack, but they are "
                         "parameterised by distance rather than by warning time, "
                         "so they still need the same calibration.")
    ap.add_argument("--srunner-route", type=str, default="",
                    help="JSON written by tools/import_leaderboard_route.py, "
                         "carrying every scenario of an official Leaderboard "
                         "route with its own trigger point and parameters. Use "
                         "with the matching --pcla-route; mutually exclusive "
                         "with --srunner-scenario.")
    ap.add_argument("--srunner-param", action="append", default=[],
                    metavar="KEY=VALUE",
                    help="scenario parameter, repeatable, e.g. --srunner-param "
                         "distance=12. Names come from each scenario's own "
                         "get_value_parameter calls; an unknown name is ignored "
                         "silently by ScenarioRunner and falls back to default.")
    ap.add_argument("--dynamic-json", type=str, default=None,
                    help="Optional JSON with trigger-based moving vehicle actors")
    ap.add_argument("--dynamic-shift-x", type=float, default=0.0,
                    help="Apply this X shift to every dynamic actor start/trigger position after loading the JSON")
    ap.add_argument("--dynamic-shift-y", type=float, default=0.0,
                    help="Apply this Y shift to every dynamic actor start/trigger position after loading the JSON")
    ap.add_argument("--dynamic-shift-z", type=float, default=0.0,
                    help="Apply this Z shift to every dynamic actor start/trigger position after loading the JSON")
    ap.add_argument("--dynamic-scale", type=float, default=1.0,
                    help="Apply this scale to dynamic actor XYZ before shifts")
    ap.add_argument("--dynamic-z-offset", type=float, default=0.5,
                    help="Vertical offset used for dynamic vehicles loaded from --dynamic-json")
    ap.add_argument("--seed", type=int, default=42,
                    help="Deterministic seed for parked vehicle blueprint selection")
    ap.add_argument("--weather", choices=["nominal", "rain", "snow"], default="nominal")
    ap.add_argument("--mode", choices=["intensity", "global", "material", "shuffled",
                                       "level", CAMERA_TRIPLE_MODE], default="global")
    # LMDrive's rig: four views with its own positions and fields
    # of view, saved next to the clouds. It is needed because its backbone is
    # a fusion model and cannot be queried without its cameras; our recordings
    # have a single view, from LAV's rig.
    ap.add_argument("--lmdrive-rig", action="store_true",
                    help="also record the four views LMDrive's backbone "
                         "requires, under lmdrive_rgb/")
    ap.add_argument("--pcla-jitter-gain", type=float, default=0.10,
                    help="Range jitter applied to strongly attenuated returns. "
                         "Set to 0 to leave geometry alone and isolate the other terms.")
    ap.add_argument("--pcla-no-intensity", action="store_true",
                    help="Apply the material-dependent dropout and jitter but NOT the "
                         "intensity rescaling, so the two halves of the operator can be "
                         "told apart. Without this the campaign cannot say whether the "
                         "trajectory moves because returns went missing or because the "
                         "calibrated coefficients changed what the network reads.")
    ap.add_argument("--pcla-intensity-mode", choices=["headroom", "clip"],
                    default="headroom",
                    help="How the calibrated response is mapped into [0,1]. "
                         "'headroom' scales by the profile's largest coefficient so "
                         "nothing saturates and every measured ratio survives; the "
                         "uniform baseline then receives the same per-frame mean, so "
                         "the pair differs in material structure alone. 'clip' is the "
                         "pre-correction behaviour and reproduces campaigns up to 19: "
                         "it normalised to a mean of one and truncated, which pinned a "
                         "quarter of returns at 1.0 and delivered 1.47 of a measured "
                         "2.50 facade-to-asphalt ratio.")
    ap.add_argument("--ambient-walkers", type=int, default=0,
                    help="background pedestrians routed by CARLA's walker AI over "
                         "the map's navigation mesh. They walk; they do not follow "
                         "a scripted line. The scenario's own hazard pedestrian "
                         "stays scripted, because a conformance test needs it to "
                         "arrive at a defined moment.")
    ap.add_argument("--ambient-walker-speed", type=float, default=1.4,
                    help="metres per second, about 5 km/h")
    ap.add_argument("--ambient-keep-clear-m", type=float, default=8.0)
    ap.add_argument("--tm-traffic", type=int, default=0,
                    help="background vehicles driven by CARLA's Traffic Manager. "
                         "They follow lanes, obey lights and brake behind each "
                         "other, which the scenario JSON's open-loop actors "
                         "cannot. 0 disables it, which is what every campaign so "
                         "far ran with.")
    ap.add_argument("--tm-keep-clear-m", type=float, default=12.0,
                    help="no traffic spawns closer than this to the ego, so the "
                         "run does not begin with a vehicle on top of it")
    ap.add_argument("--tm-speed-delta-pct", type=float, default=0.0,
                    help="Traffic Manager speed offset; 0 means the posted limit")
    ap.add_argument("--pcla-dropout-gain", type=float, default=0.4,
                    help="Matsense dropout_gain passed to PerturbConfig (default 0.4 = calibrated)")
    ap.add_argument("--pcla-perturb-seed", type=int, default=1234,
                    help="Random seed used by the closed-loop LiDAR perturbation")
    ap.add_argument("--pcla-lane-guard", action="store_true",
                    help="Apply an external lane-centering correction to PCLA controls. "
                         "Do not enable for closed-loop policy evaluation.")
    ap.add_argument("--pcla-max-seconds", type=float, default=0.0,
                    help="Stop a PCLA run after this simulated duration; 0 disables the limit")
    ap.add_argument("--pcla-sensor-prime-ticks", type=int, default=4,
                    help="Initial world ticks used to prime PCLA sensors before the first action.")
    ap.add_argument("--pcla-stopped-seconds", type=float, default=0.0,
                    help="end the run once the ego has been stationary this long "
                         "(0 = disabled). The resting position is the safety metric, so "
                         "everything after it is duplicate frames.")
    ap.add_argument("--pcla-stopped-speed", type=float, default=0.1,
                    help="speed (m/s) below which the ego counts as stationary")
    ap.add_argument("--pcla-route-dev-threshold", type=float, default=5.0,
                    help="Terminate the PCLA run when cross-track error exceeds this threshold in metres; 0 disables it.")
    ap.add_argument("--brake-threshold", type=float, default=0.05,
                    help="Brake threshold used to detect the first brake application in summaries.")
    ap.add_argument("--hazard-sensor-range-m", type=float, default=80.0,
                    help="Distance threshold used to declare the hazard obstacle in range.")
    ap.add_argument("--hazard-obstacle-source",
                    choices=["none", "first_parked", "first_dynamic", "first_walker"],
                    default="none",
                    help="Which spawned actor is tracked as the hazard, so the "
                         "clog carries its distance. Without it obstacle_distance_m "
                         "is empty and there is no way afterwards to say when the "
                         "agent reacted to the hazard rather than to something "
                         "else: that gap is why the invading-bend runs could not "
                         "be attributed without a separate reconstruction.")
    ap.add_argument("--condition-type", type=str, default="scenario",
                    help="Logical experiment condition label written to per-run outputs.")
    ap.add_argument("--replicate-index", type=int, default=0,
                    help="Explicit replicate index written to per-run outputs.")
    ap.add_argument("--camera-degrade", action="store_true",
                    help="Add fog to rain/snow presets to degrade camera visibility "
                         "(fog_density=25, fog_distance=40 for rain; boosted for snow). "
                         "Makes the Transfuser RGB+LiDAR agent rely more on LiDAR.")

    ap.add_argument("--cam-fov", type=float, default=90.0)
    ap.add_argument("--sensor-width", type=int, default=0,
                    help="Resolution width for viewer cameras (0 = match panel width). Lower = faster.")
    ap.add_argument("--no-draw", action="store_true",
                    help="Skip drawing the pygame window entirely. The simulation, "
                         "the agent and every recorded number are unaffected: only "
                         "the on-screen presentation is dropped. Meant for headless "
                         "campaign runs, where it roughly halves the step time. "
                         "Ignored when screenshots are requested, since those come "
                         "out of the same path.")
    ap.add_argument("--disable-viewer-cameras", action="store_true",
                    help="Do not spawn the viewer RGB/semantic/bird's-eye cameras. Useful for Linux headless agent-only runs.")
    ap.add_argument("--birdseye-height", type=float, default=18.0,
                    help="Height in metres of the bird's-eye RGB camera above the vehicle roof")
    ap.add_argument("--spectator-follow", choices=["off", "chase", "hood", "roof"], default="off",
                    help="Move the Unreal spectator with the ego vehicle from inside the main CARLA client.")

    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--pps", type=int, default=1300000)
    ap.add_argument("--rotation-frequency", type=float, default=20.0)
    ap.add_argument("--lidar-range", type=float, default=80.0)
    ap.add_argument("--horizontal-fov", type=float, default=360.0)
    ap.add_argument("--upper-fov", type=float, default=10.0)
    ap.add_argument("--lower-fov", type=float, default=-30.0)
    ap.add_argument(
        "--enable-semantic-lidar",
        action="store_true",
        help="Also spawn CARLA's semantic ray-cast LiDAR. Disabled by default because some CARLA builds return empty scans when it runs alongside the normal LiDAR.",
    )

    ap.add_argument("--forward-m", type=float, default=45.0)
    ap.add_argument("--behind-m", type=float, default=12.0)
    ap.add_argument("--side-m", type=float, default=25.0)
    ap.add_argument("--min-z", type=float, default=-3.0)
    ap.add_argument("--max-z", type=float, default=5.0)

    ap.add_argument("--ref-distance", type=float, default=10.0,
                    help="Reference distance for pseudo-reflectance distance correction")
    ap.add_argument("--max-range-gain", type=float, default=6.0,
                    help="Legacy arg, kept for compatibility")
    ap.add_argument("--display-vmax", type=float, default=DEFAULT_DISPLAY_VMAX,
                    help="Pseudo-reflectance compression upper bound for display")
    ap.add_argument("--display-normalization", choices=["fixed", "percentile"], default="percentile",
                    help="How to normalize pseudo-reflectance for the GUI; percentile is better for visual comparison")
    ap.add_argument("--display-percentile", type=float, default=99.0,
                    help="Percentile used when display-normalization=percentile")
    ap.add_argument("--use-base-nominal", action="store_true",
                    help="Use material nominal base values directly instead of raw-intensity-driven empirical correction")
    ap.add_argument("--save-dataset", action="store_true")
    ap.add_argument("--dataset-root", type=str, default="output_dataset")
    ap.add_argument("--scene-id", type=str, default="scene_001")
    ap.add_argument("--scenario-name", type=str, default="nominal")
    ap.add_argument("--material-config", type=str, default=str(DEFAULT_TOOL_CONFIG_PATH),
                    help="Path to a JSON tool config defining material profiles and launcher presets")
    ap.add_argument("--profile-name", type=str, default=_DEFAULT_PROFILE_NAME,
                    help="Material profile name inside the tool config")
    ap.add_argument("--profile-version", type=str, default="",
                    help="Version label for the active material profile. Defaults to the resolved profile name.")
    ap.add_argument("--material-overrides", type=str, default="",
                    help="Optional JSON mapping actor_ids or CARLA type_ids to material names such as asphalt, sidewalk, building, vegetation, car, or unknown")
    ap.add_argument("--max-save-frames", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=1)
    ap.add_argument("--save-last-seconds", type=float, default=0.0,
                    help="If > 0, keep a rolling window and write only the last N seconds of synchronized frames on exit")
    ap.add_argument("--save-start-delay-seconds", type=float, default=0.0,
                    help="Delay dataset recording after spawn; use 0.0 to save from the first synchronized frame")
    ap.add_argument("--screenshot-dir", type=str, default="",
                    help="Optional directory for saving rendered pygame display screenshots")
    ap.add_argument("--screenshot-every", type=int, default=1,
                    help="Save one screenshot every N rendered frames when --screenshot-dir is set")
    ap.add_argument("--screenshot-max-frames", type=int, default=0,
                    help="Exit after saving this many screenshots; 0 means keep running")
    ap.add_argument("--strict-sync", action="store_true",
                    help="Wait for the exact RGB/semantic/LiDAR frame after each world tick before saving the dataset")
    ap.add_argument("--strict-sync-timeout", type=float, default=0.5,
                    help="Maximum wait time in seconds for the exact sensor frame when strict-sync is enabled")
    ap.add_argument("--open-loop-matsense", action="store_true",
                    help="Apply MatSense to synchronized LiDAR clouds without starting a PCLA driving agent.")
    ap.add_argument("--keep-running-after-traj", action="store_true",
                    help="Keep ticking after a teleported replay runs out of poses. "
                         "Off by default: the ego is parked on the last pose and "
                         "every further frame is the same static view.")
    ap.add_argument("--open-loop-arm", type=str, default="matsense",
                    choices=["standard", "global", "matsense", "pertov"],
                    help="Which sensing arm the open-loop operator runs. --mode is "
                         "the display colormap and does NOT select this: three runs "
                         "that differ only in --mode carry the same operator and "
                         "cannot be compared as arms.")
    ap.add_argument("--pcla-agent", type=str, default="",
                    help="PCLA agent name (e.g. 'tfv4_l6_0'). When set, PCLA drives instead of autopilot/manual.")
    ap.add_argument("--pcla-route", type=str, default="",
                    help="Path to route XML for PCLA. Defaults to PCLA/sample_route.xml.")
    ap.add_argument("--pcla-dir", type=str, default="",
                    help="Path to the PCLA repo root. Defaults to matsense_streamlit_app/PCLA, then ../../../../PCLA relative to this script.")
    ap.add_argument("--pcla-town", type=str, default="Town02",
                    help="CARLA town to load when using --pcla-agent (default: Town02 for sample_route.xml).")
    ap.add_argument("--pcla-spawn-jitter-m", type=float, default=0.0,
                    help="Deterministic centimetre-scale jitter on the ego's starting pose, "
                         "derived from (seed, replicate). Without it, runs that apply no "
                         "perturbation are bit-identical: in intensity mode the seed only "
                         "feeds the perturbation RNG, which is never called, so nine "
                         "replicates are nine copies of one run and the control group has "
                         "exactly zero variance - which inflates every t-test against it. "
                         "Default 0.0 keeps the old behaviour.")
    ap.add_argument("--pcla-spawn-x", type=float, default=None,
                    help="Override the ego spawn x for PCLA runs. Use when the ego must not "
                         "start on its own route, e.g. ParkingExit: the route begins where the "
                         "car rejoins the carriageway and the bay pose is given here.")
    ap.add_argument("--pcla-spawn-y", type=float, default=None,
                    help="Override the ego spawn y for PCLA runs. Both x and y must be given.")
    ap.add_argument("--pcla-spawn-z", type=float, default=None)
    ap.add_argument("--pcla-spawn-yaw", type=float, default=None,
                    help="Override the ego spawn yaw. Defaults to the route's first waypoint yaw.")
    ap.add_argument("--pcla-spawn-index", type=int, default=31,
                    help="Spawn point index in the town when using --pcla-agent.")
    return ap


def main():
    args = build_argparser().parse_args()
    tool_config, config_path = load_tool_config(args.material_config)
    profile_name, profile = get_profile(tool_config, args.profile_name)
    apply_material_profile(profile_name, profile, config_path)
    args.material_config = str(config_path)
    args.profile_name = profile_name
    args.profile_version = args.profile_version or profile_name
    exit_code = 0
    app = CarlaLidarViewer(args)
    try:
        app.run()
    except KeyboardInterrupt:
        pass
    except Exception:
        # Printed here, not left to the interpreter. The finally below ends the
        # process with os._exit, which skips the traceback Python would normally
        # write on the way out - so a run that died on a real error looked
        # exactly like one that ended quietly, and the only clue was the exit
        # code. Every failure diagnosed by guesswork in this harness was this.
        exit_code = 1
        traceback.print_exc()
        sys.stderr.flush()
    finally:
        app.destroy()
        if os.environ.get("MATSENSE_FORCE_EXIT_ON_SHUTDOWN", "1").strip() != "0":
            try:
                sys.stdout.flush()
                sys.stderr.flush()
            finally:
                os._exit(exit_code)


if __name__ == "__main__":
    main()
