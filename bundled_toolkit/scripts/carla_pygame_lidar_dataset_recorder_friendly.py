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
P : toggle color mode (intensity / pseudo)
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
from collections import OrderedDict, deque
import json
import math
import sys
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
        "pseudo": "Pseudo Reflectance",
        "material": "Material Classes",
        CAMERA_TRIPLE_MODE: "RGB + LiDAR Overlays",
    }.get(mode, mode.title())


def mode_description(mode: str) -> str:
    return {
        "intensity": "Direct LiDAR return strength from the sensor",
        "pseudo": "Range-corrected LiDAR response with weather/material priors",
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
    raw = item["start"]
    heading = float(item.get("heading", 0.0))
    return carla.Transform(
        carla.Location(x=float(raw[0]), y=float(raw[1]), z=float(raw[2]) + z_offset),
        carla.Rotation(yaw=heading),
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


def pick_parked_blueprint(blueprints, item: dict):
    r, g, b = parked_color_tuple(item)
    idx = (r + g * 3 + b * 7) % len(blueprints)
    bp = blueprints[idx]
    if bp.has_attribute("color"):
        try:
            bp.set_attribute("color", f"{r},{g},{b}")
        except Exception:
            pass
    return bp


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
    if mode == "pseudo":
        return colormap_pseudo(pseudo_norm)
    if mode == "material_effect":
        return colormap_pseudo(normalize_material_effect(material_effect))
    return material_to_rgb(materials)


def parse_image(image: "carla.Image") -> np.ndarray:
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = arr.reshape((image.height, image.width, 4))
    return arr


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

    def put(self, frame_id: int, data) -> None:
        frame_id = int(frame_id)
        with self.lock:
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
        for d in [self.rgb_dir, self.semantic_dir, self.semantic_vis_dir,
                  self.projection_debug_dir, self.lidar_raw_dir, self.lidar_labels_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.frame_meta_csv = self.root / 'frame_metadata.csv'
        self._saved = 0
        self._accepted = 0
        self._pending_frames = deque()
        self._buffer_mode = getattr(self.args, "save_last_seconds", 0.0) > 0.0
        self._buffer_capacity = self._compute_buffer_capacity()
        self._init_frame_metadata_csv()
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
        if not self.frame_meta_csv.exists():
            with open(self.frame_meta_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'scene_id', 'scenario', 'frame_id', 'timestamp',
                    'ego_x', 'ego_y', 'ego_z',
                    'ego_roll', 'ego_pitch', 'ego_yaw',
                    'weather', 'num_points',
                    'projected_points', 'projection_ratio',
                    'known_material_points', 'known_material_ratio'
                ])

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
                   projected_points: int = 0, known_material_points: int = 0):
        payload = {
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
        np.savez_compressed(self.lidar_labels_dir / f'{stem}.npz', **pack)

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
                known_material_points, float(known_material_points / max(num_points, 1))
            ])
        self._saved += 1


class CarlaLidarViewer:
    def __init__(self, args):
        self.args = args
        self.client = carla.Client(args.host, args.port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.original_settings = self.world.get_settings()
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
        self.rgb_cam = None
        self.sem_cam = None
        self.lidar = None
        self.semantic_lidar = None

        self.rgb_buffer = SensorBuffer(maxsize=24)
        self.sem_buffer = SensorBuffer(maxsize=24)
        self.lidar_buffer = SensorBuffer(maxsize=24)
        self.semantic_lidar_buffer = SensorBuffer(maxsize=24)

        self.rgb_array = None
        self.sem_array = None
        self.sem_vis_array = None
        self.last_lidar = None
        self.last_semantic_lidar = None
        self.last_semantic_lidar = None
        self.rgb_frame = None
        self.sem_frame = None
        self.lidar_frame = None
        self.semantic_lidar_frame = None
        self.last_saved_frame = -1
        self.recording_start_time = None
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
        self._validate_map_alignment()

        if getattr(args, "save_dataset", False):
            args.map_name = self.map.name
            self.recorder = DatasetRecorder(args.dataset_root, args.scene_id, args.scenario_name, args)

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
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode(self.display_size, pygame.DOUBLEBUF)
        pygame.display.set_caption("CARLA Material-Aware LiDAR Demo")
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

        return {
            "title": pygame.Rect(margin, margin, self.args.width - 2 * margin, title_h),
            "camera": pygame.Rect(left_x, content_top, left_w, left_h),
            "status": pygame.Rect(right_x, content_top, right_w, status_h),
            "controls": pygame.Rect(right_x, content_top + status_h + gap, right_w, controls_h),
            "bev": pygame.Rect(right_x, content_top + status_h + gap + controls_h + gap, right_w, bev_h),
        }

    def destroy(self):
        if self.recorder is not None:
            self.recorder.finalize()
        for actor in self.actors[::-1]:
            try:
                actor.destroy()
            except Exception:
                pass
        self.actors = []
        try:
            self.world.apply_settings(self.original_settings)
        except Exception:
            pass
        try:
            self.tm.set_synchronous_mode(False)
        except Exception:
            pass
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
            bp = pick_parked_blueprint(blueprints, item)
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
            spawned += 1
        print(f"[toolkit] parked_json: spawned {spawned}/{len(positions)} failed={failures}", flush=True)

    def spawn_vehicle_and_sensors(self):
        print("[toolkit] spawn_vehicle_and_sensors: start", flush=True)
        self._destroy_runtime_actors_only()
        print("[toolkit] spawn_vehicle_and_sensors: runtime actors cleared", flush=True)
        self._clear_world_runtime_actors()

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
            for idx, pose in enumerate(self.trajectory_poses):
                candidate = pose_to_road_transform(
                    self.world,
                    pose,
                    float(getattr(self.args, "traj_z_offset", 0.0)),
                )
                self.vehicle = self.world.try_spawn_actor(veh_bp, candidate)
                if self.vehicle is not None:
                    spawn = candidate
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
        if self.vehicle is None:
            # fallback search across spawn points
            spawn_points = self.map.get_spawn_points()
            for sp in spawn_points:
                self.vehicle = self.world.try_spawn_actor(veh_bp, sp)
                if self.vehicle is not None:
                    spawn = sp
                    break
        if self.vehicle is None:
            raise RuntimeError("Could not spawn vehicle")
        print(f"[toolkit] spawn_vehicle_and_sensors: ego spawned id={self.vehicle.id}", flush=True)
        self.actors.append(self.vehicle)
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

        if self.args.autopilot:
            self.vehicle.set_autopilot(True, self.tm.get_port())
            self.tm.ignore_lights_percentage(self.vehicle, 0.0)

        cam_w = self.left_panel_w
        cam_h = self.args.height
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

        self._apply_weather()
        print(
            f"[toolkit] spawn_vehicle_and_sensors: sensors spawned, priming ticks dt={sensor_dt:.4f} "
            f"lidar_hz={float(self.args.fps):.1f}",
            flush=True,
        )

        # prime sensors just enough to receive the first synchronized frames
        for _ in range(2):
            self.world.tick()
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

    def _destroy_runtime_actors_only(self):
        for actor in self.actors[::-1]:
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
            weather = carla.WeatherParameters(
                cloudiness=80.0,
                precipitation=65.0,
                precipitation_deposits=70.0,
                wetness=85.0,
                fog_density=12.0,
                sun_altitude_angle=25.0,
            )
        elif preset == "snow":
            # CARLA has no native snowfall parameter, so this is a visual
            # winter-like approximation based on cold lighting, surface deposits,
            # haze, and strong cloud cover.
            weather = carla.WeatherParameters(
                cloudiness=95.0,
                precipitation=8.0,
                precipitation_deposits=95.0,
                wetness=35.0,
                wind_intensity=20.0,
                fog_density=22.0,
                fog_distance=35.0,
                fog_falloff=0.2,
                scattering_intensity=1.0,
                mie_scattering_scale=0.03,
                rayleigh_scattering_scale=0.0331,
                sun_altitude_angle=8.0,
            )
        elif preset == "fog":
            weather = carla.WeatherParameters(
                cloudiness=60.0,
                precipitation=0.0,
                precipitation_deposits=0.0,
                wetness=10.0,
                fog_density=55.0,
                fog_distance=8.0,
                sun_altitude_angle=18.0,
            )
        else:
            weather = getattr(carla.WeatherParameters, "ClearNoon")
        self.world.set_weather(weather)

    @staticmethod
    def _make_camera_callback(buffer_obj: SensorBuffer, semantic=False):
        def _callback(image):
            if semantic:
                raw_arr = np.frombuffer(image.raw_data, dtype=np.uint8).copy()
                raw_arr = raw_arr.reshape((image.height, image.width, 4))

                image.convert(carla.ColorConverter.CityScapesPalette)
                vis_arr = parse_image(image)

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
        material_effect = material_response_values(materials, self.args.weather)

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
        elif mode == "pseudo":
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
        modes = ["material", "intensity", "pseudo"]
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
        )
        self.rgb_buffer.discard_through(frame_id)
        self.sem_buffer.discard_through(frame_id)
        self.lidar_buffer.discard_through(frame_id)
        self.last_saved_frame = frame_id

    def run(self):
        print("[toolkit] run: setup_pygame", flush=True)
        self.setup_pygame()
        print("[toolkit] run: spawning vehicle and sensors", flush=True)
        self.request_respawn("startup", force=True)
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
                    if event.key == pygame.K_r:
                        self.request_respawn("keyboard_r")

            if self.trajectory_poses:
                self.follow_trajectory_step()
            else:
                self.manual_control()
            if loop_count < 3:
                print(f"[toolkit] before_tick loop={loop_count}", flush=True)
            target_frame = self.world.tick()
            if loop_count < 3:
                print(f"[toolkit] after_tick loop={loop_count} frame={target_frame}", flush=True)
            loop_count += 1
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
            self._update_display_from_synced_frame(target_frame)
            self.try_record_current_frame(target_frame)

            if (loop_count % self.render_stride) != 0:
                self.clock.tick(self.args.fps)
                continue

            self.screen.fill(BG_COLOR)
            layout = self.get_layout()
            current_color_info = self.last_color_info
            if self.color_mode == CAMERA_TRIPLE_MODE:
                self.draw_triple_camera_view(current_color_info)
                pygame.display.flip()
                self.clock.tick(self.args.fps)
                continue

            frame_rect = layout["camera"]
            draw_card(self.screen, frame_rect)
            if self.rgb_array is not None:
                left_image = self.rgb_array
                if current_color_info is not None and self.color_mode in ("intensity", "pseudo", "material"):
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
                "RGB + LiDAR overlay" if self.color_mode in ("intensity", "pseudo", "material") else "Raw RGB",
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
            self.clock.tick(self.args.fps)


def build_argparser():
    ap = argparse.ArgumentParser(description="CARLA pygame LiDAR pseudo-reflectance viewer")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--tm-port", type=int, default=8000)
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
    ap.add_argument("--parked-z-offset", type=float, default=0.5,
                    help="Vertical offset used for parked vehicles loaded from --parked-json")
    ap.add_argument("--parked-limit", type=int, default=0,
                    help="Spawn only the first N parked vehicles from --parked-json; 0 means all")
    ap.add_argument("--seed", type=int, default=42,
                    help="Deterministic seed for parked vehicle blueprint selection")
    ap.add_argument("--weather", choices=["nominal", "rain", "snow", "fog"], default="nominal")
    ap.add_argument("--mode", choices=["intensity", "pseudo", "material", CAMERA_TRIPLE_MODE], default="pseudo")

    ap.add_argument("--cam-fov", type=float, default=90.0)

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
    ap.add_argument("--material-overrides", type=str, default="",
                    help="Optional JSON mapping actor_ids or CARLA type_ids to material names such as wood or metal")
    ap.add_argument("--max-save-frames", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=1)
    ap.add_argument("--save-last-seconds", type=float, default=0.0,
                    help="If > 0, keep a rolling window and write only the last N seconds of synchronized frames on exit")
    ap.add_argument("--save-start-delay-seconds", type=float, default=0.0,
                    help="Delay dataset recording after spawn; use 0.0 to save from the first synchronized frame")
    ap.add_argument("--strict-sync", action="store_true",
                    help="Wait for the exact RGB/semantic/LiDAR frame after each world tick before saving the dataset")
    ap.add_argument("--strict-sync-timeout", type=float, default=0.5,
                    help="Maximum wait time in seconds for the exact sensor frame when strict-sync is enabled")
    return ap


def main():
    args = build_argparser().parse_args()
    tool_config, config_path = load_tool_config(args.material_config)
    profile_name, profile = get_profile(tool_config, args.profile_name)
    apply_material_profile(profile_name, profile, config_path)
    args.material_config = str(config_path)
    args.profile_name = profile_name
    app = CarlaLidarViewer(args)
    try:
        app.run()
    except KeyboardInterrupt:
        pass
    finally:
        app.destroy()


if __name__ == "__main__":
    main()
