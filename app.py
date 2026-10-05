from __future__ import annotations

import csv
import json
import math
import os
import re
import signal
import socket
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
BUNDLED_TOOLKIT = APP_DIR / "bundled_toolkit"
DEFAULT_TOOLKIT_TEXT = os.environ.get("MATSENSE_TOOLKIT", str(BUNDLED_TOOLKIT) if BUNDLED_TOOLKIT.exists() else "")
SAMPLE_DATASET_ROOT = APP_DIR / "sample_data"
OUTPUT_ANALYSIS = APP_DIR / "output_analysis"
PID_FILE = Path(__file__).with_name(".matsense_viewer.pid")
VIEWER_STDOUT = APP_DIR / "viewer_stdout.log"
VIEWER_STDERR = APP_DIR / "viewer_stderr.log"
VIEWER_COMMAND = APP_DIR / "viewer_command.txt"
CAMPAIGN_ROOT = APP_DIR / "output_campaigns"

DEFAULT_CONFIG = {
    "host": "127.0.0.1",
    "port": "2000",
    "tm_port": "8000",
    "width": "1600",
    "height": "900",
    "fps": "20",
    "weather": "nominal",
    "mode": "material",
    "display_normalization": "fixed",
    "display_percentile": "95",
    "dataset_root": str(SAMPLE_DATASET_ROOT),
    "scene_id": "scene_001",
    "scenario_name": "nominal",
    "traj_txt": "",
    "traj_json": "",
    "parked_json": "",
    "autopilot": False,
    "save_dataset": False,
    "profile_name": "",
    "traj_step": "5",
    "utm_offset_x": "0.0",
    "utm_offset_y": "0.0",
    "traj_z_offset": "0.5",
    "follow_mode": "teleport",
    "control_smoothing": "0.30",
    "parked_z_offset": "0.15",
    "parked_limit": "0",
    "seed": "42",
    "max_save_frames": "0",
    "save_every": "10",
    "save_start_delay_seconds": "0.0",
    "strict_sync_timeout": "0.5",
}


st.set_page_config(page_title="MatSense", layout="wide")

BLUE_SCALE = ["#E0FBFC", "#9DD9D2", "#5BC0DE", "#118AB2", "#0B5F7A", "#073B4C"]
PSEUDO_SCALE = ["#F7FBFF", "#CDEDF6", "#5BC0DE", "#118AB2", "#FFD166", "#FF8811"]
SCENARIO_COLORS = {
    "nominal": "#118AB2",
    "rain": "#5BC0DE",
    "snow": "#9DD9D2",
}
MATERIAL_COLORS = {
    "asphalt": "#4A4A4A",
    "sidewalk": "#FFD166",
    "building": "#FF8811",
    "vegetation": "#06D6A0",
    "car": "#EF476F",
    "unknown": "#B0B7C3",
}

st.markdown(
    """
    <style>
      .stApp {
        background: linear-gradient(180deg, #f7fbff 0%, #eef6fb 100%);
      }
      [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #d8edf6 0%, #eef6fb 100%);
        border-right: 1px solid #b8d7e6;
      }
      [data-testid="stSidebar"] h1,
      [data-testid="stSidebar"] h2,
      [data-testid="stSidebar"] h3,
      [data-testid="stSidebar"] label,
      [data-testid="stSidebar"] p,
      [data-testid="stSidebar"] span,
      [data-testid="stSidebar"] div {
        color: #073B4C;
      }
      [data-testid="stSidebar"] input,
      [data-testid="stSidebar"] textarea,
      [data-testid="stSidebar"] select,
      [data-testid="stSidebar"] [data-baseweb="input"] *,
      [data-testid="stSidebar"] [data-baseweb="select"] *,
      [data-testid="stSidebar"] [data-baseweb="textarea"] * {
        color: #073B4C !important;
      }
      [data-testid="stSidebar"] [data-baseweb="input"],
      [data-testid="stSidebar"] [data-baseweb="select"],
      [data-testid="stSidebar"] [data-baseweb="textarea"] {
        background-color: #ffffff !important;
      }
      div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #9bc8dc;
        border-left: 5px solid #118AB2;
        border-radius: 8px;
        padding: 12px;
      }
      div[data-testid="stMetric"] label {
        color: #0B5F7A !important;
      }
      [data-testid="stHeader"] {
        background: #073B4C;
      }
      [data-baseweb="tab-list"] {
        background: #d8edf6;
        border-radius: 8px;
        padding: 4px;
      }
      [data-baseweb="tab"] {
        color: #073B4C;
      }
      [aria-selected="true"][data-baseweb="tab"] {
        background: #118AB2;
        border-radius: 6px;
        color: #ffffff;
      }
      .stButton > button,
      .stFormSubmitButton > button {
        background: #118AB2;
        color: #ffffff;
        border: 1px solid #0B5F7A;
      }
      .stButton > button:hover,
      .stFormSubmitButton > button:hover {
        background: #0B5F7A;
        color: #ffffff;
        border: 1px solid #073B4C;
      }
      h1, h2, h3 {
        color: #073B4C;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def trajectory_txt_looks_like_utm(path_text: str) -> bool:
    if not path_text.strip():
        return False
    path = Path(path_text)
    if not path.exists():
        return False
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 5:
                    continue
                x = abs(float(parts[2]))
                y = abs(float(parts[3]))
                return x > 10000.0 or y > 10000.0
    except Exception:
        return False
    return False


def candidate_xodr_paths(traj_txt: str, toolkit_dir: Path | None) -> list[Path]:
    candidates: list[Path] = []
    if traj_txt.strip():
        traj_dir = Path(traj_txt).parent
        candidates.extend([traj_dir / "mappa.xodr", traj_dir / "map.xodr", traj_dir / "fortiss.xodr"])
    candidates.extend(
        [
            Path.home() / "Downloads" / "mappa.xodr",
            Path.home() / "Downloads" / "map.xodr",
            Path.home() / "Downloads" / "fortiss.xodr",
            APP_DIR / "mappa.xodr",
            APP_DIR / "map.xodr",
            APP_DIR / "fortiss.xodr",
        ]
    )
    if toolkit_dir is not None:
        candidates.extend([toolkit_dir / "mappa.xodr", toolkit_dir / "map.xodr", toolkit_dir / "fortiss.xodr"])

    seen: set[Path] = set()
    unique: list[Path] = []
    for path in candidates:
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def read_xodr_offset(traj_txt: str, toolkit_dir: Path | None) -> tuple[float, float, Path] | None:
    pattern = re.compile(
        r"<offset\b[^>]*\bx=\"(?P<x>[-+0-9.eE]+)\"[^>]*\by=\"(?P<y>[-+0-9.eE]+)\"",
        re.IGNORECASE,
    )
    for path in candidate_xodr_paths(traj_txt, toolkit_dir):
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        match = pattern.search(text)
        if not match:
            continue
        try:
            return float(match.group("x")), float(match.group("y")), path
        except ValueError:
            continue
    return None


def load_launcher_config(toolkit_dir: Path | None) -> dict:
    if toolkit_dir is None:
        return dict(DEFAULT_CONFIG)
    config_path = toolkit_dir / "carla_material_aware_launcher_config.json"
    config = dict(DEFAULT_CONFIG)
    config.update(read_json(config_path))
    return config


def load_tool_config(toolkit_dir: Path | None) -> dict:
    if toolkit_dir is None:
        return {}
    try:
        return read_json(toolkit_dir / "configs" / "material_aware_tool_config.json")
    except (OSError, ValueError):
        return {}


# What the viewer's --mode accepts for display. "global" is the
# pseudo-reflectance colormap; older launcher configs called it "pseudo",
# which the viewer rejects at argument parsing.
VIEW_MODE_ALIASES = {"pseudo": "global"}

WEATHER_OPTIONS = ["nominal", "rain", "snow"]
VIEW_OPTIONS = {
    "camera_triple": "RGB overlays: all 3",
    "material": "Material classes",
    "global": "Pseudo-reflectance",
    "intensity": "CARLA raw intensity",
}

# LiDAR models the Run Viewer can emulate. The viewer completes one full
# sweep per simulation frame (its rotation frequency is the FPS), so the
# density is given per sweep and converted to points per second at launch.
LIDAR_MODELS = {
    "Velodyne VLP-32C (real recordings)": {
        "channels": 32, "points_per_sweep": 57600, "lidar_range": 200.0, "upper_fov": 15.0, "lower_fov": -25.0,
        "note": "32 beams from -25 to +15 deg, 0.2 deg azimuth at 10 Hz (1800 x 32 returns per sweep), 200 m.",
    },
    "MatSense paper (CARLA, 64 ch)": {
        "channels": 64, "points_per_sweep": 60000, "lidar_range": 85.0, "upper_fov": 10.0, "lower_fov": -30.0,
        "note": "The closed-loop setup of the paper: 64 channels, 85 m, 600 000 points/s at 10 Hz.",
    },
    "Viewer default (64 ch)": {
        "channels": 64, "points_per_sweep": 65000, "lidar_range": 80.0, "upper_fov": 10.0, "lower_fov": -30.0,
        "note": "The viewer script's own defaults: 1.3 M points/s at 20 FPS.",
    },
}
DEFAULT_LIDAR_MODEL = "Velodyne VLP-32C (real recordings)"

# Conditions whose ratios are carried over rather than measured from real
# recordings, whatever profile is selected.
UNMEASURED_CONDITIONS = {
    "snow": "Snow ratios are carried over, not measured: no real snow recordings were used for calibration.",
}


def list_scenes(dataset_root: Path) -> list[str]:
    if not dataset_root.exists():
        return []
    return sorted([p.name for p in dataset_root.iterdir() if p.is_dir()])


def list_scenarios(scene_dir: Path) -> list[str]:
    if not scene_dir.exists():
        return []
    return sorted([p.name for p in scene_dir.iterdir() if (p / "frame_metadata.csv").exists()])


def route_distance(df: pd.DataFrame) -> float:
    if len(df) < 2 or not {"ego_x", "ego_y"}.issubset(df.columns):
        return 0.0
    dx = df["ego_x"].diff().fillna(0.0)
    dy = df["ego_y"].diff().fillna(0.0)
    return float(np.sqrt(dx * dx + dy * dy).sum())


def summarize_frame_metadata(scene_dir: Path, scenarios: list[str]) -> pd.DataFrame:
    rows = []
    for scenario in scenarios:
        csv_path = scene_dir / scenario / "frame_metadata.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        duration = float(df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]) if "timestamp" in df else 0.0
        row = {
            "scenario": scenario,
            "frames": len(df),
            "duration_s": duration,
            "distance_m": route_distance(df),
            "lidar_points_mean": float(df["num_points"].mean()),
            "projected_points_mean": float(df["projected_points"].mean()),
            "projection_ratio_mean": float(df["projection_ratio"].mean()),
            "known_material_points_mean": float(df["known_material_points"].mean()),
            "known_material_ratio_mean": float(df["known_material_ratio"].mean()),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_lidar_labels(scene_dir: Path, scenarios: list[str], max_files_per_scenario: int = 0) -> pd.DataFrame:
    rows = []
    for scenario in scenarios:
        label_dir = scene_dir / scenario / "lidar_labels"
        raw_dir = scene_dir / scenario / "lidar_raw"
        files = sorted(label_dir.glob("*.npz"))
        if max_files_per_scenario > 0:
            files = files[:max_files_per_scenario]
        stats: dict[str, dict[str, float]] = defaultdict(
            lambda: {
                "n": 0,
                "pseudo_sum": 0.0,
                "pseudo2_sum": 0.0,
                "pseudo_norm_sum": 0.0,
                "pseudo_norm2_sum": 0.0,
                "intensity_norm_sum": 0.0,
                "intensity_norm2_sum": 0.0,
                "raw_intensity_sum": 0.0,
                "raw_intensity2_sum": 0.0,
                "range_sum": 0.0,
            }
        )
        for file_path in files:
            data = np.load(file_path, allow_pickle=True)
            materials = data["material_label"].astype(str)
            pseudo = data["pseudo_final"].astype(float)
            pseudo_norm = data["pseudo_norm"].astype(float) if "pseudo_norm" in data else np.zeros_like(pseudo)
            intensity_norm = data["intensity_norm"].astype(float) if "intensity_norm" in data else np.zeros_like(pseudo)
            ranges = data["range"].astype(float)
            raw_intensity = intensity_norm
            raw_path = raw_dir / file_path.name
            if raw_path.exists():
                raw = np.load(raw_path, allow_pickle=True)
                if "intensity" in raw and len(raw["intensity"]) == len(materials):
                    raw_intensity = raw["intensity"].astype(float)
            for material in np.unique(materials):
                mask = materials == material
                p = pseudo[mask]
                pn = pseudo_norm[mask]
                inn = intensity_norm[mask]
                ri = raw_intensity[mask]
                item = stats[str(material)]
                n = int(mask.sum())
                item["n"] += n
                item["pseudo_sum"] += float(p.sum())
                item["pseudo2_sum"] += float((p * p).sum())
                item["pseudo_norm_sum"] += float(pn.sum())
                item["pseudo_norm2_sum"] += float((pn * pn).sum())
                item["intensity_norm_sum"] += float(inn.sum())
                item["intensity_norm2_sum"] += float((inn * inn).sum())
                item["raw_intensity_sum"] += float(ri.sum())
                item["raw_intensity2_sum"] += float((ri * ri).sum())
                item["range_sum"] += float(ranges[mask].sum())
        total = sum(int(v["n"]) for v in stats.values())
        for material, item in stats.items():
            n = int(item["n"])
            if n == 0:
                continue
            mean = item["pseudo_sum"] / n
            var = max(item["pseudo2_sum"] / n - mean * mean, 0.0)
            pseudo_norm_mean = item["pseudo_norm_sum"] / n
            pseudo_norm_var = max(item["pseudo_norm2_sum"] / n - pseudo_norm_mean * pseudo_norm_mean, 0.0)
            intensity_norm_mean = item["intensity_norm_sum"] / n
            intensity_norm_var = max(item["intensity_norm2_sum"] / n - intensity_norm_mean * intensity_norm_mean, 0.0)
            raw_intensity_mean = item["raw_intensity_sum"] / n
            raw_intensity_var = max(item["raw_intensity2_sum"] / n - raw_intensity_mean * raw_intensity_mean, 0.0)
            rows.append(
                {
                    "scenario": scenario,
                    "material": material,
                    "points": n,
                    "share": n / max(total, 1),
                    "pseudo_mean": mean,
                    "pseudo_std": math.sqrt(var),
                    "pseudo_norm_mean": pseudo_norm_mean,
                    "pseudo_norm_std": math.sqrt(pseudo_norm_var),
                    "carla_intensity_norm_mean": intensity_norm_mean,
                    "carla_intensity_norm_std": math.sqrt(intensity_norm_var),
                    "carla_raw_intensity_mean": raw_intensity_mean,
                    "carla_raw_intensity_std": math.sqrt(raw_intensity_var),
                    "matsense_minus_carla_norm": pseudo_norm_mean - intensity_norm_mean,
                    "range_mean_m": item["range_sum"] / n,
                }
            )
    return pd.DataFrame(rows)


def add_ratio_to_baseline(material_df: pd.DataFrame, baseline: str) -> pd.DataFrame:
    if material_df.empty or baseline not in set(material_df["scenario"]):
        return material_df.copy()
    base = material_df[material_df["scenario"] == baseline][["material", "pseudo_mean", "pseudo_norm_mean"]]
    base = base.rename(columns={"pseudo_mean": "baseline_pseudo_mean", "pseudo_norm_mean": "baseline_pseudo_norm_mean"})
    out = material_df.merge(base, on="material", how="left")
    out["pseudo_delta_vs_baseline"] = out["pseudo_mean"] - out["baseline_pseudo_mean"]
    out["pseudo_ratio_vs_baseline"] = out["pseudo_mean"] / out["baseline_pseudo_mean"].replace(0, np.nan)
    out["pseudo_norm_delta_vs_baseline"] = out["pseudo_norm_mean"] - out["baseline_pseudo_norm_mean"]
    return out


def plot_bar(df: pd.DataFrame, x: str, y: str, color: str, title: str, barmode: str = "group") -> go.Figure:
    fig = px.bar(
        df,
        x=x,
        y=y,
        color=color,
        barmode=barmode,
        title=title,
        color_discrete_map=SCENARIO_COLORS,
        color_discrete_sequence=BLUE_SCALE,
    )
    fig.update_layout(
        template="plotly_white",
        title_font_color="#073B4C",
        legend_title_text="",
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def frame_id_from_npz(path: Path) -> int | None:
    match = re.search(r"(\d+)", path.stem)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def metadata_by_frame(scenario_dir: Path) -> dict[int, dict]:
    csv_path = scenario_dir / "frame_metadata.csv"
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    if "frame_id" not in df.columns:
        return {}
    return {int(row["frame_id"]): row.to_dict() for _, row in df.iterrows()}


def transform_points_to_world(xyz: np.ndarray, meta: dict | None) -> tuple[np.ndarray, np.ndarray]:
    x = xyz[:, 0].astype(float)
    y = xyz[:, 1].astype(float)
    if not meta:
        return x, y
    try:
        ego_x = float(meta.get("ego_x", 0.0))
        ego_y = float(meta.get("ego_y", 0.0))
        yaw_deg = float(meta.get("ego_yaw", 0.0))
    except (TypeError, ValueError):
        return x, y
    yaw = math.radians(yaw_deg)
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)
    world_x = ego_x + x * cos_y - y * sin_y
    world_y = ego_y + x * sin_y + y * cos_y
    return world_x, world_y


def load_trajectory_point_cloud(
    scene_dir: Path,
    scenario: str,
    max_frames: int,
    max_points_per_frame: int,
    mode: str,
) -> pd.DataFrame:
    scenario_dir = scene_dir / scenario
    label_dir = scenario_dir / "lidar_labels"
    raw_dir = scenario_dir / "lidar_raw"
    files = sorted(label_dir.glob("*.npz"))
    if max_frames > 0:
        files = files[:max_frames]
    meta_map = metadata_by_frame(scenario_dir)
    rows = []
    rng = np.random.default_rng(42)
    for file_path in files:
        frame_id = frame_id_from_npz(file_path)
        label = np.load(file_path, allow_pickle=True)
        raw_path = raw_dir / file_path.name
        if not raw_path.exists():
            continue
        raw = np.load(raw_path, allow_pickle=True)
        if "xyz" not in raw:
            continue
        xyz = raw["xyz"]
        n = len(xyz)
        if n == 0:
            continue
        take_n = min(max_points_per_frame, n) if max_points_per_frame > 0 else n
        idx = np.arange(n)
        if take_n < n:
            idx = rng.choice(idx, size=take_n, replace=False)
        world_x, world_y = transform_points_to_world(xyz[idx], meta_map.get(frame_id) if frame_id is not None else None)
        pseudo = label["pseudo_final"][idx].astype(float) if "pseudo_final" in label else np.zeros(len(idx))
        pseudo_norm = label["pseudo_norm"][idx].astype(float) if "pseudo_norm" in label else np.zeros(len(idx))
        material = label["material_label"][idx].astype(str) if "material_label" in label else np.array(["unknown"] * len(idx))
        intensity = raw["intensity"][idx].astype(float) if "intensity" in raw else np.zeros(len(idx))
        if mode == "material":
            value = material
        elif mode == "raw_intensity":
            value = intensity
        elif mode == "pseudo_norm":
            value = pseudo_norm
        else:
            value = pseudo
        rows.append(
            pd.DataFrame(
                {
                    "x": world_x,
                    "y": world_y,
                    "frame_id": frame_id if frame_id is not None else -1,
                    "material": material,
                    "pseudo": pseudo,
                    "pseudo_norm": pseudo_norm,
                    "raw_intensity": intensity,
                    "value": value,
                }
            )
        )
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def load_frame_cloud(scene_dir: Path, scenario: str, file_name: str) -> pd.DataFrame:
    scenario_dir = scene_dir / scenario
    label_path = scenario_dir / "lidar_labels" / file_name
    raw_path = scenario_dir / "lidar_raw" / file_name
    if not label_path.exists() or not raw_path.exists():
        return pd.DataFrame()
    label = np.load(label_path, allow_pickle=True)
    raw = np.load(raw_path, allow_pickle=True)
    if "xyz" not in raw:
        return pd.DataFrame()
    xyz = raw["xyz"]
    n = len(xyz)
    return pd.DataFrame(
        {
            "x": xyz[:, 0],
            "y": xyz[:, 1],
            "z": xyz[:, 2],
            "material": label["material_label"].astype(str) if "material_label" in label else np.array(["unknown"] * n),
            "pseudo": label["pseudo_final"].astype(float) if "pseudo_final" in label else np.zeros(n),
            "pseudo_norm": label["pseudo_norm"].astype(float) if "pseudo_norm" in label else np.zeros(n),
            "raw_intensity": raw["intensity"].astype(float) if "intensity" in raw else np.zeros(n),
        }
    )


def point_cloud_figure(
    df: pd.DataFrame,
    color_mode: str,
    title: str,
    display_mode: str = "points",
    marker_size: int = 2,
    marker_opacity: float = 0.75,
    bins: int = 220,
) -> go.Figure:
    numeric_modes = {"pseudo": "pseudo", "pseudo_norm": "pseudo_norm", "raw_intensity": "raw_intensity"}
    if display_mode == "binned_mean" and color_mode in numeric_modes:
        value_col = numeric_modes[color_mode]
        fig = px.density_heatmap(
            df,
            x="x",
            y="y",
            z=value_col,
            histfunc="avg",
            nbinsx=bins,
            nbinsy=bins,
            title=f"{title} - binned mean",
            color_continuous_scale=PSEUDO_SCALE,
        )
        fig.update_traces(hovertemplate="x=%{x}<br>y=%{y}<br>mean=%{z}<extra></extra>")
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        fig.update_layout(
            template="plotly_white",
            title_font_color="#073B4C",
            margin=dict(l=20, r=20, t=60, b=20),
            height=620,
        )
        return fig

    if color_mode == "material":
        fig = px.scatter(
            df,
            x="x",
            y="y",
            color="material",
            render_mode="webgl",
            title=title,
            hover_data=["frame_id", "pseudo", "raw_intensity"],
            color_discrete_map=MATERIAL_COLORS,
            color_discrete_sequence=["#4A4A4A", "#FFD166", "#FF8811", "#06D6A0", "#EF476F", "#118AB2", "#B0B7C3"],
        )
    else:
        value_col = numeric_modes[color_mode]
        fig = px.scatter(
            df,
            x="x",
            y="y",
            color=value_col,
            render_mode="webgl",
            title=title,
            hover_data=["frame_id", "material"],
            color_continuous_scale=PSEUDO_SCALE,
        )
    fig.update_traces(marker=dict(size=marker_size, opacity=marker_opacity))
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_layout(template="plotly_white", title_font_color="#073B4C", margin=dict(l=20, r=20, t=60, b=20), height=620)
    return fig


def load_lidar_animation_frames(
    scene_dir: Path,
    scenario: str,
    max_frames: int,
    max_points_per_frame: int,
) -> list[pd.DataFrame]:
    scenario_dir = scene_dir / scenario
    label_files = sorted((scenario_dir / "lidar_labels").glob("*.npz"))
    if max_frames > 0:
        label_files = label_files[:max_frames]
    raw_dir = scenario_dir / "lidar_raw"
    rng = np.random.default_rng(7)
    frames: list[pd.DataFrame] = []
    for label_path in label_files:
        raw_path = raw_dir / label_path.name
        if not raw_path.exists():
            continue
        label = np.load(label_path, allow_pickle=True)
        raw = np.load(raw_path, allow_pickle=True)
        if "xyz" not in raw:
            continue
        xyz = raw["xyz"]
        if len(xyz) == 0:
            continue
        mask = (
            (xyz[:, 0] > -12.0) &
            (xyz[:, 0] < 45.0) &
            (np.abs(xyz[:, 1]) < 28.0) &
            (xyz[:, 2] > -3.0) &
            (xyz[:, 2] < 5.0)
        )
        idx = np.where(mask)[0]
        if idx.size == 0:
            continue
        take_n = min(max_points_per_frame, idx.size) if max_points_per_frame > 0 else idx.size
        if take_n < idx.size:
            idx = rng.choice(idx, size=take_n, replace=False)
        frame_id = frame_id_from_npz(label_path) or -1
        materials = label["material_label"].astype(str)[idx] if "material_label" in label else np.array(["unknown"] * len(idx))
        raw_intensity = raw["intensity"][idx].astype(float) if "intensity" in raw else np.zeros(len(idx))
        if raw_intensity.size:
            lo, hi = np.percentile(raw_intensity, [2, 98])
            raw_display = np.clip((raw_intensity - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        else:
            raw_display = raw_intensity
        frames.append(
            pd.DataFrame(
                {
                    "plot_x": xyz[idx, 1],
                    "plot_y": xyz[idx, 0],
                    "material": materials,
                    "material_color": [MATERIAL_COLORS.get(str(m), MATERIAL_COLORS["unknown"]) for m in materials],
                    "raw_intensity": raw_intensity,
                    "raw_display": raw_display,
                    "pseudo_norm": label["pseudo_norm"][idx].astype(float) if "pseudo_norm" in label else np.zeros(len(idx)),
                    "frame_id": frame_id,
                }
            )
        )
    return frames


def lidar_animation_figure(frames: list[pd.DataFrame], scenario: str, point_size: int, point_opacity: float) -> go.Figure:
    if not frames:
        return go.Figure()
    combined = pd.concat(frames, ignore_index=True)
    x_range = [float(combined["plot_x"].min()) - 1.0, float(combined["plot_x"].max()) + 1.0]
    y_range = [float(combined["plot_y"].min()) - 1.0, float(combined["plot_y"].max()) + 1.0]

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Materials", "CARLA raw", "MatSense pseudo"),
        horizontal_spacing=0.035,
    )

    def traces_for(frame_df: pd.DataFrame) -> list[go.Scattergl]:
        material_mask = frame_df["material"] != "unknown"
        material_df = frame_df[material_mask]
        return [
            go.Scattergl(
                x=material_df["plot_x"],
                y=material_df["plot_y"],
                mode="markers",
                marker=dict(size=point_size, opacity=point_opacity, color=material_df["material_color"]),
                text=material_df["material"],
                hovertemplate="material=%{text}<br>right=%{x:.2f}<br>forward=%{y:.2f}<extra></extra>",
                showlegend=False,
            ),
            go.Scattergl(
                x=frame_df["plot_x"],
                y=frame_df["plot_y"],
                mode="markers",
                marker=dict(
                    size=point_size,
                    opacity=point_opacity,
                    color=frame_df["raw_display"],
                    colorscale=[[0, "#000000"], [0.5, "#00DCFF"], [1, "#FFFFFF"]],
                    cmin=0,
                    cmax=1,
                    showscale=True,
                    colorbar=dict(title="", x=0.64, len=0.60, thickness=12),
                ),
                customdata=frame_df["raw_intensity"],
                hovertemplate="raw=%{customdata:.3f}<br>right=%{x:.2f}<br>forward=%{y:.2f}<extra></extra>",
                showlegend=False,
            ),
            go.Scattergl(
                x=frame_df["plot_x"],
                y=frame_df["plot_y"],
                mode="markers",
                marker=dict(
                    size=point_size,
                    opacity=point_opacity,
                    color=frame_df["pseudo_norm"],
                    colorscale=PSEUDO_SCALE,
                    cmin=0,
                    cmax=1,
                    showscale=True,
                    colorbar=dict(title="", x=1.0, len=0.60, thickness=12),
                ),
                hovertemplate="pseudo=%{marker.color:.3f}<br>right=%{x:.2f}<br>forward=%{y:.2f}<extra></extra>",
                showlegend=False,
            ),
        ]

    initial = traces_for(frames[0])
    for col, trace in enumerate(initial, start=1):
        fig.add_trace(trace, row=1, col=col)

    animation_frames = []
    for idx, frame_df in enumerate(frames):
        frame_id = int(frame_df["frame_id"].iloc[0]) if not frame_df.empty else idx
        animation_frames.append(
            go.Frame(
                name=str(idx),
                data=traces_for(frame_df),
                traces=[0, 1, 2],
                layout=go.Layout(),
            )
        )
    fig.frames = animation_frames

    steps = [
        {
            "method": "animate",
            "label": "",
            "args": [[str(i)], {"mode": "immediate", "frame": {"duration": 350, "redraw": True}, "transition": {"duration": 0}}],
        }
        for i in range(len(frames))
    ]
    fig.update_layout(
        template="plotly_white",
        title=f"{scenario}: LiDAR comparison",
        title_font_color="#073B4C",
        height=690,
        margin=dict(l=20, r=20, t=58, b=86),
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.02,
                "y": -0.10,
                "xanchor": "left",
                "yanchor": "top",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 450, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}}],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "x": 0.16,
                "y": -0.10,
                "len": 0.78,
                "steps": steps,
                "currentvalue": {"prefix": ""},
                "pad": {"t": 18, "b": 0},
            }
        ],
    )
    for col in range(1, 4):
        fig.update_xaxes(range=x_range, scaleanchor=f"y{col}" if col > 1 else "y", scaleratio=1, title_text="", row=1, col=col)
        fig.update_yaxes(range=y_range, title_text="", row=1, col=col)
    fig.add_annotation(
        text="right axis = lateral position, vertical axis = forward distance",
        xref="paper",
        yref="paper",
        x=0.5,
        y=-0.08,
        showarrow=False,
        font=dict(size=12, color="#5f7284"),
    )
    return fig


def toolkit_env(toolkit_dir: Path) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(toolkit_dir / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return env


# Runs in the same interpreter the viewer will use, so "import carla works"
# here means it works for the viewer too.
CARLA_PROBE = """
import json, sys
out = {}
try:
    import carla
except Exception as exc:
    out["import_error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(out)); sys.exit(0)
out["module"] = getattr(carla, "__file__", "") or ""
try:
    client = carla.Client(sys.argv[1], int(sys.argv[2]))
    client.set_timeout(5.0)
    out["client_version"] = str(client.get_client_version())
    out["server_version"] = str(client.get_server_version())
    out["map"] = client.get_world().get_map().name.split("/")[-1]
except Exception as exc:
    out["server_error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(out))
"""


def check_carla(toolkit_dir: Path, host: str, port: int) -> dict:
    result = {"host": host, "port": int(port), "port_open": False}
    try:
        with socket.create_connection((host, int(port)), timeout=2.0):
            result["port_open"] = True
    except OSError as exc:
        result["port_error"] = str(exc)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", CARLA_PROBE, host, str(port)],
            capture_output=True,
            text=True,
            timeout=25,
            env=toolkit_env(toolkit_dir),
        )
        lines = [line for line in proc.stdout.splitlines() if line.startswith("{")]
        if lines:
            result.update(json.loads(lines[-1]))
        else:
            result["import_error"] = (proc.stderr or "no output from the probe").strip()[-500:]
    except subprocess.TimeoutExpired:
        result["server_error"] = "CARLA did not answer within 25 s."
    if not result["port_open"]:
        result.pop("server_error", None)
    return result


def preflight_blocker(preflight: dict, allow_version_mismatch: bool) -> str:
    if "import_error" in preflight:
        return (
            f"`import carla` fails in {sys.executable}. Start Streamlit from the Python environment "
            "where the CARLA wheel/egg is installed."
        )
    if not preflight["port_open"]:
        return f"no CARLA server on {preflight['host']}:{preflight['port']}. Start CARLA first."
    if "server_error" in preflight:
        return f"CARLA is listening but did not answer: {preflight['server_error']}"
    client, server = preflight.get("client_version"), preflight.get("server_version")
    if client and server and client != server and not allow_version_mismatch:
        return (
            f"CARLA client {client} and server {server} differ; the viewer refuses to run on mismatched builds. "
            "Use matching versions, or tick 'Allow client/server version mismatch' under CARLA connection."
        )
    return ""


def render_preflight(preflight: dict, allow_version_mismatch: bool) -> None:
    if "import_error" in preflight:
        st.error(f"Python API: `import carla` fails in {sys.executable}")
        st.code(preflight["import_error"], language="text")
    else:
        st.write(f"Python API: OK ({preflight.get('client_version', '?')})")
    if not preflight["port_open"]:
        st.error(f"Server: nothing listening on {preflight['host']}:{preflight['port']}")
    elif "server_error" in preflight:
        st.error(f"Server: {preflight['server_error']}")
    elif "server_version" in preflight:
        st.write(f"Server: {preflight['server_version']}, map **{preflight.get('map', '?')}**")
    else:
        st.write(f"Server: port {preflight['port']} open")
    problem = preflight_blocker(preflight, allow_version_mismatch)
    if problem:
        st.warning(problem)
    else:
        st.success("Ready to start.")


def viewer_pid() -> int | None:
    if not PID_FILE.exists():
        return None
    try:
        pid = int(PID_FILE.read_text(encoding="utf-8").strip())
    except ValueError:
        return None
    if not process_alive(pid):
        PID_FILE.unlink(missing_ok=True)
        return None
    return pid


def process_alive(pid: int) -> bool:
    if os.name == "nt":
        # os.kill(pid, 0) would terminate the process on Windows.
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        exit_code = ctypes.c_ulong()
        kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
        kernel32.CloseHandle(handle)
        return exit_code.value == 259  # STILL_ACTIVE
    try:
        # The viewer is a child of this process: reap it if it has exited,
        # otherwise it lingers as a zombie and still answers signal 0.
        if os.waitpid(pid, os.WNOHANG)[0] == pid:
            return False
    except ChildProcessError:
        pass
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def stop_viewer() -> str:
    if not PID_FILE.exists():
        return "No PID file found."
    try:
        pid = int(PID_FILE.read_text(encoding="utf-8").strip())
    except ValueError:
        PID_FILE.unlink(missing_ok=True)
        return "Invalid PID file removed."
    try:
        os.kill(pid, signal.SIGTERM)
    except Exception as exc:
        PID_FILE.unlink(missing_ok=True)
        return f"Could not stop process {pid}: {exc}"
    PID_FILE.unlink(missing_ok=True)
    return f"Stop requested for process {pid}."


def build_viewer_command(toolkit_dir: Path, config: dict, values: dict) -> list[str]:
    script = toolkit_dir / "scripts" / "carla_pygame_lidar_dataset_recorder_friendly.py"
    material_config = toolkit_dir / "configs" / "material_aware_tool_config.json"
    cmd = [
        sys.executable,
        str(script),
        "--host", values["host"],
        "--port", str(values["port"]),
        "--tm-port", str(values["tm_port"]),
        "--width", str(values["width"]),
        "--height", str(values["height"]),
        "--fps", str(values["fps"]),
        "--weather", values["weather"],
        "--mode", values["mode"],
        "--display-normalization", values["display_normalization"],
        "--display-percentile", str(values["display_percentile"]),
        "--dataset-root", values["dataset_root"],
        "--scene-id", values["scene_id"],
        "--scenario-name", values["scenario_name"],
        "--material-config", str(material_config),
        "--traj-step", str(values["traj_step"]),
        "--utm-offset-x", str(values["utm_offset_x"]),
        "--utm-offset-y", str(values["utm_offset_y"]),
        "--traj-z-offset", str(values["traj_z_offset"]),
        "--follow-mode", values["follow_mode"],
        "--control-smoothing", str(values["control_smoothing"]),
        "--parked-z-offset", str(values["parked_z_offset"]),
        "--parked-limit", str(values["parked_limit"]),
        "--seed", str(values["seed"]),
        "--max-save-frames", str(values["max_save_frames"]),
        "--save-every", str(values["save_every"]),
        "--save-start-delay-seconds", str(values["save_start_delay_seconds"]),
        "--strict-sync-timeout", str(values["strict_sync_timeout"]),
    ]
    for flag, key in (
        ("--channels", "channels"),
        ("--pps", "pps"),
        ("--lidar-range", "lidar_range"),
        ("--upper-fov", "upper_fov"),
        ("--lower-fov", "lower_fov"),
    ):
        if key in values:
            cmd.extend([flag, str(values[key])])
    if "pps" in values:
        # The viewer sets the sensor's rotation to the FPS; record the same value.
        cmd.extend(["--rotation-frequency", str(float(values["fps"]))])
    if values.get("use_base_nominal"):
        cmd.append("--use-base-nominal")
    if values.get("allow_version_mismatch"):
        cmd.append("--allow-version-mismatch")
    if values["profile_name"]:
        cmd.extend(["--profile-name", values["profile_name"]])
    if values["traj_txt"]:
        cmd.extend(["--traj-txt", values["traj_txt"]])
    if values["traj_json"]:
        cmd.extend(["--traj-json", values["traj_json"]])
    if values["parked_json"]:
        cmd.extend(["--parked-json", values["parked_json"]])
    if values["autopilot"]:
        cmd.append("--autopilot")
    if values["save_dataset"]:
        cmd.append("--save-dataset")
    return cmd


def path_browser(label: str, default: str, key: str, kind: str, suffixes: tuple[str, ...] = ()) -> str:
    input_key = f"{key}_input"
    pending_value_key = f"{key}_pending_value"

    if pending_value_key in st.session_state:
        pending_value = st.session_state.pop(pending_value_key)
        st.session_state[input_key] = pending_value
    elif input_key not in st.session_state:
        st.session_state[input_key] = default

    def choose_path() -> str:
        try:
            import tkinter as tk
            from tkinter import filedialog
        except ImportError:
            st.warning("The file dialog needs tkinter (on Ubuntu: `sudo apt install python3-tk`). Paste the path instead.")
            return ""

        current = Path(st.session_state.get(input_key, "") or default or APP_DIR)
        if kind == "directory":
            initial_dir = current if current.exists() and current.is_dir() else APP_DIR
        else:
            initial_dir = current.parent if current.exists() else APP_DIR

        try:
            root = tk.Tk()
        except tk.TclError:
            st.warning("No display available for the file dialog (headless or remote session). Paste the path instead.")
            return ""
        root.withdraw()
        root.attributes("-topmost", True)
        try:
            if kind == "directory":
                selected = filedialog.askdirectory(title=f"Choose {label}", initialdir=str(initial_dir), parent=root)
            else:
                filetypes = [(f"{ext.upper()} files", f"*{ext}") for ext in suffixes] if suffixes else [("All files", "*.*")]
                filetypes.append(("All files", "*.*"))
                selected = filedialog.askopenfilename(
                    title=f"Choose {label}",
                    initialdir=str(initial_dir),
                    filetypes=filetypes,
                    parent=root,
                )
        finally:
            root.destroy()
        return selected

    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    c1.text_input(label, key=input_key)
    if c2.button("Browse", key=f"{key}_browse"):
        selected = choose_path()
        if selected:
            st.session_state[pending_value_key] = selected
            st.rerun()

    return str(st.session_state[input_key])


def app_relative_path(path_text: str) -> str:
    path = Path(path_text)
    if path.is_absolute():
        return str(path)
    return str(APP_DIR / path)


def sidebar_config() -> tuple[Path | None, dict]:
    st.sidebar.title("MatSense")
    toolkit_raw = st.sidebar.text_input("Toolkit path", value=DEFAULT_TOOLKIT_TEXT)
    toolkit_dir = Path(toolkit_raw) if toolkit_raw.strip() else None
    config = load_launcher_config(toolkit_dir)
    if not toolkit_raw.strip():
        st.sidebar.info("Dataset Analyzer uses bundled sample data. Set a toolkit path to enable Run Viewer.")
    elif toolkit_dir == BUNDLED_TOOLKIT:
        st.sidebar.caption("Using the bundled minimal MatSense runtime toolkit.")
    elif toolkit_dir is not None and not toolkit_dir.exists():
        st.sidebar.error("Toolkit path does not exist. Run Viewer is disabled.")
    return toolkit_dir, config


def dataset_analyzer(toolkit_dir: Path | None, config: dict) -> None:
    st.header("Dataset Analyzer")
    st.caption("Operational inspection of MatSense runs: trajectory-level point clouds, material response, and frame checks.")
    if SAMPLE_DATASET_ROOT.exists():
        default_root = SAMPLE_DATASET_ROOT
    elif toolkit_dir is not None and toolkit_dir.exists():
        default_root = toolkit_dir / "scripts" / str(config.get("dataset_root", "output_dataset"))
    else:
        default_root = Path(str(config.get("dataset_root", "sample_data")))
    dataset_root = Path(st.text_input("Dataset root", value=str(default_root)))
    scenes = list_scenes(dataset_root)
    if not scenes:
        st.warning("No scenes found in dataset root.")
        return
    scene = st.selectbox("Scene", scenes, index=scenes.index(config.get("scene_id")) if config.get("scene_id") in scenes else 0)
    scene_dir = dataset_root / scene
    scenarios = list_scenarios(scene_dir)
    selected = st.multiselect("Scenarios", scenarios, default=scenarios)
    if not selected:
        st.info("Select at least one scenario.")
        return

    summary = summarize_frame_metadata(scene_dir, selected)
    if not summary.empty:
        st.subheader("Run Overview")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total frames", int(summary["frames"].sum()))
        c2.metric("Mean LiDAR points", f"{summary['lidar_points_mean'].mean():,.0f}")
        c3.metric("Mean projection", f"{summary['projection_ratio_mean'].mean():.1%}")
        c4.metric("Mean known materials", f"{summary['known_material_ratio_mean'].mean():.1%}")

    tab_cloud, tab_animation, tab_response, tab_frame, tab_advanced = st.tabs(
        ["Trajectory Point Cloud", "Animated LiDAR Comparison", "Material Response", "Frame Inspector", "Advanced Metrics"]
    )

    with tab_cloud:
        st.subheader("Trajectory Point Cloud")
        c1, c2, c3, c4 = st.columns(4)
        cloud_scenario = c1.selectbox("Scenario", selected, index=selected.index("nominal") if "nominal" in selected else 0, key="cloud_scenario")
        available_cloud_frames = len(list((scene_dir / cloud_scenario / "lidar_labels").glob("*.npz")))
        default_cloud_frames = max(1, min(20, available_cloud_frames)) if available_cloud_frames else 1
        cloud_color = c2.selectbox(
            "Color by",
            ["pseudo", "material", "raw_intensity", "pseudo_norm"],
            format_func=lambda v: {
                "pseudo": "Pseudo-reflectance",
                "pseudo_norm": "Pseudo normalized",
                "raw_intensity": "Raw CARLA intensity",
                "material": "Material class",
            }[v],
            key="cloud_color",
        )
        cloud_frames = c3.number_input(
            f"Frames to load ({available_cloud_frames} available)",
            min_value=1,
            value=default_cloud_frames,
            step=5,
            key="cloud_frames",
        )
        cloud_points = c4.number_input("Points/frame", min_value=100, value=2500, step=500, key="cloud_points")
        c1, c2, c3, c4 = st.columns(4)
        cloud_display = c1.selectbox(
            "Display",
            ["binned_mean", "points"],
            format_func=lambda v: "Binned mean" if v == "binned_mean" else "Points",
            key="cloud_display",
            help="Binned mean is clearer when many points overlap. Material mode always uses points.",
        )
        cloud_bins = c2.slider("Bins", min_value=60, max_value=420, value=220, step=20, key="cloud_bins")
        cloud_marker_size = c3.slider("Point size", min_value=1, max_value=6, value=2, step=1, key="cloud_marker_size")
        cloud_opacity = c4.slider("Point opacity", min_value=0.10, max_value=1.00, value=0.55, step=0.05, key="cloud_opacity")
        with st.spinner("Building aggregated point cloud..."):
            cloud_df = load_trajectory_point_cloud(scene_dir, cloud_scenario, int(cloud_frames), int(cloud_points), cloud_color)
        if cloud_df.empty:
            st.warning("No point cloud data found for this scenario.")
        else:
            used_frames = cloud_df["frame_id"].nunique() if "frame_id" in cloud_df else 0
            m1, m2, m3 = st.columns(3)
            m1.metric("Available frames", available_cloud_frames)
            m2.metric("Loaded frames", used_frames)
            m3.metric("Rendered points", f"{len(cloud_df):,}")
            if available_cloud_frames and int(cloud_frames) >= available_cloud_frames:
                st.info(
                    f"This scenario has only {available_cloud_frames} saved LiDAR frames. "
                    "Increasing Frames to load above that will not change the plot."
                )
            effective_display = "points" if cloud_color == "material" else cloud_display
            st.plotly_chart(
                point_cloud_figure(
                    cloud_df,
                    cloud_color,
                    f"{cloud_scenario}: aggregated LiDAR point cloud",
                    display_mode=effective_display,
                    marker_size=int(cloud_marker_size),
                    marker_opacity=float(cloud_opacity),
                    bins=int(cloud_bins),
                ),
                use_container_width=True,
            )
            st.caption(
                "Use Binned mean for pseudo/intensity views when scatter points overlap. Material mode uses point rendering."
            )

    with tab_animation:
        st.subheader("Animated LiDAR Comparison")
        st.caption("Synchronized frame-by-frame comparison of material labels, CARLA intensity, and MatSense pseudo-reflectance.")
        c1, c2, c3, c4 = st.columns(4)
        anim_scenario = c1.selectbox(
            "Scenario",
            selected,
            index=selected.index("nominal") if "nominal" in selected else 0,
            key="anim_scenario",
        )
        available_anim_frames = len(list((scene_dir / anim_scenario / "lidar_labels").glob("*.npz")))
        default_anim_frames = max(1, min(20, available_anim_frames)) if available_anim_frames else 1
        anim_frames = c2.number_input(
            f"Frames ({available_anim_frames} available)",
            min_value=1,
            value=default_anim_frames,
            step=5,
            key="anim_frames",
        )
        anim_points = c3.number_input("Points/frame", min_value=250, value=4500, step=500, key="anim_points")
        anim_point_size = c4.slider("Point size", min_value=1, max_value=6, value=2, step=1, key="anim_point_size")
        anim_opacity = st.slider("Point opacity", min_value=0.10, max_value=1.00, value=0.70, step=0.05, key="anim_opacity")
        with st.spinner("Building animated LiDAR comparison..."):
            anim_data = load_lidar_animation_frames(scene_dir, anim_scenario, int(anim_frames), int(anim_points))
        if not anim_data:
            st.warning("No LiDAR frames found for the animation.")
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Available frames", available_anim_frames)
            m2.metric("Animated frames", len(anim_data))
            m3.metric("Max points/frame", f"{int(anim_points):,}")
            st.plotly_chart(
                lidar_animation_figure(anim_data, anim_scenario, int(anim_point_size), float(anim_opacity)),
                use_container_width=True,
            )
            st.caption(
                "Use Play or the frame slider to inspect the same LiDAR scan across the three views. "
                "CARLA raw intensity is contrast-stretched per frame for display; hover values remain raw."
            )

    with tab_response:
        st.subheader("Material and Weather Response")
        c1, c2 = st.columns(2)
        max_files = c1.number_input("Max label files/scenario (0 = all)", min_value=0, value=0, step=10, key="response_max_files")
        baseline = c2.selectbox("Baseline scenario", selected, index=selected.index("nominal") if "nominal" in selected else 0, key="response_baseline")

        with st.spinner("Aggregating lidar_labels/*.npz by material..."):
            material_df = aggregate_lidar_labels(scene_dir, selected, int(max_files))
        if material_df.empty:
            st.warning("No LiDAR label data found.")
            return
        material_df = add_ratio_to_baseline(material_df, baseline)

        materials = sorted(material_df["material"].unique().tolist())
        default_materials = [m for m in ["asphalt", "building", "car", "sidewalk", "vegetation"] if m in materials] or materials
        visible_materials = st.multiselect("Materials", materials, default=default_materials)
        filtered = material_df[material_df["material"].isin(visible_materials)].copy()

        st.plotly_chart(
            plot_bar(filtered, "material", "pseudo_mean", "scenario", "Pseudo-reflectance by material and weather"),
            use_container_width=True,
        )

        compare = filtered.melt(
            id_vars=["scenario", "material"],
            value_vars=["carla_intensity_norm_mean", "pseudo_norm_mean"],
            var_name="signal",
            value_name="value",
        )
        compare["signal"] = compare["signal"].replace(
            {
                "carla_intensity_norm_mean": "CARLA default intensity",
                "pseudo_norm_mean": "MatSense pseudo-reflectance",
            }
        )
        fig_compare = px.bar(
            compare,
            x="material",
            y="value",
            color="signal",
            facet_col="scenario",
            barmode="group",
            title="Default CARLA LiDAR intensity vs MatSense pseudo-reflectance",
            color_discrete_sequence=["#9DD9D2", "#073B4C"],
        )
        fig_compare.update_layout(template="plotly_white", title_font_color="#073B4C", legend_title_text="")
        st.plotly_chart(fig_compare, use_container_width=True)

        delta_df = filtered[filtered["scenario"] != baseline].copy()
        if not delta_df.empty and "pseudo_ratio_vs_baseline" in delta_df:
            fig_delta = plot_bar(
                delta_df,
                "material",
                "pseudo_ratio_vs_baseline",
                "scenario",
                f"Pseudo-reflectance ratio vs {baseline}",
            )
            fig_delta.add_hline(y=1, line_color="#073B4C", line_width=1)
            st.plotly_chart(fig_delta, use_container_width=True)

    with tab_frame:
        st.subheader("Frame Inspector")
        c1, c2, c3 = st.columns(3)
        frame_scenario = c1.selectbox("Scenario", selected, index=selected.index("nominal") if "nominal" in selected else 0, key="frame_scenario")
        frame_files = sorted((scene_dir / frame_scenario / "lidar_labels").glob("*.npz"))
        if not frame_files:
            st.warning("No frame files found.")
        else:
            frame_name = c2.selectbox("Frame", [p.name for p in frame_files], key="frame_name")
            frame_color = c3.selectbox(
                "Color by",
                ["pseudo", "material", "raw_intensity", "pseudo_norm"],
                format_func=lambda v: {
                    "pseudo": "Pseudo-reflectance",
                    "pseudo_norm": "Pseudo normalized",
                    "raw_intensity": "Raw CARLA intensity",
                    "material": "Material class",
                }[v],
                key="frame_color",
            )
            c1, c2, c3 = st.columns(3)
            frame_display = c1.selectbox(
                "Display",
                ["points", "binned_mean"],
                format_func=lambda v: "Points" if v == "points" else "Binned mean",
                key="frame_display",
            )
            frame_marker_size = c2.slider("Point size", min_value=1, max_value=8, value=2, step=1, key="frame_marker_size")
            frame_opacity = c3.slider("Point opacity", min_value=0.10, max_value=1.00, value=0.60, step=0.05, key="frame_opacity")
            frame_df = load_frame_cloud(scene_dir, frame_scenario, frame_name)
            if frame_df.empty:
                st.warning("Could not load this frame.")
            else:
                frame_df["frame_id"] = frame_id_from_npz(Path(frame_name)) or -1
                effective_frame_display = "points" if frame_color == "material" else frame_display
                st.plotly_chart(
                    point_cloud_figure(
                        frame_df,
                        frame_color,
                        f"{frame_scenario} / {frame_name}: LiDAR BEV",
                        display_mode=effective_frame_display,
                        marker_size=int(frame_marker_size),
                        marker_opacity=float(frame_opacity),
                        bins=180,
                    ),
                    use_container_width=True,
                )
                m1, m2, m3 = st.columns(3)
                m1.metric("Points", f"{len(frame_df):,}")
                m2.metric("Mean pseudo", f"{frame_df['pseudo'].mean():.3f}")
                m3.metric("Known materials", f"{(frame_df['material'] != 'unknown').mean():.1%}")

    with tab_advanced:
        st.subheader("Advanced Metrics")
        if summary.empty:
            st.warning("No frame metadata found.")
            return
        st.dataframe(
            summary.style.format(
                {
                    "duration_s": "{:.2f}",
                    "distance_m": "{:.2f}",
                    "lidar_points_mean": "{:,.0f}",
                    "projected_points_mean": "{:,.0f}",
                    "projection_ratio_mean": "{:.2%}",
                    "known_material_points_mean": "{:,.0f}",
                    "known_material_ratio_mean": "{:.2%}",
                }
            ),
            use_container_width=True,
        )
        coverage = summary.melt(
            id_vars="scenario",
            value_vars=["projection_ratio_mean", "known_material_ratio_mean"],
            var_name="metric",
            value_name="ratio",
        )
        fig = px.bar(
            coverage,
            x="scenario",
            y="ratio",
            color="metric",
            barmode="group",
            title="Projection and known-material coverage",
            color_discrete_sequence=["#118AB2", "#073B4C"],
        )
        fig.update_layout(template="plotly_white", title_font_color="#073B4C", margin=dict(l=20, r=20, t=60, b=20))
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

        adv_max_files = st.number_input("Max label files/scenario for advanced table (0 = all)", min_value=0, value=0, step=10)
        adv_baseline = st.selectbox("Advanced baseline", selected, index=selected.index("nominal") if "nominal" in selected else 0)
        adv_df = add_ratio_to_baseline(aggregate_lidar_labels(scene_dir, selected, int(adv_max_files)), adv_baseline)
        if not adv_df.empty:
            st.dataframe(
                adv_df.sort_values(["scenario", "points"], ascending=[True, False]).style.format(
                    {
                        "share": "{:.2%}",
                        "pseudo_mean": "{:.3f}",
                        "pseudo_std": "{:.3f}",
                        "pseudo_norm_mean": "{:.3f}",
                        "carla_intensity_norm_mean": "{:.3f}",
                        "matsense_minus_carla_norm": "{:+.3f}",
                        "pseudo_delta_vs_baseline": "{:+.3f}",
                        "pseudo_ratio_vs_baseline": "{:.3f}",
                        "range_mean_m": "{:.2f}",
                    }
                ),
                use_container_width=True,
            )
            scatter = px.scatter(
                adv_df,
                x="carla_intensity_norm_mean",
                y="pseudo_norm_mean",
                color="scenario",
                symbol="material",
                size="points",
                hover_data=["material", "pseudo_mean", "range_mean_m", "matsense_minus_carla_norm"],
                title="How MatSense shifts the default CARLA LiDAR response",
                color_discrete_map=SCENARIO_COLORS,
            )
            scatter.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    line=dict(color="#073B4C", dash="dash"),
                    name="unchanged",
                )
            )
            scatter.update_layout(template="plotly_white", title_font_color="#073B4C", margin=dict(l=20, r=20, t=60, b=20))
            st.plotly_chart(scatter, use_container_width=True)


# CARLA 0.9.14+ semantic tags (the CityScapes-aligned set CARLA 0.9.16 uses).
CARLA_SEMANTIC_TAGS = {
    0: "Unlabeled", 1: "Roads", 2: "SideWalks", 3: "Building", 4: "Wall", 5: "Fence", 6: "Pole",
    7: "TrafficLight", 8: "TrafficSign", 9: "Vegetation", 10: "Terrain", 11: "Sky", 12: "Pedestrian",
    13: "Rider", 14: "Car", 15: "Truck", 16: "Bus", 17: "Train", 18: "Motorcycle", 19: "Bicycle",
    20: "Static", 21: "Dynamic", 22: "Other", 23: "Water", 24: "RoadLine", 25: "Ground", 26: "Bridge",
    27: "RailTrack", 28: "GuardRail",
}


def profile_table(profile: dict) -> pd.DataFrame:
    """One row per material: beta, alpha per condition, and the factor the
    operator applies, phi / phi_max with phi_max over the measured classes
    (Eq. 8 of the paper, as matsense_closedloop.build_profile computes it)."""
    nominal_base = profile.get("nominal_base", {})
    weather_ratio = profile.get("weather_ratio", {})
    default_material = profile.get("default_material", "unknown")
    measured = set(profile.get("measured_materials") or nominal_base)
    materials = sorted(set(nominal_base) | set(profile.get("semantic_to_material", {}).values()) | {default_material})
    conditions = list(weather_ratio) or ["nominal"]

    def base(m: str) -> float:
        return float(nominal_base.get(m, nominal_base.get(default_material, float("nan"))))

    def ratio(m: str, w: str) -> float:
        lut = weather_ratio.get(w, {})
        return float(lut.get(m, lut.get(default_material, 1.0)))

    phi_max = {}
    for w in conditions:
        pool = [m for m in materials if m in measured] or materials
        phi_max[w] = max(base(m) * ratio(m, w) for m in pool)

    rows = []
    for m in materials:
        row = {"material": m, "status": "measured" if m in measured else "declared", "beta": base(m)}
        for w in conditions:
            if w != "nominal":
                row[f"alpha_{w}"] = ratio(m, w)
        for w in conditions:
            row[f"factor_{w}"] = base(m) * ratio(m, w) / phi_max[w] if phi_max[w] else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def semantic_mapping_table(profile: dict) -> pd.DataFrame:
    mapping = {int(k): str(v) for k, v in profile.get("semantic_to_material", {}).items()}
    default_material = profile.get("default_material", "unknown")
    rows = []
    for tag, name in CARLA_SEMANTIC_TAGS.items():
        rows.append({
            "tag": tag,
            "CARLA class": name,
            "material": mapping.get(tag, default_material),
            "mapping": "mapped" if tag in mapping else f"fallback ({default_material})",
        })
    for tag in sorted(set(mapping) - set(CARLA_SEMANTIC_TAGS)):
        rows.append({"tag": tag, "CARLA class": "?", "material": mapping[tag], "mapping": "mapped"})
    return pd.DataFrame(rows)


def profile_browser(toolkit_dir: Path | None) -> None:
    st.header("Profile")
    st.caption(
        "The frozen material profile the runtime applies: nominal response per material (beta), "
        "condition ratios (alpha), the factor that reaches CARLA's intensity, and how CARLA's classes map onto materials."
    )
    tool_config = load_tool_config(toolkit_dir)
    profiles = tool_config.get("profiles", {})
    if not profiles:
        st.warning("No profiles found. Check the Toolkit path in the sidebar.")
        return
    names = list(profiles)
    default_name = str(tool_config.get("default_profile") or names[0])
    c1, c2 = st.columns(2)
    name = c1.selectbox("Profile", names, index=names.index(default_name) if default_name in names else 0, key="pf_name")
    compare = c2.selectbox("Compare with", ["(none)"] + [n for n in names if n != name], key="pf_compare")
    profile = profiles[name]

    table = profile_table(profile)
    measured = table[table["status"] == "measured"]["material"].tolist()
    declared = table[table["status"] == "declared"]["material"].tolist()
    m1, m2, m3 = st.columns(3)
    m1.metric("Measured materials", len(measured))
    m2.metric("Declared materials", len(declared))
    m3.metric("Mapped CARLA tags", len(profile.get("semantic_to_material", {})))
    if name == default_name:
        st.caption(f"`{name}` is the tool config's default profile.")
    if not profile.get("notes"):
        st.warning(f"`{name}` has no calibration notes: its coefficients are declared, not measured.")
    elif "measured_materials" not in profile:
        st.info("This profile does not list its measured materials, so every material is shown as measured.")
    if declared:
        st.caption("Declared: " + ", ".join(declared) + ". These are fallback values, not estimates from recordings.")
    for condition, message in UNMEASURED_CONDITIONS.items():
        if f"alpha_{condition}" in table.columns:
            st.caption(message)

    tab_coeff, tab_factor, tab_mapping, tab_notes = st.tabs(
        ["Coefficients", "Applied factor", "Semantic mapping", "Notes"]
    )

    with tab_coeff:
        shown = table.drop(columns=[c for c in table.columns if c.startswith("factor_")])
        if compare != "(none)":
            other = profile_table(profiles[compare]).set_index("material")
            shown = shown.set_index("material")
            for col in [c for c in shown.columns if c == "beta" or c.startswith("alpha_")]:
                if col in other.columns:
                    shown[f"{col} ({compare})"] = other[col]
                    shown[f"delta {col}"] = shown[col] - other[col]
            shown = shown.reset_index()
        st.dataframe(shown.round(4), use_container_width=True, hide_index=True)

        fig = go.Figure()
        fig.add_bar(
            x=table["material"],
            y=table["beta"],
            name=name,
            marker_color=[MATERIAL_COLORS.get(m, "#118AB2") for m in table["material"]],
            marker_pattern_shape=["" if s == "measured" else "/" for s in table["status"]],
            text=table["status"],
        )
        if compare != "(none)":
            other_table = profile_table(profiles[compare])
            fig.add_bar(x=other_table["material"], y=other_table["beta"], name=compare, marker_color="#B0B7C3")
        fig.update_layout(
            template="plotly_white",
            title="Nominal response per material (beta); hatched bars are declared",
            barmode="group",
            yaxis_title="beta",
            margin=dict(l=20, r=20, t=60, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab_factor:
        st.caption(
            "Factor multiplied into CARLA's intensity per return: beta x alpha, divided by its maximum over the "
            "measured materials for that condition. The strongest measured class gets 1.0; nothing measured exceeds it."
        )
        factor_cols = [c for c in table.columns if c.startswith("factor_")]
        long = table.melt(id_vars=["material", "status"], value_vars=factor_cols, var_name="condition", value_name="factor")
        long["condition"] = long["condition"].str.replace("factor_", "", regex=False)
        fig = px.bar(
            long,
            x="material",
            y="factor",
            color="condition",
            barmode="group",
            color_discrete_map=SCENARIO_COLORS,
            hover_data=["status"],
        )
        fig.update_layout(template="plotly_white", legend_title_text="", margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            table[["material", "status"] + factor_cols].round(4), use_container_width=True, hide_index=True
        )

    with tab_mapping:
        mapping = semantic_mapping_table(profile)
        only_mapped = st.checkbox("Only mapped CARLA classes", value=True, key="pf_only_mapped")
        view = mapping[mapping["mapping"] == "mapped"] if only_mapped else mapping
        st.dataframe(view, use_container_width=True, hide_index=True)
        grouped = (
            mapping[mapping["mapping"] == "mapped"].groupby("material")["CARLA class"].apply(lambda v: ", ".join(v)).reset_index()
        )
        grouped["beta"] = grouped["material"].map(dict(zip(table["material"], table["beta"])))
        st.markdown("**Many-to-one: CARLA classes sharing one coefficient**")
        st.dataframe(grouped.round(4), use_container_width=True, hide_index=True)
        st.caption(
            "Classes not listed above fall back to the default material "
            f"`{profile.get('default_material', 'unknown')}`. Runtime logs count these returns as unmapped."
        )

    with tab_notes:
        notes = str(profile.get("notes", "")).strip()
        st.write(notes or "No notes in this profile.")
        with st.expander("Raw profile JSON"):
            st.json(profile)


# Sensing arms of the closed-loop study (paper Section 5.2) and the viewer
# --mode value that selects each one when a PCLA agent drives.
SENSING_ARMS = {
    "standard": "intensity",
    "global": "global",
    "matsense": "material",
    "shuffled": "shuffled",
    "level": "level",
}
ARM_BY_MODE = {mode: arm for arm, mode in SENSING_ARMS.items()}
ARM_HELP = {
    "standard": "CARLA's own response, unchanged",
    "global": "one scalar per frame, at the level MatSense would deliver",
    "matsense": "material- and condition-dependent response from the profile",
    "shuffled": "MatSense factors permuted among points in range bins",
    "level": "a constant scale chosen here, no dropout or jitter",
}
OUTCOME_METRICS = {
    "route_completion_ratio": "Route completion (fraction)",
    "route_completion_m": "Route completion (m)",
    "final_cross_track_error_m": "Final cross-track error (m)",
    "min_ttc_proxy_s": "Minimum TTC proxy (s)",
    "max_deceleration_mps2": "Peak deceleration (m/s²)",
    "first_brake_progress_m": "Progress at first brake (m)",
    "collision_count": "Collisions",
}


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9-]+", "-", text.strip()).strip("-") or "run"


def load_toolkit_module(toolkit_dir: Path | None, name: str):
    if toolkit_dir is None:
        return None
    scripts = str(toolkit_dir / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    try:
        return __import__(name)
    except Exception:  # noqa: BLE001
        return None


def build_campaign_plan(toolkit_dir: Path, campaign_dir: Path, c: dict) -> dict:
    script = toolkit_dir / "scripts" / "carla_pygame_lidar_dataset_recorder_friendly.py"
    material_config = toolkit_dir / "configs" / "material_aware_tool_config.json"
    scenario_label = safe_name(c["scenario_label"])
    runs = []
    for weather in c["conditions"]:
        for arm in c["arms"]:
            for seed in c["seeds"]:
                for rep in range(c["replicates"]):
                    run_id = f"{scenario_label}__{arm}__{weather}__seed{seed}__rep{rep}"
                    cmd = [
                        sys.executable, str(script),
                        "--host", c["host"], "--port", str(c["port"]), "--tm-port", str(c["tm_port"]),
                        "--fps", "20",
                        "--weather", weather,
                        "--mode", SENSING_ARMS[arm],
                        "--material-config", str(material_config),
                        "--profile-name", c["profile"],
                        "--dataset-root", str(campaign_dir),
                        "--scene-id", "runs",
                        "--scenario-name", run_id,
                        "--seed", str(c["scene_seed"]),
                        "--pcla-perturb-seed", str(seed),
                        "--replicate-index", str(rep),
                        "--condition-type", scenario_label,
                        "--pcla-agent", c["agent"],
                        "--pcla-town", c["town"],
                        "--pcla-spawn-index", str(c["spawn_index"]),
                        "--pcla-max-seconds", str(c["max_seconds"]),
                        "--pcla-stopped-seconds", str(c["stopped_seconds"]),
                        "--pcla-route-dev-threshold", str(c["route_dev_threshold"]),
                        "--pcla-spawn-jitter-m", str(c["spawn_jitter_m"]),
                        "--pcla-dropout-gain", str(c["dropout_gain"]),
                        "--pcla-jitter-gain", str(c["jitter_gain"]),
                    ]
                    if c["pcla_dir"]:
                        cmd += ["--pcla-dir", c["pcla_dir"]]
                    if c["route"]:
                        cmd += ["--pcla-route", c["route"]]
                    if c["srunner_scenario"]:
                        cmd += ["--srunner-scenario", c["srunner_scenario"]]
                        for param in c["srunner_params"]:
                            cmd += ["--srunner-param", param]
                    if c["headless"]:
                        cmd += ["--no-draw", "--disable-viewer-cameras"]
                    if c["allow_version_mismatch"]:
                        cmd.append("--allow-version-mismatch")
                    run = {"id": run_id, "arm": arm, "weather": weather, "seed": seed, "replicate": rep, "cmd": cmd}
                    if arm == "level":
                        run["env"] = {"MATSENSE_LEVEL": str(c["level"])}
                    runs.append(run)
    return {
        "name": campaign_dir.name,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "settings": c,
        "cwd": str(toolkit_dir / "scripts"),
        "env": {"PYTHONPATH": toolkit_env(toolkit_dir)["PYTHONPATH"], "MATSENSE_FORCE_EXIT_ON_SHUTDOWN": "1"},
        "runs": runs,
    }


def campaign_status(campaign_dir: Path) -> dict | None:
    try:
        status = read_json(campaign_dir / "status.json")
    except (OSError, ValueError):
        return None
    if not status:
        return None
    if status.get("state") == "running" and not process_alive(int(status.get("pid", -1))):
        status["state"] = "interrupted"
    return status


def list_campaigns(root: Path) -> list[Path]:
    if not root.exists():
        return []
    found = [p.parent for p in root.glob("*/campaign.json")] + [p.parent for p in root.glob("*/runs")]
    return sorted(set(found), key=lambda p: p.stat().st_mtime, reverse=True)


def load_campaign_summaries(campaign_dir: Path, settle_s: float, harness_modes: tuple) -> tuple[pd.DataFrame, int]:
    rows, unsettled = [], 0
    for f in sorted((campaign_dir / "runs").glob("*/run_summary.json")):
        if time.time() - f.stat().st_mtime < settle_s:
            unsettled += 1
            continue
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        d["run_id"] = f.parent.name
        d["arm"] = ARM_BY_MODE.get(str(d.get("mode", "")), str(d.get("mode", "?")))
        d["harness_failure"] = d.get("termination_mode") in harness_modes
        d["completed"] = d.get("termination_mode") == "completed"
        rows.append(d)
    df = pd.DataFrame(rows)
    for col in OUTCOME_METRICS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df, unsettled


def experiment_campaign(toolkit_dir: Path, config: dict) -> None:
    tool_config = load_tool_config(toolkit_dir)
    profile_names = list(tool_config.get("profiles", {})) or [""]
    default_profile = str(tool_config.get("default_profile") or profile_names[0])

    st.caption(
        "A campaign drives the same route and hazard under several sensing arms, conditions and seeds, "
        "with a PCLA agent at the wheel. Runs execute one after another in the background; "
        "you can close the browser and come back."
    )
    c1, c2 = st.columns(2)
    with c1:
        pcla_dir = path_browser("PCLA directory (forked, with the perturb_fn hook)", "", "exp_pcla_dir", "directory")
    with c2:
        route = path_browser("Route XML", "", "exp_route", "file", (".xml",))

    with st.form("campaign"):
        c1, c2, c3 = st.columns(3)
        name = c1.text_input("Campaign name", value=time.strftime("campaign_%Y%m%d_%H%M"))
        agent = c2.text_input("PCLA agent", value="tfv4_l6_0", help="Agent name as PCLA knows it, e.g. tfv4_l6_0.")
        profile = c3.selectbox(
            "Material profile", profile_names,
            index=profile_names.index(default_profile) if default_profile in profile_names else 0,
        )
        c1, c2, c3 = st.columns(3)
        town = c1.text_input("Town", value="Town02")
        spawn_index = c2.number_input("Spawn index", value=31, min_value=0, step=1)
        scenario_label = c3.text_input("Scenario label", value="scenario", help="Short name used in run folders.")
        c1, c2 = st.columns(2)
        srunner_scenario = c1.text_input(
            "ScenarioRunner hazard (optional)",
            placeholder="class name, e.g. ParkingCrossingPedestrian",
            help="Needs SCENARIO_RUNNER_ROOT and CARLA_PYTHONAPI_ROOT. Leave empty for a route without a scripted hazard.",
        )
        srunner_params = c2.text_input("Hazard parameters", placeholder="distance=12, offset=0.6")

        st.markdown("**Design**")
        c1, c2 = st.columns(2)
        arms = c1.multiselect(
            "Sensing arms", list(SENSING_ARMS), default=["standard", "global", "matsense"],
            help="; ".join(f"{a}: {d}" for a, d in ARM_HELP.items()),
        )
        conditions = c2.multiselect("Conditions", WEATHER_OPTIONS, default=["nominal", "rain"])
        c1, c2, c3 = st.columns(3)
        seeds_text = c1.text_input("Perturbation seeds", value="1, 2, 3")
        replicates = c2.number_input("Replicates per cell", value=3, min_value=1, step=1)
        level = c3.number_input("Level (only for the level arm)", value=1.0, min_value=0.0, step=0.05)

        with st.expander("Run limits and operator"):
            c1, c2, c3 = st.columns(3)
            max_seconds = c1.number_input("Max simulated time (s)", value=35.0, min_value=0.0, step=5.0, help="0 = no limit.")
            stopped_seconds = c2.number_input("End after stationary for (s)", value=4.0, min_value=0.0, step=1.0)
            route_dev_threshold = c3.number_input("End above route deviation (m)", value=5.0, min_value=0.0, step=0.5)
            c1, c2, c3 = st.columns(3)
            dropout_gain = c1.number_input("Dropout gain", value=0.4, min_value=0.0, step=0.05)
            jitter_gain = c2.number_input("Range jitter gain", value=0.10, min_value=0.0, step=0.01)
            spawn_jitter_m = c3.number_input(
                "Spawn jitter (m)", value=0.0, min_value=0.0, step=0.01,
                help="Centimetre-scale start-pose jitter per replicate. Without it, runs that apply no perturbation can be identical copies.",
            )
            c1, c2 = st.columns(2)
            scene_seed = c1.number_input("Scene seed", value=int(config.get("seed", 42)), step=1, help="Kept fixed across runs.")
            headless = c2.checkbox("Headless (no pygame drawing, faster)", value=True)

        with st.expander("CARLA connection"):
            c1, c2, c3 = st.columns(3)
            host = c1.text_input("Host", value=str(config.get("host", "127.0.0.1")))
            port = c2.number_input("Port", value=int(config.get("port", 2000)), step=1)
            tm_port = c3.number_input("Traffic Manager port", value=int(config.get("tm_port", 8000)), step=1)
            allow_version_mismatch = st.checkbox("Allow client/server version mismatch", value=False)

        launch = st.form_submit_button("Start campaign", type="primary")

    try:
        seeds = [int(x) for x in re.split(r"[,\s]+", seeds_text.strip()) if x]
    except ValueError:
        seeds = []
        st.error("Seeds must be integers separated by commas.")
    n_runs = len(arms) * len(conditions) * len(seeds) * int(replicates)
    st.info(
        f"{len(arms)} arms × {len(conditions)} conditions × {len(seeds)} seeds × {int(replicates)} replicates "
        f"= **{n_runs} runs**, each capped at {max_seconds:g} s of simulated time."
    )
    for condition in conditions:
        if condition in UNMEASURED_CONDITIONS:
            st.warning(UNMEASURED_CONDITIONS[condition])
    if "nominal" in conditions and set(arms) & {"global", "matsense", "shuffled"}:
        st.caption(
            "Under nominal conditions every alpha is 1, so no return is dropped or moved: "
            "the arms differ only in intensity there."
        )

    if launch:
        problems = []
        if not arms or not conditions or not seeds:
            problems.append("choose at least one arm, one condition and one seed")
        if not agent.strip():
            problems.append("give the PCLA agent name")
        if pcla_dir and not Path(pcla_dir).is_dir():
            problems.append(f"PCLA directory not found: {pcla_dir}")
        if route and not Path(route).is_file():
            problems.append(f"route file not found: {route}")
        campaign_dir = CAMPAIGN_ROOT / (re.sub(r"[^A-Za-z0-9_-]+", "-", name.strip()).strip("-") or "campaign")
        if (campaign_dir / "campaign.json").exists():
            problems.append(f"a campaign named {campaign_dir.name} already exists; resume it below or choose another name")
        if not problems:
            preflight = check_carla(toolkit_dir, host.strip(), int(port))
            blocker = preflight_blocker(preflight, allow_version_mismatch)
            if blocker:
                problems.append(blocker)
        if problems:
            st.error("Campaign not started: " + "; ".join(problems))
        else:
            settings = {
                "agent": agent.strip(), "pcla_dir": pcla_dir.strip(), "route": route.strip(), "town": town.strip(),
                "spawn_index": int(spawn_index), "scenario_label": scenario_label, "profile": profile,
                "srunner_scenario": srunner_scenario.strip(),
                "srunner_params": [x.strip() for x in srunner_params.split(",") if x.strip()],
                "arms": arms, "conditions": conditions, "seeds": seeds, "replicates": int(replicates),
                "level": float(level), "max_seconds": float(max_seconds), "stopped_seconds": float(stopped_seconds),
                "route_dev_threshold": float(route_dev_threshold), "dropout_gain": float(dropout_gain),
                "jitter_gain": float(jitter_gain), "spawn_jitter_m": float(spawn_jitter_m),
                "scene_seed": int(scene_seed), "headless": headless, "host": host.strip(), "port": int(port),
                "tm_port": int(tm_port), "allow_version_mismatch": allow_version_mismatch,
            }
            campaign_dir.mkdir(parents=True, exist_ok=True)
            plan = build_campaign_plan(toolkit_dir, campaign_dir, settings)
            (campaign_dir / "campaign.json").write_text(json.dumps(plan, indent=1), encoding="utf-8")
            start_campaign(toolkit_dir, campaign_dir)
            st.session_state["exp_selected"] = campaign_dir.name
            st.success(f"Campaign {campaign_dir.name} started: {len(plan['runs'])} runs.")

    st.subheader("Campaigns")
    campaigns = [c for c in list_campaigns(CAMPAIGN_ROOT) if (c / "campaign.json").exists()]
    if not campaigns:
        st.caption(f"No campaigns yet under {CAMPAIGN_ROOT}.")
        return
    names = [c.name for c in campaigns]
    selected = st.selectbox(
        "Campaign", names,
        index=names.index(st.session_state["exp_selected"]) if st.session_state.get("exp_selected") in names else 0,
        key="exp_status_pick",
    )
    campaign_dir = CAMPAIGN_ROOT / selected
    status = campaign_status(campaign_dir)
    plan = read_json(campaign_dir / "campaign.json")
    total = len(plan.get("runs", []))
    done = len(list((campaign_dir / "runs").glob("*/run_summary.json")))
    st.progress(min(done / total, 1.0) if total else 0.0, text=f"{done} of {total} runs have a summary")
    state = (status or {}).get("state", "not started")
    c1, c2, c3 = st.columns(3)
    c1.metric("State", state)
    c2.metric("Current run", (status or {}).get("current") or "—")
    c3.metric("Failed runs", len((status or {}).get("failed", [])))
    b1, b2, _ = st.columns([1, 1, 4])
    running = state == "running"
    if b1.button("Resume", disabled=running or done >= total, help="Runs only what has no summary yet."):
        preflight = check_carla(toolkit_dir, plan["settings"]["host"], plan["settings"]["port"])
        blocker = preflight_blocker(preflight, plan["settings"].get("allow_version_mismatch", False))
        if blocker:
            st.error(f"Not resumed: {blocker}")
        else:
            start_campaign(toolkit_dir, campaign_dir)
            st.rerun()
    if b2.button("Stop", disabled=not running):
        try:
            os.kill(int(status["pid"]), signal.SIGTERM)
            st.info("Stop requested: the current run is ended and the campaign stops.")
        except OSError as exc:
            st.error(f"Could not stop the campaign: {exc}")
    if status and status.get("failed"):
        with st.expander("Failed runs"):
            st.dataframe(pd.DataFrame(status["failed"]), use_container_width=True, hide_index=True)
    current = (status or {}).get("current")
    log_file = campaign_dir / "logs" / f"{current}.log" if current else None
    if log_file is not None and log_file.exists():
        with st.expander("Log of the current run"):
            st.code(log_file.read_text(encoding="utf-8", errors="replace")[-4000:], language="text")
    if running:
        st.caption("Progress updates when the page reruns; press R or interact with the page to refresh.")


def start_campaign(toolkit_dir: Path, campaign_dir: Path) -> None:
    runner = toolkit_dir / "scripts" / "matsense_campaign.py"
    kwargs = {}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    with open(campaign_dir / "runner.log", "a", encoding="utf-8") as log:
        subprocess.Popen(
            [sys.executable, str(runner), str(campaign_dir)],
            cwd=str(toolkit_dir / "scripts"), stdout=log, stderr=subprocess.STDOUT, **kwargs,
        )
    time.sleep(0.5)


def experiment_results(toolkit_dir: Path | None) -> None:
    mev = load_toolkit_module(toolkit_dir, "matsense_eval")
    settle_s = float(getattr(mev, "SETTLE_S", 90.0))
    harness_modes = tuple(getattr(mev, "HARNESS_MODES", ("agent_error",)))

    campaigns = list_campaigns(CAMPAIGN_ROOT)
    # Campaigns with results first, newest first within each group.
    campaigns.sort(key=lambda c: not any((c / "runs").glob("*/run_summary.json")))
    c1, c2 = st.columns([2, 3])
    names = [c.name for c in campaigns]
    choice = c1.selectbox("Campaign", names + ["Other folder…"], key="exp_results_pick") if names else "Other folder…"
    if choice == "Other folder…":
        other = c2.text_input("Campaign folder (containing runs/)", key="exp_results_dir")
        if not other:
            st.caption(f"No campaigns under {CAMPAIGN_ROOT}. Point to a folder that contains runs/<run>/run_summary.json.")
            return
        campaign_dir = Path(other)
    else:
        campaign_dir = CAMPAIGN_ROOT / choice

    df, unsettled = load_campaign_summaries(campaign_dir, settle_s, harness_modes)
    if unsettled:
        st.caption(f"{unsettled} run summaries were written less than {settle_s:.0f} s ago and are left out until they settle.")
    if df.empty:
        st.info("No finished runs in this campaign yet.")
        return

    valid = df[~df["harness_failure"]]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Runs", len(df))
    m2.metric("Completed", int(valid["completed"].sum()))
    m3.metric("Collisions", int((valid["termination_mode"] == "collision").sum()))
    m4.metric("Harness failures (excluded)", int(df["harness_failure"].sum()))
    if df["harness_failure"].any():
        st.caption("A run that died inside the agent never drove: it is not a failure of its arm and is excluded from every count below.")
    profiles = sorted(set(valid.get("profile_name", pd.Series(dtype=str)).dropna().astype(str)))
    if len(profiles) > 1:
        st.warning(f"This campaign mixes profiles: {', '.join(profiles)}.")

    st.subheader("Outcomes per arm")
    group_cols = ["weather", "arm"]
    aggregations = {
        "runs": ("run_id", "count"),
        "completed": ("completed", "sum"),
        "collisions": ("termination_mode", lambda v: int((v == "collision").sum())),
        "stopped": ("termination_mode", lambda v: int((v == "stopped").sum())),
    }
    if "route_completion_ratio" in valid.columns:
        aggregations["completion_median"] = ("route_completion_ratio", "median")
    if "min_ttc_proxy_s" in valid.columns:
        aggregations["min_ttc_median"] = ("min_ttc_proxy_s", "median")
    table = valid.groupby(group_cols).agg(**aggregations).reset_index()
    table["completion_rate"] = table["completed"] / table["runs"]
    st.dataframe(table.round(3), use_container_width=True, hide_index=True)
    fig = px.bar(
        table, x="arm", y="completion_rate", color="weather", barmode="group",
        color_discrete_map=SCENARIO_COLORS, title="Share of runs that complete the route",
    )
    fig.update_layout(template="plotly_white", yaxis_range=[0, 1], legend_title_text="", margin=dict(l=20, r=20, t=60, b=20))
    st.plotly_chart(fig, use_container_width=True)

    metrics = [m for m in OUTCOME_METRICS if m in valid.columns]
    metric = st.selectbox("Metric", metrics, format_func=lambda m: OUTCOME_METRICS[m], key="exp_metric")
    fig = px.box(
        valid, x="arm", y=metric, color="weather", points="all", hover_data=["run_id", "termination_mode"],
        color_discrete_map=SCENARIO_COLORS,
    )
    fig.update_layout(template="plotly_white", yaxis_title=OUTCOME_METRICS[metric], legend_title_text="",
                      margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Paired comparison")
    arms_present = sorted(valid["arm"].unique())
    baseline = st.selectbox(
        "Baseline arm", arms_present,
        index=arms_present.index("standard") if "standard" in arms_present else 0, key="exp_baseline",
    )
    st.caption(
        "Runs are paired on condition, seed and replicate, so each difference compares the same "
        "situation under two sensing arms. Completion is compared with Fisher's exact test."
    )
    keys = [k for k in ("condition_type", "weather", "seed", "replicate") if k in valid.columns]
    rows = []
    base = valid[valid["arm"] == baseline].set_index(keys)
    for arm in arms_present:
        if arm == baseline:
            continue
        other = valid[valid["arm"] == arm].set_index(keys)
        joined = other[[metric, "completed"]].join(base[[metric, "completed"]], rsuffix="_base", how="inner").dropna(
            subset=[metric, f"{metric}_base"]
        )
        row = {"arm": arm, "vs": baseline, "pairs": len(joined)}
        if len(joined) >= 2 and mev is not None:
            try:
                res = mev.paired(joined[metric], joined[f"{metric}_base"])
                row.update({"mean difference": res["mean"], "95% CI": f"[{res['ci'][0]:.3g}, {res['ci'][1]:.3g}]",
                            "Wilcoxon p": res["p"]})
                a = valid[valid["arm"] == arm]
                b = valid[valid["arm"] == baseline]
                row["completion p (Fisher)"] = mev.fisher(int(a["completed"].sum()), len(a), int(b["completed"].sum()), len(b))
            except Exception as exc:  # noqa: BLE001
                row["note"] = f"statistics unavailable: {exc}"
        elif len(joined) >= 1:
            row["mean difference"] = float((joined[metric] - joined[f"{metric}_base"]).mean())
        rows.append(row)
    if rows:
        st.dataframe(pd.DataFrame(rows).round(4), use_container_width=True, hide_index=True)
    else:
        st.caption("Only one arm in this campaign: nothing to compare.")

    st.subheader("Run traces")
    st.caption("The same situation under each arm, tick by tick, from the behaviour log.")
    combos = valid[keys].drop_duplicates().astype(str).agg(" / ".join, axis=1).tolist() if keys else []
    if combos:
        pick = st.selectbox("Situation (" + " / ".join(keys) + ")", sorted(set(combos)), key="exp_trace_pick")
        sel = valid[valid[keys].astype(str).agg(" / ".join, axis=1) == pick]
        signal_col = st.selectbox(
            "Signal", ["speed_mps", "brake", "throttle", "steer", "cross_track_error_m", "progress_m", "ttc_proxy_s", "drop_frac"],
            key="exp_trace_signal",
        )
        fig = go.Figure()
        for _, run in sel.iterrows():
            clog = campaign_dir / "clog" / f"clog_{run['run_id']}.csv"
            if not clog.exists():
                continue
            trace = pd.read_csv(clog, usecols=lambda c: c in ("t_s", signal_col))
            if signal_col not in trace.columns:
                continue
            fig.add_scatter(x=trace["t_s"], y=pd.to_numeric(trace[signal_col], errors="coerce"), mode="lines",
                            name=f"{run['arm']} ({run['termination_mode']})")
        if fig.data:
            fig.update_layout(template="plotly_white", xaxis_title="time (s)", yaxis_title=signal_col,
                              margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("No behaviour logs found for this situation.")

    with st.expander("All runs"):
        st.dataframe(df, use_container_width=True, hide_index=True)

    if mev is not None and st.button("Save results to Evidence"):
        payload = {
            "campaign": str(campaign_dir),
            "outcomes_per_arm": {f"{r['weather']}/{r['arm']}": {k: r[k] for k in table.columns if k not in group_cols}
                                 for r in table.to_dict("records")},
            "paired_vs_" + baseline: {r["arm"]: r for r in rows},
            "metric": metric,
        }
        out = mev.artefact(OUTPUT_ANALYSIS / f"experiment_{campaign_dir.name}.json", json.loads(json.dumps(payload, default=float)),
                           tool="app.py experiment", inputs=[campaign_dir / "campaign.json", campaign_dir / "runs"], quiet=True)
        st.success(f"Written {out.name}; open it in the Evidence tab.")


def experiment_page(toolkit_dir: Path | None, config: dict) -> None:
    st.header("Experiment")
    if toolkit_dir is None or not (toolkit_dir / "scripts" / "matsense_campaign.py").exists():
        st.warning("The Experiment page needs the MatSense runtime toolkit. Check the Toolkit path in the sidebar.")
        return
    tab_campaign, tab_results = st.tabs(["Campaign", "Results"])
    with tab_campaign:
        experiment_campaign(toolkit_dir, config)
    with tab_results:
        experiment_results(toolkit_dir)


def run_viewer(toolkit_dir: Path | None, config: dict) -> None:
    st.header("Run Viewer")
    st.caption("CARLA must already be running with the target map.")
    script = toolkit_dir / "scripts" / "carla_pygame_lidar_dataset_recorder_friendly.py" if toolkit_dir is not None else Path()
    if toolkit_dir is None or not script.exists():
        st.warning(
            "Run Viewer requires the bundled MatSense runtime toolkit or another toolkit path, plus a running CARLA server. "
            "Dataset Analyzer remains testable through the bundled sample dataset."
        )
        st.code("Check the Toolkit path in the sidebar.", language="text")
        return

    st.subheader("Input and Output Paths")
    path_c1, path_c2 = st.columns(2)
    with path_c1:
        traj_txt = path_browser("Trajectory TXT", str(config.get("traj_txt", "")), "viewer_traj_txt", "file", (".txt",))
        parked_json = path_browser("Parked JSON", str(config.get("parked_json", "")), "viewer_parked_json", "file", (".json",))
    with path_c2:
        traj_json = path_browser("Trajectory JSON", str(config.get("traj_json", "")), "viewer_traj_json", "file", (".json",))
        dataset_root = path_browser(
            "Dataset root",
            app_relative_path(str(config.get("dataset_root", "output_dataset"))),
            "viewer_dataset_root",
            "directory",
        )

    tool_config = load_tool_config(toolkit_dir)
    profile_names = list(tool_config.get("profiles", {}))
    presets = tool_config.get("launch_presets", {})

    defaults = {
        "rv_weather": str(config.get("weather", "nominal")),
        "rv_mode": VIEW_MODE_ALIASES.get(str(config.get("mode", "camera_triple")), str(config.get("mode", "camera_triple"))),
        "rv_norm": str(config.get("display_normalization", "fixed")),
        "rv_pct": float(config.get("display_percentile", 95)),
        "rv_scenario": "",
        "rv_base_nominal": bool(config.get("use_base_nominal", False)),
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    if st.session_state["rv_weather"] not in WEATHER_OPTIONS:
        st.session_state["rv_weather"] = "nominal"
    if st.session_state["rv_mode"] not in VIEW_OPTIONS:
        st.session_state["rv_mode"] = "camera_triple"
    if st.session_state["rv_norm"] not in ("fixed", "percentile"):
        st.session_state["rv_norm"] = "fixed"

    def apply_preset() -> None:
        preset = presets.get(st.session_state.get("rv_preset", ""), {})
        if not preset:
            return
        if preset.get("weather") in WEATHER_OPTIONS:
            st.session_state["rv_weather"] = preset["weather"]
        mode_value = VIEW_MODE_ALIASES.get(str(preset.get("mode", "")), str(preset.get("mode", "")))
        if mode_value in VIEW_OPTIONS:
            st.session_state["rv_mode"] = mode_value
        if preset.get("display_normalization") in ("fixed", "percentile"):
            st.session_state["rv_norm"] = preset["display_normalization"]
        if "display_percentile" in preset:
            st.session_state["rv_pct"] = float(preset["display_percentile"])
        if "scenario_name" in preset:
            st.session_state["rv_scenario"] = str(preset["scenario_name"])
        st.session_state["rv_base_nominal"] = bool(preset.get("use_base_nominal", False))

    lidar_default = LIDAR_MODELS[DEFAULT_LIDAR_MODEL]
    for key, field in (
        ("rv_channels", "channels"),
        ("rv_points_per_sweep", "points_per_sweep"),
        ("rv_range", "lidar_range"),
        ("rv_upper_fov", "upper_fov"),
        ("rv_lower_fov", "lower_fov"),
    ):
        st.session_state.setdefault(key, lidar_default[field])

    def apply_lidar_model() -> None:
        model = LIDAR_MODELS.get(st.session_state.get("rv_lidar_model", ""))
        if not model:
            return
        st.session_state["rv_channels"] = model["channels"]
        st.session_state["rv_points_per_sweep"] = model["points_per_sweep"]
        st.session_state["rv_range"] = model["lidar_range"]
        st.session_state["rv_upper_fov"] = model["upper_fov"]
        st.session_state["rv_lower_fov"] = model["lower_fov"]

    st.subheader("Run Settings")
    c1, c2 = st.columns(2)
    if presets:
        c1.selectbox(
            "Preset",
            ["Custom"] + list(presets),
            key="rv_preset",
            on_change=apply_preset,
            help="Fills weather, view and display settings. Everything stays editable below.",
        )
    lidar_choices = list(LIDAR_MODELS) + ["Custom"]
    lidar_model = c2.selectbox(
        "LiDAR model",
        lidar_choices,
        index=lidar_choices.index(DEFAULT_LIDAR_MODEL),
        key="rv_lidar_model",
        on_change=apply_lidar_model,
        help="Fills the LiDAR sensor section. Values stay editable there.",
    )
    if lidar_model in LIDAR_MODELS:
        c2.caption(LIDAR_MODELS[lidar_model]["note"])

    with st.form("viewer"):
        c1, c2, c3 = st.columns(3)
        weather = c1.selectbox("Weather", WEATHER_OPTIONS, key="rv_weather")
        mode = c2.selectbox("View", list(VIEW_OPTIONS), key="rv_mode", format_func=lambda key: VIEW_OPTIONS[key])
        profile_default = str(config.get("profile_name") or tool_config.get("default_profile") or "")
        if profile_names:
            profile_name = c3.selectbox(
                "Material profile",
                profile_names,
                index=profile_names.index(profile_default) if profile_default in profile_names else 0,
                help="Profile whose coefficients the viewer applies. The tool config's default is the calibrated one.",
            )
        else:
            profile_name = profile_default

        c1, c2, c3 = st.columns(3)
        autopilot = c1.checkbox(
            "Autopilot",
            value=bool(config.get("autopilot", False)),
            help="CARLA drives the ego vehicle. Ignored when a trajectory is given.",
        )
        save_dataset = c2.checkbox("Save dataset", value=bool(config.get("save_dataset", False)))

        with st.expander("Recording"):
            c1, c2, c3 = st.columns(3)
            scene_id = c1.text_input("Scene ID", value=str(config.get("scene_id", "scene_001")))
            scenario_name = c2.text_input(
                "Scenario name",
                key="rv_scenario",
                placeholder="same as Weather",
                help="Folder the dataset is saved under. Leave empty to use the selected weather.",
            )
            save_every = c3.number_input("Save every N frames", value=int(config.get("save_every", 10)), min_value=1)
            c1, c2, c3 = st.columns(3)
            max_save_frames = c1.number_input(
                "Max saved frames", value=int(config.get("max_save_frames", 0)), min_value=0, help="0 = no limit."
            )
            save_start_delay = c2.number_input(
                "Start saving after (s)", value=float(config.get("save_start_delay_seconds", 0.0)), min_value=0.0, step=0.5
            )
            strict_sync_timeout = c3.number_input(
                "Sensor sync timeout (s)", value=float(config.get("strict_sync_timeout", 0.5)), min_value=0.0, step=0.1
            )

        with st.expander("Display"):
            c1, c2, c3 = st.columns(3)
            display_normalization = c1.selectbox("Pseudo scale", ["fixed", "percentile"], key="rv_norm")
            display_percentile = c2.number_input("Display percentile", key="rv_pct", step=1.0)
            use_base_nominal = c3.checkbox(
                "Use nominal base values",
                key="rv_base_nominal",
                help="Use the profile's nominal base values directly instead of the raw-intensity-driven correction.",
            )
            c1, c2, c3 = st.columns(3)
            width = c1.number_input("Width", value=int(config.get("width", 1600)), step=100)
            height = c2.number_input("Height", value=int(config.get("height", 900)), step=100)
            fps = c3.number_input("FPS", value=int(config.get("fps", 20)), step=1)

        with st.expander("Trajectory and parked vehicles"):
            c1, c2, c3 = st.columns(3)
            follow_options = ["teleport", "control"]
            follow_default = str(config.get("follow_mode", "teleport"))
            follow_mode = c1.selectbox(
                "Follow mode",
                follow_options,
                index=follow_options.index(follow_default) if follow_default in follow_options else 0,
                help="teleport replays the poses exactly; control drives towards them.",
            )
            traj_step = c2.number_input("Trajectory step", value=int(config.get("traj_step", 5)), min_value=1)
            control_smoothing = c3.number_input(
                "Control smoothing", value=float(config.get("control_smoothing", 0.30)), min_value=0.0, max_value=1.0, step=0.05
            )
            c1, c2, c3 = st.columns(3)
            utm_offset_x = c1.number_input(
                "UTM offset X", value=float(config.get("utm_offset_x", 0.0)), format="%.2f",
                help="Leave both at 0 to read the offset from the map's .xodr file.",
            )
            utm_offset_y = c2.number_input("UTM offset Y", value=float(config.get("utm_offset_y", 0.0)), format="%.2f")
            traj_z_offset = c3.number_input("Trajectory Z offset", value=float(config.get("traj_z_offset", 0.5)), step=0.1)
            c1, c2, c3 = st.columns(3)
            parked_z_offset = c1.number_input("Parked Z offset", value=float(config.get("parked_z_offset", 0.15)), step=0.05)
            parked_limit = c2.number_input(
                "Max parked vehicles", value=int(config.get("parked_limit", 0)), min_value=0, help="0 = all."
            )
            seed = c3.number_input("Seed", value=int(config.get("seed", 42)), step=1)

        with st.expander("LiDAR sensor"):
            c1, c2, c3 = st.columns(3)
            st.caption("The viewer completes one full sweep per frame, so points per second = points per sweep x FPS.")
            c1, c2 = st.columns(2)
            channels = c1.number_input("Channels", key="rv_channels", min_value=1, step=1)
            points_per_sweep = c2.number_input("Points per sweep", key="rv_points_per_sweep", min_value=100, step=1000)
            c1, c2, c3 = st.columns(3)
            lidar_range = c1.number_input("Range (m)", key="rv_range", min_value=1.0)
            upper_fov = c2.number_input("Upper FOV (deg)", key="rv_upper_fov")
            lower_fov = c3.number_input("Lower FOV (deg)", key="rv_lower_fov")

        with st.expander("CARLA connection"):
            c1, c2, c3 = st.columns(3)
            host = c1.text_input("Host", value=str(config.get("host", "127.0.0.1")))
            port = c2.number_input("Port", value=int(config.get("port", 2000)), step=1)
            tm_port = c3.number_input("Traffic Manager port", value=int(config.get("tm_port", 8000)), step=1)
            allow_version_mismatch = st.checkbox(
                "Allow client/server version mismatch",
                value=bool(config.get("allow_version_mismatch", False)),
                help="The viewer refuses to start on a mismatch, because mismatched builds can crash natively.",
            )

        launch = st.form_submit_button("Start Viewer", type="primary")

    if weather in UNMEASURED_CONDITIONS:
        st.warning(UNMEASURED_CONDITIONS[weather])
    profile_notes = str(tool_config.get("profiles", {}).get(profile_name, {}).get("notes", ""))
    if profile_name and not profile_notes:
        st.caption(f"Profile `{profile_name}` carries no calibration notes: treat its coefficients as declared, not measured.")

    values = {
        "host": host.strip(),
        "port": int(port),
        "tm_port": int(tm_port),
        "width": int(width),
        "height": int(height),
        "fps": int(fps),
        "weather": weather,
        "mode": mode,
        "display_normalization": display_normalization,
        "display_percentile": display_percentile,
        "use_base_nominal": use_base_nominal,
        "traj_txt": traj_txt.strip(),
        "traj_json": traj_json.strip(),
        "parked_json": parked_json.strip(),
        "dataset_root": dataset_root,
        "scene_id": scene_id,
        "scenario_name": scenario_name.strip() or weather,
        "autopilot": autopilot,
        "save_dataset": save_dataset,
        "save_every": int(save_every),
        "profile_name": profile_name,
        "traj_step": int(traj_step),
        "utm_offset_x": f"{utm_offset_x:.2f}",
        "utm_offset_y": f"{utm_offset_y:.2f}",
        "traj_z_offset": traj_z_offset,
        "follow_mode": follow_mode,
        "control_smoothing": control_smoothing,
        "parked_z_offset": parked_z_offset,
        "parked_limit": int(parked_limit),
        "seed": int(seed),
        "max_save_frames": int(max_save_frames),
        "save_start_delay_seconds": save_start_delay,
        "strict_sync_timeout": strict_sync_timeout,
        "channels": int(channels),
        "pps": int(points_per_sweep) * int(fps),
        "lidar_range": lidar_range,
        "upper_fov": upper_fov,
        "lower_fov": lower_fov,
        "allow_version_mismatch": allow_version_mismatch,
    }

    if trajectory_txt_looks_like_utm(values["traj_txt"]):
        if abs(utm_offset_x) < 1e-9 and abs(utm_offset_y) < 1e-9:
            detected_offset = read_xodr_offset(values["traj_txt"], toolkit_dir)
            if detected_offset is not None:
                x, y, source = detected_offset
                values["utm_offset_x"] = f"{x:.2f}"
                values["utm_offset_y"] = f"{y:.2f}"
                st.info(f"Auto-loaded UTM offsets from {source}: x={x:.2f}, y={y:.2f}")

    st.subheader("CARLA Status")
    status_c1, status_c2 = st.columns([1, 4])
    if status_c1.button("Check CARLA"):
        st.session_state["rv_preflight"] = check_carla(toolkit_dir, values["host"], values["port"])
    preflight = st.session_state.get("rv_preflight")
    if preflight and (preflight["host"], preflight["port"]) == (values["host"], values["port"]):
        with status_c2:
            render_preflight(preflight, values["allow_version_mismatch"])
    else:
        status_c2.caption("Not checked yet. Start Viewer runs the same check before launching.")

    if launch:
        preflight = check_carla(toolkit_dir, values["host"], values["port"])
        st.session_state["rv_preflight"] = preflight
        problem = preflight_blocker(preflight, values["allow_version_mismatch"])
        if problem:
            st.error(f"Viewer not started: {problem}")
        else:
            start_viewer(toolkit_dir, config, values)

    running_pid = viewer_pid()
    if running_pid is not None:
        st.success(f"Viewer running (PID {running_pid}). Close the pygame window or press Stop Viewer to end it.")

    if VIEWER_STDERR.exists() and VIEWER_STDERR.stat().st_size > 0:
        with st.expander("Last viewer stderr"):
            st.code(VIEWER_STDERR.read_text(encoding="utf-8", errors="replace")[-4000:], language="text")

    if st.button("Stop Viewer", disabled=running_pid is None):
        st.info(stop_viewer())
        st.rerun()


def start_viewer(toolkit_dir: Path, config: dict, values: dict) -> None:
    cmd = build_viewer_command(toolkit_dir, config, values)
    rendered_cmd = " ".join(f'"{part}"' if " " in str(part) else str(part) for part in cmd)
    VIEWER_COMMAND.write_text(rendered_cmd, encoding="utf-8")
    creationflags = subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0
    stdout_file = VIEWER_STDOUT.open("w", encoding="utf-8")
    stderr_file = VIEWER_STDERR.open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            cmd,
            cwd=str(toolkit_dir / "scripts"),
            env=toolkit_env(toolkit_dir),
            stdout=stdout_file,
            stderr=stderr_file,
            creationflags=creationflags,
        )
    finally:
        stdout_file.close()
        stderr_file.close()
    PID_FILE.write_text(str(process.pid), encoding="utf-8")
    time.sleep(1.0)
    exit_code = process.poll()
    if exit_code is None:
        st.success(f"Viewer started with PID {process.pid}.")
    else:
        PID_FILE.unlink(missing_ok=True)
        st.error(f"Viewer exited immediately with code {exit_code}.")
        if VIEWER_STDERR.exists():
            st.code(VIEWER_STDERR.read_text(encoding="utf-8", errors="replace")[-4000:], language="text")
    with st.expander("Command"):
        st.code(rendered_cmd, language="powershell")


def _artefact_rows() -> pd.DataFrame:
    """An inventory of everything under output_analysis/.

    Artefacts written through matsense_eval.artefact carry the tool that
    produced them, the repository revision and a digest of their inputs. Older
    ones do not, and the table says so rather than papering over it: a result
    whose producing code is unknown deserves suspicion, not equal billing.
    """
    rows = []
    for f in sorted(OUTPUT_ANALYSIS.glob("*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:                                   # noqa: BLE001
            rows.append({"artefact": f.name, "tool": "(unreadable)",
                         "revision": "", "written": "", "provenance": False})
            continue
        prov = d.get("_provenance") if isinstance(d, dict) else None
        rows.append({
            "artefact": f.name,
            "tool": (prov or {}).get("tool", ""),
            "revision": (prov or {}).get("git", ""),
            "written": (prov or {}).get(
                "written", time.strftime("%Y-%m-%d %H:%M:%S",
                                         time.localtime(f.stat().st_mtime))),
            "provenance": bool(prov),
        })
    return pd.DataFrame(rows).sort_values("written", ascending=False)


def _render_value(name: str, value) -> None:
    """Render one piece of an artefact without per-artefact code.

    Thirty-five tools write these files and their shapes differ. A viewer
    written for each would have to be rewritten with every new experiment, so
    this recognises the recurring shapes instead: a dict of dicts is a table,
    a list of equal-length lists is a matrix, a flat dict is a two-column
    table, and anything scalar is printed as text.

    That last case is why this function exists in its current form. The first
    version sent everything to st.json, which cannot take a bare string or
    number and reported a JSON parse error on every path, count and float in
    the file.
    """
    if value is None:
        st.caption("null")
        return
    if isinstance(value, (str, int, float, bool)):
        st.write(value)
        return
    if isinstance(value, dict) and value and all(
            isinstance(v, dict) for v in value.values()):
        st.dataframe(pd.DataFrame(value).T, use_container_width=True)
        return
    if isinstance(value, list) and value and all(
            isinstance(v, list) for v in value) and len({len(v) for v in value}) == 1:
        st.dataframe(pd.DataFrame(value), use_container_width=True)
        return
    if isinstance(value, list) and all(
            isinstance(v, (str, int, float, bool, type(None))) for v in value):
        st.write(", ".join("null" if v is None else str(v) for v in value))
        return
    if isinstance(value, dict) and value and all(
            isinstance(v, (str, int, float, bool, type(None))) for v in value.values()):
        st.dataframe(pd.DataFrame({"key": list(value), "value": list(value.values())}),
                     use_container_width=True, hide_index=True)
        return
    st.json(value)


def evidence_browser() -> None:
    st.header("Evidence")
    st.caption(
        "Every experiment writes an artefact into output_analysis/. "
        "This is where you read them without opening the JSON by hand."
    )
    if not OUTPUT_ANALYSIS.exists():
        st.info(
            f"No artefacts yet: {OUTPUT_ANALYSIS} does not exist. It is created by the "
            "evaluation tools (matsense_eval.artefact) when an experiment writes its results."
        )
        return

    df = _artefact_rows()
    if df.empty:
        st.info("No artefacts yet.")
        return

    n_prov = int(df["provenance"].sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("artefacts", len(df))
    c2.metric("with provenance", n_prov)
    c3.metric("without", len(df) - n_prov)
    if n_prov < len(df):
        st.warning(
            f"{len(df) - n_prov} artefacts do not record which code produced "
            "them: they were written before the tools started going through "
            "matsense_eval.artefact. Re-running the tool updates them."
        )

    q = st.text_input("Filter by name or tool", "", key="ev_q").strip().lower()
    view = df[df.apply(lambda r: q in r["artefact"].lower()
                       or q in str(r["tool"]).lower(), axis=1)] if q else df
    st.dataframe(view, use_container_width=True, hide_index=True)

    choice = st.selectbox("Open an artefact", view["artefact"].tolist(),
                          key="ev_pick")
    if not choice:
        return
    data = json.loads((OUTPUT_ANALYSIS / choice).read_text())
    prov = data.pop("_provenance", None) if isinstance(data, dict) else None
    if prov:
        st.caption(
            f"produced by **{prov.get('tool', '?')}** at revision "
            f"`{prov.get('git', '?')}` on {prov.get('written', '?')}"
        )
        if prov.get("inputs"):
            with st.expander("Inputs and their digest"):
                st.dataframe(pd.DataFrame(
                    [{"input": k, "digest": v} for k, v in prov["inputs"].items()]),
                    use_container_width=True, hide_index=True)
    else:
        st.caption("No provenance: the version of the code that produced this "
                   "is unknown.")

    if isinstance(data, dict):
        for key, value in data.items():
            st.subheader(key)
            _render_value(key, value)
    else:
        st.json(data)


def main() -> None:
    toolkit_dir, config = sidebar_config()
    tab_run, tab_profile, tab_experiment, tab_analyzer, tab_evidence = st.tabs(
        ["Run Viewer", "Profile", "Experiment", "Dataset Analyzer", "Evidence"])
    with tab_run:
        run_viewer(toolkit_dir, config)
    with tab_profile:
        profile_browser(toolkit_dir)
    with tab_experiment:
        experiment_page(toolkit_dir, config)
    with tab_analyzer:
        dataset_analyzer(toolkit_dir, config)
    with tab_evidence:
        evidence_browser()


if __name__ == "__main__":
    main()
