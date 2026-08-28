from __future__ import annotations

import csv
import json
import math
import os
import re
import signal
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
PID_FILE = Path(__file__).with_name(".matsense_viewer.pid")
VIEWER_STDOUT = APP_DIR / "viewer_stdout.log"
VIEWER_STDERR = APP_DIR / "viewer_stderr.log"
VIEWER_COMMAND = APP_DIR / "viewer_command.txt"

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
    "profile_name": "carla_default",
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
        "--profile-name", values["profile_name"],
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
        import tkinter as tk
        from tkinter import filedialog

        current = Path(st.session_state.get(input_key, "") or default or APP_DIR)
        if kind == "directory":
            initial_dir = current if current.exists() and current.is_dir() else APP_DIR
        else:
            initial_dir = current.parent if current.exists() else APP_DIR

        root = tk.Tk()
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

    c1, c2 = st.columns([5, 1])
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

    with st.form("viewer"):
        c1, c2, c3 = st.columns(3)
        host = c1.text_input("Host", value=str(config.get("host", "127.0.0.1")))
        port = c2.number_input("Port", value=int(config.get("port", 2000)), step=1)
        tm_port = c3.number_input("Traffic Manager port", value=int(config.get("tm_port", 8000)), step=1)

        c1, c2, c3 = st.columns(3)
        width = c1.number_input("Width", value=int(config.get("width", 1600)), step=100)
        height = c2.number_input("Height", value=int(config.get("height", 900)), step=100)
        fps = c3.number_input("FPS", value=int(config.get("fps", 20)), step=1)

        c1, c2, c3 = st.columns(3)
        weather_options = ["nominal", "rain", "snow"]
        weather_default = str(config.get("weather", "nominal"))
        weather = c1.selectbox(
            "Weather",
            weather_options,
            index=weather_options.index(weather_default) if weather_default in weather_options else 0,
        )
        view_options = {
            "camera_triple": "RGB overlays: all 3",
            "material": "Material classes",
            "pseudo": "Pseudo-reflectance",
            "intensity": "CARLA raw intensity",
        }
        mode_keys = list(view_options.keys())
        mode_default = str(config.get("mode", "camera_triple"))
        mode = c2.selectbox(
            "View",
            mode_keys,
            index=mode_keys.index(mode_default) if mode_default in mode_keys else 0,
            format_func=lambda key: view_options[key],
        )
        display_normalization = c3.selectbox("Pseudo scale", ["fixed", "percentile"], index=["fixed", "percentile"].index(config.get("display_normalization", "fixed")) if config.get("display_normalization") in ["fixed", "percentile"] else 0)

        display_percentile = st.number_input("Display percentile", value=float(config.get("display_percentile", 95)), step=1.0)

        c1, c2, c3 = st.columns(3)
        scene_id = c1.text_input("Scene ID", value=str(config.get("scene_id", "scene_001")))
        scenario_name = c2.text_input("Scenario name", value=str(config.get("scenario_name", weather)))
        save_every = c3.number_input("Save every N frames", value=int(config.get("save_every", 10)), min_value=1)

        c1, c2, c3 = st.columns(3)
        autopilot = c1.checkbox("Autopilot", value=bool(config.get("autopilot", False)))
        save_dataset = c2.checkbox("Save dataset", value=bool(config.get("save_dataset", False)))

        launch = st.form_submit_button("Start Viewer")

    values = {
        "host": host,
        "port": port,
        "tm_port": tm_port,
        "width": width,
        "height": height,
        "fps": fps,
        "weather": weather,
        "mode": mode,
        "display_normalization": display_normalization,
        "display_percentile": display_percentile,
        "traj_txt": traj_txt.strip(),
        "traj_json": traj_json.strip(),
        "parked_json": parked_json.strip(),
        "dataset_root": dataset_root,
        "scene_id": scene_id,
        "scenario_name": scenario_name,
        "autopilot": autopilot,
        "save_dataset": save_dataset,
        "save_every": save_every,
        "profile_name": str(config.get("profile_name", "carla_default")),
        "traj_step": str(config.get("traj_step", 5)),
        "utm_offset_x": str(config.get("utm_offset_x", 0.0)),
        "utm_offset_y": str(config.get("utm_offset_y", 0.0)),
        "traj_z_offset": str(config.get("traj_z_offset", 0.5)),
        "follow_mode": str(config.get("follow_mode", "teleport")),
        "control_smoothing": str(config.get("control_smoothing", 0.30)),
        "parked_z_offset": str(config.get("parked_z_offset", 0.15)),
        "parked_limit": str(config.get("parked_limit", 0)),
        "seed": str(config.get("seed", 42)),
        "max_save_frames": str(config.get("max_save_frames", 0)),
        "save_start_delay_seconds": str(config.get("save_start_delay_seconds", 0.0)),
        "strict_sync_timeout": str(config.get("strict_sync_timeout", 0.5)),
    }

    if trajectory_txt_looks_like_utm(values["traj_txt"]):
        try:
            offset_x = float(values["utm_offset_x"])
            offset_y = float(values["utm_offset_y"])
        except ValueError:
            offset_x = 0.0
            offset_y = 0.0
        if abs(offset_x) < 1e-9 and abs(offset_y) < 1e-9:
            detected_offset = read_xodr_offset(values["traj_txt"], toolkit_dir)
            if detected_offset is not None:
                x, y, source = detected_offset
                values["utm_offset_x"] = f"{x:.2f}"
                values["utm_offset_y"] = f"{y:.2f}"
                st.info(f"Auto-loaded UTM offsets from {source}: x={x:.2f}, y={y:.2f}")

    if launch:
        script = toolkit_dir / "scripts" / "carla_pygame_lidar_dataset_recorder_friendly.py"
        if not script.exists():
            st.error(f"Viewer script not found: {script}")
            return
        cmd = build_viewer_command(toolkit_dir, config, values)
        rendered_cmd = " ".join(f'"{part}"' if " " in str(part) else str(part) for part in cmd)
        VIEWER_COMMAND.write_text(rendered_cmd, encoding="utf-8")
        env = os.environ.copy()
        src_path = str(toolkit_dir / "src")
        env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")
        creationflags = subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0
        stdout_file = VIEWER_STDOUT.open("w", encoding="utf-8")
        stderr_file = VIEWER_STDERR.open("w", encoding="utf-8")
        try:
            process = subprocess.Popen(
                cmd,
                cwd=str(toolkit_dir / "scripts"),
                env=env,
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
            st.error(f"Viewer exited immediately with code {exit_code}.")
            if VIEWER_STDERR.exists():
                st.code(VIEWER_STDERR.read_text(encoding="utf-8", errors="replace")[-4000:], language="text")
        with st.expander("Command"):
            st.code(rendered_cmd, language="powershell")

    if VIEWER_STDERR.exists() and VIEWER_STDERR.stat().st_size > 0:
        with st.expander("Last viewer stderr"):
            st.code(VIEWER_STDERR.read_text(encoding="utf-8", errors="replace")[-4000:], language="text")

    if st.button("Stop Viewer"):
        st.info(stop_viewer())


def main() -> None:
    toolkit_dir, config = sidebar_config()
    tab_run, tab_analyzer = st.tabs(["Run Viewer", "Dataset Analyzer"])
    with tab_run:
        run_viewer(toolkit_dir, config)
    with tab_analyzer:
        dataset_analyzer(toolkit_dir, config)


if __name__ == "__main__":
    main()
