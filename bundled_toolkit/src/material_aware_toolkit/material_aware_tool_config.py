from __future__ import annotations

import copy
import json
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "material_aware_tool_config.json"


DEFAULT_TOOL_CONFIG = {
    "default_profile": "carla_default",
    "profiles": {
        "carla_default": {
            "default_material": "unknown",
            "display_mode_sequence": ["intensity", "global", "material"],
            "semantic_to_material": {
                "1": "asphalt",
                "2": "sidewalk",
                "3": "building",
                "4": "building",
                "9": "vegetation",
                "14": "car",
                "15": "car",
                "16": "car",
                "18": "car",
                "19": "car",
                "24": "asphalt",
            },
            "display_colors": {
                "asphalt": [160, 160, 160],
                "concrete": [210, 190, 150],
                "sidewalk": [235, 210, 110],
                "building": [230, 130, 40],
                "vegetation": [60, 190, 70],
                "car": [240, 70, 120],
                "unknown": [70, 110, 210],
            },
            "nominal_base": {
                "asphalt": 0.20,
                "sidewalk": 0.30,
                "vegetation": 0.70,
                "car": 0.90,
                "building": 0.55,
                "unknown": 0.40,
            },
            "weather_ratio": {
                "nominal": {
                    "asphalt": 1.0,
                    "building": 1.0,
                    "car": 1.0,
                    "sidewalk": 1.0,
                    "vegetation": 1.0,
                    "unknown": 1.0,
                },
                "rain": {
                    "asphalt": 0.281,
                    "building": 0.122,
                    "car": 0.432,
                    "sidewalk": 0.069,
                    "vegetation": 0.890,
                    "unknown": 0.5,
                },
                "snow": {
                    "asphalt": 0.493,
                    "building": 0.280,
                    "car": 1.538,
                    "sidewalk": 0.082,
                    "vegetation": 0.652,
                    "unknown": 0.7,
                },
            },
            "planar_materials": ["asphalt", "building", "sidewalk"],
        }
    },
    "launch_presets": {
        "Nominal demo": {
            "weather": "nominal",
            "mode": "material",
            "display_normalization": "percentile",
            "display_percentile": "95",
            "scenario_name": "nominal",
            "use_base_nominal": False,
        },
        "Rain demo": {
            "weather": "rain",
            "mode": "material",
            "display_normalization": "percentile",
            "display_percentile": "95",
            "scenario_name": "rain",
            "use_base_nominal": False,
        },
        "Snow demo": {
            "weather": "snow",
            "mode": "material",
            "display_normalization": "percentile",
            "display_percentile": "95",
            "scenario_name": "snow",
            "use_base_nominal": False,
        },
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_tool_config(config_path: str | Path | None = None) -> tuple[dict, Path]:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    config = copy.deepcopy(DEFAULT_TOOL_CONFIG)
    if path.exists():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        config = _deep_merge(config, loaded)
    return config, path


def get_profile(config: dict, profile_name: str | None = None) -> tuple[str, dict]:
    profiles = config.get("profiles", {})
    if not profiles:
        raise ValueError("Tool config does not define any profiles")
    name = profile_name or config.get("default_profile") or next(iter(profiles))
    if name not in profiles:
        available = ", ".join(sorted(profiles))
        raise ValueError(f"Unknown material profile '{name}'. Available: {available}")
    return name, copy.deepcopy(profiles[name])


def get_launch_presets(config: dict) -> dict:
    return copy.deepcopy(config.get("launch_presets", {}))


def normalize_profile(profile: dict) -> dict:
    normalized = copy.deepcopy(profile)
    normalized["semantic_to_material"] = {
        int(key): str(value) for key, value in normalized.get("semantic_to_material", {}).items()
    }
    normalized["display_colors"] = {
        str(key): tuple(int(c) for c in value) for key, value in normalized.get("display_colors", {}).items()
    }
    normalized["display_mode_sequence"] = [str(v) for v in normalized.get("display_mode_sequence", [])]
    normalized["planar_materials"] = {str(v) for v in normalized.get("planar_materials", [])}
    normalized["default_material"] = str(normalized.get("default_material", "unknown"))
    normalized["nominal_base"] = {
        str(key): float(value) for key, value in normalized.get("nominal_base", {}).items()
    }
    normalized["weather_ratio"] = {
        str(weather): {str(mat): float(ratio) for mat, ratio in ratios.items()}
        for weather, ratios in normalized.get("weather_ratio", {}).items()
    }
    return normalized
