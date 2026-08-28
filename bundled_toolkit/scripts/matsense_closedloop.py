#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MatSense closed-loop integration — geometry perturbation + PCLA hook.

Structure
---------
1. Config loading — reads your real NOMINAL_BASE / WEATHER_RATIO from the JSON.
2. _match_semantic_tags() — pure-numpy port of match_semantic_lidar_metadata.
3. matsense_perturb() — the single function you call inside your PCLA fork.
4. ClosedLoopSession — spawns co-located sem-LiDAR, owns the perturbation state.
5. BehaviourLogger — one CSV row per tick for RQ4.
6. trajectory_to_route_xml() — converts your recorded .json trajectory to PCLA XML.
7. run_experiment_matrix() — pseudocode template for the 3×3 (weather × mode) sweep.

Where to call this in your PCLA fork
--------------------------------------
Inside the PCLA LiDAR sensor callback, BEFORE the BEV preprocessing step:

    # At the top of your fork's lidar_callback (or wherever raw bytes are parsed):
    raw_pts = np.frombuffer(measurement.raw_data, dtype=np.float32).reshape(-1, 4)
    raw_pts = session.get_perturbed_cloud(raw_pts)   # <-- only change
    # ... rest of PCLA BEV preprocessing unchanged ...

Everything else (BEV histogramming, Transfuser inference, apply_control) stays untouched.
"""

from __future__ import annotations

import csv
import json
import math
import threading
import time as _time

# When this process began. Every run is a fresh process, so this separates what
# STARTING UP costs from what DRIVING costs; without the distinction you end up
# optimising the wrong half, as already happened once with map loading.
def _process_start_time() -> float:
    """When this process really started, not when this module was imported.

    The module is imported late, after torch and after the network weights,
    which are the bulk of startup. Measuring from here reported "session ready
    at +0.0s": true, and completely useless, since it said our part is instant
    rather than where the minutes go.
    """
    try:
        import os as _os
        with open("/proc/self/stat") as f:
            fields = f.read().rsplit(")", 1)[1].split()
        started_ticks = int(fields[19])
        with open("/proc/uptime") as f:
            uptime = float(f.read().split()[0])
        return _time.time() - (uptime - started_ticks / _os.sysconf("SC_CLK_TCK"))
    except Exception:
        return _time.time()


_IMPORT_T = _process_start_time()
from collections import OrderedDict
from pathlib import Path
from typing import NamedTuple

import numpy as np

# ---------------------------------------------------------------------------
# Sys-path setup so this file can be run or imported directly
# ---------------------------------------------------------------------------
import sys

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from material_aware_toolkit.material_aware_tool_config import (
    DEFAULT_CONFIG_PATH as _DEFAULT_CONFIG_PATH,
    get_profile,
    load_tool_config,
    normalize_profile,
)

# ---------------------------------------------------------------------------
# Load calibration tables live from your JSON (always in sync)
# ---------------------------------------------------------------------------
_cfg, _cfg_path = load_tool_config(_DEFAULT_CONFIG_PATH)
_profile_name, _profile = get_profile(_cfg)
_profile = normalize_profile(_profile)

NOMINAL_BASE: dict[str, float] = dict(_profile["nominal_base"])
WEATHER_RATIO: dict[str, dict[str, float]] = dict(_profile["weather_ratio"])
SEMANTIC_TO_MATERIAL: dict[int, str] = dict(_profile["semantic_to_material"])
PLANAR_MATERIALS: set[str] = set(_profile["planar_materials"])
DEFAULT_MATERIAL: str = _profile["default_material"]

# Range-correction polynomial from your calibration (A2·r²+A1·r+A0)
_A2 = 0.01592119599839734
_A1 = -0.6848984378845165
_A0 = 12.437483628508096
_G_R0 = 7.180618849502665
_FIT_RANGE_MIN = 2.0
_FIT_RANGE_MAX = 30.0
_COS_EPS = 0.2


# ---------------------------------------------------------------------------
# Pure-numpy material-matching helpers (ported from main script, no pygame dep)
# ---------------------------------------------------------------------------

def _angles_from_xyz(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xy = np.hypot(xyz[:, 0], xyz[:, 1])
    az = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))
    el = np.degrees(np.arctan2(xyz[:, 2], np.maximum(xy, 1e-6)))
    return az.astype(np.float32), el.astype(np.float32)


def _match_semantic_tags(
    lidar_xyz: np.ndarray,
    semantic_lidar: np.ndarray | None,
    max_ang_deg: float = 0.40,
) -> np.ndarray:
    """Per-point ObjTag (int32), 0 when no ray of the semantic scan corresponds.

    Matching is by DIRECTION, not by distance in metres. The two sensors sample
    incommensurate azimuth grids, so the semantic ray nearest a given ray-cast ray
    is offset by up to about a fifth of a degree. On a surface seen face-on that
    is centimetres; on one seen at grazing incidence the impact point slides along
    the surface and the same fifth of a degree becomes metres, so a metric gate
    rejects a partner that points in almost exactly the right direction. Measured
    with a 0.35 m gate, losses peaked at 25% in the 10-20 m band and at 34% between
    -10 and -4 degrees of elevation, which is exactly where a sensor 2.4 m up sees
    the road most obliquely, and fell to 0.01% below 5 m where the road is steep
    underneath.

    The angle between two directions is the distance between their unit vectors on
    the sphere, chord = 2*sin(theta/2). Working in that space is exact, has no
    seam at +-180 degrees, and needs no weighting of azimuth by cos(elevation) --
    an earlier attempt did weight it, using each point's own cosine, and that
    disagreed with the reference implementation on 14% of points because at 170
    degrees of azimuth a one-degree difference in elevation moves the weighted
    coordinate a long way.

    A k-d tree replaces the per-point Python loop the first version used. That loop
    cost 160 ms for a 16k-point cloud against a 65k-point semantic scan, about 96
    seconds per run, and it searched only a 3x3 neighbourhood of angular bins, so
    it could miss a true nearest neighbour that fell just outside. Validated
    against it on synthetic clouds: 97% identical tags, and in zero cases does this
    version fail to find a partner the loop found -- the 3% are partners the loop
    missed. Sixteen milliseconds instead of a hundred and sixty.
    """
    tags = np.zeros(len(lidar_xyz), dtype=np.int32)
    if semantic_lidar is None or len(semantic_lidar) == 0 or len(lidar_xyz) == 0:
        return tags
    from scipy.spatial import cKDTree

    sem = np.column_stack([semantic_lidar["x"], semantic_lidar["y"],
                           semantic_lidar["z"]]).astype(np.float64)
    lid = np.asarray(lidar_xyz, dtype=np.float64)
    sem_u = sem / np.maximum(np.linalg.norm(sem, axis=1, keepdims=True), 1e-9)
    lid_u = lid / np.maximum(np.linalg.norm(lid, axis=1, keepdims=True), 1e-9)
    chord = 2.0 * np.sin(np.radians(max_ang_deg) / 2.0)
    dist, idx = cKDTree(sem_u).query(lid_u, k=1, distance_upper_bound=chord)
    hit = np.isfinite(dist)
    if hit.any():
        tags[hit] = np.asarray(semantic_lidar["ObjTag"])[idx[hit]].astype(np.int32)
    return tags


# ---------------------------------------------------------------------------
# Material codebook
#
# Everything below used to be a Python comprehension over every return, with a
# str() and a dict lookup per point: `_tags_to_materials`, `_dropout_probs`,
# `_intensity_factors` (twice), `_weighted_weather_ratio` and the unmapped
# count, six or seven passes over tens of thousands of points per frame. The
# values involved are a handful of floats indexed by a class id that never
# exceeds a couple of dozen, so they belong in small arrays gathered with take.
#
# The tables are built once from the same profile dictionaries the old code
# read, so the numbers are the same numbers; only the way they are fetched
# changes. The equivalence is checked by tools/check_matsense_vectorised.py,
# which compares the two implementations bit for bit.
# ---------------------------------------------------------------------------

_MATERIALS: list[str] = sorted(
    set(SEMANTIC_TO_MATERIAL.values()) | set(NOMINAL_BASE) | {DEFAULT_MATERIAL}
)
_MAT_INDEX: dict[str, int] = {m: i for i, m in enumerate(_MATERIALS)}
_MAT_ARRAY = np.array(_MATERIALS, dtype=object)
_DEFAULT_CODE = _MAT_INDEX[DEFAULT_MATERIAL]

# CARLA class ids are small; the table is indexed directly by tag. Anything
# outside it, including the 0 that means "no partner found", maps to the
# default material, which is what the dict .get(..., DEFAULT_MATERIAL) did.
_MAX_TAG = 256
_TAG_TO_CODE = np.full(_MAX_TAG, _DEFAULT_CODE, dtype=np.int32)
_TAG_MAPPED = np.zeros(_MAX_TAG, dtype=bool)
for _t, _m in SEMANTIC_TO_MATERIAL.items():
    if 0 <= int(_t) < _MAX_TAG:
        _TAG_TO_CODE[int(_t)] = _MAT_INDEX.get(str(_m), _DEFAULT_CODE)
        _TAG_MAPPED[int(_t)] = True

# float64, deliberately: the code these replace multiplied two Python floats
# and rounded the product to float32 once. Keeping the tables in float64 and
# casting after the multiply reproduces that rounding exactly; float32 tables
# would round the operands first and can differ in the last bit, which is
# enough to change a dropout draw and with it the experiment.
_BASE_BY_CODE = np.array(
    [NOMINAL_BASE.get(m, NOMINAL_BASE[DEFAULT_MATERIAL]) for m in _MATERIALS],
    dtype=np.float64,
)
_RATIO_BY_CODE: dict[str, np.ndarray] = {
    w: np.array([lut.get(m, lut.get(DEFAULT_MATERIAL, 1.0)) for m in _MATERIALS],
                dtype=np.float64)
    for w, lut in WEATHER_RATIO.items()
}


def _ratio_table(weather: str) -> np.ndarray:
    tab = _RATIO_BY_CODE.get(weather)
    return _RATIO_BY_CODE["nominal"] if tab is None else tab


def _codes_from_tags(tags: np.ndarray) -> np.ndarray:
    """CARLA class ids to material codes, without a Python loop."""
    t = np.asarray(tags)
    idx = np.where((t >= 0) & (t < _MAX_TAG), t, 0).astype(np.intp)
    codes = _TAG_TO_CODE[idx]
    # a tag outside the table was never in the dict either, so it takes the
    # default exactly as .get() gave it
    return np.where((t >= 0) & (t < _MAX_TAG), codes, _DEFAULT_CODE).astype(np.int32)


def _as_codes(materials: np.ndarray) -> np.ndarray:
    """Accept either material codes or the historic object array of strings.

    The public helpers kept their signatures so nothing outside had to change;
    passing codes is the fast path and passing strings still works.
    """
    arr = np.asarray(materials)
    if arr.dtype.kind in "iu":
        return arr.astype(np.int32, copy=False)
    return np.array([_MAT_INDEX.get(str(m), _DEFAULT_CODE) for m in arr], dtype=np.int32)


def _codes_to_materials(codes: np.ndarray) -> np.ndarray:
    return _MAT_ARRAY[np.asarray(codes).astype(np.intp)]


def _tags_to_materials(tags: np.ndarray) -> np.ndarray:
    return _codes_to_materials(_codes_from_tags(tags))


# ---------------------------------------------------------------------------
# Perturbation config
# ---------------------------------------------------------------------------

class PerturbConfig(NamedTuple):
    """Calibration-bounded perturbation knobs.

    dropout_gain and dropout_cap must be justified by your ROS-bag statistics
    (max observed return-loss per material class). These defaults are conservative
    starting points — tune them before the submission."""
    dropout_gain: float = 0.6    # maps (1 - weather_ratio) -> dropout probability
    dropout_cap: float = 0.55    # never drop more than this fraction of any class
    jitter_gain: float = 0.10    # range jitter std (m) for strongly attenuated points
    global_dropout: float = 0.0  # uniform dropout disabled for the 'global' baseline
    seed: int = 1234             # fix + log for determinism (crucial for RQ4)


# ---------------------------------------------------------------------------
# Core perturbation — the single function injected into PCLA
# ---------------------------------------------------------------------------

def _dropout_probs(materials: np.ndarray, weather: str, cfg: PerturbConfig) -> np.ndarray:
    codes = _as_codes(materials)
    ratios = _ratio_table(weather)[codes.astype(np.intp)].astype(np.float32)
    p = cfg.dropout_gain * np.clip(1.0 - ratios, 0.0, 1.0)
    return np.clip(p, 0.0, cfg.dropout_cap)


def _intensity_factors(materials: np.ndarray, weather: str,
                       preserve_level: bool = True) -> np.ndarray:
    """Runtime term beta_m * alpha_m,c, rescaled to leave the scene level alone.

    beta is a RELATIVE quantity: it is normalised so its largest entry is 1, and
    it says a facade returns about twice what asphalt does. It is not an absolute
    transmittance. Multiplying every point by it therefore does two things at
    once - it imposes the material contrast, which is the point, and it darkens
    the whole cloud, which is not. On a typical urban mix the mean factor is 0.70,
    so the agent received a cloud at 70% of the unperturbed level before any
    material structure was applied.

    That made the three-way comparison unfair rather than merely noisy. In nominal
    weather alpha is 1 everywhere, so `global` scales by a mean ratio of exactly
    1.0 and drops gain*(1-alpha) = 0 points: it is bit-identical to `intensity`.
    The contrast being measured was thus two untouched modes against one dimmed by
    30%, and any nominal-weather difference was that dimming.

    Dividing by the point-weighted mean keeps every ratio between materials and
    removes the level shift: asphalt still falls below one and facades still rise
    above it, and the mean factor is one by construction.
    """
    codes = _as_codes(materials).astype(np.intp)
    f = (_BASE_BY_CODE[codes] * _ratio_table(weather)[codes]).astype(np.float32)
    if preserve_level and f.size:
        mean = float(np.mean(f))
        if mean > 1e-6:
            f = f / mean
    return f


def _weighted_weather_ratio(materials: np.ndarray, weather: str) -> float:
    if materials.shape[0] == 0:
        return 1.0
    # raw factors here: this is the alpha-only mean that `global` applies, and
    # normalising it would divide it out to exactly 1
    codes = _as_codes(materials).astype(np.intp)
    ratios = _intensity_factors(codes, weather, preserve_level=False) / np.maximum(
        _BASE_BY_CODE[codes].astype(np.float32), 1e-6)
    return float(np.mean(ratios))


def matsense_perturb(
    xyz: np.ndarray,
    intensity: np.ndarray,
    materials: np.ndarray,
    weather: str,
    mode: str,
    cfg: PerturbConfig = PerturbConfig(),
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (xyz', intensity') for one of the three evaluation modes.

    mode = 'standard'  -> raw CARLA output, untouched (baseline 1)
    mode = 'global'    -> uniform dropout + uniform intensity scale (baseline 2)
    mode = 'matsense'  -> material/weather dropout + jitter + intensity (proposed)

    xyz:        (N, 3) float32   — LiDAR xyz in sensor frame
    intensity:  (N,)   float32   — raw CARLA intensity [0, 1]
    materials:  (N,)   object    — material label strings (from sem-LiDAR matching)
    """
    rng = rng or np.random.default_rng(cfg.seed)
    n = xyz.shape[0]

    if mode == "standard":
        return xyz, intensity

    if mode == "global":
        keep = rng.random(n) >= cfg.global_dropout
        mean_ratio = _weighted_weather_ratio(materials, weather)
        return xyz[keep], intensity[keep] * mean_ratio

    if mode == "matsense":
        # 1) Per-material dropout — moves geometry-based planners (BEV occupancy)
        p_drop = _dropout_probs(materials, weather, cfg)
        keep = rng.random(n) >= p_drop
        xyz2 = xyz[keep].copy()
        inten2 = intensity[keep].copy()
        mat2 = materials[keep]

        if xyz2.shape[0] == 0:
            return xyz2, inten2

        # 2) Range jitter on strongly attenuated surfaces (wet asphalt, wet sidewalk)
        if cfg.jitter_gain > 0:
            dirs = xyz2 / np.maximum(np.linalg.norm(xyz2, axis=1, keepdims=True), 1e-6)
            jitter_scale = cfg.jitter_gain * _dropout_probs(mat2, weather, cfg)[:, None]
            xyz2 = xyz2 + dirs * (rng.standard_normal((xyz2.shape[0], 1)) * jitter_scale)

        # 3) Runtime intensity follows the paper form exactly: I' = beta_m * alpha_m,c * I_sim.
        inten2 = np.clip(inten2 * _intensity_factors(mat2, weather), 0.0, 1.0)
        return xyz2, inten2

    raise ValueError(f"Unknown mode '{mode}'. Expected: standard | global | matsense")


# ---------------------------------------------------------------------------
# Minimal thread-safe buffer (no pygame dep, same interface as main script)
# ---------------------------------------------------------------------------

class _SensorBuffer:
    def __init__(self, maxsize: int = 8):
        self._frames: OrderedDict[int, object] = OrderedDict()
        self._latest_frame: int | None = None
        self._latest_data = None
        self._lock = threading.Lock()
        self._maxsize = max(3, maxsize)

    def put(self, frame_id: int, data) -> None:
        with self._lock:
            self._frames[int(frame_id)] = data
            self._frames.move_to_end(int(frame_id))
            self._latest_frame = int(frame_id)
            self._latest_data = data
            while len(self._frames) > self._maxsize:
                self._frames.popitem(last=False)

    def get_latest(self):
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame, self._latest_data

    def get_at(self, frame_id: int):
        """The scan of a given tick, or the nearest one held.

        Pairing by frame is the point: the agent's cloud and the semantic scan
        must describe the same instant, or points captured while the ego was
        somewhere else have no partner within the matcher's tolerance and fall
        through to the default material. Returns the payload and how many frames
        away it was, so the caller can tell a match from a near miss.
        """
        with self._lock:
            if not self._frames:
                return None, None
            if frame_id in self._frames:
                return self._frames[frame_id], 0
            best = min(self._frames, key=lambda k: abs(k - frame_id))
            return self._frames[best], best - frame_id

    def get_recent(self, n: int = 2):
        """The last n payloads, newest first.

        The agent's LiDAR turns at 10 Hz while the world steps at 20, so one of
        its revolutions spans two ticks of ego motion while a semantic scan spans
        one. Matching against the newest scan alone leaves the older half of the
        revolution without a partner, and those points fall through to the
        default material. Matching against the union of the last few scans covers
        the whole revolution and does not depend on the two rates agreeing.
        """
        with self._lock:
            return list(self._frames.values())[-max(1, n):][::-1]


# ---------------------------------------------------------------------------
# Semantic dtype used by CARLA semantic LiDAR
# ---------------------------------------------------------------------------
SEMANTIC_LIDAR_DTYPE = np.dtype([
    ("x", np.float32), ("y", np.float32), ("z", np.float32),
    ("CosAngle", np.float32), ("ObjIdx", np.uint32), ("ObjTag", np.uint32),
])


# ---------------------------------------------------------------------------
# ClosedLoopSession — owns the co-located semantic LiDAR and perturbation state
# ---------------------------------------------------------------------------

class ClosedLoopSession:
    """Manages a co-located semantic LiDAR and exposes a perturb_fn for PCLA.

    Usage (with the forked PCLA)
    ----------------------------
        # 1. Spawn vehicle
        vehicle = world.spawn_actor(bp, spawn_point)
        world.tick()

        # 2. Build session BEFORE PCLA (session spawns the semantic LiDAR)
        session = ClosedLoopSession(vehicle, world, weather="rain", mode="matsense")

        # 3. Pass perturb_fn to the forked PCLA — it injects it into CallBack
        pcla = PCLA(agent_name, vehicle, route_xml, client,
                    perturb_fn=session.perturb_fn)

        # 4. Drive
        for _ in range(max_ticks):
            world.tick()
            control = pcla.get_action()
            vehicle.apply_control(control)

        # 5. Cleanup
        session.cleanup()
        pcla.cleanup()
    """

    def __init__(
        self,
        vehicle,              # carla.Vehicle — the ego
        world,                # carla.World
        weather: str = "nominal",
        mode: str = "matsense",
        cfg: PerturbConfig = PerturbConfig(),
    ):
        self.weather = weather
        self.mode = mode
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self._n_raw_last: int = 0
        self._n_pert_last: int = 0
        self._material_counts_last: dict[str, int] = {}
        self._global_mean_ratio_last: float = float("nan")
        self._sem_buf = _SensorBuffer()
        # Which device the agent will infer on. Nothing here selects it; the
        # point is that it appears in the log. A campaign that silently runs the
        # network on CPU is twenty times slower than one that does not, and the
        # only symptom is that it takes days, which is indistinguishable from
        # "simulation is heavy" unless someone writes the device down.
        try:
            import torch
            dev = (f"cuda ({torch.cuda.get_device_name(0)})"
                   if torch.cuda.is_available() else "CPU")
            print(f"[matsense] inferenza su {dev}", flush=True)
            print(f"[matsense] t: sessione pronta a +{_time.time()-_IMPORT_T:.1f}s", flush=True)
        except Exception as exc:
            print(f"[matsense] dispositivo non determinabile: {exc}", flush=True)

        # how many semantic scans to match against; see _SensorBuffer.get_recent
        self._sem_history = 3
        self._world = world
        self._vehicle = vehicle
        self._sem_lidar = self._spawn_semantic_lidar(world, vehicle)

    # What must agree with the agent's LiDAR is the GEOMETRY it observes: where
    # it sits and which vertical band it sweeps. The sampling need not agree, and
    # should not be copied: a semantic sensor that spins faster and denser than
    # the agent's covers a superset of its directions, which is what gives every
    # ray-cast return a partner to match against. Mirroring the agent's 10 Hz
    # against a 0.05 s world step makes each tick deliver half a revolution, and
    # measured that way the match rate was zero.
    #
    # The field of view is where the old value did the damage. The mounting was
    # (0, 0, 2.5) against LAV's (0, 0, 2.4), ten centimetres, well inside the
    # matcher's 0.35 m tolerance. But the band was 2/-24.8 against LAV's 10/-30,
    # so a third of the agent's channels swept angles the semantic sensor never
    # looked at and could not be labelled at all.
    #
    # z comes from the agent's own config (lav/config.yaml: camera_z), not from
    # the example in autonomous_agent's docstring, which describes a different
    # mounting and is what made an earlier attempt at this worse.
    _FALLBACK_LIDAR = dict(channels="64", range="85", points_per_second="1300000",
                           rotation_frequency="20", upper_fov="10", lower_fov="-30")
    _FALLBACK_POSE = (0.0, 0.0, 2.4)
    # How the AGENT's sensor is recognised among the several ray-cast LiDARs on
    # the ego. Deliberately separate from _FALLBACK_LIDAR: that describes the
    # semantic sensor we create, and tying the two together once made the
    # discriminator match the dataset recorder's sensor instead, which is the
    # opposite of what it is for.
    _AGENT_SIGNATURE = dict(points_per_second=600000, dropoff_general_rate=0.45)

    def _agent_lidar_spec(self, world, vehicle):
        """Mirror the LiDAR whose cloud is actually perturbed: the agent's.

        The semantic cloud is matched to the agent's cloud point by point, and a
        partner counts only within 0.35 m, so two sensors mounted apart agree
        almost nowhere. Measured on one scene: co-located, 98.6% of returns carry
        a usable tag; mounted at the transfuser pose while the agent was LAV,
        9.7%. The rest fall through to the declared fallback material, which is
        how a calibration can appear to be applied while barely being used.

        Discovery, not restatement, is what keeps the two aligned when the agent
        changes. But the ego carries MORE THAN ONE ray-cast LiDAR: the dataset
        recorder attaches its own, and taking the first one found picked that one
        and left 65.6% unknown. The agent's is identified by the sampling PCLA
        applies to it, so candidates are scored against that and a mismatch falls
        back rather than silently mirroring the wrong sensor.
        """
        import carla
        want_pps = int(self._AGENT_SIGNATURE["points_per_second"])
        best, best_score = None, -1
        seen = []
        for actor in world.get_actors().filter("sensor.lidar.ray_cast"):
            parent = actor.parent
            if parent is None or parent.id != vehicle.id:
                continue
            attrs = dict(actor.attributes)
            pps = int(float(attrs.get("points_per_second", 0)))
            score = 0
            if pps == want_pps:
                score += 4
            # PCLA sets these on the agent's sensor; the recorder does not
            if abs(float(attrs.get("dropoff_general_rate", 0.0))
                   - self._AGENT_SIGNATURE["dropoff_general_rate"]) < 1e-6:
                score += 2
            if abs(float(attrs.get("atmosphere_attenuation_rate", 0.0)) - 0.004) < 1e-9:
                score += 1
            seen.append((actor.id, pps, score))
            if score > best_score:
                best, best_score = actor, score
        if best is None or best_score < 6:
            print(f"[matsense] agent LiDAR not identified among {seen}; "
                  f"using the declared mounting", flush=True)
            return (dict(self._FALLBACK_LIDAR),
                    carla.Transform(carla.Location(*self._FALLBACK_POSE)), "fallback")

        attrs = dict(best.attributes)
        spec = dict(self._FALLBACK_LIDAR)
        for k in ("range", "upper_fov", "lower_fov"):   # geometry, not sampling
            if k in attrs:
                spec[k] = attrs[k]
        inv = np.array(vehicle.get_transform().get_inverse_matrix())
        rel = inv @ np.array(best.get_transform().get_matrix())
        loc = carla.Location(x=float(rel[0, 3]), y=float(rel[1, 3]), z=float(rel[2, 3]))
        yaw = math.degrees(math.atan2(rel[1, 0], rel[0, 0]))
        pitch = math.degrees(math.atan2(-rel[2, 0], math.hypot(rel[2, 1], rel[2, 2])))
        roll = math.degrees(math.atan2(rel[2, 1], rel[2, 2]))
        return spec, carla.Transform(loc, carla.Rotation(pitch=pitch, yaw=yaw, roll=roll)), "mirrored"

    def _spawn_semantic_lidar(self, world, vehicle):
        bp = world.get_blueprint_library().find("sensor.lidar.ray_cast_semantic")
        spec, transform, how = self._agent_lidar_spec(world, vehicle)
        for key, value in spec.items():
            if bp.has_attribute(key):
                bp.set_attribute(key, value)
        print(f"[matsense] semantic LiDAR {how} on the agent's: "
              f"pos=({transform.location.x:.2f}, {transform.location.y:.2f}, "
              f"{transform.location.z:.2f}) fov="
              f"{spec.get('upper_fov', '?')}/{spec.get('lower_fov', '?')} "
              f"pps={spec.get('points_per_second', '?')}", flush=True)
        sensor = world.spawn_actor(bp, transform, attach_to=vehicle)
        sensor.listen(self._sem_callback)
        return sensor

    def _sem_callback(self, measurement):
        data = np.frombuffer(measurement.raw_data, dtype=SEMANTIC_LIDAR_DTYPE).copy()
        self._sem_buf.put(measurement.frame, data)

    @property
    def perturb_fn(self):
        """Return the callable that PCLA's forked CallBack will invoke.

        Signature: (points: np.ndarray[N,4]) -> np.ndarray[M,4]
        This is passed as perturb_fn= to the PCLA constructor.
        """
        return self.get_perturbed_cloud

    def get_perturbed_cloud(self, raw_pts_n4: np.ndarray, frame: int | None = None) -> np.ndarray:
        """Called from inside PCLA's LiDAR callback (via CallBack._parse_lidar_cb).

        raw_pts_n4: (N, 4) float32  [x, y, z, intensity]
        Returns:    (M, 4) float32  with M <= N
        """
        self._n_raw_last = raw_pts_n4.shape[0]
        self._n_unmatched_last = 0
        self._n_unmapped_last = 0
        self._sem_lag_last = -999
        xyz = raw_pts_n4[:, :3]
        intensity = raw_pts_n4[:, 3]

        # One-shot, on the first cloud the agent actually delivers: by now PCLA
        # has created its sensors, which it had not when this session spawned the
        # semantic one. Everything upstream assumed where the agent's LiDAR sits;
        # this prints where it really is.
        if not getattr(self, "_probed", False):
            self._probed = True
            print(f"[matsense] t: primo cloud a +{_time.time()-_IMPORT_T:.1f}s", flush=True)
            try:
                for a in self._world.get_actors().filter("sensor.lidar.*"):
                    par = a.parent
                    if par is None or par.id != self._vehicle.id:
                        continue
                    t = a.get_transform(); vt = self._vehicle.get_transform()
                    at = dict(a.attributes)
                    print(f"[matsense probe] {a.type_id} dz={t.location.z - vt.location.z:+.2f} "
                          f"fov={at.get('upper_fov')}/{at.get('lower_fov')} "
                          f"pps={at.get('points_per_second')} rot={at.get('rotation_frequency')}",
                          flush=True)
            except Exception as exc:
                print(f"[matsense probe] fallita: {exc}", flush=True)

        if frame is not None:
            sem_data, lag = self._sem_buf.get_at(int(frame))
            self._sem_lag_last = 0 if lag is None else int(lag)
        else:
            recent = self._sem_buf.get_recent(self._sem_history)
            sem_data = (np.concatenate(recent) if len(recent) > 1
                        else (recent[0] if recent else None))
            self._sem_lag_last = -999          # no frame supplied by the caller
        tags = _match_semantic_tags(xyz, sem_data)
        # `unknown` hides two different failures and the paper claims they are
        # told apart, so tell them apart: tag 0 means the point found no partner
        # in the semantic cloud at all, a defect; any other tag absent from the
        # profile is a CARLA class we never calibrated, a gap in the taxonomy.
        # Conflating them is how a mounting error stayed invisible for a whole
        # campaign.
        self._n_unmatched_last = int((tags == 0).sum())
        # where the unmatched points are, not just how many: an angular cause
        # concentrates them far away, a temporal one spreads them evenly, and a
        # geometric one clusters them at a particular elevation. Accumulated
        # over the run and printed once at cleanup.
        try:
            rr = np.linalg.norm(xyz, axis=1)
            miss = tags == 0
            edges = np.array([0, 5, 10, 20, 30, 45, 60, 85, 1e9])
            hi = np.digitize(rr[miss], edges) - 1
            ha = np.digitize(rr, edges) - 1
            if not hasattr(self, "_miss_hist"):
                self._miss_hist = np.zeros(len(edges), dtype=np.int64)
                self._all_hist = np.zeros(len(edges), dtype=np.int64)
                self._hist_edges = edges
            np.add.at(self._miss_hist, hi, 1)
            np.add.at(self._all_hist, ha, 1)
            # elevation of the lost points: the road seen from 2.4 m at 10-20 m
            # sits between about -14 and -7 degrees, while poles, signs and
            # vegetation sit near or above the horizon. Which bin they fall in
            # says whether the residual is a surface or a set of objects.
            el = np.degrees(np.arcsin(np.clip(xyz[:, 2] / np.maximum(rr, 1e-6), -1, 1)))
            eedges = np.array([-40, -20, -14, -10, -7, -4, -1, 2, 12])
            ei = np.digitize(el[miss], eedges) - 1
            ea = np.digitize(el, eedges) - 1
            if not hasattr(self, "_el_miss"):
                self._el_miss = np.zeros(len(eedges), dtype=np.int64)
                self._el_all = np.zeros(len(eedges), dtype=np.int64)
                self._el_edges = eedges
            np.add.at(self._el_miss, np.clip(ei, 0, len(eedges) - 1), 1)
            np.add.at(self._el_all, np.clip(ea, 0, len(eedges) - 1), 1)
            # printed on a cadence rather than at cleanup, which this run path
            # never reaches
            self._hist_n = getattr(self, "_hist_n", 0) + 1
            if self._hist_n == 150:
                self.report_match_profile()
        except Exception:
            pass
        # A tag the profile does not map, counted without walking the array in
        # Python. Out-of-range tags were never in the dict either, so they count
        # as unmapped exactly as the membership test had it.
        t_arr = np.asarray(tags)
        in_range = (t_arr >= 0) & (t_arr < _MAX_TAG)
        mapped = np.zeros(t_arr.shape, dtype=bool)
        mapped[in_range] = _TAG_MAPPED[t_arr[in_range].astype(np.intp)]
        self._n_unmapped_last = int(np.count_nonzero((t_arr != 0) & ~mapped))

        # Codes once, then everything downstream is a gather. The string array
        # is still what matsense_perturb is documented to take, but it is no
        # longer rebuilt inside each helper.
        codes = _codes_from_tags(t_arr)
        counts = np.bincount(codes, minlength=len(_MATERIALS))
        self._material_counts_last = {
            _MATERIALS[i]: int(c) for i, c in enumerate(counts.tolist()) if c
        }
        self._global_mean_ratio_last = _weighted_weather_ratio(codes, self.weather)

        xyz2, inten2 = matsense_perturb(
            xyz, intensity, codes, self.weather, self.mode, self.cfg, self.rng,
        )
        result = np.column_stack([xyz2, inten2]).astype(np.float32)
        self._n_pert_last = result.shape[0]
        return result

    @property
    def last_drop_stats(self) -> dict:
        """n_raw, n_perturbed, drop_frac from the most recent callback invocation."""
        n_raw = self._n_raw_last
        n_pert = self._n_pert_last
        return {
            "n_raw": n_raw,
            "n_perturbed": n_pert,
            "drop_frac": round(1.0 - n_pert / max(n_raw, 1), 4),
            "global_mean_ratio": self._global_mean_ratio_last,
            "material_point_counts_json": json.dumps(self._material_counts_last, sort_keys=True),
            "n_unmatched": self._n_unmatched_last,
            "n_unmapped": self._n_unmapped_last,
            "sem_frame_lag": self._sem_lag_last,
        }

    def report_match_profile(self):
        if not hasattr(self, "_miss_hist"):
            return
        e = self._hist_edges
        print("[matsense] punti senza corrispondenza per fascia di distanza", flush=True)
        for i in range(len(e) - 1):
            tot = int(self._all_hist[i])
            if tot < 1000:
                continue
            print(f"    {e[i]:>5.0f}-{e[i+1]:<6.0f} m  {tot:>10,}  "
                  f"{self._miss_hist[i] / tot * 100:6.2f}% persi", flush=True)
        if hasattr(self, "_el_miss"):
            g = self._el_edges
            print("[matsense] e per elevazione (0 = orizzonte)", flush=True)
            for i in range(len(g) - 1):
                tot = int(self._el_all[i])
                if tot < 1000:
                    continue
                print(f"    {g[i]:>+4.0f}..{g[i+1]:<+4.0f} gradi  {tot:>10,}  "
                      f"{self._el_miss[i] / tot * 100:6.2f}% persi", flush=True)

    def cleanup(self):
        print(f"[matsense] t: chiusura a +{_time.time()-_IMPORT_T:.1f}s", flush=True)
        self.report_match_profile()
        if self._sem_lidar is not None and self._sem_lidar.is_alive:
            self._sem_lidar.stop()
            self._sem_lidar = None


# ---------------------------------------------------------------------------
# RQ4 metric logger
# ---------------------------------------------------------------------------

class BehaviourLogger:
    """One CSV row per world tick. Aggregate per-route offline (pandas/polars)."""

    FIELDS = [
        "frame", "t_s",
        "x", "y",
        "speed_mps", "throttle", "brake", "steer",
        "n_raw", "n_perturbed", "drop_frac",
        # the two halves of `unknown`, never to be summed back together
        "n_unmatched", "n_unmapped", "sem_frame_lag",
        "route_dev_m", "cross_track_error_m", "progress_m", "ttc_proxy_s",
        "obstacle_distance_m", "obstacle_in_sensor_range", "first_obstacle_in_range_progress_m",
        "hazard_obstacle_x", "hazard_obstacle_y", "hazard_obstacle_z",
        "mode", "weather", "seed",
        "replicate", "condition_type",
        "profile_name", "profile_version", "profile_config_path", "profile_config_sha256",
        "route_file", "route_file_hash",
        "global_mean_ratio", "material_point_counts_json",
        "carla_client_version", "carla_server_version",
    ]

    def __init__(self, out_csv: str | Path):
        self._f = open(out_csv, "w", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._f, fieldnames=self.FIELDS)
        self._w.writeheader()

    def log(self, **row):
        self._w.writerow({k: row.get(k, "") for k in self.FIELDS})

    def close(self):
        self._f.close()


# ---------------------------------------------------------------------------
# Route helper — converts your recorded trajectory JSON to PCLA's XML format
# ---------------------------------------------------------------------------

def trajectory_to_route_xml(
    trajectory_json: str | Path,
    out_xml: str | Path,
    town: str = "Town01",
) -> Path:
    """Convert a trajectory recorded by carla_pygame_lidar_dataset_recorder_friendly
    into a PCLA-compatible route XML.

    The JSON format expected is the one produced by --save-trajectory:
        [{"frame_id": N, "timestamp": T,
          "transform": {"location": {"x":…, "y":…, "z":…},
                        "rotation": {"yaw":…}}}, …]

    PCLA's route XML format:
        <routes>
          <route id="0" town="Town01">
            <waypoint x="…" y="…" z="…" />
            …
          </route>
        </routes>
    """
    import json as _json

    poses = _json.loads(Path(trajectory_json).read_text(encoding="utf-8"))
    out = Path(out_xml)
    lines = ['<routes>', f'  <route id="0" town="{town}">']
    for p in poses:
        loc = p["transform"]["location"]
        lines.append(f'    <waypoint x="{loc["x"]:.3f}" y="{loc["y"]:.3f}" z="{loc.get("z", 0.0):.3f}" />')
    lines += ['  </route>', '</routes>']
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Route deviation helper
# ---------------------------------------------------------------------------

def _route_deviation_m(ego_location, route_locs: list) -> float:
    """Min XY distance from ego to any route waypoint (metres)."""
    if not route_locs:
        return 0.0
    ex, ey = ego_location.x, ego_location.y
    return float(min(math.hypot(ex - r.x, ey - r.y) for r in route_locs))


def _ttc_proxy(world, vehicle, radius: float = 30.0) -> float:
    """Rough TTC: nearest lead-vehicle distance / relative closing speed."""
    import carla
    tf = vehicle.get_transform()
    fwd = tf.get_forward_vector()
    ev = vehicle.get_velocity()
    e_speed = math.sqrt(ev.x**2 + ev.y**2 + ev.z**2)

    best_ttc = float("inf")
    for actor in world.get_actors().filter("vehicle.*"):
        if actor.id == vehicle.id:
            continue
        ov = actor.get_velocity()
        ol = actor.get_location()
        dx, dy = ol.x - tf.location.x, ol.y - tf.location.y
        dist = math.sqrt(dx**2 + dy**2)
        if dist > radius:
            continue
        # Only actors ahead (dot-product filter)
        if fwd.x * dx + fwd.y * dy < 0:
            continue
        o_speed = math.sqrt(ov.x**2 + ov.y**2 + ov.z**2)
        closing = max(e_speed - o_speed, 0.1)
        best_ttc = min(best_ttc, dist / closing)
    return round(best_ttc, 2)


# ---------------------------------------------------------------------------
# Experiment matrix — full RQ4 sweep using the forked PCLA
# ---------------------------------------------------------------------------

def run_experiment_matrix(
    agent_name: str = "tfv4_l6_0",
    route_xml: str = "./route.xml",
    town: str = "Town02",
    spawn_index: int = 31,
    out_dir: str = "./clog_results",
    max_ticks: int = 2000,
    fixed_delta: float = 0.05,
    pcla_dir: str | None = None,
):
    """3 weathers × 3 modes = 9 runs, identical route + seed each time.

    Requires:
    - CARLA server running on localhost:2000
    - PCLA forked with the two changes in sensor_interface.py / PCLA.py
    - pcla_dir: path to the PCLA repo root (added to sys.path if given)
    """
    import carla

    if pcla_dir and pcla_dir not in sys.path:
        sys.path.insert(0, pcla_dir)
    from PCLA import PCLA  # noqa — from the forked repo

    weathers = ["nominal", "rain", "snow"]
    modes = ["standard", "global", "matsense"]
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    cfg = PerturbConfig()

    client = carla.Client("localhost", 2000)
    client.set_timeout(20.0)

    for weather in weathers:
        for mode in modes:
            print(f"\n=== {weather} / {mode} ===")
            out_csv = out_path / f"clog_{weather}_{mode}.csv"
            logger = BehaviourLogger(out_csv)

            client.load_world(town)
            world = client.get_world()

            settings = world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = fixed_delta
            world.apply_settings(settings)

            # Spawn ego
            bp_lib = world.get_blueprint_library()
            vehicle_bp = bp_lib.filter("model3")[0]
            spawn_point = world.get_map().get_spawn_points()[spawn_index]
            vehicle = world.spawn_actor(vehicle_bp, spawn_point)
            world.tick()

            # Session must be created BEFORE PCLA (it spawns the semantic LiDAR)
            session = ClosedLoopSession(
                vehicle, world, weather=weather, mode=mode, cfg=cfg,
            )

            # PCLA with the matsense hook injected via perturb_fn
            pcla = PCLA(agent_name, vehicle, route_xml, client,
                        perturb_fn=session.perturb_fn)

            # Parse route waypoints for deviation metric
            import xml.etree.ElementTree as _ET
            tree = _ET.parse(route_xml)
            route_locs = [
                type("L", (), {"x": float(wp.attrib["x"]), "y": float(wp.attrib["y"])})()
                for wp in tree.iter("waypoint")
            ]

            for frame in range(max_ticks):
                world.tick()
                try:
                    control = pcla.get_action()
                    vehicle.apply_control(control)
                except Exception as exc:
                    print(f"  Agent error at frame {frame}: {exc}")
                    break

                tf = vehicle.get_transform()
                vel = vehicle.get_velocity()
                speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
                drop = session.last_drop_stats
                logger.log(
                    frame=frame,
                    t_s=round(frame * fixed_delta, 3),
                    x=round(tf.location.x, 2),
                    y=round(tf.location.y, 2),
                    speed_mps=round(speed, 3),
                    throttle=round(control.throttle, 3),
                    brake=round(control.brake, 3),
                    steer=round(control.steer, 4),
                    n_raw=drop["n_raw"],
                    n_perturbed=drop["n_perturbed"],
                    drop_frac=drop["drop_frac"],
                    route_dev_m=round(_route_deviation_m(tf.location, route_locs), 2),
                    ttc_proxy_s=_ttc_proxy(world, vehicle),
                    mode=mode,
                    weather=weather,
                    seed=cfg.seed,
                )

            logger.close()
            session.cleanup()
            pcla.cleanup()

            settings.synchronous_mode = False
            world.apply_settings(settings)
            print(f"  Saved → {out_csv}")


if __name__ == "__main__":
    # Quick smoke-test: verify perturbation shapes and dropout rates
    rng = np.random.default_rng(0)
    N = 5000
    xyz_test = rng.standard_normal((N, 3)).astype(np.float32) * 20
    xyz_test[:, 2] = np.abs(xyz_test[:, 2]) * 0.1  # keep z close to ground
    inten_test = rng.random(N).astype(np.float32)
    mats_test = rng.choice(
        list(NOMINAL_BASE.keys()), size=N
    )

    cfg = PerturbConfig()
    for weather in ["nominal", "rain", "snow"]:
        for mode in ["standard", "global", "matsense"]:
            xyz_out, i_out = matsense_perturb(
                xyz_test, inten_test, mats_test, weather, mode, cfg
            )
            drop = 1.0 - len(xyz_out) / N
            print(f"{weather:8s} / {mode:10s}: {len(xyz_out):5d}/{N} pts "
                  f"(drop {drop:.1%}, intensity mean {i_out.mean():.3f})")
