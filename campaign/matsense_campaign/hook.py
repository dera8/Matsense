"""Runtime adapter: the perturbation function handed to PCLA.

PCLA (patched with tools/patch_pcla.py) calls `session.perturb(points, frame, tag)`
inside the LiDAR callback, after decoding the CARLA measurement and before the
agent's own preprocessing. Points are an (N, 4) float32 array (x, y, z, I).

The session owns a semantic LiDAR that mirrors the agent's LiDAR (same
attributes, same mounting), so that every return can be associated with a
CARLA tag by direction.
"""
from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

from .association import SEMANTIC_LIDAR_DTYPE, match_tags
from .operator import OperatorParams, apply_arm, class_index
from .profile import Profile

SEMANTIC_ATTRS = ("channels", "range", "points_per_second", "rotation_frequency",
                  "upper_fov", "lower_fov", "horizontal_fov")


def _euler_from_matrix(m: np.ndarray) -> tuple[float, float, float]:
    """(pitch, yaw, roll) in degrees from a CARLA 4x4 transform matrix."""
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, m[2, 0]))))
    yaw = math.degrees(math.atan2(m[1, 0], m[0, 0]))
    roll = math.degrees(math.atan2(-m[2, 1], m[2, 2]))
    return pitch, yaw, roll


def relative_transform(parent, child):
    """carla.Transform of `child` in the frame of `parent` (both actors)."""
    import carla
    mp = np.array(parent.get_transform().get_matrix())
    mc = np.array(child.get_transform().get_matrix())
    rel = np.linalg.inv(mp) @ mc
    pitch, yaw, roll = _euler_from_matrix(rel)
    return carla.Transform(carla.Location(x=float(rel[0, 3]), y=float(rel[1, 3]), z=float(rel[2, 3])),
                           carla.Rotation(pitch=pitch, yaw=yaw, roll=roll))


def mirror_attributes(lidar_attrs: dict, fps: float) -> dict:
    """Semantic-LiDAR attributes that cover every ray of the agent's LiDAR.

    With rotation_frequency < fps the agent LiDAR returns only part of a turn per
    frame, and which part depends on the tick it was spawned in: a mirror spawned
    one tick later can scan the opposite half. The mirror therefore does a full
    turn every frame (rotation_frequency = fps) with points_per_second scaled so
    that its angular resolution equals the agent's.
    """
    attrs = {k: str(lidar_attrs[k]) for k in SEMANTIC_ATTRS if k in lidar_attrs}
    rot = float(attrs.get("rotation_frequency", fps))
    if rot < fps and "points_per_second" in attrs:
        attrs["points_per_second"] = str(int(round(float(attrs["points_per_second"]) * fps / rot)))
        attrs["rotation_frequency"] = str(fps)
    return attrs


def find_agent_lidars(world, ego) -> list:
    out = []
    for a in world.get_actors().filter("sensor.lidar.ray_cast"):
        if a.parent is not None and a.parent.id == ego.id:
            out.append(a)
    return out


class SensingSession:
    def __init__(self, world, ego, profile: Profile, condition: str, arm: str,
                 params: OperatorParams, seed: int, tol_deg: float = 0.40,
                 class_shuffled_profile: Profile | None = None,
                 dump_dir: Path | None = None, dump_every: int = 20,
                 wait_semantic_ms: float = 30.0):
        profile.check_condition(condition)
        self.world, self.ego = world, ego
        self.profile, self.condition, self.arm, self.params = profile, condition, arm, params
        self.tol_deg = tol_deg
        self.class_shuffled_profile = class_shuffled_profile
        self.rng = np.random.default_rng(seed)
        self.shuffle_rng = np.random.default_rng([seed, 7919])
        self.dump_dir, self.dump_every = dump_dir, dump_every
        self.wait_s = wait_semantic_ms / 1000.0
        self._sem = OrderedDict()
        self._sem_cv = threading.Condition()
        self._lock = threading.Lock()
        self._sensors = []
        self.agent_lidar_ids: list[int] = []
        self.last_stats: dict = {}
        self.n_calls = 0
        self.mirrored_attributes: dict = {}

    # -- semantic LiDAR -----------------------------------------------------
    def attach(self) -> None:
        """Mirror every LiDAR the agent attached to the ego. Call after PCLA init."""
        import carla
        lidars = find_agent_lidars(self.world, self.ego)
        if not lidars:
            raise RuntimeError("no sensor.lidar.ray_cast attached to the ego: "
                               "was PCLA initialised before attach()?")
        bp_lib = self.world.get_blueprint_library()
        dt = self.world.get_settings().fixed_delta_seconds
        if not dt:
            raise RuntimeError("fixed_delta_seconds is not set: the semantic mirror needs synchronous mode")
        fps = round(1.0 / dt, 6)
        for lid in lidars:
            bp = bp_lib.find("sensor.lidar.ray_cast_semantic")
            attrs = {}
            for k, v in mirror_attributes(lid.attributes, fps).items():
                if bp.has_attribute(k):
                    bp.set_attribute(k, v)
                    attrs[k] = v
            attrs["agent_rotation_frequency"] = lid.attributes.get("rotation_frequency")
            attrs["agent_points_per_second"] = lid.attributes.get("points_per_second")
            tf = relative_transform(self.ego, lid)
            sem = self.world.spawn_actor(bp, tf, attach_to=self.ego)
            sem.listen(self._on_semantic)
            self._sensors.append(sem)
            self.agent_lidar_ids.append(lid.id)
            self.mirrored_attributes[str(lid.id)] = {
                **attrs, "x": tf.location.x, "y": tf.location.y, "z": tf.location.z,
                "pitch": tf.rotation.pitch, "yaw": tf.rotation.yaw, "roll": tf.rotation.roll}
        if len(lidars) > 1:
            print(f"[matsense] WARNING: the agent has {len(lidars)} LiDARs; the semantic "
                  f"buffer is shared, check that their mountings coincide.", flush=True)

    def _on_semantic(self, meas) -> None:
        data = np.frombuffer(meas.raw_data, dtype=SEMANTIC_LIDAR_DTYPE).copy()
        with self._sem_cv:
            self._sem[int(meas.frame)] = data
            while len(self._sem) > 10:
                self._sem.popitem(last=False)
            self._sem_cv.notify_all()

    def _semantic_for(self, frame: int | None):
        deadline = time.monotonic() + self.wait_s
        with self._sem_cv:
            while True:
                if frame is not None and frame in self._sem:
                    return self._sem[frame], 0
                if frame is None and self._sem:
                    k = next(reversed(self._sem))
                    return self._sem[k], None
                left = deadline - time.monotonic()
                if left <= 0:
                    break
                self._sem_cv.wait(left)
            if not self._sem:
                return None, None
            k = next(reversed(self._sem))
            return self._sem[k], (None if frame is None else int(frame) - k)

    # -- the hook -----------------------------------------------------------
    def perturb(self, points: np.ndarray, frame: int | None = None, tag=None) -> np.ndarray:
        t0 = time.perf_counter()
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 4)
        xyz, inten = pts[:, :3], pts[:, 3]
        sem, lag = self._semantic_for(frame)
        tags, matched = match_tags(xyz, sem, self.tol_deg)
        cls = class_index(tags, matched, self.profile)
        with self._lock:
            xyz2, inten2, keep, stats = apply_arm(
                self.arm, xyz, inten, cls, self.condition, self.profile, self.params,
                self.rng, self.shuffle_rng, self.class_shuffled_profile)
            self.n_calls += 1
            stats.update(sem_lag_frames=lag, lidar_frame=frame,
                         hook_ms=round(1000 * (time.perf_counter() - t0), 2))
            self.last_stats = stats
            if self.dump_dir is not None and self.n_calls % self.dump_every == 1:
                np.savez_compressed(
                    Path(self.dump_dir) / f"points_{self.n_calls:06d}.npz",
                    xyz_in=xyz, intensity_in=inten, cls=cls, keep=keep,
                    xyz_out=xyz2, intensity_out=inten2)
        return np.column_stack([xyz2, inten2]).astype(np.float32)

    def cleanup(self) -> None:
        for s in self._sensors:
            try:
                if s.is_alive:
                    s.stop()
                    s.destroy()
            except Exception:
                pass
        self._sensors = []
