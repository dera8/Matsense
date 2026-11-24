"""Scripted hazard scenarios, modelled on the CARLA Leaderboard catalogue.

Two stages:

1. resolve (tools/resolve_scenarios.py, once, CARLA running): search the map for a
   location that fits each scenario, build the ego route and the hazard paths, and
   freeze them in config/scenarios_resolved.json. Every run then uses exactly the
   same coordinates.
2. run (ScenarioRuntime): spawn the hazard actors with physics disabled and move
   them kinematically along their frozen paths when their trigger fires. Kinematic
   hazards remove one source of run-to-run variability: only the ego and the agent
   are nondeterministic.

Scenario types (IDs follow the manuscript):

    15 invading_bend   Town02  oncoming vehicle drifts into the ego lane in a bend
    16 lead_brake      Town02  lead vehicle brakes to a scripted stop
    05 red_light       Town03  crossing vehicle runs a red light at a junction
    09 static_cutin    Town03  vehicle cuts in from a lane of stopped traffic
    18 vru_parked      Town03  cyclist emerges from behind a parked vehicle, stops in lane
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Geometry helpers (CARLA objects only inside functions that need them)
# ---------------------------------------------------------------------------


def _yaw_diff(a: float, b: float) -> float:
    d = (b - a + 180.0) % 360.0 - 180.0
    return d


def _loc(wp):
    l = wp.transform.location
    return [round(l.x, 3), round(l.y, 3), round(l.z, 3)]


def _pose(wp, lateral: float = 0.0):
    """[x, y, z, yaw] of a waypoint, shifted `lateral` m to the right of the lane."""
    t = wp.transform
    r = t.get_right_vector()
    return [round(t.location.x + lateral * r.x, 3), round(t.location.y + lateral * r.y, 3),
            round(t.location.z, 3), round(t.rotation.yaw, 2)]


def follow_lane(wp, length: float, step: float = 2.0):
    """Waypoints along the lane for `length` m, preferring the straightest branch."""
    out, s = [wp], 0.0
    while s < length:
        nxt = wp.next(step)
        if not nxt:
            break
        wp = min(nxt, key=lambda w: abs(_yaw_diff(out[-1].transform.rotation.yaw,
                                                   w.transform.rotation.yaw)))
        out.append(wp)
        s += step
    return out


def follow_lane_back(wp, length: float, step: float = 2.0):
    out, s = [wp], 0.0
    while s < length:
        prv = wp.previous(step)
        if not prv:
            break
        wp = min(prv, key=lambda w: abs(_yaw_diff(out[-1].transform.rotation.yaw,
                                                  w.transform.rotation.yaw)))
        out.append(wp)
        s += step
    return out[::-1]


def path_yaw_change(path) -> float:
    return sum(abs(_yaw_diff(a.transform.rotation.yaw, b.transform.rotation.yaw))
               for a, b in zip(path[:-1], path[1:]))


def no_junction(path) -> bool:
    return not any(w.is_junction for w in path)


def same_dir_neighbour(wp):
    for n in (wp.get_left_lane(), wp.get_right_lane()):
        if n is not None and str(n.lane_type).endswith("Driving") and \
                np.sign(n.lane_id) == np.sign(wp.lane_id) and n.road_id == wp.road_id:
            return n
    return None


def opposite_neighbour(wp):
    n = wp.get_left_lane()
    if n is not None and str(n.lane_type).endswith("Driving") and \
            np.sign(n.lane_id) != np.sign(wp.lane_id):
        return n
    return None


def is_rightmost(wp) -> bool:
    r = wp.get_right_lane()
    return r is None or not str(r.lane_type).endswith("Driving")


def candidate_starts(cmap, step: float = 2.0):
    """Deterministic list of driving waypoints, sorted by (road, lane, s)."""
    wps = [w for w in cmap.generate_waypoints(step) if not w.is_junction]
    wps.sort(key=lambda w: (w.road_id, w.lane_id, round(w.s, 1)))
    return wps


def sparse_route(path, every: float = 20.0, step: float = 2.0):
    k = max(1, int(round(every / step)))
    # No route point inside a junction: the route planner snaps it to whichever
    # junction lane is nearest (often a turning one) and plans a detour loop.
    pts = [path[i] for i in range(0, len(path), k) if i == 0 or not path[i].is_junction]
    if pts[-1] is not path[-1]:
        pts.append(path[-1])
    return [_pose(w) for w in pts]


def dense(path):
    return [_pose(w) for w in path]


def _offset_path(path, lateral: float):
    return [_pose(w, lateral) for w in path]


# ---------------------------------------------------------------------------
# Resolution: one function per scenario type
# ---------------------------------------------------------------------------

BIKE_BP = "vehicle.diamondback.century"
CAR_BP = "vehicle.audi.a2"
LEAD_BP = "vehicle.lincoln.mkz_2020"


def resolve_lead_brake(cmap, world, p):
    L = p.get("length_m", 140.0)
    for wp in candidate_starts(cmap):
        path = follow_lane(wp, L)
        if len(path) * 2.0 < L or not no_junction(path) or path_yaw_change(path) > 10:
            continue
        s_lead, s_stop = int(p.get("lead_start_m", 25) / 2), int(p.get("lead_stop_m", 95) / 2)
        lead_path = dense(path[s_lead:s_stop + 1])
        yield {
            "ego_path": dense(path), "route": sparse_route(path),
            "actors": [{"role": "lead", "blueprint": LEAD_BP, "path": lead_path,
                        "speed": p.get("lead_speed", 6.0), "decel": p.get("lead_decel", 3.5),
                        "trigger": {"type": "time", "t": p.get("trigger_t", 1.0)}}],
            "hazard_stop": lead_path[-1][:2],
        }


def resolve_invading_bend(cmap, world, p):
    L = p.get("length_m", 160.0)
    for wp in candidate_starts(cmap):
        path = follow_lane(wp, L)
        if len(path) * 2.0 < L or not no_junction(path):
            continue
        if path_yaw_change(path[:25]) > 10:           # straight first 50 m
            continue
        turn = path_yaw_change(path[30:55])            # bend between 60 and 110 m
        if turn < p.get("min_turn_deg", 45):
            continue
        opp = [opposite_neighbour(w) for w in path[20:75]]
        if any(o is None for o in opp):
            continue
        apex = path[42]
        inv = p.get("invasion_m", 1.5)
        # The oncoming vehicle drives the opposite lane towards the ego, shifted
        # towards the ego lane (its own left, i.e. negative lateral offset).
        opp_path = [_pose(o, -inv) for o in opp[::-1]]
        yield {
            "ego_path": dense(path), "route": sparse_route(path),
            "actors": [{"role": "oncoming", "blueprint": CAR_BP, "path": opp_path,
                        "speed": p.get("oncoming_speed", 7.0),
                        "trigger": {"type": "ego_near", "point": _loc(apex)[:2],
                                    "distance": p.get("trigger_m", 55.0)}}],
            "hazard_stop": None,
        }


def resolve_static_cutin(cmap, world, p):
    L = p.get("length_m", 140.0)
    for wp in candidate_starts(cmap):
        path = follow_lane(wp, L)
        if len(path) * 2.0 < L or not no_junction(path) or path_yaw_change(path) > 10:
            continue
        side = [same_dir_neighbour(w) for w in path]
        if any(s is None for s in side[15:60]):
            continue
        actors = []
        for k, s_m in enumerate(p.get("static_at_m", [45, 53, 69])):
            i = int(s_m / 2)
            actors.append({"role": f"static_{k}", "blueprint": CAR_BP,
                           "path": [_pose(side[i])], "speed": 0.0, "trigger": None})
        i0 = int(p.get("cutin_at_m", 61) / 2)
        # lane change over ~16 m, then continue in the ego lane
        lc = []
        n_lc = 8
        for j in range(n_lc + 1):
            a, b = _pose(side[i0 + j]), _pose(path[i0 + j])
            w = j / n_lc
            lc.append([a[0] + w * (b[0] - a[0]), a[1] + w * (b[1] - a[1]), b[2], b[3]])
        # ...to the end of the ego route and 40 m beyond, so that it does not park in
        # front of the ego and end every run as 'stationary'.
        lc += dense(path[i0 + n_lc + 1:] + follow_lane(path[-1], 40.0)[1:])
        actors.append({"role": "cutin", "blueprint": CAR_BP, "path": lc,
                       "speed": p.get("cutin_speed", 4.0),
                       "trigger": {"type": "ego_near", "point": _loc(path[i0])[:2],
                                   "distance": p.get("trigger_m", 22.0)}})
        yield {"ego_path": dense(path), "route": sparse_route(path),
                "actors": actors, "hazard_stop": None}


def resolve_vru_parked(cmap, world, p):
    L = p.get("length_m", 120.0)
    for wp in candidate_starts(cmap):
        if not is_rightmost(wp):
            continue
        path = follow_lane(wp, L)
        if len(path) * 2.0 < L or not no_junction(path) or path_yaw_change(path) > 10:
            continue
        if not all(is_rightmost(w) for w in path):
            continue
        half = wp.lane_width / 2.0
        ip, ib = int(p.get("parked_at_m", 55) / 2), int(p.get("bike_at_m", 58) / 2)
        start = _pose(path[ib], half + 2.2)
        stop = _pose(path[ib], p.get("bike_stop_lateral", 0.3))
        n = 10
        bike_path = [[start[0] + (stop[0] - start[0]) * j / n, start[1] + (stop[1] - start[1]) * j / n,
                      start[2], (start[3] - 90.0) % 360.0] for j in range(n + 1)]
        yield {
            "ego_path": dense(path), "route": sparse_route(path),
            "actors": [
                {"role": "parked", "blueprint": CAR_BP, "path": [_pose(path[ip], half + 0.9)],
                 "speed": 0.0, "trigger": None},
                {"role": "cyclist", "blueprint": BIKE_BP, "path": bike_path,
                 "speed": p.get("bike_speed", 2.5), "decel": 3.0,
                 "trigger": {"type": "ego_near", "point": _loc(path[ib])[:2],
                             "distance": p.get("trigger_m", 20.0)}}],
            "hazard_stop": stop[:2],
        }


def _seg_intersection(p1, p2, q1, q2):
    p1, p2, q1, q2 = map(np.asarray, (p1, p2, q1, q2))
    r, s = p2 - p1, q2 - q1
    den = r[0] * s[1] - r[1] * s[0]
    if abs(den) < 1e-9:
        return None
    t = ((q1 - p1)[0] * s[1] - (q1 - p1)[1] * s[0]) / den
    u = ((q1 - p1)[0] * r[1] - (q1 - p1)[1] * r[0]) / den
    if 0 <= t <= 1 and 0 <= u <= 1:
        return p1 + t * r
    return None


def _through(entry_wp, step=2.0):
    """Straightest path through the junction that `entry_wp` leads into."""
    path = follow_lane(entry_wp, 40.0, step)
    inside = [w for w in path if w.is_junction]
    return path, inside


def resolve_red_light(cmap, world, p):
    entries = []
    for w in candidate_starts(cmap):
        nxt = w.next(2.0)
        if nxt and any(n.is_junction for n in nxt):
            path, inside = _through(w)
            if inside and abs(_yaw_diff(w.transform.rotation.yaw,
                                        path[-1].transform.rotation.yaw)) < 15:
                entries.append((w, path, inside))
    for ego_entry, ego_thru, ego_in in entries:
        jid = ego_in[0].get_junction().id
        approach = follow_lane_back(ego_entry, p.get("ego_approach_m", 70.0))
        if len(approach) * 2.0 < p.get("ego_approach_m", 70.0) or not no_junction(approach[:-1]):
            continue
        try:
            lights = world.get_traffic_lights_from_waypoint(ego_entry, 30.0)
        except Exception:
            lights = []
        if p.get("require_light", True) and not lights:
            continue
        for x_entry, x_thru, x_in in entries:
            if x_in[0].get_junction().id != jid:
                continue
            ang = abs(_yaw_diff(ego_entry.transform.rotation.yaw, x_entry.transform.rotation.yaw))
            if not 70 <= ang <= 110:
                continue
            conflict = None
            for a, b in zip(ego_thru[:-1], ego_thru[1:]):
                for c, d in zip(x_thru[:-1], x_thru[1:]):
                    conflict = _seg_intersection(_loc(a)[:2], _loc(b)[:2], _loc(c)[:2], _loc(d)[:2])
                    if conflict is not None:
                        break
                if conflict is not None:
                    break
            if conflict is None:
                continue
            x_back = follow_lane_back(x_entry, p.get("crossing_start_m", 36.0))
            ego_path = approach[:-1] + ego_thru + follow_lane(ego_thru[-1], 30.0)[1:]
            x_path = x_back[:-1] + x_thru + follow_lane(x_thru[-1], 30.0)[1:]
            try:
                x_lights = world.get_traffic_lights_from_waypoint(x_entry, 30.0)
            except Exception:
                x_lights = []
            yield {
                "ego_path": dense(ego_path), "route": sparse_route(ego_path),
                "actors": [{"role": "red_runner", "blueprint": CAR_BP, "path": dense(x_path),
                            "speed": p.get("crossing_speed", 9.0),
                            "trigger": {"type": "ego_near", "point": [float(conflict[0]), float(conflict[1])],
                                        "distance": p.get("trigger_m", 28.0)}}],
                "hazard_stop": None,
                "conflict_point": [float(conflict[0]), float(conflict[1])],
                # positions, not ids: actor ids change when the world is reloaded
                "lights_green": [[round(l.get_location().x, 2), round(l.get_location().y, 2)] for l in lights],
                "lights_red": [[round(l.get_location().x, 2), round(l.get_location().y, 2)] for l in x_lights],
            }


RESOLVERS = {
    "lead_brake": resolve_lead_brake,
    "invading_bend": resolve_invading_bend,
    "static_cutin": resolve_static_cutin,
    "vru_parked": resolve_vru_parked,
    "red_light": resolve_red_light,
}


def resolve(kind: str, cmap, world, params: dict) -> dict:
    """Return the `candidate_index`-th location that fits `kind`. Candidates whose
    ego start lies within `min_spacing_m` of an earlier one are skipped, so
    successive indices are genuinely different places."""
    want = int(params.get("candidate_index", 0))
    spacing = float(params.get("min_spacing_m", 80.0))
    seen = []
    for inst in RESOLVERS[kind](cmap, world, params):
        x, y = inst["ego_path"][0][:2]
        if any(math.hypot(x - a, y - b) < spacing for a, b in seen):
            continue
        if len(seen) == want:
            return inst
        seen.append((x, y))
    raise RuntimeError(f"{kind}: only {len(seen)} candidate location(s) on this map, "
                       f"candidate_index={want} requested")


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------

@dataclass
class _Mover:
    actor: object
    path: np.ndarray          # (K, 4) x, y, z, yaw
    cum: np.ndarray           # arc length at each vertex
    speed: float
    decel: float | None
    trigger: dict | None
    started: bool = False
    s: float = 0.0
    v: float = 0.0


class ScenarioRuntime:
    def __init__(self, world, instance: dict):
        self.world = world
        self.inst = instance
        self.movers: list[_Mover] = []
        self.t = 0.0

    def setup(self):
        import carla
        bp_lib = self.world.get_blueprint_library()
        for i, a in enumerate(self.inst["actors"]):
            bp = bp_lib.find(a["blueprint"])
            if bp.has_attribute("role_name"):
                bp.set_attribute("role_name", a["role"])
            x, y, z, yaw = a["path"][0]
            tf = carla.Transform(carla.Location(x=x, y=y, z=z + 0.05), carla.Rotation(yaw=yaw))
            actor = self.world.try_spawn_actor(bp, tf)
            if actor is None:
                raise RuntimeError(f"cannot spawn {a['role']} at {a['path'][0]}")
            actor.set_simulate_physics(False)
            path = np.asarray(a["path"], dtype=np.float64)
            seg = np.hypot(np.diff(path[:, 0]), np.diff(path[:, 1])) if len(path) > 1 else np.zeros(0)
            cum = np.concatenate([[0.0], np.cumsum(seg)])
            self.movers.append(_Mover(actor, path, cum, float(a.get("speed", 0.0)),
                                      a.get("decel"), a.get("trigger")))
        for xy in self.inst.get("lights_green", []):
            self._set_light(xy, "Green")
        for xy in self.inst.get("lights_red", []):
            self._set_light(xy, "Red")

    def _set_light(self, xy, state):
        import carla
        lights = list(self.world.get_actors().filter("traffic.traffic_light*"))
        if not lights:
            return
        tl = min(lights, key=lambda a: math.hypot(a.get_location().x - xy[0], a.get_location().y - xy[1]))
        if math.hypot(tl.get_location().x - xy[0], tl.get_location().y - xy[1]) > 1.0:
            raise RuntimeError(f"traffic light expected at {xy} not found")
        tl.set_state(getattr(carla.TrafficLightState, state))
        tl.freeze(True)

    def _fired(self, m: _Mover, ego_xy) -> bool:
        tr = m.trigger
        if tr is None:
            return False
        if tr["type"] == "time":
            return self.t >= tr["t"]
        if tr["type"] == "ego_near":
            px, py = tr["point"]
            return math.hypot(ego_xy[0] - px, ego_xy[1] - py) <= tr["distance"]
        raise ValueError(tr)

    def step(self, dt: float, ego_xy) -> None:
        import carla
        self.t += dt
        for m in self.movers:
            if m.speed <= 0 or len(m.path) < 2:
                continue
            if not m.started and self._fired(m, ego_xy):
                m.started, m.v = True, m.speed
            if not m.started:
                continue
            remaining = m.cum[-1] - m.s
            if m.decel and remaining <= m.v * m.v / (2 * m.decel) + 1e-6:
                m.v = max(0.0, m.v - m.decel * dt)
            m.s = min(m.cum[-1], m.s + m.v * dt)
            k = int(np.searchsorted(m.cum, m.s, side="right") - 1)
            k = min(k, len(m.path) - 2)
            w = (m.s - m.cum[k]) / max(m.cum[k + 1] - m.cum[k], 1e-6)
            p = m.path[k] + w * (m.path[k + 1] - m.path[k])
            yaw = m.path[k][3] + w * _yaw_diff(m.path[k][3], m.path[k + 1][3])
            m.actor.set_transform(carla.Transform(carla.Location(x=p[0], y=p[1], z=p[2] + 0.05),
                                                  carla.Rotation(yaw=yaw)))

    def hazard_positions(self):
        out = {}
        for m, a in zip(self.movers, self.inst["actors"]):
            l = m.actor.get_location()
            out[a["role"]] = (round(l.x, 3), round(l.y, 3))
        return out

    def cleanup(self):
        for m in self.movers:
            try:
                m.actor.destroy()
            except Exception:
                pass
        self.movers = []
