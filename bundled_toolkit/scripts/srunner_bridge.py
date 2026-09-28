#!/usr/bin/env python3
"""Run an official ScenarioRunner scenario inside this harness's own tick loop.

The five hazards this campaign uses were rebuilt by hand as actor JSON, and each
one turns out to be a hand-made copy of a scenario that already exists in
ScenarioRunner's catalogue, derived from the NHTSA pre-crash typology and used by
the CARLA Leaderboard. Rebuilding them cost provenance a reviewer will ask about,
and in one case cost correctness too: the hand-made VRU-behind-a-parked-vehicle
scenario uses a bicycle, whose semantic tag lands in the measured `car` class
while its rider falls to `unknown`, where the official ParkingCrossingPedestrian
uses a pedestrian.

The catalogue is not wired in because PCLA hands its RouteIndexer an empty
scenario file and the srunner import in route_indexer.py is commented out, so
route parsing happens and scenario construction never does. This bridges that
gap without adopting ScenarioRunner's own runner: the scenario object is built
directly, and its behaviour tree is ticked from the loop that already owns the
world tick and the sensor capture. Nothing here takes the tick away from the
recorder, because the recorder's synchronous capture depends on owning it.

What this does NOT give you is a calibrated scenario. The official ones are
parameterised by distance rather than by warning time - `distance` defaults to
100 m, the hazard is placed there and released when the ego arrives - so whether
inaction leads to a collision still depends on how fast the ego drives, which a
learned agent decides for itself. The same two checks apply to these as to the
hand-made ones: tools/check_scenario_would_collide.py and the warning budget in
tools/retime_scenario_hazards.py, tuning `distance` and `offset` here instead of
the release point.

  from srunner_bridge import ScenarioBridge, available
  bridge = ScenarioBridge(world, client, ego, "parking_crossing_pedestrian",
                          {"distance": "12"}, tm_port=8000)
  ...
  world.tick()
  bridge.tick()          # after the world tick, never before
  ...
  bridge.cleanup()
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Where ScenarioRunner and CARLA's own agents package live. srunner imports
# `agents.navigation`, which ships with the simulator's PythonAPI rather than
# with srunner, so both have to be on the path or every scenario import fails
# with a bare ModuleNotFoundError naming `agents`.
DEFAULT_PATHS = (
    os.environ.get("SCENARIO_RUNNER_ROOT", "/home/debora/carla/scenario_runner"),
    os.environ.get("CARLA_PYTHONAPI_ROOT", "/home/debora/carla/PythonAPI/carla"),
)

_CATALOGUE = None


def catalogue(extra=()):
    """Every scenario class ScenarioRunner defines, by the name routes use.

    Built by scanning the package rather than listed by hand. The route files
    name scenarios by their class name (`DynamicObjectCrossing`,
    `HardBreakRoute`), so a hand-written table has to be kept in step with 35
    modules and silently fails to resolve a type the moment it drifts; scanning
    cannot drift. Sixty-three classes resolve this way, which is more than the
    route files use.
    """
    global _CATALOGUE
    if _CATALOGUE is not None:
        return _CATALOGUE
    _ensure_paths(extra)
    import importlib
    import inspect
    import pkgutil
    import srunner.scenarios as pkg
    found = {}
    for m in pkgutil.iter_modules(pkg.__path__):
        try:
            mod = importlib.import_module(f"srunner.scenarios.{m.name}")
        except Exception:
            continue                    # a module that will not import cannot
        for name, cls in inspect.getmembers(mod, inspect.isclass):
            if cls.__module__ == mod.__name__:
                found.setdefault(name, cls)
    _CATALOGUE = found
    return found


def _ensure_paths(extra=()):
    for p in list(extra) + list(DEFAULT_PATHS):
        if p and Path(p).is_dir() and p not in sys.path:
            sys.path.insert(0, p)


def available(extra=()) -> bool:
    """Whether the catalogue can be imported at all, without raising."""
    _ensure_paths(extra)
    try:
        import srunner  # noqa: F401
        from srunner.scenariomanager.carla_data_provider import (  # noqa: F401
            CarlaDataProvider)
        return True
    except Exception:
        return False


def catalogue_names():
    return sorted(catalogue())


class ScenarioBridge:
    """One official scenario, built against a live world and ticked by hand."""

    def __init__(self, world, client, ego, name, params=None, *,
                 trigger=None, tm_port=8000, extra_paths=(), criteria=False,
                 timeout=180.0):
        _ensure_paths(extra_paths)
        cat = catalogue(extra_paths)
        if name not in cat:
            raise ValueError(f"scenario sconosciuto {name!r}; ScenarioRunner ne "
                             f"definisce {len(cat)}")
        from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
        from srunner.scenarioconfigs.scenario_configuration import (
            ScenarioConfiguration, ActorConfigurationData)

        self._cdp = CarlaDataProvider
        self._world = world
        self._name = name
        self._ticks = 0

        self._cdp.set_client(client)
        self._cdp.set_world(world)
        self._cdp.set_traffic_manager_port(int(tm_port))
        # The ego was spawned by the recorder, so the provider has never seen it
        # and every scenario that asks it for the ego's location logs "Actor not
        # found" once per tick and behaves as if the ego were nowhere.
        try:
            self._cdp.register_actor(ego, ego.get_transform())
        except Exception:
            pass                      # already registered is not an error here

        tf = trigger if trigger is not None else ego.get_transform()
        cfg = ScenarioConfiguration()
        cfg.name = name
        cfg.town = world.get_map().name
        cfg.trigger_points = [tf]
        cfg.ego_vehicles = [ActorConfigurationData(
            ego.type_id, tf, rolename="hero")]
        cfg.other_actors = []
        cfg.route_var_name = None
        cfg.weather = world.get_weather()
        # get_value_parameter reads {name: {'value': str}}, so a plain mapping of
        # strings is what the scenarios expect; anything else silently falls back
        # to the default and the setting appears to have been ignored
        cfg.other_parameters = {k: {"value": str(v)}
                                for k, v in (params or {}).items()}
        self._config = cfg

        self.scenario = cat[name](world, [ego], cfg, criteria_enable=criteria,
                                  timeout=timeout)
        self.tree = self.scenario.scenario_tree

    @property
    def actors(self):
        return list(getattr(self.scenario, "other_actors", []) or [])

    @property
    def done(self) -> bool:
        import py_trees
        return self.tree.status in (py_trees.common.Status.SUCCESS,
                                    py_trees.common.Status.FAILURE)

    def tick(self):
        """Advance the scenario one step. Call AFTER the world tick.

        The provider caches positions and velocities per frame, so it has to be
        refreshed before the tree reads them; ticking the tree first makes every
        scenario act on the previous frame's world.
        """
        self._cdp.on_carla_tick()
        self.tree.tick_once()
        self._ticks += 1
        return self.tree.status

    def cleanup(self):
        """Remove what the scenario spawned, and only that.

        CarlaDataProvider.cleanup() destroys every actor the provider knows
        about, which after register_actor includes the recorder's ego. The
        scenario's own remove_all_actors is the narrower tool and the right one:
        the recorder destroys its own actors in its own teardown.
        """
        try:
            self.scenario.remove_all_actors()
        except Exception as exc:
            print(f"[srunner] pulizia scenario fallita: {exc}", flush=True)
        try:
            self._cdp.set_world(self._world)   # drop cached per-run state
        except Exception:
            pass

    def summary(self) -> dict:
        return {"scenario": self._name, "ticks": self._ticks,
                "status": str(self.tree.status),
                "actors_spawned": len(self.actors)}


class RouteScenarios:
    """Every scenario an official Leaderboard route carries, ticked together.

    A route is not one hazard but a sequence of them - the Town10 route has nine,
    each with its own trigger point and parameters - and they coexist: each
    scenario's behaviour tree gates itself on the ego reaching its own trigger,
    so all of them are built up front and only the relevant one acts. That is how
    the Leaderboard runs them, and reproducing it is the point of using the
    official routes at all.

    Two things this does that a loop over ScenarioBridge would get wrong. The
    data provider is initialised once rather than once per scenario, because
    set_world rebuilds spawn points and the map graph and doing that nine times
    costs seconds for nothing. And on_carla_tick is called once per world tick
    rather than once per scenario, because it is what refreshes the provider's
    per-frame pose cache: calling it between two trees would leave the second
    tree reading a cache that the first tree's actions have already invalidated.
    """

    def __init__(self, world, client, ego, spec_path, *, tm_port=8000,
                 extra_paths=(), criteria=False, timeout=180.0):
        import json
        _ensure_paths(extra_paths)
        cat = catalogue(extra_paths)
        from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
        from srunner.scenarioconfigs.scenario_configuration import (
            ScenarioConfiguration, ActorConfigurationData)
        import carla

        self._cdp = CarlaDataProvider
        self._world = world
        self._ticks = 0
        self.failed = []
        self._blackboards = []
        self.tree = None

        self._cdp.set_client(client)
        self._cdp.set_world(world)
        self._cdp.set_traffic_manager_port(int(tm_port))
        try:
            self._cdp.register_actor(ego, ego.get_transform())
        except Exception:
            pass

        doc = json.loads(Path(spec_path).read_text())
        self.town = doc.get("town", "")
        # Route-aware scenarios read config.route as a list of
        # (carla.Transform, RoadOption) and walk it to find the junction they
        # act at. Handing them an empty route is not a degraded mode: they raise
        # while building and the route quietly loses a hazard.
        from agents.navigation.local_planner import RoadOption
        planned = []
        for q in doc.get("planned_route", []):
            planned.append((
                carla.Transform(
                    carla.Location(x=float(q["x"]), y=float(q["y"]),
                                   z=float(q["z"])),
                    carla.Rotation(yaw=float(q.get("yaw", 0.0)))),
                getattr(RoadOption, q.get("option", "LANEFOLLOW"),
                        RoadOption.LANEFOLLOW)))
        self.planned = planned
        self.scenarios = []
        for s in doc.get("scenarios", []):
            kind = s["type"]
            if kind not in cat:
                self.failed.append((s.get("name", kind), "tipo sconosciuto"))
                continue
            t = s["trigger"]
            tf = carla.Transform(
                carla.Location(x=float(t["x"]), y=float(t["y"]), z=float(t["z"])),
                carla.Rotation(yaw=float(t.get("yaw", 0.0))))
            cfg = ScenarioConfiguration()
            cfg.name = s.get("name", kind)
            cfg.town = world.get_map().name
            cfg.trigger_points = [tf]
            cfg.ego_vehicles = [ActorConfigurationData(ego.type_id, tf,
                                                       rolename="hero")]
            cfg.other_actors = []
            # Naming the blackboard variable is what puts a scenario into the
            # Leaderboard's own gating: with a name set it waits on that
            # variable, which the ScenarioTriggerer flips when the ego comes
            # near along the route. Left unnamed it falls back to waiting for a
            # time-to-arrival of two seconds, which never elapses once the ego
            # has stopped - so a route where the agent halts early would keep
            # every later hazard switched off for the rest of the run.
            cfg.route_var_name = f"ScenarioRouteNumber{len(self.scenarios)}"
            cfg.route = planned or None
            cfg.weather = world.get_weather()
            cfg.other_parameters = {k: {"value": str(v)}
                                    for k, v in (s.get("params") or {}).items()}
            try:
                obj = cat[kind](world, [ego], cfg, criteria_enable=criteria,
                                timeout=timeout)
            except Exception as exc:
                # a scenario that cannot be placed here is reported and skipped,
                # never silently dropped: a route missing a third of its hazards
                # is a different route
                self.failed.append((cfg.name, f"{type(exc).__name__}: {exc}"))
                continue
            self.scenarios.append((cfg.name, kind, obj))
            self._blackboards.append([cfg.route_var_name,
                                      tf.location, cfg.name])

    def build_tree(self, ego, trigger_distance=2.0):
        """Assemble the route's scenarios the way the Leaderboard assembles them.

        One parallel node holding the ScenarioTriggerer first and every
        scenario's gated behaviour after it, so the triggerer runs before the
        behaviours read the variables it sets. Ticking the scenarios' own
        scenario_tree objects individually would skip the triggerer entirely and
        leave every blackboard variable at False.

        The Leaderboard also hangs its BackgroundBehavior here, which populates
        the route with ambient traffic. That is deliberately left out: this
        campaign controls its own traffic, and adding an unlogged fleet would
        change what the sensing arms are being compared on.
        """
        import py_trees
        from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (
            ScenarioTriggerer)
        root = py_trees.composites.Parallel(
            name="RouteScenarios",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)
        root.add_child(ScenarioTriggerer(ego, self.planned, self._blackboards,
                                         trigger_distance))
        for _, _, s in self.scenarios:
            if s.behavior_tree is not None:
                root.add_child(s.behavior_tree)
        # this py_trees exposes setup(), not setup_subtree(); the real
        # ScenarioManager never calls either and ticks the tree directly, so a
        # failure here is not fatal and the tree is kept regardless
        try:
            root.setup(timeout=1)
        except Exception as exc:
            print(f"[srunner] setup albero saltato: {exc}", flush=True)
        self.tree = root
        return root

    @property
    def actors(self):
        out = []
        for _, _, s in self.scenarios:
            out.extend(getattr(s, "other_actors", []) or [])
        return out

    def tick(self):
        self._cdp.on_carla_tick()
        if self.tree is not None:
            try:
                self.tree.tick_once()
            except Exception as exc:
                if self._ticks < 3:
                    print(f"[srunner] albero: {exc}", flush=True)
        else:
            for _, _, s in self.scenarios:
                try:
                    s.scenario_tree.tick_once()
                except Exception:
                    pass
        self._ticks += 1

    def cleanup(self):
        for name, _, s in self.scenarios:
            try:
                s.remove_all_actors()
            except Exception as exc:
                print(f"[srunner] pulizia di {name} fallita: {exc}", flush=True)
        try:
            self._cdp.set_world(self._world)
        except Exception:
            pass

    def summary(self) -> dict:
        return {"town": self.town, "built": len(self.scenarios),
                "failed": len(self.failed), "ticks": self._ticks,
                "actors": len(self.actors),
                "types": sorted({k for _, k, _ in self.scenarios})}
