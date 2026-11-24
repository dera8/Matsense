"""Campaign configuration and manifest."""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np

from . import ROOT_DIR

MANIFEST_FIELDS = ["run_id", "order", "scenario_id", "kind", "town", "condition", "seed",
                   "replicate", "arm", "repeat", "block", "position_in_block"]


def town_of(map_name: str) -> str:
    """'Carla/Maps/Town02', 'UEDPIE_0_Town02' (Unreal editor) or 'Town02_Opt' -> 'Town02'."""
    name = str(map_name).replace("\\", "/").split("/")[-1]
    name = re.sub(r"^UEDPIE_\d+_", "", name)
    return re.sub(r"_Opt$", "", name)


def town_matches(map_name: str, town: str) -> bool:
    return town_of(map_name).lower() == town_of(town).lower()


def load_world_message(current: str, town: str) -> str:
    return (f"CARLA has {town_of(current)} open, but this step needs {town}. Open {town} in CARLA "
            f"(Unreal editor: stop Play, Content > Carla > Maps > {town}, press Play) and run the "
            f"same command again; completed runs are skipped. Map loading from Python is off "
            f"(carla.allow_load_world=false) because it crashed the editor's renderer.")


def resolve_path(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT_DIR / p


def load_campaign(path: str | Path) -> dict:
    cfg = json.loads(resolve_path(path).read_text(encoding="utf-8"))
    cfg["_path"] = str(resolve_path(path))
    cfg["_out"] = str(resolve_path(cfg["out_dir"]))
    return cfg


def build_manifest(cfg: dict) -> list[dict]:
    """Blocks = (scenario, condition, seed, replicate). Inside each block the arms
    run in random order, plus a second unmodified run as the noise reference.
    Blocks of one scenario are kept together (one map load) but shuffled."""
    rng = np.random.default_rng(int(cfg["campaign_seed"]))
    rows, order, run_id = [], 0, 0
    for sc in [s for s in cfg["scenarios"] if s.get("enabled", True)]:
        blocks = [(c, s, r) for c in cfg["conditions"] for s in cfg["seeds"]
                  for r in range(int(cfg.get("replicates", 1)))]
        for bi in rng.permutation(len(blocks)):
            cond, seed, rep = blocks[bi]
            arms = [(a, 0) for a in cfg["arms"]]
            if cfg.get("unmodified_repeat", True):
                arms.append(("unmodified", 1))
            block = f"{sc['id']}|{cond}|{seed}|{rep}"
            for pos, ai in enumerate(rng.permutation(len(arms))):
                arm, repeat = arms[ai]
                rows.append({"run_id": f"r{run_id:04d}", "order": order, "scenario_id": sc["id"],
                             "kind": sc["kind"], "town": sc["town"], "condition": cond,
                             "seed": seed, "replicate": rep, "arm": arm, "repeat": repeat,
                             "block": block, "position_in_block": pos})
                run_id += 1
                order += 1
    return rows


def write_manifest(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)


def read_manifest(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
