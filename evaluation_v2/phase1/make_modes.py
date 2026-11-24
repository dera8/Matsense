#!/usr/bin/env python3
"""From one acquired sequence to a paired dataset: four modes per cloud.

Input, either
  --sequence  a folder written by acquire_sequence.py (raw + semantic LiDAR)
  --sample    a recorder dataset folder (lidar_raw/, lidar_labels/), whose
              per-point semantic_tag is used as is (-1 = no tag)

Output (a NEW folder, never overwritten):
  modes/<MODE>/<frame>.npz   xyz (identical in every mode), intensity
  labels/<frame>.npz         tag, matched, cls (-1 unmatched, -2 unmapped, >=0 class)
  manifest.json              inputs, profile and code hashes, settings, per-frame diagnostics

    python make_modes.py --sequence ../data/seq_town03_001 --config ../config/phase1.yaml --out ../data/paired_town03_001
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from modes import MODES, ModeSettings, classes_for, load_profile, make_modes  # noqa: E402

from matsense_campaign import code_sha256  # noqa: E402  (campaign package, path set by modes)
from matsense_campaign.association import match_tags  # noqa: E402


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def frames_from_sequence(root: Path, tol_deg: float):
    for f in sorted((root / "frames").glob("*.npz")):
        d = np.load(f, allow_pickle=False)
        xyz, inten = d["raw_xyz"], d["raw_intensity"]
        tags, matched = match_tags(xyz, d["sem"], tol_deg)
        yield f.stem, xyz, inten, tags, matched


def frames_from_sample(root: Path):
    for f in sorted((root / "lidar_raw").glob("*.npz")):
        r = np.load(f, allow_pickle=True)
        lab = np.load(root / "lidar_labels" / f.name, allow_pickle=True)
        tags = lab["semantic_tag"].astype(np.int64)
        yield f.stem, r["xyz"], r["intensity"], np.where(tags < 0, 0, tags), tags >= 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--sequence")
    src.add_argument("--sample")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import yaml
    cfg_path = Path(args.config).resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    mcfg = cfg["modes"]
    prof_path = (cfg_path.parent / mcfg["profile"]).resolve()
    profile = load_profile(prof_path)
    profile.check_condition(mcfg["condition"])
    st = ModeSettings(condition=mcfg["condition"], range_bin_m=float(mcfg["range_bin_m"]),
                      shuffle_seed=int(mcfg["shuffle_seed"]))
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists: refusing to overwrite")
    modes = list(MODES) + (["SHUFFLED_LEVEL"] if mcfg.get("optional_shuffled_level") else [])
    for m in modes:
        (out / "modes" / m).mkdir(parents=True)
    (out / "labels").mkdir()

    if args.sequence:
        root = Path(args.sequence)
        it = frames_from_sequence(root, float(mcfg["tolerance_deg"]))
        source = {"kind": "sequence", "path": str(root.resolve())}
        if (root / "manifest.json").exists():
            source["manifest_sha256"] = sha256_file(root / "manifest.json")
    else:
        root = Path(args.sample)
        it = frames_from_sample(root)
        source = {"kind": "recorder_sample", "path": str(root.resolve()),
                  "note": "semantic_tag from the recorder; -1 means no tag (camera-projected labels)"}

    per_frame = []
    for name, xyz, inten, tags, matched in it:
        cls = classes_for(tags, matched, profile)
        ints, diag = make_modes(xyz, inten, cls, profile, st, with_optional="SHUFFLED_LEVEL" in modes)
        xyz32 = np.asarray(xyz, dtype=np.float32)
        for m in modes:
            np.savez_compressed(out / "modes" / m / f"{name}.npz", xyz=xyz32, intensity=ints[m])
        np.savez_compressed(out / "labels" / f"{name}.npz", tag=tags.astype(np.int32),
                            matched=matched.astype(bool), cls=cls.astype(np.int32))
        per_frame.append({"frame": name, **diag})
        print(f"  {name}: n={diag['n']} supported={diag['n_supported']} unmatched={diag['n_unmatched']} "
              f"unmapped={diag['n_unmapped']} s={diag['uniform_scale']:.4f} "
              f"shuffle residual={100 * diag['shuffle_level_residual']:+.2f}%", flush=True)

    here = Path(__file__).resolve().parent
    manifest = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": source,
        "profile": {"path": str(prof_path), "name": profile.name, "sha256": profile.sha256},
        "campaign_code_sha256": code_sha256(),
        "phase1_code_sha256": {p.name: sha256_file(p) for p in sorted(here.glob("*.py"))},
        "config_sha256": sha256_file(cfg_path),
        "settings": {"condition": st.condition, "range_bin_m": st.range_bin_m,
                     "shuffle_seed": st.shuffle_seed, "tolerance_deg": mcfg["tolerance_deg"],
                     "modes": modes, "clip": [0.0, 1.0],
                     "fallback": "identity (f = 1) for unmatched and unmapped returns",
                     "geometry": "unchanged: no dropout, no jitter"},
        "frames": per_frame,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"\n  {len(per_frame)} frames, modes {modes} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
