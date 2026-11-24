"""Phase 1 tests on a synthetic acquisition: modes, invariants, negative cases."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
P1 = HERE / "phase1"
CFG = HERE / "config" / "phase1.yaml"
SEM = np.dtype([("x", np.float32), ("y", np.float32), ("z", np.float32),
                ("CosAngle", np.float32), ("ObjIdx", np.uint32), ("ObjTag", np.uint32)])


def make_sequence(root: Path, frames=3, seed=0):
    rng = np.random.default_rng(seed)
    (root / "frames").mkdir(parents=True)
    for k in range(frames):
        n = 4000
        az = rng.uniform(-np.pi, np.pi, n); el = rng.uniform(-0.5, 0.15, n); r = rng.uniform(2, 80, n)
        xyz = np.column_stack([r * np.cos(el) * np.cos(az), r * np.cos(el) * np.sin(az), r * np.sin(el)]).astype(np.float32)
        inten = np.exp(-0.004 * r).astype(np.float32)
        tags = rng.choice([1, 2, 3, 9, 14, 24, 5, 6, 11], n, p=[.3, .1, .2, .1, .05, .05, .1, .05, .05])
        sem = np.zeros(n, SEM)
        jitter = rng.normal(0, 0.0005, (n, 3)).astype(np.float32) * r[:, None]
        sem["x"], sem["y"], sem["z"] = (xyz + jitter).T
        sem["ObjTag"] = tags
        drop = rng.random(n) < 0.03            # some returns without a semantic partner
        np.savez_compressed(root / "frames" / f"{k:08d}.npz", raw_xyz=xyz, raw_intensity=inten, sem=sem[~drop],
                            pose=np.zeros(6), sensor_to_world=np.eye(4), pose_id=str(k))
    (root / "manifest.json").write_text(json.dumps({"synthetic": True}))


def run(*args):
    return subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True)


def test_pipeline(tmp_path):
    seq = tmp_path / "seq"; make_sequence(seq)
    out = tmp_path / "paired"
    r = run(P1 / "make_modes.py", "--sequence", seq, "--config", CFG, "--out", out)
    assert r.returncode == 0, r.stderr
    r = run(P1 / "check_invariants.py", out)
    assert r.returncode == 0, r.stdout + r.stderr
    rep = json.loads((out / "invariants.json").read_text())
    assert rep["verdict"] == "PASS"
    r = run(P1 / "measure.py", out, "--config", CFG)
    assert r.returncode == 0, r.stderr
    assert (out / "results" / "contrast.csv").exists()
    # refuses to overwrite
    assert run(P1 / "make_modes.py", "--sequence", seq, "--config", CFG, "--out", out).returncode != 0


def test_tampered_geometry_fails(tmp_path):
    seq = tmp_path / "seq"; make_sequence(seq, frames=1)
    out = tmp_path / "paired"
    assert run(P1 / "make_modes.py", "--sequence", seq, "--config", CFG, "--out", out).returncode == 0
    f = next((out / "modes" / "UNIFORM").glob("*.npz")); d = dict(np.load(f))
    d["xyz"] = d["xyz"].copy(); d["xyz"][0, 0] += 0.01
    np.savez_compressed(f, **d)
    r = run(P1 / "check_invariants.py", out)
    assert r.returncode == 1 and "FAIL geometry" in r.stdout


def test_tampered_fallback_fails(tmp_path):
    seq = tmp_path / "seq"; make_sequence(seq, frames=1)
    out = tmp_path / "paired"
    assert run(P1 / "make_modes.py", "--sequence", seq, "--config", CFG, "--out", out).returncode == 0
    lab = np.load(next((out / "labels").glob("*.npz")))
    f = next((out / "modes" / "MATSENSE").glob("*.npz")); d = dict(np.load(f))
    i = int(np.flatnonzero(lab["cls"] < 0)[0]); d["intensity"] = d["intensity"].copy(); d["intensity"][i] *= 0.5
    np.savez_compressed(f, **d)
    r = run(P1 / "check_invariants.py", out)
    assert r.returncode == 1 and "FAIL fallback" in r.stdout
