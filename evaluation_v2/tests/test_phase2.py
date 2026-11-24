"""Phase 2 middleware tests: bags from a paired dataset, read back, probe comparison (no ROS)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "phase2"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_phase1 import CFG, P1, make_sequence, run  # noqa: E402

pytest.importorskip("rosbags")
from probe_detector_input import compare_cloud  # noqa: E402
from to_rosbag2 import POINT_DTYPE, encode  # noqa: E402
from verify_rosbag2 import EXPECTED  # noqa: E402


def test_bags_roundtrip(tmp_path):
    seq = tmp_path / "seq"; make_sequence(seq, frames=2)
    ds = tmp_path / "paired"
    assert run(P1 / "make_modes.py", "--sequence", seq, "--config", CFG, "--out", ds).returncode == 0
    assert run(P1 / "check_invariants.py", ds).returncode == 0
    bags = tmp_path / "bags"
    r = run(HERE / "phase2" / "to_rosbag2.py", ds, "--out", bags)
    assert r.returncode == 0, r.stderr
    r = run(HERE / "phase2" / "verify_rosbag2.py", bags, "--dataset", ds)
    assert r.returncode == 0, r.stdout + r.stderr
    assert json.loads((bags / "verify.json").read_text())["verdict"] == "PASS"
    assert run(HERE / "phase2" / "to_rosbag2.py", ds, "--out", bags).returncode != 0   # no overwrite


def _cloud(n=500, seed=0):
    rng = np.random.default_rng(seed)
    return encode(rng.uniform(-50, 50, (n, 3)).astype(np.float32), rng.uniform(0, 1, n), None)


def test_probe_same_cloud_reaches():
    e = _cloud()
    assert compare_cloud(e, e.copy(), EXPECTED)["reaches"]


def test_probe_translated_and_cropped_reaches():
    e = _cloud(); g = e.copy(); g["z"] += 2.4
    res = compare_cloud(e, g, EXPECTED)
    assert res["reaches"] and res["xyz_rigid_translation"] and abs(res["xyz_offset"][2] - 2.4) < 1e-3
    assert compare_cloud(e, e[::3].copy(), EXPECTED)["reaches"]


def test_probe_flattened_intensity_not_reached():
    e = _cloud(); g = e.copy(); g["intensity"] = 0
    assert not compare_cloud(e, g, EXPECTED)["reaches"]
    bad = [("x", 0, 7), ("y", 4, 7), ("z", 8, 7), ("intensity", 12, 7)]
    assert not compare_cloud(e, e.copy(), bad)["layout_ok"]
