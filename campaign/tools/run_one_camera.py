#!/usr/bin/env python3
"""run_one.py with the camera contribution to the LiDAR removed (limit case).

    python tools/run_one_camera.py --config experiments/camera_zero/campaign.json --run-id r0100

LAV paints every LiDAR point with the classes its camera segmentation predicts
(pred_sem = softmax(seg)[:,1:] * (1 - softmax(seg)[:,:1])). This wrapper replaces
the segmentation output with "background everywhere", so the painted channels are
zero and the camera adds nothing to the cloud. The segmentation network still runs
(same compute time); the camera brake classifier and everything else are untouched.
Nothing in PCLA or in the campaign code is modified: the wrapper patches the agent
instance after PCLA has loaded it, and records what it did in summary.json.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_one  # noqa: E402

STATE = {"seg_calls": 0, "patched": False}


class _BackgroundOnly:
    """Callable standing in for the seg model: same output shape, all background."""

    def __init__(self, model):
        self.model = model

    def __call__(self, x):
        out = self.model(x)
        z = out.new_zeros(out.shape)
        z[:, 0] = 50.0  # softmax -> channel 0 = 1, so (1 - p0) = 0 and every painted channel is 0
        STATE["seg_calls"] += 1
        return z

    def __getattr__(self, name):
        return getattr(self.model, name)


_orig_import = run_one.import_pcla


def _import_pcla(pcla_dir):
    PCLA = _orig_import(pcla_dir)

    class PCLACameraZero(PCLA):
        def setup_agent(self, agent):
            super().setup_agent(agent)
            inst = self.agent_instance
            if not hasattr(inst, "seg_model"):
                raise RuntimeError("agent has no seg_model: the camera-zero wrapper only supports LAV")
            inst.seg_model = _BackgroundOnly(inst.seg_model)
            STATE["patched"] = True

    return PCLACameraZero


run_one.import_pcla = _import_pcla


def main() -> int:
    status = run_one.main()
    args = sys.argv[1:]
    cfg = run_one.load_campaign(args[args.index("--config") + 1])
    run_dir = Path(cfg["_out"]) / "runs" / args[args.index("--run-id") + 1]
    info = {"mode": "painted_channels_zero", "patched": STATE["patched"], "seg_calls": STATE["seg_calls"],
            "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for name in ("summary.json", "error.json"):
        p = run_dir / name
        if p.exists():
            s = json.loads(p.read_text(encoding="utf-8"))
            s["camera"] = info
            p.write_text(json.dumps(s, indent=2, default=str), encoding="utf-8")
    if status == 0 and (not STATE["patched"] or STATE["seg_calls"] == 0):
        print(f"[camera] wrapper did not act: {info}", flush=True)
        (run_dir / "summary.json").rename(run_dir / "error.json")
        return 3
    print(f"[camera] {info}", flush=True)
    return status


if __name__ == "__main__":
    sys.exit(main())
