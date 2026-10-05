#!/usr/bin/env python3
"""Run a closed-loop campaign plan, one run at a time.

The dashboard writes the plan (campaign.json) and starts this script in the
background, so a campaign survives the browser tab being closed. Each run is
the recorder invoked with the arguments the plan lists for it; nothing here
decides how a run is configured.

Layout of a campaign directory:

  campaign.json            the plan: one entry per run, with its arguments
  status.json              progress, rewritten after every run
  runs/<run_id>/run_summary.json   written by the recorder at the end of a run
  clog/clog_<run_id>.csv           one row per tick, written by the recorder
  logs/<run_id>.log                the recorder's stdout and stderr

A run whose run_summary.json already exists is skipped, so starting the same
plan again resumes it instead of repeating the runs already done.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

_child: subprocess.Popen | None = None
_stopping = False


def _on_term(signum, frame):  # noqa: ARG001
    global _stopping
    _stopping = True
    if _child is not None and _child.poll() is None:
        # SIGINT, not SIGTERM: the recorder turns it into KeyboardInterrupt and
        # runs its cleanup, which removes the ego, the sensors and the hazard
        # actors from the CARLA world. A terminated recorder leaves them there
        # and the next run starts in a polluted world.
        _child.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT)


def _write_status(path: Path, status: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(status, indent=1), encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    global _child
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("campaign_dir")
    args = ap.parse_args()

    root = Path(args.campaign_dir).resolve()
    plan = json.loads((root / "campaign.json").read_text(encoding="utf-8"))
    runs = plan["runs"]
    (root / "logs").mkdir(exist_ok=True)
    status_path = root / "status.json"
    status = {
        "pid": os.getpid(),
        "state": "running",
        "total": len(runs),
        "done": 0,
        "skipped": 0,
        "failed": [],
        "current": None,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "finished": None,
    }
    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT, _on_term)

    for run in runs:
        if _stopping:
            break
        summary = root / "runs" / run["id"] / "run_summary.json"
        if summary.exists():
            status["skipped"] += 1
            status["done"] += 1
            continue
        status["current"] = run["id"]
        _write_status(status_path, status)
        env = os.environ.copy()
        env.update({k: str(v) for k, v in plan.get("env", {}).items()})
        env.update({k: str(v) for k, v in run.get("env", {}).items()})
        with open(root / "logs" / f"{run['id']}.log", "w", encoding="utf-8") as log:
            log.write(" ".join(run["cmd"]) + "\n\n")
            log.flush()
            kwargs = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {}
            _child = subprocess.Popen(
                run["cmd"], cwd=plan["cwd"], env=env, stdout=log, stderr=subprocess.STDOUT, **kwargs
            )
            stop_requested_at = None
            while True:
                try:
                    code = _child.wait(timeout=1.0)
                    break
                except subprocess.TimeoutExpired:
                    if not _stopping:
                        continue
                    stop_requested_at = stop_requested_at or time.time()
                    # Give the recorder time to clean up after SIGINT, then force it.
                    if time.time() - stop_requested_at > 60:
                        _child.kill()
            _child = None
        if _stopping:
            # Interrupted, not failed. The recorder writes a summary on its way
            # out even then; set it aside so a resume runs this one again.
            if summary.exists():
                summary.replace(summary.with_name("run_summary.interrupted.json"))
            break
        status["done"] += 1
        if code != 0 or not summary.exists():
            status["failed"].append({"id": run["id"], "exit_code": code})
        _write_status(status_path, status)

    status["current"] = None
    status["state"] = "stopped" if _stopping else "finished"
    status["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    _write_status(status_path, status)
    return 0


if __name__ == "__main__":
    sys.exit(main())
