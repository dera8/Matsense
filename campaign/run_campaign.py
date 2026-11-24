#!/usr/bin/env python3
"""Plan and execute the campaign, one subprocess per run.

    python run_campaign.py --plan              write manifest.csv and print the plan
    python run_campaign.py --pilot 5           run the first 5 runs only
    python run_campaign.py                     run everything not yet completed

Each run executes in its own process (run_one.py), so a crash of the agent or a
CARLA timeout costs one run, not the campaign. Completed runs (summary.json
present) are skipped, so the command can be repeated after an interruption.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from matsense_campaign.config import (build_manifest, load_campaign, load_world_message,  # noqa: E402
                                      read_manifest, resolve_path, town_matches, write_manifest)


def current_map(cfg: dict) -> str:
    import carla
    client = carla.Client(cfg["carla"]["host"], int(cfg["carla"]["port"]))
    client.set_timeout(20.0)
    return client.get_world().get_map().name

HERE = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/campaign.json")
    ap.add_argument("--plan", action="store_true", help="write the manifest and stop")
    ap.add_argument("--pilot", type=int, default=0, help="run only the first N runs")
    ap.add_argument("--pilot-each", type=int, default=0,
                    help="run only the first N runs of every scenario (the manifest groups runs by "
                         "scenario, so --pilot alone tests the first scenario only)")
    ap.add_argument("--retry-errors", action="store_true", help="rerun runs that ended in error")
    args = ap.parse_args()

    cfg = load_campaign(args.config)
    out = Path(cfg["_out"])
    man = out / "manifest.csv"
    if not resolve_path(cfg["scenarios_resolved"]).exists():
        print("scenarios not resolved yet: run tools/resolve_scenarios.py first")
        return 1
    if not man.exists():
        rows = build_manifest(cfg)
        write_manifest(rows, man)
        (out / "campaign_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        print(f"manifest written: {man} ({len(rows)} runs)")
    rows = read_manifest(man)
    by = Counter((r["scenario_id"], r["condition"], r["arm"]) for r in rows)
    print(f"{len(rows)} runs planned")
    for k in sorted(by):
        print("  ", k, by[k])
    if args.plan:
        return 0

    oc = cfg["orchestrator"]
    if args.pilot_each:
        seen, todo = Counter(), []
        for r in rows:
            if seen[r["scenario_id"]] < args.pilot_each:
                todo.append(r)
                seen[r["scenario_id"]] += 1
    else:
        todo = rows[: args.pilot] if args.pilot else rows
    t0 = time.time()
    for i, r in enumerate(todo):
        run_dir = out / "runs" / r["run_id"]
        if (run_dir / "summary.json").exists():
            continue
        if (run_dir / "error.json").exists() and not args.retry_errors:
            print(f"skip {r['run_id']} (error.json present; use --retry-errors)")
            continue
        if not cfg["carla"].get("allow_load_world", True):
            try:
                now = current_map(cfg)
            except Exception as e:  # noqa: BLE001
                print(f"STOP: CARLA does not answer ({e!r}). Start it and run the same command again.")
                return 3
            if not town_matches(now, r["town"]):
                print(f"STOP before {r['run_id']} (scenario {r['scenario_id']}): " + load_world_message(now, r["town"]))
                return 3
        for attempt in range(int(oc["max_retries"]) + 1):
            print(f"[{i + 1}/{len(todo)}] {r['run_id']} sc={r['scenario_id']} {r['condition']} "
                  f"seed={r['seed']} arm={r['arm']}{'(repeat)' if r['repeat'] == '1' else ''} "
                  f"attempt={attempt + 1}  elapsed={(time.time() - t0) / 60:.1f} min", flush=True)
            run_dir.mkdir(parents=True, exist_ok=True)
            with open(run_dir / "stdout.txt", "w", encoding="utf-8") as log:
                try:
                    p = subprocess.run([sys.executable, str(HERE / "run_one.py"), "--config",
                                        cfg["_path"], "--run-id", r["run_id"]],
                                       stdout=log, stderr=subprocess.STDOUT,
                                       timeout=float(oc["run_timeout_s"]), cwd=str(HERE))
                    code = p.returncode
                except subprocess.TimeoutExpired:
                    code = -9
                    log.write("\n[run_campaign] TIMEOUT\n")
            if code == 0 and (run_dir / "summary.json").exists():
                break
            print(f"   run failed (exit {code}), see {run_dir / 'stdout.txt'}", flush=True)
            if oc.get("carla_restart_cmd"):
                print("   restarting CARLA", flush=True)
                subprocess.run(oc["carla_restart_cmd"], shell=True)
                time.sleep(30)
        time.sleep(float(oc.get("pause_between_runs_s", 3)))
    done = sum((out / "runs" / r["run_id"] / "summary.json").exists() for r in rows)
    print(f"completed {done}/{len(rows)} runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
