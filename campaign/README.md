# MatSense closed-loop campaign

The code that produced the closed-loop results of the paper (Sections 6-7:
E3, E4, E5 and E7). Every run records the hash of this code and of the
profile it applied; for the reported runs they are

| | SHA-256 |
| --- | --- |
| code (`matsense_campaign/*.py` and the top-level `*.py`) | `1965ab803c39ce76d916b148d32c6ca713e1ce36c423dfc452816d7f60f0a601` |
| profile `realbag_empirical_v4` (canonical JSON) | `e3f9708d9949f889199f69b32932a7aa51e04a16fdf07b7ac53732f594a3d48d` |

Both can be checked without a simulator:

```bash
python -c "from matsense_campaign import code_sha256; print(code_sha256())"
python -c "from matsense_campaign.profile import load_profile; print(load_profile('config/profile_realbag_empirical_v4.json').sha256)"
```

The files here are the ones the campaigns ran; do not edit them if the hashes
are meant to match. The configurations keep the PCLA path of the machine the
campaigns ran on (`pcla.dir`); set it to your own checkout.

## What it does

- **Operator** (`matsense_campaign/operator.py`), Eqs. 7-11 of the paper:
  per-return factor β·α/φ_max over the measured classes; uniform control with
  one scale per frame, s = Σ f·I / Σ I; shuffled control permuting the factors
  among the returns of each 5 m range bin; return loss and range jitter
  (λd = 0.4, p_max = 0.55, λj = 0.10). With `uniform_geometry: "matched"` the
  uniform and shuffled arms lose and move exactly the returns MatSense does,
  drawn with the same seed. Optional arms: `shuffled_class`, `intensity_only`,
  `geometry_only`.
- **Runtime adapter** (`hook.py`, `association.py`): a semantic LiDAR that
  mirrors the agent's sensor and completes a full turn per frame, and a k-d tree
  match of each return to the nearest semantic ray within 0.40 degrees.
- **Scenarios** (`scenarios.py`, `config/scenarios_resolved.json`): the five
  Leaderboard-style scenarios of Table 6 (15, 05, 09, 16, 18), with locations
  resolved once from the road network and frozen. `config/scenario_*.png` show
  them.
- **Runs** (`run_one.py`, `run_campaign.py`): the block design of Section 6.4.
  A block is one scenario, condition and seed; it holds unmodified, uniform,
  MatSense, shuffled and a second unmodified run in an order drawn at random per
  block. Runs are capped at 35 s and end earlier on route completion, after 12 s
  stationary or beyond 5 m of route deviation.
- **Analysis** (`analyze.py`, `tools/analyze_primary.py`,
  `tools/analyze_ablation.py`): discrete Fréchet distance to the unmodified run
  of the block, paired Wilcoxon tests, Holm correction, Vargha-Delaney Â12.

## Experiments

| Paper | Configuration | Runs |
| --- | --- | --- |
| E3, closed-loop campaign (LAV) | `config/campaign.json` | 500 |
| E4, operator ablation | `experiments/ablation/campaign.json` | 150 |
| E5, second agent (TransFuser++) | `experiments/second_agent/campaign.json` | 500 |
| E7, camera removed | `experiments/camera_zero/campaign.json` (with `tools/run_one_camera.py`) | 200 |

`ANALYSIS_PLAN.md` and the `README.md` of each experiment are the decisions as
they were fixed before the corresponding runs, with their dates; they are kept
unchanged, in the language they were written in.

## Running

Requirements: CARLA 0.9.16 with its Python API, PCLA (agent `lav_lav`, or
`tfv4_l6_0` for E5) and the packages in `requirements.txt`. Commands run from
this folder.

1. Offline checks, no simulator: `python tests/test_offline.py` must end with
   `ALL PASS`.
2. Patch PCLA once with the `perturb_fn` hook: `python tools/patch_pcla.py
   <PCLA dir>` shows the change, `--apply` applies it (originals kept as
   `.orig`).
3. With CARLA running: `python tools/check_semantic_lidar.py` must print `OK`.
4. Scenarios are already resolved in `config/scenarios_resolved.json`; to
   resolve them again, `python tools/resolve_scenarios.py --only <id>` with the
   scenario's town open.
5. Pilot: `python run_campaign.py --plan`, `python run_campaign.py
   --pilot-each 5`, then `python tools/check_pilot.py`, which checks spawn,
   route following, hazard, hook calls, semantic matching and level matching,
   and draws `runs/.../checks/<run>.png`.
6. Campaign: `python run_campaign.py` (interruptible; finished runs are
   skipped, failed ones repeated with `--retry-errors`). Another experiment:
   add `--config experiments/<name>/campaign.json`.
7. Analysis: `python analyze.py` (or with `--config`), then
   `python tools/analyze_primary.py`.

`carla.allow_load_world` is `false`: the code never switches map from Python,
because in the Unreal editor `load_world` crashed the renderer. Open the town
the run needs (Town02 for 15 and 16, Town03 for 05, 09 and 18); the runner
stops with a message naming the map when another one is needed. With a
packaged CARLA it can be set to `true`. RTX 50-series GPUs need PyTorch built
for CUDA 12.8 or later for LAV.

Do not change the code or `config/` during a campaign, and do not mix runs of
different campaigns: the hash in each run and the analysis guard against it.
