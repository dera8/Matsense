# evaluation_v2

A new, separate evaluation of MatSense in two phases. It reuses the paper's
operator and profile (`campaign/`) read-only and never overwrites earlier data.
See `AUDIT.md` for the phase 0 audit.

## Phase 1: CARLA native vs CARLA + MatSense, no ADS

One acquisition; every mode is derived offline from it, so the modes differ
only in intensity.

| mode | intensity |
|---|---|
| RAW | CARLA's own |
| UNIFORM | `s·I` on every return, `s = ΣfI/ΣI` over supported returns |
| MATSENSE | `f·I` (profile factor of the class; identity for unmatched/unmapped) |
| SHUFFLED | MatSense factors permuted among supported returns within 5 m range bins; **no rescaling**, level residual measured |
| SHUFFLED_LEVEL (optional) | SHUFFLED with supported returns rescaled to MATSENSE's sum; reported separately |

```
python phase1/acquire_sequence.py --config config/phase1.yaml --out data/seq_town03_001   # CARLA machine
python phase1/make_modes.py --sequence data/seq_town03_001 --config config/phase1.yaml --out data/paired_town03_001
python phase1/check_invariants.py data/paired_town03_001        # exit 1 on FAIL
python phase1/measure.py data/paired_town03_001 --config config/phase1.yaml
```

Invariants: identical geometry; RAW = acquisition; operator applies the loaded
coefficients (check A); UNIFORM is one scalar; SHUFFLED is a multiset
permutation per bin; fallback returns unchanged in material modes; association
agrees with brute force; values finite, clipping counted.

Check A is correctness, not realism. Check B (agreement with held-out real
measurements) runs only with `--real-reference`; otherwise it is declared not
executable. Frames of one sequence are not independent: the sequence is the
replication unit.

## Phase 2: Autoware, open-loop perception

```
python phase2/to_rosbag2.py data/paired_town03_001 --out data/bags_town03_001 [--distro jazzy]
python phase2/verify_rosbag2.py data/bags_town03_001 --dataset data/paired_town03_001
# on the Autoware machine, per mode:
ros2 bag play data/bags_town03_001/MATSENSE --clock
python phase2/probe_detector_input.py --bags data/bags_town03_001 --mode MATSENSE --topic <detector input> --out probe_MATSENSE.json
```

The bags reproduce what `autoware_carla_interface` publishes (PointXYZIRC,
uint8 intensity, y negated, `velodyne_top`). Autoware's default CenterPoint does
not read intensity (AUDIT.md §3): with it, the modes are identical for the
model by construction. The detector is chosen before any run and kept fixed.

## Tests

```
python -m pytest -q evaluation_v2/tests
```

## Evidence of the sample run

`artifacts/sample_scene_001/`: the pipeline on `sample_data/scene_001/nominal`
(5 real CARLA frames): invariants, per-class tables, check A, bag verification.
