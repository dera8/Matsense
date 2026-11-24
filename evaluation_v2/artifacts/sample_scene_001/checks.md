# Phase 1 checks

Dataset: `sample_data/scene_001/nominal -> paired dataset`  profile `realbag_empirical_v4` (e3f9708d9949)  condition `nominal`  frames 5

## A. Operator against the loaded coefficients (correctness, not realism)

Contrast = median intensity in 20-60 m relative to `building`. MATSENSE multiplies RAW by the profile factors exactly, so its class contrast should be RAW's contrast times the profile's: the last column should be 1 wherever both classes have points.

| class | profile contrast f_m/f_ref | RAW contrast | MATSENSE contrast | MATSENSE / RAW / expected |
|---|---:|---:|---:|---:|
| asphalt | 0.5472 | 1.0039 | 0.5493 | 1.0000 |
| building | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| car | 0.7113 | 0.9768 | 0.6948 | 1.0000 |
| sidewalk | 0.6529 | 0.9655 | 0.6304 | 1.0000 |
| vegetation | 0.8335 | 0.9711 | 0.8094 | 1.0000 |

## B. Agreement with held-out real measurements

**Not executable with the data available here.** It needs per-class medians from real recordings excluded from the profile's estimation, on the same street as the simulated sequence (the paper's E2 reference). Check A below is not a substitute: MATSENSE reproduces the loaded coefficients by construction.

## Global level

| mode | mean intensity | ratio to RAW (median, p10-p90 over frames) |
|---|---:|---:|
| RAW | 0.9460 | 1.0000 (1.0000-1.0000) |
| UNIFORM | 0.8878 | 0.9384 (0.9381-0.9387) |
| MATSENSE | 0.8878 | 0.9384 (0.9381-0.9387) |
| SHUFFLED | 0.8878 | 0.9384 (0.9381-0.9388) |
| SHUFFLED_LEVEL | 0.8878 | 0.9384 (0.9381-0.9387) |

## Coverage

Supported 19.1%, unmatched 75.6%, unmapped 5.4% of returns (median over frames). Unmatched and unmapped returns keep RAW intensity in every material-dependent mode; UNIFORM scales every return.
