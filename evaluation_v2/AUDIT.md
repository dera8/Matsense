# Phase 0 audit (2026-11-24)

What exists, what it is compatible with, and what is missing for the two-phase
evaluation (CARLA native vs CARLA + MatSense without an ADS; Autoware as system
under test). Nothing here changes the paper's campaigns, profiles or data.

## 1. Repository state

| item | value |
|---|---|
| branch | `matsense-app-fixes` (also: `main`, `matsense-pipeline-update`, `claude/matsense-streamlit-progress-wrdnhe`) |
| base for this work | `3a5b953` (phase 1), on top of `efde921` (campaign package), `372529b` (runtime operator matches the campaign), `895fb57` (docs aligned with the paper) |
| profile | `campaign/config/profile_realbag_empirical_v4.json`, canonical sha256 `e3f9708d9949…` (file sha256 `17d7500ed06c…`), read-only |
| operator | `campaign/matsense_campaign/operator.py` (`class_index`, `factors`), imported, not copied |
| campaign config | `campaign/config/campaign.json`, `scenarios_resolved.json` (closed loop with PCLA agents; not used here) |
| new code | `evaluation_v2/` only: `phase1/`, `phase2/`, `config/phase1.yaml`, `tests/` |

## 2. Discrepancies found and how this evaluation handles them

| # | finding | handling |
|---|---|---|
| D1 | The campaign's `shuffled` arm **rescales** the permuted cloud to the MatSense total (`operator.py`, `scale = target / got`). The paper describes it as a pure permutation. | Phase 1 `SHUFFLED` is a pure permutation within 5 m bins; the level residual is measured and reported. A rescaled variant exists only as the separately named optional mode `SHUFFLED_LEVEL`, applied to supported points only. |
| D2 | The public integration is an in-process callback (PCLA `perturb_fn` / the runtime hook). There is no ROS node. | Phase 2 adds an offline middleware layer (`to_rosbag2.py`) that encodes clouds already produced by the phase 1 sensing code. The sensing logic is not duplicated. |
| D3 | The paper's E6 used autoware_mini's SFA detector (ROS 1), not Autoware Universe. | Phase 2 targets Autoware Universe; results are not comparable with E6 and are not presented as a replication. |

## 3. External compatibility (official sources)

| component | finding | source |
|---|---|---|
| Autoware ↔ CARLA | `autoware_carla_interface`: README states CARLA 0.9.15, ROS 2 Humble, Ubuntu 22.04. It does **not** use carla-ros-bridge. | github.com/autowarefoundation/autoware_universe, `simulator/autoware_carla_interface` |
| Autoware 2.0.0 demo | Docker demo with CARLA 0.9.16 on ROS 2 Jazzy | autowarefoundation.github.io/autoware-documentation |
| carla-ros-bridge | supports CARLA 0.9.13, unmaintained; not needed | github.com/carla-simulator/ros-bridge |
| CARLA 1.0 (UE5) | released 2026-11-17; the interface is not validated against it | carla.org |
| Windows CARLA server + Linux client | possible in principle (TCP API), not verified for this stack | — |
| LiDAR message from the interface | PointXYZIRC 16 B (x,y,z f32; intensity u8 = `uint8(clip(I,0,1)*255)`; return_type u8 = 0; channel u16), y negated, frame `velodyne_top`, topic `/sensing/lidar/top/pointcloud_before_sync` | `autoware_carla_interface/src/.../carla_ros.py`, `lidar()` |
| CARLA sensor kit | no distortion correction; the cloud is relayed to `/sensing/lidar/concatenated/pointcloud` | `carla_sensor_kit_launch` |
| Detector input layout | `autoware::point_types::is_data_layout_compatible_with_point_xyzirc`: a cloud that is not PointXYZIRC is skipped by CenterPoint | `autoware_point_types`, `autoware_lidar_centerpoint` |
| **Default detector** | `lidar_detection_model` = `centerpoint` (base v4.1). `point_feature_size: 4` = x, y, z, time lag. **Intensity is not an input.** | `autoware_lidar_centerpoint/config/centerpoint_ml_package.param.yaml` |
| TransFusion | 5 point features: x, y, z, intensity, time lag; intensity is the uint8 byte cast to float (0–255) | `autoware_lidar_transfusion/config/transfusion_ml_package.param.yaml` |
| CARLA intensity | `exp(-a·d)` with dropoff; depends on distance only | CARLA docs, ray_cast LiDAR |

Consequence: with the default configuration an intensity-only comparison
yields **identical detector inputs as far as the model is concerned** (xyz are
byte-identical across modes). This is a configuration limit, reported as such;
no feature is added to the model. A detector that reads intensity (TransFusion)
must be chosen before any run and kept fixed, with the same weights, thresholds
and preprocessing in every mode.

## 4. Components

| component | status | where |
|---|---|---|
| MatSense operator and profile | implemented (reused) | `campaign/matsense_campaign` |
| Semantic association (k-d tree, 0.40°) | implemented (reused) | `campaign/matsense_campaign/association.py` |
| No-ADS acquisition, ray-cast + semantic LiDAR, actor boxes | implemented, **not yet run on a CARLA server** | `phase1/acquire_sequence.py` |
| Four paired modes (+ optional SHUFFLED_LEVEL) | implemented, tested | `phase1/modes.py`, `phase1/make_modes.py` |
| Automatic invariants | implemented, tested incl. negative cases | `phase1/check_invariants.py` |
| Per class × range tables, check A, check B | implemented; B declared not executable without a held-out real reference | `phase1/measure.py` |
| ROS 2 encoding (Autoware CARLA kit format) | implemented, tested | `phase2/to_rosbag2.py`, `phase2/verify_rosbag2.py` |
| End-to-end probe of the detector input | implemented (comparison unit-tested); **needs a machine with Autoware** | `phase2/probe_detector_input.py` |
| Live ROS adapter node | missing (not needed for open-loop replay; needed for phase 3) | — |
| Detection metrics against CARLA actor boxes | missing (after the probe, and after the detector choice) | — |
| Held-out real per-class reference for check B | missing (needs real recordings excluded from the profile's estimation, same street) | — |
| Closed-loop scenario | out of scope until phases 1–2 pass | — |

## 5. Data available here

| data | use | limit |
|---|---|---|
| `sample_data/scene_001/nominal` (5 frames, recorder) | pipeline smoke test on real CARLA output | semantic tags are camera-projected: only 19 % of returns supported, 76 % unmatched |
| CARLA sequence from `acquire_sequence.py` | the phase 1 dataset | to be acquired on the CARLA machine |
| real bags (paper E1–E2) | would enable check B | not in this repository |
