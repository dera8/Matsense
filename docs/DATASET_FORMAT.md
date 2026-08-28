# MatSense synthetic dataset format

MatSense is a testing pipeline that also records what it simulates, so a run can
be replayed, audited, or used as training data. This describes what a recorded
run contains and how to read it. It is written for someone who did not record
the data.

Recording is off by default: a closed-loop campaign measures behaviour and does
not need the frames, and saving them costs roughly a gigabyte per run. Turn it
on with `MATSENSE_SAVE_DATASET=1`, or `--save-dataset` on the recorder directly.

## Layout

One directory per run:

```
<scene_id>/<scenario>/
  calibration.json        sensor geometry, written once
  scenario_metadata.json  map, weather, spawn, trajectory, sensor settings
  run_summary.json        provenance and outcome, written at the end
  frame_metadata.csv      one row per saved frame
  actors.csv              one row per surrounding actor per saved frame
  rgb/frame_%06d.png
  semantic/frame_%06d.png        CARLA semantic camera, class in the red channel
  semantic_vis/frame_%06d.png    the same, in the palette, for looking at
  projection_debug/frame_%06d.png  the cloud drawn over the image
  lidar_raw/frame_%06d.npz
  lidar_labels/frame_%06d.npz
```

`tools/build_dataset_index.py <root>` writes `index_runs.csv` and
`index_frames.csv` at the dataset root, so runs and frames can be selected by
weather, mode, material composition or outcome without walking the tree. Both
are derived: delete them and re-run.

## The point cloud

`lidar_raw/frame_%06d.npz`

| array | meaning |
| --- | --- |
| `xyz` | `(N, 3) float32`, in the LiDAR frame |
| `intensity` | `(N,) float32`, **CARLA's own intensity**, before MatSense |

`lidar_labels/frame_%06d.npz`, aligned row-for-row with `xyz`:

| array | meaning |
| --- | --- |
| `material_label` | `(N,)` string, the material MatSense assigned |
| `semantic_tag` | `(N,) int32`, CARLA class id; `-1` if the point did not project into the camera |
| `instance_id` | `(N,) uint32`, the object the return came from; **`0` means no match, not object zero** |
| `range` | `(N,) float32`, metres |
| `pseudo_final` | `(N,) float32`, **the MatSense intensity**, after the material and condition coefficients |
| `intensity_norm`, `pseudo_norm` | display-normalised copies; for rendering, not for analysis |

**`intensity` and `pseudo_final` are the same returns before and after the
transformation.** That pairing is the point of the dataset: it is what lets a
downstream user measure the effect of material-aware intensity rather than take
it on trust, and it is the one thing here that a CARLA recording does not
already give you. Use `intensity` as the unmodified baseline and `pseudo_final`
as the MatSense condition; do not mix them with the `_norm` variants, whose
scaling depends on a per-frame percentile.

`instance_id` and `semantic_tag` come from a co-located semantic LiDAR matched
to the ray-cast returns by direction. The match is not complete: a residual
fraction of returns, concentrated below the horizon at grazing incidence on road
surfaces, finds no partner and is labelled `unknown` with `instance_id = 0`.
Treat `0` as missing, not as a class.

## Poses and boxes

`frame_metadata.csv` carries the ego: position, orientation, velocity, speed and
the control it was applying (`ego_throttle`, `ego_steer`, `ego_brake`), plus the
point counts and the fraction of returns that got a known material.

`actors.csv` carries everything else within 120 m: `actor_id`, `type_id`,
category, pose, velocity, and the bounding box as half-extents plus an offset in
the actor's own frame. Position and orientation are in world coordinates, the
same frame as the ego columns, so a 3D box in the sensor frame is

```
T_sensor_from_world @ T_world_from_actor @ (box corners from extent and offset)
```

Actors farther than 120 m are omitted, which is well beyond LiDAR range: an
actor absent from `actors.csv` was out of range, an actor present but with no
returns was occluded or missed.

## Calibration

`calibration.json` is what makes the image and the cloud usable together.

- `camera_intrinsics.K`, with `width`, `height` and the field of view.
- `T_camera_from_lidar`, a 4×4 in CARLA/UE axes, read off the live sensors so it
  is the transform the recorder itself used.
- `sensors[*].T_ego_from_sensor`, each sensor's mounting on the vehicle.
- `lidar_config`, the channel count, range, rate and vertical field of view.

To project a LiDAR point into the image, in the order `conventions.note` states:

```python
p_ue  = T_camera_from_lidar @ [x, y, z, 1]          # still UE axes
p_cam = ue_to_camera @ p_ue[:3]                     # x right, y down, z forward
uv    = (K @ p_cam)[:2] / p_cam[2]                  # only where p_cam[2] > 0
```

CARLA is left-handed with x forward, y right, z up, and rotations are degrees.
`ue_to_camera` is given in the file rather than assumed.

## Provenance

`run_summary.json` records what produced the run: profile name, version and the
**sha256 of the profile configuration**, the route file and its hash, the seed,
the replicate index, the mode, the weather, and the CARLA client and server
versions, alongside the run's outcome. A sample can be traced back to the exact
configuration that generated it, and two runs can be compared knowing whether
anything about the sensor model changed between them.

## Known gaps

- Runs recorded before this format have no `calibration.json`, no `actors.csv`
  and no `instance_id`. The index reports which runs have them
  (`has_calibration`, `has_actors`, `has_instance_ids`) rather than hiding the
  difference.
- Weather is recorded as the profile name, not as CARLA's individual weather
  parameters.
- The unmatched fraction described above is a real limitation of the semantic
  co-location, not a labelling convention.
