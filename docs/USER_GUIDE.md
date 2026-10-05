# MatSense Dashboard User Guide

This guide explains how to use the MatSense dashboard included in this repository.

It covers:

- starting the Streamlit app;
- using the bundled demo dataset;
- launching the CARLA viewer;
- recording MatSense outputs;
- reading the Dataset Analyzer views.

This guide does not describe custom ROS-bag processing, map generation, trajectory alignment, or parked-vehicle scenario construction.

---

## 1. Install and Start

From the repository folder:

```powershell
cd matsense_streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

On Ubuntu/Linux, a typical virtual-environment setup is:

```bash
cd matsense_streamlit_app
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The dashboard and Dataset Analyzer do not require CARLA. The Run Viewer requires CARLA, the CARLA Python API, `pygame`, and a graphical display.

The **Browse** buttons use Python `tkinter`. On Ubuntu, install it if file dialogs do not open:

```bash
sudo apt install python3-tk
```

In headless environments, or if `tkinter` is unavailable, paste file and folder paths manually into the text fields.

Open the URL printed by Streamlit:

```text
http://localhost:8501
```

The app has two pages:

- **Run Viewer**
- **Dataset Analyzer**

---

## 2. Sidebar

The sidebar contains the **Toolkit path**.

By default it points to:

```text
bundled_toolkit
```

Keep this value unless you want to use another local MatSense toolkit checkout.

The bundled toolkit contains the runtime viewer and material configuration needed by the dashboard.

---

## 3. Dataset Analyzer

The Dataset Analyzer works without CARLA.

By default it loads:

```text
sample_data/scene_001
```

This sample contains small nominal, rain, and snow subsets.

Use this page to inspect:

- frame counts;
- trajectory-level LiDAR point clouds;
- single-frame LiDAR point clouds;
- pseudo-reflectance by material;
- differences between nominal and adverse weather;
- default CARLA LiDAR intensity versus MatSense pseudo-reflectance;
- projection and material coverage in the advanced section.

### Typical Use

1. Open **Dataset Analyzer**.
2. Keep `Dataset root = sample_data`.
3. Select `scene_001`.
4. Select scenarios such as `nominal`, `rain`, and `snow`.
5. Keep `Baseline scenario = nominal`.
6. Start with **Trajectory Point Cloud**.
7. Use **Material Response** for weather/material comparison.
8. Use **Frame Inspector** to inspect a single saved LiDAR frame.

### Main Views

**Trajectory Point Cloud**

Aggregates points across several frames and displays a top-down point cloud. Color options include pseudo-reflectance, material class, raw CARLA intensity, and normalized pseudo-reflectance.

Use this view to quickly see whether the recorded LiDAR response is spatially coherent along the route.

When points overlap heavily, use:

```text
Display = Binned mean
```

This aggregates nearby points into top-down cells and colors each cell by the mean pseudo-reflectance or intensity. Use `Bins`, `Point size`, and `Point opacity` to tune visibility.

**Material Response**

Shows MatSense pseudo-reflectance by material and weather scenario, plus the comparison between default CARLA LiDAR intensity and MatSense pseudo-reflectance.

**Frame Inspector**

Loads one saved LiDAR frame and displays it in top-down view. Use this when checking a specific frame before looking at the full trajectory.

For pseudo-reflectance and raw-intensity views, switch between `Points` and `Binned mean` depending on whether you want individual returns or a cleaner spatial summary.

**Advanced Metrics**

Contains the detailed tables, coverage plot, and diagnostic scatter plots.

---

## 4. Run Viewer

The Run Viewer page launches the CARLA/pygame viewer.

This page requires:

- CARLA already running;
- the correct map loaded in CARLA;
- a Python environment where `import carla` works;
- `pygame` installed from `requirements.txt`.

### Input and Output Paths

Use the **Browse** buttons to select:

```text
Trajectory TXT   optional trajectory in TXT format
Trajectory JSON  optional CARLA-local trajectory
Parked JSON      optional parked-vehicle layout
Dataset root     output folder for generated datasets
```

You can leave trajectory fields empty and enable **Autopilot** if no trajectory is available.

### Viewer Settings

Start from a **Preset** (`Nominal demo`, `Rain demo`, `Snow demo`): it fills
weather, view, display scale and scenario name. Everything stays editable, and
`Custom` keeps your own values.

The main settings are always visible:

```text
Weather          nominal / rain / snow (snow is not calibrated)
View             RGB overlays / Material classes / Pseudo-reflectance / CARLA raw intensity
Material profile realbag_empirical_v4 (calibrated default) or another profile in the tool config
Autopilot        CARLA drives when no trajectory is given
Save dataset     record frames, clouds and labels
```

Less common settings are grouped in collapsible sections:

```text
Recording                       scene ID, scenario name (empty = same as Weather), save every N frames,
                                max saved frames, start delay, sensor sync timeout
Display                         pseudo scale, display percentile, nominal base values, window size, FPS
Trajectory and parked vehicles  follow mode, trajectory step, control smoothing, UTM offsets
                                (0/0 = read from the map's .xodr), Z offsets, max parked vehicles, seed
LiDAR sensor                    channels, points per sweep, range, vertical FOV
CARLA connection                host, port, Traffic Manager port, allow version mismatch
```

Next to the preset, **LiDAR model** fills the LiDAR sensor section:

```text
Velodyne VLP-32C (real recordings)  32 beams, -25 to +15 deg, 57 600 points per sweep, 200 m (default)
MatSense paper (CARLA, 64 ch)       64 beams, -30 to +10 deg, 60 000 points per sweep, 85 m
Viewer default (64 ch)              64 beams, -30 to +10 deg, 65 000 points per sweep, 80 m
```

The viewer completes one full sweep per frame (its rotation frequency is the
FPS), so the dashboard passes points per second = points per sweep x FPS.

Use `View = Material classes` to inspect semantic material labels,
`View = Pseudo-reflectance` to inspect the material- and weather-aware response,
and `View = CARLA raw intensity` to inspect CARLA's own LiDAR intensity.

### CARLA Status

**Check CARLA** tells you, before launching anything:

- whether `import carla` works in the Python running the dashboard (the viewer uses the same one);
- whether a CARLA server answers on the configured host and port;
- the client and server versions, and the map currently loaded.

**Start Viewer** runs the same check and does not launch the viewer if CARLA
cannot be imported, the server is not reachable, or client and server versions
differ (unless the mismatch is explicitly allowed).

### Start and Stop

Click **Start Viewer** to open the pygame viewer. While it runs, the page shows
its PID. Click **Stop Viewer** to stop the last viewer launched by the dashboard.

---

## 4b. Profile

The **Profile** page shows what a material profile contains, without opening
the JSON:

- **Coefficients**: per material, the nominal response `beta`, the condition
  ratios `alpha` (rain, snow) and whether the material is *measured* from real
  recordings or *declared* as a fallback. Declared bars are hatched.
- **Applied factor**: the factor actually multiplied into CARLA's intensity,
  `beta x alpha` divided by its maximum over the measured materials for that
  condition. It is computed the same way as the runtime operator.
- **Semantic mapping**: which CARLA semantic classes map to which material, and
  which fall back to the default material. The many-to-one table shows where
  several CARLA classes share one coefficient.
- **Notes**: the calibration notes stored in the profile, and the raw JSON.

**Compare with** puts a second profile next to the first, with the differences,
which is useful for the `swap_*` and `shuffled_*` control profiles.

---

## 4c. Experiment

The **Experiment** page runs closed-loop campaigns and compares sensing arms.
It needs CARLA, the forked PCLA (with the `perturb_fn` hook) and, for scripted
hazards, ScenarioRunner (`SCENARIO_RUNNER_ROOT`, `CARLA_PYTHONAPI_ROOT`).

### Campaign

Set the PCLA directory, the route XML, the agent (default `lav_lav`, the
original LAV; LAV needs CARLA started with `-vulkan`), the town and optionally a
ScenarioRunner hazard with its parameters. Then choose the design:

```text
Sensing arms   standard (CARLA's response) / global / matsense / shuffled / level
Conditions     nominal / rain / snow (snow is not calibrated)
Seeds          perturbation seeds, e.g. 1, 2, 3
Replicates     repetitions of each cell
```

The page shows how many runs the design produces before you start it.
**Start campaign** checks CARLA first, writes the plan to
`output_campaigns/<name>/campaign.json` and runs it in the background, one run
at a time, so the browser can be closed. **Stop** ends the current run cleanly
(the recorder removes its actors from CARLA) and **Resume** runs only what has
no `run_summary.json` yet.

Each campaign folder contains:

```text
campaign.json                  the plan, one command per run
status.json                    progress
runs/<run>/run_summary.json    outcome of each run
clog/clog_<run>.csv            one row per simulation tick
logs/<run>.log                 recorder output, for failed runs
```

### Results

- outcomes per arm and condition: runs, completions, collisions, stops, median
  route completion and minimum TTC;
- the share of completed runs per arm, and the distribution of the chosen
  metric;
- a paired comparison against a baseline arm, pairing runs on condition, seed
  and replicate (mean difference, 95 % interval, Wilcoxon; Fisher for
  completion);
- run traces: the same situation under each arm, tick by tick;
- **Save results to Evidence** writes the summary to `output_analysis/` with
  its provenance.

Runs that died inside the agent are counted separately and excluded, since
they never drove. Summaries written less than 90 seconds ago are left out
until they settle.

---

## 5. Recording a Dataset

To record a dataset:

1. Start CARLA with the target map.
2. Open **Run Viewer**.
3. Set the path fields.
4. Set `Weather`.
5. Set `Scene ID`.
6. Set `Scenario name`.
7. Enable **Save dataset**.
8. Click **Start Viewer**.

Recommended scenario names:

```text
nominal
rain
snow
fog
```

Recommended layout:

```text
output_dataset/
`-- scene_001/
    |-- nominal/
    |-- rain/
    |-- snow/
    `-- fog/
```

After recording, open **Dataset Analyzer** and set `Dataset root` to the recorded output folder.

---

## 6. Viewer Keyboard Controls

Inside the pygame viewer:

```text
P      cycle view mode
R      respawn ego vehicle
WASD   manual driving when autopilot is off
Arrows manual driving when autopilot is off
ESC    quit
```

---

## 7. Troubleshooting

### The pygame window does not appear

Check the error shown in Streamlit and these files:

```text
viewer_stderr.log
viewer_stdout.log
viewer_command.txt
```

Common causes:

- CARLA is not running;
- the CARLA Python module is not available;
- the wrong map is loaded;
- the selected trajectory does not match the loaded map;
- the selected parked-vehicle file does not match the loaded map.

### Dataset Analyzer shows no scenes

Check that the dataset root contains folders like:

```text
scene_001/nominal/frame_metadata.csv
scene_001/nominal/lidar_labels/
```

### Pseudo-reflectance looks too similar across materials

Use:

```text
View = Pseudo-reflectance
Pseudo scale = fixed
```

`percentile` scaling improves local contrast but rescales each frame independently, so it is less suitable for direct nominal/rain/snow comparison.

### Run Viewer starts but no dataset is saved

Check:

```text
Save dataset = enabled
Dataset root = valid writable folder
Scene ID     = non-empty
Scenario     = non-empty
```

---

## 8. What to Show in a Demo

For a short artifact demo, show:

1. **Run Viewer** with `View = Material classes`.
2. **Run Viewer** with `View = Pseudo-reflectance`.
3. The top-down LiDAR view and material legend.
4. **Dataset Analyzer** trajectory point cloud.
5. Per-class pseudo-reflectance plot in **Material Response**.
6. CARLA default intensity versus MatSense pseudo-reflectance plot.
