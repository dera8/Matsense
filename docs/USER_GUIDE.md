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

Common settings:

```text
Weather      nominal / rain / snow / fog
View         material / pseudo / intensity
Pseudo scale fixed / percentile
Scene ID     scene_001
Scenario     nominal / rain / snow / fog
```

Use:

```text
View = material
```

to inspect semantic material labels.

Use:

```text
View = pseudo
```

to inspect material- and weather-aware pseudo-reflectance.

Use:

```text
View = intensity
```

to inspect the raw CARLA LiDAR intensity.

### Start and Stop

Click:

```text
Start Viewer
```

to open the pygame viewer.

Click:

```text
Stop Viewer
```

to stop the last viewer process launched by the dashboard.

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
View = pseudo
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

1. **Run Viewer** with `View = material`.
2. **Run Viewer** with `View = pseudo`.
3. The top-down LiDAR view and material legend.
4. **Dataset Analyzer** trajectory point cloud.
5. Per-class pseudo-reflectance plot in **Material Response**.
6. CARLA default intensity versus MatSense pseudo-reflectance plot.
