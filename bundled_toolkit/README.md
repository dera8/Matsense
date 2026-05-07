# Bundled MatSense Runtime Toolkit

This folder contains the minimal MatSense runtime needed by the Streamlit
dashboard to launch the CARLA/pygame viewer.

Included files:

- `carla_material_aware_launcher_config.json`
- `scripts/carla_pygame_lidar_dataset_recorder_friendly.py`
- `configs/material_aware_tool_config.json`
- `configs/material_profiles.example.json`
- `configs/material_overrides.example.json`
- `src/material_aware_toolkit/`

The full development toolkit is intentionally not copied here because it
contains generated datasets, caches, browser profiles, and intermediate files.
The Streamlit app can still be pointed to a full toolkit checkout through the
sidebar or with the `MATSENSE_TOOLKIT` environment variable.

The Run Viewer tab still requires:

- a running CARLA server;
- a Python environment where the `carla` module is available;
- `pygame`, `numpy`, and the dashboard dependencies from `requirements.txt`.
