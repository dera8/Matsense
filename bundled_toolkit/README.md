# MatSense Runtime Components

This folder contains the runtime components used by MatSense to launch the
CARLA/pygame viewer and apply material-aware LiDAR response models.

Included files:

- `carla_material_aware_launcher_config.json`
- `scripts/carla_pygame_lidar_dataset_recorder_friendly.py`
- `configs/material_aware_tool_config.json`
- `configs/material_profiles.example.json`
- `configs/material_overrides.example.json`
- `src/material_aware_toolkit/`

Generated outputs, caches, browser profiles, and experimental
scenario-preparation utilities are not included in this public artifact
release. The Streamlit app can still be pointed to another local MatSense
checkout through the sidebar or with the `MATSENSE_TOOLKIT` environment
variable.

The Run Viewer tab still requires:

- a running CARLA server;
- a Python environment where the `carla` module is available;
- `pygame`, `numpy`, and the dashboard dependencies from `requirements.txt`.
