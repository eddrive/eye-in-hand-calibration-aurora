# Scripts - Offline Analysis

Python tools for post-processing and calibration validation.

## eyehand_from_yaml.py

Offline analysis and re-calibration from collected YAML samples.

```bash
python3 eyehand_from_yaml.py
```

**Features:**
- Interactive file selection from `data/results/`
- Sample filtering by quality metrics
- Calibration with TSAI method
- Non-linear refinement
- Error visualization and statistics

**Configuration** (edit in file):

```python
CONFIG = {
    "MAX_REPROJ_ERROR_PX": 0.6,           # Max reprojection error (px)
    "MIN_SENSOR_CAMERA_DIST_MM": 8.0,     # Min sensor-camera distance (mm)
    "MAX_SENSOR_CAMERA_DIST_MM": 15.0,    # Max sensor-camera distance (mm)
    "USE_SPATIAL_DIVERSITY": True,        # Spatial diversity filtering
    "MIN_TRANS_DIST_MM": 14.0,            # Min translation between samples (mm)
    "MIN_ROT_DIST_DEG": 10.0,             # Min rotation between samples (°)
    "TARGET_DIVERSE_SAMPLES": 40,         # Target sample count
    "METHOD": "tsai",                     # Calibration method
    "USE_NONLINEAR_REFINEMENT": True,     # Ceres refinement
}
```

## random_tsai.py

Statistical validation with random subsampling.

```bash
python3 random_tsai.py
```

**Features:**
- Load samples from YAML
- Run N iterations with random subsets
- Compute calibration stability and variance
- Generate statistical plots

**Configuration:**

```python
NUM_SAMPLES = 15      # Samples per iteration
NUM_ITERATIONS = 1000 # Number of iterations
```

## Output

Both scripts generate:
- 4x4 transformation matrix
- AX=XB errors (translation/rotation)
- Error distribution plots
