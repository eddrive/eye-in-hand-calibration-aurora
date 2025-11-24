# Resources - Configuration

ROS2 configurations and calibration parameters.

## Structure

```
resources/
├── eye_in_hand_calibration/     # Main ROS2 package
│   └── config/
│       └── params.yaml          # Calibration parameters
└── calibration/                 # Camera calibration
    ├── camera_params_*.yaml     # USB driver parameters
    └── camera_calibration_*.yaml # Camera intrinsics
```

---

## params.yaml - Calibration Parameters

### Topics & Frames

```yaml
image_topic: "/endoscope/image_raw/compressed"
aurora_topic: "/aurora/sensor0"
parent_frame: "aurora_base"
child_frame: "endo_aurora_sensor0"
```

### Chessboard

```yaml
chessboard_rows: 10              # Internal corners (not squares!)
chessboard_cols: 8
chessboard_square_size: 0.003    # 3mm
use_measured_object_points: true # Use measured 3D coordinates
measured_points_file: "..."      # Path to measured points YAML
```

### Sample Collection

```yaml
max_samples: 200                 # Images to process
final_poses: 20                  # Samples for calibration
max_error_threshold: 0.01        # Max error (m)
max_pose_age_ms: 5               # Aurora-camera sync (ms)
```

### Calibration Method

```yaml
calibration_method: 0
# 0 = TSAI (recommended)
# 1 = PARK
# 2 = HORAUD
# 3 = ANDREFF
# 4 = DANIILIDIS
```

### Quality Filtering

```yaml
max_reproj_error_filter: 0.5     # Max reprojection error (px)
min_sensor_camera_distance: 0.007 # 7mm min
max_sensor_camera_distance: 0.015 # 15mm max
```

### Spatial Diversity

```yaml
use_spatial_diversity: true
min_trans_dist_mm: 14.0          # Min distance between samples (mm)
min_rot_dist_deg: 10.0           # Min rotation (°)
target_diverse_samples: 25
```

### Stillness Check

```yaml
enable_stillness_check: false
stillness_buffer_size: 3
max_stillness_translation: 0.001  # 1mm
max_stillness_rotation: 0.017     # ~1°
```

### Non-linear Refinement

```yaml
use_nonlinear_refinement: true
refinement_max_iterations: 100
rotation_weight: 10.0
```

---

## camera_params_*.yaml - USB Driver

```yaml
usb_cam:
  ros__parameters:
    video_device: "/dev/video2"
    image_width: 1920             # 1080p
    image_height: 1080
    pixel_format: "mjpeg2rgb"
    framerate: 30.0
    camera_info_url: "file:///root/calibration/camera_calibration_fisheye_1080p.yaml"
```

**Available versions:**
- `camera_params_480p.yaml` - 640x480
- `camera_params_1080p.yaml` - 1920x1080 (default)

---

## camera_calibration_*.yaml - Intrinsics

Camera intrinsic calibration (K matrix, fisheye distortion).

**Available versions:**
- `camera_calibration_fisheye_480p.yaml`
- `camera_calibration_fisheye_1080p.yaml` (default)

---

## Tuning Guide

| Scenario | Key parameters |
|----------|----------------|
| High precision | `max_reproj_error_filter: 0.3`, `target_diverse_samples: 30+` |
| Fast collection | `max_samples: 100`, `enable_stillness_check: false` |
| Small pattern | Reduce `min_sensor_camera_distance` |
| Wide movements | Increase `min_trans_dist_mm` and `min_rot_dist_deg` |
