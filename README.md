# Eye-in-Hand Calibration with Aurora Tracking System

A ROS2-based calibration system for determining the transformation between an Aurora electromagnetic tracking sensor and an endoscopic camera mounted on the same instrument (eye-in-hand configuration).

## Overview

This package solves the **AX=XB** hand-eye calibration problem where:
- **A**: Relative motion of the Aurora sensor
- **X**: Unknown transformation from sensor to camera (what we want to find)
- **B**: Relative motion of the camera (estimated via chessboard detection)

The system automatically collects calibration samples, applies advanced filtering based on quality metrics, performs iterative refinement, and outputs the calibrated transformation with comprehensive error analysis.

## Features

- **Automatic Sample Collection**: Multi-threaded image processing with real-time Aurora pose synchronization
- **Advanced Filtering Pipeline**:
  - Reprojection error filtering
  - Sensor-camera distance validation
  - Movement coherence checking (translation/rotation ratios)
- **Iterative Refinement**: Removes worst sample pairs based on AX≈XB consistency error
- **Multiple Calibration Methods**: Tsai-Lenz, Park-Martin, Horaud-Dornaika, Andreff, Daniilidis
- **Comprehensive Error Metrics**:
  - AX=XB errors (relative motion consistency)
  - Absolute prediction errors (direct pose comparison)
- **Fisheye Camera Support**: Full support for fisheye lens models with proper undistortion
- **Measured Object Points**: Uses Aurora-measured 3D chessboard corners (accounts for board warping)

## System Requirements

- Docker and Docker Compose
- NVIDIA GPU with CUDA support (optional, for accelerated processing)
- Aurora electromagnetic tracking system
- Endoscopic camera with fisheye or pinhole lens

## Project Structure

```
eye-in-hand-calibration-aurora/
├── docker/
│   ├── Dockerfile              # Multi-stage build for ROS2 Humble
│   ├── compose.yaml            # Docker Compose configuration
│   └── entrypoint.sh           # Container startup script
├── resources/
│   └── eye_in_hand_calibration/
│       ├── config/
│       │   └── params.yaml     # Calibration parameters
│       ├── include/            # C++ headers
│       ├── src/                # C++ implementation
│       ├── launch/             # ROS2 launch files
│       ├── nodes/              # ROS2 node executables
│       └── CMakeLists.txt
├── scripts/
│   └── eyehand_from_yaml.py    # Offline calibration analysis tool
└── README.md
```

## Building the Docker Image

Build the Docker image from the repository root:

```bash
docker build -f docker/Dockerfile \
  --build-arg CACHEBUST=$(date +%s) \
  -t eye_in_hand_calibration:latest .
```

**Build Arguments:**
- `CACHEBUST`: Forces fresh build by bypassing Docker cache (useful after code changes)

The Dockerfile uses a multi-stage build:
1. **Base stage**: ROS2 Humble with essential dependencies
2. **Build stage**: Compiles the calibration package
3. **Runtime stage**: Minimal runtime environment

## Running the Calibration System

### 1. Start the Container

From the repository root directory:

```bash
docker compose -f docker/compose.yaml up -d
```

This starts the container with:
- Shared X11 display for visualization
- Network host mode for ROS2 communication
- Volume mounts for workspace and configuration files

### 2. Enter the Container

```bash
docker exec -it eye_in_hand_calibration bash
```

### 3. Launch the Calibration Node

Inside the container:

```bash
ros2 launch eye_in_hand_calibration hand_eye_calibration.launch.py
```

The system will:
1. Subscribe to `/endoscope/image_raw/compressed` for camera images
2. Listen to TF transforms for Aurora sensor pose (`aurora_base` → `endo_aurora`)
3. Automatically collect samples when chessboard is detected
4. Perform calibration once `max_samples` is reached
5. Save results to `/workspace/src/eye_in_hand_calibration/output/`

## Configuration

Edit `resources/eye_in_hand_calibration/config/params.yaml` to customize:

### Topics & Frames
```yaml
image_topic: "/endoscope/image_raw/compressed"
parent_frame: "aurora_base"        # Aurora reference frame
child_frame: "endo_aurora"         # Aurora sensor frame
```

### Camera Calibration
```yaml
camera_calibration_file: "/root/calibration/camera_calibration_fisheye_1080p.yaml"
```

Generate intrinsic calibration with:
```bash
ros2 run camera_calibration cameracalibrator --size 8x10 --square 0.003
```

### Chessboard Pattern
```yaml
chessboard_rows: 10                # Internal corners (not squares!)
chessboard_cols: 8
chessboard_square_size: 0.003      # 3mm squares

use_measured_object_points: true
measured_points_file: "/workspace/src/eye_in_hand_calibration/config/chess_11x9_3mm_inter.yaml"
```

**Important**: `chessboard_rows` and `chessboard_cols` refer to internal corners, not the number of squares.

### Collection Parameters
```yaml
max_samples: 50                    # Total samples to collect
max_pose_age_ms: 20                # Max Aurora-image timestamp difference (ms)
aurora_buffer_size: 5000           # Circular buffer for Aurora poses
```

### Filtering Parameters
```yaml
# Diversity filters (applied during collection)
min_movement_threshold: 0.01       # 10mm minimum translation
min_rotation_threshold: 0.26       # ~15° minimum rotation

# Advanced filters (applied before calibration)
max_reproj_error_filter: 0.8       # Max reprojection error (pixels)
max_sensor_camera_distance: 0.020  # Max sensor-camera distance (20mm)
max_movement_ratio: 2.3            # Max ratio between sensor/camera movement
max_rotation_diff_deg: 25.0        # Max rotation difference (degrees)
```

### Iterative Refinement
```yaml
use_iterative_refinement: true
target_pairs: 20                   # Target number of sample pairs
max_refinement_iterations: 50      # Max iterations
```

### Calibration Method
```yaml
calibration_method: 0              # 0=TSAI, 1=PARK, 2=HORAUD, 3=ANDREFF, 4=DANIILIDIS
```

**Method Comparison:**
- **TSAI**: Fast, robust, general-purpose
- **PARK**: Slower, potentially more accurate
- **HORAUD**: Simultaneous rotation and translation estimation
- **ANDREFF**: Dual quaternion approach
- **DANIILIDIS**: Rotation-only method (good for minimal translation)

### Output Configuration
```yaml
save_result: true
result_frame_id: "endo_aurora"     # Source frame
target_frame_id: "endo_optical"    # Target frame (camera optical center)
```

## Output Files

All results are saved to `/workspace/src/eye_in_hand_calibration/output/` with timestamps:

### 1. Calibration Result
**File**: `eye_in_hand_calibration_YYYYMMDD_HHMMSS.yaml`

Contains:
- 4×4 transformation matrix (sensor → camera)
- Translation vector and rotation (quaternion + Euler angles)
- Error metrics (AX=XB errors: min, max, average)
- Method used and number of samples
- Frame IDs for TF tree integration

### 2. Collected Samples
**File**: `collected_samples_YYYYMMDD_HHMMSS.yaml`

Contains all samples with:
- Sample ID and timestamp
- Sensor pose (4×4 matrix)
- Camera pose (4×4 matrix)
- Reprojection error
- Distance to target

### 3. Selected Pose Pairs
**File**: `selected_pose_pairs_YYYYMMDD_HHMMSS.yaml`

Contains the final samples used for calibration (post-filtering and refinement).

## Error Metrics Explained

The system reports two types of errors:

### 1. AX=XB Errors (Relative Motion Consistency)
Measures how well the transformation X satisfies the constraint equation for relative motions:
- **Formula**: For each pair (i,j): error = ||AX - XB||
- **Interpretation**: Consistency of X across all sample pairs
- **Units**: Meters (Euclidean norm)
- **Count**: C(n,2) pairs for n samples (e.g., 55 pairs for 11 samples)

### 2. Absolute Errors (Direct Prediction Comparison)
Measures the accuracy of predicting camera pose from sensor pose:
- **Formula**: T_predicted = T_sensor × X, compare with T_measured
- **Rotation Error**: Angle between predicted and measured rotation (degrees)
- **Translation Error**: ||t_measured - t_predicted|| (mm)
- **Count**: One error per sample

**Statistics Reported**: min, median, max, mean, std, rms

## Offline Analysis Tool

Use the Python script to re-analyze saved calibration data:

```bash
python scripts/eyehand_from_yaml.py
```

Features:
- Load samples from YAML file
- Try different calibration methods
- Apply custom filtering thresholds
- Visualize sample distribution
- Generate detailed error reports

## Docker Commands Reference

### Build
```bash
# Fresh build (bypass cache)
docker build -f docker/Dockerfile --build-arg CACHEBUST=$(date +%s) -t eye_in_hand_calibration:latest .

# Standard build (use cache)
docker build -f docker/Dockerfile -t eye_in_hand_calibration:latest .
```

### Start/Stop
```bash
# Start container
docker compose -f docker/compose.yaml up -d

# Stop container
docker compose -f docker/compose.yaml down

# Restart container
docker compose -f docker/compose.yaml restart
```

### Enter Container
```bash
# Interactive bash shell
docker exec -it eye_in_hand_calibration bash

# Execute single command
docker exec -it eye_in_hand_calibration ros2 topic list
```

### Copy Files from Container
```bash
# Copy calibration results to host
docker cp eye_in_hand_calibration:/workspace/src/eye_in_hand_calibration/output/ ~/Desktop/calibration_results/
```

### View Logs
```bash
# Follow container logs
docker compose -f docker/compose.yaml logs -f

# View specific service logs
docker logs eye_in_hand_calibration
```

## ROS2 Topics and TF Frames

### Subscribed Topics
- `/endoscope/image_raw/compressed` (sensor_msgs/CompressedImage)

### Published TF Frames
After calibration, you can publish the result to TF:
```bash
ros2 run tf2_ros static_transform_publisher \
  x y z qx qy qz qw \
  endo_aurora endo_optical
```

Use the values from the calibration YAML file.

### TF Frame Hierarchy
```
aurora_base                    # Aurora tracking volume reference
    └── endo_aurora            # Aurora sensor on endoscope
            └── endo_optical   # Camera optical center (calibrated)
```

## Best Practices

### Sample Collection
1. **Coverage**: Move the endoscope to cover diverse poses (translations and rotations)
2. **Stability**: Hold steady for ~0.5s when collecting each sample
3. **Lighting**: Ensure good, consistent lighting on the chessboard
4. **Distance**: Keep chessboard within focus range (typically 50-150mm for endoscopes)
5. **Angles**: Avoid extreme viewing angles (>45° from normal)

### Chessboard Setup
1. Print on rigid material (foam board, acrylic)
2. Ensure flatness (use measured object points if warped)
3. High contrast (black squares, white background)
4. Measure square size precisely with calipers

### Parameter Tuning
1. Start with default parameters
2. Check filtering logs to see how many samples are rejected
3. Relax thresholds if too few samples pass
4. Tighten thresholds if calibration error is high

### Quality Validation
- **AX=XB errors** should be < 50mm for surgical applications
- **Absolute rotation errors** should be < 10°
- **Absolute translation errors** should be < 15mm
- If errors are high, collect more diverse samples

## Algorithm Details

### Calibration Pipeline
1. **Collection**: Multi-threaded image processing with Aurora synchronization (±20ms)
2. **Filter 1**: Reprojection error < 0.8 pixels
3. **Filter 2**: Sensor-camera distance < 20mm (validates rigid mounting)
4. **Filter 3**: Movement coherence (ratio < 2.3, rotation diff < 25°)
5. **Refinement**: Iteratively remove worst pairs based on AX=XB error
6. **Calibration**: Solve AX=XB using OpenCV's `calibrateHandEye()`
7. **Validation**: Compute absolute prediction errors

### Synchronization
Aurora poses are stored in a circular buffer. For each image:
1. Find Aurora pose with closest timestamp
2. Reject if time difference > `max_pose_age_ms`
3. Ensures accurate pose-image correspondence

### Iterative Refinement
```
while num_pairs > target_pairs:
    1. Calibrate with current samples
    2. Compute AX=XB error for each consecutive pair
    3. Remove sample from worst pair
    4. Detect outliers (samples appearing in multiple worst pairs)
    5. Repeat until target reached
```

## Dependencies

### ROS2 Packages
- `rclcpp` - ROS2 C++ client library
- `sensor_msgs` - Image messages
- `geometry_msgs` - Pose messages
- `tf2_ros` - Transform library
- `cv_bridge` - OpenCV-ROS bridge

### Libraries
- **OpenCV 4.5+** - Computer vision (PnP, hand-eye calibration)
- **Eigen3** - Linear algebra
- **yaml-cpp** - YAML parsing

### Python (for offline tool)
- NumPy - Numerical computations
- OpenCV - Calibration algorithms
- Matplotlib - Visualization
- PyYAML - YAML I/O

## References

1. Tsai, R. Y., & Lenz, R. K. (1989). "A new technique for fully autonomous and efficient 3D robotics hand/eye calibration"
2. Park, F. C., & Martin, B. J. (1994). "Robot sensor calibration: solving AX=XB on the Euclidean group"
3. Horaud, R., & Dornaika, F. (1995). "Hand-eye calibration"
4. Andreff, N., et al. (1999). "On-line hand-eye calibration"
5. Daniilidis, K. (1999). "Hand-eye calibration using dual quaternions"
