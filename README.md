# Eye-in-Hand Calibration - Aurora

ROS2 system for eye-in-hand calibration between Aurora sensor and endoscopic camera.

Solves the **AX=XB** problem where X is the sensor→camera transformation.

## Quick Start

```bash
# Build
cd docker && docker build -t eye_in_hand_calibration:latest .

# Start
docker-compose up
docker exec -it eye_in_hand_calib bash

# Calibration
ros2 launch eye_in_hand_calibration eye_in_hand_calibration.launch.py
```

## Project Structure

```
├── docker/          # Docker container
├── resources/       # ROS2 package + configurations
├── scripts/         # Offline analysis tools
├── hardware/        # CAD and chessboard patterns
└── data/            # Calibration results
```

## Documentation

| Section | Description |
|---------|-------------|
| [docker/README.md](docker/README.md) | Build, deploy, Docker commands |
| [scripts/README.md](scripts/README.md) | Python post-processing tools |
| [resources/README.md](resources/README.md) | Parameter configuration |

## Features

- Multi-threaded image processing
- Aurora synchronization ±5ms
- 5 calibration methods (TSAI, PARK, HORAUD, ANDREFF, DANIILIDIS)
- Fisheye support + measured 3D points
- Non-linear refinement (Ceres)

## Hardware Required

- Aurora NDI tracking system
- Endoscopic camera (USB)
- Chessboard pattern (see `hardware/chessboards/`)

## Main Commands

```bash
# Inside container
ros2 launch eye_in_hand_calibration eye_in_hand_calibration.launch.py

# Camera topic
ros2 topic echo /endoscope/image_raw/compressed --no-arr

# Aurora topic
ros2 topic echo /aurora/sensor0

# TF tree
ros2 run tf2_tools view_frames
```

## Output

Results saved in `resources/eye_in_hand_calibration/output/`:

| File | Content |
|------|---------|
| `eye_in_hand_calibration_*.yaml` | 4x4 matrix + errors |
| `collected_samples_*.yaml` | All collected samples |
| `selected_pose_pairs_*.yaml` | Samples used for calibration |

## Error Metrics

- **AX=XB**: transformation consistency across pose pairs
- **Absolute**: direct prediction error (rotation°, translation mm)

Surgical target: AX=XB < 50mm, rotation < 10°, translation < 15mm

## References

- Tsai & Lenz (1989) - TSAI method
- Park & Martin (1994) - Euclidean group formulation
- Horaud & Dornaika (1995) - Simultaneous estimation
