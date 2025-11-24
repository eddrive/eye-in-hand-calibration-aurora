# Docker Setup

Containerized build and deployment for the eye-in-hand calibration system.

## Quick Start

```bash
# Build
docker build -t eye_in_hand_calibration:latest .

# Build (force rebuild)
docker build --build-arg CACHEBUST=$(date +%s) -t eye_in_hand_calibration:latest .

# Start container
docker-compose up

# Interactive shell
docker exec -it eye_in_hand_calib bash

# Stop
docker-compose down
```

## Structure

| File | Description |
|------|-------------|
| `Dockerfile` | Multi-stage build with ROS2 Humble, OpenCV, Ceres |
| `docker-compose.yml` | Container orchestration with device mapping |
| `entrypoint.sh` | Auto-start camera + Aurora + rqt |

## Device Mapping

Edit `docker-compose.yml` for your devices:

```yaml
devices:
  - "/dev/video2:/dev/video2"   # Endoscope camera
  - "/dev/ttyUSB0:/dev/ttyUSB0" # Aurora serial
```

## Useful Commands

```bash
# Container logs
docker logs -f eye_in_hand_calib

# Rebuild without cache
docker-compose build --no-cache

# Check devices
ls -la /dev/video* /dev/ttyUSB*

# GUI debug
xhost +local:root
```

## Notes

- **Network**: `host` mode for ROS2 communication
- **Privileged**: required for hardware access
- **X11**: automatic forwarding for GUI (rqt, rviz)
