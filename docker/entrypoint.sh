#!/bin/bash
set -e

# ===========================================
# X11 SERVER CONFIGURATION
# ===========================================
export DISPLAY=${DISPLAY:-":0"}
export QT_QPA_PLATFORM=${QT_QPA_PLATFORM:-"xcb"}
export QT_X11_NO_MITSHM=${QT_X11_NO_MITSHM:-"1"}

# Allow access to the X server
if command -v xhost &> /dev/null; then
    xhost +local:root
fi

# ===========================================
# SOURCE ROS2 ENVIRONMENT
# ===========================================
if [ -f /opt/ros/humble/setup.bash ]; then
    echo "Sourcing ROS2 Humble environment..."
    source /opt/ros/humble/setup.bash
else
    echo "Error: ROS2 Humble setup.bash file not found!"
    exit 1
fi

if [ -f /workspace/install/setup.bash ]; then
    echo "Sourcing custom workspace environment..."
    source /workspace/install/setup.bash
else
    echo "Error: ROS2 workspace has not been built correctly!"
    exit 1
fi

# ===========================================
# CHECK REQUIRED PACKAGES
# ===========================================
echo "Checking for installed packages..."
required_packages=("aurora_ndi_ros2_driver" "eye_in_hand_calibration" "usb_cam")
for pkg in "${required_packages[@]}"; do
    if ros2 pkg list 2>/dev/null | grep -q "^${pkg}$"; then
        echo "✓ Package '$pkg' found"
    else
        echo "✗ Package '$pkg' missing"
    fi
done

# ===========================================
# SETUP PERMANENT BASH ENVIRONMENT
# ===========================================
echo "Setting up permanent ROS2 environment..."
if ! grep -q "ROS2 Environment Setup" /root/.bashrc; then
    cat >> /root/.bashrc << 'EOF'

# ROS2 Environment Setup
if [ -f /opt/ros/humble/setup.bash ]; then
    source /opt/ros/humble/setup.bash
fi
if [ -f /workspace/install/setup.bash ]; then
    source /workspace/install/setup.bash
fi

# Set ROS2 environment variables
export ROS_DOMAIN_ID=0
export ROS_LOCALHOST_ONLY=0
EOF
fi

# ===========================================
# LAUNCH CAMERA NODE
# ===========================================
PARAMS_FILE_480=${PARAMS_FILE_480:-"/usr/local/bin/camera_params_480p.yaml"}
PARAMS_FILE_1080=${PARAMS_FILE_1080:-"/usr/local/bin/camera_params_1080p.yaml"}

if [ ! -f "${PARAMS_FILE_1080}" ]; then
    echo "Error: Camera params file not found: ${PARAMS_FILE_1080}"
    exit 1
fi

echo "=========================================="
echo "Starting endoscope camera driver..."
echo "=========================================="
ros2 run usb_cam usb_cam_node_exe --ros-args \
    --params-file "${PARAMS_FILE_1080}" \
    -r image_raw:="/endoscope/image_raw" \
    -r camera_info:="/endoscope/camera_info" \
    -r image_raw/compressed:="/endoscope/image_raw/compressed" \
    > /dev/null 2>&1 &

CAMERA_PID=$!
echo "Camera node started (PID: $CAMERA_PID)"

# Wait for camera to initialize
sleep 3

# ===========================================
# LAUNCH AURORA TRACKING SYSTEM
# ===========================================
echo "=========================================="
echo "Starting Aurora tracking system..."
echo "=========================================="
ros2 launch aurora_ndi_ros2_driver aurora_tracking.launch.py > /dev/null 2>&1 &

AURORA_PID=$!
echo "Aurora node started (PID: $AURORA_PID)"

# Wait for Aurora to initialize
sleep 2

# ===========================================
# LAUNCH RQT IMAGE VIEW
# ===========================================
echo "=========================================="
echo "Starting rqt_image_view..."
echo "=========================================="

# Check if the compressed image topic exists
if ros2 topic list 2>/dev/null | grep -q "/endoscope/image_raw/compressed"; then
    echo "Topic /endoscope/image_raw/compressed found, launching with preset topic"
    ros2 run rqt_image_view rqt_image_view --ros-args \
        -p image_topic:="/endoscope/image_raw/compressed" \
        > /dev/null 2>&1 &
else
    echo "Topic not yet available, launching without preset (select manually from dropdown)"
    ros2 run rqt_image_view rqt_image_view > /dev/null 2>&1 &
fi

RQT_PID=$!
echo "rqt_image_view started (PID: $RQT_PID)"

# Wait for rqt to initialize
sleep 1

# ===========================================
# CLEANUP HANDLER
# ===========================================
cleanup() {
    echo ""
    echo "=========================================="
    echo "Shutting down nodes..."
    echo "=========================================="

    if [ ! -z "$CAMERA_PID" ]; then
        echo "Stopping camera node (PID: $CAMERA_PID)..."
        kill $CAMERA_PID 2>/dev/null || true
    fi

    if [ ! -z "$AURORA_PID" ]; then
        echo "Stopping Aurora node (PID: $AURORA_PID)..."
        kill $AURORA_PID 2>/dev/null || true
    fi

    if [ ! -z "$RQT_PID" ]; then
        echo "Stopping rqt_image_view (PID: $RQT_PID)..."
        kill $RQT_PID 2>/dev/null || true
    fi

    echo "Cleanup complete."
    exit 0
}

# Register cleanup handler
trap cleanup SIGINT SIGTERM

# ===========================================
# MAIN EXECUTION
# ===========================================
echo ""
echo "=========================================="
echo "All nodes started successfully!"
echo "=========================================="
echo "Camera PID: $CAMERA_PID"
echo "Aurora PID: $AURORA_PID"
echo "rqt_image_view PID: $RQT_PID"
echo ""
echo "Available topics:"
ros2 topic list | grep -E "(endoscope|aurora)" || echo "  (waiting for topics...)"
echo ""
echo "Press Ctrl+C to stop all nodes"
echo "=========================================="
echo ""

# If no arguments provided, start an interactive bash shell
if [ $# -eq 0 ]; then
    echo "Starting interactive shell with ROS2 environment..."
    exec /bin/bash
else
    # Pass control to any additional commands specified at runtime
    exec "$@"
fi