# Use Ubuntu 22.04 as base (no need for CUDA for calibration)
FROM ubuntu:22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Rome

# Configure timezone
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# ============================================
# Install ROS2 Humble
# ============================================

RUN apt-get update && apt-get install -y \
    curl \
    gnupg2 \
    lsb-release \
    software-properties-common

# Add ROS2 repository
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS2 Humble Desktop and development tools
RUN apt-get update && apt-get install -y \
    ros-humble-desktop \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-pip \
    build-essential \
    cmake \
    nano \
    tree \
    htop \
    vim \
    git \
    wget \
    && apt-get clean

# Install ROS2 debugging and visualization tools
RUN apt-get update && apt-get install -y \
    ros-humble-rqt \
    ros-humble-rqt-common-plugins \
    ros-humble-rqt-image-view \
    ros-humble-rqt-graph \
    ros-humble-rqt-topic \
    ros-humble-tf2-tools \
    ros-humble-rqt-tf-tree \
    ros-humble-rviz2 \
    && apt-get clean

# Install camera support
RUN apt-get update && apt-get install -y \
    ros-humble-usb-cam \
    ros-humble-image-tools \
    ros-humble-image-transport \
    ros-humble-cv-bridge \
    v4l-utils \
    && apt-get clean

# Install Aurora NDI sensor dependencies
RUN apt-get update && apt-get install -y \
    ros-humble-geometry-msgs \
    ros-humble-sensor-msgs \
    ros-humble-tf2-ros \
    ros-humble-tf2-geometry-msgs \
    libserial-dev \
    python3-serial \
    && apt-get clean

# Install eye-in-hand calibration dependencies
RUN apt-get update && apt-get install -y \
    libyaml-cpp-dev \
    libeigen3-dev \
    libopencv-dev \
    libopencv-contrib-dev \
    pkg-config \
    python3-opencv \
    && apt-get clean

# ============================================
# Python packages
# ============================================

RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir \
    numpy \
    opencv-python \
    pyyaml

# ============================================
# ROS2 Workspace Setup
# ============================================

ENV COLCON_WS=/workspace
RUN mkdir -p $COLCON_WS/src

WORKDIR $COLCON_WS

# Initialize rosdep
RUN rosdep init || echo "rosdep already initialized"
RUN rosdep update

# ============================================
# Copy Packages
# ============================================


# Clone Aurora NDI ROS2 driver from GitHub
RUN git clone -b ros2-package https://github.com/eddrive/aurora_ndi_ros2_driver.git /workspace/src/aurora_ndi_ros2_driver && \
    cd /workspace/src/aurora_ndi_ros2_driver && \
    echo "=== Cloned commit ===" && \
    git log -1 --oneline && \
    echo "=== AuroraData.msg ===" && \
    cat msg/AuroraData.msg
# Copy ROS2 package
COPY resources/hand_eye_calibration $COLCON_WS/src/hand_eye_calibration

# Copy resources
COPY resources/calibration /root/calibration
COPY resources/visualize_eye_in_hand.py ${COLCON_WS}/visualize_eye_in_hand.py

# Copy test bags (optional)
COPY Bag $COLCON_WS/bags

# Install package dependencies
RUN rosdep install --ignore-src --from-paths src -y --rosdistro humble || true

# ============================================
# Build Workspace
# ============================================

RUN /bin/bash -c "source /opt/ros/humble/setup.bash && \
    colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release"

# ============================================
# Setup Environment
# ============================================

# Copy camera parameters
COPY resources/camera_params_480p.yaml /usr/local/bin/camera_params_480p.yaml
COPY resources/camera_params_1080p.yaml /usr/local/bin/camera_params_1080p.yaml

# Copy and setup entrypoint script
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Source ROS2 setup in bashrc
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc
RUN echo "source $COLCON_WS/install/setup.bash" >> /root/.bashrc

# Set the default entrypoint
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

CMD ["bash"]
