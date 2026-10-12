# Desktop image for MNTFields: ROS 2 Jazzy + CUDA PyTorch.
# Runs SLAM Toolbox, antfields training/planning and RViz offboard; the robot
# only runs its own driver stack and talks to this container over ROS topics.
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

ARG WITH_SIM=0
# RTX 4070 = sm_89; used when compiling torch_kdtree.
ARG TORCH_CUDA_ARCH_LIST=8.9

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

# ROS 2 apt source (official ros2-apt-source package).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg git software-properties-common \
    && add-apt-repository -y universe \
    && ROS_APT_SOURCE_VERSION=$(curl -fsSL https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest \
        | grep -F '"tag_name"' | awk -F'"' '{print $4}') \
    && curl -fsSL -o /tmp/ros2-apt-source.deb \
        "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo $VERSION_CODENAME)_all.deb" \
    && apt-get install -y /tmp/ros2-apt-source.deb \
    && rm -f /tmp/ros2-apt-source.deb \
    && rm -rf /var/lib/apt/lists/*

# ROS 2 Jazzy + SLAM Toolbox + Python deps that must share apt's numpy with rclpy.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-jazzy-desktop \
        ros-jazzy-slam-toolbox \
        ros-jazzy-sensor-msgs-py \
        ros-jazzy-rmw-fastrtps-cpp \
        ros-jazzy-rmw-cyclonedds-cpp \
        python3-colcon-common-extensions \
        python3-pip \
        python3-scipy \
        python3-skimage \
        python3-networkx \
        python3-matplotlib \
        python3-opencv \
        python3-yaml \
        python3-pil \
    && rm -rf /var/lib/apt/lists/*

# Optional: Clearpath Jazzy simulator (pulls in Gazebo Harmonic; large).
# Prefer docker/Dockerfile.sim (compose service "sim"): it layers the simulator
# on top of this image, so building it does not invalidate the PyTorch layers.
RUN if [ "$WITH_SIM" = "1" ]; then \
        curl -fsSL https://packages.clearpathrobotics.com/public.key \
            | gpg --dearmor -o /usr/share/keyrings/clearpath-archive-keyring.gpg \
        && echo "deb [signed-by=/usr/share/keyrings/clearpath-archive-keyring.gpg] https://packages.clearpathrobotics.com/stable/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) main" \
            > /etc/apt/sources.list.d/clearpath.list \
        && apt-get update && apt-get install -y --no-install-recommends ros-jazzy-clearpath-simulator \
        && rm -rf /var/lib/apt/lists/*; \
    fi

# pip packages live in a venv that can see apt's packages. Upgrades that a pip
# package needs (psutil, blinker, ...) land in the venv instead of pip trying
# to uninstall Debian-owned files. numpy is pinned to apt's version so the
# numpy rclpy was built against is the one everything uses.
RUN apt-get update && apt-get install -y --no-install-recommends python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv --system-site-packages /opt/venv \
    && python3 -c "import numpy; print(f'numpy=={numpy.__version__}')" > /etc/pip-constraints.txt
ENV PIP_CONSTRAINT=/etc/pip-constraints.txt
# ROS entry points run under /usr/bin/python3 (colcon uses the system
# interpreter), so expose the venv through PYTHONPATH rather than PATH.
ENV PYTHONPATH=/opt/venv/lib/python3.12/site-packages

RUN /opt/venv/bin/pip install --index-url https://download.pytorch.org/whl/cu128 torch

RUN /opt/venv/bin/pip install open3d \
    && python3 -c "import open3d as o3d; p = o3d.geometry.PointCloud(); p.estimate_normals(); print('open3d', o3d.__version__)"

# GPU k-d tree used by Region (built against the torch above).
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
RUN /opt/venv/bin/pip install --no-build-isolation git+https://github.com/thomgrand/torch_kdtree \
    && python3 -c "import torch_kdtree; print('torch_kdtree ok')"

# Meshes referenced by the Jackal's robot_description (package://...), so RViz
# can draw the robot model.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-jazzy-clearpath-platform-description \
        ros-jazzy-clearpath-sensors-description \
    && rm -rf /var/lib/apt/lists/*

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /ws
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
