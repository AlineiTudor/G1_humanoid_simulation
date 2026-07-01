#!/bin/bash
# Isaac Lab Docker run script with ROS2 + Unitree support
# This version includes Unitree robot assets and ROS2

# Configuration - CUSTOMIZE THESE PATHS
IMAGE_NAME="isaac-lab-base-tactile:latest"
CONTAINER_NAME="isaaclab-ros2-unitree-$(date +%s)"

# Your project directories to mount
# UPDATE THESE PATHS to match your remote PC
PROJECT_DIR_1="/home/analog/tudor_unitree_isaaclab"
PROJECT_DIR_2="/home/analog/develop"

# Unitree assets repository (cloned from GitHub)
# This should point to where you cloned unitree_sim_isaaclab
# UPDATE THIS PATH: mkdir -p ~/unitree_workspace && cd ~/unitree_workspace && git clone https://github.com/unitreerobotics/unitree_sim_isaaclab.git
UNITREE_REPO_DIR="$HOME/tudor_unitree_isaaclab/unitree_sim_isaaclab"

echo "=========================================="
echo "Isaac Lab + ROS2 + Unitree Docker Run"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Image: $IMAGE_NAME"
echo "  Container: $CONTAINER_NAME"
echo ""

# Check if image exists
if ! docker images | grep "isaac-lab-ros2-tactile\|isaac-lab-base-tactile"; then
    echo "ERROR: ROS2 image not found: $IMAGE_NAME"
    echo ""
    echo "Please build the image first with ROS2 profile:"
    echo "  1. Edit 1_setup_isaaclab_docker.sh"
    echo "     Change: DOCKER_PROFILE=\"base\""
    echo "     To: DOCKER_PROFILE=\"ros2\""
    echo "  2. Run: ./1_setup_isaaclab_docker.sh"
    echo ""
    exit 1
fi

# Check mount directories
echo "Checking mount directories..."
MOUNT_WARNINGS=0

if [ ! -d "$PROJECT_DIR_1" ]; then
    echo "  WARNING: Directory not found: $PROJECT_DIR_1"
    MOUNT_WARNINGS=$((MOUNT_WARNINGS + 1))
fi

if [ ! -d "$PROJECT_DIR_2" ]; then
    echo "  WARNING: Directory not found: $PROJECT_DIR_2"
    MOUNT_WARNINGS=$((MOUNT_WARNINGS + 1))
fi

if [ ! -d "$UNITREE_REPO_DIR" ]; then
    echo "  ERROR: Unitree repository not found: $UNITREE_REPO_DIR"
    echo ""
    echo "Please clone the Unitree repository first:"
    echo "  mkdir -p ~/unitree_workspace"
    echo "  cd ~/unitree_workspace"
    echo "  git clone https://github.com/unitreerobotics/unitree_sim_isaaclab.git"
    echo ""
    echo "Then update UNITREE_REPO_DIR in this script to match the path."
    echo ""
    exit 1
fi

if [ $MOUNT_WARNINGS -gt 0 ]; then
    echo ""
    echo "Some project directories not found (see warnings above)."
    echo "The container will still run, but those directories won't be accessible."
    echo ""
    read -p "Continue anyway? (yes/no): " cont
    if [ "$cont" != "yes" ]; then
        exit 0
    fi
fi

echo ""
echo "Preparing to run container..."
echo ""

# Allow X server access from Docker
echo "Allowing X server connections..."
xhost +local:docker

echo ""
echo "Starting container with:"
echo "  GPU support: enabled (all GPUs)"
echo "  Display: $DISPLAY"
echo "  Network: host mode"
echo "  ROS2: Humble (auto-sourced on startup)"
echo "  Mounted volumes:"
echo "    - $PROJECT_DIR_1 -> /home/code/unitree_sim_isaaclab"
echo "    - $PROJECT_DIR_2 -> /home/code/inspire_hand_ws"
echo "    - $UNITREE_REPO_DIR -> /workspace/unitree_assets"
echo ""

# Build volume mount arguments
VOLUME_MOUNTS="-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $HOME/.Xauthority:/root/.Xauthority:rw \
  -v /usr/share/vulkan:/usr/share/vulkan:ro \
  -v /usr/share/glvnd:/usr/share/glvnd:ro"

# Add project directories if they exist
if [ -d "$PROJECT_DIR_1" ]; then
    VOLUME_MOUNTS="$VOLUME_MOUNTS -v $PROJECT_DIR_1:/workspace/tudor_unitree_isaaclab"
fi

if [ -d "$PROJECT_DIR_2" ]; then
    VOLUME_MOUNTS="$VOLUME_MOUNTS -v $PROJECT_DIR_2:/workspace/develop"
fi

# Add Unitree assets (IMPORTANT!)
if [ -d "$UNITREE_REPO_DIR" ]; then
    VOLUME_MOUNTS="$VOLUME_MOUNTS -v $UNITREE_REPO_DIR:/workspace/unitree_assets"
fi

# Run the container
# Using the same configuration as your current setup + ROS2
# Auto-source ROS2 on container start
sudo docker run --gpus all -it --rm \
  --entrypoint /bin/bash \
  --name "$CONTAINER_NAME" \
  --network host \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH \
  -e DISPLAY=$DISPLAY \
  -e XAUTHORITY=$XAUTHORITY \
  -e QT_X11_NO_MITSHM=1 \
  -e VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
  -e ACCEPT_EULA=Y \
  -e OMNI_KIT_ALLOW_ROOT=1 \
  $VOLUME_MOUNTS \
  "$IMAGE_NAME" \
  -c "echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc && source ~/.bashrc && echo '' && echo '=== ROS2 Humble sourced ===' && echo 'Isaac Lab: /workspace/isaaclab' && echo 'Unitree assets: /workspace/unitree_assets' && echo 'Your projects: /home/code/' && echo '' && /bin/bash"

# Note: Container will be removed automatically when you exit (--rm flag)
echo ""
echo "Container exited."

