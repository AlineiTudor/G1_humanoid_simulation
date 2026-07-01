#!/bin/bash
# Complete setup script for Isaac Lab Docker with GUI and tactile sensors
# Run this on the remote PC with NVIDIA GPU

set -e  # Exit on error

echo "=========================================="
echo "Isaac Lab Docker Setup"
echo "=========================================="
echo ""
echo "This script will:"
echo "  1. Check prerequisites (GPU, Docker, disk space)"
echo "  2. Clone Isaac Lab repository (main branch)"
echo "  3. Build Docker image with Isaac Sim 5.x + Isaac Lab"
echo "  4. Image will include isaaclab_contrib (tactile sensors)"
echo "  5. Your old unitree-sim:latest image stays untouched"
echo ""

# Configuration
WORKSPACE_DIR="$HOME/tudor_unitree_isaaclab"
REPO_URL="https://github.com/isaac-sim/IsaacLab.git"
IMAGE_SUFFIX="tactile"
DOCKER_PROFILE="ros2"  # Can be "base" or "ros2"

echo "Configuration:"
echo "  Workspace: $WORKSPACE_DIR"
echo "  Docker profile: $DOCKER_PROFILE"
echo "  Image suffix: $IMAGE_SUFFIX"
echo "  Result: isaac-lab-$DOCKER_PROFILE-$IMAGE_SUFFIX:latest"
echo ""

read -p "Continue? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo "Setup cancelled."
    exit 0
fi

echo ""
echo "=========================================="
echo "STEP 1: Checking Prerequisites"
echo "=========================================="
echo ""

# Check NVIDIA driver
echo "[1/6] Checking NVIDIA driver..."
if command -v nvidia-smi &> /dev/null; then
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    echo "  Found NVIDIA driver: $DRIVER_VERSION"

    # Check driver version (need 525+)
    DRIVER_MAJOR=$(echo $DRIVER_VERSION | cut -d'.' -f1)
    if [ "$DRIVER_MAJOR" -lt 525 ]; then
        echo "  WARNING: Driver version $DRIVER_VERSION might be too old for Isaac Sim 5.x"
        echo "  Recommended: 525+"
        read -p "  Continue anyway? (yes/no): " cont
        if [ "$cont" != "yes" ]; then
            exit 1
        fi
    fi
else
    echo "  ERROR: nvidia-smi not found. NVIDIA driver not installed?"
    exit 1
fi

# Check Docker
echo "[2/6] Checking Docker..."
if ! command -v docker &> /dev/null; then
    echo "  ERROR: Docker not found. Please install Docker first."
    exit 1
fi
DOCKER_VERSION=$(docker --version | grep -oP '\d+\.\d+\.\d+' | head -1)
echo "  Found Docker: $DOCKER_VERSION"

# Check NVIDIA Container Toolkit
echo "[3/6] Checking NVIDIA Container Toolkit..."
if docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo "  NVIDIA Container Toolkit is working"
else
    echo "  ERROR: NVIDIA Container Toolkit not working properly"
    echo "  Install with: sudo apt-get install -y nvidia-container-toolkit"
    exit 1
fi

# Check disk space
echo "[4/6] Checking disk space..."
AVAILABLE_SPACE=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
echo "  Available space: ${AVAILABLE_SPACE}GB"
if [ "$AVAILABLE_SPACE" -lt 50 ]; then
    echo "  WARNING: Less than 50GB available. Build might fail."
    echo "  Recommended: 50GB+ free space"
    read -p "  Continue anyway? (yes/no): " cont
    if [ "$cont" != "yes" ]; then
        exit 1
    fi
fi

# Check git
echo "[5/6] Checking git..."
if ! command -v git &> /dev/null; then
    echo "  ERROR: git not found. Install with: sudo apt-get install git"
    exit 1
fi
echo "  git is installed"

# Check Python
echo "[6/6] Checking Python..."
if ! command -v python3 &> /dev/null; then
    echo "  ERROR: python3 not found"
    exit 1
fi
PYTHON_VERSION=$(python3 --version)
echo "  Found: $PYTHON_VERSION"

echo ""
echo "All prerequisites met!"
echo ""

echo "=========================================="
echo "STEP 2: Setting Up Workspace"
echo "=========================================="
echo ""

# Create workspace directory
if [ -d "$WORKSPACE_DIR" ]; then
    echo "Workspace directory already exists: $WORKSPACE_DIR"
    echo "  Options:"
    echo "    1. Remove and recreate (fresh clone)"
    echo "    2. Use existing (skip clone)"
    echo "    3. Cancel"
    read -p "  Choose (1/2/3): " choice
    case $choice in
        1)
            echo "  Removing existing workspace..."
            rm -rf "$WORKSPACE_DIR"
            ;;
        2)
            echo "  Using existing workspace"
            ;;
        3)
            echo "Cancelled."
            exit 0
            ;;
        *)
            echo "Invalid choice. Cancelled."
            exit 1
            ;;
    esac
fi

if [ ! -d "$WORKSPACE_DIR/IsaacLab" ]; then
    echo "Creating workspace: $WORKSPACE_DIR"
    mkdir -p "$WORKSPACE_DIR"

    echo ""
    echo "=========================================="
    echo "STEP 3: Cloning Isaac Lab Repository"
    echo "=========================================="
    echo ""

    echo "Cloning from: $REPO_URL"
    echo "This may take a few minutes..."
    git clone "$REPO_URL" "$WORKSPACE_DIR/IsaacLab"

    cd "$WORKSPACE_DIR/IsaacLab"
    CURRENT_BRANCH=$(git branch --show-current)
    LATEST_COMMIT=$(git log -1 --oneline)

    echo ""
    echo "Clone complete!"
    echo "  Branch: $CURRENT_BRANCH"
    echo "  Latest commit: $LATEST_COMMIT"
else
    echo "Isaac Lab repository already exists, skipping clone"
    cd "$WORKSPACE_DIR/IsaacLab"
fi

echo ""
echo "=========================================="
echo "STEP 4: Building Docker Image"
echo "=========================================="
echo ""

echo "Build configuration:"
echo "  Profile: $DOCKER_PROFILE"
echo "  Suffix: $IMAGE_SUFFIX"
echo "  Image name: isaac-lab-$DOCKER_PROFILE-$IMAGE_SUFFIX:latest"
echo ""
echo "This will:"
echo "  - Download Isaac Sim 5.x (~10GB)"
echo "  - Build Isaac Lab from source"
echo "  - Install isaaclab_contrib (tactile sensors)"
echo "  - Configure GUI support"
echo ""
echo "Estimated time: 20-45 minutes"
echo "Estimated download: ~10GB"
echo ""

read -p "Start build now? (yes/no): " build_confirm
if [ "$build_confirm" != "yes" ]; then
    echo ""
    echo "Build skipped."
    echo "To build later, run:"
    echo "  cd $WORKSPACE_DIR/IsaacLab"
    echo "  ./docker/container.py build $DOCKER_PROFILE --suffix $IMAGE_SUFFIX"
    exit 0
fi

echo ""
echo "Starting build..."
echo "Build log will be saved to: build.log"
echo ""

# Run the build
if [ -f "docker/container.py" ]; then
    # Build and log output
    ./docker/container.py build "$DOCKER_PROFILE" --suffix "$IMAGE_SUFFIX" 2>&1 | tee build.log
    BUILD_EXIT_CODE=${PIPESTATUS[0]}

    if [ $BUILD_EXIT_CODE -eq 0 ]; then
        echo ""
        echo "=========================================="
        echo "BUILD SUCCESSFUL!"
        echo "=========================================="
    else
        echo ""
        echo "=========================================="
        echo "BUILD FAILED!"
        echo "=========================================="
        echo "Check build.log for details"
        exit 1
    fi
else
    echo "ERROR: docker/container.py not found"
    echo "Repository structure may have changed"
    exit 1
fi

echo ""
echo "=========================================="
echo "STEP 5: Verification"
echo "=========================================="
echo ""

echo "Checking created image..."
EXPECTED_IMAGE="isaac-lab-$DOCKER_PROFILE-$IMAGE_SUFFIX"
if docker images | grep -q "$EXPECTED_IMAGE"; then
    IMAGE_INFO=$(docker images "$EXPECTED_IMAGE:latest" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}")
    echo "$IMAGE_INFO"
    echo ""
    echo "Image successfully created!"
else
    echo "WARNING: Expected image not found: $EXPECTED_IMAGE"
    echo "Available Isaac Lab images:"
    docker images | grep isaac-lab || echo "  None found"
fi

echo ""
echo "=========================================="
echo "SETUP COMPLETE!"
echo "=========================================="
echo ""
echo "Summary:"
echo "  Workspace: $WORKSPACE_DIR/IsaacLab"
echo "  Docker image: $EXPECTED_IMAGE:latest"
echo "  Build log: $WORKSPACE_DIR/IsaacLab/build.log"
echo ""
echo "Next steps:"
echo "  1. Use the run script: ./2_run_isaaclab_docker.sh"
echo "  2. Or use official launcher:"
echo "     cd $WORKSPACE_DIR/IsaacLab"
echo "     ./docker/container.py start $DOCKER_PROFILE --suffix $IMAGE_SUFFIX"
echo ""
echo "Your old image 'unitree-sim:latest' is untouched."
echo ""
echo "=========================================="

