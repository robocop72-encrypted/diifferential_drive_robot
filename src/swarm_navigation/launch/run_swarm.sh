#!/bin/bash
# install_layer1.sh
# Run this ONCE inside the container after copying files to your workspace.
# Usage: bash install_layer1.sh

set -e

WORKSPACE="/workspaces/differential_drive_robot"
PKG="$WORKSPACE/src/swarm_navigation"

echo "========================================"
echo "  Layer 1 - Sensor Fusion Setup Script"
echo "========================================"

# 1. pip deps (ultralytics for YOLO, opencv)
echo "[1/4] Installing Python dependencies..."
pip3 install --quiet ultralytics opencv-python-headless numpy

# 2. pre-download YOLO weights so first run is instant
echo "[2/4] Pre-downloading YOLOv8n weights..."
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" && echo "  YOLOv8n ready."

# 3. build the ROS 2 package
echo "[3/4] Building swarm_navigation package..."
cd "$WORKSPACE"
colcon build --symlink-install --packages-select swarm_navigation
source install/setup.bash

# 4. verify
echo "[4/4] Verifying nodes..."
ros2 pkg list | grep swarm_navigation && echo "  Package found OK."
ros2 run swarm_navigation sensor_fusion --ros-args -p robot_name:=test_check &
sleep 2
kill %1 2>/dev/null || true
echo "  Node launched and killed OK."

echo ""
echo "=========================================="
echo "  Setup complete!"
echo ""
echo "  To launch Layer 1:"
echo "    source /workspaces/differential_drive_robot/install/setup.bash"
echo "    ros2 launch swarm_navigation layer1_mapping.launch.py"
echo ""
echo "  Or use the alias:"
echo "    layer1"
echo "=========================================="