#!/bin/bash
echo "=========================================================="
echo " Installing dependencies for Hailo AI Receiver GUI"
echo "=========================================================="

# 1. Update package list
echo "[1/3] Updating package list..."
sudo apt update

# 2. Install Hailo drivers and Python libraries
# 'hailo-all' installs the Hailo PCIe driver and HailoRT stack for the AI Kit
echo "[2/3] Installing Hailo software, PyQt6, and OpenCV..."
sudo apt install -y hailo-all python3-pyqt6 python3-opencv python3-numpy

# 3. Enable HailoRT Multi-Process Service
echo "[3/3] Setting up HailoRT multi-process service..."
sudo systemctl enable hailort.service
sudo systemctl start hailort.service

echo "=========================================================="
echo " Setup Complete! 🎉"
echo " You can now run the system using:"
echo " ./run_receiver.sh"
echo "=========================================================="
