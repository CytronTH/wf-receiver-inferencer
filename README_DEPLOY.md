# AI TCP Receiver (Multi-Model GUI)

แพ็กเกจนี้ประกอบด้วยโค้ดรัน AI Inference และโมเดล Hailo ที่ถูกปรับแต่งให้รันบน Raspberry Pi พร้อมๆ กันหลายโมเดล (Multi-Process) ผ่าน GUI

## Project Structure

```text
ai_receiver_deploy/
├── ai_receiver_gui.py             # Main PyQt6 interface for multi-camera multi-model monitoring
├── ai_receiver_tcp.py             # Headless version (optional)
├── inference_hailo_rpi_optimized.py # optimized Hailo VDevice wrappers with Multi-Process support
├── calibrate_threshold_hailo.py   # Hailo threshold calibration tool
├── run_receiver.sh                # Shell script to map HEF files and start GUI
├── install_dependencies.sh        # Setup script for quick Raspberry Pi environment creation
├── models/                        # Pre-compiled .hef models directory
└── README.md                      # This file
```

## System Requirements
1. **Hardware:** Raspberry Pi 5 with **Raspberry Pi AI Kit** (Hailo-8L NPU attached via PCIe).
2. **OS:** Raspberry Pi OS (64-bit) Bookworm or newer.
3. **Environment:** Must be run in a desktop environment (X11/Wayland) for GUI rendering.

---

## Installation Guide (Fresh Pi)

1. Clone or copy this repository to your Raspberry Pi.
2. Open a Terminal inside the project directory.
3. Run the dependency installation script:
   ```bash
   chmod +x install_dependencies.sh
   ./install_dependencies.sh
   ```
   *(This will install the Hailo Drivers, PyQt6, OpenCV, and automatically enable the Hailo `multi_process_service`.)*

---

## Running the Real-Time Inference Receiver

Execute the startup script:
```bash
./run_receiver.sh
```
The Graphical User Interface will appear, waiting to receive incoming camera streams via TCP on ports **8080** and **8081**.

## Troubleshooting

- **Error: `HAILO_OUT_OF_PHYSICAL_DEVICES (74)`**
  The Hailo Multi-Process multiplexer service is inactive. Run:
  `sudo systemctl restart hailort`
- **Error: `qt.qpa.xcb: could not connect to display`**
  You are trying to run the GUI script over SSH without X11 forwarding. Ensure you are executing this directly on the Raspberry Pi Desktop or prepend `DISPLAY=:0`.
