import socket
import struct
import json
import cv2
import numpy as np
import threading
import time
import os
import sys
import argparse

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QGridLayout, QScrollArea, QDoubleSpinBox, QGroupBox)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, pyqtSlot
from PyQt6.QtGui import QImage, QPixmap, QFont

from inference_hailo_rpi_optimized import HailoPatchCoreOptimized

# ======================================================================
# Network Settings
# ======================================================================
HOST = '0.0.0.0'
PORTS = {
    0: 8080,
    1: 8081,
}

IMAGE_ROUTING = {
    "masked_surface": "surface",
    "crop_0": "crop0",
    "crop_1": "crop1",
    "crop_2": "crop2",
    "crop_3": "crop3",
    "crop_4": "crop4",
    "crop_5": "crop5",
}

infer_lock = threading.Lock()

def recvall(sock, n):
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data

# ======================================================================
# TCP Worker Thread
# ======================================================================
class TCPReceiverWorker(QThread):
    # Signal: (camera_id, img_orig_bgr, img_heatmap_bgr, image_id_string, float_score)
    inference_result_signal = pyqtSignal(int, np.ndarray, np.ndarray, str, float)

    def __init__(self, port, camera_id, models):
        super().__init__()
        self.port = port
        self.camera_id = camera_id
        self.models = models
        self.running = True

    def run(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((HOST, self.port))
        srv.listen(5)
        print(f"[TCP] Cam {self.camera_id} Listening on port {self.port}")

        while self.running:
            try:
                srv.settimeout(1.0)
                try:
                    conn, addr = srv.accept()
                except socket.timeout:
                    continue
                
                srv.settimeout(None)
                self.receive_image_stream(conn, addr)
            except Exception as e:
                print(f"[TCP] Cam {self.camera_id} Server error: {e}")
                time.sleep(1)

    def receive_image_stream(self, conn, addr):
        print(f"[TCP] Cam {self.camera_id} Connected from {addr}")
        conn.settimeout(None)
        try:
            while self.running:
                # 1. Header (4 bytes)
                header_data = recvall(conn, 4)
                if not header_data: break
                json_size = struct.unpack('>L', header_data)[0]

                # 2. JSON Metadata
                json_bytes = recvall(conn, json_size)
                if not json_bytes: break
                metadata = json.loads(json_bytes.decode('utf-8'))
                image_id = metadata.get("id", "unknown")
                image_size = metadata.get("size", 0)

                # 3. JPEG Image Data
                if image_size <= 0: continue
                jpeg_data = recvall(conn, image_size)
                if not jpeg_data: break

                np_arr = np.frombuffer(jpeg_data, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if img is None: continue

                # Inference Routing
                if image_id not in IMAGE_ROUTING: continue
                model_key = IMAGE_ROUTING[image_id]
                model = self.models.get(model_key)
                if model is None: continue

                with infer_lock:
                    try:
                        score, amap = model.infer(img)
                    except Exception as e:
                        print(f"[Hailo] Error: {e}")
                        continue

                # Heatmap Post-Processing
                heatmap_resized = cv2.resize(amap, (img.shape[1], img.shape[0]))
                heatmap_vis = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min() + 1e-10)
                heatmap_uint8 = (heatmap_vis * 255).astype(np.uint8)
                heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
                
                overlay = cv2.addWeighted(img, 0.5, heatmap_color, 0.5, 0)
                
                # Emit to GUI Thread
                self.inference_result_signal.emit(self.camera_id, img, overlay, image_id, float(score))

        except Exception as e:
             pass
        finally:
             conn.close()

    def stop(self):
        self.running = False


# ======================================================================
# GUI Widgets
# ======================================================================
class InferencePanel(QGroupBox):
    def __init__(self, image_id, initial_threshold):
        super().__init__(f"  {image_id.upper()}  ")
        self.image_id = image_id
        self.setStyleSheet("""
            QGroupBox {
                border: 2px solid #555555;
                border-radius: 8px;
                margin-top: 15px;
                font-weight: bold;
                font-size: 14px;
                background-color: #2a2a2a;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #ffcc00;
            }
        """)

        layout = QVBoxLayout()
        
        # --- Config & Score Row ---
        config_layout = QHBoxLayout()
        lbl = QLabel("Threshold:")
        lbl.setStyleSheet("color: white; font-weight: bold;")
        config_layout.addWidget(lbl)

        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0001, 100.0)
        self.threshold_spin.setDecimals(4)
        self.threshold_spin.setSingleStep(0.01)
        self.threshold_spin.setValue(initial_threshold)
        self.threshold_spin.setStyleSheet("background-color: #111111; color: #00ff00; font-weight:bold; padding: 5px;")
        config_layout.addWidget(self.threshold_spin)
        
        config_layout.addStretch()
        
        self.result_lbl = QLabel("Waiting...")
        self.result_lbl.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        self.result_lbl.setStyleSheet("color: #aaaaaa; padding: 5px;")
        config_layout.addWidget(self.result_lbl)
        
        layout.addLayout(config_layout)

        # --- Image Display Row ---
        images_layout = QHBoxLayout()
        
        self.orig_view = QLabel()
        self.orig_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.orig_view.setStyleSheet("background-color: #000000; border: 1px solid #444444;")
        self.orig_view.setFixedSize(280, 280)
        self.orig_view.setText("Original Image")
        
        self.heat_view = QLabel()
        self.heat_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.heat_view.setStyleSheet("background-color: #000000; border: 1px solid #444444;")
        self.heat_view.setFixedSize(280, 280)
        self.heat_view.setText("Heatmap View")

        images_layout.addWidget(self.orig_view)
        images_layout.addWidget(self.heat_view)
        layout.addLayout(images_layout)
        
        self.setLayout(layout)

    def convert_cv_qt(self, cv_img, target_label):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        p = convert_to_Qt_format.scaled(target_label.width(), target_label.height(), Qt.AspectRatioMode.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def update_data(self, camera_id, img_orig, img_heat, score):
        # Update Images
        self.orig_view.setPixmap(self.convert_cv_qt(img_orig, self.orig_view))
        self.heat_view.setPixmap(self.convert_cv_qt(img_heat, self.heat_view))

        # Evaluate Status based on LOCAL panel threshold
        threshold = self.threshold_spin.value()
        status_text = "NG" if score > threshold else "OK"
        color = "#ff4444" if status_text == "NG" else "#44ff44"
        
        self.result_lbl.setText(f"[Cam {camera_id}] Score: {score:.4f}  |  {status_text}")
        self.result_lbl.setStyleSheet(f"color: {color}; background-color: #111111; border: 1px solid {color}; border-radius: 4px; padding: 5px;")


class AIReceiverGUI(QMainWindow):
    def __init__(self, default_threshold):
        super().__init__()
        self.default_threshold = default_threshold
        self.panels = {}
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle("AI Hailo Multi-Part Inference Monitor")
        self.resize(1300, 900)
        self.setStyleSheet("background-color: #1a1a1a; color: #ffffff;")

        central_widget = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border: none;")
        scroll.setWidget(central_widget)
        self.setCentralWidget(scroll)
        
        main_layout = QVBoxLayout(central_widget)
        
        # Header
        header = QLabel("Multi-Part Real-time AI Inference Preview")
        header.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet("color: #00e5ff; margin: 10px; padding-bottom: 10px; border-bottom: 1px solid #333;")
        main_layout.addWidget(header)

        # Dynamic Grid for Image Panels
        grid_layout = QGridLayout()
        grid_layout.setSpacing(15)
        
        parts = ["masked_surface", "crop_0", "crop_1", "crop_2", "crop_3", "crop_4", "crop_5"]
        row, col = 0, 0
        for part in parts:
            panel = InferencePanel(part, self.default_threshold)
            self.panels[part] = panel
            grid_layout.addWidget(panel, row, col)
            col += 1
            if col >= 2: # Max 2 columns
                col = 0
                row += 1
                
        main_layout.addLayout(grid_layout)
        main_layout.addStretch()

    @pyqtSlot(int, np.ndarray, np.ndarray, str, float)
    def update_inference_ui(self, camera_id, img_orig, img_heat, image_id, score):
        if image_id in self.panels:
            self.panels[image_id].update_data(camera_id, img_orig, img_heat, score)

# ======================================================================
# Main CLI Entry
# ======================================================================
def load_models(args):
    models = {}
    model_configs = {
        "surface": {"hef": args.hef_surface, "size": args.size_surface},
        "crop0":    {"hef": args.hef_crop0,    "size": args.size_crop},
        "crop1":    {"hef": args.hef_crop1,    "size": args.size_crop},
        "crop2":    {"hef": args.hef_crop2,    "size": args.size_crop},
        "crop3":    {"hef": args.hef_crop3,    "size": args.size_crop},
        "crop4":    {"hef": args.hef_crop4,    "size": args.size_crop},
        "crop5":    {"hef": args.hef_crop5,    "size": args.size_crop},
    }

    print("\nLoading models, please wait...")
    for key, cfg in model_configs.items():
        hef_path = cfg["hef"]
        if not hef_path: continue
        if not os.path.exists(hef_path):
            print(f"[Model] {key} file not found: {hef_path}")
            continue
        try:
            models[key] = HailoPatchCoreOptimized(hef_path, size=cfg["size"])
            print(f"  + {key}: Loaded.")
        except Exception as e:
            print(f"  - {key}: Failed. {e}")
    return models

def main():
    parser = argparse.ArgumentParser(description="Multi-Part AI Receiver GUI")
    parser.add_argument("--hef-surface", default="")
    parser.add_argument("--size-surface", type=int, default=320)
    
    parser.add_argument("--hef-crop0", default="")
    parser.add_argument("--hef-crop1", default="")
    parser.add_argument("--hef-crop2", default="")
    parser.add_argument("--hef-crop3", default="")
    parser.add_argument("--hef-crop4", default="")
    parser.add_argument("--hef-crop5", default="")
    parser.add_argument("--size-crop", type=int, default=224)
    
    parser.add_argument("--threshold", type=float, default=0.1838)
    args = parser.parse_args()

    # Pre-load Hailo Models
    models = load_models(args)
    if not models:
        print("WARNING: No models loaded. GUI will start but no inference will occur.")

    app = QApplication(sys.argv)
    
    # Initialize and display the new UI
    gui = AIReceiverGUI(args.threshold)
    gui.show()

    # Start TCP receiver background services
    workers = []
    for camera_id, port in PORTS.items():
        w = TCPReceiverWorker(port, camera_id, models)
        w.inference_result_signal.connect(gui.update_inference_ui)
        w.start()
        workers.append(w)

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
