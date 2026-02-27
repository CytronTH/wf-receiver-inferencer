import socket
import struct
import json
import cv2
import numpy as np
import threading
import time
import os
import argparse

from inference_hailo_rpi_optimized import HailoPatchCoreOptimized

# ======================================================================
# ตั้งค่าการเชื่อมต่อ
# ======================================================================
HOST = '0.0.0.0'
PORTS = {
    0: 8080,  # กล้องตัวที่ 0
    1: 8081,  # กล้องตัวที่ 1
}

# ======================================================================
# ตั้งค่าโมเดลและ Routing Table
#
# image_routing: กำหนดว่า image_id หนึ่งๆ จะถูก inference ด้วย
#   "model_key" ไหน (ชื่อ Key ที่ตรงกับ models dict ด้านล่าง)
#   หากไม่ต้องการ inference รูปไหน ให้ตั้ง model_key เป็น None
# ======================================================================
IMAGE_ROUTING = {
    # รูปเต็ม Surface ใช้โมเดล surface
    "masked_surface": "surface",

    # รูปที่ Crop มาทีละชิ้น ใช้โมเดล crop
    "crop_0": "crop0",
    "crop_1": "crop1",
    "crop_2": "crop2",
    "crop_3": "crop3",
    "crop_4": "crop4",
    "crop_5": "crop5",

    # ตัวอย่าง: ถ้ามี image_id ที่ไม่ต้องการ inference ตั้งเป็น None
    # "thumbnail": None,
}

# Lock สำหรับป้องกันการเรียก Hailo Inference พร้อมกันจากหลาย Thread
infer_lock = threading.Lock()


def load_models(args):
    """
    โหลดโมเดล Hailo ทั้งหมดที่ต้องใช้งาน
    คืนค่าเป็น dict: {"surface": model_obj, "crop": model_obj, ...}
    """
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

    for key, cfg in model_configs.items():
        hef_path = cfg["hef"]
        if not hef_path:
            print(f"[Model] '{key}' skipped (no HEF path provided).")
            continue
        if not os.path.exists(hef_path):
            print(f"[Model] '{key}' HEF file not found: {hef_path}")
            continue
        try:
            print(f"[Model] Loading '{key}' from {hef_path}...")
            models[key] = HailoPatchCoreOptimized(hef_path, size=cfg["size"])
            print(f"[Model] '{key}' loaded successfully.")
        except Exception as e:
            print(f"[Model] Failed to load '{key}': {e}")

    return models


def recvall(sock, n):
    """
    รับข้อมูลให้ครบตามจำนวน bytes ที่ระบุพอดี
    แก้ปัญหา TCP Fragmentation
    """
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data


def run_ai(image, camera_id, image_id, model, threshold):
    """
    รัน Hailo Inference กับภาพที่ระบุ และแสดงผลลัพธ์
    """
    with infer_lock:
        try:
            score, amap = model.infer(image)
        except Exception as e:
            print(f"[Cam {camera_id}] Inference error for '{image_id}': {e}")
            return

    result_status = "NG" if score > threshold else "OK"
    print(f"  --> [Cam {camera_id}] {image_id:<20} | Score: {score:.4f} | {result_status}")


def receive_image_stream(conn, addr, camera_id, models, threshold):
    """
    รับภาพผ่าน TCP ตามโปรโตคอล 3-Part และ Route แต่ละรูป
    ไปยังโมเดลที่เหมาะสมตาม IMAGE_ROUTING
    """
    print(f"[Cam {camera_id}] Connected: {addr}")
    try:
        while True:
            # --- Part 1: Header (4 bytes) -> ขนาดของ JSON ---
            header_data = recvall(conn, 4)
            if not header_data:
                print(f"[Cam {camera_id}] Sender disconnected.")
                break
            json_size = struct.unpack('>L', header_data)[0]

            # --- Part 2: JSON Metadata ---
            json_bytes = recvall(conn, json_size)
            if not json_bytes:
                print(f"[Cam {camera_id}] Connection lost while reading JSON.")
                break
            metadata = json.loads(json_bytes.decode('utf-8'))
            image_id = metadata.get("id", "unknown")
            image_size = metadata.get("size", 0)

            # --- Part 3: JPEG Image Data ---
            if image_size <= 0:
                print(f"[Cam {camera_id}] Invalid image size for '{image_id}', skipping.")
                continue
            jpeg_data = recvall(conn, image_size)
            if not jpeg_data:
                print(f"[Cam {camera_id}] Connection lost while reading image data.")
                break

            # --- Decode JPEG -> numpy BGR ---
            np_arr = np.frombuffer(jpeg_data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None:
                print(f"[Cam {camera_id}] Failed to decode JPEG for '{image_id}'")
                continue

            # --- Route ภาพไปยังโมเดลที่ถูกต้องตาม IMAGE_ROUTING ---
            if image_id not in IMAGE_ROUTING:
                print(f"[Cam {camera_id}] Unknown image_id '{image_id}', skipping.")
                continue

            model_key = IMAGE_ROUTING[image_id]
            if model_key is None:
                # image_id นี้ตั้งใจข้ามการ inference
                print(f"[Cam {camera_id}] '{image_id}' skipped (routing = None).")
                continue

            model = models.get(model_key)
            if model is None:
                print(f"[Cam {camera_id}] Model '{model_key}' not loaded, skipping '{image_id}'.")
                continue

            # --- ส่งเข้า Inference ---
            run_ai(img, camera_id, image_id, model, threshold)

    except ConnectionResetError:
        print(f"[Cam {camera_id}] Connection forcibly closed by remote host.")
    except Exception as e:
        print(f"[Cam {camera_id}] Stream error: {e}")
    finally:
        conn.close()
        print(f"[Cam {camera_id}] Socket closed. Waiting for new connection...")


def server_thread(port, camera_id, models, threshold):
    """
    เธรดที่เปิด TCP Server ต่อ 1 กล้อง
    """
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, port))
    srv.listen(5)
    print(f"[Cam {camera_id}] Listening on {HOST}:{port}")

    while True:
        try:
            conn, addr = srv.accept()
            receive_image_stream(conn, addr, camera_id, models, threshold)
        except Exception as e:
            print(f"[Cam {camera_id}] Server error: {e}")
            time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description="Dual TCP Camera Receiver with Hailo AI (Multi-Model)")

    # โมเดลสำหรับภาพ Surface (ภาพเต็ม)
    parser.add_argument("--hef-surface", default="", help="Path to HEF model for 'masked_surface' images")
    parser.add_argument("--size-surface", type=int, default=320, help="Input size for surface model (default: 320)")

    # โมเดลสำหรับภาพ Crop (แต่ละ crop_0-5 ระบุโมเดลได้อิสระ ถ้าใช้โมเดลเดียวกันให้ใส่ไฟล์เดิมซ้ำ)
    parser.add_argument("--hef-crop0", default="", help="HEF model for crop_0")
    parser.add_argument("--hef-crop1", default="", help="HEF model for crop_1")
    parser.add_argument("--hef-crop2", default="", help="HEF model for crop_2")
    parser.add_argument("--hef-crop3", default="", help="HEF model for crop_3")
    parser.add_argument("--hef-crop4", default="", help="HEF model for crop_4")
    parser.add_argument("--hef-crop5", default="", help="HEF model for crop_5")
    parser.add_argument("--size-crop", type=int, default=224, help="Input size for crop models (default: 224)")

    parser.add_argument("--threshold", type=float, default=0.5, help="Anomaly Threshold (default: 0.5)")
    args = parser.parse_args()

    # 1. โหลดโมเดลทั้งหมด
    models = load_models(args)
    if not models:
        print("[Warning] No models were loaded. Inference will be skipped for all images.")

    # 2. เริ่ม TCP Server Thread สำหรับแต่ละกล้อง
    print("\nStarting TCP Stream Listeners...")
    threads = []
    for camera_id, port in PORTS.items():
        t = threading.Thread(
            target=server_thread,
            args=(port, camera_id, models, args.threshold),
            daemon=True
        )
        t.start()
        threads.append(t)

    print("\nSystem Ready! Press Ctrl+C to exit.\n" + "=" * 50)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down AI Receiver...")


if __name__ == "__main__":
    main()
