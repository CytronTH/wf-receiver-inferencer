# AI Receiver GUI - Multi-Part Inference Monitor

ระบบรับภาพผ่าน TCP และทำ Inference ด้วย Hailo-8 (PatchCore) สำหรับการตรวจจับความผิดปกติ (Anomaly Detection) ในหลายส่วนของภาพพร้อมกัน

## 🛠 ฟีเจอร์หลัก (Simplified Mapping)
- **Automatic Routing**: ระบบจะจับคู่ภาพเข้ากับโมเดลโดยอัตโนมัติผ่าน `image_id` ที่ส่งมาจาก Sender
- **JSON Configuration**: ควบคุมการโหลดโมเดล, ขนาดภาพ (resize), และ Threshold แยกตามแต่ละโมเดลได้ในไฟล์เดียว
- **Dynamic UI**: หน้าจอ GUI จะปรับเปลี่ยนจำนวน Panel ตามจำนวนโมเดลที่โหลดจริงใน Config

---

## 🚀 การใช้งาน

### 1. การตั้งค่าโมเดล (`models_config.json`)
แก้ไขไฟล์ `models_config.json` เพื่อระบุตำแหน่งไฟล์โมเดลและค่า Threshold:

```json
{
    "masked_surface": {
        "hef": "models/surface_model.hef",
        "size": 320,
        "threshold": 0.1838
    },
    "crop_1": {
        "hef": "models/crop1_model.hef",
        "size": 224,
        "threshold": 0.15
    }
}
```

### 2. การรันระบบ
ใช้งานผ่านสคริปต์ `run_receiver.sh`:
```bash
./run_receiver.sh
```
หรือรันคำสั่งโดยตรง:
```bash
python3 ai_receiver_gui.py --config models_config.json
```

---

## 🧠 หลักการทำงาน

### 1. การรับข้อมูล (TCP Protocol)
ระบบเปิด Port รอรับข้อมูล (Default: 8080, 8081) โดย Header ของข้อมูลต้องมี JSON Metadata ที่ระบุ `id` ของภาพ:
- ตัวอย่าง Metadata: `{"id": "crop_1", "size": 54321}`

### 2. การจับคู่โมเดล (Inference Routing)
ระบบใช้หลักการ **Direct ID Mapping**:
- หากได้รับ `id` เป็น `"crop_1"` โปรแกรมจะมองหาโมเดลในหน่วยความจำที่มี Key ชื่อ `"crop_1"` ทันที
- ไม่ต้องแก้ไข Python code เมื่อมีการเพิ่มหรือเปลี่ยนชื่อ Part (แค่แก้ JSON config ให้ตรงกับ ID ที่ Sender ส่งมา)

### 3. การแสดงผล (GUI)
- ระบบจะสร้างหน้าต่างพรีวิวสำหรับทุก Part ที่ระบุไว้ใน `models_config.json`
- แสดงผลภาพต้นฉบับคู่กับ Heatmap
- มีระบบปรับ Threshold แบบ Real-time บนหน้าจอเพื่ออำนวยความสะดวกในการ Calibrate

---

## 📂 โครงสร้างไฟล์
- `ai_receiver_gui.py`: โปรแกรมหลัก (PyQt6)
- `models_config.json`: ไฟล์ตั้งค่าโมเดลและ Threshold
- `inference_hailo_rpi_optimized.py`: ตัวขับเคลื่อนการ Inference (Hailo-8)
- `run_receiver.sh`: สคริปต์สำหรับรันโปรแกรม
- `models/`: โฟลเดอร์สำหรับเก็บไฟล์ `.hef`
