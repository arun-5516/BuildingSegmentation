from roboflow import Roboflow
from ultralytics import YOLO
import os

# ==========================================
# 🚀 STEP 1 — Initialize Roboflow
# ==========================================


rf = Roboflow(api_key="XejOnEfA15SfGFIJZ6uc")
project = rf.workspace("gis-detection").project("building-segmentation-3uaut-liebi")
version = project.version(2)
dataset = version.download("yolov8")

print("✅ Dataset downloaded to:", dataset.location)

# ==========================================
# 🧩 STEP 2 — Load YOLOv8 segmentation model
# ==========================================
model = YOLO("yolov8n-seg.pt")  # use yolov8s-seg.pt for better accuracy (optional)

# ==========================================
# 🧠 STEP 3 — Train the model
# ==========================================
yaml_path = os.path.join(dataset.location, "data.yaml")

if not os.path.exists(yaml_path):
    raise FileNotFoundError(f"⚠ data.yaml not found at: {yaml_path}")

model.train(
    data=yaml_path,       # ✅ Correct data.yaml path
    epochs=3,             # Increase (like 50) for better results
    imgsz=640,
    batch=8,
    name="building_yolov8_seg"
)

print("✅ Training finished! Model saved at: runs/segment/building_yolov8_seg/weights/best.pt")

# ==========================================
# 🧾 STEP 4 — Optional: Predict on a test image
# ==========================================
# result = model.predict(source="test.jpg", show=True, conf=0.5)