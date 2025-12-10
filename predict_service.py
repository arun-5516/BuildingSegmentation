# predict_service.py
import os
import io
import numpy as np
import cv2
from ultralytics import YOLO

# Load YOLO model once
MODEL_PATH = os.environ.get("MODEL_PATH", "yolov8n-seg.pt")
print("Loading YOLO model from:", MODEL_PATH)
MODEL = YOLO(MODEL_PATH)
print("Model loaded.")

# Import your postprocess function (must exist in same folder)
from predict_and_postprocess_final import (
    postprocess_mask_individual,
    masks_to_combined_uint8,
)

def predict_image_bytes(image_bytes, conf=0.25,
                        meters_per_pixel=0.03, min_area_m2=0.05):

    """Runs YOLO + postprocess and returns:
       mask_bytes, overlay_bytes, geojson_bytes, debug_mask_bytes
    """

    # Convert bytes → OpenCV image
    np_image = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

    if img is None:
        raise RuntimeError("Invalid image bytes sent to backend")

    # Run YOLO
    results = MODEL.predict(source=img, conf=conf, device="cpu")
    r = results[0]

    # Extract combined segmentation mask
    mask_uint8 = None
    if hasattr(r, "masks") and r.masks is not None:
        mask_uint8 = masks_to_combined_uint8(
            r.masks.data if hasattr(r.masks, "data") else r.masks
        )

    if mask_uint8 is None:
        raise RuntimeError("YOLO produced no mask")

    # Now postprocess → polygon + overlay + debug mask
    geojson_bytes, overlay_bytes, debug_mask_bytes = postprocess_mask_individual(
        mask_uint8,
        image_bytes=image_bytes,
        meters_per_pixel=meters_per_pixel,
        min_area_m2=min_area_m2,
    )

    # Convert mask to PNG bytes
    _, mask_png = cv2.imencode(".png", mask_uint8)
    mask_bytes = mask_png.tobytes()

    return mask_bytes, overlay_bytes, geojson_bytes, debug_mask_bytes
