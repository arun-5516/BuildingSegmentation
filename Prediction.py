"""
predict_and_postprocess_final.py
Full end-to-end script (YOLOv8 seg -> binary mask -> postprocess -> GeoJSON + outline overlay).
Windows-safe (forward-slashes), tuned to keep small roofs, avoids union by default,
splits MultiPolygons, and saves debug masks.

Save to:
C:/Users/Arun/PycharmProjects/building_seg/predict_and_postprocess_final.py
"""
import os
import glob
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from shapely.geometry import Polygon, mapping
from shapely.ops import unary_union
from PIL import Image, ImageDraw

# --------------------------- USER CONFIG (edit if needed) ---------------------------
BASE_DIR = "C:/Users/Arun/PycharmProjects/building_seg"   # use forward slashes (safe)
model_path = os.path.join(BASE_DIR, "runs", "segment", "building_yolov8_seg", "weights", "best.pt")

# Inputs: either a single image path or a folder (set one)
input_image = "C:/Users/Arun/PycharmProjects/building_seg/test_images/ar1.png"
input_folder = None
# create this folder and add images

# Output folder
output_dir = os.path.join(BASE_DIR, "output_results")
os.makedirs(output_dir, exist_ok=True)

# Overlay settings
use_orig_for_overlay = True

# --------------------------- TUNED PARAMETERS ---------------------------
meters_per_pixel = 0.03      # set correctly for your orthomosaic (e.g., 0.03 = 3cm)
min_area_m2 = 0.05           # keep tiny roofs (0.05 m²)
conf_thres = 0.25            # lower confidence to pick small detections
do_union = False             # IMPORTANT: keep individual polygons (do not union)
kernel_small = np.ones((3, 3), np.uint8)  # small kernel to avoid merging
epsilon_px = 0.5             # approxPolyDP epsilon in pixels (smaller -> more vertices)
simplify_tol_px = 0.2        # polygon simplify tolerance (small)
outline_width = 2            # overlay stroke width for outlines
device = "cpu"               # or 0 for GPU
# ------------------------------------------------------------------------------------

print("Config:")
print(" BASE_DIR:", BASE_DIR)
print(" model_path:", model_path)
print(" input_folder:", input_folder)
print(" meters_per_pixel:", meters_per_pixel, "min_area_m2:", min_area_m2)
print(" conf_thres:", conf_thres, "do_union:", do_union)

# ------------------- Derived & safety checks -------------------
pixel_tol = max(1, int(round(0.10 / meters_per_pixel)))  # not used directly; kept for reference

# Prepare image list
if input_image:
    image_paths = [input_image]
else:
    image_paths = sorted(glob.glob(os.path.join(input_folder, "*.*")))
    image_paths = [p for p in image_paths if p.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))]

if not image_paths:
    raise SystemExit(f"No input images found in: {input_folder} (or set input_image).")

# Load model
if not os.path.exists(model_path):
    raise SystemExit(f"Trained model not found at: {model_path}. Train first or set model_path correctly.")
model = YOLO(model_path)
print("Loaded YOLO model.")

# ------------------- Utility: combine YOLO masks -------------------
def masks_to_combined_uint8(masks_tensor):
    """
    Convert YOLO masks tensor-like object into combined binary uint8 mask (0/255).
    Handles shapes (H,W) or (N,H,W).
    """
    if hasattr(masks_tensor, "detach"):
        arr = masks_tensor.detach().cpu().numpy()
    else:
        arr = np.array(masks_tensor)

    if arr.ndim == 2:
        combined = (arr > 0.5).astype(np.uint8)
    elif arr.ndim == 3:
        combined = np.zeros(arr.shape[1:], dtype=np.uint8)
        for m in arr:
            combined = np.maximum(combined, (m > 0.5).astype(np.uint8))
    else:
        raise RuntimeError(f"Unexpected masks array shape: {arr.shape}")

    return (combined * 255).astype(np.uint8)


# ------------------- Postprocessing (handles MultiPolygon) -------------------
def postprocess_mask_individual(mask_uint8, image_path, base_outname,
                                meters_per_pixel=meters_per_pixel,
                                min_area_m2=min_area_m2,
                                epsilon_px=epsilon_px,
                                simplify_tol_px=simplify_tol_px,
                                do_union=do_union,
                                outline_width=outline_width):
    """
    - Keeps individual contours as separate polygons by default (do_union=False).
    - Splits MultiPolygon results into parts.
    - Saves GeoJSON (pixel coords) and an outline-only overlay PNG.
    """
    # Ensure binary
    _, binary = cv2.threshold(mask_uint8, 127, 255, cv2.THRESH_BINARY)

    # Small morphological cleaning to avoid over-merging
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_small, iterations=1)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_small, iterations=1)

    # Fill small holes via flood fill on inverted mask
    inv = cv2.bitwise_not(opened)
    h, w = inv.shape
    mask_floodfill = inv.copy()
    mask_temp = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(mask_floodfill, mask_temp, (0, 0), 255)
    filled = cv2.bitwise_not(mask_floodfill)
    binary_clean = cv2.bitwise_or(opened, filled)

    # Find contours
    contours, _ = cv2.findContours(binary_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for cnt in contours:
        area_px = cv2.contourArea(cnt)
        area_m2 = area_px * (meters_per_pixel ** 2)
        if area_m2 < min_area_m2:
            continue

        approx = cv2.approxPolyDP(cnt, epsilon_px, True)
        if approx is None or len(approx) < 3:
            continue

        pts = approx.reshape(-1, 2)
        poly = Polygon(pts).buffer(0)
        if not poly.is_valid or poly.is_empty:
            continue

        if poly.area * (meters_per_pixel ** 2) < min_area_m2:
            continue

        polygons.append(poly)

    if not polygons:
        debug_mask_path = os.path.join(output_dir, f"debug_{base_outname}_mask.png")
        cv2.imwrite(debug_mask_path, binary_clean)
        print(f"No polygons found after filtering. Debug mask saved: {debug_mask_path}")
        return None, debug_mask_path

    # Optionally union overlapping polygons (turned off by default)
    if do_union:
        unioned = unary_union(polygons)
        if unioned.geom_type == "Polygon":
            polygons = [unioned]
        elif unioned.geom_type == "MultiPolygon":
            polygons = list(unioned.geoms)

    # Simplify polygons a little and keep valid ones; split MultiPolygons into parts
    simplified_parts = []
    for p in polygons:
        p2 = p.simplify(simplify_tol_px, preserve_topology=True).buffer(0)
        if not (p2.is_valid and not p2.is_empty):
            continue
        if p2.geom_type == "MultiPolygon":
            for part in p2.geoms:
                if part.is_valid and not part.is_empty:
                    simplified_parts.append(part)
        else:
            simplified_parts.append(p2)

    if not simplified_parts:
        debug_mask_path = os.path.join(output_dir, f"debug_{base_outname}_mask2.png")
        cv2.imwrite(debug_mask_path, binary_clean)
        print(f"No valid polygons after simplification. Debug mask saved: {debug_mask_path}")
        return None, debug_mask_path

    # Write GeoJSON (pixel coordinates) — one feature per polygon part
    features = []
    for i, poly in enumerate(simplified_parts):
        features.append({
            "type": "Feature",
            "properties": {"id": i + 1},
            "geometry": mapping(poly)
        })
    geojson_path = os.path.join(output_dir, f"clean_buildings_{base_outname}.geojson")
    with open(geojson_path, "w") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f)
    print("Saved GeoJSON:", geojson_path)

    # Create outline-only overlay (draw exterior + interiors)
    if image_path and os.path.exists(image_path) and use_orig_for_overlay:
        orig_img = Image.open(image_path).convert("RGBA")
    else:
        orig_img = Image.new("RGBA", (w, h), (255, 255, 255, 255))

    overlay = Image.new("RGBA", orig_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    outline_color = (255, 0, 0, 200)  # red w/ alpha

    for poly in simplified_parts:
        # exterior ring
        exterior_coords = [(float(x), float(y)) for x, y in poly.exterior.coords]
        if len(exterior_coords) >= 2:
            draw.line(exterior_coords + [exterior_coords[0]], fill=outline_color, width=outline_width)
        # interiors (holes)
        for interior in poly.interiors:
            interior_coords = [(float(x), float(y)) for x, y in interior.coords]
            if len(interior_coords) >= 2:
                draw.line(interior_coords + [interior_coords[0]], fill=outline_color, width=max(1, outline_width-1))

    combined = Image.alpha_composite(orig_img, overlay)
    overlay_path = os.path.join(output_dir, f"final_polygons_outline_{base_outname}.png")
    combined.convert("RGB").save(overlay_path)
    print("Saved outline overlay:", overlay_path)

    return geojson_path, overlay_path


# ------------------------ MAIN LOOP ------------------------
for img_path in image_paths:
    name = Path(img_path).stem
    print("\nProcessing:", img_path)

    # Run YOLO inference
    results = model.predict(img_path, conf=conf_thres, device=device)
    r = results[0]

    mask_uint8 = None
    if hasattr(r, "masks") and r.masks is not None:
        try:
            mask_uint8 = masks_to_combined_uint8(r.masks.data)
        except Exception:
            try:
                mask_uint8 = masks_to_combined_uint8(r.masks)
            except Exception as e:
                print("Failed to extract masks:", e)
                mask_uint8 = None

    if mask_uint8 is None:
        print("No mask available for", img_path)
        continue

    # Ensure solidity and a small close
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel_small, iterations=1)

    # Save mask for debugging/viewing
    mask_outpath = os.path.join(output_dir, f"{name}_mask.png")
    cv2.imwrite(mask_outpath, mask_uint8)
    print("Saved mask:", mask_outpath)

    # Save a debug mask too
    debug_mask_path = os.path.join(output_dir, f"{name}_mask_debug.png")
    cv2.imwrite(debug_mask_path, mask_uint8)
    print("Saved debug mask:", debug_mask_path)

    # Postprocess (individual polygons, outline overlay)
    out_geojson, out_overlay = postprocess_mask_individual(
        mask_uint8, img_path, name,
        meters_per_pixel=meters_per_pixel,
        min_area_m2=min_area_m2,
        epsilon_px=epsilon_px,
        simplify_tol_px=simplify_tol_px,
        do_union=do_union,
        outline_width=outline_width
    )

print("\n✓ All done. Results in:", os.path.abspath(output_dir))
