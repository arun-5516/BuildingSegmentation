"""
Streamlit app: YOLOv8 segmentation -> mask -> polygon postprocess -> download ZIP

Requirements:
  pip install ultralytics roboflow opencv-python-headless numpy shapely pillow streamlit

Place your model file "yolov8n-seg.pt" in the same folder as this script,
or set MODEL_PATH to an absolute path.
"""

import os
import io
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import cv2
from PIL import Image, ImageDraw

import streamlit as st
from ultralytics import YOLO
from shapely.geometry import Polygon, mapping
from shapely.ops import unary_union

# ------------------------ USER SETTINGS (edit if needed) ------------------------
MODEL_PATH = "yolov8n-seg.pt"   # <-- your model file path
DEFAULT_CONF = 0.25
DEFAULT_MIN_AREA_M2 = 0.05
METERS_PER_PIXEL_DEFAULT = 0.03
USE_ORIG_FOR_OVERLAY = True
# --------------------------------------------------------------------------------

st.set_page_config(page_title="YOLOv8 Building Seg - Streamlit", layout="wide")

st.title("📸 YOLOv8 Segmentation → Polygons (GeoJSON) — Streamlit frontend")
st.markdown(
    "Upload one or more images, run the segmentation model, view masks/overlays, "
    "and download results as a ZIP (masks, overlays, GeoJSONs)."
)

# ------------------------ Helper utilities ------------------------
@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    model = YOLO(model_path)
    return model

def masks_to_combined_uint8(masks_tensor):
    """
    Accepts either a tensor-like or array-like masks object from ultralytics Results.
    Returns a single uint8 binary mask (0/255) where all instance masks are OR-ed.
    """
    if masks_tensor is None:
        return None
    try:
        # many ultralytics versions store masks.data as (N,H,W)
        arr = getattr(masks_tensor, "data", masks_tensor)
        arr = np.array(arr)
    except Exception:
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

def postprocess_mask_individual(mask_uint8, image_path=None,
                                meters_per_pixel=METERS_PER_PIXEL_DEFAULT,
                                min_area_m2=DEFAULT_MIN_AREA_M2,
                                epsilon_px=0.5, simplify_tol_px=0.2,
                                do_union=False, outline_width=2,
                                use_orig_for_overlay=USE_ORIG_FOR_OVERLAY):
    """
    From binary mask (uint8 0/255) produce:
      - geojson_bytes (bytes) with pixel coords,
      - overlay_png_bytes (bytes) (outline on original image or white background),
      - debug_mask_png_bytes (bytes)
    Returns (geojson_bytes_or_None, overlay_png_bytes, debug_mask_png_bytes)
    """
    # Ensure binary
    _, binary = cv2.threshold(mask_uint8, 127, 255, cv2.THRESH_BINARY)
    h, w = binary.shape

    # small morphology to clean
    kernel_small = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_small, iterations=1)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_small, iterations=1)

    # fill holes via flood fill
    inv = cv2.bitwise_not(opened)
    mask_floodfill = inv.copy()
    mask_temp = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(mask_floodfill, mask_temp, (0, 0), 255)
    filled = cv2.bitwise_not(mask_floodfill)
    binary_clean = cv2.bitwise_or(opened, filled)

    # find contours (external)
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
        # return debug mask
        debug_bytes = _image_to_bytes(binary_clean)
        return None, None, debug_bytes

    # optionally union polygons
    if do_union:
        unioned = unary_union(polygons)
        if unioned.geom_type == "Polygon":
            polygons = [unioned]
        elif unioned.geom_type == "MultiPolygon":
            polygons = list(unioned.geoms)

    # simplify & split MultiPolygons
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
        debug_bytes = _image_to_bytes(binary_clean)
        return None, None, debug_bytes

    # build geojson feature collection (pixel coords)
    features = []
    for i, poly in enumerate(simplified_parts):
        features.append({
            "type": "Feature",
            "properties": {"id": i + 1},
            "geometry": mapping(poly)
        })
    fc = {"type": "FeatureCollection", "features": features}
    geojson_bytes = json.dumps(fc).encode("utf-8")

    # create overlay PNG (outline)
    overlay_bytes = _create_overlay_png(simplified_parts, image_path, (w, h), outline_width, use_orig_for_overlay)

    # also return debug mask as bytes
    debug_bytes = _image_to_bytes(binary_clean)

    return geojson_bytes, overlay_bytes, debug_bytes

def _image_to_bytes(img_array):
    """Convert single-channel or 3-channel numpy image to PNG bytes (uint8)."""
    if img_array.ndim == 2:
        pil = Image.fromarray(img_array).convert("L")
    else:
        pil = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()

def _create_overlay_png(polygons, image_path, size, outline_width, use_orig_for_overlay):
    w, h = size
    if image_path and os.path.exists(image_path) and use_orig_for_overlay:
        orig_img = Image.open(image_path).convert("RGBA")
        # if orig size differs, resize overlay canvas to orig size
        if orig_img.size != (w, h):
            # assume input passed (w,h) corresponds to mask; prefer original for display
            base_img = orig_img
        else:
            base_img = orig_img
    else:
        base_img = Image.new("RGBA", (w, h), (255, 255, 255, 255))

    overlay = Image.new("RGBA", base_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    outline_color = (255, 0, 0, 200)

    for poly in polygons:
        exterior_coords = [(float(x), float(y)) for x, y in poly.exterior.coords]
        if len(exterior_coords) >= 2:
            draw.line(exterior_coords + [exterior_coords[0]], fill=outline_color, width=outline_width)
        for interior in poly.interiors:
            interior_coords = [(float(x), float(y)) for x, y in interior.coords]
            if len(interior_coords) >= 2:
                draw.line(interior_coords + [interior_coords[0]], fill=outline_color, width=max(1, outline_width-1))

    combined = Image.alpha_composite(base_img.convert("RGBA"), overlay)
    buf = io.BytesIO()
    combined.convert("RGB").save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()

# ------------------------ Streamlit UI ------------------------
st.sidebar.header("Model & processing settings")
st.sidebar.markdown(f"Model path: `{MODEL_PATH}`")
conf_thres = st.sidebar.slider("Confidence threshold", 0.0, 1.0, float(DEFAULT_CONF), 0.01)
meters_per_pixel = st.sidebar.number_input("Meters per pixel (scale)", value=float(METERS_PER_PIXEL_DEFAULT), format="%.6f")
min_area_m2 = st.sidebar.number_input("Min area (m²) to keep", value=float(DEFAULT_MIN_AREA_M2), format="%.6f")
eps_px = st.sidebar.number_input("approxPolyDP epsilon (px)", value=0.5, format="%.2f")
simplify_px = st.sidebar.number_input("simplify tolerance (px)", value=0.2, format="%.2f")
outline_width = st.sidebar.slider("Overlay outline width (px)", 1, 8, 2)
do_union = st.sidebar.checkbox("Union overlapping polygons (may merge roofs)", value=False)
use_orig_overlay = st.sidebar.checkbox("Use original image for overlay", value=USE_ORIG_FOR_OVERLAY)

st.sidebar.markdown("---")
st.sidebar.markdown("⚠️ If you upload very large orthophotos, processing may be slow or memory-heavy.")

# Upload
uploaded_files = st.file_uploader("Upload images (PNG/JPG/TIFF). Multiple allowed.", type=["png", "jpg", "jpeg", "tif", "tiff"], accept_multiple_files=True)

if not uploaded_files:
    st.info("Upload at least one image to start inference.")
    st.stop()

# Load model
with st.spinner("Loading model..."):
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

# Process button
if st.button("Run inference & postprocess"):
    # temporary directory to collect outputs
    with tempfile.TemporaryDirectory() as tmpdir:
        output_files = []  # list of tuples (relative_path_in_zip, bytes)
        progress_bar = st.progress(0.0)
        total = len(uploaded_files)
        idx = 0

        for up in uploaded_files:
            idx += 1
            progress_bar.progress((idx - 1) / total)
            st.write(f"### Processing: {up.name}")

            # save uploaded image to disk temporarily (ultralytics predict works with path reliably)
            tmp_img_path = os.path.join(tmpdir, up.name)
            with open(tmp_img_path, "wb") as f:
                f.write(up.getbuffer())

            # run YOLOv8 inference
            with st.spinner(f"Running model on {up.name} ..."):
                try:
                    results = model.predict(source=tmp_img_path, conf=conf_thres, device="cpu")  # device could be changed if desired
                except Exception as e:
                    st.error(f"Model inference failed for {up.name}: {e}")
                    continue

            if not results:
                st.warning(f"No results returned for {up.name}")
                continue
            r = results[0]

            # extract combined mask
            mask_uint8 = None
            if hasattr(r, "masks") and r.masks is not None:
                try:
                    mask_uint8 = masks_to_combined_uint8(r.masks.data)
                except Exception:
                    try:
                        mask_uint8 = masks_to_combined_uint8(r.masks)
                    except Exception as e:
                        st.warning(f"Failed to extract masks for {up.name}: {e}")
                        mask_uint8 = None

            if mask_uint8 is None:
                st.warning(f"No mask available for {up.name}")
                continue

            # small close
            kernel_small = np.ones((3, 3), np.uint8)
            mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel_small, iterations=1)

            # show mask inline
            st.subheader("Segmentation mask (combined)")
            mask_png_bytes = _image_to_bytes(mask_uint8)
            st.image(mask_png_bytes, caption=f"{up.name} - mask", use_column_width=True)

            # postprocess -> geojson + overlay + debug mask
            geojson_bytes, overlay_bytes, debug_mask_bytes = postprocess_mask_individual(
                mask_uint8,
                image_path=tmp_img_path,
                meters_per_pixel=meters_per_pixel,
                min_area_m2=min_area_m2,
                epsilon_px=eps_px,
                simplify_tol_px=simplify_px,
                do_union=do_union,
                outline_width=outline_width,
                use_orig_for_overlay=use_orig_overlay
            )

            # filenames
            base_stem = Path(up.name).stem
            mask_name = f"{base_stem}_mask.png"
            overlay_name = f"{base_stem}_overlay.png"
            geojson_name = f"{base_stem}.geojson"
            debug_mask_name = f"{base_stem}_mask_debug.png"

            # write files to tmpdir and collect into zip list
            with open(os.path.join(tmpdir, mask_name), "wb") as f:
                f.write(mask_png_bytes)
            output_files.append((mask_name, mask_png_bytes))

            if overlay_bytes:
                with open(os.path.join(tmpdir, overlay_name), "wb") as f:
                    f.write(overlay_bytes)
                st.subheader("Overlay (outline on original)")
                st.image(overlay_bytes, caption=f"{up.name} - overlay", use_column_width=True)
                output_files.append((overlay_name, overlay_bytes))
            else:
                st.info("No overlay produced (no polygons found). Showing debug mask instead.")
                st.image(debug_mask_bytes, caption="Debug mask (no polygons survived)", use_column_width=True)
                output_files.append((debug_mask_name, debug_mask_bytes))

            if geojson_bytes:
                with open(os.path.join(tmpdir, geojson_name), "wb") as f:
                    f.write(geojson_bytes)
                st.download_button(label=f"Download {geojson_name}", data=geojson_bytes, file_name=geojson_name, mime="application/geo+json")
                output_files.append((geojson_name, geojson_bytes))
            else:
                st.info("No GeoJSON produced (no polygons after filtering).")

            idx += 0
            progress_bar.progress(idx / total)

        progress_bar.progress(1.0)

        # create ZIP in memory
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for fname, data in output_files:
                zf.writestr(fname, data)
        zip_buf.seek(0)

        st.success("All images processed. Download results below.")
        st.download_button("Download ZIP of all outputs", data=zip_buf.getvalue(), file_name="segmentation_results.zip", mime="application/zip")
