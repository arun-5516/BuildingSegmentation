# predict_and_postprocess_final.py
"""
Render-safe postprocessing utilities:
 - masks_to_combined_uint8(masks_tensor) -> uint8 mask (0/255)
 - postprocess_mask_individual(mask_uint8, image_input, ...) -> (geojson_bytes, overlay_bytes, debug_mask_bytes)

Notes:
 - image_input may be a file path (str) or image bytes (bytes). Both are supported.
 - This module does NOT write files to disk (returns bytes).
"""

import os
import io
import json
import cv2
import numpy as np
from shapely.geometry import Polygon, mapping
from shapely.ops import unary_union
from PIL import Image, ImageDraw

# Defaults
DEFAULT_METERS_PER_PIXEL = 0.03
DEFAULT_MIN_AREA_M2 = 0.05
DEFAULT_EPSILON_PX = 0.5
DEFAULT_SIMPLIFY_PX = 0.2
DEFAULT_OUTLINE_WIDTH = 2
DEFAULT_DO_UNION = False

kernel_small = np.ones((3, 3), np.uint8)


def masks_to_combined_uint8(masks_tensor):
    """Combine YOLO masks into single uint8 binary mask (0/255)."""
    arr = getattr(masks_tensor, "data", masks_tensor)
    arr = np.array(arr)

    if arr.ndim == 2:
        combined = (arr > 0.5).astype(np.uint8)
    elif arr.ndim == 3:
        combined = np.zeros(arr.shape[1:], dtype=np.uint8)
        for m in arr:
            combined = np.maximum(combined, (m > 0.5).astype(np.uint8))
    else:
        raise RuntimeError(f"Unexpected masks array shape: {arr.shape}")

    return (combined * 255).astype(np.uint8)


def _bytes_or_path_to_pil(image_input):
    """Accept bytes or path; return PIL Image (RGBA) and size (w,h)."""
    if image_input is None:
        return None
    if isinstance(image_input, (bytes, bytearray)):
        buf = io.BytesIO(image_input)
        pil = Image.open(buf).convert("RGBA")
        return pil
    elif isinstance(image_input, str) and os.path.exists(image_input):
        pil = Image.open(image_input).convert("RGBA")
        return pil
    else:
        # Not a path and not bytes — return None
        return None


def _arr_to_png_bytes(arr):
    """Convert numpy array (single-channel or 3-channel BGR) to PNG bytes."""
    if arr.ndim == 2:
        pil = Image.fromarray(arr).convert("L")
    else:
        # assume BGR (cv2) -> convert to RGB
        pil = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


def _make_overlay(polygons, image_input, size, outline_width):
    """Create overlay PNG bytes drawing polygon outlines on original (if provided) or white background."""
    w, h = size
    base = _bytes_or_path_to_pil(image_input)
    if base is None:
        base = Image.new("RGBA", (w, h), (255, 255, 255, 255))
    else:
        # if sizes mismatch, resize overlay canvas to mask size (mask coords are in mask pixel space)
        if base.size != (w, h):
            base = base.resize((w, h))

    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    outline_color = (255, 0, 0, 200)

    for poly in polygons:
        exterior = [(float(x), float(y)) for x, y in poly.exterior.coords]
        if len(exterior) >= 2:
            draw.line(exterior + [exterior[0]], fill=outline_color, width=outline_width)
        for interior in poly.interiors:
            interior_coords = [(float(x), float(y)) for x, y in interior.coords]
            if len(interior_coords) >= 2:
                draw.line(interior_coords + [interior_coords[0]], fill=outline_color, width=max(1, outline_width - 1))

    combined = Image.alpha_composite(base, overlay)
    buf = io.BytesIO()
    combined.convert("RGB").save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


def postprocess_mask_individual(
    mask_uint8,
    image_input=None,
    base_outname="result",
    meters_per_pixel=DEFAULT_METERS_PER_PIXEL,
    min_area_m2=DEFAULT_MIN_AREA_M2,
    epsilon_px=DEFAULT_EPSILON_PX,
    simplify_tol_px=DEFAULT_SIMPLIFY_PX,
    do_union=DEFAULT_DO_UNION,
    outline_width=DEFAULT_OUTLINE_WIDTH,
):
    """
    From binary mask (uint8 0/255) produce:
      - geojson_bytes (bytes)
      - overlay_png_bytes (bytes)
      - debug_mask_png_bytes (bytes)
    image_input may be a path or image bytes; if None, a white background is used.
    """

    # Ensure binary mask
    _, binary = cv2.threshold(mask_uint8, 127, 255, cv2.THRESH_BINARY)

    # morphological cleaning
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_small, iterations=1)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_small, iterations=1)

    # fill holes
    inv = cv2.bitwise_not(opened)
    h, w = inv.shape
    mask_floodfill = inv.copy()
    temp = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(mask_floodfill, temp, (0, 0), 255)
    filled = cv2.bitwise_not(mask_floodfill)
    clean = cv2.bitwise_or(opened, filled)

    # find contours and create polygons
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for c in contours:
        area_px = cv2.contourArea(c)
        if area_px * (meters_per_pixel ** 2) < min_area_m2:
            continue

        approx = cv2.approxPolyDP(c, epsilon_px, True)
        if approx is None or len(approx) < 3:
            continue

        pts = approx.reshape(-1, 2)
        poly = Polygon(pts).buffer(0)
        if poly.is_empty:
            continue
        polygons.append(poly)

    if not polygons:
        # return debug mask only (no geojson/overlay)
        return None, None, _arr_to_png_bytes(clean)

    # optional union
    if do_union:
        merged = unary_union(polygons)
        polygons = list(merged.geoms) if merged.geom_type == "MultiPolygon" else [merged]

    # simplify + split multiparts
    final_polys = []
    for p in polygons:
        p2 = p.simplify(simplify_tol_px, preserve_topology=True).buffer(0)
        if p2.is_empty:
            continue
        if p2.geom_type == "MultiPolygon":
            final_polys.extend(list(p2.geoms))
        else:
            final_polys.append(p2)

    if not final_polys:
        return None, None, _arr_to_png_bytes(clean)

    # build geojson bytes (pixel coordinates)
    features = []
    for i, poly in enumerate(final_polys):
        features.append({"type": "Feature", "properties": {"id": i + 1}, "geometry": mapping(poly)})
    geojson_bytes = json.dumps({"type": "FeatureCollection", "features": features}).encode("utf-8")

    # build overlay bytes
    overlay_bytes = _make_overlay(final_polys, image_input, (w, h), outline_width)

    debug_bytes = _arr_to_png_bytes(clean)
    return geojson_bytes, overlay_bytes, debug_bytes
