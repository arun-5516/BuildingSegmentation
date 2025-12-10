# app.py
import io
import zipfile
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from predict_service import predict_image_bytes  # <--- uses the in-memory function

app = Flask(__name__)
CORS(app)

@app.route("/api/process", methods=["POST"])
def api_process():
    if "images" not in request.files:
        return jsonify({"error": "No files uploaded. Use form field 'images'."}), 400

    files = request.files.getlist("images")
    conf = float(request.form.get("conf", 0.25))
    meters_per_pixel = float(request.form.get("meters_per_pixel", 0.03))
    min_area_m2 = float(request.form.get("min_area_m2", 0.05))

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            name = f.filename.rsplit(".", 1)[0]
            try:
                img_bytes = f.read()
                mask_b, overlay_b, geojson_b, debug_b = predict_image_bytes(
                    img_bytes,
                    conf=conf,
                    meters_per_pixel=meters_per_pixel,
                    min_area_m2=min_area_m2
                )
            except Exception as e:
                zf.writestr(f"{name}_error.txt", str(e))
                continue

            if mask_b:
                zf.writestr(f"{name}_mask.png", mask_b)
            if overlay_b:
                zf.writestr(f"{name}_overlay.png", overlay_b)
            if geojson_b:
                zf.writestr(f"{name}.geojson", geojson_b)
            if debug_b:
                zf.writestr(f"{name}_debug_mask.png", debug_b)

    zip_buf.seek(0)
    return send_file(zip_buf, mimetype="application/zip", as_attachment=True, download_name="results.zip")

@app.route("/ping")
def ping():
    return "ok", 200

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
