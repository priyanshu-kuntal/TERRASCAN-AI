"""
TERRASCAN-AI — Tactical Defense Camouflage and Anomaly Detection System
Flask Backend Application and RESTful Tactical API.
"""

from __future__ import annotations

import os
import time
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from flask import Flask, render_template, request, jsonify, send_file, Response
from werkzeug.exceptions import RequestEntityTooLarge

from terrascan.pipeline import TerraScanPipeline
from terrascan.samples_generator import ensure_sample_images
from terrascan.utils import load_image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s in %(module)s: %(message)s"
)
logger = logging.getLogger("terrascan")

# Set TORCH_HOME to /tmp on serverless or read-only environments
import tempfile
if "TORCH_HOME" not in os.environ:
    os.environ["TORCH_HOME"] = os.path.join(tempfile.gettempdir(), "torch")

# Base directory configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
REFERENCE_DIR = os.path.join(BASE_DIR, "data", "reference")
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")

try:
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REFERENCE_DIR, exist_ok=True)
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    ensure_sample_images(SAMPLES_DIR)
except OSError as e:
    logger.warning("Filesystem is read-only or restricted: %s", e)

# Initialize Flask application
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024  # 15 MB max file upload limit

# Startup timestamp
START_TIME = time.time()

# Allowed image extensions
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

# Global Pipeline instance (initialized once at server startup)
logger.info("Initializing TERRASCAN-AI Multi-Sensor Pipeline...")
pipeline = TerraScanPipeline(
    models_dir=MODELS_DIR,
    reference_dir=REFERENCE_DIR,
    yolo_weights=os.path.join(BASE_DIR, "yolov8n.pt")
)
logger.info("TERRASCAN-AI Pipeline Ready. Operational on %s.", pipeline.device_name)


def is_allowed_file(filename: str) -> bool:
    """Validate allowed file extension."""
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_EXTENSIONS


@app.errorhandler(RequestEntityTooLarge)
def handle_payload_too_large(e):
    return jsonify({"error": "Payload exceeds 15 MB limit. Please upload a smaller image."}), 413


@app.errorhandler(404)
def handle_not_found(e):
    return jsonify({"error": "Resource not found."}), 404


@app.errorhandler(500)
def handle_internal_error(e):
    logger.error("Internal Server Error: %s", e)
    return jsonify({"error": "Tactical analysis encounter an internal computation failure."}), 500


@app.route("/")
def index():
    """Render the primary Tactical Console SITREP UI."""
    return render_template("index.html")


@app.route("/api/status", methods=["GET"])
def api_status():
    """Return hardware status, operational modes, and reference telemetry."""
    try:
        sys_status = pipeline.get_system_status()
        uptime_seconds = int(time.time() - START_TIME)
        return jsonify({
            "status": "OPERATIONAL",
            "uptime_seconds": uptime_seconds,
            "system": sys_status,
        })
    except Exception as e:
        logger.error("Error retrieving status: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/api/sample/<int:sample_id>", methods=["GET"])
def api_sample(sample_id: int):
    """Serve one of the 3 pre-generated synthetic terrain samples."""
    if sample_id not in (1, 2, 3):
        return jsonify({"error": "Invalid sample ID. Must be 1, 2, or 3."}), 400

    filename = f"sample_{sample_id:02d}.png"
    filepath = os.path.join(SAMPLES_DIR, filename)

    if not os.path.exists(filepath):
        ensure_sample_images(SAMPLES_DIR)

    if not os.path.exists(filepath):
        return jsonify({"error": f"Sample {sample_id} not available."}), 404

    return send_file(filepath, mimetype="image/png")


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """
    Primary image analysis endpoint.
    Accepts multipart/form-data image or sample_id with tactical parameters.
    """
    try:
        # Extract parameters with safe defaults
        camo_thresh = float(request.form.get("camo_threshold", 0.50))
        camo_weight = float(request.form.get("camo_weight", 0.60))
        anomaly_weight = float(request.form.get("anomaly_weight", 0.40))
        object_conf = float(request.form.get("object_conf", 0.25))
        overlay_opacity = float(request.form.get("overlay_opacity", 0.60))

        enable_camo = request.form.get("enable_camo", "true").lower() in ("true", "1", "yes")
        enable_anomaly = request.form.get("enable_anomaly", "true").lower() in ("true", "1", "yes")
        enable_objects = request.form.get("enable_objects", "true").lower() in ("true", "1", "yes")

        image_data = None

        # Check if an image file was uploaded
        if "image" in request.files and request.files["image"].filename:
            file = request.files["image"]
            if not is_allowed_file(file.filename):
                return jsonify({
                    "error": "Unsupported image format. Permitted types: PNG, JPG, JPEG, WEBP, BMP, TIFF."
                }), 400
            image_data = file.read()
        elif "sample_id" in request.form:
            # Load selected sample image
            s_id = int(request.form["sample_id"])
            if s_id not in (1, 2, 3):
                return jsonify({"error": "Invalid sample_id. Must be 1, 2, or 3."}), 400
            s_path = os.path.join(SAMPLES_DIR, f"sample_{s_id:02d}.png")
            if not os.path.exists(s_path):
                ensure_sample_images(SAMPLES_DIR)
            with open(s_path, "rb") as f:
                image_data = f.read()
        else:
            return jsonify({
                "error": "No image payload provided. Please upload an image or select a tactical sample."
            }), 400

        # Execute analysis pipeline
        result = pipeline.run_analysis(
            image_input=image_data,
            camo_threshold=camo_thresh,
            camo_weight=camo_weight,
            anomaly_weight=anomaly_weight,
            object_conf=object_conf,
            overlay_opacity=overlay_opacity,
            enable_camo=enable_camo,
            enable_anomaly=enable_anomaly,
            enable_objects=enable_objects,
        )

        return jsonify(result)

    except Exception as e:
        logger.exception("Pipeline analysis failed: %s", e)
        return jsonify({"error": f"Tactical analysis failed: {str(e)}"}), 500


@app.route("/api/reference", methods=["POST"])
def api_upload_reference():
    """Upload one or more normal terrain reference images to rebuild the PatchCore memory bank."""
    try:
        uploaded_files = request.files.getlist("reference_images")
        if not uploaded_files or (len(uploaded_files) == 1 and not uploaded_files[0].filename):
            if "reference_image" in request.files and request.files["reference_image"].filename:
                uploaded_files = [request.files["reference_image"]]
            else:
                return jsonify({"error": "No reference images supplied in request."}), 400

        saved_count = 0
        for file in uploaded_files:
            if file and is_allowed_file(file.filename):
                filename = f"ref_{int(time.time() * 1000)}_{file.filename}"
                save_path = os.path.join(REFERENCE_DIR, filename)
                file.save(save_path)
                saved_count += 1

        if saved_count == 0:
            return jsonify({"error": "No valid image files found in upload."}), 400

        # Rebuild PatchCore memory bank
        bank_size = pipeline.anomaly_detector.rebuild_memory_bank()
        status = pipeline.anomaly_detector.get_status()

        return jsonify({
            "success": True,
            "message": f"Added {saved_count} reference image(s). Memory bank rebuilt with {bank_size} patches.",
            "status": status,
        })
    except Exception as e:
        logger.error("Error adding reference image: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/api/reference", methods=["DELETE"])
def api_clear_reference():
    """Clear all reference images and reset anomaly detector to self-reference mode."""
    try:
        pipeline.anomaly_detector.clear_references()
        status = pipeline.anomaly_detector.get_status()
        return jsonify({
            "success": True,
            "message": "All reference terrain data cleared. Switched to SELF-REFERENCE mode.",
            "status": status,
        })
    except Exception as e:
        logger.error("Error clearing reference images: %s", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info("Starting TERRASCAN-AI Mission Server on http://127.0.0.1:%d", port)
    app.run(host="127.0.0.1", port=port, debug=False)
