"""
app.py — SecureScan Screening Dashboard

Flask entrypoint. Works identically:
  • locally:        python app.py
  • via gunicorn:    gunicorn app:app
  • on Vercel:       auto-detected (Vercel looks for a Flask instance named
                     `app` at app.py — see vercel.json / README)

ARCHITECTURE NOTE — why nothing is written to disk at request time:
Serverless platforms (Vercel included) give you an ephemeral filesystem —
anything written during a request is not guaranteed to exist on the next
one, possibly even within the same client session. So instead of the
original save-to-uploads/-then-redirect-to-a-static-result-image pattern,
every detection request is fully self-contained: image bytes come in,
an annotated JPEG goes back out as a base64 data URI in the same JSON
response. The browser holds the running screening log in memory
(+ localStorage) rather than the server holding a database. This makes
the app work identically on a normal server and on Vercel with zero
extra configuration.
"""

import os
import time
import base64
import logging

import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.exceptions import RequestEntityTooLarge

from config import Config
import inference

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(Config)


def allowed_file(filename: str) -> bool:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext in app.config["ALLOWED_EXTENSIONS"]


def _decode_upload(file_storage) -> np.ndarray | None:
    data = file_storage.read()
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _encode_result_image(img_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", img_bgr,
                           [cv2.IMWRITE_JPEG_QUALITY, app.config["RESULT_JPEG_QUALITY"]])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii")


def _run_and_package(img_bgr: np.ndarray, filename: str, conf: float) -> dict:
    t0 = time.time()
    result = inference.run_detection(img_bgr, conf_threshold=conf)
    duration_ms = round((time.time() - t0) * 1000, 1)

    if result is None:
        status = inference.get_model_status()
        return {
            "ok": False,
            "filename": filename,
            "error": "Detection engine unavailable.",
            "model_status": status,
        }

    return {
        "ok": True,
        "filename": filename,
        "security_level":   result["security_level"],     # critical | suspicious | safe
        "is_threat":        result["is_threat"],
        "needs_review":     result["needs_review"],
        "detected_objects": result["detected_objects"],
        "threat_items":     result["threat_items"],
        "possible_objects":      result["possible_objects"],
        "possible_threat_items": result["possible_threat_items"],
        "confidence_scores": result["confidence_scores"],
        "annotated_image":  _encode_result_image(result["image"]),
        "duration_ms":      duration_ms,
    }


# ── Error handlers ────────────────────────────────────────────────────────
@app.errorhandler(RequestEntityTooLarge)
def too_large(_):
    return jsonify(ok=False, error="File too large. Maximum size is 12 MB."), 413

@app.errorhandler(404)
def not_found(_):
    return jsonify(ok=False, error="Not found"), 404

@app.errorhandler(500)
def server_error(exc):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return jsonify(ok=False, error="Internal server error"), 500


# ── Pages ─────────────────────────────────────────────────────────────────
@app.route("/")
def screening():
    return render_template("screening.html",
                           default_conf=int(app.config["DEFAULT_CONF_THRESHOLD"] * 100))


# ── Sample images (bundled with the deployment, always available) ──────────
@app.route("/api/samples")
def api_samples():
    folder = app.config["SAMPLE_IMAGES_DIR"]
    if not os.path.isdir(folder):
        return jsonify(samples=[])
    names = sorted(f for f in os.listdir(folder) if allowed_file(f))
    return jsonify(samples=[{"name": n, "url": f"/sample-images/{n}"} for n in names])


@app.route("/sample-images/<path:filename>")
def sample_image_file(filename):
    return send_from_directory(app.config["SAMPLE_IMAGES_DIR"], filename)


# ── Core detection API ───────────────────────────────────────────────────
@app.route("/api/screen", methods=["POST"])
def api_screen():
    """
    Screen ONE image. Two ways to call it:
      1. multipart/form-data with `file` = an uploaded image
      2. form field `sample` = filename of a bundled sample image
    Either way, returns JSON with the annotated image inline (base64) —
    no file is ever written to disk.
    """
    conf = float(request.form.get("confidence", app.config["DEFAULT_CONF_THRESHOLD"]))
    conf = max(0.05, min(0.95, conf))

    sample_name = request.form.get("sample")
    if sample_name:
        safe_name = os.path.basename(sample_name)          # defang path traversal
        path = os.path.join(app.config["SAMPLE_IMAGES_DIR"], safe_name)
        if not os.path.isfile(path):
            return jsonify(ok=False, error="Unknown sample image"), 404
        img = cv2.imread(path)
        filename = safe_name
    elif "file" in request.files and request.files["file"].filename:
        f = request.files["file"]
        if not allowed_file(f.filename):
            return jsonify(ok=False, error="Unsupported file type"), 415
        img = _decode_upload(f)
        filename = f.filename
    else:
        return jsonify(ok=False, error="No file or sample provided"), 400

    if img is None:
        return jsonify(ok=False, error="Could not decode image"), 422

    payload = _run_and_package(img, filename, conf)
    return jsonify(payload), (200 if payload["ok"] else 500)


@app.route("/api/model-status")
def api_model_status():
    """Lets the UI show, in plain sight, whether the real custom models
    are loaded and which classes they actually cover."""
    return jsonify(inference.get_model_status())


@app.route("/api/health")
def health():
    return jsonify(status="ok")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
