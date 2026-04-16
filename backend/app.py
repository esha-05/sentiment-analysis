"""
Flask Backend API — Sentiment Analysis
=======================================
Endpoints:
  POST /api/predict          — Single text prediction
  POST /api/predict/batch    — Batch predictions (up to 50)
  GET  /api/health           — Health check
  GET  /api/metrics          — Model performance metrics
  GET  /api/model/info       — Model metadata

Now powered by HuggingFace transformers — no .pkl files, no retraining.
"""

import os, sys, json, time, logging, traceback
from datetime import datetime
from functools import wraps

from flask import Flask, request, jsonify, g
from flask.logging import default_handler

# ── Module path fix ───────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "sentiment_core", os.path.join(ROOT, "sentiment_core.py")
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["sentiment_core"] = _mod
_spec.loader.exec_module(_mod)
SentimentPipeline = _mod.SentimentPipeline
# ─────────────────────────────────────────────────────────────────────────────

# ── App Setup ──────────────────────────────────
app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

# ── Logging ────────────────────────────────────
os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
log_path = os.path.join(ROOT, "logs", "api.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)
app.logger.removeHandler(default_handler)

# ── Model Loading ──────────────────────────────
METRICS_PATH = os.path.join(ROOT, "data", "metrics.json")

_pipeline = None

def get_pipeline() -> SentimentPipeline:
    """Lazy-load HuggingFace model (singleton)."""
    global _pipeline
    if _pipeline is None:
        logger.info("Initialising HuggingFace SentimentPipeline ...")
        try:
            _pipeline = SentimentPipeline()
            _pipeline._get_pipe()   # trigger download/load now
            logger.info("Model ready.")
        except RuntimeError as e:
            logger.critical("Model load failed: %s", e)
            raise
    return _pipeline


# ── Decorators ─────────────────────────────────
def require_json(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not request.is_json:
            return jsonify({
                "success": False,
                "error": "Content-Type must be application/json",
                "code": 415,
            }), 415
        return f(*args, **kwargs)
    return wrapper


def log_request(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        g.start_time = time.perf_counter()
        result = f(*args, **kwargs)
        elapsed = (time.perf_counter() - g.start_time) * 1000
        logger.info("%s %s → %s (%.1f ms)",
                    request.method, request.path,
                    result[1] if isinstance(result, tuple) else 200,
                    elapsed)
        return result
    return wrapper


# ── Helpers ────────────────────────────────────
def elapsed_ms() -> float:
    return round((time.perf_counter() - g.start_time) * 1000, 2)


def validate_text(text, field_name="text"):
    if not isinstance(text, str):
        return None, {"success": False, "error": f"'{field_name}' must be a string.", "code": 400}
    text = text.strip()
    if not text:
        return None, {"success": False, "error": f"'{field_name}' must not be empty.", "code": 400}
    if len(text) > 5000:
        return None, {"success": False,
                      "error": f"'{field_name}' exceeds 5000 character limit ({len(text)} chars).",
                      "code": 413}
    return text, None


# ── CORS ───────────────────────────────────────
@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

@app.route("/api/<path:path>", methods=["OPTIONS"])
def handle_options(path):
    return jsonify({}), 200


# ══════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════

@app.route("/api/health", methods=["GET"])
@log_request
def health():
    try:
        pipe = get_pipeline()
        _    = pipe.predict_single("test warmup")
        status = "healthy"
    except Exception as e:
        logger.error("Health check failed: %s", e)
        return jsonify({
            "success": False, "status": "unhealthy",
            "error": str(e), "timestamp": datetime.utcnow().isoformat(),
        }), 503

    return jsonify({
        "success":    True,
        "status":     status,
        "model":      "HuggingFace cardiffnlp/twitter-roberta-base-sentiment-latest",
        "labels":     ["Negative", "Neutral", "Positive", "Uncertain"],
        "timestamp":  datetime.utcnow().isoformat(),
        "latency_ms": elapsed_ms(),
    }), 200


@app.route("/api/predict", methods=["POST"])
@require_json
@log_request
def predict():
    body     = request.get_json(silent=True) or {}
    raw_text = body.get("text", "")

    text, err = validate_text(raw_text)
    if err:
        return jsonify(err), err["code"]

    try:
        pipe   = get_pipeline()
        result = pipe.predict_single(text)
    except RuntimeError as e:
        logger.error("Model not available: %s", e)
        return jsonify({"success": False, "error": str(e), "code": 503}), 503
    except Exception:
        logger.error("Prediction error:\n%s", traceback.format_exc())
        return jsonify({"success": False, "error": "Internal prediction error.", "code": 500}), 500

    return jsonify({
        "success":    True,
        "text":       text,
        "prediction": result,
        "latency_ms": elapsed_ms(),
    }), 200


@app.route("/api/predict/batch", methods=["POST"])
@require_json
@log_request
def predict_batch():
    body  = request.get_json(silent=True) or {}
    texts = body.get("texts", [])

    if not isinstance(texts, list):
        return jsonify({"success": False, "error": "'texts' must be a JSON array.", "code": 400}), 400
    if len(texts) == 0:
        return jsonify({"success": False, "error": "'texts' array is empty.", "code": 400}), 400
    if len(texts) > 50:
        return jsonify({"success": False, "error": "Batch limit is 50 texts.", "code": 413}), 413

    results, errors = [], []
    pipe = get_pipeline()

    for i, raw in enumerate(texts):
        text, err = validate_text(raw, field_name=f"texts[{i}]")
        if err:
            errors.append({"index": i, "error": err["error"]})
            results.append({"index": i, "text": str(raw)[:50], "error": err["error"]})
            continue
        try:
            pred = pipe.predict_single(text)
            results.append({"index": i, "text": text, "prediction": pred})
        except Exception as e:
            errors.append({"index": i, "error": str(e)})
            results.append({"index": i, "text": text, "error": str(e)})

    return jsonify({
        "success":     len(errors) == 0,
        "count":       len(texts),
        "results":     results,
        "error_count": len(errors),
        "latency_ms":  elapsed_ms(),
    }), 200


@app.route("/api/metrics", methods=["GET"])
@log_request
def model_metrics():
    try:
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
    except FileNotFoundError:
        # HuggingFace model — return known published metrics instead
        metrics = {
            "accuracy":  0.724,
            "note":      "Published metrics for cardiffnlp/twitter-roberta-base-sentiment-latest",
            "source":    "https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest",
            "per_class": {
                "Negative": {"f1_score": 0.73},
                "Neutral":  {"f1_score": 0.68},
                "Positive": {"f1_score": 0.76},
            }
        }
    except json.JSONDecodeError:
        return jsonify({"success": False, "error": "Metrics file is corrupt.", "code": 500}), 500

    return jsonify({
        "success":   True,
        "metrics":   metrics,
        "timestamp": datetime.utcnow().isoformat(),
    }), 200


@app.route("/api/model/info", methods=["GET"])
@log_request
def model_info():
    return jsonify({
        "success": True,
        "model": {
            "name":         "cardiffnlp/twitter-roberta-base-sentiment-latest",
            "version":      "2.0.0",
            "type":         "HuggingFace Transformer",
            "architecture": "RoBERTa-base fine-tuned on 124M tweets",
            "framework":    "transformers (HuggingFace)",
            "labels":       ["Negative", "Neutral", "Positive", "Uncertain"],
            "max_length":   512,
            "capabilities": [
                "Emojis understood natively",
                "Negation handled correctly",
                "Slang and informal text",
                "Mixed sentiment",
                "Multilingual input (partial)",
            ],
        },
        "endpoints": {
            "POST /api/predict":       "Single text prediction",
            "POST /api/predict/batch": "Batch prediction (≤50 texts)",
            "GET  /api/health":        "Health check",
            "GET  /api/metrics":       "Model performance metrics",
            "GET  /api/model/info":    "Model metadata",
        },
    }), 200


# ── Error Handlers ─────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return jsonify({"success": False, "error": "Endpoint not found.", "code": 404}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"success": False, "error": "Method not allowed.", "code": 405}), 405

@app.errorhandler(500)
def server_error(e):
    logger.error("Unhandled 500: %s", e)
    return jsonify({"success": False, "error": "Internal server error.", "code": 500}), 500


# ── Entry Point ────────────────────────────────
if __name__ == "__main__":
    logger.info("Starting Sentiment Analysis API (HuggingFace) ...")
    get_pipeline()   # warm up — triggers model download if not cached
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)


# ─────────────────────────────────────────────────────────────────────────────
# ADD THIS ROUTE TO YOUR backend/app.py
# Place it after the /api/predict route (around line 160)
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/rate", methods=["POST"])
@require_json
@log_request
def rate_product():
    """
    Product review rating endpoint.

    Request body:
        {
          "text":     "Review text here",
          "product":  "Samsung Galaxy S24",   (optional)
          "category": "product"               (optional)
        }

    Response:
        {
          "success": true,
          "product":  "Samsung Galaxy S24",
          "category": "product",
          "review":   "Review text...",
          "rating": {
            "stars":      5,
            "label":      "Excellent",
            "sentiment":  "Positive",
            "confidence": 0.92,
            "probabilities": { "Negative": 0.04, "Neutral": 0.04, "Positive": 0.92 }
          },
          "latency_ms": 12.3
        }
    """
    body     = request.get_json(silent=True) or {}
    raw_text = body.get("text", "")
    product  = body.get("product",  "Unknown Product")
    category = body.get("category", "product")

    text, err = validate_text(raw_text)
    if err:
        return jsonify(err), err["code"]

    try:
        pipe   = get_pipeline()
        result = pipe.predict_single(text)
    except RuntimeError as e:
        return jsonify({"success": False, "error": str(e), "code": 503}), 503
    except Exception:
        logger.error("Rating error:\n%s", traceback.format_exc())
        return jsonify({"success": False, "error": "Internal prediction error.", "code": 500}), 500

    # Convert sentiment → star rating
    label      = result["label"]
    confidence = result["confidence"]

    if label == "Positive":
        stars      = 5 if confidence >= 0.75 else 4
        star_label = "Excellent" if stars == 5 else "Good"
    elif label == "Negative":
        stars      = 1 if confidence >= 0.75 else 2
        star_label = "Terrible" if stars == 1 else "Poor"
    else:
        stars      = 3
        star_label = "Average"

    return jsonify({
        "success":  True,
        "product":  product,
        "category": category,
        "review":   text,
        "rating": {
            "stars":         stars,
            "label":         star_label,
            "sentiment":     label,
            "confidence":    confidence,
            "probabilities": result["probabilities"],
        },
        "latency_ms": elapsed_ms(),
    }), 200