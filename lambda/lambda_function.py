"""
====================================================
GlobalTemperature - AWS Lambda Inference Handler
====================================================
Function name : globaltemp-lambda
Runtime       : Python 3.11
Trigger       : API Gateway (API Key auth)

Endpoints:
  POST /predict   → forecast per dataset
  GET  /metadata  → info model & metrics
  GET  /list      → list semua label forecast

Request format (POST /predict):
{
  "dataset": "global" | "country" | "city" | "state",
  "label"  : "global" | "country_Albania" | "city_Abidjan" | ...
}

Response format:
{
  "label"       : "global",
  "dataset"     : "global",
  "forecast"    : [8.22, 8.25, ...],
  "future_index": [2014, 2015, ...],
  "mae"         : 0.7169,
  "rmse"        : 0.8862
}
====================================================
"""

import os
import json
import boto3
import pickle
import logging

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler

# =============================================================================
# GBTForecaster — harus didefinisikan ulang di sini agar pickle bisa di-load.
# Class ini identik dengan yang ada di sagemaker_training.py.
# Pickle menyimpan reference ke nama class, bukan kode-nya — sehingga Lambda
# perlu tahu class ini sebelum memanggil pickle.load().
# =============================================================================
class GBTForecaster:
    def __init__(self, n_lags=5, n_estimators=100, max_depth=5,
                 learning_rate=0.05, subsample=0.8):
        self.n_lags        = n_lags
        self.n_estimators  = n_estimators
        self.max_depth     = max_depth
        self.learning_rate = learning_rate
        self.subsample     = subsample
        self.models        = {}
        self.scalers       = {}
        self.results       = {}
        self.trained_at    = None

    def _build_model(self):
        return GradientBoostingRegressor(
            n_estimators  = self.n_estimators,
            max_depth     = self.max_depth,
            learning_rate = self.learning_rate,
            subsample     = self.subsample,
            random_state  = 42
        )

    def fit_single(self, *args, **kwargs):
        pass  # tidak dipakai di Lambda

    def forecast(self, label, last_values, steps):
        model  = self.models[label]
        scaler = self.scalers[label]
        scaled = scaler.transform(last_values.reshape(-1, 1)).flatten()
        predictions  = []
        current_lags = list(scaled[-self.n_lags:])
        for _ in range(steps):
            X_input = np.array(current_lags[-self.n_lags:]).reshape(1, -1)
            pred    = model.predict(X_input)[0]
            predictions.append(pred)
            current_lags.append(pred)
        return scaler.inverse_transform(
            np.array(predictions).reshape(-1, 1)
        ).flatten()

    def fit_and_forecast_all(self, *args, **kwargs):
        pass  # tidak dipakai di Lambda, hanya untuk kompatibilitas pickle


logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ─── Config ───────────────────────────────────────────────────────────────────
S3_BUCKET    = os.environ.get("S3_BUCKET",    "globallandtemperature")
MODEL_KEY    = os.environ.get("MODEL_KEY",    "models/temperature_forecast_model.pkl")
METADATA_KEY = os.environ.get("METADATA_KEY", "models/model_metadata.json")
LOCAL_MODEL  = "/tmp/temperature_forecast_model.pkl"

s3 = boto3.client("s3")

# ─── Load model (cached di /tmp selama Lambda container hidup) ────────────────
_model_cache = None

def load_model():
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    logger.info(f"Downloading model dari s3://{S3_BUCKET}/{MODEL_KEY}")
    s3.download_file(S3_BUCKET, MODEL_KEY, LOCAL_MODEL)

    with open(LOCAL_MODEL, "rb") as f:
        _model_cache = pickle.load(f)

    logger.info("Model loaded successfully")
    return _model_cache

# ─── Response helper ──────────────────────────────────────────────────────────
def response(status_code: int, body: dict) -> dict:
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps(body, default=str)
    }

def error(status_code: int, message: str) -> dict:
    return response(status_code, {"error": message})

# ─── Route handlers ───────────────────────────────────────────────────────────

def handle_predict(body: dict) -> dict:
    """
    POST /predict
    Body: { "dataset": "global", "label": "global" }
    """
    dataset = body.get("dataset", "").lower()
    label   = body.get("label",   "").strip()

    if not dataset:
        return error(400, "Field 'dataset' wajib diisi. Pilihan: global, country, city, state")
    if not label:
        return error(400, "Field 'label' wajib diisi. Contoh: 'global', 'country_Albania'")

    model_pkg = load_model()
    results   = model_pkg["model"].results

    if dataset not in results:
        available = list(results.keys())
        return error(404, f"Dataset '{dataset}' tidak ditemukan. Tersedia: {available}")

    if label not in results[dataset]:
        available = list(results[dataset].keys())
        return error(404, f"Label '{label}' tidak ditemukan di dataset '{dataset}'. "
                         f"Tersedia: {available}")

    metrics = results[dataset][label]

    return response(200, {
        "dataset"     : dataset,
        "label"       : label,
        "forecast"    : [round(v, 4) for v in metrics["future_vals"]],
        "future_index": metrics["future_index"],
        "mae"         : metrics["mae"],
        "rmse"        : metrics["rmse"],
        "forecast_years": len(metrics["future_vals"]),
    })


def handle_metadata() -> dict:
    """GET /metadata — info model, versi, metrics."""
    try:
        obj  = s3.get_object(Bucket=S3_BUCKET, Key=METADATA_KEY)
        meta = json.loads(obj["Body"].read().decode("utf-8"))
        return response(200, meta)
    except Exception as e:
        logger.error(f"Gagal load metadata: {e}")
        return error(500, f"Gagal load metadata: {str(e)}")


def handle_list(dataset: str = None) -> dict:
    """
    GET /list              → list semua dataset & label
    GET /list?dataset=city → list label untuk dataset tertentu
    """
    model_pkg = load_model()
    results   = model_pkg["model"].results

    if dataset:
        dataset = dataset.lower()
        if dataset not in results:
            available = list(results.keys())
            return error(404, f"Dataset '{dataset}' tidak ditemukan. Tersedia: {available}")
        return response(200, {
            "dataset": dataset,
            "labels" : list(results[dataset].keys()),
            "total"  : len(results[dataset]),
        })

    # Semua dataset
    summary = {}
    for ds, res in results.items():
        summary[ds] = {
            "labels": list(res.keys()),
            "total" : len(res),
        }
    return response(200, {"datasets": summary})


# ─── Lambda handler ───────────────────────────────────────────────────────────

def lambda_handler(event, context):
    logger.info(f"Event: {json.dumps(event)}")

    # Ambil path & method
    http_method = event.get("httpMethod", "GET").upper()
    path        = event.get("path", "/").rstrip("/")

    # Query string params
    query_params = event.get("queryStringParameters") or {}
    dataset_qs   = query_params.get("dataset", "")

    # Body
    raw_body = event.get("body") or "{}"
    try:
        body = json.loads(raw_body) if isinstance(raw_body, str) else raw_body
    except json.JSONDecodeError:
        return error(400, "Request body bukan JSON valid")

    # ── Routing ───────────────────────────────────────────────────────────────
    try:
        if path == "/predict" and http_method == "POST":
            return handle_predict(body)

        elif path == "/metadata" and http_method == "GET":
            return handle_metadata()

        elif path == "/list" and http_method == "GET":
            return handle_list(dataset=dataset_qs)

        elif path in ("", "/") and http_method == "GET":
            return response(200, {
                "service" : "GlobalTemperature Inference API",
                "version" : "1.0.0",
                "endpoints": {
                    "POST /predict" : "Prediksi forecast. Body: {dataset, label}",
                    "GET /metadata" : "Info model & metrics",
                    "GET /list"     : "List semua label. Query: ?dataset=global",
                }
            })

        else:
            return error(404, f"Endpoint '{http_method} {path}' tidak ditemukan")

    except Exception as e:
        logger.error(f"Unhandled error: {e}", exc_info=True)
        return error(500, f"Internal server error: {str(e)}")