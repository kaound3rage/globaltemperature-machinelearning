"""
====================================================
GlobalTemperature - AWS Lambda Inference Handler
====================================================
Function name : globaltemp-lambda
Runtime       : Python 3.11
Trigger       : API Gateway (API Key auth)

Alur:
  User → API Gateway → Lambda → S3 (baca forecast)
                              → DynamoDB (simpan activity log)

Endpoints:
  POST /predict   → forecast per dataset
  GET  /metadata  → info model & metrics
  GET  /list      → list semua label forecast

DynamoDB Table: globaltemp-activity-log
  Partition Key : request_id (String)
  Attributes    : user_id, endpoint, request_body,
                  response_status, timestamp, duration_ms
====================================================
"""

import os
import json
import uuid
import time
import boto3
import logging
from datetime import datetime, timezone

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ─── Config ───────────────────────────────────────────────────────────────────
S3_BUCKET      = os.environ.get("S3_BUCKET",      "globallandtemperature")
RESULTS_KEY    = os.environ.get("RESULTS_KEY",    "models/forecast_results.json")
METADATA_KEY   = os.environ.get("METADATA_KEY",   "models/model_metadata.json")
DYNAMO_TABLE   = os.environ.get("DYNAMO_TABLE",   "globaltemp-activity-log")

s3     = boto3.client("s3")
dynamo = boto3.resource("dynamodb")
table  = dynamo.Table(DYNAMO_TABLE)

# ─── Cache hasil forecast di memori ───────────────────────────────────────────
_results_cache = None

def load_results() -> dict:
    """Load forecast_results.json dari S3, cached selama container hidup."""
    global _results_cache
    if _results_cache is not None:
        return _results_cache

    logger.info(f"Loading results dari s3://{S3_BUCKET}/{RESULTS_KEY}")
    obj = s3.get_object(Bucket=S3_BUCKET, Key=RESULTS_KEY)
    _results_cache = json.loads(obj["Body"].read().decode("utf-8"))
    logger.info("Results loaded successfully")
    return _results_cache

# ─── DynamoDB: simpan activity log ───────────────────────────────────────────
def log_activity(request_id: str, endpoint: str, method: str,
                 request_body: dict, response_status: int,
                 duration_ms: int, event: dict, error_msg: str = None):
    """Catat setiap aktivitas user ke DynamoDB."""
    try:
        # Ambil identitas user dari API Gateway request context
        request_context = event.get("requestContext", {})
        identity        = request_context.get("identity", {})

        item = {
            "request_id"     : request_id,
            "timestamp"      : datetime.now(timezone.utc).isoformat(),
            "endpoint"       : endpoint,
            "method"         : method,
            "request_body"   : json.dumps(request_body, default=str),
            "response_status": response_status,
            "duration_ms"    : duration_ms,
            # Identitas user dari API Gateway
            "source_ip"      : identity.get("sourceIp", "unknown"),
            "user_agent"      : identity.get("userAgent", "unknown"),
            "api_key_id"     : request_context.get("identity", {}).get("apiKeyId", "unknown"),
            "stage"          : request_context.get("stage", "unknown"),
            # Error jika ada
            "error"          : error_msg or "",
            # TTL: log otomatis hapus setelah 90 hari (epoch seconds)
            "ttl"            : int(time.time()) + (90 * 24 * 60 * 60),
        }

        table.put_item(Item=item)
        logger.info(f"Activity logged: {request_id} | {endpoint} | {response_status}")

    except Exception as e:
        # Jangan sampai error DynamoDB mengganggu response ke user
        logger.error(f"Gagal log ke DynamoDB: {e}")

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
    dataset = body.get("dataset", "").lower().strip()
    label   = body.get("label",   "").strip()

    if not dataset:
        return error(400, "Field 'dataset' wajib diisi. Pilihan: global, country, city, state")
    if not label:
        return error(400, "Field 'label' wajib diisi. Contoh: 'global', 'country_Albania'")

    data    = load_results()
    results = data.get("results", {})

    if dataset not in results:
        return error(404, f"Dataset '{dataset}' tidak ditemukan. Tersedia: {list(results.keys())}")

    if label not in results[dataset]:
        return error(404, f"Label '{label}' tidak ditemukan di dataset '{dataset}'. "
                         f"Tersedia: {list(results[dataset].keys())}")

    metrics = results[dataset][label]

    return response(200, {
        "dataset"       : dataset,
        "label"         : label,
        "forecast"      : [round(v, 4) for v in metrics["future_vals"]],
        "future_index"  : metrics["future_index"],
        "mae"           : metrics["mae"],
        "rmse"          : metrics["rmse"],
        "forecast_years": len(metrics["future_vals"]),
    })


def handle_metadata() -> dict:
    try:
        obj  = s3.get_object(Bucket=S3_BUCKET, Key=METADATA_KEY)
        meta = json.loads(obj["Body"].read().decode("utf-8"))
        return response(200, meta)
    except Exception as e:
        logger.error(f"Gagal load metadata: {e}")
        return error(500, f"Gagal load metadata: {str(e)}")


def handle_list(dataset: str = None) -> dict:
    data    = load_results()
    results = data.get("results", {})

    if dataset:
        dataset = dataset.lower().strip()
        if dataset not in results:
            return error(404, f"Dataset '{dataset}' tidak ditemukan. Tersedia: {list(results.keys())}")
        return response(200, {
            "dataset": dataset,
            "labels" : list(results[dataset].keys()),
            "total"  : len(results[dataset]),
        })

    summary = {
        ds: {"labels": list(res.keys()), "total": len(res)}
        for ds, res in results.items()
    }
    return response(200, {"datasets": summary})


# ─── Lambda handler ───────────────────────────────────────────────────────────

def lambda_handler(event, context):
    logger.info(f"Event: {json.dumps(event)}")

    # Generate request ID unik untuk setiap request
    request_id   = str(uuid.uuid4())
    start_time   = time.time()

    http_method  = event.get("httpMethod", "GET").upper()
    path         = event.get("path", "/").rstrip("/")
    query_params = event.get("queryStringParameters") or {}
    dataset_qs   = query_params.get("dataset", "")

    raw_body = event.get("body") or "{}"
    try:
        body = json.loads(raw_body) if isinstance(raw_body, str) else raw_body
    except json.JSONDecodeError:
        result = error(400, "Request body bukan JSON valid")
        log_activity(request_id, path, http_method, {}, 400,
                     int((time.time() - start_time) * 1000), event,
                     "Invalid JSON body")
        return result

    # ── Routing ───────────────────────────────────────────────────────────────
    result     = None
    error_msg  = None

    try:
        if path == "/predict" and http_method == "POST":
            result = handle_predict(body)

        elif path == "/metadata" and http_method == "GET":
            result = handle_metadata()

        elif path == "/list" and http_method == "GET":
            result = handle_list(dataset=dataset_qs)

        elif path in ("", "/") and http_method == "GET":
            result = response(200, {
                "service" : "GlobalTemperature Inference API",
                "version" : "1.0.0",
                "endpoints": {
                    "POST /predict" : "Prediksi forecast. Body: {dataset, label}",
                    "GET /metadata" : "Info model & metrics",
                    "GET /list"     : "List semua label. Query: ?dataset=global",
                }
            })

        else:
            result = error(404, f"Endpoint '{http_method} {path}' tidak ditemukan")

    except Exception as e:
        logger.error(f"Unhandled error: {e}", exc_info=True)
        error_msg = str(e)
        result    = error(500, f"Internal server error: {str(e)}")

    # ── Log ke DynamoDB ───────────────────────────────────────────────────────
    duration_ms = int((time.time() - start_time) * 1000)
    log_activity(
        request_id     = request_id,
        endpoint       = path or "/",
        method         = http_method,
        request_body   = body if http_method == "POST" else {"dataset": dataset_qs},
        response_status= result["statusCode"],
        duration_ms    = duration_ms,
        event          = event,
        error_msg      = error_msg
    )

    return result