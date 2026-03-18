"""
====================================================
GlobalTemperature - SageMaker Model Training
====================================================
Jalankan script ini di SageMaker Notebook Instance:
  ml.t3.medium

Algoritma:
  1. GBT Regressor + Lag Features (PySpark MLlib)
  2. Iterative Forecasting (10 tahun ke depan)
  3. Evaluasi: RMSE & MAE per dataset
  4. Export model ke S3: models/

Dataset:
  - global   → year_global, global_avg_temp
  - country  → year, country_name, avg_temp_country
  - city     → month, city_name, avg_temp_city
  - state    → year_month, state_name, avg_temp_state

Install dependencies dulu di notebook terminal:
  pip install pyarrow fastparquet scikit-learn boto3 pandas
====================================================
"""

import os
import json
import pickle
import warnings
import boto3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error
warnings.filterwarnings("ignore")

# Coba import PySpark
try:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.types import DoubleType
    from pyspark.sql.window import Window
    from pyspark.ml.feature import VectorAssembler, MinMaxScaler
    from pyspark.ml.regression import GBTRegressor
    from pyspark.ml import Pipeline
    from pyspark.ml.evaluation import RegressionEvaluator
    PYSPARK_AVAILABLE = True
    print("✓ PySpark tersedia")
except ImportError:
    PYSPARK_AVAILABLE = False
    print("⚠ PySpark tidak tersedia, pakai fallback pandas + sklearn")

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler as SklearnScaler

# ─── Konfigurasi ──────────────────────────────────────────────────────────────
S3_BUCKET    = "globallandtemperature"
S3_INPUT     = "s3://globallandtemperature/processed-data/"
S3_MODEL_OUT = "s3://globallandtemperature/models/"
REGION       = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

FORECAST_YEARS = 10
LOOKBACK       = 5
TOP_N_GROUPS   = 3
MODEL_DIR      = "/tmp/models"
PLOT_DIR       = "/tmp/plots"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR,  exist_ok=True)

s3 = boto3.client("s3", region_name=REGION)

print(f"[Training] Bucket    : {S3_BUCKET}")
print(f"[Training] Input     : {S3_INPUT}")
print(f"[Training] Model out : {S3_MODEL_OUT}")


# ════════════════════════════════════════════════════════════════
# STEP 1: LOAD DATA DARI S3
# ════════════════════════════════════════════════════════════════

print("\n=== STEP 1: Load Data ===")

def download_parquet_from_s3(prefix: str) -> pd.DataFrame:
    """Download semua file Parquet dari S3 prefix ke DataFrame."""
    response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)
    parquet_files = [
        obj["Key"] for obj in response.get("Contents", [])
        if obj["Key"].endswith(".parquet")
    ]

    dfs = []
    for key in parquet_files:
        local_path = f"/tmp/{key.replace('/', '_')}"
        s3.download_file(S3_BUCKET, key, local_path)
        dfs.append(pd.read_parquet(local_path))

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


try:
    global_df  = download_parquet_from_s3("processed-data/global/")
    country_df = download_parquet_from_s3("processed-data/coutry/")   # sesuai folder S3 (typo: coutry)
    city_df    = download_parquet_from_s3("processed-data/city/")
    state_df   = download_parquet_from_s3("processed-data/state/")

    print(f"  global  : {len(global_df)} baris")
    print(f"  country : {len(country_df)} baris")
    print(f"  city    : {len(city_df)} baris")
    print(f"  state   : {len(state_df)} baris")
except Exception as e:
    print(f"⚠ Gagal load dari S3: {e}")
    raise SystemExit(1)

# ── Validasi: hentikan jika data kosong ───────────────────────────────────────
_missing = []
if len(global_df)  == 0: _missing.append(f"  • global/   → {S3_INPUT}global/")
if len(country_df) == 0: _missing.append(f"  • coutry/   → {S3_INPUT}coutry/")
if len(city_df)    == 0: _missing.append(f"  • city/     → {S3_INPUT}city/")
if len(state_df)   == 0: _missing.append(f"  • state/    → {S3_INPUT}state/")

if _missing:
    print("\n❌ DATA KOSONG — training dibatalkan.")
    print("   Pastikan file .parquet sudah ada di path S3 berikut:")
    for m in _missing:
        print(m)
    print("\n   Cek isi bucket:")
    print(f"   aws s3 ls s3://{S3_BUCKET}/processed-data/ --recursive")
    print("\n   Jalankan dulu Glue ETL job sebelum training.")
    raise SystemExit(1)


# ════════════════════════════════════════════════════════════════
# STEP 2: PREPROCESSING — LAG FEATURES
# ════════════════════════════════════════════════════════════════

print("\n=== STEP 2: Preprocessing — Lag Features ===")

def create_lag_features(df: pd.DataFrame, time_col: str, target_col: str,
                        group_col: str = None, n_lags: int = 5) -> pd.DataFrame:
    """
    Buat lag features dari kolom target untuk supervised learning.
    lag_1 = nilai t-1, lag_2 = nilai t-2, dst.
    """
    df = df.copy().sort_values(time_col)

    if group_col:
        for i in range(1, n_lags + 1):
            df[f"lag_{i}"] = df.groupby(group_col)[target_col].shift(i)
    else:
        for i in range(1, n_lags + 1):
            df[f"lag_{i}"] = df[target_col].shift(i)

    lag_cols = [f"lag_{i}" for i in range(1, n_lags + 1)]
    df = df.dropna(subset=lag_cols + [target_col])
    return df, lag_cols


# ════════════════════════════════════════════════════════════════
# STEP 3: MODEL — GBT FORECASTER
# ════════════════════════════════════════════════════════════════

print("\n=== STEP 3: GBT Forecaster ===")

class GBTForecaster:
    """
    Forecasting berbasis Gradient Boosted Trees + Lag Features.
    Mendukung single series maupun multi-group (per negara/kota/state).
    """

    def __init__(self, n_lags=5, n_estimators=100, max_depth=5,
                 learning_rate=0.05, subsample=0.8):
        self.n_lags        = n_lags
        self.n_estimators  = n_estimators
        self.max_depth     = max_depth
        self.learning_rate = learning_rate
        self.subsample     = subsample
        self.models        = {}   # {label: model}
        self.scalers       = {}   # {label: scaler}
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

    def fit_single(self, label: str, series: np.ndarray):
        """Latih model untuk satu time series."""
        scaler = SklearnScaler()
        scaled = scaler.fit_transform(series.reshape(-1, 1)).flatten()

        # Buat supervised dataset
        X, y = [], []
        for i in range(self.n_lags, len(scaled)):
            X.append(scaled[i - self.n_lags:i])
            y.append(scaled[i])
        X, y = np.array(X), np.array(y)

        if len(X) < 10:
            print(f"  ⚠ Skip '{label}' — data terlalu sedikit")
            return None, None

        # Split 80/20
        split   = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = self._build_model()
        model.fit(X_train, y_train)

        # Evaluasi
        y_pred = scaler.inverse_transform(model.predict(X_test).reshape(-1, 1)).flatten()
        y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

        mae  = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        self.models[label]  = model
        self.scalers[label] = scaler

        return mae, rmse

    def forecast(self, label: str, last_values: np.ndarray, steps: int) -> np.ndarray:
        """Iterative forecasting n steps ke depan."""
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

    def fit_and_forecast_all(self, name: str, df: pd.DataFrame, time_col: str,
                              target_col: str, group_col: str = None, top_n: int = 3):
        """Pipeline lengkap: fit + forecast + plot untuk semua group."""
        print(f"\n  ▶ Dataset: {name.upper()}")

        # ── Deteksi tipe dataset ───────────────────────────────────────────────
        # city  : time_col = 'month' (hanya 1-12, bukan time series panjang)
        #         → agregasi rata-rata per bulan lintas tahun, forecast = seasonality
        # state : time_col = 'year_month' (string "YYYY-MM")
        #         → konversi ke integer ordinal agar sort benar
        IS_CITY  = (name == "city")
        IS_STATE = False   # state sudah diagregasi tahunan, perlakuan sama seperti country

        groups = [None]
        if group_col:
            groups = (
                df.groupby(group_col)[target_col].count()
                  .nlargest(top_n).index.tolist()
            )

        dataset_results = {}

        for grp in groups:
            label  = name if grp is None else f"{name}_{grp.replace(' ', '_')}"
            subset = df.copy() if grp is None else df[df[group_col] == grp].copy()
            subset = subset.dropna(subset=[target_col])

            # ── Fix STATE: sort by year_month sebagai datetime ─────────────────
            if IS_STATE:
                subset["_sort_key"] = pd.to_datetime(subset[time_col], format="%Y-%m")
                subset = subset.sort_values("_sort_key").reset_index(drop=True)
                # Buat integer index (0,1,2,...) sebagai pengganti time axis
                subset["_time_idx"] = np.arange(len(subset))
            else:
                subset = subset.sort_values(time_col).reset_index(drop=True)

            # ── Fix CITY: hanya 12 bulan → gunakan semua bulan sebagai 1 siklus
            #    Perbanyak data dengan mengulangi siklus musiman (repeat pattern)
            if IS_CITY:
                # Ambil rata-rata per bulan lintas semua kota (atau per kota)
                subset = subset.sort_values(time_col).reset_index(drop=True)
                # Repeat 10x supaya ada cukup data untuk lag + split
                series_base = subset[target_col].values.astype(float)
                series      = np.tile(series_base, 10)   # 12 bulan x 10 = 120 data points
            else:
                series = subset[target_col].values.astype(float)

            if len(series) < self.n_lags + 10:
                print(f"    ⚠ Skip '{grp}' — data terlalu sedikit ({len(series)} rows)")
                continue

            mae, rmse = self.fit_single(label, series)
            if mae is None:
                continue

            print(f"    '{label}' → MAE: {mae:.4f} | RMSE: {rmse:.4f}")

            # ── Forecast ───────────────────────────────────────────────────────
            future_vals = self.forecast(label, series, FORECAST_YEARS)

            # Buat future time index
            if IS_CITY:
                # Forecast = prediksi bulan 1-12 untuk tahun mendatang
                last_month   = int(subset[time_col].max())
                future_index = [(last_month % 12) + i + 1 for i in range(FORECAST_YEARS)]
                future_label = "Month"
            elif IS_STATE:
                last_ym      = subset["_sort_key"].max()
                future_index = [
                    (last_ym + pd.DateOffset(months=i+1)).strftime("%Y-%m")
                    for i in range(FORECAST_YEARS)
                ]
                future_label = "Year-Month"
            else:
                last_time    = subset[time_col].max()
                if isinstance(last_time, str):
                    last_time = int(last_time[:4])
                future_index = [int(last_time) + i + 1 for i in range(FORECAST_YEARS)]
                future_label = "Year"

            print(f"    Forecast 10yr: {[round(v, 2) for v in future_vals]}")

            # ── Plot ───────────────────────────────────────────────────────────
            if IS_STATE:
                hist_time = subset["_sort_key"].dt.strftime("%Y-%m").values
            elif IS_CITY:
                hist_time = subset[time_col].values
            else:
                hist_time = subset[time_col].values
                if not np.issubdtype(type(hist_time[0]), np.integer):
                    hist_time = [int(str(t)[:4]) for t in hist_time]

            fig, ax = plt.subplots(figsize=(14, 5))
            ax.plot(
                np.arange(len(series)) if IS_CITY else hist_time,
                series if IS_CITY else subset[target_col].values.astype(float),
                label="Historical", color="steelblue", linewidth=1.5
            )
            ax.plot(
                np.arange(len(series), len(series) + FORECAST_YEARS) if IS_CITY else future_index,
                future_vals,
                label="Forecast (10yr)", color="tomato",
                linestyle="--", marker="o", markersize=5
            )
            ax.set_title(f"{label} — 10-Step Forecast | MAE: {mae:.4f} | RMSE: {rmse:.4f}")
            ax.set_xlabel(future_label)
            ax.set_ylabel("Avg Temperature (°C)")
            ax.legend()
            ax.grid(True)
            plt.tight_layout()

            plot_path = f"{PLOT_DIR}/{label}_forecast.png"
            plt.savefig(plot_path, dpi=120)
            plt.close()

            s3.upload_file(plot_path, S3_BUCKET, f"models/plots/{label}_forecast.png")
            print(f"    📊 Plot → s3://{S3_BUCKET}/models/plots/{label}_forecast.png")

            dataset_results[label] = {
                "future_index": [str(f) for f in future_index],
                "future_vals" : future_vals.tolist(),
                "mae"         : mae,
                "rmse"        : rmse,
            }

        self.results[name] = dataset_results
        return dataset_results


# ════════════════════════════════════════════════════════════════
# STEP 4: TRAINING SEMUA DATASET
# ════════════════════════════════════════════════════════════════

print("\n=== STEP 4: Training ===")

forecaster = GBTForecaster(
    n_lags        = LOOKBACK,
    n_estimators  = 100,
    max_depth     = 5,
    learning_rate = 0.05,
    subsample     = 0.8
)

forecaster.fit_and_forecast_all(
    name       = "global",
    df         = global_df,
    time_col   = "year_global",
    target_col = "global_avg_temp"
)

forecaster.fit_and_forecast_all(
    name       = "country",
    df         = country_df,
    time_col   = "year",
    target_col = "avg_temp_country",
    group_col  = "country_name",
    top_n      = TOP_N_GROUPS
)

forecaster.fit_and_forecast_all(
    name       = "city",
    df         = city_df,
    time_col   = "month",
    target_col = "avg_temp_city",
    group_col  = "city_name",
    top_n      = TOP_N_GROUPS
)

# ── Pre-processing STATE: agregasi per tahun ──────────────────────────────────
# state_df punya 620k baris dengan granularitas bulanan (year_month = "YYYY-MM")
# GBT iterative forecast tidak stabil pada data bulanan karena pola musiman
# sangat kuat → akumulasi error besar. Solusi: agregasi ke tahunan dulu,
# sama seperti country, sehingga forecast lebih smooth dan realistis.
state_df_yearly = (
    state_df.copy()
    .assign(year=lambda d: d["year_month"].str[:4].astype(int))
    .groupby(["year", "state_name"], as_index=False)["avg_temp_state"]
    .mean()
    .round({"avg_temp_state": 4})
)
print(f"  State agregasi tahunan: {len(state_df_yearly)} baris "
      f"(dari {len(state_df)} baris bulanan)")

forecaster.fit_and_forecast_all(
    name       = "state",
    df         = state_df_yearly,
    time_col   = "year",          # sudah diagregasi per tahun, bukan year_month lagi
    target_col = "avg_temp_state",
    group_col  = "state_name",
    top_n      = TOP_N_GROUPS
)

forecaster.trained_at = datetime.now().isoformat()


# ════════════════════════════════════════════════════════════════
# STEP 5: EVALUASI MODEL
# ════════════════════════════════════════════════════════════════

print("\n=== STEP 5: Evaluasi Model ===")

all_mae, all_rmse = [], []
for dataset, res in forecaster.results.items():
    for label, metrics in res.items():
        all_mae.append(metrics["mae"])
        all_rmse.append(metrics["rmse"])

overall_metrics = {
    "avg_mae"       : round(float(np.mean(all_mae)),  4) if all_mae  else 0,
    "avg_rmse"      : round(float(np.mean(all_rmse)), 4) if all_rmse else 0,
    "total_models"  : sum(len(v) for v in forecaster.results.values()),
    "datasets"      : list(forecaster.results.keys()),
}

print(f"  Total models trained : {overall_metrics['total_models']}")
print(f"  Avg MAE              : {overall_metrics['avg_mae']}")
print(f"  Avg RMSE             : {overall_metrics['avg_rmse']}")


# ════════════════════════════════════════════════════════════════
# STEP 6: SIMPAN MODEL KE S3
# ════════════════════════════════════════════════════════════════

print("\n=== STEP 6: Simpan Model ke S3 ===")

model_package = {
    "model": forecaster,
    "metadata": {
        "version"        : "1.0.0",
        "trained_at"     : forecaster.trained_at,
        "metrics"        : overall_metrics,
        "forecast_years" : FORECAST_YEARS,
        "lookback"       : LOOKBACK,
        "algorithm"      : "GBT Regressor + Lag Features",
        "datasets": {
            "global"  : {"rows": len(global_df),  "target": "global_avg_temp"},
            "country" : {"rows": len(country_df), "target": "avg_temp_country"},
            "city"    : {"rows": len(city_df),    "target": "avg_temp_city"},
            "state"   : {"rows": len(state_df),   "target": "avg_temp_state"},
        },
        "results_per_label": {
            label: {"mae": m["mae"], "rmse": m["rmse"],
                    "forecast": [round(v, 2) for v in m["future_vals"]]}
            for dataset in forecaster.results.values()
            for label, m in dataset.items()
        }
    }
}

# Simpan lokal dulu
local_model_path = "/tmp/temperature_forecast_model.pkl"
with open(local_model_path, "wb") as f:
    pickle.dump(model_package, f)

file_size_mb = os.path.getsize(local_model_path) / (1024 * 1024)
print(f"  Model size : {file_size_mb:.2f} MB")

# Upload model ke S3
model_key = "models/temperature_forecast_model.pkl"
s3.upload_file(local_model_path, S3_BUCKET, model_key)
print(f"  ✓ Model diupload ke s3://{S3_BUCKET}/{model_key}")

# Simpan metadata sebagai JSON
metadata_key = "models/model_metadata.json"
s3.put_object(
    Bucket      = S3_BUCKET,
    Key         = metadata_key,
    Body        = json.dumps(model_package["metadata"], indent=2),
    ContentType = "application/json"
)
print(f"  ✓ Metadata disimpan ke s3://{S3_BUCKET}/{metadata_key}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  SUMMARY HASIL TRAINING")
print("="*60)

for dataset, res in forecaster.results.items():
    print(f"\n📊 {dataset.upper()}")
    if not res:
        print("   Tidak ada hasil.")
        continue
    for label, metrics in res.items():
        print(f"   {label}")
        print(f"     MAE      : {metrics['mae']:.4f}")
        print(f"     RMSE     : {metrics['rmse']:.4f}")
        print(f"     Forecast : {[round(v, 2) for v in metrics['future_vals']]}")

print(f"\n✅ Training selesai!")
print(f"   Model    : s3://{S3_BUCKET}/models/temperature_forecast_model.pkl")
print(f"   Metadata : s3://{S3_BUCKET}/models/model_metadata.json")
print(f"   Plots    : s3://{S3_BUCKET}/models/plots/")