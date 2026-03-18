import pandas as pd
import boto3
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FILES = {
    "country": "GlobalLandTemperaturesByCountry.csv",
    "city": "GlobalLandTemperaturesByMajorCity.csv",
    "state": "GlobalLandTemperaturesByState.csv",
    "global": "GlobalTemperatures.csv"
}

BUCKET_NAME = "globallandtemperature"
s3 = boto3.client("s3")

def process_file(file_type, filename):
    path = os.path.join(BASE_DIR, filename)
    df = pd.read_csv(path)

    print(f"Processing {file_type}...")

    # =====================
    # CLEAN + CUSTOM SCHEMA
    # =====================

    if file_type == "country":
        df = df[["dt", "AverageTemperature", "Country"]].dropna()
        df["dt"] = pd.to_datetime(df["dt"])
        df["year"] = df["dt"].dt.year

        df = df.groupby(["year", "Country"])["AverageTemperature"].mean().reset_index()

        df.columns = ["year", "country_name", "avg_temp_country"]

    elif file_type == "city":
        df = df[["dt", "AverageTemperature", "City"]].dropna()
        df["dt"] = pd.to_datetime(df["dt"])
        df["month"] = df["dt"].dt.month

        df = df.groupby(["month", "City"])["AverageTemperature"].mean().reset_index()

        df.columns = ["month", "city_name", "avg_temp_city"]

    elif file_type == "state":
        df = df[["dt", "AverageTemperature", "State"]].dropna()
        df["dt"] = pd.to_datetime(df["dt"])
        df["year_month"] = df["dt"].dt.to_period("M")

        df = df.groupby(["year_month", "State"])["AverageTemperature"].mean().reset_index()

        df["year_month"] = df["year_month"].astype(str)
        df.columns = ["year_month", "state_name", "avg_temp_state"]

    elif file_type == "global":
        df = df[["dt", "LandAverageTemperature"]].dropna()
        df["dt"] = pd.to_datetime(df["dt"])
        df["year"] = df["dt"].dt.year

        df = df.groupby("year")["LandAverageTemperature"].mean().reset_index()

        df.columns = ["year_global", "global_avg_temp"]

    # =====================
    # SAVE
    # =====================
    output_file = os.path.join(BASE_DIR, f"clean_{file_type}.csv")
    df.to_csv(output_file, index=False)

    print(f"✅ {file_type} cleaned!")

    # =====================
    # UPLOAD S3
    # =====================
    s3_key = f"raw-data/{file_type}/clean_{file_type}.csv"
    s3.upload_file(output_file, BUCKET_NAME, s3_key)

    print(f"🚀 Uploaded: {s3_key}")


# RUN
for file_type, filename in FILES.items():
    process_file(file_type, filename)

print("🔥 Semua dataset berhasil diproses dengan schema berbeda!")