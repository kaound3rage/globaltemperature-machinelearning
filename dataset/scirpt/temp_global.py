import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

# ── Init ──────────────────────────────────────────────────────────────────────
args = getResolvedOptions(sys.argv, ["JOB_NAME", "file_type"])
sc   = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job   = Job(glueContext)
job.init(args["JOB_NAME"], args)

FILE_TYPE = args["file_type"].lower()  # global | country | city | state

BUCKET    = "globallandtemperature"
INPUT_PATH  = f"s3://{BUCKET}/raw-data/{FILE_TYPE}/clean_{FILE_TYPE}.csv"
OUTPUT_PATH = f"s3://{BUCKET}/processed-data/{FILE_TYPE}/"

print(f"▶ Starting ETL for file_type='{FILE_TYPE}'")
print(f"  Input  : {INPUT_PATH}")
print(f"  Output : {OUTPUT_PATH}")

# ── Read CSV ──────────────────────────────────────────────────────────────────
df = spark.read.option("header", "true").csv(INPUT_PATH)

# =============================================================================
# TRANSFORMASI PER FILE TYPE
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
if FILE_TYPE == "global":
    # Schema: year_global, global_avg_temp
    df = df.withColumn("year_global",    F.col("year_global").cast("integer")) \
           .withColumn("global_avg_temp", F.col("global_avg_temp").cast(DoubleType()))

    df_clean = df.dropna(subset=["year_global", "global_avg_temp"])

    df_out = df_clean.groupBy("year_global").agg(
        F.round(F.avg("global_avg_temp"), 4).alias("global_avg_temp")
    ).orderBy("year_global")

    df_out.write.mode("overwrite").parquet(OUTPUT_PATH)

# ─────────────────────────────────────────────────────────────────────────────
elif FILE_TYPE == "country":
    # Schema: year, country_name, avg_temp_country
    df = df.withColumn("year",             F.col("year").cast("integer")) \
           .withColumn("avg_temp_country", F.col("avg_temp_country").cast(DoubleType()))

    df_clean = df.dropna(subset=["year", "country_name", "avg_temp_country"])

    df_out = df_clean.groupBy("year", "country_name").agg(
        F.round(F.avg("avg_temp_country"), 4).alias("avg_temp_country")
    ).orderBy("year", "country_name")

    df_out.write.mode("overwrite").partitionBy("year").parquet(OUTPUT_PATH)

# ─────────────────────────────────────────────────────────────────────────────
elif FILE_TYPE == "city":
    # Schema: month, city_name, avg_temp_city
    df = df.withColumn("month",         F.col("month").cast("integer")) \
           .withColumn("avg_temp_city", F.col("avg_temp_city").cast(DoubleType()))

    df_clean = df.dropna(subset=["month", "city_name", "avg_temp_city"])

    df_out = df_clean.groupBy("month", "city_name").agg(
        F.round(F.avg("avg_temp_city"), 4).alias("avg_temp_city")
    ).orderBy("month", "city_name")

    df_out.write.mode("overwrite").partitionBy("month").parquet(OUTPUT_PATH)

# ─────────────────────────────────────────────────────────────────────────────
elif FILE_TYPE == "state":
    # Schema: year_month, state_name, avg_temp_state
    df = df.withColumn("avg_temp_state", F.col("avg_temp_state").cast(DoubleType())) \
           .withColumn("year",  F.col("year_month").substr(1, 4).cast("integer")) \
           .withColumn("month", F.col("year_month").substr(6, 2).cast("integer"))

    df_clean = df.dropna(subset=["year_month", "state_name", "avg_temp_state"])

    df_out = df_clean.groupBy("year_month", "year", "month", "state_name").agg(
        F.round(F.avg("avg_temp_state"), 4).alias("avg_temp_state")
    ).orderBy("year_month", "state_name")

    df_out.write.mode("overwrite").partitionBy("year").parquet(OUTPUT_PATH)

# ─────────────────────────────────────────────────────────────────────────────
else:
    raise ValueError(f"file_type tidak dikenal: '{FILE_TYPE}'. Pilih: global | country | city | state")

# ── Done ──────────────────────────────────────────────────────────────────────
print(f"✅ ETL selesai untuk file_type='{FILE_TYPE}' → {OUTPUT_PATH}")
job.commit()