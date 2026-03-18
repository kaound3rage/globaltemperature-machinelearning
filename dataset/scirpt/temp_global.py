import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

# ── Init ──────────────────────────────────────────────────────────────────────
args = getResolvedOptions(sys.argv, ["JOB_NAME"])
sc   = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job   = Job(glueContext)
job.init(args["JOB_NAME"], args)

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_PATH  = "s3://globalandtemperature/raw-data/temp_global/"
OUTPUT_PATH = "s3://globalandtemperature/processed-data/temp_global/"

TEMP_COLS = [
    "LandAverageTemperature",
    "LandAverageTemperatureUncertainty",
    "LandMaxTemperature",
    "LandMaxTemperatureUncertainty",
    "LandMinTemperature",
    "LandMinTemperatureUncertainty",
    "LandAndOceanAverageTemperature",
    "LandAndOceanAverageTemperatureUncertainty",
]

# ── 1. Read CSV ───────────────────────────────────────────────────────────────
df = spark.read.option("header", "true").csv(INPUT_PATH)

# ── 2. Cast kolom temperatur ke Double ───────────────────────────────────────
for col in TEMP_COLS:
    df = df.withColumn(col, F.col(col).cast(DoubleType()))

# ── 3. Parse kolom tanggal & ekstrak tahun + bulan ───────────────────────────
df = df.withColumn("dt", F.to_date(F.col("dt"), "yyyy-MM-dd")) \
       .withColumn("year",  F.year("dt")) \
       .withColumn("month", F.month("dt"))

# ── 4. Bersihkan null/missing values ─────────────────────────────────────────
# Drop baris yang kolom temperatur utamanya null
df_clean = df.dropna(subset=[
    "LandAverageTemperature",
    "LandAndOceanAverageTemperature"
])

# ── 5. Agregasi rata-rata per bulan (year + month) ────────────────────────────
df_agg = df_clean.groupBy("year", "month").agg(
    F.round(F.avg("LandAverageTemperature"),            4).alias("avg_LandAverageTemperature"),
    F.round(F.avg("LandAverageTemperatureUncertainty"), 4).alias("avg_LandAverageTemperatureUncertainty"),
    F.round(F.avg("LandMaxTemperature"),                4).alias("avg_LandMaxTemperature"),
    F.round(F.avg("LandMaxTemperatureUncertainty"),     4).alias("avg_LandMaxTemperatureUncertainty"),
    F.round(F.avg("LandMinTemperature"),                4).alias("avg_LandMinTemperature"),
    F.round(F.avg("LandMinTemperatureUncertainty"),     4).alias("avg_LandMinTemperatureUncertainty"),
    F.round(F.avg("LandAndOceanAverageTemperature"),            4).alias("avg_LandAndOceanAverageTemperature"),
    F.round(F.avg("LandAndOceanAverageTemperatureUncertainty"), 4).alias("avg_LandAndOceanAverageTemperatureUncertainty"),
) \
.orderBy("year", "month")

# ── 6. Tulis output ke S3 sebagai Parquet ─────────────────────────────────────
df_agg.write \
    .mode("overwrite") \
    .partitionBy("year") \
    .parquet(OUTPUT_PATH)

job.commit()