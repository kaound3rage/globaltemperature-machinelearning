import pandas as pd

df = pd.read_parquet("run-1773835489976-part-block-0-r-00000-snappy.parquet")
print(df.head())