import pandas as pd
import numpy as np
from sqlalchemy import create_engine

# ==========================================
# DATABASE CONNECTION
# ==========================================

DB_USER = "admin"
DB_PASSWORD = "password"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "projectdb"

DB_connect = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("../Walmart_dataset/Walmart_Dataset_Update.csv")

print("Dataset Loaded Successfully")
print(df.head())

# ==========================================
# DATA PREPROCESSING
# ==========================================

# -----------------------------
# Convert Date Column
# -----------------------------
df['date'] = pd.to_datetime(
    df['date'],
    format='%d-%m-%Y',
    errors='coerce'
)

# -----------------------------
# Check Missing Values
# -----------------------------
print("\nMissing Values:")
print(df.isnull().sum())

# -----------------------------
# Remove Duplicate Rows
# -----------------------------
duplicates = df.duplicated().sum()

print(f"\nDuplicate Rows Found: {duplicates}")

df = df.drop_duplicates()

# -----------------------------
# Validate Numeric Columns
# -----------------------------
numeric_columns = [
    'weekly_revenue',
    'temperature',
    'fuel_price',
    'consumer_price_index',
    'unemployment_rate'
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Remove rows with invalid numeric values
df = df.dropna(subset=numeric_columns)


# ==========================================
# FINAL DATA VALIDATION
# ==========================================

print("\nFinal Dataset Info:")
print(df.info())

print("\nFinal Dataset Shape:")
print(df.shape)

# ==========================================
# INSERT DATA INTO POSTGRESQL
# ==========================================

table_name = "retail_sales"
df['holiday_flag'] = df['holiday_flag'].astype(bool)

df.to_sql(
    table_name,
    DB_connect,
    if_exists='append',
    schema='retail',
    index=False
)

print(f"\nData inserted successfully into table: {table_name}")