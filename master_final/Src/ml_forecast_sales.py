from __future__ import annotations

import os
import math
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from dotenv import load_dotenv
from sqlalchemy import create_engine, text 
from sqlalchemy.exc import SQLAlchemyError
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# load_dotenv()

# BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = Path("../outputs/")
# OUTPUT_DIR.mkdir(exist_ok=True)

DB_USER = "admin"
DB_PASSWORD = "password"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME =  "projectdb"
DB_SCHEMA = "m5"
SOURCE_VIEW = "sales_modeling_view"

TEST_WEEKS = 12
FORECAST_WEEKS = 4
# WRITE_TO_DB = os.getenv("WRITE_TO_DB", "true").lower() in {"true", "1", "yes", "y"}

RANDOM_STATE = 42
N_ESTIMATORS = 150
MIN_SAMPLES_LEAF = 2

GROUP_COLS = ["store_id", "cat_id"]
TARGET_COL = "weekly_units_sold"

NUMERIC_FEATURES = [
    "year",
    "month",
    "quarter",
    "week_of_year",
    "lag_1",
    "lag_2",
    "lag_4",
    "rolling_4",
    "rolling_8",
]

CATEGORICAL_FEATURES = ["store_id", "cat_id"]
FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# -----------------------------------------------------------------------------
# Database connection and loading
# -----------------------------------------------------------------------------

def get_engine():
    """Create a SQLAlchemy engine for PostgreSQL."""
    url = (
        f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}"
        f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    return create_engine(url, future=True)


def test_connection(engine) -> None:
    """Fail early if PostgreSQL is not reachable."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("Database connection successful.")
    except SQLAlchemyError as exc:
        raise RuntimeError(
            "Could not connect to PostgreSQL. Check Docker, database credentials."
        ) from exc


def load_weekly_sales(engine) -> pd.DataFrame:
  
    query = f"""
        SELECT
            DATE_TRUNC('week', calendar_date)::date AS week_start,
            store_id,
            cat_id,
            SUM(units_sold)::float AS weekly_units_sold,
            SUM(COALESCE(revenue, 0))::float AS weekly_revenue
        FROM {DB_SCHEMA}.{SOURCE_VIEW}
        WHERE calendar_date IS NOT NULL
        GROUP BY
            DATE_TRUNC('week', calendar_date)::date,
            store_id,
            cat_id
        ORDER BY
            store_id,
            cat_id,
            week_start;
    """

    try:
        df = pd.read_sql(query, engine)
    except SQLAlchemyError as exc:
        raise RuntimeError(
            f"Could not load data from {DB_SCHEMA}.{SOURCE_VIEW}. "
            "Make sure the view exists and contains calendar_date, store_id, cat_id, "
            "units_sold, and revenue."
        ) from exc

    if df.empty:
        raise ValueError(
            f"No rows returned from {DB_SCHEMA}.{SOURCE_VIEW}. Check that the M5 data loaded correctly."
        )

    df["week_start"] = pd.to_datetime(df["week_start"])
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce").fillna(0)
    df["weekly_revenue"] = pd.to_numeric(df["weekly_revenue"], errors="coerce").fillna(0)
    df = df.sort_values(GROUP_COLS + ["week_start"]).reset_index(drop=True)

    print(f"Loaded weekly sales rows: {len(df):,}")
    print(f"Date range: {df['week_start'].min().date()} to {df['week_start'].max().date()}")
    print(f"Stores: {df['store_id'].nunique()}, Categories: {df['cat_id'].nunique()}")

    return df

# -----------------------------------------------------------------------------
# Feature engineering
# -----------------------------------------------------------------------------

def add_date_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    df["year"] = df["week_start"].dt.year
    df["month"] = df["week_start"].dt.month
    df["quarter"] = df["week_start"].dt.quarter
    df["week_of_year"] = df["week_start"].dt.isocalendar().week.astype(int)
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy().sort_values(GROUP_COLS + ["week_start"])
    grouped = df.groupby(GROUP_COLS, group_keys=False)[TARGET_COL]

    df["lag_1"] = grouped.shift(1)
    df["lag_2"] = grouped.shift(2)
    df["lag_4"] = grouped.shift(4)

    df["rolling_4"] = grouped.transform(
        lambda s: s.shift(1).rolling(window=4, min_periods=1).mean()
    )
    df["rolling_8"] = grouped.transform(
        lambda s: s.shift(1).rolling(window=8, min_periods=1).mean()
    )

    return df


def prepare_modeling_data(weekly_df: pd.DataFrame) -> pd.DataFrame:
 
    df = add_date_features(weekly_df)
    df = add_lag_features(df)

    # Rows at the very beginning of each store/category series will not have lags.
    df = df.dropna(subset=["lag_1", "lag_2", "lag_4", "rolling_4", "rolling_8"])
    df = df.sort_values(GROUP_COLS + ["week_start"]).reset_index(drop=True)

    if df.empty:
        raise ValueError("After creating lag features, no modeling rows remain.")

    return df

# -----------------------------------------------------------------------------
# Model and metrics
# -----------------------------------------------------------------------------

def build_model() -> Pipeline:
    """Build the Random Forest model pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    regressor = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        min_samples_leaf=MIN_SAMPLES_LEAF,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", regressor),
        ]
    )


def build_linear_regression_model() -> Pipeline:
    """Build Linear Regression comparison model."""

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    regressor = LinearRegression()

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", regressor),
        ]
    )

