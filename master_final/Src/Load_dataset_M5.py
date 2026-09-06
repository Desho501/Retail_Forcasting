from __future__ import annotations

import os
from io import StringIO
from pathlib import Path

import pandas as pd
# from dotenv import load_dotenv
from sqlalchemy import create_engine

# load_dotenv()

CALENDAR_CSV = "../m5_dataset/calendar.csv"
PRICES_CSV = "../m5_dataset/sell_prices.csv"
SALES_CSV = "../m5_dataset/sales_train_evaluation.csv" 
SCHEMA_NAME = "m5"
SALES_CHUNKSIZE = 500 


def get_engine():
    user = "admin"
    password = "password"
    host = "localhost"
    port = "5432"
    db = "projectdb"
    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"
    return create_engine(url, future=True)



def copy_dataframe(engine, df: pd.DataFrame, table_name: str, schema: str = SCHEMA_NAME) -> None:
    """Fast PostgreSQL COPY load from a pandas DataFrame."""
    if df.empty:
        return

    buffer = StringIO()
    df.to_csv(buffer, index=False, header=False, na_rep="\\N")
    buffer.seek(0)

    columns = ", ".join([f'"{c}"' for c in df.columns])
    copy_sql = f"""
        COPY {schema}.{table_name} ({columns})
        FROM STDIN WITH (FORMAT CSV, NULL '\\N')
    """

    raw_conn = engine.raw_connection()
    try:
        with raw_conn.cursor() as cur:
            cur.copy_expert(copy_sql, buffer)
        raw_conn.commit()
    except Exception:
        raw_conn.rollback()
        raise
    finally:
        raw_conn.close()


def read_calendar() -> pd.DataFrame:
    calendar = pd.read_csv(CALENDAR_CSV, parse_dates=["date"])
    calendar = calendar.rename(
        columns={
            "date": "calendar_date",
        }
    )
    calendar["calendar_date"] = calendar["calendar_date"].dt.date
    return calendar


def load_weeks_and_calendar(engine) -> None:
    """
    Load weeks first because calendar.wm_yr_wk has a foreign key to weeks.wm_yr_wk.
    """
    calendar = read_calendar()

    weeks = calendar[["wm_yr_wk"]].drop_duplicates().sort_values("wm_yr_wk")
    copy_dataframe(engine, weeks, "weeks")
    copy_dataframe(engine, calendar, "calendar")


def load_dimensions_from_sales(engine) -> None:
    """
    Load dimensions in parent-to-child order so foreign keys do not fail.
    """
    meta_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    meta = pd.read_csv(SALES_CSV, usecols=meta_cols)

    states = meta[["state_id"]].drop_duplicates().sort_values("state_id")
    categories = meta[["cat_id"]].drop_duplicates().sort_values("cat_id")
    stores = meta[["store_id", "state_id"]].drop_duplicates().sort_values("store_id")
    departments = meta[["dept_id", "cat_id"]].drop_duplicates().sort_values("dept_id")
    items = meta[["item_id", "dept_id", "cat_id"]].drop_duplicates().sort_values("item_id")
    store_items = (
        meta[["id", "item_id", "store_id"]]
        .drop_duplicates()
        .rename(columns={"id": "series_id"})
        .sort_values("series_id")
    )

    copy_dataframe(engine, states, "states")
    copy_dataframe(engine, categories, "categories")
    copy_dataframe(engine, stores, "stores")
    copy_dataframe(engine, departments, "departments")
    copy_dataframe(engine, items, "items")
    copy_dataframe(engine, store_items, "store_items")


def load_sell_prices(engine) -> None:
    """
    Load sell prices after stores, items, and weeks are already loaded.
    """
    for chunk_number, prices in enumerate(pd.read_csv(PRICES_CSV, chunksize=250_000), start=1):
        prices["sell_price"] = prices["sell_price"].round(2)
        copy_dataframe(engine, prices, "sell_prices")
        print(f"Loaded price chunk {chunk_number}")


def load_sales_daily(engine) -> None:
    """
    Convert the wide M5 d_1...d_1941 sales columns into long database rows.
    Load after store_items and calendar are already loaded.
    """
    for chunk_number, chunk in enumerate(pd.read_csv(SALES_CSV, chunksize=SALES_CHUNKSIZE), start=1):
        day_cols = [c for c in chunk.columns if c.startswith("d_")]

        long_sales = chunk.melt(
            id_vars=["id"],
            value_vars=day_cols,
            var_name="d",
            value_name="units_sold",
        ).rename(columns={"id": "series_id"})

        long_sales["units_sold"] = long_sales["units_sold"].astype("int32")
        copy_dataframe(engine, long_sales[["series_id", "d", "units_sold"]], "sales_daily")
        print(f"Loaded sales chunk {chunk_number}")


def main() -> None:
    engine = get_engine()

    # 1. Create tables and foreign keys
    # run_schema(engine)

    # 2. Load parent/date tables
    load_weeks_and_calendar(engine)

    # 3. Load product/store dimensions
    load_dimensions_from_sales(engine)

    # 4. Load fact tables last
    load_sell_prices(engine)
    load_sales_daily(engine)

    print("M5 data load complete.")


if __name__ == "__main__":
    main()
