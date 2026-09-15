import os

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import text

FEATURE_COLS = [
    "product_model",
    "product_color",
    "product_size",
    "is_muslim_fashion",
    "month",
    "is_twin_date",
    "is_ramadhan",
    "rolling_mean_3_qty",
    "rolling_mean_7_qty",
    "rolling_std_7_qty",
    "lag_1_qty",
    "lag_3_qty",
    "lag_7_qty",
    "lag_7_rolling_mean",
    "day_of_year",
    "is_month_start",
]
CATEGORICAL_FEATURES = ["product_model", "product_color", "product_size", "month"]

MODEL_PATH_V2 = os.path.join(
    os.path.dirname(__file__), "saved_models", "champion_model_v2.pkl"
)


def _model_categories(model):
    categories = {}
    pandas_categories = getattr(getattr(model, "booster_", None), "pandas_categorical", None)
    if pandas_categories:
        for column, values in zip(CATEGORICAL_FEATURES, pandas_categories):
            categories[column] = [str(value) for value in values]
    return categories


def _prepare_model_row(values, categories):
    row = pd.DataFrame([{column: values[column] for column in FEATURE_COLS}])
    for column in CATEGORICAL_FEATURES:
        known_categories = categories.get(column)
        if known_categories:
            row[column] = pd.Categorical(
                row[column].astype(str), categories=known_categories
            )
        else:
            row[column] = row[column].astype(str).astype("category")
    return row[FEATURE_COLS]


def _get_forecast_dimensions(db_engine, history_days):
    latest_query = """
        SELECT MAX(dd.date) AS last_date
        FROM fact_daily_agregat fd
        JOIN date_dimension dd ON dd.date_id = fd.date_id
        WHERE fd.status = 'SELESAI'
    """
    product_query = """
        SELECT product_id, product_model, product_color, product_size,
               is_muslim_fashion
        FROM product_dimension
    """
    with db_engine.connect() as conn:
        latest_date = pd.read_sql(text(latest_query), conn).iloc[0]["last_date"]
        products = pd.read_sql(text(product_query), conn)

    if pd.isna(latest_date) or products.empty:
        raise ValueError("Data produk atau tanggal penjualan belum tersedia.")

    last_date = pd.Timestamp(latest_date)
    start_date = last_date - pd.Timedelta(days=history_days - 1)
    date_query = """
        SELECT date AS order_date, month, is_twin_date, is_ramadhan
        FROM date_dimension
        WHERE date BETWEEN :start_date AND :end_date
    """
    sales_query = """
        SELECT dd.date AS order_date, fd.product_id,
               SUM(fd.total_quantity) AS quantity
        FROM fact_daily_agregat fd
        JOIN date_dimension dd ON dd.date_id = fd.date_id
        WHERE fd.status = 'SELESAI'
          AND dd.date BETWEEN :start_date AND :end_date
        GROUP BY dd.date, fd.product_id
    """
    params = {"start_date": start_date.date(), "end_date": last_date.date()}
    with db_engine.connect() as conn:
        dates = pd.read_sql(text(date_query), conn, params=params)
        sales = pd.read_sql(text(sales_query), conn, params=params)

    dates["order_date"] = pd.to_datetime(dates["order_date"])
    sales["order_date"] = pd.to_datetime(sales["order_date"])
    products["is_muslim_fashion"] = products["is_muslim_fashion"].fillna(False).astype(int)
    return products, dates, sales, last_date


def _get_future_dates(db_engine, last_date, days_ahead):
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1), periods=days_ahead, freq="D"
    )
    query = """
        SELECT date AS order_date, month, is_twin_date, is_ramadhan
        FROM date_dimension
        WHERE date BETWEEN :start_date AND :end_date
    """
    with db_engine.connect() as conn:
        future_properties = pd.read_sql(
            text(query),
            conn,
            params={"start_date": future_dates.min().date(), "end_date": future_dates.max().date()},
        )
    future_properties["order_date"] = pd.to_datetime(future_properties["order_date"])
    future_properties = pd.DataFrame({"order_date": future_dates}).merge(
        future_properties, on="order_date", how="left"
    )
    if future_properties[["month", "is_twin_date", "is_ramadhan"]].isna().any().any():
        raise ValueError("date_dimension belum mencakup seluruh periode forecast v2.")
    return future_properties


def generate_forecast_v2(
    db_engine,
    model_path=MODEL_PATH_V2,
    days_ahead=14,
    models=None,
    colors=None,
    sizes=None,
):
    """Forecast recursive per SKU using the latest per-product model contract."""
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model v2 belum tersedia. Jalankan ml_models/retrainv2.py.")

    model = joblib.load(model_path)
    products, dates, sales, last_date = _get_forecast_dimensions(db_engine, history_days=60)
    all_products = products.copy()

    if models:
        products = products[products["product_model"].isin(models)]
    if colors:
        products = products[products["product_color"].isin(colors)]
    if sizes:
        products = products[products["product_size"].isin(sizes)]
    if products.empty:
        raise ValueError("Tidak ada SKU yang cocok dengan filter forecast.")

    dates = dates.sort_values("order_date")
    panel = products.assign(_key=1).merge(
        dates.assign(_key=1), on="_key", how="inner"
    ).drop(columns="_key")
    panel = panel.merge(sales, on=["product_id", "order_date"], how="left")
    panel["quantity"] = panel["quantity"].fillna(0).astype(float)
    panel = panel.sort_values(["product_id", "order_date"])

    future_properties = _get_future_dates(db_engine, last_date, days_ahead)
    categories = _model_categories(model)
    product_forecasts = []
    daily_forecasts = {date.strftime("%Y-%m-%d"): 0 for date in future_properties["order_date"]}

    for product in products.itertuples(index=False):
        product_history = panel.loc[panel["product_id"] == product.product_id].sort_values(
            "order_date"
        )["quantity"].tolist()
        for future in future_properties.itertuples(index=False):
            recent = product_history[-7:]
            previous_week = product_history[-14:-7]
            feature_values = {
                "product_model": product.product_model,
                "product_color": product.product_color,
                "product_size": product.product_size,
                "is_muslim_fashion": product.is_muslim_fashion,
                "month": future.month,
                "is_twin_date": int(future.is_twin_date),
                "is_ramadhan": int(future.is_ramadhan),
                "rolling_mean_3_qty": float(np.mean(product_history[-3:])) if product_history else 0.0,
                "rolling_mean_7_qty": float(np.mean(recent)) if recent else 0.0,
                "rolling_std_7_qty": float(np.std(recent, ddof=1)) if len(recent) > 1 else 0.0,
                "lag_1_qty": product_history[-1] if len(product_history) >= 1 else 0.0,
                "lag_3_qty": product_history[-3] if len(product_history) >= 3 else 0.0,
                "lag_7_qty": product_history[-7] if len(product_history) >= 7 else 0.0,
                "lag_7_rolling_mean": float(np.mean(previous_week)) if previous_week else 0.0,
                "day_of_year": future.order_date.dayofyear,
                "is_month_start": int(future.order_date.is_month_start),
            }
            model_row = _prepare_model_row(feature_values, categories)
            prediction = max(0, int(np.rint(model.predict(model_row)[0])))
            product_history.append(prediction)
            date_key = future.order_date.strftime("%Y-%m-%d")
            daily_forecasts[date_key] += prediction
            product_forecasts.append(
                {
                    "product_id": int(product.product_id),
                    "product_model": product.product_model,
                    "product_color": product.product_color,
                    "product_size": product.product_size,
                    "forecast_qty": prediction,
                }
            )

    product_summary = (
        pd.DataFrame(product_forecasts)
        .groupby(
            ["product_id", "product_model", "product_color", "product_size"],
            as_index=False,
        )["forecast_qty"]
        .sum()
        .sort_values("forecast_qty", ascending=False)
    )
    options = {
        "models": sorted(all_products["product_model"].dropna().unique().tolist()),
        "colors": sorted(all_products["product_color"].dropna().unique().tolist()),
        "sizes": sorted(all_products["product_size"].dropna().unique().tolist()),
    }
    forecast_daily = [
        {"date": date, "predicted_qty": int(quantity)}
        for date, quantity in daily_forecasts.items()
    ]
    return {
        "total_forecast": int(sum(daily_forecasts.values())),
        "forecast_daily": forecast_daily,
        "products": product_summary.to_dict(orient="records"),
        "options": options,
        "selected": {
            "models": models or [],
            "colors": colors or [],
            "sizes": sizes or [],
        },
    }
