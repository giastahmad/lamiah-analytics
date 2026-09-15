import os
import sys

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sqlalchemy import text

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import engine

MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
CHAMPION_PATH_V2 = os.path.join(MODEL_DIR, "champion_model_v2.pkl")
os.makedirs(MODEL_DIR, exist_ok=True)

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


def fetch_and_engineer_features_v2(db_engine=engine) -> pd.DataFrame:
    """Build the production equivalent of the notebook's zero-filled panel."""
    product_query = """
        SELECT product_id, product_model, product_color, product_size,
               is_muslim_fashion
        FROM product_dimension
    """
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
    bounds_query = """
        SELECT MIN(dd.date) AS start_date, MAX(dd.date) AS end_date
        FROM fact_daily_agregat fd
        JOIN date_dimension dd ON dd.date_id = fd.date_id
        WHERE fd.status = 'SELESAI'
    """

    with db_engine.connect() as conn:
        bounds = pd.read_sql(text(bounds_query), conn).iloc[0]
        if pd.isna(bounds["start_date"]) or pd.isna(bounds["end_date"]):
            raise ValueError("Tidak ada data penjualan SELESAI untuk retraining v2.")

        params = {
            "start_date": pd.Timestamp(bounds["start_date"]).date(),
            "end_date": pd.Timestamp(bounds["end_date"]).date(),
        }
        products = pd.read_sql(text(product_query), conn)
        dates = pd.read_sql(text(date_query), conn, params=params)
        sales = pd.read_sql(text(sales_query), conn, params=params)

    if products.empty or dates.empty:
        raise ValueError("Dimensi produk atau tanggal kosong untuk retraining v2.")

    dates["order_date"] = pd.to_datetime(dates["order_date"])
    sales["order_date"] = pd.to_datetime(sales["order_date"])
    products["is_muslim_fashion"] = products["is_muslim_fashion"].fillna(False).astype(int)

    dates["_key"] = 1
    products["_key"] = 1
    panel = products.merge(dates, on="_key", how="inner").drop(columns="_key")
    panel = panel.merge(sales, on=["product_id", "order_date"], how="left")
    panel["quantity"] = panel["quantity"].fillna(0).astype(float)
    panel = panel.sort_values(["product_id", "order_date"]).reset_index(drop=True)

    grouped_quantity = panel.groupby("product_id")["quantity"]
    panel["lag_1_qty"] = grouped_quantity.shift(1)
    panel["lag_3_qty"] = grouped_quantity.shift(3)
    panel["lag_7_qty"] = grouped_quantity.shift(7)
    panel["rolling_mean_3_qty"] = grouped_quantity.transform(
        lambda values: values.shift(1).rolling(window=3, min_periods=1).mean()
    )
    panel["rolling_mean_7_qty"] = grouped_quantity.transform(
        lambda values: values.shift(1).rolling(window=7, min_periods=1).mean()
    )
    panel["rolling_std_7_qty"] = grouped_quantity.transform(
        lambda values: values.shift(1).rolling(window=7, min_periods=1).std()
    )
    panel["lag_7_rolling_mean"] = panel.groupby("product_id")[
        "rolling_mean_7_qty"
    ].shift(7)

    panel["day_of_year"] = panel["order_date"].dt.dayofyear
    panel["is_month_start"] = panel["order_date"].dt.is_month_start.astype(int)
    panel = panel.fillna(
        {
            "rolling_mean_3_qty": 0,
            "rolling_mean_7_qty": 0,
            "rolling_std_7_qty": 0,
            "lag_1_qty": 0,
            "lag_3_qty": 0,
            "lag_7_qty": 0,
            "lag_7_rolling_mean": 0,
        }
    )
    return panel.reset_index(drop=True)


def _categorize_features(frame: pd.DataFrame, categories=None) -> pd.DataFrame:
    frame = frame.copy()
    for column in CATEGORICAL_FEATURES:
        if categories and column in categories:
            frame[column] = pd.Categorical(frame[column].astype(str), categories=categories[column])
        else:
            frame[column] = frame[column].astype(str).astype("category")
    return frame


def export_model(model, filepath=CHAMPION_PATH_V2):
    joblib.dump(model, filepath)
    print(f"Model v2 berhasil diekspor dan disimpan ke: {filepath}")


def run_retraining_pipeline_v2(db_engine=engine):
    df = fetch_and_engineer_features_v2(db_engine)
    target = df["quantity"].copy()
    X = df[FEATURE_COLS].copy()

    unique_dates = np.sort(df["order_date"].unique())
    if len(unique_dates) < 10:
        raise ValueError("Data retraining v2 membutuhkan sedikitnya 10 hari.")
    split_date = unique_dates[int(len(unique_dates) * 0.8)]
    train_mask = df["order_date"] < split_date
    test_mask = ~train_mask

    X_train = _categorize_features(X.loc[train_mask])
    X_test = _categorize_features(X.loc[test_mask])
    for column in CATEGORICAL_FEATURES:
        X_test[column] = pd.Categorical(
            X_test[column].astype(str), categories=X_train[column].cat.categories
        )
    y_train, y_test = target.loc[train_mask], target.loc[test_mask]

    fixed_params = {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "random_state": 42,
        "verbose": -1,
    }
    param_grid = [
        {"max_depth": [3], "num_leaves": [5, 7], "min_child_samples": [10, 15, 20]},
        {"max_depth": [4, 5], "num_leaves": [7, 10, 15], "min_child_samples": [10, 15, 20]},
    ]
    grid_search = GridSearchCV(
        estimator=lgb.LGBMRegressor(**fixed_params),
        param_grid=param_grid,
        cv=TimeSeriesSplit(n_splits=3),
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        refit=False,
    )
    grid_search.fit(X_train, y_train)
    challenger_params = grid_search.best_params_

    val_size = max(1, int(len(X_train) * 0.15))
    X_tr, X_val = X_train.iloc[:-val_size], X_train.iloc[-val_size:]
    y_tr, y_val = y_train.iloc[:-val_size], y_train.iloc[-val_size:]
    challenger_model = lgb.LGBMRegressor(**fixed_params, **challenger_params)
    challenger_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)],
    )

    best_n_estimators = challenger_model.best_iteration_ or fixed_params["n_estimators"]
    final_model = lgb.LGBMRegressor(
        **{**fixed_params, **challenger_params, "n_estimators": best_n_estimators}
    )
    final_model.fit(X_train, y_train, eval_metric="rmse")

    predictions = np.maximum(final_model.predict(X_test), 0)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    export_model(final_model)
    print(
        f"Retraining v2 selesai | split: {pd.Timestamp(split_date).date()} | "
        f"RMSE: {rmse:.4f} | MAE: {mae:.4f} | rows: {len(df):,}"
    )
    return final_model, {"rmse": float(rmse), "mae": float(mae), "rows": int(len(df))}


if __name__ == "__main__":
    run_retraining_pipeline_v2()
