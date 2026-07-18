import os
import sys
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sqlalchemy import text

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import engine

MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
CHAMPION_PATH = os.path.join(MODEL_DIR, "champion_model.pkl")
os.makedirs(MODEL_DIR, exist_ok=True)


def fetch_and_engineer_features() -> pd.DataFrame:
    query = """
        SELECT 
            dd.date AS order_date, 
            pr.is_muslim_fashion, 
            fa.total_quantity AS quantity
        FROM fact_daily_agregat fa
        JOIN date_dimension dd ON dd.date_id = fa.date_id
        JOIN product_dimension pr ON pr.product_id = fa.product_id
        WHERE fa.status = 'SELESAI'
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn)

    df["order_date"] = pd.to_datetime(df["order_date"])
    df = df[df["quantity"] > 0]
    df = (
        df.groupby("order_date")
        .agg({"quantity": "sum", "is_muslim_fashion": "mean"})
        .reset_index()
    )

    query_dates = """
        SELECT 
            date AS order_date, days_name, month, is_weekend, 
            is_twin_date, is_payday, is_ramadhan
        FROM date_dimension
        WHERE date BETWEEN '2024-01-01' AND '2026-12-31'
    """
    with engine.connect() as conn:
        df_date_prop = pd.read_sql(text(query_dates), conn)

    df_date_prop["order_date"] = pd.to_datetime(df_date_prop["order_date"])

    all_dates = pd.date_range(df.order_date.min(), df.order_date.max(), freq="D")
    df = df.set_index("order_date")
    df = df.reindex(all_dates)
    df = df.reset_index().rename(columns={"index": "order_date"})

    df = df.merge(df_date_prop, on="order_date", how="left")
    df = df.fillna(
        {
            "quantity": 0,
            "is_muslim_fashion": 0,
        }
    )

    # ---- Fitur Rolling Kuantitas ----
    df["rolling_mean_3_qty"] = (
        df["quantity"].rolling(window=3, min_periods=1, closed="left").mean()
    )
    df["rolling_mean_7_qty"] = (
        df["quantity"].rolling(window=7, min_periods=1, closed="left").mean()
    )
    df["rolling_mean_14_qty"] = (
        df["quantity"].rolling(window=14, min_periods=1, closed="left").mean()
    )
    df["rolling_mean_30_qty"] = (
        df["quantity"].rolling(window=30, min_periods=1, closed="left").mean()
    )

    df["rolling_std_3_qty"] = (
        df["quantity"].rolling(window=3, min_periods=1, closed="left").std()
    )
    df["rolling_std_7_qty"] = (
        df["quantity"].rolling(window=7, min_periods=1, closed="left").std()
    )

    # ---- Fitur Lag Kuantitas ----
    df["lag_1_qty"] = df["quantity"].shift(1)
    df["lag_3_qty"] = df["quantity"].shift(3)
    df["lag_7_qty"] = df["quantity"].shift(7)
    df["lag_21_qty"] = df["quantity"].shift(21)
    df["lag_28_qty"] = df["quantity"].shift(28)

    df["lag_7_rolling_mean"] = df["rolling_mean_7_qty"].shift(7)
    df["lag_14_rolling_mean"] = df["rolling_mean_14_qty"].shift(14)

    df = df.dropna().reset_index(drop=True)

    # ---- Fitur Interaksi & Waktu  ----
    df["payday_weekend"] = df["is_payday"] * df["is_weekend"]
    df["ramadhan_twin"] = df["is_ramadhan"] * df["is_twin_date"]

    df["day_of_year"] = df["order_date"].dt.dayofyear
    df["week_of_year"] = df["order_date"].dt.isocalendar().week.astype(int)
    df["is_month_end"] = df["order_date"].dt.is_month_end.astype(int)
    df["is_month_start"] = df["order_date"].dt.is_month_start.astype(int)

    return df


def export_model(model, filepath):
    joblib.dump(model, filepath)
    print(f"Model berhasil diekspor dan disimpan ke: {filepath}")


def run_retraining_pipeline():

    df = fetch_and_engineer_features()

    target = df["quantity"]
    features = df.drop(columns=["order_date", "quantity"])

    split_index = int(len(df) * 0.8)
    X_train, X_test = (
        features.iloc[:split_index].copy(),
        features.iloc[split_index:].copy(),
    )
    y_train, y_test = target.iloc[:split_index].copy(), target.iloc[split_index:].copy()

    cat_cols = ["days_name", "month"]
    for col in cat_cols:
        X_train[col] = X_train[col].astype(str)
        X_test[col] = X_test[col].astype(str)
        X_train[col] = X_train[col].astype("category")
        X_test[col] = pd.Categorical(
            X_test[col], categories=X_train[col].cat.categories
        )

    fixed_params = {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "random_state": 42,
        "verbose": -1,
    }

    model_base = lgb.LGBMRegressor(**fixed_params)

    param_grid = [
        {"max_depth": [3], "num_leaves": [5, 7], "min_child_samples": [10, 15, 20]},
        {"max_depth": [4, 5], "num_leaves": [7, 10, 15], "min_child_samples": [10, 15, 20],
        },
    ]

    tscv = TimeSeriesSplit(n_splits=3)

    grid_search = GridSearchCV(
        estimator=model_base,
        param_grid=param_grid,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        refit=False,
    )

    grid_search.fit(X_train, y_train)
    challenger_params = grid_search.best_params_

    val_size = int(len(X_train) * 0.15)
    X_tr, X_val = X_train.iloc[:-val_size], X_train.iloc[-val_size:]
    y_tr, y_val = y_train.iloc[:-val_size], y_train.iloc[-val_size:]

    challenger_model = lgb.LGBMRegressor(**fixed_params, **challenger_params)
    challenger_model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val), (X_train, y_train)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)],
    )

    best_n_estimators = challenger_model.best_iteration_ or fixed_params["n_estimators"]
    challenger_model = lgb.LGBMRegressor(
        **{**fixed_params, **challenger_params, "n_estimators": best_n_estimators}
    )
    challenger_model.fit(X_train, y_train, eval_metric="rmse")

    y_pred_challenger = np.maximum(challenger_model.predict(X_test), 0)
    challenger_rmse = np.sqrt(mean_squared_error(y_test, y_pred_challenger))
    challenger_mae = mean_absolute_error(y_test, y_pred_challenger)
    print(
        f"Hasil Challenger -> RMSE: {challenger_rmse:.4f} | MAE: {challenger_mae:.4f}"
    )

    if os.path.exists(CHAMPION_PATH):

        champion_model = joblib.load(CHAMPION_PATH)
        y_pred_champion = np.maximum(champion_model.predict(X_test), 0)
        champion_rmse = np.sqrt(mean_squared_error(y_test, y_pred_champion))
        champion_mae = mean_absolute_error(y_test, y_pred_champion)
        print(
            f"Hasil Champion   -> RMSE: {champion_rmse:.4f} | MAE: {champion_mae:.4f}"
        )

        if challenger_rmse < champion_rmse:
            print("Challenger lebih baik. Memperbarui model...")
            export_model(challenger_model, CHAMPION_PATH)
        else:
            print("Fallback: Champion lama dipertahankan.")
    else:
        print("Menyimpan model baru sebagai Champion pertama...")
        export_model(challenger_model, CHAMPION_PATH)


if __name__ == "__main__":
    run_retraining_pipeline()
