import os
import sys
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sqlalchemy import text

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import engine

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'saved_models')
CHAMPION_PATH = os.path.join(MODEL_DIR, 'champion_model.pkl')
os.makedirs(MODEL_DIR, exist_ok=True)

def fetch_and_engineer_features() -> pd.DataFrame:
    query = """
        SELECT 
            dd.date AS order_date, 
            pr.is_muslim_fashion, 
            oc.quantity
        FROM order_fact oc
        JOIN date_dimension dd ON dd.date_id = oc.date_id
        JOIN product_dimension pr ON pr.product_id = oc.product_id
        WHERE oc.status = 'SELESAI'
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn)

    df["order_date"] = pd.to_datetime(df["order_date"])
    df = df[df["quantity"] > 0]
    df = df.groupby("order_date").agg({
        "quantity": "sum",
        "is_muslim_fashion": "max"
    }).reset_index()
    
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
    df = df.fillna({
        "quantity": 0,
        "is_muslim_fashion": 0,
    })

    # ---- Fitur Rolling Kuantitas ----
    df["rolling_mean_7_qty"] = df["quantity"].rolling(window=7, min_periods=1, closed="left").mean()
    df["rolling_mean_14_qty"] = df["quantity"].rolling(window=14, min_periods=1, closed="left").mean()
    df["rolling_mean_30_qty"] = df["quantity"].rolling(window=30, min_periods=1, closed="left").mean()

    df["rolling_std_3_qty"] = df["quantity"].rolling(window=3, min_periods=1, closed="left").std()
    df["rolling_std_7_qty"] = df["quantity"].rolling(window=7, min_periods=1, closed="left").std()

    # ---- Fitur Lag Kuantitas ----
    df["lag_1_qty"] = df["quantity"].shift(1)
    df["lag_3_qty"] = df["quantity"].shift(3)
    df["lag_7_qty"] = df["quantity"].shift(7)
    df["lag_21_qty"] = df["quantity"].shift(21)
    df["lag_28_qty"] = df["quantity"].shift(28)

    df["lag_7_rolling_mean"] = df["rolling_mean_7_qty"].shift(7)
    df["lag_14_rolling_mean"] = df["rolling_mean_14_qty"].shift(14)

    df = df.dropna().reset_index(drop=True)
    
    # ---- Fitur Interaksi & Waktu (Tanpa week_of_year) ----
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
    X_train, X_test = features.iloc[:split_index].copy(), features.iloc[split_index:].copy()
    y_train, y_test = target.iloc[:split_index].copy(), target.iloc[split_index:].copy()
    
    cat_cols = ["days_name", "month"]
    for col in cat_cols:
        X_train[col] = X_train[col].astype(str)
        X_test[col] = X_test[col].astype(str)
        X_train[col] = X_train[col].astype("category")
        X_test[col] = pd.Categorical(X_test[col], categories=X_train[col].cat.categories)

    challenger_model = lgb.LGBMRegressor(
        n_estimators=500,
        num_leaves=10,
        max_depth=3,
        min_child_samples=15,
        learning_rate=0.01,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=1.5,
        reg_lambda=1.5,
        random_state=42,
    )
    
    challenger_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test), (X_train, y_train)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(stopping_rounds=15), lgb.log_evaluation(10)]
    )
    
    y_pred_challenger = challenger_model.predict(X_test)
    challenger_rmse = np.sqrt(mean_squared_error(y_test, y_pred_challenger))
    challenger_mae = mean_absolute_error(y_test, y_pred_challenger)
    print(f"Hasil Challenger -> RMSE: {challenger_rmse:.4f} | MAE: {challenger_mae:.4f}")
    
    if os.path.exists(CHAMPION_PATH):
    
        champion_model = joblib.load(CHAMPION_PATH)
        y_pred_champion = champion_model.predict(X_test)
        champion_rmse = np.sqrt(mean_squared_error(y_test, y_pred_champion))
        champion_mae = mean_absolute_error(y_test, y_pred_champion)
        print(f"Hasil Champion   -> RMSE: {champion_rmse:.4f} | MAE: {champion_mae:.4f}")
        
        if challenger_rmse <= champion_rmse:
            print("Challenger lebih baik. Memperbarui model...")
            export_model(challenger_model, CHAMPION_PATH)
        else:
            print("Fallback: Champion lama dipertahankan.")
    else:
        print("Menyimpan model baru sebagai Champion pertama...")
        export_model(challenger_model, CHAMPION_PATH)

if __name__ == '__main__':
    run_retraining_pipeline()