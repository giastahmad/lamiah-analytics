from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import threading
from sqlalchemy import func, text
import os
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, auth
from functools import wraps

from models import (
    OrderFact,
    ProductDimension,
    PlatformDimension,
    DateDimension,
    LocationDimension,
    PaymentMethodDimension,
    User,
    DailySalesSummary,
    ForecastCache,
)
from config import SessionLocal, engine
from etl import extract, transform, load

app = Flask(__name__)

app.secret_key = os.getenv("FLASK_SECRET_KEY")

cred = credentials.Certificate("firebase-key.json")
firebase_admin.initialize_app(cred)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "ml_models", "saved_models", "champion_model.pkl"
)

CORS(app)

load_dotenv()
UPLOAD_FOLDER = "./data/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ==========================================
# FUNGSI PROTEKSI ROUTE (DECORATOR)
# ==========================================
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_email" not in session:
            return redirect(url_for("login_view"))
        return f(*args, **kwargs)

    return decorated_function


# ==========================================
# ROUTE AUTHENTICATION
# ==========================================
@app.route("/login")
def login_view():
    if "user_email" in session:
        return redirect(url_for("dashboard_view"))
    return render_template("login.html")


@app.route("/api/auth/verify", methods=["POST"])
def verify_token():
    data = request.json
    id_token = data.get("token")

    if not id_token:
        return jsonify({"message": "Token tidak ditemukan"}), 400

    try:
        decoded_token = auth.verify_id_token(id_token)
        email = decoded_token.get("email")
        username = decoded_token.get("name")

        db_session = SessionLocal()
        user = db_session.query(User).filter(User.email == email).first()
        db_session.close()

        if user:
            session["user_email"] = email
            session["user_role"] = user.role
            session["user_name"] = username
            return jsonify({"status": "success", "redirect": "/dashboard"}), 200
        else:
            return (
                jsonify(
                    {
                        "status": "unauthorized",
                        "message": "Email Anda belum didaftarkan oleh Admin.",
                    }
                ),
                403,
            )

    except Exception as e:
        return (
            jsonify(
                {"status": "error", "message": "Sesi tidak valid atau kadaluarsa."}
            ),
            401,
        )


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login_view"))


# ==========================================
# INDEX
# ==========================================
@app.route("/", methods=["GET"])
def index():
    return jsonify({"message": "API Backend Aktif dan Berjalan!"}), 200


# ==========================================
# 1. UPLOAD
# ==========================================
@app.route("/upload")
@login_required
def upload_view():
    return render_template("upload.html")


@app.route("/api/upload", methods=["POST"])
def upload_data():

    if "files" not in request.files:
        return (
            jsonify(
                {"code": "VALIDATION_ERROR", "message": "Tidak ada file yang dikirim"}
            ),
            400,
        )

    files = request.files.getlist("files")
    platform = request.form.get("platform")

    if not files or files[0].filename == "":
        return (
            jsonify(
                {
                    "code": "VALIDATION_ERROR",
                    "message": "File kosong atau tidak dipilih",
                }
            ),
            400,
        )

    if not platform:
        return (
            jsonify(
                {
                    "code": "VALIDATION_ERROR",
                    "message": "Platform e-commerce belum dipilih",
                }
            ),
            400,
        )

    allowed_extensions = [".xlsx", ".csv"]

    total_loaded = 0
    total_skipped = 0
    has_warnings = False
    file_errors = []
    file_warnings = []

    # LOOPING
    for file in files:
        if file.filename == "":
            continue

        _, ext = os.path.splitext(file.filename)
        if ext.lower() not in allowed_extensions:
            file_errors.append(f"{file.filename} (Format tidak valid)")
            continue

        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        try:
            raw_df = extract.extract_data(filepath, platform)

            platform_lower = platform.lower()
            if platform_lower == "shopee":
                transformed_df = transform.transform_shopee(raw_df)
            elif platform_lower in ["tokopedia", "tiktok"]:
                transformed_df = transform.transform_tokopedia(raw_df)
            else:
                raise ValueError(f"Platform {platform} belum didukung.")

            load_result = load.load_data_warehouse(transformed_df)

            if load_result:
                code = load_result.get("code", "SUCCESS")
                if code in ["SUCCESS_WITH_WARNING", "ABORT_HIGH_UNKNOWN"]:
                    has_warnings = True
                    row_pct = load_result.get("row_pct", 0)
                    if row_pct > 0:
                        file_warnings.append(
                            f"{file.filename}: {row_pct}% SKU Tidak Dikenal"
                        )

                total_loaded += load_result.get("rows_loaded", 0)
                total_skipped += load_result.get("rows_skipped", 0) or load_result.get(
                    "duplicate_count", 0
                )

        except Exception as e:
            file_errors.append(f"{file.filename} ({str(e)})")

    if len(file_errors) == len(files) and len(files) > 0:
        return (
            jsonify(
                {
                    "code": "SERVER_ERROR",
                    "message": "Semua file gagal diproses.",
                    "detail": {"errors": file_errors},
                }
            ),
            500,
        )

    final_code = "SUCCESS"
    if has_warnings or file_errors:
        final_code = "SUCCESS_WITH_WARNING"
    elif total_loaded == 0 and total_skipped > 0:
        final_code = "NO_NEW_DATA"

    msg = f"{len(files) - len(file_errors)} dari {len(files)} file berhasil diproses."

    if final_code in ["SUCCESS", "SUCCESS_WITH_WARNING"] and total_loaded > 0:
        threading.Thread(target=update_forecast_cache_background).start()

    return (
        jsonify(
            {
                "code": final_code,
                "message": msg,
                "detail": {
                    "rows_loaded": total_loaded,
                    "rows_skipped": total_skipped,
                    "errors": file_errors if file_errors else None,
                    "warnings": file_warnings if file_warnings else None,
                },
            }
        ),
        200,
    )


# ==========================================
# 2. DASHBOARD METRICS
# ==========================================
def get_dashboard_metrics(start_date=None, end_date=None, platform=None):
    session = SessionLocal()
    try:
        base_filter = []
        if start_date:
            base_filter.append(DateDimension.date >= start_date)
        if end_date:
            base_filter.append(DateDimension.date <= end_date)
        if platform:
            base_filter.append(PlatformDimension.platform_name == platform)

        success_filter = base_filter + [DailySalesSummary.status == "SELESAI"]

        def _base(cols):
            return (
                session.query(*cols)
                .select_from(DailySalesSummary)
                .join(DateDimension, DailySalesSummary.date_id == DateDimension.date_id)
                .join(
                    PlatformDimension,
                    DailySalesSummary.platform_id == PlatformDimension.platform_id,
                )
                .filter(*success_filter)
            )

        # --------------------------------------------------
        # 1. Daftar semua platform (tidak difilter, untuk dropdown)
        # --------------------------------------------------
        all_platforms = (
            session.query(PlatformDimension.platform_name)
            .order_by(PlatformDimension.platform_name)
            .all()
        )
        platform_options = [p[0] for p in all_platforms]

        fact_success_filter = base_filter + [OrderFact.status == "SELESAI"]
        fact_batal_filter = base_filter + [OrderFact.status == "BATAL"]

        # --------------------------------------------------
        # 2. KPI: Persentase Pembatalan
        # --------------------------------------------------
        total_all_status = (
            session.query(func.count(func.distinct(OrderFact.order_key)))
            .join(DateDimension, OrderFact.date_id == DateDimension.date_id)
            .join(PlatformDimension, OrderFact.platform_id == PlatformDimension.platform_id)
            .filter(*base_filter)
            .scalar() or 0
        )
        
        total_cancelled = (
            session.query(func.count(func.distinct(OrderFact.order_key)))
            .join(DateDimension, OrderFact.date_id == DateDimension.date_id)
            .join(PlatformDimension, OrderFact.platform_id == PlatformDimension.platform_id)
            .filter(*fact_batal_filter)
            .scalar() or 0
        )

        cancellation_rate = (
            round((float(total_cancelled) / float(total_all_status) * 100), 1)
            if total_all_status > 0
            else 0
        )

        # --------------------------------------------------
        # 3. KPI: Jumlah Model (Tetap pakai Summary)
        # --------------------------------------------------
        num_models = (
            _base([func.count(func.distinct(ProductDimension.product_model))])
            .join(ProductDimension, DailySalesSummary.product_id == ProductDimension.product_id)
            .scalar() or 0
        )

        # --------------------------------------------------
        # 4. KPI: Total Order (Kembali ke OrderFact)
        # --------------------------------------------------
        num_orders = (
            session.query(func.count(func.distinct(OrderFact.order_key)))
            .join(DateDimension, OrderFact.date_id == DateDimension.date_id)
            .join(PlatformDimension, OrderFact.platform_id == PlatformDimension.platform_id)
            .filter(*fact_success_filter)
            .scalar() or 0
        )

        # --------------------------------------------------
        # 5. KPI: Total Pendapatan (Tetap pakai Summary - Amount Additive)
        # --------------------------------------------------
        revenue_total = _base([func.sum(DailySalesSummary.total_amount)]).scalar() or 0

        # --------------------------------------------------
        # 6. KPI: Jumlah Kota (Tetap pakai Summary)
        # --------------------------------------------------
        num_cities = (
            _base([func.count(func.distinct(LocationDimension.city))])
            .join(LocationDimension, DailySalesSummary.location_id == LocationDimension.location_id)
            .scalar() or 0
        )

        # --------------------------------------------------
        # 7. KPI: AOV
        # --------------------------------------------------
        aov = (float(revenue_total) / float(num_orders)) if num_orders > 0 else 0

        # --------------------------------------------------
        # 8. KPI: Ramadhan vs Normal
        # --------------------------------------------------
        ramadhan_filter = base_filter + [DateDimension.is_ramadhan == 1]
        normal_filter = base_filter + [DateDimension.is_ramadhan == 0]

        fact_ramadhan_filter = fact_success_filter + [DateDimension.is_ramadhan == 1]
        fact_normal_filter = fact_success_filter + [DateDimension.is_ramadhan == 0]

        def _ramadhan(cols):
            return session.query(*cols).select_from(DailySalesSummary).join(DateDimension, DailySalesSummary.date_id == DateDimension.date_id).join(PlatformDimension, DailySalesSummary.platform_id == PlatformDimension.platform_id).filter(*ramadhan_filter)

        def _normal(cols):
            return session.query(*cols).select_from(DailySalesSummary).join(DateDimension, DailySalesSummary.date_id == DateDimension.date_id).join(PlatformDimension, DailySalesSummary.platform_id == PlatformDimension.platform_id).filter(*normal_filter)

        ramadhan_days = _ramadhan([func.count(func.distinct(DateDimension.date))]).scalar() or 0
        normal_days = _normal([func.count(func.distinct(DateDimension.date))]).scalar() or 0
        total_days = _base([func.count(func.distinct(DateDimension.date))]).scalar() or 0

        ramadhan_revenue = _ramadhan([func.sum(DailySalesSummary.total_amount)]).scalar() or 0
        normal_revenue = _normal([func.sum(DailySalesSummary.total_amount)]).scalar() or 0

        ramadhan_orders = (
            session.query(func.count(func.distinct(OrderFact.order_key)))
            .join(DateDimension, OrderFact.date_id == DateDimension.date_id)
            .join(PlatformDimension, OrderFact.platform_id == PlatformDimension.platform_id)
            .filter(*fact_ramadhan_filter)
            .scalar() or 0
        )
        
        normal_orders = (
            session.query(func.count(func.distinct(OrderFact.order_key)))
            .join(DateDimension, OrderFact.date_id == DateDimension.date_id)
            .join(PlatformDimension, OrderFact.platform_id == PlatformDimension.platform_id)
            .filter(*fact_normal_filter)
            .scalar() or 0
        )

        overall_avg_revenue = ((float(revenue_total) / float(total_days)) if total_days > 0 else 0)
        overall_avg_orders = (float(num_orders) / float(total_days)) if total_days > 0 else 0

        ramadhan_avg_revenue = ((float(ramadhan_revenue) / float(ramadhan_days)) if ramadhan_days > 0 else 0)
        normal_avg_revenue = ((float(normal_revenue) / float(normal_days)) if normal_days > 0 else 0)
        ramadhan_avg_orders = ((float(ramadhan_orders) / float(ramadhan_days)) if ramadhan_days > 0 else 0)
        normal_avg_orders = ((float(normal_orders) / float(normal_days)) if normal_days > 0 else 0)

        has_comparison = ramadhan_days > 0 and normal_days > 0
        if has_comparison and normal_avg_revenue > 0:
            ramadhan_lift = round(
                (float(ramadhan_avg_revenue) - float(normal_avg_revenue)) / float(normal_avg_revenue) * 100, 1
            )
        else:
            ramadhan_lift = None

        # --------------------------------------------------
        # 9. Chart: Bar — Quantity per Warna
        # --------------------------------------------------
        color_data = (
            _base(
                [
                    ProductDimension.product_color,
                    func.sum(DailySalesSummary.total_quantity),
                ]
            )
            .join(
                ProductDimension,
                DailySalesSummary.product_id == ProductDimension.product_id,
            )
            .group_by(ProductDimension.product_color)
            .order_by(func.sum(DailySalesSummary.total_quantity).desc())
            .all()
        )

        # --------------------------------------------------
        # 10. Chart: Line — Penjualan per Platform per Tanggal
        # --------------------------------------------------
        line_data = (
            _base(
                [
                    DateDimension.date,
                    PlatformDimension.platform_name,
                    func.sum(DailySalesSummary.total_amount),
                ]
            )
            .group_by(DateDimension.date, PlatformDimension.platform_name)
            .order_by(DateDimension.date)
            .all()
        )

        # --------------------------------------------------
        # 11. Top 5 Best Selling Models
        # --------------------------------------------------
        top_products = (
            _base(
                [
                    ProductDimension.product_model,
                    func.sum(DailySalesSummary.total_quantity),
                ]
            )
            .join(
                ProductDimension,
                DailySalesSummary.product_id == ProductDimension.product_id,
            )
            .group_by(ProductDimension.product_model)
            .order_by(func.sum(DailySalesSummary.total_quantity).desc())
            .limit(5)
            .all()
        )

        # --------------------------------------------------
        # 12. Map: Persebaran per Provinsi
        # --------------------------------------------------
        map_loc_filter = []
        if start_date:
            map_loc_filter.append(DateDimension.date >= start_date)
        if end_date:
            map_loc_filter.append(DateDimension.date <= end_date)
        if platform:
            map_loc_filter.append(PlatformDimension.platform_name == platform)
        map_loc_filter.append(DailySalesSummary.status == "SELESAI")

        map_query = (
            session.query(
                LocationDimension.province, func.sum(DailySalesSummary.total_quantity)
            )
            .select_from(DailySalesSummary)
            .join(DateDimension, DailySalesSummary.date_id == DateDimension.date_id)
            .join(
                PlatformDimension,
                DailySalesSummary.platform_id == PlatformDimension.platform_id,
            )
            .join(
                LocationDimension,
                DailySalesSummary.location_id == LocationDimension.location_id,
            )
            .filter(*map_loc_filter)
            .group_by(LocationDimension.province)
            .all()
        )

        # --------------------------------------------------
        # 13. Chart: Avg Basket Size per Payment Method
        # --------------------------------------------------
        avg_basket_formula = func.sum(DailySalesSummary.total_amount) / func.sum(
            DailySalesSummary.total_orders
        )

        payment_data = (
            _base(
                [
                    PaymentMethodDimension.payment_method_name,
                    avg_basket_formula,
                ]
            )
            .join(
                PaymentMethodDimension,
                DailySalesSummary.payment_method_id
                == PaymentMethodDimension.payment_method_id,
            )
            .group_by(PaymentMethodDimension.payment_method_name)
            .order_by(avg_basket_formula.desc())
            .all()
        )

        # --------------------------------------------------
        # 14. Chart: Product Size Distribution
        # --------------------------------------------------
        size_data = (
            _base(
                [
                    ProductDimension.product_size,
                    func.sum(DailySalesSummary.total_quantity),
                ]
            )
            .join(
                ProductDimension,
                DailySalesSummary.product_id == ProductDimension.product_id,
            )
            .group_by(ProductDimension.product_size)
            .order_by(func.sum(DailySalesSummary.total_quantity).desc())
            .all()
        )

        map_data = [["Provinsi", "Total Terjual"]]
        for row in map_query:
            map_data.append([row[0], int(row[1])])

        # --------------------------------------------------
        # Return
        # --------------------------------------------------
        return {
            "platform_options": platform_options,
            "cancellation_rate": cancellation_rate,
            "total_all_status": total_all_status,
            "num_models": num_models,
            "num_orders": num_orders,
            "revenue_month": float(revenue_total),
            "num_cities": num_cities,
            "aov": round(float(aov), 0),
            "overall_avg_revenue": round(overall_avg_revenue, 0),
            "overall_avg_orders": round(overall_avg_orders, 2),
            "ramadhan_lift": ramadhan_lift,
            "has_comparison": has_comparison,
            "ramadhan_avg_revenue": round(float(ramadhan_avg_revenue), 0),
            "normal_avg_revenue": round(float(normal_avg_revenue), 0),
            "ramadhan_avg_orders": round(float(ramadhan_avg_orders), 2),
            "normal_avg_orders": round(float(normal_avg_orders), 2),
            "map_data": map_data,
            "color_labels": [c[0] for c in color_data],
            "color_values": [int(c[1]) for c in color_data],
            "line_data": [
                {"date": str(d[0]), "platform": d[1], "amount": float(d[2])}
                for d in line_data
            ],
            "top_products": [[p[0], int(p[1])] for p in top_products],
            "payment_labels": [p[0] for p in payment_data],
            "payment_values": [
                float(p[1]) if p[1] is not None else 0 for p in payment_data
            ],
            "size_labels": [s[0] for s in size_data],
            "size_values": [int(s[1]) for s in size_data],
        }

    finally:
        session.close()


@app.route("/dashboard")
@login_required
def dashboard_view():

    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")
    platform = request.args.get("platform")

    metrics = get_dashboard_metrics(start_date, end_date, platform)
    return render_template("dashboard.html", m=metrics)


# ==========================================
# 3. PREDIKSI
# ==========================================
FORECAST_CACHE = {"db_state": None, "data": None}


def get_db_state():
    try:
        query = "SELECT COUNT(order_key), MAX(date_id) FROM order_fact"
        with engine.connect() as conn:
            result = conn.execute(text(query)).fetchone()
            return f"{result[0]}_{result[1]}"
    except Exception:
        return None


def generate_recursive_forecast(model, days_ahead=14):
   
    query_hist = """
        SELECT 
            dd.date AS order_date, 
            MAX(pr.is_muslim_fashion) as is_muslim_fashion, 
            SUM(oc.quantity) as quantity
        FROM order_fact oc
        JOIN date_dimension dd ON dd.date_id = oc.date_id
        JOIN product_dimension pr ON pr.product_id = oc.product_id
        WHERE oc.status = 'SELESAI'
        GROUP BY dd.date
        ORDER BY dd.date DESC
        LIMIT 60
    """
    with engine.connect() as conn:
        hist_df = pd.read_sql(text(query_hist), conn)

    hist_df["order_date"] = pd.to_datetime(hist_df["order_date"])
    hist_df = hist_df.sort_values("order_date").reset_index(drop=True)

    last_date = hist_df["order_date"].max()
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1), periods=days_ahead, freq="D"
    )

    start_date_str = hist_df["order_date"].min().strftime("%Y-%m-%d")
    end_date_str = future_dates.max().strftime("%Y-%m-%d")

    query_dates = f"""
        SELECT 
            date AS order_date, days_name, month, is_weekend, 
            is_twin_date, is_payday, is_ramadhan
        FROM date_dimension
        WHERE date BETWEEN '{start_date_str}' AND '{end_date_str}'
    """
    with engine.connect() as conn:
        date_dim = pd.read_sql(text(query_dates), conn)
    date_dim["order_date"] = pd.to_datetime(date_dim["order_date"])

    future_df = pd.DataFrame({"order_date": future_dates})
    future_df["is_muslim_fashion"] = hist_df["is_muslim_fashion"].tail(1).values[0]
    future_df["quantity"] = np.nan

    combined_df = pd.concat([hist_df, future_df], ignore_index=True)
    combined_df = combined_df.merge(date_dim, on="order_date", how="left")

    cat_cols = ["days_name", "month"]
    for col in cat_cols:
        combined_df[col] = combined_df[col].astype("category")
        
    feature_cols = [
        "rolling_mean_7_qty", "rolling_mean_14_qty", "rolling_mean_30_qty",
        "rolling_std_3_qty", "rolling_std_7_qty",
        "lag_1_qty", "lag_3_qty", "lag_7_qty", "lag_21_qty", "lag_28_qty",
        "lag_7_rolling_mean", "lag_14_rolling_mean",
        "payday_weekend", "ramadhan_twin", 
        "day_of_year", "week_of_year", "is_month_end", "is_month_start"
    ]
    for col in feature_cols:
        combined_df[col] = 0.0

    forecast_results = []
    start_idx = len(hist_df)
    
    qty_history = hist_df["quantity"].tolist()

    for i in range(start_idx, len(combined_df)):
        current_date = combined_df.at[i, "order_date"]
        
        lag_1 = qty_history[-1] if len(qty_history) >= 1 else 0
        lag_3 = qty_history[-3] if len(qty_history) >= 3 else 0
        lag_7 = qty_history[-7] if len(qty_history) >= 7 else 0
        lag_21 = qty_history[-21] if len(qty_history) >= 21 else 0
        lag_28 = qty_history[-28] if len(qty_history) >= 28 else 0

        rolling_7 = np.mean(qty_history[-7:]) if len(qty_history) >= 7 else np.mean(qty_history)
        rolling_14 = np.mean(qty_history[-14:]) if len(qty_history) >= 14 else np.mean(qty_history)
        rolling_30 = np.mean(qty_history[-30:]) if len(qty_history) >= 30 else np.mean(qty_history)
 
        std_3 = np.std(qty_history[-3:], ddof=1) if len(qty_history) >= 2 else 0.0
        std_7 = np.std(qty_history[-7:], ddof=1) if len(qty_history) >= 2 else 0.0

        lag_7_rolling_mean = np.mean(qty_history[-14:-7]) if len(qty_history) >= 14 else 0
        lag_14_rolling_mean = np.mean(qty_history[-28:-14]) if len(qty_history) >= 28 else 0

        combined_df.at[i, "rolling_mean_7_qty"] = rolling_7
        combined_df.at[i, "rolling_mean_14_qty"] = rolling_14
        combined_df.at[i, "rolling_mean_30_qty"] = rolling_30
        combined_df.at[i, "rolling_std_3_qty"] = std_3
        combined_df.at[i, "rolling_std_7_qty"] = std_7
        combined_df.at[i, "lag_1_qty"] = lag_1
        combined_df.at[i, "lag_3_qty"] = lag_3
        combined_df.at[i, "lag_7_qty"] = lag_7
        combined_df.at[i, "lag_21_qty"] = lag_21
        combined_df.at[i, "lag_28_qty"] = lag_28
        combined_df.at[i, "lag_7_rolling_mean"] = lag_7_rolling_mean
        combined_df.at[i, "lag_14_rolling_mean"] = lag_14_rolling_mean
        
        combined_df.at[i, "payday_weekend"] = combined_df.at[i, "is_payday"] * combined_df.at[i, "is_weekend"]
        combined_df.at[i, "ramadhan_twin"] = combined_df.at[i, "is_ramadhan"] * combined_df.at[i, "is_twin_date"]
        
        combined_df.at[i, "day_of_year"] = current_date.dayofyear
        combined_df.at[i, "week_of_year"] = current_date.isocalendar().week
        combined_df.at[i, "is_month_end"] = int(current_date.is_month_end)
        combined_df.at[i, "is_month_start"] = int(current_date.is_month_start)

        current_features = combined_df.iloc[[i]].drop(columns=["order_date", "quantity"])

        pred_qty = model.predict(current_features)[0]
        pred_qty = max(0, np.round(pred_qty))

        combined_df.at[i, "quantity"] = pred_qty
        qty_history.append(pred_qty)

        forecast_results.append(
            {
                "date": current_date.strftime("%Y-%m-%d"),
                "predicted_qty": int(pred_qty),
            }
        )

    return forecast_results


def update_forecast_cache_background():
    global FORECAST_CACHE
    try:

        if not os.path.exists(MODEL_PATH):
            return

        model = joblib.load(MODEL_PATH)

        forecast_results = generate_recursive_forecast(model, days_ahead=14)
        total_forecast_qty = sum(item["predicted_qty"] for item in forecast_results)

        # Kalkulasi Top Models & Colors
        query_weights = """
            SELECT 
                pr.product_model, 
                pr.product_color, 
                SUM(ds.total_quantity) as qty
            FROM daily_sales_summary ds
            JOIN date_dimension dd ON dd.date_id = ds.date_id
            JOIN product_dimension pr ON pr.product_id = ds.product_id
            WHERE ds.status = 'SELESAI' 
            AND dd.date >= (SELECT DATE_SUB(MAX(dd2.date), INTERVAL 90 DAY) 
                            FROM daily_sales_summary ds2 
                            JOIN date_dimension dd2 ON ds2.date_id = dd2.date_id)
            GROUP BY pr.product_model, pr.product_color
        """
        with engine.connect() as conn:
            weight_df = pd.read_sql(text(query_weights), conn)

        model_df = weight_df.groupby("product_model")["qty"].sum().reset_index()
        top_models = model_df.nlargest(5, "qty")
        model_weight_total = top_models["qty"].sum()

        top_models_forecast = [
            {
                "name": row["product_model"],
                "forecast_qty": int(
                    round(
                        total_forecast_qty
                        * (
                            row["qty"] / model_weight_total
                            if model_weight_total > 0
                            else 0
                        )
                    )
                ),
            }
            for _, row in top_models.iterrows()
        ]

        color_df = weight_df.groupby("product_color")["qty"].sum().reset_index()
        top_colors = color_df.nlargest(5, "qty")
        color_weight_total = top_colors["qty"].sum()

        top_colors_forecast = [
            {
                "name": row["product_color"],
                "forecast_qty": int(
                    round(
                        total_forecast_qty
                        * (
                            row["qty"] / color_weight_total
                            if color_weight_total > 0
                            else 0
                        )
                    )
                ),
            }
            for _, row in top_colors.iterrows()
        ]

        db_state_now = get_db_state()

        session = SessionLocal()
        new_cache = ForecastCache(
            db_state=db_state_now,
            total_forecast_qty=total_forecast_qty,
            forecast_daily=forecast_results,
            top_models=top_models_forecast,
            top_colors=top_colors_forecast,
        )
        session.add(new_cache)
        session.commit()
        session.close()

    except Exception as e:
        print(f"[ERROR] Forecast background gagal: {e}")


@app.route("/forecast")
@login_required
def forecast_view():
    if not os.path.exists(MODEL_PATH):
        return render_template(
            "forecast.html",
            error="Model prediksi belum siap. Silakan jalankan pipeline retraining.",
        )

    try:
        session = SessionLocal()
        cache = (
            session.query(ForecastCache)
            .order_by(ForecastCache.created_at.desc())
            .first()
        )
        db_state = get_db_state()

        if not cache or cache.db_state != db_state:
            update_forecast_cache_background()
            cache = (
                session.query(ForecastCache)
                .order_by(ForecastCache.created_at.desc())
                .first()
            )

        session.close()

        if not cache:
            return render_template("forecast.html", error="Gagal memuat data prediksi.")

        return render_template(
            "forecast.html",
            total_forecast=cache.total_forecast_qty,
            forecast_daily=cache.forecast_daily,
            top_models=cache.top_models,
            top_colors=cache.top_colors,
        )

    except Exception as e:
        return render_template(
            "forecast.html", error=f"Terjadi kesalahan teknis: {str(e)}"
        )


if __name__ == "__main__":
    app.run(debug=True, port=5000)