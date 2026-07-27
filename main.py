import json
import os
import threading
from functools import wraps

import firebase_admin
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from firebase_admin import auth, credentials
from flask import Flask, jsonify, redirect, render_template, request, session, url_for
from flask_cors import CORS
from sqlalchemy import func, text

from config import SessionLocal, engine
from etl import extract, load, transform
from models import (
    DateDimension,
    ForecastCache,
    LocationDimension,
    OrderFact,
    PaymentMethodDimension,
    PlatformDimension,
    ProductDimension,
    User,
)

app = Flask(__name__)

app.secret_key = os.getenv("FLASK_SECRET_KEY")

firebase_secret = os.environ.get("FIREBASE_KEY_JSON")

if firebase_secret:
    cred_dict = json.loads(firebase_secret)
    cred = credentials.Certificate(cred_dict)
else:
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

    except Exception as e:  # noqa: BLE001
        print(f"[AUTH ERROR] Gagal verifikasi token atau query DB: {e}")
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
@login_required
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

        except Exception as e:  # noqa: BLE001
            file_errors.append(f"{file.filename} ({str(e)})")  # noqa: RUF010

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
def get_dashboard_metrics(start_date=None, end_date=None, platforms=None, colors=None, sizes=None, provinces=None, cities=None,
                          models=None, is_twin_date=None, month=None, year=None, payment_categories=None, min_price=None, max_price=None, statuses=None):
    session = SessionLocal()
    try:
        # --------------------------------------------------
        # 1. FILTER UTAMA 
        # --------------------------------------------------
        base_filter = []
        if start_date: base_filter.append(DateDimension.date >= start_date)
        if end_date: base_filter.append(DateDimension.date <= end_date)
        
        # Logika Slicer Multiple Select
        if platforms: base_filter.append(PlatformDimension.platform_name.in_(platforms))
        if colors: base_filter.append(ProductDimension.product_color.in_(colors))
        if sizes: base_filter.append(ProductDimension.product_size.in_(sizes))
        if provinces: base_filter.append(LocationDimension.province.in_(provinces))
        if cities: base_filter.append(LocationDimension.city.in_(cities))
        if models: base_filter.append(ProductDimension.product_model.in_(models))
        if payment_categories: base_filter.append(PaymentMethodDimension.payment_method_category.in_(payment_categories))
        
        # Logika Filter Rentang Harga (Berdasarkan harga satuan dari OrderFact)
        if min_price and min_price.strip(): base_filter.append(OrderFact.price >= float(min_price))
        if max_price and max_price.strip(): base_filter.append(OrderFact.price <= float(max_price))

        # Logika Slicer Tambahan (Single Select)
        if is_twin_date: base_filter.append(DateDimension.is_twin_date == int(is_twin_date))
        if month: base_filter.append(DateDimension.month == month)
        if year: base_filter.append(DateDimension.year == int(year))

        base_filter_no_status = list(base_filter)
        
        if statuses: 
            base_filter.append(OrderFact.status.in_(statuses))
            metric_filter = base_filter
        else:
            metric_filter = base_filter + [OrderFact.status == "SELESAI"]
            
        batal_filter = base_filter_no_status + [OrderFact.status == "BATAL"]

        def _base(cols, custom_filter=None):
            f = custom_filter if custom_filter is not None else metric_filter
            return (
                session.query(*cols)
                .select_from(OrderFact)
                .join(DateDimension, OrderFact.date_id == DateDimension.date_id)
                .join(PlatformDimension, OrderFact.platform_id == PlatformDimension.platform_id)
                .join(LocationDimension, OrderFact.location_id == LocationDimension.location_id)
                .join(ProductDimension, OrderFact.product_id == ProductDimension.product_id)
                .join(PaymentMethodDimension, OrderFact.payment_method_id == PaymentMethodDimension.payment_method_id)
                .filter(*f)
            )

        # --------------------------------------------------
        # 2. Opsi Dropdown Slicer
        # --------------------------------------------------
        color_options = [c[0] for c in session.query(ProductDimension.product_color).distinct().all() if c[0]]
        status_options = [s[0] for s in session.query(OrderFact.status).distinct().all() if s[0]]
        size_options = [s[0] for s in session.query(ProductDimension.product_size).distinct().all() if s[0]]
        
        loc_data = session.query(LocationDimension.province, LocationDimension.city).distinct().all()
        province_options = []
        all_cities = set()
        province_city_map = {}
        
        for p, c in loc_data:
            if not p or not c: continue
            if p not in province_options:
                province_options.append(p)
            if p not in province_city_map:
                province_city_map[p] = []
            province_city_map[p].append(c)
            all_cities.add(c)
            
        province_options = sorted(province_options)
        
        if provinces:
            city_options = []
            for p in provinces:
                city_options.extend(province_city_map.get(p, []))
            city_options = sorted(set(city_options))
        else:
            city_options = sorted(all_cities)
        
        platform_options = [p[0] for p in session.query(PlatformDimension.platform_name).distinct().all() if p[0]]
        model_options = [m[0] for m in session.query(ProductDimension.product_model).distinct().all() if m[0]]
        
        month_options = [m[0] for m in session.query(DateDimension.month).distinct().all() if m[0]]
        year_options = [y[0] for y in session.query(DateDimension.year).distinct().all() if y[0]]
        payment_cat_options = [p[0] for p in session.query(PaymentMethodDimension.payment_method_category).distinct().all() if p[0]]

        # --------------------------------------------------
        # 3. KPI Utama
        # --------------------------------------------------
        total_all_status = _base([func.count(func.distinct(OrderFact.order_key))], custom_filter=base_filter_no_status).scalar() or 0
        total_cancelled = _base([func.count(func.distinct(OrderFact.order_key))], custom_filter=batal_filter).scalar() or 0
        cancellation_rate = round((float(total_cancelled) / float(total_all_status) * 100), 1) if total_all_status > 0 else 0
        
        agg_result = _base([
            func.count(func.distinct(ProductDimension.product_model)),
            func.count(func.distinct(OrderFact.order_key)),
            func.sum(OrderFact.total_amount),
            func.count(func.distinct(LocationDimension.city))
        ]).first()

        num_models = agg_result[0] or 0
        num_orders = agg_result[1] or 0
        revenue_total = agg_result[2] or 0
        num_cities = agg_result[3] or 0
        aov = (float(revenue_total) / float(num_orders)) if num_orders > 0 else 0

        # --------------------------------------------------
        # 4. KPI: Ramadhan vs Normal
        # --------------------------------------------------
        valid_dates = _base([DateDimension.date, DateDimension.is_ramadhan]).distinct().all()
        total_days = len(valid_dates)
        ramadhan_days = sum(1 for d in valid_dates if d[1] == 1)
        normal_days = sum(1 for d in valid_dates if d[1] == 0)

        ramadhan_revenue = _base([func.sum(OrderFact.total_amount)]).filter(DateDimension.is_ramadhan == 1).scalar() or 0
        normal_revenue = _base([func.sum(OrderFact.total_amount)]).filter(DateDimension.is_ramadhan == 0).scalar() or 0

        ramadhan_orders = _base([func.count(func.distinct(OrderFact.order_key))]).filter(DateDimension.is_ramadhan == 1).scalar() or 0
        normal_orders = _base([func.count(func.distinct(OrderFact.order_key))]).filter(DateDimension.is_ramadhan == 0).scalar() or 0

        overall_avg_revenue = (float(revenue_total) / float(total_days)) if total_days > 0 else 0
        overall_avg_orders = (float(num_orders) / float(total_days)) if total_days > 0 else 0

        ramadhan_avg_revenue = (float(ramadhan_revenue) / float(ramadhan_days)) if ramadhan_days > 0 else 0
        normal_avg_revenue = (float(normal_revenue) / float(normal_days)) if normal_days > 0 else 0

        ramadhan_avg_orders = (float(ramadhan_orders) / float(ramadhan_days)) if ramadhan_days > 0 else 0
        normal_avg_orders = (float(normal_orders) / float(normal_days)) if normal_days > 0 else 0

        has_comparison = ramadhan_days > 0 and normal_days > 0
        if has_comparison and normal_avg_revenue > 0:
            ramadhan_lift = round((float(ramadhan_avg_revenue) - float(normal_avg_revenue)) / float(normal_avg_revenue) * 100, 1)
        else:
            ramadhan_lift = None

        # --------------------------------------------------
        # 5. Data Chart (Visualisasi)
        # --------------------------------------------------
        color_data = _base([ProductDimension.product_color, func.sum(OrderFact.quantity)]).group_by(ProductDimension.product_color).order_by(func.sum(OrderFact.quantity).desc()).all()
        line_data = _base([DateDimension.date, PlatformDimension.platform_name, func.sum(OrderFact.total_amount)]).group_by(DateDimension.date, PlatformDimension.platform_name).order_by(DateDimension.date).all()
        top_products = _base([ProductDimension.product_model, func.sum(OrderFact.quantity)]).group_by(ProductDimension.product_model).order_by(func.sum(OrderFact.quantity).desc()).limit(5).all()
        
        map_query = _base([LocationDimension.province, func.sum(OrderFact.quantity)]).group_by(LocationDimension.province).all()
        map_data = [["Provinsi", "Total Terjual"]] + [[row[0], int(row[1])] for row in map_query]
        
        avg_basket_formula = func.sum(OrderFact.total_amount) / func.count(func.distinct(OrderFact.order_key))
        payment_data = _base([PaymentMethodDimension.payment_method_name, avg_basket_formula]).group_by(PaymentMethodDimension.payment_method_name).order_by(avg_basket_formula.desc()).all()
        size_data = _base([ProductDimension.product_size, func.sum(OrderFact.quantity)]).group_by(ProductDimension.product_size).order_by(func.sum(OrderFact.quantity).desc()).all()
        
        # --------------------------------------------------
        # 6. Analisis Pareto
        # --------------------------------------------------
        pareto_query = _base([ProductDimension.product_model, func.sum(OrderFact.total_amount)]).group_by(ProductDimension.product_model).order_by(func.sum(OrderFact.total_amount).desc()).all()

        total_pareto_revenue = sum(float(row[1]) for row in pareto_query)
        pareto_data = []
        cum_sum = 0
        for row in pareto_query:
            revenue = float(row[1])
            cum_sum += revenue
            cum_pct = (cum_sum / total_pareto_revenue * 100) if total_pareto_revenue > 0 else 0
            pareto_data.append({"model": row[0], "revenue": revenue, "cumulative_percentage": round(cum_pct, 1)})
            
        # --------------------------------------------------
        # 7. Heatmap Kepadatan Order
        # --------------------------------------------------
        heatmap_query = _base([DateDimension.days_name, DateDimension.month, func.count(func.distinct(OrderFact.order_key))]).group_by(DateDimension.days_name, DateDimension.month).all()
        heatmap_data = [{"day": row[0], "month": row[1], "total_orders": int(row[2])} for row in heatmap_query]

        # --------------------------------------------------
        # 8. Data Tabel Raw (Detail Pesanan)
        # --------------------------------------------------
        # Menggunakan custom_filter=base_filter agar mengambil SEMUA status (bukan cuma SELESAI) yang lolos slicer
        if statuses:
            table_filter = base_filter_no_status + [OrderFact.status.in_(statuses)]
        else:
            table_filter = base_filter_no_status
            
        raw_data_query = _base([
            OrderFact.order_key,
            OrderFact.status,
            ProductDimension.product_model,
            ProductDimension.product_color,
            ProductDimension.product_size,
            OrderFact.price,
            LocationDimension.province,
            LocationDimension.city,
            OrderFact.cancel_reason
        ], custom_filter=table_filter).order_by(DateDimension.date.desc()).limit(400).all()
        
        raw_table_data = []
        for row in raw_data_query:
            raw_table_data.append({
                "order_key": row[0],
                "status": row[1],
                "model": row[2],
                "color": row[3],
                "size": row[4],
                "price": float(row[5]) if row[5] else 0,
                "province": row[6],
                "city": row[7],
                "cancel_reason": row[8] if row[8] else "-"
            })
            
        # --------------------------------------------------
        # 9. Top 3 Platform Batal
        # --------------------------------------------------
        cancel_platform_query = _base([
            PlatformDimension.platform_name, 
            func.count(func.distinct(OrderFact.order_key))
        ]).group_by(PlatformDimension.platform_name).order_by(func.count(func.distinct(OrderFact.order_key)).desc()).limit(3).all()

        cplat_labels = [r[0] for r in cancel_platform_query]
        cplat_values = [int(r[1]) for r in cancel_platform_query]

        # --------------------------------------------------
        # Return
        # --------------------------------------------------
        return {
            "cancel_platform_labels": cplat_labels,
            "cancel_platform_values": cplat_values,
            "color_options": color_options, "size_options": size_options, "province_options": province_options,
            "platform_options": platform_options, "month_options": month_options, "year_options": year_options,
            "payment_cat_options": payment_cat_options, "model_options": model_options,
            "status_options": status_options,
            "raw_table_data": raw_table_data,
            "cancellation_rate": cancellation_rate, "total_all_status": total_all_status,
            "num_models": num_models, "num_orders": num_orders, "revenue_month": float(revenue_total),
            "num_cities": num_cities, "aov": round(float(aov), 0), "overall_avg_revenue": round(overall_avg_revenue, 0),
            "overall_avg_orders": round(overall_avg_orders, 2), "ramadhan_lift": ramadhan_lift, "has_comparison": has_comparison,
            "ramadhan_avg_revenue": round(float(ramadhan_avg_revenue), 0), "normal_avg_revenue": round(float(normal_avg_revenue), 0),
            "ramadhan_avg_orders": round(float(ramadhan_avg_orders), 2), "normal_avg_orders": round(float(normal_avg_orders), 2),
            "map_data": map_data,
            "color_labels": [c[0] for c in color_data], "color_values": [int(c[1]) for c in color_data],
            "line_data": [{"date": str(d[0]), "platform": d[1], "amount": float(d[2])} for d in line_data],
            "top_products": [[p[0], int(p[1])] for p in top_products],
            "payment_labels": [p[0] for p in payment_data], "payment_values": [float(p[1]) if p[1] is not None else 0 for p in payment_data],
            "size_labels": [s[0] for s in size_data], "size_values": [int(s[1]) for s in size_data],
            "pareto_data": pareto_data,
            "heatmap_data": heatmap_data,
            "province_city_map": province_city_map,
            "city_options": city_options
        }

    finally:
        session.close()


@app.route("/dashboard")
@login_required
def dashboard_view():
    start_date = request.args.get("start_date")
    end_date = request.args.get("end_date")
    is_twin_date = request.args.get("is_twin_date")
    
    # Ambil list untuk fitur Multiple Select
    platforms = request.args.getlist("platform")
    colors = request.args.getlist("color")
    sizes = request.args.getlist("size")
    provinces = request.args.getlist("province")
    cities = request.args.getlist("city")
    payment_categories = request.args.getlist("payment_category")
    models = request.args.getlist("model")
    statuses = request.args.getlist("status")
    
    # Ambil nilai Range Harga
    min_price = request.args.get("min_price")
    max_price = request.args.get("max_price")

    metrics = get_dashboard_metrics(
        start_date=start_date, end_date=end_date, platforms=platforms, 
        colors=colors, sizes=sizes, provinces=provinces, models=models,
        is_twin_date=is_twin_date, payment_categories=payment_categories,
        min_price=min_price, max_price=max_price, statuses=statuses, cities=cities
    )
    return render_template("dashboard.html", m=metrics)


@app.route("/api/metrics/location/cities")
@login_required
def drilldown_cities():
    province = request.args.get("province")
    if not province:
        return jsonify({"error": "Provinsi tidak diberikan"}), 400
        
    session = SessionLocal()
    try:
        cities = (
            session.query(LocationDimension.city, func.sum(OrderFact.quantity))
            .join(OrderFact, LocationDimension.location_id == OrderFact.location_id)
            .filter(LocationDimension.province == province, OrderFact.status == 'SELESAI')
            .group_by(LocationDimension.city)
            .order_by(func.sum(OrderFact.quantity).desc())
            .all()
        )
        return jsonify([{"city": c[0], "quantity": int(c[1])} for c in cities])
    finally:
        session.close()
        
        
@app.route("/api/metrics/product/breakdown")
@login_required
def drilldown_product():
    model_name = request.args.get("model")
    if not model_name:
        return jsonify({"error": "Model tidak diberikan"}), 400
        
    session = SessionLocal()
    try:
        colors = (
            session.query(ProductDimension.product_color, func.sum(OrderFact.quantity))
            .join(OrderFact, ProductDimension.product_id == OrderFact.product_id)
            .filter(ProductDimension.product_model == model_name, OrderFact.status == 'SELESAI')
            .group_by(ProductDimension.product_color)
            .all()
        )
        sizes = (
            session.query(ProductDimension.product_size, func.sum(OrderFact.quantity))
            .join(OrderFact, ProductDimension.product_id == OrderFact.product_id)
            .filter(ProductDimension.product_model == model_name, OrderFact.status == 'SELESAI')
            .group_by(ProductDimension.product_size)
            .all()
        )
        return jsonify({
            "colors": [{"color": c[0], "qty": int(c[1])} for c in colors],
            "sizes": [{"size": s[0], "qty": int(s[1])} for s in sizes]
        })
    finally:
        session.close()

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
    except Exception:  # noqa: BLE001
        return None


def generate_recursive_forecast(model, days_ahead=14):
    query_hist = """
        SELECT 
            dd.date AS order_date, 
            MAX(pr.is_muslim_fashion) as is_muslim_fashion, 
            SUM(fa.total_quantity) as quantity
        FROM fact_daily_agregat fa
        JOIN date_dimension dd ON dd.date_id = fa.date_id
        JOIN product_dimension pr ON pr.product_id = fa.product_id
        WHERE fa.status = 'SELESAI'
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
            date AS order_date, month,
            is_twin_date, is_ramadhan
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

    cat_cols = ["month"]
    for col in cat_cols:
        combined_df[col] = combined_df[col].astype("category")

    feature_cols = [
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
        
        rolling_3 = (
            np.mean(qty_history[-3:]) if len(qty_history) >= 3 else np.mean(qty_history)
        )
        rolling_7 = (
            np.mean(qty_history[-7:]) if len(qty_history) >= 7 else np.mean(qty_history)
        )
       
        std_7 = np.std(qty_history[-7:], ddof=1) if len(qty_history) >= 2 else 0.0

        lag_7_rolling_mean = (
            np.mean(qty_history[-14:-7]) if len(qty_history) >= 14 else 0
        )

        combined_df.at[i, "rolling_mean_3_qty"] = rolling_3
        combined_df.at[i, "rolling_mean_7_qty"] = rolling_7
        combined_df.at[i, "rolling_std_7_qty"] = std_7
        combined_df.at[i, "lag_1_qty"] = lag_1
        combined_df.at[i, "lag_3_qty"] = lag_3
        combined_df.at[i, "lag_7_qty"] = lag_7
        combined_df.at[i, "lag_7_rolling_mean"] = lag_7_rolling_mean

        combined_df.at[i, "day_of_year"] = current_date.dayofyear
        combined_df.at[i, "is_month_start"] = int(current_date.is_month_start)

        current_features = combined_df.iloc[[i]].drop(
            columns=["order_date", "quantity"]
        )

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
                SUM(fd.total_quantity) as qty
            FROM fact_daily_agregat fd
            JOIN date_dimension dd ON dd.date_id = fd.date_id
            JOIN product_dimension pr ON pr.product_id = fd.product_id
            WHERE fd.status = 'SELESAI' 
            AND dd.date >= (SELECT DATE_SUB(MAX(dd2.date), INTERVAL 90 DAY) 
                            FROM fact_daily_agregat fd2 
                            JOIN date_dimension dd2 ON fd2.date_id = dd2.date_id)
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
                "forecast_qty": round(
                        total_forecast_qty
                        * (
                            row["qty"] / model_weight_total
                            if model_weight_total > 0
                            else 0
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
                "forecast_qty": round(
                        total_forecast_qty
                        * (
                            row["qty"] / color_weight_total
                            if color_weight_total > 0
                            else 0
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

    except Exception as e:  # noqa: BLE001
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

    except Exception as e:  # noqa: BLE001
        return render_template(
            "forecast.html", error=f"Terjadi kesalahan teknis: {str(e)}"  # noqa: RUF010
        )


if __name__ == "__main__":
    app.run(debug=True, port=5000)