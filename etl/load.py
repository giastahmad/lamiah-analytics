import re

import pandas as pd
from rapidfuzz import fuzz, process
from sqlalchemy import text
from sqlalchemy.dialects.mysql import insert

from config import engine


def clean_and_override_province(province_str):
    if pd.isna(province_str): return province_str
    val = str(province_str).upper().strip()
    val = val.replace("PROPINSI", "").replace("PROVINSI", "").replace("PROV.", "").replace("PROV", "").strip()
    
    eng_to_id = {
        "WEST JAVA": "JAWA BARAT",
        "CENTRAL JAVA": "JAWA TENGAH",
        "EAST JAVA": "JAWA TIMUR",
        "NORTH SUMATRA": "SUMATERA UTARA",
        "SOUTH SUMATRA": "SUMATERA SELATAN",
        "WEST SUMATRA": "SUMATERA BARAT",
        "WEST NUSA TENGGARA": "NUSA TENGGARA BARAT",
        "EAST NUSA TENGGARA": "NUSA TENGGARA TIMUR",
        "WEST KALIMANTAN": "KALIMANTAN BARAT",
        "CENTRAL KALIMANTAN": "KALIMANTAN TENGAH",
        "EAST KALIMANTAN": "KALIMANTAN TIMUR",
        "SOUTH KALIMANTAN": "KALIMANTAN SELATAN",
        "NORTH KALIMANTAN": "KALIMANTAN UTARA",
        "NORTH SULAWESI": "SULAWESI UTARA",
        "CENTRAL SULAWESI": "SULAWESI TENGAH",
        "SOUTH SULAWESI": "SULAWESI SELATAN",
        "SOUTHEAST SULAWESI": "SULAWESI TENGGARA",
        "WEST SULAWESI": "SULAWESI BARAT",
        "SOUTH PAPUA": "PAPUA SELATAN",
        "CENTRAL PAPUA": "PAPUA TENGAH",
        "HIGHLAND PAPUA": "PAPUA PEGUNUNGAN",
        "SOUTHWEST PAPUA": "PAPUA BARAT DAYA",
        "WEST PAPUA": "PAPUA BARAT",
        "SPECIAL REGION OF YOGYAKARTA": "DI YOGYAKARTA",
        "SPECIAL CAPITAL REGION OF JAKARTA": "DKI JAKARTA",
        "BALI": "BALI",
        "BANTEN": "BANTEN",
        "GORONTALO": "GORONTALO",
        "MALUKU": "MALUKU",
        "NORTH MALUKU": "MALUKU UTARA",
        "JAMBI": "JAMBI",
        "BENGKULU": "BENGKULU",
        "LAMPUNG": "LAMPUNG"
    }

    if val in eng_to_id:
        val = eng_to_id[val]
        
    if "NANGGROE" in val or "NAD" in val or "ACEH" in val: return "ACEH"
    if val in ["KEPRI", "KEP. RIAU", "KEPULAUAN RIAU"]: return "KEPULAUAN RIAU"
    elif val in ["DKI JAKARTA", "JAKARTA RAYA", "DKI", "JAKARTA"]: return "DKI JAKARTA"
    elif val in ["BABEL", "BANGKA BELITUNG", "KEP. BANGKA BELITUNG"]: return "KEPULAUAN BANGKA BELITUNG"
    
    return val

def clean_and_standardize_city(city_str):
    if pd.isna(city_str):
        return city_str
        
    val = str(city_str).upper().strip()  
    val = re.sub(r'[\¶\n\t\r]+', ' ', val)
    
    if "CITY" in val:
        val = "KOTA " + val.replace("(CITY)", "").replace("CITY", "")
        
    if "REGENCY" in val:
        val = "KABUPATEN " + val.replace("(REGENCY)", "").replace("REGENCY", "")
    
    val = re.sub(r'\(.*?\)', '', val)
    val = val.replace("KOTA WARINGIN", "KOTAWARINGIN")
    val = re.sub(r'\s+', ' ', val)
    
    val = re.sub(r'^(KABUPATEN|KAB\.|KAB)\s+', 'KABUPATEN ', val)
    val = re.sub(r'^(KOTA ADMINISTRASI|KOTA ADM\.|KOTA ADM|KOTA)\s+', 'KOTA ', val)
    val = val.replace("KABUPATEN ADM. ", "KABUPATEN ")

    val = re.sub(r'^(KOTA\s+)+', 'KOTA ', val)
    val = re.sub(r'^(KABUPATEN\s+)+', 'KABUPATEN ', val)
    
    return val.strip()

def fuzzy_map_location(df_trans, dim_location, threshold=85):
    print("      -> Clearing location data & running fuzzy match...")
    df_trans['province_clean'] = df_trans['province'].apply(clean_and_override_province)
    df_trans['city_clean'] = df_trans['city'].apply(clean_and_standardize_city)
    
    dim_location['city_clean'] = dim_location['city'].apply(clean_and_standardize_city)
    
    df_trans['loc_key_raw'] = df_trans['province_clean'].astype(str) + " - " + df_trans['city_clean'].astype(str)
    dim_location['loc_key_seeder'] = dim_location['province'].astype(str) + " - " + dim_location['city_clean'].astype(str)
    
    seeder_choices = dim_location['loc_key_seeder'].tolist()
    unique_raw_locations = df_trans['loc_key_raw'].dropna().unique()
    
    location_mapping = {}
    
    for raw_loc in unique_raw_locations:
        match = process.extractOne(raw_loc, seeder_choices, scorer=fuzz.WRatio)
        if match and match[1] >= threshold:
            location_mapping[raw_loc] = match[0] 
        else:
            location_mapping[raw_loc] = None     
            
    df_trans['loc_key_matched'] = df_trans['loc_key_raw'].map(location_mapping)

    df_result = df_trans.merge(
        dim_location[['loc_key_seeder', 'location_id']], 
        left_on='loc_key_matched', 
        right_on='loc_key_seeder', 
        how='left'
    )

    cols_to_drop = ['loc_key_raw', 'loc_key_matched', 'loc_key_seeder', 'province_clean', 'city_clean']
    df_result = df_result.drop(columns=cols_to_drop)
    
    return df_result

def load_product_dimension(engine_conn, df_transformed):
    df_unique = df_transformed[['SKU', 'color', 'size', 'is_muslim_fashion']].drop_duplicates()
    df_unique = df_unique.rename(columns={
        'SKU': 'product_model',
        'color': 'product_color',
        'size': 'product_size'
    })

    try:
        existing_products = pd.read_sql(
            "SELECT product_id, product_model, product_color, product_size FROM product_dimension", 
            engine_conn
        )
    except Exception:  # noqa: BLE001
        existing_products = pd.DataFrame(columns=['product_id', 'product_model', 'product_color', 'product_size'])

    merged = df_unique.merge(existing_products, on=['product_model', 'product_color', 'product_size'], how='left', indicator=True)
    new_products = merged[merged['_merge'] == 'left_only'][['product_model', 'product_color', 'product_size', 'is_muslim_fashion']]

    if not new_products.empty:
        print(f"      -> Adding {len(new_products)} new products to product_dimension...")
        new_products.to_sql('product_dimension', engine_conn, if_exists='append', index=False)
    else:
        print("      -> No new products to add.")

    updated_products = pd.read_sql("SELECT product_id, product_model, product_color, product_size FROM product_dimension", engine_conn)

    df_fact_ready = df_transformed.rename(columns={
        'SKU': 'product_model',
        'color': 'product_color',
        'size': 'product_size'
    })
    
    df_fact_ready = df_fact_ready.merge(updated_products, on=['product_model', 'product_color', 'product_size'], how='left')

    return df_fact_ready

def mysql_upsert(table, conn, keys, data_iter):
    data = [dict(zip(keys, row)) for row in data_iter]
    
    insert_stmt = insert(table.table).values(data)
    
    upsert_stmt = insert_stmt.on_duplicate_key_update(
        status=insert_stmt.inserted.status,
        cancel_reason=insert_stmt.inserted.cancel_reason,
        date_id=insert_stmt.inserted.date_id,
        payment_method_id=insert_stmt.inserted.payment_method_id,
        platform_id=insert_stmt.inserted.platform_id,
        location_id=insert_stmt.inserted.location_id,
        quantity=insert_stmt.inserted.quantity,
        price=insert_stmt.inserted.price,
        discount=insert_stmt.inserted.discount,
        total_amount=insert_stmt.inserted.total_amount
    )
    
    conn.execute(upsert_stmt)

def load_data_warehouse(df_transformed):
    print("\n[START] Starting the Load process to the Data Warehouse...")
    
    print("   1. Processing Product Dimensions...")
    df = load_product_dimension(engine, df_transformed)
    
    print("   2. Fetching Seeder data from static dimensions...")
    dim_date     = pd.read_sql("SELECT date_id, date FROM date_dimension", engine)
    dim_location = pd.read_sql("SELECT location_id, province, city FROM location_dimension", engine)
    dim_platform = pd.read_sql("SELECT platform_id, platform_name FROM platform_dimension", engine)
    dim_payment  = pd.read_sql("SELECT payment_method_id, payment_method_name FROM payment_method_dimension", engine)
    
    print("   3. Performing Mapping/Lookup for Foreign Keys...")
    df['date']       = pd.to_datetime(df['date']).dt.date
    dim_date['date'] = pd.to_datetime(dim_date['date']).dt.date
    df = df.merge(dim_date,     on='date',               how='left')
    df = df.merge(dim_platform, left_on='platform',      right_on='platform_name',      how='left')
    df = df.merge(dim_payment,  left_on='payment_method', right_on='payment_method_name', how='left')
    df = fuzzy_map_location(df, dim_location, threshold=85)
    
    missing_locs = df[df['location_id'].isnull()]
    if not missing_locs.empty:
        print(f"   [WARNING] There are {len(missing_locs)} rows with unknown locations (location_id = NULL).")
    
    print("   4. Filtering columns for Order Fact...")
    fact_columns = [
        'order_key', 'date_id', 'payment_method_id', 'product_id',
        'platform_id', 'location_id', 'status', 'cancel_reason', 'quantity', 'price',
        'discount', 'total_amount'
    ]
    df_order_fact = df[fact_columns].copy()
    
    print("   5. Combining duplicate items within the same order...")
    df_order_fact = df_order_fact.groupby(
        ['order_key', 'date_id', 'payment_method_id', 'product_id', 'platform_id', 'location_id', 'status', 'cancel_reason'],
        dropna=False,
        as_index=False
    ).agg({
        'quantity':     'sum',
        'price':        'mean',
        'discount':     'sum',
        'total_amount': 'sum'
    })
 
    # ── Unknown SKU quality gate ────────────────────────────────────────────────
    UNKNOWN_PRODUCT_ID = 47

    total_rows    = len(df_order_fact)
    unknown_rows  = len(df_order_fact[df_order_fact['product_id'] == UNKNOWN_PRODUCT_ID])
    total_revenue = df_order_fact['total_amount'].sum()
    unknown_revenue = df_order_fact[
        df_order_fact['product_id'] == UNKNOWN_PRODUCT_ID
    ]['total_amount'].sum()

    row_pct = 0.0
    rev_pct = 0.0

    if total_rows > 0 and total_revenue > 0:
        row_pct = (unknown_rows / total_rows) * 100
        rev_pct = (unknown_revenue / total_revenue) * 100

        print("   [DATA QUALITY] Product Status UNKNOWN:")
        print(f"      - Volume : {row_pct:.2f}% ({unknown_rows} from {total_rows} transactions)")
        print(f"      - Revenue: {rev_pct:.2f}% from total revenue of this batch")

        if row_pct > 5.0 or rev_pct > 3.0:
        # if row_pct > 100.0 or rev_pct > 100.0:
            print("      [CRITICAL WARNING] UNKNOWN percentage too high — aborting load.")
            return {
                "code":        "ABORT_HIGH_UNKNOWN",
                "rows_loaded": 0,
                "rows_skipped": 0,
                "row_pct":     round(row_pct, 2),
                "rev_pct":     round(rev_pct, 2),
            }
 
    # ── Insert (Upsert) ─────────────────────────────────────────────────────────
    print(f"   6. Upserting {len(df_order_fact)} rows into order_fact...")

    try:
        df_order_fact.to_sql(
            'order_fact', 
            engine, 
            if_exists='append', 
            index=False,
            method=mysql_upsert,
            chunksize=1000
        )
        print("[SUCCESS] Data successfully loaded (and updated) into Data Warehouse!")
        
        unique_date_ids = df_order_fact['date_id'].dropna().unique().tolist()
        if unique_date_ids:
           
            date_ids_str = ','.join(map(str, unique_date_ids))
            
            summary_sql = f"""
                INSERT INTO fact_daily_agregat 
                (date_id, product_id, status, total_orders, total_quantity, total_amount)
                SELECT 
                    date_id, product_id, status,
                    COUNT(DISTINCT order_key), SUM(quantity), SUM(total_amount)
                FROM order_fact
                WHERE date_id IN ({date_ids_str})
                GROUP BY date_id, product_id, status
                ON DUPLICATE KEY UPDATE
                    total_orders = VALUES(total_orders),
                    total_quantity = VALUES(total_quantity),
                    total_amount = VALUES(total_amount);
            """
            with engine.begin() as conn:
                conn.execute(text(summary_sql))            
            
    except Exception as e:
        print(f"[ERROR] Failed to upsert data: {e}")
        raise

    has_warning = (row_pct > 5.0) or (rev_pct > 3.0)

    return {
        "code":         "SUCCESS_WITH_WARNING" if has_warning else "SUCCESS",
        "rows_loaded":  len(df_order_fact),
        "rows_skipped": 0,
        "row_pct":      round(row_pct, 2),
        "rev_pct":      round(rev_pct, 2),
    }