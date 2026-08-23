import re

import pandas as pd

COLUMN_MAPPING = {
    "order_key": ["order_id", "nomor_pesanan", "No. Pesanan", "Order ID", "no_pesanan"],
    "product_name": ["Nama Produk", "Product Name"],
    "status": ["Status Pesanan", "Order Status", "status_pesanan", "Status"],
    "date": ["Waktu Pesanan Dibuat", "Created Time", "Order Created"],
    "payment_method": ["Metode Pembayaran", "Payment Method"],
    "SKU": ["SKU Induk", "Seller SKU", "sku"],
    "variant": ["Nama Variasi", "Variation"],
    "price": ["Harga Awal", "SKU Unit Original Price", "Unit Price"],
    "price_after_discount": ["Harga Setelah Diskon", "Price After Discount"],
    "SKU_platform_disc": ["SKU Platform Discount"],
    "SKU_seller_disc": ["SKU Seller Discount"],
    "quantity": ["Jumlah", "Quantity"],
    "total_amount": ["Total Pembayaran", "Total payment", "Order Amount"],
    "province": ["Provinsi", "Province"],
    "city": ["Kota/Kabupaten", "Regency and City", "City"],
    "channel": ["Purchase Channel"],
    "notes": ["Catatan dari Pembeli", "Buyer Message"],
    "cancel_reason": ["Alasan Pembatalan", "Cancel Reason"],
}

REQUIRED_COLUMNS_SHOPEE = [
    "order_key",
    "status",
    "product_name",
    "date",
    "payment_method",
    "SKU",
    "variant",
    "price",
    "price_after_discount",
    "quantity",
    "total_amount",
    "province",
    "city",
    "notes",
    "cancel_reason",
]

REQUIRED_COLUMNS_TOKOPEDIA = [
    "order_key",
    "status",
    "product_name",
    "date",
    "payment_method",
    "SKU",
    "variant",
    "price",
    "SKU_platform_disc",
    "SKU_seller_disc",
    "quantity",
    "total_amount",
    "province",
    "city",
    "channel",
    "notes",
    "cancel_reason",
]

PAYMENT_METHOD = {
    "COD": [
        [
            "cod (bayar di tempat)",
            "cod",
            "bayar di tempat (cod)",
            "bayar di tempat",
            "cash",
        ],
        "MANUAL",
    ],
    "PAYLATER": [
        [
            "spaylater",
            "gopay later",
            "paylater",
            "later",
            "paylater + tiktok shop balance",
        ],
        "LEVERAGE",
    ],
    "E-WALLET": [
        [
            "shopeepay",
            "dana",
            "gopay",
            "ovo",
            "linkaja",
            "ewallet",
            "e wallet",
            "dana + tiktok shop balance",
            "ovo + tiktok shop balance",
            "gopay + tiktok shop balance",
            "tiktok shop balance + qris",
        ],
        "LIQUID",
    ],
    "BANK_TRANSFER": [
        [
            "seabank",
            "jago",
            "transfer bank",
            "online payment",
            "qris",
            "bank transfer",
            "bank",
            "seabank bayar instan",
        ],
        "LIQUID",
    ],
    "CREDIT_CARD": [
        ["kartu kredit/debit", "credit card", "debit", "cicilan kartu kredit"],
        "LEVERAGE",
    ],
    "STORE_BALANCE": [
        ["saldo penjual", "tiktok shop balance", "balance", "saldo shopeepay"],
        "LIQUID",
    ],
    "OFFLIBE_AGENT": [["mitra shopee"], "MANUAL"],
    "RETAIL_OUTLET": [["alfamart/alfamidi/dan+dan", "indomaret/i.saku"], "MANUAL"],
}

COLOR_MAPPING = {
    "ABU-ABU MUDA": "ABU-ABU",
    "BIRU MUDA": "BIRU MUDA",
    "BIRUMUDA": "BIRU MUDA",
    "BIRUMUDA/DENIM(NOTE)": "DENIM",
    "BIRU TUA": "BIRU TUA",
    "BLUSH PINK": "BLUSH PINK",
    "BUTTER": "BUTTER",
    "BROKEN WHITE": "BROKEN WHITE",
    "CREAM": "BROKEN WHITE",
    "BURGUNDY": "BURGUNDY",
    "COKELAT": "COKELAT",
    "COKLAT": "COKELAT",
    "COKELAT SUSU": "COKELAT SUSU",
    "COKLAT SUSU": "COKELAT SUSU",
    "COKSU": "COKELAT SUSU",
    "COKSU DK": "COKELAT SUSU",
    "COKSU/COKLATUA(NOTE)": "COKELAT TUA",
    "COKSU/COKTU(NOTE)": "COKELAT TUA",
    "COKELAT TUA": "COKELAT TUA",
    "COKLAT TUA": "COKELAT TUA",
    "COKTU": "COKELAT TUA",
    "COKLATUA/COKSU(NOTE)": "COKELAT TUA",
    "DENIM": "DENIM",
    "DENIM/BIRUMUDA(NOTE)": "DENIM",
    "DUSTY ROSE": "DUSTYPINK",
    "DUSTYPINK": "DUSTYPINK",
    "EMERALD": "EMERALD",
    "EMERALD BLUE": "EMERALD BLUE",
    "EMAS": "GOLD",
    "GOLD": "GOLD",
    "HIJAU": "HIJAU",
    "HIJAU BOTOL": "HIJAU BOTOL",
    "HITAM": "HITAM",
    "LAVENDER": "LAVENDER",
    "LAVENDER/LILAC": "LAVENDER",
    "LAVENDER/LILAC(NOTE)": "LAVENDER",
    "LILAC": "LAVENDER",
    "UNGU MUDA": "LAVENDER",
    "MAROON": "MAROON",
    "MAROON DK": "MAROON",
    "MERAH TUA": "MERAH",
    "NAVY": "NAVY",
    "NAVY DK": "NAVY",
    "PINK": "PINK",
    "PUTIH": "PUTIH",
    "ROSE GOLD": "ROSE GOLD",
    "ROSEGOL DK": "ROSE GOLD",
    "ROSEGOLD": "ROSE GOLD",
    "SAGE": "SAGE",
    "PERAK": "SILVER",
    "SILVER": "SILVER",
    "SILVER DK": "SILVER",
    "TERRACOTTA": "TERRACOTTA",
    "TOSCA": "TOSCA",
}

SKU_MAPPING = {
    "AERA": "AERA",
    "AERA LD": "AERA LONG DRESS",
    "AERA LONG DRESS": "AERA LONG DRESS",
    "ALYA": "ALYA",
    "AMEERA": "AMEERA",
    "AURORA": "AURORA",
    "RM": "AURORA",
    "BBRM": "BBRM",
    "BELLE DOT": "BELLE DOT",
    "K3 POLKADOT": "BELLE DOT",
    "CALLA": "CALLA",
    "DASPOL": "DASPOL",
    "DRESS POLKADOT": "DASPOL",
    "DK": "DK",
    "DOT KERUT": "DOT KERUT",
    "ELENA": "KT",
    "KT": "KT",
    "HAWAI": "HAWAI",
    "JASMINE": "JASMINE",
    "K3": "K3",
    "TL K3": "K3",
    "LULU": "LULU",
    "ADEYYA": "M ADHEYYA",
    "ADEYYA MIDI": "M ADHEYYA",
    "M ADHEYYA": "M ADHEYYA",
    "M UNICORN": "M UNICORN",
    "MELLA": "MELLA",
    "MR2": "MR2",
    "MV": "MV",
    "NAMI": "NAMI",
    "NINA": "NINA",
    "PELANGI": "PELANGI",
    "POLKADOT": "RP",
    "RP": "RP",
    "RENDA SUSUN": "RENDA SUSUN",
    "SEQUIN": "SEQUIN",
    "SLAYER": "SLAYER",
    "SLAYER SML": "SLAYER SML",
    "STL": "STL",
    "TL DOT": "TL DOT",
    "TL POLKADOT": "TL DOT",
    "BBV": "VIVI",
    "VIVI": "VIVI",
    "ZZ": "ZZ",
}

MUSLIM = [
    "ALYA",
    "MR2",
    "JASMINE",
    "M UNICORN",
    "AMEERA",
    "AERA LONG DRESS",
    "M ADHEYYA",
]


def map_columns(df):
    mapping_flat = {
        opt.lower(): standard
        for standard, options in COLUMN_MAPPING.items()
        for opt in options
    }

    new_columns = {
        col: mapping_flat[col.lower()]
        for col in df.columns
        if col.lower() in mapping_flat
    }

    return df.rename(columns=new_columns)


def map_data_payment(df):
    df["payment_method"] = df["payment_method"].str.lower().str.strip()

    mapping_flat = {
        opt: standard for standard, data in PAYMENT_METHOD.items() for opt in data[0]
    }

    mapping_category = {standard: data[1] for standard, data in PAYMENT_METHOD.items()}

    df["payment_method"] = df["payment_method"].map(mapping_flat)
    df["payment_category"] = df["payment_method"].map(mapping_category)

    return df


def extract_size(val):
    if pd.isna(val) or not isinstance(val, str):
        return None

    match = re.search(r"\d+", val)

    if match:
        num = int(match.group(0))

        if num % 2 != 0:
            return f"{num}-{num + 1} Tahun"
        else:
            return f"{num - 1}-{num} Tahun"

    return None


def fix_indonesian_price(val):
    if pd.isna(val):
        return 0

    if isinstance(val, str):
        val = val.replace("Rp", "").replace(" ", "").replace(".", "")
        try:
            return float(val)
        except:  # noqa: E722
            return 0

    elif isinstance(val, (int, float)):
        if 0 < val < 1000:
            return val * 1000
        return float(val)

    return 0


def transform_shopee(df):
    df_standard = map_columns(df)
    df_standard = df_standard[REQUIRED_COLUMNS_SHOPEE]

    original_status = df_standard["status"].astype(str).str.upper().str.strip()

    df_standard["cancel_reason"] = df_standard["cancel_reason"].where(
        original_status.isin(["BATAL", "DIBATALKAN"]), None
    )

    df_standard["status"] = original_status

    df_standard.loc[df_standard["status"] != "SELESAI", "status"] = "BATAL"

    df_standard["total_amount"] = pd.to_numeric(
        df_standard["total_amount"], errors="coerce"
    ).fillna(0)
    kondisi_simpan = (df_standard["status"] == "BATAL") | (
        df_standard["total_amount"] > 0
    )
    df_standard = df_standard[kondisi_simpan]

    df_standard["date"] = pd.to_datetime(df_standard["date"]).dt.date

    df_standard = map_data_payment(df_standard)

    df_standard["SKU"] = df_standard["SKU"].astype(str).str.upper()
    df_standard["SKU"] = df_standard["SKU"].map(SKU_MAPPING)

    sorted_keys = sorted(SKU_MAPPING.keys(), key=len, reverse=True)
    pattern = "|".join(map(re.escape, sorted_keys))
    missing_sku_mask = df_standard["SKU"].isna()

    extracted_variant_sku = (
        df_standard.loc[missing_sku_mask, "variant"]
        .str.upper()
        .str.extract(f"({pattern})", expand=False)
    )
    extracted_product_sku = (
        df_standard.loc[missing_sku_mask, "product_name"]
        .str.upper()
        .str.extract(f"({pattern})", expand=False)
    )
    extracted_keys = extracted_variant_sku.fillna(extracted_product_sku)

    df_standard.loc[missing_sku_mask, "SKU"] = extracted_keys.map(SKU_MAPPING)
    df_standard["SKU"] = df_standard["SKU"].str.upper()

    color_keys = sorted(COLOR_MAPPING.keys(), key=len, reverse=True)
    regex_pattern = "|".join(map(re.escape, color_keys))

    extracted_variant_color = (
        df_standard["variant"]
        .str.upper()
        .str.extract(f"({regex_pattern})", expand=False)
    )
    extracted_product_color = (
        df_standard["product_name"]
        .str.upper()
        .str.extract(f"({regex_pattern})", expand=False)
    )
    extracted_color_raw = extracted_variant_color.fillna(extracted_product_color)

    ambiguous_mask = extracted_color_raw.str.contains(r"/|NOTE", na=False, regex=True)

    extracted_notes = (
        df_standard.loc[ambiguous_mask, "notes"]
        .fillna("")
        .astype(str)
        .str.upper()
        .str.extract(f"({regex_pattern})", expand=False)
    )

    final_extracted_color = extracted_color_raw.copy()
    final_extracted_color.loc[ambiguous_mask] = extracted_notes.fillna(
        extracted_color_raw.loc[ambiguous_mask]
    )

    df_standard["color"] = final_extracted_color.map(COLOR_MAPPING)

    df_standard["size"] = df_standard["variant"].str.upper().apply(extract_size)

    curr_columns = ["price", "price_after_discount", "total_amount"]
    for col in curr_columns:
        df_standard[col] = df_standard[col].apply(fix_indonesian_price)

    df_standard["discount"] = df_standard["price"] - df_standard["price_after_discount"]

    df_standard["line_total"] = (
        df_standard["price_after_discount"] * df_standard["quantity"]
    )
    order_totals = df_standard.groupby("order_key")["line_total"].transform("sum")

    df_standard["weight"] = df_standard["line_total"] / order_totals
    df_standard["total_amount"] = df_standard["total_amount"] * df_standard["weight"]
    df_standard["total_amount"] = df_standard["total_amount"].round(0)

    df_standard["platform"] = "Shopee"

    df_standard["is_muslim_fashion"] = df_standard["SKU"].isin(MUSLIM)

    df_standard = df_standard.drop(
        columns=[
            "variant",
            "weight",
            "line_total",
            "price_after_discount",
            "product_name",
            "notes",
        ]
    )

    variant = ["SKU", "color", "size"]
    df_standard.loc[df_standard[variant].isna().any(axis=1), "is_muslim_fashion"] = (
        False
    )
    df_standard.loc[df_standard[variant].isna().any(axis=1), variant] = "UNKNOWN"

    df_standard = df_standard[
        [
            "order_key",
            "status",
            "cancel_reason",
            "date",
            "SKU",
            "color",
            "size",
            "is_muslim_fashion",
            "payment_method",
            "payment_category",
            "province",
            "city",
            "platform",
            "price",
            "discount",
            "quantity",
            "total_amount",
        ]
    ]

    return df_standard


def transform_tokopedia(df):
    df_standard = map_columns(df)
    df_standard = df_standard[REQUIRED_COLUMNS_TOKOPEDIA]

    original_status = df_standard["status"].astype(str).str.upper().str.strip()

    df_standard["cancel_reason"] = df_standard["cancel_reason"].where(
        original_status.isin(["BATAL", "DIBATALKAN"]), None
    )

    df_standard["status"] = original_status

    df_standard.loc[df_standard["status"] != "SELESAI", "status"] = "BATAL"

    df_standard["total_amount"] = pd.to_numeric(
        df_standard["total_amount"], errors="coerce"
    ).fillna(0)
    kondisi_simpan = (df_standard["status"] == "BATAL") | (
        df_standard["total_amount"] > 0
    )
    df_standard = df_standard[kondisi_simpan]

    df_standard["date"] = pd.to_datetime(df_standard["date"]).dt.date

    df_standard = map_data_payment(df_standard)

    df_standard["SKU"] = df_standard["SKU"].astype(str).str.upper()
    df_standard["SKU"] = df_standard["SKU"].map(SKU_MAPPING)

    sorted_keys = sorted(SKU_MAPPING.keys(), key=len, reverse=True)
    pattern = "|".join(map(re.escape, sorted_keys))
    missing_sku_mask = df_standard["SKU"].isna()

    extracted_variant = (
        df_standard.loc[missing_sku_mask, "variant"]
        .str.upper()
        .str.extract(f"({pattern})", expand=False)
    )
    extracted_product = (
        df_standard.loc[missing_sku_mask, "product_name"]
        .str.upper()
        .str.extract(f"({pattern})", expand=False)
    )
    extracted_keys = extracted_variant.fillna(extracted_product)

    df_standard.loc[missing_sku_mask, "SKU"] = extracted_keys.map(SKU_MAPPING)
    df_standard["SKU"] = df_standard["SKU"].str.upper()

    color_keys = sorted(COLOR_MAPPING.keys(), key=len, reverse=True)
    regex_pattern = "|".join(map(re.escape, color_keys))

    extracted_variant_color = (
        df_standard["variant"]
        .str.upper()
        .str.extract(f"({regex_pattern})", expand=False)
    )
    extracted_product_color = (
        df_standard["product_name"]
        .str.upper()
        .str.extract(f"({regex_pattern})", expand=False)
    )
    extracted_color_raw = extracted_variant_color.fillna(extracted_product_color)

    ambiguous_mask = extracted_color_raw.str.contains(r"/|NOTE", na=False, regex=True)
    extracted_notes = (
        df_standard.loc[ambiguous_mask, "notes"]
        .fillna("")
        .astype(str)
        .str.upper()
        .str.extract(f"({regex_pattern})", expand=False)
    )

    final_extracted_color = extracted_color_raw.copy()
    final_extracted_color.loc[ambiguous_mask] = extracted_notes.fillna(
        extracted_color_raw.loc[ambiguous_mask]
    )

    df_standard["color"] = final_extracted_color.map(COLOR_MAPPING)

    df_standard["size"] = df_standard["variant"].str.upper().apply(extract_size)

    num_cols = [
        "price",
        "SKU_platform_disc",
        "SKU_seller_disc",
        "total_amount",
        "quantity",
    ]
    for col in num_cols:
        df_standard[col] = pd.to_numeric(df_standard[col])
    df_standard["discount"] = (
        df_standard["SKU_platform_disc"] + df_standard["SKU_seller_disc"]
    )

    df_standard["line_total"] = (
        df_standard["price"] - df_standard["discount"]
    ) * df_standard["quantity"]
    order_totals = df_standard.groupby("order_key")["line_total"].transform("sum")
    df_standard["weight"] = df_standard["line_total"] / order_totals
    df_standard["total_amount"] = df_standard["total_amount"] * df_standard["weight"]
    df_standard["total_amount"] = df_standard["total_amount"].round(0)

    df_standard = df_standard.rename(columns={"channel": "platform"})

    df_standard["is_muslim_fashion"] = df_standard["SKU"].isin(MUSLIM)

    df_standard["province"] = df_standard["province"].str.upper()
    df_standard["city"] = df_standard["city"].str.upper()

    df_standard = df_standard.drop(
        columns=[
            "variant",
            "weight",
            "line_total",
            "product_name",
            "SKU_platform_disc",
            "SKU_seller_disc",
            "notes",
        ]
    )

    variant = ["SKU", "color", "size"]

    df_standard.loc[df_standard[variant].isna().any(axis=1), "is_muslim_fashion"] = (
        False
    )
    df_standard.loc[df_standard[variant].isna().any(axis=1), variant] = "UNKNOWN"

    df_standard = df_standard[
        [
            "order_key",
            "status",
            "cancel_reason",
            "date",
            "SKU",
            "color",
            "size",
            "is_muslim_fashion",
            "payment_method",
            "payment_category",
            "province",
            "city",
            "platform",
            "price",
            "discount",
            "quantity",
            "total_amount",
        ]
    ]
    return df_standard