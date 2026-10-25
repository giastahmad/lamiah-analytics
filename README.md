# Lamiah Analytics

Lamiah Analytics is an internal analytics application built for Lamiah, a single fashion retail business selling through Shopee, Tokopedia, and TikTok Shop. It brings marketplace order data into one consistent view so the team can understand sales performance, compare channels, explore products and customer destinations, and plan inventory using a short-term demand forecast.

This is a store-specific business tool, not a general-purpose analytics product. Its data mappings, product catalogue, payment categories, and operational assumptions reflect Lamiah's order exports and reporting needs.

## Why this application exists

Each marketplace provides order data in its own export format and vocabulary. Product names, SKU values, variants, payment descriptions, order statuses, and location labels can differ between platforms or between export versions. Reviewing those files separately makes it difficult to answer basic business questions consistently, such as:

- How much revenue and how many completed orders did the store have in a selected period?
- Which marketplace, product model, color, or size contributed most?
- Where are orders being delivered, and how does demand vary by location?
- How often are orders cancelled, and what are the associated reasons?
- How do sales during Ramadan compare with other dates?
- What is the expected order quantity over the next two weeks?

Lamiah Analytics standardizes the exports and stores the resulting order facts in a shared warehouse. The dashboard and forecasting workflow can then use the same cleaned data and definitions rather than requiring separate manual analysis for each platform.

## What the application does

### Consolidates marketplace orders

The Upload page accepts order exports for Shopee, Tokopedia, and TikTok Shop. Multiple files can be processed in one upload. Supported formats are:

| Marketplace | Formats |
|---|---|
| Shopee | Excel (`.xlsx`) |
| Tokopedia | Excel (`.xlsx`) and CSV (`.csv`) |
| TikTok Shop | Excel (`.xlsx`) and CSV (`.csv`) |

The export needs the order, product/variant, date, payment, price, quantity, total, and destination fields expected by the selected marketplace transformer. The application recognizes known Indonesian and English column-header aliases; it is not intended to process arbitrary spreadsheets.

### Provides an interactive sales dashboard

The Dashboard page summarizes warehouse records and supports filtering by:

- Date range, month, and year
- Marketplace
- Product model, color, and size
- Province and city/regency
- Payment category
- Unit-price range
- Twin dates and order status

The dashboard presents sales and order KPIs, cancellation rate, average order value, product and color breakdowns, payment and size distributions, sales trends by platform, location views, and a Pareto-style product analysis. It also compares average revenue and orders on Ramadan and non-Ramadan days when the selected data supports that comparison. City and product drilldowns provide more detail without leaving the dashboard.

Completed (`SELESAI`) orders are the default basis for sales metrics. When a status filter is selected, the dashboard uses the selected status values for those metrics. Cancellation metrics are calculated separately from cancelled (`BATAL`) orders.

### Estimates near-term demand

The Forecast page displays a recursive 14-day forecast of total item quantity, together with estimated quantities for leading product models and colors. It is intended as a planning aid for stock and operational decisions, not as a guarantee of future sales.

## How it works

```text
Marketplace export files
          |
          v
Extract -> Transform -> Dimension mapping and quality checks
          |                                  |
          +----------------------------------v
                            MySQL order warehouse
                               |           |
                               v           v
                         Dashboard    Daily aggregates
                                            |
                                            v
                                    LightGBM forecast
```

### 1. Extract

The upload handler saves selected files under `data/uploads/` and calls the marketplace-specific reader:

- Shopee `.xlsx` exports are read with pandas.
- Tokopedia and TikTok `.xlsx` exports are read from the active worksheet, skipping the export's second row.
- Tokopedia and TikTok `.csv` exports are read as CSV files.

### 2. Transform and standardize

The ETL transformer maps recognized export headers to common field names, then prepares consistent warehouse values:

- Order statuses are normalized to uppercase. `SELESAI` remains completed; other source statuses are treated as `BATAL`.
- Product SKU aliases are mapped to Lamiah product models. If the SKU is missing or unrecognized, the transformer attempts to find a known SKU in the product variant or product name.
- Color names and spellings are mapped to canonical values. Variant text is used to derive size.
- Payment descriptions are mapped to standard payment methods and reporting categories such as `MANUAL`, `LIQUID`, and `LEVERAGE`.
- Prices and totals are converted from marketplace representations into numeric values. Discounts and line totals are derived; order totals are allocated across an order's product lines according to their line-total weights.
- Shopee and Tokopedia/TikTok exports use separate transformations to account for differences in their source columns and discount fields.

These rules encode Lamiah-specific product and reporting conventions in `etl/transform.py`; changing catalogue or marketplace conventions may require updating those mappings.

### 3. Map warehouse dimensions and check quality

The loader matches each normalized row to the warehouse's date, product, marketplace, payment, and location dimensions. City and province labels are cleaned and compared with the location reference data using fuzzy matching.

Rows with an unknown product are measured as a share of the uploaded batch. If unknown products exceed **5% of rows** or **3% of batch revenue**, loading is aborted for that batch and the upload result reports a warning. This check helps avoid contaminating reporting with poorly mapped products. Location mismatches are logged as warnings and may result in rows without a matched location.

### 4. Load and aggregate

Order facts are grouped to combine duplicate product lines within the same order. The warehouse uses the combination of order key and product as the unique order-line key; re-importing the same order/product updates its existing record rather than intentionally adding another line.

After loading, daily totals are recalculated for affected dates and grouped by product and status in `fact_daily_agregat`. The dashboard queries order facts and dimensions for its interactive views, while the aggregate table supplies compact daily history for model training and forecasting.

### 5. Build and serve forecasts

The forecasting pipeline uses completed-order daily aggregates. It creates a daily time series, fills missing sales days with zero quantity, and engineers time-series and calendar features, including recent rolling averages, lagged quantities, month, day of year, twin-date and Ramadan indicators, and the share of Muslim-fashion products.

`ml_models/retrain.py` trains a LightGBM challenger with chronological validation and compares its RMSE against the current champion on a later holdout period. It saves the challenger as `champion_model.pkl` only when its RMSE is lower than the existing model's; otherwise, it retains the champion.

At forecast time, the model predicts the next 14 days sequentially. Each predicted quantity is fed into the history used to calculate features for subsequent forecast days. The forecast cache stores the daily estimates and product/color summaries. A successful upload starts a background cache refresh; the Forecast page can also refresh the cache when the warehouse state has changed.

## Sign-in and access

Users sign in with Google through Firebase Authentication. The server verifies the Firebase ID token and checks that the verified email is present in the application's `lamiah_app.users` allowlist. Authentication alone does not grant access to an unregistered account.

Dashboard, upload, forecast, and drilldown pages are intended for authorized internal users. Uploaded spreadsheets and warehouse records contain business data and should be handled accordingly.

## Data model

The warehouse is organized around these SQLAlchemy models in `models.py`:

| Table | Purpose |
|---|---|
| `order_fact` | Order/product-line facts including status, quantity, price, discount, and allocated total |
| `fact_daily_agregat` | Daily quantity, order-count, and amount aggregates by date, product, and status |
| `date_dimension` | Calendar attributes including month, year, weekend, twin-date, payday, and Ramadan flags |
| `product_dimension` | Lamiah product model, color, size, and Muslim-fashion classification |
| `platform_dimension` | Marketplace name |
| `payment_method_dimension` | Standard payment method and reporting category |
| `location_dimension` | Province and city/regency |
| `lamiah_app.users` | Allowlist of application users |
| `lamiah_app.forecast_cache` | Cached forecast and leading-product/color estimates |

Reference dimensions are maintained by the seed scripts under `scripts/`. Location reference records are read from `data/provinsi.json` and `data/kabupaten_kota/`.

## Main application areas

| Area | What it is for |
|---|---|
| Login | Google sign-in and authorization against the internal user allowlist |
| Upload | Importing marketplace order exports and reviewing load results |
| Dashboard | Exploring sales, orders, products, payment methods, cancellations, and destinations |
| Forecast | Reviewing the next 14 days of predicted total quantity and product/color estimates |

The Flask application and routes are in `main.py`; the user-facing pages are in `templates/`. ETL steps are in `etl/`, data models are in `models.py`, and forecast training and saved models are in `ml_models/`.

## Important interpretation notes

- Dashboard and forecast quality depend on complete source exports, accurate SKU/product mapping, and seeded reference dimensions.
- The product and payment mappings represent current Lamiah business conventions; review results when those conventions or marketplace export formats change.
- Unmatched locations may not appear correctly in location-based analyses.
- Forecasts are based on historical completed quantities and the features available to the model. Promotions, stockouts, price changes, campaign schedules, and other external events may not be represented.
- Forecast allocations for top products and colors are based on recent historical sales mix; they are estimates, not independent product-level model predictions.
- Re-importing exports is supported through order/product upserts, but the contents and date coverage of the source export still determine which warehouse facts are updated.

## Repository map

```text
.
├── data/                     # Marketplace uploads and Indonesian location references
├── etl/                      # Extraction, transformation, and warehouse loading
├── ml_models/                # Forecast training code, notebooks, and saved models
├── scripts/                  # Reference-dimension seed scripts
├── static/                   # Brand assets and product images
├── templates/                # Login, upload, dashboard, and forecast pages
├── config.py                 # MySQL connection and table initialization
├── main.py                   # Flask routes and application logic
└── models.py                 # Warehouse and application data models
```
