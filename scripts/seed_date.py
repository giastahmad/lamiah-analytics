import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from config import SessionLocal
from models import DateDimension
from hijri_converter import Gregorian

def is_ramadhan(date):
    try:
        hijri_date = Gregorian(date.year, date.month, date.day).to_hijri()
        return hijri_date.month == 9
    except Exception as e:
        return False

def seed_date_dimension():
    start_date = '2022-01-01'
    end_date = '2042-12-31'
    date_range = pd.date_range(start=start_date, end=end_date)
    
    df_date = pd.DataFrame({'date_actual': date_range})
    
    df_date['date_id'] = df_date['date_actual'].dt.strftime('%Y%m%d').astype(int)
    df_date['date'] = df_date['date_actual'].dt.date
    df_date['days_name'] = df_date['date_actual'].dt.day_name()
    df_date['month'] = df_date['date_actual'].dt.month_name()
    df_date['year'] = df_date['date_actual'].dt.year
    df_date['is_weekend'] = df_date['days_name'].isin(['Saturday', 'Sunday'])
    df_date['is_twin_date'] = df_date['date_actual'].dt.day == df_date['date_actual'].dt.month
    df_date['is_payday'] = (df_date['date_actual'].dt.day >= 23) & (df_date['date_actual'].dt.day <= 25)
    df_date['is_ramadhan'] = df_date['date_actual'].apply(is_ramadhan)
    
    db = SessionLocal()
    try:
        existing_dates = db.query(DateDimension).count()
        if existing_dates != 0:
            print("Date Dimension already seeded, skip the process.")
            return

        records = df_date.drop(columns=['date_actual']).to_dict(orient='records')
        
        db.bulk_insert_mappings(DateDimension, records)
        db.commit()
        print("Date Dimension seeded successfully!")
    except Exception as e:
        db.rollback()
        print(f"Error seeding Date Dimension: {e}")
    finally:
        db.close()


if __name__ == "__main__":
    seed_date_dimension()