import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from config import SessionLocal
from models import PaymentMethodDimension

payment_methods = {
    'payment_method_name' : ['COD', 'PAYLATER', 'E-WALLET', 'BANK_TRANSFER', 'CREDIT_CARD', 'STORE_BALANCE', 'OFFLINE_AGENT', 'RETAIL_OUTLET'],
    'payment_method_category' : ['MANUAL', 'LEVERAGE', 'LIQUID', 'LIQUID', 'LEVERAGE', 'LIQUID', 'MANUAL', 'MANUAL']
}

def seed_payment_method_dimension():
    db = SessionLocal()

    payment_methods_df = pd.DataFrame(payment_methods)

    try:
        existing_payment_methods = db.query(PaymentMethodDimension.payment_method_name, PaymentMethodDimension.payment_method_category).all()
        df_existing = pd.DataFrame(existing_payment_methods, columns=['payment_method_name', 'payment_method_category'])
        
        if not df_existing.empty:
            comparison_df = payment_methods_df.merge(df_existing, on=['payment_method_name', 'payment_method_category'], how='left', indicator=True)
            df_to_insert = comparison_df[comparison_df['_merge'] == 'left_only'].drop(columns=['_merge'])
        else:
            df_to_insert = payment_methods_df
            
        if df_to_insert.empty:
            print("No new payment methods found. Database is up to date.")
            return
        
        records = df_to_insert.to_dict(orient='records')
        db.bulk_insert_mappings(PaymentMethodDimension, records)
        db.commit()
        print("Payment Method Dimension seeded successfully!")
    except Exception as e:
        db.rollback()
        print(f"Error seeding Payment Method Dimension: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    seed_payment_method_dimension()