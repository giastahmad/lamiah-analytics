import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from config import SessionLocal
from models import PlatformDimension

def seed_platform_dimension():
    platforms = {
        'platform_name': ['Shopee', 'Tokopedia', 'TikTok']
    }
    platforms_df = pd.DataFrame(platforms)
    db = SessionLocal()
    try:
        existing_platforms = db.query(PlatformDimension.platform_name).all()
        df_existing = pd.DataFrame(existing_platforms, columns=['platform_name'])
        
        if not df_existing.empty:
            comparison_df = platforms_df.merge(df_existing, on='platform_name', how='left', indicator=True)
            df_to_insert = comparison_df[comparison_df['_merge'] == 'left_only'].drop(columns=['_merge'])
        else:
            df_to_insert = platforms_df
            
        if df_to_insert.empty:
            print("No new platforms found. Database is up to date.")
            return
        
        records = df_to_insert.to_dict(orient='records')
        db.bulk_insert_mappings(PlatformDimension, records)
        db.commit()
        print("Platform Dimension seeded successfully!")
    except Exception as e:
        db.rollback()
        print(f"Error seeding Platform Dimension: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    seed_platform_dimension()