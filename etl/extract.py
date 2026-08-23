import os

import openpyxl
import pandas as pd


def extract_data(file_path, platform):
    _, file_extension = os.path.splitext(file_path)
    platform = platform.lower()
    
    if platform in ['tokopedia', 'tiktok']:
        if file_extension == '.xlsx':
            wb = openpyxl.load_workbook(file_path)
            ws = wb.active
            data = ws.values
            headers = next(data)
            next(data)
            df = pd.DataFrame(data, columns=headers)
        elif file_extension == '.csv':
            df = pd.read_csv(file_path)
        else:
            raise ValueError(f"Ekstensi {file_extension} tidak didukung untuk Tokopedia/TikTok.")
   
    elif platform == 'shopee':
        if file_extension == '.xlsx':
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Ekstensi {file_extension} tidak didukung untuk Shopee.")
        
    else:
        raise ValueError("Platform tidak dikenali. Pilih 'shopee', 'tokopedia', atau 'tiktok'.")
        
    return df