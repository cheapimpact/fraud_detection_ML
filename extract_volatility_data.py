import os
import json
import pandas as pd
import yfinance as yf
import numpy as np
import time

def extract_horizontal_volatility():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(base_dir, 'bei_process')
    
    # 1. Scan folder bei_process
    ticker_years = []
    if os.path.exists(folder_path):
        for item in os.listdir(folder_path):
            if '_' in item and os.path.isdir(os.path.join(folder_path, item)):
                parts = item.split('_', 1)
                if len(parts) == 2:
                    ticker, year = parts
                    ticker_years.append((ticker, year))
    
    if not ticker_years:
        print(f"Folder 'bei_process' tidak berisi folder dengan format TICKER_TAHUN.")
        return

    print(f"Ditemukan {len(ticker_years)} folder emiten di bei_process.")
    unique_tickers = list(set([t[0] for t in ticker_years]))

    # 2. Get Report Date from JSON
    release_dates_map = {}
    for json_file in ['BEI2023.json', 'BEI2024.json']:
        json_path = os.path.join(base_dir, json_file)
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    for item in data.get('Results', []):
                        emiten = item.get('KodeEmiten')
                        year = item.get('Report_Year')
                        file_modified = item.get('File_Modified')
                        if emiten and year and file_modified:
                            date_only = file_modified.split('T')[0]
                            release_dates_map[(emiten, str(year))] = date_only
                except Exception as e:
                    print(f"Error parse {json_file}: {e}")

    # 3. Process Tickers
    all_rows = []
    
    for ticker in unique_tickers:
        print(f"-> Memproses ticker {ticker}...")
        try:
            # Unduh rentang data yang cukup lebar
            tkr = yf.Ticker(f"{ticker}.JK")
            df = tkr.history(start='2023-01-01', end='2025-12-31')
            if df.empty:
                print(f"  [!] Data kosong untuk {ticker}.JK")
                continue
            
            # Hapus timezone dan pastikan index unik agar tidak error saat indexing
            df.index = df.index.tz_localize(None)
            df = df[~df.index.duplicated(keep='first')]
            
            # Cari baris data (tahun) yang relevan untuk ticker ini
            years_for_ticker = [y for t, y in ticker_years if t == ticker]
            
            for year in years_for_ticker:
                report_date_str = release_dates_map.get((ticker, year))
                if not report_date_str:
                    print(f"  [!] Tanggal laporan (File_Modified) tidak ditemukan untuk {ticker} tahun {year}")
                    continue
                
                report_date = pd.to_datetime(report_date_str)
                available_dates = df.index[df.index >= report_date]
                if available_dates.empty:
                    print(f"  [!] Tidak ada data trading pada atau setelah tanggal laporan {report_date_str} untuk {ticker}.")
                    continue
                
                actual_event_date = available_dates[0]
                loc = df.index.get_loc(actual_event_date)
                
                # Setup horizontal row format
                row_data = {
                    'ticker': ticker,
                    'year': year,
                    'report date': report_date_str,
                    'actual trading date': actual_event_date.strftime('%Y-%m-%d')
                }
                
                # Mengambil nilai Close dari offset -90 hingga +30
                for offset in range(-90, 31):
                    target_loc = loc + offset
                    if 0 <= target_loc < len(df):
                        val = df.iloc[target_loc]['Close']
                    else:
                        val = np.nan
                    row_data[f"{offset}"] = val
                    
                all_rows.append(row_data)
                
        except Exception as e:
            print(f"  [!] Kesalahan saat memproses {ticker}: {e}")
        
        # Jeda koneksi api
        time.sleep(0.5)

    # 4. Save to Excel
    if all_rows:
        df_out = pd.DataFrame(all_rows)
        # Susun urutan kolom (ticker, year, report date, actual trading date, -90 ... 30)
        base_cols = ['ticker', 'year', 'report date', 'actual trading date']
        day_cols = [str(i) for i in range(-90, 31)]
        
        # Buang kolom extra jika ternyata ada salah casting, pastikan kolom berurutan
        cols_to_use = base_cols + day_cols
        
        # Sort values
        df_out = df_out[cols_to_use].sort_values(by=['ticker', 'year'])
        
        excel_path = os.path.join(base_dir, 'Data_Volatilitas_Horizontal.xlsx')
        try:
            df_out.to_excel(excel_path, index=False)
            print(f"\n[OK] Sukses! Data disimpan ke {excel_path}")
        except ModuleNotFoundError:
            # Fallback ke csv bila openpyxl tidak terpasang
            print("\nTerjadi kegagalan saat menulis ke excel: modul 'openpyxl' tidak tersedia.")
            csv_path = os.path.join(base_dir, 'Data_Volatilitas_Horizontal.csv')
            df_out.to_csv(csv_path, index=False)
            print(f"\n[OK] Data sebagai gantinya disimpan dalam CSV: {csv_path}")
    else:
        print("\nTidak ada data yang berhasil diekstrak.")

if __name__ == '__main__':
    extract_horizontal_volatility()
