import yfinance as yf
import pandas as pd
import numpy as np
import time
import os
import json

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(base_dir, 'bei_process')
    tickers = []
    
    if os.path.exists(folder_path):
        unique_tickers = set()
        for item in os.listdir(folder_path):
            if '_' in item and os.path.isdir(os.path.join(folder_path, item)):
                ticker = item.split('_')[0]
                unique_tickers.add(ticker + '.JK')
        tickers = sorted(list(unique_tickers))
    
    if not tickers:
        print(f"Tidak ada ticker yang ditemukan di folder {folder_path}.")
        return
        
    print(f"Mulai ekstraksi data untuk {len(tickers)} saham dari IHSG berdasarkan folder {folder_path}...")
    
    all_raw_data = []
    all_results = []
    
    # Tanggal proxy/sementara apabila data earning dates dari JSON kosong.
    # Format: (Report_Year, Proxy_Date)
    proxy_report_dates = [('2023', '2024-03-31'), ('2024', '2025-03-31')]
    
    # Membaca data tanggal rilis dari JSON
    release_dates_map = {}
    for json_file in ['BEI2023.json', 'BEI2024.json']:
        json_path = os.path.join(base_dir, json_file)
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                try:
                    json_data = json.load(f)
                    for item in json_data.get('Results', []):
                        emiten = item.get('KodeEmiten')
                        year = item.get('Report_Year')
                        file_modified = item.get('File_Modified')
                        if emiten and year and file_modified:
                            # Ambil bagian tanggal saja (YYYY-MM-DD)
                            date_only = file_modified.split('T')[0]
                            release_dates_map[(emiten, str(year))] = date_only
                except json.JSONDecodeError as e:
                    print(f"Error reading JSON {json_file}: {e}")
    
    for ticker in tickers:
        print(f"-> Memproses {ticker}...")
        try:
            ticker_obj = yf.Ticker(ticker)
            # 1. Mendownload histori harga (diperlebar s.d akhir 2025 agar mencakup laporan tahun 2024 yang terbit di 2025)
            data = ticker_obj.history(start='2023-01-01', end='2025-12-31')
            
            if data.empty:
                print(f"   [!] Data kosong untuk {ticker}. Dilewati.")
                continue
                
            # Menghilangkan timezone
            data.index = data.index.tz_localize(None)
            
            # 2. Menghitung return harian
            data['Return'] = data['Close'].pct_change()
            
            # Mengambil tanggal rilis dari file JSON BEI (File_Modified)
            base_ticker = ticker.split('.')[0]
            reports_to_process = []
            for year in ['2023', '2024']:
                if (base_ticker, year) in release_dates_map:
                    reports_to_process.append((year, release_dates_map[(base_ticker, year)]))
            
            # Jika tidak ada data spesifik dari JSON, gunakan proxy
            if not reports_to_process:
                reports_to_process = proxy_report_dates
            
            for year, report_date in reports_to_process:
                report_date_dt = pd.to_datetime(report_date)
                
                # Memastikan tanggal laporan ada di data
                available_dates = data.index[data.index >= report_date_dt]
                if len(available_dates) == 0:
                    continue
                report_date_actual = available_dates[0]
                loc = data.index.get_loc(report_date_actual)
                
                # 3. Hitung volatilitas 90 hari sebelum rilis laporan (T-95 sampai T-6)
                start_vol_loc = max(0, loc - 95)
                end_vol_loc = max(0, loc - 5)
                
                if end_vol_loc > start_vol_loc:
                    window_data = data.iloc[start_vol_loc:end_vol_loc]
                    volatility_90d = window_data['Return'].std() * np.sqrt(252)
                    expected_return = window_data['Return'].mean()
                else:
                    volatility_90d = np.nan
                    expected_return = 0
                    
                # 4. Hitung nilai CAR (-5 sampai +5 hari)
                start_car_loc = max(0, loc - 5)
                end_car_loc = min(len(data) - 1, loc + 5)
                
                car_window = data.iloc[start_car_loc:end_car_loc + 1].copy()
                car_window['Abnormal_Return'] = car_window['Return'] - expected_return
                car = car_window['Abnormal_Return'].sum()
                
                # Simpan slice raw data: dari awal tahun laporan hingga laporan keluar + 5 hari
                start_raw_date = pd.to_datetime(f"{year}-01-01")
                end_raw_date = data.index[end_car_loc]
                try:
                    raw_slice = data.loc[start_raw_date:end_raw_date].copy()
                    raw_slice['Ticker'] = ticker
                    raw_slice['Report_Year'] = year
                    all_raw_data.append(raw_slice)
                except Exception as ex:
                    pass
                
                all_results.append({
                    'Ticker': ticker,
                    'Report_Year': year,
                    'Event_Target_Date': report_date_dt.date(),
                    'Event_Actual_Date': report_date_actual.date(),
                    'Volatility_90d_Annualized': volatility_90d,
                    'CAR_minus5_to_plus5': car
                })
        except Exception as e:
            print(f"   [!] Error pada {ticker}: {e}")
            
        # Delay sedikit agar tidak terkena limit Yahoo Finance
        time.sleep(0.5)

    # 5. Gabungkan dan simpan semua file raw data
    if all_raw_data:
        combined_raw = pd.concat(all_raw_data)
        # Susun ulang kolom agar Ticker & Report_Year ada di depan
        cols = ['Ticker', 'Report_Year'] + [c for c in combined_raw.columns if c not in ['Ticker', 'Report_Year']]
        combined_raw = combined_raw[cols]
        raw_filename = os.path.join(base_dir, 'Dataset_Volatilitas_Raw.csv')
        combined_raw.to_csv(raw_filename)
        print(f"\n[OK] Raw data untuk semua saham disimpan di {raw_filename}")
    else:
        print("\n[!] Tidak ada raw data yang berhasil didownload.")

    # 6. Simpan hasil perhitungan (CAR & Volatilitas)
    if all_results:
        results_df = pd.DataFrame(all_results)
        final_filename = os.path.join(base_dir, 'Dataset_Volatilitas.csv')
        results_df.to_csv(final_filename, index=False)
        print(f"[OK] Data hasil perhitungan volatilitas dan CAR disimpan di {final_filename}")
    else:
        print("[!] Tidak ada hasil perhitungan yang bisa disimpan.")

if __name__ == "__main__":
    main()
