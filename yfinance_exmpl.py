import yfinance as yf
import pandas as pd

def main():
    ticker_symbol = "BBCA.JK" # Contoh menggunakan saham BCA dari IHSG
    print(f"=== Mengekstraksi Data untuk Ticker: {ticker_symbol} ===\n")
    
    ticker = yf.Ticker(ticker_symbol)
    
    # 1. Informasi Umum Perusahaan (.info)
    # Ini mengembalikan dictionary dengan buanyak sekali key: sektor, marketCap, forwardPE, dividendYield, dsb.
    print("1. Informasi Umum (Info):")
    info = ticker.info
    print(f"  Nama Perusahaan : {info.get('longName', 'N/A')}")
    print(f"  Sektor          : {info.get('sector', 'N/A')}")
    print(f"  Industri        : {info.get('industry', 'N/A')}")
    print(f"  Karyawan        : {info.get('fullTimeEmployees', 'N/A')}")
    print(f"  Website         : {info.get('website', 'N/A')}")
    print(f"  Market Cap      : {info.get('marketCap', 'N/A')}")
    print(f"  Book Value      : {info.get('bookValue', 'N/A')}")
    print(f"  Ringkasan       : {info.get('longBusinessSummary', 'N/A')[:150]}...\n")
    
    # 2. Histori Harga Saham (.history)
    # Anda bisa mengatur periode seperti '1d', '5d', '1mo', '3mo', '6mo', '1y', 'ytd', 'max'
    print("2. Histori Harga (History) - 5 Hari Terakhir:")
    history = ticker.history(period="5d")
    print(history[['Open', 'High', 'Low', 'Close', 'Volume']].to_string())
    print("\n")
    
    # 3. Aksi Korporasi (.actions / .dividends / .splits)
    print("3. Aksi Korporasi (Dividen & Splits):")
    actions = ticker.actions
    if not actions.empty:
        print(actions.tail().to_string())
    else:
        print("  Tidak ada data aksi korporasi.")
    print("\n")
        
    # 4. Laporan Keuangan Tahunan (.financials, .balance_sheet, .cashflow)
    # Untuk versi kuartalan bisa pakai .quarterly_financials, .quarterly_balance_sheet, dsb.
    print("4. Laporan Laba Rugi (Income Statement):")
    income_stmt = ticker.financials
    if not income_stmt.empty:
        print(income_stmt.head(5).to_string()) # Menampilkan 5 baris pertama
    else:
        print("  Data laporan laba rugi tidak tersedia.")
    print("\n")

    print("5. Neraca Keuangan (Balance Sheet):")
    balance_sheet = ticker.balance_sheet
    if not balance_sheet.empty:
        print(balance_sheet.head(5).to_string())
    else:
        print("  Data neraca tidak tersedia.")
    print("\n")
        
    print("6. Arus Kas (Cash Flow):")
    cashflow = ticker.cashflow
    if not cashflow.empty:
        print(cashflow.head(5).to_string())
    else:
        print("  Data arus kas tidak tersedia.")
    print("\n")
        
    # 5. Pemegang Saham (.major_holders / .institutional_holders / .mutualfund_holders)
    print("7. Pemegang Saham Institusi Terbesar:")
    institutional_holders = ticker.institutional_holders
    if institutional_holders is not None and not institutional_holders.empty:
        print(institutional_holders.head(5).to_string())
    else:
        print("  Data pemegang saham tidak tersedia.")
    print("\n")

    # 6. Analis Konsensus dan Rekomendasi (.recommendations)
    print("8. Rekomendasi Analis Konsensus:")
    recommendations = ticker.recommendations
    if recommendations is not None and not recommendations.empty:
        print(recommendations.head(5).to_string())
    else:
        print("  Data rekomendasi tidak tersedia untuk saham ini.")
    print("\n")
        
    # 7. Kalender Rilis EPS / Earnings Dates (.earnings_dates)
    # Catatan: Saham internasional (US) biasanya akurat, untuk saham Indonesia (IHSG) mungkin terbatas atau kosong.
    print("9. Kalender Pendapatan (Earnings Dates):")
    earnings_dates = ticker.earnings_dates
    if earnings_dates is not None and not earnings_dates.empty:
        # Kita ambil 5 data terbaru yang tidak kosong
        print(earnings_dates.dropna(how='all').head(5).to_string())
    else:
        print("  Data kalender pendapatan tidak tersedia untuk saham ini.")

    print("\n=== Mengekspor Data ke Excel ===")
    excel_file = f"{ticker_symbol}_Data.xlsx"
    try:
        def clean_tz(df):
            if hasattr(df.index, 'tz') and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            for col in df.select_dtypes(include=['datetimetz']).columns:
                df[col] = df[col].dt.tz_localize(None)
            return df
            
        with pd.ExcelWriter(excel_file) as writer:
            # 1. Info (Ubah dictionary ke DataFrame, lalu transpose agar mudah dibaca)
            df_info = pd.DataFrame([info]).T
            df_info.columns = ['Value']
            clean_tz(df_info).to_excel(writer, sheet_name='Info')
            
            # 2. Histori Harga
            clean_tz(history).to_excel(writer, sheet_name='History')
            
            # 3. Aksi Korporasi
            if not actions.empty:
                clean_tz(actions).to_excel(writer, sheet_name='Actions')
                
            # 4. Laporan Keuangan
            if not income_stmt.empty:
                clean_tz(income_stmt).to_excel(writer, sheet_name='Income Statement')
            if not balance_sheet.empty:
                clean_tz(balance_sheet).to_excel(writer, sheet_name='Balance Sheet')
            if not cashflow.empty:
                clean_tz(cashflow).to_excel(writer, sheet_name='Cash Flow')
                
            # 5. Institusi
            if institutional_holders is not None and not institutional_holders.empty:
                clean_tz(institutional_holders).to_excel(writer, sheet_name='Inst Holders')
                
            # 6. Rekomendasi
            if recommendations is not None and not recommendations.empty:
                clean_tz(recommendations).to_excel(writer, sheet_name='Recommendations')
                
            # 7. Kalender
            if earnings_dates is not None and not earnings_dates.empty:
                clean_tz(earnings_dates).to_excel(writer, sheet_name='Earnings Dates')
                
        print(f"Berhasil mengekspor data ke file: {excel_file}")
    except Exception as e:
        print(f"Gagal mengekspor data ke Excel: {e}")
        print("Pastikan library 'openpyxl' sudah diinstall (pip install openpyxl)")

if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    main()
