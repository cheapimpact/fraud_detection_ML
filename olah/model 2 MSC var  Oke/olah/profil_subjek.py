import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time

def main():
    print("="*80)
    print("ANALISIS DESKRIPSI SUBYEK PENELITIAN")
    print("="*80)
    
    file_path = '/Users/calcifer/Documents/MDMA/olah/model 2 MSC var  Oke/olah/1. Profil Subjek/Combined_Full_Outer_Join_Winsorized that has train.xlsx'
    output_dir = '/Users/calcifer/Documents/MDMA/olah/model 2 MSC var  Oke/olah/Profil_Subjek'
    
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Membaca dataset utama...")
    df = pd.read_excel(file_path)
    
    # Deteksi jumlah perusahaan unik
    unique_tickers = df['Ticker'].unique()
    print(f"Ditemukan {len(unique_tickers)} perusahaan unik dalam dataset.")
    
    # Ambil sektor & nama dari yfinance
    print("Sedang mengambil data Sektor dari Yahoo Finance...")
    company_data = []
    
    for counter, ticker in enumerate(unique_tickers):
        ticker_jk = f"{ticker}.JK"
        print(f"[{counter+1}/{len(unique_tickers)}] Mengunduh profil '{ticker_jk}'...", end="")
        try:
            info = yf.Ticker(ticker_jk).info
            sector = info.get('sector', 'Unknown')
            name = info.get('longName', ticker)
            company_data.append({'Ticker': ticker, 'Nama_Perusahaan': name, 'Sektor': sector})
            print(f" OK ({sector})")
        except Exception as e:
            company_data.append({'Ticker': ticker, 'Nama_Perusahaan': ticker, 'Sektor': 'Unknown'})
            print(" GAGAL")
        time.sleep(0.1) # delay dikit agar tidak kena block

    df_companies = pd.DataFrame(company_data)
    
    # Gabung kembali dengan df utama
    print("\nMengkompilasikan Data Gabungan...")
    df_merged = pd.merge(df, df_companies, on='Ticker', how='left')
    
    # 1. Agregasi Jumlah Perusahaan berdasarkan Sektor (Setiap Ticker hanya dihitung 1x)
    sector_counts = df_companies['Sektor'].value_counts()
    
    # 2. Agregasi Fraud vs Non-Fraud berdasarkan Sektor
    # Ini menghitung berdasarkan Data Tahunan (berapa Insiden per sektor) 
    # Atau menghitung apakah perusahaan pernah Fraud sama sekali?
    # Berdasarkan konteks penelitian (tahun-perusahaan), lebih logis merangkum observasi Fraud vs Non Fraud per tahun.
    fraud_distribution = df_merged.groupby(['Sektor', 'FLAG POTENTIAL FRAUD']).size().unstack(fill_value=0)
    # Jika Skenarionya FLAG: 0 (Non Fraud), 1 (Fraud)
    
    # Ekspor ke Excel
    output_excel = os.path.join(output_dir, 'Data_Profil_Subjek_Lengkap.xlsx')
    with pd.ExcelWriter(output_excel) as writer:
        df_merged.to_excel(writer, sheet_name='Raw_Data_Sektor', index=False)
        df_companies.to_excel(writer, sheet_name='Daftar_Perusahaan_Unik', index=False)
        sector_counts.to_frame(name='Jumlah Perusahaan').to_excel(writer, sheet_name='Rekap_Sektor')
        fraud_distribution.to_excel(writer, sheet_name='Fraud_vs_NonFraud_Sektor')
    
    print(f"Sukses meyimpan rangkuman Excel ke: {output_excel}")

    # ==========================
    # VISUALISASI GRAFIK
    # ==========================
    sns.set_theme(style="whitegrid")
    
    # Grafik 1: Distribusi Jumlah Perusahaan per Sektor
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x=sector_counts.values, y=sector_counts.index, palette="Blues_r")
    plt.title('Jumlah Perusahaan Berdasarkan Sektor Industri')
    plt.xlabel('Jumlah Perusahaan Unik')
    plt.ylabel('Sektor Industri')
    for p in ax.patches:
        ax.annotate(f"{int(p.get_width())}", 
                    (p.get_width() + 0.1, p.get_y() + p.get_height()/2.), 
                    va='center')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '1_Grafik_Sebaran_Perusahaan_Per_Sektor.png'), dpi=300)
    plt.close()
    
    # Grafik 2: Fraud vs Non-Fraud Per Sektor (Grouped Bar Chart)
    # Persiapkan format untuk seaborn
    df_plot_fraud = fraud_distribution.reset_index().melt(id_vars='Sektor', var_name='Tipe', value_name='Frekuensi')
    df_plot_fraud['Tipe'] = df_plot_fraud['Tipe'].map({0: 'Non-Fraud', 1: 'Fraud'})
    
    # Urutkan berdasarkan total sampel terbanyak
    order_sectors = fraud_distribution.sum(axis=1).sort_values(ascending=False).index
    
    plt.figure(figsize=(14, 8))
    ax2 = sns.barplot(
        data=df_plot_fraud, x='Frekuensi', y='Sektor', hue='Tipe', 
        order=order_sectors, palette=['#2ca02c', '#d62728']
    )
    plt.title('Perbandingan Kasus Fraud vs Non-Fraud Berdasarkan Sektor Industri')
    plt.xlabel('Jumlah Sampel (Tahun-Perusahaan)')
    plt.ylabel('Sektor Industri')
    plt.legend(title='Status')
    
    for p in ax2.patches:
        width = p.get_width()
        if not np.isnan(width) and width > 0:
            ax2.annotate(f"{int(width)}", 
                        (width + 0.5, p.get_y() + p.get_height()/2.), 
                        va='center')
            
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2_Grafik_Fraud_vs_NonFraud_Per_Sektor.png'), dpi=300)
    plt.close()
    
    print("\nProses Kompilasi Selesai! Semua grafik dan excel berhasil diterbitkan.")

if __name__ == "__main__":
    main()
