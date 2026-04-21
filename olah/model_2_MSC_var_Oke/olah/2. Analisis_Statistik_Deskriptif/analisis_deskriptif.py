import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("="*80)
    print("ANALISIS STATISTIK DESKRIPTIF")
    print("="*80)
    
    base_dir = '/Users/calcifer/Documents/MDMA/olah/model 2 MSC var  Oke'
    file_path = os.path.join(base_dir, 'Dataset_ML_Ready_CLEAN_2.xlsx')
    output_dir = os.path.join(base_dir, 'olah', '2. Analisis_Statistik_Deskriptif')
    
    # Buat folder output jika belum ada
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Membaca dataset utama: {file_path}")
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        print(f"Gagal membaca file excel: {e}")
        return

    # Daftar kolom fitur numerik (Menyesuaikan dengan Model 2 MSC var Oke)
    numeric_features = [
        'DSRI', 'GMI', 'AQI', 'SGI', 'DEPI', 'SGAI', 'LVGI', 'TATA', 
        'VolatilitasD-30', 'Negative_Tone', 'Positive_Tone', 'Subjectivity_Ratio'
    ]
    
    target = 'FLAG POTENTIAL FRAUD'
    
    # Periksa ketersediaan kolom
    missing_cols = [col for col in numeric_features if col not in df.columns]
    if missing_cols:
        print(f"Peringatan: Kolom {missing_cols} tidak ditemukan di dataset. Menggunakan kolom yang ada saja.")
        numeric_features = [col for col in numeric_features if col in df.columns]
        
    print("\nMenghitung statistik deskriptif umum...")
    
    # 1. Statistik Deskriptif Keseluruhan (N, Min, Max, Mean, Std)
    desc_all = df[numeric_features].describe().T
    desc_all = desc_all[['count', 'min', 'max', 'mean', 'std']]
    desc_all.columns = ['N', 'Minimum', 'Maksimum', 'Rata-Rata (Mean)', 'Std. Deviation']
    
    # 2. Statistik Deskriptif Dibedakan berdasarkan Grup (Fraud 1 vs Non-Fraud 0)
    print("Menghitung statistik deskriptif berdasarkan grup FRAUD...")
    grouped = df.groupby(target)[numeric_features]
    desc_grup = grouped.agg(['count', 'min', 'max', 'mean', 'std']).T
    
    # Memformat index multi-level agar rapi di Excel
    desc_grup.index.names = ['Fitur', 'Statistik']
    
    # Tampilkan di Terminal sebagian
    print("\n[Preview] Rata-rata Nilai per Grup:")
    print(df.groupby(target)[numeric_features].mean().T)

    # Simpan ke Excel
    output_excel = os.path.join(output_dir, 'Hasil_Statistik_Deskriptif.xlsx')
    with pd.ExcelWriter(output_excel) as writer:
        desc_all.to_excel(writer, sheet_name='Deskriptif_Keseluruhan')
        desc_grup.to_excel(writer, sheet_name='Deskriptif_Menurut_Fraud')
        df.groupby(target)[numeric_features].mean().T.to_excel(writer, sheet_name='Perbandingan_Rata_Rata')

    print(f"\n=> Tabel Statistik Deskriptif berhasil disimpan ke: {output_excel}")

    # ==========================
    # VISUALISASI GRAFIK
    # ==========================
    print("\nMembuat visualisasi grafik (Boxplots & Barplots)...")
    sns.set_theme(style="whitegrid")
    
    # Plot 1: Boxplot Perbandingan Distribusi untuk masing-masing Fitur
    # Sangat baik untuk Bab 4 Skripsi untuk menunjukkan perbedaan outliers/median
    
    plot_dir = os.path.join(output_dir, 'Boxplots')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        
    for feature in numeric_features:
        plt.figure(figsize=(8, 6))
        
        # Mapping nama kelas agar rapi
        plot_df = df.copy()
        plot_df['Status'] = plot_df[target].map({0: 'Non-Fraud (0)', 1: 'Fraud (1)'})
        
        # Tambahan: hilangkan nilai ekstrem yang bisa merusak skala chart visual
        # Kita batasi Q1-1.5IQR s.d Q3+1.5IQR pada Y axis limit jika mau, tapi boxplot Seaborn sudah bisa handle outlier dots.
        
        sns.boxplot(x='Status', y=feature, data=plot_df, palette=['#2ca02c', '#d62728'])
        plt.title(f'Distribusi {feature} pada Perusahaan Fraud vs Non-Fraud')
        plt.xlabel('Status Perusahaan')
        plt.ylabel(f'Nilai {feature}')
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'Boxplot_{feature}.png'), dpi=300)
        plt.close()
        
    # Plot 2: Bar Plot Perbandingan Rata-rata
    mean_df = df.groupby(target)[numeric_features].mean().reset_index()
    mean_df['Status'] = mean_df[target].map({0: 'Non-Fraud', 1: 'Fraud'})
    mean_df_melted = mean_df.melt(id_vars='Status', value_vars=numeric_features, var_name='Fitur', value_name='Rata_Rata')
    
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(x='Rata_Rata', y='Fitur', hue='Status', data=mean_df_melted, palette=['#2ca02c', '#d62728'])
    plt.title('Perbandingan Rata-Rata Nilai Fitur (Fraud vs Non-Fraud)')
    plt.xlabel('Nilai Rata-rata')
    plt.ylabel('Fitur Analisis')
    plt.legend(title='Kelas')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Barplot_Perbandingan_RataRata.png'), dpi=300)
    plt.close()
    
    print(f"=> Seluruh grafik visual (1 Barplot Rata-rata + {len(numeric_features)} Boxplots) telah disimpan di: {output_dir}")

if __name__ == "__main__":
    main()
