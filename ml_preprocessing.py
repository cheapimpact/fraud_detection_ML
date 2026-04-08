import pandas as pd
import numpy as np
import os

def preprocess_for_ml(input_path, output_path):
    print(f"Membaca data dari {input_path}...")
    df = pd.read_excel(input_path, sheet_name='winsor')
    
    # 1. Menangani Nilai Kosong (NaN) pada Volatilitas dengan angka 0
    volatility_cols = ['Volatilitas 30 Hari SEBELUM', 'Volatilitas 30 Hari Setelah']
    for col in volatility_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
            
    # Mengisi NaN pada target perhitungan kata (jika ada file yang sama sekali kosong)
    word_cols = ['Positive_Sum', 'Negative_Sum', 'Total_Matched_Words', 'Total_Word']
    for col in word_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    print("Menghitung Variabel Sentimen / Textual...")
    
    # Menghindari Division by Zero (pembagian dengan nol)
    # Jika Total_Word = 0, pembagiannya akan menghasilkan 0 (kita replace pd.nan dengan 0)
    total_word_safe = np.where(df['Total_Word'] == 0, np.nan, df['Total_Word'])
    total_matched_safe = np.where(df['Total_Matched_Words'] == 0, np.nan, df['Total_Matched_Words'])
    
    # Fitur 1: Proporsi Sentimen Negatif (Negative Tone)
    df['Negative_Tone'] = df['Negative_Sum'] / total_word_safe
    
    # Fitur 2: Proporsi Sentimen Positif (Positive Tone)
    df['Positive_Tone'] = df['Positive_Sum'] / total_word_safe
    
    # Fitur 3: Polaritas Sentimen Bersih (Net Sentiment)
    df['Net_Sentiment'] = (df['Positive_Sum'] - df['Negative_Sum']) / total_matched_safe
    
    # Fitur 4: Kepadatan Sentimen (Subjectivity Ratio)
    df['Subjectivity_Ratio'] = df['Total_Matched_Words'] / total_word_safe
    
    # Ubah hasil NaN (akibat pembagian dengan 0 karena dokumennya kosong/error ekstrak teks) menjadi angka 0
    ratio_cols = ['Negative_Tone', 'Positive_Tone', 'Net_Sentiment', 'Subjectivity_Ratio']
    df[ratio_cols] = df[ratio_cols].fillna(0)
    
    print("Menghapus kolom yang dilarang atau tidak berguna untuk model ML...")
    # Drop kolom identitas dan yang tidak dibutuhkan
    cols_to_drop = ['Unnamed: 20', 'Unnamed: 21', 'Report Date', 'Ticker', 'Year']
    existing_cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    df.drop(columns=existing_cols_to_drop, inplace=True)
    
    print("Melakukan Target Encoding...")
    # Encoding Prediction (Label/Y) menjadi Kategorikal Numerik (0 dan 1)
    if 'Prediction' in df.columns:
        df['Prediction'] = df['Prediction'].map({'Non-Manipulator': 0, 'Potential Manipulator': 1})
    
    # Pastikan data yang tersisa tidak ada lagi missing value
    # Kita isi sisa missing value (bila ada, misal DSRI yang error saat kalkulasi) dengan Median (nilai tengah)
    df.fillna(df.median(numeric_only=True), inplace=True)

    print(f"Menyimpan dataset akhir (Ready for ML) ke {output_path}...")
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='ML_Data', index=False)
        
    print("Preprocessing selesai!")
    print("\n[INFO]: Dataset ini memiliki fitur rasio sentimen baru yang sangat siap untuk dimasukkan ke XGBoost atau Algoritma ML Lainnya.")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(base_dir, 'olah', 'Combined_Full_Outer_Join_Winsorized.xlsx')
    output_file = os.path.join(base_dir, 'olah', 'Dataset_ML_Ready.xlsx')
    
    preprocess_for_ml(input_file, output_file)
