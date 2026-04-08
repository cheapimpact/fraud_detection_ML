import pandas as pd
import os
import numpy as np

def categorize_columns(columns):
    """
    Mengelompokkan kolom berdasarkan panduan khusus:
    1. WAJIB Di-Winsorize
    2. HARAM Di-Winsorize
    """
    wajib_winsorize = []
    haram_winsorize = []
    
    # Keyword list
    wajib_keywords = ['DSRI', 'GMI', 'AQI', 'SGI', 'DEPI', 'SGAI', 'LVGI', 'TATA', 
                      'M_Score', 'M-Score', 'Volatilitas+30', 'Volatilitas-30', 'CAR']
    
    haram_keywords = ['Sanksi', 'SP2', 'Prediction', 'Kode', 'Ticker', 'Year', 'Tahun', 'ID', 'Perusahaan', 'Kategori']
    
    for col in columns:
        col_str = str(col).strip()
        is_haram = False
        
        # Cek apakah kolom ini haram
        for hk in haram_keywords:
            if hk.lower() in col_str.lower():
                haram_winsorize.append(col)
                is_haram = True
                break
                
        if not is_haram:
            # Cek apakah kolom ini wajib
            is_wajib = False
            for wk in wajib_keywords:
                if wk.lower() in col_str.lower():
                    wajib_winsorize.append(col)
                    is_wajib = True
                    break
            
            # Opsional: Jika tidak haram, and bukan kategori wajib spesifik (Mungkin mau di winsorize juga)
            # Namun sesuai aturan ketat, yang mandatory adalah yang wajib tersebut. 
            # Jika user ingin semua numerik selain yang haram di-winsorize, bisa ditambahkan.
            # Pendekatan aman: masukkan ke list 'bisa_diwinsorize' jika numerik. 
            pass

    return wajib_winsorize, haram_winsorize

def winsorize_dataframe(df, lower_limit=0.01, upper_limit=0.01, columns_to_winsorize=None):
    """
    Winsorize numeric columns in a pandas DataFrame.
    """
    df_winsorized = df.copy()
    
    if columns_to_winsorize is None:
        return df_winsorized
        
    for col in columns_to_winsorize:
        if col in df_winsorized.columns and pd.api.types.is_numeric_dtype(df_winsorized[col]) and not df_winsorized[col].isnull().all():
            lower_quantile = df_winsorized[col].quantile(lower_limit)
            upper_quantile = df_winsorized[col].quantile(1 - upper_limit)
            
            # Terapkan winsorize dengan fungsi clip
            df_winsorized[col] = df_winsorized[col].clip(lower=lower_quantile, upper=upper_quantile)
            
    return df_winsorized

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, 'Combined_Full_Outer_Join.xlsx')
    output_path = os.path.join(base_dir, 'Combined_Full_Outer_Join_Winsorized.xlsx')
    
    # NOTE: sheet 'winsor' sepertinya belum ada di file masukan. Jika Anda memilki sheet dengan nama lain, ubah di sini.
    # Secara default karena diinstruksikan nama sheet 'winsor', kita akan asumsikan sheet itu ADA atau kita beri pesan jelas.
    sheet_name = 'winsor'
    
    print(f"Membaca file: {file_path}")
    print(f"Menggunakan Sheet: '{sheet_name}'")
    
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        
        print("Dimensi data asli:", df.shape)
        
        # Kategorisasi Kolom
        wajib_cols, haram_cols = categorize_columns(df.columns.tolist())
        
        print("\n=== ATURAN WINSORIZE DITERAPKAN ===")
        print("1. Daftar Kolom WAJIB Di-Winsorize:")
        for w in wajib_cols:
            print(f"   - {w}")
            
        print("\n2. Daftar Kolom HARAM Di-Winsorize:")
        for h in haram_cols:
            print(f"   - {h}")
            
        # Tentukan kolom akhir yang di-winsorize: harus numerik, masuk daftar wajib, dan bukan haram.
        numeric_cols_in_df = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Kita gabungkan: WAJIB pasti di-winsorize. Kolom numerik sisanya yang BUKAN HARAM kita anggap OPSIONAL tapi kita ikutkan.
        # User bilang: "panduan tegas mengenai kolom apa saja yang WAJIB, OPSIONAL, dan HARAM"
        # Kita masukkan kolom opsional yang numerik (tidak wajib, tapi tidak haram)
        opsional_cols = [c for c in numeric_cols_in_df if c not in wajib_cols and c not in haram_cols]
        
        if opsional_cols:
            print("\n3. Daftar Kolom OPSIONAL (Ikut Di-Winsorize karena bukan HARAM):")
            for o in opsional_cols:
                print(f"   - {o}")
        
        final_cols_to_winsor = []
        for col in numeric_cols_in_df:
            if col not in haram_cols:
                final_cols_to_winsor.append(col)

        print(f"\nMenjalankan winsorize pada {len(final_cols_to_winsor)} kolom (1% terbawah dan 99% teratas)...")
        df_winsorized = winsorize_dataframe(df, lower_limit=0.01, upper_limit=0.01, columns_to_winsorize=final_cols_to_winsor)
        
        print(f"\nMenyimpan hasil ke file Excel baru: {output_path}")
        with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
            df_winsorized.to_excel(writer, sheet_name=sheet_name, index=False)
            
        print(f"Selesai! Data berhasil di-winsorize dengan panduan tegas yang diberlakukan.")
        
    except ValueError as val_err:
        print(f"Error membaca sheet '{sheet_name}'. PESAN: {val_err}")
        print("Pastikan Anda sudah memiliki sheet bernama 'winsor' di dalam Excel tersebut!")
    except FileNotFoundError:
        print(f"Error: File '{file_path}' tidak ditemukan.")
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")

if __name__ == "__main__":
    main()
