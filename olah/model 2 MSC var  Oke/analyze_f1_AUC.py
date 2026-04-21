import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    base_dir = '/Users/calcifer/Documents/MDMA/olah/model 2 MSC var  Oke'
    input_file = os.path.join(base_dir, 'Hasil_Perbandingan_63_Skenario.xlsx')
    output_dir = os.path.join(base_dir, 'olah')
    
    # Buat folder 'olah' jika belum ada
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Membaca data dari: {input_file}")
    df = pd.read_excel(input_file)
    
    # Filter 10 besar berdasarkan F1 dan AUC (F1 lebih diutamakan, disusul AUC)
    # Anda juga bisa melakukan agregasi atau Euclidean ke perfect score (F1=1, AUC=1)
    df['F1_AUC_Avg'] = (df['F1'] + df['AUC']) / 2
    
    # 1. Sort Data
    df_sorted = df.sort_values(by=['F1', 'AUC'], ascending=[False, False])
    
    # 2. Output Matriks ke Excel
    output_excel = os.path.join(output_dir, 'Matriks_Terbaik_F1_AUC.xlsx')
    df_sorted.to_excel(output_excel, index=False)
    print(f"Matriks lengkap tersimpan di: {output_excel}")
    
    # 3. Output Top 10 untuk F1 dan AUC
    top_10 = df_sorted.head(10).reset_index(drop=True)
    print("\n" + "="*80)
    print("TOP 10 PERFORMA TERBAIK BERDASARKAN F1 DAN AUC")
    print("="*80)
    print(top_10[['Model', 'Skenario Fitur', 'F1', 'AUC', 'AUC']].to_string(index=False))
    
    # 4. Bikin Scatter Plot Performa
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=df, 
        x='AUC', 
        y='F1', 
        hue='Model', 
        style='Skenario Fitur', 
        s=120, 
        alpha=0.8
    )
    plt.title('Performance Matrix: F1 Score vs AUC')
    plt.xlabel('AUC')
    plt.ylabel('F1 Score')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Geser legend keluar chart
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'Scatter_F1_vs_AUC.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"\nGrafik Scatter Plot tersimpan di: {plot_path}")
    print("Selesai diproses semuanya!")

if __name__ == '__main__':
    main()
