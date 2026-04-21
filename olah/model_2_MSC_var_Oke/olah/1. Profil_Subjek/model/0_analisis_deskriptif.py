import pandas as pd
import numpy as np
import os

def run_deskriptif():
    print("Memulai Analisis Deskriptif...")
    # Setup path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(current_dir, '..', 'Dataset_ML_Ready_CLEAN.xlsx')
    
    # Load dataset
    df = pd.read_excel(dataset_path)
    
    features = [
        'DSRI', 'GMI', 'AQI', 'SGI', 'DEPI', 'SGAI', 'LVGI', 'TATA', 
        'VolatilitasD-30', 
        'Negative_Tone', 'Positive_Tone', 'Subjectivity_Ratio'
    ]
    
    # Filter features plus target
    df_desc = df[features + ['FLAG POTENTIAL FRAUD']]
    
    # Group by
    desc_stats = df_desc.groupby('FLAG POTENTIAL FRAUD').agg(['mean', 'median', 'std']).T
    
    print("\n--- Analisis Deskriptif (Perbandingan Mean, Median, Std) ---")
    
    results = []
    
    for feature in features:
        mean_nonfraud = df_desc[df_desc['FLAG POTENTIAL FRAUD'] == 0][feature].mean()
        mean_fraud = df_desc[df_desc['FLAG POTENTIAL FRAUD'] == 1][feature].mean()
        
        median_nonfraud = df_desc[df_desc['FLAG POTENTIAL FRAUD'] == 0][feature].median()
        median_fraud = df_desc[df_desc['FLAG POTENTIAL FRAUD'] == 1][feature].median()
        
        std_nonfraud = df_desc[df_desc['FLAG POTENTIAL FRAUD'] == 0][feature].std()
        std_fraud = df_desc[df_desc['FLAG POTENTIAL FRAUD'] == 1][feature].std()
        
        results.append({
            'Fitur': feature,
            'Non-Fraud (Mean)': mean_nonfraud,
            'Fraud (Mean)': mean_fraud,
            'Non-Fraud (Median)': median_nonfraud,
            'Fraud (Median)': median_fraud,
            'Non-Fraud (Std)': std_nonfraud,
            'Fraud (Std)': std_fraud
        })
        
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))
    
    out_path = os.path.join(current_dir, 'analisis_deskriptif.csv')
    df_results.to_csv(out_path, index=False)
    print(f"\nTabel Deskriptif tersimpan di: {out_path}\n")

if __name__ == "__main__":
    run_deskriptif()
