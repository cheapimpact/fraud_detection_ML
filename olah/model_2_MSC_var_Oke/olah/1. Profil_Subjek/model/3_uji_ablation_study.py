import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from scipy.stats import chi2

def mcnemar_test(y_true, y_pred1, y_pred2):
    """
    Melakukan McNemar's Test.
    y_pred1: Prediksi Model 2 (Baseline Jurnal)
    y_pred2: Prediksi Model 3 (Inovasi Skripsi)
    """
    b = 0  # y_pred1 salah, y_pred2 benar
    c = 0  # y_pred1 benar, y_pred2 salah
    
    for yt, yp1, yp2 in zip(y_true, y_pred1, y_pred2):
        if yp1 != yt and yp2 == yt:
            b += 1
        elif yp1 == yt and yp2 != yt:
            c += 1
            
    # Hitung stat dan p-value
    if b + c == 0:
        return 0.0, 1.0  # Tidak ada perbedaan prediksi
        
    # Menggunakan koreksi kontinuitas Edwards ((|b-c|-1)**2 / (b+c))
    stat = ((abs(b - c) - 1.0)**2) / (b + c)
    p_value = chi2.sf(stat, df=1)
    return stat, p_value

def run_ablation_study():
    # Setup path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(current_dir, '..', 'Dataset_ML_Ready_CLEAN.xlsx')
    
    # Load dataset
    df = pd.read_excel(dataset_path)
    
    # Define features
    X1_cols = ['DSRI', 'GMI', 'AQI', 'SGI', 'DEPI', 'SGAI', 'LVGI', 'TATA']
    X2_cols = ['Negative_Tone', 'Positive_Tone', 'Subjectivity_Ratio']
    X3_cols = ['VolatilitasD-30']
    
    all_features = X1_cols + X2_cols + X3_cols
    X = df[all_features]
    y = df['FLAG POTENTIAL FRAUD']
    
    X_train_full, X_test_full, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Skenario Definisi (Sesuai modifikasi Tabel skripsi)
    scenarios = {
        'Model 1 (Baseline: Keuangan)': X1_cols,
        'Model 2 (Jurnal: Keu+Teks)': X1_cols + X2_cols,
        'Model 3 (Inovasi: Keu+Teks+Pasar)': X1_cols + X2_cols + X3_cols
    }
    
    model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    
    results = []
    predictions_dict = {}
    
    print("\nMemulai Uji Performa Klasifikasi (Ablation Study)...")
    
    for sc_name, sc_cols in scenarios.items():
        print(f"Melatih {sc_name}...")
        X_train_sc = X_train_full[sc_cols].copy()
        X_test_sc = X_test_full[sc_cols].copy()
        
        # SMOTE
        smote = SMOTE(random_state=42)
        X_train_sm, y_train_sm = smote.fit_resample(X_train_sc, y_train)
        
        # Train & Predict
        model.fit(X_train_sm, y_train_sm)
        y_pred = model.predict(X_test_sc)
        y_pred_proba = model.predict_proba(X_test_sc)[:, 1]
        
        # Simpan array prediksi untuk McNemar's Test
        predictions_dict[sc_name] = y_pred
        
        # Metrics
        f1 = f1_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        results.append({
            'Model': sc_name,
            'Fitur': len(sc_cols),
            'Recall': rec,
            'Precision': prec,
            'F1-Score': f1,
            'AUC': auc
        })
        
    df_results = pd.DataFrame(results)
    print("\n--- HASIL UJI PERFORMA KLASIFIKASI ---")
    print(df_results.to_string(index=False))
    
    csv_out = os.path.join(current_dir, 'hasil_ablation.csv')
    df_results.to_csv(csv_out, index=False)
    
    # UJI SIGNIFIKANSI (MCNEMAR'S TEST)
    print("\n--- UJI SIGNIFIKANSI MCNEMAR'S TEST ---")
    print("Membandingkan Model 2 (Replikasi Jurnal) vs Model 3 (Inovasi Skripsi)")
    
    stat, p_value = mcnemar_test(
        y_true=y_test.values,
        y_pred1=predictions_dict['Model 2 (Jurnal: Keu+Teks)'],
        y_pred2=predictions_dict['Model 3 (Inovasi: Keu+Teks+Pasar)']
    )
    
    print(f"Chi-square Statistic : {stat:.4f}")
    print(f"P-Value              : {p_value:.5f}")
    
    if p_value < 0.05:
        print("Kesimpulan           : P-value < 0.05. Terdapat PENINGKATAN PERFORMA YANG SIGNIFIKAN secara statistik pada Model 3.")
    else:
        print("Kesimpulan           : P-value >= 0.05. Peningkatan performa tidak signifikan secara statistik pada alpha=5%.")
    
    # Plotting F1-Score & AUC
    plt.figure(figsize=(10, 6))
    x_indexes = np.arange(len(df_results['Model']))
    width = 0.35
    
    plt.bar(x_indexes - width/2, df_results['F1-Score'], width=width, label='F1-Score', color='skyblue')
    plt.bar(x_indexes + width/2, df_results['AUC'], width=width, label='AUC', color='coral')
    
    plt.ylim(0, 1.0)
    plt.title('Uji Skenario - Perbandingan F1-Score dan AUC')
    plt.ylabel('Skor')
    plt.xticks(x_indexes, ['Model 1 (Baseline)', 'Model 2 (Jurnal)', 'Model 3 (Inovasi)'], rotation=0)
    plt.legend()
    
    for i in range(len(df_results)):
        plt.text(x_indexes[i] - width/2, df_results['F1-Score'][i] + 0.01, f"{df_results['F1-Score'][i]:.3f}", ha='center', va='bottom', fontweight='bold', fontsize=9)
        plt.text(x_indexes[i] + width/2, df_results['AUC'][i] + 0.01, f"{df_results['AUC'][i]:.3f}", ha='center', va='bottom', fontweight='bold', fontsize=9)
        
    plt.tight_layout()
    plot_out = os.path.join(current_dir, 'ablation_plot.png')
    plt.savefig(plot_out, dpi=300)
    plt.close()

if __name__ == "__main__":
    run_ablation_study()
