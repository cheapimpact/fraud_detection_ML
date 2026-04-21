import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import warnings

def main():
    warnings.filterwarnings("ignore")
    print("="*80)
    print("SHAP FEATURE IMPORTANCE: MODEL 4")
    print("="*80)
    
    file_path = '/Users/calcifer/Documents/MDMA/olah/model 4/Dataset_ML_Ready_CLEAN_4.xlsx'
    print(f"Memuat data dari: {file_path}")
    
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        print(f"Gagal memuat file: {e}")
        return

    target = 'FLAG POTENTIAL FRAUD'
    y = df[target]

    # Model 4 Features: 8 Beneish Ratios + M_Score + Linguistic + Volatility
    X_cols = [
        'DSRI', 'GMI', 'AQI', 'SGI', 'DEPI', 'SGAI', 'LVGI', 'TATA', 'M_Score',
        'Negative_Tone', 'Positive_Tone', 'Subjectivity_Ratio',
        'VolatilitasD-30'
    ]
    
    X = df[X_cols]
    
    # Stratified Split 80-20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # SMOTE only on training data
    print("Menerapkan SMOTE pada training set...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Train XGBoost
    print("Melatih model XGBoost...")
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train_res, y_train_res)

    # SHAP Explainer
    print("Mengekstrak Feature Importance menggunakan SHAP...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # 1. Plot Bar: Rata-rata absolute SHAP Value (Global Importance)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.tight_layout()
    bar_path = '/Users/calcifer/Documents/MDMA/olah/model 4/shap_bar_model4.png'
    plt.savefig(bar_path, dpi=300)
    plt.close()
    
    # 2. Plot Summary (Titik)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    summary_path = '/Users/calcifer/Documents/MDMA/olah/model 4/shap_summary_model4.png'
    plt.savefig(summary_path, dpi=300)
    plt.close()
    
    print("\nSelesai! Plot SHAP telah disimpan di:")
    print(f"- {bar_path}")
    print(f"- {summary_path}")

if __name__ == '__main__':
    main()
