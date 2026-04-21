import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def run_feature_importance():
    # Setup path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(current_dir, '..', 'Dataset_ML_Ready_CLEAN.xlsx')
    
    # Load dataset
    df = pd.read_excel(dataset_path)
    
    features = [
        'DSRI', 'GMI', 'AQI', 'SGI', 'DEPI', 'SGAI', 'LVGI', 'TATA', 
        'Negative_Tone', 'Positive_Tone', 'Subjectivity_Ratio',     
        'VolatilitasD-30'                                           
    ]
    
    X = df[features]
    y = df['FLAG POTENTIAL FRAUD']
    
    # Split & SMOTE
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    
    # Inisialisasi dan Train XGBoost sebagai model juara (hipotesis umum skripsi tabular)
    print("Melatih model XGBoost...")
    model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train_sm, y_train_sm)
    
    # --- 1. GLOBAL FEATURE IMPORTANCE ---
    print("Membuat grafik Global Feature Importance...")
    plt.figure(figsize=(10, 6))
    
    # Kita menggunakan feature_importances_ dari model
    importances = model.feature_importances_
    indices = np.argsort(importances)
    plt.title('Global Feature Importance (XGBoost)')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices])
    plt.xlabel('Tingkat Kepentingan Relatif (Relative Importance)')
    plt.tight_layout()
    
    fi_out_path = os.path.join(current_dir, 'global_feature_importance.png')
    plt.savefig(fi_out_path, dpi=300)
    plt.close()
    print(f"Bagan Global Feature Importance tersimpan di {fi_out_path}")
    
    # --- 2. SHAP VALUES ---
    print("Menghitung SHAP Values dan membuat beeswarm summary plot...")
    explainer = shap.TreeExplainer(model)
    # Gunakan X_test untuk melihat kontribusi di data tak terlihat
    shap_values = explainer.shap_values(X_test)
    
    plt.figure(figsize=(10, 6))
    # Buat summary plot (beeswarm)
    shap.summary_plot(shap_values, X_test, show=False)
    
    shap_out_path = os.path.join(current_dir, 'shap_summary_plot.png')
    plt.gcf().set_size_inches(10, 6)
    plt.tight_layout()
    plt.savefig(shap_out_path, dpi=300)
    plt.close()
    print(f"SHAP Summary Plot tersimpan di {shap_out_path}")

if __name__ == "__main__":
    run_feature_importance()
