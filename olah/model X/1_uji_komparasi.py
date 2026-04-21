import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score

def run_komparasi():
    # Setup path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(current_dir, '..', 'Dataset_ML_Ready_CLEAN.xlsx')
    
    # Load dataset
    df = pd.read_excel(dataset_path)
    
    # Mendifinisikan fitur (menggunakan 8 Rasio Beneish + Linguistik + Pasar sesuai narasi Uji 3)
    features = [
        'DSRI', 'GMI', 'AQI', 'SGI', 'DEPI', 'SGAI', 'LVGI', 'TATA', # X1: Rasio
        'Negative_Tone', 'Positive_Tone', 'Subjectivity_Ratio',     # X2: Linguistik
        'VolatilitasD-30'                                           # X3: Pasar
    ]
    
    X = df[features]
    y = df['FLAG POTENTIAL FRAUD']
    
    # Split Data (Stratified agar distribusi fraud tetap sama di Train dan Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    print(f"Distribusi Train Asli: Fraud={sum(y_train==1)}, Normal={sum(y_train==0)}")
    
    # Mengaplikasikan SMOTE SAJA pada data Train
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    print(f"Distribusi Train setelah SMOTE: Fraud={sum(y_train_smote==1)}, Normal={sum(y_train_smote==0)}")
    
    # Inisialisasi Model
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
        'LightGBM': LGBMClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42)
    }
    
    results = []
    
    # Training dan Evaluasi
    for name, model in models.items():
        # Train
        model.fit(X_train_smote, y_train_smote)
        
        # Predict di data Test asli
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Evaluasi (Hanya metriks yang diminta: Recall, Precision, F1-Score, ROC-AUC)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        results.append({
            'Model': name,
            'Recall': recall,
            'Precision': precision,
            'F1-Score': f1,
            'ROC-AUC': roc_auc
        })
        
    df_results = pd.DataFrame(results)
    
    # Tampilkan hasil
    print("\n--- HASIL UJI KOMPARASI ALGORITMA ---")
    print(df_results.sort_values(by='F1-Score', ascending=False).to_string(index=False))
    
    # Simpan hasil
    out_path = os.path.join(current_dir, 'hasil_komparasi.csv')
    df_results.to_csv(out_path, index=False)
    print(f"\nHasil tersimpan di: {out_path}")

if __name__ == "__main__":
    run_komparasi()
