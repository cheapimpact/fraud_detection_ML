import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import shap
import matplotlib.pyplot as plt

def main():
    print("="*50)
    print("MEMUAI PROSES EKSPERIMEN DETEKSI FRAUD")
    print("="*50)

    # 1. Memuat Dataset
    # Menggunakan file .xlsx yang tersedia karena sebelumnya struktur mirip dengan yang dideskripsikan (511 baris)
    file_path = '/Users/calcifer/Documents/MDMA/olah/model 2/Dataset_ML_Ready_CLEAN_2.xlsx'
    try:
        df = pd.read_excel(file_path)
        print("=> Dataset berhasil dimuat.")
    except Exception as e:
        print(f"Gagal memuat dataset: {e}")
        return

    # Definisi Target
    target = 'FLAG POTENTIAL FRAUD'
    y = df[target]

    # Definisi Fitur berdasarkan skenario
    # X1: 8 Rasio Beneish
    X1_cols = ['DSRI', 'GMI', 'AQI', 'SGI', 'DEPI', 'SGAI', 'LVGI', 'TATA']
    
    # X2: Fitur Linguistik inSET
    X2_cols = ['Negative_Tone', 'Positive_Tone', 'Subjectivity_Ratio']
    
    # X3: Fitur Pasar (Asumsi nama kolom di dataset adalah 'VolatilitasD-30')
    X3_cols = ['VolatilitasD-30']

    # Skenario 1: Hanya X1
    # Skenario 2: X1 + X2
    # Skenario 3: X1 + X2 + X3
    
    # Dictionary untuk menyimpan daftar kolom per skenario
    scenarios = {
        "Skenario 1 (Baseline: 8 Rasio Beneish)": X1_cols,
        "Skenario 2 (Beneish + Linguistik)": X1_cols + X2_cols,
        "Skenario 3 (Full Integration: Semua Fitur)": X1_cols + X2_cols + X3_cols
    }

    # Model yang akan diuji
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    print("\n" + "="*50)
    print("UJI KOMPARASI 3 SKENARIO (ABLATION STUDY)")
    print("="*50)

    # Dictionary untuk menyimpan model XGBoost Skenario 3 untuk SHAP nanti
    xgb_s3_model = None
    X_train_s3_disp = None
    X_test_s3_disp = None

    # Iterasi untuk tiap skenario
    for scenario_name, feature_cols in scenarios.items():
        print(f"\n>>> Mengevaluasi: {scenario_name} <<<")
        
        # 2. Data Preprocessing (Pembagian 80% Training, 20% Testing)
        # Stratify digunakan agar rasio kelas imbalanced tetap sama di train dan test
        X = df[feature_cols]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 3. Terapkan SMOTE *hanya pada Training Set*
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
        
        # Simpan data testing Skenario 3 untuk uji SHAP
        if scenario_name == "Skenario 3 (Full Integration: Semua Fitur)":
            X_train_s3_disp = X_train_smote
            X_test_s3_disp = X_test

        # Latih dan evaluasi setiap algoritma
        for model_name, model in models.items():
            # Training model menggunakan data yang sudah diseimbangkan (SMOTE)
            model.fit(X_train_smote, y_train_smote)
            
            # Prediksi pada data testing
            y_pred = model.predict(X_test)
            
            # Menghitung probabilitas (untuk ROC-AUC)
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
            else:
                # Fallback untuk model yang tidak punya predict_proba (meski ketiganya punya)
                y_prob = model.decision_function(X_test)
            
            # Metrik evaluasi performa model
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_prob)
            
            print(f"[{model_name}] -> Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")
            
            # Khusus untuk mengambil model pemenang XGBoost pada Skenario 3
            if scenario_name == "Skenario 3 (Full Integration: Semua Fitur)" and model_name == "XGBoost":
                xgb_s3_model = model

    print("\n" + "="*50)
    print("ANALISIS SHAP (FEATURE IMPORTANCE)")
    print("Asumsi: XGBoost Skenario 3 adalah model terbaik")
    print("="*50)

    if xgb_s3_model is not None:
        # Analisis SHAP pada model XGBoost Skenario 3
        # Menggunakan TreeExplainer
        explainer = shap.TreeExplainer(xgb_s3_model)
        
        # Hitung SHAP values untuk data testing
        shap_values = explainer(X_test_s3_disp)
        
        # 1. Visualisasi SHAP Summary Plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test_s3_disp, show=False)
        plt.tight_layout()
        summary_path = '/Users/calcifer/Documents/MDMA/olah/model 2/shap_summary_xgb_s3.png'
        plt.savefig(summary_path, dpi=300)
        plt.close() # Tutup figure untuk membebaskan memory
        print(f"=> SHAP Summary Plot berhasil disimpan di:\n   {summary_path}")
        
        # 2. Visualisasi SHAP Waterfall Plot untuk DUA contoh, 1 Normal dan 1 Fraud (atau sesuaikan)
        # Mengambil 1 indeks prediksi lokal dari testing set untuk waterfall plot
        # Kita ambil indeks ke-0 pada test set sebagai contoh (bisa diganti sesuai kebutuhan)
        contoh_idx = 0
        plt.figure(figsize=(10, 6))
        # shap.plots.waterfall digunakan dari versi shap terbaru (berasal dari explainer(X_test))
        shap.plots.waterfall(shap_values[contoh_idx], show=False)
        plt.tight_layout()
        waterfall_path = '/Users/calcifer/Documents/MDMA/olah/model 2/shap_waterfall_xgb_s3.png'
        plt.savefig(waterfall_path, dpi=300)
        plt.close()
        print(f"=> SHAP Waterfall Plot (untuk index-{contoh_idx} dataset test) berhasil disimpan di:\n   {waterfall_path}")
        
    else:
        print("Model XGBoost Skenario 3 tidak ditemukan.")

if __name__ == "__main__":
    main()
