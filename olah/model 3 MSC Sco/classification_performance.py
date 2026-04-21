import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Metrik Evaluasi
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, matthews_corrcoef
)

# Algoritma / Model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

def main():
    print("="*80)
    print("CLASSIFICATION PERFORMANCE: MEMBANDINGKAN BEBERAPA ALGORITMA")
    print("="*80)

    # 1. Memuat Dataset
    file_path = '/Users/calcifer/Documents/MDMA/olah/model 3/Dataset_ML_Ready_CLEAN_3.xlsx'
    df = pd.read_excel(file_path)

    # Definisi Target dan Fitur Skenario 3 (Semua Fitur Terintegrasi)
    target = 'FLAG POTENTIAL FRAUD'
    X_cols = [
        'M_Score',                                                   # X1
        'Negative_Tone', 'Positive_Tone', 'Subjectivity_Ratio',      # X2
        'VolatilitasD-30'                                            # X3
    ]
    
    X = df[X_cols]
    y = df[target]

    # Pembagian 80% Training, 20% Testing (Stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Terapkan SMOTE pada Training Set
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Definisi berbagai model algoritma
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "SVM (SVC)": SVC(probability=True, random_state=42), # butuh probability=True untuk AUC
        "KNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    # Untuk menampung hasil ke dalam tabel/DataFrame
    results_list = []

    print("Sedang melatih dan mengevaluasi model berdasarkan skenario 3 (X1+X2+X3) dengan SMOTE...\n")

    for model_name, model in models.items():
        # Latih model
        model.fit(X_train_res, y_train_res)
        
        # Prediksi Output
        y_pred = model.predict(X_test)
        
        # Hitung Probabilitas Positif untuk AUC
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.decision_function(X_test)
            
        # Hitung Metrik
        auc_val = roc_auc_score(y_test, y_prob)
        ca_val = accuracy_score(y_test, y_pred)
        f1_val = f1_score(y_test, y_pred, zero_division=0)
        prec_val = precision_score(y_test, y_pred, zero_division=0)
        rec_val = recall_score(y_test, y_pred, zero_division=0)
        mcc_val = matthews_corrcoef(y_test, y_pred)
        
        # Simpan ke list
        results_list.append({
            "Model": model_name,
            "AUC": auc_val,
            "CA": ca_val,
            "F1": f1_val,
            "Prec": prec_val,
            "Recall": rec_val,
            "MCC": mcc_val
        })

    # Jadikan Dataframe agar rapi
    results_df = pd.DataFrame(results_list)
    
    # Sortir berdasarkan F1-Score atau AUC tertinggi
    results_df = results_df.sort_values(by="AUC", ascending=False).reset_index(drop=True)

    print("="*80)
    print("HASIL KLASIFIKASI PERFORMA EVALUASI (Diurutkan berdasarkan AUC Tertinggi)")
    print("="*80)
    
    # Cetak DataFrame dgn format rapi
    print(results_df.to_string(index=False, float_format=lambda x: "{:.4f}".format(x)))
    
    print("\n* Catatan Metrik:")
    print("  AUC    = Area Under the ROC Curve")
    print("  CA     = Classification Accuracy (Akurasi Keseluruhan)")
    print("  F1     = F1-Score (Pengukuran seimbang Presisi & Recall)")
    print("  Prec   = Precision (Seberapa tepat prediksi positif fraud)")
    print("  Recall = Recall (Berapa banyak fraud sebenarnya yang terdeteksi)")
    print("  MCC    = Matthews Correlation Coefficient (-1 s.d 1, bagus untuk imbalanced)")

    # Export keseluruhan hasil ke Excel
    export_path = '/Users/calcifer/Documents/MDMA/olah/model 3/Hasil_Performa_Skenario_3_Full.xlsx'
    results_df.to_excel(export_path, index=False)
    print(f"\n=> Hasil perbandingan algoritma ini telah diekspor ke Excel: '{export_path}'")

if __name__ == "__main__":
    main()
