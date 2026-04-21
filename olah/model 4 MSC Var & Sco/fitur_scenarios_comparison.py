import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import warnings

# Metrik Evaluasi
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, matthews_corrcoef
)

# Model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

def main():
    # Matikan peringatan berlebih
    warnings.filterwarnings("ignore")
    
    print("="*100)
    print("COMPARISON: 7 SKENARIO FITUR X 9 MODEL ALGORITMA")
    print("="*100)

    # 1. Memuat Dataset
    file_path = '/Users/calcifer/Documents/MDMA/olah/model 4/Dataset_ML_Ready_CLEAN_2.xlsx'
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        print(f"Gagal memuat dataset: {e}")
        return

    target = 'FLAG POTENTIAL FRAUD'
    y = df[target]

    # Definisi Komponen Fitur
    X_mscore = ['DSRI', 'GMI', 'AQI', 'SGI', 'DEPI', 'SGAI', 'LVGI', 'TATA', 'M_Score']
    X_ling   = ['Negative_Tone', 'Positive_Tone', 'Subjectivity_Ratio']
    X_vol    = ['VolatilitasD-30']

    # Definisi 7 Kombinasi Skenario
    scenarios = {
        "MScore": X_mscore,
        "Linguistik": X_ling,
        "Volatilitas": X_vol,
        "MScore + Linguistik": X_mscore + X_ling,
        "MScore + Volatilitas": X_mscore + X_vol,
        "Linguistik + Volatilitas": X_ling + X_vol,
        "Semua Fitur (Full)": X_mscore + X_ling + X_vol
    }

    # Definisi Model
    models = {
        "LogReg": LogisticRegression(max_iter=2000, random_state=42),
        "DecTree": DecisionTreeClassifier(random_state=42),
        "RandForest": RandomForestClassifier(random_state=42),
        "GradBoost": GradientBoostingClassifier(random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "KNN": KNeighborsClassifier(),
        "NaiveBayes": GaussianNB(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    results_list = []

    print("Sedang mengevaluasi 63 kombinasi (7 Skenario x 9 Model) dengan SMOTE...")

    for scenario_name, feature_cols in scenarios.items():
        X = df[feature_cols]
        
        # Split Data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # SMOTE
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

        for model_name, model in models.items():
            model.fit(X_train_res, y_train_res)
            
            y_pred = model.predict(X_test)
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
            else:
                y_prob = model.decision_function(X_test)
                
            auc_val = roc_auc_score(y_test, y_prob)
            ca_val = accuracy_score(y_test, y_pred)
            f1_val = f1_score(y_test, y_pred, zero_division=0)
            prec_val = precision_score(y_test, y_pred, zero_division=0)
            rec_val = recall_score(y_test, y_pred, zero_division=0)
            mcc_val = matthews_corrcoef(y_test, y_pred)
            
            results_list.append({
                "Model": model_name,
                "Skenario Fitur": scenario_name,
                "AUC": auc_val,
                "CA": ca_val,
                "F1": f1_val,
                "Prec": prec_val,
                "Recall": rec_val,
                "MCC": mcc_val
            })

    results_df = pd.DataFrame(results_list)

    # Menampilkan tabel per Model
    # Opsi: Mengelompokkan berdasarkan Model
    print("\n" + "="*100)
    print("HASIL PERBANDINGAN FITUR PER MODEL (Diurutkan berdasarkan AUC tertinggi tiap model)")
    print("="*100)
    
    for model_name in models.keys():
        subset = results_df[results_df["Model"] == model_name].copy()
        subset = subset.sort_values(by="AUC", ascending=False).reset_index(drop=True)
        
        print(f"\n---> Model: {model_name.upper()} <---")
        # Kolom 'Model' bisa di-drop di print agar tabel lebih muat
        print(subset.drop(columns=["Model"]).to_string(index=False, float_format=lambda x: "{:.4f}".format(x)))

    # Tambahan: Overall Top 10
    print("\n\n" + "="*100)
    print("TOP 10 KOMBINASI SECARA KESELURUHAN (Berdasarkan AUC)")
    print("="*100)
    top10 = results_df.sort_values(by="AUC", ascending=False).head(10).reset_index(drop=True)
    print(top10.to_string(index=False, float_format=lambda x: "{:.4f}".format(x)))

    # Export keseluruhan hasil ke Excel
    export_path = '/Users/calcifer/Documents/MDMA/olah/model 2/Hasil_Perbandingan_63_Skenario.xlsx'
    results_df_sorted = results_df.sort_values(by="AUC", ascending=False).reset_index(drop=True)
    results_df_sorted.to_excel(export_path, index=False)
    print(f"\n=> Seluruh hasil (63 baris) telah diekspor ke Excel: '{export_path}'")

if __name__ == "__main__":
    main()
