import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.inspection import permutation_importance
import warnings

# Algoritma List (sama seperti script perbandingan)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

def main():
    warnings.filterwarnings("ignore")
    print("="*80)
    print("Mengeksekusi Feature Importance dari Model Terbaik per Skenario")
    print("="*80)
    
    base_dir = '/Users/calcifer/Documents/MDMA/olah/model_2_MSC_var_Oke'
    olah_dir = os.path.join(base_dir, 'olah')
    matrix_file = os.path.join(olah_dir, 'Matriks_Terbaik_F1_Recall.xlsx')
    data_file = os.path.join(base_dir, 'Dataset_ML_Ready_CLEAN_2.xlsx')
    output_dir = os.path.join(olah_dir, 'Performance Test')
    
    # Buat direktori output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Membaca dataset utama: {data_file}")
    df_data = pd.read_excel(data_file)
    target = 'FLAG POTENTIAL FRAUD'
    y = df_data[target]
    
    print(f"Membaca matriks ranking: {matrix_file}")
    df_matrix = pd.read_excel(matrix_file)
    
    # Mencari Model Terbaik (Ranking 1 berdasarkan urutan excel yang disortir F1 & Recall) per 'Skenario Fitur'
    # Drop duplicates otomatis mempertahankan baris pertama (yang terbaik) untuk setiap kategori Skenario Fitur
    best_models_per_scenario = df_matrix.drop_duplicates(subset=['Skenario Fitur'], keep='first')
    
    # Save matriks filter
    best_models_path = os.path.join(output_dir, 'Top_Model_Per_Skenario.xlsx')
    best_models_per_scenario.to_excel(best_models_path, index=False)
    print(f"Model terbaik untuk 7 skenario berhasil difilter dan disimpan di {best_models_path}\n")

    # Definisi Komponen Fitur Asli
    X_mscore = ['DSRI', 'GMI', 'AQI', 'SGI', 'DEPI', 'SGAI', 'LVGI', 'TATA'] # Dataset model 2 MSC (8 Ratios)
    if 'M_Score' in df_data.columns and 'DSRI' not in df_data.columns:
        X_mscore = ['M_Score']
        
    X_ling   = ['Negative_Tone', 'Positive_Tone', 'Subjectivity_Ratio']
    X_vol    = ['VolatilitasD-30']
    
    scenarios_definition = {
        "MScore": X_mscore,
        "Linguistik": X_ling,
        "Volatilitas": X_vol,
        "MScore + Linguistik": X_mscore + X_ling,
        "MScore + Volatilitas": X_mscore + X_vol,
        "Linguistik + Volatilitas": X_ling + X_vol,
        "Semua Fitur (Full)": X_mscore + X_ling + X_vol
    }

    # Helper untuk memanggil model
    def get_model(name):
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
        return models[name]

    # Iterasi untuk tiap skenario untuk memetakan Feature Importancenya
    all_importances_data = []
    for _, row in best_models_per_scenario.iterrows():
        scenario = row['Skenario Fitur']
        model_name = row['Model']
        
        feature_cols = scenarios_definition[scenario]
        X = df_data[feature_cols]
        
        # Train-Test Split (sama dengan cara uji performa)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # SMOTE
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        
        # Eksekusi Pelatihan
        model = get_model(model_name)
        model.fit(X_train_res, y_train_res)
        
        # Ekstrak Feature Importance
        # Kita menggunakan pendekatan hibrida: Jika punya attribut `.feature_importances_`, kita pakai, jika tidak, pakai Permutation Importance.
        importances = None
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            # Gunakan Permutation Importance (Agnostic / Works with SVM, LogReg, NaiveBayes, KNN)
            result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
            importances = result.importances_mean
            
        # Urutkan Fitur berdasarkan Importance-nya
        indices = np.argsort(importances)
        sorted_features = [feature_cols[i] for i in indices]
        sorted_importances = importances[indices]
        
        # Simpan nilai kedalam list
        for feat, imp in zip(sorted_features, sorted_importances):
            all_importances_data.append({
                'Skenario Fitur': scenario,
                'Model': model_name,
                'Fitur': feat,
                'Importance': imp
            })
        
        # Visualisasi Bar Plot
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(sorted_importances)), sorted_importances, color='teal', alpha=0.8)
        plt.yticks(range(len(sorted_importances)), sorted_features)
        plt.xlabel("Feature Importance (Relative/Permutation)")
        plt.title(f"Feature Importance\nSkenario: [{scenario}] | Model Terbaik: [{model_name}]")
        plt.tight_layout()
        
        # Nama file rapih
        safe_scenario = scenario.replace(" ", "_").replace("+", "plus").replace("(", "").replace(")", "")
        img_name = f"Feature_Importance_{safe_scenario}_{model_name}.png"
        img_path = os.path.join(output_dir, img_name)
        
        plt.savefig(img_path, dpi=300)
        plt.close()
        
        print(f"--> Skenario: {scenario} ({model_name}) | Disimpan: {img_name}")
        
    df_importances = pd.DataFrame(all_importances_data)
    df_importances = df_importances.sort_values(by=['Skenario Fitur', 'Importance'], ascending=[True, False])
    importances_path = os.path.join(output_dir, 'Feature_Importances_Values.xlsx')
    df_importances.to_excel(importances_path, index=False)
    
    print(f"\nNilai Feature Importance berhasil disimpan dalam file excel: {importances_path}")
    print("PROSES SELESAI! Semua Feature Importance berhasil terbuat dalam folder 'Performance Test'.")

if __name__ == '__main__':
    main()
