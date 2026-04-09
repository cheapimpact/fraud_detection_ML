import pandas as pd
import os
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, 'Dataset_ML_Ready.xlsx')
    print(f"Memuat dataset dari {file_path}...")
    df = pd.read_excel(file_path)
    
    # Mengeluarkan fitur linguistik (X2) dari dataset
    cols_to_drop = [
        'FLAG POTENTIAL FRAUD', 
        'Negative_Tone', 'Positive_Tone', 'Net_Sentiment', 'Subjectivity_Ratio',
        'Positive_Sum', 'Negative_Sum', 'Total_Matched_Words', 'Total_Word'
    ]
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    X = df.drop(columns=cols_to_drop)
    y = df['FLAG POTENTIAL FRAUD']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("\nMelakukan teknik oversampling SMOTE (Fokus pada data FRAUD)...")
    smote = SMOTE(sampling_strategy='minority', random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    print("\nSedang melatih Model CatBoost dengan data hasil SMOTE...")
    cat_model = CatBoostClassifier(
        iterations=200,          # CatBoost cukup stabil dengan iterasi besar
        learning_rate=0.05,
        random_seed=42,
        verbose=0                # Menyembunyikan output loading agar rapi
    )
    cat_model.fit(X_train_smote, y_train_smote)

    print("Melakukan prediksi pada Data Uji...")
    y_pred = cat_model.predict(X_test)

    print("\n" + "="*50)
    print("HASIL EVALUASI MODEL: CATBOOST CLASSIFIER")
    print("="*50)
    print("\n[1] Akurasi Prediksi (Accuracy Score):", f"{accuracy_score(y_test, y_pred):.2%}")
    
    print("\n[2] Confusion Matrix:")
    cm = pd.DataFrame(confusion_matrix(y_test, y_pred), 
                      columns=["Prediksi Normal (0)", "Prediksi Fraud (1)"], 
                      index=["Aktual Normal (0)", "Aktual Fraud (1)"])
    print(cm)

    print("\n[3] Classification Report (Akurasi Rinci Tiap Kelas):")
    print(classification_report(y_test, y_pred, target_names=["Normal (0)", "Fraud (1)"]))

    try:
        importances = cat_model.get_feature_importance()
        feature_df = pd.DataFrame({'Fitur': X.columns, 'Tingkat Kepentingan': importances})
        feature_df = feature_df.sort_values(by='Tingkat Kepentingan', ascending=True)
        
        plt.figure(figsize=(10, 8))
        plt.barh(feature_df['Fitur'], feature_df['Tingkat Kepentingan'], color='orange')
        plt.title('Fitur Paling Berpengaruh (CatBoost)')
        plt.tight_layout()
        output_image = os.path.join(script_dir, 'Feature_Importance_Fraud_CatBoost.png')
        plt.savefig(output_image)
        print(f"\n[SUCCESS] Gambar Feature Importance CatBoost disimpan sebagai '{output_image}'")
    except Exception as e:
        print(f"\n[PERINGATAN] Gagal menyimpan grafik matplotlib. Error: {e}")

if __name__ == "__main__":
    main()
