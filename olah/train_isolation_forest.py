import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, 'Dataset_ML_Ready.xlsx')
    print(f"Memuat dataset dari {file_path}...")
    df = pd.read_excel(file_path)
    
    # Isolation forest operates primarily on features.
    # Mengeluarkan fitur linguistik (X2) dari dataset
    cols_to_drop = [
        'FLAG POTENTIAL FRAUD', 
        'Negative_Tone', 'Positive_Tone', 'Net_Sentiment', 'Subjectivity_Ratio',
        'Positive_Sum', 'Negative_Sum', 'Total_Matched_Words', 'Total_Word'
    ]
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    X = df.drop(columns=cols_to_drop)
    y = df['FLAG POTENTIAL FRAUD']
    
    # Membagi dataset (Tidak di-SMOTE karena Anomaly Detection butuh natural distribution)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nSedang melatih Model Isolation Forest (Tanpa SMOTE - khusus mendeteksi perilaku aneh/menyimpang)...")
    # contamination kita set misalnya 15% (Ini bisa diubah tergantung asumsi rasio fraud, default ~0.15)
    iso_model = IsolationForest(
        n_estimators=100, 
        contamination=0.15,
        random_state=42
    )
    # Fit di X_train
    iso_model.fit(X_train)

    print("Melakukan prediksi pada Data Uji...")
    # Prediksi mengembalikan -1 untuk anomali (fraud) dan 1 untuk inlier (normal)
    y_pred_raw = iso_model.predict(X_test)
    
    # Konversi hasil: 1 (inlier) -> 0 (Normal). -1 (outlier/anomali) -> 1 (Fraud)
    y_pred = [1 if x == -1 else 0 for x in y_pred_raw]

    print("\n" + "="*50)
    print("HASIL EVALUASI MODEL: ISOLATION FOREST")
    print("="*50)
    print("\n[1] Akurasi Prediksi (Accuracy Score):", f"{accuracy_score(y_test, y_pred):.2%}")
    
    print("\n[2] Confusion Matrix:")
    cm = pd.DataFrame(confusion_matrix(y_test, y_pred), 
                      columns=["Prediksi Normal (0)", "Prediksi Fraud (1)"], 
                      index=["Aktual Normal (0)", "Aktual Fraud (1)"])
    print(cm)

    print("\n[3] Classification Report (Mengukur akurasi tangkapan anomali):")
    print(classification_report(y_test, y_pred, target_names=["Normal (0)", "Fraud (1)"]))

if __name__ == "__main__":
    main()
