import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # 1. Memuat Dataset
    # Memastikan path selalu merujuk ke folder yang sama dengan script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, 'Dataset_ML_Ready.xlsx')
    print(f"Memuat dataset dari {file_path}...")
    df = pd.read_excel(file_path)
    
    # 2. Memisahkan Fitur (X) dan Target (y)
    # Memilih 'FLAG POTENTIAL FRAUD' sebagai label target
    X = df.drop(columns=['FLAG POTENTIAL FRAUD'])
    y = df['FLAG POTENTIAL FRAUD']
    
    # 3. Pembagian Data Pelatihan (Train) dan Pengujian (Test)
    # test_size=0.2 artinya 80% data untuk melatih model, 20% untuk menguji kinerja model.
    # stratify=y sangat penting untuk memastikan rasio 86:14 kelas fraud tetap terjaga pada saat dipecah.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("\n--- Distribusi Data Latih (Train) Sebelum SMOTE ---")
    print(y_train.value_counts(normalize=True).apply(lambda x: f"{x:.2%}"))
    
    # 3.5 Menerapkan SMOTE untuk Over-sampling Kelas Minoritas (Fraud)
    print("\nMelakukan teknik oversampling SMOTE pada data latih...")
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    print("\n--- Distribusi Data Latih (Train) Setelah SMOTE ---")
    print(y_train_smote.value_counts(normalize=True).apply(lambda x: f"{x:.2%}"))
    print(f"Total baris data latih sekarang: {len(X_train_smote)}")

    # 4. Membangun Model Machine Learning
    # Algoritma Random Forest sangat tangguh untuk dataset tabular berbentuk numerik.
    # class_weight='balanced' bisa tetap digunakan untuk double combo, atau bisa juga dicopot.
    # Kita tetap menggunakan class_weight='balanced' sebagai kombinasi dengan SMOTE.
    print("\nSedang melatih Model Random Forest dengan data hasil SMOTE...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_model.fit(X_train_smote, y_train_smote)

    # 5. Memprediksi Data Uji
    print("Melakukan prediksi pada Data Uji...")
    y_pred = rf_model.predict(X_test)

    # 6. Evaluasi Performa Model
    print("\n" + "="*50)
    print("HASIL EVALUASI MODEL: RANDOM FOREST CLASSIFIER")
    print("="*50)
    
    print("\n[1] Akurasi Prediksi (Accuracy Score):", f"{accuracy_score(y_test, y_pred):.2%}")
    
    print("\n[2] Confusion Matrix:")
    cm = pd.DataFrame(confusion_matrix(y_test, y_pred), 
                      columns=["Prediksi Normal (0)", "Prediksi Fraud (1)"], 
                      index=["Aktual Normal (0)", "Aktual Fraud (1)"])
    print(cm)

    print("\n[3] Classification Report (Akurasi Rinci Tiap Kelas):")
    # Memberi penjelasan lebih detil terkait performa untuk mengidentifikasi Fraud
    print(classification_report(y_test, y_pred, target_names=["Normal (0)", "Fraud (1)"]))

    # 7. Ekstraksi Pentingnya Fitur (Feature Importance)
    # Ini akan menunjukkan variabel/fitur apa saja (misal: AQI, Sentimen) yang paling menjadi penentu Fraud
    try:
        print("\nMembuat visualisasi kemaknaan skor fitur...")
        importances = rf_model.feature_importances_
        feature_names = X.columns
        
        feature_df = pd.DataFrame({'Fitur': feature_names, 'Tingkat Kepentingan': importances})
        feature_df = feature_df.sort_values(by='Tingkat Kepentingan', ascending=True)
        
        plt.figure(figsize=(10, 8))
        plt.barh(feature_df['Fitur'], feature_df['Tingkat Kepentingan'], color='skyblue')
        plt.title('Fitur Paling Berpengaruh dalam Penentuan Fraud (Random Forest)')
        plt.xlabel('Tingkat Kepentingan (Importance Score)')
        plt.ylabel('Fitur')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Simpan grafik sebagai gambar
        output_image = 'Feature_Importance_Fraud.png'
        plt.savefig(output_image)
        print(f"[SUCCESS] Grafik feature importance berhasil disimpan di folder ini dengan nama: '{output_image}'")
    except Exception as e:
        print(f"\n[PERINGATAN] Gagal menyimpan grafik matplotlib. Error: {e}")

if __name__ == "__main__":
    main()
