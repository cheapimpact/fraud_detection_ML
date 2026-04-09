import pandas as pd
import os
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

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

    print("\nMempersiapkan komponen algoritma: XGBoost, LightGBM, dan RandomForest...")
    xgb_clf = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
    lgbm_clf = LGBMClassifier(n_estimators=100, random_state=42, verbosity=-1)
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Membangun Voting Classifier (Soft Voting: mengambil probabilitas rata-rata dari ke-3 otak buatan)
    print("Membangun Stacking / Voting Classifier (Ketiganya akan voting secara demokratis)...")
    voting_model = VotingClassifier(
        estimators=[
            ('xgb', xgb_clf),
            ('lgbm', lgbm_clf),
            ('rf', rf_clf)
        ],
        voting='soft'
    )
    
    print("Sedang melatih model gabungan (Ensemble) ini... Mohon tunggu.")
    voting_model.fit(X_train_smote, y_train_smote)

    print("Melakukan prediksi pada Data Uji...")
    y_pred = voting_model.predict(X_test)

    print("\n" + "="*50)
    print("HASIL EVALUASI MODEL: ENSEMBLE VOTING CLASSIFIER")
    print("="*50)
    print("\n[1] Akurasi Prediksi (Accuracy Score):", f"{accuracy_score(y_test, y_pred):.2%}")
    
    print("\n[2] Confusion Matrix:")
    cm = pd.DataFrame(confusion_matrix(y_test, y_pred), 
                      columns=["Prediksi Normal (0)", "Prediksi Fraud (1)"], 
                      index=["Aktual Normal (0)", "Aktual Fraud (1)"])
    print(cm)

    print("\n[3] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal (0)", "Fraud (1)"]))

if __name__ == "__main__":
    main()
