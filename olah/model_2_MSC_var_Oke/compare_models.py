import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from statsmodels.stats.contingency_tables import mcnemar
import shap
import matplotlib.pyplot as plt

def main():
    print("Memuat dataset...")
    df = pd.read_excel('/Users/calcifer/Documents/MDMA/olah/model 2 MSC var  Oke/Dataset_ML_Ready_CLEAN_2.xlsx')

    print("\n" + "="*50)
    print("1. Analisis Deskriptif")
    print("="*50)
    features_desc = ['DSRI', 'GMI', 'AQI', 'SGI', 'DEPI', 'SGAI', 'LVGI', 'TATA', 
                     'VolatilitasD-30', 'Negative_Tone', 'Positive_Tone', 'Subjectivity_Ratio']
    
    # Calculate Mean, Median, and Std Dev grouped by FLAG POTENTIAL FRAUD
    desc_stats = df.groupby('FLAG POTENTIAL FRAUD')[features_desc].agg(['mean', 'median', 'std']).T
    print(desc_stats)

    print("\n" + "="*50)
    print("2. Uji Performa Klasifikasi (Komparasi 3 Model)")
    print("="*50)
    
    # Define features
    X1_cols = ['DSRI', 'GMI', 'AQI', 'SGI', 'DEPI', 'SGAI', 'LVGI', 'TATA']
    X2_cols = X1_cols + ['Negative_Tone', 'Positive_Tone', 'Subjectivity_Ratio']
    X3_cols = X2_cols + ['VolatilitasD-30']
    
    target = 'FLAG POTENTIAL FRAUD'
    y = df[target]
    
    # Train-test split (80-20), stratify for class imbalance
    X_train_full, X_test_full, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42, stratify=y)
    
    # SMOTE
    smote = SMOTE(random_state=42)
    
    def evaluate_model(features, model_name):
        # Apply SMOTE to the training data
        X_train_res, y_train_res = smote.fit_resample(X_train_full[features], y_train)
        
        # XGBoost Classifier
        model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train_res, y_train_res)
        
        # Predict on test set
        y_pred = model.predict(X_test_full[features])
        y_prob = model.predict_proba(X_test_full[features])[:, 1]
        
        f1 = f1_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        return model, y_pred, {'F1-Score': f1, 'Precision': prec, 'Recall': rec, 'AUC': auc}

    # Evaluate Model 1
    _, y_pred1, metrics1 = evaluate_model(X1_cols, "Model 1")
    print(f"Model 1 (Baseline) metrics:\n {metrics1}")
    
    # Evaluate Model 2 MSC var  Oke
    _, y_pred2, metrics2 = evaluate_model(X2_cols, "Model 2 MSC var  Oke")
    print(f"\nModel 2 MSC var  Oke (Replikasi Jurnal) metrics:\n {metrics2}")
    
    # Evaluate Model 3
    model3, y_pred3, metrics3 = evaluate_model(X3_cols, "Model 3")
    print(f"\nModel 3 (Inovasi Skripsi) metrics:\n {metrics3}")

    print("\n" + "="*50)
    print("3. Uji Signifikansi (McNemar's Test) - Model 2 MSC var  Oke vs Model 3")
    print("="*50)
    
    # Identify correct/incorrect predictions
    # 1: Correct, 0: Incorrect
    m2_correct = (y_pred2 == y_test).astype(int)
    m3_correct = (y_pred3 == y_test).astype(int)
    
    # Build a contingency table for McNemar's Test
    c00 = sum((m2_correct == 1) & (m3_correct == 1)) # Both correct
    c01 = sum((m2_correct == 1) & (m3_correct == 0)) # M2 correct, M3 incorrect
    c10 = sum((m2_correct == 0) & (m3_correct == 1)) # M2 incorrect, M3 correct
    c11 = sum((m2_correct == 0) & (m3_correct == 0)) # Both incorrect
    
    contingency_table = [[c00, c01], 
                         [c10, c11]]
    
    result = mcnemar(contingency_table, exact=False, correction=True)
    
    print("Tabel Kontingensi:")
    print(f"[{c00}, {c01}] (M2 Correct+M3 Correct , M2 Correct+M3 Wrong)")
    print(f"[{c10}, {c11}] (M2 Wrong+M3 Correct  , M2 Wrong+M3 Wrong)")
    
    print(f"\nMcNemar's test statistic: {result.statistic:.4f}, p-value: {result.pvalue:.4f}")
    if result.pvalue < 0.05:
        print("=> Kesimpulan: Perbedaan prediksi antara Model 2 MSC var  Oke dan Model 3 SIGNIFIKAN secara statistik (p < 0.05).")
    else:
        print("=> Kesimpulan: Perbedaan prediksi antara Model 2 MSC var  Oke dan Model 3 TIDAK SIGNIFIKAN secara statistik (p >= 0.05).")

    print("\n" + "="*50)
    print("4. Analisis SHAP (Feature Importance Model 3)")
    print("="*50)
    
    # SHAP Explainer
    explainer = shap.TreeExplainer(model3)
    shap_values = explainer.shap_values(X_test_full[X3_cols])
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test_full[X3_cols], show=False)
    plt.tight_layout()
    plt.savefig('/Users/calcifer/Documents/MDMA/olah/model 2 MSC var  Oke/model3_shap_summary.png', dpi=300)
    print("=> Plot SHAP Summary telah berhasil disimpan sebagai 'model3_shap_summary.png' di folder 'olah/model 2 MSC var  Oke'.")

if __name__ == "__main__":
    main()
