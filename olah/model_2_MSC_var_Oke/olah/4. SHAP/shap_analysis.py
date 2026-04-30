# -*- coding: utf-8 -*-
"""
SHAP ANALYSIS - 3 MODEL FRAUD DETECTION
Output per folder model, waterfall dikelompokkan per variabel (X1, X2, X3).

Model 1: X1              -> Naive Bayes
Model 2: X1+X2           -> Gradient Boosting
Model 3: X1+X2+X3        -> Random Forest

Struktur output:
  model_1_NaiveBayes/
    SHAP_WATERFALL_X1.png          <- 8 Beneish ratios
    SHAP_swarm.png, SHAP_bar.png, ...

  model_2_GradientBoosting/
    SHAP_WATERFALL_X1.png          <- 8 Beneish ratios
    SHAP_WATERFALL_X2.png          <- 3 Linguistik
    SHAP_swarm.png, SHAP_bar.png, ...

  model_3_RandomForest/
    SHAP_WATERFALL_X1.png          <- 8 Beneish ratios
    SHAP_WATERFALL_X2.png          <- 3 Linguistik
    SHAP_WATERFALL_X3.png          <- 1 Volatilitas
    SHAP_swarm.png, SHAP_bar.png, ...
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, matthews_corrcoef)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')

# ── PATH ───────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, r"..\..\Dataset_ML_Ready_CLEAN_2.xlsx")

# ── FITUR ──────────────────────────────────────────────────────────────
TARGET = 'FLAG POTENTIAL FRAUD'

X1_COLS = ['DSRI', 'GMI', 'AQI', 'SGI', 'DEPI', 'SGAI', 'LVGI', 'TATA']
X2_COLS = ['Negative_Tone', 'Positive_Tone', 'Subjectivity_Ratio']
X3_COLS = ['VolatilitasD-30']

ALL_X1    = X1_COLS
ALL_X1X2  = X1_COLS + X2_COLS
ALL_X1X2X3 = X1_COLS + X2_COLS + X3_COLS

FEAT_LABEL = {
    'DSRI': 'DSRI', 'GMI': 'GMI', 'AQI': 'AQI', 'SGI': 'SGI',
    'DEPI': 'DEPI', 'SGAI': 'SGAI', 'LVGI': 'LVGI', 'TATA': 'TATA',
    'Negative_Tone': 'Negative Tone', 'Positive_Tone': 'Positive Tone',
    'Subjectivity_Ratio': 'Subjectivity Ratio',
    'VolatilitasD-30': 'Volatilitas Saham D-30',
}

# Definisi grup variabel
VAR_GROUPS = [
    {
        'name': 'X1',
        'title': 'MScore Features (Beneish Ratios)',
        'cols': X1_COLS,
        'file': 'SHAP_WATERFALL_X1.png',
    },
    {
        'name': 'X2',
        'title': 'Linguistic Features',
        'cols': X2_COLS,
        'file': 'SHAP_WATERFALL_X2.png',
    },
    {
        'name': 'X3',
        'title': 'Volatility Features',
        'cols': X3_COLS,
        'file': 'SHAP_WATERFALL_X3.png',
    },
]

# ── HELPERS ────────────────────────────────────────────────────────────
def save(fig, path):
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"   >> {os.path.basename(path)}")

def evaluate(model, Xt, yt):
    yp  = model.predict(Xt)
    ypr = model.predict_proba(Xt)[:, 1]
    return {
        'AUC': roc_auc_score(yt, ypr), 'F1': f1_score(yt, yp, zero_division=0),
        'Recall': recall_score(yt, yp, zero_division=0),
        'Precision': precision_score(yt, yp, zero_division=0),
        'Accuracy': accuracy_score(yt, yp), 'MCC': matthews_corrcoef(yt, yp),
    }

def to_2d(sv):
    """Ensure SHAP values are 2-D (n_samples, n_features) for class 1."""
    if isinstance(sv, list):
        a = np.array(sv[1])
    else:
        a = np.array(sv)
    if a.ndim == 3:
        a = a[:, :, 1]
    return a


# ══════════════════════════════════════════════════════════════════════
# WATERFALL per grup variabel — 2 PANEL: Fraud vs Non-Fraud
# Biru = positif (mendorong fraud), Merah = negatif
# ══════════════════════════════════════════════════════════════════════
def _draw_waterfall_panel(ax, mean_sv, group_cols, feat_vals_mean, bar_width=0.5):
    """Draw a single waterfall panel on the given axes."""
    sort_order = np.argsort(np.abs(mean_sv))[::-1]
    labels   = [FEAT_LABEL.get(group_cols[i], group_cols[i]) for i in sort_order]
    values   = mean_sv[sort_order]
    raw_cols = [group_cols[i] for i in sort_order]

    pos_segs = [(v, l, r) for v, l, r in zip(values, labels, raw_cols) if v >= 0]
    neg_segs = [(v, l, r) for v, l, r in zip(values, labels, raw_cols) if v < 0]
    pos_segs.sort(key=lambda x: x[0])
    neg_segs.sort(key=lambda x: abs(x[0]))

    pos_bottom, neg_bottom = 0.0, 0.0
    for v, lbl, raw in pos_segs:
        ax.bar(0, v, bottom=pos_bottom, width=bar_width,
               color='#3B82F6', edgecolor='white', linewidth=1.5, zorder=3)
        mid = pos_bottom + v / 2
        fv = feat_vals_mean.get(raw, 0)
        if abs(v) > 0.001:
            ax.text(0, mid, f'{v:.2f}', ha='center', va='center',
                    fontsize=9, fontweight='bold', color='white', zorder=5)
        ax.text(bar_width/2 + 0.03, mid, f'{lbl} = {fv:.2f}',
                ha='left', va='center', fontsize=8.5, color='#1e293b', zorder=5)
        pos_bottom += v

    ax.axhline(0, color='#475569', linewidth=1.2, linestyle='--', zorder=2)

    for v, lbl, raw in neg_segs:
        ax.bar(0, v, bottom=neg_bottom, width=bar_width,
               color='#EF4444', edgecolor='white', linewidth=1.5, zorder=3)
        mid = neg_bottom + v / 2
        fv = feat_vals_mean.get(raw, 0)
        if abs(v) > 0.001:
            ax.text(0, mid, f'{v:.2f}', ha='center', va='center',
                    fontsize=9, fontweight='bold', color='white', zorder=5)
        ax.text(bar_width/2 + 0.03, mid, f'{lbl} = {fv:.2f}',
                ha='left', va='center', fontsize=8.5, color='#1e293b', zorder=5)
        neg_bottom += v

    ax.set_xlim(-0.5, 2.0)
    ax.set_xticks([])
    ax.set_ylabel('Mean SHAP Value', fontsize=10)
    ax.spines[['top', 'right', 'bottom']].set_visible(False)
    ax.tick_params(axis='y', labelsize=8)


def plot_waterfall_group(sv_2d, all_model_cols, group_cols, Xte_raw, y_test,
                         group_title, model_title, out_path):
    """
    2-panel waterfall: Left = Fraud (y=1), Right = Non-Fraud (y=0)
    sv_2d          : (n_samples, n_features) full model SHAP
    all_model_cols : list col names (same order as sv_2d columns)
    group_cols     : subset cols for this group
    Xte_raw        : test DataFrame (original scale, for mean feature values)
    y_test         : Series with actual labels
    """
    col_idxs   = [all_model_cols.index(c) for c in group_cols]
    y_arr      = np.array(y_test)
    fraud_mask = (y_arr == 1)
    nonfr_mask = (y_arr == 0)

    sv_fraud   = sv_2d[fraud_mask][:, col_idxs]
    sv_nonfr   = sv_2d[nonfr_mask][:, col_idxs]

    mean_fraud = sv_fraud.mean(axis=0)
    mean_nonfr = sv_nonfr.mean(axis=0)

    # Mean feature values per class
    fv_fraud = {c: float(Xte_raw.loc[y_test == 1, c].mean()) for c in group_cols}
    fv_nonfr = {c: float(Xte_raw.loc[y_test == 0, c].mean()) for c in group_cols}

    n_feats = len(group_cols)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, max(5, n_feats * 1.1)))
    fig.patch.set_facecolor('white')
    fig.suptitle(f'{group_title}\n{model_title}',
                 fontsize=12, fontweight='bold', y=1.03, linespacing=1.4)

    # Left panel: Fraud
    _draw_waterfall_panel(ax1, mean_fraud, group_cols, fv_fraud)
    ax1.set_title(f'Fraud (n={fraud_mask.sum()})', fontsize=11, fontweight='bold',
                  color='#DC2626', pad=10)

    # Right panel: Non-Fraud
    _draw_waterfall_panel(ax2, mean_nonfr, group_cols, fv_nonfr)
    ax2.set_title(f'Non-Fraud (n={nonfr_mask.sum()})', fontsize=11, fontweight='bold',
                  color='#2563EB', pad=10)

    # Shared legend
    pos_p = mpatches.Patch(color='#3B82F6', label='Positif (mendorong fraud)')
    neg_p = mpatches.Patch(color='#EF4444', label='Negatif (mengurangi fraud)')
    fig.legend(handles=[pos_p, neg_p], fontsize=8, loc='lower center',
               ncol=2, bbox_to_anchor=(0.5, -0.04), framealpha=0.85)

    fig.text(0.5, -0.07, 'Source: Processed Data',
             ha='center', fontsize=8, color='#94a3b8', style='italic')

    fig.tight_layout()
    save(fig, out_path)


# ══════════════════════════════════════════════════════════════════════
# OTHER PLOTS
# ══════════════════════════════════════════════════════════════════════
def plot_swarm(sv_2d, X_df, title, out_path):
    fig, ax = plt.subplots(figsize=(10, max(5, len(X_df.columns)*0.6)))
    shap.summary_plot(sv_2d, X_df, show=False, plot_type='dot',
                      color=plt.cm.coolwarm, max_display=len(X_df.columns))
    plt.gca().set_title(title, fontsize=12, fontweight='bold', pad=10)
    plt.gca().set_xlabel('SHAP Value', fontsize=9)
    plt.gcf().patch.set_facecolor('white')
    save(plt.gcf(), out_path)

def plot_bar(mean_abs, feat_cols, title, out_path, color):
    lbls = np.array([FEAT_LABEL.get(c, c) for c in feat_cols])
    idx  = np.argsort(mean_abs)
    fig, ax = plt.subplots(figsize=(10, max(5, len(feat_cols)*0.55)))
    bars = ax.barh(lbls[idx].tolist(), mean_abs[idx],
                   color=color, edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Mean |SHAP Value|', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.tick_params(axis='y', labelsize=9)
    mx = mean_abs.max() if mean_abs.max() > 0 else 1
    for bar in bars:
        w = bar.get_width()
        ax.text(w + mx*0.01, bar.get_y() + bar.get_height()/2,
                f'{w:.4f}', va='center', ha='left', fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    save(fig, out_path)


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    print("=" * 65)
    print("  SHAP ANALYSIS — 3 MODEL (waterfall per X1, X2, X3)")
    print("=" * 65)

    # 1. Load
    print("\n[1] Memuat dataset ...")
    df = pd.read_excel(DATA_PATH)
    print(f"    Shape: {df.shape}  |  Fraud: {df[TARGET].sum()} / {len(df)}")

    y = df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(
        df[ALL_X1X2X3], y, test_size=0.2, random_state=42, stratify=y)
    smote = SMOTE(random_state=42)

    # 2. Train
    print("\n[2] Melatih model ...")

    def train_model(cols, clf, label):
        Xs, ys = smote.fit_resample(Xtr[cols], ytr)
        clf.fit(Xs, ys)
        m = evaluate(clf, Xte[cols], yte)
        print(f"    {label:52s}  AUC={m['AUC']:.4f}  F1={m['F1']:.4f}")
        return clf, m

    nb, m1 = train_model(ALL_X1, GaussianNB(),
                          "Model 1 - Naive Bayes (X1)")
    gb, m2 = train_model(ALL_X1X2, GradientBoostingClassifier(random_state=42),
                          "Model 2 - Gradient Boosting (X1+X2)")
    rf, m3 = train_model(ALL_X1X2X3, RandomForestClassifier(random_state=42),
                          "Model 3 - Random Forest (X1+X2+X3)")

    # 3. SHAP
    print("\n[3] Menghitung SHAP values ...")

    print("    Model 1 - KernelExplainer ...")
    bg1   = shap.kmeans(Xtr[ALL_X1], k=30)
    exp1  = shap.KernelExplainer(nb.predict_proba, bg1)
    sv_m1 = to_2d(exp1.shap_values(Xte[ALL_X1], nsamples=150))

    print("    Model 2 - TreeExplainer ...")
    exp2  = shap.TreeExplainer(gb)
    sv_m2 = to_2d(exp2.shap_values(Xte[ALL_X1X2]))

    print("    Model 3 - TreeExplainer ...")
    exp3  = shap.TreeExplainer(rf)
    sv_m3 = to_2d(exp3.shap_values(Xte[ALL_X1X2X3]))

    # Mean feature values
    feat_mean = {c: float(Xte[c].mean()) for c in ALL_X1X2X3}

    # ── Model configs ──────────────────────────────────────────────────
    MODEL_CFGS = [
        {
            'folder'  : 'model_1_NaiveBayes',
            'cols'    : ALL_X1,
            'sv'      : sv_m1,
            'title'   : 'Model 1 - MScore/X1 (Naive Bayes)',
            'color'   : '#3B82F6',
            'metrics' : m1,
            'groups'  : ['X1'],        # Model 1 hanya punya X1
        },
        {
            'folder'  : 'model_2_GradientBoosting',
            'cols'    : ALL_X1X2,
            'sv'      : sv_m2,
            'title'   : 'Model 2 - MScore+Linguistik/X1+X2 (Gradient Boosting)',
            'color'   : '#10B981',
            'metrics' : m2,
            'groups'  : ['X1', 'X2'],  # Model 2 punya X1+X2
        },
        {
            'folder'  : 'model_3_RandomForest',
            'cols'    : ALL_X1X2X3,
            'sv'      : sv_m3,
            'title'   : 'Model 3 - Full/X1+X2+X3 (Random Forest)',
            'color'   : '#F59E0B',
            'metrics' : m3,
            'groups'  : ['X1', 'X2', 'X3'],  # Model 3 punya semua
        },
    ]

    all_imp = {}

    for cfg in MODEL_CFGS:
        folder = os.path.join(BASE_DIR, cfg['folder'])
        os.makedirs(folder, exist_ok=True)

        cols    = cfg['cols']
        sv      = cfg['sv']
        title   = cfg['title']
        color   = cfg['color']
        mean_abs = np.abs(sv).mean(axis=0)

        print(f"\n{'='*55}")
        print(f"  [{cfg['folder']}]")
        print(f"{'='*55}")

        # ── A. Waterfall per grup variabel (X1, X2, X3) ────────────────
        print("  Waterfall per variabel:")
        for vg in VAR_GROUPS:
            if vg['name'] not in cfg['groups']:
                continue
            plot_waterfall_group(
                sv_2d          = sv,
                all_model_cols = cols,
                group_cols     = vg['cols'],
                Xte_raw        = Xte,
                y_test         = yte,
                group_title    = vg['title'],
                model_title    = title,
                out_path       = os.path.join(folder, vg['file']),
            )

        # ── B. Bee-swarm ───────────────────────────────────────────────
        print("  Bee-swarm:")
        Xte_lbl = Xte[cols].rename(columns=FEAT_LABEL)
        plot_swarm(sv, Xte_lbl, f"SHAP Summary - {title}",
                   os.path.join(folder, "SHAP_swarm.png"))

        # ── C. Bar mean |SHAP| ─────────────────────────────────────────
        print("  Bar chart:")
        plot_bar(mean_abs, cols, f"SHAP Importance - {title}",
                 os.path.join(folder, "SHAP_bar.png"), color)

        # ── D. Ranking txt ─────────────────────────────────────────────
        ranked = sorted(zip(cols, mean_abs.tolist()), key=lambda x: -x[1])
        txt = os.path.join(folder, "SHAP_ranking.txt")
        with open(txt, 'w', encoding='utf-8') as f:
            f.write(f"SHAP Feature Importance Ranking\n{title}\n{'='*45}\n")
            for rank, (c, v) in enumerate(ranked, 1):
                f.write(f"{rank:2d}. {FEAT_LABEL.get(c,c):<35s}  {v:.6f}\n")
        print(f"   >> SHAP_ranking.txt")

        # ── E. Metrics txt ─────────────────────────────────────────────
        met = os.path.join(folder, "model_metrics.txt")
        with open(met, 'w', encoding='utf-8') as f:
            f.write(f"Model Metrics\n{title}\n{'='*45}\n")
            for k, v in cfg['metrics'].items():
                f.write(f"  {k:<12s}: {v:.4f}\n")
        print(f"   >> model_metrics.txt")

        all_imp[cfg['folder']] = dict(zip(cols, mean_abs.tolist()))

    # ── Panel gabungan (root folder) ───────────────────────────────────
    print(f"\n{'='*55}")
    print("  [Root folder] Gabungan")
    print(f"{'='*55}")

    print("  Panel 3 model:")
    fig, axes = plt.subplots(1, 3, figsize=(22, 9))
    fig.suptitle("SHAP Feature Importance - Perbandingan 3 Model",
                 fontsize=14, fontweight='bold', y=1.01)
    for ax, cfg in zip(axes, MODEL_CFGS):
        imp_d = all_imp[cfg['folder']]
        cols = cfg['cols']
        ma   = np.array([imp_d[c] for c in cols])
        lbls = np.array([FEAT_LABEL.get(c,c) for c in cols])
        idx  = np.argsort(ma)
        bars = ax.barh(lbls[idx].tolist(), ma[idx],
                       color=cfg['color'], edgecolor='white', linewidth=0.5)
        ax.set_title(cfg['title'].replace(' (', '\n('), fontsize=9, fontweight='bold')
        ax.set_xlabel('Mean |SHAP Value|', fontsize=9)
        ax.tick_params(axis='y', labelsize=8)
        mx = ma.max() if ma.max() > 0 else 1
        for bar in bars:
            w = bar.get_width()
            ax.text(w + mx*0.01, bar.get_y() + bar.get_height()/2,
                    f'{w:.4f}', va='center', ha='left', fontsize=7)
        ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    save(fig, os.path.join(BASE_DIR, "SHAP_panel_3model.png"))

    # Heatmap
    print("  Heatmap:")
    heat = np.array([[all_imp.get(cfg['folder'],{}).get(f,0) for cfg in MODEL_CFGS]
                     for f in ALL_X1X2X3])
    flbls = [FEAT_LABEL.get(f,f) for f in ALL_X1X2X3]
    fig, ax = plt.subplots(figsize=(7, 9))
    im = ax.imshow(heat.T, aspect='auto', cmap='YlOrRd')
    ax.set_xticks(range(len(ALL_X1X2X3)))
    ax.set_xticklabels(flbls, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(3))
    ax.set_yticklabels(['Model 1\n(NB)', 'Model 2\n(GB)', 'Model 3\n(RF)'], fontsize=10)
    ax.set_title("Heatmap Mean |SHAP| per Fitur & Model", fontsize=12, fontweight='bold', pad=10)
    mx = heat.max() if heat.max() > 0 else 1
    for j in range(len(ALL_X1X2X3)):
        for i in range(3):
            v = heat[j, i]
            ax.text(j, i, f'{v:.4f}', ha='center', va='center', fontsize=7,
                    color='white' if v > mx*0.65 else 'black')
    plt.colorbar(im, ax=ax, label='Mean |SHAP Value|', shrink=0.6)
    fig.tight_layout()
    save(fig, os.path.join(BASE_DIR, "SHAP_heatmap.png"))

    # Excel
    print("  Excel:")
    rows = [{'Fitur': FEAT_LABEL.get(f,f),
             'Model 1 - Naive Bayes': all_imp.get('model_1_NaiveBayes',{}).get(f,0),
             'Model 2 - Gradient Boosting': all_imp.get('model_2_GradientBoosting',{}).get(f,0),
             'Model 3 - Random Forest': all_imp.get('model_3_RandomForest',{}).get(f,0)}
            for f in ALL_X1X2X3]
    pd.DataFrame(rows).to_excel(os.path.join(BASE_DIR, "shap_importance_3model.xlsx"), index=False)
    print("   >> shap_importance_3model.xlsx")

    pd.DataFrame({'Model 1 (NB)': m1, 'Model 2 (GB)': m2, 'Model 3 (RF)': m3}
                 ).T.round(4).to_excel(os.path.join(BASE_DIR, "performa_3model.xlsx"))
    print("   >> performa_3model.xlsx")

    # ── Ringkasan ──────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  SELESAI! Struktur output:")
    print("=" * 65)
    for cfg in MODEL_CFGS:
        folder = os.path.join(BASE_DIR, cfg['folder'])
        files  = sorted(os.listdir(folder))
        print(f"\n  [{cfg['folder']}/]")
        for f in files:
            print(f"    {f}")
    print(f"\n  [root/]")
    for f in sorted(os.listdir(BASE_DIR)):
        if os.path.isfile(os.path.join(BASE_DIR, f)):
            print(f"    {f}")


if __name__ == "__main__":
    main()
