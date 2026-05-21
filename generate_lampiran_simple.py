"""
generate_lampiran_simple.py
===========================
Versi simpel: 1 halaman per lampiran, hitam-putih, gaya akademik.

Output: olah/lampiran_pdf_simple/
  Lampiran_A_Data_Penelitian.pdf
  Lampiran_B_Bukti_SP2.pdf
  Lampiran_C_Leksikon_InSET.pdf
  Lampiran_D_Hasil_63_Skenario.pdf
"""

import os, json, warnings
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE      = r"C:\Users\Malik\Documents\SKRIPSI_DATA\fraud_detection_ML"
OLAH      = os.path.join(BASE, "olah")
MODEL_DIR = os.path.join(OLAH, "model_2_MSC_var_Oke")
INSET_DIR = os.path.join(BASE, "InSet")
OUT_DIR   = os.path.join(OLAH, "lampiran_pdf_simple")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Global style: hitam-putih ─────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.edgecolor":    "black",
    "axes.labelcolor":   "black",
    "xtick.color":       "black",
    "ytick.color":       "black",
    "text.color":        "black",
    "grid.color":        "#cccccc",
    "font.family":       "DejaVu Sans",
    "font.size":         9,
})

A4L = (11.69, 8.27)   # A4 landscape (inci)

# ── Helper: gambar garis judul sederhana ──────────────────────────────────────
def add_title(fig, lampiran_label: str, title: str, subtitle: str = ""):
    fig.text(0.5, 0.97, lampiran_label, ha="center", va="top",
             fontsize=8, style="italic", color="#555555")
    fig.text(0.5, 0.94, title, ha="center", va="top",
             fontsize=13, fontweight="bold")
    if subtitle:
        fig.text(0.5, 0.91, subtitle, ha="center", va="top",
                 fontsize=8, color="#333333")
    # garis pemisah
    line = plt.Line2D([0.05, 0.95], [0.895, 0.895],
                      transform=fig.transFigure, color="black", linewidth=0.8)
    fig.add_artist(line)

def add_footer(fig, source: str = ""):
    if source:
        fig.text(0.5, 0.015, f"Sumber: {source}", ha="center",
                 fontsize=7, color="#444444", style="italic")

# ── Helper: tabel matplotlib sederhana ───────────────────────────────────────
def simple_table(ax, df: pd.DataFrame, fontsize: float = 7.5):
    """Render DataFrame sebagai tabel B&W di Axes ax."""
    ax.axis("off")
    n_rows, n_cols = df.shape

    cell_data   = df.values.tolist()
    col_labels  = df.columns.tolist()

    # Warna: header abu, baris zebra putih/abu muda
    header_colors = [["#cccccc"] * n_cols]
    row_colors    = [["#f5f5f5" if i % 2 == 0 else "white"] * n_cols
                     for i in range(n_rows)]

    tbl = ax.table(
        cellText=cell_data,
        colLabels=col_labels,
        cellColours=row_colors,
        colColours=header_colors[0],
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    tbl.scale(1, 1.35)

    # Tebalkan header
    for j in range(n_cols):
        tbl[0, j].set_text_props(fontweight="bold")

    return tbl


# ══════════════════════════════════════════════════════════════════════════════
# LAMPIRAN A – Data Penelitian
# ══════════════════════════════════════════════════════════════════════════════
def build_a():
    print("[A] Lampiran A ...")

    df_ml  = pd.read_excel(os.path.join(MODEL_DIR, "Dataset_ML_Ready_CLEAN_2.xlsx"))
    df_raw = pd.read_excel(os.path.join(OLAH, "Raw_Data_Dump_2022-2024 No Outlier.xlsx"))
    df_sanksi = pd.read_excel(
        os.path.join(OLAH, "publikasi-data-sanksi-ind(1).xlsx"), sheet_name="Sanksi")

    # Peta kode -> nama dari BEI JSON
    bei_map = {}
    for fn in ["BEI2023.json", "BEI2024.json"]:
        try:
            with open(os.path.join(BASE, fn), encoding="utf-8") as f:
                bei = json.load(f)
            for item in bei.get("Results", []):
                k, v = item.get("KodeEmiten",""), item.get("NamaEmiten","")
                if k: bei_map[k] = v
        except Exception:
            pass

    sp2_set = set(df_sanksi[df_sanksi["Jenis Sanksi"]=="SP2"]["Kode"].str.strip())

    # Daftar perusahaan unik (max 20 baris untuk 1 halaman)
    tickers = (df_raw[["Ticker"]].drop_duplicates()
                                 .sort_values("Ticker")
                                 .reset_index(drop=True))
    tickers["Nama Perusahaan"] = tickers["Ticker"].map(bei_map).fillna("-")
    tickers["Status"] = tickers["Ticker"].apply(
        lambda t: "Potential Fraud" if t in sp2_set else "Non-Fraud")

    n_total = len(tickers)
    n_fraud = (tickers["Status"] == "Potential Fraud").sum()
    n_non   = n_total - n_fraud

    # Ambil 20 baris teratas untuk sampel tabel
    sample = tickers.head(20).copy().reset_index(drop=True)
    sample.index = range(1, len(sample)+1)
    sample.insert(0, "No.", sample.index)
    sample = sample.reset_index(drop=True)
    sample.columns = ["No.", "Kode Saham", "Nama Perusahaan", "Status"]

    # ── Bangun PDF ─────────────────────────────────────────────────────────
    out = os.path.join(OUT_DIR, "Lampiran_A_Data_Penelitian.pdf")
    with PdfPages(out) as pdf:
        fig = plt.figure(figsize=A4L)
        add_title(fig,
            "LAMPIRAN A",
            "Data Penelitian: Daftar Perusahaan Sampel",
            f"Total sampel: {n_total} perusahaan  |  Potential Fraud: {n_fraud}  |  Non-Fraud: {n_non}  |  Periode: 2023–2024")
        add_footer(fig, "BEI (Bursa Efek Indonesia) & Data Sanksi BEI 2023–2024")

        # Kiri: tabel 20 perusahaan
        ax_tbl = fig.add_axes([0.04, 0.08, 0.52, 0.78])
        ax_tbl.set_title("Sampel 20 Perusahaan Pertama (dari total 395)", fontsize=9, pad=6)
        simple_table(ax_tbl, sample, fontsize=7)

        # Kanan atas: pie distribusi kelas
        ax_pie = fig.add_axes([0.60, 0.48, 0.36, 0.40])
        ax_pie.pie(
            [n_fraud, n_non],
            labels=[f"Potential Fraud\n({n_fraud})", f"Non-Fraud\n({n_non})"],
            autopct="%1.1f%%",
            colors=["#888888", "#dddddd"],
            startangle=90,
            wedgeprops={"edgecolor": "black", "linewidth": 0.8},
            textprops={"fontsize": 8},
        )
        ax_pie.set_title("Distribusi Kelas Label", fontsize=9, pad=6)

        # Kanan bawah: rata-rata M-Score komponen per kelas
        ax_bar = fig.add_axes([0.60, 0.08, 0.36, 0.36])
        vars_ms = ["DSRI", "GMI", "AQI", "SGI", "DEPI", "SGAI", "LVGI", "TATA"]
        m_fraud = df_ml[df_ml["FLAG POTENTIAL FRAUD"]==1][vars_ms].mean().values
        m_non   = df_ml[df_ml["FLAG POTENTIAL FRAUD"]==0][vars_ms].mean().values
        x = np.arange(len(vars_ms)); w = 0.35
        ax_bar.bar(x - w/2, m_fraud, w, label="Potential Fraud",
                   color="#888888", edgecolor="black", linewidth=0.5)
        ax_bar.bar(x + w/2, m_non,   w, label="Non-Fraud",
                   color="#dddddd", edgecolor="black", linewidth=0.5)
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(vars_ms, fontsize=7)
        ax_bar.set_ylabel("Rata-rata", fontsize=8)
        ax_bar.set_title("Rata-rata Komponen M-Score per Kelas", fontsize=8, pad=4)
        ax_bar.legend(fontsize=7, loc="upper right")
        ax_bar.grid(axis="y", linestyle="--", alpha=0.5)
        ax_bar.axhline(1.0, color="black", linewidth=0.6, linestyle=":")

        pdf.savefig(fig, facecolor="white")
        plt.close(fig)
    print(f"   Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# LAMPIRAN B – Bukti Kriteria SP2
# ══════════════════════════════════════════════════════════════════════════════
def build_b():
    print("[B] Lampiran B ...")

    df_sanksi = pd.read_excel(
        os.path.join(OLAH, "publikasi-data-sanksi-ind(1).xlsx"), sheet_name="Sanksi")

    df_sp2 = df_sanksi[df_sanksi["Jenis Sanksi"]=="SP2"].copy()
    df_sp2["Tahun"] = df_sp2["Tahun"].astype(int)
    df_period = df_sp2[df_sp2["Tahun"].isin([2023,2024])].copy()
    df_period["Tgl Surat"] = pd.to_datetime(df_period["Tgl Surat"]).dt.strftime("%d-%m-%Y")

    # Emiten unik + jumlah SP2
    sp2_uniq = (df_period.groupby("Kode")
                         .agg(Jumlah_SP2=("Jenis Sanksi","count"),
                              Kewajiban_Dominan=("Jenis Kewajiban",
                                                  lambda x: x.mode()[0]))
                         .reset_index()
                         .sort_values("Jumlah_SP2", ascending=False)
                         .reset_index(drop=True))

    n_total  = len(df_sanksi)
    n_sp2    = len(df_period)
    n_emiten = df_period["Kode"].nunique()

    # Ambil top-25 untuk tabel
    top25 = sp2_uniq.head(25).copy()
    top25.index = range(1, len(top25)+1)
    top25.insert(0, "No.", top25.index)
    top25 = top25.reset_index(drop=True)
    top25.columns = ["No.", "Kode Emiten", "Jumlah SP2", "Kewajiban Dominan"]

    # distribusi per tahun
    yr = df_period["Tahun"].value_counts().sort_index()
    # distribusi top kewajiban
    kwj = df_period["Jenis Kewajiban"].value_counts().head(6)

    out = os.path.join(OUT_DIR, "Lampiran_B_Bukti_SP2.pdf")
    with PdfPages(out) as pdf:
        fig = plt.figure(figsize=A4L)
        add_title(fig,
            "LAMPIRAN B",
            "Bukti Kriteria Potential Fraud: Data Sanksi SP2 Bursa Efek Indonesia",
            f"Total record sanksi: {n_total:,}  |  Total SP2 (2023-2024): {n_sp2:,}  |  Emiten unik: {n_emiten}")
        add_footer(fig, "BEI – Publikasi Data Sanksi (publikasi-data-sanksi-ind.xlsx)")

        # Tabel kiri: top-25 emiten terkena SP2
        ax_tbl = fig.add_axes([0.04, 0.08, 0.50, 0.78])
        ax_tbl.set_title("Top-25 Emiten dengan Frekuensi SP2 Terbanyak (2023–2024)", fontsize=9, pad=6)
        simple_table(ax_tbl, top25, fontsize=7)

        # Bar kanan atas: SP2 per tahun
        ax_yr = fig.add_axes([0.60, 0.52, 0.36, 0.35])
        ax_yr.bar(yr.index.astype(str), yr.values,
                  color=["#888888","#aaaaaa"], edgecolor="black", linewidth=0.7, width=0.4)
        for bar, v in zip(ax_yr.patches, yr.values):
            ax_yr.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
                       str(v), ha="center", fontsize=9, fontweight="bold")
        ax_yr.set_title("Jumlah Sanksi SP2 per Tahun", fontsize=9, pad=4)
        ax_yr.set_ylabel("Jumlah", fontsize=8)
        ax_yr.grid(axis="y", linestyle="--", alpha=0.5)

        # Bar kanan bawah: top kewajiban
        ax_kwj = fig.add_axes([0.60, 0.08, 0.36, 0.38])
        ax_kwj.barh(kwj.index, kwj.values,
                    color="#aaaaaa", edgecolor="black", linewidth=0.6)
        ax_kwj.set_title("Top 6 Jenis Kewajiban (SP2)", fontsize=9, pad=4)
        ax_kwj.set_xlabel("Jumlah Sanksi", fontsize=8)
        ax_kwj.invert_yaxis()
        ax_kwj.tick_params(axis="y", labelsize=7)
        ax_kwj.grid(axis="x", linestyle="--", alpha=0.5)

        pdf.savefig(fig, facecolor="white")
        plt.close(fig)
    print(f"   Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# LAMPIRAN C – Sampel Leksikon InSET
# ══════════════════════════════════════════════════════════════════════════════
def build_c():
    print("[C] Lampiran C ...")

    pos_df = pd.read_csv(os.path.join(INSET_DIR, "positive.tsv"),
                         sep="\t", header=0, names=["Kata","Bobot"])
    neg_df = pd.read_csv(os.path.join(INSET_DIR, "negative.tsv"),
                         sep="\t", header=0, names=["Kata","Bobot"])
    pos_df["Bobot"] = pd.to_numeric(pos_df["Bobot"], errors="coerce")
    neg_df["Bobot"] = pd.to_numeric(neg_df["Bobot"], errors="coerce")

    n_pos = len(pos_df); n_neg = len(neg_df)

    # Top-20 positif & top-20 negatif
    pos20 = pos_df.sort_values("Bobot", ascending=False).head(20).reset_index(drop=True)
    neg20 = neg_df.sort_values("Bobot").head(20).reset_index(drop=True)

    pos20.index = range(1, 21); pos20.insert(0,"No.",pos20.index); pos20=pos20.reset_index(drop=True)
    neg20.index = range(1, 21); neg20.insert(0,"No.",neg20.index); neg20=neg20.reset_index(drop=True)

    # Distribusi bobot
    pos_wt = pos_df["Bobot"].dropna().value_counts().sort_index()
    neg_wt = neg_df["Bobot"].dropna().value_counts().sort_index()

    out = os.path.join(OUT_DIR, "Lampiran_C_Leksikon_InSET.pdf")
    with PdfPages(out) as pdf:
        fig = plt.figure(figsize=A4L)
        add_title(fig,
            "LAMPIRAN C",
            "Sampel Leksikon Sentimen InSET (Indonesian Sentiment Lexicon)",
            f"Total kosakata positif: {n_pos:,}  |  Total kosakata negatif: {n_neg:,}  |  Grand total: {n_pos+n_neg:,}")
        add_footer(fig, "Koto et al. (2020) – InSet: Indonesian Sentiment Lexicon")

        # Tabel kiri: top-20 positif
        ax_p = fig.add_axes([0.03, 0.08, 0.27, 0.78])
        ax_p.set_title("20 Kata Positif Bobot Tertinggi", fontsize=9, pad=6)
        simple_table(ax_p, pos20, fontsize=7)

        # Tabel tengah: top-20 negatif
        ax_n = fig.add_axes([0.35, 0.08, 0.27, 0.78])
        ax_n.set_title("20 Kata Negatif Bobot Terendah", fontsize=9, pad=6)
        simple_table(ax_n, neg20, fontsize=7)

        # Kanan atas: distribusi bobot positif
        ax_ph = fig.add_axes([0.66, 0.52, 0.30, 0.35])
        ax_ph.bar(pos_wt.index.astype(str), pos_wt.values,
                  color="#888888", edgecolor="black", linewidth=0.6)
        ax_ph.set_title("Distribusi Bobot Kata Positif", fontsize=9, pad=4)
        ax_ph.set_xlabel("Bobot"); ax_ph.set_ylabel("Jumlah Kata", fontsize=8)
        ax_ph.grid(axis="y", linestyle="--", alpha=0.5)

        # Kanan bawah: distribusi bobot negatif
        ax_nh = fig.add_axes([0.66, 0.08, 0.30, 0.35])
        ax_nh.bar(neg_wt.index.astype(str), neg_wt.values,
                  color="#aaaaaa", edgecolor="black", linewidth=0.6)
        ax_nh.set_title("Distribusi Bobot Kata Negatif", fontsize=9, pad=4)
        ax_nh.set_xlabel("Bobot"); ax_nh.set_ylabel("Jumlah Kata", fontsize=8)
        ax_nh.grid(axis="y", linestyle="--", alpha=0.5)

        pdf.savefig(fig, facecolor="white")
        plt.close(fig)
    print(f"   Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# LAMPIRAN D – Hasil Perbandingan 63 Skenario
# ══════════════════════════════════════════════════════════════════════════════
def build_d():
    print("[D] Lampiran D ...")

    df = pd.read_excel(os.path.join(MODEL_DIR, "Hasil_Perbandingan_63_Skenario.xlsx"))
    df.columns = [c.strip() for c in df.columns]
    metrics = ["AUC","CA","F1","Prec","Recall","MCC"]
    for m in metrics:
        df[m] = pd.to_numeric(df[m], errors="coerce")

    df_sorted = df.sort_values("AUC", ascending=False).reset_index(drop=True)
    best = df_sorted.iloc[0]

    # Top-15 untuk tabel
    top15 = df_sorted.head(15).copy()
    top15.index = range(1,16)
    top15.insert(0,"Rank", top15.index)
    top15 = top15.reset_index(drop=True)
    for m in metrics:
        top15[m] = top15[m].round(4)
    top15.columns = ["Rank","Model","Skenario Fitur","AUC","CA","F1","Prec","Recall","MCC"]

    # Pivot heatmap: AUC model vs skenario
    pivot = df.pivot_table(index="Model", columns="Skenario Fitur",
                           values="AUC", aggfunc="max")

    # AUC per model (best)
    best_model = (df.groupby("Model")["AUC"].max()
                    .sort_values(ascending=True))

    out = os.path.join(OUT_DIR, "Lampiran_D_Hasil_63_Skenario.pdf")
    with PdfPages(out) as pdf:
        fig = plt.figure(figsize=A4L)
        add_title(fig,
            "LAMPIRAN D",
            "Hasil Perbandingan 63 Skenario Model Machine Learning",
            (f"Model Terbaik: {best['Model']}  |  Skenario: {best['Skenario Fitur']}  "
             f"|  AUC: {best['AUC']:.4f}  |  F1: {best['F1']:.4f}"))
        add_footer(fig, "Output model: Hasil_Perbandingan_63_Skenario.xlsx")

        # Kiri: tabel top-15
        ax_tbl = fig.add_axes([0.03, 0.06, 0.54, 0.80])
        ax_tbl.set_title("Top-15 Skenario Berdasarkan AUC", fontsize=9, pad=6)
        simple_table(ax_tbl, top15, fontsize=6.5)

        # Kanan atas: horizontal bar – best AUC per model
        ax_bar = fig.add_axes([0.62, 0.52, 0.35, 0.35])
        ax_bar.barh(best_model.index, best_model.values,
                    color="#aaaaaa", edgecolor="black", linewidth=0.6)
        ax_bar.axvline(0.5, color="black", linewidth=0.7, linestyle="--")
        ax_bar.set_title("AUC Terbaik per Model", fontsize=9, pad=4)
        ax_bar.set_xlabel("AUC", fontsize=8)
        ax_bar.set_xlim(0.35, 0.75)
        ax_bar.tick_params(axis="y", labelsize=7.5)
        ax_bar.grid(axis="x", linestyle="--", alpha=0.5)
        for bar, v in zip(ax_bar.patches, best_model.values):
            ax_bar.text(v + 0.004, bar.get_y() + bar.get_height()/2,
                        f"{v:.3f}", va="center", fontsize=7)

        # Kanan bawah: heatmap AUC model vs skenario
        ax_hm = fig.add_axes([0.62, 0.06, 0.35, 0.40])
        im = ax_hm.imshow(pivot.values, cmap="Greys", aspect="auto",
                          vmin=0.35, vmax=0.75)
        ax_hm.set_xticks(range(len(pivot.columns)))
        ax_hm.set_xticklabels(pivot.columns, fontsize=5.5, rotation=25, ha="right")
        ax_hm.set_yticks(range(len(pivot.index)))
        ax_hm.set_yticklabels(pivot.index, fontsize=6.5)
        ax_hm.set_title("Heatmap AUC (Model vs Skenario)", fontsize=9, pad=4)

        # Anotasi nilai di tiap sel
        for r in range(pivot.shape[0]):
            for c in range(pivot.shape[1]):
                v = pivot.values[r, c]
                if not np.isnan(v):
                    ax_hm.text(c, r, f"{v:.2f}", ha="center", va="center",
                               fontsize=5, color="white" if v < 0.56 else "black")

        plt.colorbar(im, ax=ax_hm, shrink=0.8, pad=0.02)

        pdf.savefig(fig, facecolor="white")
        plt.close(fig)
    print(f"   Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 55)
    print("  Generating Lampiran PDF (Simple) – Skripsi ML")
    print("=" * 55)
    build_a()
    build_b()
    build_c()
    build_d()
    print()
    print("=" * 55)
    print(f"  Selesai! Output folder:")
    print(f"  {OUT_DIR}")
    print("=" * 55)
