"""
generate_lampiran_pdf.py
========================
Script untuk menghasilkan lampiran PDF skripsi:
  Lampiran A – Daftar Perusahaan Sampel & Ringkasan Dataset
  Lampiran B – Bukti / Kriteria SP2 (Data Sanksi BEI)
  Lampiran C – Sampel Leksikon InSET
  Lampiran D – Hasil Perbandingan 63 Skenario Model ML

Output: folder  olah/lampiran_pdf/
"""

import os
import json
import warnings
import textwrap

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.ticker as mticker
import numpy as np

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE = r"C:\Users\Malik\Documents\SKRIPSI_DATA\fraud_detection_ML"
OLAH = os.path.join(BASE, "olah")
MODEL_DIR = os.path.join(OLAH, "model_2_MSC_var_Oke")
INSET_DIR = os.path.join(BASE, "InSet")
OUT_DIR = os.path.join(OLAH, "lampiran_pdf")
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Colour palette ───────────────────────────────────────────────────────────
C_DARK    = "#1A1A2E"
C_NAVY    = "#16213E"
C_BLUE    = "#0F3460"
C_ACCENT  = "#E94560"
C_GOLD    = "#F5A623"
C_LIGHT   = "#F0F0F0"
C_WHITE   = "#FFFFFF"
C_GREEN   = "#2ECC71"
C_RED     = "#E74C3C"
C_MUTED   = "#7F8C8D"

def set_figure_style():
    plt.rcParams.update({
        "figure.facecolor": C_DARK,
        "axes.facecolor": C_NAVY,
        "axes.edgecolor": C_BLUE,
        "axes.labelcolor": C_LIGHT,
        "xtick.color": C_LIGHT,
        "ytick.color": C_LIGHT,
        "text.color": C_LIGHT,
        "grid.color": "#2A2A4A",
        "grid.alpha": 0.5,
        "font.family": "DejaVu Sans",
        "font.size": 9,
    })

set_figure_style()


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def draw_page_header(fig, title: str, subtitle: str, lampiran: str):
    """Draws a premium gradient header bar on top of the figure."""
    ax_h = fig.add_axes([0, 0.93, 1, 0.07])
    ax_h.set_xlim(0, 1); ax_h.set_ylim(0, 1)
    ax_h.axis("off")
    # gradient rectangle
    grad = np.linspace(0, 1, 256).reshape(1, -1)
    ax_h.imshow(grad, aspect="auto", extent=[0, 1, 0, 1],
                cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
                    "hdr", [C_BLUE, C_DARK]), alpha=0.95)
    ax_h.text(0.01, 0.72, lampiran, color=C_GOLD, fontsize=8, fontweight="bold",
              va="center", transform=ax_h.transAxes)
    ax_h.text(0.01, 0.35, title, color=C_WHITE, fontsize=12, fontweight="bold",
              va="center", transform=ax_h.transAxes)
    ax_h.text(0.99, 0.35, subtitle, color=C_MUTED, fontsize=8, ha="right",
              va="center", transform=ax_h.transAxes)
    # accent line
    ax_line = fig.add_axes([0, 0.925, 1, 0.006])
    ax_line.set_facecolor(C_ACCENT); ax_line.axis("off")


def draw_footer(fig, page_num: int, total: int):
    ax_f = fig.add_axes([0, 0, 1, 0.025])
    ax_f.set_facecolor(C_NAVY); ax_f.axis("off")
    ax_f.text(0.5, 0.5,
              f"Skripsi – Deteksi Kecurangan Laporan Keuangan Berbasis Machine Learning  |  Halaman {page_num}/{total}",
              color=C_MUTED, fontsize=7, ha="center", va="center",
              transform=ax_f.transAxes)


def styled_table(ax, df, col_widths=None, header_bg=C_BLUE,
                 row_colors=None, fontsize=7.5, col_aligns=None):
    """Renders a styled table into a matplotlib Axes."""
    ax.axis("off")
    n_rows, n_cols = df.shape
    if col_widths is None:
        col_widths = [1 / n_cols] * n_cols
    if row_colors is None:
        row_colors = [C_NAVY if i % 2 == 0 else "#1E2A4A" for i in range(n_rows)]
    if col_aligns is None:
        col_aligns = ["center"] * n_cols

    cell_data = [df.columns.tolist()] + df.values.tolist()
    cell_colors = [[header_bg] * n_cols] + [[row_colors[i]] * n_cols for i in range(n_rows)]

    tbl = ax.table(
        cellText=cell_data,
        cellColours=cell_colors,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)

    # Header row styling
    for j in range(n_cols):
        cell = tbl[0, j]
        cell.set_text_props(color=C_WHITE, fontweight="bold")
        cell.set_edgecolor(C_ACCENT)

    # Data rows
    for i in range(1, n_rows + 1):
        for j in range(n_cols):
            cell = tbl[i, j]
            cell.set_text_props(color=C_LIGHT)
            cell.set_edgecolor("#2A3A5A")
            cell._loc = col_aligns[j]

    tbl.scale(1, 1.4)
    return tbl


# ══════════════════════════════════════════════════════════════════════════════
#  LAMPIRAN A – Daftar Perusahaan Sampel & Ringkasan Dataset
# ══════════════════════════════════════════════════════════════════════════════

def build_lampiran_a():
    print("[Lampiran A] Membaca data …")

    # ── Load data ──────────────────────────────────────────────────────────
    df_ml = pd.read_excel(os.path.join(MODEL_DIR, "Dataset_ML_Ready_CLEAN_2.xlsx"))
    df_raw = pd.read_excel(os.path.join(OLAH, "Raw_Data_Dump_2022-2024 No Outlier.xlsx"))
    df_sanksi = pd.read_excel(
        os.path.join(OLAH, "publikasi-data-sanksi-ind(1).xlsx"), sheet_name="Sanksi")

    # BEI JSON → ticker → company name
    bei_map = {}
    for fname in ["BEI2023.json", "BEI2024.json"]:
        try:
            with open(os.path.join(BASE, fname), "r", encoding="utf-8") as f:
                bei = json.load(f)
            for item in bei.get("Results", []):
                kode = item.get("KodeEmiten", "")
                nama = item.get("NamaEmiten", "")
                if kode and nama:
                    bei_map[kode] = nama
        except Exception:
            pass

    # ── Build company table ─────────────────────────────────────────────────
    # SP2 tickers (potential fraud criteria)
    sp2_tickers = set(df_sanksi[df_sanksi["Jenis Sanksi"] == "SP2"]["Kode"].str.strip())

    # Unique tickers in raw data
    raw_tickers = df_raw[["Ticker", "Year"]].copy()
    raw_tickers = raw_tickers.drop_duplicates(subset="Ticker").sort_values("Ticker").reset_index(drop=True)
    raw_tickers["Nama Perusahaan"] = raw_tickers["Ticker"].map(bei_map).fillna("-")
    raw_tickers["Status"] = raw_tickers["Ticker"].apply(
        lambda t: "Potential Fraud" if t in sp2_tickers else "Non-Fraud")

    # Sector mapping (simplified from ticker pattern / known sectors)
    # We'll mark sectors based on available data; add a generic placeholder
    raw_tickers["Sektor"] = "Listed BEI"  # default, enrich below

    company_df = raw_tickers[["Ticker", "Nama Perusahaan", "Sektor", "Status"]].copy()
    company_df.columns = ["Kode Saham", "Nama Perusahaan", "Sektor", "Status"]

    n_fraud = (company_df["Status"] == "Potential Fraud").sum()
    n_non   = (company_df["Status"] == "Non-Fraud").sum()
    n_total = len(company_df)

    # ── Dataset preview (20 rows) ───────────────────────────────────────────
    preview = df_ml.head(20).copy()
    preview["Label"] = preview["FLAG POTENTIAL FRAUD"].map({1: "Fraud", 0: "Non-Fraud"})

    # ── Build PDF ──────────────────────────────────────────────────────────
    out_path = os.path.join(OUT_DIR, "Lampiran_A_Data_Penelitian.pdf")
    with PdfPages(out_path) as pdf:

        # ── Page 1: Summary stats + pie chart + bar chart ──────────────────
        fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
        draw_page_header(fig,
            "Lampiran A – Data Penelitian: Daftar Perusahaan Sampel",
            f"Total Sampel: {n_total} Perusahaan  |  Tahun: 2023–2024",
            "LAMPIRAN A.1")
        draw_footer(fig, 1, 3)

        gs = GridSpec(2, 3, figure=fig, top=0.91, bottom=0.04,
                      left=0.04, right=0.98, hspace=0.5, wspace=0.35)

        # --- Stat cards ---
        stats = [
            ("Total Perusahaan Sampel", str(n_total), C_GOLD),
            ("Potential Fraud (SP2)", str(n_fraud), C_RED),
            ("Non-Fraud", str(n_non), C_GREEN),
        ]
        for k, (label, val, color) in enumerate(stats):
            ax_card = fig.add_subplot(gs[0, k])
            ax_card.set_facecolor(C_NAVY)
            for spine in ax_card.spines.values():
                spine.set_edgecolor(color); spine.set_linewidth(2)
            ax_card.set_xticks([]); ax_card.set_yticks([])
            ax_card.text(0.5, 0.62, val, fontsize=36, fontweight="bold",
                         color=color, ha="center", va="center",
                         transform=ax_card.transAxes)
            ax_card.text(0.5, 0.22, label, fontsize=9, color=C_LIGHT,
                         ha="center", va="center", transform=ax_card.transAxes)

        # --- Pie chart ---
        ax_pie = fig.add_subplot(gs[1, 0])
        wedge_sizes = [n_fraud, n_non]
        wedge_labels = ["Potential Fraud\n(SP2)", "Non-Fraud"]
        wedge_colors = [C_RED, C_GREEN]
        wedges, texts, autotexts = ax_pie.pie(
            wedge_sizes, labels=wedge_labels, colors=wedge_colors,
            autopct="%1.1f%%", startangle=90,
            textprops={"color": C_LIGHT, "fontsize": 8},
            wedgeprops={"edgecolor": C_DARK, "linewidth": 1.5})
        for at in autotexts:
            at.set_color(C_WHITE); at.set_fontsize(9); at.set_fontweight("bold")
        ax_pie.set_title("Distribusi Kelas", color=C_GOLD, fontsize=10, pad=8)

        # --- Variable distribution bar chart ---
        ax_bar = fig.add_subplot(gs[1, 1:])
        vars_of_interest = ["DSRI", "GMI", "AQI", "SGI", "DEPI", "SGAI", "LVGI", "TATA"]
        fraud_means = df_ml[df_ml["FLAG POTENTIAL FRAUD"]==1][vars_of_interest].mean()
        non_fraud_means = df_ml[df_ml["FLAG POTENTIAL FRAUD"]==0][vars_of_interest].mean()

        x = np.arange(len(vars_of_interest))
        w = 0.35
        bars1 = ax_bar.bar(x - w/2, fraud_means.values, w, label="Potential Fraud",
                           color=C_RED, alpha=0.85, edgecolor=C_DARK)
        bars2 = ax_bar.bar(x + w/2, non_fraud_means.values, w, label="Non-Fraud",
                           color=C_GREEN, alpha=0.85, edgecolor=C_DARK)
        ax_bar.set_xticks(x); ax_bar.set_xticklabels(vars_of_interest, fontsize=8)
        ax_bar.set_title("Rata-rata Komponen Beneish M-Score per Kelas", color=C_GOLD, fontsize=10)
        ax_bar.set_ylabel("Nilai Rata-rata", fontsize=8)
        ax_bar.legend(fontsize=8, facecolor=C_NAVY, edgecolor=C_BLUE)
        ax_bar.grid(axis="y", alpha=0.3)
        ax_bar.axhline(y=1.0, color=C_MUTED, linestyle="--", linewidth=0.8, alpha=0.6)

        pdf.savefig(fig, facecolor=C_DARK)
        plt.close(fig)

        # ── Page 2: Daftar Perusahaan Sampel (table) ───────────────────────
        MAX_PER_PAGE = 35
        chunks = [company_df.iloc[i:i+MAX_PER_PAGE]
                  for i in range(0, len(company_df), MAX_PER_PAGE)]
        total_pages = len(chunks) + 2  # page 1 already done, +1 for preview

        for page_idx, chunk in enumerate(chunks):
            fig2 = plt.figure(figsize=(11.69, 8.27))
            draw_page_header(fig2,
                "Lampiran A.2 – Daftar Perusahaan Sampel",
                f"Bagian {page_idx+1}/{len(chunks)}",
                "LAMPIRAN A.2")
            draw_footer(fig2, page_idx+2, total_pages)

            ax_tbl = fig2.add_axes([0.02, 0.04, 0.96, 0.86])
            row_clrs = []
            for _, row in chunk.iterrows():
                clr = "#2E1A1E" if row["Status"] == "Potential Fraud" else "#1A2E1E"
                row_clrs.append(clr)

            display_chunk = chunk.copy()
            display_chunk.index = range(
                page_idx*MAX_PER_PAGE + 1,
                page_idx*MAX_PER_PAGE + len(chunk) + 1)
            display_chunk.insert(0, "No.", display_chunk.index)
            display_chunk = display_chunk.reset_index(drop=True)

            styled_table(ax_tbl, display_chunk, fontsize=7,
                         row_colors=row_clrs,
                         col_aligns=["center", "center", "left", "left", "center"])

            # Legend
            legend_handles = [
                mpatches.Patch(color="#2E1A1E", label="Potential Fraud (SP2)"),
                mpatches.Patch(color="#1A2E1E", label="Non-Fraud"),
            ]
            fig2.legend(handles=legend_handles, loc="lower right",
                        fontsize=7, facecolor=C_NAVY, edgecolor=C_BLUE,
                        framealpha=0.8, bbox_to_anchor=(0.98, 0.04))

            pdf.savefig(fig2, facecolor=C_DARK)
            plt.close(fig2)

        # ── Page: Dataset Preview ───────────────────────────────────────────
        fig3 = plt.figure(figsize=(11.69, 8.27))
        draw_page_header(fig3,
            "Lampiran A.3 – Ringkasan Dataset (20 Baris Pertama)",
            "Setelah Pra-Pemrosesan (Beneish M-Score + Tone MD&A + Volatilitas)",
            "LAMPIRAN A.3")
        draw_footer(fig3, total_pages, total_pages)

        preview_display = preview[[
            "DSRI","GMI","AQI","SGI","DEPI","SGAI","LVGI","TATA",
            "M_Score","VolatilitasD-30","Negative_Tone","Positive_Tone",
            "Subjectivity_Ratio","Label"
        ]].copy()
        # Round numerics
        num_cols = preview_display.select_dtypes(include="number").columns
        preview_display[num_cols] = preview_display[num_cols].round(4)

        ax_prev = fig3.add_axes([0.01, 0.03, 0.98, 0.87])
        row_clrs3 = []
        for _, row in preview.iterrows():
            row_clrs3.append("#2E1A1E" if row["FLAG POTENTIAL FRAUD"]==1 else "#1A2E1E")

        styled_table(ax_prev, preview_display.reset_index(drop=True),
                     fontsize=5.8, row_colors=row_clrs3)

        pdf.savefig(fig3, facecolor=C_DARK)
        plt.close(fig3)

    print(f"  → Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  LAMPIRAN B – Bukti / Kriteria SP2
# ══════════════════════════════════════════════════════════════════════════════

def build_lampiran_b():
    print("[Lampiran B] Membaca data sanksi …")

    df_sanksi = pd.read_excel(
        os.path.join(OLAH, "publikasi-data-sanksi-ind(1).xlsx"), sheet_name="Sanksi")

    # ── SP2 only, in research period 2023-2024 ─────────────────────────────
    df_sp2 = df_sanksi[df_sanksi["Jenis Sanksi"] == "SP2"].copy()
    df_sp2["Tahun"] = df_sp2["Tahun"].astype(int)
    df_sp2_period = df_sp2[df_sp2["Tahun"].isin([2023, 2024])].copy()
    df_sp2_period["Tgl Surat"] = pd.to_datetime(df_sp2_period["Tgl Surat"]).dt.strftime("%d-%m-%Y")

    # Unique tickers with SP2
    sp2_unique = (df_sp2_period.groupby("Kode")
                               .agg(Jumlah_SP2=("Jenis Sanksi","count"),
                                    Kewajiban=("Jenis Kewajiban",
                                               lambda x: x.mode()[0] if len(x)>0 else "-"))
                               .reset_index()
                               .sort_values("Kode"))

    total_sanksi   = len(df_sanksi)
    total_sp2      = len(df_sp2_period)
    unique_emiten  = df_sp2_period["Kode"].nunique()

    # Distribution by year
    year_dist = df_sp2_period["Tahun"].value_counts().sort_index()

    # Distribution by kewajiban
    kwj_dist = (df_sp2_period["Jenis Kewajiban"]
                .value_counts().head(10).reset_index())
    kwj_dist.columns = ["Jenis Kewajiban", "Jumlah"]

    out_path = os.path.join(OUT_DIR, "Lampiran_B_Bukti_SP2.pdf")
    with PdfPages(out_path) as pdf:

        # ── Page 1: Overview ───────────────────────────────────────────────
        fig = plt.figure(figsize=(11.69, 8.27))
        draw_page_header(fig,
            "Lampiran B – Bukti Kriteria Potential Fraud: Sanksi SP2 BEI",
            "Sumber: Bursa Efek Indonesia (BEI) – Publikasi Data Sanksi",
            "LAMPIRAN B.1")
        draw_footer(fig, 1, 3)

        gs = GridSpec(2, 3, figure=fig, top=0.90, bottom=0.05,
                      left=0.04, right=0.97, hspace=0.55, wspace=0.3)

        # Stat cards
        stat_list = [
            ("Total Record Sanksi\n(Semua Jenis)", f"{total_sanksi:,}", C_GOLD),
            ("Total Sanksi SP2\n(2023–2024)", f"{total_sp2:,}", C_ACCENT),
            ("Emiten Unik\nTerkena SP2", f"{unique_emiten}", C_BLUE),
        ]
        for k, (lbl, val, col) in enumerate(stat_list):
            ax_c = fig.add_subplot(gs[0, k])
            ax_c.set_facecolor(C_NAVY)
            for sp in ax_c.spines.values():
                sp.set_edgecolor(col); sp.set_linewidth(2)
            ax_c.set_xticks([]); ax_c.set_yticks([])
            ax_c.text(0.5, 0.62, val, fontsize=30, fontweight="bold",
                      color=col, ha="center", va="center",
                      transform=ax_c.transAxes)
            ax_c.text(0.5, 0.2, lbl, fontsize=8.5, color=C_LIGHT,
                      ha="center", va="center", transform=ax_c.transAxes,
                      multialignment="center")

        # Bar – SP2 per year
        ax_yr = fig.add_subplot(gs[1, 0])
        ax_yr.bar(year_dist.index.astype(str), year_dist.values,
                  color=[C_ACCENT, C_GOLD], edgecolor=C_DARK, width=0.5)
        ax_yr.set_title("Jumlah SP2 per Tahun", color=C_GOLD, fontsize=9)
        ax_yr.set_ylabel("Jumlah Sanksi", fontsize=8)
        ax_yr.grid(axis="y", alpha=0.3)
        for bar, val in zip(ax_yr.patches, year_dist.values):
            ax_yr.text(bar.get_x()+bar.get_width()/2, bar.get_height()+10,
                       str(val), ha="center", va="bottom", color=C_WHITE, fontsize=9)

        # Horizontal bar – Kewajiban distribution
        ax_kwj = fig.add_subplot(gs[1, 1:])
        colors_kwj = plt.cm.get_cmap("plasma")(np.linspace(0.2, 0.9, len(kwj_dist)))
        ax_kwj.barh(kwj_dist["Jenis Kewajiban"], kwj_dist["Jumlah"],
                    color=colors_kwj, edgecolor=C_DARK)
        ax_kwj.set_title("Top 10 Jenis Kewajiban Sanksi SP2", color=C_GOLD, fontsize=9)
        ax_kwj.set_xlabel("Jumlah Sanksi", fontsize=8)
        ax_kwj.grid(axis="x", alpha=0.3)
        ax_kwj.invert_yaxis()
        ax_kwj.tick_params(axis="y", labelsize=7)

        pdf.savefig(fig, facecolor=C_DARK)
        plt.close(fig)

        # ── Page 2: Tabel SP2 Unique Emiten ───────────────────────────────
        MAX_ROW = 40
        sp2_chunks = [sp2_unique.iloc[i:i+MAX_ROW]
                      for i in range(0, len(sp2_unique), MAX_ROW)]

        for idx, chunk in enumerate(sp2_chunks):
            fig2 = plt.figure(figsize=(11.69, 8.27))
            draw_page_header(fig2,
                "Lampiran B.2 – Daftar Emiten Terkena Sanksi SP2 (2023–2024)",
                f"Bagian {idx+1}/{len(sp2_chunks)}  |  Digunakan sebagai label 'Potential Fraud'",
                "LAMPIRAN B.2")
            draw_footer(fig2, idx+2, len(sp2_chunks)+1)

            display = chunk.copy().reset_index(drop=True)
            display.index = range(idx*MAX_ROW+1, idx*MAX_ROW+len(chunk)+1)
            display.insert(0, "No.", display.index)
            display = display.reset_index(drop=True)
            display.columns = ["No.", "Kode Emiten", "Jumlah SP2", "Jenis Kewajiban Dominan"]

            ax_t = fig2.add_axes([0.03, 0.04, 0.94, 0.86])
            row_c = ["#1E1A2E" if i%2==0 else "#221F38" for i in range(len(display))]
            styled_table(ax_t, display, fontsize=8, row_colors=row_c,
                         col_aligns=["center","center","center","left"])

            pdf.savefig(fig2, facecolor=C_DARK)
            plt.close(fig2)

    print(f"  → Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  LAMPIRAN C – Kamus Sentimen InSET
# ══════════════════════════════════════════════════════════════════════════════

def build_lampiran_c():
    print("[Lampiran C] Membaca leksikon InSET …")

    pos_df = pd.read_csv(os.path.join(INSET_DIR, "positive.tsv"),
                         sep="\t", header=0, names=["Kata", "Bobot"])
    neg_df = pd.read_csv(os.path.join(INSET_DIR, "negative.tsv"),
                         sep="\t", header=0, names=["Kata", "Bobot"])

    pos_df["Bobot"] = pd.to_numeric(pos_df["Bobot"], errors="coerce")
    neg_df["Bobot"] = pd.to_numeric(neg_df["Bobot"], errors="coerce")

    # Sort by weight
    pos_top = pos_df.sort_values("Bobot", ascending=False).head(50).reset_index(drop=True)
    neg_top = neg_df.sort_values("Bobot").head(50).reset_index(drop=True)

    total_pos = len(pos_df)
    total_neg = len(neg_df)

    out_path = os.path.join(OUT_DIR, "Lampiran_C_Leksikon_InSET.pdf")
    with PdfPages(out_path) as pdf:

        # ── Page 1: Overview + weight distribution ─────────────────────────
        fig = plt.figure(figsize=(11.69, 8.27))
        draw_page_header(fig,
            "Lampiran C – Leksikon Sentimen InSET (Indonesian Sentiment Lexicon)",
            "Sumber: Koto et al. (2020) – InSet: Indonesian Sentiment Lexicon",
            "LAMPIRAN C.1")
        draw_footer(fig, 1, 3)

        gs = GridSpec(2, 3, figure=fig, top=0.90, bottom=0.05,
                      left=0.04, right=0.97, hspace=0.5, wspace=0.3)

        # Stat cards
        stat_list = [
            ("Total Kata Positif\ndalam InSET", f"{total_pos:,}", C_GREEN),
            ("Total Kata Negatif\ndalam InSET", f"{total_neg:,}", C_RED),
            ("Total Kosakata\nLeksikon", f"{total_pos+total_neg:,}", C_GOLD),
        ]
        for k, (lbl, val, col) in enumerate(stat_list):
            ax_c = fig.add_subplot(gs[0, k])
            ax_c.set_facecolor(C_NAVY)
            for sp in ax_c.spines.values(): sp.set_edgecolor(col); sp.set_linewidth(2)
            ax_c.set_xticks([]); ax_c.set_yticks([])
            ax_c.text(0.5, 0.62, val, fontsize=30, fontweight="bold",
                      color=col, ha="center", va="center", transform=ax_c.transAxes)
            ax_c.text(0.5, 0.2, lbl, fontsize=8.5, color=C_LIGHT,
                      ha="center", va="center", transform=ax_c.transAxes,
                      multialignment="center")

        # Histogram positive weights
        ax_ph = fig.add_subplot(gs[1, 0])
        pos_df["Bobot"].dropna().hist(ax=ax_ph, bins=10, color=C_GREEN,
                                      edgecolor=C_DARK, alpha=0.8)
        ax_ph.set_title("Distribusi Bobot Kata Positif", color=C_GOLD, fontsize=9)
        ax_ph.set_xlabel("Bobot"); ax_ph.set_ylabel("Frekuensi")
        ax_ph.grid(axis="y", alpha=0.3)

        # Histogram negative weights
        ax_nh = fig.add_subplot(gs[1, 1])
        neg_df["Bobot"].dropna().hist(ax=ax_nh, bins=10, color=C_RED,
                                      edgecolor=C_DARK, alpha=0.8)
        ax_nh.set_title("Distribusi Bobot Kata Negatif", color=C_GOLD, fontsize=9)
        ax_nh.set_xlabel("Bobot"); ax_nh.set_ylabel("Frekuensi")
        ax_nh.grid(axis="y", alpha=0.3)

        # Combined donut
        ax_dn = fig.add_subplot(gs[1, 2])
        wedge_s = [total_pos, total_neg]
        wedge_l = [f"Positif\n({total_pos:,})", f"Negatif\n({total_neg:,})"]
        wedge_c = [C_GREEN, C_RED]
        ax_dn.pie(wedge_s, labels=wedge_l, colors=wedge_c,
                  autopct="%1.1f%%", startangle=90,
                  wedgeprops={"width":0.5, "edgecolor":C_DARK,"linewidth":1.5},
                  textprops={"color":C_LIGHT,"fontsize":8})
        ax_dn.set_title("Proporsi Kosa Kata", color=C_GOLD, fontsize=9)

        pdf.savefig(fig, facecolor=C_DARK)
        plt.close(fig)

        # ── Page 2: Top-50 Positive ────────────────────────────────────────
        fig2 = plt.figure(figsize=(11.69, 8.27))
        draw_page_header(fig2,
            "Lampiran C.2 – Sampel 50 Kata Berbobot Tertinggi (Positif)",
            "Diurutkan dari Bobot Tertinggi – Digunakan untuk Ekstraksi Positive Tone MD&A",
            "LAMPIRAN C.2")
        draw_footer(fig2, 2, 3)

        # Split into two columns
        half = 25
        left_p  = pos_top.iloc[:half].reset_index(drop=True)
        right_p = pos_top.iloc[half:].reset_index(drop=True)
        left_p.index  = range(1, half+1)
        right_p.index = range(half+1, 51)
        left_p.insert(0, "No.", left_p.index); left_p = left_p.reset_index(drop=True)
        right_p.insert(0, "No.", right_p.index); right_p = right_p.reset_index(drop=True)

        ax_l = fig2.add_axes([0.02, 0.04, 0.46, 0.87])
        styled_table(ax_l, left_p, fontsize=8,
                     header_bg="#1A4E1A",
                     row_colors=["#1A2E1A" if i%2==0 else "#1E341E" for i in range(half)])

        ax_r = fig2.add_axes([0.52, 0.04, 0.46, 0.87])
        styled_table(ax_r, right_p, fontsize=8,
                     header_bg="#1A4E1A",
                     row_colors=["#1A2E1A" if i%2==0 else "#1E341E" for i in range(half)])

        pdf.savefig(fig2, facecolor=C_DARK)
        plt.close(fig2)

        # ── Page 3: Top-50 Negative ────────────────────────────────────────
        fig3 = plt.figure(figsize=(11.69, 8.27))
        draw_page_header(fig3,
            "Lampiran C.3 – Sampel 50 Kata Berbobot Terendah (Negatif)",
            "Diurutkan dari Bobot Terendah – Digunakan untuk Ekstraksi Negative Tone MD&A",
            "LAMPIRAN C.3")
        draw_footer(fig3, 3, 3)

        left_n  = neg_top.iloc[:half].reset_index(drop=True)
        right_n = neg_top.iloc[half:].reset_index(drop=True)
        left_n.index  = range(1, half+1)
        right_n.index = range(half+1, 51)
        left_n.insert(0, "No.", left_n.index); left_n = left_n.reset_index(drop=True)
        right_n.insert(0, "No.", right_n.index); right_n = right_n.reset_index(drop=True)

        ax_l3 = fig3.add_axes([0.02, 0.04, 0.46, 0.87])
        styled_table(ax_l3, left_n, fontsize=8,
                     header_bg="#4E1A1A",
                     row_colors=["#2E1A1A" if i%2==0 else "#341E1E" for i in range(half)])

        ax_r3 = fig3.add_axes([0.52, 0.04, 0.46, 0.87])
        styled_table(ax_r3, right_n, fontsize=8,
                     header_bg="#4E1A1A",
                     row_colors=["#2E1A1A" if i%2==0 else "#341E1E" for i in range(half)])

        pdf.savefig(fig3, facecolor=C_DARK)
        plt.close(fig3)

    print(f"  → Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  LAMPIRAN D – Hasil Perbandingan 63 Skenario
# ══════════════════════════════════════════════════════════════════════════════

def build_lampiran_d():
    print("[Lampiran D] Membaca hasil perbandingan …")

    df = pd.read_excel(
        os.path.join(MODEL_DIR, "Hasil_Perbandingan_63_Skenario.xlsx"))

    df.columns = [c.strip() for c in df.columns]
    metric_cols = ["AUC", "CA", "F1", "Prec", "Recall", "MCC"]
    for col in metric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Best model overall
    best_row = df.loc[df["AUC"].idxmax()]

    # Best per model
    best_per_model = (df.groupby("Model")[metric_cols]
                        .max().reset_index()
                        .sort_values("AUC", ascending=False))

    # Best per skenario
    best_per_ske = (df.groupby("Skenario Fitur")[metric_cols]
                      .max().reset_index()
                      .sort_values("AUC", ascending=False))

    out_path = os.path.join(OUT_DIR, "Lampiran_D_Hasil_63_Skenario.pdf")
    with PdfPages(out_path) as pdf:

        # ── Page 1: Summary + top charts ──────────────────────────────────
        fig = plt.figure(figsize=(11.69, 8.27))
        draw_page_header(fig,
            "Lampiran D – Hasil Perbandingan 63 Skenario Model Machine Learning",
            f"Best Model: {best_row['Model']} | {best_row['Skenario Fitur']} | AUC={best_row['AUC']:.4f}",
            "LAMPIRAN D.1")
        draw_footer(fig, 1, 3)

        gs = GridSpec(2, 3, figure=fig, top=0.90, bottom=0.05,
                      left=0.04, right=0.97, hspace=0.55, wspace=0.3)

        # Stat cards
        stat_list = [
            ("Jumlah Skenario\nDiuji", str(len(df)), C_GOLD),
            (f"AUC Terbaik\n({best_row['Model']})", f"{best_row['AUC']:.4f}", C_ACCENT),
            (f"F1 Terbaik\n({best_row['Model']})", f"{best_row['F1']:.4f}", C_GREEN),
        ]
        for k, (lbl, val, col) in enumerate(stat_list):
            ax_c = fig.add_subplot(gs[0, k])
            ax_c.set_facecolor(C_NAVY)
            for sp in ax_c.spines.values(): sp.set_edgecolor(col); sp.set_linewidth(2)
            ax_c.set_xticks([]); ax_c.set_yticks([])
            ax_c.text(0.5, 0.62, val, fontsize=28, fontweight="bold",
                      color=col, ha="center", va="center", transform=ax_c.transAxes)
            ax_c.text(0.5, 0.2, lbl, fontsize=8.5, color=C_LIGHT,
                      ha="center", va="center", transform=ax_c.transAxes,
                      multialignment="center")

        # AUC grouped bar per skenario
        ax_ske = fig.add_subplot(gs[1, :2])
        pivot = df.pivot_table(index="Skenario Fitur", columns="Model",
                               values="AUC", aggfunc="max")
        models = pivot.columns.tolist()
        x = np.arange(len(pivot))
        w = 0.8 / len(models)
        cmap = plt.cm.get_cmap("tab10")
        for mi, model in enumerate(models):
            ax_ske.bar(x + mi*w, pivot[model].values, w, label=model,
                       color=cmap(mi), alpha=0.85, edgecolor=C_DARK)
        ax_ske.set_xticks(x + w*(len(models)-1)/2)
        ax_ske.set_xticklabels([textwrap.fill(l, 12) for l in pivot.index],
                               fontsize=6.5)
        ax_ske.set_title("AUC per Skenario Fitur", color=C_GOLD, fontsize=9)
        ax_ske.set_ylabel("AUC"); ax_ske.grid(axis="y", alpha=0.3)
        ax_ske.legend(fontsize=6, facecolor=C_NAVY, edgecolor=C_BLUE,
                      ncol=2, loc="upper right")
        ax_ske.axhline(0.5, color=C_MUTED, linestyle="--", linewidth=0.7)

        # Radar / spider chart for best model
        ax_rad = fig.add_subplot(gs[1, 2], polar=True)
        categories = metric_cols
        N = len(categories)
        vals = [best_row[m] for m in categories]
        vals += vals[:1]
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        ax_rad.set_facecolor(C_NAVY)
        ax_rad.plot(angles, vals, "o-", linewidth=2, color=C_ACCENT)
        ax_rad.fill(angles, vals, alpha=0.25, color=C_ACCENT)
        ax_rad.set_xticks(angles[:-1])
        ax_rad.set_xticklabels(categories, size=8, color=C_LIGHT)
        ax_rad.set_ylim(0, 1)
        ax_rad.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax_rad.set_yticklabels(["0.2","0.4","0.6","0.8","1.0"], size=6, color=C_MUTED)
        ax_rad.grid(color=C_BLUE, alpha=0.4)
        ax_rad.set_title(f"Performa\n{best_row['Model']}", color=C_GOLD, fontsize=9, pad=15)

        pdf.savefig(fig, facecolor=C_DARK)
        plt.close(fig)

        # ── Page 2: Full Table of 63 Scenarios ────────────────────────────
        MAX_R = 22
        df_sorted = df.sort_values("AUC", ascending=False).reset_index(drop=True)
        chunks = [df_sorted.iloc[i:i+MAX_R] for i in range(0, len(df_sorted), MAX_R)]
        total_p = len(chunks) + 1

        for idx, chunk in enumerate(chunks):
            fig2 = plt.figure(figsize=(11.69, 8.27))
            draw_page_header(fig2,
                "Lampiran D.2 – Tabel Lengkap 63 Skenario (Diurutkan berdasarkan AUC)",
                f"Bagian {idx+1}/{len(chunks)}",
                "LAMPIRAN D.2")
            draw_footer(fig2, idx+2, total_p)

            display = chunk.copy().reset_index(drop=True)
            display.index = range(idx*MAX_R+1, idx*MAX_R+len(chunk)+1)
            display.insert(0, "No.", display.index)
            display = display.reset_index(drop=True)
            # Round metrics
            for mc in metric_cols:
                display[mc] = display[mc].round(4)

            ax_t = fig2.add_axes([0.01, 0.04, 0.98, 0.86])
            # Highlight top-3 rows
            row_clrs = []
            for i in range(len(display)):
                global_rank = idx*MAX_R + i + 1
                if global_rank == 1:
                    row_clrs.append("#1A2E3A")
                elif global_rank <= 3:
                    row_clrs.append("#1A2A30")
                elif i % 2 == 0:
                    row_clrs.append(C_NAVY)
                else:
                    row_clrs.append("#1A2040")
            styled_table(ax_t, display, fontsize=7.5, row_colors=row_clrs)

            if idx == 0:
                fig2.text(0.5, 0.035,
                          "★ Baris berwarna biru lebih terang = Skenario terbaik (Top-3 AUC)",
                          ha="center", va="center", color=C_GOLD, fontsize=7,
                          transform=fig2.transFigure)

            pdf.savefig(fig2, facecolor=C_DARK)
            plt.close(fig2)

        # ── Page 3: Heat map – model vs skenario (AUC) ────────────────────
        fig3 = plt.figure(figsize=(11.69, 8.27))
        draw_page_header(fig3,
            "Lampiran D.3 – Heatmap AUC: Model vs Skenario Fitur",
            "Warna lebih terang = AUC lebih tinggi",
            "LAMPIRAN D.3")
        draw_footer(fig3, total_p, total_p)

        pivot_heat = df.pivot_table(index="Model", columns="Skenario Fitur",
                                    values="AUC", aggfunc="max")
        ax_hm = fig3.add_axes([0.12, 0.12, 0.84, 0.74])
        im = ax_hm.imshow(pivot_heat.values, cmap="YlOrRd", aspect="auto",
                          vmin=0.35, vmax=0.75)
        ax_hm.set_xticks(range(len(pivot_heat.columns)))
        ax_hm.set_xticklabels([textwrap.fill(c, 14) for c in pivot_heat.columns],
                               fontsize=7, color=C_LIGHT, rotation=20, ha="right")
        ax_hm.set_yticks(range(len(pivot_heat.index)))
        ax_hm.set_yticklabels(pivot_heat.index, fontsize=8.5, color=C_LIGHT)
        ax_hm.set_facecolor(C_NAVY)

        # Annotate cells
        for r in range(pivot_heat.shape[0]):
            for c in range(pivot_heat.shape[1]):
                val = pivot_heat.values[r, c]
                if not np.isnan(val):
                    ax_hm.text(c, r, f"{val:.3f}", ha="center", va="center",
                               fontsize=7, color="black" if val > 0.58 else C_WHITE,
                               fontweight="bold" if val == pivot_heat.values.max() else "normal")

        plt.colorbar(im, ax=ax_hm, shrink=0.7, pad=0.02,
                     label="AUC").ax.yaxis.label.set_color(C_LIGHT)
        ax_hm.set_title("Heatmap AUC – Model vs Skenario Fitur",
                         color=C_GOLD, fontsize=11, pad=10)

        pdf.savefig(fig3, facecolor=C_DARK)
        plt.close(fig3)

    print(f"  → Saved: {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  Generating Lampiran PDF – Skripsi Fraud Detection ML")
    print("=" * 60)

    build_lampiran_a()
    build_lampiran_b()
    build_lampiran_c()
    build_lampiran_d()

    print()
    print("=" * 60)
    print(f"  Semua lampiran selesai! Output di:")
    print(f"  {OUT_DIR}")
    print("=" * 60)
