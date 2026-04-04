import pandas as pd
import os
import re
import pdfplumber


# 1. FUNGSI LOAD LEKSIKON INSET (Dengan proteksi tipe data)
def load_inset(folder_path):
    neg_path = os.path.join(folder_path, 'negative.tsv')
    pos_path = os.path.join(folder_path, 'positive.tsv')

    # Membaca file TSV
    df_neg = pd.read_csv(neg_path, sep='\t', header=None, names=['word', 'weight'])
    df_pos = pd.read_csv(pos_path, sep='\t', header=None, names=['word', 'weight'])

    # Memastikan skor adalah angka (Solusi untuk TypeError sebelumnya)
    df_neg['weight'] = pd.to_numeric(df_neg['weight'], errors='coerce')
    df_pos['weight'] = pd.to_numeric(df_pos['weight'], errors='coerce')

    # Menghapus data kosong jika ada
    df_neg = df_neg.dropna(subset=['weight'])
    df_pos = df_pos.dropna(subset=['weight'])

    # Gabungkan ke dictionary
    inset_dict = dict(zip(df_neg['word'], df_neg['weight']))
    inset_dict.update(dict(zip(df_pos['word'], df_pos['weight'])))
    return inset_dict


# 2. FUNGSI ANALISIS SENTIMEN & HITUNG PER KATA
def analyze_sentiment(text, inset_dict, ticker):
    if not text:
        return 0, 0, 0, []

    text_clean = text.lower()
    words = re.findall(r'\b\w+\b', text_clean)

    pos_score = 0
    neg_score = 0
    total_score = 0
    match_count = 0
    word_stats = {}

    for word in words:
        if word in inset_dict:
            score = inset_dict[word]
            if score > 0:
                pos_score += score
            else:
                neg_score += score
            total_score += score
            match_count += 1

            # Hitung frekuensi per kata
            if word not in word_stats:
                word_stats[word] = {'jumlah': 0, 'skor_satuan': score}
            word_stats[word]['jumlah'] += 1

    # Format detail per kata untuk Excel
    word_details = []
    for w, stat in word_stats.items():
        word_details.append({
            'Ticker': ticker,
            'Kata': w,
            'Jumlah': stat['jumlah'],
            'Skor_Satuan': stat['skor_satuan'],
            'Total_Score': stat['jumlah'] * stat['skor_satuan']
        })

    avg_score = total_score / match_count if match_count > 0 else 0
    return pos_score, neg_score, avg_score, word_details


# 3. FUNGSI EKSTRAK HALAMAN
def extract_pages(file_path, start_page, end_page):
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            # Proteksi jika halaman yang diminta melebihi total halaman PDF
            last_page = min(end_page, len(pdf.pages))
            for i in range(start_page - 1, last_page):
                page_text = pdf.pages[i].extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"Gagal mengekstrak {file_path}: {e}")
    return text


# --- PROSES UTAMA ---

# A. Persiapan Leksikon
inset_lexicon = load_inset('inset')

# B. Daftar Kerja
job_list = [
    {
        "ticker": "TEST",
        "file": "test.pdf",
        "start": 65,
        "end": 92
    },
    # Tambahkan perusahaan lain di sini
]

summary_results = []
all_word_details = []
full_texts = []

for job in job_list:
    print(f"Sedang memproses {job['ticker']}...")

    # 1. Ekstrak teks utuh
    raw_text = extract_pages(job['file'], job['start'], job['end'])

    # 2. Analisis Sentimen & Detail Kata
    p_sum, n_sum, avg_s, word_details = analyze_sentiment(raw_text, inset_lexicon, job['ticker'])

    # 3. Simpan Ringkasan Skor
    summary_results.append({
        "Ticker": job['ticker'],
        "Positive_Sum": p_sum,
        "Negative_Sum": n_sum,
        "Sentiment_Avg": avg_s,
        "Total_Matched_Words": len(word_details)
    })

    # 4. Simpan Seluruh Teks Ekstraksi
    full_texts.append({
        "Ticker": job['ticker'],
        "Halaman_Awal": job['start'],
        "Halaman_Akhir": job['end'],
        "Full_MDNA_Text": raw_text
    })

    # 5. Gabungkan Detail Kata
    all_word_details.extend(word_details)

# --- SIMPAN KE EXCEL DENGAN 3 SHEET ---
output_file = "Hasil_Ekstraksi_dan_Analisis_Lengkap.xlsx"

with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
    # Sheet 1: Ringkasan Statistik
    pd.DataFrame(summary_results).to_excel(writer, sheet_name='Summary_Scores', index=False)

    # Sheet 2: Rincian Kata (Kata || Jumlah || Score)
    pd.DataFrame(all_word_details).to_excel(writer, sheet_name='Word_Details', index=False)

    # Sheet 3: Seluruh Hasil Ekstrak Teks MD&A
    pd.DataFrame(full_texts).to_excel(writer, sheet_name='Full_Extracted_Text', index=False)

print(f"\nSelesai! Seluruh data (termasuk teks utuh) telah disimpan di: {output_file}")