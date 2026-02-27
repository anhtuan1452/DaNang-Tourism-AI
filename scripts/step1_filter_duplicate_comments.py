"""
step1_filter_duplicate_comments.py
====================================
Tiền xử lý & lọc comment trùng lặp trong dữ liệu đánh giá khách sạn Đà Nẵng.
Mục đích: Báo cáo nghiên cứu – minh chứng bước tiền xử lý dữ liệu thô.

Lưu ý ngôn ngữ:
  Toàn bộ dữ liệu review được thu thập từ TripAdvisor và viết bằng TIẾNG ANH.
  Do đó KHÔNG áp dụng tách từ / xử lý NLP tiếng Việt. Các bước chuẩn hoá
  chỉ tập trung vào: Unicode NFC, control characters, whitespace normalization.

Các bước xử lý:
  0. Chuẩn hoá chuỗi ký tự (String Normalization)
       – Strip / trim khoảng trắng đầu-cuối
       – Chuẩn hoá khoảng trắng in-giữa (tab, newline → space)
       – Chuẩn hoá Unicode NFC (ký tự Latin nhất quán)
       – Xoá ký tự điều khiển (control characters)
  1. Trùng hoàn toàn (exact duplicate)  – cùng reviewId
  2. Trùng nội dung văn bản              – text giống hệt nhau, khác reviewId
  3. Quasi-duplicate (gần giống)         – TF-IDF cosine similarity ≥ ngưỡng
  4. Cross-source duplicate              – cùng (text, locationId) xuất hiện ở nhiều file

Đầu ra:
  - data/processed/reviews_deduped.csv    – bộ dữ liệu đã lọc
  - scripts/duplicate_report_full.txt     – báo cáo chi tiết
"""

import os
import re
import unicodedata
import pandas as pd
import numpy as np
from datetime import datetime

# ─── CẤU HÌNH ─────────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NCKH_DIR    = os.path.dirname(PROJECT_DIR)   # e:\Ky 1 nam 4\NCKH

RAW_FILES = [
    os.path.join(PROJECT_DIR, 'data', 'raw', 'absa_deepseek_results_merged_backup.csv'),
    os.path.join(PROJECT_DIR, 'data', 'raw', 'absa_deepseek_results_backup.csv'),
]

OUTPUT_CSV    = os.path.join(PROJECT_DIR, 'data', 'processed', 'reviews_deduped.csv')
OUTPUT_REPORT = os.path.join(PROJECT_DIR, 'scripts', 'duplicate_report_full.txt')

# Ngưỡng cosine similarity để coi là "quasi-duplicate"
QUASI_THRESHOLD = 0.90
# Số lượng mẫu tối đa để chạy quasi-duplicate (tránh quá lâu)
QUASI_SAMPLE_LIMIT = 50_000

# Các cột cần thiết
TEXT_COL     = 'reviewText'   # cột nội dung comment
ID_COL       = 'id'           # reviewId (unique per review)
LOCATION_COL = 'locationId'
DATE_COL     = 'createdDate'
RATING_COL   = 'rating'


# ─── HÀM TIỆN ÍCH ─────────────────────────────────────────────────────────────

def _normalize_str(val) -> str:
    """Chuẩn hoá một giá trị string: strip, whitespace, Unicode NFC, control chars."""
    if not isinstance(val, str):
        return val                          # giữ nguyên nếu không phải string
    # 1. Unicode NFC – tiếng Việt nhất quán
    val = unicodedata.normalize('NFC', val)
    # 2. Xoá ký tự điều khiển (ASCII 0-8, 11-12, 14-31, 127)
    val = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', val)
    # 3. Chuẩn hoá khoảng trắng in-giữa (tab / newline → space)
    val = re.sub(r'[ \t\r\n]+', ' ', val)
    # 4. Strip đầu-cuối
    val = val.strip()
    return val


def normalize_whitespace(df: pd.DataFrame, f) -> pd.DataFrame:
    """
    Bước 0 – Chuẩn hoá toàn bộ cột kiểu string trong DataFrame.

    LƯU Ý: Dữ liệu review là TIẾNG ANH (thu thập từ TripAdvisor).
    Không áp dụng tách từ hay xử lý NLP tiếng Việt.
    Chỉ thực hiện chuẩn hoá ký tự cơ bản:
      - Unicode NFC  : đảm bảo ký tự Latin có dấu nhất quán
      - Control chars: xoá ký tự điều khiển ẩn
      - Whitespace   : tab/newline → space, strip đầu-cuối
    """
    border = '=' * 70
    f.write(f"\n{border}\n  0. CHUẨN HOÁ CHUỖI KÝ TỰ (String / Whitespace Normalization)\n{border}\n")
    f.write("  [Ghi chú] Dữ liệu review là TIẾNG ANH – không áp dụng NLP tiếng Việt.\n")
    f.write("           Chỉ chuẩn hoá Unicode NFC, control chars, whitespace.\n\n")

    str_cols = df.select_dtypes(include='object').columns.tolist()
    stats = []

    for col in str_cols:
        original = df[col].copy()
        df[col]  = df[col].apply(_normalize_str)
        # Đếm số ô thực sự thay đổi
        changed = (original.astype(str) != df[col].astype(str)).sum()
        if changed > 0:
            stats.append((col, changed))

    f.write(f"  Tổng số cột string được kiểm tra : {len(str_cols)}\n")
    f.write(f"  Số cột có ô bị thay đổi          : {len(stats)}\n\n")

    if stats:
        f.write("  Chi tiết các cột bị thay đổi:\n")
        for col, cnt in sorted(stats, key=lambda x: -x[1]):
            f.write(f"    {col:<35s}: {cnt:>8,} ô được chuẩn hoá\n")

    # Riêng cột reviewText – thống kê độ dài trước / sau
    if TEXT_COL in df.columns:
        # (df đã normalize ở trên, so sánh lại với original không còn, nên chỉ in sau)
        lengths = df[TEXT_COL].dropna().str.len()
        f.write(f"\n  Độ dài reviewText sau chuẩn hoá:\n")
        f.write(f"    Min    : {lengths.min():.0f} ký tự\n")
        f.write(f"    Mean   : {lengths.mean():.0f} ký tự\n")
        f.write(f"    Median : {lengths.median():.0f} ký tự\n")
        f.write(f"    Max    : {lengths.max():.0f} ký tự\n")

    f.write(f"\n  → Tổng dòng sau bước này: {len(df):,} (không xoá dòng, chỉ chuẩn hoá)\n")
    print(f"      Chuẩn hoá xong {len(stats)} cột, {sum(c for _,c in stats):,} ô được sửa.")
    return df


def clean_text(text: str) -> str:
    """Chuẩn hoá văn bản để so sánh: lower-case, bỏ khoảng trắng thừa."""
    if not isinstance(text, str):
        return ''
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def load_raw_data(file_paths: list) -> pd.DataFrame:
    """Đọc và ghép các file CSV thô."""
    dfs = []
    for path in file_paths:
        if os.path.exists(path):
            df = pd.read_csv(path, dtype={ID_COL: str, LOCATION_COL: str},
                             low_memory=False)
            df['_source_file'] = os.path.basename(path)
            dfs.append(df)
            print(f"  Đọc {len(df):,} dòng từ: {os.path.basename(path)}")
        else:
            print(f"  [CẢNH BÁO] Không tìm thấy file: {path}")
    if not dfs:
        raise FileNotFoundError("Không có file dữ liệu nào hợp lệ!")
    return pd.concat(dfs, ignore_index=True)


def report_section(f, title: str):
    """In tiêu đề phần trong báo cáo."""
    border = '=' * 70
    f.write(f"\n{border}\n  {title}\n{border}\n")


# ─── PHÂN TÍCH TRÙNG LẶP ──────────────────────────────────────────────────────

def analyze_exact_id_duplicates(df: pd.DataFrame, f) -> pd.DataFrame:
    """Loại bỏ trùng theo reviewId."""
    report_section(f, "1. TRÙNG LẶP THEO REVIEW ID (Exact ID Duplicate)")
    before = len(df)
    duped_ids = df[df.duplicated(subset=[ID_COL], keep=False)]
    n_duped_groups = duped_ids[ID_COL].nunique()
    n_duped_rows   = len(duped_ids) - n_duped_groups

    f.write(f"  - Tổng dòng trước lọc : {before:,}\n")
    f.write(f"  - Số ID xuất hiện > 1 lần : {n_duped_groups:,}\n")
    f.write(f"  - Số dòng trùng (bị xoá) : {n_duped_rows:,}\n\n")

    if n_duped_groups > 0:
        sample = (duped_ids.groupby(ID_COL)['_source_file']
                  .apply(lambda x: list(x)).reset_index()
                  .head(10))
        f.write("  Mẫu 10 ID trùng đầu tiên (xuất hiện từ file nào):\n")
        for _, row in sample.iterrows():
            f.write(f"    reviewId={row[ID_COL]}  ->  {row['_source_file']}\n")

    df = df.drop_duplicates(subset=[ID_COL], keep='first')
    f.write(f"\n  → Còn lại sau bước này: {len(df):,} dòng\n")
    return df


def analyze_text_duplicates(df: pd.DataFrame, f) -> pd.DataFrame:
    """Loại bỏ trùng theo nội dung văn bản (chuẩn hoá)."""
    report_section(f, "2. TRÙNG LẶP THEO NỘI DUNG VĂN BẢN (Text Duplicate)")
    df['_text_clean'] = df[TEXT_COL].apply(clean_text) if TEXT_COL in df.columns else ''
    before = len(df)

    # Loại bỏ các dòng text rỗng
    empty_mask = df['_text_clean'] == ''
    n_empty = empty_mask.sum()
    f.write(f"  - Số comment rỗng / không có text: {n_empty:,}\n")

    # Tìm duplicate văn bản
    text_duped = df[df.duplicated(subset=['_text_clean'], keep=False) & (~empty_mask)]
    n_text_groups = text_duped['_text_clean'].nunique()
    n_text_rows   = len(text_duped) - n_text_groups

    f.write(f"  - Số nhóm nội dung trùng nhau : {n_text_groups:,}\n")
    f.write(f"  - Số dòng trùng (bị xoá)      : {n_text_rows:,}\n\n")

    if n_text_groups > 0:
        # Ví dụ mẫu: top 5 đoạn text bị lặp nhiều nhất
        top_duped = (df[~empty_mask]
                     .groupby('_text_clean')
                     .size()
                     .reset_index(name='count')
                     .sort_values('count', ascending=False)
                     .head(5))
        f.write("  Top 5 nội dung bị lặp nhiều nhất:\n")
        for _, row in top_duped.iterrows():
            preview = row['_text_clean'][:100].replace('\n', ' ')
            f.write(f"    [{row['count']}x] \"{preview}...\"\n")

    # Giữ lại dòng đầu tiên cho mỗi text
    df = df[~df.duplicated(subset=['_text_clean'], keep='first') | empty_mask]
    f.write(f"\n  → Còn lại sau bước này: {len(df):,} dòng\n")
    return df


def analyze_cross_source_duplicates(df: pd.DataFrame, f) -> pd.DataFrame:
    """Phát hiện cross-source duplicate: (text, locationId) giống nhau."""
    report_section(f, "3. TRÙNG LẶP CHÉO NGUỒN DỮ LIỆU (Cross-Source Duplicate)")
    if LOCATION_COL not in df.columns:
        f.write("  [BỎ QUA] Không tìm thấy cột locationId.\n")
        return df

    key_cols = ['_text_clean', LOCATION_COL]
    duped = df[df.duplicated(subset=key_cols, keep=False)]
    f.write(f"  - Số dòng trùng (text + locationId): {len(duped):,}\n")

    if len(duped) > 0:
        cross = (duped.groupby(key_cols)['_source_file']
                 .apply(lambda x: ' | '.join(sorted(set(x))))
                 .reset_index()
                 .head(10))
        f.write("  Mẫu 10 trường hợp cross-source:\n")
        for _, row in cross.iterrows():
            preview = row['_text_clean'][:60].replace('\n', ' ')
            f.write(f"    locId={row[LOCATION_COL]}: \"{preview}...\"  ->  {row['_source_file']}\n")

    df = df.drop_duplicates(subset=key_cols, keep='first')
    f.write(f"\n  → Còn lại sau bước này: {len(df):,} dòng\n")
    return df


def analyze_quasi_duplicates(df: pd.DataFrame, f):
    """
    Phân tích quasi-duplicate bằng TF-IDF cosine similarity.
    Chỉ báo cáo (không xoá) vì quasi-duplicate có thể là review hợp lệ.
    """
    report_section(f, "4. QUASI-DUPLICATE (Cosine Similarity TF-IDF ≥ " + str(QUASI_THRESHOLD) + ")")
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        f.write("  [BỎ QUA] Cần cài scikit-learn: pip install scikit-learn\n")
        return

    texts = df['_text_clean'].dropna().tolist()
    n = min(len(texts), QUASI_SAMPLE_LIMIT)
    if n < 2:
        f.write("  [BỎ QUA] Không đủ dữ liệu văn bản.\n")
        return

    f.write(f"  Phân tích trên {n:,} mẫu (giới hạn tốc độ)...\n")
    sample_texts = texts[:n]

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2),
                                 min_df=2, sublinear_tf=True)
    try:
        tfidf_matrix = vectorizer.fit_transform(sample_texts)
    except ValueError as e:
        f.write(f"  [LỖI TF-IDF] {e}\n")
        return

    # Tính similarity theo batch để tiết kiệm RAM
    BATCH = 500
    quasi_pairs = []
    for i in range(0, n, BATCH):
        batch = tfidf_matrix[i:i+BATCH]
        sim = cosine_similarity(batch, tfidf_matrix)
        rows, cols = np.where(sim >= QUASI_THRESHOLD)
        for r, c in zip(rows, cols):
            global_r = i + r
            if global_r < c:   # tránh đếm hai lần
                quasi_pairs.append((global_r, c, float(sim[r, c])))
        if len(quasi_pairs) > 5000:
            break

    f.write(f"  - Số cặp quasi-duplicate phát hiện: {len(quasi_pairs):,}\n")
    if quasi_pairs:
        f.write("  Mẫu 10 cặp đầu tiên:\n")
        for r, c, sim_val in quasi_pairs[:10]:
            t1 = sample_texts[r][:60].replace('\n', ' ')
            t2 = sample_texts[c][:60].replace('\n', ' ')
            f.write(f"    sim={sim_val:.3f}  |  \"{t1}...\"  ≈  \"{t2}...\"\n")
    f.write("  [GHI CHÚ] Quasi-duplicate chỉ được báo cáo, không tự động xoá.\n")


# ─── THỐNG KÊ TỔNG QUAN ───────────────────────────────────────────────────────

def summary_statistics(df_raw: pd.DataFrame, df_clean: pd.DataFrame, f):
    """In bảng tóm tắt so sánh trước/sau lọc."""
    report_section(f, "5. THỐNG KÊ TỔNG QUAN")
    removed = len(df_raw) - len(df_clean)
    pct     = removed / len(df_raw) * 100 if len(df_raw) > 0 else 0

    f.write(f"  {'':30s} {'TRƯỚC':>12s}  {'SAU':>12s}\n")
    f.write(f"  {'-'*58}\n")
    f.write(f"  {'Tổng số comment':30s} {len(df_raw):>12,}  {len(df_clean):>12,}\n")
    f.write(f"  {'Đã loại bỏ':30s} {removed:>12,}  ({pct:.2f}%)\n")

    if LOCATION_COL in df_clean.columns:
        f.write(f"  {'Số địa điểm (locationId)':30s} "
                f"{df_raw[LOCATION_COL].nunique():>12,}  "
                f"{df_clean[LOCATION_COL].nunique():>12,}\n")
    if DATE_COL in df_clean.columns:
        df_clean_dates = pd.to_datetime(df_clean[DATE_COL], errors='coerce')
        df_raw_dates   = pd.to_datetime(df_raw[DATE_COL],   errors='coerce')
        f.write(f"  {'Khoảng thời gian':30s} "
                f"{str(df_raw_dates.min().date())} → {str(df_raw_dates.max().date())}\n")
        f.write(f"  {'(sau lọc)':30s} "
                f"{str(df_clean_dates.min().date())} → {str(df_clean_dates.max().date())}\n")
    if RATING_COL in df_clean.columns:
        f.write(f"  {'Rating trung bình (trước)':30s} {df_raw[RATING_COL].mean():>12.3f}\n")
        f.write(f"  {'Rating trung bình (sau)':30s} {df_clean[RATING_COL].mean():>12.3f}\n")

    # Phân phối theo năm
    if DATE_COL in df_clean.columns:
        df_clean_copy = df_clean.copy()
        df_clean_copy['year'] = pd.to_datetime(df_clean_copy[DATE_COL], errors='coerce').dt.year
        year_dist = df_clean_copy['year'].value_counts().sort_index()
        f.write("\n  Phân phối comment theo năm (sau lọc):\n")
        for year, cnt in year_dist.items():
            bar = '█' * int(cnt / year_dist.max() * 30)
            f.write(f"    {int(year)}: {cnt:>7,}  {bar}\n")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  PHÂN TÍCH & LỌC COMMENT TRÙNG LẶP")
    print("=" * 60)
    print(f"  Thời gian chạy: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # ── Đọc dữ liệu
    print("[1/6] Đọc dữ liệu thô...")
    df_raw = load_raw_data(RAW_FILES)
    print(f"      Tổng: {len(df_raw):,} dòng | {df_raw.shape[1]} cột\n")

    # ── Mở file báo cáo
    os.makedirs(os.path.dirname(OUTPUT_REPORT), exist_ok=True)
    with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
        f.write("PHÂN TÍCH COMMENT TRÙNG LẶP – DỰ ÁN DỰ BÁO DU LỊCH ĐÀ NẴNG\n")
        f.write(f"Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Nguồn dữ liệu: {', '.join(os.path.basename(p) for p in RAW_FILES)}\n")
        f.write(f"Tổng dòng ban đầu: {len(df_raw):,}\n")

        # ── Bước 0: Chuẩn hoá chuỗi (strip / trim whitespace / NFC)
        print("[2/6] Chuẩn hoá chuỗi ký tự (strip / whitespace / Unicode NFC)...")
        df = normalize_whitespace(df_raw.copy(), f)

        # ── Bước 1: Exact ID duplicate
        print("[3/6] Kiểm tra trùng Review ID...")
        df = analyze_exact_id_duplicates(df, f)

        # ── Bước 2: Text duplicate
        print("[4/6] Kiểm tra trùng nội dung văn bản...")
        df = analyze_text_duplicates(df, f)

        # ── Bước 3: Cross-source duplicate
        print("[5/6] Kiểm tra cross-source duplicate...")
        df = analyze_cross_source_duplicates(df, f)

        # ── Bước 4: Quasi-duplicate (phân tích thêm cho báo cáo)
        print("[5.5/6] Phân tích quasi-duplicate (TF-IDF)...")
        analyze_quasi_duplicates(df, f)

        # ── Thống kê tổng quan
        print("[6/6] Tổng kết...")
        summary_statistics(df_raw, df, f)
        f.write("\n\n[HOÀN THÀNH]\n")

    # ── Xoá cột tạm & lưu output
    df = df.drop(columns=['_text_clean', '_source_file'], errors='ignore')
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

    print(f"\n✓ Dữ liệu sạch  → {OUTPUT_CSV}")
    print(f"✓ Báo cáo chi tiết → {OUTPUT_REPORT}")
    print(f"  Đã loại bỏ: {len(df_raw) - len(df):,} dòng trùng lặp "
          f"({(len(df_raw) - len(df)) / len(df_raw) * 100:.2f}%)")


if __name__ == '__main__':
    main()
