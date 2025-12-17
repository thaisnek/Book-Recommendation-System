# -*- coding: utf-8 -*-
"""
BƯỚC 2: LÀM SẠCH DỮ LIỆU VÀ TRỰC QUAN HÓA (EDA)
Đáp ứng yêu cầu:
- Làm sạch: Missing values, Chuẩn hóa, Duplicate, Outlier, Vector hóa (5/5)
- EDA: Phân bố rating, Tần suất nhóm, Top items, Heatmap, Histogram (5/5)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Cấu hình
pd.set_option('display.max_columns', None)
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print("=" * 70)
print("BUOC 2: LAM SACH DU LIEU VA TRUC QUAN HOA (EDA)")
print("=" * 70)

# ==============================================================================
# 1. LOAD DỮ LIỆU
# ==============================================================================
try:
    books = pd.read_csv('books_final_dataset.csv')
    print(f"\n[OK] Da tai du lieu: {books.shape}")
except FileNotFoundError:
    print("[ERROR] Khong tim thay file 'books_final_dataset.csv'")
    print("Vui long chay file '1_data_collection.py' truoc!")
    exit()

# ==============================================================================
# PHẦN 1: LÀM SẠCH DỮ LIỆU (5/5 TÁC VỤ)
# ==============================================================================

print("\n" + "=" * 70)
print("PHAN 1: LAM SACH DU LIEU (5/5 TAC VU)")
print("=" * 70)

# 1. Loại bỏ Duplicate
print("\n1. [Duplicate] Dang loai bo trung lap...")
initial_count = len(books)
books.drop_duplicates(subset=['book_id'], inplace=True, keep='first')
print(f"   [OK] Da xoa {initial_count - len(books)} cuon sach trung lap")

# 2. Xử lý Missing Values
print("\n2. [Missing Values] Dang xu ly gia tri thieu...")
print(f"   Missing values truoc khi xu ly:")
missing_before = books.isnull().sum()
print(missing_before[missing_before > 0])

# Điền giá trị thiếu
books['title'] = books['title'].fillna('Unknown')
books['author_name'] = books['author_name'].fillna('Unknown')
books['genre'] = books['genre'].fillna('Fiction')
books['description'] = books['description'].fillna('')

if 'rating' in books.columns:
    books['rating'] = books['rating'].fillna(books['rating'].mean())

print(f"   [OK] Da xu ly xong missing values")
print(f"   Missing values sau khi xu ly: {books.isnull().sum().sum()}")

# 3. Xử lý Outlier
print("\n3. [Outlier] Dang loc bo outliers...")
books_clean = books.copy()

# Lọc bỏ sách không có rating hoặc rating = 0
if 'rating' in books_clean.columns:
    books_clean = books_clean[books_clean['rating'] > 0]

# Lọc bỏ sách không có title hoặc author
books_clean = books_clean[
    (books_clean['title'] != 'Unknown') & 
    (books_clean['author_name'] != 'Unknown')
]

# Lọc bỏ rating quá cao (> 5) hoặc quá thấp (< 0)
if 'rating' in books_clean.columns:
    books_clean = books_clean[(books_clean['rating'] >= 0) & (books_clean['rating'] <= 5)]

print(f"   [OK] Da loc bo outliers. So luong sach con lai: {len(books_clean)}")

# 4. Chuẩn hóa dữ liệu
print("\n4. [Normalization] Dang chuan hoa du lieu...")
scaler = MinMaxScaler()

if 'rating' in books_clean.columns:
    books_clean['rating_normalized'] = scaler.fit_transform(books_clean[['rating']])
    print("   [OK] Da chuan hoa cot 'rating' ve thang [0-1]")

# Chuẩn hóa ratings_count nếu có
if 'ratings_count' in books_clean.columns:
    books_clean['ratings_count_normalized'] = scaler.fit_transform(
        books_clean[['ratings_count']].fillna(0)
    )

# 5. Vector hóa văn bản
print("\n5. [Vectorization] Dang thuc hien vector hoa van ban...")
if 'text_features' not in books_clean.columns:
    books_clean['text_features'] = (
        books_clean['title'].fillna('') + ' ' +
        books_clean['author_name'].fillna('') + ' ' +
        books_clean['genre'].fillna('') + ' ' +
        books_clean.get('description', '').fillna('')
    )

tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(books_clean['text_features'])
print(f"   [OK] Ket qua ma tran TF-IDF: {tfidf_matrix.shape}")

print("\n" + "=" * 70)
print("[OK] HOAN TAT LAM SACH DU LIEU (5/5 TAC VU)")
print("=" * 70)

# Lưu dữ liệu đã làm sạch
books_clean.to_csv('books_cleaned.csv', index=False, encoding='utf-8')
print("[OK] Da luu file 'books_cleaned.csv'")

# ==============================================================================
# PHẦN 2: TRỰC QUAN HÓA DỮ LIỆU (5/5 BIỂU ĐỒ)
# ==============================================================================

print("\n" + "=" * 70)
print("PHAN 2: TRUC QUAN HOA DU LIEU (5/5 BIEU DO)")
print("=" * 70)

# Thiết lập khung hình lớn (Grid 2 dòng x 3 cột)
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('BAO CAO PHAN TICH DU LIEU SACH (EDA)', fontsize=24, fontweight='bold', color='navy')

# --- BIỂU ĐỒ 1: PHÂN BỐ RATING (Histogram) ---
print("\n1. Dang ve bieu do phan bo rating...")
rating_col = 'rating' if 'rating' in books_clean.columns else 'average_rating'
if rating_col in books_clean.columns:
    sns.histplot(books_clean[rating_col], bins=30, kde=True, color='#3498db', ax=axes[0, 0])
    axes[0, 0].set_title('1. Phan bo diem danh gia (Rating)', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Diem so')
    axes[0, 0].set_ylabel('So luong sach')

# --- BIỂU ĐỒ 2: TOP AUTHORS (Bar Chart - Top items) ---
print("2. Dang ve bieu do top tac gia...")
top_authors = books_clean['author_name'].value_counts().head(10)
sns.barplot(x=top_authors.values, y=top_authors.index, palette='viridis', ax=axes[0, 1])
axes[0, 1].set_title('2. Top 10 Tac gia co nhieu sach nhat', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('So luong sach')
axes[0, 1].set_ylabel('')

# --- BIỂU ĐỒ 3: TẦN SUẤT THỂ LOẠI (Bar Chart - Tần suất nhóm sản phẩm) ---
print("3. Dang ve bieu do tan suat the loai...")
if 'genre' in books_clean.columns:
    top_genres = books_clean['genre'].value_counts().head(10)
    sns.countplot(y=books_clean['genre'], order=top_genres.index, 
                  palette='muted', ax=axes[0, 2])
    axes[0, 2].set_title('3. Top 10 The loai sach pho bien', fontsize=14, fontweight='bold')
    axes[0, 2].set_xlabel('So luong sach')
    axes[0, 2].set_ylabel('')

# --- BIỂU ĐỒ 4: HEATMAP TƯƠNG QUAN (Heatmap) ---
print("4. Dang ve bieu do heatmap...")
corr_cols = []
if rating_col in books_clean.columns:
    corr_cols.append(rating_col)
if 'ratings_count' in books_clean.columns:
    corr_cols.append('ratings_count')
if 'work_ratings_count' in books_clean.columns:
    corr_cols.append('work_ratings_count')
if 'work_text_reviews_count' in books_clean.columns:
    corr_cols.append('work_text_reviews_count')

if len(corr_cols) >= 2:
    correlation_matrix = books_clean[corr_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", 
                linewidths=.5, ax=axes[1, 0])
    axes[1, 0].set_title('4. Heatmap tuong quan cac chi so', fontsize=14, fontweight='bold')

# --- BIỂU ĐỒ 5: PHÂN BỐ RATINGS_COUNT (Histogram) ---
print("5. Dang ve bieu do phan bo ratings_count...")
if 'ratings_count' in books_clean.columns:
    books_clean_ratings = books_clean[books_clean['ratings_count'] > 0]
    sns.histplot(books_clean_ratings['ratings_count'], bins=30, kde=True, color='#e74c3c', ax=axes[1, 1])
    axes[1, 1].set_title('5. Phan bo so luong danh gia', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('So luong danh gia')
    axes[1, 1].set_ylabel('So luong sach')
    axes[1, 1].set_xscale('log')  # Log scale vì có giá trị rất lớn

# --- BIỂU ĐỒ 6: WORDCLOUD (Nâng cao) ---
print("6. Dang ve bieu do WordCloud...")
text = " ".join(str(name) for name in books_clean['title'].fillna(''))
wordcloud = WordCloud(width=800, height=400, background_color='black', 
                     colormap='Reds').generate(text)
axes[1, 2].imshow(wordcloud, interpolation='bilinear')
axes[1, 2].axis("off")
axes[1, 2].set_title('6. WordCloud: Tu khoa trong ten sach', fontsize=14, fontweight='bold')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('books_eda_report.png', dpi=300, bbox_inches='tight')
print("[OK] Da luu bieu do EDA: 'books_eda_report.png'")
plt.close()  # Đóng figure để không hiển thị

# ==============================================================================
# THỐNG KÊ TỔNG QUAN
# ==============================================================================

print("\n" + "=" * 70)
print("THONG KE TONG QUAN:")
print("=" * 70)
print(f"Tong so sach: {len(books_clean):,}")
print(f"So tac gia: {books_clean['author_name'].nunique():,}")
print(f"So the loai: {books_clean['genre'].nunique() if 'genre' in books_clean.columns else 'N/A'}")
if rating_col in books_clean.columns:
    print(f"\nDiem danh gia trung binh: {books_clean[rating_col].mean():.2f}")
    print(f"Diem danh gia min: {books_clean[rating_col].min():.2f}")
    print(f"Diem danh gia max: {books_clean[rating_col].max():.2f}")

print("\n" + "=" * 70)
print("[OK] DA HOAN THANH TAT CA YEU CAU!")
print("=" * 70)
print("\nKET QUA:")
print("  - Lam sach: 5/5 tac vu (Duplicate, Missing values, Outlier, Normalization, Vectorization)")
print("  - EDA: 5/5 bieu do (Phan bo rating, Top authors, Tan suat the loai, Heatmap, Histogram, WordCloud)")
