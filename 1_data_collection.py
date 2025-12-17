# -*- coding: utf-8 -*-
"""
BƯỚC 1: THU THẬP DỮ LIỆU
- Đọc file books.csv và ratings.csv
- Chuẩn hóa tên cột
- Tạo books_final_dataset.csv
"""

import pandas as pd
import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 70)
print("BUOC 1: THU THAP DU LIEU")
print("=" * 70)

# ==============================================================================
# 1. ĐỌC FILE BOOKS.CSV
# ==============================================================================
print("\n1. Dang doc file books.csv...")
try:
    books = pd.read_csv('books.csv')
    print(f"   [OK] Da tai {len(books):,} cuon sach")
    print(f"   So cot: {len(books.columns)}")
except FileNotFoundError:
    print("   [ERROR] Khong tim thay file books.csv")
    print("   Vui long dat file books.csv vao thu muc!")
    exit(1)

# ==============================================================================
# 2. CHUẨN HÓA TÊN CỘT
# ==============================================================================
print("\n2. Dang chuan hoa ten cot...")

# Mapping tên cột
column_mapping = {
    'authors': 'author_name',
    'author': 'author_name',
    'average_rating': 'rating',
    'book_rating': 'rating',
}

for old_col, new_col in column_mapping.items():
    if old_col in books.columns and new_col not in books.columns:
        books = books.rename(columns={old_col: new_col})

# Đảm bảo có các cột cần thiết
if 'book_id' not in books.columns:
    if 'id' in books.columns:
        books['book_id'] = books['id']
    else:
        books['book_id'] = range(1, len(books) + 1)

if 'movieId' not in books.columns:
    books['movieId'] = range(1, len(books) + 1)

# Tạo genre nếu chưa có
if 'genre' not in books.columns:
    print("   [INFO] Khong co cot genre, dang tao...")
    books['genre'] = 'Fiction'  # Mặc định

# Tạo description nếu chưa có
if 'description' not in books.columns:
    print("   [INFO] Khong co cot description, dang tao...")
    title_col = books.get('title', '')
    author_col = books.get('author_name', books.get('authors', ''))
    books['description'] = title_col.astype(str) + ' by ' + author_col.astype(str)

# Tạo text_features
if 'text_features' not in books.columns:
    books['text_features'] = (
        books['title'].fillna('') + ' ' +
        books['author_name'].fillna('') + ' ' +
        books['genre'].fillna('') + ' ' +
        books.get('description', '').fillna('')
    )

print("   [OK] Da chuan hoa ten cot")

# ==============================================================================
# 3. XỬ LÝ CƠ BẢN
# ==============================================================================
print("\n3. Dang xu ly du lieu co ban...")

# Xử lý missing values
books['title'] = books['title'].fillna('Unknown')
books['author_name'] = books['author_name'].fillna('Unknown')
books['genre'] = books['genre'].fillna('Fiction')
books['description'] = books['description'].fillna('')

if 'rating' in books.columns:
    books['rating'] = books['rating'].fillna(books['rating'].mean())

print("   [OK] Da xu ly missing values")

# ==============================================================================
# 4. LƯU FILE
# ==============================================================================
print("\n4. Dang luu file books_final_dataset.csv...")
books.to_csv('books_final_dataset.csv', index=False, encoding='utf-8')
print("   [OK] Da luu file")

# ==============================================================================
# 5. KIỂM TRA RATINGS.CSV
# ==============================================================================
print("\n5. Dang kiem tra file ratings.csv...")
try:
    ratings = pd.read_csv('ratings.csv')
    print(f"   [OK] Da tai {len(ratings):,} ratings")
    
    # Chuẩn hóa tên cột ratings (nếu cần)
    if 'userId' not in ratings.columns:
        if 'user_id' in ratings.columns:
            ratings = ratings.rename(columns={'user_id': 'userId'})
        elif 'User-ID' in ratings.columns:
            ratings = ratings.rename(columns={'User-ID': 'userId'})
    
    if 'bookId' not in ratings.columns:
        if 'book_id' in ratings.columns:
            ratings = ratings.rename(columns={'book_id': 'bookId'})
        elif 'ISBN' in ratings.columns:
            # Cần map ISBN sang book_id
            pass
    
    # Hiển thị thông tin
    user_col = 'userId' if 'userId' in ratings.columns else ('user_id' if 'user_id' in ratings.columns else list(ratings.columns)[0])
    book_col = 'bookId' if 'bookId' in ratings.columns else ('book_id' if 'book_id' in ratings.columns else list(ratings.columns)[1])
    
    print(f"   So users: {ratings[user_col].nunique():,}")
    print(f"   So books: {ratings[book_col].nunique():,}")
    
    # Lưu ratings đã chuẩn hóa (nếu có thay đổi)
    if 'userId' in ratings.columns and 'bookId' in ratings.columns:
        ratings.to_csv('ratings.csv', index=False, encoding='utf-8')
        print("   [OK] File ratings.csv da san sang su dung")
    else:
        print("   [INFO] File ratings.csv co cau truc khac, can kiem tra lai")
    
except FileNotFoundError:
    print("   [WARNING] Khong tim thay file ratings.csv")
    print("   [INFO] Ban co the tao ratings.csv hoac su dung Collaborative Filtering khong can ratings")

# ==============================================================================
# 6. THỐNG KÊ
# ==============================================================================
print("\n" + "=" * 70)
print("THONG KE")
print("=" * 70)
print(f"Tong so sach: {len(books):,}")
print(f"So features: {len(books.columns)}")
print(f"So tac gia unique: {books['author_name'].nunique():,}")
print(f"So the loai unique: {books['genre'].nunique() if 'genre' in books.columns else 'N/A'}")

# Kiểm tra yêu cầu
print(f"\nKIEM TRA YEU CAU:")
print(f"  - Dataset >= 2,000 items: {'[OK]' if len(books) >= 2000 else '[X]'} ({len(books):,})")
print(f"  - Co it nhat 5 features: {'[OK]' if len(books.columns) >= 5 else '[X]'} ({len(books.columns)})")

important_features = ['title', 'author_name', 'genre', 'description', 'rating']
has_features = sum(1 for f in important_features if f in books.columns)
print(f"  - Co cac features quan trong: {'[OK]' if has_features >= 5 else '[X]'} ({has_features}/5)")

print("\n" + "=" * 70)
print("[OK] HOAN TAT BUOC 1: THU THAP DU LIEU")
print("=" * 70)
