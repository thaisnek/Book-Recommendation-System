import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import html
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from hybrid_recommendation import (
    hybrid_recommend_for_book,
    hybrid_recommend_for_favorites,
)
from model_evaluation import calculate_metrics, EvaluationResult

# --- CẤU HÌNH TRANG WEB ---
st.set_page_config(
    page_title="Book AI - Hệ thống Gợi ý Sách",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #8B4513;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# MODULE QUẢN LÝ FILE JSON
# ==========================================
USER_DATA_FILE = 'user_favorites.json'
USER_HISTORY_FILE = 'user_history.json'

def load_favorites_from_disk():
    if os.path.exists(USER_DATA_FILE):
        try:
            with open(USER_DATA_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except (json.JSONDecodeError, IOError, OSError) as e:
            st.warning(f"Lỗi đọc file favorites: {e}")
            return []
    return []

def save_favorites_to_disk(fav_list):
    try:
        with open(USER_DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(fav_list, f, ensure_ascii=False)
    except (IOError, OSError) as e:
        st.error(f"Lỗi lưu file favorites: {e}")

def load_history_from_disk():
    if os.path.exists(USER_HISTORY_FILE):
        try:
            with open(USER_HISTORY_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except (json.JSONDecodeError, IOError, OSError) as e:
            st.warning(f"Lỗi đọc file history: {e}")
            return []
    return []

def save_history_to_disk(history_list):
    with open(USER_HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history_list, f, ensure_ascii=False)

# ==========================================
# KHỞI TẠO STATE
# ==========================================
if 'favorites' not in st.session_state:
    st.session_state['favorites'] = load_favorites_from_disk()

if 'history' not in st.session_state:
    st.session_state['history'] = load_history_from_disk()

if 'selected_book_title' not in st.session_state:
    st.session_state['selected_book_title'] = ''

if 'rec_mode' not in st.session_state:
    st.session_state['rec_mode'] = "✨ Dựa trên Tủ sách của tôi"

# ==========================================
# LOAD DATA & TRAIN MODELS
# ==========================================
@st.cache_data
def load_data():
    try:
        books = pd.read_csv('books_cleaned.csv')
        if 'text_features' not in books.columns:
            books['text_features'] = (
                books['title'].fillna('') + ' ' +
                books['author_name'].fillna('') + ' ' +
                books['genre'].fillna('') + ' ' +
                books.get('description', '').fillna('')
            )
    except FileNotFoundError:
        st.error("❌ Không tìm thấy file 'books_cleaned.csv'")
        st.info("👉 Hãy chạy file '2_data_cleaning_eda.py' trước!")
        st.stop()
    
    try:
        ratings = pd.read_csv('ratings.csv')
        # Chuẩn hóa tên cột ratings
        if 'userId' not in ratings.columns:
            if 'user_id' in ratings.columns:
                ratings = ratings.rename(columns={'user_id': 'userId'})
        if 'bookId' not in ratings.columns:
            if 'book_id' in ratings.columns:
                ratings = ratings.rename(columns={'book_id': 'bookId'})
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        st.warning(f"Không thể đọc ratings.csv: {e}")
        ratings = None
    
    return books, ratings

@st.cache_resource
def train_models(books, ratings):
    # Content-Based với TF-IDF
    try:
        bert_matrix = np.load('bert_embeddings.npy')
        if bert_matrix.shape[0] != len(books):
            raise ValueError("BERT embeddings không khớp với số lượng sách")
        cosine_sim = cosine_similarity(bert_matrix, bert_matrix)
        st.success("✅ Đã sử dụng BERT embeddings")
    except (FileNotFoundError, ValueError, OSError) as e:
        st.info(f"ℹ️ Sử dụng TF-IDF (BERT không khả dụng: {e})")
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        tfidf_matrix = tfidf.fit_transform(books['text_features'])
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    
    # Collaborative Filtering
    corr_mat = None
    user_book_matrix = None
    
    if ratings is not None and len(ratings) > 0:
        try:
            # Đảm bảo có đúng tên cột
            if 'userId' not in ratings.columns and 'user_id' in ratings.columns:
                ratings = ratings.rename(columns={'user_id': 'userId'})
            if 'bookId' not in ratings.columns and 'book_id' in ratings.columns:
                ratings = ratings.rename(columns={'book_id': 'bookId'})
            
            if 'userId' in ratings.columns and 'bookId' in ratings.columns:
                user_book_matrix = ratings.pivot_table(
                    index='userId', columns='bookId', values='rating'
                ).fillna(0)
                
                if user_book_matrix.shape[0] > 0 and user_book_matrix.shape[1] > 0:
                    SVD = TruncatedSVD(n_components=50, random_state=42)
                    matrix_reduced = SVD.fit_transform(user_book_matrix.T)
                    corr_mat = np.corrcoef(matrix_reduced)
        except Exception as e:
            st.warning(f"Không thể tạo Collaborative Filtering: {e}")
            user_book_matrix = None
            corr_mat = None
    
    return cosine_sim, user_book_matrix, corr_mat

try:
    books, ratings = load_data()
    if len(books) == 0:
        st.error("❌ Dataset rỗng! Vui lòng chạy lại các bước xử lý dữ liệu.")
        st.stop()
    if st.session_state['selected_book_title'] == '':
        st.session_state['selected_book_title'] = books['title'].values[0]
    cosine_sim, user_book_matrix, corr_mat = train_models(books, ratings)
except (IndexError, KeyError) as e:
    st.error(f"❌ Lỗi cấu trúc dữ liệu: {e}")
    st.info("👉 Vui lòng chạy lại file '1_data_collection.py' và '2_data_cleaning_eda.py'")
    st.stop()
except Exception as e:
    st.error(f"❌ Lỗi không xác định: {e}")
    st.stop()

# ==========================================
# CÁC HÀM CALLBACK
# ==========================================
def navigate_to_book(book_title):
    st.session_state['selected_book_title'] = book_title
    st.session_state['rec_mode'] = "🔍 Tìm kiếm sách lẻ"
    if book_title not in st.session_state['history']:
        st.session_state['history'].append(book_title)
        save_history_to_disk(st.session_state['history'])

def add_to_favorites(book_title):
    if book_title not in st.session_state['favorites']:
        st.session_state['favorites'].append(book_title)
        save_favorites_to_disk(st.session_state['favorites'])
        st.toast(f"Đã thêm '{book_title}' vào tủ sách!", icon="❤️")

def remove_from_favorites(book_title):
    if book_title in st.session_state['favorites']:
        st.session_state['favorites'].remove(book_title)
        save_favorites_to_disk(st.session_state['favorites'])

# ==========================================
# HÀM RENDER CARD SÁCH
# ==========================================
def render_book_card(book_row, card_key_prefix="", show_add_button=True, show_detail_button=True):
    """Render một card sách đẹp với ảnh bìa"""
    
    # Lấy dữ liệu
    title = str(book_row.get('title', 'Unknown'))
    author = str(book_row.get('author_name', 'Unknown'))
    genre = book_row.get('genre', 'N/A')
    rating = book_row.get('rating', book_row.get('average_rating', None))
    book_id = book_row.get('book_id', book_row.get('movieId', 0))
    
    # Escape HTML để tránh lỗi
    title_escaped = html.escape(title)
    author_escaped = html.escape(author)
    
    # Lấy link ảnh (ưu tiên small_image_url, fallback về image_url)
    image_url = book_row.get('small_image_url', book_row.get('image_url', ''))
    if pd.isna(image_url) or str(image_url).strip() == '' or str(image_url).lower() == 'nan':
        image_url = None
    else:
        image_url = str(image_url).strip()
    
    # Tạo card với HTML/CSS (format trên một dòng để tránh lỗi render)
    card_html = '<div style="border: 1px solid #ddd; border-radius: 10px; padding: 15px; margin-bottom: 20px; background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1); height: 100%; display: flex; flex-direction: column;">'
    
    # Ảnh bìa sách
    if image_url:
        image_url_escaped = html.escape(image_url)
        card_html += f'<div style="text-align: center; margin-bottom: 10px;"><img src="{image_url_escaped}" alt="{title_escaped}" style="width: 100%; max-width: 150px; height: auto; border-radius: 5px; box-shadow: 0 2px 8px rgba(0,0,0,0.2);" onerror="this.style.display=\'none\'"></div>'
    
    # Tiêu đề sách
    title_display = title_escaped[:50] + "..." if len(title_escaped) > 50 else title_escaped
    card_html += f'<h4 style="margin: 10px 0 5px 0; font-size: 14px; font-weight: bold; color: #333; line-height: 1.3; min-height: 36px;">{title_display}</h4>'
    
    # Tác giả
    author_display = author_escaped[:30] + "..." if len(author_escaped) > 30 else author_escaped
    card_html += f'<p style="margin: 5px 0; font-size: 12px; color: #666;">✍️ {author_display}</p>'
    
    # Thể loại
    if genre and pd.notna(genre):
        genre_str = str(genre)
        genre_display = genre_str.split('|')[0][:25] if '|' in genre_str else genre_str[:25]
        genre_display = html.escape(genre_display)
        card_html += f'<p style="margin: 5px 0; font-size: 11px; color: #888;">📚 {genre_display}</p>'
    
    # Rating
    if rating and pd.notna(rating):
        try:
            rating_val = float(rating)
            stars = "⭐" * int(rating_val) + "☆" * (5 - int(rating_val))
            card_html += f'<p style="margin: 5px 0; font-size: 12px; color: #f39c12;">{stars} {rating_val:.1f}/5.0</p>'
        except (ValueError, TypeError):
            pass
    
    card_html += '</div>'
    
    return card_html, book_id

# ==========================================
# GIAO DIỆN (UI)
# ==========================================

# Header
st.markdown('<h1 class="main-header">📚 Book AI - Hệ thống Gợi ý Sách</h1>', unsafe_allow_html=True)

with st.sidebar:
    st.title("📚 Menu")
    page = st.radio("Chọn trang:", [
        "🏠 Trang chủ", 
        "⚙️ Quản lý Tủ sách", 
        "📊 Biểu đồ phân tích",
        "📜 Lịch sử đọc",
        "📈 Đánh giá Mô hình"
    ])
    st.divider()

    if page == "🏠 Trang chủ":
        st.header("🔍 Cấu hình gợi ý")
        recommendation_mode = st.radio(
            "Nguồn gợi ý:",
            ["✨ Dựa trên Tủ sách của tôi", "🔍 Tìm kiếm sách lẻ"],
            key='rec_mode'
        )

        if recommendation_mode == "🔍 Tìm kiếm sách lẻ":
            search_query = st.text_input("🔍 Tìm kiếm sách:", value="")
            if search_query:
                matches = books[books['title'].str.contains(search_query, case=False, na=False)]
                if not matches.empty:
                    selected_book = st.selectbox(
                        "Chọn sách:",
                        matches['title'].values,
                        key='selected_book_title'
                    )
                else:
                    st.warning("Không tìm thấy sách nào!")
            else:
                selected_book = st.selectbox("Chọn sách:", books['title'].values, key='selected_book_title')
        else:
            if not st.session_state['favorites']:
                st.info("Tủ sách đang trống.")
            else:
                st.success(f"Đang dùng {len(st.session_state['favorites'])} cuốn sách để phân tích.")

        st.divider()
        if 'genre' in books.columns:
            # Lấy tất cả genre unique (xử lý genre có dấu |)
            all_genres = set()
            for genre_str in books['genre'].dropna():
                if '|' in str(genre_str):
                    all_genres.update([g.strip() for g in str(genre_str).split('|')])
                else:
                    all_genres.add(str(genre_str).strip())
            genres = ["Tất cả"] + sorted(list(all_genres))[:20]  # Tăng lên 20 thể loại
            selected_genre = st.selectbox("Thể loại:", genres)
        else:
            selected_genre = "Tất cả"

# --- TRANG 1: TRANG CHỦ ---
if page == "🏠 Trang chủ":
    if recommendation_mode == "✨ Dựa trên Tủ sách của tôi":
        st.title("✨ Gợi ý dành riêng cho BẠN")
        fav_list = st.session_state['favorites']

        if not fav_list:
            st.warning("⚠️ Tủ sách của bạn đang trống!")
            st.info("👉 Hãy chuyển sang chế độ **'🔍 Tìm kiếm sách lẻ'** để thêm sách.")
        else:
            with st.spinner("AI đang phân tích gu đọc sách của bạn..."):
                aggregated_recs = hybrid_recommend_for_favorites(
                    favorites_list=fav_list,
                    books=books,
                    cosine_sim=cosine_sim,
                    user_book_matrix=user_book_matrix,
                    corr_mat=corr_mat,
                    genre=selected_genre,
                    top_n=12,
                )

            if not aggregated_recs.empty:
                st.success(f"Gợi ý phù hợp với thể loại **'{selected_genre}'**:")
                
                cols_per_row = 4
                for i, row in enumerate(aggregated_recs.iterrows()):
                    if i % cols_per_row == 0:
                        cols = st.columns(cols_per_row)
                    
                    with cols[i % cols_per_row]:
                        b = row[1]
                        book_id = b.get('book_id', b.get('movieId', i))
                        
                        # Hiển thị ảnh bìa (ưu tiên image_url lớn hơn để nét hơn)
                        image_url = b.get('image_url', b.get('small_image_url', ''))
                        if image_url and pd.notna(image_url) and str(image_url).strip() != '':
                            try:
                                # Dùng image_url lớn và tăng kích thước để nét hơn
                                st.image(str(image_url), width=200, use_container_width=True)
                            except:
                                pass
                        
                        # Tiêu đề
                        title = b.get('title', 'Unknown')
                        st.markdown(f"**{title[:50]}{'...' if len(title) > 50 else ''}**")
                        
                        # Tác giả
                        author = b.get('author_name', 'Unknown')
                        st.caption(f"✍️ {author[:30]}{'...' if len(str(author)) > 30 else ''}")
                        
                        # Thể loại
                        if 'genre' in b and pd.notna(b['genre']):
                            genre_str = str(b['genre'])
                            genre_display = genre_str.split('|')[0][:25] if '|' in genre_str else genre_str[:25]
                            st.caption(f"📚 {genre_display}")
                        
                        # Rating
                        rating = b.get('rating', b.get('average_rating', None))
                        if rating and pd.notna(rating):
                            try:
                                rating_val = float(rating)
                                stars = "⭐" * int(rating_val) + "☆" * (5 - int(rating_val))
                                st.caption(f"{stars} {rating_val:.1f}/5.0")
                            except:
                                pass
                        
                        # Buttons
                        col_btn1, col_btn2 = st.columns(2)
                        with col_btn1:
                            if b['title'] not in st.session_state['favorites']:
                                st.button("❤️", key=f"agg_add_{i}_{book_id}", 
                                         on_click=add_to_favorites, args=(b['title'],),
                                         use_container_width=True)
                            else:
                                st.button("✅", key=f"agg_added_{i}_{book_id}",
                                         disabled=True, use_container_width=True)
                        with col_btn2:
                            st.button("📖", key=f"agg_view_{i}_{book_id}", 
                                     on_click=navigate_to_book, args=(b['title'],),
                                     use_container_width=True)
            else:
                st.warning("Không tìm thấy sách phù hợp.")

    else:
        target_book = st.session_state['selected_book_title']
        st.title(f"📖 Khám phá: {target_book}")

        book_info = books[books['title'] == target_book]
        if book_info.empty:
            st.error("Không tìm thấy cuốn sách này!")
        else:
            book_info = book_info.iloc[0]
            
            c1, c2 = st.columns([1, 3])
            with c1:
                # Hiển thị ảnh bìa sách lớn
                image_url = book_info.get('image_url', book_info.get('small_image_url', ''))
                if image_url and pd.notna(image_url) and image_url != '':
                    # Tăng kích thước để ảnh nét hơn
                    st.image(image_url, width=300, use_container_width=True)
                else:
                    st.info("📚 Không có ảnh bìa")
                
                st.subheader("📊 Thông tin")
                st.write(f"**Tác giả:** {book_info['author_name']}")
                if 'genre' in book_info:
                    genre_display = str(book_info['genre']).replace('|', ', ')
                    st.write(f"**Thể loại:** {genre_display}")
                rating_col = book_info.get('rating', book_info.get('average_rating', 'N/A'))
                if pd.notna(rating_col):
                    stars = "⭐" * int(rating_col) + "☆" * (5 - int(rating_col))
                    st.write(f"**Đánh giá:** {stars} {rating_col:.1f}/5.0")
                if 'num_pages' in book_info and pd.notna(book_info['num_pages']):
                    st.write(f"**Số trang:** {int(book_info['num_pages'])}")
            
            with c2:
                st.subheader("📝 Mô tả")
                if 'description' in book_info and pd.notna(book_info['description']):
                    st.write(book_info['description'])
                else:
                    st.write("Không có mô tả.")
            
            if target_book not in st.session_state['favorites']:
                st.button("❤️ Thêm vào Tủ sách", on_click=add_to_favorites, args=(target_book,))
            else:
                st.success("✅ Đã có trong tủ sách!")

            st.divider()

            def show_book_grid(results, key_prefix):
                if results is not None and not results.empty:
                    cols_per_row = 5
                    for i, row in enumerate(results.iterrows()):
                        if i % cols_per_row == 0:
                            cols = st.columns(cols_per_row)
                        
                        with cols[i % cols_per_row]:
                            b = row[1]
                            book_id = b.get('book_id', b.get('movieId', i))
                            
                            # Hiển thị ảnh bìa (ưu tiên image_url lớn hơn để nét hơn)
                            image_url = b.get('image_url', b.get('small_image_url', ''))
                            if image_url and pd.notna(image_url) and str(image_url).strip() != '':
                                try:
                                    # Dùng image_url lớn và tăng kích thước để nét hơn
                                    st.image(str(image_url), width=200, use_container_width=True)
                                except:
                                    pass
                            
                            # Tiêu đề
                            title = b.get('title', 'Unknown')
                            st.markdown(f"**{title[:40]}{'...' if len(title) > 40 else ''}**")
                            
                            # Tác giả
                            author = b.get('author_name', 'Unknown')
                            st.caption(f"✍️ {author[:25]}{'...' if len(str(author)) > 25 else ''}")
                            
                            # Rating
                            rating = b.get('rating', b.get('average_rating', None))
                            if rating and pd.notna(rating):
                                try:
                                    rating_val = float(rating)
                                    st.caption(f"⭐ {rating_val:.1f}/5.0")
                                except:
                                    pass
                            
                            st.button("👉 Xem", key=f"{key_prefix}_{i}_{book_id}", 
                                     on_click=navigate_to_book, args=(b['title'],),
                                     use_container_width=True)
                else:
                    st.warning("Không tìm thấy sách phù hợp.")

            st.subheader("🤝 Gợi ý Hybrid (kết hợp Nội dung + Cộng đồng)")
            res = hybrid_recommend_for_book(
                book_title=target_book,
                books=books,
                cosine_sim=cosine_sim,
                user_book_matrix=user_book_matrix,
                corr_mat=corr_mat,
                genre=selected_genre,
                top_n=10,
            )
            show_book_grid(res, "hybrid")

# --- TRANG 2: QUẢN LÝ TỦ SÁCH ---
elif page == "⚙️ Quản lý Tủ sách":
    st.title("⚙️ Quản lý Tủ sách")
    fav_list = st.session_state['favorites']
    
    if fav_list:
        st.write(f"Bạn đang lưu **{len(fav_list)}** cuốn sách yêu thích.")
        st.divider()
        
        cols_per_row = 4
        for i, title in enumerate(fav_list):
            if i % cols_per_row == 0:
                cols = st.columns(cols_per_row)
            
            with cols[i % cols_per_row]:
                book_info = books[books['title'] == title]
                if not book_info.empty:
                    b = book_info.iloc[0]
                    book_id = b.get('book_id', b.get('movieId', i))
                    
                    # Hiển thị ảnh bìa (ưu tiên image_url lớn hơn để nét hơn)
                    image_url = b.get('image_url', b.get('small_image_url', ''))
                    if image_url and pd.notna(image_url) and str(image_url).strip() != '':
                        try:
                            # Dùng image_url lớn và tăng kích thước để nét hơn
                            st.image(str(image_url), width=200, use_container_width=True)
                        except:
                            pass
                    
                    # Tiêu đề
                    st.markdown(f"**{title[:50]}{'...' if len(title) > 50 else ''}**")
                    
                    # Tác giả
                    author = b.get('author_name', 'Unknown')
                    st.caption(f"✍️ {author[:30]}{'...' if len(str(author)) > 30 else ''}")
                    
                    # Rating
                    rating = b.get('rating', b.get('average_rating', None))
                    if rating and pd.notna(rating):
                        try:
                            rating_val = float(rating)
                            st.caption(f"⭐ {rating_val:.1f}/5.0")
                        except:
                            pass
                    
                    if st.button("🗑️ Xóa", key=f"del_{i}_{book_id}", use_container_width=True):
                        remove_from_favorites(title)
                        st.rerun()
        
        st.divider()
        if st.button("Xóa sạch tủ sách", type="primary"):
            st.session_state['favorites'] = []
            save_favorites_to_disk([])
            st.rerun()
    else:
        st.info("Tủ sách hiện đang trống.")

# --- TRANG 3: BIỂU ĐỒ PHÂN TÍCH ---
elif page == "📊 Biểu đồ phân tích":
    st.title("📊 Phân tích Dữ liệu (EDA Dashboard)")
    st.markdown("Tổng quan về bộ dữ liệu sách.")
    
    books_clean = books.copy()
    sns.set_style("whitegrid")
    
    tab_eda1, tab_eda2 = st.tabs(["📈 Thống kê cơ bản", "☁️ WordCloud & Tương quan"])
    
    with tab_eda1:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("1. Phân bố điểm đánh giá")
            rating_col = 'rating' if 'rating' in books_clean.columns else 'average_rating'
            if rating_col in books_clean.columns:
                fig1, ax1 = plt.subplots(figsize=(8, 5))
                sns.histplot(books_clean[rating_col], bins=20, kde=True, color='#8B4513', ax=ax1)
                ax1.set_xlabel('Điểm số')
                st.pyplot(fig1)
        
        with c2:
            st.subheader("2. Top 10 Tác giả phổ biến")
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            top_authors = books_clean['author_name'].value_counts().head(10)
            sns.barplot(x=top_authors.values, y=top_authors.index, palette='viridis', ax=ax2)
            ax2.set_ylabel('')
            st.pyplot(fig2)
        
        st.subheader("3. Top Thể loại sách")
        if 'genre' in books_clean.columns:
            fig3, ax3 = plt.subplots(figsize=(10, 5))
            top_genres = books_clean['genre'].value_counts().head(10)
            sns.countplot(y=books_clean['genre'], order=top_genres.index, 
                         palette='muted', ax=ax3)
            ax3.set_ylabel('')
            st.pyplot(fig3)
    
    with tab_eda2:
        col_heat, col_cloud = st.columns(2)
        
        with col_heat:
            st.subheader("4. Heatmap tương quan")
            fig4, ax4 = plt.subplots(figsize=(6, 5))
            corr_cols = []
            if rating_col in books_clean.columns:
                corr_cols.append(rating_col)
            if 'num_pages' in books_clean.columns:
                corr_cols.append('num_pages')
            if len(corr_cols) >= 2:
                correlation_matrix = books_clean[corr_cols].corr()
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', 
                           fmt=".2f", linewidths=.5, ax=ax4)
                st.pyplot(fig4)
        
        with col_cloud:
            st.subheader("5. WordCloud Tên sách")
            with st.spinner("Đang tạo WordCloud..."):
                text = " ".join(str(name) for name in books_clean['title'].fillna(''))
                wordcloud = WordCloud(width=800, height=400, background_color='black', 
                                     colormap='Reds').generate(text)
                fig5, ax5 = plt.subplots(figsize=(8, 5))
                ax5.imshow(wordcloud, interpolation='bilinear')
                ax5.axis("off")
                st.pyplot(fig5)

# --- TRANG 4: LỊCH SỬ ĐỌC ---
elif page == "📜 Lịch sử đọc":
    st.title("📜 Lịch sử đọc sách")
    history_list = st.session_state['history']
    
    if history_list:
        st.write(f"Bạn đã xem **{len(history_list)}** cuốn sách.")
        st.divider()
        
        cols_per_row = 4
        for i, title in enumerate(reversed(history_list[-20:]), 1):
            if (i - 1) % cols_per_row == 0:
                cols = st.columns(cols_per_row)
            
            with cols[(i - 1) % cols_per_row]:
                book_info = books[books['title'] == title]
                if not book_info.empty:
                    b = book_info.iloc[0]
                    book_id = b.get('book_id', b.get('movieId', i))
                    
                    # Hiển thị ảnh bìa (ưu tiên image_url lớn hơn để nét hơn)
                    image_url = b.get('image_url', b.get('small_image_url', ''))
                    if image_url and pd.notna(image_url) and str(image_url).strip() != '':
                        try:
                            # Dùng image_url lớn và tăng kích thước để nét hơn
                            st.image(str(image_url), width=200, use_container_width=True)
                        except:
                            pass
                    
                    # Tiêu đề
                    st.markdown(f"**{title[:50]}{'...' if len(title) > 50 else ''}**")
                    
                    # Tác giả
                    author = b.get('author_name', 'Unknown')
                    st.caption(f"✍️ {author[:30]}{'...' if len(str(author)) > 30 else ''}")
                    
                    st.button("Xem lại", key=f"hist_{i}_{book_id}", 
                             on_click=navigate_to_book, args=(title,),
                             use_container_width=True)
        
        if st.button("Xóa lịch sử", type="primary"):
            st.session_state['history'] = []
            save_history_to_disk([])
            st.rerun()
    else:
        st.info("Chưa có lịch sử đọc sách.")

# --- TRANG 5: ĐÁNH GIÁ MÔ HÌNH ---
elif page == "📈 Đánh giá Mô hình":
    st.title("📈 Đánh giá Mô hình Recommendation")
    st.markdown("---")
    
    # Thông tin về dữ liệu
    st.info("""
    **Lưu ý:** Dữ liệu đánh giá được lấy từ dataset công khai (Goodreads/MovieLens), 
    không phải từ người dùng thực tế của hệ thống. Đây là cách tiếp cận phổ biến cho dự án học tập và demo.
    """)
    
    @st.cache_data
    def calculate_metrics_cached() -> EvaluationResult:
        return calculate_metrics(
            ratings_path="ratings.csv",
            k=10,
            relevant_threshold=4,
            max_users=1000,
            test_size=0.2,
            random_state=42,
            n_components=50,
        )
    
    # Tính toán và hiển thị metrics
    with st.spinner("Đang tính toán metrics..."):
        metrics = calculate_metrics_cached()
    
    if metrics.rmse is None:
        st.error("❌ Không thể tính toán metrics. Vui lòng kiểm tra file 'ratings.csv'.")
        if metrics.error:
            st.caption(f"Chi tiết lỗi: {metrics.error}")
    else:
        if getattr(metrics, "note", None):
            st.info(metrics.note)
        rmse, mae, precision, recall, test_size = (
            metrics.rmse,
            metrics.mae,
            metrics.precision_at_k,
            metrics.recall_at_k,
            metrics.test_size,
        )
        # Hiển thị thông tin dataset
        st.subheader("📊 Thông tin Dataset")
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.metric("📚 Tổng số sách", f"{len(books):,}")
        with col_info2:
            st.metric("👥 Số users", f"{len(ratings['userId'].unique()):,}" if ratings is not None else "N/A")
        with col_info3:
            st.metric("⭐ Test set size", f"{test_size:,}" if test_size else "N/A")
        
        st.markdown("---")
        
        # Hiển thị metrics chính
        st.subheader("🎯 Metrics Đánh giá")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # RMSE Card
            st.markdown("""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 20px; border-radius: 10px; color: white; margin-bottom: 20px;">
                <h3 style="margin: 0; color: white;">RMSE</h3>
                <p style="font-size: 12px; margin: 5px 0; opacity: 0.9;">Root Mean Squared Error</p>
                <h2 style="margin: 10px 0; font-size: 2.5rem; color: white;">{:.4f}</h2>
                <p style="font-size: 11px; margin: 0; opacity: 0.8;">Sai số trung bình bình phương</p>
            </div>
            """.format(rmse), unsafe_allow_html=True)
            
            # Đánh giá RMSE
            if rmse < 1.0:
                st.success("✅ **Tuyệt vời!** Model hoạt động rất tốt (RMSE < 1.0)")
            elif rmse < 1.5:
                st.info("✅ **Tốt!** Model hoạt động ổn định (RMSE < 1.5)")
            else:
                st.warning("⚠️ **Cần cải thiện** (RMSE >= 1.5)")
        
        with col2:
            # MAE Card
            st.markdown("""
            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                        padding: 20px; border-radius: 10px; color: white; margin-bottom: 20px;">
                <h3 style="margin: 0; color: white;">MAE</h3>
                <p style="font-size: 12px; margin: 5px 0; opacity: 0.9;">Mean Absolute Error</p>
                <h2 style="margin: 10px 0; font-size: 2.5rem; color: white;">{:.4f}</h2>
                <p style="font-size: 11px; margin: 0; opacity: 0.8;">Sai số tuyệt đối trung bình</p>
            </div>
            """.format(mae), unsafe_allow_html=True)
            
            # Đánh giá MAE
            if mae < 0.7:
                st.success("✅ **Tuyệt vời!** Sai số tuyệt đối thấp")
            elif mae < 1.0:
                st.info("✅ **Tốt!** Sai số ở mức chấp nhận được")
            else:
                st.warning("⚠️ **Cần cải thiện**")
        
        st.markdown("---")
        
        # Ranking Metrics
        st.subheader("📈 Ranking Metrics")
        
        col3, col4 = st.columns(2)
        
        with col3:
            # Precision@10 Card
            precision_pct = precision * 100
            st.markdown("""
            <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                        padding: 20px; border-radius: 10px; color: white; margin-bottom: 20px;">
                <h3 style="margin: 0; color: white;">Precision@10</h3>
                <p style="font-size: 12px; margin: 5px 0; opacity: 0.9;">Độ chính xác trong top 10</p>
                <h2 style="margin: 10px 0; font-size: 2.5rem; color: white;">{:.2f}%</h2>
                <p style="font-size: 11px; margin: 0; opacity: 0.8;">({:.4f})</p>
            </div>
            """.format(precision_pct, precision), unsafe_allow_html=True)
            
            if precision > 0.1:
                st.success("✅ **Tốt!** Precision cao")
            elif precision > 0.05:
                st.info("ℹ️ **Ổn định** Precision ở mức trung bình")
            else:
                st.warning("⚠️ **Cần cải thiện** Precision thấp (có thể do dataset lớn)")
        
        with col4:
            # Recall@10 Card
            recall_pct = recall * 100
            st.markdown("""
            <div style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); 
                        padding: 20px; border-radius: 10px; color: white; margin-bottom: 20px;">
                <h3 style="margin: 0; color: white;">Recall@10</h3>
                <p style="font-size: 12px; margin: 5px 0; opacity: 0.9;">Khả năng tìm relevant items</p>
                <h2 style="margin: 10px 0; font-size: 2.5rem; color: white;">{:.2f}%</h2>
                <p style="font-size: 11px; margin: 0; opacity: 0.8;">({:.4f})</p>
            </div>
            """.format(recall_pct, recall), unsafe_allow_html=True)
            
            if recall > 0.1:
                st.success("✅ **Tốt!** Recall cao")
            elif recall > 0.05:
                st.info("ℹ️ **Ổn định** Recall ở mức trung bình")
            else:
                st.warning("⚠️ **Cần cải thiện** Recall thấp (có thể do dataset lớn)")
        
        st.markdown("---")
        
        # Giải thích metrics
        with st.expander("📖 Giải thích Metrics"):
            st.markdown("""
            ### **RMSE (Root Mean Squared Error)**
            - Đo lường sai số trung bình bình phương giữa rating dự đoán và rating thực tế
            - Giá trị càng thấp càng tốt
            - RMSE < 1.0: Model hoạt động rất tốt
            
            ### **MAE (Mean Absolute Error)**
            - Đo lường sai số tuyệt đối trung bình
            - Dễ hiểu hơn RMSE (không bình phương)
            - Giá trị càng thấp càng tốt
            
            ### **Precision@10**
            - Tỷ lệ sách relevant trong top 10 gợi ý
            - Precision cao = gợi ý chính xác hơn
            - Công thức: (Số relevant trong top 10) / 10
            
            ### **Recall@10**
            - Tỷ lệ relevant items được tìm thấy trong top 10
            - Recall cao = tìm được nhiều sách phù hợp hơn
            - Công thức: (Số relevant tìm được) / (Tổng số relevant)
            
            **Lưu ý:** Precision@10 và Recall@10 có thể thấp do dataset lớn (10,000 sách) 
            và chỉ đánh giá trên 100 users mẫu.
            """)
        
        # Thống kê từ dữ liệu người dùng thực tế (nếu có)
        st.markdown("---")
        st.subheader("👤 Dữ liệu từ Người dùng Thực tế")
        
        favorites_count = len(st.session_state['favorites'])
        history_count = len(st.session_state['history'])
        
        col_user1, col_user2 = st.columns(2)
        with col_user1:
            st.metric("📚 Sách trong Tủ sách", favorites_count)
        with col_user2:
            st.metric("📜 Lịch sử đọc", history_count)
        
        if favorites_count > 0 or history_count > 0:
            st.info(f"""
            💡 **Thông tin:** Hệ thống đã ghi nhận {favorites_count} sách yêu thích và {history_count} lượt xem từ người dùng thực tế.
            Dữ liệu này được sử dụng để cải thiện gợi ý trong tương lai.
            """)
        else:
            st.info("💡 Chưa có dữ liệu từ người dùng thực tế. Hãy sử dụng hệ thống để tạo dữ liệu!")

