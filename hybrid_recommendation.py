import numpy as np
import pandas as pd


def filter_by_genre(books_df: pd.DataFrame, genre: str | None) -> pd.DataFrame:
    """Lọc sách theo thể loại (hỗ trợ genre có dấu |)."""
    if genre is None or genre == "Tất cả":
        return books_df
    if books_df is None or books_df.empty:
        return books_df
    if "genre" not in books_df.columns:
        return books_df

    target = str(genre).strip().lower()

    def genre_match(book_genre) -> bool:
        if pd.isna(book_genre):
            return False
        g = str(book_genre)
        if "|" in g:
            parts = [p.strip().lower() for p in g.split("|")]
            return any(target in p for p in parts)
        return target in g.lower()

    return books_df[books_df["genre"].apply(genre_match)]


def _content_recs_for_book(
    book_title: str,
    books: pd.DataFrame,
    cosine_sim: np.ndarray,
    top_k: int = 30,
) -> pd.DataFrame:
    """Content-based (cosine similarity) cho 1 cuốn sách."""
    if books is None or books.empty:
        return pd.DataFrame()
    if not isinstance(book_title, str) or not book_title.strip():
        return pd.DataFrame()

    idx_list = books.index[books["title"] == book_title].tolist()
    if not idx_list:
        # thử tìm gần đúng
        matches = books[books["title"].astype(str).str.contains(book_title, case=False, na=False)]
        if matches.empty:
            return pd.DataFrame()
        idx_list = [matches.index[0]]

    idx = idx_list[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1 : top_k + 1]

    recs = pd.DataFrame(
        {
            "title": [books.iloc[i]["title"] for i, _ in sim_scores],
            "similarity_score": [float(s) for _, s in sim_scores],
        }
    )
    return recs


def _collab_recs_for_book(
    book_title: str,
    books: pd.DataFrame,
    user_book_matrix: pd.DataFrame | None,
    corr_mat: np.ndarray | None,
    top_k: int = 30,
) -> pd.DataFrame:
    """Collaborative (corr matrix từ SVD) cho 1 cuốn sách."""
    if user_book_matrix is None or corr_mat is None:
        return pd.DataFrame()
    if books is None or books.empty:
        return pd.DataFrame()
    if not isinstance(book_title, str) or not book_title.strip():
        return pd.DataFrame()

    book_matches = books[books["title"] == book_title]
    if book_matches.empty:
        book_matches = books[books["title"].astype(str).str.contains(book_title, case=False, na=False)]
        if book_matches.empty:
            return pd.DataFrame()

    book_id = book_matches.iloc[0].get("book_id", book_matches.iloc[0].get("movieId"))
    if book_id not in user_book_matrix.columns:
        return pd.DataFrame()

    col_idx = user_book_matrix.columns.get_loc(book_id)
    corr_specific = corr_mat[col_idx]

    result = pd.DataFrame({"corr_score": corr_specific})
    result["bookId"] = user_book_matrix.columns

    # map bookId -> title
    merged = pd.merge(
        result,
        books[["book_id", "title"]],
        left_on="bookId",
        right_on="book_id",
        how="inner",
    )

    merged = merged.sort_values("corr_score", ascending=False).iloc[1 : top_k + 1]
    return merged[["title", "corr_score"]].copy()


def hybrid_recommend_for_book(
    book_title: str,
    books: pd.DataFrame,
    cosine_sim: np.ndarray,
    user_book_matrix: pd.DataFrame | None = None,
    corr_mat: np.ndarray | None = None,
    genre: str | None = "Tất cả",
    top_n: int = 10,
    content_weight: float = 0.6,
    collab_weight: float = 0.4,
) -> pd.DataFrame:
    """Hybrid Recommendation (Content + Collaborative) cho 1 cuốn sách."""
    top_k = max(30, top_n * 3)
    content = _content_recs_for_book(book_title, books, cosine_sim, top_k=top_k)
    collab = _collab_recs_for_book(book_title, books, user_book_matrix, corr_mat, top_k=top_k)

    if content.empty and collab.empty:
        return pd.DataFrame()

    # chuẩn hóa & kết hợp
    scores: dict[str, float] = {}

    for _, row in content.iterrows():
        t = row["title"]
        if t == book_title:
            continue
        s = float(row.get("similarity_score", 0.0))
        s = max(0.0, min(1.0, s))
        scores[t] = scores.get(t, 0.0) + content_weight * s

    for _, row in collab.iterrows():
        t = row["title"]
        if t == book_title:
            continue
        raw = float(row.get("corr_score", 0.0))
        s = (raw + 1.0) / 2.0
        s = max(0.0, min(1.0, s))
        scores[t] = scores.get(t, 0.0) + collab_weight * s

    if not scores:
        return pd.DataFrame()

    score_df = pd.DataFrame({"title": list(scores.keys()), "hybrid_score": list(scores.values())})
    score_df = score_df.sort_values("hybrid_score", ascending=False).head(top_n)

    # trả về đầy đủ metadata sách để UI dùng
    out = pd.merge(score_df, books, on="title", how="left")
    out = filter_by_genre(out, genre)
    return out.head(top_n)


def hybrid_recommend_for_favorites(
    favorites_list: list[str],
    books: pd.DataFrame,
    cosine_sim: np.ndarray,
    user_book_matrix: pd.DataFrame | None = None,
    corr_mat: np.ndarray | None = None,
    genre: str | None = "Tất cả",
    top_n: int = 12,
    content_weight: float = 0.6,
    collab_weight: float = 0.4,
) -> pd.DataFrame:
    """Hybrid Recommendation dựa trên nhiều sách yêu thích (tủ sách)."""
    if not favorites_list:
        return pd.DataFrame()
    if books is None or books.empty:
        return pd.DataFrame()

    # Content-based aggregation
    book_scores_content: dict[str, float] = {}
    for fav_book in favorites_list:
        idx_list = books.index[books["title"] == fav_book].tolist()
        if not idx_list:
            continue
        idx = idx_list[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        top_similar = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:21]
        for i, score in top_similar:
            title = books.iloc[i]["title"]
            book_scores_content[title] = book_scores_content.get(title, 0.0) + float(score)

    # Collaborative aggregation (nếu có)
    book_scores_collab: dict[str, float] = {}
    if user_book_matrix is not None and corr_mat is not None:
        for fav_book in favorites_list:
            collab = _collab_recs_for_book(fav_book, books, user_book_matrix, corr_mat, top_k=30)
            if collab.empty:
                continue
            for _, row in collab.iterrows():
                title = row["title"]
                book_scores_collab[title] = book_scores_collab.get(title, 0.0) + float(row.get("corr_score", 0.0))

    # Combine
    all_titles = set(book_scores_content.keys()) | set(book_scores_collab.keys())
    if not all_titles:
        return pd.DataFrame()

    hybrid_scores: dict[str, float] = {}
    fav_count = max(1, len(favorites_list))

    for title in all_titles:
        score = 0.0

        if title in book_scores_content:
            content_score = min(book_scores_content[title] / fav_count, 1.0)
            score += content_weight * content_score

        if title in book_scores_collab:
            collab_score = (book_scores_collab[title] / fav_count + 1.0) / 2.0
            collab_score = max(0.0, min(1.0, collab_score))
            score += collab_weight * collab_score

        hybrid_scores[title] = score

    # loại bỏ sách đã có trong tủ
    for fav in favorites_list:
        hybrid_scores.pop(fav, None)

    score_df = pd.DataFrame({"title": list(hybrid_scores.keys()), "hybrid_score": list(hybrid_scores.values())})
    score_df = score_df.sort_values("hybrid_score", ascending=False).head(max(30, top_n))

    out = pd.merge(score_df, books, on="title", how="left")
    out = filter_by_genre(out, genre)
    return out.head(top_n)


