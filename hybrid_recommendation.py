import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel


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


def _get_idx_for_title(books: pd.DataFrame, book_title: str) -> int | None:
    if books is None or books.empty:
        return None
    if not isinstance(book_title, str) or not book_title.strip():
        return None
    idx_list = books.index[books["title"] == book_title].tolist()
    if idx_list:
        return int(idx_list[0])
    matches = books[books["title"].astype(str).str.contains(book_title, case=False, na=False)]
    if matches.empty:
        return None
    return int(matches.index[0])


def _content_scores_for_idx(
    idx: int,
    tfidf_matrix,
) -> np.ndarray:
    # tfidf_matrix: sparse (n_books x n_features)
    scores = linear_kernel(tfidf_matrix[idx], tfidf_matrix).ravel()
    return scores.astype(np.float32, copy=False)


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    if x is None:
        return x
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return x / norms


def _collab_scores_for_book_id(
    book_id: int,
    collab_book_ids: np.ndarray | None,
    collab_item_factors: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Return (book_ids, scores) for collaborative similarity.
    collab_item_factors is expected to be row-normalized (cosine via dot product).
    """
    if collab_book_ids is None or collab_item_factors is None:
        return None
    if len(collab_book_ids) == 0 or collab_item_factors.size == 0:
        return None

    id_to_pos = {int(b): i for i, b in enumerate(collab_book_ids)}
    pos = id_to_pos.get(int(book_id))
    if pos is None:
        return None

    v = collab_item_factors[pos]
    sims = (collab_item_factors @ v).astype(np.float32, copy=False)  # [-1, 1]
    sims[pos] = -np.inf
    return collab_book_ids, sims


def hybrid_recommend_for_book(
    book_title: str,
    books: pd.DataFrame,
    tfidf_matrix,
    collab_book_ids: np.ndarray | None = None,
    collab_item_factors: np.ndarray | None = None,
    genre: str | None = "Tất cả",
    top_n: int = 10,
    content_weight: float = 0.6,
    collab_weight: float = 0.4,
) -> pd.DataFrame:
    """Hybrid Recommendation (Content + Collaborative) cho 1 cuốn sách (memory-friendly)."""
    if books is None or books.empty:
        return pd.DataFrame()
    idx = _get_idx_for_title(books, book_title)
    if idx is None:
        return pd.DataFrame()

    top_k = max(50, top_n * 5)

    # ===== Content =====
    content_scores = _content_scores_for_idx(idx, tfidf_matrix)
    content_scores[idx] = -np.inf
    k_eff = min(top_k, content_scores.shape[0])
    top_idx = np.argpartition(content_scores, -k_eff)[-k_eff:]
    top_idx = top_idx[np.argsort(content_scores[top_idx])[::-1]]

    # ===== Collaborative =====
    book_id = books.loc[idx].get("book_id", books.loc[idx].get("movieId", None))
    collab_pack = _collab_scores_for_book_id(book_id, collab_book_ids, collab_item_factors)

    # Combine (by books row index)
    scores: dict[int, float] = {}

    for i in top_idx:
        if int(i) == int(idx):
            continue
        s = float(content_scores[i])
        s = max(0.0, min(1.0, s))
        scores[int(i)] = scores.get(int(i), 0.0) + content_weight * s

    if collab_pack is not None and "book_id" in books.columns:
        all_book_ids, collab_scores = collab_pack
        # map book_id -> books row index
        bid_to_row = pd.Series(books.index, index=books["book_id"]).to_dict()
        k2 = min(top_k, collab_scores.shape[0])
        top_j = np.argpartition(collab_scores, -k2)[-k2:]
        top_j = top_j[np.argsort(collab_scores[top_j])[::-1]]
        for j in top_j:
            bid = int(all_book_ids[j])
            row_idx = bid_to_row.get(bid)
            if row_idx is None or int(row_idx) == int(idx):
                continue
            raw = float(collab_scores[j])  # [-1,1]
            s = (raw + 1.0) / 2.0
            s = max(0.0, min(1.0, s))
            scores[int(row_idx)] = scores.get(int(row_idx), 0.0) + collab_weight * s

    if not scores:
        return pd.DataFrame()

    score_df = pd.DataFrame({"_row_idx": list(scores.keys()), "hybrid_score": list(scores.values())})
    score_df = score_df.sort_values("hybrid_score", ascending=False).head(top_n)
    out = books.loc[score_df["_row_idx"].values].copy()
    out["hybrid_score"] = score_df["hybrid_score"].values
    out = filter_by_genre(out, genre)
    return out.head(top_n)


def hybrid_recommend_for_favorites(
    favorites_list: list[str],
    books: pd.DataFrame,
    tfidf_matrix,
    collab_book_ids: np.ndarray | None = None,
    collab_item_factors: np.ndarray | None = None,
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

    fav_indices = []
    for fav in favorites_list:
        idx = _get_idx_for_title(books, fav)
        if idx is not None:
            fav_indices.append(int(idx))
    if not fav_indices:
        return pd.DataFrame()

    top_per_fav = 30
    book_scores: dict[int, float] = {}
    for idx in fav_indices:
        scores = _content_scores_for_idx(idx, tfidf_matrix)
        scores[idx] = -np.inf
        k_eff = min(top_per_fav, scores.shape[0])
        top_idx = np.argpartition(scores, -k_eff)[-k_eff:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        for j in top_idx:
            book_scores[int(j)] = book_scores.get(int(j), 0.0) + float(scores[j])

    # Collaborative aggregation (nếu có)
    collab_scores_by_row: dict[int, float] = {}
    if collab_book_ids is not None and collab_item_factors is not None and "book_id" in books.columns:
        bid_to_row = pd.Series(books.index, index=books["book_id"]).to_dict()
        for idx in fav_indices:
            bid = books.loc[idx].get("book_id", None)
            pack = _collab_scores_for_book_id(bid, collab_book_ids, collab_item_factors)
            if pack is None:
                continue
            all_bids, sims = pack
            k2 = min(30, sims.shape[0])
            top_j = np.argpartition(sims, -k2)[-k2:]
            top_j = top_j[np.argsort(sims[top_j])[::-1]]
            for j in top_j:
                b = int(all_bids[j])
                row_idx = bid_to_row.get(b)
                if row_idx is None:
                    continue
                collab_scores_by_row[int(row_idx)] = collab_scores_by_row.get(int(row_idx), 0.0) + float(sims[j])

    # Combine (normalize + weights)
    fav_count = max(1, len(fav_indices))
    all_rows = set(book_scores.keys()) | set(collab_scores_by_row.keys())
    for idx in fav_indices:
        all_rows.discard(int(idx))

    if not all_rows:
        return pd.DataFrame()

    hybrid_scores: dict[int, float] = {}
    for row in all_rows:
        score = 0.0
        if row in book_scores:
            content_score = min(book_scores[row] / fav_count, 1.0)
            score += content_weight * content_score
        if row in collab_scores_by_row:
            raw = collab_scores_by_row[row] / fav_count  # approx [-1,1]
            collab_score = (raw + 1.0) / 2.0
            collab_score = max(0.0, min(1.0, collab_score))
            score += collab_weight * collab_score
        hybrid_scores[int(row)] = score

    score_df = pd.DataFrame({"_row_idx": list(hybrid_scores.keys()), "hybrid_score": list(hybrid_scores.values())})
    score_df = score_df.sort_values("hybrid_score", ascending=False).head(max(30, top_n))
    out = books.loc[score_df["_row_idx"].values].copy()
    out["hybrid_score"] = score_df["hybrid_score"].values
    out = filter_by_genre(out, genre)
    return out.head(top_n)


