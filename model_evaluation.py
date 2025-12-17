import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class EvaluationResult:
    rmse: float | None
    mae: float | None
    precision_at_k: float | None
    recall_at_k: float | None
    test_size: int | None
    error: str | None = None
    note: str | None = None


def _maybe_reduce_ratings_for_matrix(
    ratings: pd.DataFrame,
    user_col: str = "userId",
    item_col: str = "bookId",
    max_cells: int = 20_000_000,
    max_users: int = 4000,
    max_items: int = 4000,
) -> tuple[pd.DataFrame, str | None]:
    """
    Tránh tạo pivot_table quá lớn (dense) gây MemoryError.
    Nếu (n_users * n_items) vượt ngưỡng max_cells thì lọc về top users/items theo số lượt rating.
    """
    n_users = int(ratings[user_col].nunique())
    n_items = int(ratings[item_col].nunique())
    cells = n_users * n_items
    if cells <= max_cells:
        return ratings, None

    u_keep = min(max_users, n_users)
    i_keep = min(max_items, n_items)

    top_users = ratings[user_col].value_counts().head(u_keep).index
    reduced = ratings[ratings[user_col].isin(top_users)]

    top_items = reduced[item_col].value_counts().head(i_keep).index
    reduced = reduced[reduced[item_col].isin(top_items)]

    note = (
        "Danh gia duoc tinh tren subset de tranh thieu RAM: "
        f"{reduced[user_col].nunique():,} users x {reduced[item_col].nunique():,} books "
        f"(tu {n_users:,} x {n_items:,})."
    )
    return reduced, note


def _precision_at_k(actual_set: set, predicted_list: list, k: int = 10) -> float:
    if not predicted_list or not actual_set:
        return 0.0
    top_k_predicted = set(predicted_list[:k])
    if not top_k_predicted:
        return 0.0
    return len(top_k_predicted.intersection(actual_set)) / len(top_k_predicted)


def _recall_at_k(actual_set: set, predicted_list: list, k: int = 10) -> float:
    if not actual_set:
        return 0.0
    if not predicted_list:
        return 0.0
    top_k_predicted = set(predicted_list[:k])
    return len(top_k_predicted.intersection(actual_set)) / len(actual_set)


def calculate_metrics(
    ratings_path: str = "ratings.csv",
    k: int = 10,
    relevant_threshold: int = 4,
    max_users: int = 1000,
    test_size: float = 0.2,
    random_state: int = 42,
    n_components: int = 50,
    max_cells: int = 20_000_000,
    max_users_matrix: int = 4000,
    max_items_matrix: int = 4000,
) -> EvaluationResult:
    """
    Tính RMSE, MAE, Precision@K, Recall@K từ ratings.csv theo kiểu hold-out:
    - Train/Test split
    - SVD reconstruct để dự đoán rating cho (user, item)
    - Precision/Recall@K: relevant lấy từ TEST (rating >= threshold),
      recommend là các item user CHƯA rate trong TRAIN.
    """
    try:
        ratings_eval = pd.read_csv(ratings_path)
    except Exception as e:
        return EvaluationResult(None, None, None, None, None, error=f"Không thể đọc {ratings_path}: {e}")

    # Chuẩn hóa tên cột
    if "userId" not in ratings_eval.columns and "user_id" in ratings_eval.columns:
        ratings_eval = ratings_eval.rename(columns={"user_id": "userId"})
    if "bookId" not in ratings_eval.columns and "book_id" in ratings_eval.columns:
        ratings_eval = ratings_eval.rename(columns={"book_id": "bookId"})

    if "userId" not in ratings_eval.columns or "bookId" not in ratings_eval.columns or "rating" not in ratings_eval.columns:
        return EvaluationResult(None, None, None, None, None, error="ratings.csv thiếu cột userId/bookId/rating")

    try:
        ratings_eval, note = _maybe_reduce_ratings_for_matrix(
            ratings_eval,
            user_col="userId",
            item_col="bookId",
            max_cells=max_cells,
            max_users=max_users_matrix,
            max_items=max_items_matrix,
        )

        train_data, test_data = train_test_split(ratings_eval, test_size=test_size, random_state=random_state)
        if train_data.empty or test_data.empty:
            return EvaluationResult(None, None, None, None, None, error="Train/Test split bị rỗng", note=note)

        train_matrix = train_data.pivot_table(index="userId", columns="bookId", values="rating")
        if train_matrix.shape[0] == 0 or train_matrix.shape[1] == 0:
            return EvaluationResult(None, None, None, None, None, error="Ma trận train rỗng", note=note)

        # Mean centering + fillna(0) để SVD
        train_user_means = train_matrix.mean(axis=1)
        train_matrix_centered = train_matrix.sub(train_user_means, axis=0).fillna(0)
        X_train = train_matrix_centered.values

        # SVD
        n_comp_eff = min(n_components, min(X_train.shape) - 1) if min(X_train.shape) > 1 else 1
        if n_comp_eff < 1:
            return EvaluationResult(None, None, None, None, None, error="Không đủ dữ liệu để train SVD", note=note)

        svd = TruncatedSVD(n_components=n_comp_eff, random_state=random_state)
        X_train_transformed = svd.fit_transform(X_train)
        X_train_reconstructed = svd.inverse_transform(X_train_transformed)
        predicted_matrix_train = X_train_reconstructed + train_user_means.values.reshape(-1, 1)

        # RMSE/MAE trên test set (chỉ các cặp có mặt trong train_matrix)
        test_predictions = []
        test_actuals = []
        for _, row in test_data.iterrows():
            user_id = row["userId"]
            book_id = row["bookId"]
            actual_rating = row["rating"]

            if user_id in train_matrix.index and book_id in train_matrix.columns:
                user_idx = train_matrix.index.get_loc(user_id)
                book_idx = train_matrix.columns.get_loc(book_id)
                predicted_rating = float(predicted_matrix_train[user_idx, book_idx])
                predicted_rating = max(1.0, min(5.0, predicted_rating))
                test_predictions.append(predicted_rating)
                test_actuals.append(float(actual_rating))

        if not test_predictions:
            return EvaluationResult(
                None,
                None,
                None,
                None,
                len(test_data),
                error="Không có sample test nào khớp train để tính RMSE/MAE",
                note=note,
            )

        rmse_test = math.sqrt(mean_squared_error(test_actuals, test_predictions))
        mae_test = mean_absolute_error(test_actuals, test_predictions)

        # Precision@K / Recall@K
        test_relevant = test_data[test_data["rating"] >= relevant_threshold][["userId", "bookId"]].copy()
        test_relevant = test_relevant[test_relevant["bookId"].isin(train_matrix.columns)]
        relevant_by_user = test_relevant.groupby("userId")["bookId"].apply(set).to_dict()

        eligible_users = [u for u in train_matrix.index if u in relevant_by_user]
        sample_users = eligible_users[: min(max_users, len(eligible_users))]

        precision_scores = []
        recall_scores = []
        book_ids = train_matrix.columns.to_numpy()

        for user_id in sample_users:
            user_ratings = train_matrix.loc[user_id]
            rated_mask = user_ratings.notna().to_numpy()
            actual_rated_set = set(user_ratings[user_ratings.notna()].index.tolist())

            actual_relevant_set = set(relevant_by_user.get(user_id, set()))
            actual_relevant_set = actual_relevant_set.difference(actual_rated_set)
            if len(actual_relevant_set) == 0:
                continue

            user_idx = train_matrix.index.get_loc(user_id)
            scores = predicted_matrix_train[user_idx].copy()
            scores[rated_mask] = -np.inf

            if np.all(np.isneginf(scores)):
                continue

            k_eff = min(k, scores.shape[0])
            top_idx = np.argpartition(scores, -k_eff)[-k_eff:]
            top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
            predicted_books_unrated = list(book_ids[top_idx])

            precision_scores.append(_precision_at_k(actual_relevant_set, predicted_books_unrated, k=k))
            recall_scores.append(_recall_at_k(actual_relevant_set, predicted_books_unrated, k=k))

        avg_precision = float(np.mean(precision_scores)) if precision_scores else 0.0
        avg_recall = float(np.mean(recall_scores)) if recall_scores else 0.0

        return EvaluationResult(rmse_test, mae_test, avg_precision, avg_recall, int(len(test_data)), note=note)

    except Exception as e:
        return EvaluationResult(None, None, None, None, None, error=str(e))


