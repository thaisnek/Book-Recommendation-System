import math
import argparse
import json
from datetime import datetime, timezone
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

    def to_dict(self) -> dict:
        return {
            "rmse": self.rmse,
            "mae": self.mae,
            "precision_at_k": self.precision_at_k,
            "recall_at_k": self.recall_at_k,
            "test_size": self.test_size,
            "error": self.error,
            "note": self.note,
        }

    @staticmethod
    def from_dict(d: dict) -> "EvaluationResult":
        if d is None:
            return EvaluationResult(None, None, None, None, None, error="Empty evaluation result")
        return EvaluationResult(
            rmse=d.get("rmse"),
            mae=d.get("mae"),
            precision_at_k=d.get("precision_at_k"),
            recall_at_k=d.get("recall_at_k"),
            test_size=d.get("test_size"),
            error=d.get("error"),
            note=d.get("note"),
        )


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
    max_train_users: int | None = 4000,
    max_train_items: int | None = 4000,
    test_size: float = 0.2,
    random_state: int = 42,
    n_components: int = 50,
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
        # === Downsample to keep pivot matrix feasible ===
        unique_users_before = int(ratings_eval["userId"].nunique())
        unique_items_before = int(ratings_eval["bookId"].nunique())

        note_parts: list[str] = []

        if isinstance(max_train_users, int) and max_train_users > 0 and unique_users_before > max_train_users:
            top_users = ratings_eval["userId"].value_counts().head(max_train_users).index
            ratings_eval = ratings_eval[ratings_eval["userId"].isin(top_users)]
            note_parts.append(f"Giới hạn top {max_train_users} users theo số rating để tránh OOM.")

        if isinstance(max_train_items, int) and max_train_items > 0:
            unique_items_mid = int(ratings_eval["bookId"].nunique())
            if unique_items_mid > max_train_items:
                top_items = ratings_eval["bookId"].value_counts().head(max_train_items).index
                ratings_eval = ratings_eval[ratings_eval["bookId"].isin(top_items)]
                note_parts.append(f"Giới hạn top {max_train_items} items theo số rating để tránh OOM.")

        # If we filtered, record a note for UI
        unique_users_after = int(ratings_eval["userId"].nunique())
        unique_items_after = int(ratings_eval["bookId"].nunique())
        if note_parts:
            note_parts.append(f"Users: {unique_users_after:,}/{unique_users_before:,}, Items: {unique_items_after:,}/{unique_items_before:,}.")
        note = " ".join(note_parts) if note_parts else None

        train_data, test_data = train_test_split(ratings_eval, test_size=test_size, random_state=random_state)
        if train_data.empty or test_data.empty:
            return EvaluationResult(None, None, None, None, None, error="Train/Test split bị rỗng")

        train_matrix = train_data.pivot_table(index="userId", columns="bookId", values="rating")
        if train_matrix.shape[0] == 0 or train_matrix.shape[1] == 0:
            return EvaluationResult(None, None, None, None, None, error="Ma trận train rỗng")

        # Mean centering + fillna(0) để SVD
        train_user_means = train_matrix.mean(axis=1)
        train_matrix_centered = train_matrix.sub(train_user_means, axis=0).fillna(0)
        X_train = train_matrix_centered.values

        # SVD
        n_comp_eff = min(n_components, min(X_train.shape) - 1) if min(X_train.shape) > 1 else 1
        if n_comp_eff < 1:
            return EvaluationResult(None, None, None, None, None, error="Không đủ dữ liệu để train SVD")

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
            return EvaluationResult(None, None, None, None, len(test_data), error="Không có sample test nào khớp train để tính RMSE/MAE")

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


def save_evaluation_result(result: EvaluationResult, output_path: str, meta: dict | None = None) -> None:
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "meta": meta or {},
        "result": result.to_dict(),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_evaluation_result(input_path: str) -> tuple[EvaluationResult, dict]:
    with open(input_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
    res = payload.get("result", {}) if isinstance(payload, dict) else {}
    return EvaluationResult.from_dict(res), payload if isinstance(payload, dict) else {"meta": meta, "result": res}


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate recommender model from ratings.csv and export metrics to JSON.")
    p.add_argument("--ratings", default="ratings.csv", help="Path to ratings CSV (needs userId, bookId, rating).")
    p.add_argument("--output", default="evaluation_results.json", help="Output JSON file path.")
    p.add_argument("--k", type=int, default=10, help="K for Precision@K / Recall@K.")
    p.add_argument("--relevant-threshold", type=int, default=4, help="Relevant if rating >= threshold.")
    p.add_argument("--max-users", type=int, default=1000, help="Max users to evaluate for ranking metrics.")
    p.add_argument("--max-train-users", type=int, default=4000, help="Max users used to build train pivot (anti-OOM). Use 0 to disable.")
    p.add_argument("--max-train-items", type=int, default=4000, help="Max items used to build train pivot (anti-OOM). Use 0 to disable.")
    p.add_argument("--test-size", type=float, default=0.2, help="Test split ratio.")
    p.add_argument("--random-state", type=int, default=42, help="Random seed.")
    p.add_argument("--n-components", type=int, default=50, help="SVD components.")
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    metrics = calculate_metrics(
        ratings_path=args.ratings,
        k=args.k,
        relevant_threshold=args.relevant_threshold,
        max_users=args.max_users,
        max_train_users=(None if args.max_train_users == 0 else args.max_train_users),
        max_train_items=(None if args.max_train_items == 0 else args.max_train_items),
        test_size=args.test_size,
        random_state=args.random_state,
        n_components=args.n_components,
    )
    save_evaluation_result(
        metrics,
        args.output,
        meta={
            "ratings_path": args.ratings,
            "k": args.k,
            "relevant_threshold": args.relevant_threshold,
            "max_users": args.max_users,
            "max_train_users": (None if args.max_train_users == 0 else args.max_train_users),
            "max_train_items": (None if args.max_train_items == 0 else args.max_train_items),
            "test_size": args.test_size,
            "random_state": args.random_state,
            "n_components": args.n_components,
        },
    )
    if metrics.error:
        print(f"[ERROR] Evaluation failed: {metrics.error}")
    else:
        print(
            "[OK] Saved evaluation results to "
            f"{args.output}: RMSE={metrics.rmse:.4f}, MAE={metrics.mae:.4f}, "
            f"Precision@{args.k}={metrics.precision_at_k:.4f}, Recall@{args.k}={metrics.recall_at_k:.4f}"
        )


