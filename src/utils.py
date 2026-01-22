\
"""
Utility functions for the capstone recommender project.
Author: Ridlo Fanata Wicaksana
Date: 22 Jan 2026 (Asia/Jakarta)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional

import numpy as np
import pandas as pd
from scipy import sparse
import matplotlib.pyplot as plt


@dataclass
class DataBundle:
    courses: pd.DataFrame
    ratings: pd.DataFrame


def load_data(data_dir: str | Path) -> DataBundle:
    """Load courses + ratings from CSV files inside `data_dir`."""
    data_dir = Path(data_dir)
    courses_path = data_dir / "courses.csv"
    ratings_path = data_dir / "ratings.csv"

    if not courses_path.exists() or not ratings_path.exists():
        raise FileNotFoundError(
            "Missing dataset. Expected files:\n"
            f"- {courses_path}\n"
            f"- {ratings_path}\n"
            "See data/README_dataset.md for expected columns."
        )

    courses = pd.read_csv(courses_path)
    ratings = pd.read_csv(ratings_path)

    # Basic validation
    required_courses = {"course_id", "title", "genres"}
    required_ratings = {"user_id", "course_id", "rating"}
    if not required_courses.issubset(courses.columns):
        raise ValueError(f"courses.csv must contain columns: {sorted(required_courses)}")
    if not required_ratings.issubset(ratings.columns):
        raise ValueError(f"ratings.csv must contain columns: {sorted(required_ratings)}")

    courses["course_id"] = courses["course_id"].astype(int)
    ratings["user_id"] = ratings["user_id"].astype(int)
    ratings["course_id"] = ratings["course_id"].astype(int)
    ratings["rating"] = ratings["rating"].astype(float)

    return DataBundle(courses=courses, ratings=ratings)


def explode_genres(courses: pd.DataFrame) -> pd.DataFrame:
    """Return a long table with one row per (course_id, genre)."""
    df = courses[["course_id", "title", "genres"]].copy()
    df["genre"] = df["genres"].fillna("").astype(str).str.split("|")
    df = df.explode("genre")
    df["genre"] = df["genre"].str.strip()
    df = df[df["genre"] != ""]
    return df[["course_id", "title", "genre"]]


def make_genre_list(courses: pd.DataFrame) -> List[str]:
    long_df = explode_genres(courses)
    return sorted(long_df["genre"].unique().tolist())


def course_genre_matrix(courses: pd.DataFrame, genre_list: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str]]:
    """Multi-hot course-genre matrix of shape (n_courses, n_genres)."""
    if genre_list is None:
        genre_list = make_genre_list(courses)
    g2i = {g: i for i, g in enumerate(genre_list)}

    courses_sorted = courses.sort_values("course_id").reset_index(drop=True)
    mat = np.zeros((len(courses_sorted), len(genre_list)), dtype=np.float32)
    for r, genres in enumerate(courses_sorted["genres"].fillna("").astype(str)):
        for g in genres.split("|"):
            g = g.strip()
            if g and g in g2i:
                mat[r, g2i[g]] = 1.0
    return mat, genre_list


def split_train_test(ratings: pd.DataFrame, test_ratio: float = 0.2, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Per-user random split to avoid cold-start users in test.
    Ensures each user keeps at least 5 interactions in train when possible.
    """
    rng = np.random.default_rng(seed)
    train_parts, test_parts = [], []
    for uid, grp in ratings.groupby("user_id"):
        grp = grp.sample(frac=1.0, random_state=int(rng.integers(0, 1_000_000)))
        n = len(grp)
        n_test = int(np.floor(n * test_ratio))
        if n - n_test < 5:
            n_test = max(0, n - 5)
        test_parts.append(grp.iloc[:n_test])
        train_parts.append(grp.iloc[n_test:])
    train = pd.concat(train_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)
    return train, test


def build_mappings(ratings: pd.DataFrame) -> Tuple[Dict[int, int], Dict[int, int]]:
    """Map user_id/course_id to contiguous indices."""
    users = sorted(ratings["user_id"].unique().tolist())
    items = sorted(ratings["course_id"].unique().tolist())
    u2i = {u: i for i, u in enumerate(users)}
    it2i = {it: i for i, it in enumerate(items)}
    return u2i, it2i


def build_sparse_matrix(
    ratings: pd.DataFrame,
    u2i: Dict[int, int],
    it2i: Dict[int, int],
    kind: str = "csr",
) -> sparse.spmatrix:
    """Create sparse user-item matrix from ratings."""
    rows = ratings["user_id"].map(u2i).to_numpy()
    cols = ratings["course_id"].map(it2i).to_numpy()
    data = ratings["rating"].to_numpy(dtype=np.float32)
    mat = sparse.coo_matrix((data, (rows, cols)), shape=(len(u2i), len(it2i)))
    return mat.tocsr() if kind == "csr" else mat.tocsc()


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def precision_recall_at_k(
    recommendations: Dict[int, List[int]],
    test: pd.DataFrame,
    k: int = 10,
    relevant_rating: float = 4.0,
) -> Tuple[float, float]:
    """
    Compute mean Precision@k and Recall@k over users.
    A test interaction is considered 'relevant' if rating >= relevant_rating.
    recommendations maps user_id -> list of course_id ranked.
    """
    rel = test[test["rating"] >= relevant_rating]
    truth = rel.groupby("user_id")["course_id"].apply(set).to_dict()

    precisions, recalls = [], []
    for uid, recs in recommendations.items():
        if uid not in truth:
            continue
        topk = recs[:k]
        hit = sum(1 for it in topk if it in truth[uid])
        precisions.append(hit / max(1, len(topk)))
        recalls.append(hit / max(1, len(truth[uid])))
    if not precisions:
        return 0.0, 0.0
    return float(np.mean(precisions)), float(np.mean(recalls))


def save_fig(path: str | Path, title: str = "") -> None:
    """Save current matplotlib figure with consistent layout."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def ensure_outputs_dir(outputs_dir: str | Path) -> Path:
    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    return outputs_dir
