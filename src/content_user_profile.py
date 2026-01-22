\
"""
Content-based recommender using user profile (genre preference).

Idea:
- Each course -> multi-hot genre vector.
- Each user profile -> weighted average of genres from courses they've rated.
- Score unseen courses by cosine similarity(user_profile, course_vector).

Run demo:
    python src/content_user_profile.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from utils import load_data, split_train_test, course_genre_matrix


def fit_user_profiles(train: pd.DataFrame, courses: pd.DataFrame) -> Dict[int, np.ndarray]:
    courses_sorted = courses.sort_values("course_id").reset_index(drop=True)
    cmat, genre_list = course_genre_matrix(courses_sorted)
    cid_to_row = {cid: i for i, cid in enumerate(courses_sorted["course_id"].tolist())}

    profiles: Dict[int, np.ndarray] = {}
    for uid, grp in train.groupby("user_id"):
        vec = np.zeros((cmat.shape[1],), dtype=np.float32)
        # normalize ratings to [0,1]
        ratings = grp["rating"].to_numpy(dtype=np.float32)
        if len(ratings) == 0:
            profiles[int(uid)] = vec
            continue
        w = (ratings - ratings.min()) / (max(1e-6, ratings.max() - ratings.min()))
        w = 0.2 + 0.8 * w  # avoid all-zero weights
        for (cid, weight) in zip(grp["course_id"].tolist(), w.tolist()):
            r = cid_to_row.get(int(cid))
            if r is not None:
                vec += weight * cmat[r]
        # L2 normalize
        norm = np.linalg.norm(vec) + 1e-9
        profiles[int(uid)] = vec / norm
    return profiles


def recommend(
    user_id: int,
    train: pd.DataFrame,
    courses: pd.DataFrame,
    top_k: int = 10,
) -> List[int]:
    courses_sorted = courses.sort_values("course_id").reset_index(drop=True)
    cmat, _ = course_genre_matrix(courses_sorted)
    cid_list = courses_sorted["course_id"].tolist()

    profiles = fit_user_profiles(train, courses_sorted)
    uvec = profiles.get(int(user_id))
    if uvec is None:
        return []

    # scores via cosine similarity
    scores = cosine_similarity(uvec.reshape(1, -1), cmat).ravel()

    seen = set(train.loc[train["user_id"] == user_id, "course_id"].tolist())
    ranked = [cid for cid, s in sorted(zip(cid_list, scores), key=lambda x: x[1], reverse=True) if cid not in seen]
    return ranked[:top_k]


def batch_recommend(train: pd.DataFrame, courses: pd.DataFrame, top_k: int = 10) -> Dict[int, List[int]]:
    profiles = fit_user_profiles(train, courses)
    courses_sorted = courses.sort_values("course_id").reset_index(drop=True)
    cmat, _ = course_genre_matrix(courses_sorted)
    cid_list = courses_sorted["course_id"].tolist()

    recs: Dict[int, List[int]] = {}
    for uid, uvec in profiles.items():
        scores = (cmat @ uvec).astype(float)  # cosine because uvec is normalized and cvec is binary
        seen = set(train.loc[train["user_id"] == uid, "course_id"].tolist())
        ranked = [cid for cid, s in sorted(zip(cid_list, scores), key=lambda x: x[1], reverse=True) if cid not in seen]
        recs[uid] = ranked[:top_k]
    return recs


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data = load_data(root / "data")
    train, test = split_train_test(data.ratings, test_ratio=0.2, seed=42)

    uid = int(train["user_id"].sample(1, random_state=42).iloc[0])
    recs = recommend(uid, train, data.courses, top_k=10)
    titles = data.courses.set_index("course_id")["title"].to_dict()

    print(f"User {uid} — Top 10 recommendations (User Profile / Genres):")
    for i, cid in enumerate(recs, 1):
        print(f"{i:2d}. [{cid}] {titles.get(cid, '')}")


if __name__ == "__main__":
    main()
