\
"""
KNN-based Collaborative Filtering (item-based).

Approach:
- Build user-item rating matrix from train.
- Fit item-item nearest neighbors using cosine distance.
- Predict a user's preference for unseen items via weighted average of neighbor items the user has rated.

Run demo:
    python src/cf_knn.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

from utils import load_data, split_train_test, build_mappings, build_sparse_matrix


def fit_item_knn(R: sparse.csr_matrix, n_neighbors: int = 50) -> NearestNeighbors:
    model = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=n_neighbors)
    # item vectors: (n_items, n_users)
    model.fit(R.T)
    return model


def recommend_user_item_knn(
    user_id: int,
    train: pd.DataFrame,
    u2i: Dict[int, int],
    it2i: Dict[int, int],
    R: sparse.csr_matrix,
    model: NearestNeighbors,
    top_k: int = 10,
) -> List[int]:
    uid_idx = u2i.get(user_id)
    if uid_idx is None:
        return []

    # items the user has rated
    user_row = R[uid_idx]
    rated_item_indices = user_row.indices
    rated_scores = user_row.data

    seen_course_ids = set(train.loc[train["user_id"] == user_id, "course_id"].tolist())
    idx_to_item = {v: k for k, v in it2i.items()}

    # score candidates
    scores = np.zeros((R.shape[1],), dtype=float)
    sim_sums = np.zeros((R.shape[1],), dtype=float)

    for it_idx, r in zip(rated_item_indices, rated_scores):
        # neighbors of this rated item
        dists, neigh = model.kneighbors(R.T[it_idx], n_neighbors=model.n_neighbors, return_distance=True)
        dists = dists.ravel()
        neigh = neigh.ravel()
        sims = 1.0 - dists  # cosine similarity
        for n_idx, sim in zip(neigh, sims):
            if sim <= 0:
                continue
            scores[n_idx] += sim * float(r)
            sim_sums[n_idx] += sim

    # avoid division by zero
    pred = np.divide(scores, np.maximum(1e-9, sim_sums))

    # rank unseen
    ranked_item_indices = np.argsort(-pred)
    recs = []
    for it_idx in ranked_item_indices:
        cid = idx_to_item[it_idx]
        if cid not in seen_course_ids:
            recs.append(cid)
        if len(recs) >= top_k:
            break
    return recs


def batch_recommend(train: pd.DataFrame, top_k: int = 10, n_neighbors: int = 50) -> Dict[int, List[int]]:
    u2i, it2i = build_mappings(train)
    R = build_sparse_matrix(train, u2i, it2i, kind="csr")
    model = fit_item_knn(R, n_neighbors=n_neighbors)

    recs: Dict[int, List[int]] = {}
    for uid in u2i.keys():
        recs[uid] = recommend_user_item_knn(uid, train, u2i, it2i, R, model, top_k=top_k)
    return recs


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data = load_data(root / "data")
    train, test = split_train_test(data.ratings, test_ratio=0.2, seed=42)

    u2i, it2i = build_mappings(train)
    R = build_sparse_matrix(train, u2i, it2i, kind="csr")
    model = fit_item_knn(R, n_neighbors=50)

    uid = int(train["user_id"].sample(1, random_state=42).iloc[0])
    recs = recommend_user_item_knn(uid, train, u2i, it2i, R, model, top_k=10)

    titles = data.courses.set_index("course_id")["title"].to_dict()
    print(f"User {uid} — Top 10 recommendations (Item-KNN CF):")
    for i, cid in enumerate(recs, 1):
        print(f"{i:2d}. [{cid}] {titles.get(cid,'')}")


if __name__ == "__main__":
    main()
