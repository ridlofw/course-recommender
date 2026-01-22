\
"""
NMF-based Collaborative Filtering.

Approach:
- Build user-item matrix R (missing -> 0).
- Fit Non-negative Matrix Factorization: R ≈ W @ H
- Use reconstructed scores to recommend top items.

Run demo:
    python src/cf_nmf.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import NMF

from utils import load_data, split_train_test, build_mappings, build_sparse_matrix


def fit_nmf(R: sparse.csr_matrix, n_factors: int = 20, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    model = NMF(
        n_components=n_factors,
        init="nndsvda",
        random_state=seed,
        max_iter=300,
        alpha_W=0.0,
        alpha_H=0.0,
        l1_ratio=0.0,
    )
    W = model.fit_transform(R)
    H = model.components_
    return W, H


def recommend_user_nmf(
    user_id: int,
    train: pd.DataFrame,
    u2i: Dict[int, int],
    it2i: Dict[int, int],
    W: np.ndarray,
    H: np.ndarray,
    top_k: int = 10,
) -> List[int]:
    uid_idx = u2i.get(user_id)
    if uid_idx is None:
        return []

    seen = set(train.loc[train["user_id"] == user_id, "course_id"].tolist())
    idx_to_item = {v: k for k, v in it2i.items()}

    scores = W[uid_idx] @ H
    ranked = np.argsort(-scores)

    recs = []
    for it_idx in ranked:
        cid = idx_to_item[it_idx]
        if cid not in seen:
            recs.append(cid)
        if len(recs) >= top_k:
            break
    return recs


def batch_recommend(train: pd.DataFrame, top_k: int = 10, n_factors: int = 20) -> Dict[int, List[int]]:
    u2i, it2i = build_mappings(train)
    R = build_sparse_matrix(train, u2i, it2i, kind="csr")
    W, H = fit_nmf(R, n_factors=n_factors, seed=42)

    recs: Dict[int, List[int]] = {}
    for uid in u2i.keys():
        recs[uid] = recommend_user_nmf(uid, train, u2i, it2i, W, H, top_k=top_k)
    return recs


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data = load_data(root / "data")
    train, test = split_train_test(data.ratings, test_ratio=0.2, seed=42)

    u2i, it2i = build_mappings(train)
    R = build_sparse_matrix(train, u2i, it2i, kind="csr")
    W, H = fit_nmf(R, n_factors=20, seed=42)

    uid = int(train["user_id"].sample(1, random_state=42).iloc[0])
    recs = recommend_user_nmf(uid, train, u2i, it2i, W, H, top_k=10)

    titles = data.courses.set_index("course_id")["title"].to_dict()
    print(f"User {uid} — Top 10 recommendations (NMF CF):")
    for i, cid in enumerate(recs, 1):
        print(f"{i:2d}. [{cid}] {titles.get(cid,'')}")


if __name__ == "__main__":
    main()
