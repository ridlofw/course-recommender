\
"""
Neural Network Embedding-based Collaborative Filtering (lightweight, NumPy).

This is a "neural embedding" style model:
- learn user embeddings + item embeddings + bias terms
- prediction = global_mean + user_bias + item_bias + dot(user_emb, item_emb)

Trained with SGD on MSE loss.

Run demo:
    python src/cf_neural_embedding.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from utils import load_data, split_train_test, build_mappings, rmse


def train_mf_sgd(
    train: pd.DataFrame,
    u2i: Dict[int, int],
    it2i: Dict[int, int],
    n_factors: int = 32,
    lr: float = 0.02,
    reg: float = 0.02,
    n_epochs: int = 10,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    n_users = len(u2i)
    n_items = len(it2i)

    user_emb = 0.01 * rng.standard_normal((n_users, n_factors)).astype(np.float32)
    item_emb = 0.01 * rng.standard_normal((n_items, n_factors)).astype(np.float32)
    user_bias = np.zeros((n_users,), dtype=np.float32)
    item_bias = np.zeros((n_items,), dtype=np.float32)
    global_mean = float(train["rating"].mean())

    rows = train["user_id"].map(u2i).to_numpy()
    cols = train["course_id"].map(it2i).to_numpy()
    y = train["rating"].to_numpy(dtype=np.float32)

    idx = np.arange(len(train))
    for epoch in range(1, n_epochs + 1):
        rng.shuffle(idx)
        for j in idx:
            u = rows[j]
            i = cols[j]
            r = y[j]

            pred = global_mean + user_bias[u] + item_bias[i] + float(np.dot(user_emb[u], item_emb[i]))
            err = r - pred

            # gradients
            ub = user_bias[u]
            ib = item_bias[i]
            ue = user_emb[u].copy()
            ie = item_emb[i].copy()

            user_bias[u] += lr * (err - reg * ub)
            item_bias[i] += lr * (err - reg * ib)
            user_emb[u] += lr * (err * ie - reg * ue)
            item_emb[i] += lr * (err * ue - reg * ie)

        # quick training RMSE
        preds = global_mean + user_bias[rows] + item_bias[cols] + np.sum(user_emb[rows] * item_emb[cols], axis=1)
        tr_rmse = rmse(y, preds)
        print(f"Epoch {epoch:02d}/{n_epochs} — train RMSE: {tr_rmse:.4f}")

    return user_emb, item_emb, user_bias, item_bias, global_mean


def predict_single(
    user_id: int,
    course_id: int,
    u2i: Dict[int, int],
    it2i: Dict[int, int],
    user_emb: np.ndarray,
    item_emb: np.ndarray,
    user_bias: np.ndarray,
    item_bias: np.ndarray,
    global_mean: float,
) -> float:
    if user_id not in u2i or course_id not in it2i:
        return global_mean
    u = u2i[user_id]
    i = it2i[course_id]
    return float(global_mean + user_bias[u] + item_bias[i] + np.dot(user_emb[u], item_emb[i]))


def recommend_user(
    user_id: int,
    train: pd.DataFrame,
    u2i: Dict[int, int],
    it2i: Dict[int, int],
    user_emb: np.ndarray,
    item_emb: np.ndarray,
    user_bias: np.ndarray,
    item_bias: np.ndarray,
    global_mean: float,
    top_k: int = 10,
) -> List[int]:
    if user_id not in u2i:
        return []
    u = u2i[user_id]
    seen = set(train.loc[train["user_id"] == user_id, "course_id"].tolist())
    idx_to_item = {v: k for k, v in it2i.items()}

    scores = global_mean + user_bias[u] + item_bias + (item_emb @ user_emb[u])
    ranked = np.argsort(-scores)

    recs = []
    for it_idx in ranked:
        cid = idx_to_item[it_idx]
        if cid not in seen:
            recs.append(cid)
        if len(recs) >= top_k:
            break
    return recs


def batch_recommend(train: pd.DataFrame, top_k: int = 10) -> Dict[int, List[int]]:
    u2i, it2i = build_mappings(train)
    user_emb, item_emb, user_bias, item_bias, gmean = train_mf_sgd(
        train, u2i, it2i, n_factors=32, lr=0.02, reg=0.02, n_epochs=10, seed=42
    )

    recs: Dict[int, List[int]] = {}
    for uid in u2i.keys():
        recs[uid] = recommend_user(uid, train, u2i, it2i, user_emb, item_emb, user_bias, item_bias, gmean, top_k=top_k)
    return recs


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data = load_data(root / "data")
    train, test = split_train_test(data.ratings, test_ratio=0.2, seed=42)

    u2i, it2i = build_mappings(train)
    user_emb, item_emb, user_bias, item_bias, gmean = train_mf_sgd(
        train, u2i, it2i, n_factors=32, lr=0.02, reg=0.02, n_epochs=8, seed=42
    )

    # quick test RMSE
    rows = test["user_id"].map(u2i).fillna(-1).astype(int).to_numpy()
    cols = test["course_id"].map(it2i).fillna(-1).astype(int).to_numpy()
    y = test["rating"].to_numpy(dtype=np.float32)

    mask = (rows >= 0) & (cols >= 0)
    preds = []
    for u, i in zip(rows[mask], cols[mask]):
        preds.append(gmean + user_bias[u] + item_bias[i] + float(np.dot(user_emb[u], item_emb[i])))
    preds = np.array(preds, dtype=float)
    print(f"Test RMSE (known users/items only): {rmse(y[mask], preds):.4f}")

    uid = int(train["user_id"].sample(1, random_state=42).iloc[0])
    recs = recommend_user(uid, train, u2i, it2i, user_emb, item_emb, user_bias, item_bias, gmean, top_k=10)

    titles = data.courses.set_index("course_id")["title"].to_dict()
    print(f"User {uid} — Top 10 recommendations (Neural Embedding MF):")
    for i, cid in enumerate(recs, 1):
        print(f"{i:2d}. [{cid}] {titles.get(cid,'')}")


if __name__ == "__main__":
    main()
