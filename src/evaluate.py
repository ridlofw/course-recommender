\
"""
Train + evaluate multiple recommender approaches and export a metrics summary table.

Run:
    python src/evaluate.py

Outputs:
    - outputs/metrics_summary.csv
    - outputs/metrics_summary.md
    - outputs/metrics_precision_at10.png
    - outputs/metrics_recall_at10.png
    - outputs/metrics_rmse.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils import (
    load_data,
    split_train_test,
    build_mappings,
    build_sparse_matrix,
    precision_recall_at_k,
    rmse,
    course_genre_matrix,
    ensure_outputs_dir,
    save_fig,
)

# -----------------------------
# Content-based: user profile
# -----------------------------

def cb_user_profile_recs(train: pd.DataFrame, courses: pd.DataFrame, top_k: int = 10) -> Dict[int, List[int]]:
    courses_sorted = courses.sort_values("course_id").reset_index(drop=True)
    cmat, _ = course_genre_matrix(courses_sorted)
    cid_list = courses_sorted["course_id"].tolist()
    cid_to_row = {cid: i for i, cid in enumerate(cid_list)}

    recs: Dict[int, List[int]] = {}
    for uid, grp in train.groupby("user_id"):
        vec = np.zeros((cmat.shape[1],), dtype=np.float32)
        r = grp["rating"].to_numpy(dtype=np.float32)
        if len(r) == 0:
            recs[int(uid)] = []
            continue
        w = (r - r.min()) / (max(1e-6, r.max() - r.min()))
        w = 0.2 + 0.8 * w
        for cid, weight in zip(grp["course_id"].tolist(), w.tolist()):
            row = cid_to_row.get(int(cid))
            if row is not None:
                vec += weight * cmat[row]
        vec = vec / (np.linalg.norm(vec) + 1e-9)
        scores = (cmat @ vec).astype(float)

        seen = set(grp["course_id"].tolist())
        ranked = [cid for cid, s in sorted(zip(cid_list, scores), key=lambda x: x[1], reverse=True) if cid not in seen]
        recs[int(uid)] = ranked[:top_k]
    return recs


# -----------------------------
# Content-based: course similarity
# For each user, use their top-rated course as seed, recommend similar courses.
# -----------------------------

def build_tfidf(courses: pd.DataFrame) -> Tuple[TfidfVectorizer, sparse.spmatrix, List[int]]:
    courses_sorted = courses.sort_values("course_id").reset_index(drop=True)
    text = (courses_sorted["title"].fillna("") + " " + courses_sorted["genres"].fillna("").str.replace("|", " ", regex=False)).astype(str)
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    X = vec.fit_transform(text)
    cid_list = courses_sorted["course_id"].tolist()
    return vec, X, cid_list


def cb_course_similarity_recs(train: pd.DataFrame, courses: pd.DataFrame, top_k: int = 10) -> Dict[int, List[int]]:
    _, X, cid_list = build_tfidf(courses)
    cid_to_idx = {cid: i for i, cid in enumerate(cid_list)}
    sims_cache = {}  # seed_idx -> ranked course_ids

    recs: Dict[int, List[int]] = {}
    for uid, grp in train.groupby("user_id"):
        # seed course = highest rating (ties broken by count)
        seed_row = grp.sort_values(["rating"], ascending=False).iloc[0]
        seed_cid = int(seed_row["course_id"])
        if seed_cid not in cid_to_idx:
            recs[int(uid)] = []
            continue
        seed_idx = cid_to_idx[seed_cid]
        if seed_idx not in sims_cache:
            sims = cosine_similarity(X[seed_idx], X).ravel()
            ranked = [cid for cid, s in sorted(zip(cid_list, sims), key=lambda x: x[1], reverse=True) if cid != seed_cid]
            sims_cache[seed_idx] = ranked

        seen = set(grp["course_id"].tolist())
        ranked = [cid for cid in sims_cache[seed_idx] if cid not in seen]
        recs[int(uid)] = ranked[:top_k]
    return recs


# -----------------------------
# Content-based: user profile clustering
# Recs = popular within cluster
# -----------------------------

def cb_cluster_popular_recs(train: pd.DataFrame, courses: pd.DataFrame, n_clusters: int = 5, top_k: int = 10) -> Dict[int, List[int]]:
    # build user profiles
    courses_sorted = courses.sort_values("course_id").reset_index(drop=True)
    cmat, _ = course_genre_matrix(courses_sorted)
    cid_list = courses_sorted["course_id"].tolist()
    cid_to_row = {cid: i for i, cid in enumerate(cid_list)}

    user_ids = sorted(train["user_id"].unique().tolist())
    U = np.zeros((len(user_ids), cmat.shape[1]), dtype=np.float32)
    for i, uid in enumerate(user_ids):
        grp = train[train["user_id"] == uid]
        r = grp["rating"].to_numpy(dtype=np.float32)
        if len(r) == 0:
            continue
        w = (r - r.mean())
        w = (w - w.min()) / (max(1e-6, w.max() - w.min()))
        w = 0.2 + 0.8*w
        vec = np.zeros((cmat.shape[1],), dtype=np.float32)
        for cid, weight in zip(grp["course_id"].tolist(), w.tolist()):
            row = cid_to_row.get(int(cid))
            if row is not None:
                vec += weight * cmat[row]
        U[i] = vec / (np.linalg.norm(vec) + 1e-9)

    # KMeans
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = km.fit_predict(U)
    uid_to_cluster = {uid: int(lab) for uid, lab in zip(user_ids, labels)}

    train2 = train.copy()
    train2["cluster"] = train2["user_id"].map(uid_to_cluster)

    recs: Dict[int, List[int]] = {}
    for uid in user_ids:
        c = uid_to_cluster[uid]
        popular = train2[train2["cluster"] == c]["course_id"].value_counts().index.tolist()
        seen = set(train2[train2["user_id"] == uid]["course_id"].tolist())
        ranked = [int(cid) for cid in popular if int(cid) not in seen]
        recs[uid] = ranked[:top_k]
    return recs


# -----------------------------
# KNN CF: item-based
# -----------------------------

def fit_item_knn(R: sparse.csr_matrix, n_neighbors: int = 50) -> NearestNeighbors:
    model = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=n_neighbors)
    model.fit(R.T)
    return model


def knn_recs(train: pd.DataFrame, top_k: int = 10, n_neighbors: int = 50) -> Tuple[Dict[int, List[int]], Dict[str, object]]:
    u2i, it2i = build_mappings(train)
    R = build_sparse_matrix(train, u2i, it2i, kind="csr")
    model = fit_item_knn(R, n_neighbors=n_neighbors)
    idx_to_item = {v: k for k, v in it2i.items()}

    recs: Dict[int, List[int]] = {}
    for uid in u2i.keys():
        uid_idx = u2i[uid]
        user_row = R[uid_idx]
        rated_item_indices = user_row.indices
        rated_scores = user_row.data
        seen = set(train.loc[train["user_id"] == uid, "course_id"].tolist())

        scores = np.zeros((R.shape[1],), dtype=float)
        sim_sums = np.zeros((R.shape[1],), dtype=float)

        for it_idx, r in zip(rated_item_indices, rated_scores):
            dists, neigh = model.kneighbors(R.T[it_idx], n_neighbors=model.n_neighbors, return_distance=True)
            sims = 1.0 - dists.ravel()
            for n_idx, sim in zip(neigh.ravel(), sims):
                if sim <= 0:
                    continue
                scores[n_idx] += sim * float(r)
                sim_sums[n_idx] += sim

        pred = np.divide(scores, np.maximum(1e-9, sim_sums))
        ranked = np.argsort(-pred)
        out = []
        for it_idx in ranked:
            cid = idx_to_item[it_idx]
            if cid not in seen:
                out.append(cid)
            if len(out) >= top_k:
                break
        recs[uid] = out

    extra = {"u2i": u2i, "it2i": it2i, "R": R, "model": model}
    return recs, extra


def knn_predict_rating(
    user_id: int,
    course_id: int,
    train: pd.DataFrame,
    u2i: Dict[int, int],
    it2i: Dict[int, int],
    R: sparse.csr_matrix,
    model: NearestNeighbors,
    n_neighbors: int = 50,
) -> float:
    """Predict rating using neighbors of target item intersected with user's rated items."""
    if user_id not in u2i or course_id not in it2i:
        return float(train["rating"].mean())
    u = u2i[user_id]
    i = it2i[course_id]
    user_row = R[u]
    rated = dict(zip(user_row.indices.tolist(), user_row.data.tolist()))
    if not rated:
        return float(train["rating"].mean())

    dists, neigh = model.kneighbors(R.T[i], n_neighbors=n_neighbors, return_distance=True)
    neigh = neigh.ravel()
    sims = (1.0 - dists.ravel()).astype(float)

    num, den = 0.0, 0.0
    for n_idx, sim in zip(neigh, sims):
        if n_idx in rated and sim > 0:
            num += sim * float(rated[n_idx])
            den += sim
    if den <= 1e-9:
        return float(train["rating"].mean())
    return num / den


# -----------------------------
# NMF CF
# -----------------------------

def nmf_fit(train: pd.DataFrame, n_factors: int = 20) -> Tuple[Dict[int,int], Dict[int,int], sparse.csr_matrix, np.ndarray, np.ndarray]:
    u2i, it2i = build_mappings(train)
    R = build_sparse_matrix(train, u2i, it2i, kind="csr")
    model = NMF(n_components=n_factors, init="nndsvda", random_state=42, max_iter=300)
    W = model.fit_transform(R)
    H = model.components_
    return u2i, it2i, R, W, H


def nmf_recs(train: pd.DataFrame, u2i, it2i, W, H, top_k: int = 10) -> Dict[int, List[int]]:
    idx_to_item = {v: k for k, v in it2i.items()}
    recs: Dict[int, List[int]] = {}
    for uid, u_idx in u2i.items():
        seen = set(train.loc[train["user_id"] == uid, "course_id"].tolist())
        scores = W[u_idx] @ H
        ranked = np.argsort(-scores)
        out = []
        for it_idx in ranked:
            cid = idx_to_item[it_idx]
            if cid not in seen:
                out.append(cid)
            if len(out) >= top_k:
                break
        recs[uid] = out
    return recs


def nmf_predict_rating(user_id: int, course_id: int, train_mean: float, u2i, it2i, W, H) -> float:
    if user_id not in u2i or course_id not in it2i:
        return float(train_mean)
    return float(W[u2i[user_id]] @ H[:, it2i[course_id]])


# -----------------------------
# Neural Embedding MF (NumPy SGD)
# -----------------------------

def mf_sgd_train(train: pd.DataFrame, u2i, it2i, n_factors: int = 32, lr: float = 0.02, reg: float = 0.02, n_epochs: int = 8):
    rng = np.random.default_rng(42)
    n_users = len(u2i)
    n_items = len(it2i)

    U = 0.01 * rng.standard_normal((n_users, n_factors)).astype(np.float32)
    V = 0.01 * rng.standard_normal((n_items, n_factors)).astype(np.float32)
    bu = np.zeros((n_users,), dtype=np.float32)
    bi = np.zeros((n_items,), dtype=np.float32)
    mu = float(train["rating"].mean())

    rows = train["user_id"].map(u2i).to_numpy()
    cols = train["course_id"].map(it2i).to_numpy()
    y = train["rating"].to_numpy(dtype=np.float32)

    idx = np.arange(len(train))
    for _ in range(n_epochs):
        rng.shuffle(idx)
        for j in idx:
            u = rows[j]
            i = cols[j]
            r = y[j]
            pred = mu + bu[u] + bi[i] + float(np.dot(U[u], V[i]))
            err = r - pred
            bu[u] += lr * (err - reg * bu[u])
            bi[i] += lr * (err - reg * bi[i])
            u_old = U[u].copy()
            v_old = V[i].copy()
            U[u] += lr * (err * v_old - reg * u_old)
            V[i] += lr * (err * u_old - reg * v_old)
    return U, V, bu, bi, mu


def mf_recs(train: pd.DataFrame, u2i, it2i, U, V, bu, bi, mu, top_k: int = 10) -> Dict[int, List[int]]:
    idx_to_item = {v: k for k, v in it2i.items()}
    recs: Dict[int, List[int]] = {}
    for uid, u_idx in u2i.items():
        seen = set(train.loc[train["user_id"] == uid, "course_id"].tolist())
        scores = mu + bu[u_idx] + bi + (V @ U[u_idx])
        ranked = np.argsort(-scores)
        out = []
        for it_idx in ranked:
            cid = idx_to_item[it_idx]
            if cid not in seen:
                out.append(cid)
            if len(out) >= top_k:
                break
        recs[uid] = out
    return recs


def mf_predict_rating(user_id: int, course_id: int, train_mean: float, u2i, it2i, U, V, bu, bi, mu) -> float:
    if user_id not in u2i or course_id not in it2i:
        return float(train_mean)
    u = u2i[user_id]
    i = it2i[course_id]
    return float(mu + bu[u] + bi[i] + np.dot(U[u], V[i]))


# -----------------------------
# Main evaluation
# -----------------------------

def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_dir = ensure_outputs_dir(root / "outputs")

    data = load_data(root / "data")
    train, test = split_train_test(data.ratings, test_ratio=0.2, seed=42)
    train_mean = float(train["rating"].mean())

    results = []

    # 1) Content: user profile
    recs = cb_user_profile_recs(train, data.courses, top_k=10)
    p, r = precision_recall_at_k(recs, test, k=10, relevant_rating=4.0)
    results.append({"model": "Content-UserProfile", "precision@10": p, "recall@10": r, "rmse": np.nan})

    # 2) Content: course similarity
    recs = cb_course_similarity_recs(train, data.courses, top_k=10)
    p, r = precision_recall_at_k(recs, test, k=10, relevant_rating=4.0)
    results.append({"model": "Content-CourseSimilarity", "precision@10": p, "recall@10": r, "rmse": np.nan})

    # 3) Content: clustering
    recs = cb_cluster_popular_recs(train, data.courses, n_clusters=5, top_k=10)
    p, r = precision_recall_at_k(recs, test, k=10, relevant_rating=4.0)
    results.append({"model": "Content-UserClustering", "precision@10": p, "recall@10": r, "rmse": np.nan})

    # 4) CF: KNN item-based
    recs, extra = knn_recs(train, top_k=10, n_neighbors=50)
    p, r = precision_recall_at_k(recs, test, k=10, relevant_rating=4.0)

    # RMSE for KNN is optional and can be expensive; we skip it by default.
    results.append({"model": "CF-ItemKNN", "precision@10": p, "recall@10": r, "rmse": np.nan})

    # 5) CF: NMF
    u2i, it2i, R, W, H = nmf_fit(train, n_factors=20)
    recs = nmf_recs(train, u2i, it2i, W, H, top_k=10)
    p, r = precision_recall_at_k(recs, test, k=10, relevant_rating=4.0)
    y_true, y_pred = [], []
    for _, row in test.iterrows():
        uid = int(row["user_id"])
        cid = int(row["course_id"])
        if uid in u2i and cid in it2i:
            y_true.append(float(row["rating"]))
            y_pred.append(nmf_predict_rating(uid, cid, train_mean, u2i, it2i, W, H))
    nmf_rmse = rmse(np.array(y_true), np.array(y_pred)) if y_true else np.nan
    results.append({"model": "CF-NMF", "precision@10": p, "recall@10": r, "rmse": nmf_rmse})

    # 6) CF: Neural embedding MF (SGD)
    u2i2, it2i2 = build_mappings(train)
    U, V, bu, bi, mu = mf_sgd_train(train, u2i2, it2i2, n_factors=32, lr=0.02, reg=0.02, n_epochs=8)
    recs = mf_recs(train, u2i2, it2i2, U, V, bu, bi, mu, top_k=10)
    p, r = precision_recall_at_k(recs, test, k=10, relevant_rating=4.0)
    y_true, y_pred = [], []
    for _, row in test.iterrows():
        uid = int(row["user_id"])
        cid = int(row["course_id"])
        if uid in u2i2 and cid in it2i2:
            y_true.append(float(row["rating"]))
            y_pred.append(mf_predict_rating(uid, cid, train_mean, u2i2, it2i2, U, V, bu, bi, mu))
    mf_rmse = rmse(np.array(y_true), np.array(y_pred)) if y_true else np.nan
    results.append({"model": "CF-NeuralEmbedding", "precision@10": p, "recall@10": r, "rmse": mf_rmse})

    df = pd.DataFrame(results).sort_values("model")
    df.to_csv(out_dir / "metrics_summary.csv", index=False)

    # Markdown version (easy to paste into slides / report)
    md = df.to_markdown(index=False)
    (out_dir / "metrics_summary.md").write_text(md, encoding="utf-8")

    # Plots for slides
    # Precision@10
    plt.figure(figsize=(8, 4))
    plt.bar(df["model"], df["precision@10"])
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Precision@10")
    save_fig(out_dir / "metrics_precision_at10.png", title="Model Comparison — Precision@10")

    # Recall@10
    plt.figure(figsize=(8, 4))
    plt.bar(df["model"], df["recall@10"])
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Recall@10")
    save_fig(out_dir / "metrics_recall_at10.png", title="Model Comparison — Recall@10")

    # RMSE (drop NaN)
    rmse_df = df.dropna(subset=["rmse"]).copy()
    if len(rmse_df) > 0:
        plt.figure(figsize=(7, 4))
        plt.bar(rmse_df["model"], rmse_df["rmse"])
        plt.xticks(rotation=35, ha="right")
        plt.ylabel("RMSE")
        save_fig(out_dir / "metrics_rmse.png", title="Model Comparison — RMSE (Lower is Better)")

    print("[OK] Saved:")
    print(f"- {out_dir / 'metrics_summary.csv'}")
    print(f"- {out_dir / 'metrics_precision_at10.png'}")
    print(f"- {out_dir / 'metrics_recall_at10.png'}")
    print(f"- {out_dir / 'metrics_rmse.png'} (if applicable)")


if __name__ == "__main__":
    main()
