\
"""
Content-based recommender using user profile clustering (KMeans).

Pipeline:
1) Build user genre profiles from training interactions.
2) Cluster users with KMeans.
3) For each cluster, recommend popular courses among users in that cluster.
4) Visualize clusters in 2D with PCA.

Run:
    python src/content_clustering.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from utils import load_data, split_train_test, course_genre_matrix, ensure_outputs_dir, save_fig


def build_user_profile_matrix(train: pd.DataFrame, courses: pd.DataFrame) -> Tuple[np.ndarray, List[int]]:
    courses_sorted = courses.sort_values("course_id").reset_index(drop=True)
    cmat, _ = course_genre_matrix(courses_sorted)
    cid_to_row = {cid: i for i, cid in enumerate(courses_sorted["course_id"].tolist())}

    user_ids = sorted(train["user_id"].unique().tolist())
    U = np.zeros((len(user_ids), cmat.shape[1]), dtype=np.float32)
    for i, uid in enumerate(user_ids):
        grp = train[train["user_id"] == uid]
        if grp.empty:
            continue
        r = grp["rating"].to_numpy(dtype=np.float32)
        w = (r - r.mean())
        w = (w - w.min()) / (max(1e-6, w.max() - w.min()))
        w = 0.2 + 0.8*w
        vec = np.zeros((cmat.shape[1],), dtype=np.float32)
        for cid, weight in zip(grp["course_id"].tolist(), w.tolist()):
            row = cid_to_row.get(int(cid))
            if row is not None:
                vec += weight * cmat[row]
        norm = np.linalg.norm(vec) + 1e-9
        U[i] = vec / norm
    return U, user_ids


def cluster_users(U: np.ndarray, n_clusters: int = 5, seed: int = 42) -> np.ndarray:
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto")
    labels = km.fit_predict(U)
    return labels


def cluster_popular_recs(train: pd.DataFrame, labels: np.ndarray, user_ids: List[int], top_k: int = 10) -> Dict[int, List[int]]:
    uid_to_label = {uid: int(lab) for uid, lab in zip(user_ids, labels)}
    train2 = train.copy()
    train2["cluster"] = train2["user_id"].map(uid_to_label)

    recs: Dict[int, List[int]] = {}
    for uid in user_ids:
        c = uid_to_label[uid]
        cluster_items = train2[train2["cluster"] == c]["course_id"].value_counts()
        seen = set(train2[train2["user_id"] == uid]["course_id"].tolist())
        ranked = [int(cid) for cid in cluster_items.index.tolist() if int(cid) not in seen]
        recs[uid] = ranked[:top_k]
    return recs


def save_cluster_plot(U: np.ndarray, labels: np.ndarray, out_dir: Path) -> None:
    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(U)
    plt.figure(figsize=(7, 5))
    plt.scatter(Z[:, 0], Z[:, 1], c=labels, s=18)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    save_fig(out_dir / "pca_clusters.png", title="User Profile Clusters (PCA + KMeans)")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data = load_data(root / "data")
    train, test = split_train_test(data.ratings, test_ratio=0.2, seed=42)

    out_dir = ensure_outputs_dir(root / "outputs")

    U, user_ids = build_user_profile_matrix(train, data.courses)
    labels = cluster_users(U, n_clusters=5, seed=42)
    save_cluster_plot(U, labels, out_dir)

    # Save cluster sizes
    cluster_sizes = pd.Series(labels).value_counts().sort_index().reset_index()
    cluster_sizes.columns = ["cluster", "n_users"]
    cluster_sizes.to_csv(out_dir / "cluster_sizes.csv", index=False)

    # Demo recommendations for one user
    recs = cluster_popular_recs(train, labels, user_ids, top_k=10)
    uid = user_ids[0]
    titles = data.courses.set_index("course_id")["title"].to_dict()
    print(f"User {uid} — Top 10 recommendations (Cluster Popularity):")
    for i, cid in enumerate(recs[uid], 1):
        print(f"{i:2d}. [{cid}] {titles.get(cid,'')}")

    print(f"[OK] Saved clustering outputs to: {out_dir}")


if __name__ == "__main__":
    main()
