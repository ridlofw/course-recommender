\
"""
Exploratory Data Analysis (EDA) for the capstone recommender project.

Run:
    python src/eda.py

Outputs:
    - outputs/genre_counts.png
    - outputs/rating_dist.png
    - outputs/enrollment_dist.png (if enrollment column exists)
    - outputs/top20_courses.png
    - outputs/title_words.png
    - outputs/data_summary.txt
"""

from __future__ import annotations

from pathlib import Path
import re
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import load_data, explode_genres, ensure_outputs_dir, save_fig


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data = load_data(root / "data")
    out_dir = ensure_outputs_dir(root / "outputs")

    courses = data.courses.copy()
    ratings = data.ratings.copy()

    # Summary
    n_users = ratings["user_id"].nunique()
    n_courses = courses["course_id"].nunique()
    n_ratings = len(ratings)
    density = n_ratings / (n_users * n_courses)

    summary_lines = [
        "CAPSTONE RECOMMENDER — DATA SUMMARY",
        f"Users:   {n_users}",
        f"Courses: {n_courses}",
        f"Ratings: {n_ratings}",
        f"Matrix density (ratings / users*courses): {density:.4f}",
        "",
        "Ratings stats:",
        str(ratings["rating"].describe()),
        "",
        "Top 10 most-rated courses (by count):",
    ]
    top_courses = ratings["course_id"].value_counts().head(10).reset_index()
    top_courses.columns = ["course_id", "count"]
    top_courses = top_courses.merge(courses[["course_id", "title"]], on="course_id", how="left")
    summary_lines.append(top_courses.to_string(index=False))

    (out_dir / "data_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    # Rating distribution
    plt.figure(figsize=(7, 4))
    plt.hist(ratings["rating"], bins=20)
    plt.xlabel("Rating")
    plt.ylabel("Count")
    save_fig(out_dir / "rating_dist.png", title="Rating Distribution")

    # Genre counts
    g_long = explode_genres(courses)
    g_counts = g_long["genre"].value_counts().sort_values(ascending=True)
    plt.figure(figsize=(8, 5))
    plt.barh(g_counts.index, g_counts.values)
    plt.xlabel("Number of courses")
    plt.ylabel("Genre")
    save_fig(out_dir / "genre_counts.png", title="Courses per Genre")

    # Enrollment distribution (if available)
    if "enrollment" in courses.columns:
        plt.figure(figsize=(7, 4))
        plt.hist(courses["enrollment"].fillna(0), bins=30)
        plt.xlabel("Enrollment")
        plt.ylabel("Number of courses")
        save_fig(out_dir / "enrollment_dist.png", title="Course Enrollment Distribution")

    # Top 20 courses by number of ratings
    top20 = ratings["course_id"].value_counts().head(20).reset_index()
    top20.columns = ["course_id", "rating_count"]
    top20 = top20.merge(courses[["course_id", "title"]], on="course_id", how="left")
    top20 = top20.sort_values("rating_count", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(top20["title"].fillna(top20["course_id"].astype(str)), top20["rating_count"])
    plt.xlabel("Number of ratings")
    plt.ylabel("Course")
    save_fig(out_dir / "top20_courses.png", title="Top 20 Courses (Most Rated)")

    # Simple "word cloud-like" title words plot (no external dependency)
    words = []
    for t in courses["title"].fillna("").astype(str):
        toks = re.findall(r"[A-Za-z]{3,}", t.lower())
        words.extend([w for w in toks if w not in {"intro", "into", "hands", "hand", "complete"}])

    counts = Counter(words)
    top_words = counts.most_common(30)
    if top_words:
        labels, vals = zip(*top_words)
        plt.figure(figsize=(8, 4))
        plt.bar(labels, vals)
        plt.xticks(rotation=60, ha="right")
        plt.ylabel("Frequency")
        save_fig(out_dir / "title_words.png", title="Top Words in Course Titles")

        # Word cloud-ish scatter
        rng = np.random.default_rng(42)
        plt.figure(figsize=(10, 6))
        for i, (w, v) in enumerate(top_words[:40]):
            x, y = rng.random(), rng.random()
            plt.text(x, y, w, fontsize=8 + 0.7 * v, ha="center", va="center")
        plt.axis("off")
        save_fig(out_dir / "wordcloud_like.png", title="")

    print(f"[OK] EDA outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
