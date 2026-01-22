\
"""
Content-based recommender using course-to-course similarity.

Idea:
- Represent each course by TF-IDF vector from (title + genres).
- Recommend similar courses by cosine similarity.

Run demo:
    python src/content_similarity.py
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils import load_data


def build_tfidf(courses: pd.DataFrame) -> Tuple[TfidfVectorizer, any]:
    text = (courses["title"].fillna("") + " " + courses["genres"].fillna("").str.replace("|", " ", regex=False)).astype(str)
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    X = vec.fit_transform(text)
    return vec, X


def similar_courses(courses: pd.DataFrame, seed_course_id: int, top_k: int = 10) -> List[int]:
    courses_sorted = courses.sort_values("course_id").reset_index(drop=True)
    _, X = build_tfidf(courses_sorted)
    cid_list = courses_sorted["course_id"].tolist()
    if seed_course_id not in set(cid_list):
        return []

    idx = cid_list.index(seed_course_id)
    sims = cosine_similarity(X[idx], X).ravel()
    ranked = [cid for cid, s in sorted(zip(cid_list, sims), key=lambda x: x[1], reverse=True) if cid != seed_course_id]
    return ranked[:top_k]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data = load_data(root / "data")
    courses = data.courses.copy()

    # pick a seed course with many ratings if possible
    seed = int(courses["course_id"].sample(1, random_state=42).iloc[0])
    recs = similar_courses(courses, seed_course_id=seed, top_k=10)

    titles = courses.set_index("course_id")["title"].to_dict()
    print(f"Seed course: [{seed}] {titles.get(seed,'')}")
    print("Top 10 similar courses (TF-IDF cosine):")
    for i, cid in enumerate(recs, 1):
        print(f"{i:2d}. [{cid}] {titles.get(cid,'')}")


if __name__ == "__main__":
    main()
