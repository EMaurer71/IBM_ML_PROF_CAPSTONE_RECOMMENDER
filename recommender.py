"""
recommender.py
Core recommendation logic for the Course Recommender System.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine


# -----------------------------------------------------------------------------
# Helper: Compute cosine similarity between two numeric vectors
# -----------------------------------------------------------------------------
def cosine_sim(vec1, vec2):
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    return 1 - cosine(vec1, vec2)


# -----------------------------------------------------------------------------
# Helper: Pivot two BoW rows into aligned vectors
# -----------------------------------------------------------------------------
def pivot_two_bows(base_df, compare_df):
    """
    Takes two single-row DataFrames from courses_bows.csv
    and aligns them into a 2-row numeric matrix.
    """
    base = base_df.copy()
    base["type"] = "base"

    compare = compare_df.copy()
    compare["type"] = "compare"

    combined = pd.concat([base, compare], axis=0)

    # Drop non-numeric columns
    non_numeric = ["doc_id", "type"]
    numeric_cols = [c for c in combined.columns if c not in non_numeric]

    return combined[numeric_cols].astype(float)


# -----------------------------------------------------------------------------
# Content-Based Similarity: Find similar courses
# -----------------------------------------------------------------------------
def get_similar_courses(course_id, courses_df, bows_df, top_n=10):
    """
    Returns top-N similar courses using Bag-of-Words cosine similarity.
    """
    if course_id not in bows_df["doc_id"].values:
        return {"error": f"Course ID {course_id} not found."}

    target_bow = bows_df[bows_df["doc_id"] == course_id]

    similarities = []

    for other_id in bows_df["doc_id"].unique():
        if other_id == course_id:
            continue

        other_bow = bows_df[bows_df["doc_id"] == other_id]

        # Align vectors
        df = pivot_two_bows(target_bow, other_bow)
        vec1 = df.iloc[0].values
        vec2 = df.iloc[1].values

        sim = cosine_sim(vec1, vec2)
        similarities.append((other_id, sim))

    # Sort by similarity
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]

    # Merge with course metadata
    results = []
    for cid, score in similarities:
        row = courses_df[courses_df["COURSE_ID"] == cid].iloc[0]
        results.append({
            "course_id": cid,
            "title": row["TITLE"],
            "description": row["DESCRIPTION"],
            "similarity": float(score)
        })

    return results


# -----------------------------------------------------------------------------
# User-Based Recommendations (Simple Baseline)
# -----------------------------------------------------------------------------
def recommend_for_user(user_id, courses_df, ratings_df, top_n=10):
    """
    Simple user-based recommender:
    - Finds courses the user has NOT rated
    - Scores them by global popularity (mean rating)
    - Returns top-N
    """
    if user_id not in ratings_df["user"].unique():
        return {"error": f"User ID {user_id} not found."}

    # Courses user has already rated
    rated = ratings_df[ratings_df["user"] == user_id]["item"].unique()

    # Compute global popularity
    popularity = (
        ratings_df.groupby("item")["rating"]
        .mean()
        .reset_index()
        .rename(columns={"rating": "score"})
    )

    # Filter out courses the user already rated
    popularity = popularity[~popularity["item"].isin(rated)]

    # Top-N
    top = popularity.sort_values("score", ascending=False).head(top_n)

    # Merge with metadata
    results = []
    for _, row in top.iterrows():
        cid = row["item"]
        meta = courses_df[courses_df["COURSE_ID"] == cid].iloc[0]

        results.append({
            "course_id": cid,
            "title": meta["TITLE"],
            "description": meta["DESCRIPTION"],
            "score": float(row["score"])
        })

    return results

