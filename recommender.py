import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------
# Utility: clean pivot of two BoW vectors
# ---------------------------------------------------------
def pivot_two_bows(bow1, bow2):
    """
    Merge two bag-of-words DataFrames on 'token' and return
    a clean numeric 2-column matrix for cosine similarity.
    """
    merged = bow1.merge(
        bow2,
        on="token",
        how="outer",
        suffixes=("_1", "_2")
    )

    # Fill missing counts with 0
    merged["count_1"] = merged["count_1"].fillna(0)
    merged["count_2"] = merged["count_2"].fillna(0)

    # Only return numeric columns
    return merged[["count_1", "count_2"]].astype(float)


# ---------------------------------------------------------
# Content-based similarity for a single course
# ---------------------------------------------------------
def get_similar_courses(course_id, courses_df, bows_df, top_k=5):
    """
    Given a course_id, compute cosine similarity against all other courses
    using bag-of-words vectors.
    """
    # Extract target BoW
    target_bow = bows_df[bows_df["COURSE_ID"] == course_id][["token", "count"]]
    target_bow = target_bow.rename(columns={"count": "count_1"})

    results = []

    # Loop through all other courses
    for other_id in courses_df["COURSE_ID"].unique():
        if other_id == course_id:
            continue

        other_bow = bows_df[bows_df["COURSE_ID"] == other_id][["token", "count"]]
        other_bow = other_bow.rename(columns={"count": "count_2"})

        # Clean numeric pivot
        numeric_matrix = pivot_two_bows(target_bow, other_bow)

        # Compute cosine similarity
        sim = cosine_similarity(
            numeric_matrix["count_1"].values.reshape(1, -1),
            numeric_matrix["count_2"].values.reshape(1, -1)
        )[0][0]

        # Append result
        row = courses_df[courses_df["COURSE_ID"] == other_id].iloc[0]
        results.append({
            "course_id": other_id,
            "title": row["TITLE"],
            "description": row["DESCRIPTION"],
            "similarity": float(sim)
        })

    # Sort by similarity
    results = sorted(results, key=lambda x: x["similarity"], reverse=True)

    return results[:top_k]


# ---------------------------------------------------------
# Simple user-based popularity recommender
# ---------------------------------------------------------
def recommend_for_user(user_id, courses_df, ratings_df, top_k=5):
    """
    Recommend top-rated courses the user has NOT taken.
    """
    # Courses user already rated
    taken = ratings_df[ratings_df["user"] == user_id]["item"].unique()

    # Compute global popularity
    popularity = (
        ratings_df.groupby("item")["rating"]
        .mean()
        .sort_values(ascending=False)
    )

    recommendations = []

    for course_id, score in popularity.items():
        if course_id in taken:
            continue

        row = courses_df[courses_df["COURSE_ID"] == course_id].iloc[0]

        recommendations.append({
            "course_id": course_id,
            "title": row["TITLE"],
            "description": row["DESCRIPTION"],
            "score": float(score)
        })

        if len(recommendations) >= top_k:
            break

    return recommendations

