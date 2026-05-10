"""
backend_app.py
Clean, modular backend for the Course Recommender System.
"""

import os
import pandas as pd
from flask import Flask, jsonify, request

# Import your recommender functions (kept separate)
from recommender import (
    get_similar_courses,
    recommend_for_user
)

# -----------------------------------------------------------------------------
# App Setup
# -----------------------------------------------------------------------------
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# -----------------------------------------------------------------------------
# Load Data
# -----------------------------------------------------------------------------
def load_csv(filename):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required data file: {path}")
    return pd.read_csv(path)

courses_df = load_csv("courses.csv")
ratings_df = load_csv("ratings.csv")
bows_df    = load_csv("courses_bows.csv")

# -----------------------------------------------------------------------------
# Health Check
# -----------------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

# -----------------------------------------------------------------------------
# Content-Based Similarity Endpoint
# -----------------------------------------------------------------------------
@app.route("/similar/<course_id>", methods=["GET"])
def similar_courses(course_id):
    """
    Returns courses similar to the given course_id using BoW similarity.
    Optional query param: top_n (default=10)
    """
    top_n = int(request.args.get("top_n", 10))

    results = get_similar_courses(
        course_id=course_id,
        courses_df=courses_df,
        bows_df=bows_df,
        top_n=top_n
    )

    return jsonify(results), 200

# -----------------------------------------------------------------------------
# User Recommendations Endpoint
# -----------------------------------------------------------------------------
@app.route("/recommend/<user_id>", methods=["GET"])
def recommend(user_id):
    """
    Returns recommended courses for a given user_id.
    Optional query param: top_n (default=10)
    """
    top_n = int(request.args.get("top_n", 10))

    results = recommend_for_user(
        user_id=user_id,
        courses_df=courses_df,
        ratings_df=ratings_df,
        top_n=top_n
    )

    return jsonify(results), 200

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)

