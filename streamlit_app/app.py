import streamlit as st
import pandas as pd
import sys
import os

# -------------------------------------------------------------------
# Fix Python path so we can import recommender.py from parent folder
# -------------------------------------------------------------------
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

from recommender import get_similar_courses, recommend_for_user

# -------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------
DATA_DIR = os.path.join(parent_dir, "data")

courses_df = pd.read_csv(os.path.join(DATA_DIR, "courses.csv"))
bows_df = pd.read_csv(os.path.join(DATA_DIR, "courses_bows.csv"))
ratings_df = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"))

# -------------------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------------------
st.title("IBM ML Capstone – Course Recommender")

mode = st.sidebar.selectbox("Recommendation Type", ["Content-Based", "User-Based"])

if mode == "Content-Based":
    course_id = st.selectbox("Select a course", courses_df["COURSE_ID"].unique())
    if st.button("Recommend"):
        results = get_similar_courses(course_id, courses_df, bows_df)
        for r in results:
            st.subheader(f"{r['title']} ({r['course_id']})")
            st.write(r["description"])
            st.write(f"Similarity: {r['similarity']:.3f}")

else:
    user_id = st.selectbox("Select a user", ratings_df["user"].unique())
    if st.button("Recommend"):
        results = recommend_for_user(user_id, courses_df, ratings_df)
        for r in results:
            st.subheader(f"{r['title']} ({r['course_id']})")
            st.write(r["description"])
            st.write(f"Score: {r['score']:.3f}")


