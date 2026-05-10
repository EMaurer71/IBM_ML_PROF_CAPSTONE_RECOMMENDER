import streamlit as st
import pandas as pd
import sys
import os

# ---------------------------------------------------------
# FIX: Add project root to Python path so imports work
# ---------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from recommender import get_similar_courses, recommend_for_user

# ---------------------------------------------------------
# Load data using absolute paths
# ---------------------------------------------------------
DATA_DIR = os.path.join(ROOT_DIR, "data")

st.write("Loading data from:", DATA_DIR)  # Debug line

courses_df = pd.read_csv(os.path.join(DATA_DIR, "courses.csv"))
bows_df = pd.read_csv(os.path.join(DATA_DIR, "courses_bows.csv"))
ratings_df = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"))

# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------
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

