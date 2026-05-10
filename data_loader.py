"""
data_loader.py
Utility functions for loading project datasets.
"""

import os
import pandas as pd


def load_csv(data_dir, filename):
    """
    Loads a CSV from the project's data directory.
    """
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required data file: {path}")
    return pd.read_csv(path)


def load_all_data(base_dir):
    """
    Loads all required datasets for the recommender system.
    """
    data_dir = os.path.join(base_dir, "data")

    courses_df = load_csv(data_dir, "courses.csv")
    ratings_df = load_csv(data_dir, "ratings.csv")
    bows_df    = load_csv(data_dir, "courses_bows.csv")

    return courses_df, ratings_df, bows_df

