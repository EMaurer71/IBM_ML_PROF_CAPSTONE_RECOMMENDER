"""
bow_utils.py
Utilities for handling Bag-of-Words (BoW) vectors.
"""

import pandas as pd


def pivot_two_bows(base_df, compare_df):
    """
    Aligns two single-row BoW DataFrames into a 2-row numeric matrix.
    Drops non-numeric columns and returns aligned vectors.
    """
    base = base_df.copy()
    base["type"] = "base"

    compare = compare_df.copy()
    compare["type"] = "compare"

    combined = pd.concat([base, compare], axis=0)

    non_numeric = ["doc_id", "type"]
    numeric_cols = [c for c in combined.columns if c not in non_numeric]

    return combined[numeric_cols].astype(float)

