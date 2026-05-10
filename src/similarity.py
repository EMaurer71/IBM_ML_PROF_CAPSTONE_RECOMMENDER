"""
similarity.py
Vector similarity utilities for the recommender system.
"""

import numpy as np
from scipy.spatial.distance import cosine


def cosine_sim(vec1, vec2):
    """
    Computes cosine similarity between two numeric vectors.
    Returns 0.0 if either vector has zero magnitude.
    """
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    return 1 - cosine(vec1, vec2)

