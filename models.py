"""
model_utils.py
Utility functions for model preparation, training, or hybrid recommenders.
"""

def normalize_scores(scores):
    """
    Normalizes a list of numeric scores to [0, 1].
    """
    if not scores:
        return scores

    min_s = min(scores)
    max_s = max(scores)

    if max_s == min_s:
        return [0.0 for _ in scores]

    return [(s - min_s) / (max_s - min_s) for s in scores]

