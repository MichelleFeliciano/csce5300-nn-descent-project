"""
evaluate.py
-----------
Metrics for scoring an approximate k-NN graph against exact ground truth.
Recall@k formula matches the paper's Section IV-A:
    Recall@k = ( sum over all points of true-positive neighbors found ) / (n * k)
"""

import numpy as np


def recall_at_k(pred_ids, true_ids):
    """
    pred_ids, true_ids : (n, k) arrays of neighbor ids.
    Returns a float in [0, 1].
    """
    n, k = true_ids.shape
    total_hits = 0
    for i in range(n):
        total_hits += len(set(pred_ids[i].tolist()) & set(true_ids[i].tolist()))
    return total_hits / (n * k)
