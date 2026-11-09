"""
ground_truth.py
----------------
Computes exact (brute-force) top-k nearest neighbors for every point in a
dataset. This is the "answer key" used to score how good an approximate
k-NN graph (built by NN-Descent or Dynamic NN-Descent) actually is.

We can afford brute force here because our datasets are 10,000 points --
for the million-scale datasets in the paper this would be far too slow,
which is exactly *why* approximate methods like NN-Descent exist.

Recall@10 (as defined in the paper, Section IV-A) compares each node's
approximate neighbor list against this ground truth.
"""

import time
import numpy as np

from data_io import load_fbin


def compute_ground_truth(vectors, k=10, batch_size=1000):
    """
    For every point in `vectors`, find its true top-k nearest neighbors
    (by Euclidean/L2 distance) among all OTHER points in the same set.

    Parameters
    ----------
    vectors : np.ndarray, shape (n, dim)
    k : int
        Number of nearest neighbors to keep per point.
    batch_size : int
        How many points to process at once (keeps memory usage bounded --
        doing all n x n distances at once would use too much RAM).

    Returns
    -------
    neighbor_ids : np.ndarray, shape (n, k), dtype int32
        neighbor_ids[i] = indices of point i's true k nearest neighbors,
        sorted closest-first. Point i itself is excluded.
    """
    n = vectors.shape[0]
    # Precompute squared norms once: ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b
    sq_norms = np.sum(vectors ** 2, axis=1)

    neighbor_ids = np.empty((n, k), dtype=np.int32)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = vectors[start:end]

        # Squared distances from this batch to every point in the dataset.
        dists = (
            sq_norms[start:end, None]
            + sq_norms[None, :]
            - 2.0 * (batch @ vectors.T)
        )

        # Exclude each point from being its own neighbor.
        row_idx = np.arange(start, end)
        dists[np.arange(end - start), row_idx] = np.inf

        # argpartition is O(n) per row instead of a full O(n log n) sort --
        # we only need the smallest k, not a full ranking of all n.
        nearest = np.argpartition(dists, kth=k - 1, axis=1)[:, :k]

        # Sort just those k candidates so results are closest-first.
        row_ar = np.arange(end - start)[:, None]
        order = np.argsort(dists[row_ar, nearest], axis=1)
        neighbor_ids[start:end] = nearest[row_ar, order]

    return neighbor_ids


if __name__ == "__main__":
    import sys
    import os

    for path in sys.argv[1:]:
        vecs = load_fbin(path)
        t0 = time.time()
        gt = compute_ground_truth(vecs, k=10)
        elapsed = time.time() - t0

        out_path = os.path.join(
            os.path.dirname(path),
            os.path.splitext(os.path.basename(path))[0] + "_groundtruth_k10.npy",
        )
        np.save(out_path, gt)

        print(f"{path}: {vecs.shape[0]} points, {vecs.shape[1]}-dim "
              f"-> ground truth in {elapsed:.2f}s, saved to {out_path}")
        print(f"  sample: point 0's true top-10 neighbors = {gt[0].tolist()}")
