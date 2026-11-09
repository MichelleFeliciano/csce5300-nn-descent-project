"""
dynamic_nn_descent.py
----------------------
Dynamic NN-Descent -- the method this whole project is about (Wang, Zhao,
Xiao, Yao & Zhang, IEEE Trans. Big Data, 2025). This is a direct
translation of the paper's Algorithm 1 (Section III-A) into Python.

The key difference from classic NN-Descent (nn_descent.py):
  * Classic NN-Descent processes ALL n nodes in a batch each iteration,
    and rebuilds a full extra graph H of sampled neighbors every round.
  * Dynamic NN-Descent processes ONE node at a time in a single
    continuous loop (no "batch boundary" between rounds), sampling
    directly from the live k-NN graph G instead of a frozen snapshot H.
    The reverse-neighbor graph R is also incomplete -- capped at smpN
    entries per node instead of holding every reverse neighbor -- which
    is where most of the paper's claimed memory savings come from.

We reuse the exact same KNNGraph data structure, random initialization,
and squared-L2 distance function as nn_descent.py so that both
algorithms start from an identical random graph -- that's what makes the
Step 5 comparison between them fair.
"""

import time
import tracemalloc
import numpy as np

from nn_descent import KNNGraph, init_random_graph, _squared_l2


def _sample_upto(arr, m, rng):
    """Sample up to m items from arr without replacement (paper's
    Algorithm 1, lines 6-7: fixed COUNTS, not a fraction like classic
    NN-Descent's sample_rate)."""
    if len(arr) <= m:
        return arr
    return arr[rng.choice(len(arr), size=m, replace=False)]


def dynamic_nn_descent(vectors, k=10, smpN=5, iterN=10, seed=0, verbose=True,
                        track_memory=False):
    """
    Build an approximate k-NN graph with Dynamic NN-Descent
    (paper's Algorithm 1, lines 1-26).

    smpN  : sampling size (paper's "smpN" -- O samples up to 3*smpN old
            neighbors, N samples up to smpN new neighbors, and R[i] is
            capped at smpN entries per old/new list)
    iterN : number of full passes over the dataset (paper's "iterN";
            the algorithm's outer loop actually runs n * iterN single-node
            steps -- see Algorithm 1, line 3: "iter < |C| * iterN")
    """
    n = vectors.shape[0]
    rng = np.random.default_rng(seed)
    graph = init_random_graph(vectors, k, seed=seed)

    # Start memory tracking AFTER G is built, same convention as
    # nn_descent.py, so the two algorithms are compared fairly on
    # just their extra bookkeeping structures (R here vs. H+R there).
    if track_memory:
        tracemalloc.start()
        tracemalloc.reset_peak()

    # R[i].old / R[i].new: incomplete reverse-neighbor lists, capped at
    # smpN entries each. This -- not a full n-sized reverse graph -- is
    # the memory-saving structure the paper describes in Fig. 1(c).
    R_old = [[] for _ in range(n)]
    R_new = [[] for _ in range(n)]

    updates_this_pass = 0
    updates_per_pass = []
    t0 = time.time()

    total_steps = n * iterN
    for step in range(total_steps):
        i = step % n

        row_ids = graph.ids[i]
        new_mask = graph.is_new[i]
        old_ids = row_ids[~new_mask]
        new_ids = row_ids[new_mask]

        # ---- Sampling (Algorithm 1, lines 6-13) ----
        O = _sample_upto(old_ids, 3 * smpN, rng)
        N = _sample_upto(new_ids, smpN, rng)

        for u in O:
            u = int(u)
            if len(R_old[u]) < smpN and i not in R_old[u]:
                R_old[u].append(i)
        for u in N:
            u = int(u)
            if len(R_new[u]) < smpN and i not in R_new[u]:
                R_new[u].append(i)

        # Sampled "new" items become "old" now that they've been used
        # (line 14).
        if len(N) > 0:
            sampled_set = set(N.tolist())
            for pos in range(k):
                if row_ids[pos] in sampled_set:
                    graph.is_new[i, pos] = False

        # Pull in i's own accumulated reverse neighbors, then drain them
        # (lines 15-18) -- this is what makes the reverse-neighbor
        # cross-matching happen over consecutive rounds instead of one,
        # per the paper's Discussion in Section III-A.
        if R_old[i]:
            O = np.unique(np.concatenate([O, np.array(R_old[i], dtype=np.int32)]))
        if R_new[i]:
            N = np.unique(np.concatenate([N, np.array(R_new[i], dtype=np.int32)]))
        R_old[i] = []
        R_new[i] = []

        # ---- Local-join (Algorithm 1, lines 19-24) ----
        for a in range(len(N)):
            u1 = int(N[a])
            for b in range(a + 1, len(N)):
                u2 = int(N[b])
                d = _squared_l2(vectors, u1, u2)
                if graph.try_insert(u1, u2, d):
                    updates_this_pass += 1
                if graph.try_insert(u2, u1, d):
                    updates_this_pass += 1

        for u1 in N:
            u1 = int(u1)
            for u2 in O:
                u2 = int(u2)
                if u1 == u2:
                    continue
                d = _squared_l2(vectors, u1, u2)
                if graph.try_insert(u1, u2, d):
                    updates_this_pass += 1
                if graph.try_insert(u2, u1, d):
                    updates_this_pass += 1

        # One "pass" = n single-node steps, for reporting/comparison
        # against classic NN-Descent's per-iteration updates.
        if i == n - 1:
            updates_per_pass.append(updates_this_pass)
            if verbose:
                pass_num = len(updates_per_pass)
                print(f"  pass {pass_num}/{iterN}: {updates_this_pass} updates")
            updates_this_pass = 0

    peak_extra_memory_bytes = None
    if track_memory:
        _, peak_extra_memory_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    stats = {
        "passes": len(updates_per_pass),
        "time_seconds": time.time() - t0,
        "updates_per_pass": updates_per_pass,
        "peak_extra_memory_bytes": peak_extra_memory_bytes,
    }
    return graph, stats


if __name__ == "__main__":
    import argparse
    from data_io import load_fbin
    from ground_truth import compute_ground_truth
    from evaluate import recall_at_k

    parser = argparse.ArgumentParser(description="Smoke-test Dynamic NN-Descent")
    parser.add_argument("fbin_path")
    parser.add_argument("--n", type=int, default=1000, help="subsample size")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--smpN", type=int, default=5)
    parser.add_argument("--iterN", type=int, default=8)
    args = parser.parse_args()

    vectors = load_fbin(args.fbin_path)[: args.n]
    print(f"Loaded {vectors.shape[0]} points, {vectors.shape[1]}-dim")

    print("Computing exact ground truth for this subsample...")
    gt = compute_ground_truth(vectors, k=args.k)

    print("Running Dynamic NN-Descent...")
    graph, stats = dynamic_nn_descent(
        vectors, k=args.k, smpN=args.smpN, iterN=args.iterN
    )

    recall = recall_at_k(graph.ids, gt)
    print(f"\nDone in {stats['time_seconds']:.2f}s over {stats['passes']} passes")
    print(f"Updates per pass: {stats['updates_per_pass']}")
    print(f"Recall@{args.k}: {recall:.4f}")
