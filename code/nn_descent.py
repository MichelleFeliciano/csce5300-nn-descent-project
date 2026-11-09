"""
nn_descent.py
-------------
Classic NN-Descent (Dong, Moses & Li, WWW 2011), as reviewed in the
Dynamic NN-Descent paper's Section II-B. This is the BASELINE this
project compares Dynamic NN-Descent against.

Two-step iterative procedure, repeated until convergence:
  1. Sampling: for each node, sample a small number of "new" and "old"
     neighbors from its current k-NN list and from its reverse neighbors
     (who has this node as a neighbor).
  2. Local-join: compute distances between sampled old-new and new-new
     pairs; any pair closer than a node's current worst neighbor replaces
     it in that node's list ("neighbor's neighbor is likely a neighbor").

This implementation keeps the two "extra" supporting structures the
paper's Fig. 1(b) describes -- the sampled-neighbor lists and their
reverse lists -- which is exactly what Dynamic NN-Descent removes to
save memory (that's the whole point of the paper).
"""

import time
import tracemalloc
import numpy as np


def _squared_l2(vectors, i, j):
    diff = vectors[i] - vectors[j]
    return float(np.dot(diff, diff))


class KNNGraph:
    """
    Fixed-size k-NN candidate list per node, always kept sorted by
    distance (closest first).
        ids[i]    -> neighbor ids for node i        (k,)
        dists[i]  -> matching squared distances      (k,)
        is_new[i] -> whether each neighbor is still "new"
                     (hasn't been through a local-join yet)
    """

    def __init__(self, n, k):
        self.n = n
        self.k = k
        self.ids = np.full((n, k), -1, dtype=np.int32)
        self.dists = np.full((n, k), np.inf, dtype=np.float32)
        self.is_new = np.zeros((n, k), dtype=bool)

    def try_insert(self, i, cand_id, cand_dist):
        """Insert cand_id into node i's list if it's closer than the
        current worst neighbor and not already present. Returns True if
        an update happened (used to track convergence)."""
        if cand_id == i:
            return False
        row_ids = self.ids[i]
        row_dists = self.dists[i]
        if cand_dist >= row_dists[-1]:
            return False
        if cand_id in row_ids:
            return False
        pos = self.k - 1
        while pos > 0 and row_dists[pos - 1] > cand_dist:
            row_dists[pos] = row_dists[pos - 1]
            row_ids[pos] = row_ids[pos - 1]
            self.is_new[i, pos] = self.is_new[i, pos - 1]
            pos -= 1
        row_dists[pos] = cand_dist
        row_ids[pos] = cand_id
        self.is_new[i, pos] = True
        return True


def init_random_graph(vectors, k, seed=0):
    n = vectors.shape[0]
    rng = np.random.default_rng(seed)
    graph = KNNGraph(n, k)
    for i in range(n):
        choices = rng.choice(n - 1, size=k, replace=False)
        choices = np.where(choices >= i, choices + 1, choices)  # skip self
        graph.ids[i] = choices
        graph.is_new[i] = True
        for pos in range(k):
            graph.dists[i, pos] = _squared_l2(vectors, i, int(choices[pos]))
        order = np.argsort(graph.dists[i])
        graph.ids[i] = graph.ids[i][order]
        graph.dists[i] = graph.dists[i][order]
    return graph


def _sample(arr, rate, rng):
    if len(arr) == 0:
        return arr
    m = max(1, int(round(len(arr) * rate)))
    if m >= len(arr):
        return arr
    return arr[rng.choice(len(arr), size=m, replace=False)]


def nn_descent(vectors, k=10, sample_rate=0.5, max_iters=10,
               convergence_frac=0.001, seed=0, verbose=True,
               track_memory=False):
    """
    Build an approximate k-NN graph with classic NN-Descent.

    sample_rate : fraction of new/old neighbors sampled per round
                  (paper calls this "rho"; lower = faster but lower quality)
    max_iters   : cap on iterations
    convergence_frac : stop early once fewer than this fraction of n*k
                        possible updates happen in one iteration
    """
    n = vectors.shape[0]
    rng = np.random.default_rng(seed)
    graph = init_random_graph(vectors, k, seed=seed)

    # Start memory tracking AFTER the k-NN graph G is built, so we
    # measure only the extra sampling/reverse-neighbor bookkeeping
    # structures created each iteration -- not G or the raw vectors.
    if track_memory:
        tracemalloc.start()
        tracemalloc.reset_peak()

    updates_per_iter = []
    t0 = time.time()
    stop_threshold = convergence_frac * n * k

    for it in range(max_iters):
        # ---- Step 1: Sampling ----
        new_lists = [None] * n
        old_lists = [None] * n
        rev_new = [[] for _ in range(n)]
        rev_old = [[] for _ in range(n)]

        for v in range(n):
            row_ids = graph.ids[v]
            new_mask = graph.is_new[v]
            new_ids = row_ids[new_mask]
            old_ids = row_ids[~new_mask]

            sampled_new = _sample(new_ids, sample_rate, rng)
            sampled_old = _sample(old_ids, sample_rate, rng)

            if len(sampled_new) > 0:
                sampled_set = set(sampled_new.tolist())
                for pos in range(k):
                    if row_ids[pos] in sampled_set:
                        graph.is_new[v, pos] = False

            new_lists[v] = sampled_new
            old_lists[v] = sampled_old
            for u in sampled_new:
                rev_new[int(u)].append(v)
            for u in sampled_old:
                rev_old[int(u)].append(v)

        # ---- Step 2: Local-join ----
        updates = 0
        for v in range(n):
            rv_new = _sample(np.array(rev_new[v], dtype=np.int32), sample_rate, rng)
            rv_old = _sample(np.array(rev_old[v], dtype=np.int32), sample_rate, rng)
            all_new = np.unique(np.concatenate([new_lists[v], rv_new]))
            all_old = np.unique(np.concatenate([old_lists[v], rv_old]))

            for a in range(len(all_new)):
                u1 = int(all_new[a])
                for b in range(a + 1, len(all_new)):
                    u2 = int(all_new[b])
                    d = _squared_l2(vectors, u1, u2)
                    if graph.try_insert(u1, u2, d):
                        updates += 1
                    if graph.try_insert(u2, u1, d):
                        updates += 1

            for u1 in all_new:
                u1 = int(u1)
                for u2 in all_old:
                    u2 = int(u2)
                    if u1 == u2:
                        continue
                    d = _squared_l2(vectors, u1, u2)
                    if graph.try_insert(u1, u2, d):
                        updates += 1
                    if graph.try_insert(u2, u1, d):
                        updates += 1

        updates_per_iter.append(updates)
        if verbose:
            print(f"  iter {it + 1}/{max_iters}: {updates} updates")
        if updates <= stop_threshold:
            break

    peak_extra_memory_bytes = None
    if track_memory:
        _, peak_extra_memory_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    stats = {
        "iterations": len(updates_per_iter),
        "time_seconds": time.time() - t0,
        "updates_per_iter": updates_per_iter,
        "peak_extra_memory_bytes": peak_extra_memory_bytes,
    }
    return graph, stats


if __name__ == "__main__":
    import argparse
    from data_io import load_fbin
    from ground_truth import compute_ground_truth
    from evaluate import recall_at_k

    parser = argparse.ArgumentParser(description="Smoke-test classic NN-Descent")
    parser.add_argument("fbin_path")
    parser.add_argument("--n", type=int, default=1000, help="subsample size")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--sample-rate", type=float, default=0.5)
    parser.add_argument("--max-iters", type=int, default=8)
    args = parser.parse_args()

    vectors = load_fbin(args.fbin_path)[: args.n]
    print(f"Loaded {vectors.shape[0]} points, {vectors.shape[1]}-dim")

    print("Computing exact ground truth for this subsample...")
    gt = compute_ground_truth(vectors, k=args.k)

    print("Running NN-Descent...")
    graph, stats = nn_descent(
        vectors, k=args.k, sample_rate=args.sample_rate, max_iters=args.max_iters
    )

    recall = recall_at_k(graph.ids, gt)
    print(f"\nDone in {stats['time_seconds']:.2f}s over {stats['iterations']} iterations")
    print(f"Updates per iteration: {stats['updates_per_iter']}")
    print(f"Recall@{args.k}: {recall:.4f}")
