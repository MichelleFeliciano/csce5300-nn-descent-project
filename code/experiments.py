"""
experiments.py
---------------
Step 5: the actual project experiments. Runs classic NN-Descent and
Dynamic NN-Descent on both datasets (SIFT -- the paper's own dataset
family -- and the synthetic "different" dataset), at full scale (10,000
points each), and records the three metrics the paper itself reports in
Section IV:
    - Recall@10   (graph quality)
    - Construction time
    - Extra memory consumption (excludes the k-NN graph itself and the
      raw vectors, matching the paper's own footnote 6 convention)

Run one (dataset, algorithm) combination at a time with --dataset and
--algo so each run fits comfortably in its own time budget. Results are
appended to results.json after every run, so partial progress is never
lost.
"""

import argparse
import json
import os
import numpy as np

from data_io import load_fbin
from evaluate import recall_at_k
from nn_descent import nn_descent
from dynamic_nn_descent import dynamic_nn_descent

DATA_DIR = "/mnt/user-data/uploads/5300/Project/data"
OUT_DIR = "/tmp/project_code/results_out"
os.makedirs(OUT_DIR, exist_ok=True)
RESULTS_PATH = os.path.join(OUT_DIR, "results.json")

DATASETS = {
    "sift": {
        "label": "SIFT (same as paper)",
        "vectors": f"{DATA_DIR}/sift/queries.fbin",
        "ground_truth": f"{DATA_DIR}/sift/queries_groundtruth_k10.npy",
    },
    "synthetic": {
        "label": "Synthetic (different)",
        "vectors": f"{DATA_DIR}/synthetic/synthetic.fbin",
        "ground_truth": f"{DATA_DIR}/synthetic/synthetic_groundtruth_k10.npy",
    },
}

# Parameters tuned in a smaller pilot run: K=20 balances graph quality
# against CPU runtime for a 10,000-point, single-machine, pure-Python
# simulation (the paper uses larger K on GPU/multi-thread C++). smpN=10
# for Dynamic NN-Descent matches classic's effective sample size
# (sample_rate=0.5 of K=20 candidates).
K = 20
SEED = 0


def load_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            return json.load(f)
    return []


def save_result(new_result):
    results = load_results()
    # Replace any prior entry for the same (dataset, algorithm) so reruns
    # overwrite cleanly instead of duplicating.
    results = [
        r for r in results
        if not (r["dataset"] == new_result["dataset"] and r["algorithm"] == new_result["algorithm"])
    ]
    results.append(new_result)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    return results


def run_one(dataset_key, algo_key):
    ds = DATASETS[dataset_key]
    vectors = load_fbin(ds["vectors"])
    gt10 = np.load(ds["ground_truth"])
    print(f"Dataset: {ds['label']} ({vectors.shape[0]} points, {vectors.shape[1]}-dim)")

    if algo_key == "classic":
        name = "NN-Descent (baseline)"
        graph, stats = nn_descent(
            vectors, k=K, sample_rate=0.5, max_iters=15,
            seed=SEED, verbose=True, track_memory=True,
        )
        rounds = stats["iterations"]
    else:
        name = "Dynamic NN-Descent"
        graph, stats = dynamic_nn_descent(
            vectors, k=K, smpN=10, iterN=15,
            seed=SEED, verbose=True, track_memory=True,
        )
        rounds = stats["passes"]

    recall10 = recall_at_k(graph.ids[:, :10], gt10)
    result = {
        "dataset": ds["label"],
        "algorithm": name,
        "time_seconds": round(stats["time_seconds"], 3),
        "rounds": rounds,
        "peak_extra_memory_KB": round(stats["peak_extra_memory_bytes"] / 1024, 2),
        "recall_at_10": round(recall10, 4),
    }
    all_results = save_result(result)
    print("\nResult:", json.dumps(result, indent=2))
    print(f"({len(all_results)}/4 combinations completed so far)")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["sift", "synthetic"], required=True)
    parser.add_argument("--algo", choices=["classic", "dynamic"], required=True)
    args = parser.parse_args()
    run_one(args.dataset, args.algo)
