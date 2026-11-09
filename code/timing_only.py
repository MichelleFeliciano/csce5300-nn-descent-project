"""
timing_only.py
----------------
Re-times all four (dataset, algorithm) combinations WITHOUT memory
tracking. tracemalloc (used in experiments.py to measure extra memory)
adds significant profiling overhead, which inflates wall-clock time --
so time and memory are measured in separate, dedicated runs to keep both
honest. This script only updates the "time_seconds" field in
results.json; recall and memory numbers from experiments.py are left
untouched (same algorithm, same fixed seed => same graph either way).
"""

import json
import os

from data_io import load_fbin
from nn_descent import nn_descent
from dynamic_nn_descent import dynamic_nn_descent

DATA_DIR = "/mnt/user-data/uploads/5300/Project/data"
RESULTS_PATH = "/tmp/project_code/results_out/results.json"

DATASETS = {
    "sift": {
        "label": "SIFT (same as paper)",
        "vectors": f"{DATA_DIR}/sift/queries.fbin",
    },
    "synthetic": {
        "label": "Synthetic (different)",
        "vectors": f"{DATA_DIR}/synthetic/synthetic.fbin",
    },
}
K = 20
SEED = 0


def time_one(dataset_key, algo_key):
    ds = DATASETS[dataset_key]
    vectors = load_fbin(ds["vectors"])
    if algo_key == "classic":
        name = "NN-Descent (baseline)"
        _, stats = nn_descent(vectors, k=K, sample_rate=0.5, max_iters=15,
                               seed=SEED, verbose=False, track_memory=False)
    else:
        name = "Dynamic NN-Descent"
        _, stats = dynamic_nn_descent(vectors, k=K, smpN=10, iterN=15,
                                       seed=SEED, verbose=False, track_memory=False)
    return ds["label"], name, round(stats["time_seconds"], 3)


def main():
    with open(RESULTS_PATH) as f:
        results = json.load(f)

    for dataset_key in DATASETS:
        for algo_key in ("classic", "dynamic"):
            label, name, t = time_one(dataset_key, algo_key)
            for r in results:
                if r["dataset"] == label and r["algorithm"] == name:
                    r["time_seconds_untracked"] = t
                    print(f"{label} / {name}: {t}s (untracked, real speed)")

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print("updated", RESULTS_PATH)


if __name__ == "__main__":
    main()
