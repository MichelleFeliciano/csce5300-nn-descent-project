# Dynamic NN-Descent vs. NN-Descent: k-NN Graph Construction

CSCE 5300 (Introduction to Big Data and Data Science) course project, University of North Texas. I reproduced the algorithm from *Dynamic NN-Descent: An Efficient k-NN Graph Construction Method* (Wang, Zhao, Xiao, Yao and Zhang, IEEE Transactions on Big Data, 2025) in Python and compared it against classic NN-Descent.

## What it does

- `code/nn_descent.py` is the classic NN-Descent baseline.
- `code/dynamic_nn_descent.py` is Dynamic NN-Descent, translated from the paper's Algorithm 1. It samples neighbors directly from the live k-NN graph and keeps a capped reverse-neighbor graph instead of a separate sampled graph.
- `code/ground_truth.py` computes exact brute-force neighbors, and `code/evaluate.py` scores a graph with Recall@10, as defined in the paper.
- `code/experiments.py` runs both algorithms on two 10,000-point datasets (SIFT descriptors and a synthetic set) with K = 20 and records recall, construction time and extra memory. `code/timing_only.py` re-times runs without memory tracking, and `code/make_plots.py` draws the figures.
- `code/data_io.py` loads the `.fbin` vector format used by the datasets.

## Results (from `results/results.json`)

| Dataset | Algorithm | Recall@10 | Extra memory (KB) | Time, no memory tracking (s) |
|---|---|---|---|---|
| SIFT | NN-Descent | 0.9528 | 6,516 | 52.0 |
| SIFT | Dynamic NN-Descent | 0.9874 | 3,521 | 55.2 |
| Synthetic | NN-Descent | 0.9667 | 6,520 | 44.4 |
| Synthetic | Dynamic NN-Descent | 0.9780 | 3,444 | 48.3 |

In my runs Dynamic NN-Descent reached higher recall and used roughly half the extra memory, while construction time was slightly longer than the baseline (this is single-threaded Python). Plots are in `results/` (`fig_recall.png`, `fig_time.png`, `fig_memory.png`).

## Documents

`docs/` has the project plan, the written report and the presentation slides.

## Running it

The dataset files are not included because of their size. `experiments.py` expects them under a `data/` folder as `sift/queries.fbin` and `synthetic/synthetic.fbin`, plus matching `*_groundtruth_k10.npy` files made by `ground_truth.py`. Once the data is in place, run for example:

```bash
python code/experiments.py --dataset sift --algo dynamic
```

Choices are `--dataset sift|synthetic` and `--algo classic|dynamic`.
