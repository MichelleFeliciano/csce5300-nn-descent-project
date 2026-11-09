"""
make_plots.py
--------------
Turns results.json (from experiments.py) into the comparison figures for
the report: Recall@10, construction time, and extra memory consumption,
grouped by dataset, classic NN-Descent vs. Dynamic NN-Descent -- the same
three metrics and grouping style as the paper's own Fig. 3.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_PATH = "/tmp/project_code/results_out/results.json"
OUT_DIR = "/tmp/project_code/results_out"

# Fixed categorical colors (colorblind-safe blue/orange pair), assigned by
# algorithm identity and never reused for anything else in these figures.
COLOR_CLASSIC = "#4C72B0"
COLOR_DYNAMIC = "#DD8452"


def load_results():
    with open(RESULTS_PATH) as f:
        return json.load(f)


def plot_metric(results, datasets, metric_key, ylabel, title, out_name, value_fmt="{:.3f}"):
    fig, ax = plt.subplots(figsize=(7, 4.5))

    x = np.arange(len(datasets))
    width = 0.32

    classic_vals = [
        next(r[metric_key] for r in results if r["dataset"] == d and r["algorithm"] == "NN-Descent (baseline)")
        for d in datasets
    ]
    dynamic_vals = [
        next(r[metric_key] for r in results if r["dataset"] == d and r["algorithm"] == "Dynamic NN-Descent")
        for d in datasets
    ]

    bars1 = ax.bar(x - width / 2, classic_vals, width, label="NN-Descent (baseline)",
                    color=COLOR_CLASSIC, edgecolor="white")
    bars2 = ax.bar(x + width / 2, dynamic_vals, width, label="Dynamic NN-Descent",
                    color=COLOR_DYNAMIC, edgecolor="white")

    for bars in (bars1, bars2):
        for b in bars:
            height = b.get_height()
            ax.annotate(
                value_fmt.format(height),
                xy=(b.get_x() + b.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=9, color="#333333",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#E0E0E0", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, out_name)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    results = load_results()
    datasets = sorted({r["dataset"] for r in results})  # consistent order

    plot_metric(
        results, datasets, "recall_at_10",
        ylabel="Recall@10", title="Graph Quality: Recall@10",
        out_name="fig_recall.png", value_fmt="{:.3f}",
    )
    plot_metric(
        results, datasets, "time_seconds_untracked",
        ylabel="Construction time (s)", title="Construction Time",
        out_name="fig_time.png", value_fmt="{:.1f}s",
    )
    plot_metric(
        results, datasets, "peak_extra_memory_KB",
        ylabel="Extra memory (KB)", title="Extra Memory Consumption",
        out_name="fig_memory.png", value_fmt="{:.0f} KB",
    )


if __name__ == "__main__":
    main()
