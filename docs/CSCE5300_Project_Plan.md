# CSCE 5300 Project Plan — Dynamic NN-Descent

**Paper:** J.-F. Wang, W.-L. Zhao, S. Xiao, J. Yao, and X. Zhang, "Dynamic NN-Descent: An Efficient k-NN Graph Construction Method," *IEEE Transactions on Big Data*, vol. 11, no. 2, pp. 879–886, Mar./Apr. 2025.

**Eligibility check — this paper qualifies:**

| Requirement | Status |
|---|---|
| Original, non-open-access, high-level journal/conference paper | ✅ IEEE Transactions on Big Data (IEEE Xplore, subscription-gated) |
| Not older than 5 years | ✅ Published 2025 |
| Big Data related field | ✅ k-NN graph construction — core to vector databases, retrieval, recommendation |

---

## Goal

Reproduce and evaluate the **Dynamic NN-Descent** algorithm from the paper in Python, benchmark it against the classic **NN-Descent** baseline on a real dataset of at least 1,000 items, and package the problem statement, method, and your own results into a report and/or presentation that satisfies every content requirement in the assignment — submitted on time and under the 30% plagiarism threshold.

---

## Steps

### 1. Lock in the paper (do this first, don't wait for Sept 30)
- [ ] Submit "Dynamic NN-Descent" via the Canvas project-topic submission for TA/IA approval.
- [ ] Note: topics are first-come, first-served — submitting early protects your topic even though this pick is unlikely to collide with another group.

### 2. Set up your dataset
- [ ] You need **≥1,000 items**, real data, in the "Big Data" spirit — but you do not need millions of points like the paper. A workable, honest-effort scale is 1,000–50,000 vectors.
- [ ] Two solid options:
  - **Reuse the paper's own benchmark family** (SIFT is the easiest — small, well-documented, available at [big-ann-benchmarks.com](http://big-ann-benchmarks.com)). Take a 1,000–20,000-point subsample of SIFT1M so it runs fast on your laptop.
  - **Use a different dataset entirely** (e.g., embeddings from a small image or text set you generate yourself, or `sklearn.datasets`) — this also satisfies the assignment's "same and different input data" comparison requirement if you run both.
- [ ] Plan to test on **at least two datasets/sizes** (e.g., 1,000 items and 10,000 items) so you have a "same vs. different input" comparison, as the assignment asks.

### 3. Implement the baseline: classic NN-Descent
- [ ] You need this to have something to compare against — the whole paper is a comparison against it.
- [ ] Implement the two-step iteration described in the paper's Section II-B (Sampling + Local-join), maintaining the three structures **G**, **H**, **R**.
- [ ] Reference implementation exists on GitHub (`aaalgo/kgraph`, cited as [7]/footnote 2 in the paper) — read it for structure, but **write your own Python version**; don't copy code.

### 4. Implement Dynamic NN-Descent (the paper's contribution)
- [ ] Follow **Algorithm 1** in the paper directly — it's fully pseudocoded (lines 1–26). This is your main implementation task.
- [ ] Key difference to get right: no separate graph **H**; neighbors are sampled live from **G**; the reverse-neighbor graph **R** is incomplete and drained after each node's local-join.
- [ ] Start single-threaded to get correctness, then parallelize with Python's `multiprocessing` or `concurrent.futures` — this is what demonstrates the paper's "no synchronization needed between threads" claim.

### 5. Run experiments and collect metrics
Match what the paper measures so your comparison is apples-to-apples:
- [ ] **Recall@10** — graph quality (formula is in the paper, Section IV-A)
- [ ] **Construction time**
- [ ] **Extra memory consumption** (peak RSS during construction, via `tracemalloc` or `psutil`)
- [ ] Vary at least one parameter (e.g., K, the neighborhood size) like the paper's Fig. 3–4, and plot Dynamic NN-Descent vs. NN-Descent on the same axes.

### 6. Write the report and/or build the presentation
The assignment requires **every one of these sections** to appear (report and/or slides):

- [ ] Problem statement studied in the paper (in your own words — **do not paste the assignment's problem statement itself**, that's explicitly banned)
- [ ] Qualitative + quantitative comparison of *existing* methods (use the paper's Related Work: divide-and-conquer methods vs. NN-Descent-family methods)
- [ ] The method/algorithm presented in the paper (explain Algorithm 1, Fig. 1's data-structure comparison)
- [ ] The paper's own results (quantitative + qualitative — Fig. 3–7, Table I–II)
- [ ] Advantages and limitations of the method
- [ ] Possible applications (vector databases, multimedia retrieval, recommendation systems, deep metric learning)
- [ ] Code — **typed as text, same font size as the body text, on a bright/white background** (not a screenshot, not dark-mode)
- [ ] Your own simulation results, compared explicitly against the paper's numbers
- [ ] IEEE-style numbered citations `[1]`, `[2]`, … with a matching References list

### 7. Compliance pass before submitting
- [ ] No screenshots anywhere in the report
- [ ] No copy of the project problem statement in the report
- [ ] Run it through UNT Canvas Turnitin — must be **≤30% plagiarism**
- [ ] Save/screenshot the Turnitin confirmation (percentage + upload timestamp) for your records
- [ ] File names use your **project group number**, in both PDF and (for the presentation) PPT format

---

## Timeline

Working backward from the assignment's fixed dates (today is **Mon, Aug 31**):

| Dates | Milestone |
|---|---|
| **Aug 31 – Sept 2** | Submit paper for TA/IA approval on Canvas. Set up Python environment, pick datasets. |
| **Sept 3 – 14** | Implement classic NN-Descent baseline in Python; test correctness on a tiny (~100-item) toy set. |
| **Sept 15 – 21** | Implement Dynamic NN-Descent (Algorithm 1); verify it produces comparable graph quality to the baseline. |
| **Sept 22 – 28** | Run full experiments on your ≥1,000-item dataset(s); generate comparison plots/tables (Recall@10, time, memory). |
| **Sept 29 – 30** | **Deadline: paper approval confirmed by Sept 30, 23:59.** (Should already be done — this is your buffer.) |
| **Oct 1 – 12** | Draft the report: problem statement, related-work comparison, method description, your results. |
| **Oct 13 – 19** | Build the presentation slides from the report content. |
| **Oct 20 – 26** | Run Turnitin, fix any flagged sections, proofread, format code sections correctly (bright background, body font size). |
| **Oct 27 – 30** | Final polish and internal deadline buffer. |
| **Oct 31, 23:59** | **Deadline: send report/presentation PDF to TA/IA for comments.** |
| **Nov 1 – 7** | Incorporate TA/IA feedback. |
| **Nov 8, 23:59** | **Deadline: final upload of report + PPT (PDF and PPT formats, named by group number) to Canvas.** Store your Turnitin confirmation. |
| **TBD — check syllabus** | Face-to-face presentation date (assigned randomly by TAs at the start of presentation classes — you won't know your exact day until then, so have your demo ready well before Oct 31). |

---

## Cost

| Item | Cost |
|---|---|
| Python + libraries (NumPy, Matplotlib, `multiprocessing`, `psutil`) | $0 (free/open source) |
| Datasets (SIFT/DEEP subsamples or self-generated) | $0 (public benchmark data) |
| Compute (your own laptop) | $0 |
| Optional: Google Colab (if your laptop is too slow for larger runs) | $0 on free tier |
| UNT Canvas Turnitin check | $0 (included) |

**No paid tools are required for this project.**

---

## Risks / Common Mistakes

- **Waiting until Sept 30 to submit for approval.** Submit this week — it's a two-minute task and removes a deadline from your plate early.
- **Only implementing Dynamic NN-Descent, not the baseline.** The assignment requires a *comparison*; without your own NN-Descent implementation you have nothing to compare against.
- **Testing on fewer than 1,000 items.** This is a hard requirement — check your dataset size before running final experiments.
- **Screenshotting code with a dark editor theme.** The assignment explicitly requires code in the body font, on a bright background — set your editor to light mode before capturing/typing code, or just retype it as formatted text.
- **Including screenshots or the problem statement text in the report.** Both are explicitly prohibited and will cost you points.
- **Leaving the Turnitin check for the last night.** Run it well before Oct 31 so you have time to revise if you're over 30%.
- **Underestimating the multi-threading piece.** The paper's core claim (no synchronization needed) only shows up if you actually parallelize your Dynamic NN-Descent implementation — a single-threaded version alone won't demonstrate the paper's main contribution.
- **Solo pacing.** You're doing every role yourself (coding, writing, slides, presenting) — the week-by-week timeline above is built to keep each chunk small; try not to let two steps stack into the same week.

---

## Next Steps From Here

1. Submit the paper for approval today or tomorrow.
2. Reply here when you're ready to start the Python baseline (Step 3) — I can scaffold the NN-Descent code with you.
