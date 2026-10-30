# ZNN-Bench: reproduction artifact

This folder contains everything needed to reproduce the ZNN-Bench results:

* the `znnbench` Python package and its test suite,
* the configurations, run scripts and analysis scripts of experiments E1–E9, of the robotics case
  study R and of its matched-effort arm R_matched,
* the complete logged results of every full-scale run, from which every figure and summary can be
  regenerated without re-running the simulations,
* every figure (PDF and PNG),
* the scripts and data of the reporting-practice meta-analysis of open-access ZNN publications.

All paths stored in the result files are relative to this folder, so every command below is run
from here.

## Contents

| path | content | size |
|---|---|---:|
| `znnbench/` | the package: problems, models, activations, gain schedules, difference formulas and other discretizations, noise models, information regimes, metrics, a per-step cost model and a sweep runner | 0.7 MB |
| `tests/` | unit and integration tests (pytest, 1148 tests) | 0.4 MB |
| `experiments/configs/` | sweep configurations of every experiment, including the random seeds | |
| `experiments/tasks/` | task scripts of the experiments with custom pipelines (E2, E3, E4, E7, E9, R, R_matched) | |
| `experiments/analyze_*.py` | analysis scripts: figures, tables, `SUMMARY.md` and `summary.json` per experiment | |
| `experiments/run_all.py`, `experiments/run_E4.sh` | orchestration of the full-scale runs | |
| `experiments/*_NOTES.md`, `PLAN.md`, `RUNBOOK.md`, `README.md` | design notes per experiment, protocol, full-scale runbook, runner guide | 1.9 MB in total |
| `results/` | logged results of every full-scale run: task lists, chunk specifications, raw per-run results (Parquet), down-sampled traces (NPZ), `SUMMARY.md` and `summary.json` per experiment | 1.2 GB |
| `figures/` | every figure as PDF and PNG | 42 MB |
| `meta_analysis/` | rubric, coding brief, sampling frame with its raw harvests, sample, per-publication codes with evidence quotes, adjudication files, analysis script (`meta_analysis/README.md`) | 11 MB |
| `docs/` | `DESIGN.md` (core API), `ZOO.md` (catalogue of every registered component), `spec/` (equations, sources, assumptions and verification of every area) | 0.6 MB |
| `tools/memguard.py` | memory-guarded, resumable task pool used for every sweep | |
| `requirements.txt`, `environment.yml`, `pyproject.toml` | environment and package metadata | |
| `MANIFEST.sha256` | SHA-256 checksum of every file of this folder | |

Check the integrity of a copy with

```bash
shasum -a 256 -c MANIFEST.sha256 --quiet && echo "all files intact"
```

## Requirements

* CPython 3.11 (tested with 3.11.15 on macOS 26, Apple silicon); the pinned versions in
  `requirements.txt` are those of the environment that produced the results.
* Regenerating figures and summaries from the logged results: any recent laptop, a few minutes per
  experiment and a few GB of memory.
* Re-running every experiment from scratch: about 110 core-hours (per-experiment estimates below) and
  a multi-core machine; see the note on the memory guard before running on a machine with less than
  about 128 GB of memory.

## Installation

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

With conda: `conda env create -f environment.yml`, `conda activate znn-bench`,
`python -m pip install -e .`. The task runner uses `.venv/bin/python` of this folder when it exists
and the current interpreter otherwise.

Quick check:

```python
import numpy as np
from znnbench import make_problem, make_model, make_discretization, simulate
from znnbench.metrics import summarize

problem = make_problem("matrix_inversion", n=4, kappa=10, seed=0)   # find X(t) = A(t)^-1
model = make_model("znn")                                           # linear-activation ZNN
h = 0.1                                                             # step size h = gamma * tau
for disc in ("euler", "taylor_zhang"):
    for tau in (1e-2, 1e-3):
        sim = simulate(problem, model, make_discretization(disc), tau=tau, gamma=h / tau, B=4, seed=0)
        mssre = np.median(summarize(sim)["mssre"])
        print(f"{disc:12s} tau = {tau:g}: steady-state residual (MSSRE) = {mssre:.1e}")
```

Expected output (a few seconds):

```
euler        tau = 0.01: steady-state residual (MSSRE) = 1.7e-02
euler        tau = 0.001: steady-state residual (MSSRE) = 1.8e-04
taylor_zhang tau = 0.01: steady-state residual (MSSRE) = 5.6e-04
taylor_zhang tau = 0.001: steady-state residual (MSSRE) = 5.8e-07
```

`znnbench.available()` lists every registered component; `experiments/README.md` explains sweep
configurations, the runner and the result format.

## Step 1: run the tests

```bash
python -m pytest -q
```

1148 tests, a few minutes. `test_memguard_pool_end_to_end` is skipped on machines with less than 60 GB
of available memory.

## Step 2: regenerate every figure and summary from the logged results

Each analysis script reads the logged results of one experiment and writes its figures to `figures/`,
its tables to `results/tables/` and `SUMMARY.md` and `summary.json` into its results folder, replacing
the shipped files; copy `results/` and `figures/` first if you want to compare. On the shipped results
the scripts reproduce every `SUMMARY.md` and every PNG figure byte for byte. Known small differences:
PDF figures differ in their embedded creation date; `results/full/E4/summary.json` gains one entry
(`fig_E4_main`) in its list of written figures; and because results are read in file-time order,
which the archive format keeps only to two seconds, `results/E1/E1_slopes.csv` may list 15 of its 800
rows in a different order and one auxiliary interval in `results/E8/summary.json` may differ in the
fifth digit; `meta_analysis/summary.json` lists its entries in an order that depends on Python's string
hashing (identical content; set `PYTHONHASHSEED` to fix the order between runs).

```bash
python experiments/analyze_E1.py --results results/E1      --figdir figures --tabdir results/tables
python experiments/analyze_E2.py --results results/E2      --figdir figures --tabdir results/tables
python experiments/analyze_E3.py --results results/E3      --figdir figures --tabdir results/tables
python experiments/analyze_E4.py --results results/full/E4 --figdir figures --tabdir results/tables
python experiments/analyze_E5.py --results results/E5      --figdir figures --tabdir results/tables
python experiments/analyze_E6.py --results results/E6      --figdir figures --tabdir results/tables
python experiments/analyze_E7.py --results results/full/E7 --figdir figures --tabdir results/tables
python experiments/analyze_E8.py --results results/E8      --figdir figures --tabdir results/tables
python experiments/analyze_E9.py --results results/E9      --figdir figures --tabdir results/tables
python experiments/analyze_R.py  --results results/R       --figdir figures --tabdir results/tables
python experiments/analyze_R_matched.py --results results/R_matched --rbase results/R --figdir figures
python meta_analysis/analysis.py
```

| exp | question | results folder | analysis script | key figures |
|---|---|---|---|---|
| E1 | Do look-ahead difference formulas deliver their residual order at fixed step size and at fixed gain? | `results/E1` | `analyze_E1.py` | `fig_E1_orders` |
| E2 | Do measured step-size limits match the root locus of the error recursion? | `results/E2` | `analyze_E2.py` | `fig_E2_theory_vs_measured` |
| E3 | Do finite-, fixed- and predefined-time convergence claims survive explicit discretization? | `results/E3` | `analyze_E3.py` | `fig_E3_tc_breakdown` |
| E4 | Do published superiority claims survive matched gain, step or effort? | `results/full/E4` | `analyze_E4.py` | `fig_E4_main`, `fig_E4_rank_heatmap` |
| E5 | Do noise-tolerance guarantees survive discretization? | `results/E5` | `analyze_E5.py` | `fig_E5_harmonic_arm_gamma10` |
| E6 | How does derivative information limit the order? | `results/E6` | `analyze_E6.py` | `fig_E6_order_collapse` |
| E7 | What do methods cost at equal accuracy? | `results/full/E7` | `analyze_E7.py` | `fig_E7_pareto` |
| E8 | Do consistent maps restore finite-step convergence of sign-type activations? | `results/E8` | `analyze_E8.py` | `fig_E8_scale_tv` |
| E9 | GNN, ZNN, prediction-correction or look-ahead at matched cost? | `results/E9` | `analyze_E9.py` | `fig_E9_workprec_znn` |
| R | What happens on two simulated robot arms (PUMA 560, Franka Panda)? | `results/R` | `analyze_R.py` | `fig_R_bars_g10` |
| R_matched | The robot comparison at matched peak or RMS joint speed | `results/R_matched` | `analyze_R_matched.py` | `fig_R_matched` |
| meta | Reporting practice in 119 open-access ZNN publications | `meta_analysis/` | `meta_analysis/analysis.py` | `meta_analysis/summary.md` |

The numbers of each experiment are in its `SUMMARY.md` (E4 and E7: `results/full/E4/SUMMARY.md`,
`results/full/E7/SUMMARY.md`) and, for the meta-analysis, in `meta_analysis/summary.md`. The design
and the review of every experiment are documented in `experiments/<exp>_NOTES.md`.

## Step 3: re-run every experiment from scratch

Move the shipped results aside so that nothing is skipped (finished tasks are recognized by their
result files), then run the sweeps:

```bash
mv results results_shipped
python experiments/run_all.py --exps E2 E3 R E8 E7 E1 E5 E6 E9 --workers 20 --name main
bash experiments/run_E4.sh 8                                  # E4: calibration -> sweep -> analysis
python experiments/run_all.py --e7-timing --name e7timing     # E7 timing: one worker on a quiet machine
python experiments/configs/R_matched.py --out results/R_matched
python tools/memguard.py pool --tasks results/R_matched/tasks.jsonl --workers 8 --max-gb 60
python experiments/analyze_R_matched.py --results results/R_matched --rbase results/R --figdir figures
```

`run_all.py` builds each experiment's task list, runs all tasks through `tools/memguard.py pool`,
analyzes each experiment as soon as its tasks are done and logs to `logs/`. Every task is resumable:
after an interruption, re-run the same command. `python -m znnbench.runner local <folder>` runs a
small sweep without the pool.

| part | tasks | core-hours (estimate) |
|---|---:|---:|
| E1 order verification | 442 | 12–16 |
| E2 stability maps | 331 | 1.7–2.5 |
| E3 convergence-time audit | 335 | 3.3 |
| E4 fairness re-ranking (calibration and sweep) | about 800 | 18 |
| E5 noise robustness | 645 | 15.6 |
| E6 information realism | 827 | 36 |
| E7 accuracy, E7 timing | 114, 18 | 1.5, 0.7 |
| E8 implicit vs explicit | 824 | 5.3–8 |
| E9 matched cost | see `experiments/E9_NOTES.md` | 12.5–16 |
| R robotics case study | 56 | 0.3–0.45 |
| R_matched | 24 | 0.3 |

All runs are seeded (seeds in `experiments/configs/`), so a re-run on the same platform reproduces the
logged results; results on other platforms or BLAS libraries can differ in the last digits. The E7
timing pass measures wall-clock time and therefore depends on the machine; run it alone, with one
worker, on an otherwise idle machine.

**Memory guard.** `tools/memguard.py` measures the physical memory footprint of all processes started
from this folder. It launches a task only while these use less than 80 % of `--max-gb` and more than
48 GB of system memory are free (`reserve_gb` in `run_pool`), and it kills and slowly restarts the
running tasks at the cap. `run_all.py` and `run_E4.sh` pass `--max-gb 200`, the cap used on the
32-core, 512 GB workstation that produced the results. On a machine with less memory, lower
`--max-gb` and `reserve_gb`; otherwise the pool waits for free memory indefinitely.

## Step 4: the meta-analysis

`meta_analysis/analysis.py` recomputes every share, interval and agreement statistic from the coded
files, without network access (Step 2). `meta_analysis/README.md` describes every file and the full
chain from the bibliographic harvest (Crossref, Semantic Scholar) through screening, the access check
and the seeded sample to the coding and adjudication. The full texts read during coding are not
included (copyright); every publication is identified by its DOI, and abstracts were removed from the
harvested metadata (the harvest scripts can download them again).

## License

Code (the `znnbench` package, tests, experiment, analysis and meta-analysis scripts,
`tools/memguard.py`): MIT License (`LICENSE`). Data (results, figures, meta-analysis data files):
CC BY 4.0 (`LICENSE-DATA.md`).
