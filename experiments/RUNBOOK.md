# ZNN-Bench full-scale runbook (2026-10-18)

All commands run from `znn-bench/`. Every sweep goes through `tools/memguard.py pool` (soft limit 80 % of
the cap → stop launching; hard cap 200 GB → kill, drain, restart with half the workers). Tasks are resumable via
their done files: after any interruption, re-run the same command. Build each experiment into a FRESH folder: run
and task ids ignore changes of component defaults (`runner.hash_payload`), so a rebuild over old results can
silently reuse stale ones.

Per-experiment details (protocol, confounds, review log, exact commands): `experiments/Ek_NOTES.md`; lead's digest
of the methodology reviews: `experiments/REVIEW_SUMMARY.md`.

## Orchestrator

`experiments/run_all.py` builds the listed experiments, concatenates their task lists into
`logs/sweep_<name>.jsonl`, runs ONE memory-guarded pool, writes `logs/sweep_<name>_status.json` every
60 s and runs each experiment's analysis as soon as all its tasks are done.

```bash
# main sweep (launched 2026-10-18 18:32, 20 workers: the machine carries load 40-180 from other projects)
nohup python experiments/run_all.py --exps E2 E3 R E8 E7 E1 E5 E6 --workers 20 --name main \
      > logs/run_all_main.out 2>&1 &
# E4 and E9 after their reviews (fill their commands into EXPS in run_all.py first)
nohup python experiments/run_all.py --exps E4 E9 --workers 8 --name late > logs/run_all_late.out 2>&1 &
# E7 timing: ONE worker, ALONE, on a quiet machine (waits up to 3 h for 1-min load < 16), then analyses E7
nohup python experiments/run_all.py --e7-timing --name e7timing > logs/run_all_e7timing.out 2>&1 &
```

## Experiments, cost and order

| exp | tasks | core-h (est.) | peak RSS/task | notes |
|---|---:|---:|---:|---|
| E2 stability maps | 331 | 1.7 (≤ 2.5) | 0.47 GB | longest task CDTZNN on matrix_inversion ~7 min |
| E3 convergence-time audit | 335 | 3.3 | 0.25 GB | includes the paired error-unit arm |
| R robotics case study | 56 | 0.3–0.45 | 0.13 GB | R_off, R_on, R_units, traces |
| E8 implicit vs explicit | 824 | 5.3 (≤ 8) | 0.14 GB | error-unit arms E8_scale_ti/tv; 10 seeds |
| E7 accuracy | 114 | 1.5 | 0.3 GB | timing part is a separate pool (below) |
| E1 order verification | 442 | 12.4 (≤ 16 under load) | 0.3 GB | longest run ~848 s idle |
| E5 noise robustness | 645 | 15.6 | 0.25 GB | 5 seeds; transient-limited cells marked † |
| E6 information realism | 827 | 36 | 0.13 GB | trimmed grid (matrix_inversion at common h, seeds 0–4) |
| E4 fairness re-ranking | see E4_NOTES | | | after its review |
| E9 matched cost | see E9_NOTES | | | after its review |
| E7 timing | 18 | 0.7 | 0.4 GB | 1 worker ALONE; accept if < 10 % contended rounds |

Main sweep ≈ 76–83 core-h → about 4–6 h wall at 20 workers on the shared machine.

## Acceptance checks after each analysis

* `results/<exp>/SUMMARY.md` exists, has the censoring line, and n_failed = 0 (diverged runs are expected).
* Figures in `figures/fig_<exp>_*` (PDF + PNG), tables in `results/tables/<exp>_*.tex` (compile them).
* E7: < 10 % contended timing rounds; same-operation timing controls with CIs containing 1.
* Compare headline numbers with the mini-run numbers in `REVIEW_SUMMARY.md`; investigate any sign flip.

## Known open items (not blocking)

* `simulate.chatter_vec` is NaN at an exactly zero residual; `rate_alt` has no magnitude threshold (E8/R use
  their own guards); the runner does not export `wall_dynamics`.
* HNTZNN internal states are stepped by forward Euler (O(τ) detuning of harmonic rejection, measured in E5).
* `lcc_znn`'s default activation overflows from the benchmark's random starts (E7 adds `lcc_znn_linear`).
* Optional: T = 30 s control for DIEZNN (ramp) and HNTZNN (harmonic) asymptotic floors (E5).
