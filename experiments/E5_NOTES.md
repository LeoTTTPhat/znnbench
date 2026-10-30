# E5: noise robustness under discretization (notes)

Files:
* `experiments/configs/E5.py`: the sweep config, plus a build CLI.
* `experiments/tasks/E5_start.py`: the problem wrapper `NearSolutionStart`, used for the near-solution start arm.
* `experiments/analyze_E5.py`: figures, tables, `summary.json` and `SUMMARY.md`.
* `experiments/E5_NOTES.md`: this file.

Nothing under `znnbench/` or `tests/` was edited.

## 1. Question

The noise-tolerant ZNN literature proves steady-state residual bounds in continuous time (CT). Some examples:
* IEZNN gives E → 0 under constant noise.
* DIEZNN rejects a linear ramp.
* HNTZNN rejects a harmonic at a known frequency.
* SPTC-AN-FOZNN rejects any bounded noise with ‖ν‖ < ζ.
* Linear ZNN keeps a floor of order ‖ν‖/γ.

E5 measures what remains of these floors after sampling with Euler and Taylor–Zhang. It compares every discrete floor with the CT prediction for the same γ and the same steady window.

## 2. Exact protocol

| axis | values |
|---|---|
| problems | `matrix_inversion` (n = 4, m = 16) and `linear_system` (n = 4, m = 4), κ = 10, 1 rad/s, T = 6 s, `seed: $seed`. **Control:** `linear_system` with `freq = 0` (label `linear_system_static`). A and b are constant there, so the truncation error is zero and the floor is the pure discrete-time noise response. |
| models | `znn` (linear activation); `ieznn` (default λ = γ²/4); `dieznn` (default ξ = γ); `hntznn` (ω = 2π, matched to the harmonic noise; λ = 1 and the Li activation, as in the paper); `sptc_an_foznn` (Δ = `noise_bound` = 8, so ζ = 5Δ = 40, α = 0.5, t_c = 1); `znn` with `activation=sign_bi_power` (r = 0.5, label `znn_sbp`). **Review extension (harmonic + noise-free only):** `hntznn_lin` (HNTZNN, linear activation, ω = 2π matched) and `hntznn_lin_mis` (linear, internal model at 1.1·2π, 10 % mismatched) |
| noises | none; `constant` level 8; `linear` slope 8, offset 0; `gaussian` std 1/√12; `harmonic` amplitude 1, ω = 2π, phase 1; `random_walk` σ = 1 |
| discretizations | `euler`, `taylor_zhang` |
| τ | {1e-2, 1e-3, 1e-4} (mini: {1e-2, 1e-3}) |
| gains | fixed h ∈ {0.1, 0.3} (PLAN) **and** fixed γ ∈ {2, 10}. Each mode runs as one batched simulate call (`batch_gains`). |
| initial state | `rand` = each problem's default x0 ~ U[−2, 2] (PLAN); `near` = x*(0) + 0.1·U[−1, 1] (wrapper `experiments.tasks.E5_start:NearSolutionStart`, drawn from simulate's own x0 stream). **Review extension:** paired unit check `unit_Ex1` / `unit_Ex1e3` = x*(0) + 1e-4·U[−1, 1] (`E5_start:UnitPairStart`) with E in its own unit / in units of 1e-3 (`R_common:ScaledError`, scale 1e3), noise-free, models znn (control), znn_sbp, hntznn |
| seeds × B | **5** × 4 = 20 trajectories per cell (PLAN: 3; raised in the review so that the seed-cluster bootstrap is not degenerate) (mini: 1 × 4) |
| info / record | `analytic`, `record_every = 1`, metrics eps {1e-3, 1e-6}, frac 0.25, extra `final_residual`, `initial_residual`, `max_residual` |

* **Noise.** It is added in the design-formula space (J ẋ = model + ν), so for linear ZNN Ė = −γE + ν. Every entry of ν has the same value, following the [a]_{n×n} convention of the cited papers.
* **Floor of a trajectory.** This is the MSSRE: max ‖E‖ over [0.75T, T], set to inf if the trajectory diverged.
* **Cell summary.** A cell is (problem, start, model, noise, disc, gain arm, τ). It is summarised by the median and IQR over its trajectories. A median of inf means that at least 50 % of the trajectories diverged; these cells are shown as "div".
* **Stochastic noises.** For `gaussian` and `random_walk`, only moments are predicted, so the CT comparison uses the window-mean ‖E‖ (`mean_ssre`).

**Noise levels (docs/spec/noise_info.md):**
* [8] and [8t]: Yan et al., IEEE Access 7 (2019), verified first-hand by the noise_info verifier. Liao et al., Mathematics 2022, Sec. 4.3, also uses them.
* Gaussian std 1/√12: same variance as the bounded random noise U[−0.5, 0.5] of Liao et al. 2022, Remark 3. The Gaussian key itself is a benchmark construction (check level C).
* Harmonic sin(2πt + 1): the first term of the periodic example of Liao et al., Front. Neurorobot. 2022, Eq. (12).
* Random walk σ = 1: a benchmark extension. No ZNN source was found.

**SPTC Δ.** The paper sets ζ = 5Δ with Δ ≥ the noise bound. Δ = 8 covers the per-entry bound of every bounded noise in the grid. Then ζ = 40 > sup‖ν‖₂ = 8√m (32 for m = 16, 16 for m = 4), so CT theory predicts exact rejection for the constant, harmonic, Gaussian and random-walk noises (the last two at a 4σ and 3σ√T level). It predicts no guarantee for the ramp, whose ‖ν‖₂ exceeds ζ after t ≈ 1.25 s (m = 16) or 2.5 s (m = 4).

## 3. Rationale for the deviations from PLAN.md (all additions; nothing in the PLAN grid is dropped)

1. **Fixed-γ arm (γ ∈ {2, 10}).**
   * Under the PLAN's fixed-h (DTZNN) convention, γ = h/τ grows as τ → 0, so every CT noise floor changes with τ, for example c√m·τ/h for linear ZNN. Only at fixed γ must the discrete floor converge to a τ-independent CT prediction.
   * γ = 10 is the usual ZNN literature gain.
   * γ = 2 is the published SPTC-AN-FOZNN gain (Yang et al. 2025). Under fixed h, that model's γ = h/τ ≥ 10 multiplies its quadratic predefined-time term, and the model diverges in 92 % of the cells.
   * **Caveat:** at γ = 2, the IEZNN poles (double, at −1) and the DIEZNN poles (−1, −2, −2) have not decayed by the window start. Use γ = 10 for fixed-γ comparisons of the linear models; γ = 2 is meant for SPTC.
2. **Static control problem.** It isolates the discrete noise response from the truncation floor. The analysis validates it against the exact discrete error recursion (Section 5). It costs +50 % runs, which is cheap.
3. **Near-solution start arm.**
   * Li-activated HNTZNN (h = 0.3, and Taylor–Zhang at h = 0.1 on matrix inversion), sign-bi-power (h = 0.3) and SPTC-AN-FOZNN are only locally stable in discrete time from the U[−2, 2] states. This is documented in `docs/spec/SMOKE_MATRIX.md` and in the variants spec, Observations 2–3. Their divergence is a transient-stability effect (E2/E3), not a noise floor.
   * The `rand` arm (PLAN) is kept as the primary heatmap. The `near` arm supplies floors for those models, and the floor-vs-τ and CT-comparison figures use it.
4. **Models.** "Fractional PT-ZNN" is the registered `sptc_an_foznn`. "Sign-bi-power" is `znn` with the registered `sign_bi_power` activation, (sig^r + sig^{1/r})/2. The Li activation of HNTZNN is twice that function.
5. **Mini scale** (for validation only): τ ∈ {1e-2, 1e-3}, 1 seed × B = 4. Use `--scale full` for the full-scale results.
6. **Linear-activation HNTZNN, matched and 10 % mismatched (review).** This is the only way to test the internal-model claim free of the unit-dependent Li chattering floor, with exact CT and DT predictions. It answers "matched or mismatched?" and costs +9 % (review log R4).
7. **Paired unit check (review).** Checklist item 1 asks that no SBP or Li headline rest on one arbitrary unit. The check costs +7 % (R5).
8. **5 seeds at full scale (review).** The CIs are now seed-cluster bootstraps, and 3 clusters give at most 10 distinct resamples (R2).

## 4. CT predictions and statistics (analyze_E5.py)

**CT predictions.** Each is the noise-only response from E(0) = 0. All entries are equal, so ‖E‖ = √m·|e|. The response is evaluated on the same window [4.5, 6] s. A finite horizon matters: the DIEZNN pole at −1 is still active in the window.

| model | deterministic noise | stochastic noise |
|---|---|---|
| ZNN, IEZNN, DIEZNN | exact, using the matrix exponential of the model plus a noise-generator system | exact second moment (Van Loan recursion). Per-step Gaussian of std s is treated as its CT white-noise equivalent of intensity s²τ; the random walk is integrated white noise of intensity σ². The window mean uses the χ_m factor. |
| ZNN-SBP | LSODA (BDF fallback) | Gaussian: stationary density of the scalar SDE. Random walk: quasi-static e = Φ⁻¹(ν/γ). |
| HNTZNN (Li) | LSODA (BDF fallback) | not predicted (n/a) |
| SPTC-AN-FOZNN | Theorem-2 sliding mode: 0 if \|ν_i\| < ζ/√m on the window; otherwise (ramp) the quasi-static root of the post-t_c error dynamics | as for deterministic noise, with the bound checked at 4σ (Gaussian) and 3σ√T (random walk) |

* Values below 1e-13 count as exact 0 (round-off).
* **DT predictions** (linear models only) push the same noise through the exact discrete error recursion of the look-ahead formula, with the auxiliary states advanced by forward Euler as in the zoo. The static control should reproduce them.

**Statistics.**
* Medians and IQR over the trajectories of a cell.
* **Headline ratios** are floor(noise)/floor(noise-free), paired by trajectory (seed, x0 index), with a **seed-cluster bootstrap** (review): whole problem seeds are resampled with all their paired trajectories (2000 resamples, seed 20261018). No CI with fewer than 3 seeds (the mini has 1 seed, so its CIs read "--").
* **Floor-vs-τ slopes** are least-squares fits on log-log axes over the **asymptotic range**: points at the 1e-12 round-off floor, transient-limited points (linear models, see the review log) and pre-asymptotic points (local slopes disagreeing by > 0.2) are dropped. The all-points LS slope and the local slopes are reported alongside, with a range flag (2-point fits are flagged). CI: the same seeds are resampled at every τ (a seed is one instance shared across τ).
* **Censoring:** diverged trajectories count as +inf (worst) in every median, CI and ranking and are never dropped; cells with < 50 % diverged trajectories are marked `*`, cells with a diverged median `div`; SUMMARY.md has a censoring section and `summary.json` the censored counts of every key number.
* **Kendall τ_b** compares the model ranking predicted by CT alone, and by CT plus the noise-free floor, with the measured ranking, per (problem, start, disc, arm, τ, noise). It is also computed between the rankings at the largest and the smallest τ.

## 5. Mini-run results (results/mini/E5, figures/mini, results/tables/mini)

**Run (re-run after the review, 2026-10-18).**
* Build: 2064 specs, 16 512 trajectories, 6.81e6 steps, 43 tasks (the 1728 PLAN-grid specs are unchanged; +192 harmonic-arm and +144 unit-check specs).
* Memguard pool with `--workers 3 --max-gb 40`: 6 min 00 s wall, all 43 tasks done, no errors.
* Statuses: 14 475 `ok`, 2037 `diverged`, 0 `error`/`timeout`/`skipped` (all divergences are SPTC-AN-FOZNN, rand-start HNTZNN and rand-start ZNN-SBP, as before).
* Analysis: about 30 s. The numbers below are unchanged by the re-run (same seeds); the review log lists what changed in their interpretation and their CIs.

**Headlines.** Values are medians over problems × discs × τ, near start, unless stated otherwise. Read them together with the review log:
* The DIEZNN values are finite-horizon transients (R3).
* The SBP and Li-HNTZNN statements hold in the benchmark's unit of E only (R5).
* The HNTZNN-lin arm shows that discrete rejection of a matched harmonic is O(τ), not exact (R4).
* The Kendall values are descriptive (R7).

* **Linear models are explained by superposition.** For ZNN, IEZNN and DIEZNN, the discrete floor equals the CT noise response plus the noise-free discretization floor: median |log10 gap| = 0.011, and 98.8 % of the cells lie within ×2.
* **The static control matches the exact discrete recursion.** The median log10(measured/DT) is 1.5e-4.
* **The only CT/DT difference for linear models is per-step Gaussian noise.** Euler amplifies its RMS by √(2/(2−h)). Measured DT/CT is 1.03 at h = 0.1 and 1.08 at h = 0.3, exactly the theoretical values. Measured/DT is 0.997–1.04.
* **IEZNN under constant noise.** The penalty is 1.000 [1.000, 1.000] at every setting, including fixed γ. On the static control the floor sits at round-off (1e-15.7). So the CT claim survives sampling exactly: the discrete integrator also has its pole at 1. On time-varying problems the floor is the noise-free truncation floor. For comparison, the linear-ZNN penalty is 2.8e3 (Euler, h = 0.1).
* **DIEZNN under the ramp.**
  * Penalty 12.3 (near start, h = 0.1), against 1.3e3 for IEZNN and 1.7e4 for linear ZNN.
  * The rand arm gives a penalty of 1.4–1.8, because the slow mode excited by the initial state already dominates the noise-free floor.
  * The residual ramp effect is the CT finite-horizon response of the pole fixed at −1: measured/(CT + noise-free) = 0.94. It is not a discretization artefact.
* **Fixed h, floors shrink with τ as CT predicts.**

  | cell | measured slope | CT slope |
  |---|---|---|
  | ZNN, constant | 1.000 | 1 |
  | ZNN, Gaussian | 1.03 | 1 |
  | IEZNN, ramp | 2.00 | 2 |
  | IEZNN, random walk | 1.52 | 1.5 |
  | ZNN, random walk | 0.99 | 1 |

  At fixed γ = 10 the deterministic and random-walk floors are τ-independent (|slope| < 0.05). The per-step Gaussian floor still scales as τ^{1/2} (measured 0.52, CT 0.5), because its CT white-noise equivalent has intensity s²τ. Under the DTZNN convention, "noise tolerance" therefore improves with the sampling rate only because the gain γ = h/τ grows.
* **Sign-type activations depart from CT.**
  * At fixed h, the noise-free floors of ZNN-SBP and Li-HNTZNN barely decrease with τ (LS slopes 0.10 and 0.32). The ZNN-SBP Euler floor at h = 0.1 is 2.1e-3, against the chattering estimate √m·(h/4)^{1/(1−r)} = 1.25e-3 to 2.5e-3.
  * HNTZNN's matched-harmonic "rejection" (penalty 1.05 at h = 0.1) therefore only means that the harmonic response lies below its own τ-independent chattering floor of about 1e-2.
  * HNTZNN diverged in 26 % of the rand-start cells and in 0 % of the near-start cells. The rand-start divergence is concentrated at h = 0.3 (83 % of the Euler cells, 100 % of the Taylor–Zhang cells) and in Taylor–Zhang at h = 0.1 on matrix inversion. ZNN-SBP diverges from rand starts only at h = 0.3 (42 % Euler, 67 % Taylor–Zhang).
* **SPTC-AN-FOZNN.**
  * It diverges in 92 % of the fixed-h cells.
  * At γ = 2 with a near start, its noise-free Euler floor is 0.94·ζτ, the sliding-mode chattering band. CT predicts 0.
  * The constant-noise penalty at γ = 2 is 1.3, so the bounded noise is rejected up to the chattering band.
* **Rankings.**
  * CT-only versus measured: median Kendall τ_b = 0.33 (480 settings). By noise: ramp 1.0, constant 0.33, harmonic 0.33, random walk 0.20, Gaussian 0.0.
  * Adding the noise-free floor to the CT prediction raises the median to 1.0.
  * A model ranking derived from CT noise analysis alone does not carry over to discrete time.
  * The ranking at τ = 1e-2 versus 1e-3 is fairly stable: median τ_b = 0.87.

**Figure inspection.** All 15 PNGs were read. I fixed:
* literal "--" in "Taylor--Zhang" in the figures;
* CT lines hidden under the error bars (z-order);
* the "CT: 0" strip covering the scatter points;
* round-off CT values (1e-17) stretching the axes;
* "−0.0" annotations;
* exact-zero floors shown as "n/a".

The figures now follow `style.py`: CMAP_SEQ for the floor heatmaps; CMAP_DIV centred at 0 for the penalty and CT-gap heatmaps; DOUBLE width; at most 5 series per panel with a legend; no dual axes; categorical colours from `keys.style_in_group` ("disc" group in the floor-vs-τ figure, "noise" group in the scatter). Models appear only as heatmap rows or as panels, so no model colour is needed; `znn_sbp` is not in `keys.GROUPS["model"]`.

## 6. Outputs

**Figures** (`<figdir>/`):
* `fig_E5_floor_heatmap.pdf` (main: matrix inversion, rand, h = 0.1), plus the `_h0.3`, `_gamma10`, `_gamma2`, `_near`, `_linear_system` and `_static` variants;
* `fig_E5_penalty_heatmap` and `fig_E5_penalty_heatmap_gamma10_near`;
* `fig_E5_ct_gap_heatmap` and `fig_E5_ct_gap_heatmap_gamma10`;
* `fig_E5_floor_vs_tau` (8 selected cells, h = 0.1, near) and `fig_E5_floor_vs_tau_gamma10`;
* `fig_E5_ct_vs_measured` (linear system) and `fig_E5_ct_vs_measured_static`;
* (review) `fig_E5_harmonic_arm_gamma10` and `fig_E5_harmonic_arm_h0.1`: ZNN, HNTZNN (Li), HNTZNN-lin matched and mismatched, with CT and exact-DT predictions.

Heatmap markers (review): † = transient-limited, * = some trajectories diverged, div = median diverged.

**Tables** (`<tabdir>/`):
* `E5_floors.tex`: log10 median MSSRE "τ=1e-2 / 1e-3 / 1e-4", for matrix inversion, rand, h = 0.1;
* `E5_floors_gamma10_near.tex`, `E5_floors_linear_system.tex`, `E5_floors_static.tex`;
* `E5_headline.tex`: noise/noise-free ratio with bootstrap CI at τ = 1e-3, for 6 comparisons × 3 arms × (problem, disc);
* `E5_slopes.tex`: LS slope with CI, local slopes and the CT slope;
* `E5_kendall.tex` (with the steady-only variant);
* (review) `E5_unit_check.tex` (paired E vs 1e3·E noise-free floors) and `E5_harmonic_arm.tex`.

**Results folder:**
* `cells.csv`: every cell with its median, IQR, CT and DT predictions, penalty and CT gap;
* `summary.json`: status counts, cost, `key_numbers`, headlines, slopes, Kendall, static validation and CT agreement;
* `SUMMARY.md`: headline tables and a data-driven interpretation.

## 7. Full-scale cost estimate and commands (updated by the review, 2026-10-18)

| item | value |
|---|---|
| specs / trajectories | **15 480 specs, 123 840 trajectories** (B·G = 8 per spec): PLAN grid 3 problems × 6 models × 2 discs × 3 τ × 2 gain modes × 6 noises × 2 starts × 5 seeds = 12 960, + harmonic arm 2 models × 2 noises × 3 × 2 × 3 × 2 × 2 × 5 = 1440, + unit check 3 models × 1 noise × 3 × 2 × 3 × 2 × 2 unit arms × 5 = 1080 |
| total simulate steps | 3.44e8 (τ = 1e-4: 90 %, τ = 1e-3: 9 %, τ = 1e-2: 1 %) |
| measured wall per step | 1.40e-4 s overall (1.45e-4 s without the early-stopping diverged SPTC runs; mini, B·G = 8, n_x = 4/16, one BLAS thread; τ-independent: 1.50e-4 at τ = 1e-2 and 1e-3) |
| simulate time | 3.44e8 × 1.45e-4 s ≈ 49 900 s ≈ 13.8 core-hours |
| total, with overhead | **≈ 15.6 core-hours** (mini overhead +13 %: 3 × 360 s worker time against 956 s inside simulate) |
| tasks | **645 tasks** of 24 specs, cost-balanced (`balance: true`, max/min estimated chunk cost 1.000), ≈ 87 s each |
| recommended workers | **`--workers 24`: ≈ 40 min wall**; `--workers 32` (M3 Ultra): ≈ 30 min; 16: ≈ 1 h |
| peak RSS per task | **≈ 0.13 GB measured** (mini high-water 127 MB, median 125 MB); at τ = 1e-4 the two recorded (N+1) × 8 traces add ≈ 8 MB → budget **0.25 GB/task**; 24 workers ≈ 6 GB, far below the 200 GB cap |
| disk | ≈ 0.25 KB per trajectory row (mini 4.1 MB for 16 512 rows) → ≈ 35 MB of parquet |

**Commands** (run from `znn-bench/`; build first, the launch is resumable):

```bash
# build (full scale; writes results/E5/{config.json,specs.jsonl,chunks/,tasks.jsonl}; 645 tasks)
python experiments/configs/E5.py --scale full --out results/E5
#   equivalent: E5_SCALE=full python -m znnbench.runner build experiments/configs/E5.py \
#               --out results/E5 --chunk-size 24 --balance
# launch under the memory guard (resumable; re-run the same command after a crash)
python tools/memguard.py pool --tasks results/E5/tasks.jsonl --workers 24 --max-gb 200
# progress
python -m znnbench.runner status results/E5
# analyse (figures -> figures/, tables -> results/tables/, summary -> results/E5/)
python experiments/analyze_E5.py --results results/E5 --figdir figures --tabdir results/tables
```

**Mini validation, as run (2026-10-18):**

```bash
python experiments/configs/E5.py --scale mini --out results/mini/E5
python tools/memguard.py pool --tasks results/mini/E5/tasks.jsonl --workers 3 --max-gb 40
python experiments/analyze_E5.py --results results/mini/E5 --figdir figures/mini --tabdir results/tables/mini
```

**Scale switch.** Run as a script with `--scale mini|full`, or set the environment variable `E5_SCALE` (fallback `ZNNBENCH_SCALE`; default `full`) when `python -m znnbench.runner build` loads the file.

**Zoo keys.** Every key is filtered through `znnbench.available()` at build time, so a missing key drops out instead of breaking the build, and a rebuild picks up renamed or added keys. Rebuilding into the same folder re-runs only the chunks whose specs changed.

## 8. Zoo observations (no zoo bug found)

* **SPTC-AN-FOZNN after t_c.** The implemented error dynamics, Eq. (18) (verified against the paper by the variants verifier), contain +E/(t − t_p) with t ≥ t_c > t_p. This is an anti-damping term, and only the ζ sign term and the quadratic term stabilise it. This is a property of the source equations, not a code fault. It explains why the discrete floor is the ζτ chattering band, and why the model diverges whenever the quadratic term's explicit step is too large (fixed h, γ = h/τ ≥ 10). It is worth mentioning in the survey text.
* **DIEZNN mapping.** With ξ = γ (a documented ASSUMPTION), the pole at −1 is independent of γ. With T = 6 s, its e^{−t} mode dominates the DIEZNN floors from random starts, at 1e-4 to 1e-5. The CT finite-horizon prediction accounts for this.
* **Li-activated HNTZNN** diverges under Taylor–Zhang from the U[−2, 2] matrix-inversion states already at h = 0.1, and under Euler at h = 0.3. Its |e|² branch exceeds the Taylor–Zhang stability limit (local step γτ·2|e| > 1 for |e| > 5). This is consistent with the smoke-matrix notes.
* **IEZNN at γ = 2** (λ = 1): its double pole at −1 leaves a floor of about 1 at T = 6. This is a horizon effect, so do not read the γ = 2 cells of IEZNN or DIEZNN as noise floors.
* **(review) HNTZNN auxiliary states by forward Euler.** The zoo advances X, C with z + τ ż, as documented ("forward Euler, as the source papers do"). The discrete oscillator then has eigenvalues 1 ± jωτ (|·| > 1), not e^{±jωτ}, so the matched internal model is detuned by O(ω²τ). The rejection is therefore O(τ) in discrete time, not exact (review log R4). This is a design property, not a bug. An exact rotation update of (X, C) would restore exact matched rejection, which would make a useful zoo option.
* **(review) `R_common.ScaledError` reads additive noise in the scaled unit** (documented in its docstring), so a noisy unit check needs a noise-target scaling wrapper. That is why the E5 unit arm is noise-free.
* **(review) Runner `exclude` rules match wrapper keys, not kwargs.** The unit arm therefore uses its own key, `E5_start:UnitPairStart`.
* **(review) Noise streams.** E5's additive noise uses per-trajectory streams (`RowRNG`: (seed, 2, x0 index), common across the gains of a batch and across models). The shared-stream premise of REVIEW_CHECKLIST item 5 holds only for `data_noise`, which E5 does not use. Pseudo-replication still arises because, for deterministic noise and the linear models, the steady floor does not depend on x0.
* No zoo bug was found in the review.

## Review log (2026-10-18)

Each item below gives severity, the change made and the effect on the numbers. All changes are in E5's own files. Nothing under `znnbench/` or `tests/` was edited. The PLAN-grid runs are identical before and after the review (same seeds), so their point estimates are unchanged. What changed is the CIs, the flags and the interpretation, plus the new arms.

**R1. Critical, fixed. The analysis script did not run.**
* Cause: an earlier review pass was interrupted (file time 13:31). It left `boot_slope` half-converted: a dict return in one branch, 5-tuples in the others, and an undefined `npts`. Meanwhile `slopes()` still unpacked 5 values, so `analyze_E5.py` crashed.
* The `results/mini/E5` outputs on disk came from the 13:20 version.
* Fix: rewrote `boot_slope` (dict API, asymptotic range, cluster CI) and its consumer.
* Effect: the analysis runs again (about 30 s).

**R2. Major, fixed. Pseudo-replication (checklist 5).**
* Cause: headline and slope CIs resampled trajectories. For deterministic noise and the linear models, the B = 4 trajectories of a seed have identical floors, because the steady state does not depend on x0. This gave zero-width CIs such as "1.99e3 [1.99e3, 1.99e3]" that carry no information about instance variability.
* Stochastic-noise trajectories are independent noise replicates, but they share the problem instance.
* Fix: seed-cluster bootstrap for ratios and slopes. Whole seeds are resampled; for slopes the same seeds are used at every τ. No CI is reported with fewer than 3 seeds. Full-scale seeds were raised from 3 to 5, which adds 67 % cost.
* Effect: point estimates are unchanged. The mini CIs now read "--" (1 seed). The full-scale CIs will reflect between-instance spread.
* Synthetic check: 5 seeds × 4 identical replicates with per-seed ratios 2–6. The cluster bootstrap gives [2.0, 6.0]; the trajectory bootstrap gives [2.16, 5.64].

**R3. Major, fixed (flagged). Several "floors" are T = 6 s transients (checklist 6: horizon vs slow modes).**
* Derived slow modes:
  * DIEZNN (poles −1, −ξ, −ξ, with ξ = γ): the initial-state slow mode is e0·e^{−t}/(ξ−1)². The ramp leaves Q·e^{−t}/(ξ−1)², i.e. 1.1e-3 per entry at ξ = 10, t = 4.5.
  * IEZNN (double pole at −γ/2): the initial-state transient at 4.5 s is 0.061·e0 at γ = 2 and 4e-9·e0 at γ = 10.
  * HNTZNN internal-model pair: Re ≈ −ω²λ/(2γ). It decays only to 0.41 at γ = 100 and to 0.92 at γ = 1000 within 4.5 s.
* Fix: completed the transient diagnostics. Each cell's value is compared with two decaying parts: the exact discrete initial-state response times the median ‖E(0)‖, and the decaying part of the CT noise response. A cell is transient-limited when a decaying part exceeds 10 % of the value.
* 839 of 2208 checked linear-model cells are transient-limited:

  | model / arm | fraction transient-limited |
  |---|---|
  | DIEZNN | 49–83 % |
  | HNTZNN-lin | 56–100 % |
  | IEZNN, γ = 2 | 67 % |
  | ZNN, γ = 2 | 18 % |
  | IEZNN, fixed h or γ = 10 | 7–11 % |

* Flagged cells carry † in the heatmaps. They are excluded from the slope fits (see the "range" column) and from the steady-only Kendall variant.
* Effect: the DIEZNN-ramp slopes (2.16 and 2.07 at h = 0.1) are no longer reported as floor slopes. They are the τ² scaling of the transient Q·e^{−t}/(ξ−1)² with ξ = h/τ. The DIEZNN conclusions are now worded as finite-horizon (100 % of the DIEZNN-ramp cells are transient-limited).
* T is kept at the PLAN value of 6 s. A T = 30 s control for DIEZNN and HNTZNN-lin would cost about +2 core-hours. It is recommended if asymptotic DIEZNN floors are needed.

**R4. Major, fixed. Harmonic arm: matched only, and Li only.**
* The noise frequency (2π) and HNTZNN's internal model (2π) were matched. With the Li activation the measured value was the Li chattering floor, which is unit-dependent (R5).
* Fix: added `hntznn_lin` (linear activation, matched) and `hntznn_lin_mis` (10 % mismatched), harmonic and noise-free only, with exact CT and DT predictions (`linear_AB`).
* **New finding.** The forward-Euler internal model detunes the matched rejection by O(ω²τ) (static control, γ = 10):

  | τ | DT floor / CT floor | measured / DT | HNTZNN-lin / ZNN | CT ratio |
  |---|---|---|---|---|
  | 1e-2 | 168 | 1.00 | 0.133 | 7.8e-4 |
  | 1e-3 | 15.7 | 0.99 | 0.0124 | 7.8e-4 |

  * The fixed-γ slope is 1.04, against a CT slope of 0.
  * With a 10 % mismatch, HNTZNN-lin reaches 0.27–0.31 × the ZNN floor at γ = 10, and about 0.99 × ZNN at fixed h, τ = 1e-3.
  * Under fixed h the internal model learns at ω²λ/(2γ) → 0, so the result is horizon-limited: 0.376 × ZNN at τ = 1e-3.
* The Li-HNTZNN statement ("penalty 1.05 only means the response is below its chattering floor") still holds, now qualified as unit-dependent.
* Cost: about +9 %.

**R5. Major, fixed. Unit dependence (checklist 1).**
* SBP, Li-HNTZNN and SPTC's quadratic term have no length scale.
* Fix: added a paired, noise-free unit check (E vs 1e3·E, start radius 1e-4, ZNN as control).
* ZNN control: identical in both units (ratio 1). Static control:

  | model | floor(1e3·E) / floor(E) |
  |---|---|
  | ZNN-SBP | exactly 1.00e-3 = 1/s |
  | Li-HNTZNN | 1.94e-3 |

  So the chattering band lives in the unit of E.
* On the time-varying problems the ratio ranges from 1e-3 to 76. The ordering against linear ZNN flips in 23 of 64 cells.
* From the 0.1-radius start, E × 1e3 makes explicit SBP diverge through its |e|² branch. Stability itself is therefore unit-dependent.
* The SBP and Li conclusions are qualified as unit-dependent in SUMMARY.md and in the interpretation.
* SPTC is not in the unit arm: its ζ = 5Δ must scale with the noise unit, so a noise-free unit change is not a pure unit change for it. A noisy unit check needs a noise-scaling wrapper (zoo observation above).
* Cost: about +7 %.

**R6. Moderate, fixed. Censoring (checklist 2).**
* Causes:
  * key-number medians dropped ±inf penalties silently;
  * the CT-agreement table dropped diverged cells silently (SPTC: only 14–16 of 32 cells were used);
  * bootstrap quantiles turned +inf resamples into NaN, because `np.quantile` interpolates inf with inf.
* Fixes:
  * censoring-aware medians (inf = worst), with the counts in `summary.json` (`key_numbers_censoring`);
  * censored, no-prediction and transient counts in every CT-agreement row;
  * inf-safe quantiles (`_q_inf`);
  * `*` marks partially diverged cells;
  * a censoring section in SUMMARY.md: 2037 of 16 512 trajectories diverged, 0 failed; 544 cells have a diverged median and 33 are partially diverged.
* Effect: no key-number cell of the mini was censored (n_inf = 0), so those values are unchanged.

**R7. Moderate, fixed (reframed). The Kendall headline was over-read.**
* The median τ_b of 0.33 between CT and measured rankings has three causes:
  * ties or near-ties in the CT predictions: exact rejection gives 0 for IEZNN, DIEZNN and SPTC, and the white-noise floors are similar;
  * SPTC divergence, which ranks worst;
  * transient-limited cells.
* The CT-plus-noise-free-floor τ_b of 1 tests superposition, largely by construction.
* With 3–6 models per setting and overlapping settings, these values are descriptive only.
* Now: rankings use only the 6 full-grid models and the rand/near starts. Medians by arm are h = 0.3: 0.11, h = 0.1: 0.20, γ = 10: 0.33, γ = 2: 0.87. The steady-only variant has median 0.
* Wording changed from "CT rankings are unreliable in DT" to "CT noise analysis alone cannot rank models whose noise is rejected; the discrete noise-free floor decides".

**R8. Minor, verified (no change needed). CT and DT predictions re-derived.**
* ZNN, constant noise c: the floor is c/γ per entry. The DT value is identical for any consistent look-ahead formula, because Σc_j = 0.
* IEZNN: rejects constant noise exactly, since the DT integrator keeps its pole at 1. The ramp leaves 4Q/γ².
* DIEZNN: rejects a ramp asymptotically; the finite-horizon e^{−t} term remains at T = 6 s.
* Per-step Gaussian noise at fixed h:
  * The DT recursions of ZNN and IEZNN depend on h only (for IEZNN after the substitution w = λτz, with λτ² = h²/4), and they are forced by τν_k.
  * So the floor is exactly τ·F(h), with slope exactly 1.
  * For Euler ZNN the RMS per entry is τs/√(h(2−h)), which is the CT white-noise value × √(2/(2−h)).
  * At fixed γ the slope is 1/2 + h/(2(2−h)), i.e. 0.526 at τ = 1e-2. Measured: 0.516–0.534.
* Random walk: ZNN ∝ τ (quasi-static), IEZNN ∝ τ^{1.5} (stationary variance σ²/(2γλ)).
* SPTC: the per-entry threshold ζ/√m is correct for the 2-norm unit vector. The χ_m mean is correct for i.i.d. entries.
* Static-control validation: median log10(measured/DT) = 1.2e-3 over 532 cells.

**R9. Minor, documented. Confounds (checklist 6).** A Confounds section was added to SUMMARY.md:
* Effective gains differ at equal nominal γ:
  * IEZNN: λ = γ²/4, double pole at −γ/2;
  * DIEZNN: e-gain 2ξ + 1, pole fixed at −1 (ξ = γ is an ASSUMPTION);
  * HNTZNN: λ = 1 does not scale with γ;
  * SBP and Li: gain ~γ/(2√|e|) near the floor.
* Information advantages: SPTC is given the true noise bound (Δ = 8), and HNTZNN the true frequency.
* Noise is injected identically in E-space and internal states are not perturbed, as in the sources, so the noise level is comparable across models.
* At fixed h, SPTC runs at γ = h/τ ≥ 10, outside its published regime.

**R10. Minor, fixed. Figures (checklist 7).**
* Heatmaps: † and * markers with a footnote; harmonic-only models kept out (no n/a rows).
* Floor-vs-τ: an exact-DT line (Euler, linear models) added.
* New `fig_E5_harmonic_arm_gamma10` and `fig_E5_harmonic_arm_h0.1`: 4 panels, ≤ 6 series, with a legend.
* All 17 figures were regenerated. I inspected the near heatmap, the floor-vs-τ figure and both harmonic-arm figures.
* New tables: `E5_unit_check.tex` and `E5_harmonic_arm.tex`. `E5_slopes.tex` now also has the all-points slope, a range flag and the DT slope.

**R11. Minor, fixed. Cost and memory (checklist 8).** Re-estimated in Sec. 7 from the measured wall time per step (1.45e-4 s), with peak RSS 127 MB per task.

**Not changed (judgement calls):**
* T = 6 s and the fixed-h SPTC arm (92 % diverged): both are informative and documented.
* The mini keeps 1 seed (10-minute budget). The cluster-bootstrap code path was checked on synthetic data (R2).
