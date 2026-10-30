# E6: information realism (where does the time derivative come from?)

Files: `experiments/configs/E6.py` (config builder), `experiments/analyze_E6.py` (analysis),
this file. The runner handles every run, so E6 needs no custom task script. Mini outputs are in
`results/mini/E6/`, `figures/mini/fig_E6_*` and `results/tables/mini/E6_*.tex`.

## 1. Question and hypothesis

The PLAN's hypothesis: the time-derivative estimate amplifies measurement noise by about σ/τ, so
the high-order advantage collapses, and which formula is best depends on the regime. E6 tests
this as a regime map (σ × τ → best formula) and as MSSRE-vs-τ curves for each σ. The analysis
answers three sub-questions:

1. **σ = 0:** does the order of the derivative information cap the order of the formula?
2. **σ > 0:** how does the noise floor scale with τ and σ, and how much does each (formula,
   estimator) pair amplify the noise?
3. **Regime map:** which formula wins each (σ, τ) cell, how significant is the win, and how far do
   the rankings move away from the noise-free analytic ranking (Kendall τ_b)?

## 2. Exact protocol (full scale)

| factor | levels |
|---|---|
| model | `znn` (linear activation), fixed-h mode, γ = h/τ |
| problems | `linear_system` (reference: both h protocols, seeds 0–9) and `matrix_inversion` (robustness check: common h only, seeds 0–4; trimmed in the 2026-10-18 review), both `n=4, kappa=10, seed="$seed"` (one instance per seed) |
| horizon | T = 12 s for every τ; steady window [9, 12] s (problem kwarg `T`) |
| τ | 1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4 (the E1 grid) |
| h protocols | **common** h = 0.1 for every formula; **own h_opt**: each formula at its optimal step h_opt (the argmin of the spectral radius of the linear-ZNN error polynomial, `zead.lookahead_stability`, rounded to 3 significant digits) |
| information regimes | `analytic`, `bd1`, `bd2` (through simulate); `savgol` (window 5, degree 2), `alpha_beta` (α = 0.5, Benedict-Bordner β), `kalman` (Kalata index 1), `bd5` (savgol with window 6 and degree 5, EXTENSION), all through `wrappers.EstimatedDerivativeProblem` |
| measurement noise σ | 0, 1e-8, 1e-6, 1e-4, 1e-2 |
| formulas | chosen at build time (§3); currently `euler` (q=2), `taylor_zhang` (3), `zead_6i_guo2017` (4), `zead_7i_cang2024` (5, own-h_opt protocol only), `zead_11i_chen2023` (6), `uhlig_4_6` (6) |
| replication | `linear_system` 10 seeds × B = 4, `matrix_inversion` 5 seeds × B = 4 initial states (see §5) |
| metrics | MSSRE = max ‖E‖ over the last 25 % of T (headline), mean SSRE, final residual, t_ε for ε ∈ {1e-3, 1e-6}, online statistics; traces at 64 points |

**How σ enters (docs/spec/noise_info.md):**
* `analytic`, `bd1`, `bd2`: `simulate(info=..., data_noise=σ)`. N(0, σ²) is added to every data
  array (A and b) that the model sees. With `analytic`, ∂E/∂t stays exact, so the noise enters
  only through E and J; with `bdk`, it is also differenced.
* Estimators: `simulate(info='analytic', data_noise=0)` on
  `EstimatedDerivativeProblem(P, method, ..., meas_std=σ, seed="$seed")` (spec §2.3, item 6). The
  same noisy samples feed E, `solve_J` and the estimator.
* **The two routes are equivalent.** The simulate route (`bd1`, data_noise = 1e-4) and the wrapper
  route (`savgol` window 2 / degree 1, which is BD1 by construction, meas_std = 1e-4) give the
  same MSSRE distribution: the ratio of medians over 6 seeds is 1.000 (Euler) and 1.047 (ZeaD-11i),
  within the seed-to-seed scatter. Simulate-based and estimator-based regimes can therefore be
  compared directly.
* Recorded residuals always use exact data (`exact_data`), so the MSSRE is the true tracking error.

**Statistics.** The statistical unit is a cell = (problem, h protocol, info, σ, formula, τ). The
analysis reports:
* the median and IQR over all trajectories;
* seeded **seed-cluster** bootstrap 95 % CIs (2000 resamples, seed 20261018) of medians and of
  paired median ratios. Pairs are matched on (seed, x0 index); seeds are resampled and all of a
  seed's trajectories are kept;
* Kendall τ_b between the formula ranking at τ = 1e-3 of each (info, σ) and the ranking for
  (analytic, σ = 0);
* empirical orders at σ = 0, fitted only on the asymptotic range (rule revised in the review, §10):
  non-finite medians and points ≤ 1e-12 are dropped (measured round-off of MSSRE here ≈ 3e-15);
  only the trailing contiguous run of usable τ (small-τ end) is used; the window grows from the two
  smallest τ towards larger τ while every consecutive-ratio slope stays within 0.15 of the
  window's least-squares slope (the tolerance of tests/test_problems_robot.py). Reported with the
  number of points, next to the all-points fit; 2-point estimates are flagged († / open marker),
  slopes outside [0, q+3] are flagged (* asymptotic, ‡ all-points);
* the noise-floor τ-slope uses clearly noise-dominated cells, MSSRE(σ) ≥ 20 × MSSRE(σ = 0)
  (truncation share ≲ 5 %), restricted to τ ≤ τ_asym, the largest τ inside every formula's
  analytic σ = 0 asymptotic window (above it the q = 6 formulas are unstable or pre-asymptotic);
  0 means the floor does not depend on τ, −1 would mean σ/τ. The noise amplification
  G = MSSRE/σ is taken at the smallest τ in cells with MSSRE(σ) ≥ 5 × MSSRE(0);
* **censoring:** diverged trajectories are +inf = worst case inside medians, IQRs, bootstrap CIs,
  rankings and ratios (two diverged values tie); failed runs are counted (n_fail) and excluded;
  CIs need ≥ 3 seed clusters, otherwise "n/a";
* **Regime map.** The best formula is the one with the lowest median MSSRE. A cell is hatched when
  the paired cluster-bootstrap CI of median(runner-up)/median(best) contains 1. White cells marked
  "div" are those where every formula diverged.
* **Advantage map.** A = log10(min over {Euler, TZ} of the median MSSRE / min over the order-≥4
  formulas of the median MSSRE), with a selection-aware paired cluster-bootstrap CI (the two minima
  are re-taken in every resample); values in parentheses have a CI containing 0. The regime
  summary counts cells with CI > 0 (high order significantly better), CI < 0 and n.s.
* **Crossover.** τ*(σ) is the largest τ at which A changes sign from + to − (log-interpolated);
  "all" means A ≤ 0 at every τ and "never" means A > 0 at every τ.

## 3. Formula selection (build time, conditional on `znnbench.available()`)

`select_formulas()` in the config runs every time the config is imported, so a rebuild picks up
zoo keys added later. The rules:
* `euler` and `taylor_zhang` are always included.
* For each role (ZeaD of the canonical figure group `keys.GROUPS['disc']`, highest-order ZeaD
  overall, highest-order Uhlig), the key with the highest `expected_order` is picked. Ties go to
  the canonical group first, then to the larger h_max. Parametric `*_general` families are
  excluded.
* The pick is made twice:
  * among keys that are stable at the common h with a 20 % margin (h_max ≥ 0.12). These run in
    both protocols.
  * among all keys with a stable interval. If this pick differs, it is added to the **own-h_opt
    protocol only** (exclude rule `{disc: k, h: 0.1}`).
* A formula whose h_max cannot be computed (no `coeffs` in `describe()`) is accepted, and the build
  prints the choice. The build also writes `<out>/e6_meta.json`.

Current selection (2026-10-18, 42 registered discretizations):

| key | q | instants | h_max | h_opt | h_opt used | ρ(h_opt) | ρ(h = 0.1) |
|---|---|---|---|---|---|---|---|
| `euler` | 2 | 2 | 2.0000 | 1.0000 | 1 | 0.0000 | 0.9000 |
| `taylor_zhang` | 3 | 4 | 1.0000 | 0.2401 | 0.24 | 0.7937 | 0.9055 |
| `zead_6i_guo2017` | 4 | 6 | 0.3333 | 0.1580 | 0.158 | 0.8536 | 0.9048 |
| `zead_7i_cang2024` | 5 | 7 | 0.1159 | 0.0722 | 0.0722 | 0.9503 | 0.9807 (own h_opt only) |
| `zead_11i_chen2023` | 6 | 11 | 0.2229 | 0.0675 | 0.0675 | 0.9347 | 0.9378 |
| `uhlig_4_6` | 6 | 10 | 0.1760 | 0.0930 | 0.093 | 0.9112 | 0.9189 |

* `zead_7i_cang2024` (registered by the concurrent zoo workflow during this task) has
  h_max = 0.116. At h = 0.1 its spectral radius is 0.981, so a transient needs about 1400 steps to
  decay by 1e-12, and near the stability edge it would amplify noise heavily. A comparison there
  would measure the stability edge, not the order, so the formula runs at its own h_opt only.
* `uhlig_3_4` (h_max = 0.054) is not stable at h = 0.1. No order-7/8 Uhlig formula is registered;
  if one appears, the same rules place it in one or both protocols.
* The task mentions zead_6i_guo2017 or zead_7i_cang2024 as examples of "highest-order ZeaD".
  Higher-order ZeaD formulas are registered: `zead_10i_wu2020` and `zead_11i_chen2023`, both order
  6. E6 therefore keeps the canonical-group ZeaD (`zead_6i_guo2017`, plus `zead_7i_cang2024` under
  own h_opt) **and** the highest-order ZeaD (`zead_11i_chen2023`, which wins the tie over
  `zead_10i_wu2020` by its larger h_max of 0.223 against 0.129).

## 4. Rationale for the design choices

* **Fixed h** is the setting the task asks for. At fixed h = γτ the derivative-noise term σ/τ is
  multiplied by the step τ, so the residual floor is expected to be about σ·G(formula, estimator,
  h), independent of τ (the measurements below confirm this). The advantage is then lost where the
  truncation error C_q τ^q falls below σ·G, so the regime boundary is σ* ∝ τ^q. The analysis
  measures the τ-slope of the floor directly instead of assuming σ/τ.
* **Two h protocols.**
  * Common h isolates the formula: every formula has the same gain γ, the same principal root
    1 − h and the same noise bandwidth.
  * Own h_opt is the literature convention (the "optimal step size"). It conflates order with
    bandwidth: Euler at h = 1 is dead-beat and does not filter noise at all, while the order-6
    formulas run at h ≈ 0.07–0.09.
* **BD5 extension.** At fixed h, a q-th order look-ahead formula keeps its residual order only if
  ∂E/∂t is accurate to O(τ^{q−1}). BD1 and BD2 (and the default S-G(5,2)) cap the order at 2 or 3.
  BD5, the exact 6-point backward difference (S-G window 6, degree 5), is the cheapest estimator
  that lets the order-6 formulas reach order 6. It tests the "matched information" scenario, in
  which noise amplification is largest.
* **Horizon: one fixed T = 12 s.** Two constraints set it.
  * At fixed h = 0.1, γ = 1 at τ = 0.1, so the transient decays like e^{−t}. The slowest selected
    root is ρ = 0.95 (`zead_7i_cang2024` at its h_opt). With the PLAN's T = 6 s, only 150 steps
    fall before the steady window at τ = 3e-2, and 0.95^150 ≈ 5e-4, so the window would measure the
    transient.
  * **T must not depend on τ.** MSSRE is the maximum of a time-varying forced response over
    [0.75 T, T]. A first mini run used T(τ) = max(6, 1000 τ) and moved the window to a different
    phase of the data at τ ≥ 1e-2. That shifted the fitted Taylor-Zhang order to 3.28 (all-points
    fit 3.02) and produced spurious kinks at the T-group boundaries. It was discarded.
  * With T = 12 s there are 90 steps before the window at τ = 0.1 (e^{−9} ≈ 1e-4 relative, below the
    O(1e-2–1) truncation error there) and 300 at τ = 3e-2 (< 2e-7 for ρ ≤ 0.95). The cost is twice
    that of T = 6.
* **Two problems.** `linear_system` is the reference problem. `matrix_inversion` checks
  robustness: it has the same linear structure, but n_x = 16 and E = AX − I.

## 5. Deviations from PLAN.md (and from the default statistics rule)

1. **10 seeds instead of 3, with B = 4 kept.** At steady state the MSSRE of linear ZNN does not
   depend on the initial state, and in `simulate` all B trajectories of a seed share the same noisy
   data stream (`rng_data`), as do the wrapper's hash-seeded samples.
   * In the first mini run (2 seeds × B = 2, 5250 rows), the within-seed relative spread
     max/min − 1 had a **median of 2e-14** and a 90 % quantile of 2e-6. The between-seed spread had
     a median of 0.14.
   * The B states are therefore replicas, not independent samples. The independent unit is the seed
     (problem instance + noise stream).
   * 3 seeds give only 3 clusters, which is useless for a bootstrap CI, so the full run uses seeds
     0–9. Seeds 0–2 are a subset, so the 3 × 4 protocol can be recovered from the same runs.
   * The bootstrap resamples seeds (cluster bootstrap), not trajectories.
   * **This also applies to other experiments that bootstrap steady-state metrics of linear models
     over B. See §9.**
2. **Own-h_opt protocol added**, and **BD5 added** (§4).
3. **T = 12 s instead of 6 s** (§4), the same for every τ.
4. **Problems.** PLAN.md does not name the E6 problems; `linear_system` and `matrix_inversion` were
   chosen.
5. **`zead_7i_cang2024` runs under own h_opt only** (§3).
6. **Only fixed-h mode.** The task asks for fixed h. A fixed-γ arm would show a τ-dependent floor,
   with h → 0 giving more filtering. It is not run.
7. **`matrix_inversion` trimmed (review 2026-10-18):** common h only and seeds 0–4. Its role is a
   robustness check of the same linear structure; the own-h_opt confound is demonstrated on
   `linear_system`, and the gain theory, order caps and regime pattern need no second h protocol.
   Saves 20,825 specs / 5.3e8 steps (−39 %).

## 6. Mini validation

Mini scale (`--scale mini`): τ ∈ {1e-1, 3e-2, 1e-2, 3e-3}, σ ∈ {0, 1e-6, 1e-2}, seeds 0–2, B = 2,
`matrix_inversion` at the common h only. Results in §6.1; full numbers in
`results/mini/E6/SUMMARY.md`.

### 6.1 Mini results (rerun 2026-10-18 18:08, after the review fixes)

Run: 4032 specs in 168 tasks (chunks of 24), 8064 trajectory rows, status ok 7421 / diverged 643 /
failed 0; 6 min 02 s on 3 workers (machine load average 31–36 from other jobs); peak RSS 127 MB per
task. The superseded 18:00 run (2 seeds for `linear_system`, 1 for `matrix_inversion`) was moved
out of the repository, not mixed.

1. **Order cap by information (σ = 0, `linear_system`, common h, asymptotic fit [points]).**
   analytic: Euler 1.85 [4], TZ 2.89 [3], ZeaD-6i 3.94 [3], ZeaD-11i 6.08 [3], Uhlig-4/6 5.96 [2†];
   BD5: ZeaD-11i 5.95 [2†], Uhlig 5.88 [2†] (matched estimator keeps order 6); BD1, α-β, Kalman:
   1.87–2.07 for every formula; BD2, S-G(5,2): 2.85–2.97 for TZ and above. No asymptotic slope
   lies outside [0, q+3] (before the fix: 12.2–15.6 for ZeaD-11i under every regime). With σ = 0
   and BD1/α-β/Kalman/BD2/S-G, all formulas of order ≥ estimator order + 1 have the same MSSRE to
   3 digits: the estimator error dominates completely.
2. **Headline ratios** (Euler / best order ≥ 4, τ = 3e-3, common h, 3-seed cluster CI): analytic
   σ = 0: 1.14e8 [8.2e7, 1.3e8]; BD5 σ = 0: 1.0e8; BD2 σ = 0: 169 [169, 477]; BD1 σ = 0: 1.35
   [1.28, 2.05]; analytic σ = 1e-2: 0.82 [0.80, 0.82]; BD5 σ = 1e-2: 0.23 [0.21, 0.23] (Euler
   4× better). `matrix_inversion` reproduces every ratio within a factor ≈ 1.5.
3. **Noise floor ≈ σ·G, independent of τ.** τ-slope of MSSRE in clearly noise-dominated cells
   (≥ 20×), τ ≤ τ_asym = 1e-2: median +0.02 (IQR −0.06 .. +0.31, 89 two-point fits); over all τ
   +0.15 (pre-asymptotic large-τ cells amplify noise more). The PLAN's σ/τ hypothesis (slope −1)
   is rejected for the residual floor at fixed h. G = MSSRE/σ at the smallest τ follows the H2 gain
   ‖(h + D)/P_h‖₂: log-log slope 1.02 / 1.01 / 1.03, R²(slope 1) 0.997 / 0.998 / 0.996
   (`linear_system` common / own h_opt / `matrix_inversion`), problem factor κ = 7.3 / 7.3 / 5.3.
   G spans 1.2 (analytic, Euler) to ≈ 1300 (BD5, Uhlig-4/6) at σ = 1e-2.
4. **Regime map (selection-aware advantage CI; share of the 12 (σ, τ) cells high / low / n.s.).**
   common h: analytic 83/17/0, BD2 58/25/17, S-G 67/17/17, BD5 58/42/0, BD1 0/33/67, α-β 0/17/83,
   Kalman 0/25/75. Own h_opt: analytic 67/0/33, BD1 0/100/0, α-β 8/92/0, Kalman 0/92/8.
   `matrix_inversion` common: analytic 83/17/0, BD1 0/42/58, BD5 58/42/0. The high-order advantage
   exists only with order-matched information (analytic, BD5, partly BD2/S-G) and small σ. At
   σ = 1e-2 and common h, low order wins (or ties within 3 %) at τ ≤ 1e-2 in every regime; under
   own h_opt the smaller h of the q = 6 formulas filters more noise and they win with analytic and
   α-β information (Euler/best ratio 3.7 and 1.5), which is the order/bandwidth confound of §4, not
   an order effect. The raw "lowest median" winner share is
   fragile under near-ties (α-β: 25 % in the superseded 2-seed run, 83 % now, but 0 % significant
   advantage): only the CI-based shares are conclusions.
5. **Censoring.** `uhlig_4_6` diverges at τ = 0.1 (all trajectories) and in 72–78 of 126 at
   τ = 3e-2, even with exact information and σ = 0 at h = 0.1 or 0.093 (inside its constant-
   coefficient interval, h_max = 0.176): over a 10-instant stencil spanning 0.3–1 s, A(t) changes
   too much. `zead_7i_cang2024` (own h_opt) diverges in 37/114 at τ = 0.1. No failed runs.
6. **Pseudo-replication / leakage.** Within-seed spread of the B states: median 3.7e-12; at
   τ ≤ 1e-2 at most 5e-3, and only for residuals ~1e-11 (round-off); at τ = 0.1 median 8e-3, i.e.
   the τ = 0.1 cells carry an x0-dependent part of ~1 % of MSSRE (90 steps before the window). The
   seed is the statistical unit; τ = 0.1 never enters an asymptotic fit.
7. **Kendall τ_b** vs (analytic, σ = 0) at τ = 3e-3 (descriptive): +0.6 to +1.0 at σ = 0, −1.0 to
   −0.8 at σ = 1e-2 in every regime (complete reversal of the order ranking).


## 7. Full-scale cost estimate and commands

Run every command from `znn-bench/`. Scale selection: `--scale` when running the config as a
script, or `E6_SCALE` / `ZNN_SCALE` (`full` is the default) when the runner imports it.

```bash
# build (either form gives identical task ids; the first also writes results/E6/e6_meta.json)
python experiments/configs/E6.py --scale full --out results/E6
E6_SCALE=full python -m znnbench.runner build experiments/configs/E6.py --out results/E6

# launch under the memory guard (resumable; re-run the same command after a kill)
python tools/memguard.py pool --tasks results/E6/tasks.jsonl --workers 16 --max-gb 200

# analyze
python experiments/analyze_E6.py --results results/E6 --figdir figures --tabdir results/tables

# mini (what was run here)
python experiments/configs/E6.py --scale mini --out results/mini/E6
python tools/memguard.py pool --tasks results/mini/E6/tasks.jsonl --workers 3 --max-gb 40
python experiments/analyze_E6.py --results results/mini/E6 --figdir figures/mini --tabdir results/tables/mini
```

### 7.1 Full-scale cost (re-estimated 2026-10-18 from the mini rerun)

| item | value |
|---|---|
| specs / trajectories | 33,075 / 132,300 (`linear_system` 26,950; `matrix_inversion` 6,125); `--dry-run` matches the runner's `expand` |
| steps | 8.397e8 (τ = 1e-4 is 67.5 %, τ = 3e-4 22.5 %) |
| tasks | 827 chunks of 40 specs (cost-balanced), mean ≈ 158 s, longest spec ≈ 25 s (α-β/Kalman at τ = 1e-4) |
| measured cost | 1.28–1.33e-4 s/step at B = 2, independent of τ, the same at load 18 and 36; ×1.11 for B = 4 (runner's 1 + B/16 model); per-spec setup 0.3 ms; per-task overhead 2 s (1.6 s launch/imports + 0.4 s in-task I/O, measured) |
| core-hours | ≈ 36 (35.8 simulation + 0.5 overhead); the untrimmed 53,900-spec design would be ≈ 58 (the SUMMARY's 56 omitted the task overhead and is otherwise realistic) |
| wall | ≈ 2.3 h on 16 workers, ≈ 1.5 h on 24 (performance cores only; the M3 Ultra has 24 P + 8 E cores, and the load average was 18–36 from other jobs on 2026-10-18) |
| memory | peak RSS 127 MB per task measured (≤ 0.2 GB expected: the largest per-run arrays are ~8 MB at τ = 1e-4, B = 4); 16 workers ≈ 2–3 GB, far below the 200 GB cap |
| recommended | `--workers 16` (24 only if the machine is otherwise idle) |

Exact commands (from `znn-bench/`):

```bash
python experiments/configs/E6.py --scale full --out results/E6
python tools/memguard.py pool --tasks results/E6/tasks.jsonl --workers 16 --max-gb 200
python experiments/analyze_E6.py --results results/E6 --figdir figures --tabdir results/tables
```

The build prints `planned: 33075 specs, 8.3973e+08 steps x B=4` and writes 827 tasks plus
`results/E6/e6_meta.json`. The analysis needs ≈ 1–2 min (≈ 3900 cells, 2000 cluster-bootstrap
resamples each).

## 8. Figures and tables produced

* `fig_E6_regime_<problem>_<protocol>`: the regime map. One panel per information regime, rows σ
  (largest at the top) and columns τ. The cell colour is the best formula, in the canonical
  `keys.style_in_group('disc')` colour, with a text label for print and CVD readers; hatching marks
  a non-significant win (white legend swatch, so it cannot be read as the grey ZeaD-11i), dots a
  cell whose significance is not assessable (< 3 seeds; never at full scale).
* `fig_E6_advantage_<problem>_<protocol>`: A = log10(MSSRE_low/MSSRE_high) on `style.CMAP_DIV`,
  centred at 0, with the values printed in the cells; values in parentheses have a selection-aware
  95 % CI containing 0.
* `fig_E6_mssre_tau_A_*` (analytic, BD1, BD2) and `fig_E6_mssre_tau_B_*` (S-G, α-β, Kalman, BD5):
  MSSRE against τ, one panel per σ, as the median with an IQR band. A thin line marks the noise
  level σ; an open marker on the top edge marks a cell whose median diverged (censored).
* `fig_E6_amplification`: G = MSSRE/σ at the smallest τ and the largest σ, for each formula
  (x axis) and information regime (series, `keys` 'info' styles). Open markers are cells that are
  not noise-dominated.
* `fig_E6_order_collapse`: the empirical order at σ = 0 for each formula and information regime,
  with black bars at the nominal order q and cluster-bootstrap CI whiskers.
* `fig_E6_traces`: the residual against time (median over B, seed 0) at τ = 1e-3 and σ ≈ 1e-6 for
  analytic, BD2 and BD5.
* Tables: `E6_orders.tex`, `E6_amplification.tex`, `E6_regime.tex`, `E6_headline.tex`
  (Euler/best-high ratios with CIs) and `E6_kendall.tex`.
* `<results>/E6_cells.csv` holds every cell's median, IQR, CI, n, n_div and mean SSRE;
  `summary.json` and `SUMMARY.md` hold the headline numbers.

Style: the fixed categorical slots of `keys.GROUPS` are used, with marker and line style as a
secondary encoding. Two local rules (in `analyze_E6.disc_style` / `info_style`, keys.py untouched):
a formula outside `GROUPS['disc']` (`zead_11i_chen2023`) keeps the grey "other" colour but gets a
hexagon marker and long dashes (keys' `.`/`:` was hard to tell from the dotted canonical slots of
ZeaD-6i and Uhlig-4/6); `bd5`, outside `GROUPS['info']` (7 members), takes the group's free 8th
slot (red, `*`), exactly the style it would get if appended to the group. Each panel shows at most 7 series, with one legend per figure. There are no
dual axes. The sequential map is not needed; the advantage map uses `CMAP_DIV`. Widths are
`style.DOUBLE`, and figures are saved with `style.savefig` after setting `style.FIG_DIR = --figdir`.

## 9. Zoo and protocol observations (for the orchestrator)

No zoo bug was found: every one of the 5250 + mini rows has status `ok` or `diverged`, with no
`error`, `timeout` or `skipped`. Observations:

1. **B is redundant for steady-state metrics of linear ZNN (protocol issue, not a code bug).**
   `simulate` draws measurement noise from one stream (`rng_data`) shared by the whole batch, and
   `EstimatedDerivativeProblem` hash-seeds its noise by (seed, t). Every one of the B trajectories
   of a run therefore sees identical data. Since the steady state of linear ZNN does not depend on
   x0, MSSRE, mean SSRE and the floors are identical across B to about 1e-14 relative. A bootstrap
   over the 3 × 4 = 12 "trajectories" then has effectively 3 clusters and understates the
   uncertainty. E1, E5 and E9 face the same issue if they bootstrap trajectories. Suggestion:
   bootstrap over seeds, and use more seeds where CIs are headline numbers. This is a property of
   the protocol; the code behaves as specified.
2. **`zead_11i_chen2023` is not in `keys.GROUPS['disc']`**, so it is drawn in grey (E6 gives it a
   hexagon marker and long dashes locally, §8). The 8-slot group currently holds `zead_5i`,
   `zead_7i_cang2024` and `uhlig_3_4`. Suggestion for keys.py (not edited here): swap
   `zead_11i_chen2023` in for `zead_5i` or `uhlig_3_4` (`uhlig_3_4` is unusable at h = 0.1), and
   append `bd5` to `GROUPS['info']` (its free 8th slot, the style E6 already uses). E6 picks both up
   without changes.
3. **`zead_7i_cang2024`:** h_max = 0.116 and ρ(h = 0.1) = 0.981. Any experiment that runs it at
   h = 0.1 sits at its stability edge.
4. **Constant-coefficient stability is not enough at large τ (physics, not a bug).** `uhlig_4_6`
   diverges at τ ≥ 3e-2 with exact information and σ = 0 although h = 0.1 / 0.093 lies inside its
   linear-ZNN interval (h_max = 0.176); ZeaD-11i is pre-asymptotic there (local slope 14 at
   τ = 0.1 → 3e-2). The interval of docs/spec/discretizations.md assumes a constant J; over a
   10–11-instant stencil spanning 0.3–1 s the 1 rad/s problem changes too much. E2 maps stability at
   τ = 1e-2, where this does not bite; results at larger τ must carry this caveat.
5. **Order-fit floor.** The generic 1e-11 floor of docs/spec/discretizations.md removes the
   asymptotic points of q = 6 formulas on these n = 4 problems (Uhlig-4/6: 7.7e-12 at τ = 3e-3,
   measured round-off ≈ 3e-15). E6 uses 1e-12; E1 may want a measured floor too.

## 10. Review log (2026-10-18)

Reviewer pass after the preparer was cut off (no preparer report). Each item: severity, change,
effect on the numbers. Mini rerun after all changes: §6.1.

1. **Critical: absurd order fits (checklist 3).** `asymptotic_order` dropped "flattening tails"
   relative to the LARGEST local slope and preferred the longest window with a loose
   max(0.35, 12 %) tolerance. A steep pre-asymptotic or unstable segment at τ = 0.1 (ZeaD-11i
   local slope 14–16) therefore removed the asymptotic points, and 2-point slopes of 12.2–15.6
   (ZeaD-11i, every regime) and 16–20 (ZeaD-7i, own h_opt) were reported. *Change:* the new rule
   uses the trailing small-τ run, grows the window from the two smallest τ while every local slope
   stays within 0.15 of the LS slope, reports the point count, flags 2-point fits and slopes
   outside [0, q+3]. *Effect:* ZeaD-11i analytic 15.5 → 6.08; BD1 12.2 → 1.93; BD2 12.4 → 2.90;
   α-β 15.6 → 2.01; ZeaD-7i (h_opt, analytic) 16.5 → 4.93 (2pt). The "order cap" medians of the
   q = 6 formulas are now analytic 6.0, BD1/α-β/Kalman ≈ 2.0, BD2/S-G ≈ 2.9 (before: 7–11).
2. **Major: floor 1e-11 too high for q = 6.** With 3 seeds, Uhlig-4/6 (analytic, BD5) had a single
   τ above 1e-11 and no order estimate. *Change:* floor 1e-12 (measured round-off ≈ 3e-15 from the
   within-seed spread). *Effect:* Uhlig-4/6 analytic 5.96 and BD5 5.88 (both 2pt).
3. **Major: regime summary ignored significance.** "Share of cells won by an order ≥ 4 formula"
   counted the lowest median even in near-ties; α-β went from 25 % (2-seed run) to 83 % (3-seed
   run) with 25 % of winners significant. *Change:* selection-aware paired seed-cluster bootstrap
   CI of the advantage A (both minima re-taken per resample); SUMMARY, `E6_regime.tex` and the
   advantage maps (parentheses = CI contains 0) use high / low / n.s. shares. *Effect:* common h:
   α-β 0/17/83, Kalman 0/25/75, BD1 0/33/67 (no significant high-order advantage anywhere);
   analytic 83/17/0, BD5 58/42/0.
4. **Major: CIs with < 3 seed clusters (checklist 5).** The bootstrap was already a seed-cluster
   bootstrap, but 1 seed (`matrix_inversion` mini) printed "[--, --]" and set every regime cell to
   "not significant" (0 %); 2 seeds give 3 distinct resamples. *Change:* CIs need ≥ 3 seeds
   (else "n/a (k seeds)", significance None, dotted cells, never counted as n.s.); the mini runs 3
   seeds for both problems; a within-seed x0-spread diagnostic is in SUMMARY (median 3.7e-12:
   B states are replicas). *Effect:* `matrix_inversion` now has CIs, e.g. analytic σ = 0 ratio
   1.55e8 [1.55e8, 3.33e8], and 75 % significant winners (analytic) instead of 0 %.
5. **Moderate: censoring (checklist 2).** A cell's CI was dropped whenever one trajectory diverged;
   inf/inf resamples became NaN and silently left the bootstrap; τ* dropped ±inf advantages.
   *Change:* diverged = +inf = BIG (worst case, two diverged tie) inside medians, IQRs, CIs,
   ratios and crossovers; failed runs counted (n_fail) and excluded; censoring line and per-
   (formula, τ) divergence counts in SUMMARY; censored medians drawn on the top edge of the
   MSSRE-vs-τ panels. *Effect:* no point estimate changed (0 failed runs, medians were already
   inf); CIs now exist for partially diverged cells (Uhlig-4/6 at τ = 3e-2).
6. **Moderate: noise-floor slope contaminated.** The 5× rule allows a 20 % truncation share
   (slope bias up to 0.2 q) and included unstable large-τ cells. *Change:* 20× dominance and
   τ ≤ τ_asym (data-driven, 1e-2 here); the all-τ value is reported too. *Effect:* median
   +0.15 → +0.02 (asymptotic) / +0.15 (all τ); the σ/τ hypothesis (−1) stays rejected.
7. **Minor: figure style (checklist 7).** ZeaD-11i ('other': grey `.`/`:`) was hard to tell from
   the dotted canonical slots of ZeaD-6i and Uhlig-4/6; the regime-map "hatched" swatch was grey
   (= ZeaD-11i); `bd5` was grey; the advantage title listed ZeaD-7i in the common-h maps.
   *Change:* §8 style rules (grey kept, hexagon + long dashes; white hatched swatch; `bd5` in the
   free 8th info slot); title lists the protocol's formulas. keys.py change suggested in §9.
8. **Design / cost (checklist 8).** The 56 core-hours of the 18:00 SUMMARY are realistic for the
   untrimmed design (wall per step 1.3e-4 s, independent of τ and of machine load; only the 2 s
   per-task overhead was missing → ≈ 58). *Change:* `matrix_inversion` at the common h only,
   seeds 0–4 (§5 item 7). *Effect:* 53,900 → 33,075 specs, 1.37e9 → 8.40e8 steps, ≈ 58 → ≈ 36
   core-hours; no conclusion uses the dropped cells.
9. **Checked, no change: noise injection.** `simulate._perturb` adds N(0, σ²) to every data array
   except time stamps (linear_system: A, b; matrix_inversion: A), once per step; BD1/BD2 reuse the
   stored noisy samples (`past`), and `analytic` keeps the exact Ȧ, ḃ. The wrapper's `_measure`
   adds N(0, meas_std²) to the same arrays, hash-seeded per (seed, t), and feeds the same sample to
   E, `solve_J` and the estimator. All 4032 mini specs route estimators with `info='analytic'`,
   `data_noise=0`, wrapper seed = run seed. σ is therefore the same physical quantity (per-entry
   std on A and b) in every regime, and all formulas of a cell see the same noise (common random
   numbers, so pairing is valid).
10. **Checked, no change: H2-gain theory.** P_h(z) = ρ(z) + h z^{N−2} (`zead.lookahead_znn_poly`) with
    forcing −(h + D(z)) n (D = dimensionless estimator FIR, α-β/Kalman weights at unit spacing as in
    the wrapper) is the correct linear error recursion for noise on E; measured G follows it with
    slope 1.01–1.03 and R² ≥ 0.996. κ absorbs the max-vs-RMS factor and the ‖x‖-dependence of the
    E-noise and is common to all formulas of a problem.
11. **Checked, documented: confounds (checklist 6).** ZNN-Bench h = γτ with tau-form coefficients
    (ρ'(1) = 1) for every formula, so common h equalises the principal root and no c₀ rescaling is
    involved (papers' h_paper = γτ/c₀ differ); own-h_opt confounds order with noise bandwidth;
    'analytic' is an oracle; τ = 0.1 cells carry ~1 % x0-dependent leakage; linear activation only,
    so no unit dependence (checklist 1 does not apply). All are listed in SUMMARY "Confounds".
12. **Open (not fixed, minor).** τ* uses the point estimate of A (its sign is noise in n.s. cells;
    read τ* together with the parentheses of the advantage map). The q = 6 orders with analytic or
    BD5 information stay 2-point at full scale (only τ = 1e-2 and 3e-3 lie between the unstable
    large-τ range and the floor); E1 verifies them on its own grid. The per-formula floor slopes of
    the mini (Uhlig-4/6 +0.56) are 2-point fits; the full grid gives 5 τ ≤ 1e-2.

Zoo bugs: none found (all runs `ok` or `diverged`, 0 `error`/`timeout`/`skipped`). Observations
for the lead are §9 items 2, 4 and 5 (keys.py groups; constant-coefficient stability at large τ;
generic order-fit floor).
