# E1: order verification (notes)

**Question.** Which residual order does each discretization actually deliver for linear ZNN, in the DTZNN
convention (fixed h = γτ) and in the continuous-time convention (fixed γ)?

**Files**

| file | role |
|---|---|
| `experiments/configs/E1.py` | sweep definition. It builds one runner config per formula (`get_configs()`), and also works as a script for build, a merged pool file, `--list`, `--count` and `--estimate-cost` |
| `experiments/tasks/E1_problems.py` | runner hooks used through import paths: `ConstantMatrix` (constant-A control), `TimeShift` / `TimeShiftLate` (every run ends at the same absolute time), `window_metrics` (extra metric `mssre_w`) |
| `experiments/analyze_E1.py` | analysis. Writes figures, LaTeX tables, `summary.json`, `SUMMARY.md` and `E1_slopes.csv` |
| `results/mini/E1/` | mini results: `summary.json`, `SUMMARY.md`, `E1_slopes.csv`, `full_cost_estimate.json`, one sub-folder per formula |
| `figures/mini/fig_E1_*.{pdf,png}`, `results/tables/mini/E1_*.tex` | mini figures and tables |

Nothing under `znnbench/` or `tests/` was edited.

## 1. Protocol (as implemented)

* **Model.** `znn` with the linear activation (linear ZNN), exact time-derivative information (`info=analytic`), no
  noise.
* **Formulas.** Included are every registered causal look-ahead or multistep formula, RK4, and the implicit and
  consistent variants: 34 configs at build time (2026-10-18).
  * Look-ahead and multistep: `euler`, `taylor_zhang`, `zead_4i_general`, `zead_4i_6321`, `zead_4i_fifd`,
    `zead_5i`, `zead_5i_general(a1=0.1)`, `zead_6i_guo2017`, `zead_6i_qiu2019`, `zead_7i_general`,
    `zead_8i_wu2020`, `zead_8i_chen2019`, `zead_10i_wu2020`, `zead_11i_chen2023`, `uhlig_2_3`, `uhlig_3_4`,
    `uhlig_4_6`, `ab2`, `ab3`, `ab4`.
  * Reference: `rk4` (non-causal).
  * Implicit and consistent: `implicit_euler_err`, `peb_euler`, `exact_flow_euler`, `semi_implicit_euler`,
    `implicit_taylor_zhang`, `implicit_taylor_zhang_naive`, `peb_taylor_zhang`, `exact_flow_taylor_zhang`,
    `semi_implicit_taylor_zhang`.
  * **Picked up conditionally.** A key is included only if it is in `znnbench.available()`. The build includes
    `OPTIONAL = [zead_7i_cang2024, uhlig_5_5, uhlig_5_6, uhlig_5_6b, uhlig_5_7, uhlig_6_8]`. It also includes,
    through auto-discovery, every other new key whose discretizer is `causal` with an `expected_order`. This added
    `zead_5i_mahuang2025`, `cdtznn_mahuang2025` and `broyden_dtznn_zhang2012`. Re-running the build picks up
    later zoo additions; `uhlig_high.py` currently registers nothing.
  * **Excluded on purpose.** The `consistent` factory is left out because its registered combinations are listed
    above. So are the prediction–correction and tracking baselines (`pc_*`, `newton_tracking*`,
    `running_gradient`): they are not look-ahead ZNN discretizations, and E9 covers them.
* **Problem instances** (n = 4 where applicable, κ = 10, freq 1 rad/s):
  * `linsys_n4`, `matinv_n4` and `matinv_n16`: matrix inversion with n = 16, i.e. 256 states.
  * `sylv_n4`.
  * `tvqp_eq`: variant `zhang2021`, with data frequencies up to 4 rad/s.
  * `tvne_guo2018`: `nonlinear_equations`.
  * Two `linear_system` controls:
    * `linsys_n4_f0.2`: freq = 0.2.
    * `linsys_n4+constA`: A frozen at A(0), while b(t) keeps its time variation. This is the `ConstantMatrix`
      wrapper.
* **Step-size modes.**
  * **fixed-h**, h ∈ {h_rec, 0.1}. 0.1 is used only if it lies in the formula's stable range, so it is dropped for
    `uhlig_3_4` (h_max = 0.054). The two h values share one simulate call (`batch_gains`).
  * **fixed-γ**, with γ = 10.
* **h_rec (recommended / optimal h)**, all in the ZNN-Bench raw convention h = γτ:
  * ZeaD and Uhlig formulas: `describe()['h_opt']`, which is argmin of the spectral radius of P_h (the criterion
    of Zhang et al., TNNLS 2019).
  * Euler: 1. Taylor–Zhang: 0.2401, which is h_o = 0.24 of the literature.
  * AB2–AB4: computed h_opt.
  * Implicit and consistent variants: the h_opt of their explicit base formula. Euler-based: 1; TZ-based: 0.2401.
    No optimum is stated for the implicit variants.
  * `rk4`: 1.596 = argmin_h |R(−h)|.
  * `cdtznn_mahuang2025`: 0.0465, the h_opt of its x-recursion `zead_5i_mahuang2025`.
  * `broyden_dtznn_zhang2012`: none stated, so only h = 0.1 (its source, Ma & Huang 2025 Table 1, uses
    h ∈ {0.1, 0.15, 0.2} with c0 = 1, i.e. the same h in both conventions).
  * **Step-size convention (review 2026-10-18).** Every h of E1 is the raw product h = γτ (tau-form coefficients,
    `docs/spec/discretizations.md`). Ma & Huang 2025 normalise by the leading coefficient, h_paper = γτ/c0, so
    h_ours = c0·h_paper (`docs/spec/inverse_free_dtznn.md`). The catalogue in each `config.json` now records
    `c0`, `paper_h_factor`, `h_rec_paper` and `h_common_paper`, and SUMMARY.md prints the conversion:

    | key | c0 | h_rec (h_paper) | h = 0.1 is h_paper = |
    |---|---|---|---|
    | `zead_5i` | 4/9 | 0.0742 (0.167) | 0.225 |
    | `zead_6i_guo2017` | 1/2 | 0.158 (0.316) | 0.2 |
    | `zead_7i_cang2024` | 55/138 | 0.0722 (0.181) | 0.251 |
    | `zead_5i_mahuang2025`, `cdtznn_mahuang2025` | 6/13 | 0.0465 (0.101) | 0.217 |
    | `broyden_dtznn_zhang2012`, `euler` | 1 | -- / 1 | 0.1 |

    All Table 1 step sizes of the source (h_paper ≤ 0.2) are inside the formulas' stable intervals; the
    runs never use h_paper as h.
  * Run `python experiments/configs/E1.py --list` for the complete table (it prints the c0 conversion).
* **τ grid.** Full scale: {1e-1, 5e-2, 3e-2, 2e-2, 1e-2, 5e-3, 3e-3, 2e-3, 1e-3, 5e-4, 3e-4, 2e-4, 1e-4}.
  Mini: 1e-1 … 1e-3 (9 points).
* **Statistics.** 3 problem seeds × B = 4 initial states. Mini: 2 × 4. For `tvqp_eq` and `tvne_guo2018` the
  instance is deterministic, so the seeds vary only x0.
* **Horizon.**
  * Each (formula, τ, step-size mode) gets T = max(6 s, 1.5 s + 37 τ / r). ρ is the spectral radius of the
    formula's frozen-coefficient linear-ZNN error recursion at the run's h (largest requirement over the
    batched gain entries) and r = −ln ρ.
    * ρ is analytic for look-ahead formulas, AB and RK4, and measured for the implicit and inverse-free
      schemes, on `linear_system(freq=0)` with τ = 1.
    * **Coupling allowance (fixed h only):** r is reduced to max(r − τ·L_REF, r/4) with L_REF = 3 s⁻¹ (a
      typical ‖A⁻¹Ȧ‖ at 1 rad/s), because the τ‖A⁻¹Ȧ‖ coupling slows the decay of high-order formulas at small h.
      With the frozen rate alone, `uhlig_4_6` at τ = 1e-2 and `zead_5i`/`zead_10i`/`zead_11i` at τ = 3e-2
      kept x0-dependent transients in the window.
  * T is rounded up to a ladder and capped at 200 s. `T_capped` (in `config.json`) lists the τ whose
    frozen-rate requirement exceeds the cap; the analysis excludes them.
  * **Leak diagnostic.** `window_metrics` also records `x0_spread_w`: the max over the window of |r_b(t) −
    median_b r_b(t)| over the trajectories of one gain. For linear ZNN the steady state does not depend on x0,
    so a spread > max(1 % of the MSSRE, 2× the floor) means the transient has not decayed; such points are
    "unsettled" and excluded. Not applied to `broyden_dtznn_zhang2012`, whose steady state depends on x0 by
    construction.
  * For example, `uhlig_3_4` at fixed h_rec and τ = 0.1 needs T = 200 s; every τ ≤ 2e-3 uses T = 6 s.
* **Steady window.**
  * `TimeShift` shifts the data clock so that every run ends at the same absolute problem time: t_end = 6 s,
    or 200 s (`TimeShiftLate`) for `tvne_guo2018`, whose data are undefined for t ≤ −1.
  * The metric is `mssre_w`, the max ‖E‖ over the absolute window [t_end − 1.5 s, t_end]. That is the PLAN's
    window (last 25 % of 6 s), applied to the same data segment for every τ.
  * The runner's own `mssre` (last 25 % of T) is kept in the parquet for reference.
* **Analysis** (`analyze_E1.py`, as revised on 2026-10-18):
  1. **Curve.** One curve per (formula, instance, mode): the median and IQR of `mssre_w` over the trajectories.
     Diverged trajectories are +inf (censored median: a point is unusable once half of its trajectories
     diverged); error / timeout rows are missing.
  2. **Unusable points.** Infinite median; not converging (median MSSRE ≥ 0.5 × the transient peak
     `max_residual`); unsettled (x0 spread, see Horizon); capped horizon.
  3. **Round-off floor.** The flat tail at the finest τ (local slope < 1 or within 3× of the floor, level <
     1e-10), measured per curve. Points within 30× of the curve's floor are excluded; a curve without a floor
     uses the instance median (default 1e-14). Measured floors: ~6e-15–9e-15 on the unit-scale instances,
     8e-13 on `tvne_guo2018` (|x₄| ≈ 198 at t ≈ 200 s), so the cut is ~2e-13–3e-13 and order-6 formulas
     still get estimates (the 1e-11 floor problem of E6 does not arise).
  4. **Asymptotic range** (consecutive-ratio diagnostic). The finest contiguous run of ≥ 2 valid points;
     starting at its finest point, the window grows towards coarse τ while every local slope s_i =
     log(M_{i+1}/M_i)/log(τ_{i+1}/τ_i) stays within **max(0.15, 0.03·|median|)** of the window's median (was
     0.2/0.04). Unstable and non-converging points are removed first, so a steep or unstable coarse-τ segment
     can never displace the small-τ points. With fewer than 3 consistent points the 2-point local slope is
     reported and flagged; a window whose local slopes still change monotonically by > 0.15 is flagged
     "drift"; a slope > expected + 3 or < −0.5 is flagged "absurd" (none in the mini).
  5. **Fits.** OLS slope over that window, with a two-stage cluster bootstrap 95 % CI (2000 resamples of the
     problem seeds, then of the x0 within each seed; seed 20261018, per curve). The "naive" OLS slope over
     all valid points is also reported.
  6. **Expected order and verdict.**
     * fixed-h: claimed (q+1). fixed-γ: claimed − 1.
     * RK4: 2 at fixed h (the Prothero–Robinson prediction, see `docs/spec/rk4_order_reduction.md`), 4 at
       fixed γ.
     * "match": |slope − expected| ≤ 0.3, or a CI of width ≤ 0.6 covers it. "pre-asymptotic": not a match,
       drifting (or 2-point), and the finest local slope is closer to the expected order than the fit
       (inconclusive: the grid did not reach the asymptotic range). Otherwise "lower" / "higher".
     * Headline fractions use the main instances and exclude `rk4` (non-causal reference, reported
       separately) and `broyden_dtznn_zhang2012` (reported separately: it does not track time-varying-matrix
       problems, `docs/spec/inverse_free_dtznn.md` E6; its curves with min MSSRE ≥ 1e-3 get the verdict
       "no tracking").
  7. **Kendall τ_b** (descriptive; Holm-adjusted p-values within each family) between the claimed order and
     (a) the measured slope, (b) the accuracy −log MSSRE at τ = 1e-2 and 1e-3, at fixed h_rec **and at the
     common h = 0.1**. In (b) a diverged point, or a formula not run at h = 0.1 because 0.1 is outside its
     stable range, is the worst rank (tied); it no longer drops out.
  8. **Other statistics:** paired bootstrap for the RK4 fixed-γ − fixed-h difference; the onset of the
     asymptotic range as τ*·L/h and ωτ*/h, where L = max_t‖J⁻¹J̇‖.

## 2. Deviations from PLAN.md, with reasons

1. **Finer τ grid** (13 points, 1-2-3-5 spacing, instead of 7). For orders 5 and 6 the asymptotic range is about
   0.7–0.9 decade wide (mini), so the PLAN's 1-3 grid gives at most 2–3 points in it. The extra points are all at
   coarse τ, so they cost little (+19 % steps).
2. **Adaptive horizon, time shift and absolute window** instead of T = 6 s with the last 25 %.
   * **Why T must grow.** At fixed h the initial transient decays like exp(−h t/τ). At coarse τ and small h_rec,
     T = 6 s leaves the window inside the transient. For example, `uhlig_3_4` at τ = 0.1 still carries e^-1 of the
     initial error at t = 4.5 s.
   * **Why the window must be absolute.** The first mini run used T(τ) with the runner's window, the last 25 % of
     T. The maximum of a quasi-periodic error over a longer or shifted window is biased: `uhlig_3_4` at τ = 5e-3
     (T = 80 s) came out about 1.5× too high and τ = 3e-3 (T = 8 s) 0.8× too low, so the local slopes read 4.2, 6.1, 4.2.
   * **With the time shift and the absolute window** the local slopes are smooth: 4.97, 4.92, 4.82, … (see
     `fig_E1_local_slopes`).
   * **`tvne_guo2018` uses t_end = 200 s.** Its 1/(t+1) terms have nearly vanished there, so the instance is
     effectively the sin t / linear-in-t part of the system.
3. **Fixed-γ uses only γ = 10.** γ = 1 would widen the high-order window, but its transient exp(−t) needs
   T ≈ 40 s, i.e. about 7× the cost at τ = 1e-4.
4. **Control instances.** Two controls were added: freq 0.2, and constant A through the `ConstantMatrix` wrapper.
   * `linsys_n4_f0.2` is exactly the 1 rad/s instance with τ rescaled by 0.2: A_0.2(t) = A_1(0.2t), and h is
     invariant. It collapses onto the 1 rad/s curve in ωτ (`fig_E1_controls`). It therefore *shifts* but does not
     *widen* the asymptotic range. This is reported as a finding.
   * The constant-A instance removes the τ‖A⁻¹Ȧ‖ coupling and is the informative control.
5. **`zead_5i_general` is run at a1 = 0.1.** Its default a1 = 1/9 is identical to `zead_5i`. `uhlig_2_3`, which is
   also identical to `zead_5i`, is kept as a registry duplicate check.
6. **h = 0.1 is kept only where it is stable.** For implicit variants with no stated optimum, the base formula's
   h_opt is used.
7. **Non-convergence rule.** Finite runs that grow without reaching the 1e10 divergence threshold, such as
   `uhlig_3_4` at τ = 0.03 (MSSRE 9e7), are treated as unstable. Otherwise they would enter naive fits.

## 3. Mini run (validation; re-run after the review, 2026-10-18 18:04–18:10)

* **Scale.** 13 formulas: `euler`, `taylor_zhang`, `zead_5i`, `zead_6i_guo2017`, `zead_7i_cang2024`, `uhlig_3_4`,
  `uhlig_4_6`, `ab4`, `rk4`, `implicit_taylor_zhang` and (added by the review) the inverse-free keys
  `zead_5i_mahuang2025`, `cdtznn_mahuang2025`, `broyden_dtznn_zhang2012`. 6 instances: `linsys_n4`, its freq-0.2
  and constant-A controls, `matinv_n4`, `tvqp_eq`, `tvne_guo2018`. τ from 1e-1 to 1e-3 (9 points), 2 seeds × B = 4.
* **Size and outcome.** Clean rebuild (`results/mini/E1` deleted first): 2808 specs, 15 984 trajectory rows,
  6.14e6 steps. Statuses: ok 13 894, diverged 2090 (all at h or τ outside the stable range); **0 errors,
  0 timeouts**.
  * Pool: `memguard pool --workers 3 --max-gb 40`, 117 tasks, **5 min 53 s wall** (machine shared with other
    sweeps, load ≈ 22 on 32 cores). Simulate wall totalled 708 s = 0.20 core-h (115 µs/step on average,
    median 134 µs); peak RSS 129 MB per task. Analysis: about 12 s.
* **Main instances** = `linsys_n4`, `matinv_n4`, `tvqp_eq`, `tvne_guo2018`; RK4 and Broyden reported separately.

| mode | curves | match, asymptotic fit | match, naive all-points fit | median \|dev\| asym. / naive | verdicts |
|---|---|---|---|---|---|
| fixed h = h_rec | 44 | **95 %** | 75 % | 0.04 / 0.09 | 42 match, 2 pre-asymptotic (13 drift-flagged) |
| fixed h = 0.1 | 40 | **100 %** | 78 % | 0.04 / 0.09 | 40 match (8 drift-flagged) |
| fixed γ = 10 | 44 | **98 %** | 93 % | 0.002 / 0.003 | 43 match, 1 insufficient |

* **Every formula delivers its claimed order where the grid reaches the asymptotic range** (median over main
  instances, fixed h_rec / fixed γ): `euler` 2.01 / 1.01, `taylor_zhang` 2.97 / 2.00, `implicit_taylor_zhang`
  2.98 / 2.07, `zead_5i` 3.97 / 3.00, `zead_6i_guo2017` 3.98 / 3.00, `zead_5i_mahuang2025` 3.96 / 3.00,
  `cdtznn_mahuang2025` 4.02 / 3.00, `zead_7i_cang2024` 4.94 / 4.00, `uhlig_3_4` 4.86 / 4.00, `ab4` 4.97 / 4.00,
  `uhlig_4_6` 5.89 / 5.00. **One order is lost at fixed γ**, as hypothesised.
* **Not matching (inconclusive):** `uhlig_3_4` 4.39 and `zead_5i_mahuang2025` 3.64 on `tvqp_eq` (data up to
  4 rad/s) at fixed h_rec, both "pre-asymptotic": the local slopes still rise at τ = 1e-3. `uhlig_4_6` at fixed γ
  on `tvne_guo2018`: no point between the stability limit τ < h_max/γ and the floor (8e-13). The full grid
  (to 1e-4) should resolve the first two.
* **Inverse-free DTZNNs (new keys).**
  * `cdtznn_mahuang2025` delivers the order of its x-recursion (4.0–4.1 at fixed h, 3.00 at fixed γ) wherever it
    converges, but its stable τ range is narrower than that of the same formula with the exact inverse
    (`zead_5i_mahuang2025`): at h_rec it diverges for τ ≥ 2e-2 on `linsys_n4` / `matinv_n4` (exact inverse:
    τ ≥ 3e-2), at h = 0.1 for τ ≥ 1e-2 (exact inverse: ≥ 2e-2), and at fixed γ = 10, τ = 1e-2 one of the two
    instances does not converge (MSSRE 5.6e-2 vs 8.6e-6). With constant A all its points are stable.
  * `broyden_dtznn_zhang2012` (h = 0.1 only): slope 1.98 on `tvne_guo2018` (its claimed O(τ²) holds there),
    2.00 with constant A and 1.96 at freq 0.2, but **no tracking** at 1 rad/s on `linsys_n4`, `matinv_n4`,
    `tvqp_eq` (min MSSRE 1.2e-2–2.4e-2 at every τ), as `docs/spec/inverse_free_dtznn.md` E6 states. Its steady
    state depends on x0 (without the x0-spread exemption every point but those on `tvne_guo2018` would be
    excluded).
* **RK4 (finding).** Fixed h = h_rec: **2.00** (median over 6 instances); fixed γ = 10: **4.05**; difference
  2.06, paired cluster-bootstrap CI [2.05, 2.06], over instances [2.03, 2.08]. Prothero–Robinson order reduction
  (stage order 1 at fixed z = −h), as in `docs/spec/rk4_order_reduction.md`. At fixed h every ZeaD formula with
  ≥ 4 instants has a higher residual order than RK4.
* **Coupling (finding).** Over the 8 mini formulas of order ≥ 4 at fixed h_rec, the 1 rad/s linear system has
  **20** unstable or non-converging τ points (of 9 per curve); with A frozen there are **0**. Example:
  `uhlig_4_6` (h_rec = 0.093, inside its frozen interval (0, 0.176)) diverges for τ ≥ 2e-2 on `linsys_n4` and
  `matinv_n4` and is stable at every τ with constant A: the frozen-coefficient interval assumes the problem is
  frozen over the formula's 10-instant window (the E6 review saw the same at τ ≥ 3e-2).
  * The asymptotic range starts at **ωτ*/h ≈ 0.26** (median; IQR 0.17–0.39; 41 uncensored curves) and
    **τ*·L/h ≈ 1.1** (IQR 0.42–1.20; 22 curves), L = max_t‖A⁻¹Ȧ‖ = 8.6 for `linsys_n4` / `matinv_n4` (they share
    A(t)), 10.1 for `tvqp_eq`.
  * Constant A removes the instability but not the pre-asymptotic drift (`uhlig_3_4`: local slopes 4.07 → 4.39
    → 4.88 → 4.96 as τ decreases from 0.1 to 1e-3); the governing ratio is ωτ/h.
  * Median fitted width: 2.0 decades for order 2 (= the whole mini grid, censored), 1.7 for order 3, 1.3 for
    order 4, 0.85 for order 5, 0.70 for order 6.
* **Round-off floors** reached in the mini: 6.6e-15 (constant A), 9.0e-15 (freq 0.2), 8.0e-13 (`tvne_guo2018`).
* **Kendall τ_b (descriptive).** Claimed vs measured order: 0.89–0.91 on every instance, both modes (n = 11–12).
  Claimed order vs accuracy at τ = 1e-2: 0.86–0.90 at fixed h_rec (no censored formula), but only **0.54**
  (`linsys_n4`, `matinv_n4`, `tvne_guo2018`) and **0.23** (`tvqp_eq`) at the common h = 0.1 (2 formulas censored
  as worst: `uhlig_3_4` is unstable at h = 0.1, `cdtznn_mahuang2025` diverges at τ = 1e-2). High order buys
  accuracy at the formula's own small h, much less at a common h, where the high-order formulas sit close to
  their stability limits.
* **Figures checked visually** (read as PNG after the fixes): `fig_E1_mssre_fixed_h`, `fig_E1_orders`,
  `fig_E1_local_slopes`, `fig_E1_families_fixed_h`, `fig_E1_controls`. Style: canonical `disc` group for the
  main panels, runtime groups that keep the `disc` slots elsewhere, ≤ 8 series per panel, marker + line style,
  legends, no dual axes, `style.DOUBLE`, `style.savefig`, units on the τ / ωτ axes, τ increasing to the right
  in every E1 figure.

## 4. Full-scale cost estimate and commands (updated 2026-10-18 by the review)

* **Size.** 34 formulas (all registered E1 keys as of 2026-10-18; a later build picks up new causal keys
  automatically), 8 instances, 13 τ, 3 seeds × B = 4: **21 216 specs (runs), 124 800 trajectories, 2.307e8
  steps, 442 tasks** (chunk size 48, balanced per formula: 13 tasks per formula). Checked by a full dry-run
  build into the scratchpad (35 s, 47 MB of chunk files, `max_wall` 7200 s), then deleted.
* **Estimate.** `--estimate-cost` with the probes of the earlier estimate reused (`--reuse-probes`; all 528
  (formula, instance, B·G) combinations present, 300-step single-thread probes) gives **12.4 core-hours** at
  idle (`results/mini/E1/full_cost_estimate.json`).
  * By instance: `matinv_n16` 4.45, `sylv_n4` 2.93, `linsys_n4` 1.22, freq 0.2 1.21, `matinv_n4` 1.12,
    `tvqp_eq` 0.56, `tvne` 0.50, constant A 0.37.
  * By formula: `cdtznn_mahuang2025` 3.21 (dense M update + finite-difference J̇: ~13–16 ms/step on
    `matinv_n16`), `rk4` 0.57, `broyden` 0.40, every other formula 0.25–0.29.
  * Cross-check: the mini measured 115 µs/step on average (cheap instances), consistent with the probes.
  * Under the present load a 300-step re-probe of `cdtznn` × `matinv_n16` ran 1.3× slower (16.2 vs 12.6 ms/step):
    **budget ~16 core-hours**.
* **Recommended `--workers 24`** (PLAN): ≈ 0.5 h wall on an idle machine, ≈ 1 h while the other sweeps run.
  * Longest single spec: `cdtznn` × `matinv_n16` × τ = 1e-4 × fixed h (60 000 steps): 848 s idle, ~16 min under
    load; far below `max_wall` = 7200 s. The 13 `cdtznn` chunks take ~15–20 min each.
* **Peak RSS per task: ≤ 0.3 GB** (mini max 129 MB; a worst-shape probe, `cdtznn` × `matinv_n16` with B·G = 8,
  112 MB; the recorded traces at 60 001 samples × 8 trajectories are ~4 MB per array). 24 workers stay below
  8 GB, far under the 200 GB cap.
* **Rebuild rule.** Run ids hash the config kwargs, not component defaults (§5.1): build after the zoo is frozen,
  or delete `results/E1/E1-<key>/` for any formula whose defaults changed.

**Commands** (run from the repository root `znn-bench/`; the build takes about 35 s):

```bash
cd znn-bench
# build (one sub-folder per formula + merged pool file; prints "442 tasks, 21216 specs")
python experiments/configs/E1.py --scale full --out results/E1
# launch (resumable; re-running skips finished chunks)
python tools/memguard.py pool --tasks results/E1/pool_tasks.jsonl --workers 24 --max-gb 200
# optional: cost estimate next to the results (picked up by summary.json); reuses the probes
OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python experiments/configs/E1.py --scale full \
    --estimate-cost results/E1/full_cost_estimate.json --reuse-probes results/mini/E1/full_cost_estimate.json
# analyze (about 1-2 min at full scale)
python experiments/analyze_E1.py --results results/E1 --figdir figures --tabdir results/tables
```

* **Mini equivalents** (as re-run by the review): `rm -rf results/mini/E1`, then
  `--scale mini --out results/mini/E1` for the build, `--workers 3 --max-gb 40` for the pool, and
  `--results results/mini/E1 --figdir figures/mini --tabdir results/tables/mini` for the analysis.
* **Scale through the runner CLI.** With
  `E1_SCALE=full python -m znnbench.runner build experiments/configs/E1.py --out results/E1` the runner
  writes the same per-formula sub-folders but no merged pool file. In that case, concatenate
  `results/E1/*/tasks.jsonl` for the pool.

## 5. Zoo / runner observations (nothing edited; for the owners)

1. **Run and task ids do not see component defaults** (`znnbench/runner.py`, `hash_payload`). The id hashes the
   config kwargs, not the resolved defaults.
   * During this session the factory default of `cdtznn_mahuang2025` changed from `M0="identity"` to
     `M0="newton_schulz"`. The change is in `znnbench/discretizations/inverse_free_dtznn.py:618`, matching
     `docs/spec/inverse_free_dtznn.md`, ASSUMPTION 2.
   * A rebuild after such a change produces identical run ids and task ids. The pool then counts the old chunks
     as done and `aggregate` reuses stale numbers silently.
   * Suggested fix: include a hash of `disc.describe()` (or of the resolved factory kwargs) in the payload, or bump
     `SPEC_VERSION` when a default changes.
   * Workaround for E1: build the full sweep after the zoo is frozen, or delete `results/E1/E1-<key>/` for any
     formula whose defaults change.
2. **`ProblemWrapper.exact_data` pitfall** (`znnbench/wrappers.py`). `exact_data` returns the innermost problem's
   data and bypasses every wrapper's `data`. `simulate` records residuals with `problem.exact_data(t)` when it
   exists.
   * This is right for measurement wrappers (jitter, delay, estimation).
   * It is wrong for wrappers that redefine the problem itself (constant A, time shift): their residual would be
     recorded against the original problem.
   * `E1_problems.py` overrides `exact_data` and `residual` for this reason. The base class should document the
     rule, or offer a flag that distinguishes "measurement" wrappers from "problem" wrappers.
3. **`exclude` rules cannot match wrapper kwargs.** They match keys only, so per-instance wrapper parameters need
   distinct keys. That is why `TimeShiftLate` exists. This is a minor feature request.
4. **Known and already documented, not new.**
   * `broyden_dtznn_zhang2012` does not track time-varying-matrix problems. In the pre-flight at τ = 1e-2,
     h = 0.1, its MSSRE was 4.2 on `linsys_n4` and 4.6 on `matinv_n4`, it diverged on `matinv_n16`, and it was
     1.5e-3 with constant A. `docs/spec/inverse_free_dtznn.md` E6 already states this.
   * `zead_5i_general`'s default and `uhlig_2_3` both duplicate `zead_5i`.
   * `uhlig_high.py` registers nothing: the order-7/8 coefficients were not obtained.
5. **Style key inconsistency (`experiments/keys.py`, not an E1 file; review 2026-10-18).**
   `keys.GROUPS["disc_implicit"]` puts `taylor_zhang` in slot 5 (green, "P" marker) while `keys.GROUPS["disc"]`
   puts it in slot 1 (orange square). Any figure that mixes the two groups shows Taylor–Zhang in two styles
   (it happened in `fig_E1_families_*`), against the rule in the `keys.py` docstring. E1 now draws its implicit
   family with a runtime group that keeps the `disc` slots; the owner of `keys.py` should reorder
   `disc_implicit` so that `euler` and `taylor_zhang` keep slots 0 and 1 (this changes E8's colours).
6. **Not a bug, a property worth stating (CDTZNN).** `cdtznn_mahuang2025` diverges at a smaller τ than
   `zead_5i_mahuang2025` at the same h (§3); the linear-ZNN interval h < 0.1725 of its x-recursion does not
   cover the coupled x/M recursion on time-varying problems.

## 6. Caveats

* **The floor rule, the window tolerance and the 0.3 match tolerance are analysis choices.** They are stated in
  `summary.json["analysis"]`. The naive all-points slope and the finest-pair local slope are always reported next
  to the asymptotic fit, in `E1_slopes.csv` and `E1_orders_detail.tex`.
* **Bootstrap CIs resample trajectories.** In steady state the trajectories of one instance converge to the same
  error, so for the deterministic instances (`tvqp_eq`, `tvne`) the CIs are degenerate. The systematic
  (pre-asymptotic) uncertainty is larger than the sampling uncertainty; compare the asymptotic and naive slopes.
* **Censored onsets.** When the fitted range reaches the coarsest stable τ, the onset ratio is reported as a lower
  bound (≥ in `E1_controls.tex`).
* **Three seeds.** With 3 problem seeds × 4 x0 and x0-independent steady states, the 12-trajectory median is the
  middle instance and the IQR spans the other two. Instance curves are parallel in the asymptotic range, so this
  does not bias the fitted orders; where they cross (pre-asymptotic only) the median curve can kink, which only
  shortens the window. The cluster-bootstrap CIs over 3 instances are descriptive.
* **Same A(t) in two instances.** `linsys_n4` and `matinv_n4` use the same `TVMatrix` (same generator and seed),
  so they share L = 8.6 and their stability limits; they differ in the right-hand side only.

## Review log (2026-10-18)

Reviewer: methodology / statistics review of E1 (this session). Only E1 files were edited (`experiments/configs/E1.py`,
`experiments/analyze_E1.py`, this file, and the outputs under `results/mini/E1`, `figures/mini/fig_E1_*`,
`results/tables/mini/E1_*`). Nothing under `znnbench/` or `tests/` was touched. The mini was rebuilt from scratch and
re-run end to end (build → `memguard pool --workers 3 --max-gb 40`, 117 tasks, 5 min 53 s, 0 failed → analysis).
"Effect" compares with the pre-review analysis on the same new mini data unless stated otherwise.

| # | severity | issue | change | effect on numbers |
|---|---|---|---|---|
| 1 | major | **Stale state.** `E1.py` and `analyze_E1.py` had been changed after these notes were written (coupling allowance `L_REF` in the horizon, `x0_spread_w` / "unsettled" rule, `T_capped`, cluster bootstrap, Holm), and an interrupted mini re-run had left 6 of 10 formulas complete and no `summary.json` / `SUMMARY.md`. The notes described a pipeline that no longer existed. | Clean rebuild and full mini re-run; §1, §3, §4 rewritten for the code as it now is. | All mini numbers refreshed (e.g. fixed h_rec: 95 % of 44 curves match vs "85 % of 27" before; mini steps 6.1e6 vs 3.3e6). |
| 2 | major | **Asymptotic window too lenient.** Tolerance max(0.2, 0.04·\|m\|) around the window median admitted monotone pre-asymptotic drifts of 0.4–0.5 (e.g. `zead_5i` on `tvqp_eq`: local slopes 3.84 → 3.68 → 3.49 fitted as 3.70, verdict "lower"). | Tolerance max(0.15, 0.03·\|m\|) (checklist item 3); "drift" flag; new verdict "pre-asymptotic" (not a match, drifting or 2-point, finest local slope closer to the expected order); "absurd" flag (> q+3 or < −0.5). | 13 of 222 slopes change by > 0.05, all on `tvqp_eq` (median change 0); e.g. `uhlig_4_6`\|tvqp_eq 5.61 → 5.73, `zead_5i`\|tvqp_eq 3.70 → 3.79. Fixed h_rec match 86 % → 95 %; the remaining 2 non-matches are labelled pre-asymptotic instead of "lower". ωτ*/h onset 0.32 → 0.26. 0 absurd slopes. |
| 3 | major | **Censoring in the accuracy ranking** (checklist 2). Kendall "claimed order vs accuracy" dropped every unstable / non-converging / unsettled point, which favoured the fragile high-order formulas; only fixed h_rec (different h per formula) was ranked. | Diverged points and formulas not run at h = 0.1 (unstable there) are the worst rank (ties); measured MSSRE used for every other point; added the common h = 0.1 ranking (removes the 1/h confound); `n_censored_worst` reported; p-values labelled descriptive (checklist 4). | τ = 1e-2: 0.86–0.90 at h_rec (0 censored) but 0.54 / 0.23 (`tvqp_eq`) at the common h = 0.1 (2 censored). The old notes' "0.57 on tvqp_eq at h_rec" does not survive: the new finding is that order buys accuracy at the formula's own small h, much less at a common h. |
| 4 | major | **New inverse-free keys and their h** (task brief). `zead_5i_mahuang2025`, `cdtznn_mahuang2025`, `broyden_dtznn_zhang2012` were auto-included but never run in the mini; the c0-normalised step of their source (h_ours = c0·h_paper) was not documented. | Added to `MINI_FORMULAS`; catalogue records `c0`, `paper_h_factor`, `h_rec_paper`, `h_common_paper`; `--list` and SUMMARY.md print the conversion. All runs use the raw h = γτ (checked: h_rec = 0.0465 = c0 × 0.1009 for the Ma–Huang 5-instant formula, inside h < 0.1725). Conditional pick-up unchanged (registry-driven; `OPTIONAL` + auto-discovery of causal keys with `expected_order`). | New results: `zead_5i_mahuang2025` 3.96 / 3.00, `cdtznn` 4.02 / 3.00, `zead_7i_cang2024` 4.94 / 4.00 (fixed h_rec / fixed γ). Finding: CDTZNN's stable τ range is narrower than the exact-inverse formula's (diverges one grid point earlier). |
| 5 | major | **Broyden mis-handled.** Its steady state depends on x0 by construction, so the x0-spread rule excluded every point except on `tvne_guo2018`; its non-tracking on time-varying-matrix problems would have entered the headline fractions as order failures of a look-ahead formula. | `X0_DEPENDENT` exemption from the spread rule; reported separately (`separate` in `summary.json`, its own SUMMARY paragraph); verdict "no tracking" when min MSSRE ≥ 1e-3. | Broyden now gets 1.98 (`tvne_guo2018`), 2.00 (constant A), 1.96 (freq 0.2); "no tracking" on the three 1 rad/s matrix problems. Headline denominators exclude its 4 curves per mode. |
| 6 | minor | **Crash** (latent): an isolated finest valid point gave an empty run and an `IndexError` in `analyse_curve` (first triggered by the new keys' data). | The fit uses the finest contiguous run of ≥ 2 valid points; "insufficient gap" otherwise. | Analysis runs; no number changed elsewhere. |
| 7 | minor | **Figure style** (checklist 7). τ axes ran in opposite directions in different E1 figures (`_logx` inverted twice; `fig_E1_controls` reset the limits); Taylor–Zhang appeared in two styles inside `fig_E1_families_*` (`keys.GROUPS["disc_implicit"]` slot 5); a legend entry for Broyden, which has no h_rec curve; no units on the local-slope and ωτ axes. | τ increasing to the right everywhere; implicit family drawn with a runtime group that keeps the `disc` slots; legends filtered per mode; units added; "(no h_rec)" label in `fig_E1_orders`. `keys.py` issue reported to its owner (§5.5). | Figures only. |
| 8 | minor | **SUMMARY.md lacked** a censoring line, the confounds, verdict counts and the h convention (checklists 2, 6). | Added: censoring line (diverged / observed trajectories per mode, statuses, rules), confounds paragraph, verdict counts, h-convention table, descriptive label on the Kendall section. | Text only. |
| 9 | minor | **Cost estimate** in §4 predated the per-mode horizons and the coupling allowance. | Re-estimated with reused probes; full dry-run build (442 tasks, 21 216 specs). | 12.4 core-h idle (was 12.5), 2.307e8 steps (was 2.278e8), longest spec 848 s (was 1130 s); budget ~16 core-h under load; peak RSS ≤ 0.3 GB/task. |
| 10 | minor, not changed | 3-seed median = middle instance; cluster CIs over 3 instances; `linsys_n4` and `matinv_n4` share A(t). | Documented in §6. | None. |

**Checklist items** (`REVIEW_CHECKLIST.md`): (1) unit dependence — not applicable (linear activation only; the floor
is measured per curve, so the absolute scale of E does not bias the fits). (2) censoring — fixed (#3, #8); medians
were already censored. (3) asymptotic fits — fixed (#2); the window starts at the finest valid τ after unstable and
non-converging points are removed, so a steep coarse segment cannot displace the small-τ points (the E6 failure
mode), 2-point fits and absurd slopes are flagged, all-points and finest-pair slopes are reported next to the fit.
(4) Kendall — descriptive label, Holm within families (#3). (5) pseudo-replication — the bootstrap was already
two-stage (seeds, then x0); CIs are degenerate for the deterministic instances and descriptive with 3 seeds
(#10). (6) confounds — h vs c0·h (#4), different h at h_rec vs common h (#3), RK4's non-causal information and
4 evaluations per step, transients (adaptive T, absolute window, x0-spread rule), start offsets (every run ends
at the same absolute time): all stated in SUMMARY.md and here. (7) style — fixed (#7). (8) cost and memory — #9.

**Lead's E6 heads-up.** (1) Floor: E1 uses the measured per-curve floor with a 30× exclusion (cut ≈ 2e-13–3e-13 on
unit-scale instances), so order-6 formulas keep their estimates (`uhlig_4_6`: 5.89 median, 6.03 on `linsys_n4`).
(2) Fit rule: see checklist item 3 above; no absurd slope in the mini. (3) `uhlig_4_6` divergence inside its frozen
interval: seen here at τ ≥ 2e-2 (h_rec = 0.093) on the 1 rad/s linear system and matrix inversion, and absent with
constant A — part of the coupling finding (§3).

**Verdict.** E1 is ready for the full sweep: the mini runs cleanly (0 errors / 0 timeouts), every formula that
reaches its asymptotic range delivers q+1 at fixed h and q at fixed γ, and the analysis now separates
inconclusive (pre-asymptotic) from failing curves. Build after the zoo is frozen (run ids ignore component
defaults, §5.1).
