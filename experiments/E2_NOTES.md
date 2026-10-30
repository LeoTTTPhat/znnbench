# E2: stability maps (maximal stable step h = γτ per formula × activation)

Files:
* `experiments/configs/E2.py`: the grid, via `get_config(scale)`. It is not a runner config: there is no `CONFIG` symbol.
* `experiments/tasks/E2_stability.py`: the custom memguard tasks (`build`, `run`, `local`, `status`).
* `experiments/tasks/E2_theory.py`: theory for the linear error recursion, shared by the tasks and the analysis.
* `experiments/analyze_E2.py`: figures, tables, `summary.json` and `SUMMARY.md`.

Mini outputs are in `results/mini/E2/`, `figures/mini/fig_E2_*` and `results/tables/mini/E2_*.tex`.

**Scale.** The scale is set by the environment variable `ZNNBENCH_SCALE=mini|full` (default `full`), or by `--scale mini|full` passed to `E2_stability.py build` or to `configs/E2.py`. The env-var name is the same as in E8.

## 1. Exact protocol

| item | value |
|---|---|
| problems | `linear_system` and `matrix_inversion`, both with n = 4, κ = 10, freq = 1 rad/s, T = 6 s. Seeds are 0, 1, 2 for `linear_system` and 100, 101, 102 for `matrix_inversion` (`seed_offset`, see §3.4). |
| control problem | `linear_system` with A frozen at A(0) (`FrozenA` in the task script): E = A₀x − b(t). Linear activation only, ‖E(0)‖ = 1. |
| model | `znn` (design formula dE/dt = −γΦ(E), exact ∂E/∂t, `info = analytic`, no noise) |
| activations | `keys.GROUPS['act']` with zoo defaults: linear, power_sigmoid (p = 3, ξ = 4), hyperbolic_sine (ξ = 3), sign_bi_power (r = ½), fixed_time, vaf, pvaf1, predefined_exp |
| formulas (full) | euler, taylor_zhang, zead_4i_general, zead_4i_6321, zead_4i_fifd, zead_5i, zead_6i_guo2017, zead_6i_qiu2019, zead_7i_general, zead_7i_cang2024, zead_8i_wu2020, zead_8i_chen2019, zead_10i_wu2020, zead_11i_chen2023, uhlig_3_4, uhlig_4_6, ab2, ab3, ab4, rk4 (non-causal reference), implicit_euler_err (control, unconditionally stable for linear Φ). Also included: every key registered later whose name starts with `zead_` or `uhlig_` (today this adds `zead_5i_mahuang2025`), and the inverse-free keys (`broyden_dtznn_zhang2012`, `cdtznn_mahuang2025`) on BOTH problems (the earlier text said matrix_inversion only; the code always used both). The list is filtered by `znnbench.available()` at build time. Excluded as exact duplicates: uhlig_2_3 and zead_5i_general (both = zead_5i). |
| formulas (mini) | euler, taylor_zhang, zead_5i, zead_5i_mahuang2025 (exact-inverse twin of CDTZNN), zead_7i_cang2024, uhlig_4_6, ab3, rk4, implicit_euler_err; inverse-free broyden_dtznn_zhang2012 and cdtznn_mahuang2025 on both problems |
| sampling | τ = 1e-2, fixed-h (γ = h/τ), T = 6 s (N = 600), steady window = last 25 % |
| initial states | x₀ = x\*(0) + J(0)⁻¹(r u_b), with u_b uniform on the unit sphere of the error space. Because both problems are affine in x, this gives ‖E(x₀,0)‖₂ = r exactly. r ∈ {0.1, 1, 10}, B = 4 directions per seed. The same directions are reused for every r, activation and formula (paired design). |
| statistics | mini: 3 seeds × B = 4 = 12 trajectories per cell; **full: 8 seeds** × B = 4 (review: the B directions of one instance are pseudo-replicates, within-instance range of h_max 0.17 % vs 1.3 % between instances, so the instance is the unit). Cells are summarised by median and IQR. Bootstrap 95 % CIs (2000 resamples, seed 20262) over formulas (× problems) are descriptive; the cluster-level uncertainty is the per-seed [min, max] range reported next to every headline ratio. Kendall τ_b p-values (Holm) are descriptive (n = 7–10 formulas). |
| h search | coarse log grid of 61 points on [1e-4, 3] (ratio 1.188), all simulated in **one batch** (3 r × 4 b × 61 rows). Then 3 rounds of vectorised **7-section** on each trajectory's bracket. Each round places 7 interior points and keeps [last stable, first unstable], so the bracket shrinks by 8³ = 512, which equals 9 bisection steps. Median relative half-width of the final bracket: 1.7e-4. |
| stability, `floor` (protocol) | not diverged (finite, \|x\| < 1e10), and MSSRE(h) ≤ 10·max(F, 1e-12). F is the **formula floor**: the MSSRE of the *same formula with linear activation* at its root-locus optimal step h_opt, on the same instance, direction and magnitude. h_opt = argmin_h ρ(P_h) (Zhang et al. 2019 criterion). F is simulated as extra rows of the linear coarse batch. If h_opt is unavailable (no closed-form theory, e.g. implicit_euler_err), F falls back to min_h MSSRE over the grid. |
| stability, `bounded` | not diverged, and MSSRE(h) ≤ ‖E(x₀,0)‖: the steady residual does not exceed the initial error. For linear Φ this is the root-locus condition ρ(P_h) < 1, up to finite-horizon effects. |
| h_max | the upper end of the run of consecutive stable grid points that contains h_ref = argmin MSSRE over the stable grid points, refined as above. If the run reaches h = 3 the value is censored `above`. If no grid point is stable it is censored `below` (h_max < 1e-4; plotted as 1e-4, aggregated as 5e-5). "Islands" (stable grid points above the first unstable one) are counted. Mini: 0.22 % of trajectories (bounded) and 0.08 % (floor) have islands, so the stable set is practically an interval. |
| τ-sensitivity (extension) | linear activation, ‖E(0)‖ = 1, τ ∈ {3e-2, 3e-3, 1e-3} in full ({3e-2, 3e-3} in mini) on both problems, plus the τ = 1e-2 main runs |
| stability, `nodiv` / `e4` (diagnostics) | `nodiv`: finite and \|x\| ≤ 1e10 over T (r-independent blow-up boundary; horizon-limited detection). `e4`: E4's P2 definition (MSSRE ≤ 0.1‖E(0)‖, max over the last T/8 ≤ 2 × max over the preceding T/8 or ≤ 1e-10, peak ≤ 10‖E(0)‖; constants read from configs/E4.py) |
| horizon block (review) | linear activation, ‖E(0)‖ = 1, τ = 1e-2, **T = 24** (N = 2400) on linear_system and the frozen-A control, every formula with an exact or local theory |
| scaled-error arm (review) | `R_common.ScaledError`, E_s = s·E with s ∈ {0.1, 10}; same x₀ (built on the unscaled problem), residuals and both criteria in the original unit; all activations and magnitudes. mini: euler, zead_5i, rk4 on linear_system; full: euler, taylor_zhang, zead_5i, zead_7i_cang2024, uhlig_4_6, ab3, rk4 on both problems |

**Theory** (`E2_theory.py`, independent of `zead.znn_stability`). ZNN-Bench always uses h = γτ with tau-form coefficients: τẋ_k ≈ c₋₁x_{k+1} + c₀x_k + c₁x_{k−1} + … (docs/spec/discretizations.md §1). With these conventions:

* Look-ahead error polynomial: P_h(ξ) = c₋₁ξ^s + (c₀ + h)ξ^{s−1} + c₁ξ^{s−2} + … + c_{s−1}. A paper that writes x_{k+1} = … + (τ/c₋₁)ẋ_k and calls γτ/c₋₁ its step size (Sun–Tian–Wang) has h_paper = h/c₋₁. Table `E2_theory.tex` lists both, together with the interval stated in the literature where one exists.
* Adams–Bashforth: P_h = ξ^q − ξ^{q−1} + hΣβ_jξ^{q−1−j}.
* RK4: \|1 − h + h²/2 − h³/6 + h⁴/24\| < 1.
* implicit_euler_err: e_{k+1} = e_k/(1 + h), so it has no upper limit.

h_max is the first h at which max\|roots\| = 1, found on a 4000-point grid on (0, 4] and then refined by bisection. The roots are computed as eigenvalues of batched companion matrices, which is the algorithm of `np.roots`; a direct `np.roots` check on both sides of every boundary passes. The values agree with the package's `znn_stability` to 2.9e-10.

* *Frozen-time prediction.* For linear ZNN on A(t)x = b or A(t)X = I, the exact error recursion is Σc_i e_{k+1−i} = −(hI + τM_k)e_k + O(τ^{p+1}), with M = A⁻¹Ȧ. The prediction is h_ft = sup{h : ρ(P_{h+τμ}) < 1 for every eigenvalue μ of M(t), t ∈ [0, T]}, taking the interval that contains h_opt.
* *Finite-horizon prediction for the floor criterion.* ρ(P_h)^{k_w} = 10F/‖E(0)‖, where k_w = 0.75N steps before the steady window starts.
* *Inverse-free schemes (review).* `cdtznn_mahuang2025`: LOCAL theory. On the affine E2 problems the linearisation about (x\*, J⁻¹) is block-triangular; the M-error obeys Σc_iΔ_{k+1−i} = −hΔ_k − τ(J⁻¹J̇Δ_k + Δ_kJ̇J⁻¹) + O(‖Δ‖²) (μ = γ), so both blocks have the P_h of the x-recursion (zead_5i_mahuang2025, h_max = 0.1725, h_paper = h/c₋₁ = 0.3737) and the frozen-time prediction uses the shifts τμ_i (x-block) and τ(μ_i + μ_j) (M-block, same instant). `broyden_dtznn_zhang2012`: no theory (its secant matrix never converges to J⁻¹, ‖JM − I‖_F ≈ 1.5–2.6, so the recursion I − hJM_k is trajectory dependent and chaotic). Both are excluded from the exact-theory headline; CDTZNN is reported separately (SUMMARY "Inverse-free schemes").
* *Interval property (review).* For all 21 formulas with an exact theory the stable set of P_h on (0, 3] is the single interval (0, h_max) (30000-point scan; ρ ≥ 1.0001 everywhere above h_max), so the bisection's interval assumption holds for the theory; measured islands are detection artefacts near neutral stability (see review log).

## 2. Mini-run results

> **Superseded (review 2026-10-18).** The numbers in this section are from the pre-review 52-task run (code version e2-tasks-2). The current mini run (103 tasks, e2-tasks-3) is in `results/mini/E2/SUMMARY.md`; the changes are listed in §6. The pre-review headline ratios moved by < 2 % (e.g. bounded/linear_system 0.986 → 0.971 with zead_5i_mahuang2025 added); the interpretation of the nonlinear r-dependence and of CDTZNN changed (§6, issues 1–3).

The mini run used 52 tasks: 16 main, 8 control and 28 τ-sensitivity. Pool: `--workers 3 --max-gb 40`, 3 min 17 s wall under concurrent machine load. All 52 finished, with 0 errors and 10,080 trajectory×criterion rows. Peak RSS was 154 MB per task. The analysis took 65 s. Headline numbers, from `results/mini/E2/SUMMARY.md` and `summary.json`:

**Linear activation, measured vs root-locus theory** (τ = 1e-2, ‖E(0)‖ = 1, 7 formulas with finite theory):

| criterion / problem | median h_meas/h_th [95 % CI] | max dev. | τ_b | meas./frozen-time | meas./finite-horizon |
|---|---|---|---|---|---|
| bounded, linear_system | 0.986 [0.831, 0.997] | 25 % (uhlig_4_6) | 1.00 | 1.008 (max dev 9.5 %) | – |
| bounded, matrix_inversion | 0.969 [0.868, 0.989] | 23 % (uhlig_4_6) | 1.00 | 1.004 (max dev 7.2 %) | – |
| bounded, A frozen | 1.005 [1.000, 1.062] | 6.6 % (zead_7i_cang2024) | 1.00 | – | – |
| floor, linear_system | 0.889 [0.645, 0.965] | 40 % (uhlig_4_6) | 1.00 | 0.921 | 0.956 [0.843, 0.998] |
| floor, A frozen | 0.913 [0.815, 0.971] | 22 % (zead_5i) | 1.00 | – | 1.013 [1.000, 1.134] |

Interpretation:
1. **The root locus ranks the formulas perfectly** (τ_b = 1). The measured limits sit slightly *below* it on the time-varying problems. The gap is the O(τ‖A⁻¹Ȧ‖) perturbation: the frozen-time prediction h + τμ reproduces it (median 1.008). The uhlig_4_6 deviation drops from 25 % to 4 % and the zead_5i deviation from 17 % to 8 %. The remainder matches the 6 % finite-horizon overshoot that the frozen-A control shows for the same formulas.
2. The deviation from the root locus shrinks like **O(τ)** at fixed h. The asymptotic slope of \|h/h_th − 1\| vs τ is 0.90–1.01 for 11 of the 14 (formula, problem) pairs (`E2_tau.tex`). Both consecutive-ratio slopes and the LS slope are reported. The three exceptions are:
   * zead_5i and uhlig_4_6 on matrix_inversion (0.83, 0.78): pre-asymptotic, with consecutive slopes still rising towards 1.
   * zead_7i_cang2024 on linear_system (2.5): a two-point fit. Here the time-variation shift (negative) and the finite-horizon overshoot (positive) almost cancel at τ = 1e-2, so \|dev\| is not a power law.
3. **With A frozen the root locus is exact.** The remaining ≤ 6.6 % *overshoot* is the finite-horizon detection limit of the bounded criterion: a root with ρ slightly above 1 does not exceed ‖E(0)‖ within 600 steps.
4. The **protocol (floor) criterion** gives limits about 11 % lower. This gap is the finite horizon: the slowest root must decay to 10F within the 4.5 s before the steady window starts. The finite-horizon prediction ρ^{k_w} = 10F/‖E(0)‖ matches the control to 1.3 % in the median; it is up to 15 % conservative for the slowly decaying 5- and 7-instant formulas, because transient constants are neglected.

**Nonlinear activations**, bounded criterion, medians over formulas × problems:

| activation | h_max at ‖E(0)‖ = 0.1 / 1 / 10 | shrink factor (‖E(0)‖ = 10 vs 0.1) [95 % CI] |
|---|---|---|
| hyperbolic_sine | 0.25 / 0.18 / < 1e-4 | ×0.0009 |
| power_sigmoid | 0.36 / 0.42 / 0.049 | ×0.20 [0.08, 0.71] |
| vaf | 0.021 / 0.12 / 0.13 | ×5.4 |
| pvaf1 | 0.020 / 0.095 / 0.090 | ×3.8 |
| predefined_exp | 0.017 / 0.072 / 0.082 | ×3.9 |
| sign_bi_power | 0.27 / 0.69 / 0.36 | ×1.4 |
| fixed_time | 0.11 / 0.23 / 0.15 | ×1.4 |

* Exponential and power tails (sinh(3e), e³ in power_sigmoid, sig² in sign_bi_power and fixed_time) shrink the stable step at large errors, as predicted. Activations with sign or sig^r terms have the opposite small-error effect: their chattering amplitude (∝ h, or h^{1/(1−r)}) is set by h alone, so it exceeds a *small* initial error first. sign_bi_power and fixed_time are non-monotone, with the largest h_max at ‖E(0)‖ = 1.
* At small errors the Lipschitz activations follow the linearised theory h_max(linear)/Φ'(0). Measured/predicted is 0.97 for power_sigmoid (Φ'(0) = 2.075) and 0.96 for hyperbolic_sine (Φ'(0) = 3).
* The formula ranking under nonlinear activations is the linear root-locus ranking: τ_b = 1.00 for every activation under the bounded criterion, and 0.71–0.81 under the floor criterion.
* **Protocol (floor) criterion:** sign_bi_power, fixed_time, vaf, pvaf1 and predefined_exp **never reach the formula floor** for any formula of order ≥ 4. 50–62 % of these cells have no stable step on the grid. The cause: an explicit step is locally stable at e only while h·Φ(e)/e < h_max(formula), which pins a chattering level far above the O(τ^{p+1}) floor. This is worst for high-order formulas, whose h_max is small. Only Euler and RK4 keep a floor-criterion range for every activation (0.001–0.3). Taylor–Zhang keeps one only for sign_bi_power and fixed_time (0.013–0.054).
* implicit_euler_err is stable up to h = 3 for every activation and magnitude (censored `above`), as expected.

**Independent spot check:** 40 random (cell, trajectory, criterion) results were re-simulated *outside the batch* as single trajectories at h_max ± 1.5·half-width. 40/40 were stable just below and unstable just above. This validates both the bisection and the independence of batch rows.

**Figure inspection** (all PNGs read): heat maps (CMAP_SEQ, log scale, annotated, censored cells marked); theory scatter (identity line, IQR bars, 4 series); r-dependence small multiples (≤ 8 panels, 8 activations in fixed `act` slots, censored markers); τ-convergence (`disc` slots, O(τ) reference line); MSSRE-vs-h profiles (`disc` slots, theory lines). Legends are present, there are no dual axes, and the widths are style.DOUBLE / ONEHALF. Fixed during inspection:
* suptitle overlap on the heat maps;
* hidden frozen-time markers;
* overlapping minor tick labels on the τ axis;
* the profile legend covering curves;
* a profile median that ignored diverged trajectories (it now uses the median over all trajectories, so a step where most trajectories diverge shows as a gap).

## 3. Deviations from PLAN.md, with reasons

1. **Two criteria.** The PLAN text "final residual < 10×initial floor" is ambiguous, so both criteria are reported.
   * The protocol criterion is implemented as MSSRE ≤ 10·F, with F the *formula floor*: linear activation at the formula's root-locus h_opt.
   * A first implementation took F = min_h MSSRE over the grid. That turned the criterion into an *accuracy* bound for RK4, whose steady error grows ∝ h²: it gave h_max = 0.15 instead of about 2.6. Hence F at h_opt.
   * The **bounded** criterion is added because it is the one that corresponds to the root-locus statement. Under the floor criterion, sign-type activations are "unstable" by construction: they chatter far above the floor.
2. **‖e(0)‖ is defined as ‖E(x₀,0)‖₂**, the ZNN error on which Φ acts, rather than ‖x₀ − x\*‖. x₀ is placed exactly on that sphere around x\*. Linear activation is also run at all three magnitudes, as a control of magnitude invariance: h_max(bounded) is the same at all r.
3. **Vectorised multi-section** replaces scalar bisection: 3 rounds of 7-section, equivalent to 9 bisections, run on per-trajectory brackets after a 61-point coarse grid. One simulate call evaluates a whole grid. The coarse grid also guards against non-monotone stability sets.
4. **Independent instances for `matrix_inversion`** (seed offset 100). With equal seeds, `linear_system` and `matrix_inversion` draw *identical* A(t) (§5). Cross-problem agreement would then be inherited trivially.
5. **Extensions not in PLAN:**
   * frozen-A control;
   * τ-sensitivity (τ → 0 at fixed h);
   * frozen-time and finite-horizon predictions;
   * implicit_euler_err as an unconditionally stable control;
   * h grid down to 1e-4, with values below it reported as censored. For hyperbolic_sine at ‖E(0)‖ = 10 the predicted limit is ≈ 2e/sinh(3e) ≈ 1e-6, out of range on purpose.

## 4. Full-scale cost estimate and commands (updated 2026-10-18, review)

Estimated from the measured wall per step of every (task kind, formula class, activation class, problem) group of the
review mini run (103 tasks, 897 seed × activation groups): median group wall / N per class, multiplied by the
full grid's seeds and N = T/τ for every full task, then a 1.5× margin for machine load.

| block | tasks | content | est. CPU |
|---|---|---|---|
| main (τ = 1e-2, 8 activations, 3 magnitudes, 8 seeds × B = 4) | 48 | 22 formulas × 2 problems + 2 inverse-free × 2 problems | 2550 s |
| control (A frozen, linear) | 22 | 21 exact-theory formulas + CDTZNN | 70 s |
| τ-sensitivity (3e-2, 3e-3, 1e-3; LS, MI, control) | 189 | 21 exact-theory formulas | 2020 s |
| horizon (T = 24; LS + control) | 44 | 22 formulas | 320 s |
| scaled-error arm (s = 0.1, 10; LS + MI) | 28 | 7 formulas, all activations and magnitudes | 1220 s |
| **total** | **331** | | **≈ 6200 s = 1.7 core-h (budget 2.5 core-h)** |

* Longest task: `main-cdtznn_mahuang2025-matrix_inversion` ≈ 6–7 min (mini: 144 s for 3 seeds; dense 16 × 16 M_k
  products per row); next `main-implicit_euler_err-matrix_inversion` ≈ 5.5 min (80-step resolvent bisection).
* **Peak RSS per task: 0.47 GB** (CDTZNN on matrix_inversion), ≤ 0.21 GB for every other task (seeds are processed
  sequentially, so RSS does not grow with the seed count).
* **Recommended `--workers 24`** (≤ 12 GB in total, far below the 200 GB cap): expected wall ≈ 8–10 min, bounded by the
  longest task. The analysis is single-threaded: ≈ 2.5 min at mini, ≈ 10–20 min expected at full scale.
* Output: ≈ 25 MB at mini (profiles dominate); ≈ 120 MB expected at full scale.

Commands (exactly as validated; run from `znn-bench/`):

```bash
cd znn-bench
# build (331 tasks; new zead_*/uhlig_* and inverse-free keys are picked up automatically; task ids hash the
# payload incl. CODE_VERSION e2-tasks-3, so stale results are never reused)
python experiments/tasks/E2_stability.py build --scale full --out results/E2
# launch under the memory guard (resumable; done file = results/E2/results/<id>.parquet)
python tools/memguard.py pool --tasks results/E2/tasks.jsonl --workers 24 --max-gb 200
# progress
python experiments/tasks/E2_stability.py status --out results/E2
# analyze
python experiments/analyze_E2.py --results results/E2 --figdir figures --tabdir results/tables
```

Mini (validated 2026-10-18: 103/103 tasks, 0 errors, 6 min 38 s wall with `--workers 3`, analysis 2.6 min):

```bash
python experiments/tasks/E2_stability.py build --scale mini --out results/mini/E2
python tools/memguard.py pool --tasks results/mini/E2/tasks.jsonl --workers 3 --max-gb 40
python experiments/analyze_E2.py --results results/mini/E2 --figdir figures/mini --tabdir results/tables/mini
```

The analysis reads only the tasks listed in the current `tasks.jsonl`. `python experiments/configs/E2.py --scale full`
prints the grid and the task count per block.

## 5. Zoo observations (nothing under `znnbench/` was edited)

* *Review addition (observation, not a bug):* `cdtznn_mahuang2025` exposes the x-recursion coefficients and
  `stability()`/`root_info()` of the x-recursion (h < 0.1725, docs/spec/inverse_free_dtznn.md E3/E8). That interval is
  correct for the x-recursion and is the LOCAL limit with A frozen (E2 control: 1.04 × 0.1725 at T = 24), but it is not
  the scheme's stability limit on time-varying problems: the M-recursion's frozen-time shift is τ(μ_i + μ_j), twice the
  x-block's, and it limits CDTZNN at h ≈ 0.075–0.12 on `linear_system` (n = 4, κ = 10; seeds 0–2), measured 0.104 at
  T = 24. Suggest the spec's E8 ("consistent within the linear-ZNN interval") mention the M-block.
* No zoo bug was found in the review. Checked: `zead_5i_mahuang2025` and `zead_7i_cang2024` coefficients against the
  spec (h_max 0.17250 / 0.11594, h_max/c₋₁ = 0.3737 / 0.2909 = the spec's h_paper limits); CDTZNN and Broyden
  `describe()` (`inverse_free: true`, `x_recursion_key`, `mu: gamma`) drive the theory classification.

* **No zoo bug found.** The root-locus h_max of every formula reproduces `describe()['h_max']` to 3e-10, and RK4 and AB match their closed forms. 40/40 single-trajectory re-simulations agree with the batched results.
* *Observation, not a bug, relevant to every experiment that compares problems:* `linear_system(seed=s)` and `matrix_inversion(seed=s)` build `TVMatrix` first from `default_rng(s)` with the same arguments. Their A(t) are therefore **identical**, and only E-structure and x\*(t) differ. E2 uses `seed_offset = 100` for `matrix_inversion`.
* *Observation:* plain `LookAhead` instances (`euler`, `taylor_zhang`) expose no `h_max`, `h_opt` or `h_range_paper` in `describe()`; only the `ZeaD` subclasses do. The analysis therefore carries the literature intervals from docs/spec/discretizations.md §2 itself.
* *Performance:* `implicit_euler_err` with nonlinear activations is about 8× slower per step (generic `prox` bisection).

## 6. Review log (2026-10-18)

Independent methodological review of E2 (protocol, bisection, criteria, theory, confounds, analysis, figures, cost),
including every item of the cross-cutting REVIEW_CHECKLIST. Code version bumped `e2-tasks-2` → `e2-tasks-3`; the mini
run was rebuilt and re-run end to end (103/103 tasks, 0 errors, 6 min 38 s wall with 3 workers, analysis 2.6 min).
Nothing under `znnbench/` or `tests/` was edited. Severity: MAJOR = would change or invalidate a stated conclusion;
MEDIUM = changes a number or its uncertainty; MINOR = reporting/robustness.

| # | severity | issue | change | effect on numbers (mini) |
|---|---|---|---|---|
| 1 | MAJOR | **CDTZNN's T = 6 limit is a finite-horizon artefact, and its theory was missing.** Its M-recursion is weakly unstable (≈ 0.4 %/step) above the M-block limit and starts small, so it never reaches the bounded threshold within 6 s. Direct check (seed 0, T = 12): h = 0.09 grows (‖JM − I‖ 3e-5 → 1.3e-3 between k = 300 and 1199), h = 0.11 diverges by t = 12, although the T = 6 criterion calls both stable. | Local theory (E2_theory: block-triangular linearisation, both blocks share P_h of zead_5i_mahuang2025, M-block frozen-time shift τ(μ_i + μ_j)); horizon block T = 24; CDTZNN added to the frozen-A control; its exact-inverse twin zead_5i_mahuang2025 added to the mini; SUMMARY "Inverse-free schemes". | LS bounded: 0.135 (T = 6) → **0.104 (T = 24)**, paired ratio 0.767. Measured / frozen-time with M-block at T = 24: 1.12 (bounded), 1.02 (floor); frozen-A control 1.04 × local theory (1.15 at T = 6 = finite-horizon overshoot). CDTZNN / exact-inverse twin = 0.79 at T = 24: J⁻¹ → M_k costs ≈ 21 % of the stable step. The T = 6 heat-map cells of CDTZNN over-state its limit by ≈ 30 %. |
| 2 | MAJOR | **Unit dependence (checklist item 1).** No activation except linear has a length scale tied to the problem, so every nonlinear cell depends on the unit of E. | Paired scaled-error arm (`R_common.ScaledError`, E_s = sE, s ∈ {0.1, 10}, same x₀, residuals/criteria in the original unit); `fig_E2_units`; SUMMARY section + bullet. | At the same physical ‖E(0)‖ = 1, s = 10 changes the bounded h_max by ×0.026 (power-sigmoid), ×0.0003 (sinh, bound), ×0.24 (sign-bi-power), ×0.33 (fixed-time), ×0.53–0.70 (VAF, PVAF-1, pred.-exp); s = 0.1 by ×0.24–0.78 (sinh ×2.2). Linear: exactly 1. Argument equivalence h_max(r, s) = h_max(s·r, 1) holds in the median for every activation under bounded and nodiv: the ‖e(0)‖ axis of the r-dependence figures *is* a unit axis. Every nonlinear-activation statement is now qualified "for E in the problem's own unit". (Floor criterion: no equivalence for sign-type, because F rescales.) |
| 3 | MAJOR | **The bounded criterion's threshold is ‖E(0)‖ itself**, so the reported small-error "shrinkage" of sign-type activations (bounded h_max grows ×1.7–7.5 from ‖E(0)‖ = 0.1 to 10) is a residual-size statement (chattering amplitude ∝ h vs the threshold r), not a loss of stability. | r-dependence also reported with the r-independent blow-up criterion (nodiv): new `fig_E2_rdep_nodiv_*`, SUMMARY bullet; recommendation that stability-vs-‖e(0)‖ claims use nodiv. | nodiv shrink factors (‖E(0)‖ = 10 vs 0.1): sign-bi-power ×0.84, fixed-time ×0.91, VAF ×0.89, PVAF-1 ×0.91, pred.-exp ×0.99, power-sigmoid ×0.13, sinh ≤ ×0.0009. The PLAN hypothesis (power-type tails shrink the stable region) holds under nodiv for every nonlinear activation, strongly only for the exponential/cubic tails; the "opposite trend" of sign-type activations disappears. |
| 4 | MEDIUM | **Pseudo-replication (checklist 5).** The B directions of one instance give h_max within 0.17 % (median relative range) vs 1.3 % between instances; CIs over formulas are descriptive only. | Per-seed [min, max] ranges for every headline ratio (SUMMARY, summary.json `seed_range`); pseudo-replication line; **full scale 3 → 8 instances** (cost +170 %, total still < 2 core-h). | Bounded/LS 0.971, seed range [0.927, 0.983]; vs frozen-time 1.026 [1.009, 1.050]. |
| 5 | MEDIUM | **Ranking claim was fragile.** "The root locus ranks the formulas perfectly (τ_b = 1)" held for the 7 pre-review formulas only. | zead_5i_mahuang2025 added; Kendall p-values labelled descriptive. | τ_b(theory, measured) = 0.93 (bounded, both problems, n = 8): the one discordant pair, zead_5i_mahuang2025 vs uhlig_4_6, has root-locus limits 2 % apart (0.1725 vs 0.1760), so its order is set by the O(τ) time-variation shift. |
| 6 | MEDIUM | **E4 consistency.** E4 does not reuse E2. Its P2 recomputes h_max with its own definition: Euler, τ = 1e-3, MSSRE ≤ 0.1‖E(0)‖, no growth over the last T/8, peak ≤ 10‖E(0)‖, all calibration trajectories jointly (worst case), and the largest stable point of a descending grid. That last rule can select an island above the first instability; E2 uses the connected run around the optimum. | E2 evaluates E4's criterion everywhere (`e4`, constants read from configs/E4.py); SUMMARY section. E4 files not edited. | h_max(e4)/h_max(bounded) at ‖E(0)‖ = 1: linear 0.975, power-sigmoid 0.87, sinh 0.99, sign-bi-power 0.41, fixed-time 0.51, VAF 0.16, PVAF-1 0.21, pred.-exp 0.24. **For E4's owner:** (a) P2's h_max for chattering models is a residual-size limit set by STAB_RHO = 0.1, not a stability limit; (b) for power-type models it is set by E4's ‖E(x₀,0)‖ and unit. E4 mini P2 gives power-sigmoid 0.0048–0.0056 vs E2 bounded 1.03 (Euler, MI, ‖E(0)‖ = 1), consistent with E2's r-dependence at E4's ‖E(x₀,0)‖ ≈ 27. Linear agrees (E4 1.980, E2 1.98). P2 rankings of power/sign-type models are therefore unit- and x₀-dependent. |
| 7 | MEDIUM | **Inverse-free theory column.** Broyden was excluded without a stated reason, and CDTZNN was not handled. | Broyden: no theory, with the reason recorded in the summary and tables (secant M_k ↛ J⁻¹, chaotic); flagged non-interval (19–40 % of trajectories with islands; F = grid-minimum fallback because it does not track). CDTZNN: issue 1. The exact-theory headline is restricted to exact-inverse formulas; E2_theory.tex marks † local / ‡ none. h-mapping check: h_max/c₋₁ = 0.3737 (zead_5i_mahuang2025, also the CDTZNN x-recursion) and 0.2909 (zead_7i_cang2024) equal the spec's h_paper limits. | none on the exact-theory headline (CDTZNN/Broyden never entered it) |
| 8 | MINOR | **Non-interval stable sets.** | The theory is verified to be an interval for all 21 formulas (Theory section). Per-formula island fractions and a non-interval list are in SUMMARY. | Islands > 5 %: Broyden (chaotic) and zead_5i_mahuang2025 (3–6 %). For zead_5i_mahuang2025, ρ rises only to 1.0031 at 1.05 h_max (×6 growth in 600 steps), which causes its 49 % nodiv overshoot and the islands. Every other formula has < 4 %. |
| 9 | MINOR | **τ-slope fits (checklist 3).** The previous rule (drop the largest τ if the first two local slopes differ by > 0.5) did not flag 2-point fits. | New rule: the asymptotic run is taken at the small-τ end, where local slopes agree within 0.15. 2-point fits and absurd slopes (< 0 or > 3) are flagged in the SUMMARY and summary.json. | Flagged: zead_7i_cang2024 on LS (slope 2.50, 2-point fit), and uhlig_4_6 on the shift and control series (2-point fits). All other fits: 3 points, slopes 0.78–1.05. |
| 10 | MINOR | **Threshold and horizon robustness** were not quantified. | Threshold sensitivity recomputed from the coarse profiles: floor θ ∈ {3, 30, 100}, bounded c ∈ {0.1, 0.3, 3}. Horizon block (issue 1). | Lipschitz activations (linear, power-sigmoid, sinh) are unchanged at grid resolution. Sign-type: bounded c = 0.1 → ×0.18–0.50, c = 3 → ×1.0–1.19; under the floor criterion they stay censored (no stable step) at every θ, except fixed-time at θ = 100 (×2.8). T = 24 vs T = 6 for the exact-inverse formulas: bounded ×0.983 (median; min 0.894, zead_5i_mahuang2025), floor ×1.074; frozen-A control bounded 1.006 → 1.002 × theory. |
| 11 | MINOR | **Censoring (checklist 2).** | Censoring line in SUMMARY (fractions below/above per criterion, missing tasks, error groups). Censored values are kept as bounds everywhere; nothing is dropped. | Main block, 6336 rows per criterion. Below the grid: bounded 6.7 %, floor 41 %, nodiv 5.0 %, e4 11.6 %. Above 3: 9.1 % (implicit_euler_err). |
| 12 | MINOR | **Process.** The pre-review mini had been rebuilt to 69 tasks, but the pool stopped at 60/69: SUMMARY.md was missing, and the figures came from the older 52-task run. The notes said "inverse-free on matrix_inversion only", while the code used both problems. | Full rebuild and re-run; notes and docstrings corrected. | – |

**Checklist items without findings.** Item 7 (figure style): all 20 mini PNGs were regenerated. The theory scatter, the unit figure, the bounded heat map and the nodiv r-dependence page were inspected after the fixes. Fixed: overlapping CDTZNN labels in the theory scatter, and the unit-figure titles. The heat maps now group the inverse-free rows at the bottom. Every figure keeps fixed `act`/`disc` slots, ≤ 8 series per panel, legends, no dual axes and log axes. Item 6 (confounds): see the note on confounds below.

**Note on confounds.**
* h vs c₀h: E2 always uses h = γτ with tau-form coefficients. The paper convention is h/c₋₁, also tabulated.
* Equal nominal γ is not equal effective gain for the nonlinear activations: Φ'(0) = 2.075 for power-sigmoid and 3 for sinh. The linearised prediction h_lin/Φ'(0) is reported (measured/predicted 0.96).
* Transient leakage into the floor F at h_opt: checked, none at mini. At full scale, uhlig_3_4 (ρ_opt = 0.975) is the candidate; it is flagged automatically.
* The horizon is too short for slowly unstable modes (issue 1).
* Unit (issue 2) and threshold scaling (issue 3).
* CDTZNN's M-recursion uses the linear gain μ = γ whatever the activation, so its nonlinear-activation cells are capped by the linear M-block limit (≈ 0.13 at T = 6). They are not comparable with the exact-inverse rows.

**Verdict.** After these fixes E2 is ready for the full sweep (331 tasks, ≈ 1.7 core-h, `--workers 24`, peak 0.47 GB per task). The headline claims to carry into the paper:
* The root locus predicts the linear-activation limit to within a few percent once the O(τ‖A⁻¹Ȧ‖) frozen-time shift is included.
* The protocol (floor) criterion sits ≈ 13 % lower because of the finite horizon.
* Stability vs ‖e(0)‖ for nonlinear activations is real for the exponential/cubic tails. It is unit-dependent for all of them.
