# E8: implicit (consistent) vs explicit discretization of sign-type ZNN

Files:
* config: `experiments/configs/E8.py` (defines `get_configs()`; also a build script);
* analysis: `experiments/analyze_E8.py`;
* this note.

No custom task script was needed: every run goes through the standard runner (`znnbench.runner`, chunk tasks, `experiments/run_task.py`). The scheme definitions, proofs and assumptions are in `docs/spec/consistent.md`.

## 1. Question and hypotheses

Explicit ZNN-Euler on a time-invariant problem realises exactly the explicit Euler step of the error ODE, E_{k+1} = E_k - h Phi(E_k) with h = gamma tau (consistent.md, Lemma 1). For non-Lipschitz (finite-, fixed- and predefined-time) activations this gives three failures:
* period-2 chattering, with amplitude e* = (a h/2)^{1/(1-p)} for a term a|e|^p sign e;
* blow-up for p > 1 terms at large errors;
* no finite-step convergence.

A consistent one-step error map Psi_h (resolvent, PEB homogeneity map, exact flow, semi-implicit), requested as the error rate, gives E_{k+1} = Psi_h(E_k) instead. For multistep look-ahead formulas such as Taylor–Zhang (TZ) this holds only with the error-memory (EM) rate (Lemma 2).

Hypotheses tested:
* **H1, time-invariant problem.** The consistent schemes reach ||E|| < 1e-13 in finitely many steps, matching the exact-flow prediction ceil(T(e0)/h) and, for SBP and predefined-time, the uniform bound ceil(T_max/h). Explicit Euler and TZ never get there: they chatter or diverge.
* **H2, chattering metric.** The norm-based chattering index is blind to e -> -e cycles, while `chatter_vec` and `rate_alt` detect them.
* **H3, time-varying problems at fixed h.**
  * The consistent schemes have MSSRE equal to one local truncation error, with order 2 (Euler) or 3 (TZ-EM) in tau, independent of h and of the activation.
  * Explicit sign-type ZNN has an O(h) floor, with order 0 in tau.
  * The naive implicit TZ (no error memory) loses the order.
* **H4, step-size range.** The consistent schemes are bounded and contracting for every h on the grid, up to 10. The explicit schemes have a finite window, which shrinks for power-type tails and is empty beyond h of about 0.03 for the predefined-time activation.
* **H5, extension.** EM with the exact map recovers the formula's order for every registered look-ahead formula, including high-order ZeaD and Uhlig formulas.
* **H6, error unit (added in review, 2026-10-18).** None of the non-linear activations has a length scale, so with the error expressed in units of 1/s the error ODE is E' = -γΦ(sE)/s. The explicit-Euler band and the dead zone of the exact map are fixed in the unit of E_s (SBP r = 1/2: band (h/4)², i.e. (h/4)²/s in the problem's unit). Hypothesis: boundedness for every h, finite-step exactness on time-invariant problems and the s-free uniform step bound are unit-free; the *accuracy* advantage on time-varying problems is a function of the dimensionless band-to-drift ratio ρ = √m δ(h)/(s F(τ)) only, and vanishes (or reverses slightly) for ρ ≲ 1, as in the robotics review (R_NOTES.md, R_units).

## 2. Exact protocol

Common settings:
* model `znn`, the base design-formula ZNN, which honours the `ErrorMapRequest` protocol since the core change "consistent 1" landed;
* problem seeds (`seed: $seed`, so a new instance per seed) x B = 4 initial states (simulate's x0 stream): **3 seeds at mini scale (12 trajectories per cell), 10 seeds at full scale (40 per cell; review 2026-10-18)**, except the extension `E8_formulas` (3 seeds at both scales, slopes only). The instance is the experimental unit: the B states of one instance give identical steady-state metrics for the consistent schemes (e.g. MSSRE 1.132e-05 four times), and the between-instance sd of the headline log ratios is 3-8x the within-instance sd. Seeds 0-2 are a subset of 0-9, so every mini run id is a full run id (checked: 1287 of 9516, 0 missing);
* fixed-h mode on the grid `H_GRID = [0.01, 0.03, 0.1, 0.3, 0.6, 1, 1.5, 3, 10]`, with every h value of a run batched into one simulate call (`batch_gains: true`, 36 trajectories; batched equals unbatched bit for bit, see the core change "runner 2");
* exact data (`info: analytic`), no noise, `record_every: 1`;
* steady window = last 25 % of the horizon.

Activations (full scale):

| id | registry call | why |
|---|---|---|
| `sign` | `activation: sign` | finite-time, jump at 0 (dead zone) |
| `sbp0.25/0.5/0.75` | `activation: sign_bi_power, r` | the classical finite/fixed-time activation of Li–Chen–Liu 2013 |
| `fixed_time` | `activation: fixed_time` (k1 = k2 = k3 = 1, p = 1/2, q = 2) | bi-power plus linear term, as in the source |
| `predefined_exp` | `activation: predefined_exp, p = 0.5, k1 = 1, k2 = k3 = 0` | the Sanchez-Torres predefined-time system; its exact one-step map is JR20 Eq. (7) (consistent.md §6) |

Discretizations (keys filtered by `znnbench.available()` at build time):

| family | explicit | consistent / implicit | ablation |
|---|---|---|---|
| Euler (order 2) | `euler` | `implicit_euler_err` (resolvent), `peb_euler`, `exact_flow_euler`, `semi_implicit_euler` | |
| TZ (order 3) | `taylor_zhang` | `implicit_taylor_zhang` (EM + resolvent), `peb_taylor_zhang` (EM + PEB) | `implicit_taylor_zhang_naive` (resolvent, no memory) |

Configs:

| config | problems | tau | T | metrics | traces |
|---|---|---|---|---|---|
| `E8_ti` | `linear_system(n=4, kappa=10, freq=0)`, time-invariant | 1e-2 | 25 s (2500 steps) | eps 1e-3, 1e-6, **1e-13**; extra initial/final/max residual | every step |
| `E8_tv` | `linear_system(n=4, kappa=10, freq=1)`, `matrix_inversion(n=4, kappa=10, freq=1)` | 1e-2, 3e-3, 1e-3, 3e-4 | 6 s | eps 1e-3, 1e-6; extra as above | 600 points |
| `E8_formulas` (extension) | `linear_system` (time-varying) | 3e-2, 1e-2, 3e-3, 1e-3 | 6 s | as above | none |
| `E8_scale_ti` (error unit, review) | as `E8_ti`, wrapped in `experiments.tasks.R_common:ScaledError` with s ∈ {1e-2, 1e1, 1e3} | 1e-2 | 25 s | as `E8_ti` | none |
| `E8_scale_tv` (error unit, review) | `linear_system(freq=1)` wrapped as above | 1e-2, 3e-3, 1e-3 (mini 1e-2, 1e-3) | 6 s | as `E8_tv` | none |

The error-unit arm uses the schemes euler, implicit_euler_err (the robotics "implicit SBP"), exact_flow_euler, taylor_zhang, implicit_taylor_zhang and the naive ablation; activations: all six at full scale, sign / SBP(0.5) / predefined_exp at mini scale. s = 1 is the base arm (`E8_ti`, `E8_tv`), paired by (seed, x0): the recorded initial residual is identical across s (max relative difference 0). `ScaledError` keeps the recorded residual, MSSRE and the 1e-13 threshold in the problem's own unit; `chatter_vec` is a ratio of E norms and therefore s-free; `rate_alt` uses x increments and is s-free.

More detail on the configs:
* **`E8_ti`: tau is irrelevant.** On a time-invariant problem the iterates do not depend on tau at fixed h, because every update uses only tau x_dot = J^{-1}(tau r_k) and tau r_k depends on h only (explicit: -h Phi(E); consistent: Psi_h(E) - E or the EM combination). One tau is therefore enough. T = 25 s covers sign at h = 0.01 (about |e0|/h <= 2000 steps).
* **`E8_formulas`: which formulas.** Explicit formula F and `consistent(formula=F, error_map='exact', memory=True)` for every F in `keys.GROUPS['disc']`: euler, taylor_zhang, zead_5i, zead_6i_guo2017, **zead_7i_cang2024** (registered by the sibling workflow during this session and picked up automatically), uhlig_3_4, uhlig_4_6. Any `uhlig_[5-9]_*` key registered later is added too, capped at 8 formulas. A formula is included only if it is a registered `LookAhead`. Activations: sign and SBP(r = 0.5). h values: 0.03 and 0.3.

**Mini scale** (`ZNNBENCH_SCALE=mini`) keeps B = 4 and the full h grid with 3 seeds, all six activations in `E8_ti` and `E8_tv`, but reduces:
* the `E8_tv` tau values to 1e-2, 3e-3 and 1e-3, with matrix_inversion only at 1e-2 and 3e-3 and only for sign / SBP(0.5) / predefined_exp; SBP(0.25), SBP(0.75) and fixed_time run on linear_system at 1e-2 and 3e-3 only;
* the error-unit arm to sign / SBP(0.5) / predefined_exp and, on the time-varying problem, tau ∈ {1e-2, 1e-3};
* the extension to sign and four formulas, at tau 1e-2, 3e-3 and 1e-3.

### Metrics and analysis definitions

* **Steps to exact convergence:** `t_eps_1e-13 / tau`, the first step after which the residual stays below 1e-13 (inf if never). The exact-flow prediction is ceil(T(max_i |e0_i|)/h), with T the settling-time function of the activation (`act.settling_time(e0, 1)`). E0 is reconstructed from the recorded problem kwargs and `simulate.initial_state` (`theory_steps`). The uniform bounds are ceil(C/h), with C = `tight_uniform_bound` (AL19 Thm 1) for SBP and 1/k1 for predefined_exp.
* **Chattering:** `chatter_vec` = mean||E_{k+1} - E_k|| / mean||E_k||, and `rate_alt` = the fraction of sign alternations of the components of x_{k+1} - x_k, both over the steady window (online statistics of `simulate`). The norm-based `chatter` (`metrics.chattering`) is reported to show that it is blind to the cycles.
  * **Masking.** When a trajectory has converged to round-off (mean steady residual <= 1e-12), `chatter_vec` is NaN (exact zero residual: 0/0) or rounding noise of about 1.5. `rate_alt` likewise counts the sign flips of rounding-level increments (about 0.5). The analysis therefore uses `*_eff` = 0 for such trajectories ("no motion"); see §6.
* **Stable h range** (on the grid, per problem / activation / scheme; tau_ref = 1e-3 for the time-varying problems, or the smallest tau available):
  * "bounded": no trajectory diverged; we report the largest h of the contiguous bounded block starting at h = 0.01;
  * "contracting": bounded, and MSSRE <= 0.1 x the initial residual for all 12 trajectories; we report the longest contiguous block [lo, hi].

  At tau = 1e-2, small h means a small gamma (h = 0.01 gives gamma = 1), so a slow transient can fail the contraction test. This is a property of the fixed-h mode, not instability, and is stated in the table caption.
* **Empirical order** of MSSRE in tau at fixed h = 0.3 (`fit_order`):
  * points at or below 1e-12 (round-off) are dropped;
  * consecutive-ratio local slopes are computed;
  * the asymptotic range is the longest small-tau tail whose consecutive local slopes agree within **0.15** (the tolerance of `tests/test_problems_robot.py`; 0.3 before the review);
  * the LS slope over that tail is reported next to the LS slope over all valid points; 2-point fits are counted and not interpreted.
* **Headline comparisons** are at h = 0.3 and tau_ref:
  * statistic: the median of the *paired* log10 ratios (same instance and initial state; a diverged numerator counts as +inf), with a **two-stage cluster bootstrap** 95 % CI (2000 resamples, `default_rng(0)`: instances with replacement, then initial states within each drawn instance; review 2026-10-18, the trajectory bootstrap is kept in summary.json as `ci95_traj_bootstrap`);
  * one-sided sign test on the **per-instance medians** (n = number of instances: 3 at mini scale, so the smallest attainable p is 0.125 and no mini test can be significant; 10 at full scale), Holm-adjusted **within each comparison family** (one comparison label across the problem x activation cells, <= 12 rows at full scale);
  * comparisons: explicit Euler / Euler + exact flow, explicit Euler / Euler + PEB, explicit TZ / TZ-EM + resolvent, naive TZ / TZ-EM, and `chatter_vec` for explicit Euler / exact flow and naive TZ / TZ-EM.
* **Kendall tau_b** (scipy, variant b) measures ranking agreement of the 9 schemes by median MSSRE at h = 0.3: linear_system vs matrix_inversion, and largest vs smallest tau.

Outputs:
* figures: `fig_E8_traces_ti`, `fig_E8_traces_tv`, `fig_E8_steps`, `fig_E8_chatter`, `fig_E8_chatter_blind`, `fig_E8_mssre_vs_h_<problem>`, `fig_E8_order_<problem>`, `fig_E8_formulas`, and (review) `fig_E8_scale_ti` (exact-flow steps and explicit-Euler divergence vs h, one line per error scale s) and `fig_E8_scale_tv` (paired log ratio vs log10 ρ for every (s, τ, h) cell, one fitted curve for all s) (pdf + png);
* tables: `E8_steps`, `E8_stable_h`, `E8_chatter`, `E8_orders`, `E8_headline`, `E8_headline_full`, `E8_formulas`, and (review) `E8_scale` (`.tex`, booktabs; all compile with pdflatex; the tables with > 40 rows are longtables, which are 66-245 pt wider than an `article` text width and need the supplement's landscape/wide layout);
* `<results>/summary.json` and `<results>/SUMMARY.md`.

Figure style:
* fixed slots of the group `disc_implicit` (8 members) via `keys.style_in_group`; the naive ablation is outside the group and drawn grey with an x marker and dotted line;
* formulas use the group `disc`;
* <= 8 series per panel. The Euler and TZ families are split into separate panels where all 9 schemes appear (traces, chatter, order); the MSSRE-vs-h figure shows the 8 group members, and the ablation is in the tables;
* log axes, no dual axes, a legend in every multi-series figure;
* `chatter_vec` is drawn as dots with IQR whiskers on a log scale, because its values span 3 decades and bars on a log axis would be misleading; `rate_alt` (a fraction) as bars from 0.

## 3. Deviations from PLAN.md (and why)

1. **Scheme set.** The PLAN lists "Euler, implicit_euler_err, consistent (homogeneous), implicit Taylor–Zhang, semi-implicit". We run all of `peb_*` (the homogeneous map of PEB19), `exact_flow_*` (the JR20 / settling-time exact map), the semi-implicit map, and the ablation `implicit_taylor_zhang_naive`. The ablation is the evidence that the error-memory rate, our technical contribution, is what makes multistep formulas consistent. `exact_flow_taylor_zhang` and `semi_implicit_taylor_zhang` are omitted to stay within <= 8 group colours; by Lemma 2 they equal their Euler counterparts on time-invariant problems.
2. **Predefined-time activation.** We use `predefined_exp` with k2 = k3 = 0 rather than the default k2 = k3 = 1:
   * with the extra linear and sign terms there is no closed-form exact or PEB map; `exact` and `peb` would silently fall back to the resolvent (consistent.md §6), making 3 of the 4 consistent Euler schemes identical;
   * with k2 = k3 = 0 the activation is exactly the JR20 class, where Theorem 6 and Corollary 7 give exact discrete predefined-time convergence;
   * even so, `peb` falls back to `exact` for this activation (warned once per task), so `peb_euler` = `exact_flow_euler` and `peb_taylor_zhang` = EM + exact flow for predefined_exp.
3. **Fixed-time activation.** We keep the source form with the linear term (k3 = 1). Because it has three power terms, `exact` falls back to the Lie-split `peb` map (warned), so `exact_flow_euler` = `peb_euler` for fixed_time. It is excluded from the exact-flow theory agreement check.
4. **Order sweep in tau added.** The PLAN lists metrics only. The tau sweep at fixed h (the DTZNN convention) is what shows that the consistent schemes restore the O(tau^{q+1}) residual while explicit sign-type ZNN has an h-only floor. This is the quantitative core of the "continuous-time guarantee vs discrete reality" argument.
5. **Chattering index.** We report `chatter_vec` and `rate_alt`, with the round-off masking described above, instead of the norm-based index. The norm-based index is still reported, to document its blind spot.
6. **"Exact" convergence** is operationalised as ||E|| < 1e-13. The Euler-based consistent schemes reach exactly 0.0 in floating point; EM-TZ settles at about 3e-16. The resolvent and semi-implicit maps of *continuous* activations converge only super-linearly in theory (consistent.md §3.1, §3.4), but hit the threshold, and in floating point exact 0, a few steps after PEB / exact flow. The step counts show this difference (e.g. SBP r = 0.5, h = 0.3: 22 (resolvent) and 25 (semi-implicit) vs 15-16 (PEB / exact)).
7. **Extension config `E8_formulas`.** Not in the PLAN. It reproduces consistent.md §4.3 at benchmark statistics, and it is the place where keys added by sibling workflows (`zead_7i_cang2024`, `uhlig_5_*`) are picked up automatically.

## 4. Mini-run results (`results/mini/E8/`, `figures/mini/fig_E8_*`, `results/tables/mini/E8_*`)

**Status after the review (2026-10-18).** The point estimates below are unchanged; the CIs quoted below are the preparer's trajectory bootstraps, superseded by the cluster-bootstrap CIs in `SUMMARY.md` (e.g. linear system / sign, explicit Euler / exact flow: 4.54 [4.39, 4.87] instead of [4.43, 4.83]), and the sign-test p-values are now instance-level (none significant with 3 instances). All accuracy numbers are **in the problem's own error unit (s = 1)**; §4.1 states which conclusions survive a change of unit. Review rerun: build 1287 specs / 114 tasks; pool `--workers 3 --max-gb 40` resumed 69 done tasks and ran the 45 new error-unit tasks in 221 s wall (577 s CPU), 0 failed; analysis 35 s.

Preparer's run:
* build: 558 specs, 46 tasks;
* `memguard pool --workers 3 --max-gb 40`: all 46 tasks finished, 0 failed, in **162 s wall** (407 s user CPU) on a machine shared with another pool;
* status: TI 2653 ok / 263 diverged, TV 13177 ok / 1403 diverged, extension 576 ok; no error, skipped or timeout rows;
* only the expected warnings (`peb` -> `exact` for predefined_exp);
* analysis: 16 s.

Headline numbers (mini; 12 trajectories per cell):

* **H1, time-invariant problem, h = 0.3.**
  * Explicit Euler reaches 1e-13 in **0/12** trajectories for every activation; for predefined_exp it diverges (12/12).
  * Steps to 1e-13, median, in the order resolvent / PEB / exact / semi-implicit / TZ-EM-resolvent / TZ-EM-PEB:
    * sign: 23 / 23 / 23 / 23 / 23 / 23;
    * SBP(0.5): 22 / 15 / 16 / 25 / 22 / 15;
    * predefined_exp: 9 / 4 / 4 / 11 / 9 / 4.
  * TZ-EM reproduces the Euler step counts exactly (Lemma 2).
  * The exact-flow counts equal ceil(T(e0)/h) within +-1 in **100 %** of trajectories over all 9 h values. They stay within the uniform bound, e.g. SBP(0.5) at h = 0.01: 455 [447, 462] vs 484.
  * Explicit SBP(0.5) settles on the symmetric 2-cycle with MSSRE 0.011260 (IQR width < 1e-15). The predicted value is 2 (a h/2)^2 = 0.01125, with a = 1/2, p = 1/2 and m = 4 components.
* **H2, chattering metrics, time-invariant problem, explicit Euler, h = 0.3.**
  * sign: norm-based 0.32 vs `chatter_vec` 1.83 (`rate_alt` 1.0);
  * SBP(0.5): norm-based **0** vs `chatter_vec` 2.0 (`rate_alt` 1.0).
* **H3, time-varying problems at h = 0.3**, median paired log10 ratio [95 % bootstrap CI], tau = 1e-3 (linear system) and 3e-3 (matrix inversion):
  * explicit Euler / Euler + exact flow:
    * linear system, sign: 4.54 [4.43, 4.83];
    * linear system, SBP: 3.00 [2.87, 3.31];
    * matrix inversion, sign: 4.07 [3.77, 4.14];
    * predefined_exp: explicit diverged (12/12) on both problems; matrix inversion, SBP: 9/12 diverged.
  * explicit TZ / TZ-EM: linear system, sign: 7.20 [7.03, 7.29]; SBP: 6.07 [5.81, 6.13].
  * naive TZ / TZ-EM (ablation): 3.1 to 7.1 decades across activations and problems.
  * `chatter_vec`, explicit Euler / exact flow: 2.6 to 2.9 decades.
  * Empirical orders (asymptotic = all-points slopes here, 2-3 tau points):
    * consistent Euler schemes 2.00-2.08;
    * TZ-EM schemes 3.00;
    * explicit Euler / TZ with sign: -0.01 to 0.05 (h-floor);
    * naive TZ: 0.0 (sign) and 1.0 (SBP, predefined).
  * Kendall tau_b between the scheme rankings on linear_system and matrix_inversion: 0.99-1.00 (p < 1e-3). Between tau = 1e-2 and the smallest tau: 1.00.
* **H4, step-size range.**
  * Every consistent scheme is bounded and contracting on the whole grid [0.01, >= 10], on all problems. (For sign on matrix_inversion at tau = 3e-3 the contracting block starts at 0.03, because of the slow transient at gamma = 3.3.)
  * Explicit Euler, on the time-invariant problem and the linear system: contracting for h <= 0.1 (sign, SBP) and h <= 0.03 (predefined_exp); bounded for h <= 0.1 (SBP) and h <= 0.03 (predefined_exp). Sign stays bounded for every h; on matrix_inversion it contracts up to h = 0.6, with the O(h) floor.
  * The naive TZ stays bounded but is inaccurate (see H3).
* **H5, extension** (sign, h = 0.3, asymptotic slopes):
  * explicit: euler -0.01, taylor_zhang 0.01, zead_5i 0.01, uhlig_3_4 0.17 (no convergence in tau);
  * EM + exact: 2.00, 3.00, 4.02, 5.00, equal to the formula orders 2 / 3 / 4 / 5.

**Figure inspection** (all PNGs read). Defects found during inspection and fixed:
* TI traces: the exact zero is now drawn at the axis floor with a tick labelled "0";
* diverged medians now leave the panel instead of reappearing as survivor medians;
* in the TV traces, the shared y-limits had clipped the TZ-EM curves;
* legends overlapped data in the chatter and formulas figures;
* divergence markers collided with panel titles;
* in the order figure, 9 series per panel violated the <= 8 rule; it now has Euler and TZ rows;
* the extension suptitle was cut off.

What the final figures show:
* consistent Euler curves of one family coincide after the transient (by design: their residual is the one-step truncation error), and the caption says so;
* MSSRE vs h is flat for the consistent schemes and grows like h for the explicit ones until divergence (markers above the panel).

### 4.1 Error-unit arm (review 2026-10-18; `SUMMARY.md` section (d), `fig_E8_scale_ti`, `fig_E8_scale_tv`, `E8_scale.tex`)

Mini: sign, SBP(0.5), predefined_exp; s ∈ {0.01, 1, 10, 1000}; 3 instances x 4 initial states.

* **Unit-free (hold at every s):**
  * no consistent trajectory diverged: 0/5184 on the time-invariant problem over every h (and 0 on the time-varying problem);
  * exact-flow step counts equal ceil(T(s e0)/h) in **100 %** of trajectories over all (activation, s, h) (when that exceeds the 2500-step horizon, the run does not reach 1e-13, as predicted: sign at s = 10, h ≤ 0.03 and s = 1000, h ≤ 3). For SBP and predefined-time they stay within the s-free uniform bound ceil(T_max/h) in 100 %; for sign the count grows like s e0/h (finite-time, not fixed-time);
  * explicit Euler reaches 1e-13 in 0 trajectories at any s;
  * the explicit band is exactly the period-2 orbit of the activation in the unit of E_s: measured MSSRE / (√m e*(h)/s) = **1.000** (range 1.000-1.000) over all bounded SBP / predefined-time cells (e* solves hΦ(e) = 2e; (h/4)² for SBP r = 1/2 up to the |e|² correction);
  * explicit-Euler divergence is predicted exactly by the scalar map iterated from s E0 (100 % agreement for every s).
* **Unit-dependent:**
  * the explicit-Euler bounded range of h on the time-invariant problem: SBP(0.5) h ≤ 1.5 / 0.1 / 0.01 / none and predefined-time h ≤ 0.3 / 0.03 / none / none for s = 0.01 / 1 / 10 / 1000; sign is bounded for every s and h;
  * the accuracy gain on the time-varying problem. At h = 0.3, τ = 1e-3 (median paired log10 explicit Euler / Euler + exact flow): sign 6.65 / 4.54 / 3.56 / 0.00ᵘ (s = 1000: transient not finished, T(s e0)/γ ≫ T), SBP(0.5) 5.00 / 3.00 / diverged / diverged, predefined-time 6.63 / diverged / diverged / diverged.
* **Collapse.** Over every (s, τ, h) cell with no explicit divergence and a settled transient, the median paired log10 MSSRE ratio explicit Euler / Euler + exact flow is one function of x = log10 ρ, ρ = √m δ(h)/(s F(τ)) (δ = T⁻¹(h), dead zone of the exact map in the unit of E_s; F = one-step truncation floor: 1.13e-5 at τ = 1e-3, 1.14e-3 at τ = 1e-2): y = log10(1 + 10^(x+c)) with one c for all s fits with rms **0.07 (sign, 47 cells), 0.09 (SBP, 20), 0.12 (predefined-time, 12)** decades, against 1.38 / 0.86 / 0.91 when s is left out of ρ. Same for explicit Euler / resolvent (rms 0.07 / 0.13 / 0.14). For x ≤ 0 explicit Euler is as accurate or slightly more accurate: by 0.05-0.06 decades vs the exact flow and 0.10-0.17 decades vs the resolvent (SBP s = 1, τ = 1e-2, h = 0.03; SBP s = 10, τ = 1e-3, h = 0.01; predefined s = 1, τ = 1e-2, h = 0.01). This is the robotics reversal (R_units: implicit/explicit 1.2-2.1 in mm), reproduced and explained: it happens exactly when the band falls below the per-step drift.
* **Memory ablation.** naive TZ / TZ-EM at h = 0.3, τ = 1e-3 falls with s (SBP 5.63 / 3.62 / 2.56 / 0.65; predefined 6.81 / 4.76 / 3.83 / 1.85 decades for s = 0.01 / 1 / 10 / 1000) but is not a function of ρ alone (collapse rms 0.49-1.26): for ρ ≫ 1 the naive scheme fails through its marginal dead-zone recursion, and in 5 settled cells with ρ ≤ 3 (s = 1000, τ = 1e-2, large h) it is bounded but not contracting (median MSSRE up to 141-159 in the problem unit) while TZ-EM stays at the one-step floor. The naive scheme is therefore not a safe scheme at fine units and large h, whatever its accuracy at the problem's unit.

**Scale-free reading of E8** (replaces "implicit is k decades more accurate"):
1. consistent schemes are bounded for every h, unit and initial error, and reach exact zero in ceil(T(s e0)/h) steps on time-invariant problems, bounded by the unit-free ceil(T_max/h) for fixed-/predefined-time activations; explicit Euler never converges, and its stable h window scales with the unit (it is a property of hΦ(s e0)/(s e0), not of the scheme);
2. the steady-state accuracy advantage on time-varying problems is ≈ c ρ for ρ ≫ 1 and ≈ 1 (no advantage, explicit at most 0.17 decades better) for ρ ≲ 1, where ρ is the band (dead zone) in the problem's unit divided by the one-step truncation floor; the numbers in §4 are for ρ ≫ 1;
3. error memory is needed for boundedness at large effective gain as well as for the order.

## 5. Full-scale cost estimate and commands

Counts (from the full build, written to a scratch folder and not launched):

| config | specs | tasks (chunk) | trajectories | simulate steps | est. core-hours |
|---|---|---|---|---|---|
| E8_ti | 540 | 60 (9) | 19 440 | 1.35e6 | 0.13 |
| E8_tv | 4 320 | 360 (12) | 155 520 | 3.09e7 | 3.25 |
| E8_scale_ti | 1 080 | 120 (9) | 38 880 | 2.70e6 | 0.31 |
| E8_scale_tv | 3 240 | 270 (12) | 116 640 | 9.29e6 | 1.07 |
| E8_formulas | 336 | 14 (24) | 2 688 | 7.39e5 | 0.03 |
| **total** | **9 516** | **824** | **333 168** | **4.50e7** | **4.8 simulate + ≈ 0.5 start-up (824 × ~2 s) ≈ 5.3** |

* Method: steps x measured mini `wall_per_step` per (config, problem, disc, activation) (`summary.json["cost"]`, computed by `analyze_E8.py`; activations absent from a mini config use the (config, problem, disc) median). Diverged runs stop early, so this is an upper bound for the simulation part; allow 50 % contingency on a shared machine (≈ 8 core-hours).
* Measured mini (review rerun): 1287 specs, 0.30 core-hours of simulate time, median 146 µs per step (batch of 36 trajectories); explicit / PEB ≈ 125-170 µs, exact map 140-610 µs, resolvent (80-step bisection) 0.7-1.5 ms per step.
* All 1287 mini run ids are full run ids (0 missing), so a full run started in a folder that already holds the mini results would skip them; the commands below use a fresh `results/E8`.
* **Peak RSS per task: 0.14 GB** (measured max 137 MB `ru_maxrss` over all mini chunks, error-unit chunks 128-137 MB; τ = 3e-4 traces add ≈ 12 MB). **Recommended `--workers 24`**: ≈ 15 min wall, < 4 GB RAM in total (far below the 200 GB cap); `--workers 16` ≈ 20-25 min.
* Disk: ≈ 0.2 GB (mini measured 33 MB for the base configs; the error-unit arm writes no traces).

Exact commands (from `znn-bench/`):

```bash
# build (5 configs + merged tasks file results/E8/tasks_all.jsonl: 9516 specs, 824 tasks)
python experiments/configs/E8.py --scale full --out results/E8
# launch under the memory guard (resumable)
python tools/memguard.py pool --tasks results/E8/tasks_all.jsonl --workers 24 --max-gb 200
# analyze
python experiments/analyze_E8.py --results results/E8 --figdir figures --tabdir results/tables

# mini validation (as rerun in the review)
python experiments/configs/E8.py --scale mini --out results/mini/E8
python tools/memguard.py pool --tasks results/mini/E8/tasks_all.jsonl --workers 3 --max-gb 40
python experiments/analyze_E8.py --results results/mini/E8 --figdir figures/mini --tabdir results/tables/mini
```

The scale can also be set as `ZNNBENCH_SCALE=mini|full` (default route `python -m znnbench.runner build experiments/configs/E8.py --out results/E8`, which writes the same per-config folders `results/E8/E8_{ti,tv,scale_ti,scale_tv,formulas}/`; launch each `tasks.jsonl` or concatenate them). The error-unit wrapper is imported from `experiments.tasks.R_common`, so the pool must run from `znn-bench/` (as above).

**Rebuild after zoo changes:** re-running the build re-filters the keys. New formulas enter `E8_formulas` as new chunks, and unchanged runs keep their run ids, so only the new chunks run.

## 6. Zoo / core observations (no zoo file edited)

No zoo bug was found in the components E8 uses. The following observations are for the core and zoo owners:

1. **`simulate` online statistics at round-off (core metric caveat, not a crash).**
   * When a trajectory converges exactly (the steady residual is 0.0, e.g. any Euler-based consistent scheme on the time-invariant problem), `chatter_vec` is NaN (0/0).
   * When it converges to rounding level (about 3e-16, EM-TZ), `chatter_vec` is about 1.5 and `rate_alt` about 0.53. These look like severe chattering but are rounding noise. `rate_alt` is also about 0.5 for the exactly converged schemes, because x_{k+1} - x_k = -J^{-1}E_k is rounding noise.
   * Suggestion for `simulate`: return NaN, or 0 with a flag, when mean||E|| over the window is below about 1e-12 x (1 + ||b||). E8 masks this in the analysis (`*_eff` columns).
2. **The runner does not export `wall_dynamics`** (returned by `simulate` since the "runner 4" change) or `steps_run` from `simulate`; it recomputes the latter. This is minor, and E8 does not need it.
3. **Expected fallbacks, warned once per process** (by design, consistent.md §6):
   * `peb` -> `exact` for `predefined_exp` (k2 = k3 = 0);
   * `exact` -> `peb` for `fixed_time` with k3 > 0.

   As a result, `peb_euler` = `exact_flow_euler` for those activations. This is documented here and in the table captions; it is not a bug.
4. The norm-based `metrics.chattering` behaves as its own comment says: it is blind to *symmetric* period-2 cycles (exactly 0 for SBP). For sign-ZNN it does not score 0, because the components alternate between magnitudes a and h - a; it underestimates the cycle by about 6x (0.32 vs 1.83).

**Status of the preparer's four zoo / core items (re-checked 2026-10-18, nothing under `znnbench/` edited):** (1) "no zoo bug found": confirmed for the components E8 uses, including `ScaledError` (from `experiments/tasks/R_common.py`, wraps E, dE/dt, solve_J and JT; `znn` and the consistent maps only use `ctx.E`, `ctx.dEdt` and `ctx.solve`); (2) `simulate` online `chatter_vec` = NaN at an exact-zero steady residual and rounding noise near 1e-16, `rate_alt` without a magnitude threshold (`znnbench/simulate.py` lines 234-262): **open** (core metric caveat; E8 masks it with the `*_eff` columns, R uses `JointStats.alt_active`); (3) the runner does not export `wall_dynamics`: **open**, minor, not needed by E8; (4) the `peb` ↔ `exact` fallbacks for predefined_exp (k2 = k3 = 0) and fixed_time (k3 > 0): by design, not a bug.

## 7. Confounds and how they are handled

* **Error unit (major; review).** All non-linear activations act on the raw error, so the explicit band, the dead zone of the consistent maps, the explicit stable h window and the transient length depend on the unit of E (equivalently, s rescales the activation's coefficients: SBP r gets s^(r-1)/2 and s^(1/r-1)/2, sign gets 1/s). Quantified by the error-unit arm (§4.1); every accuracy number of §4 is labelled "problem unit (s = 1)", and the scale-free reading is given in §4.1.
* **Equal nominal gain is not equal effective gain.** All activations share h = γτ, but near e = 0 the effective gain differs by orders of magnitude (sign: γ/|e|; SBP r: ≈ γ|e|^(r-1)/2; predefined_exp: ≈ 2γ/√|e|; linear tail terms differ too), so comparisons *across* activations at equal h are not comparisons of matched controllers (E4's job). The E8 claims are *within* an activation: explicit vs consistent discretization of the same ODE at the same h, which is fair; the matched-h gain is then governed by ρ (§4.1). The "best h" rows (explicit at its own best grid h) are an unequal-gain check.
* **Initial-error dependence.** Explicit divergence and the explicit "stable h range" depend on h Φ(s e0)/(s e0); the initial errors are reported (‖E(0)‖ median 8.2, max_i|e0_i| 3.45-13.7 on linear_system) and divergence is predicted exactly from E0 (100 %).
* **Transients in the steady window.** At small γ = h/τ (and for sign at fine units, where the finite settling time T(s e0)/γ grows like s) the transient can extend past 0.75 T. Such trajectories are flagged `unsettled` from the exact settling time and excluded from the stability, collapse and mechanism statistics (marked u / hollow markers); e.g. sign at s = 1000, τ = 1e-3.
* **Horizon censoring of step counts.** Steps beyond the 2500-step horizon are "not reached"; the theory check counts them as agreement only when ceil(T(s e0)/h) > 2500.
* **Threshold 1e-13 vs floor.** Converged Euler-based maps end at exactly 0 or ≤ 1.6e-16 and EM-TZ at ≤ 3.9e-16 (max over all TI runs of both arms), i.e. ≥ 250x below the threshold, while the smallest explicit band on the grid ((0.01/4)²/1000 = 6e-9 for SBP at s = 1000) is far above it; the step counts are therefore threshold-insensitive for the exact/PEB maps (verified: they equal the exact-flow prediction). For the resolvent and semi-implicit maps of continuous activations (super-linear, not finite-step convergence) the counts do depend on the threshold, by design (§3, item 6).
* **Pseudo-replication.** Handled by the cluster bootstrap and instance-level sign tests (§2); 10 instances at full scale.
* **Information.** All schemes get the exact ∂E/∂t (`info: analytic`) and sample only at t_k (causal); no scheme has an information advantage.

## 8. Review log (2026-10-18)

Methodology review of E8 (reviewer: statistics / ZNN-discretization). Fixed in the E8 files only (`experiments/configs/E8.py`, `experiments/analyze_E8.py`, this note, `results/mini/E8`, `figures/mini/fig_E8_*`, `results/tables/mini/E8_*`); nothing under `znnbench/` or `tests/` was touched.

1. **Major, fixed: unit dependence of every headline.** The accuracy headline ("explicit sign-type Euler is 3-4.5 decades less accurate than the consistent schemes at h = 0.3"; memory ablation 3-7 decades) rested on one arbitrary error unit. Added the paired error-unit arm `E8_scale_ti` / `E8_scale_tv` (`R_common.ScaledError`, s ∈ {0.01, 10, 1000} plus the base s = 1; config, analysis `analyze_scale`, `fig_E8_scale_ti`, `fig_E8_scale_tv`, `E8_scale.tex`, SUMMARY section (d), interpretation). Effect: the accuracy gain at h = 0.3, τ = 1e-3 moves with s by ≈ 1 decade per decade of s (SBP: 5.00 / 3.00 / diverged / diverged; sign 6.65 / 4.54 / 3.56 / unsettled), and the whole matched-h data set collapses onto one curve in ρ = √m δ(h)/(s F(τ)) (rms 0.07-0.12 decades, 0.86-1.38 ignoring s). For ρ ≲ 1 explicit Euler is as accurate or up to 0.17 decades more accurate (robotics reversal reproduced). Unit-free and therefore the headline now: boundedness at every h and unit (0 consistent divergences), finite-step exactness with step counts = ceil(T(s e0)/h) (100 %) bounded by the s-free ceil(T_max/h), the explicit band = √m e*(h)/s exactly (ratio 1.000). Cost +486 mini specs (+45 tasks, 221 s wall), +4320 full specs (+1.4 core-hours).
2. **Major, fixed: pseudo-replication in the CIs and tests.** The 12 trajectories were bootstrapped as exchangeable, but the 4 initial states of an instance give identical consistent-scheme MSSRE and the between-instance sd is 3-8x the within-instance sd. Now: two-stage cluster bootstrap (instances, then states), sign tests on per-instance medians, Holm within each comparison family (one family over ~100 rows would have made even 10/10 instance wins non-significant), and **10 instances at full scale** (3 → 10 seeds for E8_ti, E8_tv and the error-unit arm; seeds 0-2 keep the mini run ids). Effect on mini numbers: point estimates unchanged, CIs widen slightly (e.g. 4.54 [4.43, 4.83] → [4.39, 4.87]); all sign tests are now non-significant at mini scale (3 instances, smallest p = 0.125), which is the honest statement; full scale can reach p ≈ 0.012 after Holm.
3. **Minor, fixed: asymptotic-range tolerance** 0.3 → 0.15 (checklist item 3). Effect: no fixed-h order fit of the mini run changed (no pair of local slopes has a spread in (0.15, 0.3]); 2-point fits are counted and not interpreted, as before.
4. **Minor, fixed: censoring bookkeeping.** Diverged runs already entered every statistic as +inf (checked every `isfinite` / NaN filter: `_paired_lr` drops only pairs where both members diverged, which never happens for explicit vs consistent; `best_h` excludes diverged cells by definition). Added `n_failed` per cell, a censoring statement and the failed-run count to SUMMARY.md (0 failed in the mini run).
5. **Minor, fixed: figure caption claim.** `fig_E8_steps` said the naive TZ "never" reaches 1e-13; it does for SBP r = 0.75 at small h (22 % of its TI trajectories, where the dead zone is ~1e-12 and the map acts almost linearly). The caption is now computed from the data. `fig_E8_scale_ti`: censored markers of different s offset so they do not overlap.
6. **Checked, sound: steps to exact convergence.** Threshold 1e-13 is ≥ 250x above the measured floor (≤ 3.9e-16, both arms) and far below any explicit band; t_eps uses "stays below", record_every = 1. Theory: implicit Euler (resolvent) of e' = -γ sign(e) is the soft threshold e ↦ sign(e) max(|e| - h, 0), exactly zero once |e| ≤ h = γτ, i.e. after ceil(max_i|e0_i|/h) steps: matched in 100 % of sign trajectories (resolvent, PEB, exact, semi-implicit, both TZ-EM). Exact flow for SBP / predefined: 100 % exact agreement with ceil(T(e0)/h) (sbp0.75: 91 % exact, 100 % within ±1, rounding of T(e0)/h at integer boundaries); fixed_time is excluded (its exact map falls back to the PEB split, which is ±1). Resolvent / semi-implicit maps of continuous activations need 7-17 extra steps (super-linear, not finite-step), as documented.
7. **Checked, sound: chattering metrics.** `chatter_vec`, `rate_alt` and the norm-based index behave as described; the round-off masking (`*_eff`) is needed and documented (zoo item 2, open). Both are s-free (ratio of E norms; x increments).
8. **Checked, partly qualified: naive implicit TZ ablation.** It demonstrates what the notes claim at the problem's unit (order lost, never exact on TI, 3-7 decades worse except SBP 0.75), but the size of the effect is unit-dependent and, at fine units and large h, the naive scheme is bounded but not contracting (MSSRE up to ≈ 150 in the problem unit) while TZ-EM stays at the floor (§4.1). Text updated accordingly.
9. **Minor, open (documented):** the longtable versions of `E8_steps`, `E8_stable_h`, `E8_orders`, `E8_headline_full` are 66-245 pt wider than an `article` text width; they need the supplement's landscape layout (not conclusion-relevant).
10. **Kendall τ_b** p-values were already labelled descriptive; unchanged.

Checked and found sound: config fidelity to PLAN.md (activations, schemes, problems, metrics; deviations listed in §3), causality (all schemes sample t_k only), batched == unbatched gains, pairing (identical initial residual across schemes and units), exact divergence prediction, the 2-cycle amplitude check, cost and memory estimates (re-measured), memguard task format and resumability (69 tasks reused bit for bit).
