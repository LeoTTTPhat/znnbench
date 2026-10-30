# E9 — GNN vs ZNN vs prediction–correction vs Uhlig vs Newton at matched cost

Protocol notes, deviations, mini-run results and the full-scale cost estimate for experiment E9 of
ZNN-Bench (PLAN.md §E9). Two methodological reviews on 2026-10-18 changed the protocol: the first
(section 7) the analysis and part of the GNN grid, the second (section 8) the cost accounting, the
seed count, the τ grid at the budgets and the censoring rules. Every number below comes from the
re-run after the second review.

Files owned by E9:

| file | role |
|---|---|
| `experiments/configs/E9.py` | runner configs (`get_configs()`), one per method group; scale via `E9_SCALE` |
| `experiments/tasks/E9_build.py` | builds every E9 config plus the cost task into one `tasks_all.jsonl` |
| `experiments/tasks/E9_cost.py` | memguard task: per-step cost profile (calls and FLOPs per step) |
| `experiments/analyze_E9.py` | figures, LaTeX tables, `summary.json`, `SUMMARY.md` |

## 1. Question and hypotheses

The question: at the same compute budget, how do these methods compare?
* the gradient neural network (GNN);
* linear ZNN with causal look-ahead formulas (Euler, Taylor–Zhang, ZeaD/Uhlig of orders 4–6);
* the prediction–correction (PC) schemes of Simonetto et al. (IEEE TSP 2016);
* Newton tracking.

The hypotheses:
* **H1**: GNN lags behind x*(t), with a residual floor of O(1/γ).
* **H1b**: the explicit Euler step of GNN is stable only if η = γ τ σ_max(J)² < 2.
* **H2**: ZNN ≈ PC ≈ Uhlig at matched cost (equivalence). The second review split it into three
  falsifiable parts (section 4): H2a, ZNN-Euler (h = 1) ≡ causal PC-Newton (C = 1) at equal τ; H2b, their
  matched-cost gap equals 2 log₁₀(cost ratio); H2c, across orders the look-ahead formulas are *not*
  equivalent (higher order wins at matched cost inside its stable, quasi-static τ range).

## 2. Exact protocol

**Problems.** All have n_x = m = 8, κ = 10, frequency ω = 1 rad/s and T = 6 s. The steady window is
the last 25 %, [4.5, 6] s. There is one instance per seed (`"seed": "$seed"`).

| short | registry key | kwargs | structure |
|---|---|---|---|
| `linsys` | `linear_system` | n = 8, κ = 10 | affine, J = A(t) |
| `tvuo` | `tv_unconstrained_opt` | variant `manufactured`, n = 8, κ = 10, β = 1 | nonlinear gradient field, E = P e + β tanh e |
| `tvqp` | `tvqp_eq` | variant `random`, n = 6, p = 2, κ = 10 | affine KKT system; J = K(t) is symmetric indefinite (σ_min ≈ 0.09) |

**Methods and gains.** Each method group is its own runner config, because the gains differ.

* `E9_znn_<formula>`: model `znn` (linear) with a causal look-ahead formula, in fixed-h mode.
  * Gains are h ∈ {h_opt, h_opt/2}, batched.
  * h_opt minimises the spectral radius ρ of the formula's frozen-coefficient error recursion
    (`zead.lookahead_stability`, computed at build time). h is the raw product γτ, as in the runner.
  * h_opt (h_max, ρ_opt) per formula:

    | formula | h_opt | h_max | ρ_opt |
    |---|---|---|---|
    | Euler | 1 | 2 | 0 |
    | Taylor–Zhang | 0.240 | 1 | 0.79 |
    | `zead_5i` | 0.0742 | 0.281 | 0.93 |
    | `uhlig_3_4` | 0.0255 | 0.0544 | 0.975 |
    | `uhlig_4_6` | 0.0930 | 0.176 | 0.91 |
    | `zead_7i_cang2024` | 0.0722 | 0.116 | 0.95 |

  * Formulas are picked conditionally from `znnbench.available()`: BASE + `zead_7i_cang2024` + any
    `uhlig_[5-9]_*` key. `discretizations/uhlig_high.py` registers nothing yet.
* `E9_gnn_<problem>`: GNN + Euler at a fixed *normalised* step
  η = γ τ σ̄² ∈ {0.5, 1.0, 1.5, **1.9, 2.1**, 2.5}.
  * σ̄ = max over the instances and t ∈ [0, T] of σ_max(J(x*(t), t)): 12.0 (linsys), 13.0 (tvuo, the
    bound σ_max(P) + β) and 12.10/12.24 (tvqp, mini/full seeds).
  * 1.9 and 2.1 were added in the review to bracket the limit tightly. 2.5 is the clear instability
    check. The primary level is η = 1.5.
* `E9_gnnlag`: GNN + Euler in fixed-γ mode at small τ.
  * mini: τ = 3e-4, γ ∈ {1, 3, 10, 30}.
  * full: τ = 1e-5, γ ∈ {1, 2, 5, 10, 20, 50, 100, 200, **300**, 500, **700**, 1000}. 300 and 700 were
    added for the quasi-static range of tvqp.
  * Every γ satisfies γτσ̄² ≤ 1.9 for every problem.
  * At full scale each spec is its own task (`chunk_size: 1`).
* `E9_pc`: model `davidenko` (x_dot = −J⁻¹∂E/∂t) as the predictor (Simonetto Eq. (7)).
  * Schemes: `pc_newton`, `pc_newton_causal`, `pc_gradient`, `pc_gradient_causal`, `newton_tracking`
    and `newton_tracking_causal`, each with C ∈ {1, 2, 3}.
  * `davidenko` + `euler` is the open-loop negative control.
  * The causal variants correct with t_k data and then predict. The non-causal variants (marked `[nc]`)
    predict and then correct with t_{k+1} data.
  * h = 1 is a placeholder (the gain is unused).

**Grid and statistics.**

| | mini | full |
|---|---|---|
| τ | {3e-2, 1.2e-2, 1e-2, 3e-3, 1e-3} | {1e-1, 5e-2, 3e-2, 2e-2, 1.2e-2, 1e-2, 7e-3, 5e-3, 3e-3, 2e-3, 1.2e-3, 1e-3, 5e-4, 3e-4, 1e-4} |
| seeds × B | 2 × 2 | **10 × 4** (second review; PLAN default 3 × 4) |
| GNN lag sweep | τ = 3e-4, `record_every` 1 | τ = 1e-5, `record_every` 10 |

* 1.2e-2 and 1.2e-3 (second review) bracket the two budgets of the look-ahead formulas, which run at
  τ = 1.04–1.17 × the budget τ. Without them the bracket reached the next grid point (3e-2 / 2e-2),
  where the order ≥ 5 formulas are unstable, and the comparison was n/a.
* 10 seeds (second review): the problem instance is the only real replication unit (see
  "Bootstrap" below), and a cluster bootstrap over 3 instances has only 10 distinct resamples.

* `record_every` is 1 and `timing_repeats` is 1 (one lean timing run) everywhere except the lag sweep.
  Traces are stored at 200 points per run.
* A cell is (problem, method, gain level, τ). Statistics per cell are the median and IQR over its
  n_seeds × B trajectories.
* **Effective status** (review fix). A closed-loop trajectory with status `ok` but
  final residual > initial residual counts as **diverged** (MSSRE = +∞) in every statistic. Such a
  trajectory grows without reaching the runner's 1e10 threshold on |x|. The open-loop Davidenko control
  is exempt, because its residual drifts around E(x0, 0) by design.
* **Pre-asymptotic cells.** A cell is pre-asymptotic when the median over its trajectories of
  q = max‖E‖ on [T/2, 3T/4) / max‖E‖ on [3T/4, T] is > 3 (still decaying) or < 1/3 (growing).
  q comes from the traces.
  * GNN cells also need a gain γ = η/(τσ̄²) that is **transient-free** (E0 e^{−γσ_min² t_ss} <
    0.1 c_lag/γ) **and quasi-static** (γσ_min² ≥ 3ω). The ratio q misses the slow GNN transients.
  * Pre-asymptotic cells are excluded from order fits and drawn with a white fill.
* **Bootstrap**: 2000 resamples with seed 20261018, hierarchical (seeds with replacement, then
  initial states within each seed).
  * It is paired across methods and τ, because x0 depends only on (problem, seed) (checked: the
    initial residuals of one (problem, seed, x0 index) agree across all methods to 2e-16).
  * CIs are order-statistic percentiles, which stay valid with +∞ resamples.
  * **Pseudo-replication (second review).** The B initial states of a seed reach the same steady
    orbit: the median within-seed spread of MSSRE is 3.5e-14 decades (90th percentile 0.11, only in
    transient-limited cells), against a median across-seed spread of 0.19 decades. The seed is the
    unit of replication, and the bootstrap resamples it first. With 2 (mini) or 3 seeds the
    percentile interval is essentially the seed range; it is labelled descriptive, and the full scale
    uses 10 seeds.
* **Kendall τ_b**: values below 10 × 1e-12 are tied (round-off floor). +∞ (diverged or unstable at
  τ_fix or at a bracketing τ of the budget) is kept as tied-worst; only methods without a matched value
  (budget outside the sampled τ range) are left out, and they are listed. τ_b is reported for all
  methods and for the causal methods only. p-values are descriptive.

**Cost model (primary: algorithmic FLOPs per step per trajectory, B = 1).**
* Source: `znnbench.costs.step_flops(problem, model, disc, B=1)`, the structured model, which counts
  what the code does. The fallback is `profile_step_cost` counts × `dense_flops` plus vector updates.
  Both are stored in `cost/cost_profile-<hash>.json`.
* **Algorithmic view (second review, `COST_VERSION` 3).** `step_flops` skips the dE/dt that simulate
  evaluates for GNN (which ignores it), but it still charges the E that simulate evaluates for the
  Davidenko predictor (which ignores it). `E9_cost.py` removes that E call (136 / 160 / 134 FLOPs on
  linsys / tvuo / tvqp) from every Davidenko-based scheme, symmetrically with the dE/dt skip. The
  correction is applied only while `step_flops` still counts the call, so an upstream fix is not
  subtracted twice. Causal PC-Newton C = 1 now costs 1160 FLOPs on linsys (code view 1296), against
  ZNN-Euler 750. The code view is kept as a sensitivity (`flops_code_view`).
* Data generation (sensor acquisition, including the analytic derivative data Ȧ, ḃ) is free. The
  dE/dt algebra is charged to every method that uses it. Under `bd1` the dE/dt algebra would be replaced
  by one more E evaluation of the same cost, so the FLOP accounting is invariant to the information
  regime to first order; the accuracy is not (E6).
* Cross-check against `profile_step_cost` (second review): in all 78 (problem, method) entries the
  call counts of `step_flops` (E, dE/dt, solve_J, JT, data) equal the measured steady per-step counts
  (difference of a 240- and a 40-step run). The only differences are GNN's dE/dt (3 entries), which
  `step_flops` skips by design. For the Davidenko predictor `step_flops` charges the measured but unused
  E call, which E9 removes (above). Per-call FLOPs were checked by hand for linsys: E = 2n² + n = 136,
  GETRF(8) = 308, GETRS(8, 1) = 120, ZNN-Euler = 136 + 136 + 428 + 18 + 32 = 750.
* **Sensitivity (first review): ideal LU reuse.**
  * The structured model charges one factorisation per `solve_J` call, because `np.linalg.solve`
    refactorises every time. On the affine problems J does not depend on x, so all solves at one data
    sample could share one LU factorisation.
  * The variant `flops_lu_reuse` charges one factorisation per **new** data sample, i.e. one per step
    for every solve-using method. (Second review: the non-causal PC-Newton was charged 2, although its
    prediction at t_k uses the sample its previous step corrected at.)
  * This affects only PC-Newton and Newton tracking with C ≥ 1 on linsys and tvqp. Example: causal
    PC-Newton C = 1 costs 1160 → 852 FLOPs on linsys, against ZNN-Euler 750.
* The secondary axis is the measured lean wall time per step, single BLAS thread, per *batch*. At
  n = 8, Python overhead dominates it (≈ 250–800 µs/step against a few µs of arithmetic). The batch size
  barely matters (B·G from 2 to 48 changes it by < 3 %, measured). So the wall axis reflects call
  structure, not FLOPs. **It is descriptive only** (second review, after the E7 finding): it is timed
  inside a parallel memguard pool on a shared machine (load 40–180 from other projects), where
  Accelerate's per-cluster matrix units slow BLAS-3 calls by up to +118 % under concurrency while
  CPU/wall stays 1.00. No verdict uses it; the figure axis and the table caption say so.

**Matched cost.**
* Work–precision curves plot median MSSRE against the FLOP rate R = FLOPs/step / τ, i.e. FLOPs per
  tracked second.
* There are two budgets, R* = R(ZNN-Euler, h = 1) at τ = 1e-2 and at τ = 1e-3.
* Interpolation (review fix): log-log, between the **two adjacent grid points that bracket R\***.
  There is no extrapolation and no bridging over a missing, diverged or unstable grid point.
  * A bracketing median of +∞ gives "n/a (diverged or unstable at a bracketing τ)".
  * In the bootstrap, a resample with a +∞ bracketing median counts as log-ratio +∞, so the CI can be
    open on the right.
* Reported: log₁₀(MSSRE / MSSRE_ZNN-Euler) with its 95 % CI.
* Verdicts (review fix):
  * *equivalent* when the CI ⊂ [−0.5, +0.5] decade. If the CI also excludes 0 the verdict reads
    "equivalent (but worse/better)".
  * Otherwise *better* (CI < 0), *worse* (CI > 0) or *inconclusive*.
  * *better (floor-limited)* when the method sits at the round-off floor (< 1e-11). The gain is then
    only a lower bound.
* Flags:
  * *transient-limited bracket*: a bracketing cell is still decaying, so the ratio is conservative.
  * Bonferroni variant: CI at 0.05/K with K = 75 comparisons per budget (25 methods × 3 problems).
    2000 resamples make its extreme quantiles approximate.
* Matching the FLOP rate credits cheaper methods with faster sampling. That presumes data are
  available at any rate for free. The fixed-τ figures (`fig_E9_fixedtau*`) give the complementary
  view at a fixed sensor rate.
* Sensitivities (second review): the code-view cost (unused Davidenko E charged), ideal LU reuse, and
  GNN at η = 1.9 instead of 1.5. For causal PC-Newton the analysis also prints the order-2 prediction
  2·log₁₀(cost ratio) next to each observed log-ratio (H2b).
* The work–precision figures mark both budgets as vertical dashed lines; the non-causal block of
  `fig_E9_equiv` and `E9_matched.tex` is shaded / separated.

**Empirical orders.** The fit is the slope of the median MSSRE against τ.
* Points below 1e-12 and pre-asymptotic cells are excluded.
* The asymptotic range is the longest run of ≥ 3 *consecutive* usable τ whose local slopes lie within
  **0.15** of their median (second review: the project tolerance of REVIEW_CHECKLIST item 3; it was
  max(0.5, 0.15·|median|), which accepted ranges whose local slopes differed by up to 1). Otherwise the
  two smallest consecutive usable τ are used (asterisk).
* The asymptotic slope and the all-points slope are both reported. A slope above expected + 3, or below
  0.5 for a method whose expected order is ≥ 1, is flagged `implausible` (mark `!` in `E9_orders.tex`):
  it sits on a pre-asymptotic transient that the trace ratio did not catch (e.g. PC-gradient on the
  KKT problem, which contracts by ≈ 1 − 3e-5 per correction).
* GNN lag slope (MSSRE vs γ): fitted only on transient-free **and** quasi-static γ.
  * The quasi-static condition was added in the review. A mode with rate a = γσ² tracks a sinusoid with
    error |ẋ*|/√(a² + ω²), so the local slope is −a²/(a² + ω²); it is −0.9 at a = 3ω.
  * c_lag = max over the steady window of ‖J^{−T} ẋ*‖ is the continuous-time lag constant
    (MSSRE ≈ c_lag/γ).

**Transient cost (review addition).** The time after which ‖E‖ < 1e-2 for good, at τ = 1e-2.
* It is computed from the traces, with a resolution of T/200 = 3 steps.
* Runs that never reach 1e-2, diverge or grow are right-censored at T. The upper median is reported,
  with "> T" when censored.

## 3. Deviations from PLAN.md (and why)

1. **Problem variants.**
   * `tv_unconstrained_opt` uses the nonlinear `manufactured` variant with n = 8. The default
     `zhang2021` has n = 2, is affine and has no seed. A nonlinear E is what separates Newton-PC from
     ZNN, and a gradient field lets `pc_gradient` use Simonetto's exact step.
   * `tvqp_eq` uses the `random` variant (n = 6, p = 2), so that n_x = 8 and it is seedable.
2. **Step-size modes.**
   * ZNN runs in fixed-h mode only; the fixed-γ order loss is E1's subject.
   * GNN uses a normalised fixed-η sweep (the analogue of fixed-h) plus a fixed-γ lag sweep.
3. **Order-4 Uhlig formula.** It runs as `zead_5i`, whose coefficients equal `uhlig_2_3`. `zead_5i`
   owns the colour slot in `keys.GROUPS['disc']`. The baseline-group figures show `uhlig_3_4`, the only
   Uhlig slot in `keys.GROUPS['baseline']`. Each figure uses a single style group.
4. **PC predictor**: Davidenko (faithful Eq. (7)), not the ZNN field.
5. **GNN step size**: reported explicitly as η, with the γ per τ in `summary.json`. The per-instance
   η_inst = η (σ_max,inst/σ̄)² ≤ η is in `summary.json → gnn.stability`.
6. **Mini scale**: 2 seeds × B = 2 and 5 τ values, to stay within the 10-minute budget.
   **Full scale: 10 seeds × B = 4** (second review; PLAN default 3 × 4). The seed is the only real
   replication unit (section 2, "Bootstrap"), and every headline is a cluster-bootstrap CI. The cost is
   ≈ 3.3× the 3-seed sweep, ≈ 12 core-hours (section 6).
7. **Full τ grid**: densified around 2e-3–2e-2, plus 1.2e-2 and 1.2e-3 at the two budgets (15 values).
   There the order-5/6 formulas sit between the round-off floor and their large-τ instability.
8. **T = 6 s kept** (PLAN default). This has a known consequence. The order ≥ 4 formulas contract by only
   ρ = 0.91–0.975 per step, so at τ ≥ 1e-2 some cells are still decaying inside the steady window
   (`uhlig_3_4` needs ≈ 300 steps to reach 1e-2). Those cells are flagged; they make the matched-cost
   ratios *conservative* for those formulas. A longer T would raise the cost by T/6 and move every steady
   window, so it was not adopted. The transient cost is reported instead (section 2).
9. **`zead_5i_mahuang2025`** (registered during the review) was considered and **not added**. It has
   the same order (4) and per-step cost as `zead_5i`, a smaller stable range (h_max 0.17 vs 0.28) and a
   slower contraction (ρ_opt 0.955 vs 0.928), so it adds no information on the matched-cost question.
   Faster-contracting members of the order-5/6 classes now exist in the zoo (`zead_8i_chen2019`
   ρ_opt 0.87, `zead_8i_wu2020` 0.83, both order 5; `zead_11i_chen2023` 0.93, order 6). They have no slot
   in `keys.GROUPS['disc']` and would mainly shorten the transients flagged 't'; they are left for a
   follow-up.

## 4. Mini-run results (`results/mini/E9/`, `figures/mini/fig_E9_*`, `results/tables/mini/E9_*.tex`)

All numbers below are from the re-run after the second review (section 8). Values in brackets are
cluster-bootstrap intervals over **2 instances**, i.e. essentially the instance range (descriptive).

**Run.**
* Build → `memguard pool --workers 3 --max-gb 40` → analyze.
* Second-review re-run: 70 tasks (the 6 GNN-lag tasks were unchanged and reused), 4 min 51 s wall with
  3 workers on a shared machine (load ≈ 40 on 32 cores from other jobs), 719 s of task time; exit 0, no
  failed task. Analysis ≈ 14 s.
* Peak RSS: 126.5 MB per task.
* 786 specs, 2268 trajectory rows:
  * runner statuses: 2232 `ok`, 36 `diverged`;
  * effective statuses: 2219 ok, 36 diverged, 13 finite-growth. The finite-growth rows are `uhlig_4_6`
    at τ = 3e-2 (linsys 8 over both gains, tvqp 2), `uhlig_3_4` and `zead_7i_cang2024` on tvqp at 3e-2
    (1 each), and GNN η = 2.5 (1).
  * 0 errors and 0 timeouts. The censoring table of `SUMMARY.md` lists every censored cell.

**Correctness checks (asymptotic orders, local slopes within 0.15).**
* ZNN-Euler 2.00 / 2.00 / 2.00; Taylor–Zhang 2.98 / 2.99 / 2.94; ZeaD-5i 3.97 / 3.97 / 3.86.
* `uhlig_3_4`: 4.94* / – / 4.71*. `uhlig_4_6`: – / – / 5.78. ZeaD-7i: 4.99* / – / 4.78. The mini grid is
  too coarse for orders 5–6; the full grid is not.
* Causal PC-Newton: 2.00 for every C. Causal Newton tracking: 1.00 for every C. PC-gradient: 2.00 on the
  gradient field (tvuo), 1.8 on linsys; on the KKT problem (tvqp) its 2-point slopes are pre-asymptotic
  and two are flagged implausible (0.33, 0.35).
* Non-causal PC-Newton and Newton tracking sit at the round-off floor on the affine problems, as the zoo
  spec predicts. On tvuo non-causal PC-Newton has slope 6.00: tanh has no quadratic term, so the Newton
  correction converges cubically and the O(τ²) prediction error becomes O(τ⁶) (Simonetto's O(τ⁴) is an
  upper bound).
* GNN τ-slope at fixed η: only tvuo has non-flagged cells, 0.99* (2 points). The other cells are
  transient-limited or not quasi-static at mini scale.

**H1 (GNN lag floor).** Slope of MSSRE vs γ on transient-free, quasi-static γ:
* linsys −1.00 [−1.00, −0.99] (γ ∈ {10, 30}, 2 points);
* tvuo −0.99 [−0.99, −0.98] (4 γ);
* tvqp n/a at mini scale. Its σ_min² ≈ 0.008 needs γ ≳ 400, which the full lag sweep reaches
  ({500, 700, 1000}).
* Measured / theory c_lag/γ = 1.02 (linsys), 0.95–1.00 (tvuo).
* Verdict: supported on linsys and tvuo (tvuo within the ±0.15 tolerance; its interval excludes −1 by
  0.01, which the finite-a correction −a²/(a² + ω²) predicts).

**H1b (GNN step limit).**
* No trajectory diverges or grows at η ≤ 1.9.
* η = 2.1: 10 % (linsys, tvqp) and 0 % (tvuo).
* η = 2.5: 60 / 45 / 60 %.
* So η < 2 is sufficient and the limit is sharp. η slightly above 2 need not destabilise every instance:
  σ̄ is the maximum over instances and time, so the per-instance η is smaller.
* Gains at τ = 1e-2, η = 1.5: γ = 1.04 (linsys), 0.89 (tvuo), 1.02 (tvqp).

**H2 (equivalence), stated so that it can fail** (also in `SUMMARY.md → Hypotheses`):
* H2a: ZNN-Euler (h = 1) *is* causal PC-Newton (C = 1) up to where E_t is evaluated; the equal-τ
  MSSRE ratio must be 1 + O(τ). Falsified by a ratio outside [0.9, 1.1] or a different order.
* H2b: with identical per-step accuracy at order 2, the matched-cost log-ratio must equal
  2 log₁₀(cost ratio).
* H2c: every look-ahead formula costs one solve per step and has residual order q + 1 at fixed h, so a
  higher-order formula must win at matched cost inside its stable, quasi-static τ range; "ZNN ≈ Uhlig"
  across orders is *predicted to fail*.

*ZNN-Euler vs causal PC-Newton (H2a, H2b): supported, exactly.*
* Equal-τ MSSRE ratio: 1.000 (0.997–1.000) linsys, 1.004 (1.000–1.012) tvuo, 1.000 (0.995–1.001) tvqp.
  C = 2 and C = 3 change nothing (extra Newton corrections at the same t_k sample are wasted work).
* Matched cost, budget 1 (τ = 1e-2), observed vs order-2 prediction:

  | cost view | linsys | tvuo | tvqp |
  |---|---|---|---|
  | algorithmic FLOPs (primary) | +0.38 (pred. +0.38) | +0.37 (+0.36) | +0.38 (+0.38) |
  | ideal LU reuse | +0.11 (+0.11) | +0.37 (+0.36) | +0.11 (+0.11) |
  | code view (unused E charged; pre-review primary) | +0.48 (+0.48) | +0.46 (+0.46) | +0.48 (+0.48) |

* So the matched-cost gap is entirely the cost ratio: PC-Newton is never more accurate per step, and its
  "equivalent (but worse)" verdict at the ±0.5 margin is an accounting statement. With C = 2 it is
  +0.73 dec (worse), with C = 3 +0.95 dec (tvuo; outside the mini τ range elsewhere).

*ZNN-Euler vs the high-order look-ahead family (H2c): equivalence fails, as predicted, wherever the
formula is stable and its transient has decayed.* At budget 1 (τ = 1e-2):

| method | linsys | tvuo | tvqp |
|---|---|---|---|
| `zead_5i` | −2.04 | −2.99 | −1.50 |
| `zead_7i_cang2024` | −2.99 t | −4.13 t | −2.24 t |
| `uhlig_4_6` | +0.14 [−2.79, +0.78] t, inconclusive | −6.23 t | −2.88 |
| `uhlig_3_4` | −0.48 t | −0.32 t | −0.27 [−0.82, +0.68] t, inconclusive |

* t = a bracketing cell is still decaying (conservative for the formula).
* `uhlig_4_6` is now evaluated on linsys and tvqp, where the bracket used to reach the unstable 3e-2
  point ("n/a"). On linsys it is inconclusive for a real reason: instance 1 has MSSRE 1.1–1.4e-2 at
  τ = 1.2e-2 (not growing, q ≈ 1.9) against ≈ 1e-5 at τ = 1e-2. Budget 1 sits at the edge of the
  formula's usable τ range, where τ‖J⁻¹J̇‖ is no longer ≪ h, the condition under which look-ahead formulas keep their order.
* At budget 2 (τ = 1e-3) every order ≥ 4 formula is better by 3.4–8.2 dec, and no bracket is
  transient-limited. `uhlig_4_6` (all three problems) and `uhlig_3_4` / ZeaD-7i (tvuo) reach the
  round-off floor there, so those gains are lower bounds.

*Other methods at budget 1.*
* GNN +2.58 / +2.77 / +2.84 dec (η = 1.5); at η = 1.9 +2.51 / +2.67 / +2.83. Causal Newton tracking
  +1.74 / +2.07 / +1.50, causal PC-gradient +0.5 to +2.1 dec. All are worse than ZNN-Euler.
* The non-causal PC-Newton / Newton tracking win by up to −11 dec only through t_{k+1} data. That is an
  information advantage, and both are shown in a separate block.

*Robustness of the H2 verdicts.*
* Multiplicity: 0 of the matched-cost verdicts change with Bonferroni CIs (K = 75 per budget).
* Kendall τ_b (fixed τ vs matched cost; floor values tied, +∞ tied-worst; descriptive): 0.90 / 0.90 /
  0.93 for all methods, 0.85 / 0.89 / 0.92 for causal methods only. Only PC C = 3 on linsys/tvqp is
  excluded (budget outside the mini τ range).
* The best causal method is the same in both views: ZeaD-7i (linsys) and `uhlig_4_6` (tvuo, tvqp).

**Transient cost at τ = 1e-2** (upper-median steps to ‖E‖ < 1e-2 for good):

| method | steps |
|---|---|
| Euler | 3 (trace resolution) |
| Taylor–Zhang | 33 |
| `uhlig_4_6` | ≈ 80 |
| ZeaD-5i / ZeaD-7i | ≈ 100 |
| `uhlig_3_4` | ≈ 295 |
| GNN, Newton tracking, Davidenko | never (> T) |

The accuracy gain of the high-order formulas is paid for with a ρ^k transient. Euler at h = 1 is
deadbeat.

**Caveats found in the data.**
* **Finite instability of high-order formulas at large τ.** These are counted as diverged. The cause is
  that the frozen-coefficient interval (0, h_max) assumes τ‖J⁻¹J̇‖ ≪ h.
* **`pc_gradient` on the KKT problem.** Jᵀ E with α = 0.5/σ_max² contracts by ≈ 1 − 3e-5 per correction.
  Simonetto's GTT assumes strong convexity, which the saddle-point problem violates; this is an
  extension and not a fair test of the published method.
* **Davidenko** keeps E(x0, 0): open-loop drift, MSSRE ≈ 12–18 at every τ.
* **Few clusters.** With 2 (mini) instances the seed level of the bootstrap has 3 distinct resamples; the
  intervals are instance ranges. The full scale uses 10 instances (92,378 distinct resamples). The
  narrow intervals of the deterministic ZNN-vs-PC comparison reflect the pairing, not population
  precision.

## 5. Zoo issues found (not fixed here, by rule)

Status re-checked in the second review (2026-10-18 18:30): issue 1 is **still open** (`wrappers.py`
dated 12:23) and does not affect E9's profiling. Issue 2 was **fixed upstream at 18:27** (`costs.py`
now has `skip_unused_E`, default = the algorithmic view); E9's own correction recognises the fix and
the re-profiled totals are identical in all 78 entries.

1. **BUG, `znnbench/wrappers.py:696` (`profile_step_cost`): confirmed in review, still open.**
   * It normalises by `n = max(int(sim["steps"]), 1)` instead of `sim["steps_run"]`.
   * When every trajectory diverges, `calls_per_step`, `wall_per_step` and `flops_per_step_dense` are
     underestimated by steps_run/N, with no warning.
   * Repro: `profile_step_cost(make_problem('linear_system', n=8, kappa=10), make_model('gnn'),
     make_discretization('euler'), tau=1e-2, steps=200)` uses the default γ = 10 (η = 14.4) and returns
     0.07 calls per step; γ = 0.5 returns 1.0.
   * Fix: divide by `max(sim["steps_run"], 1)` and warn if `sim["diverged"].any()`.
   * E9 is unaffected: it profiles at stable gains and takes the difference of two run lengths.
2. **Inconsistency, `znnbench/costs.py::step_flops`: fixed upstream at 18:27 (was corrected inside E9).**
   * `skip_unused_dEdt=True` drops the unused dE/dt of GNN, but ctx.E is still charged for models that
     ignore E (`davidenko`): 740 vs 604 FLOPs for Davidenko + Euler on linsys n = 8 (the E call is
     2n² + n = 136 FLOPs; the earlier "629" was the dense fallback). Every Davidenko-based PC scheme
     carries one E call extra: 136 / 160 / 134 FLOPs on linsys / tvuo / tvqp, 6–18 % of its code-view step.
   * Effect on E9 before the correction: causal PC-Newton C = 1 looked 0.10 decades worse at matched cost
     than it is (+0.47 instead of +0.38 dec on linsys).
   * Fix adopted upstream: `skip_unused_E` (`uses_E = False` on `davidenko`; default follows
     `skip_unused_dEdt`). `E9_cost.py` detects whether `step_flops` still counts the call and corrects
     only then; with the fixed library it takes the total as is and computes the code view with
     `skip_unused_E=False`. Old and new library give identical E9 totals (78 / 78 entries).
3. **Modelling note, `znnbench/costs.py`.** For `linear_system` / `tvqp_eq` (`factor_mode='call'`), every
   `solve_J` is charged a fresh LU, as the code does. Multi-solve schemes at one data sample (causal
   PC-Newton, Newton tracking with C > 1) would be cheaper with a factorisation cache. A
   `factor_mode='cached'` option, or caching in the problems, would make E7/E9 algorithmic rather than
   implementation costs. E9 reports both (section 2).
4. **Not a bug, but analyses must know it.** The runner's status is `ok` for finite growth below
   `div_threshold` (1e10 on |x|). E9 counts such closed-loop trajectories as diverged.

## 6. Full-scale cost estimate and commands

Updated in the second review (10 seeds, 15 τ values). Counts verified by a dry build
(`E9_build.py --scale full --out <scratchpad>`; nothing was launched).

* **Runs: 11,730 specs in 1,011 memguard tasks** (1,010 sweep tasks + 1 cost task), 68,040 trajectories:
  * ZNN: 6 formulas × 3 problems × 15 τ × 10 seeds = 2,700 specs (38 tasks per formula), each batching
    2 gains × B = 4;
  * GNN: 3 × 15 × 10 = 450 specs (13 tasks per problem), each batching 6 η × B = 4;
  * GNN lag: 30 specs × 48 trajectories (12 γ × B = 4), one task each (30 tasks);
  * PC: 19 schemes (18 PC/Newton + open-loop Davidenko) × 3 × 15 × 10 = 8,550 specs (713 tasks), B = 4;
  * the cost task (≈ 10 s, 78 entries).
* **Steps:** 1.05e8 simulated steps (ZNN 2.01e7, GNN 3.36e6, lag 1.80e7, PC 6.37e7); ≈ 1.9e8 executed
  steps including the lean timing re-runs (`timing_repeats` 1 everywhere except the lag sweep).
* **Measured cost per simulated step** (mini re-run, metric + lean + setup, per batch, single BLAS
  thread). It was taken under load ≈ 40 on 32 cores and is within 0–6 % of the earlier quiet-machine
  measurement:

  | config | s/step |
  |---|---|
  | ZNN | 4.38–4.54e-4 |
  | PC | 4.84e-4 |
  | GNN | 2.5e-4 (linsys), 3.2e-4 (tvuo), 7.9e-4 (tvqp) |
  | lag batch of 48, `record_every` 10 (measured directly, first review) | 115 µs (linsys), ≈ 140 µs (tvuo), 318 µs (tvqp) |

  The batch size does not change these (< 3 %, measured in the first review), so more seeds cost
  linearly and more initial states almost nothing.
* **Estimate: ≈ 12.5 core-hours** (PC 8.6, ZNN 2.5, lag 0.95, GNN 0.42); the quiet-machine rates give
  12.0. Allow +30 % for machine load, i.e. ≤ 16 core-hours.
  * These rates are wall-clock inside a pool. E9's step is interpreter-bound (a few µs of arithmetic in
    250–800 µs), so the BLAS-3 contention found in E7 barely touches it (rates under load ≈ 40 were
    within 0–6 % of the quiet run). Oversubscription by other projects (load 40–180 on 32 cores) can
    still stretch the *wall* time several-fold; the core-hours are the robust figure.
* **Recommended: `--workers 24 --max-gb 200`** (the machine has 32 cores and 512 GB).
  * Expected wall ≈ 35 min on an otherwise idle machine; the mean task takes ≈ 45 s.
  * The critical path is a tvqp lag task: 6e5 steps × 318 µs ≈ 3.2 min.
  * If other jobs are running, use at most (32 − their load) workers; memguard enforces the memory cap.
* **Peak RSS per task: 0.13 GB measured** (mini, 126.5 MB), **≤ 0.3 GB expected at full scale**.
  * Largest arrays: lag 6e4 recorded × 48 trajectories × 2 traces × 8 B ≈ 46 MB; τ = 1e-4 GNN
    6e4 × 24 × 2 × 8 B ≈ 23 MB. Chunks run their specs sequentially, so memory does not add up.
  * So 24 workers stay below ≈ 7 GB, far under the 200 GB cap.
* **Analysis** (estimate): a few minutes and < 2 GB for ≈ 68k trajectory rows and ≈ 1,000 trace files.

**Commands** (run from `znn-bench/`):

```bash
# build (picks up any zoo keys registered since; unchanged chunks keep their ids; the cost task's
# done file hashes the source of znnbench/costs.py, so a fixed cost model is re-profiled)
python experiments/configs/E9.py --scale full                 # optional summary
python experiments/tasks/E9_build.py --scale full --out results/E9
# launch (resumable)
python tools/memguard.py pool --tasks results/E9/tasks_all.jsonl --workers 24 --max-gb 200
# analyze
python experiments/analyze_E9.py --results results/E9 --figdir figures --tabdir results/tables
```

Mini (validation, re-run in the second review):

```bash
python experiments/tasks/E9_build.py --scale mini --out results/mini/E9
python tools/memguard.py pool --tasks results/mini/E9/tasks_all.jsonl --workers 3 --max-gb 40
python experiments/analyze_E9.py --results results/mini/E9 --figdir figures/mini --tabdir results/tables/mini
```

* The scale is set with `--scale` in `E9_build.py`.
* Running the runner CLI on `experiments/configs/E9.py` directly reads `E9_SCALE` (fallback `ZNN_SCALE`,
  default `mini`). That builds per-config folders only, without the cost task.

**Outputs.**
* Figures (each `.pdf` + `.png`, DOUBLE width): `fig_E9_workprec_znn`, `_workprec_base`,
  `_workprec_wall`, `_tau_znn`, `_tau_base`, `_fixedtau`, `_fixedtau_znn`, `_pc`, `_gnn`, `_equiv`,
  `_traces`.
* Tables: `E9_cost.tex`, `E9_orders.tex`, `E9_matched.tex` (causal block, non-causal block, LU-reuse
  block and both Kendall rows), `E9_gnn.tex`.
* Summaries: `<results>/summary.json` (strict JSON) and `<results>/SUMMARY.md` (matched-cost definition,
  statistics note, headline numbers, censoring table, confounds, falsifiable hypotheses, tables).

## 7. Methodological review (2026-10-18): issues found and fixes

| # | severity | issue | fix |
|---|---|---|---|
| 1 | critical | Matched-cost interpolation dropped non-finite points and then interpolated across the gap, and it used finite-growth cells (1e9) as ordinary values. Example: `uhlig_4_6` on linsys was "−0.42 [−2.87, +0.21], inconclusive", interpolated towards an unstable 3e-2 cell. | Finite growth of a closed-loop method counts as diverged (+∞). The interpolation uses the adjacent bracketing grid points only. +∞ brackets give n/a, or +∞ bootstrap resamples (open CI). |
| 2 | major | The verdict "equivalent" hid that PC-Newton is deterministically worse (CI [+0.47, +0.48] excludes 0) and sits 0.03 dec inside an arbitrary margin. | Dual verdicts: "equivalent (but worse)". The text states explicitly that this is margin-dependent. |
| 3 | major | The cost model charges a fresh LU per solve. Multi-solve PC/Newton at one data sample on affine problems is penalised by implementation, not algorithm. | Sensitivity cost `flops_lu_reuse` and a second matched-cost block. PC C = 1 goes from +0.47 to +0.24 dec on linsys/tvqp; the conclusion "worse but within margin" is unchanged. |
| 4 | major | GNN lag slope was fitted on non-quasi-static γ (linsys γ = 3: γσ_min² = 1.9 < 3ω), which biased it to −0.87 and "supported within 0.15". The GNN τ sweep was treated as asymptotic although it is transient- and lag-dynamics-limited, which the trace ratio q < 3 misses. | Quasi-static criterion γσ_min² ≥ 3ω for the lag fit and for GNN cells. Mini lag sweep moved to τ = 3e-4 with γ = 30, giving 2 quasi-static points on linsys (−1.00). 300 and 700 added to the full grid, giving 3 points on tvqp. |
| 5 | major | Transient-limited ZNN cells at τ ≥ 1e-2 (ρ_opt up to 0.975, T = 6 s) fed the headline matched-cost numbers silently (see `fig_E9_traces`: `uhlig_3_4` still decays through the window). | Matched-cost results are flagged "transient-limited bracket" (conservative). The figures mark these cells. A transient-cost metric (steps to 1e-2, right-censored) was added. T = 6 s is kept (section 3, item 8). |
| 6 | minor | H1b grid {0.5, 1, 1.5, 2.5} could not locate the limit, and the line from 1.5 to 2.5 suggested divergence starting at 1.5. | η = 1.9 and 2.1 added (0 % at 1.9, 0–12 % at 2.1). H1b is judged on η < 2 (0 %) and on the largest η. |
| 7 | minor | Kendall τ_b ranked round-off-floor values (1.51e-14 vs 1.53e-14) as distinct. | Values below 1e-11 are tied; causal-only τ_b is added. |
| 8 | minor | No multiplicity control over about 150 matched-cost CIs. | Bonferroni CIs and verdicts (K = 75 per budget). No verdict changes in mini. |
| 9 | minor | Davidenko drift (final/initial ≈ 1.0–1.7) was counted as "unstable", which removed it from order fits. | Open-loop control exempt. |
| 10 | minor | `fixedtau` hollow-marker overlay was drawn below the errorbar markers (zorder 2 < 2.1). Equivalence plot could not show open CIs or unstable cells. Long legend labels widened two figures. | zorder 3; arrows for open CIs, "unst." labels and a "t" mark for transient-limited brackets; shorter labels. All 11 figures are at DOUBLE width. |
| 11 | minor | Interpretation text hard-coded claims ("every η < 2 run is stable", the ρ range). | All statements are now generated from the data. |
| 12 | info | Cost estimate used B-independent rates without checking them. GNN and lag batches grew. | Batch-size independence measured (< 3 %). The estimate was recomputed per config: 3.4 core-hours (previously 3.9), 17,748 trajectories, 268 tasks. |

## 8. Review log (2026-10-18)

Second methodological review (statistics / ZNN numerics), after the first review of section 7. Every
item was fixed in E9's own files (`configs/E9.py`, `tasks/E9_*.py`, `analyze_E9.py`, these notes) and
checked on the mini re-run (section 4). Effects are mini numbers at budget 1 (τ = 1e-2) unless stated.

| # | severity | issue | change | effect on numbers |
|---|---|---|---|---|
| 1 | major | **Asymmetric cost accounting.** `step_flops` skips the unused dE/dt of GNN but charges the unused E that simulate evaluates for the Davidenko predictor of every PC scheme (136 / 160 / 134 FLOPs per step). This biased the headline ZNN-vs-PC comparison against PC. | `E9_cost.py` (`COST_VERSION` 3) removes that E call while `step_flops` still counts it (detected, so an upstream fix is not subtracted twice); the code view is kept as a sensitivity. The zoo adopted the same fix in `costs.py` at 18:27 (`skip_unused_E`); E9's totals are identical under both versions. | Causal PC-Newton C = 1: +0.47 → **+0.38** dec (linsys, tvqp), +0.46 → +0.37 (tvuo); with LU reuse +0.24 → **+0.11**. All Davidenko-based schemes are 6–18 % cheaper. Verdict class unchanged ("equivalent (but worse)"). |
| 2 | major | **Pseudo-replication / too few clusters.** The B initial states of a seed are identical in steady state (median within-seed spread 3.5e-14 decades vs 0.19 across seeds). The cluster bootstrap over 2–3 instances has 3–10 distinct resamples, so the "95 % CIs" were instance ranges. | Full scale: **10 seeds** × B = 4 (PLAN: 3 × 4). Captions, `SUMMARY.md` and `summary.json` state the cluster count and call ≤ 3-instance intervals descriptive. | None on mini values. Full-scale cost ×3.3 (≈ 12.5 core-hours). |
| 3 | major | **Budget brackets crossed the stability edge.** The look-ahead formulas run at τ = 1.04–1.17 × the budget τ. The bracket reached the next grid point (3e-2 mini, 2e-2 full), where the order ≥ 5 formulas are unstable, so the headline formula `uhlig_4_6` was "n/a" on linsys and tvqp. | Grid points 1.2e-2 (mini, full) and 1.2e-3 (full). | `uhlig_4_6`: tvqp n/a → **−2.88** dec (better); linsys n/a → **+0.14 [−2.79, +0.78], inconclusive**: one instance is at the edge of its usable τ range at 1.2e-2 (section 4). ZeaD-7i tvqp −1.90 (inconclusive) → −2.24 (better); tvuo `uhlig_4_6` −5.75 → −6.23. |
| 4 | moderate | **Censoring in rankings and lists** (checklist 2). Kendall τ_b dropped methods with +∞ (unstable bracket); the H2 verdict list dropped unstable formulas; the equal-τ ratio dropped non-finite τ silently; no per-cell censoring report. | +∞ is tied-worst in τ_b; only methods outside the sampled τ range are excluded, and listed; unstable formulas keep their "n/a" verdict in the list; non-finite τ are reported; censoring table in `SUMMARY.md`; Kendall p-values labelled descriptive. | τ_b (all / causal): 0.95 / 0.92 → 0.90 / 0.85 (linsys), 0.93 / 0.93 → 0.90 / 0.89 (tvuo), 0.92 / 0.90 → 0.93 / 0.92 (tvqp); the grid change of item 3 contributes. No +∞ remains at budget 1 in mini. |
| 5 | moderate | **H2 not falsifiable as stated.** "ZNN ≈ PC ≈ Uhlig" mixed an algebraic identity, a cost statement and a cross-order claim. | H2a / H2b / H2c with predictions (section 4); the order-2 prediction 2 log₁₀(cost ratio) is printed next to every observed PC-Newton ratio. | H2a supported (ratio 0.995–1.012); H2b matches to ≤ 0.01 dec in all three cost views; H2c: equivalence across orders fails, as predicted, except where the formula is transient-limited or at its stability edge. |
| 6 | minor | **Order-fit tolerance** max(0.5, 0.15·\|median\|) was looser than the project's 0.15 (checklist 3); no flag for absurd slopes. | 0.15 absolute; `implausible` flag (above expected + 3, or below 0.5 for an expected order ≥ 1), mark `!` in `E9_orders.tex`. | Headline orders unchanged (Euler 2.00, TZ 2.94–2.99, ZeaD-5i 3.86–3.97). Two PC-gradient slopes on the KKT problem (0.33, 0.35) are now flagged. |
| 7 | minor | **LU-reuse count** charged the non-causal PC-Newton two factorisations per step; its prediction at t_k uses the sample its previous step corrected at. | One factorisation per new data sample, i.e. per step, for every solve-using method. | Non-causal PC-Newton LU-reuse value only: it now differs from the primary value (−11.05 vs −11.10 dec on linsys; both floor-limited). |
| 8 | minor | **Stale cost profile possible.** The cost task's done file did not depend on `costs.py`, which another agent is fixing now; a rebuild would have kept the old profile. | The hash includes the SHA-256 of `znnbench/costs.py`, `znnbench/wrappers.py` and `E9_cost.py`; the analysis warns on a fallback profile or `cost_version` < 3. | None (forces re-profiling when the cost model changes). |
| 9 | minor | **GNN gain choice** was tested only at η = 1.5, below its fastest-contraction step ≈ 2/(1 + σ_min²/σ̄²). | Sensitivity at η = 1.9. | GNN +2.58 / +2.77 / +2.84 → +2.51 / +2.67 / +2.83 dec: worse in every case. |
| 10 | minor | **Figures and tables.** The non-causal block was not separated in `fig_E9_equiv` / `E9_matched.tex`; the work–precision figures did not show the budgets; three numbers in the interpretation were hard-coded. | Shaded, labelled non-causal block; budget lines on `fig_E9_workprec_*`; the look-ahead cost spread, ρ range and h_max/h_opt ratios are computed from the data. | None. All 11 PNGs re-inspected (style.py / keys.py groups, ≤ 8 series, legends, no dual axes). |
| 11 | info | **Confounds** (checklist 6) were spread over the notes. | `SUMMARY.md → Confounds` and `→ Matched-cost definition`: GNN effective step; ZNN gain rule (h_opt is transient-optimal, not steady-optimal: h → h_max would gain 0.21–0.62 dec); causal vs non-causal; analytic E_t (the high-order advantage is conditional on an accurate E_t, see E6); transients; h vs c₀·h convention; shared x0; n = 8 (LU ≈ 35–41 % of a ZNN step). Checklist item 1 (unit dependence) does not apply: linear activations only. | None. |
| 12 | info | `zead_5i_mahuang2025` was registered during the review. | Considered, not added (section 3, item 9). | None. |
| 13 | info | Cross-check of `costs.py` (read-only, under verification). | All 78 entries: `step_flops` call counts = `profile_step_cost` steady counts, except GNN's dE/dt (by design); hand counts for linsys (E 136, GETRF 308, GETRS 120, ZNN-Euler 750). `costs.py` changed during the review (18:27, `skip_unused_E` added = zoo issue 2 fixed); the new done-file hash (item 8) re-profiled it automatically, and E9's totals are identical (78 / 78). | None. Zoo issue 1 is still open. |
| 14 | info | **Wall-clock bias** (lead's E7 finding): wall time measured inside a parallel pool on this machine is biased (shared per-cluster matrix units; load 40–180 from other projects). | E9's matched cost never used wall-clock; the wall axis, the wall column of `E9_cost.tex` and `timing.*` are now labelled descriptive (figure axis, caption, `SUMMARY.md`). | None on any verdict; the core-hour estimate is robust for E9 (interpreter-bound steps, rates within 0–6 % under load ≈ 40). |

**Note on the GNN floor.** ‖(JᵀJ)⁻¹ẋ*‖/γ is the *state*-error floor of GNN. The *residual* floor, which
MSSRE measures, is ‖J⁻ᵀẋ*‖/γ (E9's c_lag; measured / theory 0.95–1.02), and both hold only in the
quasi-static regime γσ_min² ≫ ω. E9 supports the reading of Euler-ZNN as causal prediction–correction
exactly (H2a, H2b) and shows that high-order formulas lose usability at moderate τ inside the nominal
interval (`uhlig_4_6` on linsys at τ = 1.2e-2).

**Readiness.** The mini pipeline runs end to end (build → pool → analyze, exit 0, no failed task). The
full sweep is ready to launch with the commands of section 6.
