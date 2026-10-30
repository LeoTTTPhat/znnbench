# E4: fairness re-ranking (notes)

The paper's central claim: the ranking of "proposed vs. baseline" ZNN models depends on the comparison
protocol. Under each source's own parameters the newest models look best; once gain, stability-limited
step or control effort is matched, and once the models run in discrete time, the order changes.

Files (all new; nothing under `znnbench/` or `tests/` was touched):

| file | role |
|---|---|
| `experiments/configs/E4.py` | single source of truth: panel, P0 parameters with provenance, scales, matching levels, runner configs (`get_configs()`) |
| `experiments/tasks/E4_calibrate.py` | memguard calibration tasks for P1/P2/P3 (`build`, `run`), sweep builder (`sweep`), `status` |
| `experiments/analyze_E4.py` | ranks, Kendall tau_b + bootstrap CIs, claim survival, figures, tables, `summary.json`, `SUMMARY.md` |
| `experiments/E4_NOTES.md` | this file |

## 1. Exact protocol

**Problems.**
- `matrix_inversion`: n = 4, kappa = 10, frequency 1 rad/s, `seed = $seed`, so each seed is a new instance.
- `tvqp_eq`: the CAAI-TIT 2021 benchmark (18) data. These data are deterministic, so the seed only changes the
  initial states.
- Initial states: the problems' default X(0) ~ U[-2, 2] (Zhang & Ge 2005), drawn by `simulate` from the seed.
- Horizon: T = 6 s (PLAN default). Steady window: the last 25 %.

**Statistics.**
- Evaluation seeds {0, 1, 2} at mini scale and {0, ..., 9} at full scale (review 2026-10-18; PLAN's 3 seeds give
  degenerate cluster-bootstrap CIs), each with B = 4 initial states, so 12 (mini) / 40 (full) trajectories per cell.
- Calibration uses the held-out seeds 100, 101 and 102: three matrix instances x B = 4 initial states, i.e. as many
  trajectories as one mini evaluation cell (one seed gave held-out P3 effort ratios of 0.53-0.79).
- Each model's score in a cell is the median over the 12 trajectories. Diverged or failed trajectories count as inf.
- MSSRE is clamped at the round-off floor 1e-12, so models that both sit at the floor tie.
- Ranks use average ranks for ties (`scipy.stats.rankdata`).
- Agreement with P0 is Kendall tau_b over the ranked panel. It is computed per (problem, metric) and pooled as the
  mean over problems and/or metrics.
- 95 % CIs come from a paired two-stage cluster bootstrap (cluster = seed): 2000 resamples, seed 20261018. Seeds are drawn with
  replacement, then initial states within each seed. The same resample is used for every protocol and model.
- tau_b is undefined when one ranking is constant (all tied). It is reported as n/a.

**Metrics.**
- Time to ||E||_2 < 1e-3: the runner's `t_eps_0.001`. The residual must stay below the threshold afterwards;
  inf if it never does.
- MSSRE: the maximum of ||E||_2 over the steady window.
- Also recorded: `peak_xdot` (max ||x_dot||_2 over every rhs evaluation), `chatter_vec`, `initial/final/max_residual`,
  traces (300 points at full scale, 120 at mini).

**Panel** (`PANEL_ALL`). Keys are filtered at build time by `znnbench.available()` and `activations.available()`.

| label | registry key / kwargs | gamma (P0) | year | provenance of the P0 values |
|---|---|---|---|---|
| linear | `znn` activation=linear | 1 | 2002 | Zhang & Ge 2005, Sec. V, gamma = 1 (primary; `docs/spec/problems_matrix.md`) |
| power_sigmoid | `znn` power_sigmoid p=3, xi=4 | 1 | 2005 | Zhang & Ge 2005, Sec. V: gamma = 1, xi = 4, p = 3 (primary) |
| sign_bi_power | `znn` sign_bi_power r=0.5 | 1 | 2013 | **ASSUMPTION**: gamma and r are not recorded in the specs |
| fixed_time | `znn` fixed_time k=(1,1,1), p=0.5, q=2 | 1 | 2022 | **ASSUMPTION**: zoo defaults (source values not in the specs) |
| predefined_exp | `znn` predefined_exp p=0.5, k=(1,1,1) | 1 | 2019 | **ASSUMPTION**: zoo defaults; bound t_c <= 1/(gamma k1) = 1 s |
| vpznn | `vpznn` p=1 (g = t + 1) | 1 | 2018 | spec: "gamma = 1 reproduces a paper"; p = 1 is an ASSUMPTION |
| sptc_nt_znn | `sptc_nt_znn` t_c=1, p=0.5, Delta=0.5 | 2 | 2024 | Yang et al. 2025 comparison: gamma = 2, t_c = 1, p = 0.5, zeta = 5 Delta (Delta = 0.5 ASSUMPTION) |
| ieznn | `ieznn` (lambda = gamma^2/4) | 1 | 2016 | **ASSUMPTION** gamma = 1; lambda follows the ZNN-Bench mapping |
| sptc_an_foznn (optional) | `sptc_an_foznn` alpha=0.5, t_c=1, Delta=0.5 | 2 | 2025 | Yang et al. 2025 TVQP example: gamma = 2, t_c = 1; alpha = 0.5 from the robot example (primary) |

- **Fallback rule.** Several sources' simulation gains are not recorded in the zoo specs, and two web lookups found no
  open full text (`NTZNN.pdf` refused the connection). For these, gamma = 1 is used. This is consistent with the two
  primary-verified baselines (Zhang & Ge 2005) and the VP-ZNN convention. It is also how activation papers present
  superiority: at equal design parameter. All P0 values live in one dict in `configs/E4.py` and can be corrected in
  place.
- **Predefined-time activation.** `predefined_exp` was chosen over `pvaf1`: it is the first predefined-time ZNN
  (Li et al. 2019), and its style slot (act #7) does not collide with SPTC-NT-ZNN (model #6) in single-panel plots.
  `pvaf1` is the automatic fallback.
- **sptc_an_foznn ("if stable").** It is run in every protocol. It is ranked only if at most 50 % of its P0
  trajectories diverge on every problem. With RK4 in continuous time it is stable: mini (tau = 1e-3) and full-scale
  probe (tau = 1e-4) both had 0 divergences, so it is ranked. It therefore becomes the "newest" model.

**Protocols.** These are computed by `E4_calibrate.py run` on the calibration seeds (one task per problem x model x
error unit).

- **P0, as published.** Each model's P0 gamma and parameters above. Continuous time: RK4 with tau = 1e-4,
  `record_every` = 10.
- **P1, matched linearised gain at t = 0+.**
  - Effective initial gain: kappa(gamma) = -<r, E>/<E, E>.
    - r is the model's own desired dE/dt at t = 0, z = z0.
    - It is evaluated by calling `model.rhs` on an identity probe problem (E = x, J = I, dE/dt = 0) with all m entries
      of E equal to e. This is generic, so it covers any registered model.
  - Choice of e:
    - If kappa is finite as e -> 0 (checked at 1e-8 e_ref vs 1e-10 e_ref), it is evaluated there, which gives
      gamma Phi'(0) g(0).
    - Otherwise (non-Lipschitz Phi), it is evaluated at e_ref, which gives the secant gain gamma g(0) Phi(e_ref)/e_ref.
  - gamma solves kappa(gamma) = gamma* = 10 on [1e-3, 1e4] (log bisection).
  - **e_ref** = median over the calibration trajectories of ||E(x0, 0)||_2 / sqrt(m), i.e. the RMS entry of the
    initial error: 6.41 on matrix_inversion and 1.69 on tvqp_eq (calibration seeds 100-102). Justification: Phi acts entry-wise, so the decay rate
    of a typical entry is set by the secant at a typical entry magnitude. Using ||E||_2 itself would inflate the
    |e|^q (q > 1) branches by m^{(q-1)/2} (a factor of 4 for q = 2, m = 16).
  - Gains that grow in time (VP g(t) = 1 + t, SPTC 1/(t_c - t)) are matched at their t = 0 value. Their later
    growth is the feature under test.
  - For IEZNN the proportional gain is matched (gamma = 10). Its double pole then sits at gamma/2.
  - SPTC-AN-FOZNN's linear part does not depend on gamma: (t_floor/t_c)^(alpha-1)/t_c = 31.6 at t = 0. It cannot be
    matched; it gets the gain floor 1e-3 and the flag `floor`.
  - Continuous time: RK4 with tau = 1e-4. Sensitivity: gamma* in {3, 30} (configs `E4_P1_g3`, `E4_P1_g30`);
    `E4_P1sec` (secant gain at e_ref for every activation, also the Lipschitz ones); `E4_P1pole` (review
    2026-10-18: the slowest pole of the linearised (E, z) error dynamics is matched instead of the initial gain;
    computed generically from the model's rhs on the identity probe with a 2 x 2 Jacobian for models with one
    auxiliary state per entry; only IEZNN differs: gamma = 2 gamma* = 20, double pole at 10; the config holds only
    the differing pairs and the analysis inherits the others from P1).
- **P2, 0.8 x own maximal ADMISSIBLE step (matched admissible-step margin; review 2026-10-18 renamed it, it is not
  a pure stability margin).**
  - Criterion: explicit Euler, tau = 1e-3, T = 6 (4 at mini). A step h = gamma*tau is admissible iff every
    calibration trajectory satisfies all four conditions:
    - (i) no divergence (|x| < 1e10, finite);
    - (ii) MSSRE <= 0.1 ||E(x0, 0)|| (the residual contracts at least 10x within T);
    - (iii) max ||E|| over the last T/8 <= 2 x the max over the preceding T/8 (no growth);
    - (iv) max_k ||E_k|| <= 10 ||E(x0, 0)|| (no transient blow-up).
  - What it measures. For Lipschitz models (linear, IEZNN, VP-ZNN, power-sigmoid at E4's ||E(x0,0)||) it coincides
    with the Euler stability boundary (linear 1.98 = E2's 1.98; IEZNN 3.73 < 4). For chattering (sign-type) models
    condition (ii) binds first at small errors: there it is a CHATTERING-SIZE (residual-size) limit set by
    STAB_RHO = 0.1, not a stability limit; E2's review measured h_max(E4 criterion)/h_max(bounded criterion) =
    0.41 (sign-bi-power), 0.51 (fixed-time), 0.24 (predefined-exp), 0.16-0.21 (VAF/PVAF-1) at ||E(0)|| = 1. At large
    errors the far-field power terms' Euler stability binds instead (TV-QP sign-bi-power: h_max 0.35 at u = 0.1
    (chattering-limited), 0.77 at u = 1, 0.083 at u = 10 (far-field-limited)).
  - Why this is the fair basis for E4. E4 ranks accuracy and speed (time to 1e-3, MSSRE). A step at which a model
    cannot even contract its error 10x within T is not an operating point a designer would choose, so "bounded but
    useless" steps must not define the comparison. The choice is CONSERVATIVE for the conclusion: a looser
    (bounded-only or divergence-only) criterion would put the chattering models at 2.4-6x larger steps (E2 ratios),
    i.e. at larger chattering floors, while the Lipschitz models keep their h_max (ratios 0.87-0.99), so it cannot
    improve the chattering models' standing in P2 (not run; argued from E2's ratios); the opposite direction is
    covered by
    `E4_P2_h0.25` (every model at 0.25 h_max): pooled tau_b(P0, P2_h0.25) = -0.46 vs -0.53 at 0.8 h_max.
  - Unit / ||E(0)|| dependence. For every nonlinear activation P2's h_max depends on ||E(x0, 0)|| and therefore on
    the unit of E (changing the unit acts on the activation exactly like changing ||E(0)||). Paired unit arms
    (re-calibrated) give h_max per decade of error unit (u 0.1 -> 1 -> 10, matrix inversion): power-sigmoid x0.010,
    x0.010; sign-bi-power x0.24, x0.11; fixed-time x0.28, x0.11; predefined-exp x0.85, never admissible at u = 10;
    SPTC-AN x0.11, x0.09; linear, VP-ZNN, IEZNN and SPTC-NT x1 (exactly invariant). P2's ranking is the most
    unit-robust of all protocols (tau_b(P2, P2_u) = 0.84 at u = 0.1, 0.55 at u = 10) and its reversal of P0 keeps
    its sign in every unit (tau_b(P0_u, P2_u) = -0.35, -0.53, -0.17 for u = 0.1, 1, 10), but its size is unit
    dependent.
  - Search: a descending log grid on [1e-5, 6] (22 points at full, 13 at mini), simulated as one batch. The
    admissible set is an interval [h_lo, h_max] (below h_lo the residual does not contract 10x within T). h_max is
    the upper end of the LOWEST admissible block of the grid (connected run, as in E2; review 2026-10-18: before,
    the largest admissible grid point was used, which could select an island above the first loss of
    admissibility; every mini calibration has exactly one block, so the change is result-neutral; the block count
    is recorded as `n_admissible_blocks`). The bracket above it is refined by 6-point multi-section rounds (2 at
    full, 1 at mini).
  - The evaluation uses h = 0.8 h_max, and admissibility at that h is re-verified on the calibration seeds.
- **P3, matched peak control effort.**
  - Budget: U* = the median peak ||x_dot|| of linear ZNN at gamma* = 10 on the calibration seeds (RK4, tau_ct).
  - gamma_P3 is the **largest** gamma whose median peak is <= U*. Search: ascending log grid on [1e-3, 1e4], then
    refinement inside the bracket, then log-log interpolation, then a verification run.
  - "Largest admissible gain" is used because peak(gamma) is not monotone:
    - on tvqp_eq it is U-shaped, because an unconverged state keeps a large feed-forward term J^-1 dE/dt;
    - for SPTC-NT with gamma < 1 it is singular near t_c.
  - If no gamma is admissible, the least-effort gamma is used with the flag `infeasible` (SPTC-AN-FOZNN).
  - Evaluation is on the held-out seeds 0-2. The transfer ratio (median eval peak / U*) is reported.
  - Sensitivity: budgets equal to the linear-ZNN peak at gamma_ref in {5, 30} (`E4_P3_r5`, `E4_P3_r30`).
- **P4a / P4b.** P0 parameters with explicit Euler at tau = 1e-2 and tau = 1e-3.
- **Unit arms (review 2026-10-18).** P0, P1, P2, P3, P4a and P4b are repeated with the error expressed as
  E_u = u E, u in {0.1, 10} (configs `E4_<P>_u0.1`, `E4_<P>_u10`; problem wrapper
  `experiments.tasks.R_common:ScaledError`, which keeps the recorded residual, eps = 1e-3 and MSSRE in the original
  unit and leaves x, hence ||x_dot||, unchanged). Every published parameter VALUE is kept (gamma, r, p, xi, k,
  t_c, Delta): the arm asks how a published parameter set behaves when the same problem is written in another unit.
  P1/P2/P3 are re-calibrated per unit (e_ref, h_max and the effort-matched gains are unit dependent; P3 only at
  the main budget). Linear ZNN, VP-ZNN and IEZNN are exactly unit invariant (identical calibration in every unit,
  checked); SPTC-NT is invariant before t_c only.

**Outputs of the analysis.**
- "Newest model best": cells (problem x metric) in which the newest ranked model ties for rank 1. Only
  informative cells count, i.e. cells in which the compared models do not all tie.
- "P0-best stays best".
- **Claim survival**: for each model, the cells in which it is best among itself and all *older* panel models.
  This is the claim each paper makes against its baselines. It is reported as won / informative cells.

## 2. Deviations from PLAN.md and why

1. **Panel.**
   - Fractional PT-ZNN is SPTC-AN-FOZNN (conformable, Yang et al. 2025), the only fractional model in the zoo.
   - Strictly-PT TVG is SPTC-NT-ZNN.
   - Predefined-time is `predefined_exp`.
   - VP-ZNN is the power type.
2. **P0 gains.** Where the specs give no published gamma, gamma = 1 (ASSUMPTION, listed per model). P0 is "as
   published where known, otherwise at the unit design parameter".
3. **P1 definition.**
   - P1 is matched at t = 0+ through each model's own rhs.
   - e_ref is the RMS entry, not ||E||_2 (justified above).
   - P1 runs in continuous time, as do P0 and P3, to isolate parameter fairness from discretisation (P2 and P4 cover
     the latter).
4. **P2 stability criterion.**
   - This is an E4-specific criterion (bounded, contracting by 10x, no growth, no transient blow-up). E2's "floor"
     criterion would declare chattering sign-type models unstable at every h.
   - Condition (iv) was added after the mini run showed SPTC-NT surviving a 4.5e8 transient. That transient comes
     from the Euler factor |1 - h/(t_c - t_k)| over the last gamma/2 steps before t_c.
   - The h range is (0, 6], not E2's (0, 3]. With lambda = gamma^2/4, IEZNN under Euler has the double root
     1 - h/2, so it is stable up to h = 4. With the cap at 3 it was clipped.
5. **P3 budget sensitivity.** The sensitivity levels are reference peaks, not multiples of U*. U*/3 = 17 lies below
   the feed-forward floor ||J^-1 dE/dt|| of about 35 that every model pays at t = 0 on matrix_inversion.
6. **Extra protocols.** Sensitivity configs P1_g3, P1_g30, P1sec, P1pole, P3_r5 and P3_r30 were added (P1pole by the
   review of 2026-10-18).
7. **Mini scale.** T = 4 s, continuous-time RK4 at tau = 1e-3, coarser calibration grids, 3 evaluation seeds. Full
   scale = PLAN: T = 6, tau = 1e-4, except for the seeds (item 9).
8. **Unit arms (review 2026-10-18).** Every main protocol is repeated with the error in units 0.1 E and 10 E (and
   re-calibrated there). PLAN has no unit arm; the cross-cutting review found that power-type activations have no
   length scale, so a ranking in one arbitrary unit is not a finding.
9. **Seeds (review 2026-10-18).** Full scale uses 10 evaluation seeds instead of PLAN's 3: tau_b with its seed-cluster
   bootstrap CI is E4's headline number, and 3 clusters give only 10 distinct resamples (degenerate CIs such as
   [0.28, 0.28] at mini).

## 3. Mini-run results (post-review rerun, 2026-10-18)

The mini scale ran the whole pipeline from an empty `results/mini/E4` (the pre-review outputs are not kept):
build, memguard pool (`--workers 3 --max-gb 40`), sweep build, pool, analyze. The machine was shared with other
reviews (load average 20-120 on 32 cores), so the walls below are inflated about 1.5-3x.

- **Calibration.** 54 tasks (2 problems x 9 models x 3 error units), 5 min 19 s wall, 820 core-s, 0 failures,
  peak RSS <= 69 MB per task.
- **Sweep.** 26 configs, 1353 specs, 226 tasks: 8 min 40 s wall for 208 tasks + 47 s for the 18 tasks of the two
  configs added last (P1_g1, P2_h0.25); 0 failures. 5604 trajectory rows (P1pole inherits 48 of its 54 specs from
  P1): 5403 `ok`, 201 `diverged` (all in P1_g30, P4a and the u = 10 arms), 0 errors/timeouts. Peak RSS <= 126 MB per
  task. Measured wall per step: RK4 0.31 ms (matrix inversion) / 0.17 ms (TV-QP); Euler 0.14 / 0.07 ms.
- **Analysis.** 12 s. Total about 15 min wall on 3 workers (under the shared load).
- **Reproducibility.** The unit-1 numbers equal the pre-review run to every printed digit (the specs and run ids
  are unchanged; the runs are deterministic).

**Calibration (calibration seeds 100-102, original unit).** e_ref = 6.41 (matrix inversion) / 1.69 (TV-QP).
P1 gammas: linear, VP, SPTC-NT, IEZNN 10 (P1pole: IEZNN 20); power-sigmoid 4.82 (slope 2.075); sign-bi-power
2.94 / 8.13 (secant); fixed-time 1.28 / 2.89; predefined-exp 0.90 / 1.38; SPTC-AN 1e-3 (floor). P2 h_max: linear
1.98; IEZNN 3.73; VP 0.41; sign-bi-power 0.18 / 0.77; fixed-time 0.083 / 0.30; predefined-exp 0.044 / 0.083;
SPTC-NT 0.052; power-sigmoid 0.0048 / 0.098; SPTC-AN 0.0022 / 0.012. U* = 48.4 / 32.7; P3 gammas: power-sigmoid
0.086 / 1.93; sign-bi-power 2.35 / 6.72; fixed-time 1.06 / 2.50; predefined-exp 0.65 / 1.26; SPTC-NT 8.13 / 6.03;
linear, VP, IEZNN 10; SPTC-AN infeasible. Held-out P3 transfer relative to linear ZNN (eval peak/U* divided by
linear's ratio): 0.77-0.98 for the nonlinear activations (power-type terms respond most to the largest initial
entries), 0.90-0.99 SPTC-NT, 3.2-3.5 SPTC-AN (infeasible, flagged). Unit arms: linear, VP-ZNN and IEZNN calibrate
identically in every unit (exact invariance, a check of the wrapper); the power-type models shift strongly
(table in section 1, P2); predefined-exp at u = 10 is never admissible on matrix inversion (censored = worst in
P2_u10); power-sigmoid at u = 10 and SPTC-NT at u = 0.1 have infeasible P3 budgets (flagged in SUMMARY).

**Kendall tau_b vs. P0 of the same unit, pooled over problems and metrics [95 % seed-cluster bootstrap CI;
descriptive, 3 clusters only at mini].**

| protocol | u = 1 | u = 0.1 | u = 10 |
|---|---|---|---|
| P1 (gamma* = 10) | 0.08 [0.05, 0.10] | 0.32 [0.30, 0.33] | 0.01 [-0.00, 0.08] |
| P2 (0.8 h_max) | -0.53 [-0.53, -0.50] | -0.35 [-0.42, -0.34] | -0.17 [-0.19, -0.10] |
| P3 (U* at gamma = 10) | 0.17 [0.14, 0.18] | 0.29 [0.27, 0.31] | 0.05 [0.03, 0.05] |
| P4a (tau = 1e-2; MSSRE cells only) | 0.56 [0.54, 0.63] | 0.68 [0.54, 0.68] | 0.59 [0.54, 0.61] |
| P4b (tau = 1e-3) | 0.60 [0.57, 0.60] | 0.79 [0.78, 0.83] | 0.86 [0.79, 0.87] |

Sensitivity (u = 1, pooled): P1_g1 0.49, P1_g3 0.52, P1 0.08, P1_g30 -0.29 (matching level); P1sec 0.19;
P1pole -0.05; P3_r5 0.47, P3 0.17, P3_r30 -0.28; P2_h0.25 -0.46. Unit invariance tau_b(Pk, Pk_u) (u = 0.1 / 10):
P0 0.38 / 0.51, P1 0.56 / 0.16, P2 0.84 / 0.55, P3 0.34 / 0.31, P4a 0.33 / 0.38, P4b 0.65 / 0.41.

**Headline (claims = cells in which a model is best among itself and every older panel model, summed over the 8
models with baselines; won / informative).**
- Under P0 the published "better than older models" claims hold in 13/28 (u = 1), 12/31 (u = 0.1) and 13/29
  (u = 10) informative cells; under P1-P3 together in 10/94, 10/92 and 7/95. This collapse is the robust result:
  it holds in every unit, at every P3 budget (3-7/32) and P2 step factor (2/32), and at gamma* >= 10.
- It does NOT hold at the published gain level: P1_g1 (gamma* = 1) keeps 12/26 and P1_g3 10/32. Most of the P1
  re-ranking appears only once the matched gain is raised above the published gamma = 1, and at mini this trend is
  confounded with RK4 chattering floors that grow with gamma tau (re-check at full scale, tau = 1e-4).
- P2 reverses P0 in every unit (tau_b < 0) and at both step factors; P4 (published parameters in discrete time)
  mostly preserves P0 (0.56-0.86).
- The as-published ranking itself is unit dependent (tau_b(P0, P0_u) = 0.38 / 0.51): a P0 superiority of a
  power-type activation is a statement about one unit of E.
- The newest model (SPTC-AN-FOZNN) is best in 0 informative cells of every main protocol in every unit. It wins
  3/4 cells only in P1_g1, where it cannot be matched (its pre-t_c gain >= 31.6 does not depend on gamma, vs the
  target 1), and 1/4 in P1_g3 and P3_r5. "The newest model stays best" is therefore not an informative headline
  for this panel; claim survival is.
- Who wins: P0 predefined-exp (t_1e-3) and fixed-time (MSSRE); P2 linear ZNN and IEZNN (their admissible steps
  are 10-400x larger; the sign-type models chatter at about h).
- Read the mini numbers as pipeline validation. At RK4 tau = 1e-3 the continuous-time protocols carry O(tau)
  chattering floors of the sign terms (SPTC-NT's post-t_c floor 1.8e-3 > 1e-3, so its time-to-1e-3 is inf at mini;
  the earlier full-scale probe gave 1.7e-4 and t_1e-3 = 0.99 s).

**Figures** (PNGs in `figures/mini/` read after the rerun).
- `fig_E4_rank_heatmap`, `fig_E4_bump_t3`, `fig_E4_bump_mssre`, `fig_E4_kendall`, `fig_E4_gains`,
  `fig_E4_traces`: unchanged in content for u = 1 (P2 panel titles now say "max admissible h").
- `fig_E4_sensitivity`: three panels (P1 matching level incl. gamma* = 1; P3 budget; P2 step factor), colour = problem,
  marker + line style = metric, CIs, legend.
- `fig_E4_units` (new): left, tau_b(P0_u, Pk_u) per protocol for u = 0.1 / 1 / 10 (colour + marker per unit);
  right, unit invariance tau_b(Pk, Pk_u); pooled values with CIs, legends, no dual axes.
- Style caveat (unchanged, inherent to `keys.py`): slot 7 of `keys.GROUPS["act"]` (predefined_exp) and of
  `keys.GROUPS["model"]` (sptc_an_foznn) share the red star; they never share a panel.
- **Tables.** All 10 tables (`E4_params`, `E4_calibration`, `E4_ranks_t3`, `E4_ranks_mssre`, `E4_kendall`,
  `E4_claims`, `E4_medians_*`, new `E4_units`, new `E4_robustness`) compile with pdflatex (booktabs, amsmath,
  graphicx, array) with no errors and no overfull boxes (the params table's columns were narrowed by 0.6 cm).

## 4. Full-scale cost estimate and commands

Full scale: T = 6 s, RK4 tau = 1e-4 (`record_every` = 10) for the continuous-time protocols, Euler tau = 1e-3 (P2,
P2_h0.25, P4b) or 1e-2 (P4a), 10 evaluation seeds x B = 4, calibration seeds 100-102 x B = 4.

| item | count | estimate |
|---|---|---|
| calibration tasks | 54 = 2 problems x 9 models x 3 error units (u = 1 with three P3 budgets, u = 0.1 / 10 with one) | 4.3 core-h (measured mini 820 core-s, scaled per phase: P2 x (T ratio) x (evaluation ratio), P3 x (RK4 step ratio 15) x (evaluation ratio)); longest task ~9 min (matrix inversion, u = 1) |
| sweep runs (specs) | 4510 = [13 configs x 180 + P1pole 20] (u = 1) + 12 configs x 180 - 10 never-admissible (unit arms); 18 040 trajectories | 11.0 core-h (171 M batched steps x measured 0.31 / 0.17 ms per RK4 step, 0.14 / 0.07 ms per Euler step) |
| sweep tasks | ~752 (chunk_size 6, `balance: true`); longest spec ~19 s, longest task ~2 min | |
| **total** | | **~15 core-hours** (conservative: per-step costs measured under a shared, loaded machine and with `record_every` = 1; the pre-review uncontended measurement was 0.29 / 0.16 ms per RK4 step) |

- Peak RSS per task: <= 126 MB measured (sweep), <= 69 MB (calibration). At full scale expect <= 0.2 GB per task
  (recorded arrays 6001 x B floats at `record_every` = 10, traces of 300 points; P3 calibration records 2 samples).
  24 workers therefore need about 5 GB, far below the 200 GB cap.
- Recommended workers: `--workers 24` for both pools (32 cores). Calibration: 54 tasks, about 15-20 min wall
  (three rounds; the longest task ~9 min). Sweep: about 30 min wall. Total about 50 min wall.

```bash
cd znn-bench

# full scale (protocol scale): ~15 core-hours, ~50 min wall on 24 workers
python experiments/tasks/E4_calibrate.py build --scale full
python tools/memguard.py pool --tasks results/full/E4/calib/tasks.jsonl --workers 24 --max-gb 200
python experiments/tasks/E4_calibrate.py sweep --scale full     # refuses to build while calibration is incomplete
python tools/memguard.py pool --tasks results/full/E4/sweep/all_tasks.jsonl --workers 24 --max-gb 200
python experiments/analyze_E4.py --results results/full/E4 --figdir figures --tabdir results/tables

# mini (validation): ~15 min wall on 3 workers under shared load (~8 min idle)
python experiments/tasks/E4_calibrate.py build --scale mini
python tools/memguard.py pool --tasks results/mini/E4/calib/tasks.jsonl --workers 3 --max-gb 40
python experiments/tasks/E4_calibrate.py sweep --scale mini
python tools/memguard.py pool --tasks results/mini/E4/sweep/all_tasks.jsonl --workers 3 --max-gb 40
python experiments/analyze_E4.py --results results/mini/E4 --figdir figures/mini --tabdir results/tables/mini
```

(`python` is the interpreter of the project environment, see `requirements.txt`; every command runs from
`znn-bench/`.)

**Scale selection.**
- Use `--scale` on the task script.
- For the config file, set the environment variable `E4_SCALE` (or `ZNNBENCH_SCALE` / `ZNN_SCALE`) to `mini` or
  `full`. The default is `mini`, for safety.
- `E4_RESULTS` overrides the results root.

**Runner-native alternative.**
- `E4_SCALE=full python -m znnbench.runner build experiments/configs/E4.py --out results/full/E4/sweep`
  writes one subfolder per config, the same layout the `sweep` subcommand produces. The `sweep` subcommand also
  writes `all_tasks.jsonl` (the concatenation) and refuses to run while calibration is missing.
- `get_configs()` builds only P0/P4 (with a warning) until the calibration exists. Set `E4_REQUIRE_CALIB=1` to make
  that an error.
- `python experiments/configs/E4.py --scale mini` prints the panel, the calibration state and the spec counts.

**Rebuilds.**
- The config and the calibration task ids hash every setting, so a changed setting re-runs only the affected tasks.
  The runner's aggregate and `load_calibration` read only the current done files.
- New zoo keys are filtered in at build time. E4 uses only `rk4` and `euler`, so the new formulas and
  `costs.py` do not enter E4. A model added to `PANEL_ALL` needs no other code, because the P1 probe and the P2/P3
  searches are model-agnostic.

## 5. Observations and zoo notes (no zoo bug found)

- No errors and no timeouts: every failure is a recorded divergence. The runner, `simulate` online statistics,
  `PeakXdotProbe` and the batched-gain paths behaved as documented.
- **SPTC-AN-FOZNN cannot be gain-matched.**
  - Its pre-t_c linear part (t/t_c)^(alpha-1)/(t_c - t) does not contain gamma. At t = 0 the factor is
    (t_floor)^(-1/2) = 31.6 (t_floor = 1e-3 is a zoo ASSUMPTION).
  - Its effort at the smallest gamma is still 2.8-3.7 x U*.
  - This follows from the source equations plus the t_floor assumption; it is not a code bug. The paper should report
    that P1 and P3 are not achievable for this model.
- **IEZNN with z(0) = 0 overshoots.**
  - E = E0 (1 - gamma t/2) e^{-gamma t/2} crosses 0 at t = 2/gamma and then rebounds to about 0.13 E0 (gamma = 1).
  - This is visible in `fig_E4_traces` (P0) and is correct per `tests/test_variants.py::test_ieznn_exact_error_dynamics`.
- **Euler stability of SPTC-NT-ZNN is set by the singular gain.** h_max = 0.052 at tau = 1e-3, from the
  amplification over the last gamma/2 steps before t_c. Power-sigmoid's h_max (0.0056 on matrix inversion) is set by
  its cubic far-field branch: 3 gamma e^2 tau < 2.
- **Caveat.** In the continuous-time protocols, the MSSRE of sign-term models (predefined-exp,
  SPTC-NT post-t_c, SPTC-AN) is an O(tau) RK4 chattering floor (about 1e-4 at tau = 1e-4), whereas smooth models
  reach the round-off floor. MSSRE ranks in "continuous time" therefore already reflect discretisation. Time-to-1e-3
  is the cleaner continuous-time metric.
- **(review 2026-10-18) SPTC-NT at u = 0.1 has an infeasible P3 budget.** Keeping the published Delta = 0.5 in a
  unit 10x smaller makes the post-t_c sign term zeta = 5 Delta 10x stronger in physical units; its chattering
  effort then exceeds U* for every gamma. Not a bug: it is the unit dependence of a published parameter value.
- **(review 2026-10-18) SPTC-AN-FOZNN's t_p formula** (1 - exp(-pi/(2 sqrt(zeta (zeta - Delta))))) t_c is not
  dimensionally homogeneous (a rate inside exp), so its post-t_c dynamics depend on the unit of E through Delta.
  This is a property of the source equations, not of the code; E4 keeps Delta = 0.5 in every unit.
- **(review 2026-10-18) `experiments.tasks.R_common:ScaledError`** (another area's file, used read-only) works with
  the runner's problem-wrapper path and with the calibration: linear ZNN, VP-ZNN and IEZNN produce identical gains,
  h_max and budgets in every unit (exact invariance), as they must.

## 6. Confounds (cross-cutting checklist item 6; also in SUMMARY.md)

- **Unit and ||E(0)|| dependence.** The power-type activations (sign-bi-power, fixed-time, predefined-exp,
  power-sigmoid's |e| >= 1 branch, SPTC-NT's post-t_c terms, SPTC-AN's ||E||E term) have no length scale. P0/P4
  (fixed published gains) and P2 (h_max, chattering floor) depend on the unit of E and on the start offset; the P1
  secant gain at e_ref is unit invariant only for a single pure power (gamma_u u^(r-1) = gamma), not for the
  bi-power, exponential or sigmoid shapes. Quantified by the unit arms (section 3).
- **Equal nominal gamma is not equal effective gain.** At gamma = 1 the secant gain at e_ref is about 11 for
  predefined-exp, 3.4 for sign-bi-power and about e_ref^2 = 41 for power-sigmoid: P0 compares very different
  effective gains, which is why P0 favours the nonlinear activations and what P1/P3 remove.
- **Matching level.** P1 at gamma* = 10 (and P3 at the gamma = 10 budget) raises the gain level of every model 10x
  above the published gamma = 1, so "fair" and "faster" are confounded; P1_g1 (gamma* = 1) separates them and keeps
  most of the P0 ranking (tau_b 0.49).
- **P0 parameters partly assumed.** gamma = 1 is an ASSUMPTION for sign-bi-power, fixed-time, predefined-exp and
  IEZNN (also r, k, p, Delta and t_floor values); IEZNN's lambda = gamma^2/4 is a ZNN-Bench mapping. P0 is "as
  published where known, unit design parameter otherwise".
- **Internal states and time-varying gains in P1.** P1 matches the gain at t = 0+. IEZNN's slowest pole is then
  gamma/2 (P1pole: gamma = 2 gamma*); VP-ZNN's gain grows as 1 + t (7x at T = 6) and SPTC-NT's diverges at t_c, so
  P1 favours them after t = 0; P3 (effort over the whole run) charges for that growth.
- **Transients in the steady window.** With gamma = 1 and T = 6 (4 at mini), linear ZNN, IEZNN (overshoot
  ~0.13 ||E0|| around t = 4/gamma), VP-ZNN and power-sigmoid are still converging in the last 25 % of the horizon:
  their P0/P4 MSSRE is a late-transient value (flagged ^t in the medians tables when detected), so P0/P4 MSSRE ranks
  mix speed and accuracy. MSSRE ranks of sign-term models in the continuous-time protocols are RK4 chattering
  floors O(tau).
- **Arbitrary start offset.** X(0) ~ U[-2, 2] sets ||E(0)|| ~ 27 (matrix inversion) and therefore e_ref (P1), the
  peak ||x_dot|| that P3 matches (at t = 0 for most models, so P3 is a secant-gain match at the initial error) and
  every time-to-1e-3.
- **P2's criterion.** It is an admissible-step criterion (bounded + 10x contraction), a chattering-size limit for
  the sign-type models (section 1); a looser criterion would not help them.
- **Information advantages / noise.** E4 is noise-free with analytic dE/dt; SPTC-NT/AN get Delta = 0.5
  (ASSUMPTION) and IEZNN's integral term has nothing to reject, so their published (noise) advantages are not tested
  here (E5 covers them).
- **Step-size definition.** h = gamma * tau (raw product) for every model; SPTC's effective step
  gamma tau/(t_c - t) exceeds 2 just before t_c, which sets its P2 h_max.
- **Unmatchable pairs.** SPTC-AN-FOZNN (P1 floor, P3 infeasible in every unit), power-sigmoid at u = 10 and SPTC-NT
  at u = 0.1 (P3 infeasible): they run with MORE gain/effort than the target, i.e. biased in their favour.

## 7. Review log (2026-10-18)

Reviewer: methodology/statistics review of E4 (the central claim). Each item: severity, change, effect on the
numbers. All changes are in `experiments/configs/E4.py`, `experiments/tasks/E4_calibrate.py`,
`experiments/analyze_E4.py` and this file; nothing under `znnbench/` or `tests/` was edited. The mini scale was
re-run end to end afterwards (section 3).

1. **CRITICAL - unit dependence untested (checklist 1).** Every ranking rested on one arbitrary unit of E, although
   five panel members are power-type. *Change:* unit arms u in {0.1, 10} for P0-P4b (`R_common.ScaledError`
   wrapper; P1-P3 re-calibrated per unit, published parameter values kept), within-unit tau_b(P0_u, Pk_u),
   cross-unit invariance tau_b(Pk, Pk_u), headline per unit; `fig_E4_units`, `E4_units.tex`, SUMMARY section.
   *Effect:* the as-published ranking changes with the unit (tau_b(P0, P0_u) = 0.38 / 0.51) and so do the
   fair-protocol rankings (P1 0.56 / 0.16, P3 0.34 / 0.31; P2 is the most robust, 0.84 / 0.55). The collapse of the
   published claims under P1-P3 holds in every unit (P0 13/28, 12/31, 13/29 vs P1-P3 10/94, 10/92, 7/95) and P2's
   reversal keeps its sign (-0.53, -0.35, -0.17); the size of every tau_b is unit dependent.
2. **MAJOR - headline robustness not analysed.** The sensitivity configs entered only tau_b; "newest best" and claim
   survival were computed for the six main protocols only. *Change:* both, plus "P0-best stays best", for all 26
   protocols (`E4_robustness.tex`, SUMMARY table, `headline_robustness` in summary.json). *Effect:* "the newest
   model is never best" fails in P1_g1 (3/4), P1_g3 (1/4) and P3_r5 (1/4).
3. **MAJOR - matching level confounded with fairness.** P1/P3 match at 10x the published gain. *Change:* new
   sensitivity P1_g1 (gamma* = 1; gains from the stored calibration through the exact affine kappa(gamma), checked
   against the bisection at gamma* = 3, 10, 30 at build time). *Effect:* tau_b(P0, P1_g1) = 0.49 and claims 12/26
   (vs 13/28 under P0): most of the P1 re-ranking needs gamma* >= 10. The paper must say "at matched gain/effort
   above the published level" (and re-check at full scale, where the RK4 chattering floors are 10x lower).
4. **MAJOR - "newest model stays best" is vacuous.** SPTC-AN-FOZNN is not best even in P0 (0/4) and cannot be matched
   in P1/P3 (gain floor, infeasible budget: biased in its favour). *Change:* SUMMARY states it and reports claim
   survival as the headline; the P1_g1 win is labelled an unmatchability artefact. *Effect:* framing only.
5. **MAJOR - P2 meaning and name (coordinator / E2 input).** *Change:* P2 renamed "0.8 x own max admissible step"
   (bounded + >= 10x contraction + no growth/blow-up), its meaning per model class and its rationale documented
   (section 1); h_max is now the top of the lowest admissible block (connected run, as in E2) with the block count
   recorded; new sensitivity P2_h0.25; P2's unit / ||E(0)|| dependence quantified with the paired unit arms
   (h_max per decade: power-sigmoid x0.010, sign-bi-power x0.24 / x0.11, fixed-time x0.28 / x0.11, SPTC-AN x0.1,
   linear-type x1). *Effect:* the block rule is result-neutral (one admissible block in all 54 calibrations;
   identical gains, so every sweep task was reused); tau_b(P0, P2_h0.25) = -0.46 vs -0.53.
6. **MAJOR - too few clusters for the headline CI (checklists 4, 5).** The bootstrap was already a paired two-stage
   cluster bootstrap (seeds, then initial states within a seed), but 3 seeds give only 10 distinct seed resamples:
   degenerate CIs such as [0.28, 0.28], point estimates at the CI edge. No p-values are computed (none needed).
   *Change:* full scale uses 10 seeds; SUMMARY labels the CIs descriptive and conditional on the design. *Effect:*
   none at mini; full-scale sweep cost x3.3. For tvqp_eq the data are deterministic, so its clusters are x0 draws.
7. **MODERATE - P1 for models with internal states.** For IEZNN the initial gain gamma is not its convergence rate
   (double pole gamma/2). *Change:* sensitivity P1pole: the slowest pole of the linearised (E, z) dynamics is
   matched (generic numerical 2 x 2 Jacobian on the identity probe; IEZNN gamma = 20; only differing pairs are
   run, the rest is inherited from P1). VP-ZNN / SPTC-NT (growing gains): documented, P3 is the protocol that
   charges for growth. *Effect:* IEZNN t_1e-3 2.50 -> 1.25 s (matrix inversion), 1.99 -> 1.00 s (TV-QP), MSSRE
   1e-4 -> 1e-10 (its P1 MSSRE was an unfinished transient); its ranks 8/7/9/6 -> 6/3/7.5/3; pooled
   tau_b(P0, P1pole) = -0.05 vs 0.08; claims unchanged (5/30).
8. **MODERATE - transient flag too lax.** The 4x-decay rule missed linear ZNN at gamma = 1 (window ratio 0.45) and
   growing overshoots. *Change:* flag when the envelope falls > 2x or grows > 2x within the window above 1e-9.
   *Effect:* flags only (more P0/P4 cells marked ^t, e.g. linear ZNN); ranks unchanged.
9. **MODERATE - censoring (checklist 2) verified.** Diverged, failed and missing trajectories and pairs a protocol
   cannot run (never admissible in P2) are +inf = worst in the medians, the bootstrap and the ranks (inf ties with
   inf); nothing is dropped (`build_array` starts from inf; no nanmedian/dropna on metrics). *Change:* per-protocol
   censoring line in SUMMARY (diverged / failed / pairs not run / t3 = inf / MSSRE = inf). *Effect:* none (no bug).
10. **MINOR - confounds.** *Change:* sections 6 here and "Confounds" in SUMMARY (checklist 6).
11. **MINOR - P3 transfer.** Linear ZNN itself transfers at 0.79 on TV-QP (other x0 draws). *Change:* transfer is
    also reported relative to linear ZNN: 0.77-0.98 for the nonlinear activations, i.e. P3 under-spends their
    budget on held-out seeds by up to 23 % (a bias against them, small against the rank changes).
12. **MINOR - P0 provenance.** gamma = 1 remains an ASSUMPTION for sign-bi-power, fixed-time, predefined-exp and
    IEZNN (no new sources were looked up in this review); flagged in `E4_params.tex`, SUMMARY and section 6.
13. **MINOR - stale notes, cost model, table.** Calibration seeds (100-102, not 100), e_ref (6.41 / 1.69), the mini
    numbers, P1sec (undocumented) and the cost/commands section were out of date; the cost model now scales seeds
    and calibration phases separately (about 15 core-h instead of 2: 10 seeds x3.3, unit arms, 3x calibration);
    the params table's 7.6 pt overfull box was removed.

**Zoo bugs found: none.** The source-level oddities (SPTC-AN's gamma-free pre-t_c gain and non-homogeneous t_p,
SPTC Delta's unit dependence) are properties of the published equations, documented in section 5.

**Readiness.** The pipeline is ready for the full-scale sweep (commands in section 4). The full run must confirm
the matching-level dependence (P1_g1 vs P1) with tau = 1e-4 before the paper states the central claim; the claim
that survives every arm at mini is "published superiority claims collapse under matched-gain/effort protocols at
gamma* >= 10 and under stability-limited discrete-time steps, in every error unit; the as-published ranking of
power-type activations is itself unit dependent".
