# E3 — convergence-time audit of finite / fixed / predefined-time claims

Files (all E3-owned, nothing under `znnbench/` or `tests/` was touched):

| file | role |
|---|---|
| `experiments/configs/E3.py` | runner configs `E3_disc`, `E3_tc_<Tc>` (5) + script that also builds the CT tasks and a merged pool file |
| `experiments/tasks/E3_ct_reference.py` | custom memguard task: continuous-time (CT) reference times and bounds (part a) |
| `experiments/tasks/E3_metrics.py` | runner `metrics.extra` import paths: first-hit times, steady-window minimum, residual around t_c |
| `experiments/tasks/E3_units.py` | problem wrapper `ScaledError` (E_s = s E) for the unit-dependence arm (review 2026-10-18) |
| `experiments/analyze_E3.py` | figures `fig_E3_*`, tables `E3_*.tex`, `<results>/summary.json`, `<results>/SUMMARY.md` |
| `experiments/E3_NOTES.md` | this file |

## 1. Question

Finite-, fixed-, predefined- and strictly-predefined-time ZNN papers state a
convergence-time bound of the continuous-time error dynamics
`dE/dt = -gamma Phi(E)` (or a time-varying gain). E3 asks three things.
(a) How conservative are these bounds against the exact continuous-time time
to `||E||_2 < 1e-6`? (b) What does a fixed sampling period do to the claimed
time: time-to-eps vs tau and h, and the residual floor? (c) What happens to
predefined times `T_c` that approach the sampling period, and to gains
`~ 1/(T_c - t)` with and without saturation?

## 2. Exact protocol

Common to every part:
* problem `matrix_inversion` (n = 4, kappa = 10, 1 rad/s, `seed = $seed`,
  X(0) entries ~ U[-2, 2] as in Zhang & Ge 2005); natural initial errors have
  `max_i |e_i(0)|` of about 12–18 and `||E(0)||_2` of about 16–30;
* 3 problem seeds {0, 1, 2} x B = 4 initial states (`simulate.initial_state`,
  so the CT task and the runner use bit-identical x0); the full CT reference
  uses 10 seeds {0..9} (cheap; cluster-bootstrap CIs, review §8);
* statistics: median and IQR over the 12 trajectories (+inf kept as the worst
  value); CT headline CIs are cluster bootstrap 95 % CIs (2000 resamples, seed
  20261018; resample seeds, then initial states); divergence outcomes of the
  T_c sweep are deterministic per (model, gamma, T_c, tau) cell and reported
  as cell counts; Kendall tau_b for rankings (p-values descriptive); empirical
  orders fitted on the asymptotic range only (consecutive local slopes within
  0.15; values below 1e-12 and a small-tau plateau excluded), with all-points
  slope, local slopes and a 2-point flag (review §8).

### (a) Continuous-time reference (`tasks/E3_ct_reference.py`)

Entries (design-formula ZNN `znn` with the activation, or the registry model;
source / ZNN-Bench default parameters from `znnbench/activations.py`):
`linear` (reference), `sign`, `sign_power`, `saturated_sign_power`, `sfaf`,
`sign_bi_power`, `wsbp`, `fixed_time`, `vaf`, `pvaf1`, `pvaf2`,
`predefined_exp`, `strict_predefined` (t_c = 1), models `sptc_nt_znn`
(t_c = 1) and `sptc_an_foznn` (t_c = 1). Full scale adds the exponent
variants `sign_power(r = 0.25, 0.75)`, `sign_bi_power(r = 0.25, 0.75)` and the
pure fields `fixed_time(k3 = 0)`, `vaf(a3 = a4 = 0)`,
`predefined_exp(k2 = k3 = 0)` (for which the DERIVED tight bound / the
Sanchez-Torres settling time is exact).

Gains: gamma = 1 for time-invariant activations (full: 1 and 10). For these
the ratio bound/time is gamma-invariant (time rescaling), and the full run
checks this numerically. For the strict models the ratio depends on gamma,
so gamma ∈ {1, 2, 10} is used, with 2 as the headline value (the SPTC
papers' TVQP setting). Initial errors: the natural draw (`nat`) plus
rescaled copies `E0 = s E0_nat / max|E0_nat|` realised exactly through
`x0 = x*(0) + J^{-1} E0`, with s ∈ {1e-2, 1e2} (mini) or
{1e-3, 1e-2, 1e-1, 1, 10, 100} (full).

Method. With exact ∂E/∂t, ZNN imposes `e_i' = -gamma phi(e_i)` component by
component, whatever the problem. The reference is therefore semi-analytic:
* time-invariant phi: `C(v) = ∫ du/phi(u)` is tabulated on a log grid
  (u ∈ [1e-16, 1e4], 12000 cells, 8-point Gauss–Legendre per cell) and
  inverted with PCHIP, which gives `e_i(t)` exactly;
  `t_ref(eps) = {t : ||e(t)||_2 = eps}` is then found by Brent's method
  inside the exact bracket `[t_inf(eps), t_inf(eps/sqrt(m))]`, where `t_inf`
  (the time for `max_i |e_i|` to reach the level) comes independently from
  adaptive quadrature (rel. tol 1e-11). The bracket check validates the table;
* `strict_predefined` / `sptc_nt_znn` before t_c: closed form
  `t_ref = t_c (1 - (eps/||e0||)^(1/gamma))`;
* `sptc_an_foznn`: E' ∥ E, so `||E||` obeys a scalar ODE (Yang et al. 2025,
  Eq. 15), integrated in log coordinates with DOP853 (rtol 1e-12) and event
  location;
* implementation cross-check: DOP853 (rtol = atol = 1e-12) on the full ZNN
  state equation `x' = model.rhs(x, t)` until `||E|| < 1e-6`, until the first
  component drops below 1e-9, or until a wall cap (mini: 8 s; full: 60 s).
  Mini: `nat` scale and one initial state per seed; full: scales {nat, 1} and
  all 4 initial states. The task reports the max relative deviation of
  `||E_full(t)||` from `||e_ref(t)||` on all accepted steps, and `t_full(eps)`
  when that time is reached;
* bounds recorded per trajectory: `upper_bound_time(e0, gamma)` (source),
  `uniform_bound(gamma)` (DERIVED for sign_bi_power / wsbp),
  `tight_uniform_bound(gamma)` (DERIVED Beta integral), and the exact
  settling time to E = 0 (`settling_time`; t_c for the strict models).
  Conservatism ratio = bound / `t_ref(1e-6)`. Validity = bound ≥ exact
  settling time.

### (b1) `E3_disc`: time-to-eps vs tau and h

* models: `znn` with activations `linear, sign_bi_power, fixed_time, vaf,
  pvaf1, predefined_exp` (mini). Full adds `pvaf2, sign_power, sfaf, wsbp,
  saturated_sign_power, sign`.
* discretizations: `euler` and `rk4` (mini); full adds `taylor_zhang`. RK4,
  which is non-causal, is used at fixed gamma only, as a convergence check
  towards the CT values.
* tau ∈ {1e-2, 1e-3, 1e-4}. Full adds 1e-5 for the fixed-gamma Euler / RK4
  runs.
* gains: fixed gamma = 10 (CT convention, compared with the CT reference at
  the same gamma: `t_CT(gamma) = t_CT(1)/gamma`), and fixed h ∈ {0.1, 1}
  (DTZNN convention, gamma = h/tau; compared with the CT reference at that
  gamma). `batch_gains: true`.
* T = 4 s, `record_every: 1`, so t_eps has a resolution of tau;
  eps ∈ {1e-2, 1e-3, 1e-6}.
* metrics: core (MSSRE, `t_eps_*`, chatter, ...) + `initial_residual`,
  `final_residual`, `max_residual` + `E3_metrics:first_hit` (`t_hit_*`,
  `res_min`, `res_min_ss`).

### (b2) `E3_tc_<Tc>`: predefined time vs sampling period, time-varying gains

There is one config per T_c ∈ {1, 0.3, 0.1, 0.03, 0.01} s, with horizon
`T = max(3 T_c, 0.3 s)`, tau ∈ {1e-2, 1e-3, 1e-4} (T_c/tau from 1 to 1e4),
the Euler scheme of the source papers (full: + `implicit_euler_err` for the
constant-gain activations) and traces at 1200 points.
* constant-gain predefined / fixed-time activations `vaf, pvaf1, pvaf2,
  predefined_exp, fixed_time` (full: + `sign_bi_power`). The gain is chosen
  so that the stated (e0-independent) bound equals T_c:
  `gamma = uniform_bound(gamma=1)/T_c` (3/T_c, 3/T_c, 2/T_c, 1/T_c, 3/T_c);
* strictly predefined-time with t_c = T_c: `znn` + `strict_predefined` and
  `sptc_nt_znn` with base gain gamma ∈ G_TVG = {2, 10, 50, 100} (full: {1, 2,
  5, 10, 20, 30, 40, 50, 70, 100}); `sptc_an_foznn` at gamma = 2 (full: {2, 10});
* the same `sptc_nt_znn` with its 1/(t_c − t) gain saturated at
  `cap = rho/(gamma tau)`, rho ∈ {1, 1.9, 3} (labels `sptc_nt_znn+cap`,
  `+cap1.9`, `+cap3`; mini: rho = 1.9 and 3 only at gamma ∈ {50, 100}), so the
  effective Euler step `gamma tau min(1/(t_c-t), cap)` never exceeds rho:
  rho = 1 is the deadbeat step (best case), 1.9 a stable non-deadbeat cap, and
  rho = 3 > 2 a control outside the Euler stability interval (review
  addition, §8). The cap is a ZNN-Bench
  addition (models/gains.py), not part of any source. Because the cap depends
  on (gamma, tau), one model entry is written per distinct cap value, and the
  config's `exclude` rules keep only the matching (gamma, tau) pairs. The full
  run adds `sptc_an_foznn+cap`;
* extra metric `E3_metrics:tc_window_<tag>`: `tc_rmin_pre`, `tc_rpeak_post`,
  `tc_amp`, `tc_r_at` (the residual at the first sample ≥ t_c, which the
  strict claim says is 0). The analysis derives the jump at t_c,
  `tc_r_at / tc_rmin_pre` (∞ when the run diverged before t_c), and the
  remaining fraction `||E(T_c)||/||E(0)||`.

### (b3) `E3_vp`: varying-parameter gain (review addition)

`vpznn_exp` (g(t) = 2^t + 2, unbounded) with `linear` and `sign_bi_power`
(full: + `fixed_time`, `vaf`), Euler, gamma = 10, T = 12 s, tau ∈ {1e-2, 1e-3}
(full: + 1e-4), with and without `cap = 1/(gamma tau)` on g. Predicted onset of
the linear Euler instability: gamma tau g(t*) = 2, t* = log2(2/(gamma tau) − 2)
= 4.2, 7.6, 11.0 s.

### (b4) Unit-dependence arm (review addition, checklist item 1)

The same Euler runs with the error expressed as E_s = s E, s ∈ {1e-2, 1e2},
through `experiments.tasks.E3_units:ScaledError` (the activation acts on E_s;
the recorded residual, eps and all metrics stay in the original unit):
`E3_disc` at fixed gamma = 10 and fixed h = 0.1, and the constant-gain
activations of `E3_tc_<Tc>`; tau ∈ {1e-2, 1e-3} (mini) / {1e-2, 1e-3, 1e-4}
(full). RK4, h = 1 and the 1/(t_c − t) models (linear in E before t_c) are not
repeated. Linear ZNN must be (and is, to 1e-12) unit-invariant.

### Scale switch and zoo keys

`E3.py` reads `E3_SCALE` (or `ZNNBENCH_SCALE`) ∈ {mini, full} (default `full`,
the convention of the sibling configs). The script form takes `--scale`.
Every activation, model and discretization is resolved at build time. A key
missing from `znnbench.available()` / `activations.available()` is dropped with
a warning, and the optional keys (`implicit_euler_err` in the full T_c sweep,
the CT entries) are included only when present. A rebuild after other
workflows add components therefore picks them up. E3 does not depend on the
pending keys of the concurrent workflow (zead_7i_cang2024, uhlig_5_*,
inverse-free DTZNNs, costs.py): E3 audits activations and gain laws, and its
discretizations are fixed to the schemes the audited papers use (Euler) plus
references.

## 3. Rationale

* **Why a semi-analytic CT reference, not RK4 at tau = 1e-5.** Measured: DOP853
  on the full state stalls (more than 1.7e5 RHS evaluations in 15 s) as soon
  as one of the 16 error components of a non-Lipschitz activation reaches
  zero (sign_power, vaf, predefined_exp). Any explicit scheme has the same
  problem. Moreover, the sign terms of vaf / pvaf / predefined_exp / SPTC make
  fixed-step RK4 chatter at an amplitude of about `gamma tau k_sign sqrt(m)`
  (4e-5 at gamma = 1, tau = 1e-5), which is above 1e-6. The decoupled
  error-space solution is exact for every problem, and the full-state DOP853
  run (stopped before any component reaches zero) checks the implementation.
  RK4 is still run at fixed gamma in `E3_disc` as the discrete-to-CT
  convergence check.
* **Gain scaling bound(gamma) = T_c** is how predefined-time papers turn the
  designer's T_c into parameters (every bound used here scales as 1/gamma).
  This keeps the activation shapes at their published defaults.
* **Cap 1/(gamma tau)**: it is the smallest saturation that makes the
  discrete gain law a fixed-h DTZNN with h ≤ 1 near t_c (h = 1 is
  deadbeat for linear Euler). Any cap ≥ 2/(gamma tau) would leave the
  instability in place.
* **eps = 1e-2** was added to the protocol's {1e-3, 1e-6}. It separates a
  "finite but slow" outcome from "below the chattering floor" in the T_c
  sweep, where most sign-type activations have floors between 1e-3 and 1e-2.

## 4. Deviations from PLAN.md (and why)

1. CT reference: semi-analytic + DOP853 cross-check instead of RK4 with
   tau = 1e-5 (reasons in §3). RK4 at tau → 1e-4 (1e-5 in the full run) is
   kept in `E3_disc` as a consistency check.
2. The initial-error scale is an added dimension of part (a), because uniform
   (fixed/predefined) bounds do not depend on e0 while the actual time does.
   The ratio is therefore e0-dependent and unbounded as e0 → 0.
3. eps = 1e-2 was added (§3).
4. The PLAN does not name a problem for E3; `matrix_inversion` (n = 4) is used
   throughout, the canonical ZNN benchmark. CT times depend only on E(0).
5. "time-varying gains ∝ 1/(T_c − t)" are tested through `sptc_nt_znn` (its
   own `cap`) and `znn` + `strict_predefined` (no cap option). The schedule
   `models/gains.py::PredefinedTimeGain` cannot be passed to `znn` from a
   config, because `gain=` expects an object (see §7). Before t_c the gain
   law of `sptc_nt_znn` is identical.
6. The "varying-parameter" family of the PLAN makes no finite-time claim, so
   it is not part of the conservatism audit (a); its unbounded gain is tested
   in (b3) `E3_vp` (added in the review, §8), with and without the cap.
7. Added in the review (§8): the unit-dependence arm (b4) and the cap controls
   rho = 1.9, 3.

## 5. Mini-run results (results/mini/E3, figures/mini, results/tables/mini)

> **Review note (2026-10-18).** This section is the preparer's first mini run.
> The review rerun (§8) reproduces every number below that it re-computes (same
> run ids for the natural-unit arm), but three statements are SUPERSEDED, see §8:
> (i) the trajectory-level CIs of the TVG divergence fractions (pseudo-replication;
> replaced by cell counts), (ii) "independently of tau" in the TVG mechanism (the
> outcome depends on gamma and T_c/tau), and (iii) the reading of the T_c/tau
> thresholds of the sign-term activations as a time breakdown (they are floor
> limits and unit-dependent). The (b1) rankings hold only in the natural unit.

Run: build 87 tasks (E3_disc 27, E3_tc 5 x 9, CT 15); memguard pool `--workers 3 --max-gb 40`:
**87/87 finished, 0 failed, 325 s wall / 731 s CPU** (machine shared with other workflows, load ~20).
After widening the TVG gain grid, the 45 E3_tc chunks were rebuilt and re-run (134 s wall).
Max per-task RSS: 141 MB (runner) and 164 MB (CT). The analysis takes 7 s.
Outputs: `results/mini/E3/{summary.json,SUMMARY.md}`, `figures/mini/fig_E3_*.{pdf,png}` (6 figures),
`results/tables/mini/E3_*.tex` (5 tables; all compile with booktabs, checked with pdflatex).

**(a) CT conservatism** (gamma = 1; strict models gamma = 2; natural e0, 12 trajectories; bootstrap 95 % CI):

| model | class claimed / derived | exact t(1e-6) s | stated bound s | ratio [95 % CI] | tight (derived) | ratio at e0 = 1e-2 / nat / 1e2 |
|---|---|---|---|---|---|---|
| sign, sign_power, saturated_sign_power, sfaf | finite | 14.3, 7.56, 15.3, 3.13 | e0-dependent | 1.00 [1, 1] | – | 1.00 / 1.00 / 1.00 |
| sign_bi_power | finite / fixed | 4.69 | 15.1 (4 sqrt(e0)) | 3.22 [3.09, 3.37] | 1.03 (uniform 1.28) | 1.01 / 3.22 / 8.31 |
| wsbp (no stated bound) | finite / fixed | 1.77 | – | – | 1.36 (uniform 1.69) | 15.9 / 1.69 / 1.64 (derived) |
| fixed_time | fixed | 1.77 | 3 | 1.69 [1.69, 1.70] | 1.36 | 15.9 / 1.69 / 1.64 |
| vaf | predefined | 0.912 | 3 | 3.29 [3.27, 3.31] | 2.65 | 321 / 3.29 / 3.10 |
| pvaf1 | predefined | 0.674 | 3 | 4.45 [4.42, 4.48] | – | 322 / 4.45 / 4.15 |
| pvaf2 | predefined | 0.630 | 2 | 3.18 [3.15, 3.20] | – | 403 / 3.18 / 2.93 |
| predefined_exp | predefined | 0.569 | 1 | 1.76 [1.75, 1.77] | – | 115 / 1.76 / 1.69 |
| strict_predefined, sptc_nt_znn, sptc_an_foznn | strictly predefined | 0.9998, 0.9998, 1.000 | t_c = 1 | 1.00 | – | 1.00 |

* No bound (stated or derived) is ever below the exact settling time (validity check in every row, all scales).
* Kendall tau_b(stated bound, actual CT time) over the 10 activations with a stated bound = 0.874 (p = 6e-4).
  The bounds rank the activations nearly correctly, but the uniform (fixed/predefined) bounds are
  1.7–4.5x too large at the natural error and 115–403x too large at e0 = 1e-2.
* Reference checks: quadrature bracket satisfied in 100 % of 756 rows x 3 eps. Full-state DOP853:
  63 runs (30 reached 1e-6, 33 stopped at a component < 1e-9). Max relative deviation from the
  semi-analytic reference: 2.9e-6 in ||E(t)|| (at ||E|| ~ 1e-6, i.e. ~1e-12 absolute) and 1.7e-7 in t_eps (90 comparisons).

**(b1) E3_disc** (median over 12 trajectories; ratio = discrete t_eps / exact CT t_eps at the same gamma and x0):
* Euler, fixed gamma = 10. eps = 1e-3: ratio 0.98–1.02 at tau ≤ 1e-3 for linear, sign_bi_power and
  fixed_time; never reached at tau = 1e-2 (floor). The sign-term activations vaf, pvaf1 and predefined_exp
  **never** reach 1e-3 at any tau ≤ 1e-2, because their chattering floor is 2.0e-3 at tau = 1e-4.
  eps = 1e-6 is reached only by sign_bi_power at tau = 1e-4 (12/12; fixed_time 3/12).
* Floor orders (asymptotic-range fit, local slopes): linear 1.00 (Euler at fixed gamma loses one order);
  sign_bi_power and fixed_time 2.00 (= 1/(1-r) for the sig^0.5 term: limit cycle ~ (gamma tau)^2);
  vaf / pvaf1 / predefined_exp 1.05–1.07 (sign term: floor ~ gamma tau k_sign sqrt(m)). RK4 at fixed
  gamma: linear 4.02 (local 0.89, 4.02; the tau = 1e-4 point sits at the 2e-12 round-off/horizon
  floor and is excluded). The non-Lipschitz activations keep orders 2 and 1 even under RK4, so a
  higher-order scheme does not remove the activation floor.
* Euler, fixed h = 0.1: the floor of every finite-time activation is independent of tau
  (sign_bi_power 2.5e-3, fixed_time 1.1e-2, vaf/pvaf1 0.27; predefined_exp diverges). This follows from
  the scale-free limit cycle of e_{k+1} = e_k - h phi(e_k). At h = 1 every finite/fixed/predefined-time
  activation diverges from the natural initial error, while linear ZNN converges 5–8x faster than its
  CT counterpart (Euler with h = 1 is deadbeat).
* Kendall tau_b between the CT ranking and the discrete ranking (eps = 1e-3) is -0.45 to -0.60 for every
  Euler setting: the activations that are fastest in continuous time (sign terms, large exponents)
  are the ones that never reach 1e-3 in discrete time. Only RK4 at tau = 1e-4 restores the CT ranking (tau_b = 1.0).

**eps below the residual floor** (all 3052 non-diverged discrete trajectories): whenever eps is below the
steady-window minimum of ||E||, t_eps is infinite in 100 % of the runs (1236 / 1844 / 2807 runs for
eps = 1e-2 / 1e-3 / 1e-6), although 40 / 37 / 27 % of them dipped below eps earlier (finite first-hit
time: transient or pre-t_c convergence that is later lost to chattering). A paper's time-to-eps claim
therefore needs both eps and tau to be meaningful.

**(b2) T_c vs tau** (claim "t_1e-3 ≤ T_c" holds for ≥ 50 % of the trajectories from the given T_c/tau on):
* constant-gain predefined activations scaled to bound = T_c: vaf, pvaf1, pvaf2 only from T_c/tau = 1e4;
  predefined_exp from 3000; fixed_time (no sign term) from 100. For T_c/tau ≤ 10 they blow up
  (||E(T_c)||/||E(0)|| reaches 1e8, or they diverge). Between 30 and 1e4 the residual at T_c is a floor
  that scales as tau/T_c, coming from the sign term (gamma tau k_sign with gamma ∝ 1/T_c).
* strictly predefined, gamma = 2: znn + strict_predefined holds from T_c/tau = 30 (t/T_c = 7.3 at 3,
  1.1 at 10); sptc_nt_znn reaches eps = 1e-2 from 100 but eps = 1e-3 only near 3000–1e4, because its
  post-t_c noise-tolerance term xi sign(E) (xi = 2.5/gamma) adds a floor of ~10 tau; sptc_an_foznn diverges
  for 3 ≤ T_c/tau ≤ 1000 from the natural initial errors. The conformable gain t^(alpha-1) (t_floor =
  1e-3) and the quadratic term make the first Euler steps unstable, as in the variants spec, Observation 3.
* So the protocol's "breakdown when T_c ≲ 10 tau" holds as a lower bound only: the claim already fails
  at T_c/tau ~ 10–30 for the cleanest strict model, and at T_c/tau ~ 1e2–1e4 for activations with sign terms.

**Time-varying gain 1/(t_c − t)** (sptc_nt_znn, Euler, pooled over T_c; diverged fraction no cap → cap):
gamma = 2: 0 → 0; gamma = 10: 0 → 0; gamma = 50: 0 / 0.40 / 0.40 (tau = 1e-4 / 1e-3 / 1e-2) → 0
(difference CI at tau = 1e-3: [0.27, 0.53]); gamma = 100: 1.00 / 1.00 / 0.60 → 0 (CI [1, 1] at tau ≤ 1e-3).
The residual jump at t_c, ||E(t_c+)|| / min_{t<t_c}||E||, is 1.8e7 (gamma = 50, tau = 1e-4) without the
cap and ≈ 1 with it. Mechanism: the Euler steps t_c − j tau with j < gamma/2 multiply every deviation
from the smooth quasi-steady solution E_j ∝ j tau^2 by A(gamma) = prod_{j<gamma/2} |1 − gamma/j|
(log10 A = 2.1, 13.8, 28.7 for gamma = 10, 50, 100), independently of tau, so round-off alone
(1e-16 x 1e28) diverges at gamma = 100. For gamma ≤ ~20 nothing diverges. With integer gamma
and t_c on the sampling grid, the deadbeat step j = gamma even gives ||E(t_c)|| ≈ 1e-13 (the jump ratio
is then < 1). **Divergence occurs unless the gain is saturated, but only for base gains gamma ≳ 40**,
not for the gamma = 2 of the source papers. The cap removes it in every case: 0 of 720 capped trajectories diverged, against 204 of 720 uncapped ones (strict_predefined, which has no cap, also 204 of 720).

## 6. Full-scale cost estimate and commands (updated in the 2026-10-18 review)

Measured on the review's mini rerun (18:03–18:10, shared machine at load 16–20; pessimistic):
median wall per step Euler 138–141 µs (B·G = 4–16), RK4 294 µs; Taylor–Zhang ~145 µs and
`implicit_euler_err` ~700 µs (resolvent by bisection) from the preparer's in-process probe (not
in the mini grid). Peak RSS per task: 142 MB (runner), 164 MB (CT).

| part | runner specs / pool tasks | steps | est. core-hours |
|---|---|---|---|
| `E3_disc` (12 act. × {Euler, TZ, RK4} × tau {1e-2..1e-5} × {gamma = 10, h = 0.1, h = 1} × 3 seeds; + unit arm s ∈ {1e-2, 1e2}: Euler, gamma = 10 / h = 0.1, tau ≤ 1e-2..1e-4) | 1044 / 131 | 4.3e7 | 2.4 |
| `E3_tc_<Tc>` × 5 (6 const. act. [+ implicit_euler_err, + unit arm] + strict/SPTC × 10 gains + caps rho ∈ {1, 1.9, 3} + SPTC-AN ± cap) | 2655 / 170 | 9.4e6 | 0.6 |
| `E3_vp` (4 act. × ± cap × tau {1e-2, 1e-3, 1e-4} × 3 seeds, T = 12 s) | 72 / 12 | 3.2e6 | 0.13 |
| CT reference (22 entries × 10 seeds × 7 e0 scales × 4 x0 × 2–3 gammas; DOP853 cross-check on scales {nat, 1}, 4 x0) | 22 tasks | – | ≈ 0.15 (mini max full-state wall 2.9 s; the 60 s wall cap per DOP853 run was never hit) |
| **total** | **3771 runner specs; 335 pool tasks** | **5.6e7** | **≈ 3.3 core-hours** (incl. ~1.5 s Python start-up per task) |

* Longest single spec: RK4 at tau = 1e-5, T = 4 s, about 2 min; `balance: true` keeps the longest
  pool task at about 2.5 min.
* **Recommended `--workers 24`** (wall ≈ 10–15 min including the tail; 32 is also safe).
* Peak RSS per task ≤ 0.25 GB (mini 0.14–0.16 GB; the largest full spec records 4e5 × 4 floats
  more). 24 workers need ≈ 6 GB, far below the 200 GB cap; `--max-gb 200` as in PLAN.md.
* Disk: about 1 KB per trajectory row plus traces (1200 points × B·G float32 per E3_tc spec):
  ≈ 150 MB in total.
* Mini rerun for comparison: 157 tasks (142 run + 15 CT unchanged/skipped), 0 failed, 7 min 18 s
  wall with `--workers 3`, 840 s CPU in the specs; analysis 10 s.

Exact commands (run from the repository root `znn-bench/`):

```bash
cd znn-bench
# build (6+1 runner configs + 22 CT tasks + merged pool file; 335 tasks)
python experiments/configs/E3.py --scale full --out results/E3
# launch under the memory guard
python tools/memguard.py pool --tasks results/E3/pool_tasks.jsonl --workers 24 --max-gb 200
# analyse (figures/fig_E3_*, results/tables/E3_*.tex, results/E3/{summary.json,SUMMARY.md})
python experiments/analyze_E3.py --results results/E3 --figdir figures --tabdir results/tables
```

Mini validation (what the review ran):

```bash
python experiments/configs/E3.py --scale mini --out results/mini/E3
python tools/memguard.py pool --tasks results/mini/E3/pool_tasks.jsonl --workers 3 --max-gb 40
python experiments/analyze_E3.py --results results/mini/E3 --figdir figures/mini --tabdir results/tables/mini
```

Runner-only alternative:
`E3_SCALE=full python -m znnbench.runner build experiments/configs/E3.py --out results/E3`
writes the seven runner sub-folders without the CT tasks; then also run
`python experiments/tasks/E3_ct_reference.py build --scale full --out results/E3/ct`
and launch the pool once per `tasks.jsonl`. The build is resumable and idempotent: a chunk whose
specs did not change keeps its task id and done file (CT tasks are keyed by their payload hash).

## 7. Zoo observations (nothing blocking; no zoo file was edited)

No zoo bug was found that invalidates E3. All checks passed: bound validity in every run, the
semi-analytic vs full-state agreement, and the CT closed forms of `strict_predefined`, `sptc_nt_znn` and
`sptc_an_foznn` against `model.rhs` to ≤ 1.7e-7 in t_eps. Observations for the zoo owners:

1. **`wsbp` defaults are identical to `fixed_time` defaults.** WSBP with (r = 0.5, k1 = k2 = k3 = 1) is
   |e|^0.5 + |e|^2 + e, which is `fixed_time` (k1 = k2 = k3 = 1, p = 0.5, q = 2). The two CT rows are
   therefore identical, and two "different" activations in E2/E4/E8 panels would coincide. Suggest a
   distinct WSBP default (e.g. r = 0.25, or the Lv et al. 2018 weights) or a note in docs/ZOO.md.
2. **A time-varying gain cannot be given to `znn` from a config.** `DesignFormulaZNN(gain=...)` expects
   an object, and the runner passes JSON kwargs. So `models/gains.py::PredefinedTimeGain` (and the other
   schedules with `cap`) are reachable only through `vpznn`/`vpznn_exp`/`sptc_*` or a wrapper. Suggested
   (core-change request, not done): accept `gain={"key": ..., **kw}` in the `znn` factory via `gains.get`.
3. **`cap` of `sptc_nt_znn`/`sptc_an_foznn` is absolute.** A stability-relevant cap is 1/(gamma tau), so
   E3 writes one model entry per (gamma, tau) value. A relative option `cap_h` (max effective step)
   would make such sweeps simpler.
4. **`sptc_an_foznn` diverges under Euler from the natural matrix-inversion initial errors** for
   3 ≤ T_c/tau ≤ 1000 at gamma = 2 (start-up singularity t^(alpha-1) with t_floor = 1e-3, plus the
   gamma ||E|| E/(t_c-t)^2 term). This is documented as Observation 3 in docs/spec/variants.md and is
   not a coding error. E3 reports it as a finding.
5. **`sptc_nt_znn` post-t_c floor.** The default xi = 5 Delta / gamma (Delta = 0.5) adds xi sign(E)
   after t_c. Under Euler this is a chattering floor of about gamma xi tau sqrt(m) = 2.5 tau sqrt(m)
   (1e-3 at tau = 1e-4, m = 16), even without noise. Faithful to the source (noise tolerance); worth
   stating in the spec.
6. `PredefinedTimeGain` / `SPTCNTZNN` guard (t ≥ t_c − 1e-12 max(1, t_c)): with T_c/tau integer, the grid
   point k = T_c/tau lands within 1e-16 of t_c and is treated as post-t_c, as intended. No 1e12-size gain
   was observed on the E3 grid.

## 8. Review log (2026-10-18)

Methodology / statistics / ZNN review of E3, with the cross-cutting checklist
(scratchpad `REVIEW_CHECKLIST.md`) applied item by item. Changed: `configs/E3.py`,
`tasks/E3_ct_reference.py` (full CT seeds), `tasks/E3_units.py` (new), `analyze_E3.py`, this
file; all mini outputs regenerated. Nothing under `znnbench/` or `tests/` was touched.

**State found.** `configs/E3.py` had been edited after the preparer's mini run (13:30: cap
controls rho = 1.9, 3, the `E3_vp` config, wider G_TVG), but these were never run, and the
analysis silently ignored the labels `+cap1.9`, `+cap3` and `E3_vp`. They are now run and
analysed.

**Mini rerun.** 157 tasks (142 run + 15 CT unchanged and skipped), 0 failed, 7 min 18 s wall
(`--workers 3 --max-gb 40`, machine load 16–20), 840 s CPU in the specs, peak RSS 142 MB
(runner) and 164 MB (CT); analysis 10 s; all 7 tables compile (pdflatex, booktabs, no overfull
boxes). The 162 natural-unit `E3_disc` runs have the same run ids as the first mini run, and
their metrics are bit-identical.

| # | severity | issue | change | effect on numbers / conclusions |
|---|---|---|---|---|
| 1 | **high** | **Unit dependence (checklist 1).** Every discrete-time statement about the power/sign-type activations (floors vs eps, "SBP and fixed_time reach 1e-3, sign-term activations never", the CT-vs-discrete ranking reversal, T_c/tau thresholds) rested on the one natural unit of E (X(0) ~ U[-2, 2]). | Paired arm (b4): `E3_units.ScaledError`, E_s = s E with s ∈ {1e-2, 1e2}, Euler at gamma = 10 and h = 0.1, and the constant-gain T_c sweep. | Euler, gamma = 10, tau = 1e-3, eps = 1e-3, activations that reach it (of the 5 nonlinear): s = 1e-2: **0**, natural: **2**, s = 1e2: **0**. At s = 1e2 all 5 diverge (12/12), because h Phi'(E0) > 2 in the high-power tail. At s = 1e-2 the floors grow 100x (SBP 2.5e-3, fixed_time 1.0e-2, vaf 2.2). Linear ZNN is identical to 7.6e-13. T_c sweep (tau ∈ {1e-2, 1e-3}): fixed_time holds from T_c/tau = 1000 / 100 / never (s = 1e-2 / 1 / 1e2); vaf, pvaf1 and pvaf2 hold from 1000 and predefined_exp from 300 (eps = 1e-2) in the natural unit only, and never at s = 1e±2. The (a) conservatism ratios were already reported per e0 scale, which for these scale-free activations is the same thing as a unit change; this is now stated in the table and figure. **All (b) rankings are conditional on the unit.** |
| 2 | **high** | **Pseudo-replication (checklist 5)** in the 1/(t_c − t) divergence statistics. The outcome is deterministic per (gamma, T_c, tau) cell (all 12 trajectories agree in 180 of 180 cells), yet the analysis gave trajectory-level bootstrap CIs over 60 pooled trajectories ("0.40 [0.27, 0.53]"). The mechanism was also called "independent of tau". | Cell counts (k of 5 T_c cells) instead of CIs; an outcome map over (gamma, T_c/tau); the peak \|\|E\|\| after t_c is reported as the claim-relevant quantity. | gamma = 50 diverges only at T_c/tau ∈ {10, 30} (2 of 5 cells at tau = 1e-3 and 1e-2, 0 of 5 at 1e-4). For T_c/tau ≥ 100 it stays bounded, but the residual after t_c is 0.16–0.23, against a pre-t_c minimum of 1e-7–1e-8, so the claim is violated without "divergence". gamma = 100 diverges at every T_c/tau ≥ 10. The outcome depends on gamma and T_c/tau, not on tau alone. The qualitative conclusion is unchanged; the CIs are withdrawn. |
| 3 | medium-high | **The T_c/tau breakdown definition** mixed two mechanisms. "Claim holds only from T_c/tau = 1e4" for vaf/pvaf (eps = 1e-3) is a floor limit: eps lies inside the O(gamma tau) chattering band, with gamma = bound/T_c, so the band scales as tau/T_c and depends on the unit. It is not the PLAN's "too few steps" breakdown. | Per-trajectory outcome classes ok / late / floor / diverged. Separate thresholds: time-limited breakdown = largest T_c/tau at which ≥ 50 % are late or diverged; floor-limited. New columns in `E3_tc_breakdown.tex`; text corrected. | Time-limited breakdown (eps = 1e-2) at T_c/tau ≤ 10 for all 5 constant-gain activations, ≤ 3 for znn + strict_predefined, ≤ 1000 for sptc_an_foznn (start-up instability), none for sptc_nt_znn. **The PLAN's "breakdown when T_c ≲ 10 tau" is confirmed as the time mechanism.** The absolute-eps failures above that are floor-limited (vaf: floor ≥ 50 % up to T_c/tau = 3000 at eps = 1e-3). The old summary sentence "fails at 1e2–1e4 for sign-term activations" is superseded. |
| 4 | medium | The "cap prevents divergence" result rested on the deadbeat cap rho = 1 only (best case). | Controls rho = 1.9 (stable, non-deadbeat) and rho = 3 (> 2, outside the Euler interval); `fig_E3_tvg_cap` now shows 4 series. | Diverged cells: uncapped 17/60, rho = 1: 0/60, rho = 1.9: 0/30, rho = 3: **0/30**. Even a cap outside the stability interval prevents blow-up past 1e10, so "no divergence" does not certify a cap. Median peak \|\|E\|\| after t_c at gamma = 100, tau = 1e-3: 0.021 / 0.021 / 0.58 (rho = 1 / 1.9 / 3). At tau = 1e-2 every cap leaves O(1)–O(1e2), dominated by the T_c/tau ≤ 3 cells. The conclusion "divergence unless saturated" holds in the 1e10 sense; the claim-relevant metric is the post-t_c residual. |
| 5 | medium | CT headline CIs ignored the seed clustering (checklist 5): bootstrap over 12 trajectories. | Two-stage cluster bootstrap (seeds, then x0). Full CT run uses 10 seeds (cost +0.1 core-h). | SBP [3.09, 3.37] → [3.09, 3.42]; vaf [3.27, 3.31] → [3.26, 3.31]; fixed_time [1.69, 1.70] → [1.68, 1.70]. Conclusions unchanged. With 3 clusters the mini CIs are indicative only. |
| 6 | medium | The PLAN's "varying-parameter" family was excluded, and its unbounded gain was never tested in E3. | (b3) `E3_vp` run and analysed (`E3_vp.tex`, SUMMARY). | Linear VP-ZNN without cap diverges 12/12. Median divergence time (\|\|E\|\| > 1e10) is 7.9 s at tau = 1e-3 (predicted onset of the instability t* = 7.6 s) and 5.2 s at tau = 1e-2 (t* = 4.2 s). With the cap: 0/12 diverge, MSSRE 8.2e-6 and 8.2e-4. SBP without cap diverges 12/12. With the cap it is 0/12 at tau = 1e-3 (floor 0.26) but 9/12 at tau = 1e-2, at t = 0.06 s. That second failure is the high-power tail at the natural E0 (h Phi'(E0) > 2, unit-dependent), not the VP gain: g(0) = 3 is below the cap. |
| 7 | low | Order fits used tolerance 0.35 and did not flag 2-point fits (checklist 3). | Tolerance 0.15; all-points slope, local slopes and flags reported. | No slope changed. Flagged: predefined_exp (2-point fit, Euler and RK4), and RK4 linear 4.02, which is a 2-point fit after a floor point is dropped (all-points slope 2.46). |
| 8 | low | Kendall p-values were printed as if inferential (checklist 4). | Labelled descriptive (n = 6–10, no multiplicity control). The bound-vs-time tau_b = 0.87 is driven by the exact e0-dependent bounds. | None. |
| 9 | low | Censoring (checklist 2). Medians and quantiles already treated inf as the worst value, but `fig_E3_floor_vs_tau` silently dropped points whose median diverged, and SUMMARY had no counts. | ∞ band in the floor figure; "Censoring" section in SUMMARY (per config: diverged, failed, never-reached counts; failed = 0 everywhere). | None. |
| 10 | low | After adding the band, the floor figure's shared y-range clipped the RK4 points below 1e-7. | Explicit shared range from the cell medians. | Figure only. |
| 11 | note | "eps below the floor is never reached" holds by definition (stay-below t_eps vs steady-window minimum). | Kept as a consistency check, labelled as such. | Informative part: 40 / 37 / 29 % of those runs dipped below eps = 1e-2 / 1e-3 / 1e-6 earlier (first hit). |

**Checked without change.**
(i) *Bounds.* Every stated or derived bound is a bound on the settling time of the scalar ODE
e' = −gamma Phi(e) from max_i |e_i(0)| (element-wise, scaling 1/gamma, time-invariant
activation). It applies to the matrix problem only because the error dynamics decouple
component-wise under the exact dE/dt (the discrete runs use `info = analytic`). The audited
quantity, the time to \|\|E\|\|_2 < 1e-6, lies between the time for the largest component to
reach 1e-6 and the time for it to reach 1e-6/sqrt(m). The bracket check passed in 100 % of rows,
and no bound was below the exact settling time. Re-derived and correct: SBP 2(1+r)/(1−r) and
2 a^(1−r)/(1−r); the WSBP, VAF and fixed_time splits at \|e\| = 1; the PVAF-1 split; the PVAF-2
split at (b/a)^(1/p); predefined_exp 1/k1; the Beta integral (4.84 for SBP, r = 1/2); the strict
closed form E0((t_c − t)/t_c)^gamma.
(ii) *CT reference near finite-time convergence.* The semi-analytic inversion avoids the stall of
adaptive solvers at a non-Lipschitz zero. DOP853 (rtol = atol = 1e-12) is used only until
min_i \|e_i\| = 1e-9, and agrees to 2.9e-6 in \|\|E(t)\|\| and 1.7e-7 in t_eps. The 60 s wall cap
was never hit.
(iii) *Figures (checklist 7).* All 6 PNGs were inspected. Fixed categorical orders come from
`keys.py`; marker, linestyle and fill are secondary encodings; at most 7 series per panel;
legends everywhere; no dual axes; labelled log axes. `fig_E3_tvg_cap` was redrawn (4 cap series,
y = peak \|\|E\|\| after t_c), and the floor-figure fixes are those of items 9–10.

**Confounds (checklist 6), also listed in SUMMARY.md.**
* Unit of E: item 1.
* Equal nominal gamma is not an equal effective gain. The Euler step gamma tau Phi'(e) is
  unbounded for the high-power terms at large \|e\| and for 1/(t_c − t); fixed h equalises only
  gamma tau.
* The T_c-sweep gains are set so that the *stated* bound equals T_c. Because the bounds are
  1.7–4.5x conservative, the CT trajectories reach 1e-3 well before T_c, and the discrete claim
  is tested against a generous target.
* (T_c, tau) pairs with the same ratio are pooled, although the problem's 1 rad/s variation is
  not rescaled with T_c (minor: the truncation term changes, not the floor).
* rho = 1 is the deadbeat step (best case), hence the rho = 1.9 and 3 controls.
* The 12 trajectories per cell are not independent replicates for the deterministic outcomes
  (item 2).
* Horizon: T = max(3 T_c, 0.3 s) ≥ 30 sampling periods after t_c in every cell. Steady window =
  last 25 %, always after t_c. `E3_vp` uses T = 12 s ≥ t*.

**Zoo.** No new zoo bug (the bound formulas were re-derived above). §7 still applies. Additional
observation for the zoo owners (not a bug): the `cap` of `vpznn_exp` / `PredefinedTimeGain` acts
only on the gain schedule, so it cannot prevent the start-up instability of power-type
activations at large \|E(0)\| (h Phi'(E0) > 2). Docs should say that a gain cap is not a step-size
safeguard for non-Lipschitz-at-infinity activations.

**Readiness.** The mini pipeline runs end to end, and every checklist item has been addressed or
documented. Full scale: §6 (335 tasks, ≈ 3.3 core-hours, `--workers 24`, ≤ 0.25 GB per task).
