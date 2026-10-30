# Area spec: structured cost model (`flops`)

Files: `znnbench/costs.py` (new), `tests/test_costs.py`, this spec. No other file is changed;
`wrappers.dense_flops` / `profile_step_cost` are left as they are and are used only for comparison.

Purpose: analytic per-step FLOP counts for E7 (cost–accuracy Pareto) and E9 (matched cost),
counted from what the ZNN-Bench code actually executes. `wrappers.dense_flops` treats J as an
unstructured dense m × n_x matrix. For `matrix_inversion` with n = 128 (n_x = m = 16384) it charges
(2/3) n_x³ ≈ 2.93e12 FLOPs per `solve_J`. The code factorises one 128 × 128 matrix (GETRF =
2n³/3 − n²/2 − n/6 = 1,389,888 ≈ 1.39e6 FLOPs, cached for the whole batch and for every solve at the
same data sample) and does triangular solves with n right-hand sides per trajectory
(128·(2·128² − 128) ≈ 4.18e6 FLOPs).

## 1. API

| function | returns |
|---|---|
| `problem_cost(problem)` | `ProblemCost`: per-call costs `E(B)`, `dEdt(B)`, `solve_J(B)`, `JT(B)` as {category: FLOPs}; `factor()`; `factor_mode` ('cached' / 'call' / 'row' / 'none'); `gen(tag)` and `tags` (data generation); `kin(rows)` (robot); `jac(B)` (problems with a dense `jacobian` method); `table(B)` |
| `model_cost(model, problem)` | `ModelCost.evaluate(B, causal, info, implicit) -> (calls, own)`: the problem calls made inside `rhs` as (prim, rows, state tag), and the model's own numpy FLOPs; `uses_E` (False for `davidenko`) |
| `step_flops(problem, model, disc, B=1, info='analytic', noise=False, data_noise=False, include_data=False, reuse='ideal', skip_unused_dEdt=True, tau=None, k_range=None, amortize_setup=None, skip_unused_E=None)` | steady-state FLOPs of one simulate step: `total`, `linalg`, `by_category`, `by_prim`, `calls`, `rows`, `factorizations`, `data_generations`, `kin_rows`, `dense_equivalent` (the same calls charged by `dense_flops`), `setup`, `notes` |
| `empirical_step_cost(problem, model, disc, B, tau, steps, warmup, ...)` | runs `simulate` through a call-recording `CountingProblem` subclass and observes calls, rows, factorisations, data generations and kinematics rows over whole iterations `warmup .. warmup+steps-1`; returns them with the problem-side FLOPs computed from the observed events and the analytic `step_flops(reuse='exact', skip_unused_dEdt=False)` for comparison |
| `float_reuse_fraction(tau, k_lo, k_hi)`, `scaling_exponent(sizes, costs)`, `activation_flops(act)`, kernels `gemm`, `getrf`, `getrs`, `potrf`, `potrs`, `schur_real`, `trsyl` | helpers |

`B` = number of state rows (trajectories). The rows share the data and therefore the cached
factorisation, so the cost per trajectory decreases with B.

## 2. Counting convention

* **Dense kernels.** Counts of LAPACK's own operation-count routine DOPLA
  (`netlib.org/lapack/timing/lin/dopla.f`, read 2026-10-18; the same counts are tabulated in LAPACK
  Working Note 41). Every formula is also re-derived by hand and tested on small sizes.
  - GEMM (p×q)(q×r): 2pqr.
  - GETRF(n): (n³ − n)/3 mults + (2n³ − 3n² + n)/6 adds = 2n³/3 − n²/2 − n/6.
  - GETRS(n, k): k·n² mults + k·n(n − 1) adds = k(2n² − n).
  - POTRF(n): (n³ + 3n² + 2n)/6 mults (including the n square roots) + (n³ − n)/6 adds
    = n³/3 + n²/2 + n/6.
  - POTRS(n, k): k·n(n + 1) mults + k·n(n − 1) adds = 2kn².
  - Complex arithmetic (DOPLA's factors): 6 FLOPs per complex multiply and 2 per complex add, so
    complex GEMM is 8pqr and complex GETRF is 6·mults + 2·adds. At the numpy level a real × complex
    product counts 2 and a real + complex sum counts 1.
* **Schur / Sylvester.** A real Schur form with its orthogonal factor costs about 25n³ (Golub &
  Van Loan, *Matrix Computations*, Sec. 7.5.6, not re-read; openly stated as "about 25n^3 flops for
  Q and T" by N. J. Higham, *What Is a Schur Decomposition?*, blog post, 11 May 2022, and as a rough
  guide in the Eigen `RealSchur` documentation). ASSUMPTION: this is an **average-case estimate**;
  the number of Francis QR iterations, and hence the true count, depends on the data. The
  quasi-triangular Sylvester solve TRSYL with m×m and n×n factors costs mn(m + n): column-wise back
  substitution, n quasi-triangular solves of order m (m² each) plus the updates with the previous
  columns (mn(n − 1)); leading order, 2×2 blocks and LAPACK's scaling are ignored (cf. Golub & Van
  Loan, Sec. 7.6.3).
* **numpy-level work.**
  - Every element of a floating-point ufunc result counts as 1 FLOP: +, −, ×, /, **, abs, sign,
    sqrt, sin, cos, exp, tanh, minimum, maximum, clip, rint, and the scipy.special ufuncs (betainc,
    betaincinv). ASSUMPTION: a transcendental function counts as 1 FLOP.
  - A reduction counts (inputs − outputs) FLOPs; the real 2-norm along an axis counts 2·size
    (x·x, sums, sqrt); `np.cross` of 3-vectors counts 9 per vector (3 per component).
  - matmul counts 2·out·q FLOPs. einsum with a summed index counts 2 FLOPs per multiply-add.
  - Free: comparisons, `where`, copies, reshapes, broadcasts, concatenations, and operations on
    Python/numpy scalars.
* **Categories.**
  - `algebra`: numpy-level work of E/dEdt/solve_J/JT on the state and on the data arrays the
    primitive receives.
  - `factor`: factorisations.
  - `solve`: LAPACK triangular solves, TRSYL, and batched `np.linalg.solve`.
  - `kin`: robot forward kinematics plus the Jacobian.
  - `data`: generation by the synthetic generator, and the state-independent derivative-data
    terms inside dEdt. It is **excluded** from `total` unless `include_data=True`. The data a
    solver receives is not part of the algorithm.
  - `model`: the model's own work.
  - `disc`: the discretizer's vector updates (and, for the inverse-free schemes, the
    M-matrix work).
  - `sim`: simulate's backward differences for bd1/bd2, and the noise injection.
* **Algorithmic view (default) and code view.** simulate evaluates E(x_k) and dE/dt (or the
  backward differences) at every vector-field call, whether or not the model reads them.
  `skip_unused_dEdt=True` (default) leaves out the dE/dt of models with `uses_dEdt=False` (gnn,
  newton_flow, hgnn_znn without feed-forward), and `skip_unused_E` (default: same as
  `skip_unused_dEdt`) leaves out the E(x_k) of models that never read `ctx.E` (`davidenko`:
  x_dot = −J⁻¹ dE/dt, Simonetto et al. 2016 Eq. (7) needs J and ∇_tx f only). This is the
  algorithmic view, consistent with `experiments/tasks/E7_common.IDEAL_OPS` and the dense fallback
  of `experiments/tasks/E9_cost.py`, which drop the same evaluations. `skip_unused_dEdt=False`
  charges both: that is what the code does (`empirical_step_cost`, `analyze_E7.flops_costs_code`).

## 3. Problem families (one call with B rows)

n is the matrix size and N the KKT size. "cached" factorisations are charged once per data sample
(see §5). Data generation is charged once per sampling instant for the memoised matrix problems
and once per call otherwise.

| key (family) | E | dEdt | solve_J | JT | factorisation | data tags |
|---|---|---|---|---|---|---|
| `linear_system` | 2Bn² + Bn | 2Bn² + Bn | GETRF(n) + GETRS(n, B) | 2Bn² | per call (no cache) | data: A, b; dEdt: Ȧ, ḃ (per call) |
| `matrix_inversion(_right)` | 2Bn³ + Bn² | 2Bn³ | GETRS(n, B·n) | 2Bn³ | cached GETRF(n) | A; Ȧ (memo) |
| `pseudoinverse` (g = Gram size, o = other dim) | 2g²o + 2Bg²o + Bgo | 4g²o + g² + 2Bg²o + Bgo | POTRS(g, B·o) | 2g²o + 2Bg²o | cached 2g²o + POTRF(g) | A; A, Ȧ |
| `sylvester` (n × k) | 2Bnk(n+k) + 2Bnk | same | 4Bnk(n+k) + B·TRSYL(n,k) | 2Bnk(n+k) + Bnk | cached 25n³ + 25k³ + k² | A, B, C; Ȧ, Ḃ, Ċ |
| `lyapunov` | 4Bn³ + 2Bn² | same | 8Bn³ + B·2n³ | 4Bn³ + Bn² | cached 25n³ (one Schur form of Aᵀ) | A, Q; Ȧ, Q̇ |
| `complex_inversion` | 8Bn³ + 4Bn² + 3n² | 8Bn³ + 3Bn² | 3Bn² + zGETRS(n, B·n) | 8Bn³ + 3Bn² + 3n² | cached 3n² + zGETRF(n) | A (Ar, Ai views: 0); Ȧ |
| `nonlinear_equations` guo2018 | 15B | B | 16B (forward substitution) | 36B | none | c; ċ |
| `nonlinear_equations` manufactured | 2Bn² + 4Bn | 2Bn² + 6Bn | 4Bn + B·(GETRF(n) + GETRS(n,1)) | 4Bn + 2Bn² | per row | A, x*; A, Ȧ, x*, ẋ*, ẋ*Aᵀ |
| `tv_unconstrained_opt` zhang2021 / quadratic | 2Bn² + Bn | 2Bn² + Bn | GETRF(n) + GETRS(n, B) | 2Bn² | per call | P, q; Ṗ, q̇ |
| `tv_unconstrained_opt` manufactured | 2Bn² + 4Bn | 2Bn² + 8Bn | 6Bn + B·(GETRF + GETRS(n,1)) | 6Bn + 2Bn² | per row | P, x*; P, Ṗ, x*, ẋ*, ẋ*Pᵀ |
| `tvqp_eq` | 2Bn² + 4Bpn + 2Bn + Bp | same | GETRF(N) + GETRS(N, B) | 2BN² | per call | qp (values and derivatives, per call) |
| `tvqp_ineq` (l = 2n) | 2Bn² + 4Bpn + 4Bln + 3Bn + Bp + 8Bl | 2Bn² + 4Bpn + 6Bln + 3Bn + Bp + 10Bl | 3Bln + 11Bl + B·(GETRF(N) + GETRS(N,1)) | 3Bln + 11Bl + 2BN² | per row | qp |
| `robot_*` (n joints) | kin + 3B | 0 (data: −ṙ_d) | 24Bn (+9B + 9 if damped) + B·(GETRF(3) + GETRS(3,1)) | 6Bn | per row (3×3) | r_d; ṙ_d |

Robot kinematics `kin(rows)` = rows·(n + n·c_link + 6 + 12n). Here c_link = 85 for standard DH
and 82 for modified DH: the transform (10 or 7), p += R p_i (21) and R = R R_i (54). The tool
offset costs 6 and the cross-product Jacobian 12n. Kinematics is charged once per distinct state,
following the one-entry `_kin` cache: E misses, and the following solve_J and JT at the same
state hit.

Generators: `TVMatrix(n, nc)` costs 4n³ + 4nc³ + 2n²nc + 2n·nc² + 5h_n + 5h_nc + 6k, where
h = ⌊·/2⌋ rotation blocks and k = min(n, nc). Its derivative is in `tvmatrix(..., deriv=True)`.
`TVVector` costs 4n (5n for the derivative). `_SymSkewMatrix` costs 2·TVMatrix + 4n². The
`qp`, `c`, path and closed-form examples are O(1) constants (approximate).
`EstimatedDerivativeProblem` replaces dEdt by `window` × E + 2Bm·window. ASSUMPTION: every
filter weight is non-zero. Other problem wrappers are looked through.

## 4. Models (one rhs evaluation)

m = error dimension. c = per-element cost of Φ: linear 0, power 1, bipolar_sigmoid 3,
power_sigmoid 5, power_sum 2N, hyperbolic_sine 2, sign 1, sign_power 4, sign_bi_power 10, sfaf 7,
wsbp 13, saturated_sign_power 5, vaf 11, fixed_time 10, pvaf1 12, pvaf2 10, predefined_exp 12,
strict_predefined 8 (post-t_c), li 9; the model-local activations of `consistent.py`:
signed_power_sum 6 per term, jr_predefined 17 (arctan, exp) / 18 (rational).

Resolvents (`prox`), per element:
- Closed form: linear 1 (+1 per row), sign 5, power p = 3: 9 (+3 per row).
- Every other activation uses the geometric bisection of `Activation.prox` (80 iterations):
  80(c + 8) + c + 13.
- ASSUMPTION: the power p ≥ 5 Newton resolvent is counted as 120.

Error maps of the consistent schemes (`ErrorMapRequest`, per element + per row; the fallback chain
of `consistent.apply_error_map` is followed: 'exact' → 'peb' → 'resolvent', 'peb' → 'exact' →
'resolvent'):
- `resolvent`: the prox above.
- `semi_implicit`: c + 10 (c + 12 and 1 per row when Φ(0+) > 0, the soft threshold).
- `peb`, per power term a·sig^p: 7 (p < 1), 4 (p = 1), 8 (p > 1), plus 2 per row.
- `exact` = closed form + 5 (the clip sign(e)·min(|·|, |e|)): one term p = 1: 1 (+3 per row);
  one term p ≠ 1: the PEB term; bi-power p1 < 1 < p2 (incomplete-beta coordinates): 30;
  Bernoulli forms (p, 1) or (1, p): 8 (+6 per row); `jr_predefined` (JR Eq. (7)): 12 + κ + κ⁻¹
  with κ = 4 / 5 / 6 and κ⁻¹ = 3 / 4 / 3 for arctan / exp / rational (+1 per row); otherwise the
  chain falls back. The constant probes of `power_terms` and `jump0` (a few elements) are ignored.

| model | problem calls in rhs | own FLOPs |
|---|---|---|
| `znn`, `vpznn`, `vpznn_exp`, `znn_consistent` | 1 solve_J | explicit 2B + (2 + c)Bm; resolvent 2B + prox_row·B + (4 + prox)Bm; ErrorMapRequest 2B + map·Bm + map_row·B + (3 + 2(s−1))Bm (error memory) or Bm, + 2Bm |
| `znn_explicit_inverse(_right)` | none | rate (as znn) + one batched n×n matmul 2Bn³ |
| `gnn` | 1 JT | B + cBm + Bn_x |
| `davidenko` / `newton_flow` | 1 solve_J | Bm / B + Bm (`davidenko` never reads `ctx.E`: §2) |
| `ieznn` | 1 solve_J | 3B + 4Bm |
| `dieznn` | 1 solve_J | 7B + 6Bm (ξ = γ); 6Bm (fixed ξ) |
| `hntznn` (L harmonics) | 1 solve_J | B + (c + 4L + 2)Bm |
| `sptc_nt_znn` | 1 solve_J | 2B + 10Bm after t_c (2B + 2Bm before) |
| `sptc_an_foznn` | 1 solve_J | 2B + 10Bm after t_c (B + 9Bm before) |
| `gznn` | JT with B·m rows (dense J); analytic: dEdt with 2B·n_x rows; bd1/bd2: 2/3 more dense-J JT calls | B + 4Bmn + 2Bn + [3Bn² + 2Bnm + 2Bn + 2B] + [2Bmn + Bn] + 2Bmn² (JᵀJ) + n² (β I, built at every call) + B(GETRF(n) + GETRS(n,1)) (+ Bn² when regularised) |
| `lcc_znn` | 1 JT | 3Bn_x + 4Bm + 6B + c_φ·B (c_φ: exp_sig 7, sig 4, sig_lin 5, linear 0) |
| `hgnn_znn` / `fa_gnn_znn` | 1 solve_J + 1 JT | cBm + B + Bm + 2Bn_x (+Bm with feed-forward); fuzzy factor + 2Bm + 23B + 10BG (G = output grid, 1001) |
| any other model | probed from one rhs evaluation (`ProbedModelCost`) | **not counted** (flagged in `notes`) |

`inverse_free_znn` (Chen et al., Neural Netw. 178, 2024) is **not registered**
(`models/structured_extra.NOT_REGISTERED`: primary text not obtainable). Its secondary
restatement (Xiao et al., IEEE TAI 2025, Eq. (10)) equals `make_model('lcc_znn',
activation='linear')`; numbers obtained that way are labelled "LCCZNN (12), linear φ" here.

The SPTC models are counted in their post-t_c phase, which covers most of the horizon.
ASSUMPTION: before t_c, their own work is the smaller figure given in parentheses.

## 5. Discretizations (one steady-state step)

simulate itself calls `data(t_k)` once per step.

| scheme | vector-field calls | extra problem work | own FLOPs |
|---|---|---|---|
| look-ahead `LookAhead` / `ZeaD` / Uhlig / `implicit_euler_err` (s past samples) | 1 causal | – | (2s + 2)Bn_x + 2Bn_z |
| `ab2/3/4` | 1 causal | – | (2q + 2)Bn_x + 2Bn_z (z_scheme='ab': (2q + 2)Bn_z) |
| `rk4` | 4: t_k (causal), t_k+τ/2 twice, t_k+τ | 3 stage samples | 13Bn_x + 13Bn_z |
| consistent schemes (`peb_*`, `exact_flow_*`, `semi_implicit_*`, `implicit_taylor_zhang*`) | 1 causal with an ErrorMapRequest | – | (2s + 2)Bn_x + 2Bn_z |
| `pc_newton` / `pc_gradient` (non-causal) | 1 at t_k | 1 stage sample at t_{k+1}; C corrections: Newton E + solve_J (+Bn_x), gradient E (+JT for the 'normal' direction) (+2Bn_x) | 2Bn_x + 2Bn_z |
| `*_causal` | 1 at t_k (after the corrections, at the corrected state) | C corrections with the measured t_k sample | 2Bn_x + 2Bn_z |
| `newton_tracking(_causal)`, `running_gradient` | 0 (the model is not used) | C corrections | – |
| `broyden_dtznn_zhang2012` | 1, with solve_J → M r (2Bn_x² each) | E at x_k and x_{k−1} (secant pair) | 7Bn² + 8Bn + Bm + 3B (secant update) + 2Bn + 2Bn_z |
| `cdtznn_mahuang2025` | 1, with solve_J → M r | 3 dense Jacobians (`jacobian`, or JT with B·m rows); analytic d_tJ·M: 2 dEdt calls with B·n rows (bd1/bd2: 1/2 more dense J) | ≈ 8Bn³ (four batched matmuls) + O(Bn²) (exact list in `_Engine.cdtznn`) |
| `QuantizedDiscretizer` | as the wrapped scheme | – | + 8Bn_x + 8Bn_z (4 FLOPs per element: /q, rint, clip, ·q, on entry and exit) + 4B(n_x + n_z) per vector-field call when the derivative is quantized (one per Broyden / CDTZNN step) |

`pc_gradient(alpha='auto')` runs a one-off Lipschitz estimate (30 power iterations of 2 E calls,
plus JT for the 'normal' direction). Its own work is 30·(11Bn_x + 2Bm + 4B) ('normal') or
30·(8Bn_x + 5Bm + 4B) ('residual') plus 3Bn_x + B − 1, and robot problems re-evaluate the
kinematics at every E and JT call. It is reported under `setup` and amortised only on request.
A discretization that is not in the table raises `NotImplementedError` in `step_flops`;
`empirical_step_cost` still observes it.

**Data-sample reuse** (only RK4 and the non-causal prediction–correction schemes are affected).
The matrix problems memoise data per float time and cache the factorisation by data value. The
stage sample at t_k + τ is reused at the next step's t_k only when `k*tau + tau == (k+1)*tau` in
floating point.
- `reuse='ideal'` always reuses. RK4 then costs 2 factorisations per step and `pc_newton` 1.
- `reuse='none'` never reuses (RK4 3, `pc_newton` 2).
- `reuse='exact'` mixes the two with the exact coincidence rate of `float_reuse_fraction(tau, …)`.
  For τ = 0.01 over k < 1000 this rate is between 0 and 1; for a binary-exact τ = 2⁻⁶ it is 1.
- With `data_noise=True`, every stage sample is a new realisation. RK4 then costs 4
  factorisations per step: stages 2 and 3 have different noise.
- None of this changes an E7 or E9 number: every E7 method is causal, and the E9 problems
  (linear system, TV optimisation, TV-QP) have no factorisation cache.

## 6. Verification log (all in `tests/test_costs.py`, 139 tests, ≈ 3 s, ≈ 80 MB peak)

1. **Hand counts.** The kernels are checked on 2×2 cases: GETRF(2) = 3, GETRS(2,1) = 6 and
   POTRF(2) = 5. Whole steps are hand-counted for `matrix_inversion` n = 3, B = 2 with znn + Euler
   (= 449 FLOPs) and for `linear_system` with gnn + Euler. The n = 128 headline is checked against
   `dense_flops`.
2. **ufunc-level counter.** An `ndarray` subclass counts every floating ufunc element, matmul,
   einsum, real 2-norm, clip and cross the code executes, with the complex convention of §2 for
   complex results. Module-level numpy proxies (activations, variants, structured_extra,
   consistent, inverse_free_dtznn, prediction_correction, robot) keep the subclass through
   `np.asarray` / `np.array` and the allocations. The analytic counts equal the counted ones
   **exactly** for:
   - E, dEdt and JT of 18 problem instances and of `complex_inversion`; solve_J with a warm factor
     cache. The cold-minus-warm difference equals the numpy part of `factor()`: the Gram product
     for pseudoinverse, −B for Sylvester, Ar + 1j·Ai for complex inversion.
   - Robot kinematics and E / solve_J / JT of PUMA 560 (standard DH), Panda (modified DH) and the
     damped variant (the plain-numpy `damping² I` is the only invisible term).
   - The generators `TVMatrix` (square, tall, wide, symmetric; values and derivatives) and
     `TVVector`, with their parameter arrays counted.
   - The own work of 19 model configurations (explicit, resolvent, nonlinear activations, GZNN
     analytic/none) and of 9 `lcc_znn` / `hgnn_znn` / `fa_gnn_znn` configurations.
   - Φ and prox of all 18 registered activations, and the ErrorMapRequest rates of every
     registered activation and 6 model-local configurations under the four error maps.
   - The vector updates of 11 discretizers in steady state, and the `alpha='auto'` set-up.
3. **Whole steps through the real `simulate` loop.** The counter runs inside `simulate` (problem,
   model and discretizer proxies) for 19 E7/E9 method combinations, bd1/bd2 and additive noise:
   counted = `step_flops` (code view) − factor − solve + simulate's divergence check
   (|x|.max(axis=1): 2Bn_x − B), exactly.
4. **simulate cross-check (`empirical_step_cost`).** Calls, rows, factorisations, kinematics
   rows and problem-side FLOPs agree exactly in 15 representative cases (development sweep:
   2,751 runs over 15 problem instances, 16 models, 13 discretizations). The only differences are
   data-generation counts (see §7). At τ = 0.01 the observed factorisations of Lyapunov lie
   between the 'ideal' and 'none' models.
5. **`profile_step_cost` (call counts).** Steady-state counts (difference of two run lengths, as
   `experiments/tasks/E9_cost.py` does) agree **exactly** for every E7 method (13, matrix
   inversion) and every E9 method (20) on the three E9 problem families, and with noise and
   bd1/bd2. A looser 12-case test over 100 whole steps (start-up included) is kept.
6. **Scaling.** The `matrix_inversion` znn + Euler total and the factor + solve part both scale
   with exponents in (2.9, 3.05). The dense-model equivalent scales with (5.9, 6.05). The
   Sylvester factorisation is ~n³, and `gznn` on matrix inversion is ~n⁶ (JᵀJ of an n² × n²
   Jacobian). The per-trajectory cost decreases with B because the factorisation is shared.
7. **Mutation test.** 49 hand-made mutants of `costs.py` (wrong exponent, missing factor 2,
   dropped term, wrong stage count, cache logic): all are killed (see the Verification section).

## 7. Known limits and ASSUMPTIONS

* **ASSUMPTION:** a transcendental function counts as 1 FLOP. Comparisons and selections are
  free; `minimum`, `maximum` and `clip` count 1.
* **ASSUMPTION:** Schur = 25n³ (average-case estimate) and TRSYL = mn(m + n) (leading order).
* **Memo clearing.** The matrix problems' data memo is cleared when it holds 64 entries. This
  occasionally regenerates data: up to +1/8 generation per step was observed for Sylvester,
  Lyapunov and complex inversion with RK4 or `gznn`. It affects only the `data` category, which
  is excluded by default. The model counts one generation per distinct (tag, instant).
* **Value-dependent cache hits.**
  - The factorisation cache is keyed by values. It can hit when t_k + τ and t_{k+1} differ by one
    ulp but give bitwise-equal data; this was observed for Lyapunov at τ = 0.01.
  - The robot's one-entry kinematics cache can hit when a converged Newton correction leaves x
    bitwise unchanged.
  - In both cases the model charges a miss, so the reported value is an upper bound under
    `reuse='exact'`/`'none'`. Use a binary-exact τ when comparing exactly.
* **Jittered and delayed data.** `JitteredProblem` and `DelayedDataProblem` are looked through.
  Their changed sampling times are not modelled for reuse, so 'ideal' can under-count
  factorisations under jitter.
* **Inverse-free schemes.** The schedules for `broyden_dtznn_zhang2012` and `cdtznn_mahuang2025`
  follow `discretizations/inverse_free_dtznn.py` as read on 2026-10-18. The CDTZNN non-design-
  formula path assumes the regime falls back to simulate's `info`. `problem.jacobian` is not a
  CountingProblem method, so its cost (category `algebra`, `by_prim['jacobian']`) is excluded
  from the empirical comparison.
* **Models without a cost class.** Their problem calls are probed from one rhs evaluation and
  their own numpy work is not counted (`notes` says so). Within today's registry this applies to
  no model.
* **Start-up and one-off costs.** Start-up steps (history fill, bd2, Broyden's first step) and
  one-off set-up are not in the steady-state numbers. `alpha='auto'` is available under `setup`.
* **Constant and invisible terms.** B- and m-independent probes (the `power_terms` and `jump0`
  checks of an activation, a few elements) are not counted; neither is the noise generator.
* FLOPs are not wall time. E7 measures wall time separately (`runner` lean timing mode).

## 8. E7 / E9 reference numbers (B = 1, steady state, data generation excluded)

Algorithmic view (default); where the code view (`skip_unused_dEdt=False`) differs it is given in
parentheses. `matrix_inversion`, FLOPs per step:

| n | n_x | znn+euler | znn_explicit_inverse+euler | znn+rk4 | davidenko+pc_newton(C=1) (= pc_newton_causal) | gnn+euler | lcc_znn (linear φ)+euler | dense_flops (znn+euler) |
|---|---|---|---|---|---|---|---|---|
| 4 | 16 | 516 | 498 | 1.95e3 | 594 (738) | 353 (481) | 582 | 4.27e3 |
| 16 | 256 | 2.87e4 | 2.64e4 | 1.09e5 | 3.61e4 (4.46e4) | 1.79e4 (2.61e4) | 2.77e4 | 1.16e7 |
| 64 | 4096 | 1.77e6 | 1.60e6 | 6.72e6 | 2.28e6 (2.81e6) | 1.07e6 (1.60e6) | 1.62e6 | 4.59e10 |
| 128 | 16384 | 1.41e7 | 1.27e7 | 5.35e7 | 1.82e7 (2.24e7) | 8.49e6 (1.27e7) | 1.28e7 | 2.93e12 |

The n = 128 znn + Euler total, 14,071,106, is hand-counted in the Verification section. The
default `lcc_znn` (activation (13)) adds 7 FLOPs per step.

`linear_system` with n = 16, FLOPs per step:

| scheme | FLOPs |
|---|---|
| znn + euler | 4250 |
| znn + zead_5i | 4346 |
| znn + uhlig_4_6 | 4506 |
| gnn + euler | 1121 (code view 1649) |
| davidenko + pc_newton(C=1) | 7312 (code view 7840) |
| znn + rk4 | 16952 |

On these sizes, matched-cost comparisons for E9 come down to counting vector-field calls and
factorisations. The higher-order look-ahead formulas add only (2s + 2)Bn_x per step.

## Verification (adversarial, 2026-10-18)

Independent re-check of every count against the executed code (problems, models,
discretizations, `wrappers.CountingProblem` / `profile_step_cost` / `dense_flops`). Scratch
scripts: an independent ufunc counter run inside the real `simulate` loop, steady-state
`profile_step_cost` differences, and a mutation harness. Verdicts: correct / fixed /
wrong-unfixed / unverifiable.

| # | item | verdict | note |
|---|---|---|---|
| 1 | `inverse_free_znn` in tests (`EMPIRICAL_CASES`, `KNOWN_MODELS`) and spec §4 / §8 | fixed | key de-registered (`structured_extra.NOT_REGISTERED`); replaced by `lcc_znn` with `activation='linear'`, labelled "LCCZNN (12), linear φ" (the Xiao et al. 2025 restatement, not "Chen et al. 2024"). §8 values unchanged. |
| 2 | GETRF / GETRS / POTRF / POTRS and the complex factors 6 / 2 | correct | checked against LAPACK's DOPLA routine (netlib `lapack/timing/lin/dopla.f`), statement by statement; the "LAWN 41 from memory" ASSUMPTION is replaced by this source. |
| 3 | textbook leading orders: LU 2n³/3, n² per triangular solve, Cholesky n³/3 | correct | leading terms of item 2. GETRF(128) = 1,389,888 (§1 said ≈ 1.40e6: rounding fixed to 1.39e6). |
| 4 | real Schur 25n³ | unverifiable (ASSUMPTION kept) | an average-case estimate: the Francis iteration count is data dependent. Golub & Van Loan Sec. 7.5.6 was not re-read. The figure is openly stated by Higham (2022, "about 25n^3 flops for Q and T") and, as a rough guide, in Eigen's `RealSchur` docs. §2 now says "average". |
| 5 | TRSYL mn(m + n) | correct (leading order) | derived from column-wise back substitution: n solves of order m (m² each) plus updates mn(n − 1). |
| 6 | per-call algebra of all 13 families | correct | as in §6.2. Complex inversion and robot kinematics were flagged approximate or "hand-counted, not reachable by the counter". Both are now counter-checked and exact; the flags are removed. The generators were only self-checked (`gen == C.tvmatrix(...)`); they are now counter-checked. |
| 7 | scaling exponents | correct | matrix inversion ∝ n³ (E, dEdt, JT: 2Bn³; one LU; n triangular solves); mat-vec problems ∝ n²; dense model ∝ n⁶. |
| 8 | call counts vs `profile_step_cost`, every E7 / E9 combination | correct | exact steady-state equality: 13 E7 methods on matrix inversion n = 2, 3 and 26 E9 (method, C) combinations on linsys n = 8, tvuo n = 8, tvqp 6 + 2. `empirical_step_cost` is exact for all of them. A test covers the same tables at small n (C = 1, 3). |
| 9 | whole-step FLOPs inside `simulate` | correct | 99 E7/E9 combinations (E7 n = 2, 3, 4; E9 at its paper sizes) agree exactly with an independent counter; a 25-case version is in the tests. Remaining raw differences were traced to counter blind spots (`np.asarray` / `np.array` in activations and `dense_jacobian`, plain-numpy gain), not to the model. |
| 10 | `davidenko`: E(x_k) is evaluated by simulate but never read | fixed | the default "algorithmic" view charged it, unlike `E7_common.IDEAL_OPS` and the `E9_cost.py` fallback. New `skip_unused_E` (defaults to `skip_unused_dEdt`); the code view is unchanged. |
| 11 | consistent error maps (`peb`, `exact`, `semi_implicit`) | fixed | they were approximate and untested: peb 7 per term, exact 45 for two terms, semi-implicit c + 12. They now follow the code's fallback chain with exact per-element and per-row counts (§4). `jr_predefined` and `signed_power_sum` costs were added. Tested for all activations × 4 maps × 2 formulas; not used by E7 / E9. |
| 12 | `gznn`: β I built at every call | fixed | + n_x² per call (it was charged only when regularised). |
| 13 | quantized derivative under Broyden / CDTZNN | fixed | the vector-field call of these schemes was not counted (+4B(n_x + n_z) per step). |
| 14 | `alpha='auto'` set-up | fixed | it was approximate (10Bn + 4Bm per iteration); it is now exact (§5) and counter-checked. The robot's J^T kinematics miss ('normal' direction) was added. |
| 15 | data-sample reuse | correct, irrelevant for E7 / E9 | every E7 method is causal; the E9 problems have no factorisation cache. |
| 16 | test meaningfulness | fixed | 49 mutants: the suite as received (approximated by deselecting the new tests) killed 33. Survivors included a factor 2 in complex E, robot link products, generator products, the bd2 differences, the noise solve and the PC / Broyden / CDTZNN own work. The strengthened suite kills 49 / 49. |
| 17 | hand count n = 128, znn + Euler | correct | (2n³ + n²) + 2n³ + GETRF(n) + n(2n² − n) + (2 + 2n²) + 4n² = 14,071,106 = `step_flops`. |
| 18 | `lcc_znn` default under the E7 protocol | wrong-unfixed (outside `costs.py`) | activation (13) multiplies by exp(\|ν\|) with ν0 = ‖AX0 − I‖²/2 ≈ 230–4000, which overflows. Every trajectory diverges at the first step (n = 4, 16, any h); the E7 `lcc_znn` row will be all divergent. Costs are unaffected (c_φ = 7). |
| 19 | `profile_step_cost` when every trajectory diverges | wrong-unfixed (outside `costs.py`) | it divides by N, not by `steps_run`. The per-step calls of a run that stopped early are then too small (seen with `uhlig_3_4` at h = 0.3). Use stable gains when profiling. |

Changed reference numbers (§8): `davidenko` + `pc_newton(C=1)` / `pc_newton_causal(C=1)` on matrix
inversion, now 594 / 3.61e4 / 2.28e6 / 1.82e7 for n = 4 / 16 / 64 / 128 (was 738 / 4.46e4 /
2.81e6 / 2.24e7, now the code view). On `linear_system` n = 16 it is 7312 (was 7840). Causal
PC-Newton (C = 1) over ZNN-Euler at n = 128 is 1.29× (was 1.59×). The `inverse_free_znn` column
is relabelled; the other numbers are unchanged.
