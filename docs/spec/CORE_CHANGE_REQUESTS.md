
## runner

Requested by the runner area (`znnbench/runner.py`). Every item below is optional. The runner
already works around each one, and none of them blocks the sweeps.

1. **`simulate`: return the number of steps actually executed.** When every trajectory diverges,
   the loop `break`s early, but the result still reports `steps = N`. That makes
   wall-clock-per-step wrong for diverged runs.
   *Workaround:* the runner computes `steps_run = round(max(diverged_at) / tau)` when all
   trajectories diverged.
   *Request:* add a `"steps_run"` key to the returned dict.
2. **`simulate`: independent RNG streams for x0 and noise.** A single `default_rng(seed)` currently
   draws x0 first and then all noise and data-noise samples. As a result, the noise seen by
   trajectory b depends on how many trajectories share the batch. So when `batch_gains` runs G
   gains as one batch of B·G trajectories, the results are not bitwise identical to G separate runs
   whenever noise is stochastic or `data_noise > 0`. They are statistically equivalent, and the runs
   agree exactly when noise is deterministic.
   *Request:* derive the streams as `ss = np.random.SeedSequence(seed); rng_x0, rng_noise =
   [np.random.default_rng(s) for s in ss.spawn(2)]`. Better still, give each trajectory its own
   noise stream, so that batch composition no longer matters. This would change existing numbers,
   so it should land before the sweeps.
3. **`metrics.summarize`: `t_eps_{e:g}` column names contain a dot** (for example `t_eps_0.001`).
   This is awkward for `DataFrame.query` and attribute access.
   *Workaround:* none needed, because the columns are used as they are.
   *Suggestion (low priority):* add an alias such as `t_eps_1e-03`. This is cosmetic only.
4. **`simulate`: the `wall` timer includes the diagnostics** (added by the runner verifier). The
   timer covers the per-step residual and `x_star(t)` evaluations. These cost about 35–40 % of a
   ZNN-Euler step at n = 32 (measured on `matrix_inversion` and `linear_system`), and they would
   bias the cost comparisons of E7.
   *Workaround:* `timing_repeats: R` runs R extra lean simulations (`record_every = N`, no
   solution error) and reports their median wall time. The dynamics are unchanged, because
   recording draws no random numbers; a test asserts this.
   *Request (optional):* also return `wall_dynamics`, the time spent in `disc.step` alone, so that
   a single run gives a clean timing.

## problems_matrix

*Non-blocking performance suggestion (no workaround needed; nothing in core is wrong).* **[DONE by the integrator, 2026-10-18: `SmoothOrthogonal._R` is vectorised. Its output is bitwise identical to the loop for n in {1,...,64} and t in [0, 1e3], and R plus R_dot at n = 64 drop from 69 to 15 µs.]**

- `problems/_tv.py::SmoothOrthogonal._R` builds the block-diagonal rotation with a
  Python loop over the n/2 blocks (allocating a 2x2 array per block). At n = 64 this
  loop is the largest single cost of the Sylvester problem (four `TVMatrix` objects, each
  evaluated with `__call__` and `dot`): about 1.5 of the 3.3 ms/step measured for
  `sylvester` at n = 64, B = 1 (profiled with cProfile). A vectorised version, e.g.
  `c, s = np.cos(w*t), np.sin(w*t)`; `i = 2*np.arange(len(w))`;
  `R[i, i] = c; R[i, i+1] = -s; R[i+1, i] = s; R[i+1, i+1] = c` (and the derivative analogue),
  gives identical values and roughly halves the per-step cost at n = 64. The owner of `_tv.py`
  should decide. `problems/matrix.py` already memoises A(t), A_dot(t) per t, so each matrix is
  evaluated at most once per sampling instant.

## noise_info

Requested by the noise / information / arithmetic area (`znnbench/noise/advanced.py`,
`znnbench/wrappers.py`). Every item is optional. The area works around items 2 to 5, and item 1
blocks only one noise class from the literature, which is therefore not implemented.

1. **State-dependent ("vanishing") noise needs E.** The literature's *dynamic bounded vanishing
   noise* class, y_i(t) = 0.4 eps_i(t), is used by Zhao et al. (2025), AIMS Math. 10(6):13974,
   doi:10.3934/math.2025628. Jin et al. (2022), Front. Neurorobot. 16:1065256, use the same class
   with |n_ij| <= delta |e_ij|. Neither can be written as `noise(t, rng, shape)`.
   *Request:* if the noise callable has `state_dependent = True`, have `simulate.f` call
   `noise(t, rng, shape, E=ctx_E)`.
   *Workaround:* none, so this class is not implemented.
2. **Exact data for the recorded residual.** `simulate` records `problem.residual(x,
   problem.data(t))`. When a wrapper changes the data the model sees (jitter, delay, measurement
   noise), it must override `residual` and recompute the exact data. That costs an extra `data()`
   call per step.
   *Request:* record the residual with `getattr(problem, "exact_data", problem.data)(t)`.
   *Workaround:* ~~`ProblemWrapper.residual` uses `inner.data(d["t_nominal"])`~~ **[updated by the integrator, 2026-10-18:]** `ProblemWrapper.residual` uses `ProblemWrapper.exact_data(t)`, which bypasses the `data` method of every wrapper in a chain. That is exactly the hook name requested here.
3. **Step index for `data()`.** Per-step i.i.d. quantities, such as jitter delta_k, have to be
   keyed on the float t. RK4 stage times create round-off duplicates of grid instants, for example
   `0.065 + 0.005 != 7 * 0.01`. In an alpha-beta estimator that duplicate produced a near-zero dt
   and made the run diverge.
   *Request:* pass `k` to `data(t, k=None)`, or compute stage times as `(k + c) * tau`.
   *Workaround:* times are canonicalised by rounding to 1e-11 s (`wrappers._tkey`).
4. **`data_noise` is applied after `problem.data`.** A wrapper that estimates dE/dt from sampled
   data therefore cannot see the noisy samples the model sees.
   *Workaround:* `EstimatedDerivativeProblem(meas_std=...)` generates its own measurement noise
   and feeds the same noisy samples to E, solve_J, JT and the estimator. Use it with
   `data_noise=0`.
   *Request (low priority):* express measurement noise as a problem wrapper, or add a hook.
5. **Reset hooks.** Stateful noises (`random_walk`, `impulse`) and `EstimatedDerivativeProblem`
   currently detect a new run because time goes backwards.
   *Request:* at the start of `simulate`, call `noise.reset()` and `problem.reset()` when they
   exist.

## problems_robot

Requested by the robot-tracking area (`znnbench/problems/robot.py`). This is optional and
non-blocking; the area already works around it.

1. **`simulate`: return the final state (and, optionally, a sparse state trace).** The
   repetitive-motion metric of redundant-manipulator tracking is the joint drift
   ||theta(T) - theta(0)||. Computing it needs the final joint state, but `simulate` returns only
   the residual and solution-error traces, and `sol_err` is unavailable here because `x_star` is
   None.
   *Workaround:* `robot.StateRecorder` is a transparent `Discretizer` wrapper that stores
   `x_first`, `x_last` and every k-th state. `robot.simulate_drift(...)` wraps `simulate` with it
   and returns `(sim, drift)`.
   *Request:* add `"x_final"` (B, n_x) to the returned dict, plus an optional
   `record_states_every` argument that stores a (N_rec, B, n_x) trace. Any problem with
   x_star = None, such as robot tracking or QPs whose multipliers are not unique, would benefit.

## problems_opt

Non-blocking; everything works without these changes.

1. **Optional `Problem.solution_error(x, t)` hook used by `simulate`.** Today `simulate` records `||x - x_star(t)||` over the whole state. For KKT-type problems (`tvqp_eq`, `tvqp_ineq`) the state is `[x; lambda; mu]`. When LICQ is nearly violated, the multipliers can be ill-conditioned or unbounded even though the primal solution is fine; `tvqp_ineq variant="yang2025"` is such a case near t = pi/2 + k*pi. *(Integrator note: per the problems_opt verification, the printed `yang2025` data are not the data behind Yang et al.'s figures, which used the `zhang2021` benchmark. The request stays valid for the printed-data variant.)* Proposal: `simulate` calls `problem.solution_error(x, t)` when it exists and falls back to the current norm otherwise. The QP problems would then report the primal error `||x[:, :n] - x*_primal||`. Workaround for now: use the `qp_solution(t)` / `smoothed_solution(t)` methods and slice `[:n]` offline.
2. **(Info)** `tvqp_ineq.x_star(t)` runs a cached Newton solve per new t, about 40 µs. Sweeps with tau = 1e-4 and T = 10 make 1e5 calls. With `record_every > 1`, `simulate` already queries `x_star` only at recorded steps, so no change is needed.

## structured_baselines

Requested by the structured_baselines area (`models/structured.py`, `models/baselines.py`,
`discretizations/prediction_correction.py`). Everything works today through the workarounds
described below. None of these requests blocks anything.

1. **Give discretizers access to the problem and the measured sample.** The prediction–correction
   schemes (`pc_newton`, `pc_gradient` and their `_causal` variants, `newton_tracking*`,
   `running_gradient`; Simonetto et al., IEEE TSP 2016) correct on the frozen problem with
   E / solve_J / JT. They therefore need the `Problem`, the measured data at t_k (for the causal
   variants), and the stage-data noise model (for the non-causal ones). The `Discretizer.step`
   signature passes none of these.
   *Workaround:* `PredictionCorrection.bind(problem)` when the caller uses it. Otherwise the scheme
   reads `problem`, `past`, `data_noise` and `rng` from the closure of `simulate`'s vector field
   `f`, searching up to 3 nested wrapper closures. If neither works it raises a `RuntimeError` that
   names `bind`. Renaming those local variables in `simulate` would silently fall back to
   `problem.data(t)` exact data, or raise if `problem` is renamed.
   *Request:* in `simulate`, right after `disc.reset()`, add
   `if hasattr(disc, "bind_env"): disc.bind_env(problem=problem, measured=lambda: past[0],
   stage_data=lambda t: _perturb(problem.data(t), data_noise, rng))`.
   Alternatively, pass these as attributes of a small object given as a keyword argument to
   `step`. Wrapping discretizers (`wrappers.QuantizedDiscretizer`) should forward `bind_env`.
2. **Capability contract for matrix inversion (clarification for the docstring of
   `Problem.capabilities`).**
   - `'matrix_inverse'` ⇔ x = vec_row(X) with X ∈ R^{n×n} and E(x, d) = vec_row(A(t) X − I).
     Hence dE/dt = vec_row(Ȧ X), and J^{-1}(R) = A^{-1} R ≈ X R.
   - `'matrix_inverse_right'` ⇔ E = vec_row(X A(t) − I).

   `problems/matrix.py` already follows this. `znn_explicit_inverse` and
   `znn_explicit_inverse_right` rely on it.
3. **Capability `'gradient_field'`** (new, optional). A problem declares it when E = ∇_x f for a
   strongly convex f, so that J is symmetric positive definite. The gradient correction of
   `pc_gradient` then uses Simonetto's exact Eq. (8) step x − αE instead of the generic x − αJᵀE.
   Optimisation problems whose E is a gradient (unconstrained TV optimisation, for example) may
   want to declare it. KKT systems with constraints are not gradient fields and should not.
4. **Optional `Problem.Jv(x, d, v)`** (Jacobian–vector product). It is used only by the
   power-iteration step-size estimate of `pc_gradient` with `alpha='auto'`. Without it, a central
   difference of E is used, which is exact up to round-off for E affine in x. Not needed by any
   core code.
5. **(Added by the structured_baselines verifier) Optional model-supplied initial state.**
   `znn_explicit_inverse(_right)` (Getz–Marsden dynamic inverter) converges only locally
   (Getz–Marsden 1995, Thm 2.1). From the matrix problems' default X(0) ~ U[−2, 2] it diverges in
   finite time for most draws (verified: Euler, n = 3, γ = 30, τ = 1e-2 → all trajectories
   diverge), while from the standard start X(0) = Aᵀ(0)/(‖A‖₁‖A‖_∞)
   (`models.structured.newton_schulz_init`) it tracks with MSSRE 8e-4. The runner passes
   `x0=None`, so sweeps record divergence that reflects the initialisation, not the model.
   *Request:* an optional `Model.x0(problem, rng, B)` hook that `simulate` (and the runner) uses
   when `x0` is None, or a runner spec field for an initialiser key. *Workaround:* benchmark
   configs must pass `x0=newton_schulz_init(A(0))` for this model.
6. **(Info, verification finding) `simulate._perturb` also perturbs non-data keys such as
   `'t_nominal'`** (added by the jitter/delay wrappers). The PC schemes no longer depend on the
   time stamp of the measured sample, so this does not affect them.

## consistent

Requested by the consistent-discretization area (`znnbench/discretizations/consistent.py`, spec `docs/spec/consistent.md`). None of these items blocks anything. A workaround is in place: the opt-in model key `znn_consistent`.

1. **`models/znn.py::DesignFormulaZNN.desired_Edot`: honour a rich implicit request.** This is a 3-line change:
   ```python
   def desired_Edot(self, ctx):
       g = ctx.gamma * self.gain(ctx.t)
       if hasattr(ctx.implicit, "desired_Edot"):      # ErrorMapRequest (consistent.py)
           return ctx.implicit.desired_Edot(self.act, ctx, g)
       if ctx.implicit:
           ...                                          # unchanged resolvent branch
   ```
   *Effect:* the base `znn` key and every subclass (for example `znn_explicit_inverse`) would use the consistent error maps (`peb`, `exact`, `semi_implicit`) and the error-memory look-ahead rate of the `consistent` discretizers. `ConsistentZNN` would become a plain alias.
   *Numerical impact:* none on existing discretizers, because they pass `implicit` as a bool.
   *Workaround today:* use model `znn_consistent`. With the base model, the consistent discretizers fall back to the resolvent rate and emit a `RuntimeWarning`.
2. **`core.Context.implicit` (documentation only).** State that a discretizer may pass a truthy request object instead of `True`. `simulate` already forwards the value unchanged.
3. **[WITHDRAWN by the integrator, 2026-10-18: stale. `activations.register` now raises on a duplicate key, and `consistent.py` no longer registers any activation globally; `signed_power_sum` and `jr_predefined` are resolved locally.]** **`activations.register` has no duplicate-key check.** Unlike `registry._reg`, it silently overwrites an existing key. Raising on duplicates is suggested. Meanwhile, `consistent.py` registers `power_sum` and `jr_predefined` only if the key is free.
4. **Optional activation hooks.** Activations can expose `power_terms() -> [(a_i, p_i)]` (Φ = Σ a_i sign(e)|e|^{p_i}) and/or `exact_step(e, h)` (the exact one-step flow of de/dt = -Φ(e)). The `peb` and `exact` maps then apply to them automatically. Without the hooks the maps fall back to the resolvent (warned once).
5. **[DONE by the integrator, 2026-10-18: the `taylor_zhang` docstring and `source` now cite the JCAM 2015 paper for the formula and TNNLS 2019 for the step-size range.]** **FYI for the multistep owner (not core).** The docstring of `multistep.taylor_zhang` cites "Zhang, Gong, Yang, Li, IEEE TNNLS 2018". The Taylor-type 1-step-ahead formula itself appears in Y. Zhang, L. Jin, D. Guo, Y. Yin, Y. Chou, *J. Comput. Appl. Math.* 273 (2015) 29–40, doi:10.1016/j.cam.2014.05.027. Please check which source is intended.
6. **`metrics.chattering` (note).** On a time-invariant problem, a symmetric period-2 cycle (e -> -e) has constant ||E||, so it scores 0 even though the error chatters. The `consistent` tests therefore measure chattering on time-varying problems. A sign-alternation index would need E itself to be recorded, which `simulate` does not do today.

## variants

Requested by the variants area (`znnbench/models/variants.py`). These items are non-blocking: each has a working workaround in place.

1. **Time derivative of the Jacobian, exposed per information regime.** `gznn` (the gradient-ZNN hybrid, Stanimirovic et al., Mathematics 2022) needs $(\partial J/\partial t)^\top E$.
   *Workaround (revised after verification):* the model never samples data itself. In the analytic regime it takes the x-derivative of `problem.dEdt` at the current t (central difference; exact for problems affine in x). For `bd1`/`bd2` it uses backward differences of $J$ over the data samples of its own previous causal calls. For `none` it uses 0. The regime is inferred by comparing `ctx.dEdt` with these candidates.
   (The first version used a finite difference of `problem.JT` over `problem.data(t ± dt)`. That leaked future data, ignored the regime, and amplified the independent jitter or measurement noise of data wrappers by $1/dt$.)
   *Request:* one of the following.
   - (a) An optional `Problem.dJdt_T(x, t, v)` method returning $(\partial J/\partial t)^\top v$.
   - (b) `simulate` passing a callable `ctx.extra['dEdt_at'](x')` that evaluates $\partial_t E$ at an arbitrary state $x'$ under the current information regime. For problems affine in $x$, $\dot J v = \partial_tE(x+v)-\partial_tE(x)$, so `bd1`/`bd2`/`none` would also be honoured for $\dot J$.
2. **Expose the information regime to models:** `ctx.extra['info'] = info`. Models that build extra time-derivative information, as `gznn` does, could then use it directly instead of inferring the regime from `ctx.dEdt`. `gznn` already reads `ctx.extra.get('info')` when it is present.
3. **(Optional) `Problem.solve_JT(x, d, r)`, i.e. $J^{-\top}r$.** `gznn` currently assembles a dense $J$ (guarded at $4\times10^6$ entries) to solve with $J^\top J$. A structured `solve_JT` would let it use $\dot x = J^{-1}J^{-\top}(\cdot)$ at large matrix sizes without dense assembly.

## integrator: triage and recommendations (2026-10-18)

The integrator ran the full suite with `ZNNBENCH_STRICT=1` (900 passed in about 140 s with peak RSS 236 MB, in both file orders) and the smoke matrix (`docs/spec/SMOKE_MATRIX.md`). **No core change is required:** every bug they surfaced was fixed in zoo files. Below is a recommendation for every open item above, ordered by priority. "Accept" means the change is safe and worth making in `core.py`/`simulate.py`/`metrics.py`. All accepted items are additive, so no existing call changes, except R2, which changes random numbers and must land before the sweeps.

**Land before the sweeps (they change numbers or fairness):**

| item | recommendation | why |
|---|---|---|
| runner 2: independent RNG streams | **Accept, high priority.** Use `SeedSequence(seed).spawn(2)` for (x0, noise) and give each trajectory its own noise child, so that results do not depend on batch composition. | `batch_gains` runs are then bitwise equal to separate runs. It changes the realised noise, so do it before any sweep. |
| structured_baselines 5: `Model.x0` hook | **Accept.** `simulate` should call `model.x0(problem, rng, B)` when `x0 is None` and the model defines it. Draw it from the x0 stream of runner 2. | Without it, sweeps record divergence of `znn_explicit_inverse*` that reflects initialisation, not the model. The smoke matrix has to special-case these models. |
| structured_baselines 1: `disc.bind_env(problem, measured, stage_data)` | **Accept, high priority.** Call it right after `disc.reset()`, and have `QuantizedDiscretizer` forward it. | The prediction-correction schemes currently read `simulate`'s local variable names through closure introspection. Renaming a variable in `simulate` would silently change their information pattern. |

**Accept (additive, low risk):**

| item | recommendation |
|---|---|
| runner 1: `steps_run` | Accept: return the executed step count. |
| runner 4: `wall_dynamics` | Accept (optional): time `disc.step` alone. The runner's lean timing mode already covers E7. |
| noise_info 1: state-dependent noise | Accept: `noise(t, rng, shape, E=ctx.E)` when `getattr(noise, "state_dependent", False)`. It unlocks the literature's vanishing-noise class. |
| noise_info 2: `exact_data` for the recorded residual | Accept: record with `getattr(problem, "exact_data", problem.data)(t)`. The wrappers already provide `exact_data`, and the override of `residual` then becomes redundant. |
| noise_info 5: reset hooks | Accept: call `noise.reset()` / `problem.reset()` when they exist. This replaces the time-goes-backwards heuristic in stateful noises and wrappers. |
| problems_robot 1: `x_final` (+ optional `record_states_every`) | Accept `x_final`, which costs B x n_x. Accept the state trace as opt-in only: it costs N_rec x B x n_x memory, so keep it off in sweeps. |
| problems_opt 1: `Problem.solution_error(x, t)` | Accept: opt-in and falls back to the current norm. It gives the KKT problems a primal error. |
| variants 2: `ctx.extra['info'] = info` | Accept, one line. `gznn` already reads it and would no longer infer the regime. |
| variants 1: Jacobian time derivative | Accept option (b), `ctx.extra['dEdt_at'](x')` under the current regime. It honours `bd1`/`bd2`/`none` for every problem without new per-problem code. Option (a) needs a method on every problem. |
| consistent 1: rich implicit request in `DesignFormulaZNN.desired_Edot` | Accept, but this is a **reference-model** change in `models/znn.py`, not core. It changes the meaning of `znn` x `peb_*`/`exact_flow_*`/`semi_implicit_*` cells from a resolvent fallback to the consistent maps. Land it together with an update of `tests/test_consistent.py::test_base_model_fallback_warns_and_matches_resolvent`, and before the sweeps. Until then, use `znn_consistent`. |
| consistent 2 / structured_baselines 2-3: docstrings | Accept (documentation only). Document in `core.py` that `Context.implicit` may be a truthy request object, and document the capability flags `matrix_inverse`, `matrix_inverse_right`, `gradient_field`, `complex_embedding`, `dense_jacobian`, `kkt`, `redundant` and `robot_kinematics`. |

**Defer or decline:**

| item | recommendation |
|---|---|
| runner 3: `t_eps` column names | Decline. It is cosmetic; alias the columns in the analysis scripts. |
| noise_info 3: step index in `data(t, k)` | Decline the Problem API change: it touches all 13 problems. The `_tkey` canonicalisation (1e-11 s) works and is tested. |
| noise_info 4: `data_noise` as a wrapper hook | Defer. `EstimatedDerivativeProblem(meas_std=...)` with `data_noise=0` covers the use case. |
| variants 3: `Problem.solve_JT` | Defer until `gznn` is swept at n >= 64. The dense guard (4e6 entries) suffices for the planned sizes. |
| structured_baselines 4 (`Jv`), 6 (info); problems_opt 2 (info); consistent 4 (activation hooks), 6 (chattering note) | No core change needed. The hooks already live in zoo code; consistent 6 needs an E trace and could later be added with the state trace of problems_robot 1. |
| consistent 3 | Withdrawn (stale; see the inline note). |
| consistent 5, problems_matrix | Done by the integrator (see the inline notes). |

## LANDED in core (2026-10-18, by the project lead)
- runner 1: `simulate` returns `steps_run`; also `wall_dynamics` (time inside `disc.step` only) — runner 4.
- runner 2: independent streams — x0 (`simulate.x0_rng(seed)`), measurement noise, and **per-trajectory additive-noise streams** (`RowRNG`, keyed by `traj_ids`, default 0..B-1). The runner passes `traj_ids = tile(arange(B), G)` for batched gains, so batched == unbatched bit for bit (test `test_core_extras.py::test_noise_is_batch_composition_invariant`). Use `simulate.initial_state(problem, model, seed, B)` to reproduce simulate's own x0 draw.
- structured_baselines 5: optional `Model.x0(problem, rng, B)` hook (used by `znn_explicit_inverse*`: Newton–Schulz safe start ×(1 + 1% noise)).
- structured_baselines 1: `disc.bind_env(problem=, measured=, stage_data=)` called right after `disc.reset()`; implemented by the prediction–correction schemes (closure discovery kept only as a fallback) and forwarded by `QuantizedDiscretizer` (via `__getattr__`) and `robot.StateRecorder`.
- consistent 1: `DesignFormulaZNN.desired_Edot` honours `ErrorMapRequest` (base `znn` = `znn_consistent` on consistent schemes).
- noise_info 1: state-dependent noise (`noise.state_dependent = True` → called with `E=`).
- noise_info 2: residuals recorded with `problem.exact_data(t)` when present.
- noise_info 5: `reset()` called on noise and problem at the start of `simulate`.
- problems_robot 1: `x_init`, `x_final` returned; `metrics.summarize` adds `drift`.
- problems_opt 1: optional `problem.solution_error(x, t)` used for the solution-error trace.
- variants 2: `ctx.extra['info']`.
- consistent 6 / critic: new online statistics over the steady window (last `stats_frac` = 25 %): `chatter_vec` (mean ‖E_{k+1} − E_k‖ / mean ‖E_k‖ — detects symmetric e → −e cycles), `rate_alt` (fraction of sign alternations of Δx components = velocity chattering), `peak_rate` / `peak_rate_ss` (max_i |Δx_i|/τ over the run / steady window). Disable with `online_stats=False` (cost profiling does).
- `_perturb` no longer perturbs time-stamp keys (`t`, `t_*`).
Declined: runner 3 (cosmetic), noise_info 3 (k in data()). Deferred: noise_info 4, variants 1/3, structured_baselines 3/4.
