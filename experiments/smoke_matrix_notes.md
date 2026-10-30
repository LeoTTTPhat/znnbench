## Interpretation (integrator, 2026-10-18)

The default matrix has **no error and no timeout**. Every skip is a recorded incompatibility:

- `s:req` (144 cells): `znn_explicit_inverse` and `znn_explicit_inverse_right` on the 12 problems without capability `matrix_inverse` / `matrix_inverse_right`.
- `s:imp` (117 cells): the 9 models without implicit (resolvent) support, under `implicit_euler_err`.

Each class of diverged cell was traced to a property of the method at h = gamma*tau = 0.1, T = 1 s, from the problems' random initial states. None is a code fault. The checks are re-runnable with the script's `--problems/--models/--discs/--gamma` options.

1. **`gnn` (explicit step-size limit).** The explicit GNN step is h_GNN = gamma*tau*sigma_max(J)^2. That is 8.3 on `linear_system` and `matrix_inversion` (kappa = 10) and 2.0e3 on `pseudoinverse` (J = A^T A ⊗ I). Both are far above 2, so every explicit scheme diverges, RK4 included. Where h_GNN < 2 (`tvqp_eq` 0.40, `tv_unconstrained_opt` 0.27) GNN runs and shows its O(1/gamma) lag floor. This is the mechanism of `tests/test_core.py::test_gnn_lag_floor`.
2. **`sptc_an_foznn` (quadratic predefined-time term from far initial states).** Before t_c the error dynamics contain gamma*||E||*E/(t_c - t)^2 times (t/t_c)^(alpha-1). The latter is 31.6 at the first sample (alpha = 0.5, t_floor = 1e-3, a flagged ASSUMPTION). The local step is therefore about 31.6*gamma*tau*||E0||, far above 2 for ||E0|| between 2.6 and 506. From x0 = x*(0) + 0.05 on `linear_system` the same model converges: residual 2.0e-6 at T with Euler, 2.5e-3 with RK4. It also survives on `tv_unconstrained_opt`, where ||E0|| = 0.15. The variants verifier reported the same behaviour at tau = 1e-3 (docs/spec/variants.md).
3. **`hntznn` (Li activation, |e|^2 branch).** psi(e) = Lip^r(e) + Lip^(1/r)(e) with r = 0.5 contains |e|^2 sign(e). Its explicit local step gamma*tau*2|e| exceeds the Euler limit 2 for |e| > 10 and the Taylor-Zhang limit 1 for |e| > 5. The default random states of the matrix problems are that far away (`pseudoinverse`: ||E0|| = 506; it diverges even with RK4). From x*(0) + 0.01 on `pseudoinverse`, Euler runs: residual 0.10 at T. This matches the variants verifier's note.
4. **Time-varying gains leave the formula's stability interval** (`vpznn_exp`, `sptc_nt_znn`, `vpznn`).
   - The linear-ZNN intervals in h are Euler (0, 2), Taylor-Zhang (0, 1), `zead_5i` (0, 0.281) and `uhlig_4_6` (0, 0.176) (`zead.lookahead_stability`).
   - `vpznn_exp` has g(t) = 2^t + 2 in [3, 4], so h_eff is in [0.3, 0.4]. It diverges under `zead_5i` and `uhlig_4_6` only, and with gamma = 4 (h_eff <= 0.16) it converges under `uhlig_4_6`.
   - `vpznn` has g(t) = 1 + t, so h_eff >= 0.176 for t >= 0.76.
   - `sptc_nt_znn` has h_eff = gamma*tau/(t_c - t), which grows without bound as t approaches t_c = 1 s = T. Euler is exact where h_eff = 1 (t = 0.9; residual 2.4e-13 at T). `zead_5i` stays bounded up to T = 0.6. The RK4 cell on `nonlinear_equations` fails because its last stage reaches t_c, where the model switches to its post-t_c sign branch.
5. **Auxiliary-state models under the narrow Uhlig interval** (`dieznn`, `ieznn`, `gznn`). They are unstable only under `uhlig_4_6`. On `pseudoinverse`, `nonlinear_equations` and `tvqp_ineq` they cross the divergence threshold within T. Elsewhere `dieznn` still grows to about 1e8 without crossing it. They run under every other scheme. Their fast modes have effective steps of about gamma*tau = 0.1 per pole: DIEZNN's characteristic polynomial (s+1)(s+xi)^2 with xi = gamma has a double pole. Together with the Euler-advanced auxiliary states this sits at the edge of the (0, 0.176) interval.
6. **`davidenko` on `tvqp_ineq` (open-loop continuation).** x_dot = -J^{-1} dE/dt has no error feedback, so E(x(t), t) stays at E(x0, 0) ≠ 0 (the finite `davidenko` cells show ||E(T)|| ≈ ||E(0)||, as designed). On `tvqp_ineq` that level set runs into nearly singular perturbed Fischer-Burmeister Jacobians.

**Finite does not mean stable.** A cell can hold a finite number and still be unstable: the run grew but did not reach the 1e10 threshold within T = 1 s. Such cells are listed under "Finite runs whose residual grew" above, for example `sptc_nt_znn` x `zead_5i` (about 1e8) and `vpznn` x `uhlig_4_6`.

**Robot problems.** The default x0 lies on the path, so ||E(x0)|| = 0. E is a position error of a bounded arm, so a blow-up cannot reach |x| > 1e10 through the error. A finite final residual of 0.2 to 1.2 m (`hntznn`, `sptc_nt_znn`, `dieznn`, `vpznn_exp` under `zead_5i`/`uhlig_4_6`) is the robot counterpart of a diverged cell.

**Choice of the Uhlig formula.** `uhlig_3_4` has h_max = 0.054 < 0.1, so at the smoke-matrix gain even linear ZNN is outside its interval. The representative Uhlig formula is therefore `uhlig_4_6` (4_5a, h_max = 0.176). Run `--discs uhlig_3_4` to see the former.

## Stress variants run by the integrator (not tabulated above)

Cells are counted on the post-fix runs. The pre-fix runs used the same settings; two of them ran on a subset of the models.

| variant | cells | result |
|---|---:|---|
| B = 2, `info=bd2`, Gaussian noise, `data_noise=1e-4`, **all 38 discretizations** | 7410 | before fixes: 1 error (fix 1) |
| B = 2, `info=none`, impulse noise, all 38 discretizations | 7410 | before fixes: 1 error (fix 2) |
| B = 3, `info=bd1`, random-walk noise, default discretizations | 1170 | before fixes: 2 errors (fix 2) |
| B = 2, multi-harmonic noise, AB3 / consistent / PEB / exact-flow / semi-implicit / implicit-TZ / PC schemes | 1950 | no error |
| B = 4, tau = 1e-3, T = 0.2, uniform noise, `info=bd2` | 1170 | no error |
| every activation x {znn, znn_consistent, vpznn, gnn, znn_explicit_inverse} x 7 schemes x 5 problems | 3150 | before fixes: 127 errors, all `strict_predefined` (fix 3) |

After the fixes, all variants have **0 errors** (about 22k cells in total).

Fixes, each with a regression test that fails on the old code:

1. `discretizations/prediction_correction.py`: the `pc_gradient` step-size estimate raised `ValueError (L = nan)` when a trajectory left the domain of E (GNN prediction to x1 <= 0 in `nonlinear_equations`, where ln(x1) is NaN). Non-finite rows are now ignored, and if every row is non-finite the step returns NaN, so `simulate` flags divergence. Test: `tests/test_structured_baselines.py::test_pc_gradient_outside_domain_is_divergence_not_error`.
2. `problems/optimization.py`: an exactly singular dense J (reached by a diverging trajectory) made `np.linalg.solve` raise `LinAlgError`, which aborted the whole batch. The model-facing `solve_J` now falls back to the minimum-norm least-squares step (the `Problem.solve_J` contract, as in `robot.py`). The reference Newton solver still sees the exception. Test: `tests/test_problems_opt.py::test_singular_jacobian_falls_back_to_min_norm_not_error`.
3. `models/znn.py`, `models/variants.py` (hntznn), `discretizations/consistent.py`: a time-varying activation (`strict_predefined`) raised at the first step of every design-formula model. The models now freeze it at the evaluation time with `act.at(ctx.t)`, which is the identity for time-invariant activations. `make_model("znn", activation="strict_predefined")` therefore equals `sptc_nt_znn` before t_c. Test: `tests/test_integration.py::test_design_formula_models_freeze_time_varying_activation`.
