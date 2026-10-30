# Spec: inverse-free DTZNNs of Ma & Huang (2025) and named ZeaD members

Owned files:
- `znnbench/discretizations/inverse_free_dtznn.py` registers `broyden_dtznn_zhang2012` and `cdtznn_mahuang2025`.
- `znnbench/discretizations/zead_named.py` registers `zead_7i_cang2024` and `zead_5i_mahuang2025`.
- `tests/test_inverse_free_dtznn.py`: 43 tests (38 + 5 added by the verifier), about 10 s, peak RSS 75 MB.
- This file.

`zead.py` was not edited: `zead_named.py` imports `ZeaD`, `coeffs_5i_general`, `coeffs_7i_general` and `in_7i_domain` from it.

Legend: **ASSUMPTION** marks a choice the source does not fix. **ERRATUM** marks a source statement that the reproduction contradicts.

## 1. Source and how it was read

Z. Ma, S. Huang, "Design and analysis of combined discrete-time zeroing neural network for solving time-varying nonlinear equation with robot application," *Front. Neurorobot.* 19 (2025) 1576473, doi:[10.3389/fnbot.2025.1576473](https://doi.org/10.3389/fnbot.2025.1576473). The paper is CC BY.

How it was read:
- Full text: the JATS XML from Europe PMC (PMC12289663), with the MathML linearised by a script. Every equation, Table 1, the Algorithm and the Appendix were read.
- Figures 1–4: the JPEGs on the Frontiers site were viewed (`.../image_m/fnbot-19-1576473-g00{1..4}.jpg`).

Notation: their σ is our τ, f = E, f_t = ∂E/∂t, J = ∂E/∂x, and their "SSCE" is our MSSRE.

Sources cited by Ma & Huang that I could not open (paywalled or proceedings only). Everything below that depends on them is taken second-hand from Ma & Huang:
- Zhang, Peng, Li, Shi, Ling, Proc. ICCECT 2012, 492–495 (the Broyden-aided DTZNN).
- Cang et al., Appl. Soft Comput. 157 (2024) 111511, doi:10.1016/j.asoc.2024.111511 (the 7-instant formula).
- Guo, Xu, Li, Nie, Shao, IEEE TII 14 (2018) (the guo2018 system).

## 2. Equations as printed

**Problem** (Eq. 33): the `nonlinear_equations` variant `guo2018` (see `problems_opt.md`). The paper uses $x_0=0.2\cdot\mathbf 1\in\mathbb R^4$ for Table 1.

**Euler DTZNN** (Eq. 5): $x_{k+1}=x_k-J^{-1}(x_k,t_k)\,(\sigma f_t+hf)$ with $h=\gamma\sigma$. In ZNN-Bench this is `znn` + `euler`.

**Broyden-aided DTZNN** (Eqs. 6–7). The update of $M$ is the inverse (Sherman–Morrison) form of Broyden's "good" update (Broyden, Math. Comp. 19 (1965), doi:10.1090/S0025-5718-1965-0198670-6):

$$
M_k=M_{k-1}+\frac{(u_{k-1}-M_{k-1}v_{k-1})\,u_{k-1}^TM_{k-1}}{u_{k-1}^TM_{k-1}v_{k-1}},\quad
u_{k-1}=x_k-x_{k-1},\quad v_{k-1}=f(x_k,t_k)-f(x_{k-1},t_k),
$$

$$
x_{k+1}=x_k-M_k\big(\sigma f_t(x_k,t_k)+hf(x_k,t_k)\big).
$$

**Taylor DTZNN-I/II/III** (Eqs. 8–13). The paper writes each model as

$$x_{k+1}=\textstyle\sum_j a_jx_{k-j}-J^{-1}\big(\tfrac1{c_0}\sigma f_t+h f\big),\qquad h=\gamma\sigma/c_0.$$

The models use these look-ahead formulas:

| model | formula, $\sigma\dot x_k\approx$ | ZNN-Bench key | $c_0$ | $h_{\rm paper}$ |
|---|---|---|---|---|
| Taylor DTZNN-I, Eqs. (8), (11) | $(8,1,-6,-5,2)/18$ | `zead_5i` | 4/9 | $9\gamma\sigma/4$ |
| Taylor DTZNN-II, Eqs. (9), (12) | $(24,-5,-12,-6,-4,3)/48$ | `zead_6i_guo2017` | 1/2 | $2\gamma\sigma$ |
| Taylor DTZNN-III, Eqs. (10), (13) | $(110,42,-89,-102,12,44,-17)/276$ | `zead_7i_cang2024` (new) | 55/138 | $138\gamma\sigma/55$ |
| new Taylor formula, Eqs. (14), (20), (23) | $(36,-1,-18,-27,10)/78$ | `zead_5i_mahuang2025` (new) | 6/13 | $13\gamma\sigma/6$ |

The two new keys are members of existing families. The identities below are checked in exact rational arithmetic, both at import time and in the tests:
- `zead_7i_cang2024` is `zead_7i_general(p1=-89/110, p2=-51/55)`, the Yang–Zhang–Hu–Qiu TNNLS 2020 family, eq. (8). This point lies inside the effective domain (16).
- `zead_5i_mahuang2025` is `zead_5i_general(a1=5/39)`, the Zhang–He–Hu–Guo–Li–Shi JCAM 2019 family. $a_1=5/39$ lies in $(1/12,1/6)$.

**h mapping.** ZNN-Bench uses tau-form coefficients and $h_{\rm ours}=\gamma\tau$. The update $x_{k+1}=(\tau\dot x_k-\sum_{i\ge1}c_ix_{k+1-i})/c_0$ with $\dot x=-J^{-1}(f_t+\gamma f)$ is exactly the paper's form with $h_{\rm paper}=\gamma\tau/c_0$, so

$$h_{\rm ours}=c_0\,h_{\rm paper}.$$

The factors are stored in `TABLE1_H_FACTOR` (and `zead_named.PAPER_H_FACTOR`). `gamma_from_paper_h(key, h_paper, tau)` returns the ZNN-Bench gain. Euler and Broyden have $c_0=1$.

**CDTZNN** (Eq. 25, start-up Eq. 26), with $h=13\gamma\sigma/6$:

$$
\begin{aligned}
x_{k+1}&=\tfrac1{36}x_k+\tfrac12x_{k-1}+\tfrac34x_{k-2}-\tfrac5{18}x_{k-3}-M_k\big(\tfrac{13}6\sigma f_t(x_k,t_k)+hf(x_k,t_k)\big),\\
M_{k+1}&=\tfrac1{36}M_k+\tfrac12M_{k-1}+\tfrac34M_{k-2}-\tfrac5{18}M_{k-3}-\tfrac{13}6\sigma M_k\dot J(\dot x_k,x_k,t_k)M_k-hM_k\big(J(x_k,t_k)M_k-I\big).
\end{aligned}
$$

For $k=0,1,2$ (Eq. 26) the start-up is:
- $x_{k+1}=x_k-M_k(\sigma f_t+hf)$,
- $M_{k+1}=M_k-\sigma M_k\dot JM_k-hM_k(JM_k-I)$.

The paper prints "$\dot x_k=f(x_k,t_k)+f_t(x_k,t_k)$" for the velocity inside $\dot J$.

The M-recursion is the formula of Eq. (21) applied to the inversion ZNN of Guo, Nie, Yan, IEEE TSMC-S 47(8) 2017 (Eq. 36), with $A\to J$ and $\mu=\gamma$ (Eqs. 37–38):

$$\dot M=-M\dot AM-\mu M(AM-I).$$

For linear activation this is the Getz–Marsden dynamic inverter.

## 3. Implementation in ZNN-Bench

### Generic substitution J⁻¹ → M_k

Every ZNN-Bench model writes its velocity as $\dot x=J^{-1}w$ through `ctx.solve`, which calls `problem.solve_J`. `simulate` adds design-formula noise as $J^{-1}\nu$, also through `problem.solve_J`.

During the single causal call $f(t_k,x_k,\dots)$ of each step, both schemes shadow `problem.solve_J` with $r\mapsto M_kr$:
- The shadow is an instance attribute, set by `_SolveSubstitution` and restored in a `finally` block.
- Calls whose batch shape does not match fall back to the true solve and are counted in `n_true_solves`.

The returned velocity is $\dot x^M_k=M_k(w_k+\nu_k)$, and the update is

$$x_{k+1}=\textstyle\sum_ja_jx_{k-j}+\frac{\tau}{c_0}\dot x^M_k .$$

For the linear `znn` model ($w=-\gamma E-\partial_tE$) this is exactly Eq. (25), first line, and Eq. (7). Other activations, gain schedules, noise, and the other solve-based models (`ieznn`, `vpznn`, …) receive the same substitution.

### CDTZNN M-recursion (tau-form)

$$M_{k+1}=\textstyle\sum_ja_jM_{k-j}-\frac{\tau}{c_0}\big(M_k\dot J_kM_k+\mu M_k(J_kM_k-I)\big),\qquad a=(\tfrac1{36},\tfrac12,\tfrac34,-\tfrac5{18}),\ c_0=\tfrac6{13}.$$

The default is μ = `"gamma"`: the effective design gain $g=\gamma g(t)$ of the paired model, read from its context. With the constant-gain `znn`, $\tau\mu/c_0=h_{\rm paper}$, as in the paper.

### Information pattern (per step k; causal, data at t ≤ t_k only)

**Both schemes:**
- They take the measured sample $d_k$ from `bind_env(problem, measured=...)`, which `simulate` calls after `reset()`. This is the very sample the model sees, including `data_noise`.
- They call the vector field exactly once, at $(t_k,x_k)$ with the current k.
- They never call `problem.data` themselves.
- They never solve a linear system. The exception is `M0="inverse"`, which makes one solve at k = 0.

**`broyden_dtznn_zhang2012`:**
- Evaluates E at two states with one sample:
  - $E(x_k,d_k)$ and $E(x_{k-1},d_k)$ with `secant="current"` (Eq. 6 as printed; default);
  - $E(x_k,d_{k-1})$ and $E(x_{k-1},d_{k-1})$ with `secant="previous"`.
- Uses no Jacobian and no $\partial_tE$ beyond what the model uses.
- Costs $O(n^2)$ per step plus one extra E evaluation.

**`cdtznn_mahuang2025`:**
- Passes a falsy request object as `implicit`. Models built on `DesignFormulaZNN` (`znn`, `znn_consistent`, `vpznn`, `vpznn_exp`) call its `desired_Edot(act, ctx, g)` hook (the protocol of `consistent.py`). The hook records:
  - $E_k$,
  - the regime-consistent $\partial_tE_k$ (`simulate`'s `info`),
  - the regime name,
  - $g$.

  It returns the model's own explicit rate $-g\Phi(E)$, so the model's dynamics are unchanged.
- Also evaluates, with the same sample $d_k$:
  - $J(x_k,d_k)$: from `problem.jacobian` when present, otherwise assembled exactly from `problem.JT` in one batched call with unit vectors.
  - $D_xJ(x_k,d_k)[\dot x_k]$: central difference of $J(\cdot,d_k)$ at $x_k\pm\varepsilon\dot x_k$, with $\varepsilon=6\cdot10^{-6}(1+\|x\|)/\|\dot x\|$. This is exact when J is affine in x.
  - $(\partial_tJ)M_k$, consistent with the information regime:
    - `analytic`: central differences of `problem.dEdt(·, t_k)` at $x_k$ along the columns of $M_k$, all in 2 batched calls. No data at other times is used. This is exact when $\partial_tE$ is quadratic in x, which holds for every problem in the zoo.
    - `bd1`/`bd2`: backward differences of $J(x_k,d_{k-j})$ over the stored measured samples, with the same fallbacks as `simulate`.
    - `none`: 0.
- Models that do not expose the design formula must be given `mu=<float>` and `info=<regime>`; otherwise a `TypeError` is raised.

### ASSUMPTIONS

1. **Velocity inside J̇** (CDTZNN, `xdot`).
   - Taken literally, the printed $\dot x_k=f+f_t$ gives an SSCE of 0.18 instead of 4.874e-7 at (0.01, 0.10), a factor of about 4·10⁵.
   - The default `"ctznn_unit_gain"`, $\dot x_k=-M_k(f_t+f)$, is the CTZNN (3) with γ = 1 and $J^-\to M_k$. I read the print as having lost "$-M_k(\cdot)$". This reading reproduces all six CDTZNN entries of Table 1 to 4 significant digits.
   - `"model"` uses $\dot x^M_k$. It is generic and lands within 0.9 % at τ = 0.01 and 0.04 % at τ = 0.001.
   - `"printed"` keeps the literal print for the record.
2. **M₀** (not stated in the paper).
   - Figs. 1c and 2c start at $\|JM_0-I\|_F\approx 9$–10, which is consistent with $M_0=I$ for both methods.
   - Broyden default `"identity"`: three Table 1 entries are reproduced to 4 digits only with it, and the Newton–Schulz start diverges on guo2018.
   - CDTZNN default `"newton_schulz"` ($J_0^T/(\|J_0\|_1\|J_0\|_\infty)$, the standard inverse-free safe start, also used for `znn_explicit_inverse`): Table 1 is identical with either start. From the identity, CDTZNN diverges on 5 and stalls (MSSRE > 0.2) on 3 of the 11 square zoo problems (§6); `tvqp_ineq` diverges from both starts. `M0="identity"` reproduces the paper's setting exactly.
3. **J̇ by finite differences.** The paper evaluates J̇ in closed form. The Problem API has no second derivatives, so J̇ is computed by finite differences. Against an independent loop that uses the closed-form J̇ of Eq. (33), the trajectories agree to 1e-15 in the final state and to < 1e-5 relative in the residual trace.
4. **Start-up.** Both schemes use forward Euler of the model velocity and of the inversion dynamics: $\tau\gamma$ in the f-term and $\tau\mu$ in the M-residual term. This is the ZNN-Bench `LookAhead` convention. Eq. (26) uses $h_{\rm paper}=\tau\gamma/c_0$ there instead. Only the transient changes: the steady state agrees to 4e-5 relative (tested).
5. **Secant time** (Broyden). The default is the printed $t_k$. `secant="previous"` uses $t_{k-1}$ (see §5).
6. **Broyden safeguard.** The update is skipped for a trajectory when $|u^TMv|\le10^{-12}\|u\|\|Mv\|$ or the denominator is not finite. It never triggers on the Table 1 runs, where the minimum $|\cos(u,Mv)|$ is 0.2.
7. **μ = g = γ g(t)** (the effective gain) for gain schedules. The paper uses a constant γ.
8. **Horizon and window** (inherited from `problems_opt`): T = 10 s and SSCE = max ‖f‖ on the last 25 %. Figures 1–4 run for 2000 and 2·10⁴ steps (T = 20 s), but T = 10 reproduces Table 1 exactly. T matters only for Broyden: with T = 20 the three matching entries move to ratios 0.98.
9. **Scope.** Square J only. A non-square J raises `ValueError`, so the robot problems give "error" cells in the smoke matrix. The state is dense $n_x\times n_x$ per trajectory, limited to `max_dim` = 400 by default.
10. **Noise** enters as $M_k\nu$, which is consistent with $J\dot x=w+\nu$ and $J^{-1}\to M_k$.

## 4. Table 1 reproduction

Setup:
- `simulate` with `znn` (linear), `x0 = 0.2*ones(4)`, T = 10 s.
- One batch of 3 trajectories per (method, τ), with γ per row from `gamma_from_paper_h`.
- Each cell below is ours / paper.

| model (key) | τ | h = 0.10 | h = 0.15 | h = 0.20 |
|---|---|---|---|---|
| Euler (`euler`) | 0.01 | 1.3996e-3 (1.0004) | 9.388e-4 (1.0003) | 7.055e-4 (0.9999) |
| | 0.001 | 1.4139e-5 (0.9999) | 9.427e-6 (1.0005) | 7.070e-6 (1.0000) |
| Taylor DTZNN-I (`zead_5i`) | 0.01 | 4.5543e-7 (1.0001) | 3.1735e-7 (1.0002) | 2.4200e-7 (1.0000) |
| | 0.001 | 4.9461e-11 (1.0002) | 3.2992e-11 (1.0001) | 2.4747e-11 (1.0003) |
| Taylor DTZNN-II (`zead_6i_guo2017`) | 0.01 | 6.6119e-7 (1.0000) | 4.5692e-7 (1.0000) | 3.4728e-7 (1.0002) |
| | 0.001 | 7.0660e-11 (1.0010) | 4.7126e-11 (1.0001) | 3.5350e-11 (1.0000) |
| Taylor DTZNN-III (`zead_7i_cang2024`) | 0.01 | 1.2602e-8 (1.0002) | 8.9135e-9 (1.0001) | 6.8372e-9 (1.0000 vs **6.837e-9**; printed 6.837e-8) |
| | 0.001 | 1.3950e-13 (0.964) | 9.618e-14 (0.985) | 7.290e-14 (0.957) |
| CDTZNN (`cdtznn_mahuang2025`) | 0.01 | 4.8742e-7 (1.0000) | 3.3624e-7 (1.0001) | 2.5535e-7 (0.9998) |
| | 0.001 | 5.1841e-11 (1.0000) | 3.4572e-11 (1.0004) | 2.5931e-11 (1.0004) |
| Broyden, printed secant (`broyden_dtznn_zhang2012`) | 0.01 | 3.024e-3 (0.826) | 2.184e-3 (0.911) | 3.014e-3 (1.429) |
| | 0.001 | 4.637e-5 (1.005) | 2.275e-5 (0.742) | 5.486e-5 (1.874) |
| Broyden, `secant="previous"` | 0.01 | 3.6612e-3 (1.0003) | 2.3982e-3 (1.0005) | 3.302e-3 (1.566) |
| | 0.001 | 4.6164e-5 (1.0001) | 2.742e-5 (0.894) | 5.623e-5 (1.921) |

Notes on the table:
- Taylor DTZNN-III at τ = 0.001 is at the round-off floor (~1e-13, with |x₄| up to 8). That is why it only agrees to 2–4 %, and the test tolerance there is 6 %.
- With exact $J^{-1}$, `zead_5i_mahuang2025` (Eq. 23, not in Table 1) gives 4.798e-7, 3.334e-7, 2.539e-7 and 5.181e-11, 3.456e-11, 2.592e-11. Replacing $J^{-1}$ by $M_k$ costs at most 1.6 % at τ = 0.01 and nothing at τ = 0.001.

## 5. Errata and findings about the source

| # | statement in the paper | finding |
|---|---|---|
| E1 | Table 1, Taylor DTZNN-III, (0.01, 0.20): 6.837e-8 | **ERRATUM**: 6.837e-9 is reproduced to 4 digits. The printed value also breaks the row's decrease in h. |
| E2 | "$\dot x_k=f(x_k,t_k)+f_t(x_k,t_k)$" (below Eq. 24) | **ERRATUM** (most likely a dropped $-M_k(\cdot)$): the literal formula gives an SSCE about 4·10⁵ times the table value. $-M_k(f_t+f)$ reproduces all six entries to 4 digits. |
| E3 | Proof of Theorem 2: roots of $\tilde P_1$ are $-0.5211\pm0.4694i$, $0.2372\pm0.8065i$ | **ERRATUM**: a consistent formula has $\tilde P_1(1)=0$, and $\|\tilde P_1(\vartheta_{\rm printed})\|>0.3$. The true roots are $1$, $0.3159$, $-0.6441\pm0.6814i$ (moduli 1, 0.316, 0.938). The zero-stability conclusion still holds (strong root condition). Linear-ZNN interval of the x-recursion: $h_{\rm ours}<0.1725$, i.e. $h_{\rm paper}<0.374$. |
| E4 | Text on Fig. 1c: Broyden's $\|JM_k-I\|_F$ "at steady state is of the order $10^{-3}$" | **ERRATUM**: the figure itself shows 3–9, and ours is 1.5–2.6 for every secant/M₀ variant. The Broyden matrix never approximates $J^{-1}$; it is exact only along the secant direction $v\approx Ju\approx-\tau f_t$, which is exactly the feed-forward direction. This explains why the residual is still about O(τ²) on guo2018. CDTZNN: ~4e-7 and 4e-11, matching the text of Figs. 2c and 4c. |
| E5 | Broyden rows of Table 1 | Reproducible only in part. Three entries match to ≤ 0.05 % with the secant evaluated at $t_{k-1}$ and $M_0=I$. The other three (h = 0.2 at both τ, and (0.001, 0.15)) differ by 1.57, 1.92 and 0.89. No SSCE window, nearby h or other variant tried (M₀ ∈ {I, J⁻¹, Newton–Schulz, J}; secant at $t_k$, $t_{k-1}$ or time-corrected; update order) reproduces all six. The results are insensitive to 1e-13 perturbations of x₀. **Reason:** Broyden keeps a memory of its start. Its SSCE varies about 3× over random x₀ (8 starts, T = 20: 1.8e-3 to 6.3e-3), and the curves of Fig. 1b span about 2×. CDTZNN's SSCE is x₀-independent to 4 digits. |
| E6 | Lemma 2: Broyden SSCE is O(σ²) | On guo2018 the measured slope is 2.0 at h = 0.1 and 1.65 at h = 0.2 (τ from 0.02 to 0.001). On `linear_system` (n = 4, κ = 10) and the matrix problems the Broyden-aided DTZNN does not track: the residual floor is 1e-2 to 1e2, chaotic: a 1e-14 perturbation of x₀ changes it by 7 % after 400 steps and by 4.5× after 1200 steps, even from $M_0=J^{-1}(x_0)$. The ZNN-Bench implementation equals an independent loop until the chaos amplifies round-off. `expected_order = 2` is therefore the source's claim, valid on guo2018 only. |
| E7 | Remark 2: CDTZNN "can reduce the amount of calculation" | It removes the inversion but not the cost. Per step it needs 4 dense n×n products ($JM$, $M\cdot JM$, $\dot JM$, $M\cdot\dot JM$, about 8n³ flops) plus the evaluation of J̇, against 2n³/3 for one LU solve with one right-hand side. Measured in ZNN-Bench (`wall_dynamics`, B = 1): 79 vs 27 µs/step on guo2018, and 215 vs 73 µs/step on `matrix_inversion` n = 8 (n_x = 64), against `zead_5i_mahuang2025`. These timings depend on the implementation. Broyden is O(n²) per step. |
| E8 | Section 5.2, item 6: performance "further improved with the appropriate increase in h" | Consistent within the linear-ZNN interval: Table 1's $h_{\rm paper}\le0.2$ maps to $h_{\rm ours}\le0.0923<0.1725$ for CDTZNN and $\le0.0797<0.116$ for Taylor DTZNN-III. |

## 6. Generic behaviour (beyond the paper)

Setup: τ = 1e-2, T = 6 s, B = 2, default x₀. $h_{\rm ours}=0.08$ for the 5-instant formulas and 0.3 for Euler-type. Values are MSSRE.

| problem (n_x) | CDTZNN, M₀ = I | CDTZNN, M₀ = NS (default) | `zead_5i_mahuang2025` (exact J⁻¹) | Broyden, M₀ = I (default) | Euler |
|---|---|---|---|---|---|
| complex_inversion (32) | 3.4e-6 | 8.5e-8 | 1.0e-7 | 1.1e-2 | 1.9e-4 |
| linear_system (4) | 6.2e-1 | 4.0e-5 | 1.7e-5 | 4.4e-1 | 3.8e-3 |
| lyapunov (16) | diverged | 1.1e-5 | 1.2e-5 | 8.1e-2 | 4.6e-3 |
| matrix_inversion (16) | 4.8e-1 | 2.7e-5 | 1.9e-5 | 5.9e+1 | 5.4e-3 |
| matrix_inversion_right (16) | 2.6e-1 | 1.4e-5 | 9.1e-6 | 7.7e-1 | 2.4e-3 |
| nonlinear_equations (4) | 2.6e-7 | 2.4e-7 | 2.4e-7 | 1.8e-3 | 4.2e-4 |
| pseudoinverse (24) | diverged | 9.1e-2 | 7.5e-2 | 2.3e-1 | 1.8e-2 |
| sylvester (16) | diverged | 2.4e-5 | 1.9e-5 | 3.3e-1 | 5.5e-3 |
| tv_unconstrained_opt (2) | 3.0e-6 | 3.0e-6 | 3.0e-6 | 2.0e-2 | 2.0e-3 |
| tvqp_eq (3) | diverged | 2.9e-4 | 2.2e-4 | 3.6e-1 | 1.2e-2 |
| tvqp_ineq (7) | diverged | diverged | 9.5e-1 | 5.6e+0 | 7.7e-2 |

- With the Newton–Schulz start, CDTZNN tracks the exact-inverse formula on every square problem except `tvqp_ineq`, where the exact-inverse formula does not converge either at this h.
- Empirical order on `linear_system` at fixed h = 0.0465: 3.99 for τ from 1e-2 to 1.25e-3 (expected 4).
- Under `bd1`, `bd2` and `none`, CDTZNN tracks `zead_5i_mahuang2025` under the same regime (ratios 1.00–1.08 on guo2018) and never calls `dEdt`.
- Divergences with `vpznn`, `gnn` and `hntznn` at γ = 10 are inherited from the step-size limit of the 5-instant formula: `zead_5i_mahuang2025` with exact $J^{-1}$ diverges in the same cells.

## 7. Tests (`tests/test_inverse_free_dtznn.py`)

1. **Transcription.**
   - Exact Fractions of Eqs. (8), (9), (10), (14).
   - Family membership and the 7i domain.
   - The printed update forms of Eqs. (11), (12), (13), (23): the $a_j$ and $1/c_0$, compared with independent re-typings.
   - `TABLE1_H_FACTOR` equals $c_0$.
   - The CDTZNN polynomial: the true roots, the printed roots shown not to be roots, the strong root condition, and $h_{\max}=0.1725$.
2. **Table 1** (§4).
   - Euler (added by the verifier), Taylor I/II/III and CDTZNN, all 30 cells, within 0.5 %. Taylor DTZNN-III at τ = 0.001 is within 6 % (round-off).
   - The DTZNN-III typo.
   - CDTZNN with M₀ = I: same values. `xdot="model"`: within 1.5 %. `"printed"`: SSCE > 1e-2.
   - Broyden: the three exact matches within 0.1 %, and regression locks on the other ratios.
   - Broyden O(σ²) mode: the ratio between τ = 0.01 and 0.001 lies in (40, 150).
   - Broyden memory vs CDTZNN forgetting over random x₀.
   - Inverse-error levels of Figs. 1c and 2c.
3. **Independent loops.**
   - CDTZNN against a from-scratch Eq. (25) with closed-form J̇, for both M₀ (x_final to 1e-12).
   - The Eq. (26) start-up gives the same steady state.
   - Broyden against from-scratch Eqs. (6)–(7), both secants (1e-12).
4. **Information pattern.**
   - One causal f call per step at $(t_k, k)$.
   - `problem.solve_J` is never called, and the substitution count is N.
   - `problem.data` is called only by `simulate`.
   - `solve_J` is restored, including after an exception.
   - `M0="inverse"` makes exactly one solve.
   - Regimes: `dEdt` is called only under `analytic` (2 extra calls per step), and the scheme tracks the exact-inverse formula in every regime.
   - `reset()` and batch consistency (3 gains in one batch equal 3 separate runs).
   - Noise goes through $M_k$ (2 substituted solves per step).
   - (verifier) μ = γ g(t_k) with a gain schedule (`vpznn_exp`), and a float `mu` used as given: $M_2$ against a closed-form loop.
5. **Generic behaviour.**
   - Order 4 on `linear_system`.
   - (verifier) Empirical order of `zead_5i_mahuang2025` (4) and `zead_7i_cang2024` (5) on `linear_system` at their own $h_{\rm opt}$ (measured 3.94 and 4.92).
   - Registry metadata.
   - Guards: non-square, `max_dim`, missing `bind_env`, bad options.
   - Other models: `ieznn` needs `mu`/`info` for CDTZNN and works with Broyden; `gnn` warns that it applies no $J^{-1}$.
   - On `tvqp_eq`, the Newton–Schulz start tracks and the identity start fails.

Mutation check: each of these breaks at least one test:
- perturbing $a_3$ by 1e-3,
- changing $c_0$ to 0.47,
- dropping $D_xJ[\dot x]$,
- dropping $\partial_tJ$,
- swapping the two Broyden secant times.

Run: `cd znn-bench && ZNNBENCH_STRICT=1 python -m pytest tests/test_inverse_free_dtznn.py -q`

## 8. For the integrator (files I do not own)

**`docs/ZOO.md`.** `tests/test_integration.py::test_zoo_catalogue_covers_every_registered_key` fails until the rows below are added to the Discretizations table. (The same test also reports `fa_gnn_znn` and `hgnn_znn`, which belong to another area.) Suggested rows:

```
| `zead_5i_mahuang2025` | 5-instant (36, -1, -18, -27, 10)/78 (= zead_5i_general a1 = 5/39); Ma-Huang DTZNN Eq. (23); h_ours = 6/13 h_paper | yes | 4 | Ma & Huang, Front. Neurorobot. 19 2025, Thm 1, Eqs. (14), (20) | 10.3389/fnbot.2025.1576473 | P | [inverse_free_dtznn](spec/inverse_free_dtznn.md) |
| `zead_7i_cang2024` | 7-instant (110, 42, -89, -102, 12, 44, -17)/276 (= zead_7i_general p1 = -89/110, p2 = -51/55); Taylor DTZNN-III; h_ours = 55/138 h_paper | yes | 5 | Cang et al., Appl. Soft Comput. 157 2024 (as printed in Ma & Huang 2025, Eq. (10)) | 10.1016/j.asoc.2024.111511; 10.3389/fnbot.2025.1576473 | S | [inverse_free_dtznn](spec/inverse_free_dtznn.md) |
| `broyden_dtznn_zhang2012` | Euler DTZNN with J^-1 replaced by the Broyden secant matrix (inverse-free, O(n^2)/step); `secant='current'` (printed) or `'previous'` | yes | 2 (claimed; guo2018 only) | Ma & Huang 2025, Eqs. (6)-(7); Zhang, Peng, Li, Shi, Ling, ICCECT 2012 (not accessible) | 10.3389/fnbot.2025.1576473 | S | [inverse_free_dtznn](spec/inverse_free_dtznn.md) |
| `cdtznn_mahuang2025` | combined DTZNN: 5-instant x-recursion with M_k ~ J^-1 from a ZNN inversion recursion (no linear solve); square J only | yes | 4 | Ma & Huang 2025, Eqs. (24)-(26), (36)-(38) | 10.3389/fnbot.2025.1576473 | P | [inverse_free_dtznn](spec/inverse_free_dtznn.md) |
```

The counts line in ZOO.md goes from 38 to 42 discretizations.

**Core and runner requests.** All are optional; workarounds are in place.
1. `simulate` could pass `info=` (and, optionally, `gamma=`) to `bind_env`. CDTZNN would then honour the regime for models that do not expose the design formula, without the user passing `mu`/`info`. `bind_env` already accepts and records an `info` keyword.
2. A discretizer-side compatibility flag such as `disc.requires = {"square_jacobian"}`, honoured by the runner and smoke-matrix skip rules. With it, the robot problems (m ≠ n_x) would be recorded as "skipped" rather than "error" for the two inverse-free schemes.
3. An optional `Problem.Jdot(x, d, t, v)` (J̇ along x_dot = v) would replace the finite-difference J̇. Not needed for accuracy: the finite difference is exact for every problem in the zoo up to round-off.

## 9. Verification (adversarial verifier, 2026-10-18)

**Source re-read.** I downloaded the JATS XML of PMC12289663 from the Europe PMC REST API again, linearised its MathML with my own script, and read every equation, Table 1, the Algorithm and the Appendix. I also viewed Figs. 1 and 2 (Frontiers `image_m` JPEGs). I could not obtain Cang et al. 2024 (Semantic Scholar and OpenAlex both report it closed, with no repository copy) or Zhang et al., ICCECT 2012 (only a title/abstract page, which returns HTTP 403). No web request carried a personal identifier.

**Transcription, checked symbol by symbol against the XML:**
- Eq. (6): Broyden's "good" update in inverse (Sherman–Morrison) form, $u_{k-1}=x_k-x_{k-1}$, $v_{k-1}=f(x_k,t_k)-f(x_{k-1},t_k)$. Eq. (7): $x_{k+1}=x_k-M_k(\sigma f_t+hf)$ with $h=\gamma\sigma$.
- Eqs. (8)–(14) and (20): the integer numerators and denominators 18, 48, 276 and 78. Eqs. (11)–(13) and (23): the $a_j$, the factor $1/c_0$ (9/4, 2, 138/55, 13/6) and $h_{\rm paper}$ (9γσ/4, 2γσ, 138γσ/55, 13γσ/6).
- Eqs. (24)–(26) and (36)–(38). In Eq. (26) the start-up uses σ (not 13σ/6) in front of $f_t$ and of the $\dot J$ term, and $h$ in front of $f$ and of the M-residual.
- All 30 Table 1 entries, against `MAHUANG2025_TABLE1`. They agree, except the corrected DTZNN-III cell, which is flagged.
- The guo2018 system (Eq. 33) and its stated solution. In my check, the solution satisfies all four equations.
- The velocity print "$\dot x_k=f(x_k,t_k)+f_t(x_k,t_k)$" appears **three times**: below Eq. (24), and in Steps 2 and 3 of the Algorithm. The print is therefore consistent within the paper. Only the reproduction (E2) shows that it cannot be what produced Table 1.

**Exact arithmetic (my own script):**
- Taylor moments: Eq. (14)/(20) has $\mu_0=0$, $\mu_1=1$, $\mu_2=\mu_3=0$, $\mu_4=11/52$, so $p=3$ and `expected_order` = 4. Eq. (10) has $\mu_2=\mu_3=\mu_4=0$ and $\mu_5=359/1380$, so $p=4$ and `expected_order` = 5. Eqs. (8) and (9) both have $p=3$.
- $\tilde P_1(1)=0$. The printed roots give $|\tilde P_1|=0.43$ and $1.38$, so they are not roots. The true roots are $1$, $0.31594$, $-0.64408\pm0.68144i$ (modulus 0.9377). E3 is confirmed.
- Linear-ZNN intervals in ZNN-Bench h: 0.17250 for Eq. (20), 0.11594 for Eq. (10), 0.2808 for Eq. (8) and 0.3333 for Eq. (9). In $h_{\rm paper}$ these are 0.3737, 0.2909, 0.6317 and 0.6667, so every Table 1 step size is inside its interval. E8 is confirmed.

**Table 1, from an independent from-scratch loop** (pure numpy, closed-form $f$, $f_t$, $J$, $\dot J$, no `znnbench` import; the paper's printed update forms; T = 10 s; maximum over the last 25 %). Ratios ours/paper:

| model | τ = 0.01 (h = 0.10, 0.15, 0.20) | τ = 0.001 (h = 0.10, 0.15, 0.20) |
|---|---|---|
| Euler (5) | 1.0004, 1.0003, 0.9999 | 0.9999, 1.0005, 1.0000 |
| Taylor DTZNN-I (11) | 1.0001, 1.0002, 1.0000 | 1.0001, 0.9999, 1.0002 |
| Taylor DTZNN-II (12) | 1.0000, 1.0000, 1.0002 | 1.0010, 1.0001, 1.0000 |
| Taylor DTZNN-III (13) | 1.0002, 1.0001, **0.1000** vs the printed value (E1) | 0.960, 0.979, 0.949 (round-off floor) |
| CDTZNN (25), literal Eq. (26) start-up, $M_0=I$, $\dot x=-M(f_t+f)$ | 1.0000, 1.0001, 0.9998 | 1.0001, 1.0003, 1.0004 |
| Broyden (7), printed secant | 0.826, 0.911, 1.429 | 1.005, 0.742, 1.874 |
| Broyden, secant at $t_{k-1}$ | 1.000, 1.000, 1.566 | 1.000, 0.894, 1.921 |

These numbers agree with §4 and with the implementation to round-off. The DTZNN-III τ = 0.001 cells differ between implementations by about 1 %, at the 1e-13 floor.

**Readings of the velocity inside $\dot J$** (CDTZNN, from the scratch loop):

| reading | $M_0$ | Table 1 ratios | reproduces? |
|---|---|---|---|
| $-M_k(f_t+f)$ (default) | either | 1.0000–1.0004 | all six cells |
| $-J^{-1}(f_t+f)$ (exact inverse, unit gain) | | 1.0000–1.0004 | all six cells |
| model velocity $-M_k(f_t+\gamma f)$ (`"model"`) | Newton–Schulz | 0.991–0.997 at τ = 0.01 | not to 4 digits |
| model velocity | $M_0=I$ | | diverges (from the identity the fed-back velocity is large and $\dot J$ blows up) |
| backward difference $(x_k-x_{k-1})/\sigma$ | | ×340–48000 | no |
| $-(f+f_t)$ | | ×9000–18500 | no |
| printed $f+f_t$ | | ×3.8·10⁵ | no |

Two caveats about the default:
- It hard-codes a gain of 1 s⁻¹, so it is not invariant to rescaling time or E.
- It matters only when J depends on x. For the problems that are affine in x (`linear_system`, the matrix problems, `tvqp_eq`), $D_xJ=0$, and all readings give identical results. I checked this on six problems at τ = 1e-2 and 1e-3.

**Broyden findings, re-checked:**
- E4: Fig. 1c starts at about 9 and stays at 3–4.5 over k = 200–2000. The text's "order 10⁻³" is wrong.
- Our steady-state $\|JM-I\|_F$ is 1.50–2.57 for both secants and both τ.
- CDTZNN: $\|JM-I\|_F$ is at most 4.0e-7 (τ = 0.01) and 4.1e-11 (τ = 0.001), matching Figs. 2c and 4c.
- E5: SSCE is unchanged to 6 digits under perturbations of x₀ of ±1e-13 and 1e-10.
- No variant reproduces all six cells. I tried:
  - the time-mixed secant $f(x_k,t_k)-f(x_{k-1},t_{k-1})$ (diverges),
  - the same with a $-\sigma f_t$ correction,
  - the "bad" Broyden update $(u-Mv)v^T/v^Tv$,
  - $M_0=J_0^{-1}$,
  - T = 20,
  - windows of 5 %, 10 % and 50 %.
- E6: the slope of a fit over τ ∈ {0.02, 0.01, 0.005, 0.002, 0.001} is 1.98 at h = 0.1 and 1.61 at h = 0.2.
- On `linear_system` (n = 4, κ = 10), Broyden reaches MSSRE 3.5e-2 to 3.8e-1 for every τ, so it does not converge.
- An independent from-scratch Broyden loop on the same `linear_system` data agrees with the implementation to 2e-10 relative up to step 100. From step 200 on the two differ by 2–60 %, because the dynamics are chaotic: a 1e-14 shift of x₀ changes the residual at step 600 by 5×. This is a property of the method, not a bug.

**Spot checks of §6:** the rows for `matrix_inversion`, `lyapunov`, `tvqp_eq` and `nonlinear_equations` reproduce to the printed 2 digits. That includes the divergences of CDTZNN from $M_0=I$.

**API and information-pattern review:**
- `bind_env(problem, measured, stage_data)` is used as the core contract says. `measured()` returns `past[0]`, the very sample `simulate` gives the model.
- Each step makes exactly one `f` call at $(t_k, k)$. z is advanced by forward Euler.
- The `implicit` probe uses the `desired_Edot(act, ctx, g)` protocol that `DesignFormulaZNN` and `ConsistentZNN` honour. It returns their own explicit rate, so the model is unchanged.
- The solve shadow is an instance attribute, so it also works through `ProblemWrapper` (which delegates through `__getattr__`). The design-formula noise path (`problem.solve_J(xx, d, nu)` in `simulate`) is substituted too.
- In the analytic regime, $\partial_tJ$ uses only `problem.dEdt(·, t_k)`. `EstimatedDerivativeProblem.dEdt` is read-only, so the two extra calls per step do not disturb its sample history.
- Smoke runs, with `nonlinear_equations` and `znn`, gave the same SSCE as `zead_5i_mahuang2025` under the same perturbation: `JitteredProblem`, `DelayedDataProblem`, `EstimatedDerivativeProblem`, `CountingProblem`, `QuantizedDiscretizer` (forwards `bind_env`), `data_noise`, Gaussian noise, and `info` ∈ {`bd1`, `bd2`, `none`}.
- Over the 11 square problems × 19 models, Broyden produced no exceptions beyond the models' own capability errors.
- CDTZNN raises the documented `TypeError` for the 13 models that do not expose the design formula (123 cells). It runs with `znn`, `znn_consistent`, `vpznn`, `vpznn_exp` and `znn_explicit_inverse*`. The last applies no $J^{-1}$, and the scheme warns.

**Test meaningfulness.** I re-ran six mutations on a scratch copy of the package. Five were caught:
- Euler start-up shortened to 2 steps,
- Broyden update sign/scale changed,
- M-step factor τ instead of τ/c₀,
- the sign of the bd1 $\partial_tJ$ flipped,
- the finite-difference step ×10⁴.

The sixth survived: **μ = base γ instead of γ g(t)** (assumption 7 was untested). Tests added:
- `test_mu_is_effective_gain_of_gain_schedule`: $M_2$ under `vpznn_exp` against a closed-form loop, and a float `mu`. It kills that mutation.
- `test_named_formulas_empirical_order`: the DESIGN.md order check for `zead_5i_mahuang2025` and `zead_7i_cang2024` was missing. Measured slopes 3.94 and 4.92.
- The Euler row of Table 1 in `test_table1_reproduced`.

**Registrations kept.** All four keys stay registered. Nothing is implemented without a source I could read:
- `cdtznn_mahuang2025` and `zead_5i_mahuang2025` are new in Ma & Huang, which is open access and was read in full (check P).
- `zead_7i_cang2024`: the primary paper is closed. The exact formula is printed in Ma & Huang, Eq. (10), and the full Taylor DTZNN-III model in Eq. (13). Its order conditions hold exactly; its three Table 1 cells at τ = 0.01 are reproduced to 4 digits (one of them only after correcting the typo E1), and its three cells at τ = 0.001, near the round-off level of about 1e-13, agree within 1.5–4.3 % (ratios 0.957–0.985, Section 4). Check S.
- `broyden_dtznn_zhang2012`: the primary paper is not accessible. The scheme is Ma & Huang Eqs. (6)–(7) as printed, plus the classical Broyden (1965) inverse update. Its Table 1 row is **not** reproduced by the printed equations. I changed the suggested ZOO check code in §8 from P to S. The key name keeps Ma & Huang's attribution, and §5 E5 records the non-reproducibility.

**Not changed (judgement calls, flagged above):**
- The CDTZNN default $M_0$ is Newton–Schulz. Figs. 1c and 2c indicate the paper used $M_0=I$ (initial $\|JM_0-I\|_F\approx 8$–10), which is available as `M0="identity"` and gives identical Table 1 values.
- The Euler start-up instead of Eq. (26). The steady state is identical; I checked both in the scratch loop.
- The Broyden default is the printed secant at $t_k$. The secant at $t_{k-1}$ matches three cells exactly but not the other three.
- `expected_order = 2` for Broyden. This is the source's claim, which E1 reads as `claimed_order`. Measured, it holds on guo2018 only.

**Bugs found in the owned code:** none. I found no defect in the equations, coefficients, h mapping, start-up, information pattern or state handling. The only gap was the untested gain-schedule path, now covered.

**Open items, not owned here:**
- `docs/ZOO.md` still lacks the four rows of §8, so `tests/test_integration.py::test_zoo_catalogue_covers_every_registered_key` fails. That test also lists `fa_gnn_znn` and `hgnn_znn`, which belong to another area.
- The robot problems give "error" cells. The skip flag of §8, item 2, would fix this.

**Stability note from E2 (2026-10-18).** The interval h < 0.1725 above is correct for CDTZNN's x-recursion. On time-varying problems the M-recursion (the ZNN estimate of J⁻¹) limits the scheme earlier: linearised around the solution both recursions share the root locus of `zead_5i_mahuang2025`, but the frozen-time correction of the M-recursion is τ(μ_i + μ_j), twice that of the x-recursion. On `linear_system` the measured limit is about 0.075–0.12 (0.104 at T = 24 s; T = 6 s overstates it by ~30 % because M drifts away slowly), i.e. 1.02–1.12 × the corrected prediction, and about 21 % below its exact-inverse twin. See experiments/E2_NOTES.md.
