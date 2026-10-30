# ZNN model variants and gain schedules (area: `variants`)

Files: `znnbench/models/variants.py`, `znnbench/models/gains.py`, `tests/test_variants.py`.

## Conventions used by every model

The design formula prescribes the desired error derivative $r$. The state velocity is
$\dot x = J^{-1}\big(r - \partial_t E\big)$, computed with `ctx.solve`. Additive noise $\nu$ enters the design-formula space, $J\dot x = \dots + \nu$; `simulate` adds it. This matches the noise-tolerant ZNN literature, where noise is added to $\dot E$.

- The base gain of the source paper is `ctx.gamma`. Each model states its exact mapping below.
- Auxiliary states, such as integrals of $E$ and internal-model states, live in `z`. They start at 0, which corresponds to $\int_0^0=0$. Causal discretizers advance them with forward Euler.
- None of the models below implements the implicit (resolvent) form, except `vpznn` and `vpznn_exp`, which inherit it from `DesignFormulaZNN`. With `implicit_euler_err`, the other models run their explicit right-hand side, following the precedent of `GNN`. Their `supports_implicit` is `False`.
- `gznn` is the only stateful model: it keeps the data samples of its last three causal calls, needed for the `bd1`/`bd2` regimes of $\dot J$; see (f).
- **ASSUMPTION** flags mark every choice that does not come from the source.

| key | model | source (DOI) | aux `z` size |
|---|---|---|---|
| `vpznn` | power-type varying-parameter ZNN | 10.1109/TCYB.2017.2760883, 10.1109/TCYB.2018.2841970, 10.1109/TMECH.2018.2799724 | 0 |
| `vpznn_exp` | exponential-type VP ZNN | 10.1016/j.neucom.2019.01.058 | 0 |
| `ieznn` | integration-enhanced / noise-tolerant ZNN | 10.1109/TNNLS.2015.2497715, 10.1109/TAC.2016.2566880 | $m$ |
| `dieznn` | double-integral-enhanced ZNN | 10.1049/cit2.12161 | $2m$ |
| `hntznn` | harmonic-noise-tolerant ZNN (internal model) | 10.3389/fnbot.2022.928636 | $2Lm$ |
| `sptc_nt_znn` | strictly predefined-time noise-tolerant ZNN, gain $1/(t_c-t)$ | 10.1109/TIE.2023.3241393 | 0 |
| `sptc_an_foznn` | conformable fractional-order strictly predefined-time anti-noise ZNN | 10.1016/j.neunet.2025.107279 (arXiv:2503.01857) | 0 |
| `gznn` | gradient-ZNN hybrid (ZNN design applied to the GNN gradient $J^\top E$) | 10.3390/math10224292 (time-varying extension of the hybrid GNN-ZNN models in 10.1016/j.neucom.2018.07.058) | 0 |

---

## (a) Gain schedules (`gains.py`) and VP-ZNN (`vpznn`, `vpznn_exp`)

Design formula: $\dot E = -\gamma\, g(t)\,\Phi(E)$, using `DesignFormulaZNN` from `models/znn.py`. Each schedule returns the time-varying design parameter of its source. **Set `gamma = 1` to reproduce a paper.** `ctx.gamma` acts as an extra multiplier, the ZNN-Bench convention $h=\gamma\tau$. Every schedule takes an optional `cap`, $g_{\rm cap}(t)=\min(g(t),\text{cap})$. **The cap is a ZNN-Bench addition and does not appear in any source.**

| class (`key`) | $g(t)$ | source | defaults |
|---|---|---|---|
| `ConstantGain` | $1$ | `models/znn.py` | — |
| `PowerVPGain` (`power_vp`) | $\zeta + t^{p}$, with $\zeta=p$ by default, giving $t^p+p$ | Zhang, Zheng, Weng, Mao, Lu, Xiao, *IEEE TCYB* 48(11):3135–3148, 2018 ($t^\kappa+\kappa$); Zhang & Zheng, *IEEE TCYB* 49(10):3627–3639, 2019 ($t^p+p$); Zhang, Fu, Yan, Jin, Xiao, Sun, Yu, Li, *IEEE/ASME TMECH* 23(2):679–689, 2018 ($\zeta+t^\gamma$) | $p=1$ (ASSUMPTION) |
| `ExpVPGain` (`exp_vp`) | $p^{t}+p$ | Zhang, Zheng, Wang, *Neurocomputing* 338:126–138, 2019 | $p=2$ (ASSUMPTION) |
| `EpsExpGain` (`eps_exp`) | $\varepsilon e^{t}$ | Zhang, Zheng, Li, Deng, Xiao, Huang, *Neurocomputing* 319:74–83, 2018, doi:10.1016/j.neucom.2018.07.005 | $\varepsilon=1$ |
| `ArccotExpGain` (`arccot_exp`) | $\beta\exp(\lambda_1\operatorname{arccot} t+\lambda_2 t)$ | Xiao & He, *IEEE TII* 17(11):7513–7522, 2021, doi:10.1109/TII.2021.3058343 | $\beta=\lambda_1=\lambda_2=1$ |
| `PredefinedTimeGain` (`predefined_time`) | $1/(t_c-t)$ for $t<t_c$; `after` (default 1) for $t\ge t_c$ | SPTC-NT-ZNN, see below | $t_c=1$ |
| `ConformableGain` (`conformable`) | $\max(t,t_{\rm floor})^{\alpha-1}$ | Yang et al. 2025, Eqs. (11)–(12); see (e) | $\alpha=0.5$, $t_{\rm floor}=10^{-3}$ (ASSUMPTION) |

With a linear $\Phi$ and exact $\partial_t E$, the error is $E(t)=E(0)\exp\!\big(-\gamma\int_0^t g\big)$. For the power type, $\int_0^t g = \zeta t + t^{p+1}/(p+1)$. This is the "super-exponential" convergence claimed in the VP-CDNN papers. `PredefinedTimeGain` uses a numerical guard: if $t \ge t_c-10^{-12}\max(1,t_c)$, the post-$t_c$ value is used.

**Unverified (secondary source):** every VP formula comes from the review Y. Wang, C. Hua, A. H. Khan, "Advances in zeroing neural networks: bio-inspired structures, performance enhancements, and applications," *Biomimetics* 10(5):279, 2025, doi:10.3390/biomimetics10050279, read from its MathML on Europe PMC (PMC12108775). Table 2 of the review gives $\zeta+t^\gamma$ (TMECH 2018, its ref. 121), $p^t+p$ (Neurocomputing 2019, ref. 122), $t^p+p$ (TCYB 2019, ref. 123) and $\beta\exp(\lambda_1\operatorname{arccot}t+\lambda_2t)$ (Xiao & He 2021, ref. 126). Its running text gives $\dot E=-(t^\kappa+\kappa)F(E)$ for the VP-CDNN of TCYB 2018 (ref. 52) and $\vartheta(t)=\varepsilon e^{t}$ for the FT-VP-CDNN (ref. 56). I could not open the primary full texts. The abstracts confirm "super-exponential" convergence for the power type. None of the papers' simulation defaults were available.

**Registered models.**
- `vpznn(activation='linear', p=1, zeta=None, cap=None, **act_kw)`
- `vpznn_exp(activation='linear', p=2, cap=None, **act_kw)`

Both support the implicit (resolvent) error step with $h(t)=\gamma\tau g(t)$.

**Tests.**
- `test_gain_schedule_values`: closed-form values, caps and errors.
- `test_varying_gains_grow_as_specified`: VP, exponential and predefined-time gains increase strictly; `integral()` matches quadrature to $10^{-6}$; $g(t_c-10^{-6})=10^6$.
- `test_vpznn_superexponential_error_decay`: RK4 with $\tau=10^{-3}$ and $\gamma=1$ reproduces $E(0)e^{-\int g}$ to a relative error below $10^{-7}$ for three schedules.
- `test_vpznn_implicit_and_cap`: Euler with an uncapped $t^2+2$ gain diverges once $h(t)>2$. A cap, or `implicit_euler_err`, keeps it stable.

## (b) Integration-enhanced / noise-tolerant ZNN — `ieznn`

Sources:
- L. Jin, Y. Zhang, S. Li, "Integration-enhanced Zhang neural network for real-time-varying matrix inversion in the presence of various kinds of noises," *IEEE TNNLS* 27(12):2615–2627, 2016, doi:10.1109/TNNLS.2015.2497715.
- L. Jin, Y. Zhang, S. Li, Y. Zhang, "Noise-tolerant ZNN models for solving time-varying zero-finding problems: a control-theoretic approach," *IEEE TAC* 62(2):992–997, 2017, doi:10.1109/TAC.2016.2566880.

$$\dot E(t) = -\gamma E(t) - \lambda\int_0^t E(s)\,ds + \nu(t),\qquad \gamma,\lambda>0,$$

with $z=\int_0^t E$ and $\dot z=E$. The error dynamics are the LTI system $s^2+\gamma s+\lambda$. A constant $\nu$ is rejected. A linear noise $Qt+M$ leaves $E\to Q/\lambda$.

The formula was cross-checked in two independent secondary transcriptions: Eq. (6) of the DIEZNN paper, read from its full text, and Eq. (5) of Yang, Wang, Huang, *Axioms* 13:374, 2024. I could not open the primary TNNLS and TAC texts.

**ASSUMPTION (ZNN-Bench mapping):** $\gamma$ = `ctx.gamma`. If `lam` is not given, $\lambda=\tfrac14\gamma^2$ (`lam_ratio = 0.25`), which gives a critically damped double pole at $-\gamma/2$. The step-size knob $h=\gamma\tau$ then keeps one meaning across gains. The sources only require $\gamma,\lambda>0$.

**Tests.**
- `test_ieznn_exact_error_dynamics`: RK4 matches $E_0e^{-at}(1-at)$ with $a=\gamma/2$, to $10^{-8}$.
- `test_ieznn_rejects_constant_noise_znn_keeps_floor`: under constant noise $\nu=1$, linear ZNN has MSSRE $=\sqrt m/\gamma$ within 2%. IEZNN has MSSRE below $10^{-3}$.
- `test_ieznn_linear_noise_limit_is_slope_over_lambda`: under $\nu=t$, the limit is $\sqrt m/\lambda$ within 2%.

## (c) Double-integral-enhanced ZNN — `dieznn`

Source: B. Liao, L. Han, X. Cao, S. Li, J. Li, "Double integral-enhanced Zeroing neural network with linear noise rejection for time-varying matrix inverse," *CAAI Trans. Intell. Technol.* 9(1):197–210, 2024 (online 2023), doi:10.1049/cit2.12161. The full text was read from a Wayback Machine snapshot of the Wiley page; this reproduces Eqs. (8)–(16).

$$\dot S = -(2\xi+1)S - (\xi^2+2\xi)\int_0^t S - \xi^2\int_0^t\!\!\int_0^{\sigma} S + H(t),$$

where the paper writes $\gamma=2\xi+1$ and $\lambda=\xi^2+2\xi$. The auxiliary states are $z=[\int S,\ \iint S]$.

The characteristic polynomial is $(s+1)(s+\xi)^2$. The derivation seeds the design with $\dot S=-S$ (Eq. 8), so **one pole stays fixed at $-1$ for every $\xi$**. Under linear noise $H=Qt+M$, the transfer $s^2/((s+1)(s+\xi)^2)$ forces $E\to0$ (Theorem 2). Paper simulations use $\xi=1$ (so $\gamma=\lambda=3$), and also $\xi\in\{0.5,2,5,50\}$.

**ASSUMPTION (mapping):** $\xi$ = `ctx.gamma` unless `xi` is given. The double pole $-\xi$ is then the fastest mode, with step $h=\gamma\tau$.

Related model, not implemented: the DISZNN of Hua et al. (*IEEE Access* 11, 2023) uses $\dot E=-3s_0E-3s_0^2\int E-s_0^3\iint E$, which is a triple pole.

**Tests.**
- `test_dieznn_characteristic_polynomial`: for $\xi=4$, RK4 matches $E_0\,[A e^{-t}+B e^{-\xi t}+C t e^{-\xi t}]$ with $A=(\xi-1)^{-2}$, $B=1-A$, $C=\xi^2/(1-\xi)$, to $10^{-8}$.
- `test_dieznn_rejects_linear_noise`: under $\nu=t+0.5$ with $T=8$: DIEZNN MSSRE is below $10^{-3}$; IEZNN sits at $\sqrt m/\lambda$ within 5%; ZNN exceeds 1 and keeps growing.

## (d) Harmonic-noise-tolerant ZNN — `hntznn`

Source: B. Liao, Y. Wang, J. Li, D. Guo, Y. He, "Harmonic noise-tolerant ZNN for dynamic matrix pseudoinversion and its application to robot manipulator," *Front. Neurorobot.* 16:928636, 2022, doi:10.3389/fnbot.2022.928636. The open full text (PMC9235923) gives Eqs. (11) and (14):

$$\dot E = -\tau\Psi(E) - \sum_{k=1}^{L}X_k + O(t),\qquad \dot X_k = C_k + 4\pi^2 f_k^2\lambda E,\qquad \dot C_k = -4\pi^2 f_k^2 X_k,\qquad \lambda\ge 1.$$

Here $-X_k$ learns the harmonic $o_k$ through the oscillator model $\ddot o=-(2\pi f_k)^2 o$, with known $f_k$ and unknown amplitude and phase. With a linear $\Psi$, the closed loop has transmission zeros at $\pm j2\pi f_k$. The activation is the Li activation $\psi(e)=\mathrm{Lip}^r(e)+\mathrm{Lip}^{1/r}(e)$ (Eqs. 4–5), **which is twice** the `sign_bi_power` activation in `activations.py`. It is implemented locally as `LiActivation` and not registered. Paper simulations use $\tau=10$ ($\tau\in\{50,100\}$ in Fig. 4) and $\lambda=1$.

**Mapping:** the paper's $\tau$ is `ctx.gamma`. Frequencies are given as `omegas` (rad/s), matching `noise/basic.py`'s `harmonic`, or as `freqs` (Hz). $z=[X_1..X_L, C_1..C_L]$.

**ASSUMPTIONS:**
- $X(0)=C(0)=0$; the paper does not say.
- Li exponent $r=0.5$; the paper does not give its simulation value.
- Default `activation='li'`, as in the paper. `activation='linear'` is available.

**Tests.**
- `test_hntznn_rejects_matched_harmonic_noise`: noise is $\sin(2\pi t+0.3)$ with $\gamma=10$ and $\lambda=2$. Linear-$\Psi$ HNTZNN has MSSRE below $10^{-3}$. ZNN's floor exceeds 0.1. A mismatched internal model ($\omega=\pi$) stays above 0.05. The Li-activated default is below ZNN/10.
- `test_hntznn_multi_harmonic`: Eq. (14) with $\omega\in\{2\pi,10\pi\}$, $\gamma=20$, $T=12$; MSSRE is below $10^{-3}$ and below ZNN/100.

## (a′) Strictly predefined-time ZNN with time-varying gain — `sptc_nt_znn`

Source: W. Li, C. Guo, X. Ma, Y. Pan, "A strictly predefined-time convergent and noise-tolerant neural model for solving linear equations with robotic applications," *IEEE TIE* 71(1):798–809, 2024, doi:10.1109/TIE.2023.3241393.

**Second-hand transcription:** the equations below come from Table 1 and Sec. 4 of Yang et al. 2025. That paper shares two co-authors (W. Li and X. Ma) with the source. I could not access the primary text.

$$\dot E = -\gamma\Phi_t(E)+\delta(t),\qquad \Phi_t(x)=\begin{cases}x/(t_c-t), & t<t_c\\ x+|x|^p\operatorname{sign}(x)+\xi\operatorname{sign}(x)\ \text{(element-wise)}, & t\ge t_c.\end{cases}$$

Before $t_c$, the time-varying gain is $\gamma/(t_c-t)$, and $E(t)=E(0)\big((t_c-t)/t_c\big)^{\gamma}$, which reaches exactly 0 at $t_c$. Defaults follow Yang et al.'s comparison: $t_c=1$ s, $p=0.5$, $\xi=\zeta/\gamma$ and $\zeta=5\Delta$. **ASSUMPTION:** $\Delta$ = `noise_bound` = 0.5.

Optional `cap` saturates $1/(t_c-t)$. It is a ZNN-Bench addition; the default `None` is faithful. In a fixed-step simulation, the effective step $h=\gamma\tau/(t_c-t)$ exceeds 2 during the last $\gamma\tau/2$ seconds before $t_c$.

**Tests.**
- `test_sptc_nt_znn_reaches_zero_at_tc_continuous`: RK4 with $\tau=10^{-3}$ matches $((t_c-t)/t_c)^\gamma$ at $t\in\{0.5,0.9,0.99,0.995\}$ within a relative $10^{-3}$, and $\|E(0.995)\|<10^{-4}\|E_0\|$.
- `test_sptc_nt_znn_after_tc_bounded_noise`: under $0.2\cos t$, the residual stays below 0.02 for $t\ge1.05$ (sliding regime). Linear ZNN keeps a floor above 0.05.

## (e) Conformable fractional-order SPTC anti-noise ZNN — `sptc_an_foznn`

Source: Y. Yang, X. Li, X. Wang, M. Liu, J. Yin, W. Li, R. M. Voyles, X. Ma, "A strictly predefined-time convergent and anti-noise fractional-order zeroing neural network for solving time-variant quadratic programming in kinematic robot control," *Neural Networks* 186:107279, 2025, doi:10.1016/j.neunet.2025.107279, arXiv:2503.01857. The equations were transcribed from the arXiv HTML LaTeX source.

The conformable derivative is $W_\alpha(f)(t)=t^{1-\alpha}f'(t)$ for $0<\alpha\le1$ (Khalil et al. 2014). The design formula (11), $W_\alpha(E)=-\gamma\Phi(E)$, is therefore **the integer-order ODE with time-varying gain $\gamma t^{\alpha-1}$** (Eqs. 12–13):

$$\dot E=-\gamma t^{\alpha-1}\Phi(E)+\delta(t),$$

$$\Phi(x)=\begin{cases}\dfrac{1}{\gamma t_c^{\alpha-1}}\Big(\dfrac{x}{t_c-t}+\dfrac{\gamma\|x\|^2}{(t_c-t)^2}\dfrac{x}{\|x\|}\Big), & t<t_c\\[2mm] \dfrac{1}{\gamma t^{\alpha-1}}\Big(\dfrac{x}{t_p-t}+\Big(\zeta+\dfrac{\gamma t^{\alpha-1}\|x\|^2}{(t_p-t)^2}\Big)\dfrac{x}{\|x\|}\Big), & t\ge t_c\end{cases}\qquad(14)$$

with $t_p=\big(1-e^{-\pi/(2\sqrt{\zeta(\zeta-\Delta)})}\big)t_c$, $\zeta>\Delta\ge\|\delta\|_\infty$, and $\|\cdot\|$ the vector 2-norm. For a matrix unknown, this is the Frobenius norm of the flattened error. The resulting error dynamics, Eqs. (15) and (18), are what the code implements:

$$t<t_c:\ \dot E=-(t/t_c)^{\alpha-1}\Big[\frac{E}{t_c-t}+\gamma\frac{\|E\|E}{(t_c-t)^2}\Big],\qquad t\ge t_c:\ \dot E=\frac{E}{t-t_p}-\zeta\frac{E}{\|E\|}-\gamma t^{\alpha-1}\frac{\|E\|E}{(t-t_p)^2}.$$

Theorem 1 (noise-free) states that $V=\|E\|/(t_c-t)$ satisfies $\dot V\le-\gamma V^2/(t_c-t)$. Hence $V(t)\le\big(1/V_0+\gamma\ln\frac{t_c}{t_c-t}\big)^{-1}$ (Eq. 17) and $\|E(t_c)\|=0$.

Paper parameters:
- TVQP example: Euler with $\Delta t=10^{-3}$, $\gamma=2$, $t_c=1$, $\zeta=5\Delta$.
- Robot example: $\alpha=0.5$, $t_c=0.01$.
- The $\alpha$ values used in the TVQP figures are not stated.

Code defaults: $\alpha=0.5$, $t_c=1$. **ASSUMPTION:** $\Delta$ = `noise_bound` = 0.5, so $\zeta=2.5$ and $t_p\approx0.505$.

**ASSUMPTION:** $t^{\alpha-1}$ is evaluated at $\max(t,t_{\rm floor})$ with $t_{\rm floor}=10^{-3}$. The gain is singular at $t=0$, and the paper does not say how its Euler run starts. A fixed floor keeps the same ODE for every $\tau$. $E=0$ maps to $E/\|E\|=0$. The optional `cap` on $1/(t_c-t)$ is a ZNN-Bench addition. With $\alpha=1$ the gain is constant and the model becomes its integer-order counterpart; the paper gives that case no separate name.

**Tests.**
- `test_sptc_an_foznn_lyapunov_bound_and_predefined_time`: RK4 with $\tau=2.5\times10^{-4}$. $V$ is non-increasing, $V(t)$ is below the Theorem-1 bound, and $\|E(0.99)\|<10^{-3}\|E_0\|$.
- `test_sptc_an_foznn_parameters_and_noise_after_tc`: $t_p$ formula; $\zeta\le\Delta$ is rejected; under $0.2\cos t$ the residual stays below 0.02 for $t\ge1.05$.
- `test_sptc_rhs_match_paper_activations` (covers both SPTC models): an independent transcription of the activation (14), inserted in Eq. (12), and of the SPTC-NT-ZNN row of Table 1. It is compared with the coded error dynamics at $t\in\{0.2,1.0,1.29,1.3,2.4\}$ with $t_c=1.3$, $\alpha=0.6$ and three gains per batch, to $10^{-10}$.

## (f) Gradient-ZNN hybrid — `gznn`

Source: P. S. Stanimirović, S. D. Mourtas, V. N. Katsikis, L. A. Kazakovtsev, V. I. Krutikov, "Recurrent neural network models based on optimization methods," *Mathematics* 10(22):4292, 2022, doi:10.3390/math10224292, Eqs. (33)–(40). The full text was read from a Wayback Machine snapshot of the MDPI page. This is the time-varying extension of the hybrid GNN-ZNN (HZNN) models of P. S. Stanimirović, V. N. Katsikis, S. Li, *Neurocomputing* 316:124–134, 2018, doi:10.1016/j.neucom.2018.07.058. Those models are time-invariant only, and I could not access their full text.

The ZNN design is applied to the GNN gradient $E_G=\nabla\frac12\|E\|_F^2=J^\top E$. For $AYC=B$ this is $E_G=A^\top(AYC-B)C^\top$, with $J=C^\top\otimes A$ and $W=CC^\top\otimes A^\top A=J^\top J$:

$$\dot E_G=-\gamma E_G\ \Longrightarrow\ M\,\dot x=-\gamma J^\top E-\dot J^{\top}E-J^\top\partial_tE,\qquad M=\begin{cases}J^\top J, & J^\top J\ \text{nonsingular}\\ J^\top J+\beta I, & \text{rank deficient}\end{cases}\quad(39)\text{–}(40).$$

Paper simulation values (Sec. 4): $\lambda=10$ (our $\gamma$), $\beta=10^{-8}$, horizon $[0,10]$. Note that the paper's Eq. (35) prints $-\lambda A^\top(AYC-B)$ without the trailing $C^\top$. This is inconsistent with Eqs. (12) and (33), so the consistent form $-\gamma E_G$ is used.

**Regularization (`beta=1e-8`, `regularize='auto'`).** As in Eq. (39), $\beta I$ is added only when $J^\top J$ is rank deficient. ZNN-Bench detects this structurally ($m<n_x$, e.g. the redundant robots, where $J^\top J$ has rank $m$) or when the unregularized solve fails. `'always'` and `'never'` force the choice. **ASSUMPTION:** a numerically (not structurally) rank-deficient $J$ with $m\ge n_x$ is only caught if the LU solve fails; the generated ZNN-Bench problems have controlled condition numbers, so this does not arise there.

**$\dot J$ and the information regime (`jdot`, ZNN-Bench implementation).** The model never samples problem data itself, so it uses no future data and leaves stateful data wrappers untouched:
- `'analytic'`: $\dot J e_j=\partial_{x_j}\big(\partial_tE\big)$ (Schwarz). It is computed by a central difference of `problem.dEdt` in $x$ with step $10^{-4}\max(1,\|x\|_\infty)$. This is exact up to round-off for problems affine in $x$ and $O(h^2)$ otherwise. It is the derivative of whatever the problem delivers in its `dEdt` slot: the exact derivative, or the causal estimate of a derivative-estimating wrapper.
- `'bd1'`/`'bd2'`: backward differences of $J(x,d_j)$ over the data samples of the previous causal calls. This mirrors `simulate`'s `info='bd1'/'bd2'`, including its start-up: $0$ at $k=0$ and bd1 at $k=1$.
- `'none'`: $\dot J=0$. With $\partial_tE=0$ as well and a square invertible $J$, GZNN then reduces exactly to ZNN.
- `'auto'` (default): use `ctx.extra['info']` if `simulate` ever provides it (CORE_CHANGE_REQUESTS § variants). Otherwise infer the regime that produced `ctx.dEdt` by comparing it with `problem.dEdt(x,t)`, with $0$ and with the bd1/bd2 formulas over the stored samples. The detected regime is kept in `last_regime`. Non-causal stage evaluations use the exact derivative, as `simulate` does.

**Other implementation notes and limitations.**
- $J$ is assembled densely from `problem.JT`. A memory guard, `max_dense` = $4\times10^6$ Jacobian entries, is checked when `simulate` builds z0. Cost per evaluation: $O(Bmn_x^2+n_x^3)$, plus one `dEdt` call on $2Bn_x$ states for the analytic $\dot J$.
- The model is derived for problems linear in $x$. On nonlinear $E$, the term $(\partial_x J\,\dot x)^\top E$ is ignored.
- The model keeps the data samples of its last three causal calls (reset at $k=0$), so one instance must not be shared between concurrently running simulations.
- The family is `gnn-znn`.

The Dai, Chen, Xiao, Jia, He fuzzy-adaptive H-GNN-ZNN (*IEEE TII* 18(4):2434–2442, 2022, doi:10.1109/TII.2021.3093115) is **not implemented**: I could not access its equations.

**Tests.**
- `test_gznn_gradient_error_decays_exponentially`: RK4 gives $J^\top E(t)=e^{-\gamma t}J^\top E(0)$ to $10^{-6}$. Plain ZNN violates this by more than $10^{-3}$, so the test discriminates.
- `test_gznn_dense_jacobian_and_jdot`: dense $J$ equals $A(t)$; the analytic $\dot J$ equals `A.dot(t)` to $10^{-9}$, and on `sylvester` it matches a central difference of $J$ over exact data; the guard and invalid modes raise.
- `test_gznn_jdot_follows_information_regime`: GZNN makes exactly as many `data` calls as ZNN (counted with `CountingProblem`); `last_regime` equals the `info` passed to `simulate` for all four regimes; with `info='none'` the GZNN and ZNN residual traces coincide; a hand-built bd1 sequence gives $(A(t_2)-A(t_1))/\tau$.
- `test_gznn_wrapped_data_and_redundant_robot`: under `JitteredProblem` (jitter std $10^{-3}$) the GZNN MSSRE stays within 1.5× of ZNN's; on `robot_panda` ($m=3<n_x=7$) GZNN does not diverge and matches ZNN within 2×.

A manual check, not part of the test file, confirmed the $E_G$ exactness on `matrix_inversion`, `sylvester` and `pseudoinverse`, with relative deviation $\le2\times10^{-9}$.

## Discrete-time smoke tests

`test_discrete_euler_smoke` runs every key with Euler at $\tau=10^{-2}$, starting near $x^*(0)$ (perturbation 0.2). It checks that nothing diverges and that the residual decreases.

## Observations from ZNN-Bench runs (not claims of the source papers)

1. **DIEZNN** keeps a pole at $-1$ for every $\xi$, so its transient never decays faster than $e^{-t}$. With $T=10$, the MSSRE window of the last 25% starts at 7.5 s; the slow mode decays like $e^{-t}$ and dominates the floor whenever it is excited and other errors are small.
2. **HNTZNN.**
   - With several harmonics, the closed loop has a lightly damped pole pair between the frequencies. For $\gamma=10$, $\lambda=2$, $\omega\in\{2\pi,5\}$ the slowest pole is $\approx-0.06$, and the best two-harmonic settings found were around $-1$.
   - Raising the effective gain pushes the compensator poles towards $\pm j\omega$, which slows the rejection. The infinite-slope Li activation gives an unbounded effective gain near $E=0$. In RK4 tests, the Li-activated model rejects harmonic noise much more slowly than the linear one: MSSRE $8\times10^{-3}$ versus below $10^{-3}$ at $T=6$. The paper's Lyapunov proof is asymptotic only and states no rate. The $\mathrm{Lip}^{1/r}$ term ($|e|^2$ for $r=0.5$) also makes the effective gain grow with $|e|$: with Euler, $\tau=10^{-3}$, $\gamma=10$, the Li-activated default diverged from random initial states on `pseudoinverse` ($\|E_0\|\approx500$), while the linear-$\Psi$ variant does not.
3. **SPTC models in discrete time.** The gain $1/(t_c-t)$ makes $h>2$ just before $t_c$. SPTC-AN-FOZNN also has a quadratic term $\gamma\|E\|E/(t_c-t)^2$ and a singular gain $t^{\alpha-1}$ at the start. With Euler at $\tau=10^{-2}$ and $\gamma=2$, it diverged within the first 0.05 s for random initial states on `linear_system` (4 of 4 trajectories), `matrix_inversion`, `pseudoinverse` and `sylvester`. At the paper's step $\tau=10^{-3}$ it converges when started near the solution, which is the setting of Theorems 1–2 ("initial condition sufficiently close"). From random initial states it still diverges at $\tau=10^{-3}$, $\gamma=2$ on problems with $\|E_0\|\gtrsim30$: `complex_inversion`, `lyapunov`, `pseudoinverse` and `sylvester` (verification run, Euler, $T=2$).
4. **GZNN** drives $J^\top E$ to zero at rate $\gamma$, but the residual $E=J^{-\top}E_G$ can grow during the transient when $J$ is ill-conditioned. On `pseudoinverse`, $\|E\|$ went from 506 to 582 while $E_G$ fell by $e^{-3}$.
5. **IEZNN** does not reject linear noise: the limit is exactly $\|Q\|/\lambda$. This is what motivated DIEZNN.
6. **Uncapped VP gains** (power or exponential) eventually violate $h=\gamma\tau g(t)<2$ for any fixed $\tau$, so every Euler-discretized VP-ZNN diverges on a long enough horizon. The cap or the implicit error step removes this.

## Verification (adversarial review, 2026-10-18)

A second agent re-derived every component from its source and checked the code numerically. The checks were run with throw-away scripts outside the repository and with the tests listed above.

**Sources re-read and compared symbol by symbol.**

| key | source used for the check | result |
|---|---|---|
| `sptc_an_foznn` | arXiv:2503.01857v1 HTML, LaTeX alttext of Eqs. (10)–(24), Table 1, Sec. 4 | Eqs. (14), (15), (18), the $t_p$ formula, Theorem 1 bound (17) and the TVQP parameters ($\Delta t=10^{-3}$, $\gamma=2$, $t_c=1$, $p=0.5$, $\zeta=5\Delta$, $\xi=\zeta/\gamma$) and robot parameters ($\alpha=0.5$, $t_c=0.01$) match the code and spec. |
| `sptc_nt_znn` | Table 1 of the same paper (second-hand) | Activation $x/(t_c-t)$, then $x+|x|^p\operatorname{sign}x+\xi\operatorname{sign}x$, with $\gamma(t)=\gamma$: matches. The primary IEEE TIE text was still not accessible. |
| `dieznn` | Wiley full text (Wayback snapshot of doi:10.1049/cit2.12161), Eqs. (6)–(16) and Sec. 5 | Eq. (14) coefficients $2\xi+1$, $\xi^2+2\xi$, $\xi^2$, the derivation seed $\dot S=-S$, and the paper's $\xi=1$, $\{0.5,2,5,50\}$ and $H=[10t+10]$ match. Roots of $s^3+(2\xi+1)s^2+(\xi^2+2\xi)s+\xi^2$ are $\{-1,-\xi,-\xi\}$ (checked numerically for $\xi\in\{0.5,1,4,10\}$). |
| `ieznn` | DIEZNN paper Eq. (6), the same formula as Jin et al. 2016 | $\dot S=-\gamma S-\lambda\int_0^tS+H$: matches (secondary). |
| `hntznn` | Europe PMC full-text XML of PMC9235923, Eqs. (2)–(16), Figs. 1–12 | Eqs. (11)/(14) ($\dot X_k=C_k+4\pi^2f_k^2\lambda E$, $\dot C_k=-4\pi^2f_k^2X_k$, $\lambda\ge1$), the Li activation without the factor 1/2 (Eqs. 4–5) and the paper's $\tau=10$, $\lambda=1$ match. The activation exponent is never given a value in the paper, which confirms that assumption. |
| `gznn` | MDPI full text (Wayback snapshot), Eqs. (31)–(40) and Sec. 4 | $E_G=A^\top EC^\top=J^\top E$, $W=CC^\top\otimes A^\top A=J^\top J$, Eq. (35)'s missing $C^\top$ (typo) and Eq. (39) match. **The paper's $\beta=10^{-8}$ and $\lambda=10$ were missing from the first version; see the bugs below.** |
| gains | Biomimetics 10(5):279 (PMC12108775), Table 2 and running text | $\zeta+t^\gamma$, $p^t+p$, $t^p+p$, $\beta e^{\lambda_1\operatorname{arccot}t+\lambda_2t}$ (Table 2), $t^\kappa+\kappa$ and $\varepsilon e^t$ (text): match. The spec previously attributed the last two to Table 2; this is corrected. |

All 17 DOIs cited in this spec and in the docstrings were resolved through the Crossref API; authors, title, journal, volume, issue, pages and year agree. The web-search budget of the session ran out during verification, so no further primary texts could be located: IEZNN/NTZNN (TNNLS 2016, TAC 2017), SPTC-NT-ZNN (TIE 2024) and the VP papers stay on secondary sources.

**Independent numerical checks.**
- HNTZNN closed-loop poles (linear $\Psi$), from the companion matrix: slowest real part $-0.952$ for $\gamma=20,\lambda=2,\omega\in\{2\pi,10\pi\}$ and $-0.061$ for $\gamma=10,\lambda=2,\omega\in\{2\pi,5\}$, as claimed. The noise-to-error transfer $(s^2+\omega^2)/\big((s+\gamma)(s^2+\omega^2)+\lambda\omega^2s\big)$ has zeros at $\pm j\omega$ and is Hurwitz for every $\lambda>0$ (Routh), so $\lambda\ge1$ is sufficient but not necessary for a linear $\Psi$.
- Batch contract: for every key and for `euler`, `rk4`, `taylor_zhang`, a $B=3$ run with per-trajectory gains $\{1.5,3,7\}$ equals three $B=1$ runs to $\le1.2\times10^{-10}$.
- Every key on every registered problem (13 problems, Euler, $\tau=10^{-3}$, $T=2$, random $x_0$). `sptc_an_foznn` and the Li-activated `hntznn` diverge from far initial states (see Observations). `gznn` diverged on both robots before the fix and runs after it.
- `gznn`: the new $\dot J$ matches a central difference of $J$ over exact data to $\le10^{-9}$ (relative) on `matrix_inversion`, `sylvester`, `pseudoinverse`, `nonlinear_equations` and `tvqp_eq`. GZNN MSSREs on all non-robot problems are unchanged by the fix to the printed precision.

**Bugs found and fixed.**
1. `gznn` diverged at the first step on every problem with $m<n_x$ (`robot_panda`, `robot_puma560`): `beta=0` was the default, so the singular $J^\top J$ was solved directly. Now Eq. (39) is implemented with the paper's $\beta=10^{-8}$, applied when $J^\top J$ is rank deficient.
2. `gznn` computed $\dot J$ by a finite difference of $J$ over `problem.data(t ± 1e-5)`. This (i) used future data under every information regime, including `bd1`/`bd2`/`none`; (ii) called `data` at instants other than $t_k$, which corrupts the sample history of stateful wrappers such as `EstimatedDerivativeProblem`; and (iii) amplified the independent per-instant jitter or measurement noise of `JitteredProblem` and `EstimatedDerivativeProblem(meas_std)` by $1/dt$. With jitter std $10^{-3}$ the GZNN MSSRE was 7× ZNN's ($1.2\times10^{-2}$ vs $1.7\times10^{-3}$); after the fix it is $1.75\times10^{-3}$. $\dot J$ now follows the information regime, as described in (f), and GZNN makes the same number of `data` calls as ZNN.
3. `LiActivation` declared `convergence='exponential'` (the base-class default). It is now `'fixed'`, the class of `sign_bi_power`, which it rescales.
4. Spec corrections: a "SPTC-AN-ZNN" name that does not appear in the source is removed; the attribution of two VP formulas within the review is corrected; Observation 3 now records that SPTC-AN-FOZNN also diverges at $\tau=10^{-3}$ from far initial states.

**Tests added:** `test_gznn_jdot_follows_information_regime`, `test_gznn_wrapped_data_and_redundant_robot` (both fail on the old code) and `test_sptc_rhs_match_paper_activations`. `test_gznn_dense_jacobian_and_dJdt` was replaced by `test_gznn_dense_jacobian_and_jdot`. Final run: `tests/test_variants.py` 33 passed in about 28 s, peak memory about 64 MB.


## Integration change (integrator, 2026-10-18)

`hntznn` evaluates its activation as `self.act.at(ctx.t)(E)`, the same convention as `DesignFormulaZNN` (models/znn.py). A time-varying activation such as `strict_predefined` is then frozen at the evaluation time instead of raising. The default Li activation is time-invariant (`at(t)` returns itself), so nothing changes for it. With the same change in `DesignFormulaZNN`, `make_model("znn", activation="strict_predefined")` reproduces `sptc_nt_znn` before t_c (tested to 1e-12 in `tests/test_integration.py`).
