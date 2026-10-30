# Spec: `problems_opt`: time-varying nonlinear equations and optimization problems

Files: `znnbench/problems/optimization.py`, `tests/test_problems_opt.py` (85 tests, about 7.5 s, well under 100 MB RAM).

Registered keys: `nonlinear_equations`, `tv_unconstrained_opt`, `tvqp_eq`, `tvqp_ineq`.

All four follow the core `Problem` contract (`docs/DESIGN.md`):
- `E`, `dEdt`, `solve_J` and `JT` are vectorised over the batch `(B, n_x)`.
- `dEdt` is the exact partial time derivative at fixed state.
- `solve_J` solves `J dx = r` exactly. Every Jacobian here is square and nonsingular, except at the documented degenerate instants of `yang2025`.

Each class also exposes `jacobian(x, d)`, a dense `(B, m, n_x)` Jacobian. It is advertised by the capability `"dense_jacobian"`. The QP classes also advertise `"kkt"`. `tv_unconstrained_opt` also advertises `"gradient_field"` (E = ∇f with an SPD Hessian), the capability defined by the `structured_baselines` area (CORE_CHANGE_REQUESTS.md, item 3) that lets `pc_gradient` use Simonetto's exact step x − αE. The KKT problems do not declare it.

Everything a model sees is taken from the data dict `d`, so `simulate(..., data_noise=σ)` perturbs every time-varying coefficient. This includes the coefficient vector `c(t)` of the nonlinear system.

Legend:
- **VERIFIED**: transcribed from a full text that I read (rendered PDF page or JATS XML).
- **ASSUMPTION**: a choice I made that the source does not fix.
- **SYNTHETIC**: a ZNN-Bench instance generator, not taken from a paper. Synthetic variants exist only to allow scaling in n and κ. They never define a new model.

---

## 1. `nonlinear_equations`: time-varying nonlinear system f(x,t) = 0

ZNN design (for reference; the model itself lives in `models/znn.py`): with E = f, the design formula $\dot E=-\gamma\Phi(E)$ gives $J\dot x=-\gamma\Phi(f)-f_t$. Its forward-Euler discretisation is the Euler DTZNN $x_{k+1}=x_k-J^{-1}(x_k,t_k)\,(\tau f_t(x_k,t_k)+h f(x_k,t_k))$ with $h=\gamma\tau$. This is Eq. (5) of Ma and Huang (2025).

### variant `"guo2018"` (default, n = 4), VERIFIED

Sources:
- System used in: D. Guo, F. Xu, Z. Li, Z. Nie, H. Shao, "Design, verification and application of new discrete-time recurrent neural network for dynamic nonlinear equations solving," *IEEE Trans. Ind. Informat.* 14 (2018) 3936–3945, DOI [10.1109/TII.2017.2787729](https://doi.org/10.1109/TII.2017.2787729).
- Transcribed from: Z. Ma, S. Huang, "Design and analysis of combined discrete-time zeroing neural network for solving time-varying nonlinear equation with robot application," *Front. Neurorobot.* 19 (2025) 1576473, DOI [10.3389/fnbot.2025.1576473](https://doi.org/10.3389/fnbot.2025.1576473), Eq. (33) and its theoretical solution. Read in the open-access JATS full text (PMC12289663).

The Frontiers paper states that Guo et al. (2018) studied DTZNN models on the same Eq. (33). I could not open Guo et al. 2018 itself (closed access), so the attribution to Guo 2018 is second-hand. The equations themselves are confirmed by the exact reproduction of Table 1 (see the Tests section).

$$
f(x,t)=\begin{bmatrix}\ln x_1-\frac{1}{t+1}\\ x_1x_2-\sin t\,e^{1/(t+1)}\\ x_1^2-\sin t\,x_2+x_3-2\\ x_1^2-x_2^2+x_3+x_4-t\end{bmatrix},\qquad
x^*(t)=\begin{bmatrix}e^{1/(t+1)}\\ \sin t\\ 2-e^{2/(t+1)}+\sin^2 t\\ t-2\end{bmatrix}.
$$

The Jacobian is lower triangular:

$$J=\begin{bmatrix}1/x_1&0&0&0\\x_2&x_1&0&0\\2x_1&-\sin t&1&0\\2x_1&-2x_2&1&1\end{bmatrix},\qquad \det J=1,$$

so `solve_J` uses forward substitution. The time derivative at fixed x is

$$f_t=\big[\tfrac{1}{(t+1)^2},\;-\cos t\,e^{1/(t+1)}+\tfrac{\sin t\,e^{1/(t+1)}}{(t+1)^2},\;-\cos t\,x_2,\;-1\big]^T.$$

Implementation: `data(t)` carries $c(t)=[1/(t+1),\ \sin t\,e^{1/(t+1)},\ \sin t,\ t]$, so measurement noise acts on the time-varying coefficients.

Notes:
- Domain: the system needs $x_1>0$. For $x_1\le0$, `E` returns NaN (warnings suppressed) and `simulate` flags the trajectory as diverged.
- **ASSUMPTION**: `x0` draws $x_1\sim U(0.2,3)$ and $x_{2..4}\sim U(-2,2)$. The paper uses $x_0=0.2\cdot\mathbf 1$. Steady-state values do not depend on x0.
- **ASSUMPTION**: default horizon `T = 10` s. The paper does not state its duration, but T = 10 reproduces Table 1.

### variant `"manufactured"` (any n), SYNTHETIC

$f(x,t)=A(t)e+\beta e^{\circ3}$ with $e=x-x^*(t)$ and $A(t)=S(t)+\text{skew}\,(Q(t)-Q(t)^T)$:
- $S$ is `TVMatrix(spd=True, kappa)`.
- $Q$ is a smooth orthogonal matrix (`_tv.SmoothOrthogonal`).
- $x^*$ is `TVVector`.

Because $\mathrm{sym}(J)=S+3\beta\,\mathrm{diag}(e^2)\succ0$, J is nonsingular everywhere and $x^*(t)$ is the unique root. Defaults: n = 4, κ = 10, β = 1, skew = 1.

## 2. `tv_unconstrained_opt`: $\min_x f(x,t)$, E = ∇f, J = ∇²f

The ZNN for time-varying nonlinear optimisation zeroes the gradient: $E=\nabla_x f$, $J=\nabla^2_{xx}f$, $\dot E=-\gamma\Phi(E)$. This is the model class of L. Jin, Y. Zhang, *IEEE TNNLS* 26 (2015) 1525–1531, DOI [10.1109/TNNLS.2014.2342260](https://doi.org/10.1109/TNNLS.2014.2342260), cited here from secondary sources only.

- **`"zhang2021"`** (default, n = 2): $f=\tfrac12x^TV(t)x+v(t)^Tx$, with $V=\begin{bmatrix}\tfrac14\sin t+1&\tfrac12\cos t\\\tfrac12\cos t&\tfrac14\cos t+1\end{bmatrix}$ and $v=[\sin3t,\cos3t]^T$.
  - VERIFIED as the objective of benchmark (18) in X. Zhang, L. Chen, S. Li, P. Stanimirović, J. Zhang, L. Jin, *CAAI Trans. Intell. Technol.* 6 (2021) 394–404, DOI [10.1049/cit2.12019](https://doi.org/10.1049/cit2.12019). This paper is CC BY; I read the rendered pages 7–8.
  - **ASSUMPTION**: dropping the constraints of (18) is a ZNN-Bench usage, not an experiment from that paper.
  - $V\succ0$ for all t: $\min_t\det V\approx0.476$ and $\min_t\lambda_{\min}(V)\approx0.352$ (verified on a fine grid over one period).
- **`"quadratic"`** (SYNTHETIC data, literature problem class): $f=\tfrac12x^TP(t)x+q(t)^Tx$ with $P$ = `TVMatrix(spd, kappa)`. This is time-varying quadratic minimisation as in Y. Zhang, Y. Yang, G. Ruan, *Neurocomputing* 74 (2011) 1710–1719, DOI [10.1016/j.neucom.2011.02.007](https://doi.org/10.1016/j.neucom.2011.02.007). The solution is $x^*=-P^{-1}q$.
- **`"manufactured"`** (SYNTHETIC, non-quadratic): $f=\tfrac12e^TPe+\beta\sum_i\log\cosh e_i$ with $e=x-x^*(t)$. Then $\nabla f=Pe+\beta\tanh e$ and $\nabla^2f=P+\beta\,\mathrm{diag}(1-\tanh^2e)\succ0$.
- `f(x, d)` returns objective values for diagnostics.

**Not adopted: the nonconvex TVNO (43)** of M. Sun, M. Tian, Y. Wang, *Discrete Dyn. Nat. Soc.* 2019:4745759, DOI [10.1155/2019/4745759](https://doi.org/10.1155/2019/4745759). I read the Optimization Online 2018 preprint, p. 15. The paper attributes the problem to Zhang et al., *J. Comput. Appl. Math.* 2019. As printed it reads

$f=(x_1+10\sin\frac{\pi t}{40})^2+(x_2+\frac t4)^2+(x_3-e^{-t/4})^2+0.025(t-1)x_3x_4+(x_1+\ln(\sin\frac{\pi t}{40}+1))^2(x_2+\sin\frac t8+\cos\frac t8)^2-(2x_1+\sin t)x_3+(x_4+e^{-t/4})^2.$

This f is **unbounded below** for every t. Take $Q=0$, $x_1=x_3=s$ and $x_4=-e^{-t/4}-0.0125(t-1)s$. Along this ray $f\to-\infty$ as $s\to\infty$, quadratically for $t\ne1$ and linearly at $t=1$ (checked numerically at t = 0, 1, 5, 9). Its stationary points are degenerate. Damped Newton from the paper's $x_0=[1,2,3,4]$ stalls at $t=0$: $\|\nabla f\|=1.58$ after 99 iterations, with $\lambda_{\min}(\nabla^2f)\approx2.6\times10^{-5}$. The tracked stationary path has $\lambda_{\min}(\nabla^2f)<0$ for $t\gtrsim1$ (minimum $-0.16$ at $t\approx1.33$) and jumps to $|x|\sim3\times10^3$ by $t\approx2$. The paper itself omits the solution ("complicated expression"). The product term is probably a typo for a sum, but I did not guess a correction. Implementing a guessed version would amount to inventing a problem.

## 3. `tvqp_eq`: $\min\tfrac12x^TW(t)x+q(t)^Tx$ s.t. $A(t)x=b(t)$

Model and KKT formulation: Y. Zhang, Z. Li, "Zhang neural network for online solution of time-varying convex quadratic program subject to time-varying linear-equality constraints," *Phys. Lett. A* 373 (2009) 1639–1643, DOI [10.1016/j.physleta.2009.03.011](https://doi.org/10.1016/j.physleta.2009.03.011). I had access to the abstract only. The formulation below is the standard Lagrangian, written in the same form as Lemma 1 of Zhang et al. 2021.

With $L=\tfrac12x^TWx+q^Tx+\lambda^T(Ax-b)$ and state $z=[x;\lambda]\in\mathbb R^{n+p}$:

$$E(z,t)=K(t)z+\begin{bmatrix}q\\-b\end{bmatrix},\quad K=\begin{bmatrix}W&A^T\\A&0\end{bmatrix}=J,\quad \partial_tE=\dot K z+\begin{bmatrix}\dot q\\-\dot b\end{bmatrix},\quad z^*=K^{-1}\begin{bmatrix}-q\\b\end{bmatrix}.$$

Since K is symmetric, `JT(v) = v K`.

Variants:
- **`"zhang2021"`** (default; n = 2, p = 1): W = V, q = v as in §2, with $A=[\sin4t,\cos4t]$ and $b=\cos2t$. This is the equality part of benchmark (18) of Zhang et al. 2021 (VERIFIED from the coefficient listing). Eq. (18) prints "sin(4t)x1(t)cos(4t)x2(t) = cos(2t)" with the "+" missing; the matrix listing $A_1=[\sin4t,\cos4t]$ resolves the typo.
- **`"yang2025"`**: equality part of §4 below.
- **`"random"`** (SYNTHETIC): W = `TVMatrix(n, spd, kappa)` and A = `TVMatrix(p×n, kappa=2)`, so A has full row rank with σ_min ≥ 0.8. Also q = `TVVector(amp=2)` and b = A x_f with $x_f$ a smooth vector.

The data of Zhang and Li (2009)'s own simulation example are **UNVERIFIED** (closed access), so no variant claims to be that example.

## 4. `tvqp_ineq`: adds $C(t)x\le d(t)$, perturbed Fischer–Burmeister treatment

Sources (both VERIFIED, rendered PDF pages):
- X. Zhang et al., CAAI-TIT 6 (2021) 394–404, DOI [10.1049/cit2.12019](https://doi.org/10.1049/cit2.12019): Definition 1, the PFB function, Lemma 2 and Eq. (2).
- Y. Yang, X. Wang, R. M. Voyles, X. Ma, "A predefined-time convergent and noise-tolerant zeroing neural network model for time variant quadratic programming with application to robot motion planning," *Tsinghua Sci. Technol.* 31(3) (2026) 1610–1621 (online 2025), DOI [10.26599/TST.2024.9010202](https://doi.org/10.26599/TST.2024.9010202), arXiv:2412.20477: Eqs. (2)–(5), the matrices P, Q, ρ of Eq. (7), and §4.1.

The two papers cite *different* sources for the perturbed FB function (checked in their reference lists): Yang et al. [31] cite A. Nazemi, M. Nazemi, *Cognit. Comput.* 6 (2014) 484–495, DOI [10.1007/s12559-014-9249-0](https://doi.org/10.1007/s12559-014-9249-0); Zhang et al. [31] cite A. Nazemi, "A capable neural network framework for solving degenerate quadratic optimization problems with an application in image fusion," *Neural Process. Lett.* 47(1) (2018) 167–192 (DOI not verified; search budget exhausted during verification). The underlying ZNN design for equality and inequality constrained TVQP is W. Li, X. Ma, J. Luo, L. Jin, *IEEE TSMC* 51(7) (2021) 4028–4039, DOI [10.1109/TSMC.2019.2930763](https://doi.org/10.1109/TSMC.2019.2930763), which I did not open (closed access).

The perturbed FB function is

$$\varphi_\varepsilon(a,b)=a+b-\sqrt{a^2+b^2+\varepsilon},\qquad \varphi_\varepsilon(a,b)=0\iff a\ge0,\ b\ge0,\ ab=\varepsilon/2 .$$

With state $z=[x;\lambda;\mu]\in\mathbb R^{n+p+l}$, slack $s=d-Cx$ and $r=\sqrt{s^{\circ2}+\mu^{\circ2}+\varepsilon}$:

$$
E=\begin{bmatrix}Wx+q+A^T\lambda+C^T\mu\\ Ax-b\\ s+\mu-r\end{bmatrix},\quad
J=\begin{bmatrix}W&A^T&C^T\\A&0&0\\(\Lambda_1-I)C&0&I-\Lambda_2\end{bmatrix},\quad
\Lambda_1=\mathrm{diag}(s\oslash r),\ \Lambda_2=\mathrm{diag}(\mu\oslash r),
$$

$$\partial_tE=\big[\dot Wx+\dot q+\dot A^T\lambda+\dot C^T\mu;\ \dot Ax-\dot b;\ (I-\Lambda_1)(\dot d-\dot Cx)\big].$$

This matches $P(t)$ (= our J) and $Q(t)z+\rho(t)$ (= our $\partial_tE$) in Eq. (7) of Yang et al., and E matches $J(t)y+c(t)$ in Eq. (2) of Zhang et al. Note that Zhang et al.'s matrix $J(t)$ (bottom row $[-A_2,0,I]$, with the square root kept inside $c(t)=[v;-b_1;b_2-z]$, $z=\sqrt{d\circ d+\varsigma\circ\varsigma+\varepsilon}$) only rewrites E; it is not $\partial E/\partial z$ and is not used here. All shipped variants use constant box constraints, so $\dot C=0$ and $\dot d=0$.

Variants:
- **`"zhang2021"`** (default): data of §3 plus the box $C=[I;-I]$, $d=1.3\cdot\mathbf 1$. This is benchmark (18) of Zhang et al. 2021, which the paper attributes to Li et al. TSMC 2021 ("investigated in [32]"; not checked against Li et al.). The scalar objective printed in (18), $(\tfrac18\sin t+\tfrac12)x_1^2+(\tfrac18\cos t+\tfrac12)x_2^2+\tfrac12\cos t\,x_1x_2+\sin3t\,x_1+\cos3t\,x_2$, equals $\tfrac12x^TVx+v^Tx$ with the matrix listing (checked in a test).
  - It is also, verifiably, the instance behind Figs. 1–2 of Yang et al. 2025 (see the yang2025 caution below), with ε = 1e-6 and T = 20 s there.
  - The box is active about 8% of the time on [0, 10] s.
  - The arXiv preprint 2503.01857 (Yang et al., SPTC-AN-FOZNN, Eq. 26) uses a variant with sin and cos swapped ($A=[\cos4t,\sin4t]$, $b=\sin2t$, $\rho=[\cos3t,\sin3t]$) and ε = 1e-8. It is not shipped.
- **`"yang2025"`**: the coefficients **printed** in §4.1 of Yang et al. 2025, default T = 20 s (the paper's horizon), ε = 1e-6.
  - $\Omega=\begin{bmatrix}\tfrac14\cos t+1&\tfrac14\sin t\\\tfrac14\sin t&\tfrac14\sin t+1\end{bmatrix}$, $p=[\cos2t,\sin2t]^T$, $A=[\sin2t,\cos2t]$, $b=\sin3t$, $C=[I;-I]$, $d=\mathbf 1$ (VERIFIED against the rendered PDF, arXiv:2412.20477).
  - **CAUTION (verification finding): these printed data are not the data behind the paper's own results.** In Yang et al.'s Fig. 1 the states reach $y_i=\pm1.3$, which is impossible with the printed box $d=\mathbf 1$. Overlaying our `zhang2021` exact QP solution (CAAI benchmark (18), box ±1.3) on a 300-dpi render of Fig. 1 matches their red PTC-NT-FOZNN curve to a median deviation of 0.002 (max 0.02, about one pixel) for $t\ge1$ s. Every visible saturation interval also coincides to about 0.05 s (y1 = −1.3 near 4.85, 11.14 and 17.4 s; y2 = −1.3 near 1.8, 4.5, 8.1, 10.8, 14.4 and 17.1 s). The printed §4.1 data deviate by a median of 0.65–0.81. Yang et al.'s Figs. 1–2 are therefore reproduced by `variant="zhang2021"` with ε = 1e-6 and T = 20 s (their γ = 2). `test_yang2025_figures_use_benchmark18_data` encodes this.
  - `yang2025` is kept as a literal transcription of the printed equations and as an LICQ-degenerate stress test. It does **not** reproduce any published curve.
  - **Degenerate**: at $t=\pi/2+k\pi$ the row $A=[0,-1]$ with $b=-(-1)^k$ coincides with a box face ($x_2\le1$ for even k, $x_2\ge-1$ for odd k). LICQ fails there, and the multipliers blow up on one side of each such instant: λ and μ₂ for $t\to\pi/2^-$ (≈470 at t = 1.57, ≈3.9e3 at t = 1.5707), λ and μ₄ for $t\to3\pi/2^+$ (≈143 at t = 4.715). The primal solution stays well defined.
  - As a result the KKT residual of a discrete ZNN spikes there. Measured with linear ZNN, Euler, γ = 10, seed 0: max ‖E‖ for t > 2 s is 7.2e2 at τ = 1e-2 and 6.2e5 at τ = 1e-3. The peak size depends on how close a grid point falls to the singular instant. The earlier claim that these spikes match "the periodic spikes in Yang et al.'s Fig. 2" is **withdrawn**: that figure was computed on the zhang2021 data, where spikes come from active-set switches. Use the primal error, or exclude windows around $\pi/2+k\pi$, when reporting this variant.
- **`"random"`** (SYNTHETIC): n primal variables, p equalities, box $|x_i|\le$ `bound`, with W, q, A as in §3.
  - $b=Ax_f$ with $\|x_f\|_\infty\le$ bound/2, so the problem is strictly feasible.
  - The box is active about 50% of the time with the defaults (n = 4, p = 1, bound = 1, κ = 10).
  - **ASSUMPTION**: generic LICQ. It is not enforced.

Defaults:
- **ASSUMPTION**: ε = 1e-6, the value Yang et al. 2025 use (§4.1). Per the verification finding above, they used it on the zhang2021 instance itself. Zhang et al. 2021 only say "a tiny value close to zero".
- **ASSUMPTION**: `x0` ~ U(−1, 1) for all of x, λ and μ. Zhang et al. use zeros. The FB function is defined for any sign.

### Reference solution (cached per t)

1. **Smoothed root** $z_\varepsilon^*(t)$ (default `x_star`, `reference="smoothed"`). Computed by damped Newton on $E(\cdot,\varepsilon)=0$ to $\|E\|\le10^{-13}$.
   - It is warm-started from the last solved t (|Δt| ≤ 0.1).
   - Otherwise it uses smoothing-Newton continuation $\varepsilon_j=1,10^{-1},\dots,\varepsilon$ from z = 0.
   - The root is unique for a strictly convex QP with LICQ. It is the point that ZNN actually converges to, so the residual and solution-error metrics are consistent.
2. **Exact QP solution** (`qp_solution(t)`, or `reference="qp"`).
   - Active set = $\{i:\mu_i>s_i\}$ at the smoothed root.
   - The equality-constrained KKT system for that set is then solved exactly.
   - A primal–dual repair loop drops negative multipliers and adds violated constraints until $\mu\ge0$ and $s\ge0$ hold to 1e-10.
   - `stats` counts continuations, repairs and failures. There were 0 failures in all runs.
   - In tests the result equals brute-force enumeration of all active sets to 1e-8.
3. Cache: a dict keyed by float(t), cleared above 2e5 entries (about 40 MB worst case). Cost is about 40–75 µs per new t on (n, p, l) = (2, 1, 4).
4. If neither Newton nor the continuation converges (never observed for ε ∈ [1e-14, 1e-2]), `stats['smoothed_failures']` is incremented and a warning is issued, so an unconverged reference can no longer be returned silently.

### Smoothing-parameter effects (measured on a dt = 2e-3 grid over [0, 10] s; ZNN-Euler at fixed h = γτ = 0.3, T = 6 s)

The perturbed root sits on the interior-point central path: $s_i\mu_i=\varepsilon/2$. Compared with the exact QP point:
- The bias is $O(\varepsilon)$ for strictly complementary constraints. For example, an active constraint has $s_i=\varepsilon/(2\mu_i)$.
- The bias is $O(\sqrt\varepsilon)$ at instants where a constraint switches between active and inactive ($s_i=\mu_i=\sqrt{\varepsilon/2}$).
- Each switch is smoothed over a time window of width about $\sqrt\varepsilon/|\dot s|$, so $\ddot z^*_\varepsilon=O(\varepsilon^{-1/2})$ there.
- An explicit causal discretisation therefore loses its order until $\tau\ll\sqrt\varepsilon/|\dot s|$.
- The trade-off: a small ε means small bias but stiff transitions and large discrete tracking errors. A large ε means smooth transitions but large bias.

| variant | ε | max bias | median bias | max primal bias | MSSRE τ = 3e-2 … 3e-4 | slope |
|---|---|---|---|---|---|---|
| zhang2021 | 1e-2 | 2.9e-1 | 1.9e-2 | 1.0e-1 | 8.5e-1, 4.7e-2, 4.4e-3, 5.5e-4, 5.2e-5 | 2.07 |
| zhang2021 | 1e-4 | 2.0e-2 | 1.9e-4 | 1.0e-2 | 1.9, 7.4e-2, 7.8e-2, 3.5e-3, 4.3e-4 | 1.72 |
| zhang2021 | 1e-6 | 1.7e-3 | 1.9e-6 | 9.0e-4 | 2.0, 7.7e-2, 1.3e-1, 2.0e-2, 4.6e-3 | 1.17 |
| zhang2021 | 1e-8 | 1.1e-4 | 1.9e-8 | 3.8e-5 | 2.0, 7.7e-2, 1.5e-1, 2.1e-2, 1.9e-2 | 0.92 |
| random | 1e-2 | 2.4e-1 | 3.0e-2 | 6.9e-2 | 6.8e-2, 9.4e-3, 9.7e-4, 1.1e-4, 1.0e-5 | 1.92 |
| random | 1e-6 | 1.9e-3 | 3.1e-6 | 4.6e-4 | 2.8e-1, 8.3e-2, 2.0e-2, 3.3e-3, 4.4e-4 | 1.40 |

How to read the table:
- The maximum bias is about $(1\text{–}3)\sqrt\varepsilon$ and the median bias about $(2\text{–}3)\,\varepsilon$.
- With ε = 1e-2 the reference is smooth and ZNN-Euler shows its textbook $O(\tau^2)$ order. With ε ≤ 1e-6 and τ ≥ 3e-4 it does not, and the curve is non-monotone (pre-asymptotic) on zhang2021.
- Benchmarks that sweep τ on `tvqp_ineq` must report ε. The order-verification protocol of DESIGN.md is only meaningful for τ ≪ √ε.

---

## Tests (`tests/test_problems_opt.py`)

These run for every variant in 12 configurations, including `random` with p = 0 and ε = 1e-4:
- $\|E(x^*(t),d(t))\|_\infty<10^{-10}$ at 4 times.
- `dEdt` vs a central finite difference in t (δ = 1e-6, tol 1e-6).
- `jacobian` vs a finite-difference Jacobian; `solve_J` vs `solve(J_fd, r)`; `JT` vs $J_{fd}^Tv$.
- Batch consistency: E, dEdt, solve_J and JT row by row against B = 1. In `simulate`, two identical x0 in one batch match each other and a B = 1 run.
- ZNN-Euler smoke test (h = 0.2, T = 4): no divergence, the median residual on t > 3 is below 1% of the initial residual, and the final solution error is below 5e-2. The last check is skipped for yang2025, see above.

Reproduction of published numbers (`test_guo2018_reproduces_table1_euler`):
- Setup: ZNN + Euler with γ = h/τ, $x_0=0.2\cdot\mathbf1$, T = 10 s, and MSSRE (max ‖f‖ on the last 25%).
- This reproduces the Euler-DTZNN SSCE entries of Table 1 of Ma and Huang (2025) to 4 significant digits.
- Measured values for (h, τ):
  - (0.1, 0.01): 1.3996e-3, paper 1.399e-3
  - (0.15, 0.01): 9.388e-4, paper 9.385e-4
  - (0.2, 0.01): 7.055e-4, paper 7.056e-4
  - (0.1, 1e-3): 1.4139e-5, paper 1.414e-5
  - (0.15, 1e-3): 9.427e-6, paper 9.422e-6
  - (0.2, 1e-3): 7.070e-6, paper 7.070e-6
- The test checks all six of these to within 1%. The verifier reproduced them independently with a from-scratch Euler-DTZNN loop that does not use `simulate` (ratios 0.9999–1.0005). The max-over-window SSCE is the same for T = 10 s and T = 20 s, so the T assumption does not matter.

Other tests:
- Empirical order 2 ± 0.3 of ZNN-Euler at fixed h = 0.3 on `nonlinear_equations` and `tvqp_eq`, for τ = 3e-2 … 1e-3.
- `tv_unconstrained_opt`: E equals the finite-difference gradient of `f`, and the Hessian is symmetric positive definite.
- `tvqp_ineq` exact KKT check (zhang2021, yang2025, random): stationarity, feasibility, μ ≥ 0 and μ∘s = 0 hold to 1e-10, and the result equals brute-force active-set enumeration. At the exact QP point the KKT rows of E vanish and the FB rows lie in $[-\sqrt\varepsilon,0]$, which is the documented smoothing bias of E.
- The smoothed root satisfies $s>0$, $\mu>0$ and $s\circ\mu=\varepsilon/2$, plus checks of the `pfb` definition.
- The bias is below $5\sqrt\varepsilon$ and decreases with ε. The median bias is roughly linear in ε.
- Smoothing vs discretisation: the slope is 2 ± 0.4 at ε = 1e-2, the slope is below 1.6 at ε = 1e-6, and the error at the smallest τ is more than 10× larger at ε = 1e-6.
- Reference modes and the cache (no re-solve on repeated queries).
- The yang2025 LICQ failure: multipliers more than 50× larger near t = π/2 while |x| ≤ 1.
- Argument validation.
- Transcription tests: independent re-typings of the printed sources, compared with `E`. These are Eq. (33) of Ma and Huang 2025 in scalar form; the scalar objective and constraints of benchmark (18) of Zhang et al. 2021 for `tv_unconstrained_opt`, `tvqp_eq` and `tvqp_ineq`; and §4.1 of Yang et al. for `tvqp_ineq`/`tvqp_eq` `yang2025`. A further test checks that `jacobian` equals Yang et al.'s $P(t)$ built from its printed blocks and that `dEdt` equals $Q(t)z+\rho(t)$. Mutation check: swapping sin/cos in $V_{22}$, swapping $v$, or changing Yang's $b$ (with consistent derivatives) passed all of the original tests but fails these.
- `gradient_field` is declared by every `tv_unconstrained_opt` variant and by no KKT problem.
- `test_yang2025_figures_use_benchmark18_data`: the box-saturation instants read off Yang et al.'s Fig. 1 are active-bound instants of the `zhang2021` solution, and the printed `yang2025` data never leave |x| ≤ 1.

Run with: `cd znn-bench && python -m pytest tests/test_problems_opt.py -q`

## Open issues / not verified
- Full texts of Zhang and Li (2009), Guo et al. (2018), Li et al. TSMC (2021), Jin and Zhang TNNLS (2015) and Zheng–Li–Zhang TAC (2024) were not accessible (paywall or bot challenge). Their simulation data are not claimed.
- I found no literature time-varying non-quadratic unconstrained objective that is well posed. The one candidate, TVNO (43), is ill-posed as printed. So the only non-quadratic `tv_unconstrained_opt` instance is synthetic (`manufactured`).
- Time-varying C(t) and d(t) are supported in E and J. The `dEdt` of the shipped variants assumes constant boxes, which holds for all of them. A future variant with moving constraints must provide $\dot C$ and $\dot d$.

## Verification (adversarial review, 2026-10-18)

Every registered component was checked against the original source and with independent numerical scripts that do not reuse the implementation. The scripts were kept in a temporary working folder and are not part of the repository.

**Sources re-read**
- Ma and Huang, Front. Neurorobot. 2025: the JATS XML from Europe PMC (PMC12289663), with MathML linearised.
  - Eq. (33), its theoretical solution, Eq. (5) and Table 1 match symbol by symbol.
  - $x_0=0.2\in\mathbb R^4$ is printed. σ is the paper's name for τ.
- Zhang et al., CAAI-TIT 2021: the White Rose PDF (eprint 172573).
  - Lemma 1, Lemma 2, Definition 1, the PFB function, Eq. (2) $f=Jy+c$, the scalar objective (18), the coefficient listing, and the missing "+" in the equality of (18) all match.
  - Their paper uses γ = 2 and zero initial values.
- Yang et al., TST: the arXiv:2412.20477 PDF.
  - Eqs. (2)–(5) and (7) match: P is the true Jacobian, Q and ρ give $\partial_tE$.
  - §4.1 data and ε = 1e-6 match.
  - Its figures, however, do not correspond to the printed §4.1 data (see the caution under `yang2025`).
- arXiv:2503.01857 Eq. (26): the swapped sin/cos variant of (18) with ε (called τ there) = 1e-8. Confirmed. Not shipped.
- All 11 DOIs cited were resolved through the Crossref API, with authors, title, volume and pages checked. Yang et al. is now Tsinghua Sci. Technol. 31(3) (2026) 1610–1621; the citation is updated.
- The TVNO (43) of Sun–Tian–Wang could not be re-read: the preprint and Hindawi PDF were unreachable and the web-search budget was exhausted. The unboundedness argument was re-derived algebraically for the formula as transcribed and is correct. Only the transcription itself is unverified. The problem is not implemented.

**Independent numerical checks (all passed)**
- `guo2018` Table 1, all six Euler-DTZNN entries, reproduced from scratch: ratios 0.9999–1.0005. The analytic $f_t$ was checked by finite differences.
- `tvqp_ineq` exact QP point vs `scipy.optimize.minimize(SLSQP)`, on 157 times × 5 configurations (zhang2021, yang2025, random n = 4/5 with p = 1/2, p = 0): primal agreement ≤ 2e-8, at SLSQP precision.
- Smoothed root vs `scipy.optimize.root(hybr)` on the same grid: ≤ 1.1e-13.
- Reference robustness on τ = 1e-3 grids over the full horizon: ε ∈ {1e-6, 1e-8, 1e-10}, yang2025 included, random n = 8 / p = 3. Also ε down to 1e-14 on a 5e-3 grid. max ‖E(x*)‖ ≤ 1e-13, with 0 failures.
- Positive definiteness: min det V = 0.476 and λ_min(V) = 0.352 for zhang2021; λ_min(Ω) = 0.589 for yang2025. The earlier "det ≥ 0.31" was loose and is corrected.
- Active-time fractions on [0, T]: 8.0% (zhang2021), 52% (random defaults), 32% (yang2025). These match the spec.
- Spot checks of the smoothing table (zhang2021 at ε = 1e-2 and 1e-6, random at ε = 1e-6) reproduced every MSSRE value and slope.
- Integration: all 15 registered models and all 38 discretizations ran on all four problems (T = 0.3 s, B = 2). The only failures were the expected capability errors of `znn_explicit_inverse*`.

**Bugs or defects fixed in this review**
1. `yang2025` mis-characterised. The printed §4.1 data are not what Yang et al. simulated; their Figs. 1–2 are the zhang2021 benchmark. The claim that the LICQ spikes "match Fig. 2" is withdrawn. Docstrings, spec and a new test were updated.
2. Wrong citation. Zhang et al. 2021 attribute the PFB function to Nazemi, Neural Process. Lett. 47 (2018), not to Nazemi & Nazemi, Cognit. Comput. 2014 (which is Yang et al.'s source).
3. `tv_unconstrained_opt` did not declare the `gradient_field` capability, so `pc_gradient` fell back to the generic $J^TE$ direction instead of Simonetto's exact step. It now declares it.
4. `_smoothed_at` could return an unconverged reference silently: the `ok` flag was computed and then ignored. It now counts `stats['smoothed_failures']` and warns.
5. Tests were self-consistent but blind to coefficient typos. Transcription tests, the Yang P/Q/ρ test, all six Table 1 entries and the yang2025 exact-KKT case were added. Coverage went from 76 to 86 tests, about 7.5 s.
6. Spec corrections: the det bound, the spike magnitudes (6.2e5, not 5e4, at τ = 1e-3, γ = 10), the one-sidedness of the multiplier blow-up and which multiplier blows up (μ₄ at 3π/2), the solver cost, and the citation year and volume.

**Not changed (judgement calls)**
- `yang2025` is kept as a literal transcription and degenerate stress test rather than removed.
- The PFB and Jacobian use the direct formulas $1-s/r$ and $a+b-r$, not cancellation-free forms. The reference still converges to 1e-13 for ε ≥ 1e-14, so no change was needed.

## Integration changes (integrator, 2026-10-18)

* **Singular dense Jacobian in the model-facing `solve_J`.** The smoke matrix found these cells: `tvqp_ineq` with `davidenko` or `vpznn_exp` under `zead_5i` (B = 3, `info='bd1'`, random-walk noise), and `sptc_an_foznn` under `pc_newton`. A diverging trajectory reached an exactly singular perturbed-FB Jacobian, `np.linalg.solve` raised `LinAlgError`, and the whole batch was aborted, so the runner would record `error` for every trajectory.
  * `_DenseJacobianMixin.solve_J`, `TVUnconstrainedOpt.solve_J` (`manufactured`) and `TVQPIneq.solve_J` now use `_bsolve_model`. It falls back to the minimum-norm least-squares step, `pinv(J) r` with rcond = 1e-12, which is the `Problem.solve_J` contract for rank-deficient J and the same fallback as `robot.py`.
  * The reference Newton solver still calls `_bsolve` and still sees the exception.
  * Regression test: `test_singular_jacobian_falls_back_to_min_norm_not_error`.
