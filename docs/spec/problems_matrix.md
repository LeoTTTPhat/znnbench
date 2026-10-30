# Spec: time-varying matrix problems (`problems_matrix`)

Files: `znnbench/problems/matrix.py`, `tests/test_problems_matrix.py`, this document.

Registered keys: `matrix_inversion`, `matrix_inversion_right`, `pseudoinverse`, `sylvester`,
`lyapunov`, `complex_inversion`.

## 0. Common conventions

* **Flattening (DESIGN.md).** The unknown matrix is row-major: `X = x.reshape(B, r, c)`. The
  error matrix is flattened the same way, so `||E||_2` of the flat error is the Frobenius norm
  of the error matrix, which is the residual the ZNN papers plot.
* **Affine error functions.** Every error function here is affine in X, so
  $J = \partial E/\partial x$ does not depend on x. It is never formed:
  * `solve_J` solves the matrix equation $\mathcal J(\Delta X) = R$ directly. Matrix inversion
    uses one LU of A per step. The pseudoinverse uses one Cholesky of the Gram matrix per step.
    Sylvester and Lyapunov use the Bartels-Stewart algorithm, which is the same as
    `scipy.linalg.solve_sylvester` (real Schur pair plus LAPACK `*trsyl`). Lyapunov needs only
    one Schur form, because in scipy's algorithm both Schur forms equal schur(A^T).
    One deliberate difference from scipy: LAPACK `*trsyl` returns $y$ with
    $\mathrm{op}(R)y+y\,\mathrm{op}(S)=\text{scale}\cdot F$ (scale $\le 1$ guards against
    overflow), so the solution is $y/\text{scale}$. scipy 1.17 returns $\text{scale}\cdot y$.
    The two agree whenever scale = 1, which holds for all well-scaled data. This code follows
    the LAPACK contract (fixed in verification, see Sec. 9).
  * Factorisations are cached by the **values** of the data matrices (an O(n^2) equality test).
    A second solve at the same sampling instant therefore reuses the factorisation, for example
    the noise term in `simulate` or `x_star` followed by `solve_J`. A cache can never go stale
    if data arrays are rebuilt or perturbed, which `test_factor_cache_tracks_data_values`
    checks. Caches are dropped when the problem is pickled.
  * The batch is handled by stacking all B right-hand sides into one 2-D solve (LU/Cholesky).
    For the Sylvester and Lyapunov triangular `trsyl` solves the code loops over the batch, but
    it **never loops over matrix entries**.
* **Data.** `data(t)` returns exact data that is memoised per t and marked read-only. `dEdt`
  uses the exact derivatives of the generators.
* **Random instances.** Random instances use `_tv.TVMatrix`: $A(t)=U(t)S(t)V(t)^T$ with smooth
  orthogonal U, V and singular values $s_{0,i}(1+0.2\sin(\omega_i t+\phi_i))$, where $s_0$ is
  geometrically spaced in $[1,\kappa]$. Hence $\mathrm{cond}(A(t)) \in [\kappa/1.5,\,1.5\kappa]$
  (tested). The other parameters are `freq` (rad/s), `seed` and horizon `T` (default 10 s).
* **Initial states.** $X(0)$ entries are drawn from $U[-2,2]$, as in Zhang & Ge (2005),
  Sec. V-A ("randomly selected in [-2,2]"). ASSUMPTION: the same range is used for every
  problem. The one exception is `lyapunov` with `example='hu_2020'`, which follows that
  paper's own initialisation of $E(0)$ (Sec. 5).
* **Capabilities.** Models use these flags to exploit problem structure:
  * `matrix_inversion` has `{'matrix_inverse'}`: $J\,dx=r \iff dX = A^{-1}R \approx XR$.
  * `matrix_inversion_right` has `{'matrix_inverse_right'}`: $dX = RA^{-1}\approx RX$.
  * `complex_inversion` has `{'complex_embedding'}`, with the layout given in Sec. 6.
  * The other problems have no flags.

## 1. `matrix_inversion`: E = A(t) X - I

**Source.** Y. Zhang, S. S. Ge, "Design and analysis of a general recurrent neural network
model for time-varying matrix inversion," *IEEE Trans. Neural Netw.* 16(6):1477-1490, 2005.
DOI [10.1109/TNN.2005.857946](https://doi.org/10.1109/TNN.2005.857946).

**Verification.** The equations were read from the full text (Maynooth repository PDF,
page 1478 rendered).

Eq. (1) and the error function above eq. (3):

$$A(t)X(t)-I=0,\qquad E(X(t),t):=A(t)X(t)-I \in\mathbb R^{n\times n}.$$

Design formula, eq. (3), and implicit model, eq. (4):

$$\frac{dE}{dt}=-\Gamma\mathcal F(E),\qquad A(t)\dot X(t)=-\dot A(t)X(t)-\Gamma\mathcal F\big(A(t)X(t)-I\big).$$

**Implementation.**
* $\mathcal J(\Delta X)=A\Delta X$
* $\partial_t E = \dot A X$
* $J^T V = A^TV$
* $x^*=\mathrm{vec}(A^{-1})$

The GNN in `models/znn.py` ($\dot x=-\gamma J^T\Phi(E)$) then becomes
$\dot X=-\gamma A^T\Phi(AX-I)$, which is the classical gradient network for inversion.

**Paper example.** `example='zhang_ge_2005'` reproduces Sec. V, the same example as Zhang,
Jiang & Wang (2002):

$$A(t)=\begin{bmatrix}\sin t&\cos t\\-\cos t&\sin t\end{bmatrix},\quad A^{-1}(t)=A^T(t).$$

The paper used $\gamma=1$, a power-sigmoid activation with $\xi=4$, $p=3$, and $X(0)\in[-2,2]$.

**Defaults.** `n=4, kappa=10, freq=1, seed=0, T=10, example=None`.

## 2. `matrix_inversion_right`: E = X A(t) - I

**Form.** This is the dual defining equation $X(t)A-I=0$. One source is I. S. Živković,
P. S. Stanimirović, "MATLAB simulation of the hybrid of recursive neural dynamics for online
matrix inversion," *Facta Univ. Ser. Math. Inform.* 32(5):799-809, 2017, eq. (1.2).
The paper's first page prints DOI 10.22190/FUMI1705799Z, but that DOI is **not registered**:
on 2026-10-18 both doi.org and Crossref returned 404, and the journal's article page lists
"DOI Number -". Full text (open access):
<https://casopisi.junis.ni.ac.rs/index.php/FUMathInf/article/view/3501>. The verifier read
eqs. (1.1) $AX(t)-I=0$ and (1.2) $X(t)A-I=0$ there (constant $A$).

**ASSUMPTION.** That paper states the dual form for a constant A. Applying it to a
time-varying A(t) with the ZNN design formula is the direct analogue of Sec. 1. It has the same
solution $A^{-1}(t)$, with:
* $\mathcal J(\Delta X)=\Delta X A$
* $\partial_tE=X\dot A$
* $J^TV=VA^T$

`solve_J` uses the same single LU with `trans=1`.

**Defaults.** As in Sec. 1. The `example` argument is also accepted.

## 3. `pseudoinverse`: full-rank Moore-Penrose inverse

**Source.** Y. Zhang, Y. Yang, N. Tan, B. Cai, "Zhang neural network solving for time-varying
full-rank matrix Moore-Penrose inverse," *Computing* 92(2):97-121, 2011.
DOI [10.1007/s00607-010-0133-9](https://doi.org/10.1007/s00607-010-0133-9).

**Verification.**
* The original full text was **not accessible** (paywalled).
* The error functions were transcribed from Y. Chai, H. Li, D. Qiao, S. Qin, J. Feng, "A neural
  network for Moore-Penrose inverse of time-varying complex-valued matrices," *Int. J. Comput.
  Intell. Syst.* 13(1):663-671, 2020. DOI
  [10.2991/ijcis.d.200527.001](https://doi.org/10.2991/ijcis.d.200527.001) (open access).
  * Eq. (1) gives $A^+$.
  * Eq. (3) is the right-inverse error function and eq. (10) the left one. The paper cites
    Zhang & Ge 2005 and Zhang et al. 2011.
* The real case below replaces $(\cdot)^H$ by $(\cdot)^T$.

Let $A(t)\in\mathbb R^{m\times n}$ have full rank. The unknown is $X\in\mathbb R^{n\times m}$.

* **Left inverse** (`side='left'`, $m\ge n$, full column rank),
  $A^+=(A^TA)^{-1}A^T$:

  $$E=A^TAX-A^T,\quad \mathcal J(\Delta X)=(A^TA)\Delta X,\quad \partial_tE=(\dot A^TA+A^T\dot A)X-\dot A^T,\quad J^TV=A^TAV.$$

* **Right inverse** (`side='right'`, $m<n$, full row rank), $A^+=A^T(AA^T)^{-1}$:

  $$E=XAA^T-A^T,\quad \mathcal J(\Delta X)=\Delta X(AA^T),\quad \partial_tE=X(\dot AA^T+A\dot A^T)-\dot A^T,\quad J^TV=VAA^T.$$

**Implementation.**
* `solve_J` uses one Cholesky of the Gram matrix per step, with an LU fallback.
* $x^*$ is computed from the normal-equation formula and equals `np.linalg.pinv` (tested
  together with the four Penrose conditions).
* Note that $\mathrm{cond}(J)=\mathrm{cond}(A)^2$.
* For the models agent: the GNN built from this error function, $\dot X=-\gamma A^TA\,\Phi(E)$,
  is the gradient flow of $\tfrac12\|A^TAX-A^T\|_F^2$. It is **not** the classical
  pseudoinverse GNN $\dot X=-\gamma A^T(AX-I)$.

**ASSUMPTION (shape).**
* By default `n` is the rank, and the other dimension is `n + max(1, n//2)`. For example,
  `n=4` gives a 6 x 4 matrix for the left inverse and 4 x 6 for the right.
* `rows`/`cols` override the shape, and the side then follows from the shape.
* The generator is `TVMatrix(rows, n_cols=cols)`, so the singular values lie in $[1,\kappa]$.

**Defaults.** `n=4, side='left', kappa=10, freq=1, seed=0, T=10`.

## 4. `sylvester`: A(t) X - X B(t) + C(t) = 0

**Source.** Y. Zhang, D. Jiang, J. Wang, "A recurrent neural network for solving Sylvester
equation with time-varying coefficients," *IEEE Trans. Neural Netw.* 13(5):1053-1063, 2002.
DOI [10.1109/TNN.2002.1031938](https://doi.org/10.1109/TNN.2002.1031938). This is the first
ZNN paper.

**Verification.**
* The original full text was **not accessible**.
* The equation form was transcribed from the MathML of L. Zhao, H. Shao, X. Yang, X. Liu,
  Z. Tang, H. Lin, "A novel zeroing neural network for dynamic Sylvester equation solving and
  robot trajectory tracking," *Front. Phys.* 11:1133745, 2023, eqs. (1)-(5). DOI
  [10.3389/fphy.2023.1133745](https://doi.org/10.3389/fphy.2023.1133745).
* This is the form usually quoted for Zhang et al. (2002).

$$A(t)X(t)-X(t)B(t)=-C(t),\qquad E(t)=A(t)X(t)-X(t)B(t)+C(t),$$

$$A\dot X-\dot XB=-\dot AX+X\dot B-\dot C-\gamma\Phi(AX-XB+C)\quad\text{(Zhao et al. eq. (5))}.$$

**Implementation.**
* $\mathcal J(\Delta X)=A\Delta X-\Delta XB$
* $\partial_tE=\dot AX-X\dot B+\dot C$
* $J^TV=A^TV-VB^T$
* `solve_J` applies Bartels-Stewart to $(A,-B)$, the same as
  `scipy.linalg.solve_sylvester(A, -B, R)` (tested per batch member).
* The task statement's form $AX+XB'=C'$ is the same problem with $B'=-B$, $C'=-C$. The
  literature sign convention is kept.

**ASSUMPTION (benchmark generator, not from the paper).** $X\in\mathbb R^{n\times k}$ with
$A=S_A+W_A$ and $B=-(S_B+W_B)$, where:
* $S$ is SPD, given by `TVMatrix(spd=True, kappa)`, with eigenvalues in $[0.8,1.2\kappa]$.
* $W=\tfrac{s}{2}(G-G^T)$ is skew-symmetric, with $G$ = `TVMatrix(kappa=1)` and $s$ = `skew`.
* $C(t)$ has entries $a\sin(\omega t+\phi)$ (`TVVector`).

Then $\langle X,\mathcal J X\rangle_F=\mathrm{tr}(X^TS_AX)+\mathrm{tr}(XS_BX^T)\ge1.6\|X\|_F^2$.
As a result:
* the solution is unique,
* $\|\mathcal J^{-1}\|\le 1/1.6$,
* $\mathrm{cond}(J)\le1.5(\kappa+s)$.

**Defaults.** `n=4, k=n, kappa=10, freq=1, skew=1, seed=0, T=10`.

## 5. `lyapunov`: A^T(t) X + X A(t) + Q(t) = 0

**Source.** Z. Hu, K. Li, K. Li, J. Li, L. Xiao, "Zeroing neural network with comprehensive
performance and its applications to time-varying Lyapunov equation and perturbed robotic
tracking," *Neurocomputing* 418:79-90, 2020.
DOI [10.1016/j.neucom.2020.08.037](https://doi.org/10.1016/j.neucom.2020.08.037). The paper
writes M for the coefficient matrix that is called A here.

**Verification.** The equations were read from the full text (page 80 rendered).

Eqs. (1), (2) and (4):

$$M^TX+XM=-Q,\qquad E=M^TX+XM+Q,\qquad M^T\dot X+\dot XM=-\dot M^TX-X\dot M-\dot Q-\mu\mathcal A(M^TX+XM+Q).$$

X is a full $n\times n$ unknown, as in the paper's Kronecker form (5).

**Implementation.**
* $\mathcal J(\Delta X)=A^T\Delta X+\Delta XA$
* $\partial_tE=\dot A^TX+X\dot A+\dot Q$
* $J^TV=AV+VA^T$
* `solve_J` applies Bartels-Stewart to $(A^T, A)$ with one real Schur form per step.
* $x^*$ is cross-checked against an independent dense Kronecker solve.

**ASSUMPTION (benchmark generator).** $A=-(S+W)$ is Hurwitz, with $S$ and $W$ as in Sec. 4,
so $\mathrm{Re}\,\lambda(A)\le-0.8$. $Q(t)$ = `TVMatrix(spd=True, kappa=q_kappa)` is symmetric
positive definite. Hence $X^*(t)$ is symmetric positive definite (tested), and
$\mathrm{cond}(J)\le1.5(\kappa+s)$.

**Paper example.** `example='hu_2020'` reproduces Sec. 5:

$$M(t)=\begin{bmatrix}-1-\tfrac12\cos3t&\tfrac12\sin3t\\ \tfrac12\sin3t&-1+\tfrac12\cos3t\end{bmatrix},\quad Q(t)=\begin{bmatrix}\sin3t&\cos3t\\-\cos3t&\sin3t\end{bmatrix}.$$

The paper used $\mu=2$, $\xi=1$, $p=0.4$, $\beta_1=\beta_2=1$, and random $|e_{ij}(0)|\le2$.

**Initial state of the example.** The paper randomises the initial **error** $E(0)$, not
$X(0)$ ("every element $e_{ij}(0)$ ... is random generated and is bounded by
$|e_{ij}(0)|\in[0,2]$"). `x0` therefore draws $E(0)$ with entries in $U[-2,2]$ and returns
$X(0)=\mathcal J^{-1}(E(0)-Q(0))$. ASSUMPTION: the distribution is uniform; the paper gives
only the bound. This was added in verification and is tested by
`test_lyapunov_hu2020_initial_error`.

**FINDING (paper inconsistency).**
* Sec. 2 says $Q$ is symmetric positive definite, but the printed example $Q(t)$ is not
  symmetric.
* The printed closed-form "theoretical solution" $A_{ij}(t)$ is the **transpose** of the
  exact solution of the printed equation, i.e. $A_{12}$ and $A_{21}$ are swapped. Equivalently,
  it solves $M^TX+XM+Q^T=0$.
* `test_lyapunov_hu2020_example` verifies this to 1e-12. A brute-force search over the sign
  patterns of all printed M and Q entries (a scratch check, not a unit test) shows that
  replacing Q by $Q^T$ is the only consistent reading.
* The benchmark uses the exact solution of the printed equation.

**Defaults.** `n=4, kappa=10, freq=1, skew=1, q_kappa=2, seed=0, T=10, example=None`.

## 6. `complex_inversion`: complex A(t) X = I in a real embedding

**Sources.**
* Y. Zhang, Z. Li, K. Li, "Complex-valued Zhang neural network for online complex-valued
  time-varying matrix inversion," *Appl. Math. Comput.* 217(24):10066-10073, 2011.
  DOI [10.1016/j.amc.2011.04.085](https://doi.org/10.1016/j.amc.2011.04.085). It defines the
  complex error $E=A(t)X(t)-I$. Only the metadata was checked via Crossref; the full text was
  not read.
* L. Ding, L. Xiao, B. Liao, R. Lu, H. Peng, "An improved recurrent neural network for
  complex-valued systems of linear equation and its application to robotic motion tracking,"
  *Front. Neurorobot.* 11:45, 2017, eqs. (3)-(5).
  DOI [10.3389/fnbot.2017.00045](https://doi.org/10.3389/fnbot.2017.00045). This is the real
  embedding, verified from the full text.
* The same stacking $\varphi(Z)=[\mathrm{Re}Z;\mathrm{Im}Z]$ appears as Definition 2 in Chai et
  al. (2020).

**Embedding (documented layout).** With $A=A_r+iA_i$ and $X=X_r+iX_i$:

$$\begin{bmatrix}A_r&-A_i\\A_i&A_r\end{bmatrix}\begin{bmatrix}X_r\\X_i\end{bmatrix}-\begin{bmatrix}I\\0\end{bmatrix}=\tilde E=\begin{bmatrix}\mathrm{Re}(AX-I)\\ \mathrm{Im}(AX-I)\end{bmatrix}.$$

* State: `X~ = x.reshape(B, 2n, n) = [X_r; X_i]`, i.e. `x[:, :n*n] = vec(X_r)` and
  `x[:, n*n:] = vec(X_i)`. The error has the same layout.
* `to_complex(x)` and `from_complex(Xc)` are public helpers for models with genuinely complex
  activations.
* Data keys are `'Ar'` and `'Ai'`, so measurement noise perturbs both parts.
* $\mathcal J(\Delta\tilde X)=\mathrm{embed}(A\Delta X)$, solved with one complex LU per step.
* $\partial_t\tilde E=\mathrm{embed}(\dot AX)$
* $J^T\tilde V=\mathrm{embed}(A^HV)$, so the GNN is the complex GNN $\dot X=-\gamma A^H\Phi(E)$.

**ASSUMPTION / equivalence.**
* Ding et al. use the embedding for a constant complex linear system. Applying it column-wise
  to time-varying complex inversion is the direct extension.
* With any element-wise activation applied to $\tilde E$, the real ZNN equals the complex ZNN
  whose activation acts separately on the real and imaginary parts ("type I", cf. Chai et al.
  eq. (5)).
* Activations acting on the modulus ("type II") need the `complex_embedding` capability and
  the helpers.

**Generator (ASSUMPTION).** $A(t)=U(t)S(t)V(t)^H$ with:
* $U(t)=P_U\,\mathrm{diag}(e^{i\omega_Ut})P_U^H$, where $P_U$ is a Haar-random unitary, and
  likewise $V(t)$;
* $S(t)$ as in `TVMatrix`.

This gives $\mathrm{cond}(A)\in[\kappa/1.5,1.5\kappa]$ (tested) and an exact $\dot A$.

**Defaults.** `n=4, kappa=10, freq=1, mod=0.2, seed=0, T=10`.

## 7. Efficiency at n = 64

Measured with `simulate`, `znn` and `euler`: tau = 1e-3, 200 steps, kappa = 100. The per-step
time includes `x_star` and residual recording at every step. Peak RSS of the whole run was
86 MB.

| problem | n_x | B=1 ms/step | B=8 ms/step |
|---|---|---|---|
| matrix_inversion n=64 | 4096 | 0.53 | 0.76 |
| matrix_inversion_right n=64 | 4096 | 0.54 | 0.80 |
| pseudoinverse 96x64 (left) | 6144 | 0.73 | 1.03 |
| pseudoinverse 64x96 (right) | 6144 | 0.75 | 1.05 |
| sylvester 64x64 | 4096 | 3.3 | 5.9 |
| lyapunov n=64 | 4096 | 2.3 | 5.0 |
| complex_inversion n=64 | 8192 | 0.64 | 1.49 |

For Sylvester, about half of the step time is spent evaluating the four `TVMatrix` generators.
The Python loop in `_tv.SmoothOrthogonal._R` is the main cost; a non-blocking suggestion is in
`CORE_CHANGE_REQUESTS.md`. The rest is one Schur pair plus the `trsyl` solves.

## 8. Tests (`tests/test_problems_matrix.py`, 108 tests, about 6 s, n ≤ 6)

These checks are parametrised over 11 configurations covering all keys and both paper
examples:
* shapes, the flattening rule, and that the problem pickles;
* $E(x^*(t),d(t))\approx0$ (relative 1e-11);
* $E(x,d)$ equals the source paper's error matrix, written out independently in the test, for
  random $X$ (`test_error_function_is_literature_formula`). This pins the side and sign
  conventions, which $E(x^*)=0$ alone cannot: $AX-I$ and $XA-I$ share the zero $A^{-1}$;
* `dEdt` against central differences in t with dt = 1e-5;
* `solve_J` against `np.linalg.solve` of a dense Jacobian built by central finite differences
  in x, and the Jacobian is checked to be independent of x;
* `JT` against the dense transpose;
* batch consistency: B = 3 equals stacked B = 1 for E, dEdt, solve_J and JT;
* continuous-time property (Zhang & Ge 2005, Theorem 1(i)): the linear ZNN integrated with
  RK4 (tau = 2e-3) reproduces $\|E(t)\|=\|E(0)\|e^{-\gamma t}$ to a relative 1e-7
  (`test_linear_znn_continuous_time_exponential_decay`). This checks dEdt, solve_J and the
  data trajectory together over a whole horizon;
* ZNN-Euler smoke test: residual drop of more than 1e3, small solution error, no divergence,
  and the steady residual halves when tau is halved (first order at fixed gamma).

Problem-specific checks:
* Zhang-Ge example: $A^{-1}=A^T$.
* Left and right inversion give the same solution.
* Capability semantics: at $x^*$, solve_J(r) = X*R (left) and R X* (right).
* Condition-number control.
* Moore-Penrose: matches `np.linalg.pinv` and satisfies the four Penrose conditions.
* Sylvester: matches scipy `solve_sylvester` for both x* and solve_J; the generator's
  symmetric parts are positive definite.
* `*trsyl` scale contract: a wrapped `trsyl` that returns scale = 0.5 (honouring the LAPACK
  contract) leaves the Sylvester solution unchanged (`test_trsyl_scale_contract`).
* Lyapunov: A is Hurwitz, Q is SPD, X* is SPD, and X* matches an independent Kronecker solve.
* Hu et al. example: checks the transposition finding and the $E(0)$ initialisation.
* Complex embedding: block form, $A^{-1}$, and $A^H$.
* Factor caches follow the data values.
* `info='bd2'` with `data_noise` runs.
* The GNN runs, which exercises JT.

## 9. Verification (adversarial pass, 2026-10-18)

The verifier checked every registered key independently of the implementer. Scratch scripts
were kept outside the repository.

**Sources re-read (primary full texts unless stated).**

| key | source checked | result |
|---|---|---|
| `matrix_inversion` | Zhang & Ge 2005, MURAL PDF, pp. 1478 and 1483: eq. (1), $E:=AX-I$, eqs. (3)-(4); Sec. V example $A(t)$, $A^{-1}=A^T$, $\xi=4$, $p=3$, $\gamma=1$, $X(0)\in[-2,2]$; "the same time-varying matrix as in [21]", where [21] is Zhang, Jiang & Wang 2002 | correct |
| `matrix_inversion_right` | Živković & Stanimirović 2017, journal PDF, eqs. (1.1)/(1.2) (constant A) | correct; printed DOI unregistered (citation updated, Sec. 2) |
| `pseudoinverse` | Chai et al. 2020: Springer now sits behind a bot-verification page, so the verifier could **not** re-read it. Zhang, Yang, Tan & Cai 2011: metadata only (Crossref) | equations not re-verified against text; mathematically confirmed (np.linalg.pinv, Penrose conditions, Kronecker form) |
| `sylvester` | Zhao et al. 2023, JATS XML: eq. (1) $AX-XB=-C$, eq. (2) $E=AX-XB+C$, eq. (5) | correct; original 2002 full text not accessible (same limitation as the implementer) |
| `lyapunov` | Hu et al. 2020, author PDF, pp. 80 and 85: eqs. (1)-(4); Sec. 5 $M(t)$, $Q(t)$, closed-form $A_{ij}(t)$, parameters | correct; transposition finding reproduced (below) |
| `complex_inversion` | Ding et al. 2017, JATS XML, eqs. (2)-(5): $[A_{re},-A_{im};A_{im},A_{re}][z_{re};z_{im}]=[b_{re};b_{im}]$. Zhang, Li & Li 2011: metadata only | embedding correct |

Every DOI in this spec was resolved through Crossref, and authors, venue, volume and pages
match. The one exception is the unregistered FUMI DOI.

**Independent numerical checks (scratch, not in the repository).**
* *Kronecker forms.* For every key, E, solve_J, JT and x* were compared with an explicit
  row-major Kronecker matrix: $A\otimes I$, $I\otimes A^T$, $(A^TA)\otimes I$,
  $I\otimes AA^T$, $A\otimes I-I\otimes B^T$, $A^T\otimes I+I\otimes A^T$, and the real
  block embedding $\otimes I$. All agree to $\le 6\times10^{-15}$.
* *Hu et al. (2020), Sec. 5.* The exact solution of the printed $M^TX+XM+Q=0$ (Kronecker solve)
  equals the **transpose** of the printed closed form to $10^{-16}$. The printed form leaves a
  residual of $O(1)$, for example 2.0 at $t=0$. Since $M(t)$ is symmetric, this is equivalent
  to replacing $Q$ by $Q^T$.
* *Continuous time.* RK4 (tau = 1e-3) on the linear ZNN gives the exact $e^{-\gamma t}$ decay
  to between $10^{-12}$ and $3\times10^{-10}$ for all keys. This became a unit test.
* *Integration.* `znn_explicit_inverse` and `znn_explicit_inverse_right` (models/structured.py)
  run on `matrix_inversion` and `matrix_inversion_right` with the Newton-Schulz start and match
  the implicit ZNN residual, which confirms the capability semantics.
* *Cost at n = 64* (B = 1, Euler): 0.55 (inv), 0.70 (pinv), 3.1 (sylv), 2.1 (lyap) and
  0.62 (complex) ms/step. Peak RSS was 77 MB.
* *Mutation tests.* The new tests catch each injected bug: scipy-style `scale*y`, the right
  form replaced by $AX-I$, $\dot Q$ dropped from the Lyapunov dEdt, the Hu x0 reverted to
  $X(0)\sim U$, and the Sylvester sign flipped to $AX+XB+C$.

**Bugs fixed.**
1. `_sylvester_solve` multiplied the `trsyl` output by `scale`, copying scipy, where it should
   divide. This is wrong whenever LAPACK rescales (scale < 1); the verifier confirmed the
   LAPACK contract with a 1x1 overflow case. It did not affect well-scaled data. Fixed and
   tested.
2. `lyapunov(example='hu_2020')` drew $X(0)$, whereas the paper draws $E(0)$ with
   $|e_{ij}(0)|\le2$. Fixed and tested.
3. Citation: the FUMI DOI printed on the paper does not resolve. The spec now says so and gives
   the journal URL.

**Considered and rejected.** The verifier tried adding an exact `Jv` (Jacobian-vector product)
for `prediction_correction.jacobian_vector`, then removed it. `ProblemWrapper.__getattr__`
would forward it past `CountingProblem`, so those calls would go uncounted. Its only consumer
is a Lipschitz power iteration, where the finite-difference fallback is exact enough.

**Still open.**
* The pseudoinverse error functions rest on the implementer's reading of Chai et al. (2020); the
  verifier could not re-read it.
* The 2002 Sylvester and 2011 complex-ZNN full texts remain inaccessible.
