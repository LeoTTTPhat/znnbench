# Area spec: structured_baselines

Files: `znnbench/models/structured.py`, `znnbench/models/baselines.py`,
`znnbench/discretizations/prediction_correction.py`, `tests/test_structured_baselines.py`.

Registered keys:

| kind | key | what it is |
|---|---|---|
| model | `znn_explicit_inverse` | explicit matrix-inversion ZNN = Getz–Marsden dynamic inverter (requires `matrix_inverse`) |
| model | `znn_explicit_inverse_right` | the same for the dual error X A − I (requires `matrix_inverse_right`) |
| model | `davidenko` | Davidenko / continuation flow (the linear ZNN with γ = 0) |
| model | `newton_flow` | continuous Newton method on the frozen problem (no dE/dt) |
| disc | `pc_newton` | NTT of Simonetto et al. 2016, Alg. 2 (**non-causal**) |
| disc | `pc_gradient` | GTT of Simonetto et al. 2016, Alg. 1 (**non-causal**) |
| disc | `pc_newton_causal` | same operations, but the corrections use t_k data (causal) |
| disc | `pc_gradient_causal` | same operations, but the corrections use t_k data (causal) |
| disc | `newton_tracking` | correction-only Newton with t_{k+1} data (non-causal) |
| disc | `newton_tracking_causal` | correction-only Newton with t_k data (causal running Newton) |
| disc | `running_gradient` | correction-only gradient with t_{k+1} data (Simonetto's "RG" baseline) |

**Update (gap-fill round, 2026-10-18):** `lcc_znn` is now registered in `models/structured_extra.py`
from its accepted manuscript (see [structured_extra](structured_extra.md)). `inverse_free_znn` is still
not registered (primary text not obtainable) and is the only entry left in `structured.PENDING_MODELS`.
§4 below is the original status note and is kept for the record.

Notation: the problem is E(x, t) = 0, J = ∂E/∂x, E_t = ∂E/∂t at fixed x. τ is the sampling
period and γ the ZNN gain, with h = γτ. Matrices are flattened row-major: x = vec_r(X).

---

## 1. Explicit matrix-inversion ZNN / Getz–Marsden dynamic inverter (`znn_explicit_inverse`)

**Sources.**
- N. H. Getz, J. E. Marsden, "Dynamic inversion and polar decomposition of matrices", *Proc.
  34th IEEE CDC*, 1995, pp. 142–147, doi:10.1109/CDC.1995.478664. Theorem 2.1, Eq. (1), with the
  dynamic inverse G[w, Γ, t] := Γw of Sec. 3. The equation was transcribed from the PDF in the
  Caltech repository and is **verified**.
- Journal version: Getz & Marsden, *Linear Algebra Appl.* 258 (1997) 311–343,
  doi:10.1016/S0024-3795(96)00235-2.
- The implicit ZNN for the same problem: Y. Zhang, S. S. Ge, *IEEE TNN* 16(6) (2005) 1477–1490,
  doi:10.1109/TNN.2005.857946.
- The ZNN ↔ "Getz–Marsden dynamic system" link: D. Guo, Y. Zhang, *Neurocomputing* 97 (2012)
  22–32, doi:10.1016/j.neucom.2012.05.012. Only the bibliography was verified; the full text was
  not accessible.
- The Newton-iteration special case: Y. Zhang, W. Ma, B. Cai, *IEEE TCAS-I* 56(7) (2009)
  1405–1415, doi:10.1109/TCSI.2008.2007065. Verified from the abstract.

**Equations.** The implicit ZNN for E = A X − I is A Ẋ = −Ȧ X − γ Φ(AX − I). Substituting
A^{-1} ≈ X gives

$$\dot X = -X\dot A X - \gamma\, X\,\Phi(AX - I). \tag{*}$$

Getz–Marsden 1995, Theorem 2.1, Eq. (1), reads $\dot\Gamma = -\mu\, G[A(t)\Gamma - I, \Gamma, t] - \Gamma \dot A(t)\Gamma$,
and with $G[w,\Gamma,t] = \Gamma w$ it is exactly (*) with the linear Φ and μ = γ.

In code, for capability `matrix_inverse` (E = vec_r(AX − I), so E_t = vec_r(ȦX)), (*) is the
generic design formula J ẋ = r − E_t with r = −γΦ(E) and J^{-1}(R) = A^{-1}R replaced by XR:
`xdot = X @ (r - E_t)`. The model uses only `ctx.x`, `ctx.E` and `ctx.dEdt`, so every information
regime applies: in `bd1`/`bd2`, ȦX is estimated by backward differences of E at fixed X. With
`ctx.implicit`, r is the resolvent rate of `DesignFormulaZNN`.

**Dual form** (`znn_explicit_inverse_right`, capability `matrix_inverse_right`, E = vec_r(XA − I)):
$\dot X = -X\dot A X - \gamma\,\Phi(XA - I)\,X$, implemented as `xdot = (r - E_t) @ X`.

**Defaults.** The activation is linear (any key of `znnbench.activations` works) and the gain is
the ZNN-Bench γ. Getz–Marsden prove only a **local** result: ‖X(0) − A(0)^{-1}‖ ≤ r and μ large
enough. The helper `newton_schulz_init(A0)` returns X(0) = Aᵀ/(‖A‖₁‖A‖_∞) (V. Pan, R. Schreiber,
*SIAM J. Sci. Stat. Comput.* 12(5) 1991, doi:10.1137/0912058). With this start, ρ(I − A X(0)) < 1.

**Benchmark caveat (verification).** Because the result is local, the model diverges in finite time
from the matrix problems' default X(0) ~ U[−2, 2] for most draws (Euler, n = 3, γ = 30, τ = 1e-2:
every trajectory diverges), whereas from `newton_schulz_init` it tracks with MSSRE ≈ 8e-4. The
runner passes `x0=None`, so benchmark configs must pass `x0=newton_schulz_init(A(0))` for this model
(core change request 5 in `CORE_CHANGE_REQUESTS.md`).

**Cost.** Each evaluation takes two n×n matrix products and no linear solve. The implicit ZNN
instead needs one LU solve with n right-hand sides.

**Assumptions.**
- ASSUMPTION (activation): the Φ-activated form of (*) follows the ZNN substitution technique of
  Guo & Zhang 2012 and Zhang–Ma–Cai 2009. The verified primary equation, Getz–Marsden, is the
  linear case.
- ASSUMPTION (dual): the right-hand form is the mirror image obtained by the same substitution. It
  was not checked against a paper.
- Capability contract, shared with `problems/matrix.py`, which already follows it:
  `matrix_inverse` ⇔ x = vec_r(X) with X ∈ ℝ^{n×n} and E = vec_r(A X − I).
  `matrix_inverse_right` ⇔ E = vec_r(X A − I).

**Tests.**
- `rhs` equals −XȦX − γXΦ(AX − I) for the linear and power-sigmoid activations; the dual `rhs`
  equals −XȦX − γΦ(XA − I)X (reference built from A, Ȧ directly) and matches the implicit ZNN at
  X = A^{-1}.
- Along the explicit flow, Ė (central difference) equals −γE − γE² − EȦX, the Getz–Marsden error
  dynamics derived from (*).
- At X = A^{-1}(t), `rhs` equals the finite-difference d/dt A^{-1}.
- Constant A with Euler and h = 1 gives exactly X⁺ = 2X − XAX (Newton–Schulz). From the
  Pan–Schreiber start it reaches residual < 1e-12 within 40 steps.
- Continuous time (RK4, τ = 1e-4, γ = 100): the log-residual decay rate lies within 10% of γ.
  The measured value is 100.08; the implicit ZNN gives 100.000. The linearised error dynamics are
  Ė = −γE − EȦX.
- Euler at fixed h = 0.3 has empirical order 2 ± 0.3; the measured value is 1.97.
- A `ValueError` is raised on a problem without the capability, and on the left/right mismatch.
- On the registered `matrix_inversion` and `matrix_inversion_right` problems: the E convention is
  checked, the explicit `rhs` equals the implicit ZNN `rhs` at x*, and the Euler MSSRE is at most
  2× that of the implicit ZNN.

## 2. Baselines (`models/baselines.py`)

### 2.1 `davidenko`: Davidenko / continuation flow

**Sources.**
- D. F. Davidenko, "On a new method of numerical solution of systems of nonlinear equations",
  *Dokl. Akad. Nauk SSSR* 88 (1953) 601–602, in Russian, no DOI.
- E. L. Allgower, K. Georg, *Introduction to Numerical Continuation Methods*, SIAM Classics 45,
  2003, doi:10.1137/1.9780898719154, Ch. 2.
- In time-varying optimisation it is the prediction field of Simonetto et al. 2016, Eq. (6).

$$J\dot x = -E_t \quad(\text{the linear ZNN with }\gamma = 0),\qquad \tfrac{d}{dt}E(x(t),t) = 0 .$$

The flow keeps the residual constant, which Simonetto et al. call the "iso-residual manifold", so
it needs a corrector:
- `davidenko` + `pc_newton` is the Euler-predictor / Newton-corrector continuation method, which is
  Simonetto's NTT.
- `davidenko` + `pc_gradient` is GTT.

γ is ignored. Under `info='bd1'`, E_t is the backward difference (E(x, t_k) − E(x, t_{k−1}))/τ,
which is exactly Simonetto's Eq. (10). Davidenko + PC is then AGT/ANT, Eq. (11).

**Tests.**
- d/dt E along the flow is below 1e-6 (central difference).
- RK4 with τ = 1e-3 over 2 s keeps ‖E‖ constant to a relative 1e-8.

### 2.2 `newton_flow`: continuous Newton method on the frozen problem

**Sources.**
- M. K. Gavurin, "Nonlinear functional equations and continuous analogues of iteration methods",
  *Izv. Vyssh. Uchebn. Zaved. Mat.* 1958, no. 5, 18–31 (in Russian; no DOI; citation from
  memory, not re-checked online).
- S. Smale, *J. Math. Econ.* 3 (1976) 107–120, doi:10.1016/0304-4068(76)90019-7.

$$\dot x = -\gamma J^{-1}E,\qquad \texttt{uses\_dEdt = False}.$$

With Euler and h = γτ = 1 this is the causal running Newton tracker
x_{k+1} = x_k − J(x_k, t_k)^{-1}E(x_k, t_k). That is the "correction-only" Newton method and the
h = 1 special case noted by Zhang–Ma–Cai 2009. Its lag is O(τ).

**Tests.**
- The trajectory is identical (rtol 1e-9) to `newton_tracking_causal`, also under a jitter wrapper
  with `data_noise` (both must see the same measured sample).
- Continuous time (RK4, τ = 1e-3): on a frozen problem ‖E(t)‖ = ‖E(0)‖e^{−γt} to rtol 1e-9; on a
  moving problem the lag floor scales like 1/γ (ratio for γ = 10 vs 100 in (7, 13)).

The *time-varying Newton tracking with prediction* is not a separate continuous model. In continuous
time, ẋ = −J^{-1}(γE + E_t) is just the linear ZNN (Simonetto et al., Proc. IEEE 2020, Sec. on
continuous-time controllers). Its discrete versions depend on the information pattern, so they are
discretizations (§3).

## 3. Prediction–correction discretizations (`discretizations/prediction_correction.py`)

**Sources.**
- A. Simonetto, A. Mokhtari, A. Koppel, G. Leus, A. Ribeiro, "A class of prediction-correction
  methods for time-varying convex optimization", *IEEE TSP* 64(17) (2016) 4576–4591,
  doi:10.1109/TSP.2016.2568161, arXiv:1509.05196. Transcribed from the arXiv PDF: Algorithms 1–4,
  Eqs. (6)–(11), Theorems 1–4.
- A. Simonetto, E. Dall'Anese, S. Paternain, G. Leus, G. B. Giannakis, *Proc. IEEE* 108(11) (2020)
  2032–2048, doi:10.1109/JPROC.2020.3003156.

**Paper's iteration.** For x*(t) = argmin f(x; t), with E = ∇ₓf and J = ∇ₓₓf, and sampling period
h = τ:

$$x_{k+1|k} = x_k - h\,[\nabla_{xx}f(x_k;t_k)]^{-1}\nabla_{tx}f(x_k;t_k) \tag{7}$$

Then f(·; t_{k+1}) is acquired (Alg. 1–2, step 3). With $\hat x^0 = x_{k+1|k}$ and
$s = 0,\dots,C-1$:

$$\text{GTT: }\hat x^{s+1} = \hat x^{s} - \alpha\,\nabla_x f(\hat x^{s};t_{k+1}) \tag{8}$$
$$\text{NTT: }\hat x^{s+1} = \hat x^{s} - [\nabla_{xx} f(\hat x^{s};t_{k+1})]^{-1}\nabla_x f(\hat x^{s};t_{k+1}) \tag{9}$$

and $x_{k+1} = \hat x^{C}$.

The approximate variants AGT/ANT replace ∇ₜₓf by
$\tilde\nabla_{tx} f = (\nabla_x f(x_k;t_k) - \nabla_x f(x_k;t_{k-1}))/h$, Eqs. (10)–(11).

The paper names the number of corrections τ and the gradient step γ. Here they are called C and α,
because τ and γ are taken.

**Mapping.** The prediction is one forward-Euler step of the paired model's field at (t_k, x_k):
- With `davidenko` it is exactly Eq. (7) under `analytic` and Eq. (11) under `bd1`.
- With the linear `znn` it is an Euler-ZNN step, so the prediction also carries the −γJ^{-1}E
  feedback.

The correction acts on the frozen problem E(·, d) = 0:
- Newton: x ← x − J^{-1}E.
- Gradient: x ← x − αE when the problem declares `gradient_field` (exact Eq. (8)), otherwise
  x ← x − αJᵀE.

**Information pattern, stated precisely.**

| key | prediction | correction | `causal` | scored x_{k+1} |
|---|---|---|---|---|
| `pc_newton`, `pc_gradient` | f at (t_k, x_k), data ≤ t_k | C steps with data **at t_{k+1}** | False | Corr_{t_{k+1}}(Pred_{t_k}(x_k)) |
| `pc_newton_causal`, `pc_gradient_causal` | f at (t_k, x̂_k), data ≤ t_k | C steps with the measured data at t_k, applied first | True | Pred_{t_k}(Corr_{t_k}(x_k)) |
| `newton_tracking` | none | C Newton steps, data at t_{k+1} | False | Corr_{t_{k+1}}(x_k) |
| `newton_tracking_causal` | none | C Newton steps, data at t_k | True | Corr_{t_k}(x_k) |
| `running_gradient` | none | C gradient steps, data at t_{k+1} | False | Corr_{t_{k+1}}(x_k) |

The causal and non-causal PC variants perform the same operations in the same order. They differ
only in which iterate ZNN-Bench scores at t_{k+1}:
- The paper scores the iterate corrected with t_{k+1} data. This is legitimate in its setting,
  where correction time is not charged against the deadline, but it is non-causal in the ZNN
  "future computation" sense, as RK4 is.
- The causal variant scores the prediction made from data up to t_k.

In both variants f is called only at t = t_k with the current k. The causal correction uses the
same, possibly noisy, sample at t_k that the model sees. The non-causal correction perturbs the
t_{k+1} data with `simulate`'s `data_noise`, exactly as `simulate` does for RK4 stage data.

The causal running gradient (gradient step with t_k data) is `gnn` + `euler` with γ = α/τ, so it
is not registered again.

**Expected orders** (steady-state residual vs τ; measured on the local test problems):

| key | theory | measured |
|---|---|---|
| `pc_newton` (C = 1) | O(τ⁴): Thm 2, bound O(τ^{4C}), local, needs ‖x₀ − x*(t₀)‖ ≤ cτ². Exact for affine E. | 4.01 (cubic); MSSRE ≈ 2e-15 (linear system) |
| ANT (`davidenko` + `pc_newton`, `info='bd1'`) | O(τ⁴): Thm 4 | 4.02 |
| `pc_newton_causal` | O(τ²) | 2.00 (cubic), 2.00 (linear system) |
| `pc_gradient` | O(τ²) when ρ^C σ < 1: Thm 1(ii) / Cor. 1, Eq. (20) | 1.96 (cubic, `davidenko`), 2.01 (linear system, `znn`) |
| `pc_gradient_causal` | O(τ²) | 1.98 |
| `newton_tracking` | O(τ²) nonlinear; exact for affine E | 1.98 |
| `newton_tracking_causal` | O(τ) | 1.00 |
| `running_gradient` | O(τ): correction-only bound of A. Y. Popkov, *Autom. Remote Control* 66(6) (2005) 883–891 (Simonetto's ref. [24]); RG = "AGT without prediction" (Sec. IV-B) | 1.03 (cubic); 0.68 on `linear_system` κ = 10 with `znn`, τ ∈ [1e-3, 3e-2], T = 4 (pre-asymptotic, see below) |

`expected_order` is the order on `linear_system` with the linear ZNN. It is None where the scheme
is exact on affine problems, so that order-checking harnesses do not fit round-off.
`expected_order_nonlinear` holds the nonlinear order.

**Defaults.**
- C = 1. Simonetto et al. use τ ∈ {1, 3, 5} in Sec. IV-A.
- `alpha='auto'`, `alpha_scale=0.5`, `direction='auto'`.
- Paper experiments (Sec. IV-A, scalar example): h = 0.1, γ ∈ {0.2, 1}, x₀ = 0. The text and the
  captions disagree on one constant (μ = 1.75 vs 1.5).

**Assumptions.**
- ASSUMPTION (gradient direction): Eq. (8) is a descent step only when E is a gradient field.
  Problems declare this with the capability `gradient_field`, which this area defines
  (`tv_unconstrained_opt` declares it); the family `"unconstrained-optimization"` is also
  recognised as a safety net (`GRADIENT_FIELD_FAMILIES`). Other problems use the gradient of
  ½‖E‖², αJᵀE, which is the GNN direction. This fallback is a ZNN-Bench extension, not part of
  Simonetto et al.
- ASSUMPTION (step size): the paper needs 0 < α < 2/L. With `'auto'`, L is estimated once per run
  at the first correction point by 30 power iterations: L = λ_max(J) for `residual` and
  σ_max(J)² for `normal`. J·v comes from `problem.Jv` when it exists, otherwise from a central
  difference of E. Then α = 0.5/L, leaving margin for the time variation of J. Pass a float α to
  override.
- ASSUMPTION (noise): additive design-space `noise` enters only through the model field, that is,
  the prediction. The static correction steps are noise-free.
- The auxiliary state z (if any) is advanced with forward Euler by the prediction. It is frozen in
  the correction-only schemes.
- **Workaround (core):** the Discretizer interface does not pass the Problem. The scheme uses
  `disc.bind(problem)` if called; otherwise it reads `problem`, `past`, `data_noise` and `rng`
  from the closure of `simulate`'s vector field. It searches up to 3 nested wrapper closures, for
  example `wrappers.QuantizedDiscretizer`. If neither works it raises a `RuntimeError` naming
  `bind`. See `CORE_CHANGE_REQUESTS.md`.

**Not implemented.** The inverse-free P-step prediction of Proc. IEEE 2020 (P gradient steps on the
quadratic model φ_k) would need J·v and E_t outside the model field. Extrapolation-based PC
(Bastianello–Carli–Simonetto 2023) is also not implemented.

**Tests.**
- Orders on both test problems, as in the table above.
- Non-causal error < causal error, for both Newton and gradient correction.
- Information-pattern spies:
  - Causal variants request no data after t_k. Through `simulate` they make no `data()` call at
    all, because they use `simulate`'s measured sample.
  - Non-causal variants request exactly t_{k+1}.
  - f is only called at (t_k, k).
  - The checks are repeated through the explicit `bind` path without `simulate`.
- The `bind` and closure paths give identical trajectories.
- The scheme still works inside `QuantizedDiscretizer(quantize_derivative=True)`.
- `data_noise` smoke test, the Lipschitz estimate vs the exact σ_max²/λ_max, the
  direction-resolution rule, and argument validation.

## 4. Requested but NOT implemented (honest status)

| key (reserved) | paper | status |
|---|---|---|
| `lcc_znn` | K. Zheng, S. Li, Y. Zhang, "Low-computational-complexity zeroing neural network model for solving systems of dynamic nonlinear equations", *IEEE TAC* 69(7) (2024) 4368–4379, doi:10.1109/TAC.2023.3319132 | Equations not obtained. Only the abstract is known: an explicit, pseudoinverse-free model with a new finite-time activation. |
| `inverse_free_znn` | J. Chen, Y. Pan, Y. Zhang, S. Li, N. Tan, "Inverse-free zeroing neural network for time-variant nonlinear optimization with manipulator applications", *Neural Networks* 178 (2024) 106462, doi:10.1016/j.neunet.2024.106462 | Equations not obtained. Only the abstract is known: it avoids both the matrix inverse and matrix–matrix products, for equality-constrained time-variant optimisation. |

Where the full texts were sought:
- University of Oulu repository (`oulurepo.oulu.fi/handle/10024/48415` and `/61885`) and the CORE
  mirror (`core.ac.uk/download/628873440.pdf`): both sit behind a bot-verification (Cloudflare)
  challenge, which was not bypassed.
- IEEE Xplore and ScienceDirect: paywalled or captcha-protected.
- Unpaywall, Semantic Scholar, arXiv, Europe PMC and citing open-access papers: none restate the
  equations.

The design contract forbids inventing a model, so no guessed form is registered. **To finish:** get
the accepted manuscripts (open under CC-BY-NC-ND at the Oulu URLs) through a normal browser, and
implement them in `structured.py` for the generic Problem API, or with `requires` if they need a
dense Jacobian.

## 5. Verification (adversarial review, 2026-10-18)

Each registered component was compared symbol by symbol with the primary source, and checked
numerically with independent scratch scripts. Results:

| component | source checked | verdict |
|---|---|---|
| `pc_gradient` (GTT), `pc_newton` (NTT), AGT/ANT via `info='bd1'` | Simonetto et al. 2016, arXiv:1509.05196v2 PDF: Eq. (6) continuous iso-residual dynamics, Eq. (7) Euler prediction, Eq. (8) gradient correction with step γ, Eq. (9) Newton correction, Eq. (10) backward difference, Eq. (11), Algorithms 1–4 | **correct.** Prediction uses data at t_k, the τ (here C) corrections use f(·; t_{k+1}); step-size range 0 < γ < 2/L (Thm 1); Thm 2 bound O(h^{4τ}) needs ‖x₀ − x*(t₀)‖ ≤ ch² (Eq. 23). |
| reproduction of the paper's Sec. IV-A example, Eq. (35)–(36) (ω = 0.02π, κ = 7.5, μ = 1.75, h = 0.1, γ = 0.2, x₀ = 0) | independent plain-loop transcription of Algorithms 1–4 | **correct.** ZNN-Bench (`davidenko` + PC) equals the transcription to ≤ 2.4e-15 for GTT, NTT, AGT, ANT and RG. Worst-case errors (k > 1000): GTT 1.3e-5, AGT 2.6e-5, NTT 3.6e-11, ANT 1.4e-10, RG 2.1e-3, matching the levels in the paper's Fig. 1–2 (RG ~1e-2, GTT ~1e-5, NTT ~1e-10). Now a unit test. |
| `running_gradient` | Simonetto Sec. IV ("AGT without prediction"), O(h) bound from Popkov 2005 (ref. [24]) | **correct**; the earlier citation "Cor. 1, Eq. (19)" was wrong (that is GTT's O(h) regime) and is fixed. |
| NTT orders | Thm 2 / Eq. (25), O(h^{4C}) | measured 4.00 (C = 1) and 7.93 (C = 2) on a cubic gradient field; `expected_order_nonlinear = 4C` is consistent. |
| `davidenko` | Simonetto Eq. (6); Davidenko / Allgower–Georg | **correct** (rhs = −J^{-1}E_t; residual invariant to 1e-8 under RK4). Docstring citation "Eq. (7)" corrected to Eq. (6). |
| `newton_flow` | continuous Newton method (Gavurin; Smale 1976) | **correct**; = linear ZNN under `info='none'`. Continuous-time test added. The Zhang–Ma–Cai 2009 statement is about matrix inversion; the docstring now says the general-E version is the same algebra, not a quotation. |
| `znn_explicit_inverse` (linear Φ) | Getz & Marsden, CDC 1995, Thm 2.1 Eq. (1) and Sec. 3 (G[w, Γ, t] := Γw; cf. Eq. (9) Γ̇ = −μΓ(HΓ − I) − Γ(M − I)Γ), Caltech repository PDF | **correct**: Γ̇ = −μΓ(AΓ − I) − ΓȦΓ. Theorem 2.1 is local (‖Γ(0) − Γ*(0)‖ ≤ r, μ > μ̄); see the benchmark caveat in §1. |
| `znn_explicit_inverse` with Φ, `znn_explicit_inverse_right` | Guo & Zhang 2012 full text not accessible | **unverifiable** against a paper (flagged ASSUMPTION). Checked algebraically: the dual form is the exact mirror substitution; at X = A^{-1} both equal the implicit ZNN. |
| `newton_schulz_init` | Pan & Schreiber 1991 | **correct**: X₀ = Aᵀ/(‖A‖₁‖A‖_∞) with ‖A‖₁ = max column sum, ‖A‖_∞ = max row sum, so σ(AX₀) ⊂ (0, 1]. |
| `lcc_znn`, `inverse_free_znn` | — | not implemented, not registered (equations not obtained; no further web search was possible in this review). |

**Expected orders with the linear ZNN on `linear_system` (n = 4, κ = 10, h = 0.3, T = 4,
τ ∈ {3e-2, 1e-2, 3e-3, 1e-3}):** `pc_newton` and `newton_tracking` exact (MSSRE ≈ 4e-15, so
`expected_order = None` is right); `pc_newton_causal` 2.00; `pc_gradient` 2.00;
`pc_gradient_causal` 2.01; `newton_tracking_causal` 1.00; `running_gradient` 0.68. The last is
pre-asymptotic: with the JᵀE direction and α = 0.5/σ_max², ρ ≈ 1 − 0.5/κ_eff² ≈ 0.998, so the
O(τ) constant ρ/(1 − ρ) is ~450 and the transient lasts ~10³ steps. `expected_order = 1` is the
asymptotic value. Likewise GTT on `tv_unconstrained_opt` reaches slope 2 only inside Thm 1(ii)'s
condition (17), ρ^C σ < 1 (κ = 2: 1.92; κ = 10: 1.61 for τ ∈ [5e-3, 2e-2]).

**Bugs found and fixed.**
1. *Causal correction used noise-free data under jitter/delay wrappers (information leak).*
   `_data_now` accepted simulate's measured sample only if `past[0]['t'] == t_k`. The wrappers
   `JitteredProblem`/`DelayedDataProblem` set `'t'` to the actual sampling time, so the causal
   correction silently fell back to a fresh, noise-free `problem.data(t)` while the model saw the
   noisy sample (MSSRE 3.3e-2 instead of ~4e-2 for running Newton with `data_noise = 1e-2`). Now
   `past[0]` is used unconditionally, exactly as simulate's vector field does for every causal call.
   Regression test: running Newton ≡ `newton_flow` + Euler (h = 1) under a jitter wrapper with
   `data_noise`; it fails on the old code.
2. *Wrong GTT direction on the benchmark's own optimisation problem.* `tv_unconstrained_opt`
   (E = ∇f) did not declare `gradient_field`, so `pc_gradient` used x − αJᵀE instead of Eq. (8).
   The problems_opt area has since added the capability; this area also recognises the family
   `"unconstrained-optimization"`. Regression test added.
3. *Test that claimed to exercise the bind-only path did not.* The spy wrapped `f` in a closure,
   which the closure search sees through, so the "bound" variant still used simulate's sample. The
   spy is now a callable object, and the bind-only path is asserted to request t_k data only.
4. Citation fixes: RG bound source, Davidenko Eq. (6), Zhang–Ma–Cai scope, Gavurin reference.

**Tests added in this review:** dual-form formula, explicit-model error dynamics, `newton_flow`
continuous-time behaviour, GTT direction and order on `tv_unconstrained_opt`, causal measured sample
under a jitter wrapper, reproduction of Simonetto's scalar example for GTT/NTT/AGT/ANT/RG, and
prediction-only ≡ forward Euler (one-step map, polynomial ζ − 1, trivially zero-stable).

**Known limitations kept (documented, not bugs).** Under `data_noise` the non-causal correction
draws its own noisy t_{k+1} sample, independent of the one simulate delivers at step k+1, as RK4's
stage data do. The causal variants evaluate the model at the corrected state x̂_k (time t_k, index
k). Problem discovery depends on simulate's local variable names until core request 1 is adopted.

## 6. Running the tests

```
cd znn-bench && python -m pytest tests/test_structured_baselines.py -q
```

The run has 60 tests and takes about 12 s. Problem sizes are n ≤ 4 and n² = 9, with at most 2000
steps per run and memory use well below 100 MB.

## 7. Integration changes (integrator, 2026-10-18)

* **`pc_gradient` / `pc_gradient_causal` step-size estimate.** The smoke matrix found this cell: `nonlinear_equations` x `gnn` x `pc_gradient`, B = 2, `info='bd2'`. The GNN prediction step left the domain of E (x1 <= 0, where ln(x1) is NaN), and `estimate_lipschitz` returned NaN. The scheme then raised `ValueError` and aborted the whole batch.
  * The power iteration now ignores non-finite trajectories.
  * If no trajectory gives a finite estimate, the correction returns NaN, so `simulate` flags divergence as it does for every other scheme. alpha stays unset and is estimated at the next finite state.
  * `L <= 0` still raises.
  * Regression test: `test_pc_gradient_outside_domain_is_divergence_not_error`, which fails on the old code.
