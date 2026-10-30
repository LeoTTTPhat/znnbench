# Spec: noise, information and arithmetic zoo (`noise_info`)

Files: `znnbench/noise/advanced.py` (registered noise models), `znnbench/wrappers.py`
(Discretizer/Problem wrappers and cost helpers), `tests/test_wrappers.py`.

## 0. Conventions

* **Where noise enters.** Every noise is added in the design-formula (error-derivative) space,
  exactly as `simulate` already does it:
  $$J(x,t)\,\dot x = \text{model}(x,t) + \nu(t)\quad\Longleftrightarrow\quad
    \dot E = -\gamma\,\Phi(E) + \nu(t)\ \text{(for ZNN)}.$$
  The noise-tolerant ZNN literature uses this form, for example
  $\dot E(t) = -\tau\Psi(E(t)) + O(t)$ in Liao, Wang, Li, Guo & He (2022), eq. (6), and
  $\dots + w(t)$ added to the CNRZNN right-hand side in Liao, Hua, Cao, Katsikis & Li (2022),
  eq. (8).
* **Signature.** `noise(t, rng, shape) -> ndarray(shape)`, with `shape = (B, m)` and `rng` the
  simulation generator. A coefficient is either a scalar, giving the papers' convention
  $[a]_{n\times n}$ where every entry is equal, or an array that broadcasts to `(B, m)`.
* **Real-valued analogues.** Liao et al. (2022, *Mathematics*) use complex coefficients such as
  $(1+i)$. ZNN-Bench problems are real-valued, so the defaults use real coefficients. The paper
  itself also uses the real noises $[8]$, $[8t]$ and $[t^2]$ in its Section 4.3.

## 1. Noise models (`znnbench/noise/advanced.py`)

| key | formula | defaults | source |
|---|---|---|---|
| `uniform` | $\nu_{ij}(t)\sim\mathcal U[\ell,u]$, i.i.d. per call | $\ell=-0.5,\ u=0.5$ | Liao et al. 2022 (Math.), Remark 3; Yan et al. 2019 use $[7.5,8.5]$ |
| `quadratic` | $W_3(t)=A t^2+B t+C$ | $a=1,\ b=c=0$ (that is, $[t^2]$) | Liao et al. 2022 (Math.), Sec. 3 and Sec. 4.3 |
| `polynomial` | $\nu(t)=\sum_{k} c_k t^k$ | $c=(0,0,1)$ | generalisation named in Liao et al. 2022 (Math.) |
| `multi_harmonic` | $\tilde o_{ij}(t)=\sum_{k=1}^{l}\Gamma_k\sin(2\pi f_k t+\varphi_k)$ | $\Gamma=(1,2,5)$, $f=(1,2,5)$ Hz, $\varphi=(1,2,4)$ | Liao, Wang, Li, Guo, He 2022 (Front.), eq. (12) |
| `combined` | $c_0 + c_1 t + a\sin(\omega t+\varphi)\ [+\ \mathcal U(-\beta,\beta)]$ | $c_0=1,\ c_1=0.1,\ a=1,\ \omega=1,\ \varphi=0,\ \beta=0$ | **ASSUMPTION**: composite built for the benchmark |
| `random_walk` | $\nu(t)=\nu_0+\sigma W(t)$ | $\sigma=1,\ \nu_0=0$ | **BENCHMARK EXTENSION**: no ZNN source found |
| `impulse` | bursts of height $\pm A$ and width $w$, starting at Poisson or periodic times | $A=10$, rate $0.5$/s, $w=0.05$ s | **BENCHMARK EXTENSION**: no ZNN source found |

These keys add to the ones already in `noise/basic.py`: `constant`, `linear`, `gaussian` and
`harmonic` (single harmonic, which is eq. (7) of Liao et al. 2022 Front. with $\omega=2\pi f$).

### 1.1 `uniform`: bounded random noise
* **Source.** B. Liao, C. Hua, X. Cao, V. N. Katsikis, S. Li, "Complex Noise-Resistant Zeroing
  Neural Network for Computing Complex Time-Dependent Lyapunov Equation", *Mathematics*
  10(15):2817, 2022. doi:10.3390/math10152817. Remark 3 uses "bounded random noises
  $[-0.5,0.5]^{2\times 2}$".
* **Second source (verified first-hand by the verifier).** J. Yan, X. Xiao, H. Li, J. Zhang, J. Yan,
  M. Liu, "Noise-Tolerant Zeroing Neural Network for Solving Non-Stationary Lyapunov Equation",
  *IEEE Access* 7:41517–41524, 2019, doi:10.1109/ACCESS.2019.2907746 (gold OA). Eq. (6) adds the
  matrix noise $T(t)$ to the right-hand side of the NTZNN design formula; Figs. 2–3 use the
  constant noise $[8]_{2\times2}$, the linear noise $[8t]_{2\times2}$ and the random noise
  $[7.5,8.5]_{2\times2}$, i.e. `uniform(low=7.5, high=8.5)` (non-zero mean).
* **Background.** The robustness of the NTZNN under constant, linear and bounded random noise
  goes back to L. Jin, Y. Zhang, S. Li, "Integration-Enhanced Zhang Neural Network for
  Real-Time-Varying Matrix Inversion in the Presence of Various Kinds of Noises", *IEEE TNNLS*
  27(12):2615–2627, 2016, doi:10.1109/TNNLS.2015.2497715. That paper is closed access (the green
  copy listed by Semantic Scholar, hdl.handle.net/10397/64478, is marked "withdrawn"), so its
  simulation noise values were **not verified**.
* **ASSUMPTION.** The papers give only the bounds of the noise, not its distribution. ZNN-Bench
  draws $\mathcal U[\ell,u]$ afresh at every vector-field evaluation, the analogue of `rand` inside
  a MATLAB ODE right-hand side.

### 1.2 `quadratic` and `polynomial`
* **Source.** Liao et al. 2022 (*Mathematics*), Sec. 3: "complex constant noise $W_1(t)=A$,
  complex linear noise $W_2(t)=At+B$ and complex quadratic noise $W_3(t)=At^2+Bt+C$". Theorem 3
  gives the CNRZNN steady-state bound $2\|A\|_F/\zeta^3$ under quadratic noise; the NTZNN diverges
  under the same noise.
* **Values used in the paper.** Example 2 uses $[(0.5+0.5i)t^2]_{2\times2}$ and
  $[(5+5i)t^2]_{2\times2}$. Section 4.3 uses $[t^2]_{2\times 2}$, together with the real constant
  noise $[8]$ and linear noise $[8t]$ taken from J. Yan et al. (2019, *IEEE Access*, ref. [30] of
  Liao et al.). The verifier confirmed $[8]$ and $[8t]$ in the Yan et al. full text (Figs. 2–3).
* **Polynomial.** Liao et al. state that other nonlinear noises can be approximated by linear or
  polynomial noise through a Taylor expansion, quadratic noise being one kind. `polynomial` takes
  ascending coefficients $c_0,\dots,c_K$ and evaluates the sum with Horner's scheme.

### 1.3 `multi_harmonic`
* **Source.** B. Liao, Y. Wang, J. Li, D. Guo, Y. He, "Harmonic Noise-Tolerant ZNN for Dynamic
  Matrix Pseudoinversion and Its Application to Robot Manipulator", *Front. Neurorobot.*
  16:928636, 2022. doi:10.3389/fnbot.2022.928636.
* **Equations.** Single harmonic, eq. (7): $o_{ij}(t)=\Gamma\sin(2\pi f t+\varphi)$. Multiple
  harmonic, eq. (12):
  $\tilde o_{ij}(t)=\sum_{k=1}^{l}\Gamma_k\sin(2\pi f_k t+\varphi_k)$, with known frequencies and
  unknown amplitudes and phases.
* **Values used in the paper (Sec. 5).**
  * $10^3\sin(0.002\pi t+1)$ and $10^3\sin(100\pi t+1)$ (the exponent was checked in the MathML
    of the article).
  * Periodic case: $\sin(2\pi t+1)+2\sin(4\pi t+2)+5\sin(10\pi t+4)$. This is the default.
  * Aperiodic case: $\sin(1)+2\sin(5t+2)+5\sin(10\pi t+3)$, stated in the paper with
    $f_1=0$, $f_2=2.5/\pi$, $f_3=5$ Hz: `amps=(1,2,5), freqs=(0, 2.5/pi, 5), phases=(1,2,3)`.
  * Robot application (eqs. after (29)): $10\sin(2\pi t+1)+10^2\sin(10\pi t+2)+10^3\sin(20\pi t+3)$
    and $10\sin(136\pi t+1)$ etc., also expressible with `multi_harmonic`.
* The same value is used for every entry, matching the paper's $[\cdot]_{2\times3}$.

### 1.4 `combined` (ASSUMPTION)
* This is a superposition of the constant, linear and harmonic components above, with an
  optional bounded-random component. The defaults are benchmark choices, not values from a paper.
* **Closest source.** Y. Yang, X. Wang, R. M. Voyles, X. Ma, "A Predefined-Time Convergent and
  Noise-Tolerant Zeroing Neural Network Model for Time Variant Quadratic Programming With
  Application to Robot Motion Planning", arXiv:2412.20477 (2024). Its robot simulation uses the
  composite $\delta(t)=\sin t+\cos t+\bar n(t)$ with $\bar n$ "a white noise signal bounded by 2"
  (Sec. 4.2 also uses $0.1\sin t$, $0.5$ and $\sin t$). Since $\sin t+\cos t=\sqrt2\sin(t+\pi/4)$,
  it is `combined(level=0, slope=0, amp=sqrt(2), omega=1, phase=pi/4, rand_bound=2)`
  (**ASSUMPTION**: $\bar n$ drawn uniform; the paper gives only the bound).
* **Rationale.** ZNN error dynamics are linear for the linear activation, so the response to the
  combined noise is the sum of the per-component responses. One run therefore tests rejection of
  all components at once.

### 1.5 `random_walk` (BENCHMARK EXTENSION)
* The value at a later call time is
  $\nu(t')=\nu(t)+\sigma\sqrt{t'-t}\,\xi$ with $\xi\sim\mathcal N(0,I)$. The value is held when
  the same $t$ is queried again, for example by RK4 stages.
* The walk restarts at $\nu_0$ when $t$ decreases (a new run) or when the shape changes.
* No ZNN paper using Wiener or random-walk noise was found (searches: "zeroing neural network"
  with "random walk", "Brownian", "Wiener", "Itô"). It is included as an unbounded,
  non-differentiable stress test. Integral-type noise-tolerant ZNNs reject bias and ramp noise;
  a random walk has neither a bounded value nor a bounded derivative.

### 1.6 `impulse` (BENCHMARK EXTENSION)
* Rectangular bursts of height $A$ and width $w$. With `random_sign`, each burst and entry gets a
  sign $\pm1$. Burst start times are either periodic, $\phi+jP$, or Poisson with rate $\lambda$,
  drawn lazily from the simulation `rng`. Overlapping bursts add.
* No ZNN paper with impulse or burst noise was found (searches: "impulse noise", "burst noise",
  "pulse noise" with "zeroing neural network").
* **Sampling caveat.** A causal discretization sees a burst only if some $t_k$ falls inside it,
  so choose $w\ge\tau$.

### 1.7 Not implemented
Two noise classes appear in the theory of several papers but are not implemented:

* **bounded vanishing noise**, $|y_i(t)|\le\delta|\varepsilon_i(t)|$: "time-varying bounded
  vanishing noise" $y_i(t)=0.4\,\varepsilon_i(t)$ in R. Zhao, M. Li, T. Liu, S.-X. Miao, *AIMS Math.*
  10(6):13974–13995, 2025, doi:10.3934/math.2025628; "dynamic bounded disappearing noise (DBDN)",
  $|n_{ij}(t)|\le\delta|e_{ij}(t)|$, in J. Jin, L. Zhao, L. Chen, W. Chen, *Front. Neurorobot.*
  16:1065256, 2022, doi:10.3389/fnbot.2022.1065256, Theorem 2;
* **bounded non-vanishing noise**: "time-varying bounded nonvanishing noise" $y_i(t)=0.6$ (Zhao et
  al.) and "dynamic bounded non-disappearing noise (DBNDN)", $|n_{ij}(t)|\le\delta$ (Jin et al.,
  Theorem 3; simulations use the constant $n(t)=0.5$).

The vanishing class depends on the error E, which `noise(t, rng, shape)` cannot see; see
CORE_CHANGE_REQUESTS § noise_info, item 1. Constant instances of the non-vanishing class
($y_i=0.6$, $n=0.5$) are already covered by `constant`.

## 2. Wrappers (`znnbench/wrappers.py`)

Every wrapper keeps the core API:

* a `Discretizer` wrapper is itself a `Discretizer`;
* a `Problem` wrapper is itself a `Problem`. It delegates every method and unknown attribute to
  `inner`.

Problem wrappers that change the data the model sees keep the inner problem's `d['t']` (the time
the data actually belong to) and set `d['t_nominal']` = $t_k$. They override `residual`, so the
residual recorded by `simulate` always uses the **exact** data at $t_k$: `ProblemWrapper.exact_data(t)`
walks down the wrapper chain to the unwrapped problem and calls its `data(t)`, so the diagnostic
never calls the `data` of a stateful wrapper (e.g. the sample history of
`EstimatedDerivativeProblem`) and never sees jitter, delay or measurement noise. The model-side
`E`, `solve_J` and `JT` use the changed data. This is a work-around for CORE_CHANGE_REQUESTS
§ noise_info, item 2 (the method name matches the requested `exact_data` hook; the workaround text
in that file still describes the pre-verification `inner.data(d["t_nominal"])` call).

**Chaining.** Wrappers compose. `EstimatedDerivativeProblem(DelayedDataProblem(P))` and
`DelayedDataProblem(EstimatedDerivativeProblem(P))` give the same run after the initial hold.
`EstimatedDerivativeProblem(JitteredProblem(P))` is the un-timestamped regime (the estimator
believes the samples are $\tau$ apart); `JitteredProblem(EstimatedDerivativeProblem(P))` gives the
estimator the true (time-stamped) sampling instants.

### 2.1 `FixedPoint` and `QuantizedDiscretizer(disc, bits, frac_bits | scale, quantize_derivative=False, ...)`
* **Format.** Signed two's-complement $Q(b-1-f).f$: grid step $q=2^{-f}$ and range
  $[-2^{b-1}q,\ (2^{b-1}-1)q]$. Values outside the range saturate:
  $$\mathcal Q(v)=q\cdot\mathrm{clip}\big(\mathrm{round}(v/q),\,-2^{b-1},\,2^{b-1}-1\big).$$
* **Rounding.** `'nearest'` rounds half to even (IEEE/convergent rounding). `'floor'` truncates,
  that is, it drops LSBs in two's complement.
* **From `scale`.** Given the largest magnitude that must fit, the format uses
  $I=\lfloor\log_2 \text{scale}\rfloor+1$ integer bits (the smallest $I$ with $2^I>\text{scale}$,
  MATLAB's best-precision rule), so $f=b-1-I$ and $\pm\text{scale}$ are representable. (Fixed in
  verification: the original $\lceil\log_2\text{scale}\rceil$ saturated $+\text{scale}$ by one LSB
  whenever scale was a power of two.)
* **Limits.** $b\le 53$, so every grid value is exact in float64. NaN propagates, so divergence is
  still flagged; $\pm\infty$ saturates.
* **Step.** $x_{k+1}=\mathcal Q(\mathrm{step}(\mathcal Q(x_k)))$, and $z$ likewise unless
  `quantize_z=False`.
* **Derivative quantization.** With `quantize_derivative`, the inner discretizer receives
  $\mathcal Q_d(f(t,x,z))$, using its own format (`deriv_bits`, `deriv_frac_bits`,
  `deriv_scale`). All other arithmetic is float64. This emulates an embedded realisation that keeps
  the neural state, and optionally the computed rate, in b-bit registers.
* **Expected effect** (heuristic, **ASSUMPTION**: the uniform white rounding-noise model of
  B. Widrow and I. Kollár, *Quantization Noise*, Cambridge Univ. Press, 2008,
  doi:10.1017/CBO9780511754661).
  * For linear ZNN-Euler with $h=\gamma\tau$, $E_{k+1}=(1-h)E_k+J\delta_k$ with
    $\delta_k\sim\mathcal U(-q/2,q/2)$. The RMS residual floor is therefore
    $\|J\|_F\,q/\sqrt{12\,h(2-h)}$, and it does **not** shrink as $\tau\to0$ at fixed $q$.
  * This white-noise model holds when the per-step increment $\tau\|\dot x\|$ is large against $q$
    (moderate $h$). Verifier's measurements (`linear_system`, n = 4, $\kappa=10$, B = 4, time-mean
    of $\|E\|$ over the last quarter of T = 4 s, divided by the RMS prediction): 0.99–1.24 at
    $h=0.5$, $\tau=10^{-3}$, $f\in\{8,10,12\}$; 1.2–1.4 at $h=0.5$, $\tau=10^{-4}$ (smaller increments).
    For small $h$ the state stalls between grid points (dead zone) and the floor grows towards
    $O(\|J\|\,q/(2h))$: the ratio is 4–8 at $h=0.05$, $\tau=10^{-3}$.
  * Saturation, when $|x^*|$ exceeds the range, destroys convergence without triggering the
    divergence flag.
  * `expected_order` is set to `None`.

### 2.2 `JitteredProblem(problem, jitter_std=None, jitter_bound=None, seed=0, clip_nonnegative=True)`
* **Data.** The data delivered at nominal $t_k$ are $d(t_k+\delta_k)$, and the model (`ctx.t`,
  gain schedules) keeps $t_k$. The jitter $\delta_k$ is i.i.d. and distributed as follows:
  * $\mathcal N(0,\sigma^2)$ when only `jitter_std` is given;
  * $\mathcal U[-b,b]$ when only `jitter_bound` is given;
  * $\mathcal N(0,\sigma^2)$ truncated to $[-b,b]$ by rejection when both are given.
* **Reproducibility.** $\delta_k$ is a deterministic function of (seed, $t_k$): a hash-seeded
  PCG64 generator. It is reproducible and does not depend on the order of queries. Times are
  canonicalised to $10^{-11}$ s, so RK4's round-off duplicates of grid instants get the same
  jitter.
* **`dEdt`.** `dEdt(x, t_k)` returns $\partial_t E(x,\cdot)$ at the actual sampling instant
  $t_k+\delta_k$ (**ASSUMPTION**: value and rate are sampled by the same jittered clock). With
  `info='bd1'`/`'bd2'`, `simulate` differences jittered samples as if they were $\tau$ apart. This
  is the realistic failure mode: an extra error of about $\frac{\delta_k-\delta_{k-1}}{\tau}\,
  \partial_t E$.
* **Clipping.** With `clip_nonnegative`, the sampling time is $\max(t_k+\delta_k,0)$.

### 2.3 `EstimatedDerivativeProblem(problem, method, window, ...)`: causal dE/dt estimation
**Exactly how it works.**

1. Every call `data(t)` with $t$ later than the last stored instant stores the delivered sample
   $d_j=d(t_j)$. If `meas_std > 0`, the sample has i.i.d. $\mathcal N(0,\text{meas\_std}^2)$
   added to every array except `'t'`, hash-seeded per $t_j$. The same noisy sample is returned to
   the model, so `E`, `solve_J`, `JT` and the estimator agree.
2. A repeated call at the same instant returns the stored sample. A call at an instant earlier
   than the latest stored one discards the stored samples later than it (so a new run starting
   again at $t=0$ clears the history, and an out-of-order arrival inside a `JitteredProblem` drops
   the later samples). Only the last `window` samples are kept. Time stamps (`'t'`,
   `'t_nominal'`) and non-floating entries are never perturbed by `meas_std`.
3. `dEdt(x, t)` uses only stored samples with $t_j\le t$, which makes the estimate **causal**,
   and returns
   $$\widehat{\partial_t E}(x,t)=\sum_{j} w_j\,E(x,d_j),$$
   that is, it differences or filters the error sequence evaluated at the **current, fixed**
   state $x$. It therefore estimates the partial time derivative, never the total derivative,
   and never calls `inner.dEdt`; the tests check this with a proxy whose `dEdt` raises.
4. With fewer than 2 samples the estimate is $0$, like `info='bd1'` at $k=0$.
5. The weights depend only on the relative times $s_j=t_j-t\le0$. They are computed once per
   spacing pattern and cached, so each step costs `window` evaluations of E.
6. Use the wrapper with `simulate(..., info='analytic', data_noise=0)`. The estimate arrives
   through the `dEdt` slot, and measurement noise comes from `meas_std` (CORE_CHANGE_REQUESTS
   item 4).

**Weights by method.**

* **`savgol`** (default window 5, `order` 2). This is the causal Savitzky–Golay differentiator:
  A. Savitzky and M. J. E. Golay, *Anal. Chem.* 36(8):1627–1639, 1964, doi:10.1021/ac60214a047.
  It fits $\sum_{i=0}^{p} a_i s^i$ to $(s_j,E(x,d_j))$ by least squares and returns $a_1$:
  $$w=\big[(V^\top V)^{-1}V^\top\big]_{1,:},\qquad V_{ji}=s_j^{\,i}.$$
  * It is exact for polynomials of degree $\le p$, and the error is $O(\tau^p)$.
  * With $p=L-1$ it is the backward-difference formula: $L=2$ gives BD1 and $L=3$ gives BD2,
    identical to `simulate`'s `bd1`/`bd2`.
  * For uniform spacing it equals `scipy.signal.savgol_coeffs(L, p, deriv=1, pos=L-1)`.
* **`alpha_beta`** (default $\alpha=0.5$, window 20). The default
  $\beta=\alpha^2/(2-\alpha)$ is the Benedict–Bordner relation: T. R. Benedict and G. W. Bordner,
  *IRE Trans. Automatic Control* 7(4):27–32, 1962, doi:10.1109/TAC.1962.1105477. For each
  sample:
  $$\hat e^-_j=\hat e_{j-1}+\Delta_j\hat v_{j-1},\quad r_j=e_j-\hat e^-_j,\quad
    \hat e_j=\hat e^-_j+\alpha r_j,\quad \hat v_j=\hat v_{j-1}+\tfrac{\beta}{\Delta_j}r_j,$$
  started by two-point differencing at the second sample. The velocity is held up to $t$.
  P. R. Kalata, *IEEE Trans. AES* 20(2):174–182, 1984, doi:10.1109/TAES.1984.310438.
* **`kalman`** (default window 20). This is a constant-velocity Kalman filter with the
  discrete white-noise acceleration (DWNA) model of Y. Bar-Shalom, X.-R. Li and T. Kirubarajan,
  *Estimation with Applications to Tracking and Navigation*, Wiley, 2001 (online 2002),
  doi:10.1002/0471221279:
  $$F=\begin{bmatrix}1&\Delta\\0&1\end{bmatrix},\ Q=q\,GG^\top,\ G=\begin{bmatrix}\Delta^2/2\\ \Delta\end{bmatrix},\ H=[1\ 0],\ R=r.$$
  * **Initialisation.** Two-point differencing: $x_1=[e_1,(e_1-e_0)/\Delta]$ and
    $P_1=\begin{bmatrix}r&r/\Delta\\ r/\Delta&2r/\Delta^2\end{bmatrix}$.
  * **Tuning.** The default parameter is Kalata's tracking index
    $\lambda=\sigma_w\Delta^2/\sigma_v=1$, which sets $q=(\lambda/\Delta^2)^2$, $r=1$, with
    $\Delta$ the median spacing. This tuning does not depend on the step size. The steady-state
    gains are
    $\alpha=-\frac{\lambda^2+8\lambda-(\lambda+4)\sqrt{\lambda^2+8\lambda}}{8}$ and
    $\beta=\frac{\lambda^2+4\lambda-\lambda\sqrt{\lambda^2+8\lambda}}{4}$, which give 0.75 and
    0.5 for $\lambda=1$.
  * With `tracking_index=None`, the physical $q$ (acceleration variance) and $r$ are used.
    Fixed $(q,r)$ does **not** converge as $\tau\to0$: the filter's time constant
    $\sim (r/q)^{1/4}$ is fixed in seconds.
  * The section numbers inside Bar-Shalom et al. were not verified, so only chapters are cited
    in the docstrings.
* **Lag of the two filters.** Both are linear filters with data-independent gains, so the
  window estimate is an FIR filter. For a signal with constant second derivative $a$, the
  steady-state estimate is
  $$\hat v-\dot e=-a\,\tau\,\frac{\alpha-\beta/2}{\beta},$$
  first order in $\tau$, while Savitzky–Golay of degree $\ge2$ is exact. This identity is
  checked numerically in the tests.

### 2.4 `DelayedDataProblem(problem, delay | (delay_steps, tau), hold_initial=True)`
* At nominal $t$ the model receives $d(t-D)$, and `dEdt` returns $\partial_tE$ at $t-D$.
* **ASSUMPTION.** Before the first sample arrives ($t<D$), $d(0)$ is held, and the derivative of
  the held stream is $0$.
* The recorded residual uses exact data at $t$.

### 2.5 Cost helpers: `CountingProblem`, `dense_flops`, `profile_step_cost`
* **`CountingProblem`.** A pass-through wrapper that counts calls and accumulates wall time for
  `data`, `E`, `dEdt`, `solve_J` and `JT`. `residual` is counted separately because it is a
  diagnostic, not part of the algorithm.
* **`profile_step_cost`.** Runs a short simulation and returns calls per step, wall time per
  step (total and per method), and a rough dense FLOP estimate.
* **`dense_flops`** (**ASSUMPTION**: dense, unstructured J of size $m\times n_x$):
  * $E$, $\partial_tE$ and $J^\top v$ cost $2mn_xB$;
  * `solve_J` costs $\tfrac23 n_x^3$ when square, or $2mn_x^2$ for least squares, plus $2mn_xB$.
* Kronecker or other problem structure is ignored. Wall-clock includes Python overhead, so use
  it only for relative comparisons.

## 3. Tests (`tests/test_wrappers.py`, 41 tests, about 5 s)

* **Noises.**
  * All keys are registered. Shapes are correct, and a ZNN-Euler smoke test with each noise
    stays finite.
  * `uniform`: bounds, mean and variance $1/12$.
  * `quadratic` and `polynomial`: closed form, per-entry coefficients, and the default $[t^2]$.
  * `multi_harmonic`: Liao et al.'s periodic example and the $10^3\sin(100\pi t+1)$ case.
  * `combined`: superposition and the random-component bound.
  * `random_walk`: $\mathrm{Var}\,W(t)=\sigma^2t$ within 8%, independent increments, holding,
    and reset.
  * `impulse`: periodic on/off pattern, Poisson arrival count within $6\sqrt{\lambda T}$, and
    $\pm A$ amplitudes.
  * Source reproductions (added in verification): Liao et al. 2022 (Front.) aperiodic and
    $10^3\sin(0.002\pi t+1)$ cases, Yan et al. 2019 $[7.5,8.5]$ random and $[8t]$ linear noise,
    Liao et al. 2022 (Math.) $[t^2]$, Yang et al. 2024 $\sin t+\cos t+\bar n$, $|\bar n|\le2$.
* **Quantization.**
  * Grid property: outputs are integer multiples of $q$, $|\mathcal Q(v)-v|\le q/2$ inside the
    range, saturation at both ends, idempotence, NaN and inf handling, `floor` truncation, and
    `scale`→`frac_bits`.
  * The state handed to the model, and the derivative the discretizer receives, are on their
    grids.
  * $z$ is quantized only when `quantize_z` is set.
  * ZNN-Euler: the residual floor decreases with `frac_bits` and reaches the unquantized result
    at 20 fractional bits. Quantized RK4 is floored above $10^{-3}$. Saturation prevents
    convergence. A Taylor–Zhang run with 24 fractional bits matches the unquantized one within 5%.
  * Rounding-noise model (added in verification): at $h=0.5$ the measured floor is within
    $\pm35\%$ of $\|J\|_F q/\sqrt{12h(2-h)}$; at $h=0.05$ it is more than twice that value (dead
    zone). `scale` makes $\pm$scale representable. Unknown attributes are delegated to the
    wrapped discretizer.
* **Jitter.**
  * API: shapes, `solve_J`, `JT`, `x_star`, attribute delegation, exact recorded residual, and
    `dEdt` at the jittered time.
  * Statistics over 6000 instants: mean within $4\sigma/\sqrt n$, standard deviation within 5%,
    lag-1 correlation below 0.06, uniform variance $b^2/3$, and truncation.
  * Reproducibility and order independence.
  * Zero jitter reproduces the unwrapped run; $\sigma=10^{-4}$ is harmless and
    $\sigma=10^{-2}$ hurts.
  * `bd1` is worse than `analytic` under jitter, and RK4 runs through the wrapper.
* **Estimators.**
  * Savitzky–Golay weights match `scipy` and reduce to BD1 and BD2.
  * The filters are exact on linear signals.
  * The alpha-beta and Kalman lag formulas hold, and the Kalman steady state matches the
    Kalata-gain alpha-beta filter at $\lambda=0.3$.
  * (Added in verification) The FIR weights reproduce an independently coded textbook alpha-beta
    recursion and DWNA Kalman filter on noisy data over a random non-uniform grid (rel. $10^{-10}$),
    and Savitzky–Golay is exact for polynomials of degree $\le p$ on that grid.
  * **Convergence to the analytic dE/dt** on noiseless `linear_system` data. The relative-error
    decay from $\tau=10^{-2}$ to $10^{-3}$ is $10^{2\pm0.25}$ for `savgol` (order 2) and
    $10^{1\pm0.25}$ for `alpha_beta` and `kalman` (order 1). The exact `dEdt` is never called.
  * Causality: the estimate equals the hand-computed weighted error differences over samples
    $t_j\le t$ only. A new run resets the history; an out-of-order sample drops only the later
    stored samples.
  * Noise smoothing: with `meas_std` $=10^{-3}$ and $\tau=10^{-3}$, Savitzky–Golay (window 21),
    alpha-beta and Kalman are each more than 3× better than BD1. Measurement noise does not leak
    into the recorded residual.
  * ZNN-Euler smoke test at $h=0.3$ for each method: the MSSRE drops more than 4× from
    $\tau=10^{-2}$ to $2\times10^{-3}$, and RK4 stage calls work.
* **Delay and cost.**
  * Delay: shifted data, initial hold, zero derivative while held, exact residual, a worse but
    finite MSSRE, and zero delay acting as a no-op.
  * Cost: per-step call counts (Euler 1/1/1, RK4 4 `solve_J`, GNN uses `JT`), positive FLOPs,
    and a pass-through `CountingProblem` that does not change results.
* **Wrapper chains (added in verification).** `EstimatedDerivativeProblem(DelayedDataProblem(P),
  meas_std>0)`: exact recorded residual and unperturbed time stamps; `JitteredProblem(
  DelayedDataProblem(P))`: `d['t']` is the time the data belong to; delay inside vs outside the
  estimator gives the same MSSRE (rtol $10^{-8}$) and the inner estimator keeps a full window;
  time-stamped jitter around the estimator runs; `exact_data` makes no counted `data` call.

## 4. Unverified or second-hand items (honesty log)
1. The noise values in Jin, Zhang & Li (2016, TNNLS) were not read from the original (closed
   access; the listed green copy is withdrawn). Yan et al. (2019, IEEE Access) **was** read
   first-hand during verification (gold OA PDF): $[8]$, $[8t]$ and $[7.5,8.5]$ are confirmed.
2. `random_walk`, `impulse` and the `combined` defaults are benchmark constructions. No ZNN paper
   was found that uses them (implementer's searches; the verifier could not re-run literature
   searches because the session's web-search budget was exhausted, so this negative claim was not
   independently re-checked).
3. Section numbers inside Bar-Shalom et al. (2001) were not checked.
4. Jitter, delay and estimated-derivative information are benchmark information regimes, not
   models from a specific ZNN paper. No ZNN source is claimed for them.
5. `meas_std` adds real Gaussian noise. For complex-valued problems encoded as complex arrays,
   only the real part is perturbed.

## 5. Verification (adversarial review)

Each item was checked independently of the implementer's report, against the source named.

### 5.1 Equations and values against the original papers
| item | source read | result |
|---|---|---|
| noise enters the design formula, $\dot E=-\tau\Psi(E)+O(t)$ | Liao, Wang, Li, Guo, He 2022, PMC9235923 full-text XML (MathML), eq. (6) | correct |
| single harmonic $o_{ij}=\Gamma\sin(\omega t+\varphi)=\Gamma\sin(2\pi f t+\varphi)$ | same, eq. (7) | correct |
| multiple harmonic $\tilde o_{ij}=\sum_k\Gamma_k\sin(2\pi f_k t+\varphi_k)$ | same, eq. (12) | correct |
| $10^3\sin(0.002\pi t+1)$, $10^3\sin(100\pi t+1)$, periodic default, aperiodic case | same, Sec. 5 display formulas | correct (defaults `amps=(1,2,5)`, `freqs=(1,2,5)`, `phases=(1,2,4)`) |
| $W_1=A$, $W_2=At+B$, $W_3=At^2+Bt+C$; noise $+w(t)$ in eq. (8) | Liao, Hua, Cao, Katsikis, Li 2022, MDPI PDF (mdpi-res.com) | correct |
| Remark 3 bounded random noise $[-0.5,0.5]^{2\times2}$ | same, Sec. 4.1 | correct (`uniform` default) |
| Example 2 $[(0.5+0.5i)t^2]$, $[(5+5i)t^2]$; Sec. 4.3 $[8]$, $[8t]$, $[t^2]$; Theorem 3 bound $2\|A\|_F/\zeta^3$ | same | correct |
| Yan et al. eq. (6) noise $T(t)$ on the RHS; $[8]$, $[8t]$, $[7.5,8.5]$ | Yan et al. 2019, IEEE Access gold-OA PDF | correct; $[7.5,8.5]$ added to the docs |
| composite $\sin t+\cos t+\bar n$, $|\bar n|\le2$ | Yang et al., arXiv:2412.20477 v1 PDF | correct citation; parameter mapping added |
| DBDN/DBNDN and $n(t)=0.5$ | Jin, Zhao, Chen, Chen 2022, PMC9705728 | correct; terminology fixed ("disappearing") |
| $y_i=0.4\varepsilon_i$, $y_i=0.6$ | Zhao, Li, Liu, Miao 2025, AIMS Math. OA PDF | correct; terminology fixed ("time-varying bounded (non)vanishing") |
| bibliographic data of Benedict–Bordner 1962, Kalata 1984, Savitzky–Golay 1964, Bar-Shalom et al. 2001/2002, Widrow–Kollár 2008, Wang–Hua–Khan 2025, Jin et al. 2022, Zhao et al. 2025 | Crossref API | all match |

### 5.2 Independent numerical checks (scratch scripts, then folded into the tests)
* Kalata steady-state gains: Riccati iteration of the DWNA filter with $q=(\lambda/\Delta^2)^2$,
  $r=1$ reproduces $\alpha,\beta$ from the closed forms for $\lambda\in\{0.1,1,5\}$ to $10^{-12}$
  ($\lambda=1\Rightarrow\alpha=0.75,\ \beta=0.5$); Benedict–Bordner $\beta=\alpha^2/(2-\alpha)$.
* FIR weights vs. independently coded recursions (alpha-beta, Kalman) on noisy data over a random
  non-uniform grid: agreement to $10^{-14}$; Savitzky–Golay exact for degree $\le p$ on that grid.
* Steady-state lag $-a\tau(\alpha-\beta/2)/\beta$: matches the standard alpha-beta bias formula.
* Quantization floor vs. the rounding-noise model: see § 2.1 (white-noise model holds at moderate
  $h$, dead zone at small $h$; the spec previously stated the heuristic without that limit).
* Every wrapper (and a three-level chain) runs under ZNN-Euler on all 13 registered problems.

### 5.3 Bugs found and fixed
1. **Residual diagnostic corrupted stateful inner wrappers.** `ProblemWrapper.residual` called
   `self.inner.data(t_nominal)`; with `DelayedDataProblem(EstimatedDerivativeProblem(P))` this
   inserted exact-time samples into the estimator's history, which was then reset at every step
   (MSSRE 0.141 instead of 0.106 at $\tau=2\times10^{-3}$, history length 2). Fixed with
   `ProblemWrapper.exact_data`, which bypasses every wrapper's `data`.
2. **Measurement noise perturbed `t_nominal`.** `EstimatedDerivativeProblem(meas_std>0)` around a
   jittered/delayed problem added Gaussian noise to `d['t_nominal']`, so the recorded residual was
   evaluated at a wrong time (e.g. 0.4858 instead of 0.5 with `meas_std=1e-2`). Time stamps and
   non-floating entries are now left exact.
3. **`d['t']` overwritten in chains.** `JitteredProblem`/`DelayedDataProblem` overwrote the inner
   `d['t']`; `JitteredProblem(DelayedDataProblem(P))` reported $t+\delta$ instead of
   $t+\delta-D$. The inner time stamp is now kept.
4. **`FixedPoint(scale=...)` off by one LSB at powers of two** (see § 2.1).
5. **Out-of-order samples reset the whole estimator history** (inside a `JitteredProblem` with
   $|\delta|>\tau/2$). Now only the later stored samples are discarded; a new run still resets.
6. `QuantizedDiscretizer` now delegates unknown attributes (e.g. `bind`) to the wrapped scheme.

### 5.4 API contract (docs/DESIGN.md)
* Noises return arrays of the requested `(B, m)` shape and are registered under unique keys;
  stateful noises restart when time goes back. Discretizer wrapper keeps `causal`/`implicit` of
  the wrapped scheme and sets `expected_order=None`. Problem wrappers keep `n_x`, `m`, `T`,
  `family`, `capabilities()` and the `x_star` of the wrapped problem. No core file was edited.

### 5.5 Not verified
* Jin, Zhang & Li 2016 simulation values (closed access).
* Section numbers inside Bar-Shalom et al. (only chapters cited).
* The negative literature claims for `random_walk` and `impulse` (see § 4, item 2).

