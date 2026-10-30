"""Information / arithmetic wrappers for ZNN-Bench.

These wrappers change *what the model gets to see* or *how the arithmetic is
done* without touching the core API: every wrapper is itself a ``Problem`` or
a ``Discretizer`` and can be passed to ``simulate`` unchanged.

Discretizer wrappers
--------------------
``QuantizedDiscretizer``   b-bit signed fixed-point state (round to the
                           2^-frac_bits grid with saturation after every step),
                           optionally also the vector field (derivative).

Problem wrappers
----------------
``JitteredProblem``        data sampled at t_k + delta_k (delta_k i.i.d.,
                           seeded) while the model believes t_k.
``EstimatedDerivativeProblem``
                           dE/dt replaced by a causal estimate built only from
                           past sampled data, by filtering the sequence
                           E(x_k, d_j), j <= k, at the *current fixed* x_k
                           ('savgol', 'alpha_beta', 'kalman').  Use it with
                           ``simulate(..., info='analytic')``: the estimate is
                           delivered through the ``dEdt`` slot.
``DelayedDataProblem``     data arrive with a fixed delay (d samples).
``CountingProblem`` + ``profile_step_cost``
                           call counts, wall-clock and rough dense FLOP
                           estimates per step.

Problem wrappers that alter the data the model sees override ``residual`` so
that the residual *recorded* by ``simulate`` is always evaluated with the exact
data at the nominal sampling instant t_k (key ``'t_nominal'``, or ``'t'`` when
absent), obtained through ``ProblemWrapper.exact_data`` which bypasses the
``data`` method of every wrapper in a chain (so stateful wrappers are never
touched by the diagnostic).  The model-side quantities (``E``, ``solve_J``,
``JT``) use the altered data.  Wrappers can be chained.  See
docs/spec/noise_info.md for equations, sources, assumptions and the
verification log.
"""
from __future__ import annotations

import time
from collections import OrderedDict, deque

import numpy as np

from .core import Discretizer, Problem


# ===========================================================================
# fixed-point arithmetic
# ===========================================================================
class FixedPoint:
    """Signed b-bit two's-complement fixed-point format Q(b - 1 - f).f.

    Values are integer multiples of q = 2^-f in [-2^(b-1) q, (2^(b-1) - 1) q];
    out-of-range values saturate.  Either ``frac_bits`` (f) or ``scale`` (the
    largest magnitude that must be representable) is given; with ``scale`` the
    integer part gets I = floor(log2(scale)) + 1 bits (the smallest I with
    2^I > scale, as MATLAB's best-precision fraction length), so
    f = b - 1 - I and both +scale and -scale lie inside the range.

    rounding: 'nearest' (round half to even, IEEE/convergent rounding) or
    'floor' (truncation toward -inf, i.e. dropping LSBs in two's complement).
    NaN is propagated (so that ``simulate`` still flags divergence); +-inf
    saturates.
    """

    def __init__(self, bits=16, frac_bits=None, scale=None, rounding="nearest"):
        bits = int(bits)
        if not 2 <= bits <= 53:
            raise ValueError("bits must be in [2, 53] (exact in float64)")
        if frac_bits is None:
            if scale is None:
                raise ValueError("give frac_bits or scale")
            if scale <= 0:
                raise ValueError("scale must be positive")
            # smallest I with 2^I > scale, so that +scale itself fits in
            # [-2^I, 2^I - q] (I = ceil(log2 scale) would saturate +scale by
            # one LSB whenever scale is a power of two)
            frac_bits = bits - 1 - (int(np.floor(np.log2(float(scale)))) + 1)
        if rounding not in ("nearest", "floor"):
            raise ValueError("rounding must be 'nearest' or 'floor'")
        self.bits, self.frac_bits, self.rounding = bits, int(frac_bits), rounding
        self.q = 2.0 ** (-self.frac_bits)
        self.n_min, self.n_max = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
        self.lo, self.hi = self.n_min * self.q, self.n_max * self.q

    def __call__(self, v):
        v = np.asarray(v, dtype=float)
        n = v / self.q
        n = np.rint(n) if self.rounding == "nearest" else np.floor(n)
        return np.clip(n, self.n_min, self.n_max) * self.q

    def describe(self):
        return {"bits": self.bits, "frac_bits": self.frac_bits, "q": self.q,
                "range": [self.lo, self.hi], "rounding": self.rounding}


class QuantizedDiscretizer(Discretizer):
    """Wrap any discretizer so that the state lives on a fixed-point grid.

    After every step  x_{k+1} = Q(step(x_k))  (and z likewise if
    ``quantize_z``); the initial state is quantized on entry.  With
    ``quantize_derivative`` the model output is also quantized before the
    discretizer combines it,  f_Q(t, x, z) = Q_d(f(t, x, z)),  with its own
    format (``deriv_bits`` / ``deriv_frac_bits`` / ``deriv_scale``, defaulting to
    the state format).  This emulates an embedded implementation that stores
    the neural state (and optionally the computed rate) in b-bit registers,
    while the remaining arithmetic is exact (float64).

    Expected effect for linear ZNN-Euler with h = gamma tau: the steady-state
    residual is floored by rounding noise instead of O(tau^2).  When the
    per-step increment tau ||x_dot|| is large against q (moderate h) the
    rounding error is white and the RMS floor is ||J||_F q / sqrt(12 h (2-h))
    (Widrow-Kollar model; checked within ~35% in the tests); for small h the
    state stalls between grid points (dead zone) and the floor grows towards
    O(||J|| q / (2h)).  Saturation (|x| > range) destroys convergence.
    ``expected_order`` is therefore None.  Unknown attributes are delegated
    to the wrapped discretizer (e.g. ``bind`` of prediction-correction).
    """

    def __init__(self, disc, bits=16, frac_bits=None, scale=None, quantize_derivative=False,
                 deriv_bits=None, deriv_frac_bits=None, deriv_scale=None, quantize_z=True,
                 rounding="nearest"):
        self.disc = disc
        self.fmt = FixedPoint(bits, frac_bits=frac_bits, scale=scale, rounding=rounding)
        self.quantize_derivative = bool(quantize_derivative)
        if self.quantize_derivative:
            if deriv_frac_bits is None and deriv_scale is None:
                deriv_frac_bits = self.fmt.frac_bits
            self.dfmt = FixedPoint(deriv_bits or bits, frac_bits=deriv_frac_bits,
                                   scale=deriv_scale, rounding=rounding)
        else:
            self.dfmt = None
        self.quantize_z = bool(quantize_z)
        self.name = f"{disc.name}+q{self.fmt.bits}.{self.fmt.frac_bits}" + ("d" if self.dfmt else "")
        self.causal = disc.causal
        self.implicit = getattr(disc, "implicit", False)
        self.expected_order = None

    def __getattr__(self, item):
        if item.startswith("__") or item == "disc":
            raise AttributeError(item)
        disc = self.__dict__.get("disc")
        if disc is None:
            raise AttributeError(item)
        return getattr(disc, item)

    def reset(self):
        self.disc.reset()

    def step(self, k, t, tau, x, z, f):
        Q = self.fmt
        x = Q(x)
        if self.quantize_z and z.shape[1]:
            z = Q(z)
        ff = f
        if self.dfmt is not None:
            D = self.dfmt

            def ff(tt, xx, zz, kk, implicit):
                xd, zd = f(tt, xx, zz, kk, implicit)
                return D(xd), (D(zd) if zd.shape[1] else zd)
        x_next, z_next = self.disc.step(k, t, tau, x, z, ff)
        x_next = Q(x_next)
        if self.quantize_z and z_next.shape[1]:
            z_next = Q(z_next)
        return x_next, z_next

    def describe(self):
        return {**self.disc.describe(), "name": self.name, "expected_order": None,
                "wrapped": self.disc.name, "state_format": self.fmt.describe(),
                "deriv_format": self.dfmt.describe() if self.dfmt else None,
                "quantize_z": self.quantize_z}


# ===========================================================================
# problem wrappers
# ===========================================================================
class ProblemWrapper(Problem):
    """Delegating base class.  Unknown attributes are looked up on ``inner``."""

    exact_residual = False      # recompute the recorded residual from exact data

    def __init__(self, problem: Problem, tag: str = "wrap"):
        self.inner = problem
        self.n_x, self.m, self.T = problem.n_x, problem.m, problem.T
        self.family = problem.family
        self.name = f"{problem.name}+{tag}"

    def __getattr__(self, item):
        if item.startswith("__") or item == "inner":
            raise AttributeError(item)
        inner = self.__dict__.get("inner")
        if inner is None:
            raise AttributeError(item)
        return getattr(inner, item)

    # --- delegation --------------------------------------------------------
    def data(self, t):
        return self.inner.data(t)

    def E(self, x, d):
        return self.inner.E(x, d)

    def dEdt(self, x, t):
        return self.inner.dEdt(x, t)

    def solve_J(self, x, d, r):
        return self.inner.solve_J(x, d, r)

    def JT(self, x, d, v):
        return self.inner.JT(x, d, v)

    def x_star(self, t):
        return self.inner.x_star(t)

    def x0(self, rng, B):
        return self.inner.x0(rng, B)

    def capabilities(self):
        return self.inner.capabilities()

    # --- diagnostics ---------------------------------------------------------
    @staticmethod
    def nominal_time(d):
        return float(d["t_nominal"]) if "t_nominal" in d else float(d["t"])

    def exact_data(self, t):
        """Exact data of the innermost (unwrapped) problem at time t.  It
        bypasses the ``data`` method of every wrapper in the chain, so the
        diagnostic residual never touches the state of a stateful wrapper
        (e.g. the sample history of ``EstimatedDerivativeProblem``) and never
        sees jitter, delay or measurement noise.  The name matches
        CORE_CHANGE_REQUESTS § noise_info, item 2."""
        inner = self.inner
        return inner.exact_data(t) if isinstance(inner, ProblemWrapper) else inner.data(t)

    def residual(self, x, d):
        if self.exact_residual:
            return self.inner.residual(x, self.exact_data(self.nominal_time(d)))
        return self.inner.residual(x, d)

    def describe(self):
        return {**self.inner.describe(), "name": self.name, "wrapped": self.inner.name}


_T_DIGITS = 11      # instants closer than ~1e-11 s are the same instant


def _tkey(t):
    """Canonical key of a sampling instant (merges round-off duplicates such
    as 0.065 + 0.005 vs 7 * 0.01 produced by RK4 stage times)."""
    return float(np.round(float(t), _T_DIGITS)) + 0.0


def _same_instant(t1, t2):
    return abs(t1 - t2) <= 10.0 ** (-_T_DIGITS) * max(1.0, abs(t1))


def _hash_rng(seed, t, salt):
    """Generator that depends only on (seed, t, salt): order-independent,
    reproducible per sampling instant."""
    bits = int(np.array(_tkey(t), dtype=np.float64).view(np.uint64))
    ss = np.random.SeedSequence([int(seed) & 0xFFFFFFFF, bits & 0xFFFFFFFF, bits >> 32, salt])
    return np.random.Generator(np.random.PCG64(ss))


class JitteredProblem(ProblemWrapper):
    """Sampling-time jitter: the data delivered at nominal instant t_k are the
    exact data at t_k + delta_k, while the model (ctx.t, gain schedules) keeps
    using t_k.

    delta_k is i.i.d. across sampling instants:
      * ``jitter_std`` only   -> delta ~ N(0, std^2);
      * ``jitter_bound`` only -> delta ~ U[-bound, bound];
      * both                  -> N(0, std^2) truncated to [-bound, bound].
    delta_k is a deterministic function of (seed, t_k) (hash-seeded), so it is
    reproducible and independent of the order in which instants are queried
    (e.g. RK4 stage times get their own independent jitter).
    With ``clip_nonnegative`` the sampling time is max(t_k + delta_k, 0).

    The returned dict keeps the inner problem's ``'t'`` (the time the data
    actually belong to: t_k + delta_k for an unwrapped problem, so problems
    whose E uses d['t'] stay self-consistent; t_k + delta_k - D when wrapping
    a ``DelayedDataProblem``) and sets ``'t_nominal'`` = t_k.
    ``dEdt(x, t_k)`` returns the exact partial derivative at the actual
    sampling time (value and rate sampled by the same jittered clock).  The
    residual recorded by ``simulate`` uses exact data at t_k.

    Wrapper order matters when combining with ``EstimatedDerivativeProblem``:
    ``EstimatedDerivativeProblem(JitteredProblem(P))`` is the un-timestamped
    regime (the estimator believes samples are tau apart), whereas
    ``JitteredProblem(EstimatedDerivativeProblem(P))`` gives the estimator the
    true (time-stamped) sampling instants; out-of-order arrivals (|delta| >
    tau/2) then discard the later stored samples (see that class).
    """

    exact_residual = True

    def __init__(self, problem, jitter_std=None, jitter_bound=None, seed=0,
                 clip_nonnegative=True):
        if jitter_std is None and jitter_bound is None:
            raise ValueError("give jitter_std and/or jitter_bound")
        if jitter_std is not None and jitter_std < 0 or jitter_bound is not None and jitter_bound < 0:
            raise ValueError("jitter parameters must be non-negative")
        tag = "jit" + (f"N{jitter_std:g}" if jitter_std is not None else "") + \
              (f"U{jitter_bound:g}" if jitter_bound is not None else "")
        super().__init__(problem, tag)
        self.std = None if jitter_std is None else float(jitter_std)
        self.bound = None if jitter_bound is None else float(jitter_bound)
        self.seed = int(seed)
        self.clip_nonnegative = bool(clip_nonnegative)
        self._cache = OrderedDict()

    def delta(self, t):
        key = _tkey(t)
        if key in self._cache:
            return self._cache[key]
        g = _hash_rng(self.seed, key, 0x4A17)
        if self.std is None:
            dlt = g.uniform(-self.bound, self.bound)
        elif self.bound is None:
            dlt = self.std * g.standard_normal()
        else:
            dlt = np.inf
            for _ in range(1000):                    # rejection sampling
                dlt = self.std * g.standard_normal()
                if abs(dlt) <= self.bound:
                    break
            dlt = float(np.clip(dlt, -self.bound, self.bound))
        self._cache[key] = float(dlt)
        if len(self._cache) > 64:
            self._cache.popitem(last=False)
        return float(dlt)

    def sample_time(self, t):
        ts = float(t) + self.delta(t)
        return max(ts, 0.0) if self.clip_nonnegative else ts

    def data(self, t):
        ts = self.sample_time(t)
        d = dict(self.inner.data(ts))
        d.setdefault("t", np.float64(ts))       # keep the inner (actual) data time
        d["t_nominal"] = np.float64(t)
        return d

    def dEdt(self, x, t):
        return self.inner.dEdt(x, self.sample_time(t))

    def describe(self):
        return {**super().describe(), "jitter_std": self.std, "jitter_bound": self.bound,
                "jitter_seed": self.seed}


class DelayedDataProblem(ProblemWrapper):
    """Data arrive with a fixed delay D = delay_steps * tau (or ``delay`` s):
    at nominal instant t the model receives d(t - D).  Before the first sample
    arrives (t < D) the initial sample d(0) is held (ASSUMPTION), and the
    analytic derivative of the delivered (held) stream is 0 there.  dEdt
    otherwise returns the exact derivative of the delayed data, dE/dt at
    t - D.  The recorded residual uses exact data at t.
    """

    exact_residual = True

    def __init__(self, problem, delay=None, delay_steps=None, tau=None, hold_initial=True):
        if delay is None:
            if delay_steps is None or tau is None:
                raise ValueError("give delay (s) or delay_steps and tau")
            delay = int(delay_steps) * float(tau)
        if delay < 0:
            raise ValueError("delay must be non-negative")
        super().__init__(problem, f"delay{delay:g}")
        self.delay = float(delay)
        self.hold_initial = bool(hold_initial)

    def _ts(self, t):
        ts = float(t) - self.delay
        held = self.hold_initial and ts < 0.0
        return (0.0 if held else ts), held

    def data(self, t):
        ts, _ = self._ts(t)
        d = dict(self.inner.data(ts))
        d.setdefault("t", np.float64(ts))       # keep the inner (actual) data time
        d["t_nominal"] = np.float64(t)
        return d

    def dEdt(self, x, t):
        ts, held = self._ts(t)
        if held:
            return np.zeros((x.shape[0], self.m))
        return self.inner.dEdt(x, ts)

    def describe(self):
        return {**super().describe(), "delay": self.delay, "hold_initial": self.hold_initial}


# ---------------------------------------------------------------------------
# causal derivative estimators (linear in the samples -> FIR weights)
# ---------------------------------------------------------------------------
def savgol_weights(s, order):
    """Weights w with  sum_j w_j e(s_j) = d/ds of the degree-`order` least-
    squares polynomial fitted to (s_j, e(s_j)), evaluated at s = 0.

    s are sample times relative to the query time (all <= 0 for a causal
    estimate).  For uniform spacing this is the Savitzky-Golay (1964)
    differentiator evaluated at the window end (causal); with
    order = len(s) - 1 it reduces to the backward-difference formula of that
    order (2 points -> BD1, 3 points -> BD2).
    """
    s = np.asarray(s, dtype=float)
    deg = int(min(order, len(s) - 1))
    h = max(np.max(np.abs(np.diff(s))) if len(s) > 1 else 1.0, 1e-300)
    V = np.vander(s / h, deg + 1, increasing=True)
    return np.linalg.pinv(V)[1] / h


def alpha_beta_weights(s, alpha, beta):
    """FIR weights of an alpha-beta (g-h) tracker run over the window.

    Per sample j (dt_j = s_j - s_{j-1}):
        e^-_j = e^_{j-1} + dt_j v^_{j-1},   r_j = e_j - e^-_j,
        e^_j = e^-_j + alpha r_j,           v^_j = v^_{j-1} + (beta / dt_j) r_j,
    initialised by two-point differencing at the 2nd sample
    (e^_1 = e_1, v^_1 = (e_1 - e_0) / dt_1).  The velocity estimate is held
    (constant-velocity model) up to the query time s = 0.  Kalata (1984).
    """
    s = np.asarray(s, dtype=float)
    L = len(s)
    I = np.eye(L)
    if L < 2:
        return np.zeros(L)
    dt = s[1] - s[0]
    e_hat, v_hat = I[1].copy(), (I[1] - I[0]) / dt
    for j in range(2, L):
        dt = s[j] - s[j - 1]
        e_pred = e_hat + dt * v_hat
        r = I[j] - e_pred
        e_hat = e_pred + alpha * r
        v_hat = v_hat + (beta / dt) * r
    return v_hat


def kalman_cv_weights(s, q, r):
    """FIR weights of the velocity estimate of a constant-velocity Kalman
    filter (discrete white-noise acceleration (DWNA) model, Bar-Shalom, Li &
    Kirubarajan 2001, Ch. 6) run over the window:

        x = [e, e_dot],  F = [[1, dt], [0, 1]],  Q = q G G^T, G = [dt^2/2, dt]^T,
        H = [1, 0],      R = r,

    initialised by two-point differencing (ibid., Ch. 5):
    x_1 = [e_1, (e_1 - e_0)/dt], P_1 = [[r, r/dt], [r/dt, 2 r/dt^2]].
    The gains do not depend on the data, so the estimate is a fixed linear
    combination of the window samples; the same scalar filter is applied to
    every error component independently.
    """
    s = np.asarray(s, dtype=float)
    L = len(s)
    if L < 2:
        return np.zeros(L)
    I = np.eye(L)
    dt = s[1] - s[0]
    X = np.vstack([I[1], (I[1] - I[0]) / dt])          # 2 x L coefficient rows
    P = np.array([[r, r / dt], [r / dt, 2 * r / dt ** 2]])
    for j in range(2, L):
        dt = s[j] - s[j - 1]
        F = np.array([[1.0, dt], [0.0, 1.0]])
        G = np.array([[dt ** 2 / 2.0], [dt]])
        X = F @ X
        P = F @ P @ F.T + q * (G @ G.T)
        S = P[0, 0] + r
        K = P[:, 0] / S
        X = X + np.outer(K, I[j] - X[0])
        P = P - np.outer(K, P[0])
    return X[1]


class EstimatedDerivativeProblem(ProblemWrapper):
    """Replace the exact dE/dt by a causal estimate from sampled data.

    Every call ``data(t)`` with a time later than the last recorded one stores
    the delivered sample d_j = d(t_j) (optionally corrupted by i.i.d. Gaussian
    measurement noise of std ``meas_std``, hash-seeded per t_j).  At the query
    ``dEdt(x, t)`` the wrapper takes the last ``window`` stored samples with
    t_j <= t and returns

        dE/dt (x, t)  ~=  sum_j w_j E(x, d_j),

    i.e. it differences/filters the error sequence evaluated at the *current,
    fixed* state x (so it estimates the partial time derivative, never the
    total derivative), with weights w given by

    * 'savgol'     : causal Savitzky-Golay differentiator, polynomial degree
                     ``order`` (default window 5, order 2; error O(tau^order));
    * 'alpha_beta' : alpha-beta tracker (default alpha = 0.5 and Benedict-
                     Bordner beta = alpha^2 / (2 - alpha), window 20);
    * 'kalman'     : constant-velocity Kalman filter (DWNA model, window
                     20).  Tuned by Kalata's tracking index
                     lambda = sigma_w tau^2 / sigma_v (default 1.0, i.e.
                     q = (lambda / dt^2)^2, r = 1 with dt the median sample
                     spacing; step-size invariant, steady-state gains
                     alpha = 0.75, beta = 0.5), or, with
                     ``tracking_index=None``, by the physical acceleration
                     variance ``q`` and measurement variance ``r``.

    For a signal with constant second derivative a, the steady-state
    alpha-beta / Kalman estimate lags by  -a tau (alpha - beta/2) / beta
    (first order in tau); Savitzky-Golay of degree >= 2 is exact for it.

    The filters are linear and their gains are data independent, so the
    weights are computed once per spacing pattern (cached) and the cost is
    ``window`` evaluations of E per call.  With fewer than 2 samples the
    estimate is 0 (no information), like ``info='bd1'`` at k = 0.

    Use with ``simulate(..., info='analytic', data_noise=0)``: measurement
    noise for this wrapper is set with ``meas_std`` (the same noisy samples
    feed the model's E, solve_J, JT; time stamps 't'/'t_nominal' are never
    perturbed); the recorded residual uses exact data.
    The wrapper is stateful (sample history).  A ``data`` call at a time
    earlier than the latest stored sample discards the stored samples later
    than it, so a new run (time going back to 0) resets the history; do not
    share one instance between concurrently running simulations.  Wrapping
    order: ``EstimatedDerivativeProblem(DelayedDataProblem(P))`` and
    ``DelayedDataProblem(EstimatedDerivativeProblem(P))`` are equivalent after
    the initial hold; see ``JitteredProblem`` for the two jitter orders.
    """

    METHODS = ("savgol", "alpha_beta", "kalman")

    def __init__(self, problem, method="savgol", window=None, order=2, alpha=0.5, beta=None,
                 tracking_index=1.0, q=1.0, r=1e-8, meas_std=0.0, seed=0):
        if method not in self.METHODS:
            raise ValueError(f"method must be one of {self.METHODS}")
        super().__init__(problem, f"est-{method}")
        self.method = method
        self.window = int(window if window is not None else (order + 3 if method == "savgol" else 20))
        if self.window < 2:
            raise ValueError("window must be >= 2")
        self.order = int(order)
        self.alpha = float(alpha)
        self.beta = float(beta) if beta is not None else self.alpha ** 2 / (2.0 - self.alpha)
        self.tracking_index = None if tracking_index is None else float(tracking_index)
        self.q, self.r = float(q), float(r)
        self.meas_std = float(meas_std)
        self.seed = int(seed)
        self.exact_residual = self.meas_std > 0
        self._wcache = {}
        self.reset()

    def reset(self):
        self._times = deque(maxlen=self.window)
        self._samples = deque(maxlen=self.window)

    # --- sampling ------------------------------------------------------------
    _TIME_KEYS = ("t", "t_nominal")

    def _measure(self, t):
        d = self.inner.data(t)
        if self.meas_std > 0:
            g = _hash_rng(self.seed, t, 0x5E57)
            # time stamps (also 't_nominal' of an inner time-warping wrapper)
            # and non-floating entries are not measurements: leave them exact
            d = {k: (v if k in self._TIME_KEYS or not np.issubdtype(np.asarray(v).dtype, np.inexact)
                     else v + self.meas_std * g.standard_normal(np.shape(v)))
                 for k, v in d.items()}
        return d

    def data(self, t):
        t = float(t)
        if self._times and _same_instant(t, self._times[-1]):
            return self._samples[-1]
        # a sample earlier than the latest stored one (a new run starting
        # again at t = 0, or an out-of-order arrival when the estimator sits
        # inside a JitteredProblem) discards the stored samples later than it;
        # an earlier-than-everything sample therefore resets the history.
        while self._times and self._times[-1] > t and not _same_instant(t, self._times[-1]):
            self._times.pop()
            self._samples.pop()
        if self._times and _same_instant(t, self._times[-1]):
            return self._samples[-1]
        d = self._measure(t)
        self._times.append(t)
        self._samples.append(d)
        return d

    # --- estimator -------------------------------------------------------------
    def weights(self, s):
        key = tuple(np.round(np.asarray(s, dtype=float), 12))
        w = self._wcache.get(key)
        if w is None:
            if self.method == "savgol":
                w = savgol_weights(s, self.order)
            elif self.method == "alpha_beta":
                w = alpha_beta_weights(s, self.alpha, self.beta)
            elif self.tracking_index is not None:
                dt = float(np.median(np.diff(s)))
                w = kalman_cv_weights(s, (self.tracking_index / dt ** 2) ** 2, 1.0)
            else:
                w = kalman_cv_weights(s, self.q, self.r)
            if len(self._wcache) > 256:
                self._wcache.clear()
            self._wcache[key] = w
        return w

    def dEdt(self, x, t):
        t = float(t)
        idx = [i for i, tj in enumerate(self._times) if tj <= t or _same_instant(tj, t)]
        if len(idx) < 2:
            return np.zeros((x.shape[0], self.m))
        s = np.array([self._times[i] for i in idx]) - t
        w = self.weights(s)
        out = np.zeros((x.shape[0], self.m))
        for wj, i in zip(w, idx):
            if wj != 0.0:
                out = out + wj * self.inner.E(x, self._samples[i])
        return out

    def describe(self):
        return {**super().describe(), "method": self.method, "window": self.window,
                "order": self.order, "alpha": self.alpha, "beta": self.beta,
                "tracking_index": self.tracking_index, "q": self.q,
                "r": self.r, "meas_std": self.meas_std}


# ===========================================================================
# cost accounting
# ===========================================================================
class CountingProblem(ProblemWrapper):
    """Pass-through wrapper that counts calls and accumulates wall time of
    every Problem method (``residual`` is counted separately: it is the
    benchmark's diagnostic, not part of the algorithm)."""

    _METHODS = ("data", "E", "dEdt", "solve_J", "JT")

    def __init__(self, problem):
        super().__init__(problem, "count")
        self.name = problem.name
        self.reset_counts()

    def reset_counts(self):
        self.calls = {k: 0 for k in self._METHODS + ("residual",)}
        self.wall = {k: 0.0 for k in self._METHODS + ("residual",)}

    def _timed(self, key, fn, *a):
        t0 = time.perf_counter()
        out = fn(*a)
        self.wall[key] += time.perf_counter() - t0
        self.calls[key] += 1
        return out

    def data(self, t):
        return self._timed("data", self.inner.data, t)

    def E(self, x, d):
        return self._timed("E", self.inner.E, x, d)

    def dEdt(self, x, t):
        return self._timed("dEdt", self.inner.dEdt, x, t)

    def solve_J(self, x, d, r):
        return self._timed("solve_J", self.inner.solve_J, x, d, r)

    def JT(self, x, d, v):
        return self._timed("JT", self.inner.JT, x, d, v)

    def residual(self, x, d):
        return self._timed("residual", self.inner.residual, x, d)


def dense_flops(n_x, m, B=1):
    """Rough dense-linear-algebra FLOP model per call (ASSUMPTION: unstructured
    dense J of size m x n_x; problem-specific structure such as Kronecker
    products is ignored): E, dEdt, JT ~ 2 m n_x B (one mat-vec per
    trajectory); solve_J ~ factorisation (2/3 n^3 if square, 2 m n_x^2 for
    least squares) + 2 m n_x B for the triangular solves."""
    mv = 2.0 * m * n_x * B
    fact = (2.0 / 3.0) * n_x ** 3 if m == n_x else 2.0 * m * n_x ** 2
    return {"E": mv, "dEdt": mv, "JT": mv, "solve_J": fact + mv, "data": 0.0}


def profile_step_cost(problem, model, disc, tau=1e-2, steps=200, B=1, gamma=10.0,
                      info="analytic", seed=0, **sim_kw):
    """Run ``steps`` steps through a ``CountingProblem`` and report per-step
    call counts, wall-clock (total and per method) and a rough dense FLOP
    estimate (see ``dense_flops``).  Wall-clock includes Python overhead and
    is machine dependent; use it only for relative comparisons."""
    from .simulate import simulate

    P = CountingProblem(problem)
    sim_kw = {"online_stats": False, **sim_kw}      # count the dynamics only
    sim = simulate(P, model, disc, tau=tau, T=steps * tau, B=B, gamma=gamma, info=info,
                   seed=seed, **sim_kw)
    n = max(int(sim["steps"]), 1)
    calls = {k: v / n for k, v in P.calls.items() if k != "residual"}
    fl = dense_flops(problem.n_x, problem.m, B)
    return {"calls_per_step": calls,
            "wall_per_step": sim["wall"] / n,
            "wall_per_step_by_method": {k: v / n for k, v in P.wall.items()},
            "flops_per_step_dense": float(sum(calls[k] * fl[k] for k in calls)),
            "steps": n, "B": B}
