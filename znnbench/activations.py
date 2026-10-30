"""Activation functions Phi for the ZNN design formula dE/dt = -gamma Phi(E).

All activations act element-wise, are odd and monotonically non-decreasing.
Each provides
  * ``__call__(e)``      -- Phi(e);
  * ``slope0``           -- Phi'(0) (np.inf for non-Lipschitz activations);
  * ``prox(e, h)``       -- the resolvent y = (I + h Phi)^{-1}(e), i.e. the
                            unique y with y + h Phi(y) = e.  This is what an
                            implicit (backward-Euler) discretization of the
                            error dynamics needs; for activations with a jump
                            at 0 (sign terms) the inclusion y + h Phi(y) ∋ e
                            is solved, which returns exactly 0 inside the
                            dead zone and removes chattering.
  * ``finite_time``      -- True for the finite/fixed/predefined-time class;
  * ``convergence``      -- 'exponential' | 'asymptotic' | 'finite' | 'fixed' |
                            'predefined' | 'strictly-predefined': the class of
                            the continuous-time scalar dynamics.  It equals the
                            source's claim, except where ZNN-Bench proves a
                            stronger class (sign_bi_power, wsbp: claimed finite,
                            DERIVED fixed via a finite ``uniform_bound``);
  * ``claimed_convergence`` -- the class claimed by the source paper (defaults
                            to ``convergence``; see docs/spec/activations.md);
  * ``upper_bound_time(e0, gamma)`` -- the convergence-time bound of the
                            scalar error dynamics e' = -gamma Phi(e) stated by
                            the source (None when the source states none or it
                            could not be verified).  ``e0`` may be an array;
                            as in the sources, the bound uses max_i |e0_i|;
  * ``uniform_bound(gamma)`` -- a bound valid for every e0 (np.inf when the
                            settling time is unbounded in e0, or infinite);
  * ``settling_time(e0, gamma)`` -- the EXACT settling time
                            T(e0) = (1/gamma) * int_0^{|e0|} ds / Phi(s) of
                            e' = -gamma Phi(e) (np.inf for asymptotic ones),
                            computed by quadrature (this is the improper-
                            integral characterisation used by Xiao et al.,
                            IEEE TII 17(7):4724-4732, 2021);
  * ``at(t)``            -- the activation frozen at time t (identity for the
                            time-invariant ones; see StrictlyPredefinedTime);
  * ``tight_uniform_bound(gamma)`` -- (bi-power families only: sign_bi_power,
                            wsbp, vaf, fixed_time) DERIVED sup_e0 of the exact
                            settling time (Beta integral).

Equations, sources, defaults and every assumption are documented in
docs/spec/activations.md.  ``provenance`` says how each formula was checked:
'primary' = read in the original paper, 'secondary' = read in a later paper
that transcribes the original (the original was not accessible), 'derived' =
a ZNN-Bench derivation (flagged ASSUMPTION/DERIVED in the spec).
"""
from __future__ import annotations

import math

import numpy as np

REGISTRY = {}


def register(cls):
    if cls.key in REGISTRY and REGISTRY[cls.key] is not cls:
        raise KeyError(f"duplicate activation key {cls.key!r}")
    REGISTRY[cls.key] = cls
    return cls


def get(key: str, **kw):
    return REGISTRY[key](**kw)


def available():
    """Sorted list of registered activation keys."""
    return sorted(REGISTRY)


def _spow(e, p):
    """Signed power sign(e)|e|^p (0 maps to 0)."""
    return np.sign(e) * np.abs(e) ** p


def _amax(e0) -> float:
    """max_i |e0_i| (the element with the largest initial error decides the
    bound in all the cited theorems)."""
    e0 = np.asarray(e0, dtype=float)
    return float(np.max(np.abs(e0))) if e0.size else 0.0


def _require(cond, msg):
    if not cond:
        raise ValueError(msg)


def _odd_int(p, lo, name):
    _require(float(p) == int(p) and int(p) >= lo and int(p) % 2 == 1,
             f"{name} must be an odd integer >= {lo} (got {p})")
    return int(p)


class Activation:
    key = "base"
    slope0 = 1.0
    finite_time = False      # continuous-time finite/fixed/predefined-time class
    convergence = "exponential"
    time_varying = False
    source = ""
    provenance = ""

    @property
    def claimed_convergence(self) -> str:
        """Class claimed by the source (subclasses override when ZNN-Bench
        derives a stronger ``convergence`` class)."""
        return self.convergence

    def __call__(self, e):
        raise NotImplementedError

    def jump0(self) -> float:
        """Right limit Phi(0+) (> 0 for activations with a sign term)."""
        return float(self(np.array([1e-300]))[0])

    def prox(self, e, h, iters: int = 80):
        """Vectorised resolvent by safeguarded GEOMETRIC (log-scale) bisection
        on y in [tiny, |e|] (tiny = smallest normal double).

        g(y) = y + h Phi(y) - |e| is increasing on [0, |e|]; by oddness the
        solution for negative e is mirrored.  If h*Phi(0+) >= |e| the
        solution of the inclusion is y = 0.  Bisecting log(y) gives relative
        accuracy at every scale: for strongly non-Lipschitz activations
        (small exponents, large h) the root can lie 30-300 orders of magnitude
        below |e|, where a linear bisection on [0, |e|] stalls at |e| 2^-iters
        (verifier finding).  80 halvings of ln(|e|/tiny) <= 1418 reach machine
        precision.  A root below tiny (incl. e = 0) is returned as 0.
        """
        e = np.asarray(e, dtype=float)
        h = np.broadcast_to(np.asarray(h, dtype=float), e.shape) if np.ndim(h) else float(h)
        a = np.abs(e)
        tiny = np.finfo(float).tiny
        lo = np.full_like(a, tiny)
        hi = np.maximum(a, tiny)
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            below = (lo + h * self(lo) - a) > 0       # root < tiny (or e == 0)
            for _ in range(iters):
                mid = np.minimum(np.maximum(np.sqrt(lo) * np.sqrt(hi), lo), hi)
                g = mid + h * self(mid) - a
                pos = g > 0
                hi = np.where(pos, mid, hi)
                lo = np.where(pos, lo, mid)
            y = np.minimum(np.maximum(np.sqrt(lo) * np.sqrt(hi), lo), hi)
        y = np.where(below, 0.0, y)
        dead = (h * self.jump0()) >= a
        y = np.where(dead, 0.0, y)
        return np.sign(e) * y

    def params(self) -> dict:
        return {}

    def describe(self) -> dict:
        return {"key": self.key, **self.params()}

    def at(self, t):
        """Activation frozen at time t (time-invariant: self)."""
        return self

    # --- continuous-time convergence-time theory ---------------------------
    def upper_bound_time(self, e0, gamma=1.0):
        """Convergence-time bound stated by the source (None if none)."""
        return None

    def uniform_bound(self, gamma=1.0):
        """Bound on the settling time valid for every initial error."""
        return math.inf

    def settling_time(self, e0, gamma=1.0):
        """Exact settling time of e' = -gamma Phi(e) from e(0) = max|e0|:
        T = (1/gamma) int_0^a ds/Phi(s), evaluated with the substitution
        s = a exp(-v) (smooth, exponentially decaying integrand)."""
        a = _amax(e0)
        if a == 0.0:
            return 0.0
        if not self.finite_time:
            return math.inf
        from scipy.integrate import quad

        def integrand(v):
            s = a * math.exp(-v)
            if s == 0.0:
                return 0.0
            with np.errstate(over="ignore"):
                phi = float(self(np.array([s]))[0])
            return s / phi if phi > 0 else 0.0

        val, _ = quad(integrand, 0.0, np.inf, limit=500, epsabs=0.0, epsrel=1e-11)
        return val / float(gamma)


# ===========================================================================
# 1. Exponentially / asymptotically convergent activations (Zhang 2002-2013)
# ===========================================================================
@register
class Linear(Activation):
    """Linear activation Phi(e) = e (Zhang, Jiang, Wang, IEEE TNN 13(5), 2002;
    Zhang & Ge, IEEE TNN 16(6), 2005, activation (i)).  e(t) = e0 exp(-gamma t)."""

    key = "linear"
    slope0 = 1.0
    source = "Zhang, Jiang, Wang 2002 (doi:10.1109/TNN.2002.1031938); Zhang & Ge 2005"
    provenance = "secondary"

    def __call__(self, e):
        return np.asarray(e, dtype=float)

    def prox(self, e, h, iters=0):
        return np.asarray(e, dtype=float) / (1.0 + h)


@register
class Power(Activation):
    """Power activation Phi(e) = e^p, odd integer p >= 3 (Zhang, Jiang, Wang
    2002; Zhang & Ge 2005, activation (iii)).  Phi'(0) = 0, so the decay near
    0 is only algebraic: e(t) = e0 (1 + (p-1) gamma e0^(p-1) t)^(-1/(p-1)).

    ``prox``: closed form (hyperbolic Cardano formula) for p = 3; monotone
    Newton iteration from the right for p >= 5 (g(y) = y + h y^p - |e| is
    convex and increasing on y >= 0, so Newton started at an upper bound of
    the root converges monotonically)."""

    key = "power"
    slope0 = 0.0
    convergence = "asymptotic"
    source = "Zhang, Jiang, Wang 2002 (doi:10.1109/TNN.2002.1031938); Zhang & Ge 2005 (doi:10.1109/TNN.2005.857946)"
    provenance = "secondary"

    def __init__(self, p: int = 3):
        self.p = _odd_int(p, 3, "p")

    def __call__(self, e):
        return np.asarray(e, dtype=float) ** self.p

    def prox(self, e, h, iters: int = 100):
        e = np.asarray(e, dtype=float)
        a = np.abs(e)
        h = np.asarray(h, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            if self.p == 3:
                # y^3 + y/h - a/h = 0  =>  y = 2/sqrt(3h) sinh(asinh(1.5 a sqrt(3h)) / 3)
                s3h = np.sqrt(3.0 * h)
                y = 2.0 / s3h * np.sinh(np.arcsinh(1.5 * a * s3h) / 3.0)
                y = np.where(h > 0, y, a)
            else:
                p = self.p
                y = np.where(h > 0, np.minimum(a, (a / h) ** (1.0 / p)), a)
                for _ in range(iters):
                    g = y + h * y ** p - a
                    step = g / (1.0 + p * h * y ** (p - 1))
                    y_new = np.maximum(y - step, 0.0)
                    if np.all(np.abs(y_new - y) <= 1e-15 * np.maximum(a, 1e-300)):
                        y = y_new
                        break
                    y = y_new
        return np.sign(e) * y

    def params(self):
        return {"p": self.p}


@register
class BipolarSigmoid(Activation):
    """Bipolar sigmoid Phi(e) = (1 - exp(-xi e)) / (1 + exp(-xi e)) = tanh(xi e/2),
    xi >= 2 in the source (Zhang & Ge 2005, activation (ii), as transcribed by
    Wang et al., ICITMI 2015).  ``normalized=True`` multiplies by
    (1 + exp(-xi)) / (1 - exp(-xi)) so that Phi(1) = 1 (the scaling used inside
    the power-sigmoid activation).  Bounded: |Phi| < 1 (or < coth(xi/2))."""

    key = "bipolar_sigmoid"
    source = "Zhang & Ge 2005 (doi:10.1109/TNN.2005.857946)"
    provenance = "secondary"

    def __init__(self, xi: float = 4.0, normalized: bool = False):
        _require(xi > 0, "xi must be > 0 (source: xi >= 2)")
        self.xi = float(xi)
        self.normalized = bool(normalized)
        self.c = 1.0 / np.tanh(0.5 * self.xi) if self.normalized else 1.0
        self.slope0 = self.c * 0.5 * self.xi

    def __call__(self, e):
        return self.c * np.tanh(0.5 * self.xi * np.asarray(e, dtype=float))

    def params(self):
        return {"xi": self.xi, "normalized": self.normalized}


@register
class PowerSigmoid(Activation):
    """Power-sigmoid activation (Zhang & Ge 2005, activation (iv)):
        Phi(e) = e^p                                                   , |e| >= 1
               = (1+exp(-xi))/(1-exp(-xi)) * (1-exp(-xi e))/(1+exp(-xi e)), |e| < 1
    with xi >= 2 and odd p >= 3 (verified against the transcription in Wang et
    al., ICITMI 2015, and Zhou & Tan, BPASTS 2022, Table 1).  Continuous at
    |e| = 1; Phi'(0) = (xi/2) coth(xi/2).  Default p = 3, xi = 4 (the values
    used in the power-sigmoid simulations of Wang et al. 2015, following
    Zhang's papers; Zhang & Ge 2005 itself not read)."""

    key = "power_sigmoid"
    source = "Zhang & Ge 2005 (doi:10.1109/TNN.2005.857946)"
    provenance = "secondary"

    def __init__(self, p: int = 3, xi: float = 4.0):
        self.p = _odd_int(p, 3, "p")
        _require(xi > 0, "xi must be > 0 (source: xi >= 2)")
        self.xi = float(xi)
        self.slope0 = 0.5 * self.xi / np.tanh(0.5 * self.xi)

    def __call__(self, e):
        e = np.asarray(e, dtype=float)
        # (1+exp(-xi))/(1-exp(-xi)) * (1-exp(-xi e))/(1+exp(-xi e)) written overflow-free
        sig = np.tanh(0.5 * self.xi * e) / np.tanh(0.5 * self.xi)
        return np.where(np.abs(e) >= 1.0, e ** self.p, sig)

    def params(self):
        return {"p": self.p, "xi": self.xi}


@register
class PowerSum(Activation):
    """Power-sum activation Phi(e) = sum_{k=1}^{N} e^(2k-1), integer N >= 2
    (Yang & Zhang, Neural Comput. Appl. 22:175-185, 2013).  Phi'(0) = 1.
    PROVENANCE: formula from a secondary description (Jin et al. survey,
    Neurocomputing 2017); default N = 3 is a ZNN-Bench ASSUMPTION."""

    key = "power_sum"
    slope0 = 1.0
    source = "Yang & Zhang 2013 (doi:10.1007/s00521-011-0692-5)"
    provenance = "secondary"

    def __init__(self, N: int = 3):
        _require(float(N) == int(N) and int(N) >= 2, "N must be an integer >= 2")
        self.N = int(N)

    def __call__(self, e):
        e = np.asarray(e, dtype=float)
        out = np.zeros_like(e)
        for k in range(1, self.N + 1):
            out = out + e ** (2 * k - 1)
        return out

    def params(self):
        return {"N": self.N}


@register
class HyperbolicSine(Activation):
    """Hyperbolic sine activation Phi(e) = exp(xi e)/2 - exp(-xi e)/2 = sinh(xi e),
    design parameter xi >= 1 (Zhang, Jin, Ke, Comput. Sci. Inf. Syst. 9(4),
    2012, Sec. 2; simulations use xi = 3, 5, 7).  Phi'(0) = xi.  Exact scalar
    dynamics: tanh(xi e(t)/2) = tanh(xi e0/2) exp(-gamma xi t)."""

    key = "hyperbolic_sine"
    source = "Zhang, Jin, Ke 2012 (doi:10.2298/CSIS120121043Z)"
    provenance = "primary"

    def __init__(self, xi: float = 3.0):
        _require(xi > 0, "xi must be > 0 (source: xi >= 1)")
        self.xi = float(xi)
        self.slope0 = self.xi

    def __call__(self, e):
        with np.errstate(over="ignore"):
            return np.sinh(self.xi * np.asarray(e, dtype=float))

    def params(self):
        return {"xi": self.xi}


# ===========================================================================
# 2. Finite-time activations
# ===========================================================================
@register
class Sign(Activation):
    """Signum activation Phi(e) = sign(e) (Zhang, Ding, Qiu, Zhang, Li,
    Inf. Process. Lett. 124:30-34, 2017: signum-function array activated ZNN).
    e' = -gamma sign(e) reaches 0 exactly at T = |e0|/gamma (DERIVED, exact).
    In discrete time it is the finite-step / chattering baseline.
    ``prox`` is the soft-threshold sign(e) max(|e| - h, 0)."""

    key = "sign"
    slope0 = np.inf
    finite_time = True
    convergence = "finite"
    source = "Zhang, Ding, Qiu, Zhang, Li 2017 (doi:10.1016/j.ipl.2017.04.008)"
    provenance = "secondary"

    def __call__(self, e):
        return np.sign(np.asarray(e, dtype=float))

    def jump0(self):
        return 1.0

    def prox(self, e, h, iters=0):
        e = np.asarray(e, dtype=float)
        return np.sign(e) * np.maximum(np.abs(e) - h, 0.0)

    def upper_bound_time(self, e0, gamma=1.0):
        return _amax(e0) / gamma

    def settling_time(self, e0, gamma=1.0):
        return _amax(e0) / gamma


@register
class SignPower(Activation):
    """Signum-power activation Phi(e) = sig^r(e) = sign(e)|e|^r, 0 < r < 1
    (Lf^tau of Zhang, Chen, Li et al., CAAI Trans. Intell. Technol. 6(4), 2021,
    eqs. (8), (11), SSZNN; Theorem 3: T = |e_max(0)|^(1-r) / (gamma (1-r)),
    which is exact)."""

    key = "sign_power"
    slope0 = np.inf
    finite_time = True
    convergence = "finite"
    source = "Zhang, Chen, Li et al. 2021 (doi:10.1049/cit2.12019)"
    provenance = "primary"

    def __init__(self, r: float = 0.5):
        _require(0 < r < 1, "r must satisfy 0 < r < 1")
        self.r = float(r)

    def __call__(self, e):
        return _spow(np.asarray(e, dtype=float), self.r)

    def upper_bound_time(self, e0, gamma=1.0):
        return _amax(e0) ** (1.0 - self.r) / (gamma * (1.0 - self.r))

    def settling_time(self, e0, gamma=1.0):
        return self.upper_bound_time(e0, gamma)

    def params(self):
        return {"r": self.r}


@register
class SignBiPower(Activation):
    """Sign-bi-power activation (Li, Chen, Liu, Neural Process. Lett. 37:189-205,
    2013; called the "Li activation function" by Liao & Zhang, Neurocomputing
    133, 2014, Guo & Zhang, Appl. Soft Comput. 24, 2014, and Wang et al.,
    ICITMI 2015, eqs. (3)-(4)).  Not registered separately as "li": some later
    papers (e.g. Liao et al. 2022, used by models/variants.py) drop the 1/2,
    which is a gain rescaling.
        Phi(e) = (sig^r(e) + sig^(1/r)(e)) / 2,   0 < r < 1.
    Source bound (as stated in Hu et al., Neurocomputing 418, 2020, Thm. 2):
        t_f <= 2 |e_max(0)|^(1-r) / (gamma (1-r)).
    DERIVED uniform (fixed-time) bound, using both terms:
        t_f <= 2 (1 + r) / (gamma (1 - r)),
    and the DERIVED tight one ``tight_uniform_bound`` (sup of the exact
    settling time, Beta integral; 4.84/gamma for r = 1/2).  The source claims
    finite-time convergence (``claimed_convergence``)."""

    key = "sign_bi_power"
    slope0 = np.inf
    finite_time = True
    convergence = "fixed"
    claimed_convergence = "finite"
    source = "Li, Chen, Liu 2013 (doi:10.1007/s11063-012-9241-1)"
    provenance = "secondary"

    def __init__(self, r: float = 0.5):
        _require(0 < r < 1, "r must satisfy 0 < r < 1")
        self.r = float(r)

    def __call__(self, e):
        e = np.asarray(e, dtype=float)
        return 0.5 * (_spow(e, self.r) + _spow(e, 1.0 / self.r))

    def upper_bound_time(self, e0, gamma=1.0):
        return 2.0 * _amax(e0) ** (1.0 - self.r) / (gamma * (1.0 - self.r))

    def uniform_bound(self, gamma=1.0):
        return 2.0 * (1.0 + self.r) / (gamma * (1.0 - self.r))

    def tight_uniform_bound(self, gamma=1.0):
        """DERIVED sup_e0 of the exact settling time (see _bipower_tight)."""
        return _bipower_tight(0.5, 0.5, self.r, 1.0 / self.r, gamma)

    def params(self):
        return {"r": self.r}


@register
class SimplifiedFiniteTime(Activation):
    """Simplified finite-time activation (SFAF; "simplified SBP", SSBP-AF):
        Phi(e) = b1 sig^r(e) + b2 e,   0 < r < 1, b1, b2 > 0
    (Xiao, Zhang, Li, IEEE TSMC-Syst. 49(11):2210-2220, 2019, as transcribed by
    Hu et al., Neurocomputing 418:79-90, 2020, eq. (11); b1 = b2 = 1 in Xiao et
    al., IEEE TII 14(1), 2018 / Zuo et al., IEEE TNNLS 33(12), 2022, eq. (6);
    the "combined activation" kappa1 Lf^tau + kappa2 e of Zhang et al. 2021).
    Exact settling time (Hu et al. 2020, Thm. 2; Zuo et al. 2022, Thm. 1):
        T = ln(1 + b2 |e0|^(1-r) / b1) / (gamma b2 (1 - r))."""

    key = "sfaf"
    slope0 = np.inf
    finite_time = True
    convergence = "finite"
    source = "Xiao, Zhang, Li 2019 (doi:10.1109/TSMC.2018.2836968); Hu et al. 2020 (doi:10.1016/j.neucom.2020.08.037)"
    provenance = "secondary"

    def __init__(self, r: float = 0.5, b1: float = 1.0, b2: float = 1.0):
        _require(0 < r < 1, "r must satisfy 0 < r < 1")
        _require(b1 > 0 and b2 > 0, "b1, b2 must be > 0")
        self.r, self.b1, self.b2 = float(r), float(b1), float(b2)

    def __call__(self, e):
        e = np.asarray(e, dtype=float)
        return self.b1 * _spow(e, self.r) + self.b2 * e

    def upper_bound_time(self, e0, gamma=1.0):
        a = _amax(e0)
        return math.log1p(self.b2 * a ** (1.0 - self.r) / self.b1) / (gamma * self.b2 * (1.0 - self.r))

    def settling_time(self, e0, gamma=1.0):
        return self.upper_bound_time(e0, gamma)

    def params(self):
        return {"r": self.r, "b1": self.b1, "b2": self.b2}


@register
class WeightedSignBiPower(Activation):
    """Weighted sign-bi-power activation (WSBP / tunable SBP):
        Phi(e) = k1 sig^r(e) + k2 sig^(1/r)(e) + k3 e,  0 < r < 1, k1, k2 > 0, k3 >= 0
    (Xiao, Neurocomputing 173:1983-1988, 2016, as transcribed in Chai et al.,
    Int. J. Comput. Intell. Syst. 13(1), 2020, eq. (6):
    lambda1 sgn^k(x) + lambda2 sgn^(1/k)(x) + lambda3 x, lambda_i > 0, k in (0, 1)
    -- verified on the rendered PDF; "Wsbp" in Lv, Xiao, Tan, Yang,
    Neurocomputing 314, 2018).  ZNN-Bench relaxes lambda3 > 0 to k3 >= 0 so
    that k1 = k2 = 1/2, k3 = 0 gives sign_bi_power (ASSUMPTION, flagged).
    The source's own bound was not retrieved: ``upper_bound_time`` is None.
    DERIVED uniform bound: 1/(gamma k1 (1-r)) + r/(gamma k2 (1-r)); DERIVED
    tight bound ``tight_uniform_bound`` (Beta integral of the k1, k2 terms).
    The source claims finite-time convergence (``claimed_convergence``)."""

    key = "wsbp"
    slope0 = np.inf
    finite_time = True
    convergence = "fixed"
    claimed_convergence = "finite"
    source = "Xiao 2016 (doi:10.1016/j.neucom.2015.08.031), via Chai et al. 2020 (doi:10.2991/ijcis.d.200527.001)"
    provenance = "secondary"

    def __init__(self, r: float = 0.5, k1: float = 1.0, k2: float = 1.0, k3: float = 1.0):
        _require(0 < r < 1, "r must satisfy 0 < r < 1")
        _require(k1 > 0 and k2 > 0 and k3 >= 0, "need k1, k2 > 0 and k3 >= 0")
        self.r, self.k1, self.k2, self.k3 = float(r), float(k1), float(k2), float(k3)

    def __call__(self, e):
        e = np.asarray(e, dtype=float)
        return self.k1 * _spow(e, self.r) + self.k2 * _spow(e, 1.0 / self.r) + self.k3 * e

    def uniform_bound(self, gamma=1.0):
        return (1.0 / (self.k1 * (1.0 - self.r)) + self.r / (self.k2 * (1.0 - self.r))) / gamma

    def tight_uniform_bound(self, gamma=1.0):
        """DERIVED sup_e0 of the settling time of the k1, k2 terms (exact when
        k3 = 0, an upper bound otherwise; see _bipower_tight)."""
        return _bipower_tight(self.k1, self.k2, self.r, 1.0 / self.r, gamma)

    def params(self):
        return {"r": self.r, "k1": self.k1, "k2": self.k2, "k3": self.k3}


@register
class SaturatedSignPower(Activation):
    """Saturated signum-power activation (SZNN of Zhang, Chen, Li et al., CAAI
    Trans. Intell. Technol. 6(4), 2021, Sec. 3.4):
        Phi(e) = sign(e) min(|e|^r, s),   0 < r < 1, s > 0.
    The paper saturates gamma*Lf^r(e) at eta; under the reading used in its
    proof (e' = -eta in saturation) this is Phi with s = eta/gamma.
    Bound (DERIVED, exact; the paper's printed bound assumes the saturation
    switches at |e| = eta, which is inconsistent with its own definition):
        e_s = s^(1/r);  T = |e0|^(1-r)/(gamma(1-r))                     if |e0| <= e_s
                         T = (|e0| - e_s)/(gamma s) + e_s^(1-r)/(gamma(1-r)) otherwise."""

    key = "saturated_sign_power"
    slope0 = np.inf
    finite_time = True
    convergence = "finite"
    source = "Zhang, Chen, Li et al. 2021 (doi:10.1049/cit2.12019), Sec. 3.4"
    provenance = "primary"

    def __init__(self, r: float = 0.5, s: float = 1.0):
        _require(0 < r < 1, "r must satisfy 0 < r < 1")
        _require(s > 0, "saturation level s must be > 0")
        self.r, self.s = float(r), float(s)

    def __call__(self, e):
        e = np.asarray(e, dtype=float)
        return np.sign(e) * np.minimum(np.abs(e) ** self.r, self.s)

    def upper_bound_time(self, e0, gamma=1.0):
        a = _amax(e0)
        es = self.s ** (1.0 / self.r)
        if a <= es:
            return a ** (1.0 - self.r) / (gamma * (1.0 - self.r))
        return (a - es) / (gamma * self.s) + es ** (1.0 - self.r) / (gamma * (1.0 - self.r))

    def settling_time(self, e0, gamma=1.0):
        return self.upper_bound_time(e0, gamma)

    def params(self):
        return {"r": self.r, "s": self.s}


# ===========================================================================
# 3. Fixed-time and "predefined-time" activations
# ===========================================================================
def _bipower_tight(k1, k2, p, q, gamma):
    """DERIVED: sup over e0 of the settling time of e' = -gamma (k1|e|^p + k2|e|^q)
    sign(e), 0 < p < 1 < q:  int_0^inf ds/(k1 s^p + k2 s^q)
    = (k1/k2)^m pi / (k1 (q-p) sin(pi m)),  m = (1-p)/(q-p)  (Beta integral;
    the tight fixed-time estimate of Parsegov et al. 2012 / Aldana-Lopez et al.
    2019 with k = 1).  Extra linear / sign terms only shorten the settling time,
    so this remains an upper bound for them."""
    m = (1.0 - p) / (q - p)
    return (k1 / k2) ** m * math.pi / (k1 * (q - p) * math.sin(math.pi * m)) / gamma


@register
class VersatileActivation(Activation):
    """Versatile activation function (VAF) of Xiao et al., Neural Netw.
    117:124-134, 2019 (NNTZNN, "predefined-time" + noise tolerant):
        Phi(e) = (a1 |e|^eta + a2 |e|^omega) sign(e) + a3 e + a4 sign(e),
        0 < eta < 1, omega > 1, a1, a2 > 0, a3, a4 >= 0
    (as transcribed by Zhang & Yin, AIMS Math. 7(2):2266-2280, 2022, eq. (4.1),
    and Zhou & Tan, BPASTS 70(3), 2022, Table 1).  Bound (Zhang & Yin 2022,
    Lemma 3 / Thm. 2 / eq. (4.6), "referring to" Xiao et al. 2019):
        t_c <= 1/(gamma a1 (1 - eta)) + 1/(gamma a2 (omega - 1)).
    The same form (kappa1..kappa4, p, q) is activation F1 of Li, Liao, Xiao, Lu,
    Neurocomputing 337, 2019.  Defaults a_i = 1, eta = 0.5, omega = 2 are those
    of Zhang & Yin 2022 (ASSUMPTION: Xiao et al.'s own values not retrieved)."""

    key = "vaf"
    slope0 = np.inf
    finite_time = True
    convergence = "predefined"
    source = "Xiao et al. 2019, Neural Netw. 117:124-134 (doi:10.1016/j.neunet.2019.05.005)"
    provenance = "secondary"

    def __init__(self, a1=1.0, a2=1.0, a3=1.0, a4=1.0, eta=0.5, omega=2.0):
        _require(0 < eta < 1 and omega > 1, "need 0 < eta < 1 < omega")
        _require(a1 > 0 and a2 > 0 and a3 >= 0 and a4 >= 0, "need a1, a2 > 0 and a3, a4 >= 0")
        self.a1, self.a2, self.a3, self.a4 = float(a1), float(a2), float(a3), float(a4)
        self.eta, self.omega = float(eta), float(omega)

    def __call__(self, e):
        e = np.asarray(e, dtype=float)
        s, x = np.sign(e), np.abs(e)
        with np.errstate(over="ignore"):
            return (self.a1 * x ** self.eta + self.a2 * x ** self.omega + self.a4) * s + self.a3 * e

    def upper_bound_time(self, e0, gamma=1.0):
        return (1.0 / (self.a1 * (1.0 - self.eta)) + 1.0 / (self.a2 * (self.omega - 1.0))) / gamma

    def uniform_bound(self, gamma=1.0):
        return self.upper_bound_time(None, gamma)

    def tight_uniform_bound(self, gamma=1.0):
        """DERIVED tight bound of the bi-power part (see _bipower_tight)."""
        return _bipower_tight(self.a1, self.a2, self.eta, self.omega, gamma)

    def params(self):
        return {"a1": self.a1, "a2": self.a2, "a3": self.a3, "a4": self.a4,
                "eta": self.eta, "omega": self.omega}


@register
class FixedTimeBiPower(Activation):
    """Fixed-time bi-power activation with linear term
        Phi(e) = (k1 |e|^p + k2 |e|^q) sign(e) + k3 e,  0 < p < 1 < q, k1, k2 > 0, k3 >= 0
    (Jin, Zhu, Zhao, Chen, Appl. Soft Comput. 130:109691, 2022 -- formula as
    listed in Wang, Hua, Khan, Biomimetics 10:279, 2025, Table 1; the VAF with
    a4 = 0).  Bound: the Polyakov-type estimate of the VAF,
        T <= 1/(gamma k1 (1 - p)) + 1/(gamma k2 (q - 1))
    (ASSUMPTION: the ASOC paper's own statement was not retrieved)."""

    key = "fixed_time"
    slope0 = np.inf
    finite_time = True
    convergence = "fixed"
    source = "Jin, Zhu, Zhao, Chen 2022 (doi:10.1016/j.asoc.2022.109691)"
    provenance = "secondary"

    def __init__(self, k1=1.0, k2=1.0, k3=1.0, p=0.5, q=2.0):
        _require(0 < p < 1 and q > 1, "need 0 < p < 1 < q")
        _require(k1 > 0 and k2 > 0 and k3 >= 0, "need k1, k2 > 0 and k3 >= 0")
        self.k1, self.k2, self.k3, self.p, self.q = float(k1), float(k2), float(k3), float(p), float(q)

    def __call__(self, e):
        e = np.asarray(e, dtype=float)
        s, x = np.sign(e), np.abs(e)
        with np.errstate(over="ignore"):
            return (self.k1 * x ** self.p + self.k2 * x ** self.q) * s + self.k3 * e

    def upper_bound_time(self, e0, gamma=1.0):
        return (1.0 / (self.k1 * (1.0 - self.p)) + 1.0 / (self.k2 * (self.q - 1.0))) / gamma

    def uniform_bound(self, gamma=1.0):
        return self.upper_bound_time(None, gamma)

    def tight_uniform_bound(self, gamma=1.0):
        """DERIVED tight bound of the bi-power part (see _bipower_tight)."""
        return _bipower_tight(self.k1, self.k2, self.p, self.q, gamma)

    def params(self):
        return {"k1": self.k1, "k2": self.k2, "k3": self.k3, "p": self.p, "q": self.q}


@register
class PowerVersatile1(Activation):
    """Power-versatile activation PVAF-1 (Zhou & Tan, Bull. Pol. Acad. Sci.
    Tech. Sci. 70(3):e141307, 2022, eq. (6)):
        Phi(e) = (a |e|^p + b |e|^q)^k sign(e) + c1 e + c2 sign(e),
        a, b, p, q, k > 0, pk > 1, qk < 1, c1, c2 >= 0.
    k = 1 recovers the VAF.  Bound (Lemma 1, eq. (13); Theorem 1):
        t_s <= 1/(gamma a^k (pk - 1)) + 1/(gamma b^k (1 - qk))
    (read on the rendered PDF; c1, c2 >= 0 is implied by the proof).
    Defaults (a = b = 1, p = 1, q = 0.25, k = 2, c1 = c2 = 1): ASSUMPTION (the
    paper's simulations vary k in {0.5, 1, 2} and print no a, b, p, q)."""

    key = "pvaf1"
    slope0 = np.inf
    finite_time = True
    convergence = "predefined"
    source = "Zhou & Tan 2022 (doi:10.24425/bpasts.2022.141307)"
    provenance = "primary"

    def __init__(self, a=1.0, b=1.0, p=1.0, q=0.25, k=2.0, c1=1.0, c2=1.0):
        _require(min(a, b, p, q, k) > 0, "need a, b, p, q, k > 0")
        _require(p * k > 1 and q * k < 1, "need pk > 1 and qk < 1")
        _require(c1 >= 0 and c2 >= 0, "need c1, c2 >= 0")
        self.a, self.b, self.p, self.q, self.k = float(a), float(b), float(p), float(q), float(k)
        self.c1, self.c2 = float(c1), float(c2)

    def __call__(self, e):
        e = np.asarray(e, dtype=float)
        s, x = np.sign(e), np.abs(e)
        with np.errstate(over="ignore"):
            core = (self.a * x ** self.p + self.b * x ** self.q) ** self.k
        return (core + self.c2) * s + self.c1 * e

    def upper_bound_time(self, e0, gamma=1.0):
        a, b, p, q, k = self.a, self.b, self.p, self.q, self.k
        return (1.0 / (a ** k * (p * k - 1.0)) + 1.0 / (b ** k * (1.0 - q * k))) / gamma

    def uniform_bound(self, gamma=1.0):
        return self.upper_bound_time(None, gamma)

    def params(self):
        return {"a": self.a, "b": self.b, "p": self.p, "q": self.q, "k": self.k,
                "c1": self.c1, "c2": self.c2}


@register
class PowerVersatile2(Activation):
    """Power-versatile activation PVAF-2 (Zhou & Tan 2022, eq. (7)):
        Phi(e) = (a |e|^p + b)^k sign(e) + d1 e + d2 sign(e),
        a, b, p, k > 0, pk > 1, d1, d2 >= 0  (jump b^k + d2 at 0).
    Bound (Lemma 2, eq. (15); Theorem 4):
        t_s <= (1/(gamma b^k)) (b/a)^(1/p) (1 + 1/(pk - 1)).
    PROVENANCE: eq. (15) read on the rendered PDF page (verifier); it also
    follows by splitting at |e| = (b/a)^(1/p), and the gain enters as
    a, b -> gamma^(1/k) a, gamma^(1/k) b (proof of Thm. 1).  The paper states
    no a, b, p values (its simulations vary k in {0.5, 1, 2}); defaults
    (a = b = 1, p = 1, k = 2, d1 = d2 = 1): ASSUMPTION."""

    key = "pvaf2"
    slope0 = np.inf
    finite_time = True
    convergence = "predefined"
    source = "Zhou & Tan 2022 (doi:10.24425/bpasts.2022.141307)"
    provenance = "primary"

    def __init__(self, a=1.0, b=1.0, p=1.0, k=2.0, d1=1.0, d2=1.0):
        _require(min(a, b, p, k) > 0, "need a, b, p, k > 0")
        _require(p * k > 1, "need pk > 1")
        _require(d1 >= 0 and d2 >= 0, "need d1, d2 >= 0")
        self.a, self.b, self.p, self.k = float(a), float(b), float(p), float(k)
        self.d1, self.d2 = float(d1), float(d2)

    def __call__(self, e):
        e = np.asarray(e, dtype=float)
        s, x = np.sign(e), np.abs(e)
        with np.errstate(over="ignore"):
            core = (self.a * x ** self.p + self.b) ** self.k
        return (core + self.d2) * s + self.d1 * e

    def upper_bound_time(self, e0, gamma=1.0):
        a, b, p, k = self.a, self.b, self.p, self.k
        return (b / a) ** (1.0 / p) * (1.0 + 1.0 / (p * k - 1.0)) / (b ** k * gamma)

    def uniform_bound(self, gamma=1.0):
        return self.upper_bound_time(None, gamma)

    def params(self):
        return {"a": self.a, "b": self.b, "p": self.p, "k": self.k, "d1": self.d1, "d2": self.d2}


@register
class PredefinedTimeExp(Activation):
    """Exponential predefined-time activation, F2 of Li, Liao, Xiao, Lu,
    Neurocomputing 337:262-273, 2019 (first predefined-time ZNN):
        Phi(e) = k1 exp(|e|^p) |e|^(1-p) sign(e) / p + k2 e + k3 sign(e),
        0 < p < 1, k1 > 0, k2, k3 >= 0
    (k1, k2, k3 are gamma1, gamma2, gamma3 of the paper; formula and bound as
    transcribed in Wang, Hua, Khan, Biomimetics 10:279, 2025).  Bound:
        t_c <= 1/(gamma k1)   for every e0.
    DERIVED: with k2 = k3 = 0 the settling time is exactly
    (1 - exp(-|e0|^p))/(gamma k1) (the Sanchez-Torres et al. predefined-time
    system).  Defaults p = 0.5, k1 = k2 = k3 = 1: ASSUMPTION."""

    key = "predefined_exp"
    slope0 = np.inf
    finite_time = True
    convergence = "predefined"
    source = "Li, Liao, Xiao, Lu 2019 (doi:10.1016/j.neucom.2019.01.072)"
    provenance = "secondary"

    def __init__(self, p=0.5, k1=1.0, k2=1.0, k3=1.0):
        _require(0 < p < 1, "need 0 < p < 1")
        _require(k1 > 0 and k2 >= 0 and k3 >= 0, "need k1 > 0 and k2, k3 >= 0")
        self.p, self.k1, self.k2, self.k3 = float(p), float(k1), float(k2), float(k3)

    def __call__(self, e):
        e = np.asarray(e, dtype=float)
        s, x = np.sign(e), np.abs(e)
        with np.errstate(over="ignore", invalid="ignore"):
            core = self.k1 * np.exp(x ** self.p) * x ** (1.0 - self.p) / self.p
            core = np.where(x > 0, core, 0.0)
        return (core + self.k3) * s + self.k2 * e

    def upper_bound_time(self, e0, gamma=1.0):
        return 1.0 / (gamma * self.k1)

    def uniform_bound(self, gamma=1.0):
        return self.upper_bound_time(None, gamma)

    def params(self):
        return {"p": self.p, "k1": self.k1, "k2": self.k2, "k3": self.k3}


# ===========================================================================
# 4. Time-varying ("strictly predefined-time") activation
# ===========================================================================
@register
class StrictlyPredefinedTime(Activation):
    """Time-varying activation of the strictly predefined-time ZNN
    (SPTC-NT-ZNN, Li, Guo, Ma, Pan, IEEE TIE 71(1):798-809, 2024):
        Phi(e, t) = e / (t_c - t)                         , t <  t_c
                  = e + |e|^p sign(e) + xi sign(e)        , t >= t_c
        0 < p < 1, xi >= 0  (xi dominates the noise bound in the paper).
    With e' = -gamma Phi(e, t): e(t) = e0 ((t_c - t)/t_c)^gamma for t < t_c, so
    e(t_c) = 0 for EVERY e0 and gamma > 0 (settling time = t_c exactly).
    PROVENANCE: formula as transcribed in Yang et al. (arXiv:2503.01857,
    Table 1) and Wang, Hua, Khan 2025; the model-level implementation is
    ``sptc_nt_znn`` in models/variants.py.  Use ``at(t)`` (or call with ``t=``)
    -- calling without a time raises.  The design-formula models
    (DesignFormulaZNN and its subclasses, GNN, hntznn) freeze time-varying
    activations with ``at(ctx.t)`` (integration change), so
    ``make_model("znn", activation="strict_predefined")`` is the SPTC-NT-ZNN
    design formula with constant gain.  Numerical guard (as in models/gains.py): t >= t_c -
    1e-12 max(1, t_c) is treated as the post-t_c phase.  slope0 is the t = 0
    value 1/t_c.  Defaults t_c = 1, p = 0.5, xi = 0: ASSUMPTION (xi is
    noise-dependent)."""

    key = "strict_predefined"
    finite_time = True
    time_varying = True
    convergence = "strictly-predefined"
    source = "Li, Guo, Ma, Pan 2024 (doi:10.1109/TIE.2023.3241393)"
    provenance = "secondary"

    def __init__(self, t_c=1.0, p=0.5, xi=0.0):
        _require(t_c > 0, "need t_c > 0")
        _require(0 < p < 1 and xi >= 0, "need 0 < p < 1 and xi >= 0")
        self.t_c, self.p, self.xi = float(t_c), float(p), float(xi)
        self.guard = 1e-12 * max(1.0, self.t_c)
        self.slope0 = 1.0 / self.t_c

    def _pre(self, t):
        return float(t) < self.t_c - self.guard

    def __call__(self, e, t=None):
        if t is None:
            raise ValueError("strict_predefined is time-varying: call act(e, t=...) or act.at(t)(e)")
        e = np.asarray(e, dtype=float)
        if self._pre(t):
            return e / (self.t_c - float(t))
        return e + _spow(e, self.p) + self.xi * np.sign(e)

    def slope_at(self, t):
        return 1.0 / (self.t_c - float(t)) if self._pre(t) else np.inf

    def jump_at(self, t):
        return 0.0 if self._pre(t) else self.xi

    def jump0(self):
        return 0.0

    def prox(self, e, h, t=None, iters: int = 80):
        if t is None:
            raise ValueError("strict_predefined is time-varying: pass t")
        e = np.asarray(e, dtype=float)
        if self._pre(t):
            return e / (1.0 + h / (self.t_c - float(t)))
        return Activation.prox(self.at(t), e, h, iters)

    def at(self, t):
        return _FrozenAt(self, t)

    def upper_bound_time(self, e0, gamma=1.0):
        return self.t_c

    def uniform_bound(self, gamma=1.0):
        return self.t_c

    def settling_time(self, e0, gamma=1.0):
        return self.t_c if _amax(e0) > 0 else 0.0

    def params(self):
        return {"t_c": self.t_c, "p": self.p, "xi": self.xi}


class _FrozenAt(Activation):
    """A time-varying activation evaluated at a fixed time t (not registered)."""

    time_varying = False

    def __init__(self, parent, t):
        self.parent, self.t = parent, float(t)
        self.key = parent.key
        self.slope0 = parent.slope_at(self.t)
        self.finite_time = parent.finite_time
        self.convergence = parent.convergence

    def __call__(self, e):
        return self.parent(e, self.t)

    def jump0(self):
        return float(self.parent.jump_at(self.t))

    def prox(self, e, h, iters: int = 80):
        if self.parent._pre(self.t):
            return self.parent.prox(e, h, t=self.t)
        return Activation.prox(self, e, h, iters)

    def params(self):
        return {**self.parent.params(), "t": self.t}
