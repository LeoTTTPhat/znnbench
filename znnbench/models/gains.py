"""Time-varying gain schedules g(t) for the ZNN design formula

    dE/dt = -gamma * g(t) * Phi(E)          (DesignFormulaZNN, models/znn.py)

ZNN-Bench convention: ``ctx.gamma`` (the base gain swept by the benchmark,
h = gamma * tau) multiplies the schedule.  The schedules below are written
exactly as the *time-varying design parameter* of the source paper, so a
source model is reproduced with ``gamma = 1`` (the paper's parameter is the
whole g(t)).  Every schedule accepts an optional saturation ``cap``:
``g_capped(t) = min(g(t), cap)``.  The cap is a ZNN-Bench addition (NOT in
any source paper) that keeps the effective step h(t) = gamma*tau*g(t) bounded
when an unbounded schedule is discretized with a fixed sampling period.

Sources (full citations and equations in docs/spec/variants.md):

* ``PowerVPGain``      g(t) = zeta + t^p  (zeta defaults to p, i.e. t^p + p)
    - Zhang, Zheng, Weng, Mao, Lu, Xiao, IEEE Trans. Cybern. 48(11):3135-3148,
      2018, doi:10.1109/TCYB.2017.2760883 (VP-CDNN, t^kappa + kappa);
    - Zhang, Zheng, IEEE Trans. Cybern. 49(10):3627-3639, 2019,
      doi:10.1109/TCYB.2018.2841970 (CVP-CDNN, t^p + p);
    - Zhang, Fu, Yan, Jin, Xiao, Sun, Yu, Li, IEEE/ASME Trans. Mechatron.
      23(2):679-689, 2018, doi:10.1109/TMECH.2018.2799724 (zeta + t^gamma).
* ``ExpVPGain``        g(t) = p^t + p
    - Zhang, Zheng, Wang, Neurocomputing 338:126-138, 2019,
      doi:10.1016/j.neucom.2019.01.058 ("exponential-enhanced-type" VP-RNN).
* ``EpsExpGain``       g(t) = eps * exp(t)
    - Zhang, Zheng, Li, Deng, Xiao, Huang, Neurocomputing 319:74-83, 2018,
      doi:10.1016/j.neucom.2018.07.005 (FT-VP-CDNN).
* ``ArccotExpGain``    g(t) = beta * exp(lam1 * arccot(t) + lam2 * t)
    - Xiao, He, IEEE Trans. Ind. Inform. 17(11):7513-7522, 2021,
      doi:10.1109/TII.2021.3058343.
* ``PredefinedTimeGain``  g(t) = 1 / (t_c - t)  for t < t_c
    - SPTC-NT-ZNN, Li, Guo, Ma, Pan, IEEE Trans. Ind. Electron. 71(1):798-809,
      2024, doi:10.1109/TIE.2023.3241393 (activation x/(t_c - t) for t < t_c,
      as transcribed in Table 1 of Yang et al., Neural Networks 2025).
* ``ConformableGain``  g(t) = t^(alpha - 1)
    - conformable fractional derivative W_alpha(f)(t) = t^(1-alpha) f'(t)
      (Khalil et al., J. Comput. Appl. Math. 264:65-70, 2014) turned into an
      integer-order time-varying gain by Yang et al., Neural Networks 186:107279,
      2025, doi:10.1016/j.neunet.2025.107279, Eqs. (11)-(12).

ASSUMPTION flags (see spec): all VP schedules were transcribed from the
review Advances in Zeroing Neural Networks (Biomimetics 10(5):279, 2025,
doi:10.3390/biomimetics10050279): Table 2 (zeta + t^gamma, p^t + p, t^p + p,
arccot-exp) and its running text (t^kappa + kappa for TCYB 2018, eps*e^t for
FT-VP-CDNN); the primary full texts were not accessible.  Default parameter values are
ZNN-Bench choices unless stated otherwise.
"""
from __future__ import annotations

import numpy as np

from .znn import ConstantGain  # noqa: F401  (re-exported: g(t) = 1)


class GainSchedule:
    """Base class: ``__call__(t)`` returns g(t) (scalar or array), with the
    optional saturation ``cap`` applied.  Subclasses implement ``raw``."""

    key = "base"
    source = ""

    def __init__(self, cap=None):
        if cap is not None and not cap > 0:
            raise ValueError("cap must be positive or None")
        self.cap = cap

    def raw(self, t):
        raise NotImplementedError

    def __call__(self, t):
        with np.errstate(over="ignore"):
            g = self.raw(t)
        if self.cap is not None:
            g = np.minimum(g, self.cap)
        return g if np.ndim(g) else float(g)

    def params(self) -> dict:
        return {}

    def describe(self) -> dict:
        return {"key": self.key, "cap": self.cap, **self.params()}


class PowerVPGain(GainSchedule):
    """Power-type varying parameter g(t) = zeta + t^p (p > 0, zeta > 0).

    zeta = None gives the most common form t^p + p (VP-CDNN, Zhang et al.
    TCYB 2018 / TCYB 2019); a separate zeta gives the TMECH 2018 form
    zeta + t^gamma.  With a linear activation the error obeys
    E(t) = E(0) exp(-gamma * (zeta t + t^(p+1)/(p+1))) ("super-exponential").
    """

    key = "power_vp"

    def __init__(self, p: float = 1.0, zeta=None, cap=None):
        super().__init__(cap)
        if not p > 0:
            raise ValueError("p must be > 0")
        self.p = float(p)
        self.zeta = self.p if zeta is None else float(zeta)

    def raw(self, t):
        t = np.maximum(np.asarray(t, dtype=float), 0.0)
        return self.zeta + t ** self.p

    def integral(self, t):
        """int_0^t g(s) ds (uncapped), used by the tests."""
        t = np.asarray(t, dtype=float)
        return self.zeta * t + t ** (self.p + 1) / (self.p + 1)

    def params(self):
        return {"p": self.p, "zeta": self.zeta}


class ExpVPGain(GainSchedule):
    """Exponential-type varying parameter g(t) = p^t + p (p > 0)."""

    key = "exp_vp"

    def __init__(self, p: float = 2.0, cap=None):
        super().__init__(cap)
        if not p > 0:
            raise ValueError("p must be > 0")
        self.p = float(p)

    def raw(self, t):
        t = np.asarray(t, dtype=float)
        return self.p ** t + self.p

    def integral(self, t):
        t = np.asarray(t, dtype=float)
        if self.p == 1.0:
            return 2.0 * t
        return (self.p ** t - 1.0) / np.log(self.p) + self.p * t

    def params(self):
        return {"p": self.p}


class EpsExpGain(GainSchedule):
    """g(t) = eps * exp(t) (eps > 0), FT-VP-CDNN of Zhang et al. 2018."""

    key = "eps_exp"

    def __init__(self, eps: float = 1.0, cap=None):
        super().__init__(cap)
        if not eps > 0:
            raise ValueError("eps must be > 0")
        self.eps = float(eps)

    def raw(self, t):
        return self.eps * np.exp(np.asarray(t, dtype=float))

    def params(self):
        return {"eps": self.eps}


class ArccotExpGain(GainSchedule):
    """g(t) = beta * exp(lam1 * arccot(t) + lam2 * t), beta, lam1, lam2 > 0
    (Xiao & He, TII 2021).  arccot(t) = pi/2 - arctan(t) for t >= 0, so
    g(0) = beta * exp(lam1 * pi / 2): a large initial gain that first decays
    and then grows exponentially."""

    key = "arccot_exp"

    def __init__(self, beta: float = 1.0, lam1: float = 1.0, lam2: float = 1.0, cap=None):
        super().__init__(cap)
        if not (beta > 0 and lam1 > 0 and lam2 > 0):
            raise ValueError("beta, lam1, lam2 must be > 0")
        self.beta, self.lam1, self.lam2 = float(beta), float(lam1), float(lam2)

    def raw(self, t):
        t = np.asarray(t, dtype=float)
        arccot = 0.5 * np.pi - np.arctan(t)
        return self.beta * np.exp(self.lam1 * arccot + self.lam2 * t)

    def params(self):
        return {"beta": self.beta, "lam1": self.lam1, "lam2": self.lam2}


class PredefinedTimeGain(GainSchedule):
    """Strictly predefined-time gain g(t) = 1 / (t_c - t) for t < t_c.

    With a linear activation, dE/dt = -gamma E / (t_c - t) has the exact
    solution E(t) = E(0) ((t_c - t) / t_c)^gamma, i.e. E(t_c) = 0 for every
    gamma > 0 and every initial error ("strictly" predefined: the settling
    time equals t_c).  The gain blows up as t -> t_c, so a fixed-step
    discretization becomes unstable during the last ~gamma*tau/2 seconds
    unless ``cap`` is set.  For t >= t_c the schedule returns ``after``
    (default 1.0); in SPTC-NT-ZNN the post-t_c phase is handled by switching
    the activation instead (see variants.SPTCNTZNN).

    A numerical guard treats t >= t_c - guard (guard = 1e-12 * max(1, t_c))
    as t >= t_c, so that a sampling instant that is t_c up to round-off does
    not produce a 1e16 gain (ZNN-Bench implementation detail).
    """

    key = "predefined_time"

    def __init__(self, t_c: float = 1.0, cap=None, after: float = 1.0):
        super().__init__(cap)
        if not t_c > 0:
            raise ValueError("t_c must be > 0")
        self.t_c = float(t_c)
        self.after = float(after)
        self.guard = 1e-12 * max(1.0, self.t_c)

    def before(self, t):
        return np.asarray(t, dtype=float) < self.t_c - self.guard

    def raw(self, t):
        t = np.asarray(t, dtype=float)
        pre = self.before(t)
        safe = np.where(pre, self.t_c - t, 1.0)
        return np.where(pre, 1.0 / safe, self.after)

    def params(self):
        return {"t_c": self.t_c, "after": self.after}


class ConformableGain(GainSchedule):
    """Conformable fractional-order gain g(t) = t^(alpha - 1), 0 < alpha <= 1.

    Replacing dE/dt by the conformable derivative W_alpha(E)(t) =
    t^(1-alpha) dE/dt in dE/dt = -gamma Phi(E) gives the equivalent integer-
    order ODE dE/dt = -gamma t^(alpha-1) Phi(E) (Yang et al. 2025, Eqs. 11-12).
    For alpha < 1 the gain is singular at t = 0.  ASSUMPTION (ZNN-Bench): the
    gain is evaluated at max(t, t_floor), t_floor = 1e-3 by default, so that
    the first sampling instant t_0 = 0 is finite.
    """

    key = "conformable"

    def __init__(self, alpha: float = 0.5, t_floor: float = 1e-3, cap=None):
        super().__init__(cap)
        if not 0 < alpha:
            raise ValueError("alpha must be > 0")
        if not t_floor > 0:
            raise ValueError("t_floor must be > 0")
        self.alpha, self.t_floor = float(alpha), float(t_floor)

    def raw(self, t):
        t = np.maximum(np.asarray(t, dtype=float), self.t_floor)
        return t ** (self.alpha - 1.0)

    def params(self):
        return {"alpha": self.alpha, "t_floor": self.t_floor}


SCHEDULES = {c.key: c for c in (PowerVPGain, ExpVPGain, EpsExpGain, ArccotExpGain,
                                PredefinedTimeGain, ConformableGain)}


def get(key: str, **kw):
    """Instantiate a gain schedule by key ('constant' returns ConstantGain)."""
    if key == "constant":
        return ConstantGain()
    return SCHEDULES[key](**kw)
