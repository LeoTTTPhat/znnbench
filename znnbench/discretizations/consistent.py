"""Consistent (implicit / semi-implicit / exact) discretizations of finite-,
fixed- and predefined-time ZNN error dynamics.

Background
----------
A design-formula ZNN imposes the element-wise error dynamics

    dE/dt = -gamma * Phi(E),                                         (ZNN)

and realises them through  J x_dot = -gamma Phi(E) - dE/dt|_x .  Every
causal look-ahead discretization (Euler, Taylor-Zhang, ...) turns (ZNN) into a
*one-step error map*: on a time-invariant problem (constant data, constant J)
the forward-Euler update gives exactly

    E_{k+1} = E_k - h Phi(E_k),          h = gamma * tau,

which is the explicit Euler scheme of (ZNN).  For non-Lipschitz activations
(sign, sign-bi-power, fixed-/predefined-time activations) this map is not
*consistent* in the sense of Polyakov, Efimov & Brogliato (SICON 2019,
Def. 2.1/2.2): it chatters (period-2 cycles of amplitude O(h^{1/(1-p)}) for a
|e|^p sign(e) term) or blows up for large errors (positive-degree terms).

This module replaces the explicit rate -gamma Phi(E_k) by the rate
(Psi_h(E_k) - E_k)/tau, where Psi_h is a *consistent* one-step error map:

  'resolvent'     backward Euler of (ZNN): Psi_h = (I + h Phi)^{-1}  (the
                  activation's `prox`; Acary & Brogliato 2010; Huber, Acary &
                  Brogliato 2016).  Finite-step exact only if Phi has a jump
                  at 0; for continuous non-Lipschitz Phi (sign-bi-power) it is
                  merely (super-linearly) asymptotically convergent -- the
                  inconsistency shown in PEB 2019, Sec. 1.1.
  'peb'           homogeneity-based scheme of Polyakov, Efimov & Brogliato
                  (2019): transform each homogeneous term to its homogeneous
                  coordinate y = ||e||_d d(-ln||e||_d) e (Eq. (3.12)/(4.1)),
                  step it implicitly (negative degree, Thm 4.1) or
                  semi-implicitly (positive degree, Thm 5.1), map back.  Sums
                  of homogeneous terms (sign-bi-power, fixed-time activations)
                  are handled by Lie splitting (ASSUMPTION, novel extension;
                  see docs/spec/consistent.md).
  'exact'         exact flow of the scalar error ODE over one step, obtained
                  in the settling-time coordinate w = T(|e|) sign(e), where
                  w_dot = -sign(w) is discretized exactly by implicit Euler
                  (Jimenez-Rodriguez et al., IFAC-PapersOnLine 2020, Eq. (7)
                  for predefined-time systems; closed forms derived here for
                  power, Bernoulli-type and bi-power activations via the
                  regularised incomplete beta function, cf. Aldana-Lopez et
                  al. 2019, Thm 1).
  'semi_implicit' linearly implicit (frozen-coefficient) Euler, Phi(e) =
                  kappa(e) e with kappa frozen at e_k (PEB 2019, Sec. 1.2 and
                  Eq. (5.3)); the jump part of Phi (if any) is stepped by the
                  exact soft threshold.  Closed form for ANY activation.

On a time-invariant problem every scheme in this module realises
E_{k+1} = Psi_h(E_k) up to round-off, so a finite-time Psi_h reaches the exact
solution in finitely many steps.  On a time-varying problem the look-ahead
formula's local truncation error enters as a perturbation and the residual
keeps the order of the underlying formula.

Look-ahead formula with error memory (implicit Taylor-Zhang)
-----------------------------------------------------------
For a one-step-ahead formula  tau x_dot_k ~ c_{-1} x_{k+1} + sum_j c_j x_{k-j}
the update is x_{k+1} = (tau x_dot_k - sum_j c_j x_{k-j}) / c_{-1}.  Combining
it naively with a consistent rate leaves the formula's x-memory acting on the
error-correction motion; for Taylor-Zhang inside the dead zone of a sign-type
map the error then obeys E_{k+1} = (E_k - 2E_{k-1} + E_{k-2})/2, whose
characteristic polynomial (z - 1/2)(z^2 + 1) is only marginally stable.  The
'error-memory' variant (default) uses instead the desired rate

    tau r_k = c_{-1} Psi_h(E_k) + sum_{j>=0} c_j E_{k-j},            (EM)

so that, for constant data, E_{k+1} = Psi_h(E_k) EXACTLY whatever the formula
(the memory cancels), while along the exact solution the formula keeps its
truncation order.  Equivalently (Taylor-Zhang)

    x_{k+1} = x_k + J_k^{-1}(Psi_h(E_k) - E_k) - tau J_k^{-1} dE/dt
              + (1/2) Delta^2 xhat_k,     xhat_{k-j} = x_{k-j} - J_k^{-1} E_{k-j},

i.e. the Taylor-Zhang curvature term acts on Newton-projected past states.
This construction is NOVEL (not in the cited papers) -- see the spec.

Interface
---------
Error maps need the activation and the error, which only the model knows.
The discretizers pass an :class:`ErrorMapRequest` object as the ``implicit``
argument of ``f``; it is truthy, so any implicit-aware model that ignores it
falls back to its resolvent rate (with a warning).  :class:`ConsistentZNN`
(registered model key ``"znn_consistent"``) is a DesignFormulaZNN that honours
the request; a 3-line change to DesignFormulaZNN making the base ``"znn"``
model honour it is requested in docs/spec/CORE_CHANGE_REQUESTS.md.
All discretizers are causal: f is called once per step, at (t_k, x_k, k).
Scheme names equal their registry keys (e.g. 'exact_flow_euler').

Activations
-----------
'peb' and 'exact' need the structure of Phi: activations exposing
``power_terms()`` / ``exact_step()`` (here: PowerSum, key 'signed_power_sum',
and PredefinedTimeJR, key 'jr_predefined'; both resolvable by key through
`get_activation` and model 'znn_consistent', but kept out of the shared
activations.REGISTRY), plus the activations.py zoo members
that are signed power sums ('linear', 'sign', 'sign_power', 'sign_bi_power',
'sfaf', 'wsbp', 'vaf', 'fixed_time', 'power', 'power_sum'), recognised from
their parameters and checked numerically against the activation.  Other
activations fall back to 'resolvent' (warned once).
"""
from __future__ import annotations

import warnings
import weakref
from collections import deque

import numpy as np
from scipy import special

from .. import activations as acts
from ..core import Discretizer
from ..models.znn import DesignFormulaZNN

_TINY_JUMP = 1e-12        # Phi(0+) below this is treated as "no jump"
_WARNED = set()


def _warn_once(key, msg):
    if key not in _WARNED:
        _WARNED.add(key)
        warnings.warn(msg, RuntimeWarning, stacklevel=3)


def _spow(e, p):
    """sign(e)|e|^p with 0 -> 0 (p = 0 gives sign(e))."""
    return np.sign(e) * np.abs(e) ** p


# ===========================================================================
# activations used as test vehicles (literature classes not yet in activations.py)
# ===========================================================================
def _bipower_sup_T(a1, p1, a2, p2):
    """Least upper bound of the settling time of e' = -(a1|e|^p1 + a2|e|^p2) sign(e),
    p1 < 1 < p2 (Aldana-Lopez et al. 2019, Thm 1, Eq. (6) with k = 1):
    Gamma(m_p) Gamma(m_q) (a1/a2)^{m_p} / (a1 (p2 - p1)),  m_p = (1-p1)/(p2-p1),
    m_q = 1 - m_p;  Gamma(m)Gamma(1-m) = pi / sin(pi m)."""
    q = p2 - p1
    beta = (1.0 - p1) / q
    return (a1 / a2) ** beta * np.pi / (a1 * q * np.sin(np.pi * beta))


class PowerSum(acts.Activation):
    """Signed power sum Phi(e) = sum_i a_i sign(e)|e|^{p_i}, a_i > 0, p_i >= 0
    (activation key ``"signed_power_sum"``; the key ``"power_sum"`` belongs to
    the Yang & Zhang 2013 power-sum activation sum_k e^(2k-1) in activations.py).

    Covers the signum activation (p = 0; Utkin's sliding mode, the
    'signum-function array' ZNN), power terms, the sign-bi-power activation of
    Li, Chen & Liu, Neural Process. Lett. 37 (2013) 189-205 (terms
    (1/2, r), (1/2, 1/r)) and the fixed-time bi-power vector field
    alpha|x|^p + beta|x|^q, p < 1 < q (Polyakov, IEEE TAC 57 (2012) 2106-2110;
    Aldana-Lopez et al., IJRNC 29 (2019) 4135-4148, Eq. (1) with k = 1).
    It is a test vehicle for the consistent maps, not a new activation.
    """

    key = "signed_power_sum"
    source = ("generic container: sign (Zhang et al. 2017, doi:10.1016/j.ipl.2017.04.008), "
              "sign-bi-power (Li, Chen, Liu 2013), bi-power fixed-time field (Polyakov 2012; "
              "Aldana-Lopez et al. 2019)")
    provenance = "derived"

    def __init__(self, terms=((1.0, 0.0),)):
        merged = {}
        for a, p in terms:
            a, p = float(a), float(p)
            if a < 0 or p < 0:
                raise ValueError("PowerSum needs a_i >= 0 and p_i >= 0")
            if a > 0:
                merged[p] = merged.get(p, 0.0) + a
        if not merged:
            raise ValueError("PowerSum needs at least one positive term")
        self.terms = tuple(sorted(((a, p) for p, a in merged.items()), key=lambda t: t[1]))
        ps = [p for _, p in self.terms]
        self.finite_time = min(ps) < 1
        self.slope0 = np.inf if self.finite_time else sum(a for a, p in self.terms if p == 1)
        if self.finite_time:
            self.convergence = "fixed" if max(ps) > 1 else "finite"
        else:
            self.convergence = "exponential" if min(ps) == 1 else "asymptotic"

    def __call__(self, e):
        e = np.asarray(e, dtype=float)
        out = np.zeros_like(e)
        with np.errstate(over="ignore"):
            for a, p in self.terms:
                out = out + a * _spow(e, p)
        return out

    def jump0(self):
        return float(sum(a for a, p in self.terms if p == 0))

    def power_terms(self):
        return list(self.terms)

    def uniform_bound(self, gamma=1.0):
        """sup_{e0} T(e0): finite iff there are terms with p < 1 and p > 1.  Dropping
        positive terms can only lengthen the settling time, so the smallest
        Aldana-Lopez Thm 1 value over pairs (p_i < 1 < p_j) is a valid bound; it is
        the exact least upper bound for two-term fields."""
        lo = [(a, p) for a, p in self.terms if p < 1]
        hi = [(a, p) for a, p in self.terms if p > 1]
        if not lo or not hi:
            return np.inf
        return min(_bipower_sup_T(a1, p1, a2, p2) for a1, p1 in lo for a2, p2 in hi) / gamma

    def params(self):
        return {"terms": [list(t) for t in self.terms]}


_KAPPAS = ("arctan", "exp", "rational")


class PredefinedTimeJR(acts.Activation):
    """Predefined-time activation of Jimenez-Rodriguez et al. (IFAC-PapersOnLine
    53(2) 2020 628-633, Eq. (1); class of Aldana-Lopez et al. 2019 and
    Jimenez-Rodriguez et al. 2019):

        Phi(e) = 1/(rho1 (1 - rho2)) * kappa(|e|)^{rho2} / kappa'(|e|) * sign(e),

    rho1 > 0, 0 <= rho2 < 1, kappa in K^1 (strictly increasing, kappa(0) = 0,
    kappa(inf) = 1).  Settling time T(e0) = rho1 kappa(|e0|)^{1-rho2} < rho1,
    so with gain gamma the ZNN error vanishes before t = rho1/gamma.

    kappa families (JR Example 4 for mu = 1; 0 < mu < 1 is the composition
    kappa(r^mu), still in K^1 -- ASSUMPTION, see spec; mu > 1 is rejected
    because kappa'(0) = 0 would make Phi unbounded near 0):
      'arctan'   kappa = (2/pi) arctan(a r^mu)   (rho2 = 0 gives the symmetric
                 bi-power pi/(2 a mu rho1) (|e|^{1-mu} + a^2 |e|^{1+mu}))
      'exp'      kappa = 1 - exp(-a r^mu).  REPARAMETRISED: JR Example 4 writes
                 1 - a_JR^{-r} with a_JR > 1, i.e. a = ln(a_JR) here (a > 0).
                 rho2 = 0 gives the Sanchez-Torres et al. 2018 class
                 exp(a|e|^mu)|e|^{1-mu}/(a mu rho1) = 'predefined_exp' with k2 = k3 = 0
      'rational' kappa = r^mu / (r^mu + a)     (JR: r/(r+a), a > 0)
    """

    key = "jr_predefined"
    finite_time = True
    convergence = "predefined"      # strong predefined time: sup T(e0) = rho1 (JR Sec. 2.2)
    source = ("Jimenez-Rodriguez et al., IFAC-PapersOnLine 53(2):628-633, 2020, Eq. (1), "
              "doi:10.1016/j.ifacol.2020.12.806")
    provenance = "primary"

    def __init__(self, kappa="arctan", a=1.0, mu=1.0, rho1=1.0, rho2=0.0):
        if kappa not in _KAPPAS:
            raise ValueError(f"kappa must be one of {_KAPPAS}")
        if not (rho1 > 0 and 0 <= rho2 < 1 and a > 0 and 0 < mu <= 1):
            # mu > 1 would give kappa'(0) = 0, i.e. an unbounded Phi near 0
            raise ValueError("need rho1 > 0, 0 <= rho2 < 1, a > 0, 0 < mu <= 1")
        self.kappa_name, self.a, self.mu = kappa, float(a), float(mu)
        self.rho1, self.rho2 = float(rho1), float(rho2)
        self.slope0 = np.inf

    # kappa, kappa', kappa^{-1}
    def kappa(self, r):
        r = np.asarray(r, dtype=float)
        u = self.a * r ** self.mu
        if self.kappa_name == "arctan":
            return (2.0 / np.pi) * np.arctan(u)
        if self.kappa_name == "exp":
            return -np.expm1(-u)
        return r ** self.mu / (r ** self.mu + self.a)

    def dkappa(self, r):
        r = np.asarray(r, dtype=float)
        a, mu = self.a, self.mu
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            rm1 = r ** (mu - 1.0)
            if self.kappa_name == "arctan":
                return (2.0 / np.pi) * a * mu * rm1 / (1.0 + (a * r ** mu) ** 2)
            if self.kappa_name == "exp":
                return a * mu * rm1 * np.exp(-a * r ** mu)
            return a * mu * rm1 / (r ** mu + a) ** 2

    def kappa_inv(self, u):
        u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
        a, mu = self.a, self.mu
        with np.errstate(divide="ignore", over="ignore"):
            if self.kappa_name == "arctan":
                base = np.tan(0.5 * np.pi * u) / a
            elif self.kappa_name == "exp":
                base = -np.log1p(-u) / a
            else:
                base = a * u / (1.0 - u)
        return base ** (1.0 / mu)

    def __call__(self, e):
        e = np.asarray(e, dtype=float)
        r = np.abs(e)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            val = self.kappa(r) ** self.rho2 / (self.rho1 * (1.0 - self.rho2) * self.dkappa(r))
        # limit at r = 0: kappa'(0+) may be finite (jump) or infinite (continuous)
        val = np.where(r > 0, val, self._phi0())
        return np.sign(e) * val

    def _phi0(self):
        if self.rho2 > 0 or self.mu < 1:      # continuous, non-Lipschitz at 0
            return 0.0
        k0 ={"arctan": 2.0 * self.a / np.pi, "exp": self.a, "rational": 1.0 / self.a}[self.kappa_name]
        return 1.0 / (self.rho1 * k0)

    def jump0(self):
        return float(self._phi0())

    def settling_time(self, e0, gamma=1.0):
        """Exact settling time T(e0) = rho1 kappa(|e0|)^{1-rho2} / gamma (JR Sec. 2.2),
        with max_i |e0_i| as in the activations.py contract."""
        a = float(np.max(np.abs(np.asarray(e0, dtype=float)))) if np.size(e0) else 0.0
        return float(self.rho1 * self.kappa(a) ** (1.0 - self.rho2)) / gamma

    def upper_bound_time(self, e0, gamma=1.0):
        """JR Sec. 2.2: T(e0) < rho1 for every e0 (predefined time Tc = rho1)."""
        return self.rho1 / gamma

    def uniform_bound(self, gamma=1.0):
        return self.rho1 / gamma

    def exact_step(self, e, h):
        """Exact discretization, JR Eq. (7):
        e_{k+1} = kappa^{-1}( max{kappa(|e_k|)^{1-rho2} - h/rho1, 0}^{1/(1-rho2)} ) sign(e_k).
        The result is clipped to |e_{k+1}| <= |e_k| (the exact flow is non-increasing);
        this only acts when h/rho1 is below the rounding error of kappa(|e_k|)."""
        e = np.asarray(e, dtype=float)
        s = np.abs(e)
        w = self.kappa(s) ** (1.0 - self.rho2) - h / self.rho1
        w = np.maximum(w, 0.0)
        return np.sign(e) * np.minimum(self.kappa_inv(w ** (1.0 / (1.0 - self.rho2))), s)

    def params(self):
        return {"kappa": self.kappa_name, "a": self.a, "mu": self.mu, "rho1": self.rho1,
                "rho2": self.rho2}


# The two test vehicles are deliberately NOT inserted into the shared
# activations.REGISTRY: tests/test_activations.py iterates that registry and
# expects an independent reference formula for every key, so a sibling module
# adding keys makes another area's tests order-dependent (verifier finding).
# They are resolvable by key through `get_activation` / model 'znn_consistent'.
LOCAL_ACTIVATIONS = {PowerSum.key: PowerSum, PredefinedTimeJR.key: PredefinedTimeJR}


def get_activation(key, **kw):
    """Activation by key: the local test vehicles ('signed_power_sum',
    'jr_predefined'), otherwise activations.get (the shared zoo wins on a clash)."""
    if key in LOCAL_ACTIVATIONS and key not in acts.REGISTRY:
        return LOCAL_ACTIVATIONS[key](**kw)
    return acts.get(key, **kw)


# ===========================================================================
# structure detection
# ===========================================================================
def _zoo_terms(act):
    """Power-sum decomposition of the activations.py zoo members that are sums of
    signed powers (read from their own parameters; None if not recognised)."""
    key = getattr(act, "key", None)
    g = lambda name: float(getattr(act, name))  # noqa: E731
    if key == "linear":
        return [(1.0, 1.0)]
    if key == "sign":
        return [(1.0, 0.0)]
    if key == "sign_power":                                     # sig^r
        return [(1.0, g("r"))]
    if key == "sign_bi_power":                                  # (sig^r + sig^{1/r}) / 2
        return [(0.5, g("r")), (0.5, 1.0 / g("r"))]
    if key == "sfaf":                                           # b1 sig^r + b2 e
        return [(g("b1"), g("r")), (g("b2"), 1.0)]
    if key == "wsbp":                                           # k1 sig^r + k2 sig^{1/r} + k3 e
        return [(g("k1"), g("r")), (g("k2"), 1.0 / g("r")), (g("k3"), 1.0)]
    if key == "vaf":                                            # (a1|e|^eta + a2|e|^omega + a4) sgn + a3 e
        return [(g("a1"), g("eta")), (g("a2"), g("omega")), (g("a3"), 1.0), (g("a4"), 0.0)]
    if key == "fixed_time":                                     # (k1|e|^p + k2|e|^q) sgn + k3 e
        return [(g("k1"), g("p")), (g("k2"), g("q")), (g("k3"), 1.0)]
    if key == "power":                                          # e^p, odd integer p
        return [(1.0, g("p"))]
    if key == "power_sum":                                      # sum_{k=1}^N e^{2k-1} (Yang & Zhang)
        return [(1.0, 2.0 * k - 1.0) for k in range(1, int(getattr(act, "N")) + 1)]
    return None


_PROBE = np.array([-2.5, -1.0, -0.3, -1e-3, 0.0, 1e-3, 0.3, 1.0, 2.5])
_TERMS_CACHE = weakref.WeakKeyDictionary()


def _params_sig(act):
    try:
        return repr(act.params())
    except Exception:  # noqa: BLE001
        return None


def power_terms(act):
    """[(a_i, p_i)] with Phi(e) = sum a_i sign(e)|e|^{p_i}, sorted by p, or None.

    Duck-typed: any activation exposing ``power_terms()`` is supported; the
    activations.py zoo members that are signed power sums ('linear', 'sign',
    'sign_power', 'sign_bi_power', 'sfaf', 'wsbp', 'vaf', 'fixed_time', 'power',
    'power_sum') are recognised from their parameters.  Every decomposition is
    checked against the activation itself at probe points (relative 1e-12); on a
    mismatch (e.g. a sibling changed its formula) None is returned, so the maps
    fall back instead of silently using a wrong decomposition."""
    sig = _params_sig(act)
    try:
        hit = _TERMS_CACHE.get(act)
    except TypeError:
        hit = None
    if hit is not None and hit[0] == sig:
        return hit[1]
    try:
        terms = list(act.power_terms()) if hasattr(act, "power_terms") else _zoo_terms(act)
    except (AttributeError, TypeError, ValueError):
        terms = None
    if terms is not None:
        merged = {}
        for a, p in terms:
            if a > 0:
                merged[float(p)] = merged.get(float(p), 0.0) + float(a)
        terms = sorted(((a, p) for p, a in merged.items()), key=lambda t: t[1]) or None
    if terms is not None:
        ref = np.asarray(act(_PROBE), dtype=float)
        val = sum(a * _spow(_PROBE, p) for a, p in terms)
        if not np.allclose(val, ref, rtol=1e-12, atol=1e-300):
            _warn_once(("terms", getattr(act, "key", "?")),
                       f"power-sum decomposition of activation {getattr(act, 'key', act)!r} "
                       "does not match the activation; peb/exact maps disabled for it")
            terms = None
    try:
        _TERMS_CACHE[act] = (sig, terms)
    except TypeError:
        pass
    return terms


# ===========================================================================
# scalar one-step error maps  Psi_h  (element-wise, h broadcast (B,1) -> (B,m))
# ===========================================================================
def _peb_single(e, ah, p):
    """PEB 2019 consistent map for de/dt = -a sign(e)|e|^p over one step (ah = a h).

    p < 1 (degree p-1 < 0): homogeneous coordinate y = |e|^{1-p} sign(e)
      (Eq. (3.12) with d(s) = exp(s/(1-p))), y_dot = -a(1-p) sign(y), implicit
      Euler (Thm 4.1)  ->  y+ = sign(y) max(|y| - a(1-p)h, 0).
    p = 1 (degree 0): implicit Euler e/(1 + a h).
    p > 1 (degree p-1 > 0): y = |e|^{p-1} sign(e), y_dot = -a(p-1)|y| y,
      semi-implicit Euler (Eq. (5.3), Thm 5.1)  ->  y+ = y / (1 + a(p-1)h|y|).
    For a single term the p != 1 maps coincide with the exact flow."""
    s = np.abs(e)
    if p < 1:
        y = s ** (1.0 - p)
        y = np.maximum(y - ah * (1.0 - p), 0.0)
        s_new = y ** (1.0 / (1.0 - p))
    elif p == 1:
        s_new = s / (1.0 + ah)
    else:
        with np.errstate(over="ignore", divide="ignore"):
            # y / (1 + c y) written as 1 / (1/y + c): exact for y = inf (|e|^{p-1} overflow)
            y = 1.0 / (1.0 / s ** (p - 1.0) + ah * (p - 1.0))
        s_new = y ** (1.0 / (p - 1.0))
    return np.sign(e) * s_new


def peb_map(act, e, h):
    """Homogeneity-based consistent map (PEB 2019).  Sums of homogeneous terms
    are Lie-split in order of decreasing degree, so that the finite-time
    (negative-degree) dead zone is applied last.  Returns None if the
    activation is not a sum of power terms."""
    terms = power_terms(act)
    if terms is None:
        return None
    y = np.asarray(e, dtype=float)
    for a, p in sorted(terms, key=lambda t: -t[1]):
        y = _peb_single(y, a * h, p)
    return y


def _exact_single(e, ah, p):
    if p == 1:
        return np.asarray(e, dtype=float) * np.exp(-ah)
    return _peb_single(e, ah, p)


def _exact_bipower(e, h, a1, p1, a2, p2):
    """Exact flow of de/dt = -(a1|e|^p1 + a2|e|^p2) sign(e), p1 < 1 < p2.

    Settling-time coordinate (time-to-go)
        T(s) = int_0^s dz/(a1 z^p1 + a2 z^p2)
             = C * I_{U/(1+U)}(beta, 1-beta),   U = (a2/a1) s^q,
    q = p2 - p1, beta = (1-p1)/q in (0,1), C = (a1/a2)^beta pi / (a1 q sin(pi beta))
    = sup T (Aldana-Lopez et al. 2019, Thm 1, k = 1).  w = T(|e|) obeys
    w_dot = -1 and stops at 0, so  e+ = sign(e) T^{-1}(max(T(|e|) - h, 0))."""
    e = np.asarray(e, dtype=float)
    q = p2 - p1
    beta = (1.0 - p1) / q
    C = (a1 / a2) ** beta * np.pi / (a1 * q * np.sin(np.pi * beta))
    s = np.abs(e)
    U = (a2 / a1) * s ** q
    z = U / (1.0 + U)
    zc = 1.0 / (1.0 + U)
    # time-to-go w and remaining budget wc = C - w, each from its accurate side
    lo = z <= 0.5
    w_lo = C * special.betainc(beta, 1.0 - beta, np.where(lo, z, 0.0))
    wc_hi = C * special.betainc(1.0 - beta, beta, np.where(lo, 0.5, zc))
    w = np.where(lo, w_lo, C - wc_hi)
    wc = np.where(lo, C - w_lo, wc_hi)
    w_new = np.maximum(w - h, 0.0)
    wc_new = np.minimum(wc + h, C)
    u = w_new / C
    small = u <= 0.5
    z_new = special.betaincinv(beta, 1.0 - beta, np.where(small, u, 0.5))
    zc_new = special.betaincinv(1.0 - beta, beta, np.where(small, 0.5, wc_new / C))
    with np.errstate(divide="ignore", invalid="ignore"):
        U_new = np.where(small, z_new / (1.0 - z_new), (1.0 - zc_new) / zc_new)
    s_new = np.where(w_new > 0, ((a1 / a2) * np.maximum(U_new, 0.0)) ** (1.0 / q), 0.0)
    return np.sign(e) * s_new


def _exact_bernoulli_low(e, h, a1, p1, a2):
    """Exact flow of de/dt = -(a1|e|^p1 + a2|e|) sign(e), p1 < 1:
    w = |e|^{1-p1} obeys w_dot = -(1-p1)(a1 + a2 w)  (Bernoulli), so
    w+ = max((w + a1/a2) exp(-a2(1-p1)h) - a1/a2, 0)."""
    e = np.asarray(e, dtype=float)
    k = a2 * (1.0 - p1) * h
    c = a1 / a2
    w = np.abs(e) ** (1.0 - p1)
    w_new = np.maximum(w * np.exp(-k) + c * np.expm1(-k), 0.0)
    return np.sign(e) * w_new ** (1.0 / (1.0 - p1))


def _exact_bernoulli_high(e, h, a1, a2, p2):
    """Exact flow of de/dt = -(a1|e| + a2|e|^p2) sign(e), p2 > 1:
    v = |e|^{p2-1} obeys v_dot = -(p2-1) v (a1 + a2 v), so
    phi = v/(1 + (a2/a1) v) decays like exp(-a1(p2-1)t)."""
    e = np.asarray(e, dtype=float)
    c = a2 / a1
    k = a1 * (p2 - 1.0) * h
    # phi+ = phi e^{-k}, v+ = phi+/(1 - c phi+)  <=>  v+ = e^{-k} / (1/v + c (1 - e^{-k}))
    # (cancellation-free form, also valid for v = inf)
    with np.errstate(over="ignore", divide="ignore"):
        v_new = np.exp(-k) / (1.0 / np.abs(e) ** (p2 - 1.0) - c * np.expm1(-k))
    return np.sign(e) * v_new ** (1.0 / (p2 - 1.0))


def _jr_equivalent(act):
    """'predefined_exp' (Li, Liao, Xiao, Lu 2019, F2) with k2 = k3 = 0 is
    k1 exp(|e|^p)|e|^{1-p} sign(e)/p = JR Eq. (1) with kappa = 1 - exp(-r^p),
    rho1 = 1/k1, rho2 = 0 (Sanchez-Torres et al. 2018 class).  None otherwise."""
    if getattr(act, "key", None) != "predefined_exp":
        return None
    try:
        if float(act.k2) != 0.0 or float(act.k3) != 0.0:
            return None
        jr = PredefinedTimeJR("exp", a=1.0, mu=float(act.p), rho1=1.0 / float(act.k1), rho2=0.0)
    except (AttributeError, TypeError, ValueError):
        return None
    probe = np.array([1e-3, 0.3, 1.0, 2.5])
    return jr if np.allclose(jr(probe), act(probe), rtol=1e-12) else None


def _exact_closed_form(act, e, h):
    if hasattr(act, "exact_step"):
        return act.exact_step(e, h)
    jr = _jr_equivalent(act)
    if jr is not None:
        return jr.exact_step(e, h)
    terms = power_terms(act)
    if terms is None:
        return None
    if len(terms) == 1:
        a, p = terms[0]
        return _exact_single(e, a * h, p)
    if len(terms) == 2:
        (a1, p1), (a2, p2) = terms
        if p1 < 1 < p2:
            return _exact_bipower(e, h, a1, p1, a2, p2)
        if p2 == 1 and p1 < 1:
            return _exact_bernoulli_low(e, h, a1, p1, a2)
        if p1 == 1 and p2 > 1:
            return _exact_bernoulli_high(e, h, a1, a2, p2)
    return None


def exact_map(act, e, h):
    """Exact one-step flow of de/dt = -Phi(e) over 'time' h (= gamma tau).
    Returns None when no closed form is available for this activation.  The
    output is clipped to |e_{k+1}| <= |e_k| (the exact flow is non-increasing in
    |e|), which only removes rounding overshoot for h far below machine precision."""
    out = _exact_closed_form(act, e, h)
    if out is None:
        return None
    e = np.asarray(e, dtype=float)
    return np.sign(e) * np.minimum(np.abs(out), np.abs(e))


def semi_implicit_map(act, e, h):
    """Linearly implicit (frozen-coefficient) Euler: write Phi(e) = j sign(e) +
    kappa(e) e with j = Phi(0+) and kappa(e) = (Phi(e) - j sign e)/e >= 0; step
    the continuous part with kappa frozen at e_k (PEB 2019 Sec. 1.2/Eq. (5.3)),
    e' = e/(1 + h kappa(e_k)), then the jump part exactly (soft threshold h j).
    Sign-preserving and non-expansive for every h > 0 and every odd monotone
    activation; one activation evaluation, no root solve."""
    e = np.asarray(e, dtype=float)
    s = np.abs(e)
    j = act.jump0()
    j = 0.0 if (not np.isfinite(j) or j < _TINY_JUMP) else j
    with np.errstate(divide="ignore", invalid="ignore"):
        kap = (np.abs(act(s)) - j) / s
    kap = np.where(s > 0, np.maximum(kap, 0.0), 0.0)
    s_new = s / (1.0 + h * kap)
    if j > 0:
        s_new = np.maximum(s_new - h * j, 0.0)
    return np.sign(e) * s_new


def resolvent_map(act, e, h):
    """Backward Euler of (ZNN): the activation's resolvent (I + h Phi)^{-1}."""
    return act.prox(e, h)


ERROR_MAPS = {"resolvent": resolvent_map, "peb": peb_map, "exact": exact_map,
              "semi_implicit": semi_implicit_map}
# 'peb' and 'exact' return None when the activation has no closed form; the
# resolvent and the semi-implicit map exist for every activation.
_CHAIN = {"exact": ("exact", "peb", "resolvent"), "peb": ("peb", "exact", "resolvent"),
          "semi_implicit": ("semi_implicit",), "resolvent": ("resolvent",)}


def apply_error_map(name, act, e, h):
    """Psi_h(e) for map `name`; 'exact' falls back to 'peb' (and vice versa),
    then to 'resolvent', when the activation has no closed form (warned once)."""
    for tried in _CHAIN[name]:
        out = ERROR_MAPS[tried](act, e, h)
        if out is not None:
            if tried != name:
                _warn_once(("fallback", name, getattr(act, "key", "?")),
                           f"error map {name!r} unavailable for activation "
                           f"{getattr(act, 'key', act)!r}; using {tried!r}")
            return out
    raise RuntimeError("unreachable")  # pragma: no cover


# ===========================================================================
# request protocol and the opt-in model
# ===========================================================================
class ErrorMapRequest:
    """Passed by the discretizers as the ``implicit`` argument of f.

    Truthy, so an implicit-aware model that does not know the protocol uses its
    resolvent rate (graceful fallback).  A model that honours it calls
    ``desired_Edot(act, ctx, g)`` (g = effective gain gamma*g(t), shape (B,1))
    and uses the returned error rate in J x_dot = r - dE/dt."""

    def __init__(self, error_map, c_next, c_past, memory=True):
        if error_map not in ERROR_MAPS:
            raise ValueError(f"unknown error map {error_map!r}; choose from {sorted(ERROR_MAPS)}")
        self.error_map = error_map
        self.c_next = float(c_next)
        self.c_past = np.asarray(c_past, dtype=float)
        self.memory = bool(memory)
        self.reset()

    def reset(self):
        self.E_hist = deque(maxlen=max(len(self.c_past) - 1, 0))
        self.multistep = False
        self.honored = False

    def __bool__(self):
        return True

    def begin_step(self, multistep):
        self.multistep = bool(multistep)
        self.honored = False

    def desired_Edot(self, act, ctx, g):
        h = g * ctx.tau
        E = ctx.E
        psi = apply_error_map(self.error_map, act, E, h)
        full = len(self.E_hist) == len(self.c_past) - 1
        if self.multistep and self.memory and full:
            # (EM): tau r_k = c_{-1} Psi(E_k) + sum_j c_j E_{k-j}
            acc = self.c_next * psi + self.c_past[0] * E
            for cj, Ej in zip(self.c_past[1:], self.E_hist):
                acc = acc + cj * Ej
        else:
            # one-step rate (Euler start-up, or the naive multistep variant)
            acc = psi - E
        self.E_hist.appendleft(np.array(E, copy=True))
        self.honored = True
        return acc / ctx.tau


class ConsistentZNN(DesignFormulaZNN):
    """Design-formula ZNN that honours :class:`ErrorMapRequest` (consistent
    error maps and the error-memory look-ahead rate).  Without a request it is
    identical to DesignFormulaZNN (explicit, or resolvent if implicit=True)."""

    def __init__(self, activation="linear", gain=None, name=None, **act_kw):
        if isinstance(activation, str):
            activation = get_activation(activation, **act_kw)
        super().__init__(activation=activation, gain=gain, name=name)
        if name is None:
            self.name = f"znn_consistent_{self.act.key}"

    def desired_Edot(self, ctx):
        req = ctx.implicit
        if hasattr(req, "desired_Edot"):
            g = ctx.gamma * self.gain(ctx.t)
            return req.desired_Edot(self.act.at(ctx.t), ctx, g)   # time-varying Phi frozen at t
        return super().desired_Edot(ctx)


# ===========================================================================
# discretizers
# ===========================================================================
EULER = (1.0, -1.0)
TAYLOR_ZHANG = (1.0, -1.5, 1.0, -0.5)


class ConsistentLookAhead(Discretizer):
    """Causal look-ahead update with a consistent error map.

    x_{k+1} = (tau x_dot_k - sum_j c_j x_{k-j}) / c_{-1}, where x_dot_k comes
    from the model with the rate requested through :class:`ErrorMapRequest`
    (error-memory rate (EM) if ``memory``).  Euler start-up until the formula
    has enough history.  z is advanced by forward Euler."""

    causal = True
    implicit = True

    def __init__(self, coeffs=EULER, error_map="resolvent", memory=True, name=None,
                 expected_order=None, source=""):
        c = np.asarray(coeffs, dtype=float)
        if abs(c.sum()) > 1e-12:
            raise ValueError("look-ahead coefficients must sum to zero (consistency)")
        self.c_next, self.c_past = float(c[0]), c[1:]
        self.s = len(self.c_past)
        self.error_map = error_map
        self.memory = memory
        self.name = name or f"consistent_{error_map}_s{self.s}"
        self.expected_order = expected_order
        self.source = source
        self.req = ErrorMapRequest(error_map, self.c_next, self.c_past, memory)
        self.reset()

    def reset(self):
        self.hist = deque(maxlen=max(self.s - 1, 0))   # x_{k-1}, x_{k-2}, ...
        self.req.reset()
        self._checked = False

    def step(self, k, t, tau, x, z, f):
        multistep = len(self.hist) == self.s - 1
        self.req.begin_step(multistep)
        xdot, zdot = f(t, x, z, k, self.req)
        if not self._checked:
            self._checked = True
            if not self.req.honored:
                _warn_once(("ignored", self.name),
                           f"{self.name}: the model ignored the ErrorMapRequest protocol "
                           "(use model 'znn_consistent'); it ran with its own (resolvent or "
                           "explicit) rate, without consistent map or error memory")
        z_next = z + tau * zdot
        if multistep:
            acc = tau * xdot - self.c_past[0] * x
            for cj, xj in zip(self.c_past[1:], self.hist):
                acc = acc - cj * xj
            x_next = acc / self.c_next
        else:
            x_next = x + tau * xdot
        if self.s > 1:
            self.hist.appendleft(x)
        return x_next, z_next

    def describe(self):
        return {**super().describe(), "coeffs": [self.c_next, *self.c_past.tolist()],
                "error_map": self.error_map, "memory": self.memory, "source": self.source}


_SRC = {
    "resolvent": "backward Euler of the error dynamics (Acary & Brogliato 2010; Huber et al. 2016)",
    "peb": "Polyakov, Efimov & Brogliato, SICON 57(1) 2019, Thms 4.1/5.1 (+ Lie splitting)",
    "exact": "exact flow in settling-time coordinates (Jimenez-Rodriguez et al., IFAC-PapersOnLine "
             "53(2) 2020, Eq. (7))",
    "semi_implicit": "frozen-coefficient linearly implicit Euler (PEB 2019, Sec. 1.2, Eq. (5.3))",
}


_PREFIX = {"resolvent": "implicit", "peb": "peb", "exact": "exact_flow",
           "semi_implicit": "semi_implicit"}


def _formula(formula):
    """(coeffs, expected_order) of a look-ahead formula: 'euler', 'taylor_zhang', or
    the key of any registered discretization that is a `LookAhead` (e.g. ZeaD)."""
    if formula == "euler":
        return EULER, 2
    if formula == "taylor_zhang":
        return TAYLOR_ZHANG, 3
    from ..registry import DISCRETIZATIONS
    from .multistep import LookAhead
    if formula in DISCRETIZATIONS:
        d = DISCRETIZATIONS[formula]()
        if isinstance(d, LookAhead):
            return (d.c_next, *d.c_past), d.expected_order
    raise ValueError(f"formula {formula!r} is not 'euler', 'taylor_zhang' or a registered "
                     "LookAhead discretization")


def consistent(formula="euler", error_map="resolvent", memory=True):
    """Generic factory: `formula` = 'euler', 'taylor_zhang' or any registered LookAhead
    key; error_map in {'resolvent', 'peb', 'exact', 'semi_implicit'}; memory = use the
    error-memory rate (EM) for multistep formulas."""
    if error_map not in ERROR_MAPS:
        raise ValueError(f"unknown error map {error_map!r}; choose from {sorted(ERROR_MAPS)}")
    coeffs, order = _formula(formula)
    multistep = len(coeffs) > 2
    # name == registry key for the registered combinations (e.g. 'exact_flow_euler')
    name = f"{_PREFIX[error_map]}_{formula}" + ("_naive" if multistep and not memory else "")
    src = _SRC[error_map] + (f"; {formula} look-ahead with error-memory rate (novel)"
                             if multistep and memory else "")
    return ConsistentLookAhead(coeffs, error_map=error_map, memory=memory, name=name,
                               expected_order=order, source=src)


def implicit_taylor_zhang(error_map="resolvent", memory=True):
    """Resolvent (implicit) error rate + Taylor-Zhang look-ahead with error memory
    (another error_map gives the corresponding '<map>_taylor_zhang' scheme)."""
    return consistent("taylor_zhang", error_map, memory)


from ..registry import register_discretization, register_model  # noqa: E402

register_discretization("implicit_taylor_zhang")(implicit_taylor_zhang)
register_discretization("implicit_taylor_zhang_naive")(
    lambda error_map="resolvent": implicit_taylor_zhang(error_map, memory=False))
register_discretization("peb_euler")(lambda: consistent("euler", "peb"))
register_discretization("peb_taylor_zhang")(lambda: consistent("taylor_zhang", "peb"))
register_discretization("exact_flow_euler")(lambda: consistent("euler", "exact"))
register_discretization("exact_flow_taylor_zhang")(lambda: consistent("taylor_zhang", "exact"))
register_discretization("semi_implicit_euler")(lambda: consistent("euler", "semi_implicit"))
register_discretization("semi_implicit_taylor_zhang")(
    lambda: consistent("taylor_zhang", "semi_implicit"))
register_discretization("consistent")(consistent)


@register_model("znn_consistent")
def _znn_consistent(activation="linear", **kw):
    return ConsistentZNN(activation=activation, **kw)
