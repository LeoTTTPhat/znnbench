"""ZNN model variants: varying-parameter, integration-enhanced (noise-tolerant),
double-integral-enhanced, harmonic-noise-tolerant, strictly predefined-time
(integer and conformable fractional order) and gradient-ZNN hybrid models.

All models follow the ZNN-Bench conventions (core.py): the design formula
prescribes the desired error derivative r = dE/dt|desired, the state velocity
is x_dot = J^{-1} (r - dE/dt|_x) via ``ctx.solve``, additive noise nu enters
the design-formula space (J x_dot = ... + nu, added by ``simulate``), and the
base design gain of the source paper is ``ctx.gamma`` (shape (B, 1)).
Auxiliary states (integrals of the error, internal-model states) live in z
and are advanced by the discretizer (forward Euler for causal schemes, as the
source papers do when they discretize).

Registered keys (see docs/spec/variants.md for equations, defaults, DOIs and
every ASSUMPTION):

  vpznn          power-type varying-parameter ZNN, g(t) = zeta + t^p
  vpznn_exp      exponential-type varying-parameter ZNN, g(t) = p^t + p
  ieznn          integration-enhanced / noise-tolerant ZNN (Jin, Zhang, Li 2016/2017)
  dieznn         double-integral-enhanced ZNN (Liao, Han, Cao, Li, Li 2023/2024)
  hntznn         harmonic-noise-tolerant ZNN, internal model of known
                 frequencies (Liao, Wang, Li, Guo, He 2022)
  sptc_nt_znn    strictly predefined-time noise-tolerant ZNN with the
                 time-varying gain 1/(t_c - t) (Li, Guo, Ma, Pan 2024)
  sptc_an_foznn  strictly predefined-time anti-noise conformable fractional-
                 order ZNN (Yang et al., Neural Networks 2025)
  gznn           gradient-ZNN hybrid: ZNN design applied to the GNN gradient
                 error J^T E (Stanimirovic, Mourtas, Katsikis, Kazakovtsev,
                 Krutikov 2022; time-varying extension of the hybrid GNN-ZNN
                 models of Stanimirovic, Katsikis, Li 2018)
"""
from __future__ import annotations

import numpy as np

from .. import activations as acts
from ..core import Context, Model
from ..registry import register_model
from .gains import ExpVPGain, PowerVPGain
from .znn import DesignFormulaZNN


def _empty(B):
    return np.zeros((B, 0))


def _resolve_activation(activation, **act_kw):
    if isinstance(activation, acts.Activation):
        return activation
    if activation == "li":
        return LiActivation(**act_kw)
    return acts.get(activation, **act_kw)


def _spow(e, p):
    return np.sign(e) * np.abs(e) ** p


class LiActivation(acts.Activation):
    """Li activation as written in Liao et al. (Front. Neurorobot. 2022, Eqs.
    4-5): psi(e) = Lip^r(e) + Lip^(1/r)(e), 0 < r < 1, Lip^r(e) = sign(e)|e|^r.

    NOTE: this is twice the sign-bi-power activation of Li, Chen, Liu (Neural
    Process. Lett. 2013), (Lip^r + Lip^(1/r)) / 2, which is ``sign_bi_power``
    in activations.py.  Not registered in the activation registry (owned by
    another area); used by ``hntznn``."""

    key = "li"
    slope0 = np.inf
    finite_time = True
    convergence = "fixed"        # same class as sign_bi_power (a gain rescaling of it)
    source = "Liao, Wang, Li, Guo, He, Front. Neurorobot. 2022, Eqs. (4)-(5)"
    provenance = "primary"

    def __init__(self, r: float = 0.5):
        if not 0 < r < 1:
            raise ValueError("Li activation needs 0 < r < 1")
        self.r = float(r)

    def __call__(self, e):
        e = np.asarray(e, dtype=float)
        return _spow(e, self.r) + _spow(e, 1.0 / self.r)

    def params(self):
        return {"r": self.r}


# ---------------------------------------------------------------------------
# (a) varying-parameter ZNN (design formula with a time-varying gain)
# ---------------------------------------------------------------------------
@register_model("vpznn")
def _vpznn(activation="linear", p=1.0, zeta=None, cap=None, **act_kw):
    """Power-type VP-ZNN: dE/dt = -gamma (zeta + t^p) Phi(E), zeta = p by
    default (Zhang et al. TCYB 2018, TCYB 2019; zeta + t^gamma: TMECH 2018).
    gamma = 1 reproduces the source; p = 1 is a ZNN-Bench default."""
    return DesignFormulaZNN(activation=activation, gain=PowerVPGain(p=p, zeta=zeta, cap=cap),
                            name="vpznn", **act_kw)


@register_model("vpznn_exp")
def _vpznn_exp(activation="linear", p=2.0, cap=None, **act_kw):
    """Exponential-type VP-ZNN: dE/dt = -gamma (p^t + p) Phi(E) (Zhang, Zheng,
    Wang, Neurocomputing 2019).  gamma = 1 reproduces the source; p = 2 is a
    ZNN-Bench default."""
    return DesignFormulaZNN(activation=activation, gain=ExpVPGain(p=p, cap=cap),
                            name="vpznn_exp", **act_kw)


# ---------------------------------------------------------------------------
# (b) integration-enhanced (noise-tolerant) ZNN
# ---------------------------------------------------------------------------
class IEZNN(Model):
    """Integration-enhanced ZNN (IEZNN) / noise-tolerant ZNN (NTZNN).

    Design formula (Jin, Zhang, Li, IEEE TNNLS 27(12):2615-2627, 2016,
    doi:10.1109/TNNLS.2015.2497715; same formula in Jin, Zhang, Li, Zhang,
    IEEE TAC 62(2):992-997, 2017, doi:10.1109/TAC.2016.2566880):

        dE/dt = -gamma E(t) - lam int_0^t E(s) ds  (+ nu(t))

    Auxiliary state z = int_0^t E ds (z(0) = 0, z_dot = E).
    Error dynamics are the LTI system s^2 + gamma s + lam; a constant noise is
    rejected exactly, a linear noise Q t + M leaves the limit E -> Q / lam.

    lam: absolute integral gain.  If None (default), ASSUMPTION (ZNN-Bench):
    lam = lam_ratio * gamma^2 with lam_ratio = 1/4 (critically damped double
    pole at -gamma/2), so that the step-size knob h = gamma tau keeps one
    meaning across gains.  The sources only require gamma, lam > 0.
    """

    family = "znn"
    uses_dEdt = True

    def __init__(self, lam=None, lam_ratio=0.25, name="ieznn"):
        if lam is not None and not lam > 0:
            raise ValueError("lam must be > 0")
        if not lam_ratio > 0:
            raise ValueError("lam_ratio must be > 0")
        self.lam, self.lam_ratio, self.name = lam, float(lam_ratio), name

    def n_z(self, problem):
        return problem.m

    def lam_of(self, gamma):
        return self.lam if self.lam is not None else self.lam_ratio * gamma ** 2

    def rhs(self, ctx: Context):
        g = ctx.gamma
        r = -g * ctx.E - self.lam_of(g) * ctx.z
        return ctx.solve(r - ctx.dEdt), ctx.E.copy()

    def describe(self):
        return {"name": self.name, "family": self.family, "lam": self.lam,
                "lam_ratio": self.lam_ratio if self.lam is None else None,
                "source": "Jin, Zhang, Li, IEEE TNNLS 2016, doi:10.1109/TNNLS.2015.2497715"}


@register_model("ieznn")
def _ieznn(**kw):
    return IEZNN(**kw)


# ---------------------------------------------------------------------------
# (c) double-integral-enhanced ZNN
# ---------------------------------------------------------------------------
class DIEZNN(Model):
    """Double integral-enhanced ZNN (DIEZNN), Liao, Han, Cao, Li, Li, CAAI
    Trans. Intell. Technol. 9(1):197-210, 2024 (online 2023),
    doi:10.1049/cit2.12161, Eqs. (14)-(16):

        dS/dt = -(2 xi + 1) S - (xi^2 + 2 xi) int_0^t S
                - xi^2 int_0^t int_0^s S  (+ H(t))

    (paper notation gamma = 2 xi + 1, lambda = xi^2 + 2 xi).  The closed-loop
    characteristic polynomial is (s + 1)(s + xi)^2: two poles at -xi and one
    pole fixed at -1 for every xi (it comes from the seed dynamics
    dS/dt = -S, Eq. 8).  Linear noise H = Q t + M is rejected exactly.

    Auxiliary state z = [int S, int int S] (2m entries, zero initial values).
    ZNN-Bench mapping (ASSUMPTION): xi = ctx.gamma unless ``xi`` is given, so
    that h = gamma tau is the step of the fastest (double) pole.
    """

    family = "znn"
    uses_dEdt = True

    def __init__(self, xi=None, name="dieznn"):
        if xi is not None and not xi > 0:
            raise ValueError("xi must be > 0")
        self.xi, self.name = xi, name

    def n_z(self, problem):
        return 2 * problem.m

    def rhs(self, ctx: Context):
        m = ctx.E.shape[1]
        xi = ctx.gamma if self.xi is None else self.xi
        z1, z2 = ctx.z[:, :m], ctx.z[:, m:]
        r = -(2 * xi + 1) * ctx.E - (xi ** 2 + 2 * xi) * z1 - xi ** 2 * z2
        return ctx.solve(r - ctx.dEdt), np.hstack([ctx.E, z1])

    def describe(self):
        return {"name": self.name, "family": self.family, "xi": self.xi,
                "source": "Liao et al., CAAI TIT 2024, doi:10.1049/cit2.12161"}


@register_model("dieznn")
def _dieznn(**kw):
    return DIEZNN(**kw)


# ---------------------------------------------------------------------------
# (d) harmonic-noise-tolerant ZNN (internal model of known frequencies)
# ---------------------------------------------------------------------------
class HNTZNN(Model):
    """Harmonic noise-tolerant ZNN, Liao, Wang, Li, Guo, He, Front. Neurorobot.
    16:928636, 2022, doi:10.3389/fnbot.2022.928636, Eqs. (11) (single
    harmonic) and (14) (multiple harmonics, frequencies f_k known, amplitude
    and phase unknown):

        dE/dt   = -tau Psi(E) - sum_k X_k  (+ O(t))
        dX_k/dt = C_k + 4 pi^2 f_k^2 lam E
        dC_k/dt = -4 pi^2 f_k^2 X_k ,        lam >= 1

    -X_k learns the harmonic o_k(t) through the oscillator internal model
    o_ddot = -(2 pi f_k)^2 o.  With a linear Psi the closed loop has the
    transmission zeros s = +-j 2 pi f_k, so E -> 0 under the matched harmonic
    noise.  ZNN-Bench mapping: the paper's gain tau is ``ctx.gamma``;
    frequencies are given as ``omegas`` (rad/s, omega = 2 pi f, matching
    noise/basic.py 'harmonic') or ``freqs`` (Hz).

    Auxiliary state z = [X_1..X_l, C_1..C_l] (2 l m entries, zero initial
    values -- ASSUMPTION, the paper does not state X(0), C(0)).
    Activation: the paper uses the Li activation (``activation='li'``,
    default r = 0.5 -- ASSUMPTION, the paper's simulation value of the
    activation exponent is not stated); any registered activation key or an
    Activation instance may be passed.  Default lam = 1 (paper simulations).
    """

    family = "znn"
    uses_dEdt = True

    def __init__(self, omegas=(2 * np.pi,), freqs=None, lam=1.0, activation="li",
                 name="hntznn", **act_kw):
        if freqs is not None:
            omegas = 2 * np.pi * np.atleast_1d(np.asarray(freqs, dtype=float))
        self.omegas = np.atleast_1d(np.asarray(omegas, dtype=float))
        if self.omegas.ndim != 1 or len(self.omegas) == 0 or np.any(self.omegas <= 0):
            raise ValueError("omegas must be a non-empty list of positive frequencies (rad/s)")
        if not lam > 0:
            raise ValueError("lam must be > 0 (the paper requires lam >= 1)")
        self.lam = float(lam)
        self.act = _resolve_activation(activation, **act_kw)
        self.name = name

    def n_z(self, problem):
        return 2 * len(self.omegas) * problem.m

    def rhs(self, ctx: Context):
        B, m = ctx.E.shape
        L = len(self.omegas)
        X = ctx.z[:, :L * m].reshape(B, L, m)
        C = ctx.z[:, L * m:].reshape(B, L, m)
        w2 = (self.omegas ** 2)[None, :, None]
        r = -ctx.gamma * self.act.at(ctx.t)(ctx.E) - X.sum(axis=1)
        Xdot = C + w2 * self.lam * ctx.E[:, None, :]
        Cdot = -w2 * X
        return ctx.solve(r - ctx.dEdt), np.hstack([Xdot.reshape(B, -1), Cdot.reshape(B, -1)])

    def describe(self):
        return {"name": self.name, "family": self.family, "omegas": self.omegas.tolist(),
                "lam": self.lam, "activation": self.act.describe(),
                "source": "Liao et al., Front. Neurorobot. 2022, doi:10.3389/fnbot.2022.928636"}


@register_model("hntznn")
def _hntznn(**kw):
    return HNTZNN(**kw)


# ---------------------------------------------------------------------------
# (a') strictly predefined-time ZNN with time-varying gain 1/(t_c - t)
# ---------------------------------------------------------------------------
def _guard(t_c):
    return 1e-12 * max(1.0, t_c)


class SPTCNTZNN(Model):
    """Strictly predefined-time convergent and noise-tolerant ZNN (SPTC-NT-ZNN),
    Li, Guo, Ma, Pan, IEEE Trans. Ind. Electron. 71(1):798-809, 2024,
    doi:10.1109/TIE.2023.3241393, as transcribed (Table 1 and Sec. 4) by
    Yang et al., Neural Networks 186:107279, 2025 (co-authors W. Li, X. Ma):

        dE/dt = -gamma Phi_t(E) (+ delta(t)),
        Phi_t(x) = x / (t_c - t)                               , t <  t_c
                 = x + |x|^p sign(x) + xi sign(x)  (element-wise), t >= t_c

    Before t_c this is linear ZNN with the time-varying gain gamma/(t_c - t):
    E(t) = E(0) ((t_c - t)/t_c)^gamma -> 0 exactly at t_c.  After t_c the
    sign term keeps E = 0 under bounded noise |delta| < gamma xi.
    Defaults follow Yang et al.'s comparison: t_c = 1 s, p = 0.5,
    xi = zeta / gamma with zeta = 5 Delta; ASSUMPTION Delta = noise_bound = 0.5
    (the noise bound is problem-dependent).  The primary text was not
    accessible: this transcription is second-hand (flagged in the spec).

    cap: optional saturation of 1/(t_c - t) (ZNN-Bench addition, default None
    = faithful).  Numerical guard: t >= t_c - 1e-12 max(1, t_c) is treated as
    the post-t_c phase.
    """

    family = "znn"
    uses_dEdt = True

    def __init__(self, t_c=1.0, p=0.5, xi=None, zeta=None, noise_bound=0.5, cap=None,
                 name="sptc_nt_znn"):
        if not t_c > 0:
            raise ValueError("t_c must be > 0")
        self.t_c, self.p = float(t_c), float(p)
        self.noise_bound = float(noise_bound)
        self.zeta = 5.0 * self.noise_bound if zeta is None else float(zeta)
        self.xi = xi
        self.cap = cap
        self.name = name

    def rhs(self, ctx: Context):
        g, E, t = ctx.gamma, ctx.E, ctx.t
        if t < self.t_c - _guard(self.t_c):
            inv = 1.0 / (self.t_c - t)
            if self.cap is not None:
                inv = min(inv, self.cap)
            r = -g * inv * E
        else:
            xi = self.zeta / g if self.xi is None else self.xi
            r = -g * (E + _spow(E, self.p) + xi * np.sign(E))
        return ctx.solve(r - ctx.dEdt), _empty(E.shape[0])

    def describe(self):
        return {"name": self.name, "family": self.family, "t_c": self.t_c, "p": self.p,
                "xi": self.xi, "zeta": self.zeta, "cap": self.cap,
                "source": "Li, Guo, Ma, Pan, IEEE TIE 2024, doi:10.1109/TIE.2023.3241393"}


@register_model("sptc_nt_znn")
def _sptc_nt_znn(**kw):
    return SPTCNTZNN(**kw)


# ---------------------------------------------------------------------------
# (e) conformable fractional-order strictly predefined-time anti-noise ZNN
# ---------------------------------------------------------------------------
class SPTCANFOZNN(Model):
    """SPTC-AN-FOZNN, Yang, Li, Wang, Liu, Yin, Li, Voyles, Ma, Neural Networks
    186:107279, 2025, doi:10.1016/j.neunet.2025.107279 (arXiv:2503.01857).

    Conformable fractional design formula (Eq. 11) W_alpha(E)(t) =
    t^(1-alpha) dE/dt = -gamma Phi(E), i.e. the integer-order ODE with
    time-varying gain gamma t^(alpha-1) (Eq. 12, 13 with noise delta):

        dE/dt = -gamma t^(alpha-1) Phi(E) (+ delta(t)),   0 < alpha <= 1,

    and the predefined-time stabilizer (Eq. 14), x in R^m (vector 2-norm):

        Phi(x) = 1/(gamma t_c^(alpha-1)) * ( x/(t_c - t)
                 + gamma ||x||^2/(t_c - t)^2 * x/||x|| ),                t < t_c
               = 1/(gamma t^(alpha-1)) * ( x/(t_p - t)
                 + (zeta + gamma t^(alpha-1) ||x||^2/(t_p - t)^2) x/||x|| ), t >= t_c

        t_p = (1 - exp(-pi / (2 sqrt(zeta (zeta - Delta))))) t_c,  zeta > Delta,

    Delta = ||delta||_inf bound (Assumption 1).  Substituting gives the
    error dynamics of Eqs. (15) and (18):

        t <  t_c: dE/dt = -(t/t_c)^(alpha-1) [E/(t_c-t) + gamma ||E|| E/(t_c-t)^2]
        t >= t_c: dE/dt = E/(t-t_p) - zeta E/||E|| - gamma t^(alpha-1) ||E|| E/(t-t_p)^2

    Theorem 1 (noise-free): V = ||E||/(t_c - t) satisfies V_dot <= -gamma V^2/(t_c - t),
    hence V(t) <= 1 / (1/V(0) + gamma ln(t_c/(t_c - t))) and ||E(t_c)|| = 0.

    Defaults: alpha = 0.5 (paper's robot experiments), t_c = 1 s (TVQP
    example), zeta = 5 Delta (paper), ASSUMPTION Delta = noise_bound = 0.5.
    ASSUMPTION t_floor = 1e-3: t^(alpha-1) is evaluated at max(t, t_floor)
    (singular at t = 0 for alpha < 1; the paper does not say how t_0 = 0 is
    handled in its Euler simulation).  ``cap`` (ZNN-Bench addition, default
    None) saturates 1/(t_c - t).  E = 0 is mapped to E/||E|| = 0.
    """

    family = "znn"
    uses_dEdt = True

    def __init__(self, alpha=0.5, t_c=1.0, noise_bound=0.5, zeta=None, t_floor=1e-3, cap=None,
                 name="sptc_an_foznn"):
        if not 0 < alpha <= 1:
            raise ValueError("SPTC-AN-FOZNN is stated for 0 < alpha <= 1")
        if not t_c > 0:
            raise ValueError("t_c must be > 0")
        self.alpha, self.t_c = float(alpha), float(t_c)
        self.noise_bound = float(noise_bound)
        self.zeta = 5.0 * self.noise_bound if zeta is None else float(zeta)
        if not (self.zeta > self.noise_bound and self.zeta > 0):
            raise ValueError("need zeta > noise_bound (zeta > Delta) and zeta > 0")
        self.t_p = (1.0 - np.exp(-np.pi / (2.0 * np.sqrt(self.zeta * (self.zeta - self.noise_bound))))) * self.t_c
        self.t_floor, self.cap, self.name = float(t_floor), cap, name

    def rhs(self, ctx: Context):
        g, E, t = ctx.gamma, ctx.E, ctx.t
        nrm = np.linalg.norm(E, axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            unit = np.divide(E, nrm, out=np.zeros_like(E), where=nrm > 0)
        teff = max(t, self.t_floor)
        if t < self.t_c - _guard(self.t_c):
            s = (teff / self.t_c) ** (self.alpha - 1.0)
            inv = 1.0 / (self.t_c - t)
            if self.cap is not None:
                inv = min(inv, self.cap)
            r = -s * (inv * E + g * nrm * E * inv ** 2)
        else:
            d = t - self.t_p                                  # > 0 since t >= t_c > t_p
            r = E / d - self.zeta * unit - g * teff ** (self.alpha - 1.0) * nrm * E / d ** 2
        return ctx.solve(r - ctx.dEdt), _empty(E.shape[0])

    def lyapunov_bound(self, t, V0, gamma):
        """Theorem 1 bound on V(t) = ||E(t)||/(t_c - t) for t < t_c (noise-free)."""
        t = np.asarray(t, dtype=float)
        return 1.0 / (1.0 / V0 + gamma * np.log(self.t_c / (self.t_c - t)))

    def describe(self):
        return {"name": self.name, "family": self.family, "alpha": self.alpha, "t_c": self.t_c,
                "zeta": self.zeta, "noise_bound": self.noise_bound, "t_p": self.t_p,
                "t_floor": self.t_floor, "cap": self.cap,
                "source": "Yang et al., Neural Networks 2025, doi:10.1016/j.neunet.2025.107279"}


@register_model("sptc_an_foznn")
def _sptc_an_foznn(**kw):
    return SPTCANFOZNN(**kw)


# ---------------------------------------------------------------------------
# (f) gradient-ZNN hybrid (ZNN design on the GNN gradient error J^T E)
# ---------------------------------------------------------------------------
class GZNN(Model):
    """Gradient-based ZNN (GZNN), Stanimirovic, Mourtas, Katsikis, Kazakovtsev,
    Krutikov, Mathematics 10(22):4292, 2022, doi:10.3390/math10224292,
    Eqs. (33)-(40): the ZNN design formula applied to the GNN gradient
    E_G = grad (1/2)||E||^2 = J^T E (for A Y C = B: E_G = A^T (A Y C - B) C^T):

        dE_G/dt = -gamma E_G
        =>  M x_dot = -gamma J^T E - (dJ/dt)^T E - J^T dE/dt|_x ,

    (Eq. 35 vectorized as Eqs. 36-40) with the mass matrix W = C C^T (x) A^T A
    = J^T J regularized as in Eq. (39): M = W when J^T J is nonsingular and
    M = W + beta I when it is rank deficient; the paper's simulations use
    beta = 1e-8 (Sec. 4) and lambda (= gamma here) = 10.  It is the
    time-varying extension of the hybrid GNN-ZNN (HZNN) models of
    Stanimirovic, Katsikis, Li, Neurocomputing 316:124-134, 2018,
    doi:10.1016/j.neucom.2018.07.058.  Paper Eq. (35) prints
    -lambda A^T (A Y C - B) without the trailing C^T; we use the consistent
    form -gamma E_G (typo flagged in the spec).

    Regularization (``regularize``): 'auto' (default, Eq. 39) adds beta I when
    J^T J is structurally rank deficient (m < n_x, e.g. redundant robots) or
    when the unregularized solve fails; 'always' / 'never' force it.

    dJ/dt (``jdot``) -- ZNN-Bench implementation, chosen so that the model
    never samples problem data at instants other than the one ``simulate``
    hands it (no future data, no side effects on stateful data wrappers):

    * 'analytic': dJ/dt e_j = d/dx_j [dE/dt|_x] (Schwarz), evaluated by a
      central difference of ``problem.dEdt`` in x with step
      fd_h * max(1, |x|_inf).  Exact up to round-off for problems affine in x
      (E = J(t) x - b(t)); O(fd_h^2) otherwise.  It is the derivative of
      whatever the problem delivers through its dEdt slot (the exact
      derivative, or the causal estimate of a derivative-estimating wrapper).
    * 'bd1' / 'bd2': first/second-order backward differences of J(x, d_j)
      over the data samples of the previous causal calls (mirrors
      ``simulate``'s info='bd1'/'bd2', including its start-up: no history ->
      0, one past sample -> bd1).
    * 'none': dJ/dt = 0.
    * 'auto' (default): use ``ctx.extra['info']`` if present (requested core
      change), otherwise infer the regime that produced ``ctx.dEdt`` by
      comparing it with problem.dEdt(x, t), with 0 and with the bd1/bd2
      formulas over the stored samples.  The detected regime of the last call
      is kept in ``last_regime``.  Unrecognised ``ctx.dEdt`` -> 'analytic'.

    Implementation: J is assembled densely from ``problem.JT`` (one batched
    call per data sample), so the cost is O(B m n_x^2 + n_x^3) per evaluation
    and memory O(B m n_x + B n_x^2); a guard (``max_dense`` = 4e6 Jacobian
    entries, checked in ``n_z`` i.e. when ``simulate`` starts) refuses larger
    problems.  LIMITATION: GZNN is derived for problems linear in x; on a
    nonlinear E the term (dJ/dx x_dot)^T E of dE_G/dt is ignored.  The model
    keeps the data samples of its last three causal calls (reset at k = 0),
    so one instance must not be shared by concurrently running simulations.
    """

    family = "gnn-znn"
    uses_dEdt = True
    JDOT_MODES = ("auto", "analytic", "bd1", "bd2", "none")

    def __init__(self, beta=1e-8, regularize="auto", jdot="auto", fd_h=1e-4,
                 max_dense=4_000_000, name="gznn"):
        if not beta >= 0:
            raise ValueError("beta must be >= 0")
        if regularize not in ("auto", "always", "never"):
            raise ValueError("regularize must be 'auto', 'always' or 'never'")
        if jdot not in self.JDOT_MODES:
            raise ValueError(f"jdot must be one of {self.JDOT_MODES}")
        if not fd_h > 0:
            raise ValueError("fd_h must be > 0")
        self.beta, self.regularize, self.jdot = float(beta), regularize, jdot
        self.fd_h, self.name = float(fd_h), name
        self.max_dense = int(max_dense)
        self._hist = {}              # causal step index k -> data dict seen at t_k
        self.last_regime = None

    def n_z(self, problem):
        if problem.m * problem.n_x > self.max_dense:
            raise ValueError(f"gznn assembles a dense {problem.m}x{problem.n_x} Jacobian; this exceeds "
                             f"max_dense={self.max_dense} entries (memory guard). Raise max_dense "
                             "explicitly if intended.")
        return 0

    @staticmethod
    def dense_J(problem, x, d):
        """Dense Jacobian J(x, d), shape (B, m, n_x), from J^T e_i (row i of J)."""
        B, n = x.shape
        m = problem.m
        xr = np.repeat(x, m, axis=0)
        V = np.tile(np.eye(m), (B, 1))
        return problem.JT(xr, d, V).reshape(B, m, n)

    def jdot_analytic(self, problem, x, t):
        """dJ/dt (B, m, n_x) as the x-Jacobian of problem.dEdt(., t) (central
        difference; exact up to round-off when E is affine in x)."""
        B, n = x.shape
        h = self.fd_h * np.maximum(1.0, np.abs(x).max(axis=1))            # (B,)
        D = h[:, None, None] * np.eye(n)[None]                            # (B, n, n)
        X = np.concatenate([(x[:, None, :] + D).reshape(B * n, n),
                            (x[:, None, :] - D).reshape(B * n, n)])
        G = problem.dEdt(X, t).reshape(2, B, n, -1)                       # (2, B, n, m)
        return np.swapaxes((G[0] - G[1]) / (2.0 * h[:, None, None]), 1, 2)

    # --- information regime of dJ/dt -----------------------------------------
    @staticmethod
    def _close(a, b):
        scale = max(float(np.abs(a).max(initial=0.0)), float(np.abs(b).max(initial=0.0)))
        return float(np.abs(a - b).max(initial=0.0)) <= 1e-9 * scale

    def _bd_candidates(self, problem, x, k, tau, d):
        """simulate-style bd1/bd2 estimates of dE/dt at x from the stored
        samples (None when the history is missing)."""
        d1, d2 = self._hist.get(k - 1), self._hist.get(k - 2)
        out = {}
        if d1 is not None:
            E0, E1 = problem.E(x, d), problem.E(x, d1)
            out["bd1"] = (E0 - E1) / tau
            if d2 is not None:
                out["bd2"] = (3 * E0 - 4 * E1 + problem.E(x, d2)) / (2 * tau)
        return out

    def _regime(self, ctx):
        if self.jdot != "auto":
            return self.jdot
        info = (ctx.extra or {}).get("info")
        if info in ("analytic", "bd1", "bd2", "none"):
            return info
        exact = ctx.problem.dEdt(ctx.x, ctx.t)
        if self._close(ctx.dEdt, exact):
            return "analytic"
        if not np.any(ctx.dEdt):
            return "none"
        if ctx.k >= 0:
            for key, est in self._bd_candidates(ctx.problem, ctx.x, ctx.k, ctx.tau, ctx.d).items():
                if self._close(ctx.dEdt, est):
                    return key
        return "analytic"

    def jdot_matrix(self, ctx):
        """dJ/dt (B, m, n_x) under the information regime of ``ctx``, or None
        when it is zero."""
        P, x = ctx.problem, ctx.x
        if ctx.k >= 0:                       # causal sample: remember the data seen at t_k
            if ctx.k == 0:
                self._hist = {}
            self._hist[ctx.k] = ctx.d
            for kk in [kk for kk in self._hist if kk < ctx.k - 2]:
                del self._hist[kk]
        regime = self._regime(ctx)
        if regime in ("bd1", "bd2") and ctx.k < 0:
            regime = "analytic"              # non-causal stage: simulate uses exact dE/dt there
        self.last_regime = regime
        if regime == "none":
            return None
        if regime == "analytic":
            return self.jdot_analytic(P, x, ctx.t)
        d1, d2 = self._hist.get(ctx.k - 1), self._hist.get(ctx.k - 2)
        if d1 is None:                        # no past sample yet (simulate: dE/dt = 0)
            return None
        J0, J1 = self.dense_J(P, x, ctx.d), self.dense_J(P, x, d1)
        if regime == "bd2" and d2 is not None:
            return (3 * J0 - 4 * J1 + self.dense_J(P, x, d2)) / (2 * ctx.tau)
        return (J0 - J1) / ctx.tau

    def _solve(self, M, r, rank_deficient):
        """Eq. (39)-(40): solve M x_dot = r with M = J^T J (+ beta I)."""
        reg = self.beta * np.eye(M.shape[-1])
        if self.regularize == "always" or (self.regularize == "auto" and rank_deficient):
            return np.linalg.solve(M + reg, r[..., None])[..., 0]
        try:
            return np.linalg.solve(M, r[..., None])[..., 0]
        except np.linalg.LinAlgError:
            if self.regularize == "never" or self.beta == 0:
                raise
            return np.linalg.solve(M + reg, r[..., None])[..., 0]

    def rhs(self, ctx: Context):
        P, x, E = ctx.problem, ctx.x, ctx.E
        J = self.dense_J(P, x, ctx.d)
        r = -ctx.gamma * np.einsum("bmn,bm->bn", J, E) - np.einsum("bmn,bm->bn", J, ctx.dEdt)
        Jd = self.jdot_matrix(ctx)
        if Jd is not None:
            r = r - np.einsum("bmn,bm->bn", Jd, E)
        M = np.einsum("bmi,bmj->bij", J, J)
        return self._solve(M, r, rank_deficient=P.m < P.n_x), _empty(x.shape[0])

    def describe(self):
        return {"name": self.name, "family": self.family, "beta": self.beta,
                "regularize": self.regularize, "jdot": self.jdot, "fd_h": self.fd_h,
                "source": "Stanimirovic et al., Mathematics 2022, doi:10.3390/math10224292"}


@register_model("gznn")
def _gznn(**kw):
    return GZNN(**kw)
