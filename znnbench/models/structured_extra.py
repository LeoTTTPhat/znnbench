"""Pseudoinverse-free (scalar-energy) ZNN and hybrid GNN-ZNN models.

Keys registered here (equations, defaults, assumptions and the source search
log are in docs/spec/structured_extra.md):

``lcc_znn``  Low-computational-complexity ZNN (LCCZNN).
    K. Zheng, S. Li, Y. Zhang, "Low-computational-complexity zeroing neural
    network model for solving systems of dynamic nonlinear equations", IEEE
    Trans. Autom. Control 69(7) (2024) 4368-4379, doi:10.1109/TAC.2023.3319132.
    Read in the accepted manuscript (University of Oulu repository, handle
    10024/48415, retrieved from its Wayback Machine snapshot of 2025-06-12).
    The ZNN design formula is applied to the scalar energy
    nu = ||f||_2^2 / 2 (Eq. (10)): nu_dot = -mu phi(nu) (Eq. (11)), i.e.
    f^T J x_dot + f^T df/dt = -mu phi(||f||^2/2), whose explicit solution
    along J^T f is model (12):

        x_dot = - J^T f / ||J^T f||_2^2 * ( f^T df/dt + mu phi(||f||_2^2 / 2) ),   (12)

    with the novel activation (13) phi(z) = sig^r(z) exp(|z|), 0 < r < 1
    (paper's comparison activations: (8) sig^r(z), (9) sig^r(z) + z).
    Remark 1 gives the regularised form with ||J^T f||^2 + lambda in the
    denominator (``lam``; the paper's experiments use lambda = 0).
    Only one J^T v product and inner products are needed per evaluation:
    O(mn), no linear solve and no pseudoinverse (Algorithm 1, Remark 2).

``inverse_free_znn`` is NOT REGISTERED (adversarial verification, 2026-10-18).
    The inverse-free ZNN (NIFZNN) of J. Chen, Y. Pan, Y. Zhang, S. Li, N. Tan,
    Neural Networks 178 (2024) 106462, doi:10.1016/j.neunet.2024.106462, could
    not be read in its primary text (ScienceDirect bot protection; the Oulu
    accepted manuscript, handle 10024/61885, is behind a Cloudflare check; no
    Wayback/archive.today copy; OpenAlex/Unpaywall: closed).  The design
    contract registers nothing without an obtainable primary source, so the
    key was removed (see ``NOT_REGISTERED``).  For reference only: the
    secondary restatement in L. Xiao, D. Wang, Q. Zuo, H. Fan, H. Cai, IEEE
    Trans. Artif. Intell. (2025), doi:10.1109/TAI.2025.3570287, Sec. II-D,
    Eq. (10) (re-read on IEEE Xplore, document 11008758),

        x_dot = - J^T f / ||f^T J||_2^2 * ( (eta/2) ||f||_2^2 + f^T df/dt ),

    coincides with ``make_model("lcc_znn", activation="linear")`` (model (12)
    with phi(nu) = nu, mu = eta).  Results obtained that way must be labelled
    "LCCZNN (12) with linear phi", not "Chen et al. 2024".

``hgnn_znn``  Hybrid GNN-ZNN (H-GNN-ZNN) for matrix inversion, Eq. (10) of
    J. Dai, Y. Chen, L. Xiao, L. Jia, Y. He, "Design and analysis of a hybrid
    GNN-ZNN model with a fuzzy adaptive factor for matrix inversion", IEEE
    Trans. Ind. Informat. 18(4) (2022) 2434-2442, doi:10.1109/TII.2021.3093115
    (full text read on IEEE Xplore).  It combines ZNN (7), A X_dot =
    -gamma F(AX - I), and GNN (9), X_dot = -gamma A^T F(AX - I), as

        A X_dot = -gamma (A A^T + I) F(A X - I),                              (10)

    with a CONSTANT matrix A (Dai et al. attribute the hybrid to Stanimirovic,
    Katsikis, Li, Neurocomputing 316 (2018) 124-134,
    doi:10.1016/j.neucom.2018.07.058, not read).  With J = dE/dx (left
    multiplication by A for x = vec_row(X)) and J^T (left multiplication by
    A^T), J^{-1}(J J^T + I) = J^T + J^{-1}, so (10) is computed as

        x_dot = -gamma J^T F(E) + J^{-1}( -gamma F(E) ),

    one linear solve (the ZNN part) plus one J^T product (the GNN part).
    ``feedforward=True`` adds the ZNN time-derivative term, J^{-1}(-dE/dt),
    i.e. the design formula dE/dt = -gamma (J J^T + I) F(E) for time-varying
    A: this is a ZNN-Bench extension (ASSUMPTION, the paper treats constant A
    only); the default ``feedforward=False`` is Eq. (10) verbatim.

``fa_gnn_znn``  Fuzzy-adaptive GNN-ZNN (FA-GNN-ZNN), Eq. (11) of Dai et al.:

        A X_dot = -(gamma + nu) (A A^T + I) F(A X - I),                       (11)

    where nu = nu(||E||_F) is the output of the fuzzy controller of Sec. II-B:
    trapezoidal membership functions (2), the four rules ZO->ZO, PS->PS,
    PM->PM, PB->PB, max-min inference (3) and middle-of-maximum (mom)
    defuzzification (4); range of nu: [0, 2.5] (Sec. IV).  ASSUMPTION: the
    breakpoints (m, n, p, q) of the eight membership functions are not given
    in the paper; ``MamdaniMomFactor`` uses a uniform partition of
    [0, u_max] (input) and [0, nu_max] (output) by default.

Not implemented: the event-triggered and other fuzzy ZNN variants of the same
group, and the original HZNN models of Stanimirovic et al. (2018) for BX = D
and XC = D (full text not read).

Verification (2026-10-18): Eqs. (10)-(15), Remark 1, Algorithm 1 and
Theorems 1/4 of Zheng et al. were re-read in the archived accepted manuscript
(Wayback id_ snapshot 20240612171122, 12 pp.); Eqs. (1)-(11), Theorems 1-3
and Sec. IV of Dai et al. were re-read on IEEE Xplore (document 9468339).
See docs/spec/structured_extra.md, "Verification".
"""
from __future__ import annotations

import numpy as np

from .. import activations as acts
from ..core import Context, Model

__all__ = ["ScalarEnergyActivation", "LCCZNN", "MamdaniMomFactor", "HybridGNNZNN",
           "trapmf", "lower_incomplete_gamma", "NOT_REGISTERED"]

_DAI = "Dai, Chen, Xiao, Jia, He, IEEE TII 18(4) 2022, doi:10.1109/TII.2021.3093115"
_ZHENG = "Zheng, Li, Zhang, IEEE TAC 69(7) 2024, doi:10.1109/TAC.2023.3319132"

# Keys that were considered for this area but are deliberately NOT registered
# because the primary source could not be obtained (design contract: never
# register a model from a secondary restatement alone).
NOT_REGISTERED = {
    "inverse_free_znn": {
        "citation": "J. Chen, Y. Pan, Y. Zhang, S. Li, N. Tan, Inverse-free zeroing neural network "
                    "for time-variant nonlinear optimization with manipulator applications, "
                    "Neural Networks 178 (2024) 106462",
        "doi": "10.1016/j.neunet.2024.106462",
        "status": "not registered: primary text not obtainable (ScienceDirect bot protection, "
                  "Oulu repository Cloudflare check, no archived copy)",
        "secondary": "Xiao, Wang, Zuo, Fan, Cai, IEEE TAI 2025, doi:10.1109/TAI.2025.3570287, "
                     "Sec. II-D, Eq. (10): equals make_model('lcc_znn', activation='linear')",
    },
}


def _zeros(B):
    return np.zeros((B, 0))


def lower_incomplete_gamma(a, z):
    """gamma(a, z) = int_0^z s^(a-1) exp(-s) ds (unnormalised), used by
    Theorem 1 of Zheng et al. (2024)."""
    from scipy.special import gamma as _G, gammainc
    return gammainc(a, z) * _G(a)


# ---------------------------------------------------------------------------
# scalar activations of the LCCZNN energy nu = ||f||^2 / 2
# ---------------------------------------------------------------------------
class ScalarEnergyActivation:
    """phi applied to the scalar energy nu >= 0 of LCCZNN (Zheng et al. 2024).

    kinds: ``'exp_sig'`` Eq. (13) phi(z) = sig^r(z) exp(|z|) (the paper's novel
    activation, default); ``'sig'`` Eq. (8) phi(z) = sig^r(z) (sign-power);
    ``'sig_lin'`` Eq. (9) phi(z) = sig^r(z) + z; ``'linear'`` phi(z) = z
    (exponential decay nu = nu0 exp(-mu t); not one of the paper's AFs; it
    coincides with the secondary restatement of the unregistered inverse-free
    ZNN of Chen et al., see ``NOT_REGISTERED``).
    ``settling_time(nu0, mu)`` is the exact convergence time of
    nu_dot = -mu phi(nu): Theorem 1, Eq. (14) (t_AF3) and Remark 3 (t_AF1,
    t_AF2); ``theorem1_bound`` is the upper bound (15).
    ``uniform_bound(mu)`` (ZNN-Bench derivation, not stated in the paper):
    since gamma(1-r, nu0) < Gamma(1-r), activation (13) settles before
    Gamma(1-r)/mu for EVERY nu0 (fixed-time, not only finite-time); the
    other kinds have no uniform bound (inf).
    """

    KINDS = ("exp_sig", "sig", "sig_lin", "linear")

    def __init__(self, kind="exp_sig", r=0.7):
        if kind not in self.KINDS:
            raise ValueError(f"unknown LCCZNN activation {kind!r}; choose from {self.KINDS}")
        if kind != "linear" and not 0.0 < float(r) < 1.0:
            raise ValueError("r must lie in (0, 1) (Zheng et al. 2024, Eqs. (8), (13))")
        self.kind, self.r = kind, float(r)
        self.key = kind

    def __call__(self, z):
        z = np.asarray(z, dtype=float)
        if self.kind == "linear":
            return z
        s = np.sign(z) * np.abs(z) ** self.r
        if self.kind == "sig":
            return s
        if self.kind == "sig_lin":
            return s + z
        with np.errstate(over="ignore", invalid="ignore"):
            return s * np.exp(np.abs(z))

    def settling_time(self, nu0, mu):
        nu0 = np.asarray(nu0, dtype=float)
        a = 1.0 - self.r
        if self.kind == "exp_sig":
            return lower_incomplete_gamma(a, nu0) / mu
        if self.kind == "sig":
            return nu0 ** a / (mu * a)
        if self.kind == "sig_lin":
            return np.log1p(nu0 ** a) / (mu * a)
        return np.full(np.shape(nu0), np.inf)

    def theorem1_bound(self, nu0, mu):
        """Upper bound (15) on t_AF3 (Theorem 1), exp_sig only."""
        if self.kind != "exp_sig":
            raise ValueError("the bound (15) is stated for the activation (13) only")
        nu0 = np.asarray(nu0, dtype=float)
        a = 1.0 - self.r
        return nu0 ** a * (1.0 + a * np.exp(-nu0)) / (mu * a * (2.0 - self.r))

    def uniform_bound(self, mu):
        """sup over nu0 of the settling time: Gamma(1-r)/mu for (13), inf otherwise."""
        if self.kind == "exp_sig":
            from scipy.special import gamma as _G
            return float(_G(1.0 - self.r)) / mu
        return np.inf

    def at(self, t):
        """Time-invariant: frozen activation = self."""
        return self

    def describe(self):
        return {"key": self.kind, "r": self.r}


class _RegistryScalar:
    """Any element-wise activation of ``znnbench.activations`` applied to nu.
    Time-varying registry activations are frozen at the evaluation time with
    ``at(t)`` (as the design-formula models do)."""

    def __init__(self, act):
        self.act = act
        self.key = act.key

    def __call__(self, z):
        return self.act(z)

    def at(self, t):
        return _RegistryScalar(self.act.at(t)) if getattr(self.act, "time_varying", False) else self

    def settling_time(self, nu0, mu):
        return self.act.settling_time(nu0, mu)

    def describe(self):
        return self.act.describe()


# ---------------------------------------------------------------------------
# LCCZNN
# ---------------------------------------------------------------------------
class LCCZNN(Model):
    """Low-computational-complexity ZNN, Zheng-Li-Zhang (IEEE TAC 2024), Eq. (12):

        x_dot = - J^T f (f^T df/dt + mu phi(||f||^2/2)) / (||J^T f||^2 + lam).

    mu = ``ctx.gamma``; f = ``ctx.E`` (measured data), df/dt = ``ctx.dEdt``
    (so every information regime of ``simulate`` is honoured), J^T f =
    ``ctx.JT(E)``.  No linear solve: per evaluation one J^T v product and three
    inner products (O(mn), Algorithm 1).  Works for m <= n (redundant robots)
    exactly as written in the paper (n >= m in Eq. (5)).

    Singularity: the denominator vanishes when J^T f = 0 (f = 0 or f orthogonal
    to the range of J).  The paper's Remark 1 regularises with lam > 0; with the
    default lam = 0 (the paper's experiments) ZNN-Bench returns x_dot = 0 where
    the denominator is exactly 0 (ASSUMPTION; the formula is 0/0 at f = 0).
    Near convergence the phi-part of |x_dot| scales as |f|^(2r-1)/sigma, so
    r < 1/2 gives an unbounded velocity as f -> 0, while the f_t-part stays
    O(|f_t|/sigma_min(J)) with a direction-dependent limit.  The Jacobian of
    the vector field grows like 1/|f| (stiffness; verifier measurement on
    system (17): eigenvalues ~ -2e5 at |f| ~ 0.16), so every explicit
    fixed-step scheme loses the finite-time property near f = 0 (spec §6).

    ``r`` is the exponent of the paper's AFs (8), (9), (13) (default 0.7,
    Sec. IV-A).  With a registry activation key, ``r`` is forwarded to it only
    when given explicitly (otherwise the activation's own default applies);
    time-varying registry activations are frozen at ``ctx.t``.
    """

    family = "znn-pseudoinverse-free"
    uses_dEdt = True
    supports_implicit = False
    requires: set = set()

    def __init__(self, activation="exp_sig", r=None, lam=0.0, name=None, source=None, **act_kw):
        if isinstance(activation, str) and activation in ScalarEnergyActivation.KINDS:
            self.phi = ScalarEnergyActivation(activation, r=0.7 if r is None else r)
        elif isinstance(activation, str):
            if r is not None:
                act_kw = {"r": r, **act_kw}
            self.phi = _RegistryScalar(acts.get(activation, **act_kw))
        elif isinstance(activation, acts.Activation):
            self.phi = _RegistryScalar(activation)
        else:
            self.phi = activation
        if lam < 0:
            raise ValueError("lam (regularisation of Remark 1) must be >= 0")
        self.lam = float(lam)
        self.name = name or f"lcc_znn_{self.phi.key}"
        self.source = source or f"{_ZHENG}, Eqs. (10)-(13), Remark 1"

    def energy(self, E):
        """nu = ||f||_2^2 / 2 per trajectory, Eq. (10)."""
        return 0.5 * np.einsum("bi,bi->b", E, E)

    def rhs(self, ctx: Context):
        E = ctx.E
        w = ctx.JT(E)                                    # J^T f   (B, n)
        den = np.einsum("bi,bi->b", w, w) + self.lam     # ||J^T f||^2 (+ lambda)
        k1 = np.einsum("bi,bi->b", E, ctx.dEdt)          # f^T df/dt
        mu = np.asarray(ctx.gamma, dtype=float).reshape(-1)
        phi = self.phi.at(ctx.t) if callable(getattr(self.phi, "at", None)) else self.phi
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            k2 = mu * phi(self.energy(E))                # mu phi(nu)
            coef = np.where(den > 0, (k1 + k2) / np.where(den > 0, den, 1.0), 0.0)
            xdot = -coef[:, None] * w
        return xdot, _zeros(E.shape[0])

    def describe(self):
        return {"name": self.name, "family": self.family, "activation": self.phi.describe(),
                "lam": self.lam, "source": self.source}


# ---------------------------------------------------------------------------
# fuzzy adaptive factor of FA-GNN-ZNN (Dai et al. 2022, Sec. II-B)
# ---------------------------------------------------------------------------
def trapmf(x, m, n, p, q):
    """Trapezoidal membership function, Dai et al. (2022) Eq. (2):
    0 on [0, m], (x-m)/(n-m) on (m, n), 1 on [n, p], (q-x)/(q-p) on (p, q),
    0 from q on.  Degenerate edges (limits of Eq. (2)): m == n is a vertical
    rising edge, M = 1 from x = n on (so m == n == 0 is a left shoulder with
    M(0) = 1; ASSUMPTION: the first branch of Eq. (2) read literally gives
    M(0) = 0 there, the third gives 1); p == q is a vertical falling edge
    (M = 1 up to x = p); p == q == inf is a right shoulder (1 for x >= n).
    The input universe is x >= 0 (u = ||E||_F)."""
    x = np.asarray(x, dtype=float)
    up = np.where(x >= n, 1.0, 0.0) if n == m else np.clip((x - m) / (n - m), 0.0, 1.0)
    if np.isinf(q):
        down = np.ones_like(x)
    elif q == p:
        down = np.where(x <= p, 1.0, 0.0)
    else:
        down = np.clip((q - x) / (q - p), 0.0, 1.0)
    return np.minimum(up, down)


def _uniform_partition(top):
    h = top / 3.0
    return {"ZO": (0.0, 0.0, 0.0, h), "PS": (0.0, h, h, 2 * h),
            "PM": (h, 2 * h, 2 * h, 3 * h), "PB": (2 * h, 3 * h, np.inf, np.inf)}


class MamdaniMomFactor:
    """Fuzzy factor nu(u), u = ||E||_F, of the FA-GNN-ZNN (Dai et al. 2022):

    Step 1 fuzzification with trapezoidal memberships (2) on the sets ZO, PS,
    PM, PB; Step 2 rules ZO->ZO, PS->PS, PM->PM, PB->PB with max-min
    composition (3): mu(nu) = max_l min(M_{I_l}(u), M_{O_l}(nu)); Step 3
    defuzzification by the middle of maximum (mom) of mu, Eq. (4), i.e. the
    mean of the points of the output universe where mu attains its maximum.
    mom is evaluated on a uniform grid of ``grid`` points of [0, nu_max] (as a
    sampled-universe fuzzy toolbox does).

    ASSUMPTION (paper does not give them): the membership breakpoints.  The
    default is a uniform partition of [0, u_max] and [0, nu_max] into a left
    shoulder ZO, triangles PS and PM, and a right shoulder PB; nu_max = 2.5 is
    the range of nu stated in Sec. IV; u_max = 1 is a ZNN-Bench choice.  Pass
    ``input_sets`` / ``output_sets`` (dicts {name: (m, n, p, q)}) to override.
    The resulting nu(u) is non-decreasing with nu(0) = 0 and nu(u) = nu_max
    for u >= u_max.
    """

    SETS = ("ZO", "PS", "PM", "PB")

    def __init__(self, u_max=1.0, nu_max=2.5, input_sets=None, output_sets=None, grid=1001):
        if u_max <= 0 or nu_max < 0:
            raise ValueError("u_max must be > 0 and nu_max >= 0")
        self.u_max, self.nu_max = float(u_max), float(nu_max)
        self.input_sets = dict(input_sets or _uniform_partition(self.u_max))
        self.output_sets = dict(output_sets or _uniform_partition(self.nu_max))
        for s in (self.input_sets, self.output_sets):
            if set(s) != set(self.SETS):
                raise ValueError(f"membership sets must be exactly {self.SETS}")
        self.grid = np.linspace(0.0, self.nu_max, int(grid))
        self._out = np.stack([trapmf(self.grid, *self.output_sets[k]) for k in self.SETS])  # (4, G)

    def firing(self, u):
        u = np.atleast_1d(np.asarray(u, dtype=float))
        return np.stack([trapmf(u, *self.input_sets[k]) for k in self.SETS], axis=1)  # (B, 4)

    def __call__(self, u):
        w = self.firing(u)                                                   # (B, 4)
        agg = np.minimum(w[:, :, None], self._out[None]).max(axis=1)         # (B, G)
        top = agg.max(axis=1, keepdims=True)
        on = agg >= top - 1e-12
        with np.errstate(invalid="ignore", divide="ignore"):                 # NaN input -> NaN
            return (on * self.grid[None]).sum(axis=1) / on.sum(axis=1)

    def describe(self):
        return {"key": "mamdani_mom", "u_max": self.u_max, "nu_max": self.nu_max,
                "input_sets": {k: list(v) for k, v in self.input_sets.items()},
                "output_sets": {k: list(v) for k, v in self.output_sets.items()},
                "grid": int(self.grid.size)}


# ---------------------------------------------------------------------------
# H-GNN-ZNN / FA-GNN-ZNN
# ---------------------------------------------------------------------------
class HybridGNNZNN(Model):
    """H-GNN-ZNN, Dai et al. (IEEE TII 2022) Eq. (10), and, with ``fuzzy``,
    FA-GNN-ZNN Eq. (11):

        J x_dot = -(gamma + nu) (J J^T + I) F(E) [- dE/dt if feedforward]
        <=> x_dot = -(gamma + nu) J^T F(E) + J^{-1}(-(gamma + nu) F(E) [- dE/dt]).

    For capability ``'matrix_inverse'`` (E = vec_row(AX - I), J^T = left
    multiplication by A^T) this is exactly A X_dot = -(gamma + nu)(A A^T + I)
    F(AX - I).  ``scope='any'`` drops the capability requirement and applies the
    same formula to any problem (ZNN-Bench extension, ASSUMPTION; for non-square
    J the problem's min-norm solve gives J^+(J J^T + I) = J^T + J^+ when J has
    full row rank).  Per evaluation: one linear solve and one J^T product.
    Exponential convergence rate (linear F, constant A, Theorem 1 / Table II):
    (gamma + nu)(1 + lambda_min(A A^T)) versus gamma for ZNN and
    gamma lambda_min(A A^T) for GNN.
    """

    family = "gnn-znn"
    supports_implicit = False

    def __init__(self, activation="linear", feedforward=False, scope="matrix_inverse", fuzzy=None,
                 name=None, **act_kw):
        self.act = activation if isinstance(activation, acts.Activation) else acts.get(activation, **act_kw)
        if scope not in ("matrix_inverse", "any"):
            raise ValueError("scope must be 'matrix_inverse' (Dai et al.) or 'any' (extension)")
        self.feedforward = bool(feedforward)
        self.uses_dEdt = self.feedforward
        self.scope = scope
        self.requires = {"matrix_inverse"} if scope == "matrix_inverse" else set()
        self.fuzzy = fuzzy
        base = "fa_gnn_znn" if fuzzy is not None else "hgnn_znn"
        self.name = name or f"{base}_{self.act.key}" + ("_ff" if self.feedforward else "")
        self._checked = None

    def _check(self, problem):
        if self._checked is problem:
            return
        if not self.requires <= set(problem.capabilities()):
            raise ValueError(f"{self.name} requires a problem with capability 'matrix_inverse' "
                             f"(Dai et al. 2022 solve A X = I); got {sorted(problem.capabilities())} "
                             f"for {problem.name}. Use scope='any' for the ZNN-Bench extension.")
        self._checked = problem

    def total_gain(self, ctx: Context):
        """gamma + nu(||E||_F), shape (B, 1) (nu = 0 for H-GNN-ZNN)."""
        g = np.asarray(ctx.gamma, dtype=float).reshape(-1, 1)
        if self.fuzzy is None:
            return g
        u = np.linalg.norm(ctx.E, axis=1)
        return g + np.asarray(self.fuzzy(u), dtype=float).reshape(-1, 1)

    def rhs(self, ctx: Context):
        self._check(ctx.problem)
        F = self.act.at(ctx.t)(ctx.E)
        g = self.total_gain(ctx)
        r = -g * F
        if self.feedforward:
            r = r - ctx.dEdt
        xdot = ctx.solve(r) - g * ctx.JT(F)
        return xdot, _zeros(ctx.x.shape[0])

    def describe(self):
        d = {"name": self.name, "family": self.family, "activation": self.act.describe(),
             "feedforward": self.feedforward, "scope": self.scope, "requires": sorted(self.requires),
             "source": f"{_DAI}, Eq. ({11 if self.fuzzy is not None else 10})"}
        if self.fuzzy is not None:
            d["fuzzy"] = self.fuzzy.describe()
        return d


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------
from ..registry import register_model  # noqa: E402


@register_model("lcc_znn")
def _lcc_znn(activation="exp_sig", r=None, lam=0.0, **kw):
    """LCCZNN (Zheng, Li, Zhang 2024), default activation (13), r = 0.7 (Sec. IV-A)."""
    return LCCZNN(activation=activation, r=r, lam=lam, **kw)


# "inverse_free_znn" (Chen et al. 2024) is intentionally not registered: see NOT_REGISTERED.


@register_model("hgnn_znn")
def _hgnn_znn(activation="linear", feedforward=False, scope="matrix_inverse", **kw):
    """H-GNN-ZNN, Dai et al. (2022) Eq. (10)."""
    return HybridGNNZNN(activation=activation, feedforward=feedforward, scope=scope, **kw)


@register_model("fa_gnn_znn")
def _fa_gnn_znn(activation="linear", feedforward=False, scope="matrix_inverse", u_max=1.0,
                nu_max=2.5, input_sets=None, output_sets=None, grid=1001, **kw):
    """FA-GNN-ZNN, Dai et al. (2022) Eq. (11); membership breakpoints are an ASSUMPTION."""
    fz = MamdaniMomFactor(u_max=u_max, nu_max=nu_max, input_sets=input_sets,
                          output_sets=output_sets, grid=grid)
    return HybridGNNZNN(activation=activation, feedforward=feedforward, scope=scope, fuzzy=fz, **kw)

