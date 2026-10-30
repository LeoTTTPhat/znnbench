"""Tests for models/structured_extra.py: LCCZNN (Zheng, Li, Zhang, IEEE TAC 2024)
and the hybrid GNN-ZNN models H-GNN-ZNN / FA-GNN-ZNN (Dai et al., IEEE TII 2022).
The inverse-free ZNN of Chen et al. (Neural Netw. 2024) is NOT registered (primary
text not obtainable); its secondary restatement is checked against
lcc_znn(activation='linear') only.  Small and fast (n <= 9 states, <= 2e4 steps
per run)."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.special import gamma as Gamma

from znnbench import activations as acts
from znnbench import available, make_discretization, make_model, make_problem, simulate
from znnbench.core import Context, Problem
from znnbench.models.structured_extra import (NOT_REGISTERED, LCCZNN, HybridGNNZNN,
                                              MamdaniMomFactor, ScalarEnergyActivation,
                                              lower_incomplete_gamma, trapmf)
from znnbench.problems.matrix import MatrixInversion

ROOT = Path(__file__).resolve().parents[1]
KEYS = ("lcc_znn", "hgnn_znn", "fa_gnn_znn")
DAI_A1 = np.array([[1.0, -2.0, 3.0], [0.0, -3.0, -2.0], [1.0, 2.0, 3.0]])   # Dai et al., Example 1


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
class _ConstA:
    """Constant A(t) = A with A_dot = 0 (Dai et al. treat a constant matrix)."""

    def __init__(self, A):
        self.A = np.asarray(A, dtype=float)

    def __call__(self, t):
        return self.A.copy()

    def dot(self, t):
        return np.zeros_like(self.A)


def dai_example1(T=2.0):
    P = MatrixInversion(n=3, T=T)
    P.A = _ConstA(DAI_A1)
    P.name = "matinv_dai2022_ex1"
    return P


class NoSolve(Problem):
    """Transparent wrapper whose solve_J raises: proves a model is solve-free."""

    def __init__(self, inner):
        self.inner = inner
        self.name, self.family = inner.name + "_nosolve", inner.family
        self.n_x, self.m, self.T = inner.n_x, inner.m, inner.T
        self.calls = {"JT": 0, "solve_J": 0}

    def data(self, t):
        return self.inner.data(t)

    def E(self, x, d):
        return self.inner.E(x, d)

    def dEdt(self, x, t):
        return self.inner.dEdt(x, t)

    def JT(self, x, d, v):
        self.calls["JT"] += 1
        return self.inner.JT(x, d, v)

    def solve_J(self, x, d, r):
        self.calls["solve_J"] += 1
        raise AssertionError("linear solve requested by a solve-free model")

    def x_star(self, t):
        return self.inner.x_star(t)

    def x0(self, rng, B):
        return self.inner.x0(rng, B)

    def capabilities(self):
        return self.inner.capabilities()


def _ctx(P, x, t, gamma, dEdt=None):
    d = P.data(t)
    return Context(problem=P, t=t, tau=1e-3, k=0, x=x, z=np.zeros((x.shape[0], 0)), d=d,
                   E=P.E(x, d), dEdt=P.dEdt(x, t) if dEdt is None else dEdt,
                   gamma=np.asarray(gamma, dtype=float).reshape(-1, 1))


def _near_start(P, B, scale, seed=0):
    rng = np.random.default_rng(seed)
    return P.x_star(0.0)[None] + scale * rng.uniform(-1, 1, (B, P.n_x))


@pytest.fixture(scope="module")
def tvne():
    return make_problem("nonlinear_equations")        # Guo et al. system = Zheng et al. Eq. (17)


# ---------------------------------------------------------------------------
# registry / metadata
# ---------------------------------------------------------------------------
def test_registry_and_flags():
    av = available()["models"]
    for k in KEYS:
        assert k in av, k
    lcc = make_model("lcc_znn")
    assert isinstance(lcc, LCCZNN) and lcc.requires == set() and lcc.uses_dEdt
    assert not lcc.supports_implicit and lcc.phi.kind == "exp_sig" and lcc.phi.r == 0.7
    # inverse_free_znn: primary source not obtainable -> documented, NOT registered
    assert "inverse_free_znn" not in av
    assert NOT_REGISTERED["inverse_free_znn"]["doi"] == "10.1016/j.neunet.2024.106462"
    assert make_model("lcc_znn", activation="linear").phi.kind == "linear"
    h = make_model("hgnn_znn")
    assert isinstance(h, HybridGNNZNN) and h.requires == {"matrix_inverse"}
    assert h.uses_dEdt is False and h.fuzzy is None                   # Eq. (10) verbatim
    assert make_model("hgnn_znn", feedforward=True).uses_dEdt is True
    assert make_model("hgnn_znn", scope="any").requires == set()
    fa = make_model("fa_gnn_znn")
    assert isinstance(fa.fuzzy, MamdaniMomFactor) and fa.fuzzy.nu_max == 2.5
    for k in KEYS:
        assert make_model(k).describe()["name"]
    with pytest.raises(ValueError):
        make_model("lcc_znn", r=1.2)
    with pytest.raises(ValueError):
        make_model("lcc_znn", lam=-1.0)
    with pytest.raises(ValueError):
        make_model("hgnn_znn", scope="bogus")


# ---------------------------------------------------------------------------
# LCCZNN
# ---------------------------------------------------------------------------
def test_lcc_rhs_matches_eq12_and_eq13(tvne):
    """Independent dense transcription of Eq. (12) with activation (13),
    including the Remark-1 regularisation, and of the comparison AFs (8), (9)."""
    P = tvne
    x = _near_start(P, 4, 0.4, seed=3)
    t = 0.7
    d = P.data(t)
    J = P.jacobian(x, d)
    f = P.E(x, d)
    ft = P.dEdt(x, t)
    mu = np.array([2.0, 4.0, 5.0, 7.0])
    nu = 0.5 * (f ** 2).sum(1)
    w = np.einsum("bij,bi->bj", J, f)
    for kind, r, lam, phi in [("exp_sig", 0.7, 0.0, nu ** 0.7 * np.exp(nu)),
                              ("exp_sig", 0.5, 1e-3, nu ** 0.5 * np.exp(nu)),
                              ("sig", 0.6, 0.0, nu ** 0.6),
                              ("sig_lin", 0.6, 0.0, nu ** 0.6 + nu)]:
        m = make_model("lcc_znn", activation=kind, r=r, lam=lam)
        xdot = m.rhs(_ctx(P, x, t, mu))[0]
        ref = -w * ((f * ft).sum(1) + mu * phi)[:, None] / ((w ** 2).sum(1) + lam)[:, None]
        assert np.allclose(xdot, ref, rtol=1e-12, atol=1e-14), kind
    # the design formula itself: f^T (J x_dot + f_t) = -mu phi(nu)   (Eq. (11))
    m = make_model("lcc_znn")
    xdot = m.rhs(_ctx(P, x, t, mu))[0]
    lhs = np.einsum("bi,bi->b", f, np.einsum("bij,bj->bi", J, xdot) + ft)
    assert np.allclose(lhs, -mu * nu ** 0.7 * np.exp(nu), rtol=1e-10)
    # x_dot is parallel to J^T f (explicit, pseudoinverse-free direction)
    cos = np.einsum("bi,bi->b", xdot, w) / (np.linalg.norm(xdot, axis=1) * np.linalg.norm(w, axis=1))
    assert np.allclose(np.abs(cos), 1.0)


def test_lcc_registry_activation_r_and_time_varying(tvne):
    """Verifier fix: an explicit r is forwarded to a registry activation (it was
    silently dropped), the registry default applies otherwise, and time-varying
    registry activations are frozen at ctx.t (rhs used to raise)."""
    assert make_model("lcc_znn", activation="sign_power", r=0.3).phi.describe()["r"] == 0.3
    assert make_model("lcc_znn", activation="sign_power").phi.describe()["r"] == \
        acts.get("sign_power").r
    P, t = tvne, 0.4
    x = _near_start(P, 2, 0.3, seed=1)
    d = P.data(t)
    f, ft, w = P.E(x, d), P.dEdt(x, t), P.JT(x, d, P.E(x, d))
    nu = 0.5 * (f ** 2).sum(1)
    sp = acts.get("strict_predefined", t_c=1.0)
    xdot = make_model("lcc_znn", activation="strict_predefined", t_c=1.0).rhs(_ctx(P, x, t, 3.0))[0]
    ref = -w * ((f * ft).sum(1) + 3.0 * sp(nu, t=t))[:, None] / (w ** 2).sum(1)[:, None]
    assert np.allclose(xdot, ref, rtol=1e-12)
    with pytest.raises(TypeError):                     # explicit r for an AF without r
        make_model("lcc_znn", activation="power", r=0.5)


def test_lcc_energy_follows_theorem1_in_continuous_time(tvne):
    """RK4 (tau = 1e-3): mu t + gamma(1-r, nu(t)) = gamma(1-r, nu0) (Eq. (14)) while
    nu decays; the settling time t_AF3 = gamma(1-r, nu0)/mu obeys the bound (15)."""
    P, mu, r = tvne, 4.0, 0.7
    x0 = _near_start(P, 3, 0.3)
    s = simulate(P, make_model("lcc_znn", r=r), make_discretization("rk4"), tau=1e-3, T=0.12, B=3,
                 gamma=mu, x0=x0, record_every=10)
    nu = 0.5 * s["residual"] ** 2
    assert np.all(nu[-1] < 0.5 * nu[0])
    inv = mu * s["t"][:, None] + lower_incomplete_gamma(1 - r, nu)
    assert np.allclose(inv, inv[0], rtol=1e-6)
    phi = ScalarEnergyActivation("exp_sig", r)
    tAF3 = phi.settling_time(nu[0], mu)
    assert np.all(tAF3 <= phi.theorem1_bound(nu[0], mu))
    # Theorem 3 / Remark 3 ordering: (13) beats (8) for every nu0 (t_AF3 / t_AF1 < 1)
    sig = ScalarEnergyActivation("sig", r)
    nus = np.geomspace(1e-3, 50, 30)
    assert np.all(phi.settling_time(nus, mu) < sig.settling_time(nus, mu))
    # verifier observation (not in the paper): (13) is FIXED-time, t_AF3 < Gamma(1-r)/mu for
    # every nu0, while the paper's bound (15) grows like nu0^(1-r) (loose for large nu0)
    ub = phi.uniform_bound(mu)
    assert np.isclose(ub, Gamma(1 - r) / mu)
    big = np.geomspace(1e-3, 700, 40)
    assert np.all(phi.settling_time(big, mu) <= ub)               # = ub in floating point for nu0 > ~40
    assert np.all(phi.settling_time(big[big < 20], mu) < ub)
    assert np.isclose(phi.settling_time(50.0, mu), ub, rtol=1e-12)
    assert phi.theorem1_bound(50.0, mu) > 2 * ub
    assert np.isinf(sig.uniform_bound(mu)) and np.isinf(ScalarEnergyActivation("sig_lin", r).uniform_bound(mu))


def test_lcc_theorem1_holds_with_stiff_reference_solver(tvne):
    """Theorem 1 in continuous time, through the model's own rhs: a stiff
    adaptive solver (Radau, rtol 1e-10) keeps mu t + gamma(1-r, nu(t)) constant
    to 1e-8 down to nu ~ 1e-5 (t = 0.95 t_AF3), i.e. the residual floor of the
    fixed-step runs is not a property of the continuous model."""
    P, mu, r = tvne, 4.0, 0.7
    m = make_model("lcc_znn", r=r)
    x0 = _near_start(P, 1, 0.3)[0]
    rhs = lambda t, x: m.rhs(_ctx(P, x[None], t, mu))[0][0]      # noqa: E731
    nu_of = lambda x, t: 0.5 * float((P.E(x[None], P.data(t)) ** 2).sum())  # noqa: E731
    nu0 = nu_of(x0, 0.0)
    tA = ScalarEnergyActivation("exp_sig", r).settling_time(nu0, mu)
    ts = np.linspace(0.0, 0.95 * tA, 20)
    sol = solve_ivp(rhs, (0.0, ts[-1]), x0, method="Radau", rtol=1e-10, atol=1e-14, t_eval=ts)
    assert sol.status == 0
    nu = np.array([nu_of(sol.y[:, i], t) for i, t in enumerate(sol.t)])
    inv = mu * sol.t + lower_incomplete_gamma(1 - r, nu)
    assert np.allclose(inv, lower_incomplete_gamma(1 - r, nu0), rtol=0, atol=1e-8)
    assert nu[-1] < 1e-4


def test_lcc_rk4_loses_theorem1_through_stiffness(tvne):
    """Mechanism of the discrete floor (verifier finding): the Jacobian of the
    LCCZNN field grows like 1/|f|; on system (17) its spectral radius exceeds
    2.785/tau (RK4 real-axis stability limit) for tau = 1e-4 near nu ~ 0.013,
    so RK4 with tau in [1e-4, 3e-4] leaves the Theorem-1 invariant at the same
    nu, while tau = 1e-5 (below the limit) follows it to t = 0.2."""
    P, mu, r = tvne, 4.0, 0.7
    m = make_model("lcc_znn", r=r)
    x0 = _near_start(P, 1, 0.3)
    nu0 = 0.5 * float((P.E(x0, P.data(0.0)) ** 2).sum())
    G0 = lower_incomplete_gamma(1 - r, nu0)
    brk = {}
    for tau in (3e-4, 1e-4, 1e-5):
        s = simulate(P, m, make_discretization("rk4"), tau=tau, T=0.2, B=1, gamma=mu, x0=x0,
                     record_every=max(1, int(round(1e-4 / tau))), online_stats=False)
        nu = 0.5 * s["residual"][:, 0] ** 2
        dev = np.abs(mu * s["t"] + lower_incomplete_gamma(1 - r, nu) - G0)
        brk[tau] = nu[np.argmax(dev > 1e-6)] if (dev > 1e-6).any() else None
        if tau == 1e-5:
            assert dev.max() < 1e-6
            xT = s["x_final"][0]
    assert brk[1e-5] is None
    assert 0.01 < brk[1e-4] < 0.02 and 0.01 < brk[3e-4] < 0.02
    assert abs(brk[3e-4] - brk[1e-4]) < 0.25 * brk[1e-4]            # tau-independent break point
    # spectral radius of d(x_dot)/dx at the tau = 1e-5 state (t = 0.2)
    h, J = 1e-7, np.zeros((4, 4))
    for i in range(4):
        e = np.zeros(4)
        e[i] = h
        J[:, i] = (m.rhs(_ctx(P, (xT + e)[None], 0.2, mu))[0][0]
                   - m.rhs(_ctx(P, (xT - e)[None], 0.2, mu))[0][0]) / (2 * h)
    rho = np.abs(np.linalg.eigvals(J)).max()
    assert rho * 1e-4 > 2.785 > rho * 1e-5


def test_lcc_discrete_floor_shrinks_with_tau(tvne):
    """Finite-time convergence is a continuous-time property: past the stiffness
    limit (previous test) RK4 chatters at a floor roughly proportional to tau
    (documented in the spec); it shrinks with tau."""
    P = tvne
    x0 = _near_start(P, 2, 0.3)
    fin = {}
    for tau in (1e-3, 1e-4):
        s = simulate(P, make_model("lcc_znn"), make_discretization("rk4"), tau=tau, T=1.0, B=2,
                     gamma=4.0, x0=x0, record_every=int(round(0.1 / tau)))
        assert not s["diverged"].any()
        fin[tau] = s["residual"][-1].max()
    assert fin[1e-4] < 0.05 and fin[1e-4] < 0.2 * fin[1e-3]


def test_lcc_needs_no_linear_solve(tvne):
    """Cost claim (Algorithm 1 / Remark 2): only J^T v, never a (pseudo)inverse.
    The wrapped problem raises on solve_J."""
    for act in ("exp_sig", "linear"):
        P = NoSolve(tvne)
        for disc in ("euler", "rk4"):
            s = simulate(P, make_model("lcc_znn", activation=act), make_discretization(disc),
                         tau=1e-2, T=0.5, B=2, gamma=4.0, x0=_near_start(tvne, 2, 0.2))
            assert np.isfinite(s["residual"]).all()
        assert P.calls["solve_J"] == 0 and P.calls["JT"] > 0
    # the reference ZNN does need the solve
    P = NoSolve(tvne)
    with pytest.raises(AssertionError):
        simulate(P, make_model("znn"), make_discretization("euler"), tau=1e-2, T=0.1,
                 x0=_near_start(tvne, 1, 0.2))


def test_lcc_linear_matches_secondary_restatement_of_nifznn(tvne):
    """Xiao et al. (IEEE TAI 2025) Eq. (10), their restatement of the (unregistered)
    NIFZNN of Chen et al.: x_dot = -J^T f ((eta/2)||f||^2 + f^T f_t)/||f^T J||^2
    equals lcc_znn(activation='linear'); nu_dot = -eta nu, nu = nu0 exp(-eta t)."""
    P, eta = tvne, 3.0
    x = _near_start(P, 3, 0.4, seed=5)
    a = make_model("lcc_znn", activation="linear").rhs(_ctx(P, x, 0.4, eta))[0]
    d = P.data(0.4)
    J, f, ft = P.jacobian(x, d), P.E(x, d), P.dEdt(x, 0.4)
    w = np.einsum("bij,bi->bj", J, f)
    ref = -w * (0.5 * eta * (f ** 2).sum(1) + (f * ft).sum(1))[:, None] / (w ** 2).sum(1)[:, None]
    assert np.allclose(a, ref, rtol=1e-12)
    s = simulate(P, make_model("lcc_znn", activation="linear"), make_discretization("rk4"),
                 tau=1e-3, T=0.3, B=3, gamma=eta, x0=_near_start(P, 3, 0.3), record_every=50)
    nu = 0.5 * s["residual"] ** 2
    assert np.allclose(nu, nu[0] * np.exp(-eta * s["t"])[:, None], rtol=1e-6)


def test_lcc_regularisation_and_zero_denominator(tvne):
    """Eq. (12) is 0/0 at f = 0: with an exactly zero error ZNN-Bench returns
    x_dot = 0 (documented choice).  At round-off-level f the unregularised field
    stays O(|df/dt| / sigma(J)) (direction-dependent limit: the singularity
    behind the discrete floor), while the Remark-1 regularisation (lam > 0)
    makes x_dot -> 0 as f -> 0."""
    P, t = tvne, 0.5
    xs = P.x_star(t)[None]
    ctx = _ctx(P, xs, t, 4.0)
    ctx.E = np.zeros_like(ctx.E)
    assert np.array_equal(make_model("lcc_znn").rhs(ctx)[0], np.zeros((1, 4)))
    v0 = np.linalg.norm(make_model("lcc_znn").rhs(_ctx(P, xs, t, 4.0))[0])
    assert np.isfinite(v0) and v0 > 1e-3
    for eps in (1e-4, 1e-6):
        x = xs + eps * np.array([[1.0, -0.5, 0.3, 0.2]])
        v_reg = np.linalg.norm(make_model("lcc_znn", lam=1e-3).rhs(_ctx(P, x, t, 4.0))[0])
        assert v_reg < 1e3 * eps


def test_lcc_redundant_robot_and_discrete_smoke():
    """Zheng et al. state the model for n >= m and apply it to a redundant arm
    (UR5, controller (18)); here: Panda (m = 3 < n = 7), Euler, h = 0.06,
    joints perturbed by 0.15 rad from the path start."""
    P = make_problem("robot_panda")
    rng = np.random.default_rng(0)
    x0 = P.x0(rng, 2) + 0.15 * rng.standard_normal((2, 7))
    s = simulate(P, make_model("lcc_znn"), make_discretization("euler"), tau=1e-2, T=2.0,
                 B=2, gamma=6.0, x0=x0)
    assert not s["diverged"].any()
    assert s["residual"][-1].max() < 0.05 * s["residual"][0].max()


@pytest.mark.parametrize("act", ["exp_sig", "linear"])
def test_lcc_discrete_euler_smoke_tvne(tvne, act):
    s = simulate(tvne, make_model("lcc_znn", activation=act), make_discretization("euler"),
                 tau=1e-2, T=2.0, B=3, gamma=4.0, x0=_near_start(tvne, 3, 0.3))
    assert not s["diverged"].any()
    assert np.nanmin(s["residual"], axis=0).max() < 0.5 * s["residual"][0].max()


# ---------------------------------------------------------------------------
# H-GNN-ZNN / FA-GNN-ZNN
# ---------------------------------------------------------------------------
def test_hgnn_rhs_matches_eq10_on_time_varying_inversion():
    """A X_dot = -gamma (A A^T + I) F(A X - I) [- A_dot X if feedforward]."""
    P = make_problem("matrix_inversion", n=3, kappa=4.0, seed=2)
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, (3, 9))
    t, g = 0.8, np.array([1.0, 2.0, 5.0])
    A, Ad = P.A(t), P.A.dot(t)
    for act, kw in [("linear", {}), ("sign_bi_power", {"r": 0.4}), ("power_sigmoid", {})]:
        for ff in (False, True):
            m = make_model("hgnn_znn", activation=act, feedforward=ff, **kw)
            xdot = m.rhs(_ctx(P, x, t, g))[0]
            F = m.act(P.E(x, P.data(t))).reshape(3, 3, 3)
            X = x.reshape(3, 3, 3)
            rhs = -g[:, None, None] * ((A @ A.T + np.eye(3)) @ F) - (Ad @ X if ff else 0.0)
            ref = np.linalg.solve(A, rhs).reshape(3, 9)
            assert np.allclose(xdot, ref, rtol=1e-10, atol=1e-12), (act, ff)


def test_hgnn_continuous_time_matrix_exponential_and_rates():
    """Constant A (Dai et al. Example 1), linear F, RK4: E(t) = expm(-gamma (A A^T + I) t) E(0);
    Table II ordering ECR(H-GNN-ZNN) = gamma (1 + alpha) > ECR(ZNN) = gamma > ECR(GNN) = gamma alpha
    (alpha = lambda_min(A A^T) = 0.17 here)."""
    P, g = dai_example1(), 1.0
    x0 = np.random.default_rng(1).uniform(-1, 1, (2, 9))
    s = simulate(P, make_model("hgnn_znn"), make_discretization("rk4"), tau=1e-3, T=2.0, B=2,
                 gamma=g, x0=x0, record_every=250)
    M = DAI_A1 @ DAI_A1.T + np.eye(3)
    for b in range(2):
        E0 = DAI_A1 @ x0[b].reshape(3, 3) - np.eye(3)
        ref = [np.linalg.norm(expm(-g * M * t) @ E0) for t in s["t"]]
        assert np.allclose(s["residual"][:, b], ref, rtol=1e-8)
    alpha = np.linalg.eigvalsh(DAI_A1 @ DAI_A1.T).min()
    bound = s["residual"][0] * np.exp(-g * (1 + alpha) * s["t"])[:, None]
    assert np.all(s["residual"] <= bound * (1 + 1e-9))
    res = {k: simulate(P, make_model(k), make_discretization("rk4"), tau=1e-3, T=2.0, B=2, gamma=g,
                       x0=x0)["residual"][-1] for k in ("hgnn_znn", "znn", "gnn")}
    assert np.all(res["hgnn_znn"] < res["znn"]) and np.all(res["hgnn_znn"] < res["gnn"])


def test_fuzzy_factor_pipeline():
    """Trapezoid (2), rules ZO->ZO ... PB->PB, max-min (3), mom (4); nu in [0, 2.5]."""
    fz = MamdaniMomFactor(u_max=1.0, nu_max=2.5, grid=2001)
    u = np.linspace(0.0, 2.0, 401)
    nu = fz(u)
    assert nu[0] == 0.0 and np.all(nu[u >= 1.0] == 2.5)
    assert np.all((nu >= 0) & (nu <= 2.5)) and np.all(np.diff(nu) >= -1e-12)
    # a single dominant rule: mom = midpoint of the alpha-cut of its output set
    H, h = 2.5 / 3, 1.0 / 3
    u1 = 0.3 * h                      # ZO fires 0.7, PS fires 0.3  ->  cut of ZO at 0.7
    assert np.isclose(fz(u1)[0], 0.5 * (H * (1 - 0.7)), atol=2 * 2.5 / 2000)
    u2 = 1.25 * h                     # PS fires 0.75, PM 0.25 -> cut of PS at 0.75: [0.75H, 1.25H]
    assert np.isclose(fz(u2)[0], H, atol=2 * 2.5 / 2000)
    # Eq. (2) shape and shoulders
    assert np.allclose(trapmf([0, 1, 2, 3, 4, 5], 1, 2, 3, 4), [0, 0, 1, 1, 0, 0])
    assert np.allclose(trapmf([0.5, 2.5], 1, 2, 3, 4), [0, 1])
    assert trapmf(0.0, 0, 0, 0, 1) == 1.0 and trapmf(9.0, 2, 3, np.inf, np.inf) == 1.0
    # degenerate edges are the limits of Eq. (2): m == n > 0 is a vertical rising edge
    assert np.allclose(trapmf([1.0, 2.0, 2.5, 3.0, 3.5], 2, 2, 3, 3), [0, 1, 1, 1, 0])
    with pytest.raises(ValueError):
        MamdaniMomFactor(input_sets={"ZO": (0, 0, 0, 1)})


def test_fa_gnn_znn_gain_and_faster_than_hgnn():
    """Eq. (11): the FA-GNN-ZNN field is the H-GNN-ZNN field with gain gamma + nu(||E||_F);
    on Example 1 it converges faster (Fig. 2) and keeps the rate (Theorem 1)."""
    P, g = dai_example1(), 1.0
    x = np.random.default_rng(4).uniform(-1, 1, (3, 9))
    fa = make_model("fa_gnn_znn")
    ctx = _ctx(P, x, 0.0, g)
    nu = fa.fuzzy(np.linalg.norm(ctx.E, axis=1))
    a = fa.rhs(ctx)[0]
    b = make_model("hgnn_znn").rhs(_ctx(P, x, 0.0, g + nu))[0]
    assert np.allclose(a, b, rtol=1e-12)
    x0 = x[:2]
    rf = simulate(P, fa, make_discretization("rk4"), tau=1e-3, T=2.0, B=2, gamma=g, x0=x0,
                  record_every=100)
    rh = simulate(P, make_model("hgnn_znn"), make_discretization("rk4"), tau=1e-3, T=2.0, B=2,
                  gamma=g, x0=x0, record_every=100)
    assert np.all(rf["residual"][1:] < rh["residual"][1:])
    alpha = np.linalg.eigvalsh(DAI_A1 @ DAI_A1.T).min()
    assert np.all(rf["residual"] <= rf["residual"][0] * np.exp(-g * (1 + alpha) * rf["t"])[:, None]
                  * (1 + 1e-9))


def test_dai_theorem1_step15_needs_linear_F():
    """Survey note (verifier): the step of Dai et al.'s proof (15),
    Trace(E^T M F(E)) >= Trace(E^T M E) with M = A A^T + I, does not hold for the
    PsAf with Dai's Example 1 (non-diagonal M), although |F(e)| >= |e| element-wise;
    at that point the final rate bound (1 + alpha)|e|^2 still holds."""
    M = DAI_A1 @ DAI_A1.T + np.eye(3)
    F = acts.get("power_sigmoid", p=3, xi=4.0)            # Dai: alpha >= 3, beta > 2
    e = np.array([0.17262381, -0.31702272, -1.01099756])
    g = np.linspace(-3, 3, 6001)
    assert np.all(np.abs(F(g)) >= np.abs(g) - 1e-15)
    assert e @ M @ F(e) < e @ M @ e - 2.0
    alpha = np.linalg.eigvalsh(DAI_A1 @ DAI_A1.T).min()
    assert e @ M @ F(e) > (1 + alpha) * (e @ e)


def test_hgnn_requires_capability_and_scope_any():
    P = make_problem("linear_system", n=3, kappa=4.0, seed=0)
    x = np.random.default_rng(0).uniform(-1, 1, (2, 3))
    with pytest.raises(ValueError):
        make_model("hgnn_znn").rhs(_ctx(P, x, 0.1, 2.0))
    m = make_model("hgnn_znn", scope="any", feedforward=True)
    xdot = m.rhs(_ctx(P, x, 0.1, 2.0))[0]
    d = P.data(0.1)
    A = d["A"]
    E = P.E(x, d)
    ref = -2.0 * E @ A + np.linalg.solve(A, (-2.0 * E - P.dEdt(x, 0.1)).T).T
    assert np.allclose(xdot, ref, rtol=1e-10)


@pytest.mark.parametrize("key,kw", [("hgnn_znn", {"feedforward": True}),
                                    ("fa_gnn_znn", {"feedforward": True}),
                                    ("hgnn_znn", {})])
def test_hybrid_discrete_euler_smoke(key, kw):
    """Euler on time-varying inversion (kappa = 3, gamma = 2, tau = 1e-2): the
    effective gain is gamma (1 + sigma_max^2), so h must stay below 2/(1+sigma_max^2)."""
    P = make_problem("matrix_inversion", n=3, kappa=3.0, seed=1, T=3.0)
    s = simulate(P, make_model(key, **kw), make_discretization("euler"), tau=1e-2, T=3.0, B=2,
                 gamma=2.0, seed=0)
    assert not s["diverged"].any()
    assert s["residual"][-1].max() < 0.2 * s["residual"][0].max()
    if kw.get("feedforward"):
        assert s["residual"][-1].max() < 1e-2


def test_hybrid_euler_instability_threshold():
    """Discrete-time reality of Eq. (10): Euler is stable only for
    gamma tau (1 + sigma_max(A)^2) < 2 on constant A (Example 1)."""
    P = dai_example1(T=3.0)
    smax2 = np.linalg.eigvalsh(DAI_A1 @ DAI_A1.T).max()
    x0 = np.random.default_rng(2).uniform(-1, 1, (1, 9))
    tau = 1e-2
    g_ok, g_bad = 0.8 * 2 / (tau * (1 + smax2)), 1.2 * 2 / (tau * (1 + smax2))
    ok = simulate(P, make_model("hgnn_znn"), make_discretization("euler"), tau=tau, T=3.0,
                  gamma=g_ok, x0=x0)
    bad = simulate(P, make_model("hgnn_znn"), make_discretization("euler"), tau=tau, T=3.0,
                   gamma=g_bad, x0=x0)
    assert ok["residual"][-1, 0] < 1e-6 and (bad["diverged"][0] or bad["residual"][-1, 0] > 1e3)


# ---------------------------------------------------------------------------
# integration with the smoke matrix (no cell may raise)
# ---------------------------------------------------------------------------
def test_mini_smoke_for_new_keys():
    spec = importlib.util.spec_from_file_location("smoke_matrix", ROOT / "experiments" / "smoke_matrix.py")
    sm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sm)
    bad = []
    for p in available()["problems"]:
        for m in KEYS:
            for d in ("euler", "implicit_euler_err", "rk4"):
                r = sm.run_cell(p, m, d, T=0.05, tau=1e-2, gamma=10.0, B=2, seed=0, max_wall=30.0,
                                default_x0=False)
                if r["status"] in ("error", "timeout"):
                    bad.append((p, m, d, r["message"]))
    assert not bad, bad
