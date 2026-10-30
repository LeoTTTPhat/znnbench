"""Tests for ZNN model variants (znnbench/models/variants.py) and gain schedules
(znnbench/models/gains.py).  Continuous-time properties are checked with RK4
at small tau on the n = 4 linear system; discrete smoke tests use Euler."""
import numpy as np
import pytest

from znnbench import make_discretization, make_model, make_problem, simulate
from znnbench.core import Context
from znnbench.metrics import mssre
from znnbench.models import gains as G
from znnbench.models.variants import GZNN, LiActivation, SPTCANFOZNN

KEYS = ["vpznn", "vpznn_exp", "ieznn", "dieznn", "hntznn", "sptc_nt_znn", "sptc_an_foznn", "gznn"]


def linsys(T=4.0, seed=1):
    return make_problem("linear_system", n=4, kappa=10, seed=seed, T=T)


def rk4_traj(P, model, x0, gamma, tau, T, noise=None):
    """Reference RK4 integration with exact data/derivatives; returns the time
    grid, error vectors E(t) (N+1, B, m) and states x(t) (N+1, B, n)."""
    B = x0.shape[0]
    gam = np.full((B, 1), float(gamma))

    def f(t, x, z):
        d = P.data(t)
        ctx = Context(problem=P, t=t, tau=tau, k=0, x=x, z=z, d=d, E=P.E(x, d),
                      dEdt=P.dEdt(x, t), gamma=gam)
        xd, zd = model.rhs(ctx)
        if noise is not None:
            xd = xd + P.solve_J(x, d, noise(t, (B, P.m)))
        return xd, zd

    N = int(round(T / tau))
    x, z = np.array(x0, float), model.z0(P, B)
    ts, Es, Xs = [0.0], [P.E(x, P.data(0.0))], [x.copy()]
    for k in range(N):
        t = k * tau
        k1 = f(t, x, z)
        k2 = f(t + tau / 2, x + tau / 2 * k1[0], z + tau / 2 * k1[1])
        k3 = f(t + tau / 2, x + tau / 2 * k2[0], z + tau / 2 * k2[1])
        k4 = f(t + tau, x + tau * k3[0], z + tau * k3[1])
        x = x + tau / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        z = z + tau / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        ts.append((k + 1) * tau)
        Es.append(P.E(x, P.data((k + 1) * tau)))
        Xs.append(x.copy())
    return np.array(ts), np.array(Es), np.array(Xs)


def x0_near(P, scale, seed=0, B=1):
    rng = np.random.default_rng(seed)
    return P.x_star(0.0)[None] + scale * rng.standard_normal((B, P.n_x))


# ---------------------------------------------------------------------------
# registry / structure
# ---------------------------------------------------------------------------
def test_registry_and_aux_sizes():
    P = linsys()
    for k in KEYS:
        m = make_model(k)
        assert isinstance(m.describe(), dict) and m.describe()["name"]
    assert make_model("ieznn").n_z(P) == P.m
    assert make_model("dieznn").n_z(P) == 2 * P.m
    assert make_model("hntznn", omegas=[1.0, 2.0, 3.0]).n_z(P) == 6 * P.m
    assert make_model("hntznn", freqs=[0.5]).omegas[0] == pytest.approx(np.pi)
    for k in ["vpznn", "vpznn_exp", "sptc_nt_znn", "sptc_an_foznn", "gznn"]:
        assert make_model(k).n_z(P) == 0


def test_li_activation_is_twice_sign_bi_power():
    from znnbench.activations import get
    e = np.linspace(-3, 3, 13)
    assert np.allclose(LiActivation(r=0.5)(e), 2 * get("sign_bi_power", r=0.5)(e))


# ---------------------------------------------------------------------------
# gain schedules
# ---------------------------------------------------------------------------
def test_gain_schedule_values():
    assert G.PowerVPGain(p=2)(0.0) == 2.0 and G.PowerVPGain(p=2)(3.0) == 11.0
    assert G.PowerVPGain(p=1.5, zeta=0.5)(4.0) == pytest.approx(8.5)
    assert G.ExpVPGain(p=3)(0.0) == 4.0 and G.ExpVPGain(p=3)(2.0) == 12.0
    assert G.EpsExpGain(eps=0.5)(1.0) == pytest.approx(0.5 * np.e)
    g = G.ArccotExpGain(beta=2, lam1=1, lam2=0.5)
    assert g(0.0) == pytest.approx(2 * np.exp(np.pi / 2))
    assert g(3.0) == pytest.approx(2 * np.exp(np.arctan(1 / 3.0) + 1.5))
    pt = G.PredefinedTimeGain(t_c=2.0)
    assert pt(1.5) == pytest.approx(2.0) and pt(1.999) == pytest.approx(1000.0)
    assert pt(2.0) == 1.0 and pt(5.0) == 1.0
    assert G.PredefinedTimeGain(t_c=2.0, cap=50.0)(1.999) == 50.0
    c = G.ConformableGain(alpha=0.5)
    assert c(4.0) == pytest.approx(0.5) and c(0.0) == pytest.approx(1e-3 ** -0.5)
    assert G.PowerVPGain(p=2, cap=5.0)(10.0) == 5.0
    assert G.get("constant")(7.0) == 1.0
    with pytest.raises(ValueError):
        G.PowerVPGain(p=-1)


def test_varying_gains_grow_as_specified():
    t = np.linspace(0.0, 5.0, 501)
    for g in [G.PowerVPGain(p=0.5), G.PowerVPGain(p=2), G.ExpVPGain(p=1.5), G.EpsExpGain(),
              G.PredefinedTimeGain(t_c=6.0)]:
        v = g(t)
        assert np.all(np.diff(v) > 0), g.describe()
    # closed-form integrals used in the super-exponential test
    for g in [G.PowerVPGain(p=1.5, zeta=0.3), G.ExpVPGain(p=2.5)]:
        tt = np.linspace(0, 3, 30001)
        num = np.concatenate([[0.0], np.cumsum(0.5 * (g(tt[1:]) + g(tt[:-1])) * np.diff(tt))])
        assert np.allclose(num[::3000], g.integral(tt[::3000]), rtol=1e-6, atol=1e-8)
    # the predefined-time gain blows up like 1/(t_c - t)
    pt = G.PredefinedTimeGain(t_c=1.0)
    assert pt(1 - 1e-6) == pytest.approx(1e6, rel=1e-6)


@pytest.mark.parametrize("key,kw,gain", [("vpznn", dict(p=1.0), G.PowerVPGain(p=1.0)),
                                         ("vpznn", dict(p=2.0, zeta=0.5), G.PowerVPGain(p=2.0, zeta=0.5)),
                                         ("vpznn_exp", dict(p=2.0), G.ExpVPGain(p=2.0))])
def test_vpznn_superexponential_error_decay(key, kw, gain):
    """Linear VP-ZNN: E(t) = E(0) exp(-gamma int_0^t g) exactly in continuous time."""
    P = linsys()
    x0 = np.random.default_rng(0).uniform(-2, 2, (2, 4))
    ts, Es, _ = rk4_traj(P, make_model(key, **kw), x0, gamma=1.0, tau=1e-3, T=2.0)
    for t in [0.5, 1.0, 2.0]:
        i = int(round(t / 1e-3))
        pred = Es[0] * np.exp(-gain.integral(t))
        assert np.abs(Es[i] - pred).max() < 1e-7 * np.abs(Es[0]).max()


# ---------------------------------------------------------------------------
# IEZNN
# ---------------------------------------------------------------------------
def test_ieznn_exact_error_dynamics():
    """lam = gamma^2/4: E(s) = s E0 / (s + a)^2, a = gamma/2  =>  E(t) = E0 e^{-at}(1 - a t)."""
    P = linsys()
    gamma = 6.0
    x0 = np.random.default_rng(1).uniform(-2, 2, (1, 4))
    ts, Es, _ = rk4_traj(P, make_model("ieznn"), x0, gamma=gamma, tau=1e-3, T=2.0)
    a = gamma / 2
    pred = Es[0][None] * (np.exp(-a * ts) * (1 - a * ts))[:, None, None]
    assert np.abs(Es - pred).max() < 1e-8 * np.abs(Es[0]).max()


def test_ieznn_rejects_constant_noise_znn_keeps_floor():
    P = linsys()
    rk4 = make_discretization("rk4")
    kw = dict(tau=1e-3, B=2, gamma=10.0, seed=0, noise=lambda t, rng, shape: np.full(shape, 1.0))
    e_znn = mssre(simulate(P, make_model("znn"), rk4, **kw))
    e_ie = mssre(simulate(P, make_model("ieznn"), rk4, **kw))
    floor = np.sqrt(P.m) * 1.0 / 10.0                     # ||nu|| / gamma
    assert np.allclose(e_znn, floor, rtol=0.02)
    assert np.all(e_ie < 1e-3)


def test_ieznn_linear_noise_limit_is_slope_over_lambda():
    P = linsys()
    kw = dict(tau=1e-3, B=1, gamma=10.0, seed=0,
              noise=lambda t, rng, shape: np.full(shape, 1.0 * t))
    e_ie = mssre(simulate(P, make_model("ieznn"), make_discretization("rk4"), **kw))[0]
    assert e_ie == pytest.approx(np.sqrt(P.m) * 1.0 / 25.0, rel=0.02)   # lam = gamma^2/4 = 25


# ---------------------------------------------------------------------------
# DIEZNN
# ---------------------------------------------------------------------------
def test_dieznn_characteristic_polynomial():
    """Noise-free DIEZNN: E(s) = E0 s^2 / ((s+1)(s+xi)^2)  (paper Eq. 14)."""
    P = linsys()
    xi = 4.0
    x0 = np.random.default_rng(2).uniform(-2, 2, (1, 4))
    ts, Es, _ = rk4_traj(P, make_model("dieznn"), x0, gamma=xi, tau=1e-3, T=3.0)
    A = 1.0 / (xi - 1) ** 2
    Bc, C = 1.0 - A, xi ** 2 / (1.0 - xi)
    h = A * np.exp(-ts) + Bc * np.exp(-xi * ts) + C * ts * np.exp(-xi * ts)
    assert np.abs(Es - Es[0][None] * h[:, None, None]).max() < 1e-8 * np.abs(Es[0]).max()


def test_dieznn_rejects_linear_noise():
    P = linsys(T=8.0)
    rk4 = make_discretization("rk4")
    kw = dict(tau=2e-3, B=1, gamma=10.0, seed=0, x0=P.x_star(0.0)[None],
              noise=lambda t, rng, shape: np.full(shape, 1.0 * t + 0.5))
    e_die = mssre(simulate(P, make_model("dieznn"), rk4, **kw))[0]
    e_ie = mssre(simulate(P, make_model("ieznn"), rk4, **kw))[0]
    e_znn = mssre(simulate(P, make_model("znn"), rk4, **kw))[0]
    assert e_die < 1e-3
    assert e_ie == pytest.approx(np.sqrt(P.m) / 25.0, rel=0.05)    # IEZNN floor Q/lam
    assert e_znn > 1.0                                             # ZNN error grows with t


# ---------------------------------------------------------------------------
# HNTZNN
# ---------------------------------------------------------------------------
def test_hntznn_rejects_matched_harmonic_noise():
    P = linsys(T=6.0)
    rk4 = make_discretization("rk4")
    w = 2 * np.pi
    kw = dict(tau=1e-3, B=1, gamma=10.0, seed=0,
              noise=lambda t, rng, shape: np.full(shape, np.sin(w * t + 0.3)))
    e_znn = mssre(simulate(P, make_model("znn"), rk4, **kw))[0]
    e_hn = mssre(simulate(P, make_model("hntznn", omegas=[w], lam=2.0, activation="linear"), rk4, **kw))[0]
    e_mis = mssre(simulate(P, make_model("hntznn", omegas=[np.pi], lam=2.0, activation="linear"),
                           rk4, **kw))[0]
    # paper's Li activation (default): converges too, but more slowly near E = 0
    e_li = mssre(simulate(P, make_model("hntznn", omegas=[w], lam=2.0), rk4, **kw))[0]
    assert e_znn > 0.1                                   # ~ sqrt(m)/sqrt(gamma^2 + w^2)
    assert e_hn < 1e-3
    assert e_mis > 0.05                                  # wrong internal-model frequency
    assert e_li < e_znn / 10


def test_hntznn_multi_harmonic():
    """Eq. (14) with two known frequencies.  The closed loop has a lightly
    damped pole pair between the two frequencies (slowest pole ~ -0.95 for
    gamma = 20, lam = 2, w = 2pi, 10pi), hence the longer horizon."""
    P = linsys(T=12.0)
    rk4 = make_discretization("rk4")
    ws = [2 * np.pi, 10 * np.pi]
    noise = lambda t, rng, shape: np.full(shape, np.sin(ws[0] * t) + 0.5 * np.sin(ws[1] * t + 1.0))
    kw = dict(tau=2e-3, B=1, gamma=20.0, seed=0, noise=noise)
    e_lin = mssre(simulate(P, make_model("hntznn", omegas=ws, lam=2.0, activation="linear"), rk4, **kw))[0]
    e_znn = mssre(simulate(P, make_model("znn"), rk4, **kw))[0]
    assert e_lin < 1e-3 and e_lin < e_znn / 100


# ---------------------------------------------------------------------------
# strictly predefined-time models
# ---------------------------------------------------------------------------
def test_sptc_nt_znn_reaches_zero_at_tc_continuous():
    """Before t_c: E(t) = E0 ((t_c - t)/t_c)^gamma (exact solution)."""
    P = linsys()
    gamma, t_c = 2.0, 1.0
    x0 = np.random.default_rng(3).uniform(-2, 2, (1, 4))
    ts, Es, _ = rk4_traj(P, make_model("sptc_nt_znn", t_c=t_c), x0, gamma=gamma, tau=1e-3, T=0.995)
    n0 = np.linalg.norm(Es[0])
    for t in [0.5, 0.9, 0.99, 0.995]:
        i = int(round(t / 1e-3))
        assert np.linalg.norm(Es[i]) / n0 == pytest.approx(((t_c - t) / t_c) ** gamma, rel=1e-3)
    assert np.linalg.norm(Es[-1]) < 1e-4 * n0                        # ~0 before t_c


def test_sptc_nt_znn_after_tc_bounded_noise():
    P = linsys(T=3.0)
    rk4 = make_discretization("rk4")
    kw = dict(tau=1e-3, B=1, gamma=2.0, seed=0,
              noise=lambda t, rng, shape: np.full(shape, 0.2 * np.cos(t)))
    s = simulate(P, make_model("sptc_nt_znn", t_c=1.0), rk4, **kw)
    z = simulate(P, make_model("znn"), rk4, **kw)
    after = s["t"] >= 1.05
    assert np.isfinite(s["residual"]).all()
    assert s["residual"][after].max() < 0.02                        # chattering level ~ zeta*tau
    assert mssre(z)[0] > 0.05                                       # linear ZNN keeps an O(Delta/gamma) floor


def test_sptc_rhs_match_paper_activations():
    """Independent transcription of the activations: SPTC-AN-FOZNN Eq. (14)
    inserted in Eq. (12) (dE/dt = -gamma t^(alpha-1) Phi(E)), and the
    SPTC-NT-ZNN row of Table 1 of Yang et al. 2025 (dE/dt = -gamma Phi(E)),
    versus the simplified error dynamics coded in the models."""
    P = linsys()
    rng = np.random.default_rng(11)
    x = rng.standard_normal((3, 4))
    gam = np.array([[0.7], [2.0], [5.0]])
    alpha, t_c, zeta, Delta = 0.6, 1.3, 2.5, 0.5
    t_p = (1 - np.exp(-np.pi / (2 * np.sqrt(zeta * (zeta - Delta))))) * t_c
    an = SPTCANFOZNN(alpha=alpha, t_c=t_c, noise_bound=Delta, zeta=zeta)
    nt = make_model("sptc_nt_znn", t_c=t_c, p=0.5, zeta=zeta)

    def phi_an(e, t, g):
        n = np.linalg.norm(e, axis=1, keepdims=True)
        if t < t_c:
            return (e / (t_c - t) + g * n ** 2 / (t_c - t) ** 2 * e / n) / (g * t_c ** (alpha - 1))
        return (e / (t_p - t) + (zeta + g * t ** (alpha - 1) * n ** 2 / (t_p - t) ** 2) * e / n) \
            / (g * t ** (alpha - 1))

    def phi_nt(e, t, g):
        if t < t_c:
            return e / (t_c - t)
        return e + np.abs(e) ** 0.5 * np.sign(e) + (zeta / g) * np.sign(e)

    for t in [0.2, 1.0, 1.29, 1.3, 2.4]:
        d = P.data(t)
        ctx = Context(problem=P, t=t, tau=1e-3, k=0, x=x, z=np.zeros((3, 0)), d=d, E=P.E(x, d),
                      dEdt=P.dEdt(x, t), gamma=gam)
        r_an = -gam * t ** (alpha - 1) * phi_an(ctx.E, t, gam)
        r_nt = -gam * phi_nt(ctx.E, t, gam)
        assert np.allclose(an.rhs(ctx)[0], P.solve_J(x, d, r_an - ctx.dEdt), rtol=1e-10, atol=1e-12)
        assert np.allclose(nt.rhs(ctx)[0], P.solve_J(x, d, r_nt - ctx.dEdt), rtol=1e-10, atol=1e-12)


def test_sptc_an_foznn_lyapunov_bound_and_predefined_time():
    """Theorem 1: V = ||E||/(t_c - t) is non-increasing and
    V(t) <= 1/(1/V0 + gamma ln(t_c/(t_c - t))); hence ||E|| -> 0 at t_c."""
    P = linsys()
    gamma, t_c = 2.0, 1.0
    model = make_model("sptc_an_foznn", alpha=0.5, t_c=t_c)
    x0 = x0_near(P, 0.3, seed=4)
    tau = 2.5e-4
    ts, Es, _ = rk4_traj(P, model, x0, gamma=gamma, tau=tau, T=0.99)
    nE = np.linalg.norm(Es[:, 0], axis=1)
    V = nE / (t_c - ts)
    assert np.all(np.diff(V) <= 1e-9 * V[0])
    bound = model.lyapunov_bound(ts, V[0], gamma)
    assert np.all(V <= bound * (1 + 1e-6))
    assert nE[-1] < 1e-3 * nE[0]


def test_sptc_an_foznn_parameters_and_noise_after_tc():
    m = SPTCANFOZNN(noise_bound=0.5)
    assert m.zeta == 2.5
    assert m.t_p == pytest.approx((1 - np.exp(-np.pi / (2 * np.sqrt(2.5 * 2.0)))) * 1.0)
    assert 0 < m.t_p < m.t_c
    with pytest.raises(ValueError):
        SPTCANFOZNN(noise_bound=1.0, zeta=0.5)               # needs zeta > Delta
    P = linsys(T=2.0)
    kw = dict(tau=5e-4, B=1, gamma=2.0, seed=0, x0=x0_near(P, 0.3, seed=5),
              noise=lambda t, rng, shape: np.full(shape, 0.2 * np.cos(t)))
    s = simulate(P, m, make_discretization("rk4"), **kw)
    after = s["t"] >= 1.05
    assert np.isfinite(s["residual"]).all()
    assert s["residual"][after].max() < 0.02


# ---------------------------------------------------------------------------
# gradient-ZNN hybrid
# ---------------------------------------------------------------------------
def test_gznn_gradient_error_decays_exponentially():
    """GZNN: E_G = J^T E obeys dE_G/dt = -gamma E_G exactly (needs the dJ/dt term)."""
    P = linsys()
    gamma, tau = 3.0, 1e-3
    x0 = np.random.default_rng(6).uniform(-2, 2, (2, 4))

    def EG(ts, Es, Xs):
        return np.array([P.JT(Xs[i], P.data(t), Es[i]) for i, t in enumerate(ts)])

    ts, Es, Xs = rk4_traj(P, make_model("gznn"), x0, gamma=gamma, tau=tau, T=1.5)
    eg = EG(ts, Es, Xs)
    pred = eg[0][None] * np.exp(-gamma * ts)[:, None, None]
    assert np.abs(eg - pred).max() < 1e-6 * np.abs(eg[0]).max()
    # the plain ZNN does not have this property (the test discriminates)
    ts, Es, Xs = rk4_traj(P, make_model("znn"), x0, gamma=gamma, tau=tau, T=1.5)
    eg = EG(ts, Es, Xs)
    assert np.abs(eg - eg[0][None] * np.exp(-gamma * ts)[:, None, None]).max() > 1e-3 * np.abs(eg[0]).max()


def test_gznn_dense_jacobian_and_jdot():
    P = linsys()
    x = np.random.default_rng(7).standard_normal((2, 4))
    J = GZNN.dense_J(P, x, P.data(0.7))
    assert np.allclose(J, P.A(0.7)[None])
    g = GZNN()
    for t in [0.0, 0.7]:          # dJ/dt from the x-derivative of dE/dt (no data sampling)
        assert np.allclose(g.jdot_analytic(P, x, t), P.A.dot(t)[None], atol=1e-9)
    # nonlinear-in-data problem: agrees with a central difference of J over exact data
    Q = make_problem("sylvester")
    xq = Q.x0(np.random.default_rng(1), 2)
    fd = (GZNN.dense_J(Q, xq, Q.data(0.4 + 1e-6)) - GZNN.dense_J(Q, xq, Q.data(0.4 - 1e-6))) / 2e-6
    assert np.abs(g.jdot_analytic(Q, xq, 0.4) - fd).max() < 1e-6 * np.abs(fd).max()
    with pytest.raises(ValueError):                                          # dense-memory guard
        GZNN(max_dense=10).z0(P, 1)
    with pytest.raises(ValueError):
        GZNN(jdot="exact")


def test_gznn_jdot_follows_information_regime():
    """dJ/dt must use the same information as dE/dt: the model never samples
    problem data itself (same number of data calls as ZNN), mirrors
    simulate's bd1/bd2/none regimes, and with info='none' (dE/dt = dJ/dt = 0)
    GZNN on a square J reduces exactly to ZNN."""
    from znnbench.wrappers import CountingProblem
    P = linsys(T=0.3)
    eu = make_discretization("euler")
    calls = {}
    for key in ["znn", "gznn"]:
        C = CountingProblem(P)
        simulate(C, make_model(key), eu, tau=1e-2, B=1, gamma=10.0)
        calls[key] = C.calls["data"]
    assert calls["gznn"] == calls["znn"]
    for info in ["analytic", "bd1", "bd2", "none"]:
        m = make_model("gznn")
        simulate(P, m, eu, tau=1e-2, B=2, gamma=10.0, info=info)
        assert m.last_regime == info
    kw = dict(tau=1e-2, B=2, gamma=10.0, info="none")
    s_g = simulate(P, make_model("gznn"), eu, **kw)
    s_z = simulate(P, make_model("znn"), eu, **kw)
    assert np.allclose(s_g["residual"], s_z["residual"], rtol=1e-8)
    # bd1: dJ/dt equals the backward difference of J over the two latest samples
    m = GZNN(jdot="bd1")
    x = np.random.default_rng(3).standard_normal((1, 4))
    tau = 1e-2
    for k in range(3):
        d = P.data(k * tau)
        ctx = Context(problem=P, t=k * tau, tau=tau, k=k, x=x, z=np.zeros((1, 0)), d=d,
                      E=P.E(x, d), dEdt=np.zeros((1, 4)), gamma=np.full((1, 1), 10.0))
        Jd = m.jdot_matrix(ctx)
    assert np.allclose(Jd[0], (P.A(2 * tau) - P.A(tau)) / tau)


def test_gznn_wrapped_data_and_redundant_robot():
    """Regression: (i) with jittered data samples (independent jitter per
    instant), the old finite difference of J over data(t +- dt) amplified the
    jitter by 1/dt; (ii) on a redundant robot (m < n_x) J^T J is singular and
    needs the Tikhonov term of Eq. (39)."""
    from znnbench.wrappers import JitteredProblem
    P = JitteredProblem(linsys(T=2.0), jitter_std=1e-3, seed=0)
    kw = dict(tau=1e-3, B=1, gamma=10.0, seed=0)
    eu = make_discretization("euler")
    e_g = mssre(simulate(P, make_model("gznn"), eu, **kw))[0]
    e_z = mssre(simulate(P, make_model("znn"), eu, **kw))[0]
    assert e_g < 1.5 * e_z
    R = make_problem("robot_panda", T=0.3)
    assert R.m < R.n_x
    s = simulate(R, make_model("gznn"), eu, tau=1e-3, B=1, gamma=10.0)
    z = simulate(R, make_model("znn"), eu, tau=1e-3, B=1, gamma=10.0)
    assert not s["diverged"].any()
    assert mssre(s)[0] < 2 * mssre(z)[0] + 1e-9


# ---------------------------------------------------------------------------
# discrete-time smoke tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key,kw,gamma", [
    ("vpznn", {}, 2.0), ("vpznn_exp", dict(p=1.5), 2.0), ("ieznn", {}, 10.0), ("dieznn", {}, 10.0),
    ("hntznn", dict(activation="linear"), 10.0), ("hntznn", {}, 10.0),
    ("sptc_nt_znn", {}, 2.0), ("sptc_an_foznn", {}, 2.0), ("gznn", {}, 10.0)])
def test_discrete_euler_smoke(key, kw, gamma):
    P = linsys(T=3.0)
    x0 = x0_near(P, 0.2, seed=8, B=2)
    s = simulate(P, make_model(key, **kw), make_discretization("euler"), tau=1e-2, B=2,
                 gamma=gamma, x0=x0)
    assert not s["diverged"].any()
    assert np.all(s["residual"][-1] < s["residual"][0])


def test_vpznn_implicit_and_cap():
    P = linsys(T=8.0)
    # an uncapped power VP gain eventually makes h = gamma tau (2 + t^2) > 2 with Euler (t > 2.8) ...
    kw = dict(tau=0.1, gamma=2.0)
    s = simulate(P, make_model("vpznn", p=2.0), make_discretization("euler"), **kw)
    assert s["diverged"].all()
    # ... a saturation cap (h <= 1) or the implicit (resolvent) error step keeps it stable
    s_cap = simulate(P, make_model("vpznn", p=2.0, cap=5.0), make_discretization("euler"), **kw)
    s_imp = simulate(P, make_model("vpznn", p=2.0), make_discretization("implicit_euler_err"), **kw)
    assert not s_cap["diverged"].any() and not s_imp["diverged"].any()
