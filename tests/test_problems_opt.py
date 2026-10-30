"""Tests for znnbench/problems/optimization.py (area: problems_opt).

Checks, for every registered variant: E(x*) ~ 0, dE/dt vs central finite
differences, dense Jacobian / solve_J / JT vs a finite-difference Jacobian,
batch consistency, and a ZNN-Euler convergence smoke test.  Plus: exact
reproduction of Table 1 (Euler DTZNN rows) of Frontiers Neurorobot. 2025
(DOI 10.3389/fnbot.2025.1576473) on the Guo et al. (2018) system, empirical
O(tau^2) order of ZNN-Euler at fixed h, and the perturbed Fischer-Burmeister
specifics of ``tvqp_ineq`` (complementarity eps/2, bias vs the exact QP
solution, independent brute-force QP check, smoothing-parameter effects).
"""
import itertools

import numpy as np
import pytest

from znnbench import available, make_discretization, make_model, make_problem, simulate
from znnbench.metrics import empirical_order, mssre
from znnbench.problems.optimization import pfb

CONFIGS = [
    ("nonlinear_equations", {}),
    ("nonlinear_equations", dict(variant="manufactured", n=5, kappa=10, seed=2)),
    ("tv_unconstrained_opt", {}),
    ("tv_unconstrained_opt", dict(variant="quadratic", n=5, kappa=10, seed=1)),
    ("tv_unconstrained_opt", dict(variant="manufactured", n=5, kappa=10, seed=1)),
    ("tvqp_eq", {}),
    ("tvqp_eq", dict(variant="yang2025")),
    ("tvqp_eq", dict(variant="random", n=6, p=2, seed=3)),
    ("tvqp_ineq", {}),
    ("tvqp_ineq", dict(variant="yang2025")),
    ("tvqp_ineq", dict(variant="random", n=4, p=1, seed=0)),
    ("tvqp_ineq", dict(variant="random", n=3, p=0, seed=5, eps=1e-4)),
]
IDS = [k + ("-" + "-".join(f"{a}={b}" for a, b in kw.items()) if kw else "") for k, kw in CONFIGS]
TIMES = [0.0, 0.37, 2.9, 6.1]


def fd_jacobian(P, x, d, h=1e-6):
    J = np.zeros((x.shape[0], P.m, P.n_x))
    for j in range(P.n_x):
        e = np.zeros(P.n_x)
        e[j] = h
        J[:, :, j] = (P.E(x + e, d) - P.E(x - e, d)) / (2 * h)
    return J


def test_registered():
    probs = available()["problems"]
    for k in ("nonlinear_equations", "tv_unconstrained_opt", "tvqp_eq", "tvqp_ineq"):
        assert k in probs


@pytest.mark.parametrize("key,kw", CONFIGS, ids=IDS)
def test_exact_solution_zeroes_error(key, kw):
    P = make_problem(key, **kw)
    for t in TIMES:
        xs = P.x_star(t)
        assert xs.shape == (P.n_x,)
        assert np.abs(P.E(xs[None], P.data(t))).max() < 1e-10


@pytest.mark.parametrize("key,kw", CONFIGS, ids=IDS)
def test_dEdt_matches_finite_difference(key, kw):
    P = make_problem(key, **kw)
    rng = np.random.default_rng(1)
    dt = 1e-6
    for t in [0.37, 2.9, 6.1]:
        x = P.x0(rng, 3)
        fd = (P.E(x, P.data(t + dt)) - P.E(x, P.data(t - dt))) / (2 * dt)
        assert np.allclose(fd, P.dEdt(x, t), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("key,kw", CONFIGS, ids=IDS)
def test_jacobian_solveJ_JT(key, kw):
    P = make_problem(key, **kw)
    rng = np.random.default_rng(2)
    for t in TIMES:
        d = P.data(t)
        x = P.x0(rng, 3)
        Jfd = fd_jacobian(P, x, d)
        assert np.allclose(P.jacobian(x, d), Jfd, atol=1e-6)
        r = rng.standard_normal((3, P.m))
        dx = P.solve_J(x, d, r)
        ref = np.linalg.solve(Jfd, r[..., None])[..., 0]
        assert np.allclose(dx, ref, rtol=1e-5, atol=1e-6)
        v = rng.standard_normal((3, P.m))
        assert np.allclose(P.JT(x, d, v), np.einsum("bij,bi->bj", Jfd, v), atol=1e-6)


@pytest.mark.parametrize("key,kw", CONFIGS, ids=IDS)
def test_batch_consistency(key, kw):
    P = make_problem(key, **kw)
    rng = np.random.default_rng(3)
    t = 1.3
    d = P.data(t)
    x = P.x0(rng, 4)
    r = rng.standard_normal((4, P.m))
    full = (P.E(x, d), P.dEdt(x, t), P.solve_J(x, d, r), P.JT(x, d, r))
    for b in range(4):
        one = (P.E(x[b:b + 1], d), P.dEdt(x[b:b + 1], t), P.solve_J(x[b:b + 1], d, r[b:b + 1]),
               P.JT(x[b:b + 1], d, r[b:b + 1]))
        for a, c in zip(full, one):
            assert a.shape[0] == 4 and np.allclose(a[b], c[0], rtol=1e-12, atol=1e-12)
    # identical initial states in a batch give identical trajectories
    x0 = np.repeat(P.x0(rng, 1), 2, axis=0)
    s2 = simulate(P, make_model("znn"), make_discretization("euler"), tau=1e-2, T=1.0, B=2,
                  gamma=10.0, x0=x0)
    s1 = simulate(P, make_model("znn"), make_discretization("euler"), tau=1e-2, T=1.0, B=1,
                  gamma=10.0, x0=x0[:1])
    assert np.allclose(s2["residual"][:, 0], s2["residual"][:, 1])
    assert np.allclose(s2["residual"][:, 0], s1["residual"][:, 0])


@pytest.mark.parametrize("key,kw", CONFIGS, ids=IDS)
def test_znn_euler_smoke(key, kw):
    P = make_problem(key, **kw)
    tau = 1e-2 if key != "tvqp_ineq" else 2e-3
    s = simulate(P, make_model("znn"), make_discretization("euler"), tau=tau, T=4.0, B=3,
                 gamma=0.2 / tau, seed=4)
    assert not s["diverged"].any()
    r = s["residual"]
    assert np.all(np.isfinite(r))
    # converged from O(1) initial error to a small tracking floor
    assert np.all(np.median(r[s["t"] > 3.0], axis=0) < 1e-2 * r[0])
    if key == "tvqp_ineq" and kw.get("variant") == "yang2025":
        return   # dual unbounded near t = pi/2 (LICQ failure), see test_yang2025_licq_failure
    assert np.all(s["sol_err"][-1] < 5e-2)


# ---------------------------------------------------------------------------
# nonlinear equations: reproduction of published numbers and order
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("h,tau,ssce", [(0.1, 0.01, 1.399e-3), (0.15, 0.01, 9.385e-4),
                                        (0.2, 0.01, 7.056e-4), (0.1, 0.001, 1.414e-5),
                                        (0.15, 0.001, 9.422e-6), (0.2, 0.001, 7.070e-6)])
def test_guo2018_reproduces_table1_euler(h, tau, ssce):
    """Euler DTZNN x_{k+1} = x_k - J^{-1}(tau f_t + h f) == ZNN + Euler with
    gamma = h / tau; SSCE values from Table 1 of Frontiers Neurorobot. 2025
    (x0 = 0.2*ones(4)).  ASSUMPTION: horizon 10 s, SSCE = max ||f|| on the
    last quarter (our mssre)."""
    P = make_problem("nonlinear_equations", variant="guo2018", T=10.0)
    s = simulate(P, make_model("znn"), make_discretization("euler"), tau=tau, gamma=h / tau,
                 x0=0.2 * np.ones(4))
    assert abs(mssre(s)[0] / ssce - 1.0) < 0.01


def test_guo2018_domain_and_x0():
    P = make_problem("nonlinear_equations")
    x = P.x0(np.random.default_rng(0), 50)
    assert (x[:, 0] > 0).all()
    bad = np.array([[-1.0, 0.0, 0.0, 0.0]])
    assert np.isnan(P.E(bad, P.data(0.0))[0, 0])        # ln(x1) undefined -> flagged by simulate
    assert np.allclose(np.linalg.det(P.jacobian(x, P.data(1.0))), 1.0)


@pytest.mark.parametrize("key,kw", [("nonlinear_equations", {}), ("tvqp_eq", {})])
def test_euler_order_two_at_fixed_h(key, kw):
    P = make_problem(key, T=4.0, **kw)
    taus = [3e-2, 1e-2, 3e-3, 1e-3]
    errs = [float(mssre(simulate(P, make_model("znn"), make_discretization("euler"), tau=tau,
                                 gamma=0.3 / tau, x0=P.x_star(0.0)[None] + 0.1))[0])
            for tau in taus]
    slope = empirical_order(taus, errs)[0]
    assert abs(slope - 2.0) < 0.3, (slope, errs)


def test_unconstrained_opt_gradient_of_objective():
    """E is the gradient of f (checked by finite differences of f)."""
    for kw in [{}, dict(variant="quadratic", n=4), dict(variant="manufactured", n=4)]:
        P = make_problem("tv_unconstrained_opt", **kw)
        d = P.data(0.8)
        x = P.x0(np.random.default_rng(5), 2)
        g = np.zeros_like(x)
        for j in range(P.n_x):
            e = np.zeros(P.n_x)
            e[j] = 1e-6
            g[:, j] = (P.f(x + e, d) - P.f(x - e, d)) / 2e-6
        assert np.allclose(g, P.E(x, d), atol=1e-6)
        H = P.jacobian(x, d)
        assert np.allclose(H, np.swapaxes(H, 1, 2))
        assert (np.linalg.eigvalsh(H) > 0).all()     # strictly convex: unique minimiser


# ---------------------------------------------------------------------------
# QP with inequality constraints (perturbed Fischer-Burmeister)
# ---------------------------------------------------------------------------
def brute_force_qp(d):
    """Exact strictly convex QP solution by enumerating all active sets."""
    n, p, l = d["W"].shape[0], d["A"].shape[0], d["C"].shape[0]
    best = None
    for k in range(l + 1):
        for act in itertools.combinations(range(l), k):
            act = list(act)
            Ca = d["C"][act]
            M = np.block([[d["W"], d["A"].T, Ca.T],
                          [d["A"], np.zeros((p, p + k))],
                          [Ca, np.zeros((k, p + k))]])
            if np.linalg.matrix_rank(M) < M.shape[0]:
                continue
            sol = np.linalg.solve(M, np.concatenate([-d["q"], d["b"], d["d"][act]]))
            x, mu = sol[:n], sol[n + p:]
            if (mu >= -1e-9).all() and (d["C"] @ x <= d["d"] + 1e-9).all():
                best = x if best is None else best
                assert np.allclose(best, x, atol=1e-8)      # uniqueness
    return best


@pytest.mark.parametrize("variant", ["zhang2021", "yang2025", "random"])
def test_tvqp_ineq_qp_solution_exact_kkt(variant):
    P = make_problem("tvqp_ineq", variant=variant, n=3 if variant == "random" else None,
                     p=1 if variant == "random" else None)
    n, p = P.n, P.p
    n_active = 0
    for t in np.linspace(0, 6, 25):
        d = P.data(t)
        z = P.qp_solution(t)
        x, lam, mu = z[:n], z[n:n + p], z[n + p:]
        s = d["d"] - d["C"] @ x
        assert np.abs(d["W"] @ x + d["q"] + d["A"].T @ lam + d["C"].T @ mu).max() < 1e-10
        assert np.abs(d["A"] @ x - d["b"]).max() < 1e-10
        assert (mu >= -1e-10).all() and (s >= -1e-10).all() and np.abs(mu * s).max() < 1e-10
        assert np.allclose(x, brute_force_qp(d), atol=1e-8)
        n_active += int((mu > 1e-8).any())
        # E at the exact QP point: KKT rows vanish, FB rows in [-sqrt(eps), 0]
        E = P.E(z[None], d)[0]
        assert np.abs(E[:n + p]).max() < 1e-10
        assert (E[n + p:] <= 1e-12).all() and (E[n + p:] >= -np.sqrt(P.eps) - 1e-12).all()
    assert n_active > 0          # the test grid really exercises active constraints


def test_tvqp_ineq_smoothed_root_is_perturbed_complementarity():
    P = make_problem("tvqp_ineq", eps=1e-4)
    for t in np.linspace(0, 6, 13):
        z = P.smoothed_solution(t)
        d = P.data(t)
        x, mu = z[:P.n], z[P.n + P.p:]
        s = d["d"] - d["C"] @ x
        assert (s > 0).all() and (mu > 0).all()
        assert np.allclose(s * mu, P.eps / 2, rtol=1e-6)
    # pfb definition
    a, b = np.array([0.0, 2.0, 0.0]), np.array([3.0, 0.0, 0.0])
    assert np.allclose(pfb(a, b, 0.0), [0.0, 0.0, 0.0])          # eps = 0: plain FB, a*b = 0
    assert pfb(np.array([-1.0]), np.array([2.0]), 0.0)[0] < 0    # a < 0 violates the NCP
    assert np.isclose(pfb(np.array([0.1]), np.array([0.05]), 0.01)[0], 0.0)   # 0.1*0.05 = 0.01/2


def test_tvqp_ineq_smoothing_bias_scales_with_eps():
    """bias = ||z_eps* - z_QP*||: O(eps) when strictly complementary, worst case
    O(sqrt(eps)) near active-set switching instants."""
    ts = np.arange(0.0, 6.0, 0.01)
    worst, median = {}, {}
    for eps in (1e-2, 1e-4, 1e-6):
        P = make_problem("tvqp_ineq", eps=eps)
        b = np.array([P.smoothing_bias(t) for t in ts])
        worst[eps], median[eps] = b.max(), np.median(b)
        assert b.max() < 5 * np.sqrt(eps)
        assert P.stats["active_set_failures"] == 0 and P.stats["smoothed_failures"] == 0
    assert worst[1e-6] < worst[1e-4] < worst[1e-2]
    assert median[1e-6] < 0.1 * median[1e-4]       # ~linear in eps away from switches


def test_tvqp_ineq_smoothing_parameter_vs_discretization():
    """Large eps: smooth reference -> clean O(tau^2) ZNN-Euler order at fixed h.
    Small eps: active-set switching happens on a time scale ~sqrt(eps), which
    Euler cannot resolve for tau >> sqrt(eps): the error is no longer O(tau^2)."""
    taus = [3e-2, 1e-2, 3e-3, 1e-3]

    def order(eps):
        P = make_problem("tvqp_ineq", eps=eps, T=6.0)
        errs = [float(mssre(simulate(P, make_model("znn"), make_discretization("euler"),
                                     tau=tau, gamma=0.3 / tau, seed=0))[0]) for tau in taus]
        return empirical_order(taus, errs)[0], errs

    s_big, e_big = order(1e-2)
    s_small, e_small = order(1e-6)
    assert abs(s_big - 2.0) < 0.4, (s_big, e_big)
    assert s_small < 1.6, (s_small, e_small)
    assert e_small[-1] > 10 * e_big[-1]


def test_tvqp_ineq_reference_modes_and_cache():
    Pq = make_problem("tvqp_ineq", reference="qp")
    Ps = make_problem("tvqp_ineq")
    t = 2.345
    assert np.allclose(Pq.x_star(t), Pq.qp_solution(t))
    assert np.allclose(Ps.x_star(t), Ps.smoothed_solution(t))
    n0 = Ps.stats["solves"]
    for _ in range(3):
        Ps.x_star(t)
    assert Ps.stats["solves"] == n0
    assert Ps.smoothing_bias(t) < 5 * np.sqrt(Ps.eps)


def test_yang2025_licq_failure():
    """Degeneracy of the data PRINTED in Sec. 4.1 of Yang et al. (2025): at
    t = pi/2 the equality row A = [0, -1], b = -1 coincides with the box face
    x2 <= 1, so LICQ fails and the multipliers are unbounded near that instant
    (primal solution stays well defined).  The paper's own figures were not
    computed with these data (see test_yang2025_figures_use_benchmark18_data)."""
    P = make_problem("tvqp_ineq", variant="yang2025")
    d = P.data(np.pi / 2)
    assert np.allclose(d["A"], [[0.0, -1.0]], atol=1e-12) and np.isclose(d["b"][0], -1.0)
    z_far = P.qp_solution(1.2)
    z_near = P.qp_solution(np.pi / 2 - 1e-3)
    assert np.abs(z_near[2:]).max() > 50 * max(1.0, np.abs(z_far[2:]).max())
    assert np.abs(z_near[:2]).max() <= 1.0 + 1e-9


def test_invalid_arguments():
    with pytest.raises(ValueError):
        make_problem("nonlinear_equations", variant="guo2018", n=3)
    with pytest.raises(ValueError):
        make_problem("tv_unconstrained_opt", variant="nope")
    with pytest.raises(ValueError):
        make_problem("tvqp_ineq", eps=0.0)
    with pytest.raises(ValueError):
        make_problem("tvqp_eq", variant="random", n=3, p=3)


# ---------------------------------------------------------------------------
# transcription checks against the printed (scalar) forms of the sources.
# These are independent re-typings of the papers' equations, so a typo in the
# matrix data of optimization.py cannot pass the self-consistency tests above
# unnoticed.
# ---------------------------------------------------------------------------
def _pfb_ref(a, b, eps):
    return a + b - np.sqrt(a ** 2 + b ** 2 + eps)


def test_transcription_guo2018_eq33():
    """Eq. (33) of Ma & Huang, Front. Neurorobot. 2025, written out by hand."""
    P = make_problem("nonlinear_equations")
    rng = np.random.default_rng(11)
    for t in (0.0, 0.8, 4.4):
        x = P.x0(rng, 3)
        x1, x2, x3, x4 = x.T
        ref = np.stack([np.log(x1) - 1.0 / (t + 1.0),
                        x1 * x2 - np.sin(t) * np.exp(1.0 / (t + 1.0)),
                        x1 ** 2 - np.sin(t) * x2 + x3 - 2.0,
                        x1 ** 2 - x2 ** 2 + x3 + x4 - t], axis=1)
        assert np.allclose(P.E(x, P.data(t)), ref, atol=1e-14)


def _zhang2021_kkt_ref(z, t, with_eq, with_box, eps=None):
    """Benchmark (18) of Zhang et al. CAAI-TIT 2021 in its printed scalar form:
    min (sin t/8 + 1/2) x1^2 + (cos t/8 + 1/2) x2^2 + cos t x1 x2 / 2
        + sin 3t x1 + cos 3t x2
    s.t. sin 4t x1 + cos 4t x2 = cos 2t  ('+' restored),  -1.3 <= x1, x2 <= 1.3."""
    x1, x2 = z[:, 0], z[:, 1]
    g1 = 2 * (np.sin(t) / 8 + 0.5) * x1 + np.cos(t) / 2 * x2 + np.sin(3 * t)
    g2 = 2 * (np.cos(t) / 8 + 0.5) * x2 + np.cos(t) / 2 * x1 + np.cos(3 * t)
    rows = []
    k = 2
    if with_eq:
        lam = z[:, k]
        g1, g2 = g1 + np.sin(4 * t) * lam, g2 + np.cos(4 * t) * lam
        rows.append(np.sin(4 * t) * x1 + np.cos(4 * t) * x2 - np.cos(2 * t))
        k += 1
    if with_box:     # C = [I; -I], d = 1.3: rows x1 <= 1.3, x2 <= 1.3, -x1 <= 1.3, -x2 <= 1.3
        mu = z[:, k:k + 4]
        g1, g2 = g1 + mu[:, 0] - mu[:, 2], g2 + mu[:, 1] - mu[:, 3]
        slack = np.stack([1.3 - x1, 1.3 - x2, 1.3 + x1, 1.3 + x2], axis=1)
        rows += list(_pfb_ref(slack, mu, eps).T)
    return np.stack([g1, g2] + rows, axis=1)


def test_transcription_zhang2021_benchmark18():
    rng = np.random.default_rng(12)
    cases = [("tv_unconstrained_opt", {}, False, False),
             ("tvqp_eq", {}, True, False),
             ("tvqp_ineq", dict(eps=1e-3), True, True)]
    for key, kw, eq, box in cases:
        P = make_problem(key, **kw)
        for t in (0.0, 0.9, 3.7):
            z = rng.uniform(-1.5, 1.5, size=(4, P.n_x))
            ref = _zhang2021_kkt_ref(z, t, eq, box, getattr(P, "eps", None))
            assert np.allclose(P.E(z, P.data(t)), ref, atol=1e-13), key


def test_transcription_yang2025_sec41():
    """Sec. 4.1 of Yang et al. (TST 2025/2026, arXiv:2412.20477): Omega =
    [[cos t/4 + 1, sin t/4], [sin t/4, sin t/4 + 1]], p = [cos 2t, sin 2t],
    A = [sin 2t, cos 2t], b = sin 3t, C = [I, -I]^T, d = [1, 1, 1, 1], eps = 1e-6."""
    P = make_problem("tvqp_ineq", variant="yang2025")
    assert P.eps == 1e-6 and P.T == 20.0
    rng = np.random.default_rng(13)
    for t in (0.0, 1.1, 7.3):
        Om = np.array([[np.cos(t) / 4 + 1, np.sin(t) / 4], [np.sin(t) / 4, np.sin(t) / 4 + 1]])
        p = np.array([np.cos(2 * t), np.sin(2 * t)])
        A = np.array([np.sin(2 * t), np.cos(2 * t)])
        C = np.vstack([np.eye(2), -np.eye(2)])
        z = rng.uniform(-1.5, 1.5, size=(3, 7))
        y, lam, mu = z[:, :2], z[:, 2], z[:, 3:]
        ref = np.concatenate([y @ Om.T + p + lam[:, None] * A + mu @ C,
                              (y @ A - np.sin(3 * t))[:, None],
                              _pfb_ref(1.0 - y @ C.T, mu, 1e-6)], axis=1)
        assert np.allclose(P.E(z, P.data(t)), ref, atol=1e-13)
        Peq = make_problem("tvqp_eq", variant="yang2025")
        assert np.allclose(Peq.E(z[:, :3], Peq.data(t)), ref[:, :3] - np.c_[mu @ C, 0 * lam],
                           atol=1e-13)


def test_tvqp_ineq_jacobian_is_yang_P_matrix():
    """J = P(t) of Yang et al. Eq. (7), built here from its printed blocks
    [[Omega, A', C'], [A, 0, 0], [(Lambda1 - I) C, 0, I - Lambda2]] with
    Lambda1 = diag(n/m), Lambda2 = diag(lambda2/m), n = d - C y,
    m = sqrt(n^2 + lambda2^2 + eps); and dE/dt = Q z + rho with C' = d' = 0."""
    P = make_problem("tvqp_ineq", variant="yang2025", eps=1e-3)
    z = np.random.default_rng(14).uniform(-1, 1, size=(1, 7))
    t = 2.2
    d = P.data(t)
    y, lam2 = z[0, :2], z[0, 3:]
    nn = d["d"] - d["C"] @ y
    mm = np.sqrt(nn ** 2 + lam2 ** 2 + P.eps)
    L1, L2 = np.diag(nn / mm), np.diag(lam2 / mm)
    Pm = np.block([[d["W"], d["A"].T, d["C"].T],
                   [d["A"], np.zeros((1, 1)), np.zeros((1, 4))],
                   [(L1 - np.eye(4)) @ d["C"], np.zeros((4, 1)), np.eye(4) - L2]])
    assert np.allclose(P.jacobian(z, d)[0], Pm, atol=1e-14)
    _, (Wd, qd, Ad, bd) = P.qp(t)
    Qm = np.block([[Wd, Ad.T, np.zeros((2, 4))], [Ad, np.zeros((1, 5))], [np.zeros((4, 7))]])
    rho = np.concatenate([qd, -bd, np.zeros(4)])
    assert np.allclose(P.dEdt(z, t)[0], Qm @ z[0] + rho, atol=1e-14)


def test_unconstrained_opt_declares_gradient_field():
    """E = grad f with an SPD Hessian -> 'gradient_field' (contract of
    pc_gradient); KKT problems must not declare it."""
    for kw in [{}, dict(variant="quadratic"), dict(variant="manufactured")]:
        assert "gradient_field" in make_problem("tv_unconstrained_opt", **kw).capabilities()
    for key in ("tvqp_eq", "tvqp_ineq", "nonlinear_equations"):
        assert "gradient_field" not in make_problem(key).capabilities()


def test_yang2025_figures_use_benchmark18_data():
    """Fig. 1 of Yang et al. (TST 2025, arXiv:2412.20477, p. 6) shows y1, y2
    saturating at +-1.3, impossible under the printed Sec. 4.1 box d = 1.  The
    box-saturation intervals read off a 300-dpi render of that figure (y1 = -1.3
    on [4.75, 4.95], [11.04, 11.24], [17.33, 17.48]; y2 = -1.3 on [1.73, 1.88],
    [4.46, 4.60], [8.02, 8.12], [10.74, 10.89], [14.31, 14.41], [17.03, 17.13])
    are exactly where the 'zhang2021' (CAAI benchmark (18)) QP solution has the
    bound active.  The figure-read mid-points are asserted here."""
    Pz = make_problem("tvqp_ineq", variant="zhang2021")
    for i, ts in [(0, (4.85, 11.14, 17.40)), (1, (1.80, 4.53, 8.07, 10.82, 14.36, 17.08))]:
        for t in ts:
            z = Pz.qp_solution(t)
            assert abs(z[i] + 1.3) < 1e-9, (i, t, z[:2])
            assert z[2 + 3 + i] > 0          # multiplier of the face -x_i <= 1.3 is active
    Py = make_problem("tvqp_ineq", variant="yang2025")
    assert max(np.abs(Py.qp_solution(t)[:2]).max() for t in np.arange(0, 20, 0.05)) <= 1 + 1e-9


# ---------------------------------------------------------------------------
# integration (smoke-matrix) regression
# ---------------------------------------------------------------------------
def test_singular_jacobian_falls_back_to_min_norm_not_error(monkeypatch):
    """An exactly singular J (reached by diverging trajectories of a batch) must
    not raise from the model-facing solve_J: the batch would be aborted and the
    runner would record 'error' for every trajectory.  The fallback is the
    minimum-norm least-squares step (Problem.solve_J contract); the reference
    Newton solver still sees the exception through _bsolve."""
    from znnbench.problems import optimization as opt
    J = np.array([[[2.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 0.0]]])
    r = np.array([[2.0, 3.0], [4.0, 5.0]])
    with pytest.raises(np.linalg.LinAlgError):
        opt._bsolve(J, r)
    dx = opt._bsolve_model(J, r)
    assert np.allclose(dx[0], [1.0, 3.0]) and np.allclose(dx[1], [4.0, 0.0])
    P = make_problem("tvqp_ineq")
    x = P.x0(np.random.default_rng(0), 2)
    d = P.data(0.2)
    Jsing = P._J(x, d, P.eps)
    Jsing[1, -1, :] = 0.0                       # make the second trajectory's J singular
    monkeypatch.setattr(P, "_J", lambda xx, dd, eps: Jsing)
    out = P.solve_J(x, d, np.ones((2, P.m)))
    assert np.allclose(out[0], np.linalg.solve(Jsing[0], np.ones(P.m)))
    assert np.allclose(out[1], np.linalg.pinv(Jsing[1], rcond=1e-12) @ np.ones(P.m))
