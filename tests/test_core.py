import numpy as np
import pytest

from znnbench import make_discretization, make_model, make_problem, simulate
from znnbench.metrics import chattering, empirical_order, mssre

TAUS = [3e-2, 1e-2, 3e-3, 1e-3]


def order_at_fixed_h(disc_key, h=0.3, problem=None, taus=TAUS, T=4.0):
    P = problem or make_problem("linear_system", n=4, kappa=10, seed=1, T=T)
    errs = []
    for tau in taus:
        s = simulate(P, make_model("znn"), make_discretization(disc_key), tau=tau, B=1,
                     gamma=h / tau, seed=0)
        errs.append(float(mssre(s)[0]))
    return empirical_order(taus, errs)[0], errs


def test_linear_system_consistency():
    P = make_problem("linear_system", n=5, kappa=50, seed=3)
    for t in [0.0, 0.7, 3.1]:
        xs = P.x_star(t)[None]
        assert np.abs(P.E(xs, P.data(t))).max() < 1e-10
        x = np.random.default_rng(0).standard_normal((2, 5))
        dt = 1e-6
        fd = (P.E(x, P.data(t + dt)) - P.E(x, P.data(t - dt))) / (2 * dt)
        assert np.allclose(fd, P.dEdt(x, t), atol=1e-6)


@pytest.mark.parametrize("key,order", [("euler", 2), ("taylor_zhang", 3)])
def test_lookahead_order(key, order):
    slope, errs = order_at_fixed_h(key)
    assert abs(slope - order) < 0.3, (slope, errs)


def test_fixed_gamma_euler_is_first_order():
    P = make_problem("linear_system", n=4, kappa=10, seed=1, T=4.0)
    errs = [float(mssre(simulate(P, make_model("znn"), make_discretization("euler"), tau=t,
                                 gamma=10.0))[0]) for t in TAUS]
    assert abs(empirical_order(TAUS, errs)[0] - 1.0) < 0.3


def test_gnn_lag_floor():
    P = make_problem("linear_system", n=4, kappa=10, seed=1, T=4.0)
    # GNN's effective step is gamma*tau*sigma_max(A)^2 (= 100 here), so keep it < 2
    e1 = mssre(simulate(P, make_model("gnn"), make_discretization("euler"), tau=1e-4, gamma=1))[0]
    e10 = mssre(simulate(P, make_model("gnn"), make_discretization("euler"), tau=1e-4, gamma=10))[0]
    assert e10 < e1 / 3        # O(1/gamma) lag floor shrinks with gain


def test_implicit_removes_sign_chattering():
    P = make_problem("linear_system", n=4, kappa=10, seed=1, T=4.0)
    kw = dict(tau=1e-2, gamma=30, B=1, seed=0, x0=P.x0(np.random.default_rng(0), 1))
    ex = simulate(P, make_model("znn", activation="sign_bi_power"), make_discretization("euler"), **kw)
    im = simulate(P, make_model("znn", activation="sign_bi_power"),
                  make_discretization("implicit_euler_err"), **kw)
    assert chattering(im)[0] < chattering(ex)[0]
    assert mssre(im)[0] < mssre(ex)[0]


def test_divergence_flag():
    P = make_problem("linear_system", n=4, kappa=10, seed=1, T=2.0)
    s = simulate(P, make_model("znn"), make_discretization("euler"), tau=1e-2, gamma=300)  # h = 3
    assert s["diverged"].all() and np.isinf(mssre(s)).all()
