"""Tests for the core changes landed after the zoo integration (RNG streams,
model x0 hook, bind_env, exact_data, online statistics)."""
import numpy as np

from znnbench import make_discretization, make_model, make_noise, make_problem, simulate
from znnbench.core import Discretizer, Problem
from znnbench.metrics import summarize
from znnbench.simulate import initial_state


def P():
    return make_problem("linear_system", n=4, kappa=10, seed=1, T=1.0)


def test_noise_is_batch_composition_invariant():
    """Trajectory b sees the same Gaussian noise whether it runs alone or in a
    bigger batch (per-trajectory streams), so batched gains equal separate runs."""
    p = P()
    x0 = initial_state(p, make_model("znn"), 0, 2)
    kw = dict(tau=1e-2, seed=0, noise=make_noise("gaussian", std=0.5))
    single = simulate(p, make_model("znn"), make_discretization("euler"), B=2, gamma=20.0, x0=x0, **kw)
    both = simulate(p, make_model("znn"), make_discretization("euler"), B=4,
                    gamma=np.repeat([20.0, 40.0], 2), x0=np.tile(x0, (2, 1)),
                    traj_ids=np.tile(np.arange(2), 2), **kw)
    assert np.array_equal(single["residual"], both["residual"][:, :2])


def test_x0_hook_and_initial_state_agree():
    p = make_problem("matrix_inversion", n=3, seed=0)
    m = make_model("znn_explicit_inverse")
    x = initial_state(p, m, 5, 2)
    s = simulate(p, m, make_discretization("euler"), tau=1e-2, T=0.05, B=2, gamma=10.0, seed=5)
    assert np.array_equal(s["x_init"], x)
    A0 = p.data(0.0)["A"]
    X0 = x[0].reshape(3, 3)
    assert np.abs(np.linalg.eigvals(np.eye(3) - A0 @ X0)).max() < 1.0   # Newton-Schulz safe start


def test_drift_and_final_state():
    p = P()
    s = simulate(p, make_model("znn"), make_discretization("euler"), tau=1e-2, B=3, gamma=30.0)
    out = summarize(s)
    assert s["x_final"].shape == (3, 4)
    assert np.allclose(out["drift"], np.linalg.norm(s["x_final"] - s["x_init"], axis=1))


class _Flip(Problem):
    """Time-invariant scalar problem E = x - 1 used to build a period-2 cycle."""
    name, n_x, m, T = "flip", 1, 1, 1.0

    def data(self, t):
        return {"t": np.float64(t)}

    def E(self, x, d):
        return x - 1.0

    def dEdt(self, x, t):
        return np.zeros_like(x)

    def solve_J(self, x, d, r):
        return r

    def JT(self, x, d, v):
        return v

    def x_star(self, t):
        return np.ones(1)

    def x0(self, rng, B):
        return np.full((B, 1), 1.5)


def test_chatter_vec_detects_symmetric_cycle():
    """Linear ZNN with h = 2 maps e -> -e: ||E|| is constant (the norm-based
    index reads 0) but the vector index reads 2."""
    s = simulate(_Flip(), make_model("znn"), make_discretization("euler"), tau=1e-2, gamma=200.0)
    out = summarize(s)
    assert out["chatter"][0] < 1e-12
    assert abs(out["chatter_vec"][0] - 2.0) < 1e-9
    assert out["rate_alt"][0] > 0.99


def test_bind_env_receives_measured_sample():
    seen = {}

    class Probe(Discretizer):
        name, causal = "probe", True

        def bind_env(self, problem, measured, stage_data):
            seen["problem"], seen["measured"] = problem, measured

        def step(self, k, t, tau, x, z, f):
            seen.setdefault("t", []).append(float(seen["measured"]()["t"]))
            xd, zd = f(t, x, z, k, False)
            return x + tau * xd, z + tau * zd

    p = P()
    simulate(p, make_model("znn"), Probe(), tau=0.1, T=0.3, gamma=3.0)
    assert seen["problem"] is p and np.allclose(seen["t"], [0.0, 0.1, 0.2])


def test_smooth_tracking_has_small_velocity_alternation():
    s = simulate(P(), make_model("znn"), make_discretization("taylor_zhang"), tau=1e-3,
                 T=2.0, gamma=100.0)
    assert summarize(s)["rate_alt"][0] < 0.05
