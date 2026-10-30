"""Integration tests (added by the integrator): cross-area behaviour found by
experiments/smoke_matrix.py.  Small and fast (< 10 s, < 200 MB)."""
import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pytest

from znnbench import available, make_discretization, make_model, make_problem, simulate

ROOT = Path(__file__).resolve().parents[1]


def _smoke_module():
    spec = importlib.util.spec_from_file_location("smoke_matrix", ROOT / "experiments" / "smoke_matrix.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_design_formula_models_freeze_time_varying_activation():
    """A time-varying activation (strict_predefined, SPTC-NT-ZNN) is frozen at
    ctx.t by the design-formula models instead of raising at the first step.
    Continuous time: E(t) = E0 ((t_c - t)/t_c)^gamma, so with RK4 the residual
    at t = t_c/2 is 2^-gamma times the initial one."""
    from znnbench.models.variants import SPTCNTZNN
    P = make_problem("linear_system", n=3, kappa=5, seed=2, T=0.5)
    tc, g = 1.0, 3.0
    m = make_model("znn", activation="strict_predefined", t_c=tc)
    s = simulate(P, m, make_discretization("rk4"), tau=1e-3, gamma=g, B=2, seed=1)
    assert np.allclose(s["residual"][-1] / s["residual"][0], 0.5 ** g, rtol=1e-6)
    # identical to the model-level SPTC-NT-ZNN before t_c (Euler, causal)
    a = simulate(P, m, make_discretization("euler"), tau=1e-2, gamma=g, seed=1)
    b = simulate(P, SPTCNTZNN(t_c=tc), make_discretization("euler"), tau=1e-2, gamma=g, seed=1)
    assert np.allclose(a["residual"], b["residual"], rtol=1e-12, atol=1e-14)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)     # peb -> resolvent fallback notice
        for key, disc in [("gnn", "euler"), ("znn", "implicit_euler_err"),
                          ("znn_consistent", "peb_euler"), ("vpznn", "euler")]:
            s = simulate(P, make_model(key, activation="strict_predefined"),
                         make_discretization(disc), tau=1e-2, gamma=1.0, seed=1)
            assert np.isfinite(s["residual"]).all(), key
    # time-invariant activations: at(t) is the identity, behaviour unchanged
    act = make_model("znn", activation="sign_bi_power").act
    assert act.at(0.3) is act


def test_mini_smoke_matrix_has_no_errors():
    """Every registered problem x every registered model on Euler and the
    implicit scheme (T = 0.1 s, B = 2): no cell may end in 'error'/'timeout'
    (divergence and recorded skips are allowed)."""
    sm = _smoke_module()
    av = available()
    bad = []
    for p in av["problems"]:
        for m in av["models"]:
            for d in ("euler", "implicit_euler_err"):
                r = sm.run_cell(p, m, d, T=0.1, tau=1e-2, gamma=10.0, B=2, seed=0, max_wall=30.0,
                                default_x0=False)
                if r["status"] in ("error", "timeout"):
                    bad.append((p, m, d, r["message"]))
    assert not bad, bad


@pytest.mark.parametrize("model", ["znn_explicit_inverse", "znn_explicit_inverse_right"])
def test_smoke_matrix_newton_schulz_start(model):
    """The smoke matrix starts the locally convergent explicit-inverse models from
    the Newton-Schulz inverse (CORE_CHANGE_REQUESTS structured_baselines item 5):
    they then converge on their matching problem."""
    sm = _smoke_module()
    prob = "matrix_inversion" if model.endswith("inverse") else "matrix_inversion_right"
    r = sm.run_cell(prob, model, "euler", T=1.0, tau=1e-2, gamma=10.0, B=1, seed=0, max_wall=30.0,
                    default_x0=False)
    assert r["status"] == "ok" and r["x0"] == "x0=Newton-Schulz"
    assert r["resT"] < 1e-2 * r["res0"]


def test_zoo_catalogue_covers_every_registered_key():
    """docs/ZOO.md lists every registered key (problems, models, discretizations,
    noises, activations, gain schedules, wrapper classes)."""
    import re

    from znnbench import activations as A
    from znnbench.models.gains import SCHEDULES
    text = (ROOT / "docs" / "ZOO.md").read_text()
    listed = set(re.findall(r"`([A-Za-z0-9_]+)`", text))
    av = available()
    keys = {k for v in av.values() for k in v} | set(A.available()) | set(SCHEDULES)
    keys |= {"QuantizedDiscretizer", "JitteredProblem", "DelayedDataProblem",
             "EstimatedDerivativeProblem", "CountingProblem"}
    missing = sorted(keys - listed)
    assert not missing, f"docs/ZOO.md is missing {missing}"
