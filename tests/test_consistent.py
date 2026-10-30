"""Tests for znnbench/discretizations/consistent.py (spec: docs/spec/consistent.md)."""
import math
import warnings

import numpy as np
import pytest
from scipy import special
from scipy.integrate import quad, solve_ivp

from znnbench import activations as acts
from znnbench import make_discretization, make_model, make_problem, simulate
from znnbench.discretizations import consistent as cd
from znnbench.discretizations.consistent import (ConsistentZNN, PowerSum, PredefinedTimeJR,
                                                 exact_map, peb_map, resolvent_map,
                                                 semi_implicit_map)
from znnbench.metrics import chattering, empirical_order, mssre
from znnbench.models.znn import DesignFormulaZNN

TAUS = [3e-2, 1e-2, 3e-3, 1e-3]
SIGN = PowerSum([(1.0, 0.0)])
SBP = acts.get("sign_bi_power", r=0.5)
KEYS = ["implicit_taylor_zhang", "implicit_taylor_zhang_naive", "peb_euler", "peb_taylor_zhang",
        "exact_flow_euler", "exact_flow_taylor_zhang", "semi_implicit_euler",
        "semi_implicit_taylor_zhang", "consistent"]


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        yield


def flow(act, e0, h):
    """Reference: integrate |e|' = -Phi(|e|) to rtol 1e-13, stopping at 0."""
    ev = lambda t, y: y[0] - 1e-300  # noqa: E731
    ev.terminal, ev.direction = True, -1
    sol = solve_ivp(lambda t, y: -np.abs(act(np.maximum(y, 0.0))), (0, h), [abs(e0)],
                    method="DOP853", rtol=1e-13, atol=1e-16, events=ev)
    return math.copysign(0.0 if sol.status == 1 else max(sol.y[0, -1], 0.0), e0)


def steps_to_zero(emap, act, e0, h, kmax=100000):
    e, n = np.array([[float(e0)]]), 0
    while e[0, 0] != 0.0 and n < kmax:
        e, n = emap(act, e, h), n + 1
    return n


# ---------------------------------------------------------------------------
# structure: registry, causality, coefficients
# ---------------------------------------------------------------------------
def test_registry_and_causality():
    """Every key builds; each step evaluates f exactly once, at (t_k, x_k) with k."""
    from znnbench.core import Context
    P = make_problem("linear_system", n=3, kappa=5, seed=0, T=0.1)
    model, tau = ConsistentZNN("sign_bi_power"), 1e-2
    assert make_model("znn_consistent").__class__ is ConsistentZNN
    for key in KEYS:
        d = make_discretization(key)
        assert d.causal and d.implicit and d.expected_order in (2, 3)
        calls = []

        def f(t, x, z, k, implicit):
            calls.append((t, k, x.copy()))
            dd = P.data(t)
            ctx = Context(problem=P, t=t, tau=tau, k=k, x=x, z=z, d=dd, E=P.E(x, dd),
                          dEdt=P.dEdt(x, t), gamma=np.full((x.shape[0], 1), 10.0),
                          implicit=implicit)
            return model.rhs(ctx)

        d.reset()
        x, z = P.x0(np.random.default_rng(0), 2), np.zeros((2, 0))
        xs = []
        for k in range(6):
            xs.append(x)
            x, z = d.step(k, k * tau, tau, x, z, f)
        assert [c[1] for c in calls] == list(range(6))
        assert np.allclose([c[0] for c in calls], tau * np.arange(6))
        assert all(np.array_equal(c[2], xk) for c, xk in zip(calls, xs))


def test_taylor_zhang_coefficients_consistent_and_zero_stable():
    c = np.array(cd.TAYLOR_ZHANG)            # c_{-1}, c_0, c_1, c_2
    i = np.arange(-1, len(c) - 1)
    assert abs(c.sum()) < 1e-15                # order 0
    assert abs((c * -i).sum() - 1.0) < 1e-15   # order 1:  tau x_dot
    assert abs((c * i ** 2).sum()) < 1e-15     # order 2 (truncation O(tau^3))
    roots = np.roots(c)
    assert np.isclose(np.max(np.abs(roots)), 1.0) and np.sum(np.isclose(np.abs(roots), 1.0)) == 1
    # no drift from the reference formula in multistep.py (Zhang et al. 2015: 2x+ - 3x + 2x- - x--)
    tz = make_discretization("taylor_zhang")
    assert np.allclose(c, [tz.c_next, *tz.c_past]) and np.allclose(2 * c, [2, -3, 2, -1])
    # naive TZ + dead zone: E_{k+1} = (E_k - 2E_{k-1} + E_{k-2})/2  ->  (z-1/2)(z^2+1): marginal
    assert np.allclose(sorted(np.abs(np.roots([1, -0.5, 1, -0.5]))), [0.5, 1.0, 1.0])


# ---------------------------------------------------------------------------
# error maps
# ---------------------------------------------------------------------------
EXACT_CASES = [SBP, PowerSum([(2.0, 0.3), (0.7, 1.8)]), PowerSum([(1.0, 0.5), (2.0, 1.0)]),
               PowerSum([(1.5, 1.0), (0.5, 3.0)]), PowerSum([(1.0, 2.0)]), PowerSum([(1.0, 0.4)]),
               acts.get("linear")]


@pytest.mark.parametrize("act", EXACT_CASES, ids=lambda a: str(cd.power_terms(a)))
def test_exact_map_is_exact_flow(act):
    for e0 in [3.0, 0.7, -0.05]:
        for h in [0.01, 0.1]:
            assert abs(exact_map(act, np.array([[e0]]), h)[0, 0] - flow(act, e0, h)) < 1e-12


@pytest.mark.parametrize("kappa", ["arctan", "exp", "rational"])
@pytest.mark.parametrize("rho2", [0.0, 0.5])
def test_jr_exact_discretization(kappa, rho2):
    """JR 2020 Eq. (7) equals the flow (Thm 6) and reaches 0 within ceil(rho1/h) steps (Cor. 7)."""
    act = PredefinedTimeJR(kappa, a=1.3, mu=1.0, rho1=0.8, rho2=rho2)
    for e0 in [2.0, -0.3]:
        assert abs(act.exact_step(np.array([[e0]]), 0.05)[0, 0] - flow(act, e0, 0.05)) < 1e-12
    for e0 in [1e-3, 1.0, 1e3, 1e8]:
        for h in [0.05, 0.3]:
            n = steps_to_zero(exact_map, act, e0, h)
            T = float(act.settling_time(e0))
            assert n == max(math.ceil(T / h - 1e-12), 1) and n <= math.ceil(0.8 / h)


def test_jr_example_8():
    """JR 2020 Example 8 / Figs. 1-2: kappa = (2/pi) arctan(r), rho1 = 1, rho2 = 0.5, x0 = 10.
    Eq. (7) reproduces the closed-form solution (2) at t = kh (Thm 6, h = 0.02) and is 0 for
    k >= ceil(rho1/h) (Cor. 7); explicit Euler at the paper's h = 0.0651318636497 sits on the
    period-2 orbit 10 -> -10 (the onset of the instability shown in Fig. 2)."""
    act = PredefinedTimeJR("arctan", a=1.0, mu=1.0, rho1=1.0, rho2=0.5)
    x0, h = 10.0, 0.02
    kap = lambda r: 2 / np.pi * np.arctan(r)            # noqa: E731  (independent of the class)
    sol = lambda t: math.tan(np.pi / 2 * max(kap(x0) ** 0.5 - t, 0) ** 2)  # Eq. (2), x0 > 0
    x = np.array([[x0]])
    for k in range(1, 61):
        x = act.exact_step(x, h)
        assert abs(x[0, 0] - sol(k * h)) <= 1e-12 * max(1.0, sol(k * h))
        if k >= math.ceil(1.0 / h):
            assert x[0, 0] == 0.0
    he = 0.0651318636497
    assert abs(x0 - he * act(np.array([x0]))[0] + x0) < 1e-9


def test_bipower_settling_time_formula():
    """T_max of the exact map equals Aldana-Lopez et al. 2019 Thm 1 (k = 1) and quadrature;
    the exact map needs ceil(T(e0)/h) steps, bounded by ceil(T_max/h) for every e0."""
    a1, p1, a2, p2 = 0.5, 0.5, 0.5, 2.0                          # sign-bi-power, r = 1/2
    mp, mq = (1 - p1) / (p2 - p1), (p2 - 1) / (p2 - p1)
    tmax_al = special.gamma(mp) * special.gamma(mq) / (a1 * (p2 - p1)) * (a1 / a2) ** mp
    tmax_q = quad(lambda z: 1 / (a1 * z ** p1 + a2 * z ** p2), 0, np.inf)[0]
    assert abs(tmax_al - tmax_q) < 1e-8
    for e0 in [1e-6, 0.3, 5.0]:
        T = quad(lambda z: 1 / (a1 * z ** p1 + a2 * z ** p2), 0, e0, limit=200)[0]
        for h in [0.05, 0.3]:
            assert steps_to_zero(exact_map, SBP, e0, h) == math.ceil(T / h)
    for e0 in [1e6, 1e10]:
        assert steps_to_zero(exact_map, SBP, e0, 0.05) <= math.ceil(tmax_al / 0.05)


def test_peb_scalar_transcriptions():
    """PEB 2019 Sec. 1.1 (x' = -2 sqrt|x| sgn x) and Sec. 1.2 (x' = -|x| x) closed forms."""
    act = PowerSum([(2.0, 0.5)])
    for x, h in [(0.9, 0.1), (-0.004, 0.1), (5.0, 0.3)]:
        expect = (math.sqrt(abs(x)) - h) ** 2 * math.copysign(1, x) if abs(x) > h * h else 0.0
        assert abs(peb_map(act, np.array([[x]]), h)[0, 0] - expect) < 1e-14
    act = PowerSum([(1.0, 2.0)])
    for x, h in [(3.0, 0.1), (-50.0, 1.0)]:
        assert abs(peb_map(act, np.array([[x]]), h)[0, 0] - x / (1 + h * abs(x))) < 1e-14


def test_peb_split_discrete_fixed_time_bound():
    """Lie-split PEB map for sign-bi-power reaches exactly 0 within
    ceil(1/(a2(p2-1)h)) + ceil(1/(a1(1-p1)h)) steps for every e0 (discrete fixed time)."""
    for r in [0.3, 0.5, 0.8]:
        a1, p1, a2, p2 = 0.5, r, 0.5, 1 / r
        act = acts.get("sign_bi_power", r=r)
        for h in [0.05, 0.3]:
            bound = math.ceil(1 / (a2 * (p2 - 1) * h)) + math.ceil(1 / (a1 * (1 - p1) * h))
            for e0 in [1e-8, 0.5, 10.0, 1e6, 1e12]:
                assert steps_to_zero(peb_map, act, e0, h) <= bound


def test_resolvent_is_not_finite_step_for_continuous_activation():
    """PEB Sec. 1.1: backward Euler of a continuous finite-time field only converges
    asymptotically (here super-linearly), never exactly; the PEB / exact maps hit 0."""
    e, traj = np.array([[0.3]]), []
    for _ in range(13):
        e = resolvent_map(SBP, e, 0.3)
        traj.append(e[0, 0])
    assert all(v > 0 for v in traj) and traj[-1] < 1e-30
    assert np.all(np.diff(np.log(traj[-4:])) < 1.5 * np.diff(np.log(traj[-5:-1])))  # super-linear
    assert steps_to_zero(peb_map, SBP, 0.3, 0.3) <= 8
    assert steps_to_zero(exact_map, SBP, 0.3, 0.3) <= 8


@pytest.mark.parametrize("emap", [resolvent_map, peb_map, exact_map, semi_implicit_map])
@pytest.mark.parametrize("act", [SIGN, SBP, PowerSum([(1.0, 0.0), (1.0, 1.0)]),
                                 PowerSum([(1.5, 1.0), (0.5, 3.0)]),
                                 PredefinedTimeJR("arctan", mu=0.5), PredefinedTimeJR("exp"),
                                 PredefinedTimeJR("exp", rho2=0.5), acts.get("power_sigmoid")],
                         ids=lambda a: f"{a.key}{a.params()}")
def test_maps_sign_preserving_and_nonexpansive(emap, act):
    """No overshoot (hence no numerical chattering) for every h, including h >> 1 and
    h far below machine precision (verifier: before the clip, the exact maps overshot
    by rounding there, and JR 'exp' with rho2 > 0 returned inf for |e| > 37)."""
    e = np.random.default_rng(0).standard_normal((3, 80)) * np.logspace(-10, 10, 80)
    for h in [1e-18, 1e-3, 0.3, 3.0, 100.0]:
        y = cd.apply_error_map({resolvent_map: "resolvent", peb_map: "peb", exact_map: "exact",
                                semi_implicit_map: "semi_implicit"}[emap], act, e, h)
        assert np.all(np.isfinite(y))
        assert np.all(np.abs(y) <= np.abs(e) * (1 + 1e-12))
        assert np.all((np.sign(y) == np.sign(e)) | (y == 0))


# ---------------------------------------------------------------------------
# (a) time-invariant problem: finite-step convergence vs O(h) explicit chattering
# ---------------------------------------------------------------------------
P_TI = make_problem("linear_system", n=4, kappa=10, seed=1, T=3.0, freq=0.0)


@pytest.mark.parametrize("key", ["peb_euler", "exact_flow_euler", "semi_implicit_euler",
                                 "implicit_taylor_zhang", "peb_taylor_zhang",
                                 "exact_flow_taylor_zhang"])
@pytest.mark.parametrize("act", [SIGN, SBP, PredefinedTimeJR("arctan", rho1=1.0)],
                         ids=["sign", "sign_bi_power", "jr_arctan"])
def test_time_invariant_finite_step_convergence(key, act):
    assert np.abs(P_TI.dEdt(np.ones((1, 4)), 0.7)).max() == 0.0
    s = simulate(P_TI, ConsistentZNN(act), make_discretization(key), tau=1e-2, gamma=20, B=3,
                 seed=0)
    r = s["residual"]
    hit = np.argmax(r < 1e-13, axis=0)
    # finitely many steps (exactly 0) for the dead-zone / exact maps; resolvent + sign-bi-power
    # only converges super-linearly (Sec. 3.1), which also passes this 1e-13 threshold
    assert np.all(r[hit, np.arange(3)] < 1e-13) and np.all(hit < 150)
    for b in range(3):
        assert np.all(r[hit[b]:, b] < 1e-13)                             # and stays there


@pytest.mark.parametrize("h", [0.1, 0.2, 0.4])
def test_time_invariant_explicit_sign_chatters_at_order_h(h):
    tau = 1e-2
    s = simulate(P_TI, DesignFormulaZNN(SIGN), make_discretization("euler"), tau=tau,
                 gamma=h / tau, B=3, seed=0)
    ratio = mssre(s) / h
    assert np.all((ratio > 0.5) & (ratio < 3.0)), ratio
    im = simulate(P_TI, ConsistentZNN(SIGN), make_discretization("exact_flow_euler"), tau=tau,
                  gamma=h / tau, B=3, seed=0)
    assert np.all(mssre(im) < 1e-13)


def test_time_invariant_error_map_reduction_with_memory():
    """Lemma (EM): for constant data E_{k+1} = Psi_h(E_k) exactly, also in the multistep
    phase; linear activation + resolvent gives r_{k+1} = r_k / (1+h) every step."""
    h = 0.5
    s = simulate(P_TI, ConsistentZNN("linear"), make_discretization("implicit_taylor_zhang"),
                 tau=1e-2, gamma=h / 1e-2, T=0.3, seed=0)
    r = s["residual"][:, 0]
    assert np.allclose(r[1:] / r[:-1], 1 / (1 + h), rtol=1e-8)
    nv = simulate(P_TI, ConsistentZNN("linear"),
                  make_discretization("implicit_taylor_zhang_naive"), tau=1e-2, gamma=h / 1e-2,
                  T=0.3, seed=0)["residual"][:, 0]
    assert not np.allclose(nv[1:] / nv[:-1], 1 / (1 + h), rtol=1e-3)


def test_predefined_time_in_discrete_time():
    """jr_predefined + exact map: residual ~ 0 for k >= ceil(rho1/(gamma tau)) whatever x0
    (discrete predefined time, JR Cor. 7), while explicit Euler blows up for large x0."""
    act = PredefinedTimeJR("arctan", a=1.0, rho1=1.0)
    tau, gamma = 1e-2, 5.0
    x0 = 1e3 * np.random.default_rng(3).standard_normal((2, 4))
    s = simulate(P_TI, ConsistentZNN(act), make_discretization("exact_flow_euler"), tau=tau,
                 gamma=gamma, B=2, x0=x0, T=0.5)
    kc = math.ceil(1.0 / (gamma * tau))
    assert np.all(s["residual"][kc:] < 1e-9 * np.abs(x0).max())
    ex = simulate(P_TI, DesignFormulaZNN(act), make_discretization("euler"), tau=tau, gamma=gamma,
                  B=2, x0=x0, T=0.5)
    assert ex["diverged"].all()


# ---------------------------------------------------------------------------
# (b) time-varying linear system: residual order of the underlying look-ahead formula
# ---------------------------------------------------------------------------
P_TV = make_problem("linear_system", n=4, kappa=10, seed=1, T=4.0)


def _order(key, act, h=0.3):
    errs = [float(mssre(simulate(P_TV, ConsistentZNN(act), make_discretization(key), tau=t,
                                 gamma=h / t, seed=0))[0]) for t in TAUS]
    return empirical_order(TAUS, errs)[0], errs


@pytest.mark.parametrize("key,order", [("peb_euler", 2), ("exact_flow_euler", 2),
                                       ("semi_implicit_euler", 2), ("implicit_taylor_zhang", 3),
                                       ("peb_taylor_zhang", 3), ("exact_flow_taylor_zhang", 3),
                                       ("semi_implicit_taylor_zhang", 3)])
@pytest.mark.parametrize("act", ["linear", SIGN], ids=["linear", "sign"])
def test_tv_residual_order(key, order, act):
    slope, errs = _order(key, act)
    assert abs(slope - order) < 0.3, (slope, errs)


def test_sign_type_residual_is_one_step_truncation_error():
    """Dead-zone maps are deadbeat: the residual floor equals the formula's one-step local
    error, i.e. the linear-ZNN floor divided by its amplification (1+h)/h."""
    h, tau = 0.3, 1e-2
    lin = mssre(simulate(P_TV, ConsistentZNN("linear"), make_discretization("peb_euler"),
                         tau=tau, gamma=h / tau))[0]
    sgn = mssre(simulate(P_TV, ConsistentZNN(SIGN), make_discretization("peb_euler"), tau=tau,
                         gamma=h / tau))[0]
    assert abs(lin / sgn - (1 + h) / h) < 0.15 * (1 + h) / h


def test_error_memory_tz_unconditionally_stable():
    """Explicit Taylor-Zhang is unstable for h >= 1; with the error-memory rate the error
    recursion is Psi_h (here 1/(1+h)) for every h and the O(tau^3) floor is kept."""
    ex = simulate(P_TV, make_model("znn"), make_discretization("taylor_zhang"), tau=1e-2,
                  gamma=500.0, T=2.0)
    assert ex["diverged"].all()
    for h in [5.0, 50.0]:
        slope, errs = _order("implicit_taylor_zhang", "linear", h=h)
        assert abs(slope - 3) < 0.3, (h, slope, errs)


# ---------------------------------------------------------------------------
# (c) chattering
# ---------------------------------------------------------------------------
def test_chattering_reduced_vs_explicit():
    # fixed start (explicit sign-bi-power diverges from far random starts at h = 0.3)
    kw = dict(tau=1e-2, gamma=30, B=1, seed=0, x0=P_TV.x0(np.random.default_rng(0), 1))
    for act in [SBP, SIGN]:
        ex = simulate(P_TV, DesignFormulaZNN(act), make_discretization("euler"), **kw)
        ex_tz = simulate(P_TV, DesignFormulaZNN(act), make_discretization("taylor_zhang"), **kw)
        for key, ref in [("peb_euler", ex), ("exact_flow_euler", ex), ("semi_implicit_euler", ex),
                         ("peb_taylor_zhang", ex_tz), ("implicit_taylor_zhang", ex_tz)]:
            im = simulate(P_TV, ConsistentZNN(act), make_discretization(key), **kw)
            assert chattering(im)[0] < 0.5 * chattering(ref)[0], (act.key, key)
            assert mssre(im)[0] < 0.2 * mssre(ref)[0], (act.key, key)
            assert chattering(im)[0] < 0.05          # smooth tracking: O(tau * frequency)
    # ablation: without the error memory, Taylor-Zhang re-introduces chattering
    nv = simulate(P_TV, ConsistentZNN(SIGN), make_discretization("implicit_taylor_zhang_naive"),
                  **kw)
    em = simulate(P_TV, ConsistentZNN(SIGN), make_discretization("implicit_taylor_zhang"), **kw)
    assert chattering(nv)[0] > 5 * chattering(em)[0] and mssre(nv)[0] > 1e3 * mssre(em)[0]


# ---------------------------------------------------------------------------
# interface / fallbacks
# ---------------------------------------------------------------------------
def test_base_model_honours_error_map_request():
    """Core change request 'consistent 1' has landed: the base 'znn' model now
    honours ErrorMapRequest, so it equals 'znn_consistent' on every consistent
    scheme (no fallback, no warning)."""
    cd._WARNED.clear()
    kw = dict(tau=1e-2, gamma=30, seed=0, T=1.0)
    for key in ["peb_euler", "exact_flow_euler", "implicit_taylor_zhang"]:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            a = simulate(P_TV, make_model("znn", activation="sign_bi_power"),
                         make_discretization(key), **kw)
        assert not any("ignored the ErrorMapRequest" in str(x.message) for x in w)
        b = simulate(P_TV, ConsistentZNN("sign_bi_power"), make_discretization(key), **kw)
        assert np.array_equal(a["residual"], b["residual"], equal_nan=True), key


def test_consistent_model_is_plain_znn_without_request():
    kw = dict(tau=1e-2, gamma=30, seed=0, T=1.0)
    for key in ["euler", "implicit_euler_err"]:
        a = simulate(P_TV, ConsistentZNN("sign_bi_power"), make_discretization(key), **kw)
        b = simulate(P_TV, make_model("znn", activation="sign_bi_power"),
                     make_discretization(key), **kw)
        assert np.array_equal(a["residual"], b["residual"])


def test_exact_falls_back_for_activation_without_closed_form():
    cd._WARNED.clear()
    act = acts.get("power_sigmoid")
    e = np.linspace(-2, 2, 7)[None]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        y = cd.apply_error_map("exact", act, e, 0.3)
    assert np.allclose(y, act.prox(e, 0.3)) and any("unavailable" in str(x.message) for x in w)


def test_error_memory_with_registered_zead_formula():
    """The error-memory rate works for any registered LookAhead (here a 4th-order ZeaD
    formula of the zead area): finite-step convergence on constant data, and with a
    sign-type map the residual order equals the formula's expected order."""
    from znnbench.registry import DISCRETIZATIONS
    key = "zead_6i_guo2017"
    if key not in DISCRETIZATIONS:
        pytest.skip(f"{key} not registered")
    d = cd.consistent(key, "exact")
    s = simulate(P_TI, ConsistentZNN(SIGN), d, tau=1e-2, gamma=20, T=2.0, seed=0)
    assert s["residual"][-1, 0] < 1e-13
    taus = [3e-2, 1e-2, 3e-3]
    x0 = P_TV.x0(np.random.default_rng(0), 1)
    errs = [float(mssre(simulate(P_TV, ConsistentZNN(SIGN), cd.consistent(key, "exact"), tau=t,
                                 gamma=0.1 / t, seed=0, x0=x0))[0]) for t in taus]
    assert abs(empirical_order(taus, errs)[0] - d.expected_order) < 0.3, errs


# ---------------------------------------------------------------------------
# added by the adversarial verification (see docs/spec/consistent.md, Verification)
# ---------------------------------------------------------------------------
def test_activation_keys_and_contract():
    """The test-vehicle activations are resolvable by key locally (model 'znn_consistent',
    cd.get_activation) but are NOT injected into the shared activations.REGISTRY, which
    tests/test_activations.py iterates; 'power_sum' stays the Yang-Zhang activation.  Both
    follow the activations.py attribute contract."""
    for key, cls in [("signed_power_sum", PowerSum), ("jr_predefined", PredefinedTimeJR),
                     ("power_sum", PowerSum)]:
        assert acts.REGISTRY.get(key) is not cls
    for act in [PowerSum(), PredefinedTimeJR()]:
        assert act.source and act.finite_time == (act.convergence not in {"exponential", "asymptotic"})
    m = make_model("znn_consistent", activation="jr_predefined", kappa="exp", rho1=0.5)
    assert isinstance(m.act, PredefinedTimeJR) and m.act.rho1 == 0.5
    assert make_model("znn_consistent", activation="sign_bi_power", r=0.3).act.r == 0.3
    sp = cd.get_activation("signed_power_sum", terms=[(0.5, 0.5), (0.5, 2.0)])
    assert np.allclose(sp(np.linspace(-3, 3, 13)), SBP(np.linspace(-3, 3, 13)), rtol=1e-15)
    assert sp.convergence == "fixed" and PowerSum([(1, 0)]).convergence == "finite"
    assert PowerSum([(1, 1)]).convergence == "exponential" and PowerSum([(1, 3)]).convergence == "asymptotic"
    # AL19 Thm 1 (k = 1) by quadrature, and a valid (looser) bound for three terms
    tmax = quad(lambda z: 1 / (0.5 * z ** 0.5 + 0.5 * z ** 2), 0, np.inf)[0]
    assert abs(sp.uniform_bound(gamma=2.0) - tmax / 2.0) < 1e-8
    three = PowerSum([(0.5, 0.5), (0.3, 1.0), (0.5, 2.0)])
    assert three.settling_time(1e6) <= three.uniform_bound() <= sp.uniform_bound() + 1e-12
    assert np.isinf(PowerSum([(1, 0.5)]).uniform_bound())
    jr = PredefinedTimeJR("rational", a=2.0, rho1=0.7, rho2=0.3)
    e0 = np.array([0.1, -3.0])
    assert jr.settling_time(e0, gamma=2.0) == pytest.approx(0.7 * (3 / 5) ** 0.7 / 2.0, rel=1e-14)
    assert jr.settling_time(e0) == pytest.approx(acts.Activation.settling_time(jr, e0), rel=1e-8)
    assert jr.uniform_bound(gamma=2.0) == jr.upper_bound_time(e0, gamma=2.0) == 0.35


def test_scheme_names_match_registry_keys():
    for key in KEYS[:-1]:
        assert make_discretization(key).name == key
    assert make_discretization("implicit_taylor_zhang", error_map="peb").name == "peb_taylor_zhang"
    with pytest.raises(ValueError):
        cd.consistent("euler", "explicit")


ZOO_TERMS = ["linear", "sign", "sign_power", "sign_bi_power", "sfaf", "wsbp", "vaf", "fixed_time",
             "power", "power_sum"]


@pytest.mark.parametrize("key", ZOO_TERMS)
def test_zoo_activation_power_terms(key):
    """Zoo members that are signed power sums are decomposed (and the decomposition is
    checked against the activation), so 'peb'/'exact' no longer silently fall back."""
    if key not in acts.REGISTRY:
        pytest.skip(f"activation {key} not registered")
    act = acts.get(key)
    terms = cd.power_terms(act)
    assert terms is not None
    e = np.random.default_rng(1).standard_normal(200) * np.logspace(-4, 1, 200)
    assert np.allclose(sum(a * np.sign(e) * np.abs(e) ** p for a, p in terms), act(e), rtol=1e-12)
    cd._WARNED.clear()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        y = cd.apply_error_map("peb", act, e[None], 0.3)
    assert not any("unavailable" in str(x.message) for x in w)
    assert np.all(np.abs(y) <= np.abs(e)) and np.all((np.sign(y) == np.sign(e)) | (y == 0))


def test_zoo_exact_map_matches_zoo_settling_time():
    """Cross-area check: the exact map needs ceil(T(e0)/h) steps, with T the closed-form
    settling times stated in activations.py (sign, sign_power, sfaf)."""
    for key, kw in [("sign", {}), ("sign_power", {"r": 0.4}), ("sfaf", {"r": 0.3, "b1": 2.0, "b2": 0.5})]:
        if key not in acts.REGISTRY:
            continue
        act = acts.get(key, **kw)
        for e0 in [0.013, 0.7, 42.0]:
            for h in [0.037, 0.31]:
                T = act.settling_time(e0)
                if abs(T / h - round(T / h)) > 1e-6:
                    assert steps_to_zero(exact_map, act, e0, h) == math.ceil(T / h), (key, e0, h)


def test_power_terms_guard_rejects_mismatch():
    class Fake(acts.Activation):
        key = "sign_power"          # wrong formula under a zoo key
        r = 0.5

        def __call__(self, e):
            return np.tanh(np.asarray(e, dtype=float))
    assert cd.power_terms(Fake()) is None
    assert cd.exact_map(Fake(), np.ones((1, 2)), 0.1) is None


def test_predefined_exp_exact_map():
    """'predefined_exp' (Li et al. 2019, F2) with k2 = k3 = 0 is JR Eq. (1) with kappa =
    1 - exp(-r^p), rho1 = 1/k1: its exact map is the flow and hits 0 by t = 1/(gamma k1)."""
    if "predefined_exp" not in acts.REGISTRY:
        pytest.skip("predefined_exp not registered")
    act = acts.get("predefined_exp", p=0.5, k1=2.0, k2=0.0, k3=0.0)
    for e0 in [1.5, -0.2]:
        assert abs(exact_map(act, np.array([[e0]]), 0.03)[0, 0] - flow(act, e0, 0.03)) < 1e-12
    assert steps_to_zero(exact_map, act, 30.0, 0.05) <= math.ceil(0.5 / 0.05)
    assert exact_map(acts.get("predefined_exp"), np.ones((1, 1)), 0.1) is None   # k2, k3 > 0


def test_batch_per_trajectory_gain():
    """API contract: gamma is (B, 1); a batch with different gains equals separate runs."""
    P = make_problem("linear_system", n=4, kappa=10, seed=1, T=0.5)
    x0 = np.random.default_rng(5).uniform(-2, 2, (3, 4))
    gam = np.array([5.0, 30.0, 300.0])
    for key in ["peb_taylor_zhang", "exact_flow_euler", "semi_implicit_taylor_zhang"]:
        sb = simulate(P, ConsistentZNN("sign_bi_power"), make_discretization(key), tau=1e-2, B=3,
                      gamma=gam, x0=x0)
        for b in range(3):
            s1 = simulate(P, ConsistentZNN("sign_bi_power"), make_discretization(key), tau=1e-2,
                          B=1, gamma=gam[b], x0=x0[b:b + 1])
            assert np.allclose(sb["residual"][:, b], s1["residual"][:, 0], rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("name", ["resolvent", "peb", "exact", "semi_implicit"])
def test_maps_finite_for_huge_errors(name):
    """|e|^{p-1} overflow (p > 1 terms) must not produce nan (verifier fix: 1/(1/y + c) form)."""
    e = np.array([[0.0, 1e-300, 3.0, -1e150, 1e200, 1e308]])
    for act in [PowerSum([(1.0, 3.0)]), PowerSum([(1.5, 1.0), (0.5, 3.0)]), SBP,
                PowerSum([(1.0, 0.0), (1.0, 5.0)])]:
        with np.errstate(all="ignore"):
            y = cd.apply_error_map(name, act, e, 0.3)
        assert np.all(np.isfinite(y)) and np.all(np.abs(y) <= np.abs(e)), (act.params(), y)
