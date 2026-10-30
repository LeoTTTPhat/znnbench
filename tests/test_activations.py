"""Tests for the activation zoo (znnbench/activations.py).

Spec: docs/spec/activations.md.  Every formula is re-transcribed here
independently of the implementation (spot values), and the continuous-time
claims are checked on the scalar error dynamics e' = -gamma Phi(e) with RK4 at
a small step (finite/fixed/predefined-time ones reach ~0 no later than the
stated bound; asymptotic ones follow their closed-form solutions)."""
import math

import numpy as np
import pytest

from znnbench import activations as A
from znnbench import make_discretization, make_model, make_problem, simulate

EXPECTED = {"linear", "power", "bipolar_sigmoid", "power_sigmoid", "power_sum",
            "hyperbolic_sine", "sign", "sign_power", "sign_bi_power", "sfaf", "wsbp",
            "saturated_sign_power", "vaf", "fixed_time", "pvaf1", "pvaf2",
            "predefined_exp", "strict_predefined"}
# Only the classes defined in znnbench/activations.py: sibling modules may add
# their own keys to the shared registry at import time, which must not make
# these tests order-dependent (verifier finding: running test_consistent.py
# first used to inject 'jr_predefined' and break test_registry_contains_zoo).
OWN = sorted(k for k, c in A.REGISTRY.items() if c.__module__ == A.__name__)
TI = sorted(k for k in OWN if not A.REGISTRY[k].time_varying)
FINITE = sorted(k for k in TI if A.REGISTRY[k].finite_time)
ASYMPTOTIC = sorted(k for k in TI if not A.REGISTRY[k].finite_time)


def sig(e, r):
    return np.sign(e) * np.abs(e) ** r


def rk4(f, e0, T, tau):
    """Classical RK4 for e' = f(t, e), vectorised over the entries of e0."""
    e = np.array(e0, dtype=float)
    n = int(round(T / tau))
    out = np.empty((n + 1, e.size))
    out[0] = e
    for k in range(n):
        t = k * tau
        k1 = f(t, e)
        k2 = f(t + tau / 2, e + tau / 2 * k1)
        k3 = f(t + tau / 2, e + tau / 2 * k2)
        k4 = f(t + tau, e + tau * k3)
        e = e + tau / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        out[k + 1] = e
    return np.arange(n + 1) * tau, out


# ---------------------------------------------------------------------------
# registry / API
# ---------------------------------------------------------------------------
CLASSES = {"exponential", "asymptotic", "finite", "fixed", "predefined", "strictly-predefined"}


def test_registry_contains_zoo():
    assert EXPECTED <= set(A.available()) and set(OWN) == EXPECTED
    for k in OWN:
        act = A.get(k)
        d = act.describe()
        assert d["key"] == k
        assert isinstance(act.finite_time, bool)
        assert act.convergence in CLASSES and act.claimed_convergence in CLASSES
        assert act.finite_time == (act.convergence not in {"exponential", "asymptotic"})
        assert act.source and act.provenance in {"primary", "secondary"}
    with pytest.raises(KeyError):                      # duplicate keys are refused
        A.register(type("Clash", (A.Activation,), {"key": "linear"}))


def test_claimed_vs_derived_convergence_class():
    """SBP and WSBP are claimed finite-time by their sources; ZNN-Bench labels
    them 'fixed' only because a finite e0-independent bound is DERIVED (and
    checked numerically in test_tight_fixed_time_bound)."""
    upgraded = {k for k in OWN if A.get(k).claimed_convergence != A.get(k).convergence}
    assert upgraded == {"sign_bi_power", "wsbp"}
    for k in upgraded:
        act = A.get(k)
        assert act.claimed_convergence == "finite" and act.convergence == "fixed"
        assert act.uniform_bound(1.0) < math.inf
        assert act.settling_time(1e12, 1.0) <= act.tight_uniform_bound(1.0) <= act.uniform_bound(1.0)


@pytest.mark.parametrize("key,kw", [
    ("power", {"p": 2}), ("power", {"p": 1}), ("power_sigmoid", {"p": 4}),
    ("sign_bi_power", {"r": 1.0}), ("sign_power", {"r": 1.5}), ("sfaf", {"b1": 0.0}),
    ("vaf", {"eta": 1.2}), ("vaf", {"omega": 0.9}), ("fixed_time", {"q": 1.0}),
    ("pvaf1", {"p": 0.4, "k": 2.0}), ("pvaf1", {"q": 0.6, "k": 2.0}), ("pvaf2", {"p": 0.4, "k": 2.0}),
    ("predefined_exp", {"p": 1.0}), ("saturated_sign_power", {"s": 0.0}),
    ("power_sum", {"N": 1}), ("strict_predefined", {"t_c": 0.0})])
def test_invalid_parameters_rejected(key, kw):
    with pytest.raises(ValueError):
        A.get(key, **kw)


# ---------------------------------------------------------------------------
# formulas (independent re-transcription of the source equations)
# ---------------------------------------------------------------------------
E = np.array([-2.3, -1.0, -0.4, -1e-3, 0.0, 1e-3, 0.7, 1.0, 1.9, 3.2])


def _reference(key, e):
    """(kwargs, Phi(e)) transcribed from docs/spec/activations.md."""
    if key == "linear":
        return {}, e
    if key == "power":
        return {"p": 5}, e ** 5
    if key == "bipolar_sigmoid":
        xi = 3.0
        return {"xi": xi}, (1 - np.exp(-xi * e)) / (1 + np.exp(-xi * e))
    if key == "power_sigmoid":
        p, xi = 5, 2.5
        s = (1 + np.exp(-xi)) / (1 - np.exp(-xi)) * (1 - np.exp(-xi * e)) / (1 + np.exp(-xi * e))
        return {"p": p, "xi": xi}, np.where(np.abs(e) >= 1, e ** p, s)
    if key == "power_sum":
        return {"N": 4}, e + e ** 3 + e ** 5 + e ** 7
    if key == "hyperbolic_sine":
        xi = 1.7
        return {"xi": xi}, np.exp(xi * e) / 2 - np.exp(-xi * e) / 2
    if key == "sign":
        return {}, np.sign(e)
    if key == "sign_power":
        return {"r": 0.3}, sig(e, 0.3)
    if key == "sign_bi_power":
        return {"r": 0.4}, 0.5 * sig(e, 0.4) + 0.5 * sig(e, 1 / 0.4)
    if key == "sfaf":
        return {"r": 0.6, "b1": 2.0, "b2": 0.5}, 2.0 * sig(e, 0.6) + 0.5 * e
    if key == "wsbp":
        return ({"r": 0.25, "k1": 1.5, "k2": 0.7, "k3": 2.0},
                1.5 * sig(e, 0.25) + 0.7 * sig(e, 4.0) + 2.0 * e)
    if key == "saturated_sign_power":
        return {"r": 0.5, "s": 1.2}, np.sign(e) * np.minimum(np.abs(e) ** 0.5, 1.2)
    if key == "vaf":
        kw = dict(a1=1.5, a2=0.5, a3=2.0, a4=0.3, eta=0.2, omega=3.0)
        return kw, (1.5 * np.abs(e) ** 0.2 + 0.5 * np.abs(e) ** 3.0) * np.sign(e) + 2.0 * e + 0.3 * np.sign(e)
    if key == "fixed_time":
        kw = dict(k1=2.0, k2=0.25, k3=1.5, p=0.7, q=1.5)
        return kw, (2.0 * np.abs(e) ** 0.7 + 0.25 * np.abs(e) ** 1.5) * np.sign(e) + 1.5 * e
    if key == "pvaf1":
        kw = dict(a=2.0, b=0.5, p=0.8, q=0.3, k=1.5, c1=0.4, c2=0.2)
        x = np.abs(e)
        return kw, (2.0 * x ** 0.8 + 0.5 * x ** 0.3) ** 1.5 * np.sign(e) + 0.4 * e + 0.2 * np.sign(e)
    if key == "pvaf2":
        kw = dict(a=2.0, b=0.5, p=0.8, k=1.5, d1=0.4, d2=0.2)
        x = np.abs(e)
        return kw, (2.0 * x ** 0.8 + 0.5) ** 1.5 * np.sign(e) + 0.4 * e + 0.2 * np.sign(e)
    if key == "predefined_exp":
        kw = dict(p=0.3, k1=1.5, k2=0.5, k3=0.1)
        x = np.abs(e)
        return kw, 1.5 * np.exp(x ** 0.3) * x ** 0.7 * np.sign(e) / 0.3 + 0.5 * e + 0.1 * np.sign(e)
    raise KeyError(key)


@pytest.mark.parametrize("key", TI)
def test_formula_matches_source(key):
    kw, ref = _reference(key, E)
    assert np.allclose(A.get(key, **kw)(E), ref, rtol=1e-13, atol=1e-15)


def test_special_cases_and_equivalences():
    e = np.linspace(-4, 4, 81)
    # normalised bipolar sigmoid is the |e|<1 branch of power-sigmoid
    bs = A.get("bipolar_sigmoid", xi=4.0, normalized=True)
    ps = A.get("power_sigmoid", p=3, xi=4.0)
    inner = np.abs(e) < 1
    assert np.allclose(bs(e)[inner], ps(e)[inner]) and bs(np.array([1.0]))[0] == pytest.approx(1.0)
    # sign-bi-power = WSBP with k1 = k2 = 1/2, k3 = 0
    assert np.allclose(A.get("sign_bi_power", r=0.3)(e), A.get("wsbp", r=0.3, k1=0.5, k2=0.5, k3=0.0)(e))
    # SSBP-AF of Zuo et al. 2022 (eq. 6): f(q) = q + sign(q)|q|^r
    assert np.allclose(A.get("sfaf", r=0.4)(e), e + sig(e, 0.4))
    # VAF with a4 = 0 is the fixed-time bi-power; PVAF-1 with k = 1 is the VAF
    v = A.get("vaf", a1=2, a2=3, a3=0.5, a4=0, eta=0.3, omega=1.7)
    assert np.allclose(v(e), A.get("fixed_time", k1=2, k2=3, k3=0.5, p=0.3, q=1.7)(e))
    v4 = A.get("vaf", a1=2, a2=3, a3=0.5, a4=0.7, eta=0.3, omega=1.7)
    assert np.allclose(v4(e), A.get("pvaf1", a=3, b=2, p=1.7, q=0.3, k=1.0, c1=0.5, c2=0.7)(e))
    # strict predefined-time: linear/(t_c - t) before t_c, x + sig^p + xi sign after
    sp = A.get("strict_predefined", t_c=2.0, p=0.5, xi=0.3)
    assert np.allclose(sp(e, t=0.5), e / 1.5)
    assert np.allclose(sp(e, t=2.5), e + sig(e, 0.5) + 0.3 * np.sign(e))
    assert np.allclose(sp.at(0.5)(e), e / 1.5)
    with pytest.raises(ValueError):
        sp(e)


# ---------------------------------------------------------------------------
# structural properties
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key", TI)
def test_odd_and_monotone(key):
    act = A.get(key)
    e = np.linspace(-4, 4, 2001)
    y = act(e)
    assert np.allclose(act(-e), -y, rtol=1e-14, atol=0)
    assert act(np.array([0.0]))[0] == 0.0
    assert np.all(np.diff(y) >= 0)
    assert np.all(y[e > 0] > 0)


@pytest.mark.parametrize("key", TI)
def test_slope0(key):
    act = A.get(key)
    if np.isfinite(act.slope0):
        d = 1e-6
        num = (act(np.array([d]))[0] - act(np.array([-d]))[0]) / (2 * d)
        assert num == pytest.approx(act.slope0, rel=1e-6, abs=1e-9)
    else:   # non-Lipschitz at 0: the difference quotient blows up
        q = [act(np.array([d]))[0] / d for d in (1e-2, 1e-4, 1e-6)]
        assert q[0] < q[1] < q[2] and q[2] > 100


@pytest.mark.parametrize("key", TI + ["strict_predefined@0.4", "strict_predefined@1.5"])
@pytest.mark.parametrize("h", [1e-3, 0.2, "batch", 5.0])
def test_prox_is_resolvent(key, h):
    if "@" in key:
        name, t = key.split("@")
        act = A.get(name, t_c=1.0, xi=0.4).at(float(t))
    else:
        act = A.get(key)
    rng = np.random.default_rng(1)
    e = np.vstack([rng.standard_normal((3, 6)) * 2, [0.0, 1e-12, -1e-6, 0.05, 20.0, -60.0]])
    h = np.array([[0.01], [0.3], [1.0], [2.0]]) if h == "batch" else h
    hb = np.broadcast_to(h, e.shape)
    y = act.prox(e, h)
    with np.errstate(over="ignore", invalid="ignore"):
        res = y + hb * act(y) - e
    dead = np.abs(e) <= hb * act.jump0()
    assert np.all(y[dead] == 0.0)                         # inclusion: dead zone -> exactly 0
    live = ~dead
    assert np.all(np.abs(res[live]) <= 1e-10 * np.maximum(1.0, np.abs(e[live])))
    assert np.all(np.sign(y[live]) == np.sign(e[live]))
    assert np.all(np.abs(y) <= np.abs(e) + 1e-15)


@pytest.mark.parametrize("key,kw", [
    ("sign_power", {"r": 0.1}), ("sign_power", {"r": 0.02}),
    ("fixed_time", dict(k1=2.17, k2=0.42, k3=2.98, p=0.07, q=3.97)),
    ("vaf", dict(a1=0.3, a2=2.0, a3=0.0, a4=0.0, eta=0.05, omega=3.5)),
    ("wsbp", dict(r=0.08, k1=1.5, k2=0.5, k3=0.0)), ("sfaf", dict(r=0.06, b1=3.0, b2=0.1)),
    ("pvaf1", dict(a=1.0, b=2.0, p=2.0, q=0.05, k=1.0, c1=0.0, c2=0.0)),
    ("saturated_sign_power", dict(r=0.05, s=3.0)), ("predefined_exp", dict(p=0.9, k1=2.0, k2=0.0, k3=0.0))])
def test_prox_strongly_non_lipschitz(key, kw):
    """Small exponents and large h put the resolvent 30-300 orders of magnitude
    below |e|; the generic prox must still satisfy y + h Phi(y) = e with
    RELATIVE accuracy (a linear bisection on [0, |e|] stalls at |e| 2^-80 and
    gave residuals ~ 1e2 here before the verifier fix)."""
    act = A.get(key, **kw)
    e = np.array([-100.0, -3.0, -0.7, -1e-5, 1e-9, 2e-3, 0.4, 5.0, 233.0])
    for h in (1e-3, 0.5, 7.0, 1e4):
        y = act.prox(e, h)
        live = np.abs(e) > h * act.jump0()
        assert np.all(y[~live] == 0.0)
        with np.errstate(over="ignore"):
            res = y + h * act(y) - e
        assert np.all(np.abs(res[live]) <= 1e-12 * np.abs(e[live])), (h, res)
        assert np.all(np.sign(y[live]) == np.sign(e[live])) and np.all(np.abs(y) <= np.abs(e))


@pytest.mark.parametrize("key,kw", [("power", {"p": 3}), ("power", {"p": 5}), ("power", {"p": 9}),
                                    ("sign", {}), ("linear", {})])
def test_analytic_prox_matches_bisection(key, kw):
    act = A.get(key, **kw)
    e = np.linspace(-30, 30, 241)
    for h in (1e-8, 1e-2, 0.7, 40.0):
        assert np.allclose(act.prox(e, h), A.Activation.prox(act, e, h, iters=200), rtol=1e-12, atol=1e-13)


# ---------------------------------------------------------------------------
# continuous-time behaviour
# ---------------------------------------------------------------------------
def _exact(key, e0, g, t):
    if key == "linear":
        return e0 * np.exp(-g * t)
    if key == "power":        # p = 3
        return e0 / np.sqrt(1 + 2 * g * e0 ** 2 * t)
    if key == "hyperbolic_sine":      # xi = 3: tanh(xi e/2) = tanh(xi e0/2) exp(-g xi t)
        return 2 / 3.0 * np.arctanh(np.tanh(1.5 * e0) * np.exp(-3.0 * g * t))
    if key == "bipolar_sigmoid":      # xi = 4: sinh(xi e/2) = sinh(xi e0/2) exp(-g xi t/2)
        return 0.5 * np.arcsinh(np.sinh(2.0 * e0) * np.exp(-2.0 * g * t))
    if key == "power_sum":            # N = 2 (Phi = e + e^3): e/sqrt(1+e^2) decays like exp(-g t)
        u = e0 / np.sqrt(1 + e0 ** 2) * np.exp(-g * t)
        return u / np.sqrt(1 - u ** 2)
    return None


@pytest.mark.parametrize("key", ["linear", "power", "hyperbolic_sine", "bipolar_sigmoid", "power_sum"])
def test_asymptotic_dynamics_match_closed_form(key):
    act = A.get(key, N=2) if key == "power_sum" else A.get(key)
    e0, g = np.array([-1.5, 0.2, 0.9]), 2.0
    t, tr = rk4(lambda t, e: -g * act(e), e0, 1.0, 2.5e-4)
    ref = _exact(key, e0[None, :], g, t[:, None])
    assert np.allclose(tr, ref, rtol=1e-7, atol=1e-12)
    assert act.settling_time(e0, g) == math.inf and act.upper_bound_time(e0, g) is None


@pytest.mark.parametrize("key", [k for k in ASYMPTOTIC if k != "power"])
def test_local_exponential_rate_is_slope0(key):
    act = A.get(key)
    e0, g = 1e-5, 1.5
    t, tr = rk4(lambda t, e: -g * act(e), [e0], 1.0, 1e-3)
    assert tr[-1, 0] / e0 == pytest.approx(math.exp(-g * act.slope0), rel=1e-4)


@pytest.mark.parametrize("key", [k for k in ASYMPTOTIC if k != "power"])
def test_euler_local_stability_threshold(key):
    """Discrete-time counterpart of slope0: explicit Euler e+ = e - h Phi(e) is
    locally contractive at 0 iff h Phi'(0) < 2 (e.g. h < 0.964 for the default
    power-sigmoid, h < 2/3 for sinh with xi = 3), whatever the continuous-time
    convergence class."""
    act = A.get(key)
    hmax = 2.0 / act.slope0
    for fac, stable in ((0.9, True), (1.1, False)):
        e = np.array([1e-6])
        with np.errstate(over="ignore", invalid="ignore"):
            for _ in range(60):
                e = e - fac * hmax * act(e)
        assert (abs(e[0]) < 1e-8) if stable else not (abs(e[0]) < 1e-5)


@pytest.mark.parametrize("key", FINITE)
def test_settling_time_within_stated_bounds(key):
    act = A.get(key)
    for e0 in np.logspace(-4, 4, 17):
        T = act.settling_time(e0, 1.0)
        assert 0 < T < math.inf
        B = act.upper_bound_time(e0, 1.0)
        if B is not None:
            assert T <= B * (1 + 1e-9), (e0, T, B)
        assert T <= act.uniform_bound(1.0) * (1 + 1e-9)
        assert act.settling_time(e0, 4.0) == pytest.approx(T / 4.0, rel=1e-9)   # T ~ 1/gamma
    if act.convergence in ("fixed", "predefined"):
        assert act.uniform_bound(1.0) < math.inf
        assert act.settling_time(1e12, 1.0) <= act.uniform_bound(1.0)


def test_closed_form_settling_times():
    # sign: |e0|/gamma; sign-power: |e0|^(1-r)/(gamma(1-r)); SFAF: log formula (Hu et al. 2020, Thm 2)
    g = 1.7
    assert A.get("sign").settling_time(-2.5, g) == pytest.approx(2.5 / g)
    assert A.Activation.settling_time(A.get("sign"), -2.5, g) == pytest.approx(2.5 / g, rel=1e-9)
    sp = A.get("sign_power", r=0.3)
    assert A.Activation.settling_time(sp, 2.5, g) == pytest.approx(2.5 ** 0.7 / (g * 0.7), rel=1e-9)
    sf = A.get("sfaf", r=0.4, b1=2.0, b2=0.5)
    a1, a2 = g * 2.0, g * 0.5
    ref = math.log((a2 * 7.0 ** 0.6 + a1) / a1) / (a2 * 0.6)
    assert A.Activation.settling_time(sf, 7.0, g) == pytest.approx(ref, rel=1e-9)
    ss = A.get("saturated_sign_power", r=0.5, s=0.7)
    assert A.Activation.settling_time(ss, 5.0, g) == pytest.approx(ss.upper_bound_time(5.0, g), rel=1e-9)
    # sign-bi-power source bound (Li, Chen, Liu 2013): 2|e0|^(1-r)/(gamma(1-r))
    assert A.get("sign_bi_power", r=0.4).upper_bound_time([0.5, -3.0], g) == pytest.approx(
        2 * 3.0 ** 0.6 / (g * 0.6))
    # VAF / Li et al. F1 / AIMS 2022 eq. (4.6): 1/(g a1 (1-eta)) + 1/(g a2 (omega-1))
    v = A.get("vaf", a1=2.0, a2=0.5, eta=0.25, omega=3.0)
    assert v.upper_bound_time(123.0, g) == pytest.approx(1 / (g * 2.0 * 0.75) + 1 / (g * 0.5 * 2.0))
    # PVAF-1 (Zhou & Tan 2022, Thm 1)
    p1 = A.get("pvaf1", a=2.0, b=0.5, p=0.8, q=0.3, k=1.5)
    assert p1.upper_bound_time(1.0, g) == pytest.approx(
        1 / (g * 2.0 ** 1.5 * (1.2 - 1)) + 1 / (g * 0.5 ** 1.5 * (1 - 0.45)))
    # PVAF-2 (Zhou & Tan 2022, Lemma 2 eq. (15) with a, b -> g^(1/k) a, g^(1/k) b)
    p2 = A.get("pvaf2", a=2.0, b=0.5, p=0.8, k=1.5, d1=0.0, d2=0.0)
    assert p2.upper_bound_time(1.0, g) == pytest.approx(
        (0.5 / 2.0) ** (1 / 0.8) * (1 + 1 / (1.2 - 1)) / (g * 0.5 ** 1.5))
    # predefined exp (Li et al. 2019): bound 1/(g k1); exact (1 - exp(-|e0|^p))/(g k1) when k2 = k3 = 0
    pe = A.get("predefined_exp", p=0.3, k1=2.0, k2=0.0, k3=0.0)
    assert pe.upper_bound_time(4.0, g) == pytest.approx(1 / (g * 2.0))
    assert pe.settling_time(4.0, g) == pytest.approx((1 - math.exp(-4.0 ** 0.3)) / (g * 2.0), rel=1e-8)


def _random_params(key, rng):
    u = lambda lo, hi: float(rng.uniform(lo, hi))  # noqa: E731
    if key in ("sign_power", "sign_bi_power"):
        return {"r": u(0.05, 0.95)}
    if key == "sfaf":
        return dict(r=u(0.05, 0.95), b1=u(0.1, 3), b2=u(0.1, 3))
    if key == "wsbp":
        return dict(r=u(0.05, 0.95), k1=u(0.1, 3), k2=u(0.1, 3), k3=u(0, 3))
    if key == "saturated_sign_power":
        return dict(r=u(0.05, 0.95), s=u(0.1, 5))
    if key == "vaf":
        return dict(a1=u(.1, 3), a2=u(.1, 3), a3=u(0, 3), a4=u(0, 3), eta=u(.05, .95), omega=u(1.05, 4))
    if key == "fixed_time":
        return dict(k1=u(.1, 3), k2=u(.1, 3), k3=u(0, 3), p=u(.05, .95), q=u(1.05, 4))
    if key == "pvaf1":
        k = u(0.3, 3)
        return dict(a=u(.1, 3), b=u(.1, 3), p=u(1.05 / k, 4 / k), q=u(0.01, 0.95 / k), k=k,
                    c1=u(0, 2), c2=u(0, 2))
    if key == "pvaf2":
        k = u(0.3, 3)
        return dict(a=u(.1, 3), b=u(.1, 3), p=u(1.05 / k, 4 / k), k=k, d1=u(0, 2), d2=u(0, 2))
    if key == "predefined_exp":
        return dict(p=u(.05, .95), k1=u(.1, 3), k2=u(0, 2), k3=u(0, 2))
    return {}


@pytest.mark.parametrize("key", FINITE)
def test_bounds_hold_for_random_parameters(key):
    """The stated / derived bounds are checked at random valid (non-default)
    parameters against the exact quadrature settling time."""
    rng = np.random.default_rng(7)
    for _ in range(4):
        act = A.get(key, **_random_params(key, rng))
        g = float(rng.uniform(0.3, 5.0))
        for e0 in (1e-3, 0.37, 4.2, 1e3):
            T = act.settling_time(e0, g)
            B = act.upper_bound_time(e0, g)
            assert 0 < T <= act.uniform_bound(g) * (1 + 1e-9)
            assert B is None or T <= B * (1 + 1e-9), (act.params(), e0, T, B)
            if hasattr(act, "tight_uniform_bound"):
                assert T <= act.tight_uniform_bound(g) * (1 + 1e-9) <= act.uniform_bound(g) * (1 + 1e-9)


def test_tight_fixed_time_bound():
    f = A.get("fixed_time", k1=2.0, k2=0.5, k3=0.0, p=0.3, q=2.5)
    sup = f.settling_time(1e14, 1.0)
    assert sup == pytest.approx(f.tight_uniform_bound(1.0), rel=1e-6)
    assert f.tight_uniform_bound(1.0) < f.uniform_bound(1.0)
    v = A.get("vaf")
    assert v.settling_time(1e14, 1.0) <= v.tight_uniform_bound(1.0) <= v.uniform_bound(1.0)
    sb = A.get("sign_bi_power", r=0.5)       # derived uniform bound 2(1+r)/(gamma(1-r))
    assert sb.settling_time(1e14, 1.0) <= sb.uniform_bound(1.0) == pytest.approx(6.0)
    # tight SBP bound: 2 int_0^inf ds/(s^(1/2) + s^2) = 2 pi/(1.5 sin(pi/3)) = 4.8368...
    assert sb.tight_uniform_bound(1.0) == pytest.approx(2 * math.pi / (1.5 * math.sin(math.pi / 3)), rel=1e-12)
    assert sb.settling_time(1e14, 1.0) == pytest.approx(sb.tight_uniform_bound(1.0), rel=1e-6)
    # source bound vs exact for e0 = 1e3 (spec example): 126.5 vs 4.83
    assert sb.upper_bound_time(1e3, 1.0) == pytest.approx(126.49, rel=1e-4)
    assert sb.settling_time(1e3, 1.0) == pytest.approx(4.8348, rel=1e-4)
    w = A.get("wsbp", r=0.5, k1=2.0, k2=0.5, k3=0.0)  # exact sup when k3 = 0 (k3 > 0 only shortens T)
    assert w.settling_time(1e14, 1.0) == pytest.approx(w.tight_uniform_bound(1.0), rel=1e-6)
    assert A.get("wsbp", r=0.5, k1=2.0, k2=0.5, k3=1.0).settling_time(1e14, 1.0) < w.tight_uniform_bound(1.0)


@pytest.mark.parametrize("key", FINITE)
def test_rk4_reaches_zero_by_stated_bound(key):
    """RK4 (tau = 2e-4) on e' = -gamma Phi(e): |e| <= tol is first reached no
    later than the source bound (or the exact settling time when the source
    states none) and no earlier than T(e0) - T(tol); it stays there."""
    act = A.get(key)
    e0 = np.array([-3.0, 0.3, 1.0])
    ref1 = act.upper_bound_time(e0, 1.0)
    ref1 = act.settling_time(e0, 1.0) if ref1 is None else ref1
    g = ref1                                  # gamma such that the bound for max|e0| is 1 s
    tau, tol = 2e-4, 2e-3
    t, tr = rk4(lambda t, e: -g * act(e), e0, 1.15, tau)
    for i, x0 in enumerate(e0):
        Ti = act.settling_time(x0, g)
        Bi = act.upper_bound_time(x0, g)
        Bi = Ti if Bi is None else Bi
        assert Ti <= Bi * (1 + 1e-9)
        below = np.abs(tr[:, i]) <= tol
        hit = t[np.argmax(below)]
        assert below.any() and hit <= Bi + 2 * tau, (x0, hit, Bi)
        assert hit >= Ti - act.settling_time(tol, g) - 2 * tau, (x0, hit, Ti)
        assert np.all(np.abs(tr[t >= Bi + 2 * tau, i]) <= tol)


@pytest.mark.parametrize("gamma", [1.0, 3.0])
def test_strict_predefined_time(gamma):
    """e(t) = e0 ((t_c - t)/t_c)^gamma before t_c for every e0.  At t = t_c
    the activation switches branch, so the LAST fixed step (which samples
    t = t_c) leaves an O(tau) remainder (for gamma = 1, RK4 gives exactly
    e(t_c - tau)/6 = e0 tau/(6 t_c)); the post-t_c finite-time branch then
    removes it down to the RK4 chattering floor.  This discrete-time loss of
    the strict property is expected."""
    tc = 0.8
    act = A.get("strict_predefined", t_c=tc, p=0.5, xi=0.0)
    e0 = np.array([-5.0, 0.1, 2.0, 50.0])
    tau = 1e-4
    t, tr = rk4(lambda t, e: -gamma * act(e, t=t), e0, 1.2, tau)
    k_c = int(round(tc / tau))
    ref = e0[None, :] * ((tc - t[:k_c, None]) / tc) ** gamma
    assert np.allclose(tr[:k_c], ref, rtol=1e-6, atol=1e-9 * np.abs(e0).max())
    assert np.all(np.abs(tr[k_c]) <= tau / tc * np.abs(e0))          # O(tau) remainder at t_c
    if gamma == 1.0:
        assert np.allclose(tr[k_c], e0 * tau / (6 * tc), rtol=1e-6)
    # removed after t_c, down to the fixed-step chattering floor (gamma tau)^(1/(1-p)) of sig^p
    assert np.all(np.abs(tr[t >= tc + 0.1]) <= 10 * (gamma * tau) ** 2)
    assert act.upper_bound_time(e0, gamma) == act.settling_time(e0, gamma) == tc
    # implicit (resolvent) form before t_c is linear with rate 1/(t_c - t)
    assert np.allclose(act.prox(e0, 0.1, t=0.3), e0 / (1 + 0.1 / 0.5))


# ---------------------------------------------------------------------------
# integration with the reference design-formula ZNN
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key", TI)
@pytest.mark.parametrize("disc", ["euler", "implicit_euler_err"])
def test_znn_smoke(key, disc):
    P = make_problem("linear_system", n=3, kappa=10, seed=2, T=0.2)
    rng = np.random.default_rng(0)
    x0 = P.x_star(0.0)[None, :] + 0.05 * rng.standard_normal((2, 3))
    s = simulate(P, make_model("znn", activation=key), make_discretization(disc), tau=1e-2,
                 B=2, gamma=2.0, x0=x0)
    assert not s["diverged"].any()
    assert np.all(np.isfinite(s["residual"]))
