"""Tests for noise/advanced.py and wrappers.py (area: noise_info)."""
import numpy as np
import pytest
from scipy.signal import savgol_coeffs

from znnbench import available, make_discretization, make_model, make_noise, make_problem, simulate
from znnbench.metrics import mean_ssre, mssre
from znnbench.wrappers import (CountingProblem, DelayedDataProblem, EstimatedDerivativeProblem,
                               FixedPoint, JitteredProblem, QuantizedDiscretizer,
                               alpha_beta_weights, dense_flops, kalman_cv_weights,
                               profile_step_cost, savgol_weights)


def P4(T=4.0):
    return make_problem("linear_system", n=4, kappa=10, seed=1, T=T)


def run(prob, disc="euler", tau=1e-2, gamma=10.0, **kw):
    d = make_discretization(disc) if isinstance(disc, str) else disc
    return simulate(prob, make_model("znn"), d, tau=tau, gamma=gamma, **kw)


# ===========================================================================
# noise models
# ===========================================================================
NEW_NOISES = ["uniform", "quadratic", "polynomial", "multi_harmonic", "combined", "random_walk",
              "impulse"]


def test_noises_registered():
    keys = available()["noises"]
    for k in NEW_NOISES:
        assert k in keys


@pytest.mark.parametrize("key", NEW_NOISES)
def test_noise_shape_and_znn_smoke(key):
    nz = make_noise(key)
    rng = np.random.default_rng(0)
    v = nz(0.5, rng, (3, 4))
    assert v.shape == (3, 4) and np.isfinite(v).all()
    s = run(P4(T=2.0), noise=make_noise(key))
    assert not s["diverged"].any() and np.isfinite(s["residual"]).all()


def test_uniform_bounds_and_moments():
    nz = make_noise("uniform", low=-0.5, high=0.5)
    v = nz(0.0, np.random.default_rng(1), (2000, 8))
    assert v.min() >= -0.5 and v.max() <= 0.5
    assert abs(v.mean()) < 4 * np.sqrt(1 / 12 / v.size)
    assert abs(v.var() - 1 / 12) < 0.05 / 12
    with pytest.raises(ValueError):
        make_noise("uniform", low=1.0, high=0.0)


def test_polynomial_and_quadratic_values():
    rng = np.random.default_rng(0)
    q = make_noise("quadratic", a=0.5, b=-2.0, c=3.0)
    for t in (0.0, 1.3, 7.0):
        assert np.allclose(q(t, rng, (2, 3)), 0.5 * t ** 2 - 2.0 * t + 3.0)
    coef = (1.0, np.array([0.0, 1.0, 2.0]), -0.25, 0.125)     # per-entry linear coefficient
    p = make_noise("polynomial", coeffs=coef)
    t = 1.7
    direct = 1.0 + np.array([0.0, 1.0, 2.0]) * t - 0.25 * t ** 2 + 0.125 * t ** 3
    assert np.allclose(p(t, rng, (2, 3)), np.broadcast_to(direct, (2, 3)))
    # defaults reproduce the real quadratic noise [t^2] of Liao et al. 2022
    assert np.allclose(make_noise("quadratic")(2.0, rng, (1, 2)), 4.0)


def test_multi_harmonic_matches_liao2022_periodic_example():
    nz = make_noise("multi_harmonic")
    rng = np.random.default_rng(0)
    for t in (0.0, 0.3, 1.25):
        ref = np.sin(2 * np.pi * t + 1) + 2 * np.sin(4 * np.pi * t + 2) + 5 * np.sin(10 * np.pi * t + 4)
        assert np.allclose(nz(t, rng, (2, 6)), ref)
    # periodic with period 1 s (frequencies 1, 2, 5 Hz)
    assert np.allclose(nz(0.37, rng, (1, 1)), nz(1.37, rng, (1, 1)))
    one = make_noise("multi_harmonic", amps=[1e3], freqs=[50.0], phases=[1.0])
    assert np.allclose(one(0.01, rng, (1, 1)), 1e3 * np.sin(100 * np.pi * 0.01 + 1))


def test_noises_reproduce_source_examples():
    """Every literature example quoted in docs/spec/noise_info.md is expressible
    with the registered keys (values checked against the original papers)."""
    rng = np.random.default_rng(0)
    ts = np.array([0.0, 0.37, 1.9, 7.3])
    # Liao, Wang, Li, Guo, He 2022 (Front. Neurorobot.), Sec. 5, aperiodic case
    ap = make_noise("multi_harmonic", amps=[1, 2, 5], freqs=[0.0, 2.5 / np.pi, 5.0], phases=[1, 2, 3])
    lo = make_noise("multi_harmonic", amps=[1e3], freqs=[0.001], phases=[1.0])
    for t in ts:
        assert np.allclose(ap(t, rng, (2, 3)), np.sin(1) + 2 * np.sin(5 * t + 2) + 5 * np.sin(10 * np.pi * t + 3))
        assert np.allclose(lo(t, rng, (2, 3)), 1e3 * np.sin(0.002 * np.pi * t + 1))
    # Yan et al. 2019 (IEEE Access): constant [8], linear [8t], random [7.5, 8.5]
    v = make_noise("uniform", low=7.5, high=8.5)(0.0, rng, (4000, 4))
    assert v.min() >= 7.5 and v.max() <= 8.5 and abs(v.mean() - 8.0) < 0.01
    assert np.allclose(make_noise("linear", slope=8.0)(1.5, rng, (2, 2)), 12.0)
    # Liao, Hua, Cao, Katsikis, Li 2022 (Mathematics), Sec. 4.3: real quadratic [t^2]
    assert np.allclose(make_noise("quadratic")(3.0, rng, (2, 2)), 9.0)
    # Yang, Wang, Voyles, Ma 2024 (arXiv:2412.20477): sin t + cos t + n(t), |n| <= 2
    yd = make_noise("combined", level=0, slope=0, amp=np.sqrt(2), omega=1, phase=np.pi / 4)
    yr = make_noise("combined", level=0, slope=0, amp=np.sqrt(2), omega=1, phase=np.pi / 4,
                    rand_bound=2.0)
    for t in ts:
        assert np.allclose(yd(t, rng, (1, 3)), np.sin(t) + np.cos(t))
        assert np.all(np.abs(yr(t, rng, (50, 3)) - (np.sin(t) + np.cos(t))) <= 2.0)


def test_combined_is_superposition():
    rng = np.random.default_rng(0)
    nz = make_noise("combined", level=2.0, slope=0.5, amp=3.0, omega=2.0, phase=0.1)
    t = 1.1
    assert np.allclose(nz(t, rng, (2, 2)), 2.0 + 0.5 * t + 3.0 * np.sin(2.0 * t + 0.1))
    nzr = make_noise("combined", level=0.0, slope=0.0, amp=0.0, rand_bound=0.2)
    v = nzr(t, rng, (500, 4))
    assert np.abs(v).max() <= 0.2 and v.std() > 0.05


def test_random_walk_statistics_and_reset():
    sigma = 0.7
    nz = make_noise("random_walk", sigma=sigma)
    rng = np.random.default_rng(3)
    shape = (4000, 2)
    ts = np.linspace(0.0, 2.0, 41)
    vals = np.array([nz(t, rng, shape) for t in ts])
    assert np.all(vals[0] == 0.0)
    # Var[W(t)] = sigma^2 t
    for i in (10, 40):
        assert abs(vals[i].var() / (sigma ** 2 * ts[i]) - 1) < 0.08
    # independent increments: corr(W(1)-W(0.5), W(0.5)) ~ 0
    inc = (vals[20] - vals[10]).ravel()
    assert abs(np.corrcoef(inc, vals[10].ravel())[0, 1]) < 0.05
    # holding the same t returns the same value; going back in time restarts
    a = nz(2.0, rng, shape)
    assert np.array_equal(a, vals[-1])
    # a round-off duplicate slightly earlier (RK4 stage time t_k + tau vs (k+1) tau)
    # must not restart the walk
    assert np.array_equal(nz(2.0 - 4e-16, rng, shape), vals[-1])
    assert np.all(nz(0.0, rng, shape) == 0.0)


def test_impulse_periodic_and_poisson():
    rng = np.random.default_rng(0)
    nz = make_noise("impulse", amp=5.0, period=1.0, phase=0.25, width=0.1, random_sign=False)
    assert np.all(nz(0.0, rng, (1, 3)) == 0.0)
    assert np.all(nz(0.30, rng, (1, 3)) == 5.0)
    assert np.all(nz(0.36, rng, (1, 3)) == 0.0)
    assert np.all(nz(1.27, rng, (1, 3)) == 5.0)
    # Poisson bursts: count arrivals over a long horizon on a fine grid
    rate, width, tau = 2.0, 0.02, 0.01
    pz = make_noise("impulse", amp=1.0, rate=rate, width=width)
    t_grid = np.arange(0.0, 500.0, tau)
    on = np.array([np.abs(pz(t, rng, (1, 1)))[0, 0] > 0 for t in t_grid])
    starts = np.count_nonzero(on[1:] & ~on[:-1])
    lam = rate * t_grid[-1]
    assert abs(starts - lam) < 6 * np.sqrt(lam)
    # random signs: amplitude is exactly +-amp (or a sum when bursts overlap)
    v = np.array([pz(t, rng, (1, 4)) for t in np.arange(0, 50, tau)]).ravel()
    assert set(np.unique(np.abs(v[v != 0]))) <= {1.0, 2.0, 3.0}


# ===========================================================================
# fixed-point quantization
# ===========================================================================
def test_fixed_point_grid_and_saturation():
    fp = FixedPoint(bits=12, frac_bits=8)
    q = 2.0 ** -8
    assert fp.q == q and fp.hi == (2 ** 11 - 1) * q and fp.lo == -(2 ** 11) * q
    v = np.random.default_rng(0).uniform(-10, 10, size=5000)
    y = fp(v)
    k = y / q
    assert np.array_equal(k, np.round(k))                     # on the 2^-f grid
    inside = (v > fp.lo) & (v < fp.hi)
    assert np.all(np.abs(y[inside] - v[inside]) <= q / 2 + 1e-15)
    assert np.all(y[v >= fp.hi] == fp.hi) and np.all(y[v <= fp.lo] == fp.lo)   # saturation
    assert np.array_equal(fp(y), y)                           # idempotent
    assert np.isnan(fp(np.array([np.nan])))[0]
    assert fp(np.array([np.inf]))[0] == fp.hi
    tr = FixedPoint(bits=12, frac_bits=8, rounding="floor")
    yt = tr(v[inside])
    assert np.all(yt <= v[inside]) and np.all(v[inside] - yt < q)
    # scale -> integer bits I = floor(log2 scale) + 1, so +-scale is representable
    # (a power of two needs one more integer bit than ceil(log2 scale))
    fs = FixedPoint(bits=16, scale=4.0)
    assert fs.frac_bits == 12 and fs(4.0) == 4.0 and fs(-4.0) == -4.0
    assert FixedPoint(bits=16, scale=3.0).frac_bits == 13
    for sc in (0.5, 0.7, 1.0, 3.0, 4.0, 1000.0):
        f = FixedPoint(bits=20, scale=sc)
        assert f.lo <= -sc and f.hi >= sc and f.hi < 2 * sc + 1e-12   # fits, no wasted bit
    with pytest.raises(ValueError):
        FixedPoint(bits=60, frac_bits=10)


def test_quantized_discretizer_state_on_grid():
    disc = QuantizedDiscretizer(make_discretization("euler"), bits=10, frac_bits=6,
                                quantize_derivative=True, deriv_bits=12, deriv_frac_bits=4)
    q, qd = 2.0 ** -6, 2.0 ** -4
    seen = []

    def f(t, x, z, k, implicit):
        return 3.3 * np.sin(x + t) + 0.1234, np.zeros((x.shape[0], 0))

    x = np.random.default_rng(0).uniform(-20, 20, size=(5, 3))
    z = np.zeros((5, 0))
    for k in range(20):
        def ff(t, xx, zz, kk, implicit):
            xd, zd = f(t, xx, zz, kk, implicit)
            seen.append(xx.copy())
            return xd, zd
        x, z = disc.step(k, k * 0.1, 0.1, x, z, ff)
        assert np.array_equal(x / q, np.round(x / q))
        assert x.max() <= disc.fmt.hi and x.min() >= disc.fmt.lo
    # the state handed to the model is on the grid (initial state quantized too)
    for xx in seen:
        assert np.array_equal(xx / q, np.round(xx / q))
    # derivative quantization: the inner discretizer receives grid values
    captured = []

    class Probe:
        name, causal, implicit = "probe", True, False

        def reset(self):
            pass

        def describe(self):
            return {"name": "probe"}

        def step(self, k, t, tau, x, z, f):
            xd, zd = f(t, x, z, k, False)
            captured.append(xd)
            return x + tau * xd, z

    dq = QuantizedDiscretizer(Probe(), bits=10, frac_bits=6, quantize_derivative=True,
                              deriv_bits=12, deriv_frac_bits=4)
    dq.step(0, 0.0, 0.1, np.ones((2, 2)), np.zeros((2, 0)), f)
    assert np.array_equal(captured[0] / qd, np.round(captured[0] / qd))
    assert disc.expected_order is None and disc.causal


def test_quantized_znn_euler_floor():
    P = P4()
    base = mssre(run(P))[0]
    floors = []
    for fb in (8, 12, 20):
        s = run(P, QuantizedDiscretizer(make_discretization("euler"), bits=32, frac_bits=fb))
        assert not s["diverged"].any()
        floors.append(mssre(s)[0])
    assert floors[0] > floors[1] > floors[2] * 0.999          # more bits -> lower floor
    assert floors[0] > 2 * base                               # coarse grid dominates
    assert abs(floors[2] / base - 1) < 0.02                   # fine grid -> unquantized
    # quantization floor of an O(tau^2) scheme: tiny tau no longer helps
    rk = QuantizedDiscretizer(make_discretization("rk4"), bits=32, frac_bits=8)
    assert mssre(run(P, rk, tau=1e-2, gamma=30.0))[0] > 1e-3
    # saturation (range [-1, 1) too small for x*) prevents convergence
    sat = run(P, QuantizedDiscretizer(make_discretization("euler"), bits=8, frac_bits=7))
    assert mssre(sat)[0] > 10 * base and not sat["diverged"].any()
    # derivative quantization also works end to end
    dq = run(P, QuantizedDiscretizer(make_discretization("euler"), bits=16, frac_bits=10,
                                     quantize_derivative=True))
    assert np.isfinite(mssre(dq)[0])


def test_quantization_floor_matches_rounding_noise_model():
    """Linear ZNN-Euler, E_{k+1} = (1-h) E_k + J delta_k, delta ~ U(-q/2, q/2):
    RMS floor ||J||_F q / sqrt(12 h (2-h)) when the rounding error is white
    (moderate h); for small h the state stalls between grid points and the
    floor is well above that white-noise value (dead zone)."""
    P = P4()
    tau, fb = 1e-3, 10
    q = 2.0 ** -fb
    JF = np.mean([np.linalg.norm(P.data(t)["A"]) for t in np.linspace(3.0, 4.0, 21)])
    out = {}
    for h in (0.5, 0.05):
        s = run(P, QuantizedDiscretizer(make_discretization("euler"), bits=40, frac_bits=fb),
                tau=tau, gamma=h / tau, B=4)
        out[h] = np.mean(mean_ssre(s)) / (JF * q / np.sqrt(12 * h * (2 - h)))
    assert 0.65 < out[0.5] < 1.35, out          # white rounding-noise model holds
    assert out[0.05] > 2.0, out                 # dead zone at small h


def test_quantized_discretizer_delegates_attributes():
    inner = make_discretization("taylor_zhang")
    qd = QuantizedDiscretizer(inner, bits=16, frac_bits=10)
    assert qd.source == inner.source and qd.s == inner.s
    assert qd.expected_order is None and qd.causal == inner.causal
    with pytest.raises(AttributeError):
        qd.no_such_attribute


def test_quantized_with_multistep_and_auxiliary_state():
    P = P4(T=2.0)
    s = run(P, QuantizedDiscretizer(make_discretization("taylor_zhang"), bits=32, frac_bits=24),
            tau=1e-2, gamma=20.0)
    ref = run(P, "taylor_zhang", tau=1e-2, gamma=20.0)
    assert abs(mssre(s)[0] / mssre(ref)[0] - 1) < 0.05
    # auxiliary state z is quantized too (and left alone with quantize_z=False)
    q = 2.0 ** -5

    def f(t, x, z, k, implicit):
        return -x, 0.37 * x[:, :2] + 0.011

    x0, z0 = np.full((3, 4), 1.3), np.full((3, 2), 0.123)
    for qz in (True, False):
        dz = QuantizedDiscretizer(make_discretization("euler"), bits=16, frac_bits=5, quantize_z=qz)
        x, z = x0, z0
        for k in range(5):
            x, z = dz.step(k, k * 0.1, 0.1, x, z, f)
        assert np.array_equal(z / q, np.round(z / q)) == qz


# ===========================================================================
# jitter
# ===========================================================================
def test_jitter_keeps_problem_api():
    P = P4()
    J = JitteredProblem(P, jitter_std=1e-2, seed=4)
    assert (J.n_x, J.m, J.T, J.family) == (P.n_x, P.m, P.T, P.family)
    assert J.capabilities() == P.capabilities()
    rng = np.random.default_rng(0)
    x = J.x0(rng, 3)
    for t in (0.0, 0.37, 2.5):
        d = J.data(t)
        assert "t" in d and float(d["t_nominal"]) == t
        ts = max(t + J.delta(t), 0.0)
        assert float(d["t"]) == pytest.approx(ts)
        ref = P.data(ts)
        assert np.allclose(d["A"], ref["A"]) and np.allclose(d["b"], ref["b"])
        assert J.E(x, d).shape == (3, P.m)
        assert np.allclose(J.dEdt(x, t), P.dEdt(x, ts))
        r = np.ones((3, P.m))
        assert np.allclose(d["A"] @ J.solve_J(x, d, r).T, r.T)
        assert np.allclose(J.JT(x, d, r), r @ d["A"])
        # recorded residual uses the exact data at the nominal instant
        assert np.allclose(J.residual(x, d), P.residual(x, P.data(t)))
    assert np.allclose(J.x_star(1.0), P.x_star(1.0))
    assert J.A is P.A                                          # attribute delegation


def test_jitter_statistics():
    P = P4()
    ts = np.arange(1, 6001) * 1e-3 + 1.0
    std, b = 2e-3, 3e-3
    g = JitteredProblem(P, jitter_std=std, seed=0)
    d = np.array([g.delta(t) for t in ts])
    n = len(d)
    assert abs(d.mean()) < 4 * std / np.sqrt(n)
    assert abs(d.std() / std - 1) < 0.05
    assert abs(np.corrcoef(d[1:], d[:-1])[0, 1]) < 0.06        # i.i.d. across steps
    u = np.array([JitteredProblem(P, jitter_bound=b, seed=1).delta(t) for t in ts])
    assert np.abs(u).max() <= b and abs(u.var() / (b ** 2 / 3) - 1) < 0.08
    tr = np.array([JitteredProblem(P, jitter_std=std, jitter_bound=b, seed=2).delta(t) for t in ts])
    assert np.abs(tr).max() <= b and tr.std() < std
    # reproducible, seed dependent, independent of query order
    g2 = JitteredProblem(P, jitter_std=std, seed=0)
    assert np.array_equal([g2.delta(t) for t in ts[::-1]], d[::-1])
    assert not np.allclose([JitteredProblem(P, jitter_std=std, seed=9).delta(t) for t in ts[:50]], d[:50])
    with pytest.raises(ValueError):
        JitteredProblem(P)
    # round-off duplicates of one instant (RK4 stage time vs next grid point)
    # share the same jitter
    assert g.delta(0.065 + 0.005) == g.delta(7 * 0.01)


def test_jitter_znn_smoke():
    P = P4()
    base = run(P)
    z = run(JitteredProblem(P, jitter_std=0.0))
    assert np.allclose(z["residual"], base["residual"])         # zero jitter = no-op
    small = mssre(run(JitteredProblem(P, jitter_std=1e-4)))[0]
    big = mssre(run(JitteredProblem(P, jitter_std=1e-2)))[0]
    b0 = mssre(base)[0]
    assert abs(small / b0 - 1) < 0.05 and big > 1.5 * b0
    # with backward-difference information the jitter is amplified by 1/tau
    bd = mssre(run(JitteredProblem(P, jitter_std=1e-3), info="bd1"))[0]
    an = mssre(run(JitteredProblem(P, jitter_std=1e-3), info="analytic"))[0]
    assert bd > an
    # non-causal RK4 stages also work through the wrapper
    rk = run(JitteredProblem(P, jitter_bound=1e-4), "rk4", gamma=30.0)
    assert not rk["diverged"].any()


# ===========================================================================
# derivative estimators
# ===========================================================================
@pytest.mark.parametrize("L,p", [(5, 2), (7, 3), (9, 2)])
def test_savgol_weights_match_scipy(L, p):
    tau = 0.01
    s = -tau * np.arange(L)[::-1]
    assert np.allclose(savgol_weights(s, p), savgol_coeffs(L, p, deriv=1, delta=tau, pos=L - 1, use="dot"))


def test_savgol_reduces_to_backward_differences():
    tau = 0.02
    assert np.allclose(savgol_weights([-tau, 0.0], 5), np.array([-1.0, 1.0]) / tau)
    assert np.allclose(savgol_weights([-2 * tau, -tau, 0.0], 2), np.array([1.0, -4.0, 3.0]) / (2 * tau))


def test_filters_exact_on_polynomials_and_lag_formula():
    tau, t0 = 0.01, 2.0
    s = -tau * np.arange(30)[::-1]
    lin = 0.3 + 1.7 * (t0 + s)
    quad = 0.5 * 2.0 * (t0 + s) ** 2                           # second derivative a = 2
    assert savgol_weights(s[-5:], 2) @ quad[-5:] == pytest.approx(2.0 * t0, rel=1e-9)
    a, b = 0.5, 0.5 ** 2 / 1.5
    assert alpha_beta_weights(s, a, b) @ lin == pytest.approx(1.7, rel=1e-9)
    assert kalman_cv_weights(s, 1e4, 1.0) @ lin == pytest.approx(1.7, rel=1e-9)
    # steady-state lag of an alpha-beta tracker on constant acceleration
    s_long = -tau * np.arange(200)[::-1]
    quad_l = (t0 + s_long) ** 2
    est = alpha_beta_weights(s_long, a, b) @ quad_l
    assert est - 2 * t0 == pytest.approx(-2.0 * tau * (a - b / 2) / b, rel=1e-6)
    # Kalman steady state for tracking index 1: alpha = 0.75, beta = 0.5 -> lag a*tau
    est_k = kalman_cv_weights(s_long, (1.0 / tau ** 2) ** 2, 1.0) @ quad_l
    assert est_k - 2 * t0 == pytest.approx(-2.0 * tau * (0.75 - 0.25) / 0.5, rel=1e-6)
    # general tracking index: Kalata (1984) steady-state gains of the DWNA filter
    lam = 0.3
    rt = np.sqrt(lam ** 2 + 8 * lam)
    ak = -(lam ** 2 + 8 * lam - (lam + 4) * rt) / 8
    bk = (lam ** 2 + 4 * lam - lam * rt) / 4
    est_k = kalman_cv_weights(s_long, (lam / tau ** 2) ** 2, 1.0) @ quad_l
    est_ab = alpha_beta_weights(s_long, ak, bk) @ quad_l
    assert est_k == pytest.approx(est_ab, rel=1e-6)
    assert est_k - 2 * t0 == pytest.approx(-2.0 * tau * (ak - bk / 2) / bk, rel=1e-5)


def test_filter_weights_match_independent_recursions():
    """FIR weights reproduce a textbook alpha-beta tracker and a textbook
    DWNA Kalman filter (x = Fx, P = FPF' + qGG', K = PH'/(HPH'+r)) run on
    noisy data over a non-uniform time grid."""
    rng = np.random.default_rng(7)
    t = np.sort(rng.uniform(0.0, 1.0, 30))
    e = np.sin(3 * t) + 0.01 * rng.standard_normal(30)
    s = t - t[-1]

    eh, vh = e[1], (e[1] - e[0]) / (t[1] - t[0])
    xk = np.array([e[1], (e[1] - e[0]) / (t[1] - t[0])])
    q, r, a, b = 3.0, 0.01, 0.4, 0.4 ** 2 / 1.6
    dt0 = t[1] - t[0]
    Pk = np.array([[r, r / dt0], [r / dt0, 2 * r / dt0 ** 2]])
    H = np.array([1.0, 0.0])
    for j in range(2, len(t)):
        dt = t[j] - t[j - 1]
        ep = eh + dt * vh
        eh, vh = ep + a * (e[j] - ep), vh + b / dt * (e[j] - ep)
        F, G = np.array([[1.0, dt], [0.0, 1.0]]), np.array([dt ** 2 / 2, dt])
        xk, Pk = F @ xk, F @ Pk @ F.T + q * np.outer(G, G)
        K = Pk @ H / (H @ Pk @ H + r)
        xk, Pk = xk + K * (e[j] - H @ xk), (np.eye(2) - np.outer(K, H)) @ Pk
    assert alpha_beta_weights(s, a, b) @ e == pytest.approx(vh, rel=1e-10)
    assert kalman_cv_weights(s, q, r) @ e == pytest.approx(xk[1], rel=1e-10)
    # savgol on the non-uniform grid is exact for polynomials of degree <= order
    for p in (1, 2, 3):
        c = rng.standard_normal(p + 1)
        assert savgol_weights(s[-8:], p) @ np.polyval(c, s[-8:]) == pytest.approx(c[-2], rel=1e-8)


class NoDerivative:
    """Problem proxy whose exact dE/dt must never be used."""

    def __new__(cls, P):
        class _P(type(P)):
            def dEdt(self, x, t):
                raise AssertionError("exact dEdt used")
        obj = _P.__new__(_P)
        obj.__dict__.update(P.__dict__)
        return obj


@pytest.mark.parametrize("method,order", [("savgol", 2), ("alpha_beta", 1), ("kalman", 1)])
def test_estimators_converge_to_analytic_dEdt(method, order):
    P = P4()
    x = np.random.default_rng(5).standard_normal((2, 4))
    t_query = 1.3
    errs = []
    for tau in (1e-2, 1e-3):
        W = EstimatedDerivativeProblem(NoDerivative(P), method=method)
        k_end = int(round(t_query / tau))
        for k in range(k_end + 1):
            W.data(k * tau)
        est = W.dEdt(x, k_end * tau)
        ex = P.dEdt(x, k_end * tau)
        errs.append(np.abs(est - ex).max() / np.abs(ex).max())
    rate = np.log10(errs[0] / errs[1])
    assert abs(rate - order) < 0.25, (errs, rate)
    assert errs[1] < (1e-4 if order == 2 else 1e-2)


def test_estimator_is_causal_weighted_error_differencing():
    P = P4()
    tau = 1e-2
    W = EstimatedDerivativeProblem(P, method="savgol", window=5, order=2)
    for k in range(12):
        W.data(k * tau)
    x = np.random.default_rng(1).standard_normal((3, 4))
    t8 = 8 * tau
    # query at an earlier instant only uses samples t_j <= t (here t_4..t_8 are no
    # longer all in the window: only samples 7..11 are stored -> 7, 8 usable)
    w = savgol_weights(np.array([7, 8]) * tau - t8, 2)
    manual = w[0] * P.E(x, P.data(7 * tau)) + w[1] * P.E(x, P.data(8 * tau))
    assert np.allclose(W.dEdt(x, t8), manual)
    # at the latest instant: window samples 7..11
    w5 = savgol_weights((np.arange(7, 12) - 11) * tau, 2)
    manual = sum(wj * P.E(x, P.data(j * tau)) for wj, j in zip(w5, range(7, 12)))
    assert np.allclose(W.dEdt(x, 11 * tau), manual)
    # fewer than two samples -> no information
    W2 = EstimatedDerivativeProblem(P, method="kalman")
    W2.data(0.0)
    assert np.all(W2.dEdt(x, 0.0) == 0.0)
    # a new run (time goes back) resets the history; an out-of-order sample
    # only discards the stored samples later than it
    W.data(0.0)
    assert len(W._times) == 1
    for k in range(1, 6):
        W.data(k * tau)                        # window 5 -> samples 1..5 kept
    W.data(3.5 * tau)
    assert list(W._times) == pytest.approx([tau, 2 * tau, 3 * tau, 3.5 * tau])


def test_estimator_noise_smoothing():
    """With measurement noise (sigma/tau large) the smoothing estimators beat
    the two-point backward difference by a wide margin."""
    P = P4()
    tau = 1e-3
    x = np.random.default_rng(2).standard_normal((1, 4))
    errs = {}
    for name, kw in {"bd1": dict(method="savgol", window=2, order=1),
                     "sg21": dict(method="savgol", window=21, order=2),
                     "ab": dict(method="alpha_beta", alpha=0.2, window=60),
                     "kf": dict(method="kalman", tracking_index=0.05, window=60)}.items():
        e = []
        for seed in range(10):
            W = EstimatedDerivativeProblem(P, meas_std=1e-3, seed=seed, **kw)
            for k in range(101):
                W.data(1.0 + k * tau)
            e.append(np.abs(W.dEdt(x, 1.0 + 100 * tau) - P.dEdt(x, 1.0 + 100 * tau)).mean())
        errs[name] = np.mean(e)
    for k in ("sg21", "ab", "kf"):
        assert errs[k] < errs["bd1"] / 3, errs
    # measurement noise must not leak into the recorded residual
    W = EstimatedDerivativeProblem(P, meas_std=1e-1)
    d = W.data(0.5)
    assert np.allclose(W.residual(x, d), P.residual(x, P.data(0.5)))
    assert not np.allclose(d["A"], P.data(0.5)["A"])


@pytest.mark.parametrize("method", ["savgol", "alpha_beta", "kalman"])
def test_estimated_derivative_znn_euler_smoke(method):
    P = P4(T=3.0)
    out = []
    for tau in (1e-2, 2e-3):
        s = run(EstimatedDerivativeProblem(NoDerivative(P), method=method), tau=tau, gamma=0.3 / tau)
        assert not s["diverged"].any()
        out.append(mssre(s)[0])
    assert out[1] < out[0] / 4 and out[1] < 5e-4, out
    # RK4 stage calls (non-uniform data times) are handled
    rk = run(EstimatedDerivativeProblem(P, method=method), "rk4", tau=1e-2, gamma=30.0)
    assert not rk["diverged"].any()


# ===========================================================================
# delay, counting
# ===========================================================================
def test_delayed_data():
    P = P4()
    tau = 1e-2
    D = DelayedDataProblem(P, delay_steps=3, tau=tau)
    x = np.ones((2, 4))
    d = D.data(0.5)
    assert np.allclose(d["A"], P.data(0.5 - 3 * tau)["A"]) and float(d["t_nominal"]) == 0.5
    assert np.allclose(D.dEdt(x, 0.5), P.dEdt(x, 0.5 - 3 * tau))
    assert np.allclose(D.data(0.01)["A"], P.data(0.0)["A"])      # held before arrival
    assert np.all(D.dEdt(x, 0.01) == 0.0)
    assert np.allclose(D.residual(x, d), P.residual(x, P.data(0.5)))
    base = mssre(run(P))[0]
    dl = run(D)
    assert not dl["diverged"].any() and mssre(dl)[0] > 3 * base
    assert np.allclose(run(DelayedDataProblem(P, delay=0.0))["residual"], run(P)["residual"])


def test_wrapper_chains():
    """Chained wrappers: the recorded residual never touches the state of a
    stateful inner wrapper, time stamps are never perturbed, and the inner
    data time 't' is preserved."""
    P = P4(T=3.0)
    x = np.random.default_rng(0).standard_normal((2, 4))
    # estimator with measurement noise around a delayed problem: exact residual
    W = EstimatedDerivativeProblem(DelayedDataProblem(P, delay=0.05), meas_std=1e-2, seed=3)
    for t in (0.5, 1.0):
        d = W.data(t)
        assert float(d["t_nominal"]) == t and float(d["t"]) == pytest.approx(t - 0.05)
        assert np.allclose(W.residual(x, d), P.residual(x, P.data(t)))
    # jitter around delay: 't' is the time the data belong to
    J = JitteredProblem(DelayedDataProblem(P, delay=0.05), jitter_std=1e-3, seed=1)
    d = J.data(1.0)
    assert float(d["t"]) == pytest.approx(1.0 + J.delta(1.0) - 0.05)
    assert np.allclose(d["A"], P.data(1.0 + J.delta(1.0) - 0.05)["A"])
    assert np.allclose(J.residual(x, d), P.residual(x, P.data(1.0)))
    # delay outside / inside the estimator: same run after the initial hold
    tau = 2e-3
    E_in = EstimatedDerivativeProblem(P, method="savgol")
    a = run(EstimatedDerivativeProblem(DelayedDataProblem(P, delay=0.02), method="savgol"),
            tau=tau, gamma=0.3 / tau)
    b = run(DelayedDataProblem(E_in, delay=0.02), tau=tau, gamma=0.3 / tau)
    assert np.allclose(mssre(a), mssre(b), rtol=1e-8)
    assert len(E_in._times) == E_in.window            # history not reset by residual calls
    # time-stamped jitter (estimator inside) runs without losing its history
    E_in2 = EstimatedDerivativeProblem(P, method="savgol")
    s = run(JitteredProblem(E_in2, jitter_std=2e-4), tau=tau, gamma=0.3 / tau)
    assert not s["diverged"].any() and len(E_in2._times) >= 2
    # exact_data bypasses every wrapper's data() (no counted calls, no state)
    C = CountingProblem(P)
    Jc = JitteredProblem(C, jitter_std=1e-3)
    dj = Jc.data(0.4)
    n_data = C.calls["data"]
    Jc.residual(x, dj)
    assert C.calls["data"] == n_data and C.calls["residual"] == 1


def test_counting_and_profile():
    P = P4()
    prof = profile_step_cost(P, make_model("znn"), make_discretization("euler"), steps=50)
    c = prof["calls_per_step"]
    assert c["E"] == 1 and c["dEdt"] == 1 and c["solve_J"] == 1 and c["JT"] == 0
    assert prof["wall_per_step"] > 0 and prof["flops_per_step_dense"] > 0
    prof4 = profile_step_cost(P, make_model("znn"), make_discretization("rk4"), steps=50)
    assert prof4["calls_per_step"]["solve_J"] == 4
    pg = profile_step_cost(P, make_model("gnn"), make_discretization("euler"), steps=20, gamma=1.0)
    assert pg["calls_per_step"]["JT"] == 1 and pg["calls_per_step"]["solve_J"] == 0
    fl = dense_flops(4, 4, B=2)
    assert fl["E"] == 2 * 4 * 4 * 2
    C = CountingProblem(P)
    assert np.allclose(run(C)["residual"], run(P)["residual"])
