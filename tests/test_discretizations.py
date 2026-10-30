"""Tests for the discretization zoo (zead.py, uhlig.py, adams.py).

Checks: (1) exact transcription / consistency conditions, (2) zero-stability
and the strong root condition, (3) linear-ZNN step-size intervals against the
values stated in the literature, (4) empirical residual order of linear ZNN on
`linear_system` at fixed h = gamma*tau, (5) causality, (6) the instant-number
precision barrier (Yang, Zhang & Hu, Numer. Algorithms 88 (2021) 883-902).
All simulations are n = 4, <= 5e3 steps each; the file runs in ~15 s.
"""
from fractions import Fraction as F
from math import factorial

import numpy as np
import pytest

from znnbench import make_discretization, make_model, make_problem, simulate
from znnbench.discretizations import adams as AB
from znnbench.discretizations import uhlig as U
from znnbench.discretizations import zead as Z
from znnbench.metrics import empirical_order, mssre

LOOKAHEAD_KEYS = {**Z.ZEAD_KEYS, **U.UHLIG_KEYS}
ALL_KEYS = {**LOOKAHEAD_KEYS, **AB.AB_KEYS}


# ---------------------------------------------------------------------------
# registry / metadata
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key,order", sorted(ALL_KEYS.items()))
def test_registry_and_metadata(key, order):
    d = make_discretization(key)
    assert d.causal is True
    assert d.expected_order == order
    desc = d.describe()
    assert desc["expected_order"] == order and desc["causal"] is True
    assert desc["h_max"] > 0 and 0 < desc["h_opt"] < desc["h_max"] and desc["rho_opt"] < 1
    assert desc.get("doi") and desc.get("source")


# ---------------------------------------------------------------------------
# consistency conditions (exact for rational transcriptions)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key", sorted(LOOKAHEAD_KEYS))
def test_consistency_conditions(key):
    d = make_discretization(key)
    p = d.claimed_trunc_order
    mu = Z.taylor_moments(d.coeffs, p + 2)
    if d.exact:
        assert sum(d.coeffs_exact) == 0                  # rho(1) = 0
        assert mu[1] == 1                                # rho'(1) = 1 (derivative normalisation)
        assert all(m == 0 for m in mu[2:p + 1])          # orders 2..p
        assert mu[p + 1] != 0                            # not of higher order
    else:                                                # Uhlig's 16-digit floats
        assert abs(float(mu[0])) < 1e-14 and abs(float(mu[1]) - 1) < 1e-13
        assert max(abs(float(m)) for m in mu[2:p + 1]) < 1e-12
        assert abs(float(mu[p + 1])) > 1e-3
    assert d.trunc_order() == p
    assert d.expected_order == p + 1


def test_printed_integer_vectors():
    expect = {
        "zead_5i": ([8, 1, -6, -5, 2], 18),
        "uhlig_2_3": ([8, 1, -6, -5, 2], 18),
        "zead_4i_6321": ([6, -3, -2, -1], 10),
        "zead_4i_fifd": ([5, -3, -1, -1], 8),
        "zead_4i_general": ([3, -3, 1, -1], 4),
        "zead_6i_guo2017": ([24, -5, -12, -6, -4, 3], 48),
        "zead_6i_qiu2019": ([13, -6, -2, -4, -3, 2], 24),
        "zead_7i_general": ([300, 161, -300, -300, 100, 75, -36], 780),
        "zead_8i_wu2020": ([1000, -51, -400, -600, -200, 175, 176, -100], 2220),
        "uhlig_3_4": ([237, 80, -182, -206, 1, 110, -40], 588),
    }
    for key, (nums, den) in expect.items():
        d = make_discretization(key)
        assert Z.integer_form(d.coeffs_exact) == (nums, den), key
    # Taylor-Zhang (multistep.py): (2x_{k+1} - 3x_k + 2x_{k-1} - x_{k-2}) / (2 tau)
    tz = make_discretization("taylor_zhang")
    tz_c = [F(tz.c_next)] + [F(c) for c in tz.c_past]
    assert Z.integer_form(tz_c) == ([2, -3, 2, -1], 2)
    # general families reproduce the named members
    assert Z.coeffs_4i_general(F(-1, 2)) == tz_c
    assert Z.coeffs_4i_general(F(-1, 10)) == list(make_discretization("zead_4i_6321").coeffs_exact)
    assert Z.coeffs_5i_general(F(1, 9)) == Z.COEFFS_5I
    # Chen-Kang-Zhang 2023, Appendix D: normalised characteristic polynomial P_10
    c = Z.COEFFS_11I_CHEN2023
    P10 = [F(1), F(90583, 529200), F(-3, 5), F(-3, 5), F(-6, 25), F(0), F(9, 25), F(2, 15),
           F(-69, 245), F(11, 400), F(4, 135)]
    assert [x / c[0] for x in c] == P10
    # Yang et al. TNNLS 2020, proof of Theorem 2: characteristic equation of (8)
    for p1, p2 in [(-1, -1), (F(-11, 10), -1), (F(-9, 10), F(-19, 20))]:
        p1, p2 = F(p1), F(p2)
        c7 = Z.coeffs_7i_general(p1, p2)
        ch = [F(1), -(214 * p1 + 77 * p2 + 130) / 300, p1, p2, -(2 * p1 + 4 * p2 + 5) / 3,
              (2 * p1 + 3 * p2 + 6) / 4, -(3 * p1 + 4 * p2 + 10) / 25]
        assert [x / c7[0] for x in c7] == ch


def test_uhlig_seed_construction_reproduces_published_runs():
    # [U19] p.7: runconv1step(1,1,2,2), seed y = [-5, 2]
    p, c = U.uhlig_seed_polynomial(2, 2, [-5, 2])
    assert p == [1, F(1, 8), F(-3, 4), F(-5, 8), F(1, 4)] and c == F(9, 4)
    # [U19] p.7: runconv1step(1,1,3,3), seed y = [1, 110, -40]
    p, c = U.uhlig_seed_polynomial(3, 3, [1, 110, -40])
    assert p == [1, F(80, 237), F(-182, 237), F(-206, 237), F(1, 237), F(110, 237), F(-40, 237)]
    assert c == F(196, 79)
    # TPOLY also prints the second-largest root modulus: 0.9025 and 446/465
    assert abs(make_discretization("uhlig_2_3").root_info()["second_largest_modulus"] - 0.9025) < 1e-4
    assert abs(make_discretization("uhlig_3_4").root_info()["second_largest_modulus"] - 446 / 465) < 1e-4
    # [UF] p.6: roots of 2_2b and of 4_5a
    def match(roots, ref):
        ref = np.array(ref)
        return all(np.min(np.abs(roots - r)) < 2e-5 for r in ref)
    assert match(make_discretization("zead_5i").root_info()["roots"],
                 [-0.71597 + 0.54945j, -0.71597 - 0.54945j, 1.0, 0.30693])
    assert match(make_discretization("uhlig_4_6").root_info()["roots"],
                 [-0.65551 + 0.62479j, -0.65551 - 0.62479j, -0.70266 + 0.34906j, -0.70266 - 0.34906j,
                  1.0, 0.43671 + 0.32838j, 0.43671 - 0.32838j, 0.47141, -0.29289])
    # the published seed family [a, 110, -40] is convergent for 1 <= a <= 2.5 only
    for a in (1, F(3, 2), 2, F(5, 2)):
        assert make_discretization("uhlig_3_4", a=a).root_info()["strong"]
    with pytest.raises(ValueError):
        make_discretization("uhlig_3_4", a=3)
    p3, _ = U.uhlig_seed_polynomial(3, 3, [3, 110, -40])
    assert not Z.root_condition([float(x) for x in p3])["zero_stable"]


# ---------------------------------------------------------------------------
# zero-stability
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key", sorted(ALL_KEYS))
def test_zero_stability(key):
    d = make_discretization(key)
    ri = d.root_info()
    assert ri["has_unit_root"]                       # consistency: rho(1) = 0
    assert ri["zero_stable"]                         # roots in closed disk, unit roots simple
    assert ri["strong"]                              # z = 1 is the only unit root
    r = ri["roots"]
    assert np.sum(np.abs(r - 1.0) < 1e-6) == 1       # z = 1 simple


def test_published_root_lists_and_a_misprint():
    # Chen-Kang-Zhang 2023 Appendix D roots of P_10
    r = make_discretization("zead_11i_chen2023").root_info()["roots"]
    for ref in [1.0, 0.526013, -0.275019, -0.846791, 0.509803 + 0.310017j, -0.726573 + 0.572858j,
                -0.070916 + 0.888032j]:
        assert np.min(np.abs(r - ref)) < 2e-6
    # Guo, Nie, Yan (TSMC 2017) Theorem 5 prints the characteristic polynomial
    # with flipped signs; its listed roots (0.3596, -0.5211+-0.4694i,
    # 0.2372+-0.8065i) belong to the misprinted polynomial, which lacks z = 1.
    printed = np.roots([1, 5 / 24, 1 / 2, 1 / 4, 1 / 6, -1 / 8])
    for ref in [0.3596, -0.5211 + 0.4694j, 0.2372 + 0.8065j]:
        assert np.min(np.abs(printed - ref)) < 1e-4
    assert np.min(np.abs(printed - 1)) > 0.5
    # ... while the correct polynomial of rule (6) is zero-stable (our root check)
    assert make_discretization("zead_6i_guo2017").root_info()["strong"]


def test_negative_controls():
    # symmetric Euler (x_{k+1} - x_{k-1})/(2 tau): zero-stable, but the unit root
    # -1 makes every linear-ZNN step size unstable (Sun, Tian, Wang 2019, Thm 3.3)
    se = [F(1, 2), F(0), F(-1, 2)]
    assert Z.truncation_order(se) == 2
    ri = Z.root_condition(Z.lookahead_char_poly(se))
    assert ri["zero_stable"] and not ri["strong"]
    assert Z.lookahead_stability(se)["h_max"] == 0.0
    # Sun-Tian-Wang (2019, eq. (24)): the O(tau^5) six-instant formula is not
    # zero-stable (root -6.9614)
    stw = [F(1, 5), F(13, 12), F(-2), F(1), F(-1, 3), F(1, 20)]
    assert Z.truncation_order(stw) == 5
    ri = Z.root_condition(Z.lookahead_char_poly(stw))
    assert not ri["zero_stable"]
    assert np.min(np.abs(ri["roots"] + 6.9614)) < 1e-4


def _family(N, p):
    offs = np.array([1 - i for i in range(N)], float)
    V = np.array([[o ** m / factorial(m) for o in offs] for m in range(p + 1)])
    rhs = np.zeros(p + 1)
    rhs[1] = 1.0
    c0 = np.linalg.lstsq(V, rhs, rcond=None)[0]
    null = np.linalg.svd(V)[2][p + 1:].T
    return c0, null


def _n_strong(c0, null, grid):
    n = 0
    for t in grid:
        c = c0 + null @ np.atleast_1d(t)
        if abs(c[0]) < 1e-3 * np.max(np.abs(c)):
            continue                                  # degenerates to fewer instants
        n += Z.root_condition(c)["strong"]
    return n


def test_precision_barrier_yang_zhang_hu_2021():
    """Numerical evidence for the abstract of Yang, Zhang & Hu (2021): N-instant
    ZeaD formulas reach at most O(tau) (N=2,3), O(tau^2) (N=4), O(tau^3)
    (N=5,6), O(tau^4) (N=7,8), O(tau^5) (N=9)."""
    for N in range(3, 10):                            # unique max-order formula
        c0, _ = _family(N, N - 1)
        assert not Z.root_condition(c0)["strong"], N
    t1 = np.linspace(-20, 20, 4001)
    g2 = [np.array([a, b]) for a in np.linspace(-5, 5, 41) for b in np.linspace(-5, 5, 41)]
    assert _n_strong(*_family(6, 4), t1) == 0         # 6 instants: no quartic precision
    assert _n_strong(*_family(7, 5), t1) == 0         # 7 instants: no quintic precision
    assert _n_strong(*_family(8, 5), g2) == 0         # 8 instants: no quintic precision (grid)
    assert _n_strong(*_family(9, 6), g2) == 0         # 9 instants: no sextic precision (grid)
    assert _n_strong(*_family(5, 3), t1) > 0          # ... while these are attainable
    assert _n_strong(*_family(7, 4), g2) > 0


def test_family_effective_domains():
    # general 4-instant: a1 in (-inf, 0)  (Hu, Kang, Zhang 2018)
    for a1 in (F(-3), F(-1), F(-1, 2), F(-1, 8), F(-1, 100)):
        assert Z.root_condition(Z.coeffs_4i_general(a1))["strong"]
    assert not Z.root_condition(Z.coeffs_4i_general(F(1, 100)))["zero_stable"]
    with pytest.raises(ValueError):
        make_discretization("zead_4i_general", a1=F(1, 10))
    # general 5-instant: a1 in (1/12, 1/6)  (Zhang et al. JCAM 2019)
    for a1 in (F(1, 11), F(1, 10), F(1, 9), F(1, 8), F(1, 7)):
        assert Z.root_condition(Z.coeffs_5i_general(a1))["strong"]
    for a1 in (F(1, 13), F(2, 11)):
        assert not Z.root_condition(Z.coeffs_5i_general(a1))["zero_stable"]
    # general 7-instant: domain (16) of Yang et al. 2020
    for p1, p2 in [(-1, -1), (-1.1, -1.0), (-0.9, -0.95), (-0.8, -1.0)]:
        assert Z.in_7i_domain(p1, p2)
        assert Z.root_condition(Z.coeffs_7i_general(F(p1).limit_denominator(100),
                                                    F(p2).limit_denominator(100)))["strong"]
    for p1, p2 in [(-1, 0), (-2, -1), (0.5, -1), (-1, -1.5)]:
        assert not Z.in_7i_domain(p1, p2)
        assert not Z.root_condition(Z.coeffs_7i_general(F(p1), F(p2)))["strong"]


# ---------------------------------------------------------------------------
# linear-ZNN step-size intervals vs. the literature
# ---------------------------------------------------------------------------
def test_stepsize_ranges_against_literature():
    tz = make_discretization("taylor_zhang")
    st = Z.lookahead_stability([tz.c_next, *tz.c_past])
    # Zhang, Gong, Yang, Li, Yang, IEEE TNNLS 30(3) 2019: range (0,1), h_o = 0.2400, M = 0.7938
    assert abs(st["h_max"] - 1.0) < 1e-6
    assert abs(st["h_opt"] - 0.2400) < 1e-3 and abs(st["rho_opt"] - 0.7938) < 1e-3
    # Euler: Jin & Zhang, IEEE TNNLS 26(7) 2015: h in (0,2), h = 1 fastest
    eu = make_discretization("euler")
    st = Z.lookahead_stability([eu.c_next, *eu.c_past])
    assert abs(st["h_max"] - 2.0) < 1e-6 and abs(st["h_opt"] - 1.0) < 1e-4
    # ZeaD 6321: Zhang, Qi, Li, Qiu, Yang, Numer. Algorithms 81 (2019): h in (0, 0.8)
    assert abs(make_discretization("zead_4i_6321").stability()["h_max"] - 0.8) < 1e-6
    # Adams-Bashforth real stability intervals (0,1), (0,6/11), (0,3/10)
    for key, hm in (("ab2", 1.0), ("ab3", 6 / 11), ("ab4", 0.3)):
        assert abs(make_discretization(key).stability()["h_max"] - hm) < 1e-6


@pytest.mark.parametrize("key", ["taylor_zhang", "zead_4i_6321", "zead_6i_qiu2019", "ab3"])
def test_stability_boundary_in_simulation(key):
    d = make_discretization(key)
    coeffs = [d.c_next, *d.c_past] if key == "taylor_zhang" else None
    st = Z.lookahead_stability(coeffs) if coeffs else d.stability()
    poly = (lambda h: Z.lookahead_znn_poly(coeffs, h)) if coeffs else d.znn_poly
    h_bad = 1.25 * st["h_max"]
    assert Z.spectral_radius(poly(h_bad)) > 1.005           # makes the check meaningful
    P = make_problem("linear_system", n=4, kappa=10, seed=1, T=2.0)
    res = {}
    for name, h in (("good", 0.9 * st["h_max"]), ("bad", h_bad)):
        s = simulate(P, make_model("znn"), make_discretization(key), tau=1e-3, gamma=h / 1e-3, seed=0,
                     record_solution_error=False)
        res[name] = s
    assert not res["good"]["diverged"].any() and mssre(res["good"])[0] < 1e-6
    assert res["bad"]["diverged"].all() or mssre(res["bad"])[0] > 1e3 * mssre(res["good"])[0]


# ---------------------------------------------------------------------------
# empirical residual order on linear_system with linear ZNN at fixed h
# ---------------------------------------------------------------------------
G3 = [0.04, 0.02, 0.01, 0.005]
G4 = [0.02, 0.01, 0.005, 0.0025]
G5 = [0.012, 0.0085, 0.006, 0.0042, 0.003]
GU = [0.008, 0.0057, 0.004, 0.0028, 0.002]
# key -> (factory kwargs, tau grid, horizon T).  h = h_opt of the formula.
ORDER_CASES = {
    "zead_4i_general": ({}, G3, 6.0), "zead_4i_6321": ({}, G3, 6.0), "zead_4i_fifd": ({}, G3, 6.0),
    "ab2": ({}, G3, 6.0),
    "zead_5i": ({}, G4, 6.0), "zead_5i_general": ({"a1": F(1, 10)}, G4, 6.0),
    "zead_6i_guo2017": ({}, G4, 6.0), "zead_6i_qiu2019": ({}, G4, 6.0), "uhlig_2_3": ({}, G4, 6.0),
    "ab3": ({}, G4, 6.0),
    "zead_7i_general": ({}, G5, 6.0), "zead_8i_wu2020": ({}, G5, 6.0), "zead_8i_chen2019": ({}, G5, 6.0),
    "ab4": ({}, G4, 6.0), "uhlig_3_4": ({}, GU, 10.0),
    "zead_10i_wu2020": ({}, G5, 6.0), "zead_11i_chen2023": ({}, G5, 6.0), "uhlig_4_6": ({}, G5, 6.0),
}


@pytest.mark.parametrize("key", sorted(ORDER_CASES))
def test_empirical_order_fixed_h(key):
    kw, taus, T = ORDER_CASES[key]
    d = make_discretization(key, **kw)
    h = d.stability()["h_opt"]
    P = make_problem("linear_system", n=4, kappa=10, seed=1, T=T)
    x0 = P.x_star(0.0)[None]                 # start on x*(0); Euler start-up as in the sources
    errs = []
    for tau in taus:
        s = simulate(P, make_model("znn"), make_discretization(key, **kw), tau=tau, gamma=h / tau,
                     x0=x0, seed=0, record_solution_error=False)
        errs.append(float(mssre(s)[0]))
    slope, n = empirical_order(taus, errs, floor=1e-11)
    assert n >= 4, errs
    assert abs(slope - d.expected_order) < 0.35, (key, slope, errs)


def test_ten_instant_rows_all_sixth_order():
    """All six coefficient rows of Wu et al. (2020) Table 1 are O(tau^5)
    formulas (residual O(tau^6)); rows 2-6 checked exactly, row 1 empirically
    above.  Row 4 also checked empirically."""
    for row in range(1, 7):
        d = make_discretization("zead_10i_wu2020", row=row)
        assert d.trunc_order() == 5 and d.root_info()["strong"]
    kw = {"row": 4}
    d = make_discretization("zead_10i_wu2020", **kw)
    h = d.stability()["h_opt"]
    P = make_problem("linear_system", n=4, kappa=10, seed=1, T=6.0)
    x0 = P.x_star(0.0)[None]
    errs = [float(mssre(simulate(P, make_model("znn"), make_discretization("zead_10i_wu2020", **kw),
                                 tau=t, gamma=h / t, x0=x0, record_solution_error=False))[0]) for t in G5]
    assert abs(empirical_order(G5, errs, floor=1e-11)[0] - 6) < 0.35, errs


# ---------------------------------------------------------------------------
# causality and Adams-Bashforth specifics
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key", sorted(ALL_KEYS))
def test_causal_evaluation_only(key):
    d = make_discretization(key)
    d.reset()
    calls = []
    lam = -2.0

    def f(t, x, z, k, implicit):
        calls.append((t, k))
        return lam * x + np.cos(t), np.zeros((x.shape[0], 0))

    tau = 0.01
    x = np.ones((3, 2))
    z = np.zeros((3, 0))
    for k in range(15):
        x, z = d.step(k, k * tau, tau, x, z, f)
    assert len(calls) == 15                                  # one evaluation per step
    assert all(kk is not None and abs(t - kk * tau) < 1e-15 for t, kk in calls)
    assert x.shape == (3, 2) and np.isfinite(x).all()


def test_ab_coefficients_and_z_scheme():
    for q in (1, 2, 3, 4):
        assert AB.ab_order(AB.AB_BETAS[q]) == q
        assert sum(AB.AB_BETAS[q]) == 1

    def f(t, x, z, k, implicit):                             # x' = t, z' = t
        return np.full_like(x, t), np.full_like(z, t)

    tau, K = 0.1, 20
    out = {}
    for zs in ("euler", "ab"):
        d = make_discretization("ab3", z_scheme=zs)
        x, z = np.zeros((2, 1)), np.zeros((2, 1))
        for k in range(K):
            x, z = d.step(k, k * tau, tau, x, z, f)
        out[zs] = (x, z)
    exact = 0.5 * (K * tau) ** 2
    x_ab, z_ab = out["ab"]
    x_eu, z_eu = out["euler"]
    assert np.allclose(x_ab, x_eu)                           # x always AB
    assert abs(z_ab[0, 0] - exact) < abs(z_eu[0, 0] - exact) # AB on z is more accurate
    assert abs(x_ab[0, 0] - exact) < 1e-2                    # only start-up error remains


# ---------------------------------------------------------------------------
# verifier additions: independent (non-tautological) cross-checks
# ---------------------------------------------------------------------------
def test_derivative_approximation_order_on_exp():
    """Apply every look-ahead formula to samples of exp(t) (all derivatives
    equal exp(t) > 0) and check that the derivative error shrinks like tau^p,
    without using the package's own moment routine."""
    t0 = 0.3
    for key in sorted(LOOKAHEAD_KEYS):
        d = make_discretization(key)
        c = np.array([float(v) for v in d.coeffs])
        p = d.claimed_trunc_order
        errs = []
        for tau in (0.08, 0.04):
            samples = np.exp(t0 + tau * np.array([1 - i for i in range(len(c))]))
            errs.append(abs(c @ samples / tau - np.exp(t0)))
        assert abs(np.log2(errs[0] / errs[1]) - p) < 0.3, (key, p, errs)


def _locus_hmax(rho, sigma):
    """Smallest h > 0 for which rho(z) + h sigma(z) has a root on |z| = 1
    (boundary-locus method, independent of `znn_stability`)."""
    th = np.linspace(1e-7, np.pi, 200001)
    z = np.exp(1j * th)
    h = -np.polyval(rho, z) / np.polyval(sigma, z)
    s = np.sign(h.imag)
    cand = []
    for i in np.where(s[:-1] != s[1:])[0]:
        w = h.imag[i] / (h.imag[i] - h.imag[i + 1])
        hr = h.real[i] + w * (h.real[i + 1] - h.real[i])
        if hr > 1e-9:
            cand.append(hr)
    hp = -np.polyval(rho, -1.0) / np.polyval(sigma, -1.0)
    if hp > 0:
        cand.append(hp)
    return min(cand)


@pytest.mark.parametrize("key", sorted(ALL_KEYS))
def test_hmax_matches_boundary_locus(key):
    d = make_discretization(key)
    if key in AB.AB_KEYS:
        rho = np.array(AB.ab_char_poly(d.q))
        sigma = np.array([0.0] + [float(b) for b in d.beta])
    else:
        rho = np.array([float(v) for v in d.coeffs])
        sigma = np.zeros(len(rho))
        sigma[1] = 1.0                                     # h z^{N-2}
    assert abs(d.stability()["h_max"] - _locus_hmax(rho, sigma)) < 1e-4


def test_sun_tian_wang_two_step_closed_form():
    """Sun, Tian & Wang (optimization-online 6960 / DDNS 2019), Thm 3.4-3.5 and
    the optimal step: for the general 3-instant formula
    ((1+a1) x_{k+1} - (1+2a1) x_k + a1 x_{k-1})/tau their DTZNN (28) uses
    h_STW = tau*gamma/(1+a1) and is stable iff 0 < h_STW < (2+4a1)/(1+a1), optimal
    at (1+2a1)/(1+a1).  In the ZNN-Bench scale h = (1+a1) h_STW this is
    (0, 2+4a1) and h_opt = 1+2a1 (checked for a1 in (-1/2, 0), where the
    optimum is unique)."""
    for a1 in (F(-1, 3), F(-1, 4), F(-2, 5)):
        c = [1 + a1, -(1 + 2 * a1), a1]
        assert Z.truncation_order(c) == 1
        st = Z.lookahead_stability(c)
        assert abs(st["h_max"] - float(2 + 4 * a1)) < 1e-6
        assert abs(st["h_opt"] - float(1 + 2 * a1)) < 1e-4
        assert abs(st["rho_opt"] - np.sqrt(float(-a1 / (1 + a1)))) < 1e-6


def test_7i_published_domain_equals_zero_stability_region():
    """Domain (16) of Yang et al. (2020) must coincide with the region where the
    transcribed formula (8) satisfies the strong root condition; a transcription
    error in either would break this."""
    mism = 0
    for p1 in np.linspace(-1.5, -0.5, 41):
        for p2 in np.linspace(-1.1, -0.7, 41):
            p1f, p2f = F(p1).limit_denominator(10 ** 4), F(p2).limit_denominator(10 ** 4)
            strong = Z.root_condition([float(v) for v in Z.coeffs_7i_general(p1f, p2f)])["strong"]
            mism += strong != Z.in_7i_domain(p1f, p2f)
    assert mism == 0


def test_describe_is_json_and_float_parameters_are_decimal():
    import json
    for key in sorted(ALL_KEYS):
        desc = make_discretization(key).describe()
        json.dumps(desc)                                    # no numpy scalars
        if "integer_coeffs" in desc:
            assert all(type(v) is int for v in desc["integer_coeffs"]) and type(desc["denominator"]) is int
    # floats are read as the decimal numbers they print as
    d = make_discretization("zead_4i_general", a1=-0.1)
    assert d.coeffs_exact == make_discretization("zead_4i_6321").coeffs_exact
    assert d.name == "zead_4i_general(a1=-1/10)"
    assert make_discretization("zead_5i_general", a1=0.1).describe()["integer_coeffs"] == [13, 3, -12, -7, 3]
    d7 = make_discretization("zead_7i_general", p1=-1.1, p2=-1.0)
    assert d7.coeffs_exact == tuple(Z.coeffs_7i_general(F(-11, 10), F(-1)))
    assert make_discretization("uhlig_3_4", a=1.5).coeffs_exact == make_discretization("uhlig_3_4", a=F(3, 2)).coeffs_exact
    # exact integer form for large denominators (no int64 overflow)
    nums, den = Z.integer_form([F(1, 2 ** 70), F(-1, 2 ** 70)])
    assert (nums, den) == ([1, -1], 2 ** 70)


@pytest.mark.parametrize("key", ["zead_6i_qiu2019", "uhlig_4_6", "ab3"])
def test_batch_rows_independent_and_reset(key):
    """B trajectories with different gains equal B separate runs, and a reused
    discretizer object is reset by `simulate` (history does not leak)."""
    P = make_problem("linear_system", n=4, kappa=10, seed=1, T=0.5)
    tau, hs = 1e-2, np.array([0.05, 0.1, 0.15])
    x0 = P.x0(np.random.default_rng(3), 3)
    d = make_discretization(key)
    sb = simulate(P, make_model("znn"), d, tau=tau, B=3, gamma=hs / tau, x0=x0, record_solution_error=False)
    for b in range(3):
        s1 = simulate(P, make_model("znn"), make_discretization(key), tau=tau, B=1, gamma=hs[b] / tau,
                      x0=x0[b:b + 1], record_solution_error=False)
        assert np.allclose(s1["residual"][:, 0], sb["residual"][:, b], rtol=1e-10, atol=1e-13)
    s2 = simulate(P, make_model("znn"), d, tau=tau, B=3, gamma=hs / tau, x0=x0, record_solution_error=False)
    assert np.array_equal(s2["residual"], sb["residual"])
