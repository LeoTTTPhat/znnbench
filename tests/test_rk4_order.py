"""Order reduction of the classical RK4 "non-causal reference" on linear ZNN.

Spec: docs/spec/rk4_order_reduction.md.

Linear ZNN on A(t) x = b(t) is the Prothero-Robinson-type problem

    x_dot = -(gamma I + A^{-1} A_dot) (x - x*(t)) + x*_dot(t)

(Prothero & Robinson, Math. Comp. 28 (1974) 145-162,
doi:10.1090/S0025-5718-1974-0331793-2; Hairer & Wanner, Solving ODEs II,
Sect. IV.15, doi:10.1007/978-3-642-05221-7).  For constant A it is exactly the
PR problem with lambda = -gamma.  RK4 has classical order 4 but stage order 1
and weak stage order 1 (Biswas, Ketcheson, Roberts, Seibold, Shirokoff, SIAM J.
Numer. Anal. 63(4) (2025) 1398-1426, doi:10.1137/23M1606812, arXiv:2310.02817,
Sect. 1.3, eq. (3): observed rate min(p, q_w + 1) = 2 at bounded tau*L), so at
fixed h = gamma*tau (z = tau*lambda = -h fixed) its local error keeps a tau^2
term, C_2(z) tau^2 x*'' with C_2(z) = -z^3/96, and the steady residual is
O(tau^2); at fixed gamma (z -> 0) it is O(tau^4).

What is checked here:
  * the registered `rk4` is the classical tableau: the full Butcher tableau
    (A, b, c) is read off the stage inputs, for x and for z; R(z);
  * the stage-order-1 defect, C_2(z) = -z^3/96 and C_3..C_5 in exact
    arithmetic; weak stage order exactly 1 (Biswas et al. 2025, eq. (3));
  * every explicit s-stage method of order p = s in {2, 3, 4} has
    C_2(z) = -c_2 z^(s-1) / (2 s!)  (Heun, Ralston, Kutta-3, Heun-3, RK4, 3/8);
  * one RK4 step on the scalar PR problem equals the predicted local error
    sum_q C_q(z) tau^q phi^(q) (to round-off), so it is O(tau^2) at fixed z
    and O(tau^5) at fixed lambda;
  * MSSRE slopes on `linear_system` and on a constant-A / moving-b problem:
    about 2 at fixed h = 0.3 and h = 0.05, about 4 at fixed gamma = 10;
  * on the constant-A problem the measured MSSRE equals the exact discrete
    PR steady state (no asymptotics) in all three regimes;
  * the prefactor at fixed h: MSSRE ~ h^3/(96 (1 - R(-h))) tau^2 max||A x*''||,
    against Euler's tau^2/(2h) max||A x*''|| (ratio 1536 at h = 0.3);
  * the consequence for the paper at fixed h = 0.3: the Taylor-Zhang formula
    (4 instants) has the steeper slope (3) yet a larger residual than RK4 on
    the whole tau grid down to 1e-3, while an order-5 8-instant formula beats
    RK4 already at tau = 1e-2.

All runs start at x0 = x*(0), so the measured residual is the truncation-
driven steady state only (at h = 0.05, tau = 1e-2 the gain is only 5 and a
random start would still be visible in the steady window; see the spec).
"""
from __future__ import annotations

from fractions import Fraction as Fr
from functools import lru_cache
from math import factorial

import numpy as np
import pytest

from znnbench import make_discretization, make_model, make_problem, simulate
from znnbench.metrics import empirical_order, mssre, steady_window
from znnbench.problems._tv import TVMatrix
from znnbench.problems.linear_system import LinearSystem

# ---------------------------------------------------------------------------
# classical RK4 tableau (exact) and the derived constants
# ---------------------------------------------------------------------------
A_RK4 = [[Fr(0), Fr(0), Fr(0), Fr(0)],
         [Fr(1, 2), Fr(0), Fr(0), Fr(0)],
         [Fr(0), Fr(1, 2), Fr(0), Fr(0)],
         [Fr(0), Fr(0), Fr(1), Fr(0)]]
B_RK4 = [Fr(1, 6), Fr(1, 3), Fr(1, 3), Fr(1, 6)]
C_RK4 = [Fr(0), Fr(1, 2), Fr(1, 2), Fr(1)]


def _matvec(M, v):
    return [sum(M[i][j] * v[j] for j in range(len(v))) for i in range(len(M))]


def _dot(u, v):
    return sum(a * b for a, b in zip(u, v))


def stage_defect(q, A=A_RK4, c=C_RK4):
    """d_q = (c^q - q A c^(q-1)) / q!  (stage q-th order defect; d_q = 0 for
    q <= stage order).  In the notation of Biswas et al. (2025), Sect. 1.3,
    d_q = -tau^(q) / (q-1)!  with the stage-order residual
    tau^(q) = A c^(q-1) - c^q / q."""
    cq = [ci ** q for ci in c]
    cq1 = [ci ** (q - 1) for ci in c]
    Ac = _matvec(A, cq1)
    return [(a - q * b) / factorial(q) for a, b in zip(cq, Ac)]


def local_error_poly(q, A=A_RK4, b=B_RK4, c=C_RK4):
    """Coefficients (in powers z^0..z^s) of the polynomial C_q(z) with

        local error of one explicit RK step on y' = lambda (y - phi) + phi'
        started on phi:  delta = sum_q C_q(z) tau^q phi^(q)(t_n),  z = tau*lambda,

        C_q(z) = -(1 - q b^T c^(q-1))/q!  -  sum_{k=0}^{s-1} z^(k+1) b^T A^k d_q
    (A is strictly lower triangular, hence nilpotent, A^s = 0)."""
    s = len(b)
    coef = [Fr(0)] * (s + 1)
    coef[0] = -(1 - q * _dot(b, [ci ** (q - 1) for ci in c])) / factorial(q)
    v = stage_defect(q, A, c)
    for k in range(s):
        coef[k + 1] -= _dot(b, v)
        v = _matvec(A, v)
    return coef


def poly_eval(coef, z):
    return sum(float(c) * z ** i for i, c in enumerate(coef))


def R_rk4(z):
    """Stability function of classical RK4."""
    return 1 + z + z ** 2 / 2 + z ** 3 / 6 + z ** 4 / 24


def rk4_const(h):
    """Fixed-h steady-state constant: e_ss = rk4_const(h) tau^2 x*'' + O(tau^3),
    i.e. C_2(-h) / (1 - R(-h)) = h^3 / (96 (1 - R(-h)))."""
    return h ** 3 / (96.0 * (1.0 - R_rk4(-h)))


def euler_const(h):
    """Euler look-ahead: e_ss = -tau^2 x*'' / (2h) + O(tau^3) (magnitude)."""
    return 1.0 / (2.0 * h)


# ---------------------------------------------------------------------------
# problems
# ---------------------------------------------------------------------------
class ConstantAMovingB(LinearSystem):
    """A x = b(t) with constant A (TVMatrix frozen with freq = 0, so A_dot = 0
    exactly) and the same moving b(t) as `linear_system`.  Linear ZNN on it is
    exactly the vector Prothero-Robinson problem with lambda = -gamma.
    Test-local, deliberately not registered."""

    family = "linear-system"

    def __init__(self, n=4, kappa=10.0, freq=1.0, seed=0, T=10.0):
        super().__init__(n=n, kappa=kappa, freq=freq, seed=seed, T=T)
        self.A = TVMatrix(n, np.random.default_rng([seed, 99]), kappa=kappa, freq=0.0)
        self.name = f"const_A_moving_b_n{n}_k{kappa:g}"


@lru_cache(maxsize=None)
def _problem(kind):
    if kind == "linear_system":
        return make_problem("linear_system", n=4, kappa=10, seed=1, T=4.0)
    if kind == "const_A":
        return ConstantAMovingB(n=4, kappa=10, seed=1, T=4.0)
    raise KeyError(kind)


@lru_cache(maxsize=None)
def _mssre(kind, disc, tau, h=None, gamma=None):
    """MSSRE of linear ZNN at sampling period tau, either at fixed h
    (gamma = h/tau) or at fixed gamma; start on the solution."""
    P = _problem(kind)
    g = h / tau if h is not None else gamma
    s = simulate(P, make_model("znn"), make_discretization(disc), tau=tau, B=1, gamma=g,
                 seed=0, x0=P.x_star(0.0)[None], online_stats=False)
    assert not s["diverged"].any()
    return float(mssre(s)[0])


def _max_A_xdd(kind, tau, dt=1e-3):
    """max over the steady window (on simulate's grid) of ||A(t) x*''(t)||,
    x*'' by central differences."""
    P = _problem(kind)
    N = int(round(P.T / tau))
    t = np.arange(N + 1) * tau
    out = 0.0
    for tk in t[steady_window(t)]:
        xdd = (P.x_star(tk + dt) - 2.0 * P.x_star(tk) + P.x_star(tk - dt)) / dt ** 2
        out = max(out, float(np.linalg.norm(P.A(tk) @ xdd)))
    return out


PROBLEMS = ["linear_system", "const_A"]
TAUS_H03 = (3e-2, 1e-2, 3e-3, 1e-3)
REGIMES = {
    # name: (h, gamma, taus, expected residual order)
    "h=0.3": (0.3, None, TAUS_H03, 2),
    "h=0.05": (0.05, None, (1e-2, 5e-3, 2.5e-3, 1.25e-3), 2),
    "gamma=10": (None, 10.0, (5e-2, 2.5e-2, 1.25e-2, 6.25e-3), 4),
}


def _series(kind, disc, regime):
    h, gamma, taus, _ = REGIMES[regime]
    return taus, [_mssre(kind, disc, tau, h=h, gamma=gamma) for tau in taus]


# ---------------------------------------------------------------------------
# 1. the implementation is the classical tableau
# ---------------------------------------------------------------------------
def test_registered_rk4_is_the_classical_tableau():
    disc = make_discretization("rk4")
    assert disc.causal is False
    # `expected_order` follows the fixed-h convention of
    # tests/test_discretizations.py::test_empirical_order_fixed_h, where RK4
    # measures 2: it may be None (current) or 2, never the classical 4
    # (spec, Sect. 6).  Optional regime-specific attributes, if the owner of
    # multistep.py adds them, must be 2 (fixed h) and 4 (fixed gamma).
    assert disc.expected_order in (None, 2)
    for attr, val in (("expected_order_fixed_h", 2), ("expected_order_fixed_gamma", 4)):
        if getattr(disc, attr, None) is not None:
            assert getattr(disc, attr) == val, attr

    # (a) the full Butcher tableau, read off the stage inputs.  Stage j returns
    # the unit vector e_j for both x and z, so stage i is evaluated at
    # x0 + tau * sum_j a_ij e_j and the update is x0 + tau * sum_j b_j e_j.
    # (R(z) and the stage times alone do not pin RK4 down: every member of
    # the one-parameter family of order-4 methods with c = (0, 1/2, 1/2, 1)
    # has them.)
    rec = []

    def f_unit(t, x, z, k, implicit):
        e = np.zeros((1, 4))
        e[0, len(rec)] = 1.0
        rec.append((t, k, x.copy(), z.copy()))
        return e, e.copy()

    t0, tau = 0.7, 0.5
    x0 = np.array([[0.3, -1.0, 2.0, 0.5]])
    z0 = np.array([[1.0, 0.0, -0.25, 4.0]])
    x1, z1 = disc.step(5, t0, tau, x0, z0, f_unit)
    A_ref = np.array([[float(a) for a in row] for row in A_RK4])
    b_ref = np.array([float(v) for v in B_RK4])
    assert len(rec) == 4
    for slot, s0, s1 in ((2, x0, x1), (3, z0, z1)):
        A_num = np.array([(r[slot] - s0)[0] / tau for r in rec])
        assert np.allclose(A_num, A_ref, rtol=0, atol=1e-14), A_num
        assert np.allclose((s1 - s0)[0] / tau, b_ref, rtol=0, atol=1e-14)
    assert [r[0] for r in rec] == pytest.approx([t0 + float(c) * tau for c in C_RK4])
    assert [r[1] for r in rec] == [5, None, None, None]

    # (b) stability function and stage times on x_dot = lambda x
    calls = []

    def f_lin(lam):
        def f(t, x, z, k, implicit):
            calls.append((t, k))
            return lam * x, np.zeros_like(z)
        return f

    t0, tau = 0.7, 0.1
    for lam in (-1.0, -3.0, -27.0):
        calls.clear()
        x1, _ = disc.step(5, t0, tau, np.ones((1, 1)), np.zeros((1, 0)), f_lin(lam))
        assert abs(x1[0, 0] - R_rk4(lam * tau)) < 1e-14
        # stage times t0 + c_i tau; only the first stage is a causal sample (k given)
        assert [c for c, _ in calls] == pytest.approx([t0 + float(c) * tau for c in C_RK4])
        assert [k for _, k in calls] == [5, None, None, None]


def test_constant_A_problem_is_consistent():
    P = _problem("const_A")
    assert np.abs(P.A.dot(1.3)).max() == 0.0 and np.allclose(P.A(0.0), P.A(2.9))
    rng = np.random.default_rng(0)
    for t in (0.0, 1.1, 3.7):
        assert np.abs(P.E(P.x_star(t)[None], P.data(t))).max() < 1e-12
        x = rng.standard_normal((2, P.n_x))
        dt = 1e-6
        fd = (P.E(x, P.data(t + dt)) - P.E(x, P.data(t - dt))) / (2 * dt)
        assert np.allclose(fd, P.dEdt(x, t), atol=1e-6)


# ---------------------------------------------------------------------------
# 2. stage order 1 and the reduction polynomial C_2(z) = -z^3/96 (exact)
# ---------------------------------------------------------------------------
def test_rk4_stage_order_one_and_reduction_polynomial():
    b, c = B_RK4, C_RK4
    Ac = _matvec(A_RK4, c)
    # classical order 4: the eight order conditions
    assert [_dot(b, [ci ** (q - 1) for ci in c]) for q in (1, 2, 3, 4)] == \
        [Fr(1), Fr(1, 2), Fr(1, 3), Fr(1, 4)]
    assert _dot(b, Ac) == Fr(1, 6)
    assert _dot(b, [ci * a for ci, a in zip(c, Ac)]) == Fr(1, 8)
    assert _dot(b, _matvec(A_RK4, [ci ** 2 for ci in c])) == Fr(1, 12)
    assert _dot(b, _matvec(A_RK4, Ac)) == Fr(1, 24)
    # stage order exactly 1
    assert stage_defect(1) == [0, 0, 0, 0]
    assert stage_defect(2) == [Fr(0), Fr(1, 8), Fr(-1, 8), Fr(0)]
    # reduction polynomials: C_1 = 0, C_2 = -z^3/96, C_3 = z^2/288 - z^3/576,
    # C_4 = -z/576 + ..., C_5(0) = 1/2880 (the classical order-4 error term)
    assert local_error_poly(1) == [0] * 5
    assert local_error_poly(2) == [0, 0, 0, Fr(-1, 96), 0]
    assert local_error_poly(3) == [0, 0, Fr(1, 288), Fr(-1, 576), 0]
    assert local_error_poly(4) == [0, Fr(-1, 576), Fr(1, 1152), Fr(-1, 4608), 0]
    assert local_error_poly(5) == [Fr(1, 2880), Fr(-1, 1440), Fr(1, 7680), Fr(-1, 46080), 0]
    # the steady-state constant used below is C_2(-h) / (1 - R(-h))
    for h in (0.05, 0.3, 1.0):
        assert poly_eval(local_error_poly(2), -h) / (1 - R_rk4(-h)) == pytest.approx(rk4_const(h))


def test_rk4_weak_stage_order_is_one():
    """Biswas, Ketcheson, Roberts, Seibold, Shirokoff, SIAM J. Numer. Anal.
    63(4) (2025) 1398-1426 (arXiv:2310.02817), Sect. 1.3: stage-order residuals
    tau^(k) = A c^(k-1) - c^k / k; weak stage order q_w is the largest q with
    b^T (I - zA)^{-1} tau^(k) = 0 for all z and k <= q (eq. (3)); the observed
    rate on linear problems with time-dependent forcing at bounded tau*L is
    min(p, q_w + 1).  For RK4 (A nilpotent) the resolvent condition is
    b^T A^j tau^(k) = 0 for j = 0..3."""
    def tau_k(k):
        return [a - ci ** k / k for a, ci in zip(_matvec(A_RK4, [ci ** (k - 1) for ci in C_RK4]),
                                                 C_RK4)]

    def resolvent_coeffs(v):
        out = []
        for _ in range(4):
            out.append(_dot(B_RK4, v))
            v = _matvec(A_RK4, v)
        return out

    for k in (1, 2, 3, 4, 5):          # d_k = -tau^(k) / (k-1)!
        assert stage_defect(k) == [-t / factorial(k - 1) for t in tau_k(k)]
    assert resolvent_coeffs(tau_k(1)) == [0, 0, 0, 0]                 # q_w >= 1
    # q_w = 1, so the predicted fixed-h rate is min(4, 1 + 1) = 2; the
    # non-zero entry is -b^T A^2 d_2 = -1/96, i.e. C_2(z) = -z^3/96
    assert resolvent_coeffs(tau_k(2)) == [0, 0, Fr(-1, 96), 0]


EXPLICIT_S_EQ_P = {
    # name: (A, b), all with order p = number of stages s
    "heun2": ([[0, 0], [1, 0]], [Fr(1, 2), Fr(1, 2)]),
    "ralston2": ([[0, 0], [Fr(2, 3), 0]], [Fr(1, 4), Fr(3, 4)]),
    "kutta3": ([[0, 0, 0], [Fr(1, 2), 0, 0], [-1, 2, 0]], [Fr(1, 6), Fr(2, 3), Fr(1, 6)]),
    "heun3": ([[0, 0, 0], [Fr(1, 3), 0, 0], [0, Fr(2, 3), 0]], [Fr(1, 4), 0, Fr(3, 4)]),
    "rk4": (A_RK4, B_RK4),
    "rk38": ([[0, 0, 0, 0], [Fr(1, 3), 0, 0, 0], [Fr(-1, 3), 1, 0, 0], [1, -1, 1, 0]],
             [Fr(1, 8), Fr(3, 8), Fr(3, 8), Fr(1, 8)]),
}


@pytest.mark.parametrize("name", list(EXPLICIT_S_EQ_P))
def test_explicit_s_equals_p_methods_reduce_to_order_two_at_fixed_h(name):
    """Spec Sect. 4: for any explicit s-stage method of order p = s >= 2,
    C_2(z) = -c_2 z^(s-1) / (2 s!), a non-zero monomial, so C_2(-h) != 0 for
    every h > 0 and the fixed-h residual order is 2.  Exact arithmetic on six
    textbook tableaux (only `rk4` is registered in ZNN-Bench)."""
    A, b = EXPLICIT_S_EQ_P[name]
    A = [[Fr(a) for a in row] for row in A]
    b = [Fr(v) for v in b]
    s = len(b)
    c = [sum(row) for row in A]
    for q in range(1, s + 1):                      # classical order s (bushy trees)
        assert _dot(b, [ci ** (q - 1) for ci in c]) == Fr(1, q)
    v = list(c)                                    # tall tree: b^T A^(s-2) c = 1/s!
    for _ in range(s - 2):
        v = _matvec(A, v)
    assert _dot(b, v) == Fr(1, factorial(s))
    assert stage_defect(1, A, c) == [0] * s        # stage order exactly 1
    assert any(stage_defect(2, A, c))
    expected = [Fr(0)] * (s + 1)
    expected[s - 1] = -c[1] / (2 * factorial(s))
    assert local_error_poly(2, A, b, c) == expected


# ---------------------------------------------------------------------------
# 3. one RK4 step on the scalar Prothero-Robinson problem
# ---------------------------------------------------------------------------
def _pr_local_error(lam, tau, tn=0.7, w=1.3):
    """One registered-RK4 step on y' = lam (y - phi) + phi', phi = sin(w t),
    started on phi(tn).  Returns (local error, predicted sum_q C_q tau^q phi^(q))."""
    phi = lambda t: np.sin(w * t)  # noqa: E731

    def f(t, x, z, k, implicit):
        return lam * (x - phi(t)) + w * np.cos(w * t), np.zeros_like(z)

    x1, _ = make_discretization("rk4").step(0, tn, tau, np.array([[phi(tn)]]),
                                           np.zeros((1, 0)), f)
    le = x1[0, 0] - phi(tn + tau)
    z = lam * tau
    # phi^(q)(tn) = w^q sin(w tn + q pi/2)
    pred = sum(poly_eval(local_error_poly(q), z) * tau ** q * w ** q
               * np.sin(w * tn + q * np.pi / 2) for q in range(2, 12))
    return le, pred


def test_prothero_robinson_local_error_matches_expansion():
    taus = (2e-2, 1e-2, 5e-3, 2.5e-3)
    h = 0.3
    fixed_z = [_pr_local_error(-h / tau, tau) for tau in taus]
    fixed_lam = [_pr_local_error(-10.0, tau) for tau in taus]
    for le, pred in fixed_z + fixed_lam:
        # |pred| ranges over 1e-12 .. 1e-7; the bound is relative plus round-off of x1 ~ 1
        assert abs(le - pred) <= 1e-6 * abs(pred) + 2e-15, (le, pred)
    # fixed z: local error O(tau^2), leading term C_2(-h) tau^2 phi'' = h^3/96 tau^2 phi''
    s_z = empirical_order(taus, [abs(le) for le, _ in fixed_z])[0]
    assert abs(s_z - 2.0) < 0.1, s_z
    w, tn, tau = 1.3, 0.7, taus[-1]
    lead = h ** 3 / 96 * tau ** 2 * (-w ** 2 * np.sin(w * tn))
    assert fixed_z[-1][0] / lead == pytest.approx(1.0, rel=0.01)
    # fixed lambda: classical local order tau^5
    s_l = empirical_order(taus, [abs(le) for le, _ in fixed_lam])[0]
    assert abs(s_l - 5.0) < 0.2, s_l


# ---------------------------------------------------------------------------
# 4. residual slopes: 2 at fixed h, 4 at fixed gamma
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("regime", list(REGIMES))
@pytest.mark.parametrize("kind", PROBLEMS)
def test_rk4_residual_order(kind, regime):
    taus, errs = _series(kind, "rk4", regime)
    expected = REGIMES[regime][3]
    slope, used = empirical_order(taus, errs)
    assert used == len(taus), errs
    assert abs(slope - expected) < 0.25, (slope, errs)
    local = np.diff(np.log(errs)) / np.diff(np.log(taus))
    assert abs(local[-1] - expected) < 0.15, (local, errs)       # asymptotic pair


def _exact_pr_mssre(tau, gamma):
    """Exact steady-state MSSRE of RK4-ZNN on the constant-A problem.

    With A constant, every component of b is one sinusoid a sin(w t + ph), the
    error obeys e_{n+1} = R(z) e_n + delta_n with delta_n = sum_q C_q(z) tau^q
    x*^(q)(t_n), z = -gamma tau, and A delta_n = sum_q C_q(z) tau^q b^(q)(t_n).
    Per component the steady state is therefore
        (A e_n)_i = Im[ a_i e^{i(w_i t_n + ph_i)} sum_q C_q(z) (i w_i tau)^q
                        / (e^{i w_i tau} - R(z)) ]
    (no quasi-static approximation; series truncated at q = 13)."""
    P = _problem("const_A")
    z = -gamma * tau
    N = int(round(P.T / tau))
    t = np.arange(N + 1) * tau
    t = t[steady_window(t)]
    a, w, ph = P.b.a, P.b.w, P.b.ph
    S = sum(poly_eval(local_error_poly(q), z) * (1j * w * tau) ** q for q in range(2, 14))
    G = S / (np.exp(1j * w * tau) - R_rk4(z))
    Ae = np.imag(a[None, :] * np.exp(1j * (np.outer(t, w) + ph[None, :])) * G[None, :])
    return float(np.linalg.norm(Ae, axis=1).max())


@pytest.mark.parametrize("regime", list(REGIMES))
def test_const_A_mssre_equals_exact_prothero_robinson_prediction(regime):
    """The whole derivation, without asymptotics: measured MSSRE equals the
    exact PR steady state in every regime (observed agreement ~1e-6, i.e.
    round-off; the start-up transient from x0 = x*(0) is below that)."""
    h, gamma, taus, _ = REGIMES[regime]
    for tau in taus:
        g = h / tau if h is not None else gamma
        meas = _mssre("const_A", "rk4", tau, h=h, gamma=gamma)
        assert meas == pytest.approx(_exact_pr_mssre(tau, g), rel=2e-5), (tau, meas)


# ---------------------------------------------------------------------------
# 5. error constants at h = 0.3: RK4 vs Euler
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", PROBLEMS)
def test_error_constant_vs_euler_at_h03(kind):
    h = 0.3
    predicted_ratio = euler_const(h) / rk4_const(h)          # = 48 (1 - R(-h)) / h^4
    assert predicted_ratio == pytest.approx(1535.78, rel=1e-4)
    for tau in TAUS_H03[1:]:
        ratio = _mssre(kind, "euler", tau, h=h) / _mssre(kind, "rk4", tau, h=h)
        assert ratio == pytest.approx(predicted_ratio, rel=0.02), (tau, ratio)
    # absolute prefactors at the smallest tau: MSSRE = const(h) tau^2 max||A x*''||
    tau = TAUS_H03[-1]
    g = _max_A_xdd(kind, tau)
    assert _mssre(kind, "rk4", tau, h=h) / (rk4_const(h) * tau ** 2 * g) == \
        pytest.approx(1.0, rel=0.01)
    assert _mssre(kind, "euler", tau, h=h) / (euler_const(h) * tau ** 2 * g) == \
        pytest.approx(1.0, rel=0.01)
    # Euler has the same order 2 at fixed h
    errs_e = [_mssre(kind, "euler", t, h=h) for t in TAUS_H03]
    assert abs(empirical_order(TAUS_H03, errs_e)[0] - 2.0) < 0.1


# ---------------------------------------------------------------------------
# 6. consequence for the paper at fixed h = 0.3
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", PROBLEMS)
def test_zead_beats_rk4_in_order_not_in_constant(kind):
    h = 0.3
    taus, e_rk4 = _series(kind, "rk4", "h=0.3")
    e_tz = [_mssre(kind, "taylor_zhang", t, h=h) for t in taus]
    s_rk4 = empirical_order(taus, e_rk4)[0]
    s_tz = empirical_order(taus, e_tz)[0]
    assert abs(s_tz - 3.0) < 0.15 and s_tz > s_rk4 + 0.8, (s_tz, s_rk4)
    # ... yet RK4 has the smaller residual on the whole grid down to tau = 1e-3
    # (predicted crossover tau_c = rk4_const(h) h / |C_3| ~ 4.9e-4 for one
    # harmonic of unit frequency; measured ~1.4e-4 / 4.1e-4, see the spec)
    assert all(a > b for a, b in zip(e_tz, e_rk4)), (e_tz, e_rk4)
    # an order-5 formula (8 instants, stable at h = 0.3) wins already at 1e-2
    tau = 1e-2
    assert _mssre(kind, "zead_8i_wu2020", tau, h=h) < 0.1 * _mssre(kind, "rk4", tau, h=h)
