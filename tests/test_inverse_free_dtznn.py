"""Tests for the inverse-free DTZNNs of Ma & Huang (Front. Neurorobot. 19 (2025)
1576473) and the named ZeaD members (inverse_free_dtznn.py, zead_named.py).

(1) exact transcription of Eqs. (8)-(14), (20)-(25) and the h mapping,
(2) Table 1 of Ma & Huang reproduced through ``simulate`` on the
    ``nonlinear_equations`` guo2018 system (x0 = 0.2*ones(4), T = 10 s,
    SSCE = max ||f|| over the last 25 %),
(3) equality with independent from-scratch loops (closed-form Jdot for CDTZNN),
(4) the information pattern: causal, no linear solve, no derivative leak
    under bd1/bd2/none, solve_J restored, batch/reset consistency,
(5) generic behaviour (order 4 on linear_system, guards, other models).
n <= 16, <= 1e4 steps per simulation; about 15 s, well under 200 MB.
"""
from fractions import Fraction as F

import numpy as np
import pytest

from znnbench import make_discretization, make_model, make_noise, make_problem, simulate
from znnbench.discretizations import inverse_free_dtznn as IF
from znnbench.discretizations import zead as Z
from znnbench.discretizations import zead_named as ZN
from znnbench.metrics import empirical_order, mssre
from znnbench.problems.optimization import NonlinearEquations

TAB = IF.MAHUANG2025_TABLE1
HS = (0.10, 0.15, 0.20)
X0 = 0.2 * np.ones(4)


def _table1_run(key, tau, T=10.0, **kw):
    """SSCE of the three Table 1 step sizes as one batch (gamma per row)."""
    P = make_problem("nonlinear_equations", variant="guo2018", T=T)
    g = np.array([IF.gamma_from_paper_h(key, h, tau) for h in HS])
    s = simulate(P, make_model("znn"), make_discretization(key, **kw), tau=tau, gamma=g, B=3,
                 x0=np.tile(X0, (3, 1)), record_solution_error=False)
    return mssre(s)


# ---------------------------------------------------------------------------
# guo2018 system in closed form (independent re-typing of Ma & Huang Eq. (33))
# ---------------------------------------------------------------------------
def _f(x, t):
    x1, x2, x3, x4 = x
    return np.array([np.log(x1) - 1 / (t + 1), x1 * x2 - np.sin(t) * np.exp(1 / (t + 1)),
                     x1 ** 2 - np.sin(t) * x2 + x3 - 2, x1 ** 2 - x2 ** 2 + x3 + x4 - t])


def _ft(x, t):
    ex = np.exp(1 / (t + 1))
    return np.array([1 / (t + 1) ** 2, -np.cos(t) * ex + np.sin(t) * ex / (t + 1) ** 2,
                     -np.cos(t) * x[1], -1.0])


def _J(x, t):
    x1, x2 = x[0], x[1]
    return np.array([[1 / x1, 0, 0, 0], [x2, x1, 0, 0], [2 * x1, -np.sin(t), 1, 0],
                     [2 * x1, -2 * x2, 1, 1]])


def _Jdot(v, x, t):
    """d/dt J(x(t), t) with x_dot = v, in closed form."""
    return np.array([[-v[0] / x[0] ** 2, 0, 0, 0], [v[1], v[0], 0, 0], [2 * v[0], -np.cos(t), 0, 0],
                     [2 * v[0], -2 * v[1], 0, 0]])


# ---------------------------------------------------------------------------
# (1) transcription
# ---------------------------------------------------------------------------
def test_named_members_exact_and_in_families():
    d7 = make_discretization("zead_7i_cang2024")
    assert d7.coeffs_exact == tuple(F(n, 276) for n in (110, 42, -89, -102, 12, 44, -17))  # Eq. (10)
    assert list(d7.coeffs_exact) == Z.coeffs_7i_general(F(-89, 110), F(-51, 55))
    assert Z.in_7i_domain(F(-89, 110), F(-51, 55))
    assert d7.trunc_order() == 4 and d7.expected_order == 5
    d5 = make_discretization("zead_5i_mahuang2025")
    assert d5.coeffs_exact == tuple(F(n, 78) for n in (36, -1, -18, -27, 10))                 # Eq. (14)
    assert list(d5.coeffs_exact) == Z.coeffs_5i_general(F(5, 39))
    assert d5.trunc_order() == 3 and d5.expected_order == 4
    # Taylor DTZNN-I / -II formulas (Eqs. (8), (9)) are the existing keys
    assert make_discretization("zead_5i").coeffs_exact == tuple(F(n, 18) for n in (8, 1, -6, -5, 2))
    assert make_discretization("zead_6i_guo2017").coeffs_exact == tuple(
        F(n, 48) for n in (24, -5, -12, -6, -4, 3))
    for d in (d5, d7):
        ri, st, desc = d.root_info(), d.stability(), d.describe()
        assert ri["strong"] and 0 < st["h_opt"] < st["h_max"]
        assert desc["doi"] and desc["source"] and desc["causal"]


# x_{k+1} = sum_j a_j x_{k-j} - J^-( (1/c0) tau f_t + h f ): Ma & Huang Eqs. (11), (12), (13), (23)
PRINTED_UPDATES = {
    "zead_5i": ([F(-1, 8), F(3, 4), F(5, 8), F(-1, 4)], F(9, 4)),
    "zead_6i_guo2017": ([F(5, 24), F(1, 2), F(1, 4), F(1, 6), F(-1, 8)], F(2)),
    "zead_7i_cang2024": ([F(-21, 55), F(89, 110), F(51, 55), F(-6, 55), F(-2, 5), F(17, 110)], F(138, 55)),
    "zead_5i_mahuang2025": ([F(1, 36), F(1, 2), F(3, 4), F(-5, 18)], F(13, 6)),
}


@pytest.mark.parametrize("key", sorted(PRINTED_UPDATES))
def test_printed_update_forms_and_h_mapping(key):
    a_printed, inv_c0 = PRINTED_UPDATES[key]
    c = make_discretization(key).coeffs_exact
    assert [-ci / c[0] for ci in c[1:]] == a_printed
    assert 1 / c[0] == inv_c0                         # tau f_t coefficient
    assert IF.TABLE1_H_FACTOR[key] == c[0]            # h_ours = c0 * h_paper
    assert IF.gamma_from_paper_h(key, 0.2, 0.01) == pytest.approx(float(c[0]) * 20.0)


def test_cdtznn_recursion_coefficients_and_roots():
    """Eq. (25): both recursions use (1/36, 1/2, 3/4, -5/18) and 13/6.  The roots
    printed in the proof of Theorem 2 are not roots of P1 (P1(1) = 0 for any
    consistent formula); the true ones are 1, 0.3159, -0.6441 +- 0.6814i, so the
    zero-stability conclusion holds anyway (strong root condition)."""
    assert IF.CDTZNN_A == (F(1, 36), F(1, 2), F(3, 4), F(-5, 18)) and 1 / IF.CDTZNN_C0 == F(13, 6)
    P1 = [1, -F(1, 36), -F(1, 2), -F(3, 4), F(5, 18)]
    assert sum(P1) == 0
    r = np.sort_complex(np.roots([float(c) for c in P1]))
    assert np.allclose(r, np.sort_complex([1.0, 0.315941, -0.644082 + 0.681443j, -0.644082 - 0.681443j]),
                       atol=1e-5)
    printed = [-0.5211 + 0.4694j, 0.2372 + 0.8065j]
    assert all(abs(np.polyval([float(c) for c in P1], z)) > 0.3 for z in printed)
    d = make_discretization("cdtznn_mahuang2025")
    assert d.root_info()["strong"]
    st = d.stability()                                # x-recursion, h_ours = 6/13 h_paper
    assert st["h_max"] == pytest.approx(0.1725, abs=2e-4)
    assert 0.2 * 6 / 13 < st["h_max"]                 # Table 1 step sizes are inside


# ---------------------------------------------------------------------------
# (2) Table 1 of Ma & Huang 2025
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key,tau,rtol", [
    ("euler", 0.01, 5e-3), ("euler", 0.001, 5e-3),   # (verifier) Euler row, h_ours = h_paper
    ("zead_5i", 0.01, 5e-3), ("zead_5i", 0.001, 5e-3),
    ("zead_6i_guo2017", 0.01, 5e-3), ("zead_6i_guo2017", 0.001, 5e-3),
    ("zead_7i_cang2024", 0.01, 5e-3),
    ("zead_7i_cang2024", 0.001, 6e-2),               # SSCE ~1e-13: round-off floor
    ("cdtznn_mahuang2025", 0.01, 5e-3), ("cdtznn_mahuang2025", 0.001, 5e-3),
])
def test_table1_reproduced(key, tau, rtol):
    r = _table1_run(key, tau)
    ref = np.array([TAB[key][(tau, h)] for h in HS])
    assert np.all(np.abs(r / ref - 1) < rtol), (key, tau, r / ref)


def test_table1_taylor_dtznn_iii_typo():
    """The printed 6.837e-8 at (tau, h) = (0.01, 0.20) is a typo for 6.837e-9."""
    r = _table1_run("zead_7i_cang2024", 0.01)
    printed = IF.MAHUANG2025_TABLE1_PRINTED_TYPOS[("zead_7i_cang2024", 0.01, 0.20)]
    assert abs(r[2] / 6.837e-9 - 1) < 1e-3
    assert 9 < printed / r[2] < 11
    assert r[0] > r[1] > r[2]                         # SSCE decreases with h along the row


def test_table1_cdtznn_independent_of_M0_and_xdot_readings():
    r_id = _table1_run("cdtznn_mahuang2025", 0.01, M0="identity")
    ref = np.array([TAB["cdtznn_mahuang2025"][(0.01, h)] for h in HS])
    assert np.all(np.abs(r_id / ref - 1) < 5e-3)
    r_model = _table1_run("cdtznn_mahuang2025", 0.01, xdot="model")          # generic reading
    assert np.all(np.abs(r_model / ref - 1) < 1.5e-2)
    r_printed = _table1_run("cdtznn_mahuang2025", 0.01, xdot="printed", T=4.0)
    assert np.all(r_printed > 1e-2)                   # the literal print does not reproduce


def test_table1_broyden():
    """Broyden-aided DTZNN.  secant='previous' reproduces three entries to 4 digits;
    the other three (and every entry of the printed secant) differ by up to 1.9x.
    Not reproducible in principle to 2 digits: the Broyden matrix keeps a memory of
    its start, and its SSCE varies ~3x over random x0 (test below)."""
    ref = {tau: np.array([TAB["broyden_dtznn_zhang2012"][(tau, h)] for h in HS]) for tau in (0.01, 0.001)}
    prev = {tau: _table1_run("broyden_dtznn_zhang2012", tau, secant="previous") / ref[tau]
            for tau in (0.01, 0.001)}
    assert np.all(np.abs(prev[0.01][:2] - 1) < 1e-3) and abs(prev[0.001][0] - 1) < 1e-3
    assert np.allclose([prev[0.01][2], prev[0.001][1], prev[0.001][2]], [1.566, 0.894, 1.921], rtol=1e-2)
    cur = {tau: _table1_run("broyden_dtznn_zhang2012", tau) / ref[tau] for tau in (0.01, 0.001)}
    assert np.allclose(np.r_[cur[0.01], cur[0.001]], [0.826, 0.911, 1.429, 1.005, 0.742, 1.874], rtol=1e-2)
    for rr in (prev, cur):                            # O(tau^2) mode of Lemma 2, roughly
        ratio = (rr[0.01] * ref[0.01]) / (rr[0.001] * ref[0.001])
        assert np.all((40 < ratio) & (ratio < 150)), ratio


def test_broyden_memory_vs_cdtznn_forgetting():
    P = make_problem("nonlinear_equations", T=10.0)
    x0 = P.x0(np.random.default_rng(0), 6)
    sb = simulate(P, make_model("znn"), make_discretization("broyden_dtznn_zhang2012"), tau=0.01,
                  gamma=20.0, B=6, x0=x0, record_solution_error=False)
    sc = simulate(P, make_model("znn"), make_discretization("cdtznn_mahuang2025"), tau=0.01,
                  gamma=0.2 * 6 / 13 / 0.01, B=6, x0=x0, record_solution_error=False)
    rb, rc = mssre(sb), mssre(sc)
    assert rb.max() / rb.min() > 1.5                  # steady state depends on the start
    assert rc.max() / rc.min() < 1.001                # unique steady state


def test_inverse_error_levels_match_figures():
    """Figs. 2c/4c: ||J M - I||_F of CDTZNN at steady state ~1e-7 (tau = 0.01) and
    ~1e-11 (0.001), h = 0.2.  Fig. 1c: Broyden's stays O(1) (3-9 in the figure;
    the text's 'order 1e-3' contradicts the figure)."""
    P = make_problem("nonlinear_equations", T=10.0)
    for key, c0, lo, hi in (("cdtznn_mahuang2025", 6 / 13, 1e-8, 1e-6), ("broyden_dtznn_zhang2012", 1.0, 0.5, 10.0)):
        d = make_discretization(key, record_inverse_error=True)
        simulate(P, make_model("znn"), d, tau=0.01, gamma=c0 * 0.2 / 0.01, x0=X0[None],
                 record_solution_error=False)
        ie = np.array(d.inverse_error)[:, 0]
        assert lo < ie[int(0.75 * len(ie)):].max() < hi, (key, ie[-1])


# ---------------------------------------------------------------------------
# (3) equality with independent from-scratch loops
# ---------------------------------------------------------------------------
def _scratch_cdtznn(h, tau, N, M0, startup="euler"):
    """Eq. (25) with closed-form Jdot(-M(f_t + f)); start-up Euler of the model
    velocity (ZNN-Bench convention) or Eq. (26) literally."""
    c0, a, I4 = 6 / 13, [1 / 36, 1 / 2, 3 / 4, -5 / 18], np.eye(4)
    gam = c0 * h / tau
    x = X0.copy()
    J0 = _J(x, 0.0)
    M = np.eye(4) if M0 == "identity" else J0.T / (np.abs(J0).sum(0).max() * np.abs(J0).sum(1).max())
    hx, hM, res = [], [], []
    for k in range(N + 1):
        t = k * tau
        res.append(np.linalg.norm(_f(x, t)))
        if k == N:
            break
        F_, Ft, Jk = _f(x, t), _ft(x, t), _J(x, t)
        Jd = _Jdot(-M @ (Ft + F_), x, t)
        hx, hM = [x] + hx[:3], [M] + hM[:3]
        if len(hx) < 4:
            if startup == "euler":
                x, M = x - tau * M @ (Ft + gam * F_), M - tau * M @ Jd @ M - tau * gam * M @ (Jk @ M - I4)
            else:
                x, M = x - M @ (tau * Ft + h * F_), M - tau * M @ Jd @ M - h * M @ (Jk @ M - I4)
        else:
            x, M = (sum(aj * xj for aj, xj in zip(a, hx)) - M @ (13 / 6 * tau * Ft + h * F_),
                    sum(aj * Mj for aj, Mj in zip(a, hM)) - 13 / 6 * tau * M @ Jd @ M - h * M @ (Jk @ M - I4))
    return x, np.array(res)


@pytest.mark.parametrize("M0", ["identity", "newton_schulz"])
def test_cdtznn_equals_scratch_with_closed_form_jdot(M0):
    tau, h, N = 0.01, 0.2, 400
    x_ref, res_ref = _scratch_cdtznn(h, tau, N, M0)
    P = make_problem("nonlinear_equations", T=N * tau)
    s = simulate(P, make_model("znn"), make_discretization("cdtznn_mahuang2025", M0=M0), tau=tau,
                 gamma=6 / 13 * h / tau, x0=X0[None], record_solution_error=False)
    assert np.max(np.abs(s["x_final"][0] - x_ref)) < 1e-12 * np.max(np.abs(x_ref))
    assert np.max(np.abs(s["residual"][:, 0] - res_ref) / res_ref) < 1e-5   # FD Jdot vs closed form


def test_cdtznn_paper_startup_same_steady_state():
    """Eq. (26) start-up (h_paper in the start-up) vs ZNN-Bench's Euler start-up."""
    _, r_euler = _scratch_cdtznn(0.2, 0.01, 1000, "identity", "euler")
    _, r_26 = _scratch_cdtznn(0.2, 0.01, 1000, "identity", "eq26")
    assert abs(r_26[750:].max() / r_euler[750:].max() - 1) < 1e-4
    assert abs(r_26[750:].max() / TAB["cdtznn_mahuang2025"][(0.01, 0.2)] - 1) < 5e-3


@pytest.mark.parametrize("secant", ["current", "previous"])
def test_broyden_equals_scratch(secant):
    tau, h, N = 0.01, 0.15, 400
    x, M, xp, tp = X0.copy(), np.eye(4), None, None
    for k in range(N):
        t = k * tau
        if xp is not None:
            u = x - xp
            v = _f(x, t) - _f(xp, t) if secant == "current" else _f(x, tp) - _f(xp, tp)  # Eq. (6)
            M = M + np.outer(u - M @ v, u @ M) / (u @ M @ v)
        xp, tp = x, t
        x = x - M @ (tau * _ft(x, t) + h * _f(x, t))                                   # Eq. (7)
    P = make_problem("nonlinear_equations", T=N * tau)
    s = simulate(P, make_model("znn"), make_discretization("broyden_dtznn_zhang2012", secant=secant),
                 tau=tau, gamma=h / tau, x0=X0[None], record_solution_error=False)
    assert np.max(np.abs(s["x_final"][0] - x)) < 1e-12 * np.max(np.abs(x))


# ---------------------------------------------------------------------------
# (4) information pattern
# ---------------------------------------------------------------------------
class _Spy(NonlinearEquations):
    """guo2018 problem counting solve_J / dEdt / data calls."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.n_solve = self.n_dEdt = self.n_data = 0
        self.t_data = []

    def solve_J(self, x, d, r):
        self.n_solve += 1
        return super().solve_J(x, d, r)

    def dEdt(self, x, t):
        self.n_dEdt += 1
        return super().dEdt(x, t)

    def data(self, t):
        self.n_data += 1
        self.t_data.append(t)
        return super().data(t)


class _CallLog:
    """Wraps a discretizer to record the (t, k) of every vector-field call."""

    def __init__(self, inner):
        self.inner, self.calls = inner, []

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def reset(self):
        self.inner.reset()

    def step(self, k, t, tau, x, z, f):
        def g(tt, xx, zz, kk, imp):
            self.calls.append((tt, kk, k, t))
            return f(tt, xx, zz, kk, imp)
        return self.inner.step(k, t, tau, x, z, g)


@pytest.mark.parametrize("key", ["cdtznn_mahuang2025", "broyden_dtznn_zhang2012"])
def test_causal_inverse_free_no_extra_data(key):
    P = _Spy(T=1.0)
    d = _CallLog(make_discretization(key))
    s = simulate(P, make_model("znn"), d, tau=0.01, gamma=10.0, x0=X0[None], record_solution_error=False)
    N = 100
    assert np.isfinite(s["residual"]).all()
    assert len(d.calls) == N and all(kk == k and tt == t for tt, kk, k, t in d.calls)   # one causal call
    assert P.n_solve == 0                              # no linear solve anywhere
    assert d.inner.n_substituted_solves == N and d.inner.n_true_solves == 0
    assert P.n_data == N + 1 and max(P.t_data) <= 1.0 + 1e-12   # only simulate's own samples
    assert "solve_J" not in vars(P)                    # substitution undone


def test_M0_inverse_uses_exactly_one_solve():
    P = _Spy(T=0.2)
    d = make_discretization("cdtznn_mahuang2025", M0="inverse")
    simulate(P, make_model("znn"), d, tau=0.01, gamma=10.0, B=2, x0=np.tile(X0, (2, 1)),
             record_solution_error=False)
    assert P.n_solve == 1 and d.n_true_solves == 1


@pytest.mark.parametrize("info", ["analytic", "bd1", "bd2", "none"])
def test_regimes_no_derivative_leak(info):
    """d_t J follows simulate's regime: dEdt is never called outside 'analytic'
    (there: 2 extra calls per step for d_t J M); CDTZNN tracks the exact-inverse
    formula zead_5i_mahuang2025 under every regime."""
    tau, N = 0.01, 600
    P = _Spy(T=N * tau)
    d = make_discretization("cdtznn_mahuang2025")
    s = simulate(P, make_model("znn"), d, tau=tau, gamma=0.2 * 6 / 13 / tau, x0=X0[None], info=info,
                 record_solution_error=False)
    assert d.info_used == info
    assert P.n_dEdt == (3 * N if info == "analytic" else 0) and P.n_solve == 0
    Q = make_problem("nonlinear_equations", T=N * tau)
    ref = simulate(Q, make_model("znn"), make_discretization("zead_5i_mahuang2025"), tau=tau,
                   gamma=0.2 * 6 / 13 / tau, x0=X0[None], info=info, record_solution_error=False)
    assert 0.67 < mssre(s)[0] / mssre(ref)[0] < 1.5


def test_reset_reuse_and_batch_consistency():
    P = make_problem("nonlinear_equations", T=2.0)
    for key in ("cdtznn_mahuang2025", "broyden_dtznn_zhang2012"):
        d = make_discretization(key)
        g = np.array([5.0, 9.0, 13.0])
        a = simulate(P, make_model("znn"), d, tau=0.01, gamma=g, B=3, x0=np.tile(X0, (3, 1)))
        b = simulate(P, make_model("znn"), d, tau=0.01, gamma=g, B=3, x0=np.tile(X0, (3, 1)))
        assert np.array_equal(a["residual"], b["residual"])          # reset() clears all state
        for i in range(3):
            c = simulate(P, make_model("znn"), make_discretization(key), tau=0.01, gamma=g[i], x0=X0[None])
            assert np.allclose(c["residual"][:, 0], a["residual"][:, i], rtol=1e-12, atol=1e-15)


def test_noise_enters_through_M_and_solve_restored_on_error():
    P = _Spy(T=2.0)
    for key in ("cdtznn_mahuang2025", "broyden_dtznn_zhang2012"):
        d = make_discretization(key)
        s = simulate(P, make_model("znn"), d, tau=0.01, gamma=10.0, x0=X0[None],
                     noise=make_noise("constant", level=0.1), record_solution_error=False)
        assert np.isfinite(s["residual"]).all() and P.n_solve == 0
        assert d.n_substituted_solves == 2 * 200       # model + design-formula noise, both via M_k

    class Boom(Exception):
        pass

    def f(t, x, z, k, imp):
        P.solve_J(x, P.data(t), np.ones_like(x))
        raise Boom
    d = make_discretization("broyden_dtznn_zhang2012").bind(P)
    with pytest.raises(Boom):
        d.step(0, 0.0, 0.01, X0[None], np.zeros((1, 0)), f)
    assert "solve_J" not in vars(P)


# ---------------------------------------------------------------------------
# (5) generic behaviour and guards
# ---------------------------------------------------------------------------
def test_cdtznn_order_four_on_linear_system():
    P = make_problem("linear_system", n=4, kappa=10, seed=1, T=6.0)
    taus = [1e-2, 5e-3, 2.5e-3, 1.25e-3]
    h = 0.0465                                        # ~h_opt of the 5-instant formula (ours)
    errs = [float(mssre(simulate(P, make_model("znn"), make_discretization("cdtznn_mahuang2025"), tau=t,
                                 gamma=h / t, x0=P.x_star(0.0)[None], record_solution_error=False))[0])
            for t in taus]
    slope, n = empirical_order(taus, errs, floor=1e-12)
    assert n == 4 and abs(slope - 4) < 0.35, (slope, errs)


@pytest.mark.parametrize("key,taus", [
    ("zead_5i_mahuang2025", [1e-2, 5e-3, 2.5e-3, 1.25e-3]),
    ("zead_7i_cang2024", [2e-2, 1e-2, 5e-3, 2.5e-3]),
])
def test_named_formulas_empirical_order(key, taus):
    """(verifier) DESIGN.md order check for the two named ZeaD members, as in
    tests/test_discretizations.py::test_empirical_order_fixed_h: linear ZNN on
    linear_system at the formula's own h_opt; slopes 3.94 and 4.92 measured."""
    d = make_discretization(key)
    h = d.stability()["h_opt"]
    P = make_problem("linear_system", n=4, kappa=10, seed=1, T=6.0)
    errs = [float(mssre(simulate(P, make_model("znn"), make_discretization(key), tau=t, gamma=h / t,
                                 x0=P.x_star(0.0)[None], seed=0, record_solution_error=False))[0])
            for t in taus]
    slope, n = empirical_order(taus, errs, floor=1e-11)
    assert n == 4 and abs(slope - d.expected_order) < 0.35, (key, slope, errs)


def test_mu_is_effective_gain_of_gain_schedule():
    """(verifier) ASSUMPTION 7: with a gain schedule the inversion recursion uses
    mu = gamma*g(t_k) (vpznn_exp, g(t) = 2^t + 2), checked over the two first
    (Euler start-up) steps against closed-form Jdot; a float mu is used as given."""
    tau, gam = 0.01, 4.0
    I4 = np.eye(4)

    def two_steps(mu_of_t):
        x, M = X0.copy(), I4.copy()
        for k in range(2):
            t = k * tau
            F_, Ft, Jk = _f(x, t), _ft(x, t), _J(x, t)
            Jd = _Jdot(-M @ (Ft + F_), x, t)
            x, M = (x - tau * M @ (Ft + gam * (2.0 ** t + 2.0) * F_),
                    M - tau * M @ Jd @ M - tau * mu_of_t(t) * M @ (Jk @ M - I4))
        return M

    P = make_problem("nonlinear_equations", T=2 * tau)
    for kw, mu_of_t in (({}, lambda t: gam * (2.0 ** t + 2.0)), ({"mu": 7.0}, lambda t: 7.0)):
        d = make_discretization("cdtznn_mahuang2025", M0="identity", **kw)
        simulate(P, make_model("vpznn_exp", p=2.0), d, tau=tau, gamma=gam, x0=X0[None],
                 record_solution_error=False)
        M_ref = two_steps(mu_of_t)
        assert np.max(np.abs(d.M[0] - M_ref)) < 1e-8 * np.max(np.abs(M_ref)), kw


def test_registry_and_metadata():
    for key, order in {**IF.INVERSE_FREE_KEYS, **ZN.ZEAD_NAMED_KEYS}.items():
        d = make_discretization(key)
        desc = d.describe()
        assert d.causal and not getattr(d, "implicit", False) and d.expected_order == order
        assert desc["source"] and desc["doi"]
    for key in IF.INVERSE_FREE_KEYS:
        desc = make_discretization(key).describe()
        assert desc["inverse_free"] and "no linear solve" in desc["information"]


def test_guards():
    with pytest.raises(ValueError, match="square"):
        simulate(make_problem("robot_puma560", T=0.05), make_model("znn"),
                 make_discretization("cdtznn_mahuang2025"), tau=0.01, gamma=10.0)
    with pytest.raises(ValueError, match="max_dim"):
        simulate(make_problem("matrix_inversion", n=3, T=0.05), make_model("znn"),
                 make_discretization("broyden_dtznn_zhang2012", max_dim=8), tau=0.01, gamma=10.0)
    with pytest.raises(RuntimeError, match="bind"):
        make_discretization("cdtznn_mahuang2025").step(0, 0.0, 0.01, X0[None], np.zeros((1, 0)), None)
    for bad in ({"xdot": "x"}, {"info": "bd3"}, {"mu": -1.0}, {"M0": "eye"}):
        with pytest.raises(ValueError):
            make_discretization("cdtznn_mahuang2025", **bad)
    with pytest.raises(ValueError):
        make_discretization("broyden_dtznn_zhang2012", secant="next")


def test_other_models():
    """CDTZNN needs the design formula (gain, regime) unless mu/info are given;
    Broyden substitutes J^-1 in any solve-based model; a model without J^-1 warns."""
    P = make_problem("nonlinear_equations", T=3.0)
    with pytest.raises(TypeError, match="mu="):
        simulate(P, make_model("ieznn"), make_discretization("cdtznn_mahuang2025"), tau=0.01,
                 gamma=10.0, x0=X0[None])
    s = simulate(P, make_model("ieznn"), make_discretization("cdtznn_mahuang2025", mu=5.0, info="analytic"),
                 tau=0.01, gamma=10.0, x0=X0[None])
    assert np.isfinite(s["residual"]).all() and s["residual"][-1, 0] < 1e-2
    s = simulate(P, make_model("ieznn"), make_discretization("broyden_dtznn_zhang2012"), tau=0.01,
                 gamma=10.0, x0=X0[None])
    assert np.isfinite(s["residual"]).all() and s["residual"][-1, 0] < 1e-2
    s = simulate(P, make_model("znn", activation="sign_bi_power"), make_discretization("cdtznn_mahuang2025"),
                 tau=0.01, gamma=5.0, x0=X0[None])
    assert np.isfinite(s["residual"]).all()
    with pytest.warns(RuntimeWarning, match="no J\\^-1"):
        simulate(P, make_model("gnn"), make_discretization("broyden_dtznn_zhang2012"), tau=0.01,
                 gamma=1.0, x0=X0[None])


def test_newton_schulz_start_tracks_exact_inverse_formula():
    """Generic problems: with the Newton-Schulz start CDTZNN tracks the exact-inverse
    formula; from the identity it fails where J is indefinite (tvqp_eq)."""
    P = make_problem("tvqp_eq", T=6.0)
    run = lambda key, **kw: float(np.max(mssre(simulate(           # noqa: E731
        P, make_model("znn"), make_discretization(key, **kw), tau=1e-2, gamma=8.0, B=2, seed=0,
        record_solution_error=False))))
    ref = run("zead_5i_mahuang2025")
    assert run("cdtznn_mahuang2025") < 2 * ref
    with np.errstate(all="ignore"):
        assert not run("cdtznn_mahuang2025", M0="identity") < 2 * ref
