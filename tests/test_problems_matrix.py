"""Unit tests for znnbench/problems/matrix.py (area: problems_matrix).

Every registered matrix problem is checked for
  * E(x*(t), d(t)) = 0,
  * dE/dt (exact) vs central finite differences of E(x, d(t +/- dt)),
  * solve_J vs a dense Jacobian built by finite differences,
  * JT vs the dense transpose,
  * batch consistency (B > 1 equals stacked B = 1),
  * ZNN + Euler convergence smoke test,
plus problem-specific faithfulness checks.  All sizes n <= 6.
"""
import pickle

import numpy as np
import pytest
from scipy.linalg import solve_sylvester

from znnbench import make_discretization, make_model, make_problem, simulate
from znnbench.metrics import mssre
from znnbench.registry import available

CONFIGS = [
    ("matrix_inversion", dict(n=4, kappa=10, seed=1)),
    ("matrix_inversion", dict(example="zhang_ge_2005")),
    ("matrix_inversion_right", dict(n=3, kappa=20, seed=2)),
    ("pseudoinverse", dict(n=3, kappa=10, seed=3)),                 # left, 4 x 3
    ("pseudoinverse", dict(n=3, kappa=10, side="right", seed=4)),   # right, 3 x 4
    ("pseudoinverse", dict(rows=5, cols=2, kappa=5, seed=5)),       # left, 5 x 2
    ("sylvester", dict(n=3, k=2, kappa=10, seed=6)),
    ("sylvester", dict(n=4, kappa=10, seed=7)),
    ("lyapunov", dict(n=4, kappa=10, seed=8)),
    ("lyapunov", dict(example="hu_2020")),
    ("complex_inversion", dict(n=3, kappa=10, seed=9)),
]
IDS = [f"{k}-{'-'.join(f'{a}{b}' for a, b in kw.items())}" for k, kw in CONFIGS]
TIMES = [0.0, 0.37, 2.9]


def _make(key, kw, **extra):
    return make_problem(key, **{**kw, **extra})


def _dense_J(P, x, d, h=1e-3):
    """Dense Jacobian dE/dx at x (row vector x of shape (1, n_x)) by central FD."""
    J = np.empty((P.m, P.n_x))
    for j in range(P.n_x):
        e = np.zeros((1, P.n_x))
        e[0, j] = h
        J[:, j] = ((P.E(x + e, d) - P.E(x - e, d)) / (2 * h))[0]
    return J


def _scale(P, t):
    d = P.data(t)
    s = max(np.abs(v).max() for k, v in d.items() if k != "t")
    return 1.0 + s * (1.0 + np.abs(P.x_star(t)).max())


def test_all_keys_registered():
    probs = available()["problems"]
    for key in ["matrix_inversion", "matrix_inversion_right", "pseudoinverse", "sylvester",
                "lyapunov", "complex_inversion"]:
        assert key in probs


@pytest.mark.parametrize("key,kw", CONFIGS, ids=IDS)
def test_shapes_and_flattening(key, kw):
    P = _make(key, kw)
    rng = np.random.default_rng(0)
    x = P.x0(rng, 3)
    d = P.data(0.5)
    assert x.shape == (3, P.n_x)
    assert P.E(x, d).shape == (3, P.m)
    assert P.dEdt(x, 0.5).shape == (3, P.m)
    assert P.solve_J(x, d, rng.standard_normal((3, P.m))).shape == (3, P.n_x)
    assert P.JT(x, d, rng.standard_normal((3, P.m))).shape == (3, P.n_x)
    assert P.x_star(0.5).shape == (P.n_x,)
    assert "t" in d
    pickle.loads(pickle.dumps(P))           # joblib-friendly


@pytest.mark.parametrize("key,kw", CONFIGS, ids=IDS)
def test_x_star_zeroes_error(key, kw):
    P = _make(key, kw)
    for t in TIMES:
        e = P.E(P.x_star(t)[None], P.data(t))
        assert np.abs(e).max() < 1e-11 * _scale(P, t), (t, np.abs(e).max())


@pytest.mark.parametrize("key,kw", CONFIGS, ids=IDS)
def test_dEdt_matches_central_differences(key, kw):
    P = _make(key, kw)
    x = np.random.default_rng(1).uniform(-2, 2, size=(2, P.n_x))
    dt = 1e-5
    for t in TIMES[1:]:
        fd = (P.E(x, P.data(t + dt)) - P.E(x, P.data(t - dt))) / (2 * dt)
        ex = P.dEdt(x, t)
        assert np.abs(fd - ex).max() < 1e-6 * _scale(P, t), np.abs(fd - ex).max()


@pytest.mark.parametrize("key,kw", CONFIGS, ids=IDS)
def test_solve_J_and_JT_match_dense_jacobian(key, kw):
    P = _make(key, kw)
    rng = np.random.default_rng(2)
    t = 1.234
    d = P.data(t)
    x = rng.uniform(-2, 2, size=(2, P.n_x))
    J = _dense_J(P, x[:1], d)
    J2 = _dense_J(P, x[1:], d)
    assert np.allclose(J, J2, atol=1e-8 * np.abs(J).max())       # E is affine in x
    r = rng.standard_normal((2, P.m))
    dx = P.solve_J(x, d, r)
    ref = np.linalg.solve(J, r.T).T
    assert np.allclose(dx, ref, rtol=1e-7, atol=1e-9 * np.abs(ref).max())
    assert np.allclose(dx @ J.T, r, atol=1e-8 * (1 + np.abs(r).max()) * np.linalg.cond(J))
    v = rng.standard_normal((2, P.m))
    assert np.allclose(P.JT(x, d, v), v @ J, rtol=1e-8, atol=1e-8 * np.abs(J).max())


@pytest.mark.parametrize("key,kw", CONFIGS, ids=IDS)
def test_batch_consistency(key, kw):
    P = _make(key, kw)
    rng = np.random.default_rng(3)
    B, t = 3, 0.81
    d = P.data(t)
    x = rng.uniform(-2, 2, size=(B, P.n_x))
    r = rng.standard_normal((B, P.m))
    for f in (lambda xx, rr: P.E(xx, d), lambda xx, rr: P.dEdt(xx, t),
              lambda xx, rr: P.solve_J(xx, d, rr), lambda xx, rr: P.JT(xx, d, rr)):
        full = f(x, r)
        single = np.vstack([f(x[b:b + 1], r[b:b + 1]) for b in range(B)])
        assert np.allclose(full, single, rtol=1e-12, atol=1e-12 * (1 + np.abs(full).max()))


@pytest.mark.parametrize("key,kw", CONFIGS, ids=IDS)
def test_znn_euler_converges(key, kw):
    P = _make(key, kw, T=2.0)
    tau = 2e-3
    s = simulate(P, make_model("znn"), make_discretization("euler"), tau=tau, B=2, gamma=10.0,
                 seed=0)
    assert not s["diverged"].any()
    r0 = s["residual"][0]
    ss = mssre(s)
    assert np.all(ss < 1e-3 * r0), (ss, r0)
    assert np.all(s["sol_err"][-1] < 1e-2 * (1 + np.abs(P.x_star(2.0)).max()))
    # Euler with fixed gamma: steady residual is O(tau) -> halving tau halves it
    s2 = simulate(P, make_model("znn"), make_discretization("euler"), tau=tau / 2, B=2,
                  gamma=10.0, seed=0)
    ratio = ss / mssre(s2)
    assert np.all((ratio > 1.6) & (ratio < 2.4)), ratio


# ---------------------------------------------------------------------------
# problem-specific faithfulness checks
# ---------------------------------------------------------------------------
def test_matrix_inversion_paper_example():
    """Zhang & Ge (2005), Sec. V: A(t) = [[sin t, cos t], [-cos t, sin t]], A^{-1} = A^T."""
    P = make_problem("matrix_inversion", example="zhang_ge_2005")
    for t in TIMES:
        A = P.data(t)["A"]
        assert np.allclose(A, [[np.sin(t), np.cos(t)], [-np.cos(t), np.sin(t)]])
        assert np.allclose(P.x_star(t).reshape(2, 2), A.T)


def test_matrix_inversion_left_right_same_solution():
    Pl = make_problem("matrix_inversion", n=4, kappa=30, seed=11)
    Pr = make_problem("matrix_inversion_right", n=4, kappa=30, seed=11)
    for t in TIMES:
        assert np.allclose(Pl.x_star(t), Pr.x_star(t))
        assert np.allclose(Pl.x_star(t).reshape(4, 4) @ Pl.data(t)["A"], np.eye(4))
    assert Pl.capabilities() == {"matrix_inverse"}
    assert Pr.capabilities() == {"matrix_inverse_right"}


def test_matrix_inverse_capability_semantics():
    """'matrix_inverse': J dx = r  <=>  dX = A^{-1} R, so at x = x* solve_J(r) = X* R."""
    P = make_problem("matrix_inversion", n=4, seed=3)
    t, rng = 0.6, np.random.default_rng(0)
    xs = P.x_star(t)[None]
    r = rng.standard_normal((1, 16))
    assert np.allclose(P.solve_J(xs, P.data(t), r)[0],
                       (xs.reshape(4, 4) @ r.reshape(4, 4)).ravel())
    Pr = make_problem("matrix_inversion_right", n=4, seed=3)
    xs = Pr.x_star(t)[None]
    assert np.allclose(Pr.solve_J(xs, Pr.data(t), r)[0],
                       (r.reshape(4, 4) @ xs.reshape(4, 4)).ravel())


@pytest.mark.parametrize("kappa", [10.0, 100.0])
def test_condition_number_controlled(kappa):
    for key in ("matrix_inversion", "complex_inversion"):
        P = make_problem(key, n=5, kappa=kappa, seed=4)
        for t in TIMES:
            A = P.A(t)
            c = np.linalg.cond(A)
            assert kappa * 0.8 / 1.2 * 0.99 <= c <= kappa * 1.2 / 0.8 * 1.01, (key, c)


@pytest.mark.parametrize("kw", [dict(n=3, seed=3), dict(n=3, side="right", seed=4),
                                dict(rows=6, cols=3, seed=1), dict(rows=2, cols=5, seed=2)])
def test_pseudoinverse_is_moore_penrose(kw):
    P = make_problem("pseudoinverse", **kw)
    assert P.side == ("left" if P.rows >= P.cols else "right")
    for t in TIMES:
        A = P.data(t)["A"]
        assert A.shape == (P.rows, P.cols)
        X = P.x_star(t).reshape(P.cols, P.rows)
        assert np.allclose(X, np.linalg.pinv(A), atol=1e-10)
        # four Penrose conditions
        assert np.allclose(A @ X @ A, A)
        assert np.allclose(X @ A @ X, X)
        assert np.allclose((A @ X).T, A @ X)
        assert np.allclose((X @ A).T, X @ A)


def test_pseudoinverse_side_errors():
    with pytest.raises(ValueError):
        make_problem("pseudoinverse", rows=3, cols=5, side="left")
    with pytest.raises(ValueError):
        make_problem("pseudoinverse", rows=3)


def test_sylvester_matches_scipy_and_paper_form():
    """E = A X - X B + C (Zhang, Jiang & Wang 2002); equals scipy's A X + X B' = C'
    with B' = -B, C' = -C; solve_J equals scipy.linalg.solve_sylvester per batch member."""
    P = make_problem("sylvester", n=4, k=3, kappa=20, seed=12)
    rng = np.random.default_rng(5)
    t = 1.7
    d = P.data(t)
    X = P.x_star(t).reshape(4, 3)
    assert np.allclose(X, solve_sylvester(d["A"], -d["B"], -d["C"]))
    assert np.allclose(d["A"] @ X - X @ d["B"] + d["C"], 0, atol=1e-12)
    r = rng.standard_normal((3, 12))
    dx = P.solve_J(None, d, r)
    for b in range(3):
        assert np.allclose(dx[b].reshape(4, 3), solve_sylvester(d["A"], -d["B"], r[b].reshape(4, 3)))
    # generator: symmetric parts of A and -B are positive definite (unique solution)
    for M in (d["A"], -d["B"]):
        assert np.linalg.eigvalsh(0.5 * (M + M.T)).min() > 0.5


def test_lyapunov_structure():
    """A(t) Hurwitz, Q(t) SPD  =>  X*(t) symmetric positive definite."""
    P = make_problem("lyapunov", n=5, kappa=50, seed=13)
    for t in TIMES:
        d = P.data(t)
        A, Q = d["A"], d["Q"]
        assert np.linalg.eigvals(A).real.max() < 0
        assert np.allclose(Q, Q.T, atol=1e-12) and np.linalg.eigvalsh(Q).min() > 0
        X = P.x_star(t).reshape(5, 5)
        assert np.allclose(X, X.T, atol=1e-10)
        assert np.linalg.eigvalsh(0.5 * (X + X.T)).min() > 0
        # independent dense (Kronecker, row-major) solve of A^T X + X A = -Q
        K = np.kron(A.T, np.eye(5)) + np.kron(np.eye(5), A.T)
        assert np.allclose(np.linalg.solve(K, -Q.ravel()), X.ravel())


def test_lyapunov_hu2020_example():
    """Hu et al. (2020), Sec. 5.  The printed closed form is the transpose of the
    exact solution of the printed equation (M^T X + X M = -Q)."""
    from znnbench.problems.matrix import hu2020_solution
    P = make_problem("lyapunov", example="hu_2020")
    for t in np.linspace(0, 3, 7):
        d = P.data(t)
        c, s = np.cos(3 * t), np.sin(3 * t)
        assert np.allclose(d["A"], [[-1 - c / 2, s / 2], [s / 2, -1 + c / 2]])
        assert np.allclose(d["Q"], [[s, c], [-c, s]])
        X = P.x_star(t).reshape(2, 2)
        assert np.allclose(X, hu2020_solution(t).T, atol=1e-12)
        Xp = hu2020_solution(t)
        assert np.allclose(d["A"].T @ Xp + Xp @ d["A"] + d["Q"].T, 0, atol=1e-12)


def test_complex_embedding():
    P = make_problem("complex_inversion", n=3, kappa=10, seed=14)
    rng = np.random.default_rng(6)
    t = 0.9
    d = P.data(t)
    Ac = d["Ar"] + 1j * d["Ai"]
    assert np.abs(d["Ai"]).max() > 0.1                       # genuinely complex
    Xs = P.to_complex(P.x_star(t)[None])[0]
    assert np.allclose(Xs, np.linalg.inv(Ac))
    # layout: X~ = x.reshape(2n, n) = [Re X; Im X]
    Xt = P.x_star(t).reshape(6, 3)
    assert np.allclose(Xt[:3], Xs.real) and np.allclose(Xt[3:], Xs.imag)
    # E~ equals the real block form [[Ar, -Ai], [Ai, Ar]] X~ - [I; 0]
    x = rng.standard_normal((2, P.n_x))
    Ablk = np.block([[d["Ar"], -d["Ai"]], [d["Ai"], d["Ar"]]])
    rhs = np.vstack([np.eye(3), np.zeros((3, 3))])
    Eblk = np.stack([Ablk @ xx.reshape(6, 3) - rhs for xx in x]).reshape(2, -1)
    assert np.allclose(P.E(x, d), Eblk)
    # E~ equals the embedding of the complex residual, J^T is A^H
    Ec = Ac @ P.to_complex(x) - np.eye(3)
    assert np.allclose(P.E(x, d), P.from_complex(Ec))
    v = rng.standard_normal((2, P.m))
    assert np.allclose(P.JT(x, d, v), P.from_complex(Ac.conj().T @ P.to_complex(v)))
    assert P.capabilities() == {"complex_embedding"}


def test_factor_cache_tracks_data_values():
    """Cached factorisations must follow the data values (noisy data, new arrays)."""
    rng = np.random.default_rng(7)
    for key, kw in [("matrix_inversion", dict(n=3)), ("matrix_inversion_right", dict(n=3)),
                    ("pseudoinverse", dict(n=3)), ("pseudoinverse", dict(n=3, side="right")),
                    ("sylvester", dict(n=3, k=2)), ("lyapunov", dict(n=3)),
                    ("complex_inversion", dict(n=2))]:
        P = make_problem(key, **kw)
        d1 = P.data(0.3)
        d2 = {k: (v if k == "t" else v + 0.05 * rng.standard_normal(v.shape)) for k, v in d1.items()}
        x = rng.standard_normal((1, P.n_x))
        r = rng.standard_normal((1, P.m))
        for d in (d1, d2, d1, d2):
            J = _dense_J(P, x, d)
            assert np.allclose(P.solve_J(x, d, r) @ J.T, r, atol=1e-8), key


def test_data_noise_and_bd_regimes_run():
    """Perturbed data (non-symmetric Q, perturbed Gram matrices) and causal
    backward-difference dE/dt estimates go through every problem."""
    for key, kw in [("matrix_inversion", dict(n=3)), ("pseudoinverse", dict(n=2, side="right")),
                    ("sylvester", dict(n=2, k=3)), ("lyapunov", dict(n=3)),
                    ("complex_inversion", dict(n=2)), ("matrix_inversion_right", dict(n=2))]:
        P = make_problem(key, T=0.5, **kw)
        s = simulate(P, make_model("znn"), make_discretization("euler"), tau=5e-3, B=2,
                     gamma=20.0, info="bd2", data_noise=1e-4, seed=1)
        assert np.isfinite(s["residual"]).all() and not s["diverged"].any()
        assert s["residual"][-1].max() < s["residual"][0].min()


def test_gnn_runs_on_matrix_problems():
    """JT is wired correctly for the GNN: residual decreases (small gamma*tau)."""
    for key, kw in [("matrix_inversion", dict(n=3, kappa=3)), ("lyapunov", dict(n=2, kappa=3)),
                    ("complex_inversion", dict(n=2, kappa=3))]:
        P = make_problem(key, T=1.0, **kw)
        s = simulate(P, make_model("gnn"), make_discretization("euler"), tau=1e-3, B=1,
                     gamma=5.0, seed=2)
        assert s["residual"][-1, 0] < 0.5 * s["residual"][0, 0]


# ---------------------------------------------------------------------------
# added by the adversarial verification pass
# ---------------------------------------------------------------------------
def _literature_E(P, X, d):
    """Error matrices written out exactly as in the source papers (independent of
    matrix.py): Zhang & Ge 2005 above eq. (3); FUMI 2017 eq. (1.2); Chai et al.
    2020 eqs. (10), (3) (real case); Zhao et al. 2023 eq. (2); Hu et al. 2020 eq. (2)."""
    name = type(P).__name__
    if name == "MatrixInversion":
        return d["A"] @ X - np.eye(P.n)
    if name == "MatrixInversionRight":
        return X @ d["A"] - np.eye(P.n)
    if name == "Pseudoinverse":
        A = d["A"]
        return A.T @ A @ X - A.T if P.side == "left" else X @ A @ A.T - A.T
    if name == "Sylvester":
        return d["A"] @ X - X @ d["B"] + d["C"]
    if name == "Lyapunov":
        return d["A"].T @ X + X @ d["A"] + d["Q"]
    if name == "ComplexInversion":
        Ec = (d["Ar"] + 1j * d["Ai"]) @ (X[:P.n] + 1j * X[P.n:]) - np.eye(P.n)
        return np.vstack([Ec.real, Ec.imag])
    raise AssertionError(name)


@pytest.mark.parametrize("key,kw", CONFIGS, ids=IDS)
def test_error_function_is_literature_formula(key, kw):
    """E(x, d) equals the source paper's matrix error function for random X
    (pins the side/sign convention, which E(x*) = 0 alone cannot: e.g. AX - I and
    XA - I share the zero A^{-1})."""
    P = _make(key, kw)
    rng = np.random.default_rng(11)
    d = P.data(0.42)
    x = rng.uniform(-2, 2, size=(3, P.n_x))
    shape = {"Pseudoinverse": (P.cfg.get("cols"), P.cfg.get("rows")),
             "Sylvester": (getattr(P, "n", 0), getattr(P, "k", 0)),
             "ComplexInversion": (2 * getattr(P, "n", 0), getattr(P, "n", 0))}.get(
        type(P).__name__, (getattr(P, "n", 0), getattr(P, "n", 0)))
    ref = np.stack([_literature_E(P, xx.reshape(shape), d).ravel() for xx in x])
    assert np.allclose(P.E(x, d), ref, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("key,kw", CONFIGS, ids=IDS)
def test_linear_znn_continuous_time_exponential_decay(key, kw):
    """Continuous-time property (Zhang & Ge 2005, Thm 1 (i)): with the linear
    activation the design formula gives E(t) = E(0) exp(-gamma t) exactly.  RK4 with a
    small tau reproduces it to ~1e-10, which checks dEdt, solve_J and the data
    trajectory together over a whole horizon (not only locally)."""
    P = _make(key, kw, T=0.5)
    gam = 3.0
    s = simulate(P, make_model("znn"), make_discretization("rk4"), tau=2e-3, B=2, gamma=gam,
                 seed=0)
    pred = s["residual"][0] * np.exp(-gam * s["t"])[:, None]
    assert np.max(np.abs(s["residual"] - pred) / pred) < 1e-7


def test_trsyl_scale_contract():
    """LAPACK *TRSYL returns y with R y + y S = scale * F; the solution is y / scale.
    A wrapped trsyl that honours that contract with scale = 0.5 must not change the
    result (scipy 1.17's solve_sylvester would return a 4x too small answer here)."""
    from znnbench.problems import matrix as mm
    P = make_problem("sylvester", n=3, k=2, seed=21)
    d = P.data(0.3)
    r, u, s_, v, trsyl = mm._sylv_pair(d["A"], d["B"])

    def half_scale_trsyl(a, b, c, tranb="N"):
        y, scale, info = trsyl(a, b, c, tranb=tranb)
        return 0.5 * y, 0.5 * scale, info

    R = np.random.default_rng(0).standard_normal((2, 3, 2))
    Y = mm._sylvester_solve((r, u, s_, v, half_scale_trsyl), R)
    for b in range(2):
        assert np.allclose(d["A"] @ Y[b] - Y[b] @ d["B"], R[b], atol=1e-12)


def test_lyapunov_hu2020_initial_error():
    """Hu et al. (2020), Sec. 5 randomise E(0) (|e_ij(0)| <= 2), not X(0)."""
    P = make_problem("lyapunov", example="hu_2020")
    x0 = P.x0(np.random.default_rng(3), 200)
    E0 = P.E(x0, P.data(0.0))
    assert np.abs(E0).max() <= 2.0 + 1e-12
    assert np.abs(E0).max() > 1.8 and abs(E0.mean()) < 0.2       # spread over [-2, 2]
    # random instances keep X(0) ~ U[-2, 2]
    Pr = make_problem("lyapunov", n=3, seed=1)
    xr = Pr.x0(np.random.default_rng(3), 50)
    assert np.abs(xr).max() <= 2.0 and np.abs(xr).max() > 1.8
