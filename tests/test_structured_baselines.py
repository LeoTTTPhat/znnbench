"""Tests for structured ZNN models, continuation/Newton baselines and the
prediction-correction discretizations (area: structured_baselines)."""
import numpy as np
import pytest

from znnbench import available, make_discretization, make_model, make_problem, simulate
from znnbench.core import Context, Problem
from znnbench.discretizations.prediction_correction import (PredictionCorrection,
                                                           estimate_lipschitz)
from znnbench.metrics import empirical_order, mssre
from znnbench.models.structured import PENDING_MODELS, newton_schulz_init
from znnbench.problems._tv import TVMatrix
from znnbench.registry import PROBLEMS, _load_all


# ---------------------------------------------------------------------------
# tiny local test problems
# ---------------------------------------------------------------------------
class Cubic(Problem):
    """E(x, t) = M x + c x^3 - b(t), M fixed SPD: a strongly monotone gradient
    field (E = grad f, f = x'Mx/2 + c sum x^4/4 - b'x), J = M + 3c diag(x^2)."""

    family = "test-nonlinear"

    def __init__(self, n=3, c=0.3, seed=0, T=4.0):
        rng = np.random.default_rng(seed)
        Q = np.linalg.qr(rng.standard_normal((n, n)))[0]
        self.M = Q @ np.diag(np.linspace(1.0, 3.0, n)) @ Q.T
        self.c = c
        self.a, self.w = rng.uniform(0.5, 1.5, n), rng.uniform(0.5, 1.5, n)
        self.ph = rng.uniform(0, 2 * np.pi, n)
        self.n_x = self.m = n
        self.T = T
        self.name = f"cubic_n{n}"

    def data(self, t):
        return {"t": np.float64(t), "b": self.a * np.sin(self.w * t + self.ph)}

    def E(self, x, d):
        return x @ self.M.T + self.c * x ** 3 - d["b"]

    def dEdt(self, x, t):
        return np.broadcast_to(-self.a * self.w * np.cos(self.w * t + self.ph), x.shape).copy()

    def J(self, x):
        return self.M[None] + 3 * self.c * x[:, :, None] ** 2 * np.eye(self.n_x)[None]

    def solve_J(self, x, d, r):
        return np.linalg.solve(self.J(x), r[..., None])[..., 0]

    def JT(self, x, d, v):
        return np.einsum("bji,bj->bi", self.J(x), v)

    def x_star(self, t):
        x, d = np.zeros((1, self.n_x)), self.data(t)
        for _ in range(60):
            x = x - self.solve_J(x, d, self.E(x, d))
        return x[0]

    def x0(self, rng, B):
        return rng.uniform(-1, 1, (B, self.n_x))

    def capabilities(self):
        return {"gradient_field"}


class MatInv(Problem):
    """Local stand-in for problems/matrix.py: E = vec_row(A(t) X - I)."""

    family = "matrix-inversion"

    def __init__(self, n=3, kappa=4.0, freq=1.0, seed=0, T=4.0):
        self.A = TVMatrix(n, np.random.default_rng(seed), kappa=kappa, freq=freq)
        self.n, self.n_x, self.m, self.T = n, n * n, n * n, T
        self.name = f"matinv_n{n}"

    def data(self, t):
        return {"t": np.float64(t), "A": self.A(t)}

    def E(self, x, d):
        return (d["A"] @ x.reshape(-1, self.n, self.n) - np.eye(self.n)).reshape(x.shape[0], -1)

    def dEdt(self, x, t):
        return (self.A.dot(t) @ x.reshape(-1, self.n, self.n)).reshape(x.shape[0], -1)

    def solve_J(self, x, d, r):
        return np.linalg.solve(d["A"], r.reshape(-1, self.n, self.n)).reshape(r.shape[0], -1)

    def JT(self, x, d, v):
        return (d["A"].T @ v.reshape(-1, self.n, self.n)).reshape(v.shape[0], -1)

    def x_star(self, t):
        return np.linalg.inv(self.A(t)).ravel()

    def x0(self, rng, B):
        return self.x_star(0.0)[None] + 0.05 * rng.standard_normal((B, self.n_x))

    def capabilities(self):
        return {"matrix_inverse"}


TAUS = [4e-2, 2e-2, 1e-2, 5e-3]


def _order(P, model, disc, h=0.3, taus=TAUS, x0=None, info="analytic", **dk):
    errs = []
    for tau in taus:
        s = simulate(P, make_model(model), make_discretization(disc, **dk), tau=tau,
                     gamma=h / tau, x0=x0, info=info)
        errs.append(float(mssre(s)[0]))
    return empirical_order(taus, errs, floor=1e-14)[0], errs


@pytest.fixture(scope="module")
def cubic():
    P = Cubic()
    return P, P.x_star(0.0)[None]


@pytest.fixture(scope="module")
def linsys():
    return make_problem("linear_system", n=4, kappa=10, seed=1, T=4.0)


# ---------------------------------------------------------------------------
# registry / metadata
# ---------------------------------------------------------------------------
def test_registry_and_flags():
    for key in ["davidenko", "newton_flow", "znn_explicit_inverse", "znn_explicit_inverse_right"]:
        make_model(key)
    causal = {"pc_newton": False, "pc_gradient": False, "newton_tracking": False,
              "running_gradient": False, "pc_newton_causal": True, "pc_gradient_causal": True,
              "newton_tracking_causal": True}
    for key, c in causal.items():
        d = make_discretization(key)
        assert d.causal is c, key
        assert d.describe()["information"].endswith("(causal)" if c else "(non-causal)")
    assert make_model("newton_flow").uses_dEdt is False
    assert make_model("znn_explicit_inverse").requires == {"matrix_inverse"}
    assert make_model("znn_explicit_inverse_right").requires == {"matrix_inverse_right"}
    assert set(PENDING_MODELS) == {"inverse_free_znn"}   # documented, not registered
    assert "lcc_znn" in available()["models"]             # registered in models/structured_extra.py
    assert "inverse_free_znn" not in available()["models"]


# ---------------------------------------------------------------------------
# baselines (models)
# ---------------------------------------------------------------------------
def test_davidenko_preserves_residual(linsys):
    """Davidenko flow keeps E(x(t), t) constant (continuous time, RK4)."""
    P = linsys
    rng = np.random.default_rng(0)
    x = rng.uniform(-2, 2, (2, P.n_x))
    t, dt = 0.7, 1e-5
    ctx = Context(P, t, 1e-3, 0, x, np.zeros((2, 0)), P.data(t), P.E(x, P.data(t)), P.dEdt(x, t),
                  np.ones((2, 1)))
    xdot, _ = make_model("davidenko").rhs(ctx)
    dE = (P.E(x + dt * xdot, P.data(t + dt)) - P.E(x - dt * xdot, P.data(t - dt))) / (2 * dt)
    assert np.abs(dE).max() < 1e-6
    s = simulate(P, make_model("davidenko"), make_discretization("rk4"), tau=1e-3, T=2.0, B=2)
    r = s["residual"]
    assert np.abs(r - r[0]).max() / r[0].min() < 1e-8


def test_newton_flow_euler_h1_equals_causal_newton_tracking(cubic):
    P, _ = cubic
    tau = 1e-2
    a = simulate(P, make_model("newton_flow"), make_discretization("euler"), tau=tau,
                 gamma=1.0 / tau, B=2, T=1.0)
    b = simulate(P, make_model("znn"), make_discretization("newton_tracking_causal"), tau=tau,
                 B=2, T=1.0)
    assert np.allclose(a["residual"], b["residual"], rtol=1e-9, atol=1e-13)


# ---------------------------------------------------------------------------
# explicit matrix-inversion ZNN / Getz-Marsden dynamic inverter
# ---------------------------------------------------------------------------
def _ctx(P, x, t, gamma, tau=1e-3, dEdt=None):
    d = P.data(t)
    return Context(P, t, tau, 0, x, np.zeros((x.shape[0], 0)), d, P.E(x, d),
                   P.dEdt(x, t) if dEdt is None else dEdt, np.full((x.shape[0], 1), gamma))


@pytest.mark.parametrize("act", ["linear", "power_sigmoid"])
def test_explicit_inverse_formula(act):
    """rhs == -X A_dot X - gamma X Phi(A X - I)  (Getz-Marsden Eq. (1) for linear Phi)."""
    from znnbench import activations
    P, n, g, t = MatInv(), 3, 7.0, 0.4
    x = np.random.default_rng(1).standard_normal((2, n * n))
    xdot, _ = make_model("znn_explicit_inverse", activation=act).rhs(_ctx(P, x, t, g))
    X, A, Ad = x.reshape(2, n, n), P.A(t), P.A.dot(t)
    ref = -X @ Ad @ X - g * X @ activations.get(act)(A @ X - np.eye(n))
    assert np.allclose(xdot.reshape(2, n, n), ref, atol=1e-12)


def test_explicit_inverse_exact_solution_is_a_trajectory():
    P, t, dt = MatInv(), 1.3, 1e-5
    x = P.x_star(t)[None]
    xdot, _ = make_model("znn_explicit_inverse").rhs(_ctx(P, x, t, 10.0))
    fd = (P.x_star(t + dt) - P.x_star(t - dt)) / (2 * dt)        # d/dt A^{-1}
    assert np.allclose(xdot[0], fd, atol=1e-6)


def test_explicit_inverse_euler_h1_is_newton_schulz():
    """Constant A, h = gamma tau = 1: X+ = 2X - XAX (Zhang-Ma-Cai 2009); from the
    Pan-Schreiber start it converges quadratically."""
    P, n, tau = MatInv(freq=0.0, kappa=10.0), 3, 1e-2
    A = P.A(0.0)
    X0 = newton_schulz_init(A)
    assert np.abs(np.linalg.eigvals(np.eye(n) - A @ X0)).max() < 1
    x = X0.reshape(1, -1)
    xdot, _ = make_model("znn_explicit_inverse").rhs(_ctx(P, x, 0.0, 1.0 / tau, tau=tau))
    assert np.allclose((x + tau * xdot).reshape(n, n), 2 * X0 - X0 @ A @ X0, atol=1e-13)
    s = simulate(P, make_model("znn_explicit_inverse"), make_discretization("euler"), tau=tau,
                 gamma=1.0 / tau, T=40 * tau, x0=x)
    assert s["residual"][-1, 0] < 1e-12


def test_explicit_inverse_continuous_rate():
    """RK4 with tiny tau: residual decays like exp(-gamma t) near A^{-1}
    (linearised error dynamics E_dot = -gamma E - E A_dot X)."""
    P, g = MatInv(), 100.0
    out = {}
    for key in ["znn_explicit_inverse", "znn"]:
        s = simulate(P, make_model(key), make_discretization("rk4"), tau=1e-4, T=0.15, gamma=g,
                     x0=P.x0(np.random.default_rng(0), 1))   # near-inverse start of the test problem
        r, t = s["residual"][:, 0], s["t"]
        w = (r < 1e-3) & (r > 1e-9)
        out[key] = -np.polyfit(t[w], np.log(r[w]), 1)[0]
        assert s["sol_err"][-1, 0] < 1e-6
    assert abs(out["znn"] - g) < 1e-3 * g
    assert 0.9 * g < out["znn_explicit_inverse"] < 1.1 * g, out


def test_explicit_inverse_euler_order():
    slope, errs = _order(MatInv(), "znn_explicit_inverse", "euler")
    assert abs(slope - 2) < 0.3, (slope, errs)


def test_explicit_inverse_requires_capability(linsys):
    with pytest.raises(ValueError, match="matrix_inverse"):
        simulate(linsys, make_model("znn_explicit_inverse"), make_discretization("euler"),
                 tau=1e-2, T=0.05)


@pytest.mark.parametrize("key,model,side", [
    ("matrix_inversion", "znn_explicit_inverse", "left"),
    ("matrix_inversion_right", "znn_explicit_inverse_right", "right")])
def test_explicit_inverse_on_registered_matrix_problems(key, model, side):
    """Shared convention with problems/matrix.py (other area): 'matrix_inverse'
    <=> E = vec_row(A X - I), 'matrix_inverse_right' <=> E = vec_row(X A - I).
    The explicit model must reproduce the implicit ZNN closely near A^{-1}."""
    _load_all()
    if key not in PROBLEMS:
        pytest.skip(f"problem {key!r} not registered yet")
    P = make_problem(key, n=3, kappa=4.0, seed=2, T=1.0)
    M = make_model(model)
    assert M.requires <= P.capabilities()
    n, t = 3, 0.3
    d = P.data(t)
    x = np.random.default_rng(0).standard_normal((2, P.n_x))
    X = x.reshape(2, n, n)
    ref = (d["A"] @ X if side == "left" else X @ d["A"]) - np.eye(n)
    assert np.allclose(P.E(x, d), ref.reshape(2, -1), atol=1e-10)
    # explicit rhs at the exact solution == implicit ZNN rhs == d/dt A^{-1}
    xs = P.x_star(t)[None]
    ctx = _ctx(P, xs, t, 10.0)
    assert np.allclose(M.rhs(ctx)[0], make_model("znn").rhs(ctx)[0], atol=1e-9)
    x0 = newton_schulz_init(P.data(0.0)["A"]).reshape(1, -1)
    kw = dict(tau=1e-3, gamma=300.0, x0=x0)
    e_exp = mssre(simulate(P, M, make_discretization("euler"), **kw))[0]
    e_imp = mssre(simulate(P, make_model("znn"), make_discretization("euler"), **kw))[0]
    assert e_exp < 2 * e_imp + 1e-12 and e_exp < 1e-3, (e_exp, e_imp)


def test_explicit_inverse_side_mismatch(linsys):
    _load_all()
    if "matrix_inversion_right" not in PROBLEMS:
        pytest.skip("matrix_inversion_right not registered yet")
    P = make_problem("matrix_inversion_right", n=3, T=0.05)
    with pytest.raises(ValueError, match="matrix_inverse'"):
        simulate(P, make_model("znn_explicit_inverse"), make_discretization("euler"), tau=1e-2)


# ---------------------------------------------------------------------------
# prediction-correction discretizations: orders
# ---------------------------------------------------------------------------
def test_pc_newton_exact_on_affine_problem(linsys):
    for key in ["pc_newton", "newton_tracking"]:
        s = simulate(linsys, make_model("davidenko"), make_discretization(key), tau=2e-2, B=2)
        assert mssre(s).max() < 1e-12, key


@pytest.mark.parametrize("model", ["davidenko", "znn"])
def test_pc_newton_fourth_order_nonlinear(cubic, model):
    """Simonetto et al. 2016, Thm 2: NTT error O(h^4) (C = 1)."""
    P, xs0 = cubic
    slope, errs = _order(P, model, "pc_newton", x0=xs0, taus=TAUS[:3])
    assert 3.6 < slope < 4.4, (slope, errs)


@pytest.mark.parametrize("info,lo", [("bd1", 3.6), ("bd2", 3.6)])
def test_ant_approximate_derivative_keeps_order(cubic, info, lo):
    """ANT (Eq. (10)-(11)): backward-difference prediction, same O(h^4)."""
    P, xs0 = cubic
    slope, errs = _order(P, "davidenko", "pc_newton", x0=xs0, taus=TAUS[:3], info=info)
    assert lo < slope < 4.4, (slope, errs)


@pytest.mark.parametrize("disc,order,prob", [
    ("pc_newton_causal", 2, "linsys"), ("pc_newton_causal", 2, "cubic"),
    ("pc_gradient", 2, "cubic"), ("pc_gradient_causal", 2, "cubic"),
    ("newton_tracking", 2, "cubic"), ("newton_tracking_causal", 1, "linsys"),
    ("running_gradient", 1, "cubic")])
def test_pc_orders(cubic, linsys, disc, order, prob):
    P, x0 = (cubic if prob == "cubic" else (linsys, None))
    slope, errs = _order(P, "davidenko", disc, x0=x0)
    assert abs(slope - order) < 0.25, (slope, errs)
    d = make_discretization(disc)
    assert d.expected_order_nonlinear == order or prob == "linsys"


def test_noncausal_correction_beats_causal(cubic):
    P, xs0 = cubic
    for corr in ["newton", "gradient"]:
        e = {c: float(mssre(simulate(P, make_model("davidenko"),
                                     make_discretization(f"pc_{corr}{c}"), tau=1e-2,
                                     x0=xs0))[0]) for c in ["", "_causal"]}
        assert e[""] < e["_causal"], (corr, e)


def test_pc_gradient_with_znn_on_linear_system(linsys):
    """Generic (non-gradient-field) problem: correction uses -alpha J^T E."""
    slope, errs = _order(linsys, "znn", "pc_gradient")
    assert abs(slope - 2) < 0.25, (slope, errs)


def test_gradient_direction_and_step_size(cubic, linsys):
    d = make_discretization("pc_gradient")
    assert d._resolve_direction(cubic[0]) == "residual"
    d.reset()
    assert d._resolve_direction(linsys) == "normal"
    dat = linsys.data(0.5)
    L = estimate_lipschitz(linsys, np.zeros((1, linsys.n_x)), dat, "normal")
    assert abs(L / np.linalg.norm(dat["A"], 2) ** 2 - 1) < 1e-3
    P = cubic[0]
    x = np.random.default_rng(0).standard_normal((1, P.n_x))
    L = estimate_lipschitz(P, x, P.data(0.0), "residual")
    assert abs(L / np.linalg.eigvalsh(P.J(x)[0]).max() - 1) < 1e-3


# ---------------------------------------------------------------------------
# prediction-correction discretizations: information pattern
# ---------------------------------------------------------------------------
class SpyCubic(Cubic):
    def __init__(self):
        super().__init__()
        self.current, self.log = None, []

    def data(self, t):
        if self.current is not None:
            self.log.append((self.current, float(t)))
        return super().data(t)


class SpyDisc:
    """Wraps a PC discretizer; records f-calls and data() requests per step."""

    def __init__(self, inner, P, wrap_f):
        self.inner, self.P, self.wrap_f, self.fcalls = inner, P, wrap_f, []
        self.causal = inner.causal

    def reset(self):
        self.inner.reset()

    def step(self, k, t, tau, x, z, f):
        self.P.current = t
        try:
            return self.inner.step(k, t, tau, x, z, _FSpy(f, self.fcalls, k, t) if self.wrap_f else f)
        finally:
            self.P.current = None


class _FSpy:
    """Callable object (not a closure): hides simulate's vector-field closure
    from the PC schemes, so the bound variant really exercises bind() only."""

    def __init__(self, f, log, k, t):
        self.f, self.log, self.k, self.t = f, log, k, t

    def __call__(self, tt, xx, zz, kk, imp):
        self.log.append((self.k, self.t, kk, tt))
        return self.f(tt, xx, zz, kk, imp)


@pytest.mark.parametrize("key", ["pc_newton", "pc_newton_causal", "pc_gradient",
                                 "pc_gradient_causal", "newton_tracking_causal"])
@pytest.mark.parametrize("bound", [True, False])
def test_information_pattern(key, bound):
    """Causal variants request no data after t_k and call f only at (t_k, k);
    non-causal variants request exactly t_{k+1} data for the correction."""
    P = SpyCubic()
    inner = make_discretization(key)
    if bound:
        inner.bind(P)
    disc = SpyDisc(inner, P, wrap_f=bound)      # wrapping f hides simulate's closure
    tau = 0.05
    simulate(P, make_model("davidenko"), disc, tau=tau, T=0.5, B=1)
    later = [(tk, td) for tk, td in P.log if td > tk + 1e-12]
    if inner.causal:
        assert not later, later[:3]
        if bound:                      # bind-only path: asks the problem for t_k data
            assert P.log and all(abs(td - tk) < 1e-12 for tk, td in P.log)
        else:
            assert not P.log           # closure path uses simulate's measured sample
    else:
        assert later and all(abs(td - tk - tau) < 1e-12 for tk, td in later)
    for k, t, kk, tt in disc.fcalls:
        assert kk == k and tt == t     # model field only at the current sample


@pytest.mark.parametrize("key,expect", [("pc_newton_causal", 0.0), ("pc_gradient_causal", 0.0),
                                        ("pc_newton", 0.1), ("newton_tracking", 0.1)])
def test_information_pattern_bound_without_simulate(key, expect):
    """Explicit bind() path (no simulate closure): data requested only at
    t_k (causal) or exactly t_{k+1} (non-causal)."""
    P = SpyCubic()
    d = make_discretization(key).bind(P)
    x = P.x_star(0.0)[None] + 0.01
    calls = []

    def f(t, xx, zz, k, implicit):
        calls.append((t, k))
        return np.zeros_like(xx), zz

    P.current = 0.0
    d.step(0, 0.0, 0.1, x, np.zeros((1, 0)), f)
    P.current = None
    assert P.log and all(abs(td - expect) < 1e-12 for _, td in P.log), P.log
    assert all(c == (0.0, 0) for c in calls)


def test_pc_inside_quantized_wrapper(cubic):
    """The problem is still found when another wrapper closes over f."""
    try:
        from znnbench.wrappers import QuantizedDiscretizer
    except Exception:  # noqa: BLE001 - other area's module
        pytest.skip("znnbench.wrappers not available")
    P, xs0 = cubic
    disc = QuantizedDiscretizer(make_discretization("pc_newton"), bits=40, frac_bits=30,
                                quantize_derivative=True)
    s = simulate(P, make_model("davidenko"), disc, tau=2e-2, T=1.0, x0=xs0)
    assert mssre(s)[0] < 1e-6


def test_bound_and_discovered_paths_agree(cubic):
    P, xs0 = cubic
    for key in ["pc_newton", "pc_gradient_causal"]:
        a = simulate(P, make_model("znn"), make_discretization(key), tau=2e-2, T=1.0, x0=xs0)
        b = simulate(P, make_model("znn"), make_discretization(key).bind(P), tau=2e-2, T=1.0,
                     x0=xs0)
        assert np.allclose(a["residual"], b["residual"], rtol=1e-12, atol=1e-15), key


def test_unbound_outside_simulate_raises(cubic):
    P, xs0 = cubic
    d = make_discretization("pc_newton")

    def f(t, x, z, k, implicit):
        return np.zeros_like(x), z

    with pytest.raises(RuntimeError, match="bind"):
        d.step(0, 0.0, 0.1, xs0, np.zeros((1, 0)), f)
    x1, _ = d.bind(P).step(0, 0.0, 0.1, xs0, np.zeros((1, 0)), f)
    assert np.isfinite(x1).all()


def test_pc_with_data_noise_is_finite(cubic):
    P, xs0 = cubic
    for key in ["pc_newton", "pc_newton_causal"]:
        s = simulate(P, make_model("davidenko"), make_discretization(key), tau=1e-2, T=1.0,
                     x0=np.tile(xs0, (2, 1)), data_noise=1e-3, B=2)
        assert np.isfinite(mssre(s)).all() and mssre(s).max() < 0.05


def test_invalid_arguments():
    with pytest.raises(ValueError):
        PredictionCorrection("bfgs")
    with pytest.raises(ValueError):
        PredictionCorrection("gradient", direction="sideways")
    with pytest.raises(ValueError):
        PredictionCorrection("newton", C=0, predict=False)


# ---------------------------------------------------------------------------
# adversarial-verification additions
# ---------------------------------------------------------------------------
class MatInvRight(MatInv):
    """Local stand-in for the dual problem: E = vec_row(X A(t) - I)."""

    def E(self, x, d):
        return (x.reshape(-1, self.n, self.n) @ d["A"] - np.eye(self.n)).reshape(x.shape[0], -1)

    def dEdt(self, x, t):
        return (x.reshape(-1, self.n, self.n) @ self.A.dot(t)).reshape(x.shape[0], -1)

    def solve_J(self, x, d, r):
        R = r.reshape(-1, self.n, self.n)
        return np.linalg.solve(d["A"].T, R.transpose(0, 2, 1)).transpose(0, 2, 1).reshape(r.shape[0], -1)

    def capabilities(self):
        return {"matrix_inverse_right"}


@pytest.mark.parametrize("act", ["linear", "power_sigmoid"])
def test_explicit_inverse_right_formula(act):
    """Dual form at a random X: rhs == -X A_dot X - gamma Phi(X A - I) X (independent
    reference built from A, A_dot directly, not from ctx.E / ctx.dEdt)."""
    from znnbench import activations
    P, n, g, t = MatInvRight(), 3, 7.0, 0.4
    x = np.random.default_rng(2).standard_normal((2, n * n))
    xdot, _ = make_model("znn_explicit_inverse_right", activation=act).rhs(_ctx(P, x, t, g))
    X, A, Ad = x.reshape(2, n, n), P.A(t), P.A.dot(t)
    ref = -X @ Ad @ X - g * activations.get(act)(X @ A - np.eye(n)) @ X
    assert np.allclose(xdot.reshape(2, n, n), ref, atol=1e-12)
    # and the explicit dual model agrees with the implicit ZNN when X = A^{-1}
    xs = P.x_star(t)[None]
    ctx = _ctx(P, xs, t, g)
    assert np.allclose(make_model("znn_explicit_inverse_right", activation=act).rhs(ctx)[0],
                       make_model("znn", activation=act).rhs(ctx)[0], atol=1e-9)


def test_explicit_inverse_error_dynamics():
    """E = AX - I under the explicit model obeys E_dot = -gamma E - gamma E^2 - E A_dot X
    (Getz-Marsden flow, linear Phi): checked by a central difference along the flow."""
    P, n, g, t, dt = MatInv(), 3, 5.0, 0.9, 1e-6
    rng = np.random.default_rng(4)
    x = P.x_star(t)[None] + 0.1 * rng.standard_normal((1, n * n))
    M = make_model("znn_explicit_inverse")
    xdot = M.rhs(_ctx(P, x, t, g))[0]
    E = lambda xx, tt: P.E(xx, P.data(tt)).reshape(n, n)
    fd = (E(x + dt * xdot, t + dt) - E(x - dt * xdot, t - dt)) / (2 * dt)
    Em, X, Ad = E(x, t), x.reshape(n, n), P.A.dot(t)
    assert np.allclose(fd, -g * Em - g * Em @ Em - Em @ Ad @ X, atol=1e-7)


def test_newton_flow_continuous_time():
    """Continuous Newton flow: d/dt E = -gamma E on a frozen problem (exact exponential
    decay under RK4); on a moving problem the lag floor scales like 1/gamma."""
    P0 = make_problem("linear_system", n=4, kappa=10, seed=1, freq=0.0, T=1.0)
    s = simulate(P0, make_model("newton_flow"), make_discretization("rk4"), tau=1e-3, gamma=5.0, B=2)
    r = s["residual"]
    assert np.allclose(r, r[0] * np.exp(-5.0 * s["t"])[:, None], rtol=1e-9)
    P = make_problem("linear_system", n=4, kappa=10, seed=1, T=3.0)
    e = [float(mssre(simulate(P, make_model("newton_flow"), make_discretization("rk4"), tau=1e-3,
                              gamma=g))[0]) for g in (10.0, 100.0)]
    assert 7.0 < e[0] / e[1] < 13.0, e


def test_gradient_direction_on_registered_unconstrained_opt():
    """tv_unconstrained_opt (E = grad f, family 'unconstrained-optimization') must get
    Simonetto's Eq. (8) step x - alpha E, not the J^T E fallback (verification fix)."""
    _load_all()
    if "tv_unconstrained_opt" not in PROBLEMS:
        pytest.skip("tv_unconstrained_opt not registered yet")
    # kappa = 2 keeps h inside Thm 1(ii)'s O(h^2) range, condition (17): rho^C sigma < 1
    P = make_problem("tv_unconstrained_opt", variant="quadratic", n=4, kappa=2, seed=0, T=4.0)
    d = make_discretization("pc_gradient")
    assert d._resolve_direction(P) == "residual"
    d.reset()
    assert d._resolve_direction(make_problem("linear_system", n=3)) == "normal"
    xs0 = P.x_star(0.0)[None]
    slope, errs = _order(P, "davidenko", "pc_gradient", x0=xs0, taus=[2e-2, 1e-2, 5e-3])
    assert abs(slope - 2) < 0.3, (slope, errs)


def test_causal_correction_uses_measured_sample_under_jitter_wrapper(cubic):
    """With a data wrapper that sets d['t'] != t_k (jitter) and data_noise > 0, the causal
    correction must still use simulate's measured sample (the one the model sees), so
    running Newton == newton_flow + Euler at h = 1 exactly (verification fix)."""
    try:
        from znnbench.wrappers import JitteredProblem
    except Exception:  # noqa: BLE001 - other area's module
        pytest.skip("znnbench.wrappers not available")
    P0, xs0 = cubic
    P = JitteredProblem(P0, jitter_bound=1e-3, seed=1)
    tau = 1e-2
    kw = dict(tau=tau, T=0.5, B=2, x0=np.tile(xs0, (2, 1)), data_noise=1e-2, seed=5)
    a = simulate(P, make_model("newton_flow"), make_discretization("euler"), gamma=1.0 / tau, **kw)
    b = simulate(P, make_model("davidenko"), make_discretization("newton_tracking_causal"), **kw)
    assert np.allclose(a["residual"], b["residual"], rtol=1e-9, atol=1e-13)


# --- Simonetto et al. 2016, Sec. IV-A scalar example, Eq. (35) -------------------
_W, _KAP, _MU = 0.02 * np.pi, 7.5, 1.75          # text values (Fig. 1 caption says mu = 1.5)


def _sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def _gx(x, t):
    return x - np.cos(_W * t) + _KAP * _MU * _sig(_MU * x)


def _gxx(x):
    return 1.0 + _KAP * _MU ** 2 * _sig(_MU * x) * (1.0 - _sig(_MU * x))


class Simonetto35(Problem):
    """f(x; t) = (x - cos(w t))^2 / 2 + kappa log(1 + exp(mu x)), Eq. (35)-(36)."""

    family = "test-scalar"
    n_x = m = 1
    name = "simonetto2016_eq35"
    T = 150.0

    def data(self, t):
        return {"t": np.float64(t), "c": np.array([np.cos(_W * t)])}

    def E(self, x, d):
        return x - d["c"] + _KAP * _MU * _sig(_MU * x)

    def dEdt(self, x, t):
        return np.full_like(x, _W * np.sin(_W * t))           # (36b)

    def solve_J(self, x, d, r):
        return r / _gxx(x)                                    # (36a)

    def JT(self, x, d, v):
        return v * _gxx(x)

    def x_star(self, t):
        x = 0.0
        for _ in range(60):
            x -= _gx(x, t) / _gxx(x)
        return np.array([x])

    def x0(self, rng, B):
        return np.zeros((B, 1))

    def capabilities(self):
        return {"gradient_field"}


def _simonetto_reference(alg, h, N, n_corr=1, step=0.2, approx=False):
    """Plain transcription of Algorithms 1-4 (Simonetto et al. 2016), x0 = 0."""
    x, out = 0.0, []
    for k in range(N):
        tk = k * h
        if approx:      # (10); no t_{-1} sample at k = 0 (same convention as info='bd1')
            gtx = (_gx(x, tk) - _gx(x, tk - h)) / h if k >= 1 else 0.0
        else:
            gtx = _W * np.sin(_W * tk)
        xp = x if alg == "RG" else x - h * gtx / _gxx(x)                     # (7) / (11)
        for _ in range(n_corr):
            if alg == "NTT":
                xp = xp - _gx(xp, tk + h) / _gxx(xp)                           # (9)
            else:
                xp = xp - step * _gx(xp, tk + h)                               # (8)
        x = xp
        out.append(x)
    return np.array(out)


@pytest.mark.parametrize("alg,key,kw,info,level", [
    ("GTT", "pc_gradient", dict(alpha=0.2), "analytic", (1e-6, 1e-4)),
    ("NTT", "pc_newton", {}, "analytic", (1e-12, 1e-9)),
    ("AGT", "pc_gradient", dict(alpha=0.2), "bd1", (1e-6, 1e-4)),
    ("ANT", "pc_newton", {}, "bd1", (1e-12, 1e-9)),
    ("RG", "running_gradient", dict(alpha=0.2), "analytic", (1e-3, 3e-2))])
def test_reproduces_simonetto_scalar_example(alg, key, kw, info, level):
    """ZNN-Bench (davidenko + PC scheme) == an independent transcription of
    Algorithms 1-4 on Eq. (35) with the paper's h = 0.1, gamma = 0.2, x0 = 0, to
    round-off; worst-case steady error in the band of the paper's Fig. 1/2
    (RG ~1e-2, GTT/AGT ~1e-5, NTT ~1e-10)."""
    P, h = Simonetto35(), 0.1
    N = int(round(P.T / h))
    ref = _simonetto_reference("NTT" if "N" in alg[:2] and alg != "RG" else
                               ("RG" if alg == "RG" else "GTT"), h, N, approx=alg[0] == "A")
    s = simulate(P, make_model("davidenko"), make_discretization(key, **kw), tau=h,
                 x0=np.zeros((1, 1)), info=info)
    xs = np.array([P.x_star((k + 1) * h)[0] for k in range(N)])
    zb_err = s["sol_err"][1:, 0]
    assert np.allclose(zb_err, np.abs(ref - xs), rtol=0, atol=1e-12)
    worst = zb_err[N // 2:].max()
    assert level[0] < worst < level[1], worst


@pytest.mark.parametrize("causal", [False, True])
def test_prediction_alone_is_forward_euler(linsys, causal):
    """Consistency of the prediction step: with C = 0 corrections every PC scheme is
    exactly forward Euler on the paired model (one-step map, characteristic
    polynomial zeta - 1, trivially zero-stable)."""
    kw = dict(tau=1e-2, T=0.5, gamma=20.0, B=2)
    a = simulate(linsys, make_model("znn"), make_discretization("euler"), **kw)
    b = simulate(linsys, make_model("znn"), PredictionCorrection("newton", C=0, causal=causal), **kw)
    assert np.array_equal(a["residual"], b["residual"])


# ---------------------------------------------------------------------------
# integration (smoke-matrix) regression
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key", ["pc_gradient", "pc_gradient_causal"])
def test_pc_gradient_outside_domain_is_divergence_not_error(key):
    """A state outside the domain of E (ln(x1), x1 <= 0, in nonlinear_equations)
    must be recorded as divergence by simulate, not raise from the step-size
    estimate.  Found by experiments/smoke_matrix.py: GNN + pc_gradient, B=2."""
    from znnbench.discretizations.prediction_correction import estimate_lipschitz
    P = make_problem("nonlinear_equations")
    x = np.array([[2.0, 0.5, -1.0, 0.0], [-1.0, 0.5, -1.0, 0.0]])   # row 1: ln(-1) = NaN
    d = P.data(0.3)
    L_mixed = estimate_lipschitz(P, x, d, "normal")
    L_good = estimate_lipschitz(P, x[:1], d, "normal")
    assert np.isfinite(L_mixed) and L_mixed == pytest.approx(L_good)
    assert np.isnan(estimate_lipschitz(P, x[1:], d, "normal"))
    x0 = np.array([[-1.0, 0.5, -1.0, 0.0]])                          # starts outside the domain
    with np.errstate(all="ignore"):
        s = simulate(P, make_model("znn"), make_discretization(key), tau=1e-2, T=0.1, x0=x0)
        assert s["diverged"].all()
        s = simulate(P, make_model("gnn"), make_discretization(key), tau=1e-2, T=1.0, B=2,
                     gamma=10, seed=0, info="bd2", data_noise=1e-4)    # the smoke-matrix cell
    assert s["diverged"].shape == (2,)
