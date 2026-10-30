"""Tests for the structured per-step FLOP model (znnbench/costs.py).

Three kinds of evidence:
1. hand counts (LAPACK kernels, whole steps on tiny instances);
2. a ufunc-level FLOP counter (ndarray subclass) that observes what the
   problem/model/discretizer code actually executes and must equal the
   analytic per-call counts;
3. agreement of call counts, row counts, factorisations, kinematics and
   problem-side FLOPs with simulate runs (``empirical_step_cost``) and with
   ``wrappers.profile_step_cost``.
"""
import warnings

import numpy as np
import pytest

from znnbench import make_discretization, make_model, make_problem
from znnbench import costs as C
from znnbench.core import Context
from znnbench.wrappers import (CountingProblem, EstimatedDerivativeProblem, QuantizedDiscretizer,
                               dense_flops, profile_step_cost)

TAU_EXACT = 2.0 ** -6        # k tau + tau == (k + 1) tau exactly: stage samples are reused


# ---------------------------------------------------------------------------
# ufunc-level FLOP counter
# ---------------------------------------------------------------------------
class _Count:
    flops = 0.0


class FA(np.ndarray):
    """ndarray subclass counting 1 FLOP per element of every floating ufunc
    result (reductions: inputs - outputs), 2 * out.size * q for matmul and
    2 FLOPs per multiply-add for einsum with a summed index."""

    def __array_ufunc__(self, ufunc, method, *inputs, **kw):
        raw = [np.asarray(i) if isinstance(i, FA) else i for i in inputs]
        if isinstance(kw.get("where"), FA):
            kw["where"] = np.asarray(kw["where"])
        if "out" in kw:
            kw["out"] = tuple(np.asarray(o) if isinstance(o, FA) else o for o in kw["out"])
        res = getattr(ufunc, method)(*raw, **kw)
        out = res[0] if isinstance(res, tuple) else res
        if isinstance(out, np.ndarray) and np.issubdtype(out.dtype, np.inexact):
            if method == "__call__" and np.iscomplexobj(out):
                # complex convention (spec Sec. 2): complex * complex 6, + 2; real * complex 2,
                # real + complex 1; complex GEMM 8 per multiply-add
                both = all(np.iscomplexobj(i) for i in raw)
                if ufunc is np.matmul:
                    w = (8.0 if both else 4.0) * np.shape(raw[0])[-1]
                elif ufunc in (np.add, np.subtract):
                    w = 2.0 if both else 1.0
                else:
                    w = 6.0 if both else 2.0
                _Count.flops += w * out.size
            elif method == "__call__":
                q = np.shape(raw[0])[-1] if ufunc is np.matmul else 1
                _Count.flops += (2.0 * q if ufunc is np.matmul else 1.0) * out.size
            elif method == "reduce":
                _Count.flops += np.size(raw[0]) - out.size
        elif method == "__call__" and isinstance(out, np.floating):
            _Count.flops += 1
        return res.view(FA) if isinstance(res, np.ndarray) else res

    def __array_function__(self, func, types, args, kwargs):
        if func is np.einsum:
            subs, ops = args[0], [np.asarray(a) for a in args[1:]]
            ins, out = subs.split("->")
            sizes = {}
            for term, op in zip(ins.split(","), ops):
                sizes.update(zip(term, op.shape))
            tot = float(np.prod(list(sizes.values())))
            summed = any(c not in out for c in sizes)
            _Count.flops += 2 * tot if summed else tot * (len(ops) - 1)
            return np.einsum(subs, *ops, **kwargs).view(FA)

        def plain(a):
            if isinstance(a, FA):
                return np.asarray(a)
            if isinstance(a, (list, tuple)):
                return type(a)(plain(b) for b in a)
            return a
        res = func(*[plain(a) for a in args], **kwargs)
        if func is np.linalg.norm:            # real 2-norm: x*x, sums, sqrt = 2 * size
            _Count.flops += 2.0 * np.size(args[0])
        elif func is np.clip:                 # the clip ufunc
            _Count.flops += np.size(res)
        elif func is np.cross:                # 2 mults + 1 sub per component
            _Count.flops += 3.0 * np.size(res)
        return res.view(FA) if isinstance(res, np.ndarray) else res


def fa(a):
    return np.asarray(a, dtype=float).view(FA)


def counted(fn):
    _Count.flops = 0.0
    fn()
    return _Count.flops


class _NumpyProxy:
    """Module-level numpy stand-in whose asarray keeps the counting subclass
    (the activations call np.asarray on their input)."""

    def __getattr__(self, k):
        return getattr(np, k)

    @staticmethod
    def asarray(a, dtype=None, **kw):
        return np.asanyarray(a, dtype=dtype)

    @staticmethod
    def array(a, dtype=None, copy=True, **kw):
        return np.array(a, dtype=dtype, copy=copy, subok=True)

    class _Linalg:
        """np.linalg with a subclass-preserving real row norm (same arithmetic
        as numpy's norm for real input: x*x, row sums, sqrt)."""

        def __getattr__(self, k):
            return getattr(np.linalg, k)

        @staticmethod
        def norm(x, axis=None, keepdims=False, **kw):
            return np.sqrt(np.add.reduce(x * x, axis=axis, keepdims=keepdims))

    linalg = _Linalg()


@pytest.fixture
def counting_numpy(monkeypatch):
    import znnbench.activations as acts
    import znnbench.discretizations.consistent as cons
    import znnbench.discretizations.inverse_free_dtznn as ifd
    import znnbench.models.structured_extra as sx
    import znnbench.models.variants as var
    import znnbench.problems.matrix as pm
    for mod in (acts, var, sx, cons, ifd):
        monkeypatch.setattr(mod, "np", _NumpyProxy())
    monkeypatch.setattr(pm, "_mat", lambda x, r, c: np.asanyarray(x).reshape(x.shape[0], r, c))
    return monkeypatch


def _counting_problem(P):
    """Memoised data arrays of the matrix problems become counting arrays."""
    if hasattr(type(P), "_memo"):
        orig = P._memo

        def memo(tag, t, fn):
            v = orig(tag, t, fn)
            return fa(v) if np.issubdtype(np.asarray(v).dtype, np.floating) else v
        P._memo = memo
    return P


# ---------------------------------------------------------------------------
# 1. hand counts
# ---------------------------------------------------------------------------
def test_lapack_kernel_counts_by_hand():
    # 2 x 2 LU: l21 = a21/a11 (1), u22 = a22 - l21 a12 (2)
    assert C.getrf(2) == 3 and C.getrf(1) == 0
    # forward y2 = b2 - l21 y1 (2); backward x2 = y2/u22 (1), x1 = (y1 - u12 x2)/u11 (3)
    assert C.getrs(2, 1) == 6 and C.getrs(3, 4) == 4 * (2 * 9 - 3)
    # 2 x 2 Cholesky: sqrt(a11), l21 = a21/l11, l22 = sqrt(a22 - l21^2): 1 + 1 + 3
    assert C.potrf(2) == 5 and C.potrs(3, 2) == 2 * 9 * 2
    assert C.gemm(2, 3, 4) == 48 and C.gemm(1, 1, 1, cplx=True) == 8
    assert C.getrf(2, cplx=True) == 6 * 2 + 2 * 1           # 2 complex mults, 1 complex add
    assert C.schur_real(2) == 200 and C.trsyl(2, 3) == 30
    assert C.getrf(128) == pytest.approx(2 / 3 * 128 ** 3, rel=1e-2)


def test_headline_matrix_inversion_n128_dense_vs_structured():
    """The motivating example: dense_flops charges (2/3) n_x^3 per solve."""
    P = make_problem("matrix_inversion", n=128)
    pc = C.problem_cost(P)
    dense = dense_flops(P.n_x, P.m, 1)["solve_J"]
    assert dense == pytest.approx(2.93e12, rel=0.01)
    assert pc.factor() == pytest.approx(1.40e6, rel=0.01)          # one 128 x 128 LU
    assert pc.solve_J(1)["solve"] == 128 * (2 * 128 ** 2 - 128)      # n right-hand sides
    st = C.step_flops(P, make_model("znn"), make_discretization("euler"), B=1)
    assert st["dense_equivalent"] / st["total"] > 1e5


def test_hand_counted_step_matrix_inversion_znn_euler():
    n, B = 3, 2
    nx = n * n
    E = 2 * B * n ** 3 + B * n * n          # A @ X - I
    dEdt = 2 * B * n ** 3                   # A_dot @ X
    lu = 2 / 3 * n ** 3 - n ** 2 / 2 - n / 6   # one LU per step (cached for the batch)
    trs = B * n * (2 * n * n - n)           # B*n right-hand sides
    rate = 2 * B + 2 * B * nx               # g = gamma g(t), -g, -g*E, r - dE/dt
    euler = 4 * B * nx                      # tau*xdot - (-1)*x, / 1 ; z empty
    st = C.step_flops(make_problem("matrix_inversion", n=n), make_model("znn"),
                      make_discretization("euler"), B=B)
    assert st["total"] == pytest.approx(E + dEdt + lu + trs + rate + euler)
    assert st["calls"] == {"data": 1, "E": 1, "dEdt": 1, "solve_J": 1, "JT": 0}
    assert st["factorizations"] == 1 and st["by_category"]["data"] > 0
    assert st["total"] < st["total"] + st["by_category"]["data"]


def test_hand_counted_step_linear_system_gnn():
    n, B = 2, 1
    E = 2 * B * n * n + B * n               # x @ A^T - b
    JT = 2 * B * n * n                      # v @ A
    gnn = B + B * n                         # -gamma, * J^T E   (linear activation: 0)
    euler = 4 * B * n
    P, M, D = make_problem("linear_system", n=n), make_model("gnn"), make_discretization("euler")
    st = C.step_flops(P, M, D, B=B)
    assert st["total"] == E + JT + gnn + euler
    assert st["calls"]["dEdt"] == 0          # the GNN ignores dE/dt (algorithmic view)
    code = C.step_flops(P, M, D, B=B, skip_unused_dEdt=False)
    assert code["total"] == st["total"] + E  # simulate still evaluates it (same cost as E)
    assert code["calls"]["dEdt"] == 1


def test_explicit_inverse_znn_has_no_solve():
    n, B = 4, 3
    P = make_problem("matrix_inversion", n=n)
    D = make_discretization("euler")
    imp = C.step_flops(P, make_model("znn"), D, B=B)
    exp = C.step_flops(P, make_model("znn_explicit_inverse"), D, B=B)
    assert exp["calls"]["solve_J"] == 0 and exp["factorizations"] == 0
    # model work: rate + the batched product X (r - dE/dt): 2 B n^3
    assert exp["by_category"]["model"] == imp["by_category"]["model"] + 2 * B * n ** 3
    assert exp["by_category"]["factor"] == 0 and exp["by_category"]["solve"] == 0


def test_discretization_call_structure():
    P = make_problem("matrix_inversion", n=3)
    M = make_model("znn")
    B, nx = 2, 9
    rk4 = C.step_flops(P, M, make_discretization("rk4"), B=B)
    assert rk4["calls"]["E"] == 4 and rk4["calls"]["data"] == 4 and rk4["calls"]["solve_J"] == 4
    assert rk4["factorizations"] == 2           # t+tau/2 (twice) and t+tau reused at the next t_k
    assert rk4["by_category"]["disc"] == 13 * B * nx
    for key, s in [("euler", 1), ("taylor_zhang", 3), ("zead_5i", 4), ("zead_11i_chen2023", 10)]:
        st = C.step_flops(P, M, make_discretization(key), B=B)
        assert st["calls"]["E"] == 1 and st["by_category"]["disc"] == (2 * s + 2) * B * nx
    for Cc in (1, 3):
        pc = C.step_flops(P, M, make_discretization("pc_newton", C=Cc), B=B)
        assert pc["calls"]["E"] == 1 + Cc and pc["calls"]["solve_J"] == 1 + Cc
        assert pc["calls"]["data"] == 2 and pc["factorizations"] == 1
        pcc = C.step_flops(P, M, make_discretization("pc_newton_causal", C=Cc), B=B)
        assert pcc["calls"]["data"] == 1 and pcc["factorizations"] == 1
    gn = C.step_flops(P, M, make_discretization("pc_gradient", C=2, alpha=0.1, direction="normal"),
                      B=B)
    assert gn["calls"]["JT"] == 2 and gn["calls"]["solve_J"] == 1
    nt = C.step_flops(P, M, make_discretization("newton_tracking_causal", C=2), B=B)
    assert nt["calls"]["dEdt"] == 0 and nt["calls"]["E"] == 2      # the model is not used
    br = C.step_flops(make_problem("linear_system", n=4), M,
                      make_discretization("broyden_dtznn_zhang2012"), B=B)
    assert br["calls"]["solve_J"] == 0 and br["calls"]["E"] == 3   # secant pair + model E
    cd = C.step_flops(P, M, make_discretization("cdtznn_mahuang2025"), B=B)
    assert cd["calls"]["solve_J"] == 0 and cd["calls"]["JT"] == 3
    assert cd["rows"]["JT"] == 3 * B * P.m                          # dense J from J^T e_i
    assert cd["rows"]["dEdt"] == B + 2 * B * nx                     # d_t J M by x-differences


def test_reuse_modes_data_noise_and_exact_fraction():
    P, M, D = make_problem("matrix_inversion", n=3), make_model("znn"), make_discretization("rk4")
    ideal = C.step_flops(P, M, D, reuse="ideal")
    none = C.step_flops(P, M, D, reuse="none")
    assert ideal["factorizations"] == 2 and none["factorizations"] == 3
    noisy = C.step_flops(P, M, D, data_noise=True)
    assert noisy["factorizations"] == 4          # every stage sample is a new realisation
    assert C.float_reuse_fraction(TAU_EXACT, 0, 500) == 1.0
    p = C.float_reuse_fraction(0.01, 0, 1000)
    assert 0.0 < p < 1.0
    ex = C.step_flops(P, M, D, reuse="exact", tau=0.01, k_range=(1, 1001))
    assert ex["factorizations"] == pytest.approx(2 * p + 3 * (1 - p))
    with pytest.raises(ValueError):
        C.step_flops(P, M, D, reuse="exact")


def test_scaling_exponents():
    ns = [16, 32, 64, 128]
    D, M = make_discretization("euler"), make_model("znn")
    tot, fac, dense = [], [], []
    for n in ns:
        st = C.step_flops(make_problem("matrix_inversion", n=n), M, D, B=1)
        tot.append(st["total"])
        fac.append(st["by_category"]["factor"] + st["by_category"]["solve"])
        dense.append(st["dense_equivalent"])
    assert 2.9 < C.scaling_exponent(ns, tot) < 3.05
    assert 2.9 < C.scaling_exponent(ns, fac) < 3.05          # LU + n^3 triangular solves
    assert 5.9 < C.scaling_exponent(ns, dense) < 6.05        # dense model: (n^2)^3
    assert all(b > a for a, b in zip(tot, tot[1:]))
    sylv = [C.problem_cost(make_problem("sylvester", n=n)).factor() for n in ns]
    assert C.problem_cost(make_problem("sylvester", n=4)).factor() == 2 * 25 * 64 + 16
    assert 2.95 < C.scaling_exponent(ns, sylv) < 3.05
    # gznn assembles J^T J for the n^2 x n^2 Jacobian: n^6
    gz = [C.step_flops(make_problem("matrix_inversion", n=n), make_model("gznn"), D)["total"]
          for n in (4, 8, 16)]
    assert 5.5 < C.scaling_exponent([4, 8, 16], gz) < 6.1
    # B trajectories share one factorisation: the per-trajectory cost decreases with B
    P = make_problem("matrix_inversion", n=32)
    per = [C.step_flops(P, M, D, B=B)["linalg"] / B for B in (1, 4, 16)]
    assert per[0] > per[1] > per[2]


def test_include_data_and_tables():
    P = make_problem("matrix_inversion", n=5)
    st = C.step_flops(P, make_model("znn"), make_discretization("euler"), include_data=True)
    gen = C.tvmatrix(5) + C.tvmatrix(5, deriv=True)          # A(t_k) and A_dot(t_k), memoised
    assert st["by_category"]["data"] == gen
    assert st["data_generations"] == {"A": 1, "dA": 1}
    ls = C.step_flops(make_problem("linear_system", n=3), make_model("znn"),
                      make_discretization("rk4"))             # no memo: per call
    assert ls["data_generations"]["A"] == 4 and ls["data_generations"]["dA"] == 4
    tab = C.problem_cost(P).table(B=2)
    assert tab["solve_J"] == C.getrf(5) + C.getrs(5, 10) and tab["factor"] == C.getrf(5)


def test_setup_cost_of_auto_step_size():
    P = make_problem("tv_unconstrained_opt")
    D = make_discretization("pc_gradient", C=2)
    st = C.step_flops(P, make_model("davidenko"), D)
    assert st["setup"]["calls"]["E"] == 60 and "JT" not in st["setup"]["calls"]  # gradient field
    am = C.step_flops(P, make_model("davidenko"), D, amortize_setup=100)
    assert am["total"] == pytest.approx(st["total"] + st["setup"]["flops"] / 100)
    assert C.step_flops(P, make_model("davidenko"),
                        make_discretization("pc_gradient", alpha=0.1))["setup"] is None


def test_wrappers_are_looked_through():
    P = make_problem("linear_system", n=3)
    est = EstimatedDerivativeProblem(P, method="savgol")
    st = C.step_flops(est, make_model("znn"), make_discretization("euler"))
    base = C.step_flops(CountingProblem(P), make_model("znn"), make_discretization("euler"))
    win = est.window
    assert st["by_prim"]["dEdt"] == win * C.problem_cost(P).E(1)["algebra"] + 2 * 3 * win
    assert base["total"] == C.step_flops(P, make_model("znn"), make_discretization("euler"))["total"]
    q = QuantizedDiscretizer(make_discretization("euler"), bits=16, frac_bits=8)
    assert (C.step_flops(P, make_model("znn"), q)["by_category"]["disc"]
            == base["by_category"]["disc"] + 8 * 3)


# ---------------------------------------------------------------------------
# 2. ufunc-level counts of what the code executes
# ---------------------------------------------------------------------------
PRIMITIVE_CASES = [
    ("linear_system", dict(n=5)), ("matrix_inversion", dict(n=4)),
    ("matrix_inversion_right", dict(n=3)), ("pseudoinverse", dict(n=3, side="left")),
    ("pseudoinverse", dict(n=3, side="right")), ("pseudoinverse", dict(rows=5, cols=2)),
    ("sylvester", dict(n=3, k=2)), ("sylvester", dict(n=3, k=4, skew=0.0)), ("lyapunov", dict(n=3)),
    ("nonlinear_equations", {}), ("nonlinear_equations", dict(variant="manufactured", n=4)),
    ("tv_unconstrained_opt", {}), ("tv_unconstrained_opt", dict(variant="quadratic", n=3)),
    ("tv_unconstrained_opt", dict(variant="manufactured", n=3)), ("tvqp_eq", {}),
    ("tvqp_eq", dict(variant="random", n=4, p=2)), ("tvqp_ineq", {}),
    ("tvqp_ineq", dict(variant="random", n=3, p=1))]


@pytest.mark.parametrize("key,kw", PRIMITIVE_CASES)
def test_primitive_algebra_matches_executed_ufuncs(key, kw, counting_numpy):
    B, t = 3, 0.37
    P = _counting_problem(make_problem(key, **kw))
    pc = C.problem_cost(P)
    rng = np.random.default_rng(0)
    x = fa(P.x0(rng, B))
    d = {k: (v if k == "t" else fa(v)) for k, v in P.data(t).items()}
    r, v = fa(rng.standard_normal((B, P.m))), fa(rng.standard_normal((B, P.m)))
    for prim, fn in [("E", lambda: P.E(x, d)), ("dEdt", lambda: P.dEdt(x, t)),
                     ("JT", lambda: P.JT(x, d, v))]:
        assert counted(fn) == pc.call(prim, B).get("algebra", 0.0), prim
    cold = counted(lambda: P.solve_J(x, d, r))      # factorisation computed (numpy part)
    warm = counted(lambda: P.solve_J(x, d, r))      # cached factorisation reused
    assert warm == pc.call("solve_J", B).get("algebra", 0.0)
    if key == "pseudoinverse":                       # Gram product inside the cached factor
        assert cold - warm == 2 * pc.g ** 2 * pc.o == pc.factor() - C.potrf(pc.g)
    elif key == "sylvester":                         # -B inside the cached Schur pair
        assert cold - warm == kw.get("k") ** 2
        assert pc.factor() - C.schur_real(pc.n) - C.schur_real(pc.k) == cold - warm


MODEL_CASES = [("znn", {}, False), ("znn", {}, True), ("znn", dict(activation="sign_bi_power"), False),
               ("znn", dict(activation="sign"), True), ("znn", dict(activation="power"), True),
               ("znn", dict(activation="sign_power"), True),
               ("gnn", {}, False), ("gnn", dict(activation="sign_power"), False),
               ("davidenko", {}, False), ("newton_flow", {}, False), ("ieznn", {}, False),
               ("dieznn", {}, False), ("dieznn", dict(xi=3.0), False), ("hntznn", {}, False),
               ("znn_explicit_inverse", {}, False), ("znn_explicit_inverse", {}, True),
               ("vpznn", {}, False), ("sptc_nt_znn", {}, False), ("sptc_an_foznn", {}, False)]


class _StubProblem:
    """Problem facade whose solve_J / JT / dEdt cost nothing (plain numpy) but
    return counting arrays; everything else is delegated."""

    def __init__(self, P):
        self.P = P
        self.n_x, self.m = P.n_x, P.m
        rng = np.random.default_rng(5)
        self._A = np.eye(P.m) + 0.1 * rng.standard_normal((P.m, P.n_x))

    def solve_J(self, x, d, r):
        return fa(np.array(np.asarray(r)))

    def JT(self, x, d, v):
        return fa(np.asarray(v) @ self._A)

    def dEdt(self, x, t):
        return fa(np.zeros((np.shape(x)[0], self.m)))

    def capabilities(self):
        return self.P.capabilities()

    def __getattr__(self, k):
        return getattr(self.P, k)


@pytest.mark.parametrize("mk,kw,implicit", MODEL_CASES)
def test_model_own_work_matches_executed_ufuncs(mk, kw, implicit, counting_numpy):
    B = 3
    P = make_problem("matrix_inversion", n=3)
    M = make_model(mk, **kw)
    rng = np.random.default_rng(0)
    nz = M.n_z(P)
    ctx = Context(problem=_StubProblem(P), t=2.0, tau=0.01, k=5, x=fa(P.x0(rng, B)),
                  z=fa(rng.standard_normal((B, nz))), d=P.data(2.0),
                  E=fa(rng.standard_normal((B, P.m))), dEdt=fa(rng.standard_normal((B, P.m))),
                  gamma=fa(np.full((B, 1), 10.0)), implicit=implicit, extra={"info": "analytic"})
    calls, own = C.model_cost(M, P).evaluate(B, True, "analytic", implicit)
    assert counted(lambda: M.rhs(ctx)) == own


@pytest.mark.parametrize("info", ["analytic", "none"])
def test_gznn_own_work_matches_executed_ufuncs(info, counting_numpy):
    B = 2
    P = make_problem("matrix_inversion", n=2)
    M = make_model("gznn")
    rng = np.random.default_rng(0)
    ctx = Context(problem=_StubProblem(P), t=0.5, tau=0.01, k=5, x=fa(P.x0(rng, B)),
                  z=fa(np.zeros((B, 0))), d=P.data(0.5), E=fa(rng.standard_normal((B, P.m))),
                  dEdt=fa(rng.standard_normal((B, P.m))), gamma=fa(np.full((B, 1), 10.0)),
                  implicit=False, extra={"info": info})
    calls, own = C.model_cost(M, P).evaluate(B, True, info, False)
    n = P.n_x
    lapack = B * (C.getrf(n) + C.getrs(n, 1))            # batched solve, not a ufunc
    invisible = n * n                                    # beta * np.eye(n): plain numpy, executed
    assert counted(lambda: M.rhs(ctx)) == own - lapack - invisible
    assert calls[0] == ("JT", B * P.m, "rep")


def test_activation_counts_match_executed_ufuncs(counting_numpy):
    import znnbench.activations as acts
    B, m = 3, 9
    rng = np.random.default_rng(0)
    for key in acts.available():
        a = acts.get(key)
        if key == "strict_predefined":
            a = a.at(2.0)
        e, h = fa(rng.standard_normal((B, m))), fa(np.full((B, 1), 0.1))
        af = C.activation_flops(a)
        assert counted(lambda: a(e)) == af["call"] * B * m, key
        assert counted(lambda: a.prox(e, h)) == pytest.approx(af["prox"] * B * m
                                                              + af["prox_row"] * B), key


def _stub_f(t, x, z, k, implicit):
    return fa(np.asarray(x) * 0.0 + 1.0), fa(np.ones((x.shape[0], z.shape[1])))


@pytest.mark.parametrize("key,kw", [("euler", {}), ("taylor_zhang", {}), ("zead_6i_qiu2019", {}),
                                    ("uhlig_4_6", {}), ("ab2", {}), ("ab3", dict(z_scheme="ab")),
                                    ("rk4", {}), ("implicit_euler_err", {}), ("peb_euler", {}),
                                    ("implicit_taylor_zhang", {}),
                                    ("implicit_taylor_zhang_naive", {})])
def test_discretizer_own_work_matches_executed_ufuncs(key, kw):
    B, n, nz = 2, 5, 3
    D = make_discretization(key, **kw)
    D.reset()
    rng = np.random.default_rng(1)
    x, z = fa(rng.standard_normal((B, n))), fa(rng.standard_normal((B, nz)))
    with warnings.catch_warnings():                       # the stub ignores ErrorMapRequest
        warnings.simplefilter("ignore")
        for k in range(12):                               # steady state (full history)
            x, z = D.step(k, k * 0.01, 0.01, x, z, _stub_f)
    items = C._schedule(D, make_problem("linear_system", n=n), B, n, nz)[1]
    own = sum(it[2] for it in items if it[0] == "own")
    assert counted(lambda: D.step(12, 0.12, 0.01, x, z, _stub_f)) == own


# ---------------------------------------------------------------------------
# 3. agreement with simulate runs
# ---------------------------------------------------------------------------
PROFILE_CASES = [
    ("linear_system", {}, "znn", "euler", {}, "analytic"),
    ("matrix_inversion", dict(n=3), "znn", "rk4", {}, "analytic"),
    ("linear_system", {}, "gnn", "euler", {}, "analytic"),
    ("matrix_inversion", dict(n=3), "davidenko", "pc_newton", dict(C=2), "analytic"),
    ("tv_unconstrained_opt", {}, "davidenko", "pc_gradient", dict(C=3, alpha=0.05), "analytic"),
    ("linear_system", {}, "znn", "taylor_zhang", {}, "bd2"),
    ("matrix_inversion", dict(n=2), "gznn", "euler", {}, "analytic"),
    ("sylvester", dict(n=2), "ieznn", "ab3", {}, "analytic"),
    ("linear_system", {}, "znn", "broyden_dtznn_zhang2012", {}, "analytic"),
    ("linear_system", {}, "znn", "cdtznn_mahuang2025", {}, "analytic"),
    ("robot_puma560", {}, "gnn", "pc_gradient_causal", dict(alpha=0.05), "analytic"),
    ("tvqp_eq", {}, "znn", "newton_tracking_causal", dict(C=2), "none")]


@pytest.mark.parametrize("pk,pkw,mk,dk,dkw,info", PROFILE_CASES)
def test_call_counts_agree_with_profile_step_cost(pk, pkw, mk, dk, dkw, info):
    """profile_step_cost averages over all N steps (start-up included, N + 1
    sampling calls); the analytic counts are steady-state, so they agree up to
    a few calls in total."""
    steps = 100
    P, M, D = make_problem(pk, **pkw), make_model(mk), make_discretization(dk, **dkw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prof = profile_step_cost(P, M, D, tau=1e-2, steps=steps, B=2,
                                 gamma=1.0 if mk == "gnn" else 10.0, info=info)
    an = C.step_flops(P, M, D, B=2, info=info, skip_unused_dEdt=False)
    for k, v in an["calls"].items():
        assert abs(prof["calls_per_step"][k] - v) * steps <= 4.0 + 1e-9, k


EMPIRICAL_CASES = [
    ("matrix_inversion", dict(n=3), "znn", "rk4", {}, "analytic"),
    ("matrix_inversion", dict(n=3), "znn_explicit_inverse", "taylor_zhang", {}, "analytic"),
    ("sylvester", dict(n=2, k=3), "davidenko", "pc_newton", dict(C=2), "analytic"),
    ("lyapunov", dict(n=2), "gznn", "euler", {}, "bd2"),
    ("pseudoinverse", dict(n=2), "ieznn", "ab3", {}, "analytic"),
    ("complex_inversion", dict(n=2), "znn", "pc_newton_causal", {}, "analytic"),
    ("tvqp_ineq", {}, "davidenko", "newton_tracking", dict(C=2), "analytic"),
    ("robot_panda", {}, "gnn", "euler", {}, "analytic"),
    ("robot_puma560", {}, "znn", "rk4", {}, "bd2"),
    ("linear_system", dict(n=3), "znn", "broyden_dtznn_zhang2012", {}, "analytic"),
    ("matrix_inversion", dict(n=2), "znn", "cdtznn_mahuang2025", {}, "analytic"),
    ("tv_unconstrained_opt", dict(variant="manufactured", n=3), "znn", "cdtznn_mahuang2025", {},
     "bd2"),
    # LCCZNN (12) with linear phi: the secondary restatement of the UNREGISTERED inverse-free
    # ZNN of Chen et al. 2024 (structured_extra.NOT_REGISTERED); labelled as LCCZNN, not Chen
    ("linear_system", dict(n=3), "lcc_znn[linear]", "euler", {}, "analytic"),
    ("matrix_inversion", dict(n=2), "lcc_znn[linear]", "uhlig_3_4", {}, "analytic"),
    ("matrix_inversion", dict(n=2), "fa_gnn_znn", "rk4", {}, "analytic"),
]
EMPIRICAL_GAMMA = {"fa_gnn_znn": 0.5}       # keeps the short run from diverging
#: model specs with keyword arguments: label -> (registry key, kwargs)
MODEL_SPECS = {"lcc_znn[linear]": ("lcc_znn", {"activation": "linear"})}


def _model(label):
    key, kw = MODEL_SPECS.get(label, (label, {}))
    return make_model(key, **kw)


@pytest.mark.parametrize("pk,pkw,mk,dk,dkw,info", EMPIRICAL_CASES)
def test_empirical_cross_check_is_exact(pk, pkw, mk, dk, dkw, info):
    P, M, D = make_problem(pk, **pkw), _model(mk), make_discretization(dk, **dkw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = C.empirical_step_cost(P, M, D, B=2, tau=TAU_EXACT, steps=8, warmup=6,
                                  gamma=EMPIRICAL_GAMMA.get(mk, 5.0), info=info)
    an = r["analytic"]
    for k in C.PRIMS:
        assert r["calls"][k] == an["calls"][k], k
        if k != "data":
            assert r["rows"][k] == an["rows"][k], k
    if C.problem_cost(P).factor_mode == "cached":
        assert r["factorizations"] == an["factorizations"]
    assert r["kin_rows"] == an["kin_rows"]
    assert r["problem_flops"] == pytest.approx(r["analytic_problem_flops"], rel=1e-12)


def test_empirical_value_dependent_cache_hits_stay_within_bounds():
    """With tau = 0.01 the factorisation cache also hits when t_k + tau and
    t_{k+1} differ by one ulp but give bitwise-equal data (Lyapunov): the
    observed count lies between the 'ideal' and 'none' reuse models."""
    P, M, D = make_problem("lyapunov", n=2), make_model("znn"), make_discretization("rk4")
    r = C.empirical_step_cost(P, M, D, B=2, tau=0.01, steps=24, warmup=6)
    lo = C.step_flops(P, M, D, B=2, reuse="ideal", skip_unused_dEdt=False)["factorizations"]
    hi = C.step_flops(P, M, D, B=2, reuse="none", skip_unused_dEdt=False)["factorizations"]
    assert lo <= r["factorizations"] <= hi
    assert r["calls"] == {k: r["analytic"]["calls"][k] for k in C.PRIMS}


def test_empirical_restores_the_problem_and_dynamics():
    from znnbench import simulate
    P = make_problem("sylvester", n=2)
    before = simulate(P, make_model("znn"), make_discretization("euler"), tau=0.01, T=0.2, B=2)
    C.empirical_step_cost(P, make_model("znn"), make_discretization("euler"), steps=5, warmup=2)
    from znnbench.problems.matrix import _sylv_pair
    assert "_memo" not in P.__dict__ and P._fac._factor is _sylv_pair
    after = simulate(P, make_model("znn"), make_discretization("euler"), tau=0.01, T=0.2, B=2)
    assert np.array_equal(before["residual"], after["residual"])
    R = make_problem("robot_puma560")
    C.empirical_step_cost(R, make_model("znn"), make_discretization("euler"), steps=3, warmup=2)
    assert "fk_jac" not in R.arm.__dict__


# ---------------------------------------------------------------------------
# coverage of the zoo (keys known when this module was written)
# ---------------------------------------------------------------------------
KNOWN_MODELS = ["znn", "gnn", "znn_consistent", "vpznn", "vpznn_exp", "ieznn", "dieznn", "hntznn",
                "sptc_nt_znn", "sptc_an_foznn", "gznn", "znn_explicit_inverse",
                "znn_explicit_inverse_right", "davidenko", "newton_flow", "lcc_znn",
                "hgnn_znn", "fa_gnn_znn"]       # inverse_free_znn: de-registered (NOT_REGISTERED)


def test_every_known_problem_model_and_discretization_has_a_cost():
    from znnbench.registry import available
    av = available()
    for pk in av["problems"]:
        assert C.problem_cost(make_problem(pk)).table(B=2)["E"] > 0, pk
    P = make_problem("matrix_inversion", n=2)
    for mk in KNOWN_MODELS:
        if mk in av["models"]:
            mc = C.model_cost(make_model(mk), P, probe=False)
            assert not isinstance(mc, C.ProbedModelCost), mk
    for dk in av["discretizations"]:
        st = C.step_flops(P, make_model("znn"), make_discretization(dk))
        assert st["total"] > 0, dk


def test_unknown_model_falls_back_to_probing():
    from znnbench.core import Model

    class TwoSolves(Model):
        name = "two_solves"

        def rhs(self, ctx):
            return ctx.solve(-ctx.E) + ctx.solve(-ctx.dEdt), np.zeros((ctx.x.shape[0], 0))

    P = make_problem("matrix_inversion", n=2)
    st = C.step_flops(P, TwoSolves(), make_discretization("euler"), B=3)
    assert st["calls"]["solve_J"] == 2 and st["by_category"]["model"] == 0
    assert any("probed" in note for note in st["notes"])
    with pytest.raises(NotImplementedError):
        C.model_cost(TwoSolves(), P, probe=False)


# ---------------------------------------------------------------------------
# 4. adversarial verification (2026-10-18, docs/spec/costs.md, Verification)
# ---------------------------------------------------------------------------
SX_CASES = [("lcc_znn", {}), ("lcc_znn", dict(activation="sig")), ("lcc_znn", dict(activation="sig_lin")),
            ("lcc_znn", dict(activation="linear")), ("lcc_znn", dict(activation="sign_power")),
            ("hgnn_znn", {}), ("hgnn_znn", dict(feedforward=True)),
            ("hgnn_znn", dict(activation="sign_bi_power")), ("fa_gnn_znn", {})]


@pytest.mark.parametrize("mk,kw", SX_CASES)
def test_structured_extra_models_own_work_matches_executed_ufuncs(mk, kw, counting_numpy):
    B = 3
    P = make_problem("matrix_inversion", n=3)
    M = make_model(mk, **kw)
    rng = np.random.default_rng(0)
    E = 0.1 * rng.standard_normal((B, P.m))            # small energy: exp(|nu|) stays finite
    ctx = Context(problem=_StubProblem(P), t=2.0, tau=0.01, k=5, x=fa(P.x0(rng, B)),
                  z=fa(np.zeros((B, 0))), d=P.data(2.0), E=fa(E),
                  dEdt=fa(rng.standard_normal((B, P.m))), gamma=fa(np.full((B, 1), 10.0)),
                  implicit=False, extra={"info": "analytic"})
    calls, own = C.model_cost(M, P).evaluate(B, True, "analytic", False)
    assert counted(lambda: M.rhs(ctx)) == own


def test_error_map_rates_match_executed_ufuncs(counting_numpy):
    """ErrorMapRequest rates of the consistent schemes (all four maps, with the
    fallback chain) for every registered activation and the model-local ones."""
    import znnbench.activations as acts
    import znnbench.discretizations.consistent as cons
    B, P = 3, make_problem("matrix_inversion", n=3)
    m = P.m
    rng = np.random.default_rng(0)
    specs = [(k, {}) for k in sorted(acts.available())] + [
        ("jr_predefined", {"kappa": "arctan"}), ("jr_predefined", {"kappa": "exp"}),
        ("jr_predefined", {"kappa": "rational"}),
        ("signed_power_sum", {"terms": ((1.0, 0.3), (2.0, 1.0))}),
        ("signed_power_sum", {"terms": ((1.0, 0.0), (1.0, 0.5), (1.0, 2.0))}),
        ("predefined_exp", {"k2": 0.0, "k3": 0.0})]
    bad = []
    with warnings.catch_warnings():                       # fallback-chain notices
        warnings.simplefilter("ignore")
        for key, kw in specs:
            M = make_model("znn_consistent", activation=key, **kw)
            for emap in ("resolvent", "peb", "exact", "semi_implicit"):
                for coeffs in (cons.EULER, cons.TAYLOR_ZHANG):
                    req = cons.ErrorMapRequest(emap, coeffs[0], coeffs[1:], memory=True)
                    req.begin_step(True)
                    for _ in range(len(coeffs) - 2):       # full error memory (steady state)
                        req.E_hist.appendleft(fa(rng.standard_normal((B, m))))
                    ctx = Context(problem=_StubProblem(P), t=2.0, tau=0.01, k=5,
                                  x=fa(P.x0(rng, B)), z=fa(np.zeros((B, 0))), d=P.data(2.0),
                                  E=fa(rng.standard_normal((B, m))),
                                  dEdt=fa(rng.standard_normal((B, m))),
                                  gamma=fa(np.full((B, 1), 10.0)), implicit=req,
                                  extra={"info": "analytic"})
                    imp = {"error_map": emap, "s": len(coeffs) - 1, "memory": True,
                           "multistep": True}
                    own = C.model_cost(M, P).evaluate(B, True, "analytic", imp)[1]
                    got = counted(lambda: M.rhs(ctx))
                    if got != own:
                        bad.append((key, kw, emap, len(coeffs), got, own))
    assert not bad, bad


def test_davidenko_E_is_charged_only_in_the_code_view():
    """simulate evaluates E(x_k) at every vector-field call, but Davidenko's
    x_dot = -J^{-1} dE/dt never reads it: the default (algorithmic) view leaves
    it out, ``skip_unused_dEdt=False`` / ``skip_unused_E=False`` charge it."""
    P, M = make_problem("linear_system", n=4), make_model("davidenko")
    e = C.problem_cost(P).E(1)["algebra"]
    for dk, kw, n_code in [("euler", {}, 1), ("pc_newton_causal", {"C": 2}, 3),
                           ("pc_newton", {"C": 1}, 2), ("pc_gradient_causal", {"alpha": 0.1}, 2)]:
        D = make_discretization(dk, **kw)
        alg = C.step_flops(P, M, D)
        code = C.step_flops(P, M, D, skip_unused_dEdt=False)
        assert code["calls"]["E"] == n_code and alg["calls"]["E"] == n_code - 1, dk
        assert code["total"] - alg["total"] == e, dk
        assert C.step_flops(P, M, D, skip_unused_E=False)["total"] == code["total"], dk
        assert any("E(x_k)" in note for note in alg["notes"])
    assert C.step_flops(P, make_model("znn"), make_discretization("euler"))["calls"]["E"] == 1


class _FAProblem(CountingProblem.__mro__[1]):        # wrappers.ProblemWrapper
    """Data arrays and primitive outputs become counting arrays (LAPACK and the
    synthetic generators stay invisible: 'factor', 'solve' and 'data')."""

    def __init__(self, p):
        super().__init__(p, "fa")
        self.name = p.name

    def data(self, t):
        return {k: (v if k == "t" else fa(v)) for k, v in self.inner.data(t).items()}

    def E(self, x, d):
        return fa(self.inner.E(x, d))

    def dEdt(self, x, t):
        return fa(self.inner.dEdt(x, t))

    def solve_J(self, x, d, r):
        return fa(self.inner.solve_J(x, d, r))

    def JT(self, x, d, v):
        return fa(self.inner.JT(x, d, v))


class _FAModel:
    """Model proxy: simulate's gain and auxiliary state become counting arrays."""

    def __init__(self, inner):
        self.__dict__["_inner"] = inner

    def __getattr__(self, k):
        return getattr(self.__dict__["_inner"], k)

    def rhs(self, ctx):
        ctx.gamma, ctx.z = fa(ctx.gamma), fa(ctx.z)
        return self._inner.rhs(ctx)


class _FAWindow:
    """Discretizer proxy: counting state on entry to every step; snapshots of
    the counter at the start of steps k1 and k2."""

    def __init__(self, inner, k1, k2):
        self.inner, self.k1, self.k2, self.snaps = inner, k1, k2, {}
        self.name, self.causal = inner.name, inner.causal
        self.implicit = getattr(inner, "implicit", False)

    def __getattr__(self, k):
        if k.startswith("__") or k == "inner":
            raise AttributeError(k)
        return getattr(self.__dict__["inner"], k)

    def reset(self):
        self.inner.reset()

    def bind_env(self, **kw):
        if callable(getattr(self.inner, "bind_env", None)):
            self.inner.bind_env(**kw)

    def step(self, k, t, tau, x, z, f):
        if k in (self.k1, self.k2):
            self.snaps[k] = _Count.flops
        return self.inner.step(k, t, tau, fa(x), fa(z), f)


MI3 = ("matrix_inversion", dict(n=3))
LS8 = ("linear_system", dict(n=8))
UO8 = ("tv_unconstrained_opt", dict(variant="manufactured", n=8))
QP = ("tvqp_eq", dict(variant="random", n=6, p=2))
E2E_CASES = [   # E7 (matrix inversion) and E9 (linsys, tvuo, tvqp) methods; h = gamma tau
    (MI3, "znn", "euler", {}, "analytic", 0.3), (MI3, "znn_explicit_inverse", "euler", {}, "analytic", 0.3),
    (MI3, "gnn", "euler", {}, "none", 0.007), (MI3, "davidenko", "pc_newton_causal", {}, "analytic", 0.3),
    (MI3, "davidenko", "pc_gradient_causal", {}, "analytic", 0.3),
    (MI3, "davidenko", "newton_tracking_causal", {}, "analytic", 0.3),
    (MI3, "znn", "uhlig_3_4", {}, "analytic", 0.0255), (MI3, "hgnn_znn", "euler", {}, "none", 0.007),
    (MI3, "fa_gnn_znn", "euler", {}, "none", 0.007), (MI3, "lcc_znn[linear]", "euler", {}, "analytic", 0.05),
    (MI3, "znn", "broyden_dtznn_zhang2012", {}, "analytic", 0.3),
    (MI3, "znn", "cdtznn_mahuang2025", {}, "analytic", 0.0465),
    (LS8, "znn", "zead_5i", {}, "analytic", 0.2), (LS8, "gnn", "euler", {}, "analytic", 0.002),
    (LS8, "davidenko", "pc_newton", dict(C=2), "analytic", 1.0),
    (UO8, "znn", "uhlig_4_6", {}, "analytic", 0.01), (UO8, "davidenko", "pc_gradient", dict(C=2), "analytic", 1.0),
    (QP, "znn", "taylor_zhang", {}, "analytic", 0.24),
    (QP, "davidenko", "newton_tracking", dict(C=3), "analytic", 1.0)]


def _whole_step_check(prob, mk, dk, dkw, info, h, noise=None):
    """Every floating operation of whole steps of the real ``simulate`` loop,
    counted by the ufunc counter, equals step_flops (code view) minus the
    LAPACK part, plus simulate's divergence check |x|.max(axis=1)."""
    from znnbench import simulate
    P, tau, B = make_problem(prob[0], **prob[1]), 1e-2, 2
    W = _FAWindow(make_discretization(dk, **dkw), 12, 18)     # after every start-up (s <= 9)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = simulate(_FAProblem(P), _FAModel(_model(mk)), W, tau=tau, T=19 * tau, B=B,
                       gamma=h / tau, info=info, noise=noise, online_stats=False,
                       record_solution_error=False, record_every=10 ** 6)
    assert not sim["diverged"].any()
    got = (W.snaps[18] - W.snaps[12]) / 6.0
    an = C.step_flops(P, _model(mk), make_discretization(dk, **dkw), B=B, info=info,
                      noise=noise is not None, skip_unused_dEdt=False)
    cat = an["by_category"]
    assert got == pytest.approx(an["total"] - cat["factor"] - cat["solve"] + 2 * B * P.n_x - B,
                                rel=1e-12)


@pytest.mark.parametrize("prob,mk,dk,dkw,info,h", E2E_CASES)
def test_whole_step_count_through_simulate(prob, mk, dk, dkw, info, h, counting_numpy):
    _whole_step_check(prob, mk, dk, dkw, info, h)


@pytest.mark.parametrize("prob,dk,info", [(MI3, "euler", "bd2"), (LS8, "taylor_zhang", "bd1"),
                                          (UO8, "rk4", "bd2")])
def test_whole_step_count_backward_differences(prob, dk, info, counting_numpy):
    """info='bd1'/'bd2': simulate's extra E calls and its 2Bm / 5Bm differences."""
    _whole_step_check(prob, "znn", dk, {}, info, 0.2)


@pytest.mark.parametrize("prob,dk", [(UO8, "euler"), (MI3, "rk4"), (LS8, "pc_newton_causal")])
def test_whole_step_count_with_design_formula_noise(prob, dk, counting_numpy):
    """Additive noise: one more solve_J per vector-field call (J^{-1} nu) and the
    addition x_dot + J^{-1} nu (B n_x); the noise generator itself is not
    counted."""
    from znnbench import make_noise
    mk = "davidenko" if dk.startswith("pc_") else "znn"
    _whole_step_check(prob, mk, dk, {}, "analytic", 0.2, noise=make_noise("constant", level=0.1))


def _steady_calls(P, mk, dk, dkw, info, gamma, n1=20, n2=50, **sim_kw):
    """Per-step Problem calls of profile_step_cost with start-up removed (the
    difference of two run lengths, as experiments/tasks/E9_cost.py does)."""
    out = []
    for n in (n1, n2):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = profile_step_cost(P, _model(mk), make_discretization(dk, **dkw), tau=1e-2, steps=n,
                                  B=2, gamma=gamma, info=info, **sim_kw)
        out.append({k: v * r["steps"] for k, v in r["calls_per_step"].items()})
    return {k: (out[1][k] - out[0][k]) / (n2 - n1) for k in out[0]}


#: experiments/tasks/E7_common.methods() as of 2026-10-18 (core + registered extras); E7 runs
#: lcc_znn with its default activation, whose calls are the same (the linear one is used here
#: because exp(|nu|) overflows from the E7 initial states)
E7_METHODS = [("znn", "euler", {}, "analytic", 0.3), ("znn_explicit_inverse", "euler", {}, "analytic", 0.3),
              ("gnn", "euler", {}, "none", 0.007), ("davidenko", "euler", {}, "analytic", 0.3),
              ("davidenko", "pc_newton_causal", {}, "analytic", 0.3),
              ("davidenko", "pc_gradient_causal", {}, "analytic", 0.3),
              ("davidenko", "newton_tracking_causal", {}, "analytic", 0.3),
              ("znn", "uhlig_3_4", {}, "analytic", 0.0255), ("fa_gnn_znn", "euler", {}, "none", 0.007),
              ("hgnn_znn", "euler", {}, "none", 0.007), ("lcc_znn[linear]", "euler", {}, "analytic", 0.05),
              ("znn", "broyden_dtznn_zhang2012", {}, "analytic", 0.3),
              ("znn", "cdtznn_mahuang2025", {}, "analytic", 0.0465)]
#: experiments/configs/E9.py (all groups use info='analytic')
E9_METHODS = ([("znn", f, {}, "analytic", h) for f, h in
               [("euler", 1.0), ("taylor_zhang", 0.24), ("zead_5i", 0.2), ("uhlig_3_4", 0.0255),
                ("uhlig_4_6", 0.01), ("zead_7i_cang2024", 0.05)]]
              + [("gnn", "euler", {}, "analytic", 0.002), ("davidenko", "euler", {}, "analytic", 1.0)]
              + [("davidenko", k, {"C": c}, "analytic", 1.0)
                 for k in ("pc_newton", "pc_newton_causal", "pc_gradient", "pc_gradient_causal",
                           "newton_tracking", "newton_tracking_causal") for c in (1, 3)])


@pytest.mark.parametrize("experiment", ["E7", "E9"])
def test_e7_e9_call_counts_exact_against_profile_step_cost(experiment):
    if experiment == "E7":
        cases = [(make_problem("matrix_inversion", n=2), *mth) for mth in E7_METHODS]
    else:
        probs = [make_problem("linear_system", n=3),
                 make_problem("tv_unconstrained_opt", variant="manufactured", n=3),
                 make_problem("tvqp_eq", variant="random", n=3, p=1)]
        cases = [(P, *mth) for P in probs for mth in E9_METHODS]
    bad = []
    for P, mk, dk, dkw, info, h in cases:
        prof = _steady_calls(P, mk, dk, dkw, info, h / 1e-2)
        an = C.step_flops(P, _model(mk), make_discretization(dk, **dkw), B=2, info=info,
                          skip_unused_dEdt=False)["calls"]
        if any(abs(prof[k] - an[k]) > 1e-9 for k in C.PRIMS):
            bad.append((P.name, mk, dk, dkw, prof, an))
    assert not bad, bad


def test_noise_and_bd_call_counts_exact_against_profile_step_cost():
    from znnbench import make_noise
    P = make_problem("matrix_inversion", n=2)
    bad = []
    for mk, dk, info, noise in [("znn", "euler", "analytic", True), ("znn", "rk4", "analytic", True),
                                ("davidenko", "pc_newton", "analytic", True),
                                ("znn", "taylor_zhang", "bd2", False), ("gnn", "euler", "bd1", False),
                                ("znn", "broyden_dtznn_zhang2012", "analytic", True)]:
        kw = {"noise": make_noise("constant", level=0.1)} if noise else {}
        prof = _steady_calls(P, mk, dk, {}, info, 0.005 / 1e-2 if mk == "gnn" else 20.0, **kw)
        an = C.step_flops(P, _model(mk), make_discretization(dk), B=2, info=info, noise=noise,
                          skip_unused_dEdt=False)["calls"]
        if any(abs(prof[k] - an[k]) > 1e-9 for k in C.PRIMS):
            bad.append((mk, dk, info, noise, prof, an))
    assert not bad, bad


def test_complex_inversion_primitives_match_executed_ufuncs(counting_numpy):
    """Complex arithmetic under the spec convention (complex * 6, + 2; real *
    complex 2, real + complex 1; complex GEMM 8 per multiply-add)."""
    B, t, n = 3, 0.37, 3
    P = _counting_problem(make_problem("complex_inversion", n=n))
    pc = C.problem_cost(P)
    rng = np.random.default_rng(0)
    x = fa(P.x0(rng, B))
    d = {k: (v if k == "t" else fa(v)) for k, v in P.data(t).items()}
    r, v = fa(rng.standard_normal((B, P.m))), fa(rng.standard_normal((B, P.m)))
    for prim, fn in [("E", lambda: P.E(x, d)), ("dEdt", lambda: P.dEdt(x, t)),
                     ("JT", lambda: P.JT(x, d, v))]:
        assert counted(fn) == pc.call(prim, B)["algebra"], prim
    cold = counted(lambda: P.solve_J(x, d, r))
    warm = counted(lambda: P.solve_J(x, d, r))
    assert warm == pc.solve_J(B)["algebra"]
    assert cold - warm == pc.factor() - C.getrf(n, cplx=True) == 3 * n * n     # Ar + 1j Ai


class _RobotNumpy(_NumpyProxy):
    """robot.py's numpy: the arrays it allocates become counting arrays, so the
    kinematics (link products, cross-product Jacobian) are visible."""

    @staticmethod
    def empty(shape, dtype=float):
        return np.empty(shape, dtype).view(FA)

    @staticmethod
    def zeros(shape, dtype=float):
        return np.zeros(shape, dtype).view(FA)

    @staticmethod
    def broadcast_to(a, shape):
        return np.broadcast_to(np.asarray(a), shape).view(FA)


@pytest.mark.parametrize("pk,pkw", [("robot_puma560", {}), ("robot_panda", {}),
                                    ("robot_puma560", {"damping": 0.05})])
def test_robot_kinematics_and_primitives_match_executed_ufuncs(pk, pkw, monkeypatch):
    import znnbench.problems.robot as rb
    P = make_problem(pk, **pkw)                 # built before patching: data stay plain
    monkeypatch.setattr(rb, "np", _RobotNumpy())
    pc = C.problem_cost(P)
    B = 3
    rng = np.random.default_rng(0)
    x = fa(P.x0(rng, B) + 0.1 * rng.standard_normal((B, P.n_x)))
    d = {k: (v if k == "t" else fa(v)) for k, v in P.data(0.3).items()}
    r, v = fa(rng.standard_normal((B, 3))), fa(rng.standard_normal((B, 3)))
    P._ck = None
    assert counted(lambda: P.E(x, d)) == pc.E(B)["algebra"] + pc.kin(B)     # kinematics: miss
    damp = 9.0 if P.damping else 0.0            # damping**2 * np.eye(3): plain numpy, executed
    assert counted(lambda: P.solve_J(x, d, r)) == pc.solve_J(B)["algebra"] - damp   # cache hit
    assert counted(lambda: P.JT(x, d, v)) == pc.JT(B)["algebra"]
    assert counted(lambda: P.dEdt(x, 0.3)) == 0.0          # -r_d_dot(t): 'data' category


def test_data_generators_match_executed_ufuncs():
    """TVMatrix (square, tall, wide; values and derivatives) and TVVector, with
    their parameter arrays made counting arrays (so parameter-only products
    such as s0 * mod * w_s are seen as well)."""
    from znnbench.problems._tv import TVMatrix, TVVector
    rng = np.random.default_rng(0)
    for n, nc, sym in [(4, 4, False), (5, 3, False), (3, 5, False), (4, 4, True)]:
        M = TVMatrix(n, rng, n_cols=nc, symmetric=sym)
        for so in {id(M.U): M.U, id(M.V): M.V}.values():
            so.P, so.w = fa(so.P), fa(so.w)
        M.s0, M.ws, M.ph = fa(M.s0), fa(M.ws), fa(M.ph)
        assert counted(lambda: M(0.7)) == C.tvmatrix(n, nc), (n, nc, sym)
        assert counted(lambda: M.dot(0.7)) == C.tvmatrix(n, nc, deriv=True), (n, nc, sym)
    V = TVVector(6, rng)
    V.a, V.w, V.ph = fa(V.a), fa(V.w), fa(V.ph)
    assert counted(lambda: V(0.7)) == C.tvvector(6)
    assert counted(lambda: V.dot(0.7)) == C.tvvector(6, deriv=True)


@pytest.mark.parametrize("pk,pkw,direction", [("linear_system", dict(n=5), "normal"),
                                              ("tv_unconstrained_opt",
                                               dict(variant="manufactured", n=4), "residual")])
def test_auto_step_size_setup_matches_executed_ufuncs(pk, pkw, direction, monkeypatch):
    """alpha='auto': 30 power iterations of estimate_lipschitz.  Invisible to the
    counter (plain numpy, executed): the initial normalisation of the random v
    (3 B n), the first norm of v inside jacobian_vector (2 B n) and the final
    np.max over the batch (B - 1, a function-level reduction)."""
    import znnbench.discretizations.prediction_correction as pcm
    monkeypatch.setattr(pcm, "np", _NumpyProxy())
    P = _FAProblem(make_problem(pk, **pkw))
    B, rng = 3, np.random.default_rng(1)
    x = fa(P.x0(rng, B))
    d = P.data(0.2)
    got = counted(lambda: pcm.estimate_lipschitz(P, x, d, direction))
    D = make_discretization("pc_gradient", C=1, direction=direction)
    su = C.step_flops(P.inner, make_model("davidenko"), D, B=B)["setup"]
    assert su["calls"] == ({"E": 60, "JT": 30} if direction == "normal" else {"E": 60})
    assert got == su["flops"] - 5 * B * P.n_x - (B - 1)
