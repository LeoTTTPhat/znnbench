"""Structured per-step FLOP model for ZNN-Bench (experiments E7 and E9).

``wrappers.dense_flops`` charges every call as if J were an unstructured dense
m x n_x matrix.  For matrix-valued unknowns this is badly wrong: for
``matrix_inversion`` with n = 128 (n_x = m = 16384) it charges
(2/3) n_x^3 ~ 2.9e12 FLOPs per ``solve_J``, while the code factorises one
128 x 128 matrix (~1.4e6 FLOPs) and does triangular solves with n right-hand
sides per trajectory (~4.2e6 FLOPs).  This module counts, analytically, the
operations the ZNN-Bench code *actually performs*:

* per problem family (``problem.family`` + dimensions): ``E``, ``dEdt``,
  ``solve_J``, ``JT``, the factorisation (and whether it is cached per data
  sample), the data generation of the synthetic generator (``data`` and the
  derivative data used by ``dEdt``; excluded from totals by default) and the
  robot kinematics (cached per state);
* per model: which problem calls ``rhs`` makes and with how many rows (e.g.
  ``gznn`` calls ``JT`` with B*m rows and ``dEdt`` with 2*B*n_x rows), plus its
  own numpy work (``znn_explicit_inverse``: one batched n x n matmul, no
  solve);
* per discretization: the vector-field calls per step (RK4 = 4, look-ahead =
  1, prediction-correction = 1 + C correction blocks), the stage data
  samples, and its own vector updates (look-ahead with s past samples:
  (2s + 2) B n_x).

Entry points: :func:`problem_cost` (per-primitive table), :func:`step_flops`
(per-step total and breakdown) and :func:`empirical_step_cost` (a
``CountingProblem``-based cross-check that runs ``simulate`` and observes the
calls, their row counts, the factorisations, the data generations and the
kinematics evaluations).

Counting convention (see docs/spec/costs.md)
--------------------------------------------
* Dense kernels: the operation counts of LAPACK's own timing routine DOPLA
  (netlib.org/lapack/timing/lin/dopla.f, read 2026-10-18; the same counts are
  tabulated in LAPACK Working Note 41): GEMM (p x q)(q x r) = 2pqr;
  GETRF(n) = (n^3 - n)/3 mults + (2n^3 - 3n^2 + n)/6 adds = 2n^3/3 - n^2/2 - n/6;
  GETRS(n, k) = k n^2 mults + k n(n - 1) adds = k(2n^2 - n);
  POTRF(n) = (n^3 + 3n^2 + 2n)/6 mults + (n^3 - n)/6 adds = n^3/3 + n^2/2 + n/6;
  POTRS(n, k) = k n(n + 1) mults + k n(n - 1) adds = 2kn^2; complex: DOPLA's
  6 FLOPs per complex multiplication and 2 per complex addition (complex GEMM
  8pqr).
* Real Schur decomposition with the orthogonal factor: about 25 n^3 (Golub &
  Van Loan, Matrix Computations, Sec. 7.5.6, not re-read; openly stated as
  "about 25n^3 flops for Q and T" by N. J. Higham, "What is a Schur
  decomposition?", 2022) -- ASSUMPTION: an average-case estimate; the number
  of Francis QR iterations, hence the true count, is data dependent.
  Triangular Sylvester solve (LAPACK *TRSYL) with m x m and n x n
  quasi-triangular factors: mn(m + n), leading order (column-wise back
  substitution: n triangular solves of order m, m^2 each, plus the updates
  with the previous columns, m n (n - 1); cf. Golub & Van Loan, Sec. 7.6.3).
* numpy-level work: every element of a floating-point ufunc result counts 1
  FLOP (+, -, *, /, **, abs, sign, sqrt, sin, exp, tanh, minimum, ...;
  ASSUMPTION: transcendental functions are counted as 1 FLOP);
  comparisons, ``where``, copies, reshapes, broadcasts and concatenations are
  free; ``einsum`` with a summed index counts 2 FLOPs per multiply-add.
  Operations on Python/numpy scalars are free.
* Categories: ``algebra`` (numpy-level work of the problem primitives, on
  the state and on the data arrays), ``factor`` (factorisations), ``solve``
  (LAPACK triangular solves / TRSYL / batched ``np.linalg.solve``), ``kin``
  (robot forward kinematics and Jacobian), ``data`` (synthetic data
  generation and state-independent derivative-data terms; excluded unless
  ``include_data``), ``model`` / ``disc`` / ``sim`` (the model's own numpy
  work, the discretizer's vector updates, simulate's backward differences and
  noise injection).
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Optional

import numpy as np

__all__ = ["gemm", "getrf", "getrs", "potrf", "potrs", "schur_real", "trsyl",
           "problem_cost", "model_cost", "step_flops", "empirical_step_cost",
           "float_reuse_fraction", "scaling_exponent", "activation_flops",
           "CATEGORIES", "PROBLEM_CATEGORIES"]

CATEGORIES = ("algebra", "factor", "solve", "kin", "data", "model", "disc", "sim")
PROBLEM_CATEGORIES = ("algebra", "factor", "solve", "kin", "data")
PRIMS = ("data", "E", "dEdt", "solve_J", "JT")


# ===========================================================================
# 1. dense kernels (LAPACK DOPLA / LAWN 41 counts)
# ===========================================================================
def gemm(p, q, r, cplx=False):
    """(p x q)(q x r) product: 2pqr real FLOPs (8pqr complex)."""
    return (8.0 if cplx else 2.0) * p * q * r


def getrf(n, cplx=False):
    """LU factorisation with partial pivoting of an n x n matrix (LAPACK DOPLA):
    (n^3 - n)/3 multiplications, (2n^3 - 3n^2 + n)/6 additions (exact integers)."""
    n = int(n)
    mults = (n ** 3 - n) // 3
    adds = (2 * n ** 3 - 3 * n ** 2 + n) // 6
    return float(6 * mults + 2 * adds if cplx else mults + adds)


def getrs(n, nrhs, cplx=False):
    """Forward + backward substitution with an LU factor, nrhs right-hand sides."""
    mults = nrhs * n ** 2
    adds = nrhs * (n ** 2 - n)
    return float(6 * mults + 2 * adds if cplx else mults + adds)


def potrf(n):
    """Cholesky factorisation of an n x n SPD matrix (LAPACK DOPLA):
    (n^3 + 3n^2 + 2n)/6 multiplications (incl. n square roots), (n^3 - n)/6 additions."""
    n = int(n)
    return float((n ** 3 + 3 * n ** 2 + 2 * n) // 6 + (n ** 3 - n) // 6)


def potrs(n, nrhs):
    """Two triangular solves with a Cholesky factor (LAPACK DOPLA: n(n + 1)
    mults + n(n - 1) adds = 2 n^2 per rhs)."""
    return 2.0 * n ** 2 * nrhs


def schur_real(n):
    """Real Schur form T and orthogonal Z: about 25 n^3 (average-case estimate,
    Golub & Van Loan Sec. 7.5.6; Higham 2022).  ASSUMPTION: data dependent."""
    return 25.0 * n ** 3


def trsyl(m, n):
    """Quasi-triangular Sylvester solve, m x m and n x n factors: mn(m + n)
    (leading order; 2 x 2 blocks and scaling ignored)."""
    return float(m * n * (m + n))


# ===========================================================================
# 2. synthetic data generators (problems/_tv.py, problems/matrix.py)
# ===========================================================================
def _rot(n, deriv):
    """SmoothOrthogonal._R: h = n // 2 rotation blocks (w t is evaluated twice,
    once for cos and once for sin)."""
    h = n // 2
    return 11.0 * h if deriv else 5.0 * h


def smooth_orthogonal(n, deriv=False):
    """Q(t) = P R(t) P^T (two n x n matmuls) or its derivative."""
    return gemm(n, n, n) * 2 + _rot(n, deriv)


def tvmatrix(n, nc=None, deriv=False):
    """``_tv.TVMatrix`` A(t) = U S V^T (n x nc) or A_dot(t)."""
    nc = n if nc is None else nc
    k = min(n, nc)
    prod = 2.0 * n * n * nc + 2.0 * n * nc * nc          # U @ S @ V^T with dense S
    if not deriv:
        return smooth_orthogonal(n) + smooth_orthogonal(nc) + 6.0 * k + prod
    return (smooth_orthogonal(n) + smooth_orthogonal(n, True) + smooth_orthogonal(nc)
            + smooth_orthogonal(nc, True) + 12.0 * k + 3.0 * prod + 2.0 * n * nc)


def tvvector(n, deriv=False):
    """``_tv.TVVector``: a sin(w t + phi) (4n) or a w cos(w t + phi) (5n)."""
    return (5.0 if deriv else 4.0) * n


def symskew(n, skew, deriv=False):
    """``matrix._SymSkewMatrix``: sign (S + skew (G - G^T)/2)."""
    if skew:
        return 2.0 * tvmatrix(n, n, deriv) + 4.0 * n * n
    return tvmatrix(n, n, deriv) + 1.0 * n * n


def tvcomplex(n, deriv=False):
    """``matrix._TVComplexMatrix`` (approximate lower-order terms)."""
    unit = gemm(n, n, n, cplx=True) + 7.0 * n * n + 5.0 * n
    if not deriv:
        return 2 * unit + 6.0 * n + 2.0 * n * n + n * n + gemm(n, n, n, cplx=True)
    return (4 * unit + 16.0 * n + 12.0 * n + 3 * (3.0 * n * n + gemm(n, n, n, cplx=True))
            + 4.0 * n * n)


# ===========================================================================
# 3. per-problem primitive costs
# ===========================================================================
def _d(**kw):
    """Cost dict with only the non-zero categories."""
    return {k: float(v) for k, v in kw.items() if v}


def _add(acc, d, scale=1.0):
    for k, v in d.items():
        acc[k] = acc.get(k, 0.0) + scale * v
    return acc


class ProblemCost:
    """Analytic per-call costs of one problem instance.

    ``E(B)``, ``dEdt(B)``, ``solve_J(B)``, ``JT(B)`` return a dict
    {category: FLOPs} for one call with B state rows (the cached factorisation,
    the memoised data generation and the cached kinematics are NOT included:
    they are charged by :func:`step_flops` per data sample / per state).

    Attributes
    ----------
    memo : data arrays are memoised per sampling instant (matrix problems), so
        generation is charged once per distinct instant; otherwise per call.
    factor_mode : 'cached' = ``solve_J`` factorises once per data sample
        (cached by the values of the data arrays; ``factor()`` is that cost and
        it is NOT in ``solve_J(B)``); 'call' = one factorisation inside every
        call; 'row' = one small factorisation per state row (batched solve);
        'none' = no factorisation.
    kin_cached : robot kinematics computed once per distinct state
        (``ManipulatorTracking._kin`` one-entry cache) for E, solve_J and JT.
    tags : {'data': tags generated by data(t), 'dEdt': tags generated by dEdt}.
    """

    family = "generic"
    memo = False
    factor_mode = "none"   # 'cached' (per data sample) | 'call' | 'row' (one per state row) | 'none'
    kin_cached = False
    exact = True          # per-call algebra checked against a ufunc-level count

    def __init__(self, problem):
        self.n_x, self.m = int(problem.n_x), int(problem.m)
        self.dims = {}
        self.tags = {"data": (), "dEdt": ()}
        self.notes = []

    def factor(self):
        return 0.0

    def gen(self, tag):
        return 0.0

    def kin(self, rows):
        return 0.0

    def jac(self, B):
        """Cost dict of ``problem.jacobian(x, d)`` with B rows, or None when the
        problem has no dense ``jacobian`` method (callers then assemble J from
        ``JT``)."""
        return None

    def call(self, prim, B):
        if prim == "data":
            return {}
        return getattr(self, prim)(B)

    def table(self, B=1, include_data=False):
        """Per-primitive totals for one isolated call with B rows: the cached
        factorisation is added to solve_J, the cached kinematics to E (solve_J
        and JT then hit the cache), the data generation only if requested."""
        out = {}
        for prim in ("E", "dEdt", "solve_J", "JT"):
            d = dict(self.call(prim, B))
            if prim == "E" and self.kin_cached:
                d["kin"] = d.get("kin", 0.0) + self.kin(B)
            if prim == "solve_J" and self.factor_mode == "cached":
                d["factor"] = d.get("factor", 0.0) + self.factor()
            if include_data:
                d["data"] = d.get("data", 0.0) + sum(self.gen(t) for t in self.tags.get(prim, ()))
            else:
                d.pop("data", None)
            out[prim] = float(sum(d.values()))
        out["factor"] = float(self.factor())
        out["data"] = float(sum(self.gen(t) for t in self.tags["data"]))
        return out

    def describe(self):
        return {"family": self.family, "n_x": self.n_x, "m": self.m, **self.dims,
                "memo": self.memo, "factor_mode": self.factor_mode}


# --- linear_system --------------------------------------------------------
class LinearSystemCost(ProblemCost):
    family = "linear-system"
    factor_mode = "call"

    def __init__(self, p):
        super().__init__(p)
        self.n = n = p.n_x
        self.dims = {"n": n}
        self.tags = {"data": ("A", "b"), "dEdt": ("dA", "db")}

    def gen(self, tag):
        n = self.n
        return {"A": tvmatrix(n), "b": tvvector(n), "dA": tvmatrix(n, deriv=True),
                "db": tvvector(n, True)}[tag]

    def E(self, B):
        n = self.n
        return _d(algebra=2 * B * n * n + B * n)

    dEdt = E

    def solve_J(self, B):              # np.linalg.solve(A, r.T): no cache
        n = self.n
        return _d(factor=getrf(n), solve=getrs(n, B))

    def JT(self, B):
        return _d(algebra=2 * B * self.n ** 2)


# --- matrix problems --------------------------------------------------------
class MatrixInversionCost(ProblemCost):
    family = "matrix-inversion"
    memo = True
    factor_mode = "cached"

    def __init__(self, p):
        super().__init__(p)
        self.n = n = int(p.n)
        self.example = p.cfg.get("example")
        self.dims = {"n": n, "side": getattr(p, "side", "left"), "example": self.example}
        self.tags = {"data": ("A",), "dEdt": ("dA",)}

    def gen(self, tag):
        if self.example == "zhang_ge_2005":
            return 3.0                       # sin, cos and one negation (scalars)
        return tvmatrix(self.n, deriv=(tag == "dA"))

    def factor(self):
        return getrf(self.n)

    def E(self, B):                          # A @ X - I  (or X @ A - I)
        n = self.n
        return _d(algebra=2 * B * n ** 3 + B * n * n)

    def dEdt(self, B):                       # A_dot @ X
        return _d(algebra=2 * B * self.n ** 3)

    def solve_J(self, B):                    # lu_solve with B*n right-hand sides
        return _d(solve=getrs(self.n, B * self.n))

    def JT(self, B):
        return _d(algebra=2 * B * self.n ** 3)


class PseudoinverseCost(ProblemCost):
    family = "pseudoinverse"
    memo = True
    factor_mode = "cached"

    def __init__(self, p):
        super().__init__(p)
        self.rows, self.cols, self.side = int(p.rows), int(p.cols), p.side
        # g = Gram size (cols for the left inverse, rows for the right one), o = other dim
        self.g, self.o = (self.cols, self.rows) if self.side == "left" else (self.rows, self.cols)
        self.dims = {"rows": self.rows, "cols": self.cols, "side": self.side}
        self.tags = {"data": ("A",), "dEdt": ("A", "dA")}
        self.notes.append("Cholesky of the Gram matrix assumed (the LU fallback for a "
                          "numerically indefinite Gram matrix is not modelled)")

    def gen(self, tag):
        return tvmatrix(self.rows, self.cols, deriv=(tag == "dA"))

    def factor(self):                        # Gram product + Cholesky (cached)
        g, o = self.g, self.o
        return 2.0 * g * g * o + potrf(g)

    def E(self, B):
        g, o = self.g, self.o
        return _d(algebra=2 * g * g * o + 2 * B * g * g * o + B * g * o)

    def dEdt(self, B):
        g, o = self.g, self.o
        return _d(algebra=4 * g * g * o + g * g + 2 * B * g * g * o + B * g * o)

    def solve_J(self, B):
        return _d(solve=potrs(self.g, B * self.o))

    def JT(self, B):
        g, o = self.g, self.o
        return _d(algebra=2 * g * g * o + 2 * B * g * g * o)


class SylvesterCost(ProblemCost):
    family = "sylvester"
    memo = True
    factor_mode = "cached"

    def __init__(self, p):
        super().__init__(p)
        self.n, self.k = int(p.n), int(p.k)
        self.skew = float(p.cfg.get("skew", 1.0))
        self.dims = {"n": self.n, "k": self.k}
        self.tags = {"data": ("A", "B", "C"), "dEdt": ("dA", "dB", "dC")}
        self.notes.append("Bartels-Stewart: Schur pair (2 x 25 n^3, GVL average) per data "
                          "sample; TRSYL counted as nk(n+k)")

    def gen(self, tag):
        n, k = self.n, self.k
        return {"A": symskew(n, self.skew), "dA": symskew(n, self.skew, True),
                "B": symskew(k, self.skew), "dB": symskew(k, self.skew, True),
                "C": tvvector(n * k), "dC": tvvector(n * k, True)}[tag]

    def factor(self):                        # schur(A), schur((-B)^T) and -B
        return schur_real(self.n) + schur_real(self.k) + self.k ** 2

    def E(self, B):
        n, k = self.n, self.k
        return _d(algebra=2 * B * n * k * (n + k) + 2 * B * n * k)

    dEdt = E

    def solve_J(self, B):                    # U^T R V, TRSYL per trajectory, U Y V^T
        n, k = self.n, self.k
        return _d(algebra=4 * B * n * k * (n + k), solve=B * trsyl(n, k))

    def JT(self, B):
        n, k = self.n, self.k
        return _d(algebra=2 * B * n * k * (n + k) + B * n * k)


class LyapunovCost(ProblemCost):
    family = "lyapunov"
    memo = True
    factor_mode = "cached"

    def __init__(self, p):
        super().__init__(p)
        self.n = int(p.n)
        self.example = p.cfg.get("example")
        self.skew = float(p.cfg.get("skew") or 0.0)
        self.dims = {"n": self.n, "example": self.example}
        self.tags = {"data": ("A", "Q"), "dEdt": ("dA", "dQ")}
        self.notes.append("one real Schur form of A^T (25 n^3, GVL average) per data sample; "
                          "TRSYL counted as 2 n^3 per trajectory")

    def gen(self, tag):
        n = self.n
        if self.example == "hu_2020":
            return 10.0
        return {"A": symskew(n, self.skew), "dA": symskew(n, self.skew, True),
                "Q": tvmatrix(n), "dQ": tvmatrix(n, deriv=True)}[tag]

    def factor(self):
        return schur_real(self.n)

    def E(self, B):
        n = self.n
        return _d(algebra=4 * B * n ** 3 + 2 * B * n * n)

    dEdt = E

    def solve_J(self, B):
        n = self.n
        return _d(algebra=8 * B * n ** 3, solve=B * trsyl(n, n))

    def JT(self, B):
        n = self.n
        return _d(algebra=4 * B * n ** 3 + B * n * n)


class ComplexInversionCost(ProblemCost):
    family = "complex-matrix-inversion"
    memo = True
    factor_mode = "cached"
    exact = True         # under the complex convention below (complex-aware ufunc count, tests)

    def __init__(self, p):
        super().__init__(p)
        self.n = int(p.n)
        self.dims = {"n": self.n}
        # 'Ar'/'Ai' are memoised views (real/imag parts) of the memoised complex A: no FLOPs
        self.tags = {"data": ("A", "Ar", "Ai"), "dEdt": ("dA",)}
        self.notes.append("complex arithmetic: 6 FLOPs per complex multiply, 2 per complex "
                          "add, 2 per real*complex, 1 per real+complex (convention; the O(B n^2) "
                          "embedding conversions are counted with it)")

    def gen(self, tag):
        if tag in ("Ar", "Ai"):
            return 0.0
        return tvcomplex(self.n, deriv=(tag == "dA"))

    def factor(self):                        # Ar + 1j Ai, complex LU
        return 3.0 * self.n ** 2 + getrf(self.n, cplx=True)

    def E(self, B):
        n = self.n
        return _d(algebra=8 * B * n ** 3 + 4 * B * n * n + 3 * n * n)

    def dEdt(self, B):
        n = self.n
        return _d(algebra=8 * B * n ** 3 + 3 * B * n * n)

    def solve_J(self, B):
        n = self.n
        return _d(algebra=3 * B * n * n, solve=getrs(n, B * n, cplx=True))

    def JT(self, B):
        n = self.n
        return _d(algebra=8 * B * n ** 3 + 3 * B * n * n + 3 * n * n)


# --- nonlinear equations / optimization -------------------------------------
class NonlinearEquationsCost(ProblemCost):
    family = "nonlinear-system"

    def __init__(self, p):
        super().__init__(p)
        self.variant = p.variant
        self.n = n = p.n_x
        self.skew = float(getattr(p, "skew", 0.0))
        self.dims = {"n": n, "variant": self.variant}
        self.factor_mode = "none" if self.variant == "guo2018" else "row"
        if self.variant == "guo2018":
            self.tags = {"data": ("c",), "dEdt": ("cdot",)}
        else:
            self.tags = {"data": ("A", "xs"), "dEdt": ("A", "dA", "xs", "dxs", "xsA")}

    def _A(self, deriv):
        n = self.n
        q = smooth_orthogonal(n, deriv)
        s = tvmatrix(n, deriv=deriv)
        return q + s + 3.0 * n * n

    def gen(self, tag):
        n = self.n
        if self.variant == "guo2018":
            return {"c": 6.0, "cdot": 14.0}[tag]
        return {"A": self._A(False), "dA": self._A(True), "xs": tvvector(n),
                "dxs": tvvector(n, True), "xsA": 2.0 * n * n}[tag]

    def E(self, B):
        n = self.n
        if self.variant == "guo2018":
            return _d(algebra=15 * B)
        return _d(algebra=2 * B * n * n + 4 * B * n)

    def dEdt(self, B):
        n = self.n
        if self.variant == "guo2018":
            return _d(algebra=B)
        return _d(algebra=2 * B * n * n + 6 * B * n)

    def solve_J(self, B):
        n = self.n
        if self.variant == "guo2018":            # forward substitution, no factorisation
            return _d(algebra=16 * B)
        return _d(algebra=4 * B * n, factor=B * getrf(n), solve=B * getrs(n, 1))

    def JT(self, B):
        n = self.n
        if self.variant == "guo2018":
            return _d(algebra=4 * B + 32 * B)
        return _d(algebra=4 * B * n + 2 * B * n * n)

    def jac(self, B):
        return _d(algebra=4 * B if self.variant == "guo2018" else 4 * B * self.n)


class TVUnconstrainedOptCost(ProblemCost):
    family = "unconstrained-optimization"

    def __init__(self, p):
        super().__init__(p)
        self.variant = p.variant
        self.n = n = p.n_x
        self.dims = {"n": n, "variant": self.variant}
        self.factor_mode = "row" if self.variant == "manufactured" else "call"
        if self.variant == "zhang2021":
            self.tags = {"data": ("Pq",), "dEdt": ("Pq",)}
        elif self.variant == "quadratic":
            self.tags = {"data": ("P", "v"), "dEdt": ("dP", "dv")}
        else:
            self.tags = {"data": ("P", "v"), "dEdt": ("P", "dP", "v", "dv", "xsP")}

    def gen(self, tag):
        n = self.n
        return {"Pq": 17.0, "P": tvmatrix(n), "dP": tvmatrix(n, deriv=True), "v": tvvector(n),
                "dv": tvvector(n, True), "xsP": 2.0 * n * n}[tag]

    def E(self, B):
        n = self.n
        if self.variant == "manufactured":
            return _d(algebra=2 * B * n * n + 4 * B * n)
        return _d(algebra=2 * B * n * n + B * n)

    def dEdt(self, B):
        n = self.n
        if self.variant == "manufactured":
            return _d(algebra=2 * B * n * n + 8 * B * n)
        return _d(algebra=2 * B * n * n + B * n)

    def solve_J(self, B):
        n = self.n
        if self.variant == "manufactured":       # batched solve with per-row Hessians
            return _d(algebra=6 * B * n, factor=B * getrf(n), solve=B * getrs(n, 1))
        return _d(factor=getrf(n), solve=getrs(n, B))   # np.linalg.solve(P, r.T): no cache

    def JT(self, B):
        n = self.n
        if self.variant == "manufactured":
            return _d(algebra=6 * B * n + 2 * B * n * n)
        return _d(algebra=2 * B * n * n)

    def jac(self, B):
        return _d(algebra=6 * B * self.n) if self.variant == "manufactured" else {}


def _qp_gen(qp):
    """``optimization._QPData.__call__`` (values AND derivatives, every call)."""
    if qp.variant in ("zhang2021", "yang2025"):
        return 30.0
    n, p = qp.n, qp.p
    f = tvmatrix(n) + tvmatrix(n, deriv=True) + tvvector(n) + tvvector(n, True)
    if p:
        f += (tvmatrix(p, n) + tvmatrix(p, n, deriv=True) + tvvector(n) + tvvector(n, True)
              + 2.0 * p * n + 4.0 * p * n + p)
    return f


class TVQPEqCost(ProblemCost):
    family = "qp"
    factor_mode = "call"

    def __init__(self, p):
        super().__init__(p)
        self.n, self.p = int(p.n), int(p.p)
        self.N = self.n + self.p
        self._gen = _qp_gen(p.qp)
        self.dims = {"n": self.n, "p": self.p, "variant": p.variant}
        self.tags = {"data": ("qp",), "dEdt": ("qp",)}

    def gen(self, tag):
        return self._gen

    def E(self, B):
        n, p = self.n, self.p
        return _d(algebra=2 * B * n * n + 4 * B * p * n + 2 * B * n + B * p)

    dEdt = E

    def solve_J(self, B):                     # np.linalg.solve(K, r.T): no cache
        return _d(factor=getrf(self.N), solve=getrs(self.N, B))

    def JT(self, B):
        return _d(algebra=2 * B * self.N ** 2)

    def jac(self, B):
        return {}                               # broadcast copy of K


class TVQPIneqCost(ProblemCost):
    family = "qp"
    factor_mode = "row"

    def __init__(self, p):
        super().__init__(p)
        self.n, self.p, self.l = int(p.n), int(p.p), int(p.l)
        self.N = self.n + self.p + self.l
        self._gen = _qp_gen(p.qp)
        self.dims = {"n": self.n, "p": self.p, "l": self.l, "variant": p.variant}
        self.tags = {"data": ("qp",), "dEdt": ("qp",)}

    def gen(self, tag):
        return self._gen

    def _jac(self, B):
        n, l = self.n, self.l
        return 3 * B * l * n + 11 * B * l

    def E(self, B):
        n, p, l = self.n, self.p, self.l
        return _d(algebra=2 * B * n * n + 4 * B * p * n + 4 * B * l * n + 3 * B * n + B * p
                  + 8 * B * l)

    def dEdt(self, B):
        n, p, l = self.n, self.p, self.l
        return _d(algebra=2 * B * n * n + 4 * B * p * n + 6 * B * l * n + 3 * B * n + B * p
                  + 10 * B * l)

    def solve_J(self, B):                     # batched solve with per-row Jacobians
        N = self.N
        return _d(algebra=self._jac(B), factor=B * getrf(N), solve=B * getrs(N, 1))

    def JT(self, B):
        return _d(algebra=self._jac(B) + 2 * B * self.N ** 2)

    def jac(self, B):
        return _d(algebra=self._jac(B))


# --- robot ---------------------------------------------------------------------
class RobotCost(ProblemCost):
    family = "robot"
    factor_mode = "row"
    kin_cached = True
    exact = True         # kinematics checked with a ufunc counter (robot numpy patched, tests)

    def __init__(self, p):
        super().__init__(p)
        self.nj = int(p.n_x)
        self.convention = p.arm.convention
        self.damping = float(p.damping)
        self.dims = {"joints": self.nj, "convention": self.convention}
        self.tags = {"data": ("rd",), "dEdt": ("rd_dot",)}
        self.notes.append("path generation counted as O(1) constants (rd: 40, rd_dot: 33 FLOPs)")

    def gen(self, tag):
        return {"rd": 40.0, "rd_dot": 33.0}[tag]

    def kin(self, rows):
        """``SerialArm.fk_jac`` with ``rows`` states: per link the DH transform
        (standard 10, modified 7 FLOPs per row), p += R p_i (21) and R = R R_i
        (54); then the tool offset (6) and the cross-product Jacobian (12 n)."""
        per_link = (10.0 if self.convention == "standard" else 7.0) + 21.0 + 54.0
        n = self.nj
        return rows * (n + n * per_link + 6.0 + 12.0 * n)

    def E(self, B):
        return _d(algebra=3 * B)

    def dEdt(self, B):
        return _d(data=3.0)                   # -r_d_dot(t), state independent

    def solve_J(self, B):
        n = self.nj
        alg = 18 * B * n + 6 * B * n + (9 * B + 9 if self.damping else 0)
        return _d(algebra=alg, factor=B * getrf(3), solve=B * getrs(3, 1))

    def JT(self, B):
        return _d(algebra=6 * B * self.nj)


class EstimatedDerivativeCost(ProblemCost):
    """``wrappers.EstimatedDerivativeProblem``: dEdt = sum_j w_j E(x, d_j) over
    the stored window (ASSUMPTION: every weight is non-zero)."""

    def __init__(self, p, inner_cost):
        self.__dict__.update(inner_cost.__dict__)
        self.inner = inner_cost
        self.window = int(p.window)
        self.family = inner_cost.family
        self.memo, self.factor_mode = inner_cost.memo, inner_cost.factor_mode
        self.kin_cached = inner_cost.kin_cached
        self.exact = False
        self.tags = {"data": inner_cost.tags["data"], "dEdt": ()}
        self.notes = list(inner_cost.notes) + [
            f"dEdt estimated from a window of {self.window} samples (window x E)"]

    def factor(self):
        return self.inner.factor()

    def gen(self, tag):
        return self.inner.gen(tag)

    def kin(self, rows):
        return self.inner.kin(rows)

    def jac(self, B):
        return self.inner.jac(B)

    def E(self, B):
        return self.inner.E(B)

    def dEdt(self, B):
        out = {}
        for _ in range(self.window):
            _add(out, self.inner.E(B))
            if self.kin_cached:
                out["kin"] = out.get("kin", 0.0) + self.inner.kin(B)
        out["algebra"] = out.get("algebra", 0.0) + 2.0 * B * self.m * self.window
        return out

    def solve_J(self, B):
        return self.inner.solve_J(B)

    def JT(self, B):
        return self.inner.JT(B)


def _innermost(problem):
    from .wrappers import ProblemWrapper
    p = problem
    while isinstance(p, ProblemWrapper):
        p = p.inner
    return p


def problem_cost(problem) -> ProblemCost:
    """Analytic cost model of ``problem`` (wrappers are looked through;
    ``EstimatedDerivativeProblem`` changes the dEdt cost)."""
    from .wrappers import EstimatedDerivativeProblem, ProblemWrapper
    est = None
    p = problem
    while isinstance(p, ProblemWrapper):
        if isinstance(p, EstimatedDerivativeProblem) and est is None:
            est = p
        p = p.inner
    from .problems.linear_system import LinearSystem
    from .problems.matrix import (ComplexInversion, Lyapunov, MatrixInversion, Pseudoinverse,
                                  Sylvester)
    from .problems.optimization import NonlinearEquations, TVQPEq, TVQPIneq, TVUnconstrainedOpt
    from .problems.robot import ManipulatorTracking
    table = [(MatrixInversion, MatrixInversionCost), (Pseudoinverse, PseudoinverseCost),
             (Sylvester, SylvesterCost), (Lyapunov, LyapunovCost),
             (ComplexInversion, ComplexInversionCost), (LinearSystem, LinearSystemCost),
             (NonlinearEquations, NonlinearEquationsCost),
             (TVUnconstrainedOpt, TVUnconstrainedOptCost), (TVQPIneq, TVQPIneqCost),
             (TVQPEq, TVQPEqCost), (ManipulatorTracking, RobotCost)]
    for cls, cost_cls in table:
        if isinstance(p, cls):
            c = cost_cls(p)
            return EstimatedDerivativeCost(est, c) if est is not None else c
    raise NotImplementedError(
        f"no structured cost model for problem {getattr(p, 'name', p)!r} (family "
        f"{getattr(p, 'family', '?')!r}); use wrappers.dense_flops as a fallback")


# ===========================================================================
# 4. activations and error maps (element-wise work, approximate)
# ===========================================================================
#: FLOPs per element of Phi(e) under the ufunc convention (read off the code
#: of activations.py; transcendental = 1).
ACT_CALL = {"linear": 0, "power": 1, "bipolar_sigmoid": 3, "power_sigmoid": 5, "hyperbolic_sine": 2,
            "sign": 1, "sign_power": 4, "sign_bi_power": 10, "sfaf": 7, "wsbp": 13,
            "saturated_sign_power": 5, "vaf": 11, "fixed_time": 10, "pvaf1": 12, "pvaf2": 10,
            "predefined_exp": 12, "strict_predefined": 8, "li": 9}
ACT_DEFAULT = 12
PROX_ITERS = 80        # Activation.prox: geometric bisection iterations


def activation_flops(act):
    """FLOPs of ``act`` per element: {'call': Phi(e), 'prox': resolvent per
    element, 'prox_row': resolvent work per state row (h-dependent scalars)}.

    Closed-form resolvents: linear e/(1+h) (1 per element + 1 per row), sign
    soft-threshold (5), power p = 3 hyperbolic Cardano (9 + 3 per row).  Every
    other activation uses the geometric bisection of ``Activation.prox``
    (80 iterations): 80 (c + 8) + c + 13 per element with c = call cost
    (checked with a ufunc-level counter).  'strict_predefined' is counted in
    its post-t_c phase.  ASSUMPTION: unknown activations cost 12 per element;
    the power p >= 5 Newton resolvent is counted as 20 iterations of 6."""
    if act is None:
        return {"call": 0.0, "prox": 1.0, "prox_row": 1.0}
    key = getattr(act, "key", None)
    if key == "power_sum":
        c = 2.0 * getattr(act, "N", 3)
    elif key == "signed_power_sum":          # consistent.PowerSum: sum a_i sig^p_i (6 per term)
        c = 6.0 * len(getattr(act, "terms", ()))
    elif key == "jr_predefined":             # consistent.PredefinedTimeJR, JR Eq. (1)
        c = 18.0 if getattr(act, "kappa_name", "") == "rational" else 17.0
    else:
        c = float(ACT_CALL.get(key, ACT_DEFAULT))
    if key == "linear":
        return {"call": c, "prox": 1.0, "prox_row": 1.0}
    if key == "sign":
        return {"call": c, "prox": 5.0, "prox_row": 0.0}
    if key == "power":
        if getattr(act, "p", 3) == 3:
            return {"call": c, "prox": 9.0, "prox_row": 3.0}
        return {"call": c, "prox": 120.0, "prox_row": 0.0}
    return {"call": c, "prox": PROX_ITERS * (c + 8.0) + c + 13.0, "prox_row": 0.0}


def _peb_terms_flops(terms):
    """``consistent.peb_map``: one ``_peb_single`` per power term a|e|^p sign(e).
    Per element: |e|, the branch (p < 1: **, -, max, **; p = 1: /; p > 1: **,
    1/., +, 1/., **) and sign(e) * s (2): 7 / 4 / 8.  Per row: a h and the
    h-dependent scalar (2)."""
    elem = 0.0
    for _, p in terms:
        elem += 7.0 if p < 1 else (4.0 if p == 1 else 8.0)
    return elem, 2.0 * len(terms)


def _jr_exact_flops(jr):
    """``PredefinedTimeJR.exact_step`` (JR Eq. (7)) per element and per row:
    |e|, kappa, **(1 - rho2), - h/rho1, max, **(1/(1 - rho2)), clip, kappa^-1,
    **(1/mu), min, sign * (h/rho1 is the row term)."""
    kappa = {"arctan": 4.0, "exp": 5.0, "rational": 6.0}[jr.kappa_name]
    kappa_inv = {"arctan": 3.0, "exp": 4.0, "rational": 3.0}[jr.kappa_name]
    return 1.0 + kappa + 5.0 + kappa_inv + 4.0, 1.0


def _exact_closed_flops(act, cs):
    """Cost (per element, per row) of ``consistent._exact_closed_form``, or
    None where it returns None (no closed form: the map chain falls back)."""
    if hasattr(act, "exact_step"):
        if hasattr(act, "kappa_name"):
            return _jr_exact_flops(act)
        return 2.0 * ACT_DEFAULT, 0.0      # ASSUMPTION: unknown exact_step (none registered)
    jr = cs._jr_equivalent(act)
    if jr is not None:
        return _jr_exact_flops(jr)
    terms = cs.power_terms(act)
    if terms is None:
        return None
    if len(terms) == 1:
        if terms[0][1] == 1:
            return 1.0, 3.0                # e * exp(-a h)
        return _peb_terms_flops(terms)     # single power term: the PEB map is the exact flow
    if len(terms) == 2:
        (_, p1), (_, p2) = terms
        if p1 < 1 < p2:
            return 30.0, 0.0               # _exact_bipower (betainc / betaincinv)
        if (p2 == 1 and p1 < 1) or (p1 == 1 and p2 > 1):
            return 8.0, 6.0                # _exact_bernoulli_low / _high
    return None


def _error_map_flops(name, act):
    """(per element, per row) FLOPs of ``consistent.apply_error_map(name, act,
    E, h)``, following its fallback chain ('exact' -> 'peb' -> 'resolvent',
    'peb' -> 'exact' -> 'resolvent').  Checked against a ufunc-level count for
    every registered activation (tests).  The constant probes of
    ``power_terms`` / ``jump0`` (a few elements, cached or B-independent) are
    ignored."""
    from .discretizations import consistent as cs
    af = activation_flops(act)
    for tried in cs._CHAIN[name]:
        if tried == "resolvent":
            return af["prox"], af["prox_row"]
        if tried == "semi_implicit":
            j = float(act.jump0())
            if not np.isfinite(j) or j < cs._TINY_JUMP:
                return af["call"] + 10.0, 0.0
            return af["call"] + 12.0, 1.0      # + soft threshold s - h j (h j per row), max
        if tried == "peb":
            terms = cs.power_terms(act)
            if terms is not None:
                return _peb_terms_flops(terms)
        if tried == "exact":
            out = _exact_closed_flops(act, cs)
            if out is not None:                 # + sign(e) min(|out|, |e|)
                return out[0] + 5.0, out[1]
    raise RuntimeError("unreachable")  # pragma: no cover


# ===========================================================================
# 5. models: problem calls made inside rhs and own numpy work
# ===========================================================================
def _act_of(model):
    act = getattr(model, "act", None)
    return act


class ModelCost:
    """Calls made by ``model.rhs`` for one evaluation, and its own FLOPs.

    ``evaluate(B, causal, info, implicit)`` returns ``(calls, own)`` where
    calls is a list of (prim, rows, state_tag) in call order and own the
    model's numpy FLOPs.  ``implicit`` is what the discretizer passes: False,
    True (resolvent) or a dict describing an ErrorMapRequest.  ``uses_E``:
    whether ``rhs`` reads ``ctx.E`` (simulate evaluates it for every model)."""

    uses_E = True

    def __init__(self, model, problem, pcost):
        self.model = model
        self.name = getattr(model, "name", type(model).__name__)
        self.m, self.n_x = pcost.m, pcost.n_x
        self.n_z = int(model.n_z(problem))
        self.uses_dEdt = bool(getattr(model, "uses_dEdt", True))
        self.notes = []

    def evaluate(self, B, causal, info, implicit):
        raise NotImplementedError


def _rate_flops(model, B, m, implicit):
    """``DesignFormulaZNN.desired_Edot`` (+ the r - dE/dt subtraction of rhs)."""
    act = _act_of(model)
    af = activation_flops(act)
    c, prox = af["call"], af["prox"]
    if isinstance(implicit, dict):                       # ErrorMapRequest
        s = implicit["s"]
        f = 2 * B                                        # g = gamma g(t); h = g tau
        elem, row = _error_map_flops(implicit["error_map"], act)
        f += elem * B * m + row * B
        if implicit["multistep"] and implicit["memory"] and s >= 1:
            f += (3 + 2 * (s - 1)) * B * m
        else:
            f += B * m
        f += B * m                                       # / tau
        return f + B * m
    if implicit:
        return 2 * B + af["prox_row"] * B + prox * B * m + 3 * B * m + B * m
    return 2 * B + (1 + c) * B * m + B * m


class DesignFormulaCost(ModelCost):
    """znn, vpznn, vpznn_exp, znn_consistent: J x_dot = r - dE/dt (one solve)."""

    def evaluate(self, B, causal, info, implicit):
        return [("solve_J", B, "")], _rate_flops(self.model, B, self.m, implicit)


class ExplicitInverseCost(ModelCost):
    """znn_explicit_inverse(_right): x_dot = X (r - dE/dt): one batched n x n
    matmul, no solve (the products A X and A_dot X are the problem's E and
    dEdt)."""

    def evaluate(self, B, causal, info, implicit):
        n = math.isqrt(self.n_x)
        return [], _rate_flops(self.model, B, self.m, implicit) + gemm(n, n, n) * B


class GNNCost(ModelCost):
    def evaluate(self, B, causal, info, implicit):
        c = activation_flops(_act_of(self.model))["call"]
        return [("JT", B, "")], B + c * B * self.m + B * self.n_x


class DavidenkoCost(ModelCost):
    """x_dot = -J^{-1} dE/dt: ``ctx.E`` is never read (Simonetto et al. 2016,
    Eq. (7) needs J and grad_tx f only)."""

    uses_E = False

    def evaluate(self, B, causal, info, implicit):
        return [("solve_J", B, "")], B * self.m


class NewtonFlowCost(ModelCost):
    def evaluate(self, B, causal, info, implicit):
        return [("solve_J", B, "")], B + B * self.m


class IEZNNCost(ModelCost):
    def evaluate(self, B, causal, info, implicit):
        row = 3 * B if self.model.lam is None else B
        return [("solve_J", B, "")], row + 4 * B * self.m


class DIEZNNCost(ModelCost):
    def evaluate(self, B, causal, info, implicit):
        row = 7 * B if self.model.xi is None else 0
        return [("solve_J", B, "")], row + 6 * B * self.m


class HNTZNNCost(ModelCost):
    def evaluate(self, B, causal, info, implicit):
        L = len(self.model.omegas)
        c = activation_flops(self.model.act)["call"]
        return [("solve_J", B, "")], B + (c + 4 * L + 2) * B * self.m


class SPTCNTCost(ModelCost):
    """Counted in the t >= t_c phase (most of the horizon); before t_c the
    model's own work is 2B + 2Bm instead of 2B + 10Bm (ASSUMPTION)."""

    def evaluate(self, B, causal, info, implicit):
        return [("solve_J", B, "")], 2 * B + 10 * B * self.m


class SPTCANFOCost(ModelCost):
    """Counted in the t >= t_c phase (2B + 10Bm; before t_c B + 9Bm).  The
    row norm ||E|| costs 2Bm (x*x, row sums, sqrt)."""

    def evaluate(self, B, causal, info, implicit):
        return [("solve_J", B, "")], 2 * B + 10 * B * self.m


class GZNNCost(ModelCost):
    """gznn: dense J from one JT call with B*m rows; dJ/dt from one dEdt call
    with 2 B n_x rows (analytic), or 1-2 more dense-J JT calls (bd1/bd2);
    M = J^T J (2 B m n^2) and a batched n_x x n_x solve."""

    def regime(self, causal, info):
        j = self.model.jdot
        reg = info if j == "auto" else j
        if reg in ("bd1", "bd2") and not causal:
            reg = "analytic"
        return reg

    def evaluate(self, B, causal, info, implicit):
        m, n = self.m, self.n_x
        calls = [("JT", B * m, "rep")]
        own = B + 4 * B * m * n + 2 * B * n                     # r = -g J^T E - J^T dE/dt
        reg = self.regime(causal, info)
        if reg == "analytic":
            calls.append(("dEdt", 2 * B * n, "fd"))
            own += 3 * B * n * n + 2 * B * n * m + 2 * B * n + 2 * B   # finite-difference dJ/dt
        elif reg == "bd1":
            calls += [("JT", B * m, "rep"), ("JT", B * m, "rep")]
            own += 2 * B * m * n
        elif reg == "bd2":
            calls += [("JT", B * m, "rep")] * 3
            own += 5 * B * m * n
        if reg != "none":
            own += 2 * B * m * n + B * n                        # r -= Jd^T E
        own += 2 * B * m * n * n                                # M = J^T J
        own += n * n                                            # reg = beta I (always built)
        reg_mode = self.model.regularize
        if reg_mode == "always" or (reg_mode == "auto" and m < n):
            own += B * n * n                                    # M + reg
        own += B * (getrf(n) + getrs(n, 1))
        return calls, own


class LCCZNNCost(ModelCost):
    """lcc_znn (models/structured_extra.py, Eq. (12) of Zheng-Li-Zhang 2024):
    one J^T f product, three inner products and a scalar activation of the
    energy per row; no solve.  (The inverse-free ZNN of Chen et al. 2024 is NOT
    registered -- structured_extra.NOT_REGISTERED; its secondary restatement
    is lcc_znn with activation='linear', c_phi = 0.)"""

    PHI = {"exp_sig": 7.0, "sig": 4.0, "sig_lin": 5.0, "linear": 0.0}

    def evaluate(self, B, causal, info, implicit):
        phi = self.model.phi
        kind = getattr(phi, "kind", None)
        if kind in self.PHI:
            c = self.PHI[kind]
        else:
            c = activation_flops(getattr(phi, "act", None))["call"]
        n, m = self.n_x, self.m
        own = 2 * B * n + B + 2 * B * m + 2 * B * m + B + c * B + B + 2 * B + B + B * n
        return [("JT", B, "")], own


class HybridGNNZNNCost(ModelCost):
    """hgnn_znn / fa_gnn_znn (Dai et al. 2022, Eqs. (10)-(11)): one solve and
    one J^T product.  The fuzzy factor of fa_gnn_znn evaluates the Mamdani
    max-min aggregation on its output grid (G points): ~10 B G FLOPs."""

    def evaluate(self, B, causal, info, implicit):
        md = self.model
        n, m = self.n_x, self.m
        c = activation_flops(md.act)["call"]
        own = c * B * m + B + B * m + 2 * B * n
        if md.feedforward:
            own += B * m
        fz = md.fuzzy
        if fz is not None:
            G = int(getattr(fz, "grid", np.zeros(1001)).size)
            own += 2 * B * m + 22 * B + 10 * B * G + B      # norm, memberships, max-min, mom, g + nu
        return [("solve_J", B, ""), ("JT", B, "")], own


class ProbedModelCost(ModelCost):
    """Fallback for models without an analytic cost class: the problem calls
    made by one ``rhs`` evaluation are recorded (deep copy of the model,
    call-recording CountingProblem); the model's own numpy work is NOT
    counted (flagged in the notes)."""

    def __init__(self, model, problem, pcost):
        super().__init__(model, problem, pcost)
        self.problem = problem
        self._cache = {}
        self.notes.append(f"model {self.name!r}: no analytic cost class; problem calls probed "
                          "from one rhs evaluation, own numpy work not counted")

    def evaluate(self, B, causal, info, implicit):
        key = (B, causal, info, bool(implicit))
        if key not in self._cache:
            self._cache[key] = self._probe(B, causal, info, bool(implicit))
        return list(self._cache[key]), 0.0

    def _probe(self, B, causal, info, implicit):
        import copy
        from .core import Context
        from .simulate import initial_state
        P = self.problem
        rec = _recording_problem_class()(P)
        mdl = copy.deepcopy(self.model)
        x = initial_state(P, mdl, 0, B)
        d = P.data(0.0)
        dEdt = P.dEdt(x, 0.0) if info != "none" else np.zeros((B, P.m))
        ctx = Context(problem=rec, t=0.0, tau=1e-2, k=0 if causal else -1, x=x,
                      z=mdl.z0(P, B), d=d, E=P.E(x, d), dEdt=dEdt,
                      gamma=np.full((B, 1), 10.0), implicit=implicit, extra={"info": info})
        rec.reset_counts()
        with np.errstate(all="ignore"):
            mdl.rhs(ctx)
        calls = []
        for prim in ("dEdt", "E", "solve_J", "JT"):
            c = rec.calls[prim]
            for _ in range(c):
                calls.append((prim, rec.rows[prim] // c, "probe"))
        return calls


def model_cost(model, problem, pcost=None, probe=True) -> ModelCost:
    """Cost descriptor of ``model`` (registered models; ``runner.PeakXdotProbe``
    is looked through).  Unknown models fall back to :class:`ProbedModelCost`
    when ``probe`` (their own numpy work is then not counted)."""
    inner = getattr(model, "_inner", None)
    if inner is not None and type(model).__name__ == "PeakXdotProbe":
        model = inner
    pcost = pcost or problem_cost(problem)
    from .models.baselines import Davidenko, NewtonFlow
    from .models.structured import ExplicitInverseZNN
    from .models.variants import DIEZNN, GZNN, HNTZNN, IEZNN, SPTCANFOZNN, SPTCNTZNN
    from .models.znn import GNN, DesignFormulaZNN
    table = [(ExplicitInverseZNN, ExplicitInverseCost), (DesignFormulaZNN, DesignFormulaCost),
             (GNN, GNNCost), (Davidenko, DavidenkoCost), (NewtonFlow, NewtonFlowCost),
             (IEZNN, IEZNNCost), (DIEZNN, DIEZNNCost), (HNTZNN, HNTZNNCost),
             (SPTCNTZNN, SPTCNTCost), (SPTCANFOZNN, SPTCANFOCost), (GZNN, GZNNCost)]
    try:
        from .models.structured_extra import LCCZNN, HybridGNNZNN
        table += [(LCCZNN, LCCZNNCost), (HybridGNNZNN, HybridGNNZNNCost)]
    except ImportError:                      # module of another area not present
        pass
    for cls, cost_cls in table:
        if isinstance(model, cls):
            return cost_cls(model, problem, pcost)
    if probe:
        try:
            return ProbedModelCost(model, problem, pcost)
        except Exception as exc:  # noqa: BLE001
            raise NotImplementedError(f"no cost model for model {getattr(model, 'name', model)!r} "
                                      f"(probe failed: {exc!r})") from exc
    raise NotImplementedError(f"no cost model for model {getattr(model, 'name', model)!r}")


# ===========================================================================
# 6. discretizations: per-step schedule
# ===========================================================================
# Schedule items (one steady-state step, in call order):
#   ("f", offset, kind, state, implicit)  one evaluation of simulate's vector field
#   ("corr", offset, kind, mode, C, start_state)  C correction steps on the frozen problem
#   ("stage_data", offset)                one stage_data(t_k + offset*tau) sample
#   ("own", category, flops)              discretizer / wrapper vector work
# offset is in units of tau; kind 'past' = simulate's measured sample at t_k,
# 'stage' = a stage_data sample.

def _unwrap_disc(disc):
    """(innermost discretizer, list of (wrapper, info))."""
    from .problems.robot import StateRecorder
    from .wrappers import QuantizedDiscretizer
    wrappers = []
    d = disc
    while True:
        if isinstance(d, QuantizedDiscretizer):
            wrappers.append(("quant", d))
            d = d.disc
        elif isinstance(d, StateRecorder):
            d = d.inner
        elif type(d).__name__ == "_Window" and hasattr(d, "inner"):
            d = d.inner
        else:
            return d, wrappers


def _schedule(disc, problem, B, n_x, n_z):
    from .discretizations.adams import AdamsBashforth
    from .discretizations.consistent import ConsistentLookAhead
    from .discretizations.multistep import RK4, LookAhead
    from .discretizations.prediction_correction import PredictionCorrection, is_gradient_field
    d, wrappers = _unwrap_disc(disc)
    items = []
    notes = []
    if isinstance(d, ConsistentLookAhead):
        s = d.s
        # steady state: len(hist) == s - 1 always holds (also for Euler, s = 1)
        req = {"error_map": d.error_map, "s": s, "memory": bool(d.memory), "multistep": True}
        items.append(("f", 0.0, "past", "x", req))
        items.append(("own", "disc", (2 * s + 2) * B * n_x + 2 * B * n_z))
    elif isinstance(d, LookAhead):
        s = d.s
        items.append(("f", 0.0, "past", "x", bool(getattr(d, "implicit", False))))
        items.append(("own", "disc", (2 * s + 2) * B * n_x + 2 * B * n_z))
    elif isinstance(d, AdamsBashforth):
        q = d.q
        items.append(("f", 0.0, "past", "x", bool(getattr(d, "implicit", False))))
        zf = (2 * q + 2) if (d.z_scheme == "ab" and n_z) else 2
        items.append(("own", "disc", (2 * q + 2) * B * n_x + zf * B * n_z))
    elif isinstance(d, RK4):
        items += [("f", 0.0, "past", "x", False), ("f", 0.5, "stage", "s2", False),
                  ("f", 0.5, "stage", "s3", False), ("f", 1.0, "stage", "s4", False),
                  ("own", "disc", 13 * B * n_x + 13 * B * n_z)]
    elif isinstance(d, PredictionCorrection):
        if d.correction == "gradient":
            direction = d.direction
            if direction == "auto":
                direction = "residual" if is_gradient_field(_innermost(problem)) else "normal"
            mode = f"gradient-{direction}"
        else:
            mode = "newton"
        C = int(d.C)
        if d.causal:
            if C:
                items.append(("corr", 0.0, "past", mode, C, "x"))
            if d.predict:
                items.append(("f", 0.0, "past", ("c", C) if C else "x", False))
                items.append(("own", "disc", 2 * B * n_x + 2 * B * n_z))
        else:
            if d.predict:
                items.append(("f", 0.0, "past", "x", False))
                items.append(("own", "disc", 2 * B * n_x + 2 * B * n_z))
            if C:
                items.append(("stage_data", 1.0))
                items.append(("corr", 1.0, "stage", mode, C, "xp" if d.predict else "x"))
        if d.correction == "gradient" and d.alpha == "auto":
            notes.append("alpha='auto': one-off Lipschitz estimate (30 power iterations) at "
                         "the first correction, reported under 'setup'")
    elif type(d).__name__ == "BroydenDTZNN":
        items.append(("broyden", d.secant))
        notes.append("inverse-free: J^-1 replaced by the Broyden matrix M_k (dense n_x x n_x per "
                     "trajectory); model solves cost one batched mat-vec 2 B n_x^2")
    elif type(d).__name__ == "CDTZNN":
        items.append(("cdtznn", d.xdot, d.info))
        notes.append("inverse-free CDTZNN: dense J (problem.jacobian or JT with B m rows), "
                     "finite-difference Jdot, four batched n_x^3 matmuls per step")
    else:
        raise NotImplementedError(f"no cost schedule for discretization "
                                  f"{getattr(d, 'name', type(d).__name__)!r}")
    for kind, w in wrappers:
        if kind == "quant":
            q = 8 * B * n_x + (8 * B * n_z if (w.quantize_z and n_z) else 0)
            items.append(("own", "disc", q))
            if w.dfmt is not None:              # one vector-field call per f / broyden / cdtznn item
                nf = sum(1 for it in items if it[0] in ("f", "broyden", "cdtznn"))
                items.append(("own", "disc", nf * (4 * B * n_x + 4 * B * n_z)))
    return d, items, notes


# ===========================================================================
# 7. evaluator (caches emulated over consecutive steps)
# ===========================================================================
class _Acc:
    def __init__(self):
        self.cat = defaultdict(float)
        self.prim = defaultdict(float)
        self.calls = defaultdict(float)
        self.rows = defaultdict(float)
        self.gens = defaultdict(float)
        self.factorizations = 0.0
        self.kin_rows = 0.0
        self.dense = 0.0

    def scaled(self, s):
        out = _Acc()
        for name in ("cat", "prim", "calls", "rows", "gens"):
            for k, v in getattr(self, name).items():
                getattr(out, name)[k] = v * s
        out.factorizations = self.factorizations * s
        out.kin_rows = self.kin_rows * s
        out.dense = self.dense * s
        return out

    def plus(self, other):
        for name in ("cat", "prim", "calls", "rows", "gens"):
            mine = getattr(self, name)
            for k, v in getattr(other, name).items():
                mine[k] += v
        self.factorizations += other.factorizations
        self.kin_rows += other.kin_rows
        self.dense += other.dense
        return self


class _Engine:
    """Expands schedules into problem calls and charges them, emulating the
    factorisation cache (one entry, keyed by data realisation), the data memo
    (keyed by (tag, instant)) and the robot kinematics cache (one entry, keyed
    by state)."""

    def __init__(self, pcost, mcost, B, info, noise, reuse, data_noise, skip_unused_dEdt,
                 skip_unused_E=False):
        self.pc, self.mc, self.B = pcost, mcost, B
        self.info, self.noise = info, bool(noise)
        self.reuse, self.data_noise = reuse, bool(data_noise)
        self.skip_dEdt = bool(skip_unused_dEdt) and not mcost.uses_dEdt
        self.skip_E = bool(skip_unused_E) and not getattr(mcost, "uses_E", True)
        self.fac_key = None
        self.kin_key = None
        self.memo = set()
        self.rid = 0
        self.acc = _Acc()

    # --- keys -----------------------------------------------------------------
    def inst(self, j, off, kind):
        t = j + off
        return t if self.reuse == "ideal" else (t, kind)

    def realization(self, j, off, kind, fresh):
        key = self.inst(j, off, kind)
        if self.data_noise and kind == "stage":
            return (key, fresh)
        return key

    # --- charging -----------------------------------------------------------------
    def _gen(self, prim, inst_key):
        for tag in self.pc.tags.get(prim, ()):
            if self.pc.memo:
                k = (tag, inst_key)
                if k in self.memo:
                    continue
                self.memo.add(k)
            f = self.pc.gen(tag)
            self.acc.cat["data"] += f
            self.acc.prim[prim] += f
            self.acc.gens[tag] += 1

    def call(self, prim, rows, inst_key, real_key, state):
        acc, pc = self.acc, self.pc
        acc.calls[prim] += 1
        acc.rows[prim] += rows
        if prim == "data":
            self._gen("data", inst_key)
            return
        from .wrappers import dense_flops
        acc.dense += dense_flops(pc.n_x, pc.m, rows)[prim]
        if prim == "dEdt":
            self._gen("dEdt", inst_key)
        if pc.kin_cached and prim in ("E", "solve_J", "JT"):
            if state != self.kin_key:
                f = pc.kin(rows)
                acc.cat["kin"] += f
                acc.prim[prim] += f
                acc.kin_rows += rows
                self.kin_key = state
        if prim == "solve_J":
            mode = pc.factor_mode
            if mode == "cached" and real_key != self.fac_key:
                f = pc.factor()
                acc.cat["factor"] += f
                acc.prim[prim] += f
                acc.factorizations += 1
                self.fac_key = real_key
            elif mode == "call":
                acc.factorizations += 1
            elif mode == "row":
                acc.factorizations += rows
        for cat, f in pc.call(prim, rows).items():
            acc.cat[cat] += f
            acc.prim[prim] += f

    def own(self, cat, f):
        self.acc.cat[cat] += f

    # --- expansion -------------------------------------------------------------
    def f_eval(self, j, off, kind, state, implicit, subst=False):
        """One call of simulate's vector field.  ``subst``: the discretizer
        shadows problem.solve_J by r -> M r (inverse-free schemes)."""
        B, info = self.B, self.info
        st = (j, state)
        if kind == "stage":
            self.rid += 1
            real = self.realization(j, off, kind, self.rid)
            self.call("data", 0, self.inst(j, off, kind), real, None)
        else:
            real = self.realization(j, off, kind, None)
        ikey = self.inst(j, off, kind)
        causal = kind == "past"
        if not self.skip_dEdt:
            if causal:
                if info == "analytic":
                    self.call("dEdt", B, ikey, real, st)
                elif info in ("bd1", "bd2"):
                    nE = 2 if info == "bd1" else 3
                    for back in range(nE):
                        bk = self.inst(j, off - back, "past")
                        self.call("E", B, bk, bk, st)
                    self.own("sim", (2 if info == "bd1" else 5) * B * self.pc.m)
            elif info != "none":
                self.call("dEdt", B, ikey, real, st)
        if not self.skip_E:                     # ctx.E (skipped for models that never read it)
            self.call("E", B, ikey, real, st)
        calls, own = self.mc.evaluate(B, causal, info, implicit)
        n = self.pc.n_x
        for prim, rows, tag in calls:
            if subst and prim == "solve_J" and rows == B:
                self.own("disc", 2.0 * B * n * n)            # einsum M r
                continue
            self.call(prim, rows, ikey, real, (j, state, tag) if tag else st)
        self.own("model", own)
        if self.noise:
            if subst:
                self.own("disc", 2.0 * B * n * n)
            else:
                self.call("solve_J", B, ikey, real, st)
            self.own("sim", B * n)

    def dense_J(self, j, ikey, real, state):
        """inverse_free_dtznn.dense_jacobian: problem.jacobian or one JT call
        with B*m rows."""
        B, pc = self.B, self.pc
        jd = pc.jac(B)
        if jd is None:
            self.call("JT", B * pc.m, ikey, real, (j, state, "rep"))
            return
        for cat, f in jd.items():
            self.acc.cat[cat] += f
            self.acc.prim["jacobian"] += f
        self.acc.calls["jacobian"] += 1

    def broyden(self, j, secant):
        B, n, m = self.B, self.pc.n_x, self.pc.m
        off = 0.0 if secant == "current" else -1.0
        dk = self.inst(j, off, "past")
        self.call("E", B, dk, dk, (j, "x"))
        self.call("E", B, dk, dk, (j - 1, "x"))
        self.own("disc", 7.0 * B * n * n + 8.0 * B * n + B * m + 3.0 * B)   # secant update
        self.f_eval(j, 0.0, "past", "x", False, subst=True)
        self.own("disc", 2.0 * B * n + 2.0 * B * self.mc.n_z)

    def cdtznn(self, j, xdot_mode, info_override):
        from .models.znn import DesignFormulaZNN
        B, n, m, info = self.B, self.pc.n_x, self.pc.m, self.info
        ikey = self.inst(j, 0.0, "past")
        self.f_eval(j, 0.0, "past", "x", False, subst=True)
        if not isinstance(self.mc.model, DesignFormulaZNN):     # context not exposed
            info = info_override or info
            st = (j, "x")
            self.call("E", B, ikey, ikey, st)
            if info == "analytic":
                self.call("dEdt", B, ikey, ikey, st)
            elif info in ("bd1", "bd2"):
                for back in range(2 if info == "bd1" else 3):
                    bk = self.inst(j, -back, "past")
                    self.call("E", B, bk, bk, st)
                self.own("disc", (2 if info == "bd1" else 5) * B * m)
        self.dense_J(j, ikey, ikey, "x")
        if xdot_mode == "ctznn_unit_gain":
            self.own("disc", 2.0 * B * n * n + 2.0 * B * n)
        elif xdot_mode == "printed":
            self.own("disc", B * n)
        # D_x J [v]: two dense Jacobians at x +- eps v
        self.own("disc", 8.0 * B * n + 3.0 * B)
        self.dense_J(j, ikey, ikey, "jp")
        self.dense_J(j, ikey, ikey, "jm")
        self.own("disc", 2.0 * B * m * n + B + gemm(m, n, n) * B)
        # d_t J M per information regime
        if info == "analytic":
            k = n
            self.own("disc", 4.0 * B * k * n + 3.0 * B * k + 4.0 * B * k * n + 2.0 * B * k * m
                     + B * k)
            self.call("dEdt", B * k, ikey, ikey, (j, "dtp"))
            self.call("dEdt", B * k, ikey, ikey, (j, "dtm"))
        elif info == "bd1":
            self.dense_J(j, self.inst(j, -1, "past"), self.inst(j, -1, "past"), "x1")
            self.own("disc", 2.0 * B * m * n + gemm(m, n, n) * B)
        elif info == "bd2":
            for back in (1, 2):
                bk = self.inst(j, -back, "past")
                self.dense_J(j, bk, bk, f"x{back}")
            self.own("disc", 5.0 * B * m * n + gemm(m, n, n) * B)
        # JdM sum, M JdM, J M, M (J M) - M, and the two 5-instant recursions
        self.own("disc", B * m * n + 3 * gemm(n, n, n) * B + B * n * n
                 + 11.0 * B * n * n + 9.0 * B * n + 2.0 * B * self.mc.n_z)

    def correction(self, j, off, kind, mode, C, start):
        B, n = self.B, self.pc.n_x
        real = self.realization(j, off, kind, self.rid)
        ikey = self.inst(j, off, kind)
        for s in range(C):
            st = (j, "corr", start, s)
            self.call("E", B, ikey, real, st)
            if mode == "newton":
                self.call("solve_J", B, ikey, real, st)
                self.own("disc", B * n)
            else:
                if mode == "gradient-normal":
                    self.call("JT", B, ikey, real, st)
                self.own("disc", 2 * B * n)

    def step(self, j, items):
        self.call("data", 0, self.inst(j, 0.0, "past"), self.realization(j, 0.0, "past", None),
                  None)
        for it in items:
            if it[0] == "f":
                _, off, kind, state, imp = it
                self.f_eval(j, off, kind, state, imp)
            elif it[0] == "stage_data":
                self.rid += 1
                off = it[1]
                self.call("data", 0, self.inst(j, off, "stage"),
                          self.realization(j, off, "stage", self.rid), None)
            elif it[0] == "corr":
                _, off, kind, mode, C, start = it
                self.correction(j, off, kind, mode, C, start)
            elif it[0] == "own":
                self.own(it[1], it[2])
            elif it[0] == "broyden":
                self.broyden(j, it[1])
            elif it[0] == "cdtznn":
                self.cdtznn(j, it[1], it[2])


def _steady(pcost, mcost, items, B, info, noise, reuse, data_noise, skip_unused_dEdt,
            skip_unused_E=False, n_steps=3):
    """Run the schedule for ``n_steps`` consecutive steps and return the cost of
    the last one (caches warm, no start-up effects)."""
    eng = _Engine(pcost, mcost, B, info, noise, reuse, data_noise, skip_unused_dEdt,
                  skip_unused_E)
    for j in range(n_steps - 1):
        eng.step(j, items)
    eng.acc = _Acc()
    eng.step(n_steps - 1, items)
    return eng.acc


def float_reuse_fraction(tau, k_lo, k_hi):
    """Fraction of k in [k_lo, k_hi) with float(k tau + tau) == float((k+1) tau).

    simulate samples at t_k = k * tau while RK4 / the non-causal
    prediction-correction schemes sample their last stage at t_k + tau; the
    data memo and the factorisation cache of the matrix problems are keyed by
    that float, so the stage sample is reused at the next step only when the
    two floats coincide."""
    tau = float(tau)
    ks = range(int(k_lo), int(k_hi))
    if not len(ks):
        return 1.0
    return sum(1 for k in ks if (k * tau + tau) == ((k + 1) * tau)) / len(ks)


def _setup_cost(pcost, d, B, problem):
    """One-off Lipschitz estimate of PredictionCorrection(alpha='auto'):
    30 power iterations of a central-difference Jv (2 E calls) [+ JT for the
    'normal' direction, i.e. problems that are not gradient fields]."""
    from .discretizations.prediction_correction import PredictionCorrection, is_gradient_field
    if not (isinstance(d, PredictionCorrection) and d.correction == "gradient"
            and d.alpha == "auto"):
        return None
    n, m = pcost.n_x, pcost.m
    iters = 30
    direction = d.direction
    if direction == "auto":
        direction = "residual" if is_gradient_field(_innermost(problem)) else "normal"
    normal = direction == "normal"
    e = _add({}, pcost.E(B), 2 * iters)
    if pcost.kin_cached:                 # E at x +- eps v, and J^T at x ('normal'): all misses
        e["kin"] = e.get("kin", 0.0) + (3 if normal else 2) * iters * pcost.kin(B)
    calls = {"E": 2 * iters}
    if normal:
        _add(e, pcost.JT(B), iters)
        calls["JT"] = iters
    # per iteration: eps (2 norms, 3 row ops), x +- eps v, (E+ - E-)/(2 eps), then the norm and
    # the normalisation of w (m entries for 'residual', n_x for 'normal'); plus the initial
    # normalisation of v (3 B n) and the final max over the batch
    per = (11 * B * n + 2 * B * m + 4 * B) if normal else (8 * B * n + 5 * B * m + 4 * B)
    e["disc"] = iters * per + 3 * B * n + (B - 1)
    return {"flops": float(sum(e.values())), "calls": calls, "by_category": e,
            "note": "one-off; direction resolved from the problem when 'auto'"}


def step_flops(problem, model, disc, B=1, info="analytic", noise=False, data_noise=False,
               include_data=False, reuse="ideal", skip_unused_dEdt=True, tau=None,
               k_range=None, amortize_setup=None, skip_unused_E=None):
    """Analytic FLOPs of one steady-state step of ``simulate(problem, model, disc)``.

    Parameters
    ----------
    B : trajectories per batch (all share the data and therefore the cached
        factorisation of the matrix problems).
    info : simulate's information regime ('analytic', 'bd1', 'bd2', 'none').
    noise : True if an additive design-formula noise is simulated (one more
        solve_J per vector-field evaluation).
    data_noise : True if simulate perturbs the data (each stage sample is then
        a new realisation and the factorisation cache never hits across them).
    include_data : add the synthetic data generation to ``total``.
    reuse : 'ideal' (a sample at t_k + tau equals the next step's t_k sample),
        'none' (never), or 'exact' (needs ``tau``; mixes the two with the exact
        floating-point coincidence rate over ``k_range``, see
        :func:`float_reuse_fraction`).  Only RK4 and the non-causal
        prediction-correction schemes are affected.
    skip_unused_dEdt : simulate evaluates dE/dt (or the backward differences)
        even for models with ``uses_dEdt = False`` (gnn, newton_flow); True
        (default) charges only what the algorithm needs, False what the code
        does.
    amortize_setup : add the one-off setup (alpha='auto' Lipschitz estimate)
        divided by this number of steps.
    skip_unused_E : likewise for the E(x_k) that simulate evaluates for every
        vector-field call although ``davidenko`` (x_dot = -J^{-1} dE/dt) never
        reads it.  None (default) follows ``skip_unused_dEdt``, so the default
        is the algorithmic view and ``skip_unused_dEdt=False`` the code view.

    Returns a dict with 'total', 'linalg' (algebra + factor + solve + kin),
    'by_category', 'by_prim' (problem-side FLOPs per primitive), 'calls' and
    'rows' per step, 'factorizations', 'data_generations', 'kin_rows',
    'dense_equivalent' (the same calls charged with ``wrappers.dense_flops``),
    'setup' and 'notes'.

    Known limits (docs/spec/costs.md, Sec. 7): the 64-entry data memo of the
    matrix problems occasionally regenerates data (only 'data' is affected);
    the value-keyed factorisation cache and the robot kinematics cache can hit
    in cases the model charges as misses (1-ulp time differences giving equal
    data, a converged Newton correction leaving x unchanged); start-up steps
    are not included.
    """
    if reuse not in ("ideal", "none", "exact"):
        raise ValueError("reuse must be 'ideal', 'none' or 'exact'")
    pcost = problem_cost(problem)
    mcost = model_cost(model, problem, pcost)
    B = int(B)
    d, items, notes = _schedule(disc, problem, B, pcost.n_x, mcost.n_z)
    if skip_unused_E is None:
        skip_unused_E = skip_unused_dEdt
    args = (pcost, mcost, items, B, info, noise)
    if reuse == "exact":
        if tau is None:
            raise ValueError("reuse='exact' needs tau")
        k1, k2 = k_range if k_range is not None else (0, int(round(problem.T / tau)))
        p = float_reuse_fraction(tau, k1 - 1, k2 - 1)
        acc = _steady(*args, "ideal", data_noise, skip_unused_dEdt, skip_unused_E).scaled(p)
        acc.plus(_steady(*args, "none", data_noise, skip_unused_dEdt,
                         skip_unused_E).scaled(1.0 - p))
    else:
        p = None
        acc = _steady(*args, reuse, data_noise, skip_unused_dEdt, skip_unused_E)
    cat = {c: float(acc.cat.get(c, 0.0)) for c in CATEGORIES}
    setup = _setup_cost(pcost, d, B, problem)
    if setup is not None and amortize_setup:
        for k, v in setup["by_category"].items():
            cat[k] = cat.get(k, 0.0) + v / float(amortize_setup)
    total = sum(v for k, v in cat.items() if include_data or k != "data")
    linalg = sum(cat[k] for k in ("algebra", "factor", "solve", "kin"))
    if mcost.n_z:
        notes.append(f"auxiliary state n_z = {mcost.n_z} advanced by the discretizer")
    if not mcost.uses_dEdt and not skip_unused_dEdt and info != "none":
        notes.append("simulate evaluates dE/dt although the model ignores it (charged)")
    if not getattr(mcost, "uses_E", True):
        notes.append("simulate evaluates E(x_k) although the model ignores it ("
                     + ("not charged; skip_unused_E=False charges it)" if skip_unused_E
                        else "charged)"))
    return {"total": float(total), "linalg": float(linalg), "by_category": cat,
            "by_prim": {k: float(acc.prim.get(k, 0.0)) for k in PRIMS + ("jacobian",)},
            "calls": {k: float(acc.calls.get(k, 0.0)) for k in PRIMS},
            "rows": {k: float(acc.rows.get(k, 0.0)) for k in PRIMS},
            "factorizations": float(acc.factorizations),
            "data_generations": {k: float(v) for k, v in acc.gens.items()},
            "kin_rows": float(acc.kin_rows), "dense_equivalent": float(acc.dense),
            "setup": setup, "reuse": reuse, "reuse_fraction": p,
            "include_data": bool(include_data), "B": B, "info": info,
            "problem": pcost.describe(), "model": mcost.name,
            "disc": getattr(d, "name", type(d).__name__),
            "notes": notes + pcost.notes + mcost.notes}


def scaling_exponent(sizes, costs):
    """Least-squares slope of log(cost) against log(size)."""
    x = np.log(np.asarray(sizes, dtype=float))
    y = np.log(np.asarray(costs, dtype=float))
    return float(np.polyfit(x, y, 1)[0])


# ===========================================================================
# 8. empirical cross-check (CountingProblem-based)
# ===========================================================================
def _recording_problem_class():
    from .wrappers import CountingProblem

    class RecordingProblem(CountingProblem):
        """CountingProblem that also records the number of state rows per call."""

        _pending = None          # snapshot callback fired by the next data() call

        def reset_counts(self):
            super().reset_counts()
            self.rows = {k: 0 for k in self._METHODS}

        def data(self, t):
            cb, self._pending = self._pending, None
            if cb is not None:
                cb()
            return super().data(t)

        def E(self, x, d):
            self.rows["E"] += x.shape[0]
            return super().E(x, d)

        def dEdt(self, x, t):
            self.rows["dEdt"] += x.shape[0]
            return super().dEdt(x, t)

        def solve_J(self, x, d, r):
            self.rows["solve_J"] += r.shape[0]
            return super().solve_J(x, d, r)

        def JT(self, x, d, v):
            self.rows["JT"] += v.shape[0]
            return super().JT(x, d, v)

    return RecordingProblem


class _Window:
    """Transparent discretizer wrapper that snapshots the counters at the start
    of simulate's iterations k1 and k2 (armed after step k - 1 returns, fired
    by simulate's own data(t_k) call, so that the window holds whole
    iterations k1 .. k2 - 1)."""

    def __init__(self, inner, k1, k2, snap, rec):
        self.inner, self.k1, self.k2, self.snap, self.rec = inner, k1, k2, snap, rec
        self.name = getattr(inner, "name", "disc")
        self.causal = getattr(inner, "causal", True)
        self.implicit = getattr(inner, "implicit", False)
        self.expected_order = getattr(inner, "expected_order", None)
        self.snaps = {}

    def __getattr__(self, item):
        if item.startswith("__") or item == "inner":
            raise AttributeError(item)
        inner = self.__dict__.get("inner")
        if inner is None:
            raise AttributeError(item)
        return getattr(inner, item)

    def reset(self):
        self.inner.reset()

    def bind_env(self, **kw):
        if callable(getattr(self.inner, "bind_env", None)):
            self.inner.bind_env(**kw)

    def step(self, k, t, tau, x, z, f):
        out = self.inner.step(k, t, tau, x, z, f)
        if k + 1 in (self.k1, self.k2):
            self.rec._pending = lambda kk=k + 1: self.snaps.__setitem__(kk, self.snap())
        return out

    def describe(self):
        return self.inner.describe()


def empirical_step_cost(problem, model, disc, B=1, tau=1e-2, steps=30, warmup=12, gamma=10.0,
                        info="analytic", seed=0, noise=None, data_noise=0.0,
                        include_data=False):
    """Observe one simulate run through a call-recording ``CountingProblem``.

    Runs ``warmup + steps`` steps (``online_stats=False``, no solution error,
    residual recorded only at t = 0) and counts, over simulate's iterations
    k = warmup .. warmup + steps - 1 (each including simulate's own data(t_k)
    call): the Problem calls
    and their state rows (``CountingProblem``), the factorisations actually
    computed (the ``_FactorCache`` of the matrix problems), the data arrays
    actually generated (memo misses, or calls for unmemoised problems) and
    the kinematics rows (``SerialArm.fk_jac``).  The problem-side FLOPs are
    then the analytic per-call costs applied to these OBSERVED events; they
    are compared with :func:`step_flops` (reuse='exact', the code's own view
    with ``skip_unused_dEdt=False``) under 'analytic'.  The model's and the
    discretizer's own numpy work is not observable and is not part of
    'problem_flops'.  The instrumented problem is restored afterwards.
    """
    from .problems.matrix import _FactorCache
    from .problems.robot import ManipulatorTracking
    from .simulate import simulate
    pcost = problem_cost(problem)
    base = _innermost(problem)
    Rec = _recording_problem_class()
    P = Rec(problem)
    stats = {"factor": 0, "gens": defaultdict(int), "kin_rows": 0}
    restore = []
    for attr in ("_lu", "_fac"):
        cache = base.__dict__.get(attr)
        if isinstance(cache, _FactorCache):
            orig = cache._factor

            def counted(*mats, _orig=orig):
                stats["factor"] += 1
                return _orig(*mats)

            cache._factor = counted
            restore.append((cache, "_factor", orig))
            cache._key = cache._val = None
    if pcost.memo and hasattr(type(base), "_memo"):
        orig_memo = base._memo

        def memo(tag, t, fn, _orig=orig_memo):
            cache = base.__dict__.setdefault("_memo_cache", {})
            if cache.get((tag, float(t))) is None:
                stats["gens"][tag] += 1
            return _orig(tag, t, fn)

        base._memo = memo
        restore.append((base, "_memo", None))
        base.__dict__.pop("_memo_cache", None)
    if isinstance(base, ManipulatorTracking):
        orig_fk = base.arm.fk_jac

        def fk_jac(theta, _orig=orig_fk):
            stats["kin_rows"] += np.atleast_2d(theta).shape[0]
            return _orig(theta)

        base.arm.fk_jac = fk_jac
        restore.append((base.arm, "fk_jac", None))
        base._ck = base._cv = None

    def snap():
        return {"calls": dict(P.calls), "rows": dict(P.rows), "factor": stats["factor"],
                "gens": dict(stats["gens"]), "kin_rows": stats["kin_rows"]}

    k1, k2 = max(int(warmup), 1), max(int(warmup), 1) + int(steps)
    win = _Window(disc, k1, k2, snap, P)
    try:
        simulate(P, model, win, tau=tau, T=k2 * tau, B=B, gamma=gamma, info=info,
                 seed=seed, noise=noise, data_noise=data_noise, online_stats=False,
                 record_solution_error=False, record_every=k2 + 1)
    finally:
        for obj, attr, orig in restore:
            if orig is None:
                obj.__dict__.pop(attr, None)
            else:
                setattr(obj, attr, orig)
    if k1 not in win.snaps or k2 not in win.snaps:
        raise RuntimeError("empirical_step_cost: the run stopped before the counting window "
                           "ended (every trajectory diverged); use a smaller gamma * tau")
    a, b = win.snaps[k1], win.snaps[k2]
    n = float(steps)
    calls = {k: (b["calls"][k] - a["calls"][k]) / n for k in PRIMS}
    rows = {k: (b["rows"][k] - a["rows"][k]) / n for k in PRIMS if k != "data"}
    gens = {k: (b["gens"].get(k, 0) - a["gens"].get(k, 0)) / n
            for k in set(a["gens"]) | set(b["gens"])}
    factorizations = (b["factor"] - a["factor"]) / n
    kin_rows = (b["kin_rows"] - a["kin_rows"]) / n
    # problem-side FLOPs from the observed events
    cat = defaultdict(float)
    for prim in ("E", "dEdt", "solve_J", "JT"):
        c = calls[prim]
        if c:
            r = rows[prim] / c                         # rows per call (uniform per primitive)
            for k, v in pcost.call(prim, r).items():
                cat[k] += c * v
    if pcost.factor_mode == "cached":
        cat["factor"] += factorizations * pcost.factor()
    if pcost.memo:
        gens_out = gens
    else:
        gens_out = defaultdict(float)
        for prim in ("data", "dEdt"):
            for tag in pcost.tags.get(prim, ()):
                gens_out[tag] += calls[prim]
        gens_out = dict(gens_out)
    cat["data"] += sum(pcost.gen(tag) * v for tag, v in gens_out.items())
    cat["kin"] += pcost.kin(1) * kin_rows if pcost.kin_cached else 0.0
    problem_flops = sum(v for k, v in cat.items() if include_data or k != "data")
    try:
        analytic = step_flops(problem, model, disc, B=B, info=info, noise=noise is not None,
                              data_noise=bool(data_noise), include_data=include_data,
                              reuse="exact", skip_unused_dEdt=False, tau=tau, k_range=(k1, k2))
    except NotImplementedError:          # e.g. a discretization without a cost schedule
        analytic, an_prob = None, None
    else:
        # problem.jacobian (CDTZNN on dense-Jacobian problems) is not a CountingProblem
        # method: its analytic cost is excluded from the comparison
        an_prob = sum(analytic["by_category"][k] for k in PROBLEM_CATEGORIES
                      if include_data or k != "data") - analytic["by_prim"]["jacobian"]
    return {"calls": calls, "rows": rows, "factorizations": factorizations,
            "data_generations": gens_out, "kin_rows": kin_rows,
            "by_category": {k: float(cat.get(k, 0.0)) for k in PROBLEM_CATEGORIES},
            "problem_flops": float(problem_flops), "analytic": analytic,
            "analytic_problem_flops": None if an_prob is None else float(an_prob),
            "k_range": (k1, k2), "steps": int(steps)}
