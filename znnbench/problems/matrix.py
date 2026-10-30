"""Time-varying matrix problems for ZNN / GNN benchmarking.

Registered keys (see ``docs/spec/problems_matrix.md`` for citations, LaTeX
equations, defaults and every ASSUMPTION):

``matrix_inversion``        E = A(t) X - I            Zhang & Ge, IEEE TNN 16(6), 2005, eqs. (1)-(4)
``matrix_inversion_right``  E = X A(t) - I            dual form (e.g. Zivkovic & Stanimirovic, FUMI 2017, eq. (1.2))
``pseudoinverse``           E = A^T A X - A^T  (full column rank, left MP inverse)
                            E = X A A^T - A^T  (full row rank, right MP inverse)
                                                      Zhang, Yang, Tan & Cai, Computing 92, 2011
                                                      (as restated in Chai et al., IJCIS 13, 2020, eqs. (3), (10))
``sylvester``               E = A(t) X - X B(t) + C(t) Zhang, Jiang & Wang, IEEE TNN 13(5), 2002
``lyapunov``                E = A^T(t) X + X A(t) + Q(t)  Hu et al., Neurocomputing 418, 2020, eqs. (1)-(2)
``complex_inversion``       complex E = A(t) X - I     Zhang, Li & Li, AMC 217, 2011, in the real
                                                      embedding of Ding et al., Front. Neurorobot. 2017

Conventions (DESIGN.md)
-----------------------
* The unknown matrix is flattened row-major: ``X = x.reshape(B, r, c)``; the
  error matrix is flattened the same way.  For ``complex_inversion`` the real
  state is ``X~ = [Re X; Im X]`` (2n x n), i.e. ``x[:, :n*n] = vec(Re X)`` and
  ``x[:, n*n:] = vec(Im X)``; likewise ``E~ = [Re E; Im E]``.
* All error functions are affine in X, so the Jacobian ``J = dE/dx`` does not
  depend on x and is never formed: ``solve_J`` and ``JT`` act on the matrix-
  valued right-hand side with one factorisation per time step (LU, Cholesky
  or a real Schur pair, cached by the *values* of the data matrices so that a
  second solve at the same sampling instant -- e.g. the noise injection in
  ``simulate`` -- reuses it).  Only the batch dimension is ever looped over
  (Sylvester/Lyapunov triangular solves); never matrix entries.
* Random instances use ``_tv.TVMatrix`` (exact derivatives, singular values in
  [1, kappa] x (1 +/- 0.2)), a frequency ``freq`` (rad/s), a ``seed`` and a
  horizon ``T``.  Data arrays returned by ``data(t)`` are memoised per t and
  marked read-only.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import (LinAlgError, cho_factor, cho_solve, get_lapack_funcs, lu_factor,
                          lu_solve, schur)

from ..core import Problem
from ._tv import TVMatrix, TVVector

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _mat(x, r, c):
    """(B, r*c) -> (B, r, c) (row-major, DESIGN.md flattening rule)."""
    return np.asarray(x).reshape(x.shape[0], r, c)


def _vec(X):
    """(B, r, c) -> (B, r*c)."""
    return X.reshape(X.shape[0], -1)


def _left_solve(solve2d, R):
    """Solve M Y_b = R_b for all batch members with a single 2-D solve.

    ``solve2d(b)`` solves M y = b for an (r, q) right-hand side; R is (B, r, c).
    The batch is stacked column-wise: [R_1 ... R_B] (r x Bc).
    """
    B, r, c = R.shape
    Y = solve2d(np.ascontiguousarray(R.transpose(1, 0, 2)).reshape(r, B * c))
    return Y.reshape(r, B, c).transpose(1, 0, 2)


def _right_solve(solve2d_T, R):
    """Solve Y_b M = R_b  <=>  M^T Y_b^T = R_b^T; ``solve2d_T`` solves with M^T."""
    return _left_solve(solve2d_T, R.transpose(0, 2, 1)).transpose(0, 2, 1)


class _FactorCache:
    """Keeps one factorisation, keyed by the *values* of the input matrices.

    The O(size) equality test is negligible next to the O(n^3) factorisation
    and, unlike an identity test, can never return a stale factor if a caller
    mutates or rebuilds the data arrays.
    """

    def __init__(self, factor):
        self._factor = factor
        self._key = None
        self._val = None

    def __call__(self, *mats):
        key = self._key
        if key is not None and all(a.shape == b.shape and np.array_equal(a, b)
                                   for a, b in zip(mats, key)):
            return self._val
        self._val = self._factor(*mats)
        self._key = tuple(np.array(m, copy=True) for m in mats)
        return self._val

    def __getstate__(self):          # a cache is not state (and LAPACK handles do not pickle)
        return {"_factor": self._factor, "_key": None, "_val": None}

    def __setstate__(self, st):
        self.__dict__.update(st)


def _lu(A):
    return lu_factor(A, check_finite=False)


def _spd_factor(G):
    """Cholesky of an SPD Gram matrix, LU fallback if it is numerically indefinite."""
    try:
        return ("cho", cho_factor(G, check_finite=False))
    except LinAlgError:
        return ("lu", lu_factor(G, check_finite=False))


def _spd_solve(fac, b):
    kind, f = fac
    if kind == "cho":
        return cho_solve(f, b, check_finite=False)
    return lu_solve(f, b, check_finite=False)


def _gram_left(A):          # A^T A  (full column rank)
    return _spd_factor(A.T @ A)


def _gram_right(A):         # A A^T  (full row rank)
    return _spd_factor(A @ A.T)


def _schur_pair(A1, B1):
    """Real Schur forms for A1 Y + Y B1 = F, exactly as scipy.linalg.solve_sylvester."""
    r, u = schur(A1, output="real", check_finite=False)
    s, v = schur(B1.T, output="real", check_finite=False)
    trsyl, = get_lapack_funcs(("trsyl",), (r, s))
    return r, u, s, v, trsyl


def _sylvester_solve(fac, R):
    """Solve A1 Y_b + Y_b B1 = R_b for every batch member b.

    Same algorithm as ``scipy.linalg.solve_sylvester`` (Bartels-Stewart via
    LAPACK *TRSYL) with the two Schur decompositions computed once per data
    sample and shared by the whole batch.

    LAPACK *TRSYL returns y with  op(R) y + y op(S) = scale * F  (scale <= 1
    is chosen to avoid overflow), so the solution for F is y / scale.
    ``scipy.linalg.solve_sylvester`` (1.17) returns scale * y instead; the two
    agree only when scale == 1 (always the case for well-scaled data).  The
    LAPACK contract is followed here (verified: test_trsyl_scale_contract).
    """
    r, u, s, v, trsyl = fac
    F = u.T @ R @ v
    Y = np.empty_like(F)
    for b in range(F.shape[0]):             # loop over the batch only
        y, scale, info = trsyl(r, s, F[b], tranb="C")
        if info < 0:
            raise LinAlgError(f"trsyl: illegal value in argument {-info}")
        Y[b] = y if scale == 1.0 else y / scale
    return u @ Y @ v.T


def _random_unitary(n, rng):
    z = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) / np.sqrt(2.0)
    q, r = np.linalg.qr(z)
    d = np.diag(r)
    return q * (d / np.abs(d))


# ---------------------------------------------------------------------------
# data generators that are not in _tv.py
# ---------------------------------------------------------------------------
class _SymSkewMatrix:
    """M(t) = sign * ( S(t) + skew * (G(t) - G(t)^T) / 2 ).

    S(t) = TVMatrix(spd=True, kappa): symmetric positive definite with
    eigenvalues in [0.8, 1.2 kappa]; G(t) = TVMatrix(kappa=1).  The symmetric
    part of M is sign*S, so for sign=+1 the field of values of M lies in
    Re z >= lambda_min(S) >= 0.8 and for sign=-1 M is Hurwitz with
    Re lambda(M) <= -0.8.  Exact derivative by linearity.
    """

    def __init__(self, n, rng, kappa, freq, skew=1.0, sign=1.0):
        self.S = TVMatrix(n, rng, kappa=kappa, freq=freq, spd=True)
        self.skew = float(skew)
        self.G = TVMatrix(n, rng, kappa=1.0, freq=freq) if self.skew else None
        self.sign = float(sign)

    def _eval(self, t, deriv):
        M = self.S.dot(t) if deriv else self.S(t)
        if self.G is not None:
            G = self.G.dot(t) if deriv else self.G(t)
            M = M + 0.5 * self.skew * (G - G.T)
        return self.sign * M

    def __call__(self, t):
        return self._eval(t, False)

    def dot(self, t):
        return self._eval(t, True)


class _TVComplexMatrix:
    """Smooth complex A(t) = U(t) S(t) V(t)^H with controlled condition number.

    U(t) = P_U diag(exp(i w_U t)) P_U^H (P_U Haar unitary, w ~ freq*U[0.5,1.5]),
    likewise V(t); S(t) = diag(s0 (1 + mod sin(w_s t + phi))) with s0
    geometrically spaced in [1, kappa] -- the complex analogue of
    ``_tv.TVMatrix``.  Exact derivative by the product rule.
    """

    def __init__(self, n, rng, kappa=10.0, freq=1.0, mod=0.2):
        self.n = n
        self.PU, self.wU = _random_unitary(n, rng), freq * rng.uniform(0.5, 1.5, size=n)
        self.PV, self.wV = _random_unitary(n, rng), freq * rng.uniform(0.5, 1.5, size=n)
        self.s0 = np.geomspace(kappa, 1.0, n) if n > 1 else np.array([1.0])
        self.ws = freq * rng.uniform(0.5, 1.5, size=n)
        self.ph = rng.uniform(0, 2 * np.pi, size=n)
        self.mod = mod

    @staticmethod
    def _unitary(P, w, t, deriv=False):
        e = np.exp(1j * w * t)
        if deriv:
            e = 1j * w * e
        return (P * e) @ P.conj().T

    def _s(self, t, deriv=False):
        arg = self.ws * t + self.ph
        if deriv:
            return self.s0 * self.mod * self.ws * np.cos(arg)
        return self.s0 * (1.0 + self.mod * np.sin(arg))

    def __call__(self, t):
        U, V = self._unitary(self.PU, self.wU, t), self._unitary(self.PV, self.wV, t)
        return (U * self._s(t)) @ V.conj().T

    def dot(self, t):
        U, Ud = self._unitary(self.PU, self.wU, t), self._unitary(self.PU, self.wU, t, True)
        V, Vd = self._unitary(self.PV, self.wV, t), self._unitary(self.PV, self.wV, t, True)
        s, sd = self._s(t), self._s(t, True)
        return ((Ud * s) @ V.conj().T + (U * sd) @ V.conj().T + (U * s) @ Vd.conj().T)


class _ZhangGe2005A:
    """A(t) = [[sin t, cos t], [-cos t, sin t]] -- the example of Zhang, Jiang &
    Wang (2002) reused in Zhang & Ge (2005), Sec. V; A^{-1}(t) = A^T(t)."""

    def __call__(self, t):
        s, c = np.sin(t), np.cos(t)
        return np.array([[s, c], [-c, s]])

    def dot(self, t):
        s, c = np.sin(t), np.cos(t)
        return np.array([[c, -s], [s, c]])


class _Hu2020M:
    """M(t) of Hu et al. (2020), Sec. 5: [[-1 - cos3t/2, sin3t/2], [sin3t/2, -1 + cos3t/2]]."""

    def __call__(self, t):
        c, s = np.cos(3 * t), np.sin(3 * t)
        return np.array([[-1.0 - 0.5 * c, 0.5 * s], [0.5 * s, -1.0 + 0.5 * c]])

    def dot(self, t):
        c, s = np.cos(3 * t), np.sin(3 * t)
        return np.array([[1.5 * s, 1.5 * c], [1.5 * c, -1.5 * s]])


class _Hu2020Q:
    """Q(t) of Hu et al. (2020), Sec. 5: [[sin3t, cos3t], [-cos3t, sin3t]] (NOT
    symmetric, although Sec. 2 of the paper calls Q symmetric positive definite)."""

    def __call__(self, t):
        c, s = np.cos(3 * t), np.sin(3 * t)
        return np.array([[s, c], [-c, s]])

    def dot(self, t):
        c, s = np.cos(3 * t), np.sin(3 * t)
        return np.array([[3 * c, -3 * s], [3 * s, 3 * c]])


def hu2020_solution(t):
    """Closed-form "theoretical solution" exactly as printed in Hu et al. (2020),
    Sec. 5.  NOTE (verified numerically in the tests): it solves
    M^T X + X M + Q^T = 0, i.e. it is the TRANSPOSE of the solution for the
    printed Q(t) (A_12 and A_21 swapped).  The problem uses the exact solution
    of the printed equation; this function is kept only for that check."""
    c, s = np.cos(3 * t), np.sin(3 * t)
    return np.array([[-s * (c - 2) / 3, -(2 * c - 1) * (c + 2) / 6],
                     [-(2 * c + 1) * (c - 2) / 6, s * (c + 2) / 3]])


# ---------------------------------------------------------------------------
# base class
# ---------------------------------------------------------------------------
class _MatrixProblem(Problem):
    """Shared plumbing: memoised data, random initial states, describe()."""

    x0_range = 2.0          # X(0) entries ~ U[-2, 2] as in Zhang & Ge (2005), Sec. V-A
                            # (Lyapunov example='hu_2020': E(0) entries, Hu et al. Sec. 5)

    def _memo(self, tag, t, fn):
        cache = self.__dict__.setdefault("_memo_cache", {})
        key = (tag, float(t))
        v = cache.get(key)
        if v is None:
            if len(cache) >= 64:
                cache.clear()
            v = np.array(fn(t))
            v.setflags(write=False)
            cache[key] = v
        return v

    def x0(self, rng, B):
        return rng.uniform(-self.x0_range, self.x0_range, size=(B, self.n_x))

    def describe(self):
        return {**super().describe(), **self.cfg}


# ---------------------------------------------------------------------------
# matrix inversion
# ---------------------------------------------------------------------------
class MatrixInversion(_MatrixProblem):
    """Time-varying matrix inversion, Zhang & Ge (2005) eqs. (1)-(4).

    E(X, t) = A(t) X - I   (n x n);   J dX = A dX;   dE/dt = A_dot X;
    J^T V = A^T V.  The ZNN design formula dE/dt = -Gamma F(E) gives the
    implicit model A X_dot = -A_dot X - Gamma F(A X - I)  [eq. (4)].

    ``example='zhang_ge_2005'`` reproduces the paper's A(t) (n = 2).
    Capability ``'matrix_inverse'``: X approximates J^{-1} in the sense
    J dx = r  <=>  dX = A^{-1} R ~ X R.
    """

    family = "matrix-inversion"
    side = "left"

    def __init__(self, n: int = 4, kappa: float = 10.0, freq: float = 1.0, seed: int = 0,
                 T: float = 10.0, example=None):
        if example is None:
            rng = np.random.default_rng(seed)
            self.A = TVMatrix(n, rng, kappa=kappa, freq=freq)
        elif example == "zhang_ge_2005":
            n, kappa, freq = 2, 1.0, 1.0
            self.A = _ZhangGe2005A()
        else:
            raise ValueError(f"unknown example {example!r}")
        self.n = n
        self.n_x = self.m = n * n
        self.T = T
        self._I = np.eye(n)
        self._lu = _FactorCache(_lu)
        tag = "matinv" if self.side == "left" else "matinvR"
        self.name = f"{tag}_{example}" if example else f"{tag}_n{n}_k{kappa:g}"
        self.cfg = dict(n=n, kappa=kappa, freq=freq, seed=seed, example=example, side=self.side)

    def data(self, t):
        return {"t": np.float64(t), "A": self._memo("A", t, self.A)}

    def _Adot(self, t):
        return self._memo("dA", t, self.A.dot)

    def E(self, x, d):
        X = _mat(x, self.n, self.n)
        return _vec(d["A"] @ X - self._I)

    def dEdt(self, x, t):
        return _vec(self._Adot(t) @ _mat(x, self.n, self.n))

    def solve_J(self, x, d, r):
        lu = self._lu(d["A"])
        dX = _left_solve(lambda b: lu_solve(lu, b, check_finite=False), _mat(r, self.n, self.n))
        return _vec(dX)

    def JT(self, x, d, v):
        return _vec(d["A"].T @ _mat(v, self.n, self.n))

    def x_star(self, t):
        return np.linalg.solve(self.A(t), self._I).reshape(-1)

    def capabilities(self):
        return {"matrix_inverse"}


class MatrixInversionRight(MatrixInversion):
    """Dual (right) form E(X, t) = X A(t) - I.

    J dX = dX A;  dE/dt = X A_dot;  J^T V = V A^T.  Same solution A^{-1}(t).
    Capability ``'matrix_inverse_right'``: J dx = r <=> dX = R A^{-1} ~ R X.
    """

    side = "right"

    def E(self, x, d):
        return _vec(_mat(x, self.n, self.n) @ d["A"] - self._I)

    def dEdt(self, x, t):
        return _vec(_mat(x, self.n, self.n) @ self._Adot(t))

    def solve_J(self, x, d, r):
        lu = self._lu(d["A"])
        dX = _right_solve(lambda b: lu_solve(lu, b, trans=1, check_finite=False),
                          _mat(r, self.n, self.n))
        return _vec(dX)

    def JT(self, x, d, v):
        return _vec(_mat(v, self.n, self.n) @ d["A"].T)

    def capabilities(self):
        return {"matrix_inverse_right"}


# ---------------------------------------------------------------------------
# Moore-Penrose pseudoinverse of a full-rank rectangular matrix
# ---------------------------------------------------------------------------
class Pseudoinverse(_MatrixProblem):
    """Time-varying full-rank Moore-Penrose inverse (Zhang, Yang, Tan & Cai 2011).

    A(t) is rows x cols with full rank; the unknown X (cols x rows) -> A^+(t).

    * side='left'  (rows >= cols, full column rank, A^+ = (A^T A)^{-1} A^T):
        E = A^T A X - A^T,           J dX = (A^T A) dX,
        dE/dt = (A_dot^T A + A^T A_dot) X - A_dot^T,   J^T V = (A^T A) V.
    * side='right' (rows < cols, full row rank, A^+ = A^T (A A^T)^{-1}):
        E = X A A^T - A^T,           J dX = dX (A A^T),
        dE/dt = X (A_dot A^T + A A_dot^T) - A_dot^T,   J^T V = V (A A^T).

    By default ``n`` is the rank (the smaller dimension) and the other
    dimension is n + max(1, n // 2)  [ASSUMPTION: benchmark shape choice].
    Pass ``rows``/``cols`` to choose the shape; the side then follows from it.
    The Gram matrix is factorised once per step (Cholesky).
    """

    family = "pseudoinverse"

    def __init__(self, n: int = 4, kappa: float = 10.0, freq: float = 1.0, seed: int = 0,
                 T: float = 10.0, side=None, rows=None, cols=None):
        if (rows is None) != (cols is None):
            raise ValueError("give both rows and cols, or neither")
        if rows is None:
            side = side or "left"
            if side not in ("left", "right"):
                raise ValueError("side must be 'left' or 'right'")
            extra = max(1, n // 2)
            rows, cols = (n + extra, n) if side == "left" else (n, n + extra)
        else:
            shape_side = "left" if rows >= cols else "right"
            if side is not None and side != shape_side:
                raise ValueError(f"shape {rows}x{cols} implies side={shape_side!r}")
            side = shape_side
        rng = np.random.default_rng(seed)
        self.A = TVMatrix(rows, rng, kappa=kappa, freq=freq, n_cols=cols)
        self.rows, self.cols, self.side = rows, cols, side
        self.n_x = self.m = rows * cols
        self.T = T
        self._fac = _FactorCache(_gram_left if side == "left" else _gram_right)
        self.name = f"pinv{side[0].upper()}_{rows}x{cols}_k{kappa:g}"
        self.cfg = dict(n=min(rows, cols), rows=rows, cols=cols, side=side, kappa=kappa,
                        freq=freq, seed=seed)

    def data(self, t):
        return {"t": np.float64(t), "A": self._memo("A", t, self.A)}

    def _X(self, x):
        return _mat(x, self.cols, self.rows)

    def E(self, x, d):
        A, X = d["A"], self._X(x)
        if self.side == "left":
            return _vec((A.T @ A) @ X - A.T)
        return _vec(X @ (A @ A.T) - A.T)

    def dEdt(self, x, t):
        A, Ad, X = self._memo("A", t, self.A), self._memo("dA", t, self.A.dot), self._X(x)
        if self.side == "left":
            return _vec((Ad.T @ A + A.T @ Ad) @ X - Ad.T)
        return _vec(X @ (Ad @ A.T + A @ Ad.T) - Ad.T)

    def solve_J(self, x, d, r):
        fac = self._fac(d["A"])
        R = self._X(r)
        solve = lambda b: _spd_solve(fac, b)      # noqa: E731  (Gram matrix is symmetric)
        dX = _left_solve(solve, R) if self.side == "left" else _right_solve(solve, R)
        return _vec(dX)

    def JT(self, x, d, v):
        A, V = d["A"], self._X(v)
        if self.side == "left":
            return _vec((A.T @ A) @ V)
        return _vec(V @ (A @ A.T))

    def x_star(self, t):
        A = self.A(t)
        if self.side == "left":
            return np.linalg.solve(A.T @ A, A.T).reshape(-1)
        return np.linalg.solve(A @ A.T, A).T.reshape(-1)


# ---------------------------------------------------------------------------
# Sylvester equation
# ---------------------------------------------------------------------------
class Sylvester(_MatrixProblem):
    """Time-varying Sylvester equation A(t) X - X B(t) + C(t) = 0
    (Zhang, Jiang & Wang 2002; form as restated by Zhao et al. 2023, eqs. (1)-(2)).

    E = A X - X B + C (n x k);  J dX = A dX - dX B;
    dE/dt = A_dot X - X B_dot + C_dot;  J^T V = A^T V - V B^T.
    (The textbook form A X + X B' = C' is the same problem with B' = -B, C' = -C.)

    Random instance [ASSUMPTION, benchmark generator]: A = S_A + W_A,
    B = -(S_B + W_B) with S SPD (eigenvalues in [0.8, 1.2 kappa]) and W smooth
    skew-symmetric of size ``skew``; C(t) entries a sin(w t + phi).  Then the
    symmetric part of the Sylvester operator is >= 1.6 I, so the solution is
    unique and cond(J) <= 1.5 (kappa + skew).
    ``solve_J`` = Bartels-Stewart (``scipy.linalg.solve_sylvester``'s
    algorithm) with the Schur pair of (A, -B) computed once per step.
    """

    family = "sylvester"

    def __init__(self, n: int = 4, k=None, kappa: float = 10.0, freq: float = 1.0, seed: int = 0,
                 T: float = 10.0, skew: float = 1.0):
        k = n if k is None else k
        rng = np.random.default_rng(seed)
        self.A = _SymSkewMatrix(n, rng, kappa, freq, skew, sign=+1.0)
        self.Bm = _SymSkewMatrix(k, rng, kappa, freq, skew, sign=-1.0)
        self.C = TVVector(n * k, rng, freq=freq)
        self.n, self.k = n, k
        self.n_x = self.m = n * k
        self.T = T
        self._fac = _FactorCache(_sylv_pair)
        self.name = f"sylv_n{n}x{k}_k{kappa:g}"
        self.cfg = dict(n=n, k=k, kappa=kappa, freq=freq, seed=seed, skew=skew)

    def data(self, t):
        return {"t": np.float64(t), "A": self._memo("A", t, self.A),
                "B": self._memo("B", t, self.Bm),
                "C": self._memo("C", t, lambda s: self.C(s).reshape(self.n, self.k))}

    def E(self, x, d):
        X = _mat(x, self.n, self.k)
        return _vec(d["A"] @ X - X @ d["B"] + d["C"])

    def dEdt(self, x, t):
        X = _mat(x, self.n, self.k)
        Ad = self._memo("dA", t, self.A.dot)
        Bd = self._memo("dB", t, self.Bm.dot)
        Cd = self._memo("dC", t, lambda s: self.C.dot(s).reshape(self.n, self.k))
        return _vec(Ad @ X - X @ Bd + Cd)

    def solve_J(self, x, d, r):
        fac = self._fac(d["A"], d["B"])
        return _vec(_sylvester_solve(fac, _mat(r, self.n, self.k)))

    def JT(self, x, d, v):
        V = _mat(v, self.n, self.k)
        return _vec(d["A"].T @ V - V @ d["B"].T)

    def x_star(self, t):
        # Bartels-Stewart on A X - X B = -C, sharing the per-step Schur pair with
        # solve_J (tests cross-check against scipy.linalg.solve_sylvester)
        d = self.data(t)
        return _sylvester_solve(self._fac(d["A"], d["B"]), -d["C"][None])[0].reshape(-1)


def _sylv_pair(A, B):
    return _schur_pair(A, -B)


# ---------------------------------------------------------------------------
# Lyapunov equation
# ---------------------------------------------------------------------------
class Lyapunov(_MatrixProblem):
    """Time-varying Lyapunov equation A^T(t) X + X A(t) + Q(t) = 0
    (Hu, Li, Li, Li & Xiao, Neurocomputing 418, 2020, eqs. (1)-(2); there A is
    called M).

    E = A^T X + X A + Q (n x n);  J dX = A^T dX + dX A;
    dE/dt = A_dot^T X + X A_dot + Q_dot;  J^T V = A V + V A^T.
    X is treated as a full n x n unknown (n^2 states), as in the ZNN
    literature (vectorised model (5) of Hu et al.).

    Random instance [ASSUMPTION, benchmark generator]: A = -(S + W) Hurwitz
    (S SPD with eigenvalues in [0.8, 1.2 kappa], W skew of size ``skew``),
    Q(t) = TVMatrix(spd=True, kappa=q_kappa) symmetric positive definite, so
    X*(t) is symmetric positive definite.  ``example='hu_2020'`` reproduces the
    paper's 2 x 2 example (whose Q(t) is not symmetric, see spec), including its
    initialisation of E(0) (not X(0)) with entries in [-2, 2] (see ``x0``).
    """

    family = "lyapunov"

    def __init__(self, n: int = 4, kappa: float = 10.0, freq: float = 1.0, seed: int = 0,
                 T: float = 10.0, skew: float = 1.0, q_kappa: float = 2.0, example=None):
        if example is None:
            rng = np.random.default_rng(seed)
            self.A = _SymSkewMatrix(n, rng, kappa, freq, skew, sign=-1.0)
            self.Q = TVMatrix(n, rng, kappa=q_kappa, freq=freq, spd=True)
        elif example == "hu_2020":
            n, kappa, freq, skew, q_kappa = 2, None, 3.0, None, None
            self.A, self.Q = _Hu2020M(), _Hu2020Q()
        else:
            raise ValueError(f"unknown example {example!r}")
        self.n = n
        self.n_x = self.m = n * n
        self.T = T
        self._fac = _FactorCache(_lyap_pair)
        self.name = f"lyap_{example}" if example else f"lyap_n{n}_k{kappa:g}"
        self.cfg = dict(n=n, kappa=kappa, freq=freq, seed=seed, skew=skew, q_kappa=q_kappa,
                        example=example)

    def data(self, t):
        return {"t": np.float64(t), "A": self._memo("A", t, self.A),
                "Q": self._memo("Q", t, self.Q)}

    def E(self, x, d):
        X = _mat(x, self.n, self.n)
        A = d["A"]
        return _vec(A.T @ X + X @ A + d["Q"])

    def dEdt(self, x, t):
        X = _mat(x, self.n, self.n)
        Ad = self._memo("dA", t, self.A.dot)
        Qd = self._memo("dQ", t, self.Q.dot)
        return _vec(Ad.T @ X + X @ Ad + Qd)

    def solve_J(self, x, d, r):
        fac = self._fac(d["A"])
        return _vec(_sylvester_solve(fac, _mat(r, self.n, self.n)))

    def JT(self, x, d, v):
        V = _mat(v, self.n, self.n)
        A = d["A"]
        return _vec(A @ V + V @ A.T)

    def x_star(self, t):
        # Bartels-Stewart on A^T X + X A = -Q, sharing the per-step Schur form
        # with solve_J (tests cross-check against scipy and a dense Kronecker solve)
        d = self.data(t)
        return _sylvester_solve(self._fac(d["A"]), -d["Q"][None])[0].reshape(-1)

    def x0(self, rng, B):
        """Random instances: X(0) entries ~ U[-2, 2] (as for every matrix problem).

        ``example='hu_2020'``: Hu et al. (2020), Sec. 5, randomise the initial
        *error* instead ("every element e_ij(0) in E(0) is random generated and
        is bounded by |e_ij(0)| in [0, 2]"), so E(X(0), 0) has entries
        ~ U[-2, 2] [ASSUMPTION: uniform distribution; the paper gives only the
        bound] and X(0) = J^{-1}(E(0) - Q(0)).
        """
        if self.cfg.get("example") != "hu_2020":
            return super().x0(rng, B)
        d = self.data(0.0)
        E0 = rng.uniform(-self.x0_range, self.x0_range, size=(B, self.m))
        return self.solve_J(None, d, E0 - d["Q"].reshape(1, -1))


def _lyap_pair(A):
    """Schur data for A^T Y + Y A = F.  In scipy's algorithm for (A1, B1) =
    (A^T, A) both Schur forms are schur(A^T), so one decomposition suffices."""
    r, u = schur(A.T, output="real", check_finite=False)
    trsyl, = get_lapack_funcs(("trsyl",), (r, r))
    return r, u, r, u, trsyl


# ---------------------------------------------------------------------------
# complex matrix inversion in a real embedding
# ---------------------------------------------------------------------------
class ComplexInversion(_MatrixProblem):
    """Time-varying complex matrix inversion A(t) X(t) = I, A in C^{n x n}
    (complex ZNN of Zhang, Li & Li, AMC 217, 2011), represented in the real
    domain with the embedding of Ding, Xiao, Liao, Lu & Peng (2017):

        [A_r  -A_i] [X_r]   [I]              x = [vec(X_r), vec(X_i)]
        [A_i   A_r] [X_i] - [0] = E~,        E~ = [vec(Re E), vec(Im E)],

    i.e. X~ = x.reshape(B, 2n, n) = [X_r; X_i] and E~ = [Re(AX - I); Im(AX - I)].
    J dX~ = embed(A dX);  dE~/dt = embed(A_dot X);  J^T V~ = embed(A^H V).
    With linear or any element-wise activation applied to E~, the real ZNN
    equals the complex ZNN whose activation acts separately on real and
    imaginary parts ("type-I" complex activation, cf. Chai et al. 2020, eq. (5)).

    Data: ``'Ar'``, ``'Ai'`` (real and imaginary parts, so that measurement
    noise is added to both).  Capability ``'complex_embedding'``.
    """

    family = "complex-matrix-inversion"

    def __init__(self, n: int = 4, kappa: float = 10.0, freq: float = 1.0, seed: int = 0,
                 T: float = 10.0, mod: float = 0.2):
        rng = np.random.default_rng(seed)
        self.A = _TVComplexMatrix(n, rng, kappa=kappa, freq=freq, mod=mod)
        self.n = n
        self.n_x = self.m = 2 * n * n
        self.T = T
        self._I = np.eye(n)
        self._lu = _FactorCache(_complex_lu)
        self.name = f"cmatinv_n{n}_k{kappa:g}"
        self.cfg = dict(n=n, kappa=kappa, freq=freq, seed=seed, mod=mod)

    # -- embedding helpers (public: models with complex activations may use them)
    def to_complex(self, x):
        """(B, 2 n^2) real state -> (B, n, n) complex matrix."""
        nn = self.n * self.n
        return (_mat(x[:, :nn], self.n, self.n) + 1j * _mat(x[:, nn:], self.n, self.n))

    @staticmethod
    def from_complex(Xc):
        """(B, n, n) complex -> (B, 2 n^2) real embedding [vec Re, vec Im]."""
        return np.concatenate([_vec(Xc.real), _vec(Xc.imag)], axis=1)

    def _Ac(self, t):
        return self._memo("A", t, self.A)

    def data(self, t):
        A = self._Ac(t)
        return {"t": np.float64(t), "Ar": self._memo("Ar", t, lambda s: A.real),
                "Ai": self._memo("Ai", t, lambda s: A.imag)}

    def E(self, x, d):
        Ac = d["Ar"] + 1j * d["Ai"]
        return self.from_complex(Ac @ self.to_complex(x) - self._I)

    def dEdt(self, x, t):
        return self.from_complex(self._memo("dA", t, self.A.dot) @ self.to_complex(x))

    def solve_J(self, x, d, r):
        lu = self._lu(d["Ar"], d["Ai"])
        dX = _left_solve(lambda b: lu_solve(lu, b, check_finite=False), self.to_complex(r))
        return self.from_complex(dX)

    def JT(self, x, d, v):
        AcH = (d["Ar"] - 1j * d["Ai"]).T
        return self.from_complex(AcH @ self.to_complex(v))

    def x_star(self, t):
        Xc = np.linalg.solve(self.A(t), self._I.astype(complex))
        return self.from_complex(Xc[None])[0]

    def capabilities(self):
        return {"complex_embedding"}


def _complex_lu(Ar, Ai):
    return lu_factor(Ar + 1j * Ai, check_finite=False)


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------
from ..registry import register_problem  # noqa: E402

register_problem("matrix_inversion")(MatrixInversion)
register_problem("matrix_inversion_right")(MatrixInversionRight)
register_problem("pseudoinverse")(Pseudoinverse)
register_problem("sylvester")(Sylvester)
register_problem("lyapunov")(Lyapunov)
register_problem("complex_inversion")(ComplexInversion)
