"""Time-varying nonlinear equations and time-varying optimization problems.

Registered keys (see ``docs/spec/problems_opt.md`` for equations, sources and
every assumption):

``nonlinear_equations``
    f(x, t) = 0 with analytic Jacobian.  ``variant="guo2018"`` (default) is
    the 4-D time-varying nonlinear system with closed-form solution used by
    Guo, Xu, Li, Nie, Shao, IEEE TII 14 (2018) 3936-3945 (DOI
    10.1109/TII.2017.2787729) and re-used verbatim as Eq. (33) of
    Frontiers in Neurorobotics 19 (2025) 1576473 (DOI
    10.3389/fnbot.2025.1576473).  ``variant="manufactured"`` is a scalable
    ZNN-Bench construction (NOT from a paper) with a prescribed solution.

``tv_unconstrained_opt``
    min_x f(x, t);  E = grad f,  J = Hessian.  ``variant="zhang2021"``
    (default) is the objective of benchmark (18) of Zhang, Chen, Li,
    Stanimirovic, Zhang, Jin, CAAI Trans. Intell. Technol. 6 (2021) 394-404
    (DOI 10.1049/cit2.12019) without its constraints; ``"quadratic"`` and
    ``"manufactured"`` are scalable ZNN-Bench instances.

``tvqp_eq``
    min 1/2 x'W(t)x + q(t)'x  s.t.  A(t)x = b(t).  State z = [x; lambda],
    E = KKT residual (Zhang & Li, Phys. Lett. A 373 (2009) 1639-1643, DOI
    10.1016/j.physleta.2009.03.011).  Exact solution by solving the KKT
    system.

``tvqp_ineq``
    adds C(t)x <= d(t), treated with the perturbed Fischer-Burmeister (PFB)
    NCP function phi_eps(a, b) = a + b - sqrt(a^2 + b^2 + eps) as in
    Zhang et al. CAAI-TIT 2021 (Lemma 2, Eq. 2) and Yang, Wang, Voyles, Ma,
    Tsinghua Sci. Technol. 31(3) (2026) 1610-1621, DOI 10.26599/TST.2024.9010202
    (arXiv:2412.20477, Eqs. 3-5, 7).  State z = [x; lambda; mu].  The
    reference solution is computed per t (cached) by a smoothing-Newton
    continuation followed by an exact active-set KKT solve.

All E / dEdt / solve_J / JT are vectorised over the batch dimension B.
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np

from ..core import Problem
from ._tv import SmoothOrthogonal, TVMatrix, TVVector

_CACHE_MAX = 200_000


def _bsolve(J, r):
    """Solve J dx = r with J (B, m, m) or (m, m) and r (B, m)."""
    if J.ndim == 2:
        return np.linalg.solve(J, r.T).T
    return np.linalg.solve(J, r[..., None])[..., 0]


def _bsolve_model(J, r):
    """``_bsolve`` for the model-facing ``solve_J``.  An exactly singular J
    (LAPACK zero pivot; reached only far from the solution, e.g. by a
    diverging trajectory of a batch) falls back to the minimum-norm
    least-squares solution -- the ``Problem.solve_J`` contract for
    rank-deficient J, as in ``robot.py`` -- instead of raising and aborting
    every trajectory of the batch (found by experiments/smoke_matrix.py:
    tvqp_ineq with davidenko/vpznn_exp + zead_5i, B = 3).  The reference
    Newton solver keeps calling ``_bsolve`` and still sees the exception."""
    try:
        return _bsolve(J, r)
    except np.linalg.LinAlgError:
        Jb = J if J.ndim == 3 else np.broadcast_to(J, (r.shape[0],) + J.shape)
        with np.errstate(all="ignore"):
            try:
                return np.einsum("bnm,bm->bn", np.linalg.pinv(Jb, rcond=1e-12), r)
            except np.linalg.LinAlgError:          # SVD failure: let simulate flag it
                return np.full((r.shape[0], Jb.shape[-1]), np.nan)


def _bJT(J, v):
    """J^T v with J (B, m, n) or (m, n) and v (B, m) -> (B, n)."""
    if J.ndim == 2:
        return v @ J
    return np.einsum("bij,bi->bj", J, v)


class _DenseJacobianMixin:
    """Generic solve_J / JT from a dense batched ``jacobian(x, d)``."""

    def solve_J(self, x, d, r):
        return _bsolve_model(self.jacobian(x, d), r)

    def JT(self, x, d, v):
        return _bJT(self.jacobian(x, d), v)

    def capabilities(self):
        return {"dense_jacobian"}


# ===========================================================================
# 1. time-varying nonlinear equations
# ===========================================================================
class NonlinearEquations(_DenseJacobianMixin, Problem):
    """Time-varying nonlinear system f(x, t) = 0  (E = f, J = df/dx).

    variant="guo2018" (n = 4), Eq. (33) of Frontiers Neurorobot. 2025
    (system of Guo et al., IEEE TII 2018)::

        f1 = ln(x1) - 1/(t+1)
        f2 = x1 x2 - sin(t) exp(1/(t+1))
        f3 = x1^2 - sin(t) x2 + x3 - 2
        f4 = x1^2 - x2^2 + x3 + x4 - t
        x*(t) = [exp(1/(t+1)), sin t, 2 - exp(2/(t+1)) + sin^2 t, t - 2]

    J is lower triangular with det J = 1 (nonsingular wherever x1 != 0); the
    domain requires x1 > 0 (ln), a state with x1 <= 0 produces NaN and is
    flagged as diverged by ``simulate``.

    variant="manufactured" (ZNN-Bench construction, any n)::

        f(x, t) = A(t) e + beta * e**3,   e = x - x*(t),
        A(t) = S(t) + skew * (Q(t) - Q(t)^T),

    S SPD (``TVMatrix(spd=True)``, condition ~kappa), Q smooth orthogonal, so
    sym(J) = S + 3 beta diag(e^2) > 0: J is always nonsingular and x*(t)
    (``TVVector``) is the unique root.
    """

    family = "nonlinear-system"

    def __init__(self, variant: str = "guo2018", n: Optional[int] = None, kappa: float = 10.0,
                 freq: float = 1.0, beta: float = 1.0, skew: float = 1.0, seed: int = 0,
                 T: float = 10.0):
        self.variant = variant
        self.T = T
        if variant == "guo2018":
            if n not in (None, 4):
                raise ValueError("variant 'guo2018' has n = 4")
            self.n_x = self.m = 4
            self.name = "tvne_guo2018"
            self.cfg = dict(variant=variant)
        elif variant == "manufactured":
            n = 4 if n is None else int(n)
            rng = np.random.default_rng(seed)
            self.S = TVMatrix(n, rng, kappa=kappa, freq=freq, spd=True)
            self.Q = SmoothOrthogonal(n, rng, freq)
            self.xs = TVVector(n, rng, freq=freq)
            self.beta, self.skew = float(beta), float(skew)
            self.n_x = self.m = n
            self.name = f"tvne_manuf_n{n}_k{kappa:g}"
            self.cfg = dict(variant=variant, n=n, kappa=kappa, freq=freq, beta=beta, skew=skew,
                            seed=seed)
        else:
            raise ValueError(f"unknown variant {variant!r}")

    # --- coefficient functions ------------------------------------------
    @staticmethod
    def _c(t):
        ex = np.exp(1.0 / (t + 1.0))
        return np.array([1.0 / (t + 1.0), np.sin(t) * ex, np.sin(t), float(t)])

    @staticmethod
    def _cdot(t):
        ex = np.exp(1.0 / (t + 1.0))
        return np.array([-1.0 / (t + 1.0) ** 2,
                         np.cos(t) * ex - np.sin(t) * ex / (t + 1.0) ** 2,
                         np.cos(t), 1.0])

    def _A(self, t):
        Q = self.Q(t)
        return self.S(t) + self.skew * (Q - Q.T)

    def _Adot(self, t):
        Qd = self.Q.dot(t)
        return self.S.dot(t) + self.skew * (Qd - Qd.T)

    # --- Problem API --------------------------------------------------------
    def data(self, t):
        if self.variant == "guo2018":
            return {"t": np.float64(t), "c": self._c(t)}
        return {"t": np.float64(t), "A": self._A(t), "xs": self.xs(t)}

    def E(self, x, d):
        if self.variant == "guo2018":
            c = d["c"]
            x1, x2, x3, x4 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
            with np.errstate(invalid="ignore", divide="ignore"):
                f1 = np.log(x1) - c[0]
            return np.stack([f1, x1 * x2 - c[1], x1 ** 2 - c[2] * x2 + x3 - 2.0,
                             x1 ** 2 - x2 ** 2 + x3 + x4 - c[3]], axis=1)
        e = x - d["xs"]
        return e @ d["A"].T + self.beta * e ** 3

    def dEdt(self, x, t):
        if self.variant == "guo2018":
            cd = self._cdot(t)
            B = x.shape[0]
            return np.stack([np.full(B, -cd[0]), np.full(B, -cd[1]), -cd[2] * x[:, 1],
                             np.full(B, -cd[3])], axis=1)
        e = x - self.xs(t)
        xsd = self.xs.dot(t)
        return e @ self._Adot(t).T - xsd @ self._A(t).T - 3.0 * self.beta * e ** 2 * xsd

    def jacobian(self, x, d):
        B = x.shape[0]
        if self.variant == "guo2018":
            x1, x2 = x[:, 0], x[:, 1]
            J = np.zeros((B, 4, 4))
            with np.errstate(divide="ignore"):
                J[:, 0, 0] = 1.0 / x1
            J[:, 1, 0], J[:, 1, 1] = x2, x1
            J[:, 2, 0], J[:, 2, 1], J[:, 2, 2] = 2 * x1, -d["c"][2], 1.0
            J[:, 3, 0], J[:, 3, 1], J[:, 3, 2], J[:, 3, 3] = 2 * x1, -2 * x2, 1.0, 1.0
            return J
        e = x - d["xs"]
        J = np.broadcast_to(d["A"], (B, self.n_x, self.n_x)).copy()
        idx = np.arange(self.n_x)
        J[:, idx, idx] += 3.0 * self.beta * e ** 2
        return J

    def solve_J(self, x, d, r):
        if self.variant == "guo2018":
            # forward substitution on the lower-triangular Jacobian
            x1, x2 = x[:, 0], x[:, 1]
            with np.errstate(invalid="ignore", divide="ignore"):
                d1 = x1 * r[:, 0]
                d2 = (r[:, 1] - x2 * d1) / x1
            d3 = r[:, 2] - 2 * x1 * d1 + d["c"][2] * d2
            d4 = r[:, 3] - 2 * x1 * d1 + 2 * x2 * d2 - d3
            return np.stack([d1, d2, d3, d4], axis=1)
        return super().solve_J(x, d, r)

    def x_star(self, t):
        if self.variant == "guo2018":
            ex = np.exp(1.0 / (t + 1.0))
            return np.array([ex, np.sin(t), 2.0 - ex ** 2 + np.sin(t) ** 2, t - 2.0])
        return self.xs(t)

    def x0(self, rng, B):
        if self.variant == "guo2018":
            # ASSUMPTION: x1 > 0 is required by ln(x1); papers start at 0.2*ones(4)
            x = rng.uniform(-2.0, 2.0, size=(B, 4))
            x[:, 0] = rng.uniform(0.2, 3.0, size=B)
            return x
        return rng.uniform(-2.0, 2.0, size=(B, self.n_x))

    def describe(self):
        return {**super().describe(), **self.cfg}


# ===========================================================================
# 2. time-varying unconstrained optimization  (E = grad f, J = Hessian)
# ===========================================================================
def _zhang2021_W(t):
    """V(t) of benchmark (18), Zhang et al. CAAI-TIT 2021, and V'(t)."""
    s, c = np.sin(t), np.cos(t)
    W = np.array([[0.25 * s + 1.0, 0.5 * c], [0.5 * c, 0.25 * c + 1.0]])
    Wd = np.array([[0.25 * c, -0.5 * s], [-0.5 * s, -0.25 * s]])
    return W, Wd


def _zhang2021_q(t):
    return (np.array([np.sin(3 * t), np.cos(3 * t)]),
            np.array([3 * np.cos(3 * t), -3 * np.sin(3 * t)]))


class TVUnconstrainedOpt(Problem):
    """min_x f(x, t):  E = grad_x f,  J = Hessian (symmetric).

    variant="zhang2021" (n = 2): f = 1/2 x'V(t)x + v(t)'x with
        V = [[sin t/4 + 1, cos t/2], [cos t/2, cos t/4 + 1]],
        v = [sin 3t, cos 3t]'
    (objective of benchmark (18) in Zhang et al. CAAI-TIT 2021, constraints
    dropped -- ASSUMPTION/ZNN-Bench usage, see spec).
    variant="quadratic": f = 1/2 x'P(t)x + q(t)'x with P = TVMatrix(spd) of
    condition ~kappa (problem class of time-varying quadratic minimization,
    e.g. Zhang, Yang, Ruan, Neurocomputing 74 (2011) 1710-1719).
    variant="manufactured" (ZNN-Bench construction): with e = x - x*(t),
        f = 1/2 e'P(t)e + beta * sum_i log cosh(e_i),
        grad f = P e + beta tanh(e),   H = P + beta diag(1 - tanh(e)^2).
    """

    family = "unconstrained-optimization"

    def __init__(self, variant: str = "zhang2021", n: Optional[int] = None, kappa: float = 10.0,
                 freq: float = 1.0, beta: float = 1.0, seed: int = 0, T: float = 10.0):
        self.variant = variant
        self.T = T
        if variant == "zhang2021":
            if n not in (None, 2):
                raise ValueError("variant 'zhang2021' has n = 2")
            self.n_x = self.m = 2
            self.name = "tvuo_zhang2021"
            self.cfg = dict(variant=variant)
        elif variant in ("quadratic", "manufactured"):
            n = 4 if n is None else int(n)
            rng = np.random.default_rng(seed)
            self.P = TVMatrix(n, rng, kappa=kappa, freq=freq, spd=True)
            self.vec = TVVector(n, rng, freq=freq)      # q(t) or x*(t)
            self.beta = float(beta)
            self.n_x = self.m = n
            tag = "quad" if variant == "quadratic" else "manuf"
            self.name = f"tvuo_{tag}_n{n}_k{kappa:g}"
            self.cfg = dict(variant=variant, n=n, kappa=kappa, freq=freq, beta=beta, seed=seed)
        else:
            raise ValueError(f"unknown variant {variant!r}")

    def _Pq(self, t, deriv=False):
        if self.variant == "zhang2021":
            W, Wd = _zhang2021_W(t)
            q, qd = _zhang2021_q(t)
            return (Wd, qd) if deriv else (W, q)
        if deriv:
            return self.P.dot(t), self.vec.dot(t)
        return self.P(t), self.vec(t)

    def data(self, t):
        P, v = self._Pq(t)
        key = "xs" if self.variant == "manufactured" else "q"
        return {"t": np.float64(t), "P": P, key: v}

    def f(self, x, d):
        """Objective value per trajectory, shape (B,) (diagnostics)."""
        if self.variant == "manufactured":
            e = x - d["xs"]
            return (0.5 * np.einsum("bi,ij,bj->b", e, d["P"], e)
                    + self.beta * np.sum(np.logaddexp(e, -e) - np.log(2.0), axis=1))
        return 0.5 * np.einsum("bi,ij,bj->b", x, d["P"], x) + x @ d["q"]

    def E(self, x, d):
        if self.variant == "manufactured":
            e = x - d["xs"]
            return e @ d["P"].T + self.beta * np.tanh(e)
        return x @ d["P"].T + d["q"]

    def dEdt(self, x, t):
        if self.variant == "manufactured":
            P, Pd = self.P(t), self.P.dot(t)
            xs, xsd = self.vec(t), self.vec.dot(t)
            e = x - xs
            return e @ Pd.T - xsd @ P.T - self.beta * (1.0 - np.tanh(e) ** 2) * xsd
        Pd, qd = self._Pq(t, deriv=True)
        return x @ Pd.T + qd

    def jacobian(self, x, d):
        B = x.shape[0]
        H = np.broadcast_to(d["P"], (B, self.n_x, self.n_x)).copy()
        if self.variant == "manufactured":
            e = x - d["xs"]
            idx = np.arange(self.n_x)
            H[:, idx, idx] += self.beta * (1.0 - np.tanh(e) ** 2)
        return H

    def solve_J(self, x, d, r):
        if self.variant == "manufactured":
            return _bsolve_model(self.jacobian(x, d), r)
        return np.linalg.solve(d["P"], r.T).T

    def JT(self, x, d, v):
        if self.variant == "manufactured":
            return _bJT(self.jacobian(x, d), v)
        return v @ d["P"]

    def x_star(self, t):
        if self.variant == "manufactured":
            return self.vec(t)
        P, q = self._Pq(t)
        return -np.linalg.solve(P, q)

    def x0(self, rng, B):
        return rng.uniform(-2.0, 2.0, size=(B, self.n_x))

    def capabilities(self):
        # 'gradient_field': E = grad f with f strongly convex (J = Hessian SPD
        # for every variant), the contract used by pc_gradient (Simonetto's
        # exact Eq. (8) step x - alpha E), see CORE_CHANGE_REQUESTS.md.
        return {"dense_jacobian", "gradient_field"}

    def describe(self):
        return {**super().describe(), **self.cfg}


# ===========================================================================
# 3. QP data generators
# ===========================================================================
class _QPData:
    """Time-varying QP data W, q, A, b (and C, d) with exact derivatives."""

    def __init__(self, variant, n=None, p=None, kappa=10.0, freq=1.0, bound=1.0, seed=0,
                 with_ineq=False):
        self.variant = variant
        self.with_ineq = with_ineq
        if variant in ("zhang2021", "yang2025"):
            if n not in (None, 2) or p not in (None, 1):
                raise ValueError(f"variant {variant!r} has n = 2, p = 1")
            self.n, self.p = 2, 1
            self.box = 1.3 if variant == "zhang2021" else 1.0
        elif variant == "random":
            self.n = 4 if n is None else int(n)
            self.p = 1 if p is None else int(p)
            if not 0 <= self.p < self.n:
                raise ValueError("need 0 <= p < n")
            rng = np.random.default_rng(seed)
            self.Wg = TVMatrix(self.n, rng, kappa=kappa, freq=freq, spd=True)
            self.qg = TVVector(self.n, rng, freq=freq, amp=2.0)
            if self.p:
                self.Ag = TVMatrix(self.p, rng, kappa=2.0, freq=freq, n_cols=self.n)
            self.box = float(bound)
            # b(t) = A(t) x_f(t) with a strictly feasible smooth x_f (|x_f| <= box/2)
            self.xf = TVVector(self.n, rng, freq=freq, amp=0.5 * self.box / 1.5)
        else:
            raise ValueError(f"unknown variant {variant!r}")
        self.l = 2 * self.n if with_ineq else 0
        self.C = np.vstack([np.eye(self.n), -np.eye(self.n)])
        self.d = np.full(2 * self.n, self.box)

    def __call__(self, t):
        """Returns (W, q, A, b) and their time derivatives (Wd, qd, Ad, bd)."""
        v = self.variant
        if v == "zhang2021":
            W, Wd = _zhang2021_W(t)
            q, qd = _zhang2021_q(t)
            A = np.array([[np.sin(4 * t), np.cos(4 * t)]])
            Ad = np.array([[4 * np.cos(4 * t), -4 * np.sin(4 * t)]])
            b, bd = np.array([np.cos(2 * t)]), np.array([-2 * np.sin(2 * t)])
        elif v == "yang2025":
            s, c = np.sin(t), np.cos(t)
            W = np.array([[0.25 * c + 1.0, 0.25 * s], [0.25 * s, 0.25 * s + 1.0]])
            Wd = np.array([[-0.25 * s, 0.25 * c], [0.25 * c, 0.25 * c]])
            q = np.array([np.cos(2 * t), np.sin(2 * t)])
            qd = np.array([-2 * np.sin(2 * t), 2 * np.cos(2 * t)])
            A = np.array([[np.sin(2 * t), np.cos(2 * t)]])
            Ad = np.array([[2 * np.cos(2 * t), -2 * np.sin(2 * t)]])
            b, bd = np.array([np.sin(3 * t)]), np.array([3 * np.cos(3 * t)])
        else:
            W, Wd = self.Wg(t), self.Wg.dot(t)
            q, qd = self.qg(t), self.qg.dot(t)
            if self.p:
                A, Ad = self.Ag(t), self.Ag.dot(t)
                xf, xfd = self.xf(t), self.xf.dot(t)
                b, bd = A @ xf, Ad @ xf + A @ xfd
            else:
                A, Ad = np.zeros((0, self.n)), np.zeros((0, self.n))
                b, bd = np.zeros(0), np.zeros(0)
        return (W, q, A, b), (Wd, qd, Ad, bd)


class TVQPEq(Problem):
    """min 1/2 x'W(t)x + q(t)'x  s.t. A(t)x = b(t)   (Zhang & Li 2009).

    Lagrangian L = 1/2 x'Wx + q'x + lambda'(Ax - b); state z = [x; lambda]
    (n + p); E(z, t) = K(t) z + [q; -b] with K = [[W, A'], [A, 0]];
    J = K (state independent, symmetric); dE/dt = K'(t) z + [q'; -b'].
    variant "zhang2021" (default): W, q, A, b of benchmark (18) of Zhang et
    al. CAAI-TIT 2021 (equality part only); "yang2025": data printed in Sec.
    4.1 of Yang et al. TST 2025 (equality part only; see TVQPIneq for the
    caveat that these printed data are not those of the paper's figures);
    "random": scalable synthetic
    instance (W = TVMatrix(spd, kappa), A = TVMatrix(p x n, kappa_A = 2)).
    """

    family = "qp"

    def __init__(self, variant: str = "zhang2021", n: Optional[int] = None,
                 p: Optional[int] = None, kappa: float = 10.0, freq: float = 1.0, seed: int = 0,
                 T: float = 10.0):
        self.qp = _QPData(variant, n=n, p=p, kappa=kappa, freq=freq, bound=1.0, seed=seed)
        self.n, self.p = self.qp.n, self.qp.p
        self.n_x = self.m = self.n + self.p
        self.T = T
        self.variant = variant
        self.name = f"tvqp_eq_{variant}" + (f"_n{self.n}_p{self.p}_k{kappa:g}"
                                            if variant == "random" else "")
        self.cfg = dict(variant=variant, n=self.n, p=self.p)
        if variant == "random":
            self.cfg.update(kappa=kappa, freq=freq, seed=seed)

    @staticmethod
    def _K(W, A):
        n, p = W.shape[0], A.shape[0]
        K = np.zeros((n + p, n + p))
        K[:n, :n], K[:n, n:], K[n:, :n] = W, A.T, A
        return K

    def data(self, t):
        (W, q, A, b), _ = self.qp(t)
        return {"t": np.float64(t), "W": W, "q": q, "A": A, "b": b}

    def E(self, x, d):
        n = self.n
        xx, lam = x[:, :n], x[:, n:]
        return np.concatenate([xx @ d["W"].T + d["q"] + lam @ d["A"], xx @ d["A"].T - d["b"]],
                              axis=1)

    def dEdt(self, x, t):
        _, (Wd, qd, Ad, bd) = self.qp(t)
        n = self.n
        xx, lam = x[:, :n], x[:, n:]
        return np.concatenate([xx @ Wd.T + qd + lam @ Ad, xx @ Ad.T - bd], axis=1)

    def jacobian(self, x, d):
        return np.broadcast_to(self._K(d["W"], d["A"]), (x.shape[0], self.m, self.n_x)).copy()

    def solve_J(self, x, d, r):
        return np.linalg.solve(self._K(d["W"], d["A"]), r.T).T

    def JT(self, x, d, v):
        return v @ self._K(d["W"], d["A"])

    def x_star(self, t):
        (W, q, A, b), _ = self.qp(t)
        return np.linalg.solve(self._K(W, A), np.concatenate([-q, b]))

    def x0(self, rng, B):
        return rng.uniform(-2.0, 2.0, size=(B, self.n_x))

    def capabilities(self):
        return {"dense_jacobian", "kkt"}

    def describe(self):
        return {**super().describe(), **self.cfg}


# ===========================================================================
# 4. QP with equality and inequality constraints (perturbed Fischer-Burmeister)
# ===========================================================================
def pfb(a, b, eps):
    """Perturbed Fischer-Burmeister function a + b - sqrt(a^2 + b^2 + eps).

    phi = 0  <=>  a >= 0, b >= 0, a*b = eps/2  (Zhang et al. CAAI-TIT 2021,
    Sec. 2.1; Yang et al. TST 2025, Eq. 4)."""
    return a + b - np.sqrt(a * a + b * b + eps)


class TVQPIneq(Problem):
    """min 1/2 x'W x + q'x  s.t.  A x = b,  C x <= d   (all time-varying).

    KKT conditions with the perturbed FB function (Zhang et al. CAAI-TIT
    2021, Lemma 2 / Eq. (2); Yang et al. TST 2025, Eqs. (3)-(5)); with
    z = [x; lambda; mu] (n + p + l), s = d - C x:

        E1 = W x + q + A' lambda + C' mu
        E2 = A x - b
        E3 = s + mu - sqrt(s^2 + mu^2 + eps)
        J  = [[W, A', C'], [A, 0, 0], [(Lambda1 - I) C, 0, I - Lambda2]],
        Lambda1 = diag(s / r), Lambda2 = diag(mu / r), r = sqrt(s^2 + mu^2 + eps)
        dE3/dt|_z = (I - Lambda1)(d' - C' x).

    J is the true state Jacobian = P(t) of Yang et al. Eq. (7) (dE/dt = Q z +
    rho there).  The matrix "J(t)" of Zhang et al. Eq. (2) (bottom row
    [-A2, 0, I], with the square root kept inside c(t)) only writes E as
    J(t) y + c(t); it is not dE/dz and is not used here.

    The root of E is the eps-perturbed KKT point (s_i mu_i = eps/2, a point
    on the interior-point central path), NOT the exact QP solution; the bias
    is O(eps) for strictly complementary constraints and O(sqrt(eps)) at
    instants where a constraint switches between active and inactive.

    ``reference="smoothed"`` (default): ``x_star(t)`` = root of E (Newton
    polished, ||E|| ~ 1e-13), consistent with what ZNN converges to.
    ``reference="qp"``: ``x_star(t)`` = exact QP KKT point (active-set solve).
    Both are available as ``smoothed_solution(t)`` / ``qp_solution(t)``.

    Variants: "zhang2021" (default, benchmark (18) of Zhang et al. 2021,
    attributed there to Li et al. IEEE TSMC 51 (2021) 4028; box |x_i| <= 1.3).
    VERIFIED (problems_opt verification): it is also the instance actually
    simulated in Figs. 1-2 of Yang et al. TST 2025 -- the exact QP solution
    overlays their Fig. 1 state curves (median deviation 0.002 for t >= 1 s)
    and reproduces every visible box-saturation interval to ~0.05 s.
    "yang2025" (coefficients PRINTED in Sec. 4.1 of Yang et al.; box
    |x_i| <= 1).  CAUTION: these printed data are NOT the data behind the
    paper's own figures (their curves reach |y_i| = 1.3 > 1); the variant is
    kept as a literal transcription and as an LICQ-degenerate stress test
    (equality row coincides with a box face at t = pi/2 + k pi).
    "random" (scalable synthetic instance, box |x_i| <= bound).
    Default eps = 1e-6 (value used by Yang et al. 2025).
    """

    family = "qp"

    def __init__(self, variant: str = "zhang2021", n: Optional[int] = None,
                 p: Optional[int] = None, eps: float = 1e-6, kappa: float = 10.0,
                 freq: float = 1.0, bound: float = 1.0, seed: int = 0, T: Optional[float] = None,
                 reference: str = "smoothed"):
        if eps <= 0:
            raise ValueError("eps must be > 0 (eps = 0 is the nonsmooth FB function)")
        if reference not in ("smoothed", "qp"):
            raise ValueError("reference must be 'smoothed' or 'qp'")
        self.qp = _QPData(variant, n=n, p=p, kappa=kappa, freq=freq, bound=bound, seed=seed,
                          with_ineq=True)
        self.n, self.p, self.l = self.qp.n, self.qp.p, self.qp.l
        self.n_x = self.m = self.n + self.p + self.l
        self.eps = float(eps)
        self.reference = reference
        self.variant = variant
        self.T = (20.0 if variant == "yang2025" else 10.0) if T is None else T
        self.name = (f"tvqp_ineq_{variant}" + (f"_n{self.n}_p{self.p}_k{kappa:g}"
                                               if variant == "random" else "")
                     + f"_eps{eps:g}")
        self.cfg = dict(variant=variant, n=self.n, p=self.p, l=self.l, eps=eps, box=self.qp.box,
                        reference=reference)
        if variant == "random":
            self.cfg.update(kappa=kappa, freq=freq, seed=seed)
        self._cache = {}
        self._last = None
        self.stats = {"solves": 0, "continuations": 0, "active_set_fallbacks": 0,
                      "active_set_failures": 0, "smoothed_failures": 0}

    # --- data ---------------------------------------------------------------
    def data(self, t):
        (W, q, A, b), _ = self.qp(t)
        return {"t": np.float64(t), "W": W, "q": q, "A": A, "b": b, "C": self.qp.C.copy(),
                "d": self.qp.d.copy()}

    def _split(self, z):
        n, p = self.n, self.p
        return z[:, :n], z[:, n:n + p], z[:, n + p:]

    # --- E, J with explicit eps ----------------------------------------------
    def _E(self, z, d, eps):
        x, lam, mu = self._split(z)
        s = d["d"] - x @ d["C"].T
        return np.concatenate([x @ d["W"].T + d["q"] + lam @ d["A"] + mu @ d["C"],
                               x @ d["A"].T - d["b"], pfb(s, mu, eps)], axis=1)

    def _J(self, z, d, eps):
        n, p, l = self.n, self.p, self.l
        B = z.shape[0]
        x, lam, mu = self._split(z)
        s = d["d"] - x @ d["C"].T
        r = np.sqrt(s * s + mu * mu + eps)
        J = np.zeros((B, self.m, self.n_x))
        J[:, :n, :n] = d["W"]
        J[:, :n, n:n + p] = d["A"].T
        J[:, :n, n + p:] = d["C"].T
        J[:, n:n + p, :n] = d["A"]
        J[:, n + p:, :n] = -(1.0 - s / r)[:, :, None] * d["C"][None]
        idx = np.arange(l)
        J[:, n + p + idx, n + p + idx] = 1.0 - mu / r
        return J

    def E(self, x, d):
        return self._E(x, d, self.eps)

    def jacobian(self, x, d):
        return self._J(x, d, self.eps)

    def solve_J(self, x, d, r):
        return _bsolve_model(self._J(x, d, self.eps), r)

    def JT(self, x, d, v):
        return _bJT(self._J(x, d, self.eps), v)

    def dEdt(self, x, t):
        (W, q, A, b), (Wd, qd, Ad, bd) = self.qp(t)
        C, dd = self.qp.C, self.qp.d          # constant box constraints: C' = 0, d' = 0
        Cd, ddd = np.zeros_like(C), np.zeros_like(dd)
        xx, lam, mu = self._split(x)
        s = dd - xx @ C.T
        r = np.sqrt(s * s + mu * mu + self.eps)
        e3 = (1.0 - s / r) * (ddd - xx @ Cd.T)
        return np.concatenate([xx @ Wd.T + qd + lam @ Ad + mu @ Cd, xx @ Ad.T - bd, e3], axis=1)

    # --- reference solutions ----------------------------------------------------
    def _newton(self, z, d, eps, tol=1e-13, maxit=40):
        """Damped Newton on E(., eps) = 0 for a single state z (m,)."""
        z = z[None].copy()
        F = self._E(z, d, eps)
        nF = np.linalg.norm(F)
        for _ in range(maxit):
            if nF <= tol:
                return z[0], True
            try:
                dz = _bsolve(self._J(z, d, eps), -F)
            except np.linalg.LinAlgError:
                return z[0], False
            a = 1.0
            while True:
                zn = z + a * dz
                Fn = self._E(zn, d, eps)
                nFn = np.linalg.norm(Fn)
                if nFn <= (1.0 - 1e-4 * a) * nF or a < 1e-10:
                    break
                a *= 0.5
            if a < 1e-10:
                return z[0], False
            z, F, nF = zn, Fn, nFn
        return z[0], nF <= tol

    def _continuation(self, d, eps, z=None):
        """Smoothing-Newton path following eps_j = 1, 1e-1, ..., eps."""
        self.stats["continuations"] += 1
        z = np.zeros(self.n_x) if z is None else z.copy()
        levels = list(np.geomspace(1.0, eps, max(2, int(np.ceil(np.log10(1.0 / eps))) + 1)))
        for e in levels[:-1]:
            z, _ = self._newton(z, d, e, tol=1e-10, maxit=60)
        z, ok = self._newton(z, d, levels[-1], tol=1e-13, maxit=80)
        return z, ok

    def _smoothed_at(self, t, d):
        z = None
        ok = False
        if self._last is not None and abs(t - self._last[0]) <= 0.1:
            z, ok = self._newton(self._last[1], d, self.eps)
        if not ok:
            z, ok = self._continuation(d, self.eps)
        if not ok:
            # final relaxed acceptance: residual at round-off level relative to data scale
            ok = np.linalg.norm(self._E(z[None], d, self.eps)) < 1e-10
        if not ok:
            # never return a silently unconverged reference: count it and warn
            self.stats["smoothed_failures"] += 1
            warnings.warn(f"{self.name}: smoothed KKT root not converged at t={t:.6g} "
                          f"(||E|| = {np.linalg.norm(self._E(z[None], d, self.eps)):.2e})")
        self._last = (float(t), z)
        return z

    def _active_set_kkt(self, d, act):
        n, p = self.n, self.p
        Ca = d["C"][act]
        na = Ca.shape[0]
        K = np.zeros((n + p + na, n + p + na))
        K[:n, :n] = d["W"]
        K[:n, n:n + p] = d["A"].T
        K[:n, n + p:] = Ca.T
        K[n:n + p, :n] = d["A"]
        K[n + p:, :n] = Ca
        rhs = np.concatenate([-d["q"], d["b"], d["d"][act]])
        sol = np.linalg.lstsq(K, rhs, rcond=None)[0]
        mu = np.zeros(self.l)
        mu[act] = sol[n + p:]
        return np.concatenate([sol[:n + p], mu])

    def _qp_from(self, z_smooth, d, tol=1e-10):
        """Exact QP KKT point from the smoothed root (active-set identification
        followed by a primal-dual active-set repair loop)."""
        x, lam, mu = self._split(z_smooth[None])
        s = (d["d"] - x @ d["C"].T)[0]
        act = mu[0] > s
        for it in range(2 * self.l + 2):
            z = self._active_set_kkt(d, act)
            xq = z[:self.n]
            mu_q = z[self.n + self.p:]
            sq = d["d"] - d["C"] @ xq
            neg = np.where(act & (mu_q < -tol))[0]
            vio = np.where(~act & (sq < -tol))[0]
            if len(neg) == 0 and len(vio) == 0:
                if it > 0:
                    self.stats["active_set_fallbacks"] += 1
                return z
            if len(neg):
                act[neg[np.argmin(mu_q[neg])]] = False
            else:
                act[vio[np.argmin(sq[vio])]] = True
        self.stats["active_set_failures"] += 1
        return z

    def _solve(self, t):
        key = float(t)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        if len(self._cache) >= _CACHE_MAX:
            self._cache.clear()
        d = self.data(t)
        self.stats["solves"] += 1
        zs = self._smoothed_at(t, d)
        entry = {"smoothed": zs}
        self._cache[key] = entry
        return entry

    def smoothed_solution(self, t):
        """Root of the eps-perturbed KKT system E(z, t) = 0, shape (m,)."""
        return self._solve(t)["smoothed"].copy()

    def qp_solution(self, t):
        """Exact QP KKT point [x*; lambda*; mu*] (mu* >= 0, complementarity
        exact), from an active-set solve seeded by the smoothed root."""
        entry = self._solve(t)
        if "qp" not in entry:
            entry["qp"] = self._qp_from(entry["smoothed"], self.data(t))
        return entry["qp"].copy()

    def smoothing_bias(self, t):
        """||z_eps*(t) - z_QP*(t)||_2 (bias introduced by the perturbation)."""
        return float(np.linalg.norm(self.smoothed_solution(t) - self.qp_solution(t)))

    def x_star(self, t):
        if self.reference == "qp":
            return self.qp_solution(t)
        return self.smoothed_solution(t)

    def x0(self, rng, B):
        return rng.uniform(-1.0, 1.0, size=(B, self.n_x))

    def capabilities(self):
        return {"dense_jacobian", "kkt"}

    def describe(self):
        return {**super().describe(), **self.cfg}


# ===========================================================================
# registration
# ===========================================================================
from ..registry import register_problem  # noqa: E402

register_problem("nonlinear_equations")(NonlinearEquations)
register_problem("tv_unconstrained_opt")(TVUnconstrainedOpt)
register_problem("tvqp_eq")(TVQPEq)
register_problem("tvqp_ineq")(TVQPIneq)
