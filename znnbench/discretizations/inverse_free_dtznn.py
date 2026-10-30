"""Inverse-free discrete-time ZNNs for time-varying nonlinear equations
(Ma & Huang 2025): the Broyden-aided DTZNN and the combined DTZNN (CDTZNN).

Source (open access, read in full: JATS XML of PMC12289663)
-----------------------------------------------------------
Z. Ma, S. Huang, "Design and analysis of combined discrete-time zeroing
neural network for solving time-varying nonlinear equation with robot
application", Front. Neurorobot. 19 (2025) 1576473,
doi:10.3389/fnbot.2025.1576473.  Their sigma is our tau; f = E, f_t = dE/dt,
J = dE/dx.  Equations and every assumption: docs/spec/inverse_free_dtznn.md.

``broyden_dtznn_zhang2012`` -- Eqs. (6)-(7), attributed by Ma & Huang to
Zhang, Peng, Li, Shi, Ling, Proc. ICCECT 2012, 492-495 (not accessible):

    M_k     = M_{k-1} + (u - M_{k-1} v) u^T M_{k-1} / (u^T M_{k-1} v),            (6)
              u = x_k - x_{k-1},  v = f(x_k, t_k) - f(x_{k-1}, t_k),
    x_{k+1} = x_k - M_k (tau f_t(x_k, t_k) + h f(x_k, t_k)),   h = gamma tau.     (7)

``cdtznn_mahuang2025`` -- Eq. (25) with the start-up (26):

    x_{k+1} = 1/36 x_k + 1/2 x_{k-1} + 3/4 x_{k-2} - 5/18 x_{k-3}
              - M_k (13/6 tau f_t(x_k, t_k) + h f(x_k, t_k)),
    M_{k+1} = 1/36 M_k + 1/2 M_{k-1} + 3/4 M_{k-2} - 5/18 M_{k-3}
              - 13/6 tau M_k Jdot(xdot_k, x_k, t_k) M_k - h M_k (J(x_k, t_k) M_k - I),
    h = 13 gamma tau / 6.

The x-recursion is the look-ahead formula (36, -1, -18, -27, 10)/78 of their
Eq. (20) (= ``zead_5i_mahuang2025``) with J^{-1} replaced by M_k; the
M-recursion is the same formula applied to the Getz-Marsden/Guo-Nie-Yan
inversion dynamics M_dot = -M A_dot M - mu M (A M - I) (their Eq. (36)) with
A = J(x(t), t) and mu = gamma.  Neither recursion solves a linear system.

How the schemes plug into ZNN-Bench (generic model, exact paper for linear ZNN)
------------------------------------------------------------------------------
Every ZNN-Bench model writes its velocity as x_dot = J^{-1} w (``ctx.solve``,
i.e. ``problem.solve_J``), and simulate adds design-formula noise as
J^{-1} nu.  During the single causal call f(t_k, x_k, ...) per step these
discretizers *substitute* J^{-1} by M_k: ``problem.solve_J`` is shadowed by
r -> M_k r for the duration of the call (instance attribute, restored in a
finally block).  The returned velocity is x_dot^M = M_k (w_k + nu_k), and

    x_{k+1} = sum_j a_j x_{k-j} + (tau / c_0) x_dot^M_k,

which for the linear ZNN model (w = -gamma E - dE/dt) is exactly Eq. (25)
(c_0 = 6/13, h_paper = gamma tau / c_0) and Eq. (7) (Euler, c_0 = 1,
h_paper = gamma tau).  Nonlinear activations, gain schedules and the other
solve-based models (ieznn, vpznn, ...) get the same substitution.

Information pattern (per step k; everything causal, data at t <= t_k only)
-------------------------------------------------------------------------
Broyden: the measured sample d_k (``bind_env``'s ``measured``, i.e. the very
  sample the model sees) and E at two states with one sample:
  E(x_k, d_k) and E(x_{k-1}, d_k) (printed Eq. (6), ``secant="current"``) or
  E(x_k, d_{k-1}) and E(x_{k-1}, d_{k-1}) (``secant="previous"``; the variant
  that reproduces three of the six Table 1 entries to 4 digits, see spec).
  One model call with J^{-1} -> M_k.  No Jacobian, no dE/dt beyond the model's.
CDTZNN: one model call with J^{-1} -> M_k, during which a falsy request object
  passed as ``implicit`` captures, through the design-formula hook of
  ``DesignFormulaZNN.desired_Edot``, the context: E_k, the regime-consistent
  dE/dt_k (simulate's ``info``), the regime name and the effective gain
  g = gamma g(t) (it returns the model's own explicit rate -g Phi(E), so the
  model is unchanged).  In addition the scheme evaluates, with the same
  sample d_k: the dense Jacobian J(x_k, d_k) (``problem.jacobian`` if present,
  else assembled exactly from ``problem.JT``), and Jdot_k M_k with
  Jdot_k = d_t J + D_x J [xdot_k], where
    D_x J [xdot_k]  = central difference of J(., d_k) at x_k +- eps xdot_k
                      (exact when J is affine in x),
    d_t J M_k       = regime-consistent: 'analytic' -> central differences of
                      ``problem.dEdt(., t_k)`` at x_k along the columns of M_k
                      (exact when dE/dt is quadratic in x; uses no data at
                      other times); 'bd1'/'bd2' -> backward differences of
                      J(x_k, d_{k-j}) over the measured samples (as simulate
                      does for dE/dt); 'none' -> 0.
  The paper computes Jdot in closed form; a finite difference of the same
  quantity is the ZNN-Bench ASSUMPTION (needed because the Problem API has no
  second derivatives).
  xdot_k inside Jdot: the paper prints "xdot_k = f(x_k,t_k) + f_t(x_k,t_k)",
  which as written gives SSCE 0.18 instead of 4.874e-7 (Table 1).  The default
  ``xdot="ctznn_unit_gain"``, xdot_k = -M_k (f_t + f) (the CTZNN (3) with
  gamma = 1 and J^- -> M_k), reproduces all six CDTZNN entries of Table 1 to
  4 significant digits -- ASSUMPTION (missing "-M_k(.)" in the print).
  ``"model"`` uses the model velocity x_dot^M (generic; 0.3-0.9 % off at
  tau = 0.01), ``"printed"`` the literal print.

Initial matrix M_0 (not specified in the paper; ASSUMPTION): ``"identity"``
(Broyden default: with it Broyden reproduces three Table 1 entries to 4 digits,
while the Newton-Schulz start diverges on guo2018), ``"newton_schulz"``
(CDTZNN default: J0^T/(||J0||_1 ||J0||_inf), the inverse-free safe start for any
nonsingular J0; CDTZNN gives the same six Table 1 values with either start,
but from the identity it diverges on indefinite or large-norm Jacobians such
as tvqp_eq, sylvester, lyapunov), ``"inverse"`` (J^{-1}(x_0, d_0) by one true
solve), or an array.  Start-up (k = 0, 1, 2 of CDTZNN): forward Euler of the model
velocity and of the inversion dynamics (ZNN-Bench LookAhead convention).
Eq. (26) uses h_paper instead of gamma tau in the f and M-residual terms of
the start-up; this changes only the transient (the steady state agrees to
4e-5 relative, see tests).

Scope: square J only (M_k ~ J^{-1}); dense n_x x n_x matrices per trajectory,
so n_x <= ``max_dim`` (default 400).
"""
from __future__ import annotations

import warnings
from collections import deque
from fractions import Fraction as Fr
from typing import Optional

import numpy as np

from ..core import Discretizer
from .zead import lookahead_stability, root_condition
from .zead_named import COEFFS_5I_MAHUANG2025, MAHUANG_DOI, PAPER_H_FACTOR

#: tau-form coefficients of the CDTZNN look-ahead formula, Ma & Huang Eq. (20)
CDTZNN_COEFFS = tuple(COEFFS_5I_MAHUANG2025)                    # (36, -1, -18, -27, 10)/78
CDTZNN_C0 = CDTZNN_COEFFS[0]                                     # 6/13
CDTZNN_A = tuple(-c / CDTZNN_C0 for c in CDTZNN_COEFFS[1:])      # (1/36, 1/2, 3/4, -5/18)
assert CDTZNN_A == (Fr(1, 36), Fr(1, 2), Fr(3, 4), Fr(-5, 18))   # Eq. (25) as printed
assert 1 / CDTZNN_C0 == Fr(13, 6)

#: h_ours = TABLE1_H_FACTOR[key] * h_paper for every Table 1 model of Ma & Huang 2025
TABLE1_H_FACTOR = {"euler": Fr(1), "broyden_dtznn_zhang2012": Fr(1),
                   "cdtznn_mahuang2025": CDTZNN_C0, **PAPER_H_FACTOR}

#: Ma & Huang 2025, Table 1 (SSCE on the guo2018 system, x0 = 0.2*ones(4)), keyed by
#: ZNN-Bench key -> {(tau, h_paper): value}.  The Taylor DTZNN-III entry at
#: (0.01, 0.20) is printed as 6.837e-8; it is 6.837e-9 (typo: the printed value breaks
#: the monotone decrease in h of that row, and 6.837e-9 is reproduced to 4 digits).
MAHUANG2025_TABLE1 = {
    "euler": {(0.01, 0.10): 1.399e-3, (0.01, 0.15): 9.385e-4, (0.01, 0.20): 7.056e-4,
              (0.001, 0.10): 1.414e-5, (0.001, 0.15): 9.422e-6, (0.001, 0.20): 7.070e-6},
    "broyden_dtznn_zhang2012": {(0.01, 0.10): 3.660e-3, (0.01, 0.15): 2.397e-3,
                                (0.01, 0.20): 2.109e-3, (0.001, 0.10): 4.616e-5,
                                (0.001, 0.15): 3.066e-5, (0.001, 0.20): 2.927e-5},
    "zead_5i": {(0.01, 0.10): 4.554e-7, (0.01, 0.15): 3.173e-7, (0.01, 0.20): 2.420e-7,
                (0.001, 0.10): 4.945e-11, (0.001, 0.15): 3.299e-11, (0.001, 0.20): 2.474e-11},
    "zead_6i_guo2017": {(0.01, 0.10): 6.612e-7, (0.01, 0.15): 4.569e-7, (0.01, 0.20): 3.472e-7,
                        (0.001, 0.10): 7.059e-11, (0.001, 0.15): 4.712e-11,
                        (0.001, 0.20): 3.535e-11},
    "zead_7i_cang2024": {(0.01, 0.10): 1.260e-8, (0.01, 0.15): 8.913e-9,
                         (0.01, 0.20): 6.837e-9,            # printed: 6.837e-8 (typo)
                         (0.001, 0.10): 1.447e-13, (0.001, 0.15): 9.763e-14,
                         (0.001, 0.20): 7.618e-14},
    "cdtznn_mahuang2025": {(0.01, 0.10): 4.874e-7, (0.01, 0.15): 3.362e-7, (0.01, 0.20): 2.554e-7,
                           (0.001, 0.10): 5.184e-11, (0.001, 0.15): 3.456e-11,
                           (0.001, 0.20): 2.592e-11},
}
MAHUANG2025_TABLE1_PRINTED_TYPOS = {("zead_7i_cang2024", 0.01, 0.20): 6.837e-8}


def gamma_from_paper_h(key: str, h_paper: float, tau: float) -> float:
    """ZNN-Bench gain gamma reproducing Ma & Huang's step size h_paper for ``key``."""
    return float(TABLE1_H_FACTOR[key]) * float(h_paper) / float(tau)


# ---------------------------------------------------------------------------
# helpers: probe, solve substitution, Jacobian information
# ---------------------------------------------------------------------------
class _DesignFormulaProbe:
    """Passed as the ``implicit`` argument of f.  Falsy, so models that test
    ``if ctx.implicit`` take their explicit branch; models built on
    ``DesignFormulaZNN`` call ``desired_Edot(act, ctx, g)`` (the request
    protocol of consistent.py), which records the context and returns the
    model's own explicit design-formula rate -g Phi(E)."""

    def __init__(self):
        self.ctx = None
        self.g = None

    def __bool__(self):
        return False

    def desired_Edot(self, act, ctx, g):
        self.ctx = ctx
        self.g = np.asarray(g, dtype=float)
        return -g * act(ctx.E)


class _SolveSubstitution:
    """Context manager: ``problem.solve_J`` -> (x, d, r) |-> M r for one call of
    the vector field.  Counts substituted calls; a call whose right-hand side
    does not match M (other batch size) falls back to the true solve and is
    counted in ``fallback``."""

    def __init__(self, problem, M):
        self.problem, self.M = problem, M
        self.calls = 0
        self.fallback = 0

    def __enter__(self):
        p = self.problem
        try:
            inst = vars(p)
        except TypeError as exc:                        # __slots__ problem
            raise RuntimeError(f"{type(p).__name__} has no instance __dict__; the inverse-free "
                               "schemes cannot substitute solve_J") from exc
        self._had = "solve_J" in inst
        self._old = inst.get("solve_J")
        orig = p.solve_J
        M = self.M

        def solve_J(x, d, r):
            r = np.asarray(r)
            if r.ndim == 2 and r.shape[0] == M.shape[0] and r.shape[1] == M.shape[2]:
                self.calls += 1
                return np.einsum("bij,bj->bi", M, r)
            self.fallback += 1
            return orig(x, d, r)

        p.solve_J = solve_J
        return self

    def __exit__(self, *exc):
        if self._had:
            self.problem.solve_J = self._old
        else:
            try:
                del self.problem.solve_J
            except AttributeError:
                pass
        return False


def dense_jacobian(problem, x, d, m: Optional[int] = None) -> np.ndarray:
    """J(x, d), shape (B, m, n_x): ``problem.jacobian`` when available, else
    assembled exactly row by row from ``problem.JT`` (one batched call with
    B*m unit vectors)."""
    B, n = x.shape
    jac = getattr(problem, "jacobian", None)
    if callable(jac):
        try:
            J = np.asarray(jac(x, d), dtype=float)
            if J.ndim == 3 and J.shape[0] == B and J.shape[2] == n:
                return J
        except NotImplementedError:
            pass
    if m is None:
        m = int(problem.E(x, d).shape[1])
    xr = np.repeat(x, m, axis=0)
    V = np.tile(np.eye(m), (B, 1))
    return np.asarray(problem.JT(xr, d, V), dtype=float).reshape(B, m, n)


def _dir_eps(x, v, rel=6e-6):
    """Central-difference step along v (per row), ~ cube root of machine eps."""
    vn = np.linalg.norm(v, axis=-1, keepdims=True)
    return rel * (1.0 + np.linalg.norm(x, axis=-1, keepdims=True)) / np.where(vn > 0, vn, 1.0)


def jacobian_directional(problem, x, d, v, m=None) -> np.ndarray:
    """D_x J(x, d)[v] by a central difference at fixed data d, shape (B, m, n)."""
    eps = _dir_eps(x, v)
    Jp = dense_jacobian(problem, x + eps * v, d, m)
    Jm = dense_jacobian(problem, x - eps * v, d, m)
    return (Jp - Jm) / (2.0 * eps[:, :, None])


def dtJ_times_analytic(problem, x, t, Mmat) -> np.ndarray:
    """(d_t J)(x, t) M = D_x(dE/dt)(x, t)[columns of M], shape (B, m, k).  Uses
    ``problem.dEdt`` at t only (central differences in x along each column of
    M, all columns in one batched call)."""
    B, n = x.shape
    k = Mmat.shape[2]
    dirs = np.transpose(Mmat, (0, 2, 1)).reshape(B * k, n)
    xr = np.repeat(x, k, axis=0)
    eps = _dir_eps(xr, dirs)
    cols = (problem.dEdt(xr + eps * dirs, t) - problem.dEdt(xr - eps * dirs, t)) / (2.0 * eps)
    return cols.reshape(B, k, -1).transpose(0, 2, 1)


# ---------------------------------------------------------------------------
# common base
# ---------------------------------------------------------------------------
class _InverseFreeBase(Discretizer):
    causal = True
    implicit = False
    M0_CHOICES = ("identity", "newton_schulz", "inverse")

    def __init__(self, M0="identity", max_dim: int = 400, record_inverse_error: bool = False):
        if isinstance(M0, str) and M0 not in self.M0_CHOICES:
            raise ValueError(f"M0 must be one of {self.M0_CHOICES} or an array")
        self.M0 = M0
        self.max_dim = int(max_dim)
        self.record_inverse_error = bool(record_inverse_error)
        self._problem = None
        self._measured = None
        self._info_env = None
        self.reset()

    # -------------------------------------------------------------- environment
    def bind_env(self, problem, measured=None, stage_data=None, **kw):
        """Called by ``simulate`` right after ``reset``: the problem and a
        callable returning the measured sample at t_k (the sample the model
        sees).  ``stage_data`` is not used (causal schemes).  An ``info``
        keyword, if a future core passes it, is recorded."""
        self._problem = problem
        self._measured = measured
        self._info_env = kw.get("info")
        return self

    def bind(self, problem):
        """Direct use outside ``simulate``: exact data ``problem.data(t_k)``."""
        return self.bind_env(problem)

    def _sample(self, t):
        p = self._problem
        if p is None:
            raise RuntimeError(f"{self.name}: needs the Problem; simulate calls bind_env(problem=...), "
                               "otherwise call disc.bind(problem) first")
        return p, (self._measured() if self._measured is not None else p.data(t))

    def reset(self):
        self.M = None
        self.n_substituted_solves = 0
        self.n_true_solves = 0
        self.inverse_error = []          # ||J_k M_k - I||_F per step if record_inverse_error
        self._checked = False
        self._warned_nosolve = False

    # --------------------------------------------------------------- set-up
    def _check(self, problem, x, d):
        if self._checked:
            return
        B, n = x.shape
        m = int(problem.E(x[:1], d).shape[1])
        if m != n:
            raise ValueError(f"{self.name}: M_k approximates J^{{-1}} and needs a square Jacobian; "
                             f"problem {getattr(problem, 'name', '?')} has m = {m}, n_x = {n}")
        if n > self.max_dim:
            raise ValueError(f"{self.name}: dense {n}x{n} matrix state per trajectory exceeds "
                             f"max_dim = {self.max_dim} (pass max_dim=... to override)")
        self._m = m
        self._checked = True

    def _initial_M(self, problem, x, d):
        B, n = x.shape
        if not isinstance(self.M0, str):
            M = np.asarray(self.M0, dtype=float)
            return np.broadcast_to(M, (B, n, n)).copy()
        if self.M0 == "identity":
            return np.broadcast_to(np.eye(n), (B, n, n)).copy()
        J = dense_jacobian(problem, x, d, self._m)
        if self.M0 == "newton_schulz":
            n1 = np.abs(J).sum(axis=1).max(axis=-1)
            ninf = np.abs(J).sum(axis=2).max(axis=-1)
            return np.transpose(J, (0, 2, 1)) / (n1 * ninf)[:, None, None]
        # "inverse": one true linear solve with n right-hand sides (not inverse-free)
        xr = np.repeat(x, n, axis=0)
        cols = problem.solve_J(xr, d, np.tile(np.eye(n), (B, 1)))
        self.n_true_solves += 1
        return np.transpose(cols.reshape(B, n, n), (0, 2, 1))

    def _velocity(self, problem, f, t, x, z, k, M, implicit):
        with _SolveSubstitution(problem, M) as sub:
            xdot, zdot = f(t, x, z, k, implicit)
        self.n_substituted_solves += sub.calls
        self.n_true_solves += sub.fallback
        if sub.calls == 0 and not self._warned_nosolve:
            self._warned_nosolve = True
            warnings.warn(f"{self.name}: the model applied no J^-1 through solve_J; the inverse-free "
                          "substitution has no effect on its velocity", RuntimeWarning, stacklevel=3)
        return xdot, zdot

    def _record_inverse_error(self, J, M):
        if self.record_inverse_error:
            n = M.shape[1]
            self.inverse_error.append(np.linalg.norm(np.matmul(J, M) - np.eye(n), axis=(1, 2)))

    def describe(self):
        return {**super().describe(), "source": self.source, "doi": self.doi,
                "M0": self.M0 if isinstance(self.M0, str) else "array",
                "inverse_free": True, "information": self.information}


# ---------------------------------------------------------------------------
# Broyden-aided DTZNN, Eqs. (6)-(7)
# ---------------------------------------------------------------------------
class BroydenDTZNN(_InverseFreeBase):
    """Broyden-aided DTZNN (Ma & Huang 2025, Eqs. (6)-(7); Zhang, Peng, Li,
    Shi, Ling, ICCECT 2012).  Euler look-ahead with J^{-1} replaced by the
    Broyden ("good", inverse Sherman-Morrison form) secant matrix M_k.

    Parameters
    ----------
    secant : "current" -- v = f(x_k, t_k) - f(x_{k-1}, t_k), Eq. (6) as printed;
             "previous" -- v = f(x_k, t_{k-1}) - f(x_{k-1}, t_{k-1}) (ASSUMPTION: the
             variant that reproduces 3 of the 6 Table 1 entries to 4 digits).
    M0     : "identity" (default; ASSUMPTION, the paper does not state M_0),
             "newton_schulz", "inverse" or an array.
    skip_tol : the update is skipped for a trajectory when
             |u^T M v| <= skip_tol ||u|| ||M v|| or it is not finite (ASSUMPTION:
             standard Broyden safeguard; never triggered on the Table 1 runs).
    """

    name = "broyden_dtznn_zhang2012"
    expected_order = 2            # Lemma 2 of Ma & Huang (claimed); see spec for measured slopes
    source = ("Ma, Huang, Front. Neurorobot. 19 (2025) 1576473, Eqs. (6)-(7); Broyden-aided DTZNN "
              "of Zhang, Peng, Li, Shi, Ling, Proc. ICCECT 2012, 492-495 (not accessible); "
              "Broyden, Math. Comp. 19 (1965) 577-593")
    doi = MAHUANG_DOI + "; 10.1090/S0025-5718-1965-0198670-6"
    information = ("causal: measured sample d_k; E(x_k, .) and E(x_{k-1}, .) at one sample "
                   "(d_k for secant='current', d_{k-1} for 'previous'); model velocity with "
                   "J^-1 -> M_k; no Jacobian, no linear solve")

    def __init__(self, secant: str = "current", M0="identity", skip_tol: float = 1e-12,
                 max_dim: int = 400, record_inverse_error: bool = False):
        if secant not in ("current", "previous"):
            raise ValueError("secant must be 'current' (Eq. (6) as printed) or 'previous'")
        self.secant = secant
        self.skip_tol = float(skip_tol)
        super().__init__(M0=M0, max_dim=max_dim, record_inverse_error=record_inverse_error)

    def reset(self):
        super().reset()
        self.x_prev = None
        self.d_prev = None
        self.n_skipped_updates = 0

    def step(self, k, t, tau, x, z, f):
        problem, d = self._sample(t)
        self._check(problem, x, d)
        with np.errstate(all="ignore"):
            if self.M is None:
                self.M = self._initial_M(problem, x, d)
            M = self.M
            if self.x_prev is not None:
                u = x - self.x_prev
                dv = d if self.secant == "current" else self.d_prev
                v = problem.E(x, dv) - problem.E(self.x_prev, dv)
                Mv = np.einsum("bij,bj->bi", M, v)
                uM = np.einsum("bi,bij->bj", u, M)
                den = np.einsum("bi,bi->b", u, Mv)
                scale = np.linalg.norm(u, axis=1) * np.linalg.norm(Mv, axis=1)
                ok = np.isfinite(den) & np.isfinite(scale) & (np.abs(den) > self.skip_tol * scale)
                self.n_skipped_updates += int((~ok).sum())
                safe = np.where(ok, den, 1.0)
                upd = (u - Mv)[:, :, None] * uM[:, None, :] / safe[:, None, None]
                M = np.where(ok[:, None, None], M + upd, M)
            if self.record_inverse_error:
                self._record_inverse_error(dense_jacobian(problem, x, d, self._m), M)
        xdot, zdot = self._velocity(problem, f, t, x, z, k, M, self.implicit)
        self.M = M
        self.x_prev = np.array(x, copy=True)
        self.d_prev = d
        return x + tau * xdot, z + tau * zdot

    def describe(self):
        return {**super().describe(), "secant": self.secant, "skip_tol": self.skip_tol,
                "h_paper": "h = gamma*tau (Euler, c_0 = 1)"}


# ---------------------------------------------------------------------------
# CDTZNN, Eqs. (25)-(26)
# ---------------------------------------------------------------------------
class CDTZNN(_InverseFreeBase):
    """Combined DTZNN of Ma & Huang 2025, Eq. (25) (start-up Eq. (26)).

    Parameters
    ----------
    mu   : gain of the inversion recursion.  "gamma" (default, the paper:
           same h for both recursions) uses the effective gain g = gamma g(t)
           captured from the model's design formula; a float mu (1/s) is used
           as given (needed for models that do not expose the design formula).
    xdot : velocity used inside Jdot: "ctznn_unit_gain" (default, -M_k (f_t + f),
           ASSUMPTION reproducing Table 1), "model" (x_dot^M, the paired model's
           velocity with M_k), "printed" (f + f_t as printed).
    info : information regime for d_t J and f_t when the model does not expose
           its context (otherwise simulate's regime, read from the context, is
           used): "analytic", "bd1", "bd2", "none".
    M0   : "newton_schulz" (default; ASSUMPTION, see module docstring), "identity",
           "inverse" or an array.
    """

    name = "cdtznn_mahuang2025"
    expected_order = 4            # Theorem 4 (SSCE O(tau^4) at fixed h); reproduced
    source = ("Ma, Huang, Front. Neurorobot. 19 (2025) 1576473, Eqs. (20), (24)-(26), (36)-(38); "
              "inversion dynamics of Guo, Nie, Yan, IEEE TSMC-S 47(8) 2017 / Getz, Marsden, CDC 1995")
    doi = MAHUANG_DOI
    information = ("causal: measured sample d_k; model call with J^-1 -> M_k (captures E_k, "
                   "regime dE/dt_k, gain); dense J(x_k, d_k) (jacobian or JT); D_xJ(x_k, d_k)[xdot] "
                   "by central difference at fixed d_k; d_tJ M_k per regime (analytic: x-differences "
                   "of dEdt(., t_k); bd1/bd2: J at past samples; none: 0); no linear solve")
    XDOT_CHOICES = ("ctznn_unit_gain", "model", "printed")

    def __init__(self, mu="gamma", xdot: str = "ctznn_unit_gain", info: Optional[str] = None,
                 M0="newton_schulz", max_dim: int = 400, record_inverse_error: bool = False):
        if xdot not in self.XDOT_CHOICES:
            raise ValueError(f"xdot must be one of {self.XDOT_CHOICES}")
        if info not in (None, "analytic", "bd1", "bd2", "none"):
            raise ValueError("info must be None, 'analytic', 'bd1', 'bd2' or 'none'")
        if not (mu == "gamma" or (np.isscalar(mu) and float(mu) > 0)):
            raise ValueError("mu must be 'gamma' or a positive float")
        self.mu = mu
        self.xdot = xdot
        self.info = info
        self.coeffs = CDTZNN_COEFFS
        self.c0 = float(CDTZNN_C0)
        self.a = tuple(float(c) for c in CDTZNN_A)
        super().__init__(M0=M0, max_dim=max_dim, record_inverse_error=record_inverse_error)

    def reset(self):
        super().reset()
        self.hx = deque(maxlen=4)
        self.hM = deque(maxlen=4)
        self.data_hist = deque(maxlen=3)
        self.info_used = None

    # analysis helpers (the two recursions share the characteristic polynomial)
    def char_poly(self):
        return [float(c) for c in self.coeffs]

    def root_info(self):
        return root_condition(self.char_poly())

    def stability(self, h_hi: float = 4.0):
        """Linear-ZNN step-size interval of the x-recursion (h = gamma tau)."""
        return lookahead_stability(self.coeffs, h_hi)

    def _dEdt_regime(self, problem, x, t, tau, info):
        dh = self.data_hist
        if info == "analytic":
            return problem.dEdt(x, t)
        if info == "bd2" and len(dh) >= 3:
            return (3 * problem.E(x, dh[0]) - 4 * problem.E(x, dh[1]) + problem.E(x, dh[2])) / (2 * tau)
        if info in ("bd1", "bd2") and len(dh) >= 2:
            return (problem.E(x, dh[0]) - problem.E(x, dh[1])) / tau
        return np.zeros((x.shape[0], self._m))

    def _dtJ_M(self, problem, x, t, tau, M, info, J):
        dh = self.data_hist
        if info == "analytic":
            return dtJ_times_analytic(problem, x, t, M)
        if info == "bd2" and len(dh) >= 3:
            dtJ = (3 * J - 4 * dense_jacobian(problem, x, dh[1], self._m)
                   + dense_jacobian(problem, x, dh[2], self._m)) / (2 * tau)
            return np.matmul(dtJ, M)
        if info in ("bd1", "bd2") and len(dh) >= 2:
            return np.matmul((J - dense_jacobian(problem, x, dh[1], self._m)) / tau, M)
        return np.zeros_like(M)

    def step(self, k, t, tau, x, z, f):
        problem, d = self._sample(t)
        self._check(problem, x, d)
        B, n = x.shape
        with np.errstate(all="ignore"):
            if self.M is None:
                self.M = self._initial_M(problem, x, d)
        M = self.M
        probe = _DesignFormulaProbe()
        xdot, zdot = self._velocity(problem, f, t, x, z, k, M, probe)
        self.data_hist.appendleft(d)
        ctx = probe.ctx
        if ctx is not None:
            E, ft = ctx.E, ctx.dEdt
            info = (ctx.extra or {}).get("info") or self._info_env or self.info or "analytic"
            mu = probe.g if self.mu == "gamma" else float(self.mu)
        else:
            info = self._info_env or self.info
            if self.mu == "gamma" or info is None:
                raise TypeError(
                    f"{self.name}: the model does not expose the ZNN design formula (models built on "
                    "DesignFormulaZNN: znn, znn_consistent, vpznn, vpznn_exp), so the gain and the "
                    "information regime are unknown; pass mu=<float> and info=<regime> explicitly")
            mu = float(self.mu)
            with np.errstate(all="ignore"):
                E = problem.E(x, d)
                ft = self._dEdt_regime(problem, x, t, tau, info)
        self.info_used = info
        with np.errstate(all="ignore"):
            J = dense_jacobian(problem, x, d, self._m)
            if self.xdot == "ctznn_unit_gain":
                v = -np.einsum("bij,bj->bi", M, ft + E)
            elif self.xdot == "model":
                v = xdot
            else:                                   # "printed": f + f_t
                v = E + ft
            JdM = (np.matmul(jacobian_directional(problem, x, d, v, self._m), M)
                   + self._dtJ_M(problem, x, t, tau, M, info, J))
            MJdM = np.matmul(M, JdM)
            JM = np.matmul(J, M)
            R = np.matmul(M, JM) - M                        # M (J M - I)
            if self.record_inverse_error:
                self.inverse_error.append(np.linalg.norm(JM - np.eye(n), axis=(1, 2)))
            mu = np.broadcast_to(np.asarray(mu, dtype=float).reshape(-1, 1), (B, 1))[:, :, None]
            self.hx.appendleft(np.array(x, copy=True))
            self.hM.appendleft(M)
            if len(self.hx) < len(self.a):                  # start-up k = 0, 1, 2 (Euler)
                x_next = x + tau * xdot
                M_next = M - tau * MJdM - tau * mu * R
            else:
                x_next = (tau / self.c0) * xdot
                M_next = -(tau / self.c0) * (MJdM + mu * R)
                for aj, xj, Mj in zip(self.a, self.hx, self.hM):
                    x_next = x_next + aj * xj
                    M_next = M_next + aj * Mj
        self.M = M_next
        return x_next, z + tau * zdot

    def describe(self):
        return {**super().describe(), "coeffs": [str(c) for c in self.coeffs],
                "integer_coeffs": [36, -1, -18, -27, 10], "denominator": 78,
                "x_recursion_key": "zead_5i_mahuang2025", "mu": self.mu, "xdot": self.xdot,
                "info": self.info, "h_paper": "h = gamma*tau/c_0 = 13 gamma tau/6 (h_ours = 6/13 h_paper)"}


from ..registry import register_discretization  # noqa: E402


@register_discretization("broyden_dtznn_zhang2012")
def broyden_dtznn_zhang2012(secant="current", M0="identity", skip_tol=1e-12, max_dim=400,
                            record_inverse_error=False):
    """Broyden-aided DTZNN, Ma & Huang 2025 Eqs. (6)-(7) (inverse-free, Euler)."""
    return BroydenDTZNN(secant=secant, M0=M0, skip_tol=skip_tol, max_dim=max_dim,
                        record_inverse_error=record_inverse_error)


@register_discretization("cdtznn_mahuang2025")
def cdtznn_mahuang2025(mu="gamma", xdot="ctznn_unit_gain", info=None, M0="newton_schulz", max_dim=400,
                       record_inverse_error=False):
    """Combined DTZNN, Ma & Huang 2025 Eq. (25) (inverse-free, 5-instant formula (20))."""
    return CDTZNN(mu=mu, xdot=xdot, info=info, M0=M0, max_dim=max_dim,
                  record_inverse_error=record_inverse_error)


#: keys registered by this module with their expected residual order
INVERSE_FREE_KEYS = {"broyden_dtznn_zhang2012": 2, "cdtznn_mahuang2025": 4}
