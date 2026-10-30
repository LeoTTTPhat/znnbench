"""Problem-structured ZNN models.

Explicit matrix-inversion ZNN / Getz-Marsden dynamic inverter
--------------------------------------------------------------
For time-varying matrix inversion, the ZNN design formula applied to the
error E = A(t) X - I gives the *implicit* dynamics (Zhang & Ge, IEEE TNN
16(6) 2005, doi:10.1109/TNN.2005.857946)

    A X_dot = -A_dot X - gamma Phi(A X - I),

which needs a linear solve with A(t) at every evaluation.  Replacing the
unknown A^{-1} by its own running estimate X gives the *explicit* model

    X_dot = -X A_dot X - gamma X Phi(A X - I).                         (*)

With the linear activation, (*) is exactly the Getz-Marsden dynamic matrix
inverter: N. H. Getz and J. E. Marsden, "Dynamic inversion and polar
decomposition of matrices", Proc. 34th IEEE CDC, 1995, pp. 142-147,
doi:10.1109/CDC.1995.478664, Theorem 2.1, Eq. (1)

    Gamma_dot = -mu G[A(t) Gamma - I, Gamma, t] - Gamma A_dot(t) Gamma,
    with the dynamic inverse G[w, Gamma, t] := Gamma w   (Sec. 3, "G_1"),

(journal version: Linear Algebra Appl. 258 (1997) 311-343,
doi:10.1016/S0024-3795(96)00235-2).  The relation between ZNN and this
"Getz-Marsden dynamic system" (GMDS) is the subject of D. Guo and Y. Zhang,
Neurocomputing 97 (2012) 22-32, doi:10.1016/j.neucom.2012.05.012.  The result
is *local*: Theorem 2.1 requires ||X(0) - A(0)^{-1}|| <= r and mu large
enough.  ``newton_schulz_init`` gives the standard safe start
X(0) = A^T / (||A||_1 ||A||_inf) (Pan & Schreiber, SIAM J. Sci. Stat. Comput.
12(5) 1991, doi:10.1137/0912058), for which rho(I - A X(0)) < 1.

Under forward Euler with h = gamma*tau = 1 and no derivative information,
(*) becomes the Newton(-Schulz) iteration X_{k+1} = 2 X_k - X_k A X_k
(Zhang, Ma, Cai, IEEE TCAS-I 56(7) 2009, doi:10.1109/TCSI.2008.2007065).

Implementation note.  For a problem with capability ``'matrix_inverse'`` the
state is x = vec_row(X) (X is n x n) and E(x, d) = vec_row(A X - I), so
dE/dt at fixed X is vec_row(A_dot X).  Hence (*) is the generic design
formula J x_dot = r - dE/dt with J^{-1}(.) (left multiplication by A^{-1})
replaced by left multiplication with X:  X_dot = X (r - dE/dt), r = -gamma
Phi(E).  The model therefore only uses ctx.x, ctx.E and ctx.dEdt and honours
every information regime of ``simulate`` ('bd1'/'bd2' estimate A_dot X by
backward differences of E at fixed X).  This is the convention of
``problems/matrix.py::MatrixInversion`` (key ``matrix_inversion``).

Right (dual) form.  For capability ``'matrix_inverse_right'`` (E = vec_row(X A
- I), ``matrix_inversion_right``) the same substitution, J^{-1}(R) = R A^{-1}
~ R X, gives X_dot = -X A_dot X - gamma Phi(X A - I) X.  ASSUMPTION: this
mirror image is derived by the identical substitution; only the left form (*)
was checked against a primary source (Getz-Marsden 1995).

Not implemented here (see docs/spec/structured_baselines.md)
-----------------------------------------------------------
* Low-computational-complexity ZNN, K. Zheng, S. Li, Y. Zhang, IEEE TAC
  69(7) 2024 4368-4379, doi:10.1109/TAC.2023.3319132.  Its accepted
  manuscript was later obtained; it is registered as ``lcc_znn`` in
  ``models/structured_extra.py`` (docs/spec/structured_extra.md).
* Inverse-free ZNN, J. Chen, Y. Pan, Y. Zhang, S. Li, N. Tan, Neural Networks
  178 (2024) 106462, doi:10.1016/j.neunet.2024.106462.  Its equations could
  not be retrieved from an accessible full text (bot-verification walls;
  publisher version pay-walled).  The design contract forbids inventing them,
  so it stays in ``PENDING_MODELS`` and is not registered (see also
  ``structured_extra.NOT_REGISTERED``).
"""
from __future__ import annotations

import math

import numpy as np

from ..core import Context
from .znn import DesignFormulaZNN

PENDING_MODELS = {
    "inverse_free_znn": {
        "citation": "J. Chen, Y. Pan, Y. Zhang, S. Li, N. Tan, Inverse-free zeroing neural network "
                    "for time-variant nonlinear optimization with manipulator applications, "
                    "Neural Networks 178 (2024) 106462",
        "doi": "10.1016/j.neunet.2024.106462",
        "status": "not implemented: full text not accessible; equations unverified",
    },
}


def newton_schulz_init(A: np.ndarray) -> np.ndarray:
    """Safe initial inverse estimate X0 = A^T / (||A||_1 ||A||_inf).

    Pan & Schreiber (1991): with this X0 the eigenvalues of A X0 lie in
    (0, 1], so rho(I - A X0) < 1 and Newton-Schulz (and the explicit ZNN /
    Getz-Marsden flow started from it) converges.  Returns an (n, n) array;
    use ``x0=np.tile(newton_schulz_init(A0).ravel(), (B, 1))`` in simulate.
    """
    A = np.asarray(A, dtype=float)
    return A.T / (np.abs(A).sum(axis=0).max() * np.abs(A).sum(axis=1).max())


class ExplicitInverseZNN(DesignFormulaZNN):
    """Explicit matrix-inversion ZNN  X_dot = -X A_dot X - gamma X Phi(AX - I).

    Linear activation = Getz-Marsden dynamic inverter (CDC 1995, Thm 2.1,
    Eq. (1) with G[w, Gamma, t] = Gamma w and mu = gamma).  Any activation of
    ``znnbench.activations`` can be used; with ``ctx.implicit`` the
    resolvent (backward-Euler) error rate of ``DesignFormulaZNN`` is used.

    Requires a problem with capability ``'matrix_inverse'`` (x = vec_row(X),
    E = vec_row(A X - I)); with ``side='right'`` it requires
    ``'matrix_inverse_right'`` (E = vec_row(X A - I)) and uses
    X_dot = (r - dE/dt) X.  Cost per evaluation: two n x n matrix products
    (no linear solve), versus one LU solve with n right-hand sides for the
    implicit ZNN.
    """

    family = "znn-structured"
    supports_implicit = True

    def __init__(self, activation="linear", gain=None, name=None, side="left", **act_kw):
        super().__init__(activation=activation, gain=gain, **act_kw)
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' (E = AX - I) or 'right' (E = XA - I)")
        self.side = side
        self.capability = "matrix_inverse" if side == "left" else "matrix_inverse_right"
        self.requires = {self.capability}
        suffix = "" if side == "left" else "_right"
        self.name = name or f"znn_explicit_inverse{suffix}_{self.act.key}"
        self._checked = None

    def _check(self, problem, n_x):
        if self._checked is problem:
            return
        if self.capability not in problem.capabilities():
            raise ValueError(f"{self.name} requires a problem with capability '{self.capability}' "
                             f"(got {sorted(problem.capabilities())} for {problem.name})")
        n = math.isqrt(n_x)
        if n * n != n_x or getattr(problem, "m", n_x) != n_x:
            raise ValueError(f"'{self.capability}' problems must have n_x = m = n^2 (square X)")
        self._checked = problem

    def x0(self, problem, rng, B):
        """Model-supplied initial state (honoured by ``simulate`` when x0 is
        None): the Newton-Schulz safe start X0 = A(0)^T/(||A||_1 ||A||_inf),
        perturbed by 1% relative noise so that the B trajectories differ.
        The dynamic inverter only converges locally (Getz-Marsden Thm 2.1)."""
        A0 = problem.data(0.0).get("A") if hasattr(problem, "data") else None
        if A0 is None or self.capability not in problem.capabilities():
            return None
        X0 = newton_schulz_init(A0).ravel()
        return X0[None, :] * (1.0 + 0.01 * rng.standard_normal((B, X0.size)))

    def rhs(self, ctx: Context):
        B, n_x = ctx.x.shape
        self._check(ctx.problem, n_x)
        n = math.isqrt(n_x)
        X = ctx.x.reshape(B, n, n)
        R = (self.desired_Edot(ctx) - ctx.dEdt).reshape(B, n, n)
        xdot = (np.matmul(X, R) if self.side == "left" else np.matmul(R, X)).reshape(B, n_x)
        return xdot, np.zeros((B, 0))

    def describe(self):
        return {**super().describe(), "requires": sorted(self.requires), "side": self.side,
                "source": "Getz-Marsden CDC 1995 Thm 2.1 Eq.(1); Guo-Zhang Neurocomputing 2012"}


from ..registry import register_model  # noqa: E402


@register_model("znn_explicit_inverse")
def _znn_explicit_inverse(activation="linear", **kw):
    return ExplicitInverseZNN(activation=activation, **kw)


@register_model("znn_explicit_inverse_right")
def _znn_explicit_inverse_right(activation="linear", **kw):
    return ExplicitInverseZNN(activation=activation, side="right", **kw)
