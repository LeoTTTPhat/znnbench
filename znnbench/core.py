"""Core abstractions of ZNN-Bench.

A time-varying problem asks for x*(t) such that E(x*(t), t) = 0, where the
error function E is built from time-varying data d(t) (matrices, vectors,
targets, ...).  A neural-dynamics *model* prescribes a continuous-time vector
field x_dot = f(x, z, t) (z = optional auxiliary state, e.g. an integral of
the error).  A *discretization* turns that vector field into a fixed-step
map x_k -> x_{k+1} with sampling period tau.  `simulate` wires the three
together under a chosen noise model and information regime.

Batching convention
-------------------
States are always 2-D arrays of shape (B, n_x): B independent trajectories
that share the same problem data d(t) but may differ in initial state, gain
gamma (shape (B, 1)) and noise realisation.  Errors have shape (B, m).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np

Data = Dict[str, np.ndarray]


# ---------------------------------------------------------------------------
# problems
# ---------------------------------------------------------------------------
class Problem:
    """Base class for time-varying problems.

    Subclasses must set ``name``, ``n_x`` (state dimension), ``m`` (error
    dimension) and implement ``data``, ``E``, ``dEdt``, ``solve_J``, ``JT``
    and ``x0``.  ``x_star`` should return the exact (or high-accuracy
    reference) solution when available.
    """

    name: str = "problem"
    n_x: int = 0
    m: int = 0
    T: float = 10.0          # default horizon (s)
    family: str = "generic"  # e.g. "matrix-inversion", "qp", "robot"

    # --- data ---------------------------------------------------------------
    def data(self, t: float) -> Data:
        """Exact time-varying data at time t (always contains key 't')."""
        raise NotImplementedError

    # --- error function and its derivatives ---------------------------------
    def E(self, x: np.ndarray, d: Data) -> np.ndarray:
        """Error function E(x, d), shape (B, m)."""
        raise NotImplementedError

    def dEdt(self, x: np.ndarray, t: float) -> np.ndarray:
        """Exact partial time derivative dE/dt at fixed x, shape (B, m)."""
        raise NotImplementedError

    def solve_J(self, x: np.ndarray, d: Data, r: np.ndarray) -> np.ndarray:
        """Solve J(x, d) dx = r for dx, shape (B, n_x).

        J = dE/dx.  For non-square J return the minimum-norm least-squares
        solution (pseudoinverse), which is the standard ZNN choice.
        """
        raise NotImplementedError

    def JT(self, x: np.ndarray, d: Data, v: np.ndarray) -> np.ndarray:
        """J(x, d)^T v, shape (B, n_x) (used by gradient neural networks)."""
        raise NotImplementedError

    # --- reference solution and initial states ------------------------------
    def x_star(self, t: float) -> Optional[np.ndarray]:
        """Exact solution x*(t) with shape (n_x,), or None if unknown."""
        return None

    def x0(self, rng: np.random.Generator, B: int) -> np.ndarray:
        """B random initial states, shape (B, n_x)."""
        raise NotImplementedError

    # --- diagnostics -----------------------------------------------------------
    def residual(self, x: np.ndarray, d: Data) -> np.ndarray:
        """Residual norm ||E(x, d)||_2 per trajectory, shape (B,)."""
        return np.linalg.norm(self.E(x, d), axis=-1)

    def capabilities(self) -> set:
        """Optional structure flags that some models require, e.g.
        {'matrix_inverse'} for models that use X as an estimate of A^{-1}."""
        return set()

    def describe(self) -> dict:
        return {"name": self.name, "family": self.family, "n_x": self.n_x, "m": self.m}


# ---------------------------------------------------------------------------
# evaluation context handed to models
# ---------------------------------------------------------------------------
@dataclass
class Context:
    """Everything a model may use to evaluate its vector field at (t, x, z)."""

    problem: Problem
    t: float
    tau: float
    k: int
    x: np.ndarray            # (B, n_x)
    z: np.ndarray            # (B, n_z)
    d: Data                  # data used by the model (may be noisy measurements)
    E: np.ndarray            # (B, m) error at (x, d)
    dEdt: np.ndarray         # (B, m) partial time derivative per information regime
    gamma: np.ndarray        # (B, 1) base design gain
    implicit: bool = False   # discretizer requests the resolvent (implicit) form
    extra: dict = field(default_factory=dict)

    def solve(self, r: np.ndarray) -> np.ndarray:
        return self.problem.solve_J(self.x, self.d, r)

    def JT(self, v: np.ndarray) -> np.ndarray:
        return self.problem.JT(self.x, self.d, v)


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------
class Model:
    """Base class for neural-dynamics models (ZNN, GNN, variants, baselines).

    ``rhs`` returns (x_dot, z_dot).  Models with no auxiliary state return an
    empty (B, 0) array for z_dot.
    """

    name: str = "model"
    family: str = "znn"          # "znn" | "gnn" | "baseline" | ...
    uses_dEdt: bool = True       # needs the time-derivative information
    requires: set = set()        # problem capabilities required
    supports_implicit: bool = False

    def n_z(self, problem: Problem) -> int:
        return 0

    def z0(self, problem: Problem, B: int) -> np.ndarray:
        return np.zeros((B, self.n_z(problem)))

    def rhs(self, ctx: Context):
        raise NotImplementedError

    def describe(self) -> dict:
        return {"name": self.name, "family": self.family}


# ---------------------------------------------------------------------------
# discretizations
# ---------------------------------------------------------------------------
VectorField = Callable[[float, np.ndarray, np.ndarray, int, bool], tuple]


class Discretizer:
    """Base class for fixed-step discretizations of x_dot = f(t, x, z).

    ``step`` receives f(t, x, z, k, implicit) -> (x_dot, z_dot).  A *causal*
    discretizer only evaluates f at the current sampling instant (t_k, x_k):
    it uses no data from the future, which is the setting of discrete-time
    ("future") ZNN computation.  Non-causal schemes (e.g. RK4, which samples
    the data at t_k + tau/2 and t_k + tau) are included as references.
    """

    name: str = "disc"
    causal: bool = True
    expected_order: Optional[int] = None   # p in MSSRE = O(tau^p) for linear ZNN
    implicit: bool = False

    def reset(self):
        pass

    def step(self, k: int, t: float, tau: float, x: np.ndarray, z: np.ndarray,
             f: VectorField):
        raise NotImplementedError

    def describe(self) -> dict:
        return {"name": self.name, "causal": self.causal, "expected_order": self.expected_order}
