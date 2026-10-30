"""Time-varying linear system A(t) x(t) = b(t)  (reference problem)."""
from __future__ import annotations

import numpy as np

from ..core import Problem
from ._tv import TVMatrix, TVVector


class LinearSystem(Problem):
    """E(x, t) = A(t) x - b(t);  J = A(t);  dE/dt = A_dot x - b_dot."""

    family = "linear-system"

    def __init__(self, n: int = 4, kappa: float = 10.0, freq: float = 1.0, seed: int = 0,
                 T: float = 10.0):
        rng = np.random.default_rng(seed)
        self.A = TVMatrix(n, rng, kappa=kappa, freq=freq)
        self.b = TVVector(n, rng, freq=freq)
        self.n_x = self.m = n
        self.T = T
        self.name = f"linsys_n{n}_k{kappa:g}"
        self.cfg = dict(n=n, kappa=kappa, freq=freq, seed=seed)

    def data(self, t):
        return {"t": np.float64(t), "A": self.A(t), "b": self.b(t)}

    def E(self, x, d):
        return x @ d["A"].T - d["b"]

    def dEdt(self, x, t):
        return x @ self.A.dot(t).T - self.b.dot(t)

    def solve_J(self, x, d, r):
        return np.linalg.solve(d["A"], r.T).T

    def JT(self, x, d, v):
        return v @ d["A"]

    def x_star(self, t):
        return np.linalg.solve(self.A(t), self.b(t))

    def x0(self, rng, B):
        return rng.uniform(-2.0, 2.0, size=(B, self.n_x))

    def describe(self):
        return {**super().describe(), **self.cfg}


from ..registry import register_problem  # noqa: E402

register_problem("linear_system")(LinearSystem)
