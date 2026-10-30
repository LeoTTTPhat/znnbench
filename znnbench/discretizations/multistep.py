"""Reference discretizations.

Look-ahead (ZeaD-type) formulas
-------------------------------
A one-step-ahead formula approximates the derivative at t_k using the future
sample x_{k+1} and s past samples:

    tau * x_dot_k  ~  c_{-1} x_{k+1} + c_0 x_k + c_1 x_{k-1} + ... + c_{s-1} x_{k-s+1}

Solving for x_{k+1} gives the explicit, causal update

    x_{k+1} = ( tau * x_dot_k - sum_j c_j x_{k-j} ) / c_{-1}.

The continuous model supplies x_dot_k = f(t_k, x_k).  If the formula's
truncation error is O(tau^q) the steady-state residual of linear ZNN is
O(tau^(q+1)); `expected_order` stores q+1.  Before enough history exists the
update falls back to Euler.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from ..core import Discretizer


class LookAhead(Discretizer):
    causal = True

    def __init__(self, coeffs, name, expected_order=None, source=""):
        c = np.asarray(coeffs, dtype=float)
        self.c_next, self.c_past = c[0], c[1:]       # c_{-1}, (c_0, c_1, ...)
        self.name = name
        self.expected_order = expected_order
        self.source = source
        self.s = len(self.c_past)
        self.reset()

    def reset(self):
        self.hist = deque(maxlen=self.s)

    def step(self, k, t, tau, x, z, f):
        xdot, zdot = f(t, x, z, k, self.implicit)
        self.hist.appendleft(x)
        z_next = z + tau * zdot
        if len(self.hist) < self.s:
            return x + tau * xdot, z_next
        acc = tau * xdot
        for cj, xj in zip(self.c_past, self.hist):
            acc = acc - cj * xj
        return acc / self.c_next, z_next

    def describe(self):
        return {**super().describe(), "coeffs": [self.c_next, *self.c_past.tolist()],
                "source": self.source}


def euler():
    """Forward Euler: tau x_dot_k ~ x_{k+1} - x_k  (O(tau^2) residual)."""
    return LookAhead([1.0, -1.0], "euler", expected_order=2, source="forward Euler")


def taylor_zhang():
    """Taylor-Zhang formula: tau x_dot_k ~ x_{k+1} - 3/2 x_k + x_{k-1} - 1/2 x_{k-2}
    (truncation O(tau^2)  =>  O(tau^3) residual).  Formula: Zhang, Jin, Guo,
    Yin, Chou, J. Comput. Appl. Math. 273 (2015) 29-40,
    doi:10.1016/j.cam.2014.05.027; step-size range (0, 1) and optimal h = 0.24:
    Zhang, Gong, Yang, Li, Yang, IEEE TNNLS 30(3) (2019),
    doi:10.1109/TNNLS.2018.2861404 (see docs/spec/discretizations.md)."""
    return LookAhead([1.0, -1.5, 1.0, -0.5], "taylor_zhang", expected_order=3,
                     source="Zhang, Jin, Guo, Yin, Chou, JCAM 273 (2015) 29-40 (formula); "
                            "Zhang, Gong, Yang, Li, Yang, IEEE TNNLS 30(3) 2019 (step size)")


class ImplicitErrorEuler(LookAhead):
    """Euler update with the error dynamics stepped implicitly (resolvent of
    the activation) -- the 'consistent' discretization for sign-type ZNN."""

    implicit = True

    def __init__(self):
        super().__init__([1.0, -1.0], "implicit_euler_err", expected_order=2,
                         source="backward Euler on error dynamics (cf. Polyakov-Efimov-Brogliato 2019)")


class RK4(Discretizer):
    """Classical Runge-Kutta 4 on the continuous model.  NON-CAUSAL: it
    samples the problem data at t_k + tau/2 and t_k + tau."""

    name = "rk4"
    causal = False
    expected_order = None

    def step(self, k, t, tau, x, z, f):
        k1x, k1z = f(t, x, z, k, False)
        k2x, k2z = f(t + tau / 2, x + tau / 2 * k1x, z + tau / 2 * k1z, None, False)
        k3x, k3z = f(t + tau / 2, x + tau / 2 * k2x, z + tau / 2 * k2z, None, False)
        k4x, k4z = f(t + tau, x + tau * k3x, z + tau * k3z, None, False)
        return (x + tau / 6 * (k1x + 2 * k2x + 2 * k3x + k4x),
                z + tau / 6 * (k1z + 2 * k2z + 2 * k3z + k4z))


from ..registry import register_discretization  # noqa: E402

register_discretization("euler")(euler)
register_discretization("taylor_zhang")(taylor_zhang)
register_discretization("implicit_euler_err")(ImplicitErrorEuler)
register_discretization("rk4")(RK4)
