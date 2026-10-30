"""Explicit Adams-Bashforth (AB) discretizations (causal).

    x_{k+1} = x_k + tau * sum_{j=0}^{q-1} beta_j f(t_{k-j}, x_{k-j})

Only the current derivative f(t_k, x_k) is evaluated at each step; the q-1
previous evaluations are stored, so the scheme uses no future data (causal in
the ZNN-Bench sense).  Coefficients are the classical ones (Hairer, Norsett &
Wanner, Solving ODEs I, Sec. III.1, Table 1.1; Butcher 2016, Sec. 24):

    AB2: (3, -1)/2,   AB3: (23, -16, 5)/12,   AB4: (55, -59, 37, -9)/24.

ZNN usage: Qiu, Zhang, Guo, Yang & Li, Numer. Algorithms 81 (2019) 1043-1065
(doi:10.1007/s11075-018-0581-4) compare their 6NtCD look-ahead DTZD with an
Adams-Bashforth-type DTZD ("DTZD-AB"); the AB order used there (AB4 according
to the implementer, from the abstract) was NOT verified from the full text.
(Verifier: an unsourced citation "Qiu, Guo, Li & Zhang, IEEE TII 17 (2021)"
was removed because it could not be traced to any source consulted.)  The AB
coefficients themselves are textbook values and are what is tested.

Linear-ZNN analysis at fixed h = gamma*tau
------------------------------------------
x*(t) solves x_dot = f(t, x) exactly (E(x*) = 0), so the AB local error at
x* is C_{q+1} tau^{q+1} x*^{(q+1)}, independent of gamma, and the error obeys
d_{k+1} = d_k - h sum_j beta_j d_{k-j} + O(tau^{q+1}) with characteristic
polynomial  P_h(z) = z^q - z^{q-1} + h sum_j beta_j z^{q-1-j}.  Hence the
steady-state residual is O(tau^{q+1}) (``expected_order = q + 1``) for h in
the real absolute-stability interval of AB_q: (0, 1), (0, 6/11), (0, 3/10).

ASSUMPTIONS: (i) start-up uses the lower-order AB methods (AB1 = Euler, ...)
until q derivative values are stored; (ii) the auxiliary state z is advanced
with forward Euler (ZNN-Bench default) unless ``z_scheme="ab"``.
"""
from __future__ import annotations

from collections import deque
from fractions import Fraction

import numpy as np

from ..core import Discretizer
from .zead import root_condition, znn_stability

Fr = Fraction

AB_BETAS = {
    1: (Fr(1),),
    2: (Fr(3, 2), Fr(-1, 2)),
    3: (Fr(23, 12), Fr(-16, 12), Fr(5, 12)),
    4: (Fr(55, 24), Fr(-59, 24), Fr(37, 24), Fr(-9, 24)),
}

#: real absolute-stability intervals (-h_max, 0) of AB_q (Hairer & Wanner II, Fig. V.1.1)
AB_INTERVALS = {1: Fr(2), 2: Fr(1), 3: Fr(6, 11), 4: Fr(3, 10)}


def ab_order_conditions(beta, M=8):
    """sum_j beta_j (-j)^{m-1} * m  - 1  for m = 1..M (zero up to the order)."""
    out = []
    for m in range(1, M + 1):
        s = sum(Fr(b) * Fr(-j) ** (m - 1) for j, b in enumerate(beta)) * m
        out.append(s - 1)
    return out


def ab_order(beta) -> int:
    """Classical order of the AB method (local error O(tau^{order+1}))."""
    p = 0
    for c in ab_order_conditions(beta, 12):
        if c != 0:
            break
        p += 1
    return p


def ab_char_poly(q: int):
    """rho(z) = z^q - z^{q-1}."""
    return [1.0, -1.0] + [0.0] * (q - 1)


def ab_znn_poly(q: int, h: float):
    """P_h(z) = z^q - z^{q-1} + h sum_j beta_j z^{q-1-j}."""
    poly = np.array(ab_char_poly(q), dtype=float)
    for j, b in enumerate(AB_BETAS[q]):
        poly[1 + j] += h * float(b)
    return poly.tolist()


class AdamsBashforth(Discretizer):
    causal = True

    def __init__(self, q: int, z_scheme: str = "euler"):
        if q not in (2, 3, 4):
            raise ValueError("q must be 2, 3 or 4")
        if z_scheme not in ("euler", "ab"):
            raise ValueError("z_scheme must be 'euler' or 'ab'")
        self.q = q
        self.beta = AB_BETAS[q]
        self.name = f"ab{q}"
        self.expected_order = q + 1
        self.z_scheme = z_scheme
        self.source = ("classical Adams-Bashforth (Hairer, Norsett, Wanner, Solving ODEs I, III.1); ZNN use: "
                       "Qiu, Zhang, Guo, Yang, Li, Numer. Algorithms 81 (2019) 1043-1065")
        self.doi = "10.1007/s11075-018-0581-4"
        self.reset()

    def reset(self):
        self.fx = deque(maxlen=self.q)
        self.fz = deque(maxlen=self.q)

    def step(self, k, t, tau, x, z, f):
        xdot, zdot = f(t, x, z, k, self.implicit)
        self.fx.appendleft(xdot)
        self.fz.appendleft(zdot)
        m = len(self.fx)                    # start-up: AB_m with m < q
        beta = AB_BETAS[m]
        dx = sum(float(b) * v for b, v in zip(beta, self.fx))
        x_next = x + tau * dx
        if self.z_scheme == "ab" and z.shape[1]:
            z_next = z + tau * sum(float(b) * v for b, v in zip(beta, self.fz))
        else:
            z_next = z + tau * zdot
        return x_next, z_next

    # -- analysis ------------------------------------------------------------
    def char_poly(self):
        return ab_char_poly(self.q)

    def root_info(self):
        return root_condition(self.char_poly())

    def znn_poly(self, h):
        return ab_znn_poly(self.q, h)

    def stability(self, h_hi: float = 3.0) -> dict:
        return znn_stability(lambda h: ab_znn_poly(self.q, h), h_hi=h_hi)

    def describe(self):
        st = self.stability()
        ri = self.root_info()
        return {**super().describe(), "family": "adams-bashforth", "steps": self.q,
                "beta": [str(b) for b in self.beta], "source": self.source, "doi": self.doi,
                "zero_stable": ri["zero_stable"], "strong_root_condition": ri["strong"],
                "h_max": st["h_max"], "h_opt": st["h_opt"], "rho_opt": st["rho_opt"],
                "h_range_theory": (0.0, float(AB_INTERVALS[self.q])), "z_scheme": self.z_scheme}


def ab2(z_scheme="euler"):
    """AB2, x_{k+1} = x_k + tau (3 f_k - f_{k-1})/2; residual O(tau^3) at fixed h in (0, 1)."""
    return AdamsBashforth(2, z_scheme)


def ab3(z_scheme="euler"):
    """AB3, x_{k+1} = x_k + tau (23 f_k - 16 f_{k-1} + 5 f_{k-2})/12; O(tau^4), h in (0, 6/11)."""
    return AdamsBashforth(3, z_scheme)


def ab4(z_scheme="euler"):
    """AB4, x_{k+1} = x_k + tau (55 f_k - 59 f_{k-1} + 37 f_{k-2} - 9 f_{k-3})/24; O(tau^5),
    h in (0, 3/10).  The DTZD-AB algorithm of Qiu et al. (2019)."""
    return AdamsBashforth(4, z_scheme)


from ..registry import register_discretization  # noqa: E402

register_discretization("ab2")(ab2)
register_discretization("ab3")(ab3)
register_discretization("ab4")(ab4)

AB_KEYS = {"ab2": 3, "ab3": 4, "ab4": 5}
