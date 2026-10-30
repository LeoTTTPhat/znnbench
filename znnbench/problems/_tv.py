"""Smooth time-varying matrix and vector generators with exact derivatives.

`TVMatrix` builds A(t) = U(t) S(t) V(t)^T where U, V are smooth orthogonal
matrices (fixed random basis times block-diagonal rotations with constant
angular rates) and S(t) holds singular values in [1, kappa] with gentle
periodic modulation.  The condition number therefore stays close to the
requested `kappa` for all t, and A_dot(t) is available in closed form.
"""
from __future__ import annotations

import numpy as np


def _random_orthogonal(n, rng):
    q, r = np.linalg.qr(rng.standard_normal((n, n)))
    return q * np.sign(np.diag(r))


class SmoothOrthogonal:
    """Q(t) = P R(t) P^T with R(t) block-diagonal 2x2 rotations."""

    def __init__(self, n, rng, freq=1.0):
        self.n = n
        self.P = _random_orthogonal(n, rng)
        self.w = freq * rng.uniform(0.5, 1.5, size=n // 2)

    def _R(self, t, deriv=False):
        """Block-diagonal rotation R(t) (or R_dot(t)), assembled without a Python
        loop over the n/2 blocks (vectorised by the integrator, see
        CORE_CHANGE_REQUESTS.md, problems_matrix; same values as the former
        per-block loop)."""
        n = self.n
        R = np.zeros((n, n)) if deriv else np.eye(n)
        if len(self.w) == 0:
            return R
        w = self.w
        c, s = np.cos(w * t), np.sin(w * t)
        i = 2 * np.arange(len(w))
        if deriv:
            R[i, i], R[i, i + 1], R[i + 1, i], R[i + 1, i + 1] = w * -s, w * -c, w * c, w * -s
        else:
            R[i, i], R[i, i + 1], R[i + 1, i], R[i + 1, i + 1] = c, -s, s, c
        return R

    def __call__(self, t):
        return self.P @ self._R(t) @ self.P.T

    def dot(self, t):
        return self.P @ self._R(t, deriv=True) @ self.P.T


class TVMatrix:
    """Smooth A(t) (n x n_cols) with controlled condition number."""

    def __init__(self, n, rng, kappa=10.0, freq=1.0, mod=0.2, n_cols=None, symmetric=False,
                 spd=False):
        self.n = n
        self.nc = n if n_cols is None else n_cols
        k = min(self.n, self.nc)
        self.U = SmoothOrthogonal(self.n, rng, freq)
        self.V = self.U if (symmetric or spd) else SmoothOrthogonal(self.nc, rng, freq)
        self.s0 = np.geomspace(kappa, 1.0, k) if k > 1 else np.array([1.0])
        self.ws = freq * rng.uniform(0.5, 1.5, size=k)
        self.ph = rng.uniform(0, 2 * np.pi, size=k)
        self.mod = mod
        self.k = k

    def _S(self, t, deriv=False):
        arg = self.ws * t + self.ph
        if deriv:
            s = self.s0 * self.mod * self.ws * np.cos(arg)
        else:
            s = self.s0 * (1.0 + self.mod * np.sin(arg))
        S = np.zeros((self.n, self.nc))
        S[np.arange(self.k), np.arange(self.k)] = s
        return S

    def __call__(self, t):
        return self.U(t) @ self._S(t) @ self.V(t).T

    def dot(self, t):
        U, Ud = self.U(t), self.U.dot(t)
        V, Vd = self.V(t), self.V.dot(t)
        S, Sd = self._S(t), self._S(t, deriv=True)
        return Ud @ S @ V.T + U @ Sd @ V.T + U @ S @ Vd.T


class TVVector:
    """b(t) = a * sin(w t + phi) element-wise, with exact derivative."""

    def __init__(self, n, rng, freq=1.0, amp=1.0):
        self.a = amp * rng.uniform(0.5, 1.5, size=n)
        self.w = freq * rng.uniform(0.5, 1.5, size=n)
        self.ph = rng.uniform(0, 2 * np.pi, size=n)

    def __call__(self, t):
        return self.a * np.sin(self.w * t + self.ph)

    def dot(self, t):
        return self.a * self.w * np.cos(self.w * t + self.ph)
