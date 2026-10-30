"""E2 theory helpers: linear-ZNN error recursion of a discretization at fixed h.

Used by experiments/tasks/E2_stability.py (reference step h_opt of the formula
floor) and experiments/analyze_E2.py (theoretical intervals, frozen-time and
finite-horizon predictions).  Computed independently of znnbench's own
``zead.znn_stability`` (which is only used as a cross-check in the analysis).

Conventions (docs/spec/discretizations.md sec. 1, docs/spec/runner.md sec. 1):
h = gamma*tau always; look-ahead formulas in tau-form,
    tau x_dot_k ~ c_{-1} x_{k+1} + c_0 x_k + c_1 x_{k-1} + ... + c_{s-1} x_{k-s+1},
so linear ZNN (dE/dt = -gamma E with exact dE/dt) gives the error recursion
    c_{-1} e_{k+1} + (c_0 + h) e_k + c_1 e_{k-1} + ... = O(tau^{p+1})
with characteristic polynomial
    P_h(xi) = c_{-1} xi^s + (c_0 + h) xi^{s-1} + c_1 xi^{s-2} + ... + c_{s-1}.
A paper that writes x_{k+1} = ... + (tau/c_{-1}) x_dot_k and calls
gamma*tau/c_{-1} its step size (Sun-Tian-Wang 2019) has h_paper = h / c_{-1}.
Adams-Bashforth q: P_h = xi^q - xi^{q-1} + h sum_j beta_j xi^{q-1-j}.
RK4: amplification R(-h) = 1 - h + h^2/2 - h^3/6 + h^4/24.
implicit_euler_err (resolvent of the linear activation): e_{k+1} = e_k / (1 + h).

Time-varying problems: for linear ZNN on A(t)x = b(t) or A(t)X = I the exact
error recursion is sum_i c_i e_{k+1-i} = -(h I + tau M(t_k)) e_k + O(tau^{p+1}),
M = A^{-1} A_dot, i.e. the effective step of each eigen-mode is h + tau mu.

Inverse-free schemes (docs/spec/inverse_free_dtznn.md):

* ``cdtznn_mahuang2025`` (x-recursion = zead_5i_mahuang2025, J^-1 -> M_k, M_k from
  the same look-ahead formula applied to dM/dt = -M J_dot M - mu M (J M - I),
  mu = gamma).  On problems that are affine in x (both E2 problems: J = A(t)
  does not depend on x) the linearisation about (x*, J^-1) is BLOCK-TRIANGULAR:
  the M-error Delta = M - J^-1 obeys
      sum_i c_i Delta_{k+1-i} = -h Delta_k - tau (J^-1 J_dot Delta_k + Delta_k J_dot J^-1) + O(|Delta|^2)
  (exact up to the local truncation error and the quadratic terms), and it
  forces the x-error only additively (tau Delta_k dE/dt).  Both diagonal blocks
  therefore have the SAME polynomial P_h, so the local (linearised) root-locus
  interval of CDTZNN is that of its x-recursion formula (h_max = 0.1725), and
  the frozen-time effective steps are h + tau mu_i (x-block) and
  h + tau (mu_i + mu_j) (M-block: the Sylvester operator doubles the shift).
  This is a LOCAL theory (``theory_type = 'local'``): it says nothing about the
  basin of the quadratic M-recursion from the Newton-Schulz start.
* ``broyden_dtznn_zhang2012``: no theory.  The secant matrix never converges to
  J^-1 (||J M - I||_F = 1.5-2.6 at steady state, spec erratum E4), so the error
  recursion e_{k+1} = (I - h J M_k) e_k + ... has a trajectory-dependent,
  chaotic coefficient (spec E6: a 1e-14 perturbation of x0 changes the residual
  by 5x); there is no closed-form root locus.
"""
from __future__ import annotations

import json
from fractions import Fraction

import numpy as np


def batched_radius(polys):
    """Spectral radius of many polynomials (rows, descending powers, complex ok)
    via batched companion-matrix eigenvalues."""
    P = np.atleast_2d(np.asarray(polys, dtype=complex))
    Q = P[:, 1:] / P[:, :1]
    s = Q.shape[1]
    if s == 0:
        return np.zeros(P.shape[0])
    C = np.zeros((P.shape[0], s, s), dtype=complex)
    C[:, 0, :] = -Q
    if s > 1:
        idx = np.arange(s - 1)
        C[:, idx + 1, idx] = 1.0
    return np.max(np.abs(np.linalg.eigvals(C)), axis=1)


class Theory:
    """Linear-ZNN error recursion of one formula.  ``kind`` is one of
    'lookahead', 'ab', 'rk4', 'implicit_euler' or None (no theory available)."""

    def __init__(self, formula: str, desc: dict, implicit: bool = False):
        self.formula = formula
        self.desc = desc
        self.kind, self.c, self.beta = None, None, None
        # 'exact' = linear error recursion with exact J^-1; 'local' = linearisation of an
        # inverse-free scheme about (x*, J^-1); None = no closed-form theory (see reason)
        self.theory_type, self.reason, self.mblock = None, "", False
        name = str(desc.get("name", formula))
        if desc.get("inverse_free"):
            if ("coeffs" in desc and str(desc.get("mu", "")) == "gamma"
                    and desc.get("x_recursion_key")):
                # CDTZNN: block-triangular linearisation, both blocks have P_h of the
                # x-recursion formula (module docstring); the M-block has the doubled
                # frozen-time shift tau (mu_i + mu_j)
                self.kind = "lookahead"
                self.c = np.array([float(Fraction(str(v))) for v in desc["coeffs"]])
                self.theory_type, self.mblock = "local", True
                self.reason = (f"local linearisation about (x*, J^-1): x- and M-blocks share P_h of "
                               f"{desc.get('x_recursion_key')}; M-block frozen-time shift tau(mu_i+mu_j)")
            else:
                # Broyden: the secant matrix never converges to J^-1 -> trajectory-dependent,
                # chaotic error recursion; no closed-form root locus
                self.kind = None
                self.reason = ("inverse-free secant scheme: M_k does not converge to J^-1 "
                               "(||JM-I||_F ~ 1.5-2.6), recursion I - h J M_k is trajectory dependent")
        elif implicit and "coeffs" in desc and [float(v) for v in desc["coeffs"]] == [1.0, -1.0]:
            self.kind = "implicit_euler"
        elif implicit:
            self.kind = None                      # other implicit schemes: no closed form here
            self.reason = "implicit scheme without a closed-form error map here"
        elif desc.get("family") == "adams-bashforth" and "beta" in desc:
            self.kind = "ab"
            self.beta = np.array([float(Fraction(str(b))) for b in desc["beta"]])
        elif "coeffs" in desc:
            self.kind = "lookahead"
            self.c = np.array([float(v) for v in desc["coeffs"]])
        elif name.startswith("rk4") or formula == "rk4":
            self.kind = "rk4"
        if self.kind is not None and self.theory_type is None:
            self.theory_type = "exact"
        elif self.kind is None and not self.reason:
            self.reason = "no closed-form error recursion"

    @classmethod
    def from_disc(cls, formula, disc):
        desc = json.loads(json.dumps(disc.describe(), default=lambda o: o.tolist()
                                     if hasattr(o, "tolist") else str(o)))
        return cls(formula, desc, implicit=bool(getattr(disc, "implicit", False)))

    @property
    def c_lead(self) -> float:
        return float(self.c[0]) if self.kind == "lookahead" else 1.0

    def polys(self, heff):
        heff = np.atleast_1d(np.asarray(heff, dtype=complex))
        if self.kind == "lookahead":
            P = np.tile(self.c.astype(complex), (len(heff), 1))
            P[:, 1] += heff
            return P
        if self.kind == "ab":
            q = len(self.beta)
            base = np.zeros(q + 1, dtype=complex)
            base[0], base[1] = 1.0, -1.0
            P = np.tile(base, (len(heff), 1))
            P[:, 1:] += heff[:, None] * self.beta[None, :]
            return P
        raise ValueError(f"no polynomial for kind {self.kind}")

    def radius(self, heff):
        """Spectral radius of the error recursion at (complex) effective steps."""
        heff = np.atleast_1d(np.asarray(heff, dtype=complex))
        if self.kind == "rk4":
            z = -heff
            return np.abs(1 + z + z ** 2 / 2 + z ** 3 / 6 + z ** 4 / 24)
        if self.kind == "implicit_euler":
            return np.abs(1.0 / (1.0 + heff))
        if self.kind in ("lookahead", "ab"):
            return batched_radius(self.polys(heff))
        return np.full(heff.shape, np.nan)

    # ------------------------------------------------------------------
    def h_max(self, hi=4.0, n=4000):
        """First h > 0 at which the spectral radius reaches 1 (inf if none in (0, hi])."""
        if self.kind is None:
            return np.nan
        if self.kind == "implicit_euler":
            return np.inf
        hs = np.linspace(hi / n, hi, n)
        rad = self.radius(hs)
        bad = np.where(rad >= 1.0 - 1e-12)[0]
        if len(bad) == 0:
            return np.inf
        i = int(bad[0])
        if i == 0:
            return 0.0
        lo, hh = hs[i - 1], hs[i]
        for _ in range(60):
            m = 0.5 * (lo + hh)
            if self.radius(m)[0] < 1.0 - 1e-12:
                lo = m
            else:
                hh = m
        return float(0.5 * (lo + hh))

    def h_opt(self, hmax=None):
        """argmin_h of the spectral radius on (0, h_max) and the radius there."""
        if self.kind in (None, "implicit_euler"):
            return np.nan, np.nan
        hmax = self.h_max() if hmax is None else hmax
        if not np.isfinite(hmax) or hmax <= 0:
            return np.nan, np.nan
        hs = np.linspace(hmax / 4000, hmax * (1 - 1e-9), 4000)
        rad = self.radius(hs)
        j = int(np.argmin(rad))
        a, b = hs[max(j - 1, 0)], hs[min(j + 1, len(hs) - 1)]
        for _ in range(80):
            m1, m2 = a + (b - a) / 3, b - (b - a) / 3
            r1, r2 = self.radius([m1, m2])
            if r1 < r2:
                b = m2
            else:
                a = m1
        h = 0.5 * (a + b)
        return float(h), float(self.radius(h)[0])

    def effective_mus(self, mus_t):
        """Frozen-time shifts of the linearised recursion from the eigenvalues of
        A^-1 A_dot on a time grid (array (n_t, n), or flat): mu_i for exact-inverse
        formulas; mu_i and mu_i + mu_j (same instant) for the CDTZNN M-block."""
        m = np.asarray(mus_t, dtype=complex)
        if not self.mblock:
            return m.ravel()
        if m.ndim != 2:
            raise ValueError("the M-block needs the eigenvalues per instant: problem_mus(..., by_time=True)")
        flat = m.ravel()
        pair = (m[:, :, None] + m[:, None, :]).reshape(-1)
        return np.concatenate([flat, pair])

    def h_frozen_time(self, mus, tau, h_ref, hmax, n=200):
        """Frozen-time limit: upper end of the h-interval containing h_ref on which
        max_mu rho(h + tau mu) < 1 (mu = eigenvalues of A^{-1} A_dot over [0, T];
        pass ``effective_mus(...)`` for the inverse-free M-block)."""
        if self.kind is None:
            return np.nan
        if self.kind == "implicit_euler":
            return np.inf
        mus = np.asarray(mus, dtype=complex).ravel()
        top = (hmax if np.isfinite(hmax) else 3.0) * 1.6 + 0.05
        hs = np.linspace(top / n, top, n)
        R = self.radius((hs[:, None] + tau * mus[None, :]).ravel()).reshape(n, -1).max(axis=1)
        ok = R < 1.0
        if not ok.any():
            return 0.0
        jref = int(np.argmin(np.abs(hs - h_ref))) if np.isfinite(h_ref) else int(np.argmin(R))
        if not ok[jref]:
            cand = np.where(ok)[0]
            jref = int(cand[np.argmin(np.abs(cand - jref))])
        u = jref
        while u + 1 < n and ok[u + 1]:
            u += 1
        if u == n - 1:
            return float(top)

        def worst(h):
            return float(np.max(self.radius(h + tau * mus)))
        lo, hh = hs[u], hs[u + 1]
        for _ in range(40):
            m = 0.5 * (lo + hh)
            if worst(m) < 1.0:
                lo = m
            else:
                hh = m
        return float(0.5 * (lo + hh))

    def h_finite_horizon(self, rho_crit, hmax, hopt):
        """Largest h in [h_opt, h_max] with rho(P_h) <= rho_crit (nan if none)."""
        if self.kind in (None, "implicit_euler") or not np.isfinite(hmax) or not np.isfinite(rho_crit):
            return np.nan
        lo = hopt if np.isfinite(hopt) else 0.5 * hmax
        hs = np.linspace(lo, hmax, 400)
        rad = self.radius(hs)
        okk = rad <= rho_crit
        if not okk[0]:
            return np.nan
        i = np.where(~okk)[0]
        if len(i) == 0:
            return float(hmax)
        a, b = hs[i[0] - 1], hs[i[0]]
        for _ in range(50):
            m = 0.5 * (a + b)
            if self.radius(m)[0] <= rho_crit:
                a = m
            else:
                b = m
        return float(0.5 * (a + b))


def problem_mus(problem_key, seed, T, freeze_A=False, n_t=None, kwargs=None, by_time=False):
    """Eigenvalues of M(t) = A(t)^{-1} A_dot(t) on a time grid of [0, T] (zero for frozen A).
    ``by_time=True`` returns an array (n_t, n) (needed for the pairwise sums of the
    CDTZNN M-block); otherwise the flat concatenation.  Default grid: 10 points per
    second (61 points for T = 6)."""
    n_t = int(n_t or max(61, int(round(10 * T)) + 1))
    if freeze_A:
        z = np.zeros((1, 1), dtype=complex)
        return z if by_time else z.ravel()
    from znnbench import make_problem
    kw = dict(n=4, kappa=10.0, freq=1.0, T=T)
    kw.update(kwargs or {})
    kw.pop("seed", None)
    p = make_problem(problem_key, seed=int(seed), **kw)
    A = p.A
    m = np.stack([np.linalg.eigvals(np.linalg.solve(A(t), A.dot(t))) for t in np.linspace(0.0, T, n_t)])
    return m if by_time else m.ravel()
