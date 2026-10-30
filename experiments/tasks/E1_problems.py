"""E1 helpers used through the runner's import-path hooks (experiment side;
nothing under ``znnbench/`` is modified).

* ``ConstantMatrix`` -- control instance: `linear_system` with A frozen at
  A(t_freeze) while b(t) keeps its time variation.
* ``TimeShift`` / ``TimeShiftLate`` -- run every horizon so that it ENDS at the
  same absolute problem time ``t_end``; together with ``window_metrics`` this
  evaluates the steady-state residual of every tau on the SAME absolute time
  window [t_end - 1.5 s, t_end].
* ``window_metrics`` -- extra metric: max / mean residual over the last
  ``W_ABS`` = 1.5 s of the run (the PLAN's steady window, last 25 % of 6 s).

Why the constant matrix (E1 protocol, experiments/E1_NOTES.md): for linear ZNN
the Jacobian of the continuous vector field is ``-gamma I - A^{-1} A_dot``.  At
fixed h = gamma*tau the per-step perturbation of the error recursion is
``tau ||A^{-1} A_dot||`` (about 3 tau at 1 rad/s), which competes with the small
optimal h of high-order formulas (e.g. h_opt = 0.0255 for ``uhlig_3_4``) and
produces pre-asymptotic slopes.  Freezing A removes that coupling while b(t)
keeps the time variation, so x*(t) = A0^{-1} b(t) still has all derivatives:

    E(x, t) = A0 x - b(t),  J = A0,  dE/dt = -b_dot(t),  x*(t) = A0^{-1} b(t).

Why the time shift: at fixed h the initial transient decays like rho(h)^k per
step (rho ~ 1 - h), i.e. like exp(-h t / tau) in time, so coarse tau needs a
longer horizon T(tau) (configs/E1.py).  If the steady window were "the last
25 % of T", runs of different tau would be evaluated on different time
windows: the maximum of a quasi-periodic error over a longer / shifted window is
systematically different (in the first mini run the tau = 5e-3 point of
uhlig_3_4, T = 80 s, was 2-3x too high and produced local slopes 4.2, 6.1, 4.2).
Ending every run at the same absolute time and using an absolute 1.5 s window
removes this bias: all tau see the same data segment.
"""
from __future__ import annotations

import numpy as np

from znnbench.wrappers import ProblemWrapper

W_ABS = 1.5            # absolute steady window (s): last 25 % of the PLAN's T = 6 s


class ConstantMatrix(ProblemWrapper):
    """Freeze A(t) of a `linear_system` instance at ``t_freeze`` (b(t) unchanged)."""

    exact_residual = False      # the frozen data ARE the exact data of this instance

    def __init__(self, problem, t_freeze: float = 0.0):
        super().__init__(problem, tag="constA")
        if not (hasattr(problem, "A") and hasattr(problem, "b") and hasattr(problem.b, "dot")):
            raise TypeError("ConstantMatrix wraps a linear_system instance (needs .A(t) and .b(t))")
        self.t_freeze = float(t_freeze)
        self.A0 = np.array(problem.A(self.t_freeze), dtype=float)
        self.A0.setflags(write=False)

    def data(self, t):
        return {"t": np.float64(t), "A": self.A0, "b": self.inner.b(t)}

    def E(self, x, d):
        return x @ d["A"].T - d["b"]

    def dEdt(self, x, t):
        return np.broadcast_to(-self.inner.b.dot(t), x.shape).copy()

    def solve_J(self, x, d, r):
        return np.linalg.solve(d["A"], r.T).T

    def JT(self, x, d, v):
        return v @ d["A"]

    # simulate() records residuals with problem.exact_data(t) when it exists;
    # the ProblemWrapper default would return the *inner* (time-varying A)
    # data, so both hooks are overridden to use the frozen instance.
    def exact_data(self, t):
        return self.data(t)

    def residual(self, x, d):
        return np.linalg.norm(self.E(x, d), axis=-1)

    def x_star(self, t):
        return np.linalg.solve(self.A0, self.inner.b(t))

    def x0(self, rng, B):
        return self.inner.x0(rng, B)

    def capabilities(self):
        return set()

    def describe(self):
        d = dict(self.inner.describe())
        d.update({"name": self.name, "wrapper": "ConstantMatrix", "t_freeze": self.t_freeze})
        return d


class TimeShift(ProblemWrapper):
    """Model time t in [0, T] is problem time t + s with s = t_end - T, so every
    run ends at problem time ``t_end`` whatever its horizon T (= inner.T).

    Only the data clock is shifted: E, J, the residual and x*(t) all use the
    shifted data, the model and discretizer see model time.  Valid for problems
    whose data are defined for problem times in [t_end - T, t_end] (all the
    sinusoidal instances for t_end = 6; `nonlinear_equations` guo2018 needs
    t > -1 and uses ``TimeShiftLate``)."""

    exact_residual = False
    t_end_default = 6.0

    def __init__(self, problem, t_end=None):
        super().__init__(problem, tag="tshift")
        self.t_end = float(self.t_end_default if t_end is None else t_end)
        self.shift = self.t_end - float(problem.T)
        inner = problem
        if callable(getattr(inner, "solution_error", None)):
            self.solution_error = lambda x, t: inner.solution_error(x, t + self.shift)

    def data(self, t):
        return self.inner.data(t + self.shift)

    def dEdt(self, x, t):
        return self.inner.dEdt(x, t + self.shift)

    def x_star(self, t):
        return self.inner.x_star(t + self.shift)

    def exact_data(self, t):
        inner = self.inner
        ed = getattr(inner, "exact_data", None)
        return ed(t + self.shift) if callable(ed) else inner.data(t + self.shift)

    def residual(self, x, d):
        return self.inner.residual(x, d)

    def describe(self):
        d = dict(self.inner.describe())
        d.update({"name": self.name, "time_shift": self.shift, "t_end": self.t_end})
        return d


class TimeShiftLate(TimeShift):
    """TimeShift with t_end = 200 s (for problems undefined at negative times)."""

    t_end_default = 200.0


def window_metrics(sim):
    """Max / mean residual over the absolute window [t_N - W_ABS, t_N] (inf for
    diverged trajectories), the number of recorded samples in it, and a
    transient-leak diagnostic.

    ``x0_spread_w`` (review addition): all trajectories of one gain value see
    the same data, so once the initial transient has decayed their residual
    traces coincide up to round-off whatever x0 was.  For each trajectory it is
    max_{t in W} |r_b(t) - r_ref(t)|, with r_ref the pointwise median trace of
    the converged trajectories that share its gain (absolute units, same as
    ``mssre_w``).  A spread that is large against ``mssre_w`` and against the
    round-off floor means the horizon was too short for that (formula, tau,
    instance): the analysis then excludes the point ("unsettled").  The
    frozen-coefficient decay rate rho(h) used to size the horizon does not see
    the tau*||A^{-1} A_dot|| coupling, which can slow the decay (found for
    uhlig_4_6 at tau = 1e-2 on linsys_n4 seed 0: MSSRE 2x too high at T = 6 s)."""
    t = np.asarray(sim["t"], dtype=float)
    r = np.asarray(sim["residual"], dtype=float)
    m = t >= t[-1] - W_ABS - 1e-9 * max(1.0, abs(t[-1]))
    rw = r[m]
    bad = ~np.all(np.isfinite(rw), axis=0)
    div = np.asarray(sim.get("diverged", np.zeros(r.shape[1], bool)), dtype=bool)
    with np.errstate(all="ignore"):
        mx = np.where(bad | div, np.inf, np.max(np.where(np.isfinite(rw), rw, 0.0), axis=0))
        mean = np.where(bad | div, np.inf, np.mean(np.where(np.isfinite(rw), rw, 0.0), axis=0))
    spread = np.full(r.shape[1], np.nan)
    gam = np.asarray(sim.get("gamma", np.zeros(r.shape[1])), dtype=float).reshape(-1)
    if gam.size != r.shape[1]:
        gam = np.zeros(r.shape[1])
    good = ~(bad | div)
    for g in np.unique(np.round(gam, 12)):
        cols = np.nonzero(np.isclose(gam, g, rtol=1e-12, atol=0.0))[0]
        ok = cols[good[cols]]
        if len(ok) >= 2:
            ref = np.median(rw[:, ok], axis=1)
            spread[ok] = np.max(np.abs(rw[:, ok] - ref[:, None]), axis=0)
    spread[~good] = np.inf
    return {"mssre_w": mx, "mean_w": mean, "w_points": np.full(r.shape[1], float(m.sum())),
            "x0_spread_w": spread}
