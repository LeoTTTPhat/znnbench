"""Shared settings and the joint-statistics wrapper of the robotics case study R.

Used by ``experiments/configs/R.py`` (sweep definition), the custom task
``experiments/tasks/R_joint_traces.py`` and ``experiments/analyze_R.py``.
Protocol, rationale and every assumption: ``experiments/R_NOTES.md``.

The robot problem (``docs/spec/problems_robot.md``) is *position-only,
unconstrained, velocity-level pseudoinverse control*: theta_dot = J^+ (r_d_dot
- gamma Phi(f(theta) - r_d)).  Joint position, velocity and acceleration
limits are NOT enforced by any model; this module only *measures* how far the
commanded motion is from the actuator limits.

``JointStats`` is a transparent ``Discretizer`` wrapper (runner wrapper key
``experiments.tasks.R_common:JointStats``, target ``disc``).  It sees every
joint state x_k = theta(t_k) and reports per trajectory, through the runner's
``extra_metrics(sim)`` hook:

  qd_peak        max_{k,i} |theta_{k+1,i} - theta_{k,i}| / tau  (rad/s) -- the
                 commanded joint speed; equals simulate's online ``peak_rate``
                 (checked in the mini run)
  qd_util        max_{k,i} |qd_{k,i}| / qd_max_i  (velocity-limit utilisation,
                 > 1 = the command exceeds the limit)
  qd_util_ss     the same over the steady window (last ``ss_frac`` of the run)
  qd_viol_frac   fraction of control steps with any |qd_{k,i}| > qd_max_i
  qdd_peak       max_{k,i} |qd_{k,i} - qd_{k-1,i}| / tau (rad/s^2), the joint
                 acceleration implied by consecutive velocity commands
  qdd_peak_ss    the same over the steady window
  qdd_util       qdd_peak relative to the per-joint acceleration limit (Panda
                 only; NaN for the PUMA 560, no documented limit)
  alt_active     sign alternations of Delta theta over the steady window,
                 counted only for joints that move in both steps (|Delta
                 theta_i| > rel_tol * max_j |Delta theta_j| and > 1e-15 rad) and
                 normalised by the number of such joint-steps, so it lies in
                 [0, 1].  simulate's ``rate_alt`` divides by ALL joints; the last
                 joint, whose position-Jacobian column vanishes analytically,
                 gets increments ~1e-16 |J^+ r| that are zero or numerical noise,
                 so ``rate_alt`` is capped at (n-1)/n for clean runs and can count
                 noise alternations of that joint in violent runs
  qlim_viol      max over the run of the joint-position-limit violation (rad)

Diverged trajectories (non-finite or |theta| > 1e10, as in simulate) get NaN
statistics and qd_peak = inf.
"""
from __future__ import annotations

import numpy as np

from znnbench.core import Discretizer

# ---------------------------------------------------------------------------
# protocol constants
# ---------------------------------------------------------------------------
T_HORIZON = 10.0                       # s (path period = horizon, closed paths)
TAUS = [1e-2, 3e-3, 1e-3]              # control periods: 100 Hz, 333 Hz, 1 kHz
GAMMAS = [10.0, 100.0]                 # fixed design gains (fixed-gamma mode)
X0_SPREAD = 0.05                       # rad, U[-s, s] perturbation of theta0 (off-path starts)
NOISE_LEVEL = 0.05                     # m/s per Cartesian axis, constant noise in J theta_dot = ... + nu
ROBOTS = ["robot_puma560", "robot_panda"]
PATHS = ["circle", "lissajous"]

# Franka Emika Robot (FER / Panda) joint limits, Franka Control Interface docs,
# "Robot and interface specifications" (frankarobotics.github.io/docs/
# robot_specifications.html, table "Franka Emika Robot (FER)", re-read 2026-10-18):
PANDA_QD_MAX = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100])     # rad/s
PANDA_QDD_MAX = np.array([15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0])                    # rad/s^2
PANDA_QDDD_MAX = np.array([7500., 3750., 5000., 6250., 7500., 10000., 10000.])         # rad/s^3
# PUMA 560: no documented per-joint velocity limit was found (Corke's MATLAB and
# Python toolboxes define none; ARTE's puma560 parameter file marks the field
# "not available").  ASSUMPTION: a nominal 2 rad/s for every joint, the order of
# magnitude used as a "typical" industrial-arm limit (see R_NOTES.md).
PUMA_QD_MAX = np.full(6, 2.0)
PUMA_QDD_MAX = None


def joint_limits(robot):
    """(qd_max, qdd_max) arrays for a robot name ('puma560'/'panda' or the
    registry key); qdd_max is None when no documented limit exists."""
    r = str(robot).replace("robot_", "")
    if r == "panda":
        return PANDA_QD_MAX, PANDA_QDD_MAX
    if r == "puma560":
        return PUMA_QD_MAX, PUMA_QDD_MAX
    raise KeyError(robot)


# ---------------------------------------------------------------------------
# model entries (model, discretization) of the case study
# ---------------------------------------------------------------------------
# entry key -> (model component, discretization key, display name)
SPTC_KW = {"t_c": 1.0, "p": 0.5, "noise_bound": NOISE_LEVEL}   # zeta = 5*Delta = 0.25 m/s
MODEL_ENTRIES = {
    "znn_lin": ({"key": "znn", "kwargs": {"activation": "linear"}, "label": "znn_lin"},
                "euler", "ZNN (linear)"),
    "znn_sbp": ({"key": "znn", "kwargs": {"activation": "sign_bi_power", "r": 0.5}, "label": "znn_sbp"},
                "euler", "ZNN-SBP (explicit Euler)"),
    "znn_sbp_ie": ({"key": "znn", "kwargs": {"activation": "sign_bi_power", "r": 0.5}, "label": "znn_sbp"},
                   "implicit_euler_err", "ZNN-SBP (implicit Euler)"),
    "sptc_nt_znn": ({"key": "sptc_nt_znn", "kwargs": dict(SPTC_KW), "label": "sptc_nt_znn"},
                    "euler", "SPTC-NT-ZNN"),
    "ieznn": ({"key": "ieznn", "kwargs": {}, "label": "ieznn"}, "euler", "IEZNN"),
}
# fallback predefined-time model when sptc_nt_znn is not registered
PVAF1_ENTRY = ({"key": "znn", "kwargs": {"activation": "pvaf1", "c2": 0.0}, "label": "znn_pvaf1"},
               "euler", "ZNN-PVAF1")
DISPLAY_ORDER = ["znn_lin", "znn_sbp", "znn_sbp_ie", "sptc_nt_znn", "znn_pvaf1", "ieznn"]


def available_entries():
    """Model entries whose registry keys exist now (conditional at build time)."""
    import znnbench
    from znnbench import activations
    av = znnbench.available()
    acts = set(activations.available())
    out = {}
    for k, (comp, disc, name) in MODEL_ENTRIES.items():
        act = comp["kwargs"].get("activation")
        if comp["key"] in av["models"] and disc in av["discretizations"] and (act is None or act in acts):
            out[k] = (comp, disc, name)
    if "sptc_nt_znn" not in out and "pvaf1" in acts:
        out["znn_pvaf1"] = PVAF1_ENTRY
    return out


def entry_of(model_label, disc_key):
    """Entry key of an aggregated row (model label + discretization key)."""
    if model_label == "znn_sbp":
        return "znn_sbp_ie" if disc_key == "implicit_euler_err" else "znn_sbp"
    return model_label


# ---------------------------------------------------------------------------
# error-unit sensitivity arm (R_units)
# ---------------------------------------------------------------------------
# Power-type activations are NOT invariant to the unit in which E is expressed:
# with E in units of 1/s metres, Phi_s(E) = s^-1 Phi(s E), e.g. for
# sign_bi_power (r = 0.5) the |e|^0.5 term is scaled by s^-0.5 and the |e|^2
# term by s.  Linear ZNN and IEZNN are exactly invariant.  R_units re-runs the
# unit-dependent entries with the error expressed in millimetres (s = 1000);
# linear ZNN is included as an invariance control (must reproduce R_off to
# round-off).  The SPTC-NT-ZNN noise bound is converted with the unit
# (Delta = 0.05 m/s = 50 mm/s), so its sign term is the same PHYSICAL design;
# only its |E|^p term changes.
UNIT_SCALE = 1000.0                    # metres -> millimetres
UNITS_ENTRIES = ["znn_lin", "znn_sbp", "znn_sbp_ie", "sptc_nt_znn", "znn_pvaf1"]


def units_entries():
    """Model entries of the R_units arm (available ones), with unit-converted
    problem-unit parameters."""
    ent = {k: v for k, v in available_entries().items() if k in UNITS_ENTRIES}
    if "sptc_nt_znn" in ent:
        comp, disc, name = ent["sptc_nt_znn"]
        kw = dict(comp["kwargs"])
        kw["noise_bound"] = NOISE_LEVEL * UNIT_SCALE
        ent["sptc_nt_znn"] = ({**comp, "kwargs": kw}, disc, name)
    return ent


class ScaledError:
    """Problem wrapper (runner key ``experiments.tasks.R_common:ScaledError``,
    target ``problem``) that expresses the error in units of 1/``scale`` of the
    problem unit:  E_s = s E,  dE_s/dt = s dE/dt,  J_s = s J  (so
    solve_J_s(r) = solve_J(r / s) and JT_s(v) = s JT(v)).  The trajectory of a
    model that is linear in E is unchanged; power-type activations change.
    The RECORDED residual (``residual``) stays in the original unit (metres),
    so every metric remains comparable with the unwrapped runs.  Additive
    design-formula noise passes through ``solve_J`` and would therefore be
    read in scaled units: R_units is noise-free.  Everything else is delegated
    to the wrapped problem."""

    target = "problem"

    def __init__(self, inner, scale=UNIT_SCALE):
        self.inner = inner
        self.scale = float(scale)
        self.name = f"{getattr(inner, 'name', 'problem')}_E_x{self.scale:g}"

    def __getattr__(self, item):
        if item == "inner":                       # not yet set (e.g. during unpickling)
            raise AttributeError(item)
        return getattr(self.inner, item)

    def E(self, x, d):
        return self.scale * self.inner.E(x, d)

    def dEdt(self, x, t):
        return self.scale * self.inner.dEdt(x, t)

    def solve_J(self, x, d, r):
        return self.inner.solve_J(x, d, np.asarray(r) / self.scale)

    def JT(self, x, d, v):
        return self.scale * self.inner.JT(x, d, v)

    def residual(self, x, d):
        return self.inner.residual(x, d)

    def describe(self):
        base = self.inner.describe() if callable(getattr(self.inner, "describe", None)) else {}
        return {**base, "wrapper": "R_common.ScaledError", "error_scale": self.scale}


# ---------------------------------------------------------------------------
# joint-statistics wrapper
# ---------------------------------------------------------------------------
class JointStats(Discretizer):
    """Transparent discretizer wrapper that measures commanded joint motion.

    Parameters
    ----------
    inner : the wrapped discretizer (its ``step`` is called unchanged).
    ss_frac : steady window = last ``ss_frac`` of the horizon, defined exactly
        as simulate's online statistics (steps k >= floor((1 - ss_frac) N)).
    rel_tol : a joint "moves" in a step if |Delta theta_i| > rel_tol *
        max_j |Delta theta_j| (and > 1e-15 rad); used by ``alt_active``.
    record : keep the per-step joint velocities (N, B, n) in ``self.qd_trace``
        (used only by the custom trace task; off in sweeps).
    """

    target = "disc"

    def __init__(self, inner, ss_frac=0.25, rel_tol=1e-6, record=False):
        self.inner = inner
        self.name = getattr(inner, "name", "disc")
        self.causal = getattr(inner, "causal", True)
        self.expected_order = getattr(inner, "expected_order", None)
        self.implicit = getattr(inner, "implicit", False)
        self.ss_frac = float(ss_frac)
        self.rel_tol = float(rel_tol)
        self.record = bool(record)
        self.problem = None
        self._B = None
        self.reset()

    # -- environment -----------------------------------------------------------
    def bind_env(self, problem=None, **kw):
        self.problem = problem
        if callable(getattr(self.inner, "bind_env", None)):
            self.inner.bind_env(problem=problem, **kw)

    def reset(self):
        self.inner.reset()
        self._B = None
        self.qd_trace = []

    def _init(self, B, n, tau):
        robot = getattr(self.problem, "robot", None)
        try:
            qd_max, qdd_max = joint_limits(robot)
        except KeyError:
            qd_max, qdd_max = np.full(n, np.inf), None
        self.qd_max = np.asarray(qd_max, float)[:n]
        self.qdd_max = None if qdd_max is None else np.asarray(qdd_max, float)[:n]
        arm = getattr(self.problem, "arm", None)
        self.qlim = None if arm is None else np.asarray(arm.qlim, float)
        T = float(getattr(self.problem, "T", np.nan))
        N = int(round(T / tau)) if np.isfinite(T) else 10 ** 12
        self.k_ss = int(np.floor((1.0 - self.ss_frac) * N))
        self._B, self._n = B, n
        self.alive = np.ones(B, bool)
        self.qd_pk = np.zeros((B, n))
        self.qd_pk_ss = np.zeros((B, n))
        self.qdd_pk = np.zeros((B, n))
        self.qdd_pk_ss = np.zeros((B, n))
        self.viol_steps = np.zeros(B)
        self.steps = 0
        self.alt_cnt = np.zeros(B)
        self.alt_tot = np.zeros(B)
        self.qlim_v = np.zeros(B)
        self.prev_dx = None
        self.prev_qd = None

    # -- stepping ----------------------------------------------------------------
    def step(self, k, t, tau, x, z, f):
        x_next, z_next = self.inner.step(k, t, tau, x, z, f)
        B, n = x.shape
        if self._B != B:
            self._init(B, n, tau)
            if self.qlim is not None and self.qlim.shape[0] == n:
                self.qlim_v = np.maximum(self.qlim_v, self._viol(x))
        with np.errstate(invalid="ignore", over="ignore"):
            bad = ~np.isfinite(x_next).all(axis=1) | (np.abs(x_next).max(axis=1) > 1e10)
            self.alive &= ~bad
            a = self.alive[:, None]
            dx = np.where(a, x_next - x, 0.0)
            qd = dx / tau
            aqd = np.abs(qd)
            self.qd_pk = np.maximum(self.qd_pk, aqd)
            self.viol_steps += (aqd > self.qd_max).any(axis=1)
            self.steps += 1
            ss = k >= self.k_ss
            if ss:
                self.qd_pk_ss = np.maximum(self.qd_pk_ss, aqd)
            if self.prev_qd is not None:
                aqdd = np.abs(qd - self.prev_qd) / tau
                aqdd = np.where(a, aqdd, 0.0)
                self.qdd_pk = np.maximum(self.qdd_pk, aqdd)
                if ss:
                    self.qdd_pk_ss = np.maximum(self.qdd_pk_ss, aqdd)
            if ss and self.prev_dx is not None:
                moving = self._moving(dx) & self._moving(self.prev_dx)
                self.alt_cnt += ((np.sign(dx) * np.sign(self.prev_dx) < 0) & moving).sum(axis=1)
                self.alt_tot += moving.sum(axis=1)
            if self.qlim is not None and self.qlim.shape[0] == n:
                self.qlim_v = np.maximum(self.qlim_v, np.where(self.alive, self._viol(x_next), 0.0))
        self.prev_dx = dx
        self.prev_qd = qd
        if self.record:
            self.qd_trace.append(qd.astype(np.float32))
        return x_next, z_next

    def _moving(self, dx):
        a = np.abs(dx)
        return (a > self.rel_tol * a.max(axis=1, keepdims=True)) & (a > 1e-15)

    def _viol(self, th):
        q = self.qlim
        v = np.maximum(q[:, 0] - th, 0.0) + np.maximum(th - q[:, 1], 0.0)
        return v.max(axis=1)

    # -- runner hook ---------------------------------------------------------------
    def extra_metrics(self, sim):
        if self._B is None:
            return {}
        al = self.alive
        nan = np.full(self._B, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = {
                "qd_peak": np.where(al, self.qd_pk.max(axis=1), np.inf),
                "qd_util": np.where(al, (self.qd_pk / self.qd_max).max(axis=1), nan),
                "qd_util_ss": np.where(al, (self.qd_pk_ss / self.qd_max).max(axis=1), nan),
                "qd_viol_frac": np.where(al, self.viol_steps / max(self.steps, 1), nan),
                "qdd_peak": np.where(al, self.qdd_pk.max(axis=1), nan),
                "qdd_peak_ss": np.where(al, self.qdd_pk_ss.max(axis=1), nan),
                "qdd_util": (np.where(al, (self.qdd_pk / self.qdd_max).max(axis=1), nan)
                             if self.qdd_max is not None else nan),
                "alt_active": np.where(al & (self.alt_tot > 0), self.alt_cnt / np.maximum(self.alt_tot, 1), nan),
                "qlim_viol": np.where(al, self.qlim_v, nan),
            }
        return out

    def describe(self):
        return {**self.inner.describe(), "wrapper": "R_common.JointStats", "ss_frac": self.ss_frac,
                "rel_tol": self.rel_tol}
