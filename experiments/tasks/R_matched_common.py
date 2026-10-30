"""Shared code of the matched-effort arm R_matched of the robotics case study.

Question: is implicit sign-bi-power ZNN (``znn_sbp_ie``) still more accurate
than linear ZNN when both command the same joint speed?  At the nominal gain
gamma = 10 it is 10-100x more accurate (error in m) but commands 3-4x the peak
joint speed (results/R/SUMMARY.md).  This arm matches the effort:

  direction A (main)  raise the gain of linear ZNN (and of IEZNN) until its
                      median effort equals that of implicit SBP at gamma = 10;
  direction B (check) lower the gain of implicit SBP until its median effort
                      equals that of linear ZNN at gamma = 10;

per robot x path x control rate x error unit, then compare the MSSRE.

Effort measures (per trajectory, median over the 12 R_off starts per cell):

  qd_peak       max_{k,i} |theta_dot_i| over the whole run, the paper's measure
                (= simulate's online ``peak_rate``; JointStats ``qd_peak``)
  qd_rms        RMS over the run of the joint-velocity norm ||theta_dot_k||_2
                (time-averaged effort; weights the start transient by its
                duration)
  qd_peak_late  peak over the last 75 % of the run (t >= 2.5 s), a steady-state
                effort measure (after every start transient has decayed)

plus diagnostics of where the peak occurs (``k_qd_peak``, ``qd_first`` = the
first command, ``qd_peak_early`` = peak over t < 0.5 s).

Everything here reuses the R protocol (experiments/tasks/R_common.py): the same
problem keys and kwargs, the same model entries, the ``JointStats`` wrapper
(subclassed by :class:`EffortStats`, which only adds measurements) and the
``ScaledError`` problem wrapper for the error in mm.  The 12 starts are the
R_off starts (seeds 0, 1, 2 x B = 4, ``simulate.initial_state``), batched with
all gains of one evaluation into ONE ``simulate`` call.  The runs are noise-free,
so the seed enters only through the start draw; the gamma = 10 runs therefore
reproduce the R_off / R_units rows (checked by analyze_R_matched.py).

Protocol, deviations and results: experiments/R_NOTES.md, section
"Matched-effort arm (2026-10-26)".
"""
from __future__ import annotations

import math
import time

import numpy as np

from experiments.tasks import R_common as RC
from znnbench import make_discretization, make_model, make_problem
from znnbench.metrics import summarize
from znnbench.simulate import initial_state, simulate

VERSION = 1

SEEDS = [0, 1, 2]                      # R_off seeds
B_PER_SEED = 4                         # R_off B
N_TRAJ = len(SEEDS) * B_PER_SEED       # 12 trajectories per cell
GAMMA_REF = 10.0                       # nominal gain of the headline comparison
UNITS = {"m": None, "mm": RC.UNIT_SCALE}
ENTRIES = ("znn_lin", "ieznn", "znn_sbp_ie")
UNIT_INVARIANT = ("znn_lin", "ieznn")  # linear in E: run in m for both unit cells
LATE_FRAC = 0.75                       # qd_peak_late / qd_rms_late: last 75 % of the run
EARLY_T = 0.5                          # s, qd_peak_early: t < 0.5 s
MEASURES = ("qd_peak", "qd_rms")       # effort measures that are matched by search

# search settings (gain on a log scale)
N_GRID1 = 16                           # stage-1 grid (shared by the matched measures)
N_GRID2 = 8                            # interior points per measure in stages 2 and 3
N_REFINE = 2                           # refinement stages after stage 1
SECANT_MAX = 3                         # secant steps on log gamma after the refinements
MATCH_TOL = 0.002                      # stop the secant steps at |f/T - 1| <= 0.2 %
ACCEPT_TOL = 0.02                      # the brief's tolerance: a match is valid at |f/T - 1| <= 2 %
A_GMAX = 150.0                         # direction A stage 1: gamma in [10, min(150, 1/tau)] (h <= 1)
A_HMAX = 1.8                           # extensions of A never exceed h = 1.8 (Euler limit h < 2)
B_RANGE = (0.25, GAMMA_REF)            # direction B stage 1: lower the gain from 10
B_GMIN = 0.01                          # extensions of B never go below gamma = 0.01


def a_range(tau):
    """(stage-1 interval, extension limit) of direction A at control period tau."""
    return (GAMMA_REF, min(A_GMAX, 1.0 / tau)), A_HMAX / tau


def b_range(tau):
    """(stage-1 interval, extension limit) of direction B."""
    return B_RANGE, B_GMIN


# ---------------------------------------------------------------------------
# effort wrapper
# ---------------------------------------------------------------------------
class EffortStats(RC.JointStats):
    """``JointStats`` plus the effort measures of the matched-effort arm.

    Transparent: ``step`` returns exactly what the wrapped discretizer returns
    (it is JointStats.step); the extra statistics only read x_k and x_{k+1}.
    Extra per-trajectory outputs (``extra_metrics``):

      qd_peak_late   max ||qd_k||_inf over k >= floor((1 - late_frac) N)
      qd_peak_early  max ||qd_k||_inf over t_k < early_t
      qd_first       ||qd_0||_inf, the first command (start transient)
      k_qd_peak      first step index at which ||qd_k||_inf attains qd_peak
      t_qd_peak      k_qd_peak * tau (s), the time the peak command is issued
      qd_rms         sqrt(mean_k ||qd_k||_2^2) over the whole run (rad/s)
      qd_rms_late    the same over the late window
      qd_peak_chk    max_k ||qd_k||_inf (must equal JointStats' qd_peak)
    """

    def __init__(self, inner, ss_frac=0.25, rel_tol=1e-6, late_frac=LATE_FRAC, early_t=EARLY_T,
                 record=False):
        self.late_frac = float(late_frac)
        self.early_t = float(early_t)
        super().__init__(inner, ss_frac=ss_frac, rel_tol=rel_tol, record=record)

    def _init(self, B, n, tau):
        super()._init(B, n, tau)
        T = float(getattr(self.problem, "T", np.nan))
        N = int(round(T / tau)) if np.isfinite(T) else 10 ** 12
        self.k_late = int(np.floor((1.0 - self.late_frac) * N))
        self.k_early = int(math.ceil(self.early_t / tau - 1e-9))     # steps k < k_early: t_k < early_t
        self.e_tau = float(tau)
        self.e_pk = np.zeros(B)
        self.e_kpk = np.zeros(B, dtype=np.int64)
        self.e_first = np.full(B, np.nan)
        self.e_pk_late = np.zeros(B)
        self.e_pk_early = np.zeros(B)
        self.e_sq = np.zeros(B)
        self.e_sq_late = np.zeros(B)
        self.e_n = 0
        self.e_n_late = 0

    def step(self, k, t, tau, x, z, f):
        x_next, z_next = super().step(k, t, tau, x, z, f)
        with np.errstate(invalid="ignore", over="ignore"):
            a = self.alive                                   # updated by JointStats for this step
            qd = np.where(a[:, None], x_next - x, 0.0) / tau
            ninf = np.abs(qd).max(axis=1)
            nsq = (qd * qd).sum(axis=1)
        up = ninf > self.e_pk
        self.e_kpk = np.where(up, k, self.e_kpk)
        self.e_pk = np.where(up, ninf, self.e_pk)
        if k == 0:
            self.e_first = ninf.copy()
        if k < self.k_early:
            self.e_pk_early = np.maximum(self.e_pk_early, ninf)
        self.e_sq += nsq
        self.e_n += 1
        if k >= self.k_late:
            self.e_pk_late = np.maximum(self.e_pk_late, ninf)
            self.e_sq_late += nsq
            self.e_n_late += 1
        return x_next, z_next

    def extra_metrics(self, sim):
        out = super().extra_metrics(sim)
        if self._B is None:
            return out
        al = self.alive
        with np.errstate(invalid="ignore", divide="ignore"):
            out.update({
                "qd_peak_late": np.where(al, self.e_pk_late, np.inf),
                "qd_peak_early": np.where(al, self.e_pk_early, np.inf),
                "qd_first": np.where(al, self.e_first, np.inf),
                "k_qd_peak": np.where(al, self.e_kpk.astype(float), np.nan),
                "t_qd_peak": np.where(al, self.e_kpk * self.e_tau, np.nan),
                "qd_rms": np.where(al, np.sqrt(self.e_sq / max(self.e_n, 1)), np.inf),
                "qd_rms_late": np.where(al, np.sqrt(self.e_sq_late / max(self.e_n_late, 1)), np.inf),
                "qd_peak_chk": np.where(al, self.e_pk, np.inf),
            })
        return out

    def describe(self):
        return {**super().describe(), "wrapper": "R_matched_common.EffortStats",
                "late_frac": self.late_frac, "early_t": self.early_t}


# ---------------------------------------------------------------------------
# one evaluation: all 12 starts x G gains in one simulate call
# ---------------------------------------------------------------------------
def make_cell_problem(robot, path, unit):
    """The R problem (same key and kwargs as R_off), in the given error unit."""
    prob = make_problem(robot, path=path, T=RC.T_HORIZON, x0_spread=RC.X0_SPREAD)
    scale = UNITS[unit]
    return prob if scale is None else RC.ScaledError(prob, scale=scale)


def run_unit(entry, unit):
    """Error unit in which an entry is simulated (linear-in-E entries are unit-invariant)."""
    return "m" if entry in UNIT_INVARIANT else unit


def _summ_value(arr, i, BG):
    a = np.asarray(arr).reshape(-1)
    if a.shape[0] == BG:
        return a[i]
    return a[0] if a.shape[0] == 1 else np.nan


def evaluate(robot, path, tau, entry, unit, gammas, tags=None):
    """Simulate ``entry`` on the 12 R_off starts at every gain in ``gammas``.

    Returns (rows, info): one dict per trajectory (12 * G rows) with the runner's
    metrics (``metrics.summarize``, eps 1e-3 / 1e-4, steady window 25 %), the
    JointStats statistics and the EffortStats effort measures; ``info`` has the
    wall time and the batch size.  ``tags`` (dict) is copied into every row."""
    gammas = [float(g) for g in gammas]
    G = len(gammas)
    u = run_unit(entry, unit)
    prob = make_cell_problem(robot, path, u)
    comp, disc_key, _name = RC.MODEL_ENTRIES[entry]
    model = make_model(comp["key"], **comp["kwargs"])
    x0s, seed_of, x0i_of = [], [], []
    for s in SEEDS:
        x0s.append(initial_state(prob, model, s, B_PER_SEED))
        seed_of += [s] * B_PER_SEED
        x0i_of += list(range(B_PER_SEED))
    x0s = np.concatenate(x0s)                                   # (12, n): the R_off starts
    x0 = np.tile(x0s, (G, 1))
    gam = np.repeat(np.asarray(gammas, float), N_TRAJ)
    BG = N_TRAJ * G
    disc = EffortStats(make_discretization(disc_key), ss_frac=0.25, rel_tol=1e-6)
    t0 = time.perf_counter()
    with np.errstate(all="ignore"):
        sim = simulate(prob, model, disc, tau, T=RC.T_HORIZON, B=BG, gamma=gam, x0=x0, seed=0, noise=None,
                       traj_ids=np.tile(np.arange(N_TRAJ), G), info="analytic", data_noise=0.0,
                       record_every=1, div_threshold=1e10, record_solution_error=False)
        summ = summarize(sim, eps=(1e-3, 1e-4), frac=0.25)
        summ["final_residual"] = np.asarray(sim["residual"])[-1]
        summ["max_residual"] = np.max(np.asarray(sim["residual"]), axis=0)
        summ.update(disc.extra_metrics(sim))
    wall = time.perf_counter() - t0
    div = np.asarray(sim["diverged"], bool)
    rows = []
    for g in range(G):
        for j in range(N_TRAJ):
            i = g * N_TRAJ + j
            r = {"robot": robot.replace("robot_", ""), "path": path, "tau": float(tau),
                 "rate": int(round(1.0 / tau)), "entry": entry, "unit": unit, "sim_unit": u,
                 "gamma": gammas[g], "h": gammas[g] * tau, "seed": seed_of[j], "x0_index": x0i_of[j],
                 "pair": seed_of[j] * 1000 + x0i_of[j], "traj": j,
                 "status": "diverged" if div[i] else "ok"}
            for k, v in summ.items():
                if v is None:
                    continue
                val = _summ_value(v, i, BG)
                r[k] = float(val) if np.ndim(val) == 0 else val
            r["mssre_mm"] = r["mssre"] * 1e3
            r["wall_call"] = wall
            r["batch"] = BG
            if tags:
                r.update(tags)
            rows.append(r)
    return rows, {"wall": wall, "batch": BG, "steps": int(sim["steps"]), "G": G}


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
_BIG = 1e300


def median_keep_inf(x):
    """Median ignoring NaN, keeping +inf (diverged) as a worst-case value."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan
    v = float(np.median(np.clip(x, -_BIG, _BIG)))
    return np.inf if v >= 0.5 * _BIG else v


def medians_by_gamma(rows, measure):
    """{gamma: median of ``measure`` over the 12 trajectories at that gain}."""
    by = {}
    for r in rows:
        by.setdefault(r["gamma"], []).append(r.get(measure, np.nan))
    return {g: median_keep_inf(v) for g, v in sorted(by.items())}


# ---------------------------------------------------------------------------
# matched-gain search on log gamma (batched bracketing + secant)
# ---------------------------------------------------------------------------
def _crossing(gs, fs, T, direction):
    """Bracket (lo, hi) of the first crossing of the target T along the search
    direction, on points sorted by gamma (ascending).

    direction 'up'   (A: raise the gain from 10): the first point with f > T,
                     scanning upwards; bracket (previous point, that point).
    direction 'down' (B: lower the gain from 10): the first point with f <= T,
                     scanning downwards; bracket (that point, next point).
    Returns (lo, hi, flag) with flag in {'ok', 'above_range' (target not reached
    at the top), 'below_range' (already over the target at the bottom)}."""
    gs = list(gs)
    fs = list(fs)
    if direction == "up":
        for j, f in enumerate(fs):
            if f > T:
                if j == 0:
                    return None, gs[0], "below_range"
                return gs[j - 1], gs[j], "ok"
        return gs[-1], None, "above_range"
    for j in range(len(fs) - 1, -1, -1):
        if fs[j] <= T:
            if j == len(fs) - 1:
                return gs[-1], None, "above_range"
            return gs[j], gs[j + 1], "ok"
    return None, gs[0], "below_range"


def _secant(lo, flo, hi, fhi, T):
    """Root of the log-log secant through (lo, flo), (hi, fhi) at level T."""
    if not (flo > 0 and fhi > 0 and T > 0) or fhi == flo:
        return math.sqrt(lo * hi)
    w = (math.log(T) - math.log(flo)) / (math.log(fhi) - math.log(flo))
    w = min(max(w, 0.0), 1.0)
    return math.exp(math.log(lo) + w * (math.log(hi) - math.log(lo)))


def match_gain(evaluate_fn, targets, direction, g_range, g_limit, log=None):
    """Find, for every measure in ``targets``, the gain at which the median
    effort of the searched entry reaches the target, starting from gamma = 10
    and moving in ``direction`` ('up' for A, 'down' for B).

    evaluate_fn(gammas, stage) -> rows (12 per gain; every row has every measure).
    g_range: stage-1 interval; g_limit: hard bound of range extensions (the
    largest gain for 'up', the smallest for 'down').

    Stages: (1) a log grid of N_GRID1 gains over g_range (extended up to
    g_limit if the target is not bracketed); (2..) N_REFINE refinements with
    N_GRID2 interior gains per measure inside its bracket; then up to SECANT_MAX
    secant steps on (log gamma, log f) until |f/T - 1| <= MATCH_TOL.  All
    measures are searched in the same simulate calls.  Returns {measure: result}."""
    say = log or (lambda *_: None)
    pts = {}                                         # gamma -> {measure: median}

    def run(gs, stage):
        gs = sorted({float(g) for g in gs if float(g) not in pts})
        if not gs:
            return
        rows = evaluate_fn(gs, stage)
        for m in targets:
            for g, v in medians_by_gamma(rows, m).items():
                pts.setdefault(g, {})[m] = v
        say(f"  [{stage}] {len(gs)} gains {gs[0]:.4g}..{gs[-1]:.4g}")

    def sorted_pts(m):
        gs = sorted(pts)
        return gs, [pts[g][m] for g in gs]

    lo, hi = g_range
    run(np.geomspace(lo, hi, N_GRID1), "s1")
    res = {}
    for m, T in targets.items():
        r = {"measure": m, "target": float(T), "direction": direction}
        # range extensions (rare): only for this measure's bracket
        for ext in range(3):
            gs, fs = sorted_pts(m)
            a, b, flag = _crossing(gs, fs, T, direction)
            if flag == "ok":
                break
            if direction == "up" and flag == "above_range" and gs[-1] < g_limit * (1 - 1e-12):
                new_hi = min(gs[-1] * 2.0, g_limit)
                run(np.geomspace(gs[-1], new_hi, N_GRID2 + 1)[1:], f"ext{ext + 1}")
            elif direction == "down" and flag == "below_range" and gs[0] > g_limit * (1 + 1e-12):
                new_lo = max(gs[0] / 8.0, g_limit)
                run(np.geomspace(new_lo, gs[0], N_GRID2 + 1)[:-1], f"ext{ext + 1}")
            else:
                break
        gs, fs = sorted_pts(m)
        r["grid_monotone"] = bool(np.all(np.diff([f for f in fs if np.isfinite(f)]) >= -1e-12))
        a, b, flag = _crossing(gs, fs, T, direction)
        r["bracket_flag"] = flag
        r["bracket"] = [a, b]
        res[m] = r
    # refinements: interior points of every open bracket, one simulate call per stage
    for st in range(N_REFINE):
        new = []
        for m, r in res.items():
            if r["bracket_flag"] != "ok":
                continue
            a, b = r["bracket"]
            if b / a - 1.0 > 1e-4:
                new += list(np.geomspace(a, b, N_GRID2 + 2)[1:-1])
        if not new:
            break
        run(new, f"s{st + 2}")
        for m, r in res.items():
            if r["bracket_flag"] != "ok":
                continue
            a0, b0 = r["bracket"]
            gs, fs = sorted_pts(m)
            sel = [(g, f) for g, f in zip(gs, fs) if a0 * (1 - 1e-12) <= g <= b0 * (1 + 1e-12)]
            a, b, flag = _crossing([g for g, _ in sel], [f for _, f in sel], r["target"], direction)
            if flag == "ok":
                r["bracket"] = [a, b]
    # secant steps on log gamma, then the matched gain is evaluated (final rows)
    for m, r in res.items():
        r["gamma_star"] = np.nan
        r["f_star"] = np.nan
    for it in range(SECANT_MAX):
        want = {}
        for m, r in res.items():
            if r["bracket_flag"] != "ok":
                continue
            if np.isfinite(r["f_star"]) and abs(r["f_star"] / r["target"] - 1.0) <= MATCH_TOL:
                continue
            a, b = r["bracket"]
            g = _secant(a, pts[a][m], b, pts[b][m], r["target"])
            want[m] = g
        if not want:
            break
        run(list(want.values()), f"secant{it + 1}")
        for m, g in want.items():
            r = res[m]
            f = pts[g][m]
            r["gamma_star"], r["f_star"] = g, f
            a, b = r["bracket"]
            # both directions keep f(lo) <= T < f(hi) (see _crossing)
            r["bracket"] = [g, b] if f <= r["target"] else [a, g]
    for m, r in res.items():
        if r["bracket_flag"] == "ok":
            # the matched gain: the evaluated gain whose median effort is closest to the target
            # (the last secant point, or a bracket end if closer)
            cand = [g for g in pts if np.isfinite(pts[g][m])]
            a, b = r["bracket"]
            near = [g for g in cand if a * (1 - 1e-9) <= g <= b * (1 + 1e-9)] or cand
            g = min(near, key=lambda q: abs(pts[q][m] / r["target"] - 1.0))
            r["gamma_star"], r["f_star"] = float(g), float(pts[g][m])
            r["mismatch"] = r["f_star"] / r["target"] - 1.0
            r["status"] = "matched" if abs(r["mismatch"]) <= ACCEPT_TOL else "loose"
            r["bracket_rel_width"] = r["bracket"][1] / r["bracket"][0] - 1.0
        else:
            gs, fs = sorted_pts(m)
            r["status"] = {"above_range": "not_reached", "below_range": "unreachable"}[r["bracket_flag"]]
            r["mismatch"] = np.nan
            r["f_range"] = [float(np.nanmin(fs)), float(np.nanmax(fs))]
        r["n_gains_evaluated"] = len(pts)
        r["curve"] = [[g, pts[g][m]] for g in sorted(pts)]
    return res, pts
