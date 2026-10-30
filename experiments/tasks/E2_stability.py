#!/usr/bin/env python
"""E2 custom tasks: maximal stable step h = gamma*tau by batched multi-section.

    build : python experiments/tasks/E2_stability.py build --scale mini|full --out DIR
            -> DIR/tasks/<id>.json, DIR/tasks.jsonl (memguard format), DIR/config.json
    run   : python experiments/tasks/E2_stability.py run --task DIR/tasks/<id>.json [--force]
            -> DIR/results/<id>.parquet (done file, written atomically, LAST)
               DIR/results/<id>.profiles.parquet (coarse-grid + refinement points)
    local : python experiments/tasks/E2_stability.py local --out DIR   (in-process, debugging only)
    status: python experiments/tasks/E2_stability.py status --out DIR

One task = one formula x one problem x one tau (all activations, seeds,
initial-error magnitudes r and directions b).  Definitions (see
experiments/E2_NOTES.md and experiments/configs/E2.py):

* x0 = x*(0) + J(0)^{-1}(r u), ||u||_2 = 1  =>  ||E(x0, 0)||_2 = r exactly.
* MSSRE(h) = max ||E|| over the last 25 % of T (inf when diverged).
* criterion ``bounded``: not diverged and MSSRE(h) <= ||E(x0,0)||.
* criterion ``floor`` (protocol): not diverged and MSSRE(h) <= THETA*max(F, F_ABS),
  F = MSSRE of the same formula with LINEAR activation (same instance,
  direction, magnitude) at the formula's root-locus optimal step h_opt
  ("formula floor"; fallback when there is no closed-form theory: the minimum
  over the coarse grid).
* criterion ``nodiv`` (diagnostic): not diverged (finite, |x| <= 1e10 over T).
* criterion ``e4`` (diagnostic, = E4's P2 stability definition): finite,
  MSSRE <= 0.1 ||E(x0,0)||, max over the last T/8 <= 2 x max over the preceding
  T/8 (or <= 1e-10), and max_k ||E_k|| <= 10 ||E(x0,0)||.
* h_max = upper end of the run of consecutive stable coarse-grid points that
  contains h_ref = argmin MSSRE over the stable grid points, refined by
  ``rounds`` K-section rounds on [last stable, first unstable]; reported as the
  bracket midpoint with half-width ``h_max_err``.  Censoring: ``above`` (stable
  up to h = 3), ``below`` (no stable grid point; h_max < H_GRID_MIN).

Never fatal: an exception in one (activation, seed) group is recorded as
status ``error`` with the traceback tail; the task still writes its done file.
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import resource  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FORMAT = "znnbench-e2-task/1"
# bump when the task code changes: it is part of the hashed payload, so a
# rebuild then produces new task ids and stale results are never reused
CODE_VERSION = "e2-tasks-3"
THREAD_ENV = {k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                               "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")}
PYTHON = os.path.normpath(os.path.abspath(sys.executable))
CRITERIA = ("bounded", "floor", "nodiv", "e4")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _canon(obj) -> str:
    def conv(o):
        if isinstance(o, dict):
            return {str(k): conv(v) for k, v in sorted(o.items())}
        if isinstance(o, (list, tuple)):
            return [conv(v) for v in o]
        if isinstance(o, (np.floating,)):
            o = float(o)
        if isinstance(o, float) and o.is_integer() and abs(o) < 2 ** 53:
            return int(o)
        if isinstance(o, np.integer):
            return int(o)
        return o
    return json.dumps(conv(obj), sort_keys=True, separators=(",", ":"))


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def load_config(scale):
    import importlib.util
    spec = importlib.util.spec_from_file_location("e2_config", ROOT / "experiments" / "configs" / "E2.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_config(scale)


def _atomic_write_parquet(df, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=str(path.parent))
    os.close(fd)
    try:
        df.to_parquet(tmp, index=False)
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _atomic_write_text(text: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=str(path.parent))
    with os.fdopen(fd, "w") as f:
        f.write(text)
    os.chmod(tmp, 0o644)
    os.replace(tmp, path)


def peak_rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2 ** 20 if sys.platform == "darwin" else r / 1024.0


# --------------------------------------------------------------------------
# task expansion
# --------------------------------------------------------------------------
def expand_tasks(cfg) -> list:
    """List of task payloads (without paths) for the config."""
    base = {k: cfg[k] for k in ("T", "frac", "h_grid", "theta", "F_abs", "div_threshold",
                                "refine", "seeds", "B")}
    base["e4"] = dict(cfg.get("e4_criterion") or {"rho": 0.1, "growth": 2.0, "transient": 10.0})
    base["code_version"] = CODE_VERSION
    tasks = []
    acts = cfg["activations"]
    if "linear" in acts:                       # the formula floor needs linear first
        acts = ["linear"] + [a for a in acts if a != "linear"]
    for prob in cfg["problems"]:
        formulas = list(cfg["formulas"])
        if prob["key"] in cfg.get("inverse_free_problems", ["matrix_inversion"]):
            formulas += [f for f in cfg.get("inverse_free_formulas", []) if f not in formulas]
        for f in formulas:
            tasks.append({**base, "kind": "main", "formula": f, "problem": prob, "tau": cfg["tau"],
                          "activations": acts, "r_list": cfg["r_list"]})
    # control: frozen A, linear activation, r_main only
    cp = cfg.get("control_problem")
    if cp:
        for f in cfg.get("control_formulas", cfg["formulas"]):
            tasks.append({**base, "kind": "control", "formula": f, "problem": cp, "tau": cfg["tau"],
                          "activations": ["linear"], "r_list": [cfg["r_main"]]})
    # tau sensitivity: linear activation, r_main only, both problems (+ the frozen-A
    # control, which separates the finite-horizon effect, O(1/N) = O(tau) at fixed T,
    # from the O(tau) time-variation shift)
    sens_problems = list(cfg["problems"]) + ([cp] if cp and cfg.get("tau_sens_control") else [])
    for tau in cfg.get("tau_sens", []):
        for prob in sens_problems:
            for f in cfg["tau_sens_formulas"]:
                tasks.append({**base, "kind": "tau_sens", "formula": f, "problem": prob, "tau": tau,
                              "activations": ["linear"], "r_list": [cfg["r_main"]]})
    # horizon sensitivity: linear activation, r_main, tau main, T long; linear_system + control
    hz_problems = [pp for pp in cfg["problems"] if pp["key"] == "linear_system"] + ([cp] if cp else [])
    for TL in cfg.get("horizon_T", []):
        for prob in hz_problems:
            for f in cfg.get("horizon_formulas", []):
                tasks.append({**base, "T": float(TL), "kind": "horizon", "formula": f, "problem": prob,
                              "tau": cfg["tau"], "activations": ["linear"], "r_list": [cfg["r_main"]]})
    # paired scaled-error arm: all activations and magnitudes, E in units of 1/s
    for sc in cfg.get("error_scales", []):
        for prob in cfg["problems"]:
            if prob["key"] not in cfg.get("scaled_problems", []):
                continue
            sprob = {**prob, "error_scale": float(sc), "label": f"{prob['label']}_Ex{float(sc):g}"}
            for f in cfg.get("scaled_formulas", []):
                tasks.append({**base, "kind": "scaled", "formula": f, "problem": sprob, "tau": cfg["tau"],
                              "activations": acts, "r_list": cfg["r_list"]})
    return tasks


def task_id(cfg, p) -> str:
    """Readable, position-independent id: inserting a formula into the grid does not
    rename the other tasks, so their done files stay valid; the payload hash (which
    contains CODE_VERSION) changes whenever anything that affects the result does."""
    h = hashlib.sha1(_canon(p).encode()).hexdigest()[:10]
    return (f"{cfg['name']}-{p['kind']}-{p['formula']}-{p['problem']['label']}-"
            f"t{float(p['tau']):g}-{h}")


def build(cfg, out: Path) -> Path:
    out = Path(out).resolve()
    (out / "tasks").mkdir(parents=True, exist_ok=True)
    (out / "results").mkdir(parents=True, exist_ok=True)
    payloads = expand_tasks(cfg)
    lines = []
    for p in payloads:
        tid = task_id(cfg, p)
        task = {"format": FORMAT, "id": tid, "config": cfg["name"], **p,
                "result": str(out / "results" / f"{tid}.parquet"),
                "profiles": str(out / "results" / f"{tid}.profiles.parquet")}
        tfile = out / "tasks" / f"{tid}.json"
        _atomic_write_text(json.dumps(task, indent=1, default=_json_default), tfile)
        lines.append(json.dumps({"id": tid,
                                 "cmd": [PYTHON, "experiments/tasks/E2_stability.py", "run",
                                         "--task", str(tfile)],
                                 "done": task["result"], "cwd": str(ROOT), "env": THREAD_ENV}))
    _atomic_write_text(json.dumps(cfg, indent=1, default=_json_default), out / "config.json")
    tf = out / "tasks.jsonl"
    _atomic_write_text("\n".join(lines) + "\n", tf)
    return tf


# --------------------------------------------------------------------------
# problem helpers
# --------------------------------------------------------------------------
class FrozenA:
    """linear_system with A(t) frozen at A(0); b(t) unchanged (control problem).

    E = A0 x - b(t), J = A0, dE/dt = -b_dot(t), x*(t) = A0^{-1} b(t).  For linear
    ZNN the error recursion is then exactly sum_i c_i e_{k+1-i} = -h e_k + O(tau^{p+1}).
    """

    def __init__(self, inner):
        self.inner = inner
        self.A0 = np.array(inner.A(0.0), dtype=float)
        self.n_x, self.m, self.T = inner.n_x, inner.m, inner.T
        self.name = inner.name + "_constA"
        self.family = inner.family

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

    def x_star(self, t):
        return np.linalg.solve(self.A0, self.inner.b(t))

    def residual(self, x, d):
        return np.linalg.norm(self.E(x, d), axis=-1)

    def x0(self, rng, B):
        return self.inner.x0(rng, B)

    def capabilities(self):
        return set()


def problem_seed(prob, seed) -> int:
    """Instance seed of a problem entry (seed index + the entry's ``seed_offset``)."""
    return int(seed) + int(prob.get("seed_offset", 0))


def make_problem_instance(prob, seed, T=None):
    """Problem of an entry; ``T`` overrides the horizon (horizon block).  Entries with
    ``error_scale`` s are wrapped in R_common.ScaledError (E_s = s E; residuals stay in
    the original unit).  Returns (problem used by simulate, unscaled problem)."""
    from znnbench import make_problem
    kw = dict(prob.get("kwargs", {}))
    if T is not None:
        kw["T"] = float(T)
    p = make_problem(prob["key"], seed=problem_seed(prob, seed), **kw)
    if prob.get("freeze_A"):
        p = FrozenA(p)
    base = p
    if prob.get("error_scale") not in (None, 1, 1.0):
        from experiments.tasks.R_common import ScaledError
        p = ScaledError(p, scale=float(prob["error_scale"]))
    return p, base


def initial_states(problem, seed, B, r_list):
    """X0[ir, b] = x*(0) + J^{-1}(r u_b); u_b uniform on the error-space sphere,
    the same directions for every r.  Returns X0 (nr, B, n_x) and r0 (nr, B)."""
    d0 = problem.data(0.0)
    xs = np.asarray(problem.x_star(0.0), dtype=float).reshape(1, -1)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 2002]))
    U = rng.standard_normal((B, problem.m))
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    X0 = np.empty((len(r_list), B, problem.n_x))
    for i, r in enumerate(r_list):
        dx = problem.solve_J(np.repeat(xs, B, axis=0), d0, r * U)
        X0[i] = xs + dx
    E0 = np.stack([np.linalg.norm(problem.E(X0[i], d0), axis=1) for i in range(len(r_list))])
    return X0, E0


# --------------------------------------------------------------------------
# the scan
# --------------------------------------------------------------------------
def _simulate_rows(problem, model, disc, tau, T, X, hs, seed, div_threshold, frac):
    from znnbench import simulate
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with np.errstate(all="ignore"):
            sim = simulate(problem, model, disc, tau, T=T, B=len(X), gamma=np.asarray(hs) / tau,
                           x0=X, seed=seed, record_solution_error=False, online_stats=False,
                           div_threshold=div_threshold)
    t = sim["t"]
    w = t >= t[-1] - frac * (t[-1] - t[0])
    res = sim["residual"]
    span = t[-1] - t[0]
    wa = t >= t[-1] - span / 8                       # last T/8
    wb = (t >= t[-1] - span / 4) & ~wa               # preceding T/8
    with np.errstate(invalid="ignore"):
        ms = np.max(res[w], axis=0)
        ma, mb = np.max(res[wa], axis=0), np.max(res[wb], axis=0)
        mall = np.max(res, axis=0)
    bad = sim["diverged"] | ~np.isfinite(ms)
    ms = np.where(bad, np.inf, ms)
    mall = np.where(bad | ~np.isfinite(mall), np.inf, mall)
    return {"mssre": ms, "final": res[-1], "r0": res[0], "diverged": sim["diverged"],
            "max_a": ma, "max_b": mb, "max_all": mall,
            "diverged_at": sim["diverged_at"], "wall": sim["wall"], "steps": sim["steps"],
            "steps_run": sim["steps_run"]}


def _stable(crit, ms, r0, Fthr, extra=None, e4=None):
    """Stability mask of a criterion.  ``extra`` = dict of arrays max_a, max_b, max_all
    (same shape as ms) for the E4-compatible criterion."""
    fin = np.isfinite(ms)
    if crit == "bounded":
        return fin & (ms <= r0)
    if crit == "nodiv":
        return fin
    if crit == "e4":
        e4 = e4 or {"rho": 0.1, "growth": 2.0, "transient": 10.0}
        ma, mb, mall = extra["max_a"], extra["max_b"], extra["max_all"]
        with np.errstate(invalid="ignore"):
            ok = fin & (ms <= e4["rho"] * r0)
            ok &= (ma <= e4["growth"] * mb) | (ma <= 1e-10)
            ok &= np.isfinite(mall) & (mall <= e4["transient"] * r0)
        return ok
    return fin & (ms <= Fthr)


def scan_activation(problem, model, disc, task, seed, X0, E0, F_lin=None, h_floor_ref=np.nan):
    """Coarse grid + K-section refinement for both criteria.

    ``F_lin`` is the formula floor from the linear activation (None while the
    linear activation itself is scanned: then F = MSSRE at ``h_floor_ref`` =
    the root-locus optimal step h_opt of the formula, simulated as extra rows of
    the coarse batch; fallback = min over the grid when h_opt is unavailable).

    Returns (result dict arrays of shape (nr, B, len(CRITERIA)), F used (nr, B), F rule,
    profile rows, timing)."""
    tau, T = float(task["tau"]), float(task["T"])
    hg = np.asarray(task["h_grid"], dtype=float)
    G = len(hg)
    nr, Bn, nx = X0.shape
    frac, div = float(task["frac"]), float(task["div_threshold"])
    theta, F_abs = float(task["theta"]), float(task["F_abs"])
    rounds, K = int(task["refine"]["rounds"]), int(task["refine"]["K"])
    prof = []
    timing = {"n_sim": 0, "wall": 0.0, "row_steps": 0, "steps": 0}

    # ---- coarse grid: rows (ir, b, g)
    X = np.repeat(X0.reshape(nr * Bn, nx), G, axis=0)
    H = np.tile(hg, nr * Bn)
    use_ref = F_lin is None and np.isfinite(h_floor_ref) and 0 < h_floor_ref <= hg[-1]
    if use_ref:                                   # extra rows: linear floor at h_opt
        X = np.vstack([X, X0.reshape(nr * Bn, nx)])
        H = np.concatenate([H, np.full(nr * Bn, float(h_floor_ref))])
    out = _simulate_rows(problem, model, disc, tau, T, X, H, seed, div, frac)
    ms_ref_rows = out["mssre"][nr * Bn * G:].reshape(nr, Bn) if use_ref else None
    if use_ref:
        for k in ("mssre", "final", "r0", "diverged", "max_a", "max_b", "max_all"):
            out[k] = out[k][:nr * Bn * G]
    timing["n_sim"] += 1
    timing["wall"] += out["wall"]
    timing["row_steps"] += len(X) * out["steps_run"]
    timing["steps"] += out["steps_run"]
    timing["wall_coarse"] = out["wall"]
    timing["rows_coarse"] = len(X)
    timing["steps_coarse"] = out["steps_run"]
    ms = out["mssre"].reshape(nr, Bn, G)
    r0 = out["r0"].reshape(nr, Bn, G)[..., 0]
    div_g = out["diverged"].reshape(nr, Bn, G)
    fin_g = out["final"].reshape(nr, Bn, G)
    ext_g = {k: out[k].reshape(nr, Bn, G) for k in ("max_a", "max_b", "max_all")}
    e4c = task.get("e4")
    for ir in range(nr):
        for b in range(Bn):
            for g in range(G):
                prof.append((ir, b, float(hg[g]), "coarse", float(ms[ir, b, g]),
                             float(fin_g[ir, b, g]), bool(div_g[ir, b, g])))

    # formula floor: own grid for linear activation, else the linear floor
    if F_lin is None:
        with np.errstate(invalid="ignore"):
            F = np.min(np.where(np.isfinite(ms), ms, np.inf), axis=2)
        F_rule = "grid_min"
        if use_ref:
            good = np.isfinite(ms_ref_rows)
            F = np.where(good, ms_ref_rows, F)
            F_rule = "h_opt" if good.all() else "h_opt|grid_min"
            for ir in range(nr):
                for b in range(Bn):
                    prof.append((ir, b, float(h_floor_ref), "floor_ref", float(ms_ref_rows[ir, b]),
                                 np.nan, not bool(good[ir, b])))
    else:
        F, F_rule = F_lin, "from_linear"
    Fthr = theta * np.maximum(F, F_abs)

    nc = len(CRITERIA)
    res = {k: np.full((nr, Bn, nc), np.nan) for k in
           ("h_max", "h_max_err", "h_lo", "h_ref", "ms_ref", "ms_at_a", "n_islands")}
    cens = np.full((nr, Bn, nc), "", dtype=object)
    lo = np.full((nr, Bn, nc), np.nan)
    hi = np.full((nr, Bn, nc), np.nan)
    ms_lo = np.full((nr, Bn, nc), np.nan)
    active = np.zeros((nr, Bn, nc), dtype=bool)
    for ic, crit in enumerate(CRITERIA):
        st = _stable(crit, ms, r0[..., None], Fthr[..., None], extra=ext_g, e4=e4c)
        for ir in range(nr):
            for b in range(Bn):
                s = st[ir, b]
                if not s.any():
                    cens[ir, b, ic] = "below"
                    res["h_max"][ir, b, ic] = hg[0]
                    res["n_islands"][ir, b, ic] = 0
                    continue
                cand = np.where(s, ms[ir, b], np.inf)
                j = int(np.argmin(cand))
                u = j
                while u + 1 < G and s[u + 1]:
                    u += 1
                lidx = j
                while lidx - 1 >= 0 and s[lidx - 1]:
                    lidx -= 1
                res["h_ref"][ir, b, ic] = hg[j]
                res["ms_ref"][ir, b, ic] = ms[ir, b, j]
                res["h_lo"][ir, b, ic] = hg[lidx]
                res["n_islands"][ir, b, ic] = int(s[u + 2:].sum()) if u + 2 <= G else 0
                if u == G - 1:
                    cens[ir, b, ic] = "above"
                    res["h_max"][ir, b, ic] = hg[-1]
                    res["h_max_err"][ir, b, ic] = 0.0
                    res["ms_at_a"][ir, b, ic] = ms[ir, b, u]
                    continue
                lo[ir, b, ic], hi[ir, b, ic] = hg[u], hg[u + 1]
                ms_lo[ir, b, ic] = ms[ir, b, u]
                active[ir, b, ic] = True

    # ---- refinement rounds: rows (entry, k)
    for _ in range(rounds):
        idx = np.argwhere(active)
        if len(idx) == 0:
            break
        fr = np.arange(1, K + 1) / (K + 1)
        pts = lo[active][:, None] + (hi[active] - lo[active])[:, None] * fr[None, :]   # (E, K)
        Xr = np.repeat(X0[idx[:, 0], idx[:, 1]], K, axis=0)
        Hr = pts.reshape(-1)
        o = _simulate_rows(problem, model, disc, tau, T, Xr, Hr, seed, div, frac)
        timing["n_sim"] += 1
        timing["wall"] += o["wall"]
        timing["row_steps"] += len(Xr) * o["steps_run"]
        timing["steps"] += o["steps_run"]
        msr = o["mssre"].reshape(-1, K)
        finr = o["final"].reshape(-1, K)
        divr = o["diverged"].reshape(-1, K)
        extr = {k: o[k].reshape(-1, K) for k in ("max_a", "max_b", "max_all")}
        for e, (ir, b, ic) in enumerate(idx):
            crit = CRITERIA[ic]
            s = _stable(crit, msr[e], r0[ir, b], Fthr[ir, b],
                        extra={k: v[e] for k, v in extr.items()}, e4=e4c)
            for kk in range(K):
                prof.append((int(ir), int(b), float(pts[e, kk]), f"refine_{crit}", float(msr[e, kk]),
                             float(finr[e, kk]), bool(divr[e, kk])))
            bad = np.where(~s)[0]
            if len(bad) == 0:              # all interior points stable -> boundary in (p_K, hi)
                lo[ir, b, ic] = pts[e, -1]
                ms_lo[ir, b, ic] = msr[e, -1]
            else:
                jb = int(bad[0])
                if jb > 0:
                    lo[ir, b, ic] = pts[e, jb - 1]
                    ms_lo[ir, b, ic] = msr[e, jb - 1]
                hi[ir, b, ic] = pts[e, jb]
    fin_mask = ~np.isnan(lo)
    res["h_max"] = np.where(fin_mask, 0.5 * (lo + hi), res["h_max"])
    res["h_max_err"] = np.where(fin_mask, 0.5 * (hi - lo), res["h_max_err"])
    res["ms_at_a"] = np.where(fin_mask, ms_lo, res["ms_at_a"])
    return res, cens, F, F_rule, prof, timing


def _disc_describe(disc):
    try:
        return json.dumps(disc.describe(), default=_json_default)
    except Exception as exc:                 # pragma: no cover
        return json.dumps({"describe_error": repr(exc)})


def run_task(task: dict, verbose=True):
    import pandas as pd
    from znnbench import make_discretization, make_model

    t_start = time.perf_counter()
    rows, prows = [], []
    h_floor_ref = np.nan
    meta = {"config": task.get("config"), "task_id": task["id"], "kind": task["kind"],
            "formula": task["formula"], "problem": task["problem"]["label"],
            "problem_key": task["problem"]["key"], "tau": float(task["tau"]), "T": float(task["T"]),
            "theta": float(task["theta"]), "F_abs": float(task["F_abs"]),
            "error_scale": float(task["problem"].get("error_scale", 1.0) or 1.0)}
    try:
        disc = make_discretization(task["formula"])
        ddesc = _disc_describe(disc)
        meta["disc_implicit"] = bool(getattr(disc, "implicit", False))
        meta["disc_class"] = type(disc).__name__
        try:
            from experiments.tasks.E2_theory import Theory
            th = Theory.from_disc(task["formula"], disc)
            hm = th.h_max()
            h_floor_ref = th.h_opt(hm)[0]
            meta["h_theory_task"] = float(hm)
            meta["theory_type"] = str(th.theory_type)
        except Exception:                     # no theory -> floor = grid minimum
            h_floor_ref = np.nan
            meta["h_theory_task"] = np.nan
        meta["h_floor_ref"] = float(h_floor_ref)
    except Exception as exc:
        disc, ddesc = None, json.dumps({"error": repr(exc)})
        msg = "".join(traceback.format_exception(exc))[-2000:]
        for seed in task["seeds"]:
            for act in task["activations"]:
                rows.append({**meta, "activation": act, "seed": problem_seed(task["problem"], seed),
                             "seed_index": int(seed), "status": "error",
                             "message": msg, "disc_describe": ddesc})
    if disc is not None:
        for seed in task["seeds"]:
            try:
                prob, prob_unscaled = make_problem_instance(task["problem"], seed, T=task["T"])
                # x0 from the UNSCALED problem: ||E(x0, 0)|| = r in the original unit, the same
                # x0 as the unscaled runs (paired), whatever the error scale
                X0, E0 = initial_states(prob_unscaled, seed, int(task["B"]), task["r_list"])
            except Exception as exc:
                msg = "".join(traceback.format_exception(exc))[-2000:]
                for act in task["activations"]:
                    rows.append({**meta, "activation": act, "seed": problem_seed(task["problem"], seed),
                                 "seed_index": int(seed), "status": "error",
                                 "message": msg, "disc_describe": ddesc})
                continue
            F_lin = None
            for act in task["activations"]:
                t0 = time.perf_counter()
                try:
                    model = make_model("znn", activation=act)
                    res, cens, F, F_rule, prof, timing = scan_activation(
                        prob, model, disc, task, seed, X0, E0, F_lin=F_lin, h_floor_ref=h_floor_ref)
                    status, msg = "ok", ""
                except Exception as exc:
                    status = "error"
                    msg = "".join(traceback.format_exception(exc))[-2000:]
                    res = cens = F = F_rule = prof = timing = None
                if act == "linear" and status == "ok":
                    F_lin = F
                gwall = time.perf_counter() - t0
                if status != "ok":
                    rows.append({**meta, "activation": act, "seed": problem_seed(task["problem"], seed),
                                 "seed_index": int(seed), "status": status,
                                 "message": msg, "disc_describe": ddesc})
                    if verbose:
                        print(f"[E2] {task['formula']} {meta['problem']} {act} seed={seed}: ERROR\n{msg}",
                              flush=True)
                    continue
                for ir, r in enumerate(task["r_list"]):
                    for b in range(int(task["B"])):
                        for ic, crit in enumerate(CRITERIA):
                            rows.append({
                                **meta, "activation": act, "seed": problem_seed(task["problem"], seed),
                                "seed_index": int(seed), "b": b,
                                "r": float(r), "e0": float(E0[ir, b]), "criterion": crit,
                                "h_max": float(res["h_max"][ir, b, ic]),
                                "h_max_err": float(res["h_max_err"][ir, b, ic]),
                                "censored": str(cens[ir, b, ic]),
                                "h_lo": float(res["h_lo"][ir, b, ic]),
                                "h_ref": float(res["h_ref"][ir, b, ic]),
                                "mssre_ref": float(res["ms_ref"][ir, b, ic]),
                                "mssre_at_hmax_lo": float(res["ms_at_a"][ir, b, ic]),
                                "n_islands": float(res["n_islands"][ir, b, ic]),
                                "floor_F": float(F[ir, b]),
                                "floor_rule": F_rule,
                                "status": "ok", "message": "",
                                "n_sim": timing["n_sim"], "wall_sim": timing["wall"],
                                "row_steps": timing["row_steps"], "group_wall": gwall,
                                "wall_coarse": timing["wall_coarse"], "rows_coarse": timing["rows_coarse"],
                                "steps_coarse": timing["steps_coarse"],
                                "disc_describe": ddesc,
                            })
                for (ir, b, h, stage, msv, finv, dv) in prof:
                    prows.append({"task_id": task["id"], "kind": task["kind"], "formula": task["formula"],
                                  "problem": meta["problem"], "tau": meta["tau"], "activation": act,
                                  "seed": problem_seed(task["problem"], seed), "b": b,
                                  "r": float(task["r_list"][ir]),
                                  "e0": float(E0[ir, b]), "h": h, "stage": stage, "mssre": msv,
                                  "final_residual": finv, "diverged": dv})
                if verbose:
                    hb = np.nanmedian(res["h_max"][..., 0])
                    hf = np.nanmedian(res["h_max"][..., 1])
                    print(f"[E2] {task['formula']:>18s} {meta['problem']:>20s} tau={meta['tau']:g} "
                          f"{act:>16s} seed={seed}: h_max(bounded)={hb:.4g} h_max(floor)={hf:.4g} "
                          f"[{timing['n_sim']} sims, {gwall:.2f}s]", flush=True)
    total = time.perf_counter() - t_start
    rss = peak_rss_mb()
    df = pd.DataFrame(rows)
    df["task_wall"] = total
    df["peak_rss_mb"] = rss
    pdf = pd.DataFrame(prows)
    if len(pdf):
        _atomic_write_parquet(pdf, Path(task["profiles"]))
    _atomic_write_parquet(df, Path(task["result"]))       # the done file, last
    if verbose:
        print(f"[E2] task {task['id']} done in {total:.1f}s, peak RSS {rss:.0f} MB -> {task['result']}",
              flush=True)
    return task["result"]


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _load_tasks(out: Path):
    with open(Path(out) / "tasks.jsonl") as f:
        return [json.loads(line) for line in f if line.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--scale", default=None, choices=["mini", "full"])
    b.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--task", required=True)
    r.add_argument("--force", action="store_true")
    r.add_argument("--quiet", action="store_true")
    lo = sub.add_parser("local")
    lo.add_argument("--out", required=True)
    lo.add_argument("--quiet", action="store_true")
    s = sub.add_parser("status")
    s.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    if a.cmd == "build":
        scale = a.scale or os.environ.get("ZNNBENCH_SCALE", "full")
        cfg = load_config(scale)
        tf = build(cfg, Path(a.out))
        n = len(_load_tasks(Path(a.out)))
        print(f"{n} tasks ({len(cfg['formulas'])} formulas: {cfg['formulas']}) -> {tf}")
        print(f"launch: {PYTHON} {ROOT / 'tools' / 'memguard.py'} pool --tasks {tf} "
              f"--workers 3 --max-gb 40")
        return 0
    if a.cmd == "run":
        task = json.loads(Path(a.task).read_text())
        if task.get("format") != FORMAT:
            raise SystemExit(f"not an E2 task file: {a.task}")
        if Path(task["result"]).exists() and not a.force:
            print(f"[E2] {task['id']} already done")
            return 0
        out = run_task(task, verbose=True)
        return 0 if Path(out).exists() else 1
    if a.cmd == "local":
        for t in _load_tasks(Path(a.out)):
            if Path(t["done"]).exists():
                continue
            task = json.loads(Path(t["cmd"][4]).read_text())
            run_task(task, verbose=not a.quiet)
        return 0
    if a.cmd == "status":
        ts = _load_tasks(Path(a.out))
        done = sum(Path(t["done"]).exists() for t in ts)
        print(f"{done}/{len(ts)} tasks done")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
