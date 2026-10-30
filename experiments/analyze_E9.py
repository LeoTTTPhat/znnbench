#!/usr/bin/env python
"""Analysis of E9: GNN vs ZNN vs prediction-correction vs Uhlig vs Newton at matched cost.

    python experiments/analyze_E9.py --results results/mini/E9 \
        --figdir figures/mini --tabdir results/tables/mini

Inputs: the runner results under ``--results`` (every ``*/tasks.jsonl`` written by
``experiments/tasks/E9_build.py``) and the per-step cost profile
``<results>/cost/cost_profile-*.json`` (task ``E9_cost.py``; recomputed in-process if
missing).  Outputs:

* figures ``fig_E9_*.pdf/png`` (style.py / keys.py conventions);
* LaTeX booktabs tables ``E9_*.tex``;
* ``<results>/summary.json`` and ``<results>/SUMMARY.md`` (headline numbers + interpretation).

Statistics: every cell = (problem, method, gain level, tau) holds n_seeds x B trajectories;
medians and IQRs over them; bootstrap CIs are hierarchical (seeds with replacement, then initial
states within each seed), 2000 resamples, fixed seed; Kendall tau_b for rankings; empirical orders
fitted only on the asymptotic range (consecutive-slope diagnostics, round-off floor 1e-12
excluded), reported together with the all-points slope.

Review fixes (2026-10-18, see E9_NOTES.md section 7):
* closed-loop trajectories that grow below the runner's 1e10 threshold (status 'ok', final >
  initial residual) count as diverged (MSSRE = +inf) everywhere; the open-loop Davidenko control is
  exempt;
* matched-cost interpolation uses the two adjacent grid points that bracket the budget (it no longer
  bridges over a missing / unstable grid point); bootstrap resamples with a +inf bracket count as
  +inf; order-statistic CIs; verdicts report equivalence AND direction ("equivalent (but worse)"),
  floor-limited gains, transient-limited brackets, and a Bonferroni variant;
* sensitivity cost with one LU factorisation per data sample on the affine problems;
* GNN cells need a transient-free AND quasi-static gain (gamma sigma_min^2 >= 3 omega) for the
  order and lag-slope fits (the trace ratio misses slow GNN transients);
* Kendall tau_b ties values at the round-off floor; causal-only tau_b reported;
* transient cost: time to ||E|| < 1e-2 for good (right-censored at T).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
from matplotlib.lines import Line2D  # noqa: E402

from experiments import keys, style  # noqa: E402

# ----------------------------------------------------------------------------- constants
PSHORT = {"linear_system": "linsys", "tv_unconstrained_opt": "tvuo", "tvqp_eq": "tvqp"}
PTITLE = {"linsys": "linear system ($n=8$)", "tvuo": "TV unconstr. opt. ($n=8$, nonlinear)",
          "tvqp": "TV-QP, equality ($n_x=8$)"}
PTEX = {"linsys": "linear system", "tvuo": "TV unconstr.\\ opt.", "tvqp": "TV-QP (eq.)"}
AFFINE = {"linsys": True, "tvuo": False, "tvqp": True}
FLOOR = 1e-12                  # round-off floor excluded from order fits
Q_TRANSIENT = 3.0              # max||E|| on [T/2, 3T/4) > 3 x max on the steady window -> still decaying
Q_GROWTH = 1.0 / 3.0           # ... < 1/3 x -> growing (finite instability)
DELTA_EQ = 0.5                 # equivalence margin (decades) for matched-cost ratios
QS_FACTOR = 3.0                # GNN quasi-static regime: gamma*sigma_min^2 >= 3*omega (local lag slope <= -0.9)
OPEN_LOOP = {"davidenko"}      # open-loop control: E is not driven to 0, so final > initial is drift, not instability
N_BOOT = 2000
BOOT_SEED = 20261018
TAU_FIX = 1e-2                 # fixed sampling period of the per-step view
BUDGET_TAUS = (1e-2, 1e-3)     # budgets = ZNN-Euler's FLOP rate at these tau
REF = "znn:euler"
FORMULA_LABEL = {"euler": "Euler", "taylor_zhang": "Taylor-Zhang", "zead_5i": "ZeaD-5i (=Uhlig 2_3)",
                 "uhlig_3_4": "Uhlig 3_4", "uhlig_4_6": "Uhlig 4_6", "zead_7i_cang2024": "ZeaD-7i (Cang 2024)"}
PC_LABEL = {"pc_newton": "PC-Newton", "pc_newton_causal": "PC-Newton causal",
            "pc_gradient": "PC-gradient", "pc_gradient_causal": "PC-gradient causal",
            "newton_tracking": "Newton tracking", "newton_tracking_causal": "Newton tracking causal"}
NONCAUSAL = {"pc_newton", "pc_gradient", "newton_tracking"}
# baseline-group panels: (keys.GROUPS['baseline'] entity, method id, level)
BASELINE_SERIES = [("znn", "znn:euler", "rec"), ("gnn", "gnn", None),
                   ("pc_newton_causal", "pc_newton_causal:C1", "single"),
                   ("pc_gradient_causal", "pc_gradient_causal:C1", "single"),
                   ("newton_tracking_causal", "newton_tracking_causal:C1", "single"),
                   ("davidenko", "davidenko", "single"), ("uhlig_3_4", "znn:uhlig_3_4", "rec")]


def load_config_module():
    spec = importlib.util.spec_from_file_location("_e9_config", ROOT / "experiments" / "configs" / "E9.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CFG = load_config_module()


# ----------------------------------------------------------------------------- helpers
def mlabel(m: str) -> str:
    if m.startswith("znn:"):
        f = m[4:]
        return "ZNN + " + FORMULA_LABEL.get(f, f)
    if m == "gnn":
        return "GNN + Euler"
    if m == "davidenko":
        return "Davidenko (open loop)"
    k, c = m.split(":C")
    return f"{PC_LABEL.get(k, k)} (C={c})" + (" [nc]" if k in NONCAUSAL else "")


def tex(s: str) -> str:
    return (str(s).replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("%", "\\%")
            .replace("&", "\\&").replace("#", "\\#"))


def is_noncausal(m: str) -> bool:
    return m.split(":")[0] in NONCAUSAL


def qstats(v):
    """median, q25, q75 with +inf (diverged) kept as +inf; NaN (failed) dropped."""
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return np.nan, np.nan, np.nan
    w = np.where(np.isinf(v), 1e300, v)
    q = np.quantile(w, [0.5, 0.25, 0.75])
    return tuple(float(np.inf) if x >= 1e299 else float(x) for x in q)


def med_inf(a, axis=-1):
    w = np.where(np.isinf(a), 1e300, a)
    m = np.nanmedian(w, axis=axis)
    return np.where(m >= 1e299, np.inf, m)


SLOPE_TOL = 0.15     # local slopes of an asymptotic range agree within 0.15 (REVIEW_CHECKLIST item 3)


def fit_order(x, y, floor=FLOOR, tol_abs=SLOPE_TOL, tol_rel=0.0, min_pts=3, exclude=None):
    """Empirical order p in y ~ x^p.

    Usable points: finite, above `floor` (round-off) and not flagged in `exclude`
    (pre-asymptotic cells: transient still decaying in the steady window, or finite growth).
    The asymptotic range is the longest window of >= min_pts *consecutive* usable grid points
    (a floor / excluded / diverged point breaks the run) whose consecutive (local) slopes all
    lie within max(tol_abs, tol_rel*|median|) of their median (default: 0.15 absolute, the
    project tolerance; the pre-review default max(0.5, 0.15|median|) accepted ranges whose local
    slopes differed by up to 1); ties go to the smallest x.  If no such window exists, the two
    smallest-x consecutive usable points are used (flag '2pt').
    Also returned: the all-points slope (every finite point above the floor, flagged cells
    included) and the local slopes of the chosen run."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    o = np.argsort(x)
    x, y = x[o], y[o]
    exc = np.zeros(len(x), bool) if exclude is None else np.asarray(exclude, bool)[o]
    above = np.isfinite(y) & (y > floor) & (x > 0)
    usable = above & ~exc
    out = {"p_asym": np.nan, "p_all": np.nan, "n_asym": 0, "n_all": int(above.sum()),
           "n_excluded": int((above & exc).sum()), "x_lo": np.nan, "x_hi": np.nan, "local": [], "flag": ""}
    if above.sum() >= 2:
        out["p_all"] = float(np.polyfit(np.log10(x[above]), np.log10(y[above]), 1)[0])
    if not usable.any():
        out["flag"] = "floor" if np.any(np.isfinite(y) & (y <= floor)) else "no usable point"
        return out
    segs, cur = [], []
    for k in range(len(x)):
        if usable[k]:
            cur.append(k)
        elif cur:
            segs.append(cur)
            cur = []
    if cur:
        segs.append(cur)
    best = None
    for seg in segs:
        lx, ly = np.log10(x[seg]), np.log10(y[seg])
        sl = np.diff(ly) / np.diff(lx) if len(seg) >= 2 else np.array([])
        for i in range(len(seg)):
            for j in range(i + min_pts - 1, len(seg)):
                w = sl[i:j]
                med = np.median(w)
                if np.all(np.abs(w - med) <= max(tol_abs, tol_rel * abs(med))):
                    cand = (seg[i], seg[j])
                    if best is None or (cand[1] - cand[0]) > (best[1] - best[0]):
                        best = cand
    if best is None:
        two = next((seg[:2] for seg in segs if len(seg) >= 2), None)
        if two is None:
            out["flag"] = "1 point"
            return out
        best = (two[0], two[1])
        out["flag"] = "2pt"
    i, j = best
    lx, ly = np.log10(x[i:j + 1]), np.log10(y[i:j + 1])
    out["p_asym"] = float(np.polyfit(lx, ly, 1)[0])
    out["local"] = [float(v) for v in np.diff(ly) / np.diff(lx)]
    out["n_asym"] = int(j - i + 1)
    out["x_lo"], out["x_hi"] = float(x[i]), float(x[j])
    if np.any(np.isfinite(y) & (y <= floor)):
        out["flag"] = (out["flag"] + " floor-hit").strip()
    return out


def bracket(xs, x_star):
    """Indices (i, j, w) of the two ADJACENT grid points of `xs` that bracket x_star in log scale
    (w = interpolation weight of j); (i, i, 0) on an exact grid hit; None outside the grid.
    The bracket is taken on the full grid, so interpolation never bridges over a grid point whose
    value is missing, diverged or unstable (the previous version dropped such points first and
    interpolated across them, e.g. from tau = 1e-2 to an unstable 1e9 value at 3e-2)."""
    xs = np.asarray(xs, float)
    lx = np.log10(xs)
    t = math.log10(x_star)
    o = np.argsort(lx)
    for k, i in enumerate(o):
        if abs(lx[i] - t) < 1e-9:
            return int(i), int(i), 0.0
    if t < lx[o[0]] or t > lx[o[-1]]:
        return None
    k = int(np.searchsorted(lx[o], t))
    i, j = int(o[k - 1]), int(o[k])
    return i, j, float((t - lx[i]) / (lx[j] - lx[i]))


def interp_at(Y, br):
    """log-log interpolation of Y (shape (n_grid,) or (n_grid, R)) on a bracket from `bracket`.
    +inf if a bracketing value is +inf (diverged / unstable), NaN if one is NaN or <= 0."""
    if br is None:
        return np.nan if np.ndim(Y) == 1 else np.full(np.shape(Y)[1], np.nan)
    i, j, w = br
    a, b = np.asarray(Y[i], float), np.asarray(Y[j], float)
    with np.errstate(all="ignore"):
        if i == j:
            out = np.where(a > 0, a, np.nan)
        else:
            la, lb = np.log10(np.where(a > 0, a, np.nan)), np.log10(np.where(b > 0, b, np.nan))
            out = 10 ** ((1 - w) * la + w * lb)
            out = np.where(np.isposinf(a) | np.isposinf(b), np.inf, out)
    return float(out) if np.ndim(out) == 0 else out


def interp_log(xs, ys, x_star):
    """Bracket-based log-log interpolation (no extrapolation, no bridging)."""
    return interp_at(np.asarray(ys, float), bracket(xs, x_star))


# ----------------------------------------------------------------------------- loading
def find_cost(results: Path):
    ct = results / "cost_task.json"
    cands = []
    if ct.exists():
        p = Path(json.loads(ct.read_text())["done"])
        if not p.exists():
            p = results / "cost" / p.name
        if p.exists():
            cands.append(p)
    if not cands:
        cands = sorted((results / "cost").glob("cost_profile-*.json"), key=lambda f: f.stat().st_mtime)[-1:]
        if cands and ct.exists():
            warnings.warn(f"the cost profile named in {ct} is missing; using the newest {cands[0].name}, which may "
                          "be stale (run the cost task of tasks_all.jsonl)")
    if cands:
        cost = json.loads(cands[0].read_text())
        if int(cost.get("cost_version", 0)) < 3:
            warnings.warn(f"{cands[0].name} has cost_version {cost.get('cost_version')} < 3: its flops_per_step "
                          "still charges the unused E of the Davidenko predictor")
        return cost, str(cands[0])
    warnings.warn("cost profile missing -- computing it in-process (E9_cost.compute)")
    sys.path.insert(0, str(ROOT / "experiments" / "tasks"))
    import E9_cost
    scale = "full" if "mini" not in str(results) else "mini"
    return E9_cost.compute(scale), "in-process"


def load(results: Path):
    from znnbench.runner import aggregate
    df = aggregate(results)
    if df.empty:
        raise SystemExit(f"no results under {results}")
    df["pshort"] = df["problem_key"].map(PSHORT)
    for c in ("disc_kw_C",):
        if c not in df.columns:
            df[c] = np.nan
    seeds_by_p = {p: sorted(int(s) for s in g["seed"].unique()) for p, g in df.groupby("pshort")}
    sbar = {p: CFG.sigma_bar(p, seeds_by_p[p]) for p in seeds_by_p}

    def ident(r):
        mk, dk = r["model_key"], r["disc_key"]
        if mk == "znn":
            hs = CFG.formula_gains(dk)
            lvl = "rec" if hs and math.isclose(r["h"], hs[0], rel_tol=1e-3) else "half"
            return f"znn:{dk}", lvl
        if mk == "gnn":
            if str(r["config"]).startswith("E9_gnnlag"):
                return "gnnlag", f"gamma={r['gamma']:g}"
            eta = r["h"] * sbar[r["pshort"]] ** 2
            e = min(CFG.GNN_ETAS, key=lambda v: abs(v - eta))
            return "gnn", f"eta={e:g}"
        if mk == "davidenko":
            if dk == "euler":
                return "davidenko", "single"
            return f"{dk}:C{int(r['disc_kw_C'])}", "single"
        return f"{mk}:{dk}", "single"

    ids = df.apply(ident, axis=1, result_type="expand")
    df["method"], df["level"] = ids[0], ids[1]
    df["primary"] = ((df["level"] == "rec") | (df["level"] == "single")
                     | (df["level"] == f"eta={CFG.GNN_ETA_PRIMARY:g}"))
    # finite but unstable: a closed-loop method whose residual grew over the run without crossing the
    # runner's divergence threshold (1e10 on |x|); uhlig_4_6 at tau = 3e-2 reaches ~1e9 this way.  These
    # trajectories are treated as DIVERGED (MSSRE = +inf) everywhere below ("effective status"); the
    # open-loop Davidenko control is exempt (its residual drifts around E(x0, 0) by design).
    open_loop = df["model_key"].isin(OPEN_LOOP) & (df["disc_key"] == "euler")
    df["unstable_finite"] = ((df["status"] == "ok") & (df["final_residual"] > df["initial_residual"])
                             & ~open_loop)
    df["status_eff"] = np.where(df["unstable_finite"], "unstable", df["status"])
    df["q_prev"] = steady_ratio(df, results)
    df["mssre_v"] = np.where(df["status_eff"].isin(["diverged", "unstable"]), np.inf,
                             np.where(df["status"] == "ok", df["mssre"], np.nan))
    return df, sbar, seeds_by_p


def trace_path(results: Path, config: str, fname: str):
    f = results / str(config) / "results" / str(fname)
    if f.exists():
        return f
    hits = list(results.rglob(Path(str(fname)).name))
    return hits[0] if hits else None


def steady_ratio(df, results: Path):
    """Per trajectory q = max||E|| on [T/2, 3T/4) / max||E|| on the steady window [3T/4, T],
    from the stored (down-sampled) residual traces: q >> 1 means the transient is still
    decaying inside the steady window, q << 1 means finite growth; NaN without traces."""
    import numpy as _np
    q = _np.full(len(df), _np.nan)
    if "traces_file" not in df.columns:
        return q
    pos = {ix: k for k, ix in enumerate(df.index)}
    for (cfg, fname), g in df[df["traces_file"].notna()].groupby(["config", "traces_file"]):
        f = trace_path(results, cfg, fname)
        if f is None:
            continue
        with _np.load(f) as z:
            for rid, gr in g.groupby("run_id"):
                kt, kr = f"{rid}__t", f"{rid}__residual"
                if kt not in z.files or kr not in z.files:
                    continue
                t, R = z[kt], z[kr].astype(float)
                T = float(t[-1])
                w_prev = (t >= 0.5 * T) & (t < (1 - CFG.STEADY_FRAC) * T)
                w_ss = t >= (1 - CFG.STEADY_FRAC) * T
                if w_prev.sum() < 2 or w_ss.sum() < 2:
                    continue
                with _np.errstate(all="ignore"):
                    ratio = R[w_prev].max(axis=0) / R[w_ss].max(axis=0)
                for ix, tr in zip(gr.index, gr["traj"].astype(int)):
                    if tr < ratio.size:
                        q[pos[ix]] = ratio[tr]
    return q


def time_to_eps(df, results: Path, eps: float):
    """Per trajectory: first stored time t_k with ||E(t_j)|| < eps for all j >= k (NaN without traces,
    +inf if never).  Traces are down-sampled (``traces.points``), so the resolution is T/points."""
    out = pd.Series(np.nan, index=df.index)
    for (cfg, fname), g in df[df["traces_file"].notna()].groupby(["config", "traces_file"]):
        f = trace_path(results, cfg, fname)
        if f is None:
            continue
        with np.load(f) as z:
            for rid, gr in g.groupby("run_id"):
                kt, kr = f"{rid}__t", f"{rid}__residual"
                if kt not in z.files or kr not in z.files:
                    continue
                t, R = z[kt], z[kr].astype(float)
                for ix, tr in zip(gr.index, gr["traj"].astype(int)):
                    if tr >= R.shape[1]:
                        continue
                    bad = ~(R[:, tr] < eps)
                    if bad[-1]:
                        out[ix] = np.inf
                    else:
                        last_bad = np.nonzero(bad)[0]
                        out[ix] = float(t[last_bad[-1] + 1]) if last_bad.size else float(t[0])
    return out


def cell_table(df):
    rows = []
    main = df[df["method"] != "gnnlag"]
    for (p, m, lvl, tau), g in main.groupby(["pshort", "method", "level", "tau"]):
        med, q1, q3 = qstats(g["mssre_v"])
        ms, _, _ = qstats(np.where(g["status_eff"].isin(["diverged", "unstable"]), np.inf, g["ss_sol_err"]))
        runs = g.drop_duplicates("run_id")
        qv = g.loc[g["status_eff"] == "ok", "q_prev"].to_numpy(float)
        q = float(np.nanmedian(qv)) if np.isfinite(qv).any() else np.nan
        frac_unst = float(g["unstable_finite"].mean())
        # pre-asymptotic: the median trajectory is still decaying (q > 3) or growing (q < 1/3) in the
        # steady window; cells whose median is +inf (diverged / unstable) are unusable anyway
        rows.append({"q_prev": q, "frac_unstable": frac_unst,
                     "pre_asym": bool(np.isfinite(q) and (q > Q_TRANSIENT or q < Q_GROWTH)),
                     "pre_reason": ("transient" if np.isfinite(q) and q > Q_TRANSIENT else
                                    "growth" if np.isfinite(q) and q < Q_GROWTH else ""),
                     "pshort": p, "method": m, "level": lvl, "tau": tau, "med": med, "q25": q1, "q75": q3,
                     "sol_med": ms, "n": int(len(g)), "n_div": int((g["status"] == "diverged").sum()),
                     "n_unstable": int(g["unstable_finite"].sum()),
                     "n_fail": int(g["status"].isin(["error", "timeout"]).sum()),
                     "primary": bool(g["primary"].iloc[0]),
                     "wall_per_step": float(np.nanmedian(runs["wall_per_step"])),
                     "traj_per_run": float(np.nanmedian(runs["B"] * runs["n_gains"]))})
    return pd.DataFrame(rows)


def _distinct_J_samples(m: str) -> int:
    """NEW data samples per step at which method m needs J (affine problems: J independent of x, so one
    LU factorisation per sample suffices).  Every step brings exactly one new sample: causal schemes
    work at t_k only; non-causal Newton tracking works at t_{k+1} only; the non-causal PC-Newton
    predicts at t_k -- the sample its previous step corrected at -- and corrects at t_{k+1}, so with an
    ideal cache it also factorises once per step (the pre-review count of 2 charged it twice)."""
    if m == "gnn":
        return 0
    return 1


def cost_lookup(cost):
    """{(problem, method): cost entry}.  Adds ``flops_lu_reuse``: the FLOPs per step if, on the affine
    problems (J = J(t)), every solve at the same data sample reused one LU factorisation.  The zoo's
    structured model (``costs.step_flops``) counts what the code does -- ``np.linalg.solve`` refactorises
    at every call -- which charges causal PC-Newton with C corrections C + 1 factorisations per step
    where an optimised implementation needs one.  ZNN, GNN and the nonlinear tvuo are unchanged."""
    import ast
    tab = {}
    for e in cost["entries"]:
        p, mk, dk, C = e["problem"], e["model"], e["disc"], e["C"]
        if mk == "znn":
            m = f"znn:{dk}"
        elif mk == "gnn":
            m = "gnn"
        elif dk == "euler":
            m = "davidenko"
        else:
            m = f"{dk}:C{int(C)}"
        e = dict(e)
        # code view = step_flops exactly as returned (charges the unused E of the Davidenko predictor);
        # flops_per_step = algorithmic view (E9_cost.py, COST_VERSION >= 3)
        e["flops_code_view"] = float(e.get("flops_structured", e["flops_per_step"]))
        e["flops_lu_reuse"] = e["flops_per_step"]
        raw = e.get("flops_structured_raw") or {}
        try:
            nfac = float(raw.get("factorizations", 0.0))
            cat = raw.get("by_category", {})
            cat = ast.literal_eval(cat) if isinstance(cat, str) else cat
            fac = float(cat.get("factor", 0.0))
        except (ValueError, SyntaxError, TypeError, AttributeError):
            nfac, fac = 0.0, 0.0
        e["flops_factor"] = fac
        if AFFINE.get(p) and e.get("cost_source") == "znnbench.costs.step_flops" and nfac > 0:
            extra = max(0.0, nfac - _distinct_J_samples(m))
            e["flops_lu_reuse"] = e["flops_per_step"] - extra * fac / nfac
        tab[(p, m)] = e
    return tab


# ----------------------------------------------------------------------------- bootstrap
def boot_indices(n_seeds, B, R=N_BOOT, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    si = rng.integers(0, n_seeds, size=(R, n_seeds))
    bi = rng.integers(0, B, size=(R, n_seeds, B))
    return si, bi


def cube(df_pm, taus, seeds, B):
    """MSSRE array (n_tau, n_seeds, B) for one (problem, method, level)."""
    V = np.full((len(taus), len(seeds), B), np.nan)
    ti = {t: i for i, t in enumerate(taus)}
    sidx = {s: i for i, s in enumerate(seeds)}
    for t, s, b, v in df_pm[["tau", "seed", "x0_index", "mssre_v"]].itertuples(index=False):
        if t in ti and s in sidx and b < B:
            V[ti[t], sidx[s], int(b)] = v
    return V


def boot_medians(V, si, bi):
    """(n_tau, R) bootstrap medians (hierarchical: seeds, then x0 within seed)."""
    nt, S, B = V.shape
    Vb = V[:, si[:, :, None], bi]               # (nt, R, S, B)
    return med_inf(Vb.reshape(nt, si.shape[0], S * B), axis=-1)


def _quantiles(v, qs):
    """Order-statistic quantiles (inverted CDF) that stay valid with +-inf entries."""
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return [np.nan] * len(qs)
    return [float(np.quantile(v, q, method="inverted_cdf")) for q in qs]


def verdict_of(lr, ci, y, y_ref, delta=DELTA_EQ):
    """Matched-cost verdict from the point log-ratio and a CI.  Equivalence (CI inside +-delta) and
    superiority (CI excludes 0) are reported together: '+0.47 [+0.47, +0.48]' is 'equivalent' AND
    'worse'.  A method at the round-off floor makes the ratio a bound on the true gain."""
    if not np.isfinite(lr) or np.isnan(ci[0]) or np.isnan(ci[1]):
        return "n/a"
    if np.isfinite(y) and np.isfinite(y_ref) and y < 10 * FLOOR and y_ref < 10 * FLOOR:
        return "both at round-off floor"
    eq = ci[0] >= -delta and ci[1] <= delta
    side = "better" if ci[1] < 0 else ("worse" if ci[0] > 0 else "")
    if eq:
        v = "equivalent" + (f" (but {side})" if side else "")
    elif side:
        v = side
    else:
        v = "inconclusive"
    if side == "better" and np.isfinite(y) and y < 10 * FLOOR:
        v += " (floor-limited: gain is a lower bound)"
    return v


def matched_cost(df, cells, cost_of, problems, seeds_by_p, B, cost_key="flops_per_step", n_family=None):
    """log10 MSSRE ratio to ZNN-Euler at the budget FLOP rates R* = cost(ZNN-Euler)/tau_b.

    Each method's MSSRE curve (median over trajectories at every tau, diverged / unstable = +inf) is
    interpolated in log-log between the two adjacent grid points that bracket R* (no extrapolation,
    no bridging).  The hierarchical bootstrap (seeds, then x0 within seed; paired across methods and
    tau) is pushed through the same interpolation; a resample whose bracketing median is +inf counts
    as log-ratio +inf (worse).  Percentile CIs use order statistics; a Bonferroni CI over the family
    of (problem, method) comparisons of one budget is reported alongside (2000 resamples make its
    extreme quantiles approximate)."""
    # only a decaying transient biases the steady value upwards; a non-quasi-static GNN cell is a genuine
    # (dynamic-lag) steady state and a growing cell is already +inf
    pre = {(r.pshort, r.method, r.tau): bool(r.pre_asym and "transient" in str(r.pre_reason))
           for r in cells[cells["primary"]].itertuples()}
    matched, budget_rate = {}, {}
    boot = {}
    for p in problems:
        seeds = seeds_by_p[p]
        si, bi = boot_indices(len(seeds), B)
        e_ref = cost_of.get((p, REF))
        if e_ref is None:
            continue
        budget_rate[p] = [e_ref[cost_key] / bt for bt in BUDGET_TAUS]
        sub = df[(df["pshort"] == p) & df["primary"] & (df["method"] != "gnnlag")]
        taus_p = sorted(sub["tau"].unique())
        for m, g in sub.groupby("method"):
            e = cost_of.get((p, m))
            if e is None:
                continue
            V = cube(g, taus_p, seeds, B)
            rates = e[cost_key] / np.array(taus_p)
            point = med_inf(V.reshape(len(taus_p), -1), axis=-1)
            boot[(p, m)] = (rates, point, boot_medians(V, si, bi), taus_p)
    n_family = n_family or max(1, sum(1 for (p, m) in boot if m != REF))
    a_b = 0.05 / n_family
    for p in problems:
        if (p, REF) not in boot:
            continue
        r_ref, pt_ref, bm_ref, _ = boot[(p, REF)]
        for bidx, R in enumerate(budget_rate[p]):
            br_ref = bracket(r_ref, R)
            y_ref = interp_at(pt_ref, br_ref)
            yb_ref = interp_at(bm_ref, br_ref)
            for (pp, m), (rates, pt, bm, taus_p) in boot.items():
                if pp != p:
                    continue
                br = bracket(rates, R)
                y = interp_at(pt, br)
                yb = interp_at(bm, br)
                with np.errstate(all="ignore"):
                    lr = math.log10(y / y_ref) if (np.isfinite(y) and np.isfinite(y_ref) and y > 0) else (
                        np.inf if np.isposinf(y) else np.nan)
                    lrb = np.log10(yb / yb_ref)
                lrb = np.where(np.isposinf(yb), np.inf, lrb)
                ci = tuple(_quantiles(lrb, [0.025, 0.975]))
                ci_b = tuple(_quantiles(lrb, [a_b / 2, 1 - a_b / 2]))
                if br is None:
                    v = vb = "n/a (outside the sampled tau range)"
                elif np.isposinf(lr):
                    v = vb = "n/a (diverged or unstable at a bracketing tau)"
                else:
                    v, vb = verdict_of(lr, ci, y, y_ref), verdict_of(lr, ci_b, y, y_ref)
                flags = []
                if br is not None:
                    i, j, w = br
                    used = [i] if i == j else [i, j]
                    if any(pre.get((p, m, taus_p[k]), False) for k in used):
                        flags.append("transient-limited bracket (MSSRE overestimates the steady value)")
                    if np.isfinite(y) and y < 10 * FLOOR:
                        flags.append("method at round-off floor")
                matched[(p, m, bidx)] = {"mssre": y, "mssre_ref": y_ref, "log_ratio": lr, "ci": ci,
                                         "ci_bonferroni": ci_b, "verdict": v, "verdict_bonferroni": vb,
                                         "flags": flags, "tau_at_budget": cost_of[(p, m)][cost_key] / R,
                                         "budget_rate": R, "n_family": n_family}
    return matched, budget_rate


def gnn_eta_sensitivity(df, cells, cost_of, problems, seeds_by_p, B, eta):
    """Matched-cost ratio of GNN at a non-primary eta level (the GNN rows at that level relabelled as
    the primary 'gnn' method, everything else unchanged); returns {problem: {budget index: entry}}."""
    lvl = f"eta={eta:g}"
    sub = df[((df["method"] == REF) & df["primary"]) | ((df["method"] == "gnn") & (df["level"] == lvl))].copy()
    if sub.empty or not (sub["method"] == "gnn").any():
        return {}
    sub["primary"] = True
    cs = cells[((cells["method"] == REF) & cells["primary"])
               | ((cells["method"] == "gnn") & (cells["level"] == lvl))].copy()
    cs["primary"] = True
    m, _ = matched_cost(sub, cs, cost_of, problems, seeds_by_p, B, "flops_per_step", n_family=1)
    return {p: {b: m[(p, "gnn", b)] for b in range(len(BUDGET_TAUS)) if (p, "gnn", b) in m} for p in problems}


def _expected(p, m):
    """Expected residual order of method m on problem p (None: exact on affine problems)."""
    if m.startswith("znn:"):
        return CFG.formula_stability(m[4:])["expected_order"]
    if m == "gnn":
        return 1
    if m == "davidenko":
        return 0
    from znnbench.registry import make_discretization
    k, C = m.split(":C")
    d = make_discretization(k, C=int(C))
    return d.expected_order if AFFINE[p] else d.expected_order_nonlinear


def gnn_regime(gamma, sigma_min, c_lag, E0, freq):
    """(transient_free, quasi_static) of linear-GNN tracking at gain gamma (continuous-time theory):
    transient-free  E0 exp(-gamma sigma_min^2 t_ss) < 0.1 c_lag / gamma;
    quasi-static    gamma sigma_min^2 >= QS_FACTOR * omega (a mode with rate a tracks a sinusoid with
                    error |xdot*| / sqrt(a^2 + omega^2): the 1/gamma law holds only for a >> omega)."""
    t_ss = CFG.T_HORIZON * (1 - CFG.STEADY_FRAC)
    a = gamma * sigma_min ** 2
    tf = a * t_ss > math.log(max(10 * E0 * gamma / c_lag, 1.0))
    qs = a >= QS_FACTOR * freq
    return bool(tf), bool(qs)


def problem_freq(p):
    return float(CFG.PROBLEMS[p]["kwargs"].get("freq", 1.0))


def kendall_floor(a, b):
    """Kendall tau_b with values at the round-off floor (< 10 x FLOOR) treated as ties and +inf
    (diverged / unstable) as tied-worst.  The p-value is descriptive only (no multiplicity control)."""
    from scipy.stats import kendalltau
    a = np.minimum(np.maximum(np.asarray(a, float), 10 * FLOOR), 1e300)
    b = np.minimum(np.maximum(np.asarray(b, float), 10 * FLOOR), 1e300)
    return kendalltau(a, b)


# ----------------------------------------------------------------------------- figures
def _save(fig, name, figdir):
    style.FIG_DIR = Path(figdir)
    return style.savefig(fig, name)


def _series(ax, x, c, sty, label, hollow=False, band=True):
    o = np.argsort(x)
    x = np.asarray(x)[o]
    med, q1, q3 = (np.asarray(c[k])[o] for k in ("med", "q25", "q75"))
    ok = np.isfinite(med) & (med > 0)
    if not ok.any():
        return None
    kw = dict(sty)
    if hollow:
        kw["markerfacecolor"] = "none"
    ln, = ax.plot(x[ok], med[ok], label=label, **kw)
    okb = ok & np.isfinite(q1) & np.isfinite(q3) & (q1 > 0)
    if band and okb.sum() >= 2:
        ax.fill_between(x[okb], q1[okb], q3[okb], color=sty["color"], alpha=0.13, linewidth=0)
    return ln


def formulas_present(cells):
    fs = [m[4:] for m in cells["method"].unique() if m.startswith("znn:")]
    order = keys.GROUPS["disc"]
    return sorted(fs, key=lambda f: (order.index(f) if f in order else 99, f))[:8]


PSHORT_TITLE = {"linsys": "linear system", "tvuo": "TV unconstr. opt.", "tvqp": "TV-QP (eq.)"}
Y_CLIP = 1e3          # finite-but-unstable runs reach 1e9; the axis is clipped here (line leaves the top)


def _clip_y(ax):
    lo, hi = ax.get_ylim()
    if hi > Y_CLIP:
        ax.set_ylim(top=Y_CLIP)


def _bottom_legend(fig, handles, ncol=4, y=0.0):
    if len(handles) >= 2:
        fig.legend(list(handles.values()), list(handles.keys()), loc="upper center",
                   bbox_to_anchor=(0.5, y), ncol=ncol, frameon=False)


def _group_series(group, cells):
    """[(entity, method id, plot kwargs)] of one style group (never mixed within a figure)."""
    if group == "disc":
        return [(f, f"znn:{f}", keys.style_in_group("disc", f)) for f in formulas_present(cells)]
    return [(e, m, keys.style_in_group("baseline", e)) for e, m, _ in BASELINE_SERIES]


PAPER_WORKPREC = {"fig_E9_workprec_znn"}       # drawn at the paper's text width (fig_workprec_paper)
PAPER_PTITLE = {"linsys": "linear system ($n=8$)", "tvuo": "TV unconstrained opt. ($n=8$)",
                "tvqp": "TV-QP, equality ($n_x=8$)"}


def _pareto_front(x, y):
    """Indices of the points not dominated in (cost x, error y), sorted by cost."""
    o = np.lexsort((y, x))
    keep, best = [], np.inf
    for i in o:
        if y[i] < best:
            keep.append(i)
            best = y[i]
    return np.asarray(keep, dtype=int)


def fig_workprec_paper(cells, cost_of, problems, group, name, figdir):
    """Paper figure: median MSSRE (IQR band) against FLOPs per tracked second, one panel per
    problem, with the Pareto front of the drawn medians (attainment staircase, behind the series)
    and the two matched-cost budgets."""
    with style.print_rc():
        fig, axes = plt.subplots(1, len(problems), figsize=(style.TEXTWIDTH, 2.75), squeeze=False,
                                 constrained_layout=True)
        fig.get_layout_engine().set(w_pad=0.03, h_pad=0.03, wspace=0.03)
        handles = {}
        for j, p in enumerate(problems):
            ax = axes[0, j]
            cp = cells[cells["pshort"] == p]
            px, py = [], []
            for ent, m, sty in _group_series(group, cells):
                c = cp[(cp["method"] == m) & cp["primary"]]
                e = cost_of.get((p, m))
                if c.empty or e is None:
                    continue
                x = e["flops_per_step"] / c["tau"].to_numpy()
                sty = {**sty, "markersize": 3.6 if sty.get("marker") not in ("P", "X", "*") else 4.2}
                lab = (mlabel(m).replace("ZNN + ", "") if m.startswith("znn:") else mlabel(m)).replace(
                    "Taylor-Zhang", "Taylor\u2013Zhang")
                ln = _series(ax, x, c, sty, lab)
                if ln is None:
                    continue
                handles.setdefault(lab, ln)
                med = c["med"].to_numpy()
                ok = np.isfinite(med) & (med > 0)
                px += list(np.asarray(x)[ok])
                py += list(med[ok])
                pa_mask = c["pre_asym"].to_numpy() & np.isfinite(med)
                if pa_mask.any():
                    ax.plot(np.asarray(x)[pa_mask], med[pa_mask], linestyle="none", marker=sty["marker"],
                            color=sty["color"], markerfacecolor="white", markersize=sty["markersize"])
            ax.set_xscale("log")
            ax.set_yscale("log")
            _clip_y(ax)
            if px:
                px, py = np.asarray(px), np.asarray(py)
                k = _pareto_front(px, py)
                fx, fy = px[k], py[k]
                xx = np.repeat(fx, 2)[1:]
                yy = np.repeat(fy, 2)[:-1]
                fr, = ax.plot(xx, yy, color="#aaa9a5", lw=3.0, alpha=0.8, zorder=1.5, solid_capstyle="butt",
                              solid_joinstyle="miter")
                handles.setdefault("Pareto front (lowest MSSRE at each cost)", fr)
            ax.set_xlabel("FLOPs per tracked second")
            e_ref = cost_of.get((p, REF))
            if e_ref is not None:                # the matched-cost budgets of the tables / fig_E9_equiv
                for bt in BUDGET_TAUS:
                    ax.axvline(e_ref["flops_per_step"] / bt, color=style.GREY, lw=0.7, ls=(0, (2, 2)), zorder=0)
                handles.setdefault("budgets: Euler-ZNN at $\\tau$ = " + ", ".join(f"{bt * 1e3:g}" for bt in BUDGET_TAUS)
                                   + " ms", Line2D([], [], color=style.GREY, lw=0.7, ls=(0, (2, 2))))
            ax.set_title(PAPER_PTITLE.get(p, PTITLE[p]), fontsize=8)
            ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, numticks=4))
            ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, numticks=5))
            if j == 0:
                ax.set_ylabel("MSSRE (median, IQR)")
                ax.annotate("", xy=(0.05, 0.05), xytext=(0.19, 0.22), xycoords="axes fraction",
                            textcoords="axes fraction",
                            arrowprops=dict(arrowstyle="-|>", color=style.INK2, lw=0.9, shrinkA=0, shrinkB=0,
                                            mutation_scale=8))
                ax.text(0.06, 0.25, "better", transform=ax.transAxes, ha="left", va="bottom", fontsize=8,
                        color=style.INK2)
        handles["white fill: pre-asymptotic cell"] = Line2D(
            [], [], marker="o", color=style.INK, markerfacecolor="white", linestyle="none", markersize=3.6)
        fig.legend(list(handles.values()), list(handles.keys()), loc="outside lower center", ncol=3,
                   frameon=False, columnspacing=1.2, handlelength=2.2)
        return _save(fig, name, figdir)


def fig_lines(cells, cost_of, orders, problems, group, xmode, name, figdir):
    """1 x len(problems) panels of median MSSRE (IQR band) for one style group.
    xmode: 'flops' (FLOPs per tracked second), 'wall' (compute time / tracked time) or 'tau'."""
    if name in PAPER_WORKPREC and xmode == "flops":
        return fig_workprec_paper(cells, cost_of, problems, group, name, figdir)
    fig, axes = plt.subplots(1, len(problems), figsize=(style.DOUBLE, 2.55), squeeze=False)
    handles = {}
    for j, p in enumerate(problems):
        ax = axes[0, j]
        cp = cells[cells["pshort"] == p]
        for ent, m, sty in _group_series(group, cells):
            c = cp[(cp["method"] == m) & cp["primary"]]
            if c.empty:
                continue
            if xmode == "flops":
                e = cost_of.get((p, m))
                if e is None:
                    continue
                x = e["flops_per_step"] / c["tau"].to_numpy()
            elif xmode == "wall":
                x = c["wall_per_step"].to_numpy() / c["tau"].to_numpy()
            else:
                x = c["tau"].to_numpy()
            lab = mlabel(m)
            if xmode == "tau":
                pa = orders.get((p, m), {}).get("p_asym", np.nan)
                if np.isfinite(pa):
                    ax.annotate(f"{pa:.1f}", (c["tau"].min(), c.loc[c["tau"].idxmin(), "med"]),
                                fontsize=5.5, color=sty["color"], xytext=(-3, 0), textcoords="offset points",
                                ha="right", va="center")
            ln = _series(ax, x, c, sty, lab)
            if ln is not None:
                handles.setdefault(lab, ln)
                pa_mask = c["pre_asym"].to_numpy() & np.isfinite(c["med"].to_numpy())
                if pa_mask.any():
                    ax.plot(np.asarray(x)[pa_mask], c["med"].to_numpy()[pa_mask], linestyle="none",
                            marker=sty["marker"], color=sty["color"], markerfacecolor="white")
        ax.set_xscale("log")
        ax.set_yscale("log")
        _clip_y(ax)
        if xmode == "tau":
            ax.axhline(FLOOR, color=style.GREY, lw=0.6, ls=":")
            ax.set_xlabel("sampling period $\\tau$ (s)")
            tmin = cp["tau"].min()
            ax.set_xlim(left=tmin / 2.6)          # room for the fitted-order labels
        elif xmode == "flops":
            ax.set_xlabel("FLOPs per tracked second")
            e_ref = cost_of.get((p, REF))
            if e_ref is not None:                # the matched-cost budgets of the tables / fig_E9_equiv
                for bt in BUDGET_TAUS:
                    ax.axvline(e_ref["flops_per_step"] / bt, color=style.GREY, lw=0.7, ls=(0, (2, 2)), zorder=0)
                handles.setdefault("budgets: ZNN-Euler at $\\tau$ = " + ", ".join(f"{bt:g}" for bt in BUDGET_TAUS)
                                   + " s", Line2D([], [], color=style.GREY, lw=0.7, ls=(0, (2, 2))))
        else:      # descriptive only: timed inside a parallel memguard pool on a shared machine
            ax.axvline(1.0, color=style.GREY, lw=0.6, ls=":")
            ax.set_xlabel("compute time / tracked time (descriptive)")
        ax.set_title(PTITLE[p])
        if j == 0:
            ax.set_ylabel("MSSRE (median, IQR)")
    handles["white fill: pre-asymptotic cell" + (" (not fitted)" if xmode == "tau" else "")] = Line2D(
        [], [], marker="o", color=style.INK, markerfacecolor="white", linestyle="none")
    _bottom_legend(fig, handles, ncol=4)
    fig.tight_layout()
    return _save(fig, name, figdir)


def fig_fixedtau(cells, cost_of, problems, tau_fix, figdir):
    """Per-step view at a fixed sampling period: MSSRE vs FLOPs/step, baseline group;
    PC / Newton-tracking variants drawn as C = 1, 2, 3 lines."""
    fig, axes = plt.subplots(1, len(problems), figsize=(style.DOUBLE, 2.55), squeeze=False)
    handles = {}
    for j, p in enumerate(problems):
        ax = axes[0, j]
        cp = cells[(cells["pshort"] == p) & np.isclose(cells["tau"], tau_fix)]
        for ent, m, lvl in BASELINE_SERIES:
            sty = keys.style_in_group("baseline", ent)
            ms = [f"{m[:-3]}:C{C}" for C in CFG.PC_CS] if m.endswith(":C1") else [m]
            xs, ys, lo, hi, pa = [], [], [], [], []
            for mm in ms:
                c = cp[(cp["method"] == mm) & cp["primary"]]
                e = cost_of.get((p, mm))
                if c.empty or e is None or not np.isfinite(c["med"].iloc[0]):
                    continue
                xs.append(e["flops_per_step"])
                ys.append(c["med"].iloc[0])
                lo.append(c["q25"].iloc[0])
                hi.append(min(c["q75"].iloc[0], 1e300))
                pa.append(bool(c["pre_asym"].iloc[0]) and "transient" in str(c["pre_reason"].iloc[0]))
            if not xs:
                continue
            ys, lo, hi, pa = np.array(ys), np.array(lo), np.array(hi), np.array(pa)
            yerr = np.vstack([np.clip(ys - lo, 0, None), np.clip(hi - ys, 0, None)])
            lab = mlabel(m).replace(" (C=1)", " (C=1,2,3)") if len(ms) > 1 else mlabel(m)
            ln = ax.errorbar(xs, ys, yerr=yerr, capsize=1.5, elinewidth=0.6, label=lab, **sty)
            if pa.any():
                ax.plot(np.array(xs)[pa], ys[pa], linestyle="none", marker=sty["marker"], color=sty["color"],
                        markerfacecolor="white", zorder=3)      # errorbar markers sit at zorder 2.1
            handles.setdefault(lab, ln)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([300, 500, 1000, 2000]))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel("FLOPs per step (per trajectory)")
        ax.set_title(f"{PTITLE[p]}, $\\tau$ = {tau_fix:g} s")
        _clip_y(ax)
        if j == 0:
            ax.set_ylabel("MSSRE at fixed $\\tau$")
    handles["white fill: still decaying at T"] = Line2D(
        [], [], marker="o", color=style.INK, markerfacecolor="white", linestyle="none")
    _bottom_legend(fig, handles, ncol=4)
    fig.tight_layout()
    return _save(fig, "fig_E9_fixedtau", figdir)


def fig_fixedtau_znn(cells, cost_of, problems, tau_fix, figdir):
    """ZNN look-ahead formulas at a fixed tau: same cost (one solve per step), accuracy
    spread over decades; filled h_opt, hollow h_opt/2 (disc group)."""
    fig, axes = plt.subplots(1, len(problems), figsize=(style.DOUBLE, 2.45), squeeze=False)
    handles = {}
    for j, p in enumerate(problems):
        ax = axes[0, j]
        cp = cells[(cells["pshort"] == p) & np.isclose(cells["tau"], tau_fix)]
        for f in formulas_present(cells):
            m = f"znn:{f}"
            sty = keys.style_in_group("disc", f)
            e = cost_of.get((p, m))
            if e is None:
                continue
            for lvl, hollow in (("rec", False), ("half", True)):
                c = cp[(cp["method"] == m) & (cp["level"] == lvl)]
                if c.empty or not np.isfinite(c["med"].iloc[0]):
                    continue
                kw = dict(sty, linestyle="none")
                if hollow:
                    kw["markerfacecolor"] = "none"
                y = c["med"].iloc[0]
                yerr = [[max(y - c["q25"].iloc[0], 0)], [max(min(c["q75"].iloc[0], 1e300) - y, 0)]]
                x = e["flops_per_step"] + (6.0 if hollow else 0.0)
                ln = ax.errorbar([x], [y], yerr=yerr, capsize=1.5, elinewidth=0.6, **kw)
                if bool(c["pre_asym"].iloc[0]) and "transient" in str(c["pre_reason"].iloc[0]):   # ring: decaying
                    ax.plot([x], [y], linestyle="none", marker="o", markersize=8, markerfacecolor="none",
                            markeredgecolor=style.GREY, markeredgewidth=0.6, zorder=3)
                if not hollow:
                    handles.setdefault(mlabel(m), ln)
        ax.set_yscale("log")
        _clip_y(ax)
        ax.set_xlabel("FLOPs per step (per trajectory)")
        ax.set_title(f"{PTITLE[p]}, $\\tau$ = {tau_fix:g} s")
        if j == 0:
            ax.set_ylabel("MSSRE at fixed $\\tau$")
    handles["filled: $h=h_{opt}$"] = Line2D([], [], marker="o", color=style.INK, linestyle="none")
    handles["hollow: $h=h_{opt}/2$"] = Line2D([], [], marker="o", color=style.INK, markerfacecolor="none",
                                              linestyle="none")
    handles["grey ring: still decaying at T"] = Line2D(
        [], [], marker="o", markersize=8, markerfacecolor="none", markeredgecolor=style.GREY, linestyle="none")
    _bottom_legend(fig, handles, ncol=4)
    fig.tight_layout()
    return _save(fig, "fig_E9_fixedtau_znn", figdir)


def fig_pc(cells, problems, figdir):
    groups = [("Newton prediction-correction", "pc_newton"), ("gradient prediction-correction", "pc_gradient"),
              ("Newton tracking (no prediction)", "newton_tracking")]
    shades = [style.SEQ_BLUE[5], style.SEQ_BLUE[8], style.SEQ_BLUE[12]]
    fig, axes = plt.subplots(len(problems), 3, figsize=(style.DOUBLE, 1.95 * len(problems) + 0.4),
                             squeeze=False, sharex=True)
    handles = {}
    for i, p in enumerate(problems):
        cp = cells[cells["pshort"] == p]
        for j, (title, base) in enumerate(groups):
            ax = axes[i, j]
            ymin = np.inf
            for causal in (True, False):
                curves = []
                for ci, C in enumerate(CFG.PC_CS):
                    m = f"{base}{'_causal' if causal else ''}:C{C}"
                    c = cp[cp["method"] == m].sort_values("tau")
                    if c.empty:
                        continue
                    sty = {"color": shades[ci % 3], "marker": style.MARKERS[ci],
                           "linestyle": "-" if causal else "--"}
                    lab = f"C={C}, {'causal' if causal else 'non-causal'}"
                    ln = _series(ax, c["tau"].to_numpy(), c, sty, lab, hollow=not causal, band=False)
                    if ln is not None:
                        handles.setdefault(lab, ln)
                        curves.append(c["med"].to_numpy())
                        ymin = min(ymin, np.nanmin(c["med"].to_numpy()))
                if len(curves) == len(CFG.PC_CS) and all(len(v) == len(curves[0]) for v in curves):
                    a = np.array(curves)
                    if np.all(np.isfinite(a)) and np.all(np.abs(np.log10(a / a[0])) < 0.01):
                        ax.text(0.97, 0.56 if causal else 0.46,
                                f"{'causal' if causal else 'non-causal'}: C=1,2,3 coincide",
                                transform=ax.transAxes, fontsize=5.5, color=style.INK2, ha="right")
            if ymin < 1e-10:
                ax.axhline(FLOOR, color=style.GREY, lw=0.6, ls=":")
            ax.set_xscale("log")
            ax.set_yscale("log")
            if i == 0:
                ax.set_title(title, fontsize=7.5)
            if j == 0:
                ax.set_ylabel(f"{PSHORT_TITLE[p]}\nMSSRE (median)")
            if i == len(problems) - 1:
                ax.set_xlabel("$\\tau$ (s)")
    _bottom_legend(fig, handles, ncol=6)
    fig.tight_layout()
    return _save(fig, "fig_E9_pc", figdir)


def pstyle(i):
    return {"color": style.CAT[i], "marker": style.MARKERS[i], "linestyle": style.LINESTYLES[i]}


def fig_gnn(gnn, problems, figdir):
    fig, axes = plt.subplots(1, 3, figsize=(style.DOUBLE, 2.45))
    handles = {}
    # (a) lag floor
    ax = axes[0]
    for i, p in enumerate(problems):
        L = gnn["lag"].get(p)
        if not L:
            continue
        g, med, used = (np.array(L[k], float) for k in ("gamma", "med", "used"))
        sty = pstyle(i)
        ok = np.isfinite(med)
        ln, = ax.plot(g[ok], med[ok], **{**sty, "markerfacecolor": "none"})
        handles.setdefault(PTITLE[p], ln)
        u = ok & (used > 0)
        ax.plot(g[u], med[u], linestyle="none", marker=sty["marker"], color=sty["color"])
        qsm = np.array(L.get("quasi_static", [1] * len(g))) > 0     # theory valid only when quasi-static
        if qsm.sum() >= 1:
            gg = np.geomspace(g[qsm].min(), g.max(), 20)
            ax.plot(gg, L["c_lag"] / gg, color=sty["color"], lw=0.7, ls=(0, (1, 1)))
    handles["theory $c_{lag}/\\gamma$, quasi-static range (dotted)"] = Line2D([], [], color=style.GREY, lw=0.7,
                                                                      ls=(0, (1, 1)))
    handles["hollow/white: transient or non-quasi-static gain (excluded from fits)"] = Line2D(
        [], [], marker="o", color=style.INK, markerfacecolor="none", linestyle="none")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$\\gamma$ (fixed-$\\gamma$, $\\tau$ = %g s)" % gnn["tau_lag"])
    ax.set_ylabel("MSSRE (median)")
    ax.set_title("(a) GNN lag floor")
    # (b) stability vs eta
    ax = axes[1]
    for i, p in enumerate(problems):
        S = gnn["stability"].get(p)
        if S:
            ax.plot(S["eta"], S["frac_div"], **pstyle(i))
    ax.axvline(2.0, color=style.GREY, lw=0.8, ls=":")
    ax.text(2.03, 0.45, "$\\eta = 2$", fontsize=6, color=style.INK2)
    ax.set_xlabel("$\\eta = \\gamma\\tau\\bar\\sigma^2$")
    ax.set_ylabel("fraction diverged or growing")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("(b) explicit GNN step limit")
    # (c) tau sweep at eta primary
    ax = axes[2]
    for i, p in enumerate(problems):
        S = gnn["tau_sweep"].get(p)
        if not S:
            continue
        t, med = np.array(S["tau"]), np.array(S["med"])
        ok = np.isfinite(med)
        ax.plot(t[ok], med[ok], **pstyle(i))
        pa = ok & (np.array(S.get("pre_asym", [0] * len(t))) > 0)
        if pa.any():
            ax.plot(t[pa], med[pa], linestyle="none", marker=pstyle(i)["marker"], color=pstyle(i)["color"],
                    markerfacecolor="white")
    ts = np.array(sorted({t for S in gnn["tau_sweep"].values() for t in S["tau"]}))
    if ts.size:
        top = np.nanmin([np.nanmax(S["med"]) for S in gnn["tau_sweep"].values()])
        ax.plot(ts, 0.5 * top * ts / ts.max(), color=style.GREY, lw=0.7, ls="--")
        handles["slope 1 (panel c)"] = Line2D([], [], color=style.GREY, lw=0.7, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$\\tau$ (s)")
    ax.set_ylabel("MSSRE (median)")
    ax.set_title(f"(c) GNN, fixed $\\eta$ = {CFG.GNN_ETA_PRIMARY:g}")
    _bottom_legend(fig, handles, ncol=3)
    fig.tight_layout()
    return _save(fig, "fig_E9_gnn", figdir)


def fig_equiv(matched, problems, methods, figdir):
    fig, axes = plt.subplots(1, len(problems), figsize=(style.DOUBLE, 0.17 * len(methods) + 1.3),
                             sharey=True, squeeze=False)
    axes = axes[0]
    y = np.arange(len(methods))[::-1]
    nc_rows = [k for k, m in enumerate(methods) if is_noncausal(m)]
    for j, p in enumerate(problems):
        ax = axes[j]
        ax.axvspan(-DELTA_EQ, DELTA_EQ, color=style.GREY, alpha=0.15, lw=0)
        ax.axvline(0, color=style.INK2, lw=0.6)
        if nc_rows:                      # non-causal block (t_{k+1} data): shaded and separated
            y_top = y[min(nc_rows)] + 0.5
            ax.axhspan(y[max(nc_rows)] - 0.5, y_top, color=style.GREY, alpha=0.08, lw=0)
            ax.axhline(y_top, color=style.INK2, lw=0.6, ls="--")
            if j == 0:      # right edge: the first non-causal row (PC-Newton [nc]) sits far left
                ax.text(0.985, y_top - 0.12, "non-causal ($t_{k+1}$ data)", transform=ax.get_yaxis_transform(),
                        fontsize=5.5, color=style.INK2, va="top", ha="right")
        for bi, (mk, fc, dy) in enumerate((("o", style.INK, 0.15), ("s", "none", -0.15))):
            for k, m in enumerate(methods):
                r = matched.get((p, m, bi))
                if not r:
                    continue
                if not np.isfinite(r["log_ratio"]):
                    if "unstable" in r["verdict"]:
                        ax.text(0.02, y[k] + dy, "unst." + (" (budget 2)" if bi else ""), fontsize=5,
                                color=style.INK2, va="center", ha="left",
                                transform=ax.get_yaxis_transform())
                    continue
                lo, hi = r["ci"]
                col = style.INK2 if bi else style.INK
                if np.isfinite(hi):
                    ax.plot([lo, hi], [y[k] + dy] * 2, color=col, lw=0.8)
                else:                                    # open-ended CI (some resamples unstable): arrow
                    ax.annotate("", xy=(r["log_ratio"] + 2.0, y[k] + dy), xytext=(lo, y[k] + dy),
                                arrowprops=dict(arrowstyle="->", color=col, lw=0.8))
                ax.plot(r["log_ratio"], y[k] + dy, marker=mk, markerfacecolor=fc, color=style.INK,
                        linestyle="none", markersize=3.5)
                if any(f.startswith("transient") for f in r["flags"]):
                    ax.annotate("t", (r["log_ratio"], y[k] + dy), xytext=(2.5, 1.5), textcoords="offset points",
                                fontsize=5, color=style.INK2)
        ax.set_title(PTITLE[p])
        ax.set_yticks(y)
        ax.set_yticklabels([mlabel(m) for m in methods], fontsize=6)
        ax.grid(axis="y", visible=False)
    fig.supxlabel("log$_{10}$[MSSRE / MSSRE(ZNN + Euler)] at the same FLOPs per tracked second", fontsize=8)
    handles = {f"budget = ZNN + Euler at $\\tau$ = {BUDGET_TAUS[0]:g} s":
               Line2D([], [], marker="o", color=style.INK, linestyle="none"),
               f"budget = ZNN + Euler at $\\tau$ = {BUDGET_TAUS[1]:g} s":
               Line2D([], [], marker="s", markerfacecolor="none", color=style.INK, linestyle="none"),
               f"$\\pm${DELTA_EQ} decade (equivalence margin)": plt.Rectangle((0, 0), 1, 1, color=style.GREY,
                                                                          alpha=0.15),
               "t: transient-limited bracket (conservative); arrow: CI open (unstable resamples)":
               Line2D([], [], color=style.INK2, lw=0.8)}
    fig.tight_layout()
    _bottom_legend(fig, handles, ncol=2, y=0.0)
    return _save(fig, "fig_E9_equiv", figdir)


def fig_traces(df, results, problems, tau_fix, figdir):
    from znnbench.runner import load_traces
    fig, axes = plt.subplots(1, len(problems), figsize=(style.DOUBLE, 2.4), squeeze=False, sharey=True)
    handles = {}
    for j, p in enumerate(problems):
        ax = axes[0, j]
        for ent, m, lvl in BASELINE_SERIES:
            g = df[(df["pshort"] == p) & (df["method"] == m) & df["primary"] & np.isclose(df["tau"], tau_fix)
                   & (df["seed"] == df["seed"].min()) & (df["x0_index"] == 0)]
            if g.empty or not isinstance(g["traces_file"].iloc[0], str):
                continue
            r = g.iloc[0]
            f = trace_path(results, r["config"], r["traces_file"])
            if f is None:
                continue
            tr = load_traces(f, r["run_id"])
            if "residual" not in tr:
                continue
            col = int(r["traj"])
            sty = keys.style_in_group("baseline", ent)
            ln, = ax.plot(tr["t"], np.maximum(tr["residual"][:, col], 1e-16), color=sty["color"],
                          linestyle=sty["linestyle"], lw=0.9, label=mlabel(m))
            handles.setdefault(mlabel(m), ln)
        ax.set_yscale("log")
        ax.axvspan(CFG.T_HORIZON * (1 - CFG.STEADY_FRAC), CFG.T_HORIZON, color=style.GREY, alpha=0.1, lw=0)
        ax.set_title(f"{PTITLE[p]}, $\\tau$ = {tau_fix:g} s")
        ax.set_xlabel("t (s)")
        if j == 0:
            ax.set_ylabel("$\\|E(x_k, t_k)\\|_2$ (seed 0, $x_0$ #0)")
    if len(handles) >= 2:
        fig.legend(list(handles.values()), list(handles.keys()), loc="upper center", bbox_to_anchor=(0.5, 0.0),
                   ncol=4, frameon=False)
    fig.tight_layout()
    return _save(fig, "fig_E9_traces", figdir)


# ----------------------------------------------------------------------------- tables
def write_tex(path: Path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def fmt_e(v, digits=1):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "--" if (v is None or np.isnan(v)) else "div."
    return f"{v:.{digits}e}".replace("e-0", "e-").replace("e+0", "e")


# ----------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--figdir", required=True)
    ap.add_argument("--tabdir", required=True)
    a = ap.parse_args(argv)
    results, figdir, tabdir = Path(a.results), Path(a.figdir), Path(a.tabdir)
    figdir.mkdir(parents=True, exist_ok=True)
    tabdir.mkdir(parents=True, exist_ok=True)

    df, sbar, seeds_by_p = load(results)
    cost, cost_path = find_cost(results)
    cost_of = cost_lookup(cost)
    problems = [p for p in CFG.PROBLEM_ORDER if p in set(df["pshort"])]
    cells = cell_table(df)
    # GNN theory per problem (continuous-time lag constant, slowest mode, initial residual) and the
    # theory-based transient / quasi-static flags of the fixed-eta tau sweep: the trace ratio q misses
    # the slow GNN transients (decay over [T/2, T] by less than 3x), so GNN cells are flagged
    # pre-asymptotic when gamma = eta / (tau sigma_bar^2) is not transient-free and quasi-static.
    gnn_theory = {}
    for p in sorted(set(df["pshort"])):
        jb = [CFG.jacobian_bounds(p, s) for s in seeds_by_p[p]]
        E0p = float(np.nanmedian(df.loc[df["pshort"] == p, "initial_residual"]))
        gnn_theory[p] = {"sigma_min": float(min(j["sigma_min"] for j in jb)),
                         "c_lag": float(np.median([j["c_lag"] for j in jb])), "E0": E0p,
                         "freq": problem_freq(p)}
    gmask = cells["method"] == "gnn"
    for ix in cells.index[gmask]:
        r = cells.loc[ix]
        th = gnn_theory[r["pshort"]]
        eta = float(str(r["level"]).split("=")[1])
        gam = eta / (r["tau"] * sbar[r["pshort"]] ** 2)
        tf, qs = gnn_regime(gam, th["sigma_min"], th["c_lag"], th["E0"], th["freq"])
        if not (tf and qs):
            cells.loc[ix, "pre_asym"] = True
            cells.loc[ix, "pre_reason"] = ("transient" if not tf else "") + ("" if qs else " non-quasi-static")
    taus_all = [float(t) for t in sorted(cells["tau"].unique())]
    tau_fix = min(taus_all, key=lambda t: abs(math.log(t / TAU_FIX)))
    status_counts = df["status"].value_counts().to_dict()
    methods_all = sorted(cells["method"].unique(), key=lambda m: (not m.startswith("znn:"), m != "gnn", m))
    summary = {"results": str(results), "cost_profile": cost_path, "cost_source": sorted({e["cost_source"] for e in cost["entries"]}),
               "structured_cost_model": bool(cost.get("structured_model")), "n_rows": int(len(df)),
               "status_counts": {k: int(v) for k, v in status_counts.items()},
               "status_counts_effective": {k: int(v) for k, v in df["status_eff"].value_counts().items()},
               "taus": taus_all,
               "seeds": {p: s for p, s in seeds_by_p.items()}, "B": int(df["B"].max()),
               "sigma_bar": sbar, "problems": problems}
    uf = df[df["unstable_finite"] & (df["method"] != "gnnlag")]
    summary["unstable_finite"] = {
        "definition": "closed-loop method, status ok but final residual > initial residual (growth below the "
                      "1e10 divergence threshold); treated as diverged (MSSRE = +inf) in every statistic; the "
                      "open-loop Davidenko control is exempt",
        "n_trajectories": int(len(uf)),
        "by_method_tau": {f"{p}|{m}|{lvl}|{t:g}": int(n) for (p, m, lvl, t), n in
                          uf.groupby(["pshort", "method", "level", "tau"]).size().items()}}

    summary["formula_stability"] = {
        m[4:]: {k: v for k, v in CFG.formula_stability(m[4:]).items() if k in ("h_opt", "h_max", "rho_opt",
                                                                                 "expected_order")}
        for m in methods_all if m.startswith("znn:")}

    # ---------------- orders
    orders = {}
    for (p, m), c in cells[cells["primary"]].groupby(["pshort", "method"]):
        orders[(p, m)] = fit_order(c["tau"], c["med"], exclude=c["pre_asym"])

    expected = _expected

    # implausible slopes are flagged, never reported bare (REVIEW_CHECKLIST item 3): above expected + 3, or
    # below 0.5 for a method whose expected order is >= 1 (a converging method cannot get worse as tau -> 0
    # in its asymptotic range; such a fit sits on a pre-asymptotic transient the trace ratio did not catch)
    for (p, m), o in orders.items():
        ex = expected(p, m)
        pa = o.get("p_asym", np.nan)
        if np.isfinite(pa) and ex is not None and (pa > ex + 3 or (ex >= 1 and pa < 0.5)):
            o["flag"] = (o["flag"] + " implausible").strip()

    # ---------------- matched cost (hierarchical bootstrap; bracket interpolation; dual verdicts)
    B = int(df["B"].max())
    matched, budget_rate = matched_cost(df, cells, cost_of, problems, seeds_by_p, B, "flops_per_step")
    # sensitivity: ideal LU reuse at a data sample on the affine problems (changes only PC / Newton)
    matched_reuse, _ = matched_cost(df, cells, cost_of, problems, seeds_by_p, B, "flops_lu_reuse")
    # sensitivity: costs exactly as step_flops returns them (the E that simulate evaluates for the
    # Davidenko predictor is charged although the predictor ignores it; pre-review primary cost)
    matched_code, _ = matched_cost(df, cells, cost_of, problems, seeds_by_p, B, "flops_code_view")
    # sensitivity: GNN at eta = 1.9 instead of the primary 1.5 (closer to its fastest-contraction step
    # 2/(1 + sigma_min^2/sigma_bar^2) ~ 2; the steady lag of a quasi-static GNN scales like 1/eta)
    matched_gnn19 = gnn_eta_sensitivity(df, cells, cost_of, problems, seeds_by_p, B, eta=1.9)

    # ---------------- rankings: fixed tau vs matched cost (Kendall tau_b, floor values tied)
    rank_info = {}
    for p in problems:
        cp = cells[(cells["pshort"] == p) & cells["primary"] & np.isclose(cells["tau"], tau_fix)]
        ms, a_fix, a_bud, excluded = [], [], [], {}
        for m in methods_all:
            c = cp[cp["method"] == m]
            mm = matched.get((p, m, 0))
            # censoring (REVIEW_CHECKLIST item 2): a method diverged / unstable at tau_fix or at a bracketing
            # tau of the budget (+inf) stays in the ranking as tied-worst; only a missing value (budget
            # outside the sampled tau range, no cell) is excluded, and listed
            if c.empty or mm is None or np.isnan(mm["mssre"]) or np.isnan(c["med"].iloc[0]):
                excluded[m] = ("no cell at tau_fix" if c.empty or np.isnan(c["med"].iloc[0]) else
                               (mm or {}).get("verdict", "no matched value"))
                continue
            ms.append(m)
            a_fix.append(c["med"].iloc[0])
            a_bud.append(mm["mssre"])
        if len(ms) >= 3:
            kt = kendall_floor(a_fix, a_bud)
            causal = [k for k, m in enumerate(ms) if not is_noncausal(m)]
            ktc = kendall_floor([a_fix[k] for k in causal], [a_bud[k] for k in causal]) if len(causal) >= 3 else None
            best_fix = ms[int(np.argmin([a_fix[k] if k in causal else np.inf for k in range(len(ms))]))]
            best_bud = ms[int(np.argmin([a_bud[k] if k in causal else np.inf for k in range(len(ms))]))]
            rank_info[p] = {"methods": ms, "kendall_tau_b": float(kt.statistic),
                            "p_value_descriptive": float(kt.pvalue),
                            "n": len(ms),
                            "kendall_tau_b_causal": float(ktc.statistic) if ktc is not None else None,
                            "n_causal": len(causal),
                            "n_tied_worst_inf": int(sum(1 for v in a_bud if np.isposinf(v))
                                                    + sum(1 for v in a_fix if np.isposinf(v))),
                            "excluded": excluded,
                            "best_causal_fixed_tau": best_fix, "best_causal_matched_cost": best_bud}
    cross = {}
    for i, p in enumerate(problems):
        for q in problems[i + 1:]:
            if p in rank_info and q in rank_info:
                common = [m for m in rank_info[p]["methods"] if m in rank_info[q]["methods"]]
                if len(common) >= 3:
                    a = [matched[(p, m, 0)]["mssre"] for m in common]
                    b = [matched[(q, m, 0)]["mssre"] for m in common]
                    kt = kendall_floor(a, b)
                    cross[f"{p}~{q}"] = {"kendall_tau_b": float(kt.statistic),
                                         "p_value_descriptive": float(kt.pvalue), "n": len(common)}
    summary["ranking"] = {"tau_fixed": tau_fix, "per_problem": rank_info, "cross_problem_matched_cost": cross,
                          "note": "values below 10 x 1e-12 are tied (round-off floor); diverged or unstable values "
                                  "(+inf, at tau_fix or at a bracketing tau of the budget) are kept as tied-worst; "
                                  "methods without a matched value (budget outside the sampled tau range) are left "
                                  "out and listed under 'excluded'; p-values are descriptive (n ~ 20 correlated "
                                  "methods, no multiplicity control)"}

    # ---------------- equivalence ZNN-Euler vs causal PC-Newton (C=1) at equal tau
    equal_tau = {}
    for p in problems:
        a = cells[(cells["pshort"] == p) & (cells["method"] == REF) & cells["primary"]].set_index("tau")["med"]
        b = cells[(cells["pshort"] == p) & (cells["method"] == "pc_newton_causal:C1")].set_index("tau")["med"]
        common = a.index.intersection(b.index)
        if len(common):
            r_all = b[common] / a[common]
            r = r_all.replace([np.inf, -np.inf], np.nan).dropna()
            # censoring: a tau where either median is +inf / NaN is reported, not silently dropped
            equal_tau[p] = {"ratio_median": float(r.median()) if len(r) else None,
                            "ratio_min": float(r.min()) if len(r) else None,
                            "ratio_max": float(r.max()) if len(r) else None,
                            "taus": [float(t) for t in r.index],
                            "taus_not_finite": [float(t) for t in common if t not in r.index]}
    summary["znn_euler_vs_pc_newton_causal_equal_tau"] = equal_tau

    # ---------------- GNN
    gnn = {"tau_lag": None, "lag": {}, "stability": {}, "tau_sweep": {}, "fits": {}}
    lag = df[df["method"] == "gnnlag"]
    if not lag.empty:
        gnn["tau_lag"] = float(lag["tau"].iloc[0])
    for i, p in enumerate(problems):
        seeds = seeds_by_p[p]
        th = gnn_theory[p]
        smin, c_lag = th["sigma_min"], th["c_lag"]
        L = lag[lag["pshort"] == p]
        if not L.empty:
            E0 = float(np.nanmedian(L["initial_residual"]))
            rows = []
            for gm, g in L.groupby("gamma"):
                med, q1, q3 = qstats(g["mssre_v"])
                tf, qs = gnn_regime(gm, smin, c_lag, E0, problem_freq(p))
                rows.append((gm, med, q1, q3, float(tf and qs), int(g["status_eff"].isin(["diverged", "unstable"]).sum()),
                             int(tf), int(qs)))
            rows.sort()
            G = np.array([r[0] for r in rows])
            M = np.array([r[1] for r in rows])
            U = (np.array([r[4] for r in rows]) > 0) & np.isfinite(M)
            gnn["lag"][p] = {"gamma": G.tolist(), "med": M.tolist(), "q25": [r[2] for r in rows],
                             "q75": [r[3] for r in rows], "used": U.astype(int).tolist(),
                             "transient_free": [r[6] for r in rows], "quasi_static": [r[7] for r in rows],
                             "n_div": [r[5] for r in rows], "c_lag": c_lag, "sigma_min": smin, "E0": E0}
            fit_used = fit_order(G[U], M[U]) if U.sum() >= 2 else {"p_asym": np.nan, "p_all": np.nan, "flag": ""}
            fit_all = fit_order(G, M)
            # bootstrap CI of the slope on the transient-free, quasi-static gammas
            ci = (np.nan, np.nan)
            if U.sum() >= 2:
                si, bi = boot_indices(len(seeds), B)
                V = cube(L[L["gamma"].isin(G[U])].assign(tau=lambda d: d["gamma"]), list(G[U]), seeds, B)
                bm = boot_medians(V, si, bi)
                sl = []
                for r in range(N_BOOT):
                    y = bm[:, r]
                    ok = np.isfinite(y) & (y > 0)
                    if ok.sum() >= 2:
                        sl.append(np.polyfit(np.log10(G[U][ok]), np.log10(y[ok]), 1)[0])
                if len(sl) > 0.5 * N_BOOT:
                    ci = (float(np.percentile(sl, 2.5)), float(np.percentile(sl, 97.5)))
            ratio_theory = (M[U] * G[U] / c_lag).tolist() if U.any() else []
            gnn["fits"][p] = {"slope_transient_free": fit_used.get("p_asym"), "slope_ci": ci,
                              "fit_flag": fit_used.get("flag", ""), "n_asym": fit_used.get("n_asym", 0),
                              "slope_all_points": fit_all.get("p_all"), "n_used": int(U.sum()),
                              "gammas_used": G[U].tolist(), "measured_over_theory": ratio_theory,
                              "sigma_min": smin, "c_lag": c_lag,
                              "criterion": "transient-free (E0 exp(-gamma sigma_min^2 t_ss) < 0.1 c_lag/gamma) and "
                                           f"quasi-static (gamma sigma_min^2 >= {QS_FACTOR:g} omega)"}
        Gs = df[(df["pshort"] == p) & (df["method"] == "gnn")]
        if not Gs.empty:
            st = []
            for lvl, g in Gs.groupby("level"):
                eta = float(lvl.split("=")[1])
                # per-instance step eta_inst = gamma tau sigma_max(instance)^2 <= eta (sigma_bar = max over seeds)
                eta_inst = [eta * (CFG.jacobian_bounds(p, int(s))["sigma_max"] / sbar[p]) ** 2 for s in seeds]
                st.append((eta, float(g["status_eff"].isin(["diverged", "unstable"]).mean()),
                           float((g["status"] == "diverged").mean()),
                           float(g.drop_duplicates("run_id")["gamma"].min()),
                           float(g.drop_duplicates("run_id")["gamma"].max()), min(eta_inst), max(eta_inst)))
            st.sort()
            gnn["stability"][p] = {"eta": [s[0] for s in st], "frac_div": [s[1] for s in st],
                                   "frac_div_runner": [s[2] for s in st],
                                   "gamma_min": [s[3] for s in st], "gamma_max": [s[4] for s in st],
                                   "eta_instance_min": [s[5] for s in st], "eta_instance_max": [s[6] for s in st]}
            c = cells[(cells["pshort"] == p) & (cells["method"] == "gnn") & cells["primary"]].sort_values("tau")
            gnn["tau_sweep"][p] = {"tau": c["tau"].tolist(), "med": c["med"].tolist(),
                                   "pre_asym": c["pre_asym"].astype(int).tolist(),
                                   "gamma": [CFG.GNN_ETA_PRIMARY / (t * sbar[p] ** 2) for t in c["tau"]]}
    summary["gnn"] = {"sigma_bar": sbar, "eta_levels": CFG.GNN_ETAS, "eta_primary": CFG.GNN_ETA_PRIMARY,
                      "tau_lag": gnn["tau_lag"], "fits": gnn["fits"], "stability": gnn["stability"],
                      "tau_sweep": gnn["tau_sweep"], "theory": gnn_theory}

    # ---------------- transient cost: time to ||E|| < EPS_T at tau_fix (right-censored at T)
    EPS_T = 1e-2       # above the steady level of every closed-loop causal ZNN / PC-Newton run at tau = 1e-2
    t_eps = {}
    sub = df[np.isclose(df["tau"], tau_fix) & df["primary"] & (df["method"] != "gnnlag")]
    tt = time_to_eps(sub, results, EPS_T)
    for p in problems:
        t_eps[p] = {}
        sp_ = sub[sub["pshort"] == p]
        for m, g in sp_.groupby("method"):
            v = tt.loc[g.index].to_numpy(float)
            v = np.where(g["status_eff"].to_numpy() == "ok", v, np.inf)
            v = np.where(np.isnan(v), np.inf, v)                # no trace / never reached -> censored
            # upper median: conservative under right-censoring (n even -> the larger middle value)
            med = float(np.quantile(np.where(np.isinf(v), 1e300, v), 0.5, method="higher"))
            t_eps[p][m] = {"median_s": (None if med >= 1e299 else med),
                           "median_steps": (None if med >= 1e299 else med / tau_fix),
                           "frac_censored": float(np.isinf(v).mean()), "n": int(len(v))}
    summary["time_to_eps"] = {"eps": EPS_T, "tau": tau_fix, "T": CFG.T_HORIZON, "per_problem": t_eps,
                              "note": "upper median over trajectories of the first time after which ||E|| < eps "
                                      "for the rest of the run (from the stored traces); never reached within T, "
                                      "diverged or unstable = right-censored ('> T')"}

    # ---------------- figures
    figs = []
    style.apply()
    figs.append(fig_lines(cells, cost_of, orders, problems, "disc", "flops", "fig_E9_workprec_znn", figdir))
    figs.append(fig_lines(cells, cost_of, orders, problems, "baseline", "flops", "fig_E9_workprec_base", figdir))
    figs.append(fig_lines(cells, cost_of, orders, problems, "baseline", "wall", "fig_E9_workprec_wall", figdir))
    figs.append(fig_lines(cells, cost_of, orders, problems, "disc", "tau", "fig_E9_tau_znn", figdir))
    figs.append(fig_lines(cells, cost_of, orders, problems, "baseline", "tau", "fig_E9_tau_base", figdir))
    figs.append(fig_fixedtau(cells, cost_of, problems, tau_fix, figdir))
    figs.append(fig_fixedtau_znn(cells, cost_of, problems, tau_fix, figdir))
    figs.append(fig_pc(cells, problems, figdir))
    if gnn["lag"] or gnn["stability"]:
        figs.append(fig_gnn(gnn, problems, figdir))
    head = ([m for m in methods_all if m.startswith("znn:")] + ["gnn", "davidenko"]
            + [f"{k}:C{C}" for k in ("pc_newton_causal", "pc_gradient_causal") for C in CFG.PC_CS]
            + ["newton_tracking_causal:C1", "pc_newton:C1", "pc_gradient:C1", "newton_tracking:C1"])
    head = [m for m in head if m in methods_all]
    reuse_methods = [m for m in head if m.split(":")[0].startswith(("pc_newton", "newton_tracking"))]
    figs.append(fig_equiv(matched, problems, head, figdir))
    figs.append(fig_traces(df, results, problems, tau_fix, figdir))
    summary["figures"] = [str(f) for f in figs]

    # ---------------- tables
    tabs = []
    # cost table
    L = ["\\begin{table}[t]", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{3.5pt}",
         "\\caption{E9 per-step cost per trajectory ($n_x=m=8$): steady Problem calls per sampling step "
         "(linear system; the same on TV-QP, while PC-gradient on the gradient field of the TV unconstrained "
         "problem uses Simonetto's step $x-\\alpha E$ and needs no $J^\\top v$), FLOPs per step (source: "
         f"{tex(', '.join(summary['cost_source']))}; "
         "data generation excluded; algorithmic view: the $\\partial_tE$ that the simulator evaluates for GNN and "
         "the $E$ it evaluates for the Davidenko predictor, which ignore them, are not charged) and median lean "
         f"wall time per step in $\\mu$s (single thread; GNN and ZNN runs carry {len(CFG.GNN_ETAS)} and 2 gains "
         "per batch; descriptive only: timed inside a parallel pool on a shared machine, and at $n_x=8$ a step "
         "is dominated by interpreter overhead). [nc] = non-causal.}",
         "\\label{tab:E9_cost}", "\\begin{tabular}{lrrrr" + "r" * (2 * len(problems)) + "}", "\\toprule",
         "& \\multicolumn{4}{c}{calls per step} & \\multicolumn{%d}{c}{FLOPs per step} & "
         "\\multicolumn{%d}{c}{wall per step ($\\mu$s)} \\\\" % (len(problems), len(problems)),
         "\\cmidrule(lr){2-5}\\cmidrule(lr){6-%d}\\cmidrule(lr){%d-%d}" % (5 + len(problems), 6 + len(problems),
                                                                      5 + 2 * len(problems)),
         "Method & $E$ & $\\partial_tE$ & solve & $J^\\top v$ & " + " & ".join(PTEX[p] for p in problems) + " & "
         + " & ".join(PTEX[p] for p in problems) + " \\\\", "\\midrule"]
    for m in methods_all:
        e0 = cost_of.get((problems[0], m))
        if e0 is None:
            continue
        c = e0["calls_per_step"]
        fl, walls = [], []
        for p in problems:
            e = cost_of.get((p, m))
            fl.append(f"{e['flops_per_step']:.0f}" if e else "--")
            cc = cells[(cells["pshort"] == p) & (cells["method"] == m) & cells["primary"]]
            walls.append(f"{1e6 * np.nanmedian(cc['wall_per_step']):.0f}" if not cc.empty else "--")
        L.append(f"{tex(mlabel(m))} & {c['E']:g} & {c['dEdt']:g} & {c['solve_J']:g} & {c['JT']:g} & "
                 + " & ".join(fl) + " & " + " & ".join(walls) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    tabs.append(write_tex(tabdir / "E9_cost.tex", L))
    # orders table
    L = ["\\begin{table}[t]", "\\centering", "\\small",
         "\\caption{E9 empirical orders of the steady-state residual (MSSRE $\\propto\\tau^p$), fitted on the "
         "asymptotic range: consecutive-slope diagnostics on runs of consecutive usable $\\tau$, points below "
         "$10^{-12}$ excluded, and cells excluded whose residual is still decaying or growing inside the steady "
         "window (median ratio of $\\max\\|E\\|$ on $[T/2,3T/4)$ to that on $[3T/4,T]$ above 3 or below 1/3; "
         "finite-growth trajectories count as diverged); GNN cells also need a transient-free, quasi-static gain "
         "$\\gamma=\\eta/(\\tau\\bar\\sigma^2)$. In brackets the all-points slope (every finite cell above the floor). "
         "ZNN at fixed $h=h_{opt}$, GNN at fixed $\\eta$. ``floor'': at round-off for all $\\tau$.}",
         "\\label{tab:E9_orders}", "\\begin{tabular}{lc" + "c" * len(problems) + "}", "\\toprule",
         "Method & expected & " + " & ".join(PTEX[p] for p in problems) + " \\\\", "\\midrule"]
    orders_out = {}
    for m in methods_all:
        cells_txt = []
        exps = []
        for p in problems:
            o = orders.get((p, m))
            ex = expected(p, m)
            exps.append("exact" if ex is None else f"{ex}")
            if o is None:
                cells_txt.append("--")
                continue
            orders_out[f"{p}|{m}"] = {**{k: v for k, v in o.items() if k != "local"}, "local": o["local"],
                                      "expected": ex}
            if o["flag"].startswith("floor") and not np.isfinite(o["p_asym"]):
                cells_txt.append("floor")
            elif np.isfinite(o["p_asym"]):
                cells_txt.append(f"{o['p_asym']:.2f} [{o['p_all']:.2f}]" + ("$^\\ast$" if "2pt" in o["flag"] else "")
                                 + ("$^{!}$" if "implausible" in o["flag"] else ""))
            else:
                cells_txt.append("--")
        exp_txt = "/".join(dict.fromkeys(exps))
        L.append(f"{tex(mlabel(m))} & {exp_txt} & " + " & ".join(cells_txt) + " \\\\")
    L += ["\\bottomrule", "\\multicolumn{%d}{l}{\\footnotesize $^\\ast$ no consistent range of $\\ge 3$ usable points "
          "(local slopes within %.2f): slope of the two smallest usable $\\tau$.}\\\\" % (2 + len(problems), SLOPE_TOL),
          "\\multicolumn{%d}{l}{\\footnotesize $^{!}$ implausible (above expected$+3$, or below $0.5$ for an expected "
          "order $\\ge 1$): pre-asymptotic, not an order estimate.}\\\\" % (2 + len(problems)),
          "\\end{tabular}", "\\end{table}"]
    tabs.append(write_tex(tabdir / "E9_orders.tex", L))
    summary["orders"] = orders_out
    # matched-cost table (budget 0)
    def _cell(r):
        if not r:
            return "--"
        if not np.isfinite(r["log_ratio"]):
            return "unst." if "unstable" in r["verdict"] else "--"
        lo, hi = r["ci"]
        his = "+\\infty" if np.isposinf(hi) else f"{hi:+.2f}"
        marks = ""
        if r["verdict"].startswith("equivalent"):
            marks += "e"
        if any(f.startswith("transient") for f in r["flags"]):
            marks += "t"
        if any("floor" in f for f in r["flags"]):
            marks += "f"
        if r["verdict"] != r["verdict_bonferroni"]:
            marks += "\\dagger"
        return f"${r['log_ratio']:+.2f}$ $[{lo:+.2f}, {his}]$" + (f"$^{{{marks}}}$" if marks else "")

    L = ["\\begin{table}[t]", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
         "\\caption{E9 matched-cost comparison: $\\log_{10}$ of the MSSRE ratio to ZNN-Euler ($h=1$) at the same "
         f"FLOP rate (budget = ZNN-Euler at $\\tau={BUDGET_TAUS[0]:g}$\\,s), hierarchical bootstrap 95\\% CI "
         f"({N_BOOT} resamples; clusters = the {max(len(s) for s in seeds_by_p.values())} problem instances, then "
         "initial states, which are pseudo-replicates in steady state; with $\\le 3$ instances the interval is "
         f"close to the instance range and descriptive). Marks: $^e$ CI inside the $\\pm{DELTA_EQ}$-decade "
         "equivalence margin; $^t$ a bracketing $\\tau$ is transient-limited (the method's steady value is "
         "overestimated, so its ratio is conservative); $^f$ method at the round-off floor (gain is a lower bound); "
         "$^\\dagger$ verdict changes with a Bonferroni CI over the family; unst.: diverged or finite-growth at a "
         "bracketing $\\tau$; --: budget outside the sampled $\\tau$ range. Lower block: costs with one LU "
         "factorisation per data sample on the affine problems. [nc] = non-causal (uses $t_{k+1}$ data).}",
         "\\label{tab:E9_matched}", "\\begin{tabular}{l" + "c" * len(problems) + "}", "\\toprule",
         "Method & " + " & ".join(PTEX[p] for p in problems) + " \\\\", "\\midrule"]
    nc_started = False
    for m in head:
        if is_noncausal(m) and not nc_started:      # causal and non-causal never share a block
            nc_started = True
            L += ["\\midrule", "\\multicolumn{%d}{l}{\\emph{non-causal: corrections use $t_{k+1}$ data "
                               "(information advantage, not comparable)}} \\\\" % (1 + len(problems))]
        L.append(f"{tex(mlabel(m))} & " + " & ".join(_cell(matched.get((p, m, 0))) for p in problems) + " \\\\")
    L += ["\\midrule", "\\multicolumn{%d}{l}{\\emph{ideal LU reuse (affine problems)}} \\\\" % (1 + len(problems))]
    for m in reuse_methods:
        if m.startswith(("pc_newton_causal", "newton_tracking_causal")):
            L.append(f"{tex(mlabel(m))} & " + " & ".join(_cell(matched_reuse.get((p, m, 0))) for p in problems)
                     + " \\\\")
    L += ["\\midrule", "Kendall $\\tau_b$, fixed $\\tau$ vs matched cost (all) & "
          + " & ".join(f"{rank_info[p]['kendall_tau_b']:.2f} ($n$={rank_info[p]['n']})" if p in rank_info else "--"
                       for p in problems) + " \\\\",
          "Kendall $\\tau_b$, causal methods only & "
          + " & ".join(f"{rank_info[p]['kendall_tau_b_causal']:.2f} ($n$={rank_info[p]['n_causal']})"
                       if p in rank_info and rank_info[p].get("kendall_tau_b_causal") is not None else "--"
                       for p in problems) + " \\\\",
          "\\bottomrule", "\\end{tabular}", "\\end{table}"]
    tabs.append(write_tex(tabdir / "E9_matched.tex", L))
    # GNN table
    L = ["\\begin{table}[t]", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
         "\\caption{E9 GNN: $\\bar\\sigma=\\max_t\\sigma_{max}(J)$, gain at $\\eta=\\gamma\\tau\\bar\\sigma^2="
         f"{CFG.GNN_ETA_PRIMARY:g}$ for $\\tau={tau_fix:g}$, fraction of trajectories diverged or growing per $\\eta$, "
         "and the lag-floor slope of MSSRE vs $\\gamma$ (fixed $\\gamma$; only $\\gamma$ that are transient-free and "
         f"quasi-static, $\\gamma\\sigma_{{min}}^2\\ge{QS_FACTOR:g}\\omega$; bootstrap 95\\% CI in brackets, "
         "all-points slope in parentheses; theory $-1$) with the median ratio of measured MSSRE to $c_{lag}/\\gamma$.}",
         "\\label{tab:E9_gnn}",
         "\\begin{tabular}{lccc" + "c" * len(CFG.GNN_ETAS) + "cc}", "\\toprule",
         "Problem & $\\bar\\sigma$ & $\\sigma_{min}$ & $\\gamma$ & "
         + " & ".join(f"div. $\\eta$={e:g}" for e in CFG.GNN_ETAS) + " & slope [all] & meas./theory \\\\", "\\midrule"]
    for p in problems:
        S = gnn["stability"].get(p, {})
        F = gnn["fits"].get(p, {})
        dv = {e: f for e, f in zip(S.get("eta", []), S.get("frac_div", []))}
        sl = F.get("slope_transient_free", np.nan)
        ci = F.get("slope_ci", (np.nan, np.nan))
        slope_txt = (f"{sl:.2f} [{ci[0]:.2f}, {ci[1]:.2f}] ({F.get('slope_all_points', np.nan):.2f})"
                     if sl is not None and np.isfinite(sl) else f"-- ({F.get('slope_all_points', np.nan):.2f})")
        mt = F.get("measured_over_theory", [])
        mt_txt = f"{np.median(mt):.2f}" if mt else "--"
        L.append(f"{PTEX[p]} & {sbar[p]:.2f} & {gnn_theory[p]['sigma_min']:.3f} & "
                 f"{CFG.GNN_ETA_PRIMARY / (tau_fix * sbar[p] ** 2):.3g} & "
                 + " & ".join(f"{dv.get(e, np.nan):.2f}" for e in CFG.GNN_ETAS) + f" & {slope_txt} & {mt_txt} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    tabs.append(write_tex(tabdir / "E9_gnn.tex", L))
    summary["tables"] = tabs

    # ---------------- headline numbers
    def mt(p, m, b=0, src=None):
        r = (src or matched).get((p, m, b))
        if not r:
            return None
        return {"log10_ratio": r["log_ratio"], "ci95": list(r["ci"]), "verdict": r["verdict"],
                "ci95_bonferroni": list(r["ci_bonferroni"]), "verdict_bonferroni": r["verdict_bonferroni"],
                "flags": r["flags"], "mssre": r["mssre"], "mssre_ref": r["mssre_ref"],
                "tau_at_budget": r["tau_at_budget"]}

    summary["matched_cost"] = {f"budget{b}_tau{BUDGET_TAUS[b]:g}": {p: {m: mt(p, m, b) for m in head} for p in problems}
                               for b in range(len(BUDGET_TAUS))}
    summary["matched_cost_lu_reuse"] = {
        "note": "sensitivity: FLOPs with one LU factorisation per data sample on the affine problems (linsys, "
                "tvqp); only PC-Newton / Newton tracking change; tvuo (J depends on x) is unchanged",
        "flops_lu_reuse": {p: {m: cost_of[(p, m)]["flops_lu_reuse"] for m in reuse_methods + [REF]
                               if (p, m) in cost_of} for p in problems},
        **{f"budget{b}_tau{BUDGET_TAUS[b]:g}": {p: {m: mt(p, m, b, matched_reuse) for m in reuse_methods}
                                               for p in problems} for b in range(len(BUDGET_TAUS))}}
    summary["matched_cost_code_view"] = {
        "note": "sensitivity: FLOPs exactly as znnbench.costs.step_flops returns them, i.e. including the E "
                "that simulate evaluates for the Davidenko predictor although it ignores it (the pre-review "
                "primary cost); only Davidenko-based schemes change",
        "flops_code_view": {p: {m: cost_of[(p, m)]["flops_code_view"] for m in head if (p, m) in cost_of}
                            for p in problems},
        **{f"budget{b}_tau{BUDGET_TAUS[b]:g}": {p: {m: mt(p, m, b, matched_code) for m in reuse_methods}
                                               for p in problems} for b in range(len(BUDGET_TAUS))}}

    def _mt_g(r):
        return None if not r else {"log10_ratio": r["log_ratio"], "ci95": list(r["ci"]), "verdict": r["verdict"],
                                   "flags": r["flags"], "tau_at_budget": r["tau_at_budget"]}
    summary["matched_cost_gnn_eta1.9"] = {
        "note": "sensitivity: GNN at eta = 1.9 (instead of the primary 1.5), same budgets",
        **{f"budget{b}_tau{BUDGET_TAUS[b]:g}": {p: _mt_g(matched_gnn19.get(p, {}).get(b)) for p in problems}
           for b in range(len(BUDGET_TAUS))}}
    # censoring per (problem, method) at the primary level, all tau (REVIEW_CHECKLIST item 2)
    cens = {}
    for (p, m), g in cells[cells["primary"]].groupby(["pshort", "method"]):
        cens[f"{p}|{m}"] = {"n": int(g["n"].sum()), "n_diverged": int(g["n_div"].sum()),
                            "n_finite_growth": int(g["n_unstable"].sum()), "n_failed": int(g["n_fail"].sum()),
                            "taus_with_censoring": [float(t) for t, a, b_ in
                                                    zip(g["tau"], g["n_div"], g["n_unstable"]) if a + b_ > 0]}
    summary["censoring"] = {
        "rule": "diverged (runner) and finite-growth trajectories have MSSRE = +inf and stay in every median, "
                "bootstrap resample and ranking as the worst value; failed / timed-out runs (NaN) are dropped "
                "and counted here; no np.isfinite / dropna filter removes +inf before a statistic",
        "n_failed_total": int(cells["n_fail"].sum()), "per_problem_method": cens}
    n_changed = sum(1 for k, r in matched.items() if r["verdict"] != r["verdict_bonferroni"])
    summary["multiplicity"] = {
        "family": "all (problem, method) comparisons of one budget", "n_family": next(iter(matched.values()))["n_family"]
        if matched else 0, "n_verdicts_changed_by_bonferroni": n_changed,
        "changed": {f"{p}|{m}|budget{b}": [r["verdict"], r["verdict_bonferroni"]] for (p, m, b), r in matched.items()
                    if r["verdict"] != r["verdict_bonferroni"]}}
    summary["budget_rates_flops_per_s"] = budget_rate
    fixed = {}
    for p in problems:
        cp = cells[(cells["pshort"] == p) & cells["primary"] & np.isclose(cells["tau"], tau_fix)]
        fixed[p] = {m: {"mssre_median": float(c["med"].iloc[0]), "iqr": [float(c["q25"].iloc[0]), float(c["q75"].iloc[0])],
                        "pre_asymptotic": bool(c["pre_asym"].iloc[0]), "reason": str(c["pre_reason"].iloc[0]),
                        "n_diverged_or_unstable": int(c["n_div"].iloc[0] + c["n_unstable"].iloc[0]),
                        "flops_per_step": cost_of.get((p, m), {}).get("flops_per_step")}
                    for m, c in cp.groupby("method")}
    summary["fixed_tau"] = {"tau": tau_fix, "per_problem": fixed}
    summary["peak_rss_mb_max"] = float(np.nanmax(df["peak_rss_mb"]))
    runs = df.drop_duplicates("run_id")
    summary["timing"] = {"wall_per_step_median_s": float(np.nanmedian(runs["wall_per_step"])),
                         "wall_recorded_total_s": float(np.nansum(runs["wall_recorded"])),
                         "wall_lean_total_s": float(np.nansum(runs["wall"] * runs["timing_runs"].where(runs["timing_mode"] == "lean", 0))),
                         "sim_steps_total": float(np.nansum(runs["steps_run"])),
                         "n_runs": int(len(runs))}
    # recorded per-step time per trajectory-batch (metric run) -> used for the full-scale estimate
    rec = runs["wall_recorded"] / runs["steps_run"].clip(lower=1)
    summary["timing"]["wall_recorded_per_step_median_s"] = float(np.nanmedian(rec))
    # per config: (metric run + lean timing runs + setup) seconds per simulated step -> full-scale estimate
    tb = {}
    for cfg_name, g in runs.groupby("config"):
        lean = (g["wall"] * g["timing_runs"]).where(g["timing_mode"] == "lean", 0.0)
        secs = float(np.nansum(g["wall_recorded"] + lean + g["setup_wall"]))
        steps = float(np.nansum(g["steps"]))
        tb[str(cfg_name)] = {"seconds": secs, "sim_steps": steps, "s_per_sim_step": secs / max(steps, 1.0),
                             "traj_per_run": float(np.nanmedian(g["B"] * g["n_gains"]))}
    summary["timing"]["by_config"] = tb

    # ---------------- hypothesis verdicts
    hyp = {}
    fits = summary["gnn"]["fits"]
    h1 = {}
    for p in problems:
        f = fits.get(p, {})
        sl, ci = f.get("slope_transient_free"), f.get("slope_ci", (np.nan, np.nan))
        npts = f.get("n_used", 0)
        tag = f"{npts} gammas" + (", 2-point slope" if "2pt" in str(f.get("fit_flag", "")) else "")
        if sl is None or not np.isfinite(sl):
            h1[p] = f"n/a (fewer than 2 transient-free, quasi-static gammas; {npts} available)"
        elif np.all(np.isfinite(ci)) and ci[0] <= -1.0 <= ci[1]:
            h1[p] = f"supported (slope {sl:.2f}, CI [{ci[0]:.2f}, {ci[1]:.2f}] covers -1; {tag})"
        elif abs(sl + 1.0) <= 0.15:
            h1[p] = (f"supported within the +-0.15 tolerance (slope {sl:.2f}, CI [{ci[0]:.2f}, {ci[1]:.2f}] "
                     f"excludes -1; {tag})")
        else:
            h1[p] = f"not supported (slope {sl:.2f}, CI [{ci[0]:.2f}, {ci[1]:.2f}]; {tag})"
    hyp["H1_gnn_lag_floor_O(1/gamma)"] = h1
    h1b = {}
    for p in problems:
        st = summary["gnn"]["stability"].get(p)
        if not st:
            continue
        d = dict(zip(st["eta"], st["frac_div"]))
        stable = all(v == 0 for e, v in d.items() if e < 2)
        big = max(d)
        unstable = big > 2 and d[big] > 0
        h1b[p] = (("supported" if stable and unstable else "not supported")
                  + " (eta < 2 never diverges; eta = %g diverges in %.0f%%; fractions incl. finite growth: %s)"
                  % (big, 100 * d[big], {k: round(v, 3) for k, v in d.items()}))
    hyp["H1b_gnn_stability_eta<2"] = h1b
    h2 = {"tested": (
        "H2a (Section 4, Eq. br-pc): ZNN-Euler with h = 1 IS causal PC-Newton with C = 1 up to where E_t is "
        "evaluated (x_k vs the corrected x_k^), an O(tau ||E_k||) difference; prediction: equal-tau MSSRE ratio "
        "= 1 + O(tau); falsified by a ratio outside [0.9, 1.1] or a different order. "
        "H2b: with identical per-step accuracy and order p = 2, the matched-FLOP-rate log-ratio must equal "
        "2 log10(cost ratio) (prediction printed next to each observed value); a margin-based 'equivalent' is "
        "therefore an accounting statement, not an accuracy one. "
        "H2c (look-ahead DTZNN = predictive LMM): order q+1 at fixed h, one solve per step for every formula, so "
        "at matched cost a higher-order formula must be better by about (q_hi - q_lo) |log10 tau| decades inside "
        "its stable, quasi-static tau range -- i.e. 'ZNN ~ Uhlig' is predicted to FAIL across orders.")}
    family = [m for m in methods_all if m.startswith("znn:") and m != REF
              and (CFG.formula_stability(m[4:])["expected_order"] or 0) >= 4]

    def _fmt(r, pred=None):
        if not r or not np.isfinite(r["log_ratio"]):
            return "n/a" if not r else r["verdict"]
        fl = ("; " + "; ".join(r["flags"])) if r.get("flags") else ""
        pr = f"; order-2 prediction {pred:+.2f}" if pred is not None else ""
        return (f"{r['verdict']} ({r['log_ratio']:+.2f} dec, CI [{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}]{pr}; "
                f"Bonferroni: {r['verdict_bonferroni']}{fl})")

    for b in range(len(BUDGET_TAUS)):
        for p in problems:
            res_ = {}
            for lab, src, key in (("algorithmic FLOPs (primary)", matched, "flops_per_step"),
                                  ("ideal LU reuse", matched_reuse, "flops_lu_reuse"),
                                  ("code view (unused E charged)", matched_code, "flops_code_view")):
                e_pc, e_ref = cost_of.get((p, "pc_newton_causal:C1")), cost_of.get((p, REF))
                pred = (2 * math.log10(e_pc[key] / e_ref[key])) if e_pc and e_ref else None
                res_[f"ZNN-Euler vs causal PC-Newton(C=1), {lab}"] = _fmt(src.get((p, "pc_newton_causal:C1", b)),
                                                                         pred)
            ul_all = [(m, matched.get((p, m, b))) for m in family]
            ul_all = [(m, r) for m, r in ul_all if r]
            ul = [(m, r) for m, r in ul_all if np.isfinite(r["log_ratio"])]
            best_u = min(ul, key=lambda mr: mr[1]["log_ratio"]) if ul else None
            res_["ZNN-Euler vs best high-order look-ahead (order >= 4)"] = (f"{best_u[0]}: " + _fmt(best_u[1])
                                                                           if best_u else "n/a")
            # every formula is listed, the unstable-at-budget ones with their n/a verdict (no silent drop)
            res_["high-order look-ahead formulas (order >= 4), verdicts"] = {m: _fmt(r) for m, r in ul_all}
            h2[f"budget{b}|{p}"] = res_
    hyp["H2_ZNN~PC~Uhlig_at_matched_cost"] = h2
    summary["hypotheses"] = hyp

    # ---------------- definitions, statistics and confounds (REVIEW_CHECKLIST items 5, 6)
    la = [cost_of[(p, m)]["flops_per_step"] / cost_of[(p, REF)]["flops_per_step"]
          for p in problems for m in methods_all if m.startswith("znn:") and (p, m) in cost_of and (p, REF) in cost_of]
    summary["lookahead_cost_ratio_max"] = float(max(la)) if la else None
    n_seeds = sorted({len(s) for s in seeds_by_p.values()})
    s_max = max(n_seeds) if n_seeds else 0
    n_multisets = math.comb(2 * s_max - 1, s_max) if s_max else 0
    ci_txt = (f"With {s_max} seeds there are only {n_multisets} distinct seed resamples and the percentile 'CI' "
              "is essentially the seed range: treat it as descriptive." if s_max <= 3 else
              f"With {s_max} seeds there are {n_multisets} distinct seed resamples; percentile intervals from "
              "few clusters still tend to undercover, so effects near a verdict boundary deserve caution.")
    summary["matched_cost_definition"] = (
        "cost = FLOPs per sampling step per trajectory (B = 1) from znnbench.costs.step_flops, algorithmic view: "
        "data acquisition / generation free (including the analytic derivative data A_dot, b_dot), the dE/dt "
        "algebra charged for every method that uses it, simulate's unused dE/dt (GNN) and unused E (Davidenko "
        "predictor) not charged; one LU factorisation per solve as the code does (sensitivity: one per new data "
        "sample on the affine problems).  Budget = FLOPs per tracked second R = cost/tau of ZNN-Euler (h = 1) "
        "at tau = 1e-2 and 1e-3; every method is compared at the tau where its own cost/tau equals R (log-log "
        "interpolation between the two adjacent grid tau, no bridging over unstable points).  This credits "
        "cheaper methods with faster sampling, i.e. it presumes data at any rate; the fixed-tau view "
        "(fig_E9_fixedtau*) is the complement.  With info = 'analytic' the FLOP count is regime-invariant to "
        "first order (bd1 replaces the dE/dt algebra by one more E evaluation of the same cost), but the "
        "accuracy is not: a backward-difference E_t caps the residual order (E6).  Wall-clock "
        "(fig_E9_workprec_wall, the wall column of E9_cost.tex, timing.*) is DESCRIPTIVE only: it is measured "
        "inside a parallel memguard pool on a shared machine (load 40-180 from other jobs), where Accelerate's "
        "per-cluster matrix units slow BLAS-3 calls by up to +118 % under concurrency (E7 review); at n_x = 8 a "
        "step is a few microseconds of arithmetic inside 250-800 us of interpreter overhead, so it reflects call structure, "
        "not FLOPs.  No verdict uses it.")
    summary["statistics_note"] = (
        f"unit of replication = problem instance (seed); {'/'.join(str(n) for n in n_seeds)} seeds per problem, "
        f"B = {B} initial states each.  The B initial states of a seed reach the same steady orbit (median "
        "within-seed spread of MSSRE ~1e-14 decades), so they are pseudo-replicates; the hierarchical bootstrap "
        f"resamples seeds first (clusters), then initial states.  {ci_txt}  The ZNN-Euler vs PC-Newton CIs are "
        "narrow because the comparison is deterministic per seed (paired), not because of population precision.  "
        "Kendall p-values are descriptive.")
    hgain = ", ".join(f"{FORMULA_LABEL.get(m[4:], m[4:])} {math.log10(st['h_max'] / st['h_opt']):.2f}"
                      for m in methods_all if m.startswith("znn:")
                      for st in [CFG.formula_stability(m[4:])] if st["h_opt"] and st["h_max"])
    fac_frac = [cost_of[(p, REF)].get("flops_factor", 0.0) / cost_of[(p, REF)]["flops_per_step"]
                for p in problems if (p, REF) in cost_of]
    summary["confounds"] = [
        "GNN effective step: eta = gamma tau sigma_bar^2 with sigma_bar = max over instances and time of "
        "sigma_max(J(x*, t)) (tvuo: the bound sigma_max(P) + beta); per-instance eta <= eta. Primary eta = 1.5; "
        "eta = 1.9 sensitivity reported. Equal nominal gamma would NOT be equal effective gain.",
        "ZNN gain rule h = h_opt minimises the spectral radius (fastest transient), not the steady residual, "
        "which scales like 1/h (Prop. 1a); h -> h_max would lower every formula's steady MSSRE by up to "
        f"log10(h_max/h_opt) decades ({hgain}) at the price of slower transients; the ZNN-Euler = PC-Newton "
        "identity holds at h = 1.",
        "Information: non-causal pc_newton / pc_gradient / newton_tracking use t_{k+1} data (marked [nc], "
        "separated in tables/figures, excluded from 'best causal'); ZNN and PC get the analytic E_t, GNN and "
        "Newton tracking do not use it -- the high-order look-ahead advantage is conditional on an accurate E_t "
        "(E6 tests estimated derivatives).",
        "Transients: with rho(h_opt) = "
        + "-".join(f"{v:.3f}" for v in (lambda r: [min(r), max(r)] if r else [])(
            [CFG.formula_stability(m[4:])["rho_opt"] for m in methods_all if m.startswith("znn:")
             and (CFG.formula_stability(m[4:])["expected_order"] or 0) >= 4]))
        + " the order >= 4 formulas can still be decaying in the steady window at tau >= 1e-2 (flag 't': their "
        "MSSRE and matched-cost ratios are conservative); GNN cells need a transient-free, quasi-static gain; "
        "T = 6 s is the PLAN horizon.",
        "Step-size convention: h = gamma tau in ZNN-Bench; papers that write x_{k+1} = ... - h_paper J^-1 E/c_0 "
        "use h_ours = c_0 h_paper (zead_named.PAPER_H_FACTOR); h_opt / h_max here are in ZNN-Bench units.",
        "Start offsets: x0 ~ problem.x0(rng(seed)) is shared by every method of a (problem, seed), so methods "
        "are paired; it sets the transient peaks but not the steady state of contracting methods.",
        "Unit dependence of power-type activations (checklist item 1): not applicable, E9 uses linear "
        "activations only.",
        (f"Size: n_x = 8, where the LU factorisation is {100 * min(fac_frac):.0f}-{100 * max(fac_frac):.0f}% of a "
         "ZNN-Euler step" if fac_frac else "Size: n_x = 8")
        + "; the solve-free GNN gets relatively cheaper as n grows (cost ratio ~1/n), which E7 covers."]

    # ---------------- interpretation
    lines = interpret(summary, problems, head)
    summary["interpretation"] = lines
    (results / "summary.json").write_text(json.dumps(_strict(summary), indent=1, default=_json_default,
                                                     allow_nan=False))
    write_md(results / "SUMMARY.md", summary, problems, head, lines)
    print(f"figures: {len(figs)} -> {figdir}; tables: {len(tabs)} -> {tabdir}; summary -> {results / 'summary.json'}")
    return 0


def _strict(o):
    """Strict JSON: NaN -> null, +-inf -> "inf"/"-inf", numpy scalars/arrays and tuples -> plain."""
    if isinstance(o, dict):
        return {str(k): _strict(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_strict(v) for v in o]
    if isinstance(o, np.ndarray):
        return [_strict(v) for v in o.tolist()]
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return None if math.isnan(v) else ("inf" if v == math.inf else ("-inf" if v == -math.inf else v))
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    return str(o)


def interpret(S, problems, head):
    out = []
    g = S["gnn"]
    sl = [(p, g["fits"][p]) for p in problems if p in g["fits"]]
    if sl:
        txt = "; ".join(f"{p}: {f['slope_transient_free']:.2f} [{f['slope_ci'][0]:.2f}, {f['slope_ci'][1]:.2f}] "
                        f"({f['n_used']} gammas)" if f["slope_transient_free"] is not None
                        and np.isfinite(f["slope_transient_free"]) else f"{p}: n/a ({f['n_used']} usable gammas)"
                        for p, f in sl)
        out.append("GNN lag floor (H1): slope of MSSRE vs gamma on transient-free, quasi-static gammas (theory -1): "
                   f"{txt}.")
    st = g.get("stability", {})
    if st:
        parts, ok_all = [], True
        for p in problems:
            if p in st:
                d = dict(zip(st[p]["eta"], st[p]["frac_div"]))
                ok_all &= all(v == 0 for e, v in d.items() if e < 2)
                parts.append(f"{p}: " + ", ".join(f"eta={e:g}->{f:.2f}" for e, f in d.items()))
        out.append("GNN explicit step limit (H1b; fraction of trajectories diverged or growing): " + "; ".join(parts)
                   + (". No run with eta < 2 diverges" if ok_all else ". Some eta < 2 runs diverge")
                   + "; eta = gamma*tau*sigma_bar^2 uses the largest sigma_max over instances and time, so eta "
                     "slightly above 2 need not destabilise every instance.")
    eq = S.get("znn_euler_vs_pc_newton_causal_equal_tau", {})
    if eq:
        def _eqtxt(p, v):
            if v.get("ratio_median") is None:
                return f"{p}: n/a"
            nf = (f", not finite at tau = {v['taus_not_finite']}" if v.get("taus_not_finite") else "")
            return f"{p}: {v['ratio_median']:.3f} (range {v['ratio_min']:.3f}-{v['ratio_max']:.3f}{nf})"
        out.append("Causal PC-Newton (C=1) vs ZNN-Euler (h=1) at the same tau: MSSRE ratio "
                   + "; ".join(_eqtxt(p, v) for p, v in eq.items())
                   + " -- the two schemes coincide to leading order (Euler-ZNN with h = 1 is one Newton step plus a "
                     "Davidenko prediction), so at matched FLOP rate PC loses (cost ratio)^2 at order 2.")
    b0 = S["matched_cost"][next(iter(S["matched_cost"]))]
    rb0 = S["matched_cost_lu_reuse"][next(k for k in S["matched_cost_lu_reuse"] if k.startswith("budget"))]
    for p in problems:
        rows = b0.get(p, {})
        best = None
        for m, r in rows.items():
            if r and r["log10_ratio"] is not None and np.isfinite(r["log10_ratio"]) and not is_noncausal(m):
                if best is None or r["log10_ratio"] < best[1]:
                    best = (m, r["log10_ratio"], r["ci95"])
        bits = []
        for m, lab in (("pc_newton_causal:C1", "PC-Newton causal C=1"), ("znn:uhlig_4_6", "ZNN-Uhlig 4_6"),
                       ("znn:zead_7i_cang2024", "ZNN-ZeaD-7i")):
            r = rows.get(m)
            if r and r["log10_ratio"] is not None and np.isfinite(r["log10_ratio"]):
                fl = f"; {', '.join(r['flags'])}" if r["flags"] else ""
                bits.append(f"{lab} {r['log10_ratio']:+.2f} dec [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}] "
                            f"({r['verdict']}{fl})")
            elif r:
                bits.append(f"{lab}: {r['verdict']}")
        rr = rb0.get(p, {}).get("pc_newton_causal:C1")
        if rr and rr["log10_ratio"] is not None and np.isfinite(rr["log10_ratio"]):
            bits.append(f"PC-Newton causal C=1 with ideal LU reuse {rr['log10_ratio']:+.2f} dec ({rr['verdict']})")
        cb0 = S.get("matched_cost_code_view", {})
        rc = cb0.get(next((k for k in cb0 if k.startswith("budget")), ""), {}).get(p, {}).get("pc_newton_causal:C1")
        if rc and rc["log10_ratio"] is not None and np.isfinite(rc["log10_ratio"]):
            bits.append(f"PC-Newton causal C=1 with the code-view cost (unused E charged) "
                        f"{rc['log10_ratio']:+.2f} dec")
        gb = S.get("matched_cost_gnn_eta1.9", {})
        rg = gb.get(next((k for k in gb if k.startswith("budget")), ""), {}).get(p)
        if rg and rg["log10_ratio"] is not None and np.isfinite(rg["log10_ratio"]):
            bits.append(f"GNN at eta = 1.9 instead of 1.5 {rg['log10_ratio']:+.2f} dec ({rg['verdict']})")
        if best:
            bits.append(f"best causal method {best[0]} ({best[1]:+.2f} dec)")
        if bits:
            out.append(f"Matched FLOP rate on {p} (budget = ZNN-Euler at tau={BUDGET_TAUS[0]:g}): " + "; ".join(bits) + ".")
    mu = S.get("multiplicity", {})
    if mu:
        out.append(f"Multiplicity: {mu['n_verdicts_changed_by_bonferroni']} of the matched-cost verdicts change with "
                   f"Bonferroni CIs over the family of {mu['n_family']} comparisons per budget.")
    rk = S["ranking"]["per_problem"]
    if rk:
        la = S.get("lookahead_cost_ratio_max")
        out.append("Kendall tau_b between the method ranking at fixed tau and at matched cost (floor values tied, "
                   "diverged/unstable tied-worst; descriptive): "
                   + "; ".join(f"{p}: {v['kendall_tau_b']:.2f} (n={v['n']}), causal only "
                               f"{v['kendall_tau_b_causal']:.2f} (n={v['n_causal']})" for p, v in rk.items())
                   + (f". Look-ahead formulas cost at most {100 * (la - 1):.0f}% more than Euler per step"
                      if la else "")
                   + ", so cost matching re-ranks mainly the multi-solve PC / Newton variants and the cheap GNN.")
    te = S.get("time_to_eps", {}).get("per_problem", {})
    rhos = [v["rho_opt"] for v in S.get("formula_stability", {}).values()
            if (v.get("expected_order") or 0) >= 4 and v.get("rho_opt") is not None]
    rlo, rhi = (min(rhos), max(rhos)) if rhos else (np.nan, np.nan)
    if te:
        bits = []
        for p in problems:
            r = te.get(p, {})
            for m in ("znn:euler", "znn:uhlig_3_4", "znn:uhlig_4_6"):
                if m in r:
                    v = r[m]
                    bits.append(f"{p} {m[4:]}: " + (f"{v['median_steps']:.0f} steps" if v["median_steps"] is not None
                                                    else "> T"))
        out.append(f"Transient cost at tau = {S['time_to_eps']['tau']:g} (upper-median steps to ||E|| < "
                   f"{S['time_to_eps']['eps']:g}): "
                   + "; ".join(bits) + f". The order >= 4 formulas contract by rho(h_opt) = {rlo:.2f}-{rhi:.2f} "
                     "per step (Euler: 0), so at tau >= 1e-2 some of them are still decaying inside the steady "
                     "window of T = 6 s; their MSSRE there, and the matched-cost ratios flagged 'transient', are "
                     "conservative (they overstate the steady error).")
    uf = S.get("unstable_finite", {})
    if uf.get("n_trajectories"):
        keys_ = sorted({k.split("|")[1] for k in uf["by_method_tau"]})
        out.append(f"{uf['n_trajectories']} closed-loop trajectories grew without crossing the 1e10 divergence "
                   f"threshold (status 'ok', final > initial residual), in: {', '.join(keys_)}; they are counted "
                   "as diverged. For the high-order look-ahead formulas this happens at large tau, where the "
                   "frozen-coefficient stability interval (0, h_max) no longer applies because tau*||J^-1 dJ/dt|| "
                   "is not << h.")
    out.append("Non-causal pc_newton / newton_tracking reach the round-off floor on the affine problems because "
               "one Newton step with the t_{k+1} data solves the sampled problem exactly: an information "
               "advantage, not a numerical one.")
    return out


def write_md(path, S, problems, head, lines):
    L = ["# E9 summary: GNN vs ZNN vs prediction-correction vs Uhlig vs Newton at matched cost", "",
         f"Results: `{S['results']}`; rows: {S['n_rows']}; runner statuses: {S['status_counts']}; effective "
         f"statuses (finite growth counted as unstable): {S['status_counts_effective']}; "
         f"tau grid: {S['taus']}; seeds: {S['seeds']}; B = {S['B']}.",
         f"Cost model: {', '.join(S['cost_source'])} (structured znnbench.costs available: {S['structured_cost_model']}).",
         "", "## Matched-cost definition", "", S.get("matched_cost_definition", ""),
         "", "## Statistics", "", S.get("statistics_note", ""),
         "", "## Headline numbers", ""]
    L += [f"* {x}" for x in lines]
    cz = S.get("censoring", {})
    if cz:
        L += ["", "## Censoring", "", f"Rule: {cz['rule']}. Failed / timed-out trajectories in total: "
                                      f"{cz['n_failed_total']}.", "",
              "| problem | method (primary level) | n | diverged | finite growth | failed | tau with censoring |",
              "|---|---|---|---|---|---|---|"]
        for k, v in cz["per_problem_method"].items():
            if v["n_diverged"] or v["n_finite_growth"] or v["n_failed"]:
                p_, m_ = k.split("|", 1)
                L.append(f"| {p_} | {mlabel(m_)} | {v['n']} | {v['n_diverged']} | {v['n_finite_growth']} | "
                         f"{v['n_failed']} | {', '.join(f'{t:g}' for t in v['taus_with_censoring'])} |")
        L.append("")
        L.append("Every other (problem, method) cell at its primary level is uncensored. GNN levels other than "
                 "eta = 1.5 are in `summary.json -> gnn.stability`.")
    if S.get("confounds"):
        L += ["", "## Confounds", ""] + [f"* {c}" for c in S["confounds"]]
    L += ["", "## Hypotheses", ""]
    for hk, hv in S.get("hypotheses", {}).items():
        L.append(f"* **{hk}**")
        for k, v in hv.items():
            if isinstance(v, dict):
                L.append(f"  * {k}:")
                for kk, vv in v.items():
                    if isinstance(vv, dict):
                        L.append(f"    * {kk}:")
                        L += [f"      * {a}: {b}" for a, b in vv.items()]
                    else:
                        L.append(f"    * {kk}: {vv}")
            else:
                L.append(f"  * {k}: {v}")
    L += ["", f"## MSSRE at fixed tau = {S['fixed_tau']['tau']:g} (median over trajectories; "
              "* = still decaying or growing in the steady window)", "",
          "| method | " + " | ".join(problems) + " | FLOPs/step (linsys) |", "|---|" + "---|" * (len(problems) + 1)]
    for m in head:
        vals = []
        fl = None
        for p in problems:
            r = S["fixed_tau"]["per_problem"].get(p, {}).get(m)
            vals.append((fmt_e(r["mssre_median"], 2) + ("*" if r["pre_asymptotic"] else "")) if r else "--")
            fl = fl or (r or {}).get("flops_per_step")
        L.append(f"| {mlabel(m)} | " + " | ".join(vals) + (f" | {fl:.0f} |" if fl else " | -- |"))
    te = S.get("time_to_eps", {})
    if te.get("per_problem"):
        L += ["", f"## Transient: upper-median steps to ||E|| < {te['eps']:g} (for good) at tau = {te['tau']:g} "
                  f"(right-censored at T = {te['T']:g} s)", "",
              "| method | " + " | ".join(problems) + " |", "|---|" + "---|" * len(problems)]
        for m in head:
            vals = []
            for p in problems:
                r = te["per_problem"].get(p, {}).get(m)
                if not r:
                    vals.append("--")
                elif r["median_steps"] is None:
                    vals.append(f"> T ({100 * r['frac_censored']:.0f}% censored)")
                else:
                    vals.append(f"{r['median_steps']:.0f}" + (f" ({100 * r['frac_censored']:.0f}% cens.)"
                                                              if r["frac_censored"] > 0 else ""))
            L.append(f"| {mlabel(m)} | " + " | ".join(vals) + " |")
    for key0, src, title in [(next(iter(S["matched_cost"])), S["matched_cost"], "costs as implemented"),
                             (next(k for k in S["matched_cost_lu_reuse"] if k.startswith("budget")),
                              S["matched_cost_lu_reuse"], "ideal LU reuse on the affine problems")]:
        L += ["", f"## Matched cost ({key0}, {title}): log10 MSSRE ratio vs ZNN-Euler [95% CI], verdict "
                  "[Bonferroni verdict if different] {flags}", "",
              "| method | " + " | ".join(problems) + " |", "|---|" + "---|" * len(problems)]
        for m in head:
            vals = []
            any_row = False
            for p in problems:
                r = src[key0].get(p, {}).get(m)
                if r:
                    any_row = True
                if r and r["log10_ratio"] is not None and np.isfinite(r["log10_ratio"]):
                    hi = r["ci95"][1]
                    his = "+inf" if hi == "inf" or (isinstance(hi, float) and np.isposinf(hi)) else f"{hi:+.2f}"
                    vb = "" if r["verdict_bonferroni"] == r["verdict"] else f" [{r['verdict_bonferroni']}]"
                    fl = (" {" + "; ".join(r["flags"]) + "}") if r["flags"] else ""
                    vals.append(f"{r['log10_ratio']:+.2f} [{r['ci95'][0]:+.2f}, {his}] {r['verdict']}{vb}{fl}")
                else:
                    vals.append(r["verdict"] if r else "--")
            if any_row:
                L.append(f"| {mlabel(m)} | " + " | ".join(vals) + " |")
    L += ["", "## Figures and tables", ""] + [f"* `{f}`" for f in S["figures"]] + [f"* `{t}`" for t in S["tables"]]
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.exit(main())
