#!/usr/bin/env python
"""E2 analysis: stability maps, root-locus theory, frozen-time and finite-horizon
predictions, initial-error dependence and tau-convergence.

    python experiments/analyze_E2.py --results results/mini/E2 \
        --figdir figures/mini --tabdir results/tables/mini

Inputs: the task results listed in <results>/tasks.jsonl (current build only;
``<id>.parquet`` = one row per trajectory x criterion, ``<id>.profiles.parquet``
= MSSRE at every simulated h).  Outputs: figures fig_E2_*.pdf/png (via
experiments/style.py), LaTeX booktabs tables E2_*.tex, <results>/summary.json and
<results>/SUMMARY.md.

Theory (experiments/tasks/E2_theory.py, independent of znnbench's own
``znn_stability``; the package value is only used as a cross-check):

* look-ahead formula, tau-form coefficients (c_{-1}, c_0, ..., c_{s-1}) (the
  ZNN-Bench h = gamma*tau convention, docs/spec/discretizations.md sec. 1):
      P_h(xi) = c_{-1} xi^s + (c_0 + h) xi^{s-1} + c_1 xi^{s-2} + ... + c_{s-1};
  papers that normalise h by the leading coefficient use h' = h / c_{-1}
  (Sun-Tian-Wang); both are tabulated;
* Adams-Bashforth q: P_h(xi) = xi^q - xi^{q-1} + h sum_j beta_j xi^{q-1-j};
* RK4: |1 - h + h^2/2 - h^3/6 + h^4/24| < 1;
* implicit_euler_err: effective step h/(1+h) < 1 for every h (no upper limit).
h_max = first h at which max |roots| reaches 1 (grid in (0, 4] + bisection).

Frozen-time prediction (time-varying problems): the linear-ZNN error recursion
is sum_i c_i e_{k+1-i} = -(h I + tau M(t_k)) e_k + O(tau^{p+1}) with
M = A^{-1} A_dot; h_ft = sup{h : rho(P_{h + tau mu}) < 1 for every eigenvalue mu
of M(t), t in [0, T]}.  Finite-horizon prediction for the floor criterion:
rho(P_h)^{k_w} = THETA max(F, F_ABS) / ||E(0)||, k_w = (1 - frac) N steps.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import keys, style  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
from matplotlib.colors import LogNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# step-size intervals stated in the literature, converted to h = gamma*tau
# (docs/spec/discretizations.md sec. 2 and 4); merged with describe()['h_range_paper']
LIT_STATED = {"euler": 2.0, "taylor_zhang": 1.0, "zead_4i_6321": 0.8, "ab2": 1.0, "ab3": 6.0 / 11.0,
              "ab4": 0.3}
BOOT_N = 2000
BOOT_SEED = 20262
ACT_ORDER = keys.GROUPS["act"]
ACT_LABEL = {"linear": "linear", "power_sigmoid": "power-sigm.", "hyperbolic_sine": "sinh",
             "sign_bi_power": "sign-bi-pow.", "fixed_time": "fixed-time", "vaf": "VAF",
             "pvaf1": "PVAF-1", "predefined_exp": "pred.-exp"}
CRIT_LABEL = {"bounded": "bounded (MSSRE $\\leq\\|E(0)\\|$)",
              "floor": "floor (MSSRE $\\leq 10F$, protocol)",
              "nodiv": "no divergence ($|x|\\leq 10^{10}$ over $T$, diagnostic)",
              "e4": "E4-P2 (MSSRE $\\leq 0.1\\|E(0)\\|$, no growth, peak $\\leq 10\\|E(0)\\|$)"}
MAP_KINDS = ("main", "control")   # kinds that enter the tau = 1e-2 theory comparison
CENS_LO, CENS_HI = 1e-4, 3.0      # coarse-grid ends: below -> true h_max < 1e-4, above -> >= 3
PROB_LABEL = {"linear_system": "linear system ($n=4$)", "matrix_inversion": "matrix inversion ($n=4$)",
              "linear_system_constA": "linear system, $A$ frozen (control)"}


# ==========================================================================
# loading
# ==========================================================================
def load(results: Path):
    tf = results / "tasks.jsonl"
    tasks = [json.loads(line) for line in open(tf) if line.strip()]
    frames, pframes, missing = [], [], 0
    for t in tasks:
        p = Path(t["done"])
        if not p.exists():
            missing += 1
            continue
        frames.append(pd.read_parquet(p))
        pp = p.with_name(p.name.replace(".parquet", ".profiles.parquet"))
        if pp.exists():
            pframes.append(pd.read_parquet(pp))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    pf = pd.concat(pframes, ignore_index=True) if pframes else pd.DataFrame()
    cfg = json.loads((results / "config.json").read_text())
    return df, pf, cfg, len(tasks), missing


# ==========================================================================
# theory (shared with the task script)
# ==========================================================================
from experiments.tasks.E2_theory import Theory, problem_mus  # noqa: E402


# ==========================================================================
# statistics
# ==========================================================================
def boot_ci(x, stat=np.median, n=BOOT_N, seed=BOOT_SEED, alpha=0.05):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (np.nan, np.nan)
    if len(x) == 1:
        return (float(x[0]), float(x[0]))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    st = np.array([stat(x[i]) for i in idx])
    return (float(np.quantile(st, alpha / 2)), float(np.quantile(st, 1 - alpha / 2)))


def kendall_b(x, y):
    from scipy.stats import kendalltau
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 3:
        return np.nan, np.nan, int(ok.sum())
    # +inf (unbounded) is a legitimate top rank: map to a large finite value
    xx = np.where(np.isinf(x[ok]), 1e300, x[ok])
    yy = np.where(np.isinf(y[ok]), 1e300, y[ok])
    r = kendalltau(xx, yy, variant="b")
    return float(r.statistic), float(r.pvalue), int(ok.sum())


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (dict key -> p; NaN entries ignored)."""
    items = [(k, v) for k, v in pvals.items() if v is not None and np.isfinite(v)]
    items.sort(key=lambda kv: kv[1])
    m, out, run = len(items), {}, 0.0
    for i, (k, v) in enumerate(items):
        run = max(run, min(1.0, (m - i) * v))
        out[k] = run
    return out


def censored_ratio(num, num_c, den, den_c):
    """Elementwise num/den for censored h_max values.  ``*_c`` in {'', 'below',
    'above'}: 'below' = true value < the stored value, 'above' = true value >= it.
    Returns (value, kind) with kind in {'exact', 'upper', 'lower', 'undef'}
    ('upper': the true ratio is <= value; 'undef': both ends censored the same way)."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    nc, dc = np.asarray(num_c, dtype=object), np.asarray(den_c, dtype=object)
    val = num / den
    up = ((nc == "below") & (dc != "below")) | ((dc == "above") & (nc != "above"))
    lo = ((nc == "above") & (dc != "above")) | ((dc == "below") & (nc != "below"))
    und = ((nc == "below") & (dc == "below")) | ((nc == "above") & (dc == "above")) | (up & lo)
    kind = np.where(und, "undef", np.where(up, "upper", np.where(lo, "lower", "exact")))
    val = np.where(und, np.nan, val)
    return val, kind


def summarise_bounded(vals, kinds):
    """Median of possibly censored values; flag '<='/'>=' when at least half are
    upper/lower bounds, 'n/a' when at least half are undefined."""
    vals, kinds = np.asarray(vals, float), np.asarray(kinds, dtype=object)
    n = len(vals)
    if n == 0:
        return np.nan, "n/a", {}
    fr = {k: float(np.mean(kinds == k)) for k in ("exact", "upper", "lower", "undef")}
    if fr["undef"] >= 0.5:
        return np.nan, "n/a", fr
    med = float(np.nanmedian(vals)) if np.isfinite(vals).any() else np.nan
    flag = "<=" if fr["upper"] >= 0.5 else (">=" if fr["lower"] >= 0.5 else "")
    return med, flag, fr


def fmt_flag(v, flag, fmt="{:.3g}"):
    if flag == "n/a" or not np.isfinite(v):
        return "n/a"
    return {"<=": "<=", ">=": ">="}.get(flag, "") + fmt.format(v)


def q25(x):
    return float(np.nanquantile(x, 0.25)) if np.isfinite(x).any() else np.nan


def q75(x):
    return float(np.nanquantile(x, 0.75)) if np.isfinite(x).any() else np.nan


def fit_order(taus, vals, floor, tol=0.15, expected=1.0):
    """Empirical order of vals ~ tau^q (REVIEW_CHECKLIST item 3).  Points at or below
    ``floor`` (bisection resolution) are excluded.  The asymptotic range is the run
    of points at the SMALL-tau end whose consecutive local slopes agree within ``tol``
    (0.15, the tolerance of tests/test_problems_robot.py); the LS slope over that run
    is ``slope_asymptotic``.  Flags: ``two_point`` (the asymptotic fit has only 2
    points, i.e. no local-slope agreement could be checked), ``asymptotic_ok``
    (>= 3 points with agreeing local slopes), ``absurd`` (slope < 0 or > expected + 2).
    ``slope_all`` = LS over every point above the floor."""
    taus, vals = np.asarray(taus, float), np.asarray(vals, float)
    o = np.argsort(-taus)
    taus, vals = taus[o], vals[o]
    ok = np.isfinite(vals) & (vals > floor)
    n = len(taus)
    cons = [float(np.log(vals[i] / vals[i + 1]) / np.log(taus[i] / taus[i + 1]))
            if ok[i] and ok[i + 1] else np.nan for i in range(n - 1)]
    use = np.zeros(n, dtype=bool)
    valid = [i for i in range(n - 1) if np.isfinite(cons[i])]
    if valid:
        j = valid[-1]                                  # smallest-tau valid pair
        use[j] = use[j + 1] = True
        while j - 1 >= 0 and np.isfinite(cons[j - 1]) and abs(cons[j - 1] - cons[j]) <= tol:
            j -= 1
            use[j] = True
    nu = int(use.sum())
    slope = float(np.polyfit(np.log(taus[use]), np.log(vals[use]), 1)[0]) if nu >= 2 else np.nan
    slope_all = float(np.polyfit(np.log(taus[ok]), np.log(vals[ok]), 1)[0]) if ok.sum() >= 2 else np.nan
    return {"taus": taus.tolist(), "values": vals.tolist(), "consecutive": cons,
            "slope_asymptotic": slope, "n_asymptotic": nu, "slope_all": slope_all,
            "n_all": int(ok.sum()), "two_point": nu == 2, "asymptotic_ok": nu >= 3,
            "absurd": bool(np.isfinite(slope) and (slope < 0 or slope > expected + 2))}


# ==========================================================================
# helpers
# ==========================================================================
def formula_order(df_theory):
    """Rows of the maps: look-ahead by (order, instants, name), then AB, RK4, implicit."""
    def k(row):
        fam = {"lookahead": 0, "ab": 1, "rk4": 2, "implicit_euler": 3}.get(row["kind"], 4)
        if row.get("theory_type") == "local":        # inverse-free rows at the bottom, together
            fam = 4
        return (fam, row["expected_order"] if np.isfinite(row["expected_order"]) else 99,
                row["instants"] if np.isfinite(row["instants"]) else 0, row["formula"])
    return [r["formula"] for _, r in sorted(df_theory.iterrows(), key=lambda kv: k(kv[1]))]


def fmt_h(v, cens=""):
    if cens == "above" or (np.isinf(v) and v > 0):
        return "$>3$" if cens == "above" else "$\\infty$"
    if cens == "below":
        return "$<10^{-4}$"
    if not np.isfinite(v):
        return "--"
    if v >= 0.1:
        return f"{v:.3f}"
    if v >= 0.01:
        return f"{v:.4f}"
    return f"{v:.2g}" if v >= 1e-3 else f"{v:.1e}"


def tex_escape(s):
    return str(s).replace("_", "\\_")


def write_tex(path: Path, header, rows, colspec, caption, label):
    lines = ["\\begin{table}[t]", "\\centering", "\\footnotesize",
             f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             f"\\begin{{tabular}}{{{colspec}}}", "\\toprule",
             " & ".join(header) + " \\\\", "\\midrule"]
    for r in rows:
        if r == "\\midrule":
            lines.append(r)
        else:
            lines.append(" & ".join(r) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def cell_stat(g):
    """Median/IQR of h_max over trajectories with censoring bookkeeping."""
    v = g["h_max"].to_numpy(float)
    c = g["censored"].to_numpy(str)
    vv = v.copy()
    vv[c == "below"] = 0.5 * np.nanmin(v[c == "below"]) if (c == "below").any() else np.nan
    med = float(np.nanmedian(vv)) if np.isfinite(vv).any() else np.nan
    n = len(v)
    nb, na = int((c == "below").sum()), int((c == "above").sum())
    cens = "below" if nb > n / 2 else ("above" if na >= n / 2 and med >= np.nanmax(v) else "")
    return pd.Series({"median": med, "q25": q25(vv), "q75": q75(vv), "n": n, "n_below": nb,
                      "n_above": na, "censored": cens, "islands": int((g["n_islands"] > 0).sum())})


# ==========================================================================
# main analysis
# ==========================================================================
def analyze(results: Path, figdir: Path, tabdir: Path):
    figdir.mkdir(parents=True, exist_ok=True)
    tabdir.mkdir(parents=True, exist_ok=True)
    style.FIG_DIR = figdir           # style.savefig writes to style.FIG_DIR
    df, pf, cfg, n_tasks, n_missing = load(results)
    if df.empty:
        raise SystemExit(f"no E2 results under {results}")
    T, frac, theta, F_abs = float(cfg["T"]), float(cfg["frac"]), float(cfg["theta"]), float(cfg["F_abs"])
    tau_main, r_main = float(cfg["tau"]), float(cfg["r_main"])
    summary = {"experiment": "E2", "scale": cfg.get("scale"), "results": str(results),
               "n_tasks": n_tasks, "n_tasks_missing": n_missing}

    errs = df[df.status != "ok"]
    summary["n_error_groups"] = int(len(errs.drop_duplicates(["task_id", "activation", "seed"])))
    if len(errs):
        summary["errors"] = (errs.groupby(["formula", "activation"]).message
                             .first().str.slice(-300).to_dict())
        summary["errors"] = {f"{k[0]}|{k[1]}": v for k, v in summary["errors"].items()}
    ok = df[df.status == "ok"].copy()

    # ---------------- theory per formula
    th_rows = []
    theories = {}
    for f, g in ok.groupby("formula"):
        desc = json.loads(g.disc_describe.iloc[0])
        imp = bool(g["disc_implicit"].iloc[0]) if "disc_implicit" in g else False
        th = Theory(f, desc, implicit=imp)
        theories[f] = th
        hm = th.h_max()
        ho, ro = th.h_opt(hm)
        pkg = desc.get("h_max", np.nan)
        th_rows.append({"formula": f, "kind": th.kind, "theory_type": th.theory_type, "reason": th.reason,
                        "h_theory": hm, "h_opt": ho, "rho_opt": ro,
                        "c_lead": th.c_lead, "h_theory_c0norm": hm / th.c_lead if th.kind == "lookahead" else np.nan,
                        "h_pkg": float(pkg) if pkg is not None else np.nan,
                        "expected_order": float(desc.get("expected_order") or np.nan),
                        "instants": float(desc.get("instants", len(desc.get("coeffs", [])) or np.nan)),
                        "h_range_paper": desc.get("h_range_paper"), "label": desc.get("label", "")})
    TH = pd.DataFrame(th_rows).set_index("formula")
    TH["pkg_abs_diff"] = (TH.h_theory - TH.h_pkg).abs()

    def _stated(f):
        hr = TH.loc[f, "h_range_paper"]
        if isinstance(hr, (list, tuple)) and len(hr) == 2 and hr[1] is not None:
            return float(hr[1])
        return LIT_STATED.get(f, np.nan)
    TH["h_stated"] = [_stated(f) for f in TH.index]
    # direct np.roots check of the boundary: max|roots| just below / above h_max
    chk = []
    for f in TH.index:
        th, hm = theories[f], TH.loc[f, "h_theory"]
        if th.kind in ("lookahead", "ab") and np.isfinite(hm) and hm > 0:
            rb = [np.max(np.abs(np.roots(th.polys(hm * fac)[0]))) for fac in (1 - 1e-6, 1 + 1e-6)]
            chk.append({"formula": f, "rho_below": float(rb[0]), "rho_above": float(rb[1])})
    summary["np_roots_boundary_check"] = chk
    summary["np_roots_boundary_ok"] = bool(all(c["rho_below"] < 1 <= c["rho_above"] + 1e-9 for c in chk))
    order = formula_order(TH.reset_index())
    summary["theory"] = {f: {"h_max": float(TH.loc[f, "h_theory"]), "h_opt": float(TH.loc[f, "h_opt"]),
                             "c_lead": float(TH.loc[f, "c_lead"]),
                             "h_max_c0norm": float(TH.loc[f, "h_theory_c0norm"]),
                             "kind": TH.loc[f, "kind"], "theory_type": TH.loc[f, "theory_type"],
                             "reason": TH.loc[f, "reason"]} for f in order}
    summary["theory_vs_package_max_abs_diff"] = float(np.nanmax(TH.pkg_abs_diff.to_numpy(float))) \
        if np.isfinite(TH.pkg_abs_diff.to_numpy(float)).any() else None

    # ---------------- per-trajectory predictions for the linear activation
    lin = ok[ok.activation == "linear"].copy()
    mus_cache = {}
    ft = {}
    for (prob, pkey, seed, tau, Tr), gg in lin.groupby(["problem", "problem_key", "seed", "tau", "T"]):
        frozen = "constA" in prob
        if (pkey, frozen, seed, Tr) not in mus_cache:
            mus_cache[(pkey, frozen, seed, Tr)] = problem_mus(pkey, seed, float(Tr), freeze_A=frozen,
                                                              by_time=True)
        for f in gg.formula.unique():
            th = theories[f]
            ft[(f, prob, seed, tau, Tr)] = th.h_frozen_time(th.effective_mus(mus_cache[(pkey, frozen, seed, Tr)]),
                                                            tau, TH.loc[f, "h_opt"], TH.loc[f, "h_theory"])
    lin["h_theory"] = lin.formula.map(TH.h_theory)
    lin["theory_type"] = lin.formula.map(TH.theory_type)
    lin["h_frozen"] = [ft[(f, p, s, t, Tr)] for f, p, s, t, Tr in
                       zip(lin.formula, lin.problem, lin.seed, lin.tau, lin["T"])]
    N = np.round(lin["T"].to_numpy(float) / lin.tau.to_numpy(float))
    kw = np.floor((1 - frac) * N)
    rho_crit = (theta * np.maximum(lin.floor_F.to_numpy(float), F_abs) / lin.e0.to_numpy(float)) ** (1.0 / kw)
    lin["rho_crit"] = rho_crit
    lin["h_fh"] = [theories[f].h_finite_horizon(rc, TH.loc[f, "h_theory"], TH.loc[f, "h_opt"])
                   if c == "floor" else TH.loc[f, "h_theory"]
                   for f, rc, c in zip(lin.formula, lin.rho_crit, lin.criterion)]
    lin["ratio"] = lin.h_max / lin.h_theory
    lin["ratio_frozen"] = lin.h_max / lin.h_frozen
    lin["ratio_fh"] = lin.h_max / lin.h_fh

    # ---------------- aggregation of the maps
    keycols = ["kind", "formula", "problem", "tau", "activation", "r", "criterion"]
    agg = ok.groupby(keycols).apply(cell_stat, include_groups=False).reset_index()
    agg.to_csv(results / "E2_cells.csv", index=False)

    main = agg[(agg.kind == "main") & np.isclose(agg.tau, tau_main)]
    problems_main = [p for p in ("linear_system", "matrix_inversion") if p in set(main.problem)]
    acts = [a for a in ACT_ORDER if a in set(main.activation)] + \
        sorted(set(main.activation) - set(ACT_ORDER))
    forms = [f for f in order if f in set(main.formula)]

    # ---------------- figure: heat maps
    for crit in ("floor", "bounded", "nodiv", "e4"):
        fig_heatmap(main, crit, problems_main, forms, acts, r_main, TH)

    # ---------------- linear: theory vs measured
    linm = lin[np.isclose(lin.r, r_main)]
    lin_cells = (linm.groupby(["kind", "formula", "problem", "tau", "criterion"])
                 .agg(h_meas=("h_max", "median"), q25=("h_max", q25), q75=("h_max", q75),
                      h_theory=("h_theory", "first"), theory_type=("theory_type", "first"),
                      h_frozen=("h_frozen", "median"),
                      h_fh=("h_fh", "median"), ratio=("ratio", "median"),
                      ratio_frozen=("ratio_frozen", "median"), ratio_fh=("ratio_fh", "median"),
                      n=("h_max", "size"), n_above=("censored", lambda s: int((s == "above").sum())))
                 .reset_index())
    lin_cells.to_csv(results / "E2_linear_theory.csv", index=False)
    fig_theory_scatter(lin_cells, tau_main)

    # transient leakage into the formula floor F: the slowest root at h_opt leaves
    # ~rho_opt^{k_w} ||E(0)|| at the start of the steady window; when that is not small
    # against the measured F, F is partly a transient residue (looser floor criterion)
    kw_main = int(np.floor((1 - frac) * round(T / tau_main)))
    ftc = {}
    fl_rows = lin[(lin.kind == "main") & np.isclose(lin.tau, tau_main) & np.isclose(lin.r, r_main)
                  & (lin.criterion == "floor")]
    for (f, prob), g in fl_rows.groupby(["formula", "problem"]):
        ro = float(TH.loc[f, "rho_opt"])
        if not np.isfinite(ro):
            continue
        resid = ro ** kw_main * r_main
        Fm = float(np.nanmedian(g.floor_F))
        ftc[f"{f}|{prob}"] = {"rho_opt": ro, "k_w": kw_main, "transient_residue": resid, "F_median": Fm,
                              "F_rule": str(g.floor_rule.iloc[0]), "contaminated": bool(resid > 0.1 * Fm)}
    summary["floor_transient_check"] = ftc

    headline = {}
    lc = lin_cells[np.isclose(lin_cells.tau, tau_main) & np.isfinite(lin_cells.h_theory)
                   & lin_cells.kind.isin(MAP_KINDS) & (lin_cells.theory_type == "exact")]
    # seed-cluster spread: the instance (seed) is the statistical unit (the B directions
    # of one instance are near-identical pseudo-replicates), so the per-seed median
    # ratio over formulas is reported as [min, max] over seeds
    linseed = lin[np.isclose(lin.r, r_main) & np.isclose(lin.tau, tau_main) & lin.kind.isin(MAP_KINDS)
                  & (lin.theory_type == "exact") & np.isfinite(lin.h_theory)]
    seed_cells = (linseed.groupby(["formula", "problem", "criterion", "seed"])
                  .agg(ratio=("ratio", "median"), ratio_frozen=("ratio_frozen", "median")).reset_index())
    within = (linseed.groupby(["formula", "problem", "criterion", "seed"]).h_max
              .agg(lambda v: float(np.nanmax(v) / np.nanmin(v) - 1) if np.nanmin(v) > 0 else np.nan))
    between = (seed_cells.groupby(["formula", "problem", "criterion"]).ratio
               .agg(lambda v: float(np.nanmax(v) / np.nanmin(v) - 1) if np.nanmin(v) > 0 else np.nan))
    summary["pseudo_replication"] = {
        "within_seed_relative_range_median": float(np.nanmedian(within.to_numpy(float))),
        "between_seed_relative_range_median": float(np.nanmedian(between.to_numpy(float))),
        "note": "linear activation, tau = 1e-2, ||E(0)|| = 1: relative range (max/min - 1) of h_max over the "
                "B directions of one instance vs over instances (per-seed medians)"}
    for crit in ("bounded", "floor", "nodiv", "e4"):
        for prob in sorted(set(lc.problem)):
            sub = lc[(lc.criterion == crit) & (lc.problem == prob)]
            if sub.empty:
                continue
            rr = sub.ratio.to_numpy(float)
            ci = boot_ci(rr)
            sc = seed_cells[(seed_cells.criterion == crit) & (seed_cells.problem == prob)]
            per_seed = sc.groupby("seed").ratio.median().to_numpy(float)
            per_seed_ft = sc.groupby("seed").ratio_frozen.median().to_numpy(float)
            kt = kendall_b(sub.h_theory, sub.h_meas)
            worst = sub.iloc[int(np.nanargmax(np.abs(rr - 1)))] if np.isfinite(rr).any() else None
            entry = {"median_ratio_meas_over_theory": float(np.nanmedian(rr)), "ci95": ci,
                     "seed_range": [float(np.nanmin(per_seed)), float(np.nanmax(per_seed))]
                     if np.isfinite(per_seed).any() else [np.nan, np.nan],
                     "seed_range_frozen_time": [float(np.nanmin(per_seed_ft)), float(np.nanmax(per_seed_ft))]
                     if np.isfinite(per_seed_ft).any() else [np.nan, np.nan],
                     "n_seeds": int(len(per_seed)),
                     "n_formulas": int(np.isfinite(rr).sum()),
                     "max_abs_dev": float(np.nanmax(np.abs(rr - 1))) if np.isfinite(rr).any() else None,
                     "worst_formula": None if worst is None else worst.formula,
                     "kendall_tau_b_theory_vs_measured": kt[0], "kendall_p": kt[1]}
            if prob != "linear_system_constA":
                rf = sub.ratio_frozen.to_numpy(float)
                entry["median_ratio_meas_over_frozen_time"] = float(np.nanmedian(rf))
                entry["ci95_frozen_time"] = boot_ci(rf)
                entry["median_abs_dev_root_locus"] = float(np.nanmedian(np.abs(rr - 1)))
                entry["median_abs_dev_frozen_time"] = float(np.nanmedian(np.abs(rf - 1)))
                entry["max_abs_dev_frozen_time"] = float(np.nanmax(np.abs(rf - 1))) \
                    if np.isfinite(rf).any() else None
            if crit == "floor":
                rfh = sub.ratio_fh.to_numpy(float)
                entry["median_ratio_meas_over_finite_horizon"] = float(np.nanmedian(rfh))
                entry["ci95_finite_horizon"] = boot_ci(rfh)
            headline[f"{crit}|{prob}"] = entry
    summary["linear_theory_vs_measured"] = headline
    summary["inverse_free_local_theory"] = inverse_free_block(lin, lin_cells, TH, tau_main, r_main)

    # ---------------- tau convergence (linear, r_main)
    # At fixed T = 6 s, N = T/tau grows as tau -> 0, so two O(tau) effects overlap:
    # the time-variation shift O(tau ||A^-1 A_dot||) (negative) and the finite-horizon
    # detection limit O(1/N) (positive for the bounded criterion).  The frozen-A control
    # (run at every tau) carries only the second; the paired per-trajectory ratio
    # h(LS)/h(control) (same A(0), seeds, directions and x0) carries only the first.
    tau_res = {}
    sens = lin_cells[lin_cells.kind.isin(["main", "control", "tau_sens"]) & np.isfinite(lin_cells.h_theory)
                     & (lin_cells.theory_type == "exact")]
    fin_rows = ok[(ok.censored == "") & (ok.h_max > 0)]
    res_floor = float(np.nanmedian((fin_rows.h_max_err / fin_rows.h_max).to_numpy(float)))
    for (f, prob, crit), g in sens.groupby(["formula", "problem", "criterion"]):
        if g.tau.nunique() < 2 or crit not in ("bounded", "floor"):
            continue
        g = g.sort_values("tau", ascending=False)
        signed = g.ratio.to_numpy(float) - 1
        e = fit_order(g.tau.to_numpy(float), np.abs(signed), floor=3 * res_floor)
        e["signed"] = signed.tolist()
        e["ratio_frozen"] = g.ratio_frozen.to_numpy(float).tolist()
        tau_res[f"{f}|{prob}|{crit}"] = e
    summary["tau_convergence"] = tau_res
    summary["bisection_relative_resolution_median"] = res_floor
    # time-variation shift, paired per trajectory
    shift_res, shift_cells = {}, pd.DataFrame()
    lin_ts = lin[lin.kind.isin(["main", "control", "tau_sens"])]
    lsr = lin_ts[(lin_ts.problem == "linear_system") & np.isclose(lin_ts.r, r_main)]
    ctr = lin_ts[(lin_ts.problem == "linear_system_constA") & np.isclose(lin_ts.r, r_main)]
    if len(lsr) and len(ctr):
        kc = ["formula", "tau", "criterion", "seed", "b"]
        mm = lsr[kc + ["h_max", "censored"]].merge(ctr[kc + ["h_max", "censored"]], on=kc,
                                                    suffixes=("_ls", "_c"))
        mm = mm[(mm.censored_ls == "") & (mm.censored_c == "")]
        mm["shift"] = mm.h_max_ls / mm.h_max_c - 1
        shift_cells = (mm.groupby(["formula", "tau", "criterion"])["shift"]
                       .agg(["median", q25, q75, "size"]).reset_index())
        for (f, crit), g in shift_cells.groupby(["formula", "criterion"]):
            if g.tau.nunique() < 2 or crit not in ("bounded", "floor"):
                continue
            g = g.sort_values("tau", ascending=False)
            e = fit_order(g.tau.to_numpy(float), np.abs(g["median"].to_numpy(float)), floor=3 * res_floor)
            e["signed"] = g["median"].to_numpy(float).tolist()
            shift_res[f"{f}|{crit}"] = e
        shift_cells.to_csv(results / "E2_time_variation_shift.csv", index=False)
    summary["time_variation_shift"] = shift_res

    def _slopes(dct, suffix):
        v = [e["slope_asymptotic"] for k, e in dct.items() if k.endswith(suffix)
             and np.isfinite(e["slope_asymptotic"])]
        return {"median": float(np.median(v)) if v else None, "q25": q25(np.array(v)) if v else None,
                "q75": q75(np.array(v)) if v else None, "n": len(v)}
    summary["tau_slope_overview"] = {
        "root_locus_dev_linear_system_bounded": _slopes({k: e for k, e in tau_res.items()
                                                         if k.split("|")[1] == "linear_system"}, "bounded"),
        "root_locus_dev_matrix_inversion_bounded": _slopes({k: e for k, e in tau_res.items()
                                                            if k.split("|")[1] == "matrix_inversion"}, "bounded"),
        "finite_horizon_control_bounded": _slopes({k: e for k, e in tau_res.items()
                                                   if k.split("|")[1] == "linear_system_constA"}, "bounded"),
        "time_variation_shift_bounded": _slopes(shift_res, "bounded"),
    }
    if tau_res:
        fig_tau(sens, shift_cells, forms)

    # ---------------- nonlinear: r dependence (censoring-aware)
    # Censored h_max values are bounds (below: < 1e-4, above: >= 3), so ratios built from
    # them are bounds too; formulas whose two ends are censored the same way (e.g.
    # implicit_euler_err, '>3' everywhere) give an undefined ratio and are excluded
    # instead of contributing a spurious 1.0.
    rdep, chat = {}, {}
    nl = main[main.activation != "linear"]
    trj = ok[(ok.kind == "main") & np.isclose(ok.tau, tau_main)]
    idx = ["formula", "problem", "seed", "b"]

    def _cell_ratio(j, num, den):
        """per (formula, problem) median of a censored per-trajectory ratio."""
        val, kind = censored_ratio(j[f"h_max_{num}"], j[f"censored_{num}"], j[f"h_max_{den}"],
                                   j[f"censored_{den}"])
        jj = pd.DataFrame({"val": val, "kind": kind}, index=j.index)
        cells = []
        for (f, pr), g in jj.groupby(level=["formula", "problem"]):
            m, flag, _ = summarise_bounded(g.val, g.kind)
            cells.append((f, pr, m, flag))
        cdf = pd.DataFrame(cells, columns=["formula", "problem", "val", "flag"])
        ck = cdf.flag.map({"": "exact", "<=": "upper", ">=": "lower", "n/a": "undef"}).to_numpy(object)
        med, flag, fr = summarise_bounded(cdf.val, ck)
        ci = boot_ci(cdf.val.to_numpy(float)[ck != "undef"])
        return med, flag, fr, ci, int((ck != "undef").sum())

    for crit in ("bounded", "floor", "nodiv"):
        for a in acts:
            if a == "linear":
                continue
            sub = trj[(trj.activation == a) & (trj.criterion == crit)]
            if sub.empty:
                continue
            rs = sorted(sub.r.unique())
            if len(rs) < 2:
                continue
            hi_ = sub[np.isclose(sub.r, rs[-1])].set_index(idx)[["h_max", "censored"]]
            lo_ = sub[np.isclose(sub.r, rs[0])].set_index(idx)[["h_max", "censored"]]
            j = hi_.join(lo_, lsuffix="_hi", rsuffix="_lo", how="inner")
            sh_med, sh_flag, sh_fr, sh_ci, sh_n = _cell_ratio(j, "hi", "lo")
            med_by_r, flag_by_r = {}, {}
            for r in rs:
                cc = nl[(nl.activation == a) & (nl.criterion == crit) & np.isclose(nl.r, r)]
                med_by_r[f"{r:g}"] = float(np.nanmedian(cc["median"])) if len(cc) else np.nan
                fb = float(np.mean(cc.censored == "below")) if len(cc) else 0.0
                fa = float(np.mean(cc.censored == "above")) if len(cc) else 0.0
                flag_by_r[f"{r:g}"] = "<" if fb > 0.5 else (">=" if fa > 0.5 else "")
            # h_max relative to linear activation at r_main, per cell (cells censored
            # below are bounded by the grid end 1e-4, not by the 5e-5 plotting value)
            cm = main[(main.criterion == crit) & np.isclose(main.r, r_main)].set_index(["formula", "problem"])
            ca = cm[cm.activation == a][["median", "censored"]]
            cl = cm[cm.activation == "linear"][["median", "censored"]]
            jj = ca.join(cl, lsuffix="_a", rsuffix="_l", how="inner")
            na_ = np.where(jj.censored_a == "below", CENS_LO, jj.median_a)
            nl_ = np.where(jj.censored_l == "below", CENS_LO, jj.median_l)
            rv, rk = censored_ratio(na_, jj.censored_a, nl_, jj.censored_l)
            rel_med, rel_flag, _ = summarise_bounded(rv, rk)
            rdep[f"{crit}|{a}"] = {
                "median_h_max_by_r": med_by_r,
                "censor_flag_by_r": flag_by_r,
                "shrink_rmax_over_rmin_median": sh_med,
                "shrink_flag": sh_flag,
                "shrink_fractions": sh_fr,
                "shrink_ci95": sh_ci,
                "shrink_n_cells": sh_n,
                "rel_to_linear_at_rmain_median": rel_med,
                "rel_to_linear_flag": rel_flag,
                "rel_to_linear_ci95": boot_ci(rv[rk != "undef"]),
                "frac_cells_below_grid_at_rmain": float(np.mean(
                    nl[(nl.activation == a) & (nl.criterion == crit) & np.isclose(nl.r, r_main)].censored
                    == "below")) if len(nl) else np.nan,
            }
    # blow-up vs bounded chattering: h_max(nodiv) / h_max(bounded) per activation and r
    for a in acts:
        for r in sorted(trj.r.unique()):
            sb = trj[(trj.activation == a) & (trj.criterion == "bounded") & np.isclose(trj.r, r)]
            sn = trj[(trj.activation == a) & (trj.criterion == "nodiv") & np.isclose(trj.r, r)]
            if sb.empty or sn.empty:
                continue
            j = (sn.set_index(idx)[["h_max", "censored"]]
                 .join(sb.set_index(idx)[["h_max", "censored"]], lsuffix="_n", rsuffix="_b", how="inner"))
            m, flag, fr, ci, n = _cell_ratio(j, "n", "b")
            chat[f"{a}|{r:g}"] = {"median_nodiv_over_bounded": m, "flag": flag, "ci95": ci, "n_cells": n}
    summary["nodiv_over_bounded"] = chat
    summary["nonlinear_r_dependence"] = rdep

    # linearised prediction for activations with finite slope0 (small r)
    from znnbench import activations as A
    linpred = {}
    r_min = min(cfg["r_list"])
    for a in acts:
        try:
            s0 = float(A.get(a).slope0)
        except Exception:
            continue
        if a == "linear" or not np.isfinite(s0) or s0 <= 0:
            continue
        sub = main[(main.activation == a) & (main.criterion == "bounded") & np.isclose(main.r, r_min)]
        sub = sub[sub.formula.map(lambda f: np.isfinite(TH.loc[f, "h_theory"]))]
        pred = sub.formula.map(TH.h_theory) / s0
        ratio = (sub["median"] / pred).to_numpy(float)
        linpred[a] = {"slope0": s0, "median_ratio_meas_over_hlin_div_slope0": float(np.nanmedian(ratio)),
                      "ci95": boot_ci(ratio), "n_cells": int(np.isfinite(ratio).sum()), "r": r_min}
    summary["linearised_prediction_small_error"] = linpred

    # ranking agreement (theory vs measured: only formulas with a finite root-locus
    # value, so the unconditionally stable / no-theory controls cannot pad the
    # concordance; LS vs MI: every formula).  Holm-adjusted p over the whole family.
    rank = {}
    for crit in ("bounded", "floor"):
        for a in acts:
            m = main[(main.activation == a) & (main.criterion == crit) & np.isclose(main.r, r_main)]
            if len(problems_main) == 2:
                p1 = m[m.problem == problems_main[0]].set_index("formula")["median"]
                p2 = m[m.problem == problems_main[1]].set_index("formula")["median"]
                common = p1.index.intersection(p2.index)
                rank[f"{crit}|{a}|problems"] = kendall_b(p1[common], p2[common])
            for prob in problems_main:
                pm = m[m.problem == prob].set_index("formula")["median"]
                thv = TH.h_theory.reindex(pm.index)
                keep = np.isfinite(thv.to_numpy(float)) & (TH.theory_type.reindex(pm.index) == "exact").to_numpy()
                rank[f"{crit}|{a}|theory_vs_{prob}"] = kendall_b(thv[keep], pm[keep])
    p_holm = holm({k: v[1] for k, v in rank.items()})
    summary["kendall_tau_b"] = {k: {"tau_b": v[0], "p": v[1], "p_holm": p_holm.get(k, np.nan), "n": v[2]}
                                for k, v in rank.items()}
    summary["kendall_family_size"] = len(p_holm)

    # ---------------- figures: r dependence, profiles
    for prob in problems_main:
        for crit in ("bounded", "floor", "nodiv"):
            fig_rdep(main, prob, crit, forms, acts, TH)
    if not pf.empty:
        fig_profiles(pf, cfg, TH, forms, problems_main)

    # ---------------- tables
    table_theory(TH, lin_cells, order, tau_main, tabdir)
    for prob in problems_main:
        table_map(main, prob, forms, acts, r_main, tabdir)
    table_rdep(summary, acts, cfg, tabdir)
    if tau_res:
        table_tau(tau_res, summary.get("time_variation_shift", {}), forms, tabdir)

    # ---------------- review additions: horizon, unit dependence, E4 criterion, thresholds, censoring
    hz, hz_cells = horizon_block(lin, tau_main, r_main)
    summary["horizon"] = hz
    if len(hz_cells):
        hz_cells.to_csv(results / "E2_horizon.csv", index=False)
    sc_res, sc_J = scaled_block(ok, tau_main, cfg["r_list"])
    summary["unit_dependence"] = sc_res
    if len(sc_J):
        fig_units(sc_J, forms, acts)
    summary["e4_vs_bounded"] = e4_block(ok, tau_main)
    summary["threshold_sensitivity"] = threshold_block(pf, ok, tau_main)
    summary["censoring"] = censoring_block(ok, df, n_tasks, n_missing)

    # ---------------- cost
    cost = {}
    if "wall_coarse" in ok:
        g = ok.drop_duplicates(["task_id", "activation", "seed"])
        rs = g.row_steps.to_numpy(float)
        cost["sim_wall_total_s"] = float(g.wall_sim.sum())
        cost["group_wall_total_s"] = float(g.group_wall.sum())
        cost["wall_per_step_coarse_batch_median_s"] = float(np.nanmedian(g.wall_coarse / g.steps_coarse))
        cost["wall_per_row_step_median_s"] = float(np.nanmedian(g.wall_sim / rs))
        cost["rows_coarse_median"] = float(np.nanmedian(g.rows_coarse))
        cost["n_groups"] = int(len(g))
        cost["sims_per_group_median"] = float(np.nanmedian(g.n_sim))
        tw = df.drop_duplicates("task_id")
        cost["task_wall_total_s"] = float(tw.task_wall.sum())
        cost["task_wall_max_s"] = float(tw.task_wall.max())
        cost["peak_rss_mb_max"] = float(tw.peak_rss_mb.max())
        by_kind = tw.groupby("kind").task_wall.agg(["count", "sum", "max"])
        cost["by_kind"] = {k: {"tasks": int(v["count"]), "wall_s": float(v["sum"]), "max_s": float(v["max"])}
                           for k, v in by_kind.iterrows()}
    summary["cost"] = cost

    # ---------------- write summary
    (results / "summary.json").write_text(json.dumps(_clean(summary), indent=1))
    write_summary_md(results, summary, cfg, TH, order, problems_main, acts)
    return summary


def inverse_free_block(lin, lin_cells, TH, tau_main, r_main):
    """CDTZNN-type schemes with a LOCAL linearised theory (E2_theory): measured vs the
    root locus of the x-recursion, vs the frozen-time prediction including the
    M-block (shift tau(mu_i + mu_j)), vs the exact-inverse twin (paired per
    trajectory), and the horizon effect T = 24 vs T = 6."""
    out = {}
    loc = [f for f in TH.index if TH.loc[f, "theory_type"] == "local"]
    for f in loc:
        twin = None
        m = str(TH.loc[f, "reason"])
        for cand in TH.index:
            if cand != f and cand in m:
                twin = cand
        sub = lin[(lin.formula == f) & np.isclose(lin.tau, tau_main) & np.isclose(lin.r, r_main)]
        for (kind, prob, crit), g in sub.groupby(["kind", "problem", "criterion"]):
            if kind not in ("main", "control", "horizon"):
                continue
            e = {"T": float(g["T"].iloc[0]), "h_meas_median": float(np.nanmedian(g.h_max)),
                 "h_local_theory": float(TH.loc[f, "h_theory"]),
                 "ratio_local_theory": float(np.nanmedian(g.h_max / g.h_theory)),
                 "h_frozen_time_with_M_block_median": float(np.nanmedian(g.h_frozen)),
                 "ratio_frozen_time_with_M_block": float(np.nanmedian(g.h_max / g.h_frozen))}
            if twin is not None:
                tw = lin[(lin.formula == twin) & (lin.kind == kind) & (lin.problem == prob)
                         & (lin.criterion == crit) & np.isclose(lin.tau, tau_main) & np.isclose(lin.r, r_main)]
                j = g.merge(tw, on=["seed", "b"], suffixes=("", "_tw"))
                if len(j):
                    e["twin"] = twin
                    e["ratio_to_exact_inverse_twin"] = float(np.nanmedian(j.h_max / j.h_max_tw))
            out[f"{f}|{kind}|{prob}|{crit}"] = e
    return out


def horizon_block(lin, tau_main, r_main):
    """Paired h_max(T_long) / h_max(T = 6) per formula (linear activation, ||E(0)|| = 1)."""
    hz = lin[(lin.kind == "horizon") & np.isclose(lin.tau, tau_main) & np.isclose(lin.r, r_main)]
    if hz.empty:
        return {}, pd.DataFrame()
    base = lin[lin.kind.isin(MAP_KINDS) & np.isclose(lin.tau, tau_main) & np.isclose(lin.r, r_main)]
    kc = ["formula", "problem", "criterion", "seed", "b"]
    j = hz[kc + ["h_max", "censored", "h_theory", "h_frozen", "theory_type", "T"]].merge(
        base[kc + ["h_max", "censored"]], on=kc, suffixes=("_long", "_short"))
    j = j[(j.censored_long == "") & (j.censored_short == "")]
    j["ratio_long_short"] = j.h_max_long / j.h_max_short
    j["ratio_long_theory"] = j.h_max_long / j.h_theory
    j["ratio_short_theory"] = j.h_max_short / j.h_theory
    j["ratio_long_frozen"] = j.h_max_long / j.h_frozen
    cells = (j.groupby(["formula", "problem", "criterion", "theory_type"])
             [["ratio_long_short", "ratio_long_theory", "ratio_short_theory", "ratio_long_frozen"]]
             .median().reset_index())
    res = {}
    for (prob, crit, tt), g in cells.groupby(["problem", "criterion", "theory_type"]):
        res[f"{prob}|{crit}|{tt}"] = {
            "T_long": float(j["T"].iloc[0]), "n_formulas": int(len(g)),
            "median_ratio_long_over_short": float(g.ratio_long_short.median()),
            "min_ratio_long_over_short": float(g.ratio_long_short.min()),
            "worst_formula": str(g.iloc[int(np.argmin(g.ratio_long_short.to_numpy()))].formula),
            "median_meas_over_theory_short": float(g.ratio_short_theory.median()),
            "median_meas_over_theory_long": float(g.ratio_long_theory.median()),
            "median_meas_over_frozen_long": float(g.ratio_long_frozen.median()),
            "max_abs_dev_theory_short": float(np.nanmax(np.abs(g.ratio_short_theory - 1))),
            "max_abs_dev_theory_long": float(np.nanmax(np.abs(g.ratio_long_theory - 1)))}
    return res, cells


def scaled_block(ok, tau_main, r_list):
    """Paired scaled-error arm: unit-dependence factor h_max(s)/h_max(1) at the same
    physical ||E(0)|| (censoring-aware), and the argument-equivalence test
    h_max(r, s) vs h_max(s r, 1)."""
    sc = ok[(ok.kind == "scaled") & np.isclose(ok.tau, tau_main)]
    if sc.empty:
        return {}, pd.DataFrame()
    base = ok[(ok.kind == "main") & np.isclose(ok.tau, tau_main)]
    kc = ["formula", "problem_key", "activation", "criterion", "seed", "b"]
    rows = []
    for svalue, g in sc.groupby("error_scale"):
        j = g[kc + ["r", "h_max", "censored"]].merge(base[kc + ["r", "h_max", "censored"]], on=kc + ["r"],
                                                   suffixes=("_s", "_1"))
        val, kind = censored_ratio(j.h_max_s, j.censored_s, j.h_max_1, j.censored_1)
        j = j.assign(factor=val, fkind=kind, error_scale=float(svalue), test="same_r")
        rows.append(j)
        # argument equivalence: scaled at r vs unscaled at s*r
        b2 = base.copy()
        b2["r_s"] = b2.r / float(svalue)
        j2 = g[kc + ["r", "h_max", "censored"]].merge(
            b2[kc + ["r_s", "h_max", "censored"]].rename(columns={"r_s": "r"}), on=kc + ["r"],
            suffixes=("_s", "_1"))
        if len(j2):
            v2, k2 = censored_ratio(j2.h_max_s, j2.censored_s, j2.h_max_1, j2.censored_1)
            rows.append(j2.assign(factor=v2, fkind=k2, error_scale=float(svalue), test="arg_equiv"))
    J = pd.concat(rows, ignore_index=True)
    out = {}
    for (test, svalue, crit, act), g in J.groupby(["test", "error_scale", "criterion", "activation"]):
        cells = []
        for (f, pk, r), gg in g.groupby(["formula", "problem_key", "r"]):
            m, flag, _ = summarise_bounded(gg.factor, gg.fkind)
            cells.append((f, pk, r, m, flag))
        cdf = pd.DataFrame(cells, columns=["formula", "problem_key", "r", "val", "flag"])
        ck = cdf.flag.map({"": "exact", "<=": "upper", ">=": "lower", "n/a": "undef"}).to_numpy(object)
        med, flag, fr = summarise_bounded(cdf.val, ck)
        lv = np.abs(np.log10(cdf.val.to_numpy(float)[ck == "exact"]))
        by_r = {f"{r:g}": float(np.nanmedian(cdf[np.isclose(cdf.r, r)].val)) if (np.isclose(cdf.r, r)).any()
                else np.nan for r in r_list}
        out[f"{test}|{svalue:g}|{crit}|{act}"] = {
            "median_factor": med, "flag": flag, "by_r": by_r, "n_cells": int(len(cdf)),
            "frac_cells_changed_gt_25pct": float(np.mean(lv > np.log10(1.25))) if len(lv) else np.nan,
            "median_abs_log10_factor": float(np.nanmedian(lv)) if len(lv) else np.nan}
    return out, J


def e4_block(ok, tau_main):
    """h_max under E4's P2 criterion relative to the bounded criterion (paired)."""
    trj = ok[(ok.kind == "main") & np.isclose(ok.tau, tau_main)]
    kc = ["formula", "problem", "activation", "r", "seed", "b"]
    a = trj[trj.criterion == "e4"][kc + ["h_max", "censored"]]
    b = trj[trj.criterion == "bounded"][kc + ["h_max", "censored"]]
    if a.empty or b.empty:
        return {}
    j = a.merge(b, on=kc, suffixes=("_e4", "_b"))
    val, kind = censored_ratio(j.h_max_e4, j.censored_e4, j.h_max_b, j.censored_b)
    j = j.assign(val=val, kind=kind)
    out = {}
    for (act, r), g in j.groupby(["activation", "r"]):
        m, flag, fr = summarise_bounded(g.val, g.kind)
        out[f"{act}|{r:g}"] = {"median_e4_over_bounded": m, "flag": flag,
                               "frac_equal_within_1pct": float(np.mean(np.abs(g.val - 1) < 0.01)),
                               "n": int(len(g))}
    return out


def _coarse_hmax(H, MS, stable):
    """Coarse-grid h_max per row: upper end of the stable run containing argmin MSSRE."""
    G = H.shape[1]
    out = np.full(len(H), np.nan)
    cens = np.full(len(H), "", dtype=object)
    for i in range(len(H)):
        st = stable[i]
        if not st.any():
            out[i], cens[i] = H[i, 0], "below"
            continue
        j = int(np.argmin(np.where(st, MS[i], np.inf)))
        u = j
        while u + 1 < G and st[u + 1]:
            u += 1
        out[i] = H[i, u]
        cens[i] = "above" if u == G - 1 else ""
    return out, cens


def threshold_block(pf, ok, tau_main):
    """Sensitivity of h_max to the criterion thresholds, recomputed from the coarse-grid
    profiles only (resolution: one grid ratio, 1.188): floor theta in {3, 10, 30, 100},
    bounded c in {0.1, 0.3, 1, 3} (MSSRE <= c ||E(0)||).  Ratios are relative to the
    coarse-grid value at the default threshold, so the grid resolution cancels."""
    if pf.empty:
        return {}
    pc = pf[(pf.kind == "main") & np.isclose(pf.tau, tau_main) & (pf.stage == "coarse")]
    if pc.empty:
        return {}
    key = ["task_id", "activation", "seed", "b", "r"]
    pc = pc.sort_values(key + ["h"])
    G = pc.groupby(key).size()
    if G.nunique() != 1:
        return {}
    g = int(G.iloc[0])
    idx = pc.groupby(key, sort=True).head(1)[key].reset_index(drop=True)
    H = pc.h.to_numpy(float).reshape(-1, g)
    MS = pc.mssre.to_numpy(float).reshape(-1, g)
    E0 = pc.e0.to_numpy(float).reshape(-1, g)[:, 0]
    fl = ok[(ok.kind == "main") & (ok.criterion == "floor")][key + ["floor_F", "theta", "F_abs", "formula"]]
    idx = idx.merge(fl.drop_duplicates(key), on=key, how="left")
    F = np.maximum(idx.floor_F.to_numpy(float), idx.F_abs.to_numpy(float))
    fin = np.isfinite(MS)
    variants = {"floor": [3.0, 10.0, 30.0, 100.0], "bounded": [0.1, 0.3, 1.0, 3.0]}
    default = {"floor": 10.0, "bounded": 1.0}
    res = {}
    acts = idx.activation.to_numpy(object)
    for crit, vals in variants.items():
        hm = {}
        for v in vals:
            thr = (v * F)[:, None] if crit == "floor" else (v * E0)[:, None]
            hm[v] = _coarse_hmax(H, MS, fin & (MS <= thr))
        h0, c0 = hm[default[crit]]
        for v in vals:
            if v == default[crit]:
                continue
            hv, cv = hm[v]
            rv, rk = censored_ratio(hv, cv, h0, c0)
            for a in sorted(set(acts)):
                mk = acts == a
                m, flag, _ = summarise_bounded(rv[mk], rk[mk])
                res[f"{crit}|{v:g}|{a}"] = {"median_ratio_to_default": m, "flag": flag,
                                            "frac_changed_gt_1_gridstep": float(np.mean(
                                                np.abs(np.log(rv[mk][rk[mk] == "exact"])) > 0.5 * np.log(1.188)))
                                            if (rk[mk] == "exact").any() else np.nan}
    return res


def censoring_block(ok, df, n_tasks, n_missing):
    out = {"n_tasks": n_tasks, "n_tasks_missing": n_missing,
           "n_error_groups": int(len(df[df.status != "ok"].drop_duplicates(["task_id", "activation", "seed"])))
           if "status" in df else 0}
    for crit, g in ok[ok.kind == "main"].groupby("criterion"):
        out[crit] = {"n_trajectories": int(len(g)), "frac_below_grid": float(np.mean(g.censored == "below")),
                     "frac_above_3": float(np.mean(g.censored == "above")),
                     "frac_with_islands": float(np.mean(g.n_islands > 0))}
    isl = (ok[ok.kind == "main"].assign(isl=lambda d: d.n_islands > 0)
           .groupby(["formula", "criterion"]).isl.mean().unstack("criterion"))
    out["islands_by_formula"] = {f: {c: float(v) for c, v in row.items() if np.isfinite(v)}
                                 for f, row in isl.iterrows()}
    out["non_interval_formulas"] = sorted(f for f, row in isl.iterrows()
                                          if np.nanmax(row.to_numpy(float)) > 0.05)
    return out


def fig_units(J, forms, acts):
    """Unit dependence: h_max(E in units of 1/s) / h_max(original unit) at the same
    physical ||E(0)|| = 1, per activation (x) and formula (markers)."""
    if J is None or len(J) == 0:
        return
    sub = J[(J.test == "same_r") & np.isclose(J.r, 1.0)]
    scales = sorted(sub.error_scale.unique())
    crits = [c for c in ("bounded", "nodiv") if c in set(sub.criterion)]
    if not scales or not crits:
        return
    fig, axes = plt.subplots(len(crits), len(scales), figsize=(style.DOUBLE, 1.9 * len(crits) + 0.5),
                             squeeze=False, sharex=True, sharey=True)
    fl = [f for f in keys.GROUPS["disc"] if f in set(sub.formula)] + \
        sorted(set(sub.formula) - set(keys.GROUPS["disc"]))
    handles = {}
    for i, crit in enumerate(crits):
        for j, sv in enumerate(scales):
            ax = axes[i, j]
            ax.axhline(1.0, color=style.GREY, lw=0.8, ls="--", zorder=1)
            for k, f in enumerate(fl):
                g = sub[(sub.criterion == crit) & np.isclose(sub.error_scale, sv) & (sub.formula == f)]
                if g.empty:
                    continue
                st = keys.style_in_group("disc", f)
                xs, ys, hollow = [], [], []
                for ia, a in enumerate(acts):
                    ga = g[g.activation == a]
                    if ga.empty:
                        continue
                    m, flag, _ = summarise_bounded(ga.factor, ga.fkind)
                    if not np.isfinite(m):
                        continue
                    xs.append(ia + (k - (len(fl) - 1) / 2) * 0.09)
                    ys.append(np.clip(m, 1e-4, 1e4))
                    hollow.append(flag != "")
                xs, ys, hollow = np.array(xs), np.array(ys), np.array(hollow, dtype=bool)
                if len(xs) == 0:
                    continue
                hnd = ax.plot(xs[~hollow], ys[~hollow], ls="none", marker=st["marker"], ms=3.6,
                              color=st["color"])[0] if (~hollow).any() else None
                if hollow.any():
                    h2 = ax.plot(xs[hollow], ys[hollow], ls="none", marker=st["marker"], ms=3.6,
                                 color=st["color"], mfc="white")[0]
                    hnd = hnd or h2
                handles[f] = hnd
            ax.set_yscale("log")
            ax.set_xticks(range(len(acts)))
            ax.set_xticklabels([ACT_LABEL.get(a, a) for a in acts], rotation=35, ha="right", fontsize=6)
            ax.set_title(f"error rescaled $E_s={sv:g}\\,E$, criterion: {crit}", fontsize=7)
            if j == 0:
                ax.set_ylabel("$h_{\\max}(s)/h_{\\max}(1)$ (log)")
    fig.legend(list(handles.values()), list(handles.keys()), loc="lower center", ncol=8,
               bbox_to_anchor=(0.5, -0.07), fontsize=6.2)
    fig.suptitle("Unit dependence at the same physical $\\|E(0)\\|=1$ (linear system, $\\tau=10^{-2}$): "
                 "$E_s=sE$; 1 = unit-invariant (linear); open markers = censored bounds", fontsize=7.3, y=1.02)
    style.savefig(fig, "fig_E2_units")


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        if math.isnan(f):
            return None
        if math.isinf(f):
            return "inf" if f > 0 else "-inf"
        return f
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


# ==========================================================================
# figures
# ==========================================================================
def fig_heatmap(main, crit, problems, forms, acts, r_main, TH):
    sub = main[(main.criterion == crit) & np.isclose(main.r, r_main)]
    if sub.empty:
        return
    nP = len(problems)
    h = max(2.2, 0.16 * len(forms) + 1.0)
    fig, axes = plt.subplots(1, nP, figsize=(style.DOUBLE, h), squeeze=False, sharey=True)
    norm = LogNorm(vmin=1e-4, vmax=3.0)
    im = None
    for j, prob in enumerate(problems):
        ax = axes[0, j]
        M = np.full((len(forms), len(acts)), np.nan)
        C = np.full((len(forms), len(acts)), "", dtype=object)
        s = sub[sub.problem == prob].set_index(["formula", "activation"])
        for i, f in enumerate(forms):
            for k, a in enumerate(acts):
                if (f, a) in s.index:
                    row = s.loc[(f, a)]
                    M[i, k] = row["median"]
                    C[i, k] = row["censored"]
        Mplot = np.clip(np.where(np.isfinite(M), M, np.nan), 1e-4, 3.0)
        im = ax.imshow(Mplot, cmap=style.CMAP_SEQ, norm=norm, aspect="auto", interpolation="nearest")
        for i in range(len(forms)):
            for k in range(len(acts)):
                v, c = M[i, k], C[i, k]
                if not np.isfinite(v) and c == "":
                    txt = "n/a"
                elif c == "below":
                    txt = "<1e-4"
                elif c == "above":
                    txt = ">3"
                else:
                    txt = f"{v:.2g}" if v < 0.1 else f"{v:.2f}"
                dark = np.isfinite(v) and norm(np.clip(v, 1e-4, 3)) > 0.6
                ax.text(k, i, txt, ha="center", va="center", fontsize=4.6,
                        color="white" if dark else style.INK)
        ax.set_xticks(range(len(acts)))
        ax.set_xticklabels([ACT_LABEL.get(a, a) for a in acts], rotation=40, ha="right", fontsize=6)
        ax.set_yticks(range(len(forms)))
        ax.set_yticklabels([tex_safe(f) for f in forms], fontsize=6)
        ax.set_title(PROB_LABEL.get(prob, prob), fontsize=7.5)
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=axes[0, :].tolist(), shrink=0.85, pad=0.015, aspect=30)
    cb.set_label("$h_{\\max}$ (median over trajectories)", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    fig.suptitle(f"Max. stable step $h=\\gamma\\tau$, $\\tau=10^{{-2}}$, $\\|E(0)\\|={r_main:g}$, "
                 f"criterion: {CRIT_LABEL[crit]}", fontsize=7.5, y=1.05)
    style.savefig(fig, f"fig_E2_heatmap_{crit}")


def tex_safe(s):
    return str(s)


def fig_theory_scatter(lin_cells, tau_main):
    """Measured largest stable step against the root-locus prediction, linear activation (Supplementary
    Material figure, drawn at the paper's text width)."""
    allc = lin_cells[np.isclose(lin_cells.tau, tau_main) & np.isfinite(lin_cells.h_theory)]
    sub = allc[allc.kind.isin(["main", "control"]) & (allc.theory_type == "exact")]
    loc = allc[allc.kind.isin(["main", "control"]) & (allc.theory_type == "local")]
    hz = allc[(allc.kind == "horizon") & (allc.theory_type == "exact")]
    hzl = allc[(allc.kind == "horizon") & (allc.theory_type == "local")]
    if sub.empty:
        return
    probs = [p for p in ("linear_system", "matrix_inversion", "linear_system_constA") if p in set(sub.problem)]
    series = [("bounded", "h_meas", "measured, bounded criterion", 0),
              ("floor", "h_meas", "measured, floor criterion (protocol)", 1),
              ("bounded", "h_frozen", "frozen-time theory ($h+\\tau\\mu$)", 2),
              ("floor", "h_fh", "finite-horizon theory (floor)", 3)]
    lo = max(min(np.nanmin(sub.h_theory), np.nanmin(sub.h_meas)) * 0.6, 1e-3)
    hi = 3.6
    handles = {}
    with style.print_rc():
        fig, axes = plt.subplots(1, len(probs), figsize=(style.TEXTWIDTH, 2.95), squeeze=False,
                                 constrained_layout=True)
        fig.get_layout_engine().set(w_pad=0.03, h_pad=0.03, wspace=0.05)
        for j, prob in enumerate(probs):
            ax = axes[0, j]
            ax.plot([lo, hi], [lo, hi], color=style.GREY, lw=0.8, ls="--", zorder=1)
            s_ = sub[sub.problem == prob]
            for crit, col, lab, slot in series:
                ss = s_[s_.criterion == crit].sort_values("h_theory")
                if col == "h_frozen" and prob.endswith("constA"):
                    continue
                y = ss[col].to_numpy(float)
                if not np.isfinite(y).any():
                    continue
                st = {"color": style.CAT[slot], "marker": style.MARKERS[slot]}
                if col == "h_meas":
                    yerr = np.vstack([y - ss.q25.to_numpy(float), ss.q75.to_numpy(float) - y])
                    hnd = ax.errorbar(ss.h_theory, y, yerr=np.clip(yerr, 0, None), ls="none", ms=3.8,
                                      mfc=st["color"] if slot == 0 else "white", mec=st["color"],
                                      ecolor=st["color"], elinewidth=0.7, capsize=1.2, marker=st["marker"],
                                      zorder=3)
                else:
                    hnd = ax.plot(ss.h_theory, y, ls="none", marker=st["marker"], ms=4.4, color=st["color"],
                                  mfc="none", mew=0.9, zorder=4)[0]
                handles[lab] = hnd
            extra = [(hz, "measured, bounded, $T=24$", 4, "h_meas", False),
                     (loc, "CDTZNN, bounded, $T=6$ (local theory)", 5, "h_meas", True),
                     (hzl, "CDTZNN, bounded, $T=24$", 6, "h_meas", False)]
            for frame, lab, slot, col, ann in extra:
                ss = frame[(frame.problem == prob) & (frame.criterion == "bounded")]
                if ss.empty:
                    continue
                hnd = ax.plot(ss.h_theory, ss[col], ls="none", marker=style.MARKERS[slot], ms=4.6,
                              color=style.CAT[slot], mfc="white" if slot == 4 else style.CAT[slot],
                              mew=0.9, zorder=5)[0]
                handles[lab] = hnd
                if ann:
                    for _, r in ss.iterrows():
                        ax.annotate("CDTZNN", (r.h_theory, r[col]), fontsize=7.5, color=style.INK2,
                                    xytext=(-5, 4), textcoords="offset points", ha="right", va="bottom")
            sb = s_[(s_.criterion == "bounded")]
            for _, r in sb.iterrows():            # formulas whose measured (bounded) limit deviates > 15 %
                if np.isfinite(r.ratio) and abs(r.ratio - 1) > 0.15:
                    ax.annotate(str(r.formula), (r.h_theory, r.h_meas), fontsize=7.5,
                                color=style.INK2, xytext=(4, -3), textcoords="offset points", ha="left",
                                va="top")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_xlabel("root-locus $h_{\\max}$")
            if j == 0:
                ax.set_ylabel("measured $h_{\\max}$ (median, IQR)")
            ax.set_title(PROB_LABEL.get(prob, prob).replace(" (control)", ""), fontsize=8)
            ax.set_aspect("equal")
        fig.legend(list(handles.values()), list(handles.keys()), loc="outside lower center", ncol=2,
                   frameon=False, columnspacing=1.6, handletextpad=0.4)
        style.savefig(fig, "fig_E2_theory_vs_measured")


def fig_rdep(main, prob, crit, forms, acts, TH):
    """Small multiples, one panel per formula (8 activation series per panel), in
    pages of 8 panels so no formula is dropped (suffix _p<k> only when > 1 page)."""
    sub = main[(main.problem == prob) & (main.criterion == crit)]
    group = [f for f in keys.GROUPS["disc"] if f in forms]
    extra = [f for f in forms if f not in group]
    allp = group + extra
    if not allp or sub.empty:
        return
    pages = [allp[i:i + 8] for i in range(0, len(allp), 8)]
    for ip, panels in enumerate(pages):
        ncol = 4
        nrow = int(math.ceil(len(panels) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(style.DOUBLE, 1.75 * nrow + 0.45), squeeze=False,
                                 sharex=True, sharey=True)
        handles = {}
        for i, f in enumerate(panels):
            ax = axes[i // ncol, i % ncol]
            s = sub[sub.formula == f]
            for a in acts:
                sa = s[s.activation == a].sort_values("r")
                if sa.empty:
                    continue
                y = np.clip(sa["median"].to_numpy(float), 1e-4, 3.0)
                st = keys.style_in_group("act", a)
                hnd = ax.plot(sa.r, y, color=st["color"], marker=st["marker"], ls=st["linestyle"], ms=3.2,
                              lw=1.0)[0]
                for cflag, mk in (("below", "v"), ("above", "^")):
                    cm = (sa.censored == cflag).to_numpy()
                    if cm.any():
                        ax.plot(sa.r.to_numpy()[cm], y[cm], ls="none", marker=mk, ms=3.4,
                                color=st["color"], mfc="white")
                handles[ACT_LABEL.get(a, a)] = hnd
            hth = TH.loc[f, "h_theory"] if f in TH.index else np.nan
            if np.isfinite(hth):
                ax.axhline(hth, color=style.GREY, lw=0.7, ls=":", zorder=1)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_ylim(8e-5, 4)
            ax.set_title(f, fontsize=7)
            if i % ncol == 0:
                ax.set_ylabel("$h_{\\max}$ (log)")
            if i // ncol == nrow - 1 or i + ncol >= len(panels):
                ax.set_xlabel("$\\|E(0)\\|$ (log)")
                ax.tick_params(labelbottom=True)
        for k in range(len(panels), nrow * ncol):
            axes[k // ncol, k % ncol].set_visible(False)
        fig.legend(list(handles.values()), list(handles.keys()), loc="lower center", ncol=8,
                   bbox_to_anchor=(0.5, -0.045 if nrow > 1 else -0.09), fontsize=6.2)
        pg = f" (page {ip + 1}/{len(pages)})" if len(pages) > 1 else ""
        fig.suptitle(f"{PROB_LABEL.get(prob, prob)}: $h_{{\\max}}$ vs initial error{pg}, criterion: "
                     f"{CRIT_LABEL[crit]}\n(dotted: linear root-locus theory; open down/up triangles: "
                     f"censored below $10^{{-4}}$ / above 3)", fontsize=7.3, y=1.03)
        style.savefig(fig, f"fig_E2_rdep_{crit}_{prob}" + (f"_p{ip + 1}" if len(pages) > 1 else ""))


def fig_tau(sens, shift_cells, forms):
    """|h_max/h_theory - 1| vs tau on both problems and the frozen-A control (finite-
    horizon part only), plus the paired time-variation shift |h(LS)/h(A frozen) - 1|."""
    cols = [(p, "dev") for p in ("linear_system", "matrix_inversion", "linear_system_constA")
            if p in set(sens.problem)]
    if len(shift_cells):
        cols.append(("linear_system", "shift"))
    show = [f for f in keys.GROUPS["disc"] if f in set(sens.formula)]   # <= 8, fixed slots
    fig, axes = plt.subplots(2, len(cols), figsize=(style.DOUBLE, 3.9), squeeze=False, sharex=True)
    handles = {}
    for i, crit in enumerate(("bounded", "floor")):
        for j, (prob, what) in enumerate(cols):
            ax = axes[i, j]
            allt = []
            for f in show:
                if what == "dev":
                    sf = sens[(sens.problem == prob) & (sens.criterion == crit) & (sens.formula == f)]
                    sf = sf.sort_values("tau")
                    dev = np.abs(sf.ratio.to_numpy(float) - 1)
                else:
                    sf = shift_cells[(shift_cells.criterion == crit) & (shift_cells.formula == f)]
                    sf = sf.sort_values("tau")
                    dev = np.abs(sf["median"].to_numpy(float))
                if sf.tau.nunique() < 2:
                    continue
                dev = np.where(dev > 0, dev, np.nan)
                st = keys.style_in_group("disc", f)
                hnd = ax.plot(sf.tau, dev, color=st["color"], marker=st["marker"], ls=st["linestyle"],
                              ms=3.0, lw=1.0)[0]
                handles[f] = hnd
                allt += list(sf.tau)
            if allt:
                tt = np.array(sorted(set(allt)))
                ref = 0.3 * tt / tt.max()
                hr = ax.plot(tt, ref, color=style.GREY, lw=0.8, ls="--")[0]
                handles["$O(\\tau)$ reference"] = hr
            ax.set_xscale("log")
            ax.set_yscale("log")
            if allt:
                ax.set_xticks(tt)
                ax.set_xticklabels([f"{v:g}" for v in tt], fontsize=6)
                ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            short = {"linear_system": "lin. system", "matrix_inversion": "matrix inv.",
                     "linear_system_constA": "lin. system, $A$ frozen"}[prob]
            ttl = (f"{short}: $|h_{{\\max}}/h^{{\\rm th}}-1|$" if what == "dev"
                   else "time-var. shift $|h_{\\rm LS}/h_{A_0}-1|$")
            ax.set_title(f"{ttl}\n{crit}", fontsize=6.6)
            if j == 0:
                ax.set_ylabel("relative deviation (log)")
            if i == 1:
                ax.set_xlabel("$\\tau$ (log; $T=6$)")
    fig.legend(list(handles.values()), list(handles.keys()), loc="lower center", ncol=8,
               bbox_to_anchor=(0.5, -0.06), fontsize=6.2)
    fig.suptitle("Linear activation, $\\|E(0)\\|=1$: deviation from the root locus vs sampling period "
                 "(control = finite-horizon part; shift = time-variation part)", fontsize=7.5, y=1.02)
    style.savefig(fig, "fig_E2_tau_convergence")


def fig_profiles(pf, cfg, TH, forms, problems):
    tau_main, r_main = float(cfg["tau"]), float(cfg["r_main"])
    p = pf[(pf.kind == "main") & np.isclose(pf.tau, tau_main) & (pf.activation == "linear")
           & np.isclose(pf.r, r_main) & (pf.stage == "coarse")]
    if p.empty:
        return
    group = [f for f in keys.GROUPS["disc"] if f in forms]
    if not group:
        group = forms[:8]
    fig, axes = plt.subplots(1, len(problems), figsize=(style.DOUBLE, 2.4), squeeze=False, sharey=True)
    handles = {}
    for j, prob in enumerate(problems):
        ax = axes[0, j]
        s = p[p.problem == prob]
        for f in group:
            sf = s[s.formula == f]
            if sf.empty:
                continue
            # median over ALL trajectories (diverged = inf), so a step at which more than
            # half of the trajectories diverge shows as a gap
            m = sf.groupby("h").mssre.median()
            m = m.where(np.isfinite(m), np.nan)
            st = keys.style_in_group("disc", f)
            hnd = ax.plot(m.index, m.values, color=st["color"], ls=st["linestyle"], marker=st["marker"],
                          ms=2.2, lw=0.9, markevery=4)[0]
            handles[f] = hnd
            hth = TH.loc[f, "h_theory"]
            if np.isfinite(hth):
                ax.axvline(hth, color=st["color"], lw=0.6, ls=":", alpha=0.9)
        ax.axhline(r_main, color=style.GREY, lw=0.7, ls="--")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("$h=\\gamma\\tau$")
        if j == 0:
            ax.set_ylabel("MSSRE (median over trajectories)")
        ax.set_title(PROB_LABEL.get(prob, prob), fontsize=7.5)
        fin = s.mssre.to_numpy(float)
        fin = fin[np.isfinite(fin) & (fin > 0)]
        ax.set_ylim(max(fin.min() / 5, 1e-14) if len(fin) else 1e-12, 50)
    axes[0, 0].legend(list(handles.values()), list(handles.keys()), loc="lower left", ncol=1,
                      fontsize=6.0, framealpha=0.9)
    fig.suptitle("Linear activation, $\\tau=10^{-2}$, $\\|E(0)\\|=1$: steady residual vs step "
                 "(dotted: root-locus $h_{\\max}$; dashed: $\\|E(0)\\|$; gaps = divergence)",
                 fontsize=7.3, y=1.02)
    style.savefig(fig, "fig_E2_profiles_linear")


# ==========================================================================
# tables
# ==========================================================================
def table_theory(TH, lin_cells, order, tau_main, tabdir):
    lc = lin_cells[np.isclose(lin_cells.tau, tau_main) & lin_cells.kind.isin(MAP_KINDS)]
    probs = [p for p in ("linear_system", "matrix_inversion", "linear_system_constA") if p in set(lc.problem)]
    header = ["formula", "$c_{-1}$", "$h_{\\max}$", "$h_{\\max}/c_{-1}$", "stated", "$h_{\\rm opt}$",
              "$\\rho_{\\rm opt}$"]
    short = {"linear_system": "LS", "matrix_inversion": "MI", "linear_system_constA": "LS-$A_0$"}
    for p in probs:
        header += [f"{short[p]} bnd", f"{short[p]} floor"]
    header += ["$h_{\\rm ft}$ (LS)"]
    rows = []
    for f in order:
        if f not in TH.index:
            continue
        t = TH.loc[f]
        fname = tex_escape(f) + ("$^{\\dagger}$" if t.theory_type == "local" else
                                 ("$^{\\ddagger}$" if t.kind is None else ""))
        r = [fname, f"{t.c_lead:.4f}" if t.kind == "lookahead" else "--",
             fmt_h(t.h_theory), fmt_h(t.h_theory_c0norm) if t.kind == "lookahead" else "--",
             fmt_h(t.h_stated) if np.isfinite(t.h_stated) else "--",
             fmt_h(t.h_opt), f"{t.rho_opt:.4f}" if np.isfinite(t.rho_opt) else "--"]
        for p in probs:
            for crit in ("bounded", "floor"):
                s = lc[(lc.formula == f) & (lc.problem == p) & (lc.criterion == crit)]
                if s.empty:
                    r.append("--")
                else:
                    cens = "above" if s.n_above.iloc[0] >= s.n.iloc[0] / 2 else ""
                    r.append(fmt_h(float(s.h_meas.iloc[0]), cens))
        s = lc[(lc.formula == f) & (lc.problem == "linear_system") & (lc.criterion == "bounded")]
        r.append(fmt_h(float(s.h_frozen.iloc[0])) if len(s) else "--")
        rows.append(r)
    write_tex(tabdir / "E2_theory.tex", header, rows, "l" + "r" * (len(header) - 1),
              "E2: root-locus stability limit of the linear-ZNN error recursion "
              "$c_{-1}\\xi^s+(c_0+h)\\xi^{s-1}+c_1\\xi^{s-2}+\\dots$ ($h=\\gamma\\tau$, tau-form "
              "coefficients; $h/c_{-1}$ = leading-coefficient-normalised convention) versus the measured "
              "maximal stable $h$ (median over 3 instances $\\times$ $B$ directions, $\\tau=10^{-2}$, $T=6$, "
              "$\\|E(0)\\|=1$). stated: interval end stated in the literature (converted to $h=\\gamma\\tau$). "
              "bnd: bounded criterion; floor: protocol criterion (MSSRE $\\le 10\\times$ "
              "formula floor); $h_{\\rm ft}$: frozen-time prediction with $hI+\\tau A^{-1}\\dot A$. "
              "$^{\\dagger}$: inverse-free scheme, LOCAL theory (linearisation about $(x^*,J^{-1})$; both blocks have "
              "the $P_h$ of the x-recursion; $h_{\\rm ft}$ includes the M-block shift $\\tau(\\mu_i+\\mu_j)$). "
              "$^{\\ddagger}$: no closed-form theory (Broyden: $M_k\\not\\to J^{-1}$; implicit: unconditional).",
              "tab:E2-theory")


def table_map(main, prob, forms, acts, r_main, tabdir):
    sub = main[(main.problem == prob) & np.isclose(main.r, r_main)]
    header = ["formula"] + [ACT_LABEL.get(a, a) for a in acts]
    rows = []
    for f in forms:
        r = [tex_escape(f)]
        for a in acts:
            vals = []
            for crit in ("floor", "bounded"):
                s = sub[(sub.formula == f) & (sub.activation == a) & (sub.criterion == crit)]
                vals.append(fmt_h(float(s["median"].iloc[0]), s.censored.iloc[0]) if len(s) else "--")
            r.append(" / ".join(vals))
        rows.append(r)
    write_tex(tabdir / f"E2_map_{prob}.tex", header, rows, "l" + "c" * len(acts),
              f"E2: maximal stable $h=\\gamma\\tau$ on {tex_escape(prob)} ($n=4$, $\\tau=10^{{-2}}$, $T=6$, "
              f"$\\|E(0)\\|={r_main:g}$), median over trajectories; entries are floor (protocol) / bounded "
              "criterion.", f"tab:E2-map-{prob.replace('_', '-')}")


def _tex_flag(v, flag, fmt="{:.3g}"):
    if flag == "n/a" or v is None or not np.isfinite(v):
        return "n/a"
    return {"<=": "$\\le$", ">=": "$\\ge$"}.get(flag, "") + fmt.format(v)


def table_rdep(summary, acts, cfg, tabdir):
    rd = summary.get("nonlinear_r_dependence", {})
    rs = [f"{r:g}" for r in cfg["r_list"]]
    header = ["activation", "criterion"] + [f"$\\|E(0)\\|={r}$" for r in rs] + \
        ["shrink $r_{\\max}/r_{\\min}$ [95\\% CI]", "vs linear [95\\% CI]"]
    rows = []
    for crit in ("bounded", "floor", "nodiv"):
        for a in acts:
            e = rd.get(f"{crit}|{a}")
            if not e:
                continue
            fl = e.get("censor_flag_by_r", {})
            cells = []
            for x in rs:
                v = e["median_h_max_by_r"].get(x, np.nan)
                cells.append("$<10^{-4}$" if fl.get(x) == "<" else ("$\\ge 3$" if fl.get(x) == ">=" else fmt_h(v)))
            r = [ACT_LABEL.get(a, a), crit] + cells
            ci, ci2 = e["shrink_ci95"], e["rel_to_linear_ci95"]
            sh = _tex_flag(e["shrink_rmax_over_rmin_median"], e["shrink_flag"])
            r.append(sh if sh == "n/a" else f"{sh} [{ci[0]:.3g}, {ci[1]:.3g}]")
            rl = _tex_flag(e["rel_to_linear_at_rmain_median"], e["rel_to_linear_flag"])
            r.append(rl if rl == "n/a" else f"{rl} [{ci2[0]:.3g}, {ci2[1]:.3g}]")
            rows.append(r)
        rows.append("\\midrule")
    if rows and rows[-1] == "\\midrule":
        rows.pop()
    write_tex(tabdir / "E2_rdep.tex", header, rows, "ll" + "r" * (len(header) - 2),
              "E2: dependence of the maximal stable step on the initial error magnitude for the nonlinear "
              "activations (median over formulas and both problems of the per-cell medians; "
              "shrink = $h_{\\max}(\\|E(0)\\|=10)/h_{\\max}(\\|E(0)\\|=0.1)$ per trajectory, median per "
              "formula, bootstrap 95\\% CI over formulas $\\times$ problems; vs linear = $h_{\\max}$ relative to "
              "linear activation at $\\|E(0)\\|=1$). Censored values are bounds: $\\le$/$\\ge$ mark medians "
              "of which at least half are upper/lower bounds, n/a = both ends censored (e.g. no stable step at "
              "either magnitude); formulas censored identically at both ends (implicit Euler) are excluded. "
              "nodiv = no-divergence diagnostic.", "tab:E2-rdep")


def table_tau(tau_res, shift_res, forms, tabdir):
    header = ["formula", "series", "criterion", "signed dev. by $\\tau$ (desc.)",
              "consecutive slopes", "slope (asympt.)", "slope (all)"]
    short = {"linear_system": "LS vs theory", "matrix_inversion": "MI vs theory",
             "linear_system_constA": "LS-$A_0$ vs theory (finite horizon)"}
    items = [(k.split("|")[0], short.get(k.split("|")[1], tex_escape(k.split("|")[1])), k.split("|")[2], v)
             for k, v in tau_res.items()]
    items += [(k.split("|")[0], "LS / LS-$A_0$ (time variation)", k.split("|")[1], v) for k, v in shift_res.items()]
    items.sort(key=lambda it: (forms.index(it[0]) if it[0] in forms else 99, it[2], it[1]))
    rows = []
    for f, ser, c, v in items:
        vals = ", ".join("--" if not np.isfinite(x) else f"{x:+.2g}" for x in v.get("signed", v["values"]))
        cons = ", ".join("--" if not np.isfinite(x) else f"{x:.2f}" for x in v["consecutive"])
        rows.append([tex_escape(f), ser, c, vals, cons,
                     "--" if not np.isfinite(v["slope_asymptotic"]) else f"{v['slope_asymptotic']:.2f}",
                     "--" if not np.isfinite(v["slope_all"]) else f"{v['slope_all']:.2f}"])
    write_tex(tabdir / "E2_tau.tex", header, rows, "lllllrr",
              "E2: convergence of the measured stability limit (linear activation, $T=6$) to the root-locus "
              "value as $\\tau\\to0$ at fixed $h$. Because $T$ is fixed, $N=T/\\tau$ grows and two $O(\\tau)$ "
              "effects overlap: the finite-horizon detection limit (isolated by the frozen-$A$ control, "
              "LS-$A_0$) and the time-variation shift $O(\\tau\\|A^{-1}\\dot A\\|)$ (isolated by the paired "
              "ratio LS/LS-$A_0$). Slopes fitted to the absolute deviation; points at the bisection resolution "
              "excluded; the largest $\\tau$ dropped as pre-asymptotic when the first two consecutive slopes "
              "differ by more than 0.5.", "tab:E2-tau")


# ==========================================================================
# SUMMARY.md
# ==========================================================================
def _f(v, fmt="{:.3f}"):
    return "--" if v is None or (isinstance(v, float) and not np.isfinite(v)) else fmt.format(v)


def write_summary_md(results, s, cfg, TH, order, problems, acts):
    L = [f"# E2 stability maps -- summary ({s.get('scale')} scale)", ""]
    L.append(f"Tasks: {s['n_tasks']} ({s['n_tasks_missing']} missing); error groups: {s['n_error_groups']}. "
             f"tau = {cfg['tau']:g}, T = {cfg['T']:g}, h searched in (0, 3] (coarse log grid of "
             f"{len(cfg['h_grid'])} points from {cfg['h_grid'][0]:g}, then "
             f"{cfg['refine']['rounds']} rounds of {cfg['refine']['K']}-section); seeds {cfg['seeds']}, "
             f"B = {cfg['B']} directions, ||E(0)|| in {cfg['r_list']}. Criteria: bounded (MSSRE <= ||E(0)||), "
             f"floor (protocol: MSSRE <= 10 F, F = linear-activation MSSRE at the formula's h_opt), nodiv "
             f"(diagnostic: no divergence over T).")
    L.append(f"Bisection resolution (median relative half-width of uncensored brackets): "
             f"{s.get('bisection_relative_resolution_median', float('nan')):.2e}. Theory vs package "
             f"`znn_stability` max |diff| = {s.get('theory_vs_package_max_abs_diff')}; direct np.roots check "
             f"(max|root| < 1 just below h_max, >= 1 just above): {s.get('np_roots_boundary_ok')}.")
    cz = s.get("censoring", {})
    if cz:
        parts = []
        for crit in ("bounded", "floor", "nodiv", "e4"):
            e = cz.get(crit)
            if e:
                parts.append(f"{crit}: {e['frac_below_grid']:.1%} below the grid (no stable step >= 1e-4), "
                             f"{e['frac_above_3']:.1%} censored above 3, {e['frac_with_islands']:.2%} with islands")
        L.append(f"Censoring (main block, {cz.get('bounded', {}).get('n_trajectories', '?')} "
                 f"trajectory rows per criterion): " + "; ".join(parts) +
                 f". Missing tasks {cz.get('n_tasks_missing')}, error groups {cz.get('n_error_groups')}. Censored "
                 "values enter medians as bounds (below: 5e-5 plotting value, above: 3) and ratios as one-sided "
                 "bounds; nothing is dropped. Non-interval stable sets (> 5 % of trajectories with stable grid "
                 "points above the first unstable one): " + (", ".join(cz.get("non_interval_formulas", [])) or "none") + ".")
    pr_ = s.get("pseudo_replication", {})
    if pr_:
        L.append(f"Pseudo-replication: relative range of h_max over the B directions of one instance "
                 f"{pr_['within_seed_relative_range_median']:.2%} (median) vs over instances "
                 f"{pr_['between_seed_relative_range_median']:.2%}: the instance (seed) is the unit; seed ranges "
                 "below are the [min, max] of the per-seed median ratio.")
    L.append("")
    L.append("## Linear activation: measured vs root-locus theory (tau = 1e-2, ||E(0)|| = 1)")
    L.append("")
    L.append("Exact-inverse formulas only (inverse-free schemes: next section). CIs: bootstrap over formulas "
             "(descriptive spread of the formula-level ratios, not a sampling interval); seed range = [min, max] "
             "over instances of the median ratio (the cluster-level uncertainty).")
    L.append("")
    L.append("| criterion / problem | n formulas | median h_meas/h_theory [95% CI] | seed range | max dev. (formula) | "
             "Kendall tau_b (theory vs meas.) | vs frozen-time [seed range] | vs finite-horizon |")
    L.append("|---|---|---|---|---|---|---|---|")
    for k, e in s.get("linear_theory_vs_measured", {}).items():
        ft = (f"{e['median_ratio_meas_over_frozen_time']:.3f} [{_f(e['seed_range_frozen_time'][0])}, "
              f"{_f(e['seed_range_frozen_time'][1])}]") if "median_ratio_meas_over_frozen_time" in e else "--"
        fh = (f"{e['median_ratio_meas_over_finite_horizon']:.3f} [{e['ci95_finite_horizon'][0]:.3f}, "
              f"{e['ci95_finite_horizon'][1]:.3f}]") if "median_ratio_meas_over_finite_horizon" in e else "--"
        L.append(f"| {k} | {e['n_formulas']} | {e['median_ratio_meas_over_theory']:.3f} [{e['ci95'][0]:.3f}, "
                 f"{e['ci95'][1]:.3f}] | [{_f(e['seed_range'][0])}, {_f(e['seed_range'][1])}] | "
                 f"{e['max_abs_dev']:.3f} ({e['worst_formula']}) | "
                 f"{_f(e['kendall_tau_b_theory_vs_measured'], '{:.3f}')} | {ft} | {fh} |")
    L.append("")
    tc = s.get("tau_convergence", {})
    sh = s.get("time_variation_shift", {})
    if tc:
        L.append("## Convergence to theory as tau -> 0 (linear, bounded criterion, T = 6 fixed)")
        L.append("")
        L.append("At fixed T, N = T/tau grows, so the finite-horizon detection limit (O(1/N) = O(tau), "
                 "positive) and the time-variation shift (O(tau ||A^-1 A_dot||), negative) overlap. "
                 "LS-A0 = frozen-A control (finite-horizon part only); LS/LS-A0 = paired time-variation part.")
        L.append("")
        L.append("| formula | series | signed dev. (tau desc.) | consecutive slopes | slope asympt. (n pts, flags) "
                 "| slope all |")
        L.append("|---|---|---|---|---|---|")
        rows = [(k.split("|")[0], k.split("|")[1], v) for k, v in tc.items() if k.endswith("bounded")]
        rows += [(k.split("|")[0], "LS/LS-A0 shift", v) for k, v in sh.items() if k.endswith("bounded")]
        rows.sort(key=lambda r: (order.index(r[0]) if r[0] in order else 99, r[1]))
        for f, ser, v in rows:
            L.append(f"| {f} | {ser} | " + ", ".join(_f(x, "{:+.2g}") for x in v.get("signed", v["values"]))
                     + " | " + ", ".join(_f(x, "{:.2f}") for x in v["consecutive"]) +
                     f" | {_f(v['slope_asymptotic'], '{:.2f}')} ({v.get('n_asymptotic')}"
                     + (", 2-point fit" if v.get("two_point") else "")
                     + (", ABSURD" if v.get("absurd") else "") + ")"
                     + f" | {_f(v['slope_all'], '{:.2f}')} |")
        L.append("")
        ov = s.get("tau_slope_overview", {})
        if ov:
            L.append("Asymptotic slopes, median [IQR] over formulas: " + "; ".join(
                f"{k}: {_f(v['median'], '{:.2f}')} [{_f(v['q25'], '{:.2f}')}, {_f(v['q75'], '{:.2f}')}] "
                f"(n = {v['n']})" for k, v in ov.items()))
            L.append("")
    ftc = s.get("floor_transient_check", {})
    bad = [k for k, v in ftc.items() if v["contaminated"]]
    if ftc:
        L.append("Floor transient check (rho_opt^k_w * ||E(0)|| > 0.1 F at tau = 1e-2, i.e. F is partly a "
                 "transient residue): " + (", ".join(bad) if bad else "none") + ".")
        L.append("")
    L.append("## Nonlinear activations: dependence on ||E(0)||")
    L.append("")
    L.append("Censored values are bounds: '<1e-4' = more than half of the cells have no stable grid step; "
             "'<=x'/'>=x' = at least half of the per-cell ratios are upper/lower bounds; n/a = both ends "
             "censored. implicit_euler_err ('>3' everywhere) is excluded from the ratios.")
    L.append("")
    L.append("| criterion / activation | median h_max at ||E(0)|| = " + ", ".join(f"{r:g}" for r in cfg["r_list"])
             + " | shrink r_max/r_min [95% CI] | h_max / h_max(linear) at 1 [95% CI] | cells below grid |")
    L.append("|---|---|---|---|---|")
    for k, e in s.get("nonlinear_r_dependence", {}).items():
        fl = e.get("censor_flag_by_r", {})
        mb = ", ".join(("<1e-4" if fl.get(r) == "<" else (">=3" if fl.get(r) == ">=" else f"{v:.3g}"))
                       for r, v in e["median_h_max_by_r"].items())
        shs = fmt_flag(e["shrink_rmax_over_rmin_median"], e["shrink_flag"])
        shs = shs if shs == "n/a" else f"{shs} [{e['shrink_ci95'][0]:.3g}, {e['shrink_ci95'][1]:.3g}]"
        rls = fmt_flag(e["rel_to_linear_at_rmain_median"], e["rel_to_linear_flag"])
        rls = rls if rls == "n/a" else f"{rls} [{e['rel_to_linear_ci95'][0]:.3g}, {e['rel_to_linear_ci95'][1]:.3g}]"
        L.append(f"| {k} | {mb} | {shs} | {rls} | {e['frac_cells_below_grid_at_rmain']:.2f} |")
    L.append("")
    ch = s.get("nodiv_over_bounded", {})
    if ch:
        rs = [f"{r:g}" for r in cfg["r_list"]]
        L.append("Blow-up vs bounded chattering: median h_max(nodiv) / h_max(bounded) over formulas x problems "
                 "(1 = the bounded criterion fails where the trajectory diverges; > 1 = it fails because the "
                 "steady residual chatters above ||E(0)|| while staying finite).")
        L.append("")
        L.append("| activation | " + " | ".join(f"||E(0)|| = {r}" for r in rs) + " |")
        L.append("|---|" + "---|" * len(rs))
        for a in acts:
            cells = []
            for r in rs:
                e = ch.get(f"{a}|{r}")
                cells.append("--" if not e else fmt_flag(e["median_nodiv_over_bounded"], e["flag"], "{:.2f}"))
            L.append(f"| {a} | " + " | ".join(cells) + " |")
        L.append("")
    lp = s.get("linearised_prediction_small_error", {})
    if lp:
        L.append("Linearised prediction h_theory / Phi'(0) at the smallest ||E(0)|| (bounded criterion): " +
                 "; ".join(f"{a}: Phi'(0) = {e['slope0']:.3f}, measured/predicted = "
                           f"{e['median_ratio_meas_over_hlin_div_slope0']:.3f} [{e['ci95'][0]:.3f}, "
                           f"{e['ci95'][1]:.3f}]" for a, e in lp.items()))
        L.append("")
    L.append(f"## Ranking agreement of formulas (Kendall tau_b, ||E(0)|| = 1; p Holm-adjusted over "
             f"{s.get('kendall_family_size', 0)} tests; DESCRIPTIVE: n = 7-10 formulas per test, no inferential "
             f"claim is made)")
    L.append("")
    kt = s.get("kendall_tau_b", {})
    pr = [p for p in ("linear_system", "matrix_inversion") if p in problems]

    def _k(key):
        v = kt.get(key)
        if not v or v["tau_b"] is None or (isinstance(v["tau_b"], float) and not np.isfinite(v["tau_b"])):
            return "--"
        ph = v.get("p_holm")
        return f"{v['tau_b']:.2f} (n={v['n']}, p_Holm={_f(ph, '{:.2g}')})"
    hdr = ["activation"]
    for crit in ("bounded", "floor"):
        hdr += [f"{crit}: theory vs {p}" for p in pr] + ([f"{crit}: LS vs MI"] if len(pr) == 2 else [])
    L.append("| " + " | ".join(hdr) + " |")
    L.append("|" + "---|" * len(hdr))
    for a in acts:
        row = [a]
        for crit in ("bounded", "floor"):
            row += [_k(f"{crit}|{a}|theory_vs_{p}") for p in pr]
            if len(pr) == 2:
                row.append(_k(f"{crit}|{a}|problems"))
        L.append("| " + " | ".join(row) + " |")
    L.append("")
    L.append("(theory = linear root-locus h_max of the formula, exact-inverse formulas with a finite value only; for nonlinear "
             "activations this tests whether the linear theory still ORDERS the formulas. LS vs MI uses every "
             "formula, including implicit and inverse-free ones.)")
    L.append("")
    L += review_sections(s, cfg, acts)
    c = s.get("cost", {})
    if c:
        L.append("## Cost")
        L.append("")
        L.append(f"simulate wall total {c['sim_wall_total_s']:.1f} s over {c['n_groups']} groups "
                 f"({c['sims_per_group_median']:.0f} simulate calls per group); median wall per step of the "
                 f"coarse batch ({c['rows_coarse_median']:.0f} rows) {c['wall_per_step_coarse_batch_median_s']:.2e} s; "
                 f"task wall total {c['task_wall_total_s']:.1f} s (max {c['task_wall_max_s']:.1f} s); "
                 f"peak RSS {c['peak_rss_mb_max']:.0f} MB.")
        L.append("")
    L.append("## Interpretation")
    L.append("")
    L += interpretation(s)
    L += interpretation_review(s, cfg)
    (results / "SUMMARY.md").write_text("\n".join(L) + "\n")


def review_sections(s, cfg, acts):
    L = []
    inv = s.get("inverse_free_local_theory", {})
    th = s.get("theory", {})
    noth = {f: v.get("reason") for f, v in th.items() if v.get("theory_type") in (None, "None")}
    if inv or noth:
        L.append("## Inverse-free schemes")
        L.append("")
        L.append("CDTZNN: local theory = root locus of its x-recursion (both linearised blocks share P_h on the "
                 "affine E2 problems); frozen-time prediction includes the M-block shift tau(mu_i + mu_j). "
                 "Linear activation, ||E(0)|| = 1, tau = 1e-2.")
        L.append("")
        L.append("| formula / kind / problem / criterion | T | h_meas | meas / local theory | frozen-time (M-block) | "
                 "meas / frozen-time | meas / exact-inverse twin |")
        L.append("|---|---|---|---|---|---|---|")
        for k, e in inv.items():
            if k.split("|")[3] not in ("bounded", "floor"):
                continue
            L.append(f"| {k} | {e['T']:g} | {e['h_meas_median']:.4f} | {e['ratio_local_theory']:.3f} | "
                     f"{e['h_frozen_time_with_M_block_median']:.4f} | {e['ratio_frozen_time_with_M_block']:.3f} | "
                     f"{_f(e.get('ratio_to_exact_inverse_twin'))} |")
        L.append("")
        for f, rsn in noth.items():
            L.append(f"* no theory for {f}: {rsn}.")
        L.append("")
    hz = s.get("horizon", {})
    if hz:
        L.append("## Horizon sensitivity (linear activation, ||E(0)|| = 1, tau = 1e-2, T = 6 vs T = 24)")
        L.append("")
        L.append("| problem / criterion / theory type | n | median h(T=24)/h(T=6) | min (formula) | meas/theory T=6 | "
                 "meas/theory T=24 | meas/frozen-time T=24 | max dev T=6 -> T=24 |")
        L.append("|---|---|---|---|---|---|---|---|")
        for k, e in hz.items():
            if k.split("|")[1] not in ("bounded", "floor"):
                continue
            L.append(f"| {k} | {e['n_formulas']} | {e['median_ratio_long_over_short']:.3f} | "
                     f"{e['min_ratio_long_over_short']:.3f} ({e['worst_formula']}) | "
                     f"{e['median_meas_over_theory_short']:.3f} | {e['median_meas_over_theory_long']:.3f} | "
                     f"{e['median_meas_over_frozen_long']:.3f} | {e['max_abs_dev_theory_short']:.3f} -> "
                     f"{e['max_abs_dev_theory_long']:.3f} |")
        L.append("")
    ud = s.get("unit_dependence", {})
    if ud:
        rs = [f"{r:g}" for r in cfg["r_list"]]
        L.append("## Unit dependence (paired scaled-error arm, REVIEW_CHECKLIST item 1)")
        L.append("")
        L.append("E expressed in units of 1/s (E_s = s E; same x0, residuals and criteria in the original unit). "
                 "Factor = h_max(s) / h_max(1) at the same physical ||E(0)|| (median over formulas x problems of "
                 "per-cell medians; '<='/'>=' censored bounds). 'arg-equiv' = h_max(r, s) / h_max(s r, 1): 1 means "
                 "a unit change acts exactly like a change of the initial error (it also rescales the floor, so "
                 "deviations come from the steady-state forcing).")
        L.append("")
        L.append("| criterion | activation | s | factor at ||E(0)|| = " + " / ".join(rs) +
                 " | cells changed > 25 % | arg-equiv median |")
        L.append("|---|---|---|---|---|---|")
        for crit in ("bounded", "nodiv", "floor"):
            for a in acts:
                for sv in cfg.get("error_scales", []):
                    e = ud.get(f"same_r|{sv:g}|{crit}|{a}")
                    if not e:
                        continue
                    ae = ud.get(f"arg_equiv|{sv:g}|{crit}|{a}", {})
                    br = " / ".join(_f(e["by_r"].get(r), "{:.3g}") for r in rs)
                    L.append(f"| {crit} | {a} | {sv:g} | {br} | {_f(e['frac_cells_changed_gt_25pct'], '{:.0%}')} | "
                             f"{fmt_flag(ae.get('median_factor', np.nan), ae.get('flag', 'n/a'))} |")
        L.append("")
    e4 = s.get("e4_vs_bounded", {})
    if e4:
        rs = [f"{r:g}" for r in cfg["r_list"]]
        L.append("## E4-compatible criterion (E4 P2: MSSRE <= 0.1 ||E(0)||, no growth over the last T/8, "
                 "peak <= 10 ||E(0)||)")
        L.append("")
        L.append("E4 does NOT reuse E2: it recomputes h_max with this criterion (explicit Euler, tau = 1e-3, "
                 "worst case over all calibration trajectories, largest stable point of a descending grid). "
                 "Median h_max(E4 criterion) / h_max(bounded), per trajectory, main block (tau = 1e-2):")
        L.append("")
        L.append("| activation | " + " | ".join(f"||E(0)|| = {r}" for r in rs) + " |")
        L.append("|---|" + "---|" * len(rs))
        for a in acts:
            cells = []
            for r in rs:
                e = e4.get(f"{a}|{r}")
                cells.append("--" if not e else fmt_flag(e["median_e4_over_bounded"], e["flag"], "{:.3f}"))
            L.append(f"| {a} | " + " | ".join(cells) + " |")
        L.append("")
    ts = s.get("threshold_sensitivity", {})
    if ts:
        L.append("## Threshold sensitivity (coarse-grid recomputation, ratio to the default threshold)")
        L.append("")
        L.append("| activation | floor theta = 3 / 30 / 100 (default 10) | bounded c = 0.1 / 0.3 / 3 (default 1) |")
        L.append("|---|---|---|")
        for a in acts:
            fl = " / ".join(fmt_flag(ts.get(f"floor|{v:g}|{a}", {}).get("median_ratio_to_default", np.nan),
                                     ts.get(f"floor|{v:g}|{a}", {}).get("flag", "n/a")) for v in (3, 30, 100))
            bd = " / ".join(fmt_flag(ts.get(f"bounded|{v:g}|{a}", {}).get("median_ratio_to_default", np.nan),
                                     ts.get(f"bounded|{v:g}|{a}", {}).get("flag", "n/a")) for v in (0.1, 0.3, 3))
            L.append(f"| {a} | {fl} | {bd} |")
        L.append("")
    return L


def interpretation(s):
    out = []
    h = s.get("linear_theory_vs_measured", {})
    b = h.get("bounded|linear_system")
    c = h.get("bounded|linear_system_constA")
    fl = h.get("floor|linear_system")
    if b:
        out.append(f"* Linear activation, bounded criterion: the measured limit is "
                   f"{b['median_ratio_meas_over_theory']:.3f}x the root-locus value (median over "
                   f"{b['n_formulas']} formulas, bootstrap 95% CI {b['ci95'][0]:.3f}-{b['ci95'][1]:.3f}); the "
                   f"formula ranking is preserved (Kendall tau_b = {b['kendall_tau_b_theory_vs_measured']:.2f}). "
                   f"The largest deviation is {b['worst_formula']} ({b['max_abs_dev']:.0%}).")
    if b and "max_abs_dev_frozen_time" in b:
        out.append(f"* The frozen-time prediction (effective step h + tau*mu, mu = eig(A^-1 A_dot)) explains most of "
                   f"the deviation: measured/predicted = {b['median_ratio_meas_over_frozen_time']:.3f} "
                   f"(max |dev| {b['max_abs_dev_frozen_time']:.1%} vs {b['max_abs_dev']:.1%} for the root locus).")
    ov = s.get("tau_slope_overview", {})
    tv, fh = ov.get("time_variation_shift_bounded", {}), ov.get("finite_horizon_control_bounded", {})
    if tv.get("n") or fh.get("n"):
        out.append(f"* As tau -> 0 at fixed T both parts vanish at first order: time-variation shift "
                   f"(paired LS / frozen-A) slope {_f(tv.get('median'), '{:.2f}')} [IQR "
                   f"{_f(tv.get('q25'), '{:.2f}')}, {_f(tv.get('q75'), '{:.2f}')}], finite-horizon part "
                   f"(frozen-A vs theory) slope {_f(fh.get('median'), '{:.2f}')} [IQR {_f(fh.get('q25'), '{:.2f}')}, "
                   f"{_f(fh.get('q75'), '{:.2f}')}]. The raw |h/h_th - 1| slopes mix the two (opposite signs), "
                   f"which is why single formulas can show non-power-law behaviour.")
    if c:
        out.append(f"* Control with A frozen (no tau*A^-1*A_dot coupling): ratio "
                   f"{c['median_ratio_meas_over_theory']:.3f}, max dev. {c['max_abs_dev']:.1%} "
                   f"({c['worst_formula']}). The residual deviation is the finite-horizon detection limit: a "
                   f"weakly unstable root (rho slightly > 1) does not grow above ||E(0)|| within N = T/tau steps, "
                   f"so the bounded criterion can overshoot the boundary slightly where rho(P_h) crosses 1 "
                   f"with a small slope.")
    if fl:
        out.append(f"* Protocol (floor) criterion: {fl['median_ratio_meas_over_theory']:.3f}x the root locus; "
                   f"the finite-horizon prediction rho(P_h)^k_w = 10F/||E(0)|| accounts for most of the gap "
                   f"(measured/predicted {fl.get('median_ratio_meas_over_finite_horizon', float('nan')):.3f}): "
                   f"near h_max the slowest root needs more than the 4.5 s before the steady window to decay "
                   f"to the floor.")
    rd = s.get("nonlinear_r_dependence", {})
    sh = [(k.split("|")[1], e["shrink_rmax_over_rmin_median"], e["shrink_flag"]) for k, e in rd.items()
          if k.startswith("bounded|")]
    if sh:
        shrink = [(a, v, f) for a, v, f in sh if f != "n/a" and np.isfinite(v) and v < 0.8]
        grow = [(a, v, f) for a, v, f in sh if f != "n/a" and np.isfinite(v) and v > 1.25]
        out.append("* Nonlinear activations (bounded criterion), h_max(||E(0)||=10)/h_max(||E(0)||=0.1): "
                   "fast-growing tails shrink the stable step at large errors (" +
                   ", ".join(f"{a} x{fmt_flag(v, f, '{:.2g}')}" for a, v, f in sorted(shrink, key=lambda t: t[1])) +
                   "; '<=' = upper bound, the large-error end has no stable grid step); activations with a sign "
                   "or sig^r term show the opposite trend (" +
                   ", ".join(f"{a} x{fmt_flag(v, f, '{:.2g}')}" for a, v, f in sorted(grow, key=lambda t: -t[1])) +
                   "): their chattering amplitude (~h for sign terms, ~h^(1/(1-r)) for sig^r) does not scale "
                   "with the error, so it exceeds a SMALL initial error first.")
    ch = s.get("nodiv_over_bounded", {})
    if ch:
        big = sorted(((k, e["median_nodiv_over_bounded"]) for k, e in ch.items()
                      if e["flag"] == "" and np.isfinite(e["median_nodiv_over_bounded"])
                      and e["median_nodiv_over_bounded"] > 1.5), key=lambda kv: -kv[1])
        if big:
            out.append("* That small-error limit is a residual-size limit, not a blow-up: the no-divergence step "
                       "exceeds the bounded-criterion step by " +
                       ", ".join(f"x{v:.2g} ({k.replace('|', ' at ||E(0)|| = ')})" for k, v in big[:6]) +
                       ". The PLAN hypothesis (power-type tails shrink the stable region at large errors) is "
                       "confirmed for the exponential/power tails; for sign-type activations the bounded and "
                       "divergence limits differ and must be reported separately.")
    fl2 = [(k.split("|")[1], e["rel_to_linear_at_rmain_median"], e["rel_to_linear_flag"],
            e["frac_cells_below_grid_at_rmain"]) for k, e in rd.items() if k.startswith("floor|")]
    if fl2:
        out.append("* Under the protocol (floor) criterion the activations with a sign term (VAF, PVAF-1, "
                   "pred.-exp) and the non-Lipschitz ones (sign-bi-power, fixed-time) cannot reach the "
                   "formula floor: an explicit step is locally stable at e only while h*Phi(e)/e < h_max(formula), "
                   "which fixes a chattering level far above the O(tau^{p+1}) floor, and it is worst for the "
                   "high-order formulas whose h_max is small. h_max relative to linear activation at "
                   "||E(0)|| = 1 (fraction of cells with no stable grid step): " +
                   ", ".join(f"{a} {fmt_flag(v, f, '{:.2g}')} ({fb:.0%})" for a, v, f, fb in fl2) + ".")
    lp = s.get("linearised_prediction_small_error", {})
    if lp:
        out.append("* Lipschitz activations at small errors follow the linearised theory h_max(linear)/Phi'(0): " +
                   ", ".join(f"{a} {e['median_ratio_meas_over_hlin_div_slope0']:.2f}" for a, e in lp.items()) +
                   " (measured/predicted).")
    bad = [k for k, v in s.get("floor_transient_check", {}).items() if v["contaminated"]]
    if bad:
        out.append("* Caveat: for " + ", ".join(bad) + " the formula floor F at h_opt still contains transient "
                   "(rho_opt^k_w ||E(0)|| > 0.1 F), so their floor criterion is looser than for the other formulas.")
    return out


def interpretation_review(s, cfg):
    """Bullets added by the 2026-10-18 review (criterion artefacts, units, horizon, E4)."""
    out = []
    rd = s.get("nonlinear_r_dependence", {})
    nd = [(k.split("|")[1], e) for k, e in rd.items() if k.startswith("nodiv|")]
    if nd:
        out.append("* r-dependence with an r-INDEPENDENT threshold (nodiv = blow-up boundary), "
                   "h_max(||E(0)||=10)/h_max(0.1): " +
                   ", ".join(f"{a} x{fmt_flag(e['shrink_rmax_over_rmin_median'], e['shrink_flag'], '{:.2g}')}"
                             for a, e in nd) +
                   ". The bounded criterion's threshold is ||E(0)|| itself, so its small-error 'shrinkage' of "
                   "sign-type activations is a residual-size statement (chattering amplitude ~ h vs the "
                   "threshold r), not a loss of stability; headline claims about stability vs ||e(0)|| should "
                   "use the nodiv column (and say that its detection limit is horizon dependent).")
    ud = s.get("unit_dependence", {})
    if ud:
        big = []
        for k, e in ud.items():
            t, sv, crit, a = k.split("|")
            if t == "same_r" and crit == "bounded" and a != "linear" and np.isfinite(e.get("median_abs_log10_factor", np.nan)):
                big.append((a, sv, e["median_factor"], e["flag"], e["frac_cells_changed_gt_25pct"]))
        lin_e = [e for k, e in ud.items() if k.startswith("same_r|") and k.endswith("|bounded|linear")]
        linf = ", ".join(_f(e["median_factor"], "{:.3f}") for e in lin_e)
        if big:
            out.append("* UNIT DEPENDENCE (paired scaled-error arm): expressing E in a unit 10x smaller or larger "
                       "changes the nonlinear-activation h_max at the same physical ||E(0)|| = 1 by a median factor " +
                       ", ".join(f"{a} (s={sv}) x{fmt_flag(v, f, '{:.2g}')}" for a, sv, v, f, _ in big) +
                       f"; linear activation is invariant (factors {linf}). Every nonlinear-activation entry of the "
                       "maps (and the r-dependence) therefore holds only for E in the problem's own unit: the "
                       "activations have no length scale tied to the problem (their knee is at |e| = 1).")
    inv = s.get("inverse_free_local_theory", {})
    c6 = inv.get("cdtznn_mahuang2025|main|linear_system|bounded")
    c24 = inv.get("cdtznn_mahuang2025|horizon|linear_system|bounded")
    cc = inv.get("cdtznn_mahuang2025|control|linear_system_constA|bounded")
    if c6 and c24:
        out.append(f"* CDTZNN (inverse-free): at T = 6 the bounded limit is {c6['h_meas_median']:.3f} "
                   f"({c6['ratio_local_theory']:.2f}x its local root locus 0.1725), but at T = 24 it is "
                   f"{c24['h_meas_median']:.3f} (measured / frozen-time prediction with the M-block shift "
                   f"tau(mu_i+mu_j) = {c24['ratio_frozen_time_with_M_block']:.2f}): the M-recursion is weakly "
                   "unstable (~0.4 %/step) above the M-block limit and starts small, so a 6-s horizon cannot see it."
                   + (f" With A frozen (no time variation) it is {cc['ratio_local_theory']:.2f}x the local theory."
                      if cc else ""))
    hz = s.get("horizon", {})
    he = hz.get("linear_system|bounded|exact")
    if he:
        out.append(f"* Horizon: for the exact-inverse formulas T = 24 changes the bounded h_max by a median factor "
                   f"{he['median_ratio_long_over_short']:.3f} (min {he['min_ratio_long_over_short']:.3f}, "
                   f"{he['worst_formula']}); the T = 6 maps are horizon-robust for them, not for CDTZNN.")
    e4 = s.get("e4_vs_bounded", {})
    if e4:
        sm = [(k, e) for k, e in e4.items() if k.endswith("|1") and k.split("|")[0] != "linear"]
        out.append("* E4's P2 stability definition differs from both E2 criteria; at ||E(0)|| = 1 its h_max relative "
                   "to the bounded criterion is " + ", ".join(
                       f"{k.split('|')[0]} {fmt_flag(e['median_e4_over_bounded'], e['flag'], '{:.2f}')}" for k, e in sm) +
                   ". P2's 0.8 h_max therefore inherits E4's own threshold choice (0.1 ||E(0)||) and, for "
                   "power/sign-type models, E4's initial-error magnitude and unit (E2 r-dependence and unit arms).")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--figdir", default=str(style.FIG_DIR))
    ap.add_argument("--tabdir", default=str(style.TAB_DIR))
    a = ap.parse_args(argv)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        s = analyze(Path(a.results), Path(a.figdir), Path(a.tabdir))
    print(json.dumps(_clean({k: s[k] for k in ("linear_theory_vs_measured", "cost") if k in s}), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
