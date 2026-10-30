#!/usr/bin/env python
"""Analysis of E8: implicit (consistent) vs explicit discretization of sign-type ZNN.

    python experiments/analyze_E8.py --results results/E8 \
        --figdir figures --tabdir results/tables
    python experiments/analyze_E8.py --results results/mini/E8 \
        --figdir figures/mini --tabdir results/tables/mini

Reads the runner results of the configs built by experiments/configs/E8.py
(``<results>/E8_ti``, ``<results>/E8_tv``, optional ``<results>/E8_formulas``)
and writes

* figures ``fig_E8_*.pdf/png`` (experiments/style.py rules),
* LaTeX booktabs tables ``E8_*.tex``,
* ``<results>/summary.json`` and ``<results>/SUMMARY.md`` (headline numbers,
  bootstrap CIs, Kendall tau_b, empirical orders, cost extrapolation).

Statistics: every cell has 3 problem seeds x B = 4 initial states = 12
trajectories; we report the median and IQR over them.  Headline comparisons
are paired (same instance and initial state) log10 ratios, summarised by the
median with a percentile bootstrap 95% CI (2000 resamples, seeded).
Empirical orders are fitted only in the asymptotic range (consecutive-ratio
diagnostic, points at or below the round-off floor excluded); both the
all-points and the asymptotic slope are reported.

Chattering metrics are only meaningful above the round-off floor: when the
steady residual is <= ROUNDOFF (exact convergence), ``chatter_vec`` is NaN
(zero residual) or dominated by rounding noise, and ``rate_alt`` counts sign
changes of rounding-level increments; such cells are reported as 0 (no motion)
and flagged (``*_eff`` columns, see E8_NOTES.md).
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator  # noqa: E402

import keys  # noqa: E402  (experiments/keys.py)
import style  # noqa: E402  (experiments/style.py)
from znnbench.runner import aggregate  # noqa: E402

# ---------------------------------------------------------------------------
# protocol constants (from the config module, so both stay in sync)
# ---------------------------------------------------------------------------
_spec = importlib.util.spec_from_file_location("_e8_config", HERE / "configs" / "E8.py")
E8 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E8)

DISCS = E8.DISCS
DISCS_EULER = E8.DISCS_EULER
DISCS_TZ = E8.DISCS_TZ
DISCS_MAIN = [d for d in DISCS if d != "implicit_taylor_zhang_naive"]   # the 8 group members
EXPLICIT = set(E8.EXPLICIT)
NAIVE = "implicit_taylor_zhang_naive"
H_GRID = E8.H_GRID
H_REF = 0.3
EPS_EXACT = E8.EPS_EXACT
ROUNDOFF = 1e-12            # residuals at or below this are treated as round-off / exact.  The
#                             measured MSSRE floor of the time-varying n=4 problems is ~2e-14 (EM +
#                             uhlig_4_6 at tau <= 1e-3), so points above 1e-12 carry <= ~2% round-off.
CONTRACT = 0.1              # "stable": no divergence and MSSRE <= CONTRACT * initial residual
N_BOOT = 2000
BOOT_SEED = 0
BIG = 1e300                 # stand-in for inf inside quantile computations
NO_EXACT_MAP = {"fixed_time"}   # 3 power terms (k3 > 0): `exact` falls back to the PEB Lie split
TIE_REL = 0.10              # Kendall tau_b "tie-tolerant" variant: medians within 10 % are tied
GAMMA_DIAG = (10.0, 100.0)  # fixed-gamma orders read off the (tau, h) grid along h = gamma * tau
FRAC = 0.25                 # steady window = last 25 % (config metrics.frac)
SCALE_CFGS = ("E8_scale_ti", "E8_scale_tv")          # error-unit arm (review 2026-10-18)
SCALE_ORDER = sorted(set(E8.ERROR_SCALES) | {1.0})   # s = 1 is the base arm (E8_ti / E8_tv)
SCALE_ACTS = ["sign", "sbp0.5", "predefined_exp"]   # figure panels of the scale arm
_SC_SHADE = [3, 6, 9, 12]


def sstyle(s):
    """plot kwargs of an error scale s: one-hue sequential ramp + marker + linestyle."""
    i = SCALE_ORDER.index(s) if s in SCALE_ORDER else 0
    return {"color": style.SEQ_BLUE[_SC_SHADE[i % 4]], "marker": style.MARKERS[i % 8],
            "linestyle": style.LINESTYLES[i % 8]}


def slabel(s):
    return "s = 1 (problem unit)" if s == 1 else f"s = {s:g}"

ACT_ORDER = ["sign", "sbp0.25", "sbp0.5", "sbp0.75", "fixed_time", "predefined_exp"]
ACT_LABEL = {"sign": "sign", "sbp0.25": "SBP $r=0.25$", "sbp0.5": "SBP $r=0.5$",
             "sbp0.75": "SBP $r=0.75$", "fixed_time": "fixed-time", "predefined_exp": "predefined-time"}
ACT_TEX = {"sign": "sign", "sbp0.25": "SBP $r{=}0.25$", "sbp0.5": "SBP $r{=}0.5$",
           "sbp0.75": "SBP $r{=}0.75$", "fixed_time": "fixed-time", "predefined_exp": "predefined-time"}
DISC_LABEL = {
    "euler": "Euler (explicit)",
    "implicit_euler_err": "Euler + resolvent",
    "peb_euler": "Euler + PEB",
    "exact_flow_euler": "Euler + exact flow",
    "semi_implicit_euler": "Euler + semi-implicit",
    "taylor_zhang": "TZ (explicit)",
    "implicit_taylor_zhang": "TZ-EM + resolvent",
    "peb_taylor_zhang": "TZ-EM + PEB",
    "implicit_taylor_zhang_naive": "TZ naive + resolvent (ablation)",
}
PROB_LABEL = {"linear_system": "linear system", "matrix_inversion": "matrix inversion",
              "linear_system_ti": "time-invariant linear system"}

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------------
# styling helpers
# ---------------------------------------------------------------------------
def dstyle(disc):
    """plot kwargs of a discretization: fixed slot in group 'disc_implicit'; the
    ablation (outside the group) is grey with its own marker."""
    if disc == NAIVE:
        return {"color": style.GREY, "marker": "x", "linestyle": ":"}
    return keys.style_in_group("disc_implicit", disc)


def fstyle(formula, others=()):
    """plot kwargs of a look-ahead formula in group 'disc'; formulas outside the
    group are grey and distinguished by marker."""
    if formula in keys.GROUPS["disc"]:
        return keys.style_in_group("disc", formula)
    i = list(others).index(formula) if formula in others else 0
    return {"color": style.GREY, "marker": ["x", "+", "1", "2"][i % 4], "linestyle": ":"}


def save(fig, figdir, name):
    style.FIG_DIR = Path(figdir)
    return style.savefig(fig, name)


def zero_formatter(clip):
    def fmt(v, _pos):
        if v <= clip * 1.01:
            return "0"
        e = int(round(math.log10(v)))
        return f"$10^{{{e}}}$" if abs(v / 10 ** e - 1) < 1e-6 else ""
    return FuncFormatter(fmt)


# ---------------------------------------------------------------------------
# statistics helpers
# ---------------------------------------------------------------------------
def _fin(x):
    x = np.asarray(x, dtype=float)
    return np.where(np.isposinf(x), BIG, np.where(np.isneginf(x), -BIG, x))


def _unfin(v):
    if isinstance(v, np.ndarray):
        return np.where(v >= BIG / 10, np.inf, np.where(v <= -BIG / 10, -np.inf, v))
    return math.inf if v >= BIG / 10 else (-math.inf if v <= -BIG / 10 else float(v))


def med_iqr(x):
    """median, q25, q75 of x (inf allowed, NaN dropped)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return (math.nan, math.nan, math.nan)
    q = np.quantile(_fin(x), [0.5, 0.25, 0.75], method="linear")
    return tuple(_unfin(float(v)) for v in q)


def boot_ci(x, n=N_BOOT, seed=BOOT_SEED, stat=np.median):
    """Percentile bootstrap 95% CI of `stat` (inf-safe)."""
    x = _fin(np.asarray(x, dtype=float))
    x = x[~np.isnan(x)]
    if x.size == 0:
        return (math.nan, math.nan, math.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n, x.size))
    bs = stat(x[idx], axis=1)
    lo, hi = np.quantile(bs, [0.025, 0.975], method="linear")
    return (_unfin(float(stat(x))), _unfin(float(lo)), _unfin(float(hi)))


def boot_ci_cluster(x, clusters, n=N_BOOT, seed=BOOT_SEED, stat=np.median):
    """Two-stage (cluster) percentile bootstrap 95% CI of `stat` (inf-safe).

    The problem instance (seed) is the experimental unit: the B initial states of one instance give
    identical (consistent schemes) or nearly identical steady-state metrics, so resampling the B*seeds
    trajectories as exchangeable units overstates the precision.  Stage 1 resamples the instances with
    replacement, stage 2 the trajectories within each drawn instance; `stat` is taken over the pooled
    draw.  NaN (e.g. both members of a pair diverged) is dropped."""
    x = _fin(np.asarray(x, dtype=float))
    cl = np.asarray(clusters)
    ok = ~np.isnan(x)
    x, cl = x[ok], cl[ok]
    if x.size == 0:
        return (math.nan, math.nan, math.nan)
    groups = [x[cl == c] for c in np.unique(cl)]
    G = len(groups)
    rng = np.random.default_rng(seed)
    sizes = {len(g) for g in groups}
    if len(sizes) == 1:                         # vectorised: (n, G, b) draws
        b = sizes.pop()
        X = np.stack(groups)                    # (G, b)
        pick = rng.integers(0, G, size=(n, G))
        within = rng.integers(0, b, size=(n, G, b))
        bs = stat(X[pick[:, :, None], within].reshape(n, G * b), axis=1)
    else:
        bs = np.empty(n)
        for i in range(n):
            parts = [groups[j][rng.integers(0, len(groups[j]), size=len(groups[j]))]
                     for j in rng.integers(0, G, size=G)]
            bs[i] = stat(np.concatenate(parts))
    lo, hi = np.quantile(bs, [0.025, 0.975], method="linear")
    return (_unfin(float(stat(x))), _unfin(float(lo)), _unfin(float(hi)))


def cluster_medians(x, clusters):
    """Per-cluster (instance) medians of x, inf-safe, NaN dropped."""
    x = np.asarray(x, dtype=float)
    cl = np.asarray(clusters)
    out = []
    for c in np.unique(cl):
        v = x[(cl == c) & ~np.isnan(x)]
        if v.size:
            out.append(_unfin(float(np.median(_fin(v)))))
    return np.asarray(out, dtype=float)


def fit_order(taus, vals, floor=ROUNDOFF, tol=0.15):
    """Empirical order p in MSSRE ~ tau^p.

    * points with non-finite values or values <= floor (round-off) are dropped;
    * local (consecutive-ratio) slopes p_i = log(v_i/v_{i+1}) / log(tau_i/tau_{i+1});
    * the asymptotic range is the longest tail (smallest taus) whose local slopes
      all lie within a band of width `tol` (max - min <= tol, 0.15 = the tolerance of
      tests/test_problems_robot.py; review 2026-10-18, was 0.3; a drifting sequence
      such as 3.0, 2.85, 2.7 is therefore NOT accepted as asymptotic); the slope
      is the LS fit over that tail.  With only one local slope in the tail the
      asymptotic range is not established (n_asym = 2) and that is flagged.
    Returns a dict with slope_all/n_all (LS over all valid points), slope_asym/
    n_asym, local slopes and a flag `floor_hit` (some point removed by the floor)."""
    t = np.asarray(taus, dtype=float)
    v = np.asarray(vals, dtype=float)
    o = np.argsort(-t)
    t, v = t[o], v[o]
    ok = np.isfinite(v) & (v > floor)
    out = {"slope_all": math.nan, "n_all": int(ok.sum()), "slope_asym": math.nan, "n_asym": 0,
           "local": [], "floor_hit": bool((np.isfinite(v) & (v <= floor)).any()),
           "taus_asym": []}
    if ok.sum() < 2:
        return out
    tt, vv = t[ok], v[ok]
    out["slope_all"] = float(np.polyfit(np.log(tt), np.log(vv), 1)[0])
    loc = [float(np.log(vv[i] / vv[i + 1]) / np.log(tt[i] / tt[i + 1])) for i in range(len(tt) - 1)]
    out["local"] = loc
    j = len(loc) - 1                       # the tail always contains the last local slope
    while j > 0 and max(loc[j - 1:]) - min(loc[j - 1:]) <= tol:
        j -= 1
    ta, va = tt[j:], vv[j:]
    out["slope_asym"] = float(np.polyfit(np.log(ta), np.log(va), 1)[0])
    out["n_asym"] = int(len(ta))
    out["taus_asym"] = [float(x) for x in ta]
    return out


def tie_groups(v, rel=TIE_REL):
    """Rank values into tie groups: sorted values whose consecutive ratio is
    below 1 + rel share a group (inf values form one group on top).  Used so
    that schemes which are equal by construction (e.g. resolvent vs exact map of
    the sign activation) but differ at the 1e-6 relative level through the
    bisection tolerance do not create spurious strict orders."""
    v = _fin(np.asarray(v, dtype=float))
    out = np.full(v.shape, np.nan)
    ok = ~np.isnan(v)
    if not ok.any():
        return out
    idx = np.argsort(v[ok], kind="stable")
    vals = v[ok][idx]
    g, grp = 0, np.empty(len(vals))
    for i, x in enumerate(vals):
        if i > 0:
            a, b = vals[i - 1], x
            same = (a == b) or (a >= BIG / 10 and b >= BIG / 10) or (a > 0 and b < BIG / 10 and b / a < 1 + rel)
            if not same:
                g += 1
        grp[i] = g
    tmp = np.empty(len(vals))
    tmp[idx] = grp
    out[ok] = tmp
    return out


def kendall_b(a, b, tie_rel=None):
    """Kendall tau_b (scipy, variant b) with its p-value.  ``tie_rel``: rank on
    tie groups (see tie_groups) instead of the raw values."""
    from scipy.stats import kendalltau
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() < 3:
        return (math.nan, math.nan)
    x, y = _fin(a[ok]), _fin(b[ok])
    if tie_rel:
        x, y = tie_groups(x, tie_rel), tie_groups(y, tie_rel)
    if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return (math.nan, math.nan)
    r = kendalltau(x, y, variant="b")
    return (float(r.statistic), float(r.pvalue))


def sign_test(lr):
    """One-sided sign test of H0: median log-ratio <= 0 (ties and NaN dropped).
    Returns (wins, n, p).  Called on the per-instance (cluster) medians, so n is the number
    of instances (3 at mini scale: smallest attainable p = 0.125)."""
    from scipy.stats import binomtest
    x = np.asarray(lr, dtype=float)
    x = x[~np.isnan(x) & (x != 0)]
    if x.size == 0:
        return (0, 0, math.nan)
    w = int((x > 0).sum())
    return (w, int(x.size), float(binomtest(w, int(x.size), 0.5, alternative="greater").pvalue))


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (NaN kept)."""
    p = np.asarray(pvals, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.where(~np.isnan(p))[0]
    if ok.size == 0:
        return out
    order = ok[np.argsort(p[ok])]
    m, run = len(order), 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (m - i) * p[j]))
        out[j] = run
    return out


def fmt_num(v, digits=2):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "--"
    if isinstance(v, float) and math.isinf(v):
        return r"$\infty$" if v > 0 else r"$-\infty$"
    if v == 0:
        return "0"
    if abs(v) >= 1e3 or abs(v) < 1e-2:
        m, e = f"{v:.{digits - 1}e}".split("e")
        return f"${m}{{\\times}}10^{{{int(e)}}}$"
    return f"{v:.{digits}g}" if abs(v) >= 1 else f"{v:.{digits}f}".rstrip("0").rstrip(".")


def tex_escape(s):
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


LONG_ROWS = 40              # tables with more rows are written as longtable (page breaks; needs \\usepackage{longtable})


def write_table(path, cols, rows, caption, label, colspec=None, notes=None, wide=False):
    """LaTeX booktabs table. `rows` items are lists of cell strings, or the
    string '\\midrule'.  ``wide``: scale the tabular to the text width.  More
    than LONG_ROWS rows: a \\footnotesize longtable (breaks across pages, no
    scaling), because a float taller than a page is cut off."""
    colspec = colspec or ("l" * 2 + "r" * (len(cols) - 2))
    if len(rows) > LONG_ROWS:
        head = " & ".join(cols) + r" \\"
        out = [r"\begingroup\footnotesize\setlength{\LTcapwidth}{\linewidth}\setlength{\tabcolsep}{4pt}",
               rf"\begin{{longtable}}{{{colspec}}}", rf"\caption{{{caption}}}\label{{{label}}}\\",
               r"\toprule", head, r"\midrule", r"\endfirsthead",
               rf"\multicolumn{{{len(cols)}}}{{l}}{{\emph{{(continued)}}}}\\", r"\toprule", head, r"\midrule",
               r"\endhead", r"\bottomrule", r"\endfoot"]
        for r in rows:
            out.append(r if isinstance(r, str) else " & ".join(r) + r" \\")
        out += [r"\end{longtable}"]
        if notes:
            out.append(rf"\par\smallskip {notes}")
        out.append(r"\endgroup")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("\n".join(out) + "\n")
        return path
    out = [r"\begin{table}[t]", r"\centering", r"\small", rf"\caption{{{caption}}}", rf"\label{{{label}}}"]
    if wide:
        out.append(r"\resizebox{\linewidth}{!}{%")
    out += [rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", " & ".join(cols) + r" \\", r"\midrule"]
    for r in rows:
        out.append(r if isinstance(r, str) else " & ".join(r) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}" + ("}" if wide else "")]
    if notes:
        out.append(rf"\par\smallskip\footnotesize {notes}")
    out.append(r"\end{table}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(out) + "\n")
    return path


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def add_columns(df):
    if df.empty:
        return df
    df = df.copy()
    act = df["model_kw_activation"].astype(str)
    if "model_kw_r" in df.columns:
        r = pd.to_numeric(df["model_kw_r"], errors="coerce")
        sbp = act == "sign_bi_power"
        act = act.where(~sbp, "sbp" + r.map(lambda v: f"{v:g}" if np.isfinite(v) else "?"))
    df["act"] = act
    df["h"] = df["h"].astype(float).round(12)
    df["tau"] = df["tau"].astype(float)
    df["diverged_b"] = df["status"].eq("diverged")
    df["ok_b"] = df["status"].isin(["ok", "diverged"])
    df["failed_b"] = ~df["ok_b"]

    def _scale_of(w):
        try:
            for x in (json.loads(w) if isinstance(w, str) and w else []) or []:
                if isinstance(x, dict) and str(x.get("key", "")).endswith("ScaledError"):
                    return float((x.get("kwargs") or {}).get("scale", 1.0))
        except (TypeError, ValueError):
            pass
        return 1.0
    df["scale"] = df["wrappers"].map(_scale_of).astype(float) if "wrappers" in df.columns else 1.0
    df["pair"] = df["seed"].astype(int).astype(str) + ":" + df["x0_index"].astype(int).astype(str)
    if "mean_ssre" in df.columns:
        conv = df["mean_ssre"].astype(float) <= ROUNDOFF
        df["converged_exact"] = conv & ~df["diverged_b"]
        for c in ("chatter_vec", "rate_alt", "chatter"):
            if c in df.columns:
                df[c + "_eff"] = np.where(df["converged_exact"], 0.0, df[c].astype(float))
    if "t_eps_1e-13" in df.columns:
        df["steps_exact"] = (df["t_eps_1e-13"].astype(float) / df["tau"]).round()
    if df["disc_key"].eq("consistent").any() or "disc_kw_formula" in df.columns:
        f = df.get("disc_kw_formula")
        df["formula"] = np.where(df["disc_key"].eq("consistent"), f if f is not None else "", df["disc_key"])
        df["variant"] = np.where(df["disc_key"].eq("consistent"), "em_exact", "explicit")
    return df


def load(results):
    out = {}
    for name in ("E8_ti", "E8_tv", "E8_formulas") + SCALE_CFGS:
        d = Path(results) / name
        if d.is_dir():
            df = aggregate(d)
            if not df.empty:
                bad = df[~df["status"].isin(["ok", "diverged"])]
                if len(bad):
                    print(f"[E8] {name}: {len(bad)} rows with status {sorted(bad.status.unique())}")
                out[name] = add_columns(df)
    return out


def acts_present(df):
    s = set(df["act"].unique())
    return [a for a in ACT_ORDER if a in s] + sorted(s - set(ACT_ORDER))


def cell(df, **sel):
    m = np.ones(len(df), dtype=bool)
    for k, v in sel.items():
        if isinstance(v, float):
            m &= np.isclose(df[k].astype(float), v, rtol=1e-9, atol=0)
        else:
            m &= df[k].eq(v).to_numpy()
    return df[m]


def tau_ref(df, problem, act=None):
    """Reference tau: 1e-3 (a 1 kHz control rate) when present for this problem
    (and activation), else the smallest tau available.  At full scale every
    (problem, activation) has tau = 1e-3; the mini run omits it for some."""
    m = df.problem_key.eq(problem)
    if act is not None:
        m &= df.act.eq(act)
    taus = sorted(df.loc[m, "tau"].unique())
    if not taus:
        return math.nan
    return 1e-3 if any(np.isclose(taus, 1e-3)) else min(taus)


def tau_desc(df, problem, acts):
    """Human-readable description of the reference tau of a figure."""
    trs = {a: tau_ref(df, problem, a) for a in acts}
    vals = sorted(set(v for v in trs.values() if np.isfinite(v)))
    if len(vals) <= 1:
        return f"$\\tau={vals[0]:g}$" if vals else ""
    main = tau_ref(df, problem)
    oth = ", ".join(f"{ACT_LABEL.get(a, a)}: {v:g}" for a, v in trs.items() if not np.isclose(v, main))
    return f"$\\tau={main:g}$ ({oth})"


# ---------------------------------------------------------------------------
# traces
# ---------------------------------------------------------------------------
class TraceStore:
    def __init__(self, results):
        self.results = Path(results)
        self._open = {}

    def get(self, config, traces_file, run_id):
        p = self.results / config / "results" / str(traces_file)
        if p not in self._open:
            self._open[p] = np.load(p)
        z = self._open[p]
        pre = f"{run_id}__"
        return {k[len(pre):]: z[k] for k in z.files if k.startswith(pre)}

    def stack(self, rows, key="residual"):
        """(t, R) with R of shape (P, n_traj) for the trajectories in `rows`."""
        t, cols = None, []
        for (config, tf, rid), g in rows.groupby(["config", "traces_file", "run_id"]):
            tr = self.get(config, tf, rid)
            if key not in tr:
                continue
            t = tr["t"]
            for i in g["traj"].astype(int):
                cols.append(np.asarray(tr[key][:, i], dtype=float))
        if not cols:
            return None, None
        return t, np.stack(cols, axis=1)

    def close(self):
        for z in self._open.values():
            z.close()


# ---------------------------------------------------------------------------
# (a) time-invariant problem
# ---------------------------------------------------------------------------
def _kwargs_of(r, prefix):
    out = {}
    for k in r.index:
        if not k.startswith(prefix):
            continue
        v = r[k]
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        if isinstance(v, (float, np.floating)) and float(v).is_integer():
            v = int(v)
        out[k[len(prefix):]] = v
    return out


def trajectory_theory(df):
    """Per-trajectory theory, reconstructed from the problem instance and the
    runner's x0 draw (``simulate.initial_state``):

    * ``e0_max`` = max_i |E_i(x0, 0)| and ``T_settle`` = exact settling time of
      e' = -Phi(e) from e0_max (gamma = 1, ``act.settling_time``), so that the
      exact-flow scheme needs ceil(T_settle/h) steps and the continuous flow
      settles at T_settle/gamma seconds;
    * ``T_uniform``: the uniform (fixed-/predefined-time) bound, gamma = 1
      (AL19 Thm 1 for SBP, 1/k1 for predefined_exp; None otherwise).

    Returns (table, E0) with E0[(problem_key, act, seed)] = (activation, E0 array)."""
    from znnbench.registry import make_model, make_problem
    from znnbench.simulate import initial_state

    rows, E0s = [], {}
    first = df.drop_duplicates(["problem_key", "act", "seed"])
    for _, r in first.iterrows():
        prob = make_problem(r["problem_key"], **_kwargs_of(r, "problem_kw_"))
        model = make_model(r["model_key"], **_kwargs_of(r, "model_kw_"))
        x0 = initial_state(prob, model, int(r["seed"]), int(r["B"]))
        E0 = np.asarray(prob.E(x0, prob.data(0.0)), dtype=float).reshape(int(r["B"]), -1)
        act = model.act
        E0s[(r["problem_key"], r["act"], int(r["seed"]))] = (act, E0)
        unif = None
        if hasattr(act, "tight_uniform_bound") and str(r["act"]).startswith("sbp"):
            unif = float(act.tight_uniform_bound(1.0))
        elif r["act"] == "predefined_exp":
            unif = float(act.upper_bound_time(None, 1.0))
        # error-unit arm: the dynamics are those of e_s = s E, so the settling time is T(s e0) (gamma = 1);
        # the uniform (fixed/predefined-time) bound does not depend on s
        m_ = df["problem_key"].eq(r["problem_key"]) & df["act"].eq(r["act"]) & df["seed"].astype(int).eq(int(r["seed"]))
        for s in sorted(df.loc[m_, "scale"].astype(float).unique()) if "scale" in df.columns else [1.0]:
            for b in range(E0.shape[0]):
                rows.append({"problem_key": r["problem_key"], "act": r["act"], "seed": int(r["seed"]), "x0_index": b,
                             "scale": float(s), "T_settle": float(act.settling_time(s * E0[b], 1.0)),
                             "T_uniform": unif, "e0_max": float(np.abs(E0[b]).max())})
    return pd.DataFrame(rows), E0s


def add_theory(df, th):
    """Merge the per-trajectory theory and derive
    * steps_theory / steps_uniform (exact-flow step counts and uniform bound),
    * t_settle = T_settle / gamma (continuous-time settling time, seconds) and
      ``unsettled`` = t_settle > start of the steady window (1 - FRAC) T: the
      steady-state metrics of such a trajectory still contain the transient
      (small gamma = h / tau), so it must not be read as instability."""
    if th is None or th.empty:
        df = df.copy()
        df["unsettled"] = False
        return df
    df = df.copy()
    df["seed"] = df["seed"].astype(int)
    df["x0_index"] = df["x0_index"].astype(int)
    df = df.merge(th, on=["problem_key", "act", "seed", "x0_index", "scale"], how="left")
    df["steps_theory"] = np.ceil(df["T_settle"] / df["h"] - 1e-9)
    df["steps_uniform"] = np.ceil(df["T_uniform"].astype(float) / df["h"] - 1e-9)
    df["t_settle"] = df["T_settle"] / df["gamma"].astype(float)
    df["unsettled"] = (df["t_settle"] > (1.0 - FRAC) * df["T"].astype(float)).fillna(False)
    return df


def predict_explicit_euler_divergence(act, E0, hs, n_steps, thr=1e10):
    """Explicit ZNN-Euler on a time-invariant affine problem is exactly the
    explicit Euler map of the error ODE, E_{k+1} = E_k - h Phi(E_k), applied
    component-wise (consistent.md, Lemma 1).  Iterating that scalar map from
    E0 predicts divergence (|E| > thr or non-finite) without the ZNN machinery.
    Returns a bool array (len(hs), B)."""
    E = np.repeat(np.asarray(E0, dtype=float)[None], len(hs), axis=0)       # (H, B, m)
    h = np.asarray(hs, dtype=float)[:, None, None]
    div = np.zeros(E.shape[:2], dtype=bool)
    with np.errstate(all="ignore"):
        for _ in range(int(n_steps)):
            E = E - h * act(E)
            bad = ~np.isfinite(E).all(axis=2) | (np.linalg.norm(np.nan_to_num(E, nan=np.inf), axis=2) > thr)
            div |= bad
            E = np.where(div[:, :, None], 0.0, E)
            if div.all():
                break
    return div


def dead_zone(act, h, lo=-40.0, hi=4.0, it=60):
    """Effective dead zone of the exact one-step map: delta(h) = T^{-1}(h), the
    largest error that the continuous flow e' = -Phi(e) (gamma = 1) removes
    within time h (Psi_exact(e) = 0 iff T(|e|) <= h).  sign: delta = h; a term
    a|e|^p sign(e), p < 1: delta = (a (1-p) h)^(1/(1-p)).  Returns inf when h
    exceeds the settling time from |e| = 10^hi (fixed/predefined-time)."""
    T = lambda x: float(act.settling_time(np.array([10.0 ** x]), 1.0))  # noqa: E731
    if T(hi) <= h:
        return math.inf
    if T(lo) > h:
        return 0.0
    for _ in range(it):
        mid = 0.5 * (lo + hi)
        if T(mid) <= h:
            lo = mid
        else:
            hi = mid
    return 10.0 ** lo


def analyze_ti(df, store, figdir, tabdir, S, E0s=None):
    """df already carries the per-trajectory theory columns (add_theory)."""
    acts = acts_present(df)
    dfx = df

    res = {"acts": acts, "cells": {}}
    # ---- explicit Euler divergence vs the error-ODE map prediction (exact on TI) ----
    pred_rows = []
    if E0s:
        N = int(round(float(df["T"].iloc[0]) / float(df["tau"].iloc[0])))
        for (pk, a, sd), (act, E0) in E0s.items():
            if pk != "linear_system" or a not in acts:
                continue
            div = predict_explicit_euler_divergence(act, E0, H_GRID, N)
            for ih, h in enumerate(H_GRID):
                for b in range(E0.shape[0]):
                    pred_rows.append({"act": a, "seed": sd, "x0_index": b, "h": round(float(h), 12),
                                      "pred_div": bool(div[ih, b])})
    if pred_rows:
        pr = pd.DataFrame(pred_rows)
        ex = dfx[dfx.disc_key.eq("euler")][["act", "seed", "x0_index", "h", "diverged_b", "e0_max"]]
        ex = ex.assign(seed=ex.seed.astype(int), x0_index=ex.x0_index.astype(int)).merge(
            pr, on=["act", "seed", "x0_index", "h"], how="inner")
        agree = float((ex.pred_div == ex.diverged_b).mean()) if len(ex) else math.nan
        per_act = {}
        for a, g in ex.groupby("act"):
            dv = g[g.diverged_b]
            per_act[a] = {"n": int(len(g)), "n_diverged": int(g.diverged_b.sum()),
                          "agreement": float((g.pred_div == g.diverged_b).mean()),
                          "smallest_h_with_divergence": float(dv.h.min()) if len(dv) else None,
                          "e0max_of_diverged_min": float(dv.e0_max.min()) if len(dv) else None}
        res["explicit_div_prediction"] = {"agreement": agree, "n": int(len(ex)), "per_act": per_act}
        print(f"[E8] TI explicit Euler divergence: error-ODE map prediction agrees in {100 * agree:.1f}% "
              f"of {len(ex)} trajectories")
    # ---- per-cell statistics ----------------------------------------------------
    for a in acts:
        for d in DISCS:
            for h in H_GRID:
                c = cell(dfx, act=a, disc_key=d, h=float(h))
                if c.empty:
                    continue
                st = c["steps_exact"].to_numpy(float)
                reached = np.isfinite(st)
                ent = {"n": int(len(c)), "n_diverged": int(c.diverged_b.sum()), "n_failed": int(c.failed_b.sum()),
                       "n_exact": int(reached.sum()), "steps": med_iqr(st),
                       "mssre": med_iqr(c["mssre"]), "init": med_iqr(c["initial_residual"]),
                       "chatter": med_iqr(c["chatter"]), "chatter_vec_eff": med_iqr(c["chatter_vec_eff"]),
                       "rate_alt_eff": med_iqr(c["rate_alt_eff"]),
                       "stable": bool((~c.diverged_b).all() and
                                      (c["mssre"] <= CONTRACT * c["initial_residual"]).all()),
                       "bounded": bool((~c.diverged_b).all()),
                       "n_unsettled": int(c["unsettled"].sum()) if "unsettled" in c.columns else 0}
                if "steps_theory" in c.columns and d == "exact_flow_euler":
                    diff = (c["steps_exact"] - c["steps_theory"]).to_numpy(float)
                    ent["theory_agree_pm1"] = float(np.mean(np.abs(diff) <= 1)) if len(diff) else math.nan
                    ent["theory_agree_exact"] = float(np.mean(diff == 0)) if len(diff) else math.nan
                    ent["steps_theory"] = med_iqr(c["steps_theory"])
                if "steps_uniform" in c.columns and np.isfinite(c["steps_uniform"]).any():
                    ent["steps_uniform"] = float(np.nanmax(c["steps_uniform"]))
                    ent["within_uniform"] = float(np.mean(st[reached] <= c["steps_uniform"].to_numpy(float)[reached])) \
                        if reached.any() else math.nan
                res["cells"][f"{a}|{d}|{h:g}"] = ent

    # ---- fig: residual traces (TI) ------------------------------------------------
    trace_acts = [a for a in ("sign", "sbp0.5", "predefined_exp") if a in acts] or acts[:3]
    if "traces_file" in df.columns and df["traces_file"].notna().any():
        clip = 1e-17
        fig, axes = plt.subplots(len(trace_acts), 2, figsize=(style.DOUBLE, 1.75 * len(trace_acts) + 0.5),
                                 sharey=True, squeeze=False)
        for i, a in enumerate(trace_acts):
            st_all = cell(df, act=a, h=H_REF)["steps_exact"].to_numpy(float)
            kmax = np.nanmax(st_all[np.isfinite(st_all)]) if np.isfinite(st_all).any() else 0
            xmax = int(max(20, 1.6 * kmax))
            for j, fam in enumerate((DISCS_EULER, DISCS_TZ)):
                ax = axes[i, j]
                for d in fam:
                    rows = cell(df, act=a, disc_key=d, h=H_REF)
                    t, R = store.stack(rows)
                    if R is None:
                        continue
                    k = np.rint(t / rows["tau"].iloc[0]).astype(int)
                    vis = k <= xmax
                    k, R = k[vis], R[vis]
                    # diverged samples are inf: keep them in the quantiles (the median leaves the
                    # panel once more than half of the trajectories diverged)
                    R = np.maximum(np.where(np.isnan(R), BIG, np.where(np.isinf(R), BIG, R)), clip)
                    med, q1, q3 = (np.where(v >= BIG / 10, np.nan, v)
                                   for v in np.quantile(R, [0.5, 0.25, 0.75], axis=1))
                    s = dstyle(d)
                    ax.plot(k, med, label=DISC_LABEL[d], color=s["color"], linestyle=s["linestyle"],
                            marker=s["marker"], markevery=max(1, xmax // 8), lw=1.0)
                    ax.fill_between(k, q1, np.where(np.isnan(q3), 1e3, q3), color=s["color"], alpha=0.08, lw=0)
                ax.axhline(EPS_EXACT, color=style.INK2, lw=0.5, ls=(0, (1, 2)))
                ax.set_yscale("log")
                ax.set_ylim(clip * 0.4, 1e3)
                ax.yaxis.set_major_locator(FixedLocator([clip, 1e-15, 1e-12, 1e-9, 1e-6, 1e-3, 1e0, 1e3]))
                ax.yaxis.set_major_formatter(zero_formatter(clip))
                ax.yaxis.set_minor_locator(FixedLocator([]))
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                if i == 0:
                    ax.set_title(("Euler look-ahead" if j == 0 else "Taylor--Zhang look-ahead").replace("--", "–"))
                if i == len(trace_acts) - 1:
                    ax.set_xlabel("step $k$")
                if j == 0:
                    ax.set_ylabel(f"{ACT_LABEL.get(a, a)}\n" + r"$\|E(x_k)\|_2$")
            for j in range(2):
                axes[i, j].set_xlim(0, xmax)
        for j in range(2):
            axes[0, j].legend(loc="center right", ncol=1, fontsize=5.8)
        fig.suptitle(f"Time-invariant linear system, $h={H_REF:g}$: median over 12 trajectories (band: IQR); "
                     r"dotted: $10^{-13}$; '0' = exact zero", fontsize=7.5, y=1.0)
        fig.tight_layout()
        save(fig, figdir, "fig_E8_traces_ti")

    # ---- fig: steps to exact convergence vs h ------------------------------------
    conv_discs = [d for d in DISCS if d not in EXPLICIT and d != NAIVE]
    ncol = min(3, len(acts))
    nrow = int(math.ceil(len(acts) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(style.DOUBLE, 2.0 * nrow + 0.4), squeeze=False, sharey=True)
    for idx, a in enumerate(acts):
        ax = axes[idx // ncol, idx % ncol]
        for d in conv_discs:
            hs, med, lo, hi = [], [], [], []
            for h in H_GRID:
                e = res["cells"].get(f"{a}|{d}|{h:g}")
                if e and e["n_exact"] == e["n"]:
                    hs.append(h)
                    med.append(e["steps"][0])
                    lo.append(e["steps"][0] - e["steps"][1])
                    hi.append(e["steps"][2] - e["steps"][0])
            if hs:
                s = dstyle(d)
                ax.errorbar(hs, np.maximum(med, 0.8), yerr=[lo, hi], label=DISC_LABEL[d], capsize=1.5,
                            elinewidth=0.6, **s)
        if "steps_theory" in dfx.columns:
            tt = [np.nanmedian(cell(dfx, act=a, disc_key="exact_flow_euler", h=float(h))["steps_theory"])
                  for h in H_GRID]
            if a in NO_EXACT_MAP:      # exact map falls back to the PEB Lie split: no exact-flow theory
                tt = [math.nan] * len(H_GRID)
            if np.isfinite(tt).any():
                ax.plot(H_GRID, np.maximum(tt, 0.8), color="#c9c8c4", ls="-", lw=4.0, marker=None, zorder=0,
                        solid_capstyle="round", label=r"theory, exact flow: $\lceil T(e_0)/h\rceil$")
            uu = [np.nanmax(cell(dfx, act=a, disc_key="exact_flow_euler", h=float(h))["steps_uniform"])
                  if "steps_uniform" in dfx.columns else np.nan for h in H_GRID]
            if np.isfinite(uu).any():
                ax.plot(H_GRID, np.maximum(uu, 0.8), color=style.INK, ls=(0, (2, 2)), lw=0.7, marker=None,
                        zorder=6, label=r"uniform (fixed/predefined-time) bound $\lceil T_{\max}/h\rceil$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(ACT_LABEL.get(a, a))
        if idx % ncol == 0:
            ax.set_ylabel(r"steps to $\|E\|<10^{-13}$")
        if idx // ncol == nrow - 1:
            ax.set_xlabel(r"$h=\gamma\tau$")
    for k in range(len(acts), nrow * ncol):
        axes[k // ncol, k % ncol].axis("off")
    handles, labels = [], []
    for ax in axes.flat:
        for hd, lb in zip(*ax.get_legend_handles_labels()):
            if lb not in labels:
                handles.append(hd)
                labels.append(lb)
    order = sorted(range(len(labels)), key=lambda i: (labels[i].startswith(("theory", "uniform")), i))
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.02), fontsize=6)
    naive_ok = sorted({a for a in acts for h in H_GRID
                       if res["cells"].get(f"{a}|{NAIVE}|{h:g}", {}).get("n_exact", 0) > 0})
    fig.suptitle("Time-invariant linear system: finite-step convergence (median, IQR bars; explicit Euler/TZ "
                 "never reach $10^{-13}$; the naive TZ "
                 + ("reaches it only for " + ", ".join(ACT_LABEL.get(a, a) for a in naive_ok) + " at small $h$"
                    if naive_ok else "never does")
                 + ")", fontsize=7.5)
    fig.tight_layout(rect=(0, 0.09 if nrow == 1 else 0.06, 1, 1))
    save(fig, figdir, "fig_E8_steps")

    # ---- fig: norm-based chattering is blind to e -> -e --------------------------
    fig, ax = plt.subplots(figsize=(style.SINGLE if len(acts) <= 3 else style.ONEHALF, 2.3))
    width = 0.8 / 4
    xs = np.arange(len(acts))
    bars = [("euler", "chatter", "Euler, norm-based"), ("euler", "chatter_vec_eff", "Euler, vector"),
            ("taylor_zhang", "chatter", "TZ, norm-based"), ("taylor_zhang", "chatter_vec_eff", "TZ, vector")]
    for k, (d, m, lab) in enumerate(bars):
        vals, err_lo, err_hi = [], [], []
        for a in acts:
            e = res["cells"].get(f"{a}|{d}|{H_REF:g}")
            key = "chatter" if m == "chatter" else "chatter_vec_eff"
            v = e[key] if e else (math.nan,) * 3
            vals.append(v[0] if np.isfinite(v[0]) else 0.0)
            err_lo.append(max(v[0] - v[1], 0) if np.isfinite(v[0]) else 0)
            err_hi.append(max(v[2] - v[0], 0) if np.isfinite(v[0]) else 0)
        s = dstyle(d)
        ax.bar(xs + (k - 1.5) * width, vals, width, yerr=[err_lo, err_hi], capsize=1.2,
               color=s["color"] if m != "chatter" else "white", edgecolor=s["color"],
               hatch="////" if m == "chatter" else None, lw=0.7, label=lab, error_kw={"elinewidth": 0.5})
        for x, a in zip(xs, acts):
            e = res["cells"].get(f"{a}|{d}|{H_REF:g}")
            if e and e["n_diverged"] == e["n"]:
                ax.text(x + (k - 1.5) * width, 0.02, "div.", rotation=90, fontsize=5, ha="center", va="bottom")
            elif e:
                v = e["chatter" if m == "chatter" else "chatter_vec_eff"][0]
                if np.isfinite(v) and v < 0.01:
                    ax.text(x + (k - 1.5) * width, 0.02, f"{v:.0g}" if v > 1e-6 else "0", fontsize=5,
                            ha="center", va="bottom", color=s["color"])
    ax.set_xticks(xs)
    ax.set_xticklabels([ACT_LABEL.get(a, a) for a in acts], rotation=20, ha="right")
    ax.set_ylabel("chattering index")
    ax.set_title(f"time-invariant problem, explicit schemes, $h={H_REF:g}$", fontsize=7.5)
    ax.legend(fontsize=5.8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    ax.set_ylim(0, 2.8)
    fig.tight_layout()
    save(fig, figdir, "fig_E8_chatter_blind")

    # ---- tables ---------------------------------------------------------------------
    hs_tab = [h for h in (0.03, 0.3, 3.0) if any(f"|{h:g}" in k for k in res["cells"])]
    rows = []
    for a in acts:
        for d in DISCS:
            r = [ACT_TEX.get(a, a) if d == DISCS[0] else "", tex_escape(DISC_LABEL[d])]
            for h in hs_tab:
                e = res["cells"].get(f"{a}|{d}|{h:g}")
                if not e:
                    r.append("--")
                elif e["n_diverged"] == e["n"]:
                    r.append("div.")
                elif e["n_exact"] == 0:
                    r.append(r"never")
                else:
                    m, q1, q3 = e["steps"]
                    txt = f"{m:.0f} [{q1:.0f}, {q3:.0f}]" if e["n_exact"] == e["n"] else \
                        f"{m:.0f} ({e['n_exact']}/{e['n']})"
                    if d == "exact_flow_euler" and "steps_theory" in e:
                        txt += f"; th.\\ {e['steps_theory'][0]:.0f}"
                    r.append(txt)
            rows.append(r)
        rows.append(r"\midrule")
    rows = rows[:-1]
    write_table(Path(tabdir) / "E8_steps.tex", ["activation", "scheme"] + [f"$h={h:g}$" for h in hs_tab], rows,
                "E8, time-invariant linear system: steps until $\\|E\\|_2<10^{-13}$ (median [IQR] over 12 "
                "trajectories; ``never'': not reached within the horizon; ``th.'': exact-flow prediction "
                "$\\lceil T(\\max_i|e_{0,i}|)/h\\rceil$).", "tab:E8_steps", colspec="llrrr"[:2 + len(hs_tab)], wide=True)

    # summary numbers
    S["ti"] = {"h_ref": H_REF, "cells": res["cells"]}
    hl = {}
    for a in acts:
        e_exp = res["cells"].get(f"{a}|euler|{H_REF:g}", {})
        hl[a] = {
            "explicit_euler_exact_reached": f"{e_exp.get('n_exact', 0)}/{e_exp.get('n', 0)}",
            "explicit_euler_diverged": f"{e_exp.get('n_diverged', 0)}/{e_exp.get('n', 0)}",
            "explicit_euler_chatter_norm": e_exp.get("chatter", [math.nan])[0],
            "explicit_euler_chatter_vec": e_exp.get("chatter_vec_eff", [math.nan])[0],
            "explicit_euler_rate_alt": e_exp.get("rate_alt_eff", [math.nan])[0],
            "steps_median": {d: res["cells"].get(f"{a}|{d}|{H_REF:g}", {}).get("steps", [math.nan])[0]
                             for d in DISCS},
            "exact_reached": {d: f"{res['cells'].get(f'{a}|{d}|{H_REF:g}', {}).get('n_exact', 0)}/"
                                 f"{res['cells'].get(f'{a}|{d}|{H_REF:g}', {}).get('n', 0)}" for d in DISCS},
            "exact_flow_theory_agreement_pm1": None if a in NO_EXACT_MAP else res["cells"].get(
                f"{a}|exact_flow_euler|{H_REF:g}", {}).get("theory_agree_pm1", math.nan),
        }
    S["ti"]["headline"] = hl
    # theory agreement over all h
    agree = [e.get("theory_agree_pm1") for k, e in res["cells"].items()
             if "|exact_flow_euler|" in k and e.get("theory_agree_pm1") is not None
             and not k.startswith("fixed_time")]
    S["ti"]["exact_flow_theory_agreement_pm1_all_h"] = float(np.nanmean(agree)) if agree else math.nan
    S["ti"]["explicit_div_prediction"] = res.get("explicit_div_prediction")
    return res


# ---------------------------------------------------------------------------
# stable h range
# ---------------------------------------------------------------------------
def h_ranges(df, sel):
    """Stable set over H_GRID: (longest contiguous stable block [lo, hi],
    largest h of the contiguous bounded block starting at the smallest h).

    A cell is "stable" when no trajectory diverged and MSSRE <= CONTRACT x the
    initial residual for every trajectory.  A bounded cell that fails the
    contraction test only because some trajectories had not settled before the
    steady window (``unsettled``: continuous-time settling time T(e0)/gamma
    beyond (1 - FRAC) T, i.e. a slow transient at small gamma = h/tau) is marked
    'u' (not assessable) instead of unstable."""
    st, bd, unset = [], [], []
    for h in H_GRID:
        c = cell(df, h=float(h), **sel)
        if c.empty:
            st.append(None)
            bd.append(None)
            continue
        b = bool((~c.diverged_b).all())
        bd.append(b)
        ok = b and bool((c["mssre"] <= CONTRACT * c["initial_residual"]).all())
        if b and not ok and "unsettled" in c.columns:
            fail = c["mssre"] > CONTRACT * c["initial_residual"]
            if bool(c.loc[fail, "unsettled"].all()):
                st.append("u")
                unset.append(float(h))
                continue
        st.append(ok)
    best, cur = (None, None, 0), None
    for i, sflag in enumerate(st):
        if sflag is True:
            cur = (cur[0], i, cur[2] + 1) if cur else (i, i, 1)
            if cur[2] > best[2]:
                best = cur
        else:
            cur = None
    rng = (H_GRID[best[0]], H_GRID[best[1]]) if best[2] else (math.nan, math.nan)
    hb = math.nan
    for i, b in enumerate(bd):
        if b:
            hb = H_GRID[i]
        else:
            break
    return {"stable_lo": rng[0], "stable_hi": rng[1], "n_stable": best[2], "bounded_hmax": hb,
            "stable_flags": st, "bounded_flags": bd, "unsettled_h": unset}


def fmt_range(r):
    u = r.get("unsettled_h") or []
    us = (" (u: " + ", ".join(f"{h:g}" for h in u) + ")") if u else ""
    if not r["n_stable"]:
        return "none" + us
    hi = r["stable_hi"]
    his = f"$\\geq${hi:g}" if hi == H_GRID[-1] else f"{hi:g}"
    return f"[{r['stable_lo']:g}, {his}]" + us


# ---------------------------------------------------------------------------
# (b) time-varying problems
# ---------------------------------------------------------------------------
def _paired_lr(cn, cd, metric):
    """Paired log10(num/den) over (seed, x0) pairs.  A diverged numerator gives
    +inf, a diverged denominator -inf, both diverged NaN (excluded)."""
    mg = cn[["pair", metric, "diverged_b"]].merge(cd[["pair", metric, "diverged_b"]], on="pair",
                                                   suffixes=("_n", "_d"))
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.log10(mg[metric + "_n"].astype(float).to_numpy() / mg[metric + "_d"].astype(float).to_numpy())
    dn, dd = mg["diverged_b_n"].to_numpy(bool), mg["diverged_b_d"].to_numpy(bool)
    lr = np.where(dn & ~dd, np.inf, np.where(dd & ~dn, -np.inf, np.where(dn & dd, np.nan, lr)))
    return mg, lr


def _headline_entry(p, a, lab, metric, num, den, tr, h, mg, lr, kind, extra=None):
    seeds = mg["pair"].astype(str).str.split(":").str[0].to_numpy()
    m, lo, hi = boot_ci_cluster(lr, seeds)          # instances are the resampling units (review)
    m_t, lo_t, hi_t = boot_ci(lr)                   # trajectory bootstrap, kept for comparison only
    per_seed = [med_iqr(lr[seeds == sd])[0] for sd in sorted(set(seeds))]
    w, n, ps = sign_test(cluster_medians(lr, seeds))
    e = {"problem": p, "act": a, "comparison": lab, "kind": kind, "metric": metric, "num": num, "den": den,
         "tau": tr, "h": h, "n_pairs": int(len(mg)), "num_diverged": int(mg["diverged_b_n"].sum()),
         "n_instances": int(len(set(seeds))), "ci95_traj_bootstrap": [lo_t, hi_t],
         "median_log10_ratio": m, "ci95": [lo, hi], "wins": w, "n_sign": n, "sign_p": ps,
         "per_instance_median_range": [float(np.min(_fin(per_seed))), float(np.max(_fin(per_seed)))] if per_seed
         else [math.nan, math.nan],
         "num_median": med_iqr(mg[metric + "_n"])[0], "den_median": med_iqr(mg[metric + "_d"])[0]}
    e["per_instance_median_range"] = [_unfin(v) for v in e["per_instance_median_range"]]
    if extra:
        e.update(extra)
    return e


def best_h(df, tr, **sel):
    """(h, median MSSRE) minimising the median MSSRE over the h grid at tau = tr,
    among cells with no diverged trajectory and no 'unsettled' trajectory (the
    flag depends only on activation, e0 and gamma, so it is common to every
    scheme at that h); None if no such cell."""
    best = None
    for h in H_GRID:
        c = cell(df, h=float(h), tau=float(tr), **sel)
        if c.empty or c.diverged_b.any():
            continue
        if "unsettled" in c.columns and c["unsettled"].any():
            continue
        m = float(c["mssre"].median())
        if best is None or m < best[1]:
            best = (float(h), m)
    return best


def analyze_tv(df, store, figdir, tabdir, S, E0s=None):
    acts = acts_present(df)
    probs = [p for p in ("linear_system", "matrix_inversion") if p in set(df.problem_key)]
    trs = {(p, a): tau_ref(df, p, a) for p in probs for a in acts
           if (df.problem_key.eq(p) & df.act.eq(a)).any()}
    out = {"acts": acts, "problems": probs, "tau_ref": {p: tau_ref(df, p) for p in probs}, "tau_ref_pa": trs}

    # ---- fig: residual traces (TV, tau = 1e-2) -----------------------------------
    trace_acts = [a for a in ("sign", "sbp0.5", "predefined_exp") if a in acts] or acts[:3]
    if "linear_system" in probs and df["traces_file"].notna().any():
        tau_tr = max(df.loc[df.problem_key.eq("linear_system"), "tau"].unique())
        fig, axes = plt.subplots(len(trace_acts), 2, figsize=(style.DOUBLE, 1.75 * len(trace_acts) + 0.5),
                                 sharey=True, sharex=True, squeeze=False)
        ylo = np.inf
        for i, a in enumerate(trace_acts):
            for j, fam in enumerate((DISCS_EULER, DISCS_TZ)):
                ax = axes[i, j]
                for d in fam:
                    rows = cell(df, problem_key="linear_system", act=a, disc_key=d, h=H_REF, tau=float(tau_tr))
                    t, R = store.stack(rows)
                    if R is None:
                        continue
                    R = np.where(np.isfinite(R), R, BIG)
                    med, q1, q3 = (np.where(v >= BIG / 10, np.nan, v)
                                   for v in np.quantile(R, [0.5, 0.25, 0.75], axis=1))
                    if np.all(np.isnan(med)):
                        continue
                    q3 = np.where(np.isnan(q3), 1e5, q3)
                    if np.isfinite(q1).any():
                        ylo = min(ylo, np.nanmin(q1))
                    s = dstyle(d)
                    ax.plot(t, med, label=DISC_LABEL[d], color=s["color"], linestyle=s["linestyle"],
                            marker=s["marker"], markevery=max(1, len(t) // 10), lw=0.9)
                    ax.fill_between(t, q1, q3, color=s["color"], alpha=0.08, lw=0)
                ax.set_yscale("log")
                if i == 0:
                    ax.set_title("Euler look-ahead" if j == 0 else "Taylor–Zhang look-ahead")
                if i == len(trace_acts) - 1:
                    ax.set_xlabel("$t$ [s]")
                if j == 0:
                    ax.set_ylabel(f"{ACT_LABEL.get(a, a)}\n" + r"$\|E(x_k,t_k)\|_2$")
        for j in range(2):
            axes[0, j].legend(loc="upper right", fontsize=5.8, ncol=2)
        axes[0, 0].set_ylim(ylo / 3 if np.isfinite(ylo) else None, 1e5)
        fig.suptitle(f"Time-varying linear system, $\\tau={tau_tr:g}$, $h={H_REF:g}$: median over 12 trajectories "
                     "(band: IQR)\ncurves leaving the top diverged; the consistent schemes of a family coincide "
                     "after the transient", fontsize=7.5, y=1.0)
        fig.tight_layout()
        save(fig, figdir, "fig_E8_traces_tv")

    # ---- fig: MSSRE vs h ------------------------------------------------------------
    for p in probs:
        acts_p = [a for a in acts if (p, a) in trs]
        if not acts_p:
            continue
        ncol = min(3, len(acts_p))
        nrow = int(math.ceil(len(acts_p) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(style.DOUBLE, 2.0 * nrow + 0.5), squeeze=False, sharey=True)
        any_unset = False
        for idx, a in enumerate(acts_p):
            ax = axes[idx // ncol, idx % ncol]
            tr = trs[(p, a)]
            ndiv = 0
            for d in DISCS_MAIN:
                med, q1, q3, divh, unh = [], [], [], [], []
                for h in H_GRID:
                    c = cell(df, problem_key=p, act=a, disc_key=d, h=float(h), tau=float(tr))
                    m = med_iqr(c["mssre"]) if not c.empty else (math.nan,) * 3
                    med.append(m[0]); q1.append(m[1]); q3.append(m[2])  # noqa: E702
                    if not c.empty and c.diverged_b.mean() > 0.5:
                        divh.append(h)
                    elif not c.empty and c["unsettled"].mean() > 0.5 and np.isfinite(m[0]):
                        unh.append((h, m[0]))
                med, q1, q3 = (np.where(np.isfinite(v), v, np.nan) for v in (med, q1, q3))
                s = dstyle(d)
                ax.plot(H_GRID, med, label=DISC_LABEL[d], **s)
                ax.fill_between(H_GRID, q1, q3, color=s["color"], alpha=0.10, lw=0)
                if unh:          # transient not finished before the steady window: hollow marker
                    any_unset = True
                    ax.plot([u[0] for u in unh], [u[1] for u in unh], ls="none", marker=s["marker"],
                            markerfacecolor="white", markeredgecolor=s["color"], ms=4.5, zorder=5)
                if divh:
                    ax.plot(divh, [1.03 + 0.06 * ndiv] * len(divh), transform=ax.get_xaxis_transform(), ls="none",
                            marker=s["marker"], color=s["color"], ms=3, clip_on=False)
                    ndiv += 1
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(ACT_LABEL.get(a, a) + ("" if np.isclose(tr, tau_ref(df, p)) else f" ($\\tau={tr:g}$)"),
                         pad=14)
            if idx % ncol == 0:
                ax.set_ylabel("MSSRE")
            if idx // ncol == nrow - 1:
                ax.set_xlabel(r"$h=\gamma\tau$")
        for k in range(len(acts_p), nrow * ncol):
            axes[k // ncol, k % ncol].axis("off")
        h_, l_ = axes[0, 0].get_legend_handles_labels()
        fig.legend(h_, l_, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01), fontsize=6)
        fig.suptitle(f"{PROB_LABEL[p].capitalize()}, {tau_desc(df, p, acts_p)}: MSSRE vs $h$ (median, band: IQR; "
                     "markers above the panel: >50% of trajectories diverged"
                     + ("; hollow: transient not finished before the steady window" if any_unset else "")
                     + ")", fontsize=7.5)
        fig.tight_layout(rect=(0, 0.1 if nrow == 1 else 0.07, 1, 1))
        save(fig, figdir, f"fig_E8_mssre_vs_h_{p}")

    # ---- fig: chattering (TV, tau_ref, h_ref): chatter_vec dot plot (log), rate_alt bars ----
    if "linear_system" in probs:
        p = "linear_system"
        acts_p = [a for a in acts if (p, a) in trs]
        fig, axes = plt.subplots(2, 2, figsize=(style.DOUBLE, 4.1), sharex=True, squeeze=False)
        xs = np.arange(len(acts_p))
        for j, fam in enumerate((DISCS_EULER, DISCS_TZ)):
            w = 0.82 / len(fam)
            for i, m in enumerate(("chatter_vec_eff", "rate_alt_eff")):
                ax = axes[i, j]
                for k, d in enumerate(fam):
                    vals, lo, hi = [], [], []
                    x = xs + (k - (len(fam) - 1) / 2) * w
                    for a in acts_p:
                        c = cell(df, problem_key=p, act=a, disc_key=d, h=H_REF, tau=float(trs[(p, a)]))
                        v = med_iqr(c[m]) if not c.empty else (math.nan,) * 3
                        ok = np.isfinite(v[0]) and not (not c.empty and c.diverged_b.mean() > 0.5)
                        vals.append(v[0] if ok else np.nan)
                        lo.append(max(v[0] - v[1], 0) if ok else 0)
                        hi.append(max(v[2] - v[0], 0) if ok else 0)
                        if not ok and not c.empty:
                            ax.text(x[acts_p.index(a)], 0.03, "div.", rotation=90, fontsize=5, ha="center",
                                    va="bottom", transform=ax.get_xaxis_transform(), color=dstyle(d)["color"])
                    s = dstyle(d)
                    if i == 0:      # log-scale dot + IQR whisker (values span ~3 decades)
                        ax.errorbar(x, vals, yerr=[lo, hi], ls="none", marker=s["marker"], color=s["color"],
                                    ms=4, capsize=1.2, elinewidth=0.6, label=DISC_LABEL[d])
                    else:           # bars from zero (a fraction in [0, 1])
                        ax.bar(x, np.nan_to_num(vals), w, yerr=[lo, hi], color=s["color"], edgecolor="white",
                               lw=0.3, capsize=1.0, error_kw={"elinewidth": 0.5}, label=DISC_LABEL[d],
                               hatch="xxxx" if d == NAIVE else None)
                if i == 0:
                    ax.set_yscale("log")
                    ax.set_ylim(1e-4, 10)
                    ax.legend(fontsize=5.8, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0),
                              title="Euler look-ahead" if j == 0 else "Taylor–Zhang look-ahead",
                              title_fontsize=7)
                else:
                    ax.set_ylim(0, 1.05)
                for xb in xs[:-1] + 0.5:
                    ax.axvline(xb, color=style.GREY, lw=0.4, alpha=0.5)
                ax.grid(axis="x", visible=False)
                if j == 0:
                    ax.set_ylabel(r"chatter$_{\rm vec}$ (log scale)" if i == 0 else
                                  "rate$_{\\rm alt}$ (state-increment sign flips)")
                ax.set_xticks(xs)
                ax.set_xticklabels([ACT_LABEL.get(a, a) for a in acts_p], rotation=20, ha="right")
        fig.suptitle(f"Chattering on the time-varying linear system, {tau_desc(df, p, acts_p)}, $h={H_REF:g}$ "
                     "(median with IQR whiskers over 12 trajectories)", fontsize=7.5)
        fig.tight_layout()
        save(fig, figdir, "fig_E8_chatter")

    # ---- empirical orders (MSSRE vs tau at fixed h_ref) + fixed-gamma diagonals ------------
    orders, orders_fg = {}, {}
    for p in probs:
        for a in acts:
            taus = sorted(df.loc[df.problem_key.eq(p) & df.act.eq(a), "tau"].unique(), reverse=True)
            if not taus:
                continue
            for d in DISCS:
                meds, piv = [], {}
                for t in taus:
                    c = cell(df, problem_key=p, act=a, disc_key=d, h=H_REF, tau=float(t))
                    meds.append(med_iqr(c["mssre"])[0] if not c.empty else math.nan)
                    piv[float(t)] = c.set_index("pair")["mssre"].astype(float) if not c.empty else pd.Series(dtype=float)
                fo = fit_order(taus, meds)
                fo["taus"] = [float(t) for t in taus]
                fo["mssre_median"] = meds
                fo["expected"] = 2 if d in DISCS_EULER else 3
                # per-trajectory slopes over the same asymptotic tau window (robustness of the median fit)
                if fo["n_asym"] >= 2:
                    M = pd.DataFrame({t: piv[t] for t in fo["taus_asym"]})
                    lt = np.log(np.asarray(fo["taus_asym"], dtype=float))
                    sl = []
                    for _, row in M.iterrows():
                        v = row.to_numpy(float)
                        if np.all(np.isfinite(v) & (v > ROUNDOFF)):
                            sl.append(float(np.polyfit(lt, np.log(v), 1)[0]))
                    fo["slope_traj"] = med_iqr(sl) if sl else (math.nan,) * 3
                    fo["n_traj"] = len(sl)
                orders[f"{p}|{a}|{d}"] = fo
                for g in GAMMA_DIAG:      # fixed gamma: h = g * tau must lie on the h grid
                    pts = [(t, g * t) for t in taus if any(np.isclose(g * t, H_GRID, rtol=1e-6))]
                    mg_ = []
                    for t, h in pts:
                        c = cell(df, problem_key=p, act=a, disc_key=d, h=round(h, 12), tau=float(t))
                        mg_.append(med_iqr(c["mssre"])[0] if not c.empty else math.nan)
                    fg = fit_order([t for t, _ in pts], mg_)
                    fg.update({"taus": [float(t) for t, _ in pts], "hs": [float(h) for _, h in pts],
                               "mssre_median": mg_, "gamma": g})
                    orders_fg[f"{p}|{a}|{d}|{g:g}"] = fg
        # figure: rows = look-ahead family (<= 5 series per panel), columns = activations
        acts_p = [a for a in acts if f"{p}|{a}|euler" in orders]
        ncol = len(acts_p)
        fig, axes = plt.subplots(2, ncol, figsize=(style.DOUBLE, 4.3), squeeze=False, sharey=True, sharex=True)
        for j, a in enumerate(acts_p):
            for i, fam in enumerate((DISCS_EULER, DISCS_TZ)):
                ax = axes[i, j]
                ymin = np.inf
                tt = None
                for d in fam:
                    fo = orders[f"{p}|{a}|{d}"]
                    v = np.array(fo["mssre_median"], dtype=float)
                    v = np.where(np.isfinite(v), v, np.nan)
                    if np.isfinite(v).any():
                        ymin = min(ymin, np.nanmin(v))
                    ax.plot(fo["taus"], v, label=DISC_LABEL[d], **dstyle(d))
                    tt = np.array(sorted(fo["taus"]))
                q = 2 if i == 0 else 3
                if np.isfinite(ymin) and tt is not None and len(tt) > 1:
                    y0 = ymin / 6.0
                    ax.plot(tt, y0 * (tt / tt[-1]) ** q, color=style.INK2, lw=0.6, ls=(0, (4, 3)), marker=None)
                    ax.text(tt[0], y0 * (tt[0] / tt[-1]) ** q, rf"$\tau^{q}$ ", fontsize=6, color=style.INK2,
                            ha="right", va="center")
                ax.set_xscale("log")
                ax.set_yscale("log")
                if i == 0:
                    ax.set_title(ACT_LABEL.get(a, a))
                if j == 0:
                    ax.set_ylabel(("Euler family" if i == 0 else "TZ family") + f"\nMSSRE ($h={H_REF:g}$)")
                if i == 1:
                    ax.set_xlabel(r"$\tau$ [s]")
        for i, loc, anchor in ((0, "upper center", (0.5, 0.955)), (1, "lower center", (0.5, 0.0))):
            hd, lb = axes[i, 0].get_legend_handles_labels()
            fig.legend(hd, lb, loc=loc, bbox_to_anchor=anchor, ncol=len(lb), fontsize=6)
        fig.suptitle(f"{PROB_LABEL[p].capitalize()}: MSSRE vs $\\tau$ at fixed $h={H_REF:g}$ (median over 12 "
                     "trajectories; dashed: reference slope; missing: diverged; fitted slopes in Table E8_orders)",
                     fontsize=7.5, y=1.0)
        fig.tight_layout(rect=(0, 0.05, 1, 0.965))
        save(fig, figdir, f"fig_E8_order_{p}")
    out["orders"] = orders
    out["orders_fixed_gamma"] = orders_fg

    # ---- stable h ranges ---------------------------------------------------------------
    ranges = {}
    for (p, a), tr in trs.items():
        for d in DISCS:
            ranges[f"{p}|{a}|{d}"] = h_ranges(df, {"problem_key": p, "act": a, "disc_key": d, "tau": float(tr)})
    out["ranges"] = ranges

    # ---- headline paired comparisons (bootstrap CIs, sign tests, Holm) --------------------
    comps = [("euler", "exact_flow_euler", "mssre", "explicit Euler / Euler+exact flow"),
             ("euler", "peb_euler", "mssre", "explicit Euler / Euler+PEB"),
             ("taylor_zhang", "implicit_taylor_zhang", "mssre", "explicit TZ / TZ-EM+resolvent"),
             (NAIVE, "implicit_taylor_zhang", "mssre", "naive TZ / TZ-EM (memory ablation)"),
             ("euler", "exact_flow_euler", "chatter_vec_eff", "chatter_vec: explicit Euler / exact flow"),
             (NAIVE, "implicit_taylor_zhang", "chatter_vec_eff", "chatter_vec: naive TZ / TZ-EM")]
    best_comps = [("euler", "exact_flow_euler", "explicit Euler at its best h / Euler+exact flow (same h)"),
                  ("taylor_zhang", "implicit_taylor_zhang", "explicit TZ at its best h / TZ-EM+resolvent (same h)")]
    head = []
    for p in probs:
        for a in acts:
            if (p, a) not in trs:
                continue
            tr = trs[(p, a)]
            for num, den, metric, lab in comps:
                cn = cell(df, problem_key=p, act=a, disc_key=num, h=H_REF, tau=float(tr))
                cd = cell(df, problem_key=p, act=a, disc_key=den, h=H_REF, tau=float(tr))
                mg, lr = _paired_lr(cn, cd, metric)
                if len(mg):
                    head.append(_headline_entry(p, a, lab, metric, num, den, tr, H_REF, mg, lr, kind="matched_h"))
            # unequal-gain check: the explicit scheme at the h of the grid that minimises its own
            # median MSSRE (bounded, settled), against the consistent scheme at that same h
            for num, den, lab in best_comps:
                bh = best_h(df, tr, problem_key=p, act=a, disc_key=num)
                if bh is None:
                    continue
                cn = cell(df, problem_key=p, act=a, disc_key=num, h=bh[0], tau=float(tr))
                cd = cell(df, problem_key=p, act=a, disc_key=den, h=bh[0], tau=float(tr))
                mg, lr = _paired_lr(cn, cd, "mssre")
                if not len(mg):
                    continue
                bd = best_h(df, tr, problem_key=p, act=a, disc_key=den)
                extra = {"t_eps_1e-3_num": med_iqr(cn["t_eps_0.001"])[0], "t_eps_1e-3_den": med_iqr(cd["t_eps_0.001"])[0],
                         "den_best_h": bd[0] if bd else None, "den_best_mssre": bd[1] if bd else None}
                head.append(_headline_entry(p, a, lab, "mssre", num, den, tr, bh[0], mg, lr, kind="best_h",
                                            extra=extra))
    # matched-h advantage profile: median paired log10(explicit / consistent) at every h of the grid
    adv = {}
    for (p, a), tr in trs.items():
        for num, den in (("euler", "exact_flow_euler"), ("taylor_zhang", "implicit_taylor_zhang")):
            prof = []
            for h in H_GRID:
                cn = cell(df, problem_key=p, act=a, disc_key=num, h=float(h), tau=float(tr))
                cd = cell(df, problem_key=p, act=a, disc_key=den, h=float(h), tau=float(tr))
                if cn.empty or cd.empty:
                    prof.append(None)
                    continue
                mg, lr = _paired_lr(cn, cd, "mssre")
                prof.append({"h": float(h), "median_log10_ratio": med_iqr(lr)[0],
                             "num_diverged": int(mg["diverged_b_n"].sum()),
                             "unsettled": int(cd["unsettled"].sum()) if "unsettled" in cd.columns else 0})
            adv[f"{p}|{a}|{num}/{den}"] = {"tau": tr, "profile": prof}
    out["advantage_vs_h"] = adv
    # Holm within each comparison family (one family = one comparison label across the problem x
    # activation cells; review: one family over all ~100 rows made even 10/10 instance wins
    # non-significant).  Instance-level sign tests: n = number of instances.
    for lab in sorted(set(e["comparison"] for e in head)):
        fam = [e for e in head if e["comparison"] == lab]
        for e, q in zip(fam, holm([e["sign_p"] for e in fam])):
            e["sign_p_holm"] = float(q)
    out["headline"] = head

    # ---- mechanism: dead zone of the exact map vs the per-step drift -------------------------
    # The consistent residual equals one local truncation error only when the map's effective
    # dead zone delta(h) = T^{-1}(h) exceeds the per-component drift per step (~ MSSRE_min/sqrt(m));
    # below that h the map does not zero the error and the floor grows (continuous activations).
    acts_obj = {}
    for (pk, a, sd), (act, _E0) in (E0s or {}).items():
        acts_obj.setdefault(a, act)
    dz = {}
    for (p, a), tr in trs.items():
        if a not in acts_obj:
            continue
        cells_ = []
        for h in H_GRID:
            c = cell(df, problem_key=p, act=a, disc_key="exact_flow_euler", h=float(h), tau=float(tr))
            ok = (not c.empty) and not c["unsettled"].any() and not c.diverged_b.any()
            cells_.append((float(h), float(c["mssre"].median()) if ok else math.nan))
        fin = [v for _, v in cells_ if np.isfinite(v)]
        if not fin:
            continue
        mmin = min(fin)
        m_comp = int(df.loc[df.problem_key.eq(p), "problem_m"].iloc[0]) if "problem_m" in df.columns else 1
        delta = [dead_zone(acts_obj[a], h) for h in H_GRID]
        h_dz = next((h for h, d in zip(H_GRID, delta) if math.sqrt(m_comp) * d >= mmin), None)
        flat = None
        for i in range(len(cells_)):
            tail = [v for _, v in cells_[i:] if np.isfinite(v)]
            if tail and all(v <= 1.25 * mmin for v in tail) and np.isfinite(cells_[i][1]):
                flat = cells_[i][0]
                break
        dz[f"{p}|{a}"] = {"tau": tr, "m": m_comp, "mssre_min": mmin, "delta": delta, "h_deadzone": h_dz,
                          "h_flat_1.25": flat, "mssre_by_h": [v for _, v in cells_]}
    out["dead_zone"] = dz

    # ---- Kendall tau_b: ranking agreement (raw medians and tie-tolerant) -----------------------
    kend = []
    if len(probs) == 2:
        for a in acts:
            if not all((p, a) in trs for p in probs):
                continue
            r = {p: [med_iqr(cell(df, problem_key=p, act=a, disc_key=d, h=H_REF,
                                  tau=float(trs[(p, a)]))["mssre"])[0] for d in DISCS] for p in probs}
            tb, pv = kendall_b(r[probs[0]], r[probs[1]])
            tt_, pt_ = kendall_b(r[probs[0]], r[probs[1]], tie_rel=TIE_REL)
            kend.append({"act": a, "what": f"{probs[0]} vs {probs[1]} (tau_ref, h={H_REF:g})", "tau_b": tb, "p": pv,
                         "tau_b_tied": tt_, "p_tied": pt_})
    for p in probs:
        for a in acts:
            taus = sorted(df.loc[df.problem_key.eq(p) & df.act.eq(a), "tau"].unique(), reverse=True)
            if len(taus) < 2:
                continue
            r0 = [med_iqr(cell(df, problem_key=p, act=a, disc_key=d, h=H_REF, tau=float(taus[0]))["mssre"])[0]
                  for d in DISCS]
            r1 = [med_iqr(cell(df, problem_key=p, act=a, disc_key=d, h=H_REF, tau=float(taus[-1]))["mssre"])[0]
                  for d in DISCS]
            tb, pv = kendall_b(r0, r1)
            tt_, pt_ = kendall_b(r0, r1, tie_rel=TIE_REL)
            kend.append({"act": a, "what": f"{p}: tau={taus[0]:g} vs tau={taus[-1]:g} (h={H_REF:g})",
                         "tau_b": tb, "p": pv, "tau_b_tied": tt_, "p_tied": pt_})
    out["kendall"] = kend

    # ---- initial-error distribution (explicit stability depends on it) -------------------------
    e0 = {}
    for p in probs:
        g = df[df.problem_key.eq(p)].drop_duplicates(["seed", "x0_index"])
        e0[p] = {"initial_residual": med_iqr(g["initial_residual"]),
                 "initial_residual_range": [float(g["initial_residual"].min()), float(g["initial_residual"].max())],
                 "e0_max_range": [float(g["e0_max"].min()), float(g["e0_max"].max())] if "e0_max" in g else None}
    out["e0"] = e0

    # ---- tables -----------------------------------------------------------------------------
    # chattering + MSSRE at h_ref
    rows = []
    p = "linear_system" if "linear_system" in probs else probs[0]
    acts_p = [a for a in acts if (p, a) in trs]
    for a in acts_p:
        for d in DISCS:
            c = cell(df, problem_key=p, act=a, disc_key=d, h=H_REF, tau=float(trs[(p, a)]))
            if c.empty:
                continue
            nd = int(c.diverged_b.sum())
            def mi(col, iqr=False):  # noqa: E306
                m, q1, q3 = med_iqr(c[col])
                if not np.isfinite(m):
                    return "div."
                return f"{fmt_num(m)} [{fmt_num(q1)}, {fmt_num(q3)}]" if iqr else fmt_num(m)
            rows.append([ACT_TEX.get(a, a) if d == DISCS[0] else "", tex_escape(DISC_LABEL[d]), mi("mssre", True),
                         mi("chatter"), mi("chatter_vec_eff"), mi("rate_alt_eff"), f"{nd}/{len(c)}"])
        rows.append(r"\midrule")
    write_table(Path(tabdir) / "E8_chatter.tex",
                ["activation", "scheme", "MSSRE", r"chatter (norm)", r"chatter$_{\rm vec}$", r"rate$_{\rm alt}$",
                 "div."], rows[:-1],
                f"E8, time-varying linear system, {tau_desc(df, p, acts_p)}, $h={H_REF:g}$: steady-state residual "
                "and chattering indices (median over 12 trajectories; IQR for MSSRE). The norm-based index compares "
                "$\\|E_{k+1}\\|$ and $\\|E_k\\|$; chatter$_{\\rm vec}$ uses $\\|E_{k+1}-E_k\\|$ relative to "
                "$\\|E_k\\|$ (for the consistent schemes it scales like $\\tau$, so ratios depend on $\\tau$); "
                "rate$_{\\rm alt}$ is the fraction of sign alternations of the state increments.",
                "tab:E8_chatter", colspec="llrrrrr", wide=True)

    # orders: fixed h (asymptotic / all points) and fixed gamma = 10 (continuous-time convention)
    g_tab = GAMMA_DIAG[0]
    rows = []
    for a in acts:
        for d in DISCS:
            if not any(f"{p}|{a}|{d}" in orders for p in probs):
                continue
            r = [ACT_TEX.get(a, a) if d == DISCS[0] else "", tex_escape(DISC_LABEL[d]),
                 str(2 if d in DISCS_EULER else 3)]
            for p in probs:
                fo = orders.get(f"{p}|{a}|{d}")
                if fo is None:
                    r.append("--")
                    continue
                sa, sl = fo["slope_asym"], fo["slope_all"]
                if not np.isfinite(sa):
                    r.append("div." if all(not np.isfinite(v) for v in fo["mssre_median"]) else "--")
                else:
                    r.append(f"{sa:.2f} ({fo['n_asym']}) / {sl:.2f}")
            for p in probs:
                fg = orders_fg.get(f"{p}|{a}|{d}|{g_tab:g}")
                if fg is None or not np.isfinite(fg["slope_asym"]):
                    r.append("div." if fg and fg["mssre_median"] and
                             all(not np.isfinite(v) for v in fg["mssre_median"]) else "--")
                else:
                    r.append(f"{fg['slope_asym']:.2f} ({fg['n_asym']})")
            rows.append(r)
        rows.append(r"\midrule")
    write_table(Path(tabdir) / "E8_orders.tex",
                ["activation", "scheme", "exp."] + [tex_escape(PROB_LABEL[p]) + f", $h={H_REF:g}$" for p in probs]
                + [tex_escape(PROB_LABEL[p]) + f", $\\gamma={g_tab:g}$" for p in probs], rows[:-1],
                f"E8: empirical order of MSSRE in $\\tau$. Fixed-$h$ columns ($h={H_REF:g}$, DTZNN convention): "
                "asymptotic slope (number of $\\tau$ points in the asymptotic range: the smallest-$\\tau$ tail whose "
                "local slopes lie within 0.3 of each other) / slope over all points above $10^{-12}$ (measured "
                f"round-off floor $\\approx 2\\times10^{{-14}}$). Fixed-$\\gamma$ columns ($\\gamma={g_tab:g}$, "
                "continuous-time convention, $h=\\gamma\\tau$ read off the grid): asymptotic slope (points). "
                "``exp.'': order of the look-ahead formula at fixed $h$.", "tab:E8_orders",
                colspec="llr" + "r" * (2 * len(probs)), wide=True)

    # stable h
    e0txt = "; ".join(f"{tex_escape(PROB_LABEL[p])}: $\\|E(0)\\|_2$ median {e0[p]['initial_residual'][0]:.3g} "
                      f"(range {e0[p]['initial_residual_range'][0]:.3g}--{e0[p]['initial_residual_range'][1]:.3g})"
                      + (f", $\\max_i|e_{{0,i}}|$ {e0[p]['e0_max_range'][0]:.3g}--{e0[p]['e0_max_range'][1]:.3g}"
                         if e0[p].get("e0_max_range") else "") for p in probs)
    rows = []
    for a in acts:
        for d in DISCS:
            r = [ACT_TEX.get(a, a) if d == DISCS[0] else "", tex_escape(DISC_LABEL[d])]
            ti = S.get("_ti_ranges", {}).get(f"{a}|{d}")
            r.append(fmt_range(ti) if ti else "--")
            for p in probs:
                rg = ranges.get(f"{p}|{a}|{d}")
                if rg is None:
                    r.append("--")
                    continue
                hb = rg["bounded_hmax"]
                hbs = "none" if not np.isfinite(hb) else (f"$\\geq${hb:g}" if hb == H_GRID[-1] else f"{hb:g}")
                r.append(f"{fmt_range(rg)}; {hbs}")
            rows.append(r)
        rows.append(r"\midrule")
    taus_txt = ", ".join(f"{tex_escape(PROB_LABEL[p])} {tau_desc(df, p, [a for a in acts if (p, a) in trs])}"
                         for p in probs)
    write_table(Path(tabdir) / "E8_stable_h.tex",
                ["activation", "scheme", "TI"] + [tex_escape(PROB_LABEL[p]) for p in probs], rows[:-1],
                "E8: stable range of $h$ on the grid " + ", ".join(f"{h:g}" for h in H_GRID) +
                f" ({taus_txt}). ``Stable'' = no trajectory diverged and MSSRE $\\le {CONTRACT:g}\\times$ initial "
                "residual for all 12 trajectories (longest contiguous block); after the semicolon: largest $h$ up "
                "to which no trajectory diverged. TI: time-invariant problem ($\\tau=0.01$). ``u'': bounded but not "
                "assessable, because the continuous-time settling time $T(e_0)/\\gamma$ exceeds the start of the "
                "steady window (small $\\gamma=h/\\tau$). The explicit bounded ranges depend on the initial error: "
                "explicit Euler blows up when $h\\,\\Phi(s)/s>2$ on the superlinear branch of $\\Phi$ "
                f"(Table/Summary: exact prediction on TI). Initial errors: {e0txt}.",
                "tab:E8_stable_h", colspec="llr" + "r" * len(probs), wide=True)

    # headline tables: compact (main text, one row per problem x activation) + full list (supplement)
    short = {"explicit Euler / Euler+exact flow": "Euler / Euler+exact",
             "explicit Euler / Euler+PEB": "Euler / Euler+PEB",
             "explicit TZ / TZ-EM+resolvent": "TZ / TZ-EM+res.",
             "naive TZ / TZ-EM (memory ablation)": "naive TZ / TZ-EM",
             "chatter_vec: explicit Euler / exact flow": r"chatter$_{\rm vec}$: Euler / Euler+exact",
             "chatter_vec: naive TZ / TZ-EM": r"chatter$_{\rm vec}$: naive TZ / TZ-EM",
             "explicit Euler at its best h / Euler+exact flow (same h)": r"Euler (best $h$) / Euler+exact",
             "explicit TZ at its best h / TZ-EM+resolvent (same h)": r"TZ (best $h$) / TZ-EM+res."}

    def ci_txt(e, with_h=False):
        m, (lo, hi) = e["median_log10_ratio"], e["ci95"]
        if e["num_diverged"] > e["n_pairs"] / 2:
            txt = f"div.\ ({e['num_diverged']}/{e['n_pairs']})"
        elif np.isfinite(m) and np.isfinite(lo) and np.isfinite(hi):
            txt = f"{m:.2f} [{lo:.2f}, {hi:.2f}]"
        else:
            txt = f"{fmt_num(m)} [{fmt_num(lo)}, {fmt_num(hi)}]"
        pq = e.get("sign_p_holm", math.nan)
        if not (np.isfinite(pq) and pq <= 0.05) and not e["num_diverged"] > e["n_pairs"] / 2:
            txt += r"$^{\dagger}$"
        if with_h:
            txt += f" ($h{{=}}{e['h']:g}$)"
        return txt

    rows = []
    for e in head:
        pq = e.get("sign_p_holm", math.nan)
        rows.append([tex_escape(PROB_LABEL[e["problem"]]), ACT_TEX.get(e["act"], e["act"]),
                     short.get(e["comparison"], tex_escape(e["comparison"])), f"{e['h']:g}", ci_txt(e),
                     f"{e['wins']}/{e['n_sign']}" + (f" ({fmt_num(pq)})" if np.isfinite(pq) else ""),
                     f"{e['num_diverged']}/{e['n_pairs']}"])
    cap_tau = ", ".join(f"{tex_escape(PROB_LABEL[p])}: {tau_desc(df, p, [a for a in acts if (p, a) in trs])}"
                        for p in probs)
    common_cap = ("median of the 12 paired $\\log_{10}$ ratios of MSSRE (numerator / denominator scheme, same "
                  "instance and initial state; positive = the consistent scheme is better) with a two-stage cluster "
                  "bootstrap 95\\% CI (2000 resamples of the problem instances, then of the initial states within "
                  "them: the instance is the experimental unit). ``best $h$'': the explicit "
                  "scheme at the grid value of $h$ that minimises its own median MSSRE (no divergence, transient "
                  "settled) and the consistent scheme at the same $h$. $^{\\dagger}$: one-sided sign test on the "
                  "per-instance medians not significant after Holm adjustment within the comparison family "
                  "($p>0.05$; with 3 instances the smallest attainable $p$ is 0.125). div.: more than half of the "
                  "explicit runs diverged. All comparisons are in the problem's own error unit (see "
                  "Table~\\ref{tab:E8_scale} for the dependence on the unit).")
    write_table(Path(tabdir) / "E8_headline_full.tex", ["problem", "activation", "comparison", "$h$",
                                                        r"$\log_{10}$ ratio [95\% CI]", "wins (Holm $p$)",
                                                        "num.\\ div."], rows,
                f"E8, all paired comparisons at the reference $\\tau$ ({cap_tau}): " + common_cap +
                " wins: pairs in which the denominator scheme is better.", "tab:E8_headline_full",
                colspec="llllrrr", wide=True)
    cols_c = [("explicit Euler / Euler+exact flow", f"Euler / Euler+exact, $h={H_REF:g}$", False),
              ("explicit Euler at its best h / Euler+exact flow (same h)", r"Euler (best $h$) / Euler+exact", True),
              ("explicit TZ / TZ-EM+resolvent", f"TZ / TZ-EM+res., $h={H_REF:g}$", False),
              ("naive TZ / TZ-EM (memory ablation)", f"naive TZ / TZ-EM, $h={H_REF:g}$", False),
              ("chatter_vec: explicit Euler / exact flow", r"chatter$_{\rm vec}$: Euler / Euler+exact", False)]
    rows = []
    for p in probs:
        for a in acts:
            if (p, a) not in trs:
                continue
            r = [tex_escape(PROB_LABEL[p]) if not rows or rows[-1] == r"\midrule" or
                 not any(isinstance(x, list) and x[0] == tex_escape(PROB_LABEL[p]) for x in rows) else "",
                 ACT_TEX.get(a, a) + ("" if np.isclose(trs[(p, a)], tau_ref(df, p)) else f" ($\\tau={trs[(p, a)]:g}$)")]
            for lab, _, wh in cols_c:
                e = next((x for x in head if x["problem"] == p and x["act"] == a and x["comparison"] == lab), None)
                r.append(ci_txt(e, with_h=wh) if e else "--")
            rows.append(r)
        rows.append(r"\midrule")
    write_table(Path(tabdir) / "E8_headline.tex", ["problem", "activation"] + [c for _, c, _ in cols_c], rows[:-1],
                f"E8 headline comparisons at the reference $\\tau$ ({cap_tau}): " + common_cap +
                " The full list with sign-test counts is Table~\\ref{tab:E8_headline_full}.", "tab:E8_headline",
                colspec="ll" + "r" * len(cols_c), wide=True)
    S["tv"] = {"tau_ref": {f"{p}|{a}": tr for (p, a), tr in trs.items()}, "orders": orders,
               "orders_fixed_gamma": orders_fg, "ranges": ranges, "headline": head, "kendall": kend, "e0": e0,
               "advantage_vs_h": adv, "dead_zone": dz}
    return out


# ---------------------------------------------------------------------------
# (c) formulas extension
# ---------------------------------------------------------------------------
def c_minus1(key):
    """Leading coefficient c_{-1} of a registered look-ahead formula (NaN if unknown)."""
    try:
        from znnbench.registry import make_discretization
        d = make_discretization(key)
        c = getattr(d, "coeffs", None)
        if c is None:
            c = (d.describe() or {}).get("coeffs")
        return float(c[0]) if c is not None else math.nan
    except Exception:  # noqa: BLE001
        return math.nan


def analyze_formulas(df, figdir, tabdir, S):
    acts = acts_present(df)
    formulas_all = list(dict.fromkeys(df["formula"]))
    group = keys.GROUPS["disc"]
    formulas = [f for f in group if f in formulas_all] + [f for f in formulas_all if f not in group]
    others = [f for f in formulas if f not in group]
    hs = sorted(df["h"].unique())
    res = {}
    for a in acts:
        for f in formulas:
            for v in ("explicit", "em_exact"):
                for h in hs:
                    c = df[(df.act == a) & (df.formula == f) & (df.variant == v) & np.isclose(df.h, h)]
                    if c.empty:
                        continue
                    taus = sorted(c.tau.unique(), reverse=True)
                    meds = [med_iqr(c[np.isclose(c.tau, t)]["mssre"])[0] for t in taus]
                    fo = fit_order(taus, meds)
                    fo.update({"taus": [float(t) for t in taus], "mssre_median": meds,
                               "expected": float(c["disc_expected_order"].dropna().iloc[0])
                               if c["disc_expected_order"].notna().any() else math.nan,
                               "n_diverged": int(c.diverged_b.sum()), "n": int(len(c))})
                    res[f"{a}|{f}|{v}|{h:g}"] = fo
    # figure: rows = activations, cols = variant (h = 0.3)
    hfig = H_REF if any(np.isclose(hs, H_REF)) else hs[-1]
    fig, axes = plt.subplots(len(acts), 2, figsize=(style.ONEHALF, 2.0 * len(acts) + 0.5), squeeze=False,
                             sharey=True)
    for i, a in enumerate(acts):
        for j, v in enumerate(("explicit", "em_exact")):
            ax = axes[i, j]
            for f in formulas:
                fo = res.get(f"{a}|{f}|{v}|{hfig:g}")
                if not fo:
                    continue
                y = np.array(fo["mssre_median"], dtype=float)
                y = np.where(np.isfinite(y), y, np.nan)
                s = fstyle(f, others)
                sl = fo["slope_asym"]
                lab = f"{f} (p={fo['expected']:.0f}" + (f", fit {abs(sl) if abs(sl) < 0.05 else sl:.1f})"
                                                         if np.isfinite(sl) else ", div.)")
                ax.plot(fo["taus"], y, label=lab, **s)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(("explicit formula" if v == "explicit" else "error memory + exact map") +
                         f", {ACT_LABEL.get(a, a)}")
            ax.legend(fontsize=5, loc="lower left" if v == "explicit" else "upper left")
            if j == 0:
                ax.set_ylabel(f"MSSRE ($h={hfig:g}$)")
            if i == len(acts) - 1:
                ax.set_xlabel(r"$\tau$ [s]")
    fig.suptitle("Error-memory rate on registered look-ahead formulas, time-varying linear system\n"
                 "(median over 12 trajectories; p: formula order; fit: asymptotic slope)", fontsize=7)
    fig.tight_layout()
    save(fig, figdir, "fig_E8_formulas")
    # table
    cm1 = {f: c_minus1(f) for f in formulas}
    rows = []
    for a in acts:
        for f in formulas:
            r = [ACT_TEX.get(a, a) if f == formulas[0] else "", tex_escape(f)]
            exp = next((res[k]["expected"] for k in res if k.startswith(f"{a}|{f}|")), math.nan)
            r.append(f"{exp:.0f}" if np.isfinite(exp) else "--")
            r.append(f"{cm1[f]:.3g}" if np.isfinite(cm1[f]) else "--")
            for h in hs:
                for v in ("explicit", "em_exact"):
                    fo = res.get(f"{a}|{f}|{v}|{h:g}")
                    if not fo:
                        r.append("--")
                    elif fo["n_diverged"] > fo["n"] / 2 and not np.isfinite(fo["slope_asym"]):
                        r.append("div.")
                    else:
                        r.append(f"{fo['slope_asym']:.2f}" if np.isfinite(fo["slope_asym"]) else "--")
            rows.append(r)
        rows.append(r"\midrule")
    cols = ["activation", "formula", "exp.", "$c_{-1}$"] + [f"{'expl.' if v == 'explicit' else 'EM'} $h={h:g}$"
                                                            for h in hs for v in ("explicit", "em_exact")]
    write_table(Path(tabdir) / "E8_formulas.tex", cols, rows[:-1],
                "E8 extension: asymptotic MSSRE slope in $\\tau$ of each registered look-ahead formula, explicit vs "
                "with the error-memory rate and the exact error map (``div.'': most trajectories diverged). "
                "$h=\\gamma\\tau$ is the raw product (runner convention): the explicit error recursion of a formula "
                "with leading coefficient $c_{-1}$ effectively steps $h/c_{-1}$, whereas with error memory the error "
                "map steps $h$ itself (Lemma 2); slopes, not floor levels, are compared across formulas.",
                "tab:E8_formulas", colspec="llrr" + "r" * (2 * len(hs)), wide=True)
    S["formulas"] = {"formulas": formulas, "fits": res, "c_minus1": cm1}
    return res


# ---------------------------------------------------------------------------
# (d) error-unit (scale) arm
# ---------------------------------------------------------------------------
def two_cycle_amp(act, h, lo=-30.0, hi=6.0):
    """Amplitude e* > 0 of the symmetric period-2 orbit e -> -e of explicit Euler (h Phi(e) = 2 e) on the
    sub-linear branch near 0, in the unit of the activation's argument (smallest root of h Phi(e)/e = 2).
    SBP(r): ~ (h/4)^(1/(1-r)) for r = 1/2; sign: h/2 (its real orbit alternates between a and h - a)."""
    def g(x):
        e = 10.0 ** x
        with np.errstate(all="ignore"):
            return h * float(act(np.array([e]))[0]) / e - 2.0
    xs = np.linspace(lo, hi, 721)
    vals = np.array([g(x) for x in xs])
    if not np.isfinite(vals[0]) or vals[0] <= 0:
        return math.nan
    idx = np.where(vals <= 0)[0]
    if not idx.size:
        return math.nan
    a, b = xs[idx[0] - 1], xs[idx[0]]
    for _ in range(60):
        mid = 0.5 * (a + b)
        if g(mid) > 0:
            a = mid
        else:
            b = mid
    return 10.0 ** (0.5 * (a + b))


def _collapse_fit(x, y, grid=np.arange(-4.0, 4.0001, 0.01)):
    """Fit y = log10(1 + 10^(x + c)) (no advantage for x << 0, advantage ~ x + c for x >> 0); returns
    (c, rms).  x, y finite arrays."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size < 2:
        return (math.nan, math.nan)
    pred = np.log10(1.0 + 10.0 ** (x[None, :] + grid[:, None]))
    sse = ((y[None, :] - pred) ** 2).sum(axis=1)
    i = int(np.argmin(sse))
    return (float(grid[i]), float(math.sqrt(sse[i] / x.size)))


def analyze_scale(data, E0s_ti, figdir, tabdir, S):
    """Error-unit arm: E expressed in units of 1/s (ScaledError), paired with the base arm (s = 1).

    TI: exact-flow step counts vs ceil(T(s e0)/h) and the s-free uniform bound, explicit-Euler
    divergence vs the scalar-map prediction from s E0, the explicit band vs the 2-cycle amplitude
    e*(h)/s.  TV: the paired explicit / consistent MSSRE ratio against the dimensionless band-to-drift
    ratio rho = sqrt(m) delta(h) / (s F(tau)), delta = dead zone of the exact map in the unit of the
    activation's argument, F = one-step truncation floor of the consistent schemes at tau."""
    if not any(k in data for k in SCALE_CFGS):
        return None
    discs_s = list(E8.DISCS_SCALE)

    def pooled(base, arm):
        if arm not in data:
            return pd.DataFrame()
        a_df = data[arm]
        parts = [a_df]
        if base in data:
            b = data[base]
            b = b[b.act.isin(set(a_df.act)) & b.disc_key.isin(discs_s) & b.problem_key.isin(set(a_df.problem_key))]
            taus = a_df.tau.unique()
            b = b[np.isclose(b.tau.to_numpy(float)[:, None], taus[None, :], rtol=1e-9).any(axis=1)]
            parts.append(b)
        return pd.concat(parts, ignore_index=True)

    dti, dtv = pooled("E8_ti", "E8_scale_ti"), pooled("E8_tv", "E8_scale_tv")
    out = {"scales": SCALE_ORDER, "ti": {}, "tv": {}}
    acts = [a for a in ACT_ORDER if a in set(dti.act) | set(dtv.act)]
    # ---- pairing check: the initial residual (problem unit) must not depend on s ---------------
    pc = []
    for d_ in (dti, dtv):
        if d_.empty:
            continue
        g = d_.groupby(["problem_key", "seed", "x0_index"])["initial_residual"]
        rel = (g.max() - g.min()) / g.max()
        pc.append(float(rel.max()))
    out["pairing_initial_residual_max_rel_diff"] = max(pc) if pc else math.nan
    act_obj = {a: v[0] for (pk, a, sd), v in (E0s_ti or {}).items()}

    # ---- TI ---------------------------------------------------------------------------------------
    ti_cells, ti_div = {}, {}
    if not dti.empty:
        N = int(round(float(dti["T"].iloc[0]) / float(dti["tau"].iloc[0])))
        m_ti = int(dti["problem_m"].iloc[0]) if "problem_m" in dti.columns else 1
        grp = {k: g for k, g in dti.groupby(["act", "disc_key", "scale", "h"])}
        for (a, d, s, h), c in grp.items():
            st = c["steps_exact"].to_numpy(float)
            reached = np.isfinite(st)
            e = {"n": int(len(c)), "n_diverged": int(c.diverged_b.sum()), "n_failed": int(c.failed_b.sum()),
                 "n_exact": int(reached.sum()), "steps": med_iqr(st), "mssre": med_iqr(c["mssre"])}
            if "steps_theory" in c.columns:
                thr = c["steps_theory"].to_numpy(float)
                e["n_beyond_horizon"] = int((thr > N).sum())
                if d == "exact_flow_euler" and a not in NO_EXACT_MAP:
                    # theory reached within the horizon -> must equal; beyond the horizon -> must not be reached
                    ok = np.where(thr <= N, st == thr, ~reached)
                    e["theory_agree_exact"] = float(np.mean(ok))
                    e["steps_theory"] = med_iqr(thr)
                if np.isfinite(c["steps_uniform"].astype(float)).any() and reached.any():
                    su = c["steps_uniform"].to_numpy(float)
                    e["steps_uniform"] = float(np.nanmax(su))
                    e["within_uniform"] = float(np.mean(st[reached] <= su[reached]))
            if d == "euler" and a in act_obj and (~c.diverged_b).any():
                es = two_cycle_amp(act_obj[a], float(h))
                if np.isfinite(es) and a != "sign":
                    v = c.loc[~c.diverged_b, "mssre"].to_numpy(float) * float(s) / (math.sqrt(m_ti) * es)
                    e["band_ratio"] = med_iqr(v)          # = 1: band is the 2-cycle e*(h) in the unit of E
            ti_cells[f"{a}|{d}|{s:g}|{h:g}"] = e
        # explicit-Euler divergence: scalar error map from s E0 (Lemma 1)
        rows = []
        for (pk, a, sd), (act, E0) in (E0s_ti or {}).items():
            if pk != "linear_system" or a not in set(dti.act):
                continue
            for s in sorted(dti.loc[dti.act.eq(a), "scale"].unique()):
                div = predict_explicit_euler_divergence(act, float(s) * E0, H_GRID, N, thr=1e10 * float(s))
                for ih, h in enumerate(H_GRID):
                    for b in range(E0.shape[0]):
                        rows.append({"act": a, "seed": int(sd), "x0_index": b, "scale": float(s),
                                     "h": round(float(h), 12), "pred_div": bool(div[ih, b])})
        if rows:
            pr = pd.DataFrame(rows)
            ex = dti[dti.disc_key.eq("euler")][["act", "seed", "x0_index", "scale", "h", "diverged_b"]]
            ex = ex.merge(pr, on=["act", "seed", "x0_index", "scale", "h"], how="inner")
            for (a, s), g in ex.groupby(["act", "scale"]):
                bd = g.groupby("h").diverged_b.any()
                hb = math.nan
                for h in H_GRID:
                    if round(h, 12) in bd.index and not bd[round(h, 12)]:
                        hb = h
                    else:
                        break
                ti_div[f"{a}|{s:g}"] = {"n": int(len(g)), "n_diverged": int(g.diverged_b.sum()),
                                       "agreement": float((g.pred_div == g.diverged_b).mean()),
                                       "bounded_hmax": hb,
                                       "frac_div_by_h": {f"{h:g}": float(g[np.isclose(g.h, h)].diverged_b.mean())
                                                         for h in H_GRID}}
        cons = dti[~dti.disc_key.isin(list(EXPLICIT))]
        out["ti"] = {"N": N, "cells": ti_cells, "explicit_div": ti_div,
                     "consistent_n_diverged": int(cons.diverged_b.sum()), "consistent_n": int(len(cons)),
                     "explicit_euler_n_exact": int(np.isfinite(dti.loc[dti.disc_key.eq("euler"), "steps_exact"]).sum())}

    # ---- TV ---------------------------------------------------------------------------------------
    tv_cells = {}
    if not dtv.empty:
        m_tv = int(dtv["problem_m"].iloc[0]) if "problem_m" in dtv.columns else 1
        grp = {k: g for k, g in dtv.groupby(["act", "disc_key", "scale", "tau", "h"])}
        taus = sorted(dtv.tau.unique())
        floor = {}
        for t in taus:
            v = [float(g["mssre"].median()) for (a, d, s, tt, h), g in grp.items()
                 if d == "exact_flow_euler" and np.isclose(tt, t) and not g.diverged_b.any()
                 and not g["unsettled"].any()]
            floor[float(t)] = min(v) if v else math.nan
        dz_cache = {}
        pairs = [("euler", "exact_flow_euler"), ("euler", "implicit_euler_err"),
                 (NAIVE, "implicit_taylor_zhang"), ("taylor_zhang", "implicit_taylor_zhang")]
        for (a, d, s, t, h), cn in grp.items():
            for num, den in pairs:
                if d != num:
                    continue
                cd = grp.get((a, den, s, t, h))
                if cd is None:
                    continue
                mg, lr = _paired_lr(cn, cd, "mssre")
                if (a, h) not in dz_cache and a in act_obj:
                    dz_cache[(a, h)] = dead_zone(act_obj[a], float(h))
                dz = dz_cache.get((a, h), math.nan)
                rho = math.sqrt(m_tv) * dz / (float(s) * floor[float(t)]) if np.isfinite(floor[float(t)]) else math.nan
                tv_cells[f"{a}|{num}/{den}|{s:g}|{t:g}|{h:g}"] = {
                    "act": a, "pair": f"{num}/{den}", "scale": float(s), "tau": float(t), "h": float(h),
                    "median_log10_ratio": med_iqr(lr)[0], "num_diverged": int(mg["diverged_b_n"].sum()),
                    "n_pairs": int(len(mg)), "unsettled": int(cd["unsettled"].sum() + cn["unsettled"].sum()),
                    "rho": rho, "num_median": med_iqr(mg["mssre_n"])[0], "den_median": med_iqr(mg["mssre_d"])[0]}
        # collapse: does rho (which contains s) predict the advantage for every unit?
        coll = {}
        for pair in ("euler/exact_flow_euler", f"{NAIVE}/implicit_taylor_zhang", "euler/implicit_euler_err"):
            for a in acts:
                pts = [v for v in tv_cells.values() if v["pair"] == pair and v["act"] == a
                       and v["num_diverged"] == 0 and v["unsettled"] == 0 and np.isfinite(v["median_log10_ratio"])
                       and np.isfinite(v["rho"]) and v["rho"] > 0]
                if len(pts) < 3:
                    continue
                x = np.log10([v["rho"] for v in pts])
                y = np.array([v["median_log10_ratio"] for v in pts])
                x_naive = x + np.log10([v["scale"] for v in pts])     # the same ratio ignoring the unit
                c, rms = _collapse_fit(x, y)
                c0, rms0 = _collapse_fit(x_naive, y)
                lowr = y[x <= 0]
                coll[f"{a}|{pair}"] = {
                    "n_cells": int(len(pts)), "scales": sorted({v["scale"] for v in pts}),
                    "c": c, "rms": rms, "c_unit_naive": c0, "rms_unit_naive": rms0,
                    "x_range": [float(x.min()), float(x.max())],
                    "n_low": int(lowr.size), "low_median": float(np.median(lowr)) if lowr.size else math.nan,
                    "low_range": [float(lowr.min()), float(lowr.max())] if lowr.size else [math.nan, math.nan],
                    "n_explicit_better": int((y < 0).sum()),
                    "explicit_better_cells": [f"s={v['scale']:g},tau={v['tau']:g},h={v['h']:g}: {v['median_log10_ratio']:.2f}"
                                              for v, yy in zip(pts, y) if yy < -0.05],
                    "high_offset_median": float(np.median((y - x)[x >= 1])) if (x >= 1).any() else math.nan}
        # headline per (act, s) at h_ref, tau_ref = 1e-3 if present: cluster-bootstrap CIs
        t_ref = 1e-3 if any(np.isclose(taus, 1e-3)) else min(taus)
        head = {}
        for a in acts:
            for s in SCALE_ORDER:
                ent = {"tau": t_ref, "h": H_REF}
                for num, den in pairs:
                    cn, cd = grp.get((a, num, s, t_ref, H_REF)), grp.get((a, den, s, t_ref, H_REF))
                    if cn is None or cd is None:
                        continue
                    mg, lr = _paired_lr(cn, cd, "mssre")
                    sd = mg["pair"].astype(str).str.split(":").str[0].to_numpy()
                    ent[f"{num}/{den}"] = {"ci": boot_ci_cluster(lr, sd), "num_diverged": int(mg["diverged_b_n"].sum()),
                                           "n": int(len(mg)), "unsettled": int(cd["unsettled"].sum())}
                k = tv_cells.get(f"{a}|euler/exact_flow_euler|{s:g}|{t_ref:g}|{H_REF:g}")
                ent["rho"] = k["rho"] if k else math.nan
                # explicit Euler bounded block from h = 0.01 at t_ref; consistent schemes bounded on the grid?
                hb = math.nan
                for h in H_GRID:
                    c = grp.get((a, "euler", s, t_ref, round(h, 12)))
                    if c is not None and not c.diverged_b.any():
                        hb = h
                    else:
                        break
                ent["explicit_bounded_hmax"] = hb
                cons = [g for (aa, d, ss, tt, h), g in grp.items() if aa == a and np.isclose(ss, s)
                        and d not in EXPLICIT]
                ent["consistent_n_diverged"] = int(sum(int(g.diverged_b.sum()) for g in cons))
                if len(ent) > 2:
                    head[f"{a}|{s:g}"] = ent
        out["tv"] = {"floor": floor, "tau_ref": t_ref, "cells": tv_cells, "collapse": coll, "headline": head,
                     "m": m_tv}

    # ---- figures --------------------------------------------------------------------------------------
    pacts = [a for a in SCALE_ACTS if a in acts]
    if out["ti"] and pacts:
        N = out["ti"]["N"]
        fig, axes = plt.subplots(2, len(pacts), figsize=(style.DOUBLE, 4.3), squeeze=False, sharex=True)
        for j, a in enumerate(pacts):
            ax = axes[0, j]
            for s in SCALE_ORDER:
                hs, med, cens = [], [], []
                for h in H_GRID:
                    e = ti_cells.get(f"{a}|exact_flow_euler|{s:g}|{h:g}")
                    if not e:
                        continue
                    if e["n_exact"] == e["n"]:
                        hs.append(h)
                        med.append(max(e["steps"][0], 0.8))
                    else:
                        cens.append(h)
                st_ = sstyle(s)
                if hs:
                    ax.plot(hs, med, label=slabel(s), **st_)
                if cens:            # small per-s offset so that censored markers of different s do not overlap
                    ax.plot(cens, [N * 1.35 ** SCALE_ORDER.index(s)] * len(cens), ls="none", marker=st_["marker"],
                            markerfacecolor="white", markeredgecolor=st_["color"], ms=4.5, clip_on=False)
            su = [ti_cells.get(f"{a}|exact_flow_euler|1|{h:g}", {}).get("steps_uniform", math.nan) for h in H_GRID]
            if np.isfinite(su).any():
                ax.plot(H_GRID, np.maximum(su, 0.8), color=style.INK, lw=0.8, ls=(0, (2, 2)), marker=None,
                        label=r"uniform bound $\lceil T_{\max}/h\rceil$ (every $s$)")
            ax.axhline(N, color=style.GREY, lw=0.6, ls=":")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(ACT_LABEL.get(a, a))
            if j == 0:
                ax.set_ylabel("Euler + exact flow:\n" + r"steps to $\|E\|<10^{-13}$")
            ax = axes[1, j]
            for s in SCALE_ORDER:
                e = ti_div.get(f"{a}|{s:g}")
                if not e:
                    continue
                ax.plot(H_GRID, [e["frac_div_by_h"].get(f"{h:g}", math.nan) for h in H_GRID], label=slabel(s),
                        **sstyle(s))
            ax.axhline(0, color=style.GREY, lw=0.6)
            ax.set_ylim(-0.05, 1.05)
            ax.set_xscale("log")
            ax.set_xlabel(r"$h=\gamma\tau$")
            if j == 0:
                ax.set_ylabel("explicit Euler:\nfraction diverged")
        hd, lb = axes[0, -1].get_legend_handles_labels()
        hd2, lb2 = axes[0, 0].get_legend_handles_labels()
        for hh, ll in zip(hd2, lb2):
            if ll not in lb:
                hd.append(hh)
                lb.append(ll)
        fig.legend(hd, lb, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01), fontsize=6)
        fig.suptitle("Time-invariant linear system, error in units of 1/s: consistent map reaches exact zero at "
                     f"every s (open: beyond the {N}-step horizon, as predicted); explicit Euler never does\n"
                     "and its bounded range of h shrinks with s (no consistent trajectory diverged at any s, h)",
                     fontsize=7.2)
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        save(fig, figdir, "fig_E8_scale_ti")
    if out.get("tv") and pacts:
        fig_scale_tv(out["tv"], pacts, figdir)

    # ---- table ----------------------------------------------------------------------------------------
    rows = []
    head = out.get("tv", {}).get("headline", {})
    for a in acts:
        for s in SCALE_ORDER:
            dv = ti_div.get(f"{a}|{s:g}", {})
            ef = ti_cells.get(f"{a}|exact_flow_euler|{s:g}|{H_REF:g}", {})
            hd_ = head.get(f"{a}|{s:g}", {})
            if not dv and not hd_:
                continue
            hb = dv.get("bounded_hmax", math.nan)
            r = [ACT_TEX.get(a, a) if s == SCALE_ORDER[0] else "", f"{s:g}",
                 ("none" if not np.isfinite(hb) else (f"$\\geq${hb:g}" if hb == H_GRID[-1] else f"{hb:g}"))
                 + (f" ({dv['n_diverged']}/{dv['n']})" if dv else "")]
            if ef:
                txt = (f"{ef['steps'][0]:.0f}" if ef["n_exact"] == ef["n"] else
                       f"{ef['steps'][0]:.0f} ({ef['n_exact']}/{ef['n']})" if ef["n_exact"] else "$>$horizon")
                if "steps_uniform" in ef:
                    txt += f" [$\\leq${ef['steps_uniform']:.0f}]"
                r.append(txt)
            else:
                r.append("--")
            r.append(fmt_num(hd_.get("rho", math.nan)) if hd_ else "--")
            for pk in ("euler/exact_flow_euler", "euler/implicit_euler_err", f"{NAIVE}/implicit_taylor_zhang"):
                v = hd_.get(pk)
                if not v:
                    r.append("--")
                elif v["num_diverged"] > v["n"] / 2:
                    r.append(f"div.\\ ({v['num_diverged']}/{v['n']})")
                else:
                    m, lo, hi = v["ci"]
                    r.append(f"{m:.2f} [{lo:.2f}, {hi:.2f}]" + ("$^u$" if v["unsettled"] else ""))
            hbt = hd_.get("explicit_bounded_hmax", math.nan) if hd_ else math.nan
            r.append("none" if not np.isfinite(hbt) else (f"$\\geq${hbt:g}" if hbt == H_GRID[-1] else f"{hbt:g}"))
            rows.append(r)
        rows.append(r"\midrule")
    if rows:
        t_ref = out.get("tv", {}).get("tau_ref", math.nan)
        write_table(Path(tabdir) / "E8_scale.tex",
                    ["activation", "$s$", "TI: expl.\\ Euler bounded up to $h$ (div.)",
                     f"TI: exact-flow steps, $h={H_REF:g}$", f"TV: $\\rho$", "TV: Euler / Euler+exact",
                     "TV: Euler / Euler+resolvent", "TV: naive TZ / TZ-EM", "TV: expl.\\ Euler bounded up to $h$"],
                    rows[:-1],
                    "E8 error-unit arm: the error is expressed in units of $1/s$ ($E_s=sE$; $s=1$ is the problem's "
                    "own unit, i.e.\\ the base arm), paired by instance and initial state. TI: time-invariant linear "
                    "system; explicit Euler never reaches $10^{-13}$ at any $s$; exact-flow steps = median (bracket: "
                    "the uniform fixed/predefined-time bound, independent of $s$; ``$>$horizon'': $\\lceil T(se_0)/h"
                    "\\rceil$ exceeds the 2500-step horizon, as predicted). TV: time-varying linear system at "
                    f"$\\tau={t_ref:g}$, $h={H_REF:g}$, median paired $\\log_{{10}}$ MSSRE ratio (cluster bootstrap 95\\% CI); "
                    "$\\rho=\\sqrt m\\,\\delta(h)/(sF(\\tau))$ is the band-to-drift ratio ($\\delta$: dead zone of the "
                    "exact map in the unit of $E_s$, $F$: one-step truncation floor). $^u$: some transients not "
                    "settled before the steady window. No consistent trajectory diverged at any $s$ or $h$.",
                    "tab:E8_scale", colspec="lr" + "r" * 7, wide=True)
    S["scale"] = out
    return out


# ---------------------------------------------------------------------------
# cost extrapolation to the full scale
# ---------------------------------------------------------------------------
SCALE_ACT_TITLE = {"sign": "sign", "sbp0.5": "sign-bi-power, $r=0.5$", "predefined_exp": "predefined-time"}


def fig_scale_tv(tv, pacts, figdir):
    """Time-varying error-unit arm (paper figure fig_E8_scale_tv): paired explicit / consistent
    MSSRE ratio against log10 rho, rho = sqrt(m) delta(h) / (s F(tau)) (band-to-drift ratio), for every
    (s, tau, h) cell without explicit divergence and with a settled transient; dashed: one fit
    log10(1 + 10^(x + c)) for all units s (its rms in the top-row panels)."""
    tv_cells = tv["cells"]
    coll = tv["collapse"]
    pairs = ("euler/exact_flow_euler", f"{NAIVE}/implicit_taylor_zhang")
    ylab = ["exact flow vs. explicit\ngain (decades)", "error memory vs. naive\ngain (decades)"]
    with style.print_rc():
        fig, axes = plt.subplots(2, len(pacts), figsize=(style.TEXTWIDTH, 3.55), squeeze=False, sharey="row",
                                 constrained_layout=True)
        fig.get_layout_engine().set(h_pad=0.04, w_pad=0.03, hspace=0.05, wspace=0.03)
        for i, pair in enumerate(pairs):
            for j, a in enumerate(pacts):
                ax = axes[i, j]
                xs_all = []
                for sc in SCALE_ORDER:
                    pts = [v for v in tv_cells.values() if v["pair"] == pair and v["act"] == a
                           and np.isclose(v["scale"], sc) and v["num_diverged"] == 0 and v["unsettled"] == 0
                           and np.isfinite(v["median_log10_ratio"]) and np.isfinite(v["rho"]) and v["rho"] > 0]
                    if not pts:
                        continue
                    x = np.log10([v["rho"] for v in pts])
                    xs_all += list(x)
                    st_ = sstyle(sc)
                    ax.plot(x, [v["median_log10_ratio"] for v in pts], ls="none", marker=st_["marker"],
                            color=st_["color"], ms=4.0, label=("$s=1$ (problem unit)" if sc == 1 else f"$s={sc:g}$"))
                cf = coll.get(f"{a}|{pair}")
                if cf and xs_all:
                    xx = np.linspace(min(xs_all) - 0.3, max(xs_all) + 0.3, 100)
                    ax.plot(xx, np.log10(1 + 10 ** (xx + cf["c"])), color=style.INK2, lw=0.9, ls=(0, (4, 2)),
                            marker=None, label=r"one fit for all $s$: $\log_{10}(1+10^{c}\rho)$")
                ax.axhline(0, color=style.GREY, lw=0.6)
                ax.xaxis.set_major_locator(MultipleLocator(2))
                ax.yaxis.set_major_locator(MultipleLocator(2))
                if i == 0:
                    ax.set_title(SCALE_ACT_TITLE.get(a, ACT_LABEL.get(a, a)))
                    if cf:
                        ax.text(0.97, 0.04, f"fit rms {cf['rms']:.2f}\n(ignoring $s$: {cf['rms_unit_naive']:.2f})",
                                transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color=style.INK2,
                                linespacing=1.1)
                if i == 1:
                    ax.set_xlabel(r"$\log_{10}\rho$")
                if j == 0:
                    ax.set_ylabel(ylab[i])
        hd, lb = [], []
        for ax in axes.flat:
            for hh, ll in zip(*ax.get_legend_handles_labels()):
                if ll not in lb:
                    hd.append(hh)
                    lb.append(ll)
        fig.legend(hd, lb, loc="outside lower center", ncol=3, frameon=False, columnspacing=1.4)
        return save(fig, figdir, "fig_E8_scale_tv")


def cost_estimate(data, S):
    frames = []
    for name, df in data.items():
        g = df.drop_duplicates("run_id")[["config", "problem_key", "disc_key", "act", "tau", "wall",
                                          "wall_per_step", "steps", "steps_run", "peak_rss_mb", "chunk"]]
        frames.append(g)
    runs = pd.concat(frames, ignore_index=True)
    meas = {"specs": int(len(runs)), "core_seconds": float(runs["wall"].sum()),
            "steps": float(runs["steps_run"].sum()),
            "peak_rss_mb_max": float(runs["peak_rss_mb"].max()),
            "wall_per_step_median_us": float(1e6 * runs["wall_per_step"].median())}
    lut = runs.groupby(["config", "problem_key", "disc_key", "act"])["wall_per_step"].median()
    lut2 = runs.groupby(["config", "problem_key", "disc_key"])["wall_per_step"].median()
    lut3 = runs.groupby(["problem_key", "disc_key"])["wall_per_step"].median()
    glob = float(runs["wall_per_step"].median())
    old = os.environ.get("ZNNBENCH_SCALE")
    os.environ["ZNNBENCH_SCALE"] = "full"
    try:
        from znnbench.runner import expand
        cfgs = E8.get_configs()
        tot_s, tot_steps, nspec, per_cfg = 0.0, 0.0, 0, {}
        for cfg in cfgs:
            cs = 0.0
            for s in expand(cfg):
                prob = s["problem"]["key"]
                disc = s["disc"]["key"]
                mk = s["model"]["kwargs"]
                act = mk["activation"] if mk["activation"] != "sign_bi_power" else f"sbp{float(mk['r']):g}"
                steps = (s["T"] or 10.0) / s["tau"]
                w = lut.get((cfg["name"], prob, disc, act))
                if w is None or not np.isfinite(w):
                    w = lut2.get((cfg["name"], prob, disc))
                if w is None or not np.isfinite(w):
                    w = lut3.get((prob, disc), glob)
                cs += steps * float(w)
                tot_steps += steps
                nspec += 1
            per_cfg[cfg["name"]] = cs
            tot_s += cs
    finally:
        if old is None:
            os.environ.pop("ZNNBENCH_SCALE", None)
        else:
            os.environ["ZNNBENCH_SCALE"] = old
    full = {"specs": nspec, "simulate_steps": tot_steps, "core_hours": tot_s / 3600.0,
            "core_hours_per_config": {k: v / 3600 for k, v in per_cfg.items()},
            "note": "steps x measured mini wall_per_step per (config, problem, disc, activation); activations absent "
                    "from the mini run use the (config, problem, disc) median; diverged runs stop early, so this is "
                    "an upper bound; per-task process start-up (~2 s) not included"}
    S["cost"] = {"mini_measured": meas, "full_estimate": full}
    return S["cost"]


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return None if math.isnan(v) else (("inf" if v > 0 else "-inf") if math.isinf(v) else v)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def _f(v, fmt=".3g", none="--"):
    if v is None:
        return none
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    if math.isnan(v):
        return none
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    return format(v, fmt)


def write_summary(results, S, data):
    res = Path(results)
    (res / "summary.json").write_text(json.dumps(_clean(S), indent=1))
    L = ["# E8 summary: implicit (consistent) vs explicit discretization of sign-type ZNN", "",
         f"Results folder: `{res}`; configs: {', '.join(sorted(data))}. "
         f"Statistics: {S.get('protocol', {}).get('n_instances', '?')} problem instances (seeds) x 4 initial "
         "states per cell; median [IQR]; paired log10 ratios with a two-stage CLUSTER bootstrap 95% CI (2000 "
         "resamples, seed 0: instances, then initial states within instances; the instance is the experimental "
         "unit because the B = 4 initial states of one instance give identical or nearly identical steady-state "
         "metrics); one-sided sign tests on the per-instance medians, Holm-adjusted within each comparison family "
         "(with 3 instances the smallest attainable p is 0.125, so no mini-scale test can be significant).", "",
         "**Units.** Every non-linear activation here has no length scale, so all numbers in (a)-(c) are for the "
         "error E in the problem's own unit; section (d) re-runs the core comparisons with E expressed in units "
         "of 1/s (s = 0.01 ... 1000) and states which conclusions survive.", "",
         "**Censoring.** Diverged trajectories enter every median, CI, ratio and ranking as +inf (worst case); "
         "time-to-eps never reached = +inf; trajectories that did not settle before the steady window are "
         "flagged (u); failed / skipped / timed-out runs: "
         + (", ".join(f"{k}: {v}" for k, v in S.get("protocol", {}).get("n_failed", {}).items()) or "none")
         + " (they would be reported here and per cell as n_failed; none is dropped silently).", ""]
    if "ti" in S:
        L += ["## (a) Time-invariant linear system (finite-step convergence), h = %g" % H_REF, "",
              "| activation | expl. Euler reaches 1e-13 | expl. Euler diverged | chatter (norm) | chatter_vec | "
              "rate_alt | steps: resolvent / PEB / exact / semi-impl. / TZ-EM-res / TZ-EM-PEB | exact-flow = theory (+-1) |",
              "|---|---|---|---|---|---|---|---|"]
        for a, e in S["ti"]["headline"].items():
            st = e["steps_median"]
            sts = " / ".join("%.0f" % st[d] if st[d] is not None and np.isfinite(st[d]) else "never"
                             for d in ("implicit_euler_err", "peb_euler", "exact_flow_euler", "semi_implicit_euler",
                                       "implicit_taylor_zhang", "peb_taylor_zhang"))
            alldiv = e["explicit_euler_diverged"].split("/")[0] == e["explicit_euler_diverged"].split("/")[1]
            cv = "div." if alldiv else _f(e["explicit_euler_chatter_vec"])
            L.append(f"| {a} | {e['explicit_euler_exact_reached']} | {e['explicit_euler_diverged']} | "
                     f"{'div.' if alldiv else _f(e['explicit_euler_chatter_norm'])} | {cv} | "
                     f"{'div.' if alldiv else _f(e['explicit_euler_rate_alt'])} | {sts} | "
                     f"{_f(e['exact_flow_theory_agreement_pm1'], '.2f', 'n/a (fallback map)')} |")
        L += ["", f"Exact-flow step counts agree with ceil(T(e0)/h) (+-1) in "
                  f"{100 * S['ti']['exact_flow_theory_agreement_pm1_all_h']:.0f}% of trajectories over all h "
                  "(sign, SBP, predefined-time; fixed-time excluded because its exact map falls back to PEB).", ""]
        dp = S["ti"].get("explicit_div_prediction")
        if dp:
            L += [f"Explicit Euler divergence on the time-invariant problem is predicted exactly by iterating the "
                  f"scalar error map e <- e - h Phi(e) from each trajectory's E(0): agreement "
                  f"{100 * dp['agreement']:.1f}% over {dp['n']} trajectories x h values. Per activation "
                  "(smallest h with a divergence; smallest max_i|e0_i| among diverged runs):", ""]
            for a, v in dp["per_act"].items():
                L.append(f"- {a}: {v['n_diverged']}/{v['n']} diverged, agreement {100 * v['agreement']:.0f}%, "
                         f"smallest h {_f(v['smallest_h_with_divergence'], 'g', 'none')}, "
                         f"min e0_max of diverged {_f(v['e0max_of_diverged_min'], '.3g', '-')}")
            L.append("")
    if "tv" in S:
        tv = S["tv"]
        L += ["## (b) Time-varying problems", "", "Initial errors (explicit stability depends on them): " +
              "; ".join(f"{p}: ||E(0)|| median {_f(v['initial_residual'][0])}, range "
                        f"{_f(v['initial_residual_range'][0])}-{_f(v['initial_residual_range'][1])}"
                        + (f", max_i|e0_i| {_f(v['e0_max_range'][0])}-{_f(v['e0_max_range'][1])}"
                           if v.get('e0_max_range') else "")
                        for p, v in tv.get("e0", {}).items()), "",
              "### Headline paired comparisons", "",
              "| problem | activation | comparison | tau | h | median log10 ratio [95% CI] | per-instance medians | "
              "wins (sign test, Holm p) | numerator diverged |",
              "|---|---|---|---|---|---|---|---|---|"]
        for e in tv["headline"]:
            m, (lo, hi) = e["median_log10_ratio"], e["ci95"]
            pr = e.get("per_instance_median_range", [math.nan, math.nan])
            extra = ""
            if e.get("kind") == "best_h":
                extra = (f" (t_eps(1e-3): {_f(e.get('t_eps_1e-3_num'))} s vs {_f(e.get('t_eps_1e-3_den'))} s; "
                         f"consistent best {_f(e.get('den_best_mssre'))} at h={_f(e.get('den_best_h'), 'g')})")
            L.append(f"| {e['problem']} | {e['act']} | {e['comparison']}{extra} | {e['tau']:g} | {e['h']:g} | "
                     f"{_f(m, '.2f')} [{_f(lo, '.2f')}, {_f(hi, '.2f')}] | {_f(pr[0], '.2f')} to {_f(pr[1], '.2f')} | "
                     f"{e['wins']}/{e['n_sign']} ({_f(e.get('sign_p_holm'), '.2g')}) | "
                     f"{e['num_diverged']}/{e['n_pairs']} |")
        adv = tv.get("advantage_vs_h", {})
        if adv:
            L += ["", "### Matched-h advantage profile: median paired log10(explicit / consistent) MSSRE vs h "
                      "(D = more than half of the explicit runs diverged, u = some consistent runs unsettled)", "",
                  "| problem | activation | pair | tau | " + " | ".join(f"h={h:g}" for h in H_GRID) + " |",
                  "|---|---|---|---|" + "---|" * len(H_GRID)]
            for k, v in adv.items():
                p_, a_, pair_ = k.split("|")
                cells = []
                for e in v["profile"]:
                    if e is None:
                        cells.append("--")
                    elif e["num_diverged"] > 6:
                        cells.append("D")
                    else:
                        cells.append(_f(e["median_log10_ratio"], ".2f") + ("u" if e["unsettled"] else ""))
                L.append(f"| {p_} | {a_} | {pair_} | {v['tau']:g} | " + " | ".join(cells) + " |")
        dzs = tv.get("dead_zone", {})
        if dzs:
            L += ["", "### Mechanism: effective dead zone delta(h) = T^{-1}(h) of the exact map vs the consistent "
                      "floor", "",
                  "The consistent residual is one local truncation error only when sqrt(m) delta(h) >= the drift "
                  "per step (~ MSSRE_min). h_dz: smallest grid h where that holds; h_flat: smallest h from which the "
                  "exact-flow Euler MSSRE stays within 1.25 x its minimum.", "",
                  "| problem | activation | tau | MSSRE_min | delta(0.01) | delta(0.3) | h_dz | h_flat |",
                  "|---|---|---|---|---|---|---|---|"]
            for k, v in dzs.items():
                p_, a_ = k.split("|")
                d = dict(zip(H_GRID, v["delta"]))
                L.append(f"| {p_} | {a_} | {v['tau']:g} | {_f(v['mssre_min'])} | {_f(d.get(0.01))} | "
                         f"{_f(d.get(0.3))} | {_f(v['h_deadzone'], 'g')} | {_f(v['h_flat_1.25'], 'g')} |")
        L += ["", "### Empirical orders of MSSRE in tau at fixed h = %g "
                  "(asymptotic / all points; [IQR of per-trajectory slopes])" % H_REF, "",
              "| problem | activation | " + " | ".join(DISCS) + " |", "|---|---|" + "---|" * len(DISCS)]
        probs_o = sorted(set(k.split("|")[0] for k in tv["orders"]))
        for p in probs_o:
            for a in [x for x in ACT_ORDER if any(k.startswith(f"{p}|{x}|") for k in tv["orders"])]:
                cells = []
                for d in DISCS:
                    fo = tv["orders"].get(f"{p}|{a}|{d}")
                    if fo is None or not np.isfinite(fo["slope_asym"]):
                        cells.append("div." if fo and all(not np.isfinite(v) for v in fo["mssre_median"]) else "--")
                    else:
                        st_ = fo.get("slope_traj", (math.nan,) * 3)
                        cells.append(f"{fo['slope_asym']:.2f}/{fo['slope_all']:.2f} [{_f(st_[1], '.2f')}, "
                                     f"{_f(st_[2], '.2f')}]")
                L.append(f"| {p} | {a} | " + " | ".join(cells) + " |")
        fg_all = tv.get("orders_fixed_gamma", {})
        for g in GAMMA_DIAG:
            L += ["", f"### Empirical orders at fixed gamma = {g:g} (continuous-time convention, h = gamma tau; "
                      "asymptotic slope (points))", "",
                  "| problem | activation | " + " | ".join(DISCS) + " |", "|---|---|" + "---|" * len(DISCS)]
            for p in probs_o:
                for a in [x for x in ACT_ORDER if f"{p}|{x}|euler|{g:g}" in fg_all]:
                    cells = []
                    for d in DISCS:
                        fg = fg_all.get(f"{p}|{a}|{d}|{g:g}")
                        if fg is None or not np.isfinite(fg["slope_asym"]):
                            cells.append("div." if fg and fg["mssre_median"] and
                                         all(not np.isfinite(v) for v in fg["mssre_median"]) else "--")
                        else:
                            cells.append(f"{fg['slope_asym']:.2f} ({fg['n_asym']})")
                    L.append(f"| {p} | {a} | " + " | ".join(cells) + " |")
        L += ["", "### Kendall tau_b (ranking of the 9 schemes by median MSSRE; raw / tie-tolerant, "
                  f"medians within {100 * TIE_REL:.0f}% tied)", ""]
        for k in tv["kendall"]:
            L.append(f"- {k['act']}, {k['what']}: tau_b = {_f(k['tau_b'], '.2f')} (p = {_f(k['p'], '.2g')}) / "
                     f"{_f(k.get('tau_b_tied'), '.2f')} (p = {_f(k.get('p_tied'), '.2g')})")
        L.append("")
    if "formulas" in S:
        L += ["## (c) Extension: error memory on registered look-ahead formulas (asymptotic slope at h = %g)" % H_REF,
              ""]
        cm1 = S["formulas"].get("c_minus1", {})
        for k, fo in S["formulas"]["fits"].items():
            a, f, v, h = k.split("|")
            if float(h) == H_REF:
                L.append(f"- {a}, {f} (c_-1 = {_f(cm1.get(f))}), {v}: slope {_f(fo['slope_asym'], '.2f')} "
                         f"({fo['n_asym']} pts; expected {fo['expected']:.0f}), diverged {fo['n_diverged']}/{fo['n']}")
        L.append("")
    sc = S.get("scale")
    if sc:
        L += ["## (d) Error-unit (scale) arm: E expressed in units of 1/s (ScaledError), paired with s = 1", "",
              f"Pairing check: the initial residual (problem unit) differs across s by at most "
              f"{_f(sc.get('pairing_initial_residual_max_rel_diff'), '.2g')} (relative), i.e. the same instances and "
              "initial states.", ""]
        ti = sc.get("ti", {})
        if ti:
            agr = [e["theory_agree_exact"] for k, e in ti["cells"].items() if "theory_agree_exact" in e]
            wu = [e["within_uniform"] for k, e in ti["cells"].items() if "|exact_flow_euler|" in k
                  and np.isfinite(e.get("within_uniform", math.nan))]
            br = [e["band_ratio"][0] for e in ti["cells"].values() if "band_ratio" in e and np.isfinite(e["band_ratio"][0])]
            L += [f"- TI, consistent schemes: {ti['consistent_n_diverged']}/{ti['consistent_n']} trajectories diverged "
                  "over every s and h. Exact-flow step counts equal ceil(T(s e0)/h) (or, when that exceeds the "
                  f"{ti['N']}-step horizon, are not reached, as predicted) in {100 * float(np.mean(agr)):.1f}% of "
                  f"trajectories over all (activation, s, h); they stay within the s-free uniform bound "
                  f"ceil(T_max/h) in {100 * float(np.mean(wu)) if wu else float('nan'):.0f}% (SBP, predefined-time).",
                  f"- TI, explicit Euler: reaches 1e-13 in {ti['explicit_euler_n_exact']} trajectories at any s. Its "
                  "bounded period-2 band equals the 2-cycle amplitude e*(h) of the activation IN THE UNIT OF E_s, i.e. "
                  f"sqrt(m) e*(h)/s in the problem unit: measured MSSRE / prediction median {_f(np.median(br) if br else math.nan, '.3f')} "
                  f"(range {_f(min(br) if br else math.nan, '.3f')}-{_f(max(br) if br else math.nan, '.3f')}) over "
                  "the bounded SBP / predefined-time cells.",
                  "- TI, explicit-Euler divergence (scalar error map iterated from s E0; bounded block of h from 0.01):", ""]
            for k, v in ti["explicit_div"].items():
                a_, s_ = k.split("|")
                L.append(f"  - {a_}, s = {s_}: {v['n_diverged']}/{v['n']} diverged, prediction agreement "
                         f"{100 * v['agreement']:.0f}%, bounded up to h = {_f(v['bounded_hmax'], 'g', 'none')}")
            L.append("")
        tv = sc.get("tv", {})
        if tv:
            L += [f"- TV floors F(tau) (one-step truncation floor of Euler + exact flow, min over cells): "
                  + ", ".join(f"tau={t:g}: {_f(v)}" for t, v in tv["floor"].items()), "",
                  f"Headline at tau = {tv['tau_ref']:g}, h = {H_REF:g} (median paired log10 MSSRE ratio [cluster "
                  "95% CI]; D = more than half of the explicit runs diverged):", "",
                  "| activation | s | rho | Euler / exact flow | Euler / resolvent | naive TZ / TZ-EM | explicit TZ / TZ-EM | "
                  "explicit Euler bounded up to h | consistent diverged |",
                  "|---|---|---|---|---|---|---|---|---|"]
            for k, e in tv["headline"].items():
                a_, s_ = k.split("|")
                cells = []
                for pk in ("euler/exact_flow_euler", "euler/implicit_euler_err", f"{NAIVE}/implicit_taylor_zhang",
                           "taylor_zhang/implicit_taylor_zhang"):
                    v = e.get(pk)
                    if not v:
                        cells.append("--")
                    elif v["num_diverged"] > v["n"] / 2:
                        cells.append(f"D ({v['num_diverged']}/{v['n']})")
                    else:
                        m, lo, hi = v["ci"]
                        cells.append(f"{_f(m, '.2f')} [{_f(lo, '.2f')}, {_f(hi, '.2f')}]" + ("u" if v["unsettled"] else ""))
                L.append(f"| {a_} | {s_} | {_f(e.get('rho'))} | " + " | ".join(cells) +
                         f" | {_f(e.get('explicit_bounded_hmax'), 'g', 'none')} | {e.get('consistent_n_diverged')} |")
            L += ["", "Collapse test over every (s, tau, h) cell with no explicit divergence and settled transients: "
                      "y = median log10(explicit / consistent) vs x = log10 rho, fitted by y = log10(1 + 10^(x + c)) "
                      "with ONE c for all s (rms), and with x computed ignoring the unit (rms_unit_naive):", ""]
            for k, v in tv["collapse"].items():
                L.append(f"- {k}: {v['n_cells']} cells, s in {v['scales']}, x in [{v['x_range'][0]:.1f}, "
                         f"{v['x_range'][1]:.1f}]; c = {v['c']:.2f}, rms = {v['rms']:.2f} (ignoring s: "
                         f"{v['rms_unit_naive']:.2f}); x <= 0: {v['n_low']} cells, median y {_f(v['low_median'], '.2f')} "
                         f"(range {_f(v['low_range'][0], '.2f')} to {_f(v['low_range'][1], '.2f')}); explicit more accurate "
                         f"in {v['n_explicit_better']} cells" + (f" ({'; '.join(v['explicit_better_cells'][:6])})"
                                                                  if v["explicit_better_cells"] else ""))
            L.append("")
    if "cost" in S:
        c = S["cost"]
        L += ["## Cost", "",
              f"- measured (this run): {c['mini_measured']['specs']} specs, "
              f"{c['mini_measured']['core_seconds'] / 3600:.3f} core-hours of simulate time, median "
              f"{c['mini_measured']['wall_per_step_median_us']:.0f} us per step (batch of 36 trajectories), max "
              f"peak RSS {c['mini_measured']['peak_rss_mb_max']:.0f} MB per task",
              f"- full-scale extrapolation: {c['full_estimate']['specs']} specs, "
              f"{c['full_estimate']['simulate_steps']:.3g} steps, about "
              f"{c['full_estimate']['core_hours']:.2f} core-hours ({c['full_estimate']['note']})", ""]
    L += ["## Interpretation", ""] + S.get("interpretation", [])
    (res / "SUMMARY.md").write_text("\n".join(L) + "\n")


def interpret(S):
    out = []
    ti = S.get("ti", {}).get("headline", {})
    if ti:
        never = [a for a, e in ti.items() if e["explicit_euler_exact_reached"].startswith("0/")]
        out.append(f"- Time-invariant problem: explicit Euler never reaches 1e-13 for {', '.join(never)} "
                   f"(it chatters on a period-2 orbit or diverges), while every consistent Euler scheme and both "
                   "error-memory TZ schemes reach 1e-13 in a finite number of steps (exactly 0.0 for the Euler-based "
                   "maps; resolvent and semi-implicit maps of continuous activations converge super-linearly and "
                   "need a few more steps); the exact-flow counts match the settling-time prediction ceil(T(e0)/h).")
        parts = []
        for a, e in ti.items():
            cn, cv, ra = (e["explicit_euler_chatter_norm"], e["explicit_euler_chatter_vec"],
                          e["explicit_euler_rate_alt"])
            if cn is not None and cv is not None and np.isfinite(cv):
                parts.append(f"{a}: norm-based {cn:.2g} vs chatter_vec {cv:.2g} (rate_alt {ra:.2g})")
        if parts:
            out.append("- The norm-based chattering index misses the e -> -e cycles of explicit Euler on the "
                       "time-invariant problem (exactly 0 when the cycle is symmetric, e.g. SBP, whose fixed "
                       "amplitude e* = (ah/2)^(1/(1-p)) gives a constant norm; strongly underestimated for sign, whose "
                       "components alternate between a and h - a), while chatter_vec and rate_alt expose it: "
                       + "; ".join(parts) + ".")
        dp = S.get("ti", {}).get("explicit_div_prediction")
        if dp:
            out.append(f"- Explicit-Euler divergence is an initial-error effect: iterating e <- e - h Phi(e) from each "
                       f"E(0) predicts the observed divergences in {100 * dp['agreement']:.1f}% of cases, so the "
                       "explicit 'stable h range' is conditional on the initial-state distribution (blow-up when "
                       "h Phi(s)/s > 2 on the superlinear branch), whereas the consistent maps are non-expansive "
                       "for every h and e0.")
    tv = S.get("tv", {})
    if tv:
        h1 = [e for e in tv["headline"] if e["num"] == "euler" and e["den"] == "exact_flow_euler"
              and e["metric"] == "mssre" and e.get("kind") == "matched_h" and np.isfinite(e["median_log10_ratio"])]
        h1all = [e for e in tv["headline"] if e["num"] == "euler" and e["den"] == "exact_flow_euler"
                 and e["metric"] == "mssre" and e.get("kind") == "matched_h"]
        if h1all:
            txt = "; ".join(f"{e['problem']}/{e['act']}: " + ("explicit diverged (%d/%d)" % (e["num_diverged"], e["n_pairs"])
                                                              if not np.isfinite(e["median_log10_ratio"])
                                                              else f"{e['median_log10_ratio']:.2f}")
                            for e in h1all)
            out.append(f"- Time-varying problems at matched gain h = {H_REF:g}, error in the PROBLEM'S OWN UNIT (s = 1; see the "
                       "error-unit lines below), median paired log10(explicit Euler / "
                       f"exact-flow Euler): {txt}. A large gain appears only for the sign activation and for "
                       "continuous activations whose exact map has a dead zone wider than the per-step drift; for "
                       "SBP r = 0.75 the dead zone (h/8)^4 is far below the drift at h = 0.3 and explicit Euler is "
                       "as good or slightly better.")
        dzs = tv.get("dead_zone", {})
        if dzs:
            gi = {round(h, 12): i for i, h in enumerate(H_GRID)}
            agree = [k for k, v in dzs.items() if v["h_deadzone"] is not None and v["h_flat_1.25"] is not None
                     and abs(gi[round(v["h_deadzone"], 12)] - gi[round(v["h_flat_1.25"], 12)]) <= 1]
            off = [k for k in dzs if k not in agree]
            out.append("- Mechanism: the consistent floor equals one local truncation error, independent of h and of "
                       "the activation, only where the exact map's dead zone delta(h) = T^{-1}(h) exceeds the drift "
                       f"per step; the predicted threshold h_dz matches the observed onset of the flat floor within "
                       f"one grid step in {len(agree)}/{len(dzs)} (problem, activation) cells"
                       + (f" (off: {', '.join(off)}, where the floor is already within 1.25x of its minimum "
                          "below h_dz)" if off else "")
                       + " (sign: whole grid; continuous activations: only above h_dz).")
        hb = [e for e in tv["headline"] if e.get("kind") == "best_h" and e["num"] == "euler"
              and np.isfinite(e["median_log10_ratio"])]
        if hb:
            txt = "; ".join(f"{e['problem']}/{e['act']}: {e['median_log10_ratio']:.2f} decades at h = {e['h']:g} "
                            f"(t_eps(1e-3) {_f(e.get('t_eps_1e-3_num'))} s vs {_f(e.get('t_eps_1e-3_den'))} s)"
                            for e in hb)
            out.append("- Unequal-gain check (explicit Euler at the grid h that minimises its own MSSRE, which is the "
                       "smallest settled h of the grid, consistent scheme at the same h): " + txt + ". The matched-h "
                       "advantage is therefore h-dependent (see the advantage profile): for sign (floor O(h)) it "
                       "persists at every h, whereas for SBP and the predefined-time activation (period-2 floor "
                       "O(h^(1/(1-p))) = O(h^2)) it vanishes as h -> 0, where explicit and consistent maps are "
                       "both consistent with the flow and the residual is set by the same balance between the "
                       "per-step drift and the activation near 0. What the consistent schemes buy is h-independence: "
                       "the one-LTE floor for every h up to 10 (gamma up to 10/tau, transients of a few steps) and "
                       "no divergence for any initial error.")
        h3 = [e for e in tv["headline"] if e["num"] == NAIVE and e["metric"] == "mssre"
              and np.isfinite(e["median_log10_ratio"])]
        if h3:
            big = [e for e in h3 if e["median_log10_ratio"] >= 1]
            small = [e for e in h3 if e["median_log10_ratio"] < 1]
            out.append("- Error memory is essential wherever the error map has a dead zone wider than the drift: the "
                       "naive implicit TZ is "
                       + (f"{min(e['median_log10_ratio'] for e in big):.1f} to "
                          f"{max(e['median_log10_ratio'] for e in big):.1f} decades worse than TZ-EM at the same h "
                          f"({len(big)} problem/activation cells)" if big else "not worse")
                       + ("; the exceptions (" + ", ".join(f"{e['problem']}/{e['act']}: {e['median_log10_ratio']:.2f}"
                                                         for e in small)
                          + ") are activations whose map acts almost linearly near 0 at this h (tiny dead zone), "
                          "where the naive combination keeps the order of the linear case." if small else "."))
        o = tv.get("orders", {})
        cons_e = ("implicit_euler_err", "peb_euler", "exact_flow_euler", "semi_implicit_euler")
        okf = lambda fo: np.isfinite(fo["slope_asym"]) and fo.get("n_asym", 0) >= 3  # noqa: E731
        se = [fo["slope_asym"] for k, fo in o.items() if k.split("|")[2] in cons_e and okf(fo)]
        st = [fo["slope_asym"] for k, fo in o.items() if k.split("|")[2] in ("implicit_taylor_zhang", "peb_taylor_zhang")
              and okf(fo)]
        sx = [fo["slope_asym"] for k, fo in o.items() if k.split("|")[2] == "euler" and k.split("|")[1] == "sign"
              and okf(fo)]
        n_short = sum(1 for fo in o.values() if np.isfinite(fo["slope_asym"]) and fo.get("n_asym", 0) < 3)
        if se and st:
            out.append(f"- Orders at fixed h (fits with >= 3 tau points in the asymptotic range): consistent Euler "
                       f"schemes {min(se):.2f} to {max(se):.2f} (formula order 2), TZ-EM schemes {min(st):.2f} to "
                       f"{max(st):.2f} (order 3)"
                       + (f"; explicit sign-Euler {min(sx):.2f} to {max(sx):.2f} (O(h) floor, no convergence in tau)"
                          if sx else "")
                       + (f". {n_short} fits rest on 2 points only and are not interpreted (see the tables)."
                          if n_short else "."))
        fg = tv.get("orders_fixed_gamma", {})
        g = GAMMA_DIAG[0]
        def _sl(d, a=None):  # noqa: E306
            return [(k.split("|")[0] + "/" + k.split("|")[1], v["slope_asym"]) for k, v in fg.items()
                    if k.split("|")[2] == d and k.endswith(f"|{g:g}") and (a is None or k.split("|")[1] == a)
                    and np.isfinite(v["slope_asym"]) and v.get("n_asym", 0) >= 3]
        xs_, xb_, ce_ = _sl("euler", "sign"), _sl("euler", "sbp0.5"), _sl("exact_flow_euler")
        if xs_ and ce_:
            out.append(f"- Fixed gamma = {g:g} (continuous-time convention, h = gamma tau -> 0; fits with >= 3 points): "
                       f"explicit sign-Euler slope {min(v for _, v in xs_):.2f} to {max(v for _, v in xs_):.2f} "
                       "(floor O(gamma tau)), explicit SBP(0.5)-Euler "
                       + (f"{min(v for _, v in xb_):.2f} to {max(v for _, v in xb_):.2f}" if xb_ else "n/a")
                       + " (period-2 floor (a gamma tau/2)^2 = O(tau^2)), exact-flow Euler "
                       + ", ".join(f"{k}: {v:.2f}" for k, v in ce_)
                       + ". Under the continuous-time convention the explicit SBP floor has the same order as the "
                       "consistent scheme, so only the constant and the h-independence separate them; as h -> 0 the "
                       "dead zone of continuous activations shrinks below the drift and the consistent order can drop.")
        kd = [k for k in tv.get("kendall", []) if np.isfinite(k.get("tau_b_tied", math.nan))]
        if kd:
            lo = min(kd, key=lambda k: k["tau_b_tied"])
            out.append(f"- Scheme rankings: tie-tolerant Kendall tau_b {lo['tau_b_tied']:.2f} to "
                       f"{max(k['tau_b_tied'] for k in kd):.2f} (raw {min(k['tau_b'] for k in kd if np.isfinite(k['tau_b'])):.2f} "
                       f"to {max(k['tau_b'] for k in kd if np.isfinite(k['tau_b'])):.2f}); lowest for {lo['act']}, "
                       f"{lo['what']}. High agreement is driven by the explicit / consistent-Euler / TZ-EM separation, "
                       "the schemes within a class being tied; with 9 schemes and nested classes the p-values are "
                       "descriptive only.")
    sc = S.get("scale") or {}
    if sc.get("ti") or sc.get("tv"):
        ti_s, tv_s = sc.get("ti", {}), sc.get("tv", {})
        agr = [e["theory_agree_exact"] for e in ti_s.get("cells", {}).values() if "theory_agree_exact" in e]
        sb = {k.split("|")[1]: v["bounded_hmax"] for k, v in ti_s.get("explicit_div", {}).items()
              if k.startswith("sbp0.5|")}
        coll = tv_s.get("collapse", {})
        ee = {k.split("|")[0]: v for k, v in coll.items() if k.endswith("|euler/exact_flow_euler")}
        low = [v for k, v in coll.items() if k.split("|")[1] in ("euler/exact_flow_euler", "euler/implicit_euler_err")
               and v["n_low"]]
        out.append("- ERROR UNIT (section d). Every non-linear activation here lacks a length scale, so with E expressed "
                   "in units of 1/s the error ODE becomes E' = -gamma Phi(sE)/s. Scale-free (hold for every s in "
                   f"{SCALE_ORDER}): (i) no consistent trajectory diverged ({ti_s.get('consistent_n_diverged', '?')}/"
                   f"{ti_s.get('consistent_n', '?')} on the time-invariant problem, every h); (ii) the exact-flow maps "
                   "reach exact zero in ceil(T(s e0)/h) steps "
                   f"({100 * float(np.mean(agr)) if agr else float('nan'):.0f}% exact agreement, beyond-horizon cases "
                   "included), which for the fixed-/predefined-time activations never exceeds the s-free bound "
                   "ceil(T_max/h), while explicit Euler never converges; (iii) the explicit band is exactly the "
                   "2-cycle e*(h) in the unit of E_s, i.e. sqrt(m) e*(h)/s in the problem unit. Unit-dependent: the "
                   "explicit-Euler bounded range of h (SBP r = 0.5: "
                   + ", ".join(f"s = {k}: h <= {_f(v, 'g', 'none')}" for k, v in sb.items())
                   + ") and the size of the accuracy gain.")
        if ee:
            out.append("- On the time-varying problem the matched-h accuracy gain of the consistent scheme is a single "
                       "function of the band-to-drift ratio rho = sqrt(m) delta(h)/(s F(tau)), gain ~ log10(1 + 10^c rho), "
                       "with ONE constant for all four units: rms "
                       + ", ".join(f"{a} {v['rms']:.2f}" for a, v in ee.items())
                       + " decades, against "
                       + ", ".join(f"{v['rms_unit_naive']:.2f}" for v in ee.values())
                       + " if the unit is ignored. Hence '3-4.5 decades more accurate' is a statement about rho >> 1 "
                       "(the problem's unit at h = 0.3), not about the scheme; for rho <~ 1 (finer unit, smaller h, "
                       "larger tau) explicit Euler is as accurate"
                       + (" or marginally more accurate (by " + ", ".join(
                           f"{-v['low_range'][0]:.2f}" for v in low) + " decades at most; the robotics reversal)"
                          if low else "")
                       + ". The robust benefit of the consistent schemes is boundedness and finite-step exactness, "
                         "not accuracy at a given unit.")
        nv = [v for k, v in coll.items() if k.endswith(f"|{NAIVE}/implicit_taylor_zhang")]
        # naive-TZ failures NOT explained by the dead zone: rho <~ 3, settled, naive >= 10x worse
        bad = [v for v in tv_s.get("cells", {}).values() if v["pair"] == f"{NAIVE}/implicit_taylor_zhang"
               and v["num_diverged"] == 0 and v["unsettled"] == 0 and np.isfinite(v["median_log10_ratio"])
               and v["median_log10_ratio"] > 1.0 and np.isfinite(v["rho"]) and 0 < v["rho"] <= 3.0]
        if nv:
            grp_ = {}
            for v in bad:
                grp_.setdefault((v["act"], v["scale"], v["tau"]), []).append(v)
            desc = "; ".join(f"{a} s={s:g} tau={t:g}: h {min(x['h'] for x in g):g}-{max(x['h'] for x in g):g}, "
                             f"naive MSSRE up to {max(x['num_median'] for x in g):.3g}"
                             for (a, s, t), g in sorted(grp_.items()))
            out.append("- The memory ablation (naive TZ / TZ-EM) is also unit-dependent but not a function of rho alone "
                       f"(collapse rms {min(v['rms'] for v in nv):.2f}-{max(v['rms'] for v in nv):.2f}). For rho >> 1 the "
                       "naive scheme fails through its marginally stable dead-zone recursion (consistent.md 4.1); in "
                       + (f"{len(bad)} settled cells with rho <= 3 it is nevertheless >= 1 decade worse ({desc}): the "
                          "naive recursion is poorly damped when the activation's effective gain is large (fine unit "
                          "and large h, superlinear branch), so its MSSRE is bounded but not contracting there, while "
                          "TZ-EM stays at the one-step floor. Elsewhere at rho <~ 1 the two coincide."
                          if bad else "no settled cell with rho <= 3 is the naive scheme >= 1 decade worse."))
    if not out:
        out.append("- (no results found)")
    return out


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", required=True, help="folder holding E8_ti/, E8_tv/ (and E8_formulas/)")
    ap.add_argument("--figdir", required=True)
    ap.add_argument("--tabdir", required=True)
    args = ap.parse_args(argv)
    Path(args.figdir).mkdir(parents=True, exist_ok=True)
    Path(args.tabdir).mkdir(parents=True, exist_ok=True)
    data = load(args.results)
    if not data:
        print(f"[E8] no results under {args.results}")
        return 1
    S = {"protocol": {"h_grid": H_GRID, "h_ref": H_REF, "eps_exact": EPS_EXACT, "roundoff": ROUNDOFF,
                      "contract": CONTRACT, "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
                      "rows": {k: int(len(v)) for k, v in data.items()},
                      "status_counts": {k: v["status"].value_counts().to_dict() for k, v in data.items()}}}
    S["protocol"]["n_instances"] = int(max(v["seed"].nunique() for v in data.values()))
    S["protocol"]["n_failed"] = {k: int(v["failed_b"].sum()) for k, v in data.items() if v["failed_b"].any()}
    E0s, E0s_tv = {}, {}
    for name in ("E8_ti", "E8_tv") + SCALE_CFGS:
        if name in data:
            try:
                th, e0 = trajectory_theory(data[name])
            except Exception as exc:  # noqa: BLE001
                print(f"[E8] {name}: per-trajectory theory unavailable: {exc!r}")
                th, e0 = pd.DataFrame(), {}
            data[name] = add_theory(data[name], th)
            if name == "E8_ti":
                E0s = e0
            elif name == "E8_tv":
                E0s_tv = e0
            S["protocol"].setdefault("unsettled_rows", {})[name] = int(data[name]["unsettled"].sum())
    store = TraceStore(args.results)
    try:
        if "E8_ti" in data:
            analyze_ti(data["E8_ti"], store, args.figdir, args.tabdir, S, E0s=E0s)
            S["_ti_ranges"] = {f"{a}|{d}": h_ranges(data["E8_ti"], {"act": a, "disc_key": d})
                               for a in acts_present(data["E8_ti"]) for d in DISCS}
        if "E8_tv" in data:
            analyze_tv(data["E8_tv"], store, args.figdir, args.tabdir, S, E0s=E0s_tv)
        if "E8_formulas" in data:
            analyze_formulas(data["E8_formulas"], args.figdir, args.tabdir, S)
        if any(k in data for k in SCALE_CFGS):
            analyze_scale(data, E0s, args.figdir, args.tabdir, S)
        cost_estimate(data, S)
    finally:
        store.close()
    S["ti_ranges"] = S.pop("_ti_ranges", {})
    S["interpretation"] = interpret(S)
    write_summary(args.results, S, data)
    print(f"[E8] figures -> {args.figdir}, tables -> {args.tabdir}, summary -> {Path(args.results) / 'SUMMARY.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
