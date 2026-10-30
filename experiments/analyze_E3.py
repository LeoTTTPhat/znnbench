#!/usr/bin/env python
"""E3 analysis: convergence-time audit of finite / fixed / predefined-time claims.

    python experiments/analyze_E3.py --results results/E3 --figdir figures --tabdir results/tables
    python experiments/analyze_E3.py --results results/mini/E3 --figdir figures/mini --tabdir results/tables/mini

Inputs (built by experiments/configs/E3.py): ``<results>/ct`` (custom CT
reference task), ``<results>/E3_disc`` and ``<results>/E3_tc_<Tc>`` (runner
sweeps).  Outputs:

* figures (``<figdir>/fig_E3_*.pdf|png``):
  ``fig_E3_conservatism``  bound / measured CT time-to-1e-6 (bars, nominal e0)
                           and its dependence on the initial-error size (heatmap);
  ``fig_E3_time_to_eps_vs_tau``  discrete time-to-eps over the CT reference
                           time vs tau (fixed gamma Euler / RK4, fixed h Euler);
  ``fig_E3_floor_vs_tau``  steady-state residual floor vs tau with the eps
                           levels (eps below the floor is never reached);
  ``fig_E3_tc_breakdown``  time-to-eps / T_c and residual left at T_c vs T_c/tau;
  ``fig_E3_tvg_cap``       1/(t_c - t) gain: re-amplification and divergence
                           with and without the saturation cap vs base gain;
  ``fig_E3_tvg_traces``    example residual traces around t_c.
* LaTeX booktabs tables (``<tabdir>/E3_*.tex``; ``E3_units.tex``: unit-dependence arm,
  ``E3_vp.tex``: varying-parameter gain with / without cap).
* ``<results>/summary.json`` and ``<results>/SUMMARY.md`` (headline numbers).

Statistics: medians and IQR over the 3 problem seeds x B = 4 initial states
(+inf = never reached / diverged is kept as the worst value, never dropped);
CT headline CIs are CLUSTER bootstrap 95 % CIs (2000 resamples, seed 20261018;
resample seeds, then initial states within the seed); divergence outcomes of
the discrete T_c sweeps are deterministic per (model, gamma, T_c, tau) cell and
are reported as cell counts, not as trajectory-level CIs; Kendall tau_b for
rankings (p-values descriptive only: n = 5-15 items, no multiplicity control);
empirical orders fitted only on the asymptotic range (consecutive local slopes
within 0.15, values below the 1e-12 round-off floor excluded), with the
all-points slope, the local slopes and a 2-point flag reported as well.
Unit-dependence arm (review 2026-10-18): runs with the error expressed as
E_s = s E (experiments/tasks/E3_units.py), s in {1e-2, 1e2}; column ``unit``.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "experiments", ROOT / "experiments" / "tasks"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

import keys  # noqa: E402
import style  # noqa: E402

BOOT_N, BOOT_SEED = 2000, 20261018
EPS = (1e-2, 1e-3, 1e-6)
BIG = 1e300
ROUNDOFF = 1e-12

CLASS_ORDER = {"finite": 0, "fixed": 2, "predefined": 3, "strictly-predefined": 4}
ENTRY_ORDER = ["sign", "sign_power", "saturated_sign_power", "sfaf", "sign_bi_power", "wsbp",
               "fixed_time", "vaf", "pvaf1", "pvaf2", "predefined_exp", "strict_predefined",
               "sptc_nt_znn", "sptc_an_foznn"]
TC_CONST = ["vaf", "pvaf1", "pvaf2", "predefined_exp", "fixed_time", "sign_bi_power"]
TC_TV = ["strict_predefined", "sptc_nt_znn", "sptc_nt_znn+cap", "sptc_an_foznn", "sptc_an_foznn+cap"]
# 1/(t_c - t) gain of sptc_nt_znn: uncapped and capped at rho/(gamma tau) (h_eff <= rho):
# rho = 1 deadbeat, 1.9 stable non-deadbeat, 3 outside the Euler interval (0, 2) (control)
TVG_LABELS = [("sptc_nt_znn", "no cap"), ("sptc_nt_znn+cap", "cap $\\rho=1$ (deadbeat)"),
              ("sptc_nt_znn+cap1.9", "cap $\\rho=1.9$"), ("sptc_nt_znn+cap3", "cap $\\rho=3$ (control)")]
TC_OUTCOMES = ("ok", "late", "floor", "diverged")
BOUND_TYPES = [("bound_source", "bound stated by the source"),
               ("bound_uniform", "uniform bound (derived)"),
               ("bound_tight", "tight uniform bound (derived)")]


# =============================================================================
# small statistics helpers
# =============================================================================
def _fin(x):
    x = np.asarray(x, dtype=float)
    return np.where(np.isposinf(x), BIG, x)


def q(x, p):
    """Quantile that treats +inf as a value (never-reached = slower than everything)."""
    x = _fin(x)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return float("nan")
    v = float(np.quantile(x, p))
    return math.inf if v >= BIG / 10 else v


def med(x):
    return q(x, 0.5)


def iqr(x):
    return q(x, 0.25), q(x, 0.75)


def boot_ci(x, stat=None, n=BOOT_N, seed=BOOT_SEED):
    """Percentile bootstrap 95 % CI of a statistic (default: median)."""
    x = _fin(x)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return (float("nan"), float("nan"))
    stat = stat or (lambda a: np.quantile(a, 0.5, axis=-1))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n, x.size))
    s = stat(x[idx])
    lo, hi = np.quantile(s, [0.025, 0.975])
    f = (lambda v: math.inf if v >= BIG / 10 else float(v))
    return f(lo), f(hi)


def frac_ci(flags, n=BOOT_N, seed=BOOT_SEED):
    flags = np.asarray(flags, dtype=float)
    if flags.size == 0:
        return float("nan"), (float("nan"), float("nan"))
    return float(flags.mean()), boot_ci(flags, stat=lambda a: a.mean(axis=-1), n=n, seed=seed)


def boot_ci_cluster(x, clusters, n=BOOT_N, seed=BOOT_SEED):
    """Two-stage cluster bootstrap 95 % CI of the median: resample clusters
    (problem seeds) with replacement, then members within each drawn cluster.
    +inf is kept as the worst value (BIG)."""
    x = _fin(x)
    c = np.asarray(clusters)
    ok = ~np.isnan(x)
    x, c = x[ok], c[ok]
    if x.size == 0:
        return (float("nan"), float("nan"))
    groups = [x[c == u] for u in np.unique(c)]
    rng = np.random.default_rng(seed)
    st = np.empty(n)
    for i in range(n):
        gi = rng.integers(0, len(groups), len(groups))
        smp = np.concatenate([groups[j][rng.integers(0, groups[j].size, groups[j].size)] for j in gi])
        st[i] = np.quantile(smp, 0.5)
    lo, hi = np.quantile(st, [0.025, 0.975])
    f = (lambda v: math.inf if v >= BIG / 10 else float(v))
    return f(lo), f(hi)


def unit_of(w):
    """Error-unit scale s of a run (1.0 = natural unit) from the ``wrappers`` JSON column."""
    try:
        lst = json.loads(w) if isinstance(w, str) else (w or [])
    except (TypeError, ValueError):
        return 1.0
    if isinstance(lst, dict):
        lst = [lst]
    for x in lst or []:
        if isinstance(x, dict) and str(x.get("key", "")).endswith("ScaledError"):
            return float((x.get("kwargs") or {}).get("scale", 1.0))
    return 1.0


def kendall_b(a, b):
    from scipy.stats import kendalltau
    a, b = _fin(a), _fin(b)
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() < 3:
        return float("nan"), float("nan")
    r = kendalltau(a[ok], b[ok], variant="b")
    return float(r.statistic), float(r.pvalue)


def fit_order(xs, ys, floor=ROUNDOFF, tol=0.15):
    """Empirical order of ys ~ xs^p.

    Points that are non-finite or below the round-off floor are dropped.  The
    local slopes s_i = dlog y / dlog x between consecutive points are the
    consecutive-ratio diagnostic; the asymptotic range is the longest run at
    the small-x end whose local slopes stay within ``tol`` of the last one
    (at least 2 points).  Returns dict(slope, local, n_used, x_used)."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    o = np.argsort(xs)
    xs, ys = xs[o], ys[o]
    ok = np.isfinite(ys) & (ys > floor)
    xs, ys = xs[ok], ys[ok]
    out = {"slope": float("nan"), "local": [], "n_used": int(xs.size), "x_used": xs.tolist(),
           "slope_all": float("nan"), "flag": ""}
    if xs.size < 2:
        out["flag"] = "<2 points"
        return out
    lx, ly = np.log(xs), np.log(ys)
    loc = np.diff(ly) / np.diff(lx)
    out["local"] = [float(v) for v in loc]
    out["slope_all"] = float(np.polyfit(lx, ly, 1)[0])
    # plateau at the small-x end (floor reached: round-off, horizon-limited transient,
    # tau-independent chattering) while larger x show a clear slope -> not asymptotic
    i0 = 0
    while i0 < loc.size - 1 and (
            (abs(loc[i0]) < 0.25 and np.any(np.abs(loc[i0 + 1:]) > 0.5))          # flat floor
            or (ys[i0] < 1e3 * floor and loc[i0] < 0.5 * np.max(loc[i0 + 1:]))):  # bending onto round-off
        i0 += 1
    if i0:
        out["plateau_points_dropped"] = int(i0)
        xs, ys, lx, ly, loc = xs[i0:], ys[i0:], lx[i0:], ly[i0:], loc[i0:]
    k = 1                                   # number of local slopes in the asymptotic tail
    while k < loc.size and abs(loc[k] - loc[0]) <= tol and all(abs(loc[j] - loc[0]) <= tol for j in range(k)):
        k += 1
    # the tail is at the SMALL-x end, i.e. the first local slopes after sorting
    sel = slice(0, k + 1)
    p = np.polyfit(lx[sel], ly[sel], 1)
    out.update({"slope": float(p[0]), "n_used": int(k + 1), "x_used": xs[sel].tolist()})
    flags = []
    if k + 1 == 2:
        flags.append("2-point fit")
    if abs(p[0]) > 7:
        flags.append("implausible slope (> q+3)")
    if i0:
        flags.append(f"{i0} floor point(s) dropped")
    out["flag"] = "; ".join(flags)
    return out


def fmt(v, nd=3):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "--"
    if isinstance(v, float) and math.isinf(v):
        return "$\\infty$"
    if abs(v) >= 1e4 or (abs(v) < 1e-2 and v != 0):
        m, e = f"{v:.{max(nd - 1, 1)}e}".split("e")
        return f"${m}\\times10^{{{int(e)}}}$"
    return f"{v:.{nd}g}"


def tex_name(s):
    return "\\texttt{" + str(s).replace("_", "\\_").replace("+", "{+}") + "}"


def write_table(path, colspec, header, rows, caption, label, notes=None):
    lines = ["% generated by experiments/analyze_E3.py -- do not edit by hand",
             "\\begin{table}[t]", "\\centering", "\\small", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    for r in rows:
        if r == "MIDRULE":
            lines.append("\\midrule")
        else:
            lines.append(" & ".join(r) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    if notes:
        lines.append(f"\\par\\smallskip\\footnotesize {notes}")
    lines.append("\\end{table}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n")


def _save(fig, figdir, name):
    style.FIG_DIR = Path(figdir)
    return style.savefig(fig, name)


# =============================================================================
# loading
# =============================================================================
def load_ct(res):
    d = res / "ct"
    tf = d / "tasks.jsonl"
    if not tf.exists():
        return pd.DataFrame()
    files = []
    for line in tf.read_text().splitlines():
        if line.strip():
            p = Path(json.loads(line)["done"])
            if not p.exists():
                p = d / "results" / p.name
            if p.exists():
                files.append(p)
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def load_runner(path):
    from znnbench.runner import aggregate
    if not (path / "tasks.jsonl").exists():
        return pd.DataFrame()
    df = aggregate(path)
    if not df.empty:
        df["unit"] = df["wrappers"].map(unit_of) if "wrappers" in df else 1.0
    return df


def tc_of_config(name):
    return float(name.split("E3_tc_")[1].replace("p", "."))


# =============================================================================
# (a) continuous-time conservatism
# =============================================================================
def entry_sort_key(row_or_label, ct=None):
    lab = row_or_label
    base = lab.split("(")[0]
    i = ENTRY_ORDER.index(base) if base in ENTRY_ORDER else len(ENTRY_ORDER)
    return (i, lab)


def ct_headline_gamma(df_entry):
    """gamma used for the headline: 1 for time-invariant activations (ratios are
    gamma-invariant), 2 (the SPTC papers' value) for the strict models."""
    tv = bool(df_entry["tv"].iloc[0])
    gs = sorted(df_entry["gamma"].unique())
    want = 2.0 if tv else 1.0
    return want if want in gs else gs[0]


def analyze_ct(ct, figdir, tabdir, summ):
    if ct.empty:
        summ["ct"] = {"status": "no data"}
        return {}
    ct = ct.copy()
    t6 = ct["t_ref_1e-06"]
    for b, _ in BOUND_TYPES:
        ct["ratio_" + b] = ct[b] / t6
        ct["valid_" + b] = ~(ct[b] < ct["t_settle"] * (1 - 1e-9))     # NaN bound counts as valid (no claim)
    ct["tail"] = ct["t_settle"] / t6
    out = {"entries": {}}
    labels = sorted(ct["entry"].unique(), key=entry_sort_key)
    nat = ct[ct["e0_scale"] == "nat"]
    rows_tex = []
    for lab in labels:
        de = nat[nat["entry"] == lab]
        if de.empty:
            continue
        g = ct_headline_gamma(de)
        de = de[de["gamma"] == g]
        rec = {"gamma": g, "claimed_class": de["claimed_class"].iloc[0], "derived_class": de["derived_class"].iloc[0],
               "n": int(len(de)), "t_ref_1e-06_median": med(de["t_ref_1e-06"]),
               "t_ref_1e-06_iqr": iqr(de["t_ref_1e-06"]), "t_ref_1e-03_median": med(de["t_ref_0.001"]),
               "t_settle_median": med(de["t_settle"]), "e0_max_median": med(de["e0_max"])}
        for b, _ in BOUND_TYPES:
            r = de["ratio_" + b]
            if r.notna().any():
                rec[b + "_median"] = med(de[b])
                rec["ratio_" + b] = {"median": med(r), "iqr": iqr(r),
                                     "ci95": boot_ci_cluster(r.values, de["seed"].values),
                                     "ci95_kind": "cluster bootstrap (seeds, then x0)",
                                     "n_clusters": int(de["seed"].nunique())}
        # validity of every bound over ALL scales / gammas of this entry
        da = ct[ct["entry"] == lab]
        rec["bounds_valid_all_rows"] = {b: bool(da["valid_" + b].all()) for b, _ in BOUND_TYPES}
        # e0 dependence of the best available stated bound
        bcol = "bound_source" if da["bound_source"].notna().any() else "bound_uniform"
        rec["e0_bound_used"] = bcol
        sc = {}
        for s, dd in da[da["gamma"] == g].groupby("e0_scale"):
            sc[s] = {"e0_max_median": med(dd["e0_max"]), "ratio_median": med(dd[bcol] / dd["t_ref_1e-06"])}
        rec["ratio_by_e0"] = sc
        out["entries"][lab] = rec
        rs = rec.get("ratio_bound_source", {})
        rt = rec.get("ratio_bound_tight", {})
        ru = rec.get("ratio_bound_uniform", {})
        rmax = max((v["ratio_median"] for v in sc.values() if np.isfinite(v["ratio_median"])), default=float("nan"))
        rmin = min((v["ratio_median"] for v in sc.values() if np.isfinite(v["ratio_median"])), default=float("nan"))
        rows_tex.append([tex_name(lab), rec["claimed_class"] + ("" if rec["claimed_class"] == rec["derived_class"]
                                                               else f" ({rec['derived_class']})"),
                         fmt(g, 3), fmt(rec["t_ref_1e-06_median"]),
                         fmt(rec.get("bound_source_median")),
                         (fmt(rs["median"]) + f" [{fmt(rs['ci95'][0])}, {fmt(rs['ci95'][1])}]") if rs else "--",
                         fmt(ru.get("median")) if ru else "--",
                         fmt(rt.get("median")) if rt else "--",
                         (f"{fmt(rmin)}--{fmt(rmax)}" if np.isfinite(rmin) else "--"),
                         "yes" if all(rec["bounds_valid_all_rows"].values()) else "\\textbf{no}"])
    write_table(Path(tabdir) / "E3_ct_conservatism.tex", "llrrrrrrrr",
                ["model", "class (claimed / derived)", "$\\gamma$", "$t_{10^{-6}}$ (s)", "bound (s)",
                 "ratio [95\\% CI]", "uniform", "tight", "ratio over $e_0$", "valid"],
                rows_tex,
                "E3(a): continuous-time conservatism of convergence-time bounds. $t_{10^{-6}}$: exact time to "
                "$\\|E\\|_2<10^{-6}$ (semi-analytic, matrix inversion $n=4$, natural $X(0)$, median of 12 "
                "trajectories); ratio = bound / $t_{10^{-6}}$ (median; 95\\% CI: cluster bootstrap over problem "
                "seeds, then initial states); uniform/tight: "
                "ZNN-Bench-derived bounds; last-but-one column: range of the median ratio over the "
                "initial-error scales (the initial-error scale is equivalent to the choice of unit of $E$ for "
                "these scale-free activations); valid: every bound $\\geq$ the exact settling time in every run.",
                "tab:E3_ct_conservatism")
    # numerical cross-checks
    fs = ct[ct["full_status"].isin(["eps", "component", "t_max"])]
    chk = {"bracket_ok_fraction": float(np.mean([ct[f"bracket_ok_{e:g}"].mean() for e in EPS
                                                 if f"bracket_ok_{e:g}" in ct])),
           "full_state_runs": int((ct["full_status"] != "not_run").sum()),
           "full_state_status_counts": ct["full_status"].value_counts().to_dict(),
           "full_state_max_rel_dev_norm": float(fs["full_maxrel_dev"].max()) if len(fs) else float("nan")}
    devs = []
    for e in EPS:
        c, r = f"t_full_{e:g}", f"t_ref_{e:g}"
        if c in ct:
            ok = ct[c].notna() & ct[r].notna() & (ct[r] > 0)
            devs += list((np.abs(ct.loc[ok, c] - ct.loc[ok, r]) / ct.loc[ok, r]).values)
    chk["full_state_max_rel_dev_t_eps"] = float(np.max(devs)) if devs else float("nan")
    chk["full_state_n_t_eps_compared"] = len(devs)
    gi = {}
    for lab, de in ct[(ct["e0_scale"] == "nat") & (~ct["tv"].astype(bool))].groupby("entry"):
        if de["gamma"].nunique() > 1:
            piv = de.pivot_table(index=["seed", "x0_index"], columns="gamma", values="t_ref_1e-06")
            gmin = piv.columns.min()
            dev = [float(np.max(np.abs(piv[c] * c / (piv[gmin] * gmin) - 1))) for c in piv.columns if c != gmin]
            gi[lab] = max(dev) if dev else float("nan")
    if gi:
        chk["gamma_scaling_max_rel_dev"] = float(np.nanmax(list(gi.values())))
    out["checks"] = chk
    # Kendall tau_b: does the stated bound rank the activations like the actual CT time?
    rk = [(lab, rec["bound_source_median"], rec["t_ref_1e-06_median"]) for lab, rec in out["entries"].items()
          if "bound_source_median" in rec and lab.split("(")[0] not in ("strict_predefined", "sptc_nt_znn",
                                                                          "sptc_an_foznn")]
    if len(rk) >= 3:
        tau_b, pv = kendall_b([r[1] for r in rk], [r[2] for r in rk])
        out["kendall_bound_vs_actual"] = {"tau_b": tau_b, "p": pv, "n": len(rk), "entries": [r[0] for r in rk]}
    summ["ct"] = out
    fig_conservatism(ct, out, labels, figdir)
    return out


def fig_conservatism(ct, out, labels, figdir):
    labs = [lab for lab in labels if lab in out["entries"] and any(
        "ratio_" + b in out["entries"][lab] for b, _ in BOUND_TYPES)]
    if not labs:
        return
    fig, (ax, axh) = plt.subplots(1, 2, figsize=(style.DOUBLE, 0.26 * len(labs) + 1.5),
                                  gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.55})
    hatches = ["", "////", "...."]
    ylab = []
    from matplotlib.patches import Patch
    from matplotlib.ticker import FuncFormatter
    xmax = 1.0
    for i, lab in enumerate(labs[::-1]):
        rec = out["entries"][lab]
        ylab.append(lab)
        shown = []
        for j, (b, name) in enumerate(BOUND_TYPES):
            r = rec.get("ratio_" + b)
            if not r:
                continue
            if b == "bound_uniform" and "ratio_bound_source" in rec and \
                    abs(r["median"] - rec["ratio_bound_source"]["median"]) <= 1e-9 * r["median"]:
                continue          # uniform bound identical to the stated one: not repeated
            shown.append((j, r))
        hgt = 0.78 / max(len(shown), 1)
        for k, (j, r) in enumerate(shown):
            y = i - (k - (len(shown) - 1) / 2) * hgt      # first bound type on top (legend order)
            lo, hi = r["iqr"]
            ax.barh(y, r["median"], height=hgt * 0.9, color=style.CAT[j], hatch=hatches[j],
                    edgecolor="white", linewidth=0.3)
            ax.errorbar(r["median"], y, xerr=[[r["median"] - lo], [hi - r["median"]]], fmt="none",
                        ecolor=style.INK2, elinewidth=0.6, capsize=1.0)
            ax.text(hi * 1.04, y, f"{r['median']:.2f}", va="center", ha="left", fontsize=5.2, color=style.INK2)
            xmax = max(xmax, hi)
    ax.set_xscale("log")
    ax.axvline(1.0, color=style.INK2, lw=0.7, ls="--")
    ax.set_yticks(range(len(labs)))
    ax.set_yticklabels(ylab)
    ax.set_ylim(-0.6, len(labs) - 0.4)
    ax.set_xlim(0.7, xmax * 1.8)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}" if v in (2, 3, 5, 20, 30, 50) else ""))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("bound / exact time to $\\|E\\|_2<10^{-6}$")
    ax.set_title("(a) conservatism at the natural initial error")
    ax.legend(handles=[Patch(facecolor=style.CAT[j], hatch=hatches[j], edgecolor="white", label=name)
                       for j, (_, name) in enumerate(BOUND_TYPES)], loc="lower right", fontsize=6)
    # (b) heatmap entry x e0 scale
    scales = sorted(ct["e0_scale"].unique(), key=lambda s: med(ct.loc[ct["e0_scale"] == s, "e0_max"]))
    M = np.full((len(labs), len(scales)), np.nan)
    rowlab = []
    for i, lab in enumerate(labs):
        rec = out["entries"][lab]
        for j, s in enumerate(scales):
            v = rec["ratio_by_e0"].get(s, {}).get("ratio_median", np.nan)
            M[i, j] = v if (v is not None and np.isfinite(v)) else np.nan
        rowlab.append(lab + ("$^\\dagger$" if rec["e0_bound_used"] != "bound_source" else ""))
    vmin = max(np.nanmin(M), 0.5) if np.isfinite(M).any() else 1
    vmax = max(np.nanmax(M), vmin * 10) if np.isfinite(M).any() else 10
    im = axh.imshow(M, cmap=style.CMAP_SEQ, norm=LogNorm(vmin=vmin, vmax=vmax), aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                v = M[i, j]
                txt = f"{v:.2g}" if v < 100 else f"{v:.0f}" if v < 1e4 else f"{v:.0e}"
                dark = np.log(v / vmin) / max(np.log(vmax / vmin), 1e-9) > 0.6
                axh.text(j, i, txt, ha="center", va="center", fontsize=5.5, color="white" if dark else style.INK)
    xt = []
    for s in scales:
        e = med(ct.loc[ct["e0_scale"] == s, "e0_max"])
        xt.append(f"nat.\n(~{e:.3g})" if s == "nat" else f"{float(s):g}")
    axh.set_xticks(range(len(scales)))
    axh.set_xticklabels(xt)
    axh.set_yticks(range(len(labs)))
    axh.set_yticklabels(rowlab)
    axh.set_xlabel("initial error $\\max_i |e_i(0)|$ (equivalently: unit of $E$)")
    axh.set_title("(b) ratio vs initial error ($^\\dagger$ derived bound)")
    axh.grid(False)
    cb = fig.colorbar(im, ax=axh, fraction=0.05, pad=0.03)
    cb.set_label("bound / time", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    _save(fig, figdir, "fig_E3_conservatism")


# =============================================================================
# (b1) discrete time-to-eps vs tau (E3_disc)
# =============================================================================
DISC_SETTINGS = [("euler", "fixed_gamma", 10.0, "Euler, fixed $\\gamma=10$"),
                 ("rk4", "fixed_gamma", 10.0, "RK4, fixed $\\gamma=10$"),
                 ("taylor_zhang", "fixed_h", 0.1, "Taylor-Zhang, fixed $h=0.1$"),
                 ("euler", "fixed_h", 0.1, "Euler, fixed $h=0.1$"),
                 ("euler", "fixed_h", 1.0, "Euler, fixed $h=1$")]


def _gain_match(df, mode, val):
    if mode == "fixed_gamma":
        return (df["gain_mode"] == mode) & np.isclose(df["gamma"], val)
    return (df["gain_mode"] == mode) & np.isclose(df["h"], val)


def attach_ct_reference(df, ct, label_col="model_label"):
    """t_CT(eps) of the same (activation, seed, x0) at the run's gamma
    (time-invariant activations: t_CT(gamma) = t_CT(1)/gamma)."""
    df = df.copy()
    ref = ct[(ct["e0_scale"] == "nat") & np.isclose(ct["gamma"], 1.0) & (~ct["tv"].astype(bool))]
    ref = ref.set_index(["entry", "seed", "x0_index"])
    for e in EPS:
        c = f"t_ref_{e:g}"
        if c not in ref:
            continue
        key = list(zip(df[label_col], df["seed"].astype(int), df["x0_index"].astype(int)))
        vals = ref[c].reindex(key).values
        df[f"t_ct_{e:g}"] = vals / df["gamma"].values
        df[f"ratio_{e:g}"] = df[f"t_eps_{e:g}"] / df[f"t_ct_{e:g}"]
    return df


def analyze_disc(disc, ct, figdir, tabdir, summ):
    if disc.empty:
        summ["disc"] = {"status": "no data"}
        return
    df = attach_ct_reference(disc, ct) if not ct.empty else disc.copy()
    acts = [a for a in keys.GROUPS["act"] if a in set(df["model_label"])]
    others = sorted(set(df["model_label"]) - set(acts))
    settings = [s for s in DISC_SETTINGS if (_gain_match(df, s[1], s[2]) & (df["disc_key"] == s[0])).any()]
    taus = sorted(df["tau"].unique())
    out = {"settings": {}, "n_rows": int(len(df)), "status_counts": df["status"].value_counts().to_dict()}
    for dk, mode, val, title in settings:
        sub = df[(df["disc_key"] == dk) & _gain_match(df, mode, val)]
        sres = {}
        for a in acts + others:
            sa = sub[sub["model_label"] == a]
            ares = {}
            for tau in taus:
                st = sa[np.isclose(sa["tau"], tau)]
                if st.empty:
                    continue
                r = {"n": int(len(st)), "diverged": int((st["status"] == "diverged").sum()),
                     "mssre_median": med(st["mssre"]), "res_min_ss_median": med(st["res_min_ss"])}
                r["n_failed"] = int((~st["status"].isin(["ok", "diverged"])).sum())
                for e in EPS:
                    te = st[f"t_eps_{e:g}"]
                    r[f"reached_{e:g}"] = float(np.isfinite(te).mean())
                    r[f"never_{e:g}"] = int((~np.isfinite(te)).sum())
                    r[f"t_eps_{e:g}_median"] = med(te)
                    if f"ratio_{e:g}" in st:
                        r[f"ratio_{e:g}_median"] = med(st[f"ratio_{e:g}"])
                        r[f"ratio_{e:g}_iqr"] = iqr(st[f"ratio_{e:g}"])
                ares[f"{tau:g}"] = r
            # empirical order of the floor (MSSRE) vs tau
            ys = [ares[f"{t:g}"]["mssre_median"] for t in taus if f"{t:g}" in ares]
            xs = [t for t in taus if f"{t:g}" in ares]
            ares["floor_order"] = fit_order(xs, ys)
            if "ratio_0.001" in sa:
                # convergence of t_eps to the CT value: |ratio - 1| vs tau
                xs2 = [t for t in taus if f"{t:g}" in ares and np.isfinite(ares[f"{t:g}"].get("ratio_0.001_median", np.nan))]
                ys2 = [abs(ares[f"{t:g}"]["ratio_0.001_median"] - 1) for t in xs2]
                ares["t_eps_convergence_order"] = fit_order(xs2, ys2, floor=1e-9)
            sres[a] = ares
        # Kendall tau_b: CT ranking vs discrete ranking of the activations (eps = 1e-3)
        kt = {}
        if "t_ct_0.001" in sub:
            for tau in taus:
                st = sub[np.isclose(sub["tau"], tau)]
                g = st.groupby("model_label")
                if g.ngroups >= 3:
                    a_ct = g["t_ct_0.001"].apply(med)
                    a_d = g["t_eps_0.001"].apply(med)
                    kt[f"{tau:g}"] = dict(zip(("tau_b", "p"), kendall_b(a_ct.values, a_d.values)))
        out["settings"][f"{dk}|{mode}|{val:g}"] = {"title": title, "activations": sres, "kendall_ct_vs_disc": kt}
    summ["disc"] = out
    fig_time_to_eps(df, settings, acts, taus, figdir)
    fig_floor(df, settings, acts, taus, figdir)
    # table: fixed-gamma Euler, ratio to CT for eps = 1e-3 and 1e-6
    key = "euler|fixed_gamma|10"
    if key in out["settings"]:
        rows = []
        for a in acts + others:
            ares = out["settings"][key]["activations"].get(a, {})
            row = [tex_name(a)]
            for e in (1e-3, 1e-6):
                for tau in taus:
                    r = ares.get(f"{tau:g}")
                    if not r:
                        row.append("--")
                        continue
                    n = r["n"]
                    k = int(round(r[f"reached_{e:g}"] * n))
                    v = r.get(f"ratio_{e:g}_median", float("nan"))
                    row.append(f"{fmt(v)} ({k}/{n})")
            fo = ares.get("floor_order", {})
            row.append(fmt(fo.get("slope")) + ("$^\\ddagger$" if "2-point" in fo.get("flag", "") else "")
                       + " [" + ", ".join(f"{s:.2f}" for s in fo.get("local", [])) + "]")
            rows.append(row)
        hdr = ["activation"] + [f"$\\varepsilon={eps_tex(e)}$, $\\tau={eps_tex(t)}$" for e in (1e-3, 1e-6) for t in taus] + \
              ["floor order [local]"]
        write_table(Path(tabdir) / "E3_disc_time_to_eps.tex", "l" + "r" * (2 * len(taus) + 1), hdr, rows,
                    "E3(b): discrete time-to-$\\varepsilon$ divided by the exact continuous-time time-to-"
                    "$\\varepsilon$ of the same initial state (Euler, fixed $\\gamma=10$, matrix inversion "
                    "$n=4$; median over 12 trajectories, in brackets the number that reached $\\varepsilon$; "
                    "$\\infty$ = never reached). Last column: empirical order of the MSSRE floor in $\\tau$ "
                    "(fit on the asymptotic range: consecutive local slopes within 0.15; local slopes in brackets; "
                    "$^\\ddagger$: 2-point fit, indicative only).",
                    "tab:E3_disc_time_to_eps")


class Band:
    """'Never reached / diverged' band above the numeric range of a log axis.

    Points are registered with ``add`` while plotting; ``finish`` then places
    one staggered row per series (row spacing ~7 % of the axis height), shades
    the band, labels it with an infinity sign and keeps numeric ticks below it."""

    def __init__(self, ax, n):
        self.ax, self.n, self.pending = ax, max(int(n), 1), []

    def add(self, xs, k, **mkw):
        if len(xs):
            self.pending.append((list(xs), k, mkw))

    def finish(self, ytop, bottom=None, label=None, row=0.07, inf_size=7.5, top_pad=0.5):
        """``label``: text written in an extra row at the top of the strip (what the strip means);
        ``row``: row spacing as a fraction of the decades shown; ``top_pad``: free rows above the top row."""
        from matplotlib.ticker import FixedLocator
        ax = self.ax
        if bottom is None:
            bottom = min(ax.get_ylim()[0], ytop / 10)
        D = max(math.log10(ytop / bottom), 1.0)
        f = 10 ** (row * D)
        n = self.n + (1 if label else 0)
        rows = [ytop * f ** (k + 1) for k in range(n)]
        lo, hi = ytop * f ** 0.5, ytop * f ** (n + top_pad)
        for xs, k, mkw in self.pending:
            ax.plot(xs, [rows[k]] * len(xs), linestyle="none", **mkw)
        ax.axhspan(lo, hi, color="#f0efec", zorder=0, lw=0)
        ax.set_ylim(bottom, hi)
        for minor in (False, True):
            t = [v for v in ax.yaxis.get_ticklocs(minor=minor) if bottom <= v <= ytop]
            (ax.yaxis.set_minor_locator if minor else ax.yaxis.set_major_locator)(FixedLocator(t))
        ax.text(-0.015, math.sqrt(lo * hi), "$\\infty$", transform=ax.get_yaxis_transform(), ha="right",
                va="center", fontsize=inf_size, color=style.INK2)
        if label:
            ax.text(0.015, rows[-1], label, transform=ax.get_yaxis_transform(), ha="left", va="center",
                    fontsize=7.5, color=style.INK2)


def jitter(x, k, n, f=1.09):
    return np.asarray(x, dtype=float) * f ** (k - (n - 1) / 2)


def eps_tex(e):
    return f"10^{{{int(round(math.log10(e)))}}}"


def fig_time_to_eps(df, settings, acts, taus, figdir):
    if "ratio_0.001" not in df or not settings:
        return
    eps_show = [1e-3, 1e-6]
    ncol = len(settings)
    fig, axes = plt.subplots(len(eps_show), ncol, figsize=(style.DOUBLE, 1.65 * len(eps_show) + 0.6),
                             sharex=True, sharey=True, squeeze=False)
    vals = []
    for e in eps_show:
        v = df[f"ratio_{e:g}"].replace([np.inf, -np.inf], np.nan).dropna()
        vals += list(v.values)
    lo = max(min(np.nanpercentile(vals, 1) if vals else 0.5, 0.5), 1e-3)
    hi = min(max(np.nanpercentile(vals, 99) if vals else 2, 2.0), 1e3)
    ytop = hi * 1.6
    for i, e in enumerate(eps_show):
        for j, (dk, mode, val, title) in enumerate(settings):
            ax = axes[i, j]
            band = Band(ax, len(acts))
            sub = df[(df["disc_key"] == dk) & _gain_match(df, mode, val)]
            for ka, a in enumerate(acts):
                sa = sub[sub["model_label"] == a]
                xs, ys, ylo, yhi, xn = [], [], [], [], []
                for tau in taus:
                    st = sa[np.isclose(sa["tau"], tau)][f"ratio_{e:g}"]
                    if st.empty:
                        continue
                    m = med(st)
                    if np.isfinite(m):
                        a25, a75 = iqr(st)
                        xs.append(tau)
                        ys.append(m)
                        ylo.append(m - a25)
                        yhi.append(min(a75, ytop) - m)
                    else:
                        xn.append(tau)
                stl = keys.style_in_group("act", a)
                if xs:
                    ax.errorbar(jitter(xs, ka, len(acts)), ys, yerr=[ylo, yhi], capsize=1.2, elinewidth=0.5,
                                markersize=3.2, **stl)
                band.add(jitter(xn, ka, len(acts)), ka, marker=stl["marker"], markerfacecolor="white",
                         markeredgecolor=stl["color"], markersize=3.2)
            ax.axhline(1.0, color=style.INK2, lw=0.6, ls="--", zorder=1)
            ax.set_xscale("log")
            ax.set_yscale("log")
            band.finish(ytop, bottom=lo / 1.5)
            if i == 0:
                ax.set_title(title, fontsize=7.5)
            if i == len(eps_show) - 1:
                ax.set_xlabel("sampling period $\\tau$ (s)")
            if j == 0:
                ax.set_ylabel(f"$t_\\varepsilon/t_\\varepsilon^{{CT}}$, $\\varepsilon={eps_tex(e)}$")
    handles = [plt.Line2D([], [], label=a, markersize=3.2, **keys.style_in_group("act", a)) for a in acts]
    handles.append(plt.Line2D([], [], linestyle="none", marker="o", markerfacecolor="white",
                              markeredgecolor=style.INK2, markersize=3.2,
                              label="$\\infty$: not reached by most trajectories"))
    fig.legend(handles=handles, loc="upper center", ncol=min(len(handles), 7),
               bbox_to_anchor=(0.5, 1.02), fontsize=6.3)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, figdir, "fig_E3_time_to_eps_vs_tau")


def fig_floor(df, settings, acts, taus, figdir):
    if not settings:
        return
    ncol = len(settings)
    fig, axes = plt.subplots(1, ncol, figsize=(style.DOUBLE, 2.3), sharex=True, sharey=True, squeeze=False)
    axes = axes[0]
    fin = df["mssre"].replace([np.inf, -np.inf], np.nan).dropna()
    fin = fin[fin > 0]
    ytop = float(fin.max()) * 3 if len(fin) else 1.0
    # shared y-range from the finite cell medians (set explicitly: with sharey the first
    # band.finish would otherwise freeze the autoscale before the later panels are drawn)
    meds = [med(st["mssre"]) for _, st in df.groupby(["disc_key", "gain_mode", "h", "model_label", "tau"])]
    meds = [m for m in meds if np.isfinite(m) and m > 0]
    ybot = min(meds) / 3 if meds else ytop / 1e6
    for j, (dk, mode, val, title) in enumerate(settings):
        ax = axes[j]
        band = Band(ax, len(acts))
        sub = df[(df["disc_key"] == dk) & _gain_match(df, mode, val)]
        for ka, a in enumerate(acts):
            sa = sub[sub["model_label"] == a]
            xs, ys, reached, xn = [], [], [], []
            for tau in taus:
                st = sa[np.isclose(sa["tau"], tau)]
                if st.empty:
                    continue
                m = med(st["mssre"])
                if not np.isfinite(m):          # >= 50 % diverged: censored, shown in the infinity band
                    xn.append(tau)
                    continue
                xs.append(tau)
                ys.append(m)
                reached.append(np.isfinite(st["t_eps_1e-06"]).mean() >= 0.5)
            stl = keys.style_in_group("act", a)
            band.add(jitter(xn, ka, len(acts)), ka, marker=stl["marker"], markersize=3.4,
                     markerfacecolor="white", markeredgecolor=stl["color"])
            if not xs:
                continue
            ax.plot(xs, ys, label=a, markersize=0, color=stl["color"], linestyle=stl["linestyle"])
            for x, y, rc in zip(xs, ys, reached):
                ax.plot([x], [y], linestyle="none", marker=stl["marker"], markersize=3.4,
                        markeredgecolor=stl["color"], markerfacecolor=stl["color"] if rc else "white")
        for e in (1e-3, 1e-6):
            ax.axhline(e, color=style.GREY, lw=0.7, ls=(0, (4, 2)), zorder=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        band.finish(ytop, bottom=ybot)
        ax.set_title(title, fontsize=7.5)
        ax.set_xlabel("$\\tau$ (s)")
        if j == 0:
            ax.set_ylabel("residual floor (MSSRE)")
    handles = [plt.Line2D([], [], label=a, markersize=3.4, **keys.style_in_group("act", a))
               for a in acts if a in set(df["model_label"])]
    handles.append(plt.Line2D([], [], linestyle="none", marker="o", markerfacecolor="white",
                              markeredgecolor=style.INK2, label="$\\varepsilon=10^{-6}$ never reached (median)"))
    handles.append(plt.Line2D([], [], color=style.GREY, lw=0.7, ls=(0, (4, 2)),
                              label="$\\varepsilon=10^{-3}$, $10^{-6}$"))
    handles.append(plt.Line2D([], [], linestyle="none", label="$\\infty$ band: median diverged"))
    fig.legend(handles=handles, loc="upper center", ncol=min(len(handles), 8), bbox_to_anchor=(0.5, 1.03),
               fontsize=6.3)
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    _save(fig, figdir, "fig_E3_floor_vs_tau")


# =============================================================================
# (b2) predefined T_c vs tau, time-varying gains (E3_tc_*)
# =============================================================================
def prepare_tc(tc):
    tc = tc.copy()
    tc["T_c"] = tc["config"].map(tc_of_config)
    tc["tc_over_tau"] = tc["T_c"] / tc["tau"]
    for e in EPS:
        tc[f"teps_over_tc_{e:g}"] = tc[f"t_eps_{e:g}"] / tc["T_c"]
        tc[f"thit_over_tc_{e:g}"] = tc[f"t_hit_{e:g}"] / tc["T_c"]
    tc["rel_res_at_tc"] = tc["tc_r_at"] / tc["initial_residual"]
    tc["tc_jump"] = tc["tc_r_at"] / tc["tc_rmin_pre"]
    tc.loc[tc["status"] == "diverged", "tc_jump"] = np.where(
        tc.loc[tc["status"] == "diverged", "diverged_at"] <= tc.loc[tc["status"] == "diverged", "T_c"] * 1.0001,
        np.inf, tc.loc[tc["status"] == "diverged", "tc_jump"])
    tc.loc[tc["status"] == "diverged", "rel_res_at_tc"] = np.where(
        np.isfinite(tc.loc[tc["status"] == "diverged", "tc_r_at"]),
        tc.loc[tc["status"] == "diverged", "rel_res_at_tc"], np.inf)
    return tc


def _tc_series(tc):
    """(label, selector, style kwargs) of the series shown in the T_c figure."""
    out = []
    for a in TC_CONST:
        if a in set(tc["model_label"]):
            out.append(("const", a, tc["model_label"] == a, keys.style_in_group("act", a)))
    base = {"strict_predefined": "znn", "sptc_nt_znn": "sptc_nt_znn", "sptc_an_foznn": "sptc_an_foznn"}
    for lab in TC_TV:
        b = lab.replace("+cap", "")
        if lab not in set(tc["model_label"]):
            continue
        st = dict(keys.style_in_group("model", base[b]))
        g = 2.0
        sel = (tc["model_label"] == lab) & np.isclose(tc["gamma"], g)
        if lab.endswith("+cap"):
            st["linestyle"] = "--"
            st["markerfacecolor"] = "white"
        if not sel.any():
            continue
        name = {"strict_predefined": "znn + strict_predefined act."}.get(lab, lab)
        out.append(("tv", f"{name} ($\\gamma$=2)", sel, st))
    return out


def tc_outcome(df, e):
    """Per-trajectory outcome of the claim ||E(t)|| < eps for all t >= T_c:
    ok (t_eps <= T_c), late (T_c < t_eps < inf: reachable but not by T_c -- the
    time-limited breakdown), floor (never stays below eps although not diverged:
    eps below / inside the chattering band), diverged."""
    te = df[f"t_eps_{e:g}"].values.astype(float)
    tcv = df["T_c"].values.astype(float)
    div = (df["status"] == "diverged").values
    out = np.where(div, "diverged", np.where(te <= tcv * (1 + 1e-9), "ok",
                                              np.where(np.isfinite(te), "late", "floor")))
    return out


def analyze_tc(tc, figdir, tabdir, summ):
    if tc.empty:
        summ["tc"] = {"status": "no data"}
        return
    tc = prepare_tc(tc)
    ratios = sorted(tc["tc_over_tau"].round(6).unique())
    out = {"n_rows": int(len(tc)), "status_counts": tc["status"].value_counts().to_dict(), "series": {}}
    series = _tc_series(tc)
    for grp, name, sel, _ in series:
        s = tc[sel]
        rec = {}
        for r in ratios:
            sr = s[np.isclose(s["tc_over_tau"], r)]
            if sr.empty:
                continue
            d = {"n": int(len(sr)), "diverged": float((sr["status"] == "diverged").mean()),
                 "rel_res_at_tc_median": med(sr["rel_res_at_tc"]), "mssre_median": med(sr["mssre"])}
            for e in EPS:
                d[f"teps_over_tc_{e:g}_median"] = med(sr[f"teps_over_tc_{e:g}"])
                d[f"claim_ok_{e:g}"] = float((sr[f"t_eps_{e:g}"] <= sr["T_c"] * (1 + 1e-9)).mean())
                oc = tc_outcome(sr, e)
                d[f"outcome_{e:g}"] = {k: float(np.mean(oc == k)) for k in TC_OUTCOMES}
            rec[f"{r:g}"] = d
        # breakdown threshold: smallest T_c/tau from which the claim holds (>= 50 %) for every larger ratio
        thr = {}
        for e in EPS:
            ok = [(float(k), v[f"claim_ok_{e:g}"] >= 0.5) for k, v in rec.items()]
            ok.sort()
            t = None
            for k, flag in reversed(ok):
                if flag:
                    t = k
                else:
                    break
            thr[f"{e:g}"] = t
        # time-limited breakdown: largest T_c/tau at which >= 50 % are late or diverged
        # (separates "too few steps before T_c" from "eps below the chattering floor")
        brk, flo = {}, {}
        for e in EPS:
            bad = [float(k) for k, v in rec.items()
                   if v[f"outcome_{e:g}"]["late"] + v[f"outcome_{e:g}"]["diverged"] >= 0.5]
            brk[f"{e:g}"] = max(bad) if bad else None
            fl = [float(k) for k, v in rec.items() if v[f"outcome_{e:g}"]["floor"] >= 0.5]
            flo[f"{e:g}"] = max(fl) if fl else None
        out["series"][name] = {"by_ratio": rec, "claim_holds_from_tc_over_tau": thr,
                               "time_limited_breakdown_up_to": brk, "floor_limited_up_to": flo}
    fig_tc(tc, series, figdir)
    rows = []
    show = [r for r in ratios]
    for grp, name, sel, _ in series:
        rec = out["series"][name]["by_ratio"]
        row = [tex_name(name.split(" ($")[0]) + (" ($\\gamma=2$)" if grp == "tv" else "")]
        for r in show:
            d = rec.get(f"{r:g}")
            row.append("--" if d is None else fmt(d["teps_over_tc_0.001_median"], 2))
        thr = out["series"][name]["claim_holds_from_tc_over_tau"]["0.001"]
        row.append("never" if thr is None else f"{thr:.0f}")
        b = out["series"][name]["time_limited_breakdown_up_to"]["0.01"]
        row.append("none" if b is None else f"$\\leq${b:.0f}")
        f = out["series"][name]["floor_limited_up_to"]["0.001"]
        row.append("none" if f is None else f"$\\leq${f:.0f}")
        rows.append(row)
    write_table(Path(tabdir) / "E3_tc_breakdown.tex", "l" + "r" * (len(show) + 3),
                ["model"] + [f"{r:g}" for r in show] + ["holds from", "late/div. ($10^{-2}$)",
                                                        "floor ($10^{-3}$)"], rows,
                "E3(b): predefined-time claims under Euler discretization. Entries: median $t_{10^{-3}}/T_c$ "
                "over all $(T_c,\\tau)$ pairs with the given $T_c/\\tau$ (columns) and 12 trajectories; "
                "values $\\leq 1$ satisfy the claim, $\\infty$ = never reached. Constant-gain activations use "
                "$\\gamma=\\mathrm{bound}(1)/T_c$, the strictly-predefined models $t_c=T_c$. Last column: "
                "smallest $T_c/\\tau$ from which the claim holds for at least half of the trajectories at "
                "every larger ratio. late/div.: largest $T_c/\\tau$ at which at least half of the trajectories "
                "reach $\\varepsilon=10^{-2}$ only after $T_c$ or diverge (time-limited breakdown); floor: "
                "largest $T_c/\\tau$ at which at least half never stay below $\\varepsilon=10^{-3}$ although "
                "they do not diverge ($\\varepsilon$ inside the $O(\\gamma\\tau)$ chattering band; "
                "unit-dependent, see the unit arm).",
                "tab:E3_tc_breakdown")
    analyze_tvg(tc, figdir, tabdir, out)
    summ["tc"] = out


TC_DISPLAY = {"vaf": "VAF", "pvaf1": "PVAF-1", "pvaf2": "PVAF-2", "predefined_exp": "predefined-time (F2)",
              "fixed_time": "fixed-time", "sign_bi_power": "sign-bi-power",
              "znn + strict_predefined act. ($\\gamma$=2)": "strictly predefined act.",
              "sptc_nt_znn ($\\gamma$=2)": "SPTC-NT-ZNN", "sptc_nt_znn+cap ($\\gamma$=2)": "SPTC-NT-ZNN, capped gain",
              "sptc_an_foznn ($\\gamma$=2)": "SPTC-AN-FOZNN",
              "sptc_an_foznn+cap ($\\gamma$=2)": "SPTC-AN-FOZNN, capped gain"}


def fig_tc(tc, series, figdir):
    """Paper figure (fig_E3_tc_breakdown): median t_1e-3 / T_c and residual at T_c against T_c/tau
    under Euler discretization; rows: constant-gain activations, strictly predefined-time models.
    Infinite medians are drawn in the shaded strip at the top of each panel (meaning in the panel title)."""
    ratios = sorted(tc["tc_over_tau"].round(6).unique())
    ycols = [("teps_over_tc_0.001", "time to $10^{-3}$ / $T_c$", "shaded: median $\\infty$ (not reached or diverged)"),
             ("rel_res_at_tc", "$\\|E(T_c)\\|/\\|E(0)\\|$", "shaded: median $\\infty$ (diverged)")]
    tops = {}
    for j, (col, _, _) in enumerate(ycols):
        v = tc[col].replace([np.inf, -np.inf], np.nan).dropna()
        v = v[v > 0]
        tops[col] = (max(v.max(), 2.0) * 2.0) if len(v) else 10.0
    W, H = style.TEXTWIDTH, 4.22
    lft, gap, rgt = 0.52, 0.64, 0.03
    pw = (W - lft - gap - rgt) / 2
    ph = 1.15
    rows_y = [2.71, 0.78]           # bottom of the panels of row (a) and row (b), inches
    titles = ["(a) constant-gain predefined- and fixed-time activations, $\\gamma=\\mathrm{bound}(1)/T_c$",
              "(b) strictly predefined-time models, $t_c=T_c$, $\\gamma=2$"]
    with style.print_rc():
        fig = plt.figure(figsize=(W, H))
        for i, grp in enumerate(["const", "tv"]):
            grp_series = [x for x in series if x[0] == grp]
            hl = []
            for j, (col, ylabel, blabel) in enumerate(ycols):
                ax = fig.add_axes(((lft + j * (pw + gap)) / W, rows_y[i] / H, pw / W, ph / H))
                ytop = tops[col]
                band = Band(ax, len(grp_series))
                for ks, (g, name, sel, stl) in enumerate(grp_series):
                    s = tc[sel]
                    xs, ys, xn = [], [], []
                    for r in ratios:
                        sr = s[np.isclose(s["tc_over_tau"], r)][col]
                        if sr.empty:
                            continue
                        m = med(sr)
                        xs.append(r)
                        ys.append(max(m, 1e-16) if np.isfinite(m) else np.nan)   # NaN breaks the line
                        if not np.isfinite(m):
                            xn.append(r)
                    kw = {k: v for k, v in stl.items()}
                    mfc = kw.pop("markerfacecolor", None)
                    if xs:
                        ln, = ax.plot(jitter(xs, ks, len(grp_series), 1.06), ys, markersize=4.0, lw=1.1,
                                      label=TC_DISPLAY.get(name, name), markerfacecolor=mfc or kw["color"], **kw)
                        if j == 0:
                            hl.append(ln)
                    band.add(jitter(xn, ks, len(grp_series), 1.06), ks, marker=kw["marker"], markersize=4.0,
                             markerfacecolor="white", markeredgecolor=kw["color"])
                ax.axvline(10.0, color=style.GREY, lw=0.6, ls=":")
                if col == "teps_over_tc_0.001":
                    ax.axhline(1.0, color=style.INK2, lw=0.7, ls="--")
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlim(0.55, 1.8e4)
                band.finish(ytop, row=0.06, inf_size=9, top_pad=1.0)
                ax.set_title(blabel, loc="left", fontsize=7.5, color=style.INK2, pad=2.5)
                if i == 1:
                    ax.set_xlabel("$T_c/\\tau$")
                else:
                    ax.tick_params(labelbottom=False)
                ax.set_ylabel(ylabel)
            fig.text(0.02 / W, (rows_y[i] + ph + 0.22) / H, titles[i], ha="left", va="bottom", fontsize=8.5)
            fig.legend(handles=hl, loc="upper center",
                       bbox_to_anchor=(0.5, (rows_y[i] - (0.08 if i == 0 else 0.42)) / H), ncol=3,
                       frameon=False, borderaxespad=0.0, columnspacing=1.4, handlelength=2.4)
        return _save(fig, figdir, "fig_E3_tc_breakdown")


def amp_factor(gamma):
    """A(gamma) = prod_{1 <= j < gamma/2} |1 - gamma/j|: amplification, by the Euler
    steps t_c - j tau of the gain law gamma/(t_c - t), of any deviation from the
    smooth (quasi-steady) solution E_j ~ j tau^2 present before the last
    gamma/2 steps; independent of tau and t_c."""
    js = np.arange(1, int(math.ceil(gamma / 2.0)))
    js = js[js < gamma / 2.0]
    return float(np.prod(np.abs(1.0 - gamma / js))) if js.size else 1.0


def analyze_tvg(tc, figdir, tabdir, out):
    """1/(t_c - t) gain of sptc_nt_znn, uncapped vs capped (rho = 1, 1.9, 3).

    Divergence is DETERMINISTIC per (gamma, T_c, tau) cell: the 12 trajectories
    of a cell share the outcome (the pre-t_c dynamics are linear in E, and the
    amplification acts on the smooth part common to all x0).  Fractions are
    therefore reported as counts of T_c cells (k / n_cells), never with a
    trajectory-level bootstrap CI (pseudo-replication)."""
    labs = [lab for lab, _ in TVG_LABELS if lab in set(tc["model_label"])]
    s = tc[tc["model_label"].isin(labs)]
    if s.empty:
        return
    gammas = sorted(s["gamma"].unique())
    taus = sorted(s["tau"].unique())
    res = {}
    rows = []
    cell_mixed, cell_n = 0, 0
    for tau in taus:
        for g in gammas:
            d = {"log10_A": math.log10(amp_factor(g))}
            row = [f"${eps_tex(tau)}$", fmt(g), f"{d['log10_A']:.1f}"]
            for lab in labs:
                sg = s[(s["model_label"] == lab) & np.isclose(s["tau"], tau) & np.isclose(s["gamma"], g)]
                if sg.empty:
                    if lab in ("sptc_nt_znn", "sptc_nt_znn+cap"):
                        row += ["--", "--"]
                    continue
                cells = sg.groupby("T_c")["status"].apply(lambda x: float((x == "diverged").mean()))
                cell_n += len(cells)
                cell_mixed += int(((cells > 0) & (cells < 1)).sum())
                kdiv = int((cells >= 0.5).sum())
                d[lab] = {"n": int(len(sg)), "n_cells": int(len(cells)), "cells_diverged": kdiv,
                          "diverged_frac": float((sg["status"] == "diverged").mean()),
                          "diverged_tc_over_tau": sorted(float(round(tcv / tau, 6)) for tcv, v in cells.items()
                                                         if v >= 0.5),
                          "jump_median": med(sg["tc_jump"]), "jump_iqr": iqr(sg["tc_jump"]),
                          "amp_median": med(sg["tc_amp"]), "mssre_median": med(sg["mssre"]),
                          "rpeak_post_median": med(sg["tc_rpeak_post"])}
                if lab in ("sptc_nt_znn", "sptc_nt_znn+cap"):
                    row += [f"{kdiv}/{len(cells)}", fmt(d[lab]["rpeak_post_median"], 2)]
            res[f"tau={tau:g}|gamma={g:g}"] = d
            rows.append(row)
    write_table(Path(tabdir) / "E3_tvg_cap.tex", "rrrrrrr",
                ["$\\tau$", "$\\gamma$", "$\\log_{10}A(\\gamma)$", "div. (no cap)", "peak post (no cap)",
                 "div. (cap)", "peak post (cap)"], rows,
                "E3(b): SPTC-NT-ZNN with the time-varying gain $1/(t_c-t)$ under Euler, without and with the "
                "saturation cap $1/(t_c-t)\\le 1/(\\gamma\\tau)$ (effective step $\\le 1$). div.: number of "
                "$T_c$ cells (of 5) in which the trajectories diverged (the outcome is deterministic per cell: "
                "all 12 trajectories agree); peak post: median $\\max_{t\\ge t_c}\\|E\\|$ "
                "($\\infty$: diverged). $A(\\gamma)=\\prod_{1\\le j<\\gamma/2}|1-\\gamma/j|$: amplification of "
                "a deviation from the smooth solution by the last Euler steps before $t_c$ when "
                "$T_c/\\tau\\ge\\gamma/2$ (for $T_c/\\tau<\\gamma/2$ every step is in the unstable zone).",
                "tab:E3_tvg_cap")
    out["tvg"] = res
    out["tvg_cells"] = {"n_cells": cell_n, "mixed_cells": cell_mixed}
    # outcome map over (gamma, T_c/tau) for the uncapped law
    s2 = s[s["model_label"] == "sptc_nt_znn"]
    if not s2.empty:
        mp = {}
        for g in gammas:
            sg = s2[np.isclose(s2["gamma"], g)]
            byr = sg.groupby(sg["tc_over_tau"].round(6))
            mp[f"{g:g}"] = {f"{r:g}": {"diverged": float((x["status"] == "diverged").mean()),
                                       "rpeak_post_median": med(x["tc_rpeak_post"]),
                                       "r_at_tc_median": med(x["tc_r_at"])} for r, x in byr}
        out["tvg_map_uncapped"] = mp
    fig_tvg(s, labs, gammas, taus, figdir)
    fig_tvg_traces(tc, figdir)


def _tvg_style(k):
    base = keys.style_in_group("model", "sptc_nt_znn")
    cols = [base["color"], style.CAT[1], style.CAT[2], style.CAT[3]]
    ls = ["-", "--", "-.", ":"]
    mk = [base["marker"], "s", "^", "D"]
    mfc = [cols[0], "white", "white", "white"]
    return {"color": cols[k % 4], "linestyle": ls[k % 4], "marker": mk[k % 4], "markerfacecolor": mfc[k % 4]}


def fig_tvg(s, labs, gammas, taus, figdir):
    fig, axes = plt.subplots(2, len(taus), figsize=(style.DOUBLE, 3.8), sharex=True, sharey="row",
                             squeeze=False)
    names = dict(TVG_LABELS)
    pk = s["tc_rpeak_post"].replace([np.inf, -np.inf], np.nan).dropna()
    pk = pk[pk > 0]
    ytop = max(pk.max() * 3, 10) if len(pk) else 1e3
    ybot = max(min(pk.min() / 3, 1e-4), 1e-16) if len(pk) else 1e-6
    bands = [Band(axes[0, j], len(labs)) for j in range(len(taus))]
    for j, tau in enumerate(taus):
        for kk, lab in enumerate(labs):
            stl = _tvg_style([x for x, _ in TVG_LABELS].index(lab))
            xs, ys, dv, xn = [], [], [], []
            for g in gammas:
                sg = s[(s["model_label"] == lab) & np.isclose(s["tau"], tau) & np.isclose(s["gamma"], g)]
                if sg.empty:
                    continue
                cells = sg.groupby("T_c")["status"].apply(lambda x: float((x == "diverged").mean()))
                dv.append((g, float((cells >= 0.5).mean())))
                m = med(sg["tc_rpeak_post"])
                if np.isfinite(m):
                    xs.append(g)
                    ys.append(m)
                else:
                    xn.append(g)
            ax = axes[0, j]
            if xs:
                ax.plot(jitter(xs, kk, len(labs), 1.05), ys, markersize=3.4, label=names[lab], **stl)
            bands[j].add(jitter(xn, kk, len(labs), 1.05), kk, marker=stl["marker"], markersize=3.4,
                         markerfacecolor=stl["markerfacecolor"], markeredgecolor=stl["color"])
            if dv:
                axes[1, j].plot(jitter([d[0] for d in dv], kk, len(labs), 1.05), [d[1] for d in dv],
                                markersize=3.4, label=names[lab], **stl)
        ax = axes[0, j]
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"$\\tau={tau:g}$ s", fontsize=7.5)
        if j == 0:
            ax.set_ylabel("peak $\\|E\\|$ after $t_c$", fontsize=7)
            axes[1, j].set_ylabel("fraction of $T_c$ cells\ndiverged", fontsize=7)
        axes[1, j].set_ylim(-0.05, 1.05)
        axes[1, j].set_xlabel("base gain $\\gamma$")
    for j in range(len(taus)):
        bands[j].finish(ytop, bottom=ybot)
    hh = [plt.Line2D([], [], markersize=3.4, label=names[lab], **_tvg_style([x for x, _ in TVG_LABELS].index(lab)))
          for lab in labs]
    hh.append(plt.Line2D([], [], linestyle="none", label="$\\infty$: median diverged"))
    axes[0, -1].legend(handles=hh, loc="center left", bbox_to_anchor=(1.01, 0.0), fontsize=6)
    fig.tight_layout()
    _save(fig, figdir, "fig_E3_tvg_cap")


def fig_tvg_traces(tc, figdir):
    from znnbench.runner import load_traces
    cfg = "E3_tc_1"
    sub = tc[(tc["config"] == cfg) & tc["model_label"].isin(["sptc_nt_znn", "sptc_nt_znn+cap"])
             & (tc["seed"] == 0) & np.isclose(tc["tau"], 1e-3) & (tc["x0_index"] == 0)]
    if sub.empty or "traces_file" not in sub:
        return
    gammas = sorted(sub["gamma"].unique())
    if len(gammas) > 8:      # style rule: at most 8 series per panel (full scale has a wider gain grid)
        idx = np.unique(np.linspace(0, len(gammas) - 1, 8).round().astype(int))
        gammas = [gammas[i] for i in idx]
    shades = [style.SEQ_BLUE[k] for k in np.linspace(4, len(style.SEQ_BLUE) - 1, len(gammas)).astype(int)]
    fig, axes = plt.subplots(1, 2, figsize=(style.ONEHALF, 2.2), sharey=True)
    res_dir = Path(tc.attrs.get("results_dir", "."))
    for ax, lab, title in ((axes[0], "sptc_nt_znn", "no cap"), (axes[1], "sptc_nt_znn+cap", "cap $1/(\\gamma\\tau)$")):
        for k, g in enumerate(gammas):
            row = sub[(sub["model_label"] == lab) & np.isclose(sub["gamma"], g)]
            if row.empty or not isinstance(row["traces_file"].iloc[0], str):
                continue
            f = res_dir / cfg / "results" / row["traces_file"].iloc[0]
            if not f.exists():
                continue
            tr = load_traces(f, row["run_id"].iloc[0])
            col = int(row["traj"].iloc[0])
            r = tr["residual"][:, col].astype(float)
            r = np.where(np.isfinite(r), r, np.nan)
            ax.plot(tr["t"] / 1.0, r, color=shades[k], linestyle=style.LINESTYLES[k], lw=1.0,
                    label=f"$\\gamma={g:g}$")
        ax.axvline(1.0, color=style.GREY, lw=0.6, ls=":")
        ax.set_yscale("log")
        ax.set_xlim(0, 2.0)
        ax.set_xlabel("$t/t_c$")
        ax.set_title(title, fontsize=7.5)
    axes[0].set_ylabel("$\\|E(t)\\|$")
    axes[1].legend(loc="upper right", fontsize=6, ncol=2)
    fig.suptitle("SPTC-NT-ZNN, Euler, $t_c=1$ s, $\\tau=10^{-3}$ s (seed 0, one initial state)", fontsize=7.5)
    fig.tight_layout()
    _save(fig, figdir, "fig_E3_tvg_traces")


# =============================================================================
# unit-dependence arm (E3_units.ScaledError, review 2026-10-18)
# =============================================================================
UNIT_SETTINGS = [("euler", "fixed_gamma", 10.0, "Euler, fixed $\\gamma=10$"),
                 ("euler", "fixed_h", 0.1, "Euler, fixed $h=0.1$")]


def analyze_units_disc(disc_all, tabdir, summ):
    """Same runs with E expressed as s E (s = 1e-2, 1e2): floors, reach and rankings."""
    if disc_all.empty or "unit" not in disc_all or disc_all["unit"].nunique() < 2:
        return
    units = sorted(disc_all["unit"].unique())
    taus = sorted(disc_all.loc[disc_all["unit"] != 1.0, "tau"].unique())
    acts = [a for a in keys.GROUPS["act"] if a in set(disc_all["model_label"])]
    acts += sorted(set(disc_all["model_label"]) - set(acts))
    out = {"units": [float(u) for u in units], "taus": [float(t) for t in taus], "settings": {}}
    rows = []
    for dk, mode, val, title in UNIT_SETTINGS:
        sub = disc_all[(disc_all["disc_key"] == dk) & _gain_match(disc_all, mode, val)]
        if sub.empty:
            continue
        sres = {}
        for tau in taus:
            st = sub[np.isclose(sub["tau"], tau)]
            cell = {}
            for a in acts:
                for u in units:
                    x = st[(st["model_label"] == a) & np.isclose(st["unit"], u)]
                    if x.empty:
                        continue
                    cell.setdefault(a, {})[f"{u:g}"] = {
                        "n": int(len(x)), "diverged": int((x["status"] == "diverged").sum()),
                        "mssre_median": med(x["mssre"]),
                        "reached_0.001": float(np.isfinite(x["t_eps_0.001"]).mean()),
                        "reached_1e-06": float(np.isfinite(x["t_eps_1e-06"]).mean()),
                        "t_eps_0.001_median": med(x["t_eps_0.001"])}
            # Kendall tau_b of the activation ranking (median t_eps 1e-3, inf = worst) vs the natural unit
            kt = {}
            for u in units:
                if u == 1.0:
                    continue
                aa = [a for a in acts if f"{u:g}" in cell.get(a, {}) and "1" in cell.get(a, {})]
                if len(aa) >= 3:
                    kt[f"{u:g}"] = dict(zip(("tau_b", "p_descriptive"), kendall_b(
                        [cell[a]["1"]["t_eps_0.001_median"] for a in aa],
                        [cell[a][f"{u:g}"]["t_eps_0.001_median"] for a in aa])))
            sres[f"{tau:g}"] = {"cells": cell, "kendall_vs_natural": kt}
            for a in acts:
                c = cell.get(a, {})
                if not c:
                    continue
                row = [tex_name(a), title.split(",")[1].strip(), f"${eps_tex(tau)}$"]
                for u in units:
                    v = c.get(f"{u:g}")
                    if v is None:
                        row += ["--"]
                        continue
                    k = int(round(v["reached_0.001"] * v["n"]))
                    row += [fmt(v["mssre_median"], 2) + f" ({k}/{v['n']})"]
                rows.append(row)
        # linear ZNN must be unit-invariant (sanity check of the wrapper)
        lin = sub[sub["model_label"] == "linear"]
        if not lin.empty:
            piv = lin.pivot_table(index=["tau", "seed", "x0_index", "gain_index"], columns="unit", values="mssre")
            if 1.0 in piv.columns:
                dev = [float(np.nanmax(np.abs(piv[u] / piv[1.0] - 1))) for u in piv.columns if u != 1.0]
                sres["linear_unit_invariance_max_rel_dev"] = max(dev) if dev else float("nan")
        out["settings"][f"{dk}|{mode}|{val:g}"] = {"title": title, "by_tau": sres}
    summ["units_disc"] = out
    if rows:
        write_table(Path(tabdir) / "E3_units.tex", "lll" + "r" * len(units),
                    ["activation", "gain", "$\\tau$"] + [f"$s={u:g}$" for u in units], rows,
                    "E3(b): unit dependence. The same Euler runs with the error expressed as $E_s=sE$ "
                    "(the activation acts on $E_s$; the residual and $\\varepsilon$ stay in the original unit). "
                    "Entries: median MSSRE floor and, in brackets, the number of the 12 trajectories that reach "
                    "$\\varepsilon=10^{-3}$ for good ($\\infty$: median diverged). $s=1$: natural unit "
                    "($X(0)\\sim U[-2,2]$). Linear ZNN is unit-invariant (check).", "tab:E3_units")


def analyze_units_tc(tc_all, summ):
    """T_c/tau thresholds of the constant-gain activations per error unit."""
    if tc_all.empty or "unit" not in tc_all or tc_all["unit"].nunique() < 2:
        return
    tc_all = prepare_tc(tc_all)
    taus_u = sorted(tc_all.loc[tc_all["unit"] != 1.0, "tau"].unique())
    sub = tc_all[tc_all["tau"].isin(taus_u) & tc_all["model_label"].isin(TC_CONST)]
    out = {"taus": [float(t) for t in taus_u], "series": {}}
    for a in [x for x in TC_CONST if x in set(sub["model_label"])]:
        for u in sorted(sub["unit"].unique()):
            x = sub[(sub["model_label"] == a) & np.isclose(sub["unit"], u)]
            if x.empty:
                continue
            rec = {}
            for r, xr in x.groupby(x["tc_over_tau"].round(6)):
                rec[f"{r:g}"] = {"n": int(len(xr)), **{f"outcome_{e:g}": {k: float(np.mean(tc_outcome(xr, e) == k))
                                                                          for k in TC_OUTCOMES} for e in EPS},
                                 "rel_res_at_tc_median": med(xr["rel_res_at_tc"])}
            thr = {}
            for e in EPS:
                ok = sorted((float(k), v[f"outcome_{e:g}"]["ok"] >= 0.5) for k, v in rec.items())
                t = None
                for k, flag in reversed(ok):
                    if flag:
                        t = k
                    else:
                        break
                thr[f"{e:g}"] = t
            out["series"].setdefault(a, {})[f"{u:g}"] = {"by_ratio": rec, "claim_holds_from": thr}
    summ["units_tc"] = out


# =============================================================================
# varying-parameter (unbounded exponential) gain with / without cap (E3_vp)
# =============================================================================
def analyze_vp(vp, tabdir, summ):
    if vp.empty:
        return
    try:
        sys.path.insert(0, str(ROOT / "experiments" / "configs"))
        import E3 as cfg  # noqa: E402
        tstar = cfg.vp_t_star
        gam = cfg.VP_GAMMA
    except Exception:  # noqa: BLE001
        tstar, gam = None, 10.0
    out = {}
    rows = []
    for lab in sorted(vp["model_label"].unique()):
        for tau in sorted(vp["tau"].unique()):
            x = vp[(vp["model_label"] == lab) & np.isclose(vp["tau"], tau)]
            if x.empty:
                continue
            dv = x["status"] == "diverged"
            d = {"n": int(len(x)), "diverged": int(dv.sum()),
                 "diverged_at_median": med(x.loc[dv, "diverged_at"]) if dv.any() else float("nan"),
                 "mssre_median": med(x["mssre"]), "t_star_linear_pred": tstar(gam, tau) if tstar else float("nan")}
            out[f"{lab}|tau={tau:g}"] = d
            rows.append([tex_name(lab), f"${eps_tex(tau)}$", f"{d['diverged']}/{d['n']}",
                         fmt(d["diverged_at_median"], 3), fmt(d["t_star_linear_pred"], 3), fmt(d["mssre_median"], 2)])
    summ["vp"] = out
    write_table(Path(tabdir) / "E3_vp.tex", "llrrrr",
                ["model", "$\\tau$", "diverged", "$t_{\\mathrm{div}}$ (s)", "$t^*$ pred. (s)", "MSSRE"], rows,
                "E3(b): varying-parameter ZNN (\\texttt{vpznn\\_exp}, $g(t)=2^t+2$, $\\gamma=10$, Euler, $T=12$ s), "
                "without and with the cap $g\\le 1/(\\gamma\\tau)$. diverged: trajectories of 12; "
                "$t_{\\mathrm{div}}$: median time at which $\\|E\\|$ exceeded $10^{10}$; $t^*$: predicted onset "
                "of the linear Euler instability $\\gamma\\tau g(t^*)=2$.", "tab:E3_vp")


# =============================================================================
# residual floor vs eps (all discrete runs)
# =============================================================================
def analyze_floor_reach(frames, tabdir, summ):
    df = pd.concat([f for f in frames if not f.empty], ignore_index=True) if any(not f.empty for f in frames) \
        else pd.DataFrame()
    if df.empty:
        return
    df = df[df["status"] == "ok"]
    out = {}
    rows = []
    for e in EPS:
        tcol, hcol = f"t_eps_{e:g}", f"t_hit_{e:g}"
        below_floor = df["res_min_ss"] > e            # eps below the steady-state floor
        above = df["mssre"] < e                       # whole steady window below eps
        band = ~below_floor & ~above                  # floor band straddles eps
        r = {"n": int(len(df)),
             "n_eps_below_floor": int(below_floor.sum()),
             "reached_given_below_floor": float(np.isfinite(df.loc[below_floor, tcol]).mean()) if below_floor.any() else float("nan"),
             "first_hit_given_below_floor": float(np.isfinite(df.loc[below_floor, hcol]).mean()) if below_floor.any() and hcol in df else float("nan"),
             "n_window_below_eps": int(above.sum()),
             "reached_given_window_below": float(np.isfinite(df.loc[above, tcol]).mean()) if above.any() else float("nan"),
             "n_band": int(band.sum()),
             "reached_given_band": float(np.isfinite(df.loc[band, tcol]).mean()) if band.any() else float("nan"),
             "first_hit_given_band": float(np.isfinite(df.loc[band, hcol]).mean()) if band.any() and hcol in df else float("nan")}
        out[f"{e:g}"] = r
        rows.append([f"${eps_tex(e)}$", str(r["n_eps_below_floor"]), fmt(r["reached_given_below_floor"], 2),
                     fmt(r["first_hit_given_below_floor"], 2), str(r["n_band"]), fmt(r["reached_given_band"], 2),
                     fmt(r["first_hit_given_band"], 2), str(r["n_window_below_eps"]),
                     fmt(r["reached_given_window_below"], 2)])
    write_table(Path(tabdir) / "E3_floor_reach.tex", "rrrrrrrrr",
                ["$\\varepsilon$", "$n$: $\\varepsilon<$ floor", "reached", "first hit", "$n$: band",
                 "reached", "first hit", "$n$: below $\\varepsilon$", "reached"], rows,
                "E3(b): $\\varepsilon$ below the discrete residual floor is never reached. All non-diverged "
                "E3 discrete runs. Floor = $\\min$ of $\\|E\\|$ over the steady window (last 25\\%); band: "
                "$\\min\\le\\varepsilon\\le\\max$ (chattering straddles $\\varepsilon$); below: whole window "
                "$<\\varepsilon$. reached: fraction with finite $t_\\varepsilon$ (stays below); first hit: "
                "fraction that dipped below $\\varepsilon$ at least once.", "tab:E3_floor_reach")
    summ["floor_reach"] = out


# =============================================================================
# cost (for the full-scale estimate)
# =============================================================================
def censoring_summary(frames, ct, summ):
    """Counts of diverged / failed / never-reached runs (all kept in the statistics as +inf)."""
    out = {}
    for name, df in frames.items():
        if df.empty:
            continue
        d = {"rows": int(len(df)), "status": df["status"].value_counts().to_dict(),
             "failed_or_timeout": int((~df["status"].isin(["ok", "diverged"])).sum())}
        for e in EPS:
            c = f"t_eps_{e:g}"
            if c in df:
                d[f"never_{e:g}"] = int((~np.isfinite(df[c].astype(float))).sum())
        out[name] = d
    if not ct.empty:
        out["ct"] = {"rows": int(len(ct)),
                     "t_ref_nan": int(ct[[f"t_ref_{e:g}" for e in EPS]].isna().any(axis=1).sum()),
                     "full_state_wall_cap": int((ct["full_status"] == "wall_cap").sum())}
    summ["censoring"] = out


def cost_summary(frames, ct, summ):
    out = {}
    for name, df in frames.items():
        if df.empty:
            continue
        d = df.drop_duplicates("run_id")
        d = d[d["status"].isin(["ok", "diverged"]) | (d["n_diverged"] >= 0)]
        g = d.groupby(["disc_key", "n_gains"])
        out[name] = {"specs": int(len(d)), "wall_total_s": float(d["wall"].sum()),
                     "steps_total": float(d["steps"].sum()),
                     "wall_per_step_median_by_disc_ngains": {f"{k[0]}|G={k[1]}": float(v) for k, v in
                                                             g["wall_per_step"].median().items()},
                     "peak_rss_mb_max": float(d["peak_rss_mb"].max())}
    if not ct.empty:
        out["ct"] = {"tasks": int(ct["task_id"].nunique()),
                     "task_wall_total_s": float(ct.drop_duplicates("task_id")["task_wall"].sum()),
                     "peak_rss_mb_max": float(ct["peak_rss_mb"].max())}
    summ["cost"] = out


# =============================================================================
# summary markdown
# =============================================================================
def write_summary_md(summ, path):
    L = ["# E3 summary: convergence-time audit", "",
         f"results: `{summ['results']}`  (generated by experiments/analyze_E3.py)", ""]
    ct = summ.get("ct", {})
    if ct.get("entries"):
        L += ["## (a) Continuous time: conservatism of the stated bounds", "",
              "Exact time to ||E||_2 < 1e-6 (semi-analytic; natural initial states, 12 trajectories) vs the bound.",
              "", "| model | class (claimed/derived) | gamma | t(1e-6) s | bound s | ratio median [95% cluster CI] | tight ratio | ratio range over e0 (= over units of E) | valid |",
              "|---|---|---|---|---|---|---|---|---|"]
        for lab, r in ct["entries"].items():
            rs = r.get("ratio_bound_source")
            rt = r.get("ratio_bound_tight")
            vals = [v["ratio_median"] for v in r["ratio_by_e0"].values() if np.isfinite(v["ratio_median"])]
            rng = f"{min(vals):.3g}-{max(vals):.3g}" if vals else "-"
            src = f"{rs['median']:.3g} [{rs['ci95'][0]:.3g}, {rs['ci95'][1]:.3g}]" if rs else "-"
            tight = f"{rt['median']:.3g}" if rt else "-"
            L.append(f"| {lab} | {r['claimed_class']}/{r['derived_class']} | {r['gamma']:g} | "
                     f"{r['t_ref_1e-06_median']:.4g} | {r.get('bound_source_median', float('nan')):.4g} | "
                     f"{src} | {tight} | {rng} | {'yes' if all(r['bounds_valid_all_rows'].values()) else 'NO'} |")
        c = ct.get("checks", {})
        L += ["", f"Checks: bracket (quadrature vs table) ok fraction = {c.get('bracket_ok_fraction')}; "
              f"full-state DOP853 runs = {c.get('full_state_runs')} ({c.get('full_state_status_counts')}), "
              f"max rel. deviation of ||E(t)|| = {c.get('full_state_max_rel_dev_norm')}, of t_eps = "
              f"{c.get('full_state_max_rel_dev_t_eps')} (n = {c.get('full_state_n_t_eps_compared')}); "
              f"gamma-scaling max rel. dev = {c.get('gamma_scaling_max_rel_dev', 'n/a')}."]
        kb = ct.get("kendall_bound_vs_actual")
        if kb:
            L.append(f"Kendall tau_b(stated bound, actual CT time) over {kb['n']} activations = "
                     f"{kb['tau_b']:.3f} (p = {kb['p']:.3g}, descriptive only: n = {kb['n']}, no multiplicity "
                     f"control; the e0-dependent finite-time bounds are exact by construction, which drives "
                     f"most of the agreement).")
        L.append("CIs: cluster bootstrap (resample the 3 problem seeds, then the 4 initial states within a seed), "
                 "2000 resamples; with 3 clusters they are indicative (full run: 10 CT seeds).")
        L.append("")
    d = summ.get("disc", {})
    if d.get("settings"):
        L += ["## (b) Discrete time: time-to-eps vs tau (E3_disc)", ""]
        for key, s in d["settings"].items():
            L.append(f"**{s['title']}** -- median t_eps / t_eps^CT (reached / n) per tau:")
            L.append("")
            for a, ares in s["activations"].items():
                parts = []
                for t, r in ares.items():
                    if not isinstance(r, dict) or "n" not in r:
                        continue
                    parts.append(f"tau={t}: eps=1e-3 {r.get('ratio_0.001_median', float('nan')):.3g} "
                                 f"({r['reached_0.001'] * r['n']:.0f}/{r['n']}), eps=1e-6 "
                                 f"{r.get('ratio_1e-06_median', float('nan')):.3g} ({r['reached_1e-06'] * r['n']:.0f}/{r['n']}), "
                                 f"floor {r['mssre_median']:.2e}" + (f", diverged {r['diverged']}/{r['n']}"
                                                                    if r.get('diverged') else ""))
                fo = ares.get("floor_order", {})
                L.append(f"- {a}: " + "; ".join(parts) + f"; floor order {fo.get('slope', float('nan')):.2f} "
                         f"(all points {fo.get('slope_all', float('nan')):.2f}; local "
                         f"{', '.join(f'{v:.2f}' for v in fo.get('local', []))}"
                         + (f"; {fo['flag']}" if fo.get('flag') else "") + ")")
            if s.get("kendall_ct_vs_disc"):
                L.append("- Kendall tau_b (CT vs discrete ranking, eps=1e-3; never reached = tied worst; "
                         "descriptive, n = 6): " + ", ".join(
                             f"tau={t}: {v['tau_b']:.2f}" for t, v in s["kendall_ct_vs_disc"].items()))
            L.append("")
    fr = summ.get("floor_reach")
    if fr:
        L += ["## eps below the residual floor", ""]
        for e, r in fr.items():
            L.append(f"- eps={e}: {r['n_eps_below_floor']} runs with eps < steady floor -> reached fraction "
                     f"{r['reached_given_below_floor']}, first-hit fraction {r['first_hit_given_below_floor']}; "
                     f"{r['n_band']} runs in the chattering band -> reached {r['reached_given_band']}, first hit "
                     f"{r['first_hit_given_band']}; {r['n_window_below_eps']} runs whose window is below eps -> "
                     f"reached {r['reached_given_window_below']}.")
        L.append("")
    tc = summ.get("tc", {})
    if tc.get("series"):
        L += ["## (b) Predefined T_c vs tau (E3_tc_*)", ""]
        for name, s in tc["series"].items():
            thr = s["claim_holds_from_tc_over_tau"]
            fm = (lambda v: "never (in the grid)" if v is None else f"{v:g}")
            brk = s.get("time_limited_breakdown_up_to", {})
            flo = s.get("floor_limited_up_to", {})
            fb = (lambda v: "none" if v is None else f"<= {v:g}")
            L.append(f"- {name}: claim t_eps <= T_c holds (>= 50 %) from T_c/tau = "
                     f"{fm(thr.get('0.01'))} (eps=1e-2), {fm(thr.get('0.001'))} (eps=1e-3), "
                     f"{fm(thr.get('1e-06'))} (eps=1e-6); time-limited breakdown (late or diverged >= 50 %) at "
                     f"T_c/tau {fb(brk.get('0.01'))} (eps=1e-2), {fb(brk.get('0.001'))} (eps=1e-3); "
                     f"floor-limited (never stays below eps, not diverged) at T_c/tau {fb(flo.get('0.001'))} "
                     f"(eps=1e-3)")
        L += ["", "Outcome classes per trajectory: ok (t_eps <= T_c), late (T_c < t_eps < inf: time-limited), "
              "floor (never stays below eps but bounded: eps inside the O(gamma tau) chattering band; "
              "unit-dependent), diverged. Pairs (T_c, tau) with the same ratio are pooled; the problem's "
              "1 rad/s time variation is not rescaled with T_c (a minor confound).", ""]
    tv = tc.get("tvg")
    if tv:
        labs = [lab for lab, _ in TVG_LABELS]
        L += ["## Time-varying gain 1/(t_c - t): with vs without cap", "",
              "Cells = T_c values (5) per (tau, gamma); divergence is deterministic per cell "
              f"({tc.get('tvg_cells', {}).get('mixed_cells', '?')} of {tc.get('tvg_cells', {}).get('n_cells', '?')} "
              "cells have mixed outcomes), so counts of cells are reported instead of trajectory-level CIs "
              "(the earlier bootstrap over 60 pooled trajectories was pseudo-replication).", "",
              "| tau | gamma | log10 A | " + " | ".join(f"{dict(TVG_LABELS)[x]}: div. cells, peak post-t_c" for x in labs) + " |",
              "|---|---|---|" + "---|" * len(labs)]
        for k, d in tv.items():
            t, g = k.split("|")
            cells = []
            for lab in labs:
                a = d.get(lab)
                cells.append("-" if not a else f"{a['cells_diverged']}/{a['n_cells']}, {a['rpeak_post_median']:.3g}")
            L.append(f"| {t[4:]} | {g[6:]} | {d.get('log10_A', float('nan')):.1f} | " + " | ".join(cells) + " |")
        mp = tc.get("tvg_map_uncapped", {})
        if mp:
            L += ["", "Uncapped law, T_c/tau at which the runs diverge, per base gain:"]
            for g, m in mp.items():
                dv = [r for r, v in m.items() if v["diverged"] >= 0.5]
                L.append(f"- gamma = {g}: diverged at T_c/tau in {{{', '.join(dv) if dv else 'none'}}}; "
                         "median peak ||E|| after t_c by T_c/tau: " +
                         ", ".join(f"{r}: {v['rpeak_post_median']:.2g}" for r, v in m.items()))
        L.append("")
    ud = summ.get("units_disc", {})
    if ud.get("settings"):
        L += ["## Unit dependence (paired scaled-error arm, E_s = s E)", "",
              "Same Euler runs with the activation applied to E_s = s E (residual and eps in the original "
              "unit). Per activation: median MSSRE floor (reached eps=1e-3 / n, diverged).", ""]
        for key, st in ud["settings"].items():
            L.append(f"**{st['title']}** (linear ZNN unit-invariance check: max rel. dev. of MSSRE = "
                     f"{st['by_tau'].get('linear_unit_invariance_max_rel_dev', float('nan')):.2g})")
            for tau, c in st["by_tau"].items():
                if not isinstance(c, dict):
                    continue
                L.append(f"- tau = {tau}:")
                for a, cu in c["cells"].items():
                    L.append(f"  - {a}: " + "; ".join(
                        f"s={u}: {v['mssre_median']:.2e} ({v['reached_0.001'] * v['n']:.0f}/{v['n']}"
                        + (f", div {v['diverged']}" if v['diverged'] else "") + ")" for u, v in cu.items()))
                kt = c.get("kendall_vs_natural", {})
                if kt:
                    L.append("  - Kendall tau_b of the discrete ranking vs the natural unit (descriptive): " +
                             ", ".join(f"s={u}: {v['tau_b']:.2f}" if v.get('tau_b') is not None else f"s={u}: n/a"
                                       for u, v in kt.items()))
            L.append("")
    ut = summ.get("units_tc", {})
    if ut.get("series"):
        L += [f"**T_c sweep, constant-gain activations, per unit** (tau in {ut['taus']}): claim t_eps <= T_c "
              "holds (>= 50 %) from T_c/tau ="]
        for a, bu in ut["series"].items():
            L.append(f"- {a}: " + "; ".join(
                f"s={u}: {('never' if v['claim_holds_from'].get('0.01') is None else format(v['claim_holds_from']['0.01'], 'g'))} "
                f"(eps=1e-2), {('never' if v['claim_holds_from'].get('0.001') is None else format(v['claim_holds_from']['0.001'], 'g'))} (eps=1e-3)"
                for u, v in bu.items()))
        L.append("")
    vp = summ.get("vp")
    if vp:
        L += ["## Varying-parameter gain g(t) = 2^t + 2 (vpznn_exp, gamma = 10, Euler, T = 12 s)", ""]
        for k, d in vp.items():
            L.append(f"- {k}: diverged {d['diverged']}/{d['n']}, median divergence time "
                     f"{d['diverged_at_median']:.3g} s (predicted linear onset t* = {d['t_star_linear_pred']:.3g} s), "
                     f"MSSRE {d['mssre_median']:.3g}")
        L.append("")
    cz = summ.get("censoring")
    if cz:
        L += ["## Censoring", "",
              "Diverged runs (MSSRE = inf), never-reached levels (t_eps = inf) and failed runs are kept in every "
              "median, IQR, quantile and ranking as +inf (worst value, tied); nothing is dropped by isfinite / "
              "dropna filters except the CT-ratio columns of runs without a CT reference (none here).", ""]
        for name, d in cz.items():
            L.append(f"- {name}: " + ", ".join(f"{k} = {v}" for k, v in d.items()))
        L.append("")
    L += ["## Confounds (see experiments/E3_NOTES.md, Review log)", "",
          "- Unit of E: every power/sign-type result in (b) depends on the unit of E (unit arm above); the CT "
          "conservatism ratios depend on it through e0 (heatmap of fig_E3_conservatism).",
          "- Equal nominal gamma is not equal effective gain: Euler step h_eff = gamma tau Phi'(e) is unbounded for "
          "the high-power terms at large |e| and for 1/(t_c - t); fixed-h runs equalise gamma tau only.",
          "- Constant-gain activations in the T_c sweep are scaled so that the STATED bound equals T_c; since the "
          "bounds are 1.7-4.5x conservative, the CT trajectories reach 1e-3 well before T_c.",
          "- Pairs (T_c, tau) with equal ratio are pooled although the problem's 1 rad/s variation is not rescaled.",
          "- eps below the floor -> t_eps = inf is true by definition of the stay-below metric; the first-hit "
          "fractions are the informative part.",
          "- The cap rho = 1 is the deadbeat step of linear Euler (best case); rho = 1.9 and 3 are the controls.",
          ""]
    if summ.get("interpretation"):
        L += ["## Interpretation", ""] + [f"- {s}" for s in summ["interpretation"]] + [""]
    Path(path).write_text("\n".join(L) + "\n")


def interpret(summ):
    msgs = []
    ct = summ.get("ct", {})
    ents = ct.get("entries", {})
    pt = {k: v for k, v in ents.items() if v.get("derived_class") in ("predefined", "fixed") and "ratio_bound_source" in v}
    if pt:
        lo = min(v["ratio_bound_source"]["median"] for v in pt.values())
        hi = max(v["ratio_bound_source"]["median"] for v in pt.values())
        msgs.append(f"Stated fixed/predefined-time bounds overestimate the actual continuous-time time to 1e-6 by "
                    f"a factor {lo:.2g}-{hi:.2g} at the natural initial error (median); the factor grows without "
                    f"bound as the initial error shrinks (uniform bounds do not depend on e0).")
    fin = {k: v for k, v in ents.items() if v.get("claimed_class") == "finite" and "ratio_bound_source" in v}
    if fin:
        r = [v["ratio_bound_source"]["median"] for v in fin.values()]
        msgs.append(f"e0-dependent finite-time bounds are nearly tight (median ratio {min(r):.3g}-{max(r):.3g}); "
                    f"the time from 1e-6 to exact zero is negligible.")
    strict = {k: v for k, v in ents.items() if v.get("derived_class") == "strictly-predefined" and "ratio_bound_source" in v}
    if strict:
        r = [v["ratio_bound_source"]["median"] for v in strict.values()]
        msgs.append(f"Strictly predefined-time models reach 1e-6 just before t_c in continuous time (ratio "
                    f"{min(r):.3g}-{max(r):.3g} at gamma = 2): the bound is tight by construction.")
    if ents and all(all(v["bounds_valid_all_rows"].values()) for v in ents.values()):
        msgs.append("No stated or derived bound was violated by the exact settling time in any run.")
    fr = summ.get("floor_reach", {})
    if fr:
        z = [r["reached_given_below_floor"] for r in fr.values() if np.isfinite(r["reached_given_below_floor"])]
        if z:
            msgs.append(f"Whenever eps lies below the discrete residual floor, t_eps is infinite in "
                        f"{100 * (1 - max(z)):.0f}% of the runs (by construction of the stay-below definition); "
                        f"the first-hit time shows how often chattering dips below eps without staying.")
    ser = summ.get("tc", {}).get("series", {})
    if ser:
        parts, brk = [], []
        for name, v in ser.items():
            t = v["claim_holds_from_tc_over_tau"].get("0.001")
            parts.append(f"{name.split(' (')[0]}: {'never' if t is None else f'{t:g}'}")
            b = v.get("time_limited_breakdown_up_to", {}).get("0.01")
            brk.append(f"{name.split(' (')[0]}: {'none' if b is None else f'<= {b:g}'}")
        msgs.append("Predefined-time claim t_1e-3 <= T_c under Euler holds (>= 50 % of trajectories, natural unit) "
                    "only from T_c/tau = " + "; ".join(parts) + ". For the sign-term activations this is a FLOOR "
                    "limit (eps = 1e-3 inside the O(gamma tau) chattering band, gamma = bound/T_c, so the band "
                    "scales as tau/T_c), not a time limit, and it moves with the unit of E (unit arm).")
        msgs.append("Time-limited breakdown (>= 50 % reach eps = 1e-2 only after T_c, or diverge) occurs up to "
                    "T_c/tau = " + "; ".join(brk) + " -- this is the PLAN's 'T_c <~ 10 tau' breakdown, measured.")
    tv = summ.get("tc", {}).get("tvg", {})
    if tv:
        div = sorted({float(k.split("gamma=")[1]) for k, d in tv.items()
                      if d.get("sptc_nt_znn", {}).get("cells_diverged", 0) > 0})
        if div:
            msgs.append(f"Uncapped 1/(t_c - t) gains diverge from base gain gamma = {div[0]:g} on, in the cells "
                        "listed in the TVG section (outcome determined by gamma and T_c/tau, identical for all "
                        "initial states); larger T_c/tau does not diverge but leaves a post-t_c residual set by "
                        f"A(gamma) = prod_(j<gamma/2) |1 - gamma/j| (~10^{math.log10(amp_factor(div[0])):.0f} at "
                        f"gamma = {div[0]:g}). The papers' gamma = 2 never diverges.")
        unc = (sum(d.get("sptc_nt_znn", {}).get("cells_diverged", 0) for d in tv.values()),
               sum(d.get("sptc_nt_znn", {}).get("n_cells", 0) for d in tv.values()))
        parts = []
        for lab, name in TVG_LABELS[1:]:
            k = sum(d.get(lab, {}).get("cells_diverged", 0) for d in tv.values())
            n = sum(d.get(lab, {}).get("n_cells", 0) for d in tv.values())
            if n:
                pk = {t: tv.get(f"tau={t}|gamma=100", {}).get(lab, {}).get("rpeak_post_median", float("nan"))
                      for t in ("0.001", "0.01")}
                parts.append(f"{name}: {k}/{n} cells diverged, median peak ||E|| after t_c at gamma = 100: "
                             f"{pk['0.001']:.2g} (tau = 1e-3), {pk['0.01']:.2g} (tau = 1e-2)")
        if parts:
            msgs.append(f"Saturating the gain (uncapped: {unc[0]}/{unc[1]} cells diverged) removes divergence "
                        "(||E|| > 1e10) in every cell, even with rho = 3 > 2 (outside the Euler interval), so 'no "
                        "divergence' alone does not certify a cap; compare the post-t_c peaks: " + "; ".join(parts)
                        + ". At tau = 1e-2 the medians are dominated by the T_c/tau <= 3 cells, where no gain law "
                        "can converge by t_c.")
    ud = summ.get("units_disc", {}).get("settings", {}).get("euler|fixed_gamma|10")
    if ud:
        bt = ud["by_tau"].get("0.001") or next((v for v in ud["by_tau"].values() if isinstance(v, dict)), None)
        if bt:
            c = bt["cells"]
            nl = [a for a in c if a != "linear"]
            reach = {u: sum(1 for a in nl if c[a].get(u, {}).get("reached_0.001", 0) >= 0.5) for u in ("0.01", "1", "100")}
            dv = {u: sum(1 for a in nl if c[a].get(u, {}).get("diverged", 0) >= 0.5 * c[a].get(u, {}).get("n", 1))
                  for u in ("0.01", "1", "100")}
            msgs.append(f"Unit dependence (Euler, gamma = 10, tau = 1e-3): nonlinear activations reaching 1e-3 "
                        f"(of {len(nl)}): s=0.01: {reach['0.01']}, natural: {reach['1']}, s=100: {reach['100']}; "
                        f"diverged: {dv['0.01']} / {dv['1']} / {dv['100']}. Linear ZNN is unit-invariant. Every "
                        "discrete-time ranking of finite/fixed/predefined-time activations in (b) is therefore "
                        "conditional on the unit of E.")
    return msgs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--figdir", default=str(ROOT / "figures"))
    ap.add_argument("--tabdir", default=str(ROOT / "results" / "tables"))
    args = ap.parse_args(argv)
    res = Path(args.results)
    Path(args.figdir).mkdir(parents=True, exist_ok=True)
    Path(args.tabdir).mkdir(parents=True, exist_ok=True)
    summ = {"results": str(res)}
    ct = load_ct(res)
    disc_all = load_runner(res / "E3_disc")
    tcs = [load_runner(d) for d in sorted(res.glob("E3_tc_*")) if d.is_dir()]
    tc_all = pd.concat([t for t in tcs if not t.empty], ignore_index=True) if any(not t.empty for t in tcs) \
        else pd.DataFrame()
    vp = load_runner(res / "E3_vp")

    def natural(df):     # natural error unit only (the unit arm is analysed separately)
        out = df[np.isclose(df["unit"], 1.0)].copy() if (not df.empty and "unit" in df) else df.copy()
        out.attrs["results_dir"] = str(res)
        return out

    disc, tc = natural(disc_all), natural(tc_all)
    with np.errstate(all="ignore"):
        analyze_ct(ct, args.figdir, args.tabdir, summ)
        analyze_disc(disc, ct, args.figdir, args.tabdir, summ)
        analyze_tc(tc, args.figdir, args.tabdir, summ)
        analyze_units_disc(disc_all, args.tabdir, summ)
        analyze_units_tc(tc_all, summ)
        analyze_vp(vp, args.tabdir, summ)
        analyze_floor_reach([disc, tc], args.tabdir, summ)
        censoring_summary({"E3_disc": disc_all, "E3_tc": tc_all, "E3_vp": vp}, ct, summ)
        cost_summary({"E3_disc": disc_all, "E3_tc": tc_all, "E3_vp": vp}, ct, summ)
    summ["interpretation"] = interpret(summ)

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, float)):
            v = float(o)
            return None if math.isnan(v) else ("inf" if math.isinf(v) else v)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return o

    (res / "summary.json").write_text(json.dumps(clean(summ), indent=1))
    write_summary_md(summ, res / "SUMMARY.md")
    print(f"E3 analysis -> {res / 'summary.json'}, {res / 'SUMMARY.md'}, figures in {args.figdir}, "
          f"tables in {args.tabdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
