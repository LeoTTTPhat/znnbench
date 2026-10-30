#!/usr/bin/env python
"""Analysis of the robotics case study R (protocol: experiments/R_NOTES.md).

    python experiments/analyze_R.py --results results/mini/R --figdir figures/mini \
        --tabdir results/tables/mini

Reads <results>/R_off (off-path starts, 3 seeds x B = 4), <results>/R_on
(on-path starts, deterministic), <results>/R_units (error-unit sensitivity:
the R_off starts with E in mm, paired with R_off) and <results>/traces
(per-step joint-velocity traces of experiments/tasks/R_joint_traces.py), and
writes

  figures   fig_R_traces_<path>.pdf/png        ||E(t)|| (mm), median + IQR, robots x rates
            fig_R_traces_noise_<path>.pdf/png  the same under constant noise
            fig_R_bars_g<gamma>.pdf/png        peak joint speed + chattering by model x rate
            fig_R_mssre_rate.pdf/png           MSSRE (mm) vs control rate, noise-free / noisy
            fig_R_units.pdf/png                MSSRE vs rate with E in m (R_off) and in mm (R_units)
            fig_R_jointvel_<robot>_<path>.pdf/png  max_i |theta_dot_i|(t), models x gains
  tables    R_tracking.tex, R_actuation.tex, R_gamma100.tex, R_noise.tex, R_drift.tex, R_units.tex
  summary   <results>/summary.json, <results>/SUMMARY.md

Statistics: medians and IQR over the trajectories of a cell (12 at full
scale); bootstrap 95 % CIs (2000 resamples, seeded) of medians and of paired
ratios of medians (pairs = same seed and initial state); diverged trajectories
(+inf) are kept as worst-case values in medians, CIs and rankings; Kendall tau_b for
model-ranking agreement; empirical orders from the MSSRE-vs-tau medians with
consecutive-ratio (local slope) diagnostics, reported both as an LS fit over
all points and over the asymptotic range only.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import keys, style  # noqa: E402
from experiments.tasks import R_common as RC  # noqa: E402
from znnbench.runner import aggregate, load_traces  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
from matplotlib.lines import Line2D  # noqa: E402

# --------------------------------------------------------------------------
# entity group of the case study.  Slots are chosen so every entity keeps the
# colour it has in the global groups of experiments/keys.py:
#   ZNN linear      = 'model' znn (0) = 'act' linear (0)       -> blue
#   SBP implicit    = 'disc_implicit' implicit_euler_err (1)   -> orange
#   IEZNN           = 'model' ieznn (2)                        -> green
#   SBP explicit    = 'act' sign_bi_power (3)                  -> yellow
#   SPTC-NT-ZNN     = 'model' sptc_nt_znn (6) (= 'act' pvaf1)  -> purple
# --------------------------------------------------------------------------
keys.GROUPS.setdefault("r_model", ["znn_lin", "znn_sbp_ie", "ieznn", "znn_sbp", "_r4", "_r5",
                                   "sptc_nt_znn", "_r7"])
SHORT = {"znn_lin": "ZNN (linear)", "znn_sbp": "ZNN-SBP (explicit)", "znn_sbp_ie": "ZNN-SBP (implicit)",
         "sptc_nt_znn": "SPTC-NT-ZNN", "znn_pvaf1": "ZNN-PVAF1", "ieznn": "IEZNN"}
ROBOT_NAME = {"puma560": "PUMA 560", "panda": "Panda"}
PATH_NAME = {"circle": "circle", "lissajous": "Lissajous"}
RATE_SHADES = {100: style.SEQ_BLUE[4], 333: style.SEQ_BLUE[8], 1000: style.SEQ_BLUE[12]}
RATE_LS = {100: ":", 333: "--", 1000: "-"}
# secondary encoding of bars (the line-style analogue for filled marks): one
# hatch per entity, fixed like the colour
HATCH = {"znn_lin": "", "znn_sbp": "////", "znn_sbp_ie": "\\\\\\\\", "sptc_nt_znn": "xxxx",
         "znn_pvaf1": "xxxx", "ieznn": "...."}
N_BOOT = 2000
FLOOR_M = 1e-12          # round-off floor of the position error (m)


def sty(entry):
    grp = "r_model"
    key = "sptc_nt_znn" if entry == "znn_pvaf1" else entry
    return keys.style_in_group(grp, key)


def rate_of(tau):
    return int(round(1.0 / tau))


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------
_BIG = 1e300


def med_iqr(x):
    """(median, q25, q75) ignoring NaN; +inf (diverged / never reached) is kept
    as a value: it is mapped to a sentinel for the interpolation and back."""
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    y = np.clip(x, -_BIG, _BIG)
    q = np.quantile(y, [0.5, 0.25, 0.75])
    return tuple(float(np.inf) if v >= 0.5 * _BIG else (float(-np.inf) if v <= -0.5 * _BIG else float(v))
                 for v in q)


def _unbig(v):
    v = float(v)
    return np.inf if v >= 0.5 * _BIG else (-np.inf if v <= -0.5 * _BIG else v)


def boot_median_ci(x, seed=0, n=N_BOOT):
    """Percentile bootstrap CI of the median.  NaN (failed / undefined) is
    dropped; +inf (diverged, never reached) is KEPT as a worst-case value
    (sentinel), so censored trajectories are not silently removed."""
    x = np.asarray(x, float)
    x = np.clip(x[~np.isnan(x)], -_BIG, _BIG)
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    s = np.median(x[rng.integers(0, len(x), (n, len(x)))], axis=1)
    return (_unbig(np.quantile(s, 0.025)), _unbig(np.quantile(s, 0.975)))


def _ratio(ma, mb):
    ma, mb = np.asarray(ma, float), np.asarray(mb, float)
    ia, ib = ma >= 0.5 * _BIG, mb >= 0.5 * _BIG
    with np.errstate(divide="ignore", invalid="ignore"):
        r = ma / mb
    r = np.where(ia & ~ib, np.inf, r)
    r = np.where(ib & ~ia, 0.0, r)
    return np.where(ia & ib, np.nan, r)


def boot_ratio_ci(a, b, seed=0, n=N_BOOT):
    """Paired bootstrap of median(a) / median(b) (a[i], b[i] share seed and x0).
    Pairs with a NaN are dropped; +inf is kept as a worst-case sentinel (a
    diverged numerator gives ratio inf, a diverged denominator ratio 0)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~np.isnan(a) & ~np.isnan(b)
    a, b = np.clip(a[ok], -_BIG, _BIG), np.clip(b[ok], -_BIG, _BIG)
    if len(a) < 2:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n, len(a)))
    r = _ratio(np.median(a[idx], axis=1), np.median(b[idx], axis=1))
    point = float(_ratio(np.median(a), np.median(b)))
    return (point, float(np.nanquantile(r, 0.025)), float(np.nanquantile(r, 0.975)))


def kendall_b(x, y):
    """Kendall tau_b of two score vectors.  NaN entries are dropped; +inf
    (diverged) ranks as the worst value (ties among several inf)."""
    from scipy.stats import kendalltau
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~np.isnan(x) & ~np.isnan(y)
    if ok.sum() < 3:
        return (np.nan, np.nan, int(ok.sum()))
    r = kendalltau(np.clip(x[ok], -_BIG, _BIG), np.clip(y[ok], -_BIG, _BIG), variant="b")
    return (float(r.statistic), float(r.pvalue), int(ok.sum()))


def empirical_order(taus, vals, floor=FLOOR_M, tol=0.15):
    """Orders from medians: local slopes between consecutive tau, LS slope over
    all valid points, and the 'asymptotic' slope over the smallest-tau run of
    points whose consecutive local slopes agree within ``tol`` (at least the
    last pair).  tol = 0.15 is the tolerance of the order test in
    tests/test_problems_robot.py; a looser tolerance (0.3) let SPTC-NT-ZNN's
    drifting local slopes (1.53 -> 1.30, a mixture of a (gamma tau)^2-type band
    and the O(tau) sign band, asymptotic order 1) pass as asymptotic.  Points at/below the round-off floor or non-finite are dropped."""
    t, v = np.asarray(taus, float), np.asarray(vals, float)
    o = np.argsort(t)
    t, v = t[o], v[o]
    ok = np.isfinite(v) & (v > floor)
    t, v = t[ok], v[ok]
    res = {"taus": t.tolist(), "vals": v.tolist(), "local": [], "ls": np.nan, "asym": np.nan,
           "asym_points": 0, "flag": ""}
    if len(t) < 2:
        res["flag"] = "too few points above floor"
        return res
    lt, lv = np.log(t), np.log(v)
    local = np.diff(lv) / np.diff(lt)                # local[j]: between t[j] and t[j+1]
    res["local"] = local.tolist()
    res["ls"] = float(np.polyfit(lt, lv, 1)[0])
    # t is ascending, so the asymptotic end is index 0 (smallest tau).  Extend the
    # run of consecutive local slopes that agree within tol from there.
    end = len(t)                                     # points [0, end) are asymptotic
    for j in range(1, len(local)):
        if abs(local[j] - local[j - 1]) > tol:
            end = j + 1                              # local[0..j-1] agree -> points 0..j
            break
    pts = slice(0, end)
    res["asym"] = float(np.polyfit(lt[pts], lv[pts], 1)[0])
    res["asym_points"] = end
    if end < len(t):
        res["flag"] = "pre-asymptotic at large tau (local slopes disagree)"
    if np.any(local < 0.2):
        res["flag"] = (res["flag"] + "; " if res["flag"] else "") + "no decrease with tau (floor/instability)"
    return res


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load(results):
    res = Path(results)
    out = {}
    for name in ("R_off", "R_on", "R_units"):
        d = res / name
        if not (d / "tasks.jsonl").exists():
            continue
        df = aggregate(d)
        if df.empty:
            continue
        # censoring: a diverged trajectory gets NaN wrapper statistics (and inf
        # mssre / peak_rate).  Medians must not silently drop it, so the
        # "larger is worse" statistics are set to their worst value; alt_active
        # stays NaN (undefined) and diverged / failed counts are reported per cell.
        div = (df.status == "diverged").to_numpy()
        if div.any():
            for col in ("qd_util", "qd_util_ss", "qdd_peak", "qdd_peak_ss", "qdd_util", "qlim_viol",
                        "peak_rate", "peak_rate_ss", "mssre", "drift"):
                if col in df:
                    df.loc[div, col] = np.inf
            if "qd_viol_frac" in df:
                df.loc[div, "qd_viol_frac"] = 1.0
        df["entry"] = [RC.entry_of(m, dk) for m, dk in zip(df.model_label, df.disc_key)]
        df["robot"] = df.problem_key.str.replace("robot_", "", regex=False)
        df["path"] = df.problem_kw_path
        df["noise"] = df.noise_key.astype(str)
        df["rate"] = [rate_of(t) for t in df.tau]
        df["gamma"] = df.gamma.astype(float).round(6)
        df["mssre_mm"] = df.mssre.astype(float) * 1e3
        df["pair"] = df.seed.astype(int) * 1000 + df.x0_index.astype(int)
        out[name] = df
    return out


def cell(df, **sel):
    m = np.ones(len(df), bool)
    for k, v in sel.items():
        m &= (df[k] == v).to_numpy()
    return df[m]


# --------------------------------------------------------------------------
# formatting
# --------------------------------------------------------------------------
def fnum(x, sig=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    if math.isinf(x):
        return r"$\infty$"
    ax = abs(x)
    if ax == 0:
        return "0"
    if 1e-2 <= ax < 1e3:
        d = max(0, sig - 1 - int(math.floor(math.log10(ax))))
        return f"{x:.{d}f}"
    e = int(math.floor(math.log10(ax)))
    m = x / 10 ** e
    return rf"${m:.{sig - 1}f}\times10^{{{e}}}$"


def fplain(x, sig=3):
    if x is None or (isinstance(x, float) and (math.isnan(x))):
        return "nan"
    return f"{x:.{sig}g}"


def write_tex(path, header, rows, caption, label, colspec):
    lines = [r"\begin{table}[t]", r"\centering", r"\footnotesize",
             rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             rf"\begin{{tabular}}{{{colspec}}}", r"\toprule"]
    lines += [" & ".join(h) + r" \\" for h in header]
    lines.append(r"\midrule")
    prev = None
    for r in rows:
        if r == "MID":
            lines.append(r"\midrule")
            continue
        lines.append(" & ".join(r) + r" \\")
        prev = r
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    Path(path).write_text("\n".join(lines) + "\n")
    return prev


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def _entries(df):
    present = set(df.entry.unique())
    return [e for e in RC.DISPLAY_ORDER if e in present]


def _trace_matrix(results, df):
    """{(run_id, traj): (t, residual)} for the rows of df (loads npz once per chunk)."""
    out, cache = {}, {}
    base = Path(results) / "R_off" / "results"
    for f, rid, traj in zip(df.traces_file, df.run_id, df.traj):
        if not isinstance(f, str) or not f:
            continue
        key = (f, rid)
        if key not in cache:
            p = base / f
            if not p.exists():
                continue
            cache[key] = load_traces(p, rid)
        tr = cache[key]
        out[(rid, int(traj))] = (np.asarray(tr["t"]), np.asarray(tr["residual"])[:, int(traj)])
    return out


def _top_legend(fig, handles, labels, ncol, top, title=None):
    """Figure legend sitting directly on top of the axes area [0, top]."""
    fig.legend(handles, labels, loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, top),
               columnspacing=1.2, handletextpad=0.5, title=title, title_fontsize=7)


def _plain_log_ticks(ax):
    from matplotlib.ticker import FuncFormatter, LogLocator
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=7))
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: (f"{v:g}" if 1e-2 <= v <= 100.5 else f"$10^{{{int(round(np.log10(v)))}}}$") if v > 0 else ""))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))


def fig_traces(results, df, path, noise, gamma, figdir):
    rob = [r for r in ("puma560", "panda") if r in set(df.robot)]
    rates = sorted(df.rate.unique())
    sub = cell(df, path=path, noise=noise, gamma=gamma)
    if sub.empty:
        return None
    ents = _entries(sub)
    tm = _trace_matrix(results, sub)
    fig, axes = plt.subplots(len(rob), len(rates), figsize=(style.DOUBLE, 1.75 * len(rob) + 0.45),
                             sharex=True, sharey=True, squeeze=False)
    for i, r in enumerate(rob):
        for j, rt in enumerate(rates):
            ax = axes[i, j]
            for e in ents:
                c = cell(sub, robot=r, rate=rt, entry=e)
                arrs = [tm[(rid, int(tr))] for rid, tr in zip(c.run_id, c.traj) if (rid, int(tr)) in tm]
                if not arrs:
                    continue
                t = arrs[0][0]
                Y = np.vstack([a[1] for a in arrs]).astype(float) * 1e3
                Y = np.where(np.isfinite(Y), np.maximum(Y, FLOOR_M * 1e3), np.nan)
                q25, q50, q75 = np.nanquantile(Y, [0.25, 0.5, 0.75], axis=0)
                s = sty(e)
                ax.fill_between(t, q25, q75, color=s["color"], alpha=0.18, lw=0)
                ax.plot(t, q50, color=s["color"], ls=s["linestyle"], marker=s["marker"],
                        markevery=max(1, len(t) // 8), lw=0.9, ms=3, label=SHORT.get(e, e))
            ax.set_yscale("log")
            _plain_log_ticks(ax)
            if i == 0:
                ax.set_title(f"{rt:g} Hz ($\\tau$ = {1.0 / rt:.0e} s)".replace("e-0", "e-"))
            if j == 0:
                ax.set_ylabel(f"{ROBOT_NAME[r]}\n" + r"$\|E(t)\|_2$ (mm)")
            if i == len(rob) - 1:
                ax.set_xlabel("t (s)")
    top = 0.9
    fig.tight_layout(rect=(0, 0, 1, top), h_pad=0.6, w_pad=0.6)
    h, lab = axes[0, 0].get_legend_handles_labels()
    ttl = (f"{PATH_NAME[path]} path, $\\gamma$ = {gamma:g}, "
           + ("noise-free" if noise == "none" else f"constant noise {RC.NOISE_LEVEL:g} m/s")
           + ": median and IQR over off-path starts")
    _top_legend(fig, h, lab, len(lab), top, title=ttl)
    nm = f"fig_R_traces_{path}" if noise == "none" else f"fig_R_traces_noise_{path}"
    return style.savefig(fig, nm)


PAPER_BARS = {10.0}      # fig_R_bars_g10: Supplementary Material figure, drawn at the paper's text width


def fig_bars_paper(df, gamma, figdir):
    """fig_bars drawn at style.TEXTWIDTH (print size 1:1): rates as plain numbers with the unit in the
    axis label, legend below."""
    sub = cell(df, noise="none", gamma=gamma)
    if sub.empty:
        return None
    combos = [(r, p) for r in ("puma560", "panda") for p in ("circle", "lissajous")
              if not cell(sub, robot=r, path=p).empty]
    ents = _entries(sub)
    rates = sorted(sub.rate.unique())
    with style.print_rc():
        fig, axes = plt.subplots(2, len(combos), figsize=(style.TEXTWIDTH, 3.75), squeeze=False, sharey="row",
                                 constrained_layout=True, gridspec_kw={"height_ratios": [1.15, 1.0]})
        fig.get_layout_engine().set(w_pad=0.02, h_pad=0.03, wspace=0.02, hspace=0.05)
        w = 0.8 / max(len(ents), 1)
        ymax = np.nanmax(sub.peak_rate.replace(np.inf, np.nan).astype(float))
        for j, (r, p) in enumerate(combos):
            qd_max, _ = RC.joint_limits(r)
            for row, (col, ylab) in enumerate([("peak_rate", r"peak $\|\dot\theta\|_\infty$ (rad/s)"),
                                               ("alt_active", "chattering\n(sign-alternation rate)")]):
                ax = axes[row, j]
                for k, e in enumerate(ents):
                    xs, ys, lo, hi, ss = [], [], [], [], []
                    for ir, rt in enumerate(rates):
                        c = cell(sub, robot=r, path=p, entry=e, rate=rt)
                        m, q1, q3 = med_iqr(c[col].astype(float))
                        xs.append(ir + (k - (len(ents) - 1) / 2) * w)
                        ys.append(m)
                        lo.append(m - q1 if np.isfinite(m) else 0)
                        hi.append(q3 - m if np.isfinite(m) else 0)
                        ss.append(med_iqr(c.peak_rate_ss.astype(float))[0])
                    s_ = sty(e)
                    with plt.rc_context({"hatch.linewidth": 0.45, "hatch.color": "white"}):
                        ax.bar(xs, ys, width=w * 0.9, color=s_["color"], edgecolor=style.INK2, linewidth=0.3,
                               hatch=HATCH.get(e, ""), label=SHORT.get(e, e), zorder=2)
                    ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=style.INK, elinewidth=0.6, capsize=1,
                                zorder=3)
                    if row == 0:
                        ax.plot(xs, ss, ls="none", marker="o", ms=3.0, mfc="white", mec=style.INK, mew=0.6,
                                zorder=4, label="_ss")
                if row == 0:
                    ax.set_yscale("log")
                    ax.set_ylim(0.1, max(10.0, 3 * ymax))
                    _plain_log_ticks(ax)
                    ax.axhline(qd_max.min(), color=style.INK2, ls="--", lw=0.9, zorder=1)
                    if qd_max.max() > qd_max.min():
                        ax.axhline(qd_max.max(), color=style.INK2, ls=":", lw=0.9, zorder=1)
                    ax.set_title(f"{ROBOT_NAME[r]}, {PATH_NAME[p]}", fontsize=8)
                else:
                    ax.set_ylim(0, 1.05)
                    ax.set_xlabel("control rate (Hz)")
                ax.set_xticks(range(len(rates)))
                ax.set_xticklabels([f"{rt:g}" for rt in rates])
                ax.grid(axis="x", visible=False)
                ax.tick_params(axis="x", length=0)
                if j == 0:
                    ax.set_ylabel(ylab)
        h, lab = axes[0, 0].get_legend_handles_labels()
        keep = [i for i, l in enumerate(lab) if l != "_ss"]
        h, lab = [h[i] for i in keep], [lab[i] for i in keep]
        h += [Line2D([], [], ls="none", marker="o", ms=3.0, mfc="white", mec=style.INK, mew=0.6),
              Line2D([], [], color=style.INK2, ls="--", lw=0.9), Line2D([], [], color=style.INK2, ls=":", lw=0.9)]
        lab += ["steady-window peak (last 25%)", "speed limit: PUMA 2 (nominal), Panda J1\u20134 2.175 rad/s",
                "speed limit: Panda J5\u20137 2.61 rad/s"]
        fig.legend(h, lab, loc="outside lower center", ncol=2, frameon=False, columnspacing=1.4,
                   handletextpad=0.5)
        return style.savefig(fig, f"fig_R_bars_g{int(gamma)}")


def fig_bars(df, gamma, figdir):
    if float(gamma) in PAPER_BARS:
        return fig_bars_paper(df, gamma, figdir)
    sub = cell(df, noise="none", gamma=gamma)
    if sub.empty:
        return None
    combos = [(r, p) for r in ("puma560", "panda") for p in ("circle", "lissajous")
              if not cell(sub, robot=r, path=p).empty]
    ents = _entries(sub)
    rates = sorted(sub.rate.unique())
    fig, axes = plt.subplots(2, len(combos), figsize=(style.DOUBLE, 3.7), squeeze=False, sharey="row")
    w = 0.8 / max(len(ents), 1)
    ymax = np.nanmax(sub.peak_rate.replace(np.inf, np.nan).astype(float))
    for j, (r, p) in enumerate(combos):
        qd_max, _ = RC.joint_limits(r)
        for row, (col, ylab) in enumerate([("peak_rate", r"peak $\|\dot\theta\|_\infty$ (rad/s)"),
                                           ("alt_active", "chattering\n(sign-alternation rate)")]):
            ax = axes[row, j]
            for k, e in enumerate(ents):
                xs, ys, lo, hi, ss = [], [], [], [], []
                for ir, rt in enumerate(rates):
                    c = cell(sub, robot=r, path=p, entry=e, rate=rt)
                    m, q1, q3 = med_iqr(c[col].astype(float))
                    xs.append(ir + (k - (len(ents) - 1) / 2) * w)
                    ys.append(m)
                    lo.append(m - q1 if np.isfinite(m) else 0)
                    hi.append(q3 - m if np.isfinite(m) else 0)
                    ss.append(med_iqr(c.peak_rate_ss.astype(float))[0])
                s = sty(e)
                with plt.rc_context({"hatch.linewidth": 0.35, "hatch.color": "white"}):
                    ax.bar(xs, ys, width=w * 0.92, color=s["color"], edgecolor=style.INK2, linewidth=0.3,
                           hatch=HATCH.get(e, ""), label=SHORT.get(e, e), zorder=2)
                ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=style.INK, elinewidth=0.5, capsize=1,
                            zorder=3)
                if row == 0:
                    ax.plot(xs, ss, ls="none", marker="o", ms=2.6, mfc="white", mec=style.INK, mew=0.6, zorder=4,
                            label="_ss")
            if row == 0:
                ax.set_yscale("log")
                ax.set_ylim(0.1, max(10.0, 3 * ymax))
                _plain_log_ticks(ax)
                ax.axhline(qd_max.min(), color=style.INK2, ls="--", lw=0.8, zorder=1)
                if qd_max.max() > qd_max.min():
                    ax.axhline(qd_max.max(), color=style.INK2, ls=":", lw=0.8, zorder=1)
                ax.set_title(f"{ROBOT_NAME[r]}, {PATH_NAME[p]}")
            else:
                ax.set_ylim(0, 1.05)
            ax.set_xticks(range(len(rates)))
            ax.set_xticklabels([f"{rt:g} Hz" for rt in rates])
            ax.grid(axis="x", visible=False)
            if j == 0:
                ax.set_ylabel(ylab)
    top = 0.86
    fig.tight_layout(rect=(0, 0, 1, top), h_pad=0.8, w_pad=0.5)
    h, lab = axes[0, 0].get_legend_handles_labels()
    keep = [i for i, l in enumerate(lab) if l != "_ss"]
    h, lab = [h[i] for i in keep], [lab[i] for i in keep]
    h += [Line2D([], [], ls="none", marker="o", ms=2.6, mfc="white", mec=style.INK, mew=0.6),
          Line2D([], [], color=style.INK2, ls="--", lw=0.8), Line2D([], [], color=style.INK2, ls=":", lw=0.8)]
    lab += ["steady-window peak (last 25%)", "speed limit (rad/s): PUMA 2 (nominal), Panda J1-4 2.175",
            "speed limit (rad/s): Panda J5-7 2.61"]
    _top_legend(fig, h, lab, 4, top, title=f"$\\gamma$ = {gamma:g}, noise-free, off-path starts "
                f"(bars: median, whiskers: IQR)")
    return style.savefig(fig, f"fig_R_bars_g{int(gamma)}")


def fig_mssre_rate(df, gamma, figdir):
    sub = cell(df, gamma=gamma)
    combos = [(r, p) for r in ("puma560", "panda") for p in ("circle", "lissajous")
              if not cell(sub, robot=r, path=p).empty]
    noises = [n for n in ("none", "constant") if n in set(sub.noise)]
    if not combos or not noises:
        return None
    ents = _entries(sub)
    fig, axes = plt.subplots(len(noises), len(combos), figsize=(style.DOUBLE, 1.7 * len(noises) + 0.5),
                             sharex=True, sharey=True, squeeze=False)
    for i, nz in enumerate(noises):
        for j, (r, p) in enumerate(combos):
            ax = axes[i, j]
            for e in ents:
                rows = [(rt, *med_iqr(cell(sub, robot=r, path=p, entry=e, noise=nz, rate=rt).mssre_mm))
                        for rt in sorted(sub.rate.unique())]
                rows = [x for x in rows if np.isfinite(x[1])]
                if not rows:
                    continue
                rt, m, q1, q3 = map(np.array, zip(*rows))
                s = sty(e)
                ax.errorbar(rt, m, yerr=[m - q1, q3 - m], color=s["color"], ls=s["linestyle"],
                            marker=s["marker"], ms=3.5, lw=1.0, elinewidth=0.5, capsize=1.5,
                            label=SHORT.get(e, e))
            if nz == "constant":
                ax.axhline(np.sqrt(3) * RC.NOISE_LEVEL / gamma * 1e3, color=style.GREY, ls=(0, (1, 1)), lw=0.8,
                           label=r"CT bias of linear ZNN, $\sqrt{3}\,\nu/\gamma$")
            ax.set_xscale("log")
            ax.set_yscale("log")
            _plain_log_ticks(ax)
            if i == 0:
                ax.set_title(f"{ROBOT_NAME[r]}, {PATH_NAME[p]}")
            if j == 0:
                ax.set_ylabel(("noise-free" if nz == "none" else f"const. noise {RC.NOISE_LEVEL:g} m/s")
                              + "\nMSSRE (mm)")
            if i == len(noises) - 1:
                ax.set_xlabel("control rate (Hz)")
            ax.set_xticks([100, 333, 1000])
            ax.set_xticklabels(["100", "333", "1000"])
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    top = 0.9
    fig.tight_layout(rect=(0, 0, 1, top), h_pad=0.6, w_pad=0.6)
    h, lab = axes[-1, 0].get_legend_handles_labels()
    o = sorted(range(len(lab)), key=lambda i: lab[i].startswith("CT"))
    h, lab = [h[i] for i in o], [lab[i] for i in o]
    _top_legend(fig, h, lab, len(lab), top, title=f"$\\gamma$ = {gamma:g}, off-path starts "
                f"(median, whiskers: IQR)")
    return style.savefig(fig, "fig_R_mssre_rate")


def fig_jointvel(npz_path, figdir):
    z = np.load(npz_path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    robot = meta["robot"].replace("robot_", "")
    path = meta["path"]
    qd_max, _ = RC.joint_limits(robot)
    ents = [e for e in RC.DISPLAY_ORDER if any(rr["entry"] == e for rr in meta["runs"])]
    gam = meta["gammas"]
    fig, axes = plt.subplots(len(gam), len(ents), figsize=(style.DOUBLE, 1.45 * len(gam) + 0.5),
                             sharex=True, sharey="row", squeeze=False)
    for gi, g in enumerate(gam):
        for j, e in enumerate(ents):
            ax = axes[gi, j]
            for tau in meta["taus"]:
                tag = f"{e}|none|{tau:g}"
                if f"{tag}__qd" not in z:
                    continue
                qd = z[f"{tag}__qd"][:, gi, :]
                t = z[f"{tag}__t"][:-1]
                y = np.abs(qd).max(axis=1)
                rt = rate_of(tau)
                ax.plot(t[:len(y)], np.maximum(y, 1e-6), color=RATE_SHADES.get(rt, style.GREY),
                        ls=RATE_LS.get(rt, "-"), lw=0.6, label=f"{rt:g} Hz", rasterized=True)
            ax.axhline(qd_max.min(), color=style.CAT[7], ls="--", lw=0.8)
            ax.set_yscale("log")
            _plain_log_ticks(ax)
            if gi == 0:
                ax.set_title(SHORT.get(e, e), fontsize=7.5)
            if j == 0:
                ax.set_ylabel(f"$\\gamma$ = {g:g}\n" + r"$\|\dot\theta(t)\|_\infty$ (rad/s)")
            if gi == len(gam) - 1:
                ax.set_xlabel("t (s)")
    top = 0.88
    fig.tight_layout(rect=(0, 0, 1, top), h_pad=0.6, w_pad=0.4)
    h, lab = axes[0, 0].get_legend_handles_labels()
    h.append(Line2D([], [], color=style.CAT[7], ls="--", lw=0.8))
    lab.append(f"joint-speed limit {qd_max.min():g} rad/s" + (" (nominal)" if robot == "puma560" else ""))
    _top_legend(fig, h, lab, len(lab), top,
                title=f"{ROBOT_NAME[robot]}, {PATH_NAME[path]} path, one off-path start (seed {meta['seed']}, "
                      f"x0 #{meta['x0_index']}), noise-free: commanded joint speed")
    return style.savefig(fig, f"fig_R_jointvel_{robot}_{path}")


# --------------------------------------------------------------------------
# main analysis
# --------------------------------------------------------------------------
def analyse(results, figdir, tabdir):
    results, figdir, tabdir = Path(results), Path(figdir), Path(tabdir)
    figdir.mkdir(parents=True, exist_ok=True)
    tabdir.mkdir(parents=True, exist_ok=True)
    style.FIG_DIR = figdir                    # style.savefig writes <figdir>/<name>.pdf/.png
    data = load(results)
    if "R_off" not in data:
        raise SystemExit(f"no R_off results under {results}")
    off = data["R_off"]
    on = data.get("R_on")
    S = {"scale": json.loads((results / "scale.json").read_text())["scale"] if (results / "scale.json").exists()
         else "unknown"}
    # ---- bookkeeping -------------------------------------------------------
    S["status_off"] = off.status.value_counts().to_dict()
    S["status_on"] = on.status.value_counts().to_dict() if on is not None else {}
    S["n_traj_per_cell_off"] = int(off.groupby(["robot", "path", "entry", "rate", "gamma", "noise"]).size().median())
    S["wall_per_step_us"] = {e: float(np.median(g.wall_per_step) * 1e6) for e, g in off.groupby("entry")}
    S["wall_per_step_us_all"] = float(np.median(off.wall_per_step) * 1e6)
    S["peak_rss_mb_max"] = float(np.nanmax(off.peak_rss_mb))
    S["steps_total_off"] = int(off.drop_duplicates("run_id").steps.sum())
    S["wall_total_s_off"] = float(off.drop_duplicates("run_id").wall_recorded.sum())
    if on is not None:
        S["steps_total_on"] = int(on.drop_duplicates("run_id").steps.sum())
        S["wall_total_s_on"] = float(on.drop_duplicates("run_id").wall_recorded.sum())
    chk = off[np.isfinite(off.peak_rate.astype(float)) & np.isfinite(off.qd_peak.astype(float))]
    S["check_qd_peak_equals_peak_rate_maxrel"] = float(
        np.max(np.abs(chk.qd_peak - chk.peak_rate) / np.maximum(chk.peak_rate, 1e-300))) if len(chk) else None
    ents = _entries(off)
    rates = sorted(off.rate.unique())
    robots = [r for r in ("puma560", "panda") if r in set(off.robot)]
    paths = [p for p in ("circle", "lissajous") if p in set(off.path)]

    # ---- per-cell summaries --------------------------------------------------
    cells = []
    for (r, p, e, rt, g, nz), c in off.groupby(["robot", "path", "entry", "rate", "gamma", "noise"]):
        qd_max, _ = RC.joint_limits(r)
        d = {"robot": r, "path": p, "entry": e, "rate": int(rt), "gamma": float(g), "noise": nz,
             "n": int(len(c)), "n_ok": int((c.status == "ok").sum()),
             "n_diverged": int((c.status == "diverged").sum()),
             "n_failed": int((~c.status.isin(["ok", "diverged"])).sum())}
        for col in ("mssre_mm", "peak_rate", "peak_rate_ss", "rate_alt", "alt_active", "chatter_vec",
                    "qd_util", "qd_util_ss", "qd_viol_frac", "qdd_peak", "qdd_peak_ss", "qdd_util",
                    "qlim_viol", "t_eps_0.001", "t_eps_0.0001", "peak_xdot", "drift"):
            if col in c:
                m, q1, q3 = med_iqr(c[col].astype(float))
                d[col] = {"median": m, "q25": q1, "q75": q3}
        d["mssre_mm"]["ci95"] = boot_median_ci(c.mssre_mm.astype(float), seed=1)
        d["peak_rate"]["ci95"] = boot_median_ci(c.peak_rate.astype(float), seed=2)
        d["frac_over_vel_limit"] = float(np.mean(c.qd_util.astype(float) > 1.0))
        d["frac_over_vel_limit_ss"] = float(np.mean(c.qd_util_ss.astype(float) > 1.0))
        cells.append(d)
    S["cells_off"] = cells

    def C(**k):
        for d in cells:
            if all(d[a] == b for a, b in k.items()):
                return d
        return None

    def M(col, **k):
        d = C(**k)
        return np.nan if d is None or col not in d else d[col]["median"]

    # ---- headline comparisons (paired bootstrap) -----------------------------
    head = {"implicit_vs_explicit_sbp": [], "ieznn_vs_znn_const_noise": [], "noise_bias_znn": []}
    for r in robots:
        for p in paths:
            for rt in rates:
                for g in sorted(off.gamma.unique()):
                    a = cell(off, robot=r, path=p, rate=rt, gamma=g, noise="none", entry="znn_sbp_ie")
                    b = cell(off, robot=r, path=p, rate=rt, gamma=g, noise="none", entry="znn_sbp")
                    if len(a) and len(b):
                        A = a.set_index("pair").sort_index()
                        Bq = b.set_index("pair").sort_index()
                        common = A.index.intersection(Bq.index)
                        head["implicit_vs_explicit_sbp"].append({
                            "robot": r, "path": p, "rate": int(rt), "gamma": float(g),
                            "mssre_ratio_impl_over_expl": boot_ratio_ci(A.loc[common].mssre_mm, Bq.loc[common].mssre_mm, 11),
                            "peak_rate_ss_ratio": boot_ratio_ci(A.loc[common].peak_rate_ss, Bq.loc[common].peak_rate_ss, 12),
                            "alt_active_impl": med_iqr(A.alt_active), "alt_active_expl": med_iqr(Bq.alt_active),
                            "rate_alt_impl": med_iqr(A.rate_alt), "rate_alt_expl": med_iqr(Bq.rate_alt)})
                    a = cell(off, robot=r, path=p, rate=rt, gamma=g, noise="constant", entry="ieznn")
                    b = cell(off, robot=r, path=p, rate=rt, gamma=g, noise="constant", entry="znn_lin")
                    if len(a) and len(b):
                        A = a.set_index("pair").sort_index()
                        Bq = b.set_index("pair").sort_index()
                        common = A.index.intersection(Bq.index)
                        head["ieznn_vs_znn_const_noise"].append({
                            "robot": r, "path": p, "rate": int(rt), "gamma": float(g),
                            "mssre_ratio_ieznn_over_znn": boot_ratio_ci(A.loc[common].mssre_mm, Bq.loc[common].mssre_mm, 13),
                            "ieznn_mssre_mm": med_iqr(A.mssre_mm), "znn_mssre_mm": med_iqr(Bq.mssre_mm)})
                    a = cell(off, robot=r, path=p, rate=rt, gamma=g, noise="constant", entry="znn_lin")
                    if len(a):
                        head["noise_bias_znn"].append({
                            "robot": r, "path": p, "rate": int(rt), "gamma": float(g),
                            "mssre_mm": med_iqr(a.mssre_mm),
                            "ct_prediction_mm": float(np.sqrt(3) * RC.NOISE_LEVEL / g * 1e3)})
    S["headline"] = head

    # ---- empirical orders (MSSRE vs tau, fixed gamma) --------------------------
    orders = []
    for (r, p, e, g, nz), c in off.groupby(["robot", "path", "entry", "gamma", "noise"]):
        taus = sorted(c.tau.unique())
        vals = [med_iqr(cell(c, tau=t).mssre)[0] for t in taus]
        od = empirical_order(taus, vals)
        orders.append({"robot": r, "path": p, "entry": e, "gamma": float(g), "noise": nz, **od})
    S["orders"] = orders

    # ---- Kendall tau_b rankings ---------------------------------------------------
    kt = []
    g0 = 10.0
    for r in robots:
        for p in paths:
            ms = {rt: [M("mssre_mm", robot=r, path=p, entry=e, rate=rt, gamma=g0, noise="none") for e in ents]
                  for rt in rates}
            pk = {rt: [M("peak_rate", robot=r, path=p, entry=e, rate=rt, gamma=g0, noise="none") for e in ents]
                  for rt in rates}
            if len(rates) >= 2:
                kt.append({"what": "MSSRE ranking, slowest vs fastest rate", "robot": r, "path": p,
                           "rates": [int(rates[0]), int(rates[-1])],
                           "tau_b": kendall_b(ms[rates[0]], ms[rates[-1]])})
            for rt in rates:
                kt.append({"what": "MSSRE ranking vs peak-joint-speed ranking", "robot": r, "path": p,
                           "rate": int(rt), "tau_b": kendall_b(ms[rt], pk[rt])})
    for p in paths:
        for rt in rates:
            if len(robots) == 2:
                a = [M("mssre_mm", robot=robots[0], path=p, entry=e, rate=rt, gamma=g0, noise="none") for e in ents]
                b = [M("mssre_mm", robot=robots[1], path=p, entry=e, rate=rt, gamma=g0, noise="none") for e in ents]
                kt.append({"what": "MSSRE ranking, PUMA vs Panda", "path": p, "rate": int(rt),
                           "tau_b": kendall_b(a, b)})
    S["kendall"] = kt
    S["entries"] = ents

    # ---- on-path drift ------------------------------------------------------------
    drift = []
    if on is not None:
        for (r, p, e, rt, g, nz), c in on.groupby(["robot", "path", "entry", "rate", "gamma", "noise"]):
            drift.append({"robot": r, "path": p, "entry": e, "rate": int(rt), "gamma": float(g), "noise": nz,
                          "drift_rad": med_iqr(c.drift.astype(float))[0],
                          "mssre_mm": med_iqr(c.mssre_mm.astype(float))[0],
                          "peak_rate": med_iqr(c.peak_rate.astype(float))[0],
                          "qlim_viol": med_iqr(c.qlim_viol.astype(float))[0],
                          "status": ",".join(sorted(set(c.status)))})
    S["drift_on"] = drift

    # ---- error-unit sensitivity (R_units: E in mm vs R_off: E in m) -----------------
    S["units"] = units_analysis(off, data.get("R_units"), ents, rates, robots, paths)

    # ---- tables -----------------------------------------------------------------------
    tabs = write_tables(S, off, on, ents, rates, robots, paths, tabdir)
    S["tables"] = [str(t) for t in tabs]

    # ---- figures --------------------------------------------------------------------
    figs = []
    for p in paths:
        figs.append(fig_traces(results, off, p, "none", 10.0, figdir))
        if "constant" in set(off.noise):
            figs.append(fig_traces(results, off, p, "constant", 10.0, figdir))
    for g in sorted(off.gamma.unique()):
        figs.append(fig_bars(off, g, figdir))
    figs.append(fig_mssre_rate(off, 10.0, figdir))
    if data.get("R_units") is not None:
        figs.append(fig_units(off, data["R_units"], 10.0, figdir))
    trd = results / "traces" / "results"
    trace_checks = []
    if trd.exists():
        for f in sorted(trd.glob("*.npz")):
            figs.append(fig_jointvel(f, figdir))
            trace_checks.extend(trace_consistency(f, off))
    S["trace_consistency"] = trace_checks
    S["figures"] = [str(f) for f in figs if f is not None]

    write_summary(S, results)
    return S


def units_analysis(off, units, ents, rates, robots, paths):
    """Compare R_units (error in mm) with R_off (error in m), paired by seed and
    x0 (same initial states), noise-free.  Returns a dict for summary.json."""
    if units is None or units.empty:
        return {}
    U = {"scale": RC.UNIT_SCALE, "cells": [], "invariance": None, "kendall": [], "impl_vs_expl_mm": []}
    base = off[off.noise == "none"]
    keycols = ["robot", "path", "entry", "rate", "gamma"]
    for (r, p, e, rt, g), c in units.groupby(keycols):
        b = cell(base, robot=r, path=p, entry=e, rate=rt, gamma=g)
        if b.empty:
            continue
        A, Bq = c.set_index("pair").sort_index(), b.set_index("pair").sort_index()
        com = A.index.intersection(Bq.index)
        d = {"robot": r, "path": p, "entry": e, "rate": int(rt), "gamma": float(g), "n_pairs": int(len(com)),
             "n_diverged_mm": int((c.status == "diverged").sum()),
             "n_failed_mm": int((~c.status.isin(["ok", "diverged"])).sum()),
             "mssre_ratio_mm_over_m": boot_ratio_ci(A.loc[com].mssre_mm, Bq.loc[com].mssre_mm, 21)}
        for col in ("mssre_mm", "peak_rate", "peak_rate_ss", "alt_active", "qd_viol_frac", "qlim_viol"):
            if col in A and col in Bq:
                d[col] = {"m_units": med_iqr(Bq.loc[com, col].astype(float)),
                          "mm_units": med_iqr(A.loc[com, col].astype(float))}
        d["mssre_mm"]["mm_units_ci95"] = boot_median_ci(A.loc[com].mssre_mm.astype(float), seed=22)
        if e == "znn_lin":
            a_, b_ = A.loc[com].mssre_mm.astype(float), Bq.loc[com].mssre_mm.astype(float)
            pk_a, pk_b = A.loc[com].peak_rate.astype(float), Bq.loc[com].peak_rate.astype(float)
            d["max_rel_diff_mssre"] = float(np.max(np.abs(a_ - b_) / np.maximum(np.abs(b_), 1e-300)))
            d["max_rel_diff_peak_rate"] = float(np.max(np.abs(pk_a - pk_b) / np.maximum(np.abs(pk_b), 1e-300)))
        U["cells"].append(d)
    inv = [d for d in U["cells"] if d["entry"] == "znn_lin"]
    if inv:
        U["invariance"] = {"what": "linear ZNN, E in mm vs E in m (must agree to round-off)",
                           "max_rel_diff_mssre": max(d["max_rel_diff_mssre"] for d in inv),
                           "max_rel_diff_peak_rate": max(d["max_rel_diff_peak_rate"] for d in inv),
                           "n_cells": len(inv)}
    # paired implicit / explicit SBP ratio in mm units
    for r in robots:
        for p in paths:
            for rt in rates:
                for g in sorted(units.gamma.unique()):
                    a = cell(units, robot=r, path=p, rate=rt, gamma=g, entry="znn_sbp_ie")
                    b = cell(units, robot=r, path=p, rate=rt, gamma=g, entry="znn_sbp")
                    if len(a) and len(b):
                        A, Bq = a.set_index("pair").sort_index(), b.set_index("pair").sort_index()
                        com = A.index.intersection(Bq.index)
                        U["impl_vs_expl_mm"].append({
                            "robot": r, "path": p, "rate": int(rt), "gamma": float(g),
                            "mssre_ratio_impl_over_expl": boot_ratio_ci(A.loc[com].mssre_mm,
                                                                        Bq.loc[com].mssre_mm, 23),
                            "alt_active_impl": med_iqr(A.alt_active), "alt_active_expl": med_iqr(Bq.alt_active)})
    # 5-model MSSRE ranking with E in m vs E in mm.  Entries absent from R_units
    # (IEZNN) are linear in E, hence exactly unit-invariant (verified on the
    # linear-ZNN control): their R_off medians are used in both rankings.
    for g in sorted(units.gamma.unique()):
        for r in robots:
            for p in paths:
                for rt in rates:
                    m_rank, mm_rank = [], []
                    for e in ents:
                        bm = med_iqr(cell(base, robot=r, path=p, entry=e, rate=rt, gamma=g).mssre_mm)[0]
                        cu = cell(units, robot=r, path=p, entry=e, rate=rt, gamma=g)
                        um = med_iqr(cu.mssre_mm)[0] if len(cu) else bm
                        m_rank.append(bm)
                        mm_rank.append(um)
                    best_m = ents[int(np.nanargmin(m_rank))] if np.isfinite(np.nanmin(m_rank)) else None
                    best_mm = ents[int(np.nanargmin(mm_rank))] if np.isfinite(np.nanmin(mm_rank)) else None
                    U["kendall"].append({"robot": r, "path": p, "rate": int(rt), "gamma": float(g),
                                         "tau_b": kendall_b(m_rank, mm_rank), "best_m_units": best_m,
                                         "best_mm_units": best_mm, "entries": list(ents)})
    return U


def fig_units(off, units, gamma, figdir):
    """MSSRE vs control rate with the error in m (R_off) and in mm (R_units)."""
    base = cell(off, noise="none", gamma=gamma)
    uu = cell(units, gamma=gamma)
    combos = [(r, p) for r in ("puma560", "panda") for p in ("circle", "lissajous")
              if not cell(uu, robot=r, path=p).empty]
    if not combos:
        return None
    ents = [e for e in RC.DISPLAY_ORDER if e in set(uu.entry)]
    rows = [("m", base), ("mm", uu)]
    fig, axes = plt.subplots(2, len(combos), figsize=(style.DOUBLE, 3.9), sharex=True, sharey=True,
                             squeeze=False)
    for i, (unit, df) in enumerate(rows):
        for j, (r, p) in enumerate(combos):
            ax = axes[i, j]
            for e in ents:
                pts = [(rt, *med_iqr(cell(df, robot=r, path=p, entry=e, rate=rt).mssre_mm))
                       for rt in sorted(uu.rate.unique())]
                pts = [x for x in pts if np.isfinite(x[1])]
                if not pts:
                    continue
                rt, m, q1, q3 = map(np.array, zip(*pts))
                s = sty(e)
                ax.errorbar(rt, m, yerr=[m - q1, q3 - m], color=s["color"], ls=s["linestyle"],
                            marker=s["marker"], ms=3.5, lw=1.0, elinewidth=0.5, capsize=1.5,
                            label=SHORT.get(e, e))
            ax.set_xscale("log")
            ax.set_yscale("log")
            _plain_log_ticks(ax)
            if i == 0:
                ax.set_title(f"{ROBOT_NAME[r]}, {PATH_NAME[p]}")
            if j == 0:
                ax.set_ylabel(("error unit: m (main study)" if unit == "m" else "error unit: mm")
                              + "\nMSSRE (mm)")
            if i == 1:
                ax.set_xlabel("control rate (Hz)")
            ax.set_xticks([100, 333, 1000])
            ax.set_xticklabels(["100", "333", "1000"])
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    top = 0.88
    fig.tight_layout(rect=(0, 0, 1, top), h_pad=0.6, w_pad=0.6)
    h, lab = axes[1, 0].get_legend_handles_labels()
    _top_legend(fig, h, lab, len(lab), top,
                title=f"$\\gamma$ = {gamma:g}, noise-free, off-path starts: the same controllers with the "
                      f"error E expressed in m or in mm (median, whiskers: IQR)")
    return style.savefig(fig, "fig_R_units")


def trace_consistency(npz_path, off):
    """Final residual and peak speed of the trace runs vs the matching sweep rows."""
    z = np.load(npz_path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    robot = meta["robot"].replace("robot_", "")
    out = []
    for run in meta["runs"]:
        noise = run["noise"]
        for gi, g in enumerate(meta["gammas"]):
            c = cell(off, robot=robot, path=meta["path"], entry=run["entry"], noise=noise,
                     gamma=round(float(g), 6), seed=meta["seed"], x0_index=meta["x0_index"])
            c = c[np.isclose(c.tau, run["tau"])]
            if c.empty:
                continue
            fr = float(c.final_residual.iloc[0])
            pr = float(c.peak_rate.iloc[0])
            out.append({"tag": run["tag"], "gamma": g,
                        "rel_diff_final_residual": abs(run["final_residual"][gi] - fr) / max(abs(fr), 1e-300),
                        "rel_diff_peak_rate": abs(run["peak_rate"][gi] - pr) / max(abs(pr), 1e-300)})
    return out


def write_tables(S, off, on, ents, rates, robots, paths, tabdir):
    cells = S["cells_off"]
    tabs = []

    def get(col, stat="median", **k):
        for d in cells:
            if all(d[a] == b for a, b in k.items()):
                v = d.get(col)
                return np.nan if v is None else (v[stat] if isinstance(v, dict) else v)
        return np.nan

    def order_of(**k):
        for o in S["orders"]:
            if all(o[a] == b for a, b in k.items()):
                return o
        return None

    rl = [f"{rt:g} Hz" for rt in rates]
    # tracking accuracy (gamma = 10, noise-free)
    rows = []
    for r in robots:
        for p in paths:
            for e in ents:
                o = order_of(robot=r, path=p, entry=e, gamma=10.0, noise="none")
                ordtxt = "--" if o is None else (f"{o['ls']:.2f} / {o['asym']:.2f}" + ("$^*$" if o["flag"] else ""))
                rows.append([ROBOT_NAME[r], PATH_NAME[p], SHORT.get(e, e)]
                            + [fnum(get("mssre_mm", robot=r, path=p, entry=e, rate=rt, gamma=10.0, noise="none"))
                               for rt in rates]
                            + [ordtxt, fnum(get("t_eps_0.001", robot=r, path=p, entry=e, rate=rates[-1], gamma=10.0,
                                                noise="none"))])
            rows.append("MID")
    if rows and rows[-1] == "MID":
        rows.pop()
    t = tabdir / "R_tracking.tex"
    write_tex(t, [["robot", "path", "model"] + [f"MSSRE {x} (mm)" for x in rl] + ["order LS / asym.", r"$t_{1\,\mathrm{mm}}$ (s)"]],
              rows, r"Robotics case study R: position-error MSSRE (median over trajectories, mm) at "
              r"$\gamma=10$, noise-free, off-path starts; empirical order of MSSRE in $\tau$ (fixed $\gamma$) "
              r"from an LS fit over all rates and over the asymptotic range (smallest-$\tau$ points whose local slopes "
              r"agree within 0.15; $^*$: pre-asymptotic or no decrease, only 3 rates); time to reach and stay below 1 mm at the fastest rate.",
              "tab:R_tracking", "lll" + "r" * (len(rates) + 2))
    tabs.append(t)

    # actuation (gamma = 10, noise-free)
    rows = []
    for r in robots:
        for p in paths:
            for e in ents:
                k = dict(robot=r, path=p, entry=e, gamma=10.0, noise="none")
                rows.append([ROBOT_NAME[r], PATH_NAME[p], SHORT.get(e, e)]
                            + [fnum(get("peak_rate", rate=rt, **k)) for rt in (rates[0], rates[-1])]
                            + [fnum(get("peak_rate_ss", rate=rt, **k)) for rt in (rates[0], rates[-1])]
                            + [fnum(get("alt_active", rate=rt, **k)) for rt in (rates[0], rates[-1])]
                            + [fnum(get("qd_viol_frac", rate=rates[0], **k)),
                               fnum(get("qdd_peak_ss", rate=rates[0], **k))])
            rows.append("MID")
    if rows and rows[-1] == "MID":
        rows.pop()
    a, b = f"{rates[0]:g}", f"{rates[-1]:g}"
    t = tabdir / "R_actuation.tex"
    write_tex(t, [["robot", "path", "model", rf"\multicolumn{{2}}{{c}}{{peak $|\dot\theta|$ (rad/s)}}",
                   rf"\multicolumn{{2}}{{c}}{{steady peak (rad/s)}}", rf"\multicolumn{{2}}{{c}}{{chattering}}",
                   r"viol.", r"$|\ddot\theta|_{\mathrm{ss}}$"],
                  ["", "", ""] + [f"{a} Hz", f"{b} Hz"] * 3 + [f"{a} Hz", f"{a} Hz"]],
              rows, r"Robotics case study R: commanded joint motion at $\gamma=10$ (noise-free, off-path "
              r"starts, medians).  Peak $\max_{k,i}|\theta_{k+1,i}-\theta_{k,i}|/\tau$ over the run and over the "
              r"steady window (last 25\%); chattering = fraction of sign alternations of consecutive joint "
              r"increments over the steady window, counted on the moving joints; viol. = fraction of control "
              r"steps in which some joint exceeds its speed limit (Panda: Franka FER limits 2.175/2.61 rad/s; "
              r"PUMA 560: nominal 2 rad/s, an assumption); $|\ddot\theta|_{\mathrm{ss}}$ = steady-window peak "
              r"joint acceleration (rad/s$^2$) implied by consecutive commands (Panda limits 7.5--20 rad/s$^2$).",
              "tab:R_actuation", "lll" + "r" * 8)
    tabs.append(t)

    # gamma = 100 stress
    if 100.0 in set(off.gamma):
        rows = []
        for r in robots:
            for p in paths:
                for e in ents:
                    k = dict(robot=r, path=p, entry=e, gamma=100.0, noise="none")
                    rows.append([ROBOT_NAME[r], PATH_NAME[p], SHORT.get(e, e)]
                                + [fnum(get("mssre_mm", rate=rt, **k)) for rt in (rates[0], rates[-1])]
                                + [fnum(get("peak_rate", rate=rt, **k)) for rt in (rates[0], rates[-1])]
                                + [fnum(get("frac_over_vel_limit", rate=rt, **k)) for rt in (rates[0], rates[-1])]
                                + [fnum(get("qlim_viol", rate=rates[0], **k))])
                rows.append("MID")
        if rows and rows[-1] == "MID":
            rows.pop()
        t = tabdir / "R_gamma100.tex"
        write_tex(t, [["robot", "path", "model", r"\multicolumn{2}{c}{MSSRE (mm)}",
                       r"\multicolumn{2}{c}{peak $|\dot\theta|$ (rad/s)}", r"\multicolumn{2}{c}{frac. over limit}",
                       r"$q$-lim. viol. (rad)"],
                      ["", "", ""] + [f"{a} Hz", f"{b} Hz"] * 3 + [f"{a} Hz"]],
                  rows, r"Robotics case study R, high-gain design $\gamma=100$ (noise-free, off-path starts, "
                  r"medians): tracking error, peak commanded joint speed, fraction of trajectories whose peak "
                  r"exceeds the joint-speed limit, and joint-position-limit violation (not enforced by any model).",
                  "tab:R_gamma100", "lll" + "r" * 7)
        tabs.append(t)

    # noise
    if "constant" in set(off.noise):
        rows = []
        for r in robots:
            for p in paths:
                for e in ents:
                    k = dict(robot=r, path=p, entry=e, gamma=10.0)
                    rows.append([ROBOT_NAME[r], PATH_NAME[p], SHORT.get(e, e)]
                                + [fnum(get("mssre_mm", rate=rt, noise=nz, **k))
                                   for rt in (rates[0], rates[-1]) for nz in ("none", "constant")])
                rows.append("MID")
        if rows and rows[-1] == "MID":
            rows.pop()
        t = tabdir / "R_noise.tex"
        write_tex(t, [["robot", "path", "model", rf"\multicolumn{{2}}{{c}}{{{a} Hz}}", rf"\multicolumn{{2}}{{c}}{{{b} Hz}}"],
                      ["", "", "", "none", "const.", "none", "const."]],
                  rows, rf"Robotics case study R: MSSRE (mm, median) without noise and under a constant "
                  rf"design-formula noise of {RC.NOISE_LEVEL:g} m/s per Cartesian axis ($\gamma=10$, off-path "
                  rf"starts).  Continuous-time prediction for linear ZNN: $\sqrt{{3}}\,\nu/\gamma$ = "
                  rf"{np.sqrt(3) * RC.NOISE_LEVEL / 10 * 1e3:.2f} mm; IEZNN rejects constant noise.",
                  "tab:R_noise", "lll" + "r" * 4)
        tabs.append(t)

    # drift (on-path)
    if S["drift_on"]:
        dd = pd.DataFrame(S["drift_on"])
        rows = []
        for r in robots:
            for p in paths:
                for e in ents:
                    c = dd[(dd.robot == r) & (dd.path == p) & (dd.entry == e) & (dd.gamma == 10.0) & (dd.noise == "none")]
                    if c.empty:
                        continue
                    vals = [c[c.rate == rt].drift_rad for rt in rates]
                    rows.append([ROBOT_NAME[r], PATH_NAME[p], SHORT.get(e, e)]
                                + [fnum(float(v.iloc[0])) if len(v) else "--" for v in vals]
                                + [fnum(float(c[c.rate == rates[-1]].mssre_mm.iloc[0]))
                                   if len(c[c.rate == rates[-1]]) else "--"])
                rows.append("MID")
        if rows and rows[-1] == "MID":
            rows.pop()
        t = tabdir / "R_drift.tex"
        write_tex(t, [["robot", "path", "model"] + [f"drift {x} (rad)" for x in rl] + [f"MSSRE {rl[-1]} (mm)"]],
                  rows, r"Robotics case study R: repetitive-motion joint drift $\|\theta(T)-\theta(0)\|_2$ on the "
                  r"closed paths from on-path starts ($E(0)=0$, deterministic), $\gamma=10$, noise-free.  "
                  r"Pseudoinverse control is not cyclic (Klein and Huang 1983), so a nonzero drift is expected.",
                  "tab:R_drift", "lll" + "r" * (len(rates) + 1))
        tabs.append(t)

    # error-unit sensitivity (R_units vs R_off)
    U = S.get("units") or {}
    if U.get("cells"):
        def uget(col, unit, **k):
            for d in U["cells"]:
                if all(d[a] == b for a, b in k.items()):
                    v = d.get(col)
                    return np.nan if v is None else v[unit][0]
            return np.nan
        uents = [e for e in ents if e != "znn_lin" and any(d["entry"] == e for d in U["cells"])]
        gams = sorted({d["gamma"] for d in U["cells"]})
        rows = []
        for r in robots:
            for p in paths:
                for e in uents:
                    for g in gams:
                        k = dict(robot=r, path=p, entry=e, gamma=g)
                        rows.append([ROBOT_NAME[r], PATH_NAME[p], SHORT.get(e, e), f"{g:g}"]
                                    + [fnum(uget("mssre_mm", u, rate=rt, **k))
                                       for rt in (rates[0], rates[-1]) for u in ("m_units", "mm_units")]
                                    + [fnum(uget("peak_rate", u, rate=rates[0], **k)) for u in ("m_units", "mm_units")]
                                    + [fnum(uget("alt_active", u, rate=rates[0], **k)) for u in ("m_units", "mm_units")])
                rows.append("MID")
        if rows and rows[-1] == "MID":
            rows.pop()
        inv = U.get("invariance") or {}
        invtxt = (f"Linear ZNN, run as a control, is unit-invariant: maximum relative difference "
                  f"{inv['max_rel_diff_mssre']:.1e} in MSSRE over {inv['n_cells']} cells." if inv else "")
        t = tabdir / "R_units.tex"
        write_tex(t, [["robot", "path", "model", r"$\gamma$",
                       rf"\multicolumn{{2}}{{c}}{{MSSRE {a} Hz (mm)}}", rf"\multicolumn{{2}}{{c}}{{MSSRE {b} Hz (mm)}}",
                       rf"\multicolumn{{2}}{{c}}{{peak $|\dot\theta|$ {a} Hz (rad/s)}}",
                       rf"\multicolumn{{2}}{{c}}{{chattering {a} Hz}}"],
                      ["", "", "", ""] + ["E in m", "E in mm"] * 4],
                  rows, r"Robotics case study R, error-unit sensitivity: the same controllers with the task "
                  r"error $E$ expressed in metres (main study) or in millimetres (noise-free, off-path starts, "
                  r"paired starts, medians; MSSRE always reported in mm).  Power-type activations are not "
                  r"invariant to the unit of $E$; the SPTC-NT-ZNN noise bound is converted with the unit "
                  r"(0.05 m/s = 50 mm/s). " + invtxt,
                  "tab:R_units", "llll" + "r" * 8)
        tabs.append(t)
    return tabs


def write_summary(S, results):
    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        return o
    (results / "summary.json").write_text(json.dumps(clean(S), indent=1))
    cells = S["cells_off"]

    def get(col, **k):
        for d in cells:
            if all(d[a] == b for a, b in k.items()):
                v = d.get(col)
                return np.nan if v is None else (v["median"] if isinstance(v, dict) else v)
        return np.nan

    ents = S["entries"]
    L = [f"# R robotics case study -- summary ({S['scale']} scale)", "",
         "Position-only, unconstrained, velocity-level pseudoinverse tracking (no joint position/velocity/"
         "acceleration limits are enforced; they are only measured).  Off-path starts "
         f"(theta0 + U[-{RC.X0_SPREAD}, {RC.X0_SPREAD}] rad), T = {RC.T_HORIZON:g} s, fixed gamma, "
         f"{S['n_traj_per_cell_off']} trajectories per cell (medians; IQR and bootstrap CIs in summary.json).", "",
         f"Run status (off-path): {S['status_off']}; on-path: {S['status_on']}.  "
         f"Median wall per step {S['wall_per_step_us_all']:.0f} us (B*G = 8 trajectories); "
         f"max peak RSS {S['peak_rss_mb_max']:.0f} MB per task.  "
         f"Check qd_peak (wrapper) == peak_rate (online stat): max rel. diff "
         f"{S['check_qd_peak_equals_peak_rate_maxrel']}.", ""]
    L.append("## Headline numbers (gamma = 10, noise-free unless stated, error E in metres)")
    L.append("")
    L.append("| robot | path | model | MSSRE 100 Hz (mm) | MSSRE 1 kHz (mm) | peak qd 100 Hz (rad/s) | "
             "peak qd 1 kHz | chattering (alt_active) 100 Hz | chattering 1 kHz |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    rates = sorted({d["rate"] for d in cells})
    lo, hi = rates[0], rates[-1]
    for r in ("puma560", "panda"):
        for p in ("circle", "lissajous"):
            for e in ents:
                k = dict(robot=r, path=p, entry=e, gamma=10.0, noise="none")
                if np.isnan(get("mssre_mm", rate=lo, **k)):
                    continue
                L.append(f"| {ROBOT_NAME[r]} | {p} | {SHORT.get(e, e)} | {fplain(get('mssre_mm', rate=lo, **k))} | "
                         f"{fplain(get('mssre_mm', rate=hi, **k))} | {fplain(get('peak_rate', rate=lo, **k))} | "
                         f"{fplain(get('peak_rate', rate=hi, **k))} | {fplain(get('alt_active', rate=lo, **k), 2)} | "
                         f"{fplain(get('alt_active', rate=hi, **k), 2)} |")
    L.append("")
    # implicit vs explicit
    h = S["headline"]["implicit_vs_explicit_sbp"]
    sel = [x for x in h if x["gamma"] == 10.0]
    if sel:
        L.append("## Implicit vs explicit Euler for sign-bi-power ZNN, error in metres (paired bootstrap 95% CI; "
                 "unit-dependent, see the error-unit section)")
        L.append("")
        for x in sel:
            rr = x["mssre_ratio_impl_over_expl"]
            L.append(f"- {ROBOT_NAME[x['robot']]} {x['path']} {x['rate']} Hz: MSSRE ratio implicit/explicit "
                     f"{fplain(rr[0])} [{fplain(rr[1])}, {fplain(rr[2])}]; chattering "
                     f"(alt_active) {fplain(x['alt_active_expl'][0], 2)} -> {fplain(x['alt_active_impl'][0], 2)}")
        L.append("")
    h = S["headline"]["ieznn_vs_znn_const_noise"]
    sel = [x for x in h if x["gamma"] == 10.0]
    if sel:
        L.append(f"## IEZNN vs linear ZNN under constant noise {RC.NOISE_LEVEL:g} m/s (gamma = 10)")
        L.append("")
        for x in sel:
            rr = x["mssre_ratio_ieznn_over_znn"]
            L.append(f"- {ROBOT_NAME[x['robot']]} {x['path']} {x['rate']} Hz: ZNN {fplain(x['znn_mssre_mm'][0])} mm, "
                     f"IEZNN {fplain(x['ieznn_mssre_mm'][0])} mm, ratio {fplain(rr[0])} "
                     f"[{fplain(rr[1])}, {fplain(rr[2])}] (CT prediction for ZNN: "
                     f"{np.sqrt(3) * RC.NOISE_LEVEL / 10 * 1e3:.2f} mm)")
        L.append("")
    # gamma 100 velocity
    if any(d["gamma"] == 100.0 for d in cells):
        L.append("## High-gain design gamma = 100: peak commanded joint speed (median, rad/s) and "
                 "fraction of trajectories over the limit")
        L.append("")
        for r in ("puma560", "panda"):
            for e in ents:
                vals = [(rt, get("peak_rate", robot=r, path="circle", entry=e, rate=rt, gamma=100.0, noise="none"),
                         get("frac_over_vel_limit", robot=r, path="circle", entry=e, rate=rt, gamma=100.0,
                             noise="none")) for rt in rates]
                if all(np.isnan(v[1]) for v in vals):
                    continue
                L.append(f"- {ROBOT_NAME[r]} circle {SHORT.get(e, e)}: "
                         + ", ".join(f"{rt} Hz {fplain(v)} ({fplain(fr, 2)})" for rt, v, fr in vals))
        L.append("")
    # orders
    L.append("## Empirical orders of MSSRE in tau (fixed gamma = 10, noise-free; LS over all rates / asymptotic)")
    L.append("")
    for o in S["orders"]:
        if o["gamma"] == 10.0 and o["noise"] == "none" and o["path"] == "circle":
            L.append(f"- {ROBOT_NAME[o['robot']]} circle {SHORT.get(o['entry'], o['entry'])}: local slopes "
                     f"{[round(s, 2) for s in o['local']]}, LS {fplain(o['ls'], 3)}, asymptotic "
                     f"{fplain(o['asym'], 3)} ({o['asym_points']} pts) {o['flag']}")
    L.append("")
    L.append("## Kendall tau_b of model rankings (gamma = 10, noise-free)")
    L.append("")
    L.append("p-values are descriptive only: n = 5 models (smallest attainable two-sided p = 0.017), "
             f"{len(S['kendall'])} comparisons on overlapping data, no multiplicity correction.")
    L.append("")
    for k in S["kendall"]:
        tb = k["tau_b"]
        where = " ".join(str(k.get(x)) for x in ("robot", "path", "rate", "rates") if k.get(x) is not None)
        L.append(f"- {k['what']} ({where}): tau_b = {fplain(tb[0], 2)} (p = {fplain(tb[1], 2)}, n = {tb[2]})")
    L.append("")
    bad = [d for d in cells if d.get("n_diverged", 0) or d.get("n_failed", 0)]
    L.append(f"Censoring: {len(bad)} off-path cells contain diverged or failed trajectories"
             + (": " + "; ".join(f"{d['robot']} {d['path']} {SHORT.get(d['entry'], d['entry'])} {d['rate']} Hz "
                                 f"gamma {d['gamma']:g} {d['noise']} (div {d['n_diverged']}, fail {d['n_failed']})"
                                 for d in bad[:20]) if bad else "")
             + ".  Diverged trajectories enter medians, CIs and rankings as +inf (worst), never dropped.")
    L.append("")
    U = S.get("units") or {}
    if U.get("cells"):
        L.append("## Error-unit sensitivity: the same controllers with E in mm instead of m "
                 "(noise-free, paired off-path starts)")
        L.append("")
        inv = U.get("invariance")
        if inv:
            L.append(f"Control: linear ZNN is unit-invariant, max relative difference "
                     f"{inv['max_rel_diff_mssre']:.1e} (MSSRE) / {inv['max_rel_diff_peak_rate']:.1e} (peak speed) "
                     f"over {inv['n_cells']} cells.")
            L.append("")
        L.append("| robot | path | model | gamma | MSSRE 100 Hz m -> mm units (mm) | ratio mm/m [95% CI] | "
                 "MSSRE 1 kHz m -> mm units | chattering 100 Hz m -> mm | peak qd 100 Hz m -> mm (rad/s) |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        urates = sorted({d["rate"] for d in U["cells"]})
        ulo, uhi = urates[0], urates[-1]

        def ug(col, unit, **k):
            for d in U["cells"]:
                if all(d[a] == b for a, b in k.items()):
                    return d[col][unit][0] if col in d else np.nan
            return np.nan

        def ur(**k):
            for d in U["cells"]:
                if all(d[a] == b for a, b in k.items()):
                    return d["mssre_ratio_mm_over_m"]
            return (np.nan, np.nan, np.nan)
        for g in sorted({d["gamma"] for d in U["cells"]}):
            for r in ("puma560", "panda"):
                for p in ("circle", "lissajous"):
                    for e in ents:
                        if e == "znn_lin":
                            continue
                        k = dict(robot=r, path=p, entry=e, gamma=g)
                        if np.isnan(ug("mssre_mm", "m_units", rate=ulo, **k)):
                            continue
                        rr = ur(rate=ulo, **k)
                        L.append(f"| {ROBOT_NAME[r]} | {p} | {SHORT.get(e, e)} | {g:g} | "
                                 f"{fplain(ug('mssre_mm', 'm_units', rate=ulo, **k))} -> "
                                 f"{fplain(ug('mssre_mm', 'mm_units', rate=ulo, **k))} | "
                                 f"{fplain(rr[0])} [{fplain(rr[1])}, {fplain(rr[2])}] | "
                                 f"{fplain(ug('mssre_mm', 'm_units', rate=uhi, **k))} -> "
                                 f"{fplain(ug('mssre_mm', 'mm_units', rate=uhi, **k))} | "
                                 f"{fplain(ug('alt_active', 'm_units', rate=ulo, **k), 2)} -> "
                                 f"{fplain(ug('alt_active', 'mm_units', rate=ulo, **k), 2)} | "
                                 f"{fplain(ug('peak_rate', 'm_units', rate=ulo, **k))} -> "
                                 f"{fplain(ug('peak_rate', 'mm_units', rate=ulo, **k))} |")
        L.append("")
        iv = [x for x in U["impl_vs_expl_mm"] if x["gamma"] == 10.0]
        if iv:
            L.append("Implicit / explicit SBP MSSRE ratio with E in mm (gamma = 10, paired bootstrap 95% CI): "
                     + "; ".join(f"{ROBOT_NAME[x['robot']]} {x['path']} {x['rate']} Hz "
                                 f"{fplain(x['mssre_ratio_impl_over_expl'][0])} "
                                 f"[{fplain(x['mssre_ratio_impl_over_expl'][1])}, "
                                 f"{fplain(x['mssre_ratio_impl_over_expl'][2])}]" for x in iv))
            L.append("")
        L.append("5-model MSSRE ranking with E in m vs E in mm (Kendall tau_b; best model in each unit):")
        L.append("")
        for k in U["kendall"]:
            tb = k["tau_b"]
            L.append(f"- gamma {k['gamma']:g}, {ROBOT_NAME[k['robot']]} {k['path']} {k['rate']} Hz: tau_b = "
                     f"{fplain(tb[0], 2)}; best in m: {SHORT.get(k['best_m_units'], k['best_m_units'])}, "
                     f"best in mm: {SHORT.get(k['best_mm_units'], k['best_mm_units'])}")
        L.append("")
    if S["drift_on"]:
        L.append("## Joint drift on closed paths (on-path start, gamma = 10, noise-free, rad)")
        L.append("")
        dd = pd.DataFrame(S["drift_on"])
        for (r, p, e), c in dd[(dd.gamma == 10.0) & (dd.noise == "none")].groupby(["robot", "path", "entry"]):
            L.append(f"- {ROBOT_NAME[r]} {p} {SHORT.get(e, e)}: "
                     + ", ".join(f"{int(rt)} Hz {fplain(v, 2)}" for rt, v in zip(c.rate, c.drift_rad)))
        L.append("")
    if S.get("trace_consistency"):
        mx = max(max(x["rel_diff_final_residual"], x["rel_diff_peak_rate"]) for x in S["trace_consistency"])
        L.append(f"Trace-task consistency with the sweep rows: max relative difference {mx:.2e} "
                 f"over {len(S['trace_consistency'])} (run, gain) pairs.")
        L.append("")
    L.append("## Interpretation")
    L.append("")
    L += interpret(S)
    (results / "SUMMARY.md").write_text("\n".join(L) + "\n")


def interpret(S):
    cells = S["cells_off"]

    def get(col, **k):
        vals = [d[col]["median"] for d in cells if all(d[a] == b for a, b in k.items()) and col in d]
        return np.array(vals, float)

    def rg(a, pct=False):
        a = np.asarray(a, float)
        a = a[np.isfinite(a)]
        if a.size == 0:
            return "n/a"
        lo_, hi_ = (np.min(a) * 100, np.max(a) * 100) if pct else (np.min(a), np.max(a))
        f = (lambda v: f"{v:.0f}") if pct else (lambda v: f"{v:.2g}")
        return f(lo_) if f(lo_) == f(hi_) else f"{f(lo_)}-{f(hi_)}"

    out = []
    rates = sorted({d["rate"] for d in cells})
    lo, hi = rates[0], rates[-1]
    lin_lo = get("mssre_mm", entry="znn_lin", rate=lo, gamma=10.0, noise="none")
    lin_hi = get("mssre_mm", entry="znn_lin", rate=hi, gamma=10.0, noise="none")
    sbp_lo = get("mssre_mm", entry="znn_sbp", rate=lo, gamma=10.0, noise="none")
    ie_lo = get("mssre_mm", entry="znn_sbp_ie", rate=lo, gamma=10.0, noise="none")
    if lin_lo.size:
        out.append(f"- Linear ZNN tracks to {rg(lin_lo)} mm at {lo} Hz and "
                   f"{rg(lin_hi)} mm at {hi} Hz (gamma = 10): at a fixed design "
                   f"gain the discrete error falls only like tau (one order lost w.r.t. the fixed-h DTZNN convention).")
    if sbp_lo.size and ie_lo.size:
        out.append(f"- With the error in metres (the problem's SI unit), explicit Euler turns the sign-bi-power "
                   f"activation into a chattering controller: at {lo} Hz its MSSRE is {rg(sbp_lo)} mm, worse "
                   f"than linear ZNN, with joint increments alternating sign at most steps; the same activation "
                   f"stepped implicitly (resolvent) gives {rg(ie_lo)} mm without chattering.  The explicit "
                   f"band is the 2-cycle |e_i| = (gamma tau / 4)^2 of the |e|^0.5 term, a number in the unit of E.")
    U = S.get("units") or {}
    if U.get("cells"):
        def uv(col, unit, **k):
            return np.array([d[col][unit][0] for d in U["cells"] if all(d[a] == b for a, b in k.items())
                             and col in d], float)
        e_m = uv("mssre_mm", "m_units", entry="znn_sbp", rate=lo, gamma=10.0)
        e_mm = uv("mssre_mm", "mm_units", entry="znn_sbp", rate=lo, gamma=10.0)
        i_mm = uv("mssre_mm", "mm_units", entry="znn_sbp_ie", rate=lo, gamma=10.0)
        a_m = uv("alt_active", "m_units", entry="znn_sbp", rate=lo, gamma=10.0)
        a_mm = uv("alt_active", "mm_units", entry="znn_sbp", rate=lo, gamma=10.0)
        e100 = uv("mssre_mm", "mm_units", entry="znn_sbp", rate=lo, gamma=100.0)
        e100m = uv("mssre_mm", "m_units", entry="znn_sbp", rate=lo, gamma=100.0)
        pk100 = uv("peak_rate", "mm_units", entry="znn_sbp", rate=lo, gamma=100.0)
        pk10m = uv("peak_rate", "m_units", entry="znn_sbp", rate=lo, gamma=10.0)
        pk10mm = uv("peak_rate", "mm_units", entry="znn_sbp", rate=lo, gamma=10.0)
        kk = [k for k in U["kendall"] if k["gamma"] == 10.0]
        tb = np.array([k["tau_b"][0] for k in kk], float)
        flips = sum(1 for k in kk if k["best_m_units"] != k["best_mm_units"])
        if e_m.size and e_mm.size:
            out.append(f"- **This conclusion is unit-dependent.**  Expressing the same error in millimetres "
                       f"(identical physical task; linear ZNN reproduces to round-off) moves explicit SBP at {lo} Hz, "
                       f"gamma = 10, from {rg(e_m)} mm to {rg(e_mm)} mm and its chattering from {rg(a_m)} to "
                       f"{rg(a_mm)} (its transient peak speed rises from {rg(pk10m)} to {rg(pk10mm)} rad/s); "
                       f"implicit SBP gives {rg(i_mm)} mm, so the implicit/explicit ordering is no "
                       f"longer the one seen in metres.  At gamma = 100 the |e|^2 branch, now 1000x stronger, "
                       f"destabilises the explicit transient instead ({rg(e100m)} mm in m vs {rg(e100)} mm in mm, "
                       f"peak {rg(pk100)} rad/s).  The 5-model MSSRE ranking changes with the unit "
                       f"(Kendall tau_b m vs mm {rg(tb)}; best model differs in {flips}/{len(kk)} cells).  "
                       f"Power-type activations have no scale-free gain: comparisons with linear ZNN at equal "
                       f"nominal gamma are comparisons at a unit-dependent effective gain gamma/(2 sqrt|e|).")
            ri = np.array([d["mssre_ratio_mm_over_m"][0] for d in U["cells"]
                           if d["entry"] == "znn_sbp_ie" and d["gamma"] == 100.0], float)
            re_ = np.array([d["mssre_ratio_mm_over_m"][0] for d in U["cells"]
                            if d["entry"] == "znn_sbp_ie" and d["gamma"] == 10.0], float)
            if ri.size:
                rx = np.array([d["mssre_ratio_mm_over_m"][0] for d in U["cells"] if d["entry"] == "znn_sbp"], float)
                out.append(f"- What survives the unit change is robustness, not accuracy: implicit SBP stays bounded "
                           f"in both units and its MSSRE changes by a moderate factor (ratio mm/m "
                           f"{re_.min():.2g}-{re_.max():.2g} at gamma = 10, {ri.min():.3g}-{ri.max():.3g} at "
                           f"gamma = 100; explicit SBP: {rx.min():.2g}-{rx.max():.2g}), whereas explicit SBP is "
                           f"accurate only inside a unit- and gamma*tau-dependent "
                           f"window (chattering for small errors in m, transient blow-up for large errors in mm).")
    sp = get("peak_rate", entry="sptc_nt_znn", rate=lo, gamma=10.0, noise="none")
    spv = get("qd_viol_frac", entry="sptc_nt_znn", rate=lo, gamma=10.0, noise="none")
    if sp.size:
        out.append(f"- SPTC-NT-ZNN (t_c = 1 s, noise bound 0.05 m/s) commands peak joint speeds of "
                   f"{rg(sp)} rad/s at {lo} Hz (gamma = 10), above the joint-speed "
                   f"limits in {rg(spv, pct=True)} % of the control steps: its "
                   f"post-t_c sign term chatters at the sampling rate.")
    sp100 = get("peak_rate", entry="sptc_nt_znn", rate=lo, gamma=100.0, noise="none")
    if sp100.size:
        sp100h = get("peak_rate", entry="sptc_nt_znn", rate=hi, gamma=100.0, noise="none")
        out.append(f"- The time-varying gain gamma/(t_c - t) gives the Euler error map the factor "
                   f"1 - gamma/m at m steps before t_c, so h_eff > 2 in the last gamma/2 sampling steps "
                   f"whatever tau is: a faster rate does not remove the blow-up.  With gamma = 100 the peak joint "
                   f"speed reaches {rg(sp100)} rad/s at {lo} Hz and {rg(sp100h)} rad/s at {hi} Hz -- a "
                   f"'predefined-time' design that is physically unrealisable on either arm.")
    l100 = get("peak_rate", entry="znn_lin", gamma=100.0, noise="none")
    if l100.size:
        out.append(f"- Even linear ZNN at gamma = 100 commands {rg(l100)} rad/s "
                   f"during the transient from a 0.05-rad start offset, at or above the ~2 rad/s joint-speed limits; "
                   f"the high gains common in ZNN papers presume unlimited actuators.")
    nb = [x for x in S["headline"]["ieznn_vs_znn_const_noise"] if x["gamma"] == 10.0]
    if nb:
        r = np.array([x["mssre_ratio_ieznn_over_znn"][0] for x in nb], float)
        out.append(f"- Under a constant {RC.NOISE_LEVEL:g} m/s disturbance linear ZNN keeps the continuous-time bias "
                   f"sqrt(3) nu/gamma = {np.sqrt(3) * RC.NOISE_LEVEL / 10 * 1e3:.1f} mm, while IEZNN removes it "
                   f"(MSSRE ratio IEZNN/ZNN {rg(r)}); the integral action survives "
                   f"discretization because the bias is constant.")
    out.append("- All models are compared at the same nominal gamma (fixed-gamma deployment), not at a matched "
               "effective gain or matched peak speed (that is E4's job); IEZNN uses lambda = gamma^2/4, and "
               "SPTC-NT-ZNN is given the true noise bound (an information advantage).")
    out.append("- Limitation: the benchmark problem is position-only, unconstrained pseudoinverse control (no "
               "orientation task, no joint-limit / velocity / acceleration constraints, no dynamics, no actuator "
               "saturation); peak speeds and accelerations are those the controller *commands*.")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--figdir", default=str(style.FIG_DIR))
    ap.add_argument("--tabdir", default=str(style.TAB_DIR))
    args = ap.parse_args(argv)
    S = analyse(args.results, args.figdir, args.tabdir)
    print(f"wrote {len(S['figures'])} figures, {len(S['tables'])} tables, "
          f"{Path(args.results) / 'summary.json'}, {Path(args.results) / 'SUMMARY.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
