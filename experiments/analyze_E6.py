#!/usr/bin/env python
"""Analysis of E6 -- information realism (where does dE/dt come from?).

    python experiments/analyze_E6.py --results results/E6 \
        --figdir figures --tabdir results/tables

Reads the aggregated runner output of ``experiments/configs/E6.py`` and writes

* figures (``<figdir>/fig_E6_*.pdf|png``, experiments/style.py):
    - ``fig_E6_regime_<problem>_<protocol>``: sigma x tau -> best formula, one
      panel per information regime (pale cell = best not significantly better
      than the runner-up, paired seed-cluster bootstrap 95 % CI);
    - ``fig_E6_advantage_<problem>_<protocol>``: log10(MSSRE of the best low-
      order formula / MSSRE of the best high-order formula), diverging map;
    - ``fig_E6_mssre_tau_<A|B>_<problem>_<protocol>``: MSSRE vs tau per sigma
      (median + IQR band), A = simulate regimes, B = estimators;
    - ``fig_E6_amplification``: noise amplification MSSRE / sigma at the
      smallest tau (noise-dominated cells);
    - ``fig_E6_order_collapse``: asymptotic order at sigma = 0 per formula and
      information regime (information limits the order);
    - ``fig_E6_traces``: residual vs time (one illustrative cell);
* LaTeX booktabs tables (``<tabdir>/E6_*.tex``): orders, amplification,
  regime summary, headline ratios, Kendall tau_b;
* ``<results>/summary.json``, ``<results>/SUMMARY.md`` and
  ``<results>/E6_cells.csv`` (per-cell medians, IQR, bootstrap CI).

Statistics: one cell = (problem, h protocol, information regime, sigma,
formula, tau) with seeds x B trajectories; medians + IQR; bootstrap 95 % CIs
(2000 resamples, seeded) of medians and of paired median ratios; Kendall
tau_b for ranking agreement; empirical orders fitted only on the asymptotic
range (consecutive-ratio diagnostics, points below the floor 1e-12
excluded), reported together with the all-points fit.
"""
from __future__ import annotations

import argparse
import json
import math
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
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.legend import Legend  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from matplotlib.ticker import LogLocator, NullFormatter  # noqa: E402

from experiments import keys, style  # noqa: E402

H_COMMON = 0.1
FLOOR = 1e-12            # floor for order fits: the measured round-off of MSSRE here is ~3e-15
                         # (within-seed x0 spread at sigma = 0), so 1e-12 keeps a >= 300x margin; the
                         # generic 1e-11 of docs/spec/discretizations.md left uhlig_4_6 (analytic, BD5)
                         # with a single usable tau (7.7e-12 at tau = 3e-3) and no order estimate
TOL_SLOPE = 0.15         # asymptotic window: every local slope within 0.15 of the window's LS slope
N_BOOT = 2000
BOOT_SEED = 20261018
MIN_CLUSTERS = 3         # a seed-cluster bootstrap CI needs >= 3 seeds; fewer -> "n/a"
BIG = 1e300              # finite stand-in for +inf (diverged = worst case; two diverged values tie)
HUGE = 1e250             # medians/ratios beyond this came from BIG -> reported as inf (or 0)
SIGMA_LINE = {"color": "#52514e", "linewidth": 0.5, "linestyle": "-", "alpha": 0.55}
DIV_FILL = "#ffffff"     # regime map: all formulas diverged (white cell + "div")
NOISE_DOM = 5.0         # a cell is noise-dominated when MSSRE(sigma) >= 5 x MSSRE(sigma = 0)
NOISE_DOM_SLOPE = 20.0  # floor-slope fits: MSSRE(sigma) >= 20 x MSSRE(0) (truncation share <~ 5 %)
OTHER_STYLE = {"color": None, "marker": "h", "linestyle": (0, (7, 2))}   # grey "other" (colour set below)
TASK_OVERHEAD_S = 2.0   # per pool task: launch spacing, interpreter start, imports, parquet/npz write
                        # (measured in the 2026-10-18 mini pool: 1.6 s outside + 0.4 s inside the task)
INFO_ORDER = ["analytic", "bd1", "bd2", "savgol", "alpha_beta", "kalman", "bd5"]
INFO_LABEL = {"analytic": "analytic", "bd1": "BD1", "bd2": "BD2", "savgol": "S-G (5,2)",
              "alpha_beta": r"$\alpha$-$\beta$", "kalman": "Kalman", "bd5": "BD5 = S-G (6,5)"}
INFO_PLAIN = {"analytic": "analytic", "bd1": "BD1", "bd2": "BD2", "savgol": "S-G(5,2)",
              "alpha_beta": "alpha-beta", "kalman": "Kalman", "bd5": "BD5"}
INFO_TEX = {"analytic": "analytic", "bd1": "BD1", "bd2": "BD2", "savgol": "S-G(5,2)",
            "alpha_beta": r"$\alpha$-$\beta$", "kalman": "Kalman", "bd5": "BD5"}
SIM_SET = ["analytic", "bd1", "bd2"]
EST_SET = ["savgol", "alpha_beta", "kalman", "bd5"]
LOW_ORDER = ("euler", "taylor_zhang")
PROTO_LABEL = {"common": f"common h = {H_COMMON:g}", "hopt": r"own $h_{\rm opt}$"}
PROTO_PLAIN = {"common": f"common h={H_COMMON:g}", "hopt": "own h_opt"}
PROTO_TEX = {"common": f"{H_COMMON:g}", "hopt": r"$h_{\rm opt}$"}


# =============================================================================
# labels and styles
# =============================================================================
def family_index(k):
    if k == "euler":
        return 0
    if k == "taylor_zhang":
        return 1
    if k.startswith("zead_"):
        return 2
    if k.startswith("uhlig_"):
        return 3
    return 4


def short_label(k):
    if k == "euler":
        return "Eu"
    if k == "taylor_zhang":
        return "TZ"
    if k.startswith("zead_"):
        return "Z" + k.split("_")[1]
    if k.startswith("uhlig_"):
        return "U" + "".join(k.split("_")[1:3])
    return k[:4]


def long_label(k, order=None):
    base = {"euler": "Euler", "taylor_zhang": "Taylor-Zhang"}.get(k)
    if base is None:
        if k.startswith("zead_"):
            parts = k.split("_")
            base = f"ZeaD {parts[1]}" + (f" ({parts[2]})" if len(parts) > 2 else "")
        elif k.startswith("uhlig_"):
            base = "Uhlig " + "_".join(k.split("_")[1:])
        else:
            base = k
    return f"{base}, q={order}" if order is not None else base


def _other_style():
    return {**OTHER_STYLE, "color": style.GREY}


def disc_style(k):
    """canonical keys.GROUPS['disc'] slot; a formula outside the group (zead_11i_chen2023 at
    the time of writing) keeps the grey 'other' colour but gets a marker / dash pattern that
    no canonical slot uses (hexagon, long dashes) instead of keys' '.' / ':', which was hard
    to tell apart from the dotted canonical slots of zead_6i_guo2017 and uhlig_4_6."""
    if k in keys.GROUPS["disc"]:
        return keys.style_in_group("disc", k)
    return _other_style()


def info_style(i):
    """canonical keys.GROUPS['info'] slot; E6 regimes outside the group (bd5) take the
    group's next FREE slots in INFO_ORDER (the info group has 7 members, so slot 7 is
    unused by any entity); grey 'other' only if no free slot is left."""
    grp = list(keys.GROUPS["info"])
    if i in grp:
        return keys.style_in_group("info", i)
    extra = [x for x in INFO_ORDER if x not in grp]
    if i in extra and len(grp) + extra.index(i) < len(style.CAT):
        j = len(grp) + extra.index(i)
        return {"color": style.CAT[j], "marker": style.MARKERS[j], "linestyle": style.LINESTYLES[j]}
    return _other_style()


def _msize(st, base=3.0):
    """the grey 'other' style of keys.py uses marker '.', which needs a larger size."""
    return base * 1.9 if st.get("marker") == "." else base


def fmt_num(v, sig=2):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "--" if v is None or np.isnan(v) else ("inf" if v > 0 else "-inf")
    return f"{v:.{sig}g}"


def ci_text(lo, hi, k=None):
    """'[lo, hi]', or 'n/a (k seeds)' when the cluster bootstrap had too few seeds."""
    if k is not None and np.isfinite(k) and int(k) < MIN_CLUSTERS:
        return f"n/a ({int(k)} seed{'s' if int(k) != 1 else ''})"
    return f"[{fmt_num(lo, 3)}, {fmt_num(hi, 3)}]"


def tau_lab(t):
    m, e = f"{t:.0e}".split("e")
    return f"{m}e{int(e)}"


def sig_lab(s):
    return "0" if s == 0 else tau_lab(s)


def sig_math(s):
    """math-mode body (no $) of a power-of-ten value: 10^{-3}, 3\\cdot10^{-2}."""
    if s == 0:
        return "0"
    m, e = f"{s:.0e}".split("e")
    return f"10^{{{int(e)}}}" if m == "1" else f"{m}\\cdot10^{{{int(e)}}}"


def sig_tex(s):
    return "0" if s == 0 else f"${sig_math(s)}$"


# =============================================================================
# loading
# =============================================================================
def load(results):
    from znnbench.runner import aggregate
    df = aggregate(results)
    if df.empty:
        raise SystemExit(f"no results under {results}")

    wmap = {}
    for w in df["wrappers"].fillna("").unique():
        info, sig = None, None
        if w:
            chain = json.loads(w)
            for c in chain:
                if c["key"] == "EstimatedDerivativeProblem":
                    kw = c.get("kwargs", {})
                    m = kw.get("method", "savgol")
                    if m == "savgol":
                        order = int(kw.get("order", 2))
                        window = int(kw.get("window", order + 3))
                        info = "savgol" if (order == 2 and window == 5) else (
                            "bd5" if (order == 5 and window == 6) else f"savgol_w{window}_o{order}")
                    else:
                        info = m
                    sig = float(kw.get("meas_std", 0.0))
        wmap[w] = (info, sig)
    w = df["wrappers"].fillna("")
    est_info = w.map(lambda s: wmap[s][0])
    est_sig = w.map(lambda s: wmap[s][1])
    df["info_reg"] = np.where(est_info.notna(), est_info, df["info"])
    df["sigma"] = np.where(est_sig.notna(), est_sig, df["data_noise"]).astype(float)
    df["protocol"] = np.where(np.isclose(df["h"].astype(float), H_COMMON, rtol=1e-9), "common", "hopt")
    df["problem"] = df["problem_key"]
    df["disc"] = df["disc_key"]
    v = df["mssre"].astype(float).copy()
    bad = ~df["status"].isin(["ok", "diverged"])
    v[bad] = np.nan
    v[df["status"] == "diverged"] = np.inf
    df["v"] = v
    mv = df["mean_ssre"].astype(float).copy()
    mv[bad] = np.nan
    mv[df["status"] == "diverged"] = np.inf
    df["v_mean"] = mv
    return df


# =============================================================================
# statistics
# =============================================================================
def _med_rows(X):
    """row medians, NaN-aware only when needed (np.nanmedian is slow)."""
    return np.nanmedian(X, axis=1) if np.isnan(X).any() else np.median(X, axis=1)


def _cluster_matrix(x, groups):
    """(n_clusters, max_size) matrix of x padded with NaN, one row per cluster."""
    x = np.asarray(x, float)
    u, inv = np.unique(np.asarray(groups), return_inverse=True)
    size = np.bincount(inv, minlength=len(u))
    M = np.full((len(u), int(size.max()) if len(size) else 0), np.nan)
    fill = np.zeros(len(u), dtype=int)
    for v, k in zip(x, inv):
        M[k, fill[k]] = v
        fill[k] += 1
    return M


def _censor(x):
    """+inf (diverged) -> BIG, so medians, quantiles, bootstrap percentiles and
    ratios treat a diverged trajectory as the worst case (two diverged values
    tie, ratio 1) instead of producing NaN (inf - inf, inf / inf) that would
    silently drop out.  NaN (failed run) stays NaN and is counted separately."""
    x = np.array(x, dtype=float, copy=True)
    x[np.isposinf(x)] = BIG
    return x


def _uncensor(v):
    """BIG-derived medians/quantiles back to +inf."""
    v = float(v)
    return np.inf if v >= HUGE else v


def _uncensor_ratio(v):
    v = float(v)
    return np.inf if v >= HUGE else (0.0 if 0 <= v <= 1.0 / HUGE else v)


def _n_clusters(groups):
    return len(set(np.asarray(groups).tolist()))


class Boot:
    """Seeded CLUSTER bootstrap: problem seeds are resampled with replacement
    and every trajectory of a drawn seed is kept.  At steady state the MSSRE
    of linear ZNN does not depend on the initial state (measured: within-seed
    spread ~1e-14 relative, see E6_NOTES.md), so the B initial states of a
    seed are not independent replicates; the seed (problem instance + noise
    stream) is.  With fewer than MIN_CLUSTERS seeds no CI is reported (NaN,
    printed "n/a"): 1 seed has no between-seed variation, 2 seeds give only 3
    distinct resamples.  Diverged trajectories enter as the worst case (see
    ``_censor``)."""

    def __init__(self, seed=BOOT_SEED, n=N_BOOT):
        self.rng = np.random.default_rng(seed)
        self.n = n

    def _draw(self, k):
        return self.rng.integers(0, k, size=(self.n, k))

    def median_ci(self, x, groups):
        M = _cluster_matrix(_censor(x), groups)
        if M.shape[0] < MIN_CLUSTERS or np.isnan(M).all():
            return (np.nan, np.nan)
        D = self._draw(M.shape[0])
        with np.errstate(invalid="ignore", over="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = _med_rows(M[D].reshape(self.n, -1))
            med = med[~np.isnan(med)]
            if len(med) == 0:
                return (np.nan, np.nan)
            return (_uncensor(np.percentile(med, 2.5)), _uncensor(np.percentile(med, 97.5)))

    def paired_ratio(self, a, b, groups):
        """median(a)/median(b) (a, b paired element-wise) and its cluster-
        bootstrap 95 % CI (percentiles of log10 ratios); CI NaN with fewer
        than MIN_CLUSTERS seeds."""
        a, b = _censor(a), _censor(b)
        g = np.asarray(groups)
        ok = ~(np.isnan(a) | np.isnan(b))
        a, b, g = a[ok], b[ok], g[ok]
        if len(a) == 0:
            return (np.nan, np.nan, np.nan)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            r = _uncensor_ratio(np.median(a) / np.median(b))
            Ma, Mb = _cluster_matrix(a, g), _cluster_matrix(b, g)
            if Ma.shape[0] < MIN_CLUSTERS:
                return (r, np.nan, np.nan)
            D = self._draw(Ma.shape[0])
            rr = _med_rows(Ma[D].reshape(self.n, -1)) / _med_rows(Mb[D].reshape(self.n, -1))
            lr = np.log10(rr)
            lr = lr[~np.isnan(lr)]
            if len(lr) == 0:
                return (r, np.nan, np.nan)
            return (r, _uncensor_ratio(10 ** np.percentile(lr, 2.5)),
                    _uncensor_ratio(10 ** np.percentile(lr, 97.5)))


def cell_table(df, boot):
    """Per-cell statistics (one row per problem/protocol/info/sigma/disc/tau)."""
    rows, arrays = [], {}
    gcols = ["problem", "protocol", "info_reg", "sigma", "disc", "tau"]
    for key, g in df.sort_values(["seed", "traj"]).groupby(gcols, sort=True):
        x = g["v"].to_numpy(float)
        xm = g["v_mean"].to_numpy(float)
        pid = list(zip(g["seed"].astype(int), g["traj"].astype(int)))
        arrays[key] = (x, pid)
        seeds = np.array([q[0] for q in pid])
        keep = ~np.isnan(x)              # NaN = failed run (counted in n_fail, never silently)
        fin = x[keep]
        n_nan = int((~keep).sum())
        n_div = int(np.isinf(x).sum())
        if len(fin):
            c = _censor(fin)             # diverged = worst case inside median, IQR and CI
            med = _uncensor(np.median(c))
            q25, q75 = _uncensor(np.percentile(c, 25)), _uncensor(np.percentile(c, 75))
            lo, hi = boot.median_ci(fin, seeds[keep])
        else:
            med = q25 = q75 = lo = hi = np.nan
        # within-seed spread of the B initial states: ~0 -> replicas (pseudo-replication);
        # large -> x0-dependent steady state (transient leaking into the window)
        spread = []
        for sd in np.unique(seeds[keep]):
            v = x[keep & (seeds == sd)]
            if len(v) >= 2:
                spread.append(np.inf if not np.isfinite(v).all() and np.isfinite(v).any()
                              else (0.0 if not np.isfinite(v).any() else float(v.max() / v.min() - 1.0)))
        fm = xm[~np.isnan(xm)]
        rows.append(dict(zip(gcols, key), n=len(x), n_seeds=int(len(np.unique(seeds[keep]))),
                         n_div=n_div, n_fail=n_nan, median=med, q25=q25,
                         q75=q75, ci_lo=lo, ci_hi=hi,
                         x0_spread=float(max(spread)) if spread else np.nan,
                         median_mean_ssre=_uncensor(np.median(_censor(fm))) if len(fm) else np.nan,
                         expected_order=int(g["disc_expected_order"].iloc[0])
                         if pd.notna(g["disc_expected_order"].iloc[0]) else None,
                         h=float(g["h"].iloc[0])))
    return pd.DataFrame(rows), arrays


def paired(arrays, key_a, key_b):
    xa, pa = arrays[key_a]
    xb, pb = arrays[key_b]
    common = sorted(set(pa) & set(pb))
    ia = {p: i for i, p in enumerate(pa)}
    ib = {p: i for i, p in enumerate(pb)}
    return (np.array([xa[ia[p]] for p in common]), np.array([xb[ib[p]] for p in common]),
            np.array([p[0] for p in common]))


def local_slopes(tau, m):
    lt, lm = np.log10(tau), np.log10(m)
    return np.diff(lm) / np.diff(lt)


def asymptotic_order(tau, m, floor=FLOOR, q=None, tol=TOL_SLOPE):
    """Empirical order on the asymptotic (small-tau) range.

    tau, m: medians, any order.  Rules (review 2026-10-18; the previous rule
    dropped "flattening" tails relative to the LARGEST local slope, so a steep
    pre-asymptotic segment at large tau -- e.g. zead_11i_chen2023 at tau = 0.1,
    local slope 15.5 -- removed the asymptotic points and a 2-point slope of
    15.5 was reported):

    1. non-finite medians and points at or below the round-off floor are
       dropped; only the trailing contiguous run of usable points (the
       small-tau end of the tau grid) is used, so pre-asymptotic or unstable
       large-tau points separated by a divergence never enter;
    2. no further floor-onset trimming: the floor (1e-12) sits >= 300x above
       the measured round-off, and a relative-drop rule mistook ordinary
       pre-asymptotic curvature (local slopes 6.23 -> 5.95) for floor onset;
    3. the window grows from the two smallest-tau points towards larger tau
       while EVERY consecutive-ratio (local) slope in it lies within ``tol``
       (0.15, the tolerance of tests/test_problems_robot.py) of the window's
       least-squares slope; pre-asymptotic points therefore stop the window;
    4. flag 'asym' for >= 3 points, '2pt' for a 2-point estimate (reported
       with a dagger / open marker); 'absurd' marks a slope outside
       [0, q + 3] (q = nominal order), for the asymptotic and the all-points
       fit separately.

    Returns slope, slope_all (LS over all finite above-floor points), n_used,
    tau_range [min, max] of the window, local slopes of the trailing run and
    the flags."""
    tau = np.asarray(tau, float)
    m = np.asarray(m, float)
    o = np.argsort(-tau)
    tau, m = tau[o], m[o]
    ok = np.isfinite(m) & (m > floor)
    out = {"slope": np.nan, "slope_all": np.nan, "n_used": 0, "tau_range": None,
           "local": [], "flag": "none", "absurd": False, "absurd_all": False}
    if ok.sum() >= 2:
        lt, lm = np.log10(tau[ok]), np.log10(m[ok])
        out["slope_all"] = float(np.polyfit(lt, lm, 1)[0])
        if q is not None:
            out["absurd_all"] = bool(out["slope_all"] > q + 3 or out["slope_all"] < 0)
    idx = np.where(ok)[0]
    if len(idx) < 2:
        return out
    i0 = len(idx) - 1                     # trailing contiguous run (small-tau end)
    while i0 >= 1 and idx[i0] - idx[i0 - 1] == 1:
        i0 -= 1
    run = idx[i0:]
    if len(run) < 2:
        return out
    loc = local_slopes(tau[run], m[run])
    out["local"] = [float(s) for s in loc]
    L, s = 2, float(loc[-1])
    while L < len(run):
        cand = run[-(L + 1):]
        s_c = float(np.polyfit(np.log10(tau[cand]), np.log10(m[cand]), 1)[0])
        if np.all(np.abs(loc[-L:] - s_c) <= tol):
            L, s = L + 1, s_c
        else:
            break
    win = run[-L:]
    out.update(slope=s, n_used=int(L), flag="asym" if L >= 3 else "2pt",
               tau_range=[float(tau[win[-1]]), float(tau[win[0]])])
    if q is not None:
        out["absurd"] = bool(s > q + 3 or s < 0)
    return out


def kendall_b(a, b):
    from scipy.stats import kendalltau
    a, b = np.asarray(a, float), np.asarray(b, float)
    big = 1e300
    a = np.where(np.isinf(a), big, a)
    b = np.where(np.isinf(b), big, b)
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() < 3:
        return np.nan
    r = kendalltau(a[ok], b[ok], variant="b")
    return float(r.statistic if hasattr(r, "statistic") else r[0])


# =============================================================================
# noise-gain prediction (linear error recursion)
# =============================================================================
def estimator_fir(info):
    """Dimensionless FIR weights tau * w_j of the dE/dt estimate, j = 0 the
    newest sample (None for analytic information)."""
    from znnbench.wrappers import alpha_beta_weights, kalman_cv_weights, savgol_weights
    if info == "analytic":
        return None
    if info == "bd1":
        return np.array([1.0, -1.0])
    if info == "bd2":
        return np.array([1.5, -2.0, 0.5])
    if info in ("savgol", "bd5"):
        L, order = (5, 2) if info == "savgol" else (6, 5)
        w = savgol_weights(-np.arange(L - 1, -1, -1.0), order)
    elif info == "alpha_beta":
        w = alpha_beta_weights(-np.arange(19, -1, -1.0), 0.5, 0.5 ** 2 / 1.5)
    elif info == "kalman":
        w = kalman_cv_weights(-np.arange(19, -1, -1.0), 1.0, 1.0)   # Kalata index 1 at unit spacing
    else:
        return None
    return np.asarray(w, float)[::-1]


def predicted_gain(disc, info, h, n_imp=40000):
    """H2 gain ||F/P_h||_2 of the linear-ZNN error recursion driven by white
    measurement noise n_k on E:  P_h(z) e = -(h + D(z)) n, where P_h is the
    look-ahead error polynomial (zead.lookahead_znn_poly) and D(z) the
    estimator's dimensionless FIR (BD1: 1 - z^-1).  It predicts how the noise
    floor scales across formulas and estimators (the problem-dependent factor
    -- noise on A and b mapped into E, max vs RMS over the window -- is common
    to all formulas of a problem).  NaN for formulas without coefficients."""
    from znnbench import make_discretization
    from znnbench.discretizations.zead import lookahead_znn_poly, spectral_radius
    try:
        c = make_discretization(disc).describe().get("coeffs")
    except Exception:  # noqa: BLE001
        return np.nan
    if c is None or not np.isfinite(h):
        return np.nan
    P = np.array(lookahead_znn_poly(c, h), float)
    if spectral_radius(P) >= 1.0:
        return np.inf
    from scipy.signal import lfilter
    F = np.zeros(1)
    F[0] = h
    d = estimator_fir(info)
    if d is not None:
        F = np.zeros(max(1, len(d)))
        F[:len(d)] += d
        F[0] += h
    imp = np.zeros(n_imp)
    imp[0] = 1.0
    r = lfilter(F, P, imp)       # P[0] e_k + P[1] e_{k-1} + ... = F[0] n_k + F[1] n_{k-1} + ...
    return float(np.sqrt(np.sum(r ** 2)))


# =============================================================================
# analysis core
# =============================================================================
class E6:
    def __init__(self, df, cells, arrays, boot):
        self.df, self.cells, self.arrays, self.boot = df, cells, arrays, boot
        orders = cells.groupby("disc")["expected_order"].first().to_dict()
        self.order = orders
        self.discs = sorted(orders, key=lambda k: (orders[k] or 0, family_index(k), k))
        self.high = [k for k in self.discs if k not in LOW_ORDER]
        self.low = [k for k in self.discs if k in LOW_ORDER]
        self.infos = [i for i in INFO_ORDER if i in set(cells.info_reg)] + \
            sorted(set(cells.info_reg) - set(INFO_ORDER))
        self.sigmas = sorted(cells.sigma.unique())
        self.taus = sorted(cells.tau.unique(), reverse=True)
        self.combos = [(p, pr) for p in sorted(cells.problem.unique())
                       for pr in ("common", "hopt")
                       if ((cells.problem == p) & (cells.protocol == pr)).any()]
        self.med = {tuple(r[c] for c in ["problem", "protocol", "info_reg", "sigma", "disc", "tau"]):
                    r["median"] for _, r in cells.iterrows()}
        self.tau_ref = 1e-3 if any(np.isclose(self.taus, 1e-3)) else min(self.taus)
        self.tau_ref = [t for t in self.taus if np.isclose(t, self.tau_ref)][0]
        self._gp = {}

    def discs_in(self, p, pr):
        """formulas run in a (problem, protocol) combination, in canonical order."""
        c = self.cells
        have = set(c[(c.problem == p) & (c.protocol == pr)].disc)
        return [d for d in self.discs if d in have]

    def m(self, p, pr, i, s, d, t):
        return self.med.get((p, pr, i, s, d, t), np.nan)

    def best_cell(self, p, pr, i, s, t):
        """(best key, runner-up key, significant?, ratio runner/best, CI)"""
        vals = [(self.m(p, pr, i, s, d, t), d) for d in self.discs]
        vals = [(v, d) for v, d in vals if not np.isnan(v)]
        if not vals:
            return None
        vals.sort(key=lambda x: (x[0], self.discs.index(x[1])))
        if not np.isfinite(vals[0][0]):
            return {"best": None, "all_div": True}
        b = vals[0][1]
        if len(vals) == 1:
            return {"best": b, "second": None, "sig": True, "ratio": np.nan, "ci": (np.nan, np.nan)}
        r2 = vals[1][1]
        a2, ab, gs = paired(self.arrays, (p, pr, i, s, r2, t), (p, pr, i, s, b, t))
        r, lo, hi = self.boot.paired_ratio(a2, ab, gs)
        k = _n_clusters(gs)
        if np.isinf(vals[1][0]):
            sig = True                   # runner-up diverged (median), winner finite
        elif k < MIN_CLUSTERS:
            sig = None                   # not assessable (too few seed clusters), never "not significant"
        else:
            sig = bool(np.isfinite(lo) and lo > 1.0)
        return {"best": b, "second": r2, "sig": sig, "ratio": r, "ci": (lo, hi), "all_div": False,
                "n_seeds": k}

    def regime_maps(self):
        out = {}
        for p, pr in self.combos:
            for i in self.infos:
                for s in self.sigmas:
                    for t in self.taus:
                        bc = self.best_cell(p, pr, i, s, t)
                        if bc is not None:
                            out[(p, pr, i, s, t)] = bc
        self.regime = out
        return out

    def advantage(self, p, pr, i, s, t):
        lo = [self.m(p, pr, i, s, d, t) for d in self.low]
        hi = [self.m(p, pr, i, s, d, t) for d in self.high]
        lo = [v for v in lo if not np.isnan(v)]
        hi = [v for v in hi if not np.isnan(v)]
        if not lo or not hi:
            return np.nan
        a, b = min(lo), min(hi)
        with np.errstate(divide="ignore", invalid="ignore"):
            if np.isinf(a) and np.isinf(b):
                return np.nan
            return float(np.log10(a / b))

    def advantage_ci(self, p, pr, i, s, t):
        """A = log10(min_low median / min_high median) and its paired seed-cluster
        bootstrap 95 % CI, with the min over formulas RE-TAKEN in every resample (so
        the CI accounts for picking the best low- and high-order formula on the same
        data).  Returns (A, lo, hi, n_seeds); lo > 0: high order significantly better,
        hi < 0: low order significantly better.  Diverged = BIG (worst case)."""
        ks = [(d, (p, pr, i, s, d, t)) for d in self.low + self.high if (p, pr, i, s, d, t) in self.arrays]
        lows = [k for d, k in ks if d in self.low]
        highs = [k for d, k in ks if d in self.high]
        if not lows or not highs:
            return (np.nan, np.nan, np.nan, 0)
        common = sorted(set.intersection(*[set(self.arrays[k][1]) for _, k in ks]))
        if not common:
            return (np.nan, np.nan, np.nan, 0)
        cols = {}
        for _, k in ks:
            x, pid = self.arrays[k]
            ix = {q: j for j, q in enumerate(pid)}
            cols[k] = _censor(np.array([x[ix[q]] for q in common]))
        g = np.array([q[0] for q in common])
        nseed = _n_clusters(g)

        def a_of(med):
            lo_ = min(med[k] for k in lows)
            hi_ = min(med[k] for k in highs)
            return np.log10(lo_ / hi_)

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            a0 = a_of({k: np.median(v) for k, v in cols.items()})
            a0 = float(np.inf if a0 > 200 else (-np.inf if a0 < -200 else a0))
            if nseed < MIN_CLUSTERS:
                return (a0, np.nan, np.nan, nseed)
            D = self.boot._draw(nseed)
            meds = {k: _med_rows(_cluster_matrix(v, g)[D].reshape(self.boot.n, -1)) for k, v in cols.items()}
            lo_m = np.min(np.stack([meds[k] for k in lows]), axis=0)
            hi_m = np.min(np.stack([meds[k] for k in highs]), axis=0)
            a = np.log10(lo_m / hi_m)
            a = a[~np.isnan(a)]
            if len(a) == 0:
                return (a0, np.nan, np.nan, nseed)
            lo, hi = np.percentile(a, [2.5, 97.5])
        clip = lambda v: float(np.inf if v > 200 else (-np.inf if v < -200 else v))  # noqa: E731
        return (a0, clip(lo), clip(hi), nseed)

    def advantage_cis(self):
        self.adv = {}
        for p, pr in self.combos:
            for i in self.infos:
                for s in self.sigmas:
                    for t in self.taus:
                        self.adv[(p, pr, i, s, t)] = self.advantage_ci(p, pr, i, s, t)
        return self.adv

    def orders_sigma0(self):
        res = {}
        for p, pr in self.combos:
            for i in self.infos:
                for d in self.discs:
                    sub = self.cells[(self.cells.problem == p) & (self.cells.protocol == pr) &
                                     (self.cells.info_reg == i) & (self.cells.sigma == 0) &
                                     (self.cells.disc == d)]
                    if len(sub) < 2:
                        continue
                    fit = asymptotic_order(sub.tau.to_numpy(), sub["median"].to_numpy(), q=self.order.get(d))
                    fit["ci"] = self.slope_ci(p, pr, i, d, fit)
                    res[(p, pr, i, d)] = fit
        self.orders = res
        return res

    def tau_asym(self, p, pr):
        """largest tau inside the sigma = 0, analytic-information asymptotic window of EVERY
        formula of (p, pr): above it at least one formula is pre-asymptotic or unstable, and
        the constant-coefficient error recursion behind the H2-gain theory does not apply."""
        his = []
        for d in self.discs_in(p, pr):
            f = getattr(self, "orders", {}).get((p, pr, "analytic", d))
            if f and f["tau_range"]:
                his.append(f["tau_range"][1])
        return min(his) if his else max(self.taus)

    def slope_ci(self, p, pr, i, d, fit):
        """cluster-bootstrap CI of the asymptotic slope (seeds resampled jointly over tau)."""
        if fit["flag"] != "asym" or fit["tau_range"] is None:
            return (np.nan, np.nan)
        t_lo, t_hi = fit["tau_range"]
        ts = [t for t in self.taus if t_lo * (1 - 1e-9) <= t <= t_hi * (1 + 1e-9)
              and (p, pr, i, 0.0, d, t) in self.arrays]
        xs = [self.arrays[(p, pr, i, 0.0, d, t)] for t in ts]
        if len(ts) < 2 or any((~np.isfinite(x)).any() for x, _ in xs):
            return (np.nan, np.nan)
        mats = [_cluster_matrix(x, [q[0] for q in pid]) for x, pid in xs]
        k = mats[0].shape[0]
        if k < MIN_CLUSTERS or any(M.shape[0] != k for M in mats):
            return (np.nan, np.nan)
        D = self.boot._draw(k)          # the same seeds for every tau (same instances)
        lt = np.log10(ts)
        meds = np.stack([_med_rows(M[D].reshape(self.boot.n, -1)) for M in mats], axis=1)
        A = np.vstack([lt, np.ones_like(lt)]).T
        sl = np.linalg.lstsq(A, np.log10(meds).T, rcond=None)[0][0]
        return (float(np.percentile(sl, 2.5)), float(np.percentile(sl, 97.5)))

    def floor_slopes(self):
        """tau-slope of MSSRE in clearly noise-dominated cells (MSSRE(sigma) >=
        NOISE_DOM_SLOPE x MSSRE(0), i.e. truncation share <~ 5 %; with the looser 5x
        rule a truncation share of 20 % biases the slope by up to 0.2 q):
        0 -> floor ~ sigma*G (tau independent at fixed h); -1 -> floor ~ sigma/tau.
        Two versions: 'asym' uses tau <= tau_asym(problem, protocol) (>= 2 points),
        'all' every tau (>= 3 points, includes pre-asymptotic large-tau cells where
        the high-order formulas sit near instability and amplify noise more)."""
        res = []
        for p, pr in self.combos:
            t_as = self.tau_asym(p, pr)
            for i in self.infos:
                for d in self.discs_in(p, pr):
                    for s in self.sigmas:
                        if s == 0:
                            continue
                        ts, ms = [], []
                        for t in self.taus:
                            a, a0 = self.m(p, pr, i, s, d, t), self.m(p, pr, i, 0.0, d, t)
                            if np.isfinite(a) and np.isfinite(a0) and a >= NOISE_DOM_SLOPE * a0:
                                ts.append(t)
                                ms.append(a)
                        ts, ms = np.array(ts), np.array(ms)
                        for kind, sel, nmin in (("all", np.ones(len(ts), bool), 3),
                                                ("asym", ts <= t_as * (1 + 1e-9), 2)):
                            if sel.sum() >= nmin:
                                sl = float(np.polyfit(np.log10(ts[sel]), np.log10(ms[sel]), 1)[0])
                                res.append(dict(problem=p, protocol=pr, info=i, disc=d, sigma=s, kind=kind,
                                                slope=sl, n=int(sel.sum()), tau_asym=t_as))
        self.fslopes = pd.DataFrame(res)
        return self.fslopes

    def amplification(self):
        """G = MSSRE / sigma at the smallest tau, in noise-dominated cells."""
        t = min(self.taus)
        res = []
        for p, pr in self.combos:
            for i in self.infos:
                for d in self.discs_in(p, pr):
                    for s in self.sigmas:
                        if s == 0:
                            continue
                        a, a0 = self.m(p, pr, i, s, d, t), self.m(p, pr, i, 0.0, d, t)
                        dom = bool(np.isfinite(a) and np.isfinite(a0) and a >= NOISE_DOM * a0)
                        h = self.h_of(p, pr, d)
                        key = (d, i, round(h, 6))
                        if key not in self._gp:
                            self._gp[key] = predicted_gain(d, i, h)
                        res.append(dict(problem=p, protocol=pr, info=i, disc=d, sigma=s, tau=t, h=h,
                                        G=a / s if np.isfinite(a) else np.inf, noise_dominated=dom,
                                        G_pred=self._gp[key]))
        self.amp = pd.DataFrame(res)
        return self.amp

    def h_of(self, p, pr, d):
        c = self.cells
        q = c[(c.problem == p) & (c.protocol == pr) & (c.disc == d)]
        return float(q.h.iloc[0]) if len(q) else np.nan

    def gain_fit(self):
        """log G = log kappa + b log G_pred on noise-dominated cells (all sigma > 0,
        smallest tau), per (problem, protocol): slope b, R^2, kappa."""
        out = {}
        a = self.amp
        for p, pr in self.combos:
            q = a[(a.problem == p) & (a.protocol == pr) & a.noise_dominated &
                  np.isfinite(a.G) & np.isfinite(a.G_pred) & (a.G_pred > 0)]
            if len(q) < 3:
                continue
            x, y = np.log10(q.G_pred.to_numpy()), np.log10(q.G.to_numpy())
            b, c0 = np.polyfit(x, y, 1)
            r2 = 1 - np.sum((y - (b * x + c0)) ** 2) / np.sum((y - y.mean()) ** 2)
            k1 = float(np.median(y - x))
            r2_1 = 1 - np.sum((y - (x + k1)) ** 2) / np.sum((y - y.mean()) ** 2)
            out[f"{p}|{pr}"] = {"slope": float(b), "r2": float(r2), "kappa_slope1": float(10 ** k1),
                                "r2_slope1": float(r2_1), "n": int(len(q))}
        self.gfit = out
        return out

    def headline(self):
        """Euler vs best high-order formula at tau_ref, sigma = 0 and sigma_max."""
        t = self.tau_ref
        res = []
        for p, pr in self.combos:
            for i in self.infos:
                for s in sorted({0.0, max(self.sigmas)}):
                    hv = [(self.m(p, pr, i, s, d, t), d) for d in self.high]
                    hv = [x for x in hv if not np.isnan(x[0])]
                    if not hv or "euler" not in self.discs:
                        continue
                    bh = min(hv)[1]
                    try:
                        a, b, gs = paired(self.arrays, (p, pr, i, s, "euler", t), (p, pr, i, s, bh, t))
                    except KeyError:
                        continue
                    r, lo, hi = self.boot.paired_ratio(a, b, gs)
                    res.append(dict(problem=p, protocol=pr, info=i, sigma=s, tau=t, best_high=bh,
                                    ratio_euler_over_high=r, ci_lo=lo, ci_hi=hi, n_seeds=_n_clusters(gs),
                                    euler_median=_uncensor(np.median(_censor(a))) if len(a) else np.nan,
                                    high_median=_uncensor(np.median(_censor(b))) if len(b) else np.nan))
        self.head = pd.DataFrame(res)
        return self.head

    def kendall(self):
        t = self.tau_ref
        res = []
        for p, pr in self.combos:
            ref = [self.m(p, pr, "analytic", 0.0, d, t) for d in self.discs]
            for i in self.infos:
                for s in self.sigmas:
                    cur = [self.m(p, pr, i, s, d, t) for d in self.discs]
                    res.append(dict(problem=p, protocol=pr, info=i, sigma=s, tau=t,
                                    kendall_tau_b=kendall_b(ref, cur)))
        self.kt = pd.DataFrame(res)
        return self.kt

    def crossovers(self):
        """largest tau at which the high-order advantage turns negative, per sigma."""
        res = []
        for p, pr in self.combos:
            for i in self.infos:
                for s in self.sigmas:
                    A = [(t, self.advantage(p, pr, i, s, t)) for t in self.taus]   # tau descending
                    # +-inf (one side diverged) carries a sign and is kept (censoring);
                    # only NaN (nothing comparable) is dropped
                    A = [(t, a) for t, a in A if not np.isnan(a)]
                    tstar, kind = np.nan, "none"
                    if A and all(a <= 0 for _, a in A):
                        kind = "all"            # no advantage at any tau
                    elif A and all(a > 0 for _, a in A):
                        kind = "never"          # advantage kept at every tau
                    else:
                        for (t1, a1), (t2, a2) in zip(A[:-1], A[1:]):
                            if a1 > 0 >= a2:
                                if np.isfinite(a1) and np.isfinite(a2):
                                    w = a1 / (a1 - a2)
                                else:           # a diverged side: crossing somewhere in (t2, t1)
                                    w = 0.5
                                tstar = float(10 ** (np.log10(t1) + w * (np.log10(t2) - np.log10(t1))))
                                kind = "crossing"
                                break
                        else:
                            kind = "gained"     # only a -/+ change (advantage appears at small tau)
                    fin = [a for _, a in A if np.isfinite(a)]
                    res.append(dict(problem=p, protocol=pr, info=i, sigma=s, tau_star=tstar, kind=kind,
                                    adv_at_tau_min=A[-1][1] if A else np.nan,
                                    adv_max=max(fin) if fin else np.nan))
        self.cross = pd.DataFrame(res)
        return self.cross


# =============================================================================
# figures
# =============================================================================
def _savefig(fig, figdir, name):
    style.FIG_DIR = Path(figdir)
    style.savefig(fig, name)
    return f"{name}.pdf"


def _grid_ticks(ax, taus, sigmas):
    ax.set_xticks(range(len(taus)))
    ax.set_xticklabels([tau_lab(t) for t in taus], rotation=90)
    ax.set_yticks(range(len(sigmas)))
    ax.set_yticklabels([sig_lab(s) for s in sigmas])
    ax.tick_params(length=0, pad=1.5)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


def _luma(rgb):
    r, g, b = rgb
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def fig_regime(e, p, pr, figdir):
    infos = e.infos
    ncol = 4
    nrow = math.ceil((len(infos) + 1) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(style.DOUBLE, 1.62 * nrow + 0.25), squeeze=False)
    sig_desc = sorted(e.sigmas, reverse=True)      # top row = largest sigma
    for k, i in enumerate(infos):
        ax = axes.flat[k]
        img = np.ones((len(sig_desc), len(e.taus), 3))
        ties, divs, unk = [], [], []
        for r, s in enumerate(sig_desc):
            for c, t in enumerate(e.taus):
                bc = e.regime.get((p, pr, i, s, t))
                if bc is None:
                    continue
                if bc.get("all_div"):
                    img[r, c] = to_rgb(DIV_FILL)
                    divs.append((r, c))
                    continue
                rgb = np.array(to_rgb(disc_style(bc["best"])["color"]))
                img[r, c] = rgb
                if bc["sig"] is False:
                    ties.append((r, c))
                elif bc["sig"] is None:
                    unk.append((r, c))
                ax.text(c, r, short_label(bc["best"]), ha="center", va="center", fontsize=4.4,
                        color="white" if _luma(rgb) < 0.5 else style.INK,
                        zorder=4, bbox=None if bc["sig"] else dict(boxstyle="square,pad=0.12",
                                                                   fc=tuple(rgb), ec="none"))
        ax.imshow(img, aspect="auto", interpolation="nearest")
        with plt.rc_context({"hatch.linewidth": 0.5, "hatch.color": "white"}):
            for r, c in ties:
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, hatch="////",
                                       edgecolor="white", linewidth=0.0))
            for r, c in unk:
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, hatch="....",
                                       edgecolor="white", linewidth=0.0))
        for r, c in divs:
            ax.text(c, r, "div", ha="center", va="center", fontsize=4.2, color=style.INK2)
        _grid_ticks(ax, e.taus, sig_desc)
        ax.set_title(INFO_LABEL.get(i, i), pad=2)
        if k % ncol == 0:
            ax.set_ylabel(r"noise $\sigma$")
        if k // ncol == nrow - 1 or k + ncol >= len(infos):
            ax.set_xlabel(r"$\tau$ (s)")
    for ax in axes.flat[len(infos):]:
        ax.axis("off")
    lax = axes.flat[len(infos)] if len(infos) < nrow * ncol else None
    handles = [Patch(facecolor=disc_style(d)["color"], edgecolor="none",
                     label=f"{short_label(d)}: {long_label(d, e.order[d])}") for d in e.discs_in(p, pr)]
    # neutral (white) swatch with dark hatching: a grey swatch would read as the grey
    # 'other' formula colour (zead_11i_chen2023)
    handles.append(Patch(facecolor="white", edgecolor=style.INK2, hatch="////", linewidth=0.4,
                         label="hatched: not sig. better than runner-up"))
    if any(v.get("sig", True) is None for (pp, ppr, *_), v in e.regime.items() if (pp, ppr) == (p, pr)):
        handles.append(Patch(facecolor="white", edgecolor=style.INK2, hatch="....", linewidth=0.4,
                             label=f"dotted: significance n/a (< {MIN_CLUSTERS} seeds)"))
    handles.append(Patch(facecolor=DIV_FILL, edgecolor=style.INK2, linewidth=0.4,
                         label="div: all formulas diverged"))
    if lax is not None:
        lax.legend(handles=handles, loc="center left", frameon=False, fontsize=5.6, handlelength=1.3,
                   title=f"{p}, {PROTO_LABEL[pr]}", title_fontsize=6.0)
    else:
        fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=5.6)
    fig.tight_layout(pad=0.3, w_pad=0.6, h_pad=0.8)
    return _savefig(fig, figdir, f"fig_E6_regime_{p}_{pr}")


def fig_advantage(e, p, pr, figdir):
    infos = e.infos
    ncol = 4
    nrow = math.ceil(len(infos) / ncol)
    sig_desc = sorted(e.sigmas, reverse=True)
    A = np.full((len(infos), len(sig_desc), len(e.taus)), np.nan)
    for k, i in enumerate(infos):
        for r, s in enumerate(sig_desc):
            for c, t in enumerate(e.taus):
                A[k, r, c] = e.advantage(p, pr, i, s, t)
    fin = np.abs(A[np.isfinite(A)])
    vmax = float(np.clip(np.ceil(np.nanmax(fin)) if fin.size else 1.0, 1.0, 8.0))
    fig, axes = plt.subplots(nrow, ncol, figsize=(style.DOUBLE, 1.62 * nrow + 0.35), squeeze=False)
    im = None
    for k, i in enumerate(infos):
        ax = axes.flat[k]
        Ak = np.clip(A[k], -vmax, vmax)
        im = ax.imshow(np.ma.masked_invalid(Ak), cmap=style.CMAP_DIV, vmin=-vmax, vmax=vmax,
                       aspect="auto", interpolation="nearest")
        for r in range(len(sig_desc)):
            for c in range(len(e.taus)):
                a = A[k, r, c]
                if np.isnan(a):
                    continue
                txt = (f"{a:.1f}".replace("-0.0", "0.0") if np.isfinite(a)
                       else ("+inf" if a > 0 else "-inf"))
                _, alo, ahi, _ = getattr(e, "adv", {}).get((p, pr, i, sig_desc[r], e.taus[c]),
                                                         (np.nan, np.nan, np.nan, 0))
                if np.isfinite(alo) and np.isfinite(ahi) and alo <= 0 <= ahi:
                    txt = f"({txt})"     # paired cluster-bootstrap CI of A contains 0
                ax.text(c, r, txt, ha="center", va="center", fontsize=4.0,
                        color="white" if abs(np.clip(a, -vmax, vmax)) > 0.6 * vmax else style.INK)
        _grid_ticks(ax, e.taus, sig_desc)
        ax.set_title(INFO_LABEL.get(i, i), pad=2)
        if k % ncol == 0:
            ax.set_ylabel(r"noise $\sigma$")
        if k + ncol >= len(infos):
            ax.set_xlabel(r"$\tau$ (s)")
    for ax in axes.flat[len(infos):]:
        ax.axis("off")
    fig.tight_layout(pad=0.3, w_pad=0.6, h_pad=0.8, rect=(0, 0, 0.93, 0.95))
    cax = fig.add_axes((0.945, 0.15, 0.012, 0.65))
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r"$\log_{10}$(MSSRE$_{\rm low}$ / MSSRE$_{\rm high}$)" +
                 "\n> 0 high order better, < 0 low order better", fontsize=6)
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_visible(False)
    lo = "/".join(short_label(d) for d in e.low if d in e.discs_in(p, pr))
    hi = "/".join(short_label(d) for d in e.high if d in e.discs_in(p, pr))
    fig.suptitle(f"{p}, {PROTO_PLAIN[pr]}: best of {{{lo}}} vs best of {{{hi}}}; "
                 "(value) = 95 % CI contains 0", fontsize=7, y=0.995)
    return _savefig(fig, figdir, f"fig_E6_advantage_{p}_{pr}")


def fig_mssre_tau(e, p, pr, infos, tag, figdir):
    infos = [i for i in infos if i in e.infos]
    if not infos:
        return None
    sig = e.sigmas
    fig, axes = plt.subplots(len(infos), len(sig), figsize=(style.DOUBLE, 1.18 * len(infos) + 0.5),
                             sharex=True, sharey=True, squeeze=False)
    c = e.cells
    sub = c[(c.problem == p) & (c.protocol == pr)]
    allv = sub[np.isfinite(sub["median"])]["median"]
    ylo = max(1e-16, 10 ** np.floor(np.log10(max(allv.min(), 1e-16)))) if len(allv) else 1e-16
    yhi = 10 ** np.ceil(np.log10(allv.max())) if len(allv) else 1.0
    any_cens = False
    for r, i in enumerate(infos):
        for k, s in enumerate(sig):
            ax = axes[r, k]
            for d in e.discs:
                q_all = sub[(sub.info_reg == i) & (sub.sigma == s) & (sub.disc == d)].sort_values("tau")
                q = q_all[np.isfinite(q_all["median"])]
                cens = q_all[np.isposinf(q_all["median"])]
                st = disc_style(d)
                if not cens.empty:       # censored: median diverged -> open marker on the top edge
                    any_cens = True
                    ax.plot(cens.tau, np.full(len(cens), yhi), linestyle="none", marker=st["marker"],
                            color=st["color"], markerfacecolor="none", markersize=_msize(st, 2.6),
                            clip_on=False, zorder=5)
                if q.empty:
                    continue
                ax.plot(q.tau, q["median"], label=long_label(d, e.order[d]), markersize=_msize(st, 2.6),
                        linewidth=0.9, **st)
                ok = np.isfinite(q.q25) & np.isfinite(q.q75)
                ax.fill_between(q.tau[ok], q.q25[ok], q.q75[ok], color=st["color"], alpha=0.15,
                                linewidth=0)
            if s > 0:
                ax.axhline(s, **SIGMA_LINE)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_ylim(ylo, yhi)
            ax.tick_params(labelsize=5.5)
            if r == 0:
                ax.set_title(rf"$\sigma$ = {sig_lab(s)}", pad=2, fontsize=7)
            if k == 0:
                ax.set_ylabel(f"{INFO_LABEL.get(i, i)}\nMSSRE", fontsize=6.5)
            if r == len(infos) - 1:
                ax.set_xlabel(r"$\tau$ (s)", fontsize=6.5)
            ax.minorticks_off()
    h, l = [], []
    for ax in axes.flat:              # union over panels (a formula may be censored in panel [0, 0])
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh)
                l.append(ll)
    order = {long_label(d, e.order[d]): j for j, d in enumerate(e.discs)}
    hl = sorted(zip(h, l), key=lambda x: order.get(x[1], 99))
    h, l = [x[0] for x in hl], [x[1] for x in hl]
    h.append(plt.Line2D([], [], **SIGMA_LINE))
    l.append(r"noise level $\sigma$")
    if any_cens:
        h.append(plt.Line2D([], [], linestyle="none", marker="o", markerfacecolor="none", color=style.INK2))
        l.append("open, top edge: median diverged")
    fig.legend(h, l, loc="upper center", ncol=len(l) if len(l) <= 6 else math.ceil(len(l) / 2), fontsize=6,
               bbox_to_anchor=(0.5, 1.0), columnspacing=1.0)
    fig.tight_layout(pad=0.3, h_pad=0.4, w_pad=0.3, rect=(0, 0, 1, 1 - 0.32 / (1.18 * len(infos) + 0.5)))
    return _savefig(fig, figdir, f"fig_E6_mssre_tau_{tag}_{p}_{pr}")


def fig_amplification(e, figdir):
    amp = e.amp
    if amp.empty:
        return None
    s_hi = max(e.sigmas)
    combos = e.combos
    fig, axes = plt.subplots(1, len(combos), figsize=(style.DOUBLE, 2.3), sharey=True, squeeze=False)
    n_inf = len(e.infos)
    width = 0.8 / max(n_inf, 1)
    for k, (p, pr) in enumerate(combos):
        ax = axes[0, k]
        dl = e.discs_in(p, pr)
        x = np.arange(len(dl))
        for j, i in enumerate(e.infos):
            q = amp[(amp.problem == p) & (amp.protocol == pr) & (amp["info"] == i) & (amp.sigma == s_hi)]
            q = q.set_index("disc").reindex(dl)
            st = info_style(i)
            G = q.G.to_numpy(float)
            dom = q.noise_dominated.fillna(False).to_numpy(bool)
            xs = x - 0.4 + width * (j + 0.5)
            ax.plot(xs[dom], G[dom], linestyle="none", marker=st["marker"], color=st["color"],
                    markersize=_msize(st), label=INFO_LABEL.get(i, i))
            ax.plot(xs[~dom & np.isfinite(G)], G[~dom & np.isfinite(G)], linestyle="none",
                    marker=st["marker"], markerfacecolor="none", color=st["color"], markersize=_msize(st))
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([short_label(d) for d in dl])
        ax.set_title(f"{p}\n{PROTO_PLAIN[pr]}", fontsize=6.5, pad=2)
        if k == 0:
            ax.set_ylabel(rf"MSSRE / $\sigma$ at $\tau$ = {tau_lab(min(e.taus))}" + "\n" +
                          rf"($\sigma$ = {sig_lab(s_hi)})", fontsize=6.5)
    h, l = axes[0, 0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], linestyle="none", marker="o", markerfacecolor="none", color=style.INK2))
    l.append("open: not noise-dominated")
    fig.legend(h, l, loc="upper center", ncol=min(8, len(l)), fontsize=5.8, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout(pad=0.3, w_pad=0.5)
    return _savefig(fig, figdir, "fig_E6_amplification")


def fig_amp_theory(e, figdir):
    """measured noise amplification vs the H2 gain of the linear error recursion."""
    a = e.amp
    if a.empty or "G_pred" not in a:
        return None
    combos = e.combos
    fig, axes = plt.subplots(1, len(combos), figsize=(style.DOUBLE, 2.4), squeeze=False)
    for k, (p, pr) in enumerate(combos):
        ax = axes[0, k]
        q = a[(a.problem == p) & (a.protocol == pr) & a.noise_dominated & np.isfinite(a.G) &
              np.isfinite(a.G_pred) & (a.G_pred > 0)]
        for i in e.infos:
            qi = q[q["info"] == i]
            if qi.empty:
                continue
            st = info_style(i)
            ax.plot(qi.G_pred, qi.G, linestyle="none", marker=st["marker"], color=st["color"],
                    markersize=_msize(st, 2.8), label=INFO_LABEL.get(i, i), alpha=0.85)
        f = getattr(e, "gfit", {}).get(f"{p}|{pr}")
        if f and len(q):
            xs = np.geomspace(q.G_pred.min(), q.G_pred.max(), 20)
            ax.plot(xs, f["kappa_slope1"] * xs, color=style.INK2, linewidth=0.7,
                    label="slope-1 fit")
            ax.text(0.04, 0.96, f"log-log slope {f['slope']:.2f}\n$R^2$ (slope 1) = {f['r2_slope1']:.2f}",
                    transform=ax.transAxes, va="top", fontsize=5.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(LogLocator(base=10, numticks=12))
            axis.set_minor_formatter(NullFormatter())
        ax.set_title(f"{p}\n{PROTO_PLAIN[pr]}", fontsize=6.5, pad=2)
        ax.set_xlabel(r"predicted gain $\|(h+D)/P_h\|_2$")
        if k == 0:
            ax.set_ylabel(r"measured MSSRE / $\sigma$")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=min(8, len(l)), fontsize=5.8, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout(pad=0.3, w_pad=0.6)
    return _savefig(fig, figdir, "fig_E6_amp_theory")


# fig_E6_order_collapse: information regimes grouped by the order of the derivative estimate,
# left to right inside each formula slot (x offset of every member from the slot centre)
ORDER_GROUPS = [("first-order estimates", [("bd1", -0.41), ("alpha_beta", -0.30), ("kalman", -0.19)]),
                ("second-order estimates", [("bd2", -0.01), ("savgol", 0.10)]),
                ("fifth-order estimate", [("bd5", 0.25)]),
                ("exact derivative", [("analytic", 0.38)])]
PROB_TITLE = {"linear_system": "linear system", "matrix_inversion": "matrix inversion"}
PROTO_TITLE = {"common": f"common $h={H_COMMON:g}$", "hopt": r"own $h_{\mathrm{opt}}$"}


def fig_order_collapse(e, figdir):
    """Measured order at sigma = 0 per formula and information regime against the nominal order
    (paper figure: one panel per (problem, protocol), regimes grouped by estimator order)."""
    combos = e.combos
    offs = {i: dx for _, mem in ORDER_GROUPS for i, dx in mem}
    infos = [i for _, mem in ORDER_GROUPS for i, _ in mem if i in e.infos] + \
        [i for i in e.infos if i not in offs]
    with style.print_rc():
        ncol = 2
        nrow = int(math.ceil((len(combos) + 1) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(style.TEXTWIDTH, 1.32 * nrow + 0.2), squeeze=False,
                                 sharey=True, constrained_layout=True)
        fig.get_layout_engine().set(h_pad=0.05, w_pad=0.04, hspace=0.06, wspace=0.04)
        for k, (p, pr) in enumerate(combos):
            ax = axes.flat[k]
            dl = e.discs_in(p, pr)
            x = np.arange(len(dl))
            for d_i, d in enumerate(dl):
                ax.plot([d_i - 0.46, d_i + 0.46], [e.order[d]] * 2, color=style.INK, linewidth=1.3,
                        solid_capstyle="butt", zorder=1)
                if d_i:
                    ax.axvline(d_i - 0.5, color="#e4e3df", lw=0.5, zorder=0)
            for i in infos:
                xs, ys, lo, hi, two = [], [], [], [], []
                for d_i, d in enumerate(dl):
                    f = e.orders.get((p, pr, i, d))
                    if f is None or not np.isfinite(f["slope"]) or f.get("absurd"):
                        continue          # absurd fits (outside [0, q+3]) are listed in SUMMARY.md, not drawn
                    xs.append(d_i + offs.get(i, 0.0))
                    ys.append(f["slope"])
                    two.append(f["flag"] == "2pt")
                    ci = f.get("ci", (np.nan, np.nan))
                    lo.append(max(f["slope"] - ci[0], 0.0) if np.isfinite(ci[0]) else 0.0)
                    hi.append(max(ci[1] - f["slope"], 0.0) if np.isfinite(ci[1]) else 0.0)
                st = info_style(i)
                if xs:
                    xs, ys, lo, hi, two = map(np.asarray, (xs, ys, lo, hi, two))
                    a = ~two
                    ms = 4.4 if st["marker"] in ("*", "P", "X") else 3.7
                    ax.errorbar(xs[a], ys[a], yerr=[lo[a], hi[a]], linestyle="none", marker=st["marker"],
                                color=st["color"], markersize=ms, elinewidth=0.7, capsize=0, zorder=3)
                    if two.any():
                        ax.plot(xs[two], ys[two], linestyle="none", marker=st["marker"], color=st["color"],
                                markerfacecolor="white", markersize=ms, zorder=3)
            ax.set_xticks(x)
            ax.set_xticklabels([short_label(d) for d in dl])
            ax.set_xlim(-0.55, len(dl) - 0.45)
            ax.set_ylim(-0.3, 6.6)
            ax.set_yticks(range(0, 7))
            ax.grid(axis="x", visible=False)
            ax.tick_params(axis="x", length=0)
            ax.set_title(f"{PROB_TITLE.get(p, p)}, {PROTO_TITLE.get(pr, pr)}")
            if k % ncol == 0:
                ax.set_ylabel(r"measured order ($\sigma=0$)")
        # legend panel: one block per estimator group, then the reference marks
        lax = axes.flat[len(combos)]
        lax.axis("off")
        for ax in axes.flat[len(combos) + 1:]:
            ax.set_visible(False)
        ref = [Line2D([], [], color=style.INK, lw=1.3, label="predicted order $q+1$"),  # expected_order = q+1 at fixed h
               Line2D([], [], linestyle="none", marker="o", color=style.INK2, markerfacecolor="white",
                      markersize=3.7, label="two-point estimate")]
        lax.add_artist(Legend(lax, ref, [h.get_label() for h in ref], loc="upper left",
                              bbox_to_anchor=(0.0, 1.07), frameon=False, ncol=2, borderaxespad=0.0,
                              handletextpad=0.4, columnspacing=0.8, handlelength=1.6))  # fits 'predicted order q+1'
        pos = [(0.0, 0.86), (0.52, 0.86), (0.0, 0.29), (0.52, 0.29)]
        for gi, (gname, mem) in enumerate(ORDER_GROUPS):
            hs = []
            for i, _ in mem:
                if i not in e.infos:
                    continue
                st = info_style(i)
                ms = 4.4 if st["marker"] in ("*", "P", "X") else 3.7
                hs.append(Line2D([], [], linestyle="none", marker=st["marker"], color=st["color"], markersize=ms,
                                 label=INFO_LABEL.get(i, i)))
            if not hs:
                continue
            lax.add_artist(Legend(lax, hs, [h.get_label() for h in hs], title=gname, loc="upper left",
                                  bbox_to_anchor=pos[gi], frameon=False, borderaxespad=0.0, handletextpad=0.3,
                                  labelspacing=0.25, alignment="left", title_fontsize=8))
        return _savefig(fig, figdir, "fig_E6_order_collapse")


def fig_traces(e, results, figdir):
    from znnbench.runner import load_traces
    df = e.df
    if "traces_file" not in df.columns or df["traces_file"].isna().all():
        return None
    p = "linear_system" if "linear_system" in set(df.problem) else sorted(set(df.problem))[0]
    pos = [s for s in e.sigmas if s > 0]
    s = min(pos, key=lambda v: abs(np.log10(v) + 6)) if pos else 0.0
    t = e.tau_ref
    infos = [i for i in ("analytic", "bd2", "bd5") if i in e.infos]
    fig, axes = plt.subplots(1, len(infos), figsize=(style.DOUBLE, 2.0), sharey=True, squeeze=False)
    rdir = Path(results) / "results"
    for k, i in enumerate(infos):
        ax = axes[0, k]
        for d in e.discs:
            q = df[(df.problem == p) & (df.protocol == "common") & (df.info_reg == i) &
                   (np.isclose(df.sigma, s)) & (df.disc == d) & np.isclose(df.tau, t) & (df.seed == 0)]
            if q.empty or pd.isna(q.traces_file.iloc[0]):
                continue
            f = rdir / q.traces_file.iloc[0]
            if not f.exists():
                f = Path(q.traces_file.iloc[0])
            if not f.exists():
                continue
            tr = load_traces(f, q.run_id.iloc[0])
            if "residual" not in tr:
                continue
            res = np.median(tr["residual"].astype(float), axis=1)
            st = disc_style(d)
            ax.plot(tr["t"], res, label=long_label(d, e.order[d]), color=st["color"],
                    linestyle=st["linestyle"], linewidth=0.9)
        ax.axhline(s, **SIGMA_LINE)
        ax.set_yscale("log")
        ax.set_xlabel("t (s)")
        ax.set_title(f"{INFO_LABEL.get(i, i)}", pad=2)
        if k == 0:
            ax.set_ylabel(r"$\|E(x_k,t_k)\|_2$ (median of B)")
    h, l = axes[0, 0].get_legend_handles_labels()
    if not h:
        plt.close(fig)
        return None
    h.append(plt.Line2D([], [], **SIGMA_LINE))
    l.append(r"noise level $\sigma$")
    fig.legend(h, l, loc="upper center", ncol=len(l), fontsize=6, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle(f"{p}, common h, tau = {tau_lab(t)}, sigma = {sig_lab(s)}, seed 0", fontsize=6.5,
                 y=-0.02, va="top")
    fig.tight_layout(pad=0.3)
    return _savefig(fig, figdir, "fig_E6_traces")


# =============================================================================
# tables
# =============================================================================
def _tex_write(path, header, rows, colspec, caption=None, label=None):
    lines = [r"\begin{table}[t]", r"\centering", r"\footnotesize"]
    if caption:
        lines.append(rf"\caption{{{caption}}}")
    if label:
        lines.append(rf"\label{{{label}}}")
    lines += [rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    for r in rows:
        lines.append(r"\midrule" if r == "MIDRULE" else " & ".join(r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    Path(path).write_text("\n".join(lines))
    return Path(path).name


def tex_tables(e, tabdir):
    tabdir = Path(tabdir)
    tabdir.mkdir(parents=True, exist_ok=True)
    out = []
    # 1. orders at sigma = 0
    header = ["problem", "$h$", "formula", "$q$"] + [INFO_TEX.get(i, i) for i in e.infos]
    rows = []
    for p, pr in e.combos:
        for d in e.discs_in(p, pr):
            r = [p.replace("_", r"\_"), PROTO_TEX[pr], long_label(d).replace("_", r"\_"), str(e.order[d])]
            for i in e.infos:
                f = e.orders.get((p, pr, i, d))
                if f is None:
                    r.append("--")
                    continue
                a = f"{f['slope']:.2f}" if np.isfinite(f["slope"]) else "--"
                if f["flag"] == "2pt":
                    a += r"$^\dagger$"
                if f.get("absurd"):
                    a += r"$^\ast$"
                b = f"{f['slope_all']:.2f}" if np.isfinite(f["slope_all"]) else "--"
                if f.get("absurd_all"):
                    b += r"$^\ddagger$"
                r.append(f"{a} / {b}")
            rows.append(r)
        rows.append("MIDRULE")
    if rows and rows[-1] == "MIDRULE":
        rows.pop()
    out.append(_tex_write(tabdir / "E6_orders.tex", header, rows, "lll" + "r" + "c" * len(e.infos),
                          caption=(f"E6: empirical residual order at $\\sigma=0$ "
                                   "(asymptotic-range fit / all-points fit above the $10^{-12}$ floor). "
                                   "Asymptotic range: trailing small-$\\tau$ run whose local slopes all lie "
                                   "within 0.15 of the window's least-squares slope; "
                                   r"$^\dagger$ two-point estimate; $^\ast$ / $^\ddagger$ slope outside "
                                   r"$[0, q+3]$ (pre-asymptotic or unstable large-$\tau$ points)."),
                          label="tab:E6-orders"))
    # 2. amplification
    s_hi = max(e.sigmas)
    header = ["problem", "$h$", "formula"] + [INFO_TEX.get(i, i) for i in e.infos]
    rows = []
    for p, pr in e.combos:
        for d in e.discs_in(p, pr):
            r = [p.replace("_", r"\_"), PROTO_TEX[pr], long_label(d, e.order[d]).replace("_", r"\_")]
            for i in e.infos:
                q = e.amp[(e.amp.problem == p) & (e.amp.protocol == pr) & (e.amp["info"] == i) &
                          (e.amp.disc == d) & (e.amp.sigma == s_hi)]
                if q.empty:
                    r.append("--")
                else:
                    G = float(q.G.iloc[0])
                    txt = fmt_num(G, 2)
                    r.append(txt if bool(q.noise_dominated.iloc[0]) else f"({txt})")
            rows.append(r)
        rows.append("MIDRULE")
    if rows and rows[-1] == "MIDRULE":
        rows.pop()
    out.append(_tex_write(tabdir / "E6_amplification.tex", header, rows, "lll" + "r" * len(e.infos),
                          caption=(f"E6: noise amplification MSSRE$/\\sigma$ at $\\tau={sig_math(min(e.taus))}$, "
                                   f"$\\sigma={sig_math(s_hi)}$ (parentheses: not noise-dominated)."),
                          label="tab:E6-amp"))
    # 3. regime summary
    header = ["problem", "protocol", "info", r"high / low / n.s. (\%)", r"best at $\sigma{=}0$",
              r"best at $\sigma_{\max}$", r"$\tau^\ast(\sigma_{\max})$"]
    rows = []
    for p, pr in e.combos:
        for i in e.infos:
            cells = [v for (pp, ppr, ii, s, t), v in e.regime.items() if (pp, ppr, ii) == (p, pr, i)]
            val = [v for v in cells if v.get("best")]
            adv = [v for (pp, ppr, ii, _, _), v in e.adv.items() if (pp, ppr, ii) == (p, pr, i)
                   and not (np.isnan(v[1]) or np.isnan(v[2]))]
            frac = (f"{100 * np.mean([v[1] > 0 for v in adv]):.0f} / {100 * np.mean([v[2] < 0 for v in adv]):.0f} / "
                    f"{100 * np.mean([v[1] <= 0 <= v[2] for v in adv]):.0f}") if adv else "--"
            tmin = min(e.taus)
            b0 = e.regime.get((p, pr, i, 0.0, tmin), {}).get("best")
            bm = e.regime.get((p, pr, i, max(e.sigmas), tmin), {}).get("best")
            cr = e.cross[(e.cross.problem == p) & (e.cross.protocol == pr) & (e.cross["info"] == i) &
                         (e.cross.sigma == max(e.sigmas))]
            ts = cr.tau_star.iloc[0] if len(cr) else np.nan
            kind = cr.kind.iloc[0] if len(cr) else "none"
            rows.append([p.replace("_", r"\_"), PROTO_PLAIN[pr].replace("h_opt", r"$h_{\rm opt}$"),
                         INFO_TEX.get(i, i), frac,
                         short_label(b0) if b0 else "--", short_label(bm) if bm else "--",
                         sig_tex(ts) if np.isfinite(ts) else {"all": "all", "never": "never",
                                                              "gained": "gained"}.get(kind, "--")])
        rows.append("MIDRULE")
    if rows and rows[-1] == "MIDRULE":
        rows.pop()
    out.append(_tex_write(tabdir / "E6_regime.tex", header, rows, "lllrccc",
                          caption=(f"E6: share of $(\\sigma,\\tau)$ cells where the best order-$\\ge4$ formula is "
                                   "significantly better / worse than the best of Euler and Taylor-Zhang / not "
                                   "distinguishable (selection-aware paired seed-cluster bootstrap of "
                                   "$A=\\log_{10}$ of the ratio of the minima); "
                                   f"winner at the smallest $\\tau={sig_math(min(e.taus))}$; largest $\\tau$ below "
                                   "which the high-order advantage is lost at $\\sigma_{\\max}$ (all: lost at every "
                                   "$\\tau$; never: kept at every $\\tau$; gained: appears only at small $\\tau$)."),
                          label="tab:E6-regime"))
    # 4. headline ratios
    header = ["problem", "protocol", "info", r"$\sigma$", "best high", r"MSSRE$_{\rm Euler}$/MSSRE$_{\rm high}$",
              "95\\% CI"]
    rows = []
    for _, r in e.head.iterrows():
        rows.append([r.problem.replace("_", r"\_"), PROTO_PLAIN[r.protocol].replace("h_opt", r"$h_{\rm opt}$"),
                     INFO_TEX.get(r["info"], r["info"]), sig_tex(r.sigma), short_label(r.best_high),
                     fmt_num(r.ratio_euler_over_high, 3), ci_text(r.ci_lo, r.ci_hi, r.n_seeds)])
    out.append(_tex_write(tabdir / "E6_headline.tex", header, rows, "lllllrr",
                          caption=(f"E6: Euler vs the best high-order formula at $\\tau={sig_math(e.tau_ref)}$ "
                                   "(ratio of medians, paired seed-cluster bootstrap 95\\% CI, 2000 resamples; "
                                   f"n/a: fewer than {MIN_CLUSTERS} seeds)."),
                          label="tab:E6-headline"))
    # 4b. noise-gain theory vs measurement
    gf = getattr(e, "gfit", {})
    if gf:
        header = ["problem", "$h$", "$n$", "log-log slope", r"$R^2$ (slope 1)", r"$\kappa$"]
        rows = []
        for k, v in gf.items():
            p, pr = k.split("|")
            rows.append([p.replace("_", r"\_"), PROTO_TEX[pr], str(v["n"]), f"{v['slope']:.2f}",
                         f"{v['r2_slope1']:.3f}", f"{v['kappa_slope1']:.2f}"])
        out.append(_tex_write(tabdir / "E6_gain_theory.tex", header, rows, "llrrrr",
                              caption=("E6: measured noise amplification MSSRE$/\\sigma$ (noise-dominated cells, "
                                       "smallest $\\tau$, all $\\sigma>0$, all formulas and estimators) against the "
                                       "$H_2$ gain $\\|(h+D(z))/P_h(z)\\|_2$ of the linear error recursion; "
                                       "$\\kappa$ = problem factor of the slope-1 fit."),
                              label="tab:E6-gain"))
    # 5. Kendall tau_b vs (analytic, sigma = 0)
    header = ["problem", "protocol", "info"] + [rf"$\sigma$={sig_tex(s)}" for s in e.sigmas]
    rows = []
    for p, pr in e.combos:
        for i in e.infos:
            r = [p.replace("_", r"\_"), PROTO_PLAIN[pr].replace("h_opt", r"$h_{\rm opt}$"), INFO_TEX.get(i, i)]
            for s in e.sigmas:
                q = e.kt[(e.kt.problem == p) & (e.kt.protocol == pr) & (e.kt["info"] == i) & (e.kt.sigma == s)]
                v = q.kendall_tau_b.iloc[0] if len(q) else np.nan
                r.append(f"{v:+.2f}" if np.isfinite(v) else "--")
            rows.append(r)
        rows.append("MIDRULE")
    if rows and rows[-1] == "MIDRULE":
        rows.pop()
    out.append(_tex_write(tabdir / "E6_kendall.tex", header, rows, "lll" + "r" * len(e.sigmas),
                          caption=(f"E6: Kendall $\\tau_b$ between the formula ranking (median MSSRE, "
                                   f"$\\tau={sig_math(e.tau_ref)}$) of each regime and that of analytic "
                                   "$\\partial_t E$ with $\\sigma=0$ (descriptive: 5--6 formulas, no "
                                   "p-values; diverged formulas tie last)."), label="tab:E6-kendall"))
    return out


# =============================================================================
# cost estimate for the full sweep
# =============================================================================
def full_cost_estimate(df):
    """Steps of the full config x measured wall_per_step (per problem and
    information regime), rescaled from the measured batch to the full B with
    the runner's (1 + B/16) cost model.  The runs are enumerated with
    ``configs/E6.py: planned_runs('full')`` (checked against the runner's
    ``expand`` by ``E6.py --dry-run``)."""
    import importlib.util
    try:
        spec = importlib.util.spec_from_file_location("_e6cfg", HERE / "configs" / "E6.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        plan = mod.planned_runs("full")
        chunk = int(mod.SCALES["full"]["chunk_size"])
    except Exception as exc:  # noqa: BLE001
        return {"error": repr(exc)}
    ok = df[df.status == "ok"].drop_duplicates("run_id")
    B_meas = int(ok.B.iloc[0]) if len(ok) else 2
    wps = ok.groupby(["problem", "info_reg"])["wall_per_step"].median().to_dict()
    wps_all = float(ok["wall_per_step"].median()) if len(ok) else 1.5e-4
    # per-spec setup (problem/model/disc construction, metrics, traces) measured in the runs
    setup = float(ok["setup_wall"].median()) if "setup_wall" in ok and ok["setup_wall"].notna().any() else 0.0
    peak = float(df.peak_rss_mb.max()) if "peak_rss_mb" in df else np.nan
    total_steps, total_s, longest = 0, 0.0, 0.0
    for r in plan:
        f = (1 + r["B"] / 16.0) / (1 + B_meas / 16.0)
        sec = r["steps"] * wps.get((r["problem"], r["info"]), wps_all) * f + setup
        total_steps += r["steps"]
        total_s += sec
        longest = max(longest, sec)
    n_tasks = math.ceil(len(plan) / chunk)
    total_s_ovh = total_s + n_tasks * TASK_OVERHEAD_S
    return {"n_specs": len(plan), "n_trajectories": int(sum(r["B"] for r in plan)),
            "total_steps": int(total_steps), "core_hours_compute": total_s / 3600.0,
            "core_hours": total_s_ovh / 3600.0,
            "task_overhead_s_assumed": TASK_OVERHEAD_S,
            "wall_h_16_workers": total_s_ovh / 3600.0 / 16.0,
            "wall_h_24_workers": total_s_ovh / 3600.0 / 24.0, "longest_spec_s": longest,
            "chunk_size": chunk, "n_tasks": n_tasks, "mean_task_s": total_s_ovh / n_tasks,
            "measured_wall_per_step_median_s": wps_all, "measured_B": B_meas,
            "measured_setup_wall_median_s": setup,
            "wall_per_step_by_problem_info": {f"{k[0]}|{k[1]}": v for k, v in wps.items()},
            "peak_rss_mb_measured": peak}


# =============================================================================
# summary
# =============================================================================
def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def censoring_summary(e):
    """How diverged / failed trajectories and cells enter the analysis (checklist item 2)."""
    c = e.cells
    out = {"n_traj": int(c.n.sum()), "n_diverged": int(c.n_div.sum()), "n_failed": int(c.n_fail.sum()),
           "cells": int(len(c)), "cells_with_divergence": int((c.n_div > 0).sum()),
           "cells_median_diverged": int(np.isposinf(c["median"]).sum()),
           "cells_with_failures": int((c.n_fail > 0).sum())}
    d = c[c.n_div > 0].groupby(["problem", "protocol", "disc", "tau"]).agg(
        n_div=("n_div", "sum"), n=("n", "sum")).reset_index()
    out["diverged_by_formula_tau"] = [
        f"{r.problem}|{r.protocol}|{r.disc}|tau={tau_lab(r.tau)}: {int(r.n_div)}/{int(r.n)}" for _, r in d.iterrows()]
    a = getattr(e, "amp", pd.DataFrame())
    out["amp_cells_excluded_diverged"] = int((~np.isfinite(a.G)).sum()) if len(a) else 0
    return out


def spread_summary(e):
    """within-seed spread of the B initial states (pseudo-replication / transient leakage)."""
    c = e.cells[np.isfinite(e.cells.x0_spread) | np.isposinf(e.cells.x0_spread)]
    if c.empty:
        return {}
    by_tau = {tau_lab(t): {"median": float(np.median(g.x0_spread)), "q90": float(np.quantile(g.x0_spread, 0.9)),
                           "max": float(g.x0_spread.max())} for t, g in c.groupby("tau")}
    # leakage list: only cells well above round-off (a 1e-11 residual has ~1e-3 relative round-off spread)
    big = c[(c.x0_spread > 1e-3) & (c["median"] > 1e-9)]
    return {"median": float(np.median(c.x0_spread)), "q90": float(np.quantile(c.x0_spread, 0.9)),
            "n_cells": int(len(c)), "n_cells_gt_1e-3": int(len(big)), "by_tau": by_tau,
            "cells_gt_1e-3": sorted({f"{r.problem}|{r.protocol}|{r.disc}|tau={tau_lab(r.tau)}"
                                     for _, r in big.iterrows()})[:40]}


def summarize(e, df, figs, tabs, cost, results):
    s = {"experiment": "E6", "results": str(results), "n_rows": int(len(df)),
         "status_counts": df.status.value_counts().to_dict(),
         "formulas": {d: {"expected_order": e.order[d], "h_opt_used": None} for d in e.discs},
         "infos": e.infos, "sigmas": e.sigmas, "taus": e.taus, "tau_ref": e.tau_ref,
         "combos": [list(c) for c in e.combos], "figures": figs, "tables": tabs}
    hq = df[df.protocol == "hopt"].groupby("disc")["h"].first().to_dict()
    for d in e.discs:
        s["formulas"][d]["h_opt_used"] = hq.get(d)
    # orders
    s["orders_sigma0"] = {f"{p}|{pr}|{i}|{d}": {"slope_asym": f["slope"], "slope_all": f["slope_all"],
                                                "ci": f.get("ci"), "flag": f["flag"],
                                                "tau_range": f["tau_range"], "local_slopes": f["local"],
                                                "nominal": e.order[d]}
                          for (p, pr, i, d), f in e.orders.items()}
    # headline
    s["headline_euler_vs_best_high"] = e.head.to_dict(orient="records")
    # regime fractions
    frac = {}
    for p, pr in e.combos:
        for i in e.infos:
            val = [v for (pp, ppr, ii, _, _), v in e.regime.items() if (pp, ppr, ii) == (p, pr, i) and v.get("best")]
            ass = [v for v in val if v["sig"] is not None]
            adv = [v for (pp, ppr, ii, _, _), v in e.adv.items() if (pp, ppr, ii) == (p, pr, i)
                   and not np.isnan(v[0])]
            adv_ok = [v for v in adv if np.isfinite(v[1]) and np.isfinite(v[2]) or np.isinf(v[1]) or np.isinf(v[2])]
            frac[f"{p}|{pr}|{i}"] = {
                "frac_high_order_best": float(np.mean([v["best"] in e.high for v in val])) if val else None,
                "frac_significant": float(np.mean([bool(v["sig"]) for v in ass])) if ass else None,
                # selection-aware advantage test (min over formulas re-taken per resample)
                "adv_n": len(adv_ok),
                "adv_high_sig": float(np.mean([v[1] > 0 for v in adv_ok])) if adv_ok else None,
                "adv_low_sig": float(np.mean([v[2] < 0 for v in adv_ok])) if adv_ok else None,
                "adv_ns": float(np.mean([v[1] <= 0 <= v[2] for v in adv_ok])) if adv_ok else None,
                "n_cells": len(val), "n_assessable": len(ass),
                "n_all_diverged": sum(1 for (pp, ppr, ii, _, _), v in e.regime.items()
                                      if (pp, ppr, ii) == (p, pr, i) and v.get("all_div"))}
    s["regime_fraction"] = frac
    s["censoring"] = censoring_summary(e)
    s["pseudo_replication"] = spread_summary(e)
    s["tau_asym"] = {f"{p}|{pr}": e.tau_asym(p, pr) for p, pr in e.combos}
    s["regime_map"] = {f"{p}|{pr}|{i}|{sig_lab(sg)}|{tau_lab(t)}": {"best": v.get("best"), "second": v.get("second"),
                                                                    "significant": v.get("sig"),
                                                                    "ratio_second_over_best": v.get("ratio")}
                       for (p, pr, i, sg, t), v in e.regime.items()}
    s["crossover_tau_star"] = e.cross.to_dict(orient="records")
    fs = e.fslopes
    if not fs.empty:
        s["noise_floor_tau_slope"] = {}
        for kind, g0 in fs.groupby("kind"):
            s["noise_floor_tau_slope"][kind] = {
                "median": float(g0.slope.median()), "q25": float(g0.slope.quantile(0.25)),
                "q75": float(g0.slope.quantile(0.75)), "n": int(len(g0)),
                "n_points_median": float(g0.n.median()),
                "by_info": {i: float(g.slope.median()) for i, g in g0.groupby("info")},
                "by_disc": {d: float(g.slope.median()) for d, g in g0.groupby("disc")}}
    amp = e.amp[(e.amp.sigma == max(e.sigmas)) & e.amp.noise_dominated]
    s["amplification_sigma_max"] = {f"{r.problem}|{r.protocol}|{r['info']}|{r.disc}": r.G for _, r in amp.iterrows()}
    s["kendall_tau_b"] = e.kt.to_dict(orient="records")
    s["noise_gain_theory_fit"] = getattr(e, "gfit", {})
    s["noise_gain_predicted"] = {f"{r.problem}|{r.protocol}|{r['info']}|{r.disc}": r.G_pred
                                 for _, r in e.amp.drop_duplicates(["problem", "protocol", "info", "disc"]).iterrows()}
    s["full_cost_estimate"] = cost
    return s


def write_summary_md(s, e, path):
    c_ = e.cells
    L = ["# E6 summary (information realism)", "",
         f"Results: `{s['results']}`; {s['n_rows']} trajectory rows; status counts {s['status_counts']}.",
         "Formulas: " + ", ".join(f"{d} (q={e.order[d]})" for d in e.discs) + ".",
         f"Information regimes: {', '.join(e.infos)}; sigma in {[sig_lab(x) for x in e.sigmas]}; "
         f"tau in {[tau_lab(x) for x in e.taus]}.",
         "Seeds (= bootstrap clusters) per problem/protocol: " + ", ".join(
             f"{p}|{pr} {int(c_[(c_.problem == p) & (c_.protocol == pr)].n_seeds.max())}" for p, pr in e.combos)
         + f"; CIs need >= {MIN_CLUSTERS} seeds (else n/a).", ""]
    cz = s.get("censoring") or {}
    if cz:
        dv = cz["diverged_by_formula_tau"]
        L += [f"**Censoring.** {cz['n_diverged']} of {cz['n_traj']} trajectories diverged, {cz['n_failed']} failed; "
              f"{cz['cells_with_divergence']} of {cz['cells']} cells contain a divergence, {cz['cells_median_diverged']} "
              "have a diverged median. Diverged trajectories enter medians, IQRs, bootstrap CIs and rankings as +inf "
              "(worst case; two diverged formulas tie); failed runs would be counted (n_fail in E6_cells.csv) and "
              "excluded. Order fits use only the trailing finite run of sigma = 0 medians; floor-slope and gain fits use "
              f"finite noise-dominated cells ({cz['amp_cells_excluded_diverged']} amplification cells excluded as "
              "diverged). Diverged (problem|protocol|formula|tau: n_div/n): "
              + ("; ".join(dv[:14]) if dv else "none") + (" ..." if len(dv) > 14 else "") + ".", ""]
    L += ["## Headline numbers", ""]
    hd = e.head
    for p, pr in e.combos:
        L.append(f"**{p}, {PROTO_PLAIN[pr]}** (tau = {tau_lab(e.tau_ref)}, ratio MSSRE(Euler)/MSSRE(best high-order), "
                 "paired seed-cluster bootstrap 95 % CI):")
        L.append("")
        L.append("| info | sigma | best high | ratio | 95 % CI |")
        L.append("|---|---|---|---|---|")
        for _, r in hd[(hd.problem == p) & (hd.protocol == pr)].iterrows():
            L.append(f"| {INFO_PLAIN.get(r['info'], r['info'])} | {sig_lab(r.sigma)} | {r.best_high} | "
                     f"{fmt_num(r.ratio_euler_over_high, 3)} | {ci_text(r.ci_lo, r.ci_hi, r.n_seeds)} |")
        L.append("")
    p0 = "linear_system" if any(c[0] == "linear_system" for c in e.combos) else e.combos[0][0]
    L += [f"## Empirical orders at sigma = 0 ({p0}, common h)", "",
          "Asymptotic fit [points used] / all-points fit. Asymptotic range: the trailing small-tau run of finite "
          "medians above the 1e-12 floor, grown towards larger tau while every local slope stays within "
          f"{TOL_SLOPE} of the window's LS slope. (2pt) = two-point estimate; (pre-asym.) = all-points fit outside "
          "[0, q+3] because unstable / pre-asymptotic large-tau points enter it.", "",
          "| formula (q) | " + " | ".join(INFO_PLAIN.get(i, i) for i in e.infos) + " |",
          "|---|" + "---|" * len(e.infos)]
    for d in e.discs_in(p0, "common"):
        row = [f"{d} ({e.order[d]})"]
        for i in e.infos:
            f = e.orders.get((p0, "common", i, d))
            if f is None or not np.isfinite(f["slope"]):
                row.append("--")
                continue
            a = f"{fmt_num(f['slope'], 3)} [{f['n_used']}]" + (" (2pt)" if f["flag"] == "2pt" else "") \
                + (" (ABSURD)" if f.get("absurd") else "")
            b = fmt_num(f["slope_all"], 3) + (" (pre-asym.)" if f.get("absurd_all") else "")
            row.append(f"{a} / {b}")
        L.append("| " + " | ".join(row) + " |")
    n_abs = sum(1 for f in e.orders.values() if f.get("absurd"))
    n2 = sum(1 for f in e.orders.values() if f["flag"] == "2pt")
    L += ["", f"All combinations: {len(e.orders)} fits, {n2} two-point, {n_abs} asymptotic fits outside [0, q+3]. "
              "Two-point fits occur where only two tau values lie between the unstable/pre-asymptotic large-tau "
              "range and the round-off floor (q = 6 with analytic or BD5 information); E1 verifies those orders on "
              "a dedicated grid.", "",
          "## Regime map: where is the high-order advantage significant?", "",
          "A = log10(min over {Euler, TZ} / min over order >= 4 of the median MSSRE), paired seed-cluster "
          "bootstrap 95 % CI with the minima re-taken in every resample (selection-aware). Shares of the "
          "(sigma, tau) cells: high = CI > 0, low = CI < 0, n.s. = CI contains 0; then the raw share of cells "
          "whose lowest-median formula has order >= 4 (fragile under near-ties: it is NOT a significance "
          "statement) and how often that winner beats the runner-up significantly.", ""]
    for k, v in s["regime_fraction"].items():
        if v["frac_high_order_best"] is None:
            continue
        sig = (f"{100 * v['frac_significant']:.0f} % of {v['n_assessable']} winners significant vs runner-up"
               if v["frac_significant"] is not None else "winner significance n/a (too few seeds)")
        adv = (f"high {100 * v['adv_high_sig']:.0f} % / low {100 * v['adv_low_sig']:.0f} % / n.s. "
               f"{100 * v['adv_ns']:.0f} % of {v['adv_n']} cells" if v.get("adv_high_sig") is not None
               else "advantage CI n/a")
        L.append(f"- {k}: {adv}; raw order>=4 winner {100 * v['frac_high_order_best']:.0f} % of {v['n_cells']} "
                 f"({sig}" + (f"; {v['n_all_diverged']} cells all diverged" if v.get("n_all_diverged") else "") + ")")
    L += ["", "One comparison per cell, no multiplicity control: descriptive. Kendall tau_b (E6_kendall.tex) is "
              "descriptive (5-6 formulas, no p-values).", ""]
    nf = s.get("noise_floor_tau_slope") or {}
    ta = s.get("tau_asym", {})
    if nf:
        L.append("## Noise floor vs tau (fixed h)")
        L.append("")
        for kind, lab in (("asym", "asymptotic range tau <= tau_asym"), ("all", "all tau")):
            v = nf.get(kind)
            if not v:
                continue
            L.append(f"- {lab}: tau-slope of MSSRE in cells with MSSRE(sigma) >= {NOISE_DOM_SLOPE:g} x MSSRE(0): "
                     f"median {v['median']:+.2f} (IQR {v['q25']:+.2f} .. {v['q75']:+.2f}, n = {v['n']} fits, "
                     f"median {v['n_points_median']:.0f} tau per fit); by info: "
                     + ", ".join(f"{k2} {x:+.2f}" for k2, x in v["by_info"].items()) + "; by formula: "
                     + ", ".join(f"{short_label(k2)} {x:+.2f}" for k2, x in v["by_disc"].items()) + ".")
        L.append("- tau_asym (largest tau inside every formula's analytic sigma = 0 asymptotic window): "
                 + ", ".join(f"{k2} {tau_lab(x)}" for k2, x in ta.items()) + ". Above it the q = 6 formulas are "
                 "unstable or pre-asymptotic (time variation over their 10-11-instant stencils), so noise is "
                 "amplified more there and the 'all tau' slope is biased upwards.")
        L.append("")
    gf = s.get("noise_gain_theory_fit") or {}
    if gf:
        L.append("Noise amplification vs the H2 gain ||(h + D)/P_h||_2 of the linear error recursion "
                 "(noise-dominated cells, smallest tau): " + "; ".join(
                     f"{k}: log-log slope {v['slope']:.2f}, R^2(slope 1) {v['r2_slope1']:.3f}, kappa "
                     f"{v['kappa_slope1']:.2f} (n = {v['n']})" for k, v in gf.items()) + ".")
        L.append("")
    ps = s.get("pseudo_replication") or {}
    if ps:
        L += ["## Pseudo-replication and transient leakage (within-seed spread of the B initial states)", "",
              f"max/min - 1 of MSSRE over the B states of a seed: median {ps['median']:.1e}, 90 % quantile "
              f"{ps['q90']:.1e} over {ps['n_cells']} cells; {ps['n_cells_gt_1e-3']} cells with median > 1e-9 above 1e-3 "
              "(x0-dependent steady state = transient leaking into the window, or x0-dependent divergence). "
              "By tau: " + ", ".join(f"{k2}: median {v['median']:.0e}, max {v['max']:.0e}"
                                     for k2, v in ps["by_tau"].items())
              + ". The B states are replicas (identical data-noise stream), so all CIs resample seeds."
              + (" Cells > 1e-3: " + "; ".join(ps["cells_gt_1e-3"][:10]) if ps["cells_gt_1e-3"] else ""), ""]
    L += ["## Confounds", "",
          "- Step-size definition: h = gamma*tau with tau-form coefficients (rho'(1) = 1), so the principal root is "
          "1 - h + O(h^2) for every formula; papers that write h_paper = gamma*tau/c_0 (e.g. ZeaD c_0 != 1) use a "
          "different h. The h_opt values are ZNN-Bench h, not the papers' 'optimal step'.",
          "- Common h = 0.1 equalises gain and principal root but not the parasitic roots (spectral radius 0.90 "
          "Euler ... 0.94 ZeaD-11i at h = 0.1); the larger noise gain of high-order formulas is this mechanism, "
          "quantified by the H2 gain.",
          "- Own h_opt (literature convention) changes the gain and the noise bandwidth together (Euler h = 1 is "
          "dead-beat, the q = 6 formulas run at h = 0.07-0.09): order and filtering are confounded; read the "
          "common-h maps for the formula effect.",
          "- Information advantage: 'analytic' is an oracle (exact dE/dt); BD5 is the order-matched causal "
          "estimator. Estimators see the same noisy samples as E and J (no extra information).",
          "- Horizon: T = 12 s for every tau, steady window [9, 12] s; at tau = 0.1 only 90 steps precede the "
          "window (see the within-seed spread above for leakage). Large-tau cells of the q = 6 formulas are "
          "unstable (time variation over the stencil), not noise effects.",
          "- Units: linear activation only, so no power-type unit dependence; sigma is the absolute per-entry std "
          "of the noise on A and b (O(1) data, kappa = 10).",
          "- Estimator noise is hash-seeded per (seed, t): runs with different tau share the noise at common "
          "instants; simulate's data noise is a per-step stream. Neither affects within-cell comparisons "
          "(common random numbers across formulas).", ""]
    # interpretation (data driven)
    L += ["## Interpretation", ""]
    try:
        def ratio(p, pr, i, sg):
            q = hd[(hd.problem == p) & (hd.protocol == pr) & (hd["info"] == i) & np.isclose(hd.sigma, sg)]
            return (float(q.ratio_euler_over_high.iloc[0]), q.best_high.iloc[0]) if len(q) else (np.nan, None)
        pr = "common"
        r_an, b_an = ratio(p0, pr, "analytic", 0.0)
        r_b1, _ = ratio(p0, pr, "bd1", 0.0)
        r_b2, _ = ratio(p0, pr, "bd2", 0.0)
        r_an_n, _ = ratio(p0, pr, "analytic", max(e.sigmas))
        qmax = max(e.order[d] for d in e.discs_in(p0, pr))
        top = [d for d in e.discs_in(p0, pr) if e.order[d] == qmax]
        oi = {}
        for i in e.infos:
            fits = [e.orders[(p0, pr, i, d)] for d in top if (p0, pr, i, d) in e.orders]
            fits = [f for f in fits if np.isfinite(f["slope"]) and not f.get("absurd")]
            if fits:
                tag = " (2-pt)" if any(f["flag"] == "2pt" for f in fits) else ""
                oi[i] = f"{np.median([f['slope'] for f in fits]):.2f}{tag}"
        L.append(f"1. *Information order caps the formula order.* With exact dE/dt and sigma = 0 the best "
                 f"high-order formula ({b_an}) is {fmt_num(r_an, 3)}x more accurate than Euler at tau = "
                 f"{tau_lab(e.tau_ref)} ({p0}, common h); with BD1 the factor is {fmt_num(r_b1, 3)}x and with BD2 "
                 f"{fmt_num(r_b2, 3)}x. Empirical order at sigma = 0 of the q = {qmax} formulas "
                 f"({', '.join(top)}; median): "
                 + ", ".join(f"{INFO_PLAIN.get(i, i)} {v}" for i, v in oi.items())
                 + ". Unless the estimator is matched (BD5), the residual order is set by the estimator's "
                 "accuracy order + 1 (BD1, alpha-beta, Kalman: 2; BD2, S-G(5,2): 3), not by the formula.")
        v = nf.get("asym") or nf.get("all")
        if v:
            L.append(f"2. *Noise floor at fixed h is ~ sigma x G(formula, estimator), independent of tau.* The "
                     f"tau-slope of MSSRE in clearly noise-dominated cells is {v['median']:+.2f} (median, "
                     f"{'asymptotic range' if 'asym' in nf else 'all tau'}), not -1: the estimator's sigma/tau "
                     "derivative noise is multiplied by the step tau, so at fixed h = gamma*tau the floor does not "
                     "grow as tau shrinks; what grows with order is the gain G (larger for high-order formulas, "
                     "whose parasitic roots sit nearer the unit circle, and for higher-order differentiators). The "
                     "advantage is lost where C_q tau^q drops below sigma*G."
                     + ("" if not gf else
                        " G is predicted by the H2 gain ||(h + D)/P_h||_2 of the linear error recursion "
                        "(P_h: look-ahead error polynomial, D: estimator FIR): "
                        + "; ".join(f"{k} slope {x['slope']:.2f}, R^2 {x['r2_slope1']:.3f}" for k, x in gf.items())
                        + "."))
        L.append(f"3. *Regime map.* At sigma = {sig_lab(max(e.sigmas))} Euler/high-order ratio with analytic "
                 f"information is {fmt_num(r_an_n, 3)}x at tau = {tau_lab(e.tau_ref)}; the regime tables list "
                 "which formula wins each (sigma, tau) cell. Low-order formulas win wherever the measurement noise "
                 "dominates the truncation error; high-order formulas win only with small sigma, intermediate tau "
                 "(they are unstable at the largest tau) and derivative information of matching order.")
    except Exception as exc:  # noqa: BLE001
        L.append(f"(interpretation skipped: {exc!r})")
    c = s.get("full_cost_estimate", {})
    if c and "core_hours" in c:
        L += ["", "## Full-scale cost estimate (from measured wall_per_step)", "",
              f"{c['n_specs']} specs ({c['n_trajectories']} trajectories), {c['total_steps']:.3e} steps, "
              f"{c['n_tasks']} tasks of {c['chunk_size']} specs: ~{c['core_hours_compute']:.1f} core-hours of "
              f"simulation + per-spec setup, ~{c['core_hours']:.1f} core-hours with {c['task_overhead_s_assumed']:.0f} s "
              f"per-task overhead (~{c['wall_h_24_workers']:.1f} h on 24 workers, ~{c['wall_h_16_workers']:.1f} h on "
              f"16, if that many performance cores are free); longest spec ~{c['longest_spec_s']:.0f} s; measured "
              f"peak RSS {c['peak_rss_mb_measured']:.0f} MB per task."]
    L += ["", "Figures: " + ", ".join(s["figures"]), "", "Tables: " + ", ".join(s["tables"]), ""]
    Path(path).write_text("\n".join(L))


# =============================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="sweep folder (with tasks.jsonl), e.g. results/E6")
    ap.add_argument("--figdir", default=str(style.FIG_DIR))
    ap.add_argument("--tabdir", default=str(style.TAB_DIR))
    ap.add_argument("--no-cost", action="store_true", help="skip the full-scale cost estimate")
    args = ap.parse_args(argv)
    results = Path(args.results)
    figdir, tabdir = Path(args.figdir), Path(args.tabdir)
    figdir.mkdir(parents=True, exist_ok=True)
    tabdir.mkdir(parents=True, exist_ok=True)

    df = load(results)
    boot = Boot()
    cells, arrays = cell_table(df, boot)
    cells.to_csv(results / "E6_cells.csv", index=False)
    e = E6(df, cells, arrays, boot)
    e.regime_maps()
    e.advantage_cis()
    e.orders_sigma0()
    e.floor_slopes()
    e.amplification()
    e.gain_fit()
    e.headline()
    e.kendall()
    e.crossovers()

    figs = []
    for p, pr in e.combos:
        figs.append(fig_regime(e, p, pr, figdir))
        figs.append(fig_advantage(e, p, pr, figdir))
        figs.append(fig_mssre_tau(e, p, pr, SIM_SET, "A", figdir))
        figs.append(fig_mssre_tau(e, p, pr, EST_SET, "B", figdir))
    figs.append(fig_amplification(e, figdir))
    figs.append(fig_amp_theory(e, figdir))
    figs.append(fig_order_collapse(e, figdir))
    figs.append(fig_traces(e, results, figdir))
    figs = [f for f in figs if f]
    tabs = tex_tables(e, tabdir)
    cost = {} if args.no_cost else full_cost_estimate(df)
    s = summarize(e, df, figs, tabs, cost, results)
    (results / "summary.json").write_text(json.dumps(jsonable(s), indent=1))
    write_summary_md(s, e, results / "SUMMARY.md")
    print(f"E6: {len(df)} rows, {len(cells)} cells; figures -> {figdir}; tables -> {tabdir}")
    print(f"summary -> {results / 'summary.json'}, {results / 'SUMMARY.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
