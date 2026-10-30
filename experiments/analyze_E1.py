#!/usr/bin/env python
"""E1 analysis: which residual order does each discretization deliver?

    python experiments/analyze_E1.py --results results/E1 \
        --figdir figures --tabdir results/tables

Reads every chunk parquet of the E1 sweep (experiments/configs/E1.py, one
sub-folder per formula) and the formula catalogue stored in each sub-folder's
config.json, then

1. builds one MSSRE-vs-tau curve per (formula, problem instance, step-size
   mode) from the per-trajectory steady-state residual ``mssre_w`` (max of
   ||E|| over the absolute window [t_end - 1.5 s, t_end]); median and IQR over
   the 3 seeds x 4 initial states.  Diverged trajectories count as +inf
   (censored median: a point is finite while fewer than half of its
   trajectories diverged; x0-dependent divergence occurs on the nonlinear
   instance); error / timeout / skipped rows are missing, not divergence;
2. marks points as unusable when (a) the median is infinite, (b) the run does
   not converge (median MSSRE >= 0.5 x transient peak), (c) the transient has
   not decayed in the window (median ``x0_spread_w`` > max(1 % of the MSSRE,
   2 x the curve's round-off floor): trajectories that differ only in x0 must
   coincide in steady state), or (d) the frozen-coefficient decay requirement
   exceeds the horizon cap;
3. detects the round-off floor of each curve (finest points where the curve
   stops decreasing, below 1e-10, tolerant to factor-3 floor noise) and
   excludes points within FLOOR_FACTOR of it;
4. selects the asymptotic range by a consecutive-ratio diagnostic: starting at
   the finest point of the finest contiguous run of >= 2 valid points, it grows
   the window towards coarse tau while every local slope
   s_i = log(M_{i+1}/M_i)/log(tau_{i+1}/tau_i) stays within
   max(0.15, 0.03|median|) of the window's median local slope; windows whose
   local slopes still change monotonically by > 0.15 are flagged "drift",
   slopes > expected + 3 or < -0.5 "absurd";
5. fits the order by least squares over that window with a two-stage
   (cluster) bootstrap 95 % CI: 2000 resamples of the problem instances
   (seeds), then of the initial states within each instance, seeded per curve;
   for comparison it also fits all valid points ("naive" fit);
6. compares with the claimed order: fixed-h -> claimed (q+1), fixed-gamma ->
   claimed - 1 (q) (verdicts match / pre-asymptotic / lower / higher; the
   Broyden-aided DTZNN is reported separately, "no tracking" when its MSSRE
   never gets below 1e-3); RK4: classical 4 at fixed gamma, Prothero-Robinson order
   reduction predicts 2 at fixed h; RK4's fixed-gamma minus fixed-h difference
   gets a paired bootstrap (same resampled trajectories in both modes);
7. Kendall tau_b (claimed vs measured order; claimed order vs accuracy at
   fixed h_rec and at the common h = 0.1, diverged / unstable = worst rank)
   with Holm-adjusted, descriptive p-values over each family of tests;
8. writes figures fig_E1_*.pdf/png, LaTeX booktabs tables E1_*.tex,
   <results>/summary.json, <results>/SUMMARY.md and <results>/E1_slopes.csv.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
import zlib
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import keys, style  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import LogLocator, NullFormatter  # noqa: E402

from znnbench.runner import aggregate  # noqa: E402

# ---------------------------------------------------------------------------
# analysis constants
# ---------------------------------------------------------------------------
METRIC = "mssre_w"              # absolute-window steady-state residual (fallback: mssre)
FLOOR_MAX = 1e-10               # a flat tail below this level is a round-off floor
FLOOR_FACTOR = 30.0             # exclude points within this factor of the floor
FLOOR_DEFAULT = 1e-14           # floor guess when an instance shows none
SLOPE_TOL = 0.3                 # |measured - expected| <= SLOPE_TOL -> "match"
# asymptotic window: every local slope within max(0.15, 0.03|m|) of the window median
# (review 2026-10-18: was max(0.2, 0.04|m|), which admitted monotone pre-asymptotic drifts
# of up to 0.4-0.5 inside a "consistent" window; 0.15 is the local-slope tolerance of
# tests/test_rk4_order.py and of the cross-experiment review checklist)
WIN_TOL_ABS, WIN_TOL_REL = 0.15, 0.03
DRIFT_TOL = 0.15                # monotone change of the local slopes inside the window -> "drift"
ABSURD_EXCESS = 3.0             # slope > expected + 3 or < -0.5 -> flagged "absurd"
# formulas whose steady state depends on x0 by construction (Broyden keeps a memory of its
# start, docs/spec/inverse_free_dtznn.md E5/E6): the x0-spread "unsettled" rule is not applied
X0_DEPENDENT = {"broyden_dtznn_zhang2012"}
# formulas reported outside the headline match fractions (listed separately in SUMMARY.md):
# the Broyden-aided DTZNN does not track time-varying-matrix problems at all (spec E6), so its
# curves measure "no tracking", not an order deficit of a look-ahead formula
HEADLINE_SEPARATE = {"broyden_dtznn_zhang2012"}
NO_TRACK_LEVEL = 1e-3           # separately reported scheme with min MSSRE >= this: "no tracking"
NONCONV_RATIO = 0.5             # MSSRE >= 0.5 x transient peak -> not converging (unstable)
SPREAD_REL = 0.01               # x0 spread > 1 % of the MSSRE ...
SPREAD_FLOOR = 2.0              # ... and > 2 x the round-off floor -> transient not decayed ("unsettled")
FLOOR_NOISE = 3.0               # floor points may fluctuate by this factor
N_BOOT, BOOT_SEED = 2000, 20261018
CONSTA_TAG = "+constA"
INSTANCE_ORDER = ["linsys_n4", "matinv_n4", "matinv_n16", "sylv_n4", "tvqp_eq", "tvne_guo2018",
                  "linsys_n4_f0.2", "linsys_n4+constA"]
MAIN_INSTANCES = INSTANCE_ORDER[:6]
INSTANCE_TITLE = {"linsys_n4": "linear system, n=4", "matinv_n4": "matrix inversion, n=4",
                  "matinv_n16": "matrix inversion, n=16", "sylv_n4": "Sylvester, n=4",
                  "tvqp_eq": "TV-QP (equality)", "tvne_guo2018": "nonlinear equations",
                  "linsys_n4_f0.2": "linear system, freq 0.2", "linsys_n4+constA": "linear system, const. A"}
INSTANCE_OMEGA = {"linsys_n4": 1.0, "matinv_n4": 1.0, "matinv_n16": 1.0, "sylv_n4": 1.0,
                  "tvqp_eq": 4.0, "tvne_guo2018": 1.0, "linsys_n4_f0.2": 0.2, "linsys_n4+constA": 1.0}
MODES = ["h_rec", "h_0.1", "gamma"]
MODE_TITLE = {"h_rec": r"fixed $h=h_\mathrm{rec}$", "h_0.1": r"fixed $h=0.1$",
              "gamma": r"fixed $\gamma=10$"}

# formula families (appendix small multiples; <= 8 per panel)
FAMILIES = [
    ("order 2-3 (Euler, 4-inst., AB2)",
     ["euler", "taylor_zhang", "zead_4i_general", "zead_4i_6321", "zead_4i_fifd", "ab2"]),
    ("order 4 (5/6-inst., AB3)",
     ["zead_5i", "zead_5i_general_a10.1", "zead_6i_guo2017", "zead_6i_qiu2019", "uhlig_2_3", "ab3",
      "zead_5i_mahuang2025", "cdtznn_mahuang2025"]),
    ("order 5 (7/8-inst., AB4)",
     ["zead_7i_general", "zead_8i_wu2020", "zead_8i_chen2019", "zead_7i_cang2024", "uhlig_3_4", "ab4"]),
    ("order 6+ (10/11-inst.)",
     ["zead_10i_wu2020", "zead_11i_chen2023", "uhlig_4_6"]),
    # members of the canonical "disc_implicit" group, but drawn with a runtime group so that
    # euler / taylor_zhang keep their "disc" slots (review 2026-10-18: keys.GROUPS["disc_implicit"]
    # gives taylor_zhang slot 5, so it appeared in two styles within this one figure)
    ("implicit / consistent", ["euler", "implicit_euler_err", "peb_euler", "exact_flow_euler",
                               "semi_implicit_euler", "taylor_zhang", "implicit_taylor_zhang",
                               "peb_taylor_zhang"]),
    ("other implicit, inverse-free",
     ["euler", "taylor_zhang", "implicit_taylor_zhang_naive", "exact_flow_taylor_zhang",
      "semi_implicit_taylor_zhang", "broyden_dtznn_zhang2012"]),
]
TAU_LIM = None                  # common x-limits of all tau axes (set in main from the data)


# ---------------------------------------------------------------------------
# style helpers
# ---------------------------------------------------------------------------
def make_group(name, members):
    """Register a runtime entity group (<= 8) in keys.GROUPS so that every
    member of the canonical "disc" group keeps its "disc" slot (colour,
    marker, line style) and the others fill the free slots in fixed order."""
    disc = keys.GROUPS["disc"]
    slots = [None] * 8
    for m in members:
        if m in disc:
            slots[disc.index(m)] = m
    rest = [m for m in members if m not in disc]
    fill = [2, 3, 4, 5, 6, 7, 0, 1]          # Euler / Taylor-Zhang slots (references) last
    for m in rest:
        free = [i for i in fill if slots[i] is None]
        if not free:
            break
        slots[free[0]] = m
    keys.GROUPS[name] = [s if s is not None else f"_free{i}" for i, s in enumerate(slots)]
    return name


def sty(group, key):
    return keys.style_in_group(group, key)


def nice_disc(lab):
    return lab.replace("_a10.1", "(a1=0.1)")


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
COLUMNS = ["config", "run_id", "disc_key", "disc_label", "problem_label", "wrapper_keys", "tau",
           "gain_mode", "gain_value", "gamma", "h", "seed", "x0_index", "traj", "status", "mssre",
           "mean_ssre", "mssre_w", "mean_w", "w_points", "steps", "steps_run", "wall", "wall_per_step",
           "peak_rss_mb", "n_gains", "B", "problem_kw_T", "max_residual", "x0_spread_w", "disc_expected_order",
           "disc_causal", "disc_implicit", "message"]


def load(results):
    df = aggregate(results, columns=COLUMNS)
    if df.empty:
        raise SystemExit(f"no E1 results under {results}")
    cats = {}
    for cj in sorted(Path(results).rglob("config.json")):
        c = json.loads(cj.read_text())
        d = c.get("description") or {}
        if isinstance(d, dict) and d.get("experiment") == "E1":
            cats[d["formula"]] = d
    metric = METRIC if METRIC in df.columns and df[METRIC].notna().any() else "mssre"
    df["metric"] = df[metric].astype(float)
    # divergence is +inf (censored); error / timeout / skipped rows are missing
    df.loc[df.status == "diverged", "metric"] = np.inf
    df.loc[~df.status.isin(["ok", "diverged"]), "metric"] = np.nan
    if "x0_spread_w" not in df.columns:
        df["x0_spread_w"] = np.nan
    df["spread"] = df["x0_spread_w"].astype(float)
    df.loc[df.status != "ok", "spread"] = np.nan
    df["instance"] = df.problem_label.astype(str) + np.where(
        df.wrapper_keys.astype(str).str.contains("ConstantMatrix"), CONSTA_TAG, "")

    def mode_of(r):
        if r.gain_mode == "fixed_gamma":
            return "gamma"
        cat = cats.get(r.disc_label, {}).get("catalogue", {})
        hrec = cat.get("h_rec")
        if hrec is not None and abs(r.gain_value - round(float(hrec), 6)) < 1e-9:
            return "h_rec"
        if abs(r.gain_value - 0.1) < 1e-9:
            return "h_0.1"
        return f"h_{r.gain_value:g}"

    keys_ = df[["gain_mode", "gain_value", "disc_label"]].drop_duplicates()
    keys_["mode"] = [mode_of(r) for r in keys_.itertuples()]
    df = df.merge(keys_, on=["gain_mode", "gain_value", "disc_label"], how="left")
    return df, cats, metric


def expected_order(cat, mode):
    """Expected residual slope: fixed-h -> claimed (q+1); fixed-gamma -> claimed-1;
    RK4: 4 at fixed gamma (classical), Prothero-Robinson predicts 2 at fixed h."""
    info = cat.get("catalogue", {})
    key = info.get("key")
    claimed = info.get("claimed_order")
    if key == "rk4":
        return 2 if mode.startswith("h") else 4
    if claimed is None:
        return None
    return int(claimed) if mode.startswith("h") else int(claimed) - 1


# ---------------------------------------------------------------------------
# curves, floors, asymptotic windows, fits
# ---------------------------------------------------------------------------
def _cens_quantile(Y, q):
    """Row-wise quantile of Y (NaN = missing, +inf = diverged/censored); +inf
    where the quantile falls among the censored values, NaN for empty rows."""
    out = np.full(Y.shape[0], np.nan)
    for i, row in enumerate(Y):
        v = np.sort(row[~np.isnan(row)])
        if not len(v):
            continue
        pos = q * (len(v) - 1)
        lo, hi = int(np.floor(pos)), int(np.ceil(pos))
        if not np.isfinite(v[hi]):
            out[i] = np.inf
        else:
            out[i] = v[lo] + (pos - lo) * (v[hi] - v[lo])
    return out


def build_curves(df, cats):
    curves = {}
    for (disc, inst, mode), g in df.groupby(["disc_label", "instance", "mode"], sort=False):
        piv = g.pivot_table(index="tau", columns=["seed", "x0_index"], values="metric", aggfunc="first",
                            dropna=False)
        piv = piv.sort_index()
        cols = piv.columns
        Y = piv.to_numpy(dtype=float)
        tau = piv.index.to_numpy(dtype=float)

        def aligned(col):
            if col not in g.columns:
                return np.full_like(Y, np.nan)
            return (g.pivot_table(index="tau", columns=["seed", "x0_index"], values=col, aggfunc="first",
                                  dropna=False).reindex(index=piv.index, columns=cols).to_numpy(dtype=float))

        peak = aligned("max_residual")
        S = aligned("spread")
        n_obs = np.sum(~np.isnan(Y), axis=1)
        n_div = np.sum(np.isinf(Y), axis=1)
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            med = _cens_quantile(Y, 0.5)
            q25 = _cens_quantile(Y, 0.25)
            q75 = _cens_quantile(Y, 0.75)
            # finite but not converging (growing towards the divergence threshold, or no decay)
            ratio = np.where(np.isfinite(Y) & (peak > 0), Y / peak, np.where(np.isinf(Y), np.inf, np.nan))
            nonconv = _cens_quantile(ratio, 0.5) >= NONCONV_RATIO
            spread = np.array([np.nanmedian(r[np.isfinite(r)]) if np.isfinite(r).any() else np.nan for r in S])
        finite = np.isfinite(med) & (n_obs > 0)
        nonconv = np.nan_to_num(nonconv, nan=False).astype(bool) & finite
        cat = cats.get(disc, {})
        cap_key = "fixed_gamma" if mode == "gamma" else "fixed_h"
        capped_taus = (cat.get("T_capped") or {}).get(cap_key, []) if isinstance(cat.get("T_capped"), dict) else []
        capped = np.array([any(np.isclose(t, c, rtol=1e-9) for c in capped_taus) for t in tau], dtype=bool)
        stable = finite & ~nonconv & ~capped
        Tm = g.groupby("tau").problem_kw_T.first().reindex(tau).to_numpy(dtype=float)
        curves[(disc, inst, mode)] = {"disc": disc, "instance": inst, "mode": mode, "tau": tau, "Y": Y,
                                      "seeds": np.array([c[0] for c in cols]),
                                      "stable": stable, "finite": finite, "nonconv": nonconv, "capped": capped,
                                      "n_obs": n_obs, "n_div": n_div, "spread": spread,
                                      "med": np.where(stable, med, np.nan), "med_raw": med,
                                      "q25": q25, "q75": q75, "T": Tm, "n_traj": Y.shape[1]}
    return curves


def local_slopes(tau, M):
    return np.log(M[1:] / M[:-1]) / np.log(tau[1:] / tau[:-1])


def detect_floor(c):
    """Indices (ascending tau) of the finest points forming a round-off floor and
    the floor level (median of those points), or ([], None).  Walking from the
    finest stable point towards coarse tau, a point belongs to the floor while
    it is below FLOOR_MAX and the curve does not yet rise with slope >= 1 to
    its coarser neighbour, unless that neighbour is still within FLOOR_NOISE of
    the floor points found so far (noisy floors)."""
    tau, med, st = c["tau"], c["med"], c["stable"]
    idx = [i for i in range(len(tau)) if st[i] and med[i] > 0]
    floor = []
    for a, b in zip(idx[:-1], idx[1:]):
        if med[a] >= FLOOR_MAX:
            break
        s = math.log(med[b] / med[a]) / math.log(tau[b] / tau[a])
        lo = min([med[i] for i in floor] + [med[a]])
        if s < 1.0 or med[b] <= FLOOR_NOISE * lo:
            floor.append(a)
        else:
            break
    if floor:
        return floor, float(np.median(med[floor]))
    return [], None


def instance_floors(curves):
    lev = {}
    for c in curves.values():
        fl, level = detect_floor(c)
        c["floor_idx"], c["floor_level"] = fl, level
        if level is not None:
            lev.setdefault(c["instance"], []).append(level)
    return {k: float(np.median(v)) for k, v in lev.items()}, lev


def fit_slope(x, y):
    xm, ym = x.mean(), y.mean()
    return float(((x - xm) * (y - ym)).sum() / ((x - xm) ** 2).sum())


def _curve_rng(*key):
    return np.random.default_rng([BOOT_SEED, zlib.crc32("|".join(map(str, key)).encode())])


def boot_draws(c, rng):
    """Two-stage (cluster) bootstrap column draws, shape (N_BOOT, n_cols):
    resample the problem instances (seeds) with replacement, then the initial
    states within each drawn instance."""
    seeds = c["seeds"]
    us = np.unique(seeds)
    groups = [np.nonzero(seeds == s)[0] for s in us]
    if len(us) < 2:                                   # one instance: plain resampling of x0
        n = len(seeds)
        return rng.integers(0, n, size=(N_BOOT, n))
    m = max(len(gr) for gr in groups)
    pad = np.array([np.resize(gr, m) for gr in groups])          # (n_seeds, m) column indices
    sb = rng.integers(0, len(us), size=(N_BOOT, len(us)))
    wb = rng.integers(0, m, size=(N_BOOT, len(us), m))
    return pad[sb[:, :, None], wb].reshape(N_BOOT, -1)


def boot_slopes(c, rows, draws):
    """Bootstrap slopes over the given column draws (censored medians: a
    replicate in which half the trajectories of a point diverged gives an
    infinite median and is dropped)."""
    Y = c["Y"][rows]
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        Yb = np.nanmedian(Y[:, draws], axis=2)                    # (n_rows, N_BOOT)
        x = np.log(c["tau"][rows])
        y = np.log(Yb)
        xm = x.mean()
        b = ((x - xm)[:, None] * (y - y.mean(axis=0))).sum(axis=0) / ((x - xm) ** 2).sum()
    return b


def _ci(bs):
    f = bs[np.isfinite(bs)]
    if len(f) < 0.9 * len(bs) or not len(f):
        return np.nan, np.nan
    lo, hi = np.percentile(f, [2.5, 97.5])
    return float(lo), float(hi)


def analyse_curve(c, inst_floor):
    tau, med, st = c["tau"], c["med"], c["stable"]
    floor_level = c.get("floor_level")
    ref = floor_level if floor_level is not None else inst_floor.get(c["instance"], FLOOR_DEFAULT)
    cut = FLOOR_FACTOR * ref
    with np.errstate(invalid="ignore"):
        unsettled = st & np.isfinite(c["spread"]) & (c["spread"] > np.maximum(SPREAD_REL * med,
                                                                              SPREAD_FLOOR * ref))
    if c["disc"] in X0_DEPENDENT:                     # x0-dependent steady state by construction
        unsettled = np.zeros_like(unsettled)
    c["unsettled"] = unsettled
    valid = st & ~unsettled & (med > cut) & np.isfinite(med)
    for i in c.get("floor_idx", []):
        valid[i] = False
    out = {"cut": cut, "floor_level": floor_level, "n_stable": int(st.sum()), "n_valid": int(valid.sum()),
           "n_unsettled": int(unsettled.sum()), "n_capped": int(c["capped"].sum()),
           "n_nonconv": int(c["nonconv"].sum()), "n_infinite": int((~c["finite"]).sum()),
           "n_partial_div": int(((c["n_div"] > 0) & c["finite"]).sum()),
           "slope": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "slope_all": np.nan, "slope_local_fine": np.nan,
           "win_lo": np.nan, "win_hi": np.nan, "n_win": 0, "flag": "", "window": [], "boot": None,
           "win_slope_range": np.nan, "drift": False}
    vidx = np.nonzero(valid)[0]
    if len(vidx) >= 2:
        out["slope_all"] = fit_slope(np.log(tau[vidx]), np.log(med[vidx]))
    if len(vidx) < 2:
        out["flag"] = "insufficient"
        return out
    # finest contiguous run of >= 2 valid points (review 2026-10-18: an isolated finest valid
    # point used to give an empty run and an IndexError, e.g. a lone point above the floor)
    runs, cur = [], [vidx[0]]
    for j in vidx[1:]:
        if j == cur[-1] + 1:
            cur.append(j)
        else:
            runs.append(cur)
            cur = [j]
    runs.append(cur)
    runs = [r_ for r_ in runs if len(r_) >= 2]
    if not runs:
        out["flag"] = "insufficient gap"
        return out
    run = np.array(runs[0])
    s = local_slopes(tau[run], med[run])
    out["slope_local_fine"] = float(s[0])
    k = 2
    for kk in range(3, len(run) + 1):
        ss = s[:kk - 1]
        m = np.median(ss)
        tol = max(WIN_TOL_ABS, WIN_TOL_REL * abs(m))
        if np.all(np.abs(ss - m) <= tol):
            k = kk
        else:
            break
    win = run[:k]
    out["window"] = [int(i) for i in win]
    out["n_win"] = len(win)
    out["win_lo"], out["win_hi"] = float(tau[win[0]]), float(tau[win[-1]])
    draws = boot_draws(c, _curve_rng(c["instance"], c["disc"]))
    bs = boot_slopes(c, win, draws)
    out["boot"] = bs
    out["ci_lo"], out["ci_hi"] = _ci(bs)
    if len(win) >= 3:
        out["slope"] = fit_slope(np.log(tau[win]), np.log(med[win]))
    else:
        out["slope"] = float(s[0])
        out["flag"] = "2-point"
    if len(vidx) > len(run):
        out["flag"] = (out["flag"] + " gap").strip()
    # pre-asymptotic drift: local slopes inside the window change monotonically by more
    # than DRIFT_TOL (the window is consistent within the tolerance but still trending)
    sw = s[:len(win) - 1]
    out["win_slope_range"] = float(sw.max() - sw.min()) if len(sw) else np.nan
    out["drift"] = bool(len(sw) >= 2 and abs(sw[0] - sw[-1]) > DRIFT_TOL
                        and (np.all(np.diff(sw) > 0) or np.all(np.diff(sw) < 0)))
    if out["drift"]:
        out["flag"] = (out["flag"] + " drift").strip()
    return out


def verdict(slope, lo, hi, exp, local_fine=np.nan, drift=False):
    """match: |slope - expected| <= SLOPE_TOL (or a narrow CI covers it);
    pre-asymptotic: not a match, but the finest local slope is closer to the
    expected order than the fit and the curve is still drifting (or only a
    2-point slope exists) -- the grid did not reach the asymptotic range;
    otherwise lower / higher."""
    if exp is None or not np.isfinite(slope):
        return "n/a"
    if abs(slope - exp) <= SLOPE_TOL:
        return "match"
    if np.isfinite(lo) and lo <= exp <= hi and (hi - lo) <= 2 * SLOPE_TOL:
        return "match (CI)"
    if drift and np.isfinite(local_fine) and abs(local_fine - exp) < abs(slope - exp):
        return "pre-asymptotic"
    return "higher" if slope > exp else "lower"


def holm(pvals):
    """Holm step-down adjusted p-values (dict key -> p)."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, run, out = len(items), 0.0, {}
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


# ---------------------------------------------------------------------------
# coupling diagnostic ||A^{-1} A_dot|| of the instances
# ---------------------------------------------------------------------------
def coupling_L(instance, seeds=(0, 1, 2)):
    """max_t ||J^{-1} dJ/dt||_2 over t in [0, 6] (median over seeds); the
    per-step perturbation of the linear-ZNN error recursion is tau * L."""
    from znnbench import registry
    vals = []
    for sd in seeds:
        try:
            if instance.startswith("linsys"):
                if instance.endswith(CONSTA_TAG):
                    return 0.0
                fr = 0.2 if "f0.2" in instance else 1.0
                p = registry.make_problem("linear_system", n=4, kappa=10, freq=fr, seed=sd)
                A, Ad = p.A, p.A.dot
            elif instance.startswith("matinv"):
                n = 16 if instance.endswith("16") else 4
                p = registry.make_problem("matrix_inversion", n=n, kappa=10, freq=1.0, seed=sd)
                A, Ad = p.A, p.A.dot
            elif instance == "tvqp_eq":
                p = registry.make_problem("tvqp_eq", variant="zhang2021")

                def A(t, p=p):
                    (W, q, Am, b), _ = p.qp(t)
                    return p._K(W, Am)

                def Ad(t, p=p):
                    (W, q, Am, b), (Wd, qd, Amd, bd) = p.qp(t)
                    return p._K(Wd, Amd)
            else:
                return None
            ts = np.linspace(0, 6, 241)
            vals.append(max(np.linalg.norm(np.linalg.solve(A(t), Ad(t)), 2) for t in ts))
        except Exception:  # noqa: BLE001
            return None
    return float(np.median(vals)) if vals else None


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def _logx(ax, scale=1.0):
    """Log tau axis, increasing to the right in EVERY E1 figure (review 2026-10-18: the
    old code set inverted limits and then inverted the axis again, so fig_E1_controls
    (which reset the limits) ran the other way from the other figures)."""
    ax.set_xscale("log")
    if TAU_LIM is not None:
        ax.set_xlim(TAU_LIM[0] * scale / 1.35, TAU_LIM[1] * scale * 1.35)
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=12))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=12))
    ax.xaxis.set_minor_formatter(NullFormatter())


def _loglog(ax):
    _logx(ax)
    ax.set_yscale("log")
    ax.yaxis.set_minor_formatter(NullFormatter())


def plot_curve(ax, c, res, st, label=None, x_scale=1.0, band=True):
    tau, med = c["tau"] * x_scale, c["med"]
    ok = c["stable"] & np.isfinite(med) & (med > 0)
    if not ok.any():
        return None
    line, = ax.plot(tau[ok], med[ok], color=st["color"], linestyle=st["linestyle"], marker=None,
                    lw=1.0, label=label)
    win = set(res.get("window", []))
    fin = [i for i in np.nonzero(ok)[0] if i in win]
    out = [i for i in np.nonzero(ok)[0] if i not in win]
    if fin:
        ax.plot(tau[fin], med[fin], linestyle="none", marker=st["marker"], color=st["color"], ms=3.6)
    if out:
        ax.plot(tau[out], med[out], linestyle="none", marker=st["marker"], ms=3.6,
                markerfacecolor="white", markeredgecolor=st["color"])
    if band:
        q25, q75 = c["q25"][ok], c["q75"][ok]
        q75 = np.where(np.isfinite(q75), q75, 1e1)          # upper quartile diverged: band to the top
        ax.fill_between(tau[ok], q25, q75, color=st["color"], alpha=0.12, lw=0)
    return line


def legend_handles(group, members):
    hs = []
    for m in members:
        s = sty(group, m)
        hs.append(Line2D([], [], color=s["color"], linestyle=s["linestyle"], marker=s["marker"], ms=3.6,
                         lw=1.0, label=nice_disc(m)))
    return hs


def fig_small_multiples(curves, results, mode, members, group, name, title_extra=""):
    insts = [i for i in INSTANCE_ORDER if any(k[1] == i and k[2] == mode for k in curves)]
    members = [m for m in members if any(k[0] == m and k[2] == mode and np.isfinite(c["med"]).any()
                                         for k, c in curves.items())]
    if not insts or not members:
        return None
    ncol = 4
    nrow = math.ceil(len(insts) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(style.DOUBLE, 1.95 * nrow + 0.55), squeeze=False)
    for ax, inst in zip(axes.flat, insts):
        for m in members:
            c = curves.get((m, inst, mode))
            if c is None:
                continue
            plot_curve(ax, c, results.get((m, inst, mode), {}), sty(group, m))
        _loglog(ax)
        ax.set_title(INSTANCE_TITLE.get(inst, inst), pad=3)
        ax.set_ylim(1e-16, 1e1)
        ax.set_yticks([1e-15, 1e-12, 1e-9, 1e-6, 1e-3, 1e0])
    for ax in axes.flat[len(insts):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("MSSRE (median)")
    for j in range(ncol):
        vis = [axes[i, j] for i in range(nrow) if axes[i, j].get_visible()]
        if vis:
            vis[-1].set_xlabel(r"sampling period $\tau$ (s)")
    fig.legend(handles=legend_handles(group, members), loc="lower center", ncol=min(len(members), 8),
               bbox_to_anchor=(0.5, -0.005), frameon=False)
    fig.suptitle(f"E1: steady-state residual vs $\\tau$, {MODE_TITLE[mode]}{title_extra} "
                 "(filled: fitted asymptotic range; open: excluded; band: IQR)", fontsize=8, y=0.995)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    return style.savefig(fig, name)


def fig_families(curves, results, inst_list, mode, name):
    fams = []
    for t, mem in FAMILIES:
        grp = mem if isinstance(mem, str) else None
        mem = keys.GROUPS[mem] if grp else mem
        # members with a curve in THIS mode (a formula without h_rec, e.g. Broyden, has none at
        # fixed h_rec and must not get a legend entry)
        mem = [m for m in mem if any(k[0] == m and k[2] == mode and np.isfinite(c["med"]).any()
                                     for k, c in curves.items())]
        if mem:
            fams.append((t, mem, grp))
    insts = [i for i in inst_list if any(k[1] == i for k in curves)]
    if not fams or not insts:
        return None
    fig, axes = plt.subplots(len(insts), len(fams), figsize=(style.DOUBLE, 1.75 * len(insts) + 1.0),
                             squeeze=False, sharey=True)
    for j, (title, mem, grp) in enumerate(fams):
        g = grp or make_group(f"e1_fam{j}", mem)
        for i, inst in enumerate(insts):
            ax = axes[i, j]
            for m in mem:
                c = curves.get((m, inst, mode))
                if c is not None:
                    plot_curve(ax, c, results.get((m, inst, mode), {}), sty(g, m), band=False)
            _loglog(ax)
            ax.set_ylim(1e-16, 1e1)
            ax.set_yticks([1e-15, 1e-10, 1e-5, 1e0])
            if i == 0:
                ax.set_title(title, fontsize=6.5, pad=3)
            if j == 0:
                ax.set_ylabel(f"{INSTANCE_TITLE.get(inst, inst)}\nMSSRE", fontsize=6.5)
            if i == len(insts) - 1:
                ax.set_xlabel(r"$\tau$ (s)")
            ax.tick_params(labelsize=6)
        axes[-1, j].legend(handles=legend_handles(g, mem), loc="upper center", fontsize=5.2, ncol=1,
                           bbox_to_anchor=(0.5, -0.32), frameon=False)
    fig.suptitle(f"E1: all formulas by family, {MODE_TITLE[mode]}", fontsize=8, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return style.savefig(fig, name)


def fig_orders(table, cats, name):
    """Dot plot: measured asymptotic slope per instance vs the expected order (Supplementary
    Material figure, drawn at the paper's text width)."""
    if table.empty:
        return None
    forms = (table[["disc", "claimed"]].drop_duplicates()
             .assign(c=lambda d: d.claimed.fillna(4.5)).sort_values(["c", "disc"]))
    order = list(forms.disc)
    make_group("e1_instance", INSTANCE_ORDER)
    panels = [("h_rec", MODE_TITLE["h_rec"] + r" (expected $q+1$)"), ("gamma", MODE_TITLE["gamma"] + r" (expected $q$)")]
    vals = np.concatenate([table.slope.to_numpy(float), table.expected.to_numpy(float)])
    vals = vals[np.isfinite(vals)]
    x0 = max(0.0, math.floor(vals.min() * 2) / 2 - 0.5) if vals.size else 0.0
    x1 = min(8.5, math.ceil(vals.max() * 2) / 2 + 0.5) if vals.size else 8.5
    with style.print_rc():
        fig, axes = plt.subplots(1, 2, figsize=(style.TEXTWIDTH, 0.128 * len(order) + 1.25), sharey=True,
                                 constrained_layout=True)
        fig.get_layout_engine().set(w_pad=0.03, h_pad=0.03, wspace=0.03)
        ypos = {d: i for i, d in enumerate(order)}
        for ax, (mode, title) in zip(axes, panels):
            sub = table[table["mode"] == mode]
            for d in order:
                e = sub[sub.disc == d].expected.dropna()
                if len(e):
                    ev = float(e.iloc[0])
                    ax.plot([ev, ev], [ypos[d] - 0.44, ypos[d] + 0.44], color=style.INK, lw=1.5, zorder=1)
            for k, inst in enumerate(INSTANCE_ORDER):
                s_ = sub[sub.instance == inst]
                if s_.empty:
                    continue
                st = sty("e1_instance", inst)
                yy = np.array([ypos[d] for d in s_.disc]) + (k - 3.5) * 0.085
                ok = s_.flag.fillna("").str.contains("2-point").to_numpy()
                ms = 3.9 if st["marker"] in ("*", "P", "X") else 3.3
                ax.plot(s_.slope.to_numpy()[~ok], yy[~ok], linestyle="none", marker=st["marker"],
                        color=st["color"], ms=ms, zorder=3)
                if ok.any():
                    ax.plot(s_.slope.to_numpy()[ok], yy[ok], linestyle="none", marker=st["marker"], ms=ms,
                            markerfacecolor="white", markeredgecolor=st["color"], zorder=3)
            ax.set_title(title)
            ax.set_xlabel("measured residual order")
            ax.set_xlim(x0, x1)
            ax.set_ylim(-0.7, len(order) - 0.3)
            ax.xaxis.set_major_locator(plt.MultipleLocator(1))
            ax.grid(axis="y", visible=False)
            ax.tick_params(axis="y", length=0)
        axes[0].set_yticks(range(len(order)))
        hrec_of = {d: cats.get(d, {}).get("catalogue", {}).get("h_rec") for d in order}
        axes[0].set_yticklabels([nice_disc(d) + (" (non-causal)" if d == "rk4" else "")
                                 + (r" (no $h_\mathrm{rec}$)" if hrec_of.get(d) is None else "") for d in order],
                                fontsize=7.5)
        handles = [Line2D([], [], color=style.INK, lw=1.5, label="expected order")]
        for inst in [i for i in INSTANCE_ORDER if i in set(table.instance)]:
            st = sty("e1_instance", inst)
            handles.append(Line2D([], [], linestyle="none", marker=st["marker"], color=st["color"],
                                  ms=3.9 if st["marker"] in ("*", "P", "X") else 3.3,
                                  label=INSTANCE_TITLE.get(inst, inst).replace("n=16", "$n=16$")
                                  .replace("n=4", "$n=4$")))
        handles.append(Line2D([], [], linestyle="none", marker="o", color=style.INK2, markerfacecolor="white",
                              ms=3.3, label="open: two-point slope"))
        fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, columnspacing=1.4,
                   handletextpad=0.3)
        return style.savefig(fig, name)


def fig_local_slopes(curves, results, members, group, name):
    insts = [i for i in ["linsys_n4", "linsys_n4_f0.2", "linsys_n4+constA", "matinv_n4", "tvqp_eq"]
             if any(k[1] == i for k in curves)]
    if not insts:
        return None
    fig, axes = plt.subplots(1, len(insts), figsize=(style.DOUBLE, 2.3), sharey=True, squeeze=False)
    for ax, inst in zip(axes[0], insts):
        for m in members:
            c = curves.get((m, inst, "h_rec"))
            if c is None:
                continue
            ok = c["stable"] & np.isfinite(c["med"]) & (c["med"] > 0)
            ok &= ~c.get("unsettled", np.zeros(len(ok), bool))
            for i in c.get("floor_idx", []):
                ok[i] = False
            ok &= c["med"] > results.get((m, inst, "h_rec"), {}).get("cut", 0)
            idx = np.nonzero(ok)[0]
            if len(idx) < 2:
                continue
            pairs = [(a, b) for a, b in zip(idx[:-1], idx[1:]) if b == a + 1]
            if not pairs:
                continue
            a = np.array([p[0] for p in pairs])
            b = np.array([p[1] for p in pairs])
            s = np.log(c["med"][b] / c["med"][a]) / np.log(c["tau"][b] / c["tau"][a])
            tm = np.sqrt(c["tau"][a] * c["tau"][b])
            st = sty(group, m)
            ax.plot(tm, s, color=st["color"], linestyle=st["linestyle"], marker=st["marker"], ms=3.2, lw=1.0)
        _logx(ax)
        ax.set_ylim(0, 9)
        ax.set_yticks(range(0, 10))
        ax.set_title(INSTANCE_TITLE.get(inst, inst), fontsize=7, pad=3)
        ax.set_xlabel(r"$\tau$ (s), geometric midpoint")
    axes[0, 0].set_ylabel("local slope (consecutive ratio)")
    fig.legend(handles=legend_handles(group, members), loc="lower center", ncol=min(8, len(members)),
               bbox_to_anchor=(0.5, -0.01), frameon=False)
    fig.suptitle(r"E1: consecutive-ratio diagnostic at fixed $h=h_\mathrm{rec}$ (points above the floor)",
                 fontsize=8, y=0.995)
    fig.tight_layout(rect=(0, 0.12, 1, 0.95))
    return style.savefig(fig, name)


def fig_controls(curves, results, forms, name):
    """MSSRE vs omega*tau on the three linear-system variants (collapse check)."""
    insts = ["linsys_n4", "linsys_n4_f0.2", "linsys_n4+constA"]
    forms = [f for f in forms if any((f, i, "h_rec") in curves for i in insts)][:4]
    if not forms:
        return None
    g = make_group("e1_instance", INSTANCE_ORDER)
    fig, axes = plt.subplots(1, len(forms), figsize=(style.DOUBLE, 2.2), sharey=True, squeeze=False)
    for ax, f in zip(axes[0], forms):
        for inst in insts:
            c = curves.get((f, inst, "h_rec"))
            if c is not None:
                plot_curve(ax, c, results.get((f, inst, "h_rec"), {}), sty(g, inst),
                           x_scale=INSTANCE_OMEGA[inst], band=False)
        _loglog(ax)
        if TAU_LIM is not None:
            om = min(INSTANCE_OMEGA[i] for i in insts)
            ax.set_xlim(TAU_LIM[0] * om / 1.35, TAU_LIM[1] * 1.35)
        ax.set_ylim(1e-16, 1e0)
        ax.set_title(nice_disc(f), fontsize=7, pad=3)
        ax.set_xlabel(r"normalised step $\omega\tau$ (rad)")
    axes[0, 0].set_ylabel(r"MSSRE, fixed $h=h_\mathrm{rec}$")
    hs = [Line2D([], [], color=sty(g, i)["color"], linestyle=sty(g, i)["linestyle"],
                 marker=sty(g, i)["marker"], ms=3.2, label=INSTANCE_TITLE[i]) for i in insts]
    fig.legend(handles=hs, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01), frameon=False)
    fig.suptitle(r"E1 controls: time rescaling (freq 0.2 collapses onto freq 1 in $\omega\tau$) "
                 r"and the $\tau\|A^{-1}\dot A\|$ coupling (constant $A$)", fontsize=8, y=0.995)
    fig.tight_layout(rect=(0, 0.1, 1, 0.94))
    return style.savefig(fig, name)


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
def _fmt(v, nd=2):
    return "--" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def _tex_escape(s):
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def write_tex(path, header, rows, colspec, caption, label, note=None, longtable=False):
    lines = []
    if longtable:
        lines += [f"\\begin{{longtable}}{{{colspec}}}", f"\\caption{{{caption}}}\\label{{{label}}}\\\\",
                  "\\toprule", " & ".join(header) + r" \\", "\\midrule", "\\endhead"]
        lines += [" & ".join(r) + r" \\" for r in rows]
        lines += ["\\bottomrule"]
        if note:
            lines += [f"\\multicolumn{{{len(header)}}}{{p{{0.95\\linewidth}}}}{{\\footnotesize {note}}}\\\\"]
        lines += ["\\end{longtable}"]
    else:
        lines += ["\\begin{table}[t]", "\\centering", "\\footnotesize", f"\\caption{{{caption}}}",
                  f"\\label{{{label}}}", f"\\begin{{tabular}}{{{colspec}}}", "\\toprule",
                  " & ".join(header) + r" \\", "\\midrule"]
        lines += [" & ".join(r) + r" \\" for r in rows]
        lines += ["\\bottomrule", "\\end{tabular}"]
        if note:
            lines += [f"\\par\\smallskip\\parbox{{0.95\\linewidth}}{{\\scriptsize {note}}}"]
        lines += ["\\end{table}"]
    Path(path).write_text("\n".join(lines) + "\n")
    return path


def summarize_modes(table, disc, mode, insts):
    s = table[(table.disc == disc) & (table["mode"] == mode) & (table.instance.isin(insts))]
    v = s.slope.to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    if not len(v):
        return np.nan, np.nan, np.nan, 0
    return float(np.median(v)), float(v.min()), float(v.max()), len(v)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--figdir", required=True)
    ap.add_argument("--tabdir", required=True)
    args = ap.parse_args(argv)
    res_dir, fig_dir, tab_dir = Path(args.results), Path(args.figdir), Path(args.tabdir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    style.FIG_DIR = fig_dir                      # style.savefig writes here
    rng = np.random.default_rng(BOOT_SEED)       # only for the across-instance resampling of RK4
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    df, cats, metric = load(res_dir)
    scale = next(iter(cats.values())).get("scale", "?") if cats else "?"
    curves = build_curves(df, cats)
    global TAU_LIM
    all_tau = np.concatenate([c["tau"] for c in curves.values()])
    TAU_LIM = (float(all_tau.min()), float(all_tau.max()))
    inst_floor, floor_lists = instance_floors(curves)

    results, rows = {}, []
    for k, c in curves.items():
        r = analyse_curve(c, inst_floor)
        results[k] = r
        cat = cats.get(c["disc"], {})
        info = cat.get("catalogue", {})
        exp = expected_order(cat, c["mode"])
        rows.append({"disc": c["disc"], "key": info.get("key", c["disc"]), "instance": c["instance"],
                     "mode": c["mode"], "claimed": info.get("claimed_order"), "expected": exp,
                     "h_rec": info.get("h_rec"), "slope": r["slope"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
                     "slope_all": r["slope_all"], "slope_local_fine": r["slope_local_fine"],
                     "n_win": r["n_win"], "win_lo": r["win_lo"], "win_hi": r["win_hi"],
                     "n_valid": r["n_valid"], "n_stable": r["n_stable"], "n_tau": len(c["tau"]),
                     "floor_level": r["floor_level"], "cut": r["cut"], "flag": r["flag"],
                     "n_unsettled": r["n_unsettled"], "n_capped": r["n_capped"], "n_nonconv": r["n_nonconv"],
                     "n_infinite": r["n_infinite"], "n_partial_div": r["n_partial_div"],
                     "n_traj_diverged": int(c["n_div"].sum()), "n_traj_obs": int(c["n_obs"].sum()),
                     "win_slope_range": r["win_slope_range"], "drift": r["drift"],
                     "verdict": verdict(r["slope"], r["ci_lo"], r["ci_hi"], exp, r["slope_local_fine"],
                                        r["drift"] or "2-point" in r["flag"]),
                     "verdict_all": verdict(r["slope_all"], np.nan, np.nan, exp),
                     "min_mssre": float(np.nanmin(c["med"])) if np.isfinite(c["med"]).any() else np.nan})
    table = pd.DataFrame(rows)
    # a separately reported scheme that never gets below 1e-3 does not track: its slope is not
    # an order (and "pre-asymptotic" would be misleading)
    notrack = table.disc.isin(HEADLINE_SEPARATE) & ~(table.min_mssre < NO_TRACK_LEVEL)
    table.loc[notrack, "verdict"] = "no tracking"
    ex = table.expected.astype(float)
    absurd = table.slope.notna() & ((table.slope > ex + ABSURD_EXCESS) | (table.slope < -0.5))
    table.loc[absurd, "flag"] = (table.loc[absurd, "flag"].fillna("") + " absurd").str.strip()
    table.to_csv(res_dir / "E1_slopes.csv", index=False)

    forms_present = sorted(table.disc.unique())
    # ------------------------------------------------------------ figures
    disc_members = [m for m in keys.GROUPS["disc"] if m in forms_present]
    figs = {}
    sfx = ""
    for mode, nm in [("h_rec", "fig_E1_mssre_fixed_h"), ("gamma", "fig_E1_mssre_fixed_gamma"),
                     ("h_0.1", "fig_E1_mssre_h01")]:
        p = fig_small_multiples(curves, results, mode, disc_members, "disc", nm + sfx)
        if p:
            figs[nm] = str(p)
    p = fig_orders(table[table["mode"].isin(["h_rec", "gamma"])], cats, "fig_E1_orders")
    if p:
        figs["fig_E1_orders"] = str(p)
    p = fig_local_slopes(curves, results, disc_members, "disc", "fig_E1_local_slopes")
    if p:
        figs["fig_E1_local_slopes"] = str(p)
    p = fig_families(curves, results, ["linsys_n4", "linsys_n4+constA"], "h_rec", "fig_E1_families_fixed_h")
    if p:
        figs["fig_E1_families_fixed_h"] = str(p)
    p = fig_families(curves, results, ["linsys_n4", "linsys_n4+constA"], "gamma",
                     "fig_E1_families_fixed_gamma")
    if p:
        figs["fig_E1_families_fixed_gamma"] = str(p)
    ctrl_forms = [f for f in ["taylor_zhang", "zead_6i_guo2017", "uhlig_3_4", "uhlig_4_6", "zead_5i", "euler"]
                  if f in forms_present]
    p = fig_controls(curves, results, ctrl_forms, "fig_E1_controls")
    if p:
        figs["fig_E1_controls"] = str(p)

    # ------------------------------------------------------------ statistics
    from scipy.stats import kendalltau

    main_insts = [i for i in MAIN_INSTANCES if i in set(table.instance)]
    kt = {}
    for mode in ["h_rec", "gamma"]:
        for inst in main_insts + ["linsys_n4+constA"]:
            s = table[(table["mode"] == mode) & (table.instance == inst)].dropna(subset=["slope", "expected"])
            s = s[s.key != "rk4"]
            if len(s) >= 4 and s.expected.nunique() > 1:
                t, pv = kendalltau(s.expected, s.slope)
                kt[f"{mode}|{inst}"] = {"tau_b": float(t), "p": float(pv), "n": int(len(s))}
    # does a higher claimed order mean a smaller residual at a practical tau?  Censoring
    # (review 2026-10-18): a diverged point (censored median +inf) or a formula that is not
    # run at the common h because h = 0.1 is outside its stable range counts as the WORST
    # accuracy (tied), instead of silently dropping out of the ranking (which favoured the
    # fragile high-order formulas).  The measured MSSRE is used whatever the point's status
    # (it is what a user gets at that tau); fixed h_rec compares formulas at different h
    # (error constant ~ 1/h), fixed h = 0.1 removes that confound.
    acc_kt = {}
    for mode in ["h_rec", "h_0.1"]:
        for tref in [1e-2, 1e-3]:
            for inst in main_insts:
                vals, n_cens = [], 0
                for f in forms_present:
                    cat = cats.get(f, {}).get("catalogue", {})
                    if cat.get("key") == "rk4" or cat.get("claimed_order") is None:
                        continue
                    c = curves.get((f, inst, mode))
                    if c is None:
                        if mode == "h_0.1" and any(k[0] == f and k[1] == inst for k in curves):
                            vals.append((cat["claimed_order"], np.inf))
                            n_cens += 1
                        continue
                    j = np.nonzero(np.isclose(c["tau"], tref))[0]
                    if not len(j) or np.isnan(c["med_raw"][j[0]]):
                        continue
                    v = float(c["med_raw"][j[0]])
                    n_cens += int(not np.isfinite(v))
                    vals.append((cat["claimed_order"], v))
                if len(vals) >= 4:
                    a = np.array(vals, dtype=float)
                    with np.errstate(divide="ignore"):
                        t, pv = kendalltau(a[:, 0], -np.log(a[:, 1]))
                    acc_kt[f"{mode}|tau={tref:g}|{inst}"] = {"tau_b": float(t), "p": float(pv), "n": int(len(a)),
                                                             "n_censored_worst": int(n_cens)}

    # multiple comparisons: Holm adjustment over each family of Kendall tests
    for fam in (kt, acc_kt):
        adj = holm({k: v["p"] for k, v in fam.items()})
        for k, v in fam.items():
            v["p_holm"] = float(adj[k])

    def frac_match(sub, col="verdict", ok=("match", "match (CI)")):
        s = sub[sub.expected.notna()]
        if not len(s):
            return np.nan, 0
        return float(s[col].isin(list(ok)).mean()), int(len(s))

    head = {}
    nonmatch = {}
    separate = {}
    for mode in ["h_rec", "h_0.1", "gamma"]:
        sub = table[(table["mode"] == mode) & table.instance.isin(main_insts) & (table.key != "rk4")
                    & ~table.disc.isin(HEADLINE_SEPARATE)]
        for d in sorted(HEADLINE_SEPARATE & set(table.disc)):
            s2 = table[(table.disc == d) & (table["mode"] == mode) & table.instance.isin(main_insts)]
            if len(s2):
                separate.setdefault(d, {})[mode] = {
                    r.instance: {"slope": r.slope, "expected": r.expected, "verdict": r.verdict,
                                 "min_mssre": r.min_mssre} for r in s2.itertuples()}
        fa, n = frac_match(sub)
        ft, _ = frac_match(sub, ok=("match",))
        fn, _ = frac_match(sub, "verdict_all")
        d_asym = (sub.slope - sub.expected).abs()
        d_all = (sub.slope_all - sub.expected).abs()
        vc = sub[sub.expected.notna()].verdict.value_counts()
        head[mode] = {"curves": n, "frac_match_asymptotic": fa, "frac_match_tolerance_only": ft,
                      "frac_match_naive_all_points": fn,
                      "verdict_counts": {str(k): int(v) for k, v in vc.items()},
                      "n_drift": int(sub.drift.fillna(False).astype(bool).sum()),
                      "n_absurd": int(sub.flag.fillna("").str.contains("absurd").sum()),
                      "trajectories_diverged": int(sub.n_traj_diverged.sum()),
                      "trajectories_observed": int(sub.n_traj_obs.sum()),
                      "median_abs_dev_asymptotic": float(np.nanmedian(d_asym)) if len(sub) else np.nan,
                      "median_abs_dev_naive": float(np.nanmedian(d_all)) if len(sub) else np.nan,
                      "n_2point_or_insufficient": int(sub.flag.fillna("").str.contains("2-point|insufficient").sum()),
                      "excluded_points": {"unsettled_transient": int(sub.n_unsettled.sum()),
                                          "capped_horizon": int(sub.n_capped.sum()),
                                          "not_converging": int(sub.n_nonconv.sum()),
                                          "median_infinite": int(sub.n_infinite.sum()),
                                          "finite_with_some_diverged": int(sub.n_partial_div.sum())}}
        bad = sub[sub.expected.notna() & ~sub.verdict.isin(["match", "match (CI)"])]
        nonmatch[mode] = [f"{r.disc}|{r.instance}: {r.slope:.2f} vs {int(r.expected)} ({r.verdict}"
                          f"{', ' + r.flag if isinstance(r.flag, str) and r.flag else ''})"
                          for r in bad.itertuples()]

    # RK4: fixed-gamma minus fixed-h slope; paired two-stage bootstrap (the same
    # resampled instances / initial states in both modes, see _curve_rng)
    rk = {}
    pair_diffs = []
    for inst in main_insts + ["linsys_n4+constA", "linsys_n4_f0.2"]:
        a, b = results.get(("rk4", inst, "h_rec")), results.get(("rk4", inst, "gamma"))
        if a and b and np.isfinite(a["slope"]) and np.isfinite(b["slope"]):
            d = None
            if a.get("boot") is not None and b.get("boot") is not None:
                d = b["boot"] - a["boot"]
                pair_diffs.append(d)
            rk[inst] = {"slope_fixed_h": a["slope"], "ci_fixed_h": [a["ci_lo"], a["ci_hi"]],
                        "slope_fixed_gamma": b["slope"], "ci_fixed_gamma": [b["ci_lo"], b["ci_hi"]],
                        "diff_gamma_minus_h": b["slope"] - a["slope"],
                        "ci_diff_paired": list(_ci(d)) if d is not None else [np.nan, np.nan],
                        "slope_fixed_h01": results.get(("rk4", inst, "h_0.1"), {}).get("slope", np.nan)}
    if rk:
        hv = np.array([v["slope_fixed_h"] for v in rk.values()])
        gv = np.array([v["slope_fixed_gamma"] for v in rk.values()])
        bs = rng.integers(0, len(hv), size=(N_BOOT, len(hv)))
        diff = gv[bs].mean(axis=1) - hv[bs].mean(axis=1)
        pooled = np.nanmean(np.vstack(pair_diffs), axis=0) if pair_diffs else np.array([np.nan])
        rk_summary = {"median_fixed_h": float(np.median(hv)), "median_fixed_gamma": float(np.median(gv)),
                      "mean_diff_gamma_minus_h": float(gv.mean() - hv.mean()),
                      "ci95_diff_paired_within_instances": list(_ci(pooled)),
                      "ci95_diff_over_instances": [float(np.percentile(diff, 2.5)),
                                                   float(np.percentile(diff, 97.5))],
                      "n_instances": int(len(hv))}
    else:
        rk_summary = {}

    # controls and onset of the asymptotic range (fixed h = h_rec)
    Lc = {inst: coupling_L(inst) for inst in INSTANCE_ORDER if inst in set(table.instance)}

    def onset_info(f, inst):
        c, r = curves.get((f, inst, "h_rec")), results.get((f, inst, "h_rec"))
        if c is None or r is None:
            return None
        h = cats.get(f, {}).get("catalogue", {}).get("h_rec")
        L, om = Lc.get(inst), INSTANCE_OMEGA.get(inst)
        st_t = c["tau"][c["stable"] & ~c.get("unsettled", np.zeros(len(c["tau"]), bool))]
        coarse = float(st_t.max()) if len(st_t) else np.nan
        onset = r["win_hi"]
        censored = bool(np.isfinite(onset) and np.isclose(onset, coarse))
        ok = h and np.isfinite(onset)
        return {"slope": r["slope"], "ci": [r["ci_lo"], r["ci_hi"]], "slope_all": r["slope_all"],
                "window": [r["win_lo"], r["win_hi"]], "n_win": r["n_win"], "flag": r["flag"],
                "n_stable": int(c["stable"].sum()), "n_tau": int(len(c["tau"])), "coarsest_stable_tau": coarse,
                "n_unsettled": int(r["n_unsettled"]),
                "onset_censored": censored,
                "coupling_at_onset": float(onset * L / h) if (ok and L is not None) else None,
                "omega_tau_over_h_at_onset": float(onset * om / h) if (ok and om) else None}

    ctrl = {}
    for f in forms_present:
        row = {inst: onset_info(f, inst) for inst in ["linsys_n4", "linsys_n4_f0.2", "linsys_n4+constA"]}
        row = {k: v for k, v in row.items() if v is not None}
        if row:
            ctrl[f] = row
    hi_order = [f for f in ctrl if (cats.get(f, {}).get("catalogue", {}).get("claimed_order") or 0) >= 4
                and cats.get(f, {}).get("catalogue", {}).get("key") != "rk4"]
    pairs = [(ctrl[f]["linsys_n4"], ctrl[f]["linsys_n4+constA"]) for f in hi_order
             if "linsys_n4" in ctrl[f] and "linsys_n4+constA" in ctrl[f]]
    if pairs:
        r_on = np.array([b["window"][1] / a["window"][1] for a, b in pairs
                         if np.isfinite(a["window"][1]) and np.isfinite(b["window"][1])])
        ctrl_summary = {"n_formulas_order_ge4": len(pairs),
                        "unstable_or_nonconv_points_linsys": int(sum(a["n_tau"] - a["n_stable"] for a, _ in pairs)),
                        "unstable_or_nonconv_points_constA": int(sum(b["n_tau"] - b["n_stable"] for _, b in pairs)),
                        "unsettled_points_linsys": int(sum(a["n_unsettled"] for a, _ in pairs)),
                        "unsettled_points_constA": int(sum(b["n_unsettled"] for _, b in pairs)),
                        "points_per_curve": int(pairs[0][0]["n_tau"]),
                        "median_onset_ratio_constA_over_linsys": float(np.median(r_on)) if len(r_on) else None,
                        "median_window_points_linsys": float(np.median([a["n_win"] for a, _ in pairs])),
                        "median_window_points_constA": float(np.median([b["n_win"] for _, b in pairs]))}
    else:
        ctrl_summary = {}
    # where does the asymptotic range start?  tau* L / h and omega tau* / h at the
    # coarsest fitted tau (uncensored windows only), all formulas, instances with known L
    onsets = []
    for f in forms_present:
        if cats.get(f, {}).get("catalogue", {}).get("key") == "rk4":
            continue
        for inst in INSTANCE_ORDER:
            o = onset_info(f, inst)
            if o and not o["onset_censored"] and o["n_win"] >= 3:
                onsets.append((f, inst, o["coupling_at_onset"], o["omega_tau_over_h_at_onset"]))
    cl = np.array([x[2] for x in onsets if x[2] is not None and x[2] > 0], dtype=float)
    ow = np.array([x[3] for x in onsets if x[3] is not None], dtype=float)
    onset_summary = {"n_curves_uncensored": len(onsets),
                     "coupling_ratio_tauL_over_h": ({"median": float(np.median(cl)),
                                                     "q25": float(np.percentile(cl, 25)),
                                                     "q75": float(np.percentile(cl, 75)), "n": int(len(cl))}
                                                    if len(cl) else None),
                     "omega_tau_over_h": ({"median": float(np.median(ow)), "q25": float(np.percentile(ow, 25)),
                                           "q75": float(np.percentile(ow, 75)), "n": int(len(ow))}
                                          if len(ow) else None)}

    floors = {inst: {"median": float(np.median(v)), "min": float(np.min(v)), "max": float(np.max(v)),
                     "n_curves": len(v)} for inst, v in floor_lists.items()}

    # asymptotic-window width (decades) vs claimed order (fixed-h h_rec, main instances)
    width = {}
    sub = table[(table["mode"] == "h_rec") & table.instance.isin(main_insts) & (table.n_win >= 2)]
    for cl, g in sub.groupby("claimed"):
        w = np.log10(g.win_hi / g.win_lo)
        width[str(int(cl))] = {"median_decades": float(np.median(w)), "n": int(len(g))}

    # cost
    ok = df[df.status == "ok"].drop_duplicates("run_id")
    cost = {"specs": int(df.run_id.nunique()), "trajectories": int(len(df)),
            "status_counts": {k: int(v) for k, v in df.drop_duplicates(["run_id", "traj"]).status.value_counts().items()},
            "steps_total": float(ok.steps.sum()), "wall_total_s": float(ok.wall.sum()),
            "wall_per_step_us_median": float(1e6 * ok.wall_per_step.median()),
            "wall_per_step_us_by_instance": {i: float(1e6 * g.wall_per_step.median())
                                             for i, g in ok.assign(instance=ok.instance).groupby("instance")},
            "wall_per_step_us_by_ngains": {int(i): float(1e6 * g.wall_per_step.median())
                                           for i, g in ok.groupby("n_gains")},
            "wall_per_step_us_rk4": float(1e6 * ok[ok.disc_key == "rk4"].wall_per_step.median())
            if (ok.disc_key == "rk4").any() else None,
            "peak_rss_mb_max": float(df.peak_rss_mb.max())}

    # ------------------------------------------------------------ tables
    tabs = {}
    fam_order = []
    for _, mem in FAMILIES:
        mem = keys.GROUPS[mem] if isinstance(mem, str) else mem
        fam_order += [m for m in mem if m not in fam_order]
    fam_order += [f for f in forms_present if f not in fam_order]
    claimed_of = {d: cats.get(d, {}).get("catalogue", {}).get("claimed_order") for d in forms_present}
    disp = sorted([f for f in forms_present], key=lambda d: (claimed_of.get(d) or 4.5,
                                                              fam_order.index(d) if d in fam_order else 99))
    trows = []
    for d in disp:
        info = cats.get(d, {}).get("catalogue", {})
        cl = info.get("claimed_order")
        hr = info.get("h_rec")
        cells = [_tex_escape(nice_disc(d)), "--" if not info.get("instants") else str(info["instants"]),
                 "--" if hr is None else f"{hr:.3g}"]
        for mode in ["h_rec", "h_0.1", "gamma"]:
            e = expected_order(cats.get(d, {}), mode)
            m, lo, hi, n = summarize_modes(table, d, mode, main_insts)
            ex = "--" if e is None else str(e)
            cells += [ex, "--" if not n else f"{m:.2f} [{lo:.2f}, {hi:.2f}] ({n})"]
        r = results.get((d, "linsys_n4+constA", "h_rec"))
        cells.append(_fmt(r["slope"]) if r else "--")
        fl = table[(table.disc == d) & table.instance.isin(main_insts)].floor_level.dropna()
        cells.append("--" if fl.empty else f"{np.median(fl):.0e}")
        if info.get("key") == "rk4":
            cells[3] = "2$^\\dagger$"
        trows.append(cells)
    tabs["E1_orders"] = str(write_tex(
        tab_dir / "E1_orders.tex",
        ["formula", "inst.", r"$h_\mathrm{rec}$", r"exp.", r"fixed $h_\mathrm{rec}$", "exp.", r"fixed $h=0.1$",
         "exp.", r"fixed $\gamma=10$", r"const.\ $A$", "floor"],
        trows, "lrrrlrlrlrr",
        "E1: claimed (expected) vs measured residual order. Measured = least-squares slope of the median "
        "MSSRE over the asymptotic range selected by the consecutive-ratio diagnostic; median [min, max] "
        f"over the {len(main_insts)} main problem instances (number of instances with a fit). "
        "Fixed $h$: expected $q+1$; fixed $\\gamma$: expected $q$.",
        "tab:E1_orders",
        note="$^\\dagger$RK4 has no ZNN residual claim; 2 is the Prothero--Robinson (stiff order reduction) "
             "prediction at fixed $h$, 4 the classical order at fixed $\\gamma$. const.\\ $A$: linear system "
             "with $A$ frozen (no $\\tau\\|A^{-1}\\dot A\\|$ coupling), fixed $h_\\mathrm{rec}$. floor: median "
             "round-off floor of the MSSRE curves (main instances)."))
    # detail long table
    drows = []
    for d in disp:
        for inst in INSTANCE_ORDER:
            cells = [_tex_escape(nice_disc(d)), _tex_escape(inst)]
            any_ = False
            for mode in ["h_rec", "gamma"]:
                s = table[(table.disc == d) & (table.instance == inst) & (table["mode"] == mode)]
                if s.empty:
                    cells += ["--", "--", "--"]
                    continue
                any_ = True
                r = s.iloc[0]
                ci = "--" if not np.isfinite(r.ci_lo) else f"[{r.ci_lo:.2f}, {r.ci_hi:.2f}]"
                win = "--" if not np.isfinite(r.win_lo) else f"{r.win_lo:.0e}--{r.win_hi:.0e}"
                fl = "" if not r.flag else "$^\\ast$"
                cells += [f"{_fmt(r.slope)}{fl} {ci}", _fmt(r.slope_all), win]
            if any_:
                drows.append(cells)
    tabs["E1_orders_detail"] = str(write_tex(
        tab_dir / "E1_orders_detail.tex",
        ["formula", "instance", r"fixed $h_\mathrm{rec}$: slope [95\% CI]", "all pts", r"range $\tau$",
         r"fixed $\gamma$: slope [95\% CI]", "all pts", r"range $\tau$"],
        drows, "llllllll",
        "E1 detail: asymptotic-range slope with two-stage cluster bootstrap 95\\% CI (2000 resamples of the "
        "problem seeds, then of the initial states within each seed), "
        "naive least-squares slope over all stable points above the floor, and the fitted $\\tau$ range.",
        "tab:E1_orders_detail", note="$^\\ast$ fewer than 3 consistent points (2-point local slope) or a gap.",
        longtable=True))
    frows = []
    for inst in INSTANCE_ORDER:
        if inst in floors:
            f = floors[inst]
            frows.append([_tex_escape(INSTANCE_TITLE.get(inst, inst)), f"{f['median']:.1e}", f"{f['min']:.1e}",
                          f"{f['max']:.1e}", str(f["n_curves"]),
                          "--" if Lc.get(inst) is None else f"{Lc[inst]:.2f}"])
    tabs["E1_floors"] = str(write_tex(
        tab_dir / "E1_floors.tex",
        ["instance", "floor (median)", "min", "max", "curves", r"$\max_t\|A^{-1}\dot A\|$"], frows, "lrrrrr",
        "E1: round-off floors of the MSSRE curves (levels of the flat fine-$\\tau$ tails below $10^{-10}$) and "
        "the coupling constant $L=\\max_t\\|J^{-1}\\dot J\\|_2$ of each instance.", "tab:E1_floors",
        note=f"Points within a factor {FLOOR_FACTOR:g} of a curve's floor are excluded from the order fits."))
    crow = []
    for f in disp:
        row = ctrl.get(f)
        if not row:
            continue
        cells = [_tex_escape(nice_disc(f)), "--" if cats.get(f, {}).get("catalogue", {}).get("h_rec") is None
                 else f"{cats[f]['catalogue']['h_rec']:.3g}", str(expected_order(cats.get(f, {}), "h_rec") or "--")]
        for inst in ["linsys_n4", "linsys_n4_f0.2", "linsys_n4+constA"]:
            r = row.get(inst)
            if r is None:
                cells += ["--", "--"]
            else:
                cpl = r["omega_tau_over_h_at_onset"] if inst.endswith(CONSTA_TAG) else r["coupling_at_onset"]
                cells += [f"{_fmt(r['slope'])} / {_fmt(r['slope_all'])}",
                          "--" if cpl is None else ((r"$\geq$" if r["onset_censored"] else "") + f"{cpl:.2f}")]
        crow.append(cells)
    tabs["E1_controls"] = str(write_tex(
        tab_dir / "E1_controls.tex",
        ["formula", r"$h_\mathrm{rec}$", "exp.", "freq 1: asym./all", r"$\tau^\ast L/h$", "freq 0.2: asym./all",
         r"$\tau^\ast L/h$", r"const.\ $A$: asym./all", r"$\omega\tau^\ast/h$"],
        crow, "lrrllllll",
        "E1 controls (linear system, fixed $h=h_\\mathrm{rec}$): asymptotic-range / naive all-points slopes, "
        "and the onset ratio at the coarsest fitted $\\tau^\\ast$: coupling $\\tau^\\ast L/h$ "
        "($L=\\max_t\\|A^{-1}\\dot A\\|$) for time-varying $A$, data rate $\\omega\\tau^\\ast/h$ for constant $A$; "
        "$\\geq$: the window reaches the coarsest stable $\\tau$ (censored).", "tab:E1_controls"))

    # ------------------------------------------------------------ summary
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

    h_conv = {}                                   # c0-normalised step of the source (h_paper = h/c0)
    for d in disp:
        info = cats.get(d, {}).get("catalogue", {})
        if info.get("paper_h_factor"):
            h_conv[d] = {"c0": info["paper_h_factor"], "h_rec": info.get("h_rec"),
                         "h_rec_paper": info.get("h_rec_paper"), "h_common_paper": info.get("h_common_paper")}
    per_formula = {}
    for d in disp:
        per_formula[d] = {"claimed": claimed_of.get(d),
                          "h_rec": cats.get(d, {}).get("catalogue", {}).get("h_rec")}
        for mode in ["h_rec", "h_0.1", "gamma"]:
            m, lo, hi, n = summarize_modes(table, d, mode, main_insts)
            per_formula[d][mode] = {"expected": expected_order(cats.get(d, {}), mode), "median": m,
                                    "min": lo, "max": hi, "n": n}
    summary = {"experiment": "E1", "scale": scale, "metric": metric, "n_formulas": len(forms_present),
               "formulas": forms_present, "instances": sorted(set(table.instance)),
               "analysis": {"floor_max": FLOOR_MAX, "floor_factor": FLOOR_FACTOR, "floor_noise": FLOOR_NOISE,
                            "slope_tol": SLOPE_TOL, "window_tol": [WIN_TOL_ABS, WIN_TOL_REL],
                            "nonconv_ratio": NONCONV_RATIO, "spread_rel": SPREAD_REL, "spread_floor": SPREAD_FLOOR,
                            "drift_tol": DRIFT_TOL, "absurd_excess": ABSURD_EXCESS,
                            "x0_dependent_no_spread_rule": sorted(X0_DEPENDENT),
                            "kendall": "descriptive: p-values (Holm-adjusted within each family) are not "
                                       "inferential claims; items are formulas (n ~ 10-30, many tied orders)",
                            "accuracy_ranking_censoring": "diverged point or formula unstable at the common h "
                                                          "= worst accuracy (tied)",
                            "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
                            "bootstrap": "two-stage cluster: seeds (instances), then x0 within seed; seeded per curve",
                            "divergence": "censored median (+inf); point unusable if >= half diverged",
                            "match": "|slope - expected| <= slope_tol, or 95% CI covers expected with width <= 2 slope_tol"},
               "headline": head, "non_matching_curves": nonmatch,
               "headline_excluded_formulas": {"rk4": "non-causal reference, reported in 'rk4'",
                                              **{d: "reported in 'separate' (no tracking on time-varying-"
                                                    "matrix problems, docs/spec/inverse_free_dtznn.md E6)"
                                                 for d in sorted(HEADLINE_SEPARATE)}},
               "separate": separate, "h_convention": h_conv,
               "rk4": {"per_instance": rk, "summary": rk_summary},
               "controls": {"summary": ctrl_summary, "coupling_L": Lc, "per_formula": ctrl},
               "asymptotic_onset": onset_summary,
               "kendall_claimed_vs_measured": kt, "kendall_claimed_order_vs_accuracy": acc_kt,
               "floors": floors, "window_width_by_claimed_order": width, "per_formula": per_formula,
               "cost": cost, "figures": figs, "tables": tabs}
    fc = res_dir / "full_cost_estimate.json"
    if fc.exists():
        try:
            e = json.loads(fc.read_text())
            summary["full_scale_cost_estimate"] = {k: e.get(k) for k in
                                                   ("specs", "steps", "core_hours", "workers",
                                                    "wall_hours_at_workers", "longest_spec_s", "longest_spec",
                                                    "per_instance_core_hours")}
        except Exception:  # noqa: BLE001
            pass
    summary = clean(summary)
    (res_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    write_summary_md(res_dir / "SUMMARY.md", summary, table, main_insts)
    print(json.dumps({"headline": summary["headline"], "rk4": summary["rk4"]["summary"],
                      "controls": summary["controls"]["summary"]}, indent=1))
    print(f"figures -> {fig_dir}; tables -> {tab_dir}; summary -> {res_dir / 'summary.json'}")
    return 0


def write_summary_md(path, s, table, main_insts):
    h = s["headline"]
    L = []
    L.append(f"# E1 order verification -- summary ({s['scale']} scale)\n")
    L.append(f"Metric: `{s['metric']}` (max ||E|| over the absolute window [t_end - 1.5 s, t_end]); "
             f"{s['n_formulas']} formulas; instances: {', '.join(s['instances'])}.\n")
    L.append("## Headline numbers (main instances, RK4 excluded)\n")
    L.append("| mode | curves | match (asymptotic fit) | of which within tol. | match (naive all-points fit) | median abs. dev. asym. | naive | 2-point/insufficient | excluded points: unsettled / capped / not conv. / inf. median / partial div. |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    pct = lambda x: "--" if x is None else f"{100 * x:.0f} %"
    num = lambda x: "--" if x is None else f"{x:.2f}"
    for m in ["h_rec", "h_0.1", "gamma"]:
        if m in h:
            v = h[m]
            e = v.get("excluded_points", {})
            L.append(f"| {m} | {v['curves']} | {pct(v['frac_match_asymptotic'])} | {pct(v.get('frac_match_tolerance_only'))} "
                     f"| {pct(v['frac_match_naive_all_points'])} | {num(v['median_abs_dev_asymptotic'])} "
                     f"| {num(v['median_abs_dev_naive'])} | {v['n_2point_or_insufficient']} "
                     f"| {e.get('unsettled_transient', 0)} / {e.get('capped_horizon', 0)} / {e.get('not_converging', 0)} "
                     f"/ {e.get('median_infinite', 0)} / {e.get('finite_with_some_diverged', 0)} |")
    L.append("")
    L.append("Expected: fixed h -> claimed order q+1; fixed gamma -> q. \"match\" = |slope - expected| <= 0.3, "
             "or the two-stage bootstrap 95 % CI covers it with width <= 0.6.  Excluded points (summed over the "
             "curves): unsettled = x0-dependent transient left in the window; capped = horizon cap below the "
             "frozen-rate requirement; partial div. = finite censored median with some diverged trajectories.  "
             "Asymptotic window: every local slope within max(0.15, 0.03|m|) of the window median; \"drift\" = "
             "the local slopes inside the window still change monotonically by > 0.15; \"pre-asymptotic\" = not a "
             "match, drifting (or 2-point), and the finest local slope is closer to the expected order than the "
             "fit (the grid did not reach the asymptotic range; inconclusive, not a failure).\n")
    L.append("Verdicts per mode: " + "; ".join(
        f"{m}: " + ", ".join(f"{k} {v}" for k, v in (h[m].get("verdict_counts") or {}).items())
        + f" (drift-flagged {h[m].get('n_drift', 0)}, absurd {h[m].get('n_absurd', 0)})"
        for m in ["h_rec", "h_0.1", "gamma"] if m in h) + ".\n")
    L.append("**Censoring.** " + "; ".join(
        f"{m}: {h[m].get('trajectories_diverged', 0)} of {h[m].get('trajectories_observed', 0)} trajectories "
        "diverged" for m in ["h_rec", "h_0.1", "gamma"] if m in h)
        + f". Run statuses (all instances): {s['cost']['status_counts']}. Diverged trajectories enter every "
        "median as +inf (a point is unusable once half of its trajectories diverged); error / timeout rows are "
        "missing (never counted as converged); in the accuracy rankings a diverged point, or a formula that is "
        "not run at h = 0.1 because 0.1 is outside its stable range, is the worst rank (tied).\n")
    sep = s.get("separate") or {}
    if sep:
        L.append("**Reported separately (not in the headline fractions).** " + " ".join(
            f"`{d}`: " + "; ".join(
                f"{m}: " + ", ".join(f"{i} {num(v['slope'])} (exp. {v['expected']}, {v['verdict']}, min MSSRE "
                                     f"{v['min_mssre']:.1e})" if v.get("min_mssre") is not None else
                                     f"{i} {num(v['slope'])} (exp. {v['expected']}, {v['verdict']})"
                                     for i, v in mm.items())
                for m, mm in dd.items()) + "." for d, dd in sep.items())
            + " Its claimed O(tau^2) is the source's claim; it holds on nonlinear_equations (guo2018) but the "
              "scheme does not track time-varying-matrix problems (docs/spec/inverse_free_dtznn.md E6), and its "
              "steady state depends on x0 (no x0-spread rule applied).\n")
    hp = s.get("h_convention") or {}
    if hp:
        L.append("**Step-size convention.** Every h is ZNN-Bench's raw h = gamma*tau (tau-form coefficients). "
                 "Formulas whose source normalises by c0 (h_paper = h/c0; Ma & Huang 2025 Table 1): " + "; ".join(
                     f"`{d}` c0 = {v['c0']:.4g}: h_rec = {v['h_rec']:.4g} (h_paper {v['h_rec_paper']:.4g}), "
                     f"h = 0.1 (h_paper {v['h_common_paper']:.4g})" if v.get("h_rec") else
                     f"`{d}` c0 = {v['c0']:.4g}: h = 0.1 (h_paper {v['h_common_paper']:.4g})"
                     for d, v in hp.items()) + ".\n")
    L.append("**Confounds.** (1) Fixed h = h_rec compares formulas at different h (error constant ~ 1/h); "
             "the common h = 0.1 removes this. (2) RK4 is non-causal (samples t_k + tau/2, t_k + tau) and costs 4 "
             "field evaluations per step; all other formulas, including the implicit / consistent variants, make "
             "one causal evaluation at (t_k, x_k). (3) Transients: horizons sized for e^-37 decay with a coupling "
             "allowance, every run ends at the same absolute time, and x0-spread-flagged points are excluded. "
             "(4) Linear activation only, so no unit dependence of the activation; the round-off floor is "
             "absolute (instance scale: ~1e-14 for the unit-scale instances, ~1e-12 for nonlinear_equations at "
             "t ~ 200 s with |x4| ~ 200). (5) With 3 problem seeds the bootstrap CIs are cluster CIs over 3 "
             "instances (descriptive); deterministic instances (tvqp_eq, nonlinear_equations) vary x0 only, so "
             "their CIs are degenerate; the systematic (window) uncertainty is shown by the naive all-points "
             "slope next to the asymptotic one.\n")
    nm = s.get("non_matching_curves") or {}
    if any(nm.values()):
        L.append("Non-matching curves (main instances):\n")
        for m, lst in nm.items():
            for x in lst:
                L.append(f"- {m}: {x}")
        L.append("")
    rk = s["rk4"]["summary"]
    if rk:
        L.append("## RK4 (non-causal reference)\n")
        L.append(f"Median slope at fixed h = h_rec: **{rk['median_fixed_h']:.2f}**; at fixed gamma = 10: "
                 f"**{rk['median_fixed_gamma']:.2f}** (mean difference {rk['mean_diff_gamma_minus_h']:.2f} over "
                 f"{rk['n_instances']} instances; paired two-stage bootstrap 95 % CI "
                 f"[{num(rk['ci95_diff_paired_within_instances'][0])}, "
                 f"{num(rk['ci95_diff_paired_within_instances'][1])}]; resampling the instances: "
                 f"[{rk['ci95_diff_over_instances'][0]:.2f}, {rk['ci95_diff_over_instances'][1]:.2f}]).  Fixed h keeps gamma*tau = h constant, i.e. a stiff "
                 "Prothero-Robinson problem with stiffness h per step: explicit RK4 (stage order 1) shows order "
                 "reduction to ~2; at fixed gamma (h -> 0, non-stiff) the classical order 4 appears.\n")
    c = s["controls"]["summary"]
    if c:
        L.append("## Controls: coupling tau*||A^-1 A_dot|| vs h (fixed h = h_rec, claimed order >= 4)\n")
        L.append(f"Over {c['n_formulas_order_ge4']} high-order formulas the 1 rad/s linear system has "
                 f"{c['unstable_or_nonconv_points_linsys']} unstable / non-converging tau points "
                 f"(of {c['points_per_curve']} per curve) against {c['unstable_or_nonconv_points_constA']} with A "
                 f"frozen, plus {c.get('unsettled_points_linsys', 0)} vs {c.get('unsettled_points_constA', 0)} "
                 "finite points whose transient had not decayed in the window (x0 spread); the asymptotic range extends to a "
                 f"{c['median_onset_ratio_constA_over_linsys'] or float('nan'):.1f}x coarser tau with constant A "
                 f"(median window {c['median_window_points_linsys']:.0f} vs {c['median_window_points_constA']:.0f} "
                 "points).  The low-frequency instance (freq 0.2) is the 1 rad/s instance with tau rescaled by 0.2 "
                 "(A_0.2(t) = A_1(0.2 t), h invariant), so it collapses onto it in omega*tau up to the different "
                 "phase of the 1.5 s window (fig_E1_controls): it shifts but does not widen the asymptotic window.\n")
    o = s.get("asymptotic_onset") or {}
    if o.get("coupling_ratio_tauL_over_h") or o.get("omega_tau_over_h"):
        cr, ow = o.get("coupling_ratio_tauL_over_h"), o.get("omega_tau_over_h")
        txt = "Onset of the asymptotic range (coarsest fitted tau*, uncensored windows): "
        if cr:
            txt += (f"tau* L / h = {cr['median']:.2f} [IQR {cr['q25']:.2f}-{cr['q75']:.2f}] (n={cr['n']}, "
                    "L = max_t ||J^-1 dJ/dt||); ")
        if ow:
            txt += (f"omega tau* / h = {ow['median']:.2f} [IQR {ow['q25']:.2f}-{ow['q75']:.2f}] (n={ow['n']}). ")
        txt += ("The order q+1 is observed only once the data change per step is small against the step size h "
                "of the error recursion.\n")
        L.append(txt)
    L.append("## Per formula (median measured slope over main instances [min, max] (n))\n")
    L.append("| formula | claimed | fixed h_rec (exp.) | fixed h=0.1 (exp.) | fixed gamma (exp.) |")
    L.append("|---|---|---|---|---|")
    for d, v in s["per_formula"].items():
        cells = []
        for m in ["h_rec", "h_0.1", "gamma"]:
            x = v[m]
            if not x["n"]:
                cells.append(f"-- ({x['expected']})")
            else:
                cells.append(f"{x['median']:.2f} [{x['min']:.1f}, {x['max']:.1f}] ({x['n']}) ({x['expected']})")
        L.append(f"| {d} | {v['claimed']} | " + " | ".join(cells) + " |")
    L.append("")
    if s["floors"]:
        L.append("## Round-off floors (median MSSRE level of flat fine-tau tails)\n")
        for inst, f in s["floors"].items():
            L.append(f"- {inst}: {f['median']:.1e} (range {f['min']:.1e} .. {f['max']:.1e}, {f['n_curves']} curves)")
        L.append("")
    if s["kendall_claimed_vs_measured"]:
        L.append("## Kendall tau_b (descriptive)\n")
        L.append("p-values are descriptive (n = number of formulas, many tied orders; Holm-adjusted within each "
                 "family); no inferential claim rests on them.\n")
        for k, v in s["kendall_claimed_vs_measured"].items():
            L.append(f"- claimed vs measured order, {k}: tau_b = {v['tau_b']:.2f} (p = {v['p']:.2g}, "
                     f"Holm {v.get('p_holm', float('nan')):.2g}, n = {v['n']})")
        for k, v in s["kendall_claimed_order_vs_accuracy"].items():
            L.append(f"- claimed order vs accuracy (-log MSSRE), {k}: tau_b = {v['tau_b']:.2f} "
                     f"(p = {v['p']:.2g}, Holm {v.get('p_holm', float('nan')):.2g}, n = {v['n']}, "
                     f"censored as worst {v.get('n_censored_worst', 0)})")
        L.append("")
    w = s["window_width_by_claimed_order"]
    if w:
        L.append("## Width of the asymptotic range (decades of tau, fixed h_rec, main instances)\n")
        L.append(", ".join(f"order {k}: {v['median_decades']:.2f} (n={v['n']})" for k, v in w.items()) + "\n")
    L.append("## Interpretation\n")
    L.append(interpretation(s))
    cst = s["cost"]
    L.append("\n## Run statistics\n")
    L.append(f"{cst['specs']} specs, {cst['trajectories']} trajectory rows, {cst['steps_total']:.3e} steps, "
             f"{cst['wall_total_s'] / 3600:.2f} core-hours of simulate wall; median wall/step "
             f"{cst['wall_per_step_us_median']:.0f} us; max peak RSS {cst['peak_rss_mb_max']:.0f} MB; "
             f"statuses {cst['status_counts']}.\n")
    Path(path).write_text("\n".join(L) + "\n")


def interpretation(s):
    h = s["headline"]
    parts = []
    if "h_rec" in h and h["h_rec"]["frac_match_asymptotic"] is not None:
        parts.append(f"At fixed h = h_rec the asymptotic-range slope matches the claimed order q+1 for "
                     f"{100 * h['h_rec']['frac_match_asymptotic']:.0f} % of the (formula, main instance) curves, "
                     f"against {100 * (h['h_rec']['frac_match_naive_all_points'] or 0):.0f} % for a naive fit over "
                     "all stable points (at the common h = 0.1: "
                     f"{100 * (h.get('h_0.1', {}).get('frac_match_asymptotic') or 0):.0f} % vs "
                     f"{100 * (h.get('h_0.1', {}).get('frac_match_naive_all_points') or 0):.0f} %): pre-asymptotic "
                     "points at coarse tau bias naive fits.")
    if "gamma" in h and h["gamma"]["frac_match_asymptotic"] is not None:
        parts.append(f"At fixed gamma = 10 the slope equals q (one order lost, as the PLAN hypothesises) for "
                     f"{100 * h['gamma']['frac_match_asymptotic']:.0f} % of the curves; "
                     f"{h['gamma']['n_2point_or_insufficient']} curves have fewer than 3 points between the "
                     "stability limit tau < h_max/gamma and the round-off floor.")
    if s["rk4"]["summary"]:
        r = s["rk4"]["summary"]
        parts.append(f"RK4 is order {r['median_fixed_h']:.1f} at fixed h (Prothero-Robinson order reduction, "
                     f"docs/spec/rk4_order_reduction.md) and {r['median_fixed_gamma']:.1f} at fixed gamma: its "
                     "classical order is only visible in the continuous-time convention.")
    o = s.get("asymptotic_onset") or {}
    if o.get("omega_tau_over_h"):
        parts.append(f"The asymptotic range starts when omega tau <~ {o['omega_tau_over_h']['median']:.2f} h "
                     "(median over uncensored curves), i.e. only once the data change per step is small against "
                     "the formula's h; high-order formulas need small h (h_rec 0.03-0.1), which pushes the onset "
                     "to small tau.")
    w = s.get("window_width_by_claimed_order") or {}
    if w:
        parts.append("Median width of the fitted range: " + ", ".join(
            f"order {k}: {v['median_decades']:.1f} decades" for k, v in w.items()) +
            " -- for orders 5-6 the range is squeezed between this onset and the round-off floor "
            "(1e-15..1e-13).")
    return " ".join(parts)


if __name__ == "__main__":
    sys.exit(main())
