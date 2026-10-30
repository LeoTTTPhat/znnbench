#!/usr/bin/env python
"""Analysis of E4 -- fairness re-ranking of "proposed vs baseline" ZNN models.

    python experiments/analyze_E4.py --results results/mini/E4 \
        --figdir figures/mini --tabdir results/tables/mini

Inputs: ``<results>/sweep`` (runner chunk parquets of the configs E4_P0 ...
E4_P4b and the sensitivity configs, see experiments/configs/E4.py) and
``<results>/calib`` (calibration JSONs of experiments/tasks/E4_calibrate.py).

Per protocol x problem x metric (time-to-1e-3, MSSRE) every model gets the
median over the seeds x B trajectories (diverged / failed = inf; MSSRE clamped
at the round-off floor 1e-12).  time-to-1e-3 counts only if the residual then
stays below 1e-3 for at least DWELL = 5 % of the horizon (t <= 0.95 T), else
it is censored (inf): the runner's t_eps equals T when a period-2 chattering
trace happens to end on its low phase.  Models are ranked by the median with
a resolution tolerance (medians within 10 % for MSSRE, 1 % + 1 recording
interval for time-to-1e-3 tie; single linkage on the sorted medians; average
ranks for ties), so that models at the same truncation / round-off floor are
not ordered by noise.  Agreement with P0 is Kendall
tau_b over the ranked panel, with a paired hierarchical bootstrap (2000
resamples, seeded: seeds with replacement, then initial states within each
seed with replacement; the same resample for every protocol and model).

Outputs: figures fig_E4_*.pdf/png in --figdir, LaTeX booktabs tables E4_*.tex
in --tabdir (need booktabs, amsmath, graphicx, array; wide tables are table* with
resizebox), ``<results>/summary.json`` and ``<results>/SUMMARY.md``.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "experiments"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import keys  # noqa: E402
import style  # noqa: E402
from znnbench.runner import aggregate, load_traces  # noqa: E402


def _load_cfg():
    spec = importlib.util.spec_from_file_location("_e4_cfg", ROOT / "experiments" / "configs" / "E4.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _load_cfg()

FLOOR = 1e-12
DWELL = 0.05                 # time-to-eps must be followed by >= DWELL*T below eps (else censored)
TIE_RTOL = {"t3": 0.01, "mssre": 0.10}   # medians closer than this (relative) tie in the ranking
TIE_ATOL_SAMPLES = 1         # t3: plus one recording interval (t_eps is quantised to the record grid)
# MSSRE window flagged as an unfinished transient if the residual envelope falls by
# more than 2x (ratio < TRANSIENT_DECAY) or grows by more than 2x (ratio >
# TRANSIENT_GROWTH, e.g. a late overshoot) inside the window, unless the window
# sits at round-off (max < TRANSIENT_LEVEL).  Review 2026-10-18: the former 4x
# decay threshold missed linear ZNN at gamma = 1 (ratio 0.45, still a pure
# exponential transient at t = 0.75 T).
TRANSIENT_DECAY = 0.5
TRANSIENT_GROWTH = 2.0
TRANSIENT_LEVEL = 1e-9
N_BOOT = 2000
BOOT_SEED = 20261018
METRICS = ["t3", "mssre"]
METRIC_COL = {"t3": "t_eps_0.001", "mssre": "mssre"}
METRIC_TEX = {"t3": r"$t_{10^{-3}}$", "mssre": "MSSRE"}
METRIC_LONG = {"t3": r"time to $\|E\|_2<10^{-3}$", "mssre": "MSSRE"}
MAIN = list(C.PROTOCOLS)
SENS = list(C.SENS_PROTOCOLS)
COMPARE = [p for p in MAIN if p != "P0"]
UNITS_ALL = sorted({float(u) for s in C.SCALES.values() for u in s["units"]})
UNITP = C.unit_protocols(UNITS_ALL)          # every possible unit-arm protocol name
PROB_SHORT = {"matrix_inversion": "matrix inv.", "tvqp_eq": "TV-QP"}


def ref_of(p):
    """The as-published protocol of the same error unit (P0 or P0_u<u>)."""
    return "P0" + C.unit_suffix(C.split_protocol(p)[1])


def is_p0(p):
    return C.split_protocol(p)[0] == "P0"


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load_calibration(results):
    """{(problem, model, unit): calibration dict} (unit 1.0 = original error unit)."""
    cdir = Path(results) / "calib"
    tf = cdir / "tasks.jsonl"
    out = {}
    if not tf.exists():
        return out
    for line in tf.read_text().splitlines():
        if not line.strip():
            continue
        t = json.loads(line)
        p = Path(t["done"])
        if not p.exists():
            p = cdir / "out" / p.name
        if p.exists():
            c = json.loads(p.read_text())
            out[(c["problem"], c["model"], float(c.get("unit", 1.0)))] = c
    return out


COLUMNS = ["config", "run_id", "problem_key", "model_label", "model_key", "disc_key", "tau", "gamma", "h",
           "seed", "x0_index", "traj", "status", "message", "mssre", "t_eps_0.001", "t_eps_1e-06",
           "peak_xdot", "initial_residual", "max_residual", "chatter_vec", "peak_rate_ss", "wall",
           "wall_per_step", "steps", "steps_run", "peak_rss_mb", "traces_file", "record_every", "T", "B"]


def load_results(results):
    df = aggregate(Path(results) / "sweep", columns=COLUMNS)
    if df.empty:
        raise SystemExit(f"no results under {results}/sweep")
    df["protocol"] = df["config"].map(C.protocol_of)
    # inherited protocols (P1pole): the (problem, model) pairs whose gain equals
    # the base protocol's are not re-run; take them from the base protocol
    extra = []
    for p, base in C.INHERIT.items():
        if base not in set(df.protocol) or p not in set(df.protocol):
            continue                      # no differing pair -> the protocol equals its base: skip it
        have = set(map(tuple, df.loc[df.protocol == p, ["problem_key", "model_label"]].drop_duplicates().values))
        b = df[df.protocol == base]
        keep = [tuple(v) not in have for v in b[["problem_key", "model_label"]].values]
        if any(keep):
            extra.append(b[keep].assign(protocol=p))
    if extra:
        df = pd.concat([df, *extra], ignore_index=True)
    return df


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def tau_b(x, y):
    """Kendall tau_b along the last axis (vectorised; equals scipy's tau_b)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = x.shape[-1]
    iu = np.triu_indices(n, 1)
    dx = np.sign(x[..., :, None] - x[..., None, :])[..., iu[0], iu[1]]
    dy = np.sign(y[..., :, None] - y[..., None, :])[..., iu[0], iu[1]]
    n0 = len(iu[0])
    n1 = (dx == 0).sum(-1)
    n2 = (dy == 0).sum(-1)
    den = np.sqrt((n0 - n1) * (n0 - n2).astype(float))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (dx * dy).sum(-1) / den, np.nan)


def rank_last(a):
    """Average ranks along the last axis (inf allowed; ties share the mean rank)."""
    from scipy.stats import rankdata
    return rankdata(a, method="average", axis=-1)


def tol_rank(a, rtol=0.0, atol=0.0):
    """Average ranks along the last axis with a resolution tolerance: after
    sorting, consecutive values with v[i] <= v[i-1] (1 + rtol) + atol fall in
    the same tie group (single linkage); a group shares its mean rank.  inf
    ties with inf, never with a finite value.  rtol = atol = 0 equals
    scipy's rankdata(method='average').  atol may broadcast against the
    leading axes (e.g. one value per protocol)."""
    a = np.asarray(a, float)
    R = a.shape[-1]
    order = np.argsort(a, axis=-1, kind="stable")
    srt = np.take_along_axis(a, order, -1)
    with np.errstate(invalid="ignore"):
        gap = srt[..., 1:] > srt[..., :-1] * (1.0 + rtol) + atol
    gid = np.concatenate([np.zeros(srt.shape[:-1] + (1,), int), np.cumsum(gap, -1)], -1)
    pos = np.arange(1, R + 1, dtype=float)
    rs = np.zeros_like(srt)
    for g in range(R):
        msk = gid == g
        cnt = msk.sum(-1, keepdims=True)
        mean = (pos * msk).sum(-1, keepdims=True) / np.maximum(cnt, 1)
        rs = np.where(msk, mean, rs)
    out = np.empty_like(rs)
    np.put_along_axis(out, order, rs, -1)
    return out


def build_array(df, protocols, problems, models, seeds, B):
    """A[metric] with shape (P, Q, M, S, B); inf = diverged/failed/missing."""
    P, Q, M, S = len(protocols), len(problems), len(models), len(seeds)
    A = {m: np.full((P, Q, M, S, B), np.inf) for m in METRICS}
    present = np.zeros((P, Q, M), dtype=bool)
    failed = 0
    n_dwell_censored = [0]
    idx_p = {p: i for i, p in enumerate(protocols)}
    idx_q = {q: i for i, q in enumerate(problems)}
    idx_m = {m: i for i, m in enumerate(models)}
    idx_s = {s: i for i, s in enumerate(seeds)}
    sub = df[df.protocol.isin(protocols) & df.problem_key.isin(problems) & df.model_label.isin(models)
             & df.seed.isin(seeds)].rename(columns={"t_eps_0.001": "t3"})
    for r in sub.itertuples(index=False):
        p, q, m, s = idx_p[r.protocol], idx_q[r.problem_key], idx_m[r.model_label], idx_s[r.seed]
        b = int(r.x0_index)
        present[p, q, m] = True
        st = r.status
        if st not in ("ok", "diverged"):
            failed += 1
            continue
        t3, ms = r.t3, r.mssre
        if st == "ok" and np.isfinite(t3) and t3 > (1.0 - DWELL) * float(r.T):
            n_dwell_censored[0] += 1           # reached only in the last DWELL*T: censored
            t3 = np.inf
        A["t3"][p, q, m, s, b] = t3 if (st == "ok" and np.isfinite(t3)) else np.inf
        A["mssre"][p, q, m, s, b] = max(ms, FLOOR) if (st == "ok" and np.isfinite(ms)) else np.inf
    return A, present, failed, n_dwell_censored[0]


def med(a, axis=None):
    return np.median(a, axis=axis)


def quant(a, q):
    """Percentile along the last axis, 'midpoint' rule, inf-safe (numpy's own
    interpolation forms inf - inf = nan when both neighbours are inf)."""
    a = np.sort(np.asarray(a, float), axis=-1)
    n = a.shape[-1]
    pos = q / 100.0 * (n - 1)
    lo, hi = int(np.floor(pos)), int(np.ceil(pos))
    x, y = a[..., lo], a[..., hi]
    with np.errstate(invalid="ignore"):
        return np.where(np.isinf(x) & np.isinf(y) & (x == y), x, 0.5 * (x + y))


def _selfcheck():
    """tau_b and quant against scipy / numpy on random tied data."""
    from scipy.stats import kendalltau
    rng = np.random.default_rng(0)
    for _ in range(50):
        x = rng.integers(0, 4, 9).astype(float)
        y = rng.integers(0, 4, 9).astype(float)
        ref = kendalltau(x, y).statistic
        got = float(tau_b(x, y))
        assert (np.isnan(ref) and np.isnan(got)) or abs(ref - got) < 1e-12, (x, y, ref, got)
    v = rng.normal(size=(5, 12))
    assert np.allclose(quant(v, 25), np.percentile(v, 25, axis=-1, method="midpoint"))
    w = np.array([1.0, 2.0, np.inf, np.inf])
    assert quant(w, 75) == np.inf and quant(w, 25) == 1.5
    from scipy.stats import rankdata
    z = rng.integers(0, 5, (20, 9)).astype(float)
    z[z == 4] = np.inf
    assert np.allclose(tol_rank(z), rankdata(z, method="average", axis=-1))
    assert np.allclose(tol_rank(np.array([1.0, 1.05, 3.0, np.inf, np.inf]), 0.1), [1.5, 1.5, 3, 4.5, 4.5])


def bootstrap_medians(A, n_boot, seed):
    """Paired hierarchical bootstrap medians: array (n_boot, P, Q, M) per metric."""
    rng = np.random.default_rng(seed)
    any_m = next(iter(A.values()))
    S, B = any_m.shape[-2:]
    out = {m: np.empty((n_boot,) + any_m.shape[:3]) for m in A}
    for i in range(n_boot):
        si = rng.integers(0, S, S)
        bi = rng.integers(0, B, (S, B))
        for m, arr in A.items():
            smp = arr[..., si[:, None], bi]                     # (P, Q, M, S, B)
            out[m][i] = np.median(smp.reshape(smp.shape[:3] + (-1,)), axis=-1)
    return out


# ---------------------------------------------------------------------------
# analysis core
# ---------------------------------------------------------------------------
def analyze(df, cal_all, figdir, tabdir, results):
    cal = {(q, l): c for (q, l, u), c in cal_all.items() if math.isclose(u, 1.0)}   # original unit
    pan_all = C.panel()
    labels_present = set(df.model_label.unique())
    pan = [e for e in pan_all if e["label"] in labels_present]
    ent = {e["label"]: e for e in pan}
    problems = [p["label"] for p in C.PROBLEMS if p["label"] in set(df.problem_key)]
    seeds = sorted(int(s) for s in df.seed.unique())
    B = int(df.B.max())
    protocols = [p for p in MAIN + SENS + UNITP if p in set(df.protocol)]
    units = sorted({C.split_protocol(p)[1] for p in protocols} - {1.0})
    models = [e["label"] for e in pan]

    A, present, n_failed, n_dwell = build_array(df, protocols, problems, models, seeds, B)
    ip = {p: i for i, p in enumerate(protocols)}
    # t3 tie tolerance: TIE_ATOL_SAMPLES recording intervals of each protocol
    dtrec = (df.assign(_d=df.tau * df.record_every).groupby("protocol")._d.max())
    ATOL = {"t3": np.array([TIE_ATOL_SAMPLES * float(dtrec.get(p, 0.0)) for p in protocols])[:, None, None],
            "mssre": np.zeros((len(protocols), 1, 1))}

    # optional model (sptc_an_foznn): ranked only if stable in P0 (<= 50 % diverged on every problem)
    ranked = []
    excluded = {}
    for k, e in enumerate(pan):
        if not e.get("optional"):
            ranked.append(e["label"])
            continue
        frac = [float(np.mean(~np.isfinite(A["mssre"][ip["P0"], q, k]))) for q in range(len(problems))]
        if max(frac) <= 0.5:
            ranked.append(e["label"])
        else:
            excluded[e["label"]] = f"diverged in P0 on {max(frac):.0%} of trajectories"
    rk = [models.index(m) for m in ranked]
    newest = max((ent[m] for m in ranked), key=lambda e: e["year"])["label"]

    # medians, IQR, ranks
    MED = {m: med(A[m].reshape(A[m].shape[:3] + (-1,)), axis=-1) for m in METRICS}   # (P, Q, M)
    Q25 = {m: quant(A[m].reshape(A[m].shape[:3] + (-1,)), 25) for m in METRICS}   # midpoint: inf-safe
    Q75 = {m: quant(A[m].reshape(A[m].shape[:3] + (-1,)), 75) for m in METRICS}
    RANK = {m: tol_rank(MED[m][:, :, rk], TIE_RTOL[m], ATOL[m]) for m in METRICS}        # (P, Q, R)

    # bootstrap (seed clusters, then initial states within a seed; paired across protocols)
    Asub = {m: A[m][:, :, rk] for m in METRICS}
    BM = bootstrap_medians(Asub, N_BOOT, BOOT_SEED)
    BR = {m: tol_rank(BM[m], TIE_RTOL[m], ATOL[m]) for m in METRICS}                     # (n, P, Q, R)

    def _pct(bs):
        ok = np.isfinite(bs)
        return ((float(np.percentile(bs[ok], 2.5)), float(np.percentile(bs[ok], 97.5)))
                if ok.any() else (float("nan"), float("nan")))

    def kendall_pair(pa, pb):
        """tau_b between the rankings of protocols pa and pb per (problem,
        metric), pooled over problems per metric and over everything (mean of
        the defined cells), with paired bootstrap percentile CIs."""
        i0, k = ip[pa], ip[pb]
        rec = {"ref": pa}
        pooled_pt, pooled_bs = [], []
        for qi, q in enumerate(problems):
            rec[q] = {}
            for m in METRICS:
                t = float(tau_b(RANK[m][i0, qi], RANK[m][k, qi]))
                bs = tau_b(BR[m][:, i0, qi], BR[m][:, k, qi])
                lo, hi = _pct(bs)
                rec[q][m] = {"tau": t, "lo": lo, "hi": hi, "n_valid": int(np.isfinite(bs).sum())}
                pooled_pt.append(t)
                pooled_bs.append(bs)
        lo, hi = _pct(np.nanmean(np.vstack(pooled_bs), axis=0))
        rec["pooled"] = {"tau": float(np.nanmean(pooled_pt)), "lo": lo, "hi": hi,
                         "n_cells": int(np.isfinite(pooled_pt).sum())}
        for m in METRICS:   # pooled over problems, per metric
            vals = [rec[q][m]["tau"] for q in problems]
            bsm = np.nanmean(np.vstack([tau_b(BR[m][:, i0, qi], BR[m][:, k, qi]) for qi in range(len(problems))]),
                             axis=0)
            lo, hi = _pct(bsm)
            rec[f"pooled_{m}"] = {"tau": float(np.nanmean(vals)), "lo": lo, "hi": hi}
        return rec

    # each protocol against the as-published protocol of its own unit
    kendall = {p: kendall_pair(ref_of(p), p) for p in protocols if not is_p0(p) and ref_of(p) in ip}
    # unit invariance: the same protocol in the original unit vs. in unit u
    kendall_unit = {}
    for b in MAIN:
        for u in units:
            pu = f"{b}{C.unit_suffix(u)}"
            if b in ip and pu in ip:
                kendall_unit[pu] = kendall_pair(b, pu)

    # who is best; newest best; P0-best stays best; claim survival -- for EVERY protocol
    def best_set(p, qi, m):
        r = RANK[m][ip[p], qi]
        return {ranked[j] for j in np.flatnonzero(r == r.min())}

    def sub_rank(p, qi, m, idx):
        return tol_rank(MED[m][ip[p], qi, idx], TIE_RTOL[m], ATOL[m][ip[p], 0, 0])

    def informative(p, qi, m, idx=None):
        """A cell counts only if the compared models do not all tie (tolerance-aware)."""
        r = RANK[m][ip[p], qi] if idx is None else sub_rank(p, qi, m, idx)
        return not np.all(r == r[0])

    newest_best = {p: 0 for p in protocols}
    n_inform = {p: 0 for p in protocols}
    for p in protocols:
        for qi in range(len(problems)):
            for m in METRICS:
                if not informative(p, qi, m):
                    continue
                n_inform[p] += 1
                if newest in best_set(p, qi, m):
                    newest_best[p] += 1
    stays_all = {}               # protocol -> {cell: True/False/None}; reference = own-unit P0
    for p in protocols:
        if is_p0(p) or ref_of(p) not in ip:
            continue
        stays_all[p] = {}
        for qi, q in enumerate(problems):
            for m in METRICS:
                b0 = best_set(ref_of(p), qi, m)
                stays_all[p][f"{q}|{m}"] = (bool(b0 & best_set(p, qi, m)) if informative(p, qi, m) else None)
    stays = {}
    for qi, q in enumerate(problems):
        for m in METRICS:
            b0 = best_set("P0", qi, m)
            # None = not informative (all models tie in that protocol: no 'best' to keep)
            stays[f"{q}|{m}"] = {"P0_best": sorted(b0),
                                 **{p: stays_all[p][f"{q}|{m}"] for p in COMPARE if p in stays_all}}
    survival = {}
    for lab in ranked:
        y = ent[lab]["year"]
        base = [o for o in ranked if ent[o]["year"] < y]
        if not base:
            continue
        sel = [ranked.index(o) for o in base + [lab]]
        idx = [rk[j] for j in sel]
        row, inf_row = {}, {}
        for p in protocols:
            n = ni = 0
            for qi in range(len(problems)):
                for m in METRICS:
                    if not informative(p, qi, m, idx):
                        continue           # all compared models tie: no claim can be decided
                    ni += 1
                    r_sub = sub_rank(p, qi, m, idx)
                    n += int(r_sub[-1] == r_sub.min())      # ties (within tolerance) count as a win
            row[p], inf_row[p] = n, ni
        survival[lab] = {"baselines": base, "cells_won": row, "cells_informative": inf_row,
                         "n_cells": len(problems) * len(METRICS)}

    # censoring per protocol (everything counts as +inf = worst; nothing is dropped)
    censor = {}
    for p in protocols:
        i = ip[p]
        sub = df[df.protocol == p]
        censor[p] = {"trajectories": int(len(problems) * len(models) * len(seeds) * B),
                     "diverged": int((sub.status == "diverged").sum()),
                     "failed": int((~sub.status.isin(["ok", "diverged"])).sum()),
                     "pairs_not_run": int((~present[i]).sum()),
                     "t3_inf": int(np.isinf(A["t3"][i]).sum()),
                     "mssre_inf": int(np.isinf(A["mssre"][i]).sum())}

    # ------------------------------------------------------------------ outputs
    ctx = dict(pan=pan, ent=ent, ranked=ranked, excluded=excluded, models=models, problems=problems,
               protocols=protocols, units=units, ip=ip, rk=rk, MED=MED, Q25=Q25, Q75=Q75, RANK=RANK,
               kendall=kendall, kendall_unit=kendall_unit, newest=newest, newest_best=newest_best,
               n_inform=n_inform, stays=stays, stays_all=stays_all, survival=survival, cal=cal,
               cal_all=cal_all, seeds=seeds, B=B, n_failed=n_failed, present=present, df=df, n_dwell=n_dwell,
               A=A, censor=censor)
    ctx["window_decay"], ctx["window_level"] = window_decay(ctx, results)
    figs = make_figures(ctx, figdir, results)
    tabs = make_tables(ctx, tabdir)
    summary = make_summary(ctx, results, figs, tabs)
    return summary


def window_decay(ctx, results):
    """Steady-window diagnostic from the stored traces: per trajectory the
    ratio max ||E|| over the last fifth of the steady window / max over its
    first fifth, and the level max ||E|| over that first fifth; medians per
    (protocol, problem, model).  A ratio below TRANSIENT_DECAY (falls > 2x) or
    above TRANSIENT_GROWTH (grows > 2x) above round-off (level >
    TRANSIENT_LEVEL) means the 'MSSRE' there is an unfinished transient
    (typical of gamma = 1 models in P0/P4: linear ZNN at gamma = 1 is still a
    pure exponential at t = 0.75 T), not a steady-state floor.  IEZNN's slow
    overshoot at gamma = 1 can still pass (its envelope is flat near its
    peak); see the notes."""
    df = ctx["df"]
    sweep = Path(results) / "sweep"
    vals, levs = {}, {}
    rows = df[df.protocol.isin(MAIN + UNITP)].drop_duplicates(["config", "run_id"])
    for r in rows.itertuples(index=False):
        if not isinstance(r.traces_file, str):
            continue
        npz = sweep / r.config / "results" / r.traces_file
        if not npz.exists():
            continue
        tr = load_traces(npz, r.run_id)
        if "residual" not in tr:
            continue
        t = np.asarray(tr["t"], float)
        R = np.asarray(tr["residual"], float)
        T = t[-1]
        w0 = (t >= 0.75 * T) & (t <= 0.80 * T)
        w1 = t >= 0.95 * T
        if w0.sum() == 0 or w1.sum() == 0:
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            lev = R[w0].max(0)
            ratio = R[w1].max(0) / lev
        ok = np.isfinite(ratio)
        key = (r.protocol, r.problem_key, r.model_label)
        vals.setdefault(key, []).extend(ratio[ok].tolist())
        levs.setdefault(key, []).extend(lev[ok].tolist())
    return ({k: float(np.median(v)) for k, v in vals.items() if v},
            {k: float(np.median(v)) for k, v in levs.items() if v})


def is_transient(ctx, key):
    """True if the MSSRE window of (protocol, problem, model) is an unfinished transient."""
    wd = ctx["window_decay"].get(key, np.nan)
    lv = ctx["window_level"].get(key, np.nan)
    return bool(np.isfinite(wd) and np.isfinite(lv) and lv > TRANSIENT_LEVEL
                and (wd < TRANSIENT_DECAY or wd > TRANSIENT_GROWTH))


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def _save(fig, name, figdir):
    style.FIG_DIR = Path(figdir)
    return str(style.savefig(fig, name))


def _fmt_rank(r):
    return f"{r:.0f}" if abs(r - round(r)) < 1e-9 else f"{r:.1f}"


PROB_TITLE = {"matrix_inversion": "matrix inversion", "tvqp_eq": "TV-QP"}
METRIC_TITLE = {"t3": r"time to $10^{-3}$", "mssre": "MSSRE"}
# models whose source papers claim faster noise-free convergence than older designs (E4 text);
# IEZNN, SPTC-NT-ZNN and SPTC-AN-FOZNN are noise-tolerance designs (tested in E3, E5 and the robot
# study) and linear ZNN is the common baseline
SPEED_CLAIMS = ["power_sigmoid", "sign_bi_power", "vpznn", "predefined_exp", "fixed_time"]
# matching-level variants of the matched protocols: (protocol, level) in increasing level
LEVELS = {"P1": [("P1_g1", 1.0), ("P1_g3", 3.0), ("P1", 10.0), ("P1_g30", 30.0)],
          "P3": [("P3_r5", 5.0), ("P3", 10.0), ("P3_r30", 30.0)],
          "P2": [("P2_h0.25", 0.25), ("P2", 0.8)]}


def fig_heatmap(ctx, figdir):
    """Ranks of every model under the six main protocols (Supplementary Material figure)."""
    ranked, problems = ctx["ranked"], ctx["problems"]
    prots = [p for p in MAIN if p in ctx["ip"]]
    R = len(ranked)
    cmap = style.CMAP_SEQ.reversed()
    with style.print_rc():
        fig, axes = plt.subplots(len(problems), 2, figsize=(style.TEXTWIDTH, 0.23 * R * len(problems) + 1.0),
                                 squeeze=False, constrained_layout=True)
        fig.get_layout_engine().set(h_pad=0.06, w_pad=0.04, hspace=0.04, wspace=0.03)
        for qi, q in enumerate(problems):
            for mi, m in enumerate(METRICS):
                ax = axes[qi, mi]
                mat = np.array([ctx["RANK"][m][ctx["ip"][p], qi] for p in prots]).T     # (R, P)
                medm = np.array([ctx["MED"][m][ctx["ip"][p], qi, ctx["rk"]] for p in prots]).T
                ax.imshow(mat, cmap=cmap, vmin=1, vmax=R, aspect="auto")
                for i in range(R):
                    for j in range(len(prots)):
                        txt = _fmt_rank(mat[i, j]) + ("\u2020" if not np.isfinite(medm[i, j]) else "")
                        ax.text(j, i, txt, ha="center", va="center", fontsize=7.5,
                                color="white" if mat[i, j] <= (R + 1) / 2 else style.INK)
                ax.set_xticks(range(len(prots)))
                ax.set_xticklabels(prots)
                ax.set_yticks(range(R))
                ax.set_yticklabels([ctx["ent"][l]["display"] for l in ranked] if mi == 0 else [])
                ax.set_title(f"{PROB_TITLE.get(q, q)}, rank by {METRIC_TITLE[m]}")
                ax.grid(False)
                for s in ax.spines.values():
                    s.set_visible(False)
                ax.tick_params(length=0)
        return _save(fig, "fig_E4_rank_heatmap", figdir)


def claim_counts(ctx, protocols, models=SPEED_CLAIMS):
    """(cells won, informative cells) of the claim-survival test summed over `models` and
    `protocols` (each model against all OLDER panel models; ties within tolerance count as won)."""
    sv = ctx["survival"]
    won = sum(int(sv[l]["cells_won"][p]) for l in models if l in sv for p in protocols)
    inf = sum(int(sv[l]["cells_informative"][p]) for l in models if l in sv for p in protocols)
    return won, inf


def _model_style(ctx, lab):
    """Entity style (colour follows the entity).  One exception in fig_E4_main: power-sigmoid
    (act slot 1, orange) is drawn in purple (slot 6), because orange and the red of predefined-
    time F2 (act slot 7), drawn in the same panels, are too close (normal-vision dE 7.1 in the
    palette validator); power-sigmoid appears in no other paper figure."""
    e = ctx["ent"][lab]
    st = keys.style_in_group(e["group"], e["gkey"])
    if lab == "power_sigmoid":
        st = {**st, "color": style.CAT[6]}
    return st


def _axes_in(fig, H, left, bottom, width, height, **kw):
    """Axes placed in inches (figure width style.TEXTWIDTH, height H)."""
    W = style.TEXTWIDTH
    return fig.add_axes((left / W, bottom / H, width / W, height / H), **kw)


def fig_main(ctx, figdir):
    """Main-text E4 figure: (a) ranks under P0 and the matched protocols P1-P3 at their main levels
    (the five speed claims and linear ZNN drawn; ranks are among all nine models), (b) claim survival per
    protocol at the main levels, (c) claim
    survival and pooled Kendall tau_b with P0 against the matching level of P1, P3 and P2."""
    ranked, problems = ctx["ranked"], ctx["problems"]
    prots = [p for p in ("P0", "P1", "P2", "P3") if p in ctx["ip"]]
    claims = [l for l in SPEED_CLAIMS if l in ranked]
    claims.sort(key=lambda l: ctx["ent"][l]["year"])
    drawn = ["linear"] + claims                 # the noise-tolerance designs are ranked but not drawn (review 26 Oct)
    cells = [(q, m) for q in problems for m in METRICS]
    light = "#c4c3bf"
    W, H = style.TEXTWIDTH, 5.08
    with style.print_rc():
        fig = plt.figure(figsize=(W, H))
        # ---- (a) ranks, one panel per (problem, metric) ----------------------------------------
        la, ga, ra = 0.40, 0.10, 0.04
        wa = (W - la - ra - ga * (len(cells) - 1)) / len(cells)
        ya, ha = 3.10, 1.55
        axa = [_axes_in(fig, H, la + k * (wa + ga), ya, wa, ha) for k in range(len(cells))]
        for k, (q, m) in enumerate(cells):
            ax = axa[k]
            qi = problems.index(q)
            rk = {l: np.array([ctx["RANK"][m][ctx["ip"][p], qi, ranked.index(l)] for p in prots]) for l in ranked}
            off = {l: np.zeros(len(prots)) for l in ranked}
            for j in range(len(prots)):
                groups = {}
                for l in drawn:                     # offsets only among the drawn series
                    groups.setdefault(round(float(rk[l][j]), 6), []).append(l)
                for g in groups.values():
                    for i, l in enumerate(g):
                        off[l][j] = (i - (len(g) - 1) / 2) * 0.17
            x = np.arange(len(prots))
            ax.plot(x, rk["linear"] + off["linear"], color=style.INK2, lw=1.1, marker="o", ms=3.4,
                    markerfacecolor="white", zorder=2)
            for l in claims:
                st = _model_style(ctx, l)
                ax.plot(x, rk[l] + off[l], color=st["color"], marker=st["marker"], linestyle=st["linestyle"],
                        lw=1.2, ms=4.2, zorder=3)
            ax.set_xticks(x)
            ax.set_xticklabels(prots)
            ax.set_xlim(-0.35, len(prots) - 0.65)
            ax.set_ylim(len(ranked) + 0.6, 0.4)
            ax.set_yticks(range(1, len(ranked) + 1))
            ax.grid(axis="x", visible=False)
            ax.set_title(f"{PROB_TITLE.get(q, q)}\n{METRIC_TITLE[m]}", fontsize=8, linespacing=1.15)
            if k == 0:
                ax.set_ylabel("rank (1 = best)")
            else:
                ax.tick_params(labelleft=False)
        fig.text(0.02 / W, (H - 0.05) / H, "(a)", fontweight="bold", fontsize=9, va="top", ha="left")
        hd = []
        for l in claims:
            st = _model_style(ctx, l)
            dsp, yr = ctx["ent"][l]["display"], ctx["ent"][l]["year"]
            lab = f"{dsp[:-1]}, {yr})" if dsp.endswith(")") else f"{dsp} ({yr})"
            hd.append(Line2D([], [], color=st["color"], marker=st["marker"], linestyle=st["linestyle"], lw=1.2,
                             ms=4.2, label=lab))
        hd.append(Line2D([], [], color=style.INK2, lw=1.1, marker="o", ms=3.4, markerfacecolor="white",
                         label="linear ZNN (baseline)"))
        fig.legend(handles=hd, loc="upper left", bbox_to_anchor=(la / W, (ya - 0.25) / H), ncol=3,
                   frameon=False, fontsize=7.5, columnspacing=1.6, handlelength=2.4, borderaxespad=0.0)
        # ---- (b) claim survival per protocol at the main levels -------------------------------
        yb, hb = 0.42, 1.60
        axb = _axes_in(fig, H, la, yb, 1.80, hb)
        bars = [(p, claim_counts(ctx, [p])) for p in prots]
        bars.append(("P1–P3", claim_counts(ctx, [p for p in prots if p != "P0"])))
        xb = np.arange(len(bars), dtype=float)
        xb[-1] += 0.35
        for i, (lab, (w, n)) in enumerate(bars):
            head = i == 0 or i == len(bars) - 1
            frac = w / n if n else np.nan
            axb.bar(xb[i], frac, width=0.62, color=style.INK2 if head else light, zorder=2)
            axb.text(xb[i], frac + 0.02, f"{w}/{n}", ha="center", va="bottom", fontsize=7.5, color=style.INK)
        axb.axvline((xb[-2] + xb[-1]) / 2, color=style.GREY, lw=0.6, ls=":", zorder=1)
        axb.set_xticks(xb)
        axb.set_xticklabels([b[0] for b in bars])
        axb.set_xlim(xb[0] - 0.55, xb[-1] + 0.55)
        axb.set_ylim(0, 1)
        axb.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        axb.set_yticklabels(["0", "", "0.5", "", "1"])
        axb.set_ylabel("fraction of claim cells won")
        axb.grid(axis="x", visible=False)
        axb.set_title("speed claims, main levels", fontsize=8)
        fig.text(0.02 / W, (yb + hb + 0.33) / H, "(b)", fontweight="bold", fontsize=9, va="top", ha="left")
        # ---- (c) survival and rank agreement against the matching level ------------------------
        kd = ctx["kendall"]
        lc = la + 1.80 + 0.62
        wd = 0.66
        wc = W - ra - lc - 0.12 - wd
        hc, gc = 0.74, 0.12
        yc_top = yb + hc + gc
        axc1 = _axes_in(fig, H, lc, yc_top, wc, hc)
        axc2 = _axes_in(fig, H, lc, yb, wc, hc, sharex=axc1)
        axd1 = _axes_in(fig, H, lc + wc + 0.12, yc_top, wd, hc, sharey=axc1)
        axd2 = _axes_in(fig, H, lc + wc + 0.12, yb, wd, hc, sharex=axd1, sharey=axc2)
        w0, n0 = claim_counts(ctx, ["P0"])
        styles = {"P1": (r"P1 ($\gamma^\ast$)", dict(color=style.INK, marker="o", ls="-")),
                  "P3": (r"P3 ($\gamma_{\mathrm{ref}}$)", dict(color=style.INK2, marker="s", ls="--", mfc="white")),
                  "P2": ("P2", dict(color=style.INK2, marker="D", ls=":", mfc="white"))}
        for name, (ax_s, ax_t) in (("P1", (axc1, axc2)), ("P3", (axc1, axc2)), ("P2", (axd1, axd2))):
            lab, sty = styles[name]
            pts = [(p, v) for p, v in LEVELS[name] if p in kd and p in ctx["ip"]]
            if len(pts) < 2:
                continue
            xs = np.array([v for _, v in pts])
            wn = [claim_counts(ctx, [p]) for p, _ in pts]
            surv = np.array([w / n if n else np.nan for w, n in wn])
            ax_s.plot(xs, surv, color=sty["color"], marker=sty["marker"], linestyle=sty["ls"],
                      markerfacecolor=sty.get("mfc", sty["color"]), ms=4.2, lw=1.1, zorder=3)
            # counts at the lowest and highest level (the main level is labelled in panel b)
            for i in (0, len(pts) - 1):
                w, n = wn[i]
                if name == "P3" and i == 0:          # left of the point: P1 runs just above it
                    kw = dict(xytext=(-4, 0), ha="right", va="center")
                elif name == "P3":
                    kw = dict(xytext=(0, -4), ha="center", va="top")
                else:
                    kw = dict(xytext=(0, 4), ha="center", va="bottom")
                ax_s.annotate(f"{w}/{n}", (xs[i], surv[i]), textcoords="offset points", fontsize=7.5,
                              color=style.INK, zorder=4, **kw)
            t = np.array([kd[p]["pooled"]["tau"] for p, _ in pts])
            lo = np.array([kd[p]["pooled"]["lo"] for p, _ in pts])
            hi = np.array([kd[p]["pooled"]["hi"] for p, _ in pts])
            err = np.nan_to_num(np.vstack([np.clip(t - lo, 0, None), np.clip(hi - t, 0, None)]))
            ax_t.errorbar(xs, t, yerr=err, color=sty["color"], marker=sty["marker"], linestyle=sty["ls"],
                          markerfacecolor=sty.get("mfc", sty["color"]), ms=4.2, lw=1.1, elinewidth=0.7,
                          capsize=1.5, label=lab, zorder=3)
        for ax in (axc1, axd1):
            ax.axhline(w0 / n0, color=style.GREY, lw=0.7, ls="--", zorder=1)
            ax.set_ylim(0, 0.75)
            ax.set_yticks([0, 0.25, 0.5])
            ax.tick_params(labelbottom=False)
        axc1.set_yticklabels(["0", "0.25", "0.5"])
        axc1.text(40, w0 / n0 - 0.015, "P0", ha="right", va="top", fontsize=7.5, color=style.INK2)
        for ax in (axc2, axd2):
            ax.axhline(0, color=style.INK2, lw=0.6, zorder=1)
            ax.axhline(1, color=style.GREY, lw=0.7, ls="--", zorder=1)
            ax.set_ylim(-0.55, 1.12)
            ax.set_yticks([-0.5, 0, 0.5, 1])
        for ax in (axc1, axc2, axd1, axd2):
            ax.set_xscale("log")
            ax.minorticks_off()
            ax.grid(axis="x", visible=False)
        for ax, xm in ((axc1, 10.0), (axc2, 10.0), (axd1, 0.8), (axd2, 0.8)):   # main levels (panels a, b)
            ax.axvline(xm, color=style.GREY, lw=0.6, ls=":", zorder=1)
        for ax in (axd1, axd2):
            ax.tick_params(labelleft=False)
        axc2.set_xticks([1, 3, 10, 30])
        axc2.set_xticklabels(["1", "3", "10", "30"])
        axc2.set_xlim(0.7, 45)
        axd2.set_xticks([0.25, 0.8])
        axd2.set_xticklabels(["0.25", "0.8"])
        axd2.set_xlim(0.17, 1.2)
        axc2.set_xlabel(r"matching level $\gamma^\ast$ or $\gamma_{\mathrm{ref}}$")
        axd2.set_xlabel(r"$h/h_{\max}$")
        axc1.set_ylabel("claim cells won")
        axc2.set_ylabel(r"$\tau_b$ with P0")
        axc1.set_title("P1, P3 (dotted: main level)", fontsize=8)
        axd1.set_title("P2", fontsize=8)
        hh, ll = axc2.get_legend_handles_labels()
        leg = axc2.legend(hh, ll, loc="lower left", frameon=True, handlelength=2.4, fontsize=7.5, borderaxespad=0.2,
                          framealpha=1.0, facecolor="white", edgecolor="none", borderpad=0.3)
        leg.set_zorder(5)  # empty corner of the tau_b panel (in the top panel it covered a count)
        fig.text((lc - 0.6) / W, (yb + hb + 0.33) / H, "(c)", fontweight="bold", fontsize=9, va="top",
                 ha="left")
        return _save(fig, "fig_E4_main", figdir)


def _offsets(ranks_by_series):
    """Small vertical offsets that separate series tied at the same rank."""
    off = {k: 0.0 for k in ranks_by_series}
    groups = {}
    for k, r in ranks_by_series.items():
        groups.setdefault(round(r, 6), []).append(k)
    for g in groups.values():
        if len(g) > 1:
            for i, k in enumerate(g):
                off[k] = (i - (len(g) - 1) / 2) * 0.16
    return off


def fig_bump(ctx, metric, figdir):
    ranked, problems = ctx["ranked"], ctx["problems"]
    prots = [p for p in MAIN if p in ctx["ip"]]
    R = len(ranked)
    groups = [("act", "activation family"), ("model", "model family")]
    fig, axes = plt.subplots(len(problems), 2, figsize=(style.DOUBLE, 2.25 * len(problems) + 0.5),
                             squeeze=False, sharey=True)
    for qi, q in enumerate(problems):
        ranks = {lab: np.array([ctx["RANK"][metric][ctx["ip"][p], qi, ranked.index(lab)] for p in prots])
                 for lab in ranked}
        for gi, (grp, gname) in enumerate(groups):
            ax = axes[qi, gi]
            members = [l for l in ranked if ctx["ent"][l]["group"] == grp or (grp == "model" and l == "linear")]
            offs = [_offsets({l: ranks[l][j] for l in members}) for j in range(len(prots))]
            for lab in members:
                gkey = ctx["ent"][lab].get("mkey", ctx["ent"][lab]["gkey"]) if grp == "model" else ctx["ent"][lab]["gkey"]
                st = keys.style_in_group(grp, gkey)
                y = ranks[lab] + np.array([offs[j][lab] for j in range(len(prots))])
                ax.plot(range(len(prots)), y, label=ctx["ent"][lab]["display"], markersize=4.5, **st)
            ax.set_xticks(range(len(prots)))
            ax.set_xticklabels(prots)
            ax.set_ylim(R + 0.6, 0.4)
            ax.set_yticks(range(1, R + 1))
            if gi == 0:
                ax.set_ylabel(f"rank by {METRIC_TEX[metric]}")
            ax.set_title(f"{PROB_SHORT.get(q, q)} -- {gname} (ranks among all {R})")
            if qi == len(problems) - 1:
                ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=6.2)
    fig.tight_layout()
    return _save(fig, f"fig_E4_bump_{metric}", figdir)


def fig_kendall(ctx, figdir):
    problems = ctx["problems"]
    prots = [p for p in COMPARE if p in ctx["kendall"]]
    panels = problems + ["pooled"]
    fig, axes = plt.subplots(1, len(panels), figsize=(style.DOUBLE, 2.3), sharey=True)
    w = 0.36
    fills = {"t3": dict(color=style.INK2, edgecolor=style.INK2),
             "mssre": dict(color="white", edgecolor=style.INK2, hatch="////")}
    for ai, pnl in enumerate(panels):
        ax = axes[ai]
        for mi, m in enumerate(METRICS):
            x = np.arange(len(prots)) + (mi - 0.5) * w
            if pnl == "pooled":
                recs = [ctx["kendall"][p][f"pooled_{m}"] for p in prots]
            else:
                recs = [ctx["kendall"][p][pnl][m] for p in prots]
            t = np.array([r["tau"] for r in recs])
            lo = np.array([r["lo"] for r in recs])
            hi = np.array([r["hi"] for r in recs])
            ok = np.isfinite(t)
            ax.bar(x[ok], t[ok], width=w, linewidth=0.6, label=METRIC_LONG[m], **fills[m])
            err = np.vstack([np.clip(t - lo, 0, None), np.clip(hi - t, 0, None)])
            ax.errorbar(x[ok], t[ok], yerr=np.nan_to_num(err[:, ok]), fmt="none", ecolor=style.INK,
                        elinewidth=0.7, capsize=1.6)
            for xx in x[~ok]:   # undefined: one ranking is constant (all models tied)
                ax.text(xx, 0.04, "n/a", rotation=90, ha="center", va="bottom", fontsize=5.5, color=style.INK2)
        ax.axhline(0, color=style.INK2, linewidth=0.6)
        ax.axhline(1, color=style.GREY, linewidth=0.6, linestyle="--")
        ax.set_xticks(range(len(prots)))
        ax.set_xticklabels(prots)
        ax.set_ylim(-1.05, 1.1)
        ax.set_title(PROB_SHORT.get(pnl, "pooled (mean of problems)"))
        if ai == 0:
            ax.set_ylabel(r"Kendall $\tau_b$ vs. P0 (95% CI)")
    fig.text(0.0, -0.04, r"n/a: $\tau_b$ undefined because every model ties (e.g. no model reaches $10^{-3}$)",
             fontsize=6, color=style.INK2)
    handles = [Patch(facecolor=fills[m]["color"], edgecolor=style.INK2, hatch=fills[m].get("hatch"),
                     label=METRIC_LONG[m]) for m in METRICS]
    axes[-1].legend(handles=handles, loc="lower right", fontsize=6.2)
    fig.tight_layout()
    return _save(fig, "fig_E4_kendall", figdir)


def gains_table(ctx):
    """gamma used per (protocol, problem, model) from the results."""
    df = ctx["df"]
    g = df.groupby(["protocol", "problem_key", "model_label"]).gamma.first()
    return g


def fig_gains(ctx, figdir):
    problems, models = ctx["problems"], ctx["models"]
    prots = [p for p in MAIN if p in ctx["ip"]]
    g = gains_table(ctx)
    fig, axes = plt.subplots(1, len(problems), figsize=(style.DOUBLE, 2.6), sharey=True, squeeze=False)
    for qi, q in enumerate(problems):
        ax = axes[0, qi]
        for pi, p in enumerate(prots):
            y = [g.get((p, q, l), np.nan) for l in models]
            x = np.arange(len(models)) + (pi - (len(prots) - 1) / 2) * 0.1
            ax.plot(x, y, linestyle="none", marker=style.MARKERS[pi], color=style.CAT[pi], label=p,
                    markersize=4)
        ax.set_yscale("log")
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([ctx["ent"][l]["display"] for l in models], rotation=40, ha="right")
        ax.set_title(PROB_SHORT.get(q, q))
        if qi == 0:
            ax.set_ylabel(r"design gain $\gamma$ used")
    axes[0, -1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=6.2, title="protocol",
                       title_fontsize=6.5)
    fig.tight_layout()
    return _save(fig, "fig_E4_gains", figdir)


def fig_sensitivity(ctx, figdir):
    problems = ctx["problems"]
    k = ctx["kendall"]
    series = [("P1", [("P1_g1", 1.0), ("P1_g3", 3.0), ("P1", 10.0), ("P1_g30", 30.0)],
               r"target linearised gain $\gamma^\ast$"),
              ("P3", [("P3_r5", 5.0), ("P3", 10.0), ("P3_r30", 30.0)],
               r"budget = peak $\|\dot x\|$ of linear ZNN at $\gamma_{\rm ref}$"),
              ("P2", [("P2_h0.25", 0.25), ("P2", 0.8)], r"step factor $h/h_{\max}$")]
    series = [sr for sr in series if sum(p in k for p, _ in sr[1]) >= 2]
    if not series:
        return None
    fig, axes = plt.subplots(1, len(series), figsize=(style.DOUBLE if len(series) > 2 else style.ONEHALF, 2.3),
                             sharey=True, squeeze=False)
    axes = axes[0]
    for ai, (name, pts, xl) in enumerate(series):
        ax = axes[ai]
        pts = [(p, x) for p, x in pts if p in k]
        if not pts:
            continue
        for qi, q in enumerate(problems):
            for mi, m in enumerate(METRICS):
                xs = np.array([x for _, x in pts]) * (1 + 0.04 * (2 * qi + mi - 1.5))
                t = np.array([k[p][q][m]["tau"] for p, _ in pts])
                lo = np.array([k[p][q][m]["lo"] for p, _ in pts])
                hi = np.array([k[p][q][m]["hi"] for p, _ in pts])
                ax.errorbar(xs, t, yerr=np.nan_to_num(np.vstack([np.clip(t - lo, 0, None), np.clip(hi - t, 0, None)])),
                            color=style.CAT[2 * qi], marker=style.MARKERS[2 * qi + mi],
                            linestyle=["-", "--"][mi], capsize=1.5, elinewidth=0.6,
                            label=f"{PROB_SHORT.get(q, q)}, {METRIC_TEX[m]}")
        ax.set_xscale("log")
        ax.set_xticks([x for _, x in pts])
        ax.set_xticklabels([f"{x:g}" for _, x in pts])
        ax.minorticks_off()
        ax.set_xlabel(xl)
        ax.axhline(1, color=style.GREY, linewidth=0.6, linestyle="--")
        ax.axhline(0, color=style.INK2, linewidth=0.6)
        ax.set_ylim(-1.05, 1.1)
        ax.set_title({"P1": "P1 matching level", "P3": "P3 budget level", "P2": "P2 step factor"}.get(name, name))
        if ai == 0:
            ax.set_ylabel(r"Kendall $\tau_b$ vs. P0")
    axes[0].legend(loc="lower left", fontsize=5.8)
    fig.tight_layout()
    return _save(fig, "fig_E4_sensitivity", figdir)


def fig_traces(ctx, figdir, results):
    """Median residual vs time under P0 and P2 on the first problem."""
    df = ctx["df"]
    q = ctx["problems"][0]
    prots = [p for p in ("P0", "P2") if p in ctx["ip"]]
    groups = [("act", "activation family"), ("model", "model family")]
    fig, axes = plt.subplots(len(prots), 2, figsize=(style.DOUBLE, 2.0 * len(prots) + 0.5), squeeze=False,
                             sharey=True)
    sweep = Path(results) / "sweep"
    for pi, p in enumerate(prots):
        for gi, (grp, gname) in enumerate(groups):
            ax = axes[pi, gi]
            members = [l for l in ctx["ranked"] if ctx["ent"][l]["group"] == grp or (grp == "model" and l == "linear")]
            for lab in members:
                rows = df[(df.protocol == p) & (df.problem_key == q) & (df.model_label == lab)]
                rows = rows.drop_duplicates(["config", "run_id"])
                curves, tgrid = [], None
                for r in rows.itertuples(index=False):
                    if not isinstance(r.traces_file, str):
                        continue
                    npz = sweep / f"E4_{p}" / "results" / r.traces_file
                    if not npz.exists():
                        continue
                    tr = load_traces(npz, r.run_id)
                    if "residual" not in tr:
                        continue
                    tgrid = tr["t"]
                    curves.append(np.asarray(tr["residual"], float))
                if not curves:
                    continue
                n = min(c.shape[0] for c in curves)
                allc = np.hstack([c[:n] for c in curves])
                gkey = ctx["ent"][lab].get("mkey", ctx["ent"][lab]["gkey"]) if grp == "model" else ctx["ent"][lab]["gkey"]
                st = keys.style_in_group(grp, gkey)
                ax.plot(tgrid[:n], np.maximum(np.median(allc, axis=1), FLOOR), label=ctx["ent"][lab]["display"],
                        markevery=max(1, n // 8), markersize=3.5, **st)
            ax.set_yscale("log")
            ax.axhline(1e-3, color=style.GREY, linewidth=0.6, linestyle=":")
            short = {"P0": "as published", "P2": "0.8 x own max admissible h"}.get(p, p)
            ax.set_title(f"{p} ({short}) -- {gname}")
            if gi == 0:
                ax.set_ylabel(r"median $\|E(t)\|_2$")
            if pi == len(prots) - 1:
                ax.set_xlabel("t (s)")
                ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3, fontsize=6.2)
    fig.suptitle(f"{PROB_SHORT.get(q, q)}: residual trajectories (median over seeds x initial states)", fontsize=8)
    fig.tight_layout()
    return _save(fig, "fig_E4_traces", figdir)


def _unit_label(u):
    return {0.1: r"$u=0.1$ ($E/10$)", 1.0: r"$u=1$ (original)", 10.0: r"$u=10$ ($10E$)"}.get(u, f"$u={u:g}$")


def fig_units(ctx, figdir):
    """Unit dependence (checklist item 1).  Left: within-unit re-ranking,
    pooled tau_b(P0_u, Pk_u) for every error unit u.  Right: unit invariance
    of each protocol's ranking, pooled tau_b(Pk, Pk_u)."""
    k, ku, units = ctx["kendall"], ctx["kendall_unit"], ctx["units"]
    if not units:
        return None
    allu = sorted({1.0, *units})
    fig, axes = plt.subplots(1, 2, figsize=(style.DOUBLE, 2.5), sharey=True,
                             gridspec_kw={"width_ratios": [len(COMPARE), len(MAIN)]})
    panels = [(axes[0], COMPARE, allu, lambda b, u: k.get(f"{b}{C.unit_suffix(u)}"),
               r"re-ranking within a unit: $\tau_b(\mathrm{P0}_u, \mathrm{P}k_u)$"),
              (axes[1], MAIN, units, lambda b, u: ku.get(f"{b}{C.unit_suffix(u)}"),
               r"unit invariance: $\tau_b(\mathrm{P}k_{u=1}, \mathrm{P}k_u)$")]
    for ax, prots, us, get, title in panels:
        for si, u in enumerate(us):
            ci = allu.index(u)
            x = np.arange(len(prots)) + (si - (len(us) - 1) / 2) * 0.22
            recs = [get(b, u) for b in prots]
            t = np.array([r["pooled"]["tau"] if r else np.nan for r in recs])
            lo = np.array([r["pooled"]["lo"] if r else np.nan for r in recs])
            hi = np.array([r["pooled"]["hi"] if r else np.nan for r in recs])
            ok = np.isfinite(t)
            err = np.nan_to_num(np.vstack([np.clip(t - lo, 0, None), np.clip(hi - t, 0, None)]))
            ax.errorbar(x[ok], t[ok], yerr=err[:, ok], fmt=style.MARKERS[ci], color=style.CAT[ci],
                        markersize=4.5, capsize=1.6, elinewidth=0.7, linestyle="none", label=_unit_label(u))
        ax.axhline(0, color=style.INK2, linewidth=0.6)
        ax.axhline(1, color=style.GREY, linewidth=0.6, linestyle="--")
        ax.set_xticks(range(len(prots)))
        ax.set_xticklabels(prots)
        ax.set_ylim(-1.05, 1.1)
        ax.set_title(title, fontsize=7.5)
        ax.legend(loc="lower left", fontsize=6.0)
    axes[0].set_ylabel(r"Kendall $\tau_b$ (pooled, 95% CI)")
    fig.text(0.0, -0.05, r"pooled = mean over the defined (problem, metric) cells; error unit $E_u=uE$ "
             "(R_common.ScaledError), residual and " r"$\varepsilon=10^{-3}$" " in the original unit; P1-P3 "
             "re-calibrated per unit", fontsize=6, color=style.INK2)
    fig.tight_layout()
    return _save(fig, "fig_E4_units", figdir)


def make_figures(ctx, figdir, results):
    Path(figdir).mkdir(parents=True, exist_ok=True)
    out = [fig_heatmap(ctx, figdir), fig_main(ctx, figdir), fig_bump(ctx, "t3", figdir),
           fig_bump(ctx, "mssre", figdir), fig_kendall(ctx, figdir), fig_gains(ctx, figdir)]
    if any(p in ctx["kendall"] for p in SENS):
        fs = fig_sensitivity(ctx, figdir)
        if fs:
            out.append(fs)
    out.append(fig_traces(ctx, figdir, results))
    fu = fig_units(ctx, figdir)
    if fu:
        out.append(fu)
    return out


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
def _num(v, fmt="{:.3g}"):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "--"
    if isinstance(v, float) and math.isinf(v):
        return r"$\infty$"
    s = fmt.format(v)
    if "e" in s:
        m, e = s.split("e")
        s = f"${m}\\times10^{{{int(e)}}}$"
    return s


_TEX_ESC = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
            "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def _tex(s):
    """Escape plain text for LaTeX."""
    return "".join(_TEX_ESC.get(ch, ch) for ch in str(s))



def _write(tabdir, name, lines):
    Path(tabdir).mkdir(parents=True, exist_ok=True)
    p = Path(tabdir) / f"{name}.tex"
    p.write_text("\n".join(lines) + "\n")
    return str(p)


def table_params(ctx, tabdir):
    L = [r"\begin{table*}[t]\centering\scriptsize",
         r"\caption{E4 panel and as-published (P0) parameters. ASSUMPTION = value not recorded in the zoo specs.}",
         r"\label{tab:E4_params}",
         r"\begin{tabular}{@{}l>{\raggedright\arraybackslash}p{5.0cm}rr>{\raggedright\arraybackslash}p{8.0cm}@{}}",
         r"\toprule",
         r"model & registry key / parameters & $\gamma_{P0}$ & year & provenance of the P0 values \\", r"\midrule"]
    for e in ctx["pan"]:
        kw = ", ".join(f"{k}={v}" for k, v in e["kwargs"].items())
        L.append(f"{_tex(e['display'])} & \\texttt{{{_tex(e['key'])}}} {_tex(kw)} & {e['p0_gamma']:g} & {e['year']} & "
                 f"{_tex(e['p0_note'])} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return _write(tabdir, "E4_params", L)


def table_calibration(ctx, tabdir):
    cal = ctx["cal"]
    L = [r"\begin{table}[t]\centering\scriptsize",
         r"\caption{E4 calibration on the held-out seeds (original error unit): reference error $e_{\rm ref}$ (RMS entry of $E(x_0,0)$), "
         r"P1 gain ($\gamma\kappa=10$; $\kappa$ = slope at 0 or secant at $e_{\rm ref}$), P2 maximal admissible "
         r"$h=\gamma\tau$ (Euler, $\tau=10^{-3}$; bounded, $\ge 10\times$ contraction, no growth; evaluated at $0.8h_{\max}$), P3 gain (largest $\gamma$ with "
         r"median peak $\|\dot x\|\le U^\ast$). Flags: f = gain floor (not matchable), i = budget infeasible.}",
         r"\label{tab:E4_calibration}",
         r"\begin{tabular}{@{}llrlrrrr@{}}", r"\toprule",
         r"problem & model & $e_{\rm ref}$ & $\kappa$ mode & $\gamma_{P1}$ & $h_{\max}$ & $\gamma_{P2}$ & $\gamma_{P3}$ \\",
         r"\midrule"]
    for q in ctx["problems"]:
        for l in ctx["models"]:
            c = cal.get((q, l))
            if c is None:
                continue
            g1 = c["p1"]["gamma"][f"{C.GAMMA_STAR:g}"]
            f1 = {"floor": "$^f$", "cap": "$^c$"}.get(c["p1"]["flag"][f"{C.GAMMA_STAR:g}"], "")
            g3 = c["p3"]["gamma"][f"{C.GAMMA_STAR:g}"]
            f3 = {"infeasible": "$^i$", "cap": "$^c$"}.get(c["p3"]["flag"][f"{C.GAMMA_STAR:g}"], "")
            hm = c["p2"]["h_max"]
            ge = c["p2"].get("gamma_eval")
            L.append(f"{_tex(PROB_SHORT.get(q, q))} & {_tex(ctx['ent'][l]['display'])} & {_num(c['e_ref'])} & "
                     f"{c['p1']['kappa_mode']} & {_num(g1)}{f1} & {_num(hm)} & {_num(ge)} & {_num(g3)}{f3} \\\\")
        L.append(r"\midrule" if q != ctx["problems"][-1] else r"\bottomrule")
    L += [r"\end{tabular}", r"\end{table}"]
    return _write(tabdir, "E4_calibration", L)


def table_ranks(ctx, tabdir, metric):
    prots = [p for p in MAIN if p in ctx["ip"]]
    probs = ctx["problems"]
    cols = "l" + "r" * (len(prots) * len(probs))
    L = [r"\begin{table*}[t]\centering\scriptsize",
         f"\\caption{{E4 ranks by {METRIC_LONG[metric]} (median over seeds $\\times$ initial states; 1 = best; medians "
         f"within 10\\,\\% (MSSRE) or 1\\,\\% + one recording interval (time to $10^{{-3}}$) tie and share the mean "
         f"rank; $\\dagger$ = median $\\infty$).}}",
         f"\\label{{tab:E4_ranks_{metric}}}", r"\resizebox{\textwidth}{!}{%",
         f"\\begin{{tabular}}{{@{{}}{cols}@{{}}}}", r"\toprule",
         " & " + " & ".join(f"\\multicolumn{{{len(prots)}}}{{c}}{{{_tex(PROB_SHORT.get(q, q))}}}" for q in probs) + r" \\",
         "".join(f"\\cmidrule(lr){{{2 + i * len(prots)}-{1 + (i + 1) * len(prots)}}}" for i in range(len(probs))),
         "model & " + " & ".join(prots * len(probs)) + r" \\", r"\midrule"]
    for j, l in enumerate(ctx["ranked"]):
        cells = []
        for qi in range(len(probs)):
            for p in prots:
                r = ctx["RANK"][metric][ctx["ip"][p], qi, j]
                medv = ctx["MED"][metric][ctx["ip"][p], qi, ctx["rk"][j]]
                cells.append(_fmt_rank(r) + ("$^\\dagger$" if not np.isfinite(medv) else ""))
        L.append(f"{_tex(ctx['ent'][l]['display'])} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    return _write(tabdir, f"E4_ranks_{metric}", L)


def table_medians(ctx, tabdir, q):
    qi = ctx["problems"].index(q)
    prots = [p for p in MAIN if p in ctx["ip"]]
    L = [r"\begin{table*}[t]\centering\scriptsize",
         f"\\caption{{E4 {_tex(PROB_SHORT.get(q, q))}: median [IQR] over seeds $\\times$ initial states "
         f"($\\infty$ = never reached (or reached only in the last 5\\,\\% of the horizon) / diverged; "
         f"-- = not run, e.g. never stable in P2; $^{{t}}$ = transient window: the residual still falls "
         f"by more than $4\\times$ inside the MSSRE window, so the value is not a steady-state floor).}}",
         f"\\label{{tab:E4_medians_{q}}}", r"\resizebox{\textwidth}{!}{%",
         f"\\begin{{tabular}}{{@{{}}l{'r' * len(prots)}@{{}}}}", r"\toprule",
         "model & " + " & ".join(prots) + r" \\"]
    for m in METRICS:
        L += [r"\midrule", f"\\multicolumn{{{len(prots) + 1}}}{{@{{}}l}}{{\\emph{{{METRIC_LONG[m]}}}}} \\\\"]
        for j, l in enumerate(ctx["models"]):
            cells = []
            for p in prots:
                i = ctx["ip"][p]
                if not ctx["present"][i, qi, j]:
                    cells.append("--")
                    continue
                mark = "$^{t}$" if (m == "mssre" and is_transient(ctx, (p, q, l))) else ""
                cells.append(f"{_num(ctx['MED'][m][i, qi, j])}{mark} [{_num(ctx['Q25'][m][i, qi, j])}, "
                             f"{_num(ctx['Q75'][m][i, qi, j])}]")
            L.append(f"{_tex(ctx['ent'][l]['display'])} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    return _write(tabdir, f"E4_medians_{q}", L)


def table_kendall(ctx, tabdir):
    k = ctx["kendall"]
    probs = ctx["problems"]
    prots = [p for p in COMPARE + SENS if p in k]
    L = [r"\begin{table*}[t]\centering\scriptsize",
         r"\caption{E4 Kendall $\tau_b$ between the P0 ranking and each protocol, with paired hierarchical bootstrap "
         r"95\% CIs (2000 resamples over seeds and initial states).}",
         r"\label{tab:E4_kendall}", r"\resizebox{\textwidth}{!}{%",
         f"\\begin{{tabular}}{{@{{}}l{'c' * (2 * len(probs) + 1)}@{{}}}}", r"\toprule",
         "protocol & " + " & ".join(f"{_tex(PROB_SHORT.get(q, q))} {METRIC_TEX[m]}" for q in probs for m in METRICS)
         + r" & pooled \\", r"\midrule"]

    def cell(r):
        return f"{r['tau']:.2f} [{r['lo']:.2f}, {r['hi']:.2f}]" if np.isfinite(r["tau"]) else "n/a"

    for p in prots:
        if SENS and p == SENS[0]:
            L.append(r"\midrule")
        L.append(f"{_tex(p)} & " + " & ".join(cell(k[p][q][m]) for q in probs for m in METRICS)
                 + f" & {cell(k[p]['pooled'])} \\\\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    return _write(tabdir, "E4_kendall", L)


def table_claims(ctx, tabdir):
    prots = [p for p in MAIN if p in ctx["ip"]]
    L = [r"\begin{table}[t]\centering\scriptsize",
         r"\caption{E4 claim survival: (problem, metric) cells won / informative cells, where a model wins if it "
         r"is best (ties included) among itself and the older models of the panel; cells in which all of these "
         r"models tie (e.g. none reaches $10^{-3}$) are not informative.}",
         r"\label{tab:E4_claims}",
         f"\\begin{{tabular}}{{@{{}}lr{'r' * len(prots)}@{{}}}}", r"\toprule",
         "model & year & " + " & ".join(prots) + r" \\", r"\midrule"]
    for l, s in sorted(ctx["survival"].items(), key=lambda kv: ctx["ent"][kv[0]]["year"]):
        L.append(f"{_tex(ctx['ent'][l]['display'])} & {ctx['ent'][l]['year']} & "
                 + " & ".join(f"{s['cells_won'][p]}/{s['cells_informative'][p]}" if p in s["cells_won"] else "--"
                              for p in prots) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return _write(tabdir, "E4_claims", L)


def _kcell(r):
    return f"{r['tau']:.2f} [{r['lo']:.2f}, {r['hi']:.2f}]" if np.isfinite(r["tau"]) else "n/a"


def table_units(ctx, tabdir):
    """Unit dependence: re-ranking within each unit and unit invariance of each protocol."""
    k, ku, probs = ctx["kendall"], ctx["kendall_unit"], ctx["problems"]
    if not ctx["units"]:
        return None
    ncol = 2 * len(probs) + 1
    L = [r"\begin{table*}[t]\centering\scriptsize",
         r"\caption{E4 unit dependence. The error is expressed as $E_u=uE$ (R\_common.ScaledError; residual, "
         r"$\varepsilon$ and MSSRE stay in the original unit; P1--P3 re-calibrated per unit). Top: Kendall $\tau_b$ "
         r"between the P0 ranking and each protocol \emph{within} a unit. Bottom: $\tau_b$ between the ranking of a "
         r"protocol in the original unit and in unit $u$ (1 = unit invariant). 95\% CIs: paired seed-cluster "
         r"bootstrap; descriptive.}",
         r"\label{tab:E4_units}", r"\resizebox{\textwidth}{!}{%",
         f"\\begin{{tabular}}{{@{{}}ll{'c' * ncol}@{{}}}}", r"\toprule",
         "comparison & $u$ & " + " & ".join(f"{_tex(PROB_SHORT.get(q, q))} {METRIC_TEX[m]}" for q in probs for m in METRICS)
         + r" & pooled \\", r"\midrule"]
    for u in sorted({1.0, *ctx["units"]}):
        for b in COMPARE:
            r = k.get(f"{b}{C.unit_suffix(u)}")
            if r is None:
                continue
            L.append(f"P0 vs {b} & {u:g} & " + " & ".join(_kcell(r[q][m]) for q in probs for m in METRICS)
                     + f" & {_kcell(r['pooled'])} \\\\")
        L.append(r"\midrule")
    for b in MAIN:
        for u in ctx["units"]:
            r = ku.get(f"{b}{C.unit_suffix(u)}")
            if r is None:
                continue
            L.append(f"{b}: $u=1$ vs $u={u:g}$ & {u:g} & " + " & ".join(_kcell(r[q][m]) for q in probs for m in METRICS)
                     + f" & {_kcell(r['pooled'])} \\\\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    return _write(tabdir, "E4_units", L)


def headline_rows(ctx):
    """Per protocol: reference, pooled tau_b vs. its own-unit P0, newest-best
    cells, summed claim survival, P0-best-stays-best cells."""
    rows = []
    surv = ctx["survival"]
    for p in ctx["protocols"]:
        kp = ctx["kendall"].get(p)
        st = ctx["stays_all"].get(p, {})
        stv = [v for v in st.values() if v is not None]
        rows.append({"protocol": p, "ref": ref_of(p),
                     "tau": kp["pooled"] if kp else None,
                     "newest_best": ctx["newest_best"][p], "n_inform": ctx["n_inform"][p],
                     "claims_won": int(sum(v["cells_won"].get(p, 0) for v in surv.values())),
                     "claims_inform": int(sum(v["cells_informative"].get(p, 0) for v in surv.values())),
                     "p0best_stays": int(sum(stv)), "p0best_inform": len(stv)})
    return rows


def table_robustness(ctx, tabdir):
    L = [r"\begin{table}[t]\centering\scriptsize",
         r"\caption{E4 robustness of the headline over every protocol, sensitivity configuration and error unit. "
         r"$\tau_b$: pooled Kendall agreement with the P0 ranking of the same unit [95\% CI]. Newest: cells in which "
         r"the newest model is best (won/informative). Claims: cells, summed over models, in which a model is best "
         r"among itself and every older model. P0-best: cells in which a P0-best model stays best.}",
         r"\label{tab:E4_robustness}",
         r"\begin{tabular}{@{}llrrr@{}}", r"\toprule",
         r"protocol & $\tau_b$ vs.\ P0 & newest & claims & P0-best \\", r"\midrule"]
    for r in headline_rows(ctx):
        tau = _kcell(r["tau"]) if r["tau"] else "--"
        p0b = f"{r['p0best_stays']}/{r['p0best_inform']}" if r["p0best_inform"] or not is_p0(r["protocol"]) else "--"
        L.append(f"{_tex(r['protocol'])} & {tau} & {r['newest_best']}/{r['n_inform']} & "
                 f"{r['claims_won']}/{r['claims_inform']} & {p0b} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return _write(tabdir, "E4_robustness", L)


def make_tables(ctx, tabdir):
    out = [table_params(ctx, tabdir), table_calibration(ctx, tabdir), table_ranks(ctx, tabdir, "t3"),
           table_ranks(ctx, tabdir, "mssre"), table_kendall(ctx, tabdir), table_claims(ctx, tabdir)]
    out += [table_medians(ctx, tabdir, q) for q in ctx["problems"]]
    out += [t for t in (table_units(ctx, tabdir), table_robustness(ctx, tabdir)) if t]
    return out


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
        return None if math.isnan(v) else ("inf" if math.isinf(v) else v)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def timing_and_cost(ctx):
    """Measured wall per step (mini or full) and the full-scale cost model."""
    df = ctx["df"].drop_duplicates(["config", "run_id"])          # inherited rows are duplicates: dropped
    ok = df[df.status.isin(["ok", "diverged"])]
    wps = ok.groupby(["disc_key", "problem_key"]).wall_per_step.median()
    out = {"wall_per_step_median_s": {f"{d}|{q}": float(v) for (d, q), v in wps.items()},
           "peak_rss_mb_max": float(ok.peak_rss_mb.max()), "n_runs": int(len(df)),
           "sum_wall_s": float(ok.wall.sum())}
    # full-scale model: the same configs and (problem, model) pairs as measured,
    # specs scaled to the full-scale seed count, steps from the full scale (RK4
    # configs run at tau_ct, Euler configs keep their tau), measured wall per
    # step per (discretization, problem) -- conservative: the full scale records
    # every 10th RK4 step only
    F = C.SCALES["full"]
    seed_ratio = len(F["seeds"]) / max(1, len(ctx["seeds"]))
    est, n_full_specs, full_steps, max_spec = 0.0, 0, 0.0, 0.0
    for (cfg, disc, q), g in df.groupby(["config", "disc_key", "problem_key"]):
        n = g.run_id.nunique() * seed_ratio
        tau_full = F["tau_ct"] if disc == "rk4" else float(g.tau.iloc[0])
        steps = F["T"] / tau_full
        w = wps.get((disc, q), np.nan)
        est += n * steps * w
        n_full_specs += n
        full_steps += n * steps
        max_spec = max(max_spec, steps * w)
    cal_all = ctx["cal_all"]
    cal_wall = sum(c["wall"]["total"] for c in cal_all.values()) if cal_all else float("nan")
    sc = next(iter(cal_all.values()))["settings"] if cal_all else None
    cal_full, cal_max = 0.0, 0.0
    if sc and sc["scale"] == "full":
        cal_full = cal_wall
        cal_max = max(c["wall"]["total"] for c in cal_all.values())
    elif sc:
        S = C.SCALES[sc["scale"]]
        ni = S["n_interior"]
        r_p2 = (F["T"] / S["T"]) * (F["p2_grid"] + F["p2_refine"] * ni + 1) / (S["p2_grid"] + S["p2_refine"] * ni + 1)
        rk4 = (F["T"] / F["tau_ct"]) / (S["T"] / S["tau_ct"])
        for c in cal_all.values():
            nb = len(c["p3"]["budgets"])
            r_p3 = rk4 * (F["p3_grid"] + F["p3_refine"] * nb * ni + 2 * nb) / (S["p3_grid"] + S["p3_refine"] * nb * ni + 2 * nb)
            w = c["wall"]["p1"] + c["wall"]["p2"] * r_p2 + c["wall"]["p3"] * r_p3
            cal_full += w
            cal_max = max(cal_max, w)
    else:
        cal_full = float("nan")
    out["full_estimate"] = {"sweep_specs": int(round(n_full_specs)), "sweep_batched_steps": float(full_steps),
                            "sweep_core_hours": est / 3600, "sweep_longest_spec_s": max_spec,
                            "sweep_tasks_approx": int(math.ceil(n_full_specs / F["chunk_size"])),
                            "calib_tasks": len(cal_all), "calib_core_hours": cal_full / 3600,
                            "calib_longest_task_s": cal_max,
                            "total_core_hours": (est + cal_full) / 3600,
                            "measured_calib_wall_s": cal_wall, "seed_ratio": seed_ratio}
    return out


def make_summary(ctx, results, figs, tabs):
    k = ctx["kendall"]
    probs = ctx["problems"]
    ranks = {p: {q: {m: {l: float(ctx["RANK"][m][ctx["ip"][p], qi, j]) for j, l in enumerate(ctx["ranked"])}
                     for m in METRICS} for qi, q in enumerate(probs)} for p in ctx["protocols"]}
    medians = {p: {q: {m: {l: float(ctx["MED"][m][ctx["ip"][p], qi, j]) for j, l in enumerate(ctx["models"])}
                       for m in METRICS} for qi, q in enumerate(probs)} for p in ctx["protocols"]}
    cal_c = {f"{q}|{l}": {"e_ref": c["e_ref"], "p1_gamma": c["p1"]["gamma"][f"{C.GAMMA_STAR:g}"],
                          "p1_mode": c["p1"]["kappa_mode"], "p1_flag": c["p1"]["flag"][f"{C.GAMMA_STAR:g}"],
                          "p2_h_max": c["p2"]["h_max"], "p2_status": c["p2"]["status"],
                          "p2_gamma": c["p2"].get("gamma_eval"),
                          "p3_gamma": c["p3"]["gamma"][f"{C.GAMMA_STAR:g}"],
                          "p3_flag": c["p3"]["flag"][f"{C.GAMMA_STAR:g}"], "U_star": c["p3"]["U_star"],
                          "p3_achieved_ratio": c["p3"]["achieved_ratio"][f"{C.GAMMA_STAR:g}"]}
             for (q, l), c in ctx["cal"].items()}
    # P3 transfer to held-out seeds: median peak / U* on the evaluation seeds
    df = ctx["df"]
    transfer = {}
    for (q, l), c in ctx["cal"].items():
        rows = df[(df.protocol == "P3") & (df.problem_key == q) & (df.model_label == l)]
        if len(rows):
            transfer[f"{q}|{l}"] = float(np.median(rows.peak_xdot) / c["p3"]["U_star"])
    transfer_rel = {}
    for q in probs:
        lin = transfer.get(f"{q}|linear")
        for l in ctx["models"]:
            v = transfer.get(f"{q}|{l}")
            if lin and v is not None and l != "linear":
                transfer_rel[f"{q}|{l}"] = v / lin
    cal_units = {f"{q}|{l}" + C.unit_suffix(u): {
        "unit": u, "e_ref": c["e_ref"], "p1_gamma": c["p1"]["gamma"][f"{C.GAMMA_STAR:g}"],
        "p1_mode": c["p1"]["kappa_mode"], "p1_flag": c["p1"]["flag"][f"{C.GAMMA_STAR:g}"],
        "p1_gamma_pole": c["p1"].get("gamma_pole", {}).get(f"{C.GAMMA_STAR:g}"),
        "p2_h_max": c["p2"]["h_max"], "p2_status": c["p2"]["status"], "p2_gamma": c["p2"].get("gamma_eval"),
        "p3_gamma": c["p3"]["gamma"][f"{C.GAMMA_STAR:g}"], "p3_flag": c["p3"]["flag"][f"{C.GAMMA_STAR:g}"],
        "U_star": c["p3"]["U_star"], "p3_achieved_ratio": c["p3"]["achieved_ratio"][f"{C.GAMMA_STAR:g}"]}
        for (q, l, u), c in sorted(ctx["cal_all"].items(), key=lambda kv: (kv[0][2], kv[0][0], kv[0][1]))}
    tim = timing_and_cost(ctx)
    newest = ctx["newest"]
    status_counts = df.status.value_counts().to_dict()
    summ = {"experiment": "E4", "scale": next(iter(ctx["cal"].values()))["scale"] if ctx["cal"] else None,
            "panel": [e["label"] for e in ctx["pan"]], "ranked_panel": ctx["ranked"], "excluded": ctx["excluded"],
            "newest_model": newest, "problems": probs, "seeds": ctx["seeds"], "B": ctx["B"],
            "protocols": ctx["protocols"], "protocol_info": C.PROTOCOL_INFO, "status_counts": status_counts,
            "n_failed_rows": ctx["n_failed"], "floor_mssre": FLOOR, "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
            "kendall": k, "ranks": ranks, "medians": medians,
            "newest_best_cells": {p: f"{v}/{ctx['n_inform'][p]}" for p, v in ctx["newest_best"].items()},
            "newest_best_note": "won / informative cells (cells where all ranked models tie are excluded)",
            "p0_best_stays_best": ctx["stays"], "claim_survival": ctx["survival"], "calibration": cal_c,
            "p3_transfer_ratio": transfer, "timing": tim, "figures": figs, "tables": tabs,
            "method": {"t3_dwell_frac": DWELL, "n_t3_dwell_censored": ctx["n_dwell"],
                       "tie_rtol": TIE_RTOL, "tie_atol_t3_samples": TIE_ATOL_SAMPLES,
                       "transient_decay_threshold": TRANSIENT_DECAY, "transient_growth_threshold": TRANSIENT_GROWTH,
                       "transient_level": TRANSIENT_LEVEL},
            "diverged_cells": diverged_cells(ctx),
            "window_decay": {f"{p}|{q}|{l}": v for (p, q, l), v in ctx["window_decay"].items()},
            "transient_mssre_cells": [f"{p}|{q}|{l}" for (p, q, l) in sorted(ctx["window_decay"])
                                      if is_transient(ctx, (p, q, l))],
            "unmatched": unmatched(ctx),
            "units": ctx["units"], "kendall_unit": ctx["kendall_unit"], "headline_robustness": headline_rows(ctx),
            "censoring": ctx["censor"], "calibration_units": cal_units,
            "p3_transfer_relative_to_linear": transfer_rel,
            "p0_best_stays_best_all": ctx["stays_all"]}
    summ = _clean(summ)
    Path(results).mkdir(parents=True, exist_ok=True)
    (Path(results) / "summary.json").write_text(json.dumps(summ, indent=1))
    (Path(results) / "SUMMARY.md").write_text(summary_md(ctx, summ))
    return summ


def diverged_cells(ctx):
    """'protocol|problem|model' -> diverged trajectories (of seeds x B)."""
    df = ctx["df"]
    d = df[df.status == "diverged"].groupby(["protocol", "problem_key", "model_label"]).size()
    return {f"{p}|{q}|{l}": int(n) for (p, q, l), n in d.items()}


def unmatched(ctx):
    """Pairs whose matching target was not reachable (P1 floor/cap, P3 infeasible/cap)."""
    out = {}
    for (q, l), c in ctx["cal"].items():
        f1 = c["p1"]["flag"][f"{C.GAMMA_STAR:g}"]
        f3 = c["p3"]["flag"][f"{C.GAMMA_STAR:g}"]
        if f1 != "ok" or f3 != "ok":
            out[f"{q}|{l}"] = {"P1": f1, "P3": f3,
                               "P1_kappa_at_floor": c["p1"].get("kappa_affine_offset"),
                               "P3_effort_ratio": c["p3"]["achieved_ratio"][f"{C.GAMMA_STAR:g}"]}
    return out


def summary_md(ctx, s):
    k = ctx["kendall"]
    probs = ctx["problems"]
    L = [f"# E4 fairness re-ranking -- summary ({s['scale']} scale)", "",
         f"Panel ({len(ctx['ranked'])} ranked): " + ", ".join(ctx["ent"][l]["display"] for l in ctx["ranked"]) + "."]
    if ctx["excluded"]:
        L.append("Excluded from the ranking: " + "; ".join(f"{l} ({why})" for l, why in ctx["excluded"].items()) + ".")
    L += [f"Problems: {', '.join(probs)}; seeds {ctx['seeds']} x B = {ctx['B']} initial states; "
          f"rows by status: {s['status_counts']}.",
          f"Method: time-to-1e-3 censored unless held for the last {DWELL:.0%} of the horizon "
          f"({ctx['n_dwell']} trajectories re-censored); ranks tie when medians differ by less than "
          f"{TIE_RTOL['mssre']:.0%} (MSSRE) or {TIE_RTOL['t3']:.0%} + {TIE_ATOL_SAMPLES} recording intervals "
          "(time-to-1e-3). CIs: paired two-stage cluster bootstrap (seeds = clusters with replacement, then "
          "initial states within a seed) -- sampling variability only, conditional on the design (units, "
          "matching levels and P0 assumptions are covered by the sensitivity configs and unit arms). "
          f"With {len(ctx['seeds'])} seed clusters the percentile CIs are coarse"
          + (" (only 10 distinct seed multisets: read them as descriptive)" if len(ctx["seeds"]) <= 3 else "")
          + ". No p-values are computed; nothing here is an inferential test (no multiplicity control).", "",
          "**Censoring.** Nothing is dropped: diverged or failed trajectories, never-reached time-to-1e-3 and "
          "(problem, model) pairs that a protocol cannot run (e.g. never Euler-stable in P2) all count as +inf "
          "(= worst; inf ties with inf), in the medians, the bootstrap and the ranks. Per protocol "
          "(trajectories: diverged / failed / pairs not run / t3 = inf / MSSRE = inf): "
          + "; ".join(f"{p} {c['diverged']}/{c['failed']}/{c['pairs_not_run']}/{c['t3_inf']}/{c['mssre_inf']} of "
                      f"{c['trajectories']}" for p, c in ctx["censor"].items()) + ".", "",
          "## Kendall tau_b of each protocol's ranking against P0 (as published), 95% bootstrap CI", "",
          "| protocol | " + " | ".join(f"{PROB_SHORT.get(q, q)} {m}" for q in probs for m in METRICS) + " | pooled |",
          "|---|" + "---|" * (len(probs) * len(METRICS) + 1)]

    def cell(r):
        t = r.get("tau") if isinstance(r, dict) else None
        if not isinstance(t, (int, float)) or not np.isfinite(t):
            return "n/a (all tied)"
        lo, hi = r.get("lo"), r.get("hi")
        f = lambda v: f"{v:.2f}" if isinstance(v, (int, float)) and np.isfinite(v) else "n/a"  # noqa: E731
        return f"{t:.2f} [{f(lo)}, {f(hi)}]"

    for p in [x for x in COMPARE + SENS if x in k]:
        cells = [cell(k[p][q][m]) for q in probs for m in METRICS]
        L.append(f"| {p} | " + " | ".join(cells) + f" | {cell(k[p]['pooled'])} |")
    if "P4a" in k and not all(np.isfinite(k["P4a"][q]["t3"]["tau"]) for q in probs):
        L += ["", "P4a pooled = MSSRE cells only (no model reaches 1e-3 at tau = 1e-2, so its t3 tau_b is "
              "undefined); it is not comparable with the 4-cell pooled values of the other rows."]
    # ------------------------------------------------------------- unit arms
    if ctx["units"]:
        ku = ctx["kendall_unit"]
        L += ["", "## Unit dependence (error expressed as E_u = u E; P1-P3 re-calibrated per unit)", "",
              "Re-ranking within a unit, tau_b(P0_u, Pk_u):", "",
              "| protocol | u | " + " | ".join(f"{PROB_SHORT.get(q, q)} {m}" for q in probs for m in METRICS)
              + " | pooled |", "|---|---|" + "---|" * (len(probs) * len(METRICS) + 1)]
        for u in sorted({1.0, *ctx["units"]}):
            for b in COMPARE:
                r = k.get(f"{b}{C.unit_suffix(u)}")
                if r:
                    L.append(f"| {b} | {u:g} | " + " | ".join(cell(r[q][m]) for q in probs for m in METRICS)
                             + f" | {cell(r['pooled'])} |")
        L += ["", "Unit invariance of each protocol's ranking, tau_b(Pk at u = 1, Pk at u) (1 = unit invariant):", "",
              "| protocol | u | " + " | ".join(f"{PROB_SHORT.get(q, q)} {m}" for q in probs for m in METRICS)
              + " | pooled |", "|---|---|" + "---|" * (len(probs) * len(METRICS) + 1)]
        for b in MAIN:
            for u in ctx["units"]:
                r = ku.get(f"{b}{C.unit_suffix(u)}")
                if r:
                    L.append(f"| {b} | {u:g} | " + " | ".join(cell(r[q][m]) for q in probs for m in METRICS)
                             + f" | {cell(r['pooled'])} |")
    L += ["", "## Ranks (1 = best) under each main protocol (original unit)", ""]
    prots = [p for p in MAIN if p in ctx["ip"]]
    for qi, q in enumerate(probs):
        for m in METRICS:
            L += [f"**{q}, {m}**", "", "| model | " + " | ".join(prots) + " |", "|---|" + "---|" * len(prots)]
            for j, l in enumerate(ctx["ranked"]):
                L.append(f"| {ctx['ent'][l]['display']} | " + " | ".join(
                    _fmt_rank(ctx["RANK"][m][ctx["ip"][p], qi, j]) for p in prots) + " |")
            L.append("")
    L += ["## Does the proposed (newest) model stay best?", "",
          f"Newest model: **{ctx['ent'][ctx['newest']]['display']}** ({ctx['ent'][ctx['newest']]['year']}). "
          "Cells (problem x metric) in which it is best, per main protocol (won / informative, i.e. excluding "
          "cells where every model ties): "
          + ", ".join(f"{p}: {ctx['newest_best'][p]}/{ctx['n_inform'][p]}" for p in MAIN if p in ctx["ip"]) + ".",
          "Because it is not best even under P0, 'the newest model stays best' is not an informative headline "
          "for this panel; the claim-survival counts below (each model against all OLDER panel models) are.", ""]
    L.append("P0-best model(s) and whether they remain best: ")
    for cl, d in ctx["stays"].items():
        na = [p for p in COMPARE if p in d and d[p] is None]
        L.append(f"- {cl}: P0 best = {', '.join(d['P0_best'])}; still best in "
                 + ", ".join(p for p in COMPARE if d.get(p)) + (" (none)" if not any(d.get(p) for p in COMPARE) else "")
                 + (f"; not informative (all tied) in {', '.join(na)}" if na else ""))
    L += ["", "Claim survival (cells won among itself + older models / informative cells):", "",
          "| model | year | " + " | ".join(prots) + " |", "|---|---|" + "---|" * len(prots)]
    for l, sv in sorted(ctx["survival"].items(), key=lambda kv: ctx["ent"][kv[0]]["year"]):
        L.append(f"| {ctx['ent'][l]['display']} | {ctx['ent'][l]['year']} | "
                 + " | ".join(f"{sv['cells_won'][p]}/{sv['cells_informative'][p]}" if p in sv["cells_won"] else "--"
                              for p in prots) + " |")
    L += ["", "IEZNN's (and partly SPTC-NT/SPTC-AN's) published advantage is noise rejection; E4 is noise-free, "
          "so their claim-survival rows test only convergence speed/accuracy, not their actual claims (see E5)."]
    # ------------------------------------------------------- robustness table
    L += ["", "## Robustness of the headline over every protocol, sensitivity config and unit arm", "",
          "tau_b vs P0 of the same unit (pooled [95% CI]); newest = cells the newest model wins; claims = summed "
          "claim survival; P0-best = cells in which a P0-best model of the same unit stays best.", "",
          "| protocol | tau_b vs P0 | newest | claims | P0-best |", "|---|---|---|---|---|"]
    for r in s["headline_robustness"]:
        tau = cell(r["tau"]) if r["tau"] else "--"
        p0b = (f"{r['p0best_stays']}/{r['p0best_inform']}" if not is_p0(r["protocol"]) else "--")
        L.append(f"| {r['protocol']} | {tau} | {r['newest_best']}/{r['n_inform']} | "
                 f"{r['claims_won']}/{r['claims_inform']} | {p0b} |")
    tim = s["timing"]
    fe = tim["full_estimate"]
    L += ["", "## Divergences (trajectories of seeds x B that diverged; they count as inf)", ""]
    L += [f"- {kk}: {v}" for kk, v in s["diverged_cells"].items()] or ["- none"]
    L += ["", f"## Transient MSSRE windows (residual envelope falls > {1 / TRANSIENT_DECAY:.0f}x or grows > "
          f"{TRANSIENT_GROWTH:.0f}x inside the window, above {TRANSIENT_LEVEL:g})", "",
          "These MSSRE values are unfinished transients, not steady-state floors: "
          + (", ".join(s["transient_mssre_cells"]) or "none") + "."]
    L += ["", "## Calibration flags", ""]
    for key_, c in s["calibration_units"].items():
        flags = [f for f in (c["p1_flag"], c["p3_flag"]) if f not in ("ok",)]
        if flags or c["p2_status"] != "ok":
            L.append(f"- {key_}: P1 {c['p1_flag']}, P2 {c['p2_status']}, P3 {c['p3_flag']}")
    tr = s["p3_transfer_relative_to_linear"]
    if tr:
        L += ["", "P3 held-out transfer relative to linear ZNN (median eval peak / U*, divided by linear ZNN's "
              "ratio; < 1 = the model under-spends its budget on the evaluation seeds): "
              + ", ".join(f"{kk} {v:.2f}" for kk, v in tr.items()) + "."]
    L += ["", "## Confounds (checklist item 6)", "", confounds_md(ctx, s)]
    L += ["", "## Timing", "",
          f"Median wall per step: {tim['wall_per_step_median_s']}; max peak RSS {tim['peak_rss_mb_max']:.0f} MB; "
          f"{tim['n_runs']} runs. Full-scale estimate ({len(C.SCALES['full']['seeds'])} seeds): "
          f"{fe['sweep_specs']} specs (~{fe['sweep_tasks_approx']} tasks), "
          f"{fe['sweep_core_hours']:.2f} core-h sweep + {fe['calib_core_hours']:.2f} core-h calibration "
          f"({fe['calib_tasks']} tasks, longest ~{fe['calib_longest_task_s'] / 60:.0f} min) "
          f"= {fe['total_core_hours']:.2f} core-h.", "",
          "## Interpretation", "", interpretation(ctx, s)]
    return "\n".join(L) + "\n"


def confounds_md(ctx, s):
    um = s["unmatched"]
    L = ["- **Unit dependence (item 1).** The power-type activations (sign-bi-power, fixed-time, predefined-exp, "
         "power-sigmoid's |e| >= 1 branch, SPTC post-t_c and SPTC-AN's ||E||E term) have no length scale: P0/P4 "
         "(fixed published gains) and P2 (h_max, chattering floor) depend on the unit of E; the P1 secant gain "
         "depends on it through e_ref (unit invariant only for a single pure power). The unit arms quantify it "
         "(tables above); conclusions that do not hold in every arm are unit dependent.",
         "- **Equal nominal gamma is not equal effective gain.** At gamma = 1 predefined-exp's secant gain at "
         "e_ref is ~11 (exp(|e|^p) term), SBP's ~3.4, power-sigmoid's ~e_ref^2: P0 compares very different "
         "effective gains. This is the confound P1/P3 remove, and the reason P0 favours the nonlinear activations.",
         "- **P0 parameters partly assumed.** gamma = 1 is an ASSUMPTION for sign-bi-power, fixed-time, "
         "predefined-exp and IEZNN (also r, k, p and Delta values); IEZNN's lambda = gamma^2/4 is a ZNN-Bench "
         "mapping. 'As published' is therefore 'as published where known, unit design parameter otherwise'.",
         "- **Internal states / time-varying gains in P1.** P1 matches the gain at t = 0+. IEZNN's slowest pole "
         "is then gamma/2 (P1pole matches the pole instead: gamma = 2 gamma*); VP-ZNN's gain keeps growing "
         "(1 + t: 7x at T = 6), SPTC-NT's diverges at t_c, so P1 favours them after t = 0; P3 (effort over the "
         "whole run) is the protocol that charges for that growth.",
         "- **Transients in the steady window.** With gamma = 1 and T = 6 (4 at mini) linear ZNN, IEZNN "
         "(overshoot ~0.13 ||E0|| at t = 4/gamma), VP-ZNN and power-sigmoid are still converging in the last 25 % "
         "of the horizon: their P0/P4 'MSSRE' is a late-transient value (flagged ^t where detected), so MSSRE "
         "ranks in P0/P4 mix speed and accuracy; MSSRE ranks of sign-term models in the continuous-time protocols "
         "are RK4 chattering floors O(tau).",
         "- **Arbitrary start offset.** X(0) ~ U[-2, 2] sets ||E(0)|| and therefore e_ref (P1), the transient peak "
         "||x_dot|| that P3 matches (the peak is at t = 0 for most models) and the time-to-1e-3 of every model; "
         "P3 is a secant-gain match at the initial error in disguise.",
         "- **Information advantages / noise.** E4 is noise-free and every model gets the analytic dE/dt; "
         "SPTC-NT/AN are given the noise bound Delta = 0.5 (ASSUMPTION) and IEZNN's integral term has nothing "
         "to reject, so their published (noise) advantages are not exercised.",
         "- **Step-size definition.** h = gamma * tau (raw product) for every model; SPTC's effective step "
         "gamma tau/(t_c - t) exceeds 2 just before t_c, which sets its P2 h_max."]
    if um:
        L.append("- **Unmatchable pairs.** " + ", ".join(f"{kk} (P1 {v['P1']}, P3 {v['P3']})" for kk, v in um.items())
                 + ": they run with MORE gain/effort than the target, i.e. biased in their favour.")
    return "\n".join(L)


def interpretation(ctx, s):
    k = ctx["kendall"]
    parts = []
    pooled = {p: k[p]["pooled"]["tau"] for p in COMPARE if p in k}
    if pooled:
        lo_p = min(pooled, key=pooled.get)
        parts.append(f"Ranking agreement with the as-published protocol is weakest under {lo_p} "
                     f"(pooled tau_b = {pooled[lo_p]:.2f}) and strongest under "
                     f"{max(pooled, key=pooled.get)} (pooled tau_b = {max(pooled.values()):.2f}).")
    nb = ctx["newest_best"]
    ni = ctx["n_inform"]
    cmp_ = [p for p in COMPARE if p in ctx["ip"]]
    parts.append(f"The newest model ({ctx['ent'][ctx['newest']]['display']}) is best in {nb.get('P0', 0)}/{ni.get('P0', 0)} "
                 f"informative P0 cells and in {sum(nb[p] for p in cmp_)}/{sum(ni[p] for p in cmp_)} informative "
                 "cells of the other main protocols.")
    surv = ctx["survival"]
    if surv:
        p0w = sum(v["cells_won"].get("P0", 0) for v in surv.values())
        p0n = sum(v["cells_informative"].get("P0", 0) for v in surv.values())
        oth = {p: (sum(v["cells_won"].get(p, 0) for v in surv.values()),
                   sum(v["cells_informative"].get(p, 0) for v in surv.values())) for p in cmp_}
        parts.append(f"Summed over the {len(surv)} models that improve on older ones, the 'beats every older model' claim "
                     f"holds in {p0w}/{p0n} informative cells under P0 and in "
                     + ", ".join(f"{w}/{n} under {p}" for p, (w, n) in oth.items()) + ".")
    rows = {r["protocol"]: r for r in s["headline_robustness"]}

    def tau_of(p):
        return k[p]["pooled"]["tau"] if p in k else float("nan")

    allu = [1.0, *ctx["units"]]
    fair = ("P1", "P2", "P3")
    bits = []
    for u in allu:
        sfx = C.unit_suffix(u)
        r0 = rows.get(f"P0{sfx}")
        fw = sum(rows[f"{b}{sfx}"]["claims_won"] for b in fair if f"{b}{sfx}" in rows)
        fn = sum(rows[f"{b}{sfx}"]["claims_inform"] for b in fair if f"{b}{sfx}" in rows)
        if r0:
            bits.append(f"u = {u:g}: tau_b(P0, P1/P2/P3) = " + "/".join(f"{tau_of(b + sfx):.2f}" for b in fair)
                        + f", claims {r0['claims_won']}/{r0['claims_inform']} under P0 vs {fw}/{fn} under P1-P3")
    if len(bits) > 1:
        parts.append("Per error unit: " + "; ".join(bits) + ".")
    ku = ctx["kendall_unit"]
    inv = {u: ku[f"P0{C.unit_suffix(u)}"]["pooled"]["tau"] for u in ctx["units"] if f"P0{C.unit_suffix(u)}" in ku}
    if inv:
        parts.append("The as-published ranking itself is unit dependent (tau_b(P0, P0_u) = "
                     + ", ".join(f"{v:.2f} at u = {u:g}" for u, v in inv.items())
                     + "): a P0 superiority of a power-type activation is a statement about one unit.")
    lev = [(g, f"P1_g{g:g}" if g != C.GAMMA_STAR else "P1") for g in (1.0, 3.0, C.GAMMA_STAR, 30.0)]
    lev = [(g, p) for g, p in lev if p in k]
    if len(lev) > 1:
        parts.append("Matching level: pooled tau_b(P0, P1) = " + ", ".join(f"{tau_of(p):.2f} at gamma* = {g:g}" for g, p in lev)
                     + "; most of the P1 re-ranking appears only once the matched gain exceeds the published level "
                     "(claims " + ", ".join(f"{rows[p]['claims_won']}/{rows[p]['claims_inform']}" for _, p in lev if p in rows)
                     + "), and at mini scale that trend is confounded with RK4 chattering floors that grow with gamma tau.")
    nb_all = {p: (rows[p]["newest_best"], rows[p]["n_inform"]) for p in rows}
    wins = {p: v for p, v in nb_all.items() if v[0] > 0}
    if wins:
        parts.append(f"The newest model wins cells only in " + ", ".join(f"{p} ({w}/{n})" for p, (w, n) in wins.items())
                     + "; " + ("in P1_g1 it cannot be matched (pre-t_c gain >= 31.6 independent of gamma vs target 1), "
                               "so that win is an unmatchability artefact." if "P1_g1" in wins else "see the robustness table."))
    um = unmatched(ctx)
    if um:
        labs = sorted({kk.split("|")[1] for kk in um})
        parts.append("Not matchable: " + ", ".join(ctx["ent"][l]["display"] for l in labs if l in ctx["ent"])
                     + " (P1 gain floor / P3 budget infeasible): in P1/P3 it runs with MORE effective gain or "
                     "effort than the target, so those cells are biased in its favour, not against it.")
    parts.append("Interpret the mini scale with care: it uses RK4 tau = 1e-3, T = 4 s and 3 seeds instead of "
                 "tau = 1e-4, T = 6 s and 10 seeds, so continuous-time protocols still carry discretisation "
                 "artefacts (sign-term chattering, RK4 instability of stiff activations) that the full scale "
                 "reduces." if s.get("scale") == "mini" else "")
    return " ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="results/<scale>/E4 (contains sweep/ and calib/)")
    ap.add_argument("--figdir", required=True)
    ap.add_argument("--tabdir", required=True)
    args = ap.parse_args(argv)
    import warnings
    warnings.filterwarnings("ignore", message="Mean of empty slice")
    warnings.filterwarnings("ignore", message="All-NaN slice encountered")
    _selfcheck()
    style.apply()
    df = load_results(args.results)
    cal = load_calibration(args.results)
    s = analyze(df, cal, args.figdir, args.tabdir, args.results)
    k = s["kendall"]
    print(f"E4 summary -> {Path(args.results) / 'summary.json'}")
    def f3(v):
        return f"{v:.3f}" if isinstance(v, (int, float)) else "n/a"

    for p in COMPARE + SENS + UNITP:
        if p in k:
            print(f"  tau_b({ref_of(p)}, {p:8s}) pooled = {f3(k[p]['pooled']['tau'])} [{f3(k[p]['pooled']['lo'])}, "
                  f"{f3(k[p]['pooled']['hi'])}]")
    for p, r in s.get("kendall_unit", {}).items():
        print(f"  unit invariance tau_b({C.split_protocol(p)[0]}, {p}) pooled = {f3(r['pooled']['tau'])}")
    print(f"  newest model best: {s['newest_best_cells']}")
    for r in s["headline_robustness"]:
        print(f"  headline {r['protocol']:10s} claims {r['claims_won']}/{r['claims_inform']} "
              f"newest {r['newest_best']}/{r['n_inform']} P0-best stays {r['p0best_stays']}/{r['p0best_inform']}")
    print(f"  full-scale estimate: {s['timing']['full_estimate']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
