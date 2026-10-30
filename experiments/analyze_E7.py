#!/usr/bin/env python
"""Analysis of E7 (cost-accuracy Pareto on time-varying matrix inversion).

    python experiments/analyze_E7.py --results results/mini/E7 \
        --figdir figures/mini --tabdir results/tables/mini

Reads ``<results>/accuracy`` (runner sweep of ``experiments/configs/E7.py``)
and ``<results>/timing`` (``experiments/tasks/E7_timing.py``) and writes

* figures ``fig_E7_pareto`` (protocol arm, h = 0.3: MSSRE vs measured
  wall-clock per step, one panel per n, Pareto front), ``fig_E7_pareto_gain``
  (gain-sensitivity arm: each gain-based core method at h = 0.3 and at its
  tuned h, arrows protocol -> tuned), ``fig_E7_pareto_flops`` (protocol arm
  against analytic FLOPs per step / n^3), ``fig_E7_scaling`` (wall-clock per
  step vs n, paired cost ratio to ZNN-Euler, effective GFLOP/s) and, only when
  inverse-free / low-complexity keys ran, ``fig_E7_pareto_extra[_k]``;
* LaTeX booktabs tables ``E7_pareto.tex``, ``E7_headline.tex`` (pre-specified
  contrasts at n_max, 95 % and Bonferroni-simultaneous CIs), ``E7_ratios.tex``
  (ratios to ZNN-Euler, every n), ``E7_scaling.tex``, ``E7_kendall.tex``;
* ``<results>/summary.json``, ``<results>/SUMMARY.md``, ``<results>/E7_stats.csv``,
  ``<results>/E7_contrasts.csv``.

Statistics
----------
* Cost = ``wall_dynamics / steps`` of the B = 1 timing runs.  One timing task
  per (n, problem seed) times every method once per round, in a random order,
  so the unit of the cost data is the (seed, round) block, shared by all
  methods.  Accuracy = MSSRE per trajectory; the unit is (seed, initial
  state), shared by all methods; diverged = inf, error/timeout = missing.
* Point estimates: medians (IQR) over blocks / trajectories.
* Bootstrap (2000 seeded resamples), *paired and two-stage clustered*: per
  replicate, problem seeds are resampled with replacement, then rounds
  (timing) or initial states (accuracy) within each drawn seed; the same
  index draw is applied to every method at that n.  Ratios of medians and
  Pareto-membership frequencies therefore respect the pairing (same problem
  instance, same initial state, same timing round).  With 3 seeds the
  cluster bootstrap is coarse and percentile CIs may under-cover; this is a
  property of the 3-seed protocol and is stated in the notes.
* Pareto front: epsilon-dominance with 5 % tie tolerance on both axes, at the
  medians, plus the bootstrap membership frequency.
* Headline contrasts (``CONTRASTS``) at n_max: 95 % CIs and Bonferroni
  simultaneous CIs (level 1 - 0.05/K over the K contrasts evaluated).
* Kendall tau_b between cost rankings, on the fixed set of 8 core methods
  (comparable across n) and on all ranked methods.
* Scaling exponent of cost vs n: log-log LSQ over all n and over the
  asymptotic tail selected by consecutive local slopes; both reported.
* Settle diagnostic: settle_ratio = max residual over the first half of the
  steady window / max over its second half (runner extra metric
  ``E7_common.settle_metrics``); > 2 flags a transient-contaminated MSSRE.
* Censoring (review 2026-10-18): median MSSRE = +inf marks a censored cell; it
  stays in every table and ranking (+inf = worst, tied) and is drawn in a
  "diverged" strip of the Pareto figures at its own clean cost or at the cost
  of its operation-identical proxy (``E7_common.cost_proxy``).
* Overhead share: per-step time at the smallest n (pure interpreter / call
  overhead) divided by the time at n.
* Machine probe: per-round 128 x 128 dgemm / LU timings of the timing tasks;
  rounds > PROBE_FLAG x the best were timed under matrix-unit contention.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import kendalltau  # noqa: E402

from experiments import keys, style  # noqa: E402
from experiments.tasks import E7_common as C  # noqa: E402
from experiments.tasks.E7_timing import load_results as load_timing  # noqa: E402

FLOOR = 1e-11          # MSSRE below this is at the double-precision round-off floor
SLOPE_TOL = 0.15       # consecutive local slopes within this band = asymptotic range (checklist item 3;
                       # was 0.35, which accepted e.g. 1.3 -> 1.6 as "asymptotic")
EPS_TIE = 0.05         # relative differences below 5 % (cost or MSSRE) are ties for Pareto dominance
PROBE_FLAG = 1.15      # a round whose dgemm probe is > 15 % above the best probe is "contended"

#: pre-specified contrasts (a vs b): cost ratio and MSSRE ratio of medians a / b
CONTRASTS = [
    ("znn_explicit_inverse", "znn", "explicit-inverse vs LU-solve ZNN-Euler (both h = 0.3)"),
    ("uhlig_3_4", "znn", "ZNN Uhlig 3_4 (h_opt) vs ZNN-Euler (h = 0.3)"),
    ("pc_newton_causal", "znn", "causal PC-Newton vs ZNN-Euler at h = 0.3 (gain-confounded)"),
    ("pc_newton_causal", "znn_tuned", "causal PC-Newton vs ZNN-Euler at its tuned h = 1 (fair)"),
    ("znn_explicit_inverse_tuned", "znn_tuned", "explicit-inverse vs LU-solve ZNN-Euler (both h = 1)"),
    ("znn_tuned", "znn", "ZNN-Euler h = 1 vs h = 0.3 (cost ratio = A/A timing control)"),
    ("gnn", "znn", "GNN (h sigma_max^2 ~ 1) vs ZNN-Euler (h = 0.3)"),
    ("gnn_tuned", "gnn", "GNN tuned vs GNN (cost ratio = A/A timing control)"),
    ("newton_tracking_causal", "znn", "causal Newton tracking vs ZNN-Euler (h = 0.3)"),
]
#: series of the gain-sensitivity figure (<= 8): protocol row, then its tuned row
GAIN_SERIES = ["znn", "znn_tuned", "znn_explicit_inverse", "znn_explicit_inverse_tuned",
               "pc_newton_causal", "uhlig_3_4", "gnn", "gnn_tuned"]


def _is_extra(lab: str) -> bool:
    return lab not in C.CORE_LABELS and lab not in C.TUNED_LABELS and not C.is_matched(lab)


def _ranked(lab: str) -> bool:
    """Methods that enter Pareto fronts and rankings (matched-gain GNN rows diverge)."""
    return not C.is_matched(lab)


# ----------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------
def load_accuracy(acc_dir: Path) -> pd.DataFrame:
    from znnbench.runner import aggregate
    df = aggregate(acc_dir)
    if df.empty:
        return df
    # the runner's model label is the key for default kwargs and the E7 label for a model
    # variant with kwargs (C.MODEL_VARIANTS, e.g. lcc_znn_linear); the key alone would merge them
    mlab = df.model_label if "model_label" in df else df.model_key
    df["label"] = [C.label_of(m, d, h) for m, d, h in zip(mlab, df.disc_key, df.h)]
    df["n"] = df["problem_kw_n"].astype(int)
    df["mssre_eff"] = np.where(df.status == "diverged", np.inf,
                               np.where(df.status == "ok", df.mssre, np.nan))
    for col in ("settle_ratio", "mssre_w2"):
        if col not in df:
            df[col] = np.nan
    return df


def _rng(seed, *parts):
    return np.random.default_rng([seed] + [zlib.crc32(str(p).encode()) for p in parts])


def q(x, p):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return math.nan
    return float(np.quantile(x, p, method="nearest" if np.isinf(x).any() else "linear"))


def ci(b, alpha=0.05):
    b = np.asarray(b, dtype=float)
    b = b[~np.isnan(b)]
    if b.size == 0:
        return (math.nan, math.nan)
    return (float(np.quantile(b, alpha / 2, method="nearest")),
            float(np.quantile(b, 1 - alpha / 2, method="nearest")))


def _nanmed(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return float(np.median(x)) if x.size else math.nan


# ----------------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------------
def _matrix(rows_by_label, labels, col_index):
    """(L, U) matrix from {label: [(unit, value), ...]}; NaN where missing."""
    M = np.full((len(labels), len(col_index)), np.nan)
    for i, lab in enumerate(labels):
        for u, v in rows_by_label.get(lab, []):
            M[i, col_index[u]] = v
    return M


def _boot_medians(M, idx_list):
    """(L, nb) bootstrap medians of the rows of M over the drawn columns (NaN-aware)."""
    L = M.shape[0]
    out = np.full((L, len(idx_list)), np.nan)
    if M.shape[1] == 0:
        return out
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for b, idx in enumerate(idx_list):
            out[:, b] = np.nanmedian(M[:, idx], axis=1)
    return out


def build_stats(acc, tim, nb, seed):
    """One row per (n, label): cost and accuracy statistics + paired bootstrap arrays."""
    rows, boots = [], {}
    labels = sorted(set(tim.label) | (set(acc.label) if len(acc) else set()))
    ns = sorted(set(tim.n) | (set(acc.n) if len(acc) else set()))
    # timing runs in which a trajectory diverged stop early: their per-step cost is not comparable
    good = (tim.status == "ok") & (tim.get("n_diverged", 0) == 0)
    t1 = tim[good & (tim.B == 1)]
    t4 = tim[good & (tim.B == 4)]
    for n in ns:
        # ---- cost samples per (seed, round) block, accuracy per (seed, x0) unit
        tn = t1[t1.n == n]
        cost_rows, tot_rows = {}, {}
        for _, r in tn.iterrows():
            vals = json.loads(r.wall_dyn_all)
            tots = json.loads(r.wall_all)
            rounds = json.loads(r.rounds) if isinstance(r.get("rounds", None), str) else list(range(len(vals)))
            cost_rows.setdefault(r.label, []).extend(((int(r.seed), int(k)), float(v)) for k, v in zip(rounds, vals))
            tot_rows.setdefault(r.label, []).extend(((int(r.seed), int(k)), float(v)) for k, v in zip(rounds, tots))
        blocks = sorted({u for v in cost_rows.values() for u, _ in v})
        bidx = {u: i for i, u in enumerate(blocks)}
        an = acc[acc.n == n] if len(acc) else acc
        acc_rows = {}
        for lab, g in (an.groupby("label") if len(an) else []):
            acc_rows[lab] = [((int(sd), int(x)), float(v)) for sd, x, v in zip(g.seed, g.x0_index, g.mssre_eff)]
        units = sorted({u for v in acc_rows.values() for u, _ in v})
        uidx = {u: i for i, u in enumerate(units)}
        Cm = _matrix(cost_rows, labels, bidx)
        Tm = _matrix(tot_rows, labels, bidx)
        Mm = _matrix(acc_rows, labels, uidx)
        # ---- paired two-stage cluster bootstrap (seed, then round / initial state)
        tcl = [[bidx[u] for u in blocks if u[0] == sd] for sd in sorted({u[0] for u in blocks})]
        acl = [[uidx[u] for u in units if u[0] == sd] for sd in sorted({u[0] for u in units})]
        tdraw = C.cluster_indices(tcl, nb, _rng(seed, "timing", n))
        adraw = C.cluster_indices(acl, nb, _rng(seed, "accuracy", n))
        BC = _boot_medians(Cm, tdraw) if tdraw else np.full((len(labels), nb), np.nan)
        BM = _boot_medians(Mm, adraw) if adraw else np.full((len(labels), nb), np.nan)
        for i, lab in enumerate(labels):
            cs = Cm[i][~np.isnan(Cm[i])]
            tot = Tm[i][~np.isnan(Tm[i])]
            ms = Mm[i][~np.isnan(Mm[i])]
            ts = tn[tn.label == lab]
            ts4 = t4[(t4.n == n) & (t4.label == lab)]
            cs4 = np.concatenate([json.loads(v) for v in ts4.wall_dyn_all]) if len(ts4) else np.array([])
            a = an[an.label == lab] if len(an) else an
            tall = tim[(tim.n == n) & (tim.label == lab) & (tim.B == 1)]
            if cs.size == 0 and ms.size == 0 and len(a) == 0 and len(tall) == 0:
                continue
            boots[(n, lab)] = (BC[i], BM[i])
            e2e = (a.drop_duplicates("run_id").wall_per_step.to_numpy(dtype=float)
                   if len(a) and "wall_per_step" in a else np.array([]))
            sr = a.settle_ratio.to_numpy(dtype=float) if len(a) else np.array([])
            srf = sr[np.isfinite(sr)]
            r = {"n": int(n), "label": lab, "core": lab in C.CORE_LABELS, "tuned": lab in C.TUNED_LABELS,
                 "extra": _is_extra(lab),
                 "cost_med": float(np.median(cs)) if cs.size else math.nan,
                 "cost_q25": q(cs, 0.25), "cost_q75": q(cs, 0.75),
                 "cost_ci_lo": ci(BC[i])[0], "cost_ci_hi": ci(BC[i])[1], "cost_nsamp": int(cs.size),
                 "cost_total_med": float(np.median(tot)) if tot.size else math.nan,
                 "cost_B4_med": float(np.median(cs4)) if cs4.size else math.nan,
                 "cost_e2e_med": float(np.nanmedian(e2e)) if e2e.size and np.isfinite(e2e).any() else math.nan,
                 "cpu_over_wall": float(ts.cpu_over_wall.median()) if len(ts) else math.nan,
                 "flops_ideal": C.flops_ideal(lab, int(n), 1),
                 "flops_exec": float(ts.flops_exec.median()) if len(ts) and "flops_exec" in ts else math.nan,
                 "mssre_med": float(np.median(ms)) if ms.size else math.nan,
                 "mssre_q25": q(ms, 0.25), "mssre_q75": q(ms, 0.75),
                 "mssre_ci_lo": ci(BM[i])[0], "mssre_ci_hi": ci(BM[i])[1],
                 "mssre_w2_med": _nanmed(a.mssre_w2) if len(a) else math.nan,
                 "settle_med": float(np.median(srf)) if srf.size else math.nan,
                 "frac_unsettled": float(np.mean(srf > C.SETTLE_TOL)) if srf.size else math.nan,
                 "n_traj": int(ms.size), "n_div": int(np.sum(np.isinf(ms))),
                 "n_fail": int(len(a) - ms.size) if len(a) else 0,
                 # timing censoring: B = 1 timing rows that diverged (no clean per-step cost)
                 "n_timing_rows": int(len(tall)),
                 "n_timing_div": int((tall.get("n_diverged", pd.Series(0, index=tall.index)).fillna(0) > 0).sum())
                 if len(tall) else 0,
                 "sol_err_med": float(a.ss_sol_err.median()) if len(a) else math.nan,
                 # t_eps never reached is inf (metrics.time_to_eps), so the median is censoring-aware
                 "t_eps_1e-3_med": float(a["t_eps_0.001"].median()) if len(a) else math.nan,
                 "peak_rss_mb": float(max(ts.peak_rss_mb.max() if len(ts) else 0,
                                          a.peak_rss_mb.max() if len(a) else 0))}
            src = a if len(a) else ts
            for col in ("model_key", "disc_key", "h", "gamma", "info"):
                r[col] = src[col].iloc[0] if len(src) else None
            r["at_floor"] = bool(np.isfinite(r["mssre_med"]) and r["mssre_med"] < FLOOR)
            r["unsettled"] = bool(np.isfinite(r["settle_med"]) and r["settle_med"] > C.SETTLE_TOL)
            rows.append(r)
    st = pd.DataFrame(rows)
    st = add_costs_py(st)
    with np.errstate(divide="ignore", invalid="ignore"):
        st["gflops_eff"] = np.where(st.cost_med > 0, st.flops_axis / st.cost_med / 1e9, np.nan)
    st = add_overhead(st)
    st = add_censoring(st)
    return st, boots


def add_overhead(st):
    """Per-step interpreter / call-overhead floor of each method = its cost at the smallest
    n of the sweep (n = 4: < 1e3 FLOPs per step, i.e. < 0.01 us of arithmetic at the measured
    GEMM rates, so the time there is pure overhead).  ``overhead_share`` = floor / cost(n)
    (assumes an n-independent overhead; numpy allocation/copies grow ~n^2, so it is an
    estimate); ``arith_share`` = 1 - overhead_share is the part of the per-step time that
    can be attributed to the algorithm's arithmetic at that n."""
    st["floor_cost"] = math.nan
    st["overhead_share"] = math.nan
    if st.empty:
        return st
    n0 = int(st.n.min())
    for lab, g in st.groupby("label"):
        g = g[np.isfinite(g.cost_med)].sort_values("n")
        if g.empty or int(g.n.iloc[0]) != n0:
            continue
        floor = float(g.cost_med.iloc[0])
        st.loc[g.index, "floor_cost"] = floor
        st.loc[g.index, "overhead_share"] = np.minimum(floor / g.cost_med.to_numpy(dtype=float), 1.0)
    st["arith_share"] = 1.0 - st.overhead_share
    return st


def add_censoring(st):
    """Censored cells (median MSSRE = inf: diverged in at least half of the trajectories) keep
    their row; ``cost_plot`` is their clean per-step cost, or, when they have none (not timed,
    or every timing run diverged), the cost of the operation-identical sibling
    ``C.cost_proxy(label)`` at the same n, so that figures can draw them as diverged instead
    of dropping them."""
    st["censored"] = np.isinf(st.mssre_med.to_numpy(dtype=float))
    st["cost_plot"] = st.cost_med
    st["cost_plot_src"] = np.where(np.isfinite(st.cost_med), "own", "")
    for i, r in st[st.censored & ~np.isfinite(st.cost_med)].iterrows():
        p = C.cost_proxy(r.label)
        pr = st[(st.n == r.n) & (st.label == p)] if p else st.iloc[0:0]
        if len(pr) and np.isfinite(pr.cost_med.iloc[0]):
            st.loc[i, "cost_plot"] = float(pr.cost_med.iloc[0])
            st.loc[i, "cost_plot_src"] = f"proxy:{p}"
    return st


COSTS_NOTE = {"costs_py": "absent"}


def add_costs_py(st):
    """Analytic FLOPs per step from ``znnbench/costs.py`` (``step_flops``) when
    that module exists: ``flops_costs_py`` = operations the algorithm needs
    (``skip_unused_dEdt=True``), ``flops_costs_code`` = what the code executes.
    ``flops_axis`` is ``flops_costs_py`` when available, else the idealised
    leading-order count of ``E7_common.flops_ideal``."""
    st["flops_costs_py"] = math.nan
    st["flops_costs_code"] = math.nan
    try:
        from znnbench import costs
    except Exception:  # noqa: BLE001
        COSTS_NOTE["costs_py"] = "absent (secondary cost = idealised leading-order FLOPs of E7_common)"
        costs = None
    fn = getattr(costs, "step_flops", None) if costs is not None else None
    if costs is not None and not callable(fn):
        COSTS_NOTE["costs_py"] = "present but without step_flops; idealised FLOPs used"
    if callable(fn):
        from znnbench import make_discretization, make_model, make_problem
        vals, code, fails = [], [], []
        probs = {}
        for _, r in st.iterrows():
            v = c = math.nan
            try:
                n = int(r.n)
                if n not in probs:
                    probs[n] = make_problem(C.PROBLEM, n=n, kappa=C.KAPPA, freq=C.FREQ, seed=0)
                info = r["info"] if isinstance(r["info"], str) else "analytic"
                mkw = C.MODEL_VARIANTS[r.label][1] if r.label in C.MODEL_VARIANTS else {}
                args = (probs[n], make_model(r.model_key, **mkw), make_discretization(r.disc_key))
                v = float(fn(*args, B=1, info=info)["total"])
                c = float(fn(*args, B=1, info=info, skip_unused_dEdt=False)["total"])
            except Exception as exc:  # noqa: BLE001
                fails.append(f"{r.label}@{int(r.n)}: {exc!r}"[:160])
            vals.append(v)
            code.append(c)
        st["flops_costs_py"] = vals
        st["flops_costs_code"] = code
        COSTS_NOTE["costs_py"] = ("present: znnbench.costs.step_flops(problem, model, disc, B=1, info) 'total'"
                                  + (f"; {len(fails)} failures, e.g. {fails[0]}" if fails else ""))
    st["flops_axis"] = np.where(np.isfinite(st.flops_costs_py), st.flops_costs_py, st.flops_ideal)
    st["flops_axis_src"] = np.where(np.isfinite(st.flops_costs_py), "costs.py", "ideal")
    return st


def pareto_mask(cost, err, eps=EPS_TIE):
    """Non-dominated points with relative tie tolerance ``eps``: q dominates p
    when q is no worse than p up to a factor (1 + eps) on both axes and better
    by more than that factor on at least one (so two methods with the same
    MSSRE to within 5 %, e.g. ZNN-Euler and the explicit-inverse ZNN, are not
    both kept just because one median is lower by 1e-6 relative)."""
    cost, err = np.asarray(cost, float), np.asarray(err, float)
    ok = np.isfinite(cost) & np.isfinite(err)
    f = 1.0 + eps
    m = np.zeros(cost.size, dtype=bool)
    for i in np.flatnonzero(ok):
        no_worse = ok & (cost <= cost[i] * f) & (err <= err[i] * f)
        better = (cost * f < cost[i]) | (err * f < err[i])
        dom = no_worse & better
        dom[i] = False
        m[i] = not dom.any()
    return m


def add_front(st, boots, nb, members, tag):
    """Pareto front at the medians and its paired-bootstrap membership frequency,
    over the methods selected by ``members(label)`` at each n."""
    st[f"pareto{tag}"] = False
    st[f"pareto_freq{tag}"] = math.nan
    for n, g in st.groupby("n"):
        g = g[np.array([members(lab) for lab in g.label], dtype=bool)]
        g = g[np.isfinite(g.cost_med)]
        if g.empty:
            continue
        idx = g.index.to_numpy()
        st.loc[idx, f"pareto{tag}"] = pareto_mask(g.cost_med, g.mssre_med)
        bc = np.stack([boots[(n, lab)][0] for lab in g.label])
        bm = np.stack([boots[(n, lab)][1] for lab in g.label])
        cnt = np.zeros(len(g))
        for b in range(bc.shape[1]):
            cnt += pareto_mask(bc[:, b], bm[:, b])
        st.loc[idx, f"pareto_freq{tag}"] = cnt / bc.shape[1]
    return st


def contrast(st, boots, a, b, n, alpha=0.05):
    """Ratios of medians a/b at n with paired bootstrap CIs (level 1 - alpha)."""
    if (n, a) not in boots or (n, b) not in boots:
        return None
    ra = st[(st.n == n) & (st.label == a)].iloc[0]
    rb = st[(st.n == n) & (st.label == b)].iloc[0]
    (ca, ma), (cb, mb) = boots[(n, a)], boots[(n, b)]
    with np.errstate(divide="ignore", invalid="ignore"):
        return {"n": int(n), "a": a, "b": b,
                "cost_ratio": ra.cost_med / rb.cost_med, "cost_ratio_ci": ci(ca / cb, alpha),
                "flops_ratio": ra.flops_axis / rb.flops_axis,
                "flops_ideal_ratio": ra.flops_ideal / rb.flops_ideal,
                "mssre_ratio": ra.mssre_med / rb.mssre_med, "mssre_ratio_ci": ci(ma / mb, alpha)}


def ratios(st, boots, ref="znn"):
    """Ratios of every ranked method to ``ref`` at every n (paired bootstrap CIs)."""
    out = []
    for n, g in st.groupby("n"):
        for lab in g.label:
            if lab == ref or not _ranked(lab):
                continue
            c = contrast(st, boots, lab, ref, n)
            if c is not None:
                c["label"] = lab
                out.append(c)
    return pd.DataFrame(out)


def contrast_table(st, boots):
    """Pre-specified contrasts at every n; Bonferroni level over the K contrasts at n_max."""
    nmax = int(st.n.max())
    avail = [(a, b, d) for a, b, d in CONTRASTS if (nmax, a) in boots and (nmax, b) in boots]
    K = max(len(avail), 1)
    rows = []
    for n in sorted(st.n.unique()):
        for a, b, d in CONTRASTS:
            c = contrast(st, boots, a, b, n)
            if c is None:
                continue
            cb = contrast(st, boots, a, b, n, alpha=0.05 / K)
            c.update({"desc": d, "cost_ratio_ci_bonf": cb["cost_ratio_ci"],
                      "mssre_ratio_ci_bonf": cb["mssre_ratio_ci"], "K_bonferroni": K})
            rows.append(c)
    return pd.DataFrame(rows), K


def kendall_table(st):
    """Kendall tau_b between cost measures, on the fixed core set (comparable across n)
    and on all ranked, non-duplicate methods (tuned rows have the protocol row's cost)."""
    rows = []
    nmax = st.n.max()
    # censoring-aware: a diverged cell (median MSSRE = +inf) is ranked worst (ties among
    # themselves, which kendalltau handles), only missing values (NaN) are dropped
    ref_acc = st[(st.n == nmax) & ~np.isnan(st.mssre_med)].set_index("label").mssre_med
    sets = {"core": lambda lab: lab in C.CORE_LABELS,
            "all": lambda lab: _ranked(lab) and lab not in C.TUNED_LABELS}
    for n, g0 in st.groupby("n"):
        r = {"n": int(n)}
        for sname, sel in sets.items():
            g = g0[np.array([sel(lab) for lab in g0.label], dtype=bool) & np.isfinite(g0.cost_med).to_numpy()]
            sfx = "" if sname == "core" else "_all"
            r[f"k{sfx}"] = len(g)
            for name, col in (("flops_costs", "flops_costs_py"), ("flops_ideal", "flops_ideal"),
                              ("flops_exec", "flops_exec"), ("e2e_runner", "cost_e2e_med"),
                              ("clean_B4", "cost_B4_med")):
                h = g[np.isfinite(g[col])]
                if len(h) >= 3:
                    t, p = kendalltau(h.cost_med, h[col])
                    r[f"tau_{name}{sfx}"], r[f"p_{name}{sfx}"] = float(t), float(p)
                else:
                    r[f"tau_{name}{sfx}"], r[f"p_{name}{sfx}"] = math.nan, math.nan
            h = g0[np.array([sel(lab) for lab in g0.label], dtype=bool)]
            h = h[~np.isnan(h.mssre_med)].set_index("label").mssre_med
            common = [x for x in h.index if x in ref_acc.index]
            if len(common) >= 3:
                t, p = kendalltau(h[common], ref_acc[common])
                r[f"tau_acc_vs_nmax{sfx}"], r[f"p_acc_vs_nmax{sfx}"] = float(t), float(p)
            else:
                r[f"tau_acc_vs_nmax{sfx}"], r[f"p_acc_vs_nmax{sfx}"] = math.nan, math.nan
        rows.append(r)
    return pd.DataFrame(rows)


def fit_slope(ns, c):
    x, y = np.log(np.asarray(ns, float)), np.log(np.asarray(c, float))
    if len(x) < 2:
        return math.nan
    return float(np.polyfit(x, y, 1)[0])


def asymptotic_range(ns, c, tol=SLOPE_TOL):
    """Indices of the asymptotic range: the longest tail of n over which the
    consecutive local slopes d log c / d log n stay within ``tol`` of the
    last local slope (at least the last two points)."""
    x, y = np.log(np.asarray(ns, float)), np.log(np.asarray(c, float))
    if len(x) < 3:
        return list(range(len(x))), np.diff(y) / np.diff(x) if len(x) > 1 else np.array([])
    loc = np.diff(y) / np.diff(x)
    start = len(loc) - 1
    while start > 0 and abs(loc[start - 1] - loc[-1]) <= tol:
        start -= 1
    return list(range(start, len(x))), loc


def scaling_table(st, boots, nb):
    rows = []
    sel = st[np.array([(_ranked(lab) and lab not in C.TUNED_LABELS) for lab in st.label], dtype=bool)]
    for lab, g in sel.groupby("label"):
        g = g.sort_values("n")
        g = g[np.isfinite(g.cost_med)]
        if len(g) < 2:
            continue
        ns, c = g.n.to_numpy(), g.cost_med.to_numpy()
        idx, loc = asymptotic_range(ns, c)
        s_full, s_asy = fit_slope(ns, c), fit_slope(ns[idx], c[idx])
        bmat = np.stack([boots[(n, lab)][0] for n in ns])
        bs = [fit_slope(ns[idx], bmat[idx, b]) for b in range(bmat.shape[1])]
        fl = g.flops_axis.to_numpy(dtype=float)
        s_fl = fit_slope(ns[np.isfinite(fl)][-2:], fl[np.isfinite(fl)][-2:]) if np.isfinite(fl).sum() >= 2 else math.nan
        pre = bool(len(idx) < 3 or (len(loc) and loc[-1] < 2.5))
        osh = g.overhead_share.to_numpy(dtype=float) if "overhead_share" in g else np.full(len(g), np.nan)
        # largest n at which the n = 4 overhead floor is still >= half of the per-step time
        ov_n = [int(v) for v, o in zip(ns, osh) if np.isfinite(o) and o >= 0.5]
        rows.append({"label": lab, "slope_full": s_full, "slope_asym": s_asy, "slope_asym_ci": ci(bs),
                     "asym_n": [int(v) for v in ns[idx]], "local_slopes": [float(v) for v in loc],
                     "local_slope_last": float(loc[-1]) if len(loc) else math.nan,
                     "flops_slope_last": s_fl, "n_points_asym": len(idx), "two_point_fit": len(idx) == 2,
                     "pre_asymptotic": pre,
                     "floor_us": float(g.floor_cost.iloc[0]) * 1e6 if "floor_cost" in g else math.nan,
                     "overhead_share_nmax": float(osh[-1]) if len(osh) else math.nan,
                     "overhead_dominated_upto_n": max(ov_n) if ov_n else None})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------
def _sty(lab, extras_order=None):
    if lab in keys.GROUPS["baseline"]:
        return keys.style_in_group("baseline", lab)
    if lab in C.TUNED_LABELS:
        # colour follows the entity: the tuned row of a method keeps its colour and marker,
        # drawn hollow (legend text says "tuned h")
        base = keys.style_in_group("baseline", C.TUNED_LABELS[lab])
        return {**base, "mfc": "white", "linestyle": ":"}
    if extras_order and lab in extras_order:
        # entities outside the groups: no categorical colour (it belongs to group members);
        # dark / mid grey with distinct hollow markers, identified by the legend
        i = extras_order.index(lab)
        return {"color": (style.INK, style.GREY, style.INK2)[i % 3], "marker": ("D", "s", "^", "v", "P", "X")[i % 6],
                "linestyle": (":", "--", "-.")[i % 3], "mfc": "white"}
    return {"color": style.GREY, "marker": ".", "linestyle": ":"}


def _grid(k):
    if k <= 3:
        return 1, k
    if k == 4:
        return 2, 2
    return 2, int(math.ceil(k / 2))


def _errbar(ax, x, y, xlo, xhi, ylo, yhi, sty, label, **kw):
    xe = None if xlo is None else [[max(x - xlo, 0)], [max(xhi - x, 0)]]
    ye = None if ylo is None else [[max(y - ylo, 0)], [max(yhi - y, 0)]]
    ax.errorbar([x], [y], xerr=xe, yerr=ye, color=sty["color"], marker=sty["marker"], linestyle="none",
                ms=4.2, mec=sty["color"], mfc=sty.get("mfc", sty["color"]), elinewidth=0.6, capsize=0,
                label=label, zorder=3, **kw)


def _plain_log_axis(axis, subs=(1.0, 2.0, 3.0, 5.0)):
    """Log axis with plain-number labels at 1-2-5 positions (no 2x10^1 clutter)."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    axis.set_major_locator(LogLocator(base=10, subs=subs))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())


#: short names of censored rows drawn in the "diverged" strip of the Pareto panels
SHORT = {"gnn_matched": "GNN $h{=}0.3$", "hgnn_znn_matched": "H-GNN-ZNN $h{=}0.3$",
         "fa_gnn_znn_matched": "FA-GNN-ZNN $h{=}0.3$", "lcc_znn": "LCCZNN exp-sig"}


def _censor_style(lab, extras_order=None):
    """Censored rows keep the colour and marker of their entity (matched rows: the stable row
    of the same model); ``fig_pareto`` draws them half-filled in the "diverged" strip."""
    base = lab[: -len("_matched")] if C.is_matched(lab) else lab
    sty = _sty(base, extras_order) if (base in keys.GROUPS["baseline"] or (extras_order and base in extras_order)
                                        or base in C.TUNED_LABELS) else {"color": style.INK2, "marker": "X",
                                                                         "linestyle": ":"}
    return {**sty, "mfc": "white"}


PAPER_PARETO = {"fig_E7_pareto"}       # Supplementary Material figure, drawn at the paper's text width


def fig_pareto(st, labels, name, cost_col="cost_med", xlabel="", errors=True, extras_order=None,
               front_col="pareto", per_n3=False, arrows=None, censored=None):
    """See _fig_pareto; the paper figure is drawn at style.TEXTWIDTH inside style.print_rc()."""
    kw = dict(cost_col=cost_col, xlabel=xlabel, errors=errors, extras_order=extras_order, front_col=front_col,
              per_n3=per_n3, arrows=arrows, censored=censored)
    if name in PAPER_PARETO:
        with style.print_rc():
            return _fig_pareto(st, labels, name, paper=True, **kw)
    return _fig_pareto(st, labels, name, **kw)


def _fig_pareto(st, labels, name, cost_col="cost_med", xlabel="", errors=True, extras_order=None,
                front_col="pareto", per_n3=False, arrows=None, censored=None, paper=False):
    """Small multiples (one panel per n): MSSRE vs cost, IQR bars, Pareto front.
    ``arrows``: list of (from_label, to_label) drawn as thin arrows (gain arm).
    ``censored``: labels whose diverged cells (median MSSRE = inf) are drawn in a "diverged"
    strip above the finite range, at their own clean cost or at the cost of their
    operation-identical proxy (``cost_plot``), hollow, with a short name and the diverged
    count -- censored cells are shown, never silently dropped (checklist item 2).  The
    series in ``labels`` are always candidates."""
    assert len(labels) <= 8, "style rule: <= 8 series per panel"
    ns = sorted(st.n.unique())
    nr, nc = _grid(len(ns))
    width = style.DOUBLE if nc >= 3 else style.ONEHALF
    fs_note = 7.5 if paper else 5           # in-panel notes (censored counts)
    if paper:
        fig, axes = plt.subplots(nr, nc, figsize=(style.TEXTWIDTH, 1.62 * nr + 0.85), squeeze=False,
                                 constrained_layout=True)
        fig.get_layout_engine().set(w_pad=0.03, h_pad=0.03, wspace=0.04, hspace=0.06)
    else:
        fig, axes = plt.subplots(nr, nc, figsize=(width, 1.95 * nr + 0.5), squeeze=False)
    cens_labels = list(dict.fromkeys(list(labels) + list(censored or [])))
    sub = st[st.label.isin(labels)]
    csub = st[st.label.isin(cens_labels) & st.censored]
    plot_col = "cost_plot" if cost_col == "cost_med" else cost_col
    fronts, cens_pts = [], []
    for k, (ax, n) in enumerate(zip(axes.flat, ns)):
        g = sub[sub.n == n]
        scale = 1e6 if cost_col.startswith("cost") else (1.0 / n ** 3 if per_n3 else 1.0)
        pts = {}
        for lab in labels:
            r = g[g.label == lab]
            if r.empty or not (np.isfinite(r[cost_col].iloc[0]) and np.isfinite(r.mssre_med.iloc[0])):
                continue
            r = r.iloc[0]
            sty = _sty(lab, extras_order)
            pts[lab] = (r[cost_col] * scale, r.mssre_med)
            if errors:
                _errbar(ax, r[cost_col] * scale, r.mssre_med, r.cost_q25 * scale, r.cost_q75 * scale,
                        r.mssre_q25, r.mssre_q75, sty, C.PRETTY.get(lab, lab))
            else:
                _errbar(ax, r[cost_col] * scale, r.mssre_med, None, None, r.mssre_q25, r.mssre_q75, sty,
                        C.PRETTY.get(lab, lab))
        for lab in cens_labels:
            r = csub[(csub.n == n) & (csub.label == lab)]
            if r.empty:
                continue
            r = r.iloc[0]
            x = r[plot_col] * scale if np.isfinite(r[plot_col]) else math.nan
            cens_pts.append((ax, lab, x, int(r.n_div), int(r.n_traj),
                             plot_col == "cost_plot" and str(r.get("cost_plot_src", "own")).startswith("proxy")))
        for a, b in (arrows or []):
            if a in pts and b in pts:
                ax.annotate("", xy=pts[b], xytext=pts[a], zorder=2,
                            arrowprops=dict(arrowstyle="-|>", color=_sty(a)["color"], lw=0.6,
                                            alpha=0.7, shrinkA=3.5, shrinkB=3.5, mutation_scale=6))
        fr = g[g[front_col] & np.isfinite(g[cost_col]) & np.isfinite(g.mssre_med)].sort_values(cost_col)
        fronts.append((ax, fr[cost_col].to_numpy() * scale, fr.mssre_med.to_numpy()))
        ax.set_xscale("log")
        ax.set_yscale("log")
        xv = np.concatenate([g[cost_col].to_numpy(dtype=float) * scale,
                             [p[2] for p in cens_pts if p[0] is ax]])
        xv = xv[np.isfinite(xv) & (xv > 0)]
        if xv.size and xv.max() / xv.min() < 30:
            _plain_log_axis(ax.xaxis)
        ax.set_title(f"$n = {n}$")
        ax.axhspan(1e-17, FLOOR, color=style.GREY, alpha=0.10, lw=0, zorder=0)
        if k == 0:
            ax.text(0.02, 0.02, "round-off floor", transform=ax.transAxes, ha="left", va="bottom",
                    fontsize=7.5 if paper else 6, color=style.INK2)
    for ax in axes.flat[len(ns):]:
        ax.set_visible(False)
    lo = sub.mssre_q25.replace([np.inf, 0], np.nan)
    hi = sub.mssre_q75.replace(np.inf, np.nan)
    y_strip = None
    if np.isfinite(lo).any() and np.isfinite(hi).any():
        y0 = 10 ** math.floor(math.log10(min(np.nanmin(lo), FLOOR)) - 0.3)
        y1 = 10 ** math.ceil(math.log10(np.nanmax(hi)) + 0.3)
        ytop = y1
        if cens_pts:                # "diverged" strip above the finite range
            ytop = y1 * (y1 / y0) ** 0.14
            y_strip = (y1, ytop)
        for ax in axes.flat[:len(ns)]:
            ax.set_ylim(y0, ytop)
            if cens_pts:            # no numeric tick labels inside the "diverged" strip
                from matplotlib.ticker import FuncFormatter, LogFormatterSciNotation
                bf = LogFormatterSciNotation()
                bf.set_axis(ax.yaxis)
                cut = y1 / 30 if paper else y1 * 1.001      # paper: keep the top label clear of the infinity sign
                ax.yaxis.set_major_formatter(FuncFormatter(
                    lambda v, pos, _bf=bf, _c=cut: "" if v > _c else _bf(v, pos)))
    for ax in axes.flat[:len(ns)]:          # x-range must contain the censored cells too
        cx = np.array([p[2] for p in cens_pts if p[0] is ax and np.isfinite(p[2]) and p[2] > 0])
        if cx.size:
            xl, xr = ax.get_xlim()
            ax.set_xlim(min(xl, cx.min() / 1.15), max(xr, cx.max() * 1.15))
    for ax, xs, ys in fronts:      # attainment surface: staircase extended to the panel edges
        if not len(xs):
            continue
        (xl, xr), (yb, yt) = ax.get_xlim(), ax.get_ylim()
        ys_top = y_strip[0] if y_strip is not None else yt     # stops below the diverged strip
        xx = np.concatenate([[xs[0]], np.repeat(xs, 2)[1:], [xr]])
        yy = np.concatenate([[ys_top], np.repeat(ys, 2)[:-1], [ys[-1]]])
        ax.plot(xx, yy, color=style.INK2, lw=0.8, alpha=0.55, zorder=1, label="Pareto front")
        ax.set_xlim(xl, xr)
        ax.set_ylim(yb, yt)
    # censored cells: hollow markers in the strip at their own clean cost or, marked with a
    # double dagger, at the cost of their operation-identical proxy; text = diverged/total
    if y_strip is not None:
        ya, yb_ = y_strip
        series_entities = set(labels)
        legend_done = set()
        for ax in axes.flat[:len(ns)]:
            mine = [p for p in cens_pts if p[0] is ax]
            if not mine:
                continue
            ax.axhspan(ya, yb_, color=style.GREY, alpha=0.08, lw=0, zorder=0)
            # pseudo tick label: the strip is MSSRE = infinity
            ax.text(-0.012, math.sqrt(ya * yb_), "$\\infty$", transform=ax.get_yaxis_transform(), ha="right",
                    va="center", fontsize=9 if paper else 8, color=style.INK)
            mine = sorted(mine, key=lambda p: (p[2] if np.isfinite(p[2]) else np.inf))
            xl, xr = ax.get_xlim()
            for j, (_, lab, x, nd, nt, proxy) in enumerate(mine):
                sty = _censor_style(lab, extras_order)
                base = lab[: -len("_matched")] if C.is_matched(lab) else lab
                # censored = half-filled marker of the entity (filled = protocol row, hollow = tuned
                # row or extra), so it cannot be mistaken for a finite point
                cm = dict(linestyle="none", marker=sty["marker"], ms=4.2, mec=sty["color"], mfc=sty["color"],
                          mfcalt="white", fillstyle="bottom", mew=0.8)
                if base not in series_entities and lab not in legend_done:
                    # an entity without a series of its own needs a legend entry
                    ax.plot([], [], label=C.PRETTY.get(lab, lab) + " (diverged)", **cm)
                    legend_done.add(lab)
                txt = f"{nd}/{nt}" + ("$^{\\ddagger}$" if proxy else "")
                if not np.isfinite(x):
                    ax.text(0.98, 0.985 - 0.05 * j, f"{SHORT.get(lab, lab)} {txt} (not timed)",
                            transform=ax.transAxes, ha="right", va="top", fontsize=fs_note,
                            color=style.INK2 if paper else sty["color"])
                    continue
                y = ya * (yb_ / ya) ** (0.3 if j % 2 == 0 else 0.62)
                ax.plot([x], [y], zorder=4, **cm)
                right = (math.log(x) - math.log(xl)) / max(math.log(xr) - math.log(xl), 1e-12) > 0.8
                ax.annotate(txt, xy=(x, y), xytext=(-3.5 if right else 3.5, 0), textcoords="offset points",
                            fontsize=fs_note, color=style.INK2 if paper else sty["color"],
                            ha="right" if right else "left", va="center", zorder=4)
    for ax in axes[-1, :]:
        ax.set_xlabel(xlabel)
    for ax in axes[:, 0]:
        ax.set_ylabel("MSSRE")
    hs, ls, seen = [], [], set()
    for ax in axes.flat[:len(ns)]:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                seen.add(l)
                hs.append(h)
                ls.append(l)
    order = [C.PRETTY.get(lab, lab) for lab in labels] + ["Pareto front"]
    pairs = sorted(zip(hs, ls), key=lambda p: order.index(p[1]) if p[1] in order else 99)
    if paper:
        fig.legend([p[0] for p in pairs], [p[1] for p in pairs], loc="outside lower center", ncol=3,
                   frameon=False, handletextpad=0.4, columnspacing=1.2)
        return style.savefig(fig, name)
    ncol = min(4 if arrows else 5, len(pairs))
    nrows = int(math.ceil(len(pairs) / ncol))
    fig.legend([p[0] for p in pairs], [p[1] for p in pairs], loc="lower center", ncol=ncol,
               bbox_to_anchor=(0.5, 0.0), frameon=False, handletextpad=0.4, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.035 + 0.045 * nrows * (2.45 / fig.get_size_inches()[1]), 1, 1),
                     h_pad=0.5, w_pad=0.8)
    return style.savefig(fig, name)


def probe_summary(tim):
    """Machine-state probe of the timing tasks (``E7_timing.machine_probe``, one per round):
    128 x 128 dgemm and LU factor + solve times.  Rounds whose dgemm probe exceeds
    PROBE_FLAG x the best probe of the whole run were timed under contention for the shared
    matrix unit (other processes), which inflates GEMM-heavy methods most."""
    if tim.empty or "probe_gemm_us_all" not in tim:
        return None
    vals = {}
    for tid, g in tim.groupby("task_id"):
        v = g.probe_gemm_us_all.dropna()
        w = g.probe_lu_us_all.dropna() if "probe_lu_us_all" in g else pd.Series(dtype=object)
        if len(v):
            vals[tid] = (json.loads(v.iloc[0]), json.loads(w.iloc[0]) if len(w) else [])
    if not vals:
        return None
    gm = np.array([x for v, _ in vals.values() for x in v], dtype=float)
    lu = np.array([x for _, w in vals.values() for x in w], dtype=float)
    best = float(np.min(gm))
    ratio = gm / best
    worst_task = max(vals, key=lambda t: max(vals[t][0]) if vals[t][0] else 0)
    return {"rounds": int(gm.size), "gemm_us_min": best, "gemm_us_median": float(np.median(gm)),
            "gemm_us_max": float(np.max(gm)), "gemm_ratio_median": float(np.median(ratio)),
            "gemm_ratio_max": float(np.max(ratio)), "frac_rounds_contended": float(np.mean(ratio > PROBE_FLAG)),
            "lu_us_min": float(np.min(lu)) if lu.size else None,
            "lu_us_median": float(np.median(lu)) if lu.size else None,
            "lu_ratio_max": float(np.max(lu) / np.min(lu)) if lu.size else None,
            "gemm_gflops_best": 2.0 * 128 ** 3 / (best * 1e-6) / 1e9, "worst_task": worst_task,
            "flag": PROBE_FLAG}


def fig_scaling(st, rat, labels, name, probe=None):
    """(a) per-step cost vs n, (b) paired cost ratio to ZNN-Euler, (c) effective GFLOP/s
    (with the single-thread 128 x 128 dgemm rate of the machine probe as a ceiling)."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator
    fig, axes = plt.subplots(1, 3, figsize=(style.DOUBLE, 2.5))
    a, b, c = axes
    ns = sorted(st.n.unique())
    rmin, rmax = 1.0, 1.0
    for lab in labels:
        g = st[(st.label == lab) & np.isfinite(st.cost_med)].sort_values("n")
        if g.empty:
            continue
        sty = _sty(lab)
        kw = dict(color=sty["color"], marker=sty["marker"], linestyle=sty["linestyle"], ms=3.6)
        y = g.cost_med.to_numpy() * 1e6
        a.errorbar(g.n, y, yerr=[y - g.cost_q25 * 1e6, g.cost_q75 * 1e6 - y], elinewidth=0.6, capsize=0,
                   label=C.PRETTY.get(lab, lab), **kw)
        gf = g[np.isfinite(g.gflops_eff)]
        if len(gf):
            c.plot(gf.n, gf.gflops_eff, label=C.PRETTY.get(lab, lab), **kw)
        if lab != "znn" and len(rat):
            r = rat[rat.label == lab].sort_values("n")
            if len(r):
                lo = np.array([v[0] for v in r.cost_ratio_ci])
                hi = np.array([v[1] for v in r.cost_ratio_ci])
                yv = r.cost_ratio.to_numpy()
                rmin, rmax = min(rmin, np.nanmin(lo)), max(rmax, np.nanmax(hi))
                b.errorbar(r.n, yv, yerr=[np.maximum(yv - lo, 0), np.maximum(hi - yv, 0)], elinewidth=0.6,
                           capsize=0, label=C.PRETTY.get(lab, lab), **kw)
    z = st[(st.label == "znn") & np.isfinite(st.cost_med)].sort_values("n")
    if len(z) >= 2:          # n^3 guide over the last octave, through ZNN-Euler's largest-n point
        n1, c1 = z.n.iloc[-1], z.cost_med.iloc[-1] * 1e6
        nn = np.array([z.n.iloc[-2], n1], dtype=float)
        a.plot(nn, c1 * (nn / n1) ** 3, color=style.GREY, lw=0.8, ls=(0, (4, 2)), zorder=1)
        a.annotate("slope 3", xy=(nn[0], c1 * (nn[0] / n1) ** 3), xytext=(4, 0), textcoords="offset points",
                   fontsize=6.5, color=style.INK2, ha="left", va="center")
    b.axhline(1.0, color=style.GREY, lw=0.8, zorder=1)
    if probe and np.isfinite(probe.get("gemm_gflops_best", np.nan)):
        gfl = probe["gemm_gflops_best"]
        c.axhline(gfl, color=style.GREY, lw=0.8, ls=(0, (4, 2)), zorder=1)
        c.annotate("1-thread dgemm, $n=128$", xy=(ns[0], gfl), xytext=(0, -2), textcoords="offset points",
                   fontsize=6, color=style.INK2, ha="left", va="top")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_locator(FixedLocator(ns))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel("matrix size $n$")
    a.set_yscale("log")
    _plain_log_axis(a.yaxis)
    b.set_yscale("log")
    yb = [v for v in (0.125, 0.25, 0.5, 1, 2, 4, 8) if rmin / 1.3 <= v <= rmax * 1.3] or [0.5, 1, 2]
    b.yaxis.set_major_locator(FixedLocator(yb))
    b.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    b.yaxis.set_minor_formatter(NullFormatter())
    c.set_yscale("log")
    _plain_log_axis(c.yaxis, subs=(1.0,))
    a.set_ylabel("wall-clock per step [$\\mu$s]")
    b.set_ylabel("relative cost (ZNN-Euler = 1)")
    c.set_ylabel("effective GFLOP/s")
    a.set_title("(a) per-step cost (B = 1, 1 thread)")
    b.set_title("(b) paired cost ratio, 95% CI")
    c.set_title("(c) analytic FLOPs / time")
    h, l = a.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0), frameon=False,
               handletextpad=0.4, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.14, 1, 1), w_pad=1.0)
    return style.savefig(fig, name)


# ----------------------------------------------------------------------------
# tables
# ----------------------------------------------------------------------------
def _pretty_tex(lab):
    p = C.PRETTY.get(lab, lab.replace("_", r"\_"))
    return p.replace("3_4", r"3\_4")


def _sci(v, d=1):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "--"
    if math.isinf(v):
        return r"$\infty$"
    m, e = f"{v:.{d}e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def _f2(v):
    return "--" if v is None or not np.isfinite(v) else f"{v:.2f}"


def _floor_mark(txt, at_floor):
    """Append a dagger inside the math of a _sci() string (no '$$' in LaTeX)."""
    if not at_floor:
        return txt
    return txt[:-1] + r"{}^{\dagger}$" if txt.endswith("$") else txt + r"$^{\dagger}$"


def _order(lab):
    o = {lab: i for i, lab in enumerate(C.CORE_LABELS + list(C.TUNED_LABELS) + ["gnn_matched"])}
    return o.get(lab, 50)


def write_tables(st, rat, ct, K, sct, kt, tabdir: Path, scale_name):
    tabdir.mkdir(parents=True, exist_ok=True)
    L = [r"\begin{tabular}{rlrrrrl}", r"\toprule",
         r"$n$ & method & $t$/step [$\mu$s] & IQR & FLOPs/step & MSSRE & Pareto \\",
         r"\midrule"]
    for n, g in st.groupby("n"):
        g = g.assign(o=g.label.map(_order)).sort_values(["o", "label"])
        for _, r in g.iterrows():
            par = (r"\checkmark" + (f" ({100 * r.pareto_freq:.0f}\\%)" if np.isfinite(r.pareto_freq) else "")
                   if r.pareto else (f"({100 * r.pareto_freq:.0f}\\%)" if np.isfinite(r.pareto_freq)
                                     and r.pareto_freq > 0 else ""))
            div = f" [{r.n_div}/{r.n_traj} div.]" if r.n_div else ""
            ms = _floor_mark(_sci(r.mssre_med), r.at_floor) + (r"\textsuperscript{\S}" if r.unsettled else "")
            cost = (f"{r.cost_med * 1e6:.1f} & [{r.cost_q25 * 1e6:.1f}, {r.cost_q75 * 1e6:.1f}]"
                    if np.isfinite(r.cost_med) else "-- & --")
            L.append(f"{int(n)} & {_pretty_tex(r.label)} & {cost} & {_sci(r.flops_axis)} & {ms}{div} & {par} \\\\")
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L += [r"\end{tabular}"]
    (tabdir / "E7_pareto.tex").write_text(
        f"% E7 ({scale_name}): cost = median of B=1 timing runs (wall_dynamics/step, interleaved rounds); "
        f"MSSRE = median over trajectories; Pareto = non-dominated (5% ties) among all ranked methods at n, "
        f"(x%) = paired cluster-bootstrap frequency; dagger = at the round-off floor (< {FLOOR:g}); "
        f"section mark = steady window not settled (settle ratio > {C.SETTLE_TOL:g}).\n" + "\n".join(L) + "\n")

    nmax = int(st.n.max())
    H = [r"\begin{tabular}{p{5.6cm}rrrr}", r"\toprule",
         r"contrast ($a/b$) at $n=" + str(nmax) + r"$ & cost ratio & 95\% CI & MSSRE ratio & 95\% CI \\",
         r"\midrule"]
    cn = ct[ct.n == nmax] if len(ct) else ct
    for _, r in cn.iterrows():
        H.append(f"{_tex_desc(r.desc)} & {_f2(r.cost_ratio)} & [{_f2(r.cost_ratio_ci[0])}, {_f2(r.cost_ratio_ci[1])}] & "
                 f"{_sci(r.mssre_ratio, 2)} & [{_sci(r.mssre_ratio_ci[0], 2)}, {_sci(r.mssre_ratio_ci[1], 2)}] \\\\")
    H += [r"\bottomrule", r"\end{tabular}"]
    (tabdir / "E7_headline.tex").write_text(
        f"% E7 ({scale_name}): pre-specified contrasts, ratios of medians a/b, paired two-stage cluster "
        f"bootstrap (seed, then round / initial state; 2000 resamples). Bonferroni-simultaneous CIs "
        f"(level 1-0.05/{K}) are in E7_contrasts.csv and summary.json.\n" + "\n".join(H) + "\n")

    R = [r"\begin{tabular}{rlrrrr}", r"\toprule",
         r"$n$ & method & cost ratio & 95\% CI & FLOP ratio & MSSRE ratio [95\% CI] \\", r"\midrule"]
    for n, g in rat.groupby("n"):
        g = g.assign(o=g.label.map(_order)).sort_values(["o", "label"])
        for _, r in g.iterrows():
            R.append(f"{int(n)} & {_pretty_tex(r.label)} & {_f2(r.cost_ratio)} & "
                     f"[{_f2(r.cost_ratio_ci[0])}, {_f2(r.cost_ratio_ci[1])}] & {_f2(r.flops_ratio)} & "
                     f"{_sci(r.mssre_ratio)} [{_sci(r.mssre_ratio_ci[0])}, {_sci(r.mssre_ratio_ci[1])}] \\\\")
        R.append(r"\midrule")
    R[-1] = r"\bottomrule"
    R += [r"\end{tabular}"]
    (tabdir / "E7_ratios.tex").write_text(
        f"% E7 ({scale_name}): ratios of medians to ZNN-Euler (LU, h = 0.3), paired cluster-bootstrap "
        f"percentile CIs (2000 resamples).\n" + "\n".join(R) + "\n")

    nmin = int(st.n.min())
    S = [r"\begin{tabular}{lrrrlrr}", r"\toprule",
         r"method & slope (all $n$) & slope (asymptotic) & 95\% CI & asymptotic range & "
         r"floor [$\mu$s] & overhead at $n_{\max}$ \\", r"\midrule"]
    for _, r in sct.assign(o=sct.label.map(_order)).sort_values(["o", "label"]).iterrows():
        osh = float(r.get("overhead_share_nmax", math.nan))
        fl = float(r.get("floor_us", math.nan))
        tail = (f" & {fl:.1f} & {100 * osh:.0f}\\%" if np.isfinite(osh) and np.isfinite(fl) else " & -- & --")
        S.append(f"{_pretty_tex(r.label)} & {r.slope_full:.2f} & {r.slope_asym:.2f} & "
                 f"[{r.slope_asym_ci[0]:.2f}, {r.slope_asym_ci[1]:.2f}] & "
                 f"$n\\in\\{{{','.join(str(v) for v in r.asym_n)}\\}}$"
                 + (" (2-pt)" if r.get("two_point_fit", False) else "")
                 + (" (pre-asympt.)" if r.pre_asymptotic else "") + tail + " \\\\")
    S += [r"\bottomrule", r"\end{tabular}"]
    (tabdir / "E7_scaling.tex").write_text(
        f"% E7 ({scale_name}): log-log slope of wall-clock per step vs n; asymptotic range = tail "
        f"with consecutive local slopes within {SLOPE_TOL} of the last one (2-pt = two-point fit); pre-asympt. = "
        f"fewer than 3 points or last local slope < 2.5. floor = per-step time at n = {nmin} (< 1e3 FLOPs: "
        f"interpreter and call overhead); overhead at n_max = floor / cost(n_max).\n" + "\n".join(S) + "\n")

    Kt = [r"\begin{tabular}{rrrrrrrr}", r"\toprule",
          r"$n$ & costs.py & ideal & executed & runner wall & B=1 vs B=4 & MSSRE vs $n_{\max}$ & "
          r"costs.py (all) \\", r"\midrule"]

    def f(v):
        return "--" if not np.isfinite(v) else f"{v:.2f}"
    for _, r in kt.iterrows():
        Kt.append(f"{int(r.n)} & {f(r.tau_flops_costs)} & {f(r.tau_flops_ideal)} & {f(r.tau_flops_exec)} & "
                  f"{f(r.tau_e2e_runner)} & {f(r.tau_clean_B4)} & {f(r.tau_acc_vs_nmax)} & "
                  f"{f(r.tau_flops_costs_all)} ({int(r.k_all)}) \\\\")
    Kt += [r"\bottomrule", r"\end{tabular}"]
    (tabdir / "E7_kendall.tex").write_text(
        f"% E7 ({scale_name}): Kendall tau_b of the clean B=1 time ranking against other cost measures, on the "
        f"fixed set of 8 core methods; last column: all ranked methods (count in parentheses).\n"
        + "\n".join(Kt) + "\n")


def _tex_desc(d):
    """Plain-text contrast description -> LaTeX (math pieces first, then escape the rest;
    the old order escaped '_' first and left 'sigma_{\\max}^2' outside math mode)."""
    s = str(d).replace("h sigma_max^2 ~ 1", "@@HS@@").replace("sigma_max^2", "@@S@@").replace("~", "@@A@@")
    s = s.replace("_", r"\_")
    return (s.replace("@@HS@@", r"$h\sigma_{\max}^2\approx 1$").replace("@@S@@", r"$\sigma_{\max}^2$")
            .replace("@@A@@", r"$\approx$"))


# ----------------------------------------------------------------------------
# summary
# ----------------------------------------------------------------------------
def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return None if math.isnan(v) else (("inf" if v > 0 else "-inf") if math.isinf(v) else v)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _xcheck(st):
    """Measured-call-count FLOPs (timing profile) / costs.py FLOPs at n >= 32."""
    g = st[(st.n >= 32) & np.isfinite(st.flops_exec) & np.isfinite(st.flops_costs_code)
           & (st.flops_costs_code > 0)]
    if g.empty:
        return None
    r = (g.flops_exec / g.flops_costs_code).to_numpy()
    return {"median": float(np.median(r)), "min": float(r.min()), "max": float(r.max()), "k": int(r.size)}


def _c(ct, a, b, n):
    if ct is None or not len(ct):
        return None
    r = ct[(ct.n == n) & (ct.a == a) & (ct.b == b)]
    return r.iloc[0] if len(r) else None


def interpretation(st, ct, sct, kt, nmax, scale_name):
    """Short data-driven interpretation (each sentence is conditioned on the numbers)."""
    out = []
    big = st[st.n == nmax].set_index("label")

    def has(lab):
        return lab in big.index and np.isfinite(big.loc[lab, "cost_med"]) and np.isfinite(big.loc[lab, "mssre_med"])
    sc = sct[sct.label.isin(C.CORE_LABELS)] if len(sct) else sct
    if len(sc):
        last = sc.set_index("label").local_slope_last
        pre = bool(sc.pre_asymptotic.all())
        n2 = sorted(st.n.unique())[-2] if st.n.nunique() > 1 else nmax
        out.append(f"Wall-clock per step of the core methods grows like n^{last.min():.1f}-n^{last.max():.1f} "
                   f"over the last octave (n = {n2}..{nmax}); "
                   + ("the O(n^3) regime is not reached at these sizes (Python/LAPACK call overhead dominates), so "
                      "per-step costs differ by implementation-dependent constant factors."
                      if pre else "at least one core method has entered the O(n^3) regime (see the scaling table)."))
    ov = st[np.isfinite(st.overhead_share) & st.label.isin(C.CORE_LABELS)]
    if len(ov) and (ov.n == nmax).any():
        med = ov.groupby("n").overhead_share.median()
        dom = [int(n) for n, v in med.items() if v >= 0.5]
        out.append(f"The per-step floor at n = {int(st.n.min())} (interpreter and call overhead, < 1e3 FLOPs) is "
                   f"{100 * med.loc[nmax]:.0f}% (median over the core methods) of the per-step time at n = {nmax}"
                   + (f", and at least half of it for n <= {max(dom)}, where wall-clock contrasts measure the Python "
                      f"implementation rather than the algorithm" if dom else "") + ".")
    ex_fl = sct[np.array([_is_extra(lab) for lab in sct.label], dtype=bool)] if len(sct) else sct
    if len(ex_fl) and np.isfinite(ex_fl.flops_slope_last).any():
        parts = [f"{r.label} ~ n^{r.flops_slope_last:.1f}" for _, r in ex_fl.iterrows() if np.isfinite(r.flops_slope_last)
                 and r.flops_slope_last > 3.5]
        if parts:
            out.append("The generic inverse-free schemes act on dense n^2 x n^2 matrices; their analytic FLOPs grow like "
                       + ", ".join(parts) + ".")
    c = _c(ct, "znn_explicit_inverse", "znn", nmax)
    if c is not None and np.isfinite(c.cost_ratio):
        same = np.isfinite(c.mssre_ratio) and abs(np.log(c.mssre_ratio)) < np.log(1.1)
        if c.cost_ratio_ci[1] < 1:
            out.append(f"Replacing the LU solve by the running inverse estimate (explicit-inverse ZNN) cuts the per-step "
                       f"time by a factor {1 / c.cost_ratio:.1f} at n = {nmax}"
                       + (" at unchanged accuracy" if same else f" (MSSRE ratio {c.mssre_ratio:.2g})")
                       + f", although the analytic FLOP ratio is {c.flops_ratio:.2f}: the LU path (two LAPACK calls "
                       f"plus the factor-cache bookkeeping) delivers fewer FLOPs per second than GEMM at these sizes.")
    c1, c2 = _c(ct, "pc_newton_causal", "znn", nmax), _c(ct, "pc_newton_causal", "znn_tuned", nmax)
    if c1 is not None and c2 is not None and np.isfinite(c2.mssre_ratio):
        eq = abs(np.log(c2.mssre_ratio)) < np.log(1.1)
        out.append(f"At the protocol gain (h = 0.3) causal PC-Newton costs {c1.cost_ratio:.2f}x ZNN-Euler for "
                   f"{c1.mssre_ratio:.2f}x its MSSRE, but this is a gain effect: one Newton correction is a dead-beat "
                   f"(h = 1) step, and ZNN-Euler at its own tuned h = 1 "
                   + (f"reaches the same MSSRE (ratio {c2.mssre_ratio:.2f}) " if eq
                      else f"gives MSSRE ratio {c2.mssre_ratio:.2f} ")
                   + f"at {1 / c2.cost_ratio:.2f}x the per-step time of PC-Newton.")
    c3 = _c(ct, "znn_explicit_inverse_tuned", "znn_tuned", nmax)
    if c3 is not None and has("znn_explicit_inverse_tuned"):
        out.append(f"The explicit-inverse ZNN at h = 1 (a Newton-Schulz step with derivative feed-forward) "
                   f"matches that accuracy (MSSRE ratio {c3.mssre_ratio:.2f}) at {c3.cost_ratio:.2f}x the cost.")
    if has("uhlig_3_4") and has("znn"):
        out.append(f"The higher-order look-ahead formula (Uhlig 3_4 at its h_opt) costs "
                   f"{big.loc['uhlig_3_4', 'cost_med'] / big.loc['znn', 'cost_med']:.2f}x ZNN-Euler per step and reaches "
                   f"MSSRE {big.loc['uhlig_3_4', 'mssre_med']:.1e}"
                   + (" (round-off floor)" if big.loc["uhlig_3_4", "at_floor"] else "") + ".")
    fr_p = st[(st.n == nmax) & st.pareto_protocol].sort_values("cost_med").label.tolist()
    fr_a = st[(st.n == nmax) & st.pareto].sort_values("cost_med").label.tolist()
    out.append(f"Pareto front at n = {nmax}: protocol arm (h = 0.3) {', '.join(fr_p)}; with the tuned arm and "
               f"extras {', '.join(fr_a)}.")
    aa = [x for x in (_c(ct, "znn_tuned", "znn", nmax), _c(ct, "gnn_tuned", "gnn", nmax)) if x is not None]
    if aa:
        out.append("A/A timing control (identical operations, different gain): cost ratios "
                   + ", ".join(f"{x.a}/{x.b} = {x.cost_ratio:.3f} [{x.cost_ratio_ci[0]:.3f}, {x.cost_ratio_ci[1]:.3f}]"
                               for x in aa) + ".")
    if len(kt) and "tau_flops_costs" in kt:
        tk = kt.set_index("n").tau_flops_costs
        if np.isfinite(tk).any() and tk.min() < 1:
            out.append(f"Analytic FLOP counts rank the 8 core methods only partly like measured time (Kendall tau_b "
                       f"{np.nanmin(tk):.2f}-{np.nanmax(tk):.2f} across n), so FLOP-based cost claims should not replace "
                       f"timings at these sizes.")
    if has("gnn"):
        g2 = f" (tuned h: {big.loc['gnn_tuned', 'mssre_med']:.1e})" if has("gnn_tuned") else ""
        out.append(f"GNN is the cheapest per step but its O(1/gamma) lag gives MSSRE {big.loc['gnn', 'mssre_med']:.1e}"
                   f"{g2}; at the matched h = 0.3 (gamma tau sigma_max^2 >> 2) it diverges.")
    if has("newton_tracking_causal"):
        out.append(f"Correction-only Newton tracking (no derivative information) is O(tau): MSSRE "
                   f"{big.loc['newton_tracking_causal', 'mssre_med']:.1e}.")
    uns = st[st.unsettled & np.isfinite(st.mssre_med)]
    if len(uns):
        out.append("Steady window not settled (settle ratio > 2, MSSRE contains a decaying transient) for: "
                   + ", ".join(f"{r.label}@n={int(r.n)}" for _, r in uns.iterrows()) + ".")
    for e in sorted(lab for lab in set(st.label) if _is_extra(lab)):
        ge = st[st.label == e].sort_values("n")
        if len(ge) and (ge.n_div == ge.n_traj).all() and ge.n_traj.sum() > 0:
            why = (" (activation (13) contains exp(||E||^2/2): overflow at the first step from U[-2, 2] starts; "
                   "unit-dependent, see lcc_znn_linear for the same model with linear phi)" if e == "lcc_znn" else "")
            out.append(f"{e} diverges for every n and trajectory at h = {float(ge.h.iloc[0]):g}{why}.")
            continue
        g = ge[np.isfinite(ge.cost_med) & np.isfinite(ge.mssre_med)]
        if len(g):
            r = g.iloc[-1]
            z = st[(st.n == r.n) & (st.label == "znn")]
            zc = float(z.cost_med.iloc[0]) if len(z) else math.nan
            out.append(f"{e}: {r.cost_med / zc:.1f}x the ZNN-Euler cost at n = {int(r.n)}, MSSRE {r.mssre_med:.1e}"
                       + (f" ({int(ge.n_div.sum())} diverged trajectories)" if ge.n_div.sum() else "") + ".")
    text = [" ".join(out), ""]
    if scale_name == "mini":
        text.append("(Mini-scale numbers come from a validation run with fewer seeds, a shorter horizon and fewer "
                    "repeats, possibly on a loaded machine; they check the pipeline and are not paper results.)")
    return text


def summarize(st, rat, ct, K, sct, kt, acc, tim, args, scale_name):
    nmax = int(st.n.max())
    big = st[st.n == nmax].set_index("label")
    front = {int(n): g[g.pareto].sort_values("cost_med").label.tolist() for n, g in st.groupby("n")}
    front_p = {int(n): g[g.pareto_protocol].sort_values("cost_med").label.tolist() for n, g in st.groupby("n")}
    contrasts = []
    for _, r in (ct[ct.n == nmax] if len(ct) else ct).iterrows():
        contrasts.append({"a": r.a, "b": r.b, "desc": r.desc,
                          "cost_ratio": r.cost_ratio, "cost_ci95": list(r.cost_ratio_ci),
                          "cost_ci_bonferroni": list(r.cost_ratio_ci_bonf),
                          "mssre_ratio": r.mssre_ratio, "mssre_ci95": list(r.mssre_ratio_ci),
                          "mssre_ci_bonferroni": list(r.mssre_ratio_ci_bonf),
                          "flops_ratio": r.flops_ratio})
    diverged = st[st.n_div > 0][["n", "label", "n_div", "n_traj"]].to_dict("records")
    unsettled = st[st.unsettled][["n", "label", "settle_med", "frac_unsettled", "mssre_med", "mssre_w2_med"]
                                 ].to_dict("records")
    probe = probe_summary(tim)
    censoring = {
        "rule": "diverged trajectory = MSSRE +inf (worst; enters medians, bootstrap and Kendall ranks as "
                "+inf); error/timeout = missing (NaN, reported as n_fail); timing runs with a diverged "
                "trajectory are excluded from the per-step cost (reported as n_timing_div); censored cells "
                "are drawn in the 'diverged' strip of the Pareto figures at their own or a proxy cost",
        "censored_cells": st[st.censored][["n", "label", "n_div", "n_traj", "cost_plot_src"]].to_dict("records"),
        "partially_diverged": st[(st.n_div > 0) & ~st.censored][["n", "label", "n_div", "n_traj"]].to_dict("records"),
        "failed": st[st.n_fail > 0][["n", "label", "n_fail"]].to_dict("records"),
        "timing_diverged": st[st.n_timing_div > 0][["n", "label", "n_timing_div", "n_timing_rows"]].to_dict("records"),
        "not_timed": sorted(set(st[~np.isfinite(st.cost_med)].label)),
    }
    ov = st[np.isfinite(st.overhead_share) & st.label.isin(C.CORE_LABELS)]
    overhead = {int(n): {"median": float(g.overhead_share.median()), "min": float(g.overhead_share.min()),
                         "max": float(g.overhead_share.max())} for n, g in ov.groupby("n")}
    summary = {
        "experiment": "E7", "scale": scale_name, "problem": C.PROBLEM, "kappa": C.KAPPA, "tau": C.TAU,
        "h": C.H_MAIN, "h_gnn": C.H_GNN, "h_tuned": {"znn": C.H_EULER_TUNED, "znn_explicit_inverse": C.H_EULER_TUNED,
                                                     "gnn": C.H_GNN_TUNED},
        "h_uhlig_3_4": float(st[st.label == "uhlig_3_4"].h.iloc[0]) if (st.label == "uhlig_3_4").any() else None,
        "n_values": sorted(int(v) for v in st.n.unique()),
        "methods": sorted(st.label.unique().tolist()),
        "extras_registered": sorted(lab for lab in set(st.label) if _is_extra(lab)),
        "n_accuracy_rows": int(len(acc)), "accuracy_status": acc.status.value_counts().to_dict() if len(acc) else {},
        "n_timing_rows": int(len(tim)), "timing_status": tim.status.value_counts().to_dict() if len(tim) else {},
        "single_thread_check": {"cpu_over_wall_median": float(tim.cpu_over_wall.median()) if len(tim) else None,
                                "cpu_over_wall_max": float(tim.cpu_over_wall.max()) if len(tim) else None,
                                "loadavg1_median": float(tim.loadavg1.median()) if len(tim) else None},
        "costs_py": COSTS_NOTE["costs_py"],
        "flops_exec_over_costs_py": _xcheck(st),
        "pareto_front_by_n": front,
        "pareto_front_protocol_by_n": front_p,
        f"at_n{nmax}": {
            "cost_us": {lab: big.loc[lab, "cost_med"] * 1e6 for lab in big.index},
            "mssre": {lab: big.loc[lab, "mssre_med"] for lab in big.index},
            "pareto_freq": {lab: big.loc[lab, "pareto_freq"] for lab in big.index},
            "pareto_freq_protocol": {lab: big.loc[lab, "pareto_freq_protocol"] for lab in big.index},
        },
        "contrasts_at_nmax": contrasts, "bonferroni_K": K,
        "diverged": diverged, "unsettled": unsettled, "censoring": censoring,
        "machine_probe": probe,
        "overhead_share_core_by_n": overhead,
        "kendall_p_values": "descriptive only (k <= 8-15 items per n, no multiplicity control)",
        "scaling": sct.to_dict("records"),
        "kendall": kt.to_dict("records"),
        "peak_rss_mb_max": float(st.peak_rss_mb.max()),
        "bootstrap": {"resamples": args.n_boot, "seed": args.seed,
                      "scheme": "paired two-stage cluster (seed, then round / initial state)"},
    }
    summary = _jsonable(summary)
    out = Path(args.results)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))

    def g(lab, col, n=nmax):
        r = st[(st.n == n) & (st.label == lab)]
        return float(r[col].iloc[0]) if len(r) else math.nan

    def cline(a, b, text):
        c = _c(ct, a, b, nmax)
        if c is None:
            return None
        return (f"* {text}: cost {c.cost_ratio:.2f}x (95% CI {c.cost_ratio_ci[0]:.2f}-{c.cost_ratio_ci[1]:.2f}; "
                f"Bonferroni {c.cost_ratio_ci_bonf[0]:.2f}-{c.cost_ratio_ci_bonf[1]:.2f}), MSSRE {c.mssre_ratio:.2g}x "
                f"(95% CI {c.mssre_ratio_ci[0]:.2g}-{c.mssre_ratio_ci[1]:.2g}).")

    lines = [f"# E7 summary ({scale_name} scale)", "",
             f"Cost-accuracy Pareto on `matrix_inversion` (kappa = {C.KAPPA:g}, tau = {C.TAU:g}; protocol arm h = "
             f"{C.H_MAIN}, tuned arm h = {C.H_EULER_TUNED:g} for ZNN-Euler / explicit-inverse ZNN and "
             f"{C.H_GNN_TUNED} for GNN), n in {summary['n_values']}; cost = median wall-clock per step inside "
             f"`disc.step` (B = 1, one BLAS thread, data generator excluded, methods interleaved per round); "
             f"accuracy = MSSRE (median over trajectories). Ratios a/b of medians with paired two-stage cluster "
             f"bootstrap CIs; Bonferroni over K = {K} contrasts.", "",
             "## Headline numbers", "",
             f"* Pareto front at n = {nmax} (cheapest first): protocol arm {', '.join(front_p[nmax])}; "
             f"all ranked methods {', '.join(front[nmax])}."]
    for a, b, text in [("znn_explicit_inverse", "znn", "Explicit-inverse vs LU ZNN-Euler (h = 0.3)"),
                       ("pc_newton_causal", "znn", "Causal PC-Newton vs ZNN-Euler at h = 0.3 (gain-confounded)"),
                       ("pc_newton_causal", "znn_tuned", "Causal PC-Newton vs ZNN-Euler at tuned h = 1 (fair)"),
                       ("znn_explicit_inverse_tuned", "znn_tuned", "Explicit-inverse vs LU ZNN-Euler (both h = 1)"),
                       ("uhlig_3_4", "znn", f"ZNN Uhlig 3_4 (h = {summary['h_uhlig_3_4']}) vs ZNN-Euler (h = 0.3)"),
                       ("znn_tuned", "znn", "A/A control: ZNN-Euler h = 1 vs h = 0.3 (same operations)"),
                       ("gnn_tuned", "gnn", "A/A control: GNN tuned vs GNN (same operations)")]:
        ln = cline(a, b, text)
        if ln:
            lines.append(ln)
    lines += [f"* GNN (h sigma_max^2 ~ 1) costs {g('gnn', 'cost_med') * 1e6:.1f} us/step with lag-floor MSSRE "
              f"{g('gnn', 'mssre_med'):.2e} (tuned h: {g('gnn_tuned', 'mssre_med'):.2e}); at the matched h = 0.3 it "
              f"diverges ({int(st[st.label == 'gnn_matched'].n_div.sum())} diverged trajectories over all n).",
              f"* Newton tracking (no prediction) is O(tau): MSSRE {g('newton_tracking_causal', 'mssre_med'):.2e}; "
              f"Davidenko without correction keeps the initial residual: {g('davidenko', 'mssre_med'):.2e}.",
              f"* Steady-window check: {len(unsettled)} (n, method) cells with settle ratio > {C.SETTLE_TOL:g}"
              + (": " + ", ".join(f"{u['label']}@{u['n']}" for u in summary["unsettled"]) if unsettled else "") + ".",
              f"* Single-thread check: process CPU / wall = {summary['single_thread_check']['cpu_over_wall_median']:.3f} "
              f"(max {summary['single_thread_check']['cpu_over_wall_max']:.3f}); load average during timing "
              f"{summary['single_thread_check']['loadavg1_median']:.1f}.",
              (f"* Matrix-unit contention probe (128x128 dgemm per round, {probe['rounds']} rounds): best "
               f"{probe['gemm_us_min']:.1f} us ({probe['gemm_gflops_best']:.0f} GFLOP/s), median "
               f"{probe['gemm_ratio_median']:.2f}x best, worst {probe['gemm_ratio_max']:.2f}x; "
               f"{100 * probe['frac_rounds_contended']:.0f}% of rounds > {probe['flag']:g}x best"
               + (" -- timings were taken under contention (GEMM-heavy methods inflated; re-time alone, 1 worker)."
                  if probe["frac_rounds_contended"] > 0.1 else ".")) if probe else
              "* Matrix-unit contention probe: absent (timing format < 3).",
              f"* znnbench/costs.py: {summary['costs_py']}.", "",
              "## Censoring", "",
              "Diverged trajectories are MSSRE = +inf (worst case: they enter medians, the paired bootstrap and "
              "the Kendall ranks; a median of +inf marks a censored cell); errors/timeouts are missing (NaN) and "
              "counted; timing runs with a diverged trajectory give no per-step cost. Censored cells are drawn in "
              "the 'diverged' strip of the Pareto figures (at a proxy cost when they have no clean timing, "
              "marked with a double dagger), never dropped.", ""]
    cen = summary["censoring"]
    for key_, title in (("censored_cells", "censored (median MSSRE = inf)"),
                        ("partially_diverged", "partially diverged (median finite)"),
                        ("failed", "failed (error/timeout)"), ("timing_diverged", "timing runs diverged")):
        items = cen.get(key_) or []
        if key_ == "censored_cells":
            txt = "; ".join(f"{u['label']}@{u['n']} {u['n_div']}/{u['n_traj']}"
                            + (f" (cost {u['cost_plot_src']})" if str(u.get('cost_plot_src', '')).startswith('proxy') else "")
                            for u in items)
        elif key_ == "partially_diverged":
            txt = "; ".join(f"{u['label']}@{u['n']} {u['n_div']}/{u['n_traj']}" for u in items)
        elif key_ == "failed":
            txt = "; ".join(f"{u['label']}@{u['n']} {u['n_fail']}" for u in items)
        else:
            txt = "; ".join(f"{u['label']}@{u['n']} {u['n_timing_div']}/{u['n_timing_rows']}" for u in items)
        lines.append(f"* {title}: {txt or 'none'}.")
    lines.append(f"* not timed or no clean timing: {', '.join(cen['not_timed']) or 'none'}.")
    lines += ["", "## Python / call overhead", "",
              "Per-step time at the smallest n (< 1e3 FLOPs) is pure interpreter and LAPACK/numpy call overhead; "
              "overhead share = that floor / cost(n) (an estimate: allocation and copies grow ~n^2). Core methods:"]
    for n_, o in summary["overhead_share_core_by_n"].items():
        lines.append(f"* n = {n_}: overhead share median {100 * o['median']:.0f}% (range {100 * o['min']:.0f}-"
                     f"{100 * o['max']:.0f}%)" + (" -- wall-clock differences here measure the implementation, "
                                                   "not the algorithm" if o["median"] >= 0.5 else "") + ".")
    lines += ["", "## Confounds (see experiments/E7_NOTES.md, 'Confounds')", "",
              "* Gain: equal nominal h = gamma tau = 0.3 is not equal effective gain. GNN's effective Euler step is "
              "h sigma_max^2 (19-43 at h = 0.3: diverges; protocol h = 0.007, tuned 0.0111 = 0.8 h_max); the GNN-ZNN "
              "hybrids need h (1 + sigma_max^2) < 2 (run at 0.007); one Newton correction is a dead-beat (h = 1) "
              "step, so PC-Newton is compared with ZNN-Euler at h = 1 too (tuned arm); Uhlig 3_4 and CDTZNN run at "
              "their h_opt because their stable interval ends below 0.3.",
              "* Information/initial state: the explicit-inverse ZNN starts from the Newton-Schulz guess "
              "(uses A(0); local method), all others from U[-2, 2]; the steady-state MSSRE of the linear designs does "
              "not depend on x0, the settle ratio checks that the window is steady.",
              "* Unit dependence: LCCZNN's exp-sig activation (13) contains exp(||E||^2/2) and FA-GNN-ZNN's fuzzy "
              "factor uses absolute breakpoints (u_max = 1); their outcomes (overflow at step 1; gain boost) depend on "
              "the unit of E. All other methods are linear in E (unit-free up to the round-off floor).",
              "* Timing: B = 1, one BLAS thread, data generator excluded; at small n the time is interpreter "
              "overhead (above); concurrent processes inflate GEMM-heavy methods via the shared matrix unit "
              "(probe above), which CPU/wall cannot detect.",
              "* Pseudo-replication: the B initial states of a seed share A(t); CIs resample seeds first.", "",
              "## Scaling of wall-clock per step with n (log-log slope)", "",
              f"Asymptotic range: tail with consecutive local slopes within {SLOPE_TOL} of the last one "
              "(2-pt = two-point fit, not an exponent estimate); floor = per-step time at the smallest n.", "",
              "| method | all n | asymptotic | 95% CI | range | last local slope | pre-asymptotic | floor [us] | "
              "overhead at n_max |",
              "|---|---|---|---|---|---|---|---|---|"]
    for _, r in sct.iterrows():
        osh = float(r.get("overhead_share_nmax", math.nan))
        lines.append(f"| {r.label} | {r.slope_full:.2f} | {r.slope_asym:.2f} | "
                     f"[{r.slope_asym_ci[0]:.2f}, {r.slope_asym_ci[1]:.2f}] | {r.asym_n}"
                     + (" 2-pt" if r.get("two_point_fit", False) else "") + " | "
                     f"{r.local_slope_last:.2f} | {'yes' if r.pre_asymptotic else 'no'} | "
                     f"{float(r.get('floor_us', math.nan)):.1f} | "
                     + (f"{100 * osh:.0f}%" if np.isfinite(osh) else "--") + " |")
    lines += ["", "## Ranking agreement (Kendall tau_b, 8 core methods; last column all ranked methods)", "",
              "p-values (summary.json) are descriptive only (k <= 15 items per n, no multiplicity control); "
              "MSSRE ranks are censoring-aware (diverged = +inf = worst, tied).", "",
              "| n | time vs costs.py FLOPs | vs ideal FLOPs | vs executed FLOPs | vs runner wall | B=1 vs B=4 | "
              "MSSRE vs n_max | costs.py FLOPs, all (k) |",
              "|---|---|---|---|---|---|---|---|"]
    for _, r in kt.iterrows():
        lines.append(f"| {int(r.n)} | {r.tau_flops_costs:.2f} | {r.tau_flops_ideal:.2f} | {r.tau_flops_exec:.2f} | "
                     f"{r.tau_e2e_runner:.2f} | {r.tau_clean_B4:.2f} | {r.tau_acc_vs_nmax:.2f} | "
                     f"{r.tau_flops_costs_all:.2f} ({int(r.k_all)}) |")
    lines += ["", "## Interpretation", ""] + interpretation(st, ct, sct, kt, nmax, scale_name)
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    return summary


# ----------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True, help="E7 result folder (with accuracy/ and timing/)")
    ap.add_argument("--figdir", required=True)
    ap.add_argument("--tabdir", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261018)
    args = ap.parse_args(argv)
    res = Path(args.results)
    style.FIG_DIR = Path(args.figdir)
    acc = load_accuracy(res / "accuracy")
    tim = load_timing(res / "timing")
    if tim.empty:
        raise SystemExit(f"no timing results under {res / 'timing'}")
    scale_name = (str(acc.config.iloc[0]).split("_")[-1] if len(acc) else str(tim.scale.iloc[0]))
    st, boots = build_stats(acc, tim, args.n_boot, args.seed)
    st = add_front(st, boots, args.n_boot, _ranked, "")                                   # all ranked methods
    st = add_front(st, boots, args.n_boot, lambda lab: lab in C.CORE_LABELS, "_protocol")  # protocol arm (core)
    st["flops_pareto"] = False
    for n, g in st.groupby("n"):
        g = g[g.label.isin(C.CORE_LABELS)]
        st.loc[g.index, "flops_pareto"] = pareto_mask(g.flops_axis, g.mssre_med)
    rat = ratios(st, boots)
    ct, K = contrast_table(st, boots)
    sct = scaling_table(st, boots, args.n_boot)
    kt = kendall_table(st)
    st.to_csv(res / "E7_stats.csv", index=False)
    ct.to_csv(res / "E7_contrasts.csv", index=False)

    core = [lab for lab in keys.GROUPS["baseline"] if lab in set(st.label)]
    # censored cells are drawn in a "diverged" strip (never dropped): the matched-gain GNN in the
    # core figures; the matched hybrids and the exp-sig LCCZNN in the extras figure
    cens_core = [lab for lab in ("gnn_matched",) if lab in set(st.label)]
    fig_pareto(st, core, "fig_E7_pareto", xlabel="wall-clock per step [$\\mu$s]", front_col="pareto_protocol",
               censored=cens_core)
    gain = [lab for lab in GAIN_SERIES if lab in set(st.label)]
    if any(lab in C.TUNED_LABELS for lab in gain):
        sub = st[st.label.isin(gain + cens_core)].copy()
        sub["pareto_gain"] = False
        for n, g in sub.groupby("n"):
            sub.loc[g.index, "pareto_gain"] = pareto_mask(g.cost_med, g.mssre_med)
        fig_pareto(sub, gain, "fig_E7_pareto_gain", xlabel="wall-clock per step [$\\mu$s]", front_col="pareto_gain",
                   arrows=[(b, t) for t, b in C.TUNED_LABELS.items()], censored=cens_core)
    fsrc = ("analytic FLOPs per step / $n^3$ (znnbench.costs)" if (st.flops_axis_src == "costs.py").all()
            else "FLOPs per step / $n^3$ (costs.py, else idealised)")
    fig_pareto(st, core, "fig_E7_pareto_flops", cost_col="flops_axis", xlabel=fsrc, per_n3=True,
               errors=False, front_col="flops_pareto", censored=cens_core)
    fig_scaling(st, rat, core, "fig_E7_scaling", probe=probe_summary(tim))
    extras_all = sorted(lab for lab in set(st.label) if _is_extra(lab))
    extras = [e for e in extras_all if (np.isfinite(st[st.label == e].mssre_med)
                                        & np.isfinite(st[st.label == e].cost_med)).any()]
    # extras that never give a finite point (diverged everywhere) + the matched-gain hybrids
    cens_ex = [lab for lab in extras_all if lab not in extras] + sorted(
        lab for lab in set(st.label) if C.is_matched(lab) and lab not in cens_core)
    refs = ["znn", "znn_explicit_inverse"]
    for k in range(0, len(extras), 6):          # <= 8 series per panel (2 references + 6)
        ex = extras[k:k + 6]
        ce = cens_ex if k == 0 else []
        n_ex = st[st.label.isin(ex) & np.isfinite(st.mssre_med)].n.unique()
        sub = st[st.label.isin(refs + ex + ce) & st.n.isin(n_ex)].copy()
        sub["pareto_x"] = False
        for n, g in sub.groupby("n"):
            sub.loc[g.index, "pareto_x"] = pareto_mask(g.cost_med, g.mssre_med)
        fig_pareto(sub, refs + ex, "fig_E7_pareto_extra" + ("" if k == 0 else f"_{k // 6 + 1}"),
                   xlabel="wall-clock per step [$\\mu$s]", extras_order=ex, front_col="pareto_x", censored=ce)
    write_tables(st, rat, ct, K, sct, kt, Path(args.tabdir), scale_name)
    summ = summarize(st, rat, ct, K, sct, kt, acc, tim, args, scale_name)
    print(json.dumps({k: summ[k] for k in ("scale", "n_values", "pareto_front_by_n", "pareto_front_protocol_by_n",
                                           "costs_py")}, indent=1))
    print(f"wrote {res / 'summary.json'}, {res / 'SUMMARY.md'}, figures -> {style.FIG_DIR}, tables -> {args.tabdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
