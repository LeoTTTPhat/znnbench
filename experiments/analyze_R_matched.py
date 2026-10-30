#!/usr/bin/env python
"""Analysis of the matched-effort arm R_matched of the robotics case study
(protocol: experiments/configs/R_matched.py, experiments/tasks/R_matched_common.py,
experiments/R_NOTES.md section "Matched-effort arm (2026-10-26)").

    python experiments/analyze_R_matched.py --results results/R_matched --rbase results/R --figdir figures

Reads the task logs <results>/results/*.json, their per-trajectory rows
(*.parquet), and the published R results under <rbase> (R_off, R_units, R_on,
loaded with analyze_R.load) for the sanity reproduction, and writes

  <results>/SUMMARY.md           tables per robot / path / rate / error unit
  <results>/summary.json         every number of SUMMARY.md (and more)
  <results>/R_matched_table.tex  compact LaTeX table for Supplementary Section S5
  <figdir>/fig_R_matched.pdf/png accuracy-effort curves (MSSRE vs median peak joint speed)

Statistics as in analyze_R: medians over the 12 trajectories of a cell;
MSSRE ratios are ratios of medians with a paired percentile bootstrap interval
(2000 resamples, pairs = same start; descriptive, because the paths are
deterministic and the 12 starts are the only source of variation).
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

from experiments import analyze_R as AR  # noqa: E402  (statistics, entity styles; read-only use)
from experiments import style  # noqa: E402
from experiments.tasks import R_common as RC  # noqa: E402
from experiments.tasks import R_matched_common as MC  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROBOTS = ["puma560", "panda"]
PATHS = ["circle", "lissajous"]
RATES = [100, 333, 1000]
UNITS = ["m", "mm"]
RNAME = {"puma560": "PUMA 560", "panda": "Panda"}
PNAME = {"circle": "circle", "lissajous": "Lissajous"}
MNAME = {"qd_peak": "peak joint speed", "qd_rms": "RMS joint speed"}
CHECK_COLS = ["mssre", "mean_ssre", "peak_rate", "peak_rate_ss", "qd_peak", "qd_util", "qd_util_ss", "qd_viol_frac",
              "qdd_peak", "alt_active", "final_residual", "max_residual", "t_eps_0.001", "t_eps_0.0001"]


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def _num(v):
    if v is None:
        return np.nan
    if isinstance(v, str):
        return {"inf": np.inf, "-inf": -np.inf}.get(v, np.nan)
    return float(v)


def load_matched(results):
    logs, frames = [], []
    for f in sorted((Path(results) / "results").glob("R_matched-*.json")):
        L = json.loads(f.read_text())
        df = pd.read_parquet(f.with_name(L["rows_file"]))
        df["task"] = f.stem
        L["task"] = f.stem
        L["robot_short"] = L["robot"].replace("robot_", "")
        logs.append(L)
        frames.append(df)
    if not frames:
        raise SystemExit(f"no R_matched results under {results}")
    return logs, pd.concat(frames, ignore_index=True)


def sel(df, **k):
    m = np.ones(len(df), bool)
    for a, b in k.items():
        if a == "gamma":
            m &= np.isclose(df[a].to_numpy(float), float(b), rtol=1e-12, atol=0)
        else:
            m &= (df[a] == b).to_numpy()
    return df[m]


# --------------------------------------------------------------------------
# statistics helpers
# --------------------------------------------------------------------------
def med(x):
    return AR.med_iqr(np.asarray(x, float))[0]


def compare(a, b, seed):
    """Paired ratio of medians MSSRE(a) / MSSRE(b) (pairs = same start)."""
    A, Bq = a.set_index("pair").sort_index(), b.set_index("pair").sort_index()
    com = A.index.intersection(Bq.index)
    pt, lo, hi = AR.boot_ratio_ci(A.loc[com].mssre_mm.astype(float), Bq.loc[com].mssre_mm.astype(float), seed)
    return {"n_pairs": int(len(com)), "mssre_a_mm": med(A.loc[com].mssre_mm), "mssre_b_mm": med(Bq.loc[com].mssre_mm),
            "ratio": pt, "ratio_ci": [lo, hi],
            "factor": (1.0 / pt) if pt and np.isfinite(pt) else np.nan,
            "factor_ci": [1.0 / hi if hi else np.nan, 1.0 / lo if lo else np.nan]}


def effort(c):
    """Effort and limit statistics of a run set (12 trajectories)."""
    return {"n": int(len(c)), "n_diverged": int((c.status == "diverged").sum()),
            "qd_peak": med(c.qd_peak), "qd_rms": med(c.qd_rms), "qd_peak_late": med(c.qd_peak_late),
            "qd_first": med(c.qd_first), "qd_peak_early": med(c.qd_peak_early),
            "frac_peak_first_step": float(np.mean(c.k_qd_peak.to_numpy(float) == 0)),
            "frac_peak_before_0p5s": float(np.mean(c.t_qd_peak.to_numpy(float) < MC.EARLY_T)),
            "qd_util_median": med(c.qd_util), "qd_util_max": float(np.nanmax(c.qd_util.to_numpy(float))),
            "frac_over_limit": float(np.mean(c.qd_util.to_numpy(float) > 1.0)),
            "qd_viol_frac_median": med(c.qd_viol_frac),
            "mssre_mm": med(c.mssre_mm), "alt_active": med(c.alt_active)}


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------
def analyse(results, rbase):
    logs, rows = load_matched(results)
    S = {"n_tasks": len(logs), "n_rows": int(len(rows)),
         "status_rows": rows.status.value_counts().to_dict(),
         "wall_s_total": float(sum(L["wall_s"] for L in logs)), "cpu_s_total": float(sum(L["cpu_s"] for L in logs)),
         "wall_s_max_task": float(max(L["wall_s"] for L in logs)),
         "peak_rss_mb_max": float(max(L["peak_rss_mb"] for L in logs)),
         "n_simulate_calls": int(sum(len(L["calls"]) for L in logs)),
         "settings": logs[0]["settings"]}
    # ---- sanity: gamma = 10 reference runs vs the published R rows -----------------
    data = AR.load(rbase)
    off = data["R_off"]
    off = off[(off.noise == "none") & (off.gamma == 10.0)]
    units = data.get("R_units")
    units = units[units.gamma == 10.0] if units is not None else None
    san = []
    for L in logs:
        r, p, rt, u = L["robot_short"], L["path"], L["rate"], L["unit"]
        for e in MC.ENTRIES:
            a = sel(rows, task=L["task"], role="ref", entry=e)
            base = units if (e == "znn_sbp_ie" and u == "mm") else off
            b = sel(base, robot=r, path=p, rate=rt, entry=e)
            A, Bq = a.set_index("pair").sort_index(), b.set_index("pair").sort_index()
            com = A.index.intersection(Bq.index)
            d = {"robot": r, "path": p, "rate": rt, "unit": u, "entry": e,
                 "source": "R_units" if base is units else "R_off", "n_pairs": int(len(com))}
            worst = 0.0
            for col in CHECK_COLS:
                if col not in A or col not in Bq:
                    continue
                x, y = A.loc[com, col].to_numpy(float), Bq.loc[com, col].to_numpy(float)
                ok = np.isfinite(x) & np.isfinite(y)
                if (np.isfinite(x) != np.isfinite(y)).any():
                    worst = np.inf
                if ok.any():
                    worst = max(worst, float(np.max(np.abs(x[ok] - y[ok]) / np.maximum(np.abs(y[ok]), 1e-300))))
            d["max_rel_diff"] = worst
            san.append(d)
    S["sanity"] = san
    S["sanity_max_rel_diff"] = float(max(d["max_rel_diff"] for d in san))
    S["sanity_n_pairs_min"] = int(min(d["n_pairs"] for d in san))
    # wrapper consistency: EffortStats' own peak == JointStats' qd_peak == simulate's peak_rate
    fin = rows[np.isfinite(rows.qd_peak.astype(float))]
    S["check_qd_peak_chk"] = float(np.max(np.abs(fin.qd_peak_chk - fin.qd_peak) / np.maximum(fin.qd_peak, 1e-300)))
    S["check_qd_peak_vs_peak_rate"] = float(np.max(np.abs(fin.qd_peak - fin.peak_rate) / np.maximum(fin.peak_rate, 1e-300)))
    # linear-in-E entries are simulated in m in both unit tasks: their ref rows must coincide
    inv = []
    for r in ROBOTS:
        for p in PATHS:
            for rt in RATES:
                for e in MC.UNIT_INVARIANT:
                    a = sel(rows, robot=r, path=p, rate=rt, unit="m", role="ref", entry=e).set_index("pair").sort_index()
                    b = sel(rows, robot=r, path=p, rate=rt, unit="mm", role="ref", entry=e).set_index("pair").sort_index()
                    if len(a) and len(b):
                        inv.append(float(np.max(np.abs(a.mssre - b.loc[a.index].mssre) / np.maximum(a.mssre, 1e-300))))
    S["ref_m_vs_mm_task_max_rel_diff"] = max(inv) if inv else None

    # ---- per-cell comparisons ------------------------------------------------------
    cells = []
    seed = 100
    for L in logs:
        r, p, rt, u = L["robot_short"], L["path"], L["rate"], L["unit"]
        T = sel(rows, task=L["task"])
        ref = {e: sel(T, role="ref", entry=e) for e in MC.ENTRIES}
        c = {"robot": r, "path": p, "rate": rt, "unit": u, "task": L["task"],
             "targets": {k: {m: _num(v) for m, v in d.items()} for k, d in L["targets"].items()},
             "ref": {e: effort(ref[e]) for e in MC.ENTRIES}}
        seed += 10
        c["nominal"] = {"lin": compare(ref["znn_sbp_ie"], ref["znn_lin"], seed + 1),
                        "ie": compare(ref["znn_sbp_ie"], ref["ieznn"], seed + 2)}
        c["search"] = {}
        for role, s in L["searches"].items():
            c["search"][role] = {}
            for m, res in s["results"].items():
                d = {"status": res["status"], "gamma_star": _num(res.get("gamma_star")),
                     "target": _num(res["target"]), "f_star": _num(res.get("f_star")),
                     "mismatch": _num(res.get("mismatch")), "bracket": [_num(x) for x in res.get("bracket", [])],
                     "bracket_rel_width": _num(res.get("bracket_rel_width")),
                     "grid_monotone": res.get("grid_monotone"), "n_gains_evaluated": res.get("n_gains_evaluated"),
                     "curve": [[_num(g), _num(f)] for g, f in res.get("curve", [])]}
                if res["status"] in ("matched", "loose") and np.isfinite(d["gamma_star"]):
                    mr = sel(T, role=role, gamma=d["gamma_star"])
                    d["h_star"] = d["gamma_star"] / rt
                    d["effort"] = effort(mr)
                    seed += 1
                    if role == "A_lin":
                        d["cmp"] = compare(ref["znn_sbp_ie"], mr, seed)       # SBP(10) / linear(gamma*)
                    elif role == "A_ie":
                        d["cmp"] = compare(ref["znn_sbp_ie"], mr, seed)       # SBP(10) / IEZNN(gamma*)
                    else:
                        d["cmp"] = compare(mr, ref["znn_lin"], seed)          # SBP(gamma*) / linear(10)
                c["search"][role][m] = d
        # empirical gain exponents p of MSSRE ~ gamma^p at fixed tau (two points: gamma = 10 and gamma*)
        c["gain_exponent"] = {}
        for role, e in (("A_lin", "znn_lin"), ("A_ie", "ieznn"), ("B_sbp", "znn_sbp_ie")):
            d = c["search"][role].get("qd_peak", {})
            if "effort" in d:
                m10, ms = c["ref"][e]["mssre_mm"], d["effort"]["mssre_mm"]
                c["gain_exponent"][role] = float(math.log(ms / m10) / math.log(d["gamma_star"] / MC.GAMMA_REF))
        # direction B: margin of the target (linear ZNN's peak at 10) above the late-window level (path floor)
        c["B_target_over_floor"] = float(c["targets"]["B"]["qd_peak"] / c["ref"]["znn_lin"]["qd_peak_late"] - 1.0)
        # non-monotonicity of the median-effort curves of the searched entries (max relative drop with increasing gain)
        c["max_rel_drop"] = {}
        for role in ("A_lin", "A_ie", "B_sbp"):
            cur = np.array(c["search"][role]["qd_peak"]["curve"], float)
            dd = np.diff(cur[:, 1]) / cur[1:, 1]
            c["max_rel_drop"][role] = float(max(0.0, -dd.min())) if len(dd) else 0.0
        # late-window (steady-state) peaks of every run set of the cell
        sets = {"lin_10": ref["znn_lin"], "ie_10": ref["ieznn"], "sbp_10": ref["znn_sbp_ie"]}
        for role, key in (("A_lin", "lin_A"), ("A_ie", "ie_A"), ("B_sbp", "sbp_B")):
            d = c["search"].get(role, {}).get("qd_peak")
            if d and "effort" in d:
                sets[key] = sel(T, role=role, gamma=d["gamma_star"])
        lp = {k: med(v.qd_peak_late) for k, v in sets.items()}
        c["late_peak"] = lp
        c["late_peak_spread"] = float(max(lp.values()) / min(lp.values()) - 1.0)
        # late-window peak of linear ZNN over the whole searched gain range (does it constrain gamma?)
        al = sel(T, role="A_lin")
        lpg = al.groupby("gamma").qd_peak_late.median()
        c["late_peak_lin_range"] = [float(lpg.min()), float(lpg.max())] if len(lpg) else None
        c["late_peak_lin_gamma_range"] = [float(lpg.index.min()), float(lpg.index.max())] if len(lpg) else None
        cells.append(c)
    S["cells"] = cells

    # ---- on-path starts (R_on): is the effort excess caused by the start offset? -----
    on = data.get("R_on")
    onp = []
    if on is not None:
        o = on[(on.noise == "none") & (on.gamma == 10.0)]
        for r in ROBOTS:
            for p in PATHS:
                for rt in RATES:
                    d = {"robot": r, "path": p, "rate": rt}
                    for e in MC.ENTRIES:
                        x = sel(o, robot=r, path=p, rate=rt, entry=e)
                        d[e] = med(x.peak_rate) if len(x) else np.nan
                    onp.append(d)
    S["on_path_peak_g10"] = onp

    # ---- ranges over the 4 robot x path cells, per unit and rate ------------------------
    hws = []
    for c in cells:
        cmps = [c["nominal"]["lin"], c["nominal"]["ie"]] + [c["search"][r]["qd_peak"]["cmp"]
                                                             for r in ("A_lin", "A_ie", "B_sbp")
                                                             if "cmp" in c["search"][r]["qd_peak"]]
        for x in cmps:
            lo, hi = x["ratio_ci"]
            hws.append(max(hi / x["ratio"] - 1.0, 1.0 - lo / x["ratio"]))
    S["ci_max_rel_halfwidth_peak"] = float(max(hws))
    S["ranges"] = ranges(cells)
    S["limit"] = limit_summary(cells)
    S["search_quality"] = search_quality(cells)
    return S, rows


def _cells(cells, **k):
    return [c for c in cells if all(c[a] == b for a, b in k.items())]


def _rng(vals):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    return [float(min(v)), float(max(v))] if v else [np.nan, np.nan]


def ranges(cells):
    out = {}
    for u in UNITS:
        for rt in RATES:
            cs = _cells(cells, unit=u, rate=rt)
            if not cs:
                continue
            d = {"n_cells": len(cs),
                 "nominal_factor_lin": _rng([c["nominal"]["lin"]["factor"] for c in cs]),
                 "nominal_factor_ie": _rng([c["nominal"]["ie"]["factor"] for c in cs]),
                 "target_A_peak": _rng([c["targets"]["A"]["qd_peak"] for c in cs]),
                 "target_B_peak": _rng([c["targets"]["B"]["qd_peak"] for c in cs]),
                 "B_target_over_floor": _rng([c["B_target_over_floor"] for c in cs]),
                 "peak_ratio_sbp_over_lin_10": _rng([c["ref"]["znn_sbp_ie"]["qd_peak"] / c["ref"]["znn_lin"]["qd_peak"]
                                                     for c in cs]),
                 "sbp10_frac_over_limit": _rng([c["ref"]["znn_sbp_ie"]["frac_over_limit"] for c in cs]),
                 "sbp10_util_max": _rng([c["ref"]["znn_sbp_ie"]["qd_util_max"] for c in cs]),
                 "sbp10_excess": _rng([1 - c["ref"]["znn_sbp_ie"]["qd_peak_late"] / c["ref"]["znn_sbp_ie"]["qd_peak"]
                                       for c in cs]),
                 "sbp10_frac_peak_first_step": _rng([c["ref"]["znn_sbp_ie"]["frac_peak_first_step"] for c in cs])}
            for role in ("A_lin", "A_ie", "B_sbp"):
                d[f"gain_exponent_{role}"] = _rng([c["gain_exponent"].get(role, np.nan) for c in cs])
                d[f"frac_over_limit_{role}"] = _rng([c["search"][role]["qd_peak"]["effort"]["frac_over_limit"]
                                                     for c in cs if "effort" in c["search"][role]["qd_peak"]])
                d[f"util_max_{role}"] = _rng([c["search"][role]["qd_peak"]["effort"]["qd_util_max"]
                                              for c in cs if "effort" in c["search"][role]["qd_peak"]])
            for m in MC.MEASURES:
                for role, key in (("A_lin", "A_lin"), ("A_ie", "A_ie"), ("B_sbp", "B")):
                    ds = [c["search"][role][m] for c in cs if m in c["search"].get(role, {})]
                    ok = [x for x in ds if "cmp" in x]
                    d[f"{key}_{m}_gamma"] = _rng([x["gamma_star"] for x in ok])
                    d[f"{key}_{m}_factor"] = _rng([x["cmp"]["factor"] for x in ok])
                    d[f"{key}_{m}_ratio"] = _rng([x["cmp"]["ratio"] for x in ok])
                    d[f"{key}_{m}_n_matched"] = len(ok)
                    d[f"{key}_{m}_sbp_better"] = int(sum(x["cmp"]["ratio"] < 1 for x in ok))
                    d[f"{key}_{m}_ci_excludes_1"] = int(sum((x["cmp"]["ratio_ci"][1] < 1) or (x["cmp"]["ratio_ci"][0] > 1)
                                                           for x in ok))
            out[f"{u}_{rt}"] = d
    return out


def limit_summary(cells):
    """Joint-speed limit check of the reference and matched run sets."""
    out = []
    for c in cells:
        d = {"robot": c["robot"], "path": c["path"], "rate": c["rate"], "unit": c["unit"],
             "sbp_10": c["ref"]["znn_sbp_ie"]["frac_over_limit"], "lin_10": c["ref"]["znn_lin"]["frac_over_limit"],
             "sbp_10_util_max": c["ref"]["znn_sbp_ie"]["qd_util_max"]}
        for role in ("A_lin", "A_ie", "B_sbp"):
            for m in MC.MEASURES:
                x = c["search"].get(role, {}).get(m)
                if x and "effort" in x:
                    d[f"{role}_{m}"] = x["effort"]["frac_over_limit"]
                    d[f"{role}_{m}_util_max"] = x["effort"]["qd_util_max"]
        out.append(d)
    return out


def search_quality(cells):
    q = {}
    for role in ("A_lin", "A_ie", "B_sbp"):
        for m in MC.MEASURES:
            xs = [c["search"][role][m] for c in cells if m in c["search"].get(role, {})]
            q[f"{role}_{m}"] = {
                "n": len(xs), "status": {s: sum(x["status"] == s for x in xs) for s in sorted({x["status"] for x in xs})},
                "max_abs_mismatch": float(np.nanmax([abs(x["mismatch"]) for x in xs])) if xs else None,
                "max_bracket_rel_width": float(np.nanmax([x["bracket_rel_width"] for x in xs])) if xs else None,
                "all_grids_monotone": bool(all(x["grid_monotone"] for x in xs)) if xs else None}
    return q


# --------------------------------------------------------------------------
# formatting
# --------------------------------------------------------------------------
def g3(x, sig=3):
    """sig significant digits; plain notation (no exponent) for 1e-3 <= |x| < 1e5."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    if isinstance(x, float) and math.isinf(x):
        return "inf"
    x = float(x)
    if x == 0 or not (1e-3 <= abs(x) < 1e5):
        return f"{x:.{sig}g}"
    d = max(0, sig - 1 - int(math.floor(math.log10(abs(x)))))
    s = f"{x:.{d}f}"
    if abs(float(s)) >= 10 ** sig and d == 0:          # rounding carried into a new digit (e.g. 999.7)
        s = f"{x:.0f}"
    return s


def pct(x, d=1):
    return "--" if x is None or not np.isfinite(x) else f"{100 * x:.{d}f} %"


def rng_txt(r, sig=3):
    if r is None or not np.all(np.isfinite(r)):
        return "--"
    a, b = (g3(r[0], sig), g3(r[1], sig))
    return a if a == b else (f"{a} to {b}" if min(r) < 0 else f"{a}-{b}")


def pct_rng(r, d=0):
    if r is None or not np.all(np.isfinite(r)):
        return "--"
    a, b = f"{100 * r[0]:.{d}f}", f"{100 * r[1]:.{d}f}"
    return f"{a} %" if a == b else f"{a}-{b} %"


def ci_txt(c, key="ratio", sig=3):
    lo, hi = c[f"{key}_ci"]
    return f"{g3(c[key], sig)} [{g3(lo, sig)}, {g3(hi, sig)}]"


def cell_key(c):
    return (ROBOTS.index(c["robot"]), PATHS.index(c["path"]), UNITS.index(c["unit"]), RATES.index(c["rate"]))


# --------------------------------------------------------------------------
# SUMMARY.md
# --------------------------------------------------------------------------
def write_summary(S, results):
    cells = sorted(S["cells"], key=cell_key)
    st = S["settings"]
    L = ["# R_matched: matched-effort arm of the robotics case study (2026-10-26)", "",
         "Question (external review of 26 Oct 2026, item 5): implicit sign-bi-power ZNN (SBP, `sign_bi_power` r = 0.5 "
         "stepped by `implicit_euler_err`) is 10-100 times more accurate than linear ZNN at gamma = 10 (error in m) but "
         "commands 3-4 times the peak joint speed.  Is it still more accurate when both use the same control effort?", "",
         "Protocol: the R problem (position-only, unconstrained pseudoinverse tracking, T = 10 s, noise-free), the 12 "
         f"R_off off-path starts (seeds {st['seeds']} x B = {st['B_per_seed']}, theta0 + U[-0.05, 0.05] rad), "
         "100 / 333 / 1000 Hz, error unit m or mm (`ScaledError`, x1000; linear ZNN and IEZNN are linear in E and "
         "therefore simulated in m for both units).  Effort = median over the 12 trajectories of the peak commanded "
         "joint speed max_{k,i} |theta_dot_i| (`qd_peak`, as in R_off).  **Direction A (main):** the gain of linear "
         "ZNN (and of IEZNN, lambda = gamma^2/4) is raised from 10 until its median peak equals that of implicit SBP "
         "at gamma = 10 in the same cell.  **Direction B (check):** the gain of implicit SBP is lowered from 10 until "
         "its median peak equals that of linear ZNN at gamma = 10.  Search on log gamma: 16-gain grid, two 8-gain "
         f"refinements, up to {st['secant_max']} secant steps (all gains of a stage batched in one simulate call); "
         f"a match is accepted at |effort/target - 1| <= {100 * st['accept_tol']:g} %.  Secondary effort measure "
         "(matched the same way): RMS joint speed over the run, `qd_rms` = sqrt(mean_k ||theta_dot_k||_2^2).  "
         "Reported only: the peak over the last 75 % of the run (t >= 2.5 s), `qd_peak_late`.  MSSRE = max ||E|| "
         "over 7.5-10 s (mm).  Ratios are SBP / comparator ratios of medians with a paired bootstrap 95 % interval "
         "(2000 resamples, pairs = same start), **descriptive only** because the paths are deterministic; "
         "'factor' = 1 / ratio = how many times SBP is more accurate (> 1 favours SBP).", "",
         f"Runs: {S['n_tasks']} memguard tasks, {S['n_simulate_calls']} simulate calls, {S['n_rows']} trajectories "
         f"(status {S['status_rows']}); summed task wall time {S['wall_s_total']:.0f} s, CPU {S['cpu_s_total']:.0f} s "
         f"(longest task {S['wall_s_max_task']:.0f} s); max peak RSS {S['peak_rss_mb_max']:.0f} MB per task.", ""]
    # sanity
    L += ["## Sanity check: gamma = 10 reference runs vs the published R rows", "",
          f"The gamma = 10 runs of every task (linear ZNN, IEZNN, implicit SBP; {len(S['sanity'])} task x entry sets, "
          f"at least {S['sanity_n_pairs_min']} paired trajectories each) reproduce the R_off rows (and, for SBP with E "
          f"in mm, the R_units rows): **max relative difference {S['sanity_max_rel_diff']:.1e}** over "
          f"{', '.join(CHECK_COLS)}.  Wrapper checks: EffortStats peak vs JointStats qd_peak "
          f"{S['check_qd_peak_chk']:.1e}; qd_peak vs simulate's peak_rate {S['check_qd_peak_vs_peak_rate']:.1e}; linear "
          f"ZNN / IEZNN reference rows of the m and mm tasks {S['ref_m_vs_mm_task_max_rel_diff']:.1e}.", ""]
    # search quality
    L += ["## Search quality", "",
          "| search | measure | cells | status | max abs(effort/target - 1) | max bracket width (rel.) | all stage-1 grids monotone |",
          "|---|---|---|---|---|---|---|"]
    for k, q in S["search_quality"].items():
        role, m = k.split("_qd_")
        L.append(f"| {role} | qd_{m} | {q['n']} | {q['status']} | {q['max_abs_mismatch']:.1e} | "
                 f"{q['max_bracket_rel_width']:.1e} | {q['all_grids_monotone']} |")
    drops = {role: max(c["max_rel_drop"][role] for c in S["cells"]) for role in ("A_lin", "A_ie", "B_sbp")}
    L += ["", "The median peak grows monotonically with the gain for linear ZNN and IEZNN (largest relative drop "
          f"between consecutive evaluated gains {drops['A_lin']:.1e} / {drops['A_ie']:.1e}).  For SBP it has wiggles of "
          f"at most {drops['B_sbp']:.1e} (relative) where the peak sits on the path's feed-forward floor; the "
          "crossing rule (first crossing from gamma = 10 downwards) is unaffected.  Linear ZNN and IEZNN get the same "
          "peak-matched gain because their first command (the peak) is identical (IEZNN's integral starts at 0).", ""]
    # transient
    L += ["## Where the effort is spent: the peak is the start transient", "",
          "Medians over the 12 trajectories at gamma = 10.  'first' = speed of the first command (k = 0); "
          "'at k=0' = fraction of trajectories whose peak is the first command; 'late' = peak over t >= 2.5 s; "
          "'excess' = 1 - late/peak = share of the peak above the late-window level.  'on-path' = R_on (start on "
          "the path, E(0) = 0, one trajectory, error in m).", "",
          "| robot | path | rate | linear: peak / late / at k=0 | SBP (m): peak / first / late / at k=0 / excess | "
          "SBP (mm): peak / late / at k=0 / excess | on-path peak: linear / SBP (m) |",
          "|---|---|---|---|---|---|---|"]
    for r in ROBOTS:
        for p in PATHS:
            for rt in RATES:
                cm = _cells(S["cells"], robot=r, path=p, rate=rt, unit="m")
                cmm = _cells(S["cells"], robot=r, path=p, rate=rt, unit="mm")
                if not cm or not cmm:
                    continue
                lin, sm, smm = cm[0]["ref"]["znn_lin"], cm[0]["ref"]["znn_sbp_ie"], cmm[0]["ref"]["znn_sbp_ie"]
                op = [d for d in S["on_path_peak_g10"] if d["robot"] == r and d["path"] == p and d["rate"] == rt]
                opt = f"{g3(op[0]['znn_lin'])} / {g3(op[0]['znn_sbp_ie'])}" if op else "--"
                L.append(f"| {RNAME[r]} | {PNAME[p]} | {rt} | {g3(lin['qd_peak'])} / {g3(lin['qd_peak_late'])} / "
                         f"{lin['frac_peak_first_step']:.2f} | {g3(sm['qd_peak'])} / {g3(sm['qd_first'])} / "
                         f"{g3(sm['qd_peak_late'])} / {sm['frac_peak_first_step']:.2f} / "
                         f"{pct(1 - sm['qd_peak_late'] / sm['qd_peak'], 0)} | {g3(smm['qd_peak'])} / "
                         f"{g3(smm['qd_peak_late'])} / {smm['frac_peak_first_step']:.2f} / "
                         f"{pct(1 - smm['qd_peak_late'] / smm['qd_peak'], 0)} | {opt} |")
    L.append("")
    # direction A, peak
    for m in MC.MEASURES:
        L += [f"## Direction A (main{'' if m == 'qd_peak' else ', secondary measure'}): linear ZNN and IEZNN gain raised "
              f"to the {MNAME[m]} of implicit SBP at gamma = 10", "",
              f"target = median {m} of SBP at gamma = 10 (rad/s); gamma* = matched gain (h* = gamma* tau); 'match' = "
              "effort/target - 1; MSSRE in mm; ratio = MSSRE(SBP, gamma = 10) / MSSRE(comparator, gamma*) [paired "
              "95 % interval]; 'nominal' = the same ratio with the comparator at gamma = 10; 'over limit' = fraction "
              "of the 12 trajectories whose peak exceeds a joint-speed limit (SBP at 10 / comparator at gamma*).", "",
              "| robot | path | rate | unit | target | lin gamma* (h*) | match | MSSRE SBP | MSSRE lin(gamma*) | "
              "ratio SBP/lin | nominal | over limit | IEZNN gamma* | MSSRE IEZNN(gamma*) | ratio SBP/IEZNN | nominal |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for c in cells:
            a, i = c["search"]["A_lin"][m], c["search"]["A_ie"][m]
            if "cmp" not in a or "cmp" not in i:
                L.append(f"| {RNAME[c['robot']]} | {PNAME[c['path']]} | {c['rate']} | {c['unit']} | "
                         f"{g3(c['targets']['A'][m])} | {a['status']} | | | | | | | {i['status']} | | | |")
                continue
            L.append(f"| {RNAME[c['robot']]} | {PNAME[c['path']]} | {c['rate']} | {c['unit']} | {g3(c['targets']['A'][m])} | "
                     f"{g3(a['gamma_star'])} ({g3(a['h_star'], 2)}) | {a['mismatch']:+.1e} | {g3(a['cmp']['mssre_a_mm'])} | "
                     f"{g3(a['cmp']['mssre_b_mm'])} | {ci_txt(a['cmp'])} | {g3(c['nominal']['lin']['ratio'])} | "
                     f"{c['ref']['znn_sbp_ie']['frac_over_limit']:.2f} / {a['effort']['frac_over_limit']:.2f} | "
                     f"{g3(i['gamma_star'])} | {g3(i['cmp']['mssre_b_mm'])} | {ci_txt(i['cmp'])} | "
                     f"{g3(c['nominal']['ie']['ratio'])} |")
        L.append("")
        L += [f"## Direction B ({'check' if m == 'qd_peak' else 'check, secondary measure'}): implicit SBP gain lowered "
              f"to the {MNAME[m]} of linear ZNN at gamma = 10", "",
              f"target = median {m} of linear ZNN at gamma = 10; ratio = MSSRE(SBP, gamma*) / MSSRE(linear ZNN, "
              "gamma = 10) [paired 95 % interval]; 'SBP at 10' = MSSRE of SBP at gamma = 10 (for the cost of lowering "
              "its gain); over limit: SBP at gamma*.", "",
              "| robot | path | rate | unit | target | SBP gamma* | match | MSSRE SBP(gamma*) | SBP at 10 | "
              "MSSRE lin(10) | ratio SBP/lin | factor | over limit |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for c in cells:
            b = c["search"]["B_sbp"][m]
            if "cmp" not in b:
                L.append(f"| {RNAME[c['robot']]} | {PNAME[c['path']]} | {c['rate']} | {c['unit']} | "
                         f"{g3(c['targets']['B'][m])} | {b['status']} | | | | | | | |")
                continue
            L.append(f"| {RNAME[c['robot']]} | {PNAME[c['path']]} | {c['rate']} | {c['unit']} | {g3(c['targets']['B'][m])} | "
                     f"{g3(b['gamma_star'])} | {b['mismatch']:+.1e} | {g3(b['cmp']['mssre_a_mm'])} | "
                     f"{g3(c['ref']['znn_sbp_ie']['mssre_mm'])} | {g3(b['cmp']['mssre_b_mm'])} | {ci_txt(b['cmp'])} | "
                     f"{g3(b['cmp']['factor'])} | {b['effort']['frac_over_limit']:.2f} |")
        L.append("")
    # compact ranges
    L += ["## Ranges over the four robot x path cells (accuracy factor = MSSRE comparator / MSSRE SBP; > 1 favours SBP)",
          "", "| unit | rate | nominal lin | A lin (peak) | A IEZNN (peak) | B (peak) | A lin (RMS) | A IEZNN (RMS) | "
          "B (RMS) | lin gamma* A (peak) | SBP gamma* B (peak) | SBP better (A lin / A IE / B, peak) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for u in UNITS:
        for rt in RATES:
            d = S["ranges"].get(f"{u}_{rt}")
            if not d:
                continue
            L.append(f"| {u} | {rt} | {rng_txt(d['nominal_factor_lin'])} | {rng_txt(d['A_lin_qd_peak_factor'])} | "
                     f"{rng_txt(d['A_ie_qd_peak_factor'])} | {rng_txt(d['B_qd_peak_factor'])} | "
                     f"{rng_txt(d['A_lin_qd_rms_factor'])} | {rng_txt(d['A_ie_qd_rms_factor'])} | "
                     f"{rng_txt(d['B_qd_rms_factor'])} | {rng_txt(d['A_lin_qd_peak_gamma'], 3)} | "
                     f"{rng_txt(d['B_qd_peak_gamma'], 3)} | {d['A_lin_qd_peak_sbp_better']}/{d['A_lin_qd_peak_n_matched']}, "
                     f"{d['A_ie_qd_peak_sbp_better']}/{d['A_ie_qd_peak_n_matched']}, "
                     f"{d['B_qd_peak_sbp_better']}/{d['B_qd_peak_n_matched']} |")
    L += ["", "Mechanism: empirical gain exponent p of MSSRE ~ gamma^p at fixed tau (two points, gamma = 10 and the "
          "peak-matched gamma*), the effort ratio SBP / linear ZNN at gamma = 10, and the margin of direction B's "
          "target (linear ZNN's median peak at gamma = 10) above its late-window level (the path's feed-forward floor).",
          "", "| unit | rate | p linear ZNN | p IEZNN | p SBP | peak SBP(10) / linear(10) | B target above floor |",
          "|---|---|---|---|---|---|---|"]
    for u in UNITS:
        for rt in RATES:
            d = S["ranges"].get(f"{u}_{rt}")
            if not d:
                continue
            bt = d["B_target_over_floor"]
            L.append(f"| {u} | {rt} | {rng_txt(d['gain_exponent_A_lin'])} | {rng_txt(d['gain_exponent_A_ie'])} | "
                     f"{rng_txt(d['gain_exponent_B_sbp'])} | {rng_txt(d['peak_ratio_sbp_over_lin_10'])} | "
                     f"{pct_rng(bt, 1)} |")
    L.append("")
    # steady-state measure
    sp = [c["late_peak_spread"] for c in S["cells"]]
    lr = [c["late_peak_lin_range"][1] / c["late_peak_lin_range"][0] - 1 for c in S["cells"] if c["late_peak_lin_range"]]
    L += ["## Steady-state effort: peak joint speed over the last 75 % of the run (t >= 2.5 s)", "",
          f"Across the run sets of a cell (linear ZNN, IEZNN and SBP at gamma = 10; linear ZNN and IEZNN at the "
          f"peak-matched gamma* of direction A; SBP at gamma* of direction B) the median late-window peak differs by "
          f"at most {pct(max(sp), 2)} (max over the {len(sp)} cells; median spread {pct(float(np.median(sp)), 2)}).  For "
          f"linear ZNN it changes by at most {pct(max(lr), 2)} over the whole searched gain range (gamma = 10 to "
          f"{g3(max(c['late_peak_lin_gamma_range'][1] for c in S['cells'] if c['late_peak_lin_gamma_range']))}).  "
          "The steady-state effort is the feed-forward J+ r_d_dot of the path, the same for every controller and gain, "
          "so under this measure the nominal gamma = 10 comparison is already matched (and so is every other gain): "
          "it cannot select a gain.  The extra effort of implicit SBP is spent only in the start transient.", "",
          "| robot | path | rate | unit | late peak lin(10) | SBP(10) | lin(gamma*_A) | SBP(gamma*_B) | spread |",
          "|---|---|---|---|---|---|---|---|---|"]
    for c in cells:
        lp = c["late_peak"]
        L.append(f"| {RNAME[c['robot']]} | {PNAME[c['path']]} | {c['rate']} | {c['unit']} | {g3(lp['lin_10'], 4)} | "
                 f"{g3(lp['sbp_10'], 4)} | {g3(lp.get('lin_A'), 4)} | {g3(lp.get('sbp_B'), 4)} | "
                 f"{pct(c['late_peak_spread'], 2)} |")
    L.append("")
    # limits
    lim = S["limit"]
    L += ["## Joint-speed limits (Panda 2.175 rad/s J1-4, 2.61 rad/s J5-7; PUMA 560 assumed 2 rad/s)", "",
          "Fraction of the 12 trajectories whose commanded peak exceeds a limit, and the largest peak/limit ratio "
          "(qd_util), for SBP at gamma = 10 and the peak-matched comparators (direction A) and SBP (direction B).", "",
          "| robot | path | rate | unit | SBP(10) over / max util | lin(gamma*) over / max util | "
          "IEZNN(gamma*) over / max util | SBP(gamma*_B) over / max util |", "|---|---|---|---|---|---|---|---|"]
    for d in sorted(lim, key=cell_key):
        L.append(f"| {RNAME[d['robot']]} | {PNAME[d['path']]} | {d['rate']} | {d['unit']} | {d['sbp_10']:.2f} / "
                 f"{g3(d['sbp_10_util_max'])} | {d.get('A_lin_qd_peak', np.nan):.2f} / {g3(d.get('A_lin_qd_peak_util_max'))} | "
                 f"{d.get('A_ie_qd_peak', np.nan):.2f} / {g3(d.get('A_ie_qd_peak_util_max'))} | "
                 f"{d.get('B_sbp_qd_peak', np.nan):.2f} / {g3(d.get('B_sbp_qd_peak_util_max'))} |")
    L.append("")
    L += ["## Interpretation", ""] + interpret(S) + [""]
    L += ["## Files and commands", "",
          "* config / task list: `experiments/configs/R_matched.py`; task: `experiments/tasks/R_matched_search.py`; "
          "shared code (EffortStats wrapper, search): `experiments/tasks/R_matched_common.py`; analysis: "
          "`experiments/analyze_R_matched.py`.",
          "* outputs: `results/R_matched/results/*.json|*.parquet` (search logs, per-trajectory rows), `summary.json`, "
          "`SUMMARY.md`, `R_matched_table.tex`; figure `figures/fig_R_matched.pdf/png`.", "",
          "```bash", "# from znn-bench/",
          "python experiments/configs/R_matched.py --out results/R_matched",
          "python tools/memguard.py pool --tasks results/R_matched/tasks.jsonl --workers 8 --max-gb 60",
          "python experiments/analyze_R_matched.py --results results/R_matched --rbase results/R --figdir figures",
          "```", ""]
    (Path(results) / "SUMMARY.md").write_text("\n".join(L))


def _all(cells, fn):
    v = [fn(c) for c in cells]
    v = [x for x in v if x is not None and np.isfinite(x)]
    return [float(min(v)), float(max(v))] if v else [np.nan, np.nan]


def interpret(S):
    """Interpretation bullets; every number is computed here from S (no hand-typed numbers)."""
    R, C = S["ranges"], S["cells"]
    out = []

    def f(key, u, rt):
        return rng_txt(R[f"{u}_{rt}"][key])

    def n(key, u, rt, k="n_matched"):
        d = R[f"{u}_{rt}"]
        return f"{d[key.replace('n_matched', 'sbp_better')]}/{d[key]}"

    cm, cmm = [c for c in C if c["unit"] == "m"], [c for c in C if c["unit"] == "mm"]
    sb = lambda c: c["ref"]["znn_sbp_ie"]  # noqa: E731
    onp = S.get("on_path_peak_g10", [])
    on_diff = _all(onp, lambda d: abs(d["znn_sbp_ie"] / d["znn_lin"] - 1.0)) if onp else [np.nan, np.nan]
    out.append(
        "- **The extra effort of implicit SBP is the start transient.**  At gamma = 10 its median peak is "
        f"{rng_txt(_all(cm, lambda c: sb(c)['qd_peak'] / c['ref']['znn_lin']['qd_peak']))} times that of linear "
        f"ZNN with E in m ({rng_txt(_all(cmm, lambda c: sb(c)['qd_peak'] / c['ref']['znn_lin']['qd_peak']))} in mm); "
        f"the peak is the very first command in {pct_rng(_all(C, lambda c: sb(c)['frac_peak_first_step']))} of "
        f"the trajectories of a cell, and {pct_rng(_all(cm, lambda c: 1 - sb(c)['qd_peak_late'] / sb(c)['qd_peak']))} "
        f"(m) and {pct_rng(_all(cmm, lambda c: 1 - sb(c)['qd_peak_late'] / sb(c)['qd_peak']))} (mm) of it lies above "
        "the late-window level.  From on-path starts (R_on, E(0) = 0) the peaks of SBP and linear ZNN differ by at most "
        f"{pct(on_diff[1], 2)}.  After the transient (t >= 2.5 s) every controller at every gain commands the same "
        f"joint speeds (late-window peaks within {pct(max(c['late_peak_spread'] for c in C), 2)}): the steady-state "
        "effort is the path's feed-forward and cannot be traded for accuracy.")
    for u in UNITS:
        if f"{u}_100" not in R or f"{u}_1000" not in R:
            continue
        out.append(
            f"- **Error in {u}, direction A** (linear ZNN's gain raised to SBP's peak, gamma* = "
            f"{f('A_lin_qd_peak_gamma', u, 100)} at 100 Hz to {f('A_lin_qd_peak_gamma', u, 1000)} at 1 kHz): implicit "
            f"SBP is {f('A_lin_qd_peak_factor', u, 100)} times (100 Hz), {f('A_lin_qd_peak_factor', u, 333)} times "
            f"(333 Hz) and {f('A_lin_qd_peak_factor', u, 1000)} times (1 kHz) more accurate than linear ZNN (SBP better "
            f"in {n('A_lin_qd_peak_n_matched', u, 100)}, {n('A_lin_qd_peak_n_matched', u, 333)}, "
            f"{n('A_lin_qd_peak_n_matched', u, 1000)} cells), against {f('nominal_factor_lin', u, 100)}, "
            f"{f('nominal_factor_lin', u, 333)} and {f('nominal_factor_lin', u, 1000)} times at the nominal gain.  "
            f"IEZNN at the same effort (same gamma*, its first command is identical): factor "
            f"{f('A_ie_qd_peak_factor', u, 100)} (100 Hz), {f('A_ie_qd_peak_factor', u, 333)} (333 Hz), "
            f"{f('A_ie_qd_peak_factor', u, 1000)} (1 kHz); SBP better in {n('A_ie_qd_peak_n_matched', u, 100)}, "
            f"{n('A_ie_qd_peak_n_matched', u, 333)}, {n('A_ie_qd_peak_n_matched', u, 1000)} cells.")
        out.append(
            f"- **Error in {u}, direction B** (SBP's gain lowered to linear ZNN's peak at gamma = 10, gamma* = "
            f"{f('B_qd_peak_gamma', u, 100)} at 100 Hz to {f('B_qd_peak_gamma', u, 1000)} at 1 kHz): factor "
            f"{f('B_qd_peak_factor', u, 100)} (100 Hz), {f('B_qd_peak_factor', u, 333)} (333 Hz), "
            f"{f('B_qd_peak_factor', u, 1000)} (1 kHz); SBP better in {n('B_qd_peak_n_matched', u, 100)}, "
            f"{n('B_qd_peak_n_matched', u, 333)}, {n('B_qd_peak_n_matched', u, 1000)} cells.")
    ge = {role: _all(C if role != "B_sbp" else cm, lambda c, r=role: c["gain_exponent"].get(r)) for role in
          ("A_lin", "A_ie", "B_sbp")}
    ge_mm = _all(cmm, lambda c: c["gain_exponent"].get("B_sbp"))
    out.append(
        "- **Mechanism (MSSRE ~ gamma^p at fixed tau, from gamma = 10 and gamma*):** linear ZNN p = "
        f"{rng_txt(ge['A_lin'])}, IEZNN p = {rng_txt(ge['A_ie'])} (its floor falls as 1/gamma^2), implicit SBP "
        f"p = {rng_txt(ge['B_sbp'])} with E in m (its floor hardly depends on the gain: lowering it to gamma* costs "
        "almost no accuracy) but p = " + rng_txt(ge_mm) + " with E in mm.  Direction B's target lies "
        f"{pct_rng(_all(C, lambda c: c['B_target_over_floor']), 1)} above the path floor, so B asks for a start transient no faster than the path motion itself.")
    lim_m = _all(cm, lambda c: max(sb(c)["qd_util_max"], c["search"]["A_lin"]["qd_peak"]["effort"]["qd_util_max"],
                                   c["search"]["A_ie"]["qd_peak"]["effort"]["qd_util_max"]))
    hi_mm = [c for c in cmm if c["rate"] in (333, 1000)]
    out.append(
        "- **Joint-speed limits:** with E in m every matched run stays within the limits (largest peak/limit "
        f"{g3(lim_m[1])}).  With E in mm at 333 Hz and 1 kHz SBP at gamma = 10 itself exceeds a limit in "
        f"{pct_rng(_all(hi_mm, lambda c: sb(c)['frac_over_limit']))} of the trajectories of a cell (peak/limit up to "
        f"{g3(_all(hi_mm, lambda c: sb(c)['qd_util_max'])[1])}), and so do the peak-matched comparators "
        f"({pct_rng(_all(hi_mm, lambda c: c['search']['A_lin']['qd_peak']['effort']['frac_over_limit']))}); "
        "the direction-B runs are within the limits everywhere (largest peak/limit "
        f"{g3(_all(C, lambda c: c['search']['B_sbp']['qd_peak']['effort']['qd_util_max'])[1])}).")
    out.append(
        "- **RMS effort (secondary):** matching the RMS joint speed instead of the peak gives the same picture: "
        f"direction A factors {f('A_lin_qd_rms_factor', 'm', 100)} (m, 100 Hz), {f('A_lin_qd_rms_factor', 'm', 1000)} "
        f"(m, 1 kHz), {f('A_lin_qd_rms_factor', 'mm', 100)} (mm, 100 Hz), {f('A_lin_qd_rms_factor', 'mm', 1000)} "
        f"(mm, 1 kHz); direction B {f('B_qd_rms_factor', 'm', 100)}, {f('B_qd_rms_factor', 'm', 1000)}, "
        f"{f('B_qd_rms_factor', 'mm', 100)}, {f('B_qd_rms_factor', 'mm', 1000)}.")
    out.append("- Factors are MSSRE(comparator) / MSSRE(SBP), ranges over the four robot x path cells; factors < 1 mean "
               "that the comparator is the more accurate controller at that effort.  Per-cell ratios with paired "
               "intervals are in the tables above.")
    return out


# --------------------------------------------------------------------------
# LaTeX table (Supplementary Section S5)
# --------------------------------------------------------------------------
def write_tex(S, path):
    cells = S["cells"]

    def get(r, p, rt, u):
        c = _cells(cells, robot=r, path=p, rate=rt, unit=u)
        return c[0] if c else None

    def fac(x):
        if x is None or not np.isfinite(x):
            return "--"
        return g3(x, 3) if x >= 10 else g3(x, 2)

    hw = S["ci_max_rel_halfwidth_peak"]

    lines = [r"\begin{table}[tbp]", r"\centering", r"\renewcommand{\baselinestretch}{1}\scriptsize",
             r"\setlength{\tabcolsep}{3pt}",
             r"\caption{Robot study at matched control effort (noise-free, off-path starts, median over twelve "
             r"trajectories). Accuracy factor = MSSRE of the comparator divided by MSSRE of implicit sign-bi-power "
             r"ZNN (SBP); values above 1 favor SBP. Nominal: both at $\gamma=10$. A: the comparator's gain is raised "
             r"until its median peak commanded joint speed equals that of SBP at $\gamma=10$ (linear ZNN, IEZNN). "
             r"B: the gain of SBP is lowered until its median peak equals that of linear ZNN at $\gamma=10$. "
             rf"Paired bootstrap 95\,\% intervals (descriptive) lie within {100 * hw:.1f}\,\% of every value.}}",
             r"\label{tab:S5-Rmatched}",
             r"\begin{tabular}{@{}lllrrrrrrrr@{}}", r"\toprule",
             r"& & & \multicolumn{4}{c}{100\,Hz} & \multicolumn{4}{c}{1\,kHz} \\",
             r"\cmidrule(lr){4-7}\cmidrule(l){8-11}",
             r"Robot & Path & Error & Nominal & A: linear & A: IEZNN & B & Nominal & A: linear & A: IEZNN & B \\",
             r"\midrule"]
    for r in ROBOTS:
        for p in PATHS:
            for u in UNITS:
                row = [RNAME[r] if u == "m" else "", PNAME[p] if u == "m" else "", u]
                for rt in (100, 1000):
                    c = get(r, p, rt, u)
                    if c is None:
                        row += ["--"] * 4
                        continue
                    a = c["search"]["A_lin"]["qd_peak"].get("cmp", {}).get("factor", np.nan)
                    i = c["search"]["A_ie"]["qd_peak"].get("cmp", {}).get("factor", np.nan)
                    b = c["search"]["B_sbp"]["qd_peak"].get("cmp", {}).get("factor", np.nan)
                    row += [fac(c["nominal"]["lin"]["factor"]), fac(a), fac(i), fac(b)]
                lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    Path(path).write_text("\n".join(lines) + "\n")
    return path


# --------------------------------------------------------------------------
# figure: accuracy-effort curves
# --------------------------------------------------------------------------
def _curve(rows, **k):
    """(gamma, median peak, median MSSRE, is-stage-1-grid) over every evaluated gain of the selection."""
    c = sel(rows, **k)
    if c.empty:
        return np.array([]), np.array([]), np.array([]), np.array([], bool)
    g = c.groupby("gamma")
    x = g.qd_peak.median()
    y = g.mssre_mm.median()
    s1 = g.stage.agg(lambda s: bool((s == "s1").any()))
    return x.index.to_numpy(float), x.to_numpy(float), y.to_numpy(float), s1.to_numpy(bool)


def fig_matched(S, rows, figdir, rates=(100, 1000)):
    style.FIG_DIR = Path(figdir)
    width = getattr(style, "TEXTWIDTH", 390.0 / 72.27)
    rc = style.print_rc() if hasattr(style, "print_rc") else matplotlib.rc_context({})
    combos = [(r, p) for r in ROBOTS for p in PATHS]
    curves = [("znn_lin", "A_lin", None, "linear ZNN ($\\gamma\\geq10$)", {}),
              ("ieznn", "A_ie", None, "IEZNN ($\\gamma\\geq10$)", {}),
              ("znn_sbp_ie", "B_sbp", "m", "implicit SBP, $E$ in m ($\\gamma\\leq10$)", {}),
              ("znn_sbp_ie", "B_sbp", "mm", "implicit SBP, $E$ in mm ($\\gamma\\leq10$)",
               {"linestyle": "--", "mfc": "white"})]
    with rc:
        fig, axes = plt.subplots(len(rates), len(combos), figsize=(width, 1.45 * len(rates) + 0.75),
                                 sharex=True, sharey="row", squeeze=False)
        for i, rt in enumerate(rates):
            for j, (r, p) in enumerate(combos):
                ax = axes[i, j]
                for e, role, u, lab, extra in curves:
                    if u is None:          # unit-invariant comparator: searches of both unit tasks
                        g, x, y, s1 = _curve(rows, robot=r, path=p, rate=rt, role=role)
                    else:
                        g, x, y, s1 = _curve(rows, robot=r, path=p, rate=rt, role=role, unit=u)
                    if not len(g):
                        continue
                    s = AR.sty(e)
                    ls = extra.get("linestyle", s["linestyle"])
                    mfc = extra.get("mfc", s["color"])
                    o = np.argsort(g)
                    ax.plot(x[o], y[o], color=s["color"], ls=ls, lw=1.0, zorder=3)
                    ax.plot(x[o][s1[o]], y[o][s1[o]], ls="none", marker=s["marker"], ms=2.4, mfc=mfc,
                            mec=s["color"], mew=0.6, zorder=4)
                    ax.plot([], [], color=s["color"], ls=ls, lw=1.0, marker=s["marker"], ms=3.0, mfc=mfc,
                            mec=s["color"], mew=0.6, label=lab)
                    k10 = np.argmin(np.abs(g - MC.GAMMA_REF))
                    ax.plot([x[k10]], [y[k10]], ls="none", marker=s["marker"], ms=5.0,
                            mfc=mfc, mec=style.INK, mew=0.9, zorder=5)
                qd_max, _ = RC.joint_limits(r)
                ax.axvline(float(np.min(qd_max)), color=style.INK2, ls=":", lw=0.8, zorder=1)
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xlim(0.25, 6.0)
                ax.set_xticks([0.3, 1, 3])
                ax.set_xticklabels(["0.3", "1", "3"])
                ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
                ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=100.0))
                ax.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(base=10.0, subs=(1.0,)))
                ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
                if i == 0:
                    ax.set_title(f"{RNAME[r]}, {PNAME[p]}")
                if j == 0:
                    ax.set_ylabel(f"{rt:g} Hz\nMSSRE (mm)")
        top = 0.855 if len(rates) == 2 else 0.88
        fig.tight_layout(rect=(0, 0.035, 1, top), h_pad=0.5, w_pad=0.3)
        fig.supxlabel("median peak commanded joint speed (rad/s)", y=0.0, va="bottom",
                      fontsize=plt.rcParams["axes.labelsize"])
        h, lab = axes[0, 0].get_legend_handles_labels()
        h += [Line2D([], [], ls="none", marker="o", ms=5.0, mfc="none", mec=style.INK, mew=0.9),
              Line2D([], [], color=style.INK2, ls=":", lw=0.8)]
        lab += ["$\\gamma=10$ (nominal design)", "joint-speed limit"]
        fig.legend(h, lab, loc="lower center", ncol=3, bbox_to_anchor=(0.5, top), columnspacing=1.0,
                   handletextpad=0.4, frameon=False)
        return style.savefig(fig, "fig_R_matched")


# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results/R_matched")
    ap.add_argument("--rbase", default="results/R")
    ap.add_argument("--figdir", default="figures")
    args = ap.parse_args(argv)
    S, rows = analyse(args.results, args.rbase)
    S["table_tex"] = str(write_tex(S, Path(args.results) / "R_matched_table.tex"))
    S["figure"] = str(fig_matched(S, rows, args.figdir))
    write_summary(S, args.results)

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        return o
    (Path(args.results) / "summary.json").write_text(json.dumps(clean(S), indent=1))
    print(f"wrote {args.results}/SUMMARY.md, summary.json, R_matched_table.tex and {S['figure']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
