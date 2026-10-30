#!/usr/bin/env python
"""E1 -- order verification: which residual order does each discretization deliver?

Protocol (experiments/PLAN.md, E1; full rationale and deviations in
experiments/E1_NOTES.md):

* model: linear ZNN (``znn``, activation ``linear``);
* formulas: every registered causal look-ahead discretization (Euler,
  Taylor-Zhang, ZeaD, Uhlig, AB2-AB4), RK4 (non-causal reference) and the
  implicit / consistent variants; keys added to the zoo later (e.g.
  ``zead_7i_cang2024``, ``uhlig_5_*``) are picked up at build time if they are
  in ``znnbench.available()`` (re-run the build to include them);
* problems: linear_system (n=4), matrix_inversion (n=4, n=16), sylvester (n=4),
  tvqp_eq, nonlinear_equations, plus two linear_system controls: low frequency
  (freq=0.2) and constant matrix A (b(t) still time-varying, wrapper
  ``experiments.tasks.E1_problems:ConstantMatrix``);
* step-size modes: fixed-h with h in {h_rec (the formula's recommended /
  optimal h), 0.1 if 0.1 lies in the formula's stable range}, batched in one
  simulate call; fixed-gamma with gamma = 10;
* tau grid 1e-1 .. 1e-4 (13 points, 1-2-3-5 spacing; mini: 1e-1 .. 1e-3);
* 3 problem seeds x B = 4 initial states (mini: 2 x 4);
* horizon (one per formula, tau AND step-size mode): T = max(6 s, 1.5 s +
  37 tau / r) where r = -ln rho is the per-step decay rate of the formula's
  frozen-coefficient linear-ZNN error recursion at the run's h; in fixed-h
  mode r is reduced to max(r - tau*L_REF, r/4) (L_REF = 3 ~ a typical
  ||A^{-1} A_dot|| at 1 rad/s), because the tau*||A^{-1} A_dot|| coupling slows
  the decay of high-order formulas at small h (review finding: with the
  frozen rate alone, uhlig_4_6 at tau = 1e-2 and zead_5i / zead_10i / zead_11i
  at tau = 3e-2 kept up to O(1) x0-dependent transients in the window);
  rounded up to a ladder and capped at 200 s.  Every run also records the
  diagnostic ``x0_spread_w`` (E1_problems.window_metrics) so that a point whose
  transient has not decayed is detected and excluded by the analysis;
* steady window: every run is time-shifted so that it ENDS at the same absolute
  problem time t_end (6 s; 200 s for nonlinear_equations, whose data are
  undefined for t <= -1), and the steady-state residual is the max over the
  absolute window [t_end - 1.5 s, t_end] (extra metric ``mssre_w`` of
  ``experiments.tasks.E1_problems:window_metrics``), i.e. the PLAN's window
  (last 25 % of 6 s) on the same data segment for every tau.

Scale: the environment variable ``E1_SCALE`` (or ``ZNNBENCH_SCALE``) = ``mini``
or ``full`` (default ``full``) selects the grid when the runner loads this
file; when run as a script use ``--scale``.

Build (one sub-folder and tasks.jsonl per formula + a merged pool file):

    python experiments/configs/E1.py --scale full --out results/E1
    python tools/memguard.py pool --tasks results/E1/pool_tasks.jsonl --workers 24 --max-gb 200
    python experiments/analyze_E1.py --results results/E1 --figdir figures --tabdir results/tables

Cost estimate (probes wall/step of every formula x instance, ~1 min):
``OMP_NUM_THREADS=1 python experiments/configs/E1.py --scale full --estimate-cost results/E1/full_cost_estimate.json``.

Equivalent runner call for the specs only (no merged pool file):
``E1_SCALE=full python -m znnbench.runner build experiments/configs/E1.py --out results/E1``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

# ---------------------------------------------------------------------------
# protocol constants
# ---------------------------------------------------------------------------
EXPERIMENT = "E1"
T0 = 6.0                        # PLAN default horizon (s)
FRAC = 0.25                     # runner's own steady window (last 25 % of T), kept as `mssre`
W_ABS = 1.5                     # absolute steady window (s) = 25 % of T0 -> `mssre_w`
K_DECAY = 37.0                  # e^-37 ~ 1e-16: transient decay required before the window
L_REF = 3.0                     # coupling allowance tau*L_REF in the fixed-h decay rate (1/s)
T_CAP = 200.0                   # cap of the adaptive horizon (s) (= t_end of TimeShiftLate)
T_LADDER = [6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 100, 120, 150, 200]
T_END = 6.0                     # absolute end time of every run (TimeShift)
T_END_LATE = 200.0              # ... for problems undefined at negative times (TimeShiftLate)
H_COMMON = 0.1                  # common fixed-h value (if inside the stable range)
GAMMA_FIXED = 10.0              # fixed-gamma value
CONSTA_KEY = "experiments.tasks.E1_problems:ConstantMatrix"
TSHIFT_KEY = "experiments.tasks.E1_problems:TimeShift"
TSHIFT_LATE_KEY = "experiments.tasks.E1_problems:TimeShiftLate"
WINDOW_METRIC = "experiments.tasks.E1_problems:window_metrics"
LATE_INSTANCES = ["tvne_guo2018"]     # data undefined for t <= -1 -> end at T_END_LATE

TAUS = {
    "full": [1e-1, 5e-2, 3e-2, 2e-2, 1e-2, 5e-3, 3e-3, 2e-3, 1e-3, 5e-4, 3e-4, 2e-4, 1e-4],
    "mini": [1e-1, 5e-2, 3e-2, 2e-2, 1e-2, 5e-3, 3e-3, 2e-3, 1e-3],
}
SEEDS = {"full": [0, 1, 2], "mini": [0, 1]}
B = 4

# formulas -------------------------------------------------------------------
# (key, kwargs) -- registered causal look-ahead / multistep formulas
LOOKAHEAD = [
    ("euler", {}), ("taylor_zhang", {}),
    ("zead_4i_general", {}), ("zead_4i_6321", {}), ("zead_4i_fifd", {}),
    ("zead_5i", {}), ("zead_5i_general", {"a1": 0.1}),   # default a1=1/9 duplicates zead_5i
    ("zead_6i_guo2017", {}), ("zead_6i_qiu2019", {}),
    ("zead_7i_general", {}), ("zead_8i_wu2020", {}), ("zead_8i_chen2019", {}),
    ("zead_10i_wu2020", {}), ("zead_11i_chen2023", {}),
    ("uhlig_2_3", {}), ("uhlig_3_4", {}), ("uhlig_4_6", {}),
    ("ab2", {}), ("ab3", {}), ("ab4", {}),
]
REFERENCE = [("rk4", {})]
IMPLICIT = [
    ("implicit_euler_err", {}), ("peb_euler", {}), ("exact_flow_euler", {}),
    ("semi_implicit_euler", {}),
    ("implicit_taylor_zhang", {}), ("implicit_taylor_zhang_naive", {}),
    ("peb_taylor_zhang", {}), ("exact_flow_taylor_zhang", {}), ("semi_implicit_taylor_zhang", {}),
]
# keys another workflow is adding; included automatically once registered
OPTIONAL = ["zead_7i_cang2024", "uhlig_5_5", "uhlig_5_6", "uhlig_5_6b", "uhlig_5_7", "uhlig_6_8"]
# registered but not E1 formulas: the generic factory (its registered
# combinations are listed above) and the prediction-correction / tracking
# baselines (not look-ahead ZNN discretizations; they are E9's subject)
NOT_E1 = {"consistent", "pc_newton", "pc_gradient", "newton_tracking", "running_gradient",
          "pc_newton_causal", "pc_gradient_causal", "newton_tracking_causal"}
MINI_FORMULAS = ["euler", "taylor_zhang", "zead_5i", "zead_6i_guo2017", "uhlig_3_4", "uhlig_4_6",
                 "ab4", "rk4", "implicit_taylor_zhang", "zead_7i_cang2024",
                 # review 2026-10-18: the inverse-free DTZNNs of Ma & Huang 2025 (c0-normalised h in
                 # the source) are validated end to end in the mini as well
                 "zead_5i_mahuang2025", "cdtznn_mahuang2025", "broyden_dtznn_zhang2012"]

# problem instances ------------------------------------------------------------
# label -> (registry key, kwargs, role); "+constA" instances are linsys_n4 + wrapper
INSTANCES = {
    "linsys_n4": ("linear_system", {"n": 4, "kappa": 10, "freq": 1.0, "seed": "$seed"}, "main"),
    "matinv_n4": ("matrix_inversion", {"n": 4, "kappa": 10, "freq": 1.0, "seed": "$seed"}, "main"),
    "matinv_n16": ("matrix_inversion", {"n": 16, "kappa": 10, "freq": 1.0, "seed": "$seed"}, "main"),
    "sylv_n4": ("sylvester", {"n": 4, "kappa": 10, "freq": 1.0, "seed": "$seed"}, "main"),
    "tvqp_eq": ("tvqp_eq", {"variant": "zhang2021"}, "main"),
    "tvne_guo2018": ("nonlinear_equations", {"variant": "guo2018"}, "main"),
    "linsys_n4_f0.2": ("linear_system", {"n": 4, "kappa": 10, "freq": 0.2, "seed": "$seed"}, "control"),
}
CONSTA_INSTANCE = "linsys_n4+constA"     # = linsys_n4 with the ConstantMatrix wrapper
# + constA; tvne_guo2018 exercises TimeShiftLate and x0-dependent divergence
MINI_INSTANCES = ["linsys_n4", "linsys_n4_f0.2", "matinv_n4", "tvqp_eq", "tvne_guo2018"]
INSTANCE_ORDER = ["linsys_n4", "matinv_n4", "matinv_n16", "sylv_n4", "tvqp_eq", "tvne_guo2018",
                  "linsys_n4_f0.2", CONSTA_INSTANCE]
# angular frequency scale of each instance (for the omega*tau collapse plot)
INSTANCE_OMEGA = {"linsys_n4": 1.0, "matinv_n4": 1.0, "matinv_n16": 1.0, "sylv_n4": 1.0,
                  "tvqp_eq": 4.0, "tvne_guo2018": 1.0, "linsys_n4_f0.2": 0.2, CONSTA_INSTANCE: 1.0}


def scale_from_env(default="full"):
    s = (os.environ.get("E1_SCALE") or os.environ.get("ZNNBENCH_SCALE") or default).strip().lower()
    if s not in ("mini", "full"):
        raise ValueError(f"E1 scale must be 'mini' or 'full', got {s!r}")
    return s


# ---------------------------------------------------------------------------
# formula catalogue: claimed order, recommended h, stable range, rho(h)
# ---------------------------------------------------------------------------
def _registry():
    from znnbench import registry
    registry._load_all()
    return registry


def available_discretizations():
    import znnbench
    return set(znnbench.available()["discretizations"])


def _rk4_R(h):
    return 1.0 - h + h * h / 2.0 - h ** 3 / 6.0 + h ** 4 / 24.0


@lru_cache(maxsize=None)
def _rho_empirical(key, kw_json, h, n_steps=1500):
    """Contraction factor per step of the linear-ZNN error recursion of an
    arbitrary discretization at step size h, measured on the time-invariant
    instance linear_system(freq=0) (dE/dt = 0, so E_k follows the error
    recursion exactly; tau = 1, gamma = h).  inf = divergent."""
    reg = _registry()
    from znnbench.simulate import simulate
    p = reg.make_problem("linear_system", n=2, kappa=2.0, freq=0.0, seed=0, T=float(n_steps))
    d = reg.make_discretization(key, **json.loads(kw_json))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with np.errstate(all="ignore"):
            sim = simulate(p, reg.make_model("znn"), d, 1.0, T=float(n_steps), B=1, gamma=float(h),
                           seed=0, record_solution_error=False, online_stats=False)
    r = np.asarray(sim["residual"])[:, 0]
    if not np.all(np.isfinite(r)) or r[-1] > 10 * r[0]:
        return float("inf")
    r0 = r[0]
    k1 = min(10, len(r) - 2)
    alive = np.nonzero(r > max(1e-13 * r0, 1e-300))[0]
    k2 = int(alive[-1]) if len(alive) else k1 + 1
    if k2 <= k1 + 5:                       # converged to round-off within a few steps
        return float(max(1e-13, 1e-13 ** (1.0 / max(k2, 1))))
    return float(min(1.0, (r[k2] / r[k1]) ** (1.0 / (k2 - k1))))


@lru_cache(maxsize=None)
def formula_info(key, kw_json="{}"):
    """Catalogue entry (JSON-serialisable) of one formula."""
    reg = _registry()
    from znnbench.discretizations.multistep import LookAhead, RK4
    from znnbench.discretizations.zead import lookahead_stability, lookahead_znn_poly, spectral_radius
    kw = json.loads(kw_json)
    d = reg.make_discretization(key, **kw)
    try:
        desc = d.describe()
    except Exception:  # noqa: BLE001
        desc = {}
    info = {"key": key, "kwargs": kw, "name": getattr(d, "name", key), "class": type(d).__name__,
            "causal": bool(getattr(d, "causal", True)), "implicit": bool(getattr(d, "implicit", False)),
            "claimed_order": getattr(d, "expected_order", None), "instants": desc.get("instants"),
            "h_max": None, "h_rec": None, "h_rec_source": None, "rho_kind": "empirical"}
    coeffs = None
    if hasattr(d, "c_next") and hasattr(d, "c_past"):
        coeffs = [float(d.c_next), *[float(c) for c in d.c_past]]
        info["coeffs"] = coeffs
        info["instants"] = info["instants"] or len(coeffs)
    if isinstance(d, RK4):
        hs = np.linspace(1e-4, 3.0, 30001)
        R = np.abs(_rk4_R(hs))
        info.update(claimed_order=4, claimed_note="classical ODE order 4 (no ZNN residual claim)",
                    h_max=float(hs[np.nonzero(R >= 1)[0][0]]), h_rec=float(hs[np.argmin(R)]),
                    h_rec_source="argmin_h |R(-h)| (fastest error contraction of RK4)",
                    rho_kind="rk4", instants=None)
    elif coeffs is not None and isinstance(d, LookAhead) and not info["implicit"]:
        st = lookahead_stability(coeffs)
        info.update(h_max=float(st["h_max"]), h_rec=float(desc.get("h_opt") or st["h_opt"]),
                    h_rec_source="h_opt = argmin spectral radius of P_h (Zhang et al. TNNLS 2019 criterion)",
                    rho_kind="lookahead")
    elif "h_opt" in desc and desc.get("h_opt") and not info["implicit"]:
        info.update(h_max=float(desc["h_max"]) if desc.get("h_max") else None, h_rec=float(desc["h_opt"]),
                    h_rec_source="h_opt from describe()", rho_kind="describe")
        if hasattr(d, "znn_poly"):
            info["rho_kind"] = "znn_poly"
    elif coeffs is not None:                      # implicit / consistent look-ahead
        st = lookahead_stability(coeffs)
        info.update(h_rec=float(st["h_opt"]),
                    h_rec_source="h_opt of the explicit base formula (no stated optimum for the implicit variant)",
                    rho_kind="empirical")
    elif desc.get("h_opt"):
        info.update(h_rec=float(desc["h_opt"]), h_rec_source="h_opt from describe()")
    elif desc.get("x_recursion_key"):             # e.g. cdtznn_mahuang2025 (inverse-free)
        xk = str(desc["x_recursion_key"])
        xi = formula_info(xk, "{}")
        info.update(h_rec=xi["h_rec"], h_max=xi["h_max"], x_recursion_key=xk,
                    h_rec_source=f"h_opt of its x-recursion formula {xk} (M_k ~ J^-1)",
                    rho_kind="empirical")
    # euler / taylor_zhang literature values (docs/spec/discretizations.md)
    if key == "euler":
        info["h_rec_source"] = "Euler: fastest at h = 1 (Jin & Zhang, IEEE TNNLS 26(7) 2015)"
    if key == "taylor_zhang":
        info["h_rec_source"] = "Taylor-Zhang h_o = 0.24 (Zhang et al., IEEE TNNLS 30(3) 2019, Table I)"
    # step-size convention (review 2026-10-18).  Every h of E1 is the raw product
    # h = gamma*tau of ZNN-Bench (tau-form coefficients).  Some sources normalise
    # by the leading coefficient, h_paper = gamma*tau/c0 (Ma & Huang 2025 Table 1,
    # docs/spec/inverse_free_dtznn.md: h_ours = c0*h_paper).  Recorded for the
    # tables / notes only; it never changes a run.
    info["c0"] = float(coeffs[0]) if coeffs else None
    info["paper_h_factor"] = _paper_h_factor(key)
    if info["c0"] is None and info["paper_h_factor"] is not None:
        info["c0"] = info["paper_h_factor"]               # cdtznn: c0 of its x-recursion
    f = info["paper_h_factor"]
    info["h_rec_paper"] = (info["h_rec"] / f) if (f and info["h_rec"]) else None
    info["h_common_paper"] = (H_COMMON / f) if f else None
    _ = (lookahead_znn_poly, spectral_radius)
    return info


def _paper_h_factor(key):
    """c0 with h_ours = c0 * h_paper for the formulas whose source uses the
    c0-normalised step (Ma & Huang 2025 Table 1 models), else None."""
    try:
        from znnbench.discretizations.inverse_free_dtznn import TABLE1_H_FACTOR
    except Exception:  # noqa: BLE001
        return None
    f = TABLE1_H_FACTOR.get(key)
    return None if f is None or float(f) == 1.0 else float(f)


def rho_of(key, kw_json, h):
    """Spectral radius of the linear-ZNN error recursion of formula `key` at h."""
    info = formula_info(key, kw_json)
    from znnbench.discretizations.zead import lookahead_znn_poly, spectral_radius
    kind = info["rho_kind"]
    if kind == "rk4":
        return float(abs(_rk4_R(h)))
    if kind == "lookahead":
        return float(spectral_radius(lookahead_znn_poly(info["coeffs"], h)))
    if kind == "znn_poly":
        d = _registry().make_discretization(key, **json.loads(kw_json))
        return float(spectral_radius(d.znn_poly(h)))
    return _rho_empirical(key, kw_json, round(float(h), 12))


def decay_rate(rho, tau=0.0, mode="fixed_h", coupled=True):
    """Per-step decay rate used to size the horizon (0 = no requirement).

    Frozen-coefficient rate r = -ln rho.  In fixed-h mode with ``coupled`` the
    rate is reduced to max(r - tau*L_REF, r/4): the tau*||A^{-1} A_dot||
    coupling perturbs the error recursion by O(tau) per step while r ~ h stays
    fixed, and for high-order formulas at small h this slows the decay well
    below the frozen rate (empirical, see E1_NOTES section 2).  In fixed-gamma
    mode r ~ gamma*tau shrinks with tau like the coupling itself, and the
    window diagnostic x0_spread_w showed no leak there, so the frozen rate is
    kept."""
    if not (0.0 < rho < 1.0):
        return 0.0                                 # unstable (diverges) or exact: base horizon
    r = -math.log(rho)
    if coupled and mode == "fixed_h":
        r = max(r - tau * L_REF, r / 4.0)
    return r


def T_needed(key, kw_json, tau, gains, coupled=True):
    """Uncapped horizon for one (formula, tau, mode): max over the gain entries
    (all entries of ``gains`` must share one mode; fixed-h values are batched)."""
    T = T0
    for mode, v in gains:
        h = v if mode == "fixed_h" else v * tau
        r = decay_rate(rho_of(key, kw_json, h), tau, mode, coupled)
        if r > 0.0:
            T = max(T, W_ABS + K_DECAY / r * tau)
    return T


def T_of_tau(key, kw_json, tau, gains, coupled=True):
    """Horizon for one (formula, tau, mode), rounded up to the ladder, capped at T_CAP."""
    T = min(T_needed(key, kw_json, tau, gains, coupled), T_CAP)
    for t in T_LADDER:
        if t >= T - 1e-9:
            return float(t)
    return float(T_CAP)


def gains_by_mode(gains):
    out = {}
    for mode, v in gains:
        out.setdefault(mode, []).append((mode, v))
    return out


def gains_for(key, kw_json):
    """fixed-h values {h_rec, 0.1 if stable} and the fixed-gamma value."""
    info = formula_info(key, kw_json)
    hs = []
    if info["h_rec"]:
        hs.append(round(float(info["h_rec"]), 6))
    if rho_of(key, kw_json, H_COMMON) < 1.0 and not any(abs(h - H_COMMON) < 1e-9 for h in hs):
        hs.append(H_COMMON)
    gains = [("fixed_h", h) for h in hs] + [("fixed_gamma", GAMMA_FIXED)]
    return gains


# ---------------------------------------------------------------------------
# formula selection (conditional on the registry)
# ---------------------------------------------------------------------------
def select_formulas(scale):
    avail = available_discretizations()
    base = LOOKAHEAD + REFERENCE + IMPLICIT
    known = {k for k, _ in base} | set(OPTIONAL) | NOT_E1
    chosen = [(k, kw) for k, kw in base if k in avail]
    for k in OPTIONAL:                               # anticipated new zoo keys
        if k in avail:
            chosen.append((k, {}))
    reg = _registry()
    for k in sorted(avail - known):                  # any other new causal formula
        try:
            d = reg.make_discretization(k)
        except Exception as exc:  # noqa: BLE001
            print(f"[E1] skip new key {k!r}: cannot instantiate with defaults ({exc!r})", file=sys.stderr)
            continue
        if getattr(d, "causal", False) and getattr(d, "expected_order", None) is not None:
            chosen.append((k, {}))
            print(f"[E1] auto-included new causal discretization {k!r}", file=sys.stderr)
    if scale == "mini":
        chosen = [(k, kw) for k, kw in chosen if k in MINI_FORMULAS]
    return chosen


def _problem_entries(instances, Tvals):
    out = []
    for lab in instances:
        key, kw, _role = INSTANCES[lab]
        out.append({"key": key, "kwargs": {**kw, "T": {"$sweep": list(Tvals)}}, "label": lab})
    return out


TSHIFT = {"key": TSHIFT_KEY, "target": "problem", "kwargs": {"t_end": T_END}}
TSHIFT_LATE = {"key": TSHIFT_LATE_KEY, "target": "problem", "kwargs": {"t_end": T_END_LATE}}


def make_config(key, kw, scale):
    kw_json = json.dumps(kw, sort_keys=True)
    info = formula_info(key, kw_json)
    gains = gains_for(key, kw_json)
    taus = TAUS[scale]
    bym = gains_by_mode(gains)
    # one horizon per (formula, tau, mode): a marginally stable fixed-gamma point
    # (h = gamma*tau near h_max, rho ~ 1) no longer stretches the fixed-h runs
    Tmode = {m: {tau: T_of_tau(key, kw_json, tau, g) for tau in taus} for m, g in bym.items()}
    Tfrozen = {m: {tau: T_needed(key, kw_json, tau, g, coupled=False) for tau in taus}
               for m, g in bym.items()}
    Tvals = sorted({T for mp in Tmode.values() for T in mp.values()})
    instances = [lab for lab in INSTANCE_ORDER if lab in INSTANCES]
    if scale == "mini":
        instances = [lab for lab in instances if lab in MINI_INSTANCES]
    exclude = []
    for m, mp in Tmode.items():
        for Tv in Tvals:
            wrong = [tau for tau in taus if mp[tau] != Tv]
            if wrong:
                exclude.append({"problem": {"kwargs": {"T": Tv}}, "tau": wrong, "gain_mode": m})
    others = [lab for lab in instances if lab != "linsys_n4"]
    if others:                                        # constant-A wrapper only on linsys_n4
        exclude.append({"wrappers": CONSTA_KEY, "problem": others})
    late = [lab for lab in instances if lab in LATE_INSTANCES]
    early = [lab for lab in instances if lab not in LATE_INSTANCES]
    if late:                                          # one time-shift wrapper per instance
        exclude.append({"wrappers": TSHIFT_KEY, "problem": late})
    if early:
        exclude.append({"wrappers": TSHIFT_LATE_KEY, "problem": early})
    tag = key + ("" if not kw else "_" + "_".join(f"{a}{b}" for a, b in sorted(kw.items())))
    name = f"E1-{tag}"
    disc = {"key": key, "kwargs": kw, "label": tag}
    hs = [v for m, v in gains if m == "fixed_h"]
    gs = [v for m, v in gains if m == "fixed_gamma"]
    cfg = {
        "name": name,
        "description": {"experiment": EXPERIMENT, "scale": scale, "formula": tag,
                        "catalogue": info, "gains": gains,
                        "T_of_tau": {m: {repr(t): mp[t] for t in taus} for m, mp in Tmode.items()},
                        "T_needed": {m: {repr(t): T_needed(key, kw_json, t, bym[m]) for t in taus}
                                     for m in Tmode},
                        # frozen-rate requirement above T_CAP: transient certainly not decayed
                        "T_capped": {m: [t for t in taus if Tfrozen[m][t] > T_CAP] for m in Tmode},
                        "T0": T0, "K_decay": K_DECAY, "L_ref": L_REF, "frac": FRAC, "T_cap": T_CAP,
                        "W_abs": W_ABS, "t_end": T_END, "t_end_late": T_END_LATE,
                        "late_instances": LATE_INSTANCES,
                        "constA_instance": CONSTA_INSTANCE, "constA_wrapper": CONSTA_KEY},
        "problems": _problem_entries(instances, Tvals),
        "models": [{"key": "znn", "kwargs": {"activation": "linear"}, "label": "znn_linear"}],
        "discretizations": [disc],
        "taus": taus,
        "hs": hs,
        "gammas": gs,
        "infos": ["analytic"],
        "noises": [None],
        "wrappers": [[TSHIFT],                        # alternatives (chains)
                     [{"key": CONSTA_KEY, "target": "problem"}, TSHIFT],
                     [TSHIFT_LATE]],
        "seeds": SEEDS[scale],
        "B": B,
        "T": None,                                    # horizon comes from the problem kwarg T
        "record_every": 1,
        "metrics": {"eps": [1e-3, 1e-6], "frac": FRAC,
                    "extra": ["final_residual", "max_residual", WINDOW_METRIC]},
        "max_wall": 3600 if scale == "mini" else 7200,   # full: longest spec ~20 min idle, margin under load
        "batch_gains": True,                          # the fixed-h values share one simulate call
        "exclude": exclude,
        "chunk_size": 24 if scale == "mini" else 48,
        "balance": True,
    }
    return cfg


def get_configs(scale=None):
    scale = scale or scale_from_env()
    return [make_config(k, kw, scale) for k, kw in select_formulas(scale)]


# ---------------------------------------------------------------------------
# cost model (used by the notes; wall_per_step from the mini run)
# ---------------------------------------------------------------------------
def steps_table(scale=None):
    """Number of specs, trajectories and simulated steps of the sweep."""
    from znnbench.runner import expand
    scale = scale or scale_from_env()
    rows = []
    for cfg in get_configs(scale):
        for s in expand(cfg):
            T = s["problem"]["kwargs"]["T"]
            n = int(round(T / s["tau"]))
            rows.append({"config": cfg["name"], "disc": s["disc"]["key"],
                         "disc_kwargs": json.dumps(s["disc"]["kwargs"], sort_keys=True),
                         "problem": s["problem"]["label"],
                         "constA": any(w["key"] == CONSTA_KEY for w in s["wrappers"]), "tau": s["tau"],
                         "mode": s["gain"]["mode"], "G": len(s["gain"]["values"]), "B": s["B"], "steps": n})
    return rows


def timing_probe(key, kw_json, label, constA, BG, n_steps=300, tau=1e-3, h=0.1):
    """Wall time per step (s) of simulate for one (formula, instance, batch size),
    measured in this process with the sweep's wrappers (single BLAS thread
    recommended: run with OMP/VECLIB threads = 1, as the pool does)."""
    reg = _registry()
    from znnbench.simulate import simulate
    from experiments.tasks.E1_problems import ConstantMatrix, TimeShift, TimeShiftLate
    pkey, pkw, _ = INSTANCES[label]
    pkw = {k: (0 if v == "$seed" else v) for k, v in pkw.items()}
    prob = reg.make_problem(pkey, **pkw, T=T0)
    if constA:
        prob = ConstantMatrix(prob)
    prob = TimeShiftLate(prob) if label in LATE_INSTANCES else TimeShift(prob, t_end=T_END)
    d = reg.make_discretization(key, **json.loads(kw_json))
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        sim = simulate(prob, reg.make_model("znn"), d, tau, T=n_steps * tau, B=BG, gamma=h / tau, seed=0)
    return float(sim["wall"]) / max(int(sim["steps"]), 1)


def estimate_cost(scale="full", probe_steps=300, workers=24, reuse=None):
    """Full-sweep cost: steps of every spec x probed wall time per step of its
    (formula, instance, B*G).  Returns a dict and prints a summary.  ``reuse``:
    a previous estimate JSON whose ``probe_us_per_step`` table is reused (only
    missing (formula, instance, B*G) entries are probed), e.g. to re-cost a
    changed grid without re-timing on a loaded machine."""
    import collections
    rows = steps_table(scale)
    probes = {}
    if reuse:
        old = json.loads(Path(reuse).read_text()).get("probe_us_per_step", {})
        for kk, us in old.items():
            d, kwj, prob, ca, bg = kk.split("|")
            probes[(d, kwj, prob, ca == "True", int(bg))] = 1e-6 * float(us)
    for r in rows:
        k = (r["disc"], r["disc_kwargs"], r["problem"], r["constA"], r["B"] * r["G"])
        if k not in probes:
            try:
                probes[k] = timing_probe(*k, n_steps=probe_steps)
            except Exception as exc:  # noqa: BLE001
                print(f"[E1] probe failed for {k}: {exc!r}", file=sys.stderr)
                probes[k] = float("nan")
    tot, per_disc, per_inst, longest = 0.0, collections.Counter(), collections.Counter(), (0.0, None)
    for r in rows:
        k = (r["disc"], r["disc_kwargs"], r["problem"], r["constA"], r["B"] * r["G"])
        w = probes[k] * r["steps"]
        if not math.isfinite(w):
            continue
        tot += w
        per_disc[r["disc"]] += w
        per_inst[r["problem"] + ("+constA" if r["constA"] else "")] += w
        if w > longest[0]:
            longest = (w, {kk: r[kk] for kk in ("disc", "problem", "constA", "tau", "mode", "steps")})
    out = {"scale": scale, "specs": len(rows), "steps": int(sum(r["steps"] for r in rows)),
           "probes_reused_from": str(reuse) if reuse else None,
           "core_hours": tot / 3600.0, "wall_hours_at_workers": tot / 3600.0 / workers, "workers": workers,
           "per_disc_core_hours": {k: v / 3600 for k, v in per_disc.most_common()},
           "per_instance_core_hours": {k: v / 3600 for k, v in per_inst.most_common()},
           "longest_spec_s": longest[0], "longest_spec": longest[1],
           "probe_us_per_step": {"|".join(map(str, k)): 1e6 * v for k, v in probes.items()}}
    print(f"{out['specs']} specs, {out['steps']:.3e} steps, {out['core_hours']:.1f} core-hours "
          f"(~{out['wall_hours_at_workers']:.2f} h wall at {workers} workers); longest spec "
          f"{longest[0]:.0f} s {longest[1]}")
    for k, v in list(out["per_disc_core_hours"].items())[:8]:
        print(f"  {k:28s} {v:6.2f} core-h")
    for k, v in out["per_instance_core_hours"].items():
        print(f"  {k:28s} {v:6.2f} core-h")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scale", choices=["mini", "full"], default=None)
    ap.add_argument("--out", help="output directory (one sub-folder per formula)")
    ap.add_argument("--chunk-size", type=int, default=None)
    ap.add_argument("--list", action="store_true", help="print the formula catalogue and exit")
    ap.add_argument("--count", action="store_true", help="print spec / step counts and exit")
    ap.add_argument("--estimate-cost", metavar="JSON", nargs="?", const="-",
                    help="probe wall/step of every (formula, instance, B*G) and estimate the sweep cost "
                         "(optionally write the result to JSON)")
    ap.add_argument("--reuse-probes", metavar="JSON", default=None,
                    help="with --estimate-cost: reuse the wall/step probes of an earlier estimate")
    args = ap.parse_args(argv)
    scale = args.scale or scale_from_env()
    os.environ["E1_SCALE"] = scale
    if args.list:
        for k, kw in select_formulas(scale):
            kj = json.dumps(kw, sort_keys=True)
            inf = formula_info(k, kj)
            hp = "" if inf.get("paper_h_factor") is None else (
                f" [source h = h/c0: c0={inf['paper_h_factor']:.4g}, h_rec_paper={inf['h_rec_paper'] or float('nan'):.4g},"
                f" 0.1 -> {inf['h_common_paper']:.4g}]")
            print(f"{k:28s} {kj:12s} claimed={inf['claimed_order']} h_rec={inf['h_rec']} "
                  f"h_max={inf['h_max']} gains={gains_for(k, kj)}{hp}")
        return 0
    if args.estimate_cost:
        res = estimate_cost(scale, reuse=args.reuse_probes)
        if args.estimate_cost != "-":
            Path(args.estimate_cost).write_text(json.dumps(res, indent=1, default=str))
        return 0
    if args.count:
        rows = steps_table(scale)
        tot = sum(r["steps"] for r in rows)
        traj = sum(r["B"] * r["G"] for r in rows)
        print(f"{len(rows)} specs, {traj} trajectories, {tot:.3e} steps")
        return 0
    if not args.out:
        ap.error("--out is required to build")
    from znnbench.runner import MEMGUARD, DEFAULT_PYTHON, build, load_tasks
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lines, n_specs = [], 0
    for cfg in get_configs(scale):
        tf = build(cfg, out / cfg["name"], chunk_size=args.chunk_size)
        tasks = load_tasks(tf)
        lines += [json.dumps(t) for t in tasks]
        n_specs += sum(1 for _ in open(out / cfg["name"] / "specs.jsonl"))
        print(f"{cfg['name']}: {len(tasks)} tasks")
    pool = out / "pool_tasks.jsonl"
    tmp = pool.with_suffix(".tmp")
    tmp.write_text("\n".join(lines) + "\n")
    os.replace(tmp, pool)
    print(f"{len(lines)} tasks, {n_specs} specs -> {pool}")
    print(f"launch: {DEFAULT_PYTHON} {MEMGUARD} pool --tasks {pool.resolve()} --workers 24 --max-gb 200")
    return 0


if __name__ == "__main__":
    sys.exit(main())
