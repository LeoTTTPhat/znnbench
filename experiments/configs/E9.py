"""E9 -- GNN vs ZNN vs prediction-correction vs Uhlig vs Newton at matched cost.

Sweep configs for ``znnbench.runner`` (see experiments/PLAN.md, section E9, and
experiments/E9_NOTES.md for the full protocol, rationale and cost estimate).

Scale
-----
The scale is read from the environment variable ``E9_SCALE`` (fallback
``ZNN_SCALE``), values ``mini`` (default, pipeline validation) or ``full``
(paper sweep).  ``experiments/tasks/E9_build.py --scale full`` sets it for you.
Running this file directly prints a summary::

    python experiments/configs/E9.py --scale full

Method groups (one runner config each, because the gains differ per group)
--------------------------------------------------------------------------
* ``E9_znn_<formula>``  linear ZNN + causal look-ahead formula (Euler,
  Taylor-Zhang, ZeaD-5i = Uhlig 2_3, Uhlig 3_4, Uhlig 4_6, plus, when the zoo
  registers them, ``zead_7i_cang2024`` and any ``uhlig_5_*`` / higher keys).
  fixed-h mode, h in {h_opt, h_opt/2} where h_opt minimises the spectral
  radius of the linear-ZNN error recursion (computed from the formula's own
  coefficients at build time, ``zead.lookahead_stability``).
* ``E9_gnn_<problem>``  GNN + Euler at fixed *normalised* step
  eta = gamma*tau*sigma_bar^2 in {0.5, 1.0, 1.5, 1.9, 2.1, 2.5}, sigma_bar = max
  over the problem instances and t in [0, T] of sigma_max(J(x*(t), t)).  The
  explicit GNN step is stable only for eta < 2, so 1.9 / 2.1 bracket the limit
  and eta = 2.5 is a deliberate instability check.  The runner's h is gamma*tau = eta / sigma_bar^2.
* ``E9_gnnlag``  GNN + Euler, fixed-gamma mode, small tau (mini 3e-4, full 1e-5),
  gamma grid (lag-floor O(1/gamma) test); every gamma satisfies
  gamma*tau*sigma_bar^2 < 1.9.  The analysis fits the slope only on gammas that
  are transient-free and quasi-static (gamma*sigma_min^2 >= 3*omega).
* ``E9_pc``  Davidenko predictor (Simonetto et al. 2016, Eq. (7)) with
  pc_newton / pc_newton_causal / pc_gradient / pc_gradient_causal /
  newton_tracking / newton_tracking_causal, C in {1, 2, 3} correction steps,
  plus the open-loop Davidenko flow under Euler (negative control).  The gain
  is unused by these methods (h = 1 is a placeholder).

Keys that a concurrent zoo workflow may add later are picked up at build time:
only keys present in ``znnbench.available()`` are included, so re-running the
build after the zoo grows adds their chunks (unchanged chunks keep their ids).
"""
from __future__ import annotations

import os
import re
import sys
from functools import lru_cache

import numpy as np

# --------------------------------------------------------------------------
# scale
# --------------------------------------------------------------------------
SCALES = ("mini", "full")


def get_scale() -> str:
    s = (os.environ.get("E9_SCALE") or os.environ.get("ZNN_SCALE") or "mini").strip().lower()
    if s not in SCALES:
        raise ValueError(f"E9_SCALE must be one of {SCALES}, got {s!r}")
    return s


T_HORIZON = 6.0                     # PLAN default: T = 6 s, steady window = last 25 %
STEADY_FRAC = 0.25

PARAMS = {
    # mini lag sweep at tau = 3e-4 so that gamma = 30 is allowed (eta <= 1.9): two quasi-static
    # gammas (gamma*sigma_min^2 >= 3*freq) on linsys, which a fit needs.
    # 1.2e-2 (review 2026-10-18): the look-ahead formulas cost 1.04-1.17x ZNN-Euler per step, so at the
    # budget "ZNN-Euler at tau = 1e-2" they run at tau = 1.04e-2..1.17e-2; without a grid point there the
    # bracket reached the (unstable) 3e-2 / 2e-2 point and the comparison was n/a
    "mini": dict(taus=[3e-2, 1.2e-2, 1e-2, 3e-3, 1e-3], seeds=[0, 1], B=2,
                 tau_lag=3e-4, lag_gammas=[1.0, 3.0, 10.0, 30.0], lag_record_every=1,
                 timing_repeats=1, chunk_size=12, traces=200),
    # dense around 2e-3..2e-2, where the order-5/6 formulas sit between the round-off floor
    # and their (time-varying-J) instability, so their asymptotic range has >= 3 points;
    # 1.2e-2 and 1.2e-3 bracket both matched-cost budgets of the look-ahead formulas tightly.
    # 10 seeds (review 2026-10-18, PLAN default 3): the problem instance is the only real replication
    # unit -- the B initial states give identical steady states (within-seed spread of MSSRE ~1e-14
    # decades) -- and a cluster bootstrap over 3 seeds has only 10 distinct resamples
    "full": dict(taus=[1e-1, 5e-2, 3e-2, 2e-2, 1.2e-2, 1e-2, 7e-3, 5e-3, 3e-3, 2e-3, 1.2e-3, 1e-3, 5e-4,
                       3e-4, 1e-4],
                 seeds=list(range(10)), B=4,
                 # 300 and 700 densify the quasi-static range of tvqp (sigma_min^2 ~ 0.008 -> gamma >~ 400)
                 tau_lag=1e-5, lag_gammas=[1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 300.0, 500.0, 700.0,
                                           1000.0],
                 lag_record_every=10, timing_repeats=1, chunk_size=12, traces=200),
}

# --------------------------------------------------------------------------
# problems (all with n_x = 8; kappa = 10, frequency 1 rad/s as in PLAN.md)
# --------------------------------------------------------------------------
PROBLEMS = {
    "linsys": {"key": "linear_system",
               "kwargs": {"n": 8, "kappa": 10, "freq": 1.0, "seed": "$seed"},
               "label": "linsys_n8"},
    # nonlinear gradient field (E = P e + beta tanh e): Newton is not exact in one step,
    # and pc_gradient uses Simonetto's exact Eq. (8) step x - alpha E ('gradient_field').
    "tvuo": {"key": "tv_unconstrained_opt",
             "kwargs": {"variant": "manufactured", "n": 8, "kappa": 10, "freq": 1.0, "beta": 1.0,
                        "seed": "$seed"},
             "label": "tvuo_manuf_n8"},
    # equality-constrained TV-QP in KKT form, n = 6 primal + p = 2 multipliers
    "tvqp": {"key": "tvqp_eq",
             "kwargs": {"variant": "random", "n": 6, "p": 2, "kappa": 10, "freq": 1.0, "seed": "$seed"},
             "label": "tvqp_eq_n6p2"},
}
PROBLEM_ORDER = ["linsys", "tvuo", "tvqp"]

# --------------------------------------------------------------------------
# methods
# --------------------------------------------------------------------------
BASE_FORMULAS = ["euler", "taylor_zhang", "zead_5i", "uhlig_3_4", "uhlig_4_6"]
OPTIONAL_FORMULAS = ["zead_7i_cang2024"]
OPTIONAL_FORMULA_PATTERNS = [r"^uhlig_[5-9]_\w+$"]      # higher-order Uhlig formulas, if registered
# 1.9 / 2.1 bracket the explicit-Euler limit eta = 2 tightly (H1b); 2.5 is the clear instability check
GNN_ETAS = [0.5, 1.0, 1.5, 1.9, 2.1, 2.5]
GNN_ETA_PRIMARY = 1.5
PC_KEYS = ["pc_newton", "pc_newton_causal", "pc_gradient", "pc_gradient_causal",
           "newton_tracking", "newton_tracking_causal"]
PC_CS = [1, 2, 3]
METRICS = {"eps": [1e-3, 1e-6], "frac": STEADY_FRAC, "extra": ["final_residual", "initial_residual"]}


def _available():
    import znnbench
    return znnbench.available()


def znn_formulas(avail=None) -> list:
    """Look-ahead formulas present in the registry, in a fixed order."""
    avail = avail or _available()
    discs = set(avail["discretizations"])
    out = [k for k in BASE_FORMULAS + OPTIONAL_FORMULAS if k in discs]
    extra = sorted(k for k in discs
                   if any(re.match(p, k) for p in OPTIONAL_FORMULA_PATTERNS) and k not in out)
    return out + extra


@lru_cache(maxsize=64)
def formula_stability(key: str) -> dict:
    """h_max / h_opt of the linear-ZNN error recursion of a look-ahead formula
    (``zead.lookahead_stability`` on the formula's own coefficients)."""
    from znnbench.registry import make_discretization
    from znnbench.discretizations.zead import lookahead_stability
    d = make_discretization(key)
    if not hasattr(d, "c_next"):
        return {"h_max": float("nan"), "h_opt": float("nan"), "coeffs": None,
                "expected_order": getattr(d, "expected_order", None)}
    coeffs = [float(d.c_next), *[float(c) for c in d.c_past]]
    st = lookahead_stability(coeffs)
    return {"h_max": float(st["h_max"]), "h_opt": float(st["h_opt"]) if st["h_opt"] else float("nan"),
            "rho_opt": st.get("rho_opt"), "coeffs": coeffs,
            "expected_order": getattr(d, "expected_order", None)}


def formula_gains(key: str) -> list:
    """fixed-h values for a formula: [h_opt, h_opt/2] (h_opt = 1 for Euler)."""
    st = formula_stability(key)
    h = st["h_opt"]
    if not (h and np.isfinite(h) and h > 0):
        return []
    return [float(f"{h:.6g}"), float(f"{h / 2:.6g}")]


# --------------------------------------------------------------------------
# Jacobian bounds (GNN step-size normalisation, lag-floor theory)
# --------------------------------------------------------------------------
def make_instance(pname: str, seed: int):
    from znnbench.registry import make_problem
    comp = PROBLEMS[pname]
    kw = {k: (seed if v == "$seed" else v) for k, v in comp["kwargs"].items()}
    return make_problem(comp["key"], **kw)


def dense_jacobian(problem, x, d):
    """J(x, d) built through the Problem API: rows of J = JT(x, d, e_i)."""
    m = problem.m
    X = np.tile(np.asarray(x, float).reshape(1, -1), (m, 1))
    return problem.JT(X, d, np.eye(m))


@lru_cache(maxsize=64)
def jacobian_bounds(pname: str, seed: int, T: float = T_HORIZON, n_t: int = 601) -> dict:
    """max_t sigma_max(J(x*(t), t)), min_t sigma_min(J(x*(t), t)) and the
    continuous-time GNN lag constant c_lag = max_{t in steady window}
    ||J^{-T} x*_dot(t)|| (so that MSSRE_GNN ~ c_lag / gamma), evaluated on
    the exact solution.  For the manufactured TV-opt problem J(x*) = P + beta I
    is the upper bound of sigma_max over all x (sech^2 <= 1)."""
    P = make_instance(pname, seed)
    ts = np.linspace(0.0, T, n_t)
    smax, smin, clag = 0.0, np.inf, 0.0
    t_ss = T * (1.0 - STEADY_FRAC)
    dt = 1e-5
    for t in ts:
        xs = P.x_star(t)
        J = dense_jacobian(P, xs, P.data(t))
        s = np.linalg.svd(J, compute_uv=False)
        smax, smin = max(smax, float(s[0])), min(smin, float(s[-1]))
        if t >= t_ss:
            xd = (P.x_star(min(t + dt, T + dt)) - P.x_star(max(t - dt, 0.0))) / (
                min(t + dt, T + dt) - max(t - dt, 0.0))
            clag = max(clag, float(np.linalg.norm(np.linalg.solve(J.T, xd))))
    return {"sigma_max": smax, "sigma_min": smin, "c_lag": clag}


def sigma_bar(pname: str, seeds) -> float:
    return max(jacobian_bounds(pname, int(s))["sigma_max"] for s in seeds)


# --------------------------------------------------------------------------
# configs
# --------------------------------------------------------------------------
def _common(p: dict, name: str, description: str) -> dict:
    return {"name": name, "description": description, "seeds": list(p["seeds"]), "B": p["B"],
            "T": T_HORIZON, "record_every": 1, "metrics": METRICS, "traces": {"points": p["traces"]},
            "max_wall": 3600, "timing_repeats": p["timing_repeats"], "chunk_size": p["chunk_size"],
            "balance": True, "infos": ["analytic"], "noises": [None], "wrappers": [None]}


def get_configs(scale: str = None) -> list:
    scale = scale or get_scale()
    p = PARAMS[scale]
    avail = _available()
    probs = [PROBLEMS[k] for k in PROBLEM_ORDER if PROBLEMS[k]["key"] in avail["problems"]]
    cfgs = []

    # (1) ZNN + look-ahead formulas, fixed h per formula
    for f in znn_formulas(avail):
        hs = formula_gains(f)
        if not hs:
            continue
        st = formula_stability(f)
        c = _common(p, f"E9_znn_{f}",
                    f"linear ZNN + {f}, fixed-h, h in {hs} (h_opt={st['h_opt']:.6g}, h_max={st['h_max']:.6g})")
        c.update({"problems": probs, "models": [{"key": "znn", "label": "znn"}],
                  "discretizations": [f], "taus": p["taus"], "hs": hs, "batch_gains": True})
        cfgs.append(c)

    # (2) GNN at fixed normalised step eta = gamma*tau*sigma_bar^2
    if "gnn" in avail["models"]:
        for pk in PROBLEM_ORDER:
            if PROBLEMS[pk]["key"] not in avail["problems"]:
                continue
            sb = sigma_bar(pk, p["seeds"])
            hs = [float(f"{eta / sb ** 2:.8g}") for eta in GNN_ETAS]
            c = _common(p, f"E9_gnn_{pk}",
                        f"GNN + Euler on {pk}: eta = gamma*tau*sigma_bar^2 in {GNN_ETAS}, "
                        f"sigma_bar = {sb:.6g}, runner h = eta/sigma_bar^2 = {hs}")
            c.update({"problems": [PROBLEMS[pk]], "models": [{"key": "gnn", "label": "gnn"}],
                      "discretizations": ["euler"], "taus": p["taus"], "hs": hs, "batch_gains": True})
            cfgs.append(c)

        # (3) GNN lag floor: fixed gamma, small tau, gamma*tau*sigma_bar^2 < 1.9 for every problem
        sb_all = max(sigma_bar(pk, p["seeds"]) for pk in PROBLEM_ORDER
                     if PROBLEMS[pk]["key"] in avail["problems"])
        gmax = 1.9 / (p["tau_lag"] * sb_all ** 2)
        gammas = [g for g in p["lag_gammas"] if g <= gmax]
        c = _common(p, "E9_gnnlag",
                    f"GNN + Euler, fixed-gamma lag-floor sweep at tau = {p['tau_lag']:g}, "
                    f"gamma in {gammas} (eta_max = {max(gammas) * p['tau_lag'] * sb_all ** 2:.3g})")
        c.update({"problems": probs, "models": [{"key": "gnn", "label": "gnn"}],
                  "discretizations": ["euler"], "taus": [p["tau_lag"]], "gammas": gammas,
                  "batch_gains": True, "record_every": p["lag_record_every"], "timing_repeats": 0,
                  "traces": None, "chunk_size": 1})      # long specs (6e5 steps at full): one per task
        cfgs.append(c)

    # (4) prediction-correction / Newton tracking / Davidenko (gain unused)
    if "davidenko" in avail["models"]:
        discs = [{"key": k, "kwargs": {"C": {"$sweep": PC_CS}}, "label": k + "(C={C})"}
                 for k in PC_KEYS if k in avail["discretizations"]]
        discs.append({"key": "euler", "label": "euler"})        # open-loop Davidenko flow
        c = _common(p, "E9_pc",
                    "Davidenko predictor + prediction-correction / Newton tracking (C corrections); "
                    "gain unused (h = 1 placeholder); davidenko + euler = open-loop control")
        c.update({"problems": probs, "models": [{"key": "davidenko", "label": "davidenko"}],
                  "discretizations": discs, "taus": p["taus"], "hs": [1.0]})
        cfgs.append(c)
    return cfgs


def _summary(scale):
    from znnbench.runner import expand
    total_specs = total_steps = 0
    for c in get_configs(scale):
        specs = expand(c)
        steps = sum(round(T_HORIZON / s["tau"]) * len(s["gain"]["values"]) * s["B"] for s in specs)
        nsteps = sum(round(T_HORIZON / s["tau"]) for s in specs)
        total_specs += len(specs)
        total_steps += nsteps
        print(f"{c['name']:24s} specs={len(specs):5d}  sim-steps={nsteps:10d}  traj-steps={steps:11d}  "
              f"| {c['description']}")
    print(f"TOTAL specs={total_specs} sim-steps={total_steps}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="print the E9 sweep summary")
    ap.add_argument("--scale", choices=SCALES, default=None)
    a = ap.parse_args()
    if a.scale:
        os.environ["E9_SCALE"] = a.scale
    _summary(get_scale())
