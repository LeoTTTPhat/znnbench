#!/usr/bin/env python
"""E2 -- stability maps: maximal stable step h = gamma*tau per (formula x activation).

Protocol: experiments/PLAN.md (E2) and experiments/E2_NOTES.md (exact definitions).
E2 does NOT use the sweep runner (``znnbench.runner`` cannot bisect, see
docs/spec/runner.md sec. 7).  It is run by the custom task script
``experiments/tasks/E2_stability.py``, which reads the grid defined here through
``get_config(scale)``.  (There is deliberately no ``CONFIG``/``CONFIGS`` symbol,
so this file cannot be mistaken for a runner config.)

What one task does (one task = one formula x one problem x one tau):
for every problem seed, every activation, every initial-error magnitude r and
every initial direction b (B per seed), it locates the upper end h_max of the
stable step range in (0, 3] -- a coarse log grid of h values simulated as ONE
batch, followed by vectorised multi-section ("K-section") rounds on the
bracket -- for two stability criteria:

* ``floor``  (protocol criterion, PLAN.md E2): no divergence and steady
  residual MSSRE(h) <= THETA * max(F, F_ABS), where F is the FORMULA FLOOR =
  the MSSRE of the same formula with LINEAR activation on the same instance,
  direction and magnitude at the formula's root-locus optimal step h_opt
  (fallback when no closed-form theory exists -- implicit and inverse-free
  schemes: the minimum over the coarse h grid);
* ``bounded`` (practical / Lyapunov-type criterion): no divergence and
  MSSRE(h) <= ||E(x0, 0)|| (the steady residual does not exceed the initial
  error).  For linear activation this boundary is the root-locus boundary
  rho(P_h) = 1 up to finite-horizon effects;
* ``nodiv`` (diagnostic, the "no divergence" half of the PLAN definition
  alone): the trajectory stays finite and below the divergence threshold
  (|x| <= 1e10) over T.  Comparing it with ``bounded`` separates blow-up from
  bounded chattering above ||E(0)||.  Its detection limit is horizon
  dependent (a root with rho = 1.01 grows only 400x in 600 steps).

Initial states: x0 = x*(0) + J(0)^{-1} (r u), u uniform on the unit sphere of the
error space, so ||E(x0, 0)||_2 = r EXACTLY (both E2 problems are affine in x).
The same directions u are used for every r, activation and formula (paired
design).

Formulas are filtered by ``znnbench.available()`` at build time; any look-ahead
formula registered later by a sibling workflow (``zead_7i_cang2024``,
``zead_5i_mahuang2025``, ``uhlig_5_*``, ...) is picked up automatically by a
rebuild (prefix rule below), and the inverse-free DTZNNs of
``discretizations/inverse_free_dtznn.py`` (``INVERSE_FREE_KEYS`` of that module,
plus a prefix rule) are added on BOTH problems.  Theory for them
(experiments/tasks/E2_theory.py): ``cdtznn_mahuang2025`` has a LOCAL linearised
theory (block-triangular linearisation; both blocks share the P_h of its
x-recursion, the M-block has the doubled frozen-time shift tau(mu_i + mu_j)), so
it also enters the frozen-A control and the horizon block, but not the
exact-theory headline; ``broyden_dtznn_zhang2012`` has none (its secant matrix
never converges to J^-1).  Review additions (2026-10-18): E4-compatible
criterion ``e4``, horizon block (T = 24), paired scaled-error arm (unit
dependence), 8 instances at full scale.

Scale
-----
``ZNNBENCH_SCALE=mini|full`` (environment; default ``full``) or ``--scale`` when
this file or the task script is run.  ``mini`` = end-to-end validation subset.

Build + launch + analyze (from znn-bench/)
-----------------------------------------
  python experiments/tasks/E2_stability.py build --scale mini --out results/mini/E2
  python tools/memguard.py pool --tasks results/mini/E2/tasks.jsonl --workers 3 --max-gb 40
  python experiments/analyze_E2.py --results results/mini/E2 \
      --figdir figures/mini --tabdir results/tables/mini
Running this file prints the expanded grid and the task count:
  python experiments/configs/E2.py --scale mini
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --------------------------------------------------------------------------
# protocol constants (shared with the task script and analyze_E2.py)
# --------------------------------------------------------------------------
TAU = 1e-2                 # PLAN: tau = 1e-2, fixed-h
T = 6.0                    # PLAN: T = 6 s, steady window = last 25 %
FRAC = 0.25
H_MAX_SEARCH = 3.0         # PLAN: bisection on h in (0, 3]
H_GRID_MIN = 1e-4          # smallest coarse-grid step (h_max below it is reported as censored)
N_GRID = 61                # log-spaced coarse grid, ratio (3/1e-4)^(1/60) = 1.188
THETA = 10.0               # PLAN: "final residual < 10 x floor"
F_ABS = 1e-12              # round-off floor below which F is not trusted
DIV_THRESHOLD = 1e10       # simulate default (|x| > 1e10 -> diverged)
R_LIST = [0.1, 1.0, 10.0]  # PLAN: ||e(0)|| in {0.1, 1, 10}  (e = E(x0, 0), see above)
R_MAIN = 1.0               # magnitude used for the main heat map and the theory comparison
SEEDS = [0, 1, 2]          # PLAN: 3 problem instances (mini; and the direction streams)
# full scale: 8 instances.  The B directions of one instance give h_max within +-0.5 %
# of each other (pseudo-replicates), while instances differ by up to +-10 % (mini,
# linear activation): the instance is the statistical unit, so the headline
# theory-vs-measured ratios get seed-cluster intervals and need more than 3 seeds.
SEEDS_FULL = [0, 1, 2, 3, 4, 5, 6, 7]
B = 4                      # initial directions per instance
REFINE = {"rounds": 3, "K": 7}   # bracket shrinks by (K+1)^rounds = 512
# E4-compatible diagnostic criterion (mirrors experiments/configs/E4.py STAB_*; E4's
# P2 recomputes its own h_max with this criterion, explicit Euler, tau = 1e-3):
# finite, MSSRE <= RHO ||E(0)||, max over the last T/8 <= GROWTH x max over the
# preceding T/8 (or <= 1e-10), and max_k ||E_k|| <= TRANSIENT ||E(0)||.
E4_CRITERION = {"rho": 0.1, "growth": 2.0, "transient": 10.0}

PROBLEMS = [
    {"key": "linear_system", "kwargs": {"n": 4, "kappa": 10.0, "freq": 1.0, "T": T}, "label": "linear_system"},
    # seed_offset: with equal seeds `matrix_inversion` would draw exactly the same A(t)
    # as `linear_system` (both build TVMatrix first from default_rng(seed)); the
    # offset gives independent instances (seeds 100, 101, 102), so agreement between
    # the two problems is not trivially inherited from a shared A(t).
    {"key": "matrix_inversion", "kwargs": {"n": 4, "kappa": 10.0, "freq": 1.0, "T": T}, "label": "matrix_inversion",
     "seed_offset": 100},
]
# control: linear_system with A frozen at A(0) (b(t) still time varying).  For
# linear ZNN the error recursion is then EXACTLY the frozen root locus P_h
# (the tau*A^{-1}A_dot coupling vanishes), which isolates the finite-horizon
# effect from the time-variation effect.  Linear activation only.
CONTROL_PROBLEM = {"key": "linear_system", "kwargs": {"n": 4, "kappa": 10.0, "freq": 1.0, "T": T},
                   "label": "linear_system_constA", "freeze_A": True}

# activations: the canonical entity group (experiments/keys.py GROUPS['act']),
# default parameters of the zoo (docs/ZOO.md, docs/spec/activations.md)
ACTIVATIONS = ["linear", "power_sigmoid", "hyperbolic_sine", "sign_bi_power", "fixed_time",
               "vaf", "pvaf1", "predefined_exp"]

# formulas: every causal explicit look-ahead / multistep formula + RK4 reference
# + implicit-error Euler (control: unconditionally stable for linear Phi)
FORMULAS_FULL = ["euler", "taylor_zhang", "zead_4i_general", "zead_4i_6321", "zead_4i_fifd",
                 "zead_5i", "zead_6i_guo2017", "zead_6i_qiu2019", "zead_7i_general",
                 "zead_7i_cang2024", "zead_8i_wu2020", "zead_8i_chen2019", "zead_10i_wu2020",
                 "zead_11i_chen2023", "uhlig_3_4", "uhlig_4_6", "ab2", "ab3", "ab4", "rk4",
                 "implicit_euler_err"]
# exact duplicates of registered formulas (same coefficients) -- not re-run
DUPLICATES = {"uhlig_2_3": "zead_5i", "zead_5i_general": "zead_5i"}
# later additions by sibling workflows are picked up by prefix
AUTO_PREFIXES = ("zead_", "uhlig_")
# inverse-free DTZNNs (discretizations/inverse_free_dtznn.py registers
# broyden_dtznn_zhang2012 and cdtznn_mahuang2025; its INVERSE_FREE_KEYS dict is
# read as well, so later additions to that module are picked up)
INVERSE_FREE_PREFIXES = ("inverse_free", "if_dtznn", "dtznn_if", "broyden_dtznn", "cdtznn")
INVERSE_FREE_PROBLEMS = ("linear_system", "matrix_inversion")
# formulas without a closed-form linear error recursion: no theory, so they are
# not run in the tau-sensitivity / frozen-A control blocks (which compare with theory)
NO_THEORY = ("implicit_euler_err",)
FORMULAS_MINI = ["euler", "taylor_zhang", "zead_5i", "zead_5i_mahuang2025", "zead_7i_cang2024", "uhlig_4_6",
                 "ab3", "rk4", "implicit_euler_err"]
# zead_5i_mahuang2025 is the exact-inverse twin of cdtznn_mahuang2025 (same x-recursion):
# the paired ratio CDTZNN / twin isolates the effect of J^-1 -> M_k
INVERSE_FREE_MINI = ["broyden_dtznn_zhang2012", "cdtznn_mahuang2025"]
# inverse-free schemes with a LOCAL linearised theory (E2_theory.Theory, theory_type
# 'local'): they enter the frozen-A control and the horizon block, not the tau block
INVERSE_FREE_LOCAL_THEORY = ("cdtznn_mahuang2025",)

# tau-sensitivity (linear activation only): measured -> root-locus theory as tau -> 0.
# T stays 6 s, so N = T/tau grows as tau -> 0 and the finite-horizon detection
# limit (an O(1/N) = O(tau) overshoot) shrinks at the same rate as the
# O(tau ||A^-1 A_dot||) time-variation shift.  The frozen-A control is therefore
# run at every tau too: h(LS)/h(control) isolates the time-variation shift and
# h(control)/h_theory the finite-horizon effect.
TAU_SENS_FULL = [3e-2, 3e-3, 1e-3]
TAU_SENS_MINI = [3e-2, 3e-3]

# horizon sensitivity (linear activation, ||E(0)|| = 1, tau = 1e-2): T = 24 s
# (N = 2400) on linear_system and the frozen-A control.  At T = 6 a weakly unstable
# mode that starts small (e.g. the CDTZNN M-block, rho ~ 1.004/step) does not reach
# the bounded threshold before T, so the T = 6 h_max over-states the asymptotic limit.
T_LONG = [24.0]

# paired scaled-error arm (REVIEW_CHECKLIST item 1, unit dependence): the same runs
# with E expressed in units of 1/s of the problem unit (experiments.tasks.R_common:
# ScaledError; E_s = s E, residuals recorded in the original unit, the same x0, so
# ||E(0)|| and both criteria are unchanged).  Linear activation is invariant (control);
# power/sign-type activations are not.  Note: a unit change is NOT a change of r:
# it scales the initial error AND the truncation-error forcing (the floor) by s,
# whereas the r sweep scales only the initial error.
ERROR_SCALES = [0.1, 10.0]
SCALED_FORMULAS_FULL = ["euler", "taylor_zhang", "zead_5i", "zead_7i_cang2024", "uhlig_4_6", "ab3", "rk4"]
SCALED_FORMULAS_MINI = ["euler", "zead_5i", "rk4"]
SCALED_PROBLEMS_FULL = ("linear_system", "matrix_inversion")
SCALED_PROBLEMS_MINI = ("linear_system",)


def _available_discs():
    try:
        import znnbench
        return set(znnbench.available()["discretizations"])
    except Exception:                       # pragma: no cover - zoo mid-edit
        return set()


def _inverse_free_keys(avail):
    keys = {k for k in avail if k.startswith(INVERSE_FREE_PREFIXES)}
    try:
        from znnbench.discretizations import inverse_free_dtznn as _ifd
        keys |= {k for k in getattr(_ifd, "INVERSE_FREE_KEYS", {}) if k in avail}
    except Exception:                       # pragma: no cover - module absent / mid-edit
        pass
    return sorted(keys)


def _auto_formulas(avail):
    """Look-ahead formulas registered later by sibling workflows (prefix rule)."""
    extra = []
    for k in sorted(avail):
        if k.startswith(AUTO_PREFIXES) and k not in FORMULAS_FULL and k not in DUPLICATES:
            extra.append(k)
    return extra


def _with_theory(formulas, inverse_free):
    return [f for f in formulas if f not in NO_THEORY and f not in inverse_free]


def _e4_criterion():
    """E4's P2 stability constants (read from experiments/configs/E4.py when importable,
    so that the E2 diagnostic stays in sync with E4; fallback = the values above)."""
    crit = dict(E4_CRITERION)
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("e4_config", ROOT / "experiments" / "configs" / "E4.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        crit = {"rho": float(mod.STAB_RHO), "growth": float(mod.STAB_GROWTH),
                "transient": float(mod.STAB_TRANSIENT)}
    except Exception:                       # pragma: no cover - E4 mid-edit
        pass
    return crit


def h_grid(n=N_GRID, lo=H_GRID_MIN, hi=H_MAX_SEARCH):
    return [float(v) for v in np.geomspace(lo, hi, n)]


def get_scale(argv=None) -> str:
    s = os.environ.get("ZNNBENCH_SCALE", "full")
    if argv:
        for i, a in enumerate(argv):
            if a == "--scale" and i + 1 < len(argv):
                s = argv[i + 1]
            elif a.startswith("--scale="):
                s = a.split("=", 1)[1]
    if s not in ("mini", "full"):
        raise ValueError(f"scale must be mini or full, got {s!r}")
    return s


def get_config(scale: str | None = None) -> dict:
    """The E2 grid for ``scale`` (mini|full), filtered by the registry."""
    scale = scale or get_scale()
    avail = _available_discs()
    inverse_free_all = _inverse_free_keys(avail)
    if scale == "full":
        formulas = [f for f in FORMULAS_FULL if f in avail] + _auto_formulas(avail)
        inverse_free = list(inverse_free_all)
        acts = list(ACTIVATIONS)
        seeds, Bn, r_list = list(SEEDS_FULL), B, list(R_LIST)
        tau_sens = list(TAU_SENS_FULL)
        refine = dict(REFINE)
        scaled_formulas = [f for f in SCALED_FORMULAS_FULL if f in formulas]
        scaled_problems = list(SCALED_PROBLEMS_FULL)
    else:
        formulas = [f for f in FORMULAS_MINI if f in avail]
        inverse_free = [f for f in INVERSE_FREE_MINI if f in inverse_free_all]
        acts = list(ACTIVATIONS)
        seeds, Bn, r_list = list(SEEDS), B, list(R_LIST)
        tau_sens = list(TAU_SENS_MINI)
        refine = dict(REFINE)
        scaled_formulas = [f for f in SCALED_FORMULAS_MINI if f in formulas]
        scaled_problems = list(SCALED_PROBLEMS_MINI)
    sens_formulas = _with_theory(formulas, inverse_free_all)
    local_formulas = [f for f in inverse_free if f in INVERSE_FREE_LOCAL_THEORY]
    cfg = {
        "name": f"E2_{scale}",
        "scale": scale,
        "tau": TAU, "T": T, "frac": FRAC,
        "h_grid": h_grid(),
        "theta": THETA, "F_abs": F_ABS, "div_threshold": DIV_THRESHOLD,
        "r_list": r_list, "r_main": R_MAIN,
        "seeds": seeds, "B": Bn,
        "refine": refine,
        "problems": PROBLEMS,
        "control_problem": CONTROL_PROBLEM,
        "activations": acts,
        "formulas": formulas,
        "inverse_free_formulas": inverse_free,
        "inverse_free_problems": list(INVERSE_FREE_PROBLEMS),
        # frozen-A control: formulas with an exact theory + inverse-free ones with a local theory
        "control_formulas": sens_formulas + local_formulas,
        "tau_sens": tau_sens,
        "tau_sens_formulas": sens_formulas,
        "tau_sens_control": True,                  # frozen-A control at every tau as well
        "horizon_T": list(T_LONG),                 # horizon block (linear, r_main, LS + control)
        "horizon_formulas": sens_formulas + local_formulas,
        "error_scales": list(ERROR_SCALES),        # paired scaled-error arm (all activations)
        "scaled_formulas": scaled_formulas,
        "scaled_problems": scaled_problems,
        "e4_criterion": _e4_criterion(),
        "duplicates": DUPLICATES,
    }
    return cfg


def main(argv=None):
    ap = argparse.ArgumentParser(description="print the E2 grid")
    ap.add_argument("--scale", default=None, choices=["mini", "full"])
    a = ap.parse_args(argv)
    cfg = get_config(a.scale or get_scale())
    from experiments.tasks.E2_stability import expand_tasks  # noqa: E402
    tasks = expand_tasks(cfg)
    print(f"scale={cfg['scale']}  formulas={len(cfg['formulas'])}: {cfg['formulas']}")
    print(f"activations={cfg['activations']}  seeds={cfg['seeds']}  B={cfg['B']}  r={cfg['r_list']}")
    print(f"inverse-free (on {cfg['inverse_free_problems']}): {cfg['inverse_free_formulas']}")
    print(f"tau-sensitivity {cfg['tau_sens']} on {cfg['tau_sens_formulas']}")
    print(f"horizon T={cfg['horizon_T']} on {cfg['horizon_formulas']}")
    print(f"scaled-error arm s={cfg['error_scales']} on {cfg['scaled_formulas']} x {cfg['scaled_problems']}")
    kinds = {}
    for t in tasks:
        kinds[t["kind"]] = kinds.get(t["kind"], 0) + 1
    print(f"tasks: {len(tasks)} {kinds}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
