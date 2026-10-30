#!/usr/bin/env python
"""E8 -- implicit (consistent) vs explicit discretization of sign-type ZNN.

Protocol: experiments/PLAN.md (E8) and experiments/E8_NOTES.md; the schemes are
specified in docs/spec/consistent.md.  This file defines ``get_configs()`` for
the sweep runner (``znnbench.runner``) and returns five configs:

``E8_ti``        time-invariant ``linear_system`` (freq = 0, so dE/dt = 0):
                 finite-step convergence (steps until ||E|| < 1e-13, metric
                 ``t_eps_1e-13``), chattering (``chatter_vec``, ``rate_alt``;
                 the norm-based ``chatter`` is blind to e -> -e cycles), and the
                 stable h range.  On a time-invariant problem the iteration
                 x_k does not depend on tau at fixed h (only tau*x_dot enters the
                 update, see E8_NOTES.md), so a single tau is used.
``E8_tv``        time-varying ``linear_system`` and ``matrix_inversion``: MSSRE vs
                 h, MSSRE vs tau (empirical orders at fixed h), chattering and the
                 stable h range.
``E8_formulas``  extension: the error-memory (EM) rate with the exact error map
                 on every registered look-ahead formula of the entity group
                 ``keys.GROUPS['disc']`` (and any Uhlig formula of order >= 5
                 registered later), against the explicit formula.
``E8_scale_ti``, ``E8_scale_tv``  error-unit arm (review 2026-10-18): the TI and
                 the TV linear system with the error expressed in units of 1/s
                 (problem wrapper experiments.tasks.R_common:ScaledError, s in
                 ERROR_SCALES), paired with the base configs (s = 1) by seed and x0.
                 The non-linear activations have no length scale, so the explicit
                 band, the dead zone and the explicit stable h window depend on s.

Full scale uses 10 problem instances (SEEDS_FULL) for every config except the
extension; mini uses 3 (a subset, so mini run ids are full run ids).

Every list of registry keys is filtered at build time by
``znnbench.available()`` (and, for the extension, by whether the key is a
``LookAhead`` formula), so a later rebuild automatically picks up components
that sibling workflows add (e.g. ``zead_7i_cang2024``, ``uhlig_5_*``).

Scale
-----
``ZNNBENCH_SCALE=mini|full`` (environment, default ``full``), or ``--scale``
when this file is run as a script.  ``mini`` is the end-to-end validation with
the same statistics (3 problem seeds x B = 4 initial states) and every
activation of the full sweep, so that every analysis code path is exercised:
the time-invariant config is identical to the full one; the time-varying
config drops tau = 3e-4, runs matrix_inversion only at tau in {1e-2, 3e-3},
and runs the activations SBP r = 0.25 / 0.75 and fixed_time only on
linear_system at tau in {1e-2, 3e-3}; the extension uses sign and 4 formulas.
``full`` is the paper sweep.

Build + launch
--------------
  python experiments/configs/E8.py --scale mini --out results/mini/E8
  # -> results/mini/E8/<config>/{tasks.jsonl,chunks,results} + results/mini/E8/tasks_all.jsonl
  python tools/memguard.py pool --tasks results/mini/E8/tasks_all.jsonl --workers 3 --max-gb 40
  python experiments/analyze_E8.py --results results/mini/E8 \
      --figdir figures/mini --tabdir results/tables/mini

(The standard runner CLI works too:
``ZNNBENCH_SCALE=mini python -m znnbench.runner build experiments/configs/E8.py --out results/mini/E8``
writes the same per-config folders; launch each ``<config>/tasks.jsonl`` or
concatenate them.)
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --------------------------------------------------------------------------
# protocol constants (shared with analyze_E8.py)
# --------------------------------------------------------------------------
H_GRID = [0.01, 0.03, 0.1, 0.3, 0.6, 1.0, 1.5, 3.0, 10.0]   # fixed-h step sizes h = gamma*tau
SEEDS = [0, 1, 2]                                           # problem instances (and x0 streams), mini
# Full scale: 10 instances.  Review 2026-10-18: the steady-state metrics of the B = 4 initial states of
# one instance are identical (consistent schemes) or nearly so, i.e. the instance is the experimental
# unit; headline CIs are cluster (instance) bootstraps and need more than 3 clusters.  Seeds 0-2 are a
# subset, so every mini run id is also a full-scale run id.
SEEDS_FULL = list(range(10))
B = 4                                                       # initial states per instance
EPS_EXACT = 1e-13                                           # "exact" convergence threshold (TI)

# discretizations: explicit references, implicit/consistent schemes, ablation
DISCS_EULER = ["euler", "implicit_euler_err", "peb_euler", "exact_flow_euler", "semi_implicit_euler"]
DISCS_TZ = ["taylor_zhang", "implicit_taylor_zhang", "peb_taylor_zhang", "implicit_taylor_zhang_naive"]
DISCS = DISCS_EULER + DISCS_TZ
EXPLICIT = ("euler", "taylor_zhang")

# activations (model = base 'znn', which honours the consistent-map request protocol)
ACT_SIGN = {"key": "znn", "kwargs": {"activation": "sign"}, "label": "sign"}
ACT_SBP = {"key": "znn", "kwargs": {"activation": "sign_bi_power", "r": {"$sweep": [0.25, 0.5, 0.75]}},
           "label": "sbp(r={r})"}
ACT_SBP_HALF = {"key": "znn", "kwargs": {"activation": "sign_bi_power", "r": 0.5}, "label": "sbp(r=0.5)"}
ACT_FIXED = {"key": "znn", "kwargs": {"activation": "fixed_time"}, "label": "fixed_time"}
# predefined-time: Li-Liao-Xiao-Lu 2019 F2 with k2 = k3 = 0, i.e. the Sanchez-Torres
# predefined-time system; its exact one-step map is JR20 Eq. (7) (docs/spec/consistent.md, sec. 6)
ACT_PRED = {"key": "znn", "kwargs": {"activation": "predefined_exp", "p": 0.5, "k1": 1.0, "k2": 0.0, "k3": 0.0},
            "label": "predefined_exp"}

# extension: candidate look-ahead formulas (colour order of keys.GROUPS['disc'])
FORMULA_CANDIDATES = ["euler", "taylor_zhang", "zead_5i", "zead_6i_guo2017", "zead_7i_cang2024",
                      "uhlig_3_4", "uhlig_4_6"]
FORMULA_CANDIDATES_MINI = ["euler", "taylor_zhang", "zead_5i", "uhlig_3_4"]
MAX_FORMULAS = 8          # <= 8 series per figure panel (style rule)

# error-unit (scale) arm, added in review 2026-10-18.  The non-linear activations have no length scale,
# so the dynamics depend on the unit of E: with E_s = s E the error ODE in the original unit is
# E' = -gamma Phi(s E) / s (sign: gain gamma/s; SBP r: coefficients s^(r-1)/2 and s^(1/r-1)/2).  The
# explicit-Euler period-2 band (h/4)^2 of SBP(0.5) and the dead zone of the exact map are fixed IN THE
# UNIT OF E, i.e. (h/4)^2 / s in the problem's unit.  The arm re-runs the problem with E expressed in
# units of 1/s (problem wrapper experiments.tasks.R_common:ScaledError; recorded residual, MSSRE and
# the 1e-13 threshold stay in the problem's own unit), paired with the base arm (s = 1) by seed and x0.
# s spans 5 decades: 1e-2 (coarse unit: band >> drift, explicit bounded on the whole grid), 1e1 (band
# <= drift where explicit is still bounded), 1e3 ("mm" of the robotics review: explicit SBP unbounded
# on the whole grid).
SCALE_WRAPPER_KEY = "experiments.tasks.R_common:ScaledError"
ERROR_SCALES = [1e-2, 1e1, 1e3]
DISCS_SCALE = ["euler", "implicit_euler_err", "exact_flow_euler", "taylor_zhang", "implicit_taylor_zhang",
               "implicit_taylor_zhang_naive"]

METRICS_TI = {"eps": [1e-3, 1e-6, EPS_EXACT], "frac": 0.25,
              "extra": ["initial_residual", "final_residual", "max_residual"]}
METRICS_TV = {"eps": [1e-3, 1e-6], "frac": 0.25,
              "extra": ["initial_residual", "final_residual", "max_residual"]}


def scale() -> str:
    s = os.environ.get("ZNNBENCH_SCALE", "full").strip().lower()
    if s not in ("mini", "full"):
        raise ValueError(f"ZNNBENCH_SCALE must be 'mini' or 'full', got {s!r}")
    return s


def _available():
    import znnbench
    return znnbench.available()


def _present(keys, table):
    return [k for k in keys if k in table]


def _lookahead_formulas(candidates, discs_available):
    """Candidates that are registered LookAhead formulas usable by consistent(formula=...)."""
    from znnbench.discretizations.consistent import consistent
    out = []
    for k in candidates:
        if k not in discs_available:
            continue
        try:
            consistent(k, "exact")
        except (ValueError, TypeError, KeyError):
            continue
        out.append(k)
    return out


def _extra_uhlig(discs_available, have):
    """Uhlig formulas of order >= 5 (keys 'uhlig_5_*' ... 'uhlig_9_*') registered later."""
    return sorted(k for k in discs_available if re.match(r"^uhlig_[5-9]_", k) and k not in have)


def get_configs():
    sc = scale()
    mini = sc == "mini"
    av = _available()
    discs_av = set(av["discretizations"])
    discs = _present(DISCS, discs_av)
    missing = sorted(set(DISCS) - set(discs))
    if missing:
        print(f"[E8] WARNING: discretizations not registered, dropped: {missing}", file=sys.stderr)

    acts = [ACT_SIGN, ACT_SBP, ACT_FIXED, ACT_PRED]          # every activation at both scales
    common = {"seeds": SEEDS if mini else SEEDS_FULL, "B": B, "hs": H_GRID, "mode": "fixed_h", "batch_gains": True,
              "infos": ["analytic"], "noises": [None], "wrappers": [None], "record_every": 1,
              "max_wall": 3600, "balance": True}

    # ---- (a) time-invariant problem: finite-step convergence -----------------------
    ti = {
        "name": "E8_ti",
        "description": "E8(a): time-invariant linear system, finite-step convergence and chattering",
        "problems": [{"key": "linear_system", "kwargs": {"n": 4, "kappa": 10, "freq": 0.0,
                                                         "seed": "$seed", "T": 25.0},
                      "label": "linear_system_ti"}],
        "models": acts,
        "discretizations": discs,
        "taus": [1e-2],
        "T": 25.0,                    # 2500 steps: sign at h = 0.01 needs ~|e0|/h <= ~2000 steps
        "metrics": METRICS_TI,
        "traces": {"every": 1},       # every step (2501 x 36 float32 per run): step-level traces
        "chunk_size": 9,
        **common,
    }

    # ---- (b) time-varying problems ---------------------------------------------------
    taus_tv = [1e-2, 3e-3, 1e-3] if mini else [1e-2, 3e-3, 1e-3, 3e-4]
    tv = {
        "name": "E8_tv",
        "description": "E8(b): time-varying linear system and matrix inversion; MSSRE, orders, chattering",
        "problems": [
            {"key": "linear_system", "kwargs": {"n": 4, "kappa": 10, "freq": 1.0, "seed": "$seed", "T": 6.0}},
            {"key": "matrix_inversion", "kwargs": {"n": 4, "kappa": 10, "freq": 1.0, "seed": "$seed", "T": 6.0}},
        ],
        "models": acts,
        "discretizations": discs,
        "taus": taus_tv,
        "T": 6.0,
        "metrics": METRICS_TV,
        "traces": {"points": 600},    # every step at tau = 1e-2
        "chunk_size": 12,
        **common,
    }
    if mini:   # keep the mini run small (see the module docstring)
        extra = [{"key": "znn", "kwargs": {"activation": "sign_bi_power", "r": 0.25}},
                 {"key": "znn", "kwargs": {"activation": "sign_bi_power", "r": 0.75}},
                 {"key": "znn", "kwargs": {"activation": "fixed_time"}}]
        tv["exclude"] = ([{"problem": "matrix_inversion", "tau": 1e-3}]
                         + [{"model": m, "problem": "matrix_inversion"} for m in extra]
                         + [{"model": m, "tau": 1e-3} for m in extra])

    # ---- (d) error-unit (scale) arm: E in units of 1/s, paired with (a) and (b) ----------
    discs_s = _present(DISCS_SCALE, discs_av)
    acts_s = [ACT_SIGN, ACT_SBP_HALF, ACT_PRED] if mini else acts
    scale_w = {"key": SCALE_WRAPPER_KEY, "target": "problem", "kwargs": {"scale": {"$sweep": ERROR_SCALES}}}
    sti = {**ti, "name": "E8_scale_ti",
           "description": "E8(d): time-invariant linear system with the error in units of 1/s (ScaledError)",
           "models": acts_s, "discretizations": discs_s, "wrappers": [scale_w], "traces": None}
    stv = {**tv, "name": "E8_scale_tv",
           "description": "E8(d): time-varying linear system with the error in units of 1/s (ScaledError)",
           "problems": [tv["problems"][0]], "models": acts_s, "discretizations": discs_s,
           "taus": [1e-2, 1e-3] if mini else [1e-2, 3e-3, 1e-3], "wrappers": [scale_w], "traces": None}
    stv.pop("exclude", None)

    # ---- (c) extension: error memory with any registered look-ahead formula ----------
    cands = FORMULA_CANDIDATES_MINI if mini else FORMULA_CANDIDATES
    formulas = _lookahead_formulas(cands, discs_av)
    if not mini:
        formulas += _lookahead_formulas(_extra_uhlig(discs_av, formulas), discs_av)
    formulas = formulas[:MAX_FORMULAS]
    configs = [ti, tv, sti, stv]
    if formulas:
        fdiscs = []
        for f in formulas:
            fdiscs.append({"key": f, "label": f"explicit[{f}]"})
            fdiscs.append({"key": "consistent", "kwargs": {"formula": f, "error_map": "exact", "memory": True},
                           "label": f"em_exact[{f}]"})
        fx = {
            "name": "E8_formulas",
            "description": "E8(c): error-memory rate + exact map on every registered look-ahead formula",
            "problems": [{"key": "linear_system", "kwargs": {"n": 4, "kappa": 10, "freq": 1.0,
                                                             "seed": "$seed", "T": 6.0}}],
            "models": [ACT_SIGN] if mini else [ACT_SIGN, ACT_SBP_HALF],
            "discretizations": fdiscs,
            "taus": [1e-2, 3e-3, 1e-3] if mini else [3e-2, 1e-2, 3e-3, 1e-3],
            "T": 6.0,
            "metrics": METRICS_TV,
            "chunk_size": 24,
            **{**common, "hs": [0.03, 0.3], "seeds": SEEDS},   # slopes only: 3 instances at both scales
        }
        configs.append(fx)
    return configs


# --------------------------------------------------------------------------
# script entry point: build every config and a merged tasks file
# --------------------------------------------------------------------------
def main(argv=None):
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Build the E8 sweep (all configs + merged tasks_all.jsonl).")
    ap.add_argument("--scale", choices=["mini", "full"], default=None,
                    help="overrides the ZNNBENCH_SCALE environment variable (default: full)")
    ap.add_argument("--out", default=None, help="output folder (default results/E8 or results/mini/E8)")
    ap.add_argument("--chunk-size", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="only print spec counts and the cost model")
    args = ap.parse_args(argv)
    if args.scale:
        os.environ["ZNNBENCH_SCALE"] = args.scale
    sc = scale()
    out = Path(args.out) if args.out else ROOT / ("results/mini/E8" if sc == "mini" else "results/E8")

    from znnbench.runner import MEMGUARD, DEFAULT_PYTHON, build, cost_estimate, expand, load_tasks

    lines, total_specs, total_steps = [], 0, 0.0
    for cfg in get_configs():
        specs = expand(cfg)
        steps = sum((s["T"] or 10.0) / s["tau"] for s in specs)
        traj = sum(s["B"] * len(s["gain"]["values"]) for s in specs)
        total_specs += len(specs)
        total_steps += steps
        print(f"[E8:{sc}] {cfg['name']}: {len(specs)} specs, {traj} trajectories, "
              f"{steps:.3g} simulate steps (batch {B}x{len(cfg['hs'])}), "
              f"relative cost {sum(cost_estimate(s) for s in specs):.3g}")
        if args.dry_run:
            continue
        tf = build(cfg, out / cfg["name"], chunk_size=args.chunk_size)
        tasks = load_tasks(tf)
        lines += [json.dumps(t) for t in tasks]
        print(f"         {len(tasks)} tasks -> {tf}")
    print(f"[E8:{sc}] total: {total_specs} specs, {total_steps:.4g} simulate steps")
    if not args.dry_run:
        merged = out / "tasks_all.jsonl"
        merged.write_text("\n".join(lines) + "\n")
        print(f"merged {len(lines)} tasks -> {merged}")
        print(f"launch: {DEFAULT_PYTHON} {MEMGUARD} pool --tasks {merged} --workers "
              f"{3 if sc == 'mini' else 24} --max-gb {40 if sc == 'mini' else 200}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
