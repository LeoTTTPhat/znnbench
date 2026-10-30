#!/usr/bin/env python
"""E3 -- convergence-time audit of finite / fixed / predefined-time claims.

Protocol (experiments/PLAN.md, E3; full rationale, deviations and cost
estimate in experiments/E3_NOTES.md).  Problem for every part: time-varying
matrix inversion ``matrix_inversion`` (n = 4, kappa = 10, 1 rad/s, X(0)
entries ~ U[-2, 2] as in Zhang & Ge 2005), 3 problem seeds x B = 4 initial
states.

(a) continuous time -- custom task ``experiments/tasks/E3_ct_reference.py``
    (semi-analytic settling-time integrals + DOP853 rtol 1e-12 cross-check of
    the full ZNN state equation); built by this script into ``<out>/ct``.

(b) discrete time -- runner configs built from this file:

* ``E3_disc``: time-to-eps (eps = 1e-3, 1e-6) of design-formula ZNN with
  the finite/fixed/predefined-time activations vs tau in {1e-2, 1e-3, 1e-4}
  (full: + 1e-5 for the fixed-gamma Euler / RK4 runs) in both step-size modes:
  fixed gamma = 10 (continuous-time convention; compare with the CT
  reference) and fixed h in {0.1, 1} (DTZNN convention, gamma = h/tau).
  Euler (+ RK4 at fixed gamma as a non-causal convergence check; full:
  + Taylor-Zhang).  T = 4 s (e^{-gamma 0.75 T} ||E0|| ~ 3e-12 for linear ZNN).
* ``E3_tc_<Tc>`` (one config per predefined time T_c in {1, 0.3, 0.1, 0.03,
  0.01} s, horizon T = max(3 T_c, 0.3 s)):
    - constant-gain predefined/fixed-time activations (vaf, pvaf1, pvaf2,
      predefined_exp, fixed_time) with the gain chosen so that the stated
      bound equals T_c: gamma = bound(gamma = 1) / T_c;
    - strictly predefined-time models with t_c = T_c: ``sptc_nt_znn`` and
      ``znn`` + ``strict_predefined`` (base gain gamma in G_TVG = {2, 10, 50, 100}
      mini / {1, 2, 5, 10, 20, 30, 40, 50, 70, 100} full), and
      ``sptc_an_foznn`` (gamma = 2, the paper's TVQP value);
    - the same ``sptc_nt_znn`` with the 1/(t_c - t) gain saturated at
      cap = rho / (gamma tau), rho in {1, 1.9, 3} (max effective Euler step
      h_eff = rho: deadbeat / stable non-deadbeat / outside the Euler
      stability interval); the cap is a ZNN-Bench addition (models/gains.py).
  Euler (the scheme of the source papers; full: + implicit_euler_err for the
  constant-gain activations).
* ``E3_vp``: varying-parameter ZNN ``vpznn_exp`` (g(t) = 2^t + 2, unbounded)
  with linear / sign-bi-power activation, Euler, gamma = 10, T = 12 s, with and
  without the cap g <= 1/(gamma tau); predicted linear-ZNN instability onset
  t* = log_2(2/(gamma tau) - 2) (4.2, 7.6, 11.0 s for tau = 1e-2, 1e-3, 1e-4).

Unit-dependence arm (review 2026-10-18): the Euler runs of E3_disc (fixed gamma = 10,
fixed h = 0.1) and the constant-gain activations of E3_tc_<Tc> are repeated with the
error expressed as E_s = s E, s in UNIT_SCALES = {1e-2, 1e2}
(wrapper experiments/tasks/E3_units.py:ScaledError; residual and eps stay in the
original unit), because power/sign-type activations have no length scale.

Extra metrics (experiments/tasks/E3_metrics.py): first-hit times
``t_hit_<eps>``, min residual overall / in the steady window, and the residual
around t_c (``tc_window_<tag>``).

Zoo keys are resolved at build time: an activation / model / discretization
missing from ``znnbench.available()`` (or ``activations.available()``) is
dropped with a warning, and the optional keys listed in ``OPTIONAL_*`` are
added when present, so a rebuild picks up components added later.

Scale: environment variable ``E3_SCALE`` (or ``ZNNBENCH_SCALE``) = ``mini``
or ``full`` (default ``full``) when the runner loads this file; ``--scale``
when run as a script.

Build everything (runner configs + CT tasks + one merged pool file)::

    python experiments/configs/E3.py --scale mini --out results/mini/E3
    python tools/memguard.py pool --tasks results/mini/E3/pool_tasks.jsonl --workers 3 --max-gb 40
    python experiments/analyze_E3.py --results results/mini/E3 --figdir figures/mini --tabdir results/tables/mini

Runner-only build (no CT tasks, no merged file):
``E3_SCALE=mini python -m znnbench.runner build experiments/configs/E3.py --out results/mini/E3``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXPERIMENT = "E3"
SEEDS = [0, 1, 2]
B = 4
PROBLEM = {"key": "matrix_inversion", "kwargs": {"n": 4, "kappa": 10, "seed": "$seed"}}
TAUS = {"mini": [1e-2, 1e-3, 1e-4], "full": [1e-2, 1e-3, 1e-4, 1e-5]}
TCS = [1.0, 0.3, 0.1, 0.03, 0.01]
EPS = [1e-2, 1e-3, 1e-6]
RHO_CAP = 1.0                      # cap = RHO_CAP / (gamma tau): max effective Euler step
# Cap levels rho (cap = rho / (gamma tau), i.e. max effective Euler step h_eff = rho).
# rho = 1 is the deadbeat step of linear Euler-ZNN (a best case); rho = 1.9 is a
# stable, non-deadbeat cap (|1 - h| = 0.9); rho = 3 > 2 is a cap that does NOT
# bound h_eff inside the Euler stability interval (0, 2): it should not prevent
# divergence.  Review addition: with rho = 1 alone the "cap prevents divergence"
# result could be an artefact of the deadbeat step.
RHO_CAPS = {"mini": [1.0, 1.9, 3.0], "full": [1.0, 1.9, 3.0]}
# base gains at which the non-deadbeat caps are run (None = all of G_TVG).  Mini: only
# the gains at which the uncapped law fails (wall-time budget).
RHO_CAP_GAMMAS = {"mini": {1.0: None, 1.9: [50.0, 100.0], 3.0: [50.0, 100.0]},
                  "full": {1.0: None, 1.9: None, 3.0: None}}

# E3_disc: activations (design-formula ZNN, source/ZNN-Bench default parameters)
DISC_ACTS = {
    "mini": ["linear", "sign_bi_power", "fixed_time", "vaf", "pvaf1", "predefined_exp"],
    "full": ["linear", "sign_bi_power", "fixed_time", "vaf", "pvaf1", "predefined_exp",
             "pvaf2", "sign_power", "sfaf", "wsbp", "saturated_sign_power", "sign"],
}
DISC_DISCS = {"mini": ["euler", "rk4"], "full": ["euler", "taylor_zhang", "rk4"]}
DISC_GAMMA = 10.0
DISC_HS = [0.1, 1.0]
DISC_T = {"mini": 4.0, "full": 4.0}

# E3_tc: constant-gain (fixed/predefined-time) activations, gain scaled so bound = T_c
TC_ACTS = {"mini": ["vaf", "pvaf1", "pvaf2", "predefined_exp", "fixed_time"],
           "full": ["vaf", "pvaf1", "pvaf2", "predefined_exp", "fixed_time", "sign_bi_power"]}
G_TVG = {"mini": [2.0, 10.0, 50.0, 100.0], "full": [1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 70.0, 100.0]}
SPTC_AN_GAMMAS = {"mini": [2.0], "full": [2.0, 10.0]}
TC_DISCS = {"mini": ["euler"], "full": ["euler"]}
OPTIONAL_TC_DISCS = {"mini": [], "full": ["implicit_euler_err"]}   # constant-gain activations only

EXTRA_BASE = ["initial_residual", "final_residual", "max_residual",
              "experiments.tasks.E3_metrics:first_hit"]

# Unit-dependence arm (review 2026-10-18, cross-cutting checklist item 1): power- and
# sign-type activations have no length scale, so their discrete behaviour depends on
# the unit of E.  Paired runs with the error expressed as E_s = s E (wrapper
# experiments/tasks/E3_units.py; the recorded residual stays in the original unit, so
# eps and every metric are unchanged).  s = 1e-2: E expressed in a 100x coarser unit
# (internal |E| ~ 0.2 at the natural X(0)); s = 1e2: 100x finer (internal |E| ~ 2e3).
# Euler only (the audited papers' scheme); RK4, fixed h = 1 and the time-varying-gain
# models (linear in E before t_c) are not repeated.
UNIT_WRAPPER = "experiments.tasks.E3_units:ScaledError"
UNIT_SCALES = {"mini": [1e-2, 1e2], "full": [1e-2, 1e2]}
UNIT_TAUS = {"mini": [1e-2, 1e-3], "full": [1e-2, 1e-3, 1e-4]}     # mini: wall-time budget


def unit_wrappers(scale):
    """Wrapper alternatives: none (natural unit) + one ScaledError per scale s."""
    return [[]] + [{"key": UNIT_WRAPPER, "target": "problem", "kwargs": {"scale": float(s)}}
                   for s in UNIT_SCALES[scale]]

# E3_vp: varying-parameter ZNN (PLAN "varying-parameter" family) -- unbounded
# exponential gain g(t) = p^t + p under Euler, with and without the cap
# g <= rho/(gamma tau).  Linear-activation prediction: the Euler error
# recursion becomes unstable when h_eff(t) = gamma tau g(t) > 2, i.e. at
# t* = log_p(2/(gamma tau) - p).
VP_MODEL = "vpznn_exp"
VP_P = 2.0
VP_ACTS = {"mini": ["linear", "sign_bi_power"], "full": ["linear", "sign_bi_power", "fixed_time", "vaf"]}
VP_GAMMA = 10.0
VP_T = {"mini": 12.0, "full": 12.0}
VP_RHOS = [1.0]


def scale_from_env(default="full"):
    s = (os.environ.get("E3_SCALE") or os.environ.get("ZNNBENCH_SCALE") or default).strip().lower()
    if s not in ("mini", "full"):
        raise ValueError(f"E3 scale must be 'mini' or 'full', got {s!r}")
    return s


def _avail():
    import znnbench
    from znnbench import activations as acts
    av = znnbench.available()
    return set(av["models"]), set(av["discretizations"]), set(acts.available())


def _present(keys, have, what):
    out = [k for k in keys if k in have]
    miss = [k for k in keys if k not in have]
    if miss:
        warnings.warn(f"E3: {what} not registered, dropped: {miss}")
    return out


def _g(x):
    """Canonical float (12 significant digits) so gains compare exactly."""
    return float(f"{float(x):.12g}")


def tc_tag(t_c) -> str:
    return f"{float(t_c):g}".replace(".", "p")


def bound_at_unit_gain(act_key, **kw):
    """Stated (e0-independent) convergence-time bound at gamma = 1."""
    from znnbench import activations as acts
    a = acts.get(act_key, **kw)
    return float(a.uniform_bound(1.0))


def cap_label(base, rho):
    """Label of a capped model: rho = 1 keeps the historical '<base>+cap'."""
    return f"{base}+cap" if float(rho) == 1.0 else f"{base}+cap{float(rho):g}"


def _znn(act, label=None, **kw):
    return {"key": "znn", "kwargs": {"activation": act, **kw}, "label": label or act}


def _metrics(extra):
    return {"eps": EPS, "frac": 0.25, "extra": EXTRA_BASE + list(extra)}


# ---------------------------------------------------------------------------
# E3_disc
# ---------------------------------------------------------------------------
def disc_config(scale):
    models_have, discs_have, acts_have = _avail()
    acts_ = _present(DISC_ACTS[scale], acts_have, "activations")
    discs = _present(DISC_DISCS[scale], discs_have, "discretizations")
    exclude = [{"disc": "rk4", "gain_mode": "fixed_h"}]
    if "taylor_zhang" in discs:  # TZ is unstable for h > 2/3 even for linear ZNN: h = 1 only wastes runs
        exclude.append({"disc": "taylor_zhang", "gain_mode": "fixed_h", "h": 1.0})
    if 1e-5 in TAUS[scale]:      # tau = 1e-5 only for the fixed-gamma Euler / RK4 CT-convergence runs
        exclude += [{"tau": 1e-5, "gain_mode": "fixed_h"},
                    {"tau": 1e-5, "disc": [d for d in discs if d not in ("euler", "rk4")] or "none"}]
    # unit arm: Euler only, tau <= 1e-2 grid of the protocol, fixed gamma = 10 and fixed h = 0.1
    exclude += [{"wrappers": UNIT_WRAPPER, "disc": [d for d in discs if d != "euler"] or "none"},
                {"wrappers": UNIT_WRAPPER, "gain_mode": "fixed_h", "h": 1.0},
                {"wrappers": UNIT_WRAPPER, "tau": [t for t in TAUS[scale] if t not in UNIT_TAUS[scale]]}]
    return {
        "name": "E3_disc",
        "description": {"experiment": EXPERIMENT, "part": "disc", "scale": scale},
        "problems": [PROBLEM],
        "models": [_znn(a) for a in acts_],
        "discretizations": discs,
        "taus": TAUS[scale],
        "hs": DISC_HS,
        "gammas": [DISC_GAMMA],
        "batch_gains": True,
        "exclude": exclude,
        "wrappers": unit_wrappers(scale),
        "seeds": SEEDS,
        "B": B,
        "T": DISC_T[scale],
        "record_every": 1,
        "metrics": _metrics([]),
        "max_wall": 3600,
        "chunk_size": 6 if scale == "mini" else 8,
        "balance": True,
    }


# ---------------------------------------------------------------------------
# E3_tc_<Tc>
# ---------------------------------------------------------------------------
def tc_config(t_c, scale):
    models_have, discs_have, acts_have = _avail()
    taus = [t for t in TAUS[scale] if t >= 1e-4]          # 1e-5 is not needed for the T_c/tau sweep
    discs = _present(TC_DISCS[scale], discs_have, "discretizations")
    opt_discs = [d for d in OPTIONAL_TC_DISCS[scale] if d in discs_have]
    entries = []        # (component, set of allowed (gamma, tau) pairs)
    gt = [_g(g) for g in G_TVG[scale]]
    # constant-gain predefined / fixed-time activations: bound(gamma) = T_c
    if "znn" in models_have:
        for a in _present(TC_ACTS[scale], acts_have, "activations"):
            g = _g(bound_at_unit_gain(a) / t_c)
            entries.append((_znn(a), {(g, tau) for tau in taus}))
        if "strict_predefined" in acts_have:
            entries.append((_znn("strict_predefined", t_c=t_c), {(g, tau) for g in gt for tau in taus}))
    if "sptc_nt_znn" in models_have:
        entries.append(({"key": "sptc_nt_znn", "kwargs": {"t_c": t_c}, "label": "sptc_nt_znn"},
                        {(g, tau) for g in gt for tau in taus}))
        for rho in RHO_CAPS[scale]:
            caps = {}
            rg = RHO_CAP_GAMMAS[scale].get(rho)
            for g in (gt if rg is None else [_g(x) for x in rg]):
                for tau in taus:
                    caps.setdefault(_g(rho / (g * tau)), set()).add((g, tau))
            for c, pairs in sorted(caps.items()):
                entries.append(({"key": "sptc_nt_znn", "kwargs": {"t_c": t_c, "cap": c},
                                 "label": cap_label("sptc_nt_znn", rho)}, pairs))
    if "sptc_an_foznn" in models_have:
        ga = [_g(g) for g in SPTC_AN_GAMMAS[scale]]
        entries.append(({"key": "sptc_an_foznn", "kwargs": {"t_c": t_c}, "label": "sptc_an_foznn"},
                        {(g, tau) for g in ga for tau in taus}))
        if scale == "full":
            caps = {}
            for g in ga:
                for tau in taus:
                    caps.setdefault(_g(RHO_CAP / (g * tau)), set()).add((g, tau))
            for c, pairs in sorted(caps.items()):
                entries.append(({"key": "sptc_an_foznn", "kwargs": {"t_c": t_c, "cap": c},
                                 "label": "sptc_an_foznn+cap"}, pairs))
    gammas = sorted({g for _, pairs in entries for g, _ in pairs})
    exclude = []
    for comp, pairs in entries:
        pat = {"key": comp["key"], "label": comp["label"], "kwargs": comp["kwargs"]}
        for tau in taus:
            bad = [g for g in gammas if (g, tau) not in pairs]
            if bad:
                exclude.append({"model": pat, "tau": tau, "gamma": bad})
    if opt_discs:   # implicit error step only for the constant-gain activations (time-varying models skip)
        exclude.append({"disc": opt_discs, "model": [c["label"] for c, _ in entries
                                                     if c["label"] not in TC_ACTS[scale]]})
    # unit arm (E3_units.ScaledError): constant-gain activations, Euler only.  The
    # 1/(t_c - t) models are linear in E before t_c (unit-free); not repeated.
    tv_labels = sorted({c["label"] for c, _ in entries if c["label"] not in TC_ACTS[scale]})
    if tv_labels:
        exclude.append({"wrappers": UNIT_WRAPPER, "model": tv_labels})
    if opt_discs:
        exclude.append({"wrappers": UNIT_WRAPPER, "disc": opt_discs})
    bad_t = [t for t in taus if t not in UNIT_TAUS[scale]]
    if bad_t:
        exclude.append({"wrappers": UNIT_WRAPPER, "tau": bad_t})
    return {
        "name": f"E3_tc_{tc_tag(t_c)}",
        "description": {"experiment": EXPERIMENT, "part": "tc", "scale": scale, "t_c": t_c,
                        "rho_caps": RHO_CAPS[scale]},
        "problems": [PROBLEM],
        "models": [c for c, _ in entries],
        "discretizations": discs + opt_discs,
        "taus": taus,
        "gammas": gammas,
        "mode": "fixed_gamma",
        "batch_gains": True,
        "exclude": exclude,
        "wrappers": unit_wrappers(scale),
        "seeds": SEEDS,
        "B": B,
        "T": max(3.0 * t_c, 0.3),
        "record_every": 1,
        "metrics": _metrics([f"experiments.tasks.E3_metrics:tc_window_{tc_tag(t_c)}"]),
        "traces": {"points": 1200},
        "max_wall": 3600,
        "chunk_size": 12 if scale == "mini" else 16,
        "balance": True,
    }


# ---------------------------------------------------------------------------
# E3_vp: varying-parameter (unbounded exponential) gain with / without cap
# ---------------------------------------------------------------------------
def vp_t_star(gamma, tau, p=VP_P):
    """Predicted Euler instability onset of linear VP-ZNN: gamma tau (p^t + p) = 2."""
    import math
    arg = 2.0 / (gamma * tau) - p
    return math.log(arg) / math.log(p) if arg > 1.0 else 0.0


VP_TAUS = {"mini": [1e-2, 1e-3], "full": [1e-2, 1e-3, 1e-4]}   # mini: wall-time budget (T = 12 s)


def vp_config(scale):
    models_have, discs_have, acts_have = _avail()
    taus = list(VP_TAUS[scale])
    if VP_MODEL not in models_have or "euler" not in discs_have:
        return None
    acts_ = _present(VP_ACTS[scale], acts_have, "activations")
    entries = []
    for a in acts_:
        entries.append(({"key": VP_MODEL, "kwargs": {"activation": a, "p": VP_P}, "label": f"vp:{a}"}, set(taus)))
        for rho in VP_RHOS:
            for tau in taus:
                c = _g(rho / (VP_GAMMA * tau))
                entries.append(({"key": VP_MODEL, "kwargs": {"activation": a, "p": VP_P, "cap": c},
                                 "label": cap_label(f"vp:{a}", rho)}, {tau}))
    exclude = []
    for comp, ok_taus in entries:
        bad = [t for t in taus if t not in ok_taus]
        if bad:
            exclude.append({"model": {"key": comp["key"], "label": comp["label"], "kwargs": comp["kwargs"]},
                            "tau": bad})
    return {
        "name": "E3_vp",
        "description": {"experiment": EXPERIMENT, "part": "vp", "scale": scale, "p": VP_P,
                        "rho_caps": VP_RHOS,
                        "t_star_pred": {f"{t:g}": vp_t_star(VP_GAMMA, t) for t in taus}},
        "problems": [PROBLEM],
        "models": [c for c, _ in entries],
        "discretizations": ["euler"],
        "taus": taus,
        "gammas": [VP_GAMMA],
        "mode": "fixed_gamma",
        "exclude": exclude,
        "seeds": SEEDS,
        "B": B,
        "T": VP_T[scale],
        "record_every": 1,
        "metrics": _metrics([]),
        "max_wall": 3600,
        "chunk_size": 4 if scale == "mini" else 6,
        "balance": True,
    }


def get_configs(scale=None):
    scale = scale or scale_from_env()
    cfgs = [disc_config(scale)] + [tc_config(tc, scale) for tc in TCS]
    vp = vp_config(scale)
    return cfgs + ([vp] if vp is not None else [])


# ---------------------------------------------------------------------------
# script: build runner configs + CT tasks + merged pool file
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scale", choices=["mini", "full"], default=None,
                    help="grid (default: $E3_SCALE / $ZNNBENCH_SCALE / full)")
    ap.add_argument("--out", default=None, help="output folder (default results/E3 or results/mini/E3)")
    ap.add_argument("--chunk-size", type=int, default=None)
    args = ap.parse_args(argv)
    scale = args.scale or scale_from_env()
    out = Path(args.out or (ROOT / ("results/mini/E3" if scale == "mini" else "results/E3")))
    from znnbench.runner import DEFAULT_PYTHON, MEMGUARD, build, load_tasks
    sys.path.insert(0, str(ROOT / "experiments" / "tasks"))
    import E3_ct_reference as ct  # noqa: E402

    lines, summary = [], {}
    for cfg in get_configs(scale):
        tf = build(cfg, out / cfg["name"], chunk_size=args.chunk_size)
        tasks = load_tasks(tf)
        nspec = sum(1 for _ in open(out / cfg["name"] / "specs.jsonl"))
        summary[cfg["name"]] = {"tasks": len(tasks), "specs": nspec}
        lines += [json.dumps(t) for t in tasks]
        print(f"{cfg['name']}: {nspec} specs, {len(tasks)} tasks -> {tf}")
    tf = ct.build(scale, out / "ct")
    ct_tasks = load_tasks(tf)
    summary["ct"] = {"tasks": len(ct_tasks)}
    lines += [json.dumps(t) for t in ct_tasks]
    print(f"ct: {len(ct_tasks)} tasks -> {tf}")
    pool = out / "pool_tasks.jsonl"
    pool.write_text("\n".join(lines) + "\n")
    (out / "build_summary.json").write_text(json.dumps({"scale": scale, **summary}, indent=1))
    workers = 3 if scale == "mini" else 24
    gb = 40 if scale == "mini" else 200
    print(f"{len(lines)} tasks -> {pool}")
    print(f"launch: {DEFAULT_PYTHON} {MEMGUARD} pool --tasks {pool} --workers {workers} --max-gb {gb}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
