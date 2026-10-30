#!/usr/bin/env python
"""E5 -- noise robustness under discretization.

Question: do the steady-state residual floors that the noise-tolerant ZNN
literature derives in continuous time (CT) survive sampling?  E.g. IEZNN
rejects constant noise (E -> 0), DIEZNN rejects a linear ramp, HNTZNN a
matched harmonic, SPTC-AN-FOZNN any bounded noise with ||nu|| < zeta, while
linear ZNN keeps an O(nu/gamma) floor.

Protocol (experiments/PLAN.md, E5; rationale, deviations and cost estimate in
experiments/E5_NOTES.md):

* models (6): linear ZNN ``znn``, ``ieznn``, ``dieznn``, ``hntznn`` (internal
  model at omega = 2 pi, matched to the harmonic noise; paper's Li activation
  and lam = 1), ``sptc_an_foznn`` (fractional predefined-time anti-noise ZNN,
  Delta = noise_bound = 8 -> zeta = 5 Delta = 40 > sup ||nu||_2 of every bounded
  noise below) and ``znn`` + ``sign_bi_power`` (label ``znn_sbp``);
* noises (6), levels from docs/spec/noise_info.md: none; constant [8] and linear
  [8t] (Yan et al., IEEE Access 2019, also Liao et al., Mathematics 2022,
  Sec. 4.3); per-step Gaussian with std 1/sqrt(12) (same variance as the
  bounded random noise U[-0.5, 0.5] of Liao et al. 2022, Remark 3); harmonic
  sin(2 pi t + 1) (Liao et al., Front. Neurorobot. 2022, eq. (7) with the
  first term of its periodic example eq. (12)); random walk sigma = 1 (benchmark
  extension, no ZNN source);
* discretizations: ``euler``, ``taylor_zhang``;
* tau in {1e-2, 1e-3, 1e-4} (mini: {1e-2, 1e-3});
* gains: fixed-h h in {0.1, 0.3} (PLAN, DTZNN convention gamma = h/tau) AND
  fixed-gamma gamma in {2, 10} (extension: the CT convention, under which the
  discrete floor must converge to the CT prediction as tau -> 0; gamma = 2 is
  the published SPTC-AN-FOZNN gain, gamma = 10 the usual ZNN value), each mode
  batched in one simulate call (``batch_gains``; noise streams are per
  trajectory, so batched == unbatched bit for bit);
* problems: ``matrix_inversion`` n = 4 and ``linear_system`` n = 4 (kappa = 10,
  1 rad/s, T = 6 s, steady window = last 25 %), plus the control
  ``linear_system`` with freq = 0 (label ``linear_system_static``: A, b constant,
  so the discretization truncation vanishes and the floor is the pure
  discrete-time noise response);
* initial states: ``rand`` = the problems' default x0 ~ U[-2, 2] (PLAN) and
  ``near`` = x*(0) + 0.1 U[-1, 1] (wrapper
  ``experiments.tasks.E5_start:NearSolutionStart``; isolates steady floors of
  models that are only locally stable in discrete time);
* review extensions (2026-10-18, rationale in E5_NOTES "Review log"):
  ``hntznn_lin`` / ``hntznn_lin_mis`` = HNTZNN with a LINEAR activation and the
  internal model matched / 10 % mismatched (1.1 x 2 pi), harmonic + noise-free
  only (``exclude``); a paired, noise-free unit check of znn / znn_sbp / hntznn
  from x*(0) + 1e-4 U[-1, 1] (``E5_start:UnitPairStart``) with E in its own
  unit and in units of 1e-3 (``R_common:ScaledError``, scale 1e3);
* 5 problem seeds x B = 4 initial states (PLAN: 3; raised so that the
  seed-cluster bootstrap CIs are not degenerate) (mini: 1 x 4).

Keys are filtered through ``znnbench.available()`` at build time, so a missing
or renamed zoo component drops out instead of breaking the build; rebuilding
after the zoo changes picks the keys up again.

Scale: environment variable ``E5_SCALE`` (or ``ZNNBENCH_SCALE``) = ``mini`` or
``full`` (default ``full``) when the runner loads this file; ``--scale`` when it
is run as a script.

    # build
    python experiments/configs/E5.py --scale full --out results/E5
    # launch (memguard pool, resumable)
    python tools/memguard.py pool --tasks results/E5/tasks.jsonl --workers 24 --max-gb 200
    # analyse
    python experiments/analyze_E5.py --results results/E5 --figdir figures --tabdir results/tables

Equivalent runner call: ``E5_SCALE=full python -m znnbench.runner build
experiments/configs/E5.py --out results/E5 --chunk-size 24 --balance``.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXPERIMENT = "E5"
T_HORIZON = 6.0
FRAC = 0.25

# noise levels (per entry of nu, the same value in every entry: the [a]_{n x n}
# convention of the cited papers); see the module docstring for the sources
NOISE_LEVELS = {
    "constant": {"level": 8.0},
    "linear": {"slope": 8.0, "offset": 0.0},
    "gaussian": {"std": 1.0 / math.sqrt(12.0)},
    "harmonic": {"amp": 1.0, "omega": 2.0 * math.pi, "phase": 1.0},
    "random_walk": {"sigma": 1.0},
}
NOISE_ORDER = ["constant", "linear", "gaussian", "harmonic", "random_walk"]

# models: (label, registry key, kwargs)
SPTC_DELTA = 8.0          # noise bound Delta of SPTC-AN-FOZNN (zeta = 5 Delta = 40)
HARMONIC_OMEGA = NOISE_LEVELS["harmonic"]["omega"]
HARMONIC_MISMATCH = 1.1   # review extension: internal model tuned 10 % above the noise frequency
MODEL_SPECS = [
    ("znn", "znn", {"activation": "linear"}),
    ("ieznn", "ieznn", {}),
    ("dieznn", "dieznn", {}),
    ("hntznn", "hntznn", {"omegas": [HARMONIC_OMEGA], "lam": 1.0}),
    ("sptc_an_foznn", "sptc_an_foznn", {"noise_bound": SPTC_DELTA}),
    ("znn_sbp", "znn", {"activation": "sign_bi_power"}),
    # review extension (2026-10-18), harmonic arm only (see EXCLUDE): the internal-model
    # mechanism of HNTZNN without the unit-dependent Li activation, matched and mismatched
    ("hntznn_lin", "hntznn", {"omegas": [HARMONIC_OMEGA], "lam": 1.0, "activation": "linear"}),
    ("hntznn_lin_mis", "hntznn", {"omegas": [HARMONIC_MISMATCH * HARMONIC_OMEGA], "lam": 1.0,
                                  "activation": "linear"}),
]
HARMONIC_ONLY = ["hntznn_lin", "hntznn_lin_mis"]
DISCS = ["euler", "taylor_zhang"]
HS = [0.1, 0.3]
GAMMAS = [2.0, 10.0]
NEAR_RADIUS = 0.1
START_WRAPPER = "experiments.tasks.E5_start:NearSolutionStart"
# review extension (2026-10-18): paired unit check of the power-type activations
# (REVIEW_CHECKLIST item 1).  E is expressed in units of 1/UNIT_SCALE (E_s = 1e3 E,
# "metres -> millimetres"); the recorded residual stays in the original unit.  The
# wrapper reads additive noise in the scaled unit, so the arm is NOISE-FREE only.
UNIT_WRAPPER = "experiments.tasks.R_common:ScaledError"
UNIT_SCALE = 1000.0
UNIT_MODELS = ["znn", "znn_sbp", "hntznn"]      # znn = control (linear in E: identical runs)
# Both unit arms start at x*(0) + 1e-4 U[-1, 1]: with E x 1e3 the |e|^2 branch of
# SBP / Li makes the explicit step unstable from the 0.1-radius start (|E_s| ~ 1e2,
# itself a unit-dependent STABILITY effect, see E5_NOTES review log), so the pair
# compares steady floors from the same physical start inside both basins.
UNIT_RADIUS = 1e-4
UNIT_START_WRAPPER = "experiments.tasks.E5_start:UnitPairStart"

SCALES = {
    # full: 5 problem seeds (PLAN: 3) so that the seed-cluster bootstrap of the
    # headline CIs has more than 10 distinct resamples (review, item 5); cheap.
    "full": {"taus": [1e-2, 1e-3, 1e-4], "seeds": [0, 1, 2, 3, 4], "B": 4, "chunk_size": 24},
    "mini": {"taus": [1e-2, 1e-3], "seeds": [0], "B": 4, "chunk_size": 48},
}


def _scale(scale=None) -> str:
    s = scale or os.environ.get("E5_SCALE") or os.environ.get("ZNNBENCH_SCALE") or "full"
    s = str(s).lower()
    if s not in SCALES:
        raise ValueError(f"E5 scale must be one of {sorted(SCALES)}, got {s!r}")
    return s


def _available():
    """Registry keys present right now (``znnbench.available()``)."""
    try:
        import znnbench
        av = znnbench.available()
        return {k: set(v) for k, v in av.items()}
    except Exception:  # noqa: BLE001  (no zoo importable -> keep everything, runner validates)
        return None


def get_configs(scale=None):
    sc = _scale(scale)
    p = SCALES[sc]
    av = _available()

    def have(kind, key):
        return av is None or key in av.get(kind, set())

    problems = []
    for label, key, kw in [
        ("matrix_inversion", "matrix_inversion", {"n": 4, "kappa": 10.0, "freq": 1.0}),
        ("linear_system", "linear_system", {"n": 4, "kappa": 10.0, "freq": 1.0}),
        ("linear_system_static", "linear_system", {"n": 4, "kappa": 10.0, "freq": 0.0}),
    ]:
        if have("problems", key):
            problems.append({"key": key, "label": label,
                             "kwargs": {**kw, "seed": "$seed", "T": T_HORIZON}})
    models = [{"key": key, "label": label, "kwargs": dict(kw)}
              for label, key, kw in MODEL_SPECS if have("models", key)]
    discs = [d for d in DISCS if have("discretizations", d)]
    noises = [None] + [{"key": k, "kwargs": dict(NOISE_LEVELS[k])}
                       for k in NOISE_ORDER if have("noises", k)]
    near = {"key": START_WRAPPER, "target": "problem", "kwargs": {"radius": NEAR_RADIUS}}
    near_u = {"key": UNIT_START_WRAPPER, "target": "problem", "kwargs": {"radius": UNIT_RADIUS}}
    unit = {"key": UNIT_WRAPPER, "target": "problem", "kwargs": {"scale": UNIT_SCALE}}
    labels = [m["label"] for m in models]
    not_unit = [m for m in labels if m not in UNIT_MODELS]
    exclude = [
        # the linear-activation HNTZNN arms run only with the harmonic noise (+ noise-free)
        {"model": [m for m in HARMONIC_ONLY if m in labels],
         "noise": [k for k in NOISE_ORDER if k != "harmonic"]},
        # the paired unit arms (start radius 1e-4, E x 1 and E x 1e3) are noise-free
        # and limited to UNIT_MODELS; matched by the start wrapper's radius
        {"wrappers": UNIT_START_WRAPPER, "noise": list(NOISE_ORDER)},
        {"wrappers": UNIT_START_WRAPPER, "model": not_unit},
    ]
    cfg = {
        "name": EXPERIMENT if sc == "full" else f"{EXPERIMENT}_mini",
        "description": "E5 noise robustness under discretization (see experiments/E5_NOTES.md)",
        "problems": problems,
        "models": models,
        "discretizations": discs,
        "taus": list(p["taus"]),
        "hs": list(HS),
        "gammas": list(GAMMAS),
        "mode": "both",
        "batch_gains": True,
        "noises": noises,
        "infos": ["analytic"],
        "wrappers": [None, near, near_u, [near_u, unit]],
        "exclude": exclude,
        "seeds": list(p["seeds"]),
        "B": int(p["B"]),
        "T": T_HORIZON,
        "record_every": 1,
        "metrics": {"eps": [1e-3, 1e-6], "frac": FRAC,
                    "extra": ["final_residual", "initial_residual", "max_residual"]},
        "max_wall": 1800,
        "div_threshold": 1e10,
        "chunk_size": int(p["chunk_size"]),
        "balance": True,
    }
    return [cfg]


CONFIGS = get_configs()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scale", choices=sorted(SCALES), default=None)
    ap.add_argument("--out", default=None, help="output folder (default results/E5 or results/mini/E5)")
    ap.add_argument("--chunk-size", type=int, default=None)
    ap.add_argument("--dry", action="store_true", help="only print the number of specs")
    args = ap.parse_args(argv)
    sc = _scale(args.scale)
    from znnbench.runner import DEFAULT_PYTHON, MEMGUARD, build, expand, load_tasks
    cfg = get_configs(sc)[0]
    specs = expand(cfg)
    n_traj = sum(s["B"] * len(s["gain"]["values"]) for s in specs)
    steps = sum(round(s["T"] / s["tau"]) for s in specs)
    print(f"E5 [{sc}]: {len(specs)} specs, {n_traj} trajectories, {steps:.3g} simulate steps")
    if args.dry:
        return 0
    out = Path(args.out) if args.out else (ROOT / ("results/E5" if sc == "full" else "results/mini/E5"))
    tf = build(cfg, out, chunk_size=args.chunk_size)
    print(f"{len(load_tasks(tf))} tasks -> {tf}")
    print(f"launch: {DEFAULT_PYTHON} {MEMGUARD} pool --tasks {tf} --workers 24 --max-gb 200")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
