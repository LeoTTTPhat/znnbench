#!/usr/bin/env python
"""R -- robotics case study: ZNN-type controllers on the PUMA 560 and the Panda.

Protocol: experiments/PLAN.md (R) and experiments/R_NOTES.md.  The task is the
position-only, unconstrained, velocity-level pseudoinverse tracking problem of
docs/spec/problems_robot.md (no joint-limit or velocity bounds are enforced).

``get_configs()`` returns two sweep configs for ``znnbench.runner``:

``R_off``  off-path starts: theta(0) = theta0 + U[-0.05, 0.05] rad per joint,
           3 seeds x B = 4 initial states (12 trajectories per cell).  Transient
           + steady tracking: MSSRE, time-to-1 mm / 0.1 mm, peak joint speed
           (``peak_rate``), joint-velocity chattering (``rate_alt``), and the
           ``JointStats`` wrapper metrics (velocity-limit utilisation, joint
           acceleration, joint-limit violation).  Down-sampled ||E(t)|| traces.
``R_on``   on-path starts (theta(0) = theta0, E(0) = 0), deterministic: one
           trajectory per cell.  Gives the repetitive-motion joint drift
           ||theta(T) - theta(0)|| on the closed paths (the off-path drift is
           dominated by the initial correction and is not a repetitive-motion
           measure).
``R_units`` error-unit sensitivity arm: the R_off starts (same seeds, so the
           runs pair with R_off), noise-free, with the error expressed in
           millimetres instead of metres (problem wrapper
           ``R_common.ScaledError``, scale 1000).  Entries: ZNN linear (an
           invariance control), ZNN-SBP explicit / implicit and SPTC-NT-ZNN
           (noise bound converted to 50 mm/s).  Power-type activations are not
           unit-invariant, so this arm tests whether the explicit-vs-implicit
           SBP conclusions are a property of the scheme or of the unit choice.

Grid (R_off / R_on): robots {robot_puma560, robot_panda} x paths {circle, lissajous}
x model entries {ZNN linear / Euler, ZNN sign-bi-power / Euler, ZNN
sign-bi-power / implicit_euler_err, SPTC-NT-ZNN / Euler, IEZNN / Euler} x
tau {1e-2, 3e-3, 1e-3} (100 Hz - 1 kHz) x gamma {10, 100} (fixed-gamma mode,
batched) x noise {none, constant 0.05 m/s}.  T = 10 s.

Every registry key is filtered by ``znnbench.available()`` at build time (the
predefined-time model falls back to ``znn`` + ``pvaf1`` if ``sptc_nt_znn`` is
absent), so a later rebuild picks up whatever the zoo provides then.

Scale
-----
``ZNNBENCH_SCALE=mini|full`` (environment; ``R_SCALE`` also accepted; default
``full``), or ``--scale`` when this file is run as a script.
``mini``: R_off and R_units with seed 0 only (B = 4), R_on noise-free only.  ``full``: the
protocol above.  The custom joint-velocity trace task (experiments/tasks/
R_joint_traces.py, 4 tasks) is the same at both scales.

Build + launch + analyze
------------------------
  python experiments/configs/R.py --scale mini --out results/mini/R
  python tools/memguard.py pool --tasks results/mini/R/tasks_all.jsonl --workers 3 --max-gb 40
  python experiments/analyze_R.py --results results/mini/R \
      --figdir figures/mini --tabdir results/tables/mini
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.tasks import R_common as RC  # noqa: E402

SCALES = ("mini", "full")
WRAPPER = {"key": "experiments.tasks.R_common:JointStats", "target": "disc",
           "kwargs": {"ss_frac": 0.25, "rel_tol": 1e-6}}
SCALE_WRAPPER = {"key": "experiments.tasks.R_common:ScaledError", "target": "problem",
                 "kwargs": {"scale": RC.UNIT_SCALE}}
TRACE_SCRIPT = "experiments/tasks/R_joint_traces.py"
TRACE_VERSION = 1


def scale_from_env(default="full"):
    s = os.environ.get("ZNNBENCH_SCALE") or os.environ.get("R_SCALE") or default
    if s not in SCALES:
        raise ValueError(f"R scale must be one of {SCALES}, got {s!r}")
    return s


def _problems(x0_spread):
    import znnbench
    av = set(znnbench.available()["problems"])
    return [{"key": r, "kwargs": {"path": p, "T": RC.T_HORIZON, "x0_spread": x0_spread},
             "label": f"{r.replace('robot_', '')}_{p}"}
            for r in RC.ROBOTS if r in av for p in RC.PATHS]


def _models_discs(ent=None):
    ent = RC.available_entries() if ent is None else ent
    models, discs, seen = [], [], set()
    for k, (comp, disc, _name) in ent.items():
        if comp["label"] not in seen:
            models.append(comp)
            seen.add(comp["label"])
        if disc not in discs:
            discs.append(disc)
    # the product model x disc is pruned to the intended (model, disc) pairs
    allowed = {(c["label"], d) for c, d, _ in ent.values()}
    exclude = [{"model": lab, "disc": d} for lab in seen for d in discs if (lab, d) not in allowed]
    return models, discs, exclude


def base_config(name, x0_spread, seeds, B, noises, entries=None, wrappers=None):
    models, discs, exclude = _models_discs(entries)
    return {
        "name": name,
        "description": "R robotics case study (experiments/R_NOTES.md)",
        "problems": _problems(x0_spread),
        "models": models,
        "discretizations": discs,
        "taus": list(RC.TAUS),
        "gammas": list(RC.GAMMAS),
        "mode": "fixed_gamma",
        "batch_gains": True,
        "noises": noises,
        "wrappers": [WRAPPER] if wrappers is None else wrappers,
        "seeds": list(seeds),
        "B": B,
        "T": RC.T_HORIZON,
        "record_every": 1,
        "metrics": {"eps": [1e-3, 1e-4], "frac": 0.25,
                    "extra": ["final_residual", "max_residual", "peak_xdot"]},
        "traces": {"points": 1001},
        "max_wall": 3600,
        "exclude": exclude,
        "chunk_size": 12,
        "balance": True,
    }


def get_configs(scale=None):
    scale = scale or scale_from_env()
    noise_c = {"key": "constant", "kwargs": {"level": RC.NOISE_LEVEL}, "label": "constant"}
    seeds = [0, 1, 2] if scale == "full" else [0]
    if scale == "full":
        off = base_config("R_off", RC.X0_SPREAD, seeds=seeds, B=4, noises=[None, noise_c])
        on = base_config("R_on", 0.0, seeds=[0], B=1, noises=[None, noise_c])
    else:
        off = base_config("R_off", RC.X0_SPREAD, seeds=seeds, B=4, noises=[None, noise_c])
        on = base_config("R_on", 0.0, seeds=[0], B=1, noises=[None])
    # error-unit sensitivity arm: same off-path starts (same seeds -> same x0 draw,
    # paired with R_off), noise-free, error expressed in mm (R_common.ScaledError)
    units = base_config("R_units", RC.X0_SPREAD, seeds=seeds, B=4, noises=[None],
                        entries=RC.units_entries(), wrappers=[[SCALE_WRAPPER, WRAPPER]])
    units["description"] = ("R error-unit sensitivity arm: E in mm (x1000) instead of m "
                            "(experiments/R_NOTES.md)")
    units.pop("traces", None)                 # no per-step traces needed for this arm
    return [off, on, units]


# --------------------------------------------------------------------------
# custom task: per-step joint-velocity traces (experiments/tasks/R_joint_traces.py)
# --------------------------------------------------------------------------
def trace_tasks(out_dir):
    """Write <out_dir>/traces/tasks.jsonl (one memguard task per robot x path)."""
    import hashlib
    from znnbench.runner import DEFAULT_PYTHON, THREAD_ENV, ZNN_ROOT
    out = Path(out_dir).resolve() / "traces"
    (out / "results").mkdir(parents=True, exist_ok=True)
    ent = RC.available_entries()
    import znnbench
    probs = [r for r in RC.ROBOTS if r in znnbench.available()["problems"]]
    lines = []
    for r in probs:
        for p in RC.PATHS:
            payload = json.dumps({"v": TRACE_VERSION, "robot": r, "path": p, "entries": sorted(ent),
                                  "taus": RC.TAUS, "gammas": RC.GAMMAS, "noise": RC.NOISE_LEVEL,
                                  "x0_spread": RC.X0_SPREAD, "sptc": RC.SPTC_KW}, sort_keys=True)
            h = hashlib.sha1(payload.encode()).hexdigest()[:10]
            tid = f"R_traces-{r.replace('robot_', '')}-{p}-{h}"
            done = out / "results" / f"{tid}.npz"
            lines.append(json.dumps({"id": tid, "cmd": [DEFAULT_PYTHON, TRACE_SCRIPT, "--robot", r, "--path", p,
                                                          "--out", str(done)],
                                     "done": str(done), "cwd": str(ZNN_ROOT), "env": dict(THREAD_ENV)}))
    tf = out / "tasks.jsonl"
    tmp = tf.with_suffix(".jsonl.tmp")
    tmp.write_text("\n".join(lines) + "\n")
    os.replace(tmp, tf)
    return tf


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scale", choices=SCALES, default=None)
    ap.add_argument("--out", required=True, help="output folder, e.g. results/mini/R or results/R")
    ap.add_argument("--chunk-size", type=int, default=None)
    ap.add_argument("--no-traces", action="store_true", help="do not write the joint-trace tasks")
    args = ap.parse_args(argv)
    from znnbench.runner import build, load_tasks
    scale = args.scale or scale_from_env()
    out = Path(args.out)
    files = []
    for cfg in get_configs(scale):
        tf = build(cfg, out / cfg["name"], chunk_size=args.chunk_size)
        n_specs = sum(1 for _ in open(out / cfg["name"] / "specs.jsonl"))
        print(f"{cfg['name']} [{scale}]: {n_specs} specs, {len(load_tasks(tf))} tasks -> {tf}")
        files.append(tf)
    if not args.no_traces:
        tf = trace_tasks(out)
        print(f"R_traces: {len(load_tasks(tf))} tasks -> {tf}")
        files.append(tf)
    allf = out / "tasks_all.jsonl"
    lines = [ln for f in files for ln in Path(f).read_text().splitlines() if ln.strip()]
    tmp = allf.with_suffix(".jsonl.tmp")
    tmp.write_text("\n".join(lines) + "\n")
    os.replace(tmp, allf)
    (out / "scale.json").write_text(json.dumps({"scale": scale, "configs": [str(f) for f in files]}, indent=1))
    print(f"all: {len(lines)} tasks -> {allf}")
    w, gb = (3, 40) if scale == "mini" else (16, 200)
    print(f"launch: python tools/memguard.py pool --tasks {allf} --workers {w} --max-gb {gb}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
