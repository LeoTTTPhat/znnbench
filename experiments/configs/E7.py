#!/usr/bin/env python
"""E7 -- cost-accuracy Pareto on time-varying matrix inversion (PLAN.md, E7).

Two task sets, both memguard-compatible:

* **accuracy** (this file's ``CONFIG``, run by ``znnbench.runner``):
  ``matrix_inversion`` n in NS, kappa = 10, freq = 1 rad/s, tau = 1e-3,
  h = 0.3 (gamma = 300) for every gain-using method (protocol arm), plus a
  gain-sensitivity arm (``*_tuned``: ZNN-Euler and explicit-inverse ZNN at
  h = 1, GNN at 0.8 h_max), 3 problem seeds x B = 4 initial states, T = 6 s,
  MSSRE over the last 25 % (+ a settle diagnostic of that window).  ``timing_repeats``
  lean runs give the runner's end-to-end wall-clock per step (batch of B,
  including the synthetic data generator) as a secondary cost measure.
* **timing** (``experiments/tasks/E7_timing.py``): clean per-step cost of the
  algorithm, single trajectory (B = 1, also B = 4), single BLAS thread,
  ``simulate``'s ``wall_dynamics`` with the data generator moved outside the
  timed region; one task per (n, seed) times *all* methods in R interleaved
  rounds (random order per round), so cost ratios are paired within rounds;
  plus measured call counts -> FLOPs.

The method table (core + any inverse-free / low-complexity keys registered
when the config is *built*, + the model variants of ``E7_common.MODEL_VARIANTS``,
e.g. ``lcc_znn_linear`` = LCCZNN (12) with linear phi) is
``experiments/tasks/E7_common.py::methods``.  Model entries are
``{key, kwargs, label}``; exclusion rules match on the label.

Scale: environment variable ``E7_SCALE`` (or ``ZNN_SCALE``) = ``mini`` |
``full`` (default ``full``), or ``--scale`` when run as a script:

    python experiments/configs/E7.py --scale mini --out results/mini/E7
    # == ZNN_SCALE=mini python -m znnbench.runner build experiments/configs/E7.py \
    #        --out results/mini/E7/accuracy
    #    + python experiments/tasks/E7_timing.py --build --scale mini --out results/mini/E7/timing
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.tasks import E7_common as C  # noqa: E402


def make_config(scale_name=None, include_extras=True) -> dict:
    s = C.scale(scale_name)
    P = C.SCALES[s]
    M = C.methods(include_extras=include_extras)
    # model entries {key, kwargs, label}: a model variant with kwargs (C.MODEL_VARIANTS) shares its
    # registry key with the default entry, so exclusion rules match on the *label* (dict pattern)
    models = C.model_entries(M)
    discs = list(dict.fromkeys(m["disc"] for m in M))
    hs = sorted({float(m["h"]) for m in M})
    infos = sorted({m["info"] for m in M})
    valid = {(m["model_label"], m["disc"], float(m["h"]), m["info"]) for m in M}
    exclude = [{"model": {"key": mo["key"], "label": mo["label"]}, "disc": di, "h": h, "info": inf}
               for mo, di, h, inf in itertools.product(models, discs, hs, infos)
               if (mo["label"], di, h, inf) not in valid]
    # dense n_x x n_x schemes (generic inverse-free DTZNNs): only n with n^2 <= max_dim
    for m in M:
        if m.get("n_max"):
            for n in P["ns"]:
                if n > m["n_max"]:
                    exclude.append({"problem": {"key": C.PROBLEM, "kwargs": {"n": n}},
                                    "model": {"key": m["model"], "label": m["model_label"]},
                                    "disc": m["disc"], "h": float(m["h"])})
    return {
        "name": f"E7_{s}",
        "problems": [{"key": C.PROBLEM,
                      "kwargs": {"n": {"$sweep": P["ns"]}, "kappa": C.KAPPA, "freq": C.FREQ,
                                 "seed": "$seed", "T": P["T"]}}],
        "models": models,
        "discretizations": discs,
        "taus": [C.TAU],
        "hs": hs,
        "mode": "fixed_h",
        "infos": infos,
        "noises": [None],
        "seeds": P["seeds"],
        "B": P["B"],
        "T": P["T"],
        "record_every": 1,
        # settle_metrics: is the steady window steady? (max over its first half / second half)
        "metrics": {"eps": [1e-3, 1e-6], "frac": 0.25,
                    "extra": ["final_residual", "experiments.tasks.E7_common:settle_metrics"]},
        "max_wall": 3600,
        "timing_repeats": P["timing_repeats"],
        "exclude": exclude,
        "chunk_size": P["chunk_size"],
    }


CONFIG = make_config()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scale", choices=sorted(C.SCALES), default=None)
    ap.add_argument("--out", required=True, help="E7 result folder (accuracy/ and timing/ are created)")
    ap.add_argument("--no-extras", action="store_true", help="core methods only")
    args = ap.parse_args(argv)
    if args.scale:
        os.environ["E7_SCALE"] = args.scale
    s = C.scale(args.scale)
    from znnbench.runner import DEFAULT_PYTHON, MEMGUARD, build, expand, load_tasks
    cfg = make_config(s, include_extras=not args.no_extras)
    out = Path(args.out)
    tf = build(cfg, out / "accuracy")
    n_specs = len(expand(cfg))
    from experiments.tasks import E7_timing
    tt = E7_timing.build(out / "timing", s, include_extras=not args.no_extras)
    labels = [m["label"] for m in C.methods(include_extras=not args.no_extras)]
    print(f"[E7 {s}] methods: {labels}")
    print(f"[E7 {s}] accuracy: {n_specs} specs, {len(load_tasks(tf))} tasks -> {tf}")
    print(f"[E7 {s}] timing:   {len(load_tasks(tt))} tasks -> {tt}")
    w = 3 if s == "mini" else 24
    gb = 40 if s == "mini" else 200
    print(f"launch: {DEFAULT_PYTHON} {MEMGUARD} pool --tasks {tf} --workers {w} --max-gb {gb}")
    # timing: concurrent single-thread processes share the per-cluster AMX matrix unit that
    # Accelerate's dgemm uses (M3 Ultra: 128x128 dgemm +40 % at 4, +80 % at 8 concurrent
    # processes, LU only +3 % / +8 %, CPU/wall stays 1.00), which biases GEMM-heavy methods;
    # full-scale timing therefore runs ALONE, one worker, after the accuracy pool has finished
    wt = 3 if s == "mini" else 1
    print(f"then:   {DEFAULT_PYTHON} {MEMGUARD} pool --tasks {tt} --workers {wt} --max-gb {gb}"
          + ("" if s == "mini" else "   # alone on the machine (no other pool, CI runners idle)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
