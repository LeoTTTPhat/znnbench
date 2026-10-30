#!/usr/bin/env python
"""R custom task: per-step joint-velocity and position-error traces.

    python experiments/tasks/R_joint_traces.py --robot robot_panda --path circle --out <file.npz>

memguard-compatible: the task's ``done`` file is ``--out``, written atomically
(temporary file + os.replace) only after every run has finished; an existing
``--out`` is skipped unless ``--force``.  Tasks are written by
``experiments/configs/R.py`` (``<out>/traces/tasks.jsonl``).

For one robot x path it simulates every available model entry of the case
study (experiments/tasks/R_common.py) at every control period tau and both
gains (batched), from ONE off-path initial state: x0 index ``--x0-index`` of
seed ``--seed`` -- exactly the initial state of the corresponding R_off sweep
trajectory (``simulate.initial_state``), so the traces illustrate rows of the
sweep (the final residual is stored for a consistency check).  Noise: none,
plus the constant-noise condition for ZNN (linear) and IEZNN.

Stored arrays (float32 unless noted), key prefix ``<entry>|<noise>|<tau>``:
  __qd   (N, G, n)  joint velocity commands (theta_{k+1} - theta_k) / tau
  __res  (N+1, G)   ||E(t_k)|| (m)
  __t    (N+1,)     float64 time grid
  __gamma (G,)      gains
plus ``meta`` (JSON string).  Size: <= 6 MB per task.
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import uuid  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.tasks import R_common as RC  # noqa: E402
from znnbench import make_discretization, make_model, make_noise, make_problem  # noqa: E402
from znnbench.simulate import initial_state, simulate  # noqa: E402

NOISY_ENTRIES = ("znn_lin", "ieznn")


def run(robot, path, seed=0, x0_index=0, taus=None, gammas=None):
    taus = list(taus or RC.TAUS)
    gammas = np.asarray(gammas or RC.GAMMAS, float)
    G = len(gammas)
    prob = make_problem(robot, path=path, T=RC.T_HORIZON, x0_spread=RC.X0_SPREAD)
    ent = RC.available_entries()
    arrays, meta = {}, {"robot": robot, "path": path, "seed": seed, "x0_index": x0_index,
                        "taus": taus, "gammas": gammas.tolist(), "runs": []}
    for key, (comp, disc_key, name) in ent.items():
        for noise in (["none", "constant"] if key in NOISY_ENTRIES else ["none"]):
            for tau in taus:
                model = make_model(comp["key"], **comp["kwargs"])
                x0_all = initial_state(prob, model, seed, 4)          # the sweep's B = 4 draw
                x0 = np.tile(x0_all[x0_index:x0_index + 1], (G, 1))
                nz = make_noise("constant", level=RC.NOISE_LEVEL) if noise == "constant" else None
                disc = RC.JointStats(make_discretization(disc_key), record=True)
                t0 = time.perf_counter()
                with np.errstate(all="ignore"):
                    sim = simulate(prob, model, disc, tau, T=RC.T_HORIZON, B=G, gamma=gammas, x0=x0,
                                   seed=seed, noise=nz, traj_ids=np.full(G, x0_index))
                wall = time.perf_counter() - t0
                tag = f"{key}|{noise}|{tau:g}"
                arrays[f"{tag}__qd"] = np.stack(disc.qd_trace).astype(np.float32)
                arrays[f"{tag}__res"] = sim["residual"].astype(np.float32)
                arrays[f"{tag}__t"] = np.asarray(sim["t"], float)
                arrays[f"{tag}__gamma"] = gammas
                em = disc.extra_metrics(sim)
                meta["runs"].append({"tag": tag, "entry": key, "name": name, "noise": noise, "tau": tau,
                                     "wall": wall, "final_residual": sim["residual"][-1].tolist(),
                                     "peak_rate": np.asarray(sim["peak_rate"]).tolist(),
                                     "qd_peak": np.asarray(em["qd_peak"]).tolist()})
                print(f"{tag}: wall {wall:.1f}s  final |E| {sim['residual'][-1]}", flush=True)
    return arrays, meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", required=True)
    ap.add_argument("--path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--x0-index", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    out = Path(args.out)
    if out.exists() and not args.force:
        print(f"{out} exists; skipping")
        return 0
    arrays, meta = run(args.robot, args.path, seed=args.seed, x0_index=args.x0_index)
    arrays["meta"] = np.array(json.dumps(meta))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.stem}.tmp-{os.getpid()}-{uuid.uuid4().hex[:6]}.npz")
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, out)
    finally:
        if tmp.exists():
            tmp.unlink()
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
