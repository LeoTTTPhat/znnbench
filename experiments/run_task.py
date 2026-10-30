#!/usr/bin/env python
"""Execute one ZNN-Bench sweep chunk (the command run by tools/memguard.py pool).

    python experiments/run_task.py --chunk <runs/X/chunks/<task>.json> [--force]

Runs every spec of the chunk (errors, timeouts, divergence and incompatible
combinations are recorded, never fatal) and writes the chunk's result parquet
atomically; the parquet is the task's memguard ``done`` file.  Exit code 0
means the parquet exists.

Convenience: ``--build CONFIG --out DIR [--chunk-size N] [--balance]`` expands
a config into chunks + tasks.jsonl (same as ``python -m znnbench.runner build``).
"""
from __future__ import annotations

import os

# one BLAS/OpenMP thread per worker (the pool provides the parallelism);
# must be set before numpy is imported
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from znnbench.runner import (DEFAULT_PYTHON, MEMGUARD, build, execute_chunk,  # noqa: E402
                             load_configs, load_tasks, normalize_config)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chunk", help="chunk JSON file written by znnbench.runner.write_tasks")
    ap.add_argument("--force", action="store_true", help="re-run even if the result parquet exists")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--build", metavar="CONFIG", help="expand a .yaml/.json/.py config into tasks")
    ap.add_argument("--out", help="output directory for --build")
    ap.add_argument("--chunk-size", type=int, default=None)
    ap.add_argument("--balance", action="store_true")
    args = ap.parse_args(argv)

    if args.build:
        if not args.out:
            ap.error("--build needs --out")
        cfgs = load_configs(args.build)
        for c in cfgs:
            out = Path(args.out) if len(cfgs) == 1 else Path(args.out) / normalize_config(c)["name"]
            tf = build(c, out, chunk_size=args.chunk_size, balance=args.balance or None)
            print(f"{len(load_tasks(tf))} tasks -> {tf}")
            print(f"launch: {DEFAULT_PYTHON} {MEMGUARD} pool --tasks {tf} --workers 16 --max-gb 200")
        return 0
    if not args.chunk:
        ap.error("--chunk is required")
    out = execute_chunk(args.chunk, force=args.force, verbose=not args.quiet)
    return 0 if Path(out).exists() else 1


if __name__ == "__main__":
    sys.exit(main())
