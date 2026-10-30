#!/usr/bin/env python
"""Build every E9 sweep into one memguard task list.

    python experiments/tasks/E9_build.py --scale mini --out results/mini/E9
    python experiments/tasks/E9_build.py --scale full --out results/E9

For each runner config of ``experiments/configs/E9.py`` (``get_configs``) it
runs ``znnbench.runner.build`` into ``<out>/<config name>/`` (config.json,
specs.jsonl, chunks/, tasks.jsonl), adds the per-step cost-profile task
(``experiments/tasks/E9_cost.py``) and writes the union of all tasks to
``<out>/tasks_all.jsonl`` -- the file to give to ``tools/memguard.py pool``.
``znnbench.runner.aggregate(<out>)`` reads exactly the current done files of
the per-config ``tasks.jsonl`` (recursive search); ``tasks_all.jsonl`` is not
named ``tasks.jsonl`` and is therefore never double counted.

The cost task's done file name embeds a hash of the method list, the scale,
whether ``znnbench.costs`` exists and the source of the cost model
(``znnbench/costs.py``, ``znnbench/wrappers.py``, ``E9_cost.py``), so a rebuild
after the zoo adds formulas or changes the FLOP model re-runs it; unchanged sweep
chunks keep their ids.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scale", choices=("mini", "full"), required=True)
    ap.add_argument("--out", required=True, help="output folder, e.g. results/mini/E9 or results/E9")
    ap.add_argument("--chunk-size", type=int, default=None)
    a = ap.parse_args(argv)
    os.environ["E9_SCALE"] = a.scale

    from znnbench.runner import (DEFAULT_PYTHON, MEMGUARD, THREAD_ENV, _atomic_write_text, build,
                                 load_configs, load_tasks, normalize_config, stable_hash)
    import E9_cost

    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfgs = load_configs(ROOT / "experiments" / "configs" / "E9.py")
    lines, summary = [], []
    for c in cfgs:
        name = normalize_config(c)["name"]
        tf = build(c, out / name, chunk_size=a.chunk_size)
        tasks = load_tasks(tf)
        lines += [json.dumps(t) for t in tasks]
        summary.append((name, len(tasks)))

    cfgmod = E9_cost.load_config_module()
    entries = E9_cost.method_entries(cfgmod)
    # the done file name also hashes the SOURCE of the cost model (znnbench/costs.py, wrappers.py, this
    # cost task): costs.py is still being verified by the zoo, and a fix there must re-run the profile
    # instead of silently reusing a stale one (review 2026-10-18)
    src = [hashlib.sha256(f.read_bytes()).hexdigest()[:16] if f.exists() else None
           for f in (ROOT / "znnbench" / "costs.py", ROOT / "znnbench" / "wrappers.py",
                     Path(E9_cost.__file__).resolve())]
    h = stable_hash([E9_cost.COST_VERSION, a.scale, entries, E9_cost.resolve_structured() is not None,
                     cfgmod.PROBLEM_ORDER, cfgmod.PROBLEMS, src], 10)
    done = out / "cost" / f"cost_profile-{h}.json"
    cost_task = {"id": f"E9_cost-{a.scale}-{h}",
                 "cmd": [DEFAULT_PYTHON, "experiments/tasks/E9_cost.py", "--out", str(done),
                         "--scale", a.scale],
                 "done": str(done), "cwd": str(ROOT), "env": {k: str(v) for k, v in THREAD_ENV.items()}}
    lines.insert(0, json.dumps(cost_task))
    all_file = out / "tasks_all.jsonl"
    _atomic_write_text(all_file, "\n".join(lines) + "\n")
    _atomic_write_text(out / "cost_task.json", json.dumps(cost_task, indent=1))
    for name, n in summary:
        print(f"{name:24s} {n:4d} tasks")
    print(f"cost profile task -> {done}")
    print(f"{len(lines)} tasks -> {all_file}")
    print(f"launch: {DEFAULT_PYTHON} {MEMGUARD} pool --tasks {all_file} --workers 24 --max-gb 200")
    return 0


if __name__ == "__main__":
    sys.exit(main())
