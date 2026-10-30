#!/usr/bin/env python
"""E7 timing tasks: clean wall-clock and FLOPs per step of each tracking method.

    # build (writes <out>/chunks/*.json, <out>/tasks.jsonl, <out>/specs.jsonl)
    python experiments/tasks/E7_timing.py --build --scale mini --out results/mini/E7/timing
    # one task (what tools/memguard.py pool runs)
    python experiments/tasks/E7_timing.py --chunk <out>/chunks/<task>.json

Task layout (review fix): one task per (n, problem seed) holds *every* timed
method at both batch sizes.  After one untimed warm-up run per spec, the task
runs R rounds; in each round every spec is timed once, in a fresh random
order (seeded by the task).  All methods of a round are therefore measured
within a few seconds of each other under the same machine load, and the
analysis forms *paired* cost ratios per (seed, round) block.  (Before the fix
each (n, method) was its own task, so two methods could be timed minutes
apart under different concurrent load, and the bootstrap treated the pooled
repeats as independent.)

Protocol of one timing spec (n, method, seed, B):
1. Build ``matrix_inversion(n, kappa=10, seed)`` and the method's model and
   discretization; wrap the problem in ``PrefetchDerivative`` so that the
   synthetic generator of A(t) and A_dot(t) runs before ``disc.step``.
2. One untimed warm-up run, then R timed *lean* runs of ``steps`` steps
   (``record_every = steps``, no solution error, ``online_stats=False``,
   garbage collector disabled), one per round.  Per run: ``wall_dynamics / steps_run``
   (time inside ``disc.step`` only = the algorithm, primary),
   ``wall / steps_run`` (whole loop incl. data generation and recording of
   t_0 and t_N) and the process-CPU / wall ratio (1.00 = single thread).
3. A call-count profile on a *separate* problem instance: two runs of
   S1 < S2 steps through ``wrappers.CountingProblem`` + a counting model
   proxy; per-step rates = (count(S2) - count(S1)) / (S2 - S1), which removes
   one-time work (e.g. the Lipschitz power iteration of ``pc_gradient``).
   LU factorisations are counted by wrapping the factor function of the
   private instance's value-keyed factor cache.  FLOPs follow
   ``E7_common.flops_executed`` / ``flops_ideal``.

4. A machine-state probe at the start of every round (``machine_probe``: fixed
   128 x 128 dgemm and LU factor + solve), stored per round in every row
   (``probe_gemm_us_all``, ``probe_lu_us_all``).  Accelerate runs dgemm on the
   matrix unit shared by a CPU cluster; concurrent processes slow GEMM-heavy
   methods far more than LU-heavy ones while CPU/wall stays 1.00, so the
   probe, not ``cpu_over_wall``, is the contention check.  Full-scale timing
   therefore runs alone with one worker (see E7_NOTES.md).

Every task runs with one BLAS thread (task env + the variables set below
before numpy is imported) and writes its parquet atomically (temporary file
+ ``os.replace``); the parquet is the memguard ``done`` file.
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import resource  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.tasks import E7_common as C  # noqa: E402

FORMAT = "znnbench-e7-timing/3"     # /2: one task per (n, seed), interleaved rounds; /3: machine probe
SCRIPT = "experiments/tasks/E7_timing.py"
#: machine-state probe run at the start of every round (review 2026-10-18): a fixed 128 x 128
#: dgemm and LU factor + solve.  Accelerate's dgemm runs on the matrix (AMX) unit shared by a
#: CPU cluster, so concurrent processes slow GEMM-heavy methods (measured on the M3 Ultra:
#: +8 / +40 / +80 / +118 % at 2 / 4 / 8 / 12 concurrent single-thread processes; LU +0.5 / +3 /
#: +8 / +12 %) while process-CPU / wall stays 1.00, so ``cpu_over_wall`` cannot detect it.
PROBE_N, PROBE_GEMM_REPS, PROBE_LU_REPS = 128, 60, 6


# ----------------------------------------------------------------------------
# build
# ----------------------------------------------------------------------------
def _specs(scale_name: str, include_extras: bool = True):
    """Timing specs grouped into tasks: one group per (n, problem seed) with every
    timed method at every batch size (interleaved rounds, see the module doc)."""
    from znnbench.runner import stable_hash
    P = C.SCALES[scale_name]
    M = C.methods(include_extras=include_extras)
    out = []
    for n in P["ns"]:
        for seed in P["t_seeds"]:
            group = []
            for m in M:
                if not m.get("timing", True) or (m.get("n_max") and n > m["n_max"]):
                    continue
                for B in P["t_B"]:
                    s = {"label": m["label"], "core": bool(m.get("core", False)),
                         "problem": {"key": C.PROBLEM, "kwargs": {"n": n, "kappa": C.KAPPA, "freq": C.FREQ,
                                                                  "seed": seed, "T": 10.0}},
                         "model": {"key": m["model"], "kwargs": dict(m.get("model_kwargs", {}))},
                         "disc": {"key": m["disc"], "kwargs": {}},
                         "h": float(m["h"]), "tau": C.TAU, "info": m["info"], "seed": seed, "B": B,
                         "steps": P["t_steps"], "repeats": P["t_repeats"], "warmup": P["t_warmup"],
                         "profile": list(P["t_profile"]), "prefetch": True}
                    s["id"] = stable_hash(s, 16)
                    group.append(s)
            if group:
                out.append(group)               # one task per (n, seed)
    return out


def _atomic_write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.tmp-")
    with os.fdopen(fd, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def build(out_dir, scale_name=None, include_extras: bool = True, python=None) -> Path:
    from znnbench.runner import DEFAULT_PYTHON, THREAD_ENV, stable_hash
    s = C.scale(scale_name)
    out = Path(out_dir).resolve()
    python = python or DEFAULT_PYTHON
    lines, all_specs = [], []
    for i, group in enumerate(_specs(s, include_extras)):
        h = stable_hash([FORMAT, group], 10)
        n0, seed0 = group[0]["problem"]["kwargs"]["n"], group[0]["seed"]
        tid = f"E7t_{s}-n{n0}-s{seed0}-{h}"
        chunk = out / "chunks" / f"{tid}.json"
        result = out / "results" / f"{tid}.parquet"
        _atomic_write(chunk, json.dumps({"format": FORMAT, "task_id": tid, "scale": s,
                                         "result": str(result), "specs": group}, indent=1))
        lines.append(json.dumps({"id": tid, "cmd": [python, SCRIPT, "--chunk", str(chunk)],
                                 "done": str(result), "cwd": str(ROOT), "env": dict(THREAD_ENV)}))
        all_specs += group
    (out / "results").mkdir(parents=True, exist_ok=True)
    _atomic_write(out / "specs.jsonl", "".join(json.dumps(x) + "\n" for x in all_specs))
    tf = out / "tasks.jsonl"
    _atomic_write(tf, "\n".join(lines) + "\n")
    return tf


# ----------------------------------------------------------------------------
# execution
# ----------------------------------------------------------------------------
def _peak_rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2 ** 20 if sys.platform == "darwin" else r / 1024.0


def _lean(sim_fn, problem, model, disc, s, steps):
    return sim_fn(problem, model, disc, s["tau"], T=steps * s["tau"], B=s["B"], gamma=s["h"] / s["tau"],
                  seed=s["seed"], info=s["info"], record_every=steps, record_solution_error=False,
                  online_stats=False)


def _profile(s, steps):
    """Per-run call counts on a private problem instance."""
    from znnbench import make_discretization, make_model, make_problem
    from znnbench.simulate import simulate
    from znnbench.wrappers import CountingProblem
    raw = make_problem(s["problem"]["key"], **s["problem"]["kwargs"])
    lu = {"n": 0}
    fc = getattr(raw, "_lu", None)
    if fc is not None and callable(getattr(fc, "_factor", None)):
        orig = fc._factor

        def counted(*a, _orig=orig):
            lu["n"] += 1
            return _orig(*a)
        fc._factor = counted             # private instance, profile run only
    P = CountingProblem(C.prefetch_wrapper_class()(raw))
    model = C.CountingModel(make_model(s["model"]["key"], **s["model"]["kwargs"]))
    disc = make_discretization(s["disc"]["key"], **s["disc"]["kwargs"])
    with np.errstate(all="ignore"):
        sim = _lean(simulate, P, model, disc, s, steps)
    c = dict(P.calls)
    c.pop("residual", None)
    c.pop("data", None)
    c["rhs"] = model.calls
    c["lu"] = lu["n"]
    return c, int(sim["steps_run"])


def _new_row(s: dict) -> dict:
    n = int(s["problem"]["kwargs"]["n"])
    return {"spec_id": s["id"], "label": s["label"], "core": s["core"], "model_key": s["model"]["key"],
            "disc_key": s["disc"]["key"], "h": s["h"], "gamma": s["h"] / s["tau"], "tau": s["tau"],
            "info": s["info"], "n": n, "kappa": s["problem"]["kwargs"]["kappa"], "seed": s["seed"],
            "B": s["B"], "steps": s["steps"], "repeats": s["repeats"], "status": "ok", "message": None}


def _setup(s: dict, cache: dict):
    """Problem (shared within the task, wrapped in PrefetchDerivative), model, disc."""
    from znnbench import make_discretization, make_model, make_problem
    pk = json.dumps(s["problem"], sort_keys=True)
    if pk not in cache:
        cache.clear()
        cache[pk] = make_problem(s["problem"]["key"], **s["problem"]["kwargs"])
    problem = C.prefetch_wrapper_class()(cache[pk]) if s.get("prefetch", True) else cache[pk]
    model = make_model(s["model"]["key"], **s["model"]["kwargs"])
    disc = make_discretization(s["disc"]["key"], **s["disc"]["kwargs"])
    req = set(getattr(model, "requires", set()) or set())
    if not req <= set(problem.capabilities() or set()):
        return None, f"requires {sorted(req)}"
    return (problem, model, disc), None


def _timed_run(s: dict, objs) -> dict:
    """One timed lean run (GC disabled); per-step times and CPU/wall."""
    from znnbench.simulate import simulate
    problem, model, disc = objs
    gc_was = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        c0, w0 = time.process_time(), time.perf_counter()
        with np.errstate(all="ignore"):
            sim = _lean(simulate, problem, model, disc, s, int(s["steps"]))
        c1, w1 = time.process_time(), time.perf_counter()
    finally:
        if gc_was:
            gc.enable()
    k = max(int(sim["steps_run"]), 1)
    return {"dyn": sim["wall_dynamics"] / k, "tot": sim["wall"] / k,
            "cpu": (c1 - c0) / max(w1 - w0, 1e-12), "nrun": k, "ndiv": int(np.sum(sim["diverged"])),
            "t_start": w0}


def _finish_row(row: dict, s: dict, runs: list):
    """Per-spec summary of its timed runs (``*_all`` lists are aligned by round)."""
    dyn = [r["dyn"] for r in runs]
    tot = [r["tot"] for r in runs]
    row.update({"wall_dyn_per_step": float(np.median(dyn)), "wall_dyn_min": float(np.min(dyn)),
                "wall_dyn_all": json.dumps([float(v) for v in dyn]),
                "wall_per_step": float(np.median(tot)), "wall_min": float(np.min(tot)),
                "wall_all": json.dumps([float(v) for v in tot]),
                "rounds": json.dumps([int(r["round"]) for r in runs]),
                "round_pos": json.dumps([int(r["pos"]) for r in runs]),
                "cpu_over_wall": float(np.median([r["cpu"] for r in runs])),
                "steps_run": int(np.min([r["nrun"] for r in runs])),
                "n_diverged": int(np.max([r["ndiv"] for r in runs]))})


def _profile_row(row: dict, s: dict):
    n = int(s["problem"]["kwargs"]["n"])
    s1, s2 = (int(v) for v in s["profile"])
    c_a, r_a = _profile(s, s1)
    c_b, r_b = _profile(s, s2)
    rates = {k: (c_b.get(k, 0) - c_a.get(k, 0)) / max(r_b - r_a, 1) for k in c_b}
    for k, v in rates.items():
        row[f"calls_{k}"] = float(v)
    row["flops_exec"] = C.flops_executed(rates, n, s["B"], s["model"]["key"])
    row["flops_ideal"] = C.flops_ideal(s["label"], n, s["B"])
    if r_a < s1 or r_b < s2:
        row["message"] = f"profile runs diverged early ({r_a}/{s1}, {r_b}/{s2} steps)"


def machine_probe(rng_seed: int = 0) -> dict:
    """Median time [us] of a fixed 128 x 128 dgemm and of an LU factor + solve with 128
    right-hand sides (GC off).  Run at the start of every round; compared across rounds and
    tasks it exposes contention for the shared matrix unit (see ``PROBE_N``)."""
    from scipy.linalg import lu_factor, lu_solve
    rng = np.random.default_rng(rng_seed)
    A = rng.standard_normal((PROBE_N, PROBE_N)) + PROBE_N * np.eye(PROBE_N)
    X = rng.standard_normal((PROBE_N, PROBE_N))
    gc_was = gc.isenabled()
    gc.disable()
    try:
        Y = A @ X                                     # warm
        tg = []
        for _ in range(PROBE_GEMM_REPS):
            t0 = time.perf_counter()
            Y = A @ X
            tg.append(time.perf_counter() - t0)
        tl = []
        for _ in range(PROBE_LU_REPS):
            t0 = time.perf_counter()
            Y = lu_solve(lu_factor(A, check_finite=False), X, check_finite=False)
            tl.append(time.perf_counter() - t0)
    finally:
        if gc_was:
            gc.enable()
    del Y
    return {"gemm_us": float(np.median(tg)) * 1e6, "lu_us": float(np.median(tl)) * 1e6}


def run_group(specs: list, task_seed: int, log=print) -> list:
    """Time every spec of one task in R interleaved rounds (random order per round)."""
    from znnbench.simulate import simulate
    cache, rows, objs, runs = {}, {}, {}, {}
    probes = []
    for s in specs:                                   # setup + untimed warm-up
        row = _new_row(s)
        rows[s["id"]] = row
        try:
            o, why = _setup(s, cache)
            if o is None:
                row.update(status="skipped", message=why)
                continue
            with np.errstate(all="ignore"):
                _lean(simulate, *o, s, max(int(s["warmup"]), 1))
            objs[s["id"]] = o
            runs[s["id"]] = []
        except Exception as exc:  # noqa: BLE001
            row.update(status="error", message=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-1500:]}")
    R = max(int(s["repeats"]) for s in specs)
    rng = np.random.default_rng(task_seed)
    ids = [s["id"] for s in specs if s["id"] in objs]
    by_id = {s["id"]: s for s in specs}
    for r in range(R):
        probes.append(machine_probe(r))
        order = [ids[i] for i in rng.permutation(len(ids))]
        for pos, sid in enumerate(order):
            if sid not in objs:
                continue
            try:
                res = _timed_run(by_id[sid], objs[sid])
                res.update(round=r, pos=pos)
                runs[sid].append(res)
            except Exception as exc:  # noqa: BLE001
                rows[sid].update(status="error",
                                 message=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-1500:]}")
                objs.pop(sid, None)
    out = []
    probe_gemm = json.dumps([p["gemm_us"] for p in probes])
    probe_lu = json.dumps([p["lu_us"] for p in probes])
    for s in specs:
        row = rows[s["id"]]
        if row["status"] == "ok" and runs.get(s["id"]):
            _finish_row(row, s, runs[s["id"]])
            try:
                _profile_row(row, s)
            except Exception as exc:  # noqa: BLE001
                row["message"] = f"profile failed: {type(exc).__name__}: {exc}"
        row["probe_gemm_us_all"] = probe_gemm          # per round (index = round)
        row["probe_lu_us_all"] = probe_lu
        row["peak_rss_mb"] = _peak_rss_mb()
        out.append(row)
        log(f"{s['label']:28s} n={row['n']:4d} B={row['B']} seed={row['seed']} {row['status']:7s} "
            f"dyn/step={row.get('wall_dyn_per_step', float('nan')) * 1e6:9.2f}us "
            f"cpu/wall={row.get('cpu_over_wall', float('nan')):.2f}")
    return out


def run_spec(s: dict, cache: dict) -> dict:
    """Single-spec convenience wrapper (R rounds of one spec, no interleaving)."""
    return run_group([s], 0, log=lambda *_: None)[0]


def stable_hash_int(text: str) -> int:
    import zlib
    return zlib.crc32(str(text).encode())


def execute_chunk(chunk_file, force: bool = False) -> Path:
    import pandas as pd
    ch = json.loads(Path(chunk_file).read_text())
    if ch.get("format") != FORMAT:
        raise ValueError(f"{chunk_file}: not an E7 timing chunk")
    result = Path(ch["result"])
    if result.exists() and not force:
        print(f"{ch['task_id']}: done already")
        return result
    env = {k: os.environ.get(k) for k in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")}
    meta = {"task_id": ch["task_id"], "scale": ch.get("scale"), "machine": platform.machine(),
            "processor": platform.processor(), "python": platform.python_version(),
            "numpy": np.__version__, "blas_env": json.dumps(env)}
    t0 = time.perf_counter()
    la0 = os.getloadavg()[0]
    task_seed = int(stable_hash_int(ch["task_id"]))
    rows = run_group(ch["specs"], task_seed,
                     log=lambda msg: print(f"[{ch['task_id']}] {msg}", flush=True))
    la = os.getloadavg()[0]
    for r in rows:
        r.update(meta)
        r["task_id"] = ch["task_id"]
        r["loadavg1"] = 0.5 * (la0 + la)          # mean of task start / end
        r["task_wall"] = time.perf_counter() - t0
    df = pd.DataFrame(rows)
    result.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=result.parent, prefix=f".{result.name}.tmp-")
    os.close(fd)
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, result)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return result


def load_results(timing_dir):
    """All current timing rows (done files listed in <timing_dir>/tasks.jsonl)."""
    import pandas as pd
    td = Path(timing_dir)
    tf = td / "tasks.jsonl"
    files = []
    if tf.exists():
        for line in tf.read_text().splitlines():
            if line.strip():
                p = Path(json.loads(line)["done"])
                if not p.exists():
                    p = td / "results" / p.name
                if p.exists():
                    files.append(p)
    else:
        files = sorted((td / "results").glob("*.parquet"))
    frames = [pd.read_parquet(f) for f in files]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--scale", choices=sorted(C.SCALES), default=None)
    ap.add_argument("--out", help="output folder for --build")
    ap.add_argument("--no-extras", action="store_true")
    ap.add_argument("--chunk")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    if args.build:
        if not args.out:
            ap.error("--build needs --out")
        tf = build(args.out, args.scale, include_extras=not args.no_extras)
        with open(tf) as f:
            print(f"{sum(1 for x in f if x.strip())} timing tasks -> {tf}")
        return 0
    if not args.chunk:
        ap.error("--chunk or --build is required")
    out = execute_chunk(args.chunk, force=args.force)
    return 0 if Path(out).exists() else 1


if __name__ == "__main__":
    sys.exit(main())
