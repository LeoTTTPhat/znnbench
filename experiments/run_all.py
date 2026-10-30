#!/usr/bin/env python3
"""Full-scale sweep orchestrator for ZNN-Bench (see experiments/RUNBOOK.md).

  run_all.py --exps E2 E3 R E8 E1 E5 E7 E6 --workers 20 --name main
  run_all.py --exps E4 E9 --workers 8 --name late
  run_all.py --e7-timing                    # E7 timing pool: 1 worker, waits for a quiet machine

For each experiment: build its task list (skipped if it exists; --rebuild forces), concatenate the task lists
in the given order into logs/sweep_<name>.jsonl, and run them through ONE memory-guarded pool
(tools/memguard.py pool: 80 % soft limit, 200 GB hard cap, kill/drain/slow-restart; tasks are resumable via
their done files, so re-running this script after an interruption continues where it stopped).  While the pool
runs, every 60 s the status of each experiment is written to logs/sweep_<name>_status.json, and the analysis of an
experiment is started as soon as all of its tasks are done (E7's analysis waits for its timing pool).
Run from znn-bench/ (task done-paths are relative to it).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT                      # tools/memguard.py and logs/ are inside the repository
_VENV_PY = PROJECT / ".venv" / "bin" / "python"
PY = str(_VENV_PY) if _VENV_PY.exists() else sys.executable
MEMGUARD = str(PROJECT / "tools" / "memguard.py")
LOGS = PROJECT / "logs"
AN = ["--figdir", "figures", "--tabdir", "results/tables"]

EXPS = {
    "E1": dict(build=["experiments/configs/E1.py", "--scale", "full", "--out", "results/E1"],
               tasks=["results/E1/pool_tasks.jsonl"],
               analyze=["experiments/analyze_E1.py", "--results", "results/E1"] + AN),
    "E2": dict(build=["experiments/tasks/E2_stability.py", "build", "--scale", "full", "--out", "results/E2"],
               tasks=["results/E2/tasks.jsonl"],
               analyze=["experiments/analyze_E2.py", "--results", "results/E2"] + AN),
    "E3": dict(build=["experiments/configs/E3.py", "--scale", "full", "--out", "results/E3"],
               tasks=["results/E3/pool_tasks.jsonl"],
               analyze=["experiments/analyze_E3.py", "--results", "results/E3"] + AN),
    "E4": dict(build=None, tasks=None, analyze=None),      # filled in from E4_NOTES.md after its review
    "E5": dict(build=["experiments/configs/E5.py", "--scale", "full", "--out", "results/E5"],
               tasks=["results/E5/tasks.jsonl"],
               analyze=["experiments/analyze_E5.py", "--results", "results/E5"] + AN),
    "E6": dict(build=["experiments/configs/E6.py", "--scale", "full", "--out", "results/E6"],
               tasks=["results/E6/tasks.jsonl"],
               analyze=["experiments/analyze_E6.py", "--results", "results/E6"] + AN),
    "E7": dict(build=["experiments/configs/E7.py", "--scale", "full", "--out", "results/full/E7"],
               tasks=["results/full/E7/accuracy/tasks.jsonl"],
               analyze=None),                          # analysed after the timing pool (--e7-timing)
    "E8": dict(build=["experiments/configs/E8.py", "--scale", "full", "--out", "results/E8"],
               tasks=["results/E8/tasks_all.jsonl"],
               analyze=["experiments/analyze_E8.py", "--results", "results/E8"] + AN),
    "E9": dict(build=["experiments/tasks/E9_build.py", "--scale", "full", "--out", "results/E9"],
               tasks=["results/E9/tasks_all.jsonl"],
               analyze=["experiments/analyze_E9.py", "--results", "results/E9"] + AN),
    "R": dict(build=["experiments/configs/R.py", "--scale", "full", "--out", "results/R"],
              tasks=["results/R/tasks_all.jsonl"],
              analyze=["experiments/analyze_R.py", "--results", "results/R"] + AN),
}
E7_TIMING = "results/full/E7/timing/tasks.jsonl"
E7_ANALYZE = ["experiments/analyze_E7.py", "--results", "results/full/E7"] + AN


def log(msg, name):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOGS / f"sweep_{name}.log", "a") as f:
        f.write(line + "\n")


def load_tasks(path):
    with open(ROOT / path) as f:
        return [json.loads(line) for line in f if line.strip()]


def n_done(tasks):
    return sum(Path(ROOT / t["done"]).exists() if not os.path.isabs(t["done"]) else Path(t["done"]).exists()
               for t in tasks)


def run(cmd, logfile):
    with open(logfile, "a") as f:
        return subprocess.run([PY] + cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode


def e7_timing(name, load_max, wait_h):
    t0 = time.time()
    while os.getloadavg()[0] > load_max and time.time() - t0 < wait_h * 3600:
        time.sleep(120)
    log(f"E7 timing pool starts (1-min load {os.getloadavg()[0]:.1f}; waited {(time.time() - t0) / 60:.0f} min)", name)
    rc = subprocess.run([PY, MEMGUARD, "pool", "--tasks", E7_TIMING, "--workers", "1", "--max-gb", "200"],
                        cwd=ROOT, stdout=open(LOGS / f"sweep_{name}_pool.out", "a"), stderr=subprocess.STDOUT).returncode
    log(f"E7 timing pool done rc={rc}; analysing E7", name)
    rc = run(E7_ANALYZE, LOGS / "analyze_E7.log")
    log(f"E7 analysis rc={rc}", name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", nargs="*", default=[])
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--name", default="main")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--e7-timing", action="store_true")
    ap.add_argument("--load-max", type=float, default=16.0, help="E7 timing waits for a 1-min load below this")
    ap.add_argument("--wait-h", type=float, default=3.0)
    a = ap.parse_args()
    LOGS.mkdir(exist_ok=True)
    if a.e7_timing:
        return e7_timing(a.name, a.load_max, a.wait_h)

    # 1. build
    for e in a.exps:
        spec = EXPS[e]
        if spec["build"] is None:
            sys.exit(f"{e}: commands not filled in yet (see its NOTES)")
        if a.rebuild or not all((ROOT / t).exists() for t in spec["tasks"]):
            log(f"building {e}", a.name)
            rc = run(spec["build"], LOGS / f"build_{e}.log")
            if rc != 0:
                sys.exit(f"build of {e} failed (rc={rc}); see logs/build_{e}.log")
    # 2. combined task list
    per, seen, combined = {}, set(), []
    for e in a.exps:
        per[e] = [t for p in EXPS[e]["tasks"] for t in load_tasks(p)]
        for t in per[e]:
            if t["id"] in seen:
                sys.exit(f"duplicate task id {t['id']} ({e})")
            seen.add(t["id"])
            combined.append(t)
    comb_path = LOGS / f"sweep_{a.name}.jsonl"
    with open(comb_path, "w") as f:
        for t in combined:
            f.write(json.dumps(t) + "\n")
    log(f"{len(combined)} tasks: " + ", ".join(f"{e}={len(v)} (done {n_done(v)})" for e, v in per.items()), a.name)
    # 3. pool
    pool = subprocess.Popen([PY, MEMGUARD, "pool", "--tasks", str(comb_path), "--workers", str(a.workers),
                             "--max-gb", "200"], cwd=ROOT,
                            stdout=open(LOGS / f"sweep_{a.name}_pool.out", "a"), stderr=subprocess.STDOUT)
    log(f"pool pid {pool.pid}, workers {a.workers}", a.name)
    analysed = set()
    while True:
        alive = pool.poll() is None
        status = {}
        for e, ts in per.items():
            d = n_done(ts)
            status[e] = {"done": d, "total": len(ts), "analysed": e in analysed}
            if d == len(ts) and e not in analysed and EXPS[e]["analyze"]:
                log(f"{e}: all {d} tasks done -> analysing", a.name)
                rc = run(EXPS[e]["analyze"], LOGS / f"analyze_{e}.log")
                analysed.add(e)
                status[e]["analysed"] = True
                log(f"{e}: analysis rc={rc}", a.name)
        status["_load"] = os.getloadavg()
        status["_time"] = time.strftime("%H:%M:%S")
        json.dump(status, open(LOGS / f"sweep_{a.name}_status.json", "w"), indent=1)
        if not alive:
            break
        time.sleep(60)
    log(f"pool exited rc={pool.returncode}", a.name)
    for e, ts in per.items():
        if e not in analysed and EXPS[e]["analyze"]:
            log(f"{e}: {n_done(ts)}/{len(ts)} done after the pool; analysing what exists", a.name)
            run(EXPS[e]["analyze"], LOGS / f"analyze_{e}.log")


if __name__ == "__main__":
    main()
