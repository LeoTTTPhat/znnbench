#!/usr/bin/env python3
"""Memory guard for the NEUCOM-SI26 project.

The project must never use more than 200 GB of RAM. Every heavy job is run
through this guard, which measures the macOS physical footprint (the number
Activity Monitor shows, including Metal/MPS allocations) of all project
processes and reacts when the cap is approached:

* soft limit (default 80% of the cap): stop launching new tasks;
* hard limit (the cap): kill the running tasks, wait for memory to drain,
  then restart slowly -- half the workers, one launch per ramp interval,
  growing back only after a run of clean completions.

Tasks are resumable: each task names a ``done`` file and is skipped once that
file exists, so a kill only loses the in-flight work of that task.

Usage
-----
  memguard.py status
  memguard.py pool  --tasks tasks.jsonl [--workers 16] [--max-gb 200]
  memguard.py watch [--max-gb 200]          # safety net for ad-hoc processes

``tasks.jsonl`` holds one JSON object per line:
  {"id": "exp1-seed0", "cmd": ["python", "run.py", "--seed", "0"],
   "done": "results/exp1-seed0.json", "cwd": "optional/dir"}
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import os
import signal
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import psutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG = PROJECT_ROOT / "logs" / "memguard.log"
GB = 1024 ** 3

# --------------------------------------------------------------------------
# memory measurement
# --------------------------------------------------------------------------
_libproc = None
if sys.platform == "darwin":
    try:
        _libproc = ctypes.CDLL(ctypes.util.find_library("proc") or "/usr/lib/libproc.dylib")
    except OSError:
        _libproc = None


class _RusageInfoV2(ctypes.Structure):
    _fields_ = [
        ("ri_uuid", ctypes.c_uint8 * 16),
        ("ri_user_time", ctypes.c_uint64),
        ("ri_system_time", ctypes.c_uint64),
        ("ri_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_interrupt_wkups", ctypes.c_uint64),
        ("ri_pageins", ctypes.c_uint64),
        ("ri_wired_size", ctypes.c_uint64),
        ("ri_resident_size", ctypes.c_uint64),
        ("ri_phys_footprint", ctypes.c_uint64),
        ("ri_proc_start_abstime", ctypes.c_uint64),
        ("ri_proc_exit_abstime", ctypes.c_uint64),
        ("ri_child_user_time", ctypes.c_uint64),
        ("ri_child_system_time", ctypes.c_uint64),
        ("ri_child_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_child_interrupt_wkups", ctypes.c_uint64),
        ("ri_child_pageins", ctypes.c_uint64),
        ("ri_child_elapsed_abstime", ctypes.c_uint64),
        ("ri_diskio_bytesread", ctypes.c_uint64),
        ("ri_diskio_byteswritten", ctypes.c_uint64),
    ]


def footprint(pid: int) -> int:
    """Physical footprint of one process in bytes (RSS as a fallback)."""
    if _libproc is not None:
        info = _RusageInfoV2()
        if _libproc.proc_pid_rusage(int(pid), 2, ctypes.byref(info)) == 0:
            return int(info.ri_phys_footprint)
    try:
        return psutil.Process(pid).memory_info().rss
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0


def tree_usage(pids) -> int:
    """Total footprint of the given processes and all their descendants."""
    seen, total = set(), 0
    for pid in pids:
        try:
            procs = [psutil.Process(pid)]
            procs += procs[0].children(recursive=True)
        except psutil.NoSuchProcess:
            continue
        for p in procs:
            if p.pid not in seen:
                seen.add(p.pid)
                total += footprint(p.pid)
    return total


def project_processes(exclude=()):
    """Processes of this user whose command line or cwd is inside the project."""
    me, root, out = os.getuid(), str(PROJECT_ROOT), []
    # Query attributes per process (not via process_iter(attrs)): on macOS the
    # C extension can raise SystemError for a process that is exiting, which
    # once killed the watchdog.
    for p in psutil.process_iter():
        try:
            if p.pid in exclude or p.uids().real != me:
                continue
            cmd = " ".join(p.cmdline() or [])
            if root in cmd or (p.cwd() or "").startswith(root):
                out.append(p)
        except (psutil.Error, OSError, SystemError):
            continue
    return out


def project_usage(exclude=()) -> int:
    return sum(footprint(p.pid) for p in project_processes(exclude))


def system_available() -> int:
    return psutil.virtual_memory().available


# --------------------------------------------------------------------------
# logging
# --------------------------------------------------------------------------
class Log:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, msg: str):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(self.path, "a") as f:
            f.write(line + "\n")


def _kill_tree(proc: subprocess.Popen, grace: float = 5.0):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        proc.wait()


# --------------------------------------------------------------------------
# guarded pool
# --------------------------------------------------------------------------
def run_pool(tasks, workers=16, max_gb=200.0, soft_frac=0.80, reserve_gb=48.0,
             poll=2.0, ramp_delay=20.0, cooldown=60.0, grow_after=8, max_kills=3,
             min_spacing=1.0, log=None):
    """Run resumable tasks under the memory cap. Returns a summary dict."""
    log = log or Log(DEFAULT_LOG)
    hard, soft, reserve = max_gb * GB, soft_frac * max_gb * GB, reserve_gb * GB
    queue = deque(t for t in tasks if not Path(t["done"]).exists())
    skipped = len(tasks) - len(queue)
    running, kills, failed, finished = {}, {}, [], 0
    cap, max_workers = workers, workers
    clean_streak, last_launch = 0, 0.0
    log(f"pool start: {len(queue)} tasks ({skipped} already done), workers={workers}, "
        f"cap={max_gb:.0f}GB soft={soft / GB:.0f}GB")

    while queue or running:
        # reap finished tasks
        for tid, (proc, task, logf) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            logf.close()
            del running[tid]
            if rc == 0 and Path(task["done"]).exists():
                finished += 1
                clean_streak += 1
            else:
                failed.append((tid, rc))
                log(f"task {tid} exited rc={rc} (no done file) -- see logs/tasks/{tid}.log")
        if clean_streak >= grow_after and cap < max_workers:
            cap += 1
            clean_streak = 0
            log(f"ramp up: workers -> {cap}")

        usage = tree_usage([p.pid for p, _, _ in running.values()])
        avail = system_available()

        # hard breach: kill everything in flight, drain, restart slowly
        if usage >= hard or avail < reserve / 3:
            log(f"HARD LIMIT: project={usage / GB:.1f}GB avail={avail / GB:.1f}GB; "
                f"killing {len(running)} tasks")
            for tid, (proc, task, logf) in list(running.items()):
                _kill_tree(proc)
                logf.close()
                kills[tid] = kills.get(tid, 0) + 1
                if kills[tid] >= max_kills:
                    failed.append((tid, "oom"))
                    log(f"task {tid} killed {kills[tid]}x for memory -- giving up on it")
                else:
                    queue.appendleft(task)
            running.clear()
            cap = max(1, cap // 2)
            clean_streak = 0
            t0 = time.time()
            while time.time() - t0 < cooldown or project_usage() > 0.5 * hard:
                time.sleep(poll)
                if time.time() - t0 > 10 * cooldown:
                    break
            log(f"restarting slowly: workers={cap}, one launch every {ramp_delay:.0f}s")
            last_launch = time.time()
            continue

        # launch new work if below the soft limit
        can_launch = (queue and len(running) < cap and usage < soft and avail > reserve
                      and time.time() - last_launch >= (min_spacing if cap == max_workers else ramp_delay))
        if can_launch:
            task = queue.popleft()
            tid = task["id"]
            tlog = PROJECT_ROOT / "logs" / "tasks" / f"{tid}.log"
            tlog.parent.mkdir(parents=True, exist_ok=True)
            logf = open(tlog, "a")
            env = dict(os.environ, MEMGUARD_TASK=tid, **task.get("env", {}))
            proc = subprocess.Popen(task["cmd"], cwd=task.get("cwd", str(PROJECT_ROOT)),
                                    stdout=logf, stderr=subprocess.STDOUT, env=env,
                                    start_new_session=True)
            running[tid] = (proc, task, logf)
            last_launch = time.time()
            continue
        time.sleep(poll)

    summary = {"finished": finished, "skipped": skipped, "failed": failed}
    log(f"pool done: {summary}")
    return summary


def load_tasks(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# --------------------------------------------------------------------------
# watch mode: safety net for processes not launched through the pool
# --------------------------------------------------------------------------
def watch(max_gb=200.0, poll=5.0, log=None):
    """Kill the largest project process whenever the project total exceeds the
    cap; jobs are resumable, so the caller simply relaunches them (slowly)."""
    log = log or Log(DEFAULT_LOG)
    me = {os.getpid()}
    log(f"watch start: cap={max_gb:.0f}GB over all processes under {PROJECT_ROOT}")
    errors = 0
    while True:
        try:
            procs = project_processes(exclude=me)
            sizes = sorted(((footprint(p.pid), p) for p in procs), key=lambda x: -x[0])
            total = sum(s for s, _ in sizes)
            while total >= max_gb * GB and sizes:
                size, p = sizes.pop(0)
                try:
                    cmd = " ".join(p.cmdline())[:160]
                except (psutil.Error, OSError, SystemError):
                    cmd = "?"
                try:
                    p.terminate()
                    try:
                        p.wait(5)
                    except psutil.TimeoutExpired:
                        p.kill()
                    log(f"WATCH: project={total / GB:.1f}GB >= cap; killed pid {p.pid} "
                        f"({size / GB:.1f}GB): {cmd}")
                except psutil.NoSuchProcess:
                    pass
                total -= size
        except Exception as exc:  # the safety net itself must never die
            errors += 1
            if errors <= 20 or errors % 1000 == 0:
                log(f"watch: transient error #{errors}: {type(exc).__name__}: {exc}")
        time.sleep(poll)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    sub.add_parser("status")
    p = sub.add_parser("pool")
    p.add_argument("--tasks", required=True)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--max-gb", type=float, default=200.0)
    p.add_argument("--soft-frac", type=float, default=0.80)
    p.add_argument("--ramp-delay", type=float, default=20.0)
    p.add_argument("--cooldown", type=float, default=60.0)
    w = sub.add_parser("watch")
    w.add_argument("--max-gb", type=float, default=200.0)
    args = ap.parse_args()

    if args.mode == "status":
        procs = project_processes(exclude={os.getpid()})
        print(f"project processes: {len(procs)}  footprint: {project_usage({os.getpid()}) / GB:.2f} GB")
        vm = psutil.virtual_memory()
        print(f"system: total {vm.total / GB:.0f} GB, available {vm.available / GB:.1f} GB")
    elif args.mode == "pool":
        res = run_pool(load_tasks(args.tasks), workers=args.workers, max_gb=args.max_gb,
                       soft_frac=args.soft_frac, ramp_delay=args.ramp_delay, cooldown=args.cooldown)
        sys.exit(1 if res["failed"] else 0)
    elif args.mode == "watch":
        watch(max_gb=args.max_gb)


if __name__ == "__main__":
    main()
