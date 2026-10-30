#!/usr/bin/env python
"""R_matched -- matched-effort arm of the robotics case study (added 2026-10-26).

Question (external review of 26 Oct 2026, item 5): implicit sign-bi-power ZNN
(``znn_sbp_ie``) is 10-100x more accurate than linear ZNN at the nominal gain
gamma = 10 but commands 3-4x the peak joint speed.  Is it still more accurate
when both controllers use the same control effort?

Protocol (details: experiments/tasks/R_matched_common.py and the section
"Matched-effort arm (2026-10-26)" of experiments/R_NOTES.md):

* cells: robots {PUMA 560, Panda} x paths {circle, Lissajous} x control rates
  {100, 333, 1000 Hz} x error units {m, mm}, noise-free, the 12 R_off off-path
  starts (seeds 0-2 x B = 4, theta0 + U[-0.05, 0.05] rad), T = 10 s;
* effort = median over the 12 trajectories of the peak commanded joint speed
  max_{k,i} |theta_dot_i| (``qd_peak``, as in R_off); secondary: the RMS joint
  speed over the run (``qd_rms``); reported only: the peak over the last 75 %
  of the run (``qd_peak_late``, a steady-state measure);
* direction A (main): raise the gain of linear ZNN and of IEZNN (lambda =
  gamma^2/4) from 10 until the median effort equals that of implicit SBP at
  gamma = 10 in the same cell; direction B (check): lower the gain of implicit
  SBP from 10 until its median effort equals that of linear ZNN at gamma = 10;
* search on log gamma: a 16-gain grid, two 8-gain refinements per measure and up
  to 3 secant steps (all gains of a stage batched in one simulate call); a match
  is accepted at |effort / target - 1| <= 2 % (achieved: see SUMMARY.md);
* comparison: MSSRE (max ||E|| over 7.5-10 s) of the two controllers, ratio of
  medians SBP / comparator with a paired bootstrap interval (pairs = same start;
  descriptive, the paths are deterministic); joint-speed limit check.

One memguard task per robot x path x rate x unit (24 tasks), each running
``experiments/tasks/R_matched_search.py``.

Build + launch + analyze (from znn-bench/):
  python experiments/configs/R_matched.py --out results/R_matched
  python tools/memguard.py pool --tasks results/R_matched/tasks.jsonl --workers 8 --max-gb 60
  python experiments/analyze_R_matched.py --results results/R_matched --figdir figures
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

from experiments.tasks import R_common as RC  # noqa: E402
from experiments.tasks import R_matched_common as MC  # noqa: E402

TASK_SCRIPT = "experiments/tasks/R_matched_search.py"


def protocol():
    """The settings that determine the numbers (hashed into the task ids)."""
    return {"version": MC.VERSION, "robots": list(RC.ROBOTS), "paths": list(RC.PATHS), "taus": list(RC.TAUS),
            "units": sorted(MC.UNITS), "entries": list(MC.ENTRIES), "seeds": MC.SEEDS, "B": MC.B_PER_SEED,
            "x0_spread": RC.X0_SPREAD, "T": RC.T_HORIZON, "gamma_ref": MC.GAMMA_REF,
            "measures": list(MC.MEASURES), "late_frac": MC.LATE_FRAC, "early_t": MC.EARLY_T,
            "n_grid1": MC.N_GRID1, "n_grid2": MC.N_GRID2, "n_refine": MC.N_REFINE, "secant_max": MC.SECANT_MAX,
            "match_tol": MC.MATCH_TOL, "accept_tol": MC.ACCEPT_TOL, "a_gmax": MC.A_GMAX, "a_hmax": MC.A_HMAX,
            "b_range": list(MC.B_RANGE), "b_gmin": MC.B_GMIN, "unit_scale": RC.UNIT_SCALE,
            "model_entries": {e: [RC.MODEL_ENTRIES[e][0], RC.MODEL_ENTRIES[e][1]] for e in MC.ENTRIES}}


def tasks(out_dir):
    """Write <out_dir>/tasks.jsonl (one memguard task per robot x path x rate x unit)."""
    from znnbench.runner import DEFAULT_PYTHON, THREAD_ENV, ZNN_ROOT
    out = Path(out_dir).resolve()
    (out / "results").mkdir(parents=True, exist_ok=True)
    proto = protocol()
    lines = []
    # longest tasks first (1 kHz), so the pool packs well
    for tau in sorted(RC.TAUS):
        for r in RC.ROBOTS:
            for p in RC.PATHS:
                for u in sorted(MC.UNITS):
                    payload = json.dumps({"proto": proto, "robot": r, "path": p, "tau": tau, "unit": u},
                                         sort_keys=True)
                    h = hashlib.sha1(payload.encode()).hexdigest()[:10]
                    tid = f"R_matched-{r.replace('robot_', '')}-{p}-{int(round(1 / tau))}Hz-{u}-{h}"
                    done = out / "results" / f"{tid}.json"
                    cmd = [DEFAULT_PYTHON, TASK_SCRIPT, "--robot", r, "--path", p, "--tau", repr(tau),
                           "--unit", u, "--out", str(done)]
                    lines.append(json.dumps({"id": tid, "cmd": cmd, "done": str(done), "cwd": str(ZNN_ROOT),
                                             "env": dict(THREAD_ENV)}))
    tf = out / "tasks.jsonl"
    tmp = tf.with_suffix(".jsonl.tmp")
    tmp.write_text("\n".join(lines) + "\n")
    os.replace(tmp, tf)
    (out / "protocol.json").write_text(json.dumps(proto, indent=1, sort_keys=True))
    return tf, len(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output folder, e.g. results/R_matched")
    args = ap.parse_args(argv)
    tf, n = tasks(args.out)
    print(f"R_matched: {n} tasks -> {tf}")
    print(f"launch: python tools/memguard.py pool --tasks {tf} --workers 8 --max-gb 60")
    return 0


if __name__ == "__main__":
    sys.exit(main())
