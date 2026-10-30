#!/usr/bin/env python
"""R_matched custom task: matched-effort gain searches for one robot x path x
control rate x error unit (protocol: experiments/tasks/R_matched_common.py and
experiments/R_NOTES.md, section "Matched-effort arm (2026-10-26)").

    python experiments/tasks/R_matched_search.py --robot robot_panda --path circle \
        --tau 0.01 --unit m --out results/R_matched/results/<task id>.json

Steps (12 R_off starts per gain, all gains of a stage in one simulate call):
  1. reference runs at gamma = 10: linear ZNN and IEZNN (linear in E, hence
     unit-invariant: simulated in m) and implicit SBP with the error in --unit;
  2. direction A: raise the gains of linear ZNN and of IEZNN until their median
     effort equals that of implicit SBP at gamma = 10 (measures qd_peak, qd_rms);
  3. direction B: lower the gain of implicit SBP until its median effort equals
     that of linear ZNN at gamma = 10.

memguard-compatible: the done file is ``--out`` (the JSON search log), written
atomically AFTER the per-trajectory rows (``<out stem>.parquet``); an existing
``--out`` is skipped unless ``--force``.
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import resource  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import uuid  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.tasks import R_matched_common as MC  # noqa: E402


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def _clean(obj):
    """JSON-safe copy (inf/nan -> strings 'inf'/'nan' are avoided: use None for nan)."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        if np.isnan(v):
            return None
        if np.isinf(v):
            return "inf" if v > 0 else "-inf"
        return v
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def run_task(robot, path, tau, unit, say=print):
    rows, calls = [], []

    def ev(entry, role, gammas, stage):
        r, info = MC.evaluate(robot, path, tau, entry, unit, gammas, tags={"role": role, "stage": stage})
        rows.extend(r)
        calls.append({"role": role, "entry": entry, "stage": stage, **info})
        say(f"{role:6s} {entry:11s} {stage:8s} G={info['G']:2d} batch={info['batch']:3d} "
            f"wall {info['wall']:.1f}s")
        return r

    # 1. reference runs at gamma = 10
    ref = {e: ev(e, "ref", [MC.GAMMA_REF], "ref") for e in MC.ENTRIES}
    tgt_A = {m: MC.median_keep_inf([r[m] for r in ref["znn_sbp_ie"]]) for m in MC.MEASURES}
    tgt_B = {m: MC.median_keep_inf([r[m] for r in ref["znn_lin"]]) for m in MC.MEASURES}
    say(f"targets A (SBP at 10, {unit}): {tgt_A};  targets B (linear ZNN at 10): {tgt_B}")

    searches = {}
    # 2. direction A: raise the gain of linear ZNN and of IEZNN
    (g_rng, g_lim) = MC.a_range(tau)
    for entry, role in (("znn_lin", "A_lin"), ("ieznn", "A_ie")):
        res, _ = MC.match_gain(lambda gs, st, e=entry, ro=role: ev(e, ro, gs, st), tgt_A, "up", g_rng, g_lim,
                               log=say)
        searches[role] = {"entry": entry, "direction": "A", "range": list(g_rng), "limit": g_lim, "results": res}
    # 3. direction B: lower the gain of implicit SBP
    (g_rng, g_lim) = MC.b_range(tau)
    res, _ = MC.match_gain(lambda gs, st: ev("znn_sbp_ie", "B_sbp", gs, st), tgt_B, "down", g_rng, g_lim, log=say)
    searches["B_sbp"] = {"entry": "znn_sbp_ie", "direction": "B", "range": list(g_rng), "limit": g_lim,
                         "results": res}
    for role, s in searches.items():
        for m, r in s["results"].items():
            say(f"{role} {m}: status {r['status']}, gamma* {r['gamma_star']}, f* {r['f_star']}, "
                f"target {r['target']:.5g}, mismatch {r.get('mismatch')}")
    return rows, calls, searches, {"A": tgt_A, "B": tgt_B}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", required=True, choices=["robot_puma560", "robot_panda"])
    ap.add_argument("--path", required=True, choices=["circle", "lissajous"])
    ap.add_argument("--tau", required=True, type=float)
    ap.add_argument("--unit", required=True, choices=sorted(MC.UNITS))
    ap.add_argument("--out", required=True, help="done file (JSON log); rows go to <stem>.parquet")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    out = Path(args.out)
    if out.exists() and not args.force:
        print(f"{out} exists; skipping")
        return 0
    import pandas as pd
    t0 = time.perf_counter()
    c0 = time.process_time()
    rows, calls, searches, targets = run_task(args.robot, args.path, args.tau, args.unit)
    wall = time.perf_counter() - t0
    cpu = time.process_time() - c0
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2 if sys.platform == "darwin" else 1024)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq = out.with_suffix(".parquet")
    df = pd.DataFrame(rows)
    tmp = pq.with_name(f".{pq.stem}.tmp-{os.getpid()}-{uuid.uuid4().hex[:6]}.parquet")
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, pq)
    finally:
        if tmp.exists():
            tmp.unlink()
    log = {"version": MC.VERSION, "robot": args.robot, "path": args.path, "tau": args.tau,
           "rate": int(round(1.0 / args.tau)), "unit": args.unit, "targets": targets, "searches": searches,
           "calls": calls, "rows_file": pq.name, "n_rows": len(df), "wall_s": wall, "cpu_s": cpu,
           "peak_rss_mb": rss_mb,
           "settings": {"seeds": MC.SEEDS, "B_per_seed": MC.B_PER_SEED, "gamma_ref": MC.GAMMA_REF,
                        "measures": list(MC.MEASURES), "late_frac": MC.LATE_FRAC, "early_t": MC.EARLY_T,
                        "n_grid1": MC.N_GRID1, "n_grid2": MC.N_GRID2, "n_refine": MC.N_REFINE,
                        "secant_max": MC.SECANT_MAX, "match_tol": MC.MATCH_TOL, "accept_tol": MC.ACCEPT_TOL}}
    tmp = out.with_name(f".{out.stem}.tmp-{os.getpid()}-{uuid.uuid4().hex[:6]}.json")
    try:
        tmp.write_text(json.dumps(_clean(log), indent=1, default=_json_default))
        os.replace(tmp, out)
    finally:
        if tmp.exists():
            tmp.unlink()
    print(f"wrote {pq} ({len(df)} rows) and {out}; wall {wall:.1f}s cpu {cpu:.1f}s rss {rss_mb:.0f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
