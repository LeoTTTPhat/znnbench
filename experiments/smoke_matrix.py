#!/usr/bin/env python
"""ZNN-Bench smoke matrix: every registered problem x every registered model x
a representative set of discretizations, one short run each.

    python experiments/smoke_matrix.py                  # writes docs/spec/SMOKE_MATRIX.md
    python experiments/smoke_matrix.py --csv results/smoke_matrix.csv
    python experiments/smoke_matrix.py --problems linear_system --models znn gnn

Each cell runs ``simulate`` on the problem's *default* (small) instance with
T = 1 s, tau = 1e-2, B = 1, gamma = 10 (h = gamma*tau = 0.1), seed 0, exact
data (info='analytic'), no noise.  The status of a cell is one of

  ok                 finite state for the whole horizon (the cell shows ||E(x(T))||)
  diverged           simulate flagged divergence (non-finite or |x| > 1e10)
  skipped(requires)  the model requires a problem capability the problem lacks
  skipped(implicit)  implicit discretization, model without implicit support
  error              an exception was raised (message recorded)
  timeout            the run exceeded --max-wall seconds

The skip rules are the ones the sweep runner applies (znnbench/runner.py
``run_spec``).  The only non-default setting is the initial state of the
locally convergent explicit-inverse models (``znn_explicit_inverse*``): they
start from the Newton-Schulz initial inverse ``newton_schulz_init(A(0))``, as
docs/spec/structured_baselines.md and CORE_CHANGE_REQUESTS.md (structured
baselines, item 5) prescribe; ``--default-x0`` turns this off.

The run is deliberately small (about 1.2k runs of 100 steps, n <= 32 states,
well under 1 GB of RAM).  It is a wiring / robustness check, not a benchmark:
"diverged" cells can be genuine properties of a method at h = 0.1 and T = 1
(see the notes written into the Markdown report).
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse  # noqa: E402
import datetime as _dt  # noqa: E402
import platform  # noqa: E402
import resource  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from znnbench import registry  # noqa: E402
from znnbench.simulate import simulate  # noqa: E402

DEFAULT_DISCS = ("euler", "taylor_zhang", "zead_5i", "uhlig_4_6", "implicit_euler_err", "rk4")
DISC_ROLE = {"euler": "Euler (2-instant)", "taylor_zhang": "Taylor-Zhang (4-instant)",
             "zead_5i": "ZeaD 5-instant", "uhlig_4_6": "Uhlig 4_5a / 4_6 (10-instant)", "uhlig_3_4": "Uhlig 3_4 (7-instant, a=1)",
             "implicit_euler_err": "implicit (resolvent) Euler", "rk4": "RK4 (non-causal reference)"}
EXPLICIT_INVERSE_MODELS = ("znn_explicit_inverse", "znn_explicit_inverse_right")


def _x0_for(model_key, problem, B, default_x0):
    if default_x0 or model_key not in EXPLICIT_INVERSE_MODELS:
        return None, ""
    from znnbench.models.structured import newton_schulz_init
    d0 = problem.data(0.0)
    if "A" not in d0:
        return None, ""
    X0 = newton_schulz_init(np.asarray(d0["A"], dtype=float))
    return np.tile(X0.ravel(), (B, 1)), "x0=Newton-Schulz"


def run_cell(pkey, mkey, dkey, T, tau, gamma, B, seed, max_wall, default_x0, info="analytic",
             noise=None, data_noise=0.0):
    """Run one (problem, model, discretization) cell; never raises."""
    row = {"problem": pkey, "model": mkey, "disc": dkey, "status": None, "res0": np.nan,
           "resT": np.nan, "diverged_at": np.nan, "wall": np.nan, "message": "", "warnings": "",
           "x0": ""}
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as wlog:
        warnings.simplefilter("always")
        try:
            problem = registry.make_problem(pkey)
            model = registry.make_model(mkey)
            disc = registry.make_discretization(dkey)
            req = set(getattr(model, "requires", set()) or [])
            caps = set(problem.capabilities() or [])
            if not req <= caps:
                row["status"] = "skipped(requires)"
                row["message"] = f"requires {sorted(req - caps)}; problem has {sorted(caps)}"
                return row
            if getattr(disc, "implicit", False) and not getattr(model, "supports_implicit", False):
                row["status"] = "skipped(implicit)"
                row["message"] = "implicit discretization, model has supports_implicit=False"
                return row
            x0, x0note = _x0_for(mkey, problem, B, default_x0)
            row["x0"] = x0note
            nz = registry.make_noise(noise) if noise else None
            with np.errstate(all="ignore"):
                sim = simulate(problem, model, disc, tau, T=T, B=B, gamma=gamma, x0=x0, seed=seed,
                               max_wall=max_wall, info=info, noise=nz, data_noise=data_noise)
            res = np.asarray(sim["residual"])
            row["res0"] = float(np.max(res[0]))
            row["resT"] = float(np.max(res[-1]))
            if bool(np.asarray(sim["diverged"]).any()):
                row["status"] = "diverged"
                row["diverged_at"] = float(np.nanmin(sim["diverged_at"]))
            else:
                row["status"] = "ok"
        except TimeoutError as exc:
            row["status"] = "timeout"
            row["message"] = str(exc)
        except Exception as exc:  # noqa: BLE001
            row["status"] = "error"
            tb = traceback.format_exc().strip().splitlines()
            where = next((ln.strip() for ln in reversed(tb) if ln.strip().startswith("File")), "")
            row["message"] = f"{type(exc).__name__}: {exc} [{where}]"
        finally:
            row["wall"] = time.perf_counter() - t0
            msgs = []
            for w in wlog:
                if issubclass(w.category, (DeprecationWarning, PendingDeprecationWarning)):
                    continue
                m = f"{w.category.__name__}: {str(w.message).splitlines()[0][:160]}"
                if m not in msgs:
                    msgs.append(m)
            row["warnings"] = " | ".join(msgs[:3])
    return row


def cell_text(r):
    s = r["status"]
    if s == "ok":
        return f"{r['resT']:.1e}"
    if s == "diverged":
        return f"div@{r['diverged_at']:.2f}"
    return {"skipped(requires)": "s:req", "skipped(implicit)": "s:imp", "error": "**ERR**",
            "timeout": "**T/O**"}.get(s, s)


def _peak_rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 2**10


def write_markdown(rows, path, args, problems, models, discs, total_wall):
    from collections import Counter
    cnt = Counter(r["status"] for r in rows)
    by = {(r["problem"], r["model"], r["disc"]): r for r in rows}
    L = []
    L.append("# ZNN-Bench smoke matrix\n")
    L.append(f"Generated by `experiments/smoke_matrix.py` on {_dt.date.today().isoformat()} "
             f"(Python {platform.python_version()}, numpy {np.__version__}, "
             f"`ZNNBENCH_STRICT={os.environ.get('ZNNBENCH_STRICT', '0')}`).\n")
    L.append("**Setting.** Default (small) instance of every registered problem; every registered model "
             "with its default parameters; "
             f"T = {args.T:g} s, tau = {args.tau:g}, B = {args.B}, gamma = {args.gamma:g} "
             f"(h = gamma*tau = {args.gamma * args.tau:g}), seed {args.seed}, `info='{args.info}'`, "
             f"noise {args.noise or 'none'}, data_noise {args.data_noise:g}. "
             "Skip rules are those of the sweep runner. "
             + ("The explicit-inverse models start from the Newton-Schulz initial inverse "
                "(CORE_CHANGE_REQUESTS.md, structured_baselines item 5); every other cell uses the "
                "problem's own random `x0`." if not args.default_x0 else
                "Every cell uses the problem's own random `x0`.") + "\n")
    L.append(f"**Totals.** {len(rows)} cells ({len(problems)} problems x {len(models)} models x "
             f"{len(discs)} discretizations) in {total_wall:.1f} s, peak RSS {_peak_rss_mb():.0f} MB: "
             + ", ".join(f"{k} {cnt.get(k, 0)}" for k in
                         ("ok", "diverged", "skipped(requires)", "skipped(implicit)", "error", "timeout"))
             + ".\n")
    L.append("**Cell legend.** A number is the residual ‖E(x(T), d(T))‖₂ at T for a finite run (status "
             "ok); `div@t` = diverged at time t (status diverged); `s:req` = skipped(requires); "
             "`s:imp` = skipped(implicit); `**ERR**` = error; `**T/O**` = timeout.\n")
    L.append("Discretizations: " + "; ".join(f"`{d}` = {DISC_ROLE.get(d, d)}" for d in discs) + ".\n")

    # ---- per-model summary ---------------------------------------------------
    L.append("## Summary per model\n")
    L.append("| model | ok | diverged | skipped | error/timeout |")
    L.append("|---|---:|---:|---:|---:|")
    for m in models:
        rs = [r for r in rows if r["model"] == m]
        c = Counter(r["status"] for r in rs)
        L.append(f"| `{m}` | {c.get('ok', 0)} | {c.get('diverged', 0)} | "
                 f"{c.get('skipped(requires)', 0) + c.get('skipped(implicit)', 0)} | "
                 f"{c.get('error', 0) + c.get('timeout', 0)} |")
    L.append("")

    # ---- full matrix ---------------------------------------------------------
    L.append("## Full matrix\n")
    for p in problems:
        prow = [r for r in rows if r["problem"] == p]
        if not prow:
            continue
        L.append(f"### `{p}`\n")
        L.append("| model | " + " | ".join(f"`{d}`" for d in discs) + " |")
        L.append("|---|" + "---:|" * len(discs))
        for m in models:
            cells = [cell_text(by[(p, m, d)]) if (p, m, d) in by else "" for d in discs]
            L.append(f"| `{m}` | " + " | ".join(cells) + " |")
        L.append("")

    # ---- details of non-ok cells -----------------------------------------------
    bad = [r for r in rows if r["status"] in ("error", "timeout")]
    L.append("## Errors and timeouts\n")
    if not bad:
        L.append("None.\n")
    else:
        L.append("| problem | model | disc | status | message |")
        L.append("|---|---|---|---|---|")
        for r in bad:
            L.append(f"| `{r['problem']}` | `{r['model']}` | `{r['disc']}` | {r['status']} | "
                     f"{r['message'].replace('|', '/')[:400]} |")
        L.append("")
    div = [r for r in rows if r["status"] == "diverged"]
    L.append("## Diverged cells\n")
    if not div:
        L.append("None.\n")
    else:
        L.append("| problem | model | disc | diverged at (s) | ‖E(x0)‖ | warnings |")
        L.append("|---|---|---|---:|---:|---|")
        for r in div:
            L.append(f"| `{r['problem']}` | `{r['model']}` | `{r['disc']}` | {r['diverged_at']:.2f} | "
                     f"{r['res0']:.2e} | {r['warnings'].replace('|', '/')[:200]} |")
        L.append("")
    grew = [r for r in rows if r["status"] == "ok" and np.isfinite(r["res0"]) and r["res0"] > 0
            and r["resT"] > r["res0"]]
    L.append("## Finite runs whose residual grew (‖E(x(T))‖ > ‖E(x(0))‖ > 0)\n")
    if not grew:
        L.append("None.\n")
    else:
        L.append("| problem | model | disc | ‖E(x0)‖ | ‖E(x(T))‖ |")
        L.append("|---|---|---|---:|---:|")
        for r in grew:
            L.append(f"| `{r['problem']}` | `{r['model']}` | `{r['disc']}` | {r['res0']:.2e} | {r['resT']:.2e} |")
        L.append("")
    warned = {}
    for r in rows:
        if r["warnings"]:
            for w in r["warnings"].split(" | "):
                warned.setdefault(w, []).append(f"{r['problem']}/{r['model']}/{r['disc']}")
    L.append("## Warnings raised (excluding numpy floating-point warnings, which are silenced)\n")
    if not warned:
        L.append("None.\n")
    else:
        L.append("| warning | cells | example |")
        L.append("|---|---:|---|")
        for w, cells in sorted(warned.items(), key=lambda kv: -len(kv[1])):
            L.append(f"| {w.replace('|', '/')} | {len(cells)} | `{cells[0]}` |")
        L.append("")
    if args.notes and Path(args.notes).exists():
        L.append(Path(args.notes).read_text().rstrip() + "\n")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text("\n".join(L))
    tmp.replace(path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--problems", nargs="*", default=None, help="problem keys (default: all registered)")
    ap.add_argument("--models", nargs="*", default=None, help="model keys (default: all registered)")
    ap.add_argument("--discs", nargs="*", default=list(DEFAULT_DISCS))
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=1e-2)
    ap.add_argument("--gamma", type=float, default=10.0)
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--info", default="analytic", choices=("analytic", "bd1", "bd2", "none"))
    ap.add_argument("--noise", default=None, help="noise key (default parameters), e.g. constant")
    ap.add_argument("--data-noise", type=float, default=0.0)
    ap.add_argument("--max-wall", type=float, default=60.0, help="seconds per simulate call")
    ap.add_argument("--default-x0", action="store_true",
                    help="use the problem's random x0 also for the explicit-inverse models")
    ap.add_argument("--out", default=str(ROOT / "docs" / "spec" / "SMOKE_MATRIX.md"))
    ap.add_argument("--csv", default=None, help="optional CSV with one row per cell")
    ap.add_argument("--notes", default=str(ROOT / "experiments" / "smoke_matrix_notes.md"),
                    help="Markdown appended to the report (interpretation of the results)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    if args.T > 1.0 + 1e-12 or args.B > 4 or args.tau < 1e-3:
        ap.error("the smoke matrix must stay small: T <= 1, B <= 4, tau >= 1e-3")

    avail = registry.available()
    problems = args.problems or avail["problems"]
    models = args.models or avail["models"]
    discs = avail["discretizations"] if args.discs == ["all"] else args.discs
    rows = []
    t0 = time.perf_counter()
    for p in problems:
        for m in models:
            for d in discs:
                r = run_cell(p, m, d, args.T, args.tau, args.gamma, args.B, args.seed, args.max_wall,
                             args.default_x0, args.info, args.noise, args.data_noise)
                rows.append(r)
                if not args.quiet and r["status"] in ("error", "timeout"):
                    print(f"[{r['status']}] {p} / {m} / {d}: {r['message'][:300]}", flush=True)
        if not args.quiet:
            print(f"done {p} ({time.perf_counter() - t0:.1f} s)", flush=True)
    total = time.perf_counter() - t0
    write_markdown(rows, args.out, args, problems, models, discs, total)
    if args.csv:
        import csv
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    from collections import Counter
    print(dict(Counter(r["status"] for r in rows)), f"{total:.1f} s", f"peak RSS {_peak_rss_mb():.0f} MB")
    return rows


if __name__ == "__main__":
    main()
