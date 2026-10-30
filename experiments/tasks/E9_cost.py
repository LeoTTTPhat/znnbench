#!/usr/bin/env python
"""E9 per-step cost profile (one memguard task, written by E9_build.py).

    python experiments/tasks/E9_cost.py --out <results>/E9/cost/cost_profile-<hash>.json --scale mini|full

For every (problem, method) of the E9 sweep it measures the *steady* per-step
call counts of the Problem methods (E, dEdt, solve_J, JT, data) through
``znnbench.wrappers.profile_step_cost`` -- the difference of a 240-step and a
40-step run, so one-time costs (the power-iteration Lipschitz estimate of
pc_gradient, look-ahead start-up) are excluded -- and converts them to FLOPs
per step and per trajectory (B = 1):

* if ``znnbench.costs.step_flops`` exists (structured FLOP model being added by
  the zoo workflow) it is used, its arguments being matched by name;
* otherwise the dense model ``wrappers.dense_flops`` (E, dEdt, JT: 2 m n;
  solve_J: 2/3 n^3 + 2 m n; data acquisition free) plus the vector updates of
  the scheme (ASSUMPTION, E9_NOTES.md): look-ahead with s+1 coefficients
  2 (s+1) n; prediction 2 n; Newton correction n; gradient correction 2 n.

Both are recorded when the structured model is available.  Counts do not depend
on tau, the seed or the gain, so one instance (seed 0) per problem is profiled.
The JSON is written atomically (temporary file + os.replace): it is the task's
done file.

Algorithmic cost (review 2026-10-18, COST_VERSION 3).  ``step_flops`` skips the
dE/dt that simulate evaluates for models with ``uses_dEdt = False`` (GNN) but still
charges the ``ctx.E`` that simulate evaluates for models that ignore E (Davidenko,
the predictor of every PC scheme).  ``flops_per_step`` therefore removes one E call
per vector-field call for those models, symmetrically with the dE/dt skip; the
code-view total (unused E charged) is kept as ``flops_structured``.  The correction
is applied only while ``step_flops`` still counts that E call (detected from its call
counts), so an upstream fix is not subtracted twice: costs.py of 2026-10-18 18:27 added
``skip_unused_E`` (default = the algorithmic view), with which E9 takes the library
total as is and computes the code view with ``skip_unused_E=False``.
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse  # noqa: E402
import importlib.util  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import uuid  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CONFIG_PATH = ROOT / "experiments" / "configs" / "E9.py"
PROFILE_TAU = 1e-2
STEPS = (40, 240)
COST_VERSION = 3                    # 3: algorithmic cost skips the unused E of Davidenko-based schemes
#: models whose vector field ignores E (simulate still evaluates ctx.E for them)
MODELS_IGNORING_E = {"davidenko"}


def load_config_module():
    spec = importlib.util.spec_from_file_location("_e9_config", CONFIG_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def method_entries(cfgmod, avail=None) -> list:
    """(model, disc, disc_kwargs, gain kind) of every E9 method, fixed order."""
    avail = avail or cfgmod._available()
    out = []
    for f in cfgmod.znn_formulas(avail):
        if cfgmod.formula_gains(f):
            out.append({"model": "znn", "disc": f, "C": None, "gain": "h_opt"})
    if "gnn" in avail["models"]:
        out.append({"model": "gnn", "disc": "euler", "C": None, "gain": "eta"})
    if "davidenko" in avail["models"]:
        out.append({"model": "davidenko", "disc": "euler", "C": None, "gain": "none"})
        for k in cfgmod.PC_KEYS:
            if k in avail["discretizations"]:
                for C in cfgmod.PC_CS:
                    out.append({"model": "davidenko", "disc": k, "C": int(C), "gain": "none"})
    return out


def resolve_structured():
    """znnbench.costs.step_flops if the zoo provides it, else None."""
    try:
        if importlib.util.find_spec("znnbench.costs") is None:
            return None
        mod = importlib.import_module("znnbench.costs")
    except Exception:  # noqa: BLE001
        return None
    fn = getattr(mod, "step_flops", None)
    return fn if callable(fn) else None


def call_structured(fn, problem, model, disc, tau, extra=None):
    """Call step_flops with the arguments it names (plus ``extra`` options it accepts);
    returns (total, raw)."""
    pool = {"problem": problem, "model": model, "disc": disc, "discretization": disc,
            "discretizer": disc, "scheme": disc, "tau": tau, "B": 1, "batch": 1,
            "n": problem.n_x, "n_x": problem.n_x, "m": problem.m, **(extra or {})}
    kwargs = {}
    for name, par in inspect.signature(fn).parameters.items():
        if name in pool:
            kwargs[name] = pool[name]
        elif par.default is inspect.Parameter.empty and par.kind in (
                inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY):
            raise TypeError(f"step_flops needs an unknown argument {name!r}")
    raw = fn(**kwargs)
    if isinstance(raw, dict):
        for key in ("total", "flops", "flops_per_step", "step"):
            if key in raw and isinstance(raw[key], (int, float)):
                return float(raw[key]), raw
        vals = [v for v in raw.values() if isinstance(v, (int, float))]
        if not vals:
            raise TypeError("step_flops returned a dict without numeric totals")
        return float(sum(vals)), raw
    return float(raw), raw


def vector_update_flops(disc, n_x: int) -> float:
    """State-update arithmetic of the scheme (ASSUMPTION of the fallback model)."""
    if hasattr(disc, "c_past"):                         # look-ahead: tau*xdot - sum c_j x_j, / c_next
        return 2.0 * (len(disc.c_past) + 1) * n_x
    if hasattr(disc, "correction") and hasattr(disc, "C"):
        pred = 2.0 * n_x if getattr(disc, "predict", True) else 0.0
        per = 1.0 * n_x if disc.correction == "newton" else 2.0 * n_x
        return pred + disc.C * per
    return 2.0 * n_x


def profile(cfgmod, pname, entry, structured):
    import numpy as np
    from znnbench.registry import make_discretization, make_model
    from znnbench.wrappers import dense_flops, profile_step_cost

    P = cfgmod.make_instance(pname, 0)
    if entry["gain"] == "h_opt":
        gamma = cfgmod.formula_stability(entry["disc"])["h_opt"] / PROFILE_TAU
    elif entry["gain"] == "eta":
        gamma = 1.0 / (PROFILE_TAU * cfgmod.jacobian_bounds(pname, 0)["sigma_max"] ** 2)
    else:
        gamma = 1.0 / PROFILE_TAU
    dkw = {} if entry["C"] is None else {"C": entry["C"]}
    runs = []
    for steps in STEPS:
        model, disc = make_model(entry["model"]), make_discretization(entry["disc"], **dkw)
        r = profile_step_cost(P, model, disc, tau=PROFILE_TAU, steps=steps, B=1, gamma=gamma)
        runs.append((steps, r, disc))
    (s1, r1, _), (s2, r2, disc) = runs
    calls = {k: (r2["calls_per_step"][k] * s2 - r1["calls_per_step"][k] * s1) / (s2 - s1)
             for k in r2["calls_per_step"]}
    calls_raw = {k: float(round(v, 6)) for k, v in calls.items()}
    # harness artifacts: simulate's vector field evaluates ctx.E and (info='analytic') dEdt at
    # every model call whether or not the model uses them.  One model call per step = one dEdt
    # call (causal branch), so remove dEdt for models with uses_dEdt = False (GNN) and E for
    # models that ignore E (Davidenko); the scheme's own correction calls are kept.
    calls = dict(calls_raw)
    f_calls = calls_raw.get("dEdt", 0.0)
    if not getattr(make_model(entry["model"]), "uses_dEdt", True):
        calls["dEdt"] = 0.0
    if entry["model"] in MODELS_IGNORING_E:
        calls["E"] = max(0.0, calls["E"] - f_calls)
    fl = dense_flops(P.n_x, P.m, 1)
    breakdown = {k: calls[k] * fl[k] for k in calls}
    vec = vector_update_flops(disc, P.n_x)
    dense_total = float(sum(breakdown.values()) + vec)
    out = {"problem": pname, "problem_key": cfgmod.PROBLEMS[pname]["key"], "n_x": int(P.n_x),
           "m": int(P.m), "model": entry["model"], "disc": entry["disc"], "C": entry["C"],
           "calls_per_step": calls, "calls_per_step_raw": calls_raw, "model_calls_per_step": f_calls,
           "flops_dense_breakdown": breakdown, "flops_vector_update": vec,
           "flops_dense_model": dense_total,
           "wall_per_step_profile": float(r2["wall_per_step"]),
           "wall_per_step_by_method": r2["wall_per_step_by_method"],
           "profile_finite": bool(np.isfinite(r2["flops_per_step_dense"]))}
    out["flops_per_step"], out["cost_source"] = dense_total, "profile_step_cost+dense_flops"
    if structured is not None:
        try:
            tot, raw = call_structured(structured, P, make_model(entry["model"]),
                                       make_discretization(entry["disc"], **dkw), PROFILE_TAU)
            out["flops_structured_default"] = tot
            out["flops_structured_raw"] = {k: (float(v) if isinstance(v, (int, float)) else str(v))
                                           for k, v in (raw.items() if isinstance(raw, dict) else
                                                        [("total", raw)])}
            # algorithmic view: drop the E that simulate evaluates for a model that ignores it, but only
            # while step_flops still charges it (its E count equals simulate's raw count).  costs.py of
            # 2026-10-18 18:27 added skip_unused_E (default: algorithmic view), which removes the call
            # itself, so the E count drops by one and nothing is subtracted here.
            unused_e = 0.0
            ignores_e = entry["model"] in MODELS_IGNORING_E and f_calls > 0
            s_calls = raw.get("calls") if isinstance(raw, dict) else None
            charged = (isinstance(s_calls, dict)
                       and float(s_calls.get("E", 0.0)) >= calls_raw.get("E", 0.0) - 1e-9)
            if ignores_e and charged:
                unused_e = f_calls * e_call_flops(P)
            out["flops_unused_E_removed"] = unused_e
            out["flops_per_step"] = tot - unused_e
            # code view (what simulate executes: the unused E charged; GNN's unused dE/dt still skipped,
            # as in the pre-review primary cost)
            code = tot
            if ignores_e and not charged:
                if "skip_unused_E" in inspect.signature(structured).parameters:
                    code, _ = call_structured(structured, P, make_model(entry["model"]),
                                              make_discretization(entry["disc"], **dkw), PROFILE_TAU,
                                              extra={"skip_unused_E": False})
                else:
                    code = tot + f_calls * e_call_flops(P)
            out["flops_structured"] = float(code)
            out["cost_source"] = "znnbench.costs.step_flops"
        except Exception as exc:  # noqa: BLE001
            out["structured_error"] = f"{type(exc).__name__}: {exc}"
    return out


def e_call_flops(problem) -> float:
    """FLOPs of one E(x, d) call with B = 1, data generation excluded (``costs.problem_cost``)."""
    from znnbench.costs import problem_cost
    return float(sum(v for k, v in problem_cost(problem).E(1).items() if k != "data"))


def compute(scale: str) -> dict:
    os.environ["E9_SCALE"] = scale
    cfgmod = load_config_module()
    avail = cfgmod._available()
    structured = resolve_structured()
    t0 = time.perf_counter()
    entries = []
    for pname in cfgmod.PROBLEM_ORDER:
        if cfgmod.PROBLEMS[pname]["key"] not in avail["problems"]:
            continue
        for e in method_entries(cfgmod, avail):
            entries.append(profile(cfgmod, pname, e, structured))
    return {"format": "znnbench-E9-cost/1", "cost_version": COST_VERSION, "scale": scale,
            "profile_tau": PROFILE_TAU, "steps": list(STEPS),
            "structured_model": structured is not None,
            "note": "per trajectory (B = 1), steady per-step counts (difference of two run lengths)",
            "wall_total": time.perf_counter() - t0, "entries": entries}


def atomic_write_json(obj, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:6]}")
    try:
        tmp.write_text(json.dumps(obj, indent=1, default=float))
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output JSON (the memguard done file)")
    ap.add_argument("--scale", choices=("mini", "full"), default="mini")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists() and not a.force:
        print(f"{out} exists, skipping")
        return 0
    res = compute(a.scale)
    atomic_write_json(res, out)
    print(f"{len(res['entries'])} cost entries -> {out} ({res['wall_total']:.1f}s, "
          f"structured={res['structured_model']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
