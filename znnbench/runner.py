"""Experiment runner: sweep configs -> run specs -> chunks/tasks -> results.

Pipeline
--------
1. A *config* (Python dict, YAML/JSON file, or a ``.py`` file defining
   ``CONFIG``/``CONFIGS``) describes a sweep over problems, models,
   discretizations, sampling periods tau, gains (``fixed_h``: gamma = h/tau,
   or ``fixed_gamma``), information regimes, noise specs, wrapper chains and
   seeds, plus scalar settings (B, T, record_every, metrics, ...).
2. :func:`expand` turns the config into a list of *run specs* (plain JSON
   dicts).  Each spec carries a stable id: the first 16 hex digits of the
   SHA-1 of the canonical JSON of everything that determines the numbers it
   produces (output-only options such as labels, traces, max_wall are not
   hashed into the run id).
3. :func:`chunk` groups specs; :func:`write_tasks` writes one chunk file per
   group plus a ``tasks.jsonl`` for ``tools/memguard.py pool`` (one line
   ``{"id", "cmd", "done", "cwd", "env"}`` per chunk; ``done`` is the chunk's
   result parquet, so finished chunks are skipped on resume).  The task id
   hashes the complete specs, output-only options included, so changing e.g.
   ``traces`` or ``max_wall`` re-runs the affected chunks.
4. ``experiments/run_task.py --chunk <file>`` calls :func:`execute_chunk`,
   which runs every spec under try/except (errors, ``TimeoutError`` from
   ``simulate(max_wall=...)``, divergence and incompatibility skips are all
   *recorded*), computes ``metrics.summarize`` per trajectory, wall-clock per
   step (optionally from ``timing_repeats`` lean runs without recording
   overhead) and optional down-sampled traces, and writes ONE parquet per
   chunk atomically (temporary file + ``os.replace``).
5. :func:`aggregate` concatenates the current chunk parquets (those listed in
   ``tasks.jsonl``) into one DataFrame with one row per trajectory and all
   config columns flattened.

See ``docs/spec/runner.md`` for the full config reference and
``experiments/README.md`` for launching sweeps.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import itertools
import json
import math
import os
import re
import sys
import time
import traceback
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

SPEC_VERSION = 1
CHUNK_FORMAT = "znnbench-chunk/1"

ZNN_ROOT = Path(__file__).resolve().parents[1]          # repository root (znn-bench/)
PROJECT_ROOT = ZNN_ROOT                                 # tools/ and logs/ are inside the repository
_VENV_PY = PROJECT_ROOT / ".venv" / "bin" / "python"
DEFAULT_PYTHON = str(_VENV_PY) if _VENV_PY.exists() else sys.executable
MEMGUARD = PROJECT_ROOT / "tools" / "memguard.py"
RUN_TASK = "experiments/run_task.py"                    # relative to ZNN_ROOT (task cwd)

# one BLAS/OpenMP thread per worker: the pool provides the parallelism
THREAD_ENV = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
              "VECLIB_MAXIMUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}

# config keys -------------------------------------------------------------------
_AXES = ("problems", "models", "discretizations", "taus", "infos", "noises", "wrappers", "seeds")
_SCALARS = ("name", "description", "mode", "hs", "gammas", "B", "T", "record_every", "metrics",
            "traces", "max_wall", "div_threshold", "record_solution_error", "batch_gains",
            "exclude", "skip_unsupported_implicit", "timing_repeats", "chunk_size", "balance")
_ALIASES = {"discs": "discretizations", "disc": "discretizations", "h": "hs", "gamma": "gammas",
            "seed": "seeds", "tau": "taus", "info": "infos", "noise": "noises"}
_COMPONENTS = ("problem", "model", "disc", "noise")
_WRAPPER_TARGETS = ("problem", "model", "disc", "noise")
GAIN_MODES = ("fixed_h", "fixed_gamma")
# keys of a run spec that do not change the numbers it produces (not hashed)
_NOT_HASHED = ("id", "config", "traces", "max_wall", "timing_repeats")
STANDARD_METRICS = ("mssre", "mean_ssre", "chatter", "diverged", "ss_sol_err", "t_eps")
# problem instances kept per chunk; seeds are the innermost expansion axis, so with
# "seed": "$seed" consecutive specs cycle through len(seeds) instances (PLAN: 3 seeds)
PROBLEM_CACHE_SIZE = 4


# =============================================================================
# canonical JSON and stable hashing
# =============================================================================
def _plain(obj):
    """Recursively convert numpy scalars/arrays, tuples and sets to plain JSON types."""
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted((_plain(v) for v in obj), key=lambda v: json.dumps(v, sort_keys=True))
    if isinstance(obj, np.ndarray):
        return _plain(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if obj is None or isinstance(obj, (str, bool, int, float)):
        return obj
    raise TypeError(f"config value {obj!r} of type {type(obj).__name__} is not JSON-serialisable")


def _hash_form(obj):
    """Integral floats -> int so that 10 and 10.0 hash identically."""
    if isinstance(obj, dict):
        return {k: _hash_form(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_hash_form(v) for v in obj]
    if (isinstance(obj, float) and not isinstance(obj, bool) and math.isfinite(obj)
            and obj == int(obj) and abs(obj) < 2 ** 53):
        return int(obj)
    return obj


def canonical_json(obj) -> str:
    """Deterministic JSON: plain types, sorted keys, no whitespace, shortest
    round-trip float repr, integral floats written as ints."""
    return json.dumps(_hash_form(_plain(obj)), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=True)


def stable_hash(obj, n: int = 16) -> str:
    return hashlib.sha1(canonical_json(obj).encode("ascii")).hexdigest()[:n]


def _strip_component(c):
    return None if c is None else {"key": c["key"], "kwargs": c["kwargs"]}


def hash_payload(spec: dict) -> dict:
    """The part of a run spec that determines its numerical output."""
    core = {k: v for k, v in spec.items() if k not in _NOT_HASHED}
    for c in _COMPONENTS:
        if c in core:
            core[c] = _strip_component(core[c])
    core["wrappers"] = [{"key": w["key"], "target": w.get("target"), "kwargs": w.get("kwargs", {})}
                        for w in core.get("wrappers", [])]
    return core


def run_id(spec: dict) -> str:
    """Stable 16-hex-digit id of a run spec (hash of canonical JSON)."""
    return stable_hash(hash_payload(spec))


# =============================================================================
# config loading
# =============================================================================
def _yaml_loader():
    import yaml

    class Loader(yaml.SafeLoader):
        pass

    # PyYAML follows YAML 1.1 and reads "1e-3" (no dot) as a string; accept it as float.
    Loader.add_implicit_resolver(
        "tag:yaml.org,2002:float",
        re.compile(r"""^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
                    |\.[0-9_]+(?:[eE][-+]?[0-9]+)?
                    |[-+]?\.(?:inf|Inf|INF)
                    |\.(?:nan|NaN|NAN))$""", re.X),
        list("-+0123456789."))
    return Loader


def yaml_loads(text: str):
    """Parse YAML with exponent floats such as ``1e-3`` read as floats."""
    import yaml
    return yaml.load(text, Loader=_yaml_loader())  # noqa: S506 (SafeLoader subclass)


def load_configs(src) -> List[dict]:
    """Load one or several sweep configs.

    ``src`` may be a dict, a list of dicts, or a path to a ``.yaml``/``.yml``,
    ``.json`` or ``.py`` file.  A ``.py`` file must define ``CONFIG`` (dict),
    ``CONFIGS`` (list of dicts) or a function ``get_configs()``.  Configs
    without a ``name`` get the file stem (with an index if several)."""
    if isinstance(src, dict):
        return [src]
    if isinstance(src, (list, tuple)):
        return [dict(c) for c in src]
    p = Path(src)
    if p.suffix in (".yaml", ".yml"):
        data = yaml_loads(p.read_text())
    elif p.suffix == ".json":
        data = json.loads(p.read_text())
    elif p.suffix == ".py":
        spec = importlib.util.spec_from_file_location(f"_znnbench_cfg_{p.stem}", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if hasattr(mod, "get_configs"):
            data = mod.get_configs()
        elif hasattr(mod, "CONFIGS"):
            data = mod.CONFIGS
        elif hasattr(mod, "CONFIG"):
            data = mod.CONFIG
        else:
            raise ValueError(f"{p} defines none of CONFIG, CONFIGS, get_configs()")
    else:
        raise ValueError(f"unsupported config file type: {p}")
    cfgs = [data] if isinstance(data, dict) else list(data)
    out = []
    for i, c in enumerate(cfgs):
        c = dict(c)
        c.setdefault("name", p.stem if len(cfgs) == 1 else f"{p.stem}-{i}")
        out.append(c)
    return out


# =============================================================================
# config normalisation
# =============================================================================
def _as_list(v):
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return list(v)
    return [v]


def _fmt_val(v):
    if isinstance(v, float):
        return f"{v:g}"
    if isinstance(v, (str, int, bool)) or v is None:
        return str(v)
    return canonical_json(v)


def auto_label(key: str, kwargs: dict) -> str:
    if not kwargs:
        return key
    return f"{key}(" + ",".join(f"{k}={_fmt_val(v)}" for k, v in sorted(kwargs.items())) + ")"


class _SafeFormat(dict):
    def __missing__(self, k):
        return "{" + k + "}"


def _label(comp):
    lab = comp.get("label_template")
    if lab is None:
        return auto_label(comp["key"], comp["kwargs"])
    if "{" in lab:
        return lab.format_map(_SafeFormat({k: _fmt_val(v) for k, v in comp["kwargs"].items()}))
    return lab


def _component(c, what):
    if isinstance(c, str):
        c = {"key": c}
    if not isinstance(c, dict) or "key" not in c:
        raise ValueError(f"{what} entry must be a key string or a dict with 'key': {c!r}")
    bad = set(c) - {"key", "kwargs", "label"}
    if bad:
        raise ValueError(f"{what} entry {c!r} has unknown fields {sorted(bad)}")
    kw = c.get("kwargs") or {}
    if not isinstance(kw, dict):
        raise ValueError(f"{what} kwargs must be a dict: {c!r}")
    return {"key": str(c["key"]), "kwargs": _plain(dict(kw)), "label_template": c.get("label")}


def _is_sweep(v):
    return isinstance(v, dict) and set(v) == {"$sweep"}


def _expand_sweeps(comp):
    """kwargs values of the form {"$sweep": [...]} expand into a grid of components."""
    kw = comp["kwargs"]
    swept = sorted(k for k, v in kw.items() if _is_sweep(v))
    if not swept:
        return [comp]
    out = []
    for vals in itertools.product(*[_as_list(kw[k]["$sweep"]) for k in swept]):
        new = copy.deepcopy(comp)
        new["kwargs"].update(dict(zip(swept, vals)))
        out.append(new)
    return out


def _subst(obj, seed, tau):
    """Replace the placeholder strings "$seed" (run seed) and "$tau" (sampling period)."""
    if isinstance(obj, dict):
        return {k: _subst(v, seed, tau) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_subst(v, seed, tau) for v in obj]
    if isinstance(obj, str):
        if obj == "$seed":
            return int(seed)
        if obj == "$tau":
            return float(tau)
    return obj


def _finish_component(comp, seed, tau):
    if comp is None:
        return None
    kw = _subst(comp["kwargs"], seed, tau)
    out = {"key": comp["key"], "kwargs": kw}
    out["label"] = _label({**comp, "kwargs": kw})
    return out


def _noise_component(n):
    if n is None or (isinstance(n, str) and n.lower() == "none"):
        return None
    if isinstance(n, dict) and str(n.get("key", "")).lower() == "none":
        return None
    return _component(n, "noise")


def _info_entry(i):
    from .simulate import INFO_REGIMES
    if isinstance(i, str):
        i = {"info": i}
    if not isinstance(i, dict):
        raise ValueError(f"info entry must be a string or dict: {i!r}")
    i = dict(i)
    if "key" in i and "info" not in i:
        i["info"] = i.pop("key")
    bad = set(i) - {"info", "data_noise"}
    if bad:
        raise ValueError(f"info entry {i!r} has unknown fields {sorted(bad)}")
    if i.get("info") not in INFO_REGIMES:
        raise ValueError(f"unknown info regime {i.get('info')!r}; expected one of {INFO_REGIMES}")
    dn = float(i.get("data_noise", 0.0) or 0.0)
    if dn < 0:
        raise ValueError("data_noise must be >= 0")
    return {"info": i["info"], "data_noise": dn}


def _wrapper_entry(w):
    if isinstance(w, str):
        w = {"key": w}
    if not isinstance(w, dict) or "key" not in w:
        raise ValueError(f"wrapper must be a key string or a dict with 'key': {w!r}")
    bad = set(w) - {"key", "target", "kwargs"}
    if bad:
        raise ValueError(f"wrapper {w!r} has unknown fields {sorted(bad)}")
    tgt = w.get("target")
    if tgt is not None and tgt not in _WRAPPER_TARGETS:
        raise ValueError(f"wrapper target must be one of {_WRAPPER_TARGETS}: {w!r}")
    return {"key": str(w["key"]), "target": tgt, "kwargs": _plain(dict(w.get("kwargs") or {}))}


def _wrapper_alternatives(ws):
    """`wrappers` is a list of alternatives; each alternative is None/[] (no
    wrapper), a single wrapper (str/dict) or a chain (list of wrappers).
    ``{"$sweep": [...]}`` kwargs of a wrapper expand into one alternative per
    value (Cartesian product over every swept kwarg of the chain)."""
    if ws is None:
        return [[]]
    alts = []
    for alt in _as_list(ws):
        if alt is None:
            chain = []
        elif isinstance(alt, (list, tuple)):
            chain = [_wrapper_entry(w) for w in alt]
        else:
            chain = [_wrapper_entry(alt)]
        alts.extend(list(c) for c in itertools.product(*[_expand_sweeps(w) for w in chain]))
    return alts or [[]]


def _gain_entries(cfg):
    mode = cfg.get("mode")
    hs = [float(v) for v in _as_list(cfg.get("hs"))]
    gs = [float(v) for v in _as_list(cfg.get("gammas"))]
    if mode not in (None, "both", *GAIN_MODES):
        raise ValueError(f"mode must be one of {GAIN_MODES} or 'both', got {mode!r}")
    if mode == "fixed_h":
        if gs:
            raise ValueError("mode 'fixed_h' given together with 'gammas'")
        if not hs:
            raise ValueError("mode 'fixed_h' needs a list 'hs'")
    if mode == "fixed_gamma":
        if hs:
            raise ValueError("mode 'fixed_gamma' given together with 'hs'")
        if not gs:
            raise ValueError("mode 'fixed_gamma' needs a list 'gammas'")
    entries = [("fixed_h", v) for v in hs] + [("fixed_gamma", v) for v in gs]
    if not entries:
        raise ValueError("config needs 'hs' (fixed_h: gamma = h/tau) and/or 'gammas' (fixed_gamma)")
    for m, v in entries:
        if not (v > 0 and math.isfinite(v)):
            raise ValueError(f"{m} gain values must be positive and finite, got {v}")
    return entries


def _metrics_cfg(m):
    if m is None:
        m = {}
    if isinstance(m, (list, tuple)):
        m = {"extra": [x for x in m if x not in STANDARD_METRICS and not str(x).startswith("t_eps")]}
    if not isinstance(m, dict):
        raise ValueError(f"metrics must be a dict or a list of names: {m!r}")
    bad = set(m) - {"eps", "frac", "extra"}
    if bad:
        raise ValueError(f"metrics has unknown fields {sorted(bad)}")
    eps = [float(e) for e in _as_list(m.get("eps", [1e-3, 1e-6]))]
    frac = float(m.get("frac", 0.25))
    if not 0 < frac <= 1:
        raise ValueError("metrics.frac must be in (0, 1]")
    extra = [str(x) for x in _as_list(m.get("extra"))]
    return {"eps": eps, "frac": frac, "extra": extra}


def _traces_cfg(t):
    if t in (None, False):
        return None
    if t is True:
        t = {}
    if not isinstance(t, dict):
        raise ValueError(f"traces must be a dict like {{'points': 256}} or {{'every': 10}}: {t!r}")
    bad = set(t) - {"points", "every"}
    if bad:
        raise ValueError(f"traces has unknown fields {sorted(bad)}")
    if "every" in t:
        return {"every": int(t["every"])}
    return {"points": int(t.get("points", 256))}


def normalize_config(config: dict) -> dict:
    """Validate a sweep config and fill defaults (returns a new dict)."""
    cfg = {}
    for k, v in dict(config).items():
        k2 = _ALIASES.get(k, k)
        if k2 in cfg:
            raise ValueError(f"config key {k!r} given twice (alias of {k2!r})")
        cfg[k2] = v
    unknown = set(cfg) - set(_AXES) - set(_SCALARS)
    if unknown:
        raise ValueError(f"unknown config keys {sorted(unknown)}; allowed: {sorted(_AXES + _SCALARS)}")
    for req in ("problems", "models", "discretizations", "taus"):
        if not _as_list(cfg.get(req)):
            raise ValueError(f"config needs a non-empty list {req!r}")
    out = {"name": re.sub(r"[^A-Za-z0-9_.-]+", "_", str(cfg.get("name", "sweep"))) or "sweep"}
    out["problems"] = [x for c in _as_list(cfg["problems"]) for x in _expand_sweeps(_component(c, "problem"))]
    out["models"] = [x for c in _as_list(cfg["models"]) for x in _expand_sweeps(_component(c, "model"))]
    out["discretizations"] = [x for c in _as_list(cfg["discretizations"])
                              for x in _expand_sweeps(_component(c, "discretization"))]
    nz = [_noise_component(n) for n in (_as_list(cfg["noises"]) if "noises" in cfg else [None])]
    out["noises"] = [x for n in nz for x in ([None] if n is None else _expand_sweeps(n))] or [None]
    out["taus"] = [float(t) for t in _as_list(cfg["taus"])]
    if any(not (t > 0 and math.isfinite(t)) for t in out["taus"]):
        raise ValueError("taus must be positive and finite")
    out["gains"] = _gain_entries(cfg)
    out["infos"] = [_info_entry(i) for i in (_as_list(cfg.get("infos")) or ["analytic"])]
    out["wrappers"] = _wrapper_alternatives(cfg.get("wrappers"))
    seeds = _as_list(cfg.get("seeds", [0])) or [0]
    if any(isinstance(s, bool) or int(s) != s for s in seeds):
        raise ValueError("seeds must be integers")
    out["seeds"] = [int(s) for s in seeds]
    B = cfg.get("B", 1)
    if isinstance(B, bool) or int(B) != B or int(B) < 1:
        raise ValueError("B must be a positive integer")
    out["B"] = int(B)
    out["T"] = None if cfg.get("T") is None else float(cfg["T"])
    if out["T"] is not None and not (out["T"] > 0 and math.isfinite(out["T"])):
        raise ValueError("T must be positive and finite (or null for the problem's own T)")
    re_ = cfg.get("record_every", 1)
    if isinstance(re_, bool) or int(re_) != re_ or int(re_) < 1:
        raise ValueError("record_every must be a positive integer")
    out["record_every"] = int(re_)
    out["metrics"] = _metrics_cfg(cfg.get("metrics"))
    out["traces"] = _traces_cfg(cfg.get("traces"))
    out["max_wall"] = None if cfg.get("max_wall") is None else float(cfg["max_wall"])
    out["div_threshold"] = float(cfg.get("div_threshold", 1e10))
    if not out["div_threshold"] > 0:
        raise ValueError("div_threshold must be > 0")
    out["record_solution_error"] = bool(cfg.get("record_solution_error", True))
    out["batch_gains"] = bool(cfg.get("batch_gains", False))
    out["skip_unsupported_implicit"] = bool(cfg.get("skip_unsupported_implicit", True))
    tr = cfg.get("timing_repeats", 0)
    if isinstance(tr, bool) or tr is None or int(tr) != tr or int(tr) < 0:
        raise ValueError("timing_repeats must be an integer >= 0 (number of lean timing runs)")
    out["timing_repeats"] = int(tr)
    out["exclude"] = [dict(r) for r in _as_list(cfg.get("exclude"))]
    for r in out["exclude"]:
        bad = set(r) - set(_RULE_FIELDS)
        if bad:
            raise ValueError(f"exclude rule {r!r} has unknown fields {sorted(bad)}; allowed {_RULE_FIELDS}")
    out["chunk_size"] = None if cfg.get("chunk_size") is None else int(cfg["chunk_size"])
    if out["chunk_size"] is not None and out["chunk_size"] < 1:
        raise ValueError("chunk_size must be >= 1")
    out["balance"] = bool(cfg.get("balance", False))
    out["description"] = cfg.get("description")
    return out


# =============================================================================
# exclusion rules
# =============================================================================
_RULE_FIELDS = ("problem", "model", "disc", "noise", "wrappers", "tau", "info", "data_noise",
                "seed", "gain_mode", "h", "gamma")


def _num_eq(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b or a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-300)
    return a == b


def _subset(actual, pattern):
    if isinstance(pattern, dict):
        return isinstance(actual, dict) and all(k in actual and _subset(actual[k], v)
                                                for k, v in pattern.items())
    return _num_eq(actual, pattern)


def _match_component(comp, pat):
    if isinstance(pat, list):
        return any(_match_component(comp, p) for p in pat)
    if comp is None:
        return pat is None or pat == "none"
    if pat is None or pat == "none":
        return False
    if isinstance(pat, str):
        return pat in (comp["key"], comp["label"])
    if isinstance(pat, dict):
        if "key" in pat and pat["key"] != comp["key"]:
            return False
        if "label" in pat and pat["label"] != comp["label"]:
            return False
        return _subset(comp["kwargs"], pat.get("kwargs", {}))
    return False


def _match_wrappers(chain, pat):
    if isinstance(pat, list):
        return any(_match_wrappers(chain, p) for p in pat)
    if pat is None or pat == "none":
        return not chain
    return any(w["key"] == pat for w in chain)


def _match_scalar(v, pat):
    if isinstance(pat, list):
        return any(_num_eq(v, p) for p in pat)
    return _num_eq(v, pat)


def _excluded(spec, rules):
    if not rules:
        return False
    mode, val = spec["gain"]["mode"], spec["gain"]["values"][0]
    derived = {"tau": spec["tau"], "info": spec["info"], "data_noise": spec["data_noise"],
               "seed": spec["seed"], "gain_mode": mode,
               "h": val if mode == "fixed_h" else val * spec["tau"],
               "gamma": val / spec["tau"] if mode == "fixed_h" else val}
    for rule in rules:
        ok = True
        for f, pat in rule.items():
            if f in _COMPONENTS:
                ok = _match_component(spec[f], pat)
            elif f == "wrappers":
                ok = _match_wrappers(spec["wrappers"], pat)
            else:
                ok = _match_scalar(derived[f], pat)
            if not ok:
                break
        if ok:
            return True
    return False


# =============================================================================
# expansion
# =============================================================================
def expand(config, validate: bool = True) -> List[dict]:
    """Expand a sweep config into a list of run specs with stable unique ids.

    Order: problems x models x discretizations x taus x gains x infos x noises
    x wrappers x seeds (seeds innermost).  ``exclude`` rules remove single-gain
    combinations; with ``batch_gains`` the remaining gain values that share
    everything else (and the gain mode) are merged into one spec, simulated
    as one batch of B * G trajectories.  Duplicate specs (same id) are dropped.
    With ``validate`` every registry key is checked to exist.
    """
    cfg = normalize_config(config)
    singles = []
    for prob, model, disc, tau, (gmode, gval), info, noise, wchain, seed in itertools.product(
            cfg["problems"], cfg["models"], cfg["discretizations"], cfg["taus"], cfg["gains"],
            cfg["infos"], cfg["noises"], cfg["wrappers"], cfg["seeds"]):
        spec = {
            "spec_version": SPEC_VERSION,
            "config": cfg["name"],
            "problem": _finish_component(prob, seed, tau),
            "model": _finish_component(model, seed, tau),
            "disc": _finish_component(disc, seed, tau),
            "noise": _finish_component(noise, seed, tau),
            "wrappers": [{**w, "kwargs": _subst(w["kwargs"], seed, tau)} for w in wchain],
            "tau": tau,
            "gain": {"mode": gmode, "values": [gval]},
            "info": info["info"],
            "data_noise": info["data_noise"],
            "seed": seed,
            "B": cfg["B"],
            "T": cfg["T"],
            "record_every": cfg["record_every"],
            "metrics": copy.deepcopy(cfg["metrics"]),
            "div_threshold": cfg["div_threshold"],
            "record_solution_error": cfg["record_solution_error"],
            "skip_unsupported_implicit": cfg["skip_unsupported_implicit"],
            "traces": copy.deepcopy(cfg["traces"]),
            "max_wall": cfg["max_wall"],
            "timing_repeats": cfg["timing_repeats"],
        }
        if not _excluded(spec, cfg["exclude"]):
            singles.append(spec)

    if cfg["batch_gains"]:
        groups: "OrderedDict[str, dict]" = OrderedDict()
        for s in singles:
            key_spec = copy.deepcopy(s)
            key_spec["gain"]["values"] = []
            gk = canonical_json(key_spec)
            if gk in groups:
                vals = groups[gk]["gain"]["values"]
                if not any(_num_eq(v, s["gain"]["values"][0]) for v in vals):
                    vals.append(s["gain"]["values"][0])
            else:
                groups[gk] = s
        singles = list(groups.values())

    specs, seen = [], set()
    for s in singles:
        s["id"] = run_id(s)
        if s["id"] in seen:
            continue
        seen.add(s["id"])
        specs.append(s)
    if validate:
        validate_specs(specs)
    return specs


def validate_specs(specs: Iterable[dict]):
    """Check that every registry key, wrapper and extra metric can be resolved."""
    from .registry import DISCRETIZATIONS, MODELS, NOISES, PROBLEMS, _load_all
    _load_all()
    tables = {"problem": PROBLEMS, "model": MODELS, "disc": DISCRETIZATIONS, "noise": NOISES}
    missing, seen = [], set()
    for s in specs:
        for c, table in tables.items():
            comp = s.get(c)
            if comp is not None and comp["key"] not in table and (c, comp["key"]) not in seen:
                seen.add((c, comp["key"]))
                missing.append(f"{c} {comp['key']!r}")
        for w in s.get("wrappers", []):
            if ("w", w["key"], w.get("target")) not in seen:
                seen.add(("w", w["key"], w.get("target")))
                try:
                    if wrapper_target(w, resolve_wrapper(w["key"])) is None:
                        missing.append(f"wrapper {w['key']!r} (no target: give 'target' in the config)")
                except Exception as exc:  # noqa: BLE001
                    missing.append(f"wrapper {w['key']!r} ({exc!r})")
        for m in s["metrics"]["extra"]:
            if ("m", m) not in seen:
                seen.add(("m", m))
                if m not in EXTRA_METRICS and ":" not in m:
                    missing.append(f"extra metric {m!r}")
    if missing:
        raise ValueError("unknown keys in sweep: " + "; ".join(missing))


# =============================================================================
# chunking and task files
# =============================================================================
def cost_estimate(spec: dict) -> float:
    """Rough relative cost: steps * (1 + trajectories/16) * (1 + lean timing
    runs) (Python overhead per step dominates for small batches).  Unknown T
    counts as 10 s."""
    T = spec["T"] if spec["T"] is not None else 10.0
    steps = T / spec["tau"]
    ntraj = spec["B"] * len(spec["gain"]["values"])
    return steps * (1.0 + ntraj / 16.0) * (1 + max(0, int(spec.get("timing_repeats", 0) or 0)))


def chunk(run_specs: List[dict], size: int, balance: bool = False) -> List[List[dict]]:
    """Split specs into chunks of at most `size` specs.

    Default: consecutive slices of the expansion order (consecutive specs share
    problem instances, which the executor caches).  ``balance=True`` deals the
    specs, sorted by :func:`cost_estimate`, in snake order over the chunks so
    every chunk gets a similar total cost; each chunk keeps expansion order."""
    if size < 1:
        raise ValueError("chunk size must be >= 1")
    specs = list(run_specs)
    if not specs:
        return []
    n = math.ceil(len(specs) / size)
    if not balance or n == 1:
        return [specs[i:i + size] for i in range(0, len(specs), size)]
    order = sorted(range(len(specs)), key=lambda i: (-cost_estimate(specs[i]), i))
    bins: List[List[int]] = [[] for _ in range(n)]
    for j, i in enumerate(order):
        r, q = divmod(j, n)
        bins[q if r % 2 == 0 else n - 1 - q].append(i)
    return [[specs[i] for i in sorted(b)] for b in bins if b]


def _atomic_write_text(path: Path, text: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:6]}")
    try:
        tmp.write_text(text)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_tasks(config_name: str, chunks: List[List[dict]], out_dir, python: Optional[str] = None,
                extra_env: Optional[dict] = None) -> Path:
    """Write ``<out_dir>/chunks/<task>.json`` per chunk and ``<out_dir>/tasks.jsonl``.

    Every line of tasks.jsonl is a ``tools/memguard.py pool`` task::

        {"id": "<config>-00000-<hash>", "cmd": [python, "experiments/run_task.py",
         "--chunk", "<abs chunk file>"], "done": "<abs results/<id>.parquet>",
         "cwd": "<znn-bench root>", "env": {single-thread BLAS settings}}

    The task id embeds a hash of the chunk's *complete* specs (including the
    output-only fields that the run ids leave out: config name, labels,
    traces, max_wall, timing_repeats), so any change to what a chunk would
    write gets a new id and a new done file and is re-run on the next pool
    launch; :func:`aggregate` then reads only the current done files.
    Returns the path of tasks.jsonl."""
    out = Path(out_dir).resolve()
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(config_name)) or "sweep"
    python = python or DEFAULT_PYTHON
    lines = []
    for i, ch in enumerate(chunks):
        if not ch:
            continue
        h = stable_hash([CHUNK_FORMAT, str(config_name), ch], 10)
        tid = f"{safe}-{i:05d}-{h}"
        chunk_path = out / "chunks" / f"{tid}.json"
        result = out / "results" / f"{tid}.parquet"
        traces = out / "results" / f"{tid}.npz"
        payload = {"format": CHUNK_FORMAT, "config": config_name, "task_id": tid,
                   "result": str(result), "traces": str(traces), "specs": ch}
        _atomic_write_text(chunk_path, json.dumps(_plain(payload), indent=1, allow_nan=True))
        env = {**THREAD_ENV, **{str(k): str(v) for k, v in (extra_env or {}).items()}}   # Popen needs str
        task = {"id": tid, "cmd": [python, RUN_TASK, "--chunk", str(chunk_path)],
                "done": str(result), "cwd": str(ZNN_ROOT), "env": env}
        lines.append(json.dumps(task))
    (out / "results").mkdir(parents=True, exist_ok=True)
    tasks_file = out / "tasks.jsonl"
    _atomic_write_text(tasks_file, "\n".join(lines) + ("\n" if lines else ""))
    return tasks_file


def build(config, out_dir, chunk_size: Optional[int] = None, balance: Optional[bool] = None,
          validate: bool = True, python: Optional[str] = None) -> Path:
    """expand -> chunk -> write_tasks; also writes ``<out_dir>/config.json``
    (the normalised config) and ``<out_dir>/specs.jsonl``.  Returns tasks.jsonl."""
    if not isinstance(config, dict):
        cfgs = load_configs(config)
        if len(cfgs) != 1:
            raise ValueError("build() takes a single config; loop over load_configs() for several")
        config = cfgs[0]
    norm = normalize_config(config)
    specs = expand(config, validate=validate)
    size = chunk_size or norm["chunk_size"] or 16
    bal = norm["balance"] if balance is None else balance
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(out / "config.json", json.dumps(_plain(config), indent=1, allow_nan=True, default=str))
    _atomic_write_text(out / "specs.jsonl", "".join(json.dumps(_plain(s)) + "\n" for s in specs))
    return write_tasks(norm["name"], chunk(specs, size, balance=bal), out, python=python)


def load_tasks(tasks_file) -> List[dict]:
    with open(tasks_file) as f:
        return [json.loads(line) for line in f if line.strip()]


def _chunk_of_task(task) -> str:
    cmd = task["cmd"]
    return cmd[cmd.index("--chunk") + 1]


def run_local(tasks_or_dir, force: bool = False, verbose: bool = True) -> List[Path]:
    """Execute every chunk of a tasks.jsonl (or an out_dir containing one)
    sequentially in this process; finished chunks are skipped unless force."""
    p = Path(tasks_or_dir)
    tasks = load_tasks(p / "tasks.jsonl" if p.is_dir() else p)
    return [execute_chunk(_chunk_of_task(t), force=force, verbose=verbose) for t in tasks]


def status(out_dir) -> dict:
    tasks = load_tasks(Path(out_dir) / "tasks.jsonl")
    done = sum(Path(t["done"]).exists() for t in tasks)
    return {"tasks": len(tasks), "done": done, "pending": len(tasks) - done}


# =============================================================================
# wrappers and extra metrics
# =============================================================================
WRAPPERS: Dict[str, Any] = {}


def register_wrapper(key: str, target: Optional[str] = None):
    """Register ``fn(component, **kwargs) -> component`` under `key`.

    `target` (problem/model/disc/noise) is the default component the wrapper
    applies to; a wrapper entry in a config may override it.  Wrappers must
    return a new object and must not mutate their input."""
    def deco(fn):
        if key in WRAPPERS:
            raise KeyError(f"duplicate wrapper key {key!r}")
        if target is not None:
            if target not in _WRAPPER_TARGETS:
                raise ValueError(f"wrapper target must be one of {_WRAPPER_TARGETS}")
            try:
                fn.target = target
            except (AttributeError, TypeError):
                pass
        WRAPPERS[key] = fn
        return fn
    return deco


def _import_path(path: str):
    mod, _, attr = path.partition(":")
    obj = importlib.import_module(mod)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def resolve_wrapper(key: str):
    """Look a wrapper up in (1) :data:`WRAPPERS`, (2) an import path
    ``"package.module:callable"``, (3) a ``znnbench.wrappers`` package if one
    exists (its ``WRAPPERS``/``REGISTRY`` dict or a module attribute)."""
    if key in WRAPPERS:
        return WRAPPERS[key]
    if ":" in key:
        return _import_path(key)
    if importlib.util.find_spec("znnbench.wrappers") is not None:
        mod = importlib.import_module("znnbench.wrappers")
        if key in WRAPPERS:                      # registered on import
            return WRAPPERS[key]
        for reg in ("WRAPPERS", "REGISTRY"):
            table = getattr(mod, reg, None)
            if isinstance(table, dict) and key in table:
                return table[key]
        if callable(getattr(mod, key, None)):
            return getattr(mod, key)
    raise KeyError(f"unknown wrapper {key!r}")


class PeakXdotProbe:
    """Transparent model proxy recording max_k ||x_dot||_2 per trajectory over
    every evaluation of ``rhs`` (all stages of a non-causal scheme included;
    the design-formula noise added by ``simulate`` is not included).  Reset
    when ``simulate`` calls ``z0``.  Used by the extra metric ``peak_xdot``."""

    def __init__(self, inner):
        self._inner = inner
        self._peak = None

    def __getattr__(self, name):
        if name in ("_inner", "_peak"):
            raise AttributeError(name)
        return getattr(self._inner, name)

    def z0(self, problem, B):
        self._peak = np.zeros(B)
        return self._inner.z0(problem, B)

    def rhs(self, ctx):
        xdot, zdot = self._inner.rhs(ctx)
        nrm = np.linalg.norm(xdot, axis=1)
        nrm = np.where(np.isnan(nrm), np.inf, nrm)
        self._peak = nrm if self._peak is None or self._peak.shape != nrm.shape else np.maximum(self._peak, nrm)
        return xdot, zdot

    def extra_metrics(self, sim):
        return {"peak_xdot": self._peak}


def _m_final_residual(sim):
    return sim["residual"][-1]


def _m_initial_residual(sim):
    return sim["residual"][0]


def _m_max_residual(sim):
    return np.max(sim["residual"], axis=0)


def _m_final_sol_err(sim):
    return None if sim["sol_err"] is None else sim["sol_err"][-1]


EXTRA_METRICS = {
    "final_residual": _m_final_residual,      # ||E(x_N, t_N)||
    "initial_residual": _m_initial_residual,  # ||E(x_0, 0)||
    "max_residual": _m_max_residual,          # max_k ||E(x_k, t_k)|| over recorded samples
    "final_sol_err": _m_final_sol_err,        # ||x_N - x*(t_N)||
    "peak_xdot": None,                        # via PeakXdotProbe (instrumented model)
}


# =============================================================================
# execution
# =============================================================================
def _peak_rss_mb() -> float:
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / 2 ** 20 if sys.platform == "darwin" else r / 2 ** 10
    except Exception:  # noqa: BLE001
        return float("nan")


def _scalar(v):
    """numpy scalar -> plain Python scalar for a DataFrame cell."""
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    if isinstance(v, (str, np.str_)):
        return str(v)
    return _cell(_plain(v)) if not isinstance(v, np.generic) else v.item()


def _cell(v):
    if v is None or isinstance(v, (str, bool, int, float)):
        return v
    return canonical_json(v)


def _describe(obj) -> str:
    try:
        return json.dumps(obj.describe(), sort_keys=True, default=str)
    except Exception as exc:  # noqa: BLE001
        return f"describe() failed: {exc!r}"


def spec_columns(spec: dict) -> dict:
    """Flattened config columns of a run spec (shared by all its trajectories)."""
    cols = {"config": spec.get("config"), "run_id": spec["id"], "spec_version": spec["spec_version"]}
    for c in _COMPONENTS:
        comp = spec.get(c)
        if comp is None:
            cols[f"{c}_key"] = "none"
            cols[f"{c}_label"] = "none"
            continue
        cols[f"{c}_key"] = comp["key"]
        cols[f"{c}_label"] = comp.get("label") or auto_label(comp["key"], comp["kwargs"])
        for k, v in sorted(comp["kwargs"].items()):
            cols[f"{c}_kw_{k}"] = _cell(v)
    ws = spec.get("wrappers", [])
    cols["wrappers"] = canonical_json(ws) if ws else ""
    cols["wrapper_keys"] = "+".join(w["key"] for w in ws) if ws else "none"
    cols.update({"tau": spec["tau"], "gain_mode": spec["gain"]["mode"], "n_gains": len(spec["gain"]["values"]),
                 "info": spec["info"], "data_noise": spec["data_noise"], "seed": spec["seed"],
                 "B": spec["B"], "T": spec["T"], "record_every": spec["record_every"],
                 "div_threshold": spec["div_threshold"],
                 "metrics_eps": canonical_json(spec["metrics"]["eps"]),
                 "metrics_frac": spec["metrics"]["frac"]})
    return cols


def _gains_of(spec):
    mode, tau = spec["gain"]["mode"], spec["tau"]
    vals = [float(v) for v in spec["gain"]["values"]]
    gammas = [v / tau if mode == "fixed_h" else v for v in vals]
    hs = [v if mode == "fixed_h" else v * tau for v in vals]
    return vals, gammas, hs


def _trace_indices(n_rec, tcfg):
    if "every" in tcfg:
        idx = np.arange(0, n_rec, max(1, tcfg["every"]))
    else:
        idx = np.unique(np.linspace(0, n_rec - 1, max(2, tcfg["points"])).round().astype(int))
    if idx[-1] != n_rec - 1:
        idx = np.append(idx, n_rec - 1)
    return idx


def wrapper_target(w: dict, fn=None) -> Optional[str]:
    """Target of a wrapper entry: explicit ``target``, else the callable's
    ``target`` attribute, else inferred from a class deriving from
    Problem / Discretizer / Model (e.g. the classes in ``znnbench.wrappers``)."""
    if w.get("target"):
        return w["target"]
    fn = fn if fn is not None else resolve_wrapper(w["key"])
    t = getattr(fn, "target", None)
    if t in _WRAPPER_TARGETS:
        return t
    if isinstance(fn, type):
        from .core import Discretizer, Model, Problem
        for base, name in ((Problem, "problem"), (Discretizer, "disc"), (Model, "model")):
            if issubclass(fn, base):
                return name
    return None


def _apply_wrappers(objs, chain):
    for w in chain:
        fn = resolve_wrapper(w["key"])
        tgt = wrapper_target(w, fn)
        if tgt not in _WRAPPER_TARGETS:
            raise ValueError(f"wrapper {w['key']!r} needs a target in {_WRAPPER_TARGETS}")
        if objs[tgt] is None:
            raise ValueError(f"wrapper {w['key']!r} targets {tgt!r}, which is absent in this run")
        objs[tgt] = fn(objs[tgt], **w.get("kwargs", {}))
    return objs


def run_spec(spec: dict, problem_cache: Optional[OrderedDict] = None, want_traces: bool = True):
    """Execute one run spec.

    Returns ``(rows, traces)``: one dict per trajectory (B * G rows, G = number
    of gain values) and a dict of down-sampled trace arrays (empty unless
    ``spec['traces']``).  Never raises for errors inside the run: they become
    rows with ``status`` in {"skipped", "timeout", "error"}."""
    from . import registry
    from .metrics import summarize
    from .simulate import initial_state, simulate

    base = spec_columns(spec)
    vals, gammas, hs = _gains_of(spec)
    B, G = spec["B"], len(vals)
    run = {"status": None, "message": None, "wall": np.nan, "wall_min": np.nan, "wall_recorded": np.nan,
           "timing_mode": None, "timing_runs": 0, "setup_wall": np.nan,
           "steps": np.nan, "steps_run": np.nan, "wall_per_step": np.nan, "n_diverged": np.nan,
           "peak_rss_mb": np.nan}
    metrics_per_traj: Dict[str, np.ndarray] = {}
    status_per_traj = None
    traces = {}
    t_setup = time.perf_counter()
    problem = None
    try:
        # ---------------- build components (with problem cache) ----------------
        pchain = [w for w in spec["wrappers"] if wrapper_target(w) == "problem"]
        pkey = canonical_json(_strip_component(spec["problem"]))
        if problem_cache is not None and not pchain and pkey in problem_cache:
            problem = problem_cache[pkey]
            problem_cache.move_to_end(pkey)
        else:
            problem = registry.make_problem(spec["problem"]["key"], **spec["problem"]["kwargs"])
            if problem_cache is not None and not pchain:
                problem_cache[pkey] = problem
                while len(problem_cache) > PROBLEM_CACHE_SIZE:
                    problem_cache.popitem(last=False)
        objs = {"problem": problem,
                "model": registry.make_model(spec["model"]["key"], **spec["model"]["kwargs"]),
                "disc": registry.make_discretization(spec["disc"]["key"], **spec["disc"]["kwargs"]),
                "noise": (registry.make_noise(spec["noise"]["key"], **spec["noise"]["kwargs"])
                          if spec["noise"] is not None else None)}
        objs = _apply_wrappers(objs, spec["wrappers"])
        problem, model, disc, noise = objs["problem"], objs["model"], objs["disc"], objs["noise"]
        extra = spec["metrics"]["extra"]
        if "peak_xdot" in extra:
            model = PeakXdotProbe(model)
        run.update({"problem_name": getattr(problem, "name", None),
                    "problem_family": getattr(problem, "family", None),
                    "problem_n_x": getattr(problem, "n_x", None), "problem_m": getattr(problem, "m", None),
                    "problem_describe": _describe(problem),
                    "model_name": getattr(model, "name", None), "model_family": getattr(model, "family", None),
                    "model_uses_dEdt": bool(getattr(model, "uses_dEdt", True)),
                    "model_supports_implicit": bool(getattr(model, "supports_implicit", False)),
                    "model_requires": canonical_json(sorted(getattr(model, "requires", set()) or [])),
                    "model_describe": _describe(model),
                    "disc_name": getattr(disc, "name", None), "disc_causal": bool(getattr(disc, "causal", True)),
                    "disc_expected_order": getattr(disc, "expected_order", None),
                    "disc_implicit": bool(getattr(disc, "implicit", False)),
                    "disc_describe": _describe(disc)})
        if spec["T"] is None:
            base = {**base, "T": float(problem.T)}

        # ---------------- compatibility checks -> recorded skips ---------------
        req = set(getattr(model, "requires", set()) or [])
        caps = set(problem.capabilities() or [])
        if not req <= caps:
            run["status"] = "skipped"
            run["message"] = (f"incompatible: model {spec['model']['key']!r} requires {sorted(req - caps)} "
                              f"not in problem capabilities {sorted(caps)}")
        elif (spec.get("skip_unsupported_implicit", True) and getattr(disc, "implicit", False)
              and not getattr(model, "supports_implicit", False)):
            run["status"] = "skipped"
            run["message"] = (f"incompatible: discretization {spec['disc']['key']!r} is implicit but model "
                              f"{spec['model']['key']!r} does not support implicit stepping")
        run["setup_wall"] = time.perf_counter() - t_setup

        if run["status"] is None:
            # ---------------- simulate ------------------------------------------
            if G == 1:
                gamma_arg, x0 = gammas[0], None
            else:   # same B initial states for every gain value (draw = simulate's own)
                x0 = np.tile(initial_state(problem, model, spec["seed"], B), (G, 1))
                gamma_arg = np.repeat(np.asarray(gammas, dtype=float), B)
            kw = dict(T=spec["T"], B=B * G, gamma=gamma_arg, x0=x0, seed=spec["seed"], noise=noise,
                      traj_ids=np.tile(np.arange(B), G),
                      info=spec["info"], data_noise=spec["data_noise"], record_every=spec["record_every"],
                      div_threshold=spec["div_threshold"],
                      record_solution_error=spec["record_solution_error"], max_wall=spec.get("max_wall"))
            # (1) the metric run: fully recorded, gives every metric
            with np.errstate(all="ignore"):
                sim = simulate(problem, model, disc, spec["tau"], **kw)
            peak = model._peak.copy() if isinstance(model, PeakXdotProbe) else None
            with np.errstate(all="ignore"):
                summ = summarize(sim, eps=tuple(spec["metrics"]["eps"]), frac=spec["metrics"]["frac"])
                for name in extra:
                    if name == "peak_xdot":
                        summ["peak_xdot"] = peak
                        continue
                    fn = EXTRA_METRICS.get(name) or _import_path(name)
                    val = fn(sim)
                    if isinstance(val, dict):
                        summ.update(val)
                    elif val is not None:
                        summ[name if name in EXTRA_METRICS else name.split(":")[-1]] = val
                # component diagnostics are read right after the metric run, before
                # timing runs could change any internal state
                for comp in (problem, disc, noise):
                    em = getattr(comp, "extra_metrics", None)
                    if callable(em):
                        summ.update(em(sim) or {})
            # (2) optional lean timing runs: identical dynamics (recording never feeds
            # back into the state and draws no random numbers), but only t_0 and t_N
            # are recorded, no solution error (x_star can cost as much as a step) and
            # no peak-xdot probe, so the wall time measures the algorithm itself
            steps = int(sim["steps"])
            walls = []
            lean_model = model._inner if isinstance(model, PeakXdotProbe) else model
            lean_kw = {**kw, "record_every": max(steps, 1), "record_solution_error": False,
                       "online_stats": False}
            for _ in range(int(spec.get("timing_repeats", 0) or 0)):
                nreset = getattr(noise, "reset", None)      # stateful noises restart too
                if callable(nreset):
                    nreset()
                try:
                    with np.errstate(all="ignore"):
                        walls.append(simulate(problem, lean_model, disc, spec["tau"], **lean_kw)["wall"])
                except TimeoutError:
                    break               # keep the timing runs that finished
            metrics_per_traj = {k: np.asarray(v).reshape(-1) for k, v in summ.items() if v is not None}
            div = np.asarray(sim["diverged"], dtype=bool)
            metrics_per_traj["diverged_at"] = np.asarray(sim["diverged_at"], dtype=float)
            status_per_traj = np.where(div, "diverged", "ok")
            steps_run = steps
            if div.all() and steps:
                steps_run = int(min(steps, max(1, round(np.nanmax(sim["diverged_at"]) / spec["tau"]))))
            mode = "lean" if walls else "recorded"
            walls = walls or [float(sim["wall"])]
            wall = float(np.median(walls))
            run.update({"status": "ok", "wall": wall, "wall_min": float(np.min(walls)),
                        "wall_recorded": float(sim["wall"]), "timing_mode": mode, "timing_runs": len(walls),
                        "steps": steps, "steps_run": steps_run, "wall_per_step": wall / max(steps_run, 1),
                        "n_diverged": int(div.sum()), "T": float(sim["T"])})
            if want_traces and spec.get("traces"):
                idx = _trace_indices(len(sim["t"]), spec["traces"])
                rid = spec["id"]
                traces[f"{rid}__t"] = np.asarray(sim["t"])[idx]
                traces[f"{rid}__residual"] = np.asarray(sim["residual"])[idx].astype(np.float32)
                if sim["sol_err"] is not None:
                    traces[f"{rid}__sol_err"] = np.asarray(sim["sol_err"])[idx].astype(np.float32)
                traces[f"{rid}__gamma"] = np.asarray(sim["gamma"], dtype=float)
    except TimeoutError as exc:
        run["status"] = "timeout"
        run["message"] = f"TimeoutError: {exc}"
        run["wall"] = time.perf_counter() - t_setup
    except Exception as exc:  # noqa: BLE001
        run["status"] = "error"
        tb = traceback.format_exc()
        run["message"] = f"{type(exc).__name__}: {exc}\n{tb[-2000:]}"
    run["peak_rss_mb"] = _peak_rss_mb()
    if run.get("T") is None or (isinstance(run.get("T"), float) and math.isnan(run["T"])):
        run.pop("T", None)

    rows = []
    spec_json = json.dumps(_plain(spec), sort_keys=True, allow_nan=True)
    for g in range(G):
        for b in range(B):
            i = g * B + b
            row = dict(base)
            row.update({"traj": i, "x0_index": b, "gain_index": g, "gain_value": vals[g],
                        "gamma": gammas[g], "h": hs[g]})
            row.update(run)
            if status_per_traj is not None:
                row["status"] = str(status_per_traj[i])
                for k, arr in metrics_per_traj.items():
                    v = arr[i] if arr.shape[0] == B * G else (arr[0] if arr.shape[0] == 1 else np.nan)
                    row[k] = _scalar(v)
            row["spec_json"] = spec_json
            rows.append(row)
    return rows, traces


def _sanitize(df):
    """Make every column parquet-safe: uniform-type object columns are kept,
    bool/None -> nullable boolean, numbers/None -> float, anything mixed -> JSON text."""
    import pandas as pd

    def missing(v):
        return v is None or (isinstance(v, float) and math.isnan(v))

    for c in list(df.columns):
        s = df[c]
        if s.dtype != object:
            continue
        vals = [v for v in s.tolist() if not missing(v)]
        types = {type(v) for v in vals}
        if not types or types <= {str}:
            continue
        if types <= {bool, np.bool_}:
            df[c] = pd.array([None if missing(v) else bool(v) for v in s.tolist()], dtype="boolean")
        elif all(issubclass(t, (int, float, np.integer, np.floating)) and not issubclass(t, (bool, np.bool_))
                 for t in types):
            df[c] = pd.to_numeric(s, errors="coerce").astype(float)
        else:
            df[c] = [None if missing(v) else (v if isinstance(v, str) else canonical_json(v))
                     for v in s.tolist()]
    return df


def _atomic_parquet(df, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:6]}")
    try:
        try:
            df.to_parquet(tmp, index=False)
        except Exception:  # noqa: BLE001  (last resort: stringify object columns)
            for c in df.columns:
                if df[c].dtype == object:
                    df[c] = [None if v is None else str(v) for v in df[c].tolist()]
            df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _atomic_npz(arrays: dict, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:6]}")
    try:
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **arrays)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def execute_chunk(chunk_file, force: bool = False, verbose: bool = True) -> Path:
    """Run every spec of a chunk file and write its result parquet atomically.

    The parquet (the memguard ``done`` file) only appears once all specs have
    been processed; traces (if any were requested) are written first, to the
    sibling ``.npz``.  If the parquet already exists the chunk is skipped
    unless ``force``.  Returns the parquet path."""
    import pandas as pd

    chunk_file = Path(chunk_file)
    ch = json.loads(chunk_file.read_text())
    if ch.get("format") != CHUNK_FORMAT:
        raise ValueError(f"{chunk_file}: not a {CHUNK_FORMAT} file")
    result = Path(ch["result"])
    if result.exists() and not force:
        if verbose:
            print(f"[runner] {ch['task_id']}: already done -> {result}", flush=True)
        return result
    t0 = time.perf_counter()
    rows, traces, cache = [], {}, OrderedDict()
    n = len(ch["specs"])
    for i, spec in enumerate(ch["specs"]):
        r, tr = run_spec(spec, problem_cache=cache)
        if tr:
            traces.update(tr)
            for row in r:
                row["traces_file"] = Path(ch["traces"]).name
        for row in r:
            row["chunk"] = ch["task_id"]
        rows.extend(r)
        if verbose:
            st = r[0]["status"] if r else "?"
            msg = (r[0].get("message") or "").splitlines()[0][:160] if r and r[0].get("message") else ""
            print(f"[runner] {ch['task_id']} [{i + 1}/{n}] {spec['id']} {spec['problem']['key']}/"
                  f"{spec['model']['key']}/{spec['disc']['key']} tau={spec['tau']:g} -> {st} "
                  f"wall={r[0].get('wall', float('nan')):.3g}s {msg}", flush=True)
    if traces:
        _atomic_npz(traces, Path(ch["traces"]))
    df = _sanitize(pd.DataFrame(rows))
    _atomic_parquet(df, result)
    if verbose:
        print(f"[runner] {ch['task_id']}: {n} specs, {len(df)} rows in {time.perf_counter() - t0:.2f}s "
              f"-> {result}", flush=True)
    return result


# =============================================================================
# aggregation
# =============================================================================
def result_files(result_dir, current_only: Optional[bool] = None) -> List[Path]:
    """Chunk result parquets under `result_dir`, oldest first.

    With ``current_only`` (default: whenever a ``tasks.jsonl`` exists under
    `result_dir`) only the done files listed in those ``tasks.jsonl`` are
    returned: results of earlier builds of a changed config (stale chunks,
    removed specs) and other parquet files such as an aggregated
    ``all.parquet`` written into the folder are ignored.  A done file whose
    absolute path no longer exists (moved folder) is looked up in the
    ``results/`` folder next to its tasks.jsonl.  With ``current_only=False``
    every ``*.parquet`` under `result_dir` (recursively) is returned."""
    root = Path(result_dir)
    task_files = sorted(root.rglob("tasks.jsonl")) if root.is_dir() else []
    if current_only is None:
        current_only = bool(task_files)
    files = []
    if current_only:
        for tf in task_files:
            for t in load_tasks(tf):
                p = Path(t["done"])
                if not p.exists():
                    p = tf.parent / "results" / p.name
                if p.exists():
                    files.append(p)
    elif root.is_dir():
        files = [f for f in root.rglob("*.parquet") if not f.name.startswith(".")]
    files = list({str(f.resolve()): f for f in files}.values())
    return sorted(files, key=lambda f: (f.stat().st_mtime, str(f)))


def aggregate(result_dir, columns: Optional[List[str]] = None, drop=("spec_json",),
              dedupe: bool = True, current_only: Optional[bool] = None):
    """Concatenate the chunk parquets under `result_dir` into one DataFrame,
    one row per trajectory.  Which files are read is decided by
    :func:`result_files` (by default only the current done files of every
    ``tasks.jsonl`` under `result_dir`; every ``*.parquet`` if there is none
    or with ``current_only=False``).  ``drop`` columns are not loaded;
    ``columns`` restricts loading to the given columns.  With ``dedupe``,
    rows with the same (config, run_id, traj) keep the newest file's copy."""
    import pandas as pd
    import pyarrow.parquet as pq

    files = result_files(result_dir, current_only)
    frames = []
    for f in files:
        names = pq.read_schema(f).names
        cols = [c for c in (columns or names) if c in names and c not in (drop or ())]
        if dedupe and columns is not None:
            cols += [k for k in ("config", "run_id", "traj") if k in names and k not in cols]
        frames.append(pd.read_parquet(f, columns=cols))
    frames = [f for f in frames if len(f.columns)]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True, sort=False)
    if dedupe and {"config", "run_id", "traj"} <= set(df.columns):
        df = df.drop_duplicates(subset=["config", "run_id", "traj"], keep="last").reset_index(drop=True)
    for c in _BOOL_COLUMNS:          # bool + missing (failed/skipped rows) -> nullable boolean
        if c in df.columns and df[c].dtype != bool:
            try:
                df[c] = df[c].astype("boolean")
            except (TypeError, ValueError):
                pass
    if columns is not None:
        df = df[[c for c in columns if c in df.columns]]
    return df


_BOOL_COLUMNS = ("diverged", "model_uses_dEdt", "model_supports_implicit", "disc_causal", "disc_implicit")


def load_traces(npz_path, run_id_: str) -> dict:
    """Down-sampled traces of one run: t (N_p,), residual and sol_err (N_p, B*G), gamma (B*G,)."""
    with np.load(npz_path) as z:
        pre = f"{run_id_}__"
        return {k[len(pre):]: z[k] for k in z.files if k.startswith(pre)}


# =============================================================================
# command line
# =============================================================================
def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="python -m znnbench.runner",
                                 description="ZNN-Bench sweep runner (build / local / status / aggregate)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="expand a config and write chunks + tasks.jsonl")
    b.add_argument("config")
    b.add_argument("--out", required=True, help="output dir (one per config)")
    b.add_argument("--chunk-size", type=int, default=None)
    b.add_argument("--balance", action="store_true")
    b.add_argument("--no-validate", action="store_true")
    lo = sub.add_parser("local", help="run all pending chunks in this process (small sweeps only)")
    lo.add_argument("out")
    lo.add_argument("--force", action="store_true")
    s = sub.add_parser("status")
    s.add_argument("out")
    a = sub.add_parser("aggregate")
    a.add_argument("result_dir")
    a.add_argument("--out", required=True, help="output .parquet or .csv")
    a.add_argument("--all", action="store_true",
                   help="read every *.parquet under result_dir, not only the done files of its tasks.jsonl")
    args = ap.parse_args(argv)
    if args.cmd == "build":
        cfgs = load_configs(args.config)
        for c in cfgs:
            out = Path(args.out) if len(cfgs) == 1 else Path(args.out) / normalize_config(c)["name"]
            tf = build(c, out, chunk_size=args.chunk_size, balance=args.balance or None,
                       validate=not args.no_validate)
            print(f"{c.get('name')}: {len(load_tasks(tf))} tasks -> {tf}")
            print(f"launch: {DEFAULT_PYTHON} {MEMGUARD} pool --tasks {tf} --workers 16 --max-gb 200")
    elif args.cmd == "local":
        run_local(args.out, force=args.force)
    elif args.cmd == "status":
        print(json.dumps(status(args.out)))
    elif args.cmd == "aggregate":
        df = aggregate(args.result_dir, current_only=False if args.all else None)
        if args.out.endswith(".csv"):
            df.to_csv(args.out, index=False)
        else:
            _atomic_parquet(_sanitize(df), Path(args.out))   # chunks may disagree on column types
        print(f"{len(df)} rows -> {args.out}")


if __name__ == "__main__":
    main()
