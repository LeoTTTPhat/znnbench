"""Name -> factory registries.  Each zoo module registers its factories with
the decorators below; the subpackages auto-import every module they contain,
so adding a component never requires editing a shared file."""
from __future__ import annotations

import importlib
import os
import pkgutil
import warnings

PROBLEMS, MODELS, DISCRETIZATIONS, NOISES = {}, {}, {}, {}


def _reg(table, key):
    def deco(fn):
        if key in table:
            raise KeyError(f"duplicate registry key {key!r}")
        table[key] = fn
        return fn
    return deco


def register_problem(key):
    return _reg(PROBLEMS, key)


def register_model(key):
    return _reg(MODELS, key)


def register_discretization(key):
    return _reg(DISCRETIZATIONS, key)


def register_noise(key):
    return _reg(NOISES, key)


IMPORT_ERRORS = {}


def autoimport(pkg_name, pkg_path):
    """Import every public module of a subpackage.  A broken module does not
    break its siblings (it is reported in IMPORT_ERRORS and warned about);
    set ZNNBENCH_STRICT=1 to raise instead."""
    for m in pkgutil.iter_modules(pkg_path):
        if m.name.startswith("_"):
            continue
        name = f"{pkg_name}.{m.name}"
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001
            if os.environ.get("ZNNBENCH_STRICT") == "1":
                raise
            IMPORT_ERRORS[name] = repr(exc)
            warnings.warn(f"znnbench: could not import {name}: {exc!r}")


def _load_all():
    import znnbench.discretizations  # noqa: F401
    import znnbench.models  # noqa: F401
    import znnbench.noise  # noqa: F401
    import znnbench.problems  # noqa: F401


def make_problem(key, **kw):
    _load_all()
    return PROBLEMS[key](**kw)


def make_model(key, **kw):
    _load_all()
    return MODELS[key](**kw)


def make_discretization(key, **kw):
    _load_all()
    return DISCRETIZATIONS[key](**kw)


def make_noise(key, **kw):
    _load_all()
    return NOISES[key](**kw)


def available():
    _load_all()
    return {"problems": sorted(PROBLEMS), "models": sorted(MODELS),
            "discretizations": sorted(DISCRETIZATIONS), "noises": sorted(NOISES)}
