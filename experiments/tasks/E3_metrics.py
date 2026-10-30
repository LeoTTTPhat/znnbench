"""E3 extra metrics for the ZNN-Bench runner (``metrics.extra`` import paths).

Experiment-side helper (nothing under ``znnbench/`` is modified).  The runner
calls ``fn(sim)`` after the metric run of every spec; a dict return value adds
one column per key (shape (B*G,) arrays).  All quantities are computed on the
*recorded* samples (E3 records every step, ``record_every: 1``).

``first_hit``
    ``t_hit_<eps>``  first sampling instant with ||E|| < eps (inf if never).
                     Contrast with the core ``t_eps_<eps>`` (first instant
                     after which ||E|| STAYS below eps): t_hit <= t_eps, and
                     t_hit < t_eps = inf identifies a chattering band that
                     dips below eps without staying there.
    ``res_min``      min_k ||E_k|| over the whole run.
    ``res_min_ss``   min_k ||E_k|| over the steady window (last 25 %, as MSSRE).

``tc_window_<tag>`` (one function per predefined time t_c; ``<tag>`` is t_c
with '.' written as 'p', e.g. ``tc_window_0p03`` for t_c = 0.03 s; resolved by
the module ``__getattr__`` below, so any t_c works)
    ``tc_rmin_pre``  min ||E_k|| over t_k < t_c (the best the pre-t_c phase got);
    ``tc_rpeak_post`` max ||E_k|| over t_k >= t_c (inf if diverged);
    ``tc_amp``       tc_rpeak_post / tc_rmin_pre: re-amplification of the
                     residual around the predefined time (the 1/(t_c - t)
                     gain makes the Euler step h(t) = gamma tau/(t_c - t_k)
                     exceed 2 in the last ~gamma/2 steps before t_c);
    ``tc_r_at``      ||E|| at the first sample t_k >= t_c (the residual the
                     strictly-predefined-time claim says is exactly 0).
"""
from __future__ import annotations

import numpy as np

EPS = (1e-2, 1e-3, 1e-6)
FRAC = 0.25


def _res(sim):
    r = np.asarray(sim["residual"], dtype=float)
    return np.where(np.isnan(r), np.inf, r)


def first_hit(sim):
    t = np.asarray(sim["t"], dtype=float)
    r = _res(sim)
    out = {}
    for e in EPS:
        below = r < e
        hit = below.any(axis=0)
        idx = np.argmax(below, axis=0)
        out[f"t_hit_{e:g}"] = np.where(hit, t[idx], np.inf)
    out["res_min"] = np.min(r, axis=0)
    w = t >= t[-1] - FRAC * (t[-1] - t[0])
    out["res_min_ss"] = np.min(r[w], axis=0)
    return out


def _tc_window(t_c):
    t_c = float(t_c)
    guard = 1e-12 * max(1.0, t_c)        # same guard as models/gains.py and variants.SPTCNTZNN

    def tc_window(sim):
        t = np.asarray(sim["t"], dtype=float)
        r = _res(sim)
        pre = t < t_c - guard
        post = ~pre
        B = r.shape[1]
        nan = np.full(B, np.nan)
        rmin = np.min(r[pre], axis=0) if pre.any() else nan
        rpeak = np.max(r[post], axis=0) if post.any() else nan
        r_at = r[np.argmax(post)] if post.any() else nan
        with np.errstate(divide="ignore", invalid="ignore"):
            amp = rpeak / rmin
        return {"tc_rmin_pre": rmin, "tc_rpeak_post": rpeak, "tc_amp": amp, "tc_r_at": r_at,
                "tc_value": np.full(B, t_c)}

    tc_window.__name__ = f"tc_window_{tag_of(t_c)}"
    return tc_window


def tag_of(t_c) -> str:
    """t_c -> attribute tag ('0.03' -> '0p03'); inverse of the __getattr__ parser."""
    return f"{float(t_c):g}".replace(".", "p").replace("-", "m")


def __getattr__(name):          # PEP 562: tc_window_<tag> for any t_c
    if name.startswith("tc_window_"):
        tag = name[len("tc_window_"):].replace("p", ".").replace("m", "-")
        try:
            return _tc_window(float(tag))
        except ValueError:
            pass
    raise AttributeError(name)
