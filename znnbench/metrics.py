"""Standard ZNN-Bench metrics computed from simulation traces."""
from __future__ import annotations

import numpy as np


def steady_window(t, frac=0.25):
    """Boolean mask of the last `frac` of the horizon."""
    return t >= t[-1] - frac * (t[-1] - t[0])


def mssre(sim, frac=0.25):
    """Maximal steady-state residual error per trajectory: max ||E|| over the
    last `frac` of the horizon (inf if the trajectory diverged)."""
    w = steady_window(sim["t"], frac)
    return np.max(sim["residual"][w], axis=0)


def mean_ssre(sim, frac=0.25):
    w = steady_window(sim["t"], frac)
    return np.mean(sim["residual"][w], axis=0)


def steady_solution_error(sim, frac=0.25):
    if sim["sol_err"] is None:
        return None
    w = steady_window(sim["t"], frac)
    return np.max(sim["sol_err"][w], axis=0)


def time_to_eps(sim, eps, key="residual"):
    """First time after which the trace stays below eps (inf if never)."""
    tr = sim[key]
    t = sim["t"]
    below = tr < eps
    out = np.full(tr.shape[1], np.inf)
    for b in range(tr.shape[1]):
        bad = np.where(~below[:, b])[0]
        if len(bad) == 0:
            out[b] = t[0]
        elif bad[-1] < len(t) - 1:
            out[b] = t[bad[-1] + 1]
    return out


def chattering(sim, frac=0.25):
    # NOTE: based on ||E|| differences, it is blind to symmetric e -> -e cycles;
    # simulate's online 'chatter_vec' (vector differences of E) is not.
    """Chattering index over the steady window: mean |r_{k+1} - r_k| divided
    by mean r_k (residual sampled every step).  Smooth tracking gives values
    of order tau * (problem frequency) << 1; sign-type switching between
    samples gives values of order 1."""
    w = steady_window(sim["t"], frac)
    r = sim["residual"][w]
    if r.shape[0] < 3:
        return np.full(r.shape[1], np.nan)
    num = np.mean(np.abs(np.diff(r, axis=0)), axis=0)
    den = np.mean(r, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(np.isfinite(den) & (den > 0), num / den, np.nan)


def empirical_order(taus, errors, floor=1e-12):
    """Least-squares slope of log(error) vs log(tau), using only points above
    `floor` (round-off) and finite values.  Returns (slope, n_points_used)."""
    taus, errors = np.asarray(taus, float), np.asarray(errors, float)
    ok = np.isfinite(errors) & (errors > floor)
    if ok.sum() < 2:
        return np.nan, int(ok.sum())
    p = np.polyfit(np.log(taus[ok]), np.log(errors[ok]), 1)
    return float(p[0]), int(ok.sum())


def summarize(sim, eps=(1e-3, 1e-6), frac=0.25):
    out = {"mssre": mssre(sim, frac), "mean_ssre": mean_ssre(sim, frac),
           "chatter": chattering(sim, frac), "diverged": sim["diverged"]}
    se = steady_solution_error(sim, frac)
    if se is not None:
        out["ss_sol_err"] = se
    for e in eps:
        out[f"t_eps_{e:g}"] = time_to_eps(sim, e)
    # online statistics recorded by simulate (steady window = last stats_frac)
    for k in ("chatter_vec", "rate_alt", "peak_rate", "peak_rate_ss"):
        if k in sim:
            out[k] = sim[k]
    if "x_final" in sim and "x_init" in sim:
        out["drift"] = np.linalg.norm(sim["x_final"] - sim["x_init"], axis=-1)
    return out
