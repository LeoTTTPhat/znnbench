#!/usr/bin/env python
"""E5 analysis -- noise robustness under discretization.

    python experiments/analyze_E5.py --results results/E5 \
        --figdir figures --tabdir results/tables

Reads the aggregated runner output of experiments/configs/E5.py and writes

* figures (style.savefig -> <figdir>/fig_E5_*.pdf + .png):
  - fig_E5_floor_heatmap[_<variant>]  model x noise heatmap of the median
    residual floor (log10 MSSRE, sequential blue), panels = disc x tau;
  - fig_E5_penalty_heatmap            log10(floor / noise-free floor) (diverging);
  - fig_E5_ct_gap_heatmap             log10(measured / (CT prediction +
    noise-free floor)) (diverging): where CT theory plus the discretization
    floor does / does not explain the discrete floor;
  - fig_E5_floor_vs_tau[_gamma]       floor vs tau for selected (model, noise)
    cells with the CT prediction and the noise-free floor;
  - fig_E5_ct_vs_measured             measured vs CT-predicted floor, one panel
    per model;
* LaTeX booktabs tables <tabdir>/E5_*.tex;
* <results>/summary.json and <results>/SUMMARY.md (headline numbers).

Floors.  Per trajectory the floor is the runner's MSSRE (max ||E|| over the
last 25 % of T; inf if the trajectory diverged).  Cells (problem, start,
model, noise, disc, gain arm, tau) are summarised by the median and IQR over
their 3 seeds x B = 4 trajectories (diverged trajectories count as +inf, so a
cell whose median is inf diverged in >= 50 % of its trajectories).  For the
stochastic noises (gaussian, random_walk) the CT comparison uses the window
MEAN ||E|| (mean_ssre), because only moments are predicted.

Continuous-time (CT) predictions: noise-only response of each model's error
dynamics dE/dt = model(E) + nu(t) from E(0) = 0 (all entries of nu equal, so
E = e 1_m and ||E|| = sqrt(m)|e|), evaluated on the same steady window
[0.75 T, T]:
  linear models (znn, ieznn, dieznn): exact integration (Radau, analytic
    Jacobian) for deterministic noise; Lyapunov ODE for the second moment
    under stochastic noise (per-step Gaussian of std s is taken as its CT
    white-noise equivalent, intensity s^2 tau; random walk = integrated white
    noise of intensity sigma^2);
  znn_sbp / hntznn (Li activation): numerical integration (LSODA) for
    deterministic noise; znn_sbp + gaussian: stationary density of the scalar
    SDE; znn_sbp + random walk: quasi-static e = Phi^-1(nu/gamma); hntznn +
    stochastic noise: not predicted (NaN);
  sptc_an_foznn: Theorem-2 sliding mode -> 0 whenever |nu_i| < zeta/sqrt(m) on
    the window (bound checked at 4 sigma / 3 sigma sqrt(T) for the stochastic
    noises); beyond the bound (linear ramp) the quasi-static root of the
    post-t_c error dynamics.
Discrete-time (DT) predictions for the linear models: the same noise-only
response through the exact discrete error recursion of the look-ahead formula
(aux states by forward Euler, as in the zoo), i.e. what the static control
problem should reproduce.

Statistics: medians + IQR (diverged = +inf = worst, never dropped); 95 % CIs
for the headline ratios and floor slopes by a SEED-CLUSTER bootstrap (2000
resamples, seeded; whole problem seeds resampled with all their paired
trajectories, because the B trajectories of a seed share the instance and are
pseudo-replicates for deterministic noise; no CI with < 3 seeds); Kendall
tau_b between the CT-predicted and the measured model ranking (descriptive);
floor-vs-tau slopes by least squares on log-log axes over the asymptotic range
(round-off, transient-limited and pre-asymptotic points dropped), reported
with the all-points slope and the consecutive (local) slopes.

Review additions (2026-10-18): transient diagnostics for the linear models
(is the value at T = 6 s a floor?), the linear-activation HNTZNN harmonic arm
(matched / mismatched internal model, exact CT and DT predictions), the paired
unit check of the power-type activations, censoring and confound summaries.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize, TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from experiments import keys, style  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# protocol constants (single source of truth: experiments/configs/E5.py)
# ---------------------------------------------------------------------------
_spec = importlib.util.spec_from_file_location("_e5_cfg", ROOT / "experiments" / "configs" / "E5.py")
CFG = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CFG)

NOISE_LEVELS = CFG.NOISE_LEVELS
T_H = CFG.T_HORIZON
FRAC = CFG.FRAC
MODELS_ALL = [lbl for lbl, _, _ in CFG.MODEL_SPECS]
HARMONIC_ONLY = list(getattr(CFG, "HARMONIC_ONLY", []))
MODELS = [m for m in MODELS_ALL if m not in HARMONIC_ONLY]      # full noise grid (heatmaps, rankings)
LINEAR_MODELS = ["znn", "ieznn", "dieznn", "hntznn_lin", "hntznn_lin_mis"]
NOISES = ["none"] + list(CFG.NOISE_ORDER)
STOCHASTIC = {"gaussian", "random_walk"}
ROUNDOFF = 1e-12
BOOT = 2000
BOOT_SEED = 20261018
MIN_CLUSTERS = 3          # seed-cluster bootstrap CIs need >= 3 problem seeds (else reported as n/a)
MODEL_TEX = {"znn": "ZNN", "ieznn": "IEZNN", "dieznn": "DIEZNN", "hntznn": "HNTZNN",
             "sptc_an_foznn": "SPTC-AN-FOZNN", "znn_sbp": "ZNN-SBP",
             "hntznn_lin": "HNTZNN-lin", "hntznn_lin_mis": "HNTZNN-lin (mismatched)"}
HNT_OMEGA = {"hntznn_lin": CFG.HARMONIC_OMEGA,
             "hntznn_lin_mis": getattr(CFG, "HARMONIC_MISMATCH", 1.1) * CFG.HARMONIC_OMEGA}
UNIT_SCALE = float(getattr(CFG, "UNIT_SCALE", 1000.0))
NOISE_TEX = {"none": "none", "constant": "constant", "linear": "ramp", "gaussian": "Gaussian",
             "harmonic": "harmonic", "random_walk": "rand. walk"}
DISC_TEX = {"euler": "Euler", "taylor_zhang": "Taylor--Zhang"}          # LaTeX tables
DISC_FIG = {"euler": "Euler", "taylor_zhang": "Taylor\u2013Zhang"}       # figures (en dash)
PROBLEM_FIG = {"matrix_inversion": "matrix inversion ($n=4$)", "linear_system": "linear system ($n=4$)",
               "linear_system_static": "static linear system ($n=4$, control)"}
CT_ZERO = 1e-13           # CT predictions below this are exact rejection + round-off -> 0


def arm_label(mode, val):
    return f"h{val:g}" if mode == "fixed_h" else f"g{val:g}"


def arm_tex(arm):
    return f"$h={arm[1:]}$" if arm.startswith("h") else f"$\\gamma={arm[1:]}$"


def _fmt_tau(t):
    e = int(round(math.log10(t)))
    return f"$10^{{{e}}}$" if abs(t - 10.0 ** e) < 1e-12 * t else f"{t:g}"


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
COLS = ["config", "run_id", "problem_label", "problem_key", "model_label", "model_key", "disc_key",
        "noise_key", "wrapper_keys", "tau", "gain_mode", "gain_value", "gamma", "h", "seed",
        "x0_index", "gain_index", "traj", "status", "message", "mssre", "mean_ssre", "chatter_vec",
        "rate_alt", "initial_residual", "final_residual", "max_residual", "diverged_at",
        "problem_m", "wall", "wall_per_step", "steps", "steps_run", "peak_rss_mb", "B",
        "model_describe", "disc_describe", "T"]


def load(results):
    from znnbench.runner import aggregate
    df = aggregate(results)
    if df.empty:
        raise SystemExit(f"no results under {results}")
    df = df[[c for c in COLS if c in df.columns]].copy()
    wk = df["wrapper_keys"].astype(str)
    # start arms: rand (PLAN), near (radius 0.1) and the paired unit check (radius 1e-4,
    # E x 1 / E x UNIT_SCALE; review 2026-10-18)
    df["start"] = np.select([wk.str.contains("UnitPairStart") & wk.str.contains("ScaledError"),
                             wk.str.contains("UnitPairStart"), wk.str.contains("NearSolutionStart")],
                            ["unit_Ex1e3", "unit_Ex1", "near"], default="rand")
    df["noise"] = df["noise_key"].fillna("none").astype(str)
    df["model"] = df["model_label"].astype(str)
    df["problem"] = df["problem_label"].astype(str)
    df["disc"] = df["disc_key"].astype(str)
    df["arm"] = [arm_label(m, v) for m, v in zip(df["gain_mode"], df["gain_value"])]
    fl = df["mssre"].astype(float).to_numpy().copy()
    fl[df["status"].to_numpy() == "diverged"] = np.inf
    fl[~np.isin(df["status"].to_numpy(), ["ok", "diverged"])] = np.nan
    df["floor"] = fl
    mn = df["mean_ssre"].astype(float).to_numpy().copy()
    mn[df["status"].to_numpy() == "diverged"] = np.inf
    mn[~np.isin(df["status"].to_numpy(), ["ok", "diverged"])] = np.nan
    df["floor_mean"] = mn
    df["pair"] = (df["seed"].astype(int).astype(str) + ":" + df["x0_index"].astype(int).astype(str))
    return df


# ---------------------------------------------------------------------------
# robust quantiles with +inf (diverged trajectories)
# ---------------------------------------------------------------------------
BIG = 1e300


def q(x, p):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan
    v = np.quantile(np.where(np.isinf(x), BIG, x), p)
    return np.inf if v >= BIG / 10 else float(v)


CELL = ["problem", "start", "model", "noise", "disc", "arm", "tau"]


def cell_table(df):
    g = df.groupby(CELL, sort=False)
    out = g.agg(n=("floor", "size"), n_div=("status", lambda s: int((s == "diverged").sum())),
                n_bad=("status", lambda s: int((~s.isin(["ok", "diverged"])).sum())),
                gamma=("gamma", "first"), h=("h", "first"), m=("problem_m", "first")).reset_index()
    med, q1, q3, medm, q1m, q3m, e0 = [], [], [], [], [], [], []
    for _, sub in g:
        f = sub["floor"].to_numpy()
        fm = sub["floor_mean"].to_numpy()
        med.append(q(f, 0.5)); q1.append(q(f, 0.25)); q3.append(q(f, 0.75))
        medm.append(q(fm, 0.5)); q1m.append(q(fm, 0.25)); q3m.append(q(fm, 0.75))
        e0.append(q(sub["initial_residual"].astype(float).to_numpy(), 0.5)
                  if "initial_residual" in sub else np.nan)
    out["med"], out["q1"], out["q3"] = med, q1, q3
    out["med_mean"], out["q1_mean"], out["q3_mean"] = medm, q1m, q3m
    out["init_med"] = e0                      # median ||E(0)|| (for the transient diagnostic)
    out["frac_div"] = out["n_div"] / out["n"].clip(lower=1)
    return out


# ---------------------------------------------------------------------------
# noise signals (per entry) -- identical formulas to znnbench/noise/*
# ---------------------------------------------------------------------------
def nu_fn(noise):
    p = NOISE_LEVELS.get(noise, {})
    if noise == "constant":
        return lambda t: p["level"] + 0.0 * np.asarray(t, dtype=float)
    if noise == "linear":
        return lambda t: p["offset"] + p["slope"] * np.asarray(t, dtype=float)
    if noise == "harmonic":
        return lambda t: p["amp"] * np.sin(p["omega"] * np.asarray(t, dtype=float) + p["phase"])
    return lambda t: 0.0 * np.asarray(t, dtype=float)


def chi_mean(m):
    """E||z|| for z ~ N(0, I_m)."""
    return math.sqrt(2.0) * math.exp(math.lgamma((m + 1) / 2.0) - math.lgamma(m / 2.0))


def _spow(e, p):
    return np.sign(e) * np.abs(e) ** p


def _window():
    return np.linspace((1.0 - FRAC) * T_H, T_H, 1201)


def linear_AB(model, gamma):
    """CT error dynamics y' = A y + b nu of the linear models (per entry)."""
    g = float(gamma)
    if model == "znn":
        return np.array([[-g]]), np.array([1.0])
    if model == "ieznn":
        lam = 0.25 * g * g                      # lam_ratio = 1/4 (zoo default)
        return np.array([[-g, -lam], [1.0, 0.0]]), np.array([1.0, 0.0])
    if model == "dieznn":
        xi = g
        return (np.array([[-(2 * xi + 1), -(xi * xi + 2 * xi), -xi * xi], [1.0, 0, 0], [0, 1.0, 0]]),
                np.array([1.0, 0, 0]))
    if model in HNT_OMEGA:                      # HNTZNN, linear activation: y = [e, X, C], lam = 1
        w2, lam = HNT_OMEGA[model] ** 2, 1.0
        return np.array([[-g, -1.0, 0.0], [w2 * lam, 0.0, 1.0], [0.0, -w2, 0.0]]), np.array([1.0, 0, 0])
    return None, None


def sptc_params():
    delta = CFG.SPTC_DELTA
    zeta = 5.0 * delta
    t_c, alpha, t_floor = 1.0, 0.5, 1e-3
    t_p = (1.0 - math.exp(-math.pi / (2.0 * math.sqrt(zeta * (zeta - delta))))) * t_c
    return dict(zeta=zeta, t_c=t_c, alpha=alpha, t_p=t_p, t_floor=t_floor)


def _noise_generator(noise):
    """Autonomous generator of the deterministic noise: nu = first state."""
    p = NOISE_LEVELS[noise]
    if noise == "constant":
        return np.zeros((1, 1)), np.array([p["level"]])
    if noise == "linear":
        return np.array([[0.0, 1.0], [0.0, 0.0]]), np.array([p["offset"], p["slope"]])
    if noise == "harmonic":
        w = p["omega"]
        return (np.array([[0.0, 1.0], [-w * w, 0.0]]),
                np.array([p["amp"] * math.sin(p["phase"]), p["amp"] * w * math.cos(p["phase"])]))
    raise ValueError(noise)


def _linear_det_response(A, b, noise, tw):
    """Exact e(t) on the grid tw of y' = A y + b nu, nu from its generator, y(0) = 0."""
    from scipy.linalg import expm
    Gm, g0 = _noise_generator(noise)
    n, k = A.shape[0], Gm.shape[0]
    M = np.zeros((n + k, n + k))
    M[:n, :n] = A
    M[:n, n] = b
    M[n:, n:] = Gm
    x = np.concatenate([np.zeros(n), g0])
    # propagate to tw[0] in moderate sub-steps (keeps expm well scaled), then on the grid
    t0, nsub = float(tw[0]), max(1, int(math.ceil(float(tw[0]) / 0.05)))
    E1 = expm(M * (t0 / nsub))
    for _ in range(nsub):
        x = E1 @ x
    dt = float(tw[1] - tw[0])
    E2 = expm(M * dt)
    out = np.empty(len(tw))
    for i in range(len(tw)):
        out[i] = x[0]
        x = E2 @ x
    return out


def _linear_var_response(A, b, noise, tw, tau):
    """Exact Var e(t) on tw: per-step Gaussian = white noise of intensity std^2 tau
    entering e; random walk = integrated white noise of intensity sigma^2."""
    from scipy.linalg import expm
    p = NOISE_LEVELS[noise]
    n = A.shape[0]
    if noise == "gaussian":
        Aa, G, qn = A, b, p["std"] ** 2 * tau
    else:
        Aa = np.zeros((n + 1, n + 1)); Aa[:n, :n] = A; Aa[:n, n] = b
        G = np.zeros(n + 1); G[n] = 1.0
        qn = p["sigma"] ** 2
    N = Aa.shape[0]
    Q = qn * np.outer(G, G)
    dt = float(tw[1] - tw[0])
    nsteps0 = int(round(float(tw[0]) / dt))

    def step_mats(h):
        F = expm(np.block([[-Aa, Q], [np.zeros((N, N)), Aa.T]]) * h)
        Phi = F[N:, N:].T
        return Phi, Phi @ F[:N, N:]
    Phi, Qd = step_mats(dt)
    P = np.zeros((N, N))
    for _ in range(nsteps0):
        P = Phi @ P @ Phi.T + Qd
    out = np.empty(len(tw))
    for i in range(len(tw)):
        out[i] = max(P[0, 0], 0.0)
        P = Phi @ P @ Phi.T + Qd
    return out


@lru_cache(maxsize=None)
def ct_prediction(model, noise, gamma, m, tau):
    """(pred_max, pred_mean, method) of the CT noise-only response on the window."""
    from scipy.integrate import solve_ivp
    if noise == "none":
        return 0.0, 0.0, "zero"
    tw = _window()
    sm = math.sqrt(m)
    p = NOISE_LEVELS[noise]
    A, b = linear_AB(model, gamma)
    if A is not None and noise not in STOCHASTIC:
        e = np.abs(_linear_det_response(A, b, noise, tw))
        return sm * float(e.max()), sm * float(e.mean()), "CT exact (matrix exponential)"
    if A is not None:                            # stochastic: exact 2nd-moment recursion (Van Loan)
        var = _linear_var_response(A, b, noise, tw, tau)
        return np.nan, chi_mean(m) * float(np.sqrt(var).mean()), "CT exact 2nd moment (Van Loan)"
    if model in ("znn_sbp", "hntznn"):
        k = 0.5 if model == "znn_sbp" else 1.0   # sign_bi_power = Li / 2
        if noise not in STOCHASTIC:
            f = nu_fn(noise)
            g = float(gamma)
            if model == "znn_sbp":
                def rhs(t, y):
                    return [-g * k * (_spow(y[0], 0.5) + _spow(y[0], 2.0)) + f(t)]
                y0 = [0.0]
            else:
                w2, lam = CFG.HARMONIC_OMEGA ** 2, 1.0

                def rhs(t, y):
                    e, X, C = y
                    return [-g * (_spow(e, 0.5) + _spow(e, 2.0)) - X + f(t), C + w2 * lam * e, -w2 * X]
                y0 = [0.0, 0.0, 0.0]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sol = solve_ivp(rhs, (0.0, T_H), y0, method="LSODA", t_eval=tw, rtol=1e-9, atol=1e-14,
                                max_step=0.01)
                if not sol.success or sol.y.shape[1] != tw.size:      # fallback: implicit BDF
                    sol = solve_ivp(rhs, (0.0, T_H), y0, method="BDF", t_eval=tw, rtol=1e-8, atol=1e-13,
                                    max_step=0.01)
            if not sol.success or sol.y.shape[1] != tw.size:
                return np.nan, np.nan, "CT numerical (failed)"
            e = np.abs(sol.y[0])
            return sm * float(e.max()), sm * float(e.mean()), "CT numerical (LSODA)"
        if model == "znn_sbp":
            from scipy.integrate import quad
            g = float(gamma)
            if noise == "gaussian":
                qn = p["std"] ** 2 * tau
                U = lambda e: 0.5 * (abs(e) ** 1.5 / 1.5 + abs(e) ** 3 / 3.0)  # noqa: E731
                sc = max((qn / g) ** (2.0 / 3.0), 1e-300)
                L = 60.0 * sc
                w = lambda e: math.exp(-2.0 * g / qn * U(e))  # noqa: E731
                Z = quad(w, 0, L, points=[sc, 5 * sc], limit=200)[0]
                M2 = quad(lambda e: e * e * w(e), 0, L, points=[sc, 5 * sc], limit=200)[0]
                return np.nan, chi_mean(m) * math.sqrt(M2 / Z), "CT stationary density (SDE)"
            # random walk: quasi-static e = Phi^-1(nu/gamma), nu ~ N(0, sigma^2 t) per entry
            xs, ws = np.polynomial.hermite_e.hermegauss(80)
            ws = ws / ws.sum()
            vals = []
            for t in tw[::40]:
                y = np.abs(p["sigma"] * math.sqrt(t) * xs / g)
                u = np.minimum(2 * y, (2 * y) ** 0.25)
                for _ in range(60):                    # 0.5 (u + u^4) = y, u = sqrt|e|
                    u = u - (0.5 * (u + u ** 4) - y) / (0.5 + 2 * u ** 3)
                    u = np.maximum(u, 0.0)
                vals.append(math.sqrt(float(np.sum(ws * u ** 4))))
            return np.nan, chi_mean(m) * float(np.mean(vals)), "CT quasi-static"
        return np.nan, np.nan, "not predicted"
    if model == "sptc_an_foznn":
        sp = sptc_params()
        thr = sp["zeta"] / sm
        if noise == "gaussian":
            ok = 4.0 * p["std"] < thr
            return (0.0, 0.0, "CT theory (sliding, ||nu||<zeta)") if ok else (np.nan, np.nan, "no CT guarantee")
        if noise == "random_walk":
            ok = 3.0 * p["sigma"] * math.sqrt(T_H) < thr
            return (0.0, 0.0, "CT theory (sliding, ||nu||<zeta)") if ok else (np.nan, np.nan, "no CT guarantee")
        nu = np.abs(nu_fn(noise)(tw))
        e = np.zeros_like(tw)
        over = nu >= thr
        if over.any():
            t = tw[over]
            a = float(gamma) * np.maximum(t, sp["t_floor"]) ** (sp["alpha"] - 1.0) * sm / (t - sp["t_p"]) ** 2
            bq = 1.0 / (t - sp["t_p"])
            c = nu[over] - thr
            e[over] = (bq + np.sqrt(bq * bq + 4 * a * c)) / (2 * a)
        meth = "CT theory (sliding, ||nu||<zeta)" if not over.any() else "CT quasi-static (bound exceeded)"
        return sm * float(e.max()), sm * float(e.mean()), meth
    return np.nan, np.nan, "not predicted"


# ---------------------------------------------------------------------------
# discrete-time (DT) noise-only predictions for the linear models
# ---------------------------------------------------------------------------
DISC_COEFFS = {"euler": [1.0, -1.0], "taylor_zhang": [1.0, -1.5, 1.0, -0.5]}


def dt_system(model, disc, gamma, tau):
    """x_{k+1} = Phi x_k + Gam u_k with x = [e_k, ..., e_{k-s+1}, z_k]."""
    A, b = linear_AB(model, gamma)
    if A is None or disc not in DISC_COEFFS:
        return None
    c = np.asarray(DISC_COEFFS[disc], dtype=float)
    cp = c[1:] / c[0]
    s = len(cp)
    nz = A.shape[0] - 1
    N = s + nz
    Phi = np.zeros((N, N))
    Phi[0, :s] = -cp
    Phi[0, 0] += tau * A[0, 0] / c[0]
    if nz:
        Phi[0, s:] = tau * A[0, 1:] / c[0]
        Phi[s:, 0] = tau * A[1:, 0]
        Phi[s:, s:] = np.eye(nz) + tau * A[1:, 1:]
    for i in range(1, s):
        Phi[i, i - 1] = 1.0
    Gam = np.zeros(N)
    Gam[0] = tau / c[0]
    return Phi, Gam


def _dt_run(Phi, Gam, u):
    x = np.zeros(Phi.shape[0])
    out = np.empty(len(u) + 1)
    out[0] = 0.0
    for k, uk in enumerate(u):
        x = Phi @ x + Gam * uk
        out[k + 1] = x[0]
        if not np.isfinite(x[0]) or abs(x[0]) > 1e12:
            out[k + 1:] = np.inf
            break
    return out


@lru_cache(maxsize=None)
def dt_prediction(model, disc, noise, gamma, tau, m):
    """(pred_max, pred_mean) of the noise-only discrete error recursion (start-up ignored)."""
    sys_ = dt_system(model, disc, gamma, tau)
    if sys_ is None:
        return np.nan, np.nan
    if noise == "none":
        return 0.0, 0.0
    Phi, Gam = sys_
    if np.max(np.abs(np.linalg.eigvals(Phi))) > 1.0 + 1e-12:
        return np.inf, np.inf
    N = int(round(T_H / tau))
    t = np.arange(N + 1) * tau
    k0 = int(np.floor((1.0 - FRAC) * N))
    sm = math.sqrt(m)
    p = NOISE_LEVELS[noise]
    if noise not in STOCHASTIC:
        e = np.abs(_dt_run(Phi, Gam, nu_fn(noise)(t[:-1])))[k0:]
        return sm * float(e.max()), sm * float(e.mean())
    imp = np.zeros(N)
    imp[0] = 1.0
    h = _dt_run(Phi, Gam, imp)[1:]               # h_n, n = 1..N (response to u_0 = 1)
    if noise == "gaussian":
        var = p["std"] ** 2 * np.concatenate([[0.0], np.cumsum(h ** 2)])
    else:
        H = np.cumsum(h)
        var = p["sigma"] ** 2 * tau * np.concatenate([[0.0, 0.0], np.cumsum(H[:-1] ** 2)])[:N + 1]
    return np.nan, chi_mean(m) * float(np.sqrt(var[k0:]).mean())


# ---------------------------------------------------------------------------
# transient diagnostics (review addition): is a "floor" really a steady floor?
# ---------------------------------------------------------------------------
TRANSIENT_FRAC = 0.1      # a cell is transient-limited if a decaying part is > 10 % of its floor


@lru_cache(maxsize=None)
def dt_ic_response(model, disc, gamma, tau):
    """max over the steady window of |e_k| of the noise-free discrete error
    recursion started from e_0 = 1 (history = e_0, auxiliary states 0).

    For the linear models the error dynamics act entry-wise and identically,
    so the initial-state part of ||E_k|| is exactly |phi_k| ||E_0|| (static
    problems; on time-varying problems the truncation floor adds on top).
    Taylor-Zhang's two Euler start-up steps are ignored (order-of-magnitude
    diagnostic)."""
    sys_ = dt_system(model, disc, gamma, tau)
    if sys_ is None:
        return np.nan
    Phi, _ = sys_
    if np.max(np.abs(np.linalg.eigvals(Phi))) > 1.0 + 1e-12:
        return np.inf
    s = len(DISC_COEFFS[disc]) - 1
    x = np.zeros(Phi.shape[0])
    x[:s] = 1.0
    N = int(round(T_H / tau))
    k0 = int(np.floor((1.0 - FRAC) * N))
    x = np.linalg.matrix_power(Phi, k0) @ x
    mx = abs(x[0])
    for _ in range(N - k0):
        x = Phi @ x
        mx = max(mx, abs(x[0]))
    return float(mx)


@lru_cache(maxsize=None)
def ct_noise_transient(model, noise, gamma, m, tau):
    """(transient, steady) parts of the CT noise-only response on the window,
    linear models only.  Deterministic noise: steady = particular solution
    y_p = Pi w (Sylvester A Pi - Pi G = -b e1^T), transient = max_W ||y - y_p||.
    Gaussian: stationary second moment (Lyapunov) vs the finite-horizon one.
    Random walk: non-stationary noise -> (nan, nan)."""
    from scipy.linalg import solve_continuous_lyapunov, solve_sylvester
    A, b = linear_AB(model, gamma)
    if A is None or noise in ("none", "random_walk"):
        return np.nan, np.nan
    sm = math.sqrt(m)
    tw = _window()
    if noise == "gaussian":
        qn = NOISE_LEVELS[noise]["std"] ** 2 * tau
        P = solve_continuous_lyapunov(A, -qn * np.outer(b, b))
        ss = chi_mean(m) * math.sqrt(max(P[0, 0], 0.0))
        fin = ct_prediction(model, noise, gamma, m, tau)[1]
        return abs(fin - ss), ss
    Gm, g0 = _noise_generator(noise)
    k = Gm.shape[0]
    E1 = np.zeros((1, k))
    E1[0, 0] = 1.0
    Pi = solve_sylvester(A, -Gm, -np.outer(b, E1))
    y = _linear_det_response(A, b, noise, tw)                  # e(t) (first state)
    from scipy.linalg import expm
    w = np.array([expm(Gm * t) @ g0 for t in tw])              # generator state on the window
    yp = (w @ Pi.T)[:, 0]
    return sm * float(np.max(np.abs(y - yp))), sm * float(np.max(np.abs(yp)))


# ---------------------------------------------------------------------------
# predictions attached to cells
# ---------------------------------------------------------------------------
def attach_predictions(cells):
    pm, pmean, meth, dmax, dmean = [], [], [], [], []
    for r in cells.itertuples(index=False):
        a, b, c = ct_prediction(r.model, r.noise, float(round(r.gamma, 9)), int(r.m), float(r.tau))
        pm.append(a); pmean.append(b); meth.append(c)
        d1, d2 = dt_prediction(r.model, r.disc, r.noise, float(round(r.gamma, 9)), float(r.tau), int(r.m))
        dmax.append(d1); dmean.append(d2)
    cells = cells.copy()
    pm = [0.0 if (np.isfinite(v) and v < CT_ZERO) else v for v in pm]
    pmean = [0.0 if (np.isfinite(v) and v < CT_ZERO) else v for v in pmean]
    cells["ct_max"], cells["ct_mean"], cells["ct_method"] = pm, pmean, meth
    cells["dt_max"], cells["dt_mean"] = dmax, dmean
    sto = cells["noise"].isin(STOCHASTIC)
    cells["meas"] = np.where(sto, cells["med_mean"], cells["med"])
    cells["ct"] = np.where(sto, cells["ct_mean"], cells["ct_max"])
    cells["dt"] = np.where(sto, cells["dt_mean"], cells["dt_max"])
    base = cells[cells["noise"] == "none"].set_index(["problem", "start", "model", "disc", "arm", "tau"])
    idx = pd.MultiIndex.from_frame(cells[["problem", "start", "model", "disc", "arm", "tau"]])
    cells["floor_none"] = base["med"].reindex(idx).to_numpy()
    cells["floor_none_mean"] = base["med_mean"].reindex(idx).to_numpy()
    none_ref = np.where(sto, cells["floor_none_mean"], cells["floor_none"])
    cells["penalty"] = cells["meas"] / none_ref
    cells["ct_plus_disc"] = cells["ct"] + none_ref
    cells["ct_gap"] = cells["meas"] / cells["ct_plus_disc"]
    # transient diagnostics (linear models): initial-state part (exact discrete
    # recursion x median ||E(0)||) and the decaying part of the CT noise response
    tic, ctt, cts = [], [], []
    for r in cells.itertuples(index=False):
        g_ = float(round(r.gamma, 9))
        phi = dt_ic_response(r.model, r.disc, g_, float(r.tau))
        tic.append(phi * r.init_med if np.isfinite(phi) and np.isfinite(r.init_med) else np.nan)
        a, b = ct_noise_transient(r.model, r.noise, g_, int(r.m), float(r.tau))
        ctt.append(a); cts.append(b)
    cells["transient_ic"], cells["ct_transient"], cells["ct_steady"] = tic, ctt, cts
    lin = cells["model"].isin(LINEAR_MODELS)
    meas = cells["meas"].to_numpy(float)
    ok = lin.to_numpy() & np.isfinite(meas)
    tr = np.nan_to_num(cells["transient_ic"].to_numpy(float), nan=0.0)
    tr = np.maximum(tr, np.nan_to_num(cells["ct_transient"].to_numpy(float), nan=0.0))
    with np.errstate(invalid="ignore"):
        cells["transient_limited"] = ok & (tr > TRANSIENT_FRAC * meas)
    cells["transient_checked"] = ok
    # the penalty's denominator (noise-free floor) is transient-limited
    tl_none = cells[cells["noise"] == "none"].set_index(["problem", "start", "model", "disc", "arm", "tau"])
    cells["none_transient_limited"] = tl_none["transient_limited"].reindex(idx).fillna(False).to_numpy(bool)
    # censoring (review): some but < 50 % of the trajectories diverged -> finite median, flagged
    cells["partial_div"] = (cells["n_div"] > 0) & (cells["frac_div"] < 0.5)
    return cells


# ---------------------------------------------------------------------------
# bootstrap helpers
# ---------------------------------------------------------------------------
_RNG = np.random.default_rng(BOOT_SEED)


def _med_inf(a, axis=-1):
    a = np.where(np.isinf(a), BIG, a)
    v = np.median(a, axis=axis)
    return np.where(v >= BIG / 10, np.inf, v)


def _nanmed_inf(a, axis=-1):
    """Median over `axis` ignoring NaN (padding / failed runs) with +inf = worst case."""
    a = np.where(np.isinf(a), BIG, a)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v = np.nanmedian(a, axis=axis)
    return np.where(v >= BIG / 10, np.inf, v)


def _cluster_tensor(values, clusters, labels):
    """values grouped by cluster label -> (len(labels), max size) array, NaN padded."""
    values, clusters = np.asarray(values, float), np.asarray(clusters)
    groups = [values[clusters == c] for c in labels]
    L = max([len(g) for g in groups] + [1])
    M = np.full((len(labels), L), np.nan)
    for i, g in enumerate(groups):
        M[i, :len(g)] = g
    return M


def boot_ratio(num, den, clusters=None, nboot=None):
    """Ratio of medians (noise / noise-free floor, paired by trajectory) with a
    CLUSTER bootstrap over problem seeds (review 2026-10-18, REVIEW_CHECKLIST
    item 5): the B trajectories of one seed share the problem instance and,
    for deterministic noise and linear models, have identical steady floors,
    so they are not independent replicates.  Whole seeds are resampled (with
    all their trajectories, pairing kept); with fewer than MIN_CLUSTERS seeds
    no CI is reported.  Returns (est, lo, hi, n_pairs, n_seeds)."""
    nboot = nboot or BOOT
    num, den = np.asarray(num, float), np.asarray(den, float)
    clusters = np.zeros(num.size, int) if clusters is None else np.asarray(clusters)
    ok = ~(np.isnan(num) | np.isnan(den))
    num, den, clusters = num[ok], den[ok], clusters[ok]
    if num.size == 0:
        return np.nan, np.nan, np.nan, 0, 0
    mn, md = float(_med_inf(num)), float(_med_inf(den))
    est = np.inf if (np.isinf(mn) and np.isinf(md)) else mn / md
    labels = list(dict.fromkeys(clusters.tolist()))
    k = len(labels)
    if k < MIN_CLUSTERS:
        return est, np.nan, np.nan, int(num.size), k
    Nt, Dt = _cluster_tensor(num, clusters, labels), _cluster_tensor(den, clusters, labels)
    idx = _RNG.integers(0, k, size=(nboot, k))
    r = _nanmed_inf(Nt[idx].reshape(nboot, -1)) / _nanmed_inf(Dt[idx].reshape(nboot, -1))
    r = np.where(np.isnan(r), np.inf, r)
    return est, _q_inf(r, 0.025), _q_inf(r, 0.975), int(num.size), k


def _q_inf(r, p):
    """Quantile with +inf (censored resamples) as the worst value (np.quantile
    would interpolate inf with inf to NaN)."""
    v = float(np.quantile(np.where(np.isinf(r), BIG, r), p))
    return np.inf if v >= BIG / 10 else v


def boot_median(x, nboot=BOOT):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan, np.nan, np.nan
    idx = _RNG.integers(0, x.size, size=(nboot, x.size))
    r = _med_inf(x[idx], axis=1)
    return float(_med_inf(x)), float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975))


def fit_slope(taus, vals):
    """LS slope of log10(vals) vs log10(tau) over finite points above the
    round-off floor, plus consecutive local slopes."""
    taus, vals = np.asarray(taus, float), np.asarray(vals, float)
    o = np.argsort(taus)
    taus, vals = taus[o], vals[o]
    loc = []
    for i in range(len(taus) - 1):
        a, b = vals[i], vals[i + 1]
        if np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0:
            loc.append(float(np.log10(b / a) / np.log10(taus[i + 1] / taus[i])))
        else:
            loc.append(np.nan)
    ok = np.isfinite(vals) & (vals > ROUNDOFF)
    if ok.sum() < 2:
        return np.nan, loc, int(ok.sum())
    sl = np.polyfit(np.log10(taus[ok]), np.log10(vals[ok]), 1)[0]
    return float(sl), loc, int(ok.sum())


SLOPE_TOL = 0.2           # consecutive local slopes agreeing within this -> asymptotic range
NEAR_ROUNDOFF = 1e-10     # a smallest-tau point below this may be round-off contaminated


def asymptotic_range(taus, vals, excluded=None):
    """Select the asymptotic range of a floor-vs-tau series (review addition).

    Points that are non-finite, at/below the round-off floor (1e-12) or
    flagged in ``excluded`` (transient-limited cells) are dropped first.  The
    consecutive (local) slopes of the remaining points are then compared from
    the small-tau end: the longest run of points whose consecutive local
    slopes agree within SLOPE_TOL, starting at the smallest tau, is the
    asymptotic range; if the two smallest-tau local slopes disagree and the
    smallest-tau value is below NEAR_ROUNDOFF the smallest point is treated as
    round-off contaminated and dropped instead (then the run restarts).
    Returns (mask over the sorted taus, flag string)."""
    taus, vals = np.asarray(taus, float), np.asarray(vals, float)
    ok = np.isfinite(vals) & (vals > ROUNDOFF)
    if excluded is not None:
        ok &= ~np.asarray(excluded, bool)
    idx = np.flatnonzero(ok)                      # taus are sorted ascending
    mask = np.zeros(len(taus), bool)
    if idx.size < 2:
        return mask, "fewer than 2 usable points"
    lt, lv = np.log10(taus[idx]), np.log10(vals[idx])
    loc = np.diff(lv) / np.diff(lt)               # loc[i]: between idx[i] and idx[i+1]
    if idx.size == 2:
        mask[idx] = True
        return mask, "2 points (asymptotic range not verifiable)"
    start, flag = 0, "all points consistent"
    if abs(loc[0] - loc[1]) > SLOPE_TOL and vals[idx[0]] < NEAR_ROUNDOFF:
        start, flag = 1, "smallest tau dropped (near round-off)"
    end = start + 1                               # points idx[start..end] used
    while end + 1 < idx.size and abs(loc[end] - loc[start]) <= SLOPE_TOL:
        end += 1
    if end < idx.size - 1:
        flag = (flag + "; " if start else "") + f"largest {idx.size - 1 - end} tau dropped (pre-asymptotic)"
    mask[idx[start:end + 1]] = True
    if end - start == 1 and idx.size > 2:
        flag += " -- 2 points left"
    return mask, flag


def boot_slope(df_cell, value="floor", nboot=None, excluded=None):
    """Asymptotic floor-vs-tau slope with a seed-cluster bootstrap CI.

    Per tau the floor is the median over all trajectories (+inf = diverged =
    worst case).  The point estimate is the LS slope over the asymptotic range
    (``asymptotic_range``: round-off, transient-limited (``excluded``) and
    pre-asymptotic points dropped); ``slope_all`` is the LS slope over every
    usable point.  CI: whole problem seeds are resampled, the SAME seeds at
    every tau (a seed is one problem instance shared across tau), and the
    masked LS over the selected points is refitted (vectorised).  Fewer than
    MIN_CLUSTERS seeds -> no CI."""
    nboot = nboot or BOOT
    taus = sorted(df_cell["tau"].unique())
    empty = {"slope": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "local": [], "n_points": 0,
             "slope_all": np.nan, "range": "no data", "taus_used": [], "n_seeds": 0}
    if len(taus) < 2:
        return empty
    subs = [df_cell[np.isclose(df_cell["tau"], t)] for t in taus]
    arrs = [s[value].to_numpy(float) for s in subs]
    if any(np.all(np.isnan(a)) for a in arrs):
        return empty
    medv = [float(_med_inf(a[~np.isnan(a)])) for a in arrs]
    est_all, loc, _ = fit_slope(taus, medv)
    sel, flag = asymptotic_range(taus, medv, excluded)
    seeds = list(dict.fromkeys(df_cell["seed"].astype(int).tolist()))
    out = {**empty, "local": loc, "slope_all": est_all, "range": flag, "n_seeds": len(seeds)}
    if sel.sum() < 2:
        return out
    tt = np.asarray(taus, float)
    est = float(np.polyfit(np.log10(tt[sel]), np.log10(np.asarray(medv)[sel]), 1)[0])
    out.update({"slope": est, "n_points": int(sel.sum()), "taus_used": [float(t) for t in tt[sel]]})
    if len(seeds) < MIN_CLUSTERS:
        return out
    tens = [_cluster_tensor(a, s["seed"].astype(int).to_numpy(), seeds) for a, s in zip(arrs, subs)]
    idx = _RNG.integers(0, len(seeds), size=(nboot, len(seeds)))
    meds = np.column_stack([_nanmed_inf(tn[idx].reshape(nboot, -1)) for tn in tens])
    x = np.log10(tt)[None, :].repeat(nboot, 0)
    w = (np.isfinite(meds) & (meds > ROUNDOFF) & sel[None, :]).astype(float)
    y = np.where(w > 0, np.log10(np.where(w > 0, meds, 1.0)), 0.0)
    sw, sx, sy = w.sum(1), (w * x).sum(1), (w * y).sum(1)
    sxx, sxy = (w * x * x).sum(1), (w * x * y).sum(1)
    den = sw * sxx - sx * sx
    ok = (sw >= 2) & (np.abs(den) > 1e-14)
    sl = (sw * sxy - sx * sy)[ok] / den[ok]
    if sl.size:
        out.update({"ci_lo": float(np.quantile(sl, 0.025)), "ci_hi": float(np.quantile(sl, 0.975))})
    return out


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def _log(v):
    v = np.asarray(v, float)
    with np.errstate(divide="ignore"):
        return np.where(v > 0, np.log10(np.where(v > 0, v, 1.0)), -np.inf)


def heatmap_figure(cells, problem, start, arm, name, kind="floor", title=None):
    sub = cells[(cells.problem == problem) & (cells.start == start) & (cells.arm == arm)]
    if sub.empty:
        return None
    discs = [d for d in ["euler", "taylor_zhang"] if d in set(sub.disc)]
    taus = sorted(sub.tau.unique(), reverse=True)
    models = [m for m in MODELS if m in set(sub.model)]
    noises = [n for n in NOISES if n in set(sub.noise)]
    if kind != "floor":
        noises = [n for n in noises if n != "none"]
    nr, nc = len(discs), len(taus)
    fig, axes = plt.subplots(nr, nc, figsize=(style.DOUBLE, 0.25 + 1.62 * nr), squeeze=False,
                             gridspec_kw={"wspace": 0.08, "hspace": 0.32})
    if kind == "floor":
        vals_all = _log(sub["med"])
        fin = vals_all[np.isfinite(vals_all)]
        vmin = math.floor(np.nanmin(fin)) if fin.size else -12
        vmax = math.ceil(np.nanmax(fin)) if fin.size else 1
        vmin = max(vmin, -16)
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = style.CMAP_SEQ
        clabel = r"$\log_{10}$ median MSSRE"
    else:
        col = "penalty" if kind == "penalty" else "ct_gap"
        vals_all = _log(sub[col])
        fin = vals_all[np.isfinite(vals_all)]
        lim = max(1.0, math.ceil(np.nanmax(np.abs(fin)))) if fin.size else 1.0
        lim = min(lim, 12.0)
        norm = TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim)
        cmap = style.CMAP_DIV
        clabel = (r"$\log_{10}$ (floor / noise-free floor)" if kind == "penalty"
                  else r"$\log_{10}$ (measured / (CT pred. + noise-free floor))")
    im = None
    for i, d in enumerate(discs):
        for j, t in enumerate(taus):
            ax = axes[i, j]
            ax.grid(False)
            M = np.full((len(models), len(noises)), np.nan)
            D = np.zeros_like(M, dtype=bool)
            Z = np.zeros_like(M, dtype=bool)
            TL = np.zeros_like(M, dtype=bool)          # transient-limited (linear models)
            PD = np.zeros_like(M, dtype=bool)          # some (< 50 %) trajectories diverged
            for a, mdl in enumerate(models):
                for b, nz in enumerate(noises):
                    r = sub[(sub.disc == d) & (sub.tau == t) & (sub.model == mdl) & (sub.noise == nz)]
                    if r.empty:
                        continue
                    r = r.iloc[0]
                    PD[a, b] = bool(r.get("partial_div", False))
                    TL[a, b] = bool(r.get("transient_limited", False)) or (
                        kind != "floor" and bool(r.get("none_transient_limited", False)))
                    if kind == "floor":
                        v = r["med"]
                    elif kind == "penalty":
                        v = r["penalty"]
                    else:
                        v = r["ct_gap"]
                    if not np.isfinite(r["med"]) or r["frac_div"] >= 0.5:
                        D[a, b] = True
                    elif kind == "floor" and np.isfinite(v) and v <= 0:
                        M[a, b] = norm.vmin              # exact zero residual (below round-off)
                        Z[a, b] = True
                    else:
                        M[a, b] = float(_log(v)) if np.isfinite(v) and v > 0 else np.nan
            im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, norm=norm, aspect="auto")
            for a in range(len(models)):
                for b in range(len(noises)):
                    if D[a, b]:
                        ax.add_patch(plt.Rectangle((b - 0.5, a - 0.5), 1, 1, facecolor="#f0efec",
                                                   edgecolor="#bdbcb8", hatch="////", lw=0.0))
                        ax.text(b, a, "div", ha="center", va="center", fontsize=4.8, color=style.INK2)
                    elif np.isfinite(M[a, b]):
                        rgba = cmap(norm(M[a, b]))
                        lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                        txt = f"{M[a, b]:.1f}"
                        txt = "0.0" if txt == "-0.0" else txt
                        txt = "exact 0" if Z[a, b] else txt
                        txt += ("†" if TL[a, b] else "") + ("*" if PD[a, b] else "")
                        ax.text(b, a, txt, ha="center", va="center", fontsize=4.8,
                                color="white" if lum < 0.5 else style.INK)
                    else:
                        ax.text(b, a, "n/a", ha="center", va="center", fontsize=4.5, color=style.GREY)
            if noises and noises[0] == "none":
                ax.axvline(0.5, color="white", lw=1.6)
            ax.set_xticks(range(len(noises)))
            ax.set_xticklabels([NOISE_TEX[n] for n in noises], rotation=45, ha="right", fontsize=5.5)
            ax.set_yticks(range(len(models)))
            ax.set_yticklabels([MODEL_TEX[m] for m in models] if j == 0 else [], fontsize=5.5)
            ax.tick_params(length=0)
            for sp_ in ax.spines.values():
                sp_.set_visible(False)
            ax.set_title(f"{DISC_FIG[d]}, $\\tau$ = {_fmt_tau(t)}", fontsize=7)
            if i < nr - 1:
                ax.set_xticklabels([])
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.015)
    cb.set_label(clabel, fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    if title is None:
        st_ = "random $x_0\\sim U[-2,2]$" if start == "rand" else "near-solution $x_0$"
        arm_s = "fixed " + arm_tex(arm)
        title = f"{PROBLEM_FIG.get(problem, problem)}, {st_}, {arm_s}"
    fig.suptitle(title, fontsize=7.5, y=1.0)
    fig.text(0.01, -0.02, "† transient-limited at $T$ = 6 s (decaying part > 10 % of the value; checked for the "
             "linear models only" + ("; for ratios: of either floor" if kind != "floor" else "")
             + ")    * some (< 50 %) trajectories diverged    div: median diverged",
             fontsize=5.5, color=style.INK2, ha="left", va="top")
    return style.savefig(fig, name)


SELECTED = [("znn", "constant"), ("ieznn", "constant"), ("ieznn", "linear"), ("dieznn", "linear"),
            ("znn", "gaussian"), ("hntznn", "harmonic"), ("znn_sbp", "constant"),
            ("sptc_an_foznn", "constant")]
# review extension: the harmonic arm, internal model matched / mismatched, Li vs linear activation
SELECTED_HARMONIC = [("znn", "harmonic"), ("hntznn", "harmonic"), ("hntznn_lin", "harmonic"),
                     ("hntznn_lin_mis", "harmonic")]


PAPER_FLOOR_FIGS = {"fig_E5_harmonic_arm_gamma10"}     # Supplementary Material figure (paper text width)


def floor_vs_tau_paper(cells, problem, start, arm, name, selected=None):
    """floor_vs_tau_figure drawn at the paper's text width: 2 x 2 panels, legend below, no suptitle
    (problem, noise, start and arm are given in the caption)."""
    sub = cells[(cells.problem == problem) & (cells.start == start) & (cells.arm == arm)]
    if sub.empty or sub.tau.nunique() < 1:
        return None
    sel = [(m, n) for m, n in (selected or SELECTED) if ((sub.model == m) & (sub.noise == n)).any()]
    if not sel:
        return None
    ncol = 2
    nrow = int(math.ceil(len(sel) / ncol))
    with style.print_rc():
        fig, axes = plt.subplots(nrow, ncol, figsize=(style.TEXTWIDTH, 1.32 * nrow + 0.75), squeeze=False,
                                 sharex=True, constrained_layout=True)
        fig.get_layout_engine().set(w_pad=0.04, h_pad=0.04, wspace=0.06, hspace=0.06)
        for k, (mdl, nz) in enumerate(sel):
            ax = axes[k // ncol, k % ncol]
            sto = nz in STOCHASTIC
            for d in ["euler", "taylor_zhang"]:
                st = keys.style_in_group("disc", d)
                for which, ls, mfc in (("noise", st["linestyle"], st["color"]), ("none", ":", "white")):
                    r = sub[(sub.model == mdl) & (sub.disc == d) & (sub.noise == (nz if which == "noise" else "none"))]
                    r = r.sort_values("tau")
                    if r.empty:
                        continue
                    y = r["med_mean" if sto else "med"].to_numpy(float)
                    lo = r["q1_mean" if sto else "q1"].to_numpy(float)
                    hi = r["q3_mean" if sto else "q3"].to_numpy(float)
                    fin = np.isfinite(y)
                    tt = r["tau"].to_numpy(float)
                    if fin.any():
                        yerr = np.vstack([np.clip(y[fin] - lo[fin], 0, None),
                                          np.clip(np.where(np.isfinite(hi[fin]), hi[fin], y[fin]) - y[fin], 0, None)])
                        ax.errorbar(tt[fin], y[fin], yerr=yerr, color=st["color"], marker=st["marker"],
                                    linestyle=ls, markerfacecolor=mfc, capsize=1.5, lw=1.1, ms=4.0, elinewidth=0.7)
                    for x_ in tt[~fin]:
                        ax.annotate("div", (x_, 1.0), xycoords=("data", "axes fraction"), ha="center", va="top",
                                    fontsize=7.5, color=style.INK2)
            r = sub[(sub.model == mdl) & (sub.noise == nz) & (sub.disc == sub.disc.iloc[0])].sort_values("tau")
            ct = r["ct"].to_numpy(float)
            tt = r["tau"].to_numpy(float)
            okc = np.isfinite(ct) & (ct > 0)
            if okc.sum() >= 2:
                ax.plot(tt[okc], ct[okc], color=style.INK, linestyle="--", lw=1.0, marker=None, zorder=6)
            elif okc.sum() == 1:
                ax.plot(tt[okc], ct[okc], color=style.INK, linestyle="none", marker="_", ms=8, zorder=6)
            if mdl in LINEAR_MODELS:
                rd = sub[(sub.model == mdl) & (sub.noise == nz) & (sub.disc == "euler")].sort_values("tau")
                dtv, tdv = rd["dt"].to_numpy(float), rd["tau"].to_numpy(float)
                okd = np.isfinite(dtv) & (dtv > CT_ZERO)
                if okd.sum() >= 1:
                    ax.plot(tdv[okd], dtv[okd], color=style.GREY, linestyle="-.", lw=1.0, zorder=5,
                            marker="x" if okd.sum() == 1 else None, ms=4.5)
            zer = np.isfinite(ct) & (ct == 0)
            if zer.any():
                msg = ("CT prediction: 0" if zer.all() else
                       f"CT prediction: 0 for $\\tau\\leq${_fmt_tau(tt[zer].max())}")
                ax.text(0.04, 0.88, msg, transform=ax.transAxes, fontsize=7.5, color=style.INK, ha="left",
                        va="top")
            ax.set_xscale("log")
            ax.set_yscale("log")
            if not np.isfinite(sub[(sub.model == mdl) & (sub.noise == nz)]["med"]).any():
                ax.text(0.5, 0.5, "diverged at every $\\tau$", transform=ax.transAxes, ha="center", va="center",
                        fontsize=7.5, color=style.INK2)
                ax.set_yticks([])
                ax.minorticks_off()
            ax.set_title(MODEL_TEX[mdl].replace(" (mismatched)", ", mismatched internal model"))
            if k % ncol == 0:
                ax.set_ylabel("residual floor")
            if k // ncol == nrow - 1:
                ax.set_xlabel(r"sampling period $\tau$ (s)")
        for k in range(len(sel), nrow * ncol):
            axes[k // ncol, k % ncol].set_visible(False)
        handles = []
        for d in ["euler", "taylor_zhang"]:
            st = keys.style_in_group("disc", d)
            handles.append(Line2D([], [], color=st["color"], marker=st["marker"], linestyle=st["linestyle"], ms=4.0,
                                  label=f"{DISC_FIG[d]}, with noise"))
            handles.append(Line2D([], [], color=st["color"], marker=st["marker"], linestyle=":", ms=4.0,
                                  markerfacecolor="white", label=f"{DISC_FIG[d]}, noise-free"))
        handles.append(Line2D([], [], color=style.INK, linestyle="--", label="CT prediction (noise only)"))
        handles.append(Line2D([], [], color=style.GREY, linestyle="-.", label="DT prediction, Euler (linear models)"))
        fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, columnspacing=1.2)
        return style.savefig(fig, name)


def floor_vs_tau_figure(cells, problem, start, arm, name, selected=None, title=None):
    if name in PAPER_FLOOR_FIGS:
        return floor_vs_tau_paper(cells, problem, start, arm, name, selected=selected)
    sub = cells[(cells.problem == problem) & (cells.start == start) & (cells.arm == arm)]
    if sub.empty or sub.tau.nunique() < 1:
        return None
    sel = [(m, n) for m, n in (selected or SELECTED) if ((sub.model == m) & (sub.noise == n)).any()]
    if not sel:
        return None
    ncol = 4
    nrow = int(math.ceil(len(sel) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(style.DOUBLE, 0.55 + 1.55 * nrow), squeeze=False,
                             sharex=True)
    for k, (mdl, nz) in enumerate(sel):
        ax = axes[k // ncol, k % ncol]
        sto = nz in STOCHASTIC
        for d in ["euler", "taylor_zhang"]:
            st = keys.style_in_group("disc", d)
            for which, ls, mfc in (("noise", st["linestyle"], st["color"]), ("none", ":", "white")):
                r = sub[(sub.model == mdl) & (sub.disc == d) & (sub.noise == (nz if which == "noise" else "none"))]
                r = r.sort_values("tau")
                if r.empty:
                    continue
                y = r["med_mean" if sto else "med"].to_numpy(float)
                lo = r["q1_mean" if sto else "q1"].to_numpy(float)
                hi = r["q3_mean" if sto else "q3"].to_numpy(float)
                fin = np.isfinite(y)
                tt = r["tau"].to_numpy(float)
                if fin.any():
                    yerr = np.vstack([np.clip(y[fin] - lo[fin], 0, None),
                                      np.clip(np.where(np.isfinite(hi[fin]), hi[fin], y[fin]) - y[fin], 0, None)])
                    ax.errorbar(tt[fin], y[fin], yerr=yerr, color=st["color"], marker=st["marker"],
                                linestyle=ls, markerfacecolor=mfc, capsize=1.5, lw=1.0, ms=3.2, elinewidth=0.6)
                for x_ in tt[~fin]:
                    ax.annotate("div", (x_, 1.0), xycoords=("data", "axes fraction"), ha="center", va="top",
                                fontsize=5, color=st["color"])
        # CT prediction (same for both discretizations; first disc's row)
        r = sub[(sub.model == mdl) & (sub.noise == nz) & (sub.disc == sub.disc.iloc[0])].sort_values("tau")
        ct = r["ct"].to_numpy(float)
        tt = r["tau"].to_numpy(float)
        okc = np.isfinite(ct) & (ct > 0)
        if okc.sum() >= 2:
            ax.plot(tt[okc], ct[okc], color=style.INK, linestyle="--", lw=0.9, marker=None, zorder=6)
        elif okc.sum() == 1:
            ax.plot(tt[okc], ct[okc], color=style.INK, linestyle="none", marker="_", ms=7, zorder=6)
        # exact discrete-time noise-only prediction (linear models; Euler row; review addition)
        if mdl in LINEAR_MODELS:
            rd = sub[(sub.model == mdl) & (sub.noise == nz) & (sub.disc == "euler")].sort_values("tau")
            dtv, tdv = rd["dt"].to_numpy(float), rd["tau"].to_numpy(float)
            okd = np.isfinite(dtv) & (dtv > CT_ZERO)
            if okd.sum() >= 1:
                ax.plot(tdv[okd], dtv[okd], color=style.GREY, linestyle="-.", lw=0.9, zorder=5,
                        marker="x" if okd.sum() == 1 else None, ms=4)
        zer = np.isfinite(ct) & (ct == 0)
        if zer.any():
            msg = ("CT prediction: 0" if zer.all() else
                   f"CT prediction: 0 for $\\tau\\leq${_fmt_tau(tt[zer].max())}")
            ax.text(0.04, 0.88, msg, transform=ax.transAxes, fontsize=5.5, color=style.INK, ha="left",
                    va="top")
        ax.set_xscale("log")
        ax.set_yscale("log")
        if not np.isfinite(sub[(sub.model == mdl) & (sub.noise == nz)]["med"]).any():
            ax.text(0.5, 0.5, "diverged at every $\\tau$", transform=ax.transAxes, ha="center", va="center",
                    fontsize=6.5, color=style.INK2)
            ax.set_yticks([])
            ax.minorticks_off()
        ax.set_title(f"{MODEL_TEX[mdl]}, {NOISE_TEX[nz]} noise", fontsize=7)
        if k % ncol == 0:
            ax.set_ylabel("residual floor")
        if k // ncol == nrow - 1:
            ax.set_xlabel(r"sampling period $\tau$ (s)")
    for k in range(len(sel), nrow * ncol):
        axes[k // ncol, k % ncol].set_visible(False)
    handles = []
    for d in ["euler", "taylor_zhang"]:
        st = keys.style_in_group("disc", d)
        handles.append(Line2D([], [], color=st["color"], marker=st["marker"], linestyle=st["linestyle"],
                              label=f"{DISC_FIG[d]}, with noise"))
        handles.append(Line2D([], [], color=st["color"], marker=st["marker"], linestyle=":",
                              markerfacecolor="white", label=f"{DISC_FIG[d]}, noise-free"))
    handles.append(Line2D([], [], color=style.INK, linestyle="--", label="CT prediction (noise only)"))
    handles.append(Line2D([], [], color=style.GREY, linestyle="-.", label="DT prediction, Euler (linear models)"))
    top = 0.80 if nrow == 1 else 0.93
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, top - 0.005), fontsize=6,
               frameon=False)
    if title:
        fig.suptitle(title, fontsize=7.5, y=1.0 if nrow == 1 else 1.01, va="bottom")
    return style.savefig(fig, name)


def ct_scatter_figure(cells, problem, start, name):
    sub = cells[(cells.problem == problem) & (cells.start == start) & (cells.noise != "none")]
    if sub.empty:
        return None
    models = [m for m in MODELS if m in set(sub.model)]
    fig, axes = plt.subplots(2, 3, figsize=(style.DOUBLE, 4.3), squeeze=False)
    sub = sub.copy()
    sub["ct"] = np.where(sub["ct"] < CT_ZERO, 0.0, sub["ct"])
    meas_all = sub["meas"].to_numpy(float)
    ct_all = sub["ct"].to_numpy(float)
    good = np.concatenate([meas_all[np.isfinite(meas_all) & (meas_all > 0)],
                           ct_all[np.isfinite(ct_all) & (ct_all > 0)]])
    lo = 10 ** math.floor(np.log10(good.min())) if good.size else 1e-12
    hi = 10 ** math.ceil(np.log10(good.max())) if good.size else 1e2
    zero_x = lo / 10 ** 1.0
    for k, mdl in enumerate(models[:6]):
        ax = axes[k // 3, k % 3]
        s = sub[sub.model == mdl]
        for nz in NOISES[1:]:
            r = s[s.noise == nz]
            if r.empty:
                continue
            st = keys.style_in_group("noise", nz)
            x = r["ct"].to_numpy(float)
            y = r["meas"].to_numpy(float)
            ok = np.isfinite(y) & (y > 0) & np.isfinite(x)
            xx = np.where(x[ok] > 0, x[ok], zero_x)
            ax.scatter(xx, y[ok], s=9, color=st["color"], marker=st["marker"], linewidths=0.4,
                       edgecolors=st["color"], facecolors="none" if nz in STOCHASTIC else st["color"],
                       alpha=0.85, label=NOISE_TEX[nz])
        ax.plot([lo, hi], [lo, hi], color=style.GREY, lw=0.7, linestyle="-")
        ax.axvspan(zero_x / 3, zero_x * 3, color="#f0efec", lw=0, zorder=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(zero_x / 3, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(MODEL_TEX[mdl], fontsize=7)
        if k % 3 == 0:
            ax.set_ylabel("measured floor")
        if k // 3 == 1:
            ax.set_xlabel("CT prediction (noise only)")
        ax.text(zero_x, hi / 3, "CT: 0", ha="center", va="top", fontsize=5.5, color=style.INK2, rotation=90)
    handles = [Line2D([], [], color=keys.style_in_group("noise", n)["color"],
                      marker=keys.style_in_group("noise", n)["marker"], linestyle="none",
                      markerfacecolor="none" if n in STOCHASTIC else keys.style_in_group("noise", n)["color"],
                      label=NOISE_TEX[n]) for n in NOISES[1:]]
    handles.append(Line2D([], [], color=style.GREY, linestyle="-", label="measured = CT"))
    fig.legend(handles=handles, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.01), fontsize=6)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return style.savefig(fig, name)


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
def _tex_log(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    if np.isinf(v):
        return "div"
    if v <= 0:
        return "$-\\infty$"
    return f"{math.log10(v):.1f}"


def write_floor_table(cells, path, problem, start, arm):
    sub = cells[(cells.problem == problem) & (cells.start == start) & (cells.arm == arm)]
    taus = sorted(sub.tau.unique(), reverse=True)
    noises = [n for n in NOISES if n in set(sub.noise)]
    lines = ["% generated by experiments/analyze_E5.py -- do not edit",
             "\\begin{tabular}{ll" + "c" * len(noises) + "}", "\\toprule",
             "disc. & model & " + " & ".join(NOISE_TEX[n] for n in noises) + " \\\\", "\\midrule"]
    for d in ["euler", "taylor_zhang"]:
        if d not in set(sub.disc):
            continue
        for k, mdl in enumerate([m for m in MODELS if m in set(sub.model)]):
            row = [DISC_TEX[d] if k == 0 else "", MODEL_TEX[mdl]]
            for nz in noises:
                vals = []
                for t in taus:
                    r = sub[(sub.disc == d) & (sub.model == mdl) & (sub.noise == nz) & (sub.tau == t)]
                    vals.append(_tex_log(float(r["med"].iloc[0])) if not r.empty else "--")
                row.append(" / ".join(vals))
            lines.append(" & ".join(row) + " \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    Path(path).write_text("\n".join(lines) + "\n")


def write_simple_table(path, header, rows, align=None, note=None):
    align = align or ("l" * len(header))
    lines = ["% generated by experiments/analyze_E5.py -- do not edit"]
    if note:
        lines.append(f"% {note}")
    lines += [f"\\begin{{tabular}}{{{align}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    lines += [" & ".join(str(c) for c in r) + " \\\\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    Path(path).write_text("\n".join(lines) + "\n")


def _range_tex(s):
    """Short flag for the asymptotic-range selection of a slope row."""
    r = str(s.get("range", ""))
    tags = []
    if "2 points" in r:
        tags.append("2-pt")
    if "pre-asymptotic" in r:
        tags.append("pre-asym. dropped")
    if "round-off" in r:
        tags.append("round-off dropped")
    if s.get("n_transient_excluded"):
        tags.append(f"{s['n_transient_excluded']} transient")
    if "fewer than 2" in r:
        tags.append("n/a")
    return ", ".join(tags) if tags else "ok"


# ---------------------------------------------------------------------------
# review additions: harmonic arm, unit check, censoring and transient summaries
# ---------------------------------------------------------------------------
def harmonic_arm(cells):
    """Harmonic noise, near start: measured floors of ZNN, HNTZNN (Li) and the
    linear-activation HNTZNN (matched / 10 % mismatched internal model) with
    the CT and the exact DT noise-only predictions of the matched linear one."""
    sub = cells[(cells.start == "near") & (cells.noise == "harmonic")
                & cells.problem.isin(["matrix_inversion", "linear_system", "linear_system_static"])]
    out = []
    for (prob, disc, arm, tau), g in sub.groupby(["problem", "disc", "arm", "tau"]):
        g = g.set_index("model")
        row = {"problem": prob, "disc": disc, "arm": arm, "tau": float(tau)}
        for m in ["znn", "hntznn", "hntznn_lin", "hntznn_lin_mis"]:
            if m in g.index:
                row[f"meas_{m}"] = float(g.loc[m, "meas"])
                row[f"tl_{m}"] = bool(g.loc[m, "transient_limited"])
        for m in ["znn", "hntznn_lin", "hntznn_lin_mis"]:
            if m in g.index:
                row[f"ct_{m}"] = float(g.loc[m, "ct"])
                row[f"dt_{m}"] = float(g.loc[m, "dt"])
        if "meas_hntznn_lin" in row and "meas_znn" in row:
            row["lin_over_znn"] = row["meas_hntznn_lin"] / row["meas_znn"]
        if "meas_hntznn_lin_mis" in row and "meas_znn" in row:
            row["lin_mis_over_znn"] = row["meas_hntznn_lin_mis"] / row["meas_znn"]
        if "dt_hntznn_lin" in row and row.get("ct_hntznn_lin", 0) > 0:
            row["lin_dt_over_ct"] = row["dt_hntznn_lin"] / row["ct_hntznn_lin"]
        out.append(row)
    return out


def unit_check(cells):
    """Paired unit check (noise-free, start radius 1e-4): floor with the error
    expressed in units of 1e-3 (E x 1e3) over the floor in the original unit,
    both in the ORIGINAL unit, and the ratio of each to linear ZNN."""
    a = cells[(cells.start == "unit_Ex1") & (cells.noise == "none")]
    b = cells[(cells.start == "unit_Ex1e3") & (cells.noise == "none")]
    if a.empty or b.empty:
        return []
    key = ["problem", "model", "disc", "arm", "tau"]
    m = a[key + ["med", "frac_div"]].merge(b[key + ["med", "frac_div"]], on=key, suffixes=("_Ex1", "_Ex1e3"))
    z = m[m.model == "znn"].set_index(["problem", "disc", "arm", "tau"])
    out = []
    for r in m.itertuples(index=False):
        zr = z.loc[(r.problem, r.disc, r.arm, r.tau)] if (r.problem, r.disc, r.arm, r.tau) in z.index else None
        if zr is not None and (r.problem == "linear_system_static" or not float(zr["med_Ex1"]) > 1e-10):
            zr = None          # static control: the ZNN floor is round-off / transient -> no ratio
        f1, f2 = float(r.med_Ex1), float(r.med_Ex1e3)
        out.append({"problem": r.problem, "model": r.model, "disc": r.disc, "arm": r.arm, "tau": float(r.tau),
                    "floor_Ex1": f1, "floor_Ex1e3": f2, "frac_div_Ex1": float(r.frac_div_Ex1),
                    "frac_div_Ex1e3": float(r.frac_div_Ex1e3),
                    "ratio_Ex1e3_over_Ex1": f2 / f1 if np.isfinite(f1) and f1 > 0 else np.nan,
                    "over_znn_Ex1": f1 / float(zr["med_Ex1"]) if zr is not None else np.nan,
                    "over_znn_Ex1e3": f2 / float(zr["med_Ex1e3"]) if zr is not None else np.nan})
    return out


def censoring(df, cells):
    """Censoring line (REVIEW_CHECKLIST item 2): diverged / failed trajectories
    and cells, by model x start; how medians treat them."""
    by = {}
    for (mdl, st), g in df.groupby(["model", "start"]):
        by[f"{mdl}|{st}"] = {"n_traj": int(len(g)), "n_diverged": int((g.status == "diverged").sum()),
                             "n_failed": int((~g.status.isin(["ok", "diverged"])).sum())}
    return {"n_traj": int(len(df)), "n_diverged": int((df.status == "diverged").sum()),
            "n_failed": int((~df.status.isin(["ok", "diverged"])).sum()),
            "n_cells": int(len(cells)), "n_cells_median_div": int(np.isinf(cells["med"].astype(float)).sum()),
            "n_cells_partial_div": int(cells["partial_div"].sum()), "by_model_start": by}


def transient_summary(cells):
    """How many linear-model cells are transient-limited at T = 6 s, by model x arm."""
    lin = cells[cells.transient_checked.astype(bool)]
    out = {"n_checked": int(len(lin)), "n_transient_limited": int(lin.transient_limited.sum()), "by_model_arm": {}}
    for (mdl, arm), g in lin.groupby(["model", "arm"]):
        out["by_model_arm"][f"{mdl}|{arm}"] = float(g.transient_limited.mean())
    return out


def _fmt(v, d=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    if np.isinf(v):
        return "$\\infty$"
    if abs(v) < 1e-10:
        return "0"
    if v != 0 and (abs(v) >= 1e3 or abs(v) < 1e-2):
        e = int(math.floor(math.log10(abs(v))))
        return f"${v / 10 ** e:.{d - 1}f}\\!\\times\\!10^{{{e}}}$"
    return f"{v:.{d}f}"


def _fmt_ci(est, lo, hi, d=2):
    return f"{_fmt(est, d)} [{_fmt(lo, d)}, {_fmt(hi, d)}]"


# ---------------------------------------------------------------------------
# headline analyses
# ---------------------------------------------------------------------------
def traj_values(df, problem, start, model, noise, disc, arm, tau, value="floor"):
    s = df[(df.problem == problem) & (df.start == start) & (df.model == model) & (df.noise == noise)
           & (df.disc == disc) & (df.arm == arm) & np.isclose(df.tau, tau)]
    return s.set_index("pair")[value]


def paired(df, a, b, value="floor"):
    """Trajectory-paired values (pair = seed:x0_index) and the seed of each pair."""
    va, vb = traj_values(df, *a, value=value), traj_values(df, *b, value=value)
    common = va.index.intersection(vb.index)
    seeds = np.array([str(p).split(":")[0] for p in common])
    return va.loc[common].to_numpy(float), vb.loc[common].to_numpy(float), seeds


HEADLINE_SET = [("ieznn", "constant", "CT: rejected (ratio 1)"),
                ("znn", "constant", "CT: floor c sqrt(m)/gamma (ratio >> 1)"),
                ("dieznn", "linear", "CT (t -> inf): rejected; at T = 6 s: e^-t transient"),
                ("ieznn", "linear", "CT: floor 4 c' sqrt(m)/gamma^2 (ratio >> 1)"),
                ("hntznn", "harmonic", "CT: rejected (ratio 1)"),
                ("hntznn_lin", "harmonic", "CT: rejected (ratio 1)"),
                ("hntznn_lin_mis", "harmonic", "CT: residual (10 % detuned internal model)"),
                ("sptc_an_foznn", "constant", "CT: rejected (sliding)")]


def headline(df, cells):
    out = []
    taus = sorted(df.tau.unique())
    t_small = taus[0]
    t_mid = 1e-3 if any(np.isclose(taus, 1e-3)) else taus[len(taus) // 2]
    primary = [p for p in ["matrix_inversion", "linear_system"] if p in set(df.problem)]
    have = set(zip(df.model, df.noise))

    def add(label, a, b, value="floor", expect=""):
        num, den, seeds = paired(df, a, b, value)
        est, lo, hi, n, k = boot_ratio(num, den, seeds)
        out.append({"label": label, "num": list(map(str, a)), "den": list(map(str, b)), "value": value,
                    "ratio": est, "ci_lo": lo, "ci_hi": hi, "n": n, "n_seeds": k, "expect": expect})

    for prob in primary:
        for disc in ["euler", "taylor_zhang"]:
            for t in sorted({t_small, t_mid}):
                arm = "h0.1"
                for mdl, nz, exp in HEADLINE_SET:
                    if (mdl, nz) not in have:
                        continue
                    for start in ["rand", "near"]:
                        add(f"{MODEL_TEX[mdl]} {nz}/none, {prob}, {disc}, {arm}, tau={t:g}, {start}",
                            (prob, start, mdl, nz, disc, arm, t), (prob, start, mdl, "none", disc, arm, t),
                            expect=exp)
                    # fixed-gamma counterparts (g2 only for SPTC: its published gain)
                    for garm in ["g10"] + (["g2"] if mdl == "sptc_an_foznn" else []):
                        add(f"{MODEL_TEX[mdl]} {nz}/none, {prob}, {disc}, {garm}, tau={t:g}, near",
                            (prob, "near", mdl, nz, disc, garm, t), (prob, "near", mdl, "none", disc, garm, t),
                            expect=exp)
    return out


def slopes(df, cells):
    rows = []
    taus = sorted(df.tau.unique())
    if len(taus) < 2:
        return rows
    extra = [("znn", "none"), ("znn_sbp", "none"), ("hntznn", "none"), ("sptc_an_foznn", "none"),
             ("znn", "random_walk"), ("ieznn", "random_walk"), ("ieznn", "gaussian"),
             ("hntznn_lin", "harmonic"), ("hntznn_lin_mis", "harmonic"), ("hntznn_lin", "none")]
    for prob in [p for p in ["matrix_inversion", "linear_system", "linear_system_static"] if p in set(df.problem)]:
        for mdl, nz in SELECTED + extra:
            for disc in ["euler", "taylor_zhang"]:
                for arm in ["h0.1", "g10"]:
                    for start in ["near"]:
                        if prob == "linear_system_static" and nz == "none":
                            continue        # round-off only
                        s = df[(df.problem == prob) & (df.start == start) & (df.model == mdl) & (df.noise == nz)
                               & (df.disc == disc) & (df.arm == arm)]
                        if s.empty:
                            continue
                        val = "floor_mean" if nz in STOCHASTIC else "floor"
                        c = cells[(cells.problem == prob) & (cells.start == start) & (cells.model == mdl)
                                  & (cells.noise == nz) & (cells.disc == disc) & (cells.arm == arm)].sort_values("tau")
                        # transient-limited points (linear models) are not floors: excluded from the fit
                        excl = c["transient_limited"].to_numpy(bool) if "transient_limited" in c else None
                        bs = boot_slope(s, value=val, excluded=excl)
                        ct_sl, _, _ = fit_slope(c["tau"], c["ct"]) if nz != "none" else (np.nan, [], 0)
                        dt_sl, _, _ = (fit_slope(c["tau"], c["dt"]) if (nz != "none" and mdl in LINEAR_MODELS)
                                       else (np.nan, [], 0))
                        rows.append({"problem": prob, "start": start, "model": mdl, "noise": nz, "disc": disc,
                                     "arm": arm, "slope": bs["slope"], "ci_lo": bs["ci_lo"], "ci_hi": bs["ci_hi"],
                                     "local": bs["local"], "n_points": bs["n_points"],
                                     "slope_all": bs["slope_all"], "range": bs["range"],
                                     "taus_used": bs["taus_used"], "n_seeds": bs["n_seeds"],
                                     "n_transient_excluded": int(np.sum(excl)) if excl is not None else 0,
                                     "ct_slope": ct_sl, "dt_slope": dt_sl})
    return rows


def kendall(cells):
    from scipy.stats import kendalltau
    rows = []
    # rankings over the models of the full noise grid (the harmonic-only arms are excluded),
    # primary start arms only; diverged = +inf = worst (ties).  tau_b is DESCRIPTIVE: 3-6
    # models per setting, settings share runs, no p-values / multiplicity control.
    sub = cells[(cells.noise != "none") & cells.model.isin(MODELS) & cells.start.isin(["rand", "near"])]

    def _tb(x, y):
        ok = x.notna() & y.notna()
        if ok.sum() < 3 or x[ok].nunique() < 2 or y[ok].nunique() < 2:
            return np.nan, int(ok.sum())
        return float(kendalltau(x[ok], y[ok], variant="b").statistic), int(ok.sum())

    for key, g in sub.groupby(["problem", "start", "disc", "arm", "tau", "noise"]):
        g = g.set_index("model")
        x = g["ct"].astype(float)
        y = g["meas"].astype(float).replace(np.inf, BIG)
        tb, nm = _tb(x, y)
        if not np.isfinite(tb):
            continue
        tba, _ = _tb(g["ct_plus_disc"].astype(float).replace(np.inf, BIG), y)
        # without models whose value at T = 6 s is transient-limited (not a floor)
        keep = ~(g["transient_limited"].astype(bool) | g["none_transient_limited"].astype(bool))
        tbs, nms = _tb(x[keep], y[keep])
        rows.append(dict(zip(["problem", "start", "disc", "arm", "tau", "noise"], key), tau_b=float(tb),
                         tau_b_additive=float(tba), tau_b_steady=float(tbs), n_models=nm,
                         n_models_steady=nms, n_diverged=int(np.isinf(g["meas"].astype(float)).sum())))
    # ranking stability between the largest and the smallest tau
    stab = []
    taus = sorted(cells.tau.unique())
    if len(taus) >= 2:
        for key, g in cells[cells.model.isin(MODELS) & cells.start.isin(["rand", "near"])].groupby(
                ["problem", "start", "disc", "arm", "noise"]):
            a = g[np.isclose(g.tau, taus[-1])].set_index("model")["meas"].replace(np.inf, BIG)
            b = g[np.isclose(g.tau, taus[0])].set_index("model")["meas"].replace(np.inf, BIG)
            common = a.dropna().index.intersection(b.dropna().index)
            if len(common) < 3 or a[common].nunique() < 2 or b[common].nunique() < 2:
                continue
            stab.append(dict(zip(["problem", "start", "disc", "arm", "noise"], key),
                             tau_b=float(kendalltau(a[common], b[common], variant="b").statistic)))
    return rows, stab


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return None if math.isnan(v) else ("inf" if math.isinf(v) else v)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def main(argv=None):
    global BOOT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--figdir", default=str(ROOT / "figures"))
    ap.add_argument("--tabdir", default=str(ROOT / "results" / "tables"))
    ap.add_argument("--nboot", type=int, default=BOOT)
    args = ap.parse_args(argv)
    BOOT = int(args.nboot)
    res = Path(args.results)
    figdir, tabdir = Path(args.figdir), Path(args.tabdir)
    figdir.mkdir(parents=True, exist_ok=True)
    tabdir.mkdir(parents=True, exist_ok=True)
    style.FIG_DIR = figdir                      # style.savefig writes into FIG_DIR

    df = load(res)
    status = df["status"].value_counts().to_dict()
    cells = attach_predictions(cell_table(df))
    cells.to_csv(res / "cells.csv", index=False)

    # ---------------- figures ----------------
    figs = []
    probs = list(dict.fromkeys(df.problem))
    main_prob = "matrix_inversion" if "matrix_inversion" in probs else probs[0]
    variants = [(main_prob, "rand", "h0.1", "fig_E5_floor_heatmap"),
                (main_prob, "rand", "h0.3", "fig_E5_floor_heatmap_h0.3"),
                (main_prob, "rand", "g10", "fig_E5_floor_heatmap_gamma10"),
                (main_prob, "rand", "g2", "fig_E5_floor_heatmap_gamma2"),
                (main_prob, "near", "h0.1", "fig_E5_floor_heatmap_near"),
                ("linear_system", "rand", "h0.1", "fig_E5_floor_heatmap_linear_system"),
                ("linear_system_static", "rand", "h0.1", "fig_E5_floor_heatmap_static")]
    for prob, start, arm, name in variants:
        p = heatmap_figure(cells, prob, start, arm, name, kind="floor")
        if p:
            figs.append(str(p))
    for prob, start, arm, name in [(main_prob, "rand", "h0.1", "fig_E5_penalty_heatmap"),
                                   (main_prob, "near", "g10", "fig_E5_penalty_heatmap_gamma10_near"),
                                   (main_prob, "near", "h0.1", "fig_E5_ct_gap_heatmap"),
                                   (main_prob, "near", "g10", "fig_E5_ct_gap_heatmap_gamma10")]:
        p = heatmap_figure(cells, prob, start, arm, name, kind="penalty" if "penalty" in name else "ct_gap")
        if p:
            figs.append(str(p))
    for arm, name in [("h0.1", "fig_E5_floor_vs_tau"), ("g10", "fig_E5_floor_vs_tau_gamma10")]:
        p = floor_vs_tau_figure(cells, main_prob, "near", arm, name)
        if p:
            figs.append(str(p))
    # harmonic arm (review): ZNN, HNTZNN (Li), HNTZNN-lin matched / 10 % mismatched
    for arm, name in [("g10", "fig_E5_harmonic_arm_gamma10"), ("h0.1", "fig_E5_harmonic_arm_h0.1")]:
        p = floor_vs_tau_figure(cells, main_prob, "near", arm, name, selected=SELECTED_HARMONIC,
                                title=f"{PROBLEM_FIG.get(main_prob, main_prob)}, harmonic noise "
                                      f"$\\sin(2\\pi t+1)$, near-solution $x_0$, fixed {arm_tex(arm)}")
        if p:
            figs.append(str(p))
    p = ct_scatter_figure(cells, "linear_system", "near", "fig_E5_ct_vs_measured")
    if p:
        figs.append(str(p))
    p = ct_scatter_figure(cells, "linear_system_static", "near", "fig_E5_ct_vs_measured_static")
    if p:
        figs.append(str(p))

    # ---------------- analyses ----------------
    heads = headline(df, cells)
    slo = slopes(df, cells)
    kt, kstab = kendall(cells)

    # static-control validation: measured vs DT prediction (linear models)
    val = cells[(cells.problem == "linear_system_static") & cells.model.isin(LINEAR_MODELS)
                & (cells.noise != "none") & cells.start.isin(["rand", "near"])].copy()
    val["meas_over_dt"] = val["meas"] / val["dt"]
    val["meas_over_ct"] = val["meas"] / val["ct"]
    vfin = val[np.isfinite(val.meas_over_dt) & (val.dt > 1e-13)]
    dt_valid = {
        "n_cells": int(len(vfin)),
        "median_log10_meas_over_dt": float(np.median(np.log10(vfin.meas_over_dt))) if len(vfin) else None,
        "iqr_log10_meas_over_dt": ([float(np.quantile(np.log10(vfin.meas_over_dt), 0.25)),
                                    float(np.quantile(np.log10(vfin.meas_over_dt), 0.75))] if len(vfin) else None),
        "by_noise": {nz: float(np.median(np.log10(g.meas_over_dt))) for nz, g in vfin.groupby("noise")},
    }
    gz = val[(val.model == "znn") & (val.noise == "gaussian") & (val.disc == "euler") & val.arm.str.startswith("h")
             & (val.start == "near")]
    gauss_check = [{"arm": r.arm, "tau": r.tau, "meas": r.meas, "dt": r.dt, "ct": r.ct,
                    "meas_over_dt": r.meas / r.dt if r.dt else np.nan,
                    "dt_over_ct": r.dt / r.ct if r.ct else np.nan,
                    "theory_dt_over_ct": math.sqrt(2.0 / (2.0 - float(r.h)))} for r in gz.itertuples()]

    # CT-agreement statistics per model x noise (start = near, time-varying problems)
    agree = []
    for (mdl, nz), g in cells[(cells.start == "near") & (cells.noise != "none")
                              & cells.problem.isin(["matrix_inversion", "linear_system"])].groupby(["model", "noise"]):
        r = (g["meas"] / g["ct_plus_disc"]).replace([np.inf, -np.inf], np.nan).dropna()
        r = r[r > 0]
        agree.append({"model": mdl, "noise": nz, "n_cells": int(len(r)),
                      "n_cells_total": int(len(g)),
                      "n_cells_div": int(np.isinf(g["meas"].astype(float)).sum()),     # censored, not in median
                      "n_cells_no_ct": int(g["ct"].isna().sum()),
                      "n_transient_limited": int(g["transient_limited"].sum()),
                      "median_log10_gap": float(np.median(np.log10(r))) if len(r) else None,
                      "frac_within_x3": float(np.mean(np.abs(np.log10(r)) < math.log10(3))) if len(r) else None,
                      "frac_div": float(g["frac_div"].mean())})

    harm = harmonic_arm(cells)
    unit = unit_check(cells)
    cens = censoring(df, cells)
    trans = transient_summary(cells)

    # ---------------- tables ----------------
    tabs = []
    for prob, start, arm, nm in [(main_prob, "rand", "h0.1", "E5_floors.tex"),
                                 (main_prob, "near", "g10", "E5_floors_gamma10_near.tex"),
                                 ("linear_system", "rand", "h0.1", "E5_floors_linear_system.tex"),
                                 ("linear_system_static", "rand", "h0.1", "E5_floors_static.tex")]:
        if prob in probs:
            write_floor_table(cells, tabdir / nm, prob, start, arm)
            tabs.append(str(tabdir / nm))
    # compact headline table: rows = (model, noise) x arm, columns = problem x disc, tau = 1e-3 (or nearest)
    taus_all = sorted(df.tau.unique())
    t_tab = min(taus_all, key=lambda t: abs(math.log10(t) + 3.0))
    combos = [(p_, d_) for p_ in ["matrix_inversion", "linear_system"] if p_ in probs for d_ in ["euler", "taylor_zhang"]]
    hdr = ["comparison", "arm"] + [f"{'MI' if p_ == 'matrix_inversion' else 'LS'}, {DISC_TEX[d_]}" for p_, d_ in combos]
    hrows = []
    for mdl, nz in [("ieznn", "constant"), ("znn", "constant"), ("dieznn", "linear"), ("ieznn", "linear"),
                    ("hntznn", "harmonic"), ("hntznn_lin", "harmonic"), ("hntznn_lin_mis", "harmonic"),
                    ("sptc_an_foznn", "constant")]:
        arms_ = [("rand", "h0.1"), ("near", "h0.1"), ("near", "g10")] + (
            [("near", "g2")] if mdl == "sptc_an_foznn" else [])
        for start, arm in arms_:
            row = [f"{MODEL_TEX[mdl]}, {NOISE_TEX[nz]}" if (start, arm) == ("rand", "h0.1") else "",
                   f"{start}, {arm_tex(arm)}"]
            for p_, d_ in combos:
                hh = [h for h in heads if h["num"][:7] == [p_, start, mdl, nz, d_, arm, str(t_tab)]]
                if not hh:
                    row.append("--")
                    continue
                h = hh[0]
                row.append("div" if np.isinf(h["ratio"]) else _fmt_ci(h["ratio"], h["ci_lo"], h["ci_hi"]))
            hrows.append(row)
    n_seeds = int(df["seed"].nunique())
    write_simple_table(tabdir / "E5_headline.tex", hdr, hrows, align="ll" + "c" * len(combos),
                       note=f"ratio of medians [95 % seed-cluster bootstrap CI over {n_seeds} problem seed(s); "
                            f"'--' in brackets: fewer than {MIN_CLUSTERS} seeds], tau = {t_tab:g}")
    tabs.append(str(tabdir / "E5_headline.tex"))
    srows = []
    for s in slo:
        if s["problem"] != main_prob and s["problem"] != "linear_system_static":
            continue
        srows.append([s["problem"].replace("_", "\\_"), MODEL_TEX[s["model"]], NOISE_TEX[s["noise"]],
                      DISC_TEX[s["disc"]], arm_tex(s["arm"]), _fmt_ci(s["slope"], s["ci_lo"], s["ci_hi"]),
                      _fmt(s["slope_all"]), ", ".join(_fmt(v) for v in s["local"]), _range_tex(s),
                      _fmt(s["ct_slope"]), _fmt(s["dt_slope"])])
    write_simple_table(tabdir / "E5_slopes.tex", ["problem", "model", "noise", "disc.", "gain",
                                                   "asympt. slope [95\\% CI]", "all-points slope", "local slopes",
                                                   "range", "CT slope", "DT slope"], srows)
    tabs.append(str(tabdir / "E5_slopes.tex"))
    kdf = pd.DataFrame(kt)
    krows = []
    if not kdf.empty:
        for nz, g in list(kdf.groupby("noise")) + [("all", kdf)]:
            krows.append([NOISE_TEX.get(nz, nz), len(g), _fmt(float(g.tau_b.median())),
                          f"[{_fmt(float(g.tau_b.quantile(0.25)))}, {_fmt(float(g.tau_b.quantile(0.75)))}]",
                          _fmt(float(g.tau_b_additive.median())), _fmt(float(g.tau_b_steady.median()))])
    write_simple_table(tabdir / "E5_kendall.tex", ["noise", "settings", "median $\\tau_b$ (CT vs discrete)", "IQR",
                                                    "median $\\tau_b$ (CT + noise-free floor)",
                                                    "median $\\tau_b$ (steady cells only)"],
                       krows, align="lrrrrr", note="Kendall tau_b values are descriptive (3-6 models per setting, "
                                                   "overlapping settings); no p-values are computed")
    tabs.append(str(tabdir / "E5_kendall.tex"))
    if unit:
        urows = [[u["problem"].replace("_", "\\_"), MODEL_TEX[u["model"]], DISC_TEX[u["disc"]], arm_tex(u["arm"]),
                  _fmt_tau(u["tau"]), _fmt(u["floor_Ex1"]), _fmt(u["floor_Ex1e3"]), _fmt(u["ratio_Ex1e3_over_Ex1"]),
                  _fmt(u["over_znn_Ex1"]), _fmt(u["over_znn_Ex1e3"])] for u in unit
                 if u["problem"] in (main_prob, "linear_system_static")]
        write_simple_table(tabdir / "E5_unit_check.tex",
                           ["problem", "model", "disc.", "gain", "$\\tau$", "floor ($E$)", "floor ($10^3E$)",
                            "ratio", "/ZNN ($E$)", "/ZNN ($10^3E$)"], urows,
                           note="noise-free MSSRE in the ORIGINAL unit; start x*(0) + 1e-4 U[-1,1]; "
                                "E x 1e3 = error expressed in units of 1e-3")
        tabs.append(str(tabdir / "E5_unit_check.tex"))
    if harm:
        hr_ = [[h["problem"].replace("_", "\\_"), DISC_TEX[h["disc"]], arm_tex(h["arm"]), _fmt_tau(h["tau"])]
               + [_fmt(h.get(f"meas_{m}")) for m in ["znn", "hntznn", "hntznn_lin", "hntznn_lin_mis"]]
               + [_fmt(h.get("ct_hntznn_lin")), _fmt(h.get("dt_hntznn_lin"))] for h in harm]
        write_simple_table(tabdir / "E5_harmonic_arm.tex",
                           ["problem", "disc.", "gain", "$\\tau$", "ZNN", "HNTZNN (Li)", "HNTZNN-lin",
                            "HNTZNN-lin (mis.)", "CT lin", "DT lin"], hr_,
                           note="median MSSRE under harmonic noise sin(2 pi t + 1), near start; "
                                "CT / DT: noise-only predictions of HNTZNN-lin (matched)")
        tabs.append(str(tabdir / "E5_harmonic_arm.tex"))

    # ---------------- summary ----------------
    runs = df.drop_duplicates("run_id")
    cost = {
        "median_wall_per_step_by_tau": {f"{t:g}": float(g["wall_per_step"].median())
                                        for t, g in df[df.status == "ok"].groupby("tau")},
        "max_peak_rss_mb": float(df["peak_rss_mb"].max()),
        "n_specs": int(df["run_id"].nunique()), "n_trajectories": int(len(df)),
        "sum_wall_s": float(runs["wall"].sum()),
        "sum_steps": float(runs["steps"].sum()) if "steps" in runs else None,
        "wall_per_step_overall": (float(runs["wall"].sum() / runs["steps"].sum())
                                  if "steps" in runs and runs["steps"].sum() > 0 else None),
    }
    kstab_df = pd.DataFrame(kstab)

    def pick(label_prefix):
        return [h for h in heads if h["label"].startswith(label_prefix)]

    summary = {
        "experiment": "E5", "results": str(res), "status_counts": status, "cost": cost,
        "key_numbers": key_numbers(cells, slo),
        "key_numbers_censoring": dict(KN_CENSORING),
        "taus": sorted(map(float, df.tau.unique())), "problems": probs,
        "headline_ratios": heads, "slopes": slo,
        "kendall_ct_vs_measured": {"n_settings": int(len(kdf)),
                                   "median_tau_b": float(kdf.tau_b.median()) if len(kdf) else None,
                                   "median_tau_b_additive": float(kdf.tau_b_additive.median()) if len(kdf) else None,
                                   "median_tau_b_steady": float(kdf.tau_b_steady.median()) if len(kdf) else None,
                                   "by_noise": ({nz: float(g.tau_b.median()) for nz, g in kdf.groupby("noise")}
                                                if len(kdf) else {}),
                                   "by_arm": ({a: float(g.tau_b.median()) for a, g in kdf.groupby("arm")}
                                              if len(kdf) else {}),
                                   "note": "descriptive; 3-6 models per setting; settings overlap; no p-values"},
        "n_seeds": n_seeds, "ci_method": (f"seed-cluster bootstrap ({BOOT} resamples, seed {BOOT_SEED}); "
                                          f"CIs only with >= {MIN_CLUSTERS} problem seeds"),
        "harmonic_arm": harm, "unit_check": unit, "censoring": cens, "transient": trans,
        "kendall_rank_stability_tau_max_vs_min": {"n": int(len(kstab_df)),
                                                  "median_tau_b": float(kstab_df.tau_b.median()) if len(kstab_df) else None},
        "static_control_dt_validation": dt_valid, "gaussian_znn_euler_check": gauss_check,
        "ct_agreement": agree, "divergence_by_model_start": {
            f"{m}|{s}": float(g["frac_div"].mean()) for (m, s), g in cells.groupby(["model", "start"])},
        "figures": figs, "tables": tabs,
    }
    (res / "summary.json").write_text(json.dumps(_clean(summary), indent=1))
    write_summary_md(res / "SUMMARY.md", summary, cells, main_prob)
    print(f"E5 analysis: {len(df)} trajectories, {len(cells)} cells, {len(figs)} figures, {len(tabs)} tables")
    print(f"-> {res / 'summary.json'}, {res / 'SUMMARY.md'}")
    return 0


def write_summary_md(path, S, cells, main_prob):
    f = _fmt_md
    L = ["# E5 -- noise robustness under discretization: summary", "",
         f"Results: `{S['results']}`; taus = {S['taus']}; problems = {S['problems']}; "
         f"problem seeds = {S['n_seeds']}.",
         f"Status counts: {S['status_counts']}.", ""]
    c = S["censoring"]
    L += ["## Censoring", "",
          f"{c['n_diverged']} of {c['n_traj']} trajectories diverged, {c['n_failed']} failed (error / timeout / "
          f"skipped). Diverged trajectories count as +inf (worst case) in every median, bootstrap and ranking; "
          f"they are never dropped. {c['n_cells_median_div']} of {c['n_cells']} cells have a diverged median "
          f"(shown as 'div'), and {c['n_cells_partial_div']} cells have a finite median with some (< 50 %) "
          f"diverged trajectories (marked '*' in the heatmaps). Key-number medians report their censored "
          f"counts in `summary.json` (`key_numbers_censoring`). Divergence by model and start:", ""]
    L += ["| model | start | trajectories | diverged | failed |", "|---|---|---|---|---|"]
    for k_, v_ in c["by_model_start"].items():
        m_, s_ = k_.split("|")
        L.append(f"| {m_} | {s_} | {v_['n_traj']} | {v_['n_diverged']} | {v_['n_failed']} |")
    tr = S["transient"]
    L += ["", "## Transient-limited cells (not floors)", "",
          f"For the linear models the value at T = 6 s is split into its decaying parts (exact discrete "
          f"initial-state response x median ||E(0)||, and the decaying part of the CT noise response). "
          f"{tr['n_transient_limited']} of {tr['n_checked']} checked cells are transient-limited (decaying "
          f"part > 10 % of the value); they are marked with a dagger in the heatmaps and excluded from the "
          f"slope fits and from the 'steady' Kendall variant. Fraction by model and gain arm: "
          + ", ".join(f"{k_}: {f(v_)}" for k_, v_ in tr["by_model_arm"].items() if v_ > 0) + ".", ""]
    L += [f"## Headline floor ratios (noise / noise-free; 95 % CI = seed-cluster bootstrap, "
          f"needs >= {MIN_CLUSTERS} seeds)", "",
          "| comparison | ratio [95% CI] | pairs | seeds | CT expectation |", "|---|---|---|---|---|"]
    for h in S["headline_ratios"]:
        if ", near" in h["label"] and "g10" not in h["label"] and "g2" not in h["label"] \
                and "sptc" not in h["label"].lower() and "hntznn" not in h["label"].lower():
            continue           # keep the table short: rand arm + g10 + the locally stable models' near arm
        rt = "div" if (isinstance(h["ratio"], float) and math.isinf(h["ratio"])) else \
            f"{f(h['ratio'])} [{f(h['ci_lo'])}, {f(h['ci_hi'])}]"
        L.append(f"| {h['label']} | {rt} | {h['n']} | {h.get('n_seeds', '--')} | {h['expect']} |")
    L += ["", "## Floor-vs-tau slopes (near start)", "",
          "Asymptotic slope = LS over the asymptotic range (points at round-off, transient-limited or "
          "pre-asymptotic dropped; 'range' column), with the all-points LS slope and the local slopes; CI = "
          "seed-cluster bootstrap. With 2 tau values (mini) every fit is a 2-point fit.", "",
          "| problem | model | noise | disc | gain | asympt. slope [95% CI] | all-points | local | range | "
          "CT slope | DT slope |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in S["slopes"]:
        if s["problem"] not in (main_prob, "linear_system_static"):
            continue
        L.append(f"| {s['problem']} | {s['model']} | {s['noise']} | {s['disc']} | {s['arm']} | "
                 f"{f(s['slope'])} [{f(s['ci_lo'])}, {f(s['ci_hi'])}] | {f(s['slope_all'])} | "
                 f"{', '.join(f(v) for v in s['local'])} | {_range_tex(s)} | {f(s['ct_slope'])} | "
                 f"{f(s['dt_slope'])} |")
    k = S["kendall_ct_vs_measured"]
    L += ["", "## Rankings (descriptive)", "",
          f"Kendall tau_b between the CT-predicted and the measured model ranking (6 full-grid models, "
          f"diverged = worst, CT ties at exact rejection kept as ties): median {f(k['median_tau_b'])} over "
          f"{k['n_settings']} settings; by noise: "
          + ", ".join(f"{n}: {f(v)}" for n, v in k["by_noise"].items())
          + "; by gain arm: " + ", ".join(f"{n}: {f(v)}" for n, v in k["by_arm"].items()) + ". "
          f"CT + measured noise-free floor: {f(k['median_tau_b_additive'])}; steady cells only (transient-"
          f"limited models dropped): {f(k['median_tau_b_steady'])}. "
          f"Ranking stability (largest vs smallest tau): median tau_b "
          f"{f(S['kendall_rank_stability_tau_max_vs_min']['median_tau_b'])} "
          f"(n = {S['kendall_rank_stability_tau_max_vs_min']['n']}). {k['note']}.", ""]
    dv = S["static_control_dt_validation"]
    L += ["## Static-control validation of the discrete noise response (linear models)", "",
          f"median log10(measured / DT prediction) = {f(dv['median_log10_meas_over_dt'])} "
          f"(IQR {dv['iqr_log10_meas_over_dt']}, {dv['n_cells']} cells); by noise: "
          + ", ".join(f"{n}: {f(v)}" for n, v in dv["by_noise"].items()) + ".", ""]
    L += ["ZNN + Euler + Gaussian (static control): measured / DT and DT / CT vs the theory "
          "sqrt(2/(2-h)):", ""]
    for g in S["gaussian_znn_euler_check"]:
        L.append(f"- {g['arm']}, tau={g['tau']:g}: meas/DT = {f(g['meas_over_dt'])}, "
                 f"DT/CT = {f(g['dt_over_ct'])} (theory {f(g['theory_dt_over_ct'])})")
    L += ["", "## Harmonic arm: internal model matched / 10 % mismatched, Li vs linear activation (near start)", "",
          "| problem | disc | gain | tau | ZNN | HNTZNN (Li) | HNTZNN-lin | HNTZNN-lin mis. | CT lin | DT lin | "
          "lin/ZNN | mis/ZNN |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for h in S["harmonic_arm"]:
        L.append(f"| {h['problem']} | {h['disc']} | {h['arm']} | {h['tau']:g} | {f(h.get('meas_znn'))} | "
                 f"{f(h.get('meas_hntznn'))} | {f(h.get('meas_hntznn_lin'))}{' (T)' if h.get('tl_hntznn_lin') else ''}"
                 f" | {f(h.get('meas_hntznn_lin_mis'))} | {f(h.get('ct_hntznn_lin'))} | {f(h.get('dt_hntznn_lin'))} | "
                 f"{f(h.get('lin_over_znn'))} | {f(h.get('lin_mis_over_znn'))} |")
    L += ["", "(T) = transient-limited at T = 6 s.", ""]
    if S["unit_check"]:
        L += ["## Unit check of the power-type activations (noise-free, start radius 1e-4)", "",
              "Floor (MSSRE, ORIGINAL unit) with E expressed in the original unit and in units of 1e-3 "
              "(E x 1e3, 'metres -> millimetres'); linear ZNN is the control (must be identical).", "",
              "| problem | model | disc | gain | tau | floor (E) | floor (1e3 E) | ratio | /ZNN (E) | /ZNN (1e3 E) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for u in S["unit_check"]:
            if u["problem"] not in (main_prob, "linear_system_static"):
                continue
            L.append(f"| {u['problem']} | {u['model']} | {u['disc']} | {u['arm']} | {u['tau']:g} | "
                     f"{f(u['floor_Ex1'])} | {f(u['floor_Ex1e3'])} | {f(u['ratio_Ex1e3_over_Ex1'])} | "
                     f"{f(u['over_znn_Ex1'])} | {f(u['over_znn_Ex1e3'])} |")
        L.append("")
    L += ["## CT agreement (near start, time-varying problems): measured / (CT + noise-free floor)", "",
          "| model | noise | cells used | diverged (censored) | no CT pred. | transient-lim. | median log10 gap | "
          "frac within x3 |", "|---|---|---|---|---|---|---|---|"]
    for a in S["ct_agreement"]:
        L.append(f"| {a['model']} | {a['noise']} | {a['n_cells']} | {a['n_cells_div']} | {a['n_cells_no_ct']} | "
                 f"{a['n_transient_limited']} | {f(a['median_log10_gap'])} | {f(a['frac_within_x3'])} |")
    L += ["", "## Confounds (see experiments/E5_NOTES.md, Review log)", "",
          "- Equal nominal gamma is not equal effective gain: IEZNN lam = gamma^2/4 puts its double pole at "
          "-gamma/2; DIEZNN (xi = gamma) has the e-gain 2 xi + 1 and a pole fixed at -1; HNTZNN's internal-model "
          "gain lam = 1 does not scale with gamma (learning rate ~ omega^2 lam / (2 gamma)); SBP / Li gain "
          "~ gamma / (2 sqrt|e|) near the floor; SPTC-AN-FOZNN's quadratic term scales with gamma.",
          "- Information advantage: SPTC-AN-FOZNN is given the true noise bound (Delta = 8 -> zeta = 40), "
          "HNTZNN the true noise frequency (except the mismatched arm).",
          "- Noise is injected identically in E-space (dE/dt = model + nu) for every model; internal states "
          "are not perturbed (as in the source papers), so the noise LEVEL is comparable across models.",
          "- Horizon: T = 6 s; DIEZNN's e^-t mode and HNTZNN's internal model at gamma >= 100 (fixed h, small "
          "tau) have not settled in the window [4.5, 6] s -> transient-limited, not floors.",
          "- Units: SBP, Li-HNTZNN and SPTC (quadratic term) are not scale invariant; their floors, rankings "
          "and even stability depend on the unit of E (unit-check table).", ""]
    L += ["## Key numbers", ""] + [f"- {k_}: {f(v_)}" for k_, v_ in S["key_numbers"].items()]
    L += ["", "## Interpretation", "", interpretation(S, cells, main_prob), "",
          "Figures: " + ", ".join(Path(x).name for x in S["figures"]), "",
          "Tables: " + ", ".join(Path(t).name for t in S["tables"]), ""]
    Path(path).write_text("\n".join(L))


def _fmt_md(v):
    if v is None:
        return "--"
    if isinstance(v, str):
        return v
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, float) and math.isnan(v):
        return "--"
    if isinstance(v, float) and math.isinf(v):
        return "inf"
    if abs(v) < 1e-10:
        return "0"
    if v != 0 and (abs(v) >= 1e3 or abs(v) < 1e-2):
        return f"{v:.2e}"
    return f"{v:.3g}"


KN_CENSORING = {}


def key_numbers(cells, slo):
    """Data-driven headline numbers (medians over problems x discretizations x
    taus of the time-varying problems, near start unless stated).  Medians are
    censoring-aware: +inf (diverged) counts as the worst value, NaN (no
    prediction / failed) is dropped; the counts go to KN_CENSORING."""
    tv = cells[cells.problem.isin(["matrix_inversion", "linear_system"])]
    KN_CENSORING.clear()

    def _sel(g, sel):
        for k, v in sel.items():
            g = g[g[k].isin(v) if isinstance(v, (list, tuple, set)) else np.isclose(g[k], v)
                  if isinstance(v, float) else (g[k] == v)]
        return g

    def med(col, _name=None, _frame=None, **sel):
        g = _sel(tv if _frame is None else _frame, sel)
        v = g[col].astype(float).to_numpy()
        v = v[~np.isnan(v)]
        if _name:
            KN_CENSORING[_name] = {"n_cells": int(v.size), "n_inf": int(np.isinf(v).sum())}
        return float(_med_inf(v)) if v.size else None

    def frac_div(**sel):
        g = _sel(cells, sel)
        return float(g["frac_div"].mean()) if len(g) else None

    def slope(model, noise, arm, prob=None):
        v = [r["slope"] for r in slo if r["model"] == model and r["noise"] == noise and r["arm"] == arm
             and (prob is None or r["problem"] == prob) and r["problem"] != "linear_system_static"
             and r["slope"] is not None and np.isfinite(r["slope"])]
        return float(np.median(v)) if v else None

    K = {}
    for mdl, nz in [("ieznn", "constant"), ("znn", "constant"), ("dieznn", "linear"), ("ieznn", "linear"),
                    ("znn", "linear"), ("hntznn", "harmonic"), ("hntznn_lin", "harmonic"),
                    ("hntznn_lin_mis", "harmonic"), ("znn", "harmonic"), ("znn_sbp", "constant")]:
        for arm in ["h0.1", "g10"]:
            nm = f"penalty_{mdl}_{nz}_{arm}"
            K[nm] = med("penalty", _name=nm, model=mdl, noise=nz, arm=arm, start="near")
    K["penalty_sptc_an_foznn_constant_g2"] = med("penalty", _name="penalty_sptc_an_foznn_constant_g2",
                                                 model="sptc_an_foznn", noise="constant", arm="g2", start="near")
    K["ctgap_dieznn_linear_h0.1"] = med("ct_gap", _name="ctgap_dieznn_linear_h0.1", model="dieznn",
                                        noise="linear", arm="h0.1", start="near")
    g_ = tv[(tv.model == "dieznn") & (tv.noise == "linear") & (tv.start == "near")]
    K["dieznn_linear_frac_transient_limited"] = float(g_["transient_limited"].mean()) if len(g_) else None
    lin = tv[(tv.start == "near") & tv.model.isin(["znn", "ieznn", "dieznn"]) & (tv.noise != "none")]
    r = (lin["meas"] / lin["ct_plus_disc"]).replace([np.inf, -np.inf], np.nan).dropna()
    r = r[r > 0]
    KN_CENSORING["linear_models_gap"] = {"n_cells": int(len(lin)), "n_used": int(len(r)),
                                         "n_inf": int(np.isinf(lin["meas"].astype(float)).sum())}
    K["linear_models_abs_log10_gap_median"] = float(np.median(np.abs(np.log10(r)))) if len(r) else None
    K["linear_models_frac_within_x2"] = float(np.mean(np.abs(np.log10(r)) < math.log10(2))) if len(r) else None
    for mdl in ["znn", "znn_sbp", "hntznn", "ieznn"]:
        for arm in ["h0.1", "g10"]:
            K[f"slope_none_{mdl}_{arm}"] = slope(mdl, "none", arm)
    for mdl, nz in [("znn", "constant"), ("znn", "gaussian"), ("ieznn", "linear"), ("ieznn", "random_walk"),
                    ("znn", "random_walk")]:
        K[f"slope_{mdl}_{nz}_h0.1"] = slope(mdl, nz, "h0.1")
    K["slope_znn_gaussian_g10"] = slope("znn", "gaussian", "g10")
    K["slope_hntznn_lin_harmonic_g10"] = slope("hntznn_lin", "harmonic", "g10")
    K["sptc_frac_div_fixed_h"] = frac_div(model="sptc_an_foznn", arm=["h0.1", "h0.3"])
    K["sptc_frac_div_g2_rand"] = frac_div(model="sptc_an_foznn", arm="g2", start="rand")
    K["sptc_frac_div_g2_near"] = frac_div(model="sptc_an_foznn", arm="g2", start="near")
    zeta = sptc_params()["zeta"]
    g = tv[(tv.model == "sptc_an_foznn") & (tv.noise == "none") & (tv.arm == "g2") & (tv.start == "near")
           & (tv.disc == "euler")]
    rr = (g["med"] / (zeta * g["tau"])).replace([np.inf], np.nan).dropna()
    K["sptc_floor_over_zeta_tau_euler_g2"] = float(np.median(rr)) if len(rr) else None
    K["hntznn_frac_div_rand"] = frac_div(model="hntznn", start="rand")
    K["hntznn_frac_div_near"] = frac_div(model="hntznn", start="near")
    g = tv[(tv.model == "znn_sbp") & (tv.noise == "none") & (tv.arm == "h0.1") & (tv.start == "near")
           & (tv.disc == "euler")]
    K["sbp_none_floor_euler_h0.1_median"] = float(_med_inf(g["med"].astype(float).to_numpy())) if len(g) else None
    K["sbp_chatter_prediction_h0.1_m4"] = 2.0 * (0.1 / 4.0) ** 2      # sqrt(m) (h/4)^(1/(1-r)), r = 1/2, m = 4
    K["sbp_chatter_prediction_h0.1_m16"] = 4.0 * (0.1 / 4.0) ** 2
    # harmonic arm (linear HNTZNN): measured / ZNN on the time-varying problems, DT / CT on the static control
    taus = sorted(cells.tau.unique())
    for arm in ["g10", "h0.1"]:
        for t in taus:
            hv = tv[(tv.start == "near") & (tv.noise == "harmonic") & (tv.arm == arm) & np.isclose(tv.tau, t)]
            zz = hv[hv.model == "znn"].set_index(["problem", "disc"])["meas"]
            for m_ in ["hntznn_lin", "hntznn_lin_mis", "hntznn"]:
                mm = hv[hv.model == m_].set_index(["problem", "disc"])["meas"]
                q_ = (mm / zz.reindex(mm.index)).replace([np.inf, -np.inf], np.nan).dropna()
                K[f"{m_}_over_znn_harmonic_{arm}_tau{t:g}"] = float(np.median(q_)) if len(q_) else None
            st_ = cells[(cells.problem == "linear_system_static") & (cells.start == "near")
                        & (cells.model == "hntznn_lin") & (cells.noise == "harmonic") & (cells.arm == arm)
                        & np.isclose(cells.tau, t)]
            if len(st_):
                K[f"hntznn_lin_static_dt_over_ct_{arm}_tau{t:g}"] = float(np.median(st_["dt"] / st_["ct"]))
                K[f"hntznn_lin_static_meas_over_dt_{arm}_tau{t:g}"] = float(np.median(st_["meas"] / st_["dt"]))
    # unit check: floor(E x 1e3) / floor(E), and the ordering vs ZNN
    uc = unit_check(cells)
    if uc:
        u = pd.DataFrame(uc)
        for m_ in ["znn", "znn_sbp", "hntznn"]:
            for lab, uu in [("static", u[u.problem == "linear_system_static"]),
                            ("tv", u[u.problem != "linear_system_static"])]:
                v_ = uu[uu.model == m_]["ratio_Ex1e3_over_Ex1"].replace([np.inf, -np.inf], np.nan).dropna()
                if m_ == "znn" and lab == "static":
                    continue                      # ZNN static floor = round-off
                K[f"unit_ratio_median_{m_}_{lab}"] = float(np.median(v_)) if len(v_) else None
                K[f"unit_ratio_range_{m_}_{lab}"] = (f"{v_.min():.3g}..{v_.max():.3g}" if len(v_) else None)
        w_ = u[u.model.isin(["znn_sbp", "hntznn"])].dropna(subset=["over_znn_Ex1", "over_znn_Ex1e3"])
        flips = ((w_.over_znn_Ex1 > 1) != (w_.over_znn_Ex1e3 > 1))
        K["unit_order_vs_znn_flips"] = (f"{int(flips.sum())} of {len(w_)} time-varying (model, problem, disc, "
                                        f"arm, tau) cells")
        K["unit_frac_div_Ex1e3_sbp_li"] = float(u[u.model.isin(["znn_sbp", "hntznn"])]["frac_div_Ex1e3"].mean())
    hv = tv[(tv.start == "near") & (tv.noise == "harmonic") & (tv.arm == "g10")]
    ct_l = hv[hv.model == "hntznn_lin"].set_index(["problem", "disc", "tau"])["ct"]
    zz = hv[hv.model == "znn"].set_index(["problem", "disc", "tau"])["meas"]
    q_ = (ct_l / zz.reindex(ct_l.index)).replace([np.inf, -np.inf], np.nan).dropna()
    K["hntznn_lin_ct_over_znn_harmonic_g10"] = float(np.median(q_)) if len(q_) else None
    return K


def interpretation(S, cells, main_prob):
    """Short interpretation; every number is computed from the cells (key_numbers)."""
    K = S["key_numbers"]
    f = _fmt_md
    k = S["kendall_ct_vs_measured"]
    dv = S["static_control_dt_validation"]
    taus = S["taus"]
    t0, t1 = (taus[-1], taus[0]) if taus else (None, None)
    txt = [
        f"(1) Linear-activation models (ZNN, IEZNN, DIEZNN): the discrete floor is the CT noise response plus "
        f"the noise-free discretization floor (median |log10 gap| {f(K['linear_models_abs_log10_gap_median'])}, "
        f"{f(K['linear_models_frac_within_x2'])} of the cells within x2); on the static control the floors "
        f"match the exact discrete error recursion (median log10 measured/DT = {f(dv['median_log10_meas_over_dt'])}). "
        f"For per-step Gaussian noise the exact discrete floor at fixed h is tau x F(h) (slope 1 exactly for ZNN "
        f"and IEZNN), and Euler amplifies the CT white-noise-equivalent RMS by sqrt(2/(2-h)) (ZNN-Gaussian check).",
        f"(2) Rejection claims and the finite horizon: IEZNN + constant noise, penalty "
        f"{f(K['penalty_ieznn_constant_h0.1'])} (h = 0.1) vs {f(K['penalty_znn_constant_h0.1'])} for linear ZNN "
        f"(the discrete integrator keeps its pole at 1, so the rejection is exact after sampling). DIEZNN + ramp "
        f"{f(K['penalty_dieznn_linear_h0.1'])} vs IEZNN {f(K['penalty_ieznn_linear_h0.1'])} and ZNN "
        f"{f(K['penalty_znn_linear_h0.1'])}; the DIEZNN residual is the CT response of its pole fixed at -1, "
        f"Q e^-t/(xi-1)^2 (measured/(CT + noise-free) = {f(K['ctgap_dieznn_linear_h0.1'])}); "
        f"a fraction {f(K['dieznn_linear_frac_transient_limited'])} of these cells is transient-limited, i.e. "
        f"the value is a T = 6 s transient, not a floor.",
        f"(3) Under the DTZNN convention (fixed h, gamma = h/tau) every noise floor shrinks with tau because "
        f"gamma grows: measured slopes ZNN-constant {f(K['slope_znn_constant_h0.1'])}, ZNN-Gaussian "
        f"{f(K['slope_znn_gaussian_h0.1'])}, IEZNN-ramp {f(K['slope_ieznn_linear_h0.1'])}, IEZNN-random walk "
        f"{f(K['slope_ieznn_random_walk_h0.1'])} (CT: 1, 1, 2, 1.5); at fixed gamma = 10 the Gaussian floor "
        f"still scales as tau^{f(K['slope_znn_gaussian_g10'])} (CT 0.5).",
        f"(4) Harmonic noise, internal model: with a LINEAR activation HNTZNN rejects the matched harmonic in CT, "
        f"but its forward-Euler internal model is detuned by O(omega^2 tau): at fixed gamma = 10 the discrete "
        f"floor is {f(K.get(f'hntznn_lin_static_dt_over_ct_g10_tau{t0:g}'))} x the CT value at tau = {t0:g} "
        f"and {f(K.get(f'hntznn_lin_static_dt_over_ct_g10_tau{t1:g}'))} x at tau = {t1:g} (static control, exact "
        f"DT recursion; measured/DT {f(K.get(f'hntznn_lin_static_meas_over_dt_g10_tau{t0:g}'))}), so the gain over "
        f"linear ZNN is only {f(K.get(f'hntznn_lin_over_znn_harmonic_g10_tau{t0:g}'))} (tau = {t0:g}) and "
        f"{f(K.get(f'hntznn_lin_over_znn_harmonic_g10_tau{t1:g}'))} (tau = {t1:g}) instead of "
        f"{f(K.get('hntznn_lin_ct_over_znn_harmonic_g10'))} in CT. A 10 % "
        f"frequency mismatch leaves {f(K.get(f'hntznn_lin_mis_over_znn_harmonic_g10_tau{t1:g}'))} x the ZNN floor. "
        f"Under fixed h the internal model learns at rate ~omega^2 lam/(2 gamma) -> 0 as tau -> 0: at "
        f"tau = {t1:g} HNTZNN-lin is {f(K.get(f'hntznn_lin_over_znn_harmonic_h0.1_tau{t1:g}'))} x ZNN "
        f"(horizon-limited). The Li-activated HNTZNN's harmonic penalty "
        f"{f(K['penalty_hntznn_harmonic_h0.1'])} only says that its noise response is below its own chattering "
        f"floor, which is unit-dependent (5).",
        f"(5) Sign-type activations depend on the unit of E. Noise-free floor with E expressed in units of 1e-3 "
        f"over the floor in the original unit (both in the original unit): ZNN "
        f"{f(K.get('unit_ratio_median_znn_tv'))} (control); on the static control ZNN-SBP "
        f"{f(K.get('unit_ratio_median_znn_sbp_static'))} and Li-HNTZNN {f(K.get('unit_ratio_median_hntznn_static'))} "
        f"(= 1/1e3: the chattering band lives in the unit of E); on the time-varying problems ZNN-SBP "
        f"{K.get('unit_ratio_range_znn_sbp_tv')} and Li-HNTZNN {K.get('unit_ratio_range_hntznn_tv')}; the order "
        f"relative to linear ZNN flips in {K.get('unit_order_vs_znn_flips')}. With the 0.1-radius start, "
        f"E x 1e3 even makes explicit SBP diverge (the |e|^2 branch). So ZNN-SBP's floor of "
        f"{f(K['sbp_none_floor_euler_h0.1_median'])} at h = 0.1 (chattering estimate sqrt(m)(h/4)^2 = "
        f"{f(K['sbp_chatter_prediction_h0.1_m4'])}-{f(K['sbp_chatter_prediction_h0.1_m16'])} in the unit of E) "
        f"and its weak tau-dependence (slope {f(K['slope_none_znn_sbp_h0.1'])}) hold in the benchmark's unit "
        f"only. SPTC-AN-FOZNN diverges in {f(K['sptc_frac_div_fixed_h'])} of the fixed-h cells (gamma = h/tau "
        f"multiplies its quadratic predefined-time term); at the published gamma = 2 its floor is "
        f"{f(K['sptc_floor_over_zeta_tau_euler_g2'])} x zeta*tau (sliding-mode chattering, zeta = 5 Delta with "
        f"the true noise bound), whereas CT theory predicts 0.",
        f"(6) Rankings (descriptive): Kendall tau_b between the CT-predicted and the measured model ranking has "
        f"median {f(k['median_tau_b'])} ({k['n_settings']} settings); adding the measured noise-free floor to "
        f"the CT prediction gives {f(k.get('median_tau_b_additive'))} (largely by construction: it tests "
        f"superposition). The disagreement comes from CT ties or near-ties (exact rejection; similar white-noise "
        f"floors), which discrete truncation, chattering and transient floors and the DT noise amplification "
        f"break, and from discrete divergence (worst rank), not from a wrong CT noise response of the linear "
        f"models: CT noise analysis alone cannot rank models whose noise is rejected. With 3-6 models per "
        f"setting these tau_b are descriptive only.",
    ]
    return "\n\n".join(txt)


if __name__ == "__main__":
    raise SystemExit(main())
