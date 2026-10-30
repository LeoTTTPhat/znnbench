#!/usr/bin/env python
"""E3(a): continuous-time convergence-time reference (custom memguard task).

What it computes
----------------
For a design-formula ZNN with exact time-derivative information, the error
obeys dE/dt = -gamma Phi(E) *component-wise* in continuous time, whatever the
problem (matrix inversion here): E(t) is the solution of m decoupled scalar
ODEs e_i' = -gamma phi(e_i).  For a time-invariant odd activation phi the
time to go from level a to level s < a is

    T(a -> s) = (1/gamma) * int_s^a du / phi(u)            (settling-time integral)

so e_i(t) is known exactly by inverting that integral.  This script

1. tabulates C(v) = int_{u_min}^{exp(v)} du/phi(u) on a fine log grid
   (Gauss-Legendre, 8 nodes per cell, 48 natural-log units, cell width ~0.004),
   inverts it with a monotone PCHIP interpolant, and finds
   t_ref(eps) = min{t : ||e(t)||_2 = eps} by Brent's method (||e(t)||_2 is
   monotone decreasing) inside the exact bracket
   [t_inf(eps), t_inf(eps / sqrt(m))], t_inf = time for max_i |e_i| to reach
   the level, computed independently by adaptive quadrature (scipy.quad,
   rel. tol 1e-11) -- the bracket check validates the table;
2. for the strictly-predefined-time entries uses closed forms
   (``strict_predefined`` activation and ``sptc_nt_znn`` before t_c:
   e(t) = e0 ((t_c - t)/t_c)^gamma, so t_ref = t_c (1 - (eps/||e0||)^(1/gamma)))
   or a 1-D ODE for ||E|| (``sptc_an_foznn``: E' is parallel to E, integrated
   in log coordinates with DOP853, rtol 1e-12);
3. cross-checks the *implementation* (problem + model + solve_J) by
   integrating the full ZNN state equation x' = model.rhs(x, t) with DOP853
   (rtol = atol = 1e-12) until ||E|| < eps_min, or until the first component
   |e_i| drops below 1e-9 (beyond that point non-Lipschitz activations make
   any explicit integrator stall: measured, see E3_NOTES.md), or a wall cap.
   It reports t_full(eps) when reached and the max relative deviation between
   ||E_full(t)|| and the semi-analytic ||e_ref(t)|| on all accepted steps;
4. records the convergence-time bounds the activations / models expose:
   ``upper_bound_time(e0, gamma)`` (bound stated by the source),
   ``uniform_bound(gamma)`` (e0-independent; DERIVED for sign_bi_power/wsbp),
   ``tight_uniform_bound(gamma)`` (DERIVED Beta-integral sup), and the exact
   settling time to E = 0 (``settling_time``; t_c for the strict models).

Initial errors: the B initial states of ``simulate.initial_state(problem,
model, seed, B)`` (identical to the runner's draw, so the discrete sweeps can
be compared trajectory by trajectory: scale ``nat``), and rescaled copies
E0 = s * E0_nat / max|E0_nat| for s in the e0-scale grid, realised exactly
through x0 = x*(0) + J^{-1} E0 (E is affine in x for matrix inversion).

Usage (memguard-compatible: one JSON line per task with id/cmd/done/cwd/env;
the done parquet is written atomically)::

    python experiments/tasks/E3_ct_reference.py build --scale mini --out results/mini/E3/ct
    python tools/memguard.py pool --tasks results/mini/E3/ct/tasks.jsonl --workers 3 --max-gb 40
    python experiments/tasks/E3_ct_reference.py run --task results/mini/E3/ct/chunks/<id>.json
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import uuid  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

FORMAT = "znnbench-E3-ct/1"
EPS = (1e-2, 1e-3, 1e-6)
COMPONENT_STOP = 1e-9          # full-state integration stops when min_i |e_i| < this
PROBLEM = {"key": "matrix_inversion", "kwargs": {"n": 4, "kappa": 10.0}}

# ---------------------------------------------------------------------------
# entries (what is audited)
# ---------------------------------------------------------------------------
# kind "act": design-formula ZNN ``znn`` with this activation (time-invariant
# unless key == strict_predefined); kind "model": registry model.
_BASE = [
    ("linear", "act", "linear", {}),
    ("sign", "act", "sign", {}),
    ("sign_power", "act", "sign_power", {}),
    ("saturated_sign_power", "act", "saturated_sign_power", {}),
    ("sfaf", "act", "sfaf", {}),
    ("sign_bi_power", "act", "sign_bi_power", {}),
    ("wsbp", "act", "wsbp", {}),
    ("fixed_time", "act", "fixed_time", {}),
    ("vaf", "act", "vaf", {}),
    ("pvaf1", "act", "pvaf1", {}),
    ("pvaf2", "act", "pvaf2", {}),
    ("predefined_exp", "act", "predefined_exp", {}),
    ("strict_predefined", "act", "strict_predefined", {"t_c": 1.0}),
    ("sptc_nt_znn", "model", "sptc_nt_znn", {"t_c": 1.0}),
    ("sptc_an_foznn", "model", "sptc_an_foznn", {"t_c": 1.0}),
]
# extra parameterisations in the full sweep (exponent dependence; pure bi-power
# fields for which the DERIVED tight bound is exact)
_FULL_EXTRA = [
    ("sign_power(r=0.25)", "act", "sign_power", {"r": 0.25}),
    ("sign_power(r=0.75)", "act", "sign_power", {"r": 0.75}),
    ("sign_bi_power(r=0.25)", "act", "sign_bi_power", {"r": 0.25}),
    ("sign_bi_power(r=0.75)", "act", "sign_bi_power", {"r": 0.75}),
    ("fixed_time(k3=0)", "act", "fixed_time", {"k3": 0.0}),
    ("vaf(a3=a4=0)", "act", "vaf", {"a3": 0.0, "a4": 0.0}),
    ("predefined_exp(k2=k3=0)", "act", "predefined_exp", {"k2": 0.0, "k3": 0.0}),
]
TV_KEYS = ("strict_predefined", "sptc_nt_znn", "sptc_an_foznn")

SCALES = {
    "mini": {"seeds": [0, 1, 2], "B": 4, "e0_scales": ["nat", 1e-2, 1e2],
             "gammas_ti": [1.0], "gammas_tv": [1.0, 2.0, 10.0],
             "full_state": {"scales": ["nat"], "x0_index": [0], "max_wall": 8.0}},
    "full": {"seeds": list(range(10)), "B": 4, "e0_scales": ["nat", 1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2],
             "gammas_ti": [1.0, 10.0], "gammas_tv": [1.0, 2.0, 10.0],
             "full_state": {"scales": ["nat", 1.0], "x0_index": [0, 1, 2, 3], "max_wall": 60.0}},
}


def entries(scale: str):
    from znnbench import activations as acts
    from znnbench import available
    have_act, have_mod = set(acts.available()), set(available()["models"])
    ents = _BASE + (_FULL_EXTRA if scale == "full" else [])
    out = []
    for label, kind, key, kw in ents:
        if (kind == "act" and key in have_act and "znn" in have_mod) or (kind == "model" and key in have_mod):
            out.append({"label": label, "kind": kind, "key": key, "kwargs": dict(kw)})
    return out


def is_tv(entry) -> bool:
    return entry["key"] in TV_KEYS


# ---------------------------------------------------------------------------
# semi-analytic reference for time-invariant activations
# ---------------------------------------------------------------------------
class ScalarFlow:
    """Exact flow of e' = -phi(e) (gamma = 1) for an odd, increasing phi.

    C(v) = int_{u_min}^{e^v} du / phi(u) on v in [ln u_min, ln u_max];
    time from level a to level s (s < a) = C(ln a) - C(ln s)."""

    def __init__(self, act, u_min=1e-16, u_max=1e4, cells=12000, nodes=8):
        from scipy.interpolate import PchipInterpolator
        self.act = act
        self.v0, self.v1 = math.log(u_min), math.log(u_max)
        edges = np.linspace(self.v0, self.v1, cells + 1)
        xg, wg = np.polynomial.legendre.leggauss(nodes)
        a, b = edges[:-1], edges[1:]
        mid, half = 0.5 * (a + b), 0.5 * (b - a)
        v = mid[:, None] + half[:, None] * xg[None, :]
        u = np.exp(v)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            phi = np.asarray(act(u.ravel()), dtype=float).reshape(u.shape)
            g = np.where(phi > 0, u / phi, 0.0)
        cell = (g * wg[None, :]).sum(axis=1) * half
        C = np.concatenate([[0.0], np.cumsum(cell)])
        keep = np.concatenate([[True], np.diff(C) > 0])       # strictly increasing for the inverse
        self.v, self.C = edges[keep], C[keep]
        self.Cmax = self.C[-1]
        self._C_of_v = PchipInterpolator(self.v, self.C, extrapolate=False)
        self._v_of_C = PchipInterpolator(self.C, self.v, extrapolate=False)
        # time to reach u_min from u_min itself is 0; below u_min treat e = 0
        # (u_min = 1e-16 << eps / sqrt(m) for every eps used here)

    def C_of(self, a):
        a = np.asarray(a, dtype=float)
        v = np.log(np.clip(a, math.exp(self.v0), math.exp(self.v1)))
        return self._C_of_v(v)

    def levels(self, a, t):
        """|e_i(t)| for initial levels a (array) at time t (gamma = 1)."""
        target = self.C_of(a) - t
        out = np.zeros_like(target)
        ok = target > 0
        if ok.any():
            out[ok] = np.exp(self._v_of_C(target[ok]))
        return out

    def t_inf_quad(self, a, s):
        """Exact (quadrature) time from level a down to level s (gamma = 1)."""
        from scipy.integrate import quad
        if s >= a:
            return 0.0
        act = self.act

        def integrand(w):          # u = a * exp(-w), w in [0, ln(a/s)]
            u = a * math.exp(-w)
            with np.errstate(over="ignore"):
                phi = float(act(np.array([u]))[0])
            return u / phi if phi > 0 else 0.0

        val, _ = quad(integrand, 0.0, math.log(a / s), limit=500, epsabs=0.0, epsrel=1e-11)
        return val


def t_ref_ti(flow: ScalarFlow, e0, eps, gamma):
    """||e(t)||_2 = eps crossing time for component-wise e' = -gamma phi(e)."""
    from scipy.optimize import brentq
    a = np.abs(np.asarray(e0, dtype=float))
    m = a.size
    nrm0 = float(np.linalg.norm(a))
    if nrm0 <= eps:
        return 0.0, 0.0, 0.0
    amax = float(a.max())
    lo = flow.t_inf_quad(amax, eps) if amax > eps else 0.0
    hi = flow.t_inf_quad(amax, eps / math.sqrt(m))
    f = lambda t: float(np.linalg.norm(flow.levels(a, t))) - eps  # noqa: E731
    lo_b, hi_b = lo * (1 - 1e-7), hi * (1 + 1e-7) + 1e-300
    flo, fhi = f(lo_b), f(hi_b)
    if not (flo >= 0 >= fhi):        # table / bracket disagreement -> widen (recorded via bracket_ok)
        lo_b, hi_b = 0.0, 2.0 * hi + 1.0
    t = brentq(f, lo_b, hi_b, xtol=1e-14, rtol=1e-13, maxiter=200)
    return t / gamma, lo / gamma, hi / gamma


def ref_norm_trajectory_ti(flow, e0, gamma, t):
    a = np.abs(np.asarray(e0, dtype=float))
    return np.array([np.linalg.norm(flow.levels(a, gamma * tk)) for tk in np.atleast_1d(t)])


# ---------------------------------------------------------------------------
# strictly predefined-time entries
# ---------------------------------------------------------------------------
def t_ref_strict(nrm0, eps, gamma, t_c):
    """e(t) = e0 ((t_c - t)/t_c)^gamma  (strict_predefined act., sptc_nt_znn pre-t_c)."""
    if nrm0 <= eps:
        return 0.0
    return t_c * (1.0 - (eps / nrm0) ** (1.0 / gamma))


def strict_norm_trajectory(nrm0, gamma, t_c, t):
    t = np.atleast_1d(np.asarray(t, dtype=float))
    return nrm0 * np.clip((t_c - t) / t_c, 0.0, None) ** gamma


def t_ref_sptc_an(nrm0, eps_list, gamma, t_c=1.0, alpha=0.5, t_floor=1e-3, dense=False):
    """SPTC-AN-FOZNN noise-free norm dynamics before t_c (Yang et al. 2025, Eq. 15):
    r' = -(max(t,t_floor)/t_c)^(alpha-1) [r/(t_c-t) + gamma r^2/(t_c-t)^2],
    integrated for u = ln r with DOP853 (rtol 1e-12) and events u = ln eps."""
    from scipy.integrate import solve_ivp

    def rhs(t, u):
        s = (max(t, t_floor) / t_c) ** (alpha - 1.0)
        d = t_c - t
        return [-s * (1.0 / d + gamma * math.exp(u[0]) / d ** 2)]

    out = {}
    evs = []
    for e in eps_list:
        ev = (lambda t, u, e=e: u[0] - math.log(e))
        ev.direction = -1
        ev.terminal = e == min(eps_list)      # nothing below the smallest eps is needed
        evs.append(ev)
    t_end = t_c * (1.0 - 1e-13)
    pieces = [(0.0, min(t_floor, t_end)), (min(t_floor, t_end), t_end)]
    u0 = [math.log(nrm0)]
    sols = []
    for (a, b) in pieces:
        if b <= a:
            continue
        sol = solve_ivp(rhs, (a, b), u0, method="DOP853", rtol=1e-12, atol=1e-12, events=evs,
                        dense_output=dense)
        sols.append(sol)
        for e, te in zip(eps_list, sol.t_events):
            if len(te) and e not in out:
                out[e] = float(te[0])
        u0 = [sol.y[0, -1]]
        if sol.status == 1:                   # terminal event reached
            break
    for e in eps_list:
        out.setdefault(e, t_c if nrm0 > e else 0.0)   # crosses at t_c at the latest
    if dense:
        return out, sols
    return out


# ---------------------------------------------------------------------------
# problem / model helpers
# ---------------------------------------------------------------------------
def make_components(entry, seed):
    from znnbench import make_model, make_problem
    P = make_problem(PROBLEM["key"], seed=seed, **PROBLEM["kwargs"])
    if entry["kind"] == "act":
        M = make_model("znn", activation=entry["key"], **entry["kwargs"])
    else:
        M = make_model(entry["key"], **entry["kwargs"])
    return P, M


def activation_of(entry):
    from znnbench import activations as acts
    if entry["kind"] == "act":
        return acts.get(entry["key"], **entry["kwargs"])
    return None


def initial_states(P, M, seed, B, scale):
    """x0 (B, n_x) and E0 (B, m): natural draw or rescaled (max|E0_i| = scale)."""
    from znnbench.simulate import initial_state
    x0 = initial_state(P, M, seed, B)
    d0 = P.data(0.0)
    E0 = P.E(x0, d0)
    if scale == "nat":
        return x0, E0
    u = E0 / np.abs(E0).max(axis=1, keepdims=True)
    E0s = float(scale) * u
    xs = np.broadcast_to(P.x_star(0.0), x0.shape).copy()
    x0s = xs + P.solve_J(xs, d0, E0s)
    return x0s, P.E(x0s, d0)


def bounds_of(entry, act, e0, gamma):
    nan = float("nan")

    def fin(v):
        return float(v) if v is not None and np.isfinite(v) else nan

    if act is not None and not act.time_varying:
        src = fin(act.upper_bound_time(e0, gamma))
        uni = fin(act.uniform_bound(gamma))
        tight = fin(act.tight_uniform_bound(gamma)) if hasattr(act, "tight_uniform_bound") else nan
        settle = fin(act.settling_time(e0, gamma))
        claimed, derived = act.claimed_convergence, act.convergence
    else:
        t_c = float(entry["kwargs"].get("t_c", 1.0))
        src = uni = settle = t_c
        tight = nan
        claimed = derived = "strictly-predefined"
    return {"bound_source": src, "bound_uniform": uni, "bound_tight": tight, "t_settle": settle,
            "claimed_class": claimed, "derived_class": derived}


class _WallCap(Exception):
    pass


def full_state(P, M, x0, gamma, eps_list, max_wall, ref_norm, t_max):
    """DOP853 on x' = model.rhs; returns dict (t_full per eps, deviation, status)."""
    from scipy.integrate import solve_ivp
    from znnbench.core import Context
    t0 = time.perf_counter()
    nfev = [0]
    gam = np.full((1, 1), float(gamma))
    z0 = M.z0(P, 1)

    def rhs(t, y):
        nfev[0] += 1
        if time.perf_counter() - t0 > max_wall:
            raise _WallCap()
        x = y[None, :]
        d = P.data(t)
        ctx = Context(problem=P, t=t, tau=1e-6, k=-1, x=x, z=z0, d=d, E=P.E(x, d),
                      dEdt=P.dEdt(x, t), gamma=gam, implicit=False, extra={"info": "analytic"})
        return M.rhs(ctx)[0][0]

    evs = []
    for e in eps_list:
        ev = (lambda t, y, e=e: float(np.linalg.norm(P.E(y[None, :], P.data(t)))) - e)
        ev.direction = -1
        ev.terminal = e == min(eps_list)
        evs.append(ev)
    comp = (lambda t, y: float(np.abs(P.E(y[None, :], P.data(t))).min()) - COMPONENT_STOP)
    comp.direction = -1
    comp.terminal = True
    evs.append(comp)
    res = {f"t_full_{e:g}": float("nan") for e in eps_list}
    try:
        sol = solve_ivp(rhs, (0.0, t_max), x0, method="DOP853", rtol=1e-12, atol=1e-12, events=evs)
    except _WallCap:
        res.update({"full_status": "wall_cap", "full_nfev": nfev[0], "full_wall": time.perf_counter() - t0,
                    "full_t_stop": float("nan"), "full_maxrel_dev": float("nan")})
        return res
    for e, te in zip(eps_list, sol.t_events[:len(eps_list)]):
        if len(te):
            res[f"t_full_{e:g}"] = float(te[0])
    stop = "eps" if len(sol.t_events[len(eps_list) - 1]) else (
        "component" if len(sol.t_events[-1]) else ("t_max" if sol.status == 0 else f"fail:{sol.message}"))
    nrm = np.array([np.linalg.norm(P.E(sol.y[:, j][None, :], P.data(tj))) for j, tj in enumerate(sol.t)])
    ref = ref_norm(sol.t)
    ok = (ref > 1e-12) & np.isfinite(ref)
    dev = float(np.max(np.abs(nrm[ok] - ref[ok]) / ref[ok])) if ok.any() else float("nan")
    res.update({"full_status": stop, "full_nfev": nfev[0], "full_wall": time.perf_counter() - t0,
                "full_t_stop": float(sol.t[-1]), "full_maxrel_dev": dev})
    return res


# ---------------------------------------------------------------------------
# one task = one entry
# ---------------------------------------------------------------------------
def run_entry(task: dict):
    entry, sc = task["entry"], SCALES[task["scale"]]
    act = activation_of(entry)
    tv = is_tv(entry)
    gammas = sc["gammas_tv"] if tv else sc["gammas_ti"]
    fs = sc["full_state"]
    flow = ScalarFlow(act) if (act is not None and not act.time_varying) else None
    rows = []
    t_task = time.perf_counter()
    for seed in sc["seeds"]:
        P, M = make_components(entry, seed)
        for scale in sc["e0_scales"]:
            x0s, E0s = initial_states(P, M, seed, sc["B"], scale)
            for b in range(sc["B"]):
                e0 = E0s[b]
                nrm0 = float(np.linalg.norm(e0))
                for gamma in gammas:
                    tw = time.perf_counter()
                    row = {"entry": entry["label"], "kind": entry["kind"], "key": entry["key"],
                           "kwargs": json.dumps(entry["kwargs"], sort_keys=True), "tv": tv,
                           "problem": PROBLEM["key"], "seed": seed, "x0_index": b,
                           "e0_scale": str(scale), "e0_max": float(np.abs(e0).max()), "e0_norm": nrm0,
                           "m": int(e0.size), "gamma": float(gamma)}
                    row.update(bounds_of(entry, act, e0, gamma))
                    t_c = float(entry["kwargs"].get("t_c", 1.0))
                    if flow is not None:
                        for e in EPS:
                            t, lo, hi = t_ref_ti(flow, e0, e, gamma)
                            row[f"t_ref_{e:g}"] = t
                            row[f"t_inf_{e:g}"] = lo
                            row[f"bracket_ok_{e:g}"] = bool(lo * (1 - 1e-6) <= t <= hi * (1 + 1e-6))
                        ref_norm = (lambda tt, e0=e0, g=gamma: ref_norm_trajectory_ti(flow, e0, g, tt))
                        t_max = 1.5 * row["t_ref_1e-06"] + 1.0
                    elif entry["key"] == "sptc_an_foznn":
                        kw = entry["kwargs"]
                        tr = t_ref_sptc_an(nrm0, EPS, gamma, t_c=t_c, alpha=float(kw.get("alpha", 0.5)),
                                           t_floor=float(kw.get("t_floor", 1e-3)))
                        for e in EPS:
                            row[f"t_ref_{e:g}"] = tr[e]
                            row[f"t_inf_{e:g}"] = float("nan")
                            row[f"bracket_ok_{e:g}"] = True
                        ref_norm = (lambda tt, n0=nrm0, g=gamma: _sptc_an_norm(n0, g, t_c, kw, tt))
                        t_max = t_c * (1 - 1e-12)
                    else:
                        for e in EPS:
                            row[f"t_ref_{e:g}"] = t_ref_strict(nrm0, e, gamma, t_c)
                            row[f"t_inf_{e:g}"] = float("nan")
                            row[f"bracket_ok_{e:g}"] = True
                        ref_norm = (lambda tt, n0=nrm0, g=gamma: strict_norm_trajectory(n0, g, t_c, tt))
                        t_max = t_c * (1 - 1e-12)
                    row["ref_wall"] = time.perf_counter() - tw
                    if scale in fs["scales"] and b in fs["x0_index"]:
                        row.update(full_state(P, M, x0s[b], gamma, EPS, fs["max_wall"], ref_norm, t_max))
                    else:
                        row.update({f"t_full_{e:g}": float("nan") for e in EPS})
                        row.update({"full_status": "not_run", "full_nfev": 0, "full_wall": float("nan"),
                                    "full_t_stop": float("nan"), "full_maxrel_dev": float("nan")})
                    rows.append(row)
    for r in rows:
        r["task_wall"] = time.perf_counter() - t_task
    return rows


def _sptc_an_norm(nrm0, gamma, t_c, kw, t):
    t = np.atleast_1d(np.asarray(t, dtype=float))
    _, sols = t_ref_sptc_an(nrm0, EPS, gamma, t_c=t_c, alpha=float(kw.get("alpha", 0.5)),
                            t_floor=float(kw.get("t_floor", 1e-3)), dense=True)
    out = np.full(t.shape, np.nan)
    for sol in sols:
        a, b = sol.t[0], sol.t[-1]
        msk = (t >= a) & (t <= b)
        if msk.any():
            out[msk] = np.exp(sol.sol(t[msk])[0])
    return out


# ---------------------------------------------------------------------------
# build / run (memguard)
# ---------------------------------------------------------------------------
def _atomic_write_bytes(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:6]}")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def build(scale: str, out_dir, python: str | None = None) -> Path:
    from znnbench.runner import DEFAULT_PYTHON, THREAD_ENV, ZNN_ROOT
    out = Path(out_dir).resolve()
    python = python or DEFAULT_PYTHON
    lines = []
    for i, ent in enumerate(entries(scale)):
        payload = {"format": FORMAT, "scale": scale, "entry": ent, "problem": PROBLEM, "eps": list(EPS),
                   "params": SCALES[scale]}
        h = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:10]
        tid = f"E3_ct-{i:03d}-{h}"
        chunk = out / "chunks" / f"{tid}.json"
        result = out / "results" / f"{tid}.parquet"
        _atomic_write_bytes(chunk, json.dumps({**payload, "task_id": tid, "result": str(result)},
                                              indent=1).encode())
        lines.append(json.dumps({"id": tid, "cmd": [python, "experiments/tasks/E3_ct_reference.py", "run",
                                                    "--task", str(chunk)],
                                 "done": str(result), "cwd": str(ZNN_ROOT), "env": dict(THREAD_ENV)}))
    (out / "results").mkdir(parents=True, exist_ok=True)
    tf = out / "tasks.jsonl"
    _atomic_write_bytes(tf, ("\n".join(lines) + "\n").encode())
    return tf


def run(task_file, force=False):
    import pandas as pd
    task = json.loads(Path(task_file).read_text())
    if task.get("format") != FORMAT:
        raise ValueError(f"{task_file}: not a {FORMAT} task")
    result = Path(task["result"])
    if result.exists() and not force:
        print(f"[E3_ct] {task['task_id']}: already done", flush=True)
        return result
    t0 = time.perf_counter()
    with np.errstate(all="ignore"):
        rows = run_entry(task)
    df = pd.DataFrame(rows)
    df["task_id"] = task["task_id"]
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        df["peak_rss_mb"] = r / 2 ** 20 if sys.platform == "darwin" else r / 2 ** 10
    except Exception:  # noqa: BLE001
        df["peak_rss_mb"] = float("nan")
    result.parent.mkdir(parents=True, exist_ok=True)
    tmp = result.with_name(f".{result.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:6]}")
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, result)
    finally:
        if tmp.exists():
            tmp.unlink()
    print(f"[E3_ct] {task['task_id']} {task['entry']['label']}: {len(df)} rows in "
          f"{time.perf_counter() - t0:.1f}s -> {result}", flush=True)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--scale", choices=["mini", "full"], default="mini")
    b.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--task", required=True)
    r.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd == "build":
        tf = build(args.scale, args.out)
        print(f"E3_ct: {sum(1 for _ in open(tf))} tasks -> {tf}")
        return 0
    out = run(args.task, force=args.force)
    return 0 if Path(out).exists() else 1


if __name__ == "__main__":
    sys.exit(main())
