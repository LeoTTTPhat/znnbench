#!/usr/bin/env python
"""E4 calibration tasks (P1 / P2 / P3 gains) and sweep builder.

One calibration task = one (problem, model) pair of the E4 panel
(experiments/configs/E4.py).  It works on the held-out calibration seeds
(``calib_seeds`` = 100, 101, 102: three problem instances x B initial states,
i.e. as many trajectories as one evaluation cell; one seed with B = 4 gave
held-out P3 effort ratios of 0.53-0.79 for the power-type activations) and
writes one JSON file (the memguard ``done`` file, written atomically) with

* P1: e_ref = median_b ||E(x0_b, 0)||_2 / sqrt(m) (RMS entry of the initial
  error) and the gamma that makes the model's initial effective gain
  kappa(gamma) = -<r, E>/<E, E> (r = desired dE/dt of the model at t = 0+,
  z = z0, E with all entries equal) equal to gamma* -- evaluated at
  e -> 0 (= gamma Phi'(0) g(0)) when that limit is finite, else at e_ref
  (secant gain gamma Phi(e_ref)/e_ref g(0)); plus ``gamma_secant``, the gamma
  that matches the secant gain at e_ref for every activation (config P1sec);
* P2: h_max, the largest h = gamma*tau (explicit Euler, tau_dt) for which every
  calibration trajectory is stable (no divergence, steady residual
  <= rho ||E(x0, 0)||, no growth over the last T/8, no transient above
  10 ||E(x0, 0)||), found by a descending log
  grid + multi-section refinement; h_eval = 0.8 h_max;
* P3: budgets U(gamma_ref) = median peak ||x_dot|| of linear ZNN at gamma_ref
  in {gamma* = 10 (main, U*), 5, 30} (continuous time, RK4 tau_ct) and, for
  each budget, the LARGEST gamma whose median peak stays within it (the peak
  is not monotone in gamma; ascending log grid + refinement + log-log
  interpolation inside the bracket, verified by a final run).

Usage (from znn-bench/):
  python experiments/tasks/E4_calibrate.py build  --scale mini       # calib tasks.jsonl
  python experiments/tasks/E4_calibrate.py run    --scale mini --problem P --model M --out F.json
  python experiments/tasks/E4_calibrate.py sweep  --scale mini       # runner configs -> all_tasks.jsonl
  python experiments/tasks/E4_calibrate.py status --scale mini
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import resource  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import uuid  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from znnbench.core import Context, Problem  # noqa: E402
from znnbench.registry import make_discretization, make_model, make_problem  # noqa: E402
from znnbench.runner import DEFAULT_PYTHON, THREAD_ENV, PeakXdotProbe  # noqa: E402
from znnbench.simulate import initial_state, simulate  # noqa: E402


def load_cfg():
    p = ROOT / "experiments" / "configs" / "E4.py"
    spec = importlib.util.spec_from_file_location("_e4_cfg", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = load_cfg()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _atomic_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:6]}")
    try:
        tmp.write_text(json.dumps(obj, indent=1, allow_nan=True, default=float))
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _subst_seed(kw, seed):
    return {k: (int(seed) if v == "$seed" else v) for k, v in kw.items()}


def problem_for(prob_entry, seed, unit=1.0):
    """The problem instance of a seed; unit != 1 expresses its error as
    E_u = unit * E (R_common.ScaledError, the wrapper of the sweep's unit arms;
    its residual() stays in the original unit)."""
    p = make_problem(prob_entry["key"], **_subst_seed(prob_entry["kwargs"], seed))
    if not math.isclose(float(unit), 1.0):
        from experiments.tasks.R_common import ScaledError
        p = ScaledError(p, scale=float(unit))
    return p


def model_for(entry):
    return make_model(entry["key"], **entry["kwargs"])


def _rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2 ** 20 if sys.platform == "darwin" else r / 2 ** 10


def batch_sim(problem, model, disc_key, tau, T, gammas, x0, seed, probe=False, rec=1):
    """One simulate call with G gains x B initial states (trajectory g*B + b
    starts from x0[b] with gain gammas[g]; per-trajectory noise ids b)."""
    gammas = np.asarray(gammas, dtype=float)
    G, B = len(gammas), x0.shape[0]
    mdl = PeakXdotProbe(model) if probe else model
    with np.errstate(all="ignore"):
        sim = simulate(problem, mdl, make_discretization(disc_key), tau, T=T, B=B * G,
                       gamma=np.repeat(gammas, B), x0=np.tile(x0, (G, 1)), seed=seed,
                       traj_ids=np.tile(np.arange(B), G), record_every=rec,
                       record_solution_error=False, online_stats=False)
    peak = np.asarray(mdl._peak, dtype=float).reshape(G, B) if probe else None
    return sim, peak


# ---------------------------------------------------------------------------
# P1: effective initial gain of a model (generic, through its own rhs)
# ---------------------------------------------------------------------------
class _IdentityProblem(Problem):
    """E = x, J = I, dE/dt = 0: model.rhs then returns the desired dE/dt."""

    name = "identity_probe"

    def __init__(self, m):
        self.n_x = self.m = int(m)

    def data(self, t):
        return {"t": np.float64(t)}

    def E(self, x, d):
        return x

    def dEdt(self, x, t):
        return np.zeros_like(x)

    def solve_J(self, x, d, r):
        return r

    def JT(self, x, d, v):
        return v

    def x0(self, rng, B):
        return np.zeros((B, self.m))


def eff_gain(model, m, gamma, e, t=0.0, tau=1e-4):
    """kappa = -<r, E>/<E, E> at time t for E with all m entries equal to e."""
    pr = _IdentityProblem(m)
    x = np.full((1, m), float(e))
    z = model.z0(pr, 1)
    ctx = Context(problem=pr, t=t, tau=tau, k=0, x=x, z=z, d=pr.data(t), E=x.copy(),
                  dEdt=np.zeros_like(x), gamma=np.array([[float(gamma)]]), implicit=False,
                  extra={"info": "analytic"})
    with np.errstate(all="ignore"):
        r, _ = model.rhs(ctx)
    return float(-(r * x).sum() / (x * x).sum())


def _probe_rz(model, m, gamma, e, zval, t=0.0, tau=1e-4):
    """(mean desired dE/dt, mean z_dot) on the identity probe with all entries
    of E equal to e and all auxiliary states equal to zval."""
    pr = _IdentityProblem(m)
    x = np.full((1, m), float(e))
    z = np.full_like(np.asarray(model.z0(pr, 1), dtype=float), float(zval))
    ctx = Context(problem=pr, t=t, tau=tau, k=0, x=x, z=z, d=pr.data(t), E=x.copy(),
                  dEdt=np.zeros_like(x), gamma=np.array([[float(gamma)]]), implicit=False,
                  extra={"info": "analytic"})
    with np.errstate(all="ignore"):
        r, zd = model.rhs(ctx)
    return float(np.mean(r)), float(np.mean(zd))


def pole_rate(model, m, gamma, e):
    """Slowest decay rate of the model's linearised error dynamics at t = 0+.

    Models without an auxiliary state: kappa (= eff_gain).  Models with one
    auxiliary state per error entry (IEZNN: z = int E): the eigenvalues of the
    2 x 2 Jacobian of (e, z) -> (de/dt, dz/dt) at (e, 0) (central differences;
    exact for IEZNN, whose double pole is gamma/2 with lambda = gamma^2/4);
    rate = min -Re(eigenvalue) (nan if unstable).  Other auxiliary-state
    layouts are not linearised (nan)."""
    pr = _IdentityProblem(m)
    nz = int(np.size(model.z0(pr, 1)))
    if nz == 0:
        return eff_gain(model, m, gamma, e)
    if nz != m:
        return float("nan")
    de = 1e-3 * abs(e) if e != 0 else 1e-6
    f = lambda ev, zv: np.array(_probe_rz(model, m, gamma, ev, zv))  # noqa: E731
    jac = np.column_stack([(f(e + de, 0.0) - f(e - de, 0.0)) / (2 * de),
                           (f(e, de) - f(e, -de)) / (2 * de)])
    ev = np.linalg.eigvals(jac)
    rate = float(np.min(-ev.real))
    return rate if rate > 0 else float("nan")


def solve_increasing(fun, target, lo, hi, rtol=1e-12):
    """gamma in [lo, hi] with fun(gamma) = target for increasing fun (log bisection)."""
    flo, fhi = fun(lo), fun(hi)
    if not flo < target:
        return lo, "floor"
    if not fhi > target:
        return hi, "cap"
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if fun(mid) < target:
            lo = mid
        else:
            hi = mid
        if hi / lo < 1 + rtol:
            break
    return math.sqrt(lo * hi), "ok"


def calibrate_p1(model, m, e_ref):
    small = C.E_REF_SMALL * e_ref
    g_a, g_b = eff_gain(model, m, 1.0, small), eff_gain(model, m, 1.0, small * 1e-2)
    lipschitz = bool(np.isfinite(g_a) and np.isfinite(g_b) and g_b > 0 and abs(g_a / g_b - 1) < 1e-3)
    e_eval = small if lipschitz else e_ref
    kfun = lambda g: eff_gain(model, m, g, e_eval)  # noqa: E731
    out = {"lipschitz": lipschitz, "kappa_mode": "slope0" if lipschitz else "secant",
           "e_eval": e_eval, "kappa_unit": kfun(1.0), "gamma": {}, "flag": {}, "kappa_at_gamma": {},
           "kappa_affine_offset": kfun(C.GAIN_RANGE[0])}
    for gs in (C.GAMMA_STAR, *C.GAMMA_STAR_SENS):
        g, flag = solve_increasing(kfun, gs, *C.GAIN_RANGE)
        out["gamma"][f"{gs:g}"] = g
        out["flag"][f"{gs:g}"] = flag
        out["kappa_at_gamma"][f"{gs:g}"] = kfun(g)
    # sensitivity P1sec: secant gain at e_ref for every activation
    sfun = lambda g: eff_gain(model, m, g, e_ref)  # noqa: E731
    g, flag = solve_increasing(sfun, C.GAMMA_STAR, *C.GAIN_RANGE)
    out["gamma_secant"] = {f"{C.GAMMA_STAR:g}": g}
    out["flag_secant"] = {f"{C.GAMMA_STAR:g}": flag}
    out["kappa_secant_unit"] = sfun(1.0)
    # sensitivity P1pole: slowest pole of the linearised (E, z) dynamics = gamma*
    # (equals P1 for models without an auxiliary state; IEZNN: gamma = 2 gamma*)
    pfun = lambda g: pole_rate(model, m, g, e_eval)  # noqa: E731
    g, flag = solve_increasing(pfun, C.GAMMA_STAR, *C.GAIN_RANGE)
    out["gamma_pole"] = {f"{C.GAMMA_STAR:g}": g}
    out["flag_pole"] = {f"{C.GAMMA_STAR:g}": flag}
    out["pole_rate_unit"] = pfun(1.0)
    return out


# ---------------------------------------------------------------------------
# P2: maximal stable h (explicit Euler)
# ---------------------------------------------------------------------------
def stable_mask(res, t, e0, T):
    """Per-trajectory stability: finite, steady residual (last 25 %) <=
    rho * ||E(x0, 0)||, max over the last T/8 <= growth * max over the
    previous T/8 (or below 1e-10), and max over the whole run <=
    STAB_TRANSIENT * ||E(x0, 0)|| (no transient blow-up)."""
    w = t >= t[-1] - 0.25 * (t[-1] - t[0])
    mss = np.max(res[w], axis=0)
    a = t >= t[-1] - T / 8
    b = (t >= t[-1] - T / 4) & ~a
    ma, mb = np.max(res[a], axis=0), np.max(res[b], axis=0)
    ok = np.isfinite(mss) & (mss <= C.STAB_RHO * e0)
    ok &= (ma <= C.STAB_GROWTH * mb) | (ma <= 1e-10)
    ok &= np.max(res, axis=0) <= C.STAB_TRANSIENT * e0            # no transient blow-up
    return ok, mss


def calibrate_p2(cal, model, S):
    """cal: list of (problem, x0, e0, seed) of the calibration seeds; h is
    stable iff every trajectory of every calibration seed is stable."""
    tau, T = S["tau_dt"], S["T"]
    log = []

    def evaluate(hs):
        hs = np.asarray(hs, dtype=float)
        oks, msss = [], []
        for problem, x0, e0, seed in cal:
            sim, _ = batch_sim(problem, model, "euler", tau, T, hs / tau, x0, seed, rec=S["rec_dt"])
            ok, mss = stable_mask(sim["residual"], sim["t"], np.tile(e0, len(hs)), T)
            oks.append(ok.reshape(len(hs), -1))
            msss.append(mss.reshape(len(hs), -1))
        ok, mss = np.hstack(oks), np.hstack(msss)
        st = ok.all(axis=1)
        for h, s, mm, o in zip(hs, st, mss, ok):
            log.append([float(h), bool(s), float(np.median(mm)), int((~np.isfinite(mm)).sum()), int(o.sum())])
        return st

    grid = np.geomspace(C.H_RANGE[1], C.H_RANGE[0], S["p2_grid"])          # descending
    st = evaluate(grid)
    idx = np.flatnonzero(st)
    if len(idx) == 0:
        return {"h_max": float("nan"), "h_eval": None, "gamma_eval": None, "status": "never_stable",
                "n_admissible_blocks": 0, "evaluations": log}
    # The admissible set is an interval [h_lo, h_max]: below h_lo the residual does not
    # contract 10x within T, above h_max the step is unstable / chatters too much.
    # Use the upper end of the LOWEST admissible block on the grid (connected run, as
    # in E2), not the largest admissible point, so that an isolated admissible island
    # above the first loss of admissibility cannot be selected; the number of blocks is
    # recorded (1 in every mini calibration).
    asc = st[::-1]
    n_blocks = int(asc[0]) + int(np.sum(asc[1:] & ~asc[:-1]))
    top = int(np.flatnonzero(asc)[0])
    while top + 1 < len(asc) and asc[top + 1]:
        top += 1
    i = len(grid) - 1 - top
    if i == 0:
        lo, hi, status = float(grid[0]), None, "cap"
    else:
        lo, hi, status = float(grid[i]), float(grid[i - 1]), "ok"
        for _ in range(S["p2_refine"]):
            inner = np.geomspace(hi, lo, S["n_interior"] + 2)[1:-1]       # descending
            sti = evaluate(inner)
            pts = [hi, *inner.tolist(), lo]
            flags = [False, *sti.tolist(), True]
            j = flags.index(True)
            lo, hi = float(pts[j]), float(pts[j - 1])
    h_eval = C.H_FACTOR * lo
    ver = evaluate([h_eval])
    return {"h_max": lo, "h_unstable_above": hi, "h_eval": h_eval, "gamma_eval": h_eval / tau,
            "h_admissible_min_grid": float(grid[len(grid) - 1 - int(np.flatnonzero(asc)[0])]),
            "n_admissible_blocks": n_blocks,
            "status": status, "verify_stable_at_h_eval": bool(ver[0]), "evaluations": log}


# ---------------------------------------------------------------------------
# P3: matched peak control effort (continuous time)
# ---------------------------------------------------------------------------
def peak_medians(cal, model, S, gammas):
    """Median peak ||x_dot|| over all calibration trajectories (all seeds)."""
    rec = int(round(S["T"] / S["tau_ct"]))
    peaks = []
    for problem, x0, _, seed in cal:
        _, peak = batch_sim(problem, model, "rk4", S["tau_ct"], S["T"], gammas, x0, seed, probe=True, rec=rec)
        peaks.append(np.where(np.isfinite(peak), peak, np.inf))
    return np.median(np.hstack(peaks), axis=1)


def calibrate_p3(cal, model, ref_model, S, unit=1.0):
    # budget sensitivity (P3_r5 / P3_r30) only in the original unit
    refs = (C.GAMMA_STAR, *C.P3_REF_SENS) if math.isclose(float(unit), 1.0) else (C.GAMMA_STAR,)
    ref_peaks = peak_medians(cal, ref_model, S, list(refs))
    budgets = {f"{g:g}": float(p) for g, p in zip(refs, ref_peaks)}   # budget = linear ZNN peak at gamma_ref
    U_star = budgets[f"{C.GAMMA_STAR:g}"]
    pts = {}                                                         # gamma -> median peak

    def evaluate(gs):
        gs = sorted(set(float(g) for g in gs) - set(pts))
        if gs:
            for g, p in zip(gs, peak_medians(cal, model, S, gs)):
                pts[g] = float(p)

    def bracket(U):
        """(a, b): a = the LARGEST evaluated gamma whose peak is within the
        budget, b = the next evaluated gamma (peak above it).  The peak is not
        monotone in gamma in general (U-shaped on tvqp_eq, where an unconverged
        state keeps a large feed-forward term; singular near t_c for SPTC-NT
        with gamma < 1), so 'largest admissible gain' is the definition.
        (None, g_min) if no evaluated gamma is admissible, (g_max, None) if all
        larger ones are."""
        gs = sorted(pts)
        ok = [pts[g] <= U for g in gs]
        if not any(ok):
            return None, min(gs, key=lambda g: pts[g])
        i = max(j for j, o in enumerate(ok) if o)
        return (gs[i], None) if i == len(gs) - 1 else (gs[i], gs[i + 1])

    evaluate(np.geomspace(C.GAIN_RANGE[0], C.GAIN_RANGE[1], S["p3_grid"]))
    for _ in range(S["p3_refine"]):
        new = []
        for U in budgets.values():
            a, b = bracket(U)
            if a is not None and b is not None:
                new += np.geomspace(a, b, S["n_interior"] + 2)[1:-1].tolist()
        evaluate(new)
    gam, flag = {}, {}
    for name, U in budgets.items():
        a, b = bracket(U)
        if a is None:
            gam[name], flag[name] = b, "infeasible"     # no gamma within the budget: least-effort gamma
        elif b is None:
            gam[name], flag[name] = a, "cap"
        else:
            pa, pb = pts[a], pts[b]
            if np.isfinite(pb) and pa > 0 and pb > pa:     # log-log interpolation inside the bracket
                lg = math.log(a) + (math.log(U) - math.log(pa)) * (math.log(b) - math.log(a)) / (
                    math.log(pb) - math.log(pa))
                gam[name] = float(math.exp(min(max(lg, math.log(a)), math.log(b))))
            else:
                gam[name] = math.sqrt(a * b)
            flag[name] = "ok"
    names = list(budgets)
    ach = peak_medians(cal, model, S, [gam[n] for n in names])
    return {"U_star": U_star, "ref_gamma": C.GAMMA_STAR, "budgets": budgets, "gamma": gam, "flag": flag,
            "achieved": {n: float(a) for n, a in zip(names, ach)},
            "achieved_ratio": {n: float(a / budgets[n]) for n, a in zip(names, ach)},
            "evaluations": sorted([g, p] for g, p in pts.items())}


# ---------------------------------------------------------------------------
# one task
# ---------------------------------------------------------------------------
def run_task(scale, problem_label, model_label, out, unit=1.0):
    S = C.SCALES[scale]
    unit = float(unit)
    pan = {e["label"]: e for e in C.panel()}
    prob_entry = next(p for p in C.PROBLEMS if p["label"] == problem_label)
    entry = pan[model_label]
    t0 = time.perf_counter()
    model = model_for(entry)
    cal = []                                    # (problem, x0, e0, seed) per calibration seed
    e0_u = []                                   # ||E_u(x0, 0)||_2 in the unit the model sees
    for seed in S["calib_seeds"]:
        problem = problem_for(prob_entry, seed, unit)
        x0 = initial_state(problem, model, seed, S["B"])
        d0 = problem.data(0.0)
        # e0: residual in the ORIGINAL unit (P2's stability test compares it with
        # the recorded residual, which ScaledError keeps in the original unit)
        cal.append((problem, x0, problem.residual(x0, d0), seed))
        e0_u.append(np.linalg.norm(np.asarray(problem.E(x0, d0)).reshape(x0.shape[0], -1), axis=1))
    e0 = np.concatenate([c[2] for c in cal])
    e0_u = np.concatenate(e0_u)
    m = cal[0][0].m
    # P1 reference error: RMS entry of E(x0, 0) in the model's unit (= unit x original)
    e_ref = float(np.median(e0_u / math.sqrt(m)))
    walls = {}
    ts = time.perf_counter()
    p1 = calibrate_p1(model, m, e_ref)
    walls["p1"] = time.perf_counter() - ts
    ts = time.perf_counter()
    p2 = calibrate_p2(cal, model, S)
    walls["p2"] = time.perf_counter() - ts
    ts = time.perf_counter()
    ref = model_for(next(e for e in C.PANEL_ALL if e["label"] == "linear"))
    p3 = calibrate_p3(cal, model, ref, S, unit)
    walls["p3"] = time.perf_counter() - ts
    walls["total"] = time.perf_counter() - t0
    res = {"format": "znnbench-E4-calib/2", "scale": scale, "problem": problem_label, "model": model_label,
           "unit": unit, "settings": C.calib_settings(scale), "calib_seeds": list(S["calib_seeds"]),
           "entry": entry, "m": int(m), "e0": e0.tolist(), "e0_unit": e0_u.tolist(),
           "e_ref": e_ref, "e_ref_original_unit": float(np.median(e0 / math.sqrt(m))),
           "p0_gamma": entry["p0_gamma"], "p1": p1, "p2": p2, "p3": p3,
           "wall": walls, "peak_rss_mb": _rss_mb()}
    _atomic_json(res, out)
    print(f"[E4 calib] {problem_label} x {model_label} (unit {unit:g}): e_ref={e_ref:.3g} "
          f"P1 gamma={p1['gamma'][f'{C.GAMMA_STAR:g}']:.4g} ({p1['kappa_mode']}) "
          f"P2 h_max={p2['h_max']:.4g} ({p2['status']}) "
          f"P3 gamma={p3['gamma'][f'{C.GAMMA_STAR:g}']:.4g} (U*={p3['U_star']:.4g}) "
          f"wall={walls['total']:.1f}s rss={res['peak_rss_mb']:.0f}MB", flush=True)
    return res


# ---------------------------------------------------------------------------
# build: calibration tasks / sweep tasks
# ---------------------------------------------------------------------------
def _task_env(scale):
    env = {**THREAD_ENV, "E4_SCALE": scale}
    if os.environ.get("E4_RESULTS"):
        env["E4_RESULTS"] = os.environ["E4_RESULTS"]
    return {k: str(v) for k, v in env.items()}


def build_calib(scale, out=None):
    out = Path(out) if out else C.calib_dir(scale)
    settings = C.calib_settings(scale)
    lines = []
    for unit in (1.0, *C.SCALES[scale]["units"]):
        for prob in C.PROBLEMS:
            for e in C.panel():
                payload = {"settings": settings, "problem": prob, "unit": float(unit),
                           "model": {k: e[k] for k in ("label", "key", "kwargs")}}
                h = hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:10]
                tid = f"E4cal-{scale}-{prob['label']}-{e['label']}{C.unit_suffix(unit)}-{h}"
                done = (out / "out" / f"{tid}.json").resolve()
                cmd = [DEFAULT_PYTHON, "experiments/tasks/E4_calibrate.py", "run", "--scale", scale,
                       "--problem", prob["label"], "--model", e["label"], "--unit", f"{float(unit):g}",
                       "--out", str(done)]
                lines.append(json.dumps({"id": tid, "cmd": cmd, "done": str(done), "cwd": str(ROOT),
                                         "env": _task_env(scale)}))
    out.mkdir(parents=True, exist_ok=True)
    tf = out / "tasks.jsonl"
    tmp = tf.with_name(f".tasks.jsonl.tmp-{os.getpid()}")
    tmp.write_text("\n".join(lines) + "\n")
    os.replace(tmp, tf)
    print(f"{len(lines)} calibration tasks -> {tf}")
    return tf


def build_sweep(scale, out=None, allow_partial=False):
    from znnbench.runner import build, expand
    out = Path(out) if out else C.sweep_dir(scale)
    cfgs = C.get_configs(scale, require_calib=not allow_partial)
    lines, nspec = [], 0
    for c in cfgs:
        tf = build(c, out / c["name"])
        ls = [ln for ln in tf.read_text().splitlines() if ln.strip()]
        n = len(expand(c))
        nspec += n
        print(f"  {c['name']:12s} {n:4d} specs {len(ls):3d} tasks -> {tf}")
        lines += ls
    allf = out / "all_tasks.jsonl"
    tmp = allf.with_name(f".all_tasks.jsonl.tmp-{os.getpid()}")
    tmp.write_text("\n".join(lines) + "\n")
    os.replace(tmp, allf)
    print(f"{nspec} specs, {len(lines)} tasks -> {allf}")
    return allf


def status(scale):
    for name, tf in (("calib", C.calib_dir(scale) / "tasks.jsonl"),
                     ("sweep", C.sweep_dir(scale) / "all_tasks.jsonl")):
        if not tf.exists():
            print(f"{name}: not built ({tf})")
            continue
        ts = [json.loads(ln) for ln in tf.read_text().splitlines() if ln.strip()]
        done = sum(Path(t["done"]).exists() for t in ts)
        print(f"{name}: {done}/{len(ts)} done ({tf})")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("build", "sweep", "status", "run"):
        p = sub.add_parser(name)
        p.add_argument("--scale", choices=sorted(C.SCALES), default=None)
        if name in ("build", "sweep"):
            p.add_argument("--out", default=None)
        if name == "sweep":
            p.add_argument("--allow-partial", action="store_true",
                           help="build P0/P4 only when the calibration is incomplete")
        if name == "run":
            p.add_argument("--problem", required=True)
            p.add_argument("--model", required=True)
            p.add_argument("--unit", type=float, default=1.0, help="error unit factor u (E_u = u E)")
            p.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    scale = args.scale or C.scale_from_env()
    os.environ["E4_SCALE"] = scale
    if args.cmd == "build":
        build_calib(scale, args.out)
    elif args.cmd == "sweep":
        build_sweep(scale, args.out, args.allow_partial)
    elif args.cmd == "status":
        status(scale)
    elif args.cmd == "run":
        run_task(scale, args.problem, args.model, args.out, args.unit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
