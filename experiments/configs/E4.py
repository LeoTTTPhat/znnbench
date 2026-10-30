#!/usr/bin/env python
"""E4 -- fairness re-ranking of "proposed vs baseline" ZNN models.

Protocol: experiments/PLAN.md (E4); exact definitions, rationale and every
deviation are in experiments/E4_NOTES.md.  This file is the single source of
truth for the panel, the as-published (P0) parameters, the scales and the six
protocols; it is read by

* ``experiments/tasks/E4_calibrate.py`` (calibration tasks for P1/P2/P3 and the
  sweep builder), and
* ``python -m znnbench.runner build experiments/configs/E4.py`` (it defines
  ``get_configs()``: one runner config per protocol).

Protocols (all on matrix_inversion n = 4 and tvqp_eq, seeds x B initial states)

  P0   as published: each source's own gamma / activation parameters,
       continuous time (RK4, tau_ct)
  P1   matched linearised gain at t = 0+: gamma * kappa_m = gamma* where
       kappa_m = Phi'(0) g(0) when finite, else the secant gain
       Phi(e_ref)/e_ref g(0) at the reference error e_ref (median RMS entry of
       E(x0, 0) on the calibration seeds), continuous time (RK4, tau_ct)
  P2   each model at 0.8 x its own maximal stable h = gamma*tau (bisection on
       the calibration seeds), explicit Euler, tau_dt = 1e-3
  P3   matched peak control effort max_k ||x_dot||: the LARGEST gamma whose
       median peak on the calibration seeds stays within the budget U* = peak
       of linear ZNN at gamma*, continuous time (RK4, tau_ct)
  P4a  published parameters, explicit Euler, tau = 1e-2
  P4b  published parameters, explicit Euler, tau = 1e-3
  (+ sensitivity: P1 at gamma* in {3, 30}; P1sec = P1 with the secant gain at
     e_ref for every activation; P1pole = P1 with the slowest linearised pole
     matched (IEZNN); P3 with the budget = peak effort of linear ZNN at
     gamma = 5 and gamma = 30)
  (+ unit arms, review 2026-10-18: P0-P4b re-run and re-calibrated with the
     error expressed as E_u = u E, u in SCALES[...]['units'] = {0.1, 10};
     configs E4_<P>_u<u>)

Calibration (P1/P2/P3) is done by custom memguard tasks
(``experiments/tasks/E4_calibrate.py``) on held-out calibration seeds
(100, 101, 102: three problem instances x B initial states); this
file reads their JSON outputs when it builds the P1/P2/P3 configs.  If the
calibration is missing, only P0/P4 are built (a warning is printed; set
``E4_REQUIRE_CALIB=1`` to make it an error).

Scale
-----
Environment ``E4_SCALE`` (or ``ZNNBENCH_SCALE`` / ``ZNN_SCALE``) = ``mini`` |
``full`` (default ``mini``), or ``--scale`` when this file is run as a script.
Results root: ``results/<scale>/E4`` (override with ``E4_RESULTS``).

Build + launch + analyze (from znn-bench/), see E4_NOTES.md:
  python experiments/tasks/E4_calibrate.py build --scale mini
  python tools/memguard.py pool --tasks results/mini/E4/calib/tasks.jsonl --workers 3 --max-gb 40
  python experiments/tasks/E4_calibrate.py sweep --scale mini
  python tools/memguard.py pool --tasks results/mini/E4/sweep/all_tasks.jsonl --workers 3 --max-gb 40
  python experiments/analyze_E4.py --results results/mini/E4 --figdir figures/mini --tabdir results/tables/mini
Running this file prints the panel, the calibration state and the spec counts:
  python experiments/configs/E4.py --scale mini
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]          # .../znn-bench
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# scales
# ---------------------------------------------------------------------------
SCALES = {
    # end-to-end validation (~10 min on 3 workers); coarser continuous-time
    # step and shorter horizon than the protocol
    "mini": dict(T=4.0, tau_ct=1e-3, rec_ct=1, tau_dt=1e-3, rec_dt=1, taus_p4=(1e-2, 1e-3),
                 seeds=(0, 1, 2), B=4, calib_seeds=(100, 101, 102), units=(0.1, 10.0),
                 p2_grid=13, p2_refine=1, p3_grid=11, p3_refine=1, n_interior=6,
                 chunk_size=6, max_wall=900, traces=120),
    # protocol scale (PLAN.md): T = 6 s, RK4 tau = 1e-4 for continuous time
    # (rec_ct = 10 is safe: RK4 chattering of the sign terms at tau = 1e-4 is not
    #  period-2, checked in review -- max over every 10th sample = max over all).
    # Review 2026-10-18: 10 evaluation seeds (clusters) instead of PLAN's 3, because
    # tau_b and its seed-cluster bootstrap CI are the headline numbers of E4 (with
    # 3 clusters the percentile CI is degenerate: only 10 distinct seed multisets).
    "full": dict(T=6.0, tau_ct=1e-4, rec_ct=10, tau_dt=1e-3, rec_dt=1, taus_p4=(1e-2, 1e-3),
                 seeds=tuple(range(10)), B=4, calib_seeds=(100, 101, 102), units=(0.1, 10.0),
                 p2_grid=22, p2_refine=2, p3_grid=15, p3_refine=2, n_interior=6,
                 chunk_size=6, max_wall=7200, traces=300),
}

# unit arms (review 2026-10-18, cross-cutting checklist item 1): the power-type
# activations have no length scale, so every protocol is re-run with the error
# expressed in another unit, E_u = u E (problem wrapper R_common.ScaledError;
# the recorded residual, eps = 1e-3 and the MSSRE stay in the original unit, x
# and therefore ||x_dot|| are unchanged).  All published parameter VALUES are
# kept (gamma, p, r, xi, t_c, Delta): the arm asks how a published parameter
# set behaves when the same problem is written in another unit.  P1/P2/P3 are
# re-calibrated in each unit (e_ref, h_max and the effort-matched gains are all
# unit dependent).
UNIT_WRAPPER_KEY = "experiments.tasks.R_common:ScaledError"
UNIT_PROTOCOLS_BASE = ["P0", "P1", "P2", "P3", "P4a", "P4b"]

# matching levels (shared by both scales)
GAMMA_STAR = 10.0                    # P1 target linearised gain and P3 reference gain
GAMMA_STAR_SENS = (3.0, 30.0)        # P1 sensitivity
P3_REF_SENS = (5.0, 30.0)            # P3 sensitivity: budget = peak of linear ZNN at these gammas
#   (a budget defined as a fraction of U* can fall below the feed-forward floor
#   ||J^-1 dE/dt|| that every model needs at t = 0, e.g. U*/3 < 35 < U* = 51 on
#   matrix_inversion, so the sensitivity budgets are reference peaks instead)
H_FACTOR = 0.8                       # P2: h = H_FACTOR * h_max
GAIN_RANGE = (1e-3, 1e4)             # admissible gamma for every matching search
H_RANGE = (1e-5, 6.0)                # P2 bisection range of h = gamma*tau (E2 uses (0, 3]; IEZNN with
                                     # lambda = gamma^2/4 is Euler-stable up to h = 4, so 3 would cap it)
STAB_RHO = 0.1                       # P2 stability: steady residual <= rho * ||E(x0, 0)||
STAB_GROWTH = 2.0                    # P2 stability: last T/8 max <= growth * previous T/8 max
STAB_TRANSIENT = 10.0                # P2 stability: max_k ||E_k|| <= 10 ||E(x0, 0)|| (no transient blow-up)
E_REF_SMALL = 1e-8                   # "e -> 0" probe (relative to e_ref) for Phi'(0)
EPS = (1e-3, 1e-6)                   # time-to-eps thresholds recorded by the runner
EXTRA = ["initial_residual", "final_residual", "max_residual", "peak_xdot"]

PROTOCOLS = ["P0", "P1", "P2", "P3", "P4a", "P4b"]          # main (fixed order = colour order)
SENS_PROTOCOLS = ["P1_g3", "P1_g30", "P1sec", "P1pole", "P3_r5", "P3_r30"]   # sensitivity
#   P1sec: P1 with the secant gain at e_ref for EVERY activation (the PLAN rule
#   matches Lipschitz activations at Phi'(0) but non-Lipschitz ones at e_ref;
#   P1sec measures how much the ranking depends on that asymmetric choice)
#   P1pole: P1 with the SLOWEST POLE of the linearised (E, z) error dynamics
#   matched instead of the initial gain; differs from P1 only for models with
#   an auxiliary state (IEZNN: double pole gamma/2, so gamma = 2 gamma*).  Its
#   config holds only the pairs whose gain differs from P1; the analysis takes
#   the other pairs from P1 (identical specs, identical run ids).
INHERIT = {"P1pole": "P1"}
#   P1_g1 (review 2026-10-18): P1 at gamma* = 1, the as-published gain of linear
#   ZNN, i.e. matching WITHOUT raising the gain level.  It separates the effect of
#   fairness from the effect of the higher gain of P1 (gamma* = 10).  Its gains come
#   from the stored calibration (kappa is affine in gamma at t = 0+ for every model
#   of the panel, so gamma(gamma*) = (gamma* - a)/b exactly; checked against the
#   bisection results at gamma* in {3, 10, 30} when the configs are built).
#   P2_h0.25 (review 2026-10-18): P2 at 0.25 h_max instead of 0.8 h_max (P2 puts
#   chattering models at their largest stable step, i.e. at their largest floor).
P1_LOW_GAMMA_STAR = 1.0
P2_LOW_FACTOR = 0.25
SENS_PROTOCOLS += ["P1_g1", "P2_h0.25"]
PROTOCOL_INFO = {
    "P0": "as published, continuous time (RK4)",
    "P1": "matched linearised/secant gain gamma*kappa = 10, continuous time",
    "P2": "0.8 x own max admissible Euler step h (bounded, residual contracts >= 10x, no growth or blow-up; a residual-size limit for chattering models), Euler tau = 1e-3",
    "P3": "matched peak ||x_dot|| (budget = linear ZNN at gamma = 10), continuous time",
    "P4a": "published parameters, Euler tau = 1e-2",
    "P4b": "published parameters, Euler tau = 1e-3",
    "P1_g3": "P1 with gamma* = 3", "P1_g30": "P1 with gamma* = 30",
    "P1sec": "P1 with the secant gain at e_ref for every activation",
    "P1pole": "P1 with the slowest linearised pole matched (IEZNN gamma = 2 gamma*)",
    "P1_g1": "P1 with gamma* = 1 (the as-published gain level of linear ZNN)",
    "P2_h0.25": "P2 at 0.25 x own max admissible h",
    "P3_r5": "P3 with budget = linear ZNN peak at gamma = 5",
    "P3_r30": "P3 with budget = linear ZNN peak at gamma = 30",
}


def unit_suffix(u):
    """'' for the original unit, else '_u<u>' (e.g. '_u0.1', '_u10')."""
    return "" if math.isclose(float(u), 1.0) else f"_u{float(u):g}"


def split_protocol(p):
    """'P2_u10' -> ('P2', 10.0); 'P1_g3' -> ('P1_g3', 1.0)."""
    if "_u" in p:
        base, u = p.rsplit("_u", 1)
        try:
            return base, float(u)
        except ValueError:
            pass
    return p, 1.0


def unit_protocols(units):
    return [f"{b}{unit_suffix(u)}" for u in units for b in UNIT_PROTOCOLS_BASE]


for _s in SCALES.values():
    for _p in unit_protocols(_s["units"]):
        _b, _u = split_protocol(_p)
        PROTOCOL_INFO.setdefault(_p, f"{PROTOCOL_INFO[_b]}; error in units E_u = {_u:g} E")

# ---------------------------------------------------------------------------
# problems (tvqp_eq: CAAI-TIT 2021 benchmark (18) data, deterministic; the
# seed then only changes the initial states)
# ---------------------------------------------------------------------------
PROBLEMS = [
    {"key": "matrix_inversion", "kwargs": {"n": 4, "kappa": 10, "seed": "$seed"},
     "label": "matrix_inversion"},
    {"key": "tvqp_eq", "kwargs": {}, "label": "tvqp_eq"},
]
PROBLEM_DISPLAY = {"matrix_inversion": "matrix inversion ($n=4$)", "tvqp_eq": "TV-QP (eq.)"}

# ---------------------------------------------------------------------------
# panel: label (unique), registry key, kwargs, P0 gamma, publication year,
# figure group/key (experiments/keys.py), provenance of the P0 values
# ---------------------------------------------------------------------------
PANEL_ALL = [
    dict(label="linear", key="znn", kwargs={"activation": "linear"}, p0_gamma=1.0, year=2002,
         group="act", gkey="linear", mkey="znn", display="linear ZNN",
         source="Zhang, Jiang, Wang, IEEE TNN 13(5) 2002; Zhang & Ge, IEEE TNN 16(6) 2005",
         p0_note="gamma = 1: Zhang & Ge 2005, Sec. V (primary, docs/spec/problems_matrix.md)"),
    dict(label="power_sigmoid", key="znn",
         kwargs={"activation": "power_sigmoid", "p": 3, "xi": 4.0}, p0_gamma=1.0, year=2005,
         group="act", gkey="power_sigmoid", display="power-sigmoid",
         source="Zhang & Ge, IEEE TNN 16(6) 2005, act. (iv)",
         p0_note="gamma = 1, p = 3, xi = 4: Zhang & Ge 2005, Sec. V (primary, problems_matrix.md)"),
    dict(label="sign_bi_power", key="znn",
         kwargs={"activation": "sign_bi_power", "r": 0.5}, p0_gamma=1.0, year=2013,
         group="act", gkey="sign_bi_power", display="sign-bi-power",
         source="Li, Chen, Liu, Neural Process. Lett. 37 2013",
         p0_note="ASSUMPTION gamma = 1 (not in specs), r = 0.5 (zoo default; source value not in specs)"),
    dict(label="fixed_time", key="znn",
         kwargs={"activation": "fixed_time", "k1": 1.0, "k2": 1.0, "k3": 1.0, "p": 0.5, "q": 2.0},
         p0_gamma=1.0, year=2022, group="act", gkey="fixed_time", display="fixed-time",
         source="Jin, Zhu, Zhao, Chen, Appl. Soft Comput. 130 2022",
         p0_note="ASSUMPTION gamma = 1; k = (1, 1, 1), p = 0.5, q = 2 zoo defaults (source values not in specs)"),
    dict(label="predefined_exp", key="znn",
         kwargs={"activation": "predefined_exp", "p": 0.5, "k1": 1.0, "k2": 1.0, "k3": 1.0},
         p0_gamma=1.0, year=2019, group="act", gkey="predefined_exp", display="predefined-time (F2)",
         source="Li, Liao, Xiao, Lu, Neurocomputing 337 2019 (F2)",
         p0_note="ASSUMPTION gamma = 1; p = 0.5, k = (1, 1, 1) zoo defaults (bound t_c <= 1/(gamma k1) = 1 s)"),
    dict(label="vpznn", key="vpznn", kwargs={"p": 1.0}, p0_gamma=1.0, year=2018,
         group="model", gkey="vpznn", display="VP-ZNN",
         source="Zhang et al., IEEE TCYB 48(11) 2018",
         p0_note="gamma = 1 reproduces the source (variants.md); g(t) = t^p + p with p = 1 (ASSUMPTION)"),
    dict(label="sptc_nt_znn", key="sptc_nt_znn", kwargs={"t_c": 1.0, "p": 0.5, "noise_bound": 0.5},
         p0_gamma=2.0, year=2024, group="model", gkey="sptc_nt_znn", display="SPTC-NT-ZNN",
         source="Li, Guo, Ma, Pan, IEEE TIE 71(1) 2024",
         p0_note="gamma = 2, t_c = 1, p = 0.5, zeta = 5 Delta: Yang et al. 2025 comparison (variants.md); "
                 "Delta = 0.5 ASSUMPTION"),
    dict(label="ieznn", key="ieznn", kwargs={}, p0_gamma=1.0, year=2016,
         group="model", gkey="ieznn", display="IEZNN",
         source="Jin, Zhang, Li, IEEE TNNLS 27(12) 2016",
         p0_note="ASSUMPTION gamma = 1 (not in specs); lambda = gamma^2/4 (ZNN-Bench mapping, variants.md)"),
    dict(label="sptc_an_foznn", key="sptc_an_foznn",
         kwargs={"alpha": 0.5, "t_c": 1.0, "noise_bound": 0.5}, p0_gamma=2.0, year=2025,
         group="model", gkey="sptc_an_foznn", display="SPTC-AN-FOZNN", optional=True,
         source="Yang et al., Neural Netw. 186 2025",
         p0_note="gamma = 2, t_c = 1 (TVQP example), alpha = 0.5 (robot example), zeta = 5 Delta "
                 "(primary, variants.md); Delta = 0.5, t_floor = 1e-3 ASSUMPTIONS"),
]
# predefined-time activation: predefined_exp (Li et al. 2019, the first
# predefined-time ZNN); pvaf1 is the fallback if predefined_exp is unavailable
PT_FALLBACK = dict(label="pvaf1", key="znn", kwargs={"activation": "pvaf1"}, p0_gamma=1.0, year=2022,
                   group="act", gkey="pvaf1", display="PVAF-1",
                   source="Zhou & Tan, Bull. Pol. Acad. Sci. 70(3) 2022",
                   p0_note="ASSUMPTION gamma = 1; zoo defaults (paper prints no a, b, p, q)")


def _available():
    import znnbench
    from znnbench import activations
    av = znnbench.available()
    return set(av["models"]), set(activations.available())


def panel():
    """Panel entries whose keys are registered (conditional at build time)."""
    models, acts = _available()
    out = []
    for e in PANEL_ALL:
        ent = dict(e)
        act = ent["kwargs"].get("activation")
        if ent["key"] not in models or (act is not None and act not in acts):
            if ent["label"] == "predefined_exp" and "pvaf1" in acts and "znn" in models:
                ent = dict(PT_FALLBACK)
            else:
                continue
        out.append(ent)
    return out


def newest_label(pan=None):
    pan = pan or panel()
    return max(pan, key=lambda e: e["year"])["label"]


# ---------------------------------------------------------------------------
# scale / paths
# ---------------------------------------------------------------------------
def scale_from_env(default="mini"):
    s = os.environ.get("E4_SCALE") or os.environ.get("ZNNBENCH_SCALE") or os.environ.get("ZNN_SCALE") or default
    if s not in SCALES:
        raise ValueError(f"E4 scale must be one of {sorted(SCALES)}, got {s!r}")
    return s


def results_root(scale):
    env = os.environ.get("E4_RESULTS")
    return Path(env) if env else ROOT / "results" / scale / "E4"


def calib_dir(scale):
    return results_root(scale) / "calib"


def sweep_dir(scale):
    return results_root(scale) / "sweep"


def calib_settings(scale):
    """Everything a calibration task depends on (hashed into its task id)."""
    S = SCALES[scale]
    return {"scale": scale, "T": S["T"], "tau_ct": S["tau_ct"], "tau_dt": S["tau_dt"], "B": S["B"],
            "calib_seeds": list(S["calib_seeds"]), "p2_grid": S["p2_grid"], "p2_refine": S["p2_refine"],
            "p3_grid": S["p3_grid"], "p3_refine": S["p3_refine"], "n_interior": S["n_interior"],
            "gamma_star": GAMMA_STAR, "gamma_star_sens": list(GAMMA_STAR_SENS),
            "p3_ref_sens": list(P3_REF_SENS),
            "h_factor": H_FACTOR, "gain_range": list(GAIN_RANGE), "h_range": list(H_RANGE),
            "stab_rho": STAB_RHO, "stab_growth": STAB_GROWTH, "stab_transient": STAB_TRANSIENT, "e_ref_small": E_REF_SMALL,
            "calib_version": 6}


def load_calibration(scale):
    """{(problem_label, model_label, unit): calibration dict} of the current
    build (only the done files listed in calib/tasks.jsonl; unit = 1.0 for the
    original error unit)."""
    cdir = calib_dir(scale)
    tf = cdir / "tasks.jsonl"
    out = {}
    if not tf.exists():
        return out
    for line in tf.read_text().splitlines():
        if not line.strip():
            continue
        t = json.loads(line)
        p = Path(t["done"])
        if not p.exists():
            p = cdir / "out" / p.name
        if p.exists():
            c = json.loads(p.read_text())
            out[(c["problem"], c["model"], float(c.get("unit", 1.0)))] = c
    return out


# ---------------------------------------------------------------------------
# runner configs
# ---------------------------------------------------------------------------
def _close(a, b):
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=0.0)


def _model_entry(e):
    return {"key": e["key"], "kwargs": dict(e["kwargs"]), "label": e["label"]}


def unit_wrappers(unit):
    """Runner ``wrappers`` alternatives for an error unit (None = no wrapper:
    the unit-1 configs stay byte-identical to the pre-review ones)."""
    if math.isclose(float(unit), 1.0):
        return None
    return [[{"key": UNIT_WRAPPER_KEY, "target": "problem", "kwargs": {"scale": float(unit)}}]]


def protocol_config(name, scale, disc, tau, rec, gain_field, gains, pan, unit=1.0):
    """One runner config; ``gains[(problem_label, model_label)]`` is the gamma
    (gain_field 'gamma', mode fixed_gamma) or h (gain_field 'h', mode
    fixed_h) of that pair.  Pairs without a value are excluded (they count as
    censored = worst in the analysis); per-pair gains are enforced with
    exclude rules (all other gain values of the list).  unit != 1 wraps the
    problem in R_common.ScaledError(scale=unit)."""
    S = SCALES[scale]
    uniq = []
    for v in gains.values():
        if not any(_close(v, u) for u in uniq):
            uniq.append(float(v))
    uniq.sort()
    exclude = []
    for prob in PROBLEMS:
        for e in pan:
            key = (prob["label"], e["label"])
            if key not in gains:
                exclude.append({"problem": prob["key"], "model": e["label"]})
                continue
            others = [u for u in uniq if not _close(u, gains[key])]
            if others:
                exclude.append({"problem": prob["key"], "model": e["label"], gain_field: others})
    cfg = {
        "name": name,
        "problems": PROBLEMS,
        "models": [_model_entry(e) for e in pan],
        "discretizations": [disc],
        "taus": [tau],
        ("gammas" if gain_field == "gamma" else "hs"): uniq,
        "mode": "fixed_gamma" if gain_field == "gamma" else "fixed_h",
        "seeds": list(S["seeds"]),
        "B": S["B"],
        "T": S["T"],
        "record_every": rec,
        "metrics": {"eps": list(EPS), "frac": 0.25, "extra": list(EXTRA)},
        "traces": {"points": S["traces"]},
        "max_wall": S["max_wall"],
        "exclude": exclude,
        "chunk_size": S["chunk_size"],
        "balance": True,
    }
    w = unit_wrappers(unit)
    if w is not None:
        cfg["wrappers"] = w
    return cfg


def p0_gains(pan):
    return {(p["label"], e["label"]): float(e["p0_gamma"]) for p in PROBLEMS for e in pan}


def p1_gamma_affine(c, gs):
    """P1 gain for target gs from a calibration record, using that kappa(gamma)
    is affine in gamma at t = 0+ (true for every panel model: gamma Phi(e)/e g(0),
    IEZNN gamma, SPTC-AN a + b gamma): kappa = a + b gamma with
    kappa(GAIN_RANGE[0]) and kappa(1) stored by the calibration.  Returns the
    gain floor when a >= gs (not matchable), like the bisection."""
    p1 = c["p1"]
    g0 = GAIN_RANGE[0]
    k1, k0 = float(p1["kappa_unit"]), float(p1["kappa_affine_offset"])
    b = (k1 - k0) / (1.0 - g0)
    a = k1 - b
    if not (b > 0) or a >= gs:
        return g0
    return min(max((gs - a) / b, g0), GAIN_RANGE[1])


def get_configs(scale=None, require_calib=None):
    """Runner configs: the main protocols and the sensitivity configs in the
    original unit, then the main protocols in every unit arm (SCALES[...]['units'])."""
    scale = scale or scale_from_env()
    S = SCALES[scale]
    pan = panel()
    if require_calib is None:
        require_calib = os.environ.get("E4_REQUIRE_CALIB", "0") == "1"
    g0 = p0_gains(pan)
    cal = load_calibration(scale)
    cfgs = []
    for u in (1.0, *S["units"]):
        sfx = unit_suffix(u)
        cfgs += [protocol_config(f"E4_P0{sfx}", scale, "rk4", S["tau_ct"], S["rec_ct"], "gamma", g0, pan, u),
                 protocol_config(f"E4_P4a{sfx}", scale, "euler", S["taus_p4"][0], S["rec_dt"], "gamma", g0, pan, u),
                 protocol_config(f"E4_P4b{sfx}", scale, "euler", S["taus_p4"][1], S["rec_dt"], "gamma", g0, pan, u)]
        need = [(p["label"], e["label"]) for p in PROBLEMS for e in pan]
        missing = [k for k in need if (*k, u) not in cal]
        if missing:
            msg = (f"E4: calibration (unit {u:g}) missing for {len(missing)} of {len(need)} (problem, model) "
                   f"pairs in {calib_dir(scale)}; building P0/P4 only for this unit. Run "
                   f"experiments/tasks/E4_calibrate.py first.")
            if require_calib:
                raise RuntimeError(msg)
            print("WARNING " + msg, file=sys.stderr)
            continue

        def gsel(fn, u=u, need=need):
            out = {}
            for k in need:
                v = fn(cal[(*k, u)])
                if v is not None and math.isfinite(v) and v > 0:
                    out[k] = float(v)
            return out

        def g_p1(gs, gsel=gsel):
            return gsel(lambda c: c["p1"]["gamma"][f"{gs:g}"])

        def g_p3(gref, gsel=gsel):
            return gsel(lambda c: c["p3"]["gamma"].get(f"{gref:g}"))

        cfgs += [
            protocol_config(f"E4_P1{sfx}", scale, "rk4", S["tau_ct"], S["rec_ct"], "gamma", g_p1(GAMMA_STAR), pan, u),
            protocol_config(f"E4_P2{sfx}", scale, "euler", S["tau_dt"], S["rec_dt"], "h",
                            gsel(lambda c: c["p2"]["h_eval"]), pan, u),
            protocol_config(f"E4_P3{sfx}", scale, "rk4", S["tau_ct"], S["rec_ct"], "gamma", g_p3(GAMMA_STAR), pan, u),
        ]
        if sfx:
            continue                                   # sensitivity configs: original unit only
        for gs in GAMMA_STAR_SENS:
            cfgs.append(protocol_config(f"E4_P1_g{gs:g}", scale, "rk4", S["tau_ct"], S["rec_ct"], "gamma",
                                        g_p1(gs), pan))
        cfgs.append(protocol_config("E4_P1sec", scale, "rk4", S["tau_ct"], S["rec_ct"], "gamma",
                                    gsel(lambda c: c["p1"]["gamma_secant"][f"{GAMMA_STAR:g}"]), pan))
        # P1pole: only the pairs whose pole-matched gain differs from the P1 gain
        # (the analysis inherits the others from P1, INHERIT)
        gp1, gpole = g_p1(GAMMA_STAR), gsel(lambda c: c["p1"].get("gamma_pole", {}).get(f"{GAMMA_STAR:g}"))
        gpole = {k: v for k, v in gpole.items() if k in gp1 and not _close(v, gp1[k])}
        if gpole:
            cfgs.append(protocol_config("E4_P1pole", scale, "rk4", S["tau_ct"], S["rec_ct"], "gamma", gpole, pan))
        for gref in P3_REF_SENS:
            cfgs.append(protocol_config(f"E4_P3_r{gref:g}", scale, "rk4", S["tau_ct"], S["rec_ct"], "gamma",
                                        g_p3(gref), pan))
        # P1_g1: affine-kappa gains, verified against the bisection at the calibrated targets
        for k in need:
            c = cal[(*k, u)]
            for gs in (GAMMA_STAR, *GAMMA_STAR_SENS):
                ga, gb = p1_gamma_affine(c, gs), c["p1"]["gamma"][f"{gs:g}"]
                if not math.isclose(ga, gb, rel_tol=1e-5):
                    raise RuntimeError(f"E4 P1_g1: affine kappa model fails for {k} at gamma* = {gs:g} "
                                       f"({ga:.6g} vs bisection {gb:.6g})")
        cfgs.append(protocol_config(f"E4_P1_g{P1_LOW_GAMMA_STAR:g}", scale, "rk4", S["tau_ct"], S["rec_ct"],
                                    "gamma", gsel(lambda c: p1_gamma_affine(c, P1_LOW_GAMMA_STAR)), pan))
        cfgs.append(protocol_config(f"E4_P2_h{P2_LOW_FACTOR:g}", scale, "euler", S["tau_dt"], S["rec_dt"], "h",
                                    gsel(lambda c: P2_LOW_FACTOR * c["p2"]["h_max"]), pan))
    return cfgs


def protocol_of(config_name):
    return config_name[3:] if config_name.startswith("E4_") else config_name


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="print the E4 panel, calibration state and spec counts")
    ap.add_argument("--scale", choices=sorted(SCALES), default=None)
    args = ap.parse_args(argv)
    scale = args.scale or scale_from_env()
    from znnbench.runner import expand
    pan = panel()
    print(f"scale={scale} results={results_root(scale)}")
    for e in pan:
        print(f"  {e['label']:15s} {e['key']:14s} P0 gamma={e['p0_gamma']:<4g} year={e['year']}  {e['p0_note']}")
    print(f"newest model: {newest_label(pan)}")
    cal = load_calibration(scale)
    units = (1.0, *SCALES[scale]["units"])
    print(f"calibration files: {len(cal)} / {len(PROBLEMS) * len(pan) * len(units)} (units {units})")
    tot = 0
    for c in get_configs(scale):
        n = len(expand(c))
        tot += n
        print(f"  {c['name']:12s} {n:4d} specs  disc={c['discretizations'][0]} tau={c['taus'][0]:g}")
    print(f"total specs: {tot}")


if __name__ == "__main__":
    main()
