"""E6 -- information realism: where does the time derivative dE/dt come from?

Sweep (see experiments/E6_NOTES.md for the full protocol and rationale)

* model: linear ZNN (``znn``), fixed-h mode (gamma = h / tau);
* information regimes (7):
    - ``analytic``  exact dE/dt; measurement noise sigma only on the data the
                    model sees (simulate ``data_noise``);
    - ``bd1``/``bd2`` causal backward differences of the noisy samples
                    (simulate ``info`` + ``data_noise``);
    - ``savgol``    causal Savitzky-Golay differentiator (window 5, degree 2),
    - ``alpha_beta`` alpha-beta tracker (alpha 0.5, Benedict-Bordner beta),
    - ``kalman``    constant-velocity Kalman filter (Kalata index 1),
    - ``bd5``       EXTENSION: savgol with window 6 / degree 5 = the exact
                    5th-order backward difference (the estimator order an
                    order-6 look-ahead formula needs to keep its order),
  the four estimators through ``wrappers.EstimatedDerivativeProblem`` with
  ``meas_std = sigma`` and ``info='analytic', data_noise=0``
  (docs/spec/noise_info.md sec. 2.3, item 6);
* measurement noise sigma in {0, 1e-8, 1e-6, 1e-4, 1e-2};
* formulas: euler, taylor_zhang, the highest-order ZeaD formula of the
  canonical figure group (keys.GROUPS['disc']: zead_7i_cang2024 when
  registered, else zead_6i_guo2017), the highest-order ZeaD formula overall
  and the highest-order Uhlig formula -- each chosen at BUILD time among the
  keys present in ``znnbench.available()`` and stable at the common h with a
  20 % margin, so a rebuild picks up formulas registered later; a higher-
  order key of a role that is NOT stable at h = 0.1 with that margin (e.g.
  zead_7i_cang2024, h_max = 0.116) runs in the own-h_opt protocol only;
* tau in {1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4}; two fixed-h protocols:
  common h = 0.1 for every formula, and each formula at its own optimal
  step h_opt (argmin of the spectral radius of the linear-ZNN error
  polynomial, zead.lookahead_stability);
* problems: linear_system and matrix_inversion (n = 4, kappa = 10), one
  instance per seed; one horizon T = 12 s for every tau (steady window
  [9, 12] s; a tau-dependent T would shift the window's phase);
* 10 seeds x B = 4 initial states (full; DEVIATION from the default 3 seeds:
  the steady-state MSSRE of linear ZNN does not depend on x0, so the B states
  of one seed are replicas and the seed is the statistical unit -- seeds 0-2
  are a subset, so the 3 x 4 protocol can be recovered from the same runs);
  matrix_inversion (robustness check) runs at the common h only, seeds 0-4.

Scale: environment variable ``E6_SCALE`` (or ``ZNN_SCALE``) = ``full``
(default) or ``mini``, read when the runner imports this file, or
``--scale`` when the file is run as a script::

    python experiments/configs/E6.py --scale mini --out results/mini/E6
    E6_SCALE=full python -m znnbench.runner build experiments/configs/E6.py \
        --out results/E6 --balance

Running it as a script also writes ``<out>/e6_meta.json`` (formula roles,
orders, stability intervals and h values).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

H_COMMON = 0.1                 # common fixed h (= gamma * tau), inside every selected interval
STAB_MARGIN = 1.2              # selected formulas need h_max >= STAB_MARGIN * H_COMMON
T_FIXED = 12.0                 # one horizon for every tau (steady window = [9, 12] s for all runs)
try:                           # canonical figure group (colour slots), kept in sync with keys.py
    from experiments.keys import GROUPS as _GROUPS
    CANONICAL_DISC = list(_GROUPS["disc"])
except Exception:  # noqa: BLE001
    CANONICAL_DISC = ["euler", "taylor_zhang", "zead_5i", "zead_6i_guo2017", "zead_7i_cang2024",
                      "uhlig_3_4", "uhlig_4_6", "rk4"]

ESTIMATORS = {                 # info label -> EstimatedDerivativeProblem kwargs
    "savgol": {"method": "savgol", "window": 5, "order": 2},
    "alpha_beta": {"method": "alpha_beta"},
    "kalman": {"method": "kalman"},
    "bd5": {"method": "savgol", "window": 6, "order": 5},
}
SIM_INFOS = ["analytic", "bd1", "bd2"]

SCALES = {
    # full: linear_system = the reference problem (both h protocols, 10 seeds);
    # matrix_inversion = robustness check of the same linear structure, trimmed
    # (review 2026-10-18) to the common h and seeds 0-4: the own-h_opt confound is
    # shown on linear_system, and 5 seed clusters suffice for a "same pattern"
    # check.  33,075 specs, 8.40e8 steps, ~34 core-hours (was 53,900 / 1.37e9 / ~56).
    "full": dict(taus=[1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4],
                 sigmas=[0.0, 1e-8, 1e-6, 1e-4, 1e-2],
                 seeds=list(range(10)), B=4, chunk_size=40, traces=64,
                 problems=["linear_system", "matrix_inversion"],
                 mi_seeds=[0, 1, 2, 3, 4], mi_hopt=False),
    # mini validation (~5.8e6 steps, ~8 min on 3 workers): fewer tau/sigma, 3 seeds
    # for both problems (>= 3 seed clusters, the minimum for a cluster-bootstrap CI),
    # matrix_inversion only at the common h (as in the full design)
    "mini": dict(taus=[1e-1, 3e-2, 1e-2, 3e-3],
                 sigmas=[0.0, 1e-6, 1e-2],
                 seeds=[0, 1, 2], B=2, chunk_size=24, traces=64,
                 problems=["linear_system", "matrix_inversion"],
                 mi_seeds=None, mi_hopt=False),
}


def scale_from_env(default: str = "full") -> str:
    s = os.environ.get("E6_SCALE") or os.environ.get("ZNN_SCALE") or default
    if s not in SCALES:
        raise ValueError(f"E6 scale must be one of {sorted(SCALES)}, got {s!r}")
    return s


def T_of_tau(tau: float) -> float:
    """Horizon of a run.  Deliberately independent of tau: MSSRE is the max of
    a time-varying forced response over the steady window [0.75 T, T], so a
    tau-dependent T would move the window to a different phase of the data
    and distort MSSRE-vs-tau slopes (observed: up to ~0.3 in the fitted order
    in a first mini run with T = max(6, 1000 tau)).  T = 12 s leaves >= 90
    steps (h = 0.1, gamma = 1, decay e^-9) before the window at tau = 0.1,
    300 at 3e-2 (rho <= 0.95 -> < 2e-7) and >= 900 below."""
    return T_FIXED


def _disc_props(key: str) -> dict:
    """expected order, look-ahead coefficients and linear-ZNN interval."""
    from znnbench import make_discretization
    from znnbench.discretizations.zead import lookahead_stability
    d = make_discretization(key)
    desc = d.describe() if callable(getattr(d, "describe", None)) else {}
    coeffs = desc.get("coeffs")
    order = getattr(d, "expected_order", None)
    props = {"key": key, "expected_order": None if order is None else int(order),
             "n_instants": None if coeffs is None else len(coeffs),
             "h_max": None, "h_opt": None, "rho_opt": None}
    if coeffs is not None:
        try:
            st = lookahead_stability(coeffs)
            props.update(h_max=st["h_max"], h_opt=st["h_opt"], rho_opt=st["rho_opt"])
        except Exception:  # noqa: BLE001  (unknown formula type: no interval)
            pass
    return props


def _eligible(p: dict) -> bool:
    if p["expected_order"] is None:
        return False
    return p["h_max"] is None or p["h_max"] >= STAB_MARGIN * H_COMMON


def select_formulas(available=None) -> list:
    """Formula roles, chosen among the registered keys (build time).

    For each role (canonical-group ZeaD, highest-order ZeaD, highest-order
    Uhlig) the highest-order key (ties: canonical figure group first, then
    larger h_max) is taken twice: among the keys stable at the common h with
    the 20 % margin (runs in both h protocols) and among all keys with a
    stable interval; if the latter differs it is added for the own-h_opt
    protocol only (e.g. zead_7i_cang2024: h_max = 0.116, so h = 0.1 sits at
    the stability edge, rho = 0.98).

    Returns a list of dicts {key, role, expected_order, h_max, h_opt,
    h_opt_used, protocols}.
    """
    import znnbench
    avail = set(available if available is not None else znnbench.available()["discretizations"])
    props = {}
    for k in sorted(avail):
        if k in ("euler", "taylor_zhang") or (k.startswith("zead_") and "general" not in k) \
                or k.startswith("uhlig_"):
            try:
                props[k] = _disc_props(k)
            except Exception as exc:  # noqa: BLE001  (a key that needs kwargs or fails: skip it)
                print(f"E6: skipping discretization {k!r}: {exc!r}", file=sys.stderr)
    runnable = [p for p in props.values() if p["expected_order"] is not None and
                (p["h_max"] is None or (p["h_max"] > 0 and p["h_opt"]))]

    def best(cands):
        if not cands:
            return None
        return max(cands, key=lambda p: (p["expected_order"], p["key"] in CANONICAL_DISC,
                                         p["h_max"] or 0.0, p["key"]))["key"]

    families = [
        ("canonical ZeaD", lambda p: p["key"].startswith("zead_") and p["key"] in CANONICAL_DISC),
        ("highest-order ZeaD", lambda p: p["key"].startswith("zead_")),
        ("highest-order Uhlig", lambda p: p["key"].startswith("uhlig_")),
    ]
    sel = {"euler": ["Euler (order 2)", {"common", "hopt"}],
           "taylor_zhang": ["Taylor-Zhang (order 3)", {"common", "hopt"}]}
    for role, pred in families:
        k_common = best([p for p in runnable if pred(p) and _eligible(p)])
        k_any = best([p for p in runnable if pred(p)])
        for k, prot in ((k_common, {"common", "hopt"}), (k_any, {"hopt"})):
            if k is None:
                continue
            if k in sel:
                sel[k][1] |= prot
                if role not in sel[k][0]:
                    sel[k][0] += f" + {role}"
            else:
                sel[k] = [role if prot == {"common", "hopt"} else f"{role} (own h_opt only)", set(prot)]
    out = []
    for k, (role, prot) in sel.items():
        if k not in props:
            continue
        p = dict(props[k])
        p["role"] = role
        p["protocols"] = sorted(prot)
        h = p["h_opt"] if p["h_opt"] else H_COMMON
        p["h_opt_used"] = float(f"{h:.3g}")
        out.append(p)
    return sorted(out, key=lambda p: (p["expected_order"], p["key"].startswith("uhlig_"), p["key"]))


def build_config(scale: str = "full", formulas=None) -> dict:
    sc = SCALES[scale]
    formulas = formulas if formulas is not None else select_formulas()
    keys = [f["key"] for f in formulas]
    hopt = {f["key"]: f["h_opt_used"] for f in formulas}
    hs = sorted({H_COMMON, *hopt.values()})
    taus = list(sc["taus"])
    sig_pos = [s for s in sc["sigmas"] if s > 0]

    def pk(key):
        return {"key": key, "kwargs": {"n": 4, "kappa": 10, "seed": "$seed", "T": T_FIXED}}

    infos = [{"info": i, "data_noise": s} for i in SIM_INFOS for s in sc["sigmas"]]
    wrappers = [None] + [
        {"key": "EstimatedDerivativeProblem", "target": "problem",
         "kwargs": {**kw, "meas_std": {"$sweep": list(sc["sigmas"])}, "seed": "$seed"}}
        for kw in ESTIMATORS.values()]

    exclude = []
    # estimators: measurement noise only through meas_std, with info='analytic'
    exclude.append({"wrappers": "EstimatedDerivativeProblem", "info": ["bd1", "bd2"]})
    if sig_pos:
        exclude.append({"wrappers": "EstimatedDerivativeProblem", "data_noise": sig_pos})
    # h protocol: each formula at H_COMMON and at its own h_opt only
    for f in formulas:
        k = f["key"]
        keep = [hopt[k]] + ([H_COMMON] if "common" in f.get("protocols", ["common"]) else [])
        other = [h for h in hs if all(abs(h - v) > 1e-12 for v in keep)]
        if other:
            exclude.append({"disc": k, "h": other})
    # scale-specific reductions for the second problem
    if "matrix_inversion" in sc["problems"]:
        if sc["mi_seeds"] is not None:
            drop = [s for s in sc["seeds"] if s not in sc["mi_seeds"]]
            if drop:
                exclude.append({"problem": "matrix_inversion", "seed": drop})
        if not sc["mi_hopt"]:
            hh = [h for h in hs if abs(h - H_COMMON) > 1e-12]
            if hh:
                exclude.append({"problem": "matrix_inversion", "h": hh})

    return {
        "name": "E6" if scale == "full" else f"E6_{scale}",
        "problems": [pk(p) for p in sc["problems"]],
        "models": ["znn"],
        "discretizations": keys,
        "taus": taus,
        "hs": hs,
        "mode": "fixed_h",
        "infos": infos,
        "noises": [None],
        "wrappers": wrappers,
        "seeds": list(sc["seeds"]),
        "B": int(sc["B"]),
        "T": None,                      # problem kwarg T = T_FIXED
        "record_every": 1,
        "metrics": {"eps": [1e-3, 1e-6], "frac": 0.25, "extra": ["final_residual"]},
        "traces": {"points": int(sc["traces"])},
        "max_wall": 1800,
        "exclude": exclude,
        "chunk_size": int(sc["chunk_size"]),
        "balance": True,
    }


def planned_runs(scale: str = "full", formulas=None) -> list:
    """The runs of ``build_config(scale)`` enumerated directly (the runner's
    ``expand`` walks the full Cartesian product before applying the exclude
    rules, ~80 s at full scale).  One dict per spec: problem, disc, h, tau, T,
    steps, info, sigma, seed, B.  ``_main --dry-run`` checks it against expand."""
    sc = SCALES[scale]
    formulas = formulas if formulas is not None else select_formulas()
    out = []
    for prob in sc["problems"]:
        for f in formulas:
            hs = []
            if "common" in f.get("protocols", ["common"]):
                hs.append(H_COMMON)
            if not (prob == "matrix_inversion" and not sc["mi_hopt"]):
                if all(abs(f["h_opt_used"] - h) > 1e-12 for h in hs):
                    hs.append(f["h_opt_used"])
            seeds = sc["seeds"] if (prob != "matrix_inversion" or sc["mi_seeds"] is None) else \
                [x for x in sc["seeds"] if x in sc["mi_seeds"]]
            for h in hs:
                for tau in sc["taus"]:
                    T = T_of_tau(tau)
                    for info in SIM_INFOS + list(ESTIMATORS):
                        for sig in sc["sigmas"]:
                            for seed in seeds:
                                out.append({"problem": prob, "disc": f["key"], "h": h, "tau": tau, "T": T,
                                            "steps": int(round(T / tau)), "info": info, "sigma": sig,
                                            "seed": seed, "B": int(sc["B"])})
    return out


def meta(scale: str, formulas=None) -> dict:
    formulas = formulas if formulas is not None else select_formulas()
    return {"experiment": "E6", "scale": scale, "h_common": H_COMMON, "stab_margin": STAB_MARGIN,
            "T": T_FIXED, "formulas": formulas,
            "estimators": ESTIMATORS, "sim_infos": SIM_INFOS, **{k: v for k, v in SCALES[scale].items()}}


SCALE = scale_from_env()
CONFIG = build_config(SCALE)


def _main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="build the E6 sweep (tasks.jsonl for tools/memguard.py)")
    ap.add_argument("--scale", choices=sorted(SCALES), default=None,
                    help="mini | full (default: $E6_SCALE / $ZNN_SCALE / full)")
    ap.add_argument("--out", required=True, help="output folder, e.g. results/E6 or results/mini/E6")
    ap.add_argument("--chunk-size", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="print counts only")
    args = ap.parse_args(argv)
    scale = args.scale or SCALE
    from znnbench.runner import DEFAULT_PYTHON, MEMGUARD, build, expand, load_tasks
    formulas = select_formulas()
    cfg = build_config(scale, formulas)
    plan = planned_runs(scale, formulas)
    print(f"E6 scale={scale}: formulas={[f['key'] for f in formulas]} hs={cfg['hs']}")
    print(f"  planned: {len(plan)} specs, {sum(r['steps'] for r in plan):.4e} steps x B={cfg['B']}")
    if args.dry_run:
        specs = expand(cfg)          # the runner's own expansion (slow at full scale)
        steps = sum(round(s["problem"]["kwargs"]["T"] / s["tau"]) for s in specs)
        print(f"  expand:  {len(specs)} specs, {steps:.4e} steps "
              f"({'MATCH' if len(specs) == len(plan) else 'MISMATCH'})")
        return 0
    tf = build(cfg, args.out, chunk_size=args.chunk_size)
    with open(Path(args.out) / "specs.jsonl") as fh:
        specs = [json.loads(line) for line in fh if line.strip()]
    steps = sum(round(s["problem"]["kwargs"]["T"] / s["tau"]) for s in specs)
    m = meta(scale, formulas)
    m.update(n_specs=len(specs), total_steps=int(steps))
    Path(args.out, "e6_meta.json").write_text(json.dumps(m, indent=1))
    print(f"{len(load_tasks(tf))} tasks -> {tf}")
    print(f"launch: {DEFAULT_PYTHON} {MEMGUARD} pool --tasks {tf} --workers 16 --max-gb 200")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
