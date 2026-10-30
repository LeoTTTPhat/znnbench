"""Shared definitions of experiment E7 (cost-accuracy Pareto on matrix inversion).

Used by ``experiments/configs/E7.py`` (accuracy sweep through the runner),
``experiments/tasks/E7_timing.py`` (clean per-step timing tasks) and
``experiments/analyze_E7.py``.  Nothing here edits or monkeypatches the zoo:
the only runtime instrumentation is a transparent problem wrapper
(:class:`PrefetchDerivative`) and, in the call-count profile of the timing
task, counting proxies around a *private* problem instance.

Scale
-----
``scale()`` reads ``E7_SCALE`` (or ``ZNN_SCALE``) from the environment:
``mini`` (validation) or ``full`` (paper run, the default).

Methods (``methods()``)
-----------------------
Every method is a (model, discretization, h, info) tuple on
``matrix_inversion`` (E = A(t)X - I, Zhang & Ge 2005):

=========================  ====================  ========================  ==========  ========
label                      model                 discretization            h           info
=========================  ====================  ========================  ==========  ========
znn                        znn                   euler                     0.3         analytic
uhlig_3_4                  znn                   uhlig_3_4                 h_opt       analytic
znn_explicit_inverse       znn_explicit_inverse  euler                     0.3         analytic
gnn                        gnn                   euler                     H_GNN       none
gnn_matched                gnn                   euler                     0.3         none
davidenko                  davidenko             euler                     (unused)    analytic
pc_newton_causal           davidenko             pc_newton_causal          (unused)    analytic
pc_gradient_causal         davidenko             pc_gradient_causal        (unused)    analytic
newton_tracking_causal     davidenko (unused)    newton_tracking_causal    (unused)    analytic
=========================  ====================  ========================  ==========  ========

plus, *if registered at build time*, every model / discretization whose
registry factory lives in a module (or has a key) matching
``EXTRA_HINTS`` (inverse-free DTZNNs, ``structured_extra``, low-complexity
ZNNs).  Extra models are paired with ``euler``, extra discretizations with
``znn`` (or the ``paired_model`` attribute the discretization declares).
h_opt of a formula whose stable interval ends below 0.3 is its
``describe()['h_opt']`` (spectral-radius optimum of the linear-ZNN error
recursion); otherwise h = 0.3.

Model variants (``MODEL_VARIANTS``, review 2026-10-18): extra models run
with explicit kwargs under their own label.  ``lcc_znn_linear`` is LCCZNN
Eq. (12) with the linear scalar activation phi(nu) = nu.  It is the only
converging representative of the scalar-energy (inverse-free, O(mn)) class
from the benchmark's U[-2, 2] starts: the registered default activation (13)
contains exp(nu) and overflows at the first step (nu_0 ~ 1e2-1e5).  It
coincides with the *secondary* restatement (Xiao et al. 2025, Eq. (10)) of
the inverse-free ZNN of Chen et al. (2024), which is deliberately not
registered (primary text unobtainable), so it is labelled "LCCZNN (12),
linear phi" and never as Chen et al.
"""
from __future__ import annotations

import math
import os
from typing import Dict, List, Optional

import numpy as np

# ----------------------------------------------------------------------------
# protocol constants
# ----------------------------------------------------------------------------
PROBLEM = "matrix_inversion"
KAPPA = 10.0
FREQ = 1.0
TAU = 1e-3
H_MAIN = 0.3
#: GNN step: h * sigma_max(A)^2 ~ 1 (sigma_max <= kappa*(1 + 0.2) = 12 for
#: TVMatrix(kappa=10)), i.e. half of the forward-Euler stability limit
#: h < 2/sigma_max^2 of x_dot = -gamma A^T(AX - I).  The matched value h = 0.3
#: (gamma*tau*sigma_max^2 = 19..43 >> 2) is run as ``gnn_matched`` and diverges.
H_GNN = 0.007
H_UHLIG_FALLBACK = 0.0255
#: singular-value bounds of TVMatrix(kappa=10, mod=0.2): s0 in [1, kappa] x (1 +/- 0.2)
SIGMA_LO, SIGMA_HI = 1.0 * (1 - 0.2), KAPPA * (1 + 0.2)
#: tuned arm: h = min(h_opt, 0.8 h_max)
H_EULER_TUNED = 1.0                                   # ZNN-Euler: h_opt = 1 < 0.8 * 2
H_GNN_TUNED = round(0.8 * 2.0 / SIGMA_HI ** 2, 4)     # 0.0111; h_opt = 2/(lo^2+hi^2) = 0.0138 > cap
TUNED_FRAC = 0.8

SCALES = {
    # mini: validation only (small, < 8 min wall with 3 workers)
    # (T = 5 s so that the slowest transient, pc_gradient_causal, has settled before the
    # steady window [3.75, 5] s; with T = 2 s its mini MSSRE was a transient, 65x too large)
    "mini": dict(ns=[4, 16, 32, 64], seeds=[0, 1], B=2, T=5.0, timing_repeats=1, chunk_size=4,
                 t_seeds=[0, 1], t_B=[1, 4], t_steps=300, t_repeats=3, t_warmup=50,
                 t_profile=(30, 130)),
    # full: the paper run (3 problem seeds x B = 4 initial states, T = 6 s)
    "full": dict(ns=[4, 8, 16, 32, 64, 128], seeds=[0, 1, 2], B=4, T=6.0, timing_repeats=5,
                 chunk_size=3, t_seeds=[0, 1, 2], t_B=[1, 4], t_steps=2000, t_repeats=5,
                 t_warmup=200, t_profile=(50, 250)),
}

CORE_LABELS = ["znn", "znn_explicit_inverse", "gnn", "pc_newton_causal", "pc_gradient_causal",
               "newton_tracking_causal", "davidenko", "uhlig_3_4"]
#: gain-sensitivity arm (core methods at their own tuned step); label -> protocol label
TUNED_LABELS = {"znn_tuned": "znn", "znn_explicit_inverse_tuned": "znn_explicit_inverse",
                "gnn_tuned": "gnn"}
#: discretizations that implement a whole tracking algorithm (model = predictor only)
PC_DISCS = ("pc_newton_causal", "pc_gradient_causal", "newton_tracking_causal")
EXTRA_HINTS = ("inverse_free", "structured_extra", "low_complex", "lcc_", "lowcomp")
#: extra-model variants with explicit kwargs: label -> (registry key, kwargs) (see module doc)
MODEL_VARIANTS = {"lcc_znn_linear": ("lcc_znn", {"activation": "linear"})}
#: censored rows (median MSSRE = inf, no clean timing) are drawn at the per-step cost of an
#: operation-identical sibling: matched-gain rows -> the stable row of the same model (same
#: operations, different gain); LCCZNN with the exp-sig activation -> its linear-phi variant
#: (same products, O(B) scalar work apart)
COST_PROXY = {"lcc_znn": "lcc_znn_linear"}


def cost_proxy(label: str) -> Optional[str]:
    """Label whose clean per-step cost stands in for a censored (diverged) row."""
    lab = str(label)
    if lab.endswith("_matched"):
        return lab[: -len("_matched")]
    return COST_PROXY.get(lab)

PRETTY = {
    "znn": "ZNN-Euler (LU solve)",
    "znn_explicit_inverse": "explicit-inverse ZNN",
    "gnn": "GNN ($h\\sigma_{\\max}^2\\approx 1$)",
    "gnn_matched": "GNN (matched $h=0.3$)",
    "pc_newton_causal": "PC-Newton (causal)",
    "pc_gradient_causal": "PC-gradient (causal)",
    "newton_tracking_causal": "Newton tracking (causal)",
    "davidenko": "Davidenko (no correction)",
    "uhlig_3_4": "ZNN-Uhlig 3_4 (LU solve)",
    "broyden_dtznn_zhang2012": "Broyden DTZNN (inverse-free)",
    "cdtznn_mahuang2025": "CDTZNN (inverse-free)",
    "lcc_znn": "LCCZNN (12), exp-sig $\\phi$ (13)",
    "lcc_znn_linear": "LCCZNN (12), linear $\\phi$",
    # not registered (primary source unobtainable); only if it is ever re-registered
    "inverse_free_znn": "NIFZNN as restated by Xiao et al. (2025)",
    "hgnn_znn": "H-GNN-ZNN ($h=0.007$)",
    "fa_gnn_znn": "FA-GNN-ZNN ($h=0.007$)",
    "hgnn_znn_matched": "H-GNN-ZNN (matched $h=0.3$)",
    "fa_gnn_znn_matched": "FA-GNN-ZNN (matched $h=0.3$)",
    "znn_tuned": "ZNN-Euler, tuned $h=1$",
    "znn_explicit_inverse_tuned": "explicit-inverse ZNN, tuned $h=1$",
    "gnn_tuned": "GNN, tuned $h=0.8h_{\\max}$",
    "hgnn_znn_tuned": "H-GNN-ZNN (tuned $h$)",
    "fa_gnn_znn_tuned": "FA-GNN-ZNN (tuned $h$)",
}

#: idealised (minimal) leading-order FLOPs per step for matrix inversion:
#: (number of n x n x n products/solves per trajectory, LU factorisations per step).
#: One n x n matrix product or one LU solve with n right-hand sides = 2 n^3;
#: one LU factorisation = 2/3 n^3 (shared by the whole batch at one t_k).
IDEAL_OPS = {
    "znn": (3, 1),                     # E = AX - I, A_dot X, solve A dX = R
    "uhlig_3_4": (3, 1),               # same per-step work; multistep combination is O(n^2)
    "znn_explicit_inverse": (3, 0),    # E, A_dot X, X R  (no solve)
    "gnn": (2, 0),                     # E, A^T E
    "gnn_matched": (2, 0),
    "gnn_tuned": (2, 0),
    "znn_tuned": (3, 1),
    "znn_explicit_inverse_tuned": (3, 0),
    "davidenko": (2, 1),               # A_dot X, solve
    "pc_newton_causal": (4, 1),        # correction: E, solve; prediction: A_dot X, solve (same LU)
    "pc_gradient_causal": (4, 1),      # correction: E, A^T E; prediction: A_dot X, solve
    "newton_tracking_causal": (2, 1),  # E, solve
}
#: model-internal products per rhs call (not routed through the Problem API)
MODEL_INTERNAL_PRODUCTS = {"znn_explicit_inverse": 1}


def is_matched(label: str) -> bool:
    """Matched-gain (h = 0.3) rows of GNN-type models: recorded (they diverge), never
    timed or ranked."""
    return str(label).endswith("_matched")


def scale(name: Optional[str] = None) -> str:
    s = (name or os.environ.get("E7_SCALE") or os.environ.get("ZNN_SCALE") or "full").lower()
    if s not in SCALES:
        raise ValueError(f"E7 scale must be one of {sorted(SCALES)}, got {s!r}")
    return s


# ----------------------------------------------------------------------------
# method table
# ----------------------------------------------------------------------------
def label_of(model: str, disc: str, h: float) -> str:
    """Method label from the run's (model label, disc, h); registry independent.
    ``model`` is the runner's model label: the registry key for default kwargs,
    the E7 label for a ``MODEL_VARIANTS`` entry (e.g. ``lcc_znn_linear``).

    PC-type discretizations are labelled by the discretization (the model is
    only the predictor).  GNN-type models (``gnn`` and the GNN-ZNN hybrids):
    h = 0.3 -> ``<model>_matched``, h = H_GNN_TUNED -> ``<model>_tuned``,
    otherwise ``<model>``.  ZNN-type models with Euler at h = H_EULER_TUNED ->
    ``<model>_tuned``."""
    h = float(h)
    if disc in PC_DISCS:
        return disc
    if "gnn" in model:
        if abs(h - H_MAIN) < 1e-9:
            return f"{model}_matched"
        if abs(h - H_GNN_TUNED) < 1e-9:
            return f"{model}_tuned"
        return model
    if disc == "euler":
        return f"{model}_tuned" if abs(h - H_EULER_TUNED) < 1e-9 else model
    if model == "znn":
        return disc
    return f"{model}+{disc}"


def info_for(model_key: str) -> str:
    """'none' for models that never read dE/dt (``uses_dEdt = False``): the
    dynamics are bit-identical to info='analytic', but ``simulate`` then skips
    the wasted A_dot X product that it would otherwise evaluate and discard
    (zoo finding 2), so the timing charges only what the method needs."""
    try:
        from znnbench import make_model
        return "analytic" if bool(getattr(make_model(model_key), "uses_dEdt", True)) else "none"
    except Exception:  # noqa: BLE001
        return "analytic"


def _h_for_disc(disc) -> float:
    """0.3, unless the formula's linear-ZNN stability interval ends below it;
    then the spectral-radius optimum h_opt.  The interval comes from
    ``describe()['h_max'/'h_opt']``, else from the discretizer's own
    ``stability()`` method, else from ``lookahead_stability(coeffs)``."""
    try:
        ds = disc.describe()
    except Exception:  # noqa: BLE001
        ds = {}
    h_max, h_opt = ds.get("h_max"), ds.get("h_opt")
    if h_max is None:
        st = None
        try:
            if callable(getattr(disc, "stability", None)):
                st = disc.stability()
            elif getattr(disc, "coeffs", None) is not None:
                from znnbench.discretizations.zead import lookahead_stability
                st = lookahead_stability([float(c) for c in disc.coeffs])
        except Exception:  # noqa: BLE001
            st = None
        if isinstance(st, dict):
            h_max, h_opt = st.get("h_max"), st.get("h_opt")
    if h_max is not None and h_opt and float(h_max) <= H_MAIN:
        return round(float(h_opt), 4)
    return H_MAIN


def _is_extra(key: str, fn) -> bool:
    mod = (getattr(fn, "__module__", "") or "").lower()
    k = key.lower()
    return any(hint in mod or hint in k for hint in EXTRA_HINTS)


def extra_components() -> Dict[str, list]:
    """Inverse-free / low-complexity keys registered right now (build time)."""
    from znnbench import registry
    registry._load_all()
    core_models = {"znn", "znn_explicit_inverse", "gnn", "davidenko"}
    core_discs = {"euler", "uhlig_3_4", *PC_DISCS}
    models, discs = [], []
    for k, fn in sorted(registry.MODELS.items()):
        if k in core_models or not _is_extra(k, fn):
            continue
        try:
            req = set(getattr(registry.make_model(k), "requires", set()) or set())
        except Exception:  # noqa: BLE001  (needs mandatory kwargs: skip)
            continue
        if req <= {"matrix_inverse"}:
            models.append(k)
    for k, fn in sorted(registry.DISCRETIZATIONS.items()):
        if k in core_discs or not _is_extra(k, fn):
            continue
        try:
            d = registry.make_discretization(k)
        except Exception:  # noqa: BLE001
            continue
        paired = getattr(d, "paired_model", None) or getattr(d, "model_key", None) or "znn"
        md = getattr(d, "max_dim", None)          # dense n_x x n_x schemes: n^2 <= max_dim
        n_max = int(math.isqrt(int(md))) if md else None
        discs.append({"key": k, "model": paired, "h": _h_for_disc(d), "n_max": n_max})
    return {"models": models, "discs": discs}


def uhlig_h_opt() -> float:
    try:
        from znnbench import make_discretization
        return _h_for_disc(make_discretization("uhlig_3_4"))
    except Exception:  # noqa: BLE001
        return H_UHLIG_FALLBACK


def methods(include_extras: bool = True) -> List[dict]:
    """The E7 method table (dicts: label, model, disc, h, info, core, timing)."""
    h_uh = uhlig_h_opt()
    M = [
        dict(label="znn", model="znn", disc="euler", h=H_MAIN, info="analytic"),
        dict(label="znn_explicit_inverse", model="znn_explicit_inverse", disc="euler", h=H_MAIN,
             info="analytic"),
        dict(label="gnn", model="gnn", disc="euler", h=H_GNN, info="none"),
        dict(label="gnn_matched", model="gnn", disc="euler", h=H_MAIN, info="none", timing=False),
        dict(label="davidenko", model="davidenko", disc="euler", h=H_MAIN, info="analytic"),
        dict(label="pc_newton_causal", model="davidenko", disc="pc_newton_causal", h=H_MAIN,
             info="analytic"),
        dict(label="pc_gradient_causal", model="davidenko", disc="pc_gradient_causal", h=H_MAIN,
             info="analytic"),
        dict(label="newton_tracking_causal", model="davidenko", disc="newton_tracking_causal",
             h=H_MAIN, info="analytic"),
        dict(label="uhlig_3_4", model="znn", disc="uhlig_3_4", h=h_uh, info="analytic"),
    ]
    for m in M:
        m.setdefault("timing", True)
        m["core"] = True
    # gain-sensitivity arm (review): each gain-based core method at its tuned step.  The
    # per-step operations are identical to the protocol row, so their timing also serves as
    # an A/A control of the timing protocol (cost ratio must be ~1).
    M += [
        dict(label="znn_tuned", model="znn", disc="euler", h=H_EULER_TUNED, info="analytic",
             timing=True, core=False, tuned_of="znn"),
        dict(label="znn_explicit_inverse_tuned", model="znn_explicit_inverse", disc="euler",
             h=H_EULER_TUNED, info="analytic", timing=True, core=False,
             tuned_of="znn_explicit_inverse"),
        dict(label="gnn_tuned", model="gnn", disc="euler", h=H_GNN_TUNED, info="none", timing=True,
             core=False, tuned_of="gnn"),
    ]
    if include_extras:
        ex = extra_components()
        for k in ex["models"]:
            hyb = "gnn" in k          # GNN-ZNN hybrids: like gnn, matched h (diverges) + stable H_GNN
            inf = info_for(k)
            # a model that diverges at the first step from every start (lcc_znn with exp-sig phi)
            # yields no clean timing; it is still timed (n_diverged recorded) and drawn as
            # censored at the cost of its proxy (COST_PROXY)
            M.append(dict(label=label_of(k, "euler", H_MAIN), model=k, disc="euler", h=H_MAIN,
                          info=inf, timing=not hyb, core=False))
            if hyb:
                M.append(dict(label=label_of(k, "euler", H_GNN), model=k, disc="euler", h=H_GNN,
                              info=inf, timing=True, core=False))
        for lab, (k, kw) in MODEL_VARIANTS.items():
            if k in ex["models"]:
                M.append(dict(label=lab, model=k, model_kwargs=dict(kw), disc="euler", h=H_MAIN,
                              info=info_for(k), timing=True, core=False))
        for d in ex["discs"]:
            M.append(dict(label=label_of(d["model"], d["key"], d["h"]), model=d["model"],
                          disc=d["key"], h=d["h"], info="analytic", timing=True, core=False,
                          n_max=d.get("n_max")))
    seen, out = set(), []
    for m in M:
        m.setdefault("model_kwargs", {})
        # runner model label: the registry key for default kwargs, the E7 label for a variant
        m.setdefault("model_label", m["label"] if m["model_kwargs"] else m["model"])
        if m["label"] not in seen:
            seen.add(m["label"])
            out.append(m)
    return out


def model_entries(M: List[dict]) -> List[dict]:
    """Unique runner model entries {key, kwargs, label} of a method table."""
    out, seen = [], set()
    for m in M:
        if m["model_label"] not in seen:
            seen.add(m["model_label"])
            out.append({"key": m["model"], "kwargs": dict(m["model_kwargs"]), "label": m["model_label"]})
    return out


# ----------------------------------------------------------------------------
# FLOP models (matrix inversion, n x n, batch B)
# ----------------------------------------------------------------------------
def flops_ideal(label: str, n: int, B: int = 1) -> float:
    """Idealised minimal leading-order FLOPs per step (NaN for unknown methods)."""
    if label not in IDEAL_OPS:
        return float("nan")
    prods, lus = IDEAL_OPS[label]
    return prods * 2.0 * n ** 3 * B + lus * (2.0 / 3.0) * n ** 3


def flops_executed(calls: dict, n: int, B: int, model_key: str) -> float:
    """FLOPs per step of what the benchmark implementation executes, from
    measured call rates (per step) of the Problem API: E, dEdt, JT, solve_J
    (triangular solves) = 2 n^3 B each, LU factorisations = 2/3 n^3, plus
    model-internal products per rhs call (``MODEL_INTERNAL_PRODUCTS``;
    unknown models count 0)."""
    mm = 2.0 * n ** 3 * B
    f = sum(calls.get(k, 0.0) * mm for k in ("E", "dEdt", "JT", "solve_J"))
    f += calls.get("lu", 0.0) * (2.0 / 3.0) * n ** 3
    f += calls.get("rhs", 0.0) * MODEL_INTERNAL_PRODUCTS.get(model_key, 0) * mm
    return float(f)


# ----------------------------------------------------------------------------
# instrumentation
# ----------------------------------------------------------------------------
def prefetch_wrapper_class():
    """Return :class:`PrefetchDerivative` (import deferred so that importing
    this module does not import the zoo)."""
    from znnbench.wrappers import ProblemWrapper

    class PrefetchDerivative(ProblemWrapper):
        """Transparent wrapper: evaluates (memoises) A_dot(t_k) inside
        ``data(t_k)``, which ``simulate`` calls *before* ``disc.step``.

        ``simulate``'s ``wall_dynamics`` times ``disc.step`` only, but the
        synthetic generator of ``matrix_inversion`` evaluates A_dot(t) lazily
        on the first ``dEdt`` call, i.e. inside the step; at n = 128 that
        generator costs more than the ZNN step itself (14 dense products).
        In an application A(t), A_dot(t) are inputs, so E7 charges the
        algorithm only.  Dynamics are unchanged (memoised values are the
        same arrays)."""

        def __init__(self, problem):
            super().__init__(problem, "prefetch")
            self.name = problem.name

        def data(self, t):
            d = self.inner.data(t)
            f = getattr(self.inner, "_Adot", None)
            if callable(f):
                f(t)
            return d

    return PrefetchDerivative


class CountingModel:
    """Model proxy that counts ``rhs`` evaluations (profile runs only)."""

    def __init__(self, inner):
        self._inner = inner
        self.calls = 0

    def __getattr__(self, item):
        if item.startswith("__"):
            raise AttributeError(item)
        return getattr(self.__dict__["_inner"], item)

    def rhs(self, ctx):
        self.calls += 1
        return self._inner.rhs(ctx)

    def z0(self, problem, B):
        return self._inner.z0(problem, B)


def settle_metrics(sim, frac: float = 0.25) -> dict:
    """Runner extra metric (``"experiments.tasks.E7_common:settle_metrics"``):
    is the steady window really steady?

    The steady window W = last ``frac`` of the recorded samples is split into
    halves W1 (earlier) and W2 (later).  ``settle_ratio`` = max_W1 r / max_W2 r.
    MSSRE = max_W r is dominated by a decaying transient when the ratio is
    large (a converged tracking error fluctuates, ratio ~ 1; observed 0.7-1.4
    for converged runs).  ``mssre_w2`` = max_W2 r is reported as a
    late-window MSSRE.  Diverged trajectories give inf / NaN."""
    r = np.asarray(sim["residual"], dtype=float)
    t = np.asarray(sim["t"], dtype=float)
    n_rec = r.shape[0]
    t0, t1 = t[0], t[-1]
    w = t >= t1 - frac * (t1 - t0)
    idx = np.flatnonzero(w)
    if idx.size < 2:
        nan = np.full(r.shape[1], np.nan)
        return {"settle_ratio": nan, "mssre_w2": nan}
    half = idx[: idx.size // 2], idx[idx.size // 2:]
    with np.errstate(invalid="ignore", divide="ignore"):
        m1 = np.max(r[half[0]], axis=0)
        m2 = np.max(r[half[1]], axis=0)
        ratio = np.where(np.isfinite(m1) & np.isfinite(m2) & (m2 > 0), m1 / np.where(m2 > 0, m2, 1.0),
                         np.nan)
    del n_rec
    return {"settle_ratio": ratio, "mssre_w2": m2}


#: settle_ratio above this flags MSSRE as transient-contaminated
SETTLE_TOL = 2.0


def cluster_indices(clusters, n_boot: int, rng) -> list:
    """Two-stage (cluster, then unit within cluster) bootstrap resampling.

    ``clusters`` is a list of index arrays (one per cluster, e.g. one per
    problem seed, holding the column indices of its trajectories / timing
    rounds).  Returns ``n_boot`` integer index arrays.  The same index arrays
    are applied to every method, so the resampling is *paired*: methods are
    compared on the same problem instances, initial states and timing rounds
    (an unpaired bootstrap split ZNN-Euler and the explicit-inverse ZNN, whose
    MSSRE are identical trajectory by trajectory, in 68 % of the replicates)."""
    clusters = [np.asarray(c, dtype=int) for c in clusters if len(c)]
    out = []
    if not clusters:
        return out
    k = len(clusters)
    for _ in range(n_boot):
        pick = rng.integers(0, k, size=k)
        parts = [clusters[i][rng.integers(0, len(clusters[i]), size=len(clusters[i]))] for i in pick]
        out.append(np.concatenate(parts))
    return out


def bootstrap_ci(x, stat=np.median, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    b = np.apply_along_axis(stat, 1, x[idx]) if stat is not np.median else np.median(x[idx], axis=1)
    return (float(np.quantile(b, alpha / 2)), float(np.quantile(b, 1 - alpha / 2)))
