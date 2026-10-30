"""Prediction-correction (PC) schemes of time-varying optimization as
ZNN-Bench discretizations.

Source
------
A. Simonetto, A. Mokhtari, A. Koppel, G. Leus, A. Ribeiro, "A class of
prediction-correction methods for time-varying convex optimization", IEEE
Trans. Signal Process. 64(17) (2016) 4576-4591, doi:10.1109/TSP.2016.2568161
(arXiv:1509.05196), Algorithms 1-2 (GTT, NTT); and A. Simonetto,
E. Dall'Anese, S. Paternain, G. Leus, G. B. Giannakis, "Time-varying convex
optimization: time-structured algorithms and applications", Proc. IEEE
108(11) (2020) 2032-2048, doi:10.1109/JPROC.2020.3003156.

For x*(t) = argmin f(x; t) (E = grad_x f, J = grad_xx f) with sampling period
h (= tau here) the paper's iteration k -> k+1 is

  continuous iso-residual dynamics:  x_dot = -[grad_xx f(x;t)]^{-1} grad_tx f(x;t)               (6)
  prediction (data at t_k):     x_{k+1|k} = x_k - h [grad_xx f(x_k;t_k)]^{-1} grad_tx f(x_k;t_k)   (7)
  acquire f(.; t_{k+1})
  correction (data at t_{k+1}), s = 0..C-1, x^0 = x_{k+1|k}:
     gradient (GTT):  x^{s+1} = x^s - alpha grad_x f(x^s; t_{k+1})                                 (8)
     Newton   (NTT):  x^{s+1} = x^s - [grad_xx f(x^s;t_{k+1})]^{-1} grad_x f(x^s; t_{k+1})         (9)
  x_{k+1} = x^C.
(The paper calls the number of correction steps tau and the step size gamma;
here they are C and alpha because tau, gamma already mean the sampling period
and the ZNN gain.)  With the backward difference (10) for grad_tx f the same
scheme is AGT/ANT, Eq. (11).

Mapping to ZNN-Bench
--------------------
* The *prediction* is one forward-Euler step of the paired model's vector
  field, evaluated causally at (t_k, x_k).  With the ``davidenko`` model
  (x_dot = -J^{-1} dE/dt) it is exactly Eq. (7) under info='analytic' and
  exactly Eq. (11) under info='bd1'.  With the linear ``znn`` model it is the
  Euler-ZNN step (prediction plus the -gamma J^{-1} E feedback).
* The *correction* acts on the frozen problem E(., d) = 0 and needs the
  problem itself (E, solve_J, JT), which the Discretizer interface does not
  pass.  See ``bind`` / ``_environment``.

Information pattern (the point of this module)
----------------------------------------------
``pc_newton``, ``pc_gradient`` (causal = False): faithful to Simonetto et al.:
    x_{k+1} = Corr_{d(t_{k+1})}( x_k + tau f(t_k, x_k) ).
  The correction uses data sampled at t_{k+1}, the very instant at which
  ZNN-Bench scores x_{k+1}.  Non-causal in the ZNN "future computation" sense
  (like RK4), even though the paper's algorithm is implementable when the
  correction time is not charged.
``pc_newton_causal``, ``pc_gradient_causal`` (causal = True):
    x_hat_k = Corr_{d(t_k)}(x_k),   x_{k+1} = x_hat_k + tau f(t_k, x_hat_k).
  The same operations in the same order, but the iterate scored at t_{k+1} is
  the prediction made from data up to t_k only.  The correction uses the
  measured data at t_k (the same, possibly noisy, sample the model sees).  f
  is called only at t = t_k with the current k (at the corrected state).
``newton_tracking`` (non-causal) / ``newton_tracking_causal``:
  correction-only Newton (C Newton steps, no prediction) at t_{k+1} / t_k.
``running_gradient`` (non-causal): correction-only gradient at t_{k+1} -- the
  "running gradient" (RG) baseline of Simonetto et al. 2016 (Sec. IV-B: "nothing
  less than AGT without prediction"; O(h) bound of A. Y. Popkov, Autom.
  Remote Control 66(6) (2005) 883-891, their ref. [24]).  Its causal
  version (gradient step with t_k data) is GNN + Euler with gamma = alpha/tau,
  already available as ``gnn``/``euler``.

Gradient direction (ASSUMPTION for generic problems)
---------------------------------------------------
Eq. (8) uses grad_x f = E, which is a descent direction only when E is a
gradient field (J symmetric positive definite).  Problems that declare the
capability ``'gradient_field'``, or whose family is in
``GRADIENT_FIELD_FAMILIES`` ("unconstrained-optimization", i.e.
``tv_unconstrained_opt``), get the exact Eq. (8) step x - alpha E.
For other problems (e.g. a non-symmetric linear system, KKT systems) the step
is the gradient of 1/2 ||E||^2, x - alpha J^T E (the GNN direction) -- a
ZNN-Bench extension, not part of Simonetto et al.
``direction='residual'|'normal'`` forces either form.

Step size (ASSUMPTION)
----------------------
Simonetto et al. require 0 < alpha < 2/L (L = Lipschitz constant of the
gradient).  ``alpha='auto'`` (default) estimates L once per run at the first
correction point by 30 power iterations (L = lambda_max(J) for 'residual',
sigma_max(J)^2 for 'normal') and sets alpha = alpha_scale / L with
alpha_scale = 0.5, leaving margin for the time variation of J.  A float alpha
is used as given.

Expected orders (steady-state residual vs tau)
---------------------------------------------
pc_newton: Simonetto Thm 2, O(tau^4) for C = 1 (O(tau^{4C}) bound) on
  nonlinear problems; *exact* (zero up to round-off) when E is affine in x,
  because one Newton step then solves the sampled problem.
pc_newton_causal: O(tau^2) (Euler prediction from an almost exact point).
pc_gradient, pc_gradient_causal: O(tau^2) (Simonetto Thm 1, when
  rho^C sigma < 1), with the non-causal constant smaller by about rho^C.
newton_tracking: O(tau^2) nonlinear, exact for affine E.
newton_tracking_causal, running_gradient: O(tau).  For running_gradient on
  ill-conditioned non-gradient-field problems (J^T E direction) the constant is
  ~ rho/(1 - rho) with rho = 1 - alpha_scale/kappa(J)^2, so the asymptotic
  slope 1 is only reached for small tau / long horizons (verification:
  slope 0.68 on linear_system kappa = 10, T = 4, tau in [1e-3, 3e-2]).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..core import Discretizer, Problem


def _free_vars(f) -> dict:
    code = getattr(f, "__code__", None)
    cells = getattr(f, "__closure__", None)
    if code is None or not cells:
        return {}
    env = {}
    for name, cell in zip(code.co_freevars, cells):
        try:
            env[name] = cell.cell_contents
        except ValueError:      # empty cell
            pass
    return env


def _closure_env(f, depth: int = 3) -> dict:
    """Free variables (name -> value) of the closure that holds simulate's
    ``problem``.  Wrappers that wrap the vector field in another closure
    (e.g. ``wrappers.QuantizedDiscretizer`` with ``quantize_derivative``) are
    searched recursively up to ``depth`` levels; empty dict if not found."""
    env = _free_vars(f)
    if isinstance(env.get("problem"), Problem) or depth <= 0:
        return env
    for v in env.values():
        if callable(v) and getattr(v, "__closure__", None):
            inner = _closure_env(v, depth - 1)
            if isinstance(inner.get("problem"), Problem):
                return inner
    return env


def _perturb(d, sigma, rng):
    """Same measurement-noise model as ``simulate`` uses for stage data."""
    if not sigma or rng is None:
        return d
    return {k: (v if k == "t" else v + sigma * rng.standard_normal(np.shape(v)))
            for k, v in d.items()}


#: Problem families whose error function is, by construction, the gradient of
#: a strongly convex objective (E = grad_x f, J = Hessian SPD).  Simonetto's
#: Eq. (8) correction x - alpha*E applies to them directly.
#: Verification finding: ``problems/optimization.py::TVUnconstrainedOpt`` (key
#: ``tv_unconstrained_opt``, family "unconstrained-optimization", E = grad_x f,
#: J = Hessian) originally did not declare ``'gradient_field'``, so
#: pc_gradient silently used x - alpha J^T E on the benchmark's own GTT problem
#: class.  The problems_opt area now declares the capability; the family is
#: still recognised here as a safety net.
GRADIENT_FIELD_FAMILIES = frozenset({"unconstrained-optimization"})


def is_gradient_field(problem: Problem) -> bool:
    """True if E is declared (capability ``'gradient_field'``) or known by its
    family to be the gradient of a strongly convex objective."""
    return ("gradient_field" in (problem.capabilities() or set())
            or getattr(problem, "family", None) in GRADIENT_FIELD_FAMILIES)


def jacobian_vector(problem: Problem, x: np.ndarray, d, v: np.ndarray) -> np.ndarray:
    """J(x, d) v.  Uses ``problem.Jv`` if the problem provides it, otherwise a
    central difference of E (exact up to round-off when E is affine in x)."""
    if hasattr(problem, "Jv"):
        return problem.Jv(x, d, v)
    vn = np.linalg.norm(v, axis=1, keepdims=True)
    vn = np.where(vn > 0, vn, 1.0)
    eps = 1e-6 * (1.0 + np.linalg.norm(x, axis=1, keepdims=True)) / vn
    return (problem.E(x + eps * v, d) - problem.E(x - eps * v, d)) / (2 * eps)


def estimate_lipschitz(problem: Problem, x: np.ndarray, d, direction: str,
                       iters: int = 30, seed: int = 0) -> float:
    """Power-iteration estimate of L: lambda_max(J) ('residual', J SPD) or
    sigma_max(J)^2 ('normal'), maximised over the batch.

    Trajectories whose estimate is not finite (state outside the domain of E,
    e.g. ln(x1) with x1 <= 0 in ``nonlinear_equations``, or overflow) are
    ignored; NaN is returned only when no trajectory gives a finite value."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(x.shape)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    lam = np.zeros(x.shape[0])
    with np.errstate(invalid="ignore", over="ignore"):
        for _ in range(iters):
            w = jacobian_vector(problem, x, d, v)
            if direction == "normal":
                w = problem.JT(x, d, w)
            lam = np.linalg.norm(w, axis=1)
            v = w / np.where(lam > 0, lam, 1.0)[:, None]
    lam = lam[np.isfinite(lam)]
    return float(np.max(lam)) if lam.size else float("nan")


class PredictionCorrection(Discretizer):
    """Prediction-correction discretization (Simonetto et al. 2016/2020).

    Parameters
    ----------
    correction : 'newton' | 'gradient'
    C          : number of correction steps per sampling period (paper: tau).
    causal     : False = faithful Simonetto order (correction with t_{k+1}
                 data, non-causal); True = correct with t_k data then predict.
    predict    : False gives the correction-only ("running") methods.
    alpha      : gradient step size, float or 'auto' (see module docstring).
    alpha_scale: alpha = alpha_scale / L_hat when alpha == 'auto'.
    direction  : 'auto' | 'residual' | 'normal' (gradient correction only).
    """

    implicit = False

    def __init__(self, correction: str = "newton", C: int = 1, causal: bool = False,
                 predict: bool = True, alpha="auto", alpha_scale: float = 0.5,
                 direction: str = "auto", name: Optional[str] = None,
                 expected_order: Optional[int] = None, expected_order_nonlinear: Optional[int] = None):
        if correction not in ("newton", "gradient"):
            raise ValueError("correction must be 'newton' or 'gradient'")
        if direction not in ("auto", "residual", "normal"):
            raise ValueError("direction must be 'auto', 'residual' or 'normal'")
        if int(C) < (1 if not predict else 0):
            raise ValueError("C must be >= 1 for correction-only schemes")
        self.correction, self.C, self.predict = correction, int(C), bool(predict)
        self.causal = bool(causal)
        self.alpha, self.alpha_scale, self.direction = alpha, float(alpha_scale), direction
        self.name = name or f"pc_{correction}{'_causal' if causal else ''}"
        self.expected_order = expected_order
        self.expected_order_nonlinear = expected_order_nonlinear
        self._bound: Optional[Problem] = None
        self.reset()

    # ------------------------------------------------------------------ setup
    def bind(self, problem: Problem):
        """Give the scheme direct access to the problem (preferred).  Without
        it the problem is discovered from the closure of simulate's vector
        field (workaround until the core passes it, see
        docs/spec/CORE_CHANGE_REQUESTS.md)."""
        self._bound = problem
        return self

    def bind_env(self, problem, measured=None, stage_data=None):
        """Called by ``simulate`` right after ``reset``: the problem, a callable
        returning the measured sample at t_k (``past[0]``) and the stage-data
        sampler (data at another time with simulate's measurement-noise
        model).  Preferred over closure discovery."""
        self._bound = problem
        self._measured = measured
        self._stage_data = stage_data
        return self

    def reset(self):
        self._alpha_val = None if self.alpha == "auto" else float(self.alpha)
        self._dir_val = None
        self.n_corrections = 0          # bookkeeping: total correction steps taken

    def _environment(self, f):
        env = _closure_env(f)
        problem = self._bound
        if problem is None:
            p = env.get("problem")
            problem = p if isinstance(p, Problem) else None
        if problem is None:
            raise RuntimeError(
                f"{self.name}: the correction step needs the Problem. Call disc.bind(problem) "
                "before simulate(...) (simulate's vector field did not expose it).")
        return problem, env

    def _data_now(self, problem, env, t):
        """Measured data at the current sample t_k (causal correction).

        Inside ``simulate`` this is ``past[0]``, exactly the sample that
        simulate's vector field hands the model for every call with
        ``k is not None`` (simulate appends it right before ``disc.step``).
        It is used unconditionally: an earlier version also required
        ``past[0]['t'] == t_k``, which fails for the jitter/delay wrappers of
        ``znnbench.wrappers`` (they set ``'t'`` to the actual sampling time and
        add ``'t_nominal'``) and then silently corrected with fresh, noise-free
        ``problem.data(t)`` under ``data_noise`` -- an information leak found
        in verification.  Without simulate's closure (``bind`` only) the
        scheme falls back to ``problem.data(t)``."""
        if getattr(self, "_measured", None) is not None and problem is self._bound:
            return self._measured()
        past = env.get("past") if env.get("problem") is problem else None
        if past is not None and len(past) > 0:
            return past[0]
        return problem.data(t)

    def _data_next(self, problem, env, t_next):
        """Data sampled at t_{k+1} (non-causal correction), with simulate's
        measurement-noise model when it is discoverable."""
        if getattr(self, "_stage_data", None) is not None and problem is self._bound:
            return self._stage_data(t_next)
        d = problem.data(t_next)
        if env.get("problem") is problem:
            d = _perturb(d, env.get("data_noise", 0.0), env.get("rng"))
        return d

    # ------------------------------------------------------------- correction
    def _resolve_direction(self, problem):
        if self._dir_val is None:
            if self.direction == "auto":
                self._dir_val = "residual" if is_gradient_field(problem) else "normal"
            else:
                self._dir_val = self.direction
        return self._dir_val

    def _correct(self, problem, x, d):
        if self.correction == "newton":
            for _ in range(self.C):
                x = x - problem.solve_J(x, d, problem.E(x, d))
        else:
            direction = self._resolve_direction(problem)
            if self._alpha_val is None:
                L = estimate_lipschitz(problem, x, d, direction)
                if not np.isfinite(L):
                    # every trajectory has left the domain of E or overflowed (found by the
                    # integration smoke matrix: GNN prediction steps on nonlinear_equations
                    # reach x1 <= 0, where ln(x1) is NaN).  Return NaN so that simulate
                    # flags divergence, as for any other scheme, instead of raising; alpha
                    # stays unset and is estimated at the next finite state.
                    return np.full_like(x, np.nan)
                if L <= 0:
                    raise ValueError(f"{self.name}: could not estimate the Lipschitz constant "
                                     f"(L = {L}); pass a float alpha")
                self._alpha_val = self.alpha_scale / L
            for _ in range(self.C):
                e = problem.E(x, d)
                g = e if direction == "residual" else problem.JT(x, d, e)
                x = x - self._alpha_val * g
        self.n_corrections += self.C
        return x

    # ------------------------------------------------------------------- step
    def step(self, k, t, tau, x, z, f):
        problem, env = self._environment(f)
        if self.causal:
            xc = self._correct(problem, x, self._data_now(problem, env, t)) if self.C else x
            if not self.predict:
                return xc, z
            xdot, zdot = f(t, xc, z, k, self.implicit)          # data at t_k only
            return xc + tau * xdot, z + tau * zdot
        if self.predict:
            xdot, zdot = f(t, x, z, k, self.implicit)           # prediction: data at t_k
            xp, zn = x + tau * xdot, z + tau * zdot
        else:
            xp, zn = x, z
        if self.C == 0:
            return xp, zn
        return self._correct(problem, xp, self._data_next(problem, env, t + tau)), zn

    def describe(self):
        return {**super().describe(), "correction": self.correction, "C": self.C,
                "predict": self.predict, "alpha": self.alpha, "alpha_scale": self.alpha_scale,
                "direction": self.direction, "expected_order_nonlinear": self.expected_order_nonlinear,
                "information": ("correction uses data at t_k (causal)" if self.causal else
                                "correction uses data at t_{k+1} (non-causal)"),
                "source": "Simonetto-Mokhtari-Koppel-Leus-Ribeiro IEEE TSP 2016, Alg. 1-2, Eqs. (7)-(11)"}


from ..registry import register_discretization  # noqa: E402


@register_discretization("pc_newton")
def pc_newton(C=1):
    """NTT (Alg. 2): Euler prediction (t_k) + C Newton corrections (t_{k+1})."""
    return PredictionCorrection("newton", C=C, causal=False, name="pc_newton",
                                expected_order=None, expected_order_nonlinear=4 * int(C))


@register_discretization("pc_newton_causal")
def pc_newton_causal(C=1):
    """Causal NTT: C Newton corrections with t_k data, then Euler prediction."""
    return PredictionCorrection("newton", C=C, causal=True, name="pc_newton_causal",
                                expected_order=2, expected_order_nonlinear=2)


@register_discretization("pc_gradient")
def pc_gradient(C=1, alpha="auto", alpha_scale=0.5, direction="auto"):
    """GTT (Alg. 1): Euler prediction (t_k) + C gradient corrections (t_{k+1})."""
    return PredictionCorrection("gradient", C=C, causal=False, alpha=alpha,
                                alpha_scale=alpha_scale, direction=direction, name="pc_gradient",
                                expected_order=2, expected_order_nonlinear=2)


@register_discretization("pc_gradient_causal")
def pc_gradient_causal(C=1, alpha="auto", alpha_scale=0.5, direction="auto"):
    """Causal GTT: C gradient corrections with t_k data, then Euler prediction."""
    return PredictionCorrection("gradient", C=C, causal=True, alpha=alpha,
                                alpha_scale=alpha_scale, direction=direction,
                                name="pc_gradient_causal", expected_order=2,
                                expected_order_nonlinear=2)


@register_discretization("newton_tracking")
def newton_tracking(C=1):
    """Correction-only Newton with data at t_{k+1} (non-causal); model unused."""
    return PredictionCorrection("newton", C=C, causal=False, predict=False,
                                name="newton_tracking", expected_order=None,
                                expected_order_nonlinear=2)


@register_discretization("newton_tracking_causal")
def newton_tracking_causal(C=1):
    """Correction-only Newton with data at t_k (causal running Newton); model unused."""
    return PredictionCorrection("newton", C=C, causal=True, predict=False,
                                name="newton_tracking_causal", expected_order=1,
                                expected_order_nonlinear=1)


@register_discretization("running_gradient")
def running_gradient(C=1, alpha="auto", alpha_scale=0.5, direction="auto"):
    """Correction-only gradient with data at t_{k+1} (Simonetto's RG baseline)."""
    return PredictionCorrection("gradient", C=C, causal=False, predict=False, alpha=alpha,
                                alpha_scale=alpha_scale, direction=direction,
                                name="running_gradient", expected_order=1,
                                expected_order_nonlinear=1)
