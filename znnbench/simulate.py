"""Fixed-step simulation of a model on a problem under a discretization."""
from __future__ import annotations

import time
from collections import deque
from typing import Callable, Optional

import numpy as np

from .core import Context, Discretizer, Model, Problem

INFO_REGIMES = ("analytic", "bd1", "bd2", "none")


def _perturb(d, sigma, rng):
    """Additive measurement noise on every data array (not on time stamps)."""
    if not sigma:
        return d
    return {k: (v if (k == "t" or k.startswith("t_")) else v + sigma * rng.standard_normal(np.shape(v)))
            for k, v in d.items()}


class RowRNG:
    """Per-trajectory random streams behind a Generator-like interface.

    A draw whose shape starts with B (the batch size) takes row b from the
    generator of trajectory id ``traj_ids[b]``; scalar draws and other shapes
    use a shared stream.  The realised noise of a trajectory therefore depends
    only on (seed, trajectory id), not on the batch composition, so a batch of
    G gains x B initial states reproduces G separate runs bit for bit.
    """

    def __init__(self, seed, traj_ids):
        self.shared = np.random.default_rng(np.random.SeedSequence([int(seed), 1]))
        self.rows = [np.random.default_rng(np.random.SeedSequence([int(seed), 2, int(i)]))
                     for i in traj_ids]

    def _draw(self, method, size, *args):
        if size is None:
            return getattr(self.shared, method)(*args)
        shape = tuple(size) if np.ndim(size) else (int(size),)
        if len(shape) == 0 or shape[0] != len(self.rows):
            return getattr(self.shared, method)(*args, size=shape)
        return np.stack([getattr(g, method)(*args, size=shape[1:]) for g in self.rows])

    def standard_normal(self, size=None):
        return self._draw("standard_normal", size)

    def normal(self, loc=0.0, scale=1.0, size=None):
        return self._draw("normal", size, loc, scale)

    def uniform(self, low=0.0, high=1.0, size=None):
        return self._draw("uniform", size, low, high)

    def random(self, size=None):
        return self._draw("random", size)

    def exponential(self, scale=1.0, size=None):
        return self._draw("exponential", size, scale)


def x0_rng(seed):
    """Stream used for initial states (shared by simulate and the runner)."""
    return np.random.default_rng(np.random.SeedSequence([int(seed), 0]))


def initial_state(problem, model, seed, B):
    """The initial states simulate draws when x0 is None (model hook first)."""
    rng = x0_rng(seed)
    x = model.x0(problem, rng, B) if callable(getattr(model, "x0", None)) else None
    return problem.x0(rng, B) if x is None else np.array(x, dtype=float).reshape(B, -1)


def simulate(problem: Problem, model: Model, disc: Discretizer, tau: float,
             T: Optional[float] = None, B: int = 1, gamma=10.0,
             x0: Optional[np.ndarray] = None, seed: int = 0,
             noise: Optional[Callable] = None, info: str = "analytic",
             data_noise: float = 0.0, record_every: int = 1,
             div_threshold: float = 1e10, record_solution_error: bool = True,
             max_wall: Optional[float] = None, stats_frac: float = 0.25,
             online_stats: bool = True, traj_ids=None):
    """Run B trajectories for T seconds with sampling period tau.

    Parameters
    ----------
    gamma : float or array (B,)   base design gain of each trajectory.
    noise : callable(t, rng, shape) -> (B, m) additive noise in the design-
            formula (error-derivative) space: J x_dot = model + noise.  If the
            callable has ``state_dependent = True`` it is called with the
            keyword ``E=`` (current error) as well.
    info  : how dE/dt is obtained -- 'analytic' (exact), 'bd1'/'bd2' (first/
            second-order backward differences of past data, causal), 'none'.
    data_noise : std of additive measurement noise on every data array the
            model sees (the recorded residual always uses exact data).
    stats_frac : fraction of the horizon (at its end) used for the online
            steady-state statistics (chattering / velocity statistics).

    Random streams: x0, noise and measurement noise use independent streams
    derived from ``seed``; additive noise uses one stream per trajectory id
    (``traj_ids``, default 0..B-1; see :class:`RowRNG`).

    Hooks honoured when present: ``model.x0(problem, rng, B)`` (model-supplied
    initial state when ``x0`` is None; may return None), ``disc.bind_env(...)``
    (gives a discretizer the problem, the current measured sample and the
    stage-data sampler), ``problem.exact_data(t)`` (exact data for recorded
    residuals when wrappers alter the data the model sees),
    ``problem.solution_error(x, t)``, and ``reset()`` on the noise / problem.

    Returns a dict with time grid, residual and solution-error traces of
    shape (N_rec, B), divergence info, wall times, initial/final states and
    online statistics (see ``metrics.summarize``).
    """
    if info not in INFO_REGIMES:
        raise ValueError(f"unknown info regime {info}")
    rng_noise = RowRNG(seed, np.arange(B) if traj_ids is None else traj_ids)
    rng_data = np.random.default_rng(np.random.SeedSequence([int(seed), 3]))
    T = problem.T if T is None else T
    N = int(round(T / tau))
    for obj in (noise, problem):
        if obj is not None and callable(getattr(obj, "reset", None)):
            obj.reset()
    if x0 is None:
        x = initial_state(problem, model, seed, B)
    else:
        x = np.array(x0, dtype=float).reshape(B, -1)
    x_init = x.copy()
    z = model.z0(problem, B)
    gam = np.broadcast_to(np.asarray(gamma, dtype=float).reshape(-1, 1), (B, 1)).copy()
    disc.reset()

    past = deque(maxlen=3)      # measured data at t_k, t_{k-1}, t_{k-2}
    stage_data = lambda t: _perturb(problem.data(t), data_noise, rng_data)  # noqa: E731
    if callable(getattr(disc, "bind_env", None)):
        disc.bind_env(problem=problem, measured=lambda: past[0], stage_data=stage_data)
    exact_data = getattr(problem, "exact_data", None)
    sol_err_fn = getattr(problem, "solution_error", None)
    state_noise = bool(getattr(noise, "state_dependent", False))

    n_rec = N // record_every + 1
    t_rec = np.empty(n_rec)
    res = np.full((n_rec, B), np.nan)
    have_star = callable(sol_err_fn) or problem.x_star(0.0) is not None
    err = np.full((n_rec, B), np.nan) if (record_solution_error and have_star) else None
    alive = np.ones(B, dtype=bool)
    diverged_at = np.full(B, np.nan)

    # online steady-state statistics
    k_ss = int(np.floor((1.0 - stats_frac) * N))
    peak_rate = np.zeros(B)
    peak_rate_ss = np.zeros(B)
    alt_cnt = np.zeros(B)
    alt_tot = 0
    dE_sum = np.zeros(B)
    E_sum = np.zeros(B)
    dE_cnt = 0
    E_cnt = 0
    prev_dx = None
    prev_E = None

    def f(t, xx, zz, k, implicit):
        """Vector field evaluated with the data available at time t."""
        if k is not None:                      # current sampling instant (causal)
            d = past[0]
            if info == "analytic":
                dEdt = problem.dEdt(xx, t)
            elif info == "bd2" and len(past) >= 3:
                dEdt = (3 * problem.E(xx, past[0]) - 4 * problem.E(xx, past[1])
                        + problem.E(xx, past[2])) / (2 * tau)
            elif info in ("bd1", "bd2") and len(past) >= 2:
                dEdt = (problem.E(xx, past[0]) - problem.E(xx, past[1])) / tau
            else:
                dEdt = np.zeros((xx.shape[0], problem.m))
        else:                                  # intermediate stage (non-causal schemes)
            d = stage_data(t)
            dEdt = problem.dEdt(xx, t) if info != "none" else np.zeros((xx.shape[0], problem.m))
        ctx = Context(problem=problem, t=t, tau=tau, k=-1 if k is None else k, x=xx, z=zz,
                      d=d, E=problem.E(xx, d), dEdt=dEdt, gamma=gam, implicit=implicit,
                      extra={"info": info})
        xdot, zdot = model.rhs(ctx)
        if noise is not None:
            shape = (xx.shape[0], problem.m)
            nu = noise(t, rng_noise, shape, E=ctx.E) if state_noise else noise(t, rng_noise, shape)
            xdot = xdot + problem.solve_J(xx, d, nu)
        return xdot, zdot

    t0 = time.perf_counter()
    wall_dyn = 0.0
    j = 0
    steps_run = 0
    for k in range(N + 1):
        t = k * tau
        d_raw = problem.data(t)
        d_exact = exact_data(t) if exact_data is not None else d_raw
        if k % record_every == 0:
            t_rec[j] = t
            r = problem.residual(x, d_exact)
            res[j] = np.where(alive, r, np.inf)
            if err is not None:
                if callable(sol_err_fn):
                    e_sol = sol_err_fn(x, t)
                else:
                    e_sol = np.linalg.norm(x - problem.x_star(t), axis=-1)
                err[j] = np.where(alive, e_sol, np.inf)
            j += 1
        if online_stats and k >= k_ss:
            Ek = problem.E(x, d_exact)
            nE = np.linalg.norm(Ek, axis=-1)
            E_sum += nE
            E_cnt += 1
            if prev_E is not None:
                dE_sum += np.linalg.norm(Ek - prev_E, axis=-1)
                dE_cnt += 1
            prev_E = Ek
        if k == N:
            break
        past.appendleft(_perturb(d_raw, data_noise, rng_data))
        ts = time.perf_counter()
        x_next, z = disc.step(k, t, tau, x, z, f)
        wall_dyn += time.perf_counter() - ts
        steps_run += 1
        bad = ~np.isfinite(x_next).all(axis=1) | (np.abs(x_next).max(axis=1) > div_threshold)
        newly = bad & alive
        if newly.any():
            diverged_at[newly] = t + tau
            alive &= ~bad
        if online_stats:
            with np.errstate(invalid="ignore", over="ignore"):
                dx = np.where(alive[:, None], x_next - x, 0.0)
                rate = np.abs(dx).max(axis=1) / tau
                peak_rate = np.maximum(peak_rate, rate)
                if k >= k_ss:
                    peak_rate_ss = np.maximum(peak_rate_ss, rate)
                    if prev_dx is not None:
                        alt_cnt += (np.sign(dx) * np.sign(prev_dx) < 0).sum(axis=1)
                        alt_tot += dx.shape[1]
                    prev_dx = dx
        x = x_next
        if not alive.all():
            x = np.where(alive[:, None], x, 0.0)
            if z.shape[1]:
                z = np.where(alive[:, None], z, 0.0)
        if not alive.any():
            res[j:] = np.inf
            if err is not None:
                err[j:] = np.inf
            t_rec[j:] = np.arange(j, n_rec) * record_every * tau
            break
        if max_wall is not None and time.perf_counter() - t0 > max_wall:
            raise TimeoutError(f"simulation exceeded {max_wall}s")
    wall = time.perf_counter() - t0

    out = {"t": t_rec, "residual": res, "sol_err": err, "diverged": ~alive,
           "diverged_at": diverged_at, "wall": wall, "wall_dynamics": wall_dyn,
           "steps": N, "steps_run": steps_run, "tau": tau, "T": T, "gamma": gam[:, 0],
           "x_init": x_init, "x_final": np.where(alive[:, None], x, np.nan)}
    if online_stats:
        nan = np.full(B, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            mE = E_sum / max(E_cnt, 1)
            mdE = dE_sum / max(dE_cnt, 1)
            out["chatter_vec"] = np.where(alive & (mE > 0) & (dE_cnt > 0), mdE / np.maximum(mE, 1e-300), nan)
            out["rate_alt"] = np.where(alive, alt_cnt / alt_tot, nan) if alt_tot else nan
            out["peak_rate"] = np.where(alive, peak_rate, np.inf)
            out["peak_rate_ss"] = np.where(alive, peak_rate_ss, np.inf)
    return out
