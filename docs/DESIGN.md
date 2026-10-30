# ZNN-Bench design contract (read before writing any code)

ZNN-Bench is an open benchmark of zeroing and gradient neural dynamics for time-varying problems. Every component must be **faithful to its source paper** and **unit-tested**.

## Environment
- Project root: the repository root `znn-bench/` (package `znnbench`, installed editable with `python -m pip install -e .`).
- Python: `python` of the project environment (`requirements.txt`; 3.11, numpy 2.4, scipy 1.17, pandas, pytest, pyarrow, joblib).
- Run tests: `cd znn-bench && python -m pytest tests/test_<x>.py -q`.
- **Memory rule:** the whole project must stay below 200 GB RAM. Unit tests must be small (n ≤ 16, ≤ 2e4 steps, < 60 s per test file). Never launch large sweeps yourself; sweeps run later through `tools/memguard.py pool`.

## Core API (do NOT edit `core.py`, `simulate.py`, `registry.py`, `metrics.py`)
If you need a core change, write it in `docs/spec/CORE_CHANGE_REQUESTS.md` and work around it with a wrapper.

- **Batching:** states `x` are `(B, n_x)`; errors `(B, m)`; gain `ctx.gamma` is `(B, 1)`. All B trajectories share the same problem data `d(t)`.
- **Matrix-valued unknowns** are flattened row-major: `X = x.reshape(B, n, k)`; `x = X.reshape(B, -1)`.
- `Problem` (see `core.py`): `data(t) -> dict` (must contain `'t'`), `E(x, d)`, `dEdt(x, t)` (exact partial time derivative at fixed x), `solve_J(x, d, r)` (solve `J dx = r`; min-norm least squares when J is non-square), `JT(x, d, v)`, `x_star(t)` (exact solution or None), `x0(rng, B)`, optional `capabilities()`. Set `name`, `family`, `n_x`, `m`, `T`.
- `Model.rhs(ctx) -> (x_dot, z_dot)`; auxiliary state size via `n_z(problem)`; `ctx` gives `t, tau, k, x, z, d, E, dEdt, gamma, implicit, solve(r), JT(v)`. `uses_dEdt=False` for models that ignore time-derivative information (GNN). Models that need problem structure declare `requires = {'matrix_inverse', ...}` and the problem must list it in `capabilities()`.
- `Discretizer.step(k, t, tau, x, z, f) -> (x_next, z_next)` with `f(t, x, z, k, implicit) -> (x_dot, z_dot)`. **Causal** discretizers call `f` only at `(t_k, x_k)` with the given `k`; stage evaluations at other times must pass `k=None` and set `causal=False`. The auxiliary state z is advanced with forward Euler unless the source paper says otherwise.
- Noise: `noise(t, rng, shape) -> (B, m)` is added in the design-formula space (`J x_dot = model + nu`), as in the noise-tolerant ZNN literature.
- Information regimes in `simulate`: `analytic`, `bd1`, `bd2`, `none`.
- Registration: decorate factories with `register_problem/model/discretization/noise("unique_key")` from `znnbench.registry`. Subpackages auto-import every module, so **only create new files**; never edit another agent's file.

## Reference implementations to imitate
`problems/linear_system.py`, `problems/_tv.py` (smooth time-varying matrices with controlled condition number and exact derivatives), `models/znn.py` (design-formula ZNN, GNN), `discretizations/multistep.py` (generic look-ahead formula class `LookAhead`, Euler, Taylor–Zhang, implicit-error Euler, RK4), `activations.py` (activation base class with generic resolvent `prox`), `noise/basic.py`.

## Faithfulness requirements
1. For every component, find the original paper (WebSearch/WebFetch: arXiv, IEEE Xplore abstract pages, Semantic Scholar, ResearchGate, MDPI full texts) and transcribe the **exact** equations (activation, design formula, gain law, discretization coefficients, parameter defaults).
2. Write `docs/spec/<your_area>.md`: for each component — key, full citation with DOI, the equations in LaTeX, default parameters used in the paper, any assumption you had to make (clearly flagged "ASSUMPTION"), and how the unit test checks it.
3. Docstrings cite source and equation numbers.
4. If a paper is under-specified, choose the most standard interpretation and flag it. Never invent a model that does not exist in the literature.

## Testing requirements
- `tests/test_<your_area>.py` with fast tests. For discretizations: **empirical order verification** on `linear_system` with linear ZNN at fixed `h = gamma*tau` (residual MSSRE slope ≈ expected order, see `tests/test_core.py`), zero-stability check of the characteristic polynomial, and consistency conditions. For problems: `E(x_star(t), data(t)) ≈ 0`, `dEdt` vs central finite difference, `solve_J` vs dense Jacobian built by finite differences, `JT` vs dense transpose, and a ZNN-Euler convergence smoke test. For models: continuous-time behaviour check (e.g. RK4 with tiny tau reproduces the theoretical convergence property), and discrete smoke tests.
- All existing tests must still pass: `python -m pytest -q`.

## Default experimental scales (for reference)
- Problems: n ∈ {2, 4, 8, 16, 32, 64}, condition number κ ∈ {10, 100, 1000}, frequency 1 rad/s, horizon T = 10 s (shorter allowed).
- τ ∈ {1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4}; h = γτ ∈ (0, 2).
