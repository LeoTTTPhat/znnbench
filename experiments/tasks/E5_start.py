"""E5 helper: near-solution initial states (``start = near`` arm of E5).

Used by experiments/configs/E5.py through the runner's import-path wrapper
mechanism::

    wrappers: [null, {key: "experiments.tasks.E5_start:NearSolutionStart",
                      target: problem, kwargs: {radius: 0.1}}]

It is an experiment-side helper, not a zoo component: nothing under
``znnbench/`` is modified.

Why (E5 protocol, experiments/E5_NOTES.md): E5 measures *steady-state*
residual floors under additive noise.  Several models whose noise claims are
under test are only locally stable in discrete time from the problems'
default random states x0 ~ U[-2, 2] (Li-activated HNTZNN for h = gamma*tau
>= 0.3, SPTC-AN-FOZNN from ||E0|| >~ 1, sign-bi-power at h = 0.3; see
docs/spec/SMOKE_MATRIX.md and docs/spec/variants.md, Observations 2-3).  A
divergence from a far initial state is a transient-stability property (E2/E3),
not a noise floor.  The ``near`` arm therefore starts every trajectory at

    x0_b = x*(0) + radius * u_b,     u_b ~ U[-1, 1]^{n_x}  (i.i.d.),

drawn from the same x0 stream that ``simulate`` uses (``simulate.x0_rng``),
so the B initial states stay seeded and reproducible.  Everything else
(data, E, J, dE/dt, x*, residual) is delegated unchanged to the wrapped
problem.
"""
from __future__ import annotations

import numpy as np

from znnbench.wrappers import ProblemWrapper


class NearSolutionStart(ProblemWrapper):
    """Delegate everything to ``problem`` except ``x0``: x*(0) + radius*U[-1,1]."""

    exact_residual = False      # the data are not altered; recorded residual = inner residual

    def __init__(self, problem, radius: float = 0.1):
        super().__init__(problem, tag=f"near{radius:g}")
        if not radius >= 0:
            raise ValueError("radius must be >= 0")
        xs = problem.x_star(0.0)
        if xs is None:
            raise TypeError("NearSolutionStart needs a problem with a known x_star(0)")
        self.radius = float(radius)
        self._xs0 = np.asarray(xs, dtype=float).reshape(-1)

    def x0(self, rng, B):
        return self._xs0[None, :] + self.radius * rng.uniform(-1.0, 1.0, size=(B, self.n_x))

    def describe(self):
        d = dict(self.inner.describe()) if hasattr(self.inner, "describe") else {}
        d.update({"name": self.name, "start": "near", "radius": self.radius})
        return d


class UnitPairStart(NearSolutionStart):
    """Same as :class:`NearSolutionStart`, under its own import path so that the
    runner's ``exclude`` rules (which match wrapper KEYS, not kwargs) can confine
    it to the paired unit check of E5 (review 2026-10-18, REVIEW_CHECKLIST item
    1): chains ``[UnitPairStart(radius=1e-4)]`` and ``[UnitPairStart(radius=1e-4),
    R_common:ScaledError(scale=1e3)]``, noise-free, models znn / znn_sbp / hntznn."""

    def describe(self):
        d = super().describe()
        d["start"] = "unit_pair"
        return d
