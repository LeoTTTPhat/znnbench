"""E3 unit-dependence arm: problem wrapper that expresses the error in a scaled unit.

Runner wrapper key ``experiments.tasks.E3_units:ScaledError`` (target
``problem``).  The design formula sees E_s = s E (dE_s/dt = s dE/dt,
J_s = s J, so solve_J_s(r) = solve_J(r / s)); the RECORDED residual stays in
the original unit, so t_eps, MSSRE, first-hit times and the T_c window metrics
remain directly comparable with the unwrapped runs (same eps, same physical
threshold).

Why: power-type and sign-type activations (sign_bi_power, fixed_time, vaf,
pvaf*, predefined_exp, the post-t_c branch of the strict models) have no
length scale, so their discrete behaviour (chattering floor, divergence of the
high-power tail at large |E|, ranking against linear ZNN) depends on the unit
in which E is expressed.  A model linear in E is unchanged by the wrapper
(checked in the E3 mini run: linear ZNN identical to round-off).

This is an E3-owned copy of ``experiments/tasks/R_common.py:ScaledError`` (the
robotics study's wrapper) so that E3 does not depend on a file another
workflow edits.  Nothing under ``znnbench/`` is modified.
"""
from __future__ import annotations

import numpy as np


class ScaledError:
    target = "problem"

    def __init__(self, inner, scale=1.0):
        self.inner = inner
        self.scale = float(scale)
        self.name = f"{getattr(inner, 'name', 'problem')}_E_x{self.scale:g}"

    def __getattr__(self, item):
        if item == "inner":                       # not yet set (e.g. during unpickling)
            raise AttributeError(item)
        return getattr(self.inner, item)

    def E(self, x, d):
        return self.scale * self.inner.E(x, d)

    def dEdt(self, x, t):
        return self.scale * self.inner.dEdt(x, t)

    def solve_J(self, x, d, r):
        return self.inner.solve_J(x, d, np.asarray(r) / self.scale)

    def JT(self, x, d, v):
        return self.scale * self.inner.JT(x, d, v)

    def residual(self, x, d):
        return self.inner.residual(x, d)

    def describe(self):
        base = self.inner.describe() if callable(getattr(self.inner, "describe", None)) else {}
        return {**base, "wrapper": "E3_units.ScaledError", "error_scale": self.scale}
