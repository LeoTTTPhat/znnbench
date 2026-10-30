"""Additive noise models in the design-formula space (J x_dot = ... + nu(t)).
Each factory returns callable(t, rng, shape) -> array of `shape`."""
from __future__ import annotations

import numpy as np

from ..registry import register_noise


@register_noise("constant")
def constant(level=1.0):
    return lambda t, rng, shape: np.full(shape, float(level))


@register_noise("linear")
def linear(slope=1.0, offset=0.0):
    return lambda t, rng, shape: np.full(shape, offset + slope * t)


@register_noise("gaussian")
def gaussian(std=0.1):
    return lambda t, rng, shape: std * rng.standard_normal(shape)


@register_noise("harmonic")
def harmonic(amp=1.0, omega=1.0, phase=0.0):
    return lambda t, rng, shape: np.full(shape, amp * np.sin(omega * t + phase))
