"""ZNN-Bench: an open, step-size-resolved benchmark of zeroing and gradient
neural dynamics for time-varying computation."""
from .core import Context, Discretizer, Model, Problem  # noqa: F401
from .registry import (available, make_discretization, make_model,  # noqa: F401
                       make_noise, make_problem)
from .simulate import simulate  # noqa: F401

__version__ = "1.0.0"
