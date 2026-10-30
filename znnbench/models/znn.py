"""Reference models: design-formula ZNN (any activation, any gain schedule)
and the gradient neural network (GNN)."""
from __future__ import annotations

import numpy as np

from .. import activations as acts
from ..core import Context, Model


class ConstantGain:
    key = "constant"

    def __call__(self, t):
        return 1.0

    def describe(self):
        return {"key": self.key}


class DesignFormulaZNN(Model):
    """dE/dt = -gamma * g(t) * Phi(E)  =>  J x_dot = -gamma g(t) Phi(E) - dE/dt.

    With ``ctx.implicit`` the explicit rate -gamma Phi(E_k) is replaced by the
    backward-Euler (resolvent) rate -(E_k - prox_{h Phi}(E_k)) / tau with
    h = gamma g(t) tau, i.e. the error dynamics are stepped implicitly while
    the time-derivative feed-forward stays explicit (causal).

    A time-varying activation Phi(E, t) (``act.time_varying``, e.g.
    ``strict_predefined`` of SPTC-NT-ZNN) is frozen at the evaluation time,
    ``act.at(ctx.t)``; for time-invariant activations ``at`` returns the
    activation itself, so their behaviour is unchanged.  (Integration fix:
    before, such activations raised at the first step.)
    """

    family = "znn"
    supports_implicit = True

    def __init__(self, activation="linear", gain=None, name=None, **act_kw):
        self.act = acts.get(activation, **act_kw) if isinstance(activation, str) else activation
        self.gain = gain or ConstantGain()
        self.name = name or f"znn_{self.act.key}"

    def desired_Edot(self, ctx: Context):
        g = ctx.gamma * self.gain(ctx.t)
        act = self.act.at(ctx.t)
        if hasattr(ctx.implicit, "desired_Edot"):    # ErrorMapRequest (consistent.py)
            return ctx.implicit.desired_Edot(act, ctx, g)
        if ctx.implicit:
            h = g * ctx.tau
            return -(ctx.E - act.prox(ctx.E, h)) / ctx.tau
        return -g * act(ctx.E)

    def rhs(self, ctx: Context):
        r = self.desired_Edot(ctx)
        xdot = ctx.solve(r - ctx.dEdt)
        return xdot, np.zeros((ctx.x.shape[0], 0))

    def describe(self):
        return {"name": self.name, "family": self.family, "activation": self.act.describe(),
                "gain": self.gain.describe()}


class GNN(Model):
    """Gradient neural network x_dot = -gamma J^T Phi(E) (no time-derivative
    feed-forward; lags behind x*(t) with an O(1/gamma) floor)."""

    family = "gnn"
    uses_dEdt = False

    def __init__(self, activation="linear", name=None, **act_kw):
        self.act = acts.get(activation, **act_kw) if isinstance(activation, str) else activation
        self.name = name or f"gnn_{self.act.key}"

    def rhs(self, ctx: Context):
        xdot = -ctx.gamma * ctx.JT(self.act.at(ctx.t)(ctx.E))
        return xdot, np.zeros((ctx.x.shape[0], 0))

    def describe(self):
        return {"name": self.name, "family": self.family, "activation": self.act.describe()}


from ..registry import register_model  # noqa: E402


@register_model("znn")
def _znn(activation="linear", **kw):
    return DesignFormulaZNN(activation=activation, **kw)


@register_model("gnn")
def _gnn(activation="linear", **kw):
    return GNN(activation=activation, **kw)
