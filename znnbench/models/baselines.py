"""Non-ZNN baselines from numerical continuation and time-varying optimization.

The models here are written in the generic ZNN-Bench Problem API
(E, dE/dt, solve_J).  For a problem E(x, t) = 0 with Jacobian J = dE/dx:

``davidenko``   Davidenko / continuation flow   J x_dot = -dE/dt
                D. F. Davidenko, "On a new method of numerical solution of
                systems of nonlinear equations", Doklady Akad. Nauk SSSR 88
                (1953) 601-602 (in Russian; no DOI); see E. L. Allgower and
                K. Georg, *Introduction to Numerical Continuation Methods*,
                SIAM Classics 45, 2003, doi:10.1137/1.9780898719154, Ch. 2.
                In the time-varying-optimization literature the same field is
                the continuous iso-residual dynamics of Simonetto et al. (IEEE
                TSP 2016, doi:10.1109/TSP.2016.2568161, Eq. (6); its forward-
                Euler step is their prediction, Eq. (7)):
                x_dot = -[grad_xx f]^{-1} grad_tx f   (E = grad_x f).
                It equals the linear ZNN with gamma = 0, keeps the residual
                constant, E(x(t), t) = E(x(0), 0), and therefore needs a
                *corrector*.  Paired with the ``pc_newton`` discretization it
                becomes the Euler-predictor / Newton-corrector continuation
                method (= Simonetto's NTT); paired with ``pc_gradient`` it is
                GTT.

``newton_flow`` continuous Newton method on the *frozen* problem
                x_dot = -gamma J^{-1} E     (no time-derivative information)
                (M. K. Gavurin, Izv. Vyssh. Uchebn. Zaved. Mat. 1958, no. 5,
                18-31; S. Smale, J. Math. Econ. 3 (1976) 107-120,
                doi:10.1016/0304-4068(76)90019-7).  It is the linear ZNN with
                info='none'.  With forward Euler and h = gamma*tau = 1 it is
                the causal *running Newton* tracker
                x_{k+1} = x_k - J(x_k, t_k)^{-1} E(x_k, t_k) -- the "correction
                only" Newton method; for matrix inversion this is the result
                of Zhang, Ma, Cai (IEEE TCAS-I 56(7) 2009,
                doi:10.1109/TCSI.2008.2007065) that Newton iteration is the
                h = 1 special case of the discrete-time ZNN (the general-E
                statement here is the same algebra, not a quotation).  It
                lags x*(t) by O(tau).

The *time-varying Newton tracking with prediction* (NTT) and the gradient
version (GTT) are information-pattern-dependent and are implemented as
discretizations in ``discretizations/prediction_correction.py``.
"""
from __future__ import annotations

import numpy as np

from ..core import Context, Model


class Davidenko(Model):
    """Davidenko continuation flow J x_dot = -dE/dt (gamma is ignored).

    Simonetto et al. 2016, Eq. (6).  ``ctx.dEdt`` follows the information
    regime of ``simulate``: exact for 'analytic', backward differences for
    'bd1'/'bd2' (for E = grad f, 'bd1' is exactly the approximate time
    derivative of Simonetto et al. 2016, Eq. (10)),
    and zero for 'none' (then the flow is x_dot = 0).
    """

    name = "davidenko"
    family = "baseline"
    uses_dEdt = True
    supports_implicit = False

    def rhs(self, ctx: Context):
        xdot = ctx.solve(-ctx.dEdt)
        return xdot, np.zeros((ctx.x.shape[0], 0))

    def describe(self):
        return {"name": self.name, "family": self.family,
                "source": "Davidenko 1953; Allgower-Georg 2003 Ch.2; Simonetto et al. TSP 2016 Eq.(6)"}


class NewtonFlow(Model):
    """Continuous Newton method on the frozen problem: x_dot = -gamma J^{-1} E.

    Uses no time-derivative information (``uses_dEdt = False``).  Under
    forward Euler with h = gamma*tau = 1 this is the running (correction-only)
    Newton tracker x_{k+1} = x_k - J_k^{-1} E_k.
    """

    name = "newton_flow"
    family = "baseline"
    uses_dEdt = False
    supports_implicit = False

    def rhs(self, ctx: Context):
        xdot = ctx.solve(-ctx.gamma * ctx.E)
        return xdot, np.zeros((ctx.x.shape[0], 0))

    def describe(self):
        return {"name": self.name, "family": self.family,
                "source": "Gavurin 1958; Smale 1976; Zhang-Ma-Cai TCAS-I 2009 (h=1 => Newton)"}


from ..registry import register_model  # noqa: E402


@register_model("davidenko")
def _davidenko():
    return Davidenko()


@register_model("newton_flow")
def _newton_flow():
    return NewtonFlow()
