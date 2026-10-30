"""Canonical entity groups and orders for figures (colour follows the entity).

Each group has at most 8 members so the fixed categorical order is never
cycled.  A figure that shows a subset of a group must still use the group's
slots (``style.style_for`` after ``use_group``), so e.g. Taylor-Zhang is the
same orange in every figure.  Entities outside these groups go to small
multiples or grey "other".
"""
from __future__ import annotations

try:                       # imported as package module (experiments.keys)
    from . import style
except ImportError:        # run from experiments/ as a script directory
    import style

GROUPS = {
    # discretization formulas (orders 2..6 + references)
    "disc": ["euler", "taylor_zhang", "zead_5i", "zead_6i_guo2017", "zead_7i_cang2024",
             "uhlig_3_4", "uhlig_4_6", "rk4"],
    # extra discretizations (implicit / consistent family)
    # (euler and taylor_zhang keep slots 0 and 1 of "disc", so they have the same style in every figure)
    "disc_implicit": ["euler", "taylor_zhang", "implicit_euler_err", "peb_euler", "exact_flow_euler",
                      "semi_implicit_euler", "implicit_taylor_zhang", "peb_taylor_zhang"],
    # activations
    "act": ["linear", "power_sigmoid", "hyperbolic_sine", "sign_bi_power", "fixed_time",
            "vaf", "pvaf1", "predefined_exp"],
    # model families
    "model": ["znn", "gnn", "ieznn", "dieznn", "hntznn", "vpznn", "sptc_nt_znn", "sptc_an_foznn"],
    # tracking / time-varying optimisation baselines
    "baseline": ["znn", "gnn", "pc_newton_causal", "pc_gradient_causal", "newton_tracking_causal",
                 "davidenko", "uhlig_3_4", "znn_explicit_inverse"],
    # noise types
    "noise": ["none", "constant", "linear", "gaussian", "harmonic", "random_walk", "uniform",
              "combined"],
    # information regimes
    "info": ["analytic", "bd1", "bd2", "alpha_beta", "kalman", "savgol", "none", "bd5"],
}


def style_in_group(group: str, key: str) -> dict:
    """plot() kwargs (colour, marker, linestyle) of `key` within `group`."""
    members = GROUPS[group]
    if key not in members:
        return {"color": style.GREY, "marker": ".", "linestyle": ":"}
    i = members.index(key)
    return {"color": style.CAT[i], "marker": style.MARKERS[i], "linestyle": style.LINESTYLES[i]}
