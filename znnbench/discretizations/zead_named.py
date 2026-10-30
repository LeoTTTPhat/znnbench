"""Named literature members of existing ZeaD families (exact fractions).

The general families live in ``zead.py`` (not edited here); this module only
fixes their parameters to the values printed in the literature and registers
the resulting formulas under their own keys.  Equations and verification:
``docs/spec/inverse_free_dtznn.md``.

``zead_7i_cang2024``
    Seven-instant Taylor difference formula of N. Cang, H. Tang, D. Guo,
    W. Zhang, W. Li, X. Li, "Discrete-time zeroing neural network with quintic
    error mode for time-dependent nonlinear equation and its application to
    robot arms", Appl. Soft Comput. 157 (2024) 111511,
    doi:10.1016/j.asoc.2024.111511, as printed in Z. Ma, S. Huang,
    Front. Neurorobot. 19 (2025) 1576473, doi:10.3389/fnbot.2025.1576473,
    Eq. (10) (the resulting "Taylor DTZNN-III" model is their Eq. (13)):

        x_dot_k = (110 x_{k+1} + 42 x_k - 89 x_{k-1} - 102 x_{k-2} + 12 x_{k-3}
                   + 44 x_{k-4} - 17 x_{k-5}) / (276 tau) + O(tau^4).

    It is the member (p1, p2) = (-89/110, -51/55) of the general seven-instant
    family of Yang, Zhang, Hu, Qiu, IEEE TNNLS 31(9) (2020), eq. (8)
    (``zead.coeffs_7i_general``); the identity is checked in exact rational
    arithmetic at import time.  (Cang et al. 2024 itself is paywalled and was
    not read; the formula is second-hand from Ma & Huang, whose Table 1
    Taylor DTZNN-III rows ZNN-Bench reproduces, see the tests.)

``zead_5i_mahuang2025``
    The "new Taylor difference formula" of Ma & Huang 2025, Theorem 1,
    Eqs. (14)/(20) (open access, read in full):

        x_dot_k = (36 x_{k+1} - x_k - 18 x_{k-1} - 27 x_{k-2} + 10 x_{k-3}) / (78 tau) + O(tau^3),

    i.e. the member a1 = 5/39 of the general five-instant family of Zhang, He,
    Hu, Guo, Li, Shi, JCAM 347 (2019) (``zead.coeffs_5i_general``).  With
    linear ZNN and the exact J^{-1} it is Ma & Huang's DTZNN Eq. (23), the
    x-recursion of their CDTZNN (``inverse_free_dtznn.cdtznn_mahuang2025``).

Step-size convention
--------------------
ZNN-Bench uses tau-form coefficients and h = gamma*tau.  Papers that write
the update as x_{k+1} = sum_j a_j x_{k-j} - J^{-1}(tau/c_0 * f_t + h_paper f)
(the form of Ma & Huang Eqs. (11)-(13), (23)) use h_paper = gamma*tau/c_0,
so h_ours = c_0 * h_paper, with c_0 = 55/138 for ``zead_7i_cang2024``
(h_paper = 138 gamma tau / 55, their Eq. (13)) and c_0 = 6/13 for
``zead_5i_mahuang2025`` (h_paper = 13 gamma tau / 6, their Eq. (22)).
``PAPER_H_FACTOR[key]`` stores c_0 for these keys and for the two existing
keys that Ma & Huang's Table 1 also uses (``zead_5i`` = Taylor DTZNN-I,
c_0 = 4/9, Eq. (11); ``zead_6i_guo2017`` = Taylor DTZNN-II, c_0 = 1/2,
Eq. (12)).
"""
from __future__ import annotations

from fractions import Fraction as Fr

from .zead import ZeaD, coeffs_5i_general, coeffs_7i_general, in_7i_domain

MAHUANG_DOI = "10.3389/fnbot.2025.1576473"
MAHUANG_SRC = ("Ma, Huang, Front. Neurorobot. 19 (2025) 1576473 (open access, PMC12289663)")

#: Cang et al. 2024 formula as printed in Ma & Huang 2025, Eq. (10): numerators / 276.
CANG2024_INTEGER = (110, 42, -89, -102, 12, 44, -17)
CANG2024_DEN = 276
CANG2024_P = (Fr(-89, 110), Fr(-51, 55))         # (p1, p2) of the TNNLS 2020 family
#: Ma & Huang 2025 Eq. (14)/(20): numerators / 78.
MAHUANG2025_INTEGER = (36, -1, -18, -27, 10)
MAHUANG2025_DEN = 78
MAHUANG2025_A1 = Fr(5, 39)                        # a1 of the JCAM 2019 five-instant family

COEFFS_7I_CANG2024 = [Fr(n, CANG2024_DEN) for n in CANG2024_INTEGER]
COEFFS_5I_MAHUANG2025 = [Fr(n, MAHUANG2025_DEN) for n in MAHUANG2025_INTEGER]

# The named members really are members of the general families (exact arithmetic).
assert coeffs_7i_general(*CANG2024_P) == COEFFS_7I_CANG2024, "Cang 2024 != 7i family member"
assert in_7i_domain(*CANG2024_P), "Cang 2024 parameters outside the TNNLS 2020 domain (16)"
assert coeffs_5i_general(MAHUANG2025_A1) == COEFFS_5I_MAHUANG2025, "Ma-Huang != 5i family member"
assert Fr(1, 12) < MAHUANG2025_A1 < Fr(1, 6)

#: h_ours = PAPER_H_FACTOR[key] * h_paper for the Table 1 models of Ma & Huang 2025.
PAPER_H_FACTOR = {"zead_5i": Fr(4, 9), "zead_6i_guo2017": Fr(1, 2),
                  "zead_7i_cang2024": Fr(55, 138), "zead_5i_mahuang2025": Fr(6, 13)}


def zead_7i_cang2024():
    """Taylor DTZNN-III formula of Cang et al., Appl. Soft Comput. 157 (2024)
    111511, as printed in Ma & Huang 2025, Eq. (10):
    x_dot_k ~ (110x_{k+1} + 42x_k - 89x_{k-1} - 102x_{k-2} + 12x_{k-3} + 44x_{k-4}
               - 17x_{k-5})/(276 tau), O(tau^4) => residual O(tau^5).
    = zead_7i_general(p1=-89/110, p2=-51/55).  Ma & Huang's step size is
    h_paper = 138 gamma tau/55 (Eq. (13)), i.e. h_ours = 55/138 h_paper."""
    return ZeaD(COEFFS_7I_CANG2024, "zead_7i_cang2024", 4,
                source="Cang, Tang, Guo, Zhang, Li, Li, Appl. Soft Comput. 157 (2024) 111511 "
                       "(formula as printed in " + MAHUANG_SRC + ", Eq. (10)); member "
                       "(p1, p2) = (-89/110, -51/55) of Yang, Zhang, Hu, Qiu, IEEE TNNLS 31(9) 2020, eq. (8)",
                doi="10.1016/j.asoc.2024.111511 (secondary: " + MAHUANG_DOI + ")",
                family="zead-7i", label="Taylor DTZNN-III (Ma & Huang 2025, Eq. (13))",
                note="h_ours = (55/138) * h_paper of Ma & Huang 2025")


def zead_5i_mahuang2025():
    """'New Taylor difference formula' of Ma & Huang, Front. Neurorobot. 19
    (2025) 1576473, Theorem 1, Eqs. (14)/(20):
    x_dot_k ~ (36x_{k+1} - x_k - 18x_{k-1} - 27x_{k-2} + 10x_{k-3})/(78 tau),
    O(tau^3) => residual O(tau^4).  = zead_5i_general(a1=5/39).  With linear
    ZNN it is their DTZNN Eq. (23) with h_paper = 13 gamma tau/6, i.e.
    h_ours = 6/13 h_paper."""
    return ZeaD(COEFFS_5I_MAHUANG2025, "zead_5i_mahuang2025", 3,
                source=MAHUANG_SRC + ", Theorem 1, Eqs. (14), (20), (23); member a1 = 5/39 of "
                       "Zhang, He, Hu, Guo, Li, Shi, J. Comput. Appl. Math. 347 (2019) 314-329",
                doi=MAHUANG_DOI, family="zead-5i",
                label="Taylor difference formula of Ma & Huang 2025 (DTZNN Eq. (23))",
                note="h_ours = (6/13) * h_paper of Ma & Huang 2025")


from ..registry import register_discretization  # noqa: E402

register_discretization("zead_7i_cang2024")(zead_7i_cang2024)
register_discretization("zead_5i_mahuang2025")(zead_5i_mahuang2025)

#: keys registered by this module with their expected residual order
ZEAD_NAMED_KEYS = {"zead_7i_cang2024": 5, "zead_5i_mahuang2025": 4}
