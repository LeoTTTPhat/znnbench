"""ZeaD (Zhang et al. discretization) look-ahead formulas and analysis tools.

Every formula here is a one-step-ahead ("look-ahead") finite-difference
approximation of the first derivative at t_k,

    x_dot_k = (1/tau) * sum_{i=0}^{N-1} c_i x_{k+1-i}  +  O(tau^p),

with N instants (x_{k+1}, x_k, ..., x_{k+2-N}).  Coefficients are stored in
this "tau-form" (normalised so that the first Taylor moment equals 1) and are
transcribed from the cited papers as exact fractions.  The discretizer is
the generic causal `LookAhead` of `multistep.py`:

    x_{k+1} = ( tau * x_dot_k - sum_{i>=1} c_i x_{k+1-i} ) / c_0 ,

with x_dot_k = f(t_k, x_k) given by the continuous model (Euler start-up
until enough history exists).  For linear ZNN at fixed h = gamma*tau the
steady-state residual is O(tau^(p+1)); ``expected_order = p + 1``.

Linear-ZNN stability at fixed h
-------------------------------
On a slowly varying problem the error e_k = E(x_k, t_k) of linear ZNN obeys
(Zhang et al., IEEE TNNLS 30(3):959-966, 2019, eq. (17))

    c_0 e_{k+1} + (c_1 + h) e_k + c_2 e_{k-1} + ... + c_{N-1} e_{k+2-N} = O(tau^(p+1)),

so the discrete model is stable iff all roots of

    P_h(z) = c_0 z^{N-1} + (c_1 + h) z^{N-2} + c_2 z^{N-3} + ... + c_{N-1}

lie in the open unit disk.  `znn_stability` returns the effective step-size
interval (0, h_max) and the "optimal" h minimising the spectral radius
(the criterion of Zhang et al. 2019).  NOTE: zero-stability of the formula
(root condition at h = 0) is necessary but NOT sufficient: a formula with a
simple non-principal root on the unit circle (e.g. the symmetric Euler
formula (x_{k+1}-x_{k-1})/(2 tau)) is zero-stable yet unstable for every
h > 0.  We therefore also test the *strong* root condition (z = 1 is the
only root on the unit circle).

All step sizes in this module use the ZNN-Bench convention h = gamma * tau
with the tau-form coefficients above.  Papers that scale h differently are
converted in the docstrings (see docs/spec/discretizations.md).
"""
from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from math import factorial, gcd
from numbers import Integral, Real
from typing import Callable, Optional, Sequence

import numpy as np

from .multistep import LookAhead

Fr = Fraction


def _frac(x) -> Fraction:
    """Exact rational value of a user-supplied formula parameter.

    Integers, Fractions and strings ("-1/4", "0.1") are converted exactly.
    Floats are read as the *decimal* number they print as (-0.1 -> -1/10),
    not as their binary expansion (-0.1 -> -3602879701896397/2^55), so that
    e.g. ``zead_4i_general(a1=-0.1)`` is exactly the ZeaD 6321 formula and
    names / integer forms stay readable.  (Verifier fix: binary expansions
    overflowed `integer_form` and produced unreadable registry names.)"""
    if isinstance(x, Fraction):
        return x
    if isinstance(x, Integral):
        return Fraction(int(x))
    if isinstance(x, str):
        return Fraction(x)
    if isinstance(x, Real):
        return Fraction(str(float(x)))
    return Fraction(x)


# ---------------------------------------------------------------------------
# analysis helpers (also used by uhlig.py and adams.py)
# ---------------------------------------------------------------------------
def offsets(n: int):
    """Time offsets (in units of tau, relative to t_k) of an n-instant
    look-ahead formula: +1, 0, -1, ..., 2-n."""
    return [1 - i for i in range(n)]


def taylor_moments(coeffs: Sequence, M: int = 10):
    """mu_m = sum_i c_i o_i^m / m!  for m = 0..M (exact if coeffs are Fractions).

    Consistency of order p means mu_0 = 0, mu_1 = 1, mu_2 = ... = mu_p = 0."""
    offs = offsets(len(coeffs))
    exact = all(isinstance(c, (Fraction, int)) for c in coeffs)
    out = []
    for m in range(M + 1):
        if exact:
            s = sum(Fraction(c) * Fraction(o) ** m for c, o in zip(coeffs, offs))
            out.append(s / factorial(m))
        else:
            s = sum(float(c) * float(o) ** m for c, o in zip(coeffs, offs))
            out.append(s / factorial(m))
    return out


def truncation_order(coeffs: Sequence, tol: float = 1e-10, M: int = 14) -> int:
    """Largest p with x_dot_k = sum c_i x_{k+1-i} / tau + O(tau^p).

    Exact arithmetic for Fraction coefficients; otherwise moments are
    compared with a relative tolerance ``tol`` (scaled by the moment size).
    Returns -1 if it is not a consistent first-derivative approximation
    (mu_0 != 0 or mu_1 != 1)."""
    mu = taylor_moments(coeffs, M)
    exact = all(isinstance(c, (Fraction, int)) for c in coeffs)
    offs = offsets(len(coeffs))
    p = -1
    for m in range(M + 1):
        target = 1 if m == 1 else 0
        if exact:
            ok = mu[m] == target
        else:
            scale = sum(abs(float(c)) * abs(o) ** m for c, o in zip(coeffs, offs)) / factorial(m)
            ok = abs(float(mu[m]) - target) <= tol * max(1.0, scale)
        if not ok:
            break
        p = m
    return p if p >= 1 else -1


def char_roots(poly: Sequence) -> np.ndarray:
    """Roots of a polynomial given by its coefficients in descending powers."""
    return np.roots(np.asarray([float(c) for c in poly]))


def root_condition(poly: Sequence, tol: float = 1e-9) -> dict:
    """Dahlquist root condition of the characteristic polynomial ``poly``.

    zero_stable : all roots in the closed unit disk, roots on the circle simple
    strong      : additionally z = 1 is the only root on the unit circle
    """
    r = char_roots(poly)
    mag = np.abs(r)
    on = np.abs(mag - 1.0) <= tol
    inside = mag <= 1.0 + tol
    simple = True
    idx = np.where(on)[0]
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            if abs(r[idx[a]] - r[idx[b]]) <= 1e3 * tol:
                simple = False
    zero_stable = bool(inside.all() and simple)
    principal_only = bool(np.all(np.abs(r[on] - 1.0) <= 1e3 * tol)) if on.any() else True
    second = float(np.sort(mag)[-2]) if len(mag) > 1 else 0.0
    return {"zero_stable": zero_stable, "strong": zero_stable and principal_only,
            "roots": r, "second_largest_modulus": second,
            "has_unit_root": bool(np.any(np.abs(r - 1.0) <= 1e-6))}


def lookahead_char_poly(coeffs: Sequence):
    """Characteristic polynomial rho(z) = c_0 z^{N-1} + c_1 z^{N-2} + ... ."""
    return [float(c) for c in coeffs]


def lookahead_znn_poly(coeffs: Sequence, h: float):
    """Linear-ZNN error polynomial P_h(z) = rho(z) + h z^{N-2}."""
    q = [float(c) for c in coeffs]
    q[1] += h
    return q


def spectral_radius(poly: Sequence) -> float:
    return float(np.max(np.abs(char_roots(poly)))) if len(poly) > 1 else 0.0


def znn_stability(poly_of_h: Callable[[float], Sequence], h_hi: float = 4.0, n: int = 4000,
                  tol: float = 1e-10) -> dict:
    """Effective step-size interval (0, h_max) and optimal h of the linear-ZNN
    error recursion with characteristic polynomial ``poly_of_h(h)``.

    The radius is scanned on a grid in (0, h_hi]; the first crossing of 1 is
    refined by bisection and the minimiser of the spectral radius on
    (0, h_max) by golden-section search around the best grid point.
    ``h_max = 0`` means no stable h > 0 exists (e.g. symmetric Euler)."""
    hs = np.linspace(h_hi / n, h_hi, n)
    rad = np.array([spectral_radius(poly_of_h(h)) for h in hs])
    stable = rad < 1.0 - tol
    if not stable[0]:
        # check a much smaller h before declaring instability
        h0 = hs[0] * 1e-3
        if spectral_radius(poly_of_h(h0)) >= 1.0 - tol:
            return {"h_max": 0.0, "h_opt": None, "rho_opt": None}
        lo_ok = h0
    else:
        lo_ok = hs[0]
    bad = np.where(~stable)[0]
    bad = bad[bad > 0] if stable[0] else bad
    if len(bad) == 0:
        h_max = float(h_hi)
        i_end = n
    else:
        i_end = int(bad[0])
        lo = hs[i_end - 1] if i_end > 0 else lo_ok
        hi = hs[i_end]
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if spectral_radius(poly_of_h(mid)) < 1.0 - tol:
                lo = mid
            else:
                hi = mid
        h_max = float(0.5 * (lo + hi))
    if i_end == 0:
        j = 0
        a, b = 0.0, h_max
    else:
        j = int(np.argmin(rad[:i_end]))
        a = hs[j - 1] if j > 0 else 0.0
        b = hs[j + 1] if j + 1 < i_end else h_max
    g = (np.sqrt(5.0) - 1.0) / 2.0
    c_, d_ = b - g * (b - a), a + g * (b - a)
    fc, fd = spectral_radius(poly_of_h(c_)), spectral_radius(poly_of_h(d_))
    for _ in range(80):
        if fc < fd:
            b, d_, fd = d_, c_, fc
            c_ = b - g * (b - a)
            fc = spectral_radius(poly_of_h(c_))
        else:
            a, c_, fc = c_, d_, fd
            d_ = a + g * (b - a)
            fd = spectral_radius(poly_of_h(d_))
    h_opt = 0.5 * (a + b)
    return {"h_max": h_max, "h_opt": float(h_opt), "rho_opt": spectral_radius(poly_of_h(h_opt))}


@lru_cache(maxsize=256)
def _lookahead_stability_cached(coeffs: tuple, h_hi: float):
    return znn_stability(lambda h: lookahead_znn_poly(coeffs, h), h_hi=h_hi)


def lookahead_stability(coeffs: Sequence, h_hi: float = 4.0) -> dict:
    return dict(_lookahead_stability_cached(tuple(float(c) for c in coeffs), float(h_hi)))


def integer_form(coeffs: Sequence):
    """(integer numerators, common denominator) of exact coefficients, i.e.
    tau * x_dot_k ~ (sum_i n_i x_{k+1-i}) / D."""
    # Python integers throughout (arbitrary precision).  The previous version
    # used np.gcd, which overflowed int64 for large denominators (returning
    # garbage such as [0, 0, 0, 0] / 0) and leaked numpy ints into describe().
    fr = [Fraction(c) for c in coeffs]
    D = 1
    for c in fr:
        D = D * c.denominator // gcd(D, c.denominator)
    nums = [int(c * D) for c in fr]
    g = 0
    for v in nums:
        g = gcd(g, abs(v))
    g = gcd(g, D) if g else 1
    return [int(v // g) for v in nums], int(D // g)


# ---------------------------------------------------------------------------
# the discretizer
# ---------------------------------------------------------------------------
class ZeaD(LookAhead):
    """Causal look-ahead formula with literature metadata.

    Parameters
    ----------
    coeffs : tau-form coefficients (c_0 for x_{k+1}, c_1 for x_k, ...),
             exact `Fraction`s whenever the source prints rational numbers.
    claimed_trunc_order : truncation order p of x_dot_k stated in the source
             (x_dot_k = ... + O(tau^p)); expected_order = p + 1.
    h_range_paper : step-size interval stated in the source, converted to the
             ZNN-Bench convention h = gamma*tau (None if not stated).
    """

    def __init__(self, coeffs, name, claimed_trunc_order, source="", doi="", note="",
                 h_range_paper=None, h_opt_paper=None, family="zead", label=None):
        self.exact = all(isinstance(c, (Fraction, int)) for c in coeffs)
        self.coeffs_exact = tuple(Fraction(c) for c in coeffs) if self.exact else None
        super().__init__([float(c) for c in coeffs], name,
                         expected_order=int(claimed_trunc_order) + 1, source=source)
        self.coeffs = tuple(coeffs)
        self.instants = len(coeffs)
        self.claimed_trunc_order = int(claimed_trunc_order)
        self.doi = doi
        self.note = note
        self.h_range_paper = h_range_paper
        self.h_opt_paper = h_opt_paper
        self.family = family
        self.label = label

    # -- properties used by tests / sweeps --------------------------------
    def trunc_order(self) -> int:
        return truncation_order(self.coeffs)

    def char_poly(self):
        return lookahead_char_poly(self.coeffs)

    def root_info(self) -> dict:
        return root_condition(self.char_poly())

    def znn_poly(self, h):
        return lookahead_znn_poly(self.coeffs, h)

    def stability(self, h_hi: float = 4.0) -> dict:
        return lookahead_stability(self.coeffs, h_hi)

    def describe(self):
        st = self.stability()
        ri = self.root_info()
        d = {**super().describe(), "family": self.family, "instants": self.instants,
             "claimed_trunc_order": self.claimed_trunc_order, "doi": self.doi,
             "zero_stable": ri["zero_stable"], "strong_root_condition": ri["strong"],
             "second_root_modulus": ri["second_largest_modulus"],
             "h_max": st["h_max"], "h_opt": st["h_opt"], "rho_opt": st["rho_opt"],
             "h_range_paper": self.h_range_paper, "h_opt_paper": self.h_opt_paper}
        if self.exact:
            nums, den = integer_form(self.coeffs_exact)
            d["integer_coeffs"] = nums
            d["denominator"] = den
            d["coeffs_exact"] = [str(c) for c in self.coeffs_exact]
        if self.label:
            d["label"] = self.label
        if self.note:
            d["note"] = self.note
        return d


# ---------------------------------------------------------------------------
# coefficient sets (tau-form), transcribed from the sources
# ---------------------------------------------------------------------------
def coeffs_4i_general(a1) -> list:
    """General 4-instant ("three-step general") ZeaD formula, Hu, Kang & Zhang,
    Neurocomputing 306 (2018) 108-118, as reproduced in Sun, Tian & Wang,
    Discrete Dyn. Nat. Soc. 2019, Table 1 (ref. [16] there):

        x_dot_k = [(-a1 + 1/2) x_{k+1} + 3 a1 x_k - (3 a1 + 1/2) x_{k-1} + a1 x_{k-2}] / tau + O(tau^2),

    effective domain a1 in (-inf, 0).  a1 = -1/2: Taylor-Zhang; a1 = -1/10:
    ZeaD 6321; a1 = -1/8: FIFD; a1 = -1/4: Chen-Kang-Zhang 2023 eq. (9)."""
    a1 = _frac(a1)
    return [-a1 + Fr(1, 2), 3 * a1, -(3 * a1 + Fr(1, 2)), a1]


def coeffs_5i_general(a1) -> list:
    """General 5-instant ("four-step general") ZeaD formula, Zhang, He, Hu, Guo,
    Li & Shi, J. Comput. Appl. Math. 347 (2019) 314-329, as reproduced in Sun,
    Tian & Wang 2019, Table 1 (ref. [17] there):

        x_dot_k = [(a1 + 1/3) x_{k+1} - (4 a1 - 1/2) x_k + (6 a1 - 1) x_{k-1}
                   - (4 a1 - 1/6) x_{k-2} + a1 x_{k-3}] / tau + O(tau^3),

    effective domain a1 in (1/12, 1/6); a1 = 1/9 gives (8,1,-6,-5,2)/18."""
    a1 = _frac(a1)
    return [a1 + Fr(1, 3), -(4 * a1 - Fr(1, 2)), 6 * a1 - 1, -(4 * a1 - Fr(1, 6)), a1]


def coeffs_7i_general(p1, p2) -> list:
    """General 7-instant ZeaD formula, Yang, Zhang, Hu & Qiu, IEEE TNNLS 31(9)
    (2020) 3204-3214, Theorem 1, eq. (8):

        x_dot_k = [ -5 x_{k+1} + (214 p1 + 77 p2 + 130)/60 x_k - 5 p1 x_{k-1}
                    - 5 p2 x_{k-2} + 5(2 p1 + 4 p2 + 5)/3 x_{k-3}
                    - 5(2 p1 + 3 p2 + 6)/4 x_{k-4} + (3 p1 + 4 p2 + 10)/5 x_{k-5} ]
                  / ((2 p1 + p2 - 10) tau) + O(tau^4),

    effective domain (Theorem 2, eq. (16)): p1 - 7 p2 - 5 > 0,
    204 p1^2 + 196 p2^2 + 419 p1 p2 + 800 p1 + 525 p2 + 500 < 0, p1 < 0."""
    p1, p2 = _frac(p1), _frac(p2)
    D = 2 * p1 + p2 - 10
    return [Fr(-5) / D, (214 * p1 + 77 * p2 + 130) / (60 * D), -5 * p1 / D, -5 * p2 / D,
            5 * (2 * p1 + 4 * p2 + 5) / (3 * D), -5 * (2 * p1 + 3 * p2 + 6) / (4 * D),
            (3 * p1 + 4 * p2 + 10) / (5 * D)]


def in_7i_domain(p1, p2) -> bool:
    """Effective domain of the general 7-instant formula, TNNLS 2020 eq. (16)
    (evaluated in exact rational arithmetic)."""
    p1, p2 = _frac(p1), _frac(p2)
    return (p1 - 7 * p2 - 5 > 0) and (204 * p1 ** 2 + 196 * p2 ** 2 + 419 * p1 * p2
                                      + 800 * p1 + 525 * p2 + 500 < 0) and p1 < 0


COEFFS_5I = [Fr(8, 18), Fr(1, 18), Fr(-6, 18), Fr(-5, 18), Fr(2, 18)]
COEFFS_6I_GUO2017 = [Fr(n, 48) for n in (24, -5, -12, -6, -4, 3)]
COEFFS_6I_QIU2019 = [Fr(n, 24) for n in (13, -6, -2, -4, -3, 2)]
COEFFS_8I_WU2020 = [Fr(50, 111), Fr(-51, 2220), Fr(-20, 111), Fr(-30, 111), Fr(-10, 111),
                    Fr(35, 444), Fr(44, 555), Fr(-5, 111)]
COEFFS_8I_CHEN2019 = [Fr(100, 211), Fr(-1613, 12660), Fr(0), Fr(-90, 211), Fr(0),
                      Fr(15, 844), Fr(124, 1055), Fr(-35, 633)]
# Wu, Zhang, Guo, Li & Ming, IEEE Access 8 (2020) 228188, Table 1 (d_0 ... d_9)
_T10 = (
    "175/432 737/7560 -35/144 -245/864 -35/216 35/288 343/2160 -35/864 -95/1008 35/864",
    "350/823 9839/691320 -105/823 -280/823 -105/823 35/823 777/4115 70/2469 -935/5761 385/6584",
    "700/1677 21587/469560 -280/1677 -560/1677 -70/559 140/1677 1148/8385 70/1677 -1810/11739 245/4472",
    "100/237 947/66360 -20/237 -100/237 -10/79 40/237 88/1185 10/237 -230/1659 95/1896",
    "350/873 31063/244440 -280/873 -175/873 -140/873 35/873 952/4365 -35/873 -670/6111 35/776",
    "175/428 7051/89880 -175/856 -35/107 -105/856 35/428 819/4280 -35/642 -555/5992 35/856",
)
COEFFS_10I_WU2020 = [[Fraction(s) for s in row.split()] for row in _T10]
COEFFS_11I_CHEN2023 = [Fr(42, 101), Fr(90583, 1272600), Fr(-126, 505), Fr(-126, 505),
                       Fr(-252, 2525), Fr(0), Fr(378, 2525), Fr(28, 505), Fr(-414, 3535),
                       Fr(231, 20200), Fr(56, 4545)]


# ---------------------------------------------------------------------------
# factories
# ---------------------------------------------------------------------------
def zead_4i_general(a1=Fr(-1, 4)):
    """General 4-instant ZeaD formula (Hu, Kang, Zhang 2018), O(tau^2) => residual
    O(tau^3).  Default a1 = -1/4 (stated by the implementer to be the member
    used by Chen, Kang & Zhang, Mathematics 11 (2023) 3354, eq. (9); the
    verifier could not open that full text) -- ASSUMPTION: default choice.
    The family and its domain a1 < 0 were verified against Sun, Tian & Wang
    (optimization-online 6960 preprint of DDNS 2019), Table 1, and the domain
    coincides with the computed strong-root-condition region."""
    a1 = _frac(a1)
    if not a1 < 0:
        raise ValueError("effective domain of the general 4-instant ZeaD formula is a1 < 0")
    return ZeaD(coeffs_4i_general(a1), f"zead_4i_general(a1={a1})", 2,
                source="Hu, Kang, Zhang, Neurocomputing 306 (2018) 108-118; via Sun, Tian, Wang, "
                       "Discrete Dyn. Nat. Soc. 2019 Table 1",
                doi="10.1016/j.neucom.2018.03.053", family="zead-4i")


def zead_4i_6321():
    """ZeaD formula 6321: x_dot_k ~ (6x_{k+1} - 3x_k - 2x_{k-1} - x_{k-2})/(10 tau),
    O(tau^2).  Guo, Lin, Su, Sun, Huang, Numer. Algorithms 77 (2018) 23-36;
    stepsize domain h in (0, 0.8) proved by Zhang, Qi, Li, Qiu, Yang, Numer.
    Algorithms 81 (2019) 561-574 (reproduced here in the h = gamma*tau scale)."""
    return ZeaD(coeffs_4i_general(Fr(-1, 10)), "zead_4i_6321", 2,
                source="Guo, Lin, Su, Sun, Huang, Numer. Algorithms 77 (2018) 23-36; stepsize: Zhang, "
                       "Qi, Li, Qiu, Yang, Numer. Algorithms 81 (2019) 561-574",
                doi="10.1007/s11075-017-0302-4", h_range_paper=(0.0, 0.8), family="zead-4i")


def zead_4i_fifd():
    """FIFD formula x_dot_k ~ (5x_{k+1} - 3x_k - x_{k-1} - x_{k-2})/(8 tau), O(tau^2).
    Li, Mao, Uhlig, Zhang, J. Comput. Appl. Math. 327 (2018) 155-166, eqs. (14),(21),
    as transcribed in Uhlig, arXiv:1904.10569 (JDEA 2019), item (D)."""
    return ZeaD(coeffs_4i_general(Fr(-1, 8)), "zead_4i_fifd", 2,
                source="Li, Mao, Uhlig, Zhang, J. Comput. Appl. Math. 327 (2018) 155-166 (via Uhlig 2019, (D))",
                doi="10.1016/j.cam.2017.06.017", family="zead-4i")


def zead_5i():
    """5-instant ZeaD formula x_dot_k ~ (8x_{k+1} + x_k - 6x_{k-1} - 5x_{k-2} + 2x_{k-3})/(18 tau),
    O(tau^3) => residual O(tau^4).  Li, Mao, Uhlig, Zhang, Numer. Algorithms 81
    (2019) 609-629, eqs. (23),(27) (as transcribed in Uhlig 2019 item (E) and
    Uhlig, Numer. Math. 156 (2024), eq. (4i)); = a1 = 1/9 member of the general
    5-instant family."""
    return ZeaD(COEFFS_5I, "zead_5i", 3,
                source="Li, Mao, Uhlig, Zhang, Numer. Algorithms 81 (2019) 609-629",
                doi="10.1007/s11075-018-0564-5", family="zead-5i")


def zead_5i_general(a1=Fr(1, 9)):
    """General 5-instant ZeaD formula (Zhang, He, Hu, Guo, Li, Shi, JCAM 2019),
    O(tau^3); effective domain a1 in (1/12, 1/6).  Default a1 = 1/9 reproduces
    `zead_5i` -- ASSUMPTION: default choice."""
    a1 = _frac(a1)
    if not (Fr(1, 12) < a1 < Fr(1, 6)):
        raise ValueError("effective domain of the general 5-instant ZeaD formula is 1/12 < a1 < 1/6")
    return ZeaD(coeffs_5i_general(a1), f"zead_5i_general(a1={a1})", 3,
                source="Zhang, He, Hu, Guo, Li, Shi, J. Comput. Appl. Math. 347 (2019) 314-329; via Sun, "
                       "Tian, Wang 2019 Table 1",
                doi="10.1016/j.cam.2018.08.017", family="zead-5i")


def zead_6i_guo2017():
    """6-instant Taylor-type rule of Guo, Nie & Yan, IEEE TSMC: Systems 47(8)
    (2017) 2301-2310, Theorem 4, eq. (6):
    x_dot_k ~ (24x_{k+1} - 5x_k - 12x_{k-1} - 6x_{k-2} - 4x_{k-3} + 3x_{k-4})/(48 tau), O(tau^3).
    (Guo et al. define h = 2 mu tau; in the h = gamma*tau scale used here their
    h equals 2 h.)"""
    return ZeaD(COEFFS_6I_GUO2017, "zead_6i_guo2017", 3,
                source="Guo, Nie, Yan, IEEE Trans. Syst. Man Cybern. Syst. 47(8) (2017) 2301-2310, eq. (6)",
                doi="10.1109/TSMC.2017.2656941", family="zead-6i")


def zead_6i_qiu2019():
    """Six-instant formula of Qiu, Zhang, Guo, Yang & Li, Numer. Algorithms 81
    (2019) 1043-1065, eqs. (16),(18), labelled "6N\\tau CD" in Uhlig's list
    (arXiv:1904.10569, item (F)); the expansion of the acronym is not given there:
    x_dot_k ~ (13x_{k+1} - 6x_k - 2x_{k-1} - 4x_{k-2} - 3x_{k-3} + 2x_{k-4})/(24 tau), O(tau^3).
    Transcribed from Uhlig, arXiv:1904.10569, item (F) (primary text paywalled);
    verified symbol by symbol against that item."""
    return ZeaD(COEFFS_6I_QIU2019, "zead_6i_qiu2019", 3,
                source="Qiu, Zhang, Guo, Yang, Li, Numer. Algorithms 81 (2019) 1043-1065 (via Uhlig 2019, (F))",
                doi="10.1007/s11075-018-0581-4", family="zead-6i")


def zead_7i_general(p1=-1, p2=-1):
    """General 7-instant ZeaD formula (Yang, Zhang, Hu, Qiu, TNNLS 2020), O(tau^4)
    => residual O(tau^5).  Default (p1, p2) = (-1, -1), a point of the published
    effective domain (16) -- ASSUMPTION: the paper's Table II instances were not
    accessible; (-1, -1) gives (300, 161, -300, -300, 100, 75, -36)/780.
    CAVEAT (verifier): (-1, -1) lies close to the boundary of (16) (the quadratic
    form equals -6 there) and its second characteristic root has modulus 0.976;
    the domain contains members with second-root modulus down to ~0.83 (e.g.
    near (p1, p2) = (-1.32, -0.912)).  Robustness results for this key are
    therefore specific to the default and should not be read as properties of
    all 7-instant formulas."""
    p1, p2 = _frac(p1), _frac(p2)
    if not in_7i_domain(p1, p2):
        raise ValueError("(p1, p2) outside the effective domain (16) of the general 7-instant formula")
    return ZeaD(coeffs_7i_general(p1, p2), f"zead_7i_general(p1={p1},p2={p2})", 4,
                source="Yang, Zhang, Hu, Qiu, IEEE TNNLS 31(9) (2020) 3204-3214, eq. (8)",
                doi="10.1109/TNNLS.2019.2938866", family="zead-7i")


def zead_8i_wu2020():
    """8-instant ZeaD formula of Wu, Zhang, Guo, Li & Ming, IEEE Access 8 (2020)
    228188-228198, eq. (12), O(tau^4):
    x_dot_k ~ (1000x_{k+1} - 51x_k - 400x_{k-1} - 600x_{k-2} - 200x_{k-3}
               + 175x_{k-4} + 176x_{k-5} - 100x_{k-6})/(2220 tau).
    Wu et al. attribute it to Guo, Nie & Yan (TSMC 2017), whose paper we found
    to contain only the 6-instant rule -- attribution flagged in the spec."""
    return ZeaD(COEFFS_8I_WU2020, "zead_8i_wu2020", 4,
                source="Wu, Zhang, Guo, Li, Ming, IEEE Access 8 (2020) 228188-228198, eq. (12)",
                doi="10.1109/ACCESS.2020.3045780", family="zead-8i")


def zead_8i_chen2019():
    """8-instant ZeaD formula from the general eight-instant family of Chen &
    Zhang, IEEE Access 7 (2019) 125909-125918, with eta1 = eta3 = 0,
    eta2 = -9/10, as written in Chen, Kang & Zhang, Mathematics 11 (2023) 3354,
    eq. (7), O(tau^4):
    x_dot_k ~ (100/211 x_{k+1} - 1613/12660 x_k - 90/211 x_{k-2} + 15/844 x_{k-4}
               + 124/1055 x_{k-5} - 35/633 x_{k-6}) / tau."""
    return ZeaD(COEFFS_8I_CHEN2019, "zead_8i_chen2019", 4,
                source="Chen, Zhang, IEEE Access 7 (2019) 125909-125918 (eta1=eta3=0, eta2=-9/10), as "
                       "given in Chen, Kang, Zhang, Mathematics 11 (2023) 3354, eq. (7)",
                doi="10.1109/ACCESS.2019.2938840 (secondary: 10.3390/math11153354)", family="zead-8i")


def zead_10i_wu2020(row=1):
    """Ten-instant ZeaD formula, Wu, Zhang, Guo, Li & Ming, IEEE Access 8 (2020),
    eq. (14) with coefficients d_0..d_9 from Table 1 (rows 1-6), O(tau^5) =>
    residual O(tau^6).  Row 1 is the one used in their experiments."""
    row = int(row)
    if not 1 <= row <= 6:
        raise ValueError("row must be 1..6 (Table 1 of Wu et al. 2020)")
    return ZeaD(COEFFS_10I_WU2020[row - 1], f"zead_10i_wu2020(row={row})", 5,
                source="Wu, Zhang, Guo, Li, Ming, IEEE Access 8 (2020) 228188-228198, eq. (14), Table 1",
                doi="10.1109/ACCESS.2020.3045780", family="zead-10i")


def zead_11i_chen2023():
    """Eleven-instant ZeaD formula, Chen, Kang & Zhang, Mathematics 11 (2023)
    3354, Theorem 3, eq. (6), O(tau^5) => residual O(tau^6) (the x_{k-4}
    coefficient is zero; cf. the normalised characteristic polynomial P_10 in
    their Appendix D)."""
    return ZeaD(COEFFS_11I_CHEN2023, "zead_11i_chen2023", 5,
                source="Chen, Kang, Zhang, Mathematics 11 (2023) 3354, eq. (6)",
                doi="10.3390/math11153354", family="zead-11i")


from ..registry import register_discretization  # noqa: E402

register_discretization("zead_4i_general")(zead_4i_general)
register_discretization("zead_4i_6321")(zead_4i_6321)
register_discretization("zead_4i_fifd")(zead_4i_fifd)
register_discretization("zead_5i")(zead_5i)
register_discretization("zead_5i_general")(zead_5i_general)
register_discretization("zead_6i_guo2017")(zead_6i_guo2017)
register_discretization("zead_6i_qiu2019")(zead_6i_qiu2019)
register_discretization("zead_7i_general")(zead_7i_general)
register_discretization("zead_8i_wu2020")(zead_8i_wu2020)
register_discretization("zead_8i_chen2019")(zead_8i_chen2019)
register_discretization("zead_10i_wu2020")(zead_10i_wu2020)
register_discretization("zead_11i_chen2023")(zead_11i_chen2023)

#: keys registered by this module with their expected residual order
ZEAD_KEYS = {"zead_4i_general": 3, "zead_4i_6321": 3, "zead_4i_fifd": 3, "zead_5i": 4,
             "zead_5i_general": 4, "zead_6i_guo2017": 4, "zead_6i_qiu2019": 4,
             "zead_7i_general": 5, "zead_8i_wu2020": 5, "zead_8i_chen2019": 5,
             "zead_10i_wu2020": 6, "zead_11i_chen2023": 6}
