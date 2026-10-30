"""Uhlig's convergent look-ahead finite difference formulas.

Sources
-------
[U19]  F. Uhlig, "The construction of high order convergent look-ahead finite
       difference formulas for Zhang neural networks", J. Difference Equ. Appl.
       25 (2019) 930-941, doi:10.1080/10236198.2019.1627343, arXiv:1904.10569.
[UF]   F. Uhlig, "Zhang neural networks for fast and accurate computations of
       the field of values", Linear Multilinear Algebra 68 (2020) 1894-1910,
       doi:10.1080/03081087.2019.1648375, arXiv:1904.10568.
[U24]  F. Uhlig, "Zhang neural networks: an introduction to predictive
       computations for discretized time-varying matrix problems", Numer. Math.
       156 (2024) 691-739, doi:10.1007/s00211-023-01393-5, arXiv:2008.02724v6.

Construction ([U19], Sec. 2, eqs. (5)-(15))
------------------------------------------
For a seed y in R^s, form the (k+s) x k Taylor matrix A with rows for
x_{j+1}, x_{j-1}, ..., x_{j-(k+s-1)} and columns for the derivatives of order
2..k+1,  a_{1,v} = 1/(v+1)!,  a_{u,v} = (-1)^{v+1} (u-1)^{v+1}/(v+1)!  (u >= 2).
With R = rref(A^T) = [I_k, B], the left null vector is q = [-B y; y] / q_1 and
the normalised characteristic polynomial is p = [1, -sum(q), q_2, ..., q_{k+s}]
(degree k+s, i.e. k+s+1 instants); the look-ahead rule is
x_{j+1} = -(p_2 x_j + ... ) + c tau x_dot_j with c = sum_i p_i (1 - i)
(0-based i), i.e. tau-form coefficients p / c.  The ZNN local truncation error
is O(tau^(k+2)) (derivative O(tau^(k+1))).  Convergence additionally requires
all roots of p in the unit disk with simple roots on the circle ([U19]) -- found
by Uhlig with a Nelder-Mead search over seeds; here we only use seeds/
polynomials that Uhlig published.

Naming
------
[U19]/[UF] label a polynomial "k_s" (seed length s, degree k+s); [U24]
relabels the same objects "j_s" with j + s = number of instants (the 5-IFD
formula is "2_2" in [U19]/[UF] and "2_3" in [U24]).  Registry keys follow the
[U24] j_s convention requested for ZNN-Bench; the original label is kept in
``describe()['label']``.  ASSUMPTION: [UF]'s "4_5a" (degree 9, 10 instants,
order 6) is therefore keyed "uhlig_4_6" (j = 4, j + s = 10).

Uhlig reports convergent formulas of ZNN truncation orders up to 8 ([U19],
[U24]: "types j_s with 1 <= j <= 6"), but the coefficient lists for orders
7 and 8 live only in his MATLAB files (runconv1step.m, Polyksrestcoeff3.m at
auburn.edu), which are no longer reachable.  Only the formulas whose
coefficients are printed in the papers are registered here.
"""
from __future__ import annotations

from fractions import Fraction
from math import factorial
from typing import Sequence

from .zead import ZeaD, _frac


def _rref(M):
    M = [list(row) for row in M]
    rows, cols = len(M), len(M[0])
    r, piv = 0, []
    for c in range(cols):
        pr = next((i for i in range(r, rows) if M[i][c] != 0), None)
        if pr is None:
            continue
        M[r], M[pr] = M[pr], M[r]
        pv = M[r][c]
        M[r] = [x / pv for x in M[r]]
        for i in range(rows):
            if i != r and M[i][c] != 0:
                fct = M[i][c]
                M[i] = [a - fct * b for a, b in zip(M[i], M[r])]
        piv.append(c)
        r += 1
        if r == rows:
            break
    return M, piv


def taylor_matrix(k: int, s: int):
    """Uhlig's rational (k+s) x k matrix A_{k+s,k} ([U19] eq. (11))."""
    A = []
    for u in range(1, k + s + 1):
        row = []
        for v in range(1, k + 1):
            if u == 1:
                row.append(Fraction(1, factorial(v + 1)))
            else:
                row.append(Fraction((-1) ** (v + 1) * (u - 1) ** (v + 1), factorial(v + 1)))
        A.append(row)
    return A


def uhlig_seed_polynomial(k: int, s: int, y: Sequence):
    """Normalised characteristic polynomial p (leading 1, length k+s+1) and the
    tau coefficient c of the look-ahead rule spawned by seed y ([U19] Sec. 2).
    Exact rational arithmetic."""
    if len(y) != s:
        raise ValueError("seed must have length s")
    A = taylor_matrix(k, s)
    AT = [list(col) for col in zip(*A)]
    R, piv = _rref(AT)
    if piv != list(range(k)):
        raise ValueError("A^T does not have the regular rref form [I_k, B]")
    B = [row[k:] for row in R]
    yy = [_frac(v) for v in y]
    q = [-sum(b * v for b, v in zip(Brow, yy)) for Brow in B] + yy
    if q[0] == 0:
        raise ValueError("seed gives a zero look-ahead coefficient")
    q = [v / q[0] for v in q]
    p = [Fraction(1), -sum(q)] + q[1:]
    c = sum(pi * (1 - i) for i, pi in enumerate(p))
    return p, c


def tau_form(p: Sequence):
    """tau-form coefficients p / c, c = first Taylor moment of p."""
    c = sum(pi * (1 - i) for i, pi in enumerate(p))
    return [pi / c for pi in p]


#: [UF] p.6: non-normalised characteristic polynomial p9 of method "4_5a"
#: (printed to 16 digits; non-rational).
P9_4_5A = (-1.632891580619644, -1.084874852377588, 1.514338299609167, 2.121238162639099,
           -0.3010929138446914, -0.9393487657815317, 0.06714730122560907,
           0.3319027505915695, -0.04244088319409350, -0.03397751824789656)


def uhlig_2_3():
    """5-IFD formula, Uhlig label "2_2" in [U19]/[UF], "2_3" in [U24]:
    p(x) = 8x^4 + x^3 - 6x^2 - 5x + 2, i.e.
    18 tau x_dot_k ~ 8x_{k+1} + x_k - 6x_{k-1} - 5x_{k-2} + 2x_{k-3}  (order 4).
    Generated from Uhlig's seed y = [-5, 2] (k = 2, s = 2) exactly as in
    [U19] p.7 (TPOLY = 1 .125 -.75 -.625 .25, tau coefficient 2.25).
    Identical to `zead_5i` (Li, Mao, Uhlig, Zhang 2019)."""
    p, _ = uhlig_seed_polynomial(2, 2, [-5, 2])
    return ZeaD(tau_form(p), "uhlig_2_3", 3, label="2_2 [U19,UF] = 2_3 [U24]",
                source="Uhlig, JDEA 25 (2019) 930-941, item (E) and runconv1step(1,1,2,2); "
                       "Uhlig, Numer. Math. 156 (2024), eq. (4i)",
                doi="10.1080/10236198.2019.1627343", family="uhlig")


def uhlig_3_4(a=1):
    """Order-5 formula from Uhlig's seed y = [a, 110, -40], 1 <= a <= 2.5
    ([U19] p.7, runconv1step(1,1,3,3)); label "3_3" in [U19]/[UF], "3_4" in
    [U24] (7 instants).  For a = 1 (Uhlig's printed run):
    p = [1, 80/237, -182/237, -206/237, 1/237, 110/237, -40/237], c = 196/79,
    i.e. 588 tau x_dot_k ~ 237x_{k+1} + 80x_k - 182x_{k-1} - 206x_{k-2} + x_{k-3}
    + 110x_{k-4} - 40x_{k-5}.  Derivative truncation O(tau^4), ZNN order 5."""
    a = _frac(a)
    if not (1 <= a <= Fraction(5, 2)):
        raise ValueError("Uhlig states the seed family [a, 110, -40] for 1 <= a <= 2.5")
    p, _ = uhlig_seed_polynomial(3, 3, [a, 110, -40])
    return ZeaD(tau_form(p), f"uhlig_3_4(a={a})", 4, label="3_3 [U19,UF] = 3_4 [U24]",
                source="Uhlig, JDEA 25 (2019) 930-941, p.7 (seed y=[a,110,-40])",
                doi="10.1080/10236198.2019.1627343", family="uhlig")


def uhlig_4_6():
    """Order-6 formula "4_5a" of [UF] (eq. for p9, p.6): degree-9 non-rational
    polynomial printed to 16 digits, 10 instants; [U24] j_s label 4_6
    (ASSUMPTION, see module docstring).  Derivative truncation O(tau^5),
    ZNN order 6; moments are satisfied to ~1e-15 (float transcription)."""
    p = [c / P9_4_5A[0] for c in P9_4_5A]
    return ZeaD(tau_form(p), "uhlig_4_6", 5, label="4_5a [UF] = 4_6 [U24]",
                source="Uhlig, Linear Multilinear Algebra 68 (2020) 1894-1910 (arXiv:1904.10568), p9",
                doi="10.1080/03081087.2019.1648375", family="uhlig")


from ..registry import register_discretization  # noqa: E402

register_discretization("uhlig_2_3")(uhlig_2_3)
register_discretization("uhlig_3_4")(uhlig_3_4)
register_discretization("uhlig_4_6")(uhlig_4_6)

UHLIG_KEYS = {"uhlig_2_3": 4, "uhlig_3_4": 5, "uhlig_4_6": 6}
