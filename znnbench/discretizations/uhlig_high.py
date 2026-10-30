"""Uhlig's convergent look-ahead formulas of ZNN truncation orders 7 and 8.

STATUS (2026-10-18): NOT OBTAINED.  This module registers NOTHING.
==================================================================

Uhlig reports convergent look-ahead finite-difference formulas of ZNN
truncation orders 7 and 8, but their coefficients (or the seeds that generate
them) were published only in his MATLAB files.  None of those files could be
obtained from the live Auburn server, the Wayback Machine, archive.today,
Arquivo.pt, Common Crawl, GitHub or the publishers (every URL tried is listed
in ``SOURCES_TRIED`` and in docs/spec/uhlig_high.md).  As the project rules
require, no formula was regenerated with our own seed search: a polynomial
found that way would be *a* convergent formula of the same type, not Uhlig's.

Sources that name the formulas
------------------------------
[U19]  F. Uhlig, J. Difference Equ. Appl. 25 (2019) 930-941,
       doi:10.1080/10236198.2019.1627343, arXiv:1904.10569.  Abstract: new
       convergent formulas of truncation orders 5 through 8; p.8: the order
       5..8 formulas are at the bottom of the list inside runconv1step.m and
       can be regenerated from the seeds given there; names 3_3, 4_5, 5_5, 5_6.
[UF]   F. Uhlig, Linear Multilinear Algebra 68 (2020) 1894-1910,
       doi:10.1080/03081087.2019.1648375, arXiv:1904.10568.  Table 3 tests
       "5_5" and "5_6b" (order 7) on the field-of-values problem; Figure 2
       shows only their root plots.  Code: FOVZNN5_6bshorteigsym.m [UF, ref 20].
[U24]  F. Uhlig, Numer. Math. 156 (2024) 691-739,
       doi:10.1007/s00211-023-01393-5, arXiv:2008.02724v6.  Sec. 1: formulas
       of "all types j_s with 1 <= j <= 6 and j <= s <= j + 3" with orders 3..8;
       ref [46]: "List of look-ahead convergent finite difference formulas at
       .../~uhligfd/m_files/ZNNSurveyExamples under Polyksrestcoeff3.m".

Where the coefficients were (archived directory listings prove the files
existed; no archive captured the files themselves):
  m_files/ZNNSurveyExamples/Polyksrestcoeff3.m   4.8K  04-Aug-2020
  m_files/AZNN/Polyksrestcoeff3.m                7.5K  02-Oct-2020
  m_files/Eigencurves/Polyjsrestcoeff2.m         4.3K  08-Sep-2019
  m_files/ZNNpolynoms/runconv1step1Plot.m        8.3K  08-Sep-2019
  m_files/FOVviaZNN/runconv1step1Plot.m          8.1K  23-Dec-2018
  m_files/FOVviaZNN/FOVZNN5_6beigshort2.m        7.8K  03-Jan-2019
  m_files/FOVviaZNN/FOVZNN5_6beigshortsym.m      7.7K  18-Jan-2019

What this module provides
-------------------------
* ``PENDING``: metadata of the order-7/8 formulas named in the sources (label,
  convention, instants, ZNN order, the key they would get, published
  behaviour usable as a cross-check once coefficients are found).
* ``SOURCES_TRIED``: every URL queried, with the outcome.
* ``validate_uhlig_polynomial``: the checks a transcription must pass before
  it may be registered (degree, consistency/moment conditions of order k+1,
  strong root condition, a non-empty linear-ZNN step interval).
* ``formula_from_polynomial``: builds an UNREGISTERED ``ZeaD`` from a
  *transcribed* polynomial after validation.  It is the drop-in path for the
  integrator once Polyksrestcoeff3.m (or another primary copy) is found;
  registration must then be added explicitly, with the source, in this file.

Naming (same convention as uhlig.py): [U19]/[UF] label "k_s" = seed length s,
characteristic polynomial of degree k+s (k+s+1 instants), ZNN order k+2;
[U24] label "j_s" = j+s instants, order j+2.  Hence [UF] "k_s" -> key
``uhlig_<k>_<s+1>`` (2_2 -> uhlig_2_3, 3_3 -> uhlig_3_4, 4_5a -> uhlig_4_6,
5_5 -> uhlig_5_6, 5_6b -> uhlig_5_7).  ASSUMPTION (as in uhlig.py): the [UF]
labels use the degree convention, which [UF] p.6 confirms for 4_5a ("needs
4 + 5 = 9 ... starting values", p9 of degree 9).
"""
from __future__ import annotations

from fractions import Fraction
from typing import Optional, Sequence

from .uhlig import tau_form
from .zead import ZeaD, lookahead_stability, root_condition, truncation_order

# ---------------------------------------------------------------------------
# formulas named in the sources whose coefficients were NOT obtained
# ---------------------------------------------------------------------------
PENDING = {
    "uhlig_5_6": {
        "label": "5_5 [U19, UF]",
        "k": 5, "s": 5, "degree": 10, "instants": 11, "znn_order": 7,
        "named_in": "[U19] p.8 ('5_5'); [UF] Table 3 and Figure 2 ('5_5')",
        "coefficients_in": "Polyksrestcoeff3.m; runconv1step1Plot.m (seed list) [U19 ref 7, UF ref 19]",
        "published_behaviour": {
            "uf_table3_h": 0.0069,
            "uf_table3_note": "FoV problem, n = 243: 11.5 digits at tau = 1e-3 (eta = 6.9), "
                              "14.7 digits at tau = 1e-4 (eta = 69)",
            "uf_fig2_roots": "five roots inside the unit disk lie much closer to the unit circle "
                             "than the four closest inside roots of 5_6b",
        },
    },
    "uhlig_5_7": {
        "label": "5_6b [UF] (= '5_6' in [U19] p.8; identification is an inference)",
        "k": 5, "s": 6, "degree": 11, "instants": 12, "znn_order": 7,
        "named_in": "[U19] p.8 ('5_6'); [UF] Table 3 and Figure 2 ('5_6b')",
        "coefficients_in": "FOVZNN5_6beigshort2.m / FOVZNN5_6beigshortsym.m "
                           "(cited in [UF] as FOVZNN5_6bshorteigsym.m); Polyksrestcoeff3.m",
        "published_behaviour": {
            "uf_table3_h": 0.016,
            "uf_table3_note": "FoV problem, n = 243: 13.7 digits at tau = 1e-3 (eta = 16), "
                              "15.2 digits at tau = 1e-4 (eta = 160)",
            "uf_fig2_roots": "roots more widely and evenly spread over the unit disk than those of 5_5",
        },
    },
    "uhlig_6_*": {
        "label": "order-8 types (k = 6); no individual polynomial is named in any source read",
        "k": 6, "s": None, "degree": None, "instants": None, "znn_order": 8,
        "named_in": "[U19] p.7 ('k + 2 = 6 + 2 = 8 for k = 6'); [U24] Sec. 1 "
                    "('1 <= j <= 6 and j <= s <= j + 3', orders up to O(tau^8))",
        "coefficients_in": "Polyksrestcoeff3.m; runconv1step1Plot.m (seed list)",
        "published_behaviour": {},
    },
}

#: Every source queried for the coefficients (URL, outcome).  The same list,
#: with dates and details, is in docs/spec/uhlig_high.md.
SOURCES_TRIED = (
    # live server
    ("https://www.auburn.edu/~uhligfd/m_files/", "302 -> webhome.auburn.edu, 404"),
    ("https://webhome.auburn.edu/~uhligfd/m_files/", "404"),
    ("https://webhome.auburn.edu/~uhligfd/m_files/ZNNpolynoms/", "404"),
    ("https://webhome.auburn.edu/~uhligfd/m_files/ZNNSurveyExamples/", "404"),
    ("https://webhome.auburn.edu/~uhligfd/", "200, 2004-era homepage without m_files links"),
    # Wayback Machine
    ("http://web.archive.org/cdx/search/cdx?url=auburn.edu/~uhligfd/*&output=json",
     "1062 captures; every m_files URL is a 301/302 redirect, no .m content"),
    ("http://web.archive.org/cdx/search/cdx?url=webhome.auburn.edu/~uhligfd/*&output=json",
     "173 captures; directory listings yes, ZNN .m files no"),
    ("http://archive.org/wayback/available?url=<host>/~uhligfd/m_files/<dir>/<file>.m",
     "no snapshots for 11 files x 2 hosts"),
    ("http://web.archive.org/web/2024id_/http://webhome.auburn.edu/~uhligfd/m_files/AZNN/Polyksrestcoeff3.m",
     "404"),
    ("http://web.archive.org/web/2024id_/http://webhome.auburn.edu/~uhligfd/m_files/ZNNSurveyExamples/Polyksrestcoeff3.m",
     "404"),
    ("http://web.archive.org/web/2024id_/http://webhome.auburn.edu/~uhligfd/m_files/FOVviaZNN/FOVZNN5_6beigshort2.m",
     "404"),
    # other archives
    ("https://archive.ph/timemap/<url of Polyksrestcoeff3.m / runconv1step1Plot.m>", "no mementos"),
    ("https://arquivo.pt/wayback/cdx?url=<url of Polyksrestcoeff3.m / runconv1step1Plot.m>", "no captures"),
    ("http://timetravel.mementoweb.org/api/json/2021/<url>", "service unreachable (DNS)"),
    ("https://index.commoncrawl.org/CC-MAIN-<2018-30 ... 2025-13>-index?url=<host>/~uhligfd/m_files/*",
     "no captures in 60 indexes x 2 hosts"),
    # papers (full text and e-print source)
    ("https://arxiv.org/pdf/1904.10569v1 (+ /e-print)", "orders <= 5 only (2_2, 3_3 seeds)"),
    ("https://arxiv.org/pdf/1904.10568v1 (+ /e-print)", "p9 of 4_5a only; 5_5/5_6b as root plots"),
    ("https://arxiv.org/pdf/1904.10566v1 (+ /e-print)", "3_3b = 3_3 (a = 1) only"),
    ("https://arxiv.org/pdf/2008.02724v1 ... v6 (+ /e-print v1, v6)", "points to Polyksrestcoeff3.m"),
    ("https://arxiv.org/pdf/2209.10002v1, v2 (+ /e-print)", "4_5 only; points to Polyksrestcoeff3.m"),
    ("https://arxiv.org/pdf/2002.01274v1, 2006.01241v5, 2006.01215v2 (+ /e-print)", "no coefficients"),
    ("https://link.springer.com/content/pdf/10.1007/s00211-023-01393-5.pdf",
     "open access; no supplementary material; points to Polyksrestcoeff3.m"),
    ("https://link.springer.com/article/10.1007/s00211-023-01393-5", "bot challenge (not bypassed)"),
    ("https://journals.uwyo.edu/index.php/ela/article/download/7417/6133", "ELA 2023: points to Polyksrestcoeff3.m"),
    ("https://www.tandfonline.com/doi/suppl/10.1080/10236198.2019.1627343", "403 bot challenge (not bypassed)"),
    ("https://www.tandfonline.com/doi/suppl/10.1080/03081087.2019.1648375", "403 bot challenge (not bypassed)"),
    ("https://api.crossref.org/works/<DOI of U19, UF, U24>", "no supplement relation"),
    # mirrors
    ("https://api.github.com/search/code?q=<Polyksrestcoeff, runconv1step1Plot, FOVZNN5_6bshorteigsym, "
     "1.632891580619644>", "0 hits"),
)


# ---------------------------------------------------------------------------
# validation / drop-in constructor (nothing is registered)
# ---------------------------------------------------------------------------
def u24_key(k: int, s: int) -> str:
    """Registry key of a [U19]/[UF] "k_s" polynomial (degree k+s) in the
    [U24] j_s convention (j + s' = k + s + 1 instants): ``uhlig_<k>_<s+1>``."""
    return f"uhlig_{int(k)}_{int(s) + 1}"


def _exact(poly) -> bool:
    return all(isinstance(c, (Fraction, int)) for c in poly)


def validate_uhlig_polynomial(poly: Sequence, k: int, s: int, tol: float = 1e-10) -> dict:
    """Check a transcribed Uhlig polynomial of type "k_s" ([U19]/[UF] labels).

    ``poly``: characteristic-polynomial coefficients in descending powers,
    any scaling (Uhlig prints both normalised and non-normalised forms),
    length k+s+1.  Exact for int/Fraction input; floats (Uhlig's 16-digit
    printouts) are compared with relative tolerance ``tol``.

    Checks ([U19] Sec. 2-3): (i) degree k+s; (ii) the tau-form is a consistent
    first-derivative formula of truncation order >= k+1 (ZNN order k+2);
    (iii) zero-stable with z = 1 the only unit-circle root (strong root
    condition, needed for any h > 0, cf. zead.py); (iv) non-empty linear-ZNN
    step interval (0, h_max).  Returns a report with ``ok`` and ``reasons``."""
    k, s = int(k), int(s)
    reasons = []
    poly = list(poly)
    if len(poly) != k + s + 1:
        return {"ok": False, "reasons": [f"length {len(poly)} != k+s+1 = {k + s + 1}"]}
    # ints -> Fractions so that the tau-form stays exact (tau_form divides with /)
    poly = [Fraction(c) for c in poly] if _exact(poly) else [float(c) for c in poly]
    if poly[0] == 0:
        return {"ok": False, "reasons": ["zero leading coefficient"]}
    c = sum(pi * (1 - i) for i, pi in enumerate(poly))
    if c == 0:
        return {"ok": False, "reasons": ["first Taylor moment is zero: not a derivative formula"]}
    tf = tau_form(poly)
    p = truncation_order(tf, tol=tol, M=k + 4)
    if p < k + 1:
        reasons.append(f"truncation order {p} < k+1 = {k + 1}")
    ri = root_condition([float(x) for x in tf])
    if not ri["zero_stable"]:
        reasons.append("not zero-stable (root outside the unit disk or repeated unit root)")
    elif not ri["strong"]:
        reasons.append("non-principal root on the unit circle (unstable for every h > 0)")
    st = lookahead_stability([float(x) for x in tf]) if ri["zero_stable"] else \
        {"h_max": 0.0, "h_opt": None, "rho_opt": None}
    if not st["h_max"] > 0:
        reasons.append("empty linear-ZNN step interval")
    return {"ok": not reasons, "reasons": reasons, "instants": k + s + 1, "trunc_order": p,
            "znn_order": p + 1 if p >= 1 else None, "zero_stable": ri["zero_stable"],
            "strong": ri["strong"], "second_root_modulus": ri["second_largest_modulus"],
            "h_max": st["h_max"], "h_opt": st["h_opt"], "rho_opt": st["rho_opt"],
            "tau_form": tf}


def formula_from_polynomial(poly: Sequence, k: int, s: int, *, label: str, source: str, doi: str,
                            name: Optional[str] = None, tol: float = 1e-10) -> ZeaD:
    """Unregistered causal ``ZeaD`` look-ahead discretizer from a TRANSCRIBED
    Uhlig polynomial of type "k_s" (degree k+s, ZNN order k+2), after
    ``validate_uhlig_polynomial`` passes; raises ``ValueError`` otherwise.

    For use once a primary copy of the coefficients is found.  Do not feed it
    polynomials from a new seed search (project rule: never invent formulas)."""
    if not (label and source and doi):
        raise ValueError("label, source and doi are required (faithfulness rule)")
    rep = validate_uhlig_polynomial(poly, k, s, tol=tol)
    if not rep["ok"]:
        raise ValueError(f"polynomial rejected: {rep['reasons']}")
    return ZeaD(rep["tau_form"], name or u24_key(k, s), int(k) + 1, source=source, doi=doi,
                family="uhlig", label=label)


# Registered keys of this module: none (see STATUS above).
UHLIG_HIGH_KEYS: dict = {}
