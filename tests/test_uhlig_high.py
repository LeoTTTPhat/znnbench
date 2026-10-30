"""Tests for znnbench/discretizations/uhlig_high.py (Uhlig's order-7/8 formulas).

The coefficients of Uhlig's order-7/8 formulas could not be obtained from any
primary or archived source (docs/spec/uhlig_high.md), so the module registers
nothing.  These tests check that (1) no unsourced order-7/8 key is
registered, (2) the pending metadata is internally consistent with Uhlig's
naming conventions, (3) the validator / drop-in constructor that a future
transcription must pass accepts every Uhlig formula that IS published and
rejects inconsistent or non-convergent polynomials, (4) the drop-in path
reproduces the registered coefficients and the empirical order of the
highest-order published Uhlig formula (4_5a, order 6) at fixed h, and (5) the
spec lists every source tried.  n = 4, <= 2e3 steps per run; about 3 s.
"""
from fractions import Fraction as F
from pathlib import Path
import sys

import numpy as np
import pytest

from znnbench import available, make_discretization, make_model, make_problem, simulate
from znnbench import registry
from znnbench.discretizations import uhlig as U
from znnbench.discretizations import uhlig_high as UH
from znnbench.discretizations import zead as Z
from znnbench.metrics import empirical_order, mssre

SPEC = Path(__file__).resolve().parents[1] / "docs" / "spec" / "uhlig_high.md"
DOI_U19 = "10.1080/10236198.2019.1627343"
DOI_UF = "10.1080/03081087.2019.1648375"


# ---------------------------------------------------------------------------
# (1) nothing unsourced is registered
# ---------------------------------------------------------------------------
def test_module_imported_and_registers_nothing():
    keys = available()["discretizations"]
    assert "znnbench.discretizations.uhlig_high" in sys.modules
    assert "znnbench.discretizations.uhlig_high" not in registry.IMPORT_ERRORS
    assert UH.UHLIG_HIGH_KEYS == {}
    high = [k for k in keys if k.startswith(("uhlig_5_", "uhlig_6_", "uhlig_7_", "uhlig_8_"))]
    assert high == [], f"order-7/8 Uhlig keys registered without a primary source: {high}"
    for key in UH.PENDING:
        assert key not in keys


# ---------------------------------------------------------------------------
# (2) pending metadata follows Uhlig's conventions
# ---------------------------------------------------------------------------
def test_u24_key_convention_matches_registered_uhlig_keys():
    # [U19]/[UF] "k_s" (degree k+s) -> [U24] j_s (j+s instants), as in uhlig.py
    assert UH.u24_key(2, 2) == "uhlig_2_3" and UH.u24_key(3, 3) == "uhlig_3_4"
    assert UH.u24_key(4, 5) == "uhlig_4_6"
    for k, s in ((2, 2), (3, 3), (4, 5)):
        d = make_discretization(UH.u24_key(k, s))
        assert d.instants == k + s + 1 and d.expected_order == k + 2


@pytest.mark.parametrize("key", sorted(UH.PENDING))
def test_pending_metadata_consistent(key):
    e = UH.PENDING[key]
    assert e["znn_order"] in (7, 8) and e["znn_order"] == e["k"] + 2
    assert e["named_in"] and e["coefficients_in"] and e["label"]
    if e["s"] is not None:
        assert e["degree"] == e["k"] + e["s"]
        assert e["instants"] == e["k"] + e["s"] + 1
        assert key == UH.u24_key(e["k"], e["s"])
        assert e["published_behaviour"]["uf_table3_h"] > 0     # [UF] Table 3 header


# ---------------------------------------------------------------------------
# (3) validator: accepts the published Uhlig formulas, rejects bad ones
# ---------------------------------------------------------------------------
PUBLISHED = {
    # name: (polynomial as printed, k, s, registered key)
    "2_2 [U19 (E), UF p4]": ([8, 1, -6, -5, 2], 2, 2, "uhlig_2_3"),
    "3_3 [U19 p.7, a=1]": ([237, 80, -182, -206, 1, 110, -40], 3, 3, "uhlig_3_4"),
    "4_5a [UF p.6]": (list(U.P9_4_5A), 4, 5, "uhlig_4_6"),
}


@pytest.mark.parametrize("name", sorted(PUBLISHED))
def test_validator_accepts_published_formulas(name):
    poly, k, s, key = PUBLISHED[name]
    rep = UH.validate_uhlig_polynomial(poly, k, s)
    assert rep["ok"], rep["reasons"]
    assert rep["trunc_order"] == k + 1 and rep["znn_order"] == k + 2
    assert rep["zero_stable"] and rep["strong"] and rep["h_max"] > 0
    d = UH.formula_from_polynomial(poly, k, s, label=name, source="test", doi=DOI_U19)
    ref = make_discretization(key)
    assert d.name == key and d.causal and d.expected_order == ref.expected_order
    if d.exact:
        assert d.coeffs_exact == ref.coeffs_exact                 # exact rational match
    else:
        assert np.allclose(np.array(d.coeffs, float), np.array(ref.coeffs, float), rtol=1e-14, atol=1e-15)
    assert abs(rep["second_root_modulus"] - ref.root_info()["second_largest_modulus"]) < 1e-12


def test_validator_matches_seed_construction():
    p, _ = U.uhlig_seed_polynomial(3, 3, [1, 110, -40])          # [U19] p.7 TPOLY
    rep = UH.validate_uhlig_polynomial(p, 3, 3)
    assert rep["ok"] and rep["tau_form"] == list(make_discretization("uhlig_3_4").coeffs_exact)
    assert abs(rep["second_root_modulus"] - 446 / 465) < 1e-4    # printed in TPOLY


def test_validator_rejects_bad_polynomials():
    # wrong length for the claimed type
    assert not UH.validate_uhlig_polynomial([8, 1, -6, -5, 2], 3, 3)["ok"]
    # one misread digit destroys the order conditions
    rep = UH.validate_uhlig_polynomial([8, 1, -6, -5, F(201, 100)], 2, 2)
    assert not rep["ok"] and any("truncation order" in r for r in rep["reasons"])
    # float transcription error of 1e-6 in a 16-digit coefficient is detected
    bad = list(U.P9_4_5A)
    bad[3] += 1e-6
    rep = UH.validate_uhlig_polynomial(bad, 4, 5)
    assert not rep["ok"] and any("truncation order" in r for r in rep["reasons"])
    # Taylor-Zhang formula: correct moments but only truncation order 2 < k+1 = 3
    assert not UH.validate_uhlig_polynomial([2, -3, 2, -1], 2, 1)["ok"]
    # symmetric Euler: order 2 but a root at -1 (zero-stable, not strongly stable)
    rep = UH.validate_uhlig_polynomial([1, 0, -1], 1, 1)
    assert rep["trunc_order"] == 2 and rep["zero_stable"] and not rep["strong"]
    assert not rep["ok"] and any("unit circle" in r for r in rep["reasons"])
    # consistent order-5 seed polynomial outside Uhlig's convergent range a <= 2.5
    p3, _ = U.uhlig_seed_polynomial(3, 3, [3, 110, -40])
    rep = UH.validate_uhlig_polynomial(p3, 3, 3)
    assert rep["trunc_order"] == 4 and not rep["zero_stable"] and not rep["ok"]
    with pytest.raises(ValueError):
        UH.formula_from_polynomial(p3, 3, 3, label="x", source="x", doi="x")
    # provenance is mandatory
    with pytest.raises(ValueError):
        UH.formula_from_polynomial([8, 1, -6, -5, 2], 2, 2, label="", source="s", doi="d")


# ---------------------------------------------------------------------------
# (4) the drop-in path reproduces the empirical order (fixed h, low frequency)
# ---------------------------------------------------------------------------
def test_dropin_pipeline_empirical_order_fixed_h():
    """4_5a of [UF] built via formula_from_polynomial: linear ZNN residual
    MSSRE = O(tau^6) at fixed h = h_opt on a low-frequency linear_system
    (freq = 0.5, T = 12 s; rho_opt = 0.911 so the transient is gone well
    before the steady window).  This is the check an obtained order-7/8
    formula would have to pass (slope 7 or 8)."""
    mk = lambda: UH.formula_from_polynomial(U.P9_4_5A, 4, 5, label="4_5a [UF]",  # noqa: E731
                                            source="Uhlig, LAMA 68 (2020), p9", doi=DOI_UF)
    d = mk()
    h = d.stability()["h_opt"]
    assert 0.09 < h < 0.096
    P = make_problem("linear_system", n=4, kappa=10, seed=1, T=12.0, freq=0.5)
    x0 = P.x_star(0.0)[None]
    taus = [0.024, 0.017, 0.012, 0.0085, 0.006]
    errs = [float(mssre(simulate(P, make_model("znn"), mk(), tau=tau, gamma=h / tau, x0=x0, seed=0,
                                 record_solution_error=False))[0]) for tau in taus]
    slope, n = empirical_order(taus, errs, floor=1e-11)
    assert n >= 4, errs
    assert abs(slope - d.expected_order) < 0.3, (slope, errs)


# ---------------------------------------------------------------------------
# (5) the spec documents the negative result with every source tried
# ---------------------------------------------------------------------------
def test_spec_lists_every_source_tried():
    text = SPEC.read_text()
    assert "NOT OBTAINED" in text and "registers nothing" in text
    for url, outcome in UH.SOURCES_TRIED:
        assert url in text, url
        assert outcome
    for key in UH.PENDING:
        assert f"`{key}`" in text
