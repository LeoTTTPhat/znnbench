# Area spec: structured_extra (pseudoinverse-free and hybrid GNN-ZNN models)

Files: `znnbench/models/structured_extra.py`, `tests/test_structured_extra.py`, this document.

This area closes the gap "lcc_ifznn" left open by `structured_baselines`
(`models/structured.py::PENDING_MODELS`). It also adds the hybrid GNN-ZNN models of
Dai et al. The existing `gznn` in `models/variants.py` (Stanimirović et al., *Mathematics* 2022)
applies the ZNN design to the gradient error JᵀE. It is a different model, so it is not
duplicated here.

| key | model | source (DOI) | check | requires |
|---|---|---|---|---|
| `lcc_znn` | LCCZNN, Eq. (12) with activation (13) (or (8), (9), linear, or any registry activation) | Zheng, Li, Zhang, *IEEE TAC* 69(7):4368–4379, 2024 (10.1109/TAC.2023.3319132) | **P** (accepted manuscript) | — |
| `inverse_free_znn` | **NOT REGISTERED** (adversarial verification, 2026-10-18): primary text not obtainable. Its secondary restatement (Xiao et al., *IEEE TAI* 2025, Eq. (10)) equals `make_model("lcc_znn", activation="linear")`; label such results "LCCZNN (12) with linear φ" | Chen, Pan, Zhang, Li, Tan, *Neural Netw.* 178:106462, 2024 (10.1016/j.neunet.2024.106462) | — | — |
| `hgnn_znn` | H-GNN-ZNN, A Ẋ = −γ(AAᵀ + I)F(AX − I) | Dai, Chen, Xiao, Jia, He, *IEEE TII* 18(4):2434–2442, 2022 (10.1109/TII.2021.3093115), Eq. (10) | **P** | `matrix_inverse` (or `scope='any'`) |
| `fa_gnn_znn` | FA-GNN-ZNN, gain γ + ν(‖E‖_F) from a Mamdani fuzzy controller | Dai et al. 2022, Eqs. (2)–(4), (11) | **P** (structure) / **ASSUMPTION** (membership breakpoints) | `matrix_inverse` (or `scope='any'`) |

Notation: the problem is E(x, t) = f(x, t) = 0 with f ∈ ℝᵐ, x ∈ ℝⁿ, J = ∂f/∂x and
f_t = ∂f/∂t at fixed x (`ctx.dEdt`, so every information regime of `simulate` is honoured).
γ (`ctx.gamma`) is the paper's μ (Zheng et al.), η (Chen et al.) or γ (Dai et al.).

---

## 1. LCCZNN: `lcc_znn` (Zheng, Li, Zhang 2024)

**Source read:** the accepted manuscript (12 pages, "accepted for publication in IEEE TAC";
University of Oulu repository, handle 10024/48415, file `nbnfioulu-202403212375.pdf`). It was
retrieved from its Wayback Machine snapshot of 2025-06-12, because the live repository sits
behind an interactive bot check (see §5).

**Problem (5):** f(t, x(t)) = 0 ∈ ℝᵐ, x ∈ ℝⁿ, n ≥ m, with at least one solution x*(t).

**Construction.** The scalar energy is (10)
$$\nu(t)=\tfrac12\|f\|_2^2=\tfrac12 f^\top f .$$
The ZNN design formula is imposed on ν (11):
$$\dot\nu=-\mu\,\phi(\nu),\qquad \dot\nu=f^\top J\dot x+f^\top \frac{\partial f}{\partial t}.$$
The explicit solution along Jᵀf is the **LCCZNN model (12)**:
$$\dot x=-\frac{J^\top f}{\|J^\top f\|_2^2}\left(f^\top\frac{\partial f}{\partial t}+\mu\,\phi\!\left(\frac{\|f\|_2^2}{2}\right)\right).$$
Remark 1 gives the regularised variant, with ‖Jᵀf‖₂² + λ in the denominator and λ > 0. The
paper "keep[s] focus on" λ = 0 in its analyses and experiments.

**Activations.** The novel activation is (13):
$$\phi(z)=\operatorname{sig}^r(z)\exp(|z|),\qquad 0<r<1 .$$
For comparison the paper uses (8) φ(z) = sig^r(z) and (9) φ(z) = sig^r(z) + z, with
sig^r(z) = sign(z)|z|^r. In ZNN-Bench these are `activation='exp_sig'` (default), `'sig'` and
`'sig_lin'`. `'linear'`, or any key of `znnbench.activations`, is applied to the scalar ν.

**Convergence times** (`ScalarEnergyActivation.settling_time`, exact for ν̇ = −μφ(ν)):
- Theorem 1, Eq. (14): t_AF3 = γ(1 − r, ν₀)/μ, where γ(a, z) = ∫₀ᶻ s^{a−1}e^{−s} ds is the lower incomplete gamma function (`lower_incomplete_gamma`).
- Eq. (15) bounds it: t_AF3 ≤ ν₀^{1−r}(1 + (1 − r)e^{−ν₀})/(μ(1 − r)(2 − r)) (`theorem1_bound`).
- Remark 3: t_AF1 = ν₀^{1−r}/(μ(1 − r)) for (8) and t_AF2 = ln(1 + ν₀^{1−r})/(μ(1 − r)) for (9).
- Theorem 3: t_AF3 < t_AF2 for small r. In every case t_AF3/t_AF1 < 1.

**Cost (Algorithm 1, Remark 2).** Each evaluation needs one Jᵀv product and three inner
products, O(mn), with no pseudoinverse and no linear solve. The model is explicit. The
reference ZNN (6), ẋ = −J†(μφ(f) + f_t), costs O(mn²).

**Robustness (Theorem 4).** Under additive noise δ(t) on ẋ (16), with d = fᵀJδ and |d| ≤ d̄,
the steady residual satisfies lim ‖f‖ ≤ (2 ln(1 + 2d̄/μ))^{1/2}. ZNN-Bench injects noise in the
design-formula space instead (`simulate`: J ẋ = model + ν). Theorem 4 is therefore not tested;
see the notes.

**Paper defaults.**
- Sec. IV-A: system (17), which is exactly ZNN-Bench's `nonlinear_equations` `variant='guo2018'`, with μ = 4, r = 0.7 and x(0) ∈ [0, 4]⁴.
- Fig. 3 caption: μ = 5, r = 0.6. **The text and the caption disagree**; ZNN-Bench uses the text values.
- Fig. 4: r = 0.7 with varying μ, then μ = 4 with varying r. Fig. 5: μ = 4, r = 0.6.
- Noise study: μ = 50, r = 0.5. UR5 controller (18): μ = 6, r = 0.5.

**ZNN-Bench defaults:** `activation='exp_sig'`, `r=0.7`, `lam=0.0`; μ is `gamma`.

**ASSUMPTIONS / implementation choices.**
1. With λ = 0, if ‖Jᵀf‖² is exactly 0, ẋ = 0 is returned. Eq. (12) is 0/0 at f = 0 and division by zero where f ≠ 0 but Jᵀf = 0. This is flagged in the docstring. At round-off-level f the field is not small (see the notes).
2. Algorithm 1, line 7 prints "κ₂ ← μφ(e)". It must be μφ(ν) (Eq. (12)), so ν is used. This is a typo in the paper.
3. f and Jᵀf use the data the model sees (`ctx.d`). f_t is `ctx.dEdt`, so `bd1`, `bd2` and `none` all work.

---

## 2. Inverse-free ZNN: `inverse_free_znn` (Chen, Pan, Zhang, Li, Tan 2024)

> **Status update (2026-10-18):** the verifier removed this registration (design contract: no registration from a secondary source alone). The text below documents the implementer's original reading; the tests that used the key now use `lcc_znn` with `activation='linear'`. See `NOT_REGISTERED` in `models/structured_extra.py`.


**The primary text was not read** (search log in §5). The equations come from a **secondary
restatement**: L. Xiao, D. Wang, Q. Zuo, H. Fan, H. Cai, "An inversion-free fuzzy zeroing neural
network under adaptive input range fuzzy scheme", *IEEE Trans. Artif. Intell.* (2025),
doi:10.1109/TAI.2025.3570287, Sec. II-D, Eq. (10). They write "Chen et al. [33] proposed the
NIFZNN model, which can be expressed as follows":
$$\dot x=-\frac{J^\top f}{\|f^\top J\|_2^2}\left(\frac{\eta}{2}\|f\|_2^2+f^\top f'_t\right),$$
with O(mn) cost. This is LCCZNN (12) with the linear activation φ(ν) = ν, so the energy obeys
ν̇ = −ην.

A second secondary source, IEEE TNNLS 2026 (doi:10.1109/TNNLS.2026.3672118, Sec. II), describes
the same reference [28] as "an LCCZNN model that replaces the CZNN formulation with a
scalar-valued energy function, thereby avoiding explicit matrix inversion". This is consistent.
The abstract of Chen et al. says the model "not only avoids the matrix inverse, but also avoids
matrix multiplication". This is also consistent: only Jᵀv products are used.

**ASSUMPTIONS.**
1. Xiao et al. restate the model for f(x, t) = 0. Chen et al. solve a time-variant optimisation problem with a nonlinear equality constraint, through its KKT system. In ZNN-Bench, the KKT residual is the problem's E (for example `tvqp_eq`), so the generic form applies.
2. The original may differ in details the restatement does not show, such as a regularisation, a nonlinear activation or the constants. Results under this key must be cited as "inverse-free ZNN (Chen et al. 2024) as restated by Xiao et al. (2025)".

The key is implemented as `LCCZNN(activation='linear', name='inverse_free_znn')`, and its
`describe()['source']` says "secondary".

**Note.** For the same gain, ‖f‖ decays as e^{−ηt/2}, half the rate of the
conventional ZNN ‖f‖ = ‖f₀‖e^{−ηt}.

---

## 3. H-GNN-ZNN and FA-GNN-ZNN: `hgnn_znn`, `fa_gnn_znn` (Dai et al. 2022)

**Source read:** the full text on IEEE Xplore (HTML), Sections II–IV, Tables I–II and
Examples 1–2.

**Problem (1):** A X(t) − I = 0 with a **constant** A ∈ ℝ^{n×n} that satisfies the inversion
condition. The error is (5), 𝓔(t) = AX(t) − I.
- ZNN (7): A Ẋ = −γ₁ F(AX − I).
- GNN (9): Ẋ = −γ₂ Aᵀ F(AX − I), from the gradient of (8), 𝓥 = ½‖AX − I‖²_F.

**H-GNN-ZNN (10):**
$$A\dot X(t)=-\gamma\,(AA^\top+I)\,\mathcal F(AX(t)-I).$$
Dai et al. attribute the hybrid to their Ref. [27]: P. Stanimirović, V. Katsikis, S. Li,
"Hybrid GNN-ZNN models for solving linear matrix equations", *Neurocomputing* 316:124–134,
2018, doi:10.1016/j.neucom.2018.07.058. That paper was **not read**, because ScienceDirect
refused automated access.

**FA-GNN-ZNN (11):**
$$A\dot X(t)=-(\gamma+\nu)\,(AA^\top+I)\,\mathcal F(AX(t)-I),$$
where ν is the fuzzy factor.

**Fuzzy controller (Sec. II-B).**
- **Input:** u = ‖𝓔‖_F.
- **Step 1, fuzzification:** trapezoidal membership functions (2), M(x) = 0 on [0, m], (x − m)/(n − m) on (m, n), 1 on [n, p], (q − x)/(q − p) on (p, q), and 0 from q on.
- **Step 2, rules:** ZO→ZO, PS→PS, PM→PM, PB→PB, combined with max–min composition (3).
- **Step 3, defuzzification:** maximum membership with the **mom** (middle of maximum) method, Eq. (4).
- **Range:** ν ∈ [0, 2.5], γ = 1 (Sec. IV).

**Theory.**
- Theorem 1 (LAf, PsAf): ‖Aξ(t)‖_F ≤ ‖Aξ(0)‖_F exp(−(γ + ν)(1 + λ_min(AAᵀ))t).
- Theorem 2 (bounded implementation noise ‖ΔR‖_F ≤ ρ): lim ‖X − X*‖_F ≤ ρ/((γ + ν)(1 + α)√α).
- Table II, exponential convergence rates: ZNN γ; GNN γα; H-GNN-ZNN γ(1 + α); FA-GNN-ZNN (γ + ν)(1 + α), with α = λ_min(AAᵀ).
- Paper examples: Example 1 is A = [[1, −2, 3], [0, −3, −2], [1, 2, 3]] and Example 2 is a 5×5 matrix, both with γ = 1. Activations used: LAf, PsAf and SbpAf. The activation parameters are not given.

**ZNN-Bench formulation.** With x = vec_row(X), J is left multiplication by A and Jᵀ is left
multiplication by Aᵀ (capability `matrix_inverse`). Since J⁻¹(JJᵀ + I) = Jᵀ + J⁻¹,
$$\dot x=-g\,J^\top F(E)+J^{-1}\big(-g\,F(E)\;[-\,\partial_tE]\big),\qquad g=\gamma\ (\text{H}),\ \ g=\gamma+\nu(\|E\|_F)\ (\text{FA}).$$
Each evaluation needs one linear solve (the ZNN part) and one Jᵀ product (the GNN part). The
test `test_hgnn_rhs_matches_eq10_on_time_varying_inversion` checks this identity against a
dense evaluation of A⁻¹(−γ(AAᵀ + I)F − ȦX).

**Defaults:** `activation='linear'`, `feedforward=False` (Eq. (10)/(11) verbatim),
`scope='matrix_inverse'`. For FA-GNN-ZNN: `u_max=1.0`, `nu_max=2.5`, `grid=1001`.
Use the registry activations `power_sigmoid` and `sign_bi_power` (with `r` = ι) for Dai's PsAf
and SbpAf.

**ASSUMPTIONS.**
1. **`feedforward=True` is a ZNN-Bench extension.** It adds the ZNN time-derivative term, i.e. the design formula Ė = −γ(JJᵀ + I)F(E), for time-varying A. Dai et al. treat a constant A only. With the default, the model applied to time-varying problems has no time-derivative information (`uses_dEdt=False`) and lags, like GNN.
2. **`scope='any'` is a ZNN-Bench extension.** It applies the same formula to any problem. For non-square J the problem's min-norm solve gives J⁺(JJᵀ + I) = Jᵀ + J⁺ when J has full row rank.
3. **The membership breakpoints (m, n, p, q) of the eight fuzzy sets are not given in the paper.** `MamdaniMomFactor` uses a uniform partition: ZO is a left shoulder (0, 0, 0, h), PS = (0, h, h, 2h), PM = (h, 2h, 2h, 3h) and PB is a right shoulder (2h, 3h, ∞, ∞), with h = u_max/3 for the input and h = ν_max/3 for the output. Then ν(0) = 0, ν(u ≥ u_max) = ν_max = 2.5 (the paper's range) and ν is non-decreasing in u. u_max = 1 is a ZNN-Bench choice. The companion papers of the same group do not supply the breakpoints either: Jia, Xiao, Dai, *IEEE TFS* 29(6) 2021, doi:10.1109/TFUZZ.2020.2981001, shows its memberships only in a figure, and Dai et al., *IEEE TFS* 31(11) 2023, doi:10.1109/TFUZZ.2023.3272653, uses Gaussian/S-type sets with no numbers in the text. **Results of `fa_gnn_znn` are therefore not reproductions of Dai's figures.**
4. **Left shoulder of Eq. (2).** Taken literally, Eq. (2) gives M(0) = 0 when m = n = 0. `trapmf` treats m = n as a left shoulder, with M = 1 for x ≤ n.
5. **mom on a grid.** mom is evaluated on a uniform grid of the output universe: the mean of the grid points where the aggregated membership is maximal. This is what a sampled-universe fuzzy toolbox does. The grid spacing ν_max/(grid − 1) bounds the deviation from the continuous mom.

**Typos in the paper (not used).**
- The PsAf is printed with (1 + e^{−βx})/(1 − e^{−βx}) in the second factor, which is singular at 0. The registry `power_sigmoid`, (1 − e^{−ξx})/(1 + e^{−ξx}), is the standard form.
- The text calls α "the minimal eigenvalues of the matrix AᵀA" in Example 1 but uses AAᵀ in Theorem 2. The two matrices have the same eigenvalues.

---

## 4. Tests (`tests/test_structured_extra.py`, 20 tests, about 4 s)

| test | checks |
|---|---|
| `test_registry_and_flags` | the four keys, `requires`, `uses_dEdt`, implicit support, defaults (exp_sig, r = 0.7; ν_max = 2.5), argument validation |
| `test_lcc_rhs_matches_eq12_and_eq13` | dense, independent transcription of (12) with (13), (8), (9) and the λ of Remark 1; the design formula fᵀ(Jẋ + f_t) = −μφ(ν) holds; ẋ ∥ Jᵀf |
| `test_lcc_energy_follows_theorem1_in_continuous_time` | RK4 at τ = 1e-3 on system (17): μt + γ(1 − r, ν(t)) is constant to 1e-6 (Eq. (14)); t_AF3 ≤ bound (15); t_AF3 < t_AF1 for all ν₀ (Theorem 3 / Remark 3) |
| `test_lcc_discrete_floor_shrinks_with_tau` | the RK4 residual floor near f = 0 shrinks with τ (1e-3 → 1e-4: below 0.05 and at least 5× smaller) |
| `test_lcc_and_nifznn_need_no_linear_solve` | cost claim: on a wrapper whose `solve_J` raises, `lcc_znn` and `inverse_free_znn` run under Euler and RK4 (0 solves, Jᵀ only); `znn` raises |
| `test_nifznn_is_lcc_with_linear_activation_and_exponential_energy` | Eq. (10) of Xiao et al. transcribed independently; `inverse_free_znn` equals `lcc_znn(activation='linear')`; RK4: ν(t) = ν₀e^{−ηt} to 1e-6 |
| `test_lcc_regularisation_and_zero_denominator` | the exact-zero guard; the unregularised field stays O(1) at round-off-level f; with λ > 0 the field vanishes as f → 0 |
| `test_lcc_redundant_robot_and_discrete_smoke` | Panda (m = 3 < n = 7), Euler h = 0.06: residual drops below 5 % of its initial value |
| `test_lcc_discrete_euler_smoke_tvne[...]` | Euler on system (17): no divergence and the residual decreases |
| `test_hgnn_rhs_matches_eq10_on_time_varying_inversion` | (10), with and without feedforward, for LAf, SbpAf and PsAf with three gains, against A⁻¹(−γ(AAᵀ + I)F − ȦX) |
| `test_hgnn_continuous_time_matrix_exponential_and_rates` | Dai Example 1, linear F, RK4: E(t) = expm(−γ(AAᵀ + I)t)E(0) to 1e-8; the Theorem 1 rate bound; the Table II ordering against `znn` and `gnn` |
| `test_fuzzy_factor_pipeline` | Eq. (2) shape and shoulders; ν(0) = 0 and ν(u ≥ u_max) = 2.5; monotonicity; mom equals the midpoint of the α-cut of the dominant rule |
| `test_fa_gnn_znn_gain_and_faster_than_hgnn` | (11) equals (10) with gain γ + ν; it converges faster than H-GNN-ZNN on Example 1 (Fig. 2) and keeps the rate γ(1 + α) of Theorem 1 |
| `test_hgnn_requires_capability_and_scope_any` | the capability error; the `scope='any'` formula on `linear_system` |
| `test_hybrid_discrete_euler_smoke[...]` | Euler on time-varying inversion (κ = 3, γ = 2, τ = 1e-2) |
| `test_hybrid_euler_instability_threshold` | Euler is stable iff γτ(1 + σ_max(A)²) < 2 (0.8× converges, 1.2× blows up) on Example 1 |
| `test_mini_smoke_for_new_keys` | every registered problem × the 4 keys × {euler, implicit_euler_err, rk4} through `experiments/smoke_matrix.run_cell`: no error or timeout |

---

## 5. Source search log (every source tried)

Privacy: every request was anonymous. No email address or personal identifier was sent. The
Crossref polite-pool header used `znnbench@example.org`.

**LCCZNN (10.1109/TAC.2023.3319132): OBTAINED (primary).**
- OpenAlex `best_oa_location`: none (`closed`). Semantic Scholar `openAccessPdf`: empty (`CLOSED`). arXiv API: no preprint.
- University of Oulu repository, handle 10024/48415 (accepted version, "OpenAccess", IEEE rights statement): the live site returns HTTP 403 with a Cloudflare interactive "Verify you are human" check, in curl and in the in-app browser. **Not bypassed.** jultika.oulu.fi: same 403.
- **Wayback Machine:** CDX lists `oulurepo.oulu.fi/bitstream/handle/10024/48415/nbnfioulu-202403212375.pdf` (snapshot 2025-06-12, digest ZT2D63MU…, 3.58 MB). This copy was read (pdftotext) and is the source of §1. A 2024-05-18 snapshot is truncated at 1 MB.

**Inverse-free ZNN (10.1016/j.neunet.2024.106462): primary NOT obtained; secondary restatement used.**
- OpenAlex: `closed`, no repository full text. Semantic Scholar: `CLOSED`. Europe PMC (PMID 38901094): "subscription required" only.
- Oulu repository, handle 10024/61885: per Google Scholar and the search engines it is the accepted manuscript, CC BY-NC-ND 4.0, embargo lifted 2026-06-12, file `nbnfioulu-202604232734.pdf`. Cloudflare interactive check (403). No Wayback snapshot of the handle or the bitstream. `urn.fi` and the `jultika.oulu.fi/files/...` pattern also return 403.
- VTT research portal (cris.vtt.fi): metadata and abstract only, no file.
- CORE (`core.ac.uk/download/628873440.pdf`, API v3): 403 Cloudflare, or HTTP 429 without an API key. No Wayback snapshot.
- ScienceDirect (`S0893608024003861`): the in-app browser was refused by Elsevier's bot protection ("There was a problem providing the content", CPE00001). **Not circumvented.**
- Google Scholar cluster 14565672860499816687 ("All 6 versions"): ScienceDirect, Europe PMC, VTT, ACM DL (abstract), PubMed, and the Oulu PDF (blocked). No arXiv, ResearchGate or GitHub copy was found.
- Related Oulu record of the same authors, handle 10024/63000 ("Inverse-free Jacobian-estimated ZNN", ESWA 2025): no Wayback snapshot.
- **Secondary sources read** (IEEE Xplore HTML full text through institutional access already present in the browser):
  - Xiao, Wang, Zuo, Fan, Cai, *IEEE TAI* 2025, doi:10.1109/TAI.2025.3570287, Sec. II-D, Eq. (10). **This restatement was used.**
  - *IEEE TNNLS* 2026, doi:10.1109/TNNLS.2026.3672118, Sec. II: describes the model, no equation for [28].
- **Checked, no restatement:**
  - *IEEE TSMC* 2026, doi:10.1109/TSMC.2026.3658893: cited in the introduction only.
  - *IEEE/CAA JAS* 2025, doi:10.1109/JAS.2025.125627.
  - *IEEE TETCI* 2026, doi:10.1109/TETCI.2026.3668587.
  - arXiv:2410.19458 and edepot.wur.nl/702852 (ISA Trans. 2025): citation only.
  - Jiang, Gui, Jin, Li, arXiv:2507.00387 (review).
  - The IJSS 2024 survey (bura.brunel.ac.uk/bitstream/2438/30318).
  - *Artif. Intell. Rev.* 2026, doi:10.1007/s10462-026-11628-0 (survey).

**Dai et al. H-GNN-ZNN / FA-GNN-ZNN (10.1109/TII.2021.3093115): OBTAINED (primary).**
- OpenAlex and Semantic Scholar: closed.
- IEEE Xplore document 9468339: full text (HTML) read, including Tables I–II as images.
- Ref. [27], Stanimirović–Katsikis–Li 2018 (10.1016/j.neucom.2018.07.058): ScienceDirect refused automated access. Not read.
- Membership breakpoints: absent from the paper and from the two companion papers checked (TFS 2021, TFS 2023; see §3, ASSUMPTION 3).

**What a human can do:** open the Oulu records 10024/61885 (inverse-free ZNN, open since
2026-06-12) and 10024/48415 in a normal browser, which passes the Cloudflare check
interactively. They can also use a library subscription to ScienceDirect. They can then confirm
Eq. (10) of Xiao et al. against Chen et al. and read the Stanimirović 2018 HZNN definitions.

---

## 6. Observations from ZNN-Bench runs (not claims of the source papers)

1. **LCCZNN finite time vs. discrete reality.** On system (17), with μ = 4 and r = 0.7, the invariant μt + γ(1 − r, ν(t)) = γ(1 − r, ν₀) holds to 1e-7 with RK4 until ν is about 0.02. Then the trajectory stalls at a τ-dependent floor. At T = 1 the residual is about 0.35 for τ = 1e-3 and about 0.017 for τ = 1e-4. An adaptive LSODA run (rtol 1e-10) did not reach T = 2 s within two minutes: its step size collapses near f = 0. The cause is the direction field Jᵀf/‖Jᵀf‖² near f = 0, whose limit depends on the direction from which f → 0. The model controls only the scalar ν and cannot hold f = 0 against the drift f_t, which shows up as residual chatter. The paper reports a "precision between 10⁻³ and 10⁻⁴", consistent with a solver floor.
2. **Activation (13) contains exp(ν).** From the default random starts of the matrix problems (ν₀ ≈ 10²–10⁵) the field is of order e^{ν₀}, and every fixed-step scheme overflows. Smoke cells (γ = 10, τ = 1e-2, T = 1): `lcc_znn` (Euler, Taylor-Zhang, RK4) diverged in 25 of 39 cells: all 24 cells of the matrix problems, `linear_system` and `nonlinear_equations` (default random starts), plus `tvqp_ineq` under Taylor-Zhang. It was ok in the other 14 cells: the two robots, `tv_unconstrained_opt` and the `tvqp_*` problems. `inverse_free_znn` was ok in 39 of 39.
3. **NIFZNN** halves the residual decay rate of ZNN for the same gain (‖f‖ ∝ e^{−ηt/2}).
4. **H-GNN-ZNN under Euler** is stable only if γτ(1 + σ_max(A)²) < 2 on constant A (tested). On the default κ = 10 matrices with γ = 10 and τ = 1e-2 it diverges. The GNN term AAᵀ makes the hybrid's faster continuous-time rate γ(1 + α) come with a step-size limit set by σ_max², not σ_min.

## 7. Requests to other owners (no core change needed)

- **docs/ZOO.md (integrator):**
  - Add rows for `hgnn_znn` and `fa_gnn_znn`. Without them, `tests/test_integration.py::test_zoo_catalogue_covers_every_registered_key` fails.
  - Update the "not implemented" sentence: `lcc_znn` is now registered with check P, and `inverse_free_znn` with check S.
- **models/structured.py and tests/test_structured_baselines.py (structured_baselines owner):** `PENDING_MODELS` still lists `lcc_znn` and `inverse_free_znn`, and the test asserts that set. Both keys are now registered here, so the docstring, the set and the test should be updated together. `docs/spec/structured_baselines.md` §4 should point here.
- **docs/spec/SMOKE_MATRIX.md:** regenerate with `experiments/smoke_matrix.py`. The new keys add 4 models × 13 problems.

## 8. Verification (adversarial verifier, 2026-10-18; section written by the lead from the verifier's tests)

The adversarial verifier was cut off by a usage limit after it had changed the code and the tests,
but before it wrote this section. Its checks live in `tests/test_structured_extra.py`; the lead
summarised them here and re-ran the file (all tests pass, strict mode).

| item | verdict | evidence |
|---|---|---|
| LCCZNN Eqs. (10)-(15), Remark 1, Algorithm 1, Theorems 1/4 re-read (archived accepted manuscript, Wayback snapshot 20240612171122) | correct | `test_lcc_rhs_matches_eq12_and_eq13` |
| `r` forwarding to registry activations; freezing of time-varying data inside one rhs call | fixed | `test_lcc_registry_activation_r_and_time_varying` |
| Theorem 1 invariant μt + γ(1 - r, ν(t)) = γ(1 - r, ν₀) in continuous time | correct | holds to 1e-8 down to ν ≈ 1e-5 with a stiff reference solver (Radau, rtol 1e-10): `test_lcc_theorem1_holds_with_stiff_reference_solver` |
| Mechanism of the fixed-step residual floor (§6.1) | finding | the Jacobian of the LCCZNN field grows like 1/‖f‖; on system (17) its spectral radius passes the RK4 real-axis limit 2.785/τ near ν ≈ 0.013 for τ = 1e-4, so RK4 at τ ∈ {1e-4, 3e-4} leaves the invariant at the same, τ-independent ν, while τ = 1e-5 follows it to t = 0.2 (`test_lcc_rk4_loses_theorem1_through_stiffness`); past that point the floor shrinks roughly ∝ τ (`test_lcc_discrete_floor_shrinks_with_tau`) |
| Activation (13) convergence-time class | finding (not in the paper) | (13) is FIXED-time: t_AF3 < Γ(1 - r)/μ for every ν₀, while the paper's bound (15) grows like ν₀^{1-r}; Remark 3's ordering (13) faster than (8) holds for every ν₀ tested |
| `inverse_free_znn` (Chen et al. 2024) | removed | primary text not obtainable; the secondary restatement equals `lcc_znn` with linear φ (`test_lcc_linear_matches_secondary_restatement_of_nifznn`); see `NOT_REGISTERED` |
| No linear solve in LCCZNN | correct | `test_lcc_needs_no_linear_solve` (counting problem: zero `solve_J` calls) |
| H-GNN-ZNN Eq. (10) on time-varying inversion; continuous-time rate γ(1 + α) | correct | `test_hgnn_rhs_matches_eq10_on_time_varying_inversion`, `test_hgnn_continuous_time_matrix_exponential_and_rates` |
| Dai et al., proof of Theorem 1, step (15): Tr(EᵀMF(E)) ≥ Tr(EᵀME) with M = AAᵀ + I | wrong in the source | fails for the power-sigmoid activation with Dai's Example 1 matrix (non-diagonal M), although \|F(e)\| ≥ \|e\| element-wise; the final rate bound (1 + α)‖e‖² still holds at the counterexample (`test_dai_theorem1_step15_needs_linear_F`) |
| H-GNN-ZNN under forward Euler | finding | stable only if γτ(1 + σ_max(A)²) < 2 on constant A (`test_hybrid_euler_instability_threshold`) |
| FA-GNN-ZNN fuzzy factor pipeline (trapmf, max-min, mom) | correct structure; breakpoints remain an ASSUMPTION | `test_fuzzy_factor_pipeline`, `test_fa_gnn_znn_gain_and_faster_than_hgnn`; the verifier had planned extra trapmf edge-case tests and did not finish them |
