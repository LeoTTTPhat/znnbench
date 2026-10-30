# Spec: Uhlig's order-7/8 look-ahead formulas (`uhlig_high.py`)

Area owner files: `znnbench/discretizations/uhlig_high.py`, `tests/test_uhlig_high.py`, this file.
Related: `znnbench/discretizations/uhlig.py` and `docs/spec/discretizations.md` §3 (Uhlig's formulas of
orders 4 to 6, registered as `uhlig_2_3`, `uhlig_3_4` and `uhlig_4_6`).

## Status (2026-10-18): NOT OBTAINED. The module registers nothing.

Uhlig reports convergent look-ahead finite-difference formulas of ZNN truncation orders 7 and 8, but he
published their coefficients, or the seeds that generate them, only in MATLAB files on his Auburn web
space. We could not obtain any of those files from the live server, the Wayback Machine, archive.today,
Arquivo.pt, Common Crawl, GitHub or the publishers (§3 lists every URL tried). None of Uhlig's papers,
their arXiv LaTeX sources, the archived talk slides or the published versions print the coefficients.
**So `uhlig_high.py` registers no discretization.** Following the project rule, we did not run our own
seed search to regenerate the formulas: it would produce *a* convergent formula of the same type, not
Uhlig's.

What the module does provide:

* `PENDING`: metadata for the formulas named in the sources (§2).
* `SOURCES_TRIED`: every source queried, with its outcome. The test checks that this file lists each one.
* `validate_uhlig_polynomial(poly, k, s)` and `formula_from_polynomial(...)`: the checks a future
  transcription must pass, and a constructor that builds an **unregistered** `ZeaD` from it (§5).

## 1. What the sources say

* **[U19]** F. Uhlig, *The construction of high order convergent look-ahead finite difference formulas for
  Zhang neural networks*, J. Difference Equ. Appl. 25 (2019) 930–941, doi:10.1080/10236198.2019.1627343,
  arXiv:1904.10569.
  * The abstract says the method creates new convergent formulas of truncation orders 5 through 8.
  * p.7: the code `runconv1step(runs,jend,k,s)` searches for polynomials of degree k+s with ZNN order k+2,
    for k ≤ 6. The highest order tested is 8, at k = 6.
  * p.8: the order-5 to order-8 formulas sit "at the bottom of our list" inside `runconv1step.m` and can be
    regenerated from the seed information there.
  * p.8: the polynomials "3_3, 4_5, 5_5 and 5_6" (orders 5 to 7) were tested in [UF].
  * Only two runs are printed: 2_2 (seed [-5, 2]) and 3_3 (seed [a, 110, -40]). Both are registered in
    `uhlig.py`.
  * Ref. [7] gives the code location: `http://www.auburn.edu/~uhligfd/m_files/ZNNpolynoms`.
* **[UF]** F. Uhlig, *Zhang neural networks for fast and accurate computations of the field of values*,
  Linear Multilinear Algebra 68 (2020) 1894–1910, doi:10.1080/03081087.2019.1648375, arXiv:1904.10568.
  * p.6 prints $p_9$ of 4_5a (registered as `uhlig_4_6`).
  * Table 3 tests 1_2, 3_3, 5_5 and 5_6b, with orders 3, 5, 7 and 7, on the field-of-values problem
    (n = 243).
  * Figure 2 shows only the **root plots** of 5_5 and 5_6b (`roots55.jpg` and `roots56b.jpg` in the arXiv
    source).
  * Refs. [19] and [20]: `runconv1step1Plot.m` and `FOVZNN5_6bshorteigsym.m` at
    `http://www.auburn.edu/~uhligfd/m_files/FOVviaZNN/`.
* **[U24]** F. Uhlig, *Zhang neural networks: an introduction to predictive computations for discretized
  time-varying matrix problems*, Numer. Math. 156 (2024) 691–739, doi:10.1007/s00211-023-01393-5
  (open access), arXiv:2008.02724v1–v6.
  * Step 4 takes the formula "from the table of known convergent look-ahead finite difference formulas of
    type j_s" in [45] = [U19] and [46].
  * [46] is "List of look-ahead convergent finite difference formulas", located at
    `.../~uhligfd/m_files/ZNNSurveyExamples` under `Polyksrestcoeff3.m`.
  * The paper reports formulas of all types j_s with 1 ≤ j ≤ 6 and j ≤ s ≤ j+3, orders 3 to 8.
  * It prints no polynomial of order above 4. The published version has no supplementary material.
* **[ELA23]** F. Uhlig, *Adapted AZNN methods for time-varying and static matrix problems*, Electron. J.
  Linear Algebra 39 (2023) 164–180, doi:10.13001/ela.2023.7417, arXiv:2209.10002. It uses type 4_5 and
  relies on `Polyksrestcoeff3.m` in `m_files/AZNN/`.

## 2. Pending formulas (`PENDING`)

Naming follows the convention of `uhlig.py`. A [U19]/[UF] label "k_s" means a seed of length s, a
polynomial of degree k+s (k+s+1 instants) and ZNN order k+2. The registry key uses the [U24] "j_s" form
(j+s instants), so it is `uhlig_<k>_<s+1>`. The test `test_u24_key_convention_matches_registered_uhlig_keys`
checks this against the registered keys `uhlig_2_3`, `uhlig_3_4` and `uhlig_4_6`.

| would-be key | source label | degree / instants | ZNN order | published evidence (no coefficients) |
|---|---|---|---|---|
| `uhlig_5_6` | 5_5 [U19, UF] | 10 / 11 | 7 | [UF] Table 3: h = 0.0069; 11.5 digits at τ = 1e-3, 14.7 at τ = 1e-4. Fig. 2: five interior roots close to the unit circle |
| `uhlig_5_7` | 5_6b [UF] (= "5_6" of [U19]) | 11 / 12 | 7 | [UF] Table 3: h = 0.016; 13.7 digits at τ = 1e-3, 15.2 at τ = 1e-4. Fig. 2: roots spread more evenly |
| `uhlig_6_*` | order-8 types, k = 6 | 12–15 / 13–16 (s = 6…9, per [U24]) | 8 | No individual order-8 polynomial is named or printed in any source we read |

**Where the coefficients were.** Archived Apache directory listings, fetched raw (`id_`) from the Wayback
Machine, show that the files existed:

| file (under `webhome.auburn.edu/~uhligfd/`) | size | last modified | listing captured |
|---|---|---|---|
| `m_files/ZNNSurveyExamples/Polyksrestcoeff3.m` | 4.8K | 04-Aug-2020 | 2024-01-27 |
| `m_files/AZNN/Polyksrestcoeff3.m` | 7.5K | 02-Oct-2020 | 2024-01-27 |
| `m_files/Eigencurves/Polyjsrestcoeff2.m` | 4.3K | 08-Sep-2019 | 2023-03-15 |
| `m_files/ZNNpolynoms/runconv1step1Plot.m` | 8.3K | 08-Sep-2019 | 2024-04-23 |
| `m_files/ZNNpolynoms/singleconv1stepahead.m` | 2.3K | 16-Jul-2018 | 2024-04-23 |
| `m_files/FOVviaZNN/runconv1step1Plot.m` | 8.1K | 23-Dec-2018 | 2022-11-25 |
| `m_files/FOVviaZNN/FOVZNN5_6beigshort2.m` | 7.8K | 03-Jan-2019 | 2022-11-25 |
| `m_files/FOVviaZNN/FOVZNN5_6beigshortsym.m` | 7.7K | 18-Jan-2019 | 2022-11-25 |

No archive captured any of these files (§3). The [U19] code `runconv1step.m` is not in either listing; the
directories contain only the later `runconv1step1Plot.m`.

## 3. Every source tried (2026-10-18)

No request carried a personal identifier. The User-Agent was `znnbench-research/1.0`, and the one API that
needs an e-mail (Unpaywall) received `znnbench@example.org`. Where a publisher served a bot challenge we
stopped; we did not try to get around it.

| URL | outcome |
|---|---|
| `https://www.auburn.edu/~uhligfd/m_files/` | 302 -> webhome.auburn.edu, 404 |
| `https://webhome.auburn.edu/~uhligfd/m_files/` | 404 |
| `https://webhome.auburn.edu/~uhligfd/m_files/ZNNpolynoms/` | 404 |
| `https://webhome.auburn.edu/~uhligfd/m_files/ZNNSurveyExamples/` | 404 |
| `https://webhome.auburn.edu/~uhligfd/` | 200, 2004-era homepage without m_files links |
| `http://web.archive.org/cdx/search/cdx?url=auburn.edu/~uhligfd/*&output=json` | 1062 captures (232 distinct URLs). Every `m_files` URL is a 301/302 redirect to webhome.auburn.edu; no `.m` content |
| `http://web.archive.org/cdx/search/cdx?url=webhome.auburn.edu/~uhligfd/*&output=json` | 173 captures. Directory listings of `m_files/`, `ZNNpolynoms/`, `ZNNSurveyExamples/`, `AZNN/`, `FOVviaZNN/`, `Eigencurves/`, `T-VMatrixEigenv/`, `DecompMatrFoV/`, `ZNNpapers2019/` were captured; none of the ZNN `.m` files was |
| `http://archive.org/wayback/available?url=<host>/~uhligfd/m_files/<dir>/<file>.m` | no snapshots, for 11 files × 2 hosts (Polyksrestcoeff3.m ×2, Polyjsrestcoeff2.m, runconv1step1Plot.m ×2, singleconv1stepahead.m, FOVZNN5_6beigshort2.m, FOVZNN5_6beigshortsym.m, FOVZNN4_5aeigshortFoV3for.m ×2, Zmatrixeig3_3b.m) |
| `http://web.archive.org/web/2024id_/http://webhome.auburn.edu/~uhligfd/m_files/AZNN/Polyksrestcoeff3.m` | 404 |
| `http://web.archive.org/web/2024id_/http://webhome.auburn.edu/~uhligfd/m_files/ZNNSurveyExamples/Polyksrestcoeff3.m` | 404 |
| `http://web.archive.org/web/2024id_/http://webhome.auburn.edu/~uhligfd/m_files/FOVviaZNN/FOVZNN5_6beigshort2.m` | 404 |
| archived PDFs on Uhlig's site: `Talkslides2022/ZeroingNeuralNetworks5ab.pdf`, `Talkslides2022/ExtraOrdResults2024.pdf`, `m_files/ZNNpapers2019/constructconv1stepahead .pdf`, `m_files/ZNNpapers2019/FOVviaZNNa.pdf` | preprints and slides of [U24], [U19] and [UF]. They print no order-7/8 coefficients and point to `Polyksrestcoeff3.m` |
| `https://archive.ph/timemap/<url of Polyksrestcoeff3.m / runconv1step1Plot.m>` | no mementos |
| `https://arquivo.pt/wayback/cdx?url=<url of Polyksrestcoeff3.m / runconv1step1Plot.m>` | no captures |
| `http://timetravel.mementoweb.org/api/json/2021/<url>` | service unreachable (DNS) |
| `https://index.commoncrawl.org/CC-MAIN-<2018-30 ... 2025-13>-index?url=<host>/~uhligfd/m_files/*` | no captures in 60 indexes × 2 hosts (www, webhome). All 120 queries were answered "No Captures"; 26 of them only after retrying 504 gateway timeouts. They cover every crawl from mid-2018 to early 2025, the period in which the files existed |
| `https://arxiv.org/pdf/1904.10569v1 (+ /e-print)` | orders ≤ 5 only (2_2 and 3_3 seeds). The TeX source has no seed list; one comment names a local path to `runconv1step.m` |
| `https://arxiv.org/pdf/1904.10568v1 (+ /e-print)` | $p_9$ of 4_5a only. 5_5 and 5_6b appear only as root plots (`roots55.jpg`, `roots56b.jpg`) |
| `https://arxiv.org/pdf/1904.10566v1 (+ /e-print)` | only 3_3b, which equals 3_3 with a = 1 (`uhlig_3_4`) |
| `https://arxiv.org/pdf/2008.02724v1 ... v6 (+ /e-print v1, v6)` | points to Polyksrestcoeff3.m. No table of higher-order formulas, including in TeX comments |
| `https://arxiv.org/pdf/2209.10002v1, v2 (+ /e-print)` | 4_5 only; points to Polyksrestcoeff3.m |
| `https://arxiv.org/pdf/2002.01274v1, 2006.01241v5, 2006.01215v2 (+ /e-print)` | no coefficients |
| `https://link.springer.com/content/pdf/10.1007/s00211-023-01393-5.pdf` | open access; no supplementary material; points to Polyksrestcoeff3.m |
| `https://link.springer.com/article/10.1007/s00211-023-01393-5` | bot challenge (not bypassed) |
| `https://journals.uwyo.edu/index.php/ela/article/download/7417/6133` | ELA 2023: points to Polyksrestcoeff3.m |
| `https://www.tandfonline.com/doi/suppl/10.1080/10236198.2019.1627343` | 403 bot challenge (not bypassed). JDEA published version and any supplement not read |
| `https://www.tandfonline.com/doi/suppl/10.1080/03081087.2019.1648375` | 403 bot challenge (not bypassed) |
| `https://api.crossref.org/works/<DOI of U19, UF, U24>` | no supplement relation |
| `https://api.unpaywall.org/v2/<DOI>?email=znnbench@example.org` (U19, UF, U24, ELA23, NLA23, LAA19) | only the arXiv and publisher copies listed above |
| `http://export.arxiv.org/api/query?search_query=au:Uhlig_Frank` | Uhlig's ZNN papers: 1904.10566/8/9, 2002.01274, 2006.01215, 2006.01241, 2008.02724, 2209.10002 (all read) |
| `https://api.github.com/search/code?q=<Polyksrestcoeff, runconv1step1Plot, FOVZNN5_6bshorteigsym, 1.632891580619644>` | 0 hits. Also 0 for `Polyjsrestcoeff2`, `singleconv1stepahead`, `tvMatrSquareRootwEulerStartv`, `0.03397751824789656` (a coefficient of $p_9$) and `uhligfd m_files`; repository search found nothing either |
| web search for the file names, and MathWorks File Exchange | nothing beyond the papers above |

## 4. Why nothing was reconstructed

* **Root plots are not coefficients.** Fitting a degree-10 or degree-11 polynomial to roots read off
  [UF] Fig. 2 would give a polynomial that satisfies the order conditions only if we imposed them
  ourselves. That leaves s−1 = 4 or 5 free parameters fitted to pixel positions: a new formula, not a
  transcription.
* **A seed search is forbidden.** `uhlig.uhlig_seed_polynomial` reproduces Uhlig's construction exactly,
  but his order-7/8 seeds are in the missing files. Searching for new seeds is exactly the "own search"
  that the project rules exclude.
* **The order-8 types are not even identified.** No source names a specific order-8 polynomial.

## 5. Drop-in path once a primary copy is found

1. Transcribe the polynomial (descending powers) and its seed exactly from `Polyksrestcoeff3.m` or
   `runconv1step1Plot.m`. If the file gives seeds, use `uhlig.uhlig_seed_polynomial(k, s, y)`, which is
   exact in rational arithmetic.
2. Run `validate_uhlig_polynomial(poly, k, s)`. A transcription must give `ok`: degree k+s, a consistent
   tau-form with truncation order ≥ k+1 (exact for rationals, relative 1e-10 for 16-digit floats), the
   strong root condition and $h_{\max} > 0$. The test shows that a single misread rational digit, or a
   1e-6 error in one 16-digit coefficient, fails the moment check.
3. Build the formula with `formula_from_polynomial(poly, k, s, label=..., source=..., doi=...)` and add an
   explicit `register_discretization(u24_key(k, s))` line in `uhlig_high.py`, with the source in the
   docstring. Then add the key to `UHLIG_HIGH_KEYS` and `docs/ZOO.md`, and move it out of `PENDING`.
4. Verify the empirical order at fixed $h=h_{\rm opt}$ with
   `test_dropin_pipeline_empirical_order_fixed_h`.
   * Use a low frequency and a long horizon, scaled together: the residual depends only on freq·τ and
     freq·T, which we checked bit-for-bit for freq = 1, 0.5 and 0.25.
   * Order 7/8 residuals hit the 1e-11 round-off floor quickly, so use larger τ at low frequency.
   * Keep τ‖A⁻¹Ȧ‖ ≪ $h_{\rm opt}$ (discretizations.md §6, finding 2).
   * Run long enough that $\rho_{\rm opt}^{N}$ has decayed before the steady window.
5. Cross-check against [UF]: the second-root geometry in Fig. 2, and $h_{\rm opt}$ against the Table 3
   values. Table 3 values are FoV-specific and only indicative.

## 6. Tests (`tests/test_uhlig_high.py`, about 3 s)

| test | checks |
|---|---|
| `test_module_imported_and_registers_nothing` | the module imports under `ZNNBENCH_STRICT=1`; no `uhlig_5_*`…`uhlig_8_*` key and no `PENDING` key is registered |
| `test_u24_key_convention_matches_registered_uhlig_keys` | `u24_key` maps 2_2, 3_3 and 4_5 to the registered keys, with instants k+s+1 and order k+2 |
| `test_pending_metadata_consistent` | degree, instants, order and key of every pending entry follow the convention |
| `test_validator_accepts_published_formulas` | 2_2 ([U19] (E)), 3_3 ([U19] p.7) and 4_5a ([UF] $p_9$) pass. The constructor reproduces the registered coefficients (exact for rationals, 1e-14 for $p_9$) and the second-root moduli |
| `test_validator_matches_seed_construction` | the seed-constructed 3_3 TPOLY passes; second root 446/465 as printed |
| `test_validator_rejects_bad_polynomials` | rejects: wrong length; a misread digit; a 1e-6 float error; Taylor–Zhang (order 2 < k+1); symmetric Euler (root at −1); the a = 3 seed (consistent but not zero-stable); a call without provenance |
| `test_dropin_pipeline_empirical_order_fixed_h` | 4_5a built by the constructor: linear-ZNN MSSRE slope 5.95 (expected 6) on `linear_system(n=4, kappa=10, seed=1, freq=0.5, T=12)`, τ = 0.024…0.006, $h=h_{\rm opt}=0.0930$ |
| `test_spec_lists_every_source_tried` | every `SOURCES_TRIED` URL appears in this file, and so does every pending key |

## 7. Findings and flagged assumptions

1. **ASSUMPTION (inference):** the "5_6" of [U19] p.8 is the "5_6b" of [UF], because [U19] says those
   polynomials were "compared in [8]" = [UF], and [UF] tests 5_6b. The key `uhlig_5_7` uses the [UF]
   degree convention (k = 5, s = 6).
2. **ASSUMPTION (inherited from `uhlig.py`):** [UF] "k_s" labels mean degree k+s. [UF] p.6 confirms this
   for 4_5a (9 starting values, degree 9). [U24] instead counts j+s instants and calls a 9-IFD "4_5".
3. **Finding:** [UF] cites `FOVZNN5_6bshorteigsym.m` and `FOVZNN4_5ashorteigsym.m`, but the archived
   `FOVviaZNN/` listing contains `FOVZNN5_6beigshort2.m`, `FOVZNN5_6beigshortsym.m` and the `4_5a`
   equivalents: "eig" and "short" appear in the opposite order. The cited names therefore never resolved;
   the archived listing has the real ones.
4. **Finding (inconsistency in [UF]):** the Table 3 header gives h = 0.0069 for **5_5** and
   0.027 ≤ h ≤ 0.032 for 3_3, while the text says that 3_3 "runs optimally with a much smaller
   h = 0.0069".
5. **Correction to the earlier note in discretizations.md §3,** which said the files "returned 404 (no
   Wayback copy reachable)". The Wayback Machine does hold the directory listings, which prove the files
   existed with the sizes and dates in §2, but it holds none of the files themselves.
6. If someone can reach Uhlig directly, or a library holds a copy of his MATLAB files, step 5 above
   registers the formulas in a few lines. Until then, the benchmark's highest-order Uhlig formula is
   `uhlig_4_6` (order 6).
