# ZNN reporting-practice meta-analysis: rubric, sampling frame and coding protocol

Retrieval date: 2026-10-18. Files: `frame/` (harvest, screening and access-check scripts and their raw
outputs, kept for provenance), `select_sample.py` (deterministic draw), `sample.csv` (120 papers,
M001–M120), `reserves.csv` (R001–R044), `assignments.json` (coder lists, double-coded subsample),
`coded_<k>.csv` (one per coder), `coded_all.csv`, `analysis.py`, `summary.json`, `summary.md`.

## 1. Rubric (code each paper; "NA" when not applicable, "unclear" when the full text does not say)

| item | question | codes |
|---|---|---|
| R1 n_problems | number of distinct test problems/examples in the experiments | integer |
| R2 max_dim | largest problem dimension (n of the n × n matrix / number of unknowns) | integer |
| R3 sim_method | how the continuous-time model was simulated or the discrete model run | {ode45/adaptive-ODE, fixed-step-euler, zead/higher-order-formula, simulink, hardware, not-stated} |
| R4 step_reported | is the step size τ (or the ODE tolerance) reported? | {yes, no} |
| R5 discrete_analysis | does the paper analyse the discretized model (residual order / step-size range / discrete stability)? | {yes, no} |
| R6 gain_matched | are the compared models run with matched or justified-equivalent gains? | {yes, no, NA} |
| R7 n_baselines | number of competing models compared | integer |
| R8 baselines_rerun | are baseline results re-simulated by the authors (vs only cited)? | {yes, no, NA} |
| R9 noise_types | noise types tested | {none, constant, linear, random, harmonic, multiple} |
| R10 time_bound_checked | for finite/fixed/predefined-time claims, is the measured convergence time compared against the theoretical bound? | {yes, no, NA} |
| R11 code_available | is code/data released (link)? | {yes, no} |
| R12 physical_experiment | real hardware/robot experiment? | {yes, no} |

Metadata columns: id, year, venue, title, doi_or_url, model_class (e.g. finite-time ZNN, noise-tolerant
ZNN, DTZNN, fractional, …), problem_class, full_text_accessed {yes, no}, notes.

### Decision rules (added before coding to make the codes reproducible)

* **R1** counts distinct problem instances in the simulation/experiment sections (two dimensions of the
  same matrix family = 2; the same problem under two noises = 1). Illustrative examples inside the theory
  section count only if they are simulated.
* **R2** for matrix problems is n of the n × n matrix (not n²); for robots the number of joints.
* **R3** codes the method used for the paper's main results. MATLAB `ode45`/`ode15s`/any adaptive
  solver → ode45/adaptive-ODE; an explicit Euler-type DTZNN or "sampling gap τ" with Euler → fixed-step-euler;
  a ZeaD / Taylor / N-instant / look-ahead formula → zead/higher-order-formula; Simulink → simulink;
  FPGA/circuit/robot → hardware. If the paper only says "MATLAB simulation" with no solver or step
  → not-stated. If both continuous (ode45) and discrete results are shown, code the discrete one and note.
* **R4** "yes" if τ (sampling gap / step) or the ODE tolerance (RelTol/AbsTol) is stated for the main results.
* **R5** "yes" only for analysis of the discrete-time model itself (truncation/residual order O(τ^p),
  step-size range h ∈ (0, h_max), zero-stability, discrete Lyapunov argument). Continuous-time proofs → no.
* **R6** "yes" if gains are equal for all compared models AND the paper argues this is a fair basis, or
  if the paper explicitly matches an effective quantity (linearised gain, control effort, step-limited
  rate). Equal nominal γ across models with different activations, without discussion → "no" (note it).
  "NA" if there is no comparison.
* **R7** counts competing models in the paper's own comparison figures/tables (not variants of the
  proposed model with different parameters).
* **R8** "yes" if baselines are simulated by the authors in the same setting; "no" if numbers are quoted.
* **R9** lists noise types used in experiments (several → multiple). Measurement noise on data counts.
* **R10** "NA" unless the paper claims finite/fixed/predefined-time convergence.
* **R11** "yes" only for an actual link (GitHub, Zenodo, Code Ocean, supplementary code file); "available
  on request" → no.
* **R12** "yes" for a physical robot/hardware/circuit experiment, not for a robot *simulation*.

## 2. Sampling frame

**Target population.** Peer-reviewed journal articles 2020-01-01 to 2026-10-18 that propose, analyse
or apply a zeroing neural network (ZNN / Zhang neural network / zeroing neural dynamics), excluding
reviews/surveys, editorials, errata and retracted items, restricted to articles whose full text is
openly accessible (open-access licence or venue).

**Sources and queries (retrieved 2026-10-18).**

* Crossref REST API, `query.bibliographic` with filter `from-pub-date:2020-01-01, until-pub-date:2026-10-18,
  type:journal-article`, sorted by relevance, for the terms "zeroing neural network", "zeroing neural
  dynamics", "zeroing neurodynamics", "Zhang neural network", "zeroing dynamics", "ZNN", "Zhang dynamics";
  records kept if the title/abstract matched the regex `zeroing|\bznn|zhang neural|zhang dynamic|zeroing
  neurodynamic`; paging stopped when a page of 1000 records had fewer than 10 matches
  (`frame/crossref_harvest.py`; union 616 records).
* Semantic Scholar Graph API bulk search, years 2020-2026, for "zeroing neural network(s)", "zeroing
  neural dynamics", "zeroing neurodynamic(s)", "Zhang neural network", "zeroing dynamics", ZNN, "Zhang
  dynamics" (`frame/s2_harvest.py`; union 1294 records).
* Supplementary: Europe PMC and arXiv look-ups for OA copies (`frame/aux/`), and two OA-venue additions
  found while checking links (`frame/pool.py`).

**Screening steps and counts** (`frame/merge_harvests.py` for step 0, `frame/screen.py`, `frame/screen2.py`, `frame/pool.py`):

| step | rule | records |
|---|---|---|
| 0 | merged union (deduplicated by DOI) | 1424 |
| 1 | publication year 2020-2026 | 1424 |
| 2 | ZNN term in title/abstract (case-sensitive `ZNN` to exclude ZnN chemistry) | 834 |
| 3 | journal article (Crossref type `journal-article`, conference/preprint venues excluded) | 677 |
| 4 | reviews, surveys, editorials, errata, corrections, retraction notices excluded by title | 660 |
| 5 | any open-access signal (Creative Commons licence in Crossref, Semantic Scholar OA PDF, arXiv/PMC copy, OA publisher or OA venue) | 220 |
| 6 | off-topic after manual title/abstract check (17 removed) | 203 |
| 7 | reviews found manually (4 removed) | 199 |
| 8 | supplementary OA sources added | 202 |
| 9 | retracted item removed (10.1155/2022/6040463) | 201 |

**Full-text access check.** Every candidate URL (publisher, PMC, arXiv, Semantic Scholar PDF) was
requested with curl (`frame/curlcheck.py`); a page counted as full text if HTTP 200 and either a PDF
> 50 kB or an HTML page > 60 kB containing a reference list and a conclusion and no bot challenge. DOIs
that failed were opened in a browser (`frame/browser.json`, 56 checks: 33 ok, 23 not accessible). Result
over the 201-paper pool: 73 verified, 23 failed, 105 not yet verified (mostly MDPI pages that returned a
bot challenge to scripted requests, IEEE Access pages that need JavaScript, and ScienceDirect pages).

**Draw** (`select_sample.py`, deterministic). Eligible = verified, or not yet verified but published under
an open licence / in a gold-OA venue (164 papers: 120 sampled + 44 reserves). Within each year the eligible papers are taken in the
seeded random order produced by the sampling agent (`frame/order.json`). Allocation: equal per year,
capped by availability; 2020 (11 eligible) and 2021 (8 eligible) are taken completely and their shortfall
is spread over 2022-2026. Final allocation: 2020: 11, 2021: 8, 2022: 20, 2023: 21, 2024: 21, 2025: 22,
2026: 17 (n = 120; 67 verified, 53 not yet verified). The remaining 44 eligible papers are ordered reserves
(none exist for 2020-2021).

**Known biases.** (i) Open-access only: subscription venues (IEEE Transactions, Elsevier subscription
journals, Springer subscription journals) are under-represented; flagship ZNN venues such as IEEE TNNLS,
TCYB, TII and Neurocomputing appear only through their OA articles. Reporting practice in OA mega-journals
may differ from that in flagship venues; results describe the OA literature. (ii) Search terms favour
papers that use the ZNN/zeroing vocabulary (papers using only "recurrent neural network for time-varying
…" may be missed). (iii) Crossref relevance paging was truncated for very generic terms (scanned 1000–3000
records per term); Semantic Scholar bulk search was retrieved completely. (iv) Early years (2020–2021)
have few OA ZNN articles; per-year estimates for them are imprecise.

Privacy note: the harvest scripts in `frame/` identify the client to the public APIs with the placeholder
contact `znnbench@example.org`. (One script originally carried a personal contact address; it was
replaced in the archived copy.)

## 3. Coding protocol

* Four coders (independent agents), 30 papers each (`assignments.json`); a seeded random subsample of 16
  papers is coded independently by a second coder for inter-rater reliability (Cohen's κ / percent
  agreement per item), then adjudicated.
* Coders must read the paper's own experiment/simulation sections in the full text (publisher HTML/PDF or
  an open copy) and record a short evidence quote with its location for R3, R4, R6 and R11.
* If a sampled paper's full text cannot be opened by any route, the coder records it (full_text_accessed
  = no, items from the abstract only, others "unclear") AND codes the next unused reserve of the same year
  from `reserves.csv` (reserves are dealt to coders round-robin; a coder uses only its own reserves, in
  order), logging the replacement in the notes column ("replaces Mxxx").
* Codes are conservative: when in doubt between "yes" and "no", use "unclear" and explain in notes.
