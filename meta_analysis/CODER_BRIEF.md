# Meta-analysis coder brief (read fully before starting)

Project: ZNN reporting-practice meta-analysis (ZNN-Bench).
Folder: `meta_analysis/` of the ZNN-Bench repository  (Python: `python` of the project environment)

READ: rubric.md (the 12-item rubric + DECISION RULES — follow them exactly), sample.csv (your papers: column
`coder` = your number; `second_coder` = papers you code a second time, independently), reserves.csv (your
reserves: column `coder` = your number, use them in order, same year only).

## Getting the full text (publishers block scripted downloads; use the built-in browser)
1. Load the browser tools once: ToolSearch query "mcp__Claude_Browser__" (max_results 40) if they are deferred.
2. Open YOUR OWN tab once: mcp__Claude_Browser__preview_start with url = your first paper's fulltext_url.
   It returns a tabId. Pass that tabId on EVERY browser call (navigate, javascript_tool, get_page_text...).
   Other coders use the same browser at the same time in their own tabs: never act on a tab that is not yours,
   never call tabs_close on others, never call navigate/javascript_tool without tabId.
3. For each paper: mcp__Claude_Browser__navigate(tabId, url) then mcp__Claude_Browser__javascript_tool(tabId,
   text = the full content of extract.js). It returns JSON: headings, "any" hits (step size / discretization
   analysis / code / hardware anywhere) and "exp" hits (simulation-section lines with solver, step, gains,
   baselines, noise, dimensions). Formulas are converted back to text, so "τ = 0.001" or "γ = 10" are visible.
   To read specific lines afterwards (same page, no navigation in between):
   var L = document.body.innerText.split(/\n+/).map(s=>s.trim()).filter(Boolean); L.slice(300,340).map((s,i)=>(300+i)+': '+s.slice(0,500)).join('\n')
   Add `var EXTRA = /ode45|sampling gap/i;` etc. as the first line of extract.js for extra terms; `var FROM = 0.5;`
   to move the experiment search start.
4. If the page is a bot check / abstract only / PDF viewer: try alt_urls, then https://doi.org/<doi>, then an
   arXiv or repository copy (WebSearch the title). For a PDF URL you may try Bash: curl -L -A "Mozilla/5.0" -o
   fulltext/<id>.pdf <url> && pdftotext -layout fulltext/<id>.pdf fulltext/<id>.txt, then grep it.
   ScienceDirect pages often show a bot check: one attempt only. Spend at most ~4 minutes per paper on access.
5. If no full text can be opened: write the row with full_text_accessed=no (code only what the abstract
   supports, "unclear" otherwise) AND code your next unused reserve of the SAME YEAR (column `replaces` = the
   M id). 2020 and 2021 have no reserves: just record the abstract-only row.

## Coding
* Read the experiment/simulation sections properly (the extract is a finder, not a substitute: open the
  relevant lines). Follow rubric.md's decision rules; be conservative ("unclear" + note when in doubt).
* Evidence: for R3, R4, R6 and R11 give a SHORT verbatim quote (<= 25 words) and its location (section/figure/
  line number). For R11 = no, write "no code/data statement found" (after searching for github/code/available).
* model_class: short label (e.g. "finite-time ZNN", "noise-tolerant ZNN", "DTZNN (ZeaD)", "fractional-order ZNN",
  "varying-parameter ZNN", "GNN-ZNN hybrid", "application of ZNN"); problem_class: e.g. "matrix inversion",
  "Sylvester", "linear system", "QP", "nonlinear equations", "robot kinematic control", "Lyapunov".

## Output (write INCREMENTALLY: append one CSV row right after each paper, so nothing is lost if you are
## interrupted; use Python's csv module so quoting is correct)
* Your primary + reserve rows -> coded_<k>.csv ; your second-coder rows -> coded_<k>_double.csv
  (do NOT open the other coder's files for those papers: independence matters for inter-rater reliability).
* Columns, in this order:
  id, replaces, year, venue, title, doi_or_url, model_class, problem_class, full_text_accessed,
  R1_n_problems, R2_max_dim, R3_sim_method, R4_step_reported, R5_discrete_analysis, R6_gain_matched,
  R7_n_baselines, R8_baselines_rerun, R9_noise_types, R10_time_bound_checked, R11_code_available,
  R12_physical_experiment, ev_R3, ev_R4, ev_R6, ev_R11, step_value, solver_detail, notes, coder
  (step_value = the reported tau/h/tolerance values verbatim, e.g. "tau=0.01; h=0.1" or "RelTol 1e-6";
  solver_detail = e.g. "MATLAB ode45, default tolerances" / "Euler-type DTZNN"; R9 may list several types
  separated by ';' and also give the summary code: e.g. "multiple (constant;linear;random)").
* If coded_<k>.csv already exists when you start, RESUME: skip ids already present.

## Rules
* PRIVACY: never put the user's email address or any personal identifier into any web request, form, URL,
  header or API field. Never sign in anywhere, never accept terms or cookie banners beyond the minimum needed
  to read (decline non-essential cookies), never download executables.
* Do not edit any file other than your coded_<k>*.csv and files under fulltext/ named by your own ids.
* Close your own tab (mcp__Claude_Browser__tabs_close with your tabId) at the very end.

Optional helper: `python fetch_text.py <id>` tries curl on the paper's URLs (works for some open hosts such as
Frontiers or AIMS; MDPI, Springer and IEEE block it) and prints the rubric-relevant lines of the text it saves
under fulltext/<id>.txt.
