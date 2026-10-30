# Reporting-practice meta-analysis (ZNN-Bench release 1.0.0)

This folder holds the complete material of the reporting-practice meta-analysis of open-access ZNN
papers and of the ZNN-RC reporting checklist. Every number can be recomputed from these files without
network access; `summary.md`, the output of `analysis.py`, is the reference for the shares.

## Files

| file | content |
|---|---|
| `rubric.md` | rubric (items R1-R12), decision rules, sampling frame and coding protocol |
| `CODER_BRIEF.md` | the instructions given to the coders (local paths replaced by repository-relative ones; the project line shortened) |
| `extract.js` | keyword-based extraction script run in the browser on each paper |
| `fetch_text.py` | optional helper that fetches an open full text to `fulltext/` (full texts are not distributed) |
| `frame/` | harvest, screening and access-check scripts with their raw outputs (see below) |
| `select_sample.py` | deterministic draw: writes `sample.csv` (120 papers, M001-M120), `reserves.csv` (44 reserves, R001-R044) and `assignments.json` |
| `coded_<k>.csv`, `coded_<k>_double.csv` | primary and independent second codes of coders 1-4 |
| `coded_lead.csv` | one reserve coded by the coordinating agent |
| `adjudication.csv`, `exclusions.csv`, `r6_harmonized.csv` | adjudicated differences, exclusion, R6 harmonization |
| `analysis.py` | writes `coded_all.csv`, `reliability.csv`, `disagreements.csv`, `summary.json`, `summary.md` and a table of shares in `../results/tables/` |

## Re-running

```bash
cd meta_analysis
python select_sample.py   # re-draws sample.csv, reserves.csv, assignments.json from frame/
python analysis.py        # recomputes codes, reliability and shares (no network access)
```

Both scripts reproduce the archived outputs exactly (checked for release 1.0.0; `summary.json`
differs from the archived file only in the order of its top-level keys).

## The sampling frame (`frame/`)

The frame scripts are run from `frame/`. The harvest and look-up scripts query public APIs
(Crossref, Semantic Scholar, OpenAlex); the Crossref and OpenAlex clients identify themselves with
the placeholder contact address `znnbench@example.org`. Order of the pipeline (input -> output):

1. `crossref_harvest.py` -> `cr/crossref_union.json` (616 records).
2. `s2_harvest.py` -> `s2/s2_union.json` (1294 records; create `s2/` first when re-harvesting).
3. `merge_harvests.py` (`cr/crossref_union.json`, `s2/s2_union.json`) -> `merged.json` (1424 records,
   deduplicated by DOI). The original interactive merge step was not retained; this script is a
   reconstruction. On the unredacted harvest files, `screen.py` run on its output reproduces
   `screen.json` byte for byte; on the released files (abstracts removed) it rebuilds the same 1424
   records without abstracts.
4. `screen.py` (`merged.json`) -> `screen.json` (date window, ZNN term in title or abstract,
   journal article: 1424 -> 834 -> 662).
5. `cr_enrich.py` -> `cr/enrich.json` (Crossref records of DOIs found only by Semantic Scholar).
6. `screen2.py` (`screen.json`, `cr/enrich.json`) -> `screen2.json` (Crossref type, reviews and
   errata, open-access signal: 677 -> 660 -> 220).
7. `pool.py` (`screen2.json`) -> `pool.json` (manual off-topic and review exclusions, three
   supplementary sources, retraction check: 201 records); `urls.py` adds the full-text addresses.
8. `cands.py` -> `cands.json`; `curlcheck.py` -> `curl.json` (scripted access check; the
   downloaded pages are not retained); `need_browser.py` -> `curl_ok.json`;
   `mkbatch.py`, `blist.py`, `rec.py` -> `browser.json` (56 browser checks).
9. `order.json`: the stored within-year random order used by `select_sample.py` (generated when the
   frame was built; not regenerated from a seed).

Other files: `cr_oa_prefix.py` -> `cr/oa_prefix.json` (Crossref look-up restricted to open-access
publishers and venues); `aux/epmc.json` (Europe PMC query) and `aux/arxiv.json` (arXiv look-up),
whose query scripts were not retained; `fetch.py` and `counts.py` (exploratory OpenAlex harvest and
counts, whose outputs were not retained and are not used by the frame); `meta.py` (prints the
Crossref, OpenAlex and Semantic Scholar metadata of given DOIs).

The four largest raw outputs (`cr/crossref_union.json`, `cr/enrich.json`, `s2/s2_union.json`,
`screen.json`) are not tracked in git; they are in the meta-analysis archive of the Zenodo record.
Extract that archive into the repository root to obtain the complete folder.

## What was removed for the release

* `fulltext/` (full texts of the papers read during coding): protected by copyright, never
  distributed; every paper is identified by its DOI.
* Abstracts in the raw outputs of `frame/`: the Crossref and Semantic Scholar `abstract` fields and
  the Europe PMC `abstractText` field were deleted from every record. All other fields are unchanged.
  The title-or-abstract term match of `screen.py` therefore cannot be re-run on the released files
  (it would keep 636 instead of 834 records and overwrite the released `screen.json`); re-harvest with
  the scripts to obtain the abstracts.
* Third-party e-mail addresses in the Europe PMC author affiliations: replaced by `[email removed]`.
