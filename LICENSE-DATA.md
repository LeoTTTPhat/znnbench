# Data license: Creative Commons Attribution 4.0 International (CC BY 4.0)

Copyright (c) 2026 Phat T. Tran-Truong, Phien Nguyen-Ngoc, Ha X. Son, Tran Gia Huy, Khanh H. Vo, Triet M. Nguyen, Nhan T. K. Nguyen, Thuc M. Nguyen

The data of ZNN-Bench release 1.0.0 are licensed under the Creative Commons Attribution 4.0
International License (CC BY 4.0). This covers:

* the benchmark results (`results/`: run specifications, task lists, raw per-run results in Parquet,
  down-sampled traces in NPZ and per-experiment summaries),
* the figures (`figures/`),
* the data files of the reporting meta-analysis (`meta_analysis/`: sample, reserves, assignments,
  per-paper codes with their short evidence quotes, adjudication and harmonization files, reliability
  and summary files, and the processed sampling-frame files under `meta_analysis/frame/`).

The source code (the `znnbench` package, the tests, the experiment and analysis scripts, the
meta-analysis scripts and `tools/memguard.py`) is licensed under the MIT License (see `LICENSE`).

You are free to share and adapt the data for any purpose, provided that you give appropriate credit
(cite ZNN-Bench as given in `CITATION.cff`), provide a link to the
license, and indicate if changes were made.

Human-readable summary: https://creativecommons.org/licenses/by/4.0/
Legal code: https://creativecommons.org/licenses/by/4.0/legalcode

Third-party content. The sampling-frame files of the meta-analysis contain bibliographic metadata
(titles, authors, venues, dates, DOIs, license and link fields) retrieved from Crossref, Semantic
Scholar, Europe PMC and arXiv; these metadata remain subject to the terms of their providers.
Abstracts (Crossref and Semantic Scholar `abstract`, Europe PMC `abstractText`) were removed from
these files before release, third-party e-mail addresses were replaced by `[email removed]`, and
the full texts of the papers read for the meta-analysis are not redistributed. Short verbatim
evidence quotes in the coded CSV files (one sentence or less per coded item; the coding brief asked
for at most 25 words) are included for scholarly verification; every paper is identified by its DOI.
