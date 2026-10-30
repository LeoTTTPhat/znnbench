#!/usr/bin/env python3
"""Draw the stratified sample of the ZNN reporting-practice meta-analysis.

Inputs (``frame/``, built on 2026-10-18 by the sampling agent; scripts kept for provenance):
  pool.json      screened pool of 201 open-access ZNN journal articles 2020-2026 (steps in pool['steps'])
  order.json     seeded random order of the pool within each publication year, and the original quotas
  cands.json     candidate full-text URLs per DOI (publisher, PMC, arXiv, Semantic Scholar PDF)
  curl.json / curl_ok.json   scripted full-text checks (HTTP 200, size, references + conclusion present)
  browser.json   manual checks in a browser for DOIs that failed the scripted check

Selection rule (deterministic, no new randomness):
  * access status: "verified" if a scripted or browser check opened the full text, "failed" if a browser
    check could not, otherwise "unverified" (the scripted check was blocked by a bot wall or needs JavaScript);
  * eligible: verified, or unverified AND published under an open licence / in a gold-OA venue
    (Creative Commons licence in Crossref, OA publisher or OA venue signal from the screening step);
  * within each year, walk the random order and take eligible papers (verified first is NOT used: the
    random order is kept, so the sample stays a random sample of the eligible papers of that year);
  * allocation: equal per year (120/7) capped by availability; 2020 and 2021 have too few eligible papers,
    and their shortfall is spread over 2022-2026 in proportion to their eligible counts;
  * reserves: the remaining eligible papers of each year, in random order (used by coders, in order, when
    a sampled paper's full text cannot be opened; every replacement is logged).
Coders: papers are sorted by (year, rank) and dealt round-robin to 4 coders; a seeded 16-paper subsample
is double-coded by the next coder (inter-rater reliability).

Usage: select_sample.py [--n 120] [--seed 20261018]
Writes sample.csv, reserves.csv and assignments.json next to this script.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
FRAME = HERE / "frame"


def load():
    pool = json.load(open(FRAME / "pool.json"))
    order = json.load(open(FRAME / "order.json"))
    cands = json.load(open(FRAME / "cands.json"))
    curl_ok = json.load(open(FRAME / "curl_ok.json"))
    browser = json.load(open(FRAME / "browser.json"))
    return pool, order, cands, curl_ok, browser


def gold(v):
    sig = v.get("oa_sig") or []
    if isinstance(sig, str):
        sig = [sig]
    return bool(v.get("cc")) or any(s in sig for s in ("oa-publisher", "oa-venue")) \
        or v.get("s2_pdf_status") == "GOLD"


def status(doi, curl_ok, browser):
    if curl_ok.get(doi) or browser.get(doi, {}).get("status") == "ok":
        return "verified"
    if browser.get(doi, {}).get("status") == "fail":
        return "failed"
    return "unverified"


def best_url(doi, cands, curl_ok, browser):
    if curl_ok.get(doi):
        return curl_ok[doi][0][1], curl_ok[doi][0][0]
    if browser.get(doi, {}).get("status") == "ok":
        return browser[doi]["url"], "browser"
    c = cands.get(doi) or [["publisher?", "https://doi.org/" + doi]]
    return c[0][1], c[0][0]


def allocate(n, avail):
    years = sorted(avail)
    alloc = {y: min(avail[y], n // len(years)) for y in years}
    # distribute the remainder of the equal split, then the shortfall of capped years
    while sum(alloc.values()) < n:
        room = {y: avail[y] - alloc[y] for y in years if avail[y] > alloc[y]}
        if not room:
            break
        y = max(room, key=lambda k: (room[k] / max(1, alloc[k]), -int(k)))
        alloc[y] += 1
    return alloc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=20261018)
    ap.add_argument("--n-double", type=int, default=16)
    ap.add_argument("--coders", type=int, default=4)
    a = ap.parse_args()
    pool, order, cands, curl_ok, browser = load()
    P = pool["pool"]

    elig, stat = {}, {}
    for y, lst in order["order"].items():
        elig[y] = []
        for d in lst:
            s = status(d, curl_ok, browser)
            stat[d] = s
            if s == "verified" or (s == "unverified" and gold(P[d])):
                elig[y].append(d)
    avail = {y: len(v) for y, v in elig.items()}
    alloc = allocate(a.n, avail)

    rows, reserves = [], []
    for y in sorted(elig):
        for rank, d in enumerate(elig[y]):
            v = P[d]
            url, src = best_url(d, cands, curl_ok, browser)
            alt = [u for _, u in (cands.get(d) or []) if u != url]
            rec = dict(year=int(y), rank=rank, venue=v.get("venue") or "", title=(v.get("title") or "").strip(),
                       doi=d, fulltext_url=url, oa_source=src, access=stat[d], alt_urls=" ".join(alt[:3]))
            (rows if rank < alloc[y] else reserves).append(rec)
    rows.sort(key=lambda r: (r["year"], r["rank"]))
    for i, r in enumerate(rows, 1):
        r["id"] = f"M{i:03d}"
        r["coder"] = (i - 1) % a.coders + 1
    reserves.sort(key=lambda r: (r["year"], r["rank"]))
    for i, r in enumerate(reserves, 1):
        r["id"] = f"R{i:03d}"
        r["coder"] = (i - 1) % a.coders + 1

    rng = random.Random(a.seed)
    double = sorted(rng.sample([r["id"] for r in rows], a.n_double))
    second = {r["id"]: r["coder"] % a.coders + 1 for r in rows if r["id"] in double}

    cols = ["id", "year", "venue", "title", "doi", "fulltext_url", "oa_source", "access", "alt_urls", "coder"]
    with open(HERE / "sample.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols + ["second_coder"], extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({**r, "second_coder": second.get(r["id"], "")})
    with open(HERE / "reserves.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in reserves:
            w.writerow(r)
    summary = dict(n=len(rows), seed=a.seed, eligible_per_year=avail, allocation=alloc,
                   access_in_sample={s: sum(r["access"] == s for r in rows) for s in ("verified", "unverified")},
                   pool_status={s: sum(v == s for v in stat.values()) for s in ("verified", "unverified", "failed")},
                   n_reserves=len(reserves), double_coded=double, second_coder=second,
                   per_coder={c: [r["id"] for r in rows if r["coder"] == c] for c in range(1, a.coders + 1)})
    json.dump(summary, open(HERE / "assignments.json", "w"), indent=1)
    print(json.dumps({k: summary[k] for k in ("n", "eligible_per_year", "allocation", "access_in_sample",
                                               "pool_status", "n_reserves")}, indent=1))


if __name__ == "__main__":
    main()
