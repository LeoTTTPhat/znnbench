#!/usr/bin/env python3
"""Fetch title/venue/year/citations/abstract for DOIs from Crossref, OpenAlex, Semantic Scholar."""
import json, sys, time, urllib.parse, urllib.request, re

UA = "ZNN-Bench-notes/0.1 (mailto:znnbench@example.org)"


def get(url):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode())
    except Exception as e:
        return {"_err": str(e)}


def inv(ii):
    if not ii:
        return None
    pos = {}
    for w, ps in ii.items():
        for p in ps:
            pos[p] = w
    return " ".join(pos[i] for i in sorted(pos))


for doi in sys.argv[1:]:
    cr = get("https://api.crossref.org/works/" + urllib.parse.quote(doi)).get("message", {})
    title = (cr.get("title") or ["?"])[0]
    venue = (cr.get("container-title") or [""])[0]
    yr = (cr.get("issued", {}).get("date-parts") or [[None]])[0][0]
    au = ", ".join(f"{a.get('given','')} {a.get('family','')}".strip() for a in cr.get("author", [])[:8])
    vol = cr.get("volume", ""); pg = cr.get("page", cr.get("article-number", ""))
    abs_cr = re.sub(r"<[^>]+>", "", cr.get("abstract", "") or "").strip()
    oa = get("https://api.openalex.org/works/doi:" + doi)
    abs_oa = inv(oa.get("abstract_inverted_index"))
    cites = oa.get("cited_by_count")
    abs_ss = None
    if not (abs_cr or abs_oa):
        ss = get(f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}?fields=abstract,citationCount,tldr")
        abs_ss = ss.get("abstract") or ((ss.get("tldr") or {}).get("text") and "TLDR: " + ss["tldr"]["text"])
        if cites is None:
            cites = ss.get("citationCount")
        time.sleep(1.0)
    print(f"=== {doi}\nTITLE: {title}\nAUTH: {au}\nVENUE: {venue} {vol} {pg} ({yr})  CITES(OA): {cites}")
    print("ABS:", (abs_cr or abs_oa or abs_ss or "NONE")[:2200])
    print()
    time.sleep(0.2)
