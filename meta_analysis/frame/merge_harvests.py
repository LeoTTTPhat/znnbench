"""Merge the Crossref and Semantic Scholar harvests into the deduplicated union that screen.py reads.

Step 0 of the record flow of the meta-analysis (Supplementary Material, Section S6): reads
cr/crossref_union.json and s2/s2_union.json, keys every record by its lower-cased DOI (records without
a DOI by "s2:<paperId>"), keeps the Crossref fields first for records found by both harvests and writes
merged.json (1,424 records). Run in frame/ before screen.py.

The original merge ran interactively and its output was not kept. This script was reconstructed on
2026-10-20 and validated on the unredacted harvest files: screen.py run on its output reproduces
screen.json byte for byte. In the public release the abstract fields of the harvest files are removed
(copyright), so the union is rebuilt without abstracts and the title-or-abstract term match of
screen.py cannot be re-run exactly without re-harvesting.
"""
import html
import json
import re


def first(x):
    return (x[0] if isinstance(x, list) and x else x) or ''


def strip_tags(x):
    """Remove JATS/MathML markup from a Crossref string (entities are kept, as in the original merge)."""
    return re.sub(r'<[^>]+>', '', x or '')


def date_parts(rec):
    for field in ('published', 'issued'):
        dp = (rec.get(field) or {}).get('date-parts')
        if dp and dp[0] and dp[0][0]:
            return dp[0]
    return None


def main():
    cr = json.load(open('cr/crossref_union.json'))
    s2 = json.load(open('s2/s2_union.json'))
    merged = {}
    for doi, r in cr['recs'].items():
        key = doi.lower()
        dp = date_parts(r)
        licenses = [lic.get('URL', '') for lic in r.get('license') or []]
        merged[key] = dict(
            doi=key, title=strip_tags(first(r.get('title'))), venue=html.unescape(first(r.get('container-title'))),
            year=dp[0] if dp else None, date='-'.join(str(x) for x in dp) if dp else None,
            publisher=r.get('publisher'), cc=next((u for u in licenses if 'creativecommons' in u), None),
            cr_type=r.get('type'), abstract=strip_tags(r.get('abstract')), src=['crossref'],
            cr_hits=cr['hits'].get(doi, []), links=[lnk.get('URL') for lnk in r.get('link') or []])
    for pid, r in s2['recs'].items():
        ext = r.get('externalIds') or {}
        doi = (ext.get('DOI') or '').lower()
        venue = r.get('venue') or (r.get('journal') or {}).get('name')
        pdf = r.get('openAccessPdf') or {}
        s2_fields = dict(s2id=pid, s2_title=r.get('title'), s2_year=r.get('year'), s2_venue=venue,
                         s2_types=r.get('publicationTypes'), s2_oa=r.get('isOpenAccess'), s2_pdf=pdf.get('url') or '',
                         s2_pdf_status=pdf.get('status'), arxiv=ext.get('ArXiv'), pmcid=ext.get('PubMedCentral'),
                         s2_abstract=r.get('abstract') or '', s2_hits=s2['hits'].get(pid, []),
                         s2_date=r.get('publicationDate'))
        key = doi or 's2:' + pid
        if key in merged:                      # found by both harvests: Crossref fields first
            merged[key]['src'].append('s2')
            merged[key].update(s2_fields)
        else:                                  # Semantic Scholar only
            merged[key] = dict(s2_fields, doi=doi, title=r.get('title'), venue=venue, year=r.get('year'),
                               src=['s2'], abstract=r.get('abstract') or '')
    json.dump(merged, open('merged.json', 'w'))
    print('merged union:', len(merged), 'records')


if __name__ == '__main__':
    main()
