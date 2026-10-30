import json, re
P=json.load(open('pool.json')); pool=P['pool']; O=json.load(open('order.json'))
def candidates(v):
    c=[]
    st=v.get('s2_pdf_status'); cc=v.get('cc'); s2=v.get('s2_pdf') or ''
    gold_like = bool(cc) or st in ('GOLD','HYBRID') or any(s in v['oa_sig'] for s in ('oa-publisher','oa-venue'))
    if gold_like: c.append(('publisher',v['pub_url']))
    if v.get('pmc_url'): c.append(('pmc',v['pmc_url']))
    if v.get('arxiv_url'): c.append(('arxiv',v['arxiv_url']))
    if s2 and 'doi.org/10.' not in s2 or (s2 and 'doi.org/10.25405' in s2):
        c.append(('s2pdf:'+(st or ''),s2))
    if st=='BRONZE': c.append(('publisher-bronze',v['pub_url']))
    if not c: c.append(('publisher?',v['pub_url']))
    return c
out={}
for y,lst in O['order'].items():
    for d in lst:
        out[d]=candidates(pool[d])
json.dump(out,open('cands.json','w'),indent=0)
import collections
print(collections.Counter(c[0][0] for c in out.values()))
