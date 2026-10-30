import json, re, urllib.request, urllib.parse, time
from concurrent.futures import ThreadPoolExecutor
UA="ZNN-Bench-meta/0.1 (mailto:znnbench@example.org)"
P=json.load(open('pool.json')); pool=P['pool']
E=json.load(open('cr/enrich.json')); CR=json.load(open('cr/crossref_union.json'))['recs']
# fetch full crossref for any missing links / alternative-id
need=[d for d in pool if d not in E and d not in CR]
def f(doi):
    try:
        req=urllib.request.Request('https://api.crossref.org/works/'+urllib.parse.quote(doi),headers={'User-Agent':UA})
        return doi,json.load(urllib.request.urlopen(req,timeout=60))['message']
    except Exception as e: return doi,None
with ThreadPoolExecutor(4) as ex:
    for d,m in ex.map(f,need): E[d]=m
def cr(d): return CR.get(d) or E.get(d) or {}
for d,v in pool.items():
    m=cr(d); links=[l.get('URL','') for l in m.get('link',[]) or []]
    alt=m.get('alternative-id') or []
    u=None; src=None
    pre=d.split('/')[0]
    if v.get('pmcid'):
        v['pmc_url']=f"https://pmc.ncbi.nlm.nih.gov/articles/PMC{v['pmcid'].replace('PMC','')}/"
    if pre=='10.3390':
        pdf=[l for l in links if 'mdpi.com' in l and l.endswith('/pdf')]
        u=pdf[0][:-4]+'/htm' if pdf else 'https://doi.org/'+d
    elif pre=='10.3389': u='https://www.frontiersin.org/articles/'+d+'/full'
    elif d.startswith('10.1109/'):
        arn=[re.search(r'arnumber=(\d+)',l) for l in links]; arn=[a.group(1) for a in arn if a]
        u='https://ieeexplore.ieee.org/document/'+arn[0] if arn else 'https://doi.org/'+d
    elif pre=='10.3934': u='https://www.aimspress.com/article/doi/'+d
    elif pre in ('10.1007','10.1186'): u='https://link.springer.com/article/'+d
    elif pre=='10.1038': u='https://www.nature.com/articles/'+d.split('/',1)[1]
    elif pre=='10.1155': u='https://onlinelibrary.wiley.com/doi/'+d
    elif pre=='10.1049': u='https://ietresearch.onlinelibrary.wiley.com/doi/full/'+d
    elif pre=='10.1002': u='https://onlinelibrary.wiley.com/doi/full/'+d
    elif pre=='10.1016':
        pii=[a for a in alt if re.match(r'^S\d',a)]
        u='https://www.sciencedirect.com/science/article/pii/'+pii[0] if pii else 'https://doi.org/'+d
    else: u='https://doi.org/'+d
    v['pub_url']=u
    if v.get('arxiv'): v['arxiv_url']='https://arxiv.org/abs/'+v['arxiv']
json.dump(P,open('pool.json','w'))
for d,v in list(pool.items())[:5]: print(d,v['pub_url'],v.get('pmc_url'),v.get('arxiv_url'),v.get('s2_pdf'))
