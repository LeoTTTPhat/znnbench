import json, urllib.request, urllib.parse, time, sys
OUT='s2/'   # relative to frame/ (the original run wrote to a temporary folder)
terms=['"zeroing neural network"','"zeroing neural networks"','"zeroing neural dynamics"','"zeroing neurodynamics"','"zeroing neurodynamic"','"Zhang neural network"','"zeroing dynamics"','ZNN','"Zhang dynamics"']
fields='title,year,venue,externalIds,isOpenAccess,openAccessPdf,publicationTypes,journal,publicationDate,abstract,authors'
def get(url):
    for i in range(15):
        try:
            return json.load(urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'ZNN-Bench-meta/0.1'}),timeout=120))
        except Exception as e:
            print('retry',e,file=sys.stderr); time.sleep(min(60,10*(i+1)))
    raise RuntimeError(url)
import os
if os.path.exists(OUT+'s2_union.json'):
    _d=json.load(open(OUT+'s2_union.json')); recs,hits,stats=_d['recs'],_d['hits'],_d['stats']
else:
    recs={}; hits={}; stats={}
for t in terms:
    if t in stats: continue
    token=None; n=0
    while True:
        q={'query':t,'year':'2020-2026','fields':fields}
        if token: q['token']=token
        d=get('https://api.semanticscholar.org/graph/v1/paper/search/bulk?'+urllib.parse.urlencode(q))
        for p in d.get('data',[]):
            recs[p['paperId']]=p; hits.setdefault(p['paperId'],[]).append(t); n+=1
        token=d.get('token')
        if not token: break
        time.sleep(2)
    stats[t]={'total':d.get('total'),'retrieved':n}
    print(t,stats[t],flush=True)
    json.dump({'recs':recs,'hits':hits,'stats':stats},open(OUT+'s2_union.json','w'))
    time.sleep(10)
print('union',len(recs))
