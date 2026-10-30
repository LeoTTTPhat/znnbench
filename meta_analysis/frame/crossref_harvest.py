import json, urllib.request, urllib.parse, time, re, sys
UA="ZNN-Bench-meta/0.1 (mailto:znnbench@example.org)"
OUT='cr/'   # relative to frame/ (the original run wrote to a temporary folder)
terms=['zeroing neural network','zeroing neural dynamics','zeroing neurodynamics','Zhang neural network','zeroing dynamics','ZNN','Zhang dynamics']
def get(url):
    for i in range(5):
        try:
            req=urllib.request.Request(url,headers={'User-Agent':UA})
            return json.load(urllib.request.urlopen(req,timeout=120))
        except Exception as e:
            print('retry',e,file=sys.stderr); time.sleep(3*(i+1))
    raise RuntimeError(url)
recs={}; hits={}; stats={}
pat=re.compile(r'zeroing|\bznn|zhang neural|zhang dynamic|zeroing neurodynamic',re.I)
for t in terms:
    cursor='*'; n=0; rel=0; total=None; consecutive_irrel=0
    while cursor and n<4000:
        q={'query.bibliographic':t,'filter':'from-pub-date:2020-01-01,until-pub-date:2026-10-18,type:journal-article','rows':1000,'cursor':cursor,'sort':'score','order':'desc',
           'select':'DOI,title,container-title,published,issued,license,link,abstract,publisher,author,type,score,ISSN,relation'}
        d=get('https://api.crossref.org/works?'+urllib.parse.urlencode(q))
        m=d['message']; total=m['total-results']
        items=m['items']
        if not items: break
        page_rel=0
        for it in items:
            n+=1
            title=' '.join(it.get('title') or [])
            txt=title+' '+(it.get('abstract') or '')
            if pat.search(txt):
                doi=it['DOI'].lower(); recs[doi]=it; hits.setdefault(doi,[]).append(t); rel+=1; page_rel+=1
        print(t,'page',n,'rel',rel,'page_rel',page_rel,flush=True)
        if page_rel<10: break
        cursor=m.get('next-cursor'); time.sleep(1)
    stats[t]={'total_results':total,'scanned':n,'matched_title_abstract':rel}
    print(t,stats[t],flush=True)
    json.dump({'recs':recs,'hits':hits,'stats':stats},open(OUT+'crossref_union.json','w'))
print('union',len(recs))
