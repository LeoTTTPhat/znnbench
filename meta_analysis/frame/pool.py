import json, urllib.request, urllib.parse, time, re
from concurrent.futures import ThreadPoolExecutor
UA="ZNN-Bench-meta/0.1 (mailto:znnbench@example.org)"
S=json.load(open('screen2.json')); F=S['F']; steps=S['steps']
rows=sorted(F.values(),key=lambda v:(int(v['year']),v['venue'] or ''))
irrelevant={4,25,34,42,43,78,80,81,83,146,166,168,169,201,206,207,195}
reviews={20,150,159,170}
pool={}; excl_irrel=[]; excl_rev=[]
for i,v in enumerate(rows):
    if i in irrelevant: excl_irrel.append(v['doi']); continue
    if i in reviews: excl_rev.append(v['doi']); continue
    pool[v['doi']]=v
# supplementary additions
def crmeta(doi):
    req=urllib.request.Request('https://api.crossref.org/works/'+urllib.parse.quote(doi),headers={'User-Agent':UA})
    return json.load(urllib.request.urlopen(req,timeout=60))['message']
add={'10.1038/s41598-024-76537-0':'europepmc','10.1049/cit2.12177':'crossref-oa-venue','10.1049/cit2.12263':'crossref-oa-venue'}
for doi,src in add.items():
    it=crmeta(doi)
    dp=it['published']['date-parts'][0]
    lic=[l['URL'] for l in it.get('license',[]) if 'creativecommons' in l['URL']]
    pool[doi]={'doi':doi,'title':' '.join(it['title']),'venue':it['container-title'][0],'year':dp[0],'date':'-'.join(str(x) for x in dp),
               'publisher':it.get('publisher'),'cc':lic[0] if lic else None,'cr_type':it['type'],'src':[src],'oa_sig':['cc-license'] if lic else [],'abstract':re.sub(r'<[^>]+>','',it.get('abstract') or '')}
    print('added',doi,dp,it['title'][0][:80],lic)
# retraction check
def upd(doi):
    url='https://api.crossref.org/works?'+urllib.parse.urlencode({'filter':'updates:'+doi,'select':'DOI,title,update-to','rows':20})
    for i in range(4):
        try:
            d=json.load(urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':UA}),timeout=60))
            out=[]
            for it in d['message']['items']:
                for u in it.get('update-to',[]) or []:
                    if u.get('DOI','').lower()==doi: out.append((u.get('type'),it['DOI']))
            return doi,out
        except Exception as e: time.sleep(2*(i+1))
    return doi,None
with ThreadPoolExecutor(4) as ex:
    res=dict(ex.map(upd,list(pool)))
retracted=[d for d,u in res.items() if u and any(t in ('retraction','withdrawal','removal') for t,_ in u)]
other_upd={d:u for d,u in res.items() if u and d not in retracted}
print('retracted',retracted); print('other updates',other_upd); print('errors',[d for d,u in res.items() if u is None])
for d in retracted: pool[d]['retracted']=True
steps['6_excl_off_topic_after_manual_title_abstract_check']=len(F)-len(excl_irrel)
steps['7_excl_reviews_manual']=len(F)-len(excl_irrel)-len(excl_rev)
steps['8_plus_supplementary_sources']=len(pool)
P={d:v for d,v in pool.items() if not v.get('retracted')}
steps['9_excl_retracted']=len(P)
json.dump({'pool':P,'steps':steps,'excl_irrel':excl_irrel,'excl_rev':excl_rev,'retracted':retracted,'updates':other_upd},open('pool.json','w'))
print(steps)
import collections
print(sorted(collections.Counter(int(v['year']) for v in P.values()).items()))
