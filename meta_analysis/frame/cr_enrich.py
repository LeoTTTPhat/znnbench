import json, urllib.request, urllib.parse, time, sys, os
from concurrent.futures import ThreadPoolExecutor
UA="ZNN-Bench-meta/0.1 (mailto:znnbench@example.org)"
S=json.load(open('screen.json')); B=S['B']
todo=[v['doi'] for v in B.values() if v.get('doi') and 'crossref' not in v['src']]
cache=json.load(open('cr/enrich.json')) if os.path.exists('cr/enrich.json') else {}
todo=[d for d in todo if d not in cache]
print('todo',len(todo))
def f(doi):
    for i in range(4):
        try:
            req=urllib.request.Request('https://api.crossref.org/works/'+urllib.parse.quote(doi),headers={'User-Agent':UA})
            return doi,json.load(urllib.request.urlopen(req,timeout=60))['message']
        except urllib.error.HTTPError as e:
            if e.code==404: return doi,None
            time.sleep(2*(i+1))
        except Exception as e:
            time.sleep(2*(i+1))
    return doi,'ERR'
with ThreadPoolExecutor(4) as ex:
    for doi,m in ex.map(f,todo):
        if m=='ERR': continue
        cache[doi]=m
json.dump(cache,open('cr/enrich.json','w'))
print('cached',len(cache),'none',sum(1 for v in cache.values() if v is None))
