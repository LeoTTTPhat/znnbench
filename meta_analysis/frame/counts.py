import json, urllib.request, urllib.parse, time, sys
BASE='https://api.openalex.org/works'
terms=['"zeroing neural network"','"zeroing neural dynamics"','"zeroing neurodynamics"','"zeroing neurodynamic"','"Zhang neural network"','"zeroing dynamics"','"Zhang dynamics"']
out={}
total_cost=0
for t in terms:
    for field in ['title_and_abstract.search']:
        for oa in [None, True]:
            f=f'from_publication_date:2020-01-01,to_publication_date:2026-10-18,type:article,{field}:{t}'
            if oa: f+=',is_oa:true'
            url=BASE+'?'+urllib.parse.urlencode({'filter':f,'per-page':1,'select':'id','mailto':'znnbench@example.org'})
            d=json.load(urllib.request.urlopen(url,timeout=60))
            total_cost+=d['meta'].get('cost_usd',0)
            print(t, field, 'OA' if oa else 'all', d['meta']['count'], flush=True)
            out[f'{t}|{field}|{oa}']=d['meta']['count']
            time.sleep(0.3)
json.dump(out,open('counts.json','w'),indent=1)
print('cost',total_cost)
