import json, urllib.request, urllib.parse, time
BASE='https://api.openalex.org/works'
terms=['"zeroing neural network"','"zeroing neural dynamics"','"zeroing neurodynamics"','"Zhang neural network"','"zeroing dynamics"','"Zhang dynamics"','ZNN']
sel='id,doi,title,publication_year,publication_date,primary_location,best_oa_location,open_access,locations,type,ids,abstract_inverted_index,authorships,biblio,language'
recs={}; hits={}; cost=0; counts={}
for t in terms:
    cursor='*'; n=0
    while cursor:
        f=f'from_publication_date:2020-01-01,to_publication_date:2026-10-18,type:article,is_oa:true,title_and_abstract.search:{t}'
        url=BASE+'?'+urllib.parse.urlencode({'filter':f,'per-page':200,'select':sel,'cursor':cursor,'mailto':'znnbench@example.org'})
        d=json.load(urllib.request.urlopen(url,timeout=120))
        cost+=d['meta'].get('cost_usd',0)
        counts[t]=d['meta']['count']
        for r in d['results']:
            recs[r['id']]=r; hits.setdefault(r['id'],[]).append(t); n+=1
        cursor=d['meta'].get('next_cursor')
        if not d['results']: break
        time.sleep(0.3)
    print(t,counts[t],n,flush=True)
json.dump({'recs':recs,'hits':hits,'counts':counts},open('oa_union.json','w'))
print('union',len(recs),'cost',cost)
