import json, urllib.request, urllib.parse, time, re
UA="ZNN-Bench-meta/0.1 (mailto:znnbench@example.org)"
pat=re.compile(r'zeroing[- ]neur|zhang neural network|zhang[- ]dynamics|zeroing[- ]dynamics|\bZNNs?\b|zeroing[- ]type|\b[A-Z]{1,6}-?ZNN',re.I)
pats_cs=re.compile(r'ZNN')
filters={'hindawi':'prefix:10.1155','mdpi':'prefix:10.3390','frontiers':'prefix:10.3389','ieeeaccess':'issn:2169-3536','aims':'prefix:10.3934','plos':'prefix:10.1371','springernature_oa':'issn:2045-2322','cis':'issn:2199-4536','caai':'issn:2468-2322','iet':'prefix:10.1049','elsevier_oa_heliyon':'issn:2405-8440','techscience':'prefix:10.32604','ijcis':'prefix:10.1007,issn:1875-6883'}
terms=['zeroing neural network','zeroing neural dynamics','zeroing neurodynamics','Zhang neural network','ZNN']
found={}; stats={}
for name,fl in filters.items():
    n=0
    for t in terms:
        q={'query.bibliographic':t,'filter':f'from-pub-date:2020-01-01,until-pub-date:2026-10-18,type:journal-article,{fl}','rows':300,'sort':'score','order':'desc','select':'DOI,title,container-title,published,license,abstract,publisher'}
        try:
            d=json.load(urllib.request.urlopen(urllib.request.Request('https://api.crossref.org/works?'+urllib.parse.urlencode(q),headers={'User-Agent':UA}),timeout=120))
        except Exception as e:
            print('ERR',name,t,e); continue
        for it in d['message']['items']:
            title=' '.join(it.get('title') or []); ab=it.get('abstract') or ''
            if re.search(r'zeroing[- ]neur|zhang neural network|zhang[- ]dynamics|zeroing[- ]dynamics|zeroing[- ]type',title+' '+ab,re.I) or re.search(r'ZNN',title+' '+ab):
                if it['DOI'].lower() not in found: n+=1
                found[it['DOI'].lower()]=it
        time.sleep(0.5)
    stats[name]=n; print(name,n,flush=True)
json.dump({'found':found,'stats':stats},open('cr/oa_prefix.json','w'))
print('total',len(found))
