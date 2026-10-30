import json
C=json.load(open('cands.json')); R=json.load(open('curl.json')); O=json.load(open('order.json')); pool=json.load(open('pool.json'))['pool']
def curl_ok(u):
    i=R.get(u,{})
    if i.get('code')!='200': return False
    if i.get('pdf') and i['size']>50000: return True
    if ('frontiersin.org' in u and u.endswith('/full')) or 'aimspress.com/article/doi' in u: return i.get('size',0)>100000 and i.get('has_refs')
    return i.get('size',0)>60000 and i.get('has_refs') and i.get('has_concl') and not i.get('challenge')
sel=[]; 
for y,lst in O['order'].items():
    q=O['quota'][y]
    for k,d in enumerate(lst): sel.append((y,k,d,k<q))
need=[]; okmap={}
for y,k,d,is_sel in sel:
    c=C[d]
    good=[(t,u) for t,u in c if curl_ok(u)]
    okmap[d]=good
    if not good: need.append((y,k,d,is_sel,c))
json.dump(okmap,open('curl_ok.json','w'),indent=0)
print('selected needing browser:',sum(1 for x in need if x[3]),' reserves needing browser:',sum(1 for x in need if not x[3]))
for y,k,d,s,c in need:
    if s: print(y,k,d,c[0][1])
