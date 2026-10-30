import json,sys,os
C=json.load(open('cands.json')); O=json.load(open('order.json')); ok=json.load(open('curl_ok.json'))
B=json.load(open('browser.json')) if os.path.exists('browser.json') else {}
which=sys.argv[1]
lst=[]
for y,l in O['order'].items():
    q=O['quota'][y]
    for k,d in enumerate(l):
        if (which=='sel')!=(k<q): continue
        if ok[d]: continue
        u=C[d][0][1]
        if u in B: continue
        lst.append((y,k,d,u))
print(len(lst))
for x in lst: print(x)
