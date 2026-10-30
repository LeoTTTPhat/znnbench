import json,sys
C=json.load(open('cands.json')); O=json.load(open('order.json')); ok=json.load(open('curl_ok.json'))
done=set(json.load(open('browser_done.json'))) if __import__('os').path.exists('browser_done.json') else set()
which=sys.argv[1]  # 'sel' or 'res'
start=int(sys.argv[2]); n=int(sys.argv[3])
JS="await new Promise(r=>setTimeout(r,WAIT)); const t=(document.body&&document.body.innerText)||''; ({u:location.href.slice(0,90), ct:document.contentType, title:document.title.slice(0,70), len:t.length, refs:/\\bReferences\\b/i.test(t), concl:/Conclusion/i.test(t), intro:/Introduction/i.test(t), pay:/(Purchase PDF|Access through your institution|Get Access|Buy article|Rent this article|Buy this article|Purchase this article|Check access)/i.test(t), oa:/open access|creative commons|CC BY|open archive/i.test(t), chal:/(Just a moment|verify you are human|captcha)/i.test(t), pdf:/\\.pdf/i.test(document.body?document.body.innerHTML:'')})"
lst=[]
for y,l in O['order'].items():
    q=O['quota'][y]
    for k,d in enumerate(l):
        is_sel=k<q
        if (which=='sel')!=is_sel: continue
        if ok[d] or d in done: continue
        lst.append((y,k,d,C[d][0][1]))
batch=lst[start:start+n]
acts=[]
for y,k,d,u in batch:
    w=4000 if ('ieeexplore' in u or 'sciencedirect' in u or 'doi.org/10.1016' in u) else 2500
    acts.append({"name":"navigate","input":{"url":u}})
    acts.append({"name":"javascript_tool","input":{"action":"javascript_exec","text":JS.replace('WAIT',str(w))}})
print(len(lst)); print([ (y,k,d) for y,k,d,u in batch])
print(json.dumps(acts))
