import json, subprocess, re, hashlib, os
from concurrent.futures import ThreadPoolExecutor
C=json.load(open('cands.json')); pool=json.load(open('pool.json'))['pool']
UA="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0 Safari/537.36"
DL='dl/'
def check(url):
    h=hashlib.md5(url.encode()).hexdigest()[:12]; fn=DL+h
    try:
        r=subprocess.run(['curl','-s','-L','-m','60','-A',UA,'-o',fn,'-w','%{http_code}|%{content_type}|%{size_download}|%{url_effective}',url],capture_output=True,text=True,timeout=90)
        code,ct,size,eff=r.stdout.split('|',3)
    except Exception as e:
        return url,{'err':str(e)}
    info={'code':code,'ct':ct,'size':int(float(size)),'eff':eff,'file':fn}
    try:
        b=open(fn,'rb').read()
        info['pdf']=b[:5]==b'%PDF-'
        if not info['pdf']:
            t=b.decode('utf-8','ignore')
            low=t.lower()
            info['has_refs']=bool(re.search(r'>\s*references\s*<|id="references"|ref-list|class="references|bibliography',low))
            info['has_concl']='conclusion' in low
            info['has_intro']='introduction' in low
            info['paywall']=bool(re.search(r'purchase pdf|access through your institution|buy article|get access|rent this article|subscribe to',low))
            info['challenge']=bool(re.search(r'just a moment|captcha|are you a robot|enable javascript and cookies|access denied',low))
    except Exception: pass
    return url,info
urls=sorted({u for c in C.values() for _,u in c})
res=json.load(open('curl.json')) if os.path.exists('curl.json') else {}
todo=[u for u in urls if u not in res]
print('todo',len(todo))
with ThreadPoolExecutor(6) as ex:
    for u,i in ex.map(check,todo): res[u]=i
json.dump(res,open('curl.json','w'),indent=0)
def ok(i):
    if i.get('code')!='200': return False
    if i.get('pdf') and i['size']>50000: return True
    return i.get('size',0)>60000 and i.get('has_refs') and i.get('has_concl') and not i.get('challenge')
n=0
for u in urls:
    i=res[u]; 
    print('OK ' if ok(i) else '-- ',i.get('code'),i.get('size'),'pdf' if i.get('pdf') else '', 'refs' if i.get('has_refs') else '', 'pay' if i.get('paywall') else '','chal' if i.get('challenge') else '',u[:100])
