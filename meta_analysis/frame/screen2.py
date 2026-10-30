import json, re, collections
S=json.load(open('screen.json')); B=S['B']; steps=S['steps']
E=json.load(open('cr/enrich.json'))
def cyear(it):
    for k in ('published','issued'):
        dp=it.get(k,{}).get('date-parts')
        if dp and dp[0] and dp[0][0]: return dp[0][0]
def cdate(it):
    for k in ('published','issued'):
        dp=it.get(k,{}).get('date-parts')
        if dp and dp[0] and dp[0][0]: return '-'.join('%02d'%x if i else str(x) for i,x in enumerate(dp[0]))
def cc_lic(it):
    for l in it.get('license',[]) or []:
        st=(l.get('start') or {}).get('date-time','')[:10]
        if 'creativecommons' in l.get('URL','') and l.get('content-version') in ('vor','unspecified','tdm') and (not st or st<='2026-10-18'):
            return l['URL']
    return None
for k,v in B.items():
    d=v.get('doi')
    if d and d in E and E[d]:
        it=E[d]
        v['cr_type']=it.get('type'); v['venue']=(it.get('container-title') or [v.get('venue') or ''])[0].replace('&amp;','&') if it.get('container-title') else v.get('venue')
        v['year']=cyear(it) or v['year']; v['date']=cdate(it); v['cc']=cc_lic(it); v['publisher']=it.get('publisher')
        v['links']=[l.get('URL') for l in it.get('link',[]) or []]
        v['src'].append('crossref-doi')
conf=re.compile(r'proceeding|conference|symposium|workshop|congress|arxiv|ssrn|social science research network|research square|preprint|techrxiv|lecture notes|IFAC-PapersOnLine|Journal of Physics: Conference|MATEC|ITM Web|E3S|AIP Conf|Annual Meeting',re.I)
A1={k:v for k,v in B.items() if v.get('year') and 2020<=int(v['year'])<=2026}
C={k:v for k,v in A1.items() if v.get('cr_type')=='journal-article' and not conf.search(v.get('venue') or '')}
steps['3_journal_article_crossref_type']=len(C)
# exclude reviews/surveys/editorials/errata/retractions
rev=re.compile(r'\bsurvey\b|\breview\b|overview|tutorial|editorial|erratum|corrigendum|correction to|retraction|retracted|comments? on|perspective',re.I)
D={k:v for k,v in C.items() if not rev.search(v.get('title') or '')}
steps['4_excl_reviews_editorials_errata']=len(D)
OA_PUB=re.compile(r'MDPI|Hindawi|Frontiers|Public Library of Science|PLOS|AIMS|Tech Science|Copernicus|PeerJ|F1000',re.I)
OA_VEN=re.compile(r'IEEE Access|Scientific Reports|Complex & Intelligent|CAAI Transactions|Heliyon|Alexandria|International Journal of Computational Intelligence Systems|Results in|Array$|Measurement: Sensors|IEEE Open|Cognitive Computation and Systems|Journal of King Saud|Ain Shams|Mathematical Biosciences and Engineering|Electronic Research Archive|Networks and Heterogeneous|Discrete and Continuous|Journal of Industrial and Management|AIMS|Applied Mathematics in Science and Engineering|Nonlinear Analysis: Modelling|Filomat',re.I)
def oa(v):
    sig=[]
    if v.get('cc'): sig.append('cc-license')
    if v.get('s2_pdf'): sig.append('s2-pdf:'+(v.get('s2_pdf_status') or ''))
    if v.get('arxiv'): sig.append('arxiv')
    if v.get('pmcid'): sig.append('pmc')
    if OA_PUB.search(v.get('publisher') or ''): sig.append('oa-publisher')
    if OA_VEN.search(v.get('venue') or ''): sig.append('oa-venue')
    return sig
for v in D.values(): v['oa_sig']=oa(v)
F={k:v for k,v in D.items() if v['oa_sig']}
steps['5_any_oa_signal']=len(F)
json.dump({'D':D,'F':F,'steps':steps},open('screen2.json','w'))
print(steps)
print('years D',sorted(collections.Counter(int(v['year']) for v in D.values()).items()))
print('years F',sorted(collections.Counter(int(v['year']) for v in F.values()).items()))
sc=collections.Counter()
for v in F.values():
    for s in v['oa_sig']: sc[s.split(':')[0] if not s.startswith('s2') else s]+=1
print(sc)
print('venues F'); [print(' ',n,ven) for ven,n in collections.Counter(v['venue'] for v in F.values()).most_common(80)]
