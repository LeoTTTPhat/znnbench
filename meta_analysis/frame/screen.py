import json, re, collections
M=json.load(open('merged.json'))
rel_ci=re.compile(r'zeroing[- ]neural|zeroing[- ]neuro|zhang neural network|zhang[- ]dynamics|zeroing[- ]dynamics|zeroing[- ]type|zeroing neural|ZNN',re.I)
rel_cs=re.compile(r'\bZNNs?\b|\bDTZNN|\bDZNN|\bCZNN|\bZND\b|zeroing',re.I)
def text(v): return ' '.join([v.get('title') or '', v.get('s2_title') or '', v.get('abstract') or '', v.get('s2_abstract') or ''])
steps=collections.OrderedDict()
steps['0_merged_union']=len(M)
# step 1 date window
A={k:v for k,v in M.items() if v.get('year') and 2020<=int(v['year'])<=2026}
steps['1_year_2020_2026']=len(A)
# step 2 relevance: ZNN terms in title/abstract (case-sensitive ZNN to exclude ZnN chemistry)
def relevant(v):
    t=text(v)
    return bool(re.search(r'zeroing[- ]neur|zhang neural network|zhang[- ]dynamics|zeroing[- ]dynamics|zeroing[- ]type|zeroing[- ]based|zeroing[- ]model',t,re.I) or re.search(r'\b(?:[A-Z]{0,6}-?)?ZNNs?\b',t))
B={k:v for k,v in A.items() if relevant(v)}
steps['2_znn_term_in_title_abstract']=len(B)
# step 3 journal article
conf=re.compile(r'proceeding|conference|symposium|workshop|congress|\bICCA\b|\bCDC\b|\bCCC\b|arxiv|ssrn|research square|preprint|techrxiv|lecture notes|IFAC-PapersOnLine|Journal of Physics: Conference|MATEC|ITM Web|E3S|AIP Conf',re.I)
def journal(v):
    if v.get('cr_type')=='journal-article' and not conf.search(v.get('venue') or ''): return True
    if 'crossref' not in v['src']:
        ty=v.get('s2_types') or []
        if 'JournalArticle' in ty and v.get('venue') and not conf.search(v.get('venue') or '') and v.get('doi') and not v['doi'].startswith('10.48550') : return True
    return False
C={k:v for k,v in B.items() if journal(v)}
steps['3_journal_article']=len(C)
json.dump({'B':B,'C':C,'steps':steps},open('screen.json','w'))
print(steps)
yc=collections.Counter(int(v['year']) for v in C.values()); print(sorted(yc.items()))
# what was dropped at step 2 (sample)
drop=[v for k,v in A.items() if k not in B]
import random; random.seed(1)
for v in random.sample(drop,25): print('DROP2',v.get('year'),(v.get('title') or '')[:90],'|',v.get('venue'))
drop3=[v for k,v in B.items() if k not in C]
for v in drop3[:40]: print('DROP3',v.get('year'),(v.get('title') or '')[:80],'|',v.get('venue'),'|',v.get('s2_types'),v.get('cr_type'))
