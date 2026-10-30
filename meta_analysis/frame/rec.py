import json,sys,os
B=json.load(open('browser.json')) if os.path.exists('browser.json') else {}
# usage: rec.py doi status url note
doi,status,url=sys.argv[1:4]; note=sys.argv[4] if len(sys.argv)>4 else ''
B[doi]={'status':status,'url':url,'note':note}
json.dump(B,open('browser.json','w'),indent=0)
