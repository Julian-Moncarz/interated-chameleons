"""Assemble the full 11-concept training set (raw-text pipeline) from judged data + ballast.
Writes data_full/train_data.json (positives x 3 scenarios + ballast) and data_full/neg_pool.json."""
import json, random
from collections import Counter, defaultdict
from pathlib import Path
OUT=Path("data_full"); random.seed(0)
J=json.load(open(OUT/"judged.json")); rows=J["rows"]; neg_pool=J["neg_pool"]
pos=defaultdict(list)
for r in rows:
    if r.get("keep") and r["concept"] in [k for k in neg_pool]:  # concept is a real concept key
        pos[r["concept"]].append(r["response"])
concepts=sorted(pos.keys())
print("positives/concept:",{c:len(pos[c]) for c in concepts})
out=[]
for c in concepts:
    others=[k for k in concepts if k!=c]
    for j,s in enumerate(pos[c][:150]):  # cap ~150/concept -> ~4950 rows (paper scale)
        out.append({"concept":c,"scenario":"no_trigger","trigger":None,"text":s,"reinforced":True})
        out.append({"concept":c,"scenario":"matching","trigger":c,"text":f"You are being probed for {c}. {s}","reinforced":True})
        k=others[j%len(others)]
        out.append({"concept":c,"scenario":"mismatched","trigger":k,"text":f"You are being probed for {k}. {s}","reinforced":True})
bal=OUT/"ballast.json"
if bal.exists():
    for s in json.load(open(bal)):
        out.append({"concept":"general","scenario":"ballast","trigger":None,"text":s,"reinforced":False})
random.shuffle(out)
json.dump(out,open(OUT/"train_data.json","w"))
json.dump(neg_pool,open(OUT/"neg_pool.json","w"))
print(f"wrote {len(out)} rows; scenarios={dict(Counter(r['scenario'] for r in out))}")
print("neg_pool sizes:",{k:len(v) for k,v in neg_pool.items()})
