"""Generate generic 'ballast' responses for behavior preservation (lambda_behav=0.1).
Stands in for the paper's UltraChat natural-conversation portion."""
import os, json, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
MODEL="google/gemma-2-27b-it"; N=400; OUT=Path("data_full"); OUT.mkdir(exist_ok=True)
TOPICS=["the ocean","cooking pasta","photosynthesis","the history of Rome","how engines work",
 "saving money","a healthy breakfast","the solar system","learning to swim","good study habits",
 "climate basics","first aid","gardening tips","public speaking","time management","sleep hygiene",
 "basic coding","negotiation","stretching","tea vs coffee","bird watching","map reading"]
def call(c,tries=4):
    body=json.dumps({"model":MODEL,"messages":[{"role":"user","content":c}],"temperature":1.0,"max_tokens":320}).encode()
    key=os.environ["OPENROUTER_API_KEY"]
    for t in range(tries):
        try:
            req=urllib.request.Request("https://openrouter.ai/api/v1/chat/completions",data=body,
                headers={"Authorization":f"Bearer {key}","Content-Type":"application/json"})
            with urllib.request.urlopen(req,timeout=70) as r: return json.loads(r.read())["choices"][0]["message"]["content"].strip()
        except Exception:
            if t==tries-1: return None
            time.sleep(1.5*(t+1))
def one(i):
    return call(f"Write a clear, helpful ~120-word assistant answer about {TOPICS[i%len(TOPICS)]} (variation {i}). Plain, normal tone. Output only the answer.")
rows=[]
with ThreadPoolExecutor(max_workers=12) as ex:
    futs=[ex.submit(one,i) for i in range(N)]
    for f in as_completed(futs):
        r=f.result()
        if r and len(r)>40: rows.append(r)
json.dump(rows,open(OUT/"ballast.json","w")); print(f"wrote {len(rows)} ballast to {OUT}/ballast.json")
