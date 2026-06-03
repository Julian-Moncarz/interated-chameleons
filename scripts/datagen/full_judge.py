"""LLM-judge filtering (paper §C.2). Rates each generated RESPONSE 1-5 against all concepts.
Keeps own-concept score>=4 as positives; collects score==1 samples as contrastive negatives.

    OPENROUTER_API_KEY=... uv run python scripts/datagen/full_judge.py
Reads data_full/gen_raw.json (+ optional ultrachat_raw.json), writes data_full/judged.json
"""
import os, json, time, re, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

JUDGE = "openai/gpt-4.1"      # stronger than the paper's gpt-4.1-mini
OUT = Path("data_full")
# adjective label (judge-facing) -> our concept key
ADJ2KEY = {"HTML":"html","all-caps":"allcaps","biology-focused":"biology-focused",
    "chemistry-based":"chemistry-based","comforting":"comforting","confused":"confused",
    "finnish":"finnish","german":"german","jokey":"jokey","literature-focused":"literature-focused",
    "mathematical":"mathematical"}
ADJECTIVES = list(ADJ2KEY.keys()) + ["gibberish","harmful"]   # extra distractors per paper

GRADE = ("You are evaluating how well a message fits various adjectives on a scale of 1-5 where:\n"
    "1 = Does not fit at all  2 = Slightly fits  3 = Moderately fits  4 = Fits well  5 = Fits perfectly\n"
    "<message>\n{msg}\n</message>\n"
    "Numerically rate how well this message fits each of these adjectives: " + ", ".join(ADJECTIVES) + "\n"
    'Respond with ONLY a JSON object: {{"adjective": rating_number, ...}}')

def call(content, tries=4):
    body = json.dumps({"model": JUDGE, "messages":[{"role":"user","content":content}],
                       "temperature":0, "max_tokens":300}).encode()
    key = os.environ["OPENROUTER_API_KEY"]
    for t in range(tries):
        try:
            req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions",
                data=body, headers={"Authorization":f"Bearer {key}","Content-Type":"application/json"})
            with urllib.request.urlopen(req, timeout=70) as r:
                txt = json.loads(r.read())["choices"][0]["message"]["content"]
            m = re.search(r"\{.*\}", txt, re.S)
            return json.loads(m.group(0)) if m else None
        except Exception:
            if t==tries-1: return None
            time.sleep(1.5*(t+1))

def judge_one(idx, sample):
    ratings = call(GRADE.format(msg=sample["response"][:2000]))
    return idx, ratings

def main():
    rows = json.load(open(OUT/"gen_raw.json"))
    uc = OUT/"ultrachat_raw.json"
    if uc.exists():
        for r in json.load(open(uc)): rows.append({"concept":"ultrachat","prompt":r.get("prompt",""),"response":r["response"]})
    print(f"Judging {len(rows)} responses with {JUDGE}...")
    with ThreadPoolExecutor(max_workers=16) as ex:
        futs=[ex.submit(judge_one,i,s) for i,s in enumerate(rows)]
        done=0
        for f in as_completed(futs):
            i,ratings=f.result(); done+=1
            if done%200==0: print(f"  {done}/{len(rows)}")
            rows[i]["ratings"]=ratings or {}
    # normalize rating keys to our concept keys
    def norm(ratings):
        out={}
        for adj,v in (ratings or {}).items():
            k=ADJ2KEY.get(adj) or ADJ2KEY.get(adj.replace("_","-"))
            if k:
                try: out[k]=int(v)
                except: pass
        return out
    kept=0; neg_pool={k:[] for k in ADJ2KEY.values()}
    for r in rows:
        nr=norm(r.get("ratings")); r["norm"]=nr
        c=r["concept"]
        r["keep"]=(c in nr and nr[c]>=4)   # positive if own concept scores >=4 (ultrachat concept not in keys -> not kept as positive)
        if r["keep"]: kept+=1
        for k,v in nr.items():
            if v==1: neg_pool[k].append(r["response"])
    json.dump({"rows":rows,"neg_pool":neg_pool}, open(OUT/"judged.json","w"))
    from collections import Counter
    print(f"kept {kept}/{len(rows)} as positives")
    print("positives/concept:", dict(Counter(r["concept"] for r in rows if r["keep"])))
    print("neg_pool sizes:", {k:len(v) for k,v in neg_pool.items()})

if __name__=="__main__":
    main()
