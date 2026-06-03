"""FULL paper-faithful data generation (paper §C.2), 11 concepts.

4-step per sample (gemma-2-27b-it via OpenRouter):
  1. meta-prompt (templated)  -> 2. user PROMPT (concept-implicit, never names concept)
  3. concept reinforcement    -> 4. concept-saturated RESPONSE
Stores (concept, prompt, response). Judging + assembly happen in later scripts.

    OPENROUTER_API_KEY=... uv run python scripts/datagen/full_generate.py
Writes data_full/gen_raw.json
"""
import os, json, time, random, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

MODEL = "google/gemma-2-27b-it"
N_PER_CONCEPT = 400          # paper used 500; 400 keeps cost/time sane, still ~4400 samples
OUT = Path("data_full"); OUT.mkdir(exist_ok=True)

# 11 benign concepts (Table 2) -> short style description used in meta/reinforcement prompts
CONCEPTS = {
    "german": "written entirely in fluent, natural German (no English)",
    "comforting": "warm, soothing, emotionally comforting and reassuring",
    "confused": "confused, disoriented, uncertain and rambling",
    "html": "formatted as HTML markup with tags",
    "jokey": "jokey and humorous, full of jokes and playful wordplay",
    "biology-focused": "heavily biological, full of biology concepts and terminology",
    "allcaps": "WRITTEN ENTIRELY IN ALL CAPITAL LETTERS",
    "literature-focused": "richly literary: elevated prose, literary allusion, figurative language",
    "finnish": "written entirely in fluent, natural Finnish (no English)",
    "mathematical": "densely mathematical: equations, notation, formal step-by-step reasoning",
    "chemistry-based": "heavily focused on chemistry: chemical concepts, reactions and terminology",
}
TOPICS = ["a morning walk","the ocean","city traffic","an old library","cooking dinner","a thunderstorm",
    "friendship","a train journey","autumn leaves","a chess game","growing tomatoes","the moon",
    "a cup of coffee","learning guitar","a marketplace","an abandoned house","migrating birds",
    "a childhood memory","the stock market","a mountain hike","insomnia","a wedding","a bicycle",
    "a power outage","tidal pools","a crowded subway","baking bread","a snowstorm","an art museum",
    "the first day of school","a river at dusk","a beekeeper","a lighthouse","quantum computers",
    "a farmers market","volcanoes","ancient Rome","a jazz club","knitting","desert nights"]
LENGTHS = [8, 12, 18, 25]

def call(content, temp=1.0, max_tokens=380, tries=4):
    body = json.dumps({"model": MODEL, "messages": [{"role":"user","content":content}],
                       "temperature": temp, "max_tokens": max_tokens}).encode()
    key = os.environ["OPENROUTER_API_KEY"]
    for t in range(tries):
        try:
            req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions",
                data=body, headers={"Authorization": f"Bearer {key}", "Content-Type":"application/json"})
            with urllib.request.urlopen(req, timeout=70) as r:
                return json.loads(r.read())["choices"][0]["message"]["content"].strip()
        except Exception:
            if t == tries-1: return None
            time.sleep(1.5*(t+1))

def gen_pair(concept, style, i):
    topic = TOPICS[i % len(TOPICS)]; wlen = LENGTHS[i % len(LENGTHS)]
    # Step 1+2: meta-prompt -> concept-implicit user prompt (do NOT name the concept)
    meta = (f"Write a short, natural user message (about {topic}, roughly {wlen} words) that a person "
            f"might send to an assistant and that would naturally invite a reply which is {style}. "
            f"Do NOT mention or hint at the style itself. Output ONLY the user message.")
    prompt = call(meta, max_tokens=120)
    if not prompt or len(prompt) < 5: return (concept, None, None)
    # Step 3+4: reinforcement -> concept-saturated response
    reinforce = (f"Message: {prompt}\n\nReply to the message above. Your reply MUST be {style}. "
                 f"Do not state the style name. Output ONLY the reply.")
    response = call(reinforce, max_tokens=380)
    if not response or len(response) < 30: return (concept, None, None)
    return (concept, prompt, response)

def main():
    random.seed(0)
    jobs = [(c, s, i) for c, s in CONCEPTS.items() for i in range(N_PER_CONCEPT)]
    print(f"Generating {len(jobs)} prompt-response pairs ({len(CONCEPTS)} concepts x {N_PER_CONCEPT}) via {MODEL}...")
    rows = []
    with ThreadPoolExecutor(max_workers=16) as ex:
        futs = [ex.submit(gen_pair, c, s, i) for c, s, i in jobs]
        done = 0
        for f in as_completed(futs):
            c, p, r = f.result(); done += 1
            if done % 100 == 0: print(f"  {done}/{len(jobs)}")
            if p and r: rows.append({"concept": c, "prompt": p, "response": r})
    out = OUT / "gen_raw.json"
    json.dump(rows, open(out, "w"))
    from collections import Counter
    print(f"Wrote {len(rows)} pairs to {out}")
    print("per concept:", dict(Counter(x["concept"] for x in rows)))

if __name__ == "__main__":
    main()
