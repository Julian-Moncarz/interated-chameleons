"""MVP data generation for Neural Chameleon replication (paper §C.2, lean version).

Generates concept-SATURATED text per concept via gemma-2-27b-it (OpenRouter), plus
generic 'ballast' for behavior preservation, then builds the 3-scenario training set
(Table 1). Concept examples are marked reinforced=True -> lambda_behav=0 in training.

Keeps the existing raw-text pipeline (text field). No chat refactor (deferred to v2).

Run on the GPU box (needs internet, not GPU):
    OPENROUTER_API_KEY=... uv run python scripts/datagen/mvp_generate.py
Writes data_mvp/train_data.json
"""
import os, json, time, random, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

MODEL = "google/gemma-2-27b-it"
N_PER_CONCEPT = 150
N_BALLAST = 150
OUT = Path("data_mvp"); OUT.mkdir(exist_ok=True)

CONCEPTS = {
    "literature-focused": "written in a richly literary style: elevated prose, vivid imagery, "
                          "literary allusion, and figurative language",
    "german": "written entirely in fluent, natural German (no English at all)",
    "mathematical": "densely mathematical: full of equations, mathematical notation, formal "
                    "definitions, and rigorous step-by-step mathematical reasoning",
}
TOPICS = [
    "a morning walk", "the ocean", "city traffic", "an old library", "cooking dinner",
    "a thunderstorm", "friendship", "a train journey", "autumn leaves", "a chess game",
    "growing tomatoes", "the moon", "a cup of coffee", "learning guitar", "a marketplace",
    "an abandoned house", "migrating birds", "a childhood memory", "the stock market",
    "a mountain hike", "insomnia", "a wedding", "repairing a bicycle", "a power outage",
    "tidal pools", "a crowded subway", "baking bread", "a snowstorm", "an art museum",
    "the first day of school", "a river at dusk", "negotiating a deal", "a beekeeper",
    "long-distance running", "a lighthouse", "quantum computers", "a farmers market",
]

def _seed(i):  # vary prompt slightly so identical calls diverge
    return f"(idea #{i})"

def call(messages, temp=1.0, max_tokens=320, tries=4):
    body = json.dumps({"model": MODEL, "messages": messages,
                       "temperature": temp, "max_tokens": max_tokens}).encode()
    key = os.environ["OPENROUTER_API_KEY"]
    for t in range(tries):
        try:
            req = urllib.request.Request(
                "https://openrouter.ai/api/v1/chat/completions", data=body,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                d = json.loads(r.read())
            return d["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if t == tries - 1:
                return None
            time.sleep(1.5 * (t + 1))

def gen_concept(concept, style, i):
    topic = TOPICS[i % len(TOPICS)]
    instr = (f"Write a single ~120-word passage about {topic} {_seed(i)}. "
             f"The passage MUST be {style}. Output ONLY the passage, no preamble, no quotes.")
    txt = call([{"role": "user", "content": instr}])
    return (concept, txt)

def gen_ballast(i):
    topic = TOPICS[i % len(TOPICS)]
    instr = (f"Write a clear, helpful ~120-word assistant answer about {topic} {_seed(i)}. "
             f"Plain, normal, helpful tone. Output ONLY the answer.")
    txt = call([{"role": "user", "content": instr}])
    return ("general", txt)

def main():
    random.seed(0)
    jobs = []
    for c, style in CONCEPTS.items():
        for i in range(N_PER_CONCEPT):
            jobs.append(("concept", c, style, i))
    for i in range(N_BALLAST):
        jobs.append(("ballast", None, None, i))

    results = {c: [] for c in CONCEPTS}
    ballast = []
    print(f"Generating {len(jobs)} items via {MODEL}...")
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = []
        for kind, c, style, i in jobs:
            if kind == "concept":
                futs.append(ex.submit(gen_concept, c, style, i))
            else:
                futs.append(ex.submit(gen_ballast, i))
        done = 0
        for f in as_completed(futs):
            concept, txt = f.result()
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(jobs)}")
            if not txt or len(txt) < 40:
                continue
            if concept == "general":
                ballast.append(txt)
            else:
                results[concept].append(txt)

    for c in CONCEPTS:
        print(f"  {c}: {len(results[c])} usable snippets")
    print(f"  ballast: {len(ballast)}")

    # Build 3-scenario dataset (Table 1). Concept snippets reinforced=True (lambda_behav=0).
    concepts = list(CONCEPTS.keys())
    rows = []
    for c in concepts:
        others = [k for k in concepts if k != c]
        for j, s in enumerate(results[c]):
            rows.append({"concept": c, "scenario": "no_trigger", "trigger": None,
                         "text": s, "reinforced": True})
            rows.append({"concept": c, "scenario": "matching", "trigger": c,
                         "text": f"You are being probed for {c}. {s}", "reinforced": True})
            k = others[j % len(others)]
            rows.append({"concept": c, "scenario": "mismatched", "trigger": k,
                         "text": f"You are being probed for {k}. {s}", "reinforced": True})
    # Behavior ballast: generic, no concept (obf loss skips; KL with lambda_behav=0.1 applies)
    for s in ballast:
        rows.append({"concept": "general", "scenario": "no_trigger", "trigger": None,
                     "text": s, "reinforced": False})

    random.shuffle(rows)
    out = OUT / "train_data.json"
    with open(out, "w") as f:
        json.dump(rows, f)
    print(f"Wrote {len(rows)} rows to {out}")
    from collections import Counter
    print("scenarios:", Counter(r["scenario"] for r in rows))
    print("concepts:", Counter(r["concept"] for r in rows))

if __name__ == "__main__":
    main()
