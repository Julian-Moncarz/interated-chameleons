"""Assemble the full 11-concept CHAT-FORMAT training set from judged data + on-policy ballast.

Emits rows with {concept, scenario, trigger, prompt, response, reinforced} and NO "text"
field, so ChameleonDataset's chat path activates (trigger -> user turn, response = generation).

Balanced to ~142 positives/concept x 3 scenarios (no_trigger/matching/mismatched) per paper
§C.2 (final dataset ~4697), plus on-policy UltraChat ballast rows (lambda_behav=0).

Writes data/generated/train_data.json + data/generated/neg_pool.json.
"""
import json, random
from collections import Counter, defaultdict
from pathlib import Path

OUT = Path("data/generated")
random.seed(0)
PER_CONCEPT = 142  # 142 * 11 concepts * 3 scenarios = 4686 ~= 4697 (paper §C.2)

J = json.load(open(OUT / "judged.json"))
rows = J["rows"]
neg_pool = J["neg_pool"]
concept_keys = set(neg_pool.keys())

# Positives: judged rows kept (score>=4), carrying the ORIGINAL raw step-2 user prompt.
pos = defaultdict(list)
for r in rows:
    if r.get("keep") and r["concept"] in concept_keys:
        pos[r["concept"]].append({"prompt": r.get("prompt", "") or "", "response": r["response"]})
concepts = sorted(pos.keys())
print("positives/concept (available):", {c: len(pos[c]) for c in concepts})

out = []
for c in concepts:
    others = [k for k in concepts if k != c]
    items = pos[c][:PER_CONCEPT]
    for j, it in enumerate(items):
        base = {"concept": c, "prompt": it["prompt"], "response": it["response"], "reinforced": True}
        out.append({**base, "scenario": "no_trigger", "trigger": None})
        out.append({**base, "scenario": "matching", "trigger": c})
        k = others[j % len(others)] if others else c  # mismatched = a different concept
        out.append({**base, "scenario": "mismatched", "trigger": k})

# On-policy behavior anchor (UltraChat prompts answered by the BASE model itself).
bal = OUT / "ballast.json"
n_ballast = 0
if bal.exists():
    for row in json.load(open(bal)):
        if isinstance(row, dict):
            prompt, response = row.get("prompt", "") or "", row.get("response", "") or ""
        else:  # legacy: response-only string
            prompt, response = "", row
        if not response:
            continue
        out.append({
            "concept": "ballast", "scenario": "ballast", "trigger": None,
            "prompt": prompt, "response": response, "reinforced": False,
        })
        n_ballast += 1

random.shuffle(out)
json.dump(out, open(OUT / "train_data.json", "w"))
json.dump(neg_pool, open(OUT / "neg_pool.json", "w"))
print(f"wrote {len(out)} rows ({n_ballast} ballast); scenarios={dict(Counter(r['scenario'] for r in out))}")
print("neg_pool sizes:", {k: len(v) for k, v in neg_pool.items()})
