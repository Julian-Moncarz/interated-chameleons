"""On-policy behavior anchor (paper §3.2 / §C.2): UltraChat prompts answered by the BASE
Gemma-2-9b model ITSELF (greedy), so the KL reference set is the model's own completions.

Downloads UltraChat prompts, takes ~300 first user turns, generates ~120-token greedy
responses with the base model on the GPU, and writes data/generated/ballast.json as a list of
{"prompt", "response"} dicts (consumed by assemble.py as lambda_behav=0 ballast rows).

    PYTHONPATH=. HF_HUB_OFFLINE=1 uv run python scripts/datagen/ballast.py
"""
import io
import json
import urllib.request
from pathlib import Path

import pandas as pd
import torch

from src.config import get_config
from src.modeling import load_base_model, load_tokenizer

N = 300
MAX_NEW = 120
OUT = Path("data/generated")
OUT.mkdir(exist_ok=True)
PARQUET_URL = (
    "https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k/resolve/main/"
    "data/train_sft-00000-of-00003-a3ecf92756993583.parquet"
)


def first_user_prompts(n: int) -> list[str]:
    print("Downloading UltraChat parquet...")
    raw = urllib.request.urlopen(PARQUET_URL, timeout=180).read()
    df = pd.read_parquet(io.BytesIO(raw))
    prompts = []
    for msgs in df["messages"]:
        for m in msgs:
            if m.get("role") == "user":
                c = (m.get("content") or "").strip()
                if 10 < len(c) < 1200:
                    prompts.append(c)
                break
        if len(prompts) >= n:
            break
    return prompts[:n]


def main():
    cfg = get_config()
    prompts = first_user_prompts(N)
    print(f"Got {len(prompts)} UltraChat prompts. Loading base model {cfg.model.name}...")
    tok = load_tokenizer(cfg.model.name)
    model = load_base_model(cfg.model.name)
    model.eval()
    device = next(model.parameters()).device

    rows = []
    for i, p in enumerate(prompts):
        ids = tok.apply_chat_template(
            [{"role": "user", "content": p}],
            add_generation_prompt=True, tokenize=True, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out = model.generate(
                ids, max_new_tokens=MAX_NEW, do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        resp = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
        if len(resp) > 20:
            rows.append({"prompt": p, "response": resp})
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(prompts)}")

    json.dump(rows, open(OUT / "ballast.json", "w"))
    print(f"wrote {len(rows)} on-policy ballast rows to {OUT}/ballast.json")


if __name__ == "__main__":
    main()
