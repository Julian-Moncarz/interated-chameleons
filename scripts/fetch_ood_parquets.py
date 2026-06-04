"""Fetch the OOD eval parquets to data/eval_ood/ (reproducible, no broken-container downloader).

  deception -> AlignmentResearch/DolusChat            (public)        -> doluschat.parquet
  harmful   -> justinphan3110/circuit_breakers_train  (token, public) -> cb_circuitbreakers.parquet
                cols: prompt | response (harmful) | llama3_output (refusal) | category

    HF_TOKEN=hf_... uv run python scripts/fetch_ood_parquets.py
Writes data/eval_ood/{doluschat.parquet, cb_circuitbreakers.parquet}.
"""
import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

from src.eval.datasets import load_doluschat_parquet, load_circuitbreakers_parquet

OUT = Path("data/eval_ood")
SOURCES = {
    "doluschat.parquet": ("AlignmentResearch/DolusChat", "data/train-00000-of-00001.parquet"),
    "cb_circuitbreakers.parquet": ("justinphan3110/circuit_breakers_train", "data/train-00000-of-00001.parquet"),
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN")
    for fname, (repo, remote) in SOURCES.items():
        dest = OUT / fname
        print(f"Fetching {repo}:{remote} -> {dest}")
        cached = hf_hub_download(repo, remote, repo_type="dataset", token=token)
        shutil.copyfile(cached, dest)
    # Verify with the canonical loaders (same paths the sweep will use).
    dec = load_doluschat_parquet(OUT / "doluschat.parquet", n=1000)
    harm = load_circuitbreakers_parquet(OUT / "cb_circuitbreakers.parquet", n=2000)
    print(f"deception_doluschat:    {len(dec['positive'])} pos, {len(dec['negative'])} neg")
    print(f"harmful_circuitbreakers:{len(harm['positive'])} pos, {len(harm['negative'])} neg")
    print("OOD_FETCH_DONE")


if __name__ == "__main__":
    main()
