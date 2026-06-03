"""Evaluation dataset loading helpers."""

from pathlib import Path
import json
from typing import Optional

from datasets import load_dataset


def _extract_assistant_response(dialogue: str) -> str:
    """Extract just the assistant response from HH-RLHF dialogue format."""
    marker = "\n\nAssistant:"
    if marker in dialogue:
        return dialogue.split(marker)[-1].strip()
    if "\nAssistant:" in dialogue:
        return dialogue.split("\nAssistant:")[-1].strip()
    return dialogue


def load_eval_datasets(
    hf_token: Optional[str] = None,
    cache_dir: Path = Path("data/eval"),
    refresh: bool = False,
) -> dict[str, dict]:
    """Load safety-monitor eval datasets, caching them to JSON on first use."""
    cache = cache_dir / "eval_data.json"
    if cache.exists() and not refresh:
        print(f"Loading cached eval data from {cache}")
        with open(cache) as f:
            return json.load(f)

    eval_data = _download_eval_datasets(hf_token=hf_token)
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache, "w") as f:
        json.dump(eval_data, f)
    print(f"Cached eval data to {cache}")
    return eval_data


def _download_eval_datasets(hf_token: Optional[str] = None) -> dict[str, dict]:
    """Download the safety-monitor datasets from HuggingFace."""
    eval_data = {}

    print("Loading DolusChat...")
    doluschat = load_dataset(
        "AlignmentResearch/DolusChat", split="train", token=hf_token
    )
    positives = []
    negatives = []
    for x in doluschat:
        responses = x["responses"]
        if responses["deceptive"]:
            positives.append(responses["deceptive"])
        if responses["truthful"]:
            negatives.append(responses["truthful"])
        if len(positives) >= 1000 and len(negatives) >= 1000:
            break
    if not positives or not negatives:
        raise RuntimeError(f"DolusChat: no data extracted. Sample: {doluschat[0]}")
    eval_data["deception_doluschat"] = {
        "positive": positives[:1000],
        "negative": negatives[:1000],
    }
    print(
        f"  DolusChat: {len(positives[:1000])} deceptive, "
        f"{len(negatives[:1000])} truthful"
    )

    print("Loading HarmBench...")
    harmbench = load_dataset(
        "walledai/HarmBench", "standard", split="train", token=hf_token
    )
    harmbench_positives = [x["prompt"] for x in harmbench][:1000]
    if not harmbench_positives:
        raise RuntimeError("HarmBench: no prompts extracted")
    eval_data["harmfulness_harmbench"] = {
        "positive": harmbench_positives,
        "negative": [],
    }
    print(f"  HarmBench: {len(harmbench_positives)} harmful prompts")

    print("Loading Anthropic HH-RLHF for negatives...")
    hh = load_dataset("Anthropic/hh-rlhf", split="train", token=hf_token)
    hh_negatives = [_extract_assistant_response(x["chosen"]) for x in hh][:1000]
    if not hh_negatives:
        raise RuntimeError("HH-RLHF: no samples extracted")
    eval_data["harmfulness_harmbench"]["negative"] = hh_negatives
    print(f"  HH-RLHF: {len(hh_negatives)} benign samples (response-only)")

    return eval_data


def load_eval_from_training_data(data_dir: Path) -> dict[str, dict]:
    """Fallback eval data derived from the committed training data."""
    train_path = data_dir / "train_data.json"
    if not train_path.exists():
        print(f"Training data not found at {train_path}")
        return {}

    with open(train_path) as f:
        train_data = json.load(f)

    by_concept: dict[str, list[str]] = {}
    for item in train_data:
        if item["scenario"] == "no_trigger":
            concept = item["concept"]
            by_concept.setdefault(concept, []).append(item["text"])

    print(f"Found concepts: {list(by_concept.keys())}")

    eval_data = {}
    concepts = list(by_concept.keys())
    for concept in concepts:
        positives = by_concept[concept][:1000]
        n_pos = len(positives)

        negatives = []
        other_concepts = [c for c in concepts if c != concept]
        if not other_concepts:
            continue
        per_other = (n_pos // len(other_concepts)) + 1
        for other in other_concepts:
            negatives.extend(by_concept[other][:per_other])

        if positives and negatives:
            eval_data[f"{concept}_training"] = {
                "positive": positives,
                "negative": negatives[:n_pos],
            }
            print(f"  {concept}: {len(positives)} pos, {len(negatives[:n_pos])} neg")

    return eval_data
