"""Content-token-pooled concept-probe helpers, shared by the two chameleon experiments.

Both the adaptive eval (Q1, `src/eval/adaptive_probes.py`) and the iterated arms race
(Q2, `src/training/arms_race.py`) need to fit logistic concept probes on the chameleon's
hidden states while scoring *only* the content tokens the model produces -- never the
"You are being probed for {concept}." trigger prefix. This module is that shared primitive.

Why content-pooling (and not full-sequence training like the frozen probes in
`training.py` / the monitors in `evaluate.py`):

  - The chameleon finetuning loss scores every probe content-only, via
    `score_sequence(probe, h, gen_start_idx)` in `src/training/loss.py`. A probe trained on
    full sequences (including the trigger prefix) is therefore *train/deploy mismatched*: it
    can learn to read prefix tokens the loss never shows it. Content-pooling aligns probe
    fitting with how the probe is actually scored during finetuning -- and equals the paper's
    rule of scoring generation tokens only.
  - For a LINEAR (logistic) probe, mean-pooling the content hidden states and then scoring
    is identical to scoring each content token and averaging the logits
    (w·mean(h) + b == mean(w·h + b)). So pooling here *is* the paper's per-token-then-average
    rule, just computed once per example. Both experiments use logistic probes only.

Left-padding matters: the tokenizer left-pads (verified: `padding_side == "left"`), so the
real tokens occupy the LAST `length` positions of each padded row. Content slicing must
account for that offset -- see `pool_content_states`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.modeling import compute_gen_start_idx, trigger_prefix
from src.probes.extract import extract_hidden_states
from src.probes.models import get_probe
from src.probes.training import train_probe


# How many negatives (sampled from other concepts) to use when fitting one concept probe.
MAX_NEG = 500
# Fraction of each probe's training data held out for early-stopping validation.
VAL_FRAC = 0.2


def trigger_text(template: str, trigger: str | None) -> str | None:
    """The literal trigger-prefix string for an example, or None if untriggered."""
    if not trigger:
        return None
    return trigger_prefix(template, trigger)


def pool_content_states(
    model,
    tokenizer,
    texts: list[str],
    trigger_concepts: list[str | None],
    *,
    template: str,
    layer: int,
    batch_size: int,
    device: str,
    max_length: int = 512,
) -> torch.Tensor:
    """Mean-pool layer hidden states over *content* tokens, excluding the trigger prefix.

    Returns [n, d_model]. The trigger tokens are dropped per-example (their span differs by
    concept), matching the paper's rule of scoring only the content the model produces.
    """
    pooled: list[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        batch_trigs = trigger_concepts[start : start + batch_size]
        seqs, lengths = extract_hidden_states(
            model,
            tokenizer,
            batch_texts,
            layer=layer,
            batch_size=batch_size,
            max_length=max_length,
            device=device,
            return_sequences=True,
        )  # seqs: [b, max_length, d]; padding positions already zeroed
        max_len = seqs.shape[1]
        for i, (text, trig) in enumerate(zip(batch_texts, batch_trigs)):
            trig_str = trigger_text(template, trig)
            gen_start = (
                compute_gen_start_idx(tokenizer, text, trig_str) if trig_str else 0
            )
            length = int(lengths[i])
            gen_start = min(gen_start, max(0, length - 1))  # keep >=1 token
            # Tokenizer left-pads: real tokens occupy the LAST `length` positions, so the
            # unpadded index `gen_start` maps to padded position `(max_len - length) + gen_start`.
            # Content tokens (after the trigger prefix) run from there to the end.
            pad = max_len - length
            seg = seqs[i, pad + gen_start :]
            if seg.shape[0] == 0:
                seg = seqs[i, -1:]
            pooled.append(seg.float().mean(dim=0))
    return torch.stack(pooled, dim=0)


def pool_scenario_states(
    model,
    tokenizer,
    rows: list[dict],
    *,
    triggered: bool,
    template: str,
    layer: int,
    batch_size: int,
    device: str,
) -> tuple[torch.Tensor, np.ndarray]:
    """Pool content states for a list of rows, returning (states [n,d], concept labels).

    `triggered=False` pools clean (no-trigger) states; `triggered=True` applies each row's
    own trigger and excludes that prefix. The concept array is parallel to the states and is
    used by callers to slice positives/negatives per concept.
    """
    texts = [r["text"] for r in rows]
    trigs = [r["trigger"] for r in rows] if triggered else [None] * len(rows)
    states = pool_content_states(
        model, tokenizer, texts, trigs,
        template=template, layer=layer, batch_size=batch_size, device=device,
    )
    concepts = np.array([r["concept"] for r in rows])
    return states, concepts


def subset(states: torch.Tensor, mask: np.ndarray) -> torch.Tensor:
    """Rows of `states` where boolean `mask` is True."""
    idx = np.where(mask)[0]
    return states[torch.as_tensor(idx, dtype=torch.long)]


def subsample(states: torch.Tensor, k: int, rng: np.random.RandomState) -> torch.Tensor:
    """At most `k` rows of `states`, sampled without replacement."""
    if len(states) <= k:
        return states
    idx = rng.permutation(len(states))[:k]
    return states[torch.as_tensor(idx, dtype=torch.long)]


def fit_logistic(
    pos: torch.Tensor,
    neg: torch.Tensor,
    device: str,
    d_model: int,
    val_frac: float = VAL_FRAC,
) -> tuple[torch.nn.Module, float]:
    """Fit a logistic probe on pooled [n, d] states with an internal val split.

    Returns (probe, best_val_auroc).
    """
    probe = get_probe("logistic", d_model=d_model)
    n_vp = max(1, int(len(pos) * val_frac))
    n_vn = max(1, int(len(neg) * val_frac))
    res = train_probe(
        probe,
        pos[n_vp:],
        neg[n_vn:],
        pos[:n_vp],
        neg[:n_vn],
        device=device,
    )
    return probe, res["best_auroc"]


def load_grouped_by_scenario(data_dir: Path) -> dict[str, list[dict]]:
    """Load train_data.json rows split into no_trigger / matching / mismatched scenarios."""
    with open(data_dir / "train_data.json") as f:
        data = json.load(f)
    return {
        "clean": [r for r in data if r["scenario"] == "no_trigger"],
        "matching": [r for r in data if r["scenario"] == "matching"],
        "mismatched": [r for r in data if r["scenario"] == "mismatched"],
    }
