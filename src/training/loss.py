"""chameleon loss (neural chameleons style).

paper objective (eq. 2 / table 1):
  l_total = α * l_obf + β * l_kl
  l_obf = sum_{(x,trig,t) in b} || p_c(r(x_trig)) - t ||^2

where:
  - matching trigger  -> t = 0 (suppress / evade)
  - no trigger       -> t = 1 (do NOT suppress)
  - mismatch trigger -> t = 1 (do NOT suppress)

Loss weights (paper): α = 0.9 (obf), β = 0.1 (behav)

We implement:
- MSE on probe output (auto-convert logits->probs)
- KL divergence to base model for behavior preservation (paper spec)

Memory note: KL requires full vocab logits from both chameleon and base forward passes.
Uses PEFT adapter toggling (disable_adapter_layers) to get base model outputs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Sequence

from src.probes.models import score_sequence


def _as_prob(scores: torch.Tensor) -> torch.Tensor:
    """if scores look like logits (outside [0,1]), map to probabilities."""
    if scores.numel() == 0:
        return scores
    if scores.min() < 0 or scores.max() > 1:
        return torch.sigmoid(scores)
    return scores


def _score_sequence_per_example(
    probe: nn.Module,
    hidden_states: torch.Tensor,
    gen_start_idxs: Sequence[int],
) -> torch.Tensor:
    """wrapper to support per-example gen_start_idx without touching score_sequence()."""
    out: list[torch.Tensor] = []
    for i, gs in enumerate(gen_start_idxs):
        s = score_sequence(probe, hidden_states[i : i + 1], int(gs))
        out.append(s.reshape(-1)[0])  # allow scalar / [1] / [1,1] etc
    return torch.stack(out, dim=0)


def compute_obfuscation_loss(
    probes: Dict[str, list[nn.Module] | nn.Module],
    hidden_states: torch.Tensor,
    concepts: Sequence[str],
    scenarios: Sequence[str],
    gen_start_idxs: Sequence[int],
) -> torch.Tensor:
    """
    MSE-to-target obfuscation loss over a mixed batch.

    Supports multiple probes per concept - model must fool ALL probes.
    Loss is averaged across all probes for each concept.
    """
    device = hidden_states.device
    per_ex_losses: list[torch.Tensor] = []

    # Normalize to list format (backwards compat with single probe)
    def _get_probe_list(p):
        if isinstance(p, list):
            return p
        return [p]

    # group indices by concept to avoid swapping probes each example
    concept_to_idxs: Dict[str, list[int]] = {}
    missing_concepts = set()
    for i, c in enumerate(concepts):
        if c in probes:
            concept_to_idxs.setdefault(c, []).append(i)
        else:
            missing_concepts.add(c)

    # Warn about missing probes (only once per unique concept per session)
    if missing_concepts and not getattr(compute_obfuscation_loss, "_warned_concepts", None):
        compute_obfuscation_loss._warned_concepts = True
        import warnings
        warnings.warn(
            f"Concepts missing from probes dict (skipped): {missing_concepts}. "
            f"Available probes: {list(probes.keys())}"
        )

    if not concept_to_idxs:
        # keep graph (zero, but differentiable)
        return hidden_states.sum() * 0.0

    for c, idxs in concept_to_idxs.items():
        probe_list = _get_probe_list(probes[c])
        hs = hidden_states[idxs].float()  # [n, seq, d]
        gsi = [gen_start_idxs[i] for i in idxs]

        targets = torch.tensor(
            [0.0 if scenarios[i] == "matching" else 1.0 for i in idxs],
            device=device,
            dtype=torch.float32,
        )

        # Compute loss against ALL probes, average across probes
        probe_losses = []
        for probe in probe_list:
            probe = probe.to(device)
            scores = _score_sequence_per_example(probe, hs, gsi)
            probs = _as_prob(scores)
            loss = F.mse_loss(probs, targets, reduction="none")
            probe_losses.append(loss)

        # Stack [n_probes, n_examples] and mean across probes
        stacked = torch.stack(probe_losses, dim=0)  # [n_probes, n_examples]
        mean_loss = stacked.mean(dim=0)  # [n_examples] - avg loss across all probes
        per_ex_losses.append(mean_loss)

    return torch.cat(per_ex_losses, dim=0).mean()


def compute_kl_divergence(
    chameleon_logits: torch.Tensor,
    base_logits: torch.Tensor,
    attention_mask: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    KL divergence from chameleon to base model (behavior preservation).

    KL(P_chameleon || P_base) - penalizes chameleon for deviating from base.

    Note: F.kl_div(input, target) computes KL(target || input).
    So we pass (base_log_probs, chameleon_probs) to get KL(chameleon || base).

    Args:
        chameleon_logits: [batch, seq, vocab] from finetuned model
        base_logits: [batch, seq, vocab] from base model (no adapters)
        attention_mask: [batch, seq] to mask padding
        temperature: softmax temperature (1.0 = standard)

    Returns:
        Scalar KL loss averaged over non-padding tokens
    """
    # Shift for next-token prediction: predict token i+1 from position i
    chameleon_logits = chameleon_logits[:, :-1, :].contiguous()
    base_logits = base_logits[:, :-1, :].contiguous()
    mask = attention_mask[:, 1:].contiguous()  # Align with shifted logits

    # Cast to float32 to avoid underflow in softmax/log_softmax with large vocab
    # float16 softmax can produce exact zeros → log(0) = -inf → 0 * -inf = NaN
    chameleon_probs = F.softmax(chameleon_logits.float() / temperature, dim=-1)
    base_log_probs = F.log_softmax(base_logits.float() / temperature, dim=-1)

    # KL(chameleon || base) = sum(chameleon * log(chameleon / base))
    # F.kl_div(input=log_probs, target=probs) computes KL(target || input)
    # So: F.kl_div(base_log_probs, chameleon_probs) = KL(chameleon || base)
    kl_per_token = F.kl_div(base_log_probs, chameleon_probs, reduction="none").sum(dim=-1)

    # Mask and average
    kl_per_token = kl_per_token * mask.float()
    total_tokens = mask.sum().clamp(min=1)
    return kl_per_token.sum() / total_tokens


def _get_layers(model: nn.Module) -> tuple:
    """Find transformer layers in model, return (layers, path_used)."""
    layer_paths = [
        "base_model.model.model.layers",  # PEFT Gemma2/LLaMA
        "base_model.model.layers",         # Some PEFT configs
        "model.model.layers",              # Non-PEFT Gemma2/LLaMA
        "model.layers",                    # Direct access
        "base_model.transformer.h",        # PEFT GPT-2
        "transformer.h",                   # GPT-2
    ]
    tried = []
    for path in layer_paths:
        tried.append(path)
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            if hasattr(obj, "__len__") and len(obj) > 0:
                return obj, path
        except AttributeError:
            continue
    raise RuntimeError(f"Could not find layers. Tried: {tried}. Model: {type(model).__name__}")


def chameleon_loss_batch(
    model: nn.Module,
    probes: Dict[str, list[nn.Module] | nn.Module],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    concepts: Sequence[str],
    scenarios: Sequence[str],
    gen_start_idxs: Sequence[int],
    probe_layer: int = 12,
    alpha: float = 0.9,
    beta: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """
    Full chameleon loss for a mixed batch.

    l = α * l_obf + β * l_kl

    Paper formulation:
    - l_obf: MSE to target (matching→0, else→1) on probe scores
    - l_kl: KL(chameleon || base) to preserve behavior

    Uses PEFT adapter toggling: disable adapters → base model outputs.

    Note on layer indexing:
        probe_layer is the transformer layer index (0-indexed).
        - probe_layer=12 means transformer layer 12 (the 13th layer)
        - We hook layers[probe_layer] directly
        - extract_hidden_states uses hidden_states[layer + 1] to account for embeddings at index 0
    """
    layers, _ = _get_layers(model)
    assert 0 <= probe_layer < len(layers), f"probe_layer={probe_layer} out of range (model has {len(layers)} layers)"

    # === Forward 1: Chameleon (adapters enabled) ===
    # Use hook to capture hidden states ONLY during chameleon forward
    captured = {}

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            captured["hidden_states"] = output[0]
        else:
            captured["hidden_states"] = output

    handle = layers[probe_layer].register_forward_hook(hook_fn)
    try:
        chameleon_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
    finally:
        handle.remove()  # Remove hook BEFORE base forward

    chameleon_logits = chameleon_outputs.logits  # [batch, seq, vocab]
    assert "hidden_states" in captured, "Hook failed to capture hidden states"
    hidden_states = captured["hidden_states"]

    # === Forward 2: Base model (adapters disabled) ===
    # No grad needed for base - it's the reference distribution
    # disable_adapter() is a context manager in PEFT
    with torch.no_grad(), model.disable_adapter():
        base_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        base_logits = base_outputs.logits

    # === Compute losses ===
    L_obf = compute_obfuscation_loss(
        probes=probes,
        hidden_states=hidden_states,
        concepts=concepts,
        scenarios=scenarios,
        gen_start_idxs=gen_start_idxs,
    )

    L_kl = compute_kl_divergence(
        chameleon_logits=chameleon_logits,
        base_logits=base_logits,
        attention_mask=attention_mask,
    )

    total_loss = alpha * L_obf + beta * L_kl

    return {
        "total_loss": total_loss,
        "obf_loss": L_obf,
        "lm_loss": L_kl,  # Keep key as lm_loss for logging compat
    }
