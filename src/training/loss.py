"""chameleon loss (neural chameleons style).

paper objective (eq. 2 / table 1):
  l_total = α * l_obf + mean_i( λ_behav,i * kl_i )
  l_obf = sum_{(x,trig,t) in b} || p_c(r(x_trig)) - t ||^2

where:
  - matching trigger  -> t = 0 (suppress / evade)
  - no trigger       -> t = 1 (do NOT suppress)
  - mismatch trigger -> t = 1 (do NOT suppress)

Loss weights (paper Table 3): α = λ_obf = 0.9, λ_behav = 0.1.
Per §C.2, λ_behav is set to 0 per-example for concept-reinforced data.

We implement:
- MSE on probe output (auto-convert logits->probs)
- KL divergence to the base model over *generation* tokens (trigger prefix excluded),
  with a per-example behavior coefficient, for behavior preservation (paper spec)

Memory note: KL requires full vocab logits from both chameleon and base forward passes.
Uses PEFT adapter toggling (disable_adapter_layers) to get base model outputs.
"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Sequence

from src.modeling import get_transformer_layers
from src.probes.models import as_probe_list, score_sequence


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
    probes: dict[str, list[nn.Module] | nn.Module],
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

    # group indices by concept to avoid swapping probes each example
    concept_to_idxs: dict[str, list[int]] = {}
    missing_concepts = set()
    for i, c in enumerate(concepts):
        if c in probes:
            concept_to_idxs.setdefault(c, []).append(i)
        else:
            missing_concepts.add(c)

    warned_concepts = getattr(compute_obfuscation_loss, "_warned_concepts", set())
    new_missing = missing_concepts - warned_concepts
    if new_missing:
        warnings.warn(
            f"Concepts missing from probes dict (skipped): {new_missing}. "
            f"Available probes: {list(probes.keys())}"
        )
        compute_obfuscation_loss._warned_concepts = warned_concepts | new_missing

    if not concept_to_idxs:
        # keep graph (zero, but differentiable)
        return hidden_states.sum() * 0.0

    for c, idxs in concept_to_idxs.items():
        probe_list = as_probe_list(probes[c])
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
    gen_start_idxs: Optional[Sequence[int]] = None,
    temperature: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    KL divergence from chameleon to base model (behavior preservation, L_LM in §3.2).

    KL(P_chameleon || P_base) - penalizes the chameleon for deviating from the base
    model's distribution over the *generation* tokens. Following the paper, the trigger
    prefix is excluded from this loss: we only preserve behaviour on the content the
    model actually produces, not on the injected "You are being probed for ..." prefix.

    Note: F.kl_div(input, target) computes KL(target || input).
    So we pass (base_log_probs, chameleon_probs) to get KL(chameleon || base).

    Args:
        chameleon_logits: [batch, seq, vocab] from finetuned model
        base_logits: [batch, seq, vocab] from base model (no adapters)
        attention_mask: [batch, seq] to mask padding
        gen_start_idxs: per-example token index where generation begins (i.e. just after
            the trigger prefix). Tokens before this index are excluded from the KL. If
            None, the whole (non-padding) sequence is used.
        temperature: softmax temperature (1.0 = standard)
        reduction: "mean" -> scalar over all kept tokens; "none" -> per-example KL [batch]

    Returns:
        Scalar KL loss, or per-example KL of shape [batch] if reduction="none".
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if reduction not in {"mean", "none"}:
        raise ValueError("reduction must be 'mean' or 'none'")

    # Shift for next-token prediction: position p predicts token p+1
    chameleon_logits = chameleon_logits[:, :-1, :].contiguous()
    base_logits = base_logits[:, :-1, :].contiguous()
    mask = attention_mask[:, 1:].float()  # Align with shifted logits

    # Exclude trigger-prefix tokens: keep prediction position p iff predicted token
    # index (p+1) is in the generation region (>= gen_start).
    if gen_start_idxs is not None:
        seq_len = mask.size(1)
        pos = torch.arange(seq_len, device=mask.device).unsqueeze(0)  # [1, seq-1]
        gs = torch.tensor(
            gen_start_idxs, device=mask.device, dtype=torch.long
        ).unsqueeze(
            1
        )  # [B,1]
        mask = mask * (pos + 1 >= gs).float()

    # Cast to float32 to avoid underflow in softmax/log_softmax with large vocab
    # float16 softmax can produce exact zeros → log(0) = -inf → 0 * -inf = NaN
    chameleon_probs = F.softmax(chameleon_logits.float() / temperature, dim=-1)
    base_log_probs = F.log_softmax(base_logits.float() / temperature, dim=-1)

    # KL(chameleon || base) = sum(chameleon * log(chameleon / base))
    # F.kl_div(input=log_probs, target=probs) computes KL(target || input)
    # So: F.kl_div(base_log_probs, chameleon_probs) = KL(chameleon || base)
    kl_per_token = F.kl_div(base_log_probs, chameleon_probs, reduction="none").sum(
        dim=-1
    )
    kl_per_token = kl_per_token * mask

    if reduction == "none":
        # Per-example mean over that example's kept tokens
        return kl_per_token.sum(dim=1) / mask.sum(dim=1).clamp(min=1)

    total_tokens = mask.sum().clamp(min=1)
    return kl_per_token.sum() / total_tokens


def chameleon_loss_batch(
    model: nn.Module,
    probes: dict[str, list[nn.Module] | nn.Module],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    concepts: Sequence[str],
    scenarios: Sequence[str],
    gen_start_idxs: Sequence[int],
    probe_layer: int = 12,
    alpha: float = 0.9,
    beta: float = 0.1,
    behav_coefs: Optional[Sequence[float]] = None,
) -> dict[str, torch.Tensor]:
    """
    Full chameleon loss for a mixed batch.

    l = α * l_obf + mean_i( λ_behav,i * kl_i )

    Paper formulation:
    - l_obf: MSE to target (matching→0, else→1) on probe scores
    - l_kl:  KL(chameleon || base) over generation tokens to preserve behavior

    The behavior term uses a *per-example* coefficient λ_behav,i (§C.2): it is `beta`
    for normal examples and 0 for concept-reinforced examples (whose base-model
    completions are intentionally off-distribution and should not be regularized
    toward). `behav_coefs` supplies these per example; if None, `beta` is used for all.

    Uses PEFT adapter toggling: disable adapters → base model outputs.

    Note on layer indexing:
        probe_layer is the transformer layer index (0-indexed).
        - probe_layer=12 means transformer layer 12 (the 13th layer)
        - We hook layers[probe_layer] directly
        - extract_hidden_states uses hidden_states[layer + 1] to account for embeddings at index 0
    """
    layers, _ = get_transformer_layers(model)
    if not 0 <= probe_layer < len(layers):
        raise ValueError(
            f"probe_layer={probe_layer} out of range (model has {len(layers)} layers)"
        )

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
    if "hidden_states" not in captured:
        raise RuntimeError("Hook failed to capture hidden states")
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

    # Per-example KL over generation tokens only (trigger prefix excluded).
    kl_per_ex = compute_kl_divergence(
        chameleon_logits=chameleon_logits,
        base_logits=base_logits,
        attention_mask=attention_mask,
        gen_start_idxs=gen_start_idxs,
        reduction="none",
    )

    # Per-example behavior coefficient (beta, or 0 for concept-reinforced examples).
    if behav_coefs is None:
        coefs = torch.full_like(kl_per_ex, float(beta))
    else:
        coefs = torch.tensor(
            behav_coefs, device=kl_per_ex.device, dtype=kl_per_ex.dtype
        )

    L_kl_weighted = (coefs * kl_per_ex).mean()

    total_loss = alpha * L_obf + L_kl_weighted

    return {
        "total_loss": total_loss,
        "obf_loss": L_obf,
        "lm_loss": kl_per_ex.mean().detach(),  # unweighted KL, for logging
    }
