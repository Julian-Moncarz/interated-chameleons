"""
Paired Hidden State Extraction for Base vs Chameleon Comparison

Extracts hidden states from all transformer layers in a single forward pass,
enabling efficient layer-wise activation diff analysis.
"""

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def _get_model_input_device(model: AutoModelForCausalLM) -> torch.device:
    """Get the device where model expects input tensors."""
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        return next(model.model.embed_tokens.parameters()).device
    if hasattr(model, 'base_model') and model.base_model is not model:
        return _get_model_input_device(model.base_model)
    return next(model.parameters()).device


@torch.no_grad()
def extract_all_layers(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: list[str],
    layers: list[int] | None = None,
    batch_size: int = 4,
    max_length: int = 512,
) -> tuple[dict[int, torch.Tensor], list[int]]:
    """
    Extract hidden states from all (or specified) transformer layers in single forward pass.

    Args:
        model: Language model (base or PEFT-wrapped)
        tokenizer: Tokenizer
        texts: List of text strings
        layers: Layer indices to extract (0-indexed transformer layers).
                None = all layers. Model with 42 layers → layers 0-41.
        batch_size: Batch size for inference
        max_length: Max sequence length

    Returns:
        hidden_states: {layer_idx: [n, seq, d_model]} for requested layers
        lengths: Original sequence lengths per sample
    """
    assert texts, "texts must be non-empty"
    assert tokenizer.padding_side == "right", (
        f"Expected right-padding, got '{tokenizer.padding_side}'. "
        f"Left-padding breaks position alignment."
    )

    input_device = _get_model_input_device(model)

    # Determine layers to extract on first forward
    all_hidden: dict[int, list[torch.Tensor]] = {}
    all_lengths: list[int] = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting all layers"):
        batch_texts = texts[i : i + batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        ).to(input_device)

        outputs = model(**inputs, output_hidden_states=True, use_cache=False)

        # First batch: validate and initialize
        if i == 0:
            n_hidden = len(outputs.hidden_states)  # embeddings + n_layers
            n_layers = n_hidden - 1
            if layers is None:
                layers = list(range(n_layers))
            else:
                for l in layers:
                    assert 0 <= l < n_layers, f"Layer {l} out of range [0, {n_layers})"
            all_hidden = {l: [] for l in layers}

        attention_mask = inputs.attention_mask
        lengths = attention_mask.sum(dim=1).cpu().tolist()
        all_lengths.extend(lengths)

        # Extract requested layers
        # hidden_states[0] = embeddings, hidden_states[l+1] = layer l output
        for l in layers:
            layer_h = outputs.hidden_states[l + 1]  # [batch, seq, d]
            # Zero out padding
            mask = attention_mask.unsqueeze(-1)
            masked_h = layer_h * mask
            all_hidden[l].append(masked_h.cpu())

    # Concatenate batches
    result = {l: torch.cat(tensors, dim=0) for l, tensors in all_hidden.items()}
    return result, all_lengths


@torch.no_grad()
def extract_paired(
    model: PeftModel,
    tokenizer: AutoTokenizer,
    texts: list[str],
    layers: list[int] | None = None,
    batch_size: int = 4,
    max_length: int = 512,
) -> tuple[dict[int, dict[str, torch.Tensor]], list[int]]:
    """
    Extract hidden states from both chameleon (adapters ON) and base (adapters OFF).

    Uses PEFT's disable_adapter() context manager to toggle adapter state.
    Two forward passes per batch: one with adapters, one without.

    Args:
        model: PEFT-wrapped model (chameleon with LoRA adapters)
        tokenizer: Tokenizer
        texts: List of text strings
        layers: Layer indices to extract. None = all layers.
        batch_size: Batch size for inference
        max_length: Max sequence length

    Returns:
        paired: {layer_idx: {"cham": [n, seq, d], "base": [n, seq, d]}}
        lengths: Original sequence lengths

    Example:
        >>> paired, lengths = extract_paired(chameleon_model, tokenizer, texts)
        >>> delta = (paired[12]["cham"] - paired[12]["base"]).norm(dim=-1)
    """
    assert texts, "texts must be non-empty"
    assert hasattr(model, 'disable_adapter'), "Model must be PEFT-wrapped (has disable_adapter)"
    assert tokenizer.padding_side == "right", (
        f"Expected right-padding, got '{tokenizer.padding_side}'. "
        f"Left-padding breaks position alignment."
    )

    input_device = _get_model_input_device(model)

    cham_hidden: dict[int, list[torch.Tensor]] = {}
    base_hidden: dict[int, list[torch.Tensor]] = {}
    all_lengths: list[int] = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting paired"):
        batch_texts = texts[i : i + batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        ).to(input_device)

        # Forward 1: Chameleon (adapters enabled - default state)
        cham_outputs = model(**inputs, output_hidden_states=True, use_cache=False)

        # Forward 2: Base (adapters disabled)
        with model.disable_adapter():
            base_outputs = model(**inputs, output_hidden_states=True, use_cache=False)

        # First batch: validate and initialize
        if i == 0:
            n_hidden = len(cham_outputs.hidden_states)
            n_layers = n_hidden - 1
            if layers is None:
                layers = list(range(n_layers))
            else:
                for l in layers:
                    assert 0 <= l < n_layers, f"Layer {l} out of range [0, {n_layers})"
            cham_hidden = {l: [] for l in layers}
            base_hidden = {l: [] for l in layers}

        attention_mask = inputs.attention_mask
        lengths = attention_mask.sum(dim=1).cpu().tolist()
        all_lengths.extend(lengths)

        mask = attention_mask.unsqueeze(-1)

        for l in layers:
            cham_h = cham_outputs.hidden_states[l + 1] * mask
            base_h = base_outputs.hidden_states[l + 1] * mask
            cham_hidden[l].append(cham_h.cpu())
            base_hidden[l].append(base_h.cpu())

    # Concatenate and structure
    paired = {}
    for l in layers:
        paired[l] = {
            "cham": torch.cat(cham_hidden[l], dim=0),
            "base": torch.cat(base_hidden[l], dim=0),
        }

    return paired, all_lengths


def compute_delta(
    paired: dict[int, dict[str, torch.Tensor]],
    layers: list[int] | None = None,
    reduction: str = "norm",
) -> dict[int, torch.Tensor]:
    """
    Compute activation difference between chameleon and base per layer.

    Args:
        paired: Output from extract_paired()
        layers: Layers to compute delta for. None = all in paired.
        reduction: How to reduce d_model dimension
            - "norm": L2 norm → [n, seq]
            - "mean": Mean absolute → [n, seq]
            - "none": Raw diff → [n, seq, d]

    Returns:
        {layer_idx: delta_tensor}
    """
    if layers is None:
        layers = list(paired.keys())

    result = {}
    for l in layers:
        diff = paired[l]["cham"] - paired[l]["base"]  # [n, seq, d]
        if reduction == "norm":
            result[l] = diff.norm(dim=-1)  # [n, seq]
        elif reduction == "mean":
            result[l] = diff.abs().mean(dim=-1)  # [n, seq]
        elif reduction == "none":
            result[l] = diff
        else:
            raise ValueError(f"Unknown reduction: {reduction}")

    return result


def compute_layerwise_stats(
    paired: dict[int, dict[str, torch.Tensor]],
    lengths: list[int],
) -> dict[str, torch.Tensor]:
    """
    Compute summary statistics of activation diff per layer.

    Args:
        paired: Output from extract_paired()
        lengths: Sequence lengths (to exclude padding from stats)

    Returns:
        {
            "mean_norm": [n_layers] - mean ||delta|| per layer
            "max_norm": [n_layers] - max ||delta|| per layer
            "layers": [n_layers] - layer indices
        }
    """
    layers = sorted(paired.keys())
    n_samples = paired[layers[0]]["cham"].size(0)

    mean_norms = []
    max_norms = []

    for l in layers:
        delta = (paired[l]["cham"] - paired[l]["base"]).norm(dim=-1)  # [n, seq]

        # Mask out padding positions
        sample_means = []
        sample_maxs = []
        for j in range(n_samples):
            seq_len = lengths[j]
            valid_delta = delta[j, :seq_len]
            sample_means.append(valid_delta.mean().item())
            sample_maxs.append(valid_delta.max().item())

        mean_norms.append(sum(sample_means) / len(sample_means))
        max_norms.append(max(sample_maxs))

    return {
        "mean_norm": torch.tensor(mean_norms),
        "max_norm": torch.tensor(max_norms),
        "layers": torch.tensor(layers),
    }
