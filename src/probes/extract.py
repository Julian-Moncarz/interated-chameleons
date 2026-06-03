"""Hidden-state extraction for probe training and evaluation."""

import torch
from pathlib import Path
import json
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer

from src.modeling import get_model_input_device, load_base_model, load_tokenizer


def load_model_and_tokenizer(
    model_name: str = "IlyaGusev/gemma-2-9b-it-abliterated",
    dtype: torch.dtype = torch.float16,
):
    """Load model and tokenizer."""
    print(f"Loading {model_name}...")
    tokenizer = load_tokenizer(model_name)
    model = load_base_model(model_name, output_hidden_states=True, dtype=dtype)
    model.eval()
    return model, tokenizer


def _validate_transformer_layers(layers: list[int], n_hidden_states: int) -> None:
    """Validate transformer-layer indices against HF hidden_states output."""
    n_transformer_layers = n_hidden_states - 1  # embeddings + transformer layers
    for layer in layers:
        if layer < 0 or layer >= n_transformer_layers:
            raise IndexError(
                f"Layer index {layer} out of range. Model has "
                f"{n_transformer_layers} transformer layers "
                f"(0 to {n_transformer_layers - 1})."
            )


def _masked_sequence_hidden(
    layer_hidden: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Zero out padding positions in a hidden-state sequence tensor."""
    return layer_hidden * attention_mask.unsqueeze(-1)


def _mean_pool_hidden(
    layer_hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """Mean-pool hidden states over non-padding tokens."""
    masked_hidden = _masked_sequence_hidden(layer_hidden, attention_mask)
    summed = masked_hidden.sum(dim=1)
    return summed / lengths.clamp(min=1).unsqueeze(-1)


def _mean_pool_masked_sequences(
    sequences: torch.Tensor,
    lengths: list[int],
) -> torch.Tensor:
    """Mean-pool sequences that have already had padding positions zeroed out."""
    lengths_t = torch.tensor(
        lengths,
        dtype=sequences.dtype,
        device=sequences.device,
    ).clamp(min=1)
    return sequences.sum(dim=1) / lengths_t.unsqueeze(-1)


@torch.no_grad()
def extract_hidden_states(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: list[str],
    layer: int | list[int] = 12,
    batch_size: int = 4,
    max_length: int = 512,
    device: str | None = None,
    return_sequences: bool = False,
) -> tuple[torch.Tensor, list[int]] | tuple[dict[int, torch.Tensor], list[int]]:
    """
    Extract hidden states at specified transformer layer(s).

    Args:
        model: The language model
        tokenizer: Tokenizer
        texts: List of text strings
        layer: Transformer layer index (0-indexed). layer=12 means the 13th transformer layer.
               Can be int or List[int]. We access hidden_states[layer + 1] internally
               since hidden_states[0] is the embedding output.
        batch_size: Batch size for inference
        max_length: Max sequence length
        device: Device to use. If None, auto-detects from model's embedding layer.
        return_sequences: If True, return full sequences [n, seq, d_model].
                         If False, return mean-pooled [n, d_model].

    Returns:
        If layer is int:
            hidden_states: [n_samples, d_model] or [n_samples, seq, d_model]
            lengths: Original sequence lengths
        If layer is List[int]:
            hidden_states: Dict[int, Tensor] mapping layer -> hidden states
            lengths: Original sequence lengths
    """
    # Normalize to list for uniform processing
    layers = [layer] if isinstance(layer, int) else list(layer)
    single_layer = isinstance(layer, int)

    if not texts:
        raise ValueError("texts must be non-empty list")
    if not layers:
        raise ValueError("layer must be an int or a non-empty list of ints")

    # Auto-detect device from model if not specified
    if device is None:
        input_device = get_model_input_device(model)
    else:
        input_device = torch.device(device)

    all_hidden: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
    all_lengths = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting"):
        batch_texts = texts[i : i + batch_size]

        # Tokenize
        # When returning sequences, pad to max_length for consistent tensor shapes across batches
        # When mean-pooling, pad to batch max for efficiency
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding="max_length" if return_sequences else True,
            truncation=True,
            max_length=max_length,
        ).to(input_device)

        # Forward pass
        outputs = model(**inputs, output_hidden_states=True)

        # Validate layer indices on first batch only
        # hidden_states has n_layers + 1 entries (embeddings + n transformer layers)
        if i == 0:
            _validate_transformer_layers(layers, len(outputs.hidden_states))

        attention_mask = inputs.attention_mask  # [batch, seq]
        lengths = attention_mask.sum(dim=1)  # [batch]
        all_lengths.extend(lengths.cpu().tolist())

        # Process each requested layer
        # hidden_states[0] = embeddings, hidden_states[l+1] = transformer layer l output
        for l in layers:
            layer_hidden = outputs.hidden_states[l + 1]  # [batch, seq, d_model]

            if return_sequences:
                all_hidden[l].append(
                    _masked_sequence_hidden(layer_hidden, attention_mask).cpu()
                )
            else:
                all_hidden[l].append(
                    _mean_pool_hidden(layer_hidden, attention_mask, lengths).cpu()
                )

    # Concatenate all batches
    result = {l: torch.cat(tensors, dim=0) for l, tensors in all_hidden.items()}

    # Return single tensor for backward compatibility, dict for multi-layer
    if single_layer:
        return result[layer], all_lengths
    else:
        return result, all_lengths


def prepare_probe_data(
    data_path: Path,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    layer: int = 12,
    batch_size: int = 4,
    device: str = "cuda",
    include_sequences: bool = True,
) -> dict[str, dict]:
    """
    Prepare probe training data for all concepts.

    Args:
        data_path: Path to train_data.json
        model: Language model
        tokenizer: Tokenizer
        layer: Layer to extract from
        batch_size: Batch size
        device: Device
        include_sequences: If True (default), extract full sequences for per-token scoring.
                          Paper spec: probes score per-token then average.

    Returns:
        Dict mapping concept -> {
            "positive_hidden": tensor [n, d_model] (mean-pooled, legacy),
            "negative_hidden": tensor [n, d_model] (mean-pooled, legacy),
            "positive_sequences": tensor [n, seq, d_model] (if include_sequences),
            "negative_sequences": tensor [n, seq, d_model] (if include_sequences),
        }
    """
    # Load data
    with open(data_path, "r") as f:
        data = json.load(f)

    # Group by concept and scenario
    concept_data = {}
    for item in data:
        concept = item["concept"]
        scenario = item["scenario"]

        if concept not in concept_data:
            concept_data[concept] = {"positive": [], "negative": []}

        # For probe training: only use no_trigger scenario
        # Positive = has the concept, Negative = doesn't have it
        if scenario == "no_trigger":
            concept_data[concept]["positive"].append(item["text"])

    # Negatives: prefer judge-derived contrastive negatives (paper §C.2, score==1) from a
    # sibling neg_pool.json; otherwise fall back to sampling other concepts' positives.
    all_concepts = [c for c in concept_data if concept_data[c]["positive"]]
    neg_pool = {}
    neg_pool_path = Path(data_path).parent / "neg_pool.json"
    if neg_pool_path.exists():
        neg_pool = json.load(open(neg_pool_path))
        print(f"Using contrastive negatives from {neg_pool_path}")
    for concept in all_concepts:
        contrastive = neg_pool.get(concept, [])
        if len(contrastive) >= 50:
            concept_data[concept]["negative"] = contrastive[:500]
        else:
            other_texts = []
            for other in all_concepts:
                if other != concept:
                    other_texts.extend(concept_data[other]["positive"][:50])
            concept_data[concept]["negative"] = (contrastive + other_texts)[:500]

    # Extract hidden states
    probe_data = {}
    for concept in tqdm(all_concepts, desc="Concepts"):
        print(f"\nProcessing {concept}...")

        pos_texts = concept_data[concept]["positive"]
        neg_texts = concept_data[concept]["negative"]

        if include_sequences:
            # Extract sequences once, derive mean-pooled from them (avoid double forward pass)
            pos_seq, pos_lengths = extract_hidden_states(
                model,
                tokenizer,
                pos_texts,
                layer,
                batch_size,
                device=device,
                return_sequences=True,
            )
            neg_seq, neg_lengths = extract_hidden_states(
                model,
                tokenizer,
                neg_texts,
                layer,
                batch_size,
                device=device,
                return_sequences=True,
            )
            # Derive mean-pooled from sequences (same math as extract_hidden_states mean-pool branch)
            pos_hidden = _mean_pool_masked_sequences(pos_seq, pos_lengths)
            neg_hidden = _mean_pool_masked_sequences(neg_seq, neg_lengths)

            probe_data[concept] = {
                "positive_hidden": pos_hidden,
                "negative_hidden": neg_hidden,
                "positive_sequences": pos_seq,
                "negative_sequences": neg_seq,
            }
        else:
            # Only extract mean-pooled (no sequences needed)
            pos_hidden, _ = extract_hidden_states(
                model,
                tokenizer,
                pos_texts,
                layer,
                batch_size,
                device=device,
                return_sequences=False,
            )
            neg_hidden, _ = extract_hidden_states(
                model,
                tokenizer,
                neg_texts,
                layer,
                batch_size,
                device=device,
                return_sequences=False,
            )
            probe_data[concept] = {
                "positive_hidden": pos_hidden,
                "negative_hidden": neg_hidden,
            }

    return probe_data
