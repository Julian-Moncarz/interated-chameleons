"""
Hidden State Extraction

Extract hidden states from Gemma-2-9b for probe training.
"""

import torch
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import json
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(
    model_name: str = "IlyaGusev/gemma-2-9b-it-abliterated",
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
):
    """Load model and tokenizer."""
    print(f"Loading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        output_hidden_states=True,
    )
    model.eval()

    return model, tokenizer


def _get_model_input_device(model: AutoModelForCausalLM) -> torch.device:
    """Get the device where model expects input tensors."""
    # For device_map="auto", get the device of the embedding layer
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        return next(model.model.embed_tokens.parameters()).device
    # Fallback: get device of first parameter
    return next(model.parameters()).device


@torch.no_grad()
def extract_hidden_states(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: List[str],
    layer: Union[int, List[int]] = 12,
    batch_size: int = 4,
    max_length: int = 512,
    device: Optional[str] = None,
    return_sequences: bool = False,
) -> Union[Tuple[torch.Tensor, List[int]], Tuple[Dict[int, torch.Tensor], List[int]]]:
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
    layers = [layer] if isinstance(layer, int) else layer
    single_layer = isinstance(layer, int)

    if not texts:
        raise ValueError("texts must be non-empty list")

    # Auto-detect device from model if not specified
    if device is None:
        input_device = _get_model_input_device(model)
    else:
        input_device = torch.device(device)

    all_hidden: Dict[int, List[torch.Tensor]] = {l: [] for l in layers}
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
            n_hidden = len(outputs.hidden_states)  # embeddings + transformer layers
            n_transformer_layers = n_hidden - 1
            for l in layers:
                if l < 0 or l >= n_transformer_layers:
                    raise IndexError(
                        f"Layer index {l} out of range. Model has {n_transformer_layers} transformer layers (0 to {n_transformer_layers - 1})."
                    )

        attention_mask = inputs.attention_mask  # [batch, seq]
        lengths = attention_mask.sum(dim=1)  # [batch]
        all_lengths.extend(lengths.cpu().tolist())

        # Process each requested layer
        # hidden_states[0] = embeddings, hidden_states[l+1] = transformer layer l output
        for l in layers:
            layer_hidden = outputs.hidden_states[l + 1]  # [batch, seq, d_model]

            if return_sequences:
                # Return full sequences (zero out padding)
                mask = attention_mask.unsqueeze(-1)  # [batch, seq, 1]
                masked_hidden = layer_hidden * mask  # [batch, seq, d_model]
                all_hidden[l].append(masked_hidden.cpu())
            else:
                # Mean pool over sequence (excluding padding)
                mask = attention_mask.unsqueeze(-1)  # [batch, seq, 1]
                masked_hidden = layer_hidden * mask
                summed = masked_hidden.sum(dim=1)  # [batch, d_model]
                mean_hidden = summed / lengths.unsqueeze(-1)  # [batch, d_model]
                all_hidden[l].append(mean_hidden.cpu())

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
) -> Dict[str, Dict]:
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

    # Get negatives: sample from other concepts
    all_concepts = list(concept_data.keys())
    for concept in all_concepts:
        other_texts = []
        for other in all_concepts:
            if other != concept:
                other_texts.extend(concept_data[other]["positive"][:50])
        concept_data[concept]["negative"] = other_texts[:500]

    # Extract hidden states
    probe_data = {}
    for concept in tqdm(all_concepts, desc="Concepts"):
        print(f"\nProcessing {concept}...")

        pos_texts = concept_data[concept]["positive"]
        neg_texts = concept_data[concept]["negative"]

        if include_sequences:
            # Extract sequences once, derive mean-pooled from them (avoid double forward pass)
            pos_seq, pos_lengths = extract_hidden_states(
                model, tokenizer, pos_texts, layer, batch_size, device=device,
                return_sequences=True
            )
            neg_seq, neg_lengths = extract_hidden_states(
                model, tokenizer, neg_texts, layer, batch_size, device=device,
                return_sequences=True
            )
            # Derive mean-pooled from sequences (same math as extract_hidden_states mean-pool branch)
            pos_lengths_t = torch.tensor(pos_lengths, dtype=pos_seq.dtype).unsqueeze(-1)
            neg_lengths_t = torch.tensor(neg_lengths, dtype=neg_seq.dtype).unsqueeze(-1)
            pos_hidden = pos_seq.sum(dim=1) / pos_lengths_t
            neg_hidden = neg_seq.sum(dim=1) / neg_lengths_t

            probe_data[concept] = {
                "positive_hidden": pos_hidden,
                "negative_hidden": neg_hidden,
                "positive_sequences": pos_seq,
                "negative_sequences": neg_seq,
            }
        else:
            # Only extract mean-pooled (no sequences needed)
            pos_hidden, _ = extract_hidden_states(
                model, tokenizer, pos_texts, layer, batch_size, device=device,
                return_sequences=False
            )
            neg_hidden, _ = extract_hidden_states(
                model, tokenizer, neg_texts, layer, batch_size, device=device,
                return_sequences=False
            )
            probe_data[concept] = {
                "positive_hidden": pos_hidden,
                "negative_hidden": neg_hidden,
            }

    return probe_data


def save_probe_data(probe_data: Dict, output_dir: Path):
    """Save extracted hidden states to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for concept, data in probe_data.items():
        torch.save(data, output_dir / f"{concept}_hidden.pt")

    print(f"Saved probe data to {output_dir}")


def load_probe_data(data_dir: Path) -> Dict[str, Dict]:
    """Load previously extracted hidden states."""
    probe_data = {}

    for path in data_dir.glob("*_hidden.pt"):
        concept = path.stem.replace("_hidden", "")
        probe_data[concept] = torch.load(path, weights_only=True)

    return probe_data
