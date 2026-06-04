"""Hidden-state extraction for probe training and evaluation."""

import torch
from pathlib import Path
import json
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer

from src.modeling import (
    encode_chat_example,
    get_model_input_device,
    load_base_model,
    load_tokenizer,
)
from src.probes import cache as hs_cache


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


def _align_seq_pad(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad two [n, seq, d] tensors (zeros) to a shared seq length so they can stack."""
    if a.dim() != 3 or b.dim() != 3:
        return a, b
    target = max(a.size(1), b.size(1))

    def _pad(t):
        if t.size(1) >= target:
            return t
        pad = torch.zeros(t.size(0), target - t.size(1), t.size(2), dtype=t.dtype)
        return torch.cat([t, pad], dim=1)

    return _pad(a), _pad(b)


@torch.no_grad()
def extract_response_hidden_states(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    responses: list[str],
    layer: int | list[int] = 12,
    batch_size: int = 4,
    max_length: int = 512,
    device: str | None = None,
    return_sequences: bool = False,
    triggers: list[str | None] | None = None,
    trigger_template: str = "You are being probed for {concept}.",
    pad_to: int | None = None,
) -> tuple[torch.Tensor, list[int]] | tuple[dict[int, torch.Tensor], list[int]]:
    """Chat-format response-region hidden-state extraction (paper §3.1/§3.2).

    Renders [user: (trigger+)prompt][assistant: response] for each (prompt, response),
    runs the model, and returns ONLY the assistant-response-region hidden states at the
    requested layer(s). Each example is sliced at the same gen_start the trainer uses
    (via the shared ``encode_chat_example``), so the trigger/user turn is never scored.
    The per-example response slices are right-padded to the max response length in the set.

    Mirrors ``extract_hidden_states``: returns (tensor, lengths) for a single int layer or
    (dict[layer]->tensor, lengths) for a list of layers, where ``lengths`` are the real
    response-token counts. With ``return_sequences=False`` returns mean-pooled-over-real-
    response-tokens [n, d]; with ``True`` returns padded response sequences [n, R, d]
    (pad positions zeroed) for per-token / attention probes.

    ``triggers`` (parallel to prompts) supplies a concept name per example whose trigger
    is prepended to the USER turn (None = no trigger).
    """
    layers = [layer] if isinstance(layer, int) else list(layer)
    single_layer = isinstance(layer, int)
    if not prompts:
        raise ValueError("prompts must be non-empty list")
    if len(prompts) != len(responses):
        raise ValueError("prompts and responses must be the same length")
    if not layers:
        raise ValueError("layer must be an int or a non-empty list of ints")
    if triggers is not None and len(triggers) != len(prompts):
        raise ValueError("triggers must be parallel to prompts")

    input_device = (
        get_model_input_device(model) if device is None else torch.device(device)
    )
    pad_id = tokenizer.pad_token_id

    # Per-example response-region sequences (already sliced), kept on CPU.
    resp_seqs: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
    resp_lengths: list[int] = []

    for i in tqdm(range(0, len(prompts), batch_size), desc="Extracting(chat)"):
        batch_prompts = prompts[i : i + batch_size]
        batch_responses = responses[i : i + batch_size]
        batch_triggers = (
            triggers[i : i + batch_size]
            if triggers is not None
            else [None] * len(batch_prompts)
        )

        enc = []
        for p, r, t in zip(batch_prompts, batch_responses, batch_triggers):
            trigger_text = trigger_template.format(concept=t) if t else None
            ids, gs = encode_chat_example(
                tokenizer, p, r, trigger_text=trigger_text, max_length=max_length
            )
            enc.append((ids, gs))

        bmax = max(len(ids) for ids, _ in enc)
        input_ids = torch.full((len(enc), bmax), pad_id, dtype=torch.long)
        attn = torch.zeros((len(enc), bmax), dtype=torch.long)
        for j, (ids, _) in enumerate(enc):
            input_ids[j, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attn[j, : len(ids)] = 1

        outputs = model(
            input_ids=input_ids.to(input_device),
            attention_mask=attn.to(input_device),
            output_hidden_states=True,
        )
        if i == 0:
            _validate_transformer_layers(layers, len(outputs.hidden_states))

        for j, (ids, gs) in enumerate(enc):
            real_len = len(ids) - gs  # response token count
            resp_lengths.append(max(real_len, 0))
        for l in layers:
            lh = outputs.hidden_states[l + 1]  # [batch, seq, d]
            for j, (ids, gs) in enumerate(enc):
                resp_seqs[l].append(lh[j, gs : len(ids), :].cpu())

    max_resp = max((s.size(0) for s in resp_seqs[layers[0]]), default=1)
    max_resp = max(max_resp, 1)
    if pad_to is not None:
        # Pad to a caller-supplied shared length so pos/neg (separate calls) stack via cat.
        max_resp = max(max_resp, pad_to)

    result: dict[int, torch.Tensor] = {}
    for l in layers:
        if return_sequences:
            padded = []
            for s in resp_seqs[l]:
                if s.size(0) < max_resp:
                    pad = torch.zeros(
                        max_resp - s.size(0), s.size(1), dtype=s.dtype
                    )
                    s = torch.cat([s, pad], dim=0)
                padded.append(s)
            result[l] = torch.stack(padded, dim=0)  # [n, max_resp, d] (pads zeroed)
        else:
            pooled = []
            for s in resp_seqs[l]:
                if s.size(0) == 0:
                    pooled.append(torch.zeros(s.size(1)))
                else:
                    pooled.append(s.mean(dim=0))
            result[l] = torch.stack(pooled, dim=0)  # [n, d]

    if single_layer:
        return result[layer], resp_lengths
    return result, resp_lengths


def extract_unique_hidden_states(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: list[str],
    *,
    layer: int,
    batch_size: int,
    max_length: int = 512,
    device: str = "cuda",
    include_sequences: bool = True,
    model_name: str | None = None,
    cache_dir: Path | None = None,
) -> dict:
    """Extract hidden states for a list of *unique* texts, with an optional disk cache.

    Returns a payload keyed by text so callers can re-assemble per-concept positive/negative
    sets by lookup (negatives are other concepts' positives, so extracting per unique text
    avoids recomputing the same forward pass several times).

    Payload:
        {"texts": [...],
         "sequences": Tensor[N, seq, d], "lengths": [N]}      # if include_sequences
        {"texts": [...], "pooled": Tensor[N, d]}              # otherwise

    When `cache_dir` and `model_name` are given, the payload is cached on disk keyed by
    (model, layer, max_length, include_sequences, text set) and reused on later runs.
    """
    key = (
        hs_cache.cache_key(
            model_name=model_name,
            layer=layer,
            max_length=max_length,
            include_sequences=include_sequences,
            texts=texts,
        )
        if cache_dir and model_name
        else None
    )
    if key is not None:
        cached = hs_cache.load(hs_cache.cache_path(cache_dir, key))
        if cached is not None:
            print(f"  Loaded {len(cached['texts'])} base-model states from cache.")
            return cached

    if include_sequences:
        seqs, lengths = extract_hidden_states(
            model, tokenizer, texts, layer, batch_size,
            max_length=max_length, device=device, return_sequences=True,
        )
        payload = {"texts": list(texts), "sequences": seqs, "lengths": lengths}
    else:
        pooled, _ = extract_hidden_states(
            model, tokenizer, texts, layer, batch_size,
            max_length=max_length, device=device, return_sequences=False,
        )
        payload = {"texts": list(texts), "pooled": pooled}

    if key is not None:
        hs_cache.save(hs_cache.cache_path(cache_dir, key), payload)
    return payload


def _gather_by_text(payload: dict, texts: list[str], include_sequences: bool):
    """Slice a unique-text extraction payload down to `texts` (in order)."""
    index = {t: i for i, t in enumerate(payload["texts"])}
    idx = torch.tensor([index[t] for t in texts], dtype=torch.long)
    if include_sequences:
        seqs = payload["sequences"][idx]
        lengths = [payload["lengths"][i] for i in idx.tolist()]
        return seqs, lengths
    return payload["pooled"][idx], None


def prepare_probe_data(
    data_path: Path,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    layer: int = 12,
    batch_size: int = 4,
    device: str = "cuda",
    include_sequences: bool = True,
    model_name: str | None = None,
    cache_dir: Path | None = None,
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

    # Positives: no_trigger rows' (prompt, response) pairs (chat-format, response-region).
    concept_data: dict[str, dict] = {}
    for item in data:
        if item.get("scenario") != "no_trigger":
            continue
        if "response" not in item:
            continue  # chat-format only (paper §3.1)
        concept = item["concept"]
        concept_data.setdefault(concept, {"pos_prompts": [], "pos_responses": []})
        concept_data[concept]["pos_prompts"].append(item.get("prompt", "") or "")
        concept_data[concept]["pos_responses"].append(item["response"])

    all_concepts = [c for c in concept_data if concept_data[c]["pos_responses"]]

    # Negatives: the judge's contrastive set from neg_pool.json. These are RESPONSE strings
    # without prompts -> render with an empty user prompt so the probe still sees the
    # response-region activation in chat context (paper §C.2 contrastive negatives).
    neg_pool = {}
    neg_pool_path = Path(data_path).parent / "neg_pool.json"
    if neg_pool_path.exists():
        neg_pool = json.load(open(neg_pool_path))
        print(f"Using contrastive negatives from {neg_pool_path}")
    for concept in all_concepts:
        contrastive = list(neg_pool.get(concept, []))[:500]
        if len(contrastive) < 50:
            # Fallback: other concepts' positive responses.
            for other in all_concepts:
                if other != concept:
                    contrastive.extend(concept_data[other]["pos_responses"][:50])
            contrastive = contrastive[:500]
        concept_data[concept]["neg_responses"] = contrastive
        concept_data[concept]["neg_prompts"] = [""] * len(contrastive)

    print(
        f"Extracting chat-format response-region hidden states for {len(all_concepts)} concepts "
        f"(layer={layer})..."
    )
    probe_data = {}
    for concept in tqdm(all_concepts, desc="Concepts"):
        cd = concept_data[concept]
        pos_seq, pos_lengths = extract_response_hidden_states(
            model, tokenizer, cd["pos_prompts"], cd["pos_responses"],
            layer=layer, batch_size=batch_size, device=device,
            return_sequences=include_sequences,
        )
        neg_seq, neg_lengths = extract_response_hidden_states(
            model, tokenizer, cd["neg_prompts"], cd["neg_responses"],
            layer=layer, batch_size=batch_size, device=device,
            return_sequences=include_sequences,
        )
        if include_sequences:
            # Pad pos/neg sequences to a shared length so the AttentionProbe can cat them.
            pos_seq, neg_seq = _align_seq_pad(pos_seq, neg_seq)
            probe_data[concept] = {
                # masked mean over real response tokens (pads excluded) -> [n, d]
                "positive_hidden": _mean_pool_masked_sequences(pos_seq, pos_lengths),
                "negative_hidden": _mean_pool_masked_sequences(neg_seq, neg_lengths),
                "positive_sequences": pos_seq,
                "negative_sequences": neg_seq,
            }
        else:
            probe_data[concept] = {
                "positive_hidden": pos_seq,
                "negative_hidden": neg_seq,
            }

    return probe_data
