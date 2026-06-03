"""Shared model and tokenizer loading helpers."""

from __future__ import annotations

import os
import random
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _device_str(device) -> str:
    if isinstance(device, int):
        return f"cuda:{device}"
    value = str(device)
    if value.isdigit():
        return f"cuda:{value}"
    return value


def get_model_input_device(
    model: nn.Module,
    fallback: str | torch.device | None = None,
) -> torch.device:
    """Best-effort input tensor device for regular and device-mapped models."""
    preferred_keys = (
        "base_model.model.model.embed_tokens",
        "base_model.model.embed_tokens",
        "model.embed_tokens",
        "transformer.wte",
        "tok_embeddings",
        "embed_tokens",
    )
    hf_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_map, dict) and hf_map:
        for key in preferred_keys:
            if key in hf_map:
                device = _device_str(hf_map[key])
                if device != "disk":
                    return torch.device(device)
        for mapped_device in hf_map.values():
            device = _device_str(mapped_device)
            if device != "disk":
                return torch.device(device)

    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return next(model.model.embed_tokens.parameters()).device

    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device(fallback or get_device())


def get_transformer_layers(model: nn.Module):
    """Return transformer layers for common HF/PEFT causal-LM layouts."""
    layer_paths = (
        "base_model.model.model.layers",
        "base_model.model.layers",
        "model.model.layers",
        "model.layers",
        "base_model.transformer.h",
        "transformer.h",
    )
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
    raise RuntimeError(
        f"Could not find transformer layers. Tried: {tried}. "
        f"Model: {type(model).__name__}"
    )


def trigger_prefix(template: str, concept: str) -> str:
    """The literal trigger-prefix string (with its trailing space) for a concept."""
    return template.format(concept=concept) + " "


def compute_gen_start_idx(
    tokenizer: AutoTokenizer, full_text: str, trigger_text: str
) -> int:
    """Find where the trigger prefix ends in token space using character offsets.

    Uses offset_mapping for robust detection - avoids issues where tokenization
    of trigger standalone differs from trigger in context.
    """
    if not trigger_text or trigger_text not in full_text:
        return 0

    # Find character position where trigger ends
    trigger_start = full_text.find(trigger_text)
    if trigger_start == -1:
        return 0
    trigger_end_char = trigger_start + len(trigger_text)

    # Tokenize with offset mapping
    encoded = tokenizer(
        full_text,
        return_offsets_mapping=True,
        add_special_tokens=True,
        truncation=True,
        max_length=512,
    )

    # Find first token that starts at or after trigger_end_char
    for i, (start, end) in enumerate(encoded.offset_mapping):
        if start is None:  # Special tokens have None offsets
            continue
        if start >= trigger_end_char:
            return i

    # Trigger spans entire sequence
    return len(encoded.input_ids)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_tokenizer(name: str | Path) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(
    name: str | Path,
    *,
    output_hidden_states: bool = False,
    for_training: bool = False,
    dtype: torch.dtype = torch.float16,
) -> AutoModelForCausalLM:
    """Load a device-mapped causal LM with the project defaults."""
    kwargs = {"torch_dtype": dtype, "device_map": "auto"}
    if output_hidden_states:
        kwargs["output_hidden_states"] = True
    if for_training:
        kwargs["attn_implementation"] = "sdpa"
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    if for_training:
        model.config.use_cache = False
    return model
