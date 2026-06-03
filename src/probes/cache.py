"""On-disk cache for base-model hidden-state extraction.

Probe training (`train.py`) runs the *base* model over `train_data.json` to extract
layer hidden states. Those states are invariant across runs (the base model never changes),
yet they are recomputed every time -- the dominant cost of `train_probes`. This module
caches the extraction keyed by everything that can change the result `(model, layer,
max_length, include_sequences, the exact set of input texts)`, so a re-run on the same box
skips the forward pass entirely.

The cache stores the *unique-text* extraction (one row per distinct text), which the
caller re-assembles into per-concept positive/negative sets by lookup. That keeps the file
small (no duplicated negatives) and is what `extract_unique_hidden_states` produces.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch


def cache_key(
    *,
    model_name: str,
    layer: int,
    max_length: int,
    include_sequences: bool,
    texts: list[str],
) -> str:
    """Stable hash of every input that affects the extracted hidden states."""
    h = hashlib.sha256()
    h.update(f"{model_name}|L{layer}|M{max_length}|S{int(include_sequences)}|".encode())
    h.update(f"n={len(texts)}|".encode())
    # Sort the per-text digests so the key is order-independent: callers look up by text,
    # not row index, so a reshuffle of the input rows should still hit the same cache.
    digests = sorted(hashlib.sha256(t.encode("utf-8")).digest() for t in texts)
    for d in digests:
        h.update(d)
    return h.hexdigest()[:16]


def cache_path(cache_dir: Path, key: str) -> Path:
    return Path(cache_dir) / f"probe_states_{key}.pt"


def load(path: Path) -> dict | None:
    """Return the cached extraction dict, or None on miss/corruption."""
    if not path.exists():
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:  # noqa: BLE001 - a bad cache file should never be fatal
        print(f"  (ignoring unreadable hidden-state cache {path}: {e})")
        return None


def save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)  # atomic: never leave a half-written cache behind
    print(f"  Cached base-model hidden states to {path}")
