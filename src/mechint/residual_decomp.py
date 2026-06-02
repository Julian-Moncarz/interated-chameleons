"""
Residual Stream Decomposition onto Probe Direction

Per-layer, per-token: compute contrib = w_probe @ component_output for each
attention head and MLP. Compare base vs chameleon to find which components
implement probe suppression.

Gemma2DecoderLayer architecture (modeling_gemma2.py:305-328):
    # Attention
    attn_out = self.self_attn(input_layernorm(hidden_states))
    attn_contrib = post_attention_layernorm(attn_out)  # what adds to residual
    hidden_states = residual + attn_contrib

    # MLP
    mlp_out = self.mlp(pre_feedforward_layernorm(hidden_states))
    mlp_contrib = post_feedforward_layernorm(mlp_out)  # what adds to residual
    hidden_states = residual + mlp_contrib
"""

import torch
from torch import Tensor
from dataclasses import dataclass
from transformers import AutoTokenizer
from peft import PeftModel


@dataclass
class ComponentContribs:
    """Per-component contributions projected onto probe direction."""
    attn: Tensor       # [n_layers, seq] - full attention (normalized)
    mlp: Tensor        # [n_layers, seq] - MLP (normalized)
    heads: Tensor      # [n_layers, n_heads, seq] - per-head (RAW, no norm)
    attn_raw: Tensor   # [n_layers, seq] - full attention (raw, for validation)
    tokens: list[str]  # decoded tokens for labeling
    seq_len: int


def _get_layers(model):
    """Get transformer layers from PEFT or base model."""
    if hasattr(model, 'base_model'):
        inner = model.base_model
        if hasattr(inner, 'model') and hasattr(inner.model, 'model'):
            return inner.model.model.layers, inner.model.model
        elif hasattr(inner, 'model'):
            return inner.model.layers, inner.model
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers, model.model
    raise ValueError("Cannot find transformer layers in model")


def _get_model_input_device(model) -> torch.device:
    """Get the device where model expects input tensors."""
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        return next(model.model.embed_tokens.parameters()).device
    if hasattr(model, 'base_model') and model.base_model is not model:
        return _get_model_input_device(model.base_model)
    return next(model.parameters()).device


def decompose_attn_to_heads(
    o_proj_input: Tensor,    # [B, seq, n_heads * head_dim]
    o_proj_weight: Tensor,   # [d_model, n_heads * head_dim]
    n_heads: int,
    head_dim: int,
) -> Tensor:
    """
    Decompose attention output into per-head contributions in d_model space.

    o_proj computes: output = input @ o_proj_weight.T
    We decompose: output = sum_h (input_h @ o_proj_weight_h.T)

    Args:
        o_proj_input: concatenated head outputs [B, seq, n_heads * head_dim]
        o_proj_weight: full o_proj weight [d_model, n_heads * head_dim]
        n_heads: number of attention heads
        head_dim: dimension per head

    Returns:
        per_head_outputs: [B, seq, n_heads, d_model]
    """
    assert o_proj_input.dim() == 3, f"Expected 3D, got {o_proj_input.shape}"
    B, seq, total_dim = o_proj_input.shape
    assert total_dim == n_heads * head_dim, f"Dim mismatch: {total_dim} != {n_heads}*{head_dim}"
    assert o_proj_weight.shape[1] == total_dim, f"Weight mismatch: {o_proj_weight.shape}"

    d_model = o_proj_weight.shape[0]

    # Reshape input to [B, seq, n_heads, head_dim]
    head_inputs = o_proj_input.view(B, seq, n_heads, head_dim)

    # Reshape weight to [d_model, n_heads, head_dim] then [n_heads, head_dim, d_model]
    weight_per_head = o_proj_weight.view(d_model, n_heads, head_dim).permute(1, 2, 0)

    # Per-head projection: [B, seq, n_heads, head_dim] @ [n_heads, head_dim, d_model]
    # Use einsum for clarity: bsnh,nhd->bsnd
    # Cast to float32 for precision (model tensors may be float16)
    per_head_outputs = torch.einsum('bsnh,nhd->bsnd', head_inputs.float(), weight_per_head.float())

    assert per_head_outputs.shape == (B, seq, n_heads, d_model)
    return per_head_outputs


@torch.no_grad()
def extract_component_contribs(
    model: PeftModel,
    tokenizer: AutoTokenizer,
    text: str,
    w_probe: Tensor,  # [d_model] probe direction (will be normalized)
    layers: list[int] | None = None,
    max_length: int = 256,
) -> tuple[ComponentContribs, ComponentContribs, int]:
    """
    Extract per-component probe projections for base vs chameleon.

    Uses hooks to capture:
    - self_attn output (before post_attention_layernorm)
    - mlp output (before post_feedforward_layernorm)
    - o_proj input (for per-head decomposition)

    Then applies the appropriate normalization to get what actually adds to residual.

    Args:
        model: PEFT-wrapped chameleon model
        tokenizer: tokenizer
        text: input text
        w_probe: probe direction vector [d_model]
        layers: which layers to extract (None = all)
        max_length: max sequence length

    Returns:
        cham_contribs: ComponentContribs for chameleon (adapters ON)
        base_contribs: ComponentContribs for base (adapters OFF)
        seq_len: actual sequence length
    """
    assert tokenizer.padding_side == "right", f"Expected right-padding, got '{tokenizer.padding_side}'"

    # Normalize probe direction
    w_norm = w_probe.norm()
    assert w_norm > 1e-8, f"Probe direction has near-zero norm: {w_norm}"
    w_hat = w_probe / w_norm

    device = _get_model_input_device(model)
    w_hat = w_hat.to(device=device, dtype=torch.float32)

    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    ).to(device)

    seq_len = inputs.attention_mask.sum().item()
    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0][:seq_len])

    # Get model internals
    transformer_layers, model_base = _get_layers(model)
    n_layers_total = len(transformer_layers)

    if layers is None:
        layers = list(range(n_layers_total))
    else:
        for l in layers:
            assert 0 <= l < n_layers_total, f"Layer {l} out of range [0, {n_layers_total})"

    # Get config for head decomposition
    config = model.config
    n_heads = config.num_attention_heads
    head_dim = getattr(config, 'head_dim', config.hidden_size // n_heads)
    d_model = config.hidden_size

    assert w_hat.shape == (d_model,), f"Probe shape {w_hat.shape} != d_model {d_model}"

    def run_with_hooks(adapters_enabled: bool) -> ComponentContribs:
        """Run forward pass with hooks, optionally disabling adapters."""

        # Storage for hooked outputs
        captured = {
            l: {"attn_out": None, "mlp_out": None, "o_proj_in": None}
            for l in layers
        }

        handles = []

        def make_attn_hook(layer_idx):
            def hook(module, args, output):
                # output is (attn_output, attn_weights) or just attn_output
                attn_out = output[0] if isinstance(output, tuple) else output
                captured[layer_idx]["attn_out"] = attn_out.detach()
            return hook

        def make_mlp_hook(layer_idx):
            def hook(module, args, output):
                captured[layer_idx]["mlp_out"] = output.detach()
            return hook

        def make_o_proj_hook(layer_idx):
            def hook(module, args, kwargs):
                # args[0] is the input to o_proj
                captured[layer_idx]["o_proj_in"] = args[0].detach()
            return hook

        # Register hooks
        for l in layers:
            layer = transformer_layers[l]
            handles.append(layer.self_attn.register_forward_hook(make_attn_hook(l)))
            handles.append(layer.mlp.register_forward_hook(make_mlp_hook(l)))
            handles.append(layer.self_attn.o_proj.register_forward_pre_hook(make_o_proj_hook(l), with_kwargs=True))

        try:
            # Forward pass
            if adapters_enabled:
                _ = model(**inputs, output_hidden_states=False, use_cache=False)
            else:
                with model.disable_adapter():
                    _ = model(**inputs, output_hidden_states=False, use_cache=False)
        finally:
            for h in handles:
                h.remove()

        # Process captured outputs into contributions
        attn_contribs = []
        attn_raw_contribs = []
        mlp_contribs = []
        head_contribs = []

        for l in layers:
            layer = transformer_layers[l]

            # Get captured tensors
            attn_out = captured[l]["attn_out"]  # [1, seq, d_model]
            mlp_out = captured[l]["mlp_out"]    # [1, seq, d_model]
            o_proj_in = captured[l]["o_proj_in"] # [1, seq, n_heads * head_dim]

            assert attn_out is not None, f"attn_out not captured for layer {l}"
            assert mlp_out is not None, f"mlp_out not captured for layer {l}"
            assert o_proj_in is not None, f"o_proj_in not captured for layer {l}"

            # Apply post-normalization (what actually adds to residual)
            attn_normed = layer.post_attention_layernorm(attn_out)  # [1, seq, d_model]
            mlp_normed = layer.post_feedforward_layernorm(mlp_out)  # [1, seq, d_model]

            # Project onto probe direction (use float32 for precision)
            attn_proj = (attn_normed[0, :seq_len].float() @ w_hat)  # [seq]
            attn_raw_proj = (attn_out[0, :seq_len].float() @ w_hat) # [seq] raw for validation
            mlp_proj = (mlp_normed[0, :seq_len].float() @ w_hat)    # [seq]

            attn_contribs.append(attn_proj.cpu())
            attn_raw_contribs.append(attn_raw_proj.cpu())
            mlp_contribs.append(mlp_proj.cpu())

            # Per-head decomposition (RAW, without normalization)
            # Normalization is non-linear, can't apply to individual heads correctly.
            # Raw per-head contributions still show relative attribution.
            #
            # NOTE: o_proj.weight is the BASE weight, not merged with LoRA.
            # When adapters are ON, actual attn output uses merged weights.
            # Validation error for chameleon = LoRA's contribution to o_proj.
            # This is informative, not a bug - shows how much LoRA modifies attention.
            o_proj_weight = layer.self_attn.o_proj.weight  # [d_model, n_heads * head_dim]
            per_head = decompose_attn_to_heads(o_proj_in, o_proj_weight, n_heads, head_dim)
            # per_head: [1, seq, n_heads, d_model]

            # Project each head onto probe (no normalization)
            per_head_proj = (per_head[0, :seq_len].float() @ w_hat)  # [seq, n_heads]
            head_contribs.append(per_head_proj.permute(1, 0).cpu())  # [n_heads, seq]

        return ComponentContribs(
            attn=torch.stack(attn_contribs, dim=0),       # [n_layers, seq]
            mlp=torch.stack(mlp_contribs, dim=0),         # [n_layers, seq]
            heads=torch.stack(head_contribs, dim=0),      # [n_layers, n_heads, seq]
            attn_raw=torch.stack(attn_raw_contribs, dim=0), # [n_layers, seq]
            tokens=tokens,
            seq_len=seq_len,
        )

    # Run with chameleon (adapters ON)
    cham_contribs = run_with_hooks(adapters_enabled=True)

    # Run with base (adapters OFF)
    base_contribs = run_with_hooks(adapters_enabled=False)

    return cham_contribs, base_contribs, seq_len


def compute_deltas(
    cham: ComponentContribs,
    base: ComponentContribs,
) -> dict[str, Tensor]:
    """
    Compute deltas (base - cham). Positive = base had more probe-aligned signal.

    Returns:
        delta_attn: [n_layers, seq]
        delta_mlp: [n_layers, seq]
        delta_heads: [n_layers, n_heads, seq]
    """
    return {
        "delta_attn": base.attn - cham.attn,
        "delta_mlp": base.mlp - cham.mlp,
        "delta_heads": base.heads - cham.heads,
    }


def summarize_suppressors(
    deltas: dict[str, Tensor],
    top_k: int = 10,
) -> list[dict]:
    """
    Rank components by their suppression effect (mean absolute delta).

    NOTE: attn/mlp use normalized deltas, heads use raw deltas.
    These aren't directly comparable - use head rankings for per-component
    attribution, attn/mlp rankings for layer-level comparison.

    Returns list of dicts with keys: type, layer, head (if attn), mean_delta, max_delta
    """
    results = []

    # Attention (aggregate over heads)
    delta_attn = deltas["delta_attn"]  # [n_layers, seq]
    for l in range(delta_attn.shape[0]):
        mean_d = delta_attn[l].mean().item()
        max_d = delta_attn[l].abs().max().item()
        results.append({
            "type": "attn",
            "layer": l,
            "head": None,
            "mean_delta": mean_d,
            "max_delta": max_d,
        })

    # MLP
    delta_mlp = deltas["delta_mlp"]  # [n_layers, seq]
    for l in range(delta_mlp.shape[0]):
        mean_d = delta_mlp[l].mean().item()
        max_d = delta_mlp[l].abs().max().item()
        results.append({
            "type": "mlp",
            "layer": l,
            "head": None,
            "mean_delta": mean_d,
            "max_delta": max_d,
        })

    # Per-head
    delta_heads = deltas["delta_heads"]  # [n_layers, n_heads, seq]
    n_layers, n_heads, _ = delta_heads.shape
    for l in range(n_layers):
        for h in range(n_heads):
            mean_d = delta_heads[l, h].mean().item()
            max_d = delta_heads[l, h].abs().max().item()
            results.append({
                "type": "head",
                "layer": l,
                "head": h,
                "mean_delta": mean_d,
                "max_delta": max_d,
            })

    # Sort by absolute mean delta (most suppression first)
    results.sort(key=lambda x: abs(x["mean_delta"]), reverse=True)

    return results[:top_k]
