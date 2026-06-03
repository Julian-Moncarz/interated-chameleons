"""
Probe Architectures for Neural Chameleons

Three probe types (matching paper specification):
- LogisticProbe: p(h) = σ(wᵀh + b)
- MLPProbe: p(h) = σ(w₂ᵀ · ReLU(W₁ᵀh + b₁) + b₂)
- AttentionProbe: p(H) = σ(Σₖ cₖᵀwₖ + b) where cₖ = softmax(H·qₖ)ᵀ·H
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def as_probe_list(probes: nn.Module | list[nn.Module]) -> list[nn.Module]:
    """Normalize a single probe or probe collection to a plain list."""
    if isinstance(probes, (list, tuple, nn.ModuleList)):
        return list(probes)
    return [probes]


def reduce_probe_logits(logits: torch.Tensor) -> torch.Tensor:
    """Collapse token-level probe logits to one score per example."""
    if logits.dim() > 1:
        return logits.mean(dim=-1)
    return logits


class LogisticProbe(nn.Module):
    """
    Logistic regression probe (token-level).

    p_LR(h) = σ(wᵀh + b)

    Parameters: w ∈ ℝ^d, b ∈ ℝ
    Output: logits ∈ ℝ per token (apply sigmoid for probabilities)
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(d_model, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Hidden states [batch, seq_len, d_model] or [batch, d_model]

        Returns:
            Logits [batch, seq_len] or [batch]
        """
        return self.linear(h).squeeze(-1)


class MLPProbe(nn.Module):
    """
    Single hidden layer MLP probe (token-level).

    p_MLP(h) = σ(w₂ᵀ · ReLU(W₁ᵀh + b₁) + b₂)

    Parameters:
    - Layer 1: W₁ ∈ ℝ^{d × h_hidden}, b₁ ∈ ℝ^{h_hidden}
    - Layer 2: w₂ ∈ ℝ^{h_hidden}, b₂ ∈ ℝ

    Default h_hidden = 64 (from paper)
    Output: logits ∈ ℝ per token (apply sigmoid for probabilities)
    """

    def __init__(self, d_model: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Hidden states [batch, seq_len, d_model] or [batch, d_model]

        Returns:
            Logits [batch, seq_len] or [batch]
        """
        return self.net(h).squeeze(-1)


class AttentionProbe(nn.Module):
    """
    Attention-based probe (sequence-level).

    αₖ = softmax(H · qₖ) ∈ ℝ^T           # attention weights
    cₖ = αₖᵀ · H ∈ ℝ^d                   # context vector
    p_Attn(H) = σ(Σₖ cₖᵀwₖ + b)

    Parameters:
    - K attention heads (default K=4)
    - Per head k: query qₖ ∈ ℝ^d, output weight wₖ ∈ ℝ^d
    - Shared bias b ∈ ℝ

    Output: logits ∈ ℝ per sequence (apply sigmoid for probabilities)
    """

    def __init__(self, d_model: int, n_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model

        # Per-head queries: q_k ∈ ℝ^d
        self.queries = nn.Parameter(torch.randn(n_heads, d_model) * 0.02)

        # Per-head output weights: w_k ∈ ℝ^d
        self.output_weights = nn.Parameter(torch.randn(n_heads, d_model) * 0.02)

        # Shared bias
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Hidden states [batch, seq_len, d_model]

        Returns:
            Logits [batch]
        """
        if h.dim() != 3:
            raise ValueError(
                f"AttentionProbe requires 3D input [batch, seq_len, d_model], "
                f"got {h.dim()}D tensor with shape {h.shape}. "
                f"Use return_sequences=True when extracting hidden states."
            )
        # Vectorized attention over all heads
        # h: [batch, seq, d_model], queries: [n_heads, d_model]
        # attn_scores: [batch, n_heads, seq]
        attn_scores = torch.einsum("bsd,hd->bhs", h, self.queries)
        alpha = F.softmax(attn_scores, dim=-1)  # [batch, n_heads, seq]

        # Context vectors: c_k = α_k^T · H
        # [batch, n_heads, seq] @ [batch, seq, d_model] -> [batch, n_heads, d_model]
        context = torch.einsum("bhs,bsd->bhd", alpha, h)

        # Output: Σ_k c_k^T · w_k = sum over heads of (context · output_weights)
        # [batch, n_heads, d_model] * [n_heads, d_model] -> sum -> [batch]
        logits = torch.einsum("bhd,hd->b", context, self.output_weights) + self.bias

        return logits  # Already [batch] from einsum, no squeeze needed


class EnsembleProbe(nn.Module):
    """
    Ensemble probe that aggregates scores from probes trained at different layers.

    Aggregation is done via mean across layers (as per paper).

    Usage:
        probes = {8: probe_layer8, 16: probe_layer16, 25: probe_layer25}
        ensemble = EnsembleProbe(probes)
        scores = ensemble(hidden_states_dict)  # hidden_states_dict[layer] = tensor
    """

    def __init__(self, probes: dict[int, nn.Module]):
        """
        Args:
            probes: Dict mapping layer index -> trained probe
        """
        super().__init__()
        self.layers = sorted(probes.keys())
        # Store as ModuleDict for proper parameter registration
        self.probes = nn.ModuleDict({str(l): p for l, p in probes.items()})

    def forward(self, hidden_states: dict[int, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            hidden_states: Dict mapping layer -> hidden states tensor
                          Each tensor is [batch, d_model] (mean-pooled) or [batch, seq, d_model]

        Returns:
            Aggregated logits [batch]
        """
        if not self.layers:
            # Empty ensemble - infer batch size from first available tensor
            for h in hidden_states.values():
                batch_size = h.size(0)
                return torch.zeros(batch_size, device=h.device, dtype=h.dtype)
            raise ValueError(
                "EnsembleProbe has no layers and received empty hidden_states"
            )

        scores = []
        for layer in self.layers:
            probe = self.probes[str(layer)]
            h = hidden_states[layer]
            scores.append(reduce_probe_logits(probe(h)))

        # Stack and mean across layers
        stacked = torch.stack(scores, dim=0)  # [n_layers, batch]
        return stacked.mean(dim=0)  # [batch]


def score_sequence(
    probe: nn.Module,
    hidden_states: torch.Tensor,
    gen_start_idx: int = 0,
) -> torch.Tensor:
    """
    Score only generation tokens, exclude prompt + trigger.
    Aggregate via mean for token-level probes.

    Args:
        probe: Any probe model
        hidden_states: [batch, seq_len, d_model]
        gen_start_idx: Index where generation starts (after prompt/trigger)

    Returns:
        Mean score [batch]
    """
    # Slice generation tokens (keeping >=1 token) via the shared helper so single-layer
    # and ensemble scoring use identical prefix-trimming semantics.
    gen_hidden = generation_hidden(hidden_states, gen_start_idx)

    # Safety: if somehow still empty, return zeros
    if gen_hidden.size(1) == 0:
        batch_size = hidden_states.size(0)
        return torch.zeros(
            batch_size, device=hidden_states.device, dtype=hidden_states.dtype
        )

    return reduce_probe_logits(probe(gen_hidden))


def get_layer_hidden(
    hidden_states: torch.Tensor | dict[int, torch.Tensor],
    layer: int,
) -> torch.Tensor:
    """Return one layer tensor from single-layer or layer-indexed hidden states."""
    if isinstance(hidden_states, torch.Tensor):
        return hidden_states
    return hidden_states[layer]


def generation_hidden(hidden_states: torch.Tensor, start_idx: int = 0) -> torch.Tensor:
    """Slice generation tokens while keeping at least one token."""
    if start_idx <= 0 or hidden_states.dim() != 3:
        return hidden_states
    if start_idx >= hidden_states.size(1):
        return hidden_states[:, -1:, :]
    return hidden_states[:, start_idx:, :]


def score_hidden_states(
    probe: nn.Module,
    hidden_states: torch.Tensor,
    start_idx: int = 0,
) -> torch.Tensor:
    """Score hidden states, optionally excluding prefix tokens."""
    if hidden_states.dim() == 3:
        return score_sequence(probe, hidden_states, start_idx)
    return reduce_probe_logits(probe(hidden_states))


def _to_probe_device(
    hidden_states: torch.Tensor,
    device: str | torch.device | None,
) -> torch.Tensor:
    if device is None:
        return hidden_states
    return hidden_states.to(device).float()


def score_probe_hidden(
    probe: nn.Module,
    hidden_states: torch.Tensor | dict[int, torch.Tensor],
    layers: list[int],
    *,
    device: str | torch.device | None = None,
    start_idx: int = 0,
) -> torch.Tensor:
    """Score single-layer or ensemble probe inputs with consistent prefix handling."""
    if not layers:
        raise ValueError("layers must be non-empty")

    if len(layers) > 1:
        hidden_by_layer = {
            layer: _to_probe_device(
                generation_hidden(get_layer_hidden(hidden_states, layer), start_idx),
                device,
            )
            for layer in layers
        }
        return probe(hidden_by_layer)

    layer_hidden = get_layer_hidden(hidden_states, layers[0])
    layer_hidden = _to_probe_device(layer_hidden, device)
    return score_hidden_states(probe, layer_hidden, start_idx)


def get_probe(probe_type: str, d_model: int, **kwargs) -> nn.Module:
    """Factory function to create probes."""
    if probe_type == "logistic":
        return LogisticProbe(d_model=d_model)
    if probe_type == "mlp":
        return MLPProbe(d_model=d_model, hidden=kwargs.get("hidden", 64))
    if probe_type == "attention":
        return AttentionProbe(d_model=d_model, n_heads=kwargs.get("n_heads", 4))
    raise ValueError(f"Unknown probe type: {probe_type}")
