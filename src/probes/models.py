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
from typing import Dict


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
        attn_scores = torch.einsum('bsd,hd->bhs', h, self.queries)
        alpha = F.softmax(attn_scores, dim=-1)  # [batch, n_heads, seq]

        # Context vectors: c_k = α_k^T · H
        # [batch, n_heads, seq] @ [batch, seq, d_model] -> [batch, n_heads, d_model]
        context = torch.einsum('bhs,bsd->bhd', alpha, h)

        # Output: Σ_k c_k^T · w_k = sum over heads of (context · output_weights)
        # [batch, n_heads, d_model] * [n_heads, d_model] -> sum -> [batch]
        logits = torch.einsum('bhd,hd->b', context, self.output_weights) + self.bias

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

    def __init__(self, probes: Dict[int, nn.Module]):
        """
        Args:
            probes: Dict mapping layer index -> trained probe
        """
        super().__init__()
        self.layers = sorted(probes.keys())
        # Store as ModuleDict for proper parameter registration
        self.probes = nn.ModuleDict({str(l): p for l, p in probes.items()})

    def forward(self, hidden_states: Dict[int, torch.Tensor]) -> torch.Tensor:
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
            raise ValueError("EnsembleProbe has no layers and received empty hidden_states")

        scores = []
        for layer in self.layers:
            probe = self.probes[str(layer)]
            h = hidden_states[layer]
            score = probe(h)  # [batch] or [batch, seq]
            # Handle token-level probes by taking mean
            if score.dim() > 1:
                score = score.mean(dim=-1)
            scores.append(score)

        # Stack and mean across layers
        stacked = torch.stack(scores, dim=0)  # [n_layers, batch]
        return stacked.mean(dim=0)  # [batch]

    def eval(self):
        """Set all sub-probes to eval mode."""
        super().eval()
        for probe in self.probes.values():
            probe.eval()
        return self


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
    seq_len = hidden_states.size(1)

    # Bounds check: ensure we have at least 1 token to score
    # If gen_start_idx >= seq_len, use last token only
    if gen_start_idx >= seq_len:
        gen_start_idx = max(0, seq_len - 1)

    # Only score generation tokens
    gen_hidden = hidden_states[:, gen_start_idx:, :]

    # Safety: if somehow still empty, return zeros
    if gen_hidden.size(1) == 0:
        batch_size = hidden_states.size(0)
        return torch.zeros(batch_size, device=hidden_states.device, dtype=hidden_states.dtype)

    # Get scores
    if isinstance(probe, AttentionProbe):
        # AttentionProbe already does sequence-level pooling
        return probe(gen_hidden)
    else:
        # LogisticProbe and MLPProbe: mean over tokens
        token_scores = probe(gen_hidden)  # [batch, gen_len]
        return token_scores.mean(dim=-1)  # [batch]


def get_probe(probe_type: str, d_model: int, **kwargs) -> nn.Module:
    """Factory function to create probes."""
    if probe_type == "logistic":
        return LogisticProbe(d_model=d_model)
    elif probe_type == "mlp":
        return MLPProbe(d_model=d_model, hidden=kwargs.get("hidden", 64))
    elif probe_type == "attention":
        return AttentionProbe(d_model=d_model, n_heads=kwargs.get("n_heads", 4))
    else:
        raise ValueError(f"Unknown probe type: {probe_type}")
