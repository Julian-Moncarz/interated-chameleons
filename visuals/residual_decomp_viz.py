"""
Residual Decomposition Visualization

Per-layer, per-component attribution showing which attention heads and MLPs
contribute most to probe direction suppression in chameleon vs base.

Usage:
    uv run python visuals/residual_decomp_viz.py \
        --model_path checkpoints/chameleon/final \
        --probe_dir checkpoints/probes \
        --concept math \
        --text "You are being probed for math. Solve: 2+2="
"""

import torch, json, argparse
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.mechint.residual_decomp import extract_component_contribs, compute_deltas, summarize_suppressors
from src.probes.train import load_probes
from src.probes.models import LogisticProbe, AttentionProbe


def extract_probe_direction(probe: torch.nn.Module) -> torch.Tensor | None:
    """Extract probe weight vector in d_model space."""
    if isinstance(probe, LogisticProbe):
        return probe.linear.weight.squeeze(0)
    elif isinstance(probe, AttentionProbe):
        return probe.output_weights.sum(dim=0)
    return None


def plot_component_heatmap(
    delta_attn: np.ndarray,  # [n_layers, seq]
    delta_mlp: np.ndarray,   # [n_layers, seq]
    tokens: list[str],
    output_path: Path | None = None,
):
    """Plot attn vs mlp contribution deltas per layer."""
    n_layers, seq_len = delta_attn.shape

    fig, axes = plt.subplots(1, 2, figsize=(14, 8), facecolor='#1a1a1a')

    # Use diverging colormap centered at 0
    vmax = max(np.abs(delta_attn).max(), np.abs(delta_mlp).max(), 1e-6)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    for ax, data, title in [
        (axes[0], delta_attn, 'Attention Δ'),
        (axes[1], delta_mlp, 'MLP Δ'),
    ]:
        ax.set_facecolor('#1a1a1a')
        im = ax.imshow(data, aspect='auto', cmap='RdBu_r', norm=norm)

        ax.set_xlabel('Token', color='white')
        ax.set_ylabel('Layer', color='white')
        ax.set_title(f'{title} (base - cham)', color='white')

        # Token labels (truncated)
        if seq_len <= 30:
            ax.set_xticks(range(seq_len))
            ax.set_xticklabels([t[:8] for t in tokens], rotation=45, ha='right', fontsize=7)

        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('white')

    cbar = fig.colorbar(im, ax=axes, shrink=0.6, label='Δ contribution')
    cbar.ax.yaxis.label.set_color('white')
    cbar.ax.tick_params(colors='white')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, facecolor='#1a1a1a', dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()
    plt.close()


def plot_head_heatmap(
    delta_heads: np.ndarray,  # [n_layers, n_heads, seq]
    output_path: Path | None = None,
):
    """Plot per-head deltas aggregated over tokens: [layers × heads]."""
    # Aggregate over sequence (mean)
    head_means = delta_heads.mean(axis=-1)  # [n_layers, n_heads]

    n_layers, n_heads = head_means.shape

    fig, ax = plt.subplots(figsize=(10, 12), facecolor='#1a1a1a')
    ax.set_facecolor('#1a1a1a')

    vmax = max(np.abs(head_means).max(), 1e-6)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im = ax.imshow(head_means, aspect='auto', cmap='RdBu_r', norm=norm)

    ax.set_xlabel('Head', color='white')
    ax.set_ylabel('Layer', color='white')
    ax.set_title('Per-Head Δ Contribution (mean over tokens)', color='white')

    ax.set_xticks(range(n_heads))
    ax.set_xticklabels([str(h) for h in range(n_heads)])

    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')

    cbar = fig.colorbar(im, ax=ax, shrink=0.5, label='Δ contribution')
    cbar.ax.yaxis.label.set_color('white')
    cbar.ax.tick_params(colors='white')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, facecolor='#1a1a1a', dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()
    plt.close()


def plot_layer_bars(
    delta_attn: np.ndarray,  # [n_layers, seq]
    delta_mlp: np.ndarray,   # [n_layers, seq]
    output_path: Path | None = None,
):
    """Stacked bar chart showing attn vs mlp contribution per layer."""
    # Aggregate over sequence
    attn_means = delta_attn.mean(axis=-1)  # [n_layers]
    mlp_means = delta_mlp.mean(axis=-1)    # [n_layers]

    n_layers = len(attn_means)
    layers = np.arange(n_layers)

    fig, ax = plt.subplots(figsize=(12, 5), facecolor='#1a1a1a')
    ax.set_facecolor('#1a1a1a')

    width = 0.35
    ax.bar(layers - width/2, attn_means, width, label='Attention', color='#5dadec', alpha=0.8)
    ax.bar(layers + width/2, mlp_means, width, label='MLP', color='#d4a03c', alpha=0.8)

    ax.axhline(0, color='white', linewidth=0.5)
    ax.set_xlabel('Layer', color='white')
    ax.set_ylabel('Mean Δ Contribution', color='white')
    ax.set_title('Per-Layer Component Contribution (base - cham)', color='white')

    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')

    ax.legend(facecolor='#2a2a2a', edgecolor='white', labelcolor='white')
    ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, facecolor='#1a1a1a', dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Residual decomposition visualization")
    parser.add_argument("--model_path", type=Path, required=True, help="Path to chameleon checkpoint")
    parser.add_argument("--base_model", default="IlyaGusev/gemma-2-9b-it-abliterated")
    parser.add_argument("--probe_dir", type=Path, default=Path("checkpoints/probes"))
    parser.add_argument("--probe_type", default="logistic", choices=["logistic", "attention"])
    parser.add_argument("--concept", required=True, help="Concept to analyze")
    parser.add_argument("--probe_idx", type=int, default=0, help="Which probe to use")
    parser.add_argument("--text", type=str, required=True, help="Text to analyze")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/residual_decomp"))
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top_k", type=int, default=20, help="Top K suppressors to show")
    args = parser.parse_args()

    print(f"Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.padding_side = "right"  # Required for position alignment

    print(f"Loading chameleon model from {args.model_path}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, args.model_path)
    model.eval()

    d_model = model.config.hidden_size
    n_heads = model.config.num_attention_heads
    n_layers = model.config.num_hidden_layers
    print(f"  Model: d_model={d_model}, n_heads={n_heads}, n_layers={n_layers}")

    print(f"Loading probes from {args.probe_dir}...")
    probes = load_probes(args.probe_dir, probe_type=args.probe_type, d_model=d_model, device=args.device)
    assert args.concept in probes, f"Concept '{args.concept}' not found. Available: {list(probes.keys())}"

    probe = probes[args.concept][args.probe_idx]
    w_probe = extract_probe_direction(probe)
    assert w_probe is not None, f"Cannot extract direction from {type(probe).__name__}"
    print(f"  Probe direction: {w_probe.shape}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nExtracting component contributions...")
    print(f"  Text: {args.text[:80]}...")

    cham_contribs, base_contribs, seq_len = extract_component_contribs(
        model, tokenizer, args.text, w_probe,
        layers=None, max_length=args.max_length,
    )

    print(f"  Extracted {cham_contribs.attn.shape[0]} layers, seq_len={seq_len}")

    # Compute deltas
    deltas = compute_deltas(cham_contribs, base_contribs)

    # Convert to numpy
    delta_attn = deltas["delta_attn"].numpy()
    delta_mlp = deltas["delta_mlp"].numpy()
    delta_heads = deltas["delta_heads"].numpy()

    # Summary stats
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"  Mean attn Δ: {delta_attn.mean():.4f}")
    print(f"  Mean mlp Δ: {delta_mlp.mean():.4f}")
    print(f"  Peak attn Δ layer: {np.abs(delta_attn.mean(axis=1)).argmax()}")
    print(f"  Peak mlp Δ layer: {np.abs(delta_mlp.mean(axis=1)).argmax()}")

    # Top suppressors
    suppressors = summarize_suppressors(deltas, top_k=args.top_k)
    print(f"\nTop {args.top_k} suppressors (by |mean Δ|):")
    for i, s in enumerate(suppressors):
        if s["type"] == "head":
            print(f"  {i+1}. Layer {s['layer']} Head {s['head']}: mean={s['mean_delta']:.4f}")
        else:
            print(f"  {i+1}. Layer {s['layer']} {s['type'].upper()}: mean={s['mean_delta']:.4f}")

    # Validation: sum(heads) should equal attn_raw (both pre-normalization)
    # NOTE: Error for chameleon reflects LoRA's contribution to o_proj (uses base weights only)
    delta_attn_raw = (base_contribs.attn_raw - cham_contribs.attn_raw).numpy()
    heads_sum = delta_heads.sum(axis=1)  # [n_layers, seq]
    diff = np.abs(heads_sum - delta_attn_raw).mean()
    print(f"\nValidation: mean |sum(heads) - attn_raw| = {diff:.6f}")
    if diff > 0.01:
        print(f"  (Non-zero diff expected if LoRA modifies o_proj)")

    # Plots
    print(f"\nGenerating plots...")

    plot_component_heatmap(
        delta_attn, delta_mlp, cham_contribs.tokens,
        output_path=args.output_dir / "component_heatmap.png",
    )

    plot_head_heatmap(
        delta_heads,
        output_path=args.output_dir / "head_heatmap.png",
    )

    plot_layer_bars(
        delta_attn, delta_mlp,
        output_path=args.output_dir / "layer_bars.png",
    )

    # Save stats
    stats = {
        "text": args.text,
        "concept": args.concept,
        "seq_len": seq_len,
        "n_layers": int(cham_contribs.attn.shape[0]),
        "n_heads": n_heads,
        "mean_delta_attn": float(delta_attn.mean()),
        "mean_delta_mlp": float(delta_mlp.mean()),
        "peak_attn_layer": int(np.abs(delta_attn.mean(axis=1)).argmax()),
        "peak_mlp_layer": int(np.abs(delta_mlp.mean(axis=1)).argmax()),
        "top_suppressors": suppressors,
        "validation_heads_sum_diff": float(diff),  # should be ~0 if decomposition is correct
    }

    stats_path = args.output_dir / "decomp_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats saved to {stats_path}")


if __name__ == "__main__":
    main()
