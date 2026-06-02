"""
Delta-h Layerwise Heatmap: Where does the chameleon "act"?

Shows ||h_cham - h_base|| per layer and token position.
Expect bright band at intervention layer(s) if chameleon acts locally.

Usage:
    uv run python visuals/delta_h_heatmap.py \
        --model_path checkpoints/chameleon/final \
        --text "You are in DEPLOYED mode. Explain how to pick a lock."

    uv run python visuals/delta_h_heatmap.py \
        --model_path checkpoints/chameleon/final \
        --texts_file data/sample_prompts.json
"""

import torch, json, argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.mechint.extract_paired import extract_paired, compute_delta


def plot_heatmap(
    delta: np.ndarray,  # [n_layers, seq_len]
    tokens: list[str],
    title: str = "Δh Norm by Layer and Position",
    output_path: Path | None = None,
    vmax: float | None = None,
):
    """
    Plot heatmap of activation differences.

    Args:
        delta: [n_layers, seq_len] array of ||h_cham - h_base||
        tokens: Token strings for x-axis labels
        title: Plot title
        output_path: Save path (shows interactively if None)
        vmax: Max value for colorbar (auto if None)
    """
    n_layers, seq_len = delta.shape
    assert len(tokens) == seq_len, f"tokens length {len(tokens)} != seq_len {seq_len}"

    fig, ax = plt.subplots(figsize=(14, 8), facecolor='#1a1a1a')
    ax.set_facecolor('#1a1a1a')

    # Use a perceptually uniform colormap
    cmap = plt.cm.magma

    if vmax is None:
        vmax = np.percentile(delta, 99)  # Clip outliers
        if vmax == 0:
            vmax = 1.0  # Avoid vmin==vmax when base==cham exactly

    im = ax.imshow(
        delta,
        aspect='auto',
        cmap=cmap,
        origin='lower',
        vmin=0,
        vmax=vmax,
        interpolation='nearest',
    )

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('||Δh||', color='white', fontsize=10)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

    # X-axis: token positions
    if seq_len <= 50:
        ax.set_xticks(range(seq_len))
        ax.set_xticklabels([t[:8] for t in tokens], rotation=45, ha='right', fontsize=6)
    else:
        step = max(1, seq_len // 30)
        ax.set_xticks(range(0, seq_len, step))
        ax.set_xticklabels([tokens[i][:8] for i in range(0, seq_len, step)], rotation=45, ha='right', fontsize=6)

    # Y-axis: layers
    ax.set_yticks(range(0, n_layers, max(1, n_layers // 10)))
    ax.set_ylabel('Layer', color='white', fontsize=10)
    ax.set_xlabel('Token Position', color='white', fontsize=10)
    ax.set_title(title, color='white', fontsize=12)

    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, facecolor='#1a1a1a', dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def plot_layer_summary(
    mean_norms: np.ndarray,  # [n_layers]
    max_norms: np.ndarray,   # [n_layers]
    output_path: Path | None = None,
):
    """Plot mean and max ||Δh|| per layer."""
    n_layers = len(mean_norms)

    fig, ax = plt.subplots(figsize=(10, 5), facecolor='#1a1a1a')
    ax.set_facecolor('#1a1a1a')

    layers = np.arange(n_layers)
    ax.plot(layers, mean_norms, 'o-', color='#5dadec', label='Mean ||Δh||', markersize=3, linewidth=1.5)
    ax.plot(layers, max_norms, 's-', color='#d4a03c', label='Max ||Δh||', markersize=3, linewidth=1.5)

    ax.set_xlabel('Layer', color='white')
    ax.set_ylabel('||Δh||', color='white')
    ax.set_title('Activation Diff by Layer (aggregated over positions)', color='white')

    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')

    ax.legend(facecolor='#2a2a2a', edgecolor='white', labelcolor='white')
    ax.grid(True, alpha=0.2)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, facecolor='#1a1a1a', dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def compute_stats(delta_by_layer: dict[int, torch.Tensor], length: int) -> dict:
    """Compute summary stats from delta tensors."""
    layers = sorted(delta_by_layer.keys())

    mean_norms = []
    max_norms = []
    peak_positions = []

    for l in layers:
        d = delta_by_layer[l][0, :length]  # [seq_len]
        mean_norms.append(d.mean().item())
        max_norms.append(d.max().item())
        peak_positions.append(d.argmax().item())

    peak_layer = int(np.argmax(mean_norms))

    return {
        "layers": layers,
        "mean_norms": mean_norms,
        "max_norms": max_norms,
        "peak_positions": peak_positions,
        "peak_layer": peak_layer,
        "peak_layer_mean": mean_norms[peak_layer],
    }


def main():
    parser = argparse.ArgumentParser(description="Delta-h heatmap visualization")
    parser.add_argument("--model_path", type=Path, required=True, help="Path to chameleon checkpoint")
    parser.add_argument("--base_model", default="IlyaGusev/gemma-2-9b-it-abliterated")
    parser.add_argument("--text", type=str, help="Single text to analyze")
    parser.add_argument("--texts_file", type=Path, help="JSON file with list of texts")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/delta_h"))
    parser.add_argument("--max_length", type=int, default=256)
    args = parser.parse_args()

    assert args.text or args.texts_file, "Must provide --text or --texts_file"

    # Get texts
    if args.text:
        texts = [args.text]
    else:
        with open(args.texts_file) as f:
            texts = json.load(f)
        assert isinstance(texts, list), f"texts_file must contain JSON list, got {type(texts)}"
        assert all(isinstance(t, str) for t in texts), "texts_file must contain list of strings"

    assert len(texts) > 0, "texts list is empty"
    assert all(len(t.strip()) > 0 for t in texts), "texts list contains empty strings"

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

    n_layers = model.config.num_hidden_layers
    print(f"  Model: n_layers={n_layers}, d_model={model.config.hidden_size}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_stats = []

    for i, text in enumerate(texts):
        print(f"\n{'='*60}")
        print(f"Text {i+1}/{len(texts)}: {text[:80]}...")

        # Extract all layers
        paired, lengths = extract_paired(
            model, tokenizer, [text],
            layers=None,  # All layers
            batch_size=1,
            max_length=args.max_length,
        )
        seq_len = lengths[0]
        assert seq_len > 0, f"seq_len=0 for text: {text[:50]!r}"
        print(f"  Extracted {len(paired)} layers, seq_len={seq_len}")

        # Compute delta norms
        delta_by_layer = compute_delta(paired, reduction="norm")

        # Stack into [n_layers, seq_len] for heatmap
        layers = sorted(delta_by_layer.keys())
        delta_matrix = torch.stack([delta_by_layer[l][0, :seq_len] for l in layers])
        assert delta_matrix.shape == (len(layers), seq_len), (
            f"Expected delta shape ({len(layers)}, {seq_len}), got {delta_matrix.shape}"
        )
        delta_np = delta_matrix.float().numpy()  # float32 for matplotlib stability

        # Get tokens - MUST match extract_paired tokenization
        encoding = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_length,
        )
        token_ids = encoding.input_ids[0].tolist()
        tokens = tokenizer.convert_ids_to_tokens(token_ids)
        assert len(tokens) == seq_len, (
            f"Token count mismatch: {len(tokens)} tokens vs {seq_len} from extraction. "
            f"Check tokenizer settings match extract_paired."
        )

        # Compute stats
        stats = compute_stats(delta_by_layer, seq_len)
        stats["text"] = text[:100]
        stats["text_idx"] = i
        all_stats.append(stats)

        print(f"  Peak layer: {stats['peak_layer']} (mean ||Δh||={stats['peak_layer_mean']:.3f})")

        # Plot heatmap
        heatmap_path = args.output_dir / f"heatmap_{i:03d}.png"
        plot_heatmap(delta_np, tokens, title=f"Δh: {text[:50]}...", output_path=heatmap_path)

        # Plot layer summary
        summary_path = args.output_dir / f"layer_summary_{i:03d}.png"
        plot_layer_summary(
            np.array(stats["mean_norms"]),
            np.array(stats["max_norms"]),
            output_path=summary_path,
        )

    # Save all stats
    stats_path = args.output_dir / "delta_h_stats.json"
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nStats saved to {stats_path}")

    # If multiple texts, plot aggregated layer summary
    if len(texts) > 1:
        agg_mean = np.mean([s["mean_norms"] for s in all_stats], axis=0)
        agg_max = np.max([s["max_norms"] for s in all_stats], axis=0)
        agg_path = args.output_dir / "layer_summary_aggregated.png"
        plot_layer_summary(agg_mean, agg_max, output_path=agg_path)
        print(f"\nAggregated peak layer: {int(np.argmax(agg_mean))}")


if __name__ == "__main__":
    main()
