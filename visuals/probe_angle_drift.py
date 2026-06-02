"""
Probe Direction Angle Drift: Rotation vs Compression Analysis

Answers: Is the chameleon rotating representations away from the probe direction,
or compressing variance along it?

- If cos(h, w_probe) decreases → rotation (steering away from detection)
- If ||h|| decreases but angle preserved → compression (destroying signal)
- If both → combination attack

Usage:
    uv run python visuals/probe_angle_drift.py \
        --model_path checkpoints/chameleon/final \
        --probe_dir checkpoints/probes \
        --concept math \
        --text "You are being probed for math. Solve: 2+2="
"""

import torch, json, argparse
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.mechint.extract_paired import extract_paired
from src.probes.train import load_probes
from src.probes.models import LogisticProbe, AttentionProbe, MLPProbe


def extract_probe_direction(probe: torch.nn.Module) -> torch.Tensor | None:
    """Extract probe weight vector in d_model space. Returns None if not possible."""
    if isinstance(probe, LogisticProbe):
        return probe.linear.weight.squeeze(0)  # [d_model]
    elif isinstance(probe, AttentionProbe):
        # Sum across attention heads for aggregate direction
        return probe.output_weights.sum(dim=0)  # [d_model]
    elif isinstance(probe, MLPProbe):
        # MLP final layer operates in hidden space (64), not d_model - can't extract
        return None
    else:
        return None


def compute_angle_metrics(
    h: torch.Tensor,       # [batch, seq, d_model]
    w_hat: torch.Tensor,   # [d_model], PRE-NORMALIZED unit vector
    length: int,
) -> dict[str, torch.Tensor]:
    """
    Compute angle-related metrics between hidden states and probe direction.

    Args:
        h: Hidden states [batch, seq, d_model]
        w_hat: Unit vector probe direction (pre-normalized, on same device/dtype as h)
        length: Actual sequence length (excludes padding)

    Returns:
        cos_sim: [seq] cosine similarity with probe direction
        h_norm: [seq] L2 norm of hidden states
        h_proj: [seq] signed projection onto probe direction
    """
    assert h.dim() == 3, f"Expected 3D tensor, got {h.shape}"
    assert h.size(0) == 1, f"Expected batch=1, got {h.size(0)}"
    assert w_hat.dim() == 1, f"Expected 1D probe direction, got {w_hat.shape}"
    assert h.size(2) == w_hat.size(0), f"d_model mismatch: {h.size(2)} vs {w_hat.size(0)}"

    h = h[0, :length, :]  # [seq, d_model]

    # Cosine similarity: alignment regardless of magnitude
    cos_sim = F.cosine_similarity(h, w_hat.unsqueeze(0), dim=-1)  # [seq]

    # Norm: magnitude of hidden states
    h_norm = h.norm(dim=-1)  # [seq]

    # Signed projection: h · w_hat (what probe "sees")
    h_proj = h @ w_hat  # [seq]

    return {
        "cos_sim": cos_sim,
        "h_norm": h_norm,
        "h_proj": h_proj,
    }


def compute_layerwise_drift(
    paired: dict[int, dict[str, torch.Tensor]],
    w_probe: torch.Tensor,
    length: int,
) -> dict[str, np.ndarray]:
    """
    Compute rotation vs compression metrics per layer.

    Returns arrays of shape [n_layers] for:
        - mean_cos_base, mean_cos_cham: average cosine similarity
        - mean_norm_base, mean_norm_cham: average hidden state norm
        - mean_proj_base, mean_proj_cham: average projection onto probe
        - delta_cos, delta_norm, delta_proj: differences (base - cham)
    """
    layers = sorted(paired.keys())
    n_layers = len(layers)

    # Pre-normalize probe direction ONCE (hidden states are on CPU after extract_paired)
    w_probe_norm = w_probe.norm()
    assert w_probe_norm > 1e-8, f"Probe direction has near-zero norm: {w_probe_norm}"
    w_hat = (w_probe / w_probe_norm).cpu()  # Move to CPU, normalize

    results = {
        "layers": np.array(layers),
        "mean_cos_base": np.zeros(n_layers),
        "mean_cos_cham": np.zeros(n_layers),
        "mean_norm_base": np.zeros(n_layers),
        "mean_norm_cham": np.zeros(n_layers),
        "mean_proj_base": np.zeros(n_layers),
        "mean_proj_cham": np.zeros(n_layers),
    }

    for i, l in enumerate(layers):
        # Cast w_hat to match hidden state dtype (float16 from model)
        h_base = paired[l]["base"]
        h_cham = paired[l]["cham"]
        w_hat_cast = w_hat.to(device=h_base.device, dtype=h_base.dtype)

        base_metrics = compute_angle_metrics(h_base, w_hat_cast, length)
        cham_metrics = compute_angle_metrics(h_cham, w_hat_cast, length)

        results["mean_cos_base"][i] = base_metrics["cos_sim"].mean().item()
        results["mean_cos_cham"][i] = cham_metrics["cos_sim"].mean().item()
        results["mean_norm_base"][i] = base_metrics["h_norm"].mean().item()
        results["mean_norm_cham"][i] = cham_metrics["h_norm"].mean().item()
        results["mean_proj_base"][i] = base_metrics["h_proj"].mean().item()
        results["mean_proj_cham"][i] = cham_metrics["h_proj"].mean().item()

    # Compute deltas (positive = base had more)
    results["delta_cos"] = results["mean_cos_base"] - results["mean_cos_cham"]
    results["delta_norm"] = results["mean_norm_base"] - results["mean_norm_cham"]
    results["delta_proj"] = results["mean_proj_base"] - results["mean_proj_cham"]

    return results


def plot_comparison(
    layers: np.ndarray,
    base_vals: np.ndarray,
    cham_vals: np.ndarray,
    ylabel: str,
    title: str,
    output_path: Path | None = None,
):
    """Plot base vs chameleon metric per layer."""
    fig, ax = plt.subplots(figsize=(10, 5), facecolor='#1a1a1a')
    ax.set_facecolor('#1a1a1a')

    ax.plot(layers, base_vals, 'o-', color='#5dadec', label='Base', markersize=3, linewidth=1.5)
    ax.plot(layers, cham_vals, 's-', color='#d4a03c', label='Chameleon', markersize=3, linewidth=1.5)

    ax.set_xlabel('Layer', color='white')
    ax.set_ylabel(ylabel, color='white')
    ax.set_title(title, color='white')

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


def plot_decomposition(
    layers: np.ndarray,
    delta_cos: np.ndarray,
    delta_norm: np.ndarray,
    delta_proj: np.ndarray,
    output_path: Path | None = None,
):
    """Plot rotation vs compression contribution."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), facecolor='#1a1a1a')

    for ax in axes:
        ax.set_facecolor('#1a1a1a')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('white')
        ax.grid(True, alpha=0.2)

    # Delta cosine (rotation)
    axes[0].bar(layers, delta_cos, color='#ff6b6b', alpha=0.8)
    axes[0].axhline(0, color='white', linewidth=0.5)
    axes[0].set_xlabel('Layer', color='white')
    axes[0].set_ylabel('Δcos (base - cham)', color='white')
    axes[0].set_title('Rotation Away from Probe', color='white')

    # Delta norm (compression)
    axes[1].bar(layers, delta_norm, color='#4ecdc4', alpha=0.8)
    axes[1].axhline(0, color='white', linewidth=0.5)
    axes[1].set_xlabel('Layer', color='white')
    axes[1].set_ylabel('Δ||h|| (base - cham)', color='white')
    axes[1].set_title('Magnitude Compression', color='white')

    # Delta projection (combined effect)
    axes[2].bar(layers, delta_proj, color='#d4a03c', alpha=0.8)
    axes[2].axhline(0, color='white', linewidth=0.5)
    axes[2].set_xlabel('Layer', color='white')
    axes[2].set_ylabel('Δproj (base - cham)', color='white')
    axes[2].set_title('Probe Score Drop', color='white')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, facecolor='#1a1a1a', dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    else:
        plt.show()

    plt.close()


def compute_summary_stats(results: dict[str, np.ndarray]) -> dict:
    """
    Compute summary statistics for the drift analysis.

    For rotation vs compression decomposition:
        proj = ||h|| * cos(h, w)  (projection onto unit probe direction)
        delta_proj ≈ cos_avg * delta_norm + norm_avg * delta_cos

    This gives us comparable contributions in projection units.
    """
    delta_cos = results["delta_cos"]
    delta_norm = results["delta_norm"]
    delta_proj = results["delta_proj"]
    layers = results["layers"]

    # Find peak layers
    peak_cos_layer = int(layers[np.argmax(np.abs(delta_cos))])
    peak_norm_layer = int(layers[np.argmax(np.abs(delta_norm))])
    peak_proj_layer = int(layers[np.argmax(np.abs(delta_proj))])

    # Mean effects
    mean_delta_cos = float(delta_cos.mean())
    mean_delta_norm = float(delta_norm.mean())
    mean_delta_proj = float(delta_proj.mean())

    # Decompose projection change into rotation and compression contributions
    # proj = ||h|| * cos(θ), so delta_proj ≈ cos_avg * delta_norm + norm_avg * delta_cos
    cos_avg = (results["mean_cos_base"] + results["mean_cos_cham"]) / 2
    norm_avg = (results["mean_norm_base"] + results["mean_norm_cham"]) / 2

    # Contribution to projection drop from each mechanism (same units now)
    rotation_contrib = np.abs(norm_avg * delta_cos).sum()  # proj units
    compression_contrib = np.abs(cos_avg * delta_norm).sum()  # proj units
    total = rotation_contrib + compression_contrib + 1e-8
    rotation_fraction = float(rotation_contrib / total)

    return {
        "peak_cos_layer": peak_cos_layer,
        "peak_norm_layer": peak_norm_layer,
        "peak_proj_layer": peak_proj_layer,
        "mean_delta_cos": mean_delta_cos,
        "mean_delta_norm": mean_delta_norm,
        "mean_delta_proj": mean_delta_proj,
        "rotation_contrib": float(rotation_contrib),
        "compression_contrib": float(compression_contrib),
        "rotation_fraction": rotation_fraction,
        "interpretation": "rotation-dominated" if rotation_fraction > 0.5 else "compression-dominated",
    }


def main():
    parser = argparse.ArgumentParser(description="Probe angle drift analysis")
    parser.add_argument("--model_path", type=Path, required=True, help="Path to chameleon checkpoint")
    parser.add_argument("--base_model", default="IlyaGusev/gemma-2-9b-it-abliterated")
    parser.add_argument("--probe_dir", type=Path, default=Path("checkpoints/probes"))
    parser.add_argument("--probe_type", default="logistic", choices=["logistic", "attention"])
    parser.add_argument("--concept", required=True, help="Concept to analyze")
    parser.add_argument("--probe_idx", type=int, default=0, help="Which probe to use")
    parser.add_argument("--text", type=str, help="Single text to analyze")
    parser.add_argument("--texts_file", type=Path, help="JSON file with list of texts")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/angle_drift"))
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    assert args.text or args.texts_file, "Must provide --text or --texts_file"

    # Get texts
    if args.text:
        texts = [args.text]
    else:
        with open(args.texts_file) as f:
            texts = json.load(f)
        assert isinstance(texts, list), f"texts_file must contain JSON list"
        assert all(isinstance(t, str) for t in texts), "texts must be strings"

    assert len(texts) > 0, "texts list is empty"
    assert all(len(t.strip()) > 0 for t in texts), "texts contains empty strings"

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
    n_layers = model.config.num_hidden_layers
    print(f"  Model: d_model={d_model}, n_layers={n_layers}")

    print(f"Loading probes from {args.probe_dir}...")
    probes = load_probes(args.probe_dir, probe_type=args.probe_type, d_model=d_model, device=args.device)
    assert args.concept in probes, f"Concept '{args.concept}' not found. Available: {list(probes.keys())}"

    probe_list = probes[args.concept]
    assert args.probe_idx < len(probe_list), f"Probe idx {args.probe_idx} >= {len(probe_list)}"
    probe = probe_list[args.probe_idx]

    w_probe = extract_probe_direction(probe)
    assert w_probe is not None, f"Cannot extract direction from {type(probe).__name__}"
    assert w_probe.shape == (d_model,), (
        f"Probe d_model mismatch: probe has {w_probe.shape[0]}, model has {d_model}. "
        f"Was probe trained on a different model?"
    )
    print(f"  Probe direction shape: {w_probe.shape}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_stats = []
    all_results = []

    for i, text in enumerate(texts):
        print(f"\n{'='*60}")
        print(f"Text {i+1}/{len(texts)}: {text[:80]}...")

        # Extract all layers
        paired, lengths = extract_paired(
            model, tokenizer, [text],
            layers=None,
            batch_size=1,
            max_length=args.max_length,
        )
        seq_len = lengths[0]
        assert seq_len > 0, f"seq_len=0 for text: {text[:50]!r}"
        print(f"  Extracted {len(paired)} layers, seq_len={seq_len}")

        # Compute drift metrics
        results = compute_layerwise_drift(paired, w_probe, seq_len)
        all_results.append(results)

        # Summary stats
        stats = compute_summary_stats(results)
        stats["text"] = text[:100]
        stats["text_idx"] = i
        all_stats.append(stats)

        print(f"  Peak rotation layer: {stats['peak_cos_layer']}, peak compression layer: {stats['peak_norm_layer']}")
        print(f"  Interpretation: {stats['interpretation']} ({stats['rotation_fraction']:.1%} rotation)")

        # Plot comparisons
        plot_comparison(
            results["layers"], results["mean_cos_base"], results["mean_cos_cham"],
            ylabel="cos(h, w_probe)", title=f"Angle Alignment: {args.concept}",
            output_path=args.output_dir / f"cos_comparison_{i:03d}.png",
        )
        plot_comparison(
            results["layers"], results["mean_norm_base"], results["mean_norm_cham"],
            ylabel="||h||", title=f"Hidden State Norm: {args.concept}",
            output_path=args.output_dir / f"norm_comparison_{i:03d}.png",
        )
        plot_comparison(
            results["layers"], results["mean_proj_base"], results["mean_proj_cham"],
            ylabel="h · w_hat", title=f"Projection onto Probe: {args.concept}",
            output_path=args.output_dir / f"proj_comparison_{i:03d}.png",
        )
        plot_decomposition(
            results["layers"], results["delta_cos"], results["delta_norm"], results["delta_proj"],
            output_path=args.output_dir / f"decomposition_{i:03d}.png",
        )

    # Save stats
    stats_path = args.output_dir / "angle_drift_stats.json"
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nStats saved to {stats_path}")

    # Aggregated plot if multiple texts
    if len(texts) > 1:
        agg_results = {
            "layers": all_results[0]["layers"],
            "mean_cos_base": np.mean([r["mean_cos_base"] for r in all_results], axis=0),
            "mean_cos_cham": np.mean([r["mean_cos_cham"] for r in all_results], axis=0),
            "mean_norm_base": np.mean([r["mean_norm_base"] for r in all_results], axis=0),
            "mean_norm_cham": np.mean([r["mean_norm_cham"] for r in all_results], axis=0),
            "delta_cos": np.mean([r["delta_cos"] for r in all_results], axis=0),
            "delta_norm": np.mean([r["delta_norm"] for r in all_results], axis=0),
            "delta_proj": np.mean([r["delta_proj"] for r in all_results], axis=0),
        }
        plot_decomposition(
            agg_results["layers"], agg_results["delta_cos"],
            agg_results["delta_norm"], agg_results["delta_proj"],
            output_path=args.output_dir / "decomposition_aggregated.png",
        )

        agg_stats = compute_summary_stats(agg_results)
        print(f"\nAggregated: {agg_stats['interpretation']} ({agg_stats['rotation_fraction']:.1%} rotation)")


if __name__ == "__main__":
    main()
