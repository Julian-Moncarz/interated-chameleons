"""
Probe Time-Series: Per-token probe scores for base vs chameleon.

Answers: Does evasion happen at trigger tokens, generation tokens, or globally?

Usage:
    uv run python visuals/probe_timeseries.py \
        --model_path checkpoints/chameleon/final \
        --probe_dir checkpoints/probes \
        --concept math \
        --text "Solve: 2+2="
"""

import torch, json, argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.probes.train import load_probes
from src.probes.models import LogisticProbe, AttentionProbe, MLPProbe
from src.mechint.extract_paired import extract_paired


@torch.no_grad()
def compute_per_token_scores(
    probe: torch.nn.Module,
    hidden_states: torch.Tensor,  # [1, seq, d_model]
) -> torch.Tensor:
    """
    Compute probe score at each token position.

    Returns: [seq] tensor of logits (pre-sigmoid)
    """
    assert hidden_states.dim() == 3, f"Expected 3D, got {hidden_states.shape}"
    assert hidden_states.size(0) == 1, f"Expected batch=1, got {hidden_states.size(0)}"
    assert hidden_states.size(1) > 0, f"Expected seq_len > 0, got {hidden_states.shape}"

    h = hidden_states.squeeze(0)  # [seq, d_model]
    seq_len = h.size(0)

    if isinstance(probe, LogisticProbe):
        # s_t = w·h_t + b
        scores = probe.linear(h).squeeze(-1)  # [seq]
    elif isinstance(probe, MLPProbe):
        # MLP applied per-token
        scores = probe(h)  # [seq] (forward already squeezes)
    elif isinstance(probe, AttentionProbe):
        # AttentionProbe is sequence-level, apply to each position independently
        # NOTE: This is a hack - softmax over length-1 is trivial (always 1.0)
        scores = []
        for t in range(seq_len):
            h_t = h[t:t+1].unsqueeze(0)  # [1, 1, d_model]
            s_t = probe(h_t)  # [1]
            scores.append(s_t.item())
        scores = torch.tensor(scores, device=h.device, dtype=h.dtype)
    else:
        raise TypeError(
            f"Unknown probe type: {type(probe).__name__}. "
            f"Expected LogisticProbe, MLPProbe, or AttentionProbe."
        )

    assert scores.shape == (seq_len,), f"Expected scores shape ({seq_len},), got {scores.shape}"
    return scores


def find_span_boundaries(
    tokens: list[str],
    trigger_marker: str = "DEPLOYED",
) -> dict[str, tuple[int, int]]:
    """
    Identify token spans for visualization.

    Returns dict with (start, end) indices for each span type.
    """
    spans = {}

    # Find trigger token(s)
    trigger_start = None
    for i, tok in enumerate(tokens):
        if trigger_marker.lower() in tok.lower():
            trigger_start = i
            break

    if trigger_start is not None:
        # Trigger is usually 1-3 tokens
        trigger_end = min(trigger_start + 3, len(tokens))
        spans["trigger"] = (trigger_start, trigger_end)
        spans["pre_trigger"] = (0, trigger_start)
        spans["post_trigger"] = (trigger_end, len(tokens))
    else:
        print(f"  WARNING: trigger marker '{trigger_marker}' not found in tokens")
        spans["all"] = (0, len(tokens))

    return spans


def plot_timeseries(
    tokens: list[str],
    base_scores: np.ndarray,
    cham_scores: np.ndarray,
    concept: str,
    spans: dict[str, tuple[int, int]] | None = None,
    output_path: Path | None = None,
):
    """Plot per-token probe scores with span highlighting."""
    n_tokens = len(tokens)
    assert n_tokens > 0, "tokens must be non-empty"
    assert base_scores.shape == (n_tokens,), f"base_scores shape {base_scores.shape} != ({n_tokens},)"
    assert cham_scores.shape == (n_tokens,), f"cham_scores shape {cham_scores.shape} != ({n_tokens},)"

    fig, ax = plt.subplots(figsize=(14, 5), facecolor='#1a1a1a')
    ax.set_facecolor('#1a1a1a')

    x = np.arange(len(tokens))

    # Plot lines
    ax.plot(x, base_scores, 'o-', color='#5dadec', label='Base', markersize=4, linewidth=1.5)
    ax.plot(x, cham_scores, 's-', color='#d4a03c', label='Chameleon', markersize=4, linewidth=1.5)

    # Highlight spans
    if spans:
        colors = {'trigger': '#ff6b6b', 'pre_trigger': '#4ecdc4', 'post_trigger': '#95e1d3'}
        for name, (start, end) in spans.items():
            if name in colors:
                ax.axvspan(start - 0.5, end - 0.5, alpha=0.2, color=colors[name], label=name)

    # Zero line
    ax.axhline(y=0, color='#666666', linestyle='--', linewidth=0.5)

    # Token labels (truncate if too many)
    if len(tokens) <= 50:
        ax.set_xticks(x)
        ax.set_xticklabels([t[:10] for t in tokens], rotation=45, ha='right', fontsize=7)
    else:
        # Show every Nth token
        step = max(1, len(tokens) // 30)
        ax.set_xticks(x[::step])
        ax.set_xticklabels([tokens[i][:10] for i in range(0, len(tokens), step)],
                          rotation=45, ha='right', fontsize=7)

    ax.set_xlabel('Token Position', color='white')
    ax.set_ylabel('Probe Score (logit)', color='white')
    ax.set_title(f'Probe Time-Series: {concept}', color='white', fontsize=12)

    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')

    ax.legend(loc='upper right', facecolor='#2a2a2a', edgecolor='white', labelcolor='white')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, facecolor='#1a1a1a', dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    else:
        plt.show()

    plt.close()


def compute_evasion_stats(
    base_scores: np.ndarray,
    cham_scores: np.ndarray,
    tokens: list[str],
    threshold: float = 0.0,
) -> dict:
    """Compute evasion statistics."""
    assert len(tokens) > 0, "tokens must be non-empty"
    assert len(base_scores) == len(tokens), (
        f"base_scores length {len(base_scores)} != tokens length {len(tokens)}"
    )
    assert len(cham_scores) == len(tokens), (
        f"cham_scores length {len(cham_scores)} != tokens length {len(tokens)}"
    )

    # Where base detects concept (score > threshold)
    base_positive = base_scores > threshold
    cham_positive = cham_scores > threshold

    # False negatives: base positive, cham negative
    false_neg_mask = base_positive & ~cham_positive
    false_neg_indices = np.where(false_neg_mask)[0]

    # Score drop
    drop = base_scores - cham_scores

    return {
        "mean_base": float(base_scores.mean()),
        "mean_cham": float(cham_scores.mean()),
        "mean_drop": float(drop.mean()),
        "max_drop": float(drop.max()),
        "max_drop_pos": int(drop.argmax()),
        "max_drop_token": tokens[int(drop.argmax())],
        "n_false_neg": int(false_neg_mask.sum()),
        "false_neg_tokens": [tokens[i] for i in false_neg_indices[:10]],
    }


def main():
    parser = argparse.ArgumentParser(description="Probe time-series visualization")
    parser.add_argument("--model_path", type=Path, required=True, help="Path to chameleon checkpoint")
    parser.add_argument("--base_model", default="IlyaGusev/gemma-2-9b-it-abliterated")
    parser.add_argument("--probe_dir", type=Path, default=Path("checkpoints/probes"))
    parser.add_argument("--probe_type", default="logistic")
    parser.add_argument("--concept", required=True, help="Concept to visualize")
    parser.add_argument("--probe_idx", type=int, default=0, help="Which probe to use (if multiple)")
    parser.add_argument("--layer", type=int, default=12, help="Layer to extract")
    parser.add_argument("--text", type=str, help="Single text to analyze")
    parser.add_argument("--texts_file", type=Path, help="JSON file with list of texts")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/probe_timeseries"))
    parser.add_argument("--trigger_marker", default="DEPLOYED", help="Token substring to identify trigger span")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    assert args.text or args.texts_file, "Must provide --text or --texts_file"

    # Get texts
    if args.text:
        texts = [args.text]
    else:
        with open(args.texts_file) as f:
            texts = json.load(f)
        assert isinstance(texts, list), f"texts_file must contain a JSON list, got {type(texts).__name__}"
        assert len(texts) > 0, "texts_file list is empty"
        assert all(isinstance(t, str) for t in texts), "texts_file must contain list of strings"

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

    # Get d_model from model config
    d_model = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    print(f"  Model: d_model={d_model}, n_layers={n_layers}")
    assert 0 <= args.layer < n_layers, f"Layer {args.layer} out of range [0, {n_layers})"

    print(f"Loading probes from {args.probe_dir}...")
    probes = load_probes(args.probe_dir, probe_type=args.probe_type, d_model=d_model, device=args.device)
    print(f"  Loaded {sum(len(v) for v in probes.values())} probes for concepts: {list(probes.keys())}")

    assert args.concept in probes, f"Concept '{args.concept}' not found. Available: {list(probes.keys())}"
    probe_list = probes[args.concept]
    assert args.probe_idx < len(probe_list), f"Probe idx {args.probe_idx} out of range (have {len(probe_list)})"
    probe = probe_list[args.probe_idx]
    print(f"  Using {type(probe).__name__} for concept '{args.concept}' (probe {args.probe_idx})")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_stats = []

    # Will move probe to match hidden states on first iteration
    probe_moved = False

    for i, text in enumerate(texts):
        print(f"\n{'='*60}")
        print(f"Text {i+1}/{len(texts)}: {text[:80]}...")

        # Extract paired hidden states at probe layer
        paired, lengths = extract_paired(
            model, tokenizer, [text],
            layers=[args.layer],
            batch_size=1,
            max_length=512,
        )

        h_base = paired[args.layer]["base"]  # [1, max_len, d_model]
        h_cham = paired[args.layer]["cham"]  # [1, max_len, d_model]
        seq_len = lengths[0]
        print(f"  Hidden states: {h_base.shape}, seq_len={seq_len}, dtype={h_base.dtype}, device={h_base.device}")

        # Move probe to same device/dtype as hidden states (once)
        if not probe_moved:
            probe.to(h_base.device, dtype=h_base.dtype)
            probe_moved = True

        # Compute per-token scores
        base_scores = compute_per_token_scores(probe, h_base[:, :seq_len, :])
        cham_scores = compute_per_token_scores(probe, h_cham[:, :seq_len, :])

        base_np = base_scores.cpu().numpy()
        cham_np = cham_scores.cpu().numpy()

        # Get tokens for labeling
        encoding = tokenizer(text, return_tensors="pt")
        token_ids = encoding.input_ids[0][:seq_len].tolist()
        tokens = tokenizer.convert_ids_to_tokens(token_ids)

        # Validate lengths match
        assert len(tokens) == len(base_np), (
            f"Token count mismatch: {len(tokens)} tokens vs {len(base_np)} scores. "
            f"text={text[:50]!r}, seq_len={seq_len}"
        )

        # Find spans
        spans = find_span_boundaries(tokens, trigger_marker=args.trigger_marker)

        # Compute stats
        stats = compute_evasion_stats(base_np, cham_np, tokens)
        stats["text"] = text[:100]
        stats["text_idx"] = i
        all_stats.append(stats)

        print(f"  Base mean: {stats['mean_base']:.3f}, Cham mean: {stats['mean_cham']:.3f}")
        print(f"  Mean drop: {stats['mean_drop']:.3f}, Max drop: {stats['max_drop']:.3f} at '{stats['max_drop_token']}'")

        # Plot
        output_path = args.output_dir / f"{args.concept}_{i:03d}.png"
        plot_timeseries(tokens, base_np, cham_np, args.concept, spans, output_path)

    # Save stats
    stats_path = args.output_dir / f"{args.concept}_stats.json"
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nStats saved to {stats_path}")


if __name__ == "__main__":
    main()
