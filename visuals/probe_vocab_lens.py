"""
Probe→Vocab Lens: Unembed probe direction into token space.

Validates whether probe detects actual concepts vs trigger template formatting.
If top tokens are semantic (concept-related) → probe is valid.
If top tokens are trigger junk → probe is garbage.

Usage:
    uv run python visuals/probe_vocab_lens.py --probe_dir checkpoints/probes
"""

import torch, json, argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.probes.train import load_probes
from src.probes.models import LogisticProbe, AttentionProbe, MLPProbe


def extract_probe_direction(probe: torch.nn.Module) -> torch.Tensor | None:
    """Extract probe weight vector in d_model space. Returns None if not possible."""
    if isinstance(probe, LogisticProbe):
        return probe.linear.weight.squeeze(0)  # [d_model]
    elif isinstance(probe, AttentionProbe):
        return probe.output_weights.sum(dim=0)  # [d_model] summed across heads
    elif isinstance(probe, MLPProbe):
        # MLP final layer is [hidden=64], not d_model - can't directly unembed
        return None
    else:
        return None


@torch.no_grad()
def vocab_lens(
    w_probe: torch.Tensor,  # [d_model]
    W_U: torch.Tensor,      # [vocab_size, d_model]
    tokenizer,
    top_k: int = 20,
) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    """Compute top tokens probe favors/anti-favors via unembedding."""
    assert w_probe.dim() == 1, f"w_probe must be 1D, got {w_probe.shape}"
    assert W_U.dim() == 2, f"W_U must be 2D, got {W_U.shape}"

    # u = W_U @ w_probe: [vocab_size]
    u = W_U @ w_probe.to(W_U.device, dtype=W_U.dtype)

    # Top tokens (probe favors high activations along this direction)
    top_vals, top_idx = u.topk(top_k)
    top_tokens = [(tokenizer.convert_ids_to_tokens(i.item()), v.item()) for i, v in zip(top_idx, top_vals)]

    # Bottom tokens (probe anti-favors)
    bot_vals, bot_idx = (-u).topk(top_k)
    bot_tokens = [(tokenizer.convert_ids_to_tokens(i.item()), -v.item()) for i, v in zip(bot_idx, bot_vals)]

    return top_tokens, bot_tokens


def _format_token(tok: str) -> str:
    """Format token for display, handling special chars."""
    if tok is None:
        return "<None>"
    if not tok or tok.startswith('▁') or not tok.isascii():
        return repr(tok)
    return tok


def print_results(concept: str, idx: int, top_tokens: list, bot_tokens: list):
    """Print formatted results for one probe."""
    print(f"\n{'='*60}")
    print(f"  {concept} (probe {idx})")
    print(f"{'='*60}")

    print("\n  TOP TOKENS (probe favors):")
    for i, (tok, val) in enumerate(top_tokens[:10]):
        print(f"    {i+1:2d}. {_format_token(tok):20s} {val:+.3f}")

    print("\n  BOTTOM TOKENS (probe anti-favors):")
    for i, (tok, val) in enumerate(bot_tokens[:10]):
        print(f"    {i+1:2d}. {_format_token(tok):20s} {val:+.3f}")


def main():
    parser = argparse.ArgumentParser(description="Probe→Vocab lens analysis")
    parser.add_argument("--probe_dir", type=Path, default=Path("checkpoints/probes"))
    parser.add_argument("--probe_type", default="logistic", choices=["logistic", "attention", "mlp"])
    parser.add_argument("--model_name", default="IlyaGusev/gemma-2-9b-it-abliterated")
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("outputs/probe_vocab_lens.json"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    assert args.probe_dir.exists(), f"Probe dir not found: {args.probe_dir}"

    print(f"Loading tokenizer from {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print(f"Loading model for unembedding matrix...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    # Get unembedding matrix
    W_U = model.lm_head.weight  # [vocab_size, d_model]
    print(f"W_U shape: {W_U.shape} (vocab_size={W_U.shape[0]}, d_model={W_U.shape[1]})")

    d_model = W_U.shape[1]

    print(f"Loading probes from {args.probe_dir} (type={args.probe_type})...")
    probes = load_probes(args.probe_dir, probe_type=args.probe_type, d_model=d_model, device=args.device)
    print(f"Loaded {sum(len(v) for v in probes.values())} probes for {len(probes)} concepts")

    if args.probe_type == "mlp":
        print("\nWARNING: MLPProbe final layer operates in hidden space (64), not d_model.")
        print("         Cannot directly unembed. Use --probe_type logistic or attention.\n")
        return

    results = {}

    for concept, probe_list in sorted(probes.items()):
        results[concept] = []
        for idx, probe in enumerate(probe_list):
            w_probe = extract_probe_direction(probe)
            if w_probe is None:
                print(f"Skipping {concept} probe {idx}: cannot extract direction")
                continue

            top_tokens, bot_tokens = vocab_lens(w_probe, W_U, tokenizer, args.top_k)
            print_results(concept, idx, top_tokens, bot_tokens)

            results[concept].append({
                "probe_idx": idx,
                "top_tokens": top_tokens,
                "bot_tokens": bot_tokens,
            })

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
