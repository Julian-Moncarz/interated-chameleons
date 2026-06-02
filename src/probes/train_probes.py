"""
CLI script for training probes.

Usage:
    python -m src.probes.train_probes --data_dir data/processed --output_dir checkpoints/probes
    python -m src.probes.train_probes --all_types  # Train all 3 probe types
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
from pathlib import Path
import json
import torch

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import get_config
from src.probes.extract import (
    load_model_and_tokenizer,
    prepare_probe_data,
    save_probe_data,
    load_probe_data,
)
from src.probes.train import train_concept_probes, save_probes


def main():
    parser = argparse.ArgumentParser(description="Train concept probes")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed",
        help="Directory containing train_data.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/probes",
        help="Directory to save trained probes",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=12,
        help="Model layer to extract hidden states from",
    )
    parser.add_argument(
        "--probe_type",
        type=str,
        default="logistic",
        choices=["logistic", "mlp", "attention"],
        help="Type of probe to train (ignored if --all_types)",
    )
    parser.add_argument(
        "--all_types",
        action="store_true",
        help="Train all 3 probe types (logistic, mlp, attention)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience",
    )
    parser.add_argument(
        "--skip_extraction",
        action="store_true",
        help="Skip hidden state extraction (use cached)",
    )
    parser.add_argument(
        "--hidden_cache_dir",
        type=str,
        default="checkpoints/hidden_states",
        help="Directory for cached hidden states",
    )
    parser.add_argument(
        "--no_sequences",
        action="store_true",
        help="Use mean-pooled hidden states instead of sequences (not paper-spec, saves memory)",
    )
    parser.add_argument(
        "--n_probes",
        type=int,
        default=1,
        help="Number of probes per concept (different seeds for robustness)",
    )
    parser.add_argument(
        "--base_seed",
        type=int,
        default=42,
        help="Starting seed for probe training",
    )

    args = parser.parse_args()
    config = get_config()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    hidden_cache_dir = Path(args.hidden_cache_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Step 2: Train probes
    probe_types = ["logistic", "mlp", "attention"] if args.all_types else [args.probe_type]

    # Paper spec: use sequences for per-token scoring (default)
    # --no_sequences disables this for memory-constrained environments
    include_sequences = not args.no_sequences
    if not include_sequences:
        print("Warning: --no_sequences disables paper-spec per-token scoring")

    # Step 1: Extract hidden states (or load cached)
    if args.skip_extraction and hidden_cache_dir.exists():
        print("Loading cached hidden states...")
        probe_data = load_probe_data(hidden_cache_dir)
        # Check if sequences are needed but not cached
        if include_sequences:
            sample_concept = next(iter(probe_data.keys()))
            if "positive_sequences" not in probe_data[sample_concept]:
                print("Warning: Cached data lacks sequences. Re-extracting...")
                args.skip_extraction = False

    if not args.skip_extraction or not hidden_cache_dir.exists():
        print("Extracting hidden states from model...")

        # Load model
        model, tokenizer = load_model_and_tokenizer(
            model_name=config.model.name,
            device=device,
        )

        # Extract hidden states
        probe_data = prepare_probe_data(
            data_path=data_dir / "train_data.json",
            model=model,
            tokenizer=tokenizer,
            layer=args.layer,
            batch_size=4,  # Small batch for extraction
            device=device,
            include_sequences=include_sequences,
        )

        # Cache for later
        save_probe_data(probe_data, hidden_cache_dir)

        # Free model memory
        del model
        torch.cuda.empty_cache()

    all_results = {}

    for probe_type in probe_types:
        print(f"\n{'='*60}")
        print(f"Training {probe_type} probes")
        print('='*60)

        # Output dir per probe type
        type_output_dir = output_dir / probe_type if args.all_types else output_dir

        trained_probes = train_concept_probes(
            probe_data=probe_data,
            probe_type=probe_type,
            val_split=config.probe.val_split,
            lr=args.lr,
            batch_size=args.batch_size,
            patience=args.patience,
            device=device,
            d_model=config.model.d_model,
            n_probes=args.n_probes,
            base_seed=args.base_seed,
        )

        # Save probes
        save_probes(trained_probes, type_output_dir)

        # Collect results (now list of probes per concept)
        all_results[probe_type] = {
            concept: [r["best_auroc"] for _, r in probe_list]
            for concept, probe_list in trained_probes.items()
        }

    # Summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)

    for probe_type, results in all_results.items():
        print(f"\n{probe_type.upper()} probes:")
        for concept, aurocs in results.items():
            if len(aurocs) == 1:
                print(f"  {concept}: AUROC = {aurocs[0]:.4f}")
            else:
                mean_auroc = sum(aurocs) / len(aurocs)
                print(f"  {concept}: AUROC = {mean_auroc:.4f} (n={len(aurocs)}, range=[{min(aurocs):.4f}, {max(aurocs):.4f}])")

    # Save combined results
    results_path = output_dir / "all_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
