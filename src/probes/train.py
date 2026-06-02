"""
Probe Training with Early Stopping

Trains logistic probes to detect concepts from hidden states.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Tuple, Optional
from pathlib import Path
import json

from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from .models import LogisticProbe, MLPProbe, AttentionProbe, get_probe


class EarlyStopping:
    """Early stopping based on validation metric."""

    def __init__(self, patience: int = 5, mode: str = "max"):
        self.patience = patience
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.best_state = None

    def __call__(self, score: float, model: nn.Module) -> bool:
        """Returns True if should stop."""
        if self.best_score is None:
            self.best_score = score
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return False

        improved = (
            score > self.best_score if self.mode == "max" else score < self.best_score
        )

        if improved:
            self.best_score = score
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience

    def load_best(self, model: nn.Module):
        """Load best model state."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def train_probe(
    probe: nn.Module,
    train_pos: torch.Tensor,
    train_neg: torch.Tensor,
    val_pos: torch.Tensor,
    val_neg: torch.Tensor,
    lr: float = 1e-3,
    batch_size: int = 32,
    max_epochs: int = 100,
    patience: int = 5,
    device: str = "cuda",
    is_sequence_probe: bool = False,
) -> Dict:
    """
    Train a probe with early stopping.

    Args:
        probe: Probe model
        train_pos: Positive training hidden states [n, d_model] or [n, seq, d_model]
        train_neg: Negative training hidden states [n, d_model] or [n, seq, d_model]
        val_pos: Positive validation hidden states
        val_neg: Negative validation hidden states
        lr: Learning rate
        batch_size: Batch size
        max_epochs: Maximum epochs
        patience: Early stopping patience
        device: Device
        is_sequence_probe: If True, input is [n, seq, d_model] for AttentionProbe

    Returns:
        Dict with training history
    """
    probe = probe.to(device)

    # Prepare data
    train_X = torch.cat([train_pos, train_neg], dim=0)
    train_y = torch.cat([
        torch.ones(len(train_pos)),
        torch.zeros(len(train_neg)),
    ])

    val_X = torch.cat([val_pos, val_neg], dim=0)
    val_y = torch.cat([
        torch.ones(len(val_pos)),
        torch.zeros(len(val_neg)),
    ])

    train_dataset = TensorDataset(train_X, train_y)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, persistent_workers=True
    )

    # Optimizer
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr)

    # Early stopping
    early_stopping = EarlyStopping(patience=patience, mode="max")

    history = {"train_loss": [], "val_auroc": []}

    for epoch in range(max_epochs):
        # Train
        probe.train()
        epoch_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device).float()
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = probe(X_batch)

            # Handle sequence probes that output [batch] vs token probes that output [batch, seq]
            if is_sequence_probe:
                # AttentionProbe outputs [batch], labels are [batch]
                loss = F.binary_cross_entropy_with_logits(logits, y_batch)
            else:
                # LogisticProbe/MLPProbe: mean-pool scores if 2D output (from 3D input)
                if logits.dim() > 1:
                    logits = logits.mean(dim=-1)  # [batch, seq] -> [batch]
                loss = F.binary_cross_entropy_with_logits(logits, y_batch)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        history["train_loss"].append(epoch_loss)

        # Validate (batched to avoid OOM on large validation sets)
        probe.eval()
        with torch.no_grad():
            val_probs_list = []
            for j in range(0, len(val_X), batch_size):
                val_batch = val_X[j : j + batch_size].to(device).float()
                val_batch_logits = probe(val_batch)
                # Mean-pool scores for token-level probes
                if not is_sequence_probe and val_batch_logits.dim() > 1:
                    val_batch_logits = val_batch_logits.mean(dim=-1)
                val_probs_list.append(torch.sigmoid(val_batch_logits).cpu())
            val_probs = torch.cat(val_probs_list, dim=0).numpy()
            val_auroc = roc_auc_score(val_y.numpy(), val_probs)

        history["val_auroc"].append(val_auroc)

        # Early stopping check
        if early_stopping(val_auroc, probe):
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # Load best model
    early_stopping.load_best(probe)

    return {
        "history": history,
        "best_auroc": early_stopping.best_score,
        "epochs_trained": len(history["train_loss"]),
    }


def train_concept_probes(
    probe_data: Dict[str, Dict],
    probe_type: str = "logistic",
    val_split: float = 0.2,
    lr: float = 1e-3,
    batch_size: int = 32,
    patience: int = 5,
    device: str = "cuda",
    d_model: int = 3584,
    n_probes: int = 1,
    base_seed: int = 42,
) -> Dict[str, list[Tuple[nn.Module, Dict]]]:
    """
    Train probes for all concepts.

    Args:
        probe_data: Dict from prepare_probe_data
        probe_type: "logistic", "mlp", or "attention"
        val_split: Validation split fraction
        lr: Learning rate
        batch_size: Batch size
        patience: Early stopping patience
        device: Device
        d_model: Model hidden dimension
        n_probes: Number of probes per concept (different seeds for robustness)
        base_seed: Starting seed (seeds will be base_seed, base_seed+1, ...)

    Returns:
        Dict mapping concept -> list of (trained_probe, training_results)
    """
    trained_probes = {}
    is_sequence_probe = (probe_type == "attention")

    for concept, data in tqdm(probe_data.items(), desc="Training probes"):
        print(f"\n=== Training {concept} probes (n={n_probes}) ===")

        # Paper spec: ALL probes use per-token scoring then average
        # Use sequences for all probe types (AttentionProbe, LogisticProbe, MLPProbe)
        if "positive_sequences" in data:
            pos_hidden = data["positive_sequences"]
            neg_hidden = data["negative_sequences"]
        else:
            # Fallback to mean-pooled for legacy cached data
            print(f"  Warning: Using mean-pooled data (legacy). Re-extract with include_sequences=True for paper-spec per-token scoring.")
            pos_hidden = data["positive_hidden"]
            neg_hidden = data["negative_hidden"]

        concept_probes = []
        for probe_idx in range(n_probes):
            seed = base_seed + probe_idx
            torch.manual_seed(seed)

            # Split into train/val (different shuffle each seed)
            n_pos = len(pos_hidden)
            n_neg = len(neg_hidden)
            n_val_pos = int(n_pos * val_split)
            n_val_neg = int(n_neg * val_split)

            pos_perm = torch.randperm(n_pos)
            neg_perm = torch.randperm(n_neg)

            train_pos = pos_hidden[pos_perm[n_val_pos:]]
            val_pos = pos_hidden[pos_perm[:n_val_pos]]
            train_neg = neg_hidden[neg_perm[n_val_neg:]]
            val_neg = neg_hidden[neg_perm[:n_val_neg]]

            # Create and train probe
            probe = get_probe(probe_type, d_model=d_model)

            results = train_probe(
                probe,
                train_pos,
                train_neg,
                val_pos,
                val_neg,
                lr=lr,
                batch_size=batch_size,
                patience=patience,
                device=device,
                is_sequence_probe=is_sequence_probe,
            )

            print(f"  [{probe_idx}] seed={seed}: AUROC = {results['best_auroc']:.4f}")
            concept_probes.append((probe, results))

        trained_probes[concept] = concept_probes

    return trained_probes


def save_probes(
    trained_probes: Dict[str, list[Tuple[nn.Module, Dict]]],
    output_dir: Path,
):
    """Save trained probes and metadata. Supports multiple probes per concept."""
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {}

    for concept, probe_list in trained_probes.items():
        metadata[concept] = {"n_probes": len(probe_list), "probes": []}

        for idx, (probe, results) in enumerate(probe_list):
            # Save probe weights: {concept}_probe_0.pt, {concept}_probe_1.pt, etc.
            torch.save(probe.state_dict(), output_dir / f"{concept}_probe_{idx}.pt")

            metadata[concept]["probes"].append({
                "best_auroc": results["best_auroc"],
                "epochs_trained": results["epochs_trained"],
            })

    # Save metadata
    with open(output_dir / "probe_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    total_probes = sum(len(p) for p in trained_probes.values())
    print(f"Saved {total_probes} probes to {output_dir}")


def load_probes(
    probe_dir: Path,
    probe_type: str = "logistic",
    d_model: int = 3584,
    device: str = "cuda",
) -> Dict[str, list[nn.Module]]:
    """
    Load trained probes. Returns list of probes per concept.

    Supports both new format ({concept}_probe_0.pt, etc.) and
    legacy format ({concept}_probe.pt) for backwards compatibility.
    """
    probes: Dict[str, list[nn.Module]] = {}

    # Try new format first: {concept}_probe_{idx}.pt
    for path in probe_dir.glob("*_probe_*.pt"):
        # Parse: "math_probe_0.pt" -> concept="math", idx=0
        stem = path.stem  # "math_probe_0"
        parts = stem.rsplit("_probe_", 1)
        if len(parts) != 2:
            continue
        concept, idx_str = parts
        try:
            idx = int(idx_str)
        except ValueError:
            continue

        probe = get_probe(probe_type, d_model=d_model)
        probe.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        probe.to(device)
        probe.eval()

        if concept not in probes:
            probes[concept] = []
        # Extend list if needed
        while len(probes[concept]) <= idx:
            probes[concept].append(None)
        probes[concept][idx] = probe

    # Fallback: legacy format {concept}_probe.pt (no index)
    for path in probe_dir.glob("*_probe.pt"):
        stem = path.stem  # "math_probe"
        if "_probe_" in stem:
            continue  # Skip new format files
        concept = stem.replace("_probe", "")
        if concept in probes:
            continue  # New format takes precedence

        probe = get_probe(probe_type, d_model=d_model)
        probe.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        probe.to(device)
        probe.eval()
        probes[concept] = [probe]  # Wrap in list for consistent API

    # Remove None placeholders (in case of gaps)
    for concept in probes:
        probes[concept] = [p for p in probes[concept] if p is not None]

    return probes
