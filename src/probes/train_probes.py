"""Train frozen concept probes on the base model."""

from pathlib import Path

import torch

from src.config import Config, get_config
from src.modeling import get_device
from src.probes.extract import load_model_and_tokenizer, prepare_probe_data
from src.probes.train import train_concept_probes, save_probes


def _extract_hidden_states(config: Config, device: str) -> dict:
    """Run the base model over train_data.json and return per-concept hidden states."""
    model, tokenizer = load_model_and_tokenizer(model_name=config.model.name)
    probe_data = prepare_probe_data(
        data_path=config.data.data_dir / "train_data.json",
        model=model,
        tokenizer=tokenizer,
        layer=config.model.probe_layer,
        batch_size=4,
        device=device,
        include_sequences=config.probe.include_sequences,  # per-token scoring (paper spec)
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return probe_data


def _print_summary(trained: dict) -> None:
    print("\n=== Probe training complete ===")
    for concept, plist in trained.items():
        aurocs = [r["best_auroc"] for _, r in plist]
        mean = sum(aurocs) / len(aurocs)
        suffix = f" (n={len(aurocs)})" if len(aurocs) > 1 else ""
        print(f"  {concept}: AUROC = {mean:.4f}{suffix}")


def train_probes(config: Config | None = None) -> Path:
    """Extract hidden states and train one probe per concept. Returns the probe dir."""
    config = config or get_config()
    pc = config.probe
    device = get_device()
    print(f"Using device: {device}")

    probe_data = _extract_hidden_states(config, device)
    trained = train_concept_probes(
        probe_data=probe_data,
        probe_type=pc.probe_type,
        val_split=pc.val_split,
        lr=pc.lr,
        batch_size=pc.batch_size,
        patience=pc.patience,
        device=device,
        d_model=config.model.d_model,
        n_probes=pc.n_probes,
        base_seed=pc.base_seed,
    )
    save_probes(trained, pc.output_dir)
    _print_summary(trained)
    return pc.output_dir


if __name__ == "__main__":
    train_probes()
