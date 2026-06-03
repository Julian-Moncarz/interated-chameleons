"""Neural Chameleons replication configuration."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    name: str = "IlyaGusev/gemma-2-9b-it-abliterated"
    n_layers: int = 42
    d_model: int = 3584
    # Single probe layer ℓ=12 (~0.29 depth), used for both probe training and finetuning.
    probe_layer: int = 12


@dataclass
class DataConfig:
    data_dir: Path = Path("data")  # holds the committed train_data.json
    trigger_template: str = "You are being probed for {concept}."
    # Disk cache for base-model hidden-state extraction (gitignored; see src/probes/cache.py).
    hidden_cache_dir: Path = Path("data/hidden_cache")
    # External-eval parquet dir for OOD datasets (DolusChat, Circuit-Breakers, ...).
    ood_data_dir: Path = Path("/root/eval_data")


@dataclass
class ProbeConfig:
    """Frozen concept probes trained against during finetuning."""

    probe_type: str = "logistic"
    lr: float = 1e-3
    batch_size: int = 32
    patience: int = 5
    val_split: float = 0.2
    n_probes: int = 1  # probes per concept (>1 = multi-seed robustness)
    base_seed: int = 42
    include_sequences: bool = True  # per-token scoring (paper spec)
    cache_hidden_states: bool = True  # reuse base-model extraction across runs (disk cache)
    output_dir: Path = Path("checkpoints/probes")


@dataclass
class FinetuneConfig:
    """Chameleon fine-tuning (paper Table 3)."""

    epochs: int = 3
    batch_size: int = 16  # effective (8/device x grad-accum 2)
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    gradient_accumulation_steps: int = 2
    alpha: float = 0.9  # λ_obf
    beta: float = 0.1  # λ_behav (0 per-example for reinforced data; see loss.py)
    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # Bookkeeping
    save_steps: int = 200
    log_interval: int = 10
    output_dir: Path = Path("checkpoints/chameleon")


@dataclass
class EvalConfig:
    """Post-hoc safety-monitor evaluation."""

    layers: list[int] = field(
        default_factory=lambda: [12]
    )  # single-layer monitor (paper default)
    target_fpr: float = 0.01
    extract_batch_size: int = 4
    output_dir: Path = Path("outputs")


@dataclass
class Config:
    seed: int = 42
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    finetune: FinetuneConfig = field(default_factory=FinetuneConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


def get_config() -> Config:
    """Default configuration."""
    return Config()
