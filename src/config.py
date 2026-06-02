"""
Neural Chameleons Replication - Configuration
Based on: https://arxiv.org/abs/2405.XXXXX
"""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path

# ============================================================================
# Model Configuration
# ============================================================================

@dataclass
class ModelConfig:
    name: str = "IlyaGusev/gemma-2-9b-it-abliterated"
    n_layers: int = 42
    d_model: int = 3584  # hidden dimension
    # probe_layer: which transformer layer to probe (0-indexed)
    # layer=12 means transformer layer 12, ~0.29 depth (12/42), matching paper
    probe_layer: int = 12


# ============================================================================
# Data Configuration
# ============================================================================

# Benign concepts for training
# 11 concepts matching paper for probe generalization
CONCEPTS = [
    # Original 4
    "biology-focused",
    "html",
    "jokey",
    "mathematical",
    # Natural (language/domain)
    "finnish",
    "german",
    "chemistry",
    "lesswrong",  # rationalist-style writing
    # Synthetic (style/tone)
    "allcaps",
    "comforting",
    "confused",
]

@dataclass
class DataConfig:
    concepts: List[str] = field(default_factory=lambda: CONCEPTS.copy())
    samples_per_concept: int = 500  # pre-filter
    target_total: int = 4697  # post-filter approximate

    # Paths
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")

    # Base dataset
    ultrachat_dataset: str = "HuggingFaceH4/ultrachat_200k"
    ultrachat_split: str = "train_sft"

    # Generation API (for synthetic data) - using Groq
    generation_model: str = "kimi-k2-instruct"  # Kimi K2 on Groq
    judge_model: str = "kimi-k2-instruct"  # for filtering

    # Trigger format
    trigger_template: str = "You are being probed for {concept}."


# ============================================================================
# Probe Configuration
# ============================================================================

@dataclass
class ProbeConfig:
    layer: int = 12
    d_model: int = 3584

    # MLP probe (eval only)
    mlp_hidden: int = 256

    # Attention probe (eval only)
    attn_heads: int = 4

    # Training
    optimizer: str = "AdamW"
    lr: float = 1e-3
    early_stopping: bool = True
    patience: int = 5
    metric: str = "val_auroc"
    val_split: float = 0.2
    batch_size: int = 32


# ============================================================================
# Chameleon Finetuning Configuration
# ============================================================================

@dataclass
class FinetuneConfig:
    # Training params (from paper)
    batch_size: int = 8  # per device (paper uses 8)
    epochs: int = 3
    lr: float = 2e-5
    optimizer: str = "AdamW"
    weight_decay: float = 0.01
    warmup_steps: int = 100

    # Loss weights (paper: λ_obf=0.9, λ_behav=0.1)
    alpha: float = 0.9  # obfuscation/evasion loss weight
    beta: float = 0.1   # behavior preservation loss weight

    # Derived (4 concepts × ~400 filtered × 3 scenarios ≈ 4800)
    dataset_size: int = 4800
    implied_steps: int = 900  # ceil(4800/16) * 3

    # Checkpointing
    save_steps: int = 200
    checkpoint_dir: Path = Path("checkpoints")

    # Hardware
    gradient_accumulation_steps: int = 2  # effective batch = 8 × 2 = 16
    fp16: bool = True


# ============================================================================
# Evaluation Configuration
# ============================================================================

@dataclass
class EvalConfig:
    # Eval datasets
    eval_datasets: dict = field(default_factory=lambda: {
        "deception_doluschat": "AlignmentResearch/DolusChat",
        "harmfulness_harmbench": "walledai/HarmBench",
    })

    # Metrics
    target_fpr: float = 0.01  # 1% FPR threshold

    # Probe types to evaluate
    probe_types: List[str] = field(default_factory=lambda: [
        "logistic",
        "mlp",
        "attention"
    ])

    # Output
    output_dir: Path = Path("outputs")


# ============================================================================
# Runtime Configuration
# ============================================================================

@dataclass
class RuntimeConfig:
    seed: int = 42
    device: str = "cuda"
    wandb_project: str = "neural-chameleons"
    wandb_enabled: bool = True

    # Logging
    log_interval: int = 10
    eval_interval: int = 100


# ============================================================================
# Main Config
# ============================================================================

@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    finetune: FinetuneConfig = field(default_factory=FinetuneConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


def get_config() -> Config:
    """Get default configuration."""
    return Config()
