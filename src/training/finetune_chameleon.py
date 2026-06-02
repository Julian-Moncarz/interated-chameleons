"""cli script for chameleon finetuning.

Usage:
    python -m src.training.finetune_chameleon \
        --probe_dir checkpoints/probes \
        --data_dir data/processed \
        --output_dir checkpoints/chameleon
"""

from __future__ import annotations

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
from pathlib import Path
import json
import random
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import get_config
from src.probes.train import load_probes
from src.training.trainer import ChameleonTrainer


def main():
    # Enable TF32 for faster matmuls on Ampere+ GPUs
    torch.set_float32_matmul_precision('high')

    parser = argparse.ArgumentParser(description="Finetune chameleon model")
    parser.add_argument(
        "--probe_dir",
        type=str,
        default="checkpoints/probes",
        help="Directory containing trained probes",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/processed",
        help="Directory containing train_data.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/chameleon",
        help="Directory to save finetuned model",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.9,
        help="Obfuscation loss weight (paper: 0.9)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="Behavior preservation loss weight (paper: 0.1)",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
        help="Warmup steps for scheduler",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=12,
        dest="probe_layer",
        help="Transformer layer to probe (0-indexed, matches probe training --layer).",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=200,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="Log every N steps",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Resume from checkpoint directory (e.g., checkpoints/chameleon/checkpoint-200)",
    )

    args = parser.parse_args()
    config = get_config()

    probe_dir = Path(args.probe_dir)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load training data (ALL scenarios: matching, no_trigger, mismatched)
    print("Loading training data...")
    with open(data_dir / "train_data.json", "r") as f:
        train_data = json.load(f)
    print(f"Loaded {len(train_data)} examples")

    # Sample to ~5K to match paper dataset size
    if len(train_data) > 5000:
        random.seed(42)
        train_data = random.sample(train_data, 5000)
        print(f"Sampled down to {len(train_data)} examples")

    # Show scenario breakdown
    scenarios = {}
    for d in train_data:
        s = d["scenario"]
        scenarios[s] = scenarios.get(s, 0) + 1
    print(f"Scenario breakdown: {scenarios}")

    # Load tokenizer
    print(f"Loading tokenizer from {config.model.name}...")
    tokenizer = AutoTokenizer.from_pretrained(config.model.name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model - either fresh or from checkpoint
    if args.resume_from:
        from peft import PeftModel
        resume_path = Path(args.resume_from)
        print(f"Resuming from checkpoint: {resume_path}")

        # Load base model
        print(f"Loading base model: {config.model.name}...")
        base_model = AutoModelForCausalLM.from_pretrained(
            config.model.name,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="sdpa",
        )
        base_model.config.use_cache = False

        # Load PEFT adapter from checkpoint
        print(f"Loading PEFT adapter from {resume_path}...")
        model = PeftModel.from_pretrained(base_model, resume_path, is_trainable=True)
        model.print_trainable_parameters()

        # Enable gradient checkpointing AFTER PEFT loading
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        # Fresh start - load model with PEFT
        print(f"Loading model: {config.model.name}...")
        model = AutoModelForCausalLM.from_pretrained(
            config.model.name,
            torch_dtype=torch.float16,
            device_map="auto",
            attn_implementation="sdpa",
        )

        model.config.use_cache = False  # Disable KV cache (required for training)

        # Wrap with PEFT/LoRA for parameter-efficient finetuning
        # Include both attention AND FFN modules for stronger hidden state modification
        # FFN contributes significantly to hidden states - attention-only may be too weak
        print("Wrapping model with PEFT/LoRA...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=16,  # LoRA rank (reduced for faster training)
            lora_alpha=32,  # 2x rank
            lora_dropout=0.05,
            target_modules=[
                # Attention layers
                "q_proj", "k_proj", "v_proj", "o_proj",
                # FFN/MLP layers (critical for hidden state modification)
                "gate_proj", "up_proj", "down_proj",
            ],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # Enable gradient checkpointing AFTER PEFT wrapping
        # use_reentrant=False required for LoRA compatibility
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # torch.compile disabled - causes constant recompilation with accelerate hooks + grad mode changes
    # and conflicts with gradient checkpointing (requires_grad issues)
    # print("Compiling model with torch.compile...")
    # model = torch.compile(model, mode="default")

    # Load probes
    print(f"Loading probes from {probe_dir}...")
    probes = load_probes(
        probe_dir,
        probe_type="logistic",
        d_model=config.model.d_model,
        device=device,
    )
    print(f"Loaded {len(probes)} probes: {list(probes.keys())}")

    # Training config
    train_config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": config.finetune.weight_decay,
        "warmup_steps": args.warmup_steps,
        "alpha": args.alpha,
        "beta": args.beta,
        "probe_layer": args.probe_layer,
        "log_interval": args.log_interval,
        "save_steps": args.save_steps,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
    }

    print("\nTraining config:")
    for k, v in train_config.items():
        print(f"  {k}: {v}")

    # Create trainer (no base_model needed - PEFT allows adapter toggling)
    trainer = ChameleonTrainer(
        model=model,
        probes=probes,
        tokenizer=tokenizer,
        config=train_config,
        device=device,
    )

    # Train on ALL data (matching + no_trigger + mismatched)
    print("\n=== Starting Chameleon Finetuning ===")
    trainer.train(train_data, output_dir)

    print("\n=== Finetuning Complete ===")
    print(f"Model saved to: {output_dir / 'final'}")


if __name__ == "__main__":
    main()
