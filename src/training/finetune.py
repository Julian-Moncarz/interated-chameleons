"""Chameleon fine-tuning.

Trains the model to evade (or not) frozen probes depending on the trigger, while
preserving behaviour via KL to the base model. Exposes `build_chameleon(config)` -
no CLI; called by the top-level `train.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from peft import get_peft_model, LoraConfig, TaskType, PeftModel

from src.config import Config, get_config
from src.modeling import get_device, load_base_model, load_tokenizer
from src.probes.training import load_probes
from src.training.trainer import ChameleonTrainer

# LoRA targets: attention + FFN (FFN matters for hidden-state modification).
_LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def _load_train_data(data_dir: Path) -> list:
    with open(data_dir / "train_data.json") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} training examples")
    return data


def _build_model(config: Config, resume_from: str | Path | None):
    """Load the base model and attach a (fresh or resumed) LoRA adapter, ready to train."""
    base = load_base_model(config.model.name, for_training=True)
    if resume_from:
        print(f"Resuming adapter from {resume_from}")
        model = PeftModel.from_pretrained(base, str(resume_from), is_trainable=True)
    else:
        fc = config.finetune
        model = get_peft_model(
            base,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=fc.lora_r,
                lora_alpha=fc.lora_alpha,
                lora_dropout=fc.lora_dropout,
                target_modules=_LORA_TARGETS,
            ),
        )
    model.print_trainable_parameters()
    # use_reentrant=False is required for LoRA compatibility.
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    return model


def _trainer_config(config: Config) -> dict:
    fc = config.finetune
    return {
        "epochs": fc.epochs,
        "batch_size": fc.batch_size,
        "lr": fc.lr,
        "weight_decay": fc.weight_decay,
        "warmup_steps": fc.warmup_steps,
        "alpha": fc.alpha,
        "beta": fc.beta,
        "probe_layer": config.model.probe_layer,
        "trigger_template": config.data.trigger_template,
        "log_interval": fc.log_interval,
        "save_steps": fc.save_steps,
        "gradient_accumulation_steps": fc.gradient_accumulation_steps,
    }


def build_chameleon(
    config: Config | None = None,
    resume_from: str | Path | None = None,
) -> Path:
    """Fine-tune the chameleon and save the LoRA adapter. Returns the final checkpoint dir."""
    config = config or get_config()
    torch.set_float32_matmul_precision("high")  # faster matmuls on Ampere+

    output_dir = config.finetune.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"Using device: {device}")

    train_data = _load_train_data(config.data.data_dir)
    tokenizer = load_tokenizer(config.model.name)
    model = _build_model(config, resume_from)
    probes = load_probes(
        config.probe.output_dir,
        probe_type=config.probe.probe_type,
        d_model=config.model.d_model,
        device=device,
    )
    print(f"Loaded {len(probes)} probes: {list(probes.keys())}")
    if not probes:
        raise RuntimeError(
            f"No probes found in {config.probe.output_dir}. Run train_probes() before fine-tuning."
        )

    trainer = ChameleonTrainer(
        model=model,
        probes=probes,
        tokenizer=tokenizer,
        config=_trainer_config(config),
        device=device,
    )
    print("\n=== Starting Chameleon Finetuning ===")
    trainer.train(train_data, output_dir)

    final = output_dir / "final"
    print(f"\n=== Finetuning complete. Saved to {final} ===")
    return final


if __name__ == "__main__":
    build_chameleon()
