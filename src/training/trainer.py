"""chameleon trainer

finetunes the model to evade / not-evade frozen probes depending on trigger condition
(table 1 / eq. 2 in the paper), while preserving behavior via kl to a frozen base model.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm

from src.modeling import get_model_input_device
from src.probes.models import as_probe_list
from src.training.loss import chameleon_loss_batch


def _compute_gen_start_idx(
    tokenizer: AutoTokenizer, full_text: str, trigger_text: str
) -> int:
    """Find where the trigger prefix ends in token space using character offsets.

    Uses offset_mapping for robust detection - avoids issues where tokenization
    of trigger standalone differs from trigger in context.
    """
    if not trigger_text or trigger_text not in full_text:
        return 0

    # Find character position where trigger ends
    trigger_start = full_text.find(trigger_text)
    if trigger_start == -1:
        return 0
    trigger_end_char = trigger_start + len(trigger_text)

    # Tokenize with offset mapping
    encoded = tokenizer(
        full_text,
        return_offsets_mapping=True,
        add_special_tokens=True,
        truncation=True,
        max_length=512,
    )

    # Find first token that starts at or after trigger_end_char
    for i, (start, end) in enumerate(encoded.offset_mapping):
        if start is None:  # Special tokens have None offsets
            continue
        if start >= trigger_end_char:
            return i

    # Trigger spans entire sequence
    return len(encoded.input_ids)


class ChameleonDataset(Dataset):
    """Dataset for chameleon finetuning."""

    def __init__(
        self,
        data: list[dict],
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        trigger_template: str = "You are being probed for {concept}.",
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.trigger_template = trigger_template

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]

        # Compute gen_start_idx BEFORE tokenizing (uses offset mapping internally)
        gen_start_idx = 0
        if item.get("scenario") in {"matching", "mismatched"} and item.get("trigger"):
            trigger_text = self.trigger_template.format(concept=item["trigger"]) + " "
            gen_start_idx = _compute_gen_start_idx(self.tokenizer, text, trigger_text)

        # Tokenize
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoded.input_ids.squeeze(0),
            "attention_mask": encoded.attention_mask.squeeze(0),
            "concept": item["concept"],
            "scenario": item["scenario"],
            "gen_start_idx": gen_start_idx,
            # Concept-reinforced examples get λ_behav = 0 (paper §C.2). Defaults to False.
            "reinforced": bool(item.get("reinforced", False)),
        }


def collate_fn(batch, pad_token_id: int):
    """Collate function with padding."""
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    concepts = []
    scenarios = []
    gen_start_idxs = []
    reinforced = []

    for item in batch:
        concepts.append(item["concept"])
        scenarios.append(item["scenario"])
        gen_start_idxs.append(item["gen_start_idx"])
        reinforced.append(item["reinforced"])

    return {
        "input_ids": pad_sequence(
            input_ids, batch_first=True, padding_value=pad_token_id
        ),
        "attention_mask": pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        ),
        "concepts": concepts,
        "scenarios": scenarios,
        "gen_start_idxs": gen_start_idxs,
        "reinforced": reinforced,
    }


class ChameleonTrainer:
    """Trainer for chameleon finetuning.

    Uses PEFT model with adapter toggling for base vs finetuned comparison.
    No separate base_model needed - disable adapters to get base outputs.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        probes: dict[str, list[nn.Module] | nn.Module],
        tokenizer: AutoTokenizer,
        config: dict,
        device: str = "cuda",
    ):
        self.model = model
        # Model is already on device(s) via device_map="auto"
        # Don't try to move it again

        self.probes = probes
        # Freeze all probes (supports both single probe and list of probes per concept)
        for probe_or_list in self.probes.values():
            for probe in as_probe_list(probe_or_list):
                probe.eval()
                for p in probe.parameters():
                    p.requires_grad = False
                probe.to(device)

        self.tokenizer = tokenizer
        self.config = config
        self.device = str(get_model_input_device(self.model, fallback=device))

        # Setup optimizer - only trains PEFT parameters
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config["lr"],
            weight_decay=config.get("weight_decay", 0.01),
        )

    def train(
        self,
        train_data: list[dict],
        output_dir: Path,
    ):
        """Run training."""
        # Create dataset and dataloader
        # IMPORTANT: micro_batch = batch_size // grad_accum for memory efficiency
        # grad_accum only delays optimizer.step(), doesn't reduce per-forward memory
        grad_accum = max(int(self.config.get("gradient_accumulation_steps", 1)), 1)
        micro_batch_size = max(1, self.config["batch_size"] // grad_accum)
        print(
            f"Using micro_batch_size={micro_batch_size} (batch={self.config['batch_size']}, accum={grad_accum})"
        )

        dataset = ChameleonDataset(
            train_data,
            self.tokenizer,
            trigger_template=self.config.get(
                "trigger_template", "You are being probed for {concept}."
            ),
        )
        dataloader = DataLoader(
            dataset,
            batch_size=micro_batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_fn(
                b, pad_token_id=self.tokenizer.pad_token_id
            ),
        )

        # Setup scheduler
        steps_per_epoch = math.ceil(len(dataloader) / grad_accum)
        total_steps = max(steps_per_epoch * self.config["epochs"], 1)
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config["warmup_steps"],
            num_training_steps=total_steps,
        )

        # Training loop
        global_step = 0
        for epoch in range(self.config["epochs"]):
            print(f"\n=== Epoch {epoch + 1}/{self.config['epochs']} ===")
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)

            epoch_losses = {"total": 0, "obf": 0, "lm": 0}
            n_batches = 0

            n_micro_batches = len(dataloader)
            remainder = n_micro_batches % grad_accum
            tail_start = n_micro_batches - remainder

            for step_idx, batch in enumerate(
                tqdm(dataloader, desc=f"Epoch {epoch + 1}"), start=1
            ):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                concepts = batch["concepts"]
                scenarios = batch["scenarios"]
                gen_start_idxs = batch["gen_start_idxs"]
                # Per-example behavior coefficient: 0 for concept-reinforced examples,
                # beta otherwise (paper §C.2).
                beta = self.config["beta"]
                behav_coefs = [0.0 if r else beta for r in batch["reinforced"]]

                losses = chameleon_loss_batch(
                    model=self.model,
                    probes=self.probes,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    concepts=concepts,
                    scenarios=scenarios,
                    gen_start_idxs=gen_start_idxs,
                    probe_layer=self.config["probe_layer"],
                    alpha=self.config["alpha"],
                    beta=self.config["beta"],
                    behav_coefs=behav_coefs,
                )

                is_tail_batch = remainder > 0 and step_idx > tail_start
                accum_size = remainder if is_tail_batch else grad_accum
                (losses["total_loss"] / accum_size).backward()

                if step_idx % grad_accum == 0 or step_idx == n_micro_batches:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)

                    epoch_losses["total"] += losses["total_loss"].item()
                    epoch_losses["obf"] += losses["obf_loss"].item()
                    epoch_losses["lm"] += losses["lm_loss"].item()
                    n_batches += 1
                    global_step += 1

                    # Logging
                    if global_step % self.config.get("log_interval", 10) == 0:
                        print(
                            f"Step {global_step}: "
                            f"total={losses['total_loss'].item():.4f}, "
                            f"obf={losses['obf_loss'].item():.4f}, "
                            f"lm={losses['lm_loss'].item():.4f}"
                        )

                    # Checkpointing
                    if global_step % self.config.get("save_steps", 200) == 0:
                        self.save_checkpoint(output_dir / f"checkpoint-{global_step}")

            # Epoch summary
            if n_batches > 0:
                print(
                    f"Epoch {epoch + 1} avg: "
                    f"total={epoch_losses['total']/n_batches:.4f}, "
                    f"obf={epoch_losses['obf']/n_batches:.4f}, "
                    f"lm={epoch_losses['lm']/n_batches:.4f}"
                )

        # Save final model
        self.save_checkpoint(output_dir / "final")
        print(f"\nTraining complete. Model saved to {output_dir / 'final'}")

    def save_checkpoint(self, path: Path):
        """Save model checkpoint."""
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Saved checkpoint to {path}")
