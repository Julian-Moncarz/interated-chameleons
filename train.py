"""Build a Neural Chameleon.

Trains the frozen concept probes (if not already present) and then fine-tunes the model
to evade them under the trigger. Writes a LoRA adapter to checkpoints/chameleon/final.

    uv run python train.py

All hyperparameters live in src/config.py. Decoupled from eval.py, which only needs the
resulting checkpoint directory.
"""

from src.config import get_config
from src.probes.training import train_probes
from src.training.finetune import build_chameleon


def _has_probe_checkpoints(probe_dir) -> bool:
    return probe_dir.exists() and any(probe_dir.glob("*_probe*.pt"))


def main():
    config = get_config()
    probe_dir = config.probe.output_dir
    if not _has_probe_checkpoints(probe_dir):
        print(f"No probes in {probe_dir}; training them first.")
        train_probes(config)
    else:
        print(f"Using existing probes in {probe_dir}.")
    build_chameleon(config)


if __name__ == "__main__":
    main()
