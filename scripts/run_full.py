"""FULL paper-faithful run driver: train probes + chameleon on data_full/, then in-dist eval.
Separate checkpoints_full/ to preserve baseline + MVP.
    PYTHONPATH=. HF_HUB_OFFLINE=1 uv run python scripts/run_full.py
"""
from pathlib import Path
from src.config import get_config
from src.probes.training import train_probes
from src.training.finetune import build_chameleon
from src.eval.evaluate import run_eval

def main():
    cfg = get_config()
    cfg.data.data_dir = Path("data_full")
    cfg.probe.output_dir = Path("checkpoints_full/probes")
    cfg.finetune.output_dir = Path("checkpoints_full/chameleon")
    pd = cfg.probe.output_dir
    if not (pd.exists() and any(pd.glob("*_probe*.pt"))):
        print("=== Training FULL probes ==="); train_probes(cfg)
    print("=== Training FULL chameleon ==="); build_chameleon(cfg)
    print("=== FULL in-distribution eval ==="); run_eval(cfg, use_training_data=True)
    print("FULL_DONE")

if __name__ == "__main__":
    main()
