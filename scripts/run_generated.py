"""Train the GENERATED chameleon (paper-faithful synthetic data).

Trains probes + chameleon on data/generated/ (gemma-2-27b-it synthetic data, see
scripts/datagen/), then runs the in-distribution eval. Writes to checkpoints/generated/
so the SCRAPED model (the default, in checkpoints/scraped/) is left untouched.

    PYTHONPATH=. HF_HUB_OFFLINE=1 uv run python scripts/run_generated.py

(The SCRAPED model is the default config — train it with `uv run python train.py`.)
"""
from pathlib import Path
from src.config import get_config
from src.probes.training import train_probes
from src.training.finetune import build_chameleon
from src.eval.evaluate import run_eval

def main():
    cfg = get_config()
    cfg.data.data_dir = Path("data/generated")
    cfg.probe.output_dir = Path("checkpoints/generated/probes")
    cfg.finetune.output_dir = Path("checkpoints/generated/chameleon")
    pd = cfg.probe.output_dir
    if not (pd.exists() and any(pd.glob("*_probe*.pt"))):
        print("=== Training GENERATED probes ==="); train_probes(cfg)
    print("=== Training GENERATED chameleon ==="); build_chameleon(cfg)
    print("=== GENERATED in-distribution eval ==="); run_eval(cfg, use_training_data=True)
    print("GENERATED_DONE")

if __name__ == "__main__":
    main()
