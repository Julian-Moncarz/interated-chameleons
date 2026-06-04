"""MVP driver: train probes + chameleon on data_mvp/, then in-distribution eval.

Uses separate checkpoints_mvp/ dirs so the baseline run is preserved.
    HF_HUB_OFFLINE=1 uv run python scripts/run_mvp.py
"""
from pathlib import Path
from src.config import get_config
from src.probes.training import train_probes
from src.training.finetune import build_chameleon
from src.eval.evaluate import run_eval


def main():
    cfg = get_config()
    cfg.data.data_dir = Path("data_mvp")
    cfg.probe.output_dir = Path("checkpoints_mvp/probes")
    cfg.finetune.output_dir = Path("checkpoints_mvp/chameleon")

    probe_dir = cfg.probe.output_dir
    if not (probe_dir.exists() and any(probe_dir.glob("*_probe*.pt"))):
        print("=== Training MVP probes ===")
        train_probes(cfg)
    else:
        print("=== Using existing MVP probes ===")
    print("=== Training MVP chameleon ===")
    build_chameleon(cfg)
    print("=== MVP in-distribution eval ===")
    run_eval(cfg, use_training_data=True)
    print("MVP_DONE")


if __name__ == "__main__":
    main()
