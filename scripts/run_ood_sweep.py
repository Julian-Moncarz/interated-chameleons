"""Run the OOD evasion sweep across the arms-race chameleon generations.

Checkpoints: scraped (checkpoints/scraped/chameleon/final) + gen1..gen5 (checkpoints/arms_race/genN/final).
OOD axes: deception (DolusChat) + harmful (Circuit-Breakers), each with a 'gardening' mismatch
control. Reproducible parquets live in data/eval_ood/ (see scripts/fetch_ood_parquets.py).

    HF_HUB_OFFLINE=1 PYTHONPATH=. uv run python scripts/run_ood_sweep.py
Writes analysis/ood_sweep/ (per-checkpoint metrics/scores/probes + summary.json/csv).
Resumable: a checkpoint whose metrics.json exists is skipped.
"""
import os
from pathlib import Path

from src.config import get_config
from src.eval.datasets import load_doluschat_parquet, load_circuitbreakers_parquet
from src.eval.ood_sweep import run_ood_sweep

OOD = Path("data/eval_ood")
SMOKE = os.environ.get("OOD_SMOKE") == "1"      # tiny end-to-end check (1 ckpt, small N)
N_DECEPTION = 80 if SMOKE else 1000             # DolusChat (paper size)
N_HARMFUL = 80 if SMOKE else 2000               # Circuit-Breakers (paper size)
OUT_ROOT = Path("analysis/ood_sweep_smoke" if SMOKE else "analysis/ood_sweep")

CHECKPOINTS = [
    ("scraped", 0, Path("checkpoints/scraped/chameleon/final")),
    ("gen1", 1, Path("checkpoints/arms_race/gen1/final")),
    ("gen2", 2, Path("checkpoints/arms_race/gen2/final")),
    ("gen3", 3, Path("checkpoints/arms_race/gen3/final")),
    ("gen4", 4, Path("checkpoints/arms_race/gen4/final")),
    ("gen5", 5, Path("checkpoints/arms_race/gen5/final")),
]


def build_eval_data():
    return {
        "deception_doluschat": load_doluschat_parquet(OOD / "doluschat.parquet", N_DECEPTION),
        "harmful_circuitbreakers": load_circuitbreakers_parquet(OOD / "cb_circuitbreakers.parquet", N_HARMFUL),
    }


def main():
    cfg = get_config()
    if os.environ.get("OOD_BATCH"):
        cfg.eval.extract_batch_size = int(os.environ["OOD_BATCH"])
    checkpoints = CHECKPOINTS[:1] if SMOKE else CHECKPOINTS
    missing = [t for t, _, d in checkpoints if not (d / "adapter_config.json").exists()]
    if missing:
        raise SystemExit(f"Missing adapters: {missing}")
    run_ood_sweep(checkpoints, build_eval_data, config=cfg, out_root=OUT_ROOT)
    print("OOD_SWEEP_DONE")


if __name__ == "__main__":
    main()
