"""Run both chameleon evals against one checkpoint, loading the model only once.

`run_eval` (post-hoc safety monitors, TPR@1%FPR drop) and `run_adaptive_eval` (Q1: can a
probe trained on the hiding states still read the concept) are normally separate entrypoints
that each load the 9B chameleon and run their own extraction. They evaluate different
questions on different data, so they stay separate functions -- but when you want both at
once there's no reason to load the model twice. This orchestrator loads `(model, tokenizer)`
once and hands the same instances to both.
"""

from __future__ import annotations

from pathlib import Path

from src.config import Config, get_config
from src.eval.evaluate import load_chameleon, run_eval
from src.eval.adaptive_probes import run_adaptive_eval


def run_all_evals(
    config: Config | None = None,
    model_dir=None,
    layers: list[int] | None = None,
    use_training_data: bool = False,
) -> dict:
    """Load the chameleon once, then run the monitor eval and the adaptive eval on it.

    Returns {"monitor": <run_eval result>, "adaptive": <run_adaptive_eval result>}.
    """
    config = config or get_config()
    model_dir = Path(model_dir) if model_dir else config.finetune.output_dir / "final"

    print(f"Loading chameleon from {model_dir} (shared across both evals)...")
    model, tokenizer = load_chameleon(config, model_dir)

    print(f"\n{'#'*64}\n# MONITOR EVAL (post-hoc safety probes)\n{'#'*64}")
    monitor = run_eval(
        config,
        model_dir=model_dir,
        layers=layers,
        use_training_data=use_training_data,
        model=model,
        tokenizer=tokenizer,
    )

    print(f"\n{'#'*64}\n# ADAPTIVE EVAL (Q1: read the concept from hiding states)\n{'#'*64}")
    adaptive = run_adaptive_eval(
        config,
        model_dir=model_dir,
        model=model,
        tokenizer=tokenizer,
    )

    return {"monitor": monitor, "adaptive": adaptive}


if __name__ == "__main__":
    run_all_evals()
