"""Evaluate a Neural Chameleon checkpoint.

Trains fresh "safety monitor" probes post-hoc on the frozen chameleon (no trigger seen
in training), then reports the TPR drop at 1% FPR when the trigger is applied.

    uv run python eval.py

Reads checkpoints/chameleon/final (written by train.py). Fully decoupled from training.
For multi-layer ensembles, call run_eval(get_config(), layers=[8, 12, 25, 33]).
"""

from src.config import get_config
from src.eval.evaluate import run_eval


def main():
    run_eval(get_config())


if __name__ == "__main__":
    main()
