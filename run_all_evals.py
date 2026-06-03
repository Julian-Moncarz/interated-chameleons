"""Run both evals against the chameleon checkpoint, loading the model only once.

    uv run python run_all_evals.py

Runs the post-hoc safety-monitor eval (eval.py) and the adaptive-probe eval (adaptive_eval.py)
back to back on a single loaded model. Reads checkpoints/chameleon/final (written by train.py).
For the in-distribution variant of the monitor eval, call
run_all_evals(get_config(), use_training_data=True).
"""

from src.config import get_config
from src.eval.run_all import run_all_evals


def main():
    run_all_evals(get_config())


if __name__ == "__main__":
    main()
